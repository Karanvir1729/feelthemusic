"""Deterministic worker-barrier tests. Written 2026-09-19; no hardware/network."""
from dataclasses import FrozenInstanceError, replace
import threading
import time
import unittest

from lamp.dispatch import ClipIntent, Dispatcher, JOINTS, LightIntent, MoveIntent


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.now = 10_000_000_000
        self.motion_entered, self.motion_release = threading.Event(), threading.Event()
        self.light_entered = threading.Event()
        self.motion_calls, self.light_calls = [], []
        self.result = {"state": "succeeded", "result": {"completed": True, "reached": True}}

        def motion(intent):
            self.motion_calls.append(intent)
            self.motion_entered.set()
            if not self.motion_release.wait(2):
                raise RuntimeError("test barrier timed out")
            return self.result

        def light(intent):
            self.light_calls.append(intent)
            self.light_entered.set()
            return {"state": "succeeded"}

        self.dispatchers = []
        self.motion_callback, self.light_callback = motion, light
        self.d = self.make()

    def make(self, **kwargs):
        d = Dispatcher(kwargs.pop("run_motion", self.motion_callback),
                       kwargs.pop("run_light", self.light_callback), clock=lambda: self.now,
                       clip_allowlist={"verified": 3_000_000_000}, **kwargs)
        self.dispatchers.append(d)
        self.assertTrue(d.configure(1, 0, "follow", True, True, armed=True))
        return d

    def tearDown(self):
        self.motion_release.set()
        for d in self.dispatchers:
            d.close()

    def move(self, **kwargs):
        values = dict(positions=dict.fromkeys(JOINTS, 0), start_due_ns=self.now,
                      duration_ns=2_000_000_000, epoch=1, gen=0)
        values.update(kwargs)
        return MoveIntent(**values)

    def light(self, **kwargs):
        values = dict(rgb=(0, .1, .2), due_ns=self.now, epoch=1, gen=0)
        values.update(kwargs)
        return LightIntent(**values)

    def wait_for(self, predicate):
        deadline = time.monotonic() + 1
        while not predicate() and time.monotonic() < deadline:
            threading.Event().wait(.001)
        self.assertTrue(predicate())

    def start_motion(self):
        self.assertTrue(self.d.submit_move(self.move()))
        self.d.tick()
        self.assertTrue(self.motion_entered.wait(1))

    def test_light_progresses_during_motion_and_off_is_responsive(self):
        self.start_motion()
        self.now += 500_000_000
        self.assertTrue(self.d.submit_light(self.light()))
        self.d.tick()
        self.assertTrue(self.light_entered.wait(1))
        self.assertFalse(self.motion_release.is_set())
        self.assertTrue(self.d.configure(1, 1, "off", False, True))
        self.assertTrue(self.d.snapshot()["motion_inflight"])
        self.assertFalse(self.d.submit_move(self.move(gen=1)))

    def test_off_on_clears_pending_and_noop_preserves_it(self):
        self.assertTrue(self.d.submit_move(self.move()))
        self.d.configure(1, 0, "follow", True, True, armed=True)
        self.assertTrue(self.d.snapshot()["pending_motion"])
        self.d.configure(1, 1, "off", False, True)
        self.d.configure(1, 2, "follow", True, True, armed=True)
        self.d.tick()
        self.assertFalse(self.d.snapshot()["pending_motion"])
        self.assertEqual(self.motion_calls, [])
        self.assertFalse(self.d.submit_move(self.move()))
        self.assertFalse(self.d.configure(1, 1, "dance", True, True, armed=True))
        self.assertTrue(self.d.configure(2, 0, "dance", True, True, armed=True))

    def test_move_requires_explicit_completed_and_reached(self):
        self.result = {"state": "succeeded", "result": {"reached": True}}
        self.start_motion()
        self.motion_release.set()
        self.wait_for(lambda: self.d.snapshot()["latched"])
        self.assertFalse(self.d.submit_light(self.light()))
        self.assertFalse(self.d.configure(2, 0, "follow", True, True, armed=True))

    def test_timeout_retains_owner_until_real_completion(self):
        self.start_motion()
        self.now += 3_000_000_001
        self.d.tick()
        state = self.d.snapshot()
        self.assertTrue(state["latched"])
        self.assertTrue(state["motion_inflight"])
        self.assertEqual(state["motion_status"], "timed_out")
        self.assertFalse(self.d.submit_move(self.move()))
        self.motion_release.set()
        self.wait_for(lambda: not self.d.snapshot()["motion_inflight"])
        self.assertTrue(self.d.snapshot()["latched"])
        self.assertEqual(len(self.motion_calls), 1)

    def test_callback_error_latches_all_outputs(self):
        def fail(_):
            raise RuntimeError("SDK response lost")
        d = self.make(run_light=fail)
        self.assertTrue(d.submit_light(self.light()))
        d.tick()
        self.wait_for(lambda: d.snapshot()["latched"])
        self.assertTrue(d.snapshot()["light_inflight"])
        self.assertFalse(d.submit_move(self.move()))

    def test_no_early_fire_and_worker_drops_late(self):
        self.assertTrue(self.d.submit_light(self.light(due_ns=self.now + 10_000_000_000)))
        self.d.tick(self.now + 10_000_000_000)
        # Even a future explicit tick hint cannot override the worker's clock.
        self.wait_for(lambda: self.d.snapshot()["stats"].get("light_early_wait", 0) >= 1)
        self.assertEqual(self.light_calls, [])
        self.now += 10_080_000_001
        self.d.tick()
        self.wait_for(lambda: not self.d.snapshot()["pending_light"])
        self.assertEqual(self.light_calls, [])
        self.assertEqual(self.d.snapshot()["stats"]["light_late"], 1)

    def test_horizon_late_and_bool_validation(self):
        self.assertFalse(self.d.submit_light(self.light(due_ns=self.now + 30_000_000_001)))
        self.assertTrue(self.d.submit_light(self.light(due_ns=self.now + 30_000_000_000)))
        self.assertFalse(self.d.submit_light(self.light(due_ns=self.now - 80_000_001)))
        for field in ("due_ns", "epoch", "gen"):
            self.assertFalse(self.d.submit_light(self.light(**{field: True})))
        for value in (float("nan"), float("inf"), True, 10**1000):
            self.assertFalse(self.d.submit_light(self.light(rgb=(value, .1, .2))))
            self.assertFalse(self.d.submit_move(self.move(positions=dict.fromkeys(JOINTS, value))))

    def test_shared_spacing_and_rolling_budget(self):
        d = self.make(max_posts_per_minute=2, late_tolerance_ns=1_000_000_000)
        self.assertTrue(d.submit_light(self.light()))
        d.tick()
        self.wait_for(lambda: d.snapshot()["stats"].get("light_completed") == 1)
        self.assertTrue(d.submit_move(self.move()))
        d.tick()
        self.wait_for(lambda: d.snapshot()["stats"].get("rate_deferred", 0) >= 1)
        self.assertEqual(self.motion_calls, [])
        self.now += 500_000_000
        d.tick()
        self.assertTrue(self.motion_entered.wait(1))
        self.motion_release.set()
        self.wait_for(lambda: not d.snapshot()["motion_inflight"])
        self.now += 500_000_000
        self.assertTrue(d.submit_light(self.light()))
        d.tick()
        self.wait_for(lambda: d.snapshot()["stats"].get("rate_deferred", 0) >= 2)
        self.assertEqual(len(self.light_calls), 1)
        self.now += 59_000_000_000
        self.assertTrue(d.submit_light(self.light()))
        d.tick()
        self.wait_for(lambda: d.snapshot()["stats"].get("light_completed") == 2)

    def test_allowlist_mode_and_duration(self):
        good = ClipIntent("verified", self.now, 3_000_000_000, 1, 0)
        self.assertFalse(self.d.submit_clip(good))
        self.d.configure(1, 1, "dance", True, True, armed=True)
        good = replace(good, gen=1)
        self.assertFalse(self.d.submit_clip(replace(good, clip_id="unverified")))
        self.assertFalse(self.d.submit_clip(replace(good, duration_ns=3_000_000_001)))
        self.assertFalse(self.d.submit_clip(replace(good, duration_ns=True)))
        self.assertTrue(self.d.submit_clip(good))
        self.result = {"state": "succeeded"}
        self.motion_release.set()
        self.d.tick()
        self.wait_for(lambda: self.d.snapshot()["stats"].get("motion_completed") == 1)

    def test_only_latest_intents_and_immutable_copy(self):
        positions = dict.fromkeys(JOINTS, 0)
        intent = self.move(positions=positions)
        positions["base_yaw"] = 90
        with self.assertRaises(TypeError):
            intent.positions["base_yaw"] = 90
        with self.assertRaises(FrozenInstanceError):
            intent.gen = 2
        self.assertEqual(intent.positions["base_yaw"], 0)
        for i in range(100):
            self.assertTrue(self.d.submit_light(self.light(rgb=(0, .1, i / 100))))
        self.d.tick()
        self.wait_for(lambda: self.d.snapshot()["stats"].get("light_completed") == 1)
        self.assertEqual(len(self.light_calls), 1)
        self.assertEqual(self.light_calls[0].rgb, (0, .1, .3))

    def test_light_armed_sync_toggle_and_red_guard(self):
        self.d.configure(1, 1, "light", True, True)
        self.assertFalse(self.d.submit_light(self.light(gen=1)))
        self.d.configure(1, 2, "light", True, False, armed=True)
        self.assertFalse(self.d.submit_light(self.light(gen=2)))
        self.d.configure(1, 3, "light", False, True, armed=True)
        self.assertFalse(self.d.submit_light(self.light(gen=3)))
        self.d.configure(1, 4, "light", True, True, armed=True)
        self.assertFalse(self.d.submit_light(self.light(gen=4, rgb=(1, 0, 0))))
        self.assertTrue(self.d.submit_light(self.light(gen=4, rgb=(1, 1, 1))))
        self.d.tick()
        self.assertTrue(self.light_entered.wait(1))
        self.assertEqual(self.light_calls[-1].rgb, (.3, .3, .3))

    def test_close_does_not_wait_for_or_cancel_admitted_motion(self):
        self.start_motion()
        self.d.close()
        self.assertTrue(self.d.snapshot()["closed"])
        self.assertTrue(self.d.snapshot()["motion_inflight"])
        self.assertFalse(self.motion_release.is_set())
        self.assertFalse(self.d.submit_move(self.move()))

    def test_worker_rechecks_off_after_tick_before_admission(self):
        # Deliberately hold the state lock across the tick/off race. The worker
        # cannot pass its admission check until the trusted owner has set off.
        with self.d._condition:
            self.d.submit_move(self.move())
            self.d.tick()
            self.d.configure(1, 1, "off", False, True)
        self.assertFalse(self.d.snapshot()["motion_inflight"])
        self.assertFalse(self.d.snapshot()["pending_motion"])
        self.assertEqual(self.motion_calls, [])

    def test_worker_rejects_lease_expired_after_tick_before_admission(self):
        lease_end = self.now + 10_000_000
        # Hold the admission lock so the tick runs while ready, then advance
        # the worker's authoritative clock before it can admit a callback.
        with self.d._condition:
            self.assertTrue(self.d.configure(1, 0, "follow", True, True, armed=True,
                                             valid_until_ns=lease_end))
            self.assertTrue(self.d.submit_light(self.light()))
            self.d.tick()
            self.now = lease_end + 1
        self.wait_for(lambda: self.d.snapshot()["stats"].get("readiness_expired") == 1)
        self.assertEqual(self.light_calls, [])
        self.assertFalse(self.d.snapshot()["pending_light"])
        self.assertFalse(self.d.snapshot()["light_inflight"])

    def test_refreshing_only_readiness_lease_preserves_queued_work(self):
        due = self.now + 10_000_000
        self.assertTrue(self.d.configure(1, 0, "follow", True, True, armed=True,
                                         valid_until_ns=self.now + 5_000_000))
        self.assertTrue(self.d.submit_light(self.light(due_ns=due)))
        for extension in (20_000_000, 30_000_000):
            self.assertTrue(self.d.configure(1, 0, "follow", True, True, armed=True,
                                             valid_until_ns=self.now + extension))
        self.now = due
        self.d.tick()
        self.wait_for(lambda: self.d.snapshot()["stats"].get("light_completed") == 1)
        self.assertEqual(len(self.light_calls), 1)
        self.assertEqual(self.light_calls[0].due_ns, due)
        self.assertEqual(self.d.snapshot()["stats"].get("invalidated", 0), 0)

    def test_pending_motion_waits_for_terminal_completion_without_overlap(self):
        self.start_motion()
        self.now += 500_000_000
        self.assertTrue(self.d.submit_move(self.move(positions=dict.fromkeys(JOINTS, 1))))
        self.d.tick()
        self.assertEqual(len(self.motion_calls), 1)
        self.assertTrue(self.d.snapshot()["pending_motion"])
        self.motion_release.set()
        self.wait_for(lambda: self.d.snapshot()["stats"].get("motion_completed") == 2)
        self.assertEqual(len(self.motion_calls), 2)
        self.assertEqual(self.motion_calls[1].positions["base_yaw"], 1)

    def test_failure_and_uncertain_records_latch_with_no_followup(self):
        for record in ({"state": "failed"}, {"state": "running"}, None,
                       {"state": "succeeded", "result": {"completed": True, "reached": False}},
                       {"state": "succeeded", "result": {"completed": 1, "reached": True}}):
            with self.subTest(record=record):
                d = self.make(run_motion=lambda _, result=record: result)
                d.submit_move(self.move())
                d.tick()
                self.wait_for(lambda: d.snapshot()["latched"])
                self.assertFalse(d.submit_light(self.light()))
                self.assertEqual(d.snapshot()["stats"]["motion_started"], 1)
                self.assertEqual(d.snapshot()["motion_inflight"], record != {"state": "failed"})

    def test_idle_external_latch_and_policy_limits(self):
        self.d.latch("thermal_cutoff")
        self.assertTrue(self.d.snapshot()["latched"])
        self.assertFalse(self.d.submit_light(self.light()))
        for kwargs in ({"max_posts_per_minute": 121}, {"min_post_interval_ns": 499_999_999},
                       {"light_timeout_ns": True}, {"max_future_ns": 30_000_000_001}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Dispatcher(self.motion_callback, self.light_callback, **kwargs)

    def test_empty_clip_allowlist_full_move_and_duration_validation(self):
        d = Dispatcher(self.motion_callback, self.light_callback, clock=lambda: self.now)
        self.dispatchers.append(d)
        d.configure(1, 0, "dance", True, True, armed=True)
        self.assertFalse(d.submit_clip(ClipIntent("verified", self.now, 1, 1, 0)))
        self.assertFalse(self.d.submit_move(self.move(positions={"base_yaw": 0})))
        for duration in (True, 1_999_999_999, 30_000_000_001, float("nan")):
            self.assertFalse(self.d.submit_move(self.move(duration_ns=duration)))

    def test_light_watchdog_retains_worker_until_return(self):
        entered, release = threading.Event(), threading.Event()
        def blocked_light(_):
            entered.set()
            release.wait(1)
            return {"state": "succeeded"}
        d = self.make(run_light=blocked_light)
        try:
            d.submit_light(self.light())
            d.tick()
            self.assertTrue(entered.wait(1))
            self.now += 8_000_000_001
            d.tick()
            self.assertTrue(d.snapshot()["latched"])
            self.assertEqual(d.snapshot()["light_status"], "timed_out")
            self.assertTrue(d.snapshot()["light_inflight"])
            release.set()
            self.wait_for(lambda: not d.snapshot()["light_inflight"])
            self.assertTrue(d.snapshot()["latched"])
        finally:
            release.set()


if __name__ == "__main__":
    unittest.main()
