"""Deterministic scheduling regression tests, written 2026-09-19."""
import unittest

from simulation.controller import Controller, JOINTS


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.now = 10_000_000_000
        self.sent = []
        self.c = Controller("test-session", self.sent.append, clock=lambda: self.now)
        self.c.set_synced(True)
        self.c.set_mode("follow", armed=True)
        self.c.lock_target("listener")

    def command(self, seq=1, **changes):
        result = dict(session_id="test-session", generation=self.c.generation,
                      seq=seq, due_ns=self.now, kind=changes.get("kind", "head_target"))
        if result["kind"] == "head_target":
            result.update(target_id="listener", frame="lamp_base", position_m=[.1, .8, .4],
                          capture_ns=self.now, valid_until_ns=self.now + self.c.max_target_age_ns)
        result.update(changes)
        return result

    def test_starts_disarmed_and_requires_sync(self):
        c = Controller("s", self.sent.append, clock=lambda: self.now)
        self.assertEqual((c.mode, c.armed, c.synced), ("hold", False, False))
        with self.assertRaises(ValueError):
            c.set_mode("follow", armed=True)
        self.c.set_synced(False)
        self.assertFalse(self.c.submit(self.command()))

    def test_future_waits_without_adding_latency(self):
        self.assertTrue(self.c.submit(self.command(kind="light", rgb=[0, .1, .1],
                                                   due_ns=self.now + 10_000_000_000)))
        self.assertEqual(self.c.tick(), 0)
        self.now += 9_999_999_999
        self.assertEqual(self.c.tick(), 0)
        self.now += 1
        self.assertEqual(self.c.tick(), 1)
        self.assertEqual(self.sent[0]["due_ns"], self.now)

    def test_missing_boolean_nonfinite_and_far_future_rejected(self):
        for field in ("due_ns", "seq", "generation"):
            for bad in (None, True, 1.5, float("nan"), float("inf"), -1):
                self.assertFalse(self.c.submit(self.command(**{field: bad})))
        for bad in (True, float("nan"), float("inf"), "0.5", 10**1000):
            self.assertFalse(self.c.submit(self.command(position_m=[bad, 1, 1])))
        c = self.command()
        del c["due_ns"]
        self.assertFalse(self.c.submit(c))
        self.assertFalse(self.c.submit(self.command(due_ns=self.now + 31_000_000_000)))

    def test_late_admission_and_dispatch_dropped(self):
        self.c.late_tolerance_ns = 0
        self.assertFalse(self.c.submit(self.command(due_ns=self.now - 1)))
        self.assertTrue(self.c.submit(self.command(due_ns=self.now + 10)))
        self.now += 11
        self.assertEqual(self.c.tick(), 0)
        self.assertEqual(self.c.stats["late"], 2)

    def test_explicit_lateness_tolerance(self):
        self.c.late_tolerance_ns = 5
        self.assertTrue(self.c.submit(self.command(due_ns=self.now - 5, capture_ns=self.now - 10,
                                                  valid_until_ns=self.now)))
        self.assertEqual(self.c.tick(), 1)

    def test_sequence_session_and_capacity(self):
        self.c.capacity = 1
        self.assertTrue(self.c.submit(self.command(seq=2, kind="light", rgb=[0, .1, .1])))
        for seq in (1, 2):
            self.assertFalse(self.c.submit(self.command(seq=seq)))
        self.assertFalse(self.c.submit(self.command(seq=3, kind="light", rgb=[0, .1, .1])))
        self.assertEqual(self.c.stats["capacity"], 1)
        self.assertFalse(self.c.submit(self.command(seq=3, session_id="other")))

    def test_head_lock_frame_and_loss(self):
        self.assertFalse(self.c.lock_target("other"))
        self.assertFalse(self.c.submit(self.command(target_id="other")))
        self.assertFalse(self.c.submit(self.command(frame="mac_camera")))
        stale = self.command()
        self.assertTrue(self.c.submit(stale))
        self.c.release_target()
        self.assertEqual(self.c.tick(), 0)
        self.assertFalse(self.c.submit(stale))
        self.assertTrue(self.c.lock_target("other"))

    def test_mode_change_invalidates_queue_but_retains_action_ownership(self):
        self.c.submit(self.command())
        self.c.tick()
        first = self.c.inflight
        self.c.submit(self.command(seq=2, due_ns=self.now + 100))
        old = self.c.generation
        self.c.set_mode("dance", armed=True)
        self.assertEqual(self.c.generation, old + 1)
        self.assertEqual(self.c.queued, 0)
        self.assertEqual(self.c.inflight, first)
        self.assertTrue(self.c.complete(first, "succeeded"))

    def test_light_runs_while_motion_busy_and_motion_cannot_overlap(self):
        self.c.submit(self.command())
        self.c.tick()
        self.c.submit(self.command(seq=2))
        self.c.submit(self.command(seq=3, kind="light", rgb=[0, .4, .7]))
        self.assertEqual(self.c.tick(), 1)
        self.assertEqual([c["kind"] for c in self.sent], ["head_target", "light"])
        self.assertGreater(self.c.stats["motion_waits"], 0)
        self.assertEqual(self.c.queued, 1)

    def test_unknown_or_failed_action_latches(self):
        for outcome in ("unknown", "failed"):
            with self.subTest(outcome=outcome):
                self.setUp()
                self.c.submit(self.command())
                self.c.tick()
                action = self.c.inflight
                self.assertFalse(self.c.complete(("bad", 1, 1), outcome))
                self.assertTrue(self.c.complete(action, outcome))
                self.assertFalse(self.c.submit(self.command(seq=2)))
                with self.assertRaises(ValueError):
                    self.c.set_mode("dance", armed=True)
                self.assertFalse(self.c.submit(self.command(seq=3, kind="light", rgb=[0, .4, .7])))

    def test_callback_exception_latches_motion(self):
        def fail(_):
            raise RuntimeError("simulator lost action")
        self.c.output = fail
        self.c.submit(self.command())
        self.assertEqual(self.c.tick(), 0)
        self.assertTrue(self.c.motion_latched)

    def test_completion_allows_next_motion_without_releasing_target(self):
        self.c.submit(self.command())
        self.c.tick()
        self.assertTrue(self.c.complete(self.c.inflight, "succeeded"))
        self.assertTrue(self.c.submit(self.command(seq=2)))
        self.assertEqual(self.c.tick(), 1)
        self.assertEqual(self.c.target_id, "listener")

    def test_deadline_order_is_independent_of_submission_order(self):
        self.c.submit(self.command(seq=1, kind="light", rgb=[0, .1, .1], due_ns=self.now + 20))
        self.c.submit(self.command(seq=2, kind="light", rgb=[0, .2, .2], due_ns=self.now + 10))
        self.now += 10
        self.assertEqual(self.c.tick(), 1)
        self.assertEqual(self.sent[0]["seq"], 2)
        self.now += 10
        self.assertEqual(self.c.tick(), 1)
        self.assertEqual(self.sent[1]["seq"], 1)

    def test_sync_loss_invalidates_and_disarms(self):
        old = self.command()
        self.c.submit(old)
        self.c.set_synced(False)
        self.c.set_synced(True)
        self.assertEqual(self.c.tick(), 0)
        self.assertFalse(self.c.armed)
        self.assertFalse(self.c.submit(old))

    def test_dance_requires_complete_bounded_pose_and_duration(self):
        self.c.set_mode("dance", armed=True)
        pose = dict.fromkeys(JOINTS, 0.)
        cmd = self.command(kind="dance", positions=pose, duration_ns=2_000_000_000)
        self.assertFalse(self.c.submit(dict(cmd, positions={"base_yaw": 1})))
        self.assertFalse(self.c.submit(dict(cmd, duration_ns=1)))
        self.assertFalse(self.c.submit(dict(cmd, positions=dict(pose, elbow_pitch=101))))
        self.assertTrue(self.c.submit(cmd))
        pose["base_yaw"] = 99
        self.c.tick()
        self.assertEqual(self.sent[0]["positions"]["base_yaw"], 0)

    def test_light_validates_values_and_renderer_owns_flash_safety(self):
        for rgb in ([True, 0, 0], [1.1, 0, 0], [float("nan"), 0, 0], [0, 1], [10**1000, 0, 0]):
            self.assertFalse(self.c.submit(self.command(kind="light", rgb=rgb)))
        self.assertTrue(self.c.submit(self.command(kind="light", rgb=[0, .2, .2])))
        self.assertEqual(self.c.tick(), 1)

    def test_old_capture_cannot_be_refreshed_by_a_fresh_deadline(self):
        self.assertFalse(self.c.submit(self.command(
            capture_ns=self.now - self.c.max_target_age_ns - 1)))
        self.assertEqual(self.c.stats["stale_target"], 1)
        self.assertEqual(self.c.queued, 0)

    def test_target_expiry_overrides_general_lateness_tolerance(self):
        self.c.late_tolerance_ns = 1_000_000_000
        self.assertTrue(self.c.submit(self.command(due_ns=self.now + 10, valid_until_ns=self.now + 10)))
        self.now += 11
        self.assertEqual(self.c.tick(), 0)
        self.assertEqual(self.c.stats["stale_target"], 1)
        self.assertIsNone(self.c.inflight)

    def test_target_time_fields_are_required_bounded_integers(self):
        for field in ("capture_ns", "valid_until_ns"):
            cmd = self.command()
            del cmd[field]
            self.assertFalse(self.c.submit(cmd))
            for bad in (None, True, 1.5, float("nan"), float("inf"), -1):
                self.assertFalse(self.c.submit(self.command(**{field: bad})))
        for changes in (
            {"capture_ns": self.now + 1},
            {"due_ns": self.now + 11, "valid_until_ns": self.now + 10},
            {"valid_until_ns": self.now + self.c.max_target_age_ns + 1},
        ):
            self.assertFalse(self.c.submit(self.command(**changes)))
        self.c.late_tolerance_ns = 10
        self.assertFalse(self.c.submit(self.command(due_ns=self.now - 1)))

    def test_loss_and_expiry_do_not_clear_running_action(self):
        self.c.submit(self.command())
        self.c.tick()
        owned = self.c.inflight
        self.c.late_tolerance_ns = 1_000_000_000
        self.c.submit(self.command(seq=2, due_ns=self.now + 10, valid_until_ns=self.now + 10))
        self.now += 11
        self.c.tick()
        self.assertEqual(self.c.inflight, owned)
        self.c.release_target()
        self.assertEqual(self.c.inflight, owned)

    def test_unknown_fields_rejected_before_copying_nested_input(self):
        nested = []
        for _ in range(2000):
            nested = [nested]
        for fields in ({}, {"kind": "light", "rgb": [0, .1, .1]},
                       {"kind": "dance", "positions": dict.fromkeys(JOINTS, 0),
                        "duration_ns": 2_000_000_000}):
            self.assertFalse(self.c.submit(self.command(extra=nested, **fields)))
        self.assertFalse(self.c.submit(self.command(kind=["head_target"])))
        self.assertEqual(self.c.queued, 0)

    def test_configured_target_age_boundary(self):
        self.c.max_target_age_ns = 100
        self.assertTrue(self.c.submit(self.command(capture_ns=self.now - 100, valid_until_ns=self.now)))
        self.assertEqual(self.c.tick(), 1)

    def test_default_80ms_tick_tolerance_and_exact_limit(self):
        self.assertEqual(self.c.late_tolerance_ns, 80_000_000)
        self.c.submit(self.command())
        self.now += 80_000_000
        self.assertEqual(self.c.tick(), 1)
        self.c.complete(self.c.inflight, "succeeded")
        self.c.submit(self.command(seq=2))
        self.now += 80_000_001
        self.assertEqual(self.c.tick(), 0)
        self.assertEqual(self.c.stats["late"], 1)

    def test_30second_horizon_is_inclusive_and_never_fires_early(self):
        cmd = self.command(kind="light", rgb=[0, .1, .1], due_ns=self.now + 30_000_000_000)
        self.assertFalse(self.c.submit(dict(cmd, due_ns=cmd["due_ns"] + 1)))
        self.assertTrue(self.c.submit(cmd))
        self.now += 29_999_999_999
        self.assertEqual(self.c.tick(), 0)
        self.now += 1
        self.assertEqual(self.c.tick(), 1)

    def test_external_idle_latch_blocks_all_output_and_cannot_rearm(self):
        for reason in ("thermal", "torque_off", "409", "operator"):
            with self.subTest(reason=reason):
                self.setUp()
                self.c.submit(self.command(kind="light", rgb=[0, .1, .1]))
                self.c.safety_latch(reason)
                self.assertEqual(self.c.queued, 0)
                self.assertEqual(self.c.tick(), 0)
                self.assertTrue(self.c.safety_latched)
                self.assertEqual(self.c.latch_reason, reason)
                self.assertFalse(self.c.submit(self.command(seq=2)))
                self.assertFalse(self.c.submit(self.command(seq=3, kind="light", rgb=[0, .1, .1])))
                with self.assertRaises(ValueError):
                    self.c.set_mode("follow", armed=True)
                self.c.set_synced(False)
                self.c.set_synced(True)
                self.assertTrue(self.c.safety_latched)

    def test_light_output_failure_latches_and_preserves_active_motion(self):
        self.c.submit(self.command())
        self.c.tick()
        action = self.c.inflight
        def fail(_):
            raise RuntimeError("renderer unavailable")
        self.c.output = fail
        self.c.submit(self.command(seq=2, kind="light", rgb=[0, .1, .1]))
        self.c.submit(self.command(seq=3))
        self.assertEqual(self.c.tick(), 0)
        self.assertTrue(self.c.safety_latched)
        self.assertEqual(self.c.inflight, action)
        self.assertEqual(self.c.queued, 0)
        self.assertTrue(self.c.complete(action, "succeeded"))
        self.assertTrue(self.c.safety_latched)

    def test_process_interrupt_latches_then_propagates(self):
        for error in (KeyboardInterrupt, SystemExit):
            with self.subTest(error=error):
                self.setUp()
                def fail(_):
                    raise error()
                self.c.output = fail
                self.c.submit(self.command())
                with self.assertRaises(error):
                    self.c.tick()
                self.assertTrue(self.c.safety_latched)
                self.assertIsNotNone(self.c.inflight)

    def test_watchdog_survives_sync_and_mode_changes_without_clearing_ownership(self):
        self.c.submit(self.command())
        self.c.tick()
        action = self.c.inflight
        self.c.set_mode("hold")
        self.c.set_synced(False)
        self.now += 3_000_000_000
        self.c.tick()
        self.assertFalse(self.c.safety_latched)
        self.now += 1
        self.c.tick()
        self.assertTrue(self.c.safety_latched)
        self.assertEqual(self.c.inflight, action)
        self.assertEqual(self.c.stats["motion_timeout"], 1)
        self.c.tick()
        self.assertEqual(self.c.stats["motion_timeout"], 1)
        self.assertTrue(self.c.complete(action, "succeeded"))
        self.assertTrue(self.c.safety_latched)

    def test_late_completion_itself_checks_watchdog(self):
        self.c.submit(self.command())
        self.c.tick()
        action = self.c.inflight
        self.now += 3_000_000_001
        self.assertTrue(self.c.complete(action, "succeeded"))
        self.assertTrue(self.c.safety_latched)

    def test_dance_watchdog_uses_command_duration_plus_margin(self):
        self.c.set_mode("dance", armed=True)
        self.c.submit(self.command(kind="dance", positions=dict.fromkeys(JOINTS, 0), duration_ns=5_000_000_000))
        self.c.tick()
        self.now += 6_000_000_000
        self.c.tick()
        self.assertFalse(self.c.safety_latched)
        self.now += 1
        self.c.tick()
        self.assertTrue(self.c.safety_latched)

    def test_mode_switch_waits_and_dispatches_only_if_original_deadline_is_fresh(self):
        self.c.submit(self.command())
        self.c.tick()
        first = self.c.inflight
        self.c.set_mode("dance", armed=True)
        self.c.submit(self.command(seq=2, kind="dance", positions=dict.fromkeys(JOINTS, 0), duration_ns=2_000_000_000))
        self.assertEqual(self.c.tick(), 0)
        self.assertEqual(self.c.queued, 1)
        self.now += 20_000_000
        self.c.complete(first, "succeeded")
        self.assertEqual(self.c.tick(), 1)
        self.assertEqual(self.sent[-1]["kind"], "dance")

    def test_waiting_motion_drops_instead_of_retiming_after_completion(self):
        self.c.submit(self.command())
        self.c.tick()
        first = self.c.inflight
        self.c.submit(self.command(seq=2))
        self.c.tick()
        self.now += 80_000_001
        self.c.complete(first, "succeeded")
        self.assertEqual(self.c.tick(), 0)
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(self.c.stats["late"], 1)

    def test_follow_coalesces_latest_capture_and_rejects_capture_regression(self):
        self.c.submit(self.command())
        self.now += 10
        self.c.submit(self.command(seq=2, position_m=[.2, .8, .4]))
        self.assertEqual(self.c.queued, 1)
        self.assertFalse(self.c.submit(self.command(seq=3, capture_ns=self.now - 1,
                                                  valid_until_ns=self.now + 100)))
        self.assertEqual(self.c.stats["capture_reordered"], 1)
        self.assertEqual(self.c.tick(), 1)
        self.assertEqual(self.sent[0]["seq"], 2)
        self.assertEqual(self.sent[0]["position_m"], [.2, .8, .4])

    def test_current_target_evicts_farthest_future_light_at_capacity(self):
        self.c.capacity = 2
        self.c.submit(self.command(seq=1, kind="light", rgb=[0, .1, .1], due_ns=self.now + 1_000_000_000))
        self.c.submit(self.command(seq=2, kind="light", rgb=[0, .2, .2], due_ns=self.now + 30_000_000_000))
        self.assertTrue(self.c.submit(self.command(seq=3)))
        self.assertEqual(self.c.queued, 2)
        self.assertEqual(self.c.stats["future_light_evicted"], 1)
        self.assertEqual(self.c.tick(), 1)
        self.c.complete(self.c.inflight, "succeeded")
        self.now += 1_000_000_000
        self.assertEqual(self.c.tick(), 1)
        self.assertEqual(self.sent[-1]["seq"], 1)

    def test_future_capture_rejected_even_with_valid_future_deadline(self):
        self.assertFalse(self.c.submit(self.command(capture_ns=self.now + 1, due_ns=self.now + 2,
                                                   valid_until_ns=self.now + 100)))
        self.assertEqual(self.c.queued, 0)

    def test_hold_rejects_lights_and_invalidates_previously_queued_light(self):
        self.c.submit(self.command(kind="light", rgb=[0, .1, .1]))
        self.c.set_mode("hold")
        self.assertEqual(self.c.tick(), 0)
        self.assertFalse(self.c.submit(self.command(seq=2, kind="light", rgb=[0, .1, .1])))

    def test_bounded_configuration_and_sequence_jump(self):
        for values in ({"capacity": 4097}, {"capacity": True}, {"max_future_ns": 30_000_000_001},
                       {"late_tolerance_ns": 1_000_000_001}, {"max_target_age_ns": 5_000_000_001},
                       {"watchdog_margin_ns": 5_000_000_001}, {"head_motion_duration_ns": 1},
                       {"max_motion_duration_ns": 30_000_000_001}, {"max_sequence_jump": 65537}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Controller("s", self.sent.append, **values)
        self.assertFalse(self.c.submit(self.command(seq=2**63 - 1)))
        self.assertEqual(self.c.last_seq, -1)
        self.assertTrue(self.c.submit(self.command(seq=1)))


if __name__ == "__main__":
    unittest.main()
