"""Synthetic PR #15/#16 contract checks, written 2026-09-19.

The light adapter below is TEST-ONLY. It is not an application event mapping,
an SDK renderer, a network client, or evidence of physical light/motion safety.
Native packet layout and timestamp semantics are source-reported, not captured.
"""

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lamp"))
try:
    from ftm_client import BassEnvelope, EventPacket, decode
    from ftm_clock import ClockEstimator
    from ftm_events import Normalizer
    from simulation.controller import Controller
except ModuleNotFoundError as exc:
    if exc.name in {"ftm_client", "ftm_clock", "ftm_events", "simulation", "simulation.controller"}:
        raise ImportError(
            "Native scheduler contract suite requires PR #15 simulation/controller.py "
            "and PR #16 lamp/ftm_client.py, ftm_clock.py, ftm_events.py. "
            "Assemble those dependencies before running; see tests/integration/README.md."
        ) from exc
    raise


MS = 1_000_000
SECOND = 1_000_000_000


def native_event(master_ns, *, seq=1, kind=1, flags=0, intensity=128):
    """Hand-assemble reported type-3 bytes without using production encode()."""
    return (b"\x03" + seq.to_bytes(4, "little") + bytes((kind, flags, intensity, 80))
            + (30).to_bytes(2, "little") + (60).to_bytes(2, "little") + b"\xff"
            + master_ns.to_bytes(8, "little") + (777).to_bytes(4, "little"))


def native_bass(start_ns, *, seq=1, step_ms=20, samples=(0, 128, 255)):
    """Hand-assemble reported type-12 layout; timestamp policy is a separate opt-in."""
    return (b"\x0c" + seq.to_bytes(4, "little") + start_ns.to_bytes(8, "little")
            + bytes((step_ms, len(samples))) + bytes(samples))


def add_symmetric_probe(estimator, *, received_ns, offset_ns, one_way_ns):
    """Synthetic physical stamps; offset = conductor clock minus local clock."""
    sent = received_ns - 2 * one_way_ns
    conductor = sent + one_way_ns + offset_ns
    estimator.add_sample(sent, conductor, conductor, received_ns)


class FakeClock:
    def __init__(self, now=10 * SECOND):
        self.now = now

    def __call__(self):
        return self.now


class SyntheticLightAdapter:
    """Exercise component boundaries, not the unfinished production integration.

    TEST POLICY: normalized kick and explicitly opted-in bass samples map to an
    inert light record. Green is 0.05 * intensity, solely to observe scalars.
    Flags, targets and lead_us do not drive faults, motion or routing here.
    The actual application must define those policies and apply a flash limiter.
    Clock readiness is refreshed before receipt and every tick; no event can
    preserve stale synchronization merely because it was queued while synced.
    """

    def __init__(self, *, offset_ns=0, trim_ns=0, sync=True, max_age_ns=SECOND,
                 bass_time_policy=None):
        self.clock = FakeClock()
        self.estimator = ClockEstimator(keep=3, min_samples=3, drift_ppm=0,
                                        max_age_ns=max_age_ns)
        if sync:
            for age in (300 * MS, 200 * MS, 100 * MS):
                add_symmetric_probe(self.estimator, received_ns=self.clock.now - age,
                                    offset_ns=offset_ns, one_way_ns=MS)
        self.normalizer = Normalizer(self.estimator, trim_ns=trim_ns,
                                     bass_time_policy=bass_time_policy)
        self._epoch = self.normalizer.epoch
        self._next_seq = 0
        self.output = []
        self.controller = Controller("synthetic-contract-session", self._record,
                                     clock=self.clock, late_tolerance_ns=0)
        self.refresh_sync()
        self.controller.set_mode("dance")

    def _record(self, command):
        self.output.append((self.clock.now, command))

    def refresh_sync(self):
        if self._epoch != self.normalizer.epoch:
            # Keep the same Controller: invalidate queued work without losing
            # ownership of an active motion. Local sequencing does not reset.
            self.controller.set_synced(False)
            self._epoch = self.normalizer.epoch
        synced = self.estimator.estimate(now_ns=self.clock.now) is not None
        if self.controller.synced != synced:
            self.controller.set_synced(synced)

    def receive(self, datagram):
        self.refresh_sync()
        packet = decode(datagram)
        if isinstance(packet, BassEnvelope):
            samples = self.normalizer.normalize_bass(packet, now_ns=self.clock.now)
            admitted = [self.submit_light(s.due_ns, s.level, s.epoch) for s in samples]
            return bool(admitted) and all(admitted)
        if not isinstance(packet, EventPacket):
            return False
        event = self.normalizer.normalize_event(packet, now_ns=self.clock.now)
        return self.submit_event(event)

    def submit_event(self, event):
        if event is None or event.kind != "kick":
            return False
        return self.submit_light(event.due_ns, event.intensity, event.epoch)

    def submit_light(self, due_ns, intensity, epoch):
        self.refresh_sync()
        if epoch != self.normalizer.epoch:
            return False
        seq = self._next_seq
        self._next_seq += 1
        return self.controller.submit({
            "session_id": self.controller.session_id,
            "generation": self.controller.generation,
            "seq": seq, "due_ns": due_ns, "kind": "light",
            "rgb": [0.0, 0.05 * intensity, 0.0],
        })

    def tick(self):
        self.refresh_sync()
        return self.controller.tick()


class NativeSchedulerContractTests(unittest.TestCase):
    def test_offsets_and_trim_convert_once_without_another_room_budget(self):
        for offset in (2 * SECOND, -2 * SECOND):
            for trim in (0, 7 * MS):
                with self.subTest(offset_ns=offset, trim_ns=trim):
                    adapter = SyntheticLightAdapter(offset_ns=offset, trim_ns=trim)
                    local_due = adapter.clock.now + 100 * MS
                    # masterTs is already the presentation deadline; no L added here.
                    raw = native_event(local_due + offset + trim, intensity=128)
                    self.assertTrue(adapter.receive(raw))
                    self.assertEqual(adapter.tick(), 0)
                    adapter.clock.now = local_due - 1
                    self.assertEqual(adapter.tick(), 0)
                    self.assertEqual(adapter.output, [])
                    adapter.clock.now = local_due
                    self.assertEqual(adapter.tick(), 1)
                    fired_at, command = adapter.output[0]
                    self.assertEqual(fired_at, local_due)
                    self.assertEqual(command["due_ns"], local_due)
                    self.assertAlmostEqual(command["rgb"][1], 0.05 * 128 / 255)
                    adapter.clock.now += 300 * MS
                    self.assertEqual(adapter.tick(), 0)
                    self.assertEqual(len(adapter.output), 1)

    def test_late_arrival_drops_instead_of_firing_on_receipt(self):
        adapter = SyntheticLightAdapter(offset_ns=-SECOND, trim_ns=5 * MS)
        due = adapter.clock.now - 1
        self.assertFalse(adapter.receive(native_event(due - SECOND + 5 * MS)))
        self.assertEqual(adapter.controller.stats["late"], 1)
        self.assertEqual(adapter.tick(), 0)
        self.assertEqual(adapter.output, [])

    def test_queued_event_missing_its_deadline_is_dropped(self):
        adapter = SyntheticLightAdapter()
        due = adapter.clock.now + 100 * MS
        self.assertTrue(adapter.receive(native_event(due)))
        adapter.clock.now = due + 1
        self.assertEqual(adapter.tick(), 0)
        self.assertEqual(adapter.controller.queued, 0)
        self.assertEqual(adapter.controller.stats["late"], 1)
        self.assertEqual(adapter.output, [])

    def test_unsynced_clock_never_supplies_output(self):
        adapter = SyntheticLightAdapter(sync=False)
        self.assertFalse(adapter.receive(native_event(adapter.clock.now)))
        self.assertEqual(adapter.normalizer.not_synced, 1)
        self.assertEqual(adapter.tick(), 0)
        self.assertEqual(adapter.controller.queued, 0)
        self.assertEqual(adapter.output, [])

    def test_expired_clock_invalidates_queued_and_new_output(self):
        adapter = SyntheticLightAdapter()
        due = adapter.clock.now + 2 * SECOND
        self.assertTrue(adapter.receive(native_event(due)))
        adapter.clock.now = due
        self.assertEqual(adapter.tick(), 0)
        self.assertFalse(adapter.controller.synced)
        self.assertEqual(adapter.controller.queued, 0)
        self.assertFalse(adapter.receive(native_event(due, seq=2)))
        self.assertEqual(adapter.normalizer.not_synced, 1)
        self.assertEqual(adapter.output, [])

    def test_unknown_wire_kind_is_never_reinterpreted_as_kick(self):
        adapter = SyntheticLightAdapter()
        for seq, kind in enumerate((6, 127, 255), start=1):
            self.assertFalse(adapter.receive(native_event(adapter.clock.now, seq=seq, kind=kind)))
        self.assertEqual(adapter.normalizer.unknown_kind, 3)
        self.assertEqual(adapter.normalizer.normalized, 0)
        self.assertEqual(adapter.tick(), 0)
        self.assertEqual(adapter.output, [])
        self.assertTrue(adapter.receive(native_event(adapter.clock.now, seq=4, kind=1)))
        self.assertEqual(adapter.tick(), 1)

    def test_mode_transition_invalidates_already_normalized_work(self):
        adapter = SyntheticLightAdapter()
        due = adapter.clock.now + 100 * MS
        self.assertTrue(adapter.receive(native_event(due)))
        old_generation = adapter.controller.generation
        adapter.controller.set_mode("follow")
        self.assertGreater(adapter.controller.generation, old_generation)
        adapter.clock.now = due
        self.assertEqual(adapter.tick(), 0)
        self.assertEqual(adapter.output, [])
        self.assertTrue(adapter.receive(native_event(due, seq=2)))
        self.assertEqual(adapter.tick(), 1)
        self.assertEqual(adapter.output[0][1]["generation"], adapter.controller.generation)

    def test_faults_are_external_state_not_invented_wire_flag_semantics(self):
        adapter = SyntheticLightAdapter()
        # Unknown bits still parse and do not constitute an invented fault signal.
        self.assertTrue(adapter.receive(native_event(adapter.clock.now, flags=0x80)))
        self.assertEqual(adapter.tick(), 1)
        self.assertFalse(adapter.controller.safety_latched)
        # Control text is not interpreted by this explicitly limited test adapter.
        self.assertFalse(adapter.receive(b'\x0d{"thermal_fault":true}'))
        self.assertFalse(adapter.controller.safety_latched)
        due = adapter.clock.now + 100 * MS
        self.assertTrue(adapter.receive(native_event(due, seq=2)))
        adapter.controller.safety_latch("synthetic_external_sensor_fault")
        adapter.clock.now = due
        self.assertEqual(adapter.tick(), 0)
        self.assertFalse(adapter.receive(native_event(due, seq=3)))
        self.assertEqual(len(adapter.output), 1)
        self.assertEqual(adapter.controller.queued, 0)
        self.assertTrue(adapter.controller.safety_latched)

    def test_bass_has_no_output_without_explicit_timestamp_policy(self):
        adapter = SyntheticLightAdapter()
        self.assertFalse(adapter.receive(native_bass(adapter.clock.now)))
        self.assertEqual(adapter.normalizer.bass_policy_unset, 1)
        self.assertEqual(adapter.normalizer.bass_normalized, 0)
        self.assertEqual(adapter.controller.queued, 0)
        self.assertEqual(adapter.tick(), 0)
        self.assertEqual(adapter.output, [])

    def test_opted_in_bass_converts_offset_and_trim_once_for_each_sample(self):
        for offset in (SECOND, -SECOND):
            with self.subTest(offset_ns=offset):
                trim = 9 * MS
                adapter = SyntheticLightAdapter(offset_ns=offset, trim_ns=trim,
                                                bass_time_policy="presentation")
                first_due = adapter.clock.now + 100 * MS
                self.assertTrue(adapter.receive(native_bass(first_due + offset + trim)))
                self.assertEqual(adapter.normalizer.bass_normalized, 1)
                for index, level in enumerate((0, 128, 255)):
                    due = first_due + index * 20 * MS
                    adapter.clock.now = due - 1
                    self.assertEqual(adapter.tick(), 0)
                    adapter.clock.now = due
                    self.assertEqual(adapter.tick(), 1)
                    fired_at, command = adapter.output[-1]
                    self.assertEqual((fired_at, command["due_ns"]), (due, due))
                    self.assertAlmostEqual(command["rgb"][1], 0.05 * level / 255)
                self.assertEqual(len(adapter.output), 3)

    def test_native_u32_wrap_does_not_reset_local_admission_order(self):
        adapter = SyntheticLightAdapter()
        for native_seq in (2**32 - 1, 0):
            self.assertTrue(adapter.receive(native_event(adapter.clock.now, seq=native_seq)))
            self.assertEqual(adapter.tick(), 1)
        self.assertEqual([command["seq"] for _, command in adapter.output], [0, 1])

    def test_new_epoch_invalidates_old_work_and_requires_clock_rewarming(self):
        adapter = SyntheticLightAdapter()
        due = adapter.clock.now + 100 * MS
        old_event = adapter.normalizer.normalize_event(decode(native_event(due)),
                                                       now_ns=adapter.clock.now)
        self.assertTrue(adapter.submit_event(old_event))
        controller = adapter.controller
        new_epoch = adapter.normalizer.new_epoch()
        self.assertGreater(new_epoch, old_event.epoch)
        self.assertEqual(adapter.tick(), 0)
        self.assertIs(adapter.controller, controller)
        self.assertFalse(controller.synced)
        self.assertEqual(controller.queued, 0)
        self.assertFalse(adapter.receive(native_event(due)))
        offset = 200 * MS
        for index, age in enumerate((90 * MS, 60 * MS, 30 * MS)):
            add_symmetric_probe(adapter.estimator, received_ns=adapter.clock.now - age,
                                offset_ns=offset, one_way_ns=MS)
            adapter.refresh_sync()
            self.assertEqual(controller.synced, index == 2)
        self.assertFalse(adapter.submit_event(old_event))
        self.assertTrue(adapter.receive(native_event(due + offset, seq=0)))
        adapter.clock.now = due
        self.assertEqual(adapter.tick(), 1)
        self.assertEqual(len(adapter.output), 1)
        self.assertEqual(adapter.output[0][1]["seq"], 1)

    def test_epoch_change_preserves_inflight_motion_ownership(self):
        # This synthetic motion is a scheduler fixture, NOT a mapping from any
        # native kind into robot joints. It exercises the consumer's epoch reset.
        adapter = SyntheticLightAdapter()
        controller = adapter.controller
        controller.set_mode("dance", armed=True)
        motion = {
            "session_id": controller.session_id, "generation": controller.generation,
            "seq": 0, "due_ns": adapter.clock.now, "kind": "dance",
            "positions": dict.fromkeys(("base_yaw", "base_pitch", "elbow_pitch",
                                         "wrist_roll", "wrist_pitch"), 0),
            "duration_ns": 2 * SECOND,
        }
        self.assertTrue(controller.submit(motion))
        adapter._next_seq = 1
        self.assertEqual(adapter.tick(), 1)
        owned = controller.inflight
        self.assertIsNotNone(owned)
        self.assertTrue(adapter.receive(native_event(adapter.clock.now + 100 * MS)))
        adapter.normalizer.new_epoch()
        adapter.tick()
        self.assertIs(adapter.controller, controller)
        self.assertEqual(controller.queued, 0)
        self.assertEqual(controller.inflight, owned)
        self.assertFalse(controller.armed)
        self.assertTrue(controller.complete(owned, "succeeded"))
        self.assertIsNone(controller.inflight)

    def test_fresh_higher_rtt_samples_restore_output_after_old_minima_expire(self):
        # Regression: historical lowest-delay samples must not permanently crowd
        # fresh, higher-delay samples out of the estimator's bounded sample pool.
        adapter = SyntheticLightAdapter(sync=False, max_age_ns=MS)
        offset = 5 * MS
        for age in (30_000, 20_000, 10_000):
            add_symmetric_probe(adapter.estimator, received_ns=adapter.clock.now - age,
                                offset_ns=offset, one_way_ns=1_000)
        adapter.refresh_sync()
        self.assertTrue(adapter.controller.synced)
        adapter.clock.now += 2 * MS
        adapter.tick()
        self.assertFalse(adapter.controller.synced)
        for age in (90_000, 60_000, 30_000):
            add_symmetric_probe(adapter.estimator, received_ns=adapter.clock.now - age,
                                offset_ns=offset, one_way_ns=10_000)
        due = adapter.clock.now + 100_000
        self.assertTrue(adapter.receive(native_event(due + offset)),
                        "fresh higher-RTT probes must replace expired historical minima")
        adapter.clock.now = due
        self.assertEqual(adapter.tick(), 1)
        self.assertEqual(adapter.output[0][1]["due_ns"], due)


if __name__ == "__main__":
    unittest.main()
