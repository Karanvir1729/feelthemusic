"""Lamp application contracts, written 2026-09-19; injected I/O only."""

from dataclasses import dataclass
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import LampApp, OutputTypes, SDKActions
from dispatch import ClipIntent, LightIntent, MoveIntent

NS = 1_000_000_000
MS = 1_000_000
JOINTS = ("base_yaw", "base_pitch", "elbow_pitch", "wrist_roll", "wrist_pitch")


@dataclass
class EventOut:
    event: object


@dataclass
class BassOut:
    samples: list


@dataclass
class ModeOut:
    gen: int


@dataclass
class SessionOut:
    epoch: int


TYPES = OutputTypes(EventOut, BassOut, ModeOut, SessionOut)


def event(due, intensity=1.0, epoch=0):
    return EventOut(SimpleNamespace(due_ns=due, intensity=intensity, epoch=epoch))


def bass(due, level=1.0, epoch=0):
    return SimpleNamespace(due_ns=due, level=level, epoch=epoch)


class FakeSession:
    def __init__(self):
        self.epoch, self.mode, self.lights, self.synced = 0, "light", True, True
        self.estimator = self

    def estimate(self, *, now_ns):
        return object() if self.synced else None

    def current_mode(self, now_ns):
        return self.mode

    def current_lights(self, now_ns):
        return self.lights


class FakeClient:
    def __init__(self):
        self.session = FakeSession()
        self.pending, self.statuses = [], []

    def step(self, now_ns):
        outputs, self.pending = self.pending, []
        return outputs

    def set_status(self, status):
        self.statuses.append(status)


class RecordingDispatcher:
    """Record the app boundary only; no claim to reproduce dispatcher safety."""

    def __init__(self):
        self.configs, self.lights, self.moves, self.clips, self.ticks = [], [], [], [], []
        self.closed = False

    def configure(self, **config):
        self.configs.append(config)

    def submit_light(self, intent):
        self.lights.append(intent)

    def submit_move(self, intent):
        self.moves.append(intent)

    def submit_clip(self, intent):
        self.clips.append(intent)

    def tick(self, now_ns):
        self.ticks.append(now_ns)

    def snapshot(self):
        return {}

    def close(self):
        self.closed = True


class LampAppTests(unittest.TestCase):
    def make_app(self, **kwargs):
        client, dispatcher = FakeClient(), RecordingDispatcher()
        app = LampApp(client, dispatcher, output_types=TYPES, **kwargs)
        return app, client, dispatcher

    def test_future_light_waits_until_deadline_and_trim_is_subtracted_once(self):
        app, client, dispatcher = self.make_app(light_trim_ns=7 * MS)
        client.pending = [event(NS + 107 * MS)]
        app.step(NS)
        self.assertEqual(dispatcher.lights, [])
        app.step(NS + 100 * MS - 1)
        self.assertEqual(dispatcher.lights, [])
        app.step(NS + 100 * MS)
        self.assertEqual(len(dispatcher.lights), 1)
        self.assertEqual(dispatcher.lights[0].due_ns, NS + 100 * MS)
        app.step(NS + 400 * MS)
        self.assertEqual(len(dispatcher.lights), 1)

    def test_continuous_100hz_future_samples_do_not_postpone_due_samples(self):
        app, client, dispatcher = self.make_app()
        for index in range(100):
            now = NS + index * 10 * MS
            client.pending = [BassOut([bass(now + 100 * MS)])]
            app.step(now)
            self.assertEqual(len(dispatcher.lights), max(0, index - 9))
            self.assertTrue(all(intent.due_ns <= now for intent in dispatcher.lights))
        self.assertEqual(len(dispatcher.lights), 90)
        self.assertEqual(dispatcher.lights[0].due_ns, NS + 100 * MS)
        self.assertEqual(dispatcher.lights[-1].due_ns, NS + 990 * MS)

    def test_latest_due_light_coalesces_without_losing_future_light(self):
        app, client, dispatcher = self.make_app()
        client.pending = [BassOut([bass(NS + 10 * MS, .1), bass(NS + 20 * MS, .2),
                                   bass(NS + 30 * MS, .3)])]
        app.step(NS)
        app.step(NS + 25 * MS)
        self.assertEqual(len(dispatcher.lights), 1)
        self.assertEqual(dispatcher.lights[0].due_ns, NS + 20 * MS)
        self.assertAlmostEqual(dispatcher.lights[0].rgb[1], .06)
        app.step(NS + 30 * MS)
        self.assertEqual(len(dispatcher.lights), 2)
        self.assertEqual(dispatcher.lights[-1].due_ns, NS + 30 * MS)

    def test_timeline_accepts_at_most_256_pending_samples(self):
        app, client, dispatcher = self.make_app()
        client.pending = [BassOut([bass(NS + (i + 1) * MS) for i in range(300)])]
        app.step(NS)
        self.assertEqual(app.dropped, 44)
        self.assertEqual(dispatcher.lights, [])
        for index in range(1, 301):
            app.step(NS + index * MS)
        self.assertEqual(len(dispatcher.lights), 256)
        self.assertEqual(dispatcher.lights[-1].due_ns, NS + 256 * MS)

    def test_mode_epoch_and_sync_changes_each_clear_pending_timeline(self):
        for change in ("mode", "epoch", "sync"):
            with self.subTest(change=change):
                app, client, dispatcher = self.make_app()
                client.pending = [event(NS + 100 * MS)]
                app.step(NS)
                if change == "mode":
                    client.session.mode = "follow"
                    client.pending = [ModeOut(1)]
                elif change == "epoch":
                    client.session.epoch = 1
                    client.pending = [SessionOut(1)]
                else:
                    client.session.synced = False
                app.step(NS + 50 * MS)
                if change == "sync":
                    client.session.synced = True
                    app.step(NS + 75 * MS)
                app.step(NS + 100 * MS)
                self.assertEqual(dispatcher.lights, [])

    def test_stale_event_and_bass_epochs_produce_no_submissions(self):
        app, client, dispatcher = self.make_app()
        client.session.epoch = 2
        client.pending = [event(NS, epoch=1), BassOut([bass(NS, epoch=1)])]
        app.step(NS)
        self.assertEqual(dispatcher.lights, [])
        self.assertEqual(dispatcher.clips, [])
        self.assertEqual(dispatcher.moves, [])

    def test_last_mode_or_session_barrier_discards_earlier_batch_outputs(self):
        for barrier in (ModeOut(4), SessionOut(0)):
            with self.subTest(barrier=barrier):
                app, client, dispatcher = self.make_app()
                client.pending = [event(NS, .9), ModeOut(2), event(NS, .7), barrier,
                                  event(NS + MS, .2)]
                app.step(NS)
                self.assertEqual(dispatcher.lights, [])
                app.step(NS + MS)
                self.assertEqual(len(dispatcher.lights), 1)
                self.assertAlmostEqual(dispatcher.lights[0].rgb[1], .06)

    def test_missing_follow_and_dance_planners_are_reported(self):
        for mode in ("follow", "dance"):
            app, client, dispatcher = self.make_app(armed=True)
            client.session.mode = mode
            app.step(NS)
            self.assertEqual(client.statuses[-1]["sdk"], "planner unavailable")
            self.assertEqual(client.statuses[-1]["state"], "error")
            self.assertEqual(dispatcher.moves + dispatcher.clips, [])

    def test_constructor_is_disarmed_and_configures_observation_only(self):
        app, client, dispatcher = self.make_app()
        client.pending = [event(NS)]
        app.step(NS)
        self.assertFalse(dispatcher.configs[-1]["armed"])
        self.assertEqual(client.statuses[-1]["sdk"], "observation only")
        self.assertEqual(client.statuses[-1]["state"], "idle")
        # The real dispatcher enforces arming; this recording fake only observes
        # its configured state and does not pretend to execute SDK callbacks.

    def test_armed_unsynchronized_app_reports_error_and_submits_no_light(self):
        app, client, dispatcher = self.make_app(armed=True)
        client.session.synced = False
        client.pending = [event(NS)]
        app.step(NS)
        self.assertEqual(dispatcher.lights, [])
        self.assertFalse(dispatcher.configs[-1]["synced"])
        self.assertEqual(client.statuses[-1]["state"], "error")
        self.assertEqual(client.statuses[-1]["sdk"], "clock unsynchronized")

    def test_readiness_must_survive_future_admission_lease(self):
        for expiry_source in ("clock", "mode", "lights"):
            with self.subTest(expiry_source=expiry_source):
                app, client, dispatcher = self.make_app(armed=True)
                expires = NS + 10 * MS
                if expiry_source == "clock":
                    client.session.estimate = lambda *, now_ns: object() if now_ns < expires else None
                elif expiry_source == "mode":
                    client.session.current_mode = lambda now_ns: "light" if now_ns < expires else "off"
                else:
                    client.session.current_lights = lambda now_ns: now_ns < expires
                # Each signal is valid now; a simple current-time-only check
                # would wrongly admit this immediate event across expiry.
                self.assertIsNotNone(client.session.estimate(now_ns=NS))
                self.assertEqual(client.session.current_mode(NS), "light")
                self.assertTrue(client.session.current_lights(NS))
                client.pending = [event(NS)]
                app.step(NS)
                self.assertEqual(dispatcher.lights, [])
                self.assertFalse(dispatcher.configs[-1]["synced"])
                self.assertEqual(dispatcher.configs[-1]["valid_until_ns"], NS + 20 * MS)
                self.assertEqual(client.statuses[-1]["sdk"], "clock unsynchronized")

    def test_dance_status_requires_actual_inflight_motion(self):
        app, client, dispatcher = self.make_app(
            armed=True, planner=SimpleNamespace(on_event=lambda *_: []))
        client.session.mode = "dance"
        inflight = [False]
        dispatcher.snapshot = lambda: {"motion_inflight": inflight[0]}
        app.step(NS)
        self.assertEqual(client.statuses[-1]["state"], "idle")
        inflight[0] = True
        app.step(NS + MS)
        self.assertEqual(client.statuses[-1]["state"], "dancing")
        inflight[0] = False
        app.step(NS + 2 * MS)
        self.assertEqual(client.statuses[-1]["state"], "idle")

    def test_planner_intents_retain_stale_identity_for_dispatcher_rejection(self):
        move = MoveIntent(dict.fromkeys(JOINTS, 0), NS, 2 * NS, 8, 9)
        clip = ClipIntent("synthetic-clip", NS, 2 * NS, 8, 9)
        planner = SimpleNamespace(follow=lambda *_: move, on_event=lambda *_: [clip])
        app, client, dispatcher = self.make_app(planner=planner, armed=True)
        client.session.mode = "follow"
        app.step(NS)
        client.session.mode = "dance"
        client.pending = [ModeOut(1), event(NS)]
        app.step(NS)
        self.assertEqual((dispatcher.moves[0].epoch, dispatcher.moves[0].gen), (8, 9))
        self.assertEqual((dispatcher.clips[0].epoch, dispatcher.clips[0].gen), (8, 9))

    def test_nonfinite_or_boolean_levels_are_dropped(self):
        app, client, dispatcher = self.make_app()
        client.pending = [BassOut([bass(NS, level) for level in
                                  (float("nan"), float("inf"), -float("inf"), True)])]
        app.step(NS)
        self.assertEqual(app.dropped, 4)
        self.assertEqual(dispatcher.lights, [])

    def test_oversized_numeric_level_is_rejected_without_crashing_loop(self):
        app, client, dispatcher = self.make_app()
        client.pending = [event(NS, 10 ** 400)]
        app.step(NS)
        self.assertEqual(app.dropped, 1)
        self.assertEqual(dispatcher.lights, [])

    def test_light_values_are_clamped_to_dim_nonred_output(self):
        app, client, dispatcher = self.make_app()
        client.pending = [event(NS, 1000)]
        app.step(NS)
        self.assertEqual(dispatcher.lights[0].rgb, (0.0, .3, .15))

    def test_close_clears_pending_timeline_and_closes_dispatcher(self):
        app, client, dispatcher = self.make_app(armed=True)
        client.pending = [event(NS + MS)]
        app.step(NS)
        app.close()
        self.assertFalse(app.armed)
        self.assertTrue(dispatcher.closed)
        app.step(NS + MS)
        self.assertEqual(dispatcher.lights, [])


class FakeClock:
    def __init__(self):
        self.now, self.sleeps = NS, []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += round(seconds * NS)


def action(state="succeeded", action_id="test-action", **extra):
    return {"action": {"action_id": action_id, "state": state, **extra}}


class FakeSDK:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def _call(self, method, path, **kwargs):
        self.calls.append((method, path, kwargs))
        response = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(response, Exception):
            raise response
        return response


class SDKActionsTests(unittest.TestCase):
    def make_actions(self, motion, light=None):
        clock = FakeClock()
        light = light or FakeSDK(action())
        return SDKActions(motion, light, clock=clock, sleep=clock.sleep), clock, light

    def move(self):
        return MoveIntent(dict.fromkeys(JOINTS, 0), NS, 2 * NS, 0, 0)

    def test_post_once_and_read_poll_until_matching_terminal_action(self):
        sdk = FakeSDK(action("accepted"), action("running"),
                      action(result={"completed": True, "reached": True}))
        actions, clock, light = self.make_actions(sdk)
        result = actions.motion(self.move())
        self.assertEqual(result["state"], "succeeded")
        self.assertEqual([method for method, _, _ in sdk.calls], ["POST", "GET", "GET"])
        self.assertEqual([path for _, path, _ in sdk.calls],
                         ["/api/sdk/v1/actions", "/api/sdk/v1/actions/test-action",
                          "/api/sdk/v1/actions/test-action"])
        self.assertEqual(sdk.calls[0][2]["json"]["type"], "motion.move")
        self.assertTrue(sdk.calls[0][2]["json"]["idempotency_key"])
        self.assertEqual(light.calls, [])
        self.assertEqual(len(clock.sleeps), 2)
        self.assertTrue(all(0 < kwargs["timeout"] <= 2 for _, _, kwargs in sdk.calls))

    def test_motion_and_light_use_separate_clients_and_sdk_payloads(self):
        motion, light = FakeSDK(action()), FakeSDK(action())
        actions, _, _ = self.make_actions(motion, light)
        actions.motion(ClipIntent("test-clip", NS, 2 * NS, 0, 0))
        self.assertEqual(light.calls, [])
        actions.light(LightIntent((0.0, .3, .15), NS, 0, 0))
        self.assertEqual(len(motion.calls), 1)
        self.assertEqual(len(light.calls), 1)
        self.assertEqual(motion.calls[0][2]["json"]["payload"], {"clip_id": "test-clip"})
        body = light.calls[0][2]["json"]
        self.assertEqual(body["type"], "light.glow")
        self.assertEqual(body["payload"]["luminance"], .3)
        self.assertEqual(body["payload"]["color"], [0, 76, 38])

    def test_terminal_failures_are_returned_as_failures_without_retry(self):
        for state in ("failed", "rejected", "canceled"):
            with self.subTest(state=state):
                sdk = FakeSDK(action(state))
                actions, _, _ = self.make_actions(sdk)
                self.assertEqual(actions.motion(self.move())["state"], state)
                self.assertEqual(len(sdk.calls), 1)

    def test_malformed_responses_and_initial_ids_are_rejected(self):
        for response in (None, [], {}, {"action": []}, action(action_id=None),
                         action(action_id="../unsafe"), action(action_id=""),
                         action(action_id="x" * 129), action(state="unknown")):
            with self.subTest(response=response):
                sdk = FakeSDK(response)
                actions, _, _ = self.make_actions(sdk)
                with self.assertRaises(RuntimeError):
                    actions.motion(self.move())
                self.assertEqual(len(sdk.calls), 1)

    def test_mismatched_poll_identity_is_rejected_without_new_post(self):
        sdk = FakeSDK(action("accepted"), action(action_id="different-action"))
        actions, _, _ = self.make_actions(sdk)
        with self.assertRaises(RuntimeError):
            actions.motion(self.move())
        self.assertEqual([method for method, _, _ in sdk.calls], ["POST", "GET"])

    def test_timeout_only_polls_never_cancels_retries_or_uses_raw_routes(self):
        sdk = FakeSDK(action("running"))
        actions, clock, _ = self.make_actions(sdk)
        with self.assertRaises(RuntimeError):
            actions.motion(self.move())
        self.assertEqual(sum(method == "POST" for method, _, _ in sdk.calls), 1)
        self.assertTrue(all((method, path) in {
            ("POST", "/api/sdk/v1/actions"), ("GET", "/api/sdk/v1/actions/test-action")
        } for method, path, _ in sdk.calls))
        self.assertEqual(clock.now, 4 * NS)

    def test_ambiguous_post_failure_is_not_retried(self):
        sdk = FakeSDK(OSError("synthetic transport failure"))
        actions, clock, _ = self.make_actions(sdk)
        with self.assertRaises(OSError):
            actions.motion(self.move())
        self.assertEqual(len(sdk.calls), 1)
        self.assertEqual(clock.sleeps, [])


if __name__ == "__main__":
    unittest.main()
