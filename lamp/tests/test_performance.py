import json
import math
import socket
import time
import pytest

from performance import MusicPerformer, PerformanceConfig, PerformanceBridge
from safety.flash import FlashLimiter
from sdk import LampSDK, SDKError


class MockLampSDK(LampSDK):
    fake_sink: bool = True

    def __init__(self):
        self.fake_sink = True
        self.glows = []
        self.animations = []
        self.clips = []

    def _ensure_session(self):
        pass

    def glow(self, rgb, luminance=None):
        self.glows.append((rgb, luminance))
        return {"state": "succeeded"}

    def play_animation(self, name, wait_s=30.0):
        self.animations.append(name)
        return {"state": "succeeded"}

    def play_clip(self, clip_id, wait_s=30.0):
        self.clips.append(clip_id)
        return {"state": "succeeded"}


def test_render_light_scales_luminance_and_respects_limiter():
    sdk = MockLampSDK()
    simulated_time = [100.0]
    performer = MusicPerformer(sdk, clock_fn=lambda: simulated_time[0])

    # Zero bass level -> min_luminance (0.10)
    result = performer.render_light(simulated_time[0], 0.0)
    assert result is not None
    rgb, lum = result
    assert lum == pytest.approx(0.10)
    assert all(0 <= c <= 255 for c in rgb)

    # Move time forward and apply max bass level (1.0) -> max_luminance (0.80)
    simulated_time[0] += 0.20
    result = performer.render_light(simulated_time[0], 1.0)
    assert result is not None
    rgb, lum = result
    assert lum == pytest.approx(0.80)
    assert all(0 <= c <= 255 for c in rgb)


def test_render_light_throttles_negligible_changes():
    sdk = MockLampSDK()
    simulated_time = [100.0]
    config = PerformanceConfig(min_glow_interval_s=0.2)
    performer = MusicPerformer(sdk, config=config, clock_fn=lambda: simulated_time[0])

    # Initial render
    res1 = performer.render_light(simulated_time[0], 0.5)
    assert res1 is not None

    # Immediate second call with tiny change should throttle (return None)
    simulated_time[0] += 0.05
    res2 = performer.render_light(simulated_time[0], 0.51)
    assert res2 is None

    # Substantial change after interval should pass
    simulated_time[0] += 0.20
    res3 = performer.render_light(simulated_time[0], 0.9)
    assert res3 is not None


def test_flash_limiter_caps_rapid_strobing():
    sdk = MockLampSDK()
    simulated_time = [100.0]
    limiter = FlashLimiter(margin_s=0.0)
    config = PerformanceConfig(min_glow_interval_s=0.0)  # no throttle to test limiter
    performer = MusicPerformer(sdk, config=config, limiter=limiter, clock_fn=lambda: simulated_time[0])

    # Strobe 10 times in 1 second between 0 and 1
    outputs = []
    for i in range(20):
        simulated_time[0] += 0.05
        level = 1.0 if i % 2 == 0 else 0.0
        res = performer.render_light(simulated_time[0], level)
        if res:
            outputs.append(res[0])

    # Verify that the limiter held outputs once 3 flashes/s was hit
    held_count = sum(1 for i in range(1, len(outputs)) if outputs[i] == outputs[i - 1])
    assert held_count > 0


def test_gesture_cooldown_prevents_spamming():
    sdk = MockLampSDK()
    simulated_time = [100.0]
    config = PerformanceConfig(gesture_cooldown_s=3.0, default_gesture_duration_s=2.0)
    performer = MusicPerformer(sdk, config=config, clock_fn=lambda: simulated_time[0])

    # First gesture succeeds
    assert performer.play_musical_gesture("nod") is True
    assert sdk.animations == ["nod"]
    assert performer.stats.gestures_played == 1

    # Immediate second gesture is rejected during cooldown
    simulated_time[0] += 1.0
    assert performer.play_musical_gesture("excited") is False
    assert sdk.animations == ["nod"]
    assert performer.stats.gestures_skipped == 1

    # After cooldown finishes (2.0s duration + 3.0s cooldown = 5.0s total)
    simulated_time[0] += 4.5
    assert performer.play_musical_gesture("excited") is True
    assert sdk.animations == ["nod", "excited"]
    assert performer.stats.gestures_played == 2


def test_handle_event_drops_late_events():
    sdk = MockLampSDK()
    simulated_time = [100.0]
    performer = MusicPerformer(sdk, clock_fn=lambda: simulated_time[0])
    performer.clock_offset = 0.0

    # Event with pts = 90.0 (target fire was 90.27s, current time is 100.0s -> over 9s late!)
    late_event = {"pts": 90.0, "kind": "kick", "strength": 1.0}
    fired = performer.handle_event(late_event)

    assert fired is False
    assert performer.stats.events_dropped_late == 1
    assert performer.stats.events_fired == 0


def test_clock_sync_works_when_clocks_far_apart():
    sdk = MockLampSDK()
    # Local clock at 5000.0, conductor at 12.0
    simulated_time = [5000.0]
    performer = MusicPerformer(sdk, clock_fn=lambda: simulated_time[0])

    sync_event = {"pts": 12.0, "kind": "clock_sync"}
    assert performer.handle_event(sync_event) is True
    # Offset is 12.0 - 5000.0 = -4988.0
    assert performer.clock_offset == pytest.approx(-4988.0)

    # Now a music event on conductor clock 12.0 + 0.5 = 12.5
    # target = (12.5 - (-4988.0)) + 0.300 - 0.030 = 5000.5 + 0.27 = 5000.77
    music_event = {"pts": 12.0 + (simulated_time[0] - 5000.0) - 0.27, "kind": "bass", "value": 0.5}
    assert performer.handle_event(music_event) is True


def test_nan_and_inf_pts_safely_rejected():
    sdk = MockLampSDK()
    simulated_time = [100.0]
    performer = MusicPerformer(sdk, clock_fn=lambda: simulated_time[0])

    nan_event = {"pts": float("nan"), "kind": "kick"}
    assert performer.handle_event(nan_event) is False

    inf_event = {"pts": float("inf"), "kind": "kick"}
    assert performer.handle_event(inf_event) is False


def test_refusal_limit_latches_off():
    sdk = MockLampSDK()
    simulated_time = [100.0]
    # Simulate repeated 409 refusals
    sdk.play_animation = lambda *args, **kwargs: (_ for _ in ()).throw(
        SDKError(409, "lost_track", "action outcome unknown")
    )
    performer = MusicPerformer(sdk, clock_fn=lambda: simulated_time[0])

    for i in range(3):
        simulated_time[0] += 5.0
        assert performer.play_musical_gesture("nod") is False

    assert performer.latched_off is True
    assert performer.stats.terminal_refusals >= 1

    # Further attempts immediately rejected without invoking SDK
    assert performer.play_musical_gesture("nod") is False


def test_weak_kick_does_not_trigger_arm_nod():
    sdk = MockLampSDK()
    simulated_time = [100.0]
    config = PerformanceConfig(mode="dance")  # Dance mode allows arm gestures
    performer = MusicPerformer(sdk, config=config, clock_fn=lambda: simulated_time[0])

    on_time_pts = 100.0 - (0.300 - 0.030)  # target = 100.0s

    # Weak kick (strength 0.3) -> pulses light, but does NOT play whole-arm animation
    weak_kick = {"pts": on_time_pts, "kind": "kick", "strength": 0.3}
    assert performer.handle_event(weak_kick) is True
    assert len(sdk.animations) == 0

    # Strong kick (strength 0.85) in dance mode -> triggers whole-arm nod
    strong_kick = {"pts": on_time_pts, "kind": "kick", "strength": 0.85}
    assert performer.handle_event(strong_kick) is True
    assert sdk.animations == ["nod"]


def test_mode_arbitration_follow_vs_dance():
    sdk = MockLampSDK()
    simulated_time = [100.0]
    # Default mode is "follow" (head tracking locked, arm gestures suppressed)
    performer = MusicPerformer(sdk, clock_fn=lambda: simulated_time[0])
    on_time_pts = 100.0 - (0.300 - 0.030)

    strong_kick = {"pts": on_time_pts, "kind": "kick", "strength": 0.95}

    # In follow mode: light glows, but whole-arm gestures are strictly suppressed
    assert performer.handle_event(strong_kick) is True
    assert len(sdk.animations) == 0
    assert performer.stats.gestures_skipped == 1
    assert len(sdk.glows) == 1

    # Switch to dance mode: whole-arm gestures now fire
    performer.set_mode("dance")
    assert performer.handle_event(strong_kick) is True
    assert sdk.animations == ["nod"]
    assert performer.stats.gestures_played == 1

    # Switch to off mode: everything suppressed
    performer.set_mode("off")
    assert performer.handle_event(strong_kick) is False


def test_events_without_pts_rejected():
    sdk = MockLampSDK()
    performer = MusicPerformer(sdk)
    # Rule 5: Events without pts violate shared-clock scheduling
    assert performer.handle_event({"kind": "kick", "strength": 0.9}) is False
    assert performer.stats.events_dropped_no_pts == 1


def test_far_future_events_dropped():
    sdk = MockLampSDK()
    simulated_time = [100.0]
    performer = MusicPerformer(sdk, clock_fn=lambda: simulated_time[0])

    # Event target is 102.0s (2.0s in future, way beyond max_early_hold_s 0.350s)
    future_pts = 102.0 - (0.300 - 0.030)
    assert performer.handle_event({"pts": future_pts, "kind": "kick"}) is False
    assert performer.stats.events_dropped_far_future == 1
    assert performer.stats.events_fired == 0


def test_performance_bridge_udp_sender_pinning():
    sdk = MockLampSDK()
    performer = MusicPerformer(sdk)
    bridge = PerformanceBridge(performer, host="127.0.0.1", port=0)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    bridge.port = sock.getsockname()[1]
    sock.close()

    bridge.start()
    time.sleep(0.1)

    try:
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Supply valid pts in packet
        target_pts = time.monotonic() - (0.300 - 0.030)
        packet = json.dumps({"pts": target_pts, "kind": "bass", "value": 0.65}).encode("utf-8")
        sender.sendto(packet, ("127.0.0.1", bridge.port))
        sender.close()

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and performer.stats.events_received == 0:
            time.sleep(0.02)

        assert performer.stats.events_received >= 1
    finally:
        bridge.stop()
        bridge.join(timeout=1.0)


def test_handle_conductor_wire_events():
    sdk = MockLampSDK()
    simulated_time = [500.0]
    config = PerformanceConfig(mode="dance")
    performer = MusicPerformer(sdk, config=config, clock_fn=lambda: simulated_time[0])
    performer.clock_offset = 0.0

    # Conductor Event with integer ns pts and fixed-point amp (0..1000)
    # target_time = pts_s + 0.270 => for target_time == 500.0, pts_s = 500.0 - 0.270
    target_pts_ns = int((500.0 - 0.27) * 1e9)
    wire_kick = {
        "v": 1,
        "t": "event",
        "seq": 10,
        "kind": "kick",
        "pts": target_pts_ns,
        "payload": {"amp": 850},
    }
    assert performer.handle_event(wire_kick) is True
    assert sdk.animations == ["nod"]
    assert len(sdk.glows) == 1

    # Conductor Bass event with fixed-point level
    wire_bass = {
        "v": 1,
        "t": "event",
        "seq": 11,
        "kind": "bass",
        "pts": target_pts_ns,
        "payload": {"level": 600},
    }
    assert performer.handle_event(wire_bass) is True


def test_bridge_answers_conductor_probe():
    sdk = MockLampSDK()
    performer = MusicPerformer(sdk)
    bridge = PerformanceBridge(performer, host="127.0.0.1", port=0)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    bridge.port = sock.getsockname()[1]
    sock.close()

    bridge.start()
    time.sleep(0.05)

    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client.settimeout(1.0)
    try:
        probe = json.dumps({"v": 1, "t": "probe", "id": 42, "t0": 1000000}).encode("utf-8")
        client.sendto(probe, ("127.0.0.1", bridge.port))
        data, _ = client.recvfrom(2048)
        reply = json.loads(data.decode("utf-8"))
        assert reply["t"] == "probe_reply"
        assert reply["id"] == 42
        assert reply["t0"] == 1000000
        assert "t1" in reply and "t2" in reply
    finally:
        client.close()
        bridge.stop()
        bridge.join(timeout=1.0)


def test_dry_run_subclass_wrapper_does_not_emit_real_calls():
    class WrappedLampSDK(LampSDK):
        def __init__(self):
            self.glows = []
            self.animations = []

        def _ensure_session(self):
            pass

        def glow(self, rgb, luminance=None):
            self.glows.append(rgb)
            return {"state": "succeeded"}

        def play_animation(self, name, wait_s=30.0):
            self.animations.append(name)
            return {"state": "succeeded"}

    wrapped_sdk = WrappedLampSDK()
    # live_mode is False by default, fake_sink is False by default
    performer = MusicPerformer(wrapped_sdk, clock_fn=lambda: 100.0)
    performer.clock_offset = 0.0

    # Test bass glow in dry run
    ok = performer.handle_event({"pts": 99.73, "kind": "bass", "value": 0.5})
    assert ok is True
    assert len(wrapped_sdk.glows) == 0  # Dry-run suppressed calls to wrapped SDK!

    # Test gesture in dry run
    performer.set_mode("dance")
    ok = performer.play_musical_gesture("nod")
    assert ok is True
    assert len(wrapped_sdk.animations) == 0  # Dry-run suppressed gesture!


def test_mode_switch_to_off_during_sleep_invalidates_queued_event():
    sdk = MockLampSDK()
    simulated_time = [100.0]

    def sleep_advance_and_turn_off(dur):
        simulated_time[0] += dur
        performer.set_mode("off")

    performer = MusicPerformer(
        sdk,
        clock_fn=lambda: simulated_time[0],
        sleep_fn=sleep_advance_and_turn_off,
    )
    performer.clock_offset = 0.0

    # Event arrives with lead time (target_time = 99.83 + 0.300 - 0.030 = 100.10)
    ok = performer.handle_event({"pts": 99.83, "kind": "bass", "value": 0.5})
    assert ok is False
    assert len(sdk.glows) == 0  # No glow was emitted because mode became 'off' during sleep!


def test_mode_switch_from_dance_to_follow_during_sleep_suppresses_gesture():
    sdk = MockLampSDK()
    simulated_time = [100.0]

    def sleep_advance_and_switch_follow(dur):
        simulated_time[0] += dur
        performer.set_mode("follow")

    performer = MusicPerformer(
        sdk,
        config=PerformanceConfig(mode="dance"),
        clock_fn=lambda: simulated_time[0],
        sleep_fn=sleep_advance_and_switch_follow,
    )
    performer.clock_offset = 0.0

    # Event arrives with lead time in dance mode, but during sleep switched to follow
    ok = performer.handle_event({"pts": 99.83, "kind": "kick", "strength": 0.9})
    assert ok is False  # Invalidation dropped the stale generation event
    assert len(sdk.animations) == 0  # No dance gesture emitted

