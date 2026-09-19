import json
import socket
import time
import pytest

from performance import MusicPerformer, PerformanceConfig, PerformanceBridge
from safety.flash import FlashLimiter
from sdk import LampSDK, SDKError


class MockLampSDK(LampSDK):
    def __init__(self):
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
    # (Consecutive identical outputs mean the limiter held the last state)
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
    # L = 0.3s, trim = 0.05s, target = pts + 0.25s. max_drop_late = 0.08s
    performer = MusicPerformer(sdk, clock_fn=lambda: simulated_time[0])
    performer.clock_offset = 0.0

    # Event with pts = 90.0 (target fire was 90.25s, current time is 100.0s -> over 9s late!)
    late_event = {"pts": 90.0, "kind": "kick", "strength": 1.0}
    fired = performer.handle_event(late_event)

    assert fired is False
    assert performer.stats.events_dropped_late == 1
    assert performer.stats.events_fired == 0


def test_handle_event_dispatches_kinds():
    sdk = MockLampSDK()
    simulated_time = [100.0]
    performer = MusicPerformer(sdk, clock_fn=lambda: simulated_time[0])

    # On-time bass event: target = 99.75 + 0.25 = 100.0
    bass_event = {"pts": 99.75, "kind": "bass", "value": 0.8}
    assert performer.handle_event(bass_event) is True
    assert len(sdk.glows) == 1

    # On-time drop event
    drop_event = {"pts": 99.75, "kind": "drop"}
    assert performer.handle_event(drop_event) is True
    assert sdk.animations == ["excited"]

    # Clock sync event
    sync_event = {"pts": 200.0, "kind": "clock_sync"}
    assert performer.handle_event(sync_event) is True
    assert performer.clock_offset == pytest.approx(100.0)  # 200.0 - 100.0


def test_handle_error_gracefully():
    sdk = MockLampSDK()
    simulated_time = [100.0]
    sdk.glow = lambda *args, **kwargs: (_ for _ in ()).throw(SDKError(400, "bad_request", "invalid"))
    sdk.play_animation = lambda *args, **kwargs: (_ for _ in ()).throw(SDKError(409, "rejected", "busy"))

    performer = MusicPerformer(sdk, clock_fn=lambda: simulated_time[0])

    # Glow error caught
    assert performer.handle_bass_envelope(simulated_time[0], 0.7) is False
    assert "glow failed" in performer.stats.last_error

    # Gesture error caught
    assert performer.play_musical_gesture("nod") is False
    assert "gesture failed" in performer.stats.last_error


def test_performance_bridge_udp():
    sdk = MockLampSDK()
    performer = MusicPerformer(sdk)
    bridge = PerformanceBridge(performer, host="127.0.0.1", port=0)
    # Bind socket to an OS-assigned ephemeral port
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    bridge.port = sock.getsockname()[1]
    sock.close()

    bridge.start()
    time.sleep(0.1)

    try:
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        packet = json.dumps({"kind": "bass", "value": 0.65}).encode("utf-8")
        sender.sendto(packet, ("127.0.0.1", bridge.port))
        sender.close()

        # Wait briefly for bridge to receive and process
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and performer.stats.events_received == 0:
            time.sleep(0.02)

        assert performer.stats.events_received >= 1
    finally:
        bridge.stop()
        bridge.join(timeout=1.0)
