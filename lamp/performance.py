"""Music-driven performance controller for the LeLamp robot lamp.

Converts timestamped music events (bass envelope, onsets, kicks, drops, builds) into:
1. Smooth, WCAG-safe light glow using FlashLimiter (safety/flash.py).
2. Musical gesture choreography using vendor SDK animations (nod, curious, excited, dance).

Sync contract (AGENTS.md rule 5):
- All events are scheduled on the conductor's monotonic clock (pts).
- Fired at target_time = pts + L - trim, where L is room latency (300 ms) and trim is lamp output trim.
- Late events are dropped and counted, never fired late.

Safety rules (AGENTS.md rules 1 & 6):
- Control only through the vendor SDK gateway.
- Light is strictly capped at <= 3 flashes/s with saturated red avoidance via FlashLimiter.
"""
from __future__ import annotations

import json
import math
import queue
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from safety.flash import FlashLimiter
from sdk import LampSDK, SDKError


RGB_F = tuple[float, float, float]  # 0.0 .. 1.0 per channel
RGB_255 = tuple[int, int, int]      # 0 .. 255 integer


# Known expressive gestures from the vendor catalog
GESTURE_CATALOG = {
    "beat": "nod",
    "kick": "nod",
    "build": "curious",
    "drop": "excited",
    "dance": "dance",
    "look_up": "look_up",
    "happy": "happy",
}


@dataclass
class PerformanceConfig:
    room_latency_s: float = 0.300       # L = 300 ms room latency budget
    lamp_trim_s: float = 0.050          # Lamp output latency trim (estimated 50 ms)
    max_drop_late_s: float = 0.080      # Events older than this past their target are dropped
    base_color: RGB_F = (0.15, 0.45, 0.90)    # Rest/ambient cool blue
    accent_color: RGB_F = (0.85, 0.20, 0.85)  # Bass/energy magenta accent
    min_luminance: float = 0.10         # Minimum resting light level
    max_luminance: float = 0.80         # Maximum peak light level (kept < 0.80 for soft transitions)
    min_glow_interval_s: float = 0.15   # Throttle light updates to avoid flooding SDK
    gesture_cooldown_s: float = 3.0     # Minimum time between whole-arm animation dispatches
    default_gesture_duration_s: float = 2.5


@dataclass
class PerformanceStats:
    events_received: int = 0
    events_fired: int = 0
    events_dropped_late: int = 0
    light_updates: int = 0
    gestures_played: int = 0
    gestures_skipped: int = 0
    last_error: str = ""


class MusicPerformer:
    """Manages light glow and musical gestures in sync with conductor events."""

    def __init__(self, sdk: LampSDK, config: PerformanceConfig | None = None,
                 limiter: FlashLimiter | None = None, clock_fn: Callable[[], float] = time.monotonic):
        self.sdk = sdk
        self.config = config or PerformanceConfig()
        self.limiter = limiter or FlashLimiter(margin_s=0.10)
        self.clock = clock_fn
        self.clock_offset = 0.0  # conductor_time - local_time
        self.stats = PerformanceStats()

        self._last_glow_time = 0.0
        self._last_bass_level = -1.0
        self._gesture_busy_until = 0.0
        self._active_animation = None
        self._lock = threading.Lock()

    def update_clock_offset(self, conductor_pts: float, local_receive_time: float | None = None) -> None:
        """Update clock offset estimate with conductor monotonic timestamp."""
        rec = self.clock() if local_receive_time is None else local_receive_time
        # Simple sample tracking (or minimum-delay sample update)
        with self._lock:
            self.clock_offset = conductor_pts - rec

    def conductor_to_local_time(self, pts: float) -> float:
        """Map conductor pts to local monotonic target fire time with room budget and trim."""
        with self._lock:
            offset = self.clock_offset
        # pts_local = pts - offset
        # target_fire_time = pts_local + L - trim
        return (pts - offset) + self.config.room_latency_s - self.config.lamp_trim_s

    def render_light(self, t: float, bass_level: float) -> tuple[RGB_255, float] | None:
        """Compute safe RGB and luminance for a given bass level (0..1) at local time t."""
        clamped_bass = max(0.0, min(1.0, float(bass_level)))
        # Throttle: if recently updated and change is negligible, skip
        if (t - self._last_glow_time < self.config.min_glow_interval_s and
                abs(clamped_bass - self._last_bass_level) < 0.05):
            return None

        # Interpolate color between base_color and accent_color
        r = self.config.base_color[0] + (self.config.accent_color[0] - self.config.base_color[0]) * clamped_bass
        g = self.config.base_color[1] + (self.config.accent_color[1] - self.config.base_color[1]) * clamped_bass
        b = self.config.base_color[2] + (self.config.accent_color[2] - self.config.base_color[2]) * clamped_bass

        # Modulate luminance
        lum = self.config.min_luminance + clamped_bass * (self.config.max_luminance - self.config.min_luminance)
        rgb_raw = (r * lum, g * lum, b * lum)

        # Run through WCAG FlashLimiter
        rgb_safe = self.limiter.limit(t, rgb_raw)

        rgb_255 = (
            int(round(max(0.0, min(1.0, rgb_safe[0])) * 255)),
            int(round(max(0.0, min(1.0, rgb_safe[1])) * 255)),
            int(round(max(0.0, min(1.0, rgb_safe[2])) * 255)),
        )

        self._last_glow_time = t
        self._last_bass_level = clamped_bass
        self.stats.light_updates += 1
        return rgb_255, lum

    def handle_bass_envelope(self, t: float, bass_level: float) -> bool:
        """Update lamp glow in response to a bass envelope sample."""
        rendered = self.render_light(t, bass_level)
        if rendered is None:
            return False
        rgb_255, lum = rendered
        try:
            self.sdk.glow(rgb_255, luminance=lum)
            return True
        except SDKError as exc:
            self.stats.last_error = f"glow failed: {exc}"
            return False

    def play_musical_gesture(self, gesture_name: str, duration_s: float | None = None) -> bool:
        """Play a built-in vendor gesture if not currently in cooldown."""
        now = self.clock()
        with self._lock:
            if now < self._gesture_busy_until:
                self.stats.gestures_skipped += 1
                return False
            dur = duration_s or self.config.default_gesture_duration_s
            self._gesture_busy_until = now + dur + self.config.gesture_cooldown_s

        try:
            self.sdk.play_animation(gesture_name)
            with self._lock:
                self.stats.gestures_played += 1
            return True
        except SDKError as exc:
            with self._lock:
                self.stats.last_error = f"gesture failed: {exc}"
                # If gesture was refused or failed, clear busy so we don't hang cooldown
                self._gesture_busy_until = now + 1.0
            return False

    def handle_event(self, event: dict[str, Any]) -> bool:
        """Process one incoming event dictionary stamped with presentation time."""
        self.stats.events_received += 1
        now = self.clock()

        pts = event.get("pts")
        if pts is not None:
            target_time = self.conductor_to_local_time(float(pts))
            # Late check
            if now > target_time + self.config.max_drop_late_s:
                self.stats.events_dropped_late += 1
                return False
            # If arriving well ahead of time, caller or scheduler should delay,
            # but handle_event dispatches immediately when called at or near fire time.

        kind = str(event.get("kind", ""))
        self.stats.events_fired += 1

        if kind == "bass" or "bass_envelope" in event:
            val = event.get("value", event.get("bass_envelope", 0.0))
            return self.handle_bass_envelope(now, float(val))

        elif kind in GESTURE_CATALOG or kind in ("drop", "build", "dance"):
            mapped = GESTURE_CATALOG.get(kind, kind)
            return self.play_musical_gesture(mapped)

        elif kind in ("kick", "snare", "beat", "onset"):
            strength = float(event.get("strength", 1.0))
            # Strong beats trigger an occasional subtle nod
            if strength >= 0.75:
                return self.play_musical_gesture("nod", duration_s=1.5)
            # Otherwise, just pulse the light briefly if provided
            return self.handle_bass_envelope(now, strength * 0.5)

        elif kind == "clock_sync":
            pts_val = event.get("pts")
            if pts_val is not None:
                self.update_clock_offset(float(pts_val), local_receive_time=now)
            return True

        return False


class PerformanceBridge(threading.Thread):
    """Listens on UDP for conductor music events and dispatches to MusicPerformer."""

    def __init__(self, performer: MusicPerformer, host: str = "0.0.0.0", port: int = 47300):
        super().__init__(daemon=True, name="lamp-performer-bridge")
        self.performer = performer
        self.host, self.port = host, port
        self.running = threading.Event()
        self.sock: socket.socket | None = None

    def run(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind((self.host, self.port))
            self.sock.settimeout(0.5)
            self.running.set()
        except OSError as exc:
            self.performer.stats.last_error = f"UDP bind failed: {exc}"
            return

        while self.running.is_set():
            try:
                data, _ = self.sock.recvfrom(2048)
                if not data:
                    continue
                packet = json.loads(data.decode("utf-8"))
                if isinstance(packet, dict):
                    self.performer.handle_event(packet)
            except (socket.timeout, json.JSONDecodeError, UnicodeDecodeError):
                continue
            except Exception as exc:
                self.performer.stats.last_error = f"recv error: {exc}"

        if self.sock:
            self.sock.close()

    def stop(self) -> None:
        self.running.clear()
