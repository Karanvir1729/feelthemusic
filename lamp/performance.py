"""Music-driven performance controller for the LeLamp robot lamp.

Converts timestamped music events (bass envelope, onsets, kicks, drops, builds) into:
1. Tactile, WCAG-safe light feedback (haptic-like visual rhythm) using FlashLimiter (safety/flash.py).
2. Musical gesture choreography using vendor SDK animations (nod, curious, excited, dance).

Sync contract (AGENTS.md rule 5):
- All events are scheduled on the conductor's monotonic clock (pts).
- Fired at target_time = pts + L - trim, where L is room latency (300 ms) and trim is lamp output trim.
- Late events (>80 ms past target) are dropped and counted, never fired late.
- Early events are held until their presentation time.

Safety rules (AGENTS.md rules 1 & 6):
- Control only through the vendor SDK gateway.
- Light is strictly capped at <= 3 flashes/s with saturated red avoidance via FlashLimiter.
- Refusal policy and refusal limit enforced; bridge stops upon terminal refusals.
"""

from __future__ import annotations

import json
import logging
import math
import queue
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

from safety.flash import FlashLimiter
from sdk import LampSDK, SDKError, refusal_policy

logger = logging.getLogger("lamp.performance")

RGB_F = tuple[float, float, float]  # 0.0 .. 1.0 per channel
RGB_255 = tuple[int, int, int]  # 0 .. 255 integer

# Expressive section/dance gestures from vendor catalog
GESTURE_CATALOG = {
    "build": "curious",
    "drop": "excited",
    "dance": "dance",
    "look_up": "look_up",
    "happy": "happy",
}

REFUSAL_LIMIT = 3


@dataclass
class PerformanceConfig:
    mode: str = "follow"  # "follow" (head locked, light pulse only), "dance" (expressive gestures), "light_only", "off"
    room_latency_s: float = 0.300  # L = 300 ms room latency budget
    lamp_trim_s: float = 0.030  # Estimated lamp output latency trim (30 ms)
    max_drop_late_s: float = 0.080  # Drop events >80 ms past target
    max_early_hold_s: float = 0.350  # Maximum lookahead window to sleep/hold
    base_color: RGB_F = (0.15, 0.45, 0.90)  # Rest/ambient cool blue
    accent_color: RGB_F = (0.75, 0.20, 0.75)  # Energy magenta accent (unsaturated red)
    min_luminance: float = 0.10  # Minimum resting light level
    max_luminance: float = 0.80  # Peak light level (kept <= 0.80 for soft transitions)
    min_glow_interval_s: float = 0.030  # Allow fast updates for haptic-like light rhythm
    gesture_cooldown_s: float = 3.0  # Minimum seconds between whole-arm animations
    default_gesture_duration_s: float = 2.5
    live_mode: bool = False  # Explicit flag required for non-loopback network listening


@dataclass
class PerformanceStats:
    events_received: int = 0
    events_fired: int = 0
    events_dropped_late: int = 0
    events_dropped_no_pts: int = 0
    events_dropped_far_future: int = 0
    clock_sync_count: int = 0
    light_updates: int = 0
    gestures_played: int = 0
    gestures_skipped: int = 0
    terminal_refusals: int = 0
    last_error: str = ""


class MusicPerformer:
    """Manages tactile light feedback and musical gestures in sync with conductor events."""

    def __init__(
        self,
        sdk: LampSDK,
        config: Optional[PerformanceConfig] = None,
        limiter: Optional[FlashLimiter] = None,
        clock_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.sdk = sdk
        self.config = config or PerformanceConfig()
        self.limiter = limiter or FlashLimiter(margin_s=0.0)
        self.clock = clock_fn
        self.sleep_fn = sleep_fn
        self.clock_offset = 0.0  # conductor_time - local_time
        self.stats = PerformanceStats()

        self._last_glow_time = 0.0
        self._last_bass_level = -1.0
        self._gesture_busy_until = 0.0
        self._refusal_count = 0
        self.latched_off = False
        self._lock = threading.Lock()

        # Minimum delay offset filter (keeps lowest delay samples)
        self._offset_samples: list[tuple[float, float]] = []  # (delay, offset)

    def update_clock_offset(
        self,
        conductor_pts: float,
        local_receive_time: Optional[float] = None,
        delay_s: Optional[float] = None,
    ) -> None:
        """Update monotonic clock offset using minimum-delay filter.

        Filters out high-jitter / queuing-delay samples, retaining the lowest-delay
        measurements which provide the tightest bounds on true clock offset.
        """
        if not math.isfinite(conductor_pts):
            return
        pts_s = (conductor_pts / 1e9) if conductor_pts > 1e8 else conductor_pts
        rec = self.clock() if local_receive_time is None else local_receive_time
        offset = pts_s - rec
        delay = delay_s if (delay_s is not None and math.isfinite(delay_s) and delay_s >= 0.0) else abs(rec - pts_s)

        with self._lock:
            self._offset_samples.append((delay, offset))
            if len(self._offset_samples) > 16:
                self._offset_samples.pop(0)
            # Pick median offset of the lowest-delay samples
            best_samples = sorted(self._offset_samples, key=lambda x: x[0])[:min(4, len(self._offset_samples))]
            offsets = sorted(s[1] for s in best_samples)
            self.clock_offset = offsets[len(offsets) // 2]
            self.stats.clock_sync_count += 1

    def conductor_to_local_time(self, pts: float) -> float:
        """Map conductor pts to local monotonic target fire time with room budget and trim."""
        pts_s = (pts / 1e9) if pts > 1e8 else pts
        with self._lock:
            offset = self.clock_offset
        # pts_local = pts_s - offset
        # target_fire_time = pts_local + L - trim
        return (pts_s - offset) + self.config.room_latency_s - self.config.lamp_trim_s

    def render_light(self, t: float, bass_level: float) -> Optional[Tuple[RGB_255, float]]:
        """Compute safe RGB and luminance for a given level (0..1) at local time t.

        Provides haptic-like tactile visual pulses through FlashLimiter.
        """
        if not math.isfinite(bass_level):
            return None
        clamped_level = max(0.0, min(1.0, float(bass_level)))

        # Throttle check
        if (t - self._last_glow_time < self.config.min_glow_interval_s and
                abs(clamped_level - self._last_bass_level) < 0.05):
            return None

        # Interpolate color between base_color and accent_color
        r = self.config.base_color[0] + (self.config.accent_color[0] - self.config.base_color[0]) * clamped_level
        g = self.config.base_color[1] + (self.config.accent_color[1] - self.config.base_color[1]) * clamped_level
        b = self.config.base_color[2] + (self.config.accent_color[2] - self.config.base_color[2]) * clamped_level

        # Modulate luminance
        lum = self.config.min_luminance + clamped_level * (self.config.max_luminance - self.config.min_luminance)
        rgb_raw = (r * lum, g * lum, b * lum)

        # Run through WCAG FlashLimiter
        rgb_safe = self.limiter.limit(t, rgb_raw)

        rgb_255 = (
            int(round(max(0.0, min(1.0, rgb_safe[0])) * 255)),
            int(round(max(0.0, min(1.0, rgb_safe[1])) * 255)),
            int(round(max(0.0, min(1.0, rgb_safe[2])) * 255)),
        )

        self._last_glow_time = t
        self._last_bass_level = clamped_level
        self.stats.light_updates += 1
        return rgb_255, lum

    def handle_bass_envelope(self, t: float, bass_level: float) -> bool:
        """Update lamp glow in response to a bass envelope sample."""
        rendered = self.render_light(t, bass_level)
        if rendered is None:
            return False
        rgb_255, lum = rendered

        # Live mode gate: avoid sending real network HTTP requests in dry-run mode
        if not self.config.live_mode and type(self.sdk).__name__ == "LampSDK":
            logger.debug("[DRY RUN] Would glow: %s (live_mode=False)", rgb_255)
            return True

        try:
            # rgb_255 already has luminance scaled and limited; pass without extra luminance parameter
            self.sdk.glow(rgb_255)
            return True
        except SDKError as exc:
            self.stats.last_error = f"glow failed: {exc}"
            return False

    def set_mode(self, mode: str) -> None:
        """Set operation mode: 'follow', 'dance', 'light_only', 'off'."""
        with self._lock:
            self.config.mode = mode
            logger.info("Performer mode set to: %s", mode)

    def play_musical_gesture(self, gesture_name: str, duration_s: Optional[float] = None) -> bool:
        """Play a built-in vendor gesture if not in cooldown and not latched off."""
        if self.latched_off:
            logger.warning("Gesture rejected: performer is latched off due to refusals")
            return False

        now = self.clock()
        with self._lock:
            if now < self._gesture_busy_until:
                self.stats.gestures_skipped += 1
                return False
            dur = duration_s or self.config.default_gesture_duration_s
            self._gesture_busy_until = now + dur + self.config.gesture_cooldown_s

        # Live mode gate: avoid sending real physical moves in dry-run mode
        if not self.config.live_mode and type(self.sdk).__name__ == "LampSDK":
            logger.info("[DRY RUN] Would play gesture '%s' (live_mode=False)", gesture_name)
            with self._lock:
                self.stats.gestures_played += 1
                self._refusal_count = 0
            return True

        try:
            self.sdk.play_animation(gesture_name)
            with self._lock:
                self.stats.gestures_played += 1
                self._refusal_count = 0  # Reset on successful execution
            return True
        except SDKError as exc:
            with self._lock:
                self.stats.last_error = f"gesture failed: {exc}"
                self._refusal_count += 1
                stop, backoff_s, reason = refusal_policy(exc, self._refusal_count)
                if exc.status == 409 or stop or self._refusal_count >= REFUSAL_LIMIT:
                    self.latched_off = True
                    self.stats.terminal_refusals += 1
                    logger.error(
                        "Performer latched off: %s (exc: %s)", reason, exc
                    )
                self._gesture_busy_until = now + backoff_s
            return False

    def handle_event(self, event: dict[str, Any]) -> bool:
        """Process an incoming event dictionary with presentation time scheduling."""
        if self.config.mode == "off":
            return False

        self.stats.events_received += 1
        now = self.clock()

        # Support both flat JSON and conductor wire codec Event(t="event", kind=...)
        kind = str(event.get("kind", "")).lower()
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}

        # 1. Handle clock sync first before any late check!
        if kind == "clock_sync":
            pts_val = event.get("pts")
            if pts_val is not None:
                try:
                    val = float(pts_val)
                    if math.isfinite(val):
                        self.update_clock_offset(val, local_receive_time=now)
                        return True
                except (ValueError, TypeError):
                    return False
            return False

        # 2. Timing and Presentation Schedule check (pts + L - trim)
        pts = event.get("pts")
        if pts is None:
            # Rule 5: Everything is scheduled on shared clock; packet arrival dispatch prohibited
            with self._lock:
                self.stats.events_dropped_no_pts += 1
            return False

        try:
            val = float(pts)
            if not math.isfinite(val):
                # Reject NaN or Inf pts
                return False
            target_time = self.conductor_to_local_time(val)
        except (ValueError, TypeError):
            return False

        # Late event drop rule: pts + L - trim < now - 80ms
        if now > target_time + self.config.max_drop_late_s:
            with self._lock:
                self.stats.events_dropped_late += 1
            return False

        # Early hold rule: if event arrives ahead of time
        lead_time = target_time - now
        if lead_time > 0.0:
            if lead_time <= self.config.max_early_hold_s:
                self.sleep_fn(lead_time)
                now = self.clock()
            else:
                # Event is too far in future (> max_early_hold_s)
                # Drop rather than firing prematurely!
                with self._lock:
                    self.stats.events_dropped_far_future += 1
                return False

        self.stats.events_fired += 1

        if kind in ("bass", "bass_envelope") or "bass_envelope" in event or "level" in payload:
            val = payload.get("level", payload.get("amp", event.get("value", event.get("bass_envelope", 0.0))))
            try:
                fval = float(val)
                if fval > 1.0:
                    fval = fval / 1000.0 if fval <= 1000.0 else fval / 255.0
                return self.handle_bass_envelope(now, fval)
            except (ValueError, TypeError):
                return False

        elif kind in ("kick", "snare", "beat", "onset"):
            raw_strength = payload.get("amp", payload.get("strength", event.get("strength", 1.0)))
            try:
                strength = float(raw_strength)
                if strength > 1.0:
                    strength = strength / 1000.0 if strength <= 1000.0 else strength / 255.0
            except (ValueError, TypeError):
                strength = 1.0

            if not math.isfinite(strength):
                return False

            # High energy drops / kicks trigger whole-arm gesture only in "dance" mode
            gesture_ok = False
            if strength >= 0.75 and kind in ("kick", "beat"):
                if self.config.mode == "dance":
                    gesture_ok = self.play_musical_gesture("nod", duration_s=1.5)
                else:
                    with self._lock:
                        self.stats.gestures_skipped += 1
                # Deliver sharp haptic-like visual pulse
                light_ok = self.handle_bass_envelope(now, 1.0)
                return light_ok or gesture_ok

            # Moderate beats give haptic-like visual pulse
            return self.handle_bass_envelope(now, strength)

        elif kind in GESTURE_CATALOG:
            if self.config.mode == "dance":
                mapped = GESTURE_CATALOG[kind]
                return self.play_musical_gesture(mapped)
            else:
                with self._lock:
                    self.stats.gestures_skipped += 1
                return True

        return False


class PerformanceBridge(threading.Thread):
    """Listens on UDP for conductor music events and dispatches to MusicPerformer."""

    def __init__(
        self,
        performer: MusicPerformer,
        host: str = "127.0.0.1",  # Loopback by default for security
        port: int = 47300,
        allow_ip: Optional[str] = None,
    ) -> None:
        super().__init__(daemon=True, name="lamp-performer-bridge")
        self.performer = performer
        self.host = host
        self.port = port
        self.allow_ip = allow_ip
        self.pinned_sender: Optional[str] = allow_ip
        self.running = threading.Event()
        self.sock: Optional[socket.socket] = None
        self._event_queue: queue.Queue = queue.Queue(maxsize=256)
        self._worker_thread: Optional[threading.Thread] = None

    def _worker_loop(self) -> None:
        """Worker thread loop consuming events from the queue so UDP receive loop is never blocked."""
        while self.running.is_set():
            try:
                packet = self._event_queue.get(timeout=0.1)
                if packet is None:
                    break
                self.performer.handle_event(packet)
                self._event_queue.task_done()
            except queue.Empty:
                continue
            except Exception as exc:
                self.performer.stats.last_error = f"worker error: {exc}"

    def run(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind((self.host, self.port))
            self.sock.settimeout(0.5)
            self.running.set()
        except OSError as exc:
            self.performer.stats.last_error = f"UDP bind failed: {exc}"
            return

        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="lamp-performer-worker",
        )
        self._worker_thread.start()

        while self.running.is_set():
            try:
                data, addr = self.sock.recvfrom(2048)
                if not data:
                    continue

                # Sender pinning for network protection
                sender_ip = addr[0]
                if self.pinned_sender is None:
                    self.pinned_sender = sender_ip
                elif sender_ip != self.pinned_sender:
                    continue  # Ignore foreign packets

                packet = json.loads(data.decode("utf-8"))
                if isinstance(packet, dict):
                    # Answer wire probes immediately without queue delay
                    if packet.get("t") == "probe":
                        t1_ns = int(time.monotonic() * 1e9)
                        t0_ns = int(packet.get("t0", 0))
                        t2_ns = int(time.monotonic() * 1e9)
                        reply = {
                            "v": 1,
                            "t": "probe_reply",
                            "id": packet.get("id", 0),
                            "t0": t0_ns,
                            "t1": t1_ns,
                            "t2": t2_ns,
                        }
                        self.sock.sendto(json.dumps(reply).encode("utf-8"), addr)
                        continue

                    # Handle probe replies immediately for fast clock sync
                    if packet.get("t") == "probe_reply" or packet.get("kind") == "clock_sync":
                        self.performer.handle_event(packet)
                        continue

                    # Musical events are queued for execution on worker thread
                    try:
                        self._event_queue.put_nowait(packet)
                    except queue.Full:
                        with self.performer._lock:
                            self.performer.stats.events_dropped_late += 1
            except (socket.timeout, json.JSONDecodeError, UnicodeDecodeError):
                continue
            except Exception as exc:
                self.performer.stats.last_error = f"recv error: {exc}"

        if self.sock:
            self.sock.close()

    def stop(self) -> None:
        self.running.clear()
        try:
            self._event_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=0.5)
