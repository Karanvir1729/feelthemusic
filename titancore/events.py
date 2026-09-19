"""Musical event mapping and presentation scheduling for TITAN Core haptic kit.

Translates conductor timing and music analysis events (kicks, snares, bass envelope)
into TITAN Core serial commands, scheduled against the shared monotonic clock:
    schedule_time = pts + L - trim
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Optional

from titancore.driver import TitanDriver


class TitanEventMapper:
    """Dispatches musical events to TITAN Core driver with presentation time gating."""

    DEFAULT_BUDGET_S: float = 0.300  # 300 ms room latency budget L
    DEFAULT_TRIM_S: float = 0.005  # 5 ms serial transport trim
    LATE_DROP_THRESHOLD_S: float = 0.080  # Drop events >80 ms past target time

    def __init__(
        self,
        driver: TitanDriver,
        latency_budget_s: float = DEFAULT_BUDGET_S,
        trim_s: float = DEFAULT_TRIM_S,
        time_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        self.driver = driver
        self.latency_budget_s = latency_budget_s
        self.trim_s = trim_s
        self.time_fn = time_fn or time.monotonic
        self._phase: float = 0.0

        # Metrics
        self.processed_events: int = 0
        self.dropped_late_events: int = 0
        self.kicks_fired: int = 0
        self.snares_fired: int = 0
        self.bass_frames_sent: int = 0

    def synthesize_bass_frame(
        self,
        intensity: float,
        freq_hz: float = 65.0,
        sample_rate: int = 200,
        num_samples: int = 16,
    ) -> List[int]:
        """Synthesize a continuous sub-bass waveform frame centered at 128."""
        clamped_intensity = max(0.0, min(1.0, float(intensity)))
        samples: List[int] = []
        phase_step = 2.0 * math.pi * freq_hz / sample_rate

        for _ in range(num_samples):
            # Scale +/- 127 around rest value 128
            osc = math.sin(self._phase)
            sample_val = int(round(128.0 + (127.0 * clamped_intensity * osc)))
            samples.append(max(0, min(255, sample_val)))
            self._phase += phase_step
            if self._phase > 2.0 * math.pi:
                self._phase -= 2.0 * math.pi

        return samples

    def handle_event(self, event: Dict[str, Any], now: Optional[float] = None) -> bool:
        """Process a conductor event and dispatch to TITAN Core hardware.

        Returns True if the event was dispatched, False if dropped (e.g. late).
        """
        current_time = self.time_fn() if now is None else now

        # Parse presentation timestamp (pts)
        raw_pts = event.get("pts")
        if raw_pts is not None:
            # Check if integer nanoseconds (> 1e12) or floating seconds
            pts_s = (float(raw_pts) / 1e9) if raw_pts > 1e12 else float(raw_pts)
            target_presentation_s = pts_s + self.latency_budget_s - self.trim_s

            # Late event drop rule: pts + L - trim < now - 80ms
            if current_time > (target_presentation_s + self.LATE_DROP_THRESHOLD_S):
                self.dropped_late_events += 1
                return False

        event_type = str(event.get("type", "")).lower()
        intensity = float(event.get("intensity", 1.0))
        intensity = max(0.0, min(1.0, intensity))

        self.processed_events += 1

        if event_type == "kick":
            # Channel M high-impact punch + low-end thud on L/R
            amplitude = int(round(255 * intensity))
            duration_ms = int(event.get("duration_ms", 30))
            transient_ok = self.driver.send_transient(amplitude=amplitude, duration_ms=duration_ms)
            if transient_ok:
                self.kicks_fired += 1
            # Complement with synthesized kick sub-bass thud
            thud = self.synthesize_bass_frame(intensity=intensity, freq_hz=55.0, num_samples=16)
            self.driver.send_pcm(thud)
            return True

        elif event_type == "snare":
            # Channel M crisp high-energy snap
            amplitude = int(round(200 * intensity))
            duration_ms = int(event.get("duration_ms", 15))
            transient_ok = self.driver.send_transient(amplitude=amplitude, duration_ms=duration_ms)
            if transient_ok:
                self.snares_fired += 1
            return True

        elif event_type in ("bass_envelope", "bass", "envelope"):
            # Continuous sub-bass wave on Channels L/R (PAM8403)
            freq_hz = float(event.get("freq_hz", 65.0))
            frame = self.synthesize_bass_frame(intensity=intensity, freq_hz=freq_hz, num_samples=16)
            self.driver.send_pcm(frame)
            self.bass_frames_sent += 1
            return True

        elif event_type == "stop":
            self.driver.emergency_stop()
            return True

        return False
