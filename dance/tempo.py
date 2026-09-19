"""Conservative pulse prediction from presentation timestamps, not packet arrival.

Written 2026-09-19. This estimates a regular event pulse, not musical metre or
the true downbeat. Half/double-time ambiguity cannot be solved from timestamps
alone. An upstream adapter deduplicates packets and supplies local-clock times.
The caller must reset on session, mode or synchronization changes.
"""

from collections import deque
from dataclasses import dataclass
from statistics import median


@dataclass(frozen=True)
class Pulse:
    period_ns: int
    anchor_ns: int

    def next_at_or_after(self, earliest_ns: int) -> int:
        """Choose an observed-phase pulse after the required preparation window."""
        if type(earliest_ns) is not int or earliest_ns < 0:
            raise ValueError("earliest_ns must be a nonnegative integer")
        steps = max(0, (earliest_ns - self.anchor_ns + self.period_ns - 1) // self.period_ns)
        return self.anchor_ns + steps * self.period_ns


class PulseTracker:
    """Require six regular onsets before predicting; lose lock on disagreement.

    Pass a single selected onset stream, such as kicks, in presentation order.
    Do not combine kick and snare streams blindly: their subdivision is unknown.
    Bounds are algorithmic policy, not physical robot limits. A missing beat
    deliberately loses lock rather than guessing an octave or silently coasting.
    """

    def __init__(self, *, min_period_ns=250_000_000, max_period_ns=1_500_000_000,
                 tolerance_ns=25_000_000, stale_after_ns=2_000_000_000):
        values = (min_period_ns, max_period_ns, tolerance_ns, stale_after_ns)
        if any(type(v) is not int or v <= 0 for v in values):
            raise ValueError("timing policies must be positive integers")
        if not tolerance_ns < min_period_ns <= max_period_ns <= stale_after_ns:
            raise ValueError("inconsistent timing policies")
        self.min_period_ns = min_period_ns
        self.max_period_ns = max_period_ns
        self.tolerance_ns = tolerance_ns
        self.stale_after_ns = stale_after_ns
        self._times = deque(maxlen=6)

    def reset(self):
        self._times.clear()

    def observe(self, presentation_ns: int) -> bool:
        if type(presentation_ns) is not int or presentation_ns < 0:
            raise ValueError("presentation_ns must be a nonnegative integer")
        if self._times and presentation_ns <= self._times[-1]:
            return False
        if self._times and presentation_ns - self._times[-1] > self.stale_after_ns:
            self.reset()
        self._times.append(presentation_ns)
        return True

    def estimate(self, now_ns: int) -> Pulse | None:
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("now_ns must be a nonnegative integer")
        if len(self._times) < 6:
            return None
        # Events arrive early; freshness is measured against presentation time.
        if now_ns - self._times[-1] > self.stale_after_ns:
            return None
        times = list(self._times)
        intervals = [b - a for a, b in zip(times, times[1:])]
        period = int(median(intervals))
        if not self.min_period_ns <= period <= self.max_period_ns:
            return None
        # Check cumulative phase as well as neighbouring gaps; slow drift must
        # not masquerade as a stable grid that will miss future visible beats.
        residuals = [t - i * period for i, t in enumerate(times)]
        phase = int(median(residuals))
        if any(abs(value - phase) > self.tolerance_ns for value in residuals):
            return None
        return Pulse(period, phase + (len(times) - 1) * period)
