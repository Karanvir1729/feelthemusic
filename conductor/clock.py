"""One clock: integer nanoseconds on the conductor's monotonic clock.

No wall clock anywhere in this module. The conductor stamps events with its own
monotonic time; a client measures its offset to that clock with request/response
probes and converts with ``to_local_time`` / ``to_conductor_time``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional

NS_PER_MS = 1_000_000
ROOM_BUDGET_NS = 300 * NS_PER_MS      # L, from docs/architecture.md
MIN_LEAD_NS = 250 * NS_PER_MS         # conductor sends at least this far ahead of pts

ClockSource = Callable[[], int]


def monotonic_ns() -> int:
    """Default clock source: the OS monotonic clock, integer nanoseconds."""
    return time.monotonic_ns()


class FakeClock:
    """Injectable clock for tests. Only moves when told to."""

    def __init__(self, start_ns: int = 0) -> None:
        self._now = int(start_ns)

    def __call__(self) -> int:
        return self._now

    def now_ns(self) -> int:
        return self._now

    def advance(self, delta_ns: int) -> int:
        if delta_ns < 0:
            raise ValueError("a monotonic clock cannot go backwards")
        self._now += int(delta_ns)
        return self._now


@dataclass(frozen=True)
class ProbeSample:
    """One probe round trip. t0/t3 on the client clock, t1/t2 on the conductor clock."""

    t0: int
    t1: int
    t2: int
    t3: int

    @property
    def delay_ns(self) -> int:
        """Round-trip network delay: total elapsed minus conductor processing time."""
        return (self.t3 - self.t0) - (self.t2 - self.t1)

    @property
    def offset2_ns(self) -> int:
        """Twice the NTP offset (conductor minus client), kept doubled to stay exact."""
        return (self.t1 - self.t0) + (self.t2 - self.t3)

    @property
    def offset_ns(self) -> int:
        return self.offset2_ns // 2

    def is_sane(self) -> bool:
        return self.t3 >= self.t0 and self.t2 >= self.t1 and self.delay_ns >= 0


@dataclass(frozen=True)
class OffsetEstimate:
    offset_ns: int          # conductor_time = local_time + offset_ns
    uncertainty_ns: int     # true offset is within +/- this (under the stated assumptions)
    best_delay_ns: int      # smallest round-trip delay among kept samples
    samples: int


class OffsetEstimator:
    """Client-side offset estimator.

    Each sample gives ``theta = ((t1-t0)+(t2-t3))/2`` and ``delay = (t3-t0)-(t2-t1)``.
    Forward and return one-way delays d1, d2 are both >= 0 and sum to ``delay``, so the
    error of ``theta`` is ``(d1-d2)/2`` and is bounded by ``delay/2``. That makes every
    sample a *hard interval* ``theta +/- delay/2`` containing the true offset (assuming
    the two clocks do not drift apart meanwhile). We keep the ``keep`` lowest-delay
    samples (queueing only ever adds delay) and intersect their intervals: the estimate
    is the midpoint, the uncertainty is the half-width. The intersection is never wider
    than the best single sample's ``delay/2``.

    Drift: with ``now_local_ns`` given, each interval is widened by ``age * drift_ppm``
    and samples older than ``max_age_ns`` are ignored. ``drift_ppm`` is a guess (see
    PROTOCOL.md). If the widened intervals still do not intersect (a step in the
    offset), fall back to the freshest of the kept samples.
    """

    def __init__(self, keep: int = 8, drift_ppm: int = 100, max_age_ns: int = 60_000_000_000) -> None:
        if keep < 1:
            raise ValueError("keep must be >= 1")
        self.keep = keep
        self.drift_ppm = drift_ppm
        self.max_age_ns = max_age_ns
        self._samples: List[ProbeSample] = []

    def add(self, t0: int, t1: int, t2: int, t3: int) -> bool:
        """Add a probe round trip. Returns False (and ignores it) if it is not physical."""
        s = ProbeSample(t0, t1, t2, t3)
        if not s.is_sane():
            return False
        self._samples.append(s)
        if len(self._samples) > self.keep:
            # drop the worst (highest delay); on ties drop the oldest
            worst = max(range(len(self._samples)), key=lambda i: (self._samples[i].delay_ns, -i))
            del self._samples[worst]
        return True

    @property
    def samples(self) -> List[ProbeSample]:
        return list(self._samples)

    def estimate(self, now_local_ns: Optional[int] = None) -> Optional[OffsetEstimate]:
        pool = self._samples
        if now_local_ns is not None:
            pool = [s for s in pool if now_local_ns - s.t3 <= self.max_age_ns]
        if not pool:
            return None
        lo2 = hi2 = None
        for s in pool:
            widen2 = 0
            if now_local_ns is not None:
                widen2 = 2 * (max(0, now_local_ns - s.t3) * self.drift_ppm // 1_000_000)
            a, b = s.offset2_ns - s.delay_ns - widen2, s.offset2_ns + s.delay_ns + widen2
            lo2 = a if lo2 is None else max(lo2, a)
            hi2 = b if hi2 is None else min(hi2, b)
        if lo2 > hi2:  # inconsistent (offset stepped): trust only the freshest sample
            s = max(pool, key=lambda x: x.t3)
            lo2, hi2 = s.offset2_ns - s.delay_ns, s.offset2_ns + s.delay_ns
        offset = (lo2 + hi2) // 4
        unc = -(-(hi2 - lo2) // 4) + 1  # ceil of half-width in ns, +1 ns for integer rounding
        return OffsetEstimate(offset, unc, min(s.delay_ns for s in pool), len(pool))


def to_conductor_time(local_ns: int, offset_ns: int) -> int:
    return local_ns + offset_ns


def to_local_time(conductor_ns: int, offset_ns: int) -> int:
    return conductor_ns - offset_ns
