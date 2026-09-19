"""Clock offset estimator for the native FTM Conductor stamps (task #19). Pure standard library.

No sockets, no wall clock: every time is an integer nanosecond passed in by the caller, including
"now" (``now_ns``). Nothing here has been run against the real conductor; the stamp layout it
consumes (SyncResp t1/t2, EventPacket masterTs) is source-reported by a teammate, see ftm_client.py.

Where the algorithm comes from
------------------------------
This is a port of ``OffsetEstimator`` in conductor/clock.py (tested minimum-delay estimator on
origin/claude/6-conductor-core, PROTOCOL.md there), NOT a new design. Each probe gives
``theta = ((t1-t0)+(t2-t3))/2`` and ``delay = (t3-t0)-(t2-t1)``. The forward and return one-way delays
are both >= 0 and sum to ``delay``, so theta is off by at most ``delay/2``: every sample is a hard
interval ``theta +/- delay/2`` around the true offset (assuming no drift in between). We keep the
``keep`` lowest-delay samples (queueing only adds delay), intersect their intervals, report the midpoint
and use the half-width (+1 ns for integer rounding) as ``bound_ns``. If drift-widened intervals do not
intersect (offset stepped) the freshest kept sample alone is used.

Adaptations, all ours: names (``add_sample``, ``bound_ns``, ``NotSynced``), strict type/range validation
that RAISES instead of returning False, ``min_samples`` warm-up (default 3; the original answers after
one sample, and one sample is still a valid hard bound), ``now_ns`` passed explicitly, ``ready``,
``to_local_due_ns``.

Convention: offset = conductor_time - local_time, so local_time = conductor_time - offset and
conductor_time = local_time + offset. The estimator is only as good as its assumptions: a CONSTANT path
asymmetry is not removed, it is inside the bound; drift beyond ``drift_ppm`` (a guess, 100 ppm as in
the original) is not covered; the hard bound holds only if the samples are physical. The bound also
does not include the conductor's own clock accuracy.

Aging: ``estimate(now_ns)`` ignores samples with ``now_ns - t3 > max_age_ns`` and widens each remaining
interval by ``age * drift_ppm``. With ``now_ns=None`` there is no expiry and no widening (so the bound
then ignores drift). Samples are never deleted by aging, so a later query with an earlier ``now_ns`` sees
them again.

Contract (docs/integration.md section 3): downstream code gets a LOCAL MONOTONIC ``due_ns`` and never
adds an offset itself. ``to_local_due_ns`` subtracts the offset and a trim and adds NO room budget L; the
native conductor is reported (source-reported, unverified) to have added L into masterTs already.
"""
from __future__ import annotations

from dataclasses import dataclass

INT63_MAX = 2**63 - 1


class NotSynced(RuntimeError):
    """No usable offset estimate exists (too few samples, or all expired)."""


@dataclass(frozen=True)
class ClockEstimate:
    offset_ns: int  # conductor_time = local_time + offset_ns
    bound_ns: int  # true offset is within +/- this, under the assumptions in the module docstring
    best_delay_ns: int  # smallest round-trip delay among the samples used
    samples: int  # samples used


@dataclass(frozen=True)
class _Sample:
    t0: int
    t1: int
    t2: int
    t3: int

    @property
    def delay_ns(self) -> int:
        return (self.t3 - self.t0) - (self.t2 - self.t1)

    @property
    def offset2_ns(self) -> int:  # twice the offset, kept doubled to stay exact
        return (self.t1 - self.t0) + (self.t2 - self.t3)


def _check_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int, got {type(value).__name__}")
    return value


def _check_time(name: str, value: object) -> int:
    v = _check_int(name, value)
    if not 0 <= v <= INT63_MAX:
        raise ValueError(f"{name} out of range 0..2**63-1: {v}")
    return v


class ClockEstimator:
    def __init__(self, keep: int = 8, drift_ppm: int = 100, max_age_ns: int = 60_000_000_000,
                 min_samples: int = 3) -> None:
        _check_int("keep", keep)
        _check_int("drift_ppm", drift_ppm)
        _check_int("max_age_ns", max_age_ns)
        _check_int("min_samples", min_samples)
        if keep < 1 or drift_ppm < 0 or max_age_ns < 0 or not 1 <= min_samples <= keep:
            raise ValueError("need keep >= 1, drift_ppm >= 0, max_age_ns >= 0, 1 <= min_samples <= keep")
        self.keep = keep
        self.drift_ppm = drift_ppm
        self.max_age_ns = max_age_ns
        self.min_samples = min_samples
        self._samples: list[_Sample] = []

    def reset(self) -> None:
        """Forget every sample (a new session or a changed conductor): the estimator is unsynced again."""
        self._samples = []

    def add_sample(self, t0: int, t1: int, t2: int, t3: int) -> None:
        """t0/t3 local ns (client send/receive), t1/t2 conductor ns (receive/send).

        Raises ValueError on a non-physical sample and leaves the state untouched.
        """
        t0, t1, t2, t3 = (_check_time(n, v) for n, v in (("t0", t0), ("t1", t1), ("t2", t2), ("t3", t3)))
        if t3 < t0:
            raise ValueError("t3 < t0: client received before it sent")
        if t2 < t1:
            raise ValueError("t2 < t1: conductor sent before it received")
        s = _Sample(t0, t1, t2, t3)
        if s.delay_ns < 0:
            raise ValueError("negative round-trip delay after subtracting the conductor hold time")
        kept = self._samples + [s]
        if len(kept) > self.keep:
            # drop the worst (highest delay); on ties drop the oldest
            worst = max(range(len(kept)), key=lambda i: (kept[i].delay_ns, -i))
            del kept[worst]
        self._samples = kept

    @property
    def ready(self) -> bool:
        """True when enough samples are stored (ignores age; use estimate(now_ns) for that)."""
        return len(self._samples) >= self.min_samples

    def estimate(self, now_ns: int | None = None) -> ClockEstimate | None:
        if now_ns is not None:
            _check_time("now_ns", now_ns)
        pool = self._samples
        if now_ns is not None:
            pool = [s for s in pool if now_ns - s.t3 <= self.max_age_ns]
        if len(pool) < self.min_samples:
            return None
        lo2 = hi2 = None
        for s in pool:
            widen2 = 0
            if now_ns is not None:
                widen2 = 2 * (max(0, now_ns - s.t3) * self.drift_ppm // 1_000_000)
            a, b = s.offset2_ns - s.delay_ns - widen2, s.offset2_ns + s.delay_ns + widen2
            lo2 = a if lo2 is None else max(lo2, a)
            hi2 = b if hi2 is None else min(hi2, b)
        if lo2 > hi2:  # inconsistent (offset stepped): trust only the freshest sample
            f = max(pool, key=lambda x: x.t3)
            lo2, hi2 = f.offset2_ns - f.delay_ns, f.offset2_ns + f.delay_ns
        offset = (lo2 + hi2) // 4
        bound = -(-(hi2 - lo2) // 4) + 1
        return ClockEstimate(offset, bound, min(s.delay_ns for s in pool), len(pool))

    def to_local_due_ns(self, master_ts_ns: int, trim_ns: int = 0, now_ns: int | None = None) -> int:
        """master_ts - offset - trim. Adds NO room budget L. Raises NotSynced or ValueError."""
        _check_int("master_ts_ns", master_ts_ns)
        _check_int("trim_ns", trim_ns)
        est = self.estimate(now_ns)
        if est is None:
            raise NotSynced("no clock offset estimate yet")
        due = master_ts_ns - est.offset_ns - trim_ns
        if not 0 <= due <= INT63_MAX:
            raise ValueError(f"converted time out of range 0..2**63-1: {due}")
        return due
