"""Normalise decoded FTM packets into events with a LOCAL MONOTONIC due time (task #19).

Sits on top of ftm_client.py (structure-only parser) and ftm_clock.py (offset estimator). Pure: no
sockets, no wall clock. Nothing here was checked against the real conductor.

Time contract (docs/integration.md section 3): ``due_ns`` is local monotonic and is what the consumer
fires at. This layer subtracts the estimated offset and ``trim_ns`` and adds NO room budget L.
UNCONFIRMED: whether masterTs (and BassEnvelope.startTs) already include L. A teammate reports that the
native conductor adds L into masterTs (source-reported, not verified); nothing here relies on it beyond
declining to add L. If that report is wrong, every due_ns is 300 ms early and only the consumer/adapter
config can fix it, not this code.

No offset means no output: a consumer must hold or drop such events, it never fires with no offset.

Freshness cannot be switched off by forgetting an argument: every normalize_* call needs the current
monotonic time, either as ``now_ns=`` or from ``Normalizer(clock=callable)``. With neither it raises
TypeError. (ClockEstimator.estimate(now_ns=None) remains an explicit, documented diagnostic.)

BassEnvelope timestamps are OFF by default. The reported semantics of BassEnvelope.startTs are not
verified (masterTs is reported to be pre-budgeted; nothing says startTs is), and a silently active
assumption would show up as wrong visible timing. ``Normalizer(bass_time_policy="presentation")`` opts in to
treating startTs like masterTs (a native presentation timestamp, no L added). Until someone verifies
that against the real conductor, normalize_bass returns [] and counts ``bass_policy_unset``.

Sessions: ``seq`` is the raw native u32 sequence. It wraps and it is NOT an ordering key; a consumer
keeps its own session-local sequence. ``Normalizer.new_epoch()`` (call it when a new session/Assign
arrives or the conductor changes) resets the clock estimator and bumps ``epoch``; every Event and
BassSample carries the epoch it was made in, so a consumer can drop queued work from an old epoch.
There is no native stop kind (kinds are 0..5), so safety stops are a separate consumer API.

Other reported-not-verified items: intensity, sharpness and bass samples are u8 values reported as
"x255" scaled, so we divide by 255 (0 -> 0.0, 255 -> 1.0). Kind names come from ftm_client.KIND_NAMES;
an unknown kind is dropped and counted, NEVER mapped to another kind. Unknown flag bits are ignored.
``target`` is kept raw (0xFF reported as "all"). ``lead_us`` is not used: it is not in the contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ftm_client import BassEnvelope, EventPacket, flag_names, kind_name
from ftm_clock import ClockEstimator, NotSynced

NS_PER_MS = 1_000_000


@dataclass(frozen=True)
class Event:
    seq: int
    kind: str
    intensity: float  # 0..1 (raw/255)
    sharpness: float  # 0..1 (raw/255)
    duration_ms: int
    freq_hz: int
    target: int  # raw
    flags: frozenset[str]
    due_ns: int  # local monotonic ns
    epoch: int = 0  # Normalizer.epoch when this was made; drop work from an older epoch


@dataclass(frozen=True)
class BassSample:
    due_ns: int  # local monotonic ns
    level: float  # 0..1 (raw/255)
    epoch: int = 0


class Normalizer:
    BASS_POLICIES = (None, "presentation")

    def __init__(self, estimator: ClockEstimator, trim_ns: int = 0, bass_time_policy: str | None = None,
                 clock: Callable[[], int] | None = None) -> None:
        if bass_time_policy not in self.BASS_POLICIES:
            raise ValueError("unknown bass_time_policy (only None or 'presentation' exist)")
        self.estimator = estimator
        self.trim_ns = trim_ns
        self.bass_time_policy = bass_time_policy
        self.clock = clock
        self.epoch = 0
        self.bass_policy_unset = 0
        self.unknown_kind = 0
        self.not_synced = 0
        self.bad_time = 0
        self.normalized = 0  # events
        self.bass_normalized = 0  # non-empty bass envelopes

    def _now(self, now_ns: int | None) -> int:
        if now_ns is not None:
            return now_ns
        if self.clock is None:
            raise TypeError("normalize needs now_ns=... or Normalizer(clock=...): freshness is never optional")
        now = self.clock()
        if type(now) is not int:
            raise TypeError(f"the injected clock must return an int of nanoseconds, got {type(now).__name__}")
        return now

    def new_epoch(self) -> int:
        """Start a new session: forget the clock samples and bump the epoch. Returns the new epoch."""
        self.estimator.reset()
        self.epoch += 1
        return self.epoch

    def normalize_event(self, pkt: EventPacket, now_ns: int | None = None) -> Event | None:
        """Event, or None (counted) if not synced, kind unknown, or the time cannot be converted."""
        now_ns = self._now(now_ns)
        name = kind_name(pkt.kind)
        if name is None:
            self.unknown_kind += 1
            return None
        try:
            due = self.estimator.to_local_due_ns(pkt.master_ts, self.trim_ns, now_ns)
        except NotSynced:
            self.not_synced += 1
            return None
        except ValueError:
            self.bad_time += 1
            return None
        self.normalized += 1
        return Event(
            seq=pkt.seq, kind=name, intensity=pkt.intensity / 255, sharpness=pkt.sharpness / 255,
            duration_ms=pkt.duration_ms, freq_hz=pkt.freq_hz, target=pkt.target,
            flags=frozenset(flag_names(pkt.flags)), due_ns=due, epoch=self.epoch,
        )

    def normalize_bass(self, pkt: BassEnvelope, now_ns: int | None = None) -> list[BassSample]:
        """Sample i is due at start_ts + i*step_ms (conductor time) converted once; [] on any failure.

        n == 0 gives [] with no counter. step_ms == 0 is accepted only when n <= 1 (no semantics
        are invented for several samples at one instant); otherwise it counts as bad_time.
        """
        now_ns = self._now(now_ns)
        if self.bass_time_policy is None:
            self.bass_policy_unset += 1
            return []
        n = len(pkt.samples)
        if n == 0:
            return []
        if pkt.step_ms == 0 and n > 1:
            self.bad_time += 1
            return []
        try:
            out = [
                BassSample(
                    self.estimator.to_local_due_ns(pkt.start_ts + i * pkt.step_ms * NS_PER_MS, self.trim_ns, now_ns),
                    raw / 255,
                    self.epoch,
                )
                for i, raw in enumerate(pkt.samples)
            ]
        except NotSynced:
            self.not_synced += 1
            return []
        except ValueError:
            self.bad_time += 1
            return []
        self.bass_normalized += 1
        return out
