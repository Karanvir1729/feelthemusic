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

Other reported-not-verified items: intensity, sharpness and bass samples are u8 values reported as
"x255" scaled, so we divide by 255 (0 -> 0.0, 255 -> 1.0). Kind names come from ftm_client.KIND_NAMES;
an unknown kind is dropped and counted, NEVER mapped to another kind. Unknown flag bits are ignored.
``target`` is kept raw (0xFF reported as "all"). ``lead_us`` is not used: it is not in the contract.
"""
from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class BassSample:
    due_ns: int  # local monotonic ns
    level: float  # 0..1 (raw/255)


class Normalizer:
    def __init__(self, estimator: ClockEstimator, trim_ns: int = 0) -> None:
        self.estimator = estimator
        self.trim_ns = trim_ns
        self.unknown_kind = 0
        self.not_synced = 0
        self.bad_time = 0
        self.normalized = 0  # events
        self.bass_normalized = 0  # non-empty bass envelopes

    def normalize_event(self, pkt: EventPacket, now_ns: int | None = None) -> Event | None:
        """Event, or None (counted) if not synced, kind unknown, or the time cannot be converted."""
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
            flags=frozenset(flag_names(pkt.flags)), due_ns=due,
        )

    def normalize_bass(self, pkt: BassEnvelope, now_ns: int | None = None) -> list[BassSample]:
        """Sample i is due at start_ts + i*step_ms (conductor time) converted once; [] on any failure.

        n == 0 gives [] with no counter. step_ms == 0 is accepted only when n <= 1 (no semantics
        are invented for several samples at one instant); otherwise it counts as bad_time.
        """
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
