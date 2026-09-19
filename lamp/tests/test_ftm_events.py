import random

import pytest

import ftm_client
from ftm_client import BassEnvelope, EventPacket
from ftm_clock import ClockEstimator
from ftm_events import BassSample, Event, Normalizer

MS = 1_000_000
OFFSET = 5_000_000_000  # conductor = local + OFFSET; symmetric probes make it exact


def synced(n=3):
    est = ClockEstimator()
    for i in range(n):
        t0 = 10**12 + i * 100 * MS
        est.add_sample(t0, t0 + MS + OFFSET, t0 + MS + OFFSET, t0 + 2 * MS)
    assert est.estimate().offset_ns == OFFSET
    return est


def pkt(**kw):
    base = dict(seq=1, kind=1, flags=0, intensity=0, sharpness=0, duration_ms=0, freq_hz=0, target=0,
                master_ts=OFFSET + 10**10, lead_us=0)
    base.update(kw)
    return EventPacket(**base)


@pytest.mark.parametrize("kind,name", [(0, "click"), (1, "kick"), (2, "snare"), (3, "bass"), (4, "build"),
                                       (5, "drop")])
def test_every_reported_kind_maps(kind, name):
    n = Normalizer(synced())
    ev = n.normalize_event(pkt(kind=kind))
    assert isinstance(ev, Event) and ev.kind == name
    assert n.normalized == 1 and n.unknown_kind == 0


@pytest.mark.parametrize("kind", [6, 7, 200, 255])
def test_unknown_kinds_are_dropped_and_counted_never_defaulted(kind):
    n = Normalizer(synced())
    assert n.normalize_event(pkt(kind=kind)) is None
    assert n.unknown_kind == 1 and n.normalized == 0 and n.not_synced == 0 and n.bad_time == 0


def test_unknown_kind_when_unsynced_counts_as_unknown_kind_only():
    n = Normalizer(ClockEstimator())
    assert n.normalize_event(pkt(kind=200)) is None
    assert n.unknown_kind == 1 and n.not_synced == 0


def test_due_time_conversion_no_l_and_trim():
    n = Normalizer(synced(), trim_ns=3 * MS)
    ev = n.normalize_event(pkt(master_ts=OFFSET + 10**10))
    assert ev.due_ns == 10**10 - 3 * MS  # offset and trim subtracted, no +300 ms
    assert Normalizer(synced()).normalize_event(pkt(master_ts=OFFSET + 10**10)).due_ns == 10**10


def test_fields_pass_through():
    ev = Normalizer(synced()).normalize_event(
        pkt(seq=77, kind=2, flags=3, duration_ms=123, freq_hz=456, target=0xFF, intensity=51, sharpness=102))
    assert (ev.seq, ev.duration_ms, ev.freq_hz, ev.target) == (77, 123, 456, 0xFF)
    assert ev.intensity == pytest.approx(0.2) and ev.sharpness == pytest.approx(0.4)


def test_intensity_scaling_endpoints_and_range():
    n = Normalizer(synced())
    assert n.normalize_event(pkt(intensity=255, sharpness=255)).intensity == 1.0
    assert n.normalize_event(pkt(intensity=255, sharpness=255)).sharpness == 1.0
    z = n.normalize_event(pkt(intensity=0, sharpness=0))
    assert z.intensity == 0.0 and z.sharpness == 0.0
    for raw in range(256):
        ev = n.normalize_event(pkt(intensity=raw, sharpness=raw))
        assert ev.intensity == pytest.approx(raw / 255.0, abs=1e-12) and 0.0 <= ev.intensity <= 1.0
    assert n.normalize_event(pkt(intensity=128)).intensity == pytest.approx(0.50196, abs=1e-5)  # not 0.5


@pytest.mark.parametrize("flags,expected", [
    (0, set()), (1, {"audio"}), (2, {"haptic"}), (4, {"measure"}), (7, {"audio", "haptic", "measure"}),
    (8, set()), (0x80, set()), (0xFF, {"audio", "haptic", "measure"}), (0b1010_1001, {"audio"}),
])
def test_flags_decode_and_unknown_bits_are_ignored(flags, expected):
    ev = Normalizer(synced()).normalize_event(pkt(flags=flags))
    assert ev is not None and ev.flags == frozenset(expected)


def test_not_synced_counts_and_returns_none():
    n = Normalizer(ClockEstimator())
    assert n.normalize_event(pkt()) is None
    assert n.not_synced == 1 and n.normalized == 0
    assert n.normalize_bass(BassEnvelope(1, OFFSET, 20, (1, 2))) == []
    assert n.not_synced == 2


def test_expired_estimate_counts_as_not_synced():
    est = ClockEstimator(max_age_ns=10 * 10**9)
    for i in range(3):
        t0 = 10**12 + i
        est.add_sample(t0, t0 + OFFSET, t0 + OFFSET, t0 + 2)
    n = Normalizer(est)
    assert n.normalize_event(pkt(), now_ns=10**12 + 5 * 10**9) is not None
    assert n.normalize_event(pkt(), now_ns=10**12 + 30 * 10**9) is None
    assert n.not_synced == 1


def test_unconvertible_time_counts_bad_time():
    n = Normalizer(synced())
    assert n.normalize_event(pkt(master_ts=0)) is None  # would be negative locally
    assert n.normalize_event(pkt(master_ts=2**64 - 1)) is None  # beyond 2**63-1
    assert n.bad_time == 2 and n.normalized == 0


def test_bass_n0_is_empty_without_counters():
    n = Normalizer(synced())
    assert n.normalize_bass(BassEnvelope(1, OFFSET + 10**10, 20, ())) == []
    assert (n.bad_time, n.not_synced, n.bass_normalized) == (0, 0, 0)


def test_bass_n1_and_step_zero():
    n = Normalizer(synced())
    out = n.normalize_bass(BassEnvelope(1, OFFSET + 10**10, 0, (255,)))
    assert out == [BassSample(10**10, 1.0)]
    out = n.normalize_bass(BassEnvelope(1, OFFSET + 10**10, 20, (0,)))
    assert out == [BassSample(10**10, 0.0)]


def test_bass_step_zero_with_several_samples_is_bad_time():
    n = Normalizer(synced())
    assert n.normalize_bass(BassEnvelope(1, OFFSET + 10**10, 0, (1, 2))) == []
    assert n.bad_time == 1 and n.bass_normalized == 0


def test_bass_255_samples_due_times_and_levels():
    n = Normalizer(synced(), trim_ns=MS)
    step = 20
    raw = tuple((i * 7) % 256 for i in range(255))
    start = OFFSET + 10**10
    out = n.normalize_bass(BassEnvelope(9, start, step, raw))
    assert len(out) == 255
    for i, s in enumerate(out):
        assert s.due_ns == 10**10 + i * step * MS - MS  # written out by hand, no +L
        assert s.level == pytest.approx(raw[i] / 255.0, abs=1e-12)
    assert out[1].due_ns - out[0].due_ns == 20 * MS
    assert n.bass_normalized == 1


def test_bass_with_out_of_range_sample_fails_whole_envelope():
    n = Normalizer(synced())
    assert n.normalize_bass(BassEnvelope(1, 2**63 - 1 + OFFSET - MS, 250, (1, 2, 3))) == []
    assert n.bad_time == 1


def test_bass_max_step_255_ms():
    n = Normalizer(synced())
    out = n.normalize_bass(BassEnvelope(1, OFFSET + 10**10, 255, (1, 2, 3)))
    assert [s.due_ns for s in out] == [10**10, 10**10 + 255 * MS, 10**10 + 510 * MS]


def test_round_trip_from_wire_bytes():
    wire = pkt(kind=5, flags=6, intensity=255, master_ts=OFFSET + 10**10 + 12345).encode()
    ev = Normalizer(synced()).normalize_event(ftm_client.decode(wire))
    assert ev.kind == "drop" and ev.flags == {"haptic", "measure"} and ev.due_ns == 10**10 + 12345


def test_fuzz_never_raises_only_event_none_or_list():
    rng = random.Random(5150)
    n = Normalizer(synced(), trim_ns=rng.randint(-10 * MS, 10 * MS))
    for _ in range(5000):
        if rng.random() < 0.5:
            r = n.normalize_event(EventPacket(
                rng.getrandbits(32), rng.getrandbits(8), rng.getrandbits(8), rng.getrandbits(8),
                rng.getrandbits(8), rng.getrandbits(16), rng.getrandbits(16), rng.getrandbits(8),
                rng.choice([rng.getrandbits(64), rng.getrandbits(34), 0, 2**64 - 1, 2**63]),
                rng.getrandbits(32)))
            assert r is None or isinstance(r, Event)
            if r is not None:
                assert 0.0 <= r.intensity <= 1.0 and 0.0 <= r.sharpness <= 1.0 and 0 <= r.due_ns < 2**63
        else:
            m = rng.randint(0, 255)
            r = n.normalize_bass(BassEnvelope(
                rng.getrandbits(32), rng.choice([rng.getrandbits(64), rng.getrandbits(34), OFFSET + 10**10]),
                rng.choice([0, 1, 20, 255, rng.getrandbits(8)]), tuple(rng.getrandbits(8) for _ in range(m))))
            assert isinstance(r, list) and all(isinstance(s, BassSample) for s in r)
    assert n.normalized > 0 and n.unknown_kind > 0 and n.bad_time > 0  # the fuzz exercised every path
