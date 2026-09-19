import struct
import tracemalloc
import wave

import numpy as np
import pytest

import analysis.conductor as conductor
from analysis import analyze, analyze_file, load_wav

SR = 44100
BPM = 120.0
BEAT = 60.0 / BPM
BARS = 8
LEAD_IN = 0.25  # seconds of noise floor before the first hit; see README


def _noise(rng, n):
    return 0.004 * rng.standard_normal(n).astype(np.float32)


def _kick_sine(rng):
    tt = np.arange(int(0.18 * SR)) / SR
    return np.sin(2 * np.pi * (60 + 40 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 18)


def _kick_sweep(rng):
    tt = np.arange(int(0.2 * SR)) / SR
    phase = 2 * np.pi * np.cumsum(60 + 40 * np.exp(-tt * 25)) / SR
    return np.sin(phase) * np.exp(-tt * 16)


def _snare_tonal(rng):
    tt = np.arange(int(0.12 * SR)) / SR
    return (rng.standard_normal(len(tt)) * np.exp(-tt * 35)
            + 0.5 * np.sin(2 * np.pi * 220 * tt) * np.exp(-tt * 30))


def _snare_noise(rng):
    tt = np.arange(int(0.15 * SR)) / SR
    return (rng.standard_normal(len(tt)) * np.exp(-tt * 30)
            + 0.5 * np.sin(2 * np.pi * 220 * tt) * np.exp(-tt * 25))


def synth(bpm=BPM, bars=BARS, seed=0, kick=_kick_sine, snare=_snare_tonal,
          kick_beats=None, snare_beats=None, tail=0.0):
    """Kick and snare hits over a noise floor. Returns (audio, kick_t, snare_t).

    By default a kick lands on every beat and a snare on beats 2 and 4 (the
    kick and snare are then coincident on those beats). `kick_beats` /
    `snare_beats` are predicates on the beat index.
    """
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    kick_beats = kick_beats or (lambda b: True)
    snare_beats = snare_beats or (lambda b: b % 2 == 1)
    n = int(SR * (LEAD_IN + beat * 4 * bars + tail))
    x = _noise(rng, n)
    kick_t, snare_t = [], []
    for b in range(4 * bars):
        t0 = LEAD_IN + b * beat
        i = int(t0 * SR)
        if kick_beats(b):
            k = _kick_sine(rng) if kick is _kick_sine else kick(rng)
            x[i:i + len(k)] += 0.8 * k[: n - i]
            kick_t.append(t0)
        if snare_beats(b):
            s = snare(rng)
            x[i:i + len(s)] += 0.5 * s[: n - i]
            snare_t.append(t0)
    return x, kick_t, snare_t


def alternating(bpm=BPM, bars=BARS, seed=1):
    """Kick on even beats, snare on odd beats: no coincident hits."""
    return synth(bpm=bpm, bars=bars, seed=seed, kick=_kick_sweep, snare=_snare_noise,
                 kick_beats=lambda b: b % 2 == 0, snare_beats=lambda b: b % 2 == 1)


def times(a, kind):
    return [e.t for e in a.events if e.kind == kind]


def match(found, expected, tol):
    return sum(any(abs(f - e) <= tol for f in found) for e in expected)


def errors(found, expected, tol=0.05):
    out = []
    for e in expected:
        near = [f for f in found if abs(f - e) <= tol]
        if near:
            out.append(min(near, key=lambda f: abs(f - e)) - e)
    return out


def test_tempo_and_phase():
    x, kick_t, _ = synth()
    a = analyze(x, SR)
    assert a.bpm == pytest.approx(BPM, abs=2.0)
    assert match(a.beat_times, kick_t[:16], 0.04) >= 15
    phase = a.beat_phase(kick_t[4])
    assert min(phase, 1 - phase) < 0.1


def test_kicks_found_with_low_false_positives():
    x, kick_t, _ = synth()
    a = analyze(x, SR)
    found = times(a, "kick")
    assert match(found, kick_t, 0.04) == len(kick_t)
    assert len(found) <= len(kick_t) + 2


def test_snares_found_on_backbeat():
    x, _, snare_t = synth()
    a = analyze(x, SR)
    found = times(a, "snare")
    assert match(found, snare_t, 0.05) == len(snare_t)
    assert len(found) <= len(snare_t) + 3


@pytest.mark.parametrize("bpm", [90.0, 120.0, 140.0])
def test_snares_do_not_fire_kicks(bpm):
    """Regression: broadband snare noise used to trigger a kick on every hit."""
    x, kick_t, snare_t = alternating(bpm=bpm)
    a = analyze(x, SR)
    kicks, snares = times(a, "kick"), times(a, "snare")
    assert match(kicks, kick_t, 0.05) == len(kick_t)
    assert match(kicks, snare_t, 0.05) == 0, "kick events fired on snare hits"
    assert len(kicks) <= len(kick_t) + 1
    assert match(snares, snare_t, 0.05) == len(snare_t)
    assert match(snares, kick_t, 0.05) == 0, "snare events fired on kick hits"
    assert len(snares) <= len(snare_t) + 1


@pytest.mark.parametrize("make", [synth, alternating])
def test_event_times_are_unbiased(make):
    """Regression: times used to land ~13 ms before the onset."""
    x, kick_t, snare_t = make()
    a = analyze(x, SR)
    assert abs(np.mean(errors(times(a, "kick"), kick_t))) < 0.005
    assert abs(np.mean(errors(times(a, "snare"), snare_t))) < 0.005


def _snare_tight(rng):
    """A short, fast-decaying snare: nearly silent 23 ms after the hit."""
    tt = np.arange(int(0.08 * SR)) / SR
    return rng.standard_normal(len(tt)) * np.exp(-tt * 90)


def test_snare_coinciding_with_kick_is_still_reported():
    """A snare that lands together with a kick must still be reported, even a
    tight one, and so must the kick. Behaviour guard: this passes on the
    original code too, so the review's 'real snares are wrongly suppressed'
    symptom was not reproduced on synthetic audio. The off-by-two frame read
    itself is covered by test_hits_in_last_frames_do_not_crash."""
    x, kick_t, snare_t = synth(snare=_snare_tight, kick_beats=lambda b: True,
                               snare_beats=lambda b: b % 4 in (1, 3))
    a = analyze(x, SR)
    both = [t for t in snare_t if t in kick_t]
    assert both, "test setup: need coincident hits"
    assert match(times(a, "snare"), snare_t, 0.05) == len(snare_t)
    assert match(times(a, "kick"), kick_t, 0.05) == len(kick_t)


def test_hits_in_last_frames_do_not_crash():
    """A coincident kick and snare right at the end of the file must not index
    past the last frame."""
    hop_s = conductor.HOP / SR
    for extra in (0.02, 0.03, 0.05, 0.08):
        x, _, _ = synth(bars=2, tail=extra)
        i = len(x) - int(extra * SR)
        k, s = _kick_sine(None), _snare_tonal(np.random.default_rng(3))
        x[i:i + len(k)] += 0.8 * k[: len(x) - i]
        x[i:i + len(s)] += 0.5 * s[: len(x) - i]
        analyze(x, SR)  # must not raise
    assert hop_s > 0


def test_bass_envelope_tracks_kicks():
    x, kick_t, _ = synth()
    a = analyze(x, SR)
    env = a.bass_envelope
    assert 0.0 <= env.min() and env.max() == pytest.approx(1.0, abs=0.05)
    hop = a.hop_seconds
    at_kick = np.mean([env[int((t + 0.03) / hop)] for t in kick_t[:16]])
    between = np.mean([env[int((t + BEAT * 0.75) / hop)] for t in kick_t[:16]])
    assert at_kick > 3 * between


def test_silence_is_quiet():
    a = analyze(np.zeros(SR * 3, dtype=np.float32), SR)
    assert a.events == []
    assert a.bpm is None
    assert a.bass_envelope.max() == 0.0


def test_deterministic():
    x, _, _ = synth()
    a, b = analyze(x, SR), analyze(x, SR)
    assert a.events == b.events and a.bpm == b.bpm


def test_chunking_does_not_change_the_result(monkeypatch):
    x, _, _ = alternating(bars=8)
    whole = analyze(x, SR)
    monkeypatch.setattr(conductor, "CHUNK_FRAMES", 37)
    chunked = analyze(x, SR)
    assert chunked.events == whole.events
    np.testing.assert_allclose(chunked.bass_envelope, whole.bass_envelope, rtol=1e-5)
    assert chunked.bpm == pytest.approx(whole.bpm)


def test_memory_stays_bounded():
    """Regression: peak memory was ~1.3 GB for a 4 minute track (about 5.6 MB
    per second). Chunking keeps it far below that; 60 s must stay well under
    100 MB."""
    x, _, _ = alternating(bars=32, bpm=128.0)
    x = np.tile(x, 3)[: 60 * SR]
    tracemalloc.start()
    try:
        analyze(x, SR)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 100e6, f"peak {peak / 1e6:.0f} MB for 60 s of audio"


def _write_wav(path, x, channels=1):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(SR)
        pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
        if channels == 2:
            pcm = np.column_stack([pcm, pcm])
        w.writeframes(pcm.tobytes())


def test_wav_roundtrip(tmp_path):
    x, kick_t, _ = synth()
    path = tmp_path / "beat.wav"
    _write_wav(path, x, channels=2)
    data, rate = load_wav(str(path))
    assert rate == SR and len(data) == len(x)
    a = analyze_file(str(path))
    assert a.bpm == pytest.approx(BPM, abs=2.0)
    assert match(times(a, "kick"), kick_t, 0.04) == len(kick_t)


def test_extensible_wav_gives_a_friendly_error(tmp_path):
    """Python's wave module cannot open WAVE_FORMAT_EXTENSIBLE (tag 0xFFFE)."""
    path = tmp_path / "ext.wav"
    fmt = struct.pack("<HHIIHHHHIH14s", 0xFFFE, 2, SR, SR * 6, 6, 24, 22, 24, 3,
                      1, b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"[:14])
    data = b"\x00" * 600
    body = (b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
            + b"data" + struct.pack("<I", len(data)) + data)
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
    with pytest.raises(ValueError, match="WAVE_FORMAT_EXTENSIBLE"):
        load_wav(str(path))
