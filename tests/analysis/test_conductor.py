import wave

import numpy as np
import pytest

from analysis import analyze, analyze_file, load_wav

SR = 44100
BPM = 120.0
BEAT = 60.0 / BPM
BARS = 8


def synth(bpm=BPM, bars=BARS, seed=0):
    """Kick on every beat, snare on beats 2 and 4, over a quiet noise floor."""
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    n = int(SR * beat * 4 * bars)
    x = 0.005 * rng.standard_normal(n).astype(np.float32)
    kick_t, snare_t = [], []
    for b in range(4 * bars):
        t0 = b * beat
        i = int(t0 * SR)
        tt = np.arange(int(0.18 * SR)) / SR
        kick = np.sin(2 * np.pi * (60 + 40 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 18)
        x[i:i + len(kick)] += 0.8 * kick[: n - i]
        kick_t.append(t0)
        if b % 2 == 1:
            tt = np.arange(int(0.12 * SR)) / SR
            sn = (rng.standard_normal(len(tt)) * np.exp(-tt * 35)
                  + 0.5 * np.sin(2 * np.pi * 220 * tt) * np.exp(-tt * 30))
            x[i:i + len(sn)] += 0.5 * sn[: n - i]
            snare_t.append(t0)
    return x, kick_t, snare_t


def times(a, kind):
    return [e.t for e in a.events if e.kind == kind]


def match(found, expected, tol):
    return sum(any(abs(f - e) <= tol for f in found) for e in expected)


def test_tempo_and_phase():
    x, kick_t, _ = synth()
    a = analyze(x, SR)
    assert a.bpm == pytest.approx(BPM, abs=2.0)
    # Beat grid lands on kicks.
    assert match(a.beat_times, kick_t[:16], 0.04) >= 15
    phase = a.beat_phase(kick_t[4])
    assert min(phase, 1 - phase) < 0.1


def test_kicks_found_with_low_false_positives():
    x, kick_t, _ = synth()
    a = analyze(x, SR)
    found = times(a, "kick")
    assert match(found, kick_t, 0.04) >= len(kick_t) - 1
    assert len(found) <= len(kick_t) + 2


def test_snares_found_on_backbeat():
    x, _, snare_t = synth()
    a = analyze(x, SR)
    found = times(a, "snare")
    assert match(found, snare_t, 0.05) >= len(snare_t) - 1
    assert len(found) <= len(snare_t) + 3


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


def test_wav_roundtrip(tmp_path):
    x, kick_t, _ = synth()
    path = tmp_path / "beat.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
        w.writeframes(np.column_stack([pcm, pcm]).tobytes())
    data, rate = load_wav(str(path))
    assert rate == SR and len(data) == len(x)
    a = analyze_file(str(path))
    assert a.bpm == pytest.approx(BPM, abs=2.0)
    assert match(times(a, "kick"), kick_t, 0.04) >= len(kick_t) - 1
