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


def _snare_dark(shell_hz=180.0, cutoff_hz=1200.0):
    """A dark, shell-heavy snare: low-passed noise plus strong shell modes. Little
    energy above a few kHz, so a kick/snare rule that only compares against
    2-8 kHz cannot tell it from a kick."""
    def make(rng):
        n = int(0.22 * SR)
        tt = np.arange(n) / SR
        a = 1 - np.exp(-2 * np.pi * cutoff_hz / SR)
        kernel = a * (1 - a) ** np.arange(int(8 / a))            # one-pole low-pass
        noise = np.convolve(rng.standard_normal(n), kernel)[:n]
        noise = noise / (np.std(noise) + 1e-9) * np.exp(-tt * 22) * 0.6
        shell = (np.sin(2 * np.pi * shell_hz * tt)
                 + 0.5 * np.sin(2 * np.pi * shell_hz * 1.83 * tt)) * np.exp(-tt * 16)
        return noise + shell
    return make


def _hats(rng, n_samples, bpm, bars, level=0.25):
    """Closed hi-hats on every eighth note: short, high-passed noise bursts. Adds no snare body."""
    x = np.zeros(n_samples, np.float32)
    beat = 60.0 / bpm
    for k in range(bars * 8):
        i = int((LEAD_IN + k * beat / 2) * SR)
        n = int(0.05 * SR)
        tt = np.arange(n) / SR
        burst = np.diff(rng.standard_normal(n + 1)) * np.exp(-tt * 90)
        seg = level * burst[: max(0, n_samples - i)]
        x[i:i + len(seg)] += seg
    return x


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


@pytest.mark.parametrize("shell_hz,cutoff_hz", [(180, 600), (180, 1200), (180, 2500), (240, 1200)])
def test_dark_snares_do_not_fire_kicks(shell_hz, cutoff_hz):
    """Regression: with the kick rule comparing only against 2-8 kHz, a dark
    shell-heavy snare fired a kick on 9-114 of 144 hits (52/144 at 180 Hz shell,
    1.2 kHz low-pass)."""
    fired = hits = 0
    for bpm in (90.0, 120.0, 140.0):
        for seed in (0, 1, 2):
            x, kick_t, snare_t = synth(bpm=bpm, seed=seed, kick=_kick_sweep,
                                       snare=_snare_dark(shell_hz, cutoff_hz),
                                       kick_beats=lambda b: b % 2 == 0,
                                       snare_beats=lambda b: b % 2 == 1)
            a = analyze(x, SR)
            fired += match(times(a, "kick"), snare_t, 0.05)
            hits += len(snare_t)
            assert match(times(a, "kick"), kick_t, 0.05) == len(kick_t)
            assert match(times(a, "snare"), snare_t, 0.05) == len(snare_t)
    assert fired <= 0.02 * hits, f"kick fired on {fired}/{hits} dark snare hits"


@pytest.mark.parametrize("shell_hz,cutoff_hz", [(180, 600), (180, 1200), (240, 1200)])
def test_kick_landing_with_a_dark_snare_is_still_found(shell_hz, cutoff_hz):
    """The stricter kick rule must not lose the kick when a shell-heavy snare
    hits at the same instant. The margin here is thin (see KICK_DOMINANCE), so
    allow a couple of misses out of 96 but not a systematic loss."""
    found = total = 0
    for bpm in (90.0, 120.0, 140.0):
        for seed in (0, 1, 2):
            x, kick_t, snare_t = synth(bpm=bpm, seed=seed, kick=_kick_sweep,
                                       snare=_snare_dark(shell_hz, cutoff_hz),
                                       kick_beats=lambda b: True,
                                       snare_beats=lambda b: b % 2 == 1)
            a = analyze(x, SR)
            found += match(times(a, "kick"), kick_t, 0.05)
            total += len(kick_t)
            assert match(times(a, "snare"), snare_t, 0.05) == len(snare_t)
    assert found >= total - 3, f"lost {total - found} of {total} kicks under a dark snare"


@pytest.mark.xfail(reason="Known limitation: a snare whose fundamental sits inside the kick band "
                          "(~120 Hz) is ambiguous by frequency alone and fires a kick. Measured "
                          "94-133 of 144 hits. Needs the real track, or a duration/decay feature.",
                   strict=False)
def test_snare_with_fundamental_in_kick_band_does_not_fire_kicks():
    fired = hits = 0
    for seed in (0, 1, 2):
        x, kick_t, snare_t = synth(seed=seed, kick=_kick_sweep, snare=_snare_dark(120, 1200),
                                   kick_beats=lambda b: b % 2 == 0,
                                   snare_beats=lambda b: b % 2 == 1)
        fired += match(times(analyze(x, SR), "kick"), snare_t, 0.05)
        hits += len(snare_t)
    assert fired <= 0.02 * hits


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


def test_snare_with_a_body_landing_with_a_kick_is_still_reported():
    """A shell-heavy snare that lands together with a kick must be reported, and so must the kick.
    (Behaviour guard: this passes on the original code too, so the review's 'real snares are wrongly
    suppressed' symptom was not reproduced on synthetic audio. The off-by-two frame read itself is
    covered by test_hits_in_last_frames_do_not_crash.)"""
    x, kick_t, snare_t = synth(snare=_snare_dark(180, 1200), kick_beats=lambda b: True,
                               snare_beats=lambda b: b % 4 in (1, 3))
    a = analyze(x, SR)
    assert match(times(a, "snare"), snare_t, 0.05) == len(snare_t)
    assert match(times(a, "kick"), kick_t, 0.05) == len(kick_t)


@pytest.mark.xfail(reason="Known trade-off: a very short white-noise snare landing with a kick raises the "
                          "body band no more than a hi-hat on the same beat does (body/low ratio median "
                          "0.063 vs kick+hat <= 0.088), so only the kick is reported. Hats on kick beats "
                          "are far more common than snares on them.", strict=False)
def test_tight_white_noise_snare_landing_with_a_kick_is_reported():
    x, kick_t, snare_t = synth(snare=_snare_tight, kick_beats=lambda b: True,
                               snare_beats=lambda b: b % 4 in (1, 3))
    assert match(times(analyze(x, SR), "snare"), snare_t, 0.05) == len(snare_t)


@pytest.mark.parametrize("bpm", [90.0, 120.0, 140.0])
def test_hi_hats_do_not_fire_snares(bpm):
    """Regression: the snare channel summed body and noise-band flux, so a hi-hat (all top end, no
    body) fired it on its own: 131 false snares for 96 real ones with eighth-note hats."""
    fired = real = 0
    for seed in (0, 1, 2):
        x, kick_t, snare_t = alternating(bpm=bpm, seed=seed)
        x = x + _hats(np.random.default_rng(seed + 100), len(x), bpm, BARS)
        a = analyze(x, SR)
        snares = times(a, "snare")
        assert match(times(a, "kick"), kick_t, 0.05) == len(kick_t)
        assert match(snares, snare_t, 0.05) == len(snare_t)
        fired += len(snares) - match(snares, snare_t, 0.05)
        real += len(snare_t)
    assert fired <= 0.02 * real, f"{fired} false snares for {real} real ones"


@pytest.mark.parametrize("level", [0.1, 0.25, 0.5])
def test_hat_on_a_kick_beat_is_not_a_snare(level):
    """The hardest hat case: kick and hat landing together, on every beat. Only kicks may be reported."""
    x, kick_t, _ = synth(kick=_kick_sweep, snare_beats=lambda b: False)
    x = x + _hats(np.random.default_rng(7), len(x), BPM, BARS, level=level)
    a = analyze(x, SR)
    assert match(times(a, "kick"), kick_t, 0.05) == len(kick_t)
    assert match(times(a, "snare"), kick_t, 0.05) == 0


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


@pytest.mark.parametrize("bpm", [145.0, 150.0, 160.0, 174.0, 180.0])
def test_fast_tempos_are_not_reported_at_half_time(bpm):
    """Regression: 145-180 BPM came back at half (174 -> 87). The slower level repeats more
    strongly in a kick/snare pattern, so it used to win the comb."""
    x, _, _ = alternating(bpm=bpm, bars=16)
    a = analyze(x, SR)
    assert a.bpm == pytest.approx(bpm, rel=0.02)
    assert a.tempo_confidence > 0.5


@pytest.mark.parametrize("bpm", [70.0, 80.0, 90.0, 100.0])
def test_slow_tempos_are_not_doubled(bpm):
    """The octave fold must not turn a slow tempo into twice its speed, with or without hats."""
    for with_hats in (False, True):
        x, _, _ = alternating(bpm=bpm, bars=16)
        if with_hats:
            x = x + _hats(np.random.default_rng(5), len(x), bpm, 16)
        assert analyze(x, SR).bpm == pytest.approx(bpm, rel=0.02), f"hats={with_hats}"


@pytest.mark.xfail(reason="Known ambiguity: at or below ~80 BPM with strong eighth-note hats, the "
                          "half-lag peak is as strong as the beat, so the fold doubles the tempo "
                          "(65 BPM -> 130). Subdivision vs beat cannot be told apart here.", strict=False)
def test_65_bpm_with_hats_is_not_doubled():
    x, _, _ = alternating(bpm=65.0, bars=16)
    x = x + _hats(np.random.default_rng(5), len(x), 65.0, 16)
    assert analyze(x, SR).bpm == pytest.approx(65.0, rel=0.02)


@pytest.mark.parametrize("bpm", [100.0, 120.0])
def test_beat_grid_does_not_drift_over_a_song(bpm):
    """Regression: one global period extrapolated across the track put the last beat 400-650 ms
    off after three minutes, past the 300 ms room budget, while the +/-2 BPM tests still passed."""
    bars = int(150 / (60.0 / bpm * 4))
    x, _, _ = alternating(bpm=bpm, bars=bars)
    a = analyze(x, SR)
    beat = 60.0 / bpm
    truth = [LEAD_IN + k * beat for k in range(int((len(x) / SR - LEAD_IN) / beat))]
    n = min(len(a.beat_times), len(truth))
    assert n > 0.9 * len(truth)
    err = np.abs(np.array(a.beat_times[:n]) - np.array(truth[:n]))
    assert err.max() < 0.03, f"worst beat error {err.max() * 1000:.0f} ms"


def _pulseless(kind):
    rng = np.random.default_rng(0)
    n = 30 * SR
    tt = np.arange(n) / SR
    if kind == "room tone":
        return 0.01 * rng.standard_normal(n).astype(np.float32)
    if kind == "chord":
        return (sum(np.sin(2 * np.pi * f * tt) for f in (220.0, 277.18, 329.63, 440.0)) * 0.1).astype(np.float32)
    if kind == "two-partial chord":
        return (sum(np.sin(2 * np.pi * f * tt) for f in (196.0, 392.0)) * 0.1).astype(np.float32)
    swell = np.convolve(np.abs(rng.standard_normal(n)), np.ones(400) / 400, mode="same")
    return (0.2 * rng.standard_normal(n) * (0.4 + swell)).astype(np.float32)


@pytest.mark.parametrize("kind", ["room tone", "chord", "two-partial chord", "applause"])
def test_pulseless_material_reports_no_tempo(kind):
    """Regression: every pulseless clip got a confident BPM and a full beat grid (room tone ->
    167 BPM with 33 beats). A lamp driven by that dances to nothing."""
    a = analyze(_pulseless(kind), SR)
    assert a.bpm is None
    assert a.beat_times == []
    assert a.beat_phase(1.0) is None
    assert a.tempo_confidence < 0.3


def test_pulsed_material_has_high_tempo_confidence_even_when_quiet_or_noisy():
    x, _, _ = alternating(bpm=120.0, bars=16)
    rng = np.random.default_rng(3)
    for variant in (x * 0.02, x + 0.05 * rng.standard_normal(len(x)).astype(np.float32)):
        a = analyze(variant.astype(np.float32), SR)
        assert a.bpm == pytest.approx(120.0, rel=0.02)
        assert a.tempo_confidence > 0.5


def test_beat_phase_interpolates_between_the_bracketing_beats():
    """The grid follows the music, so a beat interval is not constant: phase is measured between the
    two beats that bracket the time, not against one global period."""
    a = analyze(np.zeros(SR, dtype=np.float32), SR)
    a.bpm = 60.0
    a.beat_times = [0.0, 1.0, 2.5, 4.5]                  # intervals 1.0, 1.5, 2.0 s
    assert a.beat_phase(0.5) == pytest.approx(0.5)
    assert a.beat_phase(1.75) == pytest.approx(0.5)       # halfway through the 1.5 s interval
    assert a.beat_phase(3.5) == pytest.approx(0.5)        # halfway through the 2.0 s interval
    assert a.beat_phase(2.5) == pytest.approx(0.0)


def test_eighth_note_kicks_are_all_found():
    """Coverage gap found by mutation: with one kick per beat at 120 BPM a min-gap of 0.5 s (which
    would silently drop every kick faster than every 500 ms) passed the suite."""
    x, kick_t, _ = synth(bpm=120.0, kick=_kick_sweep, snare_beats=lambda b: False)
    beat = 60.0 / 120.0
    extra = [t + beat / 2 for t in kick_t[:-1]]
    rng = np.random.default_rng(11)
    for t in extra:
        i = int(t * SR)
        k = 0.8 * _kick_sweep(rng)
        x[i:i + len(k)] += k[: len(x) - i]
    a = analyze(x, SR)
    every = sorted(kick_t + extra)
    assert match(times(a, "kick"), every, 0.05) >= len(every) - 1


def test_bass_envelope_ignores_energy_outside_the_bass_band():
    """Coverage gap found by mutation: widening BASS_BAND from 50-100 Hz to 50-400 Hz passed the
    suite. A 300 Hz burst must barely register; a 70 Hz burst of the same level must."""
    rng = np.random.default_rng(2)
    n = 6 * SR
    x = _noise(rng, n)
    tt = np.arange(int(0.25 * SR)) / SR
    env = np.exp(-tt * 12)
    for at, hz in ((1.0, 70.0), (3.0, 300.0), (5.0, 70.0)):
        i = int(at * SR)
        x[i:i + len(tt)] += 0.6 * np.sin(2 * np.pi * hz * tt) * env
    a = analyze(x, SR)
    at_bass = max(a.bass_envelope[int(t / a.hop_seconds)] for t in (1.05, 5.05))
    at_mid = max(a.bass_envelope[int((3.0 + d) / a.hop_seconds)] for d in np.arange(0.0, 0.2, 0.01))
    assert at_mid < 0.15 * at_bass, f"300 Hz burst reached {at_mid / at_bass:.0%} of the 70 Hz burst"


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


def _wav_bytes(samples_le: bytes, *, tag=1, channels=1, rate=SR, bits=16, extensible=False, pad_list=False):
    """Hand-built RIFF/WAVE bytes, so the loader is tested independently of the stdlib `wave` module."""
    block = channels * bits // 8
    if extensible:
        fmt = struct.pack("<HHIIHHHHIH14s", 0xFFFE, channels, rate, rate * block, block, bits, 22, bits, 3,
                          tag, bytes.fromhex("000000001000800000aa00389b71"))
    else:
        fmt = struct.pack("<HHIIHH", tag, channels, rate, rate * block, block, bits)
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    if pad_list:                                        # an odd-sized extra chunk before the data
        chunks += b"LIST" + struct.pack("<I", 3) + b"abc" + bytes(1)
    chunks += b"data" + struct.pack("<I", len(samples_le)) + samples_le
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks


def _pcm24(values):
    out = bytearray()
    for v in values:
        out += int(v).to_bytes(3, "little", signed=True)
    return bytes(out)


@pytest.mark.parametrize("extensible", [False, True])
@pytest.mark.parametrize("pad_list", [False, True])
def test_24_bit_pcm_loads_the_same_plain_or_extensible(tmp_path, extensible, pad_list):
    """WAVE_FORMAT_EXTENSIBLE is what most tools emit for 24-bit. The stdlib `wave` module rejected
    it before Python 3.12 and reads it from 3.12, so this must not depend on the interpreter."""
    values = [0, 4194304, -4194304, 8388607, -8388608, 1234567]
    path = tmp_path / "x.wav"
    path.write_bytes(_wav_bytes(_pcm24(values), bits=24, extensible=extensible, pad_list=pad_list))
    data, rate = load_wav(str(path))
    assert rate == SR
    np.testing.assert_allclose(data, np.array(values) / 8388608.0, atol=1e-6)


def test_extensible_stereo_16_bit_is_downmixed(tmp_path):
    left, right = np.array([1000, -2000, 3000], "<i2"), np.array([3000, -4000, 5000], "<i2")
    pcm = np.column_stack([left, right]).tobytes()
    path = tmp_path / "s.wav"
    path.write_bytes(_wav_bytes(pcm, channels=2, bits=16, extensible=True))
    data, _ = load_wav(str(path))
    np.testing.assert_allclose(data, (left.astype(float) + right) / 2 / 32768.0, atol=1e-6)


@pytest.mark.parametrize("extensible", [False, True])
def test_float_wav_gives_a_friendly_error(tmp_path, extensible):
    path = tmp_path / "f.wav"
    path.write_bytes(_wav_bytes(np.zeros(8, "<f4").tobytes(), tag=3, bits=32, extensible=extensible))
    with pytest.raises(ValueError, match="not integer PCM"):
        load_wav(str(path))


def test_truncated_data_chunk_loads_what_is_there(tmp_path):
    """A streamed or cut-off file can claim more data than it holds."""
    good = _wav_bytes(np.arange(100, dtype="<i2").tobytes())
    path = tmp_path / "t.wav"
    path.write_bytes(good[:-51])                        # cuts mid-sample; the odd byte must be dropped
    data, _ = load_wav(str(path))
    assert len(data) == 74


def test_not_a_wav_file(tmp_path):
    path = tmp_path / "n.wav"
    path.write_bytes(b"this is not audio at all")
    with pytest.raises(ValueError, match="not a RIFF/WAVE"):
        load_wav(str(path))
