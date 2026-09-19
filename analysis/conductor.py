"""Conductor-side music analysis.

Pure numpy. Takes mono audio, returns timestamped events, a bass envelope and
beat/tempo info. Everything is expressed in seconds on the track's own clock;
the conductor maps that to the shared presentation clock.

The analysis is whole-track: strengths, thresholds and the envelope are
normalised by the track's global maximum, so it cannot run on a live stream.
That suits a demo that plays a chosen file.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass, field

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

N_FFT = 2048
HOP = 512

BASS_BAND = (50.0, 100.0)
KICK_BAND = (40.0, 120.0)
SNARE_BAND = (180.0, 450.0)
SNARE_NOISE_BAND = (2000.0, 8000.0)

# Frames per STFT chunk. Keeps peak memory flat in track length.
CHUNK_FRAMES = 512

# Log-compressed flux peaks while an onset is still entering the tail of the
# analysis window, so raw event times land early. Measured on synthetic
# kick/snare tracks (tests/analysis); added back so times land on the onset.
ONSET_LATENCY_S = 0.013

# A low-band hit is a kick only if the per-bin rise in 40-120 Hz magnitude beats
# the per-bin rise in 2-8 kHz magnitude by this factor. White noise (a snare)
# raises every bin equally; a kick raises only the low bins. Linear magnitude,
# not log: log flux saturates on a loud kick and cannot tell a kick that lands
# with a snare from a snare alone.
# Measured on synthetic tracks: snare alone <= 3.1, kick + snare >= 16,
# kick alone >= 5000. NOT measured on real music, where snares are not white
# noise and the ratio for a snare alone will be higher; tune on the demo track.
KICK_DOMINANCE = 8.0
# A snare that lands with a kick must still carry this share of the loudest
# broadband noise flux, or it is the kick's own click.
SNARE_MIN_NOISE_WHEN_WITH_KICK = 0.3
# Frames either side of a peak searched when comparing two band measures.
PEAK_SLACK = 2


@dataclass(frozen=True)
class Event:
    t: float
    kind: str  # "kick" | "snare" | "onset"
    strength: float  # 0..1, relative to the loudest event of that kind


@dataclass
class Analysis:
    sample_rate: int
    hop_seconds: float
    events: list[Event]
    bass_envelope: np.ndarray  # one value per frame, 0..1
    bpm: float | None
    beat_times: list[float] = field(default_factory=list)

    def beat_phase(self, t: float) -> float | None:
        """Position in the current beat, 0 <= phase < 1. None without a tempo."""
        if not self.bpm or not self.beat_times:
            return None
        period = 60.0 / self.bpm
        return ((t - self.beat_times[0]) / period) % 1.0


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Read a PCM WAV file as mono float32 in [-1, 1]."""
    try:
        with wave.open(path, "rb") as w:
            rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(w.getnframes())
    except wave.Error as e:
        raise ValueError(
            f"cannot read {path} as PCM WAV ({e}). WAVE_FORMAT_EXTENSIBLE and "
            "float WAV files are not supported; re-export as plain 16- or "
            "24-bit PCM."
        ) from e
    if width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        v = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8)
             | (b[:, 2].astype(np.int32) << 16))
        v = np.where(v & 0x800000, v - 0x1000000, v)
        data = v.astype(np.float32) / 8388608.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width: {width}")
    if channels > 1:
        data = data[: len(data) // channels * channels].reshape(-1, channels).mean(axis=1)
    return data, rate


def _bin_range(sr: int, lo: float, hi: float) -> tuple[int, int]:
    """Half-open FFT bin range for a band, always at least one bin wide."""
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    sel = np.nonzero((freqs >= lo) & (freqs <= min(hi, sr / 2)))[0]
    if sel.size == 0:
        return 0, 0
    return int(sel[0]), int(sel[-1]) + 1


def _band_sums(x: np.ndarray, sr: int, bands: list[tuple[float, float]]) -> np.ndarray:
    """STFT magnitude summed over each band: shape (n_frames, len(bands)).

    Chunked and windowed as a view, so memory stays flat in track length; only
    the per-band sums are kept.
    """
    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
    frames = sliding_window_view(x, N_FFT)[::HOP]
    ranges = [_bin_range(sr, lo, hi) for lo, hi in bands]
    window = np.hanning(N_FFT).astype(np.float32)
    out = np.zeros((len(frames), len(bands)), dtype=np.float32)
    for a in range(0, len(frames), CHUNK_FRAMES):
        mag = np.abs(np.fft.rfft(frames[a:a + CHUNK_FRAMES] * window, axis=1))
        for j, (lo, hi) in enumerate(ranges):
            if hi > lo:
                out[a:a + len(mag), j] = mag[:, lo:hi].sum(axis=1)
    return out


def _flux(band_mag: np.ndarray) -> np.ndarray:
    """Half-wave rectified log-magnitude difference: onset strength per frame."""
    logm = np.log1p(100.0 * band_mag)
    d = np.diff(logm, prepend=logm[0])
    return np.maximum(d, 0.0)


def _rise(band_mag: np.ndarray) -> np.ndarray:
    """Frame-to-frame increase in linear band magnitude (never negative)."""
    return np.maximum(np.diff(band_mag, prepend=band_mag[0]), 0.0)


def _pick_peaks(env: np.ndarray, hop_s: float, min_gap_s: float,
                rel_threshold: float = 0.3) -> list[int]:
    if env.max() <= 0:
        return []
    win = max(3, int(round(0.5 / hop_s)))
    kernel = np.ones(win) / win
    local = np.convolve(env, kernel, mode="same")
    thresh = local * 1.5 + rel_threshold * env.max()
    gap = max(1, int(round(min_gap_s / hop_s)))
    peaks: list[int] = []
    for i in range(1, len(env) - 1):
        if env[i] < thresh[i] or env[i] < env[i - 1] or env[i] < env[i + 1]:
            continue
        if peaks and i - peaks[-1] < gap:
            if env[i] > env[peaks[-1]]:
                peaks[-1] = i
            continue
        peaks.append(i)
    return peaks


def _peak_max(env: np.ndarray, frame: int) -> float:
    """Largest value within PEAK_SLACK frames of `frame`; safe at the edges."""
    lo, hi = max(0, frame - PEAK_SLACK), min(len(env), frame + PEAK_SLACK + 1)
    return float(env[lo:hi].max())


def _frame_time(frame: int, hop_s: float, sr: int) -> float:
    """Track time of a frame's onset: window centre plus the measured latency."""
    return frame * hop_s + (N_FFT / 2) / sr + ONSET_LATENCY_S


def _events(frames: list[int], env: np.ndarray, kind: str,
            hop_s: float, sr: int) -> list[Event]:
    if not frames:
        return []
    top = max(env[f] for f in frames)
    return [Event(t=_frame_time(f, hop_s, sr), kind=kind, strength=float(env[f] / top))
            for f in frames]


def _autocorr(x: np.ndarray) -> np.ndarray:
    """Non-negative-lag autocorrelation via FFT (O(n log n), unlike np.correlate)."""
    n = len(x)
    size = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, size)
    return np.fft.irfft(f * np.conj(f), size)[:n]


def _tempo(onset_env: np.ndarray, hop_s: float, sr: int,
           bpm_range: tuple[float, float] = (60.0, 180.0)) -> tuple[float | None, list[float]]:
    if onset_env.max() <= 0 or len(onset_env) < 16:
        return None, []
    x = onset_env - onset_env.mean()
    n = len(x)
    ac = _autocorr(x)
    lo = int(round(60.0 / bpm_range[1] / hop_s))
    hi = min(n - 1, int(round(60.0 / bpm_range[0] / hop_s)))
    if hi <= lo:
        return None, []
    lags = np.arange(lo, hi + 1)
    # Prefer moderate tempi so a half/double-time peak does not win by accident.
    prior = np.exp(-0.5 * (np.log2((60.0 / (lags * hop_s)) / 110.0) / 0.9) ** 2)
    lag = int(lags[np.argmax(ac[lo:hi + 1] * prior)])
    # Refine the lag with a parabola through the neighbours.
    if 0 < lag < len(ac) - 1:
        a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
        denom = a - 2 * b + c
        if denom != 0:
            lag = lag + 0.5 * (a - c) / denom
    period = lag * hop_s
    bpm = 60.0 / period
    # Phase: pick the offset whose comb of beats collects the most onset energy.
    steps = max(1, int(round(period / hop_s)))
    best_off, best_score = 0, -1.0
    for off in range(steps):
        pos = np.arange(off, n, period / hop_s)
        score = onset_env[np.clip(pos.round().astype(int), 0, n - 1)].sum()
        if score > best_score:
            best_off, best_score = off, score
    duration = n * hop_s
    first = _frame_time(best_off, hop_s, sr)
    beats = list(np.arange(first, duration, period))
    return float(bpm), [float(b) for b in beats]


def analyze(samples: np.ndarray, sample_rate: int) -> Analysis:
    """Analyse mono float samples. Deterministic; no randomness, no I/O."""
    x = np.asarray(samples, dtype=np.float32)
    hop_s = HOP / sample_rate
    bands = [BASS_BAND, KICK_BAND, SNARE_BAND, SNARE_NOISE_BAND, (0.0, sample_rate / 2)]
    sums = _band_sums(x, sample_rate, bands)
    bass, kick, snare_body, snare_noise, full_band = sums.T

    # Per-bin means make the low and high bands comparable: white noise has the
    # same energy per bin everywhere, a kick only in the low bins.
    widths = [max(1, _bin_range(sample_rate, *b)[1] - _bin_range(sample_rate, *b)[0])
              for b in bands]
    kick_flux = _flux(kick / widths[1])
    noise_flux = _flux(snare_noise / widths[3])
    body_flux = _flux(snare_body / widths[2])
    snare_flux = body_flux + noise_flux

    low_rise = _rise(kick / widths[1])
    high_rise = _rise(snare_noise / widths[3])
    kick_frames = [f for f in _pick_peaks(kick_flux, hop_s, min_gap_s=0.12)
                   if _peak_max(low_rise, f) >= KICK_DOMINANCE * _peak_max(high_rise, f)]
    noise_top = float(noise_flux.max()) if len(noise_flux) else 0.0
    snare_frames = []
    for f in _pick_peaks(snare_flux, hop_s, min_gap_s=0.12):
        with_kick = any(abs(f - k) <= 2 for k in kick_frames)
        if with_kick and _peak_max(noise_flux, f) < SNARE_MIN_NOISE_WHEN_WITH_KICK * noise_top:
            continue
        snare_frames.append(f)

    full_flux = _flux(full_band)
    onset_frames = _pick_peaks(full_flux, hop_s, min_gap_s=0.05)

    # Light smoothing (~40 ms) so the envelope is safe to drive a light or motor,
    # then normalise so the loudest frame is exactly 1. This de-jitters; it is
    # not a flash-safety limiter, which belongs in the lamp renderer.
    k = max(1, int(round(0.04 / hop_s)))
    env = np.convolve(bass, np.ones(k) / k, mode="same")
    peak = env.max()
    env = env / peak if peak > 0 else env

    bpm, beats = _tempo(kick_flux + snare_flux + full_flux, hop_s, sample_rate)
    events = sorted(
        _events(kick_frames, kick_flux, "kick", hop_s, sample_rate)
        + _events(snare_frames, snare_flux, "snare", hop_s, sample_rate)
        + _events(onset_frames, full_flux, "onset", hop_s, sample_rate),
        key=lambda e: e.t)
    return Analysis(sample_rate=sample_rate, hop_seconds=hop_s, events=events,
                    bass_envelope=env.astype(np.float32), bpm=bpm, beat_times=beats)


def analyze_file(path: str) -> Analysis:
    samples, rate = load_wav(path)
    return analyze(samples, rate)
