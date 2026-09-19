"""Conductor-side music analysis.

Pure numpy. Takes mono audio, returns timestamped events, a bass envelope and
beat/tempo info. Everything is expressed in seconds on the track's own clock;
the conductor maps that to the shared presentation clock.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass, field

import numpy as np

N_FFT = 2048
HOP = 512

BASS_BAND = (50.0, 100.0)
KICK_BAND = (40.0, 120.0)
SNARE_BAND = (180.0, 450.0)
SNARE_NOISE_BAND = (2000.0, 8000.0)


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
    with wave.open(path, "rb") as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
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


def _stft_mag(x: np.ndarray) -> np.ndarray:
    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
    n_frames = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = x[idx] * np.hanning(N_FFT).astype(np.float32)
    return np.abs(np.fft.rfft(frames, axis=1))


def _band(mag: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    sel = (freqs >= lo) & (freqs <= min(hi, sr / 2))
    if not sel.any():
        return np.zeros(mag.shape[0])
    return mag[:, sel].sum(axis=1)


def _flux(band_mag: np.ndarray) -> np.ndarray:
    """Half-wave rectified log-magnitude difference: onset strength per frame."""
    logm = np.log1p(100.0 * band_mag)
    d = np.diff(logm, prepend=logm[0])
    return np.maximum(d, 0.0)


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


def _events(env: np.ndarray, kind: str, hop_s: float, min_gap_s: float) -> list[Event]:
    peaks = _pick_peaks(env, hop_s, min_gap_s)
    if not peaks:
        return []
    top = max(env[p] for p in peaks)
    # An STFT frame is centred N_FFT/2 samples after its start.
    return [Event(t=p * hop_s + (N_FFT / 2) * (hop_s / HOP), kind=kind,
                  strength=float(env[p] / top)) for p in peaks]


def _tempo(onset_env: np.ndarray, hop_s: float,
           bpm_range: tuple[float, float] = (60.0, 180.0)) -> tuple[float | None, list[float]]:
    if onset_env.max() <= 0 or len(onset_env) < 16:
        return None, []
    x = onset_env - onset_env.mean()
    n = len(x)
    ac = np.correlate(x, x, mode="full")[n - 1:]
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
    first = best_off * hop_s + (N_FFT / 2) * (hop_s / HOP)
    beats = list(np.arange(first, duration, period))
    return float(bpm), [float(b) for b in beats]


def analyze(samples: np.ndarray, sample_rate: int) -> Analysis:
    """Analyse mono float samples. Deterministic; no randomness, no I/O."""
    x = np.asarray(samples, dtype=np.float32)
    mag = _stft_mag(x)
    hop_s = HOP / sample_rate

    bass = _band(mag, sample_rate, *BASS_BAND)
    kick = _band(mag, sample_rate, *KICK_BAND)
    snare_body = _band(mag, sample_rate, *SNARE_BAND)
    snare_noise = _band(mag, sample_rate, *SNARE_NOISE_BAND)

    kick_flux = _flux(kick)
    snare_flux = _flux(snare_body) + _flux(snare_noise)

    kicks = _events(kick_flux, "kick", hop_s, min_gap_s=0.12)
    snares = _events(snare_flux, "snare", hop_s, min_gap_s=0.12)
    # A kick also excites the snare band; drop snare hits that coincide with a
    # stronger low-band hit and carry no extra broadband noise.
    kick_times = [k.t for k in kicks]
    snares = [s for s in snares
              if not any(abs(s.t - kt) < 0.03 for kt in kick_times)
              or snare_noise[int(s.t / hop_s)] > 0.5 * snare_noise.max()]

    full = _flux(mag.sum(axis=1))
    onsets = _events(full, "onset", hop_s, min_gap_s=0.05)

    # Light smoothing (~40 ms) so the envelope is safe to drive a light or motor,
    # then normalise so the loudest frame is exactly 1.
    k = max(1, int(round(0.04 / hop_s)))
    env = np.convolve(bass, np.ones(k) / k, mode="same")
    peak = env.max()
    env = env / peak if peak > 0 else env

    bpm, beats = _tempo(kick_flux + snare_flux + full, hop_s)
    events = sorted(kicks + snares + onsets, key=lambda e: e.t)
    return Analysis(sample_rate=sample_rate, hop_seconds=hop_s, events=events,
                    bass_envelope=env.astype(np.float32), bpm=bpm, beat_times=beats)


def analyze_file(path: str) -> Analysis:
    samples, rate = load_wav(path)
    return analyze(samples, rate)
