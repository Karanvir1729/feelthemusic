"""Conductor-side music analysis.

Pure numpy. Takes mono audio, returns timestamped events, a bass envelope and
beat/tempo info. Everything is expressed in seconds on the track's own clock;
the conductor maps that to the shared presentation clock.

The analysis is whole-track: strengths, thresholds and the envelope are
normalised by the track's global maximum, so it cannot run on a live stream.
That suits a demo that plays a chosen file.
"""

from __future__ import annotations

import bisect
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
# the per-bin rise in BOTH the snare body (180-450 Hz, where a snare's shell
# lives) and 2-8 kHz by this factor. A snare raises the body band about as much
# as the low band (its shell), a kick raises almost only the low bins. Linear
# magnitude, not log: log flux saturates on a loud kick. Comparing against 2-8
# kHz alone fails on dark, shell-heavy snares, which have little energy up there.
# Measured on synthetic tracks (see tests/analysis): kick alone >= 12, kick and
# snare together >= 2.85, snare alone <= 1.94, so 2.4 sits in the gap, but the
# margin is thin. A snare whose fundamental is inside the kick band (120 Hz)
# reaches 5.3 and WILL fire a kick: that is ambiguous by frequency alone. NOT
# measured on real music; tune on the demo track.
KICK_DOMINANCE = 2.4
# A snare must raise the body band (180-450 Hz) at least this much per bin
# relative to 2-8 kHz. A hi-hat is all top end and no body, so it fails this; a
# snare raises the body about as much as the top (white-noise snare) or far more
# (dark, shell-heavy snare). Measured on synthetic audio: hi-hat <= 0.42, snare
# >= 1.75, so 0.8 sits in the gap. NOT measured on real hats or snares.
SNARE_MIN_BODY_TO_NOISE = 0.8
# A snare that lands together with a kick must raise the body band at least
# this fraction as much as the low band did. A kick leaks into the body band
# (kick alone <= 0.08) and a hi-hat on the same beat adds nothing there (kick +
# hat <= 0.088 even with a hat as loud as the kick; <= 0.078 at realistic
# levels). A snare adds body: kick + white-noise snare >= 0.097, kick +
# shell-heavy snare >= 0.17. THIN MARGIN for the white-noise case. Kick and
# snare landing together are rare in real music; kick and hat landing together
# are everywhere, so when unsure this errs toward reporting only the kick.
SNARE_MIN_BODY_TO_LOW_WITH_KICK = 0.09
# No tempo is reported (bpm None, no beat grid) unless the onset envelope has a pulse.
# Two independent tests, because autocorrelation alone is fooled by steady tones (a sustained
# chord's partials beat against the frame rate and score 0.69 to 1.00):
#  - confidence: how far the winning autocorrelation peak stands above the median of the search
#    band (0..1). Pulsed synthetic material >= 0.84, room tone / applause / noise <= 0.06.
#  - onset concentration: the share of all onset flux carried by the strongest 5% of frames.
#    Pulsed >= 0.48, pulseless (chords, noise) <= 0.28. A slowly swept sine is NOT rejected: a 50 to 5000 Hz
#    linear chirp over 20 s reads bpm 172 at confidence 0.68 (measured); log chirps, steady tones and vibrato are.
# NOT measured on real music: a dense track with a constant texture may spread its onset energy
# more than these synthetic ones and be wrongly reported as pulseless. That fails safe (no beat
# grid) but must be checked on the demo track (task #7).
TEMPO_MIN_CONFIDENCE = 0.3
TEMPO_MIN_ONSET_CONCENTRATION = 0.38
# Fold a slow tempo up to twice or three times its speed when the faster autocorrelation peak has at
# least this share of the slower one's height, and the slower is at or below this BPM.
OCTAVE_FOLD_BPM = 80.0
OCTAVE_FOLD_SUPPORT = 0.75
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
    tempo_confidence: float = 0.0  # 0..1 autocorrelation salience; 0.0 and bpm None when there is no pulse

    def beat_phase(self, t: float) -> float | None:
        """Position in the current beat, 0 <= phase < 1. None without a tempo.

        Interpolates between the two beats that bracket t, so a beat grid that follows a drifting
        tempo is honoured; outside the grid it extrapolates with the nearest beat interval.
        """
        if not self.bpm or len(self.beat_times) < 2:
            return None
        beats = self.beat_times
        i = bisect.bisect_right(beats, t) - 1
        i = min(max(i, 0), len(beats) - 2)
        return ((t - beats[i]) / (beats[i + 1] - beats[i])) % 1.0


def _parse_wav(blob: bytes, path: str) -> tuple[int, int, int, bytes]:
    """Return (sample_rate, channels, bytes_per_sample, pcm_bytes) from a RIFF/WAVE blob.

    Own parser instead of the stdlib `wave` module: `wave` rejected
    WAVE_FORMAT_EXTENSIBLE (what most tools emit for 24-bit) before Python 3.12
    and accepts it from 3.12, so its behaviour depended on the interpreter.
    """
    if len(blob) < 12 or blob[:4] != b"RIFF" or blob[8:12] != b"WAVE":
        raise ValueError(f"{path} is not a RIFF/WAVE file")
    fmt = None
    pos = 12
    while pos + 8 <= len(blob):
        chunk_id = blob[pos:pos + 4]
        size = int.from_bytes(blob[pos + 4:pos + 8], "little")
        body = pos + 8
        if chunk_id == b"fmt ":
            if size < 16 or body + 16 > len(blob):
                raise ValueError(f"{path} has a truncated fmt chunk")
            tag = int.from_bytes(blob[body:body + 2], "little")
            channels = int.from_bytes(blob[body + 2:body + 4], "little")
            rate = int.from_bytes(blob[body + 4:body + 8], "little")
            bits = int.from_bytes(blob[body + 14:body + 16], "little")
            if tag == 0xFFFE:                               # WAVE_FORMAT_EXTENSIBLE
                if size < 40 or body + 26 > len(blob):
                    raise ValueError(f"{path} has a truncated extensible fmt chunk")
                tag = int.from_bytes(blob[body + 24:body + 26], "little")   # sub-format GUID, first 2 bytes
            fmt = (tag, channels, rate, bits)
        elif chunk_id == b"data":
            if fmt is None:
                raise ValueError(f"{path} has a data chunk before its fmt chunk")
            tag, channels, rate, bits = fmt
            if tag != 1:
                raise ValueError(
                    f"{path} is not integer PCM (format tag {tag}); float and compressed WAV "
                    "are not supported. Re-export as 16- or 24-bit PCM.")
            if bits not in (8, 16, 24, 32) or channels < 1 or rate < 1:
                raise ValueError(f"{path}: unsupported PCM layout ({bits}-bit, {channels} ch, {rate} Hz)")
            end = min(len(blob), body + size)               # a streamed file may claim more than it holds
            width = bits // 8
            end = body + (end - body) // (width * channels) * (width * channels)
            return rate, channels, width, blob[body:end]
        pos = body + size + (size & 1)                      # chunks are word-aligned
    raise ValueError(f"{path} has no data chunk")


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Read an integer-PCM WAV file (8/16/24/32-bit, plain or extensible) as mono float32 in [-1, 1]."""
    with open(path, "rb") as f:
        rate, channels, width, raw = _parse_wav(f.read(), path)
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
    else:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
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


def _lag_value(curve: np.ndarray, lag: np.ndarray) -> np.ndarray:
    """curve[lag] for integer lags, 0 beyond the end."""
    out = np.zeros(len(lag))
    ok = lag < len(curve)
    out[ok] = curve[lag[ok]]
    return out


def _parabola(curve: np.ndarray, i: int) -> float:
    """Sub-sample position of the peak near index i (parabola through i-1, i, i+1)."""
    if 0 < i < len(curve) - 1:
        a, b, c = curve[i - 1], curve[i], curve[i + 1]
        d = a - 2 * b + c
        if d != 0:
            return i + float(np.clip(0.5 * (a - c) / d, -1.0, 1.0))
    return float(i)


def _tempo_period(onset_env: np.ndarray, hop_s: float,
                  bpm_range: tuple[float, float]) -> tuple[float, float] | None:
    """Beat period in (fractional) frames and a 0..1 confidence, or None.

    Unbiased autocorrelation (divided by the number of overlapping samples, so long lags are not
    penalised) and a comb score that also counts the 2x and 4x multiples of a candidate.

    Octave: a kick-then-snare pattern repeats more strongly every two beats than every beat, so the
    slower twin can win the comb. When the winner is at or below OCTAVE_FOLD_BPM and a peak at half
    or a third of its lag has at least OCTAVE_FOLD_SUPPORT of its height, the faster level is taken.
    """
    n = len(onset_env)
    x = onset_env - onset_env.mean()
    ac = _autocorr(x)
    if ac[0] <= 0:
        return None
    idx = np.arange(n)
    unb = ac / np.maximum(n - idx, 1) * n / ac[0]            # unb[0] == 1
    lo = max(1, int(round(60.0 / bpm_range[1] / hop_s)))
    hi = min(n - 1, int(round(60.0 / bpm_range[0] / hop_s)))
    if hi <= lo or n < 4 * lo:
        return None
    lags = np.arange(lo, hi + 1)
    comb = unb[lags] + 0.5 * _lag_value(unb, 2 * lags) + 0.25 * _lag_value(unb, 4 * lags)
    bpm = 60.0 / (lags * hop_s)
    prior = np.exp(-0.5 * (np.log2(bpm / 120.0) / 1.5) ** 2)   # weak: breaks ties, does not decide
    best = int(lags[int(np.argmax(comb * prior))])
    if 60.0 / (best * hop_s) <= OCTAVE_FOLD_BPM:
        for d in (3, 2):
            cand = int(round(best / d))
            near = np.arange(max(1, cand - 2), min(n - 2, cand + 2) + 1)
            top = int(near[int(np.argmax(unb[near]))])
            if not (lo <= top <= hi) or unb[top] < unb[top - 1] or unb[top] < unb[top + 1]:
                continue                                       # outside the search band, or not a peak
            if unb[top] >= OCTAVE_FOLD_SUPPORT * unb[best]:
                best = top
                break
    confidence = float(np.clip(unb[best] - np.median(unb[lo:hi + 1]), 0.0, 1.0))
    return _parabola(unb, best), confidence


def _track_beats(env: np.ndarray, period_f: float) -> np.ndarray | None:
    """Beat positions in frames, following the onsets the beats actually land on.

    A grid extrapolated from one global period walks off the music: a 0.2% period error is 400 ms
    after three minutes, and rounding the period to whole frames is worse. So: anchor on the
    strongest short run of beats, then track outward in both directions, snapping each beat to the
    strongest onset near where the last intervals predict it, and re-estimating the period from
    recent intervals. A beat with no supporting onset coasts on the prediction.
    """
    n = len(env)
    x = np.arange(n, dtype=float)
    run = 12
    best_score, anchor = -1.0, 0.0
    for off in np.arange(0.0, period_f, 0.5):
        vals = np.interp(np.arange(off, n - 1, period_f), x, env)
        if len(vals) < run:
            return None
        csum = np.concatenate([[0.0], np.cumsum(vals)])
        window = csum[run:] - csum[:-run]
        j = int(np.argmax(window))
        if window[j] > best_score:
            best_score, anchor = float(window[j]), off + (j + run // 2) * period_f
    reach = max(2, int(round(0.12 * period_f)))

    def snap(pred: float) -> tuple[float, float] | None:
        c = int(round(pred))
        a, b = max(0, c - reach), min(n, c + reach + 1)
        if b - a < 3:
            return None
        m = a + int(np.argmax(env[a:b]))
        return _parabola(env, m), float(env[m])

    first = snap(anchor)
    if first is None:
        return None
    anchor, ref_height = first
    if ref_height <= 0:
        return None

    def walk(direction: int) -> list[float]:
        out, at, period, intervals = [], anchor, period_f, []
        while True:
            pred = at + direction * period
            if pred < 0 or pred > n - 1:
                return out
            hit = snap(pred)
            if hit is not None and hit[1] >= 0.3 * ref_height \
                    and abs(abs(hit[0] - at) - period_f) <= 0.25 * period_f:
                intervals.append(abs(hit[0] - at))
                period = float(np.clip(np.median(intervals[-8:]), 0.9 * period_f, 1.1 * period_f))
                at = hit[0]
            else:
                at = pred                                     # no supporting onset: coast
            out.append(at)

    beats = np.array(sorted(walk(-1) + [anchor] + walk(+1)))
    return beats if len(beats) >= 4 else None


def _onset_concentration(env: np.ndarray) -> float:
    """Share of the total onset flux carried by the strongest 5% of frames (0..1)."""
    total = float(env.sum())
    if total <= 0:
        return 0.0
    top = np.sort(env)[::-1][: max(1, len(env) // 20)]
    return float(top.sum() / total)


def _tempo(onset_env: np.ndarray, hop_s: float, sr: int,
           bpm_range: tuple[float, float] = (60.0, 180.0)) -> tuple[float | None, list[float], float]:
    """(bpm, beat_times, confidence). bpm is None and beat_times empty when there is no pulse."""
    if onset_env.max() <= 0 or len(onset_env) < 16:
        return None, [], 0.0
    coarse = _tempo_period(onset_env, hop_s, bpm_range)
    if coarse is None:
        return None, [], 0.0
    period_f, confidence = coarse
    if _onset_concentration(onset_env) < TEMPO_MIN_ONSET_CONCENTRATION:
        return None, [], 0.0                                   # no pulse structure at all
    if confidence < TEMPO_MIN_CONFIDENCE:
        return None, [], confidence
    beats = _track_beats(onset_env, period_f)
    if beats is None:
        return None, [], confidence
    slope = float(np.polyfit(np.arange(len(beats)), beats, 1)[0])      # mean period over the whole track
    return 60.0 / (slope * hop_s), [_frame_time(float(b), hop_s, sr) for b in beats], confidence


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
    snare_flux = body_flux                         # not body + noise: a hi-hat is all noise band and no body

    low_rise = _rise(kick / widths[1])
    reference_rise = np.maximum(_rise(snare_body / widths[2]), _rise(snare_noise / widths[3]))
    kick_frames = [f for f in _pick_peaks(kick_flux, hop_s, min_gap_s=0.12)
                   if _peak_max(low_rise, f) >= KICK_DOMINANCE * _peak_max(reference_rise, f)]
    body_rise = _rise(snare_body / widths[2])
    noise_rise = _rise(snare_noise / widths[3])
    snare_frames = []
    for f in _pick_peaks(snare_flux, hop_s, min_gap_s=0.12):
        if _peak_max(body_rise, f) < SNARE_MIN_BODY_TO_NOISE * _peak_max(noise_rise, f):
            continue                                      # top end without a body: a hi-hat
        if (any(abs(f - k) <= 2 for k in kick_frames)
                and _peak_max(body_rise, f) < SNARE_MIN_BODY_TO_LOW_WITH_KICK * _peak_max(low_rise, f)):
            continue                                      # a kick (plus hat, plus its own leakage), not a snare
        snare_frames.append(f)

    full_flux = _flux(full_band)
    onset_frames = _pick_peaks(full_flux, hop_s, min_gap_s=0.05)

    # The envelope is the 50-100 Hz band magnitude per frame, scaled so the loudest frame is 1. It is
    # already smoothed by the 46 ms analysis window (a further 12, 35 or 104 ms moving average was
    # measured to change nothing), and it is a spike train, one spike per kick. It is NOT safe to
    # drive a light directly: measured on synthetic kicks at 128 BPM it crosses 0.5 upward 2.1 times
    # a second for four-on-the-floor, 4.2 for eighth notes and 8.5 for sixteenths, against the cap of
    # three flashes a second in AGENTS.md rule 6. The limiter belongs in the lamp renderer (task #11).
    peak = bass.max()
    env = bass / peak if peak > 0 else bass

    bpm, beats, tempo_confidence = _tempo(kick_flux + snare_flux + full_flux, hop_s, sample_rate)
    events = sorted(
        _events(kick_frames, kick_flux, "kick", hop_s, sample_rate)
        + _events(snare_frames, snare_flux, "snare", hop_s, sample_rate)
        + _events(onset_frames, full_flux, "onset", hop_s, sample_rate),
        key=lambda e: e.t)
    return Analysis(sample_rate=sample_rate, hop_seconds=hop_s, events=events,
                    bass_envelope=env.astype(np.float32), bpm=bpm, beat_times=beats,
                    tempo_confidence=tempo_confidence)


def analyze_file(path: str) -> Analysis:
    samples, rate = load_wav(path)
    return analyze(samples, rate)
