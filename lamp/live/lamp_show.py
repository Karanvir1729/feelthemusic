#!/usr/bin/env python3
"""The LeLamp joins the Feel the Music show as a peer and performs it locally.

Operator tool for the live demo, outside the repo (the team's unified dispatcher, #24, does not exist
yet). It replaces the Mac panel's light pushing -- run one or the other, never both.

v4: colour follows the MELODY (a chroma analysis of the show's own decoded audio, published when the
packet is actually played so it lines up with the speaker), flashes scale with the hit and with an
excitement score, DROP is a white-hot burst, BUILD ramps over the event's duration, and the dance is
beat-locked: a tempo tracker on the scheduled KICK instants predicts beats and the generated
`beat_<tier>_<bpm>` clips are posted so their first beat pose lands on one. The new conductor lists
us as a lamp ("role":"lamp"), streams audio only if we ask ("audio":true), and can switch our mode
from the dashboard (Control "lamp" key); the old conductor ignores all of that and the CLI decides.
v3.1's UDP, clock and audio machinery is unchanged (lamp_show_v31.py is the reference copy).

v2: the lamp drives its own 93-pixel panel directly at 50 fps, so a kick can be a real off->on
flash instead of a 2 s ramp through the runtime's HTTP route (which accepts ~1.6 changes/s). The
runtime must have released the panel first: LELAMP_LED_ENABLED=0 in /etc/lelamp/runtime.env, then
restart lelamp-runtime. If the panel cannot be opened, this falls back to the HTTP route.

Flash safety is enforced HERE, in the process that renders: at most 3 flash onsets in any one
second (the general flash threshold), extra ones are dropped rather than smeared (a refused hit may
become a 0.15 pulse, which is below the 0.2 "flash" threshold), and flashes never go to saturated
red. The hardware brightness is 0.6 (~2.0 A full white of a 3.0 A rating; 5.12-5.14 V on the rail);
a DROP burst may lift it to PEAK_BRIGHTNESS for at most 400 ms. Every number that reaches the panel
passes through safe_output(), the one place that clamps.

Wire facts, from a byte capture of the real conductor:
  hello      0x04 + JSON {"t":"hi",...}; re-sent every 10 s so a conductor restart re-adopts us
  keepalive  SyncReq 0x01 + u16le seq; SyncResp 0x02 | u16 seq | u64 t1 | u64 t2
  Event      26 B: 03 | u32 seq | kind | flags | intensity | sharpness | u16 durMs | u16 freqHz | target
             | u64 masterTs | u32 leadUs; arrives 3x; kinds 1 KICK 2 SNARE 4 BUILD 5 DROP
  Bass       0c | u32 seq | u64 startTs | u8 stepMs | u8 N | N x u8; arrives 2x
  Anchor     09 | u32 seqBase | u64 pts | u16 frames | u32 sampleRate | u8 codec (0 pcm, 1 aac-eld, 2 opus)
  Compact    08 | u16 seq16 | u16 sendLag100us | opus/aac payload (one 10 ms access unit, 480 frames)
  Control    0d + JSON; the new conductor adds "lamp": {"mode": off|light|follow|dance, "lights": bool, "gen": int,
             "bold": 0..1 (the operator's "Bolder moves" slider), "track": face|phone (what follow mode looks for;
             missing keeps the current one, default face; a change while following restarts the follower:
             stop, wait for the old one to leave (FOLLOW_EXIT_S + RESTART_WAIT_S, then SIGKILL), start; a
             newer restart supersedes a pending one, and one that still finds the old follower alive gives
             up loudly and forgets the running track so the next Control with a lamp key retries)}
  Telemetry we send: hello, `cs` and `lamp` (once a second; 5 times a second while the follower sees its
             target). `lamp` carries "target": {kind, seen, center 0..1, aim_deg, age_s} read from follow.py's
             target.json (TargetReport), so the conductor can vibrate the phone harder the nearer the middle
             of the lamp's view it is. Never `au`.

  python3 lamp_show.py --conductor <conductor-ip>                 # light only, safe with follow.py
  python3 lamp_show.py --conductor <conductor-ip> --dance --audio # + beat-locked clips + the show on the speaker
  python3 lamp_show.py --conductor <conductor-ip> --dance --vendor-clips   # v3.1's clips instead (fallback)
  python3 lamp_show.py --conductor <conductor-ip> --dance --pattern rise # one choreography, at the detected tempo
  python3 lamp_show.py --conductor <conductor-ip> --dance --no-live-clips --bold 1  # full-amplitude library only

v4.1: LIVE clips. The dance clips are generated on the lamp, one ahead, at the tracker's exact bpm and the
dashboard's "Bolder moves" slider (Control lamp.bold, 0..1, default 0.6): beat_clips.make_clip() on a worker
thread, validated with the lamp's own calibration (~70 ms on the Pi 5), written atomically into the runtime's
animation pack under one of six pooled names live_0..live_5 (never the one playing or in flight, so the pack
never grows), md5-verified and confirmed in GET /api/animations before it is posted. The post instants, the
one-in-flight rule, the tail-hold re-post and the home on stop are the library's, unchanged; whenever the
live clip is not ready (generation, validation or write failed, no model, no time), the pre-generated
library is allowed only at bold 1. Lower amplitudes wait for matching live generation. Auto chooses the
nearest bucket with a/b/c rotation; forced patterns require their exact tier and variant in that library.
Ready live clips must still match the selected tier, variant, bold and tempo at the posting boundary.
  python3 lamp_show.py --burst-test                             # 5 white bursts, prints the 5 V rail (Pi only)
"""
from __future__ import annotations

import argparse, atexit, colorsys, heapq, json, math, os, signal, socket, struct, subprocess, sys, threading, time, urllib.request

from beat_clips import DANCE_PATTERNS

LAMP = "http://127.0.0.1:8081"
KINDS = {0: "CLICK", 1: "KICK", 2: "SNARE", 3: "BASS", 4: "BUILD", 5: "DROP"}
CLIPS = [("dance_fwd", 12.4), ("robot_dance_fwd", 10.8)]   # vendor fallback, forward-levelled: base_pitch floor -65
PIXELS, PIN, HW_BRIGHTNESS = 93, "D10", 0.6   # full white ~2.0 A of a 3.0 A rating
PEAK_BRIGHTNESS = 0.75                        # DROP burst only, <= 400 ms; still to be verified on the rail (--burst-test)
PEAK_MAX_S = 0.4                              # the panel is never above HW_BRIGHTNESS for longer than this
NOTE_NAMES = ["C", "G", "D", "A", "E", "B", "F#", "C#", "G#", "D#", "A#", "F"]   # by circle-of-fifths index
MODES = ("off", "light", "follow", "dance")
SDK = "lamp_show v4"
HOLD_NS = 600_000_000                         # the 0.6 s hold at the head (and tail) of every generated clip
CLIP_BEATS = 8
# beat_clips.py v2 writes MANIFEST.json next to the staged clips: ~/feelthemusic-lamp/beat_clips/ on the
# lamp, the STAGE dir of lamp-tools/install_clips.py, which copies the CSVs (not the manifest) into the
# runtime's animation pack (see lamp-tools/README.md). The manifest lists the a/b/c variants of every
# beat_<tier>_<bpm> and the "build" tier. Without it the scheduler posts the unsuffixed v1 names, which
# v2 keeps as aliases, and never picks build. Looked for at DEFAULT_MANIFEST, then in beat_clips/ next
# to this file (the same place when lamp_show.py runs from ~/feelthemusic-lamp).
DEFAULT_MANIFEST = os.path.expanduser("~/feelthemusic-lamp/beat_clips/MANIFEST.json")
MANIFEST_CANDIDATES = (DEFAULT_MANIFEST,
                       os.path.join(os.path.dirname(os.path.abspath(__file__)), "beat_clips", "MANIFEST.json"))
# The runtime's animation pack: a CSV written here is listed by GET /api/animations without a restart, but a
# half-written one blocks the runtime at boot (hence the atomic writes and the startup sweep of live_*).
PACK_DIR = os.path.expanduser("~/lelamp-hackathon-2026/static/robots/lelamp_v1/pi5_feetech_r1/animations/factory_v1")
DEFAULT_BOLD = 0.6                            # the dashboard slider's default
# A live clip made within this of the tracker's bpm is the right clip: the beat tracker's PLL trims the
# period continuously (with real kick jitter the bpm wanders +-1 several times per 8-beat cycle), and a
# 2 bpm mismatch over 8 beats is ~60 ms at the last beat, well inside the hold. The scheduler uses the
# same band when it takes a ready clip and when it decides whether one must be regenerated, so only a
# bold/tier/variant change (or a real tempo re-lock) costs a worker job, a CSV write and a listing check.
LIVE_BPM_TOL = 2.0
# follow.py --live (run_face.sh) writes what it sees here every cycle; see TargetFile.
TARGET_FILE = os.path.expanduser("~/feelthemusic-lamp/target.json")
TRACKS = ("face", "phone")                    # Control lamp.track: what follow mode looks for
FOLLOW_LOG = "/tmp/follow-face.log"
# Restarting the follower on a track change: the outgoing follow.py settles the arm and restores the
# runtime's idle on its way out (poster.wait_idle up to 5 s, a settle POST, idle_restore), so it is given
# FOLLOW_EXIT_S after SIGTERM before the start is queued, the start polls another RESTART_WAIT_S for it to
# disappear, then SIGKILLs it and waits FOLLOW_KILL_GRACE_S. run_face.sh inherits FOLLOW_ARGS (extra
# follow.py options, e.g. the step and period) from this process's environment.
FOLLOW_EXIT_S, RESTART_WAIT_S, FOLLOW_KILL_GRACE_S, FOLLOW_POLL_S = 6.0, 3.0, 2.0, 0.2
LAMP_PERIOD_S, LAMP_PERIOD_SEEN_S = 1.0, 0.2  # `lamp` telemetry cadence: idle, and while a target is seen


def lamp(path: str, body: dict | None = None, timeout: float = 4.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(LAMP + path, data=data, method="POST" if data else "GET",
                                 headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as exc:
        return {"error": str(exc)}


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """NaN-proof clamp: a NaN anywhere in the colour maths must not reach the LEDs."""
    if x != x:
        return lo
    return lo if x < lo else hi if x > hi else x


def safe_output(hue: float, sat: float, value: float, brightness: float, dark: bool = False):
    """The ONE place where colour numbers become panel bytes. Everything is clamped here, whatever
    the model upstream did: value and saturation to [0, 1], brightness to [0, PEAK_BRIGHTNESS]."""
    brightness = clamp(brightness, 0.0, PEAK_BRIGHTNESS)
    if dark:
        return (0, 0, 0), min(brightness, HW_BRIGHTNESS)
    hue = hue % 1.0 if hue == hue else 0.0
    r, g, b = colorsys.hsv_to_rgb(hue, clamp(sat), clamp(value))
    return (int(clamp(r) * 255), int(clamp(g) * 255), int(clamp(b) * 255)), brightness


# ---- pure pieces (importable and testable on the Mac: no board/neopixel/av/sounddevice here) -------
class FlashLimiter:
    """Counts flash onsets. Allows at most `max_per_s` in any 1 s window; the rest are dropped."""
    def __init__(self, max_per_s=3):
        self.max_per_s, self.onsets = max_per_s, []

    def allow(self, now: float) -> bool:
        self.onsets = [t for t in self.onsets if now - t <= 1.0]
        if len(self.onsets) >= self.max_per_s:
            return False
        self.onsets.append(now)
        return True


def flash_target_for(kind: str, intensity: float, excite: float) -> float:
    """How bright a hit wants to flash. Scales with the conductor's intensity and, for kicks, with how
    excited the last two seconds were, so a crazy section flashes harder than a verse."""
    if kind == "KICK":
        return min(1.0, 0.45 + 0.45 * intensity + (0.1 if excite > 0.6 else 0.0))
    if kind == "SNARE":
        return 0.35 + 0.30 * intensity
    if kind == "DROP":
        return 1.0
    return 0.0


class Chroma:
    """Which note is the music on? A rolling 4096-sample window of the decoded 48 kHz mono stream,
    Hann-windowed rfft per 10 ms packet, 70-5000 Hz weighted toward the mid band (bass is muddy,
    treble is mostly harmonics). Energy is attributed by SPECTRAL PEAK, with the peak's frequency
    interpolated between bins: at 4096 samples a bin is 11.7 Hz, close to a semitone around middle C,
    so a plain per-bin pitch class splits one note over two classes depending on where it falls on
    the grid. Peaks are summed by pitch class into 12 CIRCLE-OF-FIFTHS bins so that neighbouring
    bins are related keys (and a tritone is opposite). Loudness has its own slow AGC: a
    running peak in dB that rises instantly and decays 3 dB/s, with the usable range 30 dB below it,
    so a quiet passage is dim and the loud one after it is bright without anyone touching a gain."""
    SR, N, HOP = 48000, 4096, 480
    F_LO, F_HI, SILENCE_DB, FLOOR_DB, WINDOW_DB, DECAY_DB_S = 70.0, 5000.0, -55.0, -45.0, 30.0, 3.0

    def __init__(self):
        import numpy as np
        self.np = np
        self.buf = np.zeros(self.N, dtype=np.float32)
        self.window = np.hanning(self.N).astype(np.float32)
        freqs = np.fft.rfftfreq(self.N, 1.0 / self.SR)
        band = (freqs >= self.F_LO) & (freqs <= self.F_HI)
        self.bins = np.nonzero(band)[0]
        f = freqs[self.bins]
        w = np.ones_like(f)
        w[f < 250.0] = (f[f < 250.0] / 250.0) ** 1.0
        w[f > 2500.0] = (2500.0 / f[f > 2500.0]) ** 1.5
        self.weights = w
        midi = np.rint(12.0 * np.log2(f / 440.0) + 69.0).astype(int)
        self.fifths = ((midi % 12) * 7) % 12                   # per-bin classes, for reference/tests
        self.lo, self.hi = int(self.bins[0]), int(self.bins[-1])
        self.chroma = np.zeros(12)
        self.peak_db = self.FLOOR_DB
        self.log12 = math.log(12.0)

    def peaks_to_fifths(self, spec):
        """Local maxima in the band, each with a parabolic frequency estimate, summed by fifths index."""
        np = self.np
        lo, hi = self.lo, self.hi
        mid = spec[lo:hi + 1]; left = spec[lo - 1:hi]; right = spec[lo + 1:hi + 2]
        is_peak = (mid > left) & (mid >= right) & (mid > 1e-6)
        idx = np.nonzero(is_peak)[0]
        if len(idx) == 0:
            return np.zeros(12)
        a, b, c = left[idx], mid[idx], right[idx]
        denom = a - 2.0 * b + c
        d = np.where(np.abs(denom) > 1e-12, 0.5 * (a - c) / np.where(np.abs(denom) > 1e-12, denom, 1.0), 0.0)
        f = (lo + idx + d) * (self.SR / self.N)
        midi = np.rint(12.0 * np.log2(np.maximum(f, 1.0) / 440.0) + 69.0).astype(int)
        fifths = ((midi % 12) * 7) % 12
        return np.bincount(fifths, weights=b * self.weights[idx], minlength=12)

    def feed(self, pcm) -> tuple:
        """One decoded packet in, (fifths_index or -1, confidence, loudness, tonal) out."""
        np = self.np
        n = len(pcm)
        if n >= self.N:
            self.buf[:] = pcm[-self.N:]
        elif n:
            self.buf[:-n] = self.buf[n:]
            self.buf[-n:] = pcm
        rms = float(np.sqrt(np.mean(np.square(pcm, dtype=np.float32)))) if n else 0.0
        level_db = 20.0 * math.log10(rms + 1e-9)
        # AGC: the peak rises instantly, decays 3 dB/s, never below the floor (so hiss is not "loud").
        dt = n / self.SR
        self.peak_db = max(level_db, self.peak_db - self.DECAY_DB_S * dt, self.FLOOR_DB)
        loud = clamp((level_db - (self.peak_db - self.WINDOW_DB)) / self.WINDOW_DB)
        spec = np.abs(np.fft.rfft(self.buf * self.window))
        raw = self.peaks_to_fifths(spec)
        self.chroma = 0.8 * self.chroma + 0.2 * raw
        total = float(self.chroma.sum())
        if level_db < self.SILENCE_DB or total <= 1e-9:
            return (-1, 0.0, 0.0 if level_db < self.SILENCE_DB else loud, 0.0)
        share = self.chroma / total
        idx = int(share.argmax())
        conf = float(share[idx])
        nz = share[share > 0]
        entropy = float(-(nz * np.log(nz)).sum()) / self.log12
        tonal = clamp(1.0 - 1.15 * entropy)
        return (idx, conf, loud, tonal)


class ColourModel:
    """The panel's colour, one step per 50 fps frame. Pure: the caller feeds audio analysis, bass,
    excitement and events; step() returns (hue, sat, value, brightness) BEFORE the final clamp.
    Hue follows the note wheel (a note must persist 3 frames at conf >= 0.22 and 120 ms after the last
    change, so a passing harmonic does not jerk the colour), saturation follows tonal purity, value is
    loudness + bass + excitement, and hits, BUILD and DROP sit on top. Without audio analysis the hue is
    nudged by events as in v3.1."""
    FPS, NOTE_TAU, STALE_AUDIO_S = 50.0, 0.1, 2.0
    NOTE_CONF, NOTE_FRAMES, NOTE_GAP_S = 0.22, 3, 0.12
    LOUD_RISE, LOUD_FALL = 0.35, 0.15     # per-frame smoothing of the 10 ms loudness (see step)

    def __init__(self):
        self.hue, self.bass, self.flash, self.flash_target = 0.62, 0.0, 0.0, 0.0
        self.note, self.cand, self.cand_frames, self.note_changed_at, self.note_flash = -1, -1, 0, -1e9, 0.0
        self.loud, self.tonal, self.conf, self.excite, self.music = 0.0, 1.0, 0.0, 0.0, False
        self.audio, self.audio_at = None, -1e9
        self.build_start, self.build_end = None, None
        self.drop_at, self.peak_since = None, None
        self.build = 0.0

    # inputs
    def set_audio(self, ana, now: float):
        """A new tuple means a packet was played; the same tuple again means the audio has stopped."""
        if ana is not None and ana is not self.audio:
            self.audio, self.audio_at = ana, now

    def has_audio(self, now: float) -> bool:
        return self.audio is not None and now - self.audio_at < self.STALE_AUDIO_S

    def nudge_hue(self, kind: str, intensity: float, now: float):
        """v3.1's event-driven hue moves; only when there is no melody to follow."""
        if self.has_audio(now):
            return
        if kind == "DROP":    self.hue = (self.hue + 0.5) % 1.0
        elif kind == "BUILD": self.hue = (self.hue + 0.12) % 1.0
        elif kind == "KICK":  self.hue = (self.hue + 0.06 * intensity) % 1.0
        elif kind == "SNARE": self.hue = (self.hue - 0.04) % 1.0

    def request_flash(self, target: float, limiter: FlashLimiter, now: float) -> bool:
        """Every flash >= 0.2 goes through the limiter; a refused one may still be a 0.15 pulse (not a
        flash by the general threshold). Returns whether a real flash was started."""
        if target < 0.2:
            self.flash_target = max(self.flash_target, target)
            return False
        if limiter.allow(now):
            self.flash_target = max(self.flash_target, target)
            return True
        self.flash_target = max(self.flash_target, 0.15)
        return False

    def start_build(self, now: float, dur_s: float):
        self.build_start, self.build_end = now, now + max(0.05, dur_s)

    def start_drop(self, now: float):
        self.drop_at = now
        self.build_start = self.build_end = None

    # one frame
    def step(self, now: float):
        dt = 1.0 / self.FPS
        if self.has_audio(now):
            note, conf, loud, tonal = self.audio
            # The loudness is a 10 ms RMS: a kick over near-silence would step the value by its full
            # 0.30 weight in one frame, a flash that never met the limiter. Smooth it (faster up than
            # down) so only the limiter-gated flash path can move the value by >= 0.2 per frame.
            self.loud += (self.LOUD_RISE if loud > self.loud else self.LOUD_FALL) * (loud - self.loud)
            self.tonal, self.conf = tonal, conf
            if note >= 0 and conf >= self.NOTE_CONF:
                if note == self.cand: self.cand_frames += 1
                else: self.cand, self.cand_frames = note, 1
            else:
                self.cand, self.cand_frames = -1, 0
            if (self.cand >= 0 and self.cand != self.note and self.cand_frames >= self.NOTE_FRAMES
                    and now - self.note_changed_at >= self.NOTE_GAP_S):
                self.note, self.note_changed_at, self.note_flash = self.cand, now, 0.12
            if self.note >= 0:
                target = self.note / 12.0
                d = ((target - self.hue + 0.5) % 1.0) - 0.5          # shortest way round the wheel
                self.hue = (self.hue + 0.6 * d) % 1.0
        else:
            self.loud, self.tonal, self.conf = 0.0, 1.0, 0.0
        if not self.music:
            self.hue = (self.hue + 0.02 * dt) % 1.0
        self.note_flash *= math.exp(-dt / self.NOTE_TAU)
        # BUILD: 0 -> 1 over the event's duration, then released.
        if self.build_end is not None and now < self.build_end:
            self.build = clamp((now - self.build_start) / (self.build_end - self.build_start))
        else:
            self.build, self.build_start, self.build_end = 0.0, None, None
        b2 = self.build * self.build
        build_scale = 1.0 - 0.5 * b2
        base = 0.06 + 0.30 * self.loud + 0.14 * self.bass + 0.25 * self.excite + 0.35 * b2
        value = min(1.0, base + self.flash + self.note_flash)
        sat = (0.55 + 0.45 * self.tonal) * (1.0 - 0.6 * self.flash) * build_scale
        brightness = HW_BRIGHTNESS
        if self.drop_at is not None:
            age = now - self.drop_at
            if age < 0.15:
                value, sat, brightness = 1.0, 0.0, PEAK_BRIGHTNESS
            elif age < 0.9:
                u = (age - 0.15) / 0.75
                value, sat = max(value, 1.0 - 0.7 * u), min(sat, 0.4 + 0.6 * u)
            elif age < 2.0:
                value = max(value, 0.7)
            else:
                self.drop_at = None
        if not self.music:
            value = max(value, 0.10)
        # Never above the hardware brightness for longer than PEAK_MAX_S, however many DROPs arrive.
        if brightness > HW_BRIGHTNESS:
            self.peak_since = self.peak_since if self.peak_since is not None else now
            if now - self.peak_since > PEAK_MAX_S:
                brightness = HW_BRIGHTNESS
        else:
            self.peak_since = None
        # v3.1's flash envelope: 0.2 -> 0.9 in 5 frames, then 0.78 per frame back down.
        if self.flash_target > self.flash + 0.01:
            self.flash = min(self.flash_target, self.flash + 0.70 / 5)
        else:
            self.flash_target = 0.0
            self.flash *= 0.78
            if self.flash < 0.02: self.flash = 0.0
        return self.hue, sat, value, brightness


class BeatTracker:
    """Tempo and phase from the scheduled KICK instants (ns on the lamp clock). Period = median of the
    inter-onset intervals in 0.28-0.75 s (80-214 bpm) over the last 16 kicks, with one x2 / x0.5 fold
    for a skipped or doubled kick; phase by a PLL: each kick pulls the beat grid 30 % of the way to it
    and trims the period. Four consistent kicks at a clearly different tempo re-lock at once, so a
    tempo change is followed within a bar instead of being averaged away."""
    LO, HI, MAX_KICKS, EXPIRE_NS = 0.28, 0.75, 16, 4_000_000_000
    ON_GRID = 0.15                 # a kick within this fraction of a period of the grid counts as on the beat

    def __init__(self):
        self.kicks: list[int] = []
        self.period, self.grid, self.last_ns = None, None, None
        self.errors: list[float] = []
        self.on_grid: list[bool] = []   # per kick since the period was set: landed within ON_GRID of the grid

    @classmethod
    def fold(cls, ioi_s: float):
        if cls.HI < ioi_s <= 2 * cls.HI: ioi_s /= 2.0          # one kick skipped
        elif cls.LO / 2 <= ioi_s < cls.LO: ioi_s *= 2.0        # a double hit
        return ioi_s if cls.LO <= ioi_s <= cls.HI else None

    def iois(self, last: int | None = None) -> list[float]:
        ks = self.kicks if last is None else self.kicks[-(last + 1):]
        out = [self.fold((b - a) / 1e9) for a, b in zip(ks, ks[1:])]
        return [x for x in out if x is not None]

    def feed(self, t_ns: int):
        if self.last_ns is not None and t_ns - self.last_ns > self.EXPIRE_NS:
            self.__init__()
        if self.kicks and t_ns - self.kicks[-1] < 60_000_000:      # the same hit reported twice
            return
        self.kicks = (self.kicks + [t_ns])[-self.MAX_KICKS:]
        self.last_ns = t_ns
        iois = self.iois()
        if len(iois) < 2:
            return
        median = sorted(iois)[len(iois) // 2]
        if self.period is None:
            self.period, self.grid = median, t_ns
            return
        beats = max(1, round((t_ns - self.grid) / (self.period * 1e9)))
        predicted = self.grid + beats * self.period * 1e9
        error = (t_ns - predicted) / 1e9
        self.on_grid = (self.on_grid + [abs(error) <= self.ON_GRID * self.period])[-8:]
        if abs(error) <= 0.2 * self.period:
            # On or near the beat: pull the grid and trim the period. A kick further off (a half-beat
            # of double time, a syncopation) is not a beat and must not drag the grid toward it.
            self.errors = (self.errors + [error])[-16:]
            self.grid = predicted + 0.3 * error * 1e9
            self.period = clamp(self.period + 0.1 * error / beats, self.LO, self.HI)
        recent = self.iois(last=4)
        if len(recent) >= 4 and max(recent) - min(recent) <= 0.06 * min(recent):
            m = sorted(recent)[2]
            if abs(m - self.period) > 0.04 * self.period:            # a real tempo change: re-lock now
                self.period, self.grid, self.errors, self.on_grid = m, t_ns, [], []

    @property
    def bpm(self) -> float:
        return 60.0 / self.period if self.period else 0.0

    def locked(self, now_ns: int) -> bool:
        return self.stable() and self.last_ns is not None and now_ns - self.last_ns <= self.EXPIRE_NS

    def stable(self) -> bool:
        """Locked when the kicks land on the grid, not when they are evenly spaced: real drum patterns
        skip beats and add off-beat kicks (measured tonight: a 120 bpm track sat at a +-14 ms phase
        error for three minutes while the old uniform-spacing test refused to lock). Four of the last
        six kicks within ON_GRID of a beat, after at least four kicks, is a lock."""
        if self.period is None or len(self.kicks) < 4 or not self.on_grid:
            return False
        recent = self.on_grid[-8:]
        return sum(recent) >= math.ceil(0.6 * len(recent))   # 5 of 8 once warm; cleaner while few kicks

    def next_beats(self, after_ns: int, n: int) -> list[int]:
        if not self.locked(after_ns):
            return []
        p = self.period * 1e9
        k = math.floor((after_ns + 1_000_000 - self.grid) / p) + 1     # beats more than 1 ms after `after_ns`
        return [int(self.grid + (k + i) * p) for i in range(n)]

    def phase_error_ms(self) -> float:
        return 1e3 * sum(self.errors) / len(self.errors) if self.errors else 0.0


class ClipScheduler:
    """Beat-locked dance. When the music is on and the tracker is stable it posts the generated clip
    of the right tier and bpm bucket so that the clip's first beat pose (HOLD after the first frame
    reaches the servos, itself START_LATENCY after the POST) lands on a predicted beat, then re-posts
    every 8 beats. Posts run on their own thread; never two in flight.

    Re-post timing: the clip's pattern ends back at START exactly on the boundary beat (beat + 8 P)
    and the arm then holds START for HOLD. The next clip's first frame reaches the servos HOLD before
    its own first beat, so that beat must be at least HOLD (plus the runtime's ~100 ms start jitter)
    after the boundary, or the first frame arrives while the arm is still on the way back from pose
    7 and the runtime crossfades 0.5-2 s from there instead of skipping its blend-in -- which smeared
    the first beats of every clip and let the beat lock slip. The arm rests 1-3 beats every 8."""
    BUCKETS = tuple(range(80, 181, 4))
    LATE_NS = 100_000_000
    START_JITTER_NS = 100_000_000      # POST -> first frame is 250-450 ms; plan for the early end

    LIVE_BPM_TOL = LIVE_BPM_TOL        # a ready live clip within this of the tracker's bpm is still the right clip
    VARIANTS = ("a", "b", "c")         # the live rotation (the library's letters come from the manifest)

    def __init__(self, post=None, status=None, start_latency_ns: int = 350_000_000, hold_ns: int = HOLD_NS,
                 available: set | None = None, log=print, manifest: dict | None = None,
                 live: "LiveClips | None" = None, bold: float = DEFAULT_BOLD, pattern: str = "auto"):
        self.post_fn = post or (lambda name: lamp("/api/animations/play", {"name": name}))
        self.status_fn = status or (lambda: lamp("/api/animations/status", timeout=1.0))
        self.start_latency_ns, self.hold_ns, self.log = int(start_latency_ns), int(hold_ns), log
        self.available = available
        self.live, self.bold = live, clamp(float(bold))
        self.pattern = "auto"
        self.set_pattern(pattern)
        self.live_posts = 0                        # clips posted from the live generator (the rest: the library)
        self.variants: dict[str, list[str]] = {}   # beat_<tier>_<bpm> -> its a/b/c clip names (v2 manifest)
        self.last_variant: dict[str, str] = {}     # tier -> the variant letter played last (rotation state)
        self.build_until_ns = None                 # a BUILD event is pending until this instant
        if manifest:
            self.load_manifest(manifest)
        self.boundary_ns, self.inflight, self.playing, self.drop_pending = None, False, False, False
        self.pending_home = None           # (hold_ns, after) asked for while a post was in flight
        self.not_before_ns = 0             # no clip posts before this (the pre-move to home settles)
        self.moves, self.refused = 0, 0
        self.phase_ms: list[float] = []
        self.lock = threading.Lock()

    @staticmethod
    def post_instant(beat_ns: int, start_latency_ns: int, hold_ns: int = HOLD_NS) -> int:
        """When to POST so the clip's first beat pose lands on `beat_ns`."""
        return int(beat_ns - start_latency_ns - hold_ns)

    @staticmethod
    def bucket(bpm: float) -> int:
        return min(ClipScheduler.BUCKETS, key=lambda b: abs(b - bpm))

    @staticmethod
    def tier_for(excite: float, drop: bool, build: bool = False) -> str:
        return "drop" if drop else ("build" if build else ("groove" if excite < 0.55 else "hype"))

    @staticmethod
    def base_of(name: str) -> str:
        """beat_<tier>_<bpm>_<v> -> beat_<tier>_<bpm>; any other name unchanged."""
        p = name.split("_")
        return "_".join(p[:3]) if len(p) == 4 and p[0] == "beat" and p[2].isdigit() and len(p[3]) == 1 else name

    def load_manifest(self, manifest: dict) -> int:
        """beat_clips v2 MANIFEST.json: every clip carries base + variant; the unsuffixed v1 names are
        listed with alias_of and skipped. Returns how many bases have variants."""
        groups: dict[str, list[tuple[str, str]]] = {}
        for c in manifest.get("clips", []) if isinstance(manifest, dict) else []:
            if not isinstance(c, dict) or c.get("alias_of") or not c.get("variant") or not c.get("name"):
                continue
            groups.setdefault(c.get("base") or self.base_of(c["name"]), []).append((str(c["variant"]), c["name"]))
        self.variants = {b: [n for _, n in sorted(v)] for b, v in groups.items()}
        return len(self.variants)

    def has_tier(self, tier: str) -> bool:
        if self.live is not None and self.live.usable():      # the generator makes every tier
            return True
        prefix = f"beat_{tier}_"
        return any(b.startswith(prefix) for b in self.variants) or \
            (self.available is not None and any(n.startswith(prefix) for n in self.available))

    def set_bold(self, bold: float) -> bool:
        """The dashboard slider (Control lamp.bold), clamped to 0..1. Old prepared clips are not
        substitutes for the new amplitude; the next tick prepares a matching clip if time permits."""
        b = round(clamp(float(bold)), 2)
        if b == self.bold:
            return False
        self.log(f"{time.strftime('%H:%M:%S')} bold {self.bold:.2f} -> {b:.2f}")
        self.bold = b
        if self.live is not None:
            self.live.cancel()
        return True

    def set_pattern(self, pattern: str) -> bool:
        """Choose one existing choreography, or auto tier selection and variant rotation."""
        if not isinstance(pattern, str) or pattern not in DANCE_PATTERNS:
            raise ValueError(f"unknown dance pattern {pattern!r}; choose {', '.join(DANCE_PATTERNS)}")
        if pattern == self.pattern:
            return False
        self.log(f"{time.strftime('%H:%M:%S')} pattern {self.pattern} -> {pattern}")
        self.pattern = pattern
        if self.live is not None:
            self.live.cancel()
        return True

    def selection(self, tier: str) -> tuple[str, str]:
        """One tier/variant resolution shared by live generation, take-time checks and the library."""
        return DANCE_PATTERNS[self.pattern] or (tier, self.next_letter(tier))

    def next_letter(self, tier: str) -> str:
        """The variant the a -> b -> c rotation of `tier` plays next (pick()'s rule), without committing."""
        letters = list(self.VARIANTS)
        last = self.last_variant.get(tier)
        if last in letters:
            return letters[(letters.index(last) + 1) % len(letters)]
        return next((l for l in letters if last is not None and l > last), letters[0])

    def prepare(self, tier: str, bpm: float, post_at: int, now_ns: int):
        """Ahead of the post: ask the generator for this tier's next variant at the tracker's exact bpm and
        the current bold, unless the ready (or in-progress) clip already is that, or there is no time
        left before the post instant (then only matching ready clips can be posted)."""
        live = self.live
        if live is None or not live.usable(now_ns):
            return
        tier, letter = self.selection(tier)
        if live.covers(tier, letter, bpm, self.bold):
            return
        if post_at - now_ns < live.PREP_NS:
            return
        live.request(tier, letter, bpm, self.bold, post_at)

    def take_live(self, tier: str, bpm: float):
        """Only a ready clip matching current choreography, amplitude and tempo may be posted."""
        if self.live is None:
            return None
        tier, letter = self.selection(tier)
        return self.live.take(lambda r: r.covers(tier, letter, bpm, self.bold, self.LIVE_BPM_TOL))

    def note_build(self, now_ns: int, dur_ms: int):
        """A BUILD event with a known duration: the build tier is chosen for clips whose first half lies
        inside the build (until it ends, or a DROP resolves it)."""
        if dur_ms > 0:
            self.build_until_ns = int(now_ns + dur_ms * 1_000_000)

    def pick(self, tier: str, bpm: float) -> str | None:
        """The clip to post: the base of the nearest bucket (choose_clip, on base names) and, when
        variants of it exist, the next one in the a -> b -> c rotation of that tier -- never the variant
        played last. Without variants the base name itself (v1 library, or a v2 alias).
        Library clips are full amplitude: a lower bold request must wait for live generation.
        Forced patterns never substitute a different tier, variant or unverified alias."""
        if self.bold < 1.0:
            return None
        if self.pattern != "auto":
            tier, letter = self.selection(tier)
            known = self.available if self.available is not None else {n for names in self.variants.values() for n in names}
            matching = {n for n in known if len(n.split("_")) == 4 and n.startswith(f"beat_{tier}_")
                        and n.endswith(f"_{letter}") and n.split("_")[2].isdigit()}
            base = self.choose_clip(tier, bpm, {self.base_of(n) for n in matching})
            if base is None:
                return None
            self.last_variant[tier] = letter
            return f"{base}_{letter}"
        bases = None if self.available is None else {self.base_of(n) for n in self.available}
        base = self.choose_clip(tier, bpm, bases)
        if base is None:
            return None
        names = [n for n in self.variants.get(base, []) if self.available is None or n in self.available]
        if not names and self.available is not None:      # variants on the lamp, no manifest here
            names = sorted(n for n in self.available if n != base and self.base_of(n) == base)
        if not names:
            return base
        t = base.split("_")[1]
        letters = [n.rsplit("_", 1)[1] for n in names]
        last = self.last_variant.get(t)
        if last in letters:
            i = (letters.index(last) + 1) % len(letters)
        else:
            i = next((k for k, l in enumerate(letters) if last is not None and l > last), 0)
        self.last_variant[t] = letters[i]
        return names[i]

    @staticmethod
    def choose_clip(tier: str, bpm: float, available=None) -> str | None:
        """Nearest bucket of the tier; if the library lacks it, the nearest bucket that exists in
        this tier, then in the other tiers. None when there is nothing to play."""
        want = ClipScheduler.bucket(bpm)
        if available is None:
            return f"beat_{tier}_{want}"
        for t in (tier, "hype", "groove", "drop"):
            names = [n for n in available if n.startswith(f"beat_{t}_")]
            if not names:
                continue
            def key(n):
                try: return abs(int(n.rsplit("_", 1)[1]) - bpm)
                except ValueError: return 1e9
            best = min(names, key=key)
            if key(best) <= 6:
                return best
        return None

    def plan(self, now_ns: int, tracker: BeatTracker):
        """(post_at_ns, beat_ns, period_ns) for the next clip, or None."""
        beats = tracker.next_beats(now_ns, 32)
        if not beats:
            return None
        period_ns = int(tracker.period * 1e9)
        if self.boundary_ns is None:
            # The first beat whose post instant has not passed by more than LATE_NS: the loop polls
            # every few ms, so "not yet passed at all" would slide to the next beat forever.
            for b in beats:
                if self.post_instant(b, self.start_latency_ns, self.hold_ns) >= now_ns - self.LATE_NS:
                    return (self.post_instant(b, self.start_latency_ns, self.hold_ns), b, period_ns)
            return None
        # A re-post: the first predicted beat whose first frame (HOLD before it, up to START_JITTER
        # early) arrives once the current clip is holding START, i.e. at or after the boundary.
        earliest = self.boundary_ns + self.hold_ns + self.START_JITTER_NS
        for b in beats:                                              # ascending: the first that fits
            if b >= earliest - 1_000_000 and self.post_instant(b, self.start_latency_ns, self.hold_ns) >= now_ns - self.LATE_NS:
                return (self.post_instant(b, self.start_latency_ns, self.hold_ns), b, period_ns)
        return None

    def reset(self):
        self.boundary_ns, self.drop_pending, self.build_until_ns = None, False, None

    def tick(self, now_ns: int, active: bool, tracker: BeatTracker, excite: float):
        if not active:
            if self.playing or self.boundary_ns is not None:
                self.stop()
            return
        with self.lock:
            if self.inflight:
                return
        if now_ns < self.not_before_ns:
            # The hold behind the home pre-move: no post, but the first clip can be made meanwhile (its
            # content depends on tier, variant, bpm and bold, not on which beat it will land on).
            if tracker.locked(now_ns):
                self.prepare(self.tier_for(excite, self.drop_pending), tracker.bpm, self.not_before_ns, now_ns)
            return
        p = self.plan(now_ns, tracker)
        if p is None:
            return
        post_at, beat, period_ns = p
        if self.build_until_ns is not None and now_ns >= self.build_until_ns:
            self.build_until_ns = None
        # build only when the BUILD outlasts at least half the clip: the crouch is deepest on beat 6, so a
        # build that ends a beat after the first one would crouch through and past the drop hit
        build = (self.build_until_ns is not None and self.has_tier("build")
                 and beat + (CLIP_BEATS // 2) * period_ns <= self.build_until_ns)
        tier = self.tier_for(excite, self.drop_pending, build)
        if now_ns < post_at:
            self.prepare(tier, tracker.bpm, post_at, now_ns)      # the next clip, generated ahead of its post
            return
        live = self.take_live(tier, tracker.bpm)
        if live is not None:
            name = live.name
            self.last_variant[live.tier] = live.variant             # the rotation moves on as pick() would
        else:
            name = self.pick(tier, tracker.bpm)
        if name is None:
            self.refused += 1; self.boundary_ns = beat + CLIP_BEATS * period_ns
            return
        if self.drop_pending:
            self.build_until_ns = None                        # the drop resolves the build
        self.drop_pending = False
        self.boundary_ns = beat + CLIP_BEATS * period_ns
        with self.lock:
            self.inflight = True
        threading.Thread(target=self._post, args=(name, beat, now_ns - post_at), daemon=True).start()

    def _post(self, name: str, beat_ns: int, late_ns: int):
        t0 = time.monotonic_ns()
        r = self.post_fn(name)
        ok = r.get("status") == "started"
        if ok:
            self.moves += 1; self.playing = True
        else:
            self.refused += 1
        if self.live is not None and self.live.is_pool_name(name):
            self.live.posted(name, ok)
            if ok: self.live_posts += 1
        self.log(f"{time.strftime('%H:%M:%S')} clip {name} for beat {beat_ns / 1e9:.3f}s posted {late_ns / 1e6:+.0f}ms "
                 f"after its instant -> {r.get('status') or r.get('error')}")
        if ok:
            self._measure(name, beat_ns, t0)
        with self.lock:
            self.inflight = False
            pend, self.pending_home = self.pending_home, None
        if pend is not None:                                  # a stop/mode change arrived meanwhile
            self.home(*pend)

    def _measure(self, name: str, beat_ns: int, t0: int):
        """Poll the runtime until it reports the clip running; the first frame's instant vs the planned
        one (beat - HOLD) is the phase error the operator can see."""
        deadline = t0 + 1_500_000_000
        while time.monotonic_ns() < deadline:
            st = self.status_fn()
            if st.get("current_animation") == name and st.get("playing"):
                el = float(st.get("elapsed_seconds") or 0.0)
                first = time.monotonic_ns() - int(el * 1e9)
                err = (first - (beat_ns - self.hold_ns)) / 1e6
                self.phase_ms = (self.phase_ms + [err])[-50:]
                mean = sum(self.phase_ms) / len(self.phase_ms)
                self.log(f"{time.strftime('%H:%M:%S')} clip {name} running: first frame {err:+.0f}ms vs plan "
                         f"(post->start {(first - t0) / 1e6:.0f}ms); mean {mean:+.0f}ms over {len(self.phase_ms)} "
                         f"moves={self.moves} refused={self.refused}")
                return
            time.sleep(0.02)
        self.log(f"{time.strftime('%H:%M:%S')} clip {name}: runtime never reported it running")

    def stop(self):
        """Music ended or the mode changed: home once, then nothing."""
        self.home(0, None, why="dance stop")

    def home(self, hold_ns: int = 0, after=None, why: str = "dance start"):
        """Post the vendor `home` clip on the worker thread (never on the UDP loop) and refuse clip
        posts for `hold_ns` after it, so the arm reaches home through the runtime's own planned path
        before a beat clip blends from it. `after` (e.g. loading the clip library) runs on the worker
        once home is posted. While a post is in flight the request is kept and honoured by that
        post's thread; a second request during the home itself is folded into it (no second post),
        so nothing can be left pending once the worker is idle."""
        with self.lock:
            if self.inflight:
                self.pending_home = (hold_ns, after); return
            self.inflight, self.pending_home = True, None
        self.boundary_ns, self.drop_pending, self.playing = None, False, False
        self.build_until_ns = None
        self.not_before_ns = max(self.not_before_ns, time.monotonic_ns() + hold_ns)
        def go():
            r = self.post_fn("home")
            self.log(f"{time.strftime('%H:%M:%S')} {why} -> home: {r.get('status') or r.get('error')}")
            fns = [after]
            while fns:
                fn = fns.pop(0)
                if fn is not None:
                    try: fn()
                    except Exception as exc: self.log(f"{time.strftime('%H:%M:%S')} after home: {exc}")
                with self.lock:                               # fold a request made meanwhile into this home
                    pend, self.pending_home = self.pending_home, None
                    if pend is None and not fns:
                        self.inflight = False
                if pend is not None:
                    self.not_before_ns = max(self.not_before_ns, time.monotonic_ns() + pend[0])
                    fns.append(pend[1])
        threading.Thread(target=go, daemon=True).start()


def animation_names(r) -> set[str]:
    """The names in a GET /api/animations answer (a list of strings, or of {"name"|"stem": ...})."""
    names = set()
    try:
        items = r.get("animations") if isinstance(r, dict) else r
        for it in items or []:
            n = it if isinstance(it, str) else (it.get("name") or it.get("stem") or "")
            if n: names.add(n)
    except Exception:
        pass
    return names


class LiveClip:
    """A clip the generator wrote into the pack: what it is and what it is called."""
    __slots__ = ("tier", "variant", "bpm", "bold", "name", "md5", "meta")

    def __init__(self, tier, variant, bpm, bold, name, md5, meta):
        self.tier, self.variant, self.bpm, self.bold = tier, variant, float(bpm), float(bold)
        self.name, self.md5, self.meta = name, md5, meta

    def covers(self, tier, variant, bpm, bold, bpm_tol: float) -> bool:
        return self.tier == tier and self.variant == variant and self.bold == float(bold) and abs(self.bpm - bpm) <= bpm_tol


class LiveClips:
    """Generates the scheduler's next clip AHEAD of its post, on one worker thread (never the UDP loop):
    beat_clips.make_clip at the exact bpm and bold, validated with the lamp's own model, written atomically
    into the pack dir under a pooled name live_<k> (six names, k rotating; never the name playing or in
    flight, so the pack never grows), md5-verified, then confirmed in GET /api/animations within 300 ms.
    One job at a time; a newer request replaces a queued one. Any exception in the worker is logged once
    and disables generation for DISABLE_S (the library covers meanwhile); a failed clip (validation, write,
    listing) just leaves nothing ready, so that post falls back to the library."""
    POOL = 6
    PREP_NS = 700_000_000              # generate + write + sync + listing check, with margin, on the Pi 5
    LIST_NS = 300_000_000              # the runtime must list the name within this, or the clip is not used
    DISABLE_S = 60.0
    BPM_TOL = LIVE_BPM_TOL             # the tracker's PLL trims the period continuously: do not chase that (the
                                       # scheduler's own take-time band; tighter and every wobble is a job)

    def __init__(self, pack_dir: str = PACK_DIR, list_fn=None, make_fn=None, write_fn=None, model_fn=None,
                 log=print, enabled: bool = True):
        self.pack_dir, self.log = pack_dir, log
        self.list_fn = list_fn or (lambda: animation_names(lamp("/api/animations", timeout=0.3)))
        self.make_fn, self.write_fn, self.model_fn = make_fn, write_fn, model_fn
        self.model = None
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.wanted = None                 # (tier, variant, bpm, bold, deadline_ns): the newest request
        self.busy = None                   # the job the worker is on
        self.ready: LiveClip | None = None
        self.k = -1                        # the pool index used last
        self.playing, self.inflight = None, None
        self.disabled_until_ns, self.enabled = 0, enabled
        self.generated, self.used, self.failed, self.disables = 0, 0, 0, 0
        self.thread = None
        if enabled and not os.path.isdir(pack_dir):
            self.enabled = False
            self.log(f"live clips: pack dir {pack_dir} missing -> library only")
        if self.enabled:
            self.clean_pack_dir()

    # ---- pool ---------------------------------------------------------------------------
    @classmethod
    def pool_name(cls, k: int) -> str:
        return f"live_{k % cls.POOL}"

    @classmethod
    def is_pool_name(cls, name: str) -> bool:
        return name in {cls.pool_name(k) for k in range(cls.POOL)}

    def next_name(self) -> str:
        """The next pooled name that is not playing, in flight or ready (under the lock)."""
        reserved = {self.playing, self.inflight, self.ready.name if self.ready else None}
        for _ in range(self.POOL):
            self.k = (self.k + 1) % self.POOL
            name = self.pool_name(self.k)
            if name not in reserved:
                return name
        raise RuntimeError("live clip pool exhausted")     # cannot happen: at most three names are reserved

    def clean_pack_dir(self) -> int:
        """Startup: stale live_*.csv (and any temp of ours) from an earlier run go, so a clip of an old bpm
        or bold is never posted and the pack holds nothing half-written."""
        n = 0
        try:
            for f in os.listdir(self.pack_dir):
                if (f.startswith("live_") and f.endswith(".csv")) or (f.startswith(".live_") and f.endswith(".tmp")):
                    try:
                        os.unlink(os.path.join(self.pack_dir, f)); n += 1
                    except OSError as exc:
                        self.log(f"live clips: could not remove stale {f}: {exc}")
            if n and hasattr(os, "sync"):
                os.sync()
        except OSError as exc:
            self.log(f"live clips: cannot read {self.pack_dir}: {exc}")
        if n:
            self.log(f"live clips: removed {n} stale live_* file(s) from the pack")
        return n

    # ---- what the scheduler asks (UDP loop; cheap, one lock) ------------------------------
    def usable(self, now_ns: int | None = None) -> bool:
        if not self.enabled:
            return False
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        return now_ns >= self.disabled_until_ns

    def covers(self, tier, variant, bpm, bold) -> bool:
        """Is that clip ready, being made, or queued already?"""
        with self.lock:
            if self.ready is not None and self.ready.covers(tier, variant, bpm, bold, self.BPM_TOL):
                return True
            for job in (self.busy, self.wanted):
                if job is not None and job[0] == tier and job[1] == variant and job[3] == float(bold) \
                        and abs(job[2] - bpm) <= self.BPM_TOL:
                    return True
        return False

    def request(self, tier: str, variant: str, bpm: float, bold: float, deadline_ns: int):
        with self.cond:
            self.wanted = (tier, variant, float(bpm), float(bold), int(deadline_ns))
            if self.thread is None:
                self.thread = threading.Thread(target=self._worker, name="live-clips", daemon=True)
                self.thread.start()
            self.cond.notify()

    def cancel(self):
        """Discard queued/ready work after a selection change; an old busy result still must pass take()."""
        with self.cond:
            self.wanted, self.ready = None, None

    def peek(self) -> LiveClip | None:
        with self.lock:
            return self.ready

    def take(self, accept=None) -> LiveClip | None:
        """The ready clip if `accept(clip)` says so, now in flight (its name stays reserved until posted()
        says it is playing). Checked and taken under one lock: a regeneration cannot swap the clip
        between the two."""
        with self.lock:
            r = self.ready
            if r is None or (accept is not None and not accept(r)):
                return None
            self.ready, self.inflight = None, r.name
            return r

    def posted(self, name: str, ok: bool):
        with self.lock:
            if self.inflight == name:
                self.inflight = None
            if ok:
                self.playing = name
                self.used += 1

    def state(self) -> str:
        with self.lock:
            if not self.enabled: return "off"
            now = time.monotonic_ns()
            if now < self.disabled_until_ns: return f"disabled{(self.disabled_until_ns - now) // 1_000_000_000}s"
            if self.ready is not None: return f"ready({self.ready.name})"
            if self.busy is not None: return "busy"
            return "idle"

    # ---- the worker -----------------------------------------------------------------------
    def _worker(self):
        while True:
            with self.cond:
                while self.wanted is None:
                    self.cond.wait()
                job, self.wanted = self.wanted, None
                self.busy = job
            try:
                self._run(job)
            except Exception as exc:
                with self.lock:
                    self.disabled_until_ns = time.monotonic_ns() + int(self.DISABLE_S * 1e9)
                    self.ready, self.wanted, self.disables = None, None, self.disables + 1
                self.log(f"{time.strftime('%H:%M:%S')} live clips: {type(exc).__name__}: {exc} -> library only for "
                         f"{self.DISABLE_S:.0f}s")
            finally:
                with self.lock:
                    self.busy = None

    def _load(self):
        """beat_clips + the lamp's model, once, on the worker (an import and ~10 ms of file reading)."""
        if self.make_fn is None or self.write_fn is None:
            import beat_clips
            self.make_fn = self.make_fn or beat_clips.make_clip
            self.write_fn = self.write_fn or beat_clips.write_clip_atomic
            self.model_fn = self.model_fn or beat_clips.Validator.for_lamp
        if self.model is None and self.model_fn is not None:
            self.model = self.model_fn()

    def _run(self, job):
        tier, variant, bpm, bold, deadline = job
        t0 = time.monotonic_ns()
        self._load()
        rows, meta = self.make_fn(tier, variant, bpm, bold, model=self.model)
        if meta.get("ok") is not True:
            with self.lock: self.failed += 1
            self.log(f"{time.strftime('%H:%M:%S')} live clip {tier} {variant} {bpm:.1f}bpm bold {bold:.2f} FAILED "
                     f"validation: {'; '.join(meta.get('reasons') or ['no verdict'])} -> library")
            return
        with self.lock:
            name = self.next_name()
        path = os.path.join(self.pack_dir, name + ".csv")
        try:
            md5 = self.write_fn(path, rows)
        except Exception as exc:                              # a bad write is a failed clip, not a dead generator
            with self.lock: self.failed += 1
            self.log(f"{time.strftime('%H:%M:%S')} live clip {name}: write failed ({exc}) -> library")
            return
        t1 = time.monotonic_ns()
        listed = False
        while True:
            try:
                listed = name in self.list_fn()
            except Exception:
                listed = False
            if listed or time.monotonic_ns() - t1 > self.LIST_NS:
                break
            time.sleep(0.03)
        if not listed:
            with self.lock: self.failed += 1
            self.log(f"{time.strftime('%H:%M:%S')} live clip {name}: not listed by the runtime within "
                     f"{self.LIST_NS // 1_000_000}ms -> library")
            return
        clip = LiveClip(tier, variant, bpm, bold, name, md5, meta)
        with self.lock:
            self.ready = clip
            self.generated += 1
        late = time.monotonic_ns() - deadline
        self.log(f"{time.strftime('%H:%M:%S')} live clip {name} = {tier} {variant} {bpm:.1f}bpm bold {bold:.2f} "
                 f"x{meta.get('multiplier', 0):.2f} A={meta.get('amplitude', 0):.1f} {meta.get('frames')}f md5 {md5[:8]} "
                 f"in {(time.monotonic_ns() - t0) / 1e6:.0f}ms" + (f" ({late / 1e6:+.0f}ms past its post instant)" if late > 0 else ""))


def load_manifest_file(path: str | None) -> dict | None:
    """The staged clips' MANIFEST.json (beat_clips v2), or None when absent or unreadable: the
    scheduler then behaves as v4 did (unsuffixed names, no build tier) -- and says so, once, so the
    operator can tell from the log which library is in play. With no path the candidates are tried
    in order (DEFAULT_MANIFEST, then beat_clips/ next to this file)."""
    candidates = (path,) if path else MANIFEST_CANDIDATES
    why = []
    for cand in candidates:
        try:
            with open(cand) as f:
                m = json.load(f)
        except (OSError, ValueError) as exc:
            why.append(f"{cand}: {exc}")
            continue
        if not isinstance(m, dict) or not isinstance(m.get("clips"), list):
            why.append(f"{cand}: not a beat_clips manifest (no clips list)")
            continue
        n = sum(1 for c in m["clips"] if isinstance(c, dict) and c.get("variant") and not c.get("alias_of"))
        print(f"dance: manifest v{m.get('version')} at {cand}: {n} variant clips, tiers {m.get('tiers')}", flush=True)
        return m
    print("dance: no usable manifest (" + "; ".join(why) + "); v4 names only, no build tier", flush=True)
    return None


# ---- protocol helpers (pure) --------------------------------------------------------------------
def parse_event(d: bytes) -> dict | None:
    """EventPacket, 26 B: 03 | u32 seq | kind | flags | intensity | sharpness | u16 durMs | u16 freqHz
    | target | u64 masterTs | u32 leadUs."""
    if len(d) < 26 or d[0] != 3:
        return None
    seq, kind, flags, inten, sharp, dur_ms, freq, target, master, lead = struct.unpack_from("<IBBBBHHBQI", d, 1)
    return {"seq": seq, "kind": KINDS.get(kind, str(kind)), "flags": flags, "intensity": inten / 255.0,
            "sharpness": sharp / 255.0, "dur_ms": dur_ms, "freq": freq, "target": target, "master": master, "lead_us": lead}


def parse_control(d: bytes) -> dict:
    """Control JSON; returns {"lat": float|None, "session": int|None,
    "lamp": {"mode","lights","gen","bold","track"}|None}.
    The old conductor sends no lamp key (then the CLI flags stay in charge). `session` is the
    conductor's per-launch random id: a new one means its seq counters start again from 0. `bold` is
    the "Bolder moves" slider, clamped to 0..1; missing or unreadable -> None (keep the current value).
    `track` is what follow mode looks for ("face" | "phone"); missing or unknown -> None (keep the current)."""
    out = {"lat": None, "session": None, "lamp": None}
    try:
        j = json.loads(d[1:] if d and d[0] == 13 else d)
    except Exception:
        return out
    if not isinstance(j, dict):
        return out
    if "lat" in j:
        try: out["lat"] = float(j["lat"])
        except (TypeError, ValueError): pass
    if "session" in j:
        try: out["session"] = int(j["session"])
        except (TypeError, ValueError): pass
    l = j.get("lamp")
    if isinstance(l, dict):
        mode = l.get("mode")
        try: gen = int(l.get("gen", 0) or 0)
        except (TypeError, ValueError): gen = 0
        bold = None
        if "bold" in l:
            try:
                bold = clamp(float(l["bold"]))                  # NaN -> 0 by clamp(); strings that parse are fine
            except (TypeError, ValueError):
                bold = None
        track = l.get("track")
        out["lamp"] = {"mode": mode if mode in MODES else None,
                       "lights": bool(l["lights"]) if "lights" in l else None,
                       "gen": gen, "bold": bold, "track": track if track in TRACKS else None}
    return out


def hello_json(audio: bool) -> dict:
    """role:lamp lists us as a lamp on the new conductor; audio:true is REQUIRED to receive audio."""
    return {"t": "hi", "role": "lamp", "audio": bool(audio), "v": 1, "m": "lelamp", "name": "LeLamp",
            "app": "lelamp-show", "hap": False, "os": "Raspberry Pi OS", "route": "Light"}


def excitement(hits_2s: int, loud: float) -> float:
    return clamp(0.6 * hits_2s / 8.0 + 0.4 * loud)


def pi_temperature() -> float | None:
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return round(int(f.read().strip()) / 1000.0, 1)
    except Exception:
        return None


# ---- hardware-facing pieces ---------------------------------------------------------------------
class TargetFile:
    """follow.py --live writes ~/feelthemusic-lamp/target.json every cycle (follow.TargetReport): what it
    sees and how centred it is. Read here by mtime: at most one stat every `poll` seconds on the UDP
    loop, a read and a parse (a ~150 B file) only when it changed. `seen` is true only while the file
    is fresh (FRESH_S), so a follower that died does not keep claiming a target."""
    FRESH_S = 2.0
    EMPTY = {"kind": None, "seen": False, "center": 0.0, "aim_deg": None, "age_s": None}

    def __init__(self, path: str = TARGET_FILE, poll: float = 0.1):
        self.path, self.poll = path, poll
        self.mtime_ns, self.next_stat, self.body = None, float("-inf"), None

    def read(self, now: float, wall: float | None = None) -> dict:
        """{kind, seen, center, aim_deg, age_s}; `now` is monotonic (the poll throttle), `wall` time.time()."""
        if now >= self.next_stat:
            self.next_stat = now + self.poll
            try:
                st = os.stat(self.path)
            except OSError:
                self.mtime_ns, self.body = None, None
            else:
                if st.st_mtime_ns != self.mtime_ns:
                    self.mtime_ns = st.st_mtime_ns
                    try:
                        with open(self.path, "rb") as f:
                            self.body = json.loads(f.read())
                    except (OSError, ValueError):
                        self.body = None
        b = self.body
        if self.mtime_ns is None or not isinstance(b, dict):
            return dict(self.EMPTY)
        age = max(0.0, (time.time() if wall is None else wall) - self.mtime_ns / 1e9)
        try:
            center = clamp(float(b.get("center") or 0.0))
        except (TypeError, ValueError):
            center = 0.0
        try:
            aim = b.get("aim_deg")
            aim = None if aim is None or not math.isfinite(float(aim)) else round(float(aim), 1)
        except (TypeError, ValueError):
            aim = None
        kind = b.get("kind")
        return {"kind": kind if isinstance(kind, str) else None, "seen": bool(b.get("seen")) and age <= self.FRESH_S,
                "center": round(center, 3), "aim_deg": aim, "age_s": round(age, 2)}


class Panel(threading.Thread):
    """Renders at 50 fps. State is written by the network thread under a lock; this thread only reads
    it and talks to the LEDs. Direct NeoPixel if the runtime has released the panel, else HTTP."""
    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        self.model = ColourModel()
        self.dark = False
        self.audio_source = lambda: None
        self.frames, self.mode, self.px = 0, "none", None
        self.brightness = HW_BRIGHTNESS
        self.http_pending, self.http_sent = None, 0
        try:
            os.chdir("/home/lelamp/feelthemusic-lamp")          # Blinka's Pi 5 backend wants a writable cwd
            import board, neopixel
            self.px = neopixel.NeoPixel(getattr(board, PIN), PIXELS, brightness=HW_BRIGHTNESS,
                                        auto_write=False, pixel_order=neopixel.GRB)
            self.px.fill((0, 0, 0)); self.px.show()
            self.mode = "direct"
        except Exception as exc:
            self.mode = "http"
            print(f"panel: direct open failed ({exc}); falling back to the HTTP route", flush=True)
        atexit.register(self.off)

    def off(self):
        try:
            if self.px is not None:
                self.px.brightness = HW_BRIGHTNESS
                self.px.fill((0, 0, 0)); self.px.show()
            else:
                lamp("/api/light/solid", {"r": 0, "g": 0, "b": 0}, timeout=2)
        except Exception:
            pass

    def frame(self, now: float):
        with self.lock:
            self.model.set_audio(self.audio_source(), now)
            hue, sat, value, brightness = self.model.step(now)
            dark = self.dark
        return safe_output(hue, sat, value, brightness, dark)

    def run(self):
        period = 1 / 50
        http_thread = None
        if self.mode == "http":
            http_thread = threading.Thread(target=self._http_loop, daemon=True); http_thread.start()
        while True:
            t = time.monotonic()
            rgb, brightness = self.frame(t)
            if self.mode == "direct":
                try:
                    if abs(brightness - self.brightness) > 1e-3:
                        self.px.brightness = brightness; self.brightness = brightness
                    self.px.fill(rgb); self.px.show()
                except Exception as exc:
                    print(f"panel: show failed: {exc}", flush=True); time.sleep(0.5)
            else:
                with self.lock:
                    self.http_pending = rgb
            self.frames += 1
            time.sleep(max(0.0, period - (time.monotonic() - t)))

    def _http_loop(self):
        while True:
            with self.lock:
                rgb, self.http_pending = self.http_pending, None
            if rgb is None:
                time.sleep(0.02); continue
            lamp("/api/light/solid", {"r": rgb[0], "g": rgb[1], "b": rgb[2]}, timeout=2.5)
            self.http_sent += 1


class AudioPlayer:
    """Plays the conductor's Opus stream on the lamp's speaker, clock-locked to the shared clock when
    synced (jitter buffer of ~L otherwise). Silently disables itself on any decoder or device error.
    v4: every decoded packet is analysed (Chroma) at decode time, and that analysis is PUBLISHED as
    self.now when the packet is played, so the colour lines up with what the speaker is playing."""
    def __init__(self, depth_packets: int = 30):
        import av, numpy as np, sounddevice as sd
        self.np, self.av = np, av
        self.cc = av.CodecContext.create("libopus", "r")
        self.cc.sample_rate = 48000
        try: self.cc.layout = "mono"
        except Exception: pass
        self.buf: dict[int, "np.ndarray"] = {}
        self.ana: dict[int, tuple] = {}
        self.now: tuple | None = None
        try:
            self.chroma = Chroma()
        except Exception as exc:
            self.chroma = None
            print(f"audio: chroma analysis disabled ({exc})", flush=True)
        self.depth, self.next_seq, self.last16 = depth_packets, None, None
        self.codec, self.decoded, self.played, self.gaps, self.errors = None, 0, 0, 0, 0
        self.anchor, self.clock, self.late, self.aud_leads = None, None, 0, []
        self.stream = sd.OutputStream(samplerate=48000, channels=1, dtype="float32",
                                      blocksize=480, callback=self._cb)
        self.stream.start()

    def unwrap(self, s16: int) -> int:
        if self.last16 is None:
            self.last16 = s16; return s16
        d = (s16 - (self.last16 & 0xFFFF) + 0x8000) % 0x10000 - 0x8000
        self.last16 += d
        return self.last16

    def on_anchor(self, codec: int, seq_base: int = 0, pts: int = 0, frames: int = 480, sr: int = 48000):
        self.codec = codec
        self.anchor = (seq_base, pts, frames, sr)
        # The compact packets carry only 16 bits of sequence; the anchor carries the full 32-bit
        # base. Unwrap RELATIVE TO THAT (as the phone's SeqUnwrapper.reset(to:) does), or
        # (seq - seq_base) is off by tens of thousands and every due time is in the past.
        if self.last16 is None or abs(self.last16 - seq_base) > 0x4000:
            self.last16 = seq_base
            self.buf.clear(); self.ana.clear(); self.next_seq = None

    def due_ns(self, seq: int, offset_ns: int, lat_ms: float) -> int:
        seq_base, pts, frames, sr = self.anchor
        return int(pts + (seq - seq_base) * frames * 1_000_000_000 // sr + lat_ms * 1_000_000 - offset_ns)

    def on_compact(self, seq16: int, payload: bytes):
        if self.codec not in (None, 2):
            return
        try:
            seq = self.unwrap(seq16)
            frames = self.cc.decode(self.av.Packet(payload))
            if not frames:
                return
            chunks = []
            for f in frames:
                a = f.to_ndarray()
                a = a[0] if a.ndim == 2 else a
                if a.dtype.kind == "i":
                    a = a.astype("float32") / 32768.0
                chunks.append(a.astype("float32"))
            pcm = self.np.concatenate(chunks)
            # Analyse BEFORE the packet is offered to the callback: a packet arriving right at its
            # due instant is popped at once, and an analysis written after that pop would never be
            # pruned (the callback and the prune only drop ana keys together with buf keys).
            if self.chroma is not None:
                try: self.ana[seq] = self.chroma.feed(pcm)
                except Exception: self.chroma = None
            self.buf[seq] = pcm
            self.decoded += 1
            if len(self.buf) > self.depth * 2:                # never let latency grow without bound
                for k in sorted(self.buf)[: len(self.buf) - self.depth]:
                    self.buf.pop(k, None); self.ana.pop(k, None)
                self.next_seq = min(self.buf)
        except Exception:
            self.errors += 1

    def _cb(self, out, frames, t, status):
        out[:] = 0
        if self.anchor is not None and self.clock is not None and self.clock()[0] is not None:
            # Clock-locked: play the packet whose due instant matches when THIS block reaches the DAC.
            offset_ns, lat_ms = self.clock()
            try: dac_lead = t.outputBufferDacTime - t.currentTime
            except Exception: dac_lead = 0.0
            play_at = time.monotonic_ns() + int(dac_lead * 1e9)
            while self.buf:
                seq = min(self.buf); lead = self.due_ns(seq, offset_ns, lat_ms) - play_at
                if lead < -30_000_000:                          # already late: drop, do not smear
                    self.buf.pop(seq); self.ana.pop(seq, None); self.late += 1; continue
                if lead > 8_000_000:                            # not yet: silence this block
                    return
                pcm = self.buf.pop(seq); n = min(len(pcm), frames)
                out[:n, 0] = pcm[:n]; self.played += 1
                a = self.ana.pop(seq, None)
                if a is not None: self.now = a
                self.aud_leads = (self.aud_leads + [lead / 1e6])[-100:]
                return
            self.gaps += 1
            return
        if self.next_seq is None:
            if len(self.buf) >= self.depth:
                self.next_seq = min(self.buf)
            return
        pcm = self.buf.pop(self.next_seq, None)
        a = self.ana.pop(self.next_seq, None)
        self.next_seq += 1
        if pcm is None:
            self.gaps += 1
            return
        n = min(len(pcm), frames)
        out[:n, 0] = pcm[:n]
        self.played += 1
        if a is not None: self.now = a


class Show:
    FOLLOW_SETTLE_S, HOME_HOLD_S = 1.0, 5.0   # entering dance: kill follow.py, wait, home, then clips after 5 s

    def __init__(self, conductor: str, mode: str, audio: bool, vendor_clips: bool = False,
                 start_latency_ms: float = 350.0, lights: bool = True, manifest: str | None = None,
                 live_clips: bool = True, bold: float = DEFAULT_BOLD, pack_dir: str = PACK_DIR, track: str = "face",
                 pattern: str = "auto"):
        if vendor_clips and pattern != "auto":
            raise ValueError("dance pattern selection requires beat-locked clips, not --vendor-clips")
        self.dst = (conductor, 47300)
        self.cli_mode, self.cli_lights, self.vendor_clips = mode, lights, vendor_clips
        self.mode, self.lights, self.gen = mode, lights, None
        self.track = track if track in TRACKS else "face"     # what follow.py looks for (Control lamp.track)
        self.targets, self.target = TargetFile(), dict(TargetFile.EMPTY)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
        self.sock.setblocking(False)
        self.panel = Panel(); self.panel.start()
        self.audio = None
        if audio:
            try:
                self.audio = AudioPlayer()
                self.audio.clock = lambda: (self.offset_ns, self.lat_ms)
                self.panel.audio_source = lambda: self.audio.now
                print("audio: opus decoder + reSpeaker output ready (clock-locked when synced)", flush=True)
            except Exception as exc:
                print(f"audio: disabled ({exc})", flush=True)
        self.limiter = FlashLimiter()
        self.tracker = BeatTracker()
        live = LiveClips(pack_dir) if live_clips and not vendor_clips else None
        self.scheduler = ClipScheduler(start_latency_ns=int(start_latency_ms * 1e6), manifest=load_manifest_file(manifest),
                                       live=live, bold=bold, pattern=pattern)
        self.follow_proc, self.follow_lock, self.follow_atexit = None, threading.Lock(), False
        # what the running (or queued) follower looks for; None = unknown after a failed restart, so the
        # next Control with a lamp key restarts it. follow_gen numbers start requests: a start that finds
        # a newer one queued gives up (the newer one spawns the newest track).
        self.follow_track, self.follow_gen = None, 0
        self.seq, self.index, self.session = 0, None, None
        self.seen_event, self.seen_bass = set(), set()
        self.sent_at, self.rtts, self.min_rtt = {}, [], float("inf")
        self.last_sync_rx, self.rediscover_at = time.monotonic(), time.monotonic() + 20.0
        # Shared clock. offset_ns = conductor host ns - lamp monotonic ns, from the lowest-rtt sync
        # sample (the same NTP arithmetic the phones use). Everything the conductor stamps is then
        # fired at its due instant on THIS clock instead of on arrival: hits arrive ~267 ms early.
        self.offset_ns, self.lat_ms = None, 300.0
        self.pending_events, self.pending_bass = [], []        # heaps of (due_lamp_ns, ...)
        self.ev_leads, self.lead_ok = [], 0
        self.bass_raw, self.loud_since, self.quiet_since = 0.0, None, time.monotonic()
        # "Is music playing?" is answered by EVENTS, not the bass envelope. The conductor's envelope
        # is normalised, so in silence the noise floor is scaled up to 1.0 and it never reads quiet;
        # the analyzer only emits KICK/SNARE when there is a real signal. Three hits in three
        # seconds means music; none for four seconds means the song ended.
        self.hits: list[float] = []
        self.music, self.excite = False, 0.0
        self.clip_until, self.clip_i = 0.0, 0
        self.counts = {"events": 0, "bass": 0, "control": 0, "flashes": 0, "clips": 0}

    # ---- wire ---------------------------------------------------------------------------
    def send(self, b: bytes):
        self.sock.sendto(b, self.dst)

    def hello(self):
        self.send(b"\x04" + json.dumps(hello_json(self.audio is not None), sort_keys=True, separators=(",", ":")).encode())

    def sync(self):
        self.seq = (self.seq + 1) & 0xFFFF
        self.sent_at[self.seq] = time.monotonic_ns()
        if len(self.sent_at) > 64:
            for k in sorted(self.sent_at)[:-32]: self.sent_at.pop(k, None)
        self.send(b"\x01" + struct.pack("<H", self.seq))

    def send_cs(self):
        if not self.rtts: return
        r = sorted(self.rtts); p = lambda q: r[min(len(r) - 1, int(q * len(r)))]
        j = {"t": "cs", "rtt50": round(p(0.5), 1), "rtt95": round(p(0.95), 1), "min": round(self.min_rtt, 1),
             "off": 0.0, "bnd": round(self.min_rtt / 2, 2), "n": len(self.rtts), "skew": 0.0, "rej": 0,
             "ev": self.counts["events"], "late": 0, "th": "ok"}
        if self.index is not None: j["i"] = self.index
        self.send(b"\x04" + json.dumps(j, sort_keys=True, separators=(",", ":")).encode())

    def lamp_status(self) -> dict:
        state = ("off" if self.mode == "off" else "syncing" if self.offset_ns is None
                 else "music" if self.music else "idle")
        s = self.scheduler
        moves = s.moves + (self.counts["clips"] if self.vendor_clips else 0)
        self.target = self.targets.read(time.monotonic())
        return {"t": "lamp", "state": state, "mode": self.mode, "locked": self.tracker.locked(time.monotonic_ns()),
                "piC": pi_temperature(), "sdk": SDK, "moves": moves, "refused": s.refused, "lights": self.lights,
                "bpm": round(self.tracker.bpm, 1), "target": self.target}

    def send_lamp(self):
        self.send(b"\x04" + json.dumps(self.lamp_status(), sort_keys=True, separators=(",", ":")).encode())

    def lamp_period(self) -> float:
        """Seconds until the next `lamp` telemetry: 5 Hz while the follower sees its target (the
        conductor drives the phone's vibration from it live), 1 Hz otherwise."""
        return LAMP_PERIOD_SEEN_S if self.target.get("seen") else LAMP_PERIOD_S

    def handle(self, d: bytes, now: float):
        t = d[0]
        if t == 8 and len(d) > 5:
            if self.audio: self.audio.on_compact(struct.unpack_from("<H", d, 1)[0], d[5:])
        elif t == 12 and len(d) >= 15:
            seq = struct.unpack_from("<I", d, 1)[0]
            if seq in self.seen_bass: return
            self.seen_bass.add(seq)
            n = d[14]
            start_ts, step_ms = struct.unpack_from("<Q", d, 5)[0], d[13]
            if self.offset_ns is None:
                if n: self.bass_raw = d[15 + n - 1] / 255.0
            else:
                for k in range(n):
                    heapq.heappush(self.pending_bass, (start_ts + k * step_ms * 1_000_000 - self.offset_ns, d[15 + k] / 255.0))
            self.counts["bass"] += 1
        elif t == 3 and len(d) >= 26:
            e = parse_event(d)
            if e is None or e["seq"] in self.seen_event: return
            self.seen_event.add(e["seq"])
            self.counts["events"] += 1
            kind = e["kind"]
            if kind in ("KICK", "SNARE"):
                self.hits = [t for t in self.hits if now - t < 3.0] + [now]
            if self.offset_ns is None:
                self.on_event(kind, e["intensity"], now, e["dur_ms"])
            else:
                due = e["master"] - self.offset_ns
                self.ev_leads = (self.ev_leads + [(due - time.monotonic_ns()) / 1e6])[-50:]
                if kind == "KICK":
                    self.tracker.feed(due)                     # the tracker sees the beat before it happens
                heapq.heappush(self.pending_events, (due, kind, e["intensity"], e["dur_ms"]))
        elif t == 2 and len(d) >= 19:
            seq, t1, t2 = struct.unpack_from("<HQQ", d, 1)
            t0 = self.sent_at.pop(seq, None)
            if t0 is not None:
                t3 = time.monotonic_ns(); rtt = ((t3 - t0) - (t2 - t1)) / 1e6
                if 0 <= rtt < 2000:
                    if rtt <= self.min_rtt:                  # best sample so far: trust its offset
                        self.offset_ns = ((t1 - t0) + (t2 - t3)) // 2
                    self.rtts = (self.rtts + [rtt])[-40:]; self.min_rtt = min(self.min_rtt, rtt)
                    self.last_sync_rx = time.monotonic()
        elif t == 9 and len(d) >= 20:
            seq_base, pts, frames, sr, codec = struct.unpack_from("<IQHIB", d, 1)
            if self.audio: self.audio.on_anchor(codec, seq_base, pts, frames, sr)
        elif t == 5 and len(d) >= 2:
            self.index = d[1]
            if len(d) >= 6: self.on_session(struct.unpack_from("<I", d, 2)[0])
        elif t == 13:
            self.counts["control"] += 1
            c = parse_control(d)
            if c["session"] is not None: self.on_session(c["session"])
            if c["lat"] is not None: self.lat_ms = c["lat"]
            if c["lamp"] is not None: self.on_lamp_control(c["lamp"], now)
        for s in (self.seen_event, self.seen_bass):
            if len(s) > 4000:
                for k in sorted(s)[:2000]: s.discard(k)

    # ---- modes --------------------------------------------------------------------------
    def on_session(self, session: int):
        """The conductor restarted (Control and Assign carry a per-launch random session id): its
        event and bass seq counters start again from 0, so the old seqs must not be treated as
        duplicates or every hit of the new run is dropped until it passes the old high-water mark.
        Queued work stamped by the old run goes too; the shared clock survives (the host clock is
        mach_absolute_time based, so the offset stays valid) and is simply re-measured."""
        if session == self.session:
            return
        first, self.session = self.session is None, session
        if first:
            return
        print(f"{time.strftime('%H:%M:%S')} conductor session changed -> {session}: forgetting its old seqs", flush=True)
        self.seen_event.clear(); self.seen_bass.clear()
        self.pending_events.clear(); self.pending_bass.clear()
        self.tracker.__init__()
        self.scheduler.reset()
        self.rtts, self.min_rtt = [], float("inf")              # the next lowest-rtt sample re-anchors the offset

    def on_lamp_control(self, l: dict, now: float):
        """The dashboard's word wins over the CLI whenever the conductor sends a lamp key."""
        mode = l["mode"] or self.mode
        lights = self.lights if l["lights"] is None else l["lights"]
        track = l.get("track")
        retrack = track in TRACKS and track != self.track
        if retrack:
            print(f"{time.strftime('%H:%M:%S')} track {self.track} -> {track}", flush=True)
            self.track = track                                   # before set_mode: a follower it starts looks for it
        changed = self.set_mode(mode, now)   # before the gen reset: leaving dance must see what is playing
        if self.mode == "follow" and not changed and self.follow_track != self.track:
            # already following, on another target (or on an unknown one after a failed restart): restart
            print(f"{time.strftime('%H:%M:%S')} follow: restarting on {self.track}"
                  + ("" if retrack else " (retry)"), flush=True)
            self.follow_track = self.track                       # the queued start is for it: no second restart
            self.stop_follow(then=lambda: self.start_follow(wait_s=RESTART_WAIT_S), exit_s=FOLLOW_EXIT_S)
        if l["gen"] != self.gen:
            self.gen = l["gen"]
            self.scheduler.reset()                               # drop work queued for the old mode
            if not (changed and self.mode == "dance"):           # but keep the hold behind the home pre-move
                self.clip_until = now
        if lights != self.lights:
            self.lights = lights
            print(f"{time.strftime('%H:%M:%S')} lights -> {'on' if lights else 'off'}", flush=True)
        if l.get("bold") is not None:
            self.scheduler.set_bold(l["bold"])                   # logs on change; missing -> keep the current value
        self.apply_lights()

    def apply_lights(self):
        with self.panel.lock:
            self.panel.dark = (not self.lights) or self.mode == "off"

    def set_mode(self, mode: str, now: float) -> bool:
        """Returns whether the mode changed."""
        if mode not in MODES or mode == self.mode:
            return False
        old, self.mode = self.mode, mode
        print(f"{time.strftime('%H:%M:%S')} mode {old} -> {mode}", flush=True)
        if old == "follow":
            self.stop_follow()
        if old == "dance":
            s = self.scheduler
            # A post in flight has neither `playing` nor (after a gen reset) a boundary yet: it must
            # still be homed, or the clip plays out in light/off mode.
            if s.playing or s.boundary_ns is not None or s.inflight: s.stop()
            else: s.reset()
            self.clip_until = now
        if mode == "follow":
            self.start_follow()
        elif mode == "dance":
            self.enter_dance(now)
        self.apply_lights()
        return True

    def enter_dance(self, now: float):
        """The safety pre-move, from every path into dance (CLI start and the dashboard alike). A
        clip blends over ~2 s from wherever the arm is parked, and the envelope only governs the
        clip's own frames: after a reboot the arm rests at base_pitch ~-91 (from the sleep pose,
        elbow -98), so the first blend-in crossed the flip region. Start from the vendor's `home`
        pose instead, through the runtime's own planned path, and post no clip for 5 s. A stray
        follow.py (left by mode.sh or a crash) would drive the arm at the same time: kill it first.
        The scheduler is held from THIS thread at once, so no beat clip slips in before home."""
        hold = self.FOLLOW_SETTLE_S + self.HOME_HOLD_S
        self.scheduler.not_before_ns = max(self.scheduler.not_before_ns, time.monotonic_ns() + int(hold * 1e9))
        self.clip_until = max(self.clip_until, now + hold)
        self.stop_follow(wait_s=self.FOLLOW_SETTLE_S, then=self._enter_dance_home)

    def _enter_dance_home(self):
        if self.mode != "dance":
            return
        if self.vendor_clips:
            self.clip_until = time.monotonic() + self.HOME_HOLD_S
            self.scheduler.home(0, None, why="dance start")
        else:
            self.scheduler.home(int(self.HOME_HOLD_S * 1e9),
                                self.load_clip_library if self.scheduler.available is None else None)

    def start_follow(self, wait_s: float = 0.0):
        """Spawns ./run_face.sh (setsid, log FOLLOW_LOG, FOLLOW_TARGET = the track at spawn time) on a
        worker thread: process spawns take tens of ms on the Pi and must not stall the UDP loop. With
        `wait_s` (a restart, after stop_follow) a follower still on its way out gets that long to
        disappear, then SIGKILL and FOLLOW_KILL_GRACE_S more; one that is still there after that is
        given up on loudly and follow_track is forgotten (the next Control with a lamp key retries).
        Without `wait_s` a follower already running (mode.sh, a hand start) is left alone. A start
        that finds a newer start queued (follow_gen moved on) gives up: that one spawns the newest
        track, so two quick track changes never leave the lamp on the older one."""
        here = os.path.dirname(os.path.abspath(__file__))
        with self.follow_lock:
            self.follow_gen += 1
            gen = self.follow_gen
            self.follow_track = self.track
        def superseded() -> bool:
            if gen == self.follow_gen: return False
            print(f"follow: start #{gen} superseded by #{self.follow_gen}", flush=True); return True
        def go():
            try:
                deadline, killed = time.monotonic() + wait_s, False
                while True:
                    if superseded(): return
                    if subprocess.run(["pgrep", "-f", "[f]ollow.py"], capture_output=True).returncode != 0:
                        break
                    with self.follow_lock:
                        ours, on = self.follow_proc, self.follow_track
                    if ours is not None and ours.poll() is None:  # a sibling start spawned it, on the newest track
                        print(f"follow: already running on {on} (pid {ours.pid})", flush=True); return
                    if time.monotonic() >= deadline:
                        if wait_s <= 0:
                            print("follow: already running", flush=True); return
                        if not killed:                              # the one we stopped will not leave: SIGKILL
                            print(f"follow: the old follower is still running {wait_s:g} s after the stop: SIGKILL", flush=True)
                            subprocess.run(["pkill", "-KILL", "-f", "[f]ollow.py"], capture_output=True)
                            killed, deadline = True, time.monotonic() + FOLLOW_KILL_GRACE_S
                            continue
                        print(f"follow: COULD NOT RESTART on {self.track}: the old follower survived SIGKILL; "
                              "the next Control retries", flush=True)
                        with self.follow_lock:
                            if gen == self.follow_gen: self.follow_track = None
                        return
                    time.sleep(FOLLOW_POLL_S)
                if superseded(): return
                track = self.track                                   # the newest wish, not the one at the call
                log = open(FOLLOW_LOG, "ab")
                proc = subprocess.Popen(["setsid", os.path.join(here, "run_face.sh")], cwd=here,
                                        env={**os.environ, "FOLLOW_TARGET": track},
                                        stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                with self.follow_lock:
                    self.follow_proc, self.follow_track = proc, track
                    if not self.follow_atexit:                # our tracker dies with us (systemctl stop, a crash)
                        self.follow_atexit = True
                        atexit.register(self.stop_follow_at_exit)
                print(f"follow: started run_face.sh --target {track} (pid {proc.pid}, log {FOLLOW_LOG})", flush=True)
            except Exception as exc:
                print(f"follow: could not start ({exc})", flush=True)
        threading.Thread(target=go, daemon=True).start()

    def stop_follow_at_exit(self):
        if self.follow_proc is not None:
            self.stop_follow(sync=True)

    def stop_follow(self, wait_s: float = 0.0, then=None, sync: bool = False, exit_s: float = 2.0):
        """pkill -f '[f]ollow.py' on a worker thread ('[f]ollow.py' matches run_face.sh's exec'd
        `python follow.py ...` and neither this process nor the pkill itself), reap our own child so
        it is not left a zombie (up to `exit_s`), then optionally wait and run `then` (the dance
        pre-move, or the restart's start). `sync` runs it inline (atexit: a daemon thread would not
        get to finish)."""
        with self.follow_lock:
            proc, self.follow_proc = self.follow_proc, None
        def go():
            try:
                subprocess.run(["pkill", "-f", "[f]ollow.py"], capture_output=True)
            except Exception as exc:
                print(f"follow: pkill failed ({exc})", flush=True)
            if proc is not None:
                try: proc.wait(timeout=exit_s)
                except Exception: pass
            print("follow: stopped", flush=True)
            if then is not None:
                if wait_s > 0: time.sleep(wait_s)
                try: then()
                except Exception as exc: print(f"follow: after stop: {exc}", flush=True)
        if sync: go()
        else: threading.Thread(target=go, daemon=True).start()

    # ---- performance --------------------------------------------------------------------
    def on_event(self, kind: str, intensity: float, now: float, dur_ms: int = 0):
        p = self.panel
        with p.lock:
            p.model.nudge_hue(kind, intensity, now)
            if kind == "BUILD":
                p.model.start_build(now, dur_ms / 1000.0)
                self.scheduler.note_build(int(now * 1e9), dur_ms)
            target = flash_target_for(kind, intensity, self.excite)
            ok = target > 0 and p.model.request_flash(target, self.limiter, now)
            if ok:
                self.counts["flashes"] += 1
            if kind == "DROP":
                # The white-hot burst is the biggest flash the show makes, so it is gated like any
                # other: a refused DROP keeps its 0.15 pulse and still schedules the drop clip.
                if ok: p.model.start_drop(now)
                self.scheduler.drop_pending = True
        if self.mode == "dance" and self.vendor_clips and kind in ("BUILD", "DROP") and now >= self.clip_until:
            self.play_clip(kind, now)

    def play_clip(self, why: str, now: float):
        name, secs = CLIPS[self.clip_i % len(CLIPS)]; self.clip_i += 1
        # Reserve the slot BEFORE the HTTP round trip (~0.6 s). The tick runs at 20 Hz, so without
        # this a second tick fired play_clip again inside that window and restarted the clip -- a
        # visible snap at every trigger ("started" logged twice at the same second).
        self.clip_until = now + 3.0
        r = lamp("/api/animations/play", {"name": name})
        ok = r.get("status") == "started"
        self.clip_until = now + (secs + 0.8 if ok else 3.0)
        self.counts["clips"] += 1
        if not ok: self.scheduler.refused += 1
        print(f"{time.strftime('%H:%M:%S')} {why} -> clip {name}: {r.get('status') or r.get('error')}", flush=True)

    def tick(self, now: float):
        # Smooth bass for the floor colour (the flashes carry the beat, this carries the energy).
        self.hits = [t for t in self.hits if now - t < 4.0]
        recent3 = sum(1 for t in self.hits if now - t < 3.0)
        if not self.music and recent3 >= 3:
            self.music = True; print(f"{time.strftime('%H:%M:%S')} music: on", flush=True)
        elif self.music and not self.hits:
            self.music = False; print(f"{time.strftime('%H:%M:%S')} music: off", flush=True)
        music = self.music
        loud = self.audio.now[2] if (self.audio and self.audio.now) else 0.0
        self.excite = excitement(sum(1 for t in self.hits if now - t < 2.0), loud)
        with self.panel.lock:
            # When no music is detected the colour floor decays to dark instead of sitting at the
            # envelope's phantom 1.0.
            target = self.bass_raw if music else 0.0
            m = self.panel.model
            m.bass += 0.35 * (target - m.bass)
            m.excite, m.music = self.excite, music
        if music:
            self.loud_since = self.loud_since or now; self.quiet_since = None
        else:
            self.quiet_since = self.quiet_since or now; self.loud_since = None
        if self.mode == "dance" and self.vendor_clips:
            # v3.1: while the music has been loud for 2 s, keep a vendor clip going. Stop when quiet.
            if self.loud_since and now - self.loud_since > 2.0 and now >= self.clip_until:
                self.play_clip("music", now)

    def schedule(self, now_ns: int):
        """Runs every loop pass (~3 ms), so a post lands within a few ms of its instant."""
        if self.mode == "dance" and not self.vendor_clips:
            self.scheduler.tick(now_ns, self.music and self.offset_ns is not None, self.tracker, self.excite)

    def load_clip_library(self):
        names = animation_names(lamp("/api/animations"))
        beat = sorted(n for n in names if n.startswith("beat_"))
        self.scheduler.available = names if names else None
        print(f"dance: {len(beat)} beat clips in the runtime's library" + (f" ({beat[0]} .. {beat[-1]})" if beat else
              " -- none; the scheduler will post by name and count refusals"), flush=True)

    # ---- loop ---------------------------------------------------------------------------
    def rediscover(self, now: float):
        """No SyncResp for 15 s: the conductor may have moved (hotspot lease, laptop swap, restart on
        another Mac). One short mDNS lookup (ftm_discover); switch and re-hello if the answer differs.
        Discovery blocks this thread for up to ~1.5 s, which costs nothing while nothing is arriving."""
        self.rediscover_at = now + 20.0
        try:
            import ftm_discover
            found = ftm_discover.discover(timeout=0.7, attempts=2)
        except Exception as exc:
            print(f"{time.strftime('%H:%M:%S')} discovery failed: {exc}", flush=True); found = None
        if found and (found[0], int(found[1])) != self.dst:
            print(f"{time.strftime('%H:%M:%S')} conductor moved: {self.dst[0]}:{self.dst[1]} -> {found[0]}:{found[1]}", flush=True)
            self.dst = (found[0], int(found[1])); self.hello(); self.sync()

    def run(self):
        print(f"panel: {self.panel.mode}", flush=True)
        self.hello(); self.sync()
        print(f"joined {self.dst[0]}:{self.dst[1]} as LeLamp, mode={self.mode}, lights={'on' if self.lights else 'off'}, "
              f"audio={'on' if self.audio else 'off'}, clips={'vendor' if self.vendor_clips else 'beat'}", flush=True)
        self.apply_lights()
        if self.mode == "dance":
            self.enter_dance(time.monotonic())                  # home first, clips after 5 s (see enter_dance)
        elif self.mode == "follow":
            self.start_follow()
        nxt = {"sync": 0.0, "hello": time.monotonic() + 10, "tick": 0.0, "report": time.monotonic() + 5, "lamp": time.monotonic() + 1}
        while True:
            now = time.monotonic()
            if now >= nxt["sync"]:  self.sync(); self.send_cs(); nxt["sync"] = now + 2.0
            if now >= nxt["hello"]: self.hello(); nxt["hello"] = now + 10.0
            if now - self.last_sync_rx > 15.0 and now >= self.rediscover_at: self.rediscover(now)
            if now >= nxt["lamp"]:  self.send_lamp(); nxt["lamp"] = now + self.lamp_period()
            drained = 0
            while drained < 512:
                try: d, _ = self.sock.recvfrom(4096)
                except (BlockingIOError, InterruptedError): break
                if not d: break
                self.handle(d, now); drained += 1
            if drained == 0: time.sleep(0.003)
            now_ns = time.monotonic_ns()
            while self.pending_events and self.pending_events[0][0] <= now_ns:
                _, kind, inten, dur = heapq.heappop(self.pending_events); self.on_event(kind, inten, now, dur); self.lead_ok += 1
            newest = None
            while self.pending_bass and self.pending_bass[0][0] <= now_ns:
                newest = heapq.heappop(self.pending_bass)[1]
            if newest is not None: self.bass_raw = newest
            if len(self.pending_bass) > 2000: self.pending_bass = self.pending_bass[-1000:]; heapq.heapify(self.pending_bass)
            if now >= nxt["tick"]:  self.tick(now); nxt["tick"] = now + 0.05
            self.schedule(now_ns)
            if now >= nxt["report"]:
                c = self.counts; a = self.audio; m = self.panel.model
                extra = (f" audio dec={a.decoded} play={a.played} gaps={a.gaps} late={a.late} "
                         f"aud_lead={(sum(a.aud_leads)/len(a.aud_leads)) if a.aud_leads else 0:+.0f}ms") if a else ""
                ev_lead = (sum(self.ev_leads) / len(self.ev_leads)) if self.ev_leads else 0.0
                extra += f" off={'?' if self.offset_ns is None else round(self.offset_ns/1e6)}ms L={self.lat_ms:.0f} ev_lead={ev_lead:+.0f}ms fired={self.lead_ok}"
                note = NOTE_NAMES[m.note] if 0 <= m.note < 12 else "-"
                s = self.scheduler
                phase = (sum(s.phase_ms) / len(s.phase_ms)) if s.phase_ms else 0.0
                lv = s.live
                live = "off" if lv is None else f"{lv.state()} gen={lv.generated} used={lv.used} fail={lv.failed}"
                print(f"{time.strftime('%H:%M:%S')} peer#{self.index} {self.mode} ev={c['events']} bass={c['bass']} "
                      f"flashes={c['flashes']} clips={c['clips'] + s.moves} level={m.bass:.2f} "
                      f"note={note} conf={m.conf:.2f} loud={m.loud:.2f} excite={self.excite:.2f} "
                      f"bpm={self.tracker.bpm:.1f}{'*' if self.tracker.locked(now_ns) else ''} beat_err={self.tracker.phase_error_ms():+.0f}ms "
                      f"clip_phase={phase:+.0f}ms bold={s.bold:.2f} live={live} fps={self.panel.frames / 5:.0f} "
                      f"rtt={self.min_rtt if self.rtts else -1:.1f}ms{extra}", flush=True)
                self.panel.frames = 0; nxt["report"] = now + 5.0


def burst_test():
    """Five white bursts at PEAK_BRIGHTNESS (300 ms on, 700 ms off) while sampling the 5 V rail through
    vcgencmd, to check the rail holds before PEAK_BRIGHTNESS is trusted in a show. Pi only."""
    import re
    try:
        os.chdir("/home/lelamp/feelthemusic-lamp")
        import board, neopixel
    except Exception as exc:
        print(f"burst-test: needs the Pi's board/neopixel ({exc})"); return 1
    px = neopixel.NeoPixel(getattr(board, PIN), PIXELS, brightness=HW_BRIGHTNESS, auto_write=False, pixel_order=neopixel.GRB)
    atexit.register(lambda: (px.fill((0, 0, 0)), px.show()))
    samples: list[float] = []
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            try:
                out = subprocess.run(["vcgencmd", "pmic_read_adc", "EXT5V_V"], capture_output=True, text=True, timeout=1).stdout
                m = re.search(r"=\s*([0-9.]+)V", out)
                if m: samples.append(float(m.group(1)))
            except Exception:
                pass
            time.sleep(0.05)
    th = threading.Thread(target=sample, daemon=True); th.start()
    peak = clamp(PEAK_BRIGHTNESS, 0.0, PEAK_BRIGHTNESS)
    for i in range(5):
        px.brightness = peak; px.fill((255, 255, 255)); px.show(); time.sleep(0.3)
        px.brightness = HW_BRIGHTNESS; px.fill((0, 0, 0)); px.show(); time.sleep(0.7)
    stop.set(); th.join(timeout=1)
    if samples:
        print(f"burst-test: EXT5V_V min {min(samples):.3f} V mean {sum(samples) / len(samples):.3f} V over {len(samples)} samples "
              f"(brightness {peak}, {PIXELS} px white x5, 300 ms)")
    else:
        print("burst-test: no rail samples (vcgencmd pmic_read_adc unavailable?)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--conductor")
    ap.add_argument("--dance", action="store_true", help="start in dance mode (same as --mode dance)")
    ap.add_argument("--mode", choices=MODES, help="starting mode; the new conductor's dashboard can change it")
    ap.add_argument("--audio", action="store_true")
    ap.add_argument("--no-lights", action="store_true")
    ap.add_argument("--vendor-clips", action="store_true", help="v3.1's vendor clips instead of the beat-locked ones")
    ap.add_argument("--start-latency-ms", type=float, default=350.0, help="POST -> first frame on the servos")
    ap.add_argument("--manifest", default=None, help="beat_clips MANIFEST.json with the clip variants (default: "
                    + ", then ".join(MANIFEST_CANDIDATES) + ")")
    ap.add_argument("--bold", type=float, default=DEFAULT_BOLD, help="\"Bolder moves\" 0..1 until the dashboard sends lamp.bold")
    ap.add_argument("--pattern", choices=DANCE_PATTERNS, default="auto",
                    help="beat-locked choreography: auto rotates with music; other choices keep one pattern")
    ap.add_argument("--no-live-clips", action="store_true",
                    help="pre-generated library only; its full-amplitude clips require --bold 1")
    ap.add_argument("--pack-dir", default=PACK_DIR, help="the runtime's animation pack, where live clips are written")
    ap.add_argument("--track", choices=TRACKS, default="face",
                    help="what follow mode looks for until the dashboard sends lamp.track (default face)")
    ap.add_argument("--burst-test", action="store_true")
    a = ap.parse_args()
    if a.vendor_clips and a.pattern != "auto":
        ap.error("--pattern requires beat-locked clips, not --vendor-clips")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    if a.burst_test:
        sys.exit(burst_test())
    if not a.conductor:
        ap.error("--conductor is required")
    mode = a.mode or ("dance" if a.dance else "light")
    try:
        Show(a.conductor, mode, a.audio, vendor_clips=a.vendor_clips, start_latency_ms=a.start_latency_ms,
             lights=not a.no_lights, manifest=a.manifest, live_clips=not a.no_live_clips, bold=a.bold,
             pack_dir=a.pack_dir, track=a.track, pattern=a.pattern).run()
    except KeyboardInterrupt:
        sys.exit(0)
