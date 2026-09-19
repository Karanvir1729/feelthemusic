"""Music -> haptics and lamp light on one presentation clock, for the robot-free twin.

  synthetic_song / from_runlog / from_analysis   event sources: a Song (MusicEvents, bass envelope, sections)
  LightDesign                                    what an IDEAL 93-pixel panel would show, as 60 Hz LightFrames
  apply_flash_limit / flash_report               WCAG 2.3.1 flash safety on the light actually emitted
  SdkLamp                                        what the REAL lamp shows for a list of light.glow calls
  reactive_calls / BestSdkDesign                 two ways to drive it: the ideal design sent literally
                                                 (fails, as the spec predicts) and the spec's Option A + B
  PhoneLane / TitanLane                          when each haptic output is FELT
  sync_report                                    light onset vs felt haptic, colour switches, flashes
  run_show                                       all of the above for one song

Time is float seconds on the twin's clock (twin/contract.py). An event's `t` is its presentation time:
masterTs on the real system, which already includes the room latency budget L. It is the moment the hit
must be felt and seen.

Colour: "linear" rgb 0..1 = LED drive fraction (see twin/panel.py). A LightFrame holds what the panel
physically shows, so it includes the 0.3 hardware cap. Flash analysis works on the COMMANDED light
(frame / cap, i.e. as if there were no cap: conservative, research report twin-spec/interfaces.md 5.4),
converted to sRGB exactly once, because safety.flash expects sRGB.

HONESTY. The ideal panel needs per-pixel frames. The lamp's SDK has no per-pixel route (research report
twin-spec/light.md section 5), so the ideal frames are a design reference, NOT something the real lamp can
show. Everything under "SDK light path" models what light.glow really does; its numbers are cited below and
the ones marked ASSUMPTION are not measured. All of this is simulation evidence, not hardware approval.
"""
from __future__ import annotations

import bisect
import dataclasses
import heapq
import json
import math
import subprocess
import sys
import types
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import panel as P
from . import sim_light as L
from .contract import LightFrame, MusicEvent

ROOT = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------ constants (every number says where it is from)
FPS = 60.0                  # our panel frame rate; the vendor's fades also draw at <= 60 Hz (vendor source
                            # modules/robot_base/lights/rendering/transitions.py:9)
LATENCY_S = 0.300           # room latency budget L: Mac conductor source Show/Show.swift:56 (default latencyMs 300);
                            # masterTs = capture pts + L (Show.swift:370)
FTM_KINDS = ("CLICK", "KICK", "SNARE", "BASS", "BUILD", "DROP")    # twin/contract.py MusicEvent.kind

# Haptic defaults (sharpness, duration ms): Mac conductor source Show/HapticAnalyzer.swift:33-38.
HAPTIC_DEFAULTS = {"KICK": (0.22, 70), "SNARE": (0.85, 40), "BASS": (0.08, 300), "BUILD": (0.40, 2000),
                   "DROP": (0.45, 700), "CLICK": (0.50, 10)}
KICK_REFRACTORY_S = 0.100   # the analyser never emits two kicks closer than this: HapticAnalyzer.swift:116
SNARE_REFRACTORY_S = 0.090  # same for snares: HapticAnalyzer.swift:134

# How long before `t` an event reaches a consumer (masterTs - sendH): median 262 ms, min 156 ms; detection
# lag median 37 ms, p90 58 ms. Measured on the Mac conductor run log 2026-09-19T091049Z-wifi-music
# (research report twin-spec/light.md section 6.1).
LEAD_MEDIAN_S = 0.262
LEAD_MIN_S = 0.156
DETECT_LAG_MEDIAN_S = 0.037
DETECT_LAG_P90_S = 0.058

# Light-touch simultaneity (research report deaf-hoh-design.md:128-138): visuo-tactile temporal-order JND
# 30 ms (Fujisaki & Nishida 2009); first judged asynchronous at about 45 ms (Vogels 2004). Rule: light onset
# within +-30 ms of the haptic, aim +-20 ms, light may lead but never lag.
SYNC_WINDOW_S = 0.030
SYNC_AIM_S = 0.020
ASYNC_NOTICED_S = 0.045

# Flash safety. WCAG 2.2 SC 2.3.1: at most 3 flashes in any 1 s. The project's stricter governor: at most 2
# up-down excursions per second, leading edges >= 500 ms apart (research report deaf-hoh-design.md:149-166).
WCAG_FLASHES_PER_S = 3
GOVERNOR_FLASHES_PER_S = 2
GOVERNOR_EDGE_SPACING_S = 0.5
FLASH_DELTA = 0.10          # WCAG: a transition is a relative-luminance change of 10 % of max ...
FLASH_DARKER_BELOW = 0.80   # ... where the darker state is below 0.80
RED_FRACTION = 0.8          # saturated red: linear R / (R + G + B) >= 0.8 (deaf-hoh-design.md:166)

# SDK light path (research report twin-spec/light.md, which cites the vendor source for each). The per-frame
# rules (the 600 ms smootherstep fade drawn at 60 Hz, the canned effects at 20 Hz, the start state, the 4 s
# orphan timeout) live once, in twin/sim_light.py, which the simulated gateway uses too; here they are only
# named. SdkLamp below adds what the gateway side does not: admission latency and the shared rate window.
TRANSITION_S = L.TRANSITION_MS / 1e3    # every colour glow is a 600 ms crossfade: vendor source config/default.yaml:63;
                                        # measured on the lamp (live config 600, research report
                                        # pi-control-surface.md:156)
FADE_HZ = 1.0 / L.FADE_FRAME_S          # fade frames at <= 60 Hz, first after 1/60 s: vendor source
                                        # rendering/transitions.py:9-38
EFFECT_HZ = L.EFFECT_FRAME_HZ           # canned effects draw at 20 Hz on a frame counter: vendor source
                                        # effects/player.py:48,56-69
RATE_LIMIT_PER_MIN = 30     # per SDK session, sliding 60 s, motion + light share it: vendor source
                            # config/default.yaml:451, policy.py:166-187. 120 is in the lamp's .env but NOT in
                            # effect until a runtime restart (research report pi-internet-deps.md:231): a switch.
ORPHAN_TIMEOUT_S = L.LIFECYCLE_ACK_S    # a colour glow whose fade is cancelled gets no result; 503 4 s after its
                                        # handler ends (the cancel): vendor source sdk_gateway/mapper.py:176
START_RGB = L.START_COLOR               # ambient warm white before our first call: vendor source
                                        # config/default.yaml:66-73
START_BRIGHTNESS = L.START_BRIGHTNESS   # the lamp's live brightness_level: measured on the lamp (research report
                                        # pi-control-surface.md:158); a colour-only glow keeps it (mapper.py:154-157)
MOTION_PER_MIN = 26.0       # settle-chained head tracking needs <= 26 motion.move per minute on the same
                            # session (research report twin-spec/motion.md section 8)

# The canned effects our clients may use (vendor source rendering/effects/player.py:13-44; drawn by
# twin/sim_light.py effect_frame). Only two are safe for the audience.
EFFECTS = ("breathing", "flowing")
FORBIDDEN_EFFECTS = {       # research report twin-spec/light.md section 2; deaf-hoh-design.md:155-166
    "rainbow": "contains saturated red",
    "sparkle": "re-rolls 16 times a second: the photosensitive peak band",
    "police": "saturated red/blue at 2.5 Hz",
}

# Colour palettes per section, LINEAR drive colours. Normalised below to one relative luminance (iso-
# luminant), so a colour switch changes hue, not total light, and never counts as a flash (research report
# deaf-hoh-design.md:163-164: "shift hue at constant luminance"). Hue is a stable code per section
# (deaf-hoh-design.md section 6). Every colour keeps linear R/(R+G+B) well under 0.8 (no saturated red).
# Starting points: base cool blue (0.15, 0.45, 0.90) and accent magenta (0.75, 0.20, 0.75) from the team's
# lamp/performance.py:59-60 (branch gemini38/lamp-follower-music). The rest are design choices (ASSUMPTION),
# picked in CIE u'v' so that bar steps inside a section are 0.07-0.11 apart (colour_switches counts > 0.04)
# and the DROP (break blue -> electric lime, 0.17) is the biggest change of the song.
_RAW_PALETTES = {
    "intro": [(0.15, 0.45, 0.90), (0.05, 0.85, 0.65)],                      # cool blue, teal
    "verse": [(0.45, 0.35, 1.00), (0.10, 0.85, 1.00)],                      # violet-blue, cyan
    "groove": [(0.15, 0.45, 0.90), (0.05, 0.85, 0.65)],                     # run logs: no named sections
    "build": [(1.00, 0.50, 0.08), (1.00, 0.30, 0.55)],                      # amber, rose: warm
    "break": [(0.30, 0.35, 1.00)],                                          # dim deep blue, held
    "drop": [(0.50, 1.00, 0.10), (0.10, 0.85, 1.00)],                       # electric lime, cyan
    "outro": [(1.00, 0.85, 0.65), (0.75, 0.20, 0.75)],                      # warm white, magenta accent
}


def _iso_palettes(raw: dict) -> dict:
    """Scale every colour to the same relative luminance, the highest all of them can reach."""
    y_ref = min(float(P.luminance(c)) / max(c) for cs in raw.values() for c in cs)
    return {k: [np.asarray(c, float) * (y_ref / float(P.luminance(c))) for c in cs] for k, cs in raw.items()}


PALETTES = _iso_palettes(_RAW_PALETTES)


# ------------------------------------------------------------------ small helpers
def _smootherstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def _q(values, q: float) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.quantile(values, q)) if values.size else float("nan")


def _frame_times(t0: float, t1: float, fps: float) -> np.ndarray:
    n = int(math.floor((t1 - t0) * fps + 1e-9)) + 1
    return t0 + np.arange(n) / fps


def _is_saturated_red(rgb_linear) -> np.ndarray:
    rgb = np.asarray(rgb_linear, dtype=float)
    total = rgb.sum(axis=-1)
    return (total > 1e-6) & (rgb[..., 0] >= RED_FRACTION * np.maximum(total, 1e-12))


def _load_team_module(name: str, branch: str, path: str, *, try_import: bool = True):
    """A teammate's module: normal import (unless try_import is False), else its source from
    `git show origin/<branch>:<path>` executed in memory (nothing is written). Returns (module or None, how
    it was loaded). Research report twin-spec/interfaces.md section 1 asks for exactly this until the
    branches are merged."""
    if try_import:
        try:
            return __import__(name, fromlist=["_"]), "import"
        except ImportError:
            pass
    try:
        git = ["git", "-C", str(ROOT)]
        src = subprocess.run(git + ["show", f"origin/{branch}:{path}"], capture_output=True, text=True,
                             timeout=15, check=True).stdout
        commit = subprocess.run(git + ["rev-parse", f"origin/{branch}"], capture_output=True, text=True,
                                timeout=15, check=True).stdout.strip()
        mod_name = "_twin_team_" + name.replace(".", "_")
        mod = types.ModuleType(mod_name)
        mod.__file__ = f"git:origin/{branch}:{path}"
        sys.modules[mod_name] = mod            # dataclasses look their module up while the class is built
        exec(compile(src, mod.__file__, "exec"), mod.__dict__)
        return mod, f"git-show origin/{branch}@{commit[:12]}"
    except Exception:                          # no git, no branch, or the module fails: the caller decides
        return None, "unavailable"


_FLASH_CACHE: list = []


def load_flash():
    """(safety.flash module or None, provenance). Team branch claude/flash-limiter (PR #5).

    A module that imports as `safety.flash` but lacks the team limiter's API (an unrelated installed package
    named `safety`, say) is not trusted: the branch's own source is loaded instead, and None if that fails."""
    if not _FLASH_CACHE:
        mod, how = _load_team_module("safety.flash", "claude/flash-limiter", "safety/flash.py")
        if mod is not None and not all(callable(getattr(mod, name, None)) for name in ("FlashLimiter", "analyze")):
            mod, how = _load_team_module("safety.flash", "claude/flash-limiter", "safety/flash.py", try_import=False)
            if mod is not None and not all(callable(getattr(mod, n, None)) for n in ("FlashLimiter", "analyze")):
                mod, how = None, "unavailable"
        _FLASH_CACHE.append((mod, how))
    return _FLASH_CACHE[0]


# ------------------------------------------------------------------ songs
@dataclass(frozen=True, eq=False)
class Song:
    events: list                     # MusicEvent, sorted by t (presentation time)
    bass_t: np.ndarray               # bass envelope sample times (s); may be empty
    bass_level: np.ndarray           # 0..1
    sections: list                   # (t_start, t_end, name)
    bpm: float | None
    beats: np.ndarray                # beat times if the song's grid is known (a "cue sheet"), else empty
    bars: np.ndarray                 # bar-line (downbeat) times if known, else empty
    lead_s: np.ndarray               # per event: how long before t it reaches a consumer (masterTs - sendH)
    latency_s: float                 # L the events were produced with
    origin: str                      # provenance for the evidence
    truth: list = field(default_factory=list)   # like the conductor's "truth" line: {"k", "s", "len"?}
    titan: dict = field(default_factory=dict)   # the first "titan" line of a run log (reportedMs, trimMs, ...)

    @property
    def start(self) -> float:
        return self.events[0].t if self.events else 0.0

    @property
    def end(self) -> float:
        ends = [e.t for e in self.events] + [s[1] for s in self.sections]
        return max(ends) if ends else 0.0

    def section_at(self, t: float) -> str:
        for t0, t1, name in self.sections:
            if t0 <= t < t1:
                return name
        return "groove"

    def bass_at(self, t) -> np.ndarray:
        if not len(self.bass_t):
            return np.full(np.shape(t), np.nan)
        return np.interp(t, self.bass_t, self.bass_level, left=0.0, right=0.0)

    def arrival(self, i: int) -> float:
        """When event i reaches a consumer on the real system: t - lead."""
        return self.events[i].t - float(self.lead_s[i])

    def window(self, t0: float, t1: float) -> "Song":
        """The part of the song with t0 <= t < t1 (same clock; nothing is shifted)."""
        keep = [i for i, e in enumerate(self.events) if t0 <= e.t < t1]
        b = (self.bass_t >= t0) & (self.bass_t < t1)
        return Song(events=[self.events[i] for i in keep], bass_t=self.bass_t[b], bass_level=self.bass_level[b],
                    sections=[(max(a, t0), min(z, t1), n) for a, z, n in self.sections if z > t0 and a < t1],
                    bpm=self.bpm, beats=self.beats[(self.beats >= t0) & (self.beats < t1)],
                    bars=self.bars[(self.bars >= t0) & (self.bars < t1)], lead_s=self.lead_s[keep],
                    latency_s=self.latency_s, origin=f"{self.origin} [window {t0:g}..{t1:g} s]",
                    truth=self.truth, titan=self.titan)


def _event(t: float, kind: str, intensity: float) -> MusicEvent:
    sharp, dur = HAPTIC_DEFAULTS[kind]
    return MusicEvent(t=float(t), kind=kind, intensity=float(min(1.0, max(0.0, intensity))),
                      sharpness=sharp, duration_ms=dur)


def _modelled_leads(n: int, latency_s: float, rng) -> np.ndarray:
    """ASSUMPTION (shape): lead = L - detection lag, the lag a shifted exponential whose median (37 ms) and
    p90 (58 ms) are the measured ones, clipped so the lead never drops below the measured minimum 156 ms
    (at L = 300 ms). Gives a median lead of about 263 ms against the measured 262 ms."""
    scale = (DETECT_LAG_P90_S - DETECT_LAG_MEDIAN_S) / (math.log(10.0) - math.log(2.0))
    lag = DETECT_LAG_MEDIAN_S - scale * math.log(2.0) + rng.exponential(scale, n)
    return np.clip(latency_s - lag, latency_s - (LATENCY_S - LEAD_MIN_S), latency_s)


def synthetic_song(bpm: float = 120.0, bars: int = 16, seed: int = 0, *, start_t: float = 2.0,
                   latency_s: float = LATENCY_S) -> Song:
    """The Mac conductor's synthetic track, as the events its analyser emits, with named sections.

    Mirrors Mac conductor source Show/FileSource.swift:68-126 bar for bar at the defaults (120 BPM, 16 bars,
    32 s): groove (KICK every beat, SNARE on 2 and 4, two-beat bass notes), a four-bar build (snare roll at
    2, 4, 8, 16 per beat, bass fading out), a drop bar (first half silence, DROP on beat 3, a heavy kick on
    beat 4, sub bass) and heavy groove to the end. Section names for colour: intro (first quarter), verse
    (rest of the groove), build, break (the silent half bar), drop (from the DROP), outro (last bar).
    For other lengths the same plan is scaled by quarters.

    `start_t` shifts the music so a tracker gets some search time first (research report
    twin-spec/interfaces.md 4.1 suggests 2 s). `seed` varies the analyser's per-hit strength a little
    (ASSUMPTION: +-0.05) and the modelled arrival leads; beat timing is exact (the analyser's median timing
    error on this track is 0 ms: HapticAnalyzer.swift:47-48).
    """
    if bars < 6:
        raise ValueError("synthetic_song needs at least 6 bars (groove, build, drop, outro)")
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    bar_len = 4 * beat
    n_intro = max(1, bars // 4)
    n_verse = max(1, bars // 2 - n_intro)
    n_build = max(1, bars // 4)
    drop_bar = n_intro + n_verse + n_build
    outro_bar = bars - 1
    events: list[MusicEvent] = []
    truth: list[dict] = []
    notes: list[tuple[float, float, float]] = []          # bass notes (start, length, gain)
    last = {"KICK": -1e9, "SNARE": -1e9}

    def jitter() -> float:
        return float(rng.uniform(-0.05, 0.05))

    def kick(t: float, strength: float) -> None:
        truth.append({"k": "KICK", "s": t - start_t})
        if t - last["KICK"] > KICK_REFRACTORY_S:
            last["KICK"] = t
            events.append(_event(t, "KICK", 0.5 + 0.5 * min(1.0, strength + jitter())))   # HapticAnalyzer.swift:127

    def snare(t: float, strength: float) -> None:
        truth.append({"k": "SNARE", "s": t - start_t})
        if t - last["SNARE"] > SNARE_REFRACTORY_S:          # the analyser drops the rest of a fast roll
            last["SNARE"] = t
            events.append(_event(t, "SNARE", 0.45 + 0.4 * min(1.0, strength + jitter())))  # HapticAnalyzer.swift:139

    def bass(t: float, length: float, gain: float) -> None:
        notes.append((t, length, gain))
        truth.append({"k": "BASS", "s": t - start_t, "len": length})

    for b in range(bars):
        b0 = start_t + b * bar_len
        if b < n_intro + n_verse or (drop_bar < b):
            heavy = b > drop_bar
            for q in range(4):
                # Strength: 1.0 on groove downbeats, lower elsewhere (ASSUMPTION per interfaces.md 4.1);
                # the heavy groove after the drop is played at full gain (FileSource.swift:117-121).
                kick(b0 + q * beat, 1.0 if (heavy or q == 0) else 0.5)
            snare(b0 + beat, 0.9 if heavy else 0.7)
            snare(b0 + 3 * beat, 0.9 if heavy else 0.7)
            if heavy:
                bass(b0, bar_len - 0.02, 0.5)
            else:
                bass(b0, 2 * beat - 0.02, 0.35)
                bass(b0 + 2 * beat, 2 * beat - 0.02, 0.35)
        elif b < drop_bar:                                       # the build
            j = b - (n_intro + n_verse)
            div = (2, 4, 8, 16)[min(3, j * 4 // n_build)]
            for k in range(4 * div):
                # the roll's gain rises 0.25 -> 0.60 (FileSource.swift:107); strength = gain / 0.6 (ASSUMPTION)
                snare(b0 + k * beat / div, (0.25 + 0.35 * j / max(1, n_build - 1)) / 0.6)
            if j == 0:
                bass(b0, bar_len, 0.2)
                truth.append({"k": "BUILD_START", "s": b0 - start_t})
        else:                                                    # the drop bar: silence, then DROP on beat 3
            t_drop = b0 + 2 * beat
            truth.append({"k": "DROP", "s": t_drop - start_t})
            events.append(_event(t_drop, "DROP", 1.0))           # the analyser emits DROP instead of that kick
            last["KICK"] = t_drop
            kick(t_drop + beat, 1.0)
            bass(t_drop, 2 * beat - 0.02, 0.5)

    # BUILD fires once the bass has been gone for 1 s with >= 8 snares in 2 s (HapticAnalyzer.swift:168-171):
    # about 1 s into the build (ASSUMPTION, research report twin-spec/interfaces.md 4.1).
    t_build = start_t + (n_intro + n_verse) * bar_len + 1.0
    n_snares = sum(1 for e in events if e.kind == "SNARE" and t_build - 2.0 < e.t <= t_build)
    events.append(_event(t_build, "BUILD", min(1.0, 0.5 + 0.03 * n_snares)))   # HapticAnalyzer.swift:171
    events.sort(key=lambda e: (e.t, e.kind))

    # Bass envelope, a point every 10 ms like the analyser (HapticAnalyzer.swift:143-151). Level = note gain
    # relative to the loudest note, with 10 ms attack / 30 ms release (FileSource.swift:93) and the
    # analyser's duck to 0.35 for 80 ms after each kick (HapticAnalyzer.swift:146). Shape ASSUMPTION.
    end = start_t + bars * bar_len
    bass_t = np.arange(start_t, end, 0.010)
    level = np.zeros_like(bass_t)
    top = max(g for _, _, g in notes)
    for t0, length, gain in notes:
        u = bass_t - t0
        env = np.clip(np.minimum(u / 0.010, (length - u) / 0.030), 0.0, 1.0)
        level = np.maximum(level, env * gain / top)
    for e in events:
        if e.kind in ("KICK", "DROP"):
            level[(bass_t >= e.t) & (bass_t < e.t + 0.080)] *= 0.35

    bar_t = start_t + np.arange(bars) * bar_len
    t_drop = bar_t[drop_bar] + 2 * beat
    sections = [(bar_t[0], bar_t[n_intro], "intro"),
                (bar_t[n_intro], bar_t[n_intro + n_verse], "verse"),
                (bar_t[n_intro + n_verse], bar_t[drop_bar], "build"),
                (bar_t[drop_bar], t_drop, "break"),
                (t_drop, bar_t[outro_bar], "drop"),
                (bar_t[outro_bar], end, "outro")]
    return Song(events=events, bass_t=bass_t, bass_level=level, sections=sections, bpm=float(bpm),
                beats=start_t + np.arange(bars * 4) * beat, bars=bar_t,
                lead_s=_modelled_leads(len(events), latency_s, rng), latency_s=latency_s,
                origin=f"synthetic_song(bpm={bpm:g}, bars={bars}, seed={seed}) mirroring the Mac synthetic track",
                truth=truth)


def loop_song(song: Song, t_end: float) -> Song:
    """`song` played again and again until t_end, as the Mac conductor loops its file or synthetic track
    (Mac conductor source Show/FileSource.swift:4). One period is the song's sections from start to end;
    every copy has the same events, leads and bass, shifted by whole periods."""
    if not song.sections:
        return song
    t_start, period = song.sections[0][0], song.sections[-1][1] - song.sections[0][0]
    n = max(1, math.ceil((t_end - t_start) / period - 1e-9))
    if n == 1 or period <= 0:
        return song
    shifts = [k * period for k in range(n)]
    events = [dataclasses.replace(e, t=e.t + d) for d in shifts for e in song.events]
    return Song(events=events, bass_t=np.concatenate([song.bass_t + d for d in shifts]),
                bass_level=np.tile(song.bass_level, n),
                sections=[(a + d, b + d, name) for d in shifts for a, b, name in song.sections],
                bpm=song.bpm, beats=np.concatenate([song.beats + d for d in shifts]),
                bars=np.concatenate([song.bars + d for d in shifts]), lead_s=np.tile(song.lead_s, n),
                latency_s=song.latency_s, origin=f"{song.origin}, looped x{n} (FileSource.swift:4)",
                truth=[dict(x, s=x["s"] + d) for d in shifts for x in song.truth], titan=song.titan)


def derive_sections(events: list, t_end: float, drop_s: float = 8.0) -> list:
    """Coarse sections from BUILD/DROP when nothing else is known: build from a BUILD to the next DROP, drop
    for 8 s after a DROP, groove otherwise (research report twin-spec/interfaces.md 4.2; 8 s ASSUMPTION)."""
    marks = sorted((e.t, e.kind) for e in events if e.kind in ("BUILD", "DROP"))
    t0 = events[0].t if events else 0.0
    out, cur, since = [], "groove", t0
    for t, kind in marks:
        if kind == "BUILD" and cur != "build":
            out.append((since, t, cur))
            cur, since = "build", t
        elif kind == "DROP":
            out.append((since, t, cur))
            cur, since = "drop", t
            nxt = [m for m in marks if m[0] > t]
            if not nxt or nxt[0][0] > t + drop_s:
                out.append((t, min(t + drop_s, t_end), "drop"))
                cur, since = "groove", min(t + drop_s, t_end)
    out.append((since, max(t_end, since), cur))
    return [s for s in out if s[1] > s[0]]


def estimate_bpm(kick_times) -> float | None:
    """Median kick interval folded into 0.33..1.0 s (60..180 BPM). None with fewer than 4 kicks."""
    k = np.sort(np.asarray(kick_times, dtype=float))
    if k.size < 4:
        return None
    iv = np.diff(k)
    iv = iv[iv > 0.1]
    if not iv.size:
        return None
    iv = np.where(iv < 0.333, iv * 2, iv)
    iv = np.where(iv >= 1.0, iv / 2, iv)
    return 60.0 / float(np.median(iv))


def from_runlog(path, *, include_measure: bool = False, origin: str = "first_event",
                start_t: float = 2.0) -> Song:
    """A real Mac conductor run.jsonl as a Song.

    Event lines: {"t":"event", "kind", "masterTs" (Mac host ns when it must be felt; already includes L),
    "intensity" (after hapticGain), "sharpness", "durationMs", "sendH", "flags"} (Mac conductor source
    Show/Show.swift:352-361). Times become seconds relative to the first event (origin="first_event", placed
    at `start_t`) or to the start line's host time (origin="start"). Sorted by masterTs, because log order
    is send order. Events flagged "measure" (flags & 4) are clicks-mode test slots and are skipped unless
    asked. The bass envelope is not in the log, so Song.bass is empty; sections come from BUILD/DROP.
    Local paths in the log ("out", "file") are never read into the Song.
    """
    start_h, latency_ms, titan, rows = None, None, {}, []
    for line in Path(path).read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict):
            continue
        kind = row.get("t")
        if kind == "start" and start_h is None:
            start_h, latency_ms = row.get("h"), row.get("latencyMs")
        elif kind == "titan" and not titan and "reportedMs" in row:
            titan = {k: row[k] for k in ("reportedMs", "trimMs", "latencyMs", "transport") if k in row}
        elif kind == "event":
            name = str(row.get("kind", "")).upper()
            if name not in FTM_KINDS or "masterTs" not in row:
                continue                        # never map an unknown kind onto another
            if int(row.get("flags", 0)) & 4 and not include_measure:
                continue
            rows.append(row | {"kind": name})
    if not rows:
        raise ValueError(f"{path}: no event lines")
    rows.sort(key=lambda r: int(r["masterTs"]))
    base = int(rows[0]["masterTs"]) if origin == "first_event" or start_h is None else int(start_h)
    offset = start_t if origin == "first_event" or start_h is None else 0.0
    events, leads = [], []
    for r in rows:
        default_sharp, default_dur = HAPTIC_DEFAULTS[r["kind"]]
        events.append(MusicEvent(t=offset + (int(r["masterTs"]) - base) / 1e9, kind=r["kind"],
                                 intensity=float(r.get("intensity", 1.0)),
                                 sharpness=float(r.get("sharpness", default_sharp)),
                                 duration_ms=int(r.get("durationMs", default_dur))))
        leads.append((int(r["masterTs"]) - int(r["sendH"])) / 1e9 if "sendH" in r else float("nan"))
    kicks = [e.t for e in events if e.kind == "KICK"]
    latency_s = (latency_ms if latency_ms is not None else LATENCY_S * 1e3) / 1e3
    leads_arr = np.array(leads)
    leads_arr[np.isnan(leads_arr)] = LEAD_MEDIAN_S            # unknown lead: the measured median
    return Song(events=events, bass_t=np.zeros(0), bass_level=np.zeros(0),
                sections=derive_sections(events, events[-1].t + 1.0), bpm=estimate_bpm(kicks),
                beats=np.zeros(0), bars=np.zeros(0), lead_s=leads_arr, latency_s=latency_s,
                origin=f"run.jsonl ({len(events)} events, origin={origin})", titan=titan)


def from_analysis(wav, *, latency_s: float = LATENCY_S, start_t: float = 2.0, seed: int = 0) -> Song:
    """Events from a WAV through the team's analysis.conductor (branch claude/music-analysis).

    Mapping (research report twin-spec/interfaces.md 4.3): t = start_t + e.t + L (so t is the presentation
    time), kick/snare upper-cased, "onset" dropped, intensity from the Mac analyser's formulas, bass envelope
    from the per-hop envelope, beats from its beat grid, bars every 4 beats from the first (ASSUMPTION).
    Note from that module: its bass envelope is not safe to drive a light directly, so LightDesign smooths it.
    """
    conductor, how = _load_team_module("analysis.conductor", "claude/music-analysis", "analysis/conductor.py")
    if conductor is None:
        raise RuntimeError("from_analysis needs analysis.conductor (team branch claude/music-analysis): it is "
                           "neither importable nor loadable with `git show origin/claude/music-analysis:"
                           "analysis/conductor.py`. Use synthetic_song() or from_runlog() instead.")
    a = conductor.analyze_file(str(wav))
    shift = start_t + latency_s
    events = []
    for e in a.events:
        kind = str(e.kind).upper()
        if kind == "KICK":
            events.append(_event(shift + e.t, kind, 0.5 + 0.5 * e.strength))    # HapticAnalyzer.swift:127
        elif kind == "SNARE":
            events.append(_event(shift + e.t, kind, 0.45 + 0.4 * e.strength))   # HapticAnalyzer.swift:139
    events.sort(key=lambda e: e.t)
    env = np.asarray(a.bass_envelope, dtype=float)
    beats = shift + np.asarray(a.beat_times, dtype=float)
    end = shift + len(env) * a.hop_seconds
    return Song(events=events, bass_t=shift + np.arange(len(env)) * a.hop_seconds, bass_level=env,
                sections=[(shift, end, "groove")], bpm=a.bpm, beats=beats, bars=beats[::4],
                lead_s=_modelled_leads(len(events), latency_s, np.random.default_rng(seed)),
                latency_s=latency_s, origin=f"analysis.conductor ({how}) on a WAV")


# ------------------------------------------------------------------ beat grid
def estimate_grid(song: Song) -> tuple[np.ndarray, str]:
    """Bar lines: the song's own when known, else walked from the KICKs (non-causal; first kick = downbeat,
    ASSUMPTION). Only the IDEAL design uses this; the SDK client predicts causally (BeatTracker)."""
    if len(song.bars):
        return np.asarray(song.bars, dtype=float), "song grid (known track)"
    kicks = np.array([e.t for e in song.events if e.kind in ("KICK", "DROP")])
    bpm = song.bpm or estimate_bpm(kicks)
    if bpm is None or kicks.size < 4:
        return np.zeros(0), "none (too few kicks)"
    period = 60.0 / bpm
    beats, b = [], float(kicks[0])
    while b <= kicks[-1] + 1e-9:
        near = kicks[np.abs(kicks - b) < 0.2 * period]
        b = float(near[0]) if near.size else b                   # snap to a kick when one is close
        beats.append(b)
        b += period
    return np.array(beats[::4]), "estimated from KICKs (first kick = downbeat, ASSUMPTION)"


class BeatTracker:
    """Causal beat prediction from KICK presentation times as they ARRIVE (what an SDK client can know).

    Period: median of the last 8 kick intervals, each folded into 0.33..1.0 s. Beat numbers count from the
    first kick, which is taken as a downbeat (ASSUMPTION). Stops predicting 4 s after the last kick
    (ASSUMPTION: do not paint bar lines into silence)."""

    def __init__(self, coast_s: float = 4.0):
        self.kicks: list[float] = []
        self.index: list[int] = []
        self.coast_s = coast_s

    def add(self, t: float) -> None:
        if self.kicks and t - self.kicks[-1] <= KICK_REFRACTORY_S:
            return
        p = self.period()
        n = self.index[-1] + max(1, round((t - self.kicks[-1]) / p)) if (self.kicks and p) else len(self.kicks)
        self.kicks.append(t)
        self.index.append(n)

    def period(self) -> float | None:
        if len(self.kicks) < 4:
            return None
        iv = np.diff(self.kicks[-9:])
        iv = np.where(iv < 0.333, iv * 2, iv)
        iv = np.where(iv >= 1.0, iv / 2, iv)
        return float(np.median(iv))

    def bars_in(self, a: float, b: float) -> list[float]:
        """Predicted bar lines in [a, b)."""
        p = self.period()
        if p is None:
            return []
        t_last, n_last = self.kicks[-1], self.index[-1]
        out, j = [], 1
        while True:
            t = t_last + j * p
            if t >= b or t > t_last + self.coast_s:
                return out
            if t >= a and (n_last + j) % 4 == 0:
                out.append(t)
            j += 1


# ------------------------------------------------------------------ the ideal light design
@dataclass
class DesignPlan:
    t: np.ndarray                    # frame times
    base: np.ndarray                 # pedestal level per frame (0..1 of full drive), smoothed
    colour: np.ndarray               # (n, 3) iso-luminant colour per frame
    accent: np.ndarray               # (n, 3) snare accent colour per frame
    pulses: list                     # (t, kind, depth above the pedestal)
    skipped: list                    # (t, kind, why)
    snares: list                     # (t, side +1 right / -1 left)
    switches: list                   # (t, section, reason)
    grid: str                        # where the bar lines came from

    def at(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.t, t, side="right") - 1, 0, len(self.t) - 1))


@dataclass
class LightDesign:
    """What an IDEAL panel should show (not reachable through the lamp's SDK: design reference only).

    kick  -> a fast whole-panel pulse on a pedestal: rises over `attack_s` and peaks ON the kick (so the
             light leads by at most one frame), exponential decay (95 % gone after `decay_s`).
    snare -> half of the outer ring flashes the section's accent colour, alternating sides, while the inner
             pixels dim so TOTAL light stays constant (research: snares get no luminance change,
             deaf-hoh-design.md section 6; the "outer ring flash" is a moving lit region, not a flash).
    bass  -> the pedestal, smoothed over >= 300 ms so it can never count as a flash (deaf-hoh-design.md s.6).
    colour-> a palette per section; steps on every bar line, switches on BUILD and DROP (DROP = the biggest
             hue jump), quick smootherstep cross-fade centred on the switch time. Iso-luminant.
    safety-> pulses no closer than 500 ms (the governor's leading-edge rule); no colour is a saturated red;
             level <= 1 of full drive, then the 0.3 hardware cap.
    Levels are fractions of full SDK drive. Values marked ASSUMPTION are design choices, not findings.
    """
    fps: float = FPS
    pedestal: float = 0.30           # ASSUMPTION: research starting point 35 % (deaf-hoh-design.md:165), a little lower
    bass_depth: float = 0.12         # ASSUMPTION: how much the smoothed bass lifts the pedestal
    bass_smooth_s: float = 0.30      # >= 300 ms smoothing: deaf-hoh-design.md section 6 (bass row)
    kick_depth: float = 0.40         # ASSUMPTION: research starting points pedestal 35 % / peak 70 %
                                     # (deaf-hoh-design.md:165). A FIXED depth above the pedestal gives every
                                     # kick the same shape, so its counted fall always comes at the same delay
                                     # and a 2 Hz train stays at exactly 2 flashes/s.
    drop_peak: float = 1.0           # DROP spends the one pulse at the cap (deaf-hoh-design.md:233)
    attack_s: float = 0.016          # <= 20 ms asked for; research: rise 20-60 ms keeps onset in the window
    decay_s: float = 0.20            # research: decay 150-300 ms (deaf-hoh-design.md:165)
    min_pulse_spacing_s: float = GOVERNOR_EDGE_SPACING_S
    snare_boost: float = 0.35        # ASSUMPTION
    snare_decay_s: float = 0.15      # ASSUMPTION
    snare_min_spacing_s: float = 0.25   # <= 4 per second: sync is not judgeable above ~4 Hz (deaf-hoh-design.md:131)
    colour_fade_s: float = 0.25      # ASSUMPTION: "smooth but quick"
    break_level: float = 0.18        # ASSUMPTION: silence = dim steady pedestal (deaf-hoh-design.md s.6)
    build_level: float = 0.52        # ASSUMPTION: the build ramps the pedestal up to this, smoothly, no strobe
    palettes: dict = field(default_factory=lambda: {k: list(v) for k, v in PALETTES.items()})
    hardware_cap: float = P.HARDWARE_CAP

    # ---------------------------------------------------------------- plan
    def plan(self, song: Song, t0: float | None = None, t1: float | None = None) -> DesignPlan:
        t0 = 0.0 if t0 is None else t0
        t1 = song.end + 1.5 if t1 is None else t1
        t = _frame_times(t0, t1, self.fps)
        bars, grid = estimate_grid(song)

        # Pulses on KICK / DROP / CLICK, thinned to the governor's 500 ms leading-edge spacing. A DROP
        # always pulses; a kick is skipped for a DROP that follows within the spacing if the DROP was
        # already known (arrived) when the kick would fire. A DROP that comes too soon after a pulse that
        # already fired keeps its colour change but not its pulse.
        pulses, skipped, last = [], [], -1e9
        idx = {id(e): i for i, e in enumerate(song.events)}
        drops = [(e.t, song.arrival(idx[id(e)])) for e in song.events if e.kind == "DROP"]
        for e in song.events:
            if e.kind not in ("KICK", "DROP", "CLICK"):
                continue
            if e.kind != "DROP" and any(0 < td - e.t < self.min_pulse_spacing_s and arr <= e.t for td, arr in drops):
                skipped.append((e.t, e.kind, "a DROP follows within 500 ms"))
                continue
            if e.t - last < self.min_pulse_spacing_s - 1e-6:
                skipped.append((e.t, e.kind, "governor: leading edges >= 500 ms apart"))
                continue
            pulses.append((e.t, e.kind))
            last = e.t

        snares, last_s, side = [], -1e9, 1
        for e in song.events:
            if e.kind == "SNARE" and e.t - last_s >= self.snare_min_spacing_s - 1e-6:
                snares.append((e.t, side))
                last_s, side = e.t, -side

        # Colour switches, in time order: every bar line steps through the section's palette (a new
        # section starts at its first colour); BUILD steps to the build palette; DROP jumps to the drop
        # palette's first colour (the biggest hue change). A bar line within one fade of an event switch
        # is dropped, so the event wins.
        marks = sorted([(float(b), "bar") for b in bars] +
                       [(e.t, e.kind) for e in song.events if e.kind in ("BUILD", "DROP")])
        switches, cur_sec, step = [], None, 0
        for tm, what in marks:
            if what == "bar":
                if switches and switches[-1][2] in ("BUILD", "DROP") and tm - switches[-1][0] < self.colour_fade_s:
                    continue
                sec = song.section_at(tm + 1e-6)
                why = "bar" if sec == cur_sec else "section"
                step = step + 1 if sec == cur_sec else 0
            else:
                if switches and tm - switches[-1][0] < self.colour_fade_s:
                    switches.pop()
                sec, why = ("build", "BUILD") if what == "BUILD" else ("drop", "DROP")
                step = step + 1 if (what == "BUILD" and cur_sec == "build") else 0
            switches.append((tm, sec, why, step))
            cur_sec = sec

        # Colour per frame: walk the switches, cross-fading over colour_fade_s centred on each.
        first_sec = song.section_at(song.start)
        colour = np.tile(self.palettes.get(first_sec, self.palettes["groove"])[0], (len(t), 1))
        accent = np.tile(self._accent(first_sec, 1), (len(t), 1))
        cur = colour[0].copy()
        for j, (ts, sec, _why, k) in enumerate(switches):
            pal = self.palettes.get(sec, self.palettes["groove"])
            new = pal[k % len(pal)]
            a = ts - self.colour_fade_s / 2
            upto = switches[j + 1][0] - self.colour_fade_s / 2 if j + 1 < len(switches) else math.inf
            i0, i1 = np.searchsorted(t, a), np.searchsorted(t, upto)   # only until the next switch takes over
            u = _smootherstep((t[i0:i1] - a) / self.colour_fade_s)[:, None]
            colour[i0:] = new
            colour[i0:i1] = cur * (1 - u) + new * u
            accent[i0:] = self._accent(sec, k + 1)
            cur = colour[i1 - 1].copy() if i1 < len(t) and i1 > i0 else new

        # Pedestal: bass (or a steady mid level when the song has no bass envelope), overridden by build
        # (smooth ramp) and break (dim steady), then smoothed over bass_smooth_s.
        bass = song.bass_at(t)
        bass = np.where(np.isnan(bass), 0.5, bass)               # ASSUMPTION: no envelope -> mid level
        base = self.pedestal + self.bass_depth * bass
        for s0, s1, name in song.sections:
            m = (t >= s0) & (t < s1)
            if name == "build":
                base[m] = self.pedestal + (self.build_level - self.pedestal) * (t[m] - s0) / max(1e-9, s1 - s0)
            elif name == "break":
                base[m] = self.break_level
        n = max(1, int(round(self.bass_smooth_s * self.fps)))
        base = np.convolve(np.pad(base, (n // 2, n - 1 - n // 2), mode="edge"), np.ones(n) / n, mode="valid")

        # Pulse depth above the pedestal: fixed for kicks, up to the cap for a DROP.
        pulses = [(tp, kind, (self.drop_peak - base[int(np.clip(np.searchsorted(t, tp), 0, len(t) - 1))])
                   if kind == "DROP" else self.kick_depth) for tp, kind in pulses]
        return DesignPlan(t=t, base=base, colour=colour, accent=accent, pulses=pulses, skipped=skipped,
                          snares=snares, switches=[(ts, sec, why) for ts, sec, why, _ in switches], grid=grid)

    def _accent(self, section: str, k: int) -> np.ndarray:
        pal = self.palettes.get(section, self.palettes["groove"])
        return pal[k % len(pal)] if len(pal) > 1 else self.palettes["drop"][1]

    def pulse_lift(self, t: np.ndarray, pulses: list) -> np.ndarray:
        """Level added above the pedestal per frame: depth x envelope, the envelope rising linearly over
        attack_s to peak exactly at the pulse time, then decaying exponentially (5 % left after decay_s)."""
        lift = np.zeros_like(t)
        tau = self.decay_s / 3.0
        for tp, _, depth in pulses:
            i0, i1 = np.searchsorted(t, tp - self.attack_s), np.searchsorted(t, tp + 12 * tau)   # e^-12: nothing left
            tt = t[i0:i1]
            env = np.where(tt <= tp, np.clip((tt - (tp - self.attack_s)) / self.attack_s, 0.0, 1.0),
                           np.exp(-np.maximum(tt - tp, 0.0) / tau))
            lift[i0:i1] = np.maximum(lift[i0:i1], depth * env)
        return lift

    # ---------------------------------------------------------------- frames
    def render(self, song: Song, t0: float | None = None, t1: float | None = None,
               geometry: P.PanelGeometry | None = None, plan: DesignPlan | None = None) -> list:
        """IDEAL panel frames (list[LightFrame]) at `fps`. Not flash-limited: see apply_flash_limit."""
        geometry = geometry or P.load_geometry()
        plan = plan or self.plan(song, t0, t1)
        t = plan.t
        level = np.clip(plan.base + self.pulse_lift(t, plan.pulses), 0.0, 1.0)

        # Snare: which half of the outer ring, and how strongly, per frame.
        outer = geometry.outer(1)
        halves = {1: outer[geometry.xy[outer, 0] >= 0], -1: outer[geometry.xy[outer, 0] < 0]}
        s_env, s_side = np.zeros_like(t), np.zeros(len(t), dtype=int)
        tau_s = self.snare_decay_s / 3.0
        for ts, side in plan.snares:
            i0, i1 = np.searchsorted(t, ts), np.searchsorted(t, ts + 12 * tau_s)
            e = np.exp(-(t[i0:i1] - ts) / tau_s)
            better = e > s_env[i0:i1]
            s_env[i0:i1][better], s_side[i0:i1][better] = e[better], side
        rests = {s: np.setdiff1d(np.arange(P.PIXEL_COUNT), r) for s, r in halves.items()}

        frames = []
        for k, tk in enumerate(t):
            rgb = np.tile(plan.colour[k] * level[k], (P.PIXEL_COUNT, 1))
            if s_env[k] > 1e-3:
                region, rest = halves[s_side[k]], rests[s_side[k]]
                boost = min(self.snare_boost * s_env[k], 1.0 - level[k])
                mix = plan.colour[k] * (1 - s_env[k]) + plan.accent[k] * s_env[k]
                rgb[region] = mix * (level[k] + boost)
                rgb[rest] = plan.colour[k] * max(0.0, level[k] - boost * len(region) / len(rest))
            frames.append(LightFrame(t=float(tk), rgb=np.clip(rgb, 0.0, 1.0) * self.hardware_cap))
        return frames


# ------------------------------------------------------------------ flash safety
def _perceived(frames: list) -> tuple[np.ndarray, np.ndarray]:
    """(times, perceived linear colour (n, 3)) of a list of LightFrames (panel.perceived_colour, vectorised)."""
    w = P.diffuser_weights()
    t = np.array([f.t for f in frames], dtype=float)
    rgb = np.array([w @ f.rgb for f in frames], dtype=float).reshape(-1, 3)
    return t, rgb


def _commanded(frames: list, cap: float = P.HARDWARE_CAP) -> tuple[np.ndarray, np.ndarray]:
    """(times, perceived linear colour / cap) for a list of LightFrames."""
    t, rgb = _perceived(frames)
    return t, np.clip(rgb / cap, 0.0, 1.0)


def _swings(t: np.ndarray, y: np.ndarray, delta: float = FLASH_DELTA, darker_below: float = FLASH_DARKER_BELOW):
    """Built-in WCAG 2.3.1 transition counter (measurement only): (time, +1 rise / -1 fall) of every
    counted relative-luminance change of >= delta whose darker end is below darker_below. A monotonic ramp
    is one change however long it takes."""
    out, ref, cand, direction = [], None, 0.0, 0
    for tk, yk in zip(t, y, strict=True):
        if ref is None:
            ref = cand = yk
            continue
        if direction == 0:
            if abs(yk - ref) >= delta:
                direction, cand = (1 if yk > ref else -1), yk
                if min(ref, yk) < darker_below:
                    out.append((tk, direction))
        elif direction > 0:
            if cand - yk >= delta:
                if min(cand, yk) < darker_below:
                    out.append((tk, -1))
                ref, direction, cand = cand, -1, yk
            elif yk > cand:
                cand = yk
        else:
            if yk - cand >= delta:
                if min(cand, yk) < darker_below:
                    out.append((tk, 1))
                ref, direction, cand = cand, 1, yk
            elif yk < cand:
                cand = yk
    return out


def _max_in_window(times, window: float = 1.0) -> int:
    best, lo = 0, 0
    for hi, tk in enumerate(times):
        while times[lo] <= tk - window + 1e-9:
            lo += 1
        best = max(best, hi - lo + 1)
    return best


def _red_transitions(t: np.ndarray, rgb: np.ndarray) -> list:
    """Built-in red-flash transitions: changes into or out of a saturated red (measurement only)."""
    red = _is_saturated_red(rgb) & (P.luminance(rgb) > 1e-4)
    return [float(t[k]) for k in range(1, len(t)) if red[k] != red[k - 1]]


def flash_report(frames: list, *, use_team_meter: bool = True, cap: float = P.HARDWARE_CAP) -> dict:
    """Flashes per second (max over any 1 s window) of the light a list of frames emits, on the commanded
    light (frame / cap), perceived as one colour (every pixel counts: the lamp lights the room, so no area
    exemption, deaf-hoh-design.md:161-162). Uses safety.flash.analyze when available, else the built-in
    meter. Also: the smallest spacing between counted leading edges (rises), and saturated-red pixels."""
    t, rgb = _commanded(frames, cap)
    srgb = P.linear_to_srgb(rgb)
    y = P.luminance(rgb)
    swings = _swings(t, y)
    rises = [tk for tk, d in swings if d > 0]
    flash, how = load_flash() if use_team_meter else (None, "not used")
    if flash is not None:
        rep = flash.analyze([(float(tk), tuple(float(c) for c in s)) for tk, s in zip(t, srgb, strict=True)])
        general, red = rep.general_flashes_per_second, rep.red_flashes_per_second
        meter = f"safety.flash.analyze ({how})"
    else:
        general = math.ceil(_max_in_window([tk for tk, _ in swings]) / 2)
        red = math.ceil(_max_in_window(_red_transitions(t, rgb)) / 2)
        meter = "built-in WCAG 2.3.1 meter (safety.flash unavailable: measurement only)"
    red_px = sum(int(np.sum(_is_saturated_red(f.rgb) & (P.luminance(f.rgb) > 1e-5))) for f in frames)
    spacing = float(np.min(np.diff(rises))) if len(rises) > 1 else float("inf")
    return {
        "meter": meter,
        "general_flashes_per_s": int(general),
        "red_flashes_per_s": int(red),
        "counted_transitions": len(swings),
        "min_leading_edge_spacing_s": spacing,
        "saturated_red_pixel_frames": red_px,
        "wcag_ok": bool(general <= WCAG_FLASHES_PER_S and red <= WCAG_FLASHES_PER_S),
        "governor_ok": bool(general <= GOVERNOR_FLASHES_PER_S and red == 0
                            and spacing >= GOVERNOR_EDGE_SPACING_S - 1e-6),
    }


def apply_flash_limit(frames: list, *, max_flashes_per_second: int = GOVERNOR_FLASHES_PER_S,
                      margin_s: float = 0.0, cap: float = P.HARDWARE_CAP) -> tuple[list, dict]:
    """Run the team's FlashLimiter on the light actually emitted; when it holds, the whole previous frame is
    re-emitted (a held output cannot create a transition). margin_s = 0: the governor allows exactly
    2 flashes/s, which a 120 BPM pulse train is; the limiter's default 0.1 s margin would trim it
    (research report twin-spec/interfaces.md 5.4 suggests 0.1; see the deviation note in the report).
    FAILS CLOSED: without safety.flash no light is rendered at all (every frame dark), and the info says so
    (research report twin-spec/interfaces.md section 1: safety pieces fail closed)."""
    flash, how = load_flash()
    if flash is None:
        dark = [LightFrame(t=f.t, rgb=np.zeros_like(f.rgb)) for f in frames]
        return dark, {"limiter": "light suppressed: limiter unavailable (safety.flash could not be loaded)",
                      "held_frames": 0, "suppressed": True}
    lim = flash.FlashLimiter(max_flashes_per_second=max_flashes_per_second, margin_s=margin_s)
    out, prev = [], None
    _, commanded = _commanded(frames, cap)
    srgb = P.linear_to_srgb(commanded)
    for f, s in zip(frames, srgb, strict=True):
        held_before = lim.held
        lim.limit(float(f.t), (float(s[0]), float(s[1]), float(s[2])))
        if lim.held > held_before and prev is not None:
            out.append(LightFrame(t=f.t, rgb=prev.rgb.copy()))
        else:
            out.append(f)
            prev = f
    return out, {"limiter": f"applied: safety.flash.FlashLimiter(max_flashes_per_second="
                            f"{max_flashes_per_second}, margin_s={margin_s}) via {how}",
                 "held_frames": int(lim.held), "suppressed": False}


# ------------------------------------------------------------------ SDK light path (what the real lamp shows)
@dataclass
class SdkLightParams:
    transition_s: float = TRANSITION_S      # "hardware" preset; anything else is a labelled what-if (Option C)
    admission_s: tuple = (0.010, 0.040)     # POST -> fade start, typical. ASSUMPTION built on Mac->lamp ping
                                            # 3.99/20.65/103 ms + an audit fsync (twin-spec/light.md 1.2)
    admission_tail_s: float = 0.100         # ASSUMPTION: the tail (same source)
    admission_tail_p: float = 0.05          # ASSUMPTION: how often the tail happens
    reply_s: float = 0.015                  # fade end -> HTTP reply at the client (reply at 620-660 ms after
                                            # the POST: twin-spec/light.md 1.2). ASSUMPTION
    effect_first_frame_s: float = 0.005     # next loop turn + 2.79 ms wire + 1 ms driver gap. ASSUMPTION built
                                            # on vendor source simulation.yaml:18-19 and research report
                                            # ftm-integration.md:564
    effect_overrun: float = L.EFFECT_OVERRUN    # the 20 Hz effect clock runs 3-10 % slow. ASSUMPTION
                                                # (research report twin-spec/light.md section 2)
    rate_limit_per_min: int = RATE_LIMIT_PER_MIN
    hardware_cap: float = P.HARDWARE_CAP
    seed: int = 0

    def admission_quantile(self, q: float) -> float:
        """The q-quantile of the admission latency drawn by SdkLamp: uniform over admission_s, except a share
        admission_tail_p that is uniform from its top up to admission_tail_s."""
        lo, hi = self.admission_s
        body = 1.0 - self.admission_tail_p
        if q <= body:
            return lo + (hi - lo) * q / max(body, 1e-12)
        return hi + (self.admission_tail_s - hi) * (q - body) / max(self.admission_tail_p, 1e-12)


@dataclass
class SdkCall:
    """One light.glow POST and, after SdkLamp.run, what the lamp did with it."""
    t_send: float
    payload: dict                    # {"color": [r,g,b] 0..255, "luminance": x} | {"effect": name} | {"luminance": x}
    why: str = ""
    due: float = float("nan")        # the presentation time it is meant for
    aim: str = "onset"               # which point of the change is meant to land on `due`: onset | mid
    status: str = ""
    t_admit: float = float("nan")
    t_start: float = float("nan")    # fade or effect start: the light starts changing
    t_end: float = float("nan")      # fade end (or cancel)
    t_reply: float = float("nan")    # HTTP reply back at the client
    queued_behind: int = 0           # colour glows ahead of it on the lock when admitted

    def reset(self) -> None:
        self.status, self.queued_behind = "", 0
        self.t_admit = self.t_start = self.t_end = self.t_reply = float("nan")


@dataclass
class _Seg:
    t0: float
    kind: str                        # hold | fade | effect
    px0: np.ndarray                  # (93, 3) pixel values 0..255 at the start
    bri0: float
    px1: np.ndarray | None = None
    bri1: float = 0.0
    dur: float = 0.0
    effect: str = ""


def effect_pixels(name: str, elapsed_s: float, n: int = P.PIXEL_COUNT) -> np.ndarray:
    """Pixel values (n, 3), 0..255, of a canned effect `elapsed_s` into it on its own frame clock (the one
    shared rule: twin/sim_light.py effect_frame)."""
    return L.effect_frame(name, elapsed_s, n)


def fade_point(transition_s: float, frac: float) -> float:
    """When a light.glow fade has drawn `frac` of its step: frames every 1/60 s, the first after 1/60 s,
    smootherstep of elapsed / T (vendor source rendering/transitions.py:9-38). For T = 0.6 s: 10 % at
    150 ms, 50 % at 300 ms, 90 % at 467 ms (the continuous curve gives 148 / 300 / 452 ms)."""
    k = 1
    while k / FADE_HZ < transition_s and float(_smootherstep(k / FADE_HZ / transition_s)) < frac:
        k += 1
    return min(k / FADE_HZ, transition_s)


def _seg_state(seg: _Seg, t: float, p: SdkLightParams) -> tuple[np.ndarray, float]:
    if seg.kind == "hold":
        return seg.px0, seg.bri0
    if seg.kind == "fade":
        e = L.fade_progress(t - seg.t0, seg.dur)                     # the one shared rule (twin/sim_light.py)
        return np.round(seg.px0 + (seg.px1 - seg.px0) * e), seg.bri0 + (seg.bri1 - seg.bri0) * e
    k = math.floor((t - seg.t0) * EFFECT_HZ / (1.0 + p.effect_overrun) + 1e-9)   # frame-counted, runs slow
    return effect_pixels(seg.effect, max(0, k) / EFFECT_HZ), seg.bri0


@dataclass
class SdkRun:
    calls: list
    segs: list
    params: SdkLightParams
    motion_granted: int = 0         # head-tracking moves the shared rate limit let through
    motion_refused: int = 0         # ... and refused (429) because light used the window

    def state_at(self, t: float) -> tuple[np.ndarray, float]:
        for seg in reversed(self.segs):              # the simulation asks about "now": the last segments
            if seg.t0 <= t:
                return _seg_state(seg, t, self.params)
        return _seg_state(self.segs[0], t, self.params)

    def luminance_at(self, t: float) -> float:
        """The perceived luminance of the panel at any t (as frames() would show it): for onset times finer
        than the 60 Hz frames."""
        px, bri = self.state_at(t)
        return float(P.luminance(P.diffuser_weights() @ (px / 255.0 * bri * self.params.hardware_cap)))

    def frames(self, t0: float, t1: float, fps: float = FPS) -> list:
        """What the panel physically shows: pixel/255 x driver brightness x hardware cap."""
        starts = [s.t0 for s in self.segs]
        out = []
        for tk in _frame_times(t0, t1, fps):
            i = max(0, bisect.bisect_right(starts, tk) - 1)
            px, bri = _seg_state(self.segs[i], tk, self.params)
            out.append(LightFrame(t=float(tk), rgb=px / 255.0 * bri * self.params.hardware_cap))
        return out

    def summary(self) -> dict:
        status = {}
        for c in self.calls:
            key = c.status.split(":")[0]
            status[key] = status.get(key, 0) + 1
        colour = [c for c in self.calls if "color" in c.payload and not math.isnan(c.t_start)]
        waits = [c.t_start - c.t_admit for c in colour]
        sent = sorted(c.t_send for c in self.calls if not c.status.startswith("refused"))   # POSTs, 429s too
        admitted = sorted(c.t_send for c in self.calls if not math.isnan(c.t_admit))
        # True lag of the visible change behind the time it was meant for, from the call records: a colour
        # fade's 10 % point (aim "onset") or midpoint (aim "mid"), an effect's first frame. Only calls that
        # were shown and have a due time. Positive = late.
        point = {"onset": fade_point(self.params.transition_s, 0.1),
                 "mid": fade_point(self.params.transition_s, 0.5)}
        lag = [(c.t_start + (point[c.aim] if "color" in c.payload else 0.0)) - c.due for c in self.calls
               if c.status == "ok" and not math.isnan(c.due) and not math.isnan(c.t_start)]
        return {
            "calls": len(self.calls),
            "status": status,
            "visible_lag_ms": {"n": len(lag), "median": round(1e3 * _q(lag, 0.5), 1),
                               "p95": round(1e3 * _q(lag, 0.95), 1),
                               "max": round(1e3 * max(lag), 1) if lag else float("nan"),
                               # lead-only rule: a visible change may come early, never late
                               "share_not_late": round(float(np.mean(np.array(lag) <= 1e-9)), 4) if lag
                               else float("nan")},
            "colour_glows_started": len(colour),
            "lock_wait_s_median": _q(waits, 0.5),
            "lock_wait_s_max": float(max(waits)) if waits else float("nan"),
            "max_queue_depth": max((c.queued_behind for c in self.calls), default=0),
            "light_calls_max_in_60s": _max_in_window(sent, 60.0),
            "light_admitted_max_in_60s": _max_in_window(admitted, 60.0),
            "motion_moves_granted": self.motion_granted,
            "motion_moves_refused_429": self.motion_refused,
            "rate_limit_per_min": self.params.rate_limit_per_min,
            "transition_s": self.params.transition_s,
        }


class SdkLamp:
    """The lamp's light.glow path, modelled from the vendor source (research report twin-spec/light.md):

    * colour glow ({"color", "luminance"}): a smootherstep crossfade of every pixel AND the driver
      brightness over transition_s, frames at 60 Hz; serialised FIFO behind one lock; the HTTP call returns
      when the fade ends. Without "luminance" the current brightness_level is kept (starts at 0.03).
    * effect glow ({"effect"}): no lock; cancels a running colour fade (that call gets no result and a 503
      after 4 s although the light changed) and starts the effect from frame 0 on the next loop turn; the
      call returns on RUNNING. An unknown effect name is reported as success and shows nothing.
    * luminance-only glow: no lock; cancels a running fade; 600 ms brightness fade. (What happens to a
      luminance-only call when something later cuts ITS fade short is not modelled: our clients never send
      one. twin/sim_light.py, the simulated gateway's light, models the rest of the vendor behaviour.)
    * rate limit: per session, sliding 60 s, shared with motion.move (motion_times take slots too).
    Each call is admitted at t_send + its admission latency; a later POST can overtake an earlier one.
    No per-pixel access exists, so nothing here can draw the ideal design's pixels.
    """

    def __init__(self, params: SdkLightParams | None = None):
        self.p = params or SdkLightParams()

    def _admission(self, rng) -> float:
        lo, hi = self.p.admission_s
        a = rng.uniform(lo, hi)
        tail = rng.random() < self.p.admission_tail_p
        return float(rng.uniform(hi, self.p.admission_tail_s)) if tail else float(a)

    def run(self, calls: list, motion_times=()) -> SdkRun:
        p = self.p
        calls = sorted(calls, key=lambda c: c.t_send)
        rng = np.random.default_rng(p.seed)

        # 1. rate limit (sliding 60 s, per session; motion.move takes slots from the same window) and admission
        adm = {id(c): self._admission(rng) for c in calls}   # drawn in send order: same list -> same draws
        items = sorted([(float(m), 0, None) for m in motion_times] + [(c.t_send, 1, c) for c in calls],
                       key=lambda x: (x[0], x[1]))
        slots: deque = deque()
        granted = refused = 0
        for t, _, c in items:
            while slots and slots[0] <= t - 60.0:
                slots.popleft()
            if c is None:                                   # a head-tracking move on the same session
                if len(slots) < p.rate_limit_per_min:
                    slots.append(t)
                    granted += 1
                else:
                    refused += 1
                continue
            c.reset()
            name = str(c.payload.get("effect", "")).lower()
            if name in FORBIDDEN_EFFECTS:
                c.status = f"refused: {name} {FORBIDDEN_EFFECTS[name]} (client whitelist)"
                continue
            if len(slots) >= p.rate_limit_per_min:
                c.status = "429: rate limited (audited on the lamp, nothing queued)"
                c.t_reply = c.t_send + adm[id(c)]
                continue
            slots.append(c.t_send)
            c.t_admit = c.t_send + adm[id(c)]

        # 2. the lamp's light loop, as a discrete-event simulation in admission order
        segs = [_Seg(t0=-1e9, kind="hold", px0=np.tile(np.array(START_RGB, float), (P.PIXEL_COUNT, 1)),
                     bri0=START_BRIGHTNESS)]
        run = SdkRun(calls=calls, segs=segs, params=p, motion_granted=granted, motion_refused=refused)

        def push(seg: _Seg) -> None:
            while len(segs) > 1 and segs[-1].t0 > seg.t0:   # a later-starting segment that never showed
                segs.pop()
            segs.append(seg)

        heap, order = [], 0
        for c in calls:
            if not math.isnan(c.t_admit):
                heap.append((c.t_admit, order, "admit", c))
                order += 1
        heapq.heapify(heap)
        queue: deque = deque()
        running: list = []                                   # [call, token] of the colour fade holding the lock
        level_at_admit: dict = {}                            # driver level when a colour glow was admitted

        def start_fade(c: SdkCall, t: float) -> None:
            nonlocal order
            px, bri = run.state_at(t)
            target = np.tile(np.clip(np.asarray(c.payload["color"], float), 0, 255), (P.PIXEL_COUNT, 1))
            lum = float(c.payload["luminance"]) if "luminance" in c.payload else level_at_admit[id(c)]
            push(_Seg(t0=t, kind="fade", px0=px.copy(), bri0=bri, px1=target, bri1=lum, dur=p.transition_s))
            c.t_start = t
            running[:] = [c, order]
            heapq.heappush(heap, (t + p.transition_s, order, "end", order))
            order += 1

        def cancel(t: float) -> None:
            nonlocal order
            c = running[0]
            px, bri = run.state_at(t)
            push(_Seg(t0=t, kind="hold", px0=px.copy(), bri0=bri))
            c.status = "503: fade cancelled by a later effect/luminance call; light changed, no result"
            c.t_end, c.t_reply = t, t + ORPHAN_TIMEOUT_S     # the 4 s start when its handler ends (the cancel)
            running.clear()
            if queue:                                        # the lock frees on the next loop turn
                heapq.heappush(heap, (t + 0.001, order, "lock", None))
                order += 1

        while heap:
            t, _, kind, obj = heapq.heappop(heap)
            if kind == "admit":
                c, pay = obj, obj.payload
                if "color" in pay:
                    level_at_admit[id(c)] = run.state_at(t)[1]   # mapper fills a missing luminance from the level now
                    if not running and not queue:
                        start_fade(c, t)
                    else:
                        c.queued_behind = len(queue) + (1 if running else 0)
                        queue.append(c)
                elif "effect" in pay:
                    name = str(pay["effect"]).lower()        # names match case-insensitively
                    if name not in EFFECTS:                  # raises before cancelling anything, but the SDK
                        c.status = f"ok: unknown effect {name!r} reported as success, nothing shown"
                        c.t_reply = t + p.reply_s            # already answered RUNNING (controller.py:172-176)
                        continue
                    if running:
                        cancel(t)
                    px, bri = run.state_at(t)
                    push(_Seg(t0=t + p.effect_first_frame_s, kind="effect", px0=px.copy(), bri0=bri, effect=name))
                    c.status, c.t_start, c.t_reply = "ok", t + p.effect_first_frame_s, t + p.reply_s
                elif "luminance" in pay:
                    if running:
                        cancel(t)
                    px, bri = run.state_at(t)
                    if not px.any():                         # nothing showing: snaps to white first
                        px = np.full((P.PIXEL_COUNT, 3), 255.0)
                    push(_Seg(t0=t, kind="fade", px0=px.copy(), bri0=bri, px1=px.copy(),
                              bri1=float(pay["luminance"]), dur=p.transition_s))
                    c.status, c.t_start = "ok", t
                    c.t_end, c.t_reply = t + p.transition_s, t + p.transition_s + p.reply_s
                else:
                    c.status = "ok: empty payload plays the default effect (not modelled)"
            elif kind == "end":
                if running and running[1] == obj:
                    c = running[0]
                    px, bri = run.state_at(t)
                    push(_Seg(t0=t, kind="hold", px0=px.copy(), bri0=bri))
                    c.status, c.t_end, c.t_reply = "ok", t, t + p.reply_s
                    running.clear()
                    if queue:
                        start_fade(queue.popleft(), t)
            elif kind == "lock" and not running and queue:
                start_fade(queue.popleft(), t)
        return run


def glow(color, luminance: float | None, t_send: float, why: str = "", due: float = float("nan"),
         aim: str = "onset") -> SdkCall:
    """A colour glow call. `color` is linear drive 0..1 (one colour for the whole panel)."""
    payload = {"color": [int(round(255 * min(1.0, max(0.0, float(c))))) for c in color]}
    if luminance is not None:
        payload["luminance"] = float(min(1.0, max(0.0, luminance)))
    return SdkCall(t_send=float(t_send), payload=payload, why=why, due=due, aim=aim)


def effect(name: str, t_send: float, why: str = "", due: float = float("nan")) -> SdkCall:
    """An effect call. Forbidden effects are refused here (client whitelist), before anything is sent."""
    if name.lower() in FORBIDDEN_EFFECTS:
        raise ValueError(f"effect {name!r} is not allowed: {FORBIDDEN_EFFECTS[name.lower()]}")
    return SdkCall(t_send=float(t_send), payload={"effect": name}, why=why, due=due)


def _one_in_flight(cands: list, lamp: SdkLamp, motion_times, late_tolerance_s: float) -> tuple[list, int]:
    """Send candidates one at a time: wait for each reply, drop any that is more than late_tolerance_s
    past its due time by the time it could go out (lamp/performance.py :57, :329-333 on branch
    gemini38/lamp-follower-music)."""
    out, free_at, dropped = [], -1e9, 0
    for c in sorted(cands, key=lambda c: c.t_send):
        s = max(c.t_send, free_at)
        if not math.isnan(c.due) and s > c.due + late_tolerance_s:
            dropped += 1
            continue
        c.t_send = s
        out.append(c)
        lamp.run(out, motion_times)
        free_at = c.t_reply if not math.isnan(c.t_reply) else s
    return out, dropped


def reactive_calls(song: Song, plan: DesignPlan, lamp: SdkLamp, *, mode: str = "queue", trim_s: float = 0.030,
                   late_tolerance_s: float = 0.080, motion_times=()) -> tuple[list, dict]:
    """The IDEAL design sent literally through light.glow, the way an event-driven client would.

    Each pulse becomes an "up" glow (colour at the peak level) sent when the event arrives or at due - trim,
    whichever is later, and a "down" glow back to the pedestal right behind it; each colour switch becomes a
    glow at its due - trim. trim 30 ms is lamp/performance.py's lamp_trim_s (branch
    gemini38/lamp-follower-music :56), which assumes an instant LED. Every glow carries luminance (the fixed
    version of that client). Snare ring flashes need per-pixel access: not representable, counted.
    mode "queue": everything is POSTed and the lamp queues it (lag grows 0.6 s per excess call).
    mode "one_in_flight": wait for each reply, drop what is > 80 ms late (performance.py's policy).
    """
    idx_of = {round(e.t, 6): i for i, e in enumerate(song.events)}
    cands = []
    for tp, kind, depth in plan.pulses:
        i = idx_of.get(round(tp, 6))
        arrival = song.arrival(i) if i is not None else -1e9
        s = max(arrival, tp - trim_s)
        k = plan.at(tp)
        cands.append(glow(plan.colour[k], min(1.0, plan.base[k] + depth), s, f"{kind} up", tp))
        cands.append(glow(plan.colour[k], plan.base[k], s + 0.001, f"{kind} down", tp + 0.1))
    for ts, sec, why in plan.switches:
        k = plan.at(ts)
        cands.append(glow(plan.colour[plan.at(ts + 0.2)], plan.base[k], ts - trim_s,   # the colour it switches to
                          f"colour {why} ({sec})", ts))
    info = {"mode": mode, "not_representable_snare_flashes": len(plan.snares), "dropped_late": 0}
    if mode == "queue":
        return sorted(cands, key=lambda c: c.t_send), info
    calls, dropped = _one_in_flight(cands, lamp, motion_times, late_tolerance_s)
    info["dropped_late"] = dropped
    return calls, info


def motion_slots_for(song: Song, per_min: float = MOTION_PER_MIN, t0: float = 0.0) -> np.ndarray:
    """Evenly spaced motion.move times standing in for head tracking on the same SDK session (show-only
    runs; the full twin passes its real motion call times instead)."""
    if per_min <= 0:
        return np.zeros(0)
    step = 60.0 / per_min * (1 + 1e-9)
    return np.arange(t0 + step / 2, song.end + 2.0, step)


@dataclass
class BestSdkDesign:
    """The spec's recommended SDK light design (research report twin-spec/light.md section 6.3):

    Option A: colour switches on bar lines / sections, each ONE colour glow sent T/2 + admission_trim before
              the bar, so the 600 ms fade's midpoint (its steepest point) lands on the bar line.
    Option B: an "effect kick" for DROP (and bar downbeats in drop/outro when the budget allows): the
              "flowing" effect is the only near-instant SDK light change, sent admission_trim early, then a
              colour glow back to the base colour as a 600 ms decay. Two calls per hit.
    One call in flight, never queue; never send an effect while our colour fade runs (it would orphan it);
    every glow carries luminance. The light budget is what the session's rate limit leaves after head
    tracking; a reserve is kept for the DROP. There is no effect whose period can match the BPM: periods
    are fixed (breathing matches one breath per bar only at about 100.8 BPM), so none is used for beats.
    Colours are sent as dim pixel values at a high luminance, so the full-value cyan crescent of "flowing"
    is brighter than the pedestal and reads as a hit.
    """
    drive_luminance: float = 0.9     # ASSUMPTION (design choice)
    pedestal_px: float = 0.18        # ASSUMPTION: base colour pixel scale; crescent total light ~1.7x pedestal
    # How early a hit's POST goes out. None: the admission latency's trim_quantile plus the effect's first frame
    # (45 ms with the ASSUMED 10-40 ms admission and 5 ms first frame), so the visible change lands 0-30 ms
    # BEFORE the haptic whenever admission is not in its tail: the spec's light may lead, never lag. A
    # smaller trim lands late about half the time. Measure admission on the lamp (twin-spec/light.md 6.4).
    admission_trim_s: float | None = None
    trim_quantile: float = 0.95      # ASSUMPTION: the tail beyond it lands late; more trim would lead by > 30 ms
    hold_s: float = 0.090            # ASSUMPTION, inside the spec's 60-120 ms
    motion_per_min: float = MOTION_PER_MIN
    drop_reserve: int = 2
    effect_name: str = "flowing"
    downbeat_kicks: bool = True
    # False (live use): bars and sections are predicted from events as they ARRIVE, as the real conductor's
    # live capture allows. True: a known track's cue sheet (Song.bars, Song.sections): a what-if only.
    use_cue_sheet: bool = False
    palettes: dict = field(default_factory=lambda: {k: list(v) for k, v in PALETTES.items()})

    def trim_s(self, params: SdkLightParams) -> float:
        """The admission trim in use with these lamp parameters."""
        if self.admission_trim_s is not None:
            return float(self.admission_trim_s)
        return params.admission_quantile(self.trim_quantile) + params.effect_first_frame_s

    def _colour(self, section: str, step: int) -> np.ndarray:
        pal = self.palettes.get(section, self.palettes["groove"])
        return pal[step % len(pal)] * self.pedestal_px

    def calls(self, song: Song, lamp: SdkLamp, motion_times=()) -> tuple[list, dict]:
        T = lamp.p.transition_s
        trim = self.trim_s(lamp.p)
        budget = max(0, int(lamp.p.rate_limit_per_min - math.ceil(self.motion_per_min - 1e-9)))
        cue = self.use_cue_sheet and len(song.bars) > 0          # a known track's cue sheet (what-if)
        kicks_at = {round(e.t, 3) for e in song.events if e.kind == "KICK"}

        # Bar lines as (bar time, when it is known). Known track: known in advance. Otherwise predicted
        # causally from kick ARRIVALS; each bar is decided with what had arrived by its send time.
        bars = []
        if cue:
            bars = [(float(b), -1e9) for b in song.bars]
        else:
            tracker, arrivals = BeatTracker(), sorted((song.arrival(i), e) for i, e in enumerate(song.events))
            for j, (a, e) in enumerate(arrivals):
                if e.kind in ("KICK", "DROP"):
                    tracker.add(e.t)
                nxt = arrivals[j + 1][0] if j + 1 < len(arrivals) else a + 4.0
                adv = T / 2 + trim
                for b in tracker.bars_in(a + adv, nxt + adv):
                    if not bars or b - bars[-1][0] > 0.3:
                        bars.append((b, a))

        # Sections as the client can know them: the cue sheet, else from BUILD/DROP as they arrive.
        def section_for(b: float, known_by: float) -> str:
            if cue:
                return song.section_at(b + 1e-6)
            cur = "groove"
            for i, e in enumerate(song.events):
                if e.t > b or song.arrival(i) > known_by:
                    break
                if e.kind == "BUILD":
                    cur = "build"
                elif e.kind == "DROP":
                    cur = "drop" if b < e.t + 8.0 else "groove"     # 8 s drop section, ASSUMPTION
            return cur

        cands = []                                                    # (send, priority, kind, due, section)
        # The first colour: before the music on a known track, else as soon as the first event arrives.
        if cue:
            t_prime = min(bars[0][0], song.start) - 1.5
        else:
            t_prime = min(song.arrival(i) for i in range(len(song.events))) if song.events else 0.0
        cands.append((t_prime, 1, "prime", float("nan"), section_for(song.start, t_prime)))
        adv = T / 2 + trim
        for b, _known in bars:
            sec = section_for(b, b - adv)
            hit = self.downbeat_kicks and sec in ("drop", "outro") and round(b, 3) in kicks_at
            cands.append((b - (trim if hit else adv), 2, "kickbar" if hit else "bar", b, sec))
        for i, e in enumerate(song.events):
            if e.kind == "DROP":
                cands.append((max(song.arrival(i), e.t - trim), 0, "drop", e.t, "drop"))
        cands.sort(key=lambda c: (c[0], c[1]))

        calls, own, prev_sec, step, last_colour = [], [], None, 0, None
        info = {"budget_per_min": budget, "skipped_busy": 0, "skipped_budget": 0, "skipped_same_colour": 0,
                "drop_effect_skipped_busy": 0, "late_sends": 0, "admission_trim_s": round(trim, 4),
                "grid": "known track (cue sheet)" if cue else "causal BeatTracker on kick arrivals",
                "per_kick_light": "no: the SDK cannot pair a light hit with every kick (twin-spec/light.md 6.2)"}

        def remaining(s: float) -> int:
            return budget - sum(1 for x in own if s - 60.0 < x <= s)

        def busy(s: float) -> bool:
            if not calls:
                return False
            lamp.run(calls, motion_times)
            return any(not math.isnan(c.t_reply) and c.t_reply > s for c in calls)

        for s, _, kind, due, sec in cands:
            colour_step = step + 1 if sec == prev_sec else 0
            need = 1 if kind in ("bar", "prime") else 2
            # Budget floors, so that what matters most still has calls left (causal: no look-ahead needed).
            # DROP and the first colour take anything left; a change INTO build/break/drop keeps the DROP's
            # reserve; other section changes keep one more; bar steps and downbeat kicks keep two more.
            if kind in ("drop", "prime"):
                floor = 0
            elif sec != prev_sec:
                floor = self.drop_reserve + (0 if sec in ("build", "break", "drop") else 1)
            else:
                floor = self.drop_reserve + 2
            if kind == "drop" and remaining(s) < 2 and remaining(s) >= 1:
                need = 1                                              # no effect: just the colour change
            if remaining(s) < need + floor:
                info["skipped_budget"] += 1
                continue
            if busy(s):
                if kind != "drop":
                    info["skipped_busy"] += 1
                    continue
                # Our colour fade is still running: an effect now would cut it and orphan that call. The
                # DROP keeps its colour change, sent as soon as the lamp has answered.
                s, need = max(c.t_reply for c in calls if not math.isnan(c.t_reply)), 1
                info["drop_effect_skipped_busy"] += 1
            colour = self._colour(sec, colour_step)
            payload_colour = glow(colour, None, 0.0).payload["color"]
            if kind in ("bar", "prime") and payload_colour == last_colour:
                info["skipped_same_colour"] += 1              # nothing would change: keep the budget
                continue
            last_colour = payload_colour
            if kind in ("drop", "kickbar") and need == 2:
                calls.append(effect(self.effect_name, s, f"{kind}: effect kick", due))
                calls.append(glow(colour, self.drive_luminance, max(s, due) + self.hold_s,
                                  f"{kind}: decay to {sec}"))           # self-clocking, not aimed at a beat
                own += [s, max(s, due) + self.hold_s]
                info["late_sends"] += int(s > due - trim + 1e-9)
            else:
                calls.append(glow(colour, self.drive_luminance, s, f"{kind}: colour ({sec})",
                                  due if kind == "bar" else float("nan"), aim="mid"))
                own.append(s)
            prev_sec, step = sec, colour_step
        lamp.run(calls, motion_times)
        return calls, info


# ------------------------------------------------------------------ haptic lanes
@dataclass(frozen=True)
class PhoneLane:
    """An iPhone's Core Haptics transient. The phone schedules it at t - trim_s; it is felt output_latency_s
    later, give or take spread_s. Defaults: haptic output latency is NOT measured, so 0 with a perfect trim
    (ASSUMPTION); the spread is the +-2 ms band measured for acoustic clicks between two phones (Mac repo
    docs/field-notes-2026-09-15.md:23-24; audio, not haptics). For an uncorrected what-if use
    phone_audio_whatif()."""
    name: str = "phone"
    output_latency_s: float = 0.0
    trim_s: float = 0.0
    spread_s: float = 0.002
    kinds: tuple = ("CLICK", "KICK", "SNARE", "BASS", "BUILD", "DROP")

    def felt(self, events: list, rng=None) -> np.ndarray:
        rng = rng or np.random.default_rng(0)
        t = np.array([e.t for e in events], dtype=float)
        out = t - self.trim_s + self.output_latency_s + rng.uniform(-self.spread_s, self.spread_s, len(t))
        return np.where([e.kind in self.kinds for e in events], out, np.nan)


def phone_audio_whatif() -> PhoneLane:
    """What-if: haptics as late as the iPhone 17's acoustic clicks (p50 33.2 ms after masterTs, Mac repo
    docs/field-notes-2026-09-15.md:23) with no trim. Audio-derived, not a haptic measurement."""
    return PhoneLane(name="phone (audio-derived what-if, +33 ms)", output_latency_s=0.0332)


@dataclass(frozen=True)
class TitanLane:
    """TITAN Core over Bluetooth via the Mac's TitanSink (Mac conductor source Show/TitanSink.swift).

    The sample captured at pts is handed to Core Audio at pts + L - reported - trim (TitanSink.swift:9-11,
    150), never less than 35 ms after capture (:31, :156); it is felt reported + (the real extra delay)
    later. With a correct trim the thump lands on pts + L = t. Only KICK, SNARE and DROP make thumps
    (:127-135). reported 137.7 ms: Core Audio's value logged by the conductor 2026-09-19 (run.jsonl titan
    line), not a measurement of the board. trim 100 ms: that run's SETTING. Real Bluetooth delay to the
    board: NOT measured; `actual_extra_s=None` assumes the trim is right (ASSUMPTION). Jitter: +-1 ms
    measured 2026-09-19 with the built-in speakers as a stand-in (Mac repo AGENTS.md:79)."""
    name: str = "titan"
    latency_s: float = LATENCY_S
    reported_s: float = 0.1377
    trim_s: float = 0.100
    actual_extra_s: float | None = None
    jitter_s: float = 0.001
    min_behind_s: float = 0.035
    kinds: tuple = ("KICK", "SNARE", "DROP")

    @classmethod
    def from_song(cls, song: Song, **kw) -> "TitanLane":
        """Take L, reported and trim from a run log's titan line when there is one."""
        t = song.titan
        args = {"latency_s": song.latency_s}
        if "reportedMs" in t:
            args["reported_s"] = float(t["reportedMs"]) / 1e3
        if "trimMs" in t:
            args["trim_s"] = float(t["trimMs"]) / 1e3
        return cls(**(args | kw))

    @property
    def late_s(self) -> float:
        """How late every thump is because L is too small for reported + trim (TitanSink.swift:150-151)."""
        return max(0.0, self.reported_s + self.trim_s + self.min_behind_s - self.latency_s)

    def felt(self, events: list, rng=None) -> np.ndarray:
        rng = rng or np.random.default_rng(0)
        extra = self.trim_s if self.actual_extra_s is None else self.actual_extra_s
        t = np.array([e.t for e in events], dtype=float)
        pts = t - self.latency_s
        handoff = pts + max(self.latency_s - self.reported_s - self.trim_s, self.min_behind_s)
        out = handoff + self.reported_s + extra + rng.normal(0.0, self.jitter_s, len(t))
        return np.where([e.kind in self.kinds for e in events], out, np.nan)


# ------------------------------------------------------------------ sync and colour measurement
def _uv(rgb_linear: np.ndarray) -> np.ndarray:
    """CIE 1976 u'v' of linear sRGB colours (n, 3)."""
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = rgb_linear @ m.T
    d = xyz[:, 0] + 15 * xyz[:, 1] + 3 * xyz[:, 2]
    d = np.where(d <= 1e-12, np.nan, d)
    return np.stack([4 * xyz[:, 0] / d, 9 * xyz[:, 1] / d], axis=1)


def colour_switches(frames: list, *, min_dist: float = 0.04, settle_s: float = 0.15,
                    settle_tol: float = 0.008) -> list:
    """Times of hue changes of the perceived colour: a new SETTLED colour (its u'v' moved less than
    settle_tol over settle_s) more than min_dist in u'v' from the previous settled colour. The time
    reported is when the change was half done. min_dist 0.04 is several colour JNDs (ASSUMPTION);
    brief ring flashes and pulses do not settle, so they do not count."""
    if not frames:
        return []
    t, rgb = _perceived(frames)
    uv = _uv(rgb)
    for k in range(1, len(uv)):                         # dark frames keep the previous chromaticity
        if np.isnan(uv[k]).any():
            uv[k] = uv[k - 1]
    n = max(1, int(round(settle_s / max(1e-9, t[1] - t[0])))) if len(t) > 1 else 1
    if len(t) <= n:
        return []
    windows = np.lib.stride_tricks.sliding_window_view(uv, (n + 1, 2))[:, 0]    # frames k-n .. k, for k >= n
    spread = np.linalg.norm(windows - uv[n:, None, :], axis=2).max(axis=1)
    settled = np.flatnonzero(spread <= settle_tol) + n                           # NaN spread is never settled
    out, anchor, anchor_k = [], None, 0
    for k in settled:
        if anchor is None:
            anchor, anchor_k = uv[k], k
        elif float(np.hypot(*(uv[k] - anchor))) > min_dist:
            span = uv[anchor_k:k + 1]
            half = np.linalg.norm(span - anchor, axis=1) >= 0.5 * float(np.hypot(*(uv[k] - anchor)))
            out.append(float(t[anchor_k + int(np.argmax(half))]))
            anchor, anchor_k = uv[k], k
        else:
            anchor_k = k
    return out


def _stats_ms(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if not x.size:
        return {"n": 0}
    a = np.abs(x)
    return {"n": int(x.size), "median_ms": round(1e3 * float(np.median(x)), 2),
            "median_abs_ms": round(1e3 * float(np.median(a)), 2), "p95_abs_ms": round(1e3 * _q(a, 0.95), 2),
            "max_abs_ms": round(1e3 * float(a.max()), 2),
            "share_within_window": round(float(np.mean(a <= SYNC_WINDOW_S + 1e-9)), 4),
            "share_within_aim": round(float(np.mean(a <= SYNC_AIM_S + 1e-9)), 4),
            # the spec's rule: the light may lead, never lag (deaf-hoh-design.md:128-138)
            "share_light_not_late": round(float(np.mean(x <= 1e-9)), 4),
            "share_in_sync": round(float(np.mean((x <= 1e-9) & (x >= -SYNC_WINDOW_S - 1e-9))), 4)}


def light_rises(frames: list, *, rise_frac: float = 0.10, fine=None) -> tuple[np.ndarray, np.ndarray]:
    """Every rise of the panel's luminance: (t10, t50) per rise. A rise starts counting from the lowest
    level since the last fall; its onset (t10) is the first frame >= rise_frac of the show's maximum above
    that low, t50 when half of the rise to its peak is done. After a rise, the light must fall by the same
    amount before another rise can count.

    fine: a function t -> luminance (SdkRun.luminance_at). With it the onset is found to 1 ms between the
    frame before and the frame that crossed; without it the 60 Hz frames can only place it up to 16.7 ms
    late, which the lead-only rule would count as lag."""
    t, rgb = _perceived(frames)
    y = P.luminance(rgb)
    if y.size < 2 or float(y.max()) <= 0:
        return np.zeros(0), np.zeros(0)
    thresh = rise_frac * float(y.max())
    onsets, lows, ends = [], [], []
    low, armed, peak = float(y[0]), True, float(y[0])
    for k in range(1, len(y)):
        v = float(y[k])
        if armed:
            if v < low:
                low = v
            elif v - low >= thresh:
                onsets.append(k)
                lows.append(low)
                armed, peak = False, v
        else:
            if v > peak:
                peak = v
            elif peak - v >= thresh:                     # it fell back: the rise is over
                ends.append(k)
                armed, low = True, v
    ends += [len(y)] * (len(onsets) - len(ends))
    t10 = t[onsets] if onsets else np.zeros(0)
    if fine is not None and onsets:
        t10 = t10.copy()
        for i, (k, lo) in enumerate(zip(onsets, lows, strict=True)):
            grid = np.arange(t[k - 1], t[k] + 5e-4, 1e-3)
            above = [tk for tk in grid if fine(float(tk)) - lo >= thresh]
            t10[i] = min(float(above[0]), t[k]) if above else t[k]
    t50 = np.full(len(onsets), np.nan)
    for i, (k, lo, end) in enumerate(zip(onsets, lows, ends, strict=True)):
        seg = y[k:min(end, k + int(0.7 * FPS))]
        top = float(seg.max())
        t50[i] = t[k + int(np.argmax(seg >= lo + 0.5 * (top - lo)))]
    return t10, t50


def light_onsets(events: list, frames: list, *, before_s: float = 0.2, after_s: float = 0.6,
                 rise_frac: float = 0.10, fine=None) -> tuple[np.ndarray, np.ndarray, int]:
    """(t10, t50, unpaired) of the light's response to each event. Every rise of the light (light_rises)
    is credited to the NEAREST event in time, if it lies within [t - before_s, t + after_s] of it; an event
    takes the earliest rise credited to it; NaN = no light response. `unpaired` counts rises that belong to
    no event. Nearest-in-time is exact for the ideal panel; for a light that lags by more than half the gap
    between hits it credits the rise to the next hit, so SdkRun.summary() also reports the SDK model's
    true per-call lag from its call records."""
    r10, r50 = light_rises(frames, rise_frac=rise_frac, fine=fine)
    times = np.array([e.t for e in events], dtype=float)
    t10, t50 = np.full(len(events), np.nan), np.full(len(events), np.nan)
    unpaired = 0
    for r, h in zip(r10, r50, strict=True):
        j = int(np.searchsorted(times, r))
        cands = [i for i in (j - 1, j) if 0 <= i < len(times) and times[i] - before_s <= r <= times[i] + after_s]
        if not cands:
            unpaired += 1
            continue
        i = min(cands, key=lambda i: abs(r - times[i]))
        if np.isnan(t10[i]):
            t10[i], t50[i] = r, h
    return t10, t50, unpaired


def sync_report(events: list, light_frames: list, lanes: list, *, kinds=("KICK", "DROP"), seed: int = 0,
                fine=None) -> dict:
    """Light vs haptic for every event of `kinds`: per lane, offset = light onset - felt time (positive =
    the light lags). Onset = first frame where luminance rises by >= 10 % of the show's max towards the
    pulse (t10); t50 is also given. Plus colour switches per minute and the flash report of the frames.

    The window is +-30 ms, aim +-20 ms, light may lead never lag (visuo-tactile temporal-order JND 30 ms,
    Fujisaki & Nishida 2009; asynchrony first noticed at ~45 ms, Vogels 2004: research report
    deaf-hoh-design.md:128-138)."""
    sel = [e for e in events if e.kind in kinds]
    t10, t50, unpaired = light_onsets(sel, light_frames, fine=fine)
    rng = np.random.default_rng(seed)
    per_lane, per_event = {}, [{"t": round(e.t, 4), "kind": e.kind,
                                "light_t10": None if np.isnan(a) else round(float(a), 4),
                                "light_t50": None if np.isnan(b) else round(float(b), 4), "felt": {},
                                "offset_ms": {}}
                               for e, a, b in zip(sel, t10, t50, strict=True)]
    for lane in lanes:
        felt = lane.felt(sel, rng)
        off = t10 - felt
        st = _stats_ms(off)
        st["events"] = int(np.sum(~np.isnan(felt)))
        st["missed_light"] = int(np.sum(~np.isnan(felt) & np.isnan(t10)))
        st["t50_offset"] = _stats_ms(t50 - felt)
        per_lane[lane.name] = st
        for row, f, o in zip(per_event, felt, off, strict=True):
            row["felt"][lane.name] = None if np.isnan(f) else round(float(f), 4)
            row["offset_ms"][lane.name] = None if np.isnan(o) else round(1e3 * float(o), 2)
    switches = colour_switches(light_frames)
    dur = (light_frames[-1].t - light_frames[0].t) if len(light_frames) > 1 else 0.0
    # Colour switches per minute are only a steady-state number over at least one full 60 s rate window: a
    # shorter stretch can spend a whole minute's light budget in less than a minute.
    return {
        "kinds": list(kinds),
        "events": len(sel),
        "light_missed": int(np.sum(np.isnan(t10))),
        "light_rises_unpaired": unpaired,
        "window_ms": SYNC_WINDOW_S * 1e3, "aim_ms": SYNC_AIM_S * 1e3, "async_noticed_ms": ASYNC_NOTICED_S * 1e3,
        "window_source": "research report deaf-hoh-design.md:128-138 (Fujisaki & Nishida 2009; Vogels 2004)",
        "lanes": per_lane,
        "colour": {"switches": len(switches), "per_minute": round(60.0 * len(switches) / dur, 2) if dur else 0.0,
                   "seconds": round(dur, 2), "steady_state": bool(dur >= 60.0),
                   "max_in_any_60s": _max_in_window(switches, 60.0) if switches else 0,
                   "times": [round(x, 3) for x in switches]},
        "flash": flash_report(light_frames),
        "per_event": per_event,
    }


# ------------------------------------------------------------------ evidence helpers
def json_safe(x):
    """Plain JSON types; NaN and infinity become None (the replay tooling writes allow_nan=False)."""
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return json_safe(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    return x


def assumptions(design: LightDesign, sdk: SdkLightParams, best: BestSdkDesign, lanes: list) -> dict:
    """Every ASSUMPTION value this run used (not measured anywhere), for the evidence sidecar."""
    out = {
        "panel_index_order": P.INDEX_ORDER,
        "panel_layout": P.load_geometry().source,
        "event_lead_model": "L - (shifted exponential detection lag, median 37 ms, p90 58 ms), lead >= 156 ms",
        "sdk_admission_s": list(sdk.admission_s), "sdk_admission_tail_s": sdk.admission_tail_s,
        "sdk_admission_tail_p": sdk.admission_tail_p, "sdk_reply_s": sdk.reply_s,
        "sdk_effect_first_frame_s": sdk.effect_first_frame_s, "sdk_effect_overrun": sdk.effect_overrun,
        "best_drive_luminance": best.drive_luminance, "best_pedestal_px": best.pedestal_px,
        "best_admission_trim_s": best.trim_s(sdk), "best_hold_s": best.hold_s,
        "design_levels": {"pedestal": design.pedestal, "bass_depth": design.bass_depth,
                          "kick_depth": design.kick_depth, "snare_boost": design.snare_boost,
                          "break_level": design.break_level, "build_level": design.build_level,
                          "colour_fade_s": design.colour_fade_s},
        "palettes": "design choice, iso-luminant, see show._RAW_PALETTES",
    }
    for lane in lanes:
        if isinstance(lane, PhoneLane):
            out[f"{lane.name}_output_latency_s"] = lane.output_latency_s
        elif isinstance(lane, TitanLane):
            out[f"{lane.name}_actual_extra_s"] = "equals trim" if lane.actual_extra_s is None else lane.actual_extra_s
    return out


# ------------------------------------------------------------------ everything for one song
@dataclass
class ShowResult:
    song: Song
    plan: DesignPlan
    ideal_frames: list               # after the flash limiter (when it is available)
    naive_frames: list               # the ideal design sent literally through light.glow
    best_frames: list                # the spec's Option A + B through light.glow
    naive_run: SdkRun
    best_run: SdkRun
    evidence: dict


def run_show(song: Song, *, design: LightDesign | None = None, sdk: SdkLightParams | None = None,
             best: BestSdkDesign | None = None, lanes: list | None = None, motion_times=None,
             naive_mode: str = "queue", seed: int = 0, t0: float | None = None,
             t1: float | None = None) -> ShowResult:
    """The ideal design, the naive SDK path and the best SDK path for one song, and the evidence.

    Frames cover t0..t1 (default: 2 s before the first event to 1.5 s after the end). A real run log can
    hold long gaps (one 2026-09-19 log spans 54 min of presentation time for 5 min of music); use
    Song.window() to cut a stretch first, since every frame holds 93 pixels."""
    design = design or LightDesign()
    sdk = sdk or SdkLightParams(seed=seed)
    best = best or BestSdkDesign()
    lanes = lanes or [PhoneLane(), TitanLane.from_song(song)]
    motion = motion_slots_for(song, best.motion_per_min) if motion_times is None else np.asarray(motion_times)
    t0 = max(0.0, song.start - 2.0) if t0 is None else t0
    t1 = song.end + 1.5 if t1 is None else t1

    plan = design.plan(song, t0, t1)
    ideal, limit_info = apply_flash_limit(design.render(song, plan=plan))

    lamp = SdkLamp(sdk)
    naive_calls, naive_info = reactive_calls(song, plan, lamp, mode=naive_mode, motion_times=motion)
    naive_run = lamp.run(naive_calls, motion)
    naive = naive_run.frames(t0, t1)
    best_calls, best_info = best.calls(song, lamp, motion)
    best_run = lamp.run(best_calls, motion)
    best_frames = best_run.frames(t0, t1)

    def report(frames, fine=None):
        r = sync_report(song.events, frames, lanes, seed=seed, fine=fine)
        r.pop("per_event")
        r["onset_resolution"] = "1 ms (from the SDK model's state)" if fine else f"{1e3 / FPS:.1f} ms frames"
        return r

    # What-if: the same design with a known track's cue sheet (bars and sections known in advance). The live
    # conductor captures audio and has no cue sheet, so this is never the headline.
    known = None
    if len(song.bars) and not best.use_cue_sheet:
        known_design = dataclasses.replace(best, use_cue_sheet=True)
        known_calls, known_info = known_design.calls(song, lamp, motion)
        known_run = lamp.run(known_calls, motion)
        known = {"what": "WHAT-IF: sdk_best with a known track's cue sheet (bars and sections in advance); the "
                         "live conductor has none", "client": known_info, "lamp": known_run.summary(),
                 "sync": report(known_run.frames(t0, t1), known_run.luminance_at)}

    flash, how = load_flash()
    evidence = {
        "song": {"origin": song.origin, "events": len(song.events), "bpm": song.bpm, "latency_s": song.latency_s,
                 "sections": [(round(a, 3), round(b, 3), n) for a, b, n in song.sections],
                 "lead_ms_median": round(1e3 * float(np.median(song.lead_s)), 1) if len(song.lead_s) else None},
        "safety_flash": how,
        "ideal": {"what": "IDEAL per-pixel panel: design reference, NOT reachable through the SDK",
                  "grid": plan.grid, "pulses": len(plan.pulses), "skipped_pulses": len(plan.skipped),
                  "snare_ring_flashes": len(plan.snares), "colour_switches_planned": len(plan.switches),
                  "limiter": limit_info, "sync": report(ideal)},
        "sdk_naive": {"what": f"the ideal design sent literally through light.glow ({naive_mode})",
                      "client": naive_info, "lamp": naive_run.summary(), "sync": report(naive, naive_run.luminance_at)},
        "sdk_best": {"what": "Option A colour switches on bars + Option B effect kick on DROP (twin-spec/light.md "
                             "6.3); bars and sections " + ("from a cue sheet (what-if)" if best.use_cue_sheet and
                                                            len(song.bars) else "predicted live from event arrivals"),
                     "client": best_info, "lamp": best_run.summary(),
                     "sync": report(best_frames, best_run.luminance_at)},
        "sdk_best_known_track": known,
        "haptic_lanes": {lane.name: {k: v for k, v in vars(lane).items() if k != "kinds"} for lane in lanes},
        "assumptions": assumptions(design, sdk, best, lanes),
        "honesty": "simulation evidence only; no hardware approval. SDK timings marked ASSUMPTION are not "
                   "measured; the ideal panel cannot be shown by the real lamp.",
    }
    evidence = json_safe(evidence)
    return ShowResult(song=song, plan=plan, ideal_frames=ideal, naive_frames=naive, best_frames=best_frames,
                      naive_run=naive_run, best_run=best_run, evidence=evidence)
