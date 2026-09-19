#!/usr/bin/env python3
"""Offline generator + validator for beat-locked dance clips, v2 (run on the Mac, never on the lamp).

Why clips: single joint commands through the runtime reach only 40-60 % of the commanded delta on
base_yaw (75-99 % elsewhere) and a plan's last goal is not chased once the plan ends. Only a
continuous 30 fps stream -- a CSV clip played by the runtime's animation player -- moves the arm
faithfully. So the dance is a library of short clips, one per (tier, bpm, variant), that lamp_show's
ClipScheduler triggers so the pattern's beats fall on the music's beats.

Shape of every clip (FPS 30), unchanged from v1 because the scheduler depends on it:
    0.6 s hold at START  |  8 beats of pattern, poses landing ON k*60/bpm after the hold  |  0.6 s hold at START
Between two beats the arm holds 30 % of the beat, then a cosine-eased move takes the last 70 %, so it
arrives exactly on the next beat instant. Starting and ending at START (a neutral-ish pose) means a
re-trigger blends nothing: the runtime skips its blend-in when the first frames equal the current pose.

v2 -- what changed and why (the v1 dance was timid on the arm: at 132 bpm it gave base_yaw +-12..18):
* THREE variants per tier and bpm, beat_<tier>_<bpm>_<a|b|c>, that differ in choreography, not just
  amplitude (groove: sway+bob / figure-eight / nod-led; hype: wide sway with elbow pump / head circles /
  stabs; drop: spring + sweep, then the matching hype; build: a progressive crouch with shivers on a
  different joint). The scheduler rotates a -> b -> c so the audience never sees the same 8 beats twice
  in a row. The unsuffixed v1 names stay as ALIASES (an identical copy of variant a, listed in the
  manifest with "alias_of") so a v4 scheduler that only knows beat_<tier>_<bpm> keeps working.
* Bar accents: beat 1 of each bar (beats 1 and 5) moves x1.3; beat 5 adds a head flick (wrist_pitch up,
  wrist_roll). The drop's beat 1 is the spring itself (the spec's pose, exactly) so it carries no x1.3,
  and the build's crouch is a progressive ramp that carries neither the accent nor the flick.
* Amplitude per JOINT, not one A for all: every pattern is written at its nominal (envelope-sized)
  size and each joint's excursion from START is then scaled by the largest s_j <= 1 whose commanded
  (post-gain) trajectory keeps that joint <= 140 units/s -- the vendor simulation limit and the design
  ceiling (the SDK refuses > 300). Cosine easing peaks at pi/2 x the mean speed, so a one-beat move can
  cover at most 140 * 0.7 * 60/bpm / (pi/2) commanded units (47 at 80 bpm, 28 at 132, 21 at 180); the
  patterns with 4- and 8-beat sweeps get proportionally more travel from the same budget, which is why
  they exist.
* Command gains for the runtime's under-delivery live in one GAIN TABLE (GAINS) at the top of the file
  and can be overridden with --gains '{"base_yaw": 1.6, ...}' (a JSON object or @file) once the operator
  has measured ratios. Validation is ALWAYS on the commanded trajectory, and the envelope clamp runs after
  the gain.
* A "build" tier: over beats 1-6 the lamp crouches progressively (elbow first so the flip rule never
  binds: base_pitch -49 -> -60, elbow -22 -> -55, wrist_pitch 30 -> 15, small shivers on eighth notes),
  then releases over beats 7-8 back to START -- which IS the drop clip's first frame, so the drop that
  follows blends from nothing. The manifest documents the crouch pose ("crouch") next to first/last.

Safety envelope (non-negotiable, validated tonight on the URDF and on the arm): base_pitch >= -65 always;
base_pitch < -52 only with elbow_pitch <= -45 commanded (so the landed elbow is under -40; shoulder back
with the elbow not lifted is the flip region); |joint| <= 94; wrist_pitch <= 60 (it stops physically near
+94); peak speed <= 140 units/s; LampModel.problems() empty on every frame (table and base clearance);
ZMP: head_y >= +0.020 m and zmp_y >= -0.012 m on every frame with the 5-frame CoM smoothing of
analysis/clip_zmp.py; first and last frame == START. The envelope clamp here is continuous (it lifts
base_pitch's floor smoothly as the elbow comes down through -45..-58) so clamping never adds a speed
spike; it is a safety net -- the patterns are designed to sit inside the envelope, and the validator is
the final word. Failing clips are never written and a stale CSV of the same name is removed, so the out
dir always equals the manifest; the exit status is non-zero on any failure.

Writes are atomic (temp name in the same dir, fsync, os.replace, os.sync, md5 re-read) because a corrupt
CSV in the runtime's animations dir blocks the whole runtime at boot.

LIVE generation (the operator's "Bolder moves" slider): lamp_show's ClipScheduler calls make_clip() on the
lamp itself, at the tracker's exact bpm and the slider's `bold`, and writes the result into the runtime's
pack with write_clip_atomic(). bold maps to an amplitude multiplier m = 0.35 + 0.65 * bold on every joint's
excursion from START, applied BEFORE the per-joint speed-budget scale, so bold 1.0 is exactly the library
(byte-identical files when the bpm equals a bucket) and 0.0 is about a third of it; the budget, the gains,
the envelope clamp and the validation are the same code. Validator.for_lamp() builds the model from the
lamp's own vendor checkout and servo calibration (10 ms; a 145-frame validation takes ~57 ms on the Pi 5).

    .venv/bin/python beat_clips.py --one groove a 127.3 0.8 --out /tmp/x   # one live-style clip, exact bpm

    .venv/bin/python beat_clips.py                                  # all tiers, bpm 80..180 step 4, default out dir
    .venv/bin/python beat_clips.py --bpm 128 --tier drop --out /tmp/x --gains '{"base_yaw": 1.6}'

A partial run (any --tier/--bpm/--variant subset) merges its entries into the out dir's existing
MANIFEST.json (entries replaced by name, the rest kept while their CSV exists), so the scheduler never
sees a four-clip library after a quick regeneration; a full run rebuilds the manifest. Install: see
lamp-tools/README.md (install_clips.py copies the CSVs from ~/feelthemusic-lamp/beat_clips into the
runtime's animation pack atomically and md5-verifies them; MANIFEST.json stays in beat_clips/ where
lamp_show.py reads it).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spatial import JOINTS, LampModel, _rot_z  # noqa: E402

SCRATCH = Path("/private/tmp/claude-501/-Users-meharkhanna-feelthemusic/a2b4cfc6-081d-4df6-b3d6-864bf6cf8fa1/scratchpad")
DEFAULT_OUT = SCRATCH / "beat_clips"
DEFAULT_ROBOTDESC = Path(os.environ.get("LAMP_ROBOTDESC", SCRATCH / "robotdesc"))
LAMP_CALIBRATION = Path("/var/lib/lelamp/user-data/v1/calibration/lelamp.json")   # the lamp's own servo calibration

# ----------------------------------------------------------------------------- GAIN TABLE
# Command gain per joint: the runtime under-delivers step-like moves (base_yaw worst), so the commanded
# excursion from START is the pattern's excursion times this. Override with --gains once measured.
GAINS = {"base_yaw": 1.8, "base_pitch": 1.15, "elbow_pitch": 1.2, "wrist_roll": 1.3, "wrist_pitch": 1.1}

FPS = 30
HOLD_S = 0.6
BEATS = 8
MOVE_FRAC = 0.7                      # the last 70 % of each beat is the move; the first 30 % a hold
TIERS = ("groove", "hype", "drop", "build")
VARIANTS = ("a", "b", "c")
ALIAS_VARIANT = "a"                  # the unsuffixed v1 name is a copy of this variant
BPMS = tuple(range(80, 181, 4))
CSV_HEADER = "timestamp," + ",".join(f"{j}.pos" for j in JOINTS)
T0 = 1000.0                          # timestamps are absolute seconds, monotone

# joint order: base_yaw, base_pitch, elbow_pitch, wrist_roll, wrist_pitch
START = np.array([0.0, -49.0, -22.0, 0.0, 30.0])
GAIN = np.array([GAINS[j] for j in JOINTS])
SPEED_DESIGN = 140.0                 # per-joint amplitude is chosen against this (vendor simulation limit)
SPEED_LIMIT = 140.0                  # ... and the clip must pass this
SPEED_MARGIN = 0.99                  # design to 99 % of the limit so rounding never trips the validator
SCALE_STEP = 0.005
BOLD_MIN = 0.35                      # bold 0.0 -> 35 % of the library's excursion; bold 1.0 -> the library itself
BPM_RANGE = (40.0, 300.0)            # make_clip refuses tempi outside this (the tracker only reports 80-214)
JOINT_MAX = 94.0
BP_MIN, BP_FLIP, ELBOW_FLIP, WP_MAX = -65.0, -52.0, -45.0, 60.0
FLIP_BLEND = 13.0                    # elbow units over which the base_pitch floor drops from -52 to -65
HEAD_Y_MIN, ZMP_Y_MIN = 0.020, -0.012
G = 9.81
YAW, BP, EL, WR, WP = range(5)

# choreography sizes (RAW pattern units, i.e. what the arm should do; the gain is applied afterwards)
Y = 30.0                             # yaw sway: x1.8 -> 54 commanded, x1.3 accent -> 70, under the 94 box
R = 16.0                             # wrist_roll tilt
ACCENT = 1.3                         # beat 1 of every bar
ACCENT_BEATS = (1, 5)
FLICK_BEAT = 5                       # beat 1 of bar 2: an extra head flick
FLICK = np.array([0.0, 0.0, 0.0, 8.0, 10.0])
SPRING = np.array([0.0, -36.0, 2.0, 0.0, 50.0])       # drop beat 1 (spec numbers, raw)
CROUCH = np.array([0.0, -60.0, -55.0, 0.0, 15.0])     # build's deepest pose (raw)
SHIVER = {"a": {YAW: 4.0}, "b": {WR: 6.0, YAW: 2.0}, "c": {YAW: 4.0, WP: 3.0}}


# ----------------------------------------------------------------------------- pure pieces
def beat_period(bpm: float) -> float:
    return 60.0 / bpm


def beat_instants(bpm: float, hold_s: float = HOLD_S, beats: int = BEATS) -> np.ndarray:
    """Seconds from clip start at which pattern pose k (k = 0..beats) lands. Pose 0 is START."""
    return hold_s + np.arange(beats + 1) * beat_period(bpm)


def clip_seconds(bpm: float) -> float:
    return 2 * HOLD_S + BEATS * beat_period(bpm)


def max_move(bpm: float, limit: float = SPEED_DESIGN, move_frac: float = MOVE_FRAC) -> float:
    """Largest commanded one-beat excursion that a cosine-eased move keeps under `limit` units/s."""
    return limit * move_frac * beat_period(bpm) / (math.pi / 2)


def ease(u) -> np.ndarray:
    """Cosine ease-in-out on 0..1: zero speed at both ends, peak speed pi/2 times the average."""
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(math.pi * u)


def gain_vector(gains: dict | None = None) -> np.ndarray:
    """The gain table as a joint-ordered vector; `gains` overrides entries (unknown joints rejected)."""
    table = dict(GAINS)
    for j, g in (gains or {}).items():
        if j not in table:
            raise ValueError(f"unknown joint in gains: {j!r}")
        g = float(g)
        if not 0.2 <= g <= 4.0:
            raise ValueError(f"gain for {j} out of range: {g}")
        table[j] = g
    return np.array([table[j] for j in JOINTS])


def parse_gains(text: str | None) -> dict:
    """--gains '{"base_yaw": 1.6}' or --gains @measured.json."""
    if not text:
        return {}
    if text.startswith("@"):
        text = Path(text[1:]).read_text()
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("--gains must be a JSON object of joint -> gain")
    gain_vector(obj)                                    # validates
    return {k: float(v) for k, v in obj.items()}


def apply_gain(U: np.ndarray, gain: np.ndarray = GAIN, start: np.ndarray = START) -> np.ndarray:
    """Command gain on the excursion from START (START itself is unchanged)."""
    return start + gain * (np.asarray(U, dtype=float) - start)


def clamp_envelope(U: np.ndarray) -> np.ndarray:
    """Hard safety envelope, continuous in the input so it never adds a speed spike.

    base_pitch's floor is -52 while the elbow is above -45, and drops linearly to -65 as the elbow comes
    down to -58: everywhere the floor is below -52 the elbow is already <= -45, so base_pitch < -52 only
    ever happens with elbow <= -45 (and the landed elbow is under -40 even at 75 % delivery of a -45
    command). Then the joint box and the wrist_pitch mechanical stop.
    """
    V = np.array(U, dtype=float, copy=True)
    if V.ndim == 1:
        return clamp_envelope(V[None, :])[0]
    V = np.clip(V, -JOINT_MAX, JOINT_MAX)
    V[:, WP] = np.minimum(V[:, WP], WP_MAX)
    lowered = np.clip(ELBOW_FLIP - V[:, EL], 0.0, FLIP_BLEND)             # 0 when elbow > -45
    floor = BP_FLIP - (BP_FLIP - BP_MIN) * lowered / FLIP_BLEND          # -52 .. -65
    V[:, BP] = np.maximum(V[:, BP], floor)
    return V


def envelope_violations(U: np.ndarray) -> list[str]:
    U = np.asarray(U, dtype=float)
    out = []
    if (U[:, BP] < BP_MIN - 1e-9).any():
        out.append(f"base_pitch below {BP_MIN:+.0f} (min {U[:, BP].min():+.1f})")
    flip = (U[:, BP] < BP_FLIP - 1e-9) & (U[:, EL] > ELBOW_FLIP + 1e-9)
    if flip.any():
        out.append(f"flip region on {int(flip.sum())} frames (base_pitch < -52 with elbow > -45)")
    if (np.abs(U) > JOINT_MAX + 1e-9).any():
        out.append(f"a joint outside +-{JOINT_MAX:.0f}")
    if (U[:, WP] > WP_MAX + 1e-9).any():
        out.append(f"wrist_pitch above {WP_MAX:.0f}")
    return out


def peak_speed(U: np.ndarray, fps: float = FPS) -> np.ndarray:
    """Per-joint peak |delta| per second between consecutive frames (stricter than np.gradient)."""
    U = np.asarray(U, dtype=float)
    if len(U) < 2:
        return np.zeros(U.shape[1])
    return np.abs(np.diff(U, axis=0)).max(axis=0) * fps


def bold_multiplier(bold: float) -> float:
    """The "Bolder moves" slider as an excursion multiplier: BOLD_MIN + (1 - BOLD_MIN) * bold, clamped to
    0..1 on the way in. Exactly 1.0 at bold 1.0 (0.35 + 0.65 is 1.0 in binary floating point too), so the
    library clips are the bold-1.0 case and nothing else."""
    b = float(bold)
    b = 0.0 if b != b else min(1.0, max(0.0, b))          # NaN -> 0
    return BOLD_MIN + (1.0 - BOLD_MIN) * b


# ----------------------------------------------------------------------------- patterns
def _pose(yaw=0.0, bp=0.0, el=0.0, wr=None, wp=0.0, roll=-0.4) -> np.ndarray:
    """An excursion from START. wrist_roll defaults to a counter-tilt of the yaw (head stays level-ish)."""
    return np.array([yaw, bp, el, roll * yaw if wr is None else wr, wp], dtype=float)


UP = dict(bp=5.0, el=14.0, wp=12.0)        # head up and out
DOWN = dict(bp=-2.0, el=-14.0, wp=-12.0)   # head down and in (base_pitch stays >= -52 commanded)
FOLD = dict(bp=-8.0, el=-20.0, wp=-10.0)   # deeper: elbow -46 commanded lets base_pitch go to -58


def hype_excursions(variant: str) -> list[np.ndarray]:
    """Hype beats 1..7 as excursions from START (nominal size, no accents yet)."""
    s4 = [math.sin(k * math.pi / 2) for k in range(8)]                  # 4-beat sine: 0 +1 0 -1 ...
    c4 = [math.cos(k * math.pi / 2) for k in range(8)]
    if variant == "a":      # wide sway (4-beat sine) with the elbow pumping every beat
        return [_pose(yaw=1.2 * Y * s4[k], roll=-0.35, **(UP if k % 2 else FOLD)) for k in range(1, 8)]
    if variant == "b":      # head circles: yaw and pitch in quadrature (sin / cos over 4 beats)
        out = []
        for k in range(1, 8):
            pitch = c4[k]
            kw = UP if pitch > 0.5 else (DOWN if pitch < -0.5 else {})
            out.append(_pose(yaw=Y * s4[k], roll=-0.3, **kw))
        return out
    if variant == "c":      # stabs: a thrust on the beat, a small recoil, one two-beat hold
        recoil = _pose(0.0, -1.0, -6.0, 0.0, -4.0)
        return [_pose(Y, 4.0, 18.0, 0.3 * Y, 12.0),        # 1 stab right and up
                recoil,                                     # 2
                _pose(-Y, -2.0, -16.0, -0.3 * Y, -12.0),    # 3 stab left and down
                recoil,                                     # 4
                _pose(0.0, 8.0, 22.0, 0.0, 12.0),           # 5 big stab centre-up (held ...)
                _pose(0.0, 8.0, 22.0, 0.0, 12.0),           # 6 ... for two beats)
                _pose(Y, -2.0, -16.0, 0.3 * Y, -12.0)]      # 7 stab right and down
    raise ValueError(f"unknown variant {variant!r}")


def groove_excursions(variant: str) -> list[np.ndarray]:
    """Groove beats 1..7 as excursions from START."""
    if variant == "a":      # sway alternating each beat, wrist_roll counter-tilt, bob on odd beats
        bob, rise = dict(bp=-2.0, el=-10.0, wp=-6.0), dict(bp=2.0, el=6.0, wp=4.0)
        return [_pose(yaw=Y * (1.0 if k % 2 else -1.0), **(bob if k % 2 else rise)) for k in range(1, 8)]
    if variant == "b":      # figure-eight: yaw over 8 beats, wrist_roll over 4 (90 degrees apart)
        out = []
        for k in range(1, 8):
            yaw = Y * math.sin(k * math.pi / 4)
            wr = R * math.sin(k * math.pi / 2)
            kw = dict(el=8.0, wp=8.0) if k in (2, 6) else (dict(bp=-2.0, el=-8.0, wp=-6.0) if k == 4 else {})
            out.append(_pose(yaw=yaw, wr=wr, **kw))
        return out
    if variant == "c":      # nod-led: wrist_pitch nods every beat; the head looks right for two beats,
        # passes the centre on a nod, looks left for two, centre, right -- so a reversal never lands in
        # a single beat (a +0.8Y -> -0.8Y move would bind the yaw scale at half the size)
        yaws = [0.8, 0.8, 0.0, -0.8, -0.8, 0.0, 0.8]
        out = []
        for k in range(1, 8):
            nod = dict(el=4.0, wp=14.0) if k % 2 else dict(bp=-2.0, el=-8.0, wp=-12.0)
            out.append(_pose(yaw=Y * yaws[k - 1], roll=0.3, **nod))
        return out
    raise ValueError(f"unknown variant {variant!r}")


MIRROR = np.array([-1.0, 1.0, 1.0, -1.0, 1.0])    # the same move looking the other way (yaw and wrist_roll)
RECOIL = _pose(0.0, -2.0, -8.0, 0.0, -6.0)           # centre, head dipped: the spring's recoil


def drop_excursions(variant: str) -> list[np.ndarray]:
    """Drop: beat 1 the spring (the spec's pose); beats 2-4 a wide sweep that crosses the centre on beat 3
    (the spring's recoil) so no single beat carries more than Y of yaw (+Y -> -Y in one beat bound the
    yaw scale at half the size); beats 5-7 the matching hype, the sweep's direction chosen so the hype's
    first beat continues it. a sweeps left first and lands in hype a; b sweeps right first and plays
    hype b mirrored (its circle runs the other way); c sweeps right in a fold and lands in the stabs."""
    hype = hype_excursions(variant)
    if variant == "a":
        sign, roll, sweep, tail = -1.0, -0.4, UP, hype[4:7]
    elif variant == "b":
        sign, roll, sweep, tail = 1.0, 0.3, UP, [e * MIRROR for e in hype[4:7]]
    elif variant == "c":
        sign, roll, sweep, tail = 1.0, -0.4, FOLD, hype[4:7]
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return [SPRING - START,
            _pose(yaw=sign * Y, roll=roll, **sweep),
            RECOIL,
            _pose(yaw=-sign * Y, roll=roll, **sweep)] + tail


def build_excursions(variant: str) -> list[np.ndarray]:
    """Build: crouch progressively over beats 1-6 (elbow first, so base_pitch < -52 never meets an
    elbow above -45), release over beats 7-8. a sinks straight down; b leans (yaw and wrist_roll
    grow with the crouch, the head tilts as it sinks); c nods on the way down (wrist_pitch alternates
    around the ramp). The eighth-note shivers are added by shiver_overlay()."""
    el_f = [0.35, 0.60, 0.80, 0.90, 1.00, 1.00, 0.65]
    bp_f = [0.00, 0.15, 0.27, 0.50, 0.80, 1.00, 0.40]
    wp_f = [0.20, 0.40, 0.55, 0.70, 0.85, 1.00, 0.50]
    d = CROUCH - START
    out = []
    for i in range(7):
        yaw, wr, nod = 0.0, 0.0, 0.0
        if variant == "b":
            yaw, wr = -12.0 * el_f[i], 20.0 * el_f[i]
        elif variant == "c":      # looks left/right and nods on alternate beats while sinking
            yaw, nod = 9.0 * (1.0 if i % 2 == 0 else -1.0) * el_f[i], 10.0 * (1.0 if i % 2 == 0 else -1.0)
        out.append(np.array([yaw, d[BP] * bp_f[i], d[EL] * el_f[i], wr, d[WP] * wp_f[i] + nod]))
    return out


def keyframes(tier: str, variant: str = "a") -> np.ndarray:
    """(BEATS+1, 5) RAW poses, pose k landing on beat instant k. Pose 0 and pose BEATS are START.
    Bar accents (x1.3 on beats 1 and 5) and the beat-5 head flick are applied here, except on the
    drop's spring (already the accent, and the spec's exact pose) and the build tier, whose crouch is a
    progressive ramp and carries neither the accent nor the flick (a flick on beat 5 made it stutter)."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    exc = {"groove": groove_excursions, "hype": hype_excursions,
           "drop": drop_excursions, "build": build_excursions}[tier](variant)
    K = np.tile(START, (BEATS + 1, 1))
    for k, e in enumerate(exc, start=1):
        e = np.array(e, dtype=float)
        if tier not in ("build",) and k in ACCENT_BEATS and not (tier == "drop" and k == 1):
            e = e * ACCENT
        if k == FLICK_BEAT and tier != "build":
            e = e + FLICK
        K[k] = START + e
    return K


def shiver_overlay(tier: str, variant: str, bpm: float, t: np.ndarray, hold_s: float = HOLD_S) -> np.ndarray:
    """(n, 5) RAW additive wobble for the build tier: +-amplitude alternating on every eighth note
    (-cos at the beat rate: an extreme on each beat and each half beat), zero at START, windowed in
    over the first beat and out over the last so the clip still starts and ends exactly at START."""
    n = len(t)
    out = np.zeros((n, 5))
    if tier != "build":
        return out
    P = beat_period(bpm)
    x = (t - hold_s) / P                                          # beats since the pattern started
    inside = (x > 0) & (x < BEATS)
    w = np.where(x < 1.0, ease(x), np.where(x > BEATS - 1.0, ease(BEATS - x), 1.0)) * inside
    wave = -np.cos(2 * math.pi * x) * w
    for j, amp in SHIVER[variant].items():
        out[:, j] = amp * wave
    return out


def trajectory(keys: np.ndarray, bpm: float, fps: float = FPS, hold_s: float = HOLD_S,
               move_frac: float = MOVE_FRAC) -> np.ndarray:
    """Sample the piecewise hold-then-ease path through the keyframes at fps. Raw (pre-gain) units."""
    P = beat_period(bpm)
    n_beats = len(keys) - 1
    total = 2 * hold_s + n_beats * P
    n = int(round(total * fps))
    t = np.arange(n) / fps
    U = np.empty((n, keys.shape[1]))
    for i, ti in enumerate(t):
        x = (ti - hold_s) / P
        if x < 0 or x >= n_beats:
            U[i] = keys[0] if x < 0 else keys[-1]
            continue
        k = int(math.floor(x))
        u = x - k
        if u < 1.0 - move_frac:
            U[i] = keys[k]
        else:
            e = float(ease((u - (1.0 - move_frac)) / move_frac))
            U[i] = keys[k] + (keys[k + 1] - keys[k]) * e
    return U


def raw_trajectory(tier: str, bpm: float, variant: str = "a") -> np.ndarray:
    """The nominal-size pattern sampled at fps: keyframe path plus the build's shiver overlay."""
    U = trajectory(keyframes(tier, variant), bpm)
    t = np.arange(len(U)) / FPS
    return U + shiver_overlay(tier, variant, bpm, t)


def joint_scales(raw: np.ndarray, bpm: float, gain: np.ndarray = GAIN, limit: float = SPEED_DESIGN) -> np.ndarray:
    """Per joint, the largest s_j <= 1 (step 0.005) whose commanded excursion gain_j * s_j * (raw - START)
    stays <= limit * SPEED_MARGIN units/s. The trajectory is linear in the excursion, so the peak speed
    scales linearly and no search is needed; the envelope clamp afterwards can only slow a joint."""
    peak = peak_speed(apply_gain(raw, gain))
    s = np.ones(len(JOINTS))
    for j in range(len(JOINTS)):
        if peak[j] > 0:
            s[j] = min(1.0, math.floor(limit * SPEED_MARGIN / peak[j] / SCALE_STEP) * SCALE_STEP)
    return s


def scaled(raw: np.ndarray, scales: np.ndarray, start: np.ndarray = START) -> np.ndarray:
    return start + np.asarray(scales, dtype=float) * (np.asarray(raw, dtype=float) - start)


def bold_trajectory(tier: str, bpm: float, variant: str = "a", bold: float = 1.0) -> np.ndarray:
    """The nominal pattern with the slider's multiplier on every joint's excursion from START. Applied
    BEFORE the speed budget (joint_scales), so a bold clip is scaled down by the budget where it must be
    and a timid one is simply smaller. At bold 1.0 the multiplier is 1.0 and the raw trajectory is
    returned untouched -- not even a float round trip -- so the library stays byte-identical."""
    raw = raw_trajectory(tier, bpm, variant)
    m = bold_multiplier(bold)
    return raw if m == 1.0 else scaled(raw, m)


def commanded(tier: str, bpm: float, variant: str = "a", gain: np.ndarray = GAIN,
              scales: np.ndarray | None = None, bold: float = 1.0) -> np.ndarray:
    """The trajectory as the runtime will be asked to play it: bold, per-joint scale, gain, then the
    clamp. scales=None chooses them against the speed budget."""
    raw = bold_trajectory(tier, bpm, variant, bold)
    if scales is None:
        scales = joint_scales(raw, bpm, gain)
    return clamp_envelope(apply_gain(scaled(raw, scales), gain))


def build_clip(tier: str, bpm: float, variant: str = "a", gain: np.ndarray = GAIN,
               bold: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """(scales, commanded trajectory). The batch library is bold 1.0; the live path passes the slider."""
    raw = bold_trajectory(tier, bpm, variant, bold)
    scales = joint_scales(raw, bpm, gain)
    return scales, clamp_envelope(apply_gain(scaled(raw, scales), gain))


# ----------------------------------------------------------------------------- validation
class Validator:
    """LampModel.problems() on every frame + the ZMP criterion of analysis/clip_zmp.py."""

    def __init__(self, robotdesc: Path = DEFAULT_ROBOTDESC, *, model: LampModel | None = None,
                 robot_dir: Path | None = None):
        """From a robotdesc dir (the Mac: pi5_feetech_r1/ + lelamp-calibration.json read in place), or
        from an existing LampModel plus the robot dir its URDF inertials come from (the lamp)."""
        if model is not None:
            self.model = model
            self.robot = Path(robot_dir if robot_dir is not None else model.robot_dir)
        else:
            robotdesc = Path(robotdesc)
            self.robot = robotdesc / "pi5_feetech_r1"
            self.model = LampModel(self.robot, calibration=robotdesc / "lelamp-calibration.json")
        root = ET.parse(self.robot / "robot.urdf").getroot()
        self.inertials = []
        for link in root.findall("link"):
            it = link.find("inertial")
            if it is None:
                continue
            o = it.find("origin")
            self.inertials.append((float(it.find("mass").get("value")),
                                   np.array([float(v) for v in o.get("xyz").split()])))

    @classmethod
    def for_lamp(cls, robot_dir: Path | None = None, calibration: Path = LAMP_CALIBRATION) -> "Validator":
        """The validator lamp_show uses on the lamp itself: the vendor checkout at spatial.DEFAULT_ROBOT_DIR
        and the lamp's own servo calibration (the calibrated span IS the physical span). Loads in ~10 ms."""
        from spatial import DEFAULT_ROBOT_DIR
        robot_dir = Path(robot_dir if robot_dir is not None else DEFAULT_ROBOT_DIR)
        return cls(model=LampModel(robot_dir, calibration=Path(calibration)), robot_dir=robot_dir)

    def com_and_head(self, u) -> tuple[np.ndarray, np.ndarray]:
        model = self.model
        m, acc, tot, fs = np.eye(4), np.zeros(3), 0.0, [np.eye(4)]
        for (motor, offset, origin), value in zip(model._chain, u):
            m = m @ origin @ _rot_z((float(value) - model.neutral[motor]) * model._scale[motor] + offset)
            fs.append(m.copy())
        for (mass, local), f in zip(self.inertials, fs):
            acc += mass * (f @ np.append(local, 1.0))[:3]
            tot += mass
        return acc / tot, fs[-1][:3, 3]

    def zmp(self, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(head_y, zmp_y) per frame, exactly as clip_zmp.py computes them (5-frame CoM smoothing,
        np.gradient twice, ZMP = CoM_xy - z_com / g * a_xy)."""
        coms, heads = zip(*(self.com_and_head(u) for u in U))
        coms, heads = np.array(coms), np.array(heads)
        k = np.ones(5) / 5
        sm = np.stack([np.convolve(coms[:, i], k, mode="same") for i in range(3)], 1)
        vel = np.gradient(sm, 1 / FPS, axis=0)
        acc = np.gradient(vel, 1 / FPS, axis=0)
        zmp_y = sm[:, 1] - (sm[:, 2] / G) * acc[:, 1]
        return heads[:, 1], zmp_y

    def validate(self, U: np.ndarray) -> dict:
        U = np.asarray(U, dtype=float)
        reasons = envelope_violations(U)
        if not (np.allclose(U[0], START, atol=1e-6) and np.allclose(U[-1], START, atol=1e-6)):
            reasons.append("first/last frame is not the START pose")
        bad = [i for i, u in enumerate(U) if self.model.problems(dict(zip(JOINTS, u)))]
        if bad:
            reasons.append(f"LampModel.problems on {len(bad)} frames, e.g. frame {bad[0]}: "
                           + "; ".join(self.model.problems(dict(zip(JOINTS, U[bad[0]])))))
        speed = peak_speed(U)
        if speed.max() > SPEED_LIMIT:
            reasons.append(f"peak speed {speed.max():.0f} > {SPEED_LIMIT:.0f} units/s")
        head_y, zmp_y = self.zmp(U)
        if head_y.min() < HEAD_Y_MIN:
            reasons.append(f"head_y min {head_y.min():+.3f} < {HEAD_Y_MIN:+.3f}")
        if zmp_y.min() < ZMP_Y_MIN:
            reasons.append(f"zmp_y min {zmp_y.min():+.3f} < {ZMP_Y_MIN:+.3f}")
        return {"ok": not reasons, "reasons": reasons, "head_y_min": float(head_y.min()),
                "zmp_y_min": float(zmp_y.min()), "peak_speed": speed, "problem_frames": len(bad)}


def validate_rows(rows, model) -> tuple[bool, dict]:
    """(ok, report) for a clip given as rows (list of 5-tuples, or an (n, 5) array) -- the batch
    generator's exact checks: envelope, START at both ends, LampModel.problems() per frame, peak speed,
    ZMP. `model` is a Validator, or a LampModel whose robot_dir holds the URDF. The report is plain
    Python (json-able) so the scheduler can log it."""
    v = model if isinstance(model, Validator) else Validator(model=model)
    U = np.asarray(rows, dtype=float)
    if U.ndim != 2 or U.shape[1] != len(JOINTS) or len(U) < 2:
        return False, {"ok": False, "reasons": [f"rows must be (n >= 2, {len(JOINTS)}), got {U.shape}"]}
    r = v.validate(U)
    report = {"ok": r["ok"], "reasons": list(r["reasons"]), "frames": int(len(U)),
              "head_y_min": round(r["head_y_min"], 4), "zmp_y_min": round(r["zmp_y_min"], 4),
              "peak_speed": {j: round(float(sp), 1) for j, sp in zip(JOINTS, r["peak_speed"])},
              "problem_frames": int(r["problem_frames"])}
    return bool(r["ok"]), report


def make_clip(tier: str, variant: str, bpm: float, bold: float, gains: dict | None = GAINS,
              model=None) -> tuple[list[tuple], dict]:
    """ONE clip at an exact bpm and boldness, as (rows, meta): rows are 30 fps 5-tuples of commanded
    joint values (csv_text() turns them into the runtime's CSV), meta describes it. Same maths as the
    batch generator (build_clip), so bold 1.0 at a bucket bpm reproduces the library file byte for byte.
    With `model` (a Validator or a LampModel) the clip is validated and meta carries ok/reasons/report;
    without it meta["ok"] is None and the caller must validate before the file reaches the runtime."""
    bpm = float(bpm)
    if not (BPM_RANGE[0] <= bpm <= BPM_RANGE[1]):
        raise ValueError(f"bpm {bpm} outside {BPM_RANGE[0]:.0f}..{BPM_RANGE[1]:.0f}")
    gain = gain_vector(gains)
    scales, U = build_clip(tier, bpm, variant, gain, bold)
    rows = [tuple(float(x) for x in u) for u in U]
    meta = {"tier": tier, "variant": variant, "bpm": bpm, "bold": float(bold), "multiplier": bold_multiplier(bold),
            "frames": int(len(U)), "seconds": round(len(U) / FPS, 4), "first_beat_s": HOLD_S + beat_period(bpm),
            "beats": BEATS, "amplitude": round(float(np.abs(U[:, YAW]).max()), 2),
            "scale": {j: round(float(sc), 3) for j, sc in zip(JOINTS, scales)},
            "gain": dict(zip(JOINTS, gain.tolist())),
            "range": {j: [round(float(lo), 4), round(float(hi), 4)] for j, lo, hi in zip(JOINTS, U.min(axis=0), U.max(axis=0))},
            "first": _pose_dict(U[0]), "last": _pose_dict(U[-1]), "ok": None, "reasons": []}
    if model is not None:
        ok, report = validate_rows(U, model)
        meta.update(ok=ok, reasons=report["reasons"], report=report)
    return rows, meta


# ----------------------------------------------------------------------------- files
def csv_text(U: np.ndarray, t0: float = T0, fps: float = FPS) -> str:
    lines = [CSV_HEADER]
    for i, u in enumerate(U):
        lines.append(f"{t0 + i / fps:.6f}," + ",".join(f"{v:.4f}" for v in u))
    return "\n".join(lines) + "\n"


def write_atomic(path: Path, text: str) -> str:
    """Temp name in the same dir (starting with '.', so the runtime never lists it), fsync, os.replace,
    os.sync, then re-read and md5-verify against the bytes written. Returns the md5. A failure anywhere
    leaves neither the temp file nor a half-written target: the runtime's animations dir must only ever
    hold complete CSVs (a corrupt one blocks the whole runtime at boot)."""
    path = Path(path)
    data = text.encode()
    want = hashlib.md5(data).hexdigest()
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    if hasattr(os, "sync"):
        os.sync()
    got = hashlib.md5(path.read_bytes()).hexdigest()
    if got != want:
        try:
            os.unlink(path)                                  # not what we wrote: it must not be played
        except OSError:
            pass
        raise IOError(f"{path}: md5 mismatch after write ({got} != {want})")
    return want


def write_clip_atomic(path: Path, rows) -> str:
    """A clip (rows from make_clip) into the runtime's pack dir, atomically; returns the md5 of the
    bytes written (and re-read). The caller checks the runtime lists the name before posting it."""
    return write_atomic(path, csv_text(rows))


def one_name(tier: str, variant: str, bpm: float, bold: float) -> str:
    """File stem for a --one clip: live_<tier>_<variant>_<bpm>_<bold> with '.' -> 'p' (the runtime plays
    by stem, so the stem must not look like an extension)."""
    return f"live_{tier}_{variant}_{bpm:g}_{bold:.2f}".replace(".", "p")


def remove_stale(path: Path) -> bool:
    """Drop a CSV of this name left by an earlier run, so the out dir always equals the manifest
    (a failing clip must not survive as a stale file that a glob copy would ship). True if removed."""
    path = Path(path)
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    if hasattr(os, "sync"):
        os.sync()
    return True


def base_name(tier: str, bpm: int) -> str:
    """The v1 (unsuffixed) name; in v2 an alias of variant a."""
    return f"beat_{tier}_{bpm}"


def clip_name(tier: str, bpm: int, variant: str = "a") -> str:
    return f"{base_name(tier, bpm)}_{variant}"


def _pose_dict(u) -> dict:
    return {j: round(float(x), 4) for j, x in zip(JOINTS, u)}


def generate(out: Path, tiers=TIERS, bpms=BPMS, robotdesc: Path = DEFAULT_ROBOTDESC,
             log=print, variants=VARIANTS, gains: dict | None = None,
             aliases: bool = True) -> tuple[list[dict], list[str]]:
    """Generate, validate and write every clip; returns (manifest entries, failure descriptions).
    With `aliases`, the unsuffixed v1 name is written as an identical copy of variant a (listed with
    "alias_of") so an older scheduler and install_clips.py keep working from the same manifest. When
    variant a is not part of this run the alias is left alone (neither written nor removed); when a
    was generated and failed, the stale alias goes with it."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    gain = gain_vector(gains)
    validator = Validator(robotdesc)
    entries, failures = [], []
    head = (f"{'clip':<20}{'frames':>7}{'sec':>6}{'A':>6}  {'scale y/bp/el/wr/wp':<26}{'head_y':>7}{'zmp_y':>7}  "
            + "".join(f"{j[:5]:>6}" for j in JOINTS) + "  verdict")
    log(head)
    for tier in tiers:
        for bpm in bpms:
            variant_a = None
            for variant in variants:
                name = clip_name(tier, bpm, variant)
                scales, U = build_clip(tier, bpm, variant, gain)
                v = validator.validate(U)
                A = float(np.abs(U[:, YAW]).max())
                row = (f"{name:<20}{len(U):>7}{len(U) / FPS:>6.2f}{A:>6.1f}  "
                       + f"{'/'.join(f'{s:.2f}' for s in scales):<26}{v['head_y_min']:>+7.3f}{v['zmp_y_min']:>+7.3f}  "
                       + "".join(f"{sp:>6.0f}" for sp in v["peak_speed"]))
                if not v["ok"]:
                    failures.append(f"{name}: " + "; ".join(v["reasons"]))
                    log(row + "  FAIL  " + "; ".join(v["reasons"]))
                    remove_stale(out / f"{name}.csv")
                    continue
                text = csv_text(U)
                md5 = write_atomic(out / f"{name}.csv", text)
                log(row + "  PASS")
                e = {"name": name, "base": base_name(tier, bpm), "variant": variant, "bpm": bpm, "tier": tier,
                     "frames": int(len(U)), "md5": md5, "seconds": round(len(U) / FPS, 4), "amplitude": round(A, 2),
                     "scale": {j: round(float(s), 3) for j, s in zip(JOINTS, scales)},
                     "first": _pose_dict(U[0]), "last": _pose_dict(U[-1]),
                     "range": {j: [round(float(lo), 4), round(float(hi), 4)]
                               for j, lo, hi in zip(JOINTS, U.min(axis=0), U.max(axis=0))},
                     "first_beat_s": HOLD_S + beat_period(bpm), "beats": BEATS,
                     "accent_beats": [] if tier == "build" else ([5] if tier == "drop" else list(ACCENT_BEATS)),
                     "flick_beat": None if tier == "build" else FLICK_BEAT,
                     "head_y_min": round(v["head_y_min"], 4), "zmp_y_min": round(v["zmp_y_min"], 4),
                     "peak_speed": {j: round(float(sp), 1) for j, sp in zip(JOINTS, v["peak_speed"])}}
                if tier == "build":
                    i6 = int(round(beat_instants(bpm)[6] * FPS))
                    e["crouch"] = _pose_dict(U[i6])
                    e["note"] = ("crouch deepest on beat 6, released over beats 7-8 back to START; "
                                 "the drop clip's first frame is START, so it blends from nothing")
                entries.append(e)
                if variant == ALIAS_VARIANT:
                    variant_a = (name, text, md5)
            if aliases and ALIAS_VARIANT in variants:
                alias = base_name(tier, bpm)
                if variant_a is None:
                    remove_stale(out / f"{alias}.csv")
                    continue
                target, text, md5 = variant_a
                got = write_atomic(out / f"{alias}.csv", text)
                assert got == md5
                a_entry = next(e for e in entries if e["name"] == target)
                entries.append({**a_entry, "name": alias, "alias_of": target, "variant": None})
    return entries, failures


def run_names(tiers=TIERS, bpms=BPMS, variants=VARIANTS, aliases: bool = True) -> set[str]:
    """Every clip name a generate() call with these arguments writes or removes (its aliases included
    when variant a is part of the run): the manifest entries a partial run replaces."""
    names = {clip_name(t, b, v) for t in tiers for b in bpms for v in variants}
    if aliases and ALIAS_VARIANT in variants:
        names |= {base_name(t, b) for t in tiers for b in bpms}
    return names


def read_manifest(out: Path) -> dict | None:
    """The MANIFEST.json already in `out` (a v2 one with a clips list), or None."""
    try:
        m = json.loads((Path(out) / "MANIFEST.json").read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(m, dict) or m.get("version") != 2 or not isinstance(m.get("clips"), list):
        return None
    return m


def write_manifest(out: Path, entries: list[dict], gains: dict | None = None,
                   replace: set[str] | None = None, log=print) -> Path:
    """Write MANIFEST.json for `entries`. With `replace` (the names a PARTIAL run touched, from
    run_names()) the existing manifest's other entries are kept -- provided their CSV is still in the
    out dir -- so `--tier drop --bpm 128` refreshes four entries instead of leaving the scheduler with
    a four-clip library and no build tier. Without `replace` the manifest is rebuilt from `entries`."""
    out = Path(out)
    gain = gain_vector(gains)
    gain_d = dict(zip(JOINTS, gain.tolist()))
    entries = [{**e, "gain": e.get("gain", gain_d)} for e in entries]
    if replace is not None:
        old = read_manifest(out)
        new_names = {e["name"] for e in entries}
        kept, dropped = [], 0
        for c in (old or {}).get("clips", []):
            if not isinstance(c, dict) or not c.get("name") or c["name"] in replace or c["name"] in new_names:
                continue
            if not (out / f"{c['name']}.csv").exists():
                dropped += 1
                continue
            kept.append(c)
        if old is not None and old.get("gain") != gain_d and kept:
            log(f"manifest: merging into a library generated with gains {old.get('gain')} (this run: {gain_d}); "
                "the per-entry \"gain\" says which is which")
        log(f"manifest: partial run, {len(entries)} entries replaced, {len(kept)} kept from the existing manifest"
            + (f", {dropped} dropped (CSV missing)" if dropped else ""))
        entries = kept + entries
    real = [e for e in entries if not e.get("alias_of")]
    manifest = {"version": 2, "generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "fps": FPS, "hold_s": HOLD_S,
                "beats": BEATS, "move_frac": MOVE_FRAC, "gain": gain_d,
                "speed_limit": SPEED_LIMIT, "start_pose": dict(zip(JOINTS, START.tolist())),
                "tiers": sorted({e["tier"] for e in real}, key=TIERS.index), "variants": list(VARIANTS),
                "accent": ACCENT, "accent_beats": list(ACCENT_BEATS), "flick_beat": FLICK_BEAT,
                "aliases": {e["name"]: e["alias_of"] for e in entries if e.get("alias_of")},
                "clips": sorted(entries, key=lambda e: (TIERS.index(e["tier"]), e["bpm"], e.get("variant") or ""))}
    path = out / "MANIFEST.json"
    write_atomic(path, json.dumps(manifest, indent=1) + "\n")
    return path


def one(spec: list[str], out: Path, robotdesc: Path, gains: dict | None = None, log=print) -> int:
    """--one TIER VARIANT BPM BOLD: make_clip + validate + write_clip_atomic, one table row, exit status."""
    tier, variant, bpm, bold = spec[0], spec[1], float(spec[2]), float(spec[3])
    if tier not in TIERS or variant not in VARIANTS:
        log(f"--one: tier must be one of {TIERS} and variant one of {VARIANTS}")
        return 2
    rows, meta = make_clip(tier, variant, bpm, bold, {**GAINS, **(gains or {})}, Validator(robotdesc))
    name = one_name(tier, variant, bpm, bold)
    r = meta["report"]
    scales = "/".join(f"{meta['scale'][j]:.2f}" for j in JOINTS)
    speeds = "".join(f"{r['peak_speed'][j]:>6.0f}" for j in JOINTS)
    row = (f"{name:<28}{meta['frames']:>7}{meta['seconds']:>6.2f}{meta['amplitude']:>6.1f}  x{meta['multiplier']:.2f}  "
           f"{scales:<26}{r['head_y_min']:>+7.3f}{r['zmp_y_min']:>+7.3f}  {speeds}")
    if not meta["ok"]:
        log(row + "  FAIL  " + "; ".join(meta["reasons"]))
        return 1
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    md5 = write_clip_atomic(out / f"{name}.csv", rows)
    log(row + f"  PASS  {out / name}.csv md5 {md5}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--robotdesc", type=Path, default=DEFAULT_ROBOTDESC,
                    help="dir holding pi5_feetech_r1/ and lelamp-calibration.json (never copied)")
    ap.add_argument("--bpm", type=int, nargs="*", default=list(BPMS))
    ap.add_argument("--tier", nargs="*", default=list(TIERS), choices=TIERS)
    ap.add_argument("--variant", nargs="*", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--gains", help='per-joint command gain override, JSON object or @file, e.g. \'{"base_yaw": 1.6}\'')
    ap.add_argument("--no-aliases", action="store_true", help="do not write the unsuffixed v1 names")
    ap.add_argument("--one", nargs=4, metavar=("TIER", "VARIANT", "BPM", "BOLD"),
                    help="ONE clip at an exact bpm and boldness (what lamp_show generates live), written to "
                         "--out as live_<tier>_<variant>_<bpm>_<bold>.csv; no manifest")
    args = ap.parse_args(argv)
    if not (args.robotdesc / "pi5_feetech_r1" / "robot.urdf").exists():
        print(f"robot description not found under {args.robotdesc}", file=sys.stderr)
        return 2
    try:
        gains = parse_gains(args.gains)
    except (ValueError, OSError) as exc:
        print(f"--gains: {exc}", file=sys.stderr)
        return 2
    if args.one:
        return one(args.one, args.out, args.robotdesc, gains)
    print("gains: " + ", ".join(f"{j} x{g:.2f}" for j, g in zip(JOINTS, gain_vector(gains))))
    entries, failures = generate(args.out, args.tier, args.bpm, args.robotdesc, variants=args.variant,
                                 gains=gains, aliases=not args.no_aliases)
    full = set(args.tier) == set(TIERS) and set(args.bpm) == set(BPMS) and set(args.variant) == set(VARIANTS)
    replace = None if full else run_names(args.tier, args.bpm, args.variant, not args.no_aliases)
    path = write_manifest(args.out, entries, gains, replace=replace)
    real = sum(1 for e in entries if not e.get("alias_of"))
    print(f"\n{real} clips (+{len(entries) - real} aliases) written to {args.out}, manifest {path}")
    if failures:
        print(f"{len(failures)} clip(s) FAILED and were not written:", file=sys.stderr)
        for f in failures:
            print("  " + f, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
