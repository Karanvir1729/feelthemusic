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
  amplitude (groove: sway+bob / figure-eight / nod-led; hype: wide sway with elbow pump / bottom-up /
  crossing diagonals; drop: spring + sweep, then the matching hype; build: a progressive crouch with shivers on a
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
ZMP: head_y >= +0.020 m and -0.012 m <= zmp_y <= +0.050 m on every frame with the 5-frame CoM smoothing
of analysis/clip_zmp.py (the forward bound is v4's: the dance now reaches the arm out over the table, so
that is a direction the lamp can tip in); first and last frame == START. The envelope clamp here is continuous (it lifts
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

FACING (dancing AT the person, not at the lamp's home heading): build_clip and make_clip take `facing` in
base_yaw's own joint units -- follow.py's flashlight/phone tracker is what knows which way the person with
the light is -- and it is added to the COMMANDED base_yaw column after the gain (the gain corrects the
runtime's under-delivery of a MOVE, and a constant offset is not a move), then clamped and validated like
any other command. facing 0.0 is the home-facing dance byte for byte. The turn is first reduced to what
the clip's own yaw excursion leaves inside the +-94 box (facing_limit), so it rotates the choreography
instead of flattening it against the box; validation then runs on the ROTATED trajectory, because turning
the arm carries the head sideways over the table and that is what the base clearance and the ZMP see. A
turned clip starts and ends at base_yaw = facing instead of 0, which is what lets consecutive clips at the
same facing still chain with nothing for the runtime to blend. Measured 2026-09-19 against this servo
calibration, every tier/variant/tempo validates out to +-25 units; the first to fail is drop c at 80 bpm
at +30, whose head comes within 0.017 m of the base against the 0.020 m floor -- so a caller that turns
the dance must check meta["ok"], exactly as the live path already does.

    .venv/bin/python beat_clips.py --one groove a 127.3 0.8 --out /tmp/x   # one live-style clip, exact bpm
    .venv/bin/python beat_clips.py --one groove a 127.3 0.8 --facing 20 --out /tmp/x   # ... danced turned

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
MOVE_FRAC = 0.85                     # the last 85 % of each beat is the move; the first 15 % a hold. The
                                     # excursion a beat can buy is proportional to this, so a longer move per
                                     # beat is free amplitude; the hold is only there to punctuate the beat.
TIERS = ("groove", "hype", "drop", "build")
VARIANTS = ("a", "b", "c")
ALIAS_VARIANT = "a"                  # the unsuffixed v1 name is a copy of this variant
BPMS = tuple(range(80, 181, 4))
CSV_HEADER = "timestamp," + ",".join(f"{j}.pos" for j in JOINTS)
T0 = 1000.0                          # timestamps are absolute seconds, monotone

# joint order: base_yaw, base_pitch, elbow_pitch, wrist_roll, wrist_pitch
START = np.array([0.0, -49.0, -22.0, 0.0, 30.0])
GAIN = np.array([GAINS[j] for j in JOINTS])
SPEED_DESIGN = 380.0                 # per-joint amplitude is chosen against this. 380 is what the VENDOR's own
                                     # animation clips reach, so it is the runtime's demonstrated ceiling, not a
                                     # guess; the servos deliver about half of a command at beat rates anyway.
SPEED_LIMIT = 380.0                  # ... and the clip must pass this
SPEED_MARGIN = 0.99                  # design to 99 % of the limit so rounding never trips the validator
SCALE_STEP = 0.005
BOLD_MIN = 0.35                      # bold 0.0 -> 35 % of the library's excursion; bold 1.0 -> the library itself
BPM_RANGE = (40.0, 300.0)            # make_clip refuses tempi outside this (the tracker only reports 80-214)
JOINT_MAX = 98.0                     # the servo's calibrated range is +-100; this is the margin to it
BP_MIN, BP_FLIP, ELBOW_FLIP, WP_MAX = -65.0, -52.0, -45.0, 60.0
FLIP_BLEND = 13.0                    # elbow units over which the base_pitch floor drops from -52 to -65
HEAD_Y_MIN, ZMP_Y_MIN = 0.020, -0.012
ZMP_Y_MAX = 0.050                    # ... and forward, half the radius of the vendor base cylinder the
                                     # model already carries (safety.yaml cylinder_radius_m 0.10, read as
                                     # LampModel.base["radius"]). v3 never leaned forward so it needed no
                                     # forward bound; v4 reaches the arm out over the table (REACH) to buy
                                     # sideways travel, and that is the side the lamp can now tip toward.
G = 9.81
YAW, BP, EL, WR, WP = range(5)

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


# ----------------------------------------------------------------------------- choreography (v4, wide)
# Written in COMMANDED units against the lamp's own servo calibration: LampModel maps each servo's calibrated
# range_min..range_max to -100..100, the safety box is +-98, and the model's rules (table, the lamp's own
# base, the base_pitch floor, ZMP) were scanned at the calibrated scales on 2026-09-19: the lowest safe head
# is the shoulder on its -65 floor with the elbow folded to -98 (0.165 m), the highest is the elbow straight
# up at +94 with the shoulder near -4 (0.435 m); START is 0.324 m. Every phrase is a LIFT profile on the
# beats (0 = BOTTOM, 1 = TOP) plus a yaw layer, and both are sized by the TEMPO: a joint may move at most
# max_move(bpm) in one beat (the speed budget), so a slow song sweeps the whole range in a beat or two and a
# fast one takes three or four beats per sweep -- the full range either way, and every landing on a beat.
#
# v4 -- the dance was tall but narrow. Measured through LampModel.head over the commanded trajectory, v3's
# head spanned 0.271 m vertically (0.167..0.438 against a reachable 0.165..0.435: nothing left to win) and
# only 0.107 m sideways. Sideways travel is r * sin(yaw angle), where r is the head's distance from the yaw
# axis, and r is NOT constant along the lift: straight up the BOTTOM..TOP line it is 0.067 m with the arm
# folded at the bottom, 0.114 m at mid-lift and 0.082 m with the elbow straight up. v3 spent its +-60-unit
# (+-45 deg) swings near the ends of that line, where a big angle moves the head hardly at all. Two changes,
# both measured against this calibration and this URDF:
#   * REACH extends the arm through the middle of the lift, taking r at lift 0.5/0.6 to 0.195/0.200 m.
#   * LIFTS and YAWS are now one table read in parallel, and every row is written so the beats carrying
#     |yaw| = 1 sit at a lift of 0.45..0.75 -- the band where r is within 3 % of its maximum -- while the
#     beats that visit lift 0 and 1 carry yaw near 0. Each phrase is therefore a circle in the frontal
#     plane (out to one side at mid height, over the top, out to the other side, down to the bottom)
#     instead of a vertical pump with a wiggle on it, and it spends the same speed budget as before.
BOTTOM = np.array([0.0, -65.0, -98.0, 0.0, -10.0])   # commanded: shoulder on its floor, elbow folded, head tucked
TOP = np.array([0.0, -4.0, 94.0, 0.0, 45.0])         # commanded: elbow straight up, shoulder up, head looking out
LIFT = TOP - BOTTOM
LIFT_JOINT = EL                                       # the joint with the longest way to go sizes the lift steps
START_LIFT = float((START[LIFT_JOINT] - BOTTOM[LIFT_JOINT]) / LIFT[LIFT_JOINT])   # 0.40: where START sits
# REACH: how far the arm reaches out through the middle of the lift, in base_pitch units, shaped by
# reach_layer(). Leaning out over the table is what it spends: measured over the whole library (every tier,
# variant and bucket tempo) the worst forward ZMP is +0.028 m with no reach, +0.042 m at REACH 13, +0.044 m
# at 15 and +0.048 m at 17, against ZMP_Y_MAX. 15 leaves about a tenth of that bound in hand for the live
# path, which generates clips at tempi this table never saw, and buys the head 0.260 m of sideways travel
# against v3's 0.107 m. Validator.validate enforces the bound rather than trusting this number.
REACH = 15.0
WIGGLE, SWAY = 0.55, 0.95            # of the beat's budget: the groove tier's swing and the hype/drop tier's.
                                     # At 80..180 bpm the budget is 152..68 units, so YAW_MAX/YAW_CALM is what
                                     # actually binds and the swing is the same width at every demo tempo; the
                                     # fractions bind again above ~200 bpm, where the tracker can still go.
ACCENT = 1.3                         # beat 1 of each bar swings 30 % wider, up to the ceiling
ACCENT_BEATS = (1, 5)
YAW_MAX = 68.0                       # the head never turns further than this from the crowd (commanded).
                                     # 68 units is 51 deg on this calibration, and sideways travel is
                                     # r * sin of it: 60 -> 68 is worth 0.023 m of head travel and costs
                                     # the turn nothing in tipping margin (a turned head leans less far
                                     # forward, not more). What it costs is room for `facing`: the +-98 box
                                     # leaves 30 units for the turn, and facing_limit measures the clips
                                     # validating to +-25 anyway, so the turn is not what binds here.
YAW_CALM = 45.0                      # ... and the groove tier, the calm one, stays inside this
ROLL = -0.4                          # wrist_roll counter-tilts the yaw so the head stays level-ish
NOD = 10.0                           # wrist_pitch nod on the crowd sweep
SHIVER = {"a": {YAW: 4.0}, "b": {WR: 6.0, YAW: 2.0}, "c": {YAW: 4.0, WP: 3.0}}
LOOKS_C = [0.8, 0.8, 0.0, -0.8, -0.8, 0.0, 0.8]              # groove c: right for two beats, centre, left ...
SWEEP_C = [-0.33, -0.83, -1.0, 0.0, 1.0, 0.83, 0.33]         # hype/drop c: across the crowd, left to right

# Lift and yaw on beats 1..7 (beats 0 and 8 are START, lift 0.40, yaw 0), read as one table: LIFTS is
# 0 = BOTTOM .. 1 = TOP and YAWS is -1..+1 of the tier's yaw ceiling. Rate-limited to the tempo by
# lift_profile() and yaw_layer(). Every |yaw| = 1 beat below sits at a lift of 0.45..0.75, which is where
# the head is furthest from the yaw axis and a swing is worth the most travel; every lift 0 or lift 1 beat
# carries little yaw, because there a swing is worth almost nothing.
LIFTS = {
    ("hype", "a"): [0.60, 0.85, 1.00, 0.80, 0.60, 0.30, 0.00],   # the wheel: out right at mid height, over
    ("hype", "b"): [0.00, 0.45, 0.90, 0.45, 0.00, 0.45, 0.90],   # bottom-up: rises out of the floor twice,
    ("hype", "c"): [0.00, 0.30, 0.60, 0.90, 0.60, 0.30, 0.00],   # the crowd sweep: an arch from the bottom
    ("groove", "a"): [0.50, 0.65, 0.60, 0.45, 0.50, 0.65, 0.60],   # a bob that stays in the wide band
    ("groove", "b"): [0.30, 0.55, 0.75, 0.55, 0.30, 0.55, 0.75],   # figure-eight: yaw over 4 beats, roll too
    ("groove", "c"): [0.55, 0.55, 0.85, 0.55, 0.55, 0.20, 0.55],   # nod-led: looks right, up, left, down
    ("drop", "a"): [1.00, 0.80, 0.60, 0.30, 0.00, 0.30, 0.60],   # the hit on top, then the wheel the other
    ("drop", "b"): [1.00, 0.45, 0.00, 0.45, 0.90, 0.45, 0.00],   # the hit, then two bottom-up diagonals
    ("drop", "c"): [1.00, 0.60, 0.60, 0.90, 0.60, 0.30, 0.00],   # the hit, the sweep across, the dive
    ("build", "a"): [0.3, 0.2, 0.12, 0.06, 0.0, 0.0, 0.25],  # the crouch: down to the floor, released on 7
    ("build", "b"): [0.3, 0.2, 0.12, 0.06, 0.0, 0.0, 0.25],
    ("build", "c"): [0.3, 0.2, 0.12, 0.06, 0.0, 0.0, 0.25],
}
YAWS = {
    ("hype", "a"): [1.00, 0.60, 0.00, -0.60, -1.00, -0.60, 0.00],   # ... the top, out left, down to the floor
    ("hype", "b"): [0.00, 1.00, 0.35, -0.35, 0.00, -1.00, -0.35],   # ... leaning right, then leaning left
    ("hype", "c"): SWEEP_C,                                          # ... left, over the apex, to the right
    ("groove", "a"): [0.70, 1.00, 0.70, 0.00, -0.70, -1.00, -0.70],
    ("groove", "b"): [0.00, 1.00, 0.00, -1.00, 0.00, 1.00, 0.00],
    ("groove", "c"): LOOKS_C,
    ("drop", "a"): [0.00, -0.60, -1.00, -0.60, 0.00, 0.60, 1.00],   # ... way round from hype a
    ("drop", "b"): [0.00, 1.00, 0.35, -0.35, 0.00, -1.00, -0.35],
    ("drop", "c"): SWEEP_C,
}
DEFAULT_BPM = 128.0                                          # keyframes() without a tempo: the demo tempo


def beat_budget(bpm: float, limit: float = SPEED_DESIGN) -> float:
    """The most a joint may move in ONE beat (commanded units) and stay under the speed budget."""
    return max_move(bpm, limit) * SPEED_MARGIN * 0.995


def lift_profile(tier: str, variant: str, bpm: float) -> np.ndarray:
    """Lift on beats 0..8 (0 = BOTTOM, 1 = TOP), beats 0 and 8 at START's lift, with no step larger than
    the tempo allows the lift's slowest joint. A profile that asks for more is rate-limited (each beat
    goes as far toward its target as one beat can), forward from START and backward from the return to
    START, so the phrase always reaches as far as the tempo allows and always gets home on beat 8.

    The elbow is still the slowest joint after REACH: per unit of lift it commands 192 * 1.2 = 230 units,
    against base_pitch's (61 + REACH * pi) * 1.15 = 160 at the steepest point of the hump."""
    if (tier, variant) not in LIFTS:
        raise ValueError(f"unknown tier/variant {tier!r} {variant!r}")
    d = beat_budget(bpm) / abs(LIFT[LIFT_JOINT])
    want = np.array([START_LIFT] + LIFTS[(tier, variant)] + [START_LIFT], dtype=float)
    L = want.copy()
    for _ in range(4):
        for k in range(1, BEATS):
            L[k] = float(np.clip(want[k], L[k - 1] - d, L[k - 1] + d))
        for k in range(BEATS - 1, 0, -1):
            L[k] = float(np.clip(L[k], L[k + 1] - d, L[k + 1] + d))
    return np.clip(L, 0.0, 1.0)


def reach_layer(L: np.ndarray, yaw: np.ndarray, reach: float = REACH) -> np.ndarray:
    """How far the arm reaches out on each beat, in base_pitch units, given the lift and the yaw.

    Two factors, and both of them are the same argument: reach only pays where a yaw swing pays.
    * sin(pi * L) -- zero at BOTTOM and at TOP, so the head's 0.165..0.435 m vertical envelope is exactly
      what it was and only the middle of the lift changes. It is also where reach is possible: folded at
      the bottom or straight up at the top the head is on the yaw axis whatever the shoulder does.
    * |yaw| / YAW_MAX -- full reach on a beat turned to the ceiling, none on a beat that passes through
      the centre, and part of it in between. Leaning out costs forward tipping margin, and a turned head
      spends less of it, because its distance from the axis lands sideways rather than in front; so the
      beats that would pay the most for the lean are exactly the ones that get none of it. The tier's own
      swing is what it is scaled by, not the phrase's peak: the groove tier turns to YAW_CALM rather than
      YAW_MAX, and a shallower turn really does carry more of the reach forward.
    """
    turned = np.minimum(np.abs(np.asarray(yaw, dtype=float)) / YAW_MAX, 1.0)
    return reach * np.sin(math.pi * np.asarray(L, dtype=float)) * turned


def lift_pose(L, reach=0.0) -> np.ndarray:
    """The arm pose(s) at lift L: the BOTTOM..TOP line, plus `reach` on base_pitch (reach_layer()).
    Reach only ever RAISES base_pitch, so it cannot push the shoulder toward its -65 floor or into the
    flip region; at lift 0.6 and full reach the head's distance from the yaw axis goes 0.114 -> 0.20 m."""
    L = np.asarray(L, dtype=float)
    K = BOTTOM[None, :] + L[..., None] * LIFT[None, :]
    K[..., BP] += np.asarray(reach, dtype=float)
    return K


def yaw_layer(tier: str, variant: str, bpm: float, L: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(yaw, wrist_roll, wrist_pitch nod) on beats 0..8 for the phrase, commanded units, each sized so no
    beat asks the yaw for more than the tempo's budget and the head never turns past the tier's ceiling.

    The shape is YAWS[(tier, variant)] on beats 1..7, written to peak where LIFTS puts the head furthest
    from the yaw axis. The bar accent multiplies beats 1 and 5 by ACCENT and is then clipped back to the
    row's own peak: a beat already at the ceiling cannot swing wider (that was true in v3 too, where
    YAW_MAX absorbed the accent at every tempo), and clipping rather than rescaling is what keeps a sweep
    symmetric -- an asymmetric sweep would cost the side that was not accented its share of the travel.
    """
    cap = beat_budget(bpm)
    k = np.arange(BEATS + 1)
    yaw = np.zeros(BEATS + 1)
    nod = np.zeros(BEATS + 1)
    if (tier, variant) == ("build", "b"):          # leans as it sinks: yaw and roll grow with the crouch
        sink = np.clip((START_LIFT - L) / START_LIFT, 0.0, 1.0)
        return -25.0 * sink, 20.0 * sink, nod
    if (tier, variant) == ("build", "c"):          # looks left and right on alternate beats while sinking
        sink = np.clip((START_LIFT - L) / START_LIFT, 0.0, 1.0)
        yaw = 12.0 * np.where(k % 2 == 1, 1.0, -1.0) * sink
        nod = NOD * 0.6 * np.where(k % 2 == 1, 1.0, -1.0) * (sink > 0)
    elif (tier, variant) in YAWS:
        shape = np.array(YAWS[(tier, variant)], dtype=float)
        peak = float(np.abs(shape).max())
        shape = np.clip(shape * np.where(np.isin(k[1:BEATS], ACCENT_BEATS), ACCENT, 1.0), -peak, peak)
        calm = tier == "groove"
        yaw[1:BEATS] = min(YAW_CALM if calm else YAW_MAX, (WIGGLE if calm else SWAY) * cap) * shape
        if (tier, variant) in (("hype", "c"), ("drop", "c")):
            nod[1:BEATS] = NOD * np.where(k[1:BEATS] % 2 == 1, 1.0, -1.0)
        if (tier, variant) == ("groove", "b"):     # the figure-eight's roll runs at the yaw's own rate
            roll = 16.0 * np.sin(2 * math.pi * k / 4)
            roll[0] = roll[BEATS] = 0.0
            return yaw, roll, nod
    yaw[0] = yaw[BEATS] = 0.0
    nod[0] = nod[BEATS] = 0.0
    step = np.abs(np.diff(yaw)).max()
    if step > cap:                                   # never ask the yaw for more than one beat can do
        yaw = yaw * (cap / step)
    return yaw, ROLL * yaw, nod


def commanded_keyframes(tier: str, variant: str, bpm: float) -> np.ndarray:
    """(BEATS+1, 5) COMMANDED poses landing on the beat instants; poses 0 and BEATS are START."""
    L = lift_profile(tier, variant, bpm)
    yaw, roll, nod = yaw_layer(tier, variant, bpm, L)
    # The build tier crouches and never swings the yaw, so reach would buy it no travel at all and would
    # turn its crouch (shoulder back and down) into a lean forward. It dances the bare BOTTOM..TOP line.
    K = lift_pose(L, 0.0 if tier == "build" else reach_layer(L, yaw))
    K[:, YAW] = yaw
    K[:, WR] = roll
    K[:, WP] = np.minimum(K[:, WP] + nod, WP_MAX)
    K[0] = START
    K[BEATS] = START
    return K


def keyframes(tier: str, variant: str = "a", bpm: float = DEFAULT_BPM) -> np.ndarray:
    """(BEATS+1, 5) RAW poses (what apply_gain turns into the commanded keyframes), pose k landing on beat
    instant k. Pose 0 and pose BEATS are START. The choreography is designed in commanded units; the gain
    table is divided out here so that the library's gain step reproduces the design exactly."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    K = commanded_keyframes(tier, variant, float(bpm))
    return START + (K - START) / GAIN


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
    U = trajectory(keyframes(tier, variant, bpm), bpm)
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


def facing_limit(U: np.ndarray, facing: float) -> float:
    """`facing` reduced to the largest turn in the same direction that keeps the ROTATED base_yaw column
    inside the +-JOINT_MAX box -- past that the envelope clamp would flatten the choreography against the
    box instead of turning it, which is the one thing a turn must not cost.

    `U` is the COMMANDED trajectory before clamp_envelope (the clamp is what this avoids). The clip's own
    yaw excursion is the room the turn has to fit in: with hi = max(base_yaw) and lo = min(base_yaw) the
    turn f must satisfy hi + f <= JOINT_MAX and lo + f >= -JOINT_MAX. Measured 2026-09-19 over every tier,
    variant and bucket tempo at bold 1.0, that leaves +-34 units at worst (hype b at 80 bpm, whose yaw
    rides the lift up to YAW_MAX) and +-54 at best (groove a, whose sway is +-40).

    The box is only the limit that can be checked here, without the robot model: turning the arm swings
    the head sideways over the table, so Validator.validate on the rotated clip stays the final word.
    Measured the same day against this calibration, every clip validates out to +-25 units; the first to
    fail is drop c at 80 bpm at +30, head_y 0.017 m against the 0.020 m floor.
    """
    f = float(facing)
    if f == 0.0 or f != f:              # no turn asked (NaN -> none, as bold_multiplier reads it): today's clip
        return 0.0
    yaw = np.asarray(U, dtype=float)[:, YAW]
    top, bottom = float(yaw.max()), float(yaw.min())
    lo, hi = -JOINT_MAX - bottom, JOINT_MAX - top
    if lo > hi:                         # the clip already fills the box on both sides: no room to turn at all
        return 0.0
    f = min(hi, max(lo, f))
    # On the edge the sum can land a ulp outside the box -- top + (JOINT_MAX - top) is not exactly
    # JOINT_MAX in binary floating point -- and clamp_envelope would then shave that one frame and change
    # its speed, which is the one thing a turn must not do. Step the turn toward 0 by single representable
    # values until the rotated column really fits; in practice that is a ulp or two, and never a design
    # margin invented here.
    while f != 0.0 and (top + f > JOINT_MAX or bottom + f < -JOINT_MAX):
        f = math.nextafter(f, 0.0)
    return f


def face_trajectory(U: np.ndarray, facing: float) -> np.ndarray:
    """The commanded trajectory turned to `facing`: the whole dance rotated about the base by a constant
    on base_yaw. At facing 0.0 the input is returned untouched -- not even a float round trip, the way
    bold_trajectory does at bold 1.0 -- so the home-facing library stays byte-identical: base_yaw carries
    exact -0.0 values, and -0.0 + 0.0 is +0.0, which csv_text writes as "0.0000" and not "-0.0000"."""
    if float(facing) == 0.0:
        return U
    V = np.array(U, dtype=float, copy=True)
    V[:, YAW] += float(facing)
    return V


def commanded(tier: str, bpm: float, variant: str = "a", gain: np.ndarray = GAIN,
              scales: np.ndarray | None = None, bold: float = 1.0, facing: float = 0.0) -> np.ndarray:
    """The trajectory as the runtime will be asked to play it: bold, per-joint scale, gain, the turn to
    `facing`, then the clamp. scales=None chooses them against the speed budget."""
    raw = bold_trajectory(tier, bpm, variant, bold)
    if scales is None:
        scales = joint_scales(raw, bpm, gain)
    U = apply_gain(scaled(raw, scales), gain)
    return clamp_envelope(face_trajectory(U, facing_limit(U, facing)))


def build_clip(tier: str, bpm: float, variant: str = "a", gain: np.ndarray = GAIN,
               bold: float = 1.0, facing: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """(scales, commanded trajectory). The batch library is bold 1.0 facing 0.0; the live path passes the
    slider and, when it has a direction to dance at, the bearing of the person holding the light.

    `facing` (base_yaw's own joint units) turns the whole dance about the base: it goes onto the commanded
    base_yaw column AFTER the gain -- the gain corrects the runtime's under-delivery of a MOVE, and a
    constant offset is not a move -- and BEFORE the envelope clamp, so the turn is clamped and validated
    like any other command. facing 0.0 returns today's clip byte for byte.

    The turned clip starts and ends at base_yaw = facing rather than 0. That is intended: the ends still
    match each other, so the runtime has nothing to blend when the next clip at the same facing starts.
    """
    raw = bold_trajectory(tier, bpm, variant, bold)
    scales = joint_scales(raw, bpm, gain)
    U = apply_gain(scaled(raw, scales), gain)
    # The turn is a constant added to one column, so every frame-to-frame difference is the same one it
    # was (to the last bit of the subtraction: (a+f)-(b+f) need not round to exactly a-b) and the turn
    # cannot raise any joint's peak speed. The speed budget above (joint_scales) needs no redoing, and
    # the budget's own 1 % margin (SPEED_MARGIN) is orders of magnitude more than that rounding.
    return scales, clamp_envelope(face_trajectory(U, facing_limit(U, facing)))


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
            mass, origin = it.find("mass"), it.find("origin")
            if mass is None or origin is None:
                raise ValueError(f"{self.robot / 'robot.urdf'}: inertial requires mass and origin elements")
            mass_value, xyz = mass.get("value"), origin.get("xyz")
            if mass_value is None or xyz is None:
                raise ValueError(f"{self.robot / 'robot.urdf'}: inertial requires mass value and origin xyz")
            self.inertials.append((float(mass_value), np.array([float(v) for v in xyz.split()])))

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
        # The START pose at both ends, base_yaw aside: a clip built with a `facing` starts and ends turned
        # by it (build_clip). What the runtime's blend-in skip actually needs is that the two ends MATCH,
        # so a re-trigger at the same facing has nothing to blend; a clip whose ends point different ways
        # is still a fault.
        rest = [j for j in range(len(JOINTS)) if j != YAW]
        if not (np.allclose(U[0][rest], START[rest], atol=1e-6) and np.allclose(U[-1][rest], START[rest], atol=1e-6)
                and abs(float(U[0][YAW]) - float(U[-1][YAW])) <= 1e-6):
            reasons.append("first/last frame is not the START pose (base_yaw aside, which carries the facing)")
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
        if zmp_y.max() > ZMP_Y_MAX:
            reasons.append(f"zmp_y max {zmp_y.max():+.3f} > {ZMP_Y_MAX:+.3f}")
        return {"ok": not reasons, "reasons": reasons, "head_y_min": float(head_y.min()),
                "zmp_y_min": float(zmp_y.min()), "zmp_y_max": float(zmp_y.max()),
                "peak_speed": speed, "problem_frames": len(bad)}


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
              "zmp_y_max": round(r["zmp_y_max"], 4),
              "peak_speed": {j: round(float(sp), 1) for j, sp in zip(JOINTS, r["peak_speed"])},
              "problem_frames": int(r["problem_frames"])}
    return bool(r["ok"]), report


def make_clip(tier: str, variant: str, bpm: float, bold: float, gains: dict | None = GAINS,
              model=None, facing: float = 0.0) -> tuple[list[tuple], dict]:
    """ONE clip at an exact bpm, boldness and facing, as (rows, meta): rows are 30 fps 5-tuples of
    commanded joint values (csv_text() turns them into the runtime's CSV), meta describes it. Same maths
    as the batch generator (build_clip), so bold 1.0 facing 0.0 at a bucket bpm reproduces the library
    file byte for byte. `facing` (base_yaw units) turns the whole dance to point that way -- at the person
    holding the light -- and meta["facing"] reports the turn actually applied, which facing_limit may have
    reduced to what the joint box allows. A turned clip starts and ends at base_yaw = facing, not 0.
    With `model` (a Validator or a LampModel) the clip is validated -- on the TURNED trajectory, since the
    turn carries the head sideways over the table -- and meta carries ok/reasons/report; without it
    meta["ok"] is None and the caller must validate before the file reaches the runtime."""
    bpm = float(bpm)
    if not (BPM_RANGE[0] <= bpm <= BPM_RANGE[1]):
        raise ValueError(f"bpm {bpm} outside {BPM_RANGE[0]:.0f}..{BPM_RANGE[1]:.0f}")
    gain = gain_vector(gains)
    scales, U = build_clip(tier, bpm, variant, gain, bold, facing)
    rows = [tuple(float(x) for x in u) for u in U]
    # The turn build_clip actually made, which facing_limit may have reduced: frame 0 is START, whose
    # base_yaw is 0, and facing_limit keeps the turned column inside the box, so the clamp leaves that
    # frame alone and its base_yaw IS the applied facing. No second pass over the trajectory for it.
    applied = float(U[0][YAW])
    meta = {"tier": tier, "variant": variant, "bpm": bpm, "bold": float(bold), "facing": applied,
            "multiplier": bold_multiplier(bold),
            "frames": int(len(U)), "seconds": round(len(U) / FPS, 4), "first_beat_s": HOLD_S + beat_period(bpm),
            # how wide the dance sways, measured from the facing it was danced at rather than from 0, so a
            # turned clip reports the size of its choreography and not the size of the turn
            "beats": BEATS, "amplitude": round(float(np.abs(U[:, YAW] - applied).max()), 2),
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


def one_name(tier: str, variant: str, bpm: float, bold: float, facing: float = 0.0) -> str:
    """File stem for a --one clip: live_<tier>_<variant>_<bpm>_<bold> with '.' -> 'p' (the runtime plays
    by stem, so the stem must not look like an extension). A nonzero facing is appended as _f<facing>, so
    rendering the same clip turned never overwrites the home-facing one sitting next to it."""
    stem = f"live_{tier}_{variant}_{bpm:g}_{bold:.2f}" + (f"_f{facing:g}" if facing else "")
    return stem.replace(".", "p")


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
    head = (f"{'clip':<20}{'frames':>7}{'sec':>6}{'A':>6}  {'scale y/bp/el/wr/wp':<26}{'head_y':>7}"
            + f"{'zmp-':>7}{'zmp+':>7}  " + "".join(f"{j[:5]:>6}" for j in JOINTS) + "  verdict")
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
                       + f"{'/'.join(f'{s:.2f}' for s in scales):<26}{v['head_y_min']:>+7.3f}"
                       + f"{v['zmp_y_min']:>+7.3f}{v['zmp_y_max']:>+7.3f}  "
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
                     "head_y_min": round(v["head_y_min"], 4), "zmp_y_min": round(v["zmp_y_min"], 4),
                     "zmp_y_max": round(v["zmp_y_max"], 4),
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
                "accent": ACCENT, "accent_beats": list(ACCENT_BEATS),
                "aliases": {e["name"]: e["alias_of"] for e in entries if e.get("alias_of")},
                "clips": sorted(entries, key=lambda e: (TIERS.index(e["tier"]), e["bpm"], e.get("variant") or ""))}
    path = out / "MANIFEST.json"
    write_atomic(path, json.dumps(manifest, indent=1) + "\n")
    return path


def one(spec: list[str], out: Path, robotdesc: Path, gains: dict | None = None, log=print,
        facing: float = 0.0) -> int:
    """--one TIER VARIANT BPM BOLD [--facing F]: make_clip + validate + write_clip_atomic, one table row,
    exit status. A turn the joint box does not allow is reduced and said so, not silently dropped."""
    tier, variant, bpm, bold = spec[0], spec[1], float(spec[2]), float(spec[3])
    if tier not in TIERS or variant not in VARIANTS:
        log(f"--one: tier must be one of {TIERS} and variant one of {VARIANTS}")
        return 2
    rows, meta = make_clip(tier, variant, bpm, bold, {**GAINS, **(gains or {})}, Validator(robotdesc), facing)
    if meta["facing"] != float(facing):
        log(f"--facing {facing:g} reduced to {meta['facing']:+.2f}: what this clip's own yaw leaves "
            f"inside the +-{JOINT_MAX:.0f} joint box")
    name = one_name(tier, variant, bpm, bold, meta["facing"])
    r = meta["report"]
    scales = "/".join(f"{meta['scale'][j]:.2f}" for j in JOINTS)
    speeds = "".join(f"{r['peak_speed'][j]:>6.0f}" for j in JOINTS)
    turn = f"  f{meta['facing']:+.1f}" if meta["facing"] else ""
    row = (f"{name:<28}{meta['frames']:>7}{meta['seconds']:>6.2f}{meta['amplitude']:>6.1f}  x{meta['multiplier']:.2f}{turn}  "
           f"{scales:<26}{r['head_y_min']:>+7.3f}{r['zmp_y_min']:>+7.3f}{r['zmp_y_max']:>+7.3f}  {speeds}")
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
    ap.add_argument("--facing", type=float, default=0.0,
                    help="with --one: dance the clip turned to this base_yaw (joint units, the units the CSV "
                         "carries), i.e. facing whoever the tracker has found, instead of the lamp's home "
                         "heading. Reduced to what the clip's own yaw leaves inside the +-94 box; the "
                         "validator is still the final word, because a big turn swings the head sideways "
                         "over the table (default 0: the home-facing dance)")
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
        return one(args.one, args.out, args.robotdesc, gains, facing=args.facing)
    if args.facing:
        print("--facing applies to --one only: the batch library is the home-facing one", file=sys.stderr)
        return 2
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
