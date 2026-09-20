#!/usr/bin/env python3
"""What the runtime does to a clip BEFORE the servos see it: an offline model of the animation route.

Why this exists. ground_truth_2026-09-20.md: `cha_look_right` commands base_yaw 0 -> +55 in 0.4 s and
holds it there for 1.37 s, and the arm delivered a peak of +19.9. `cha_turn` (0 -> -65 -> +65 -> 0 over
3.9 s) went left into the stop and never went right. `beat_hype_124_c` (5.07 s) delivered its +66
swing almost in full. Single tracking moves given >= 2 s land 90-99 %. So the loss is not the servo:
it is what the runtime does to a clip on the way to the servo, and that pipeline had not been
modelled. The ground-truth note guessed at fusion.py's own defaults (blend 500 ms, window 300 ms,
90 units/s, 300 units/s^2); config/default.yaml overrides every one of them, and the live values are
what make a 2 s clip disappear into its own blend-in.

What runs, read in place from the vendor runtime (runtime_ref; nothing is copied here, every step
below is re-derived from the named function and its equations), for a POST that ends in
MotionController.play_animation (control/runtime.py):

  1. CsvMotionPlanTransformer.transform (motion/transformers/csv_transformer.py): one waypoint per
     row, timestamp_ms = round((t_s - t_s[0]) * 1000, 3).
  2. process_recorded_motion_plan (motion/trajectory_processing.py), because motion.playback.
     smooth_enabled is true: normalize_timestamps -> remove_position_outliers (20 units per frame)
     -> resample_motion_plan (a 30 fps grid of int(duration_s * fps) + 1 samples -- which DROPS one
     sample from any clip whose duration is not a whole number of frames, e.g. 59 frames -> 58)
     -> smooth_motion_plan (5-frame moving average, first and last 3 frames untouched)
     -> limit_motion_velocity(300 units/s, by stretching timestamps).
  3. fuse_into_motion_plan (motion/fusion.py) with config/default.yaml's motion.transition block:
     blend_duration_ms 1100, lookahead_ms 1500, blend_window_ms 1500, threshold_deg 2.0,
     velocity_threshold 5.0, max_velocity_dps 55, max_acceleration_dps2 45; plus the safety filter's
     position_limits (+-100) and execution_velocity_limit (300 units/s, because
     robotdesc safety.yaml has enforce_velocity: true). This is where a short clip loses itself:
     the blend START is the clip frame nearest lookahead_ms (1.5 s) AFTER the first moving frame,
     everything before it is dropped, and the window from there to blend_window_ms later (or the
     clip's end) is squeezed into an 1100 ms crossfade against a quintic that starts at the current
     pose. cha_look_right is 1.93 s long (59 rows): its blend starts at 1.53 s, on the return home, so the
     whole +55 look is dropped and what streams is an 1100 ms bump that peaks in the low twenties.
  4. MotionController._submit_direct: with enforce_velocity the plan gets a LIVE BASELINE (the
     measured pose as a new frame at t = 0, every other frame pushed back one frame, 33.3 ms) and
     MotionSafetyFilter.validate (control/safety_filter.py) clamps to +-100 and REFUSES the whole
     plan if any frame is more than 0.001 outside the envelope or any frame pair exceeds 300 units/s.
  5. MotionExecutor.execute (control/executor.py) streams the frames at their timestamps
     (asyncio.sleep until each one, at least 8 ms apart); no resampling, no filter, no further
     limit. The servo's own dynamics (P 24, torque 700: 100-150 ms of lag and the beat-rate
     attenuation measured with beattrace) come AFTER this point and are servo_model.py's job, and
     the 250-450 ms POST-to-first-write latency is a parameter of simulate.py, which chains the two:
     start_latency_ms here only offsets the returned timestamps onto the POST clock.

Units. Plans are in the calibration's -100..100 units (robotdesc actuation.yaml coordinate_space:
normalized_m100_100; 0.74 deg per unit on base_yaw from the calibration's 1693-tick span). The YAML
keys say "dps" and fusion.py says "deg", but NOTHING converts: motion/transition_settings.py
fusion_kwargs reads `max_velocity_dps` and `max_acceleration_dps2` with _float_value and hands them
to fuse_into_motion_plan as `max_velocity` / `max_acceleration` as they are, and fusion.py clamps
unit velocities against them (make_transition_quintic, _clamp_dict). So 55 means 55 units/s and 45
means 45 units/s^2, threshold_deg 2.0 is 2.0 units, and the safety filter's 300 "deg/s" is 300
units/s (the same number safe_motion_info reports as max_velocity_units_s). An earlier revision of
this module kept a per-joint "deg" reading because transition_settings.py had not been pulled; it
has now, and that reading is gone. Resolved 2026-09-20 03:10 (ground_truth_2026-09-20.md, RESOLVED).

What the two knobs do, and do not do: max_velocity_dps clamps only the boundary velocities the
quintic must MATCH at its two ends (the arm's and the clip's at win_end); it never limits the speed
of the path in between (cha_turn's fade slides 60 units at up to 154 units/s). The whole loss on a
short clip is find_blend_window: the blend starts lookahead_ms after the first moving frame and
everything before it is dropped, unless the skip rule streams the clip raw.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

JOINTS = ("base_yaw", "base_pitch", "elbow_pitch", "wrist_roll", "wrist_pitch")
NJ = len(JOINTS)
TICKS_PER_TURN = 4096.0          # Feetech STS3215 encoder (spatial.LampModel, calibrate_servos.py)
UNIT_SPAN = 200.0                # range_min..range_max ticks map onto -100..100 units (yaw_limits.units)


class PlanRefused(ValueError):
    """The runtime would raise and stream nothing: MotionSafetyFilter.validate with enforce_velocity."""


@dataclass(frozen=True)
class RuntimeConfig:
    """Every knob the animation route applies, with the LIVE value and where it is read from.

    The transition numbers are config/default.yaml `motion.transition` (fusion.py's own keyword
    defaults -- 500 / 300 / 300 / 90 / 300 -- are what a caller with no config would get and are NOT
    what runs; FUSION_PY_DEFAULTS below keeps them for the comparison the ground-truth note made).
    The processing numbers are the literals inside process_recorded_motion_plan; the safety numbers
    are robotdesc safety.yaml (limits +-100, enforce_velocity true) and safety_filter.py's
    DEFAULT_MAX_VELOCITY 300, the same 300 follow.py measured the tracking route refusing above."""

    fps: int = 30                             # motion.fps; also MotionExecutor.fps and the resample grid
    smooth_enabled: bool = True               # motion.playback.smooth_enabled -> process_recorded_motion_plan
    transition_enabled: bool = True           # motion.playback.transition_enabled and transition.mode fuse
    blend_duration_ms: float = 1100.0         # motion.transition.blend_duration_ms: the quintic AND the crossfade
    lookahead_ms: float = 1500.0              # motion.transition.lookahead_ms: how far past the first moving frame the blend starts
    blend_window_ms: float = 1500.0           # motion.transition.blend_window_ms: how much clip the crossfade swallows
    threshold_deg: float = 2.0                # motion.transition.threshold_deg: the pos_close half of the skip rule
    velocity_threshold: float = 5.0           # motion.transition.velocity_threshold: the velocity half, and find_blend_target's min_velocity
    max_velocity_dps: float = 55.0            # motion.transition.max_velocity_dps -> make_transition_quintic max_velocity, UNITS/s
                                              # (transition_settings.fusion_kwargs passes the number through unconverted)
    max_acceleration_dps2: float = 45.0       # motion.transition.max_acceleration_dps2 -> max_acceleration, units/s^2, likewise
    position_limits: tuple[tuple[float, float], ...] = ((-100.0, 100.0),) * NJ   # safety.yaml limits, via safety.position_limits
    execution_velocity_limit: float | None = 300.0   # safety.execution_velocity_limit: fusion's final limit_motion_velocity
    live_baseline: bool = True                # safety.requires_live_baseline (== enforce_velocity): _submit_direct prepends the pose
    safety_max_velocity: float = 300.0        # MotionSafetyFilter._max_velocity: validate() refuses above it
    outlier_max_step: float = 20.0            # remove_position_outliers default_max_step, units per frame
    smooth_window: int = 5                    # smooth_motion_plan window
    smooth_passes: int = 1                    # smooth_motion_plan passes
    processing_velocity_limit: float = 300.0  # process_recorded_motion_plan's limit_motion_velocity(default_max_velocity=300)
    start_latency_ms: float = 0.0             # measured 250-450 ms POST -> first servo write; not part of the plan, added on request

    def limits_units(self) -> tuple[np.ndarray, np.ndarray]:
        """(max_velocity, max_acceleration) per joint, units/s and units/s^2: the YAML numbers on every
        joint, because transition_settings.fusion_kwargs converts nothing (module docstring, Units)."""
        return np.full(NJ, float(self.max_velocity_dps)), np.full(NJ, float(self.max_acceleration_dps2))


LIVE_CONFIG = RuntimeConfig()
# What fusion.py's blend_into_motion_plan applies when nobody passes kwargs: the numbers the
# ground-truth note reasoned with. Kept only so the Validate phase can show they do not fit.
FUSION_PY_DEFAULTS = RuntimeConfig(blend_duration_ms=500.0, lookahead_ms=300.0, blend_window_ms=300.0,
                                   max_velocity_dps=90.0, max_acceleration_dps2=300.0)


def deg_per_unit_from_calibration(path: str | os.PathLike | None = None) -> tuple[float, ...]:
    """Degrees per unit per joint from the servo calibration: (range_max - range_min) ticks over
    4096 per turn, spread over 200 units. base_yaw: 1693 / 4096 * 360 / 200 = 0.744 (the "0.74" of
    the ground-truth note). Same formula as spatial.LampModel, in degrees instead of radians. For
    reporting a unit figure in degrees only: the runtime never applies it (module docstring, Units)."""
    path = Path(path or os.environ.get("LELAMP_CALIBRATION_PATH", "/var/lib/lelamp/user-data/v1/calibration/lelamp.json"))
    cal = json.loads(path.read_text())
    return tuple((float(cal[j]["range_max"]) - float(cal[j]["range_min"])) / TICKS_PER_TURN * 360.0 / UNIT_SPAN for j in JOINTS)


class Streamed(NamedTuple):
    """The frames the executor writes, in order: t_ms[i] is when positions[i] (units, JOINTS order)
    goes to the bus, relative to the first write (plus config.start_latency_ms)."""
    t_ms: np.ndarray
    positions: np.ndarray
    decisions: dict


# ----------------------------------------------------------------------------- small pieces
def minimum_jerk(t: float) -> float:
    """The crossfade weight w(t) = 10 t^3 - 15 t^4 + 6 t^5, clamped to 0..1 (fusion.minimum_jerk_scalar):
    zero slope at both ends, so the fade neither kinks in nor kinks out."""
    t = max(0.0, min(1.0, float(t)))
    return 10.0 * t ** 3 - 15.0 * t ** 4 + 6.0 * t ** 5


def clip_timestamps_ms(n: int, fps: int = 30) -> np.ndarray:
    """What the CSV route yields for an n-frame clip written at fps (beat_clips.csv_text writes
    1000 + i/fps to 6 decimals; csv_transformer subtracts the first and rounds to 3 decimals of a
    millisecond): round(i * 1000 / fps, 3). The rounding matters downstream -- resample_motion_plan
    takes int(duration_ms / 1000 * fps) of it, and 1933.333 ms is 57.99999 frames, not 58."""
    return np.array([round(i * 1000.0 / fps, 3) for i in range(n)], dtype=float)


def _norm(v: np.ndarray) -> float:
    """fusion._velocity_magnitude: the Euclidean norm over joints."""
    return float(math.sqrt(float(np.sum(np.asarray(v, dtype=float) ** 2))))


def _estimate_velocity(ts: np.ndarray, pos: np.ndarray, index: int, window: int = 2) -> np.ndarray:
    """fusion.estimate_velocity: central difference over +-window frames, clamped to the plan's ends
    (so at frame 0 it is (p[2] - p[0]) / (t[2] - t[0])); zero when the span is under 1 ms."""
    n = len(ts)
    if n == 0 or index < 0 or index >= n:
        return np.zeros(NJ)
    lo, hi = max(0, index - window), min(n - 1, index + window)
    dt = (ts[hi] - ts[lo]) / 1000.0
    if dt > 0.001:
        return (pos[hi] - pos[lo]) / dt
    return np.zeros(NJ)


def _estimate_acceleration(ts: np.ndarray, pos: np.ndarray, index: int, window: int = 2) -> np.ndarray:
    """fusion.estimate_acceleration: the velocity (window 1) at the two ends of the +-window span,
    differenced over that span."""
    n = len(ts)
    if n == 0 or index < 0 or index >= n:
        return np.zeros(NJ)
    lo, hi = max(0, index - window), min(n - 1, index + window)
    if lo >= hi:
        return np.zeros(NJ)
    v_lo = _estimate_velocity(ts, pos, lo, window=max(1, window // 2))
    v_hi = _estimate_velocity(ts, pos, hi, window=max(1, window // 2))
    dt = (ts[hi] - ts[lo]) / 1000.0
    if dt > 0.001:
        return (v_hi - v_lo) / dt
    return np.zeros(NJ)


def _first_moving(pos: np.ndarray, threshold: float = 0.05) -> int:
    """fusion.find_first_moving_waypoint: the first frame where any joint differs from frame 0 by more
    than 0.05 units (dead frames are what the blend window skips); 0 when nothing ever moves."""
    for i in range(len(pos)):
        if np.any(np.abs(pos[i] - pos[0]) > threshold):
            return i
    return 0


def _blend_target(ts: np.ndarray, pos: np.ndarray, *, lookahead_ms: float, min_index: int,
                  min_velocity: float, max_scan_ms: float = 1000.0) -> int:
    """fusion.find_blend_target: the frame nearest lookahead_ms after the first moving frame; if the
    clip is not moving there (|v| < min_velocity), the first moving frame within the next second."""
    n = len(ts)
    start = max(0, min(min_index, n - 1))
    target_ts = ts[start] + lookahead_ms
    best, best_dist = start, float("inf")
    for i in range(start, n):
        dist = abs(ts[i] - target_ts)
        if dist < best_dist:
            best_dist, best = dist, i
    if _norm(_estimate_velocity(ts, pos, best)) < min_velocity:
        limit_ts = ts[best] + max_scan_ms
        for i in range(best + 1, n):
            if ts[i] > limit_ts:
                break
            if _norm(_estimate_velocity(ts, pos, i)) >= min_velocity:
                best = i
                break
    return min(best, n - 1)


def _blend_window(ts: np.ndarray, pos: np.ndarray, *, lookahead_ms: float, blend_window_ms: float,
                  min_velocity: float) -> tuple[int, int]:
    """fusion.find_blend_window: (start, end) frame indices. end is the first frame at or past
    start + blend_window_ms, or the last frame of the clip when the clip is shorter than that --
    which is the case for every clip under lookahead + window = 3 s."""
    n = len(ts)
    if n < 2:
        return 0, 0
    first_move = _first_moving(pos, 0.05)
    start = _blend_target(ts, pos, lookahead_ms=lookahead_ms, min_index=first_move, min_velocity=min_velocity)
    start = max(first_move, start)
    target_end_ts = ts[start] + blend_window_ms
    end = start + 1
    for i in range(start + 1, n):
        if ts[i] >= target_end_ts:
            end = i
            break
    else:
        end = n - 1
    return start, end


def _sample_at(ts: np.ndarray, pos: np.ndarray, t_ms: float) -> np.ndarray:
    """fusion.sample_plan_at_time: linear interpolation, clamped to the ends."""
    t = max(0.0, float(t_ms))
    if t <= ts[0]:
        return pos[0].copy()
    if t >= ts[-1]:
        return pos[-1].copy()
    for i in range(len(ts) - 1):
        t0, t1 = ts[i], ts[i + 1]
        if t0 <= t <= t1:
            if t1 - t0 < 0.001:
                return pos[i].copy()
            w = (t - t0) / (t1 - t0)
            return pos[i] + (pos[i + 1] - pos[i]) * w
    return pos[-1].copy()


def quintic_transition(p0, p1, v0, v1, a0, a1, *, duration_ms: float, max_velocity, max_acceleration,
                       fps: int = 30) -> tuple[np.ndarray, np.ndarray, dict]:
    """fusion.make_transition_quintic: per joint, the degree-5 polynomial through p(0)=p0, p'(0)=v0,
    p''(0)=a0, p(T)=p1, p'(T)=v1, p''(T)=a1, sampled at max(2, int(T * fps)) points, positions
    rounded to 4 decimals and timestamps to 3. The boundary velocities and accelerations are CLAMPED
    to +-max_velocity / +-max_acceleration first (that is the whole effect of max_velocity_dps: it
    never limits the path's speed, only what the blend is asked to match at its ends). The vendor
    solves the 3x3 system for c3..c5 by Cramer's rule; the closed form below is the same solution.
    Returns (ts, pos, clamps) where clamps says which boundary values the limits bit."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    vmax = np.broadcast_to(np.asarray(max_velocity, float), (NJ,))
    amax = np.broadcast_to(np.asarray(max_acceleration, float), (NJ,))
    raw = {"v0": np.asarray(v0, float), "v1": np.asarray(v1, float), "a0": np.asarray(a0, float), "a1": np.asarray(a1, float)}
    lim = {"v0": vmax, "v1": vmax, "a0": amax, "a1": amax}
    cl = {k: np.clip(raw[k], -lim[k], lim[k]) for k in raw}
    clamps = {k: {JOINTS[j]: {"asked": float(raw[k][j]), "clamped_to": float(cl[k][j])}
                  for j in range(NJ) if cl[k][j] != raw[k][j]} for k in raw}

    T = duration_ms / 1000.0
    steps = max(2, int(T * fps))
    dt = duration_ms / (steps - 1)
    if T < 0.001:
        c = np.zeros((6, NJ))
        c[0] = p1
    else:
        dp = p1 - p0 - cl["v0"] * T - cl["a0"] * T * T / 2.0
        dv = cl["v1"] - cl["v0"] - cl["a0"] * T
        da = cl["a1"] - cl["a0"]
        T2, T3, T4, T5 = T * T, T ** 3, T ** 4, T ** 5
        c = np.zeros((6, NJ))
        c[0], c[1], c[2] = p0, cl["v0"], cl["a0"] / 2.0
        # The unique solution of [T^3 T^4 T^5; 3T^2 4T^3 5T^4; 6T 12T^2 20T^3] (c3 c4 c5)^T = (dp dv da)^T
        c[3] = (20.0 * dp - 8.0 * dv * T + da * T2) / (2.0 * T3)
        c[4] = (-30.0 * dp + 14.0 * dv * T - 2.0 * da * T2) / (2.0 * T4)
        c[5] = (12.0 * dp - 6.0 * dv * T + da * T2) / (2.0 * T5)
    ts = np.array([round(i * dt, 3) for i in range(steps)], dtype=float)
    pos = np.zeros((steps, NJ))
    for i in range(steps):
        t = i / max(steps - 1, 1) * T
        powers = np.array([1.0, t, t ** 2, t ** 3, t ** 4, t ** 5])
        pos[i] = np.round(powers @ c, 4)
    return ts, pos, clamps


# ----------------------------------------------------------------------------- trajectory_processing.py
def _normalize_timestamps(ts: np.ndarray, fps: int) -> np.ndarray:
    """trajectory_processing.normalize_timestamps: start at 0, never negative, strictly monotonic
    (a repeat is pushed one frame on), round 3; values over 100 are taken as seconds. All-zero
    timestamps become a fixed-fps grid."""
    first = float(ts[0])
    scale = 1000.0 if first > 100.0 else 1.0
    out = []
    for t in ts:
        v = max(0.0, (float(t) - first) * scale)
        if out and v <= out[-1]:
            v = out[-1] + 1000.0 / fps
        out.append(round(v, 3))
    if all(v == 0.0 for v in out):
        out = [round(i * 1000.0 / fps, 3) for i in range(len(out))]
    return np.array(out, dtype=float)


def _remove_outliers(pos: np.ndarray, max_step: float) -> tuple[np.ndarray, int]:
    """trajectory_processing.remove_position_outliers: a frame that jumps more than max_step from the
    (already-repaired) previous frame, while the frame after it is within 2 * max_step of that
    previous frame, is replaced by the midpoint of its neighbours. Ends are never touched. At 30 fps
    the 20-unit step is 600 units/s, above anything beat_clips validates (380), so this is a no-op
    on our clips; it is here because it runs, and a hand-edited clip could trip it."""
    pos = pos.copy()
    count = 0
    if len(pos) < 3:
        return pos, 0
    for i in range(1, len(pos) - 1):
        modified = False
        for j in range(NJ):
            prev_v, next_v = pos[i - 1, j], pos[i + 1, j]
            if abs(pos[i, j] - prev_v) > max_step and abs(next_v - prev_v) <= max_step * 2:
                pos[i, j] = (prev_v + next_v) / 2.0
                modified = True
        if modified:
            count += 1
    return pos, count


def _resample(ts: np.ndarray, pos: np.ndarray, fps: int) -> tuple[np.ndarray, np.ndarray]:
    """trajectory_processing.resample_motion_plan: max(2, int(duration_ms / 1000 * fps) + 1) samples
    on linspace(0, ts[-1]), each joint np.interp'd. The int() is a floor of a float product, so a
    58-frame duration of 1933.333 ms (57.99999 frames) gets 58 samples spaced 33.9 ms, not 59."""
    if len(ts) < 2:
        return ts, pos
    duration = float(ts[-1])
    count = max(2, int(duration / 1000.0 * fps) + 1)
    new_ts = np.linspace(0.0, duration, count)
    new_pos = np.column_stack([np.interp(new_ts, ts, pos[:, j]) for j in range(NJ)])
    return np.array([round(float(t), 3) for t in new_ts]), new_pos


def _smooth(pos: np.ndarray, window: int, passes: int) -> np.ndarray:
    """trajectory_processing.smooth_motion_plan: edge-padded moving average of `window` (made odd),
    with the first and last half+1 samples copied back unsmoothed; skipped for clips shorter than
    the window. A 5-frame box at 30 fps is a 167 ms average: it rounds the corners of a snap."""
    if len(pos) < window or window < 3:
        return pos
    if window % 2 == 0:
        window += 1
    half = window // 2
    kernel = np.ones(window) / window
    out = pos.copy()
    for _ in range(passes):
        for j in range(NJ):
            arr = out[:, j].copy()
            conv = np.convolve(np.pad(arr, (half, half), mode="edge"), kernel, mode="valid")
            conv[:half + 1] = arr[:half + 1]
            conv[-half - 1:] = arr[-half - 1:]
            out[:, j] = conv
    return out


def _limit_velocity(ts: np.ndarray, pos: np.ndarray, max_velocity: float) -> tuple[np.ndarray, np.ndarray, int]:
    """trajectory_processing.limit_motion_velocity: where a frame pair exceeds 99 % of max_velocity on
    any joint, the pair's dt is stretched by the worst ratio; positions are untouched. Faithful to a
    quirk of the vendor loop: dt is the ORIGINAL timestamp of this frame minus the already-STRETCHED
    timestamp of the previous one, so after one stretch the following pairs look faster than they are
    (and once the stretched clock overtakes the original, dt <= 0 falls back to one frame). This
    cascades a single over-fast pair into several stretched ones."""
    if len(ts) < 2:
        return ts, pos, 0
    effective = max_velocity * 0.99
    new_ts = [float(ts[0])]
    count = 0
    for i in range(1, len(ts)):
        dt = float(ts[i]) - new_ts[-1]
        if dt <= 0:
            dt = 1000.0 / 30
        vel = np.abs(pos[i] - pos[i - 1]) / (dt / 1000.0)
        max_ratio = 1.0
        for v in vel:
            if v > effective and v / effective > max_ratio:
                max_ratio = v / effective
        if max_ratio > 1.0:
            count += 1
        new_ts.append(round(new_ts[-1] + dt * max_ratio, 3))
    return np.array(new_ts), pos.copy(), count


def process_recorded(ts: np.ndarray, pos: np.ndarray, config: RuntimeConfig = LIVE_CONFIG) -> tuple[np.ndarray, np.ndarray, dict]:
    """trajectory_processing.process_recorded_motion_plan, in its order: normalize, outliers,
    resample, smooth, limit at processing_velocity_limit (300)."""
    ts = _normalize_timestamps(ts, config.fps)
    pos, outliers = _remove_outliers(pos, config.outlier_max_step)
    ts, pos = _resample(ts, pos, config.fps)
    resampled = len(ts)
    pos = _smooth(pos, config.smooth_window, config.smooth_passes)
    ts, pos, stretched = _limit_velocity(ts, pos, config.processing_velocity_limit)
    return ts, pos, {"outliers_removed": outliers, "resampled_frames": resampled,
                     "velocity_stretch_count": stretched, "duration_ms": float(ts[-1])}


def processed_peak_speed(clip, timestamps_ms=None, config: RuntimeConfig = LIVE_CONFIG) -> np.ndarray:
    """Per-joint peak frame-to-frame speed (units/s) of the plan AS limit_motion_velocity SEES IT: after
    normalize_timestamps, remove_position_outliers, the 30 fps resample and the 5-frame smooth, and
    before any stretch. This is the number a clip has to keep under 0.99 * processing_velocity_limit
    (297 units/s) to keep its timing: the limiter stretches the first pair over it and the stretch
    cascades (see _limit_velocity), so a clip written to beat_clips' 380 can leave the runtime longer
    than its beat count. The smoothing takes a few percent off an eased peak and the resample's
    slightly longer frame (33.9 ms for a 59-row clip) a little more, so this is lower than
    beat_clips.peak_speed on the same rows -- and it is the one that matters."""
    pos = np.asarray(clip, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != NJ or len(pos) < 2:
        raise ValueError(f"clip must be (N >= 2, {NJ}), got {pos.shape}")
    ts = clip_timestamps_ms(len(pos), config.fps) if timestamps_ms is None else np.asarray(timestamps_ms, dtype=float)
    ts = _normalize_timestamps(ts, config.fps)
    pos, _ = _remove_outliers(pos, config.outlier_max_step)
    ts, pos = _resample(ts, pos, config.fps)
    pos = _smooth(pos, config.smooth_window, config.smooth_passes)
    dt = np.diff(ts)[:, None] / 1000.0
    dt[dt <= 0] = 1.0 / config.fps
    return (np.abs(np.diff(pos, axis=0)) / dt).max(axis=0)


# ----------------------------------------------------------------------------- fusion.py
def fuse(ts: np.ndarray, pos: np.ndarray, current: np.ndarray, current_velocity: np.ndarray,
         current_acceleration: np.ndarray, config: RuntimeConfig = LIVE_CONFIG) -> tuple[np.ndarray, np.ndarray, dict]:
    """fusion.blend_into_motion_plan (what fuse_into_motion_plan delegates to), then the
    execution_velocity_limit pass at its end. Returns the fused plan and what it decided."""
    vmax, amax = config.limits_units()
    dec: dict = {
        "effective_max_velocity_units": [float(v) for v in vmax],
        "effective_max_acceleration_units": [float(a) for a in amax],
    }
    if len(ts) == 0:
        dec.update(skipped_blend=True, skip_rule={"reason": "empty plan"})
        return ts, pos, dec

    # The skip rule (blend_into_motion_plan): position within threshold_deg on EVERY joint AND both
    # the arm's velocity and the clip's entry velocity (central difference at frame 0, so it sees
    # frames 0..2) under velocity_threshold. A clip that opens with a hold at the current pose
    # passes; a clip that moves off its first frame within two frames does not, however close.
    cv_mag = _norm(current_velocity)
    first_vel = _estimate_velocity(ts, pos, 0)
    max_diff = float(np.max(np.abs(current - pos[0])))
    pos_close = max_diff <= config.threshold_deg
    dec["skip_rule"] = {"pos_close": bool(pos_close), "max_pos_diff": max_diff,
                        "current_velocity_magnitude": cv_mag, "clip_entry_velocity_magnitude": _norm(first_vel),
                        "threshold_deg": config.threshold_deg, "velocity_threshold": config.velocity_threshold}
    if pos_close and cv_mag < config.velocity_threshold and _norm(first_vel) < config.velocity_threshold:
        dec["skipped_blend"] = True
        dec["blend"] = None
        return ts, pos, dec
    dec["skipped_blend"] = False

    win_start, win_end = _blend_window(ts, pos, lookahead_ms=config.lookahead_ms,
                                       blend_window_ms=config.blend_window_ms, min_velocity=config.velocity_threshold)
    # compute_boundary_state at the window END: that is what the quintic aims at.
    end_pos = pos[win_end].copy()
    end_vel = _estimate_velocity(ts, pos, win_end)
    end_acc = _estimate_acceleration(ts, pos, win_end)

    tr_ts, tr_pos, clamps = quintic_transition(current, end_pos, current_velocity, end_vel, current_acceleration, end_acc,
                                               duration_ms=config.blend_duration_ms, max_velocity=vmax,
                                               max_acceleration=amax, fps=config.fps)

    # The crossfade: fade_steps frames over blend_duration_ms; frame i takes the transition at its own
    # time and the clip at the SAME fraction of the blend window -- so a window longer than the fade
    # (1500 ms into 1100 ms) plays that stretch of the clip 1.36x fast, and a window cut short by the
    # clip's end plays it slow. Then w(t) mixes them and the position limits clamp.
    fade_steps = max(2, int(config.blend_duration_ms / 1000.0 * config.fps))
    fade_dt = config.blend_duration_ms / (fade_steps - 1)
    win_start_ts, win_end_ts = float(ts[win_start]), float(ts[win_end])
    lo = np.array([l for l, _ in config.position_limits], dtype=float)
    hi = np.array([h for _, h in config.position_limits], dtype=float)
    fade_ts, fade_pos = [], []
    constrained_frames, constrained_joints = 0, set()
    for i in range(fade_steps):
        t_ms = round(i * fade_dt, 3)
        t_norm = i / max(fade_steps - 1, 1)
        tp = _sample_at(tr_ts, tr_pos, t_ms)
        ap = _sample_at(ts, pos, win_start_ts + t_norm * (win_end_ts - win_start_ts))
        w = minimum_jerk(t_norm)
        value = np.round((1.0 - w) * tp + w * ap, 4)
        constrained = np.clip(value, lo, hi)
        bit = constrained != value
        if bit.any():
            constrained_frames += 1
            constrained_joints.update(JOINTS[j] for j in np.flatnonzero(bit))
        fade_ts.append(t_ms)
        fade_pos.append(constrained)

    # After the fade, the rest of the clip past the window end, re-timed to follow the fade's last
    # frame; the first remaining frame is dropped if the fade already landed on it (within 0.01).
    remaining_pos = pos[win_end + 1:]
    remaining_ts = ts[win_end + 1:]
    offset = fade_ts[-1]
    skip_first = len(remaining_pos) > 0 and bool(np.all(np.abs(fade_pos[-1] - remaining_pos[0]) < 0.01))
    start_idx = 1 if skip_first else 0
    all_ts = list(fade_ts) + [round(offset + (float(t) - win_end_ts), 3) for t in remaining_ts[start_idx:]]
    all_pos = list(fade_pos) + [p.copy() for p in remaining_pos[start_idx:]]

    # _rebase_waypoints: to 0, strictly monotonic.
    base = all_ts[0]
    out_ts: list[float] = []
    for t in all_ts:
        v = round(t - base, 3)
        if out_ts and v <= out_ts[-1]:
            v = out_ts[-1] + 1000.0 / config.fps
        out_ts.append(v)
    fused_ts, fused_pos = np.array(out_ts), np.array(all_pos)

    dec["blend"] = {
        "start_index": int(win_start), "end_index": int(win_end),
        "start_ms": win_start_ts, "end_ms": win_end_ts,
        "dropped_frames": int(win_start),                 # fusion's metadata "dropped_waypoints"
        "frames_in_window": int(win_end - win_start + 1),
        "transition_duration_ms": float(config.blend_duration_ms),
        "fade_frames": int(fade_steps),
        "target_positions": [float(v) for v in end_pos],
        "target_velocities": [float(v) for v in end_vel],
        "target_accelerations": [float(v) for v in end_acc],
        "velocity_clamp": {k: clamps[k] for k in ("v0", "v1")},
        "acceleration_clamp": {k: clamps[k] for k in ("a0", "a1")},
        "position_clamp": {"waypoints": constrained_frames, "joints": sorted(constrained_joints)},
        "skip_first_remaining": skip_first,
        "remaining_frames": int(len(remaining_pos) - start_idx),
    }
    if config.execution_velocity_limit is not None and config.execution_velocity_limit > 0:
        before = float(fused_ts[-1])
        fused_ts, fused_pos, stretched = _limit_velocity(fused_ts, fused_pos, config.execution_velocity_limit)
        dec["blend"]["execution_velocity_stretch"] = {"limit_units_s": float(config.execution_velocity_limit),
                                                      "count": stretched, "duration_ms_before": before,
                                                      "duration_ms_after": float(fused_ts[-1])}
    return fused_ts, fused_pos, dec


# ----------------------------------------------------------------------------- runtime.py / safety_filter.py / executor.py
def _live_baseline(ts: np.ndarray, pos: np.ndarray, current: np.ndarray, fps: int) -> tuple[np.ndarray, np.ndarray]:
    """MotionController._submit_direct with requires_live_baseline: the measured pose at t = 0, every
    plan frame one frame later. The first write is therefore always "stay where you are"."""
    frame_ms = 1000.0 / max(fps, 1)
    return (np.concatenate([[0.0], np.array([round(float(t) + frame_ms, 3) for t in ts])]),
            np.vstack([np.asarray(current, float)[None, :], pos]))


def _safety_validate(ts: np.ndarray, pos: np.ndarray, config: RuntimeConfig) -> tuple[np.ndarray, dict]:
    """MotionSafetyFilter.validate with enforce_velocity (safety.yaml): a frame more than 0.001 outside
    its calibrated range, or a frame pair over safety_max_velocity on any joint, and the runtime raises
    -- nothing streams. Inside the envelope, values are clamped (no-op after the check) and reported."""
    lo = np.array([l for l, _ in config.position_limits], dtype=float)
    hi = np.array([h for _, h in config.position_limits], dtype=float)
    if config.live_baseline and len(ts) < 2:
        raise PlanRefused("Physical motion requires a live position baseline before the first command")
    outside = (pos < lo - 1e-3) | (pos > hi + 1e-3)
    if config.live_baseline and outside.any():
        i, j = np.argwhere(outside)[0]
        raise PlanRefused(f"Motion target outside calibrated safety envelope for {JOINTS[j]}: {pos[i, j]} at frame {i}")
    clamped = np.clip(pos, lo, hi)
    clamped_joints = sorted({JOINTS[j] for j in np.flatnonzero((clamped != pos).any(axis=0))})
    worst = 0.0
    if config.safety_max_velocity > 0:
        for i in range(1, len(ts)):
            dt = (ts[i] - ts[i - 1]) / 1000.0 if ts[i] > ts[i - 1] else 1.0 / 30.0
            v = float(np.max(np.abs(clamped[i] - clamped[i - 1]) / dt))
            worst = max(worst, v)
            if v > config.safety_max_velocity:
                if config.live_baseline:
                    raise PlanRefused(f"Velocity limit exceeded at frame {i}: {v:.0f} units/s > {config.safety_max_velocity}")
    return clamped, {"clamped_joints": clamped_joints, "peak_velocity_units_s": worst}


def simulate_playback(clip, current_pose, current_velocity=None, config: RuntimeConfig = LIVE_CONFIG, *,
                      current_acceleration=None, timestamps_ms=None) -> Streamed:
    """The frames the runtime streams to the servos for `clip`, from the arm's current state.

    clip            (N, 5) units at config.fps, JOINTS order (base_yaw, base_pitch, elbow_pitch,
                    wrist_roll, wrist_pitch): the rows of the CSV the runtime is asked to play.
    current_pose    (5,) the measured pose the runtime reads before fusing (and again for the live
                    baseline; modelled as the same read).
    current_velocity, current_acceleration
                    (5,) units/s and units/s^2, what MotionExecutor.get_motion_state reports from its
                    last writes; None means the executor has no history (or the arm is still).
    timestamps_ms   the CSV's own timestamps relative to its first row, if not the fps grid.

    Returns Streamed(t_ms, positions, decisions): what goes on the bus and when, and a dict of every
    decision on the way -- the skip rule and its inputs, the blend window, the transition duration,
    every clamp that bit, the processing counts, the safety result. Raises PlanRefused where the
    runtime raises. Pure: the same inputs give the same output, bit for bit.
    """
    pos = np.asarray(clip, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != NJ or len(pos) == 0:
        raise ValueError(f"clip must be (N, {NJ}) with N >= 1, got {pos.shape}")
    current = np.asarray(current_pose, dtype=float).reshape(NJ)
    cur_vel = np.zeros(NJ) if current_velocity is None else np.asarray(current_velocity, dtype=float).reshape(NJ)
    cur_acc = np.zeros(NJ) if current_acceleration is None else np.asarray(current_acceleration, dtype=float).reshape(NJ)
    ts = clip_timestamps_ms(len(pos), config.fps) if timestamps_ms is None else np.asarray(timestamps_ms, dtype=float)
    if len(ts) != len(pos):
        raise ValueError("timestamps_ms must have one entry per clip row")

    decisions: dict = {"config": asdict(config), "frames_in": int(len(pos)), "process": None}
    if config.smooth_enabled:
        ts, pos, decisions["process"] = process_recorded(ts, pos, config)
    if config.transition_enabled:
        ts, pos, fusion_dec = fuse(ts, pos, current, cur_vel, cur_acc, config)
        decisions.update(fusion_dec)
    else:
        decisions.update(skipped_blend=True, skip_rule={"reason": "transition disabled"}, blend=None)
    decisions["live_baseline"] = bool(config.live_baseline)
    if config.live_baseline:
        ts, pos = _live_baseline(ts, pos, current, config.fps)
    pos, decisions["safety"] = _safety_validate(ts, pos, config)
    # MotionExecutor.execute: frame i is written at timeline_start + (ts[i] - ts[0]); nothing else.
    t_out = ts - ts[0] + config.start_latency_ms
    decisions["frames_out"] = int(len(t_out))
    decisions["duration_ms"] = float(t_out[-1] - t_out[0])
    decisions["start_latency_ms"] = float(config.start_latency_ms)
    return Streamed(t_out, pos, decisions)


def read_clip(path: str | os.PathLike) -> tuple[np.ndarray, np.ndarray]:
    """(timestamps_ms relative to the first row, (N, 5) rows) from an animation CSV, columns by name
    (lamp_path.read_clip_rows), timestamps the way csv_transformer takes them."""
    from lamp_path import read_clip_rows
    rows = np.array(read_clip_rows(str(path), JOINTS), dtype=float)
    with open(path) as f:
        header = [h.strip() for h in f.readline().split(",")]
        if "timestamp" not in header:
            return clip_timestamps_ms(len(rows)), rows
        k = header.index("timestamp")
        raw = [float(line.split(",")[k]) for line in f if line.strip()]
    first_ms = raw[0] * 1000.0
    return np.array([round(t * 1000.0 - first_ms, 3) for t in raw]), rows


def delivered_summary(streamed: Streamed, joint: str = "base_yaw", every_ms: float = 500.0) -> dict:
    """The stream the way the ground-truth note prints a trace: the joint sampled every 0.5 s, its
    range and its peak, so a model run and a measured run can sit side by side."""
    j = JOINTS.index(joint)
    t, q = streamed.t_ms, streamed.positions[:, j]
    grid = np.arange(0.0, float(t[-1]) + 1e-9, every_ms)
    return {"every_ms": every_ms, "samples": [round(float(v), 1) for v in np.interp(grid, t, q)],
            "min": float(q.min()), "max": float(q.max()), "duration_ms": float(t[-1] - t[0]),
            "frames": int(len(t))}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Print what the runtime would stream for a clip, from home.")
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--pose", default="0,-49,-22,0,30", help="current pose, 5 units")
    ap.add_argument("--fusion-defaults", action="store_true", help="fusion.py's own defaults instead of default.yaml")
    args = ap.parse_args()
    cfg = FUSION_PY_DEFAULTS if args.fusion_defaults else LIVE_CONFIG
    pose = [float(v) for v in args.pose.split(",")]
    for path in args.csv:
        ts, rows = read_clip(path)
        out = simulate_playback(rows, pose, config=cfg, timestamps_ms=ts)
        b = out.decisions["blend"]
        s = delivered_summary(out)
        print(f"{Path(path).stem}: {len(rows)} frames in, {out.decisions['frames_out']} out, {s['duration_ms']:.0f} ms; "
              f"skipped_blend={out.decisions['skipped_blend']}"
              + (f" window [{b['start_index']}, {b['end_index']}] dropped {b['dropped_frames']}" if b else ""))
        print(f"  base_yaw every 0.5 s: {' '.join(f'{v:+.0f}' for v in s['samples'])}   range {s['min']:+.1f}..{s['max']:+.1f}")
