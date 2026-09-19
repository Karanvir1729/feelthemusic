"""The vendor SDK's motion path, modelled: what the lamp does with motion.move and clip.play.

The lamp is borrowed and may only be moved through the vendor's SDK gateway (/api/sdk/v1). This file is a
robot-free model of that gateway's motion path, so head search and lock-in can be designed before the arm
moves. It is design evidence, not hardware approval: nothing here replaces a real SDK outcome.

What SDKMotionModel reproduces (the rules were read in the vendor runtime's source; file:line citations are
relative to that private repository, which is not in this one, and no vendor code is copied here):
  * every motion.move is a rest-to-rest quintic planned from the MEASURED pose, at least 2.0 s long;
  * a per-session rate limit over a sliding 60 s window, whose slot is taken before admission;
  * at most 4 motion tasks at once, and admissions one at a time;
  * a new move pre-empts the running one: the old goal freezes, the old action ends "canceled", and the new
    plan starts about 0.14 s later from the measured pose at zero velocity;
  * a settle check with per-joint tolerances after each move: "succeeded" or "failed";
  * clip.play: an upload (POST /clips) into the vendor's persistent clip store (at most 100 files), then an
    entry move of at least 2.0 s from the measured pose, then the clip, no settle check;
  * compute time that grows with the plan: every validation pass (upload, admission, re-plan) runs the
    vendor's collision check on every row;
  * the vendor's idle animation: paused by any SDK motion, resumed only after a success, crouched;
  * a servo plant: first-order lag with a speed cap, plus gravity sag on the loaded joints (ASSUMPTION;
    pose dependent when given a function, see twin/model.py GravitySag).

The run loop owns the clock: it calls step(t) with a non-decreasing t and submits moves and clips at the
same t. Everything is deterministic: there is no randomness, and results do not depend on how finely the
caller steps, because the servo is integrated in closed form between the model's own event times.

Joint values are SDK units (-100..100 over each calibrated range), time is seconds on the twin's clock.
"""
from __future__ import annotations

import csv
import heapq
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from twin.contract import JOINTS, ActionResult, MotionState, Units

N = len(JOINTS)
TERMINAL = ("succeeded", "failed", "canceled", "rejected")
IDLE = "idle"

# ------------------------------------------------------------------ the planner (sources per line)
FPS = 30.0                     # waypoint rate: vendor source config/default.yaml:104-105; control/runtime.py:37
MAX_VELOCITY = 300.0           # units/s ceiling: vendor source control/safety_filter.py:16, enforced by the
                               # robot's safety.yaml:10 (the /joints route reports it as max_velocity_units_s)
SPEED_FRACTION = 0.5           # vendor source control/safe_motion.py:15 (moves run at half the selected speed)
VELOCITY_HEADROOM = 0.9        # vendor source control/safe_motion.py:114-116 (10% below the ceiling)
MIN_BASELINE_S = 1.0           # vendor source control/safe_motion.py:116; with SPEED_FRACTION -> 2.0 s minimum
QUINTIC_PEAK_SLOPE = 1.875     # peak slope of 10u^3-15u^4+6u^5: vendor source control/safe_motion.py:114
MAX_PLAN_S = 600.0             # vendor source control/safe_motion.py:13
MAX_FRAMES = 18000             # vendor source control/safe_motion.py:12
CALIBRATED_RANGE = (-100.0, 100.0)   # every joint: vendor source robot safety.yaml:3-8

# Settle check after a motion.move. Tolerances: default 2.0 and wider on two joints, "measured on the
# loaded arm" by the vendor: vendor source robot safety.yaml:11-16; applied in control/runtime.py:312-329.
TOLERANCES = {"base_yaw": 2.0, "base_pitch": 2.0, "elbow_pitch": 10.0, "wrist_roll": 2.0, "wrist_pitch": 3.0}
SETTLE_TIMEOUT_S = 1.5         # vendor source control/runtime.py:39
SETTLE_POLL_S = 0.05           # vendor source control/runtime.py:40

# Gateway limits.
CAPACITY = 4                   # running motion tasks + admissions: vendor source sdk_gateway/service.py:392
ADMISSION_TIMEOUT_S = 10.0     # vendor source sdk_gateway/service.py:397
RATE_LIMIT_PER_MIN = 30        # default: vendor source config/default.yaml:451; sdk_gateway/policy.py:175-178.
                               # 120/min is written in the lamp's .env but not in effect until the runtime
                               # restarts (research report pi-internet-deps.md:231).
RATE_WINDOW_S = 60.0           # sliding window per session: vendor source sdk_gateway/policy.py:172-179

# Time from the POST to the first waypoint. Structure: vendor source control/runtime.py:149-174 (admission),
# :197-215 (pre-roll: pause idle, stop playback, sleep 2 frames, read the pose again, re-plan), :431-468
# (a third pose read is written first as a live frame and the plan is shifted by one frame).
HTTP_S = 0.015                 # research report pi-control-surface.md:201 (HTTP on the lamp, measured there)
POSE_READ_S = 0.008            # research report pi-control-surface.md:201 (servo bus read, measured there)
PREROLL_FRAMES = 2             # sleep 2/fps before the re-plan: vendor source control/runtime.py:208

# Compute. Every validation pass runs the vendor's clip parser and its self-collision check on EVERY row of
# the plan: a clip upload (control/runtime.py:143-147, via sdk_gateway/research.py:66), clip.play admission
# (research.py:116-117: the stored CSV is validated again, then the entry + clip plan is checked,
# runtime.py:163), motion.move admission (runtime.py:163) and the re-plan at execution (runtime.py:215).
# Timed on this Mac 2026-09-19 with the vendor's own code (SelfCollisionChecker.validate over 61, 400 and
# 1340 rows: 227-231 us/row; safe_motion.parse_clip over 61 and 1283 rows: 6 us/row).
MAC_ROW_S = 236e-6             # measured on the Mac 2026-09-19 (vendor checker + parser, see above)
PI_SLOWDOWN = 3.0              # ASSUMPTION: the Pi 5 against this Mac, 2-4x; not measured on the lamp
COMPUTE_ROW_S = MAC_ROW_S * PI_SLOWDOWN
COMPUTE_BASE_S = 0.005         # ASSUMPTION: a pass's fixed cost (thread hop, plan objects), not measured
MOVE_ROWS = 61                 # a minimum move: 2.0 s at 30 Hz, both ends (vendor source safe_motion.py:115-135)

# Clip upload and the vendor's clip store. POST /clips sends the CSV body, then validation and the save run
# under the gateway's admission lock (vendor source sdk_gateway/research.py:61-70, the same lock motion
# admissions take) and the store's own lock (clip_store.py:16, :24). The store lives on disk across sessions
# and refuses a save once it holds 100 files, or once its bytes plus twice the new CSV pass 50 MiB
# (clip_store.py:26-32). clips.delete is a resource call (no rate slot) under the store lock (:95-97).
UPLOAD_S_PER_BYTE = 0.00442 / 63308    # research report pi-vision-readiness.md:158: a 63 KB body in 4.4 ms median
FSYNC_S = 0.010                # ASSUMPTION: the save fsyncs the file and its directory (clip_store.py:44-55)
STORE_MAX_FILES = 100          # vendor source sdk_gateway/clip_store.py:26-32
STORE_MAX_BYTES = 50 * 1024 * 1024     # vendor source sdk_gateway/clip_store.py:26-32
STORE_RECORD_OVERHEAD_B = 300  # ASSUMPTION: the JSON record's id, robot and calibration ids and hash
DELETE_S = 0.003               # ASSUMPTION: one unlink on the Pi's SD card

# The vendor idle animation and the "fuse" blend that brings it in.
IDLE_BLEND_S = 1.1             # vendor source config/default.yaml:113 (blend_duration_ms 1100)
IDLE_LOOKAHEAD_S = 1.5         # vendor source config/default.yaml:114 (lookahead_ms 1500)
IDLE_WINDOW_S = 1.5            # vendor source config/default.yaml:115 (blend_window_ms 1500)
IDLE_SKIP_THRESHOLD = 2.0      # no blend when this close: vendor source config/default.yaml:116; fusion.py:376-380
IDLE_MOVING_THRESHOLD = 0.05   # "first moving frame": vendor source motion/fusion.py:126,181

# Synthetic stand-in for the vendor idle, used only when idle.csv cannot be read. It is shaped by summary
# statistics of the vendor clip and nothing else (vendor data animations/factory_v1/idle.csv, summarised in
# the twin motion spec, section 8: a 75.5 s loop, crouched near the sleep pose, these per-joint ranges).
# The trajectory itself is an ASSUMPTION: three slow cosines per joint that loop seamlessly.
SYNTHETIC_IDLE_PERIOD_S = 75.5
SYNTHETIC_IDLE_RANGES = {"base_yaw": (-4.6, 1.0), "base_pitch": (-92.0, -53.0), "elbow_pitch": (-98.0, -60.0),
                         "wrist_roll": (-28.0, 24.0), "wrist_pitch": (-80.0, 54.0)}

# Servo plant. The vendor's own simulated STS3215 caps speed at 140 units/s and quantises readings to 0.05
# (vendor source robot simulation.yaml:8, :13; robots/backends/feetech/simulation.py:187-231). It has no
# gravity. The twin uses a first-order lag instead of its bang-bang loop.
SERVO_TAU_S = 0.05             # ASSUMPTION: of the same order as the vendor simulator, which needs about 0.1 s
                               # for a 1-unit step at 420 units/s^2 (robot simulation.yaml:9)
SERVO_MAX_VELOCITY = 140.0     # vendor source robot simulation.yaml:8
QUANTUM = 0.05                 # vendor source robot simulation.yaml:13
# Gravity sag: a loaded joint holds off its goal by a few units, in the direction gravity pulls it.
# ASSUMPTION, NOT MEASURED. The only bound is the vendor's completion tolerances "measured on the loaded arm"
# (robot safety.yaml:12-16: base_pitch 2, elbow_pitch 10, wrist_pitch 3): the vendor's own moves would fail
# their settle check if the sag were larger. So the twin sweeps it (twin/run.py --sag-sweep). The levels are
# magnitudes at the reference lock pose (a seated head straight ahead); twin/model.py GravitySag scales
# them with the gravity torque of the URDF masses at each pose and signs them by the torque's direction.
SAG_LEVELS = {
    "none": {},
    "assumed": {"base_pitch": 1.5, "elbow_pitch": 6.0, "wrist_pitch": 2.0},    # ASSUMPTION: the twin's first guess
    "tolerance": {"base_pitch": 2.0, "elbow_pitch": 10.0, "wrist_pitch": 3.0},  # = the vendor tolerances
    "1.5x": {"base_pitch": 3.0, "elbow_pitch": 15.0, "wrist_pitch": 4.5},       # 1.5 x the vendor tolerances
}
SAG = SAG_LEVELS["assumed"]
# The direction gravity pulls each loaded joint at the lamp's lock poses, in SDK units: from the gravity
# torque of the vendor URDF's link masses (twin/model.py GravitySag, read at run time; the elbow folds down,
# base_pitch and wrist_pitch are pulled toward +units). Used by the constant (pose-independent) sag only.
SAG_SIGN = {"base_pitch": 1.0, "elbow_pitch": -1.0, "wrist_pitch": 1.0}


def clip_csv(frames) -> bytes:
    """The CSV a client uploads for `frames` [(t, units)]: a header, then one row per frame (timestamps to
    0.1 ms, positions to 0.001 unit). Its size is what the clip store counts."""
    lines = ["timestamp," + ",".join(f"{j}.pos" for j in JOINTS)]
    lines += [f"{float(t):.4f}," + ",".join(f"{float(p[j]):.3f}" for j in JOINTS) for t, p in frames]
    return ("\n".join(lines) + "\n").encode()


def compute_s(rows: int, *, base_s: float = COMPUTE_BASE_S, row_s: float = COMPUTE_ROW_S) -> float:
    """One validation pass over `rows` plan rows on the Pi (see COMPUTE_*)."""
    return base_s + row_s * max(0, int(rows))


def move_start_delay_s(rows: int = MOVE_ROWS) -> float:
    """POST motion.move -> its first waypoint, with free locks: HTTP, pose read, the admission pass, then the
    pre-roll (2 frames, pose read, the re-plan pass, pose read, one prepended frame). A client-side replica."""
    return (HTTP_S + POSE_READ_S + compute_s(rows) + PREROLL_FRAMES / FPS + 2 * POSE_READ_S + compute_s(rows)
            + 1.0 / FPS)


def clip_arbitration_delay_s(clip_rows: int, entry_rows: int = MOVE_ROWS, clip_bytes: int | None = None) -> float:
    """POST /clips -> the moment its clip.play is admitted and freezes whatever plays, with free locks: the
    upload (body, validation, save, reply), then the clip.play POST (validation again, pose read, the entry +
    clip pass). A client-side replica of the model below."""
    size = clip_bytes if clip_bytes is not None else 45 * clip_rows         # about 45 bytes a row (clip_csv)
    plan_rows = entry_rows + clip_rows - 1
    upload = HTTP_S + size * UPLOAD_S_PER_BYTE + compute_s(clip_rows) + FSYNC_S + HTTP_S
    return upload + HTTP_S + compute_s(clip_rows) + POSE_READ_S + compute_s(plan_rows)


def clip_start_delay_s(clip_rows: int, entry_rows: int = MOVE_ROWS, clip_bytes: int | None = None) -> float:
    """POST /clips -> the entry move's first waypoint, with free locks: clip_arbitration_delay_s, then the
    pre-roll with the re-plan over entry + clip. A client-side replica of the model below."""
    plan_rows = entry_rows + clip_rows - 1
    return (clip_arbitration_delay_s(clip_rows, entry_rows, clip_bytes) + PREROLL_FRAMES / FPS + 2 * POSE_READ_S
            + compute_s(plan_rows) + 1.0 / FPS)


class PlanError(ValueError):
    """What the SDK refuses with 422 invalid_request (vendor source sdk_gateway/service.py:403-413)."""


# ------------------------------------------------------------------ the plan, replicable by a client
def ease(u):
    """The SDK's easing, 10u^3 - 15u^4 + 6u^5 on 0..1 (vendor source control/safe_motion.py:124)."""
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    return u ** 3 * (10.0 + u * (-15.0 + 6.0 * u))


def plan_seconds(delta: float, *, max_velocity: float = MAX_VELOCITY, speed_fraction: float = SPEED_FRACTION) -> float:
    """Length of a move whose largest joint travels `delta` units: 2.0 s up to 144 units, then delta/72
    (vendor source control/safe_motion.py:115-117)."""
    baseline = max(MIN_BASELINE_S, QUINTIC_PEAK_SLOPE * abs(delta) / (max_velocity * VELOCITY_HEADROOM))
    return baseline / speed_fraction


def _quintic(current: np.ndarray, target: np.ndarray, fps: float, max_velocity: float,
             speed_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Relative times and waypoints of one rest-to-rest move, straight in joint space, one clock for all joints
    (vendor source control/safe_motion.py:109-135)."""
    seconds = plan_seconds(float(np.max(np.abs(target - current))), max_velocity=max_velocity,
                           speed_fraction=speed_fraction)
    if seconds > MAX_PLAN_S:
        raise PlanError("the move would last longer than the plan limit")
    steps = min(MAX_FRAMES - 1, max(2, math.ceil(seconds * fps)))
    u = np.arange(steps + 1) / steps
    return seconds * u, current + np.outer(ease(u), target - current)


def target_plan(current: Units, target: Units, *, fps: float = FPS, max_velocity: float = MAX_VELOCITY,
                speed_fraction: float = SPEED_FRACTION) -> tuple[np.ndarray, np.ndarray]:
    """The waypoints the SDK would plan from `current` (all five joints) to `target` (any subset; unnamed joints
    keep their current value, vendor source control/safe_motion.py:113). Returns (times in s from the plan's
    start, positions of shape (n, 5) in JOINTS order). A tracker can use this as its client-side replica of
    an in-flight move."""
    cur = np.array([float(current[j]) for j in JOINTS])
    tgt = np.array([float(target.get(j, current[j])) for j in JOINTS])
    return _quintic(cur, tgt, fps, max_velocity, speed_fraction)


def _resample(times: np.ndarray, positions: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Linear resampling onto an even control-rate grid (vendor source control/safe_motion.py:138-165)."""
    duration = float(times[-1])
    steps = min(MAX_FRAMES - 1, max(1, math.ceil(duration * fps)))
    grid = duration * np.arange(steps + 1) / steps
    return grid, np.column_stack([np.interp(grid, times, positions[:, j]) for j in range(positions.shape[1])])


# ------------------------------------------------------------------ the rate budget (shareable with light)
class RateWindow:
    """One SDK session's action budget: at most `per_minute` actions in any sliding 60 s window (vendor source
    sdk_gateway/policy.py:166-187). Every POST /actions that passes policy takes its slot before admission,
    so an action refused later (capacity, 422) still counts; a refused-for-rate request does not
    (sdk_gateway/service.py:307-313). motion.move, clip.play and light.glow on one session share one window:
    pass the same RateWindow to the light model to model that."""

    def __init__(self, per_minute: int = RATE_LIMIT_PER_MIN, window_s: float = RATE_WINDOW_S):
        self.per_minute = max(1, int(per_minute))
        self.window_s = float(window_s)
        self.stamps: list[float] = []

    def _prune(self, t: float) -> None:
        start = t - self.window_s
        self.stamps = [s for s in self.stamps if s >= start]     # the vendor keeps stamps >= window start

    def take(self, t: float) -> bool:
        self._prune(t)
        if len(self.stamps) >= self.per_minute:
            return False
        self.stamps.append(float(t))
        return True

    def used(self, t: float) -> int:
        self._prune(t)
        return len(self.stamps)


# ------------------------------------------------------------------ the idle trajectory
def load_vendor_idle(robot_dir: str | Path | None = None, fps: float = FPS) -> tuple[np.ndarray, np.ndarray] | None:
    """The vendor's idle clip (animations/factory_v1/idle.csv in the robot package), read at run time from
    FTM_ROBOT_DIR and resampled to the control rate. None when it is not there or not readable. It is vendor
    data: never copy it into this repository."""
    robot_dir = robot_dir or os.environ.get("FTM_ROBOT_DIR")
    if not robot_dir:
        return None
    path = Path(robot_dir) / "animations" / "factory_v1" / "idle.csv"
    try:
        with path.open(newline="", encoding="utf-8-sig") as f:
            rows = [row for row in csv.reader(f) if row]
        header = rows[0]
        columns = [header.index("timestamp")] + [header.index(f"{j}.pos") for j in JOINTS]
        data = np.array([[float(row[c]) for c in columns] for row in rows[1:]], dtype=float)
    except (OSError, ValueError, IndexError, csv.Error):
        return None
    if len(data) < 2 or not np.all(np.isfinite(data)):
        return None
    times = data[:, 0] - data[0, 0]
    if not np.all(np.diff(times) > 0):
        return None
    return _resample(times, data[:, 1:], fps)


def synthetic_idle(fps: float = FPS) -> tuple[np.ndarray, np.ndarray]:
    """A slow crouched wander standing in for the vendor idle (see SYNTHETIC_IDLE_*). It loops seamlessly:
    every cosine has a whole number of cycles per period."""
    n = round(SYNTHETIC_IDLE_PERIOD_S * fps)
    t = np.arange(n + 1) * (SYNTHETIC_IDLE_PERIOD_S / n)
    w = 2.0 * np.pi * t / SYNTHETIC_IDLE_PERIOD_S
    out = np.empty((n + 1, N))
    for j, name in enumerate(JOINTS):
        lo, hi = SYNTHETIC_IDLE_RANGES[name]
        phase = 0.9 * j
        wave = 0.5 * np.cos(w) + 0.3 * np.cos(4 * w + phase) + 0.2 * np.cos(9 * w + 2 * phase)
        out[:, j] = lo + (hi - lo) * (0.5 - 0.5 * wave)
    return t, out


# ------------------------------------------------------------------ the servo
def _servo(x: np.ndarray, setpoint: np.ndarray, dt: float, tau: float, vmax: float) -> np.ndarray:
    """First-order lag toward `setpoint` with a speed cap, solved exactly over dt with a constant setpoint:
    at the cap while the error exceeds vmax*tau, then an exponential decay with time constant tau."""
    if dt <= 0.0:
        return x.copy()
    err = setpoint - x
    mag, sign = np.abs(err), np.sign(err)
    if tau <= 0.0:
        return x + sign * np.minimum(mag, vmax * dt)
    knee = vmax * tau
    t_capped = np.maximum(mag - knee, 0.0) / vmax
    return np.where(dt <= t_capped, x + sign * vmax * dt,
                    setpoint - sign * np.minimum(mag, knee) * np.exp(-(dt - t_capped) / tau))


# ------------------------------------------------------------------ records
@dataclass
class MotionAction:
    """One SDK action as the twin saw it. Times are on the twin's clock; None means not reached."""
    action_id: str
    kind: str                                  # "move" | "clip"
    t_post: float
    requested: dict                            # move: the joints the client named; clip: {}
    planned_duration_s: float = 0.0            # admission-time plan length, as the SDK reports it (no pre-roll/settle)
    rows: int = 0                              # rows of the admitted plan: the re-plan's collision pass costs this
    clip_id: str = ""                          # clip.play: the stored clip it plays
    t_upload: float | None = None              # clip.play: the upload POST that came before it
    state: str = "admitted"                    # admitted -> preroll -> running -> settling (moves) -> an outcome
    outcome: str | None = None                 # succeeded | failed | canceled | rejected
    reason: str = ""
    t_arbitrated: float | None = None          # stop_all(): whatever was playing froze here
    t_preroll: float | None = None             # pause idle, stop playback, 2-frame sleep starts
    t_replan: float | None = None              # the pose read that the executed plan starts from
    replan_from: dict | None = None            # that measured pose
    t_live_frame: float | None = None          # a third pose read is written as the first goal
    t_plan_start: float | None = None          # the plan's first waypoint (ease 0), one frame later
    executed_duration_s: float | None = None   # length of the plan actually executed (entry + clip for clips)
    plan_times: np.ndarray | None = None       # absolute write times, live frame first
    plan_positions: np.ndarray | None = None   # goals written, shape (n, 5), JOINTS order
    target: dict | None = None                 # the last waypoint: what the settle check compares against
    t_last_write: float | None = None
    settle_errors: dict | None = None          # errors at the settle sample that decided the outcome
    settle_actual: dict | None = None          # the measured pose at that sample (the SDK's failure reports it)
    t_finished: float | None = None
    clip: tuple | None = field(default=None, repr=False)        # (times, positions) as uploaded
    generation: int = field(default=-1, repr=False)
    superseded: bool = field(default=False, repr=False)
    next_check: float = field(default=math.inf, repr=False)     # next moment its settle loop looks up
    plan_rel: tuple | None = field(default=None, repr=False)    # re-planned (times, positions) before timing


@dataclass
class _Playback:
    owner: str                                 # an action id, or IDLE
    times: np.ndarray
    positions: np.ndarray
    next_i: int = 0


# ------------------------------------------------------------------ the model
class SDKMotionModel:
    """contract.MotionLike for the vendor SDK: motion.move and clip.play on one SDK session.

    neutral: the pose the arm holds at t0 (all five joints, SDK units).
    limits: joint -> (lo, hi) that targets must lie in. The SDK itself refuses only outside the calibrated
        range (-100..100); pass tighter limits only to model a client envelope. The measured pose is always
        checked against `calibrated_range`, as the SDK does (vendor source control/safe_motion.py:112).
    pose_ok: the SDK's self-collision check, pose -> True when allowed, called on every waypoint at
        admission and at re-plan (vendor source control/runtime.py:163, :215). The vendor checks only the
        head shade against the base cylinder; do not pass a table check here, that belongs to the client.
    idle: "auto" (the vendor idle.csv from robot_dir or FTM_ROBOT_DIR if readable, else the synthetic
        wander), "vendor", "synthetic", None (no idle: an owner-side lamp change nobody has approved), or
        (times, positions) arrays. self.idle_source says which one is in use.
    sag: None (the constant SAG, signed by SAG_SIGN), a dict of constant magnitudes (same signs), or a
        function goal (np.ndarray, JOINTS order) -> signed deflection (np.ndarray, units) for a
        pose-dependent sag such as twin/model.py GravitySag.
    clip store: store_initial_files / store_initial_bytes are clips already on the lamp's disk from earlier
        sessions (the store persists); submit_clip refuses once the store is full, delete_clip frees a slot.
    Every other keyword is one of the constants above; the defaults carry their sources
    (servo_max_velocity 0 or None means no speed cap).

    MotionState.finished lists outcomes of ACCEPTED actions only: succeeded, failed (settle), canceled
    (pre-empted or cancelled) and rejected (refused at the execution-time re-plan). A refusal at the POST
    (429 rate, 503 capacity, 422 invalid) is reported once, in the ActionResult. self.actions keeps a
    MotionAction record of every action, with its stage times, for evidence and tests.
    """

    def __init__(self, neutral: Units, limits: dict | None = None, *,
                 fps: float = FPS, max_velocity: float = MAX_VELOCITY, speed_fraction: float = SPEED_FRACTION,
                 tolerances: dict | None = None, settle_timeout_s: float = SETTLE_TIMEOUT_S,
                 settle_poll_s: float = SETTLE_POLL_S,
                 http_s: float = HTTP_S, pose_read_s: float = POSE_READ_S,
                 compute_base_s: float = COMPUTE_BASE_S, compute_row_s: float = COMPUTE_ROW_S,
                 preroll_sleep_s: float | None = None, admission_timeout_s: float = ADMISSION_TIMEOUT_S,
                 rate_limit_per_min: int = RATE_LIMIT_PER_MIN, rate_window: RateWindow | None = None,
                 capacity: int = CAPACITY,
                 upload_s_per_byte: float = UPLOAD_S_PER_BYTE, fsync_s: float = FSYNC_S, delete_s: float = DELETE_S,
                 store_max_files: int = STORE_MAX_FILES, store_max_bytes: int = STORE_MAX_BYTES,
                 store_initial_files: int = 0, store_initial_bytes: int = 0,
                 sag=None, servo_tau_s: float = SERVO_TAU_S,
                 servo_max_velocity: float = SERVO_MAX_VELOCITY, quantum: float = QUANTUM,
                 calibrated_range: tuple[float, float] = CALIBRATED_RANGE,
                 idle="auto", robot_dir: str | Path | None = None, idle_at_start: bool = True,
                 idle_blend_s: float = IDLE_BLEND_S, idle_lookahead_s: float = IDLE_LOOKAHEAD_S,
                 idle_window_s: float = IDLE_WINDOW_S, idle_skip_threshold: float = IDLE_SKIP_THRESHOLD,
                 pose_ok: Callable[[Units], bool] | None = None, t0: float = 0.0):
        self.neutral = {j: float(neutral[j]) for j in JOINTS}
        self.limits = {j: tuple(map(float, (limits or {}).get(j, calibrated_range))) for j in JOINTS}
        self.calibrated_range = (float(calibrated_range[0]), float(calibrated_range[1]))
        self.fps, self.max_velocity, self.speed_fraction = float(fps), float(max_velocity), float(speed_fraction)
        self.tolerances = {**TOLERANCES, **(tolerances or {})}
        self.settle_timeout_s, self.settle_poll_s = float(settle_timeout_s), float(settle_poll_s)
        self.http_s, self.pose_read_s = float(http_s), float(pose_read_s)
        self.compute_base_s, self.compute_row_s = float(compute_base_s), float(compute_row_s)
        self.preroll_sleep_s = PREROLL_FRAMES / self.fps if preroll_sleep_s is None else float(preroll_sleep_s)
        self.admission_timeout_s = float(admission_timeout_s)
        self.rate = rate_window if rate_window is not None else RateWindow(rate_limit_per_min)
        self.capacity = int(capacity)
        self.upload_s_per_byte, self.fsync_s, self.delete_s = float(upload_s_per_byte), float(fsync_s), float(delete_s)
        self.store_max_files, self.store_max_bytes = int(store_max_files), int(store_max_bytes)
        # The clip store: id -> bytes on disk. Clips left by earlier sessions stay there (the store persists).
        n_old = max(0, int(store_initial_files))
        self.store: dict[str, int] = {f"earlier-{k:03d}": int(store_initial_bytes) // max(1, n_old)
                                      for k in range(n_old)}
        self._store_free_at = -math.inf          # the store's lock (vendor clip_store.py:16)
        self._clips = 0
        # Sag: a function of the goal (pose dependent), or constant magnitudes signed by SAG_SIGN.
        self._sag_fn = sag if callable(sag) else None
        levels = {} if callable(sag) else ({**SAG} if sag is None else {k: float(v) for k, v in sag.items()})
        self.sag = {j: 0.0 for j in JOINTS} | levels
        # (a joint without a known direction is taken to sag toward -units, as the vendor's crouch does)
        self._sag_const = np.array([SAG_SIGN.get(j, -1.0) * self.sag[j] for j in JOINTS])
        self._sag_cache: tuple[bytes, np.ndarray] | None = None
        self.servo_tau_s, self.quantum = float(servo_tau_s), float(quantum)
        self.servo_max_velocity = float(servo_max_velocity) if servo_max_velocity else 1e9
        self.idle_blend_s, self.idle_lookahead_s = float(idle_blend_s), float(idle_lookahead_s)
        self.idle_window_s, self.idle_skip_threshold = float(idle_window_s), float(idle_skip_threshold)
        self.pose_ok = pose_ok

        self._idle_clip, self.idle_source = self._pick_idle(idle, robot_dir)

        # State: the servo position x and the goal being written, both committed at time self._t.
        self._t = float(t0)
        self._clock = float(t0)                  # latest time any caller used; time never goes back
        self._goal = self._vec(self.neutral)
        self._x = self._setpoint(self._goal)     # at rest at t0, already sagged
        self._exec: _Playback | None = None
        self._heap: list = []
        self._seq = 0
        self._generation = 0                     # bumped by stop_all / stop_playback (vendor runtime.py:758-760,
                                                 # :898-900)
        self._idle_paused = False
        self._foreground: str | None = None      # the SDK action that owns the arm, or is waiting to
        self._admission_free_at = -math.inf      # admissions are serialised (vendor sdk_gateway/research.py:77)
        self._preroll_free_at = -math.inf        # pre-rolls are serialised (vendor control/runtime.py:199)
        self._finished: list[tuple[str, str]] = []
        self._open: list[str] = []               # accepted, not yet terminal, in POST order
        self._count = 0
        self.actions: dict[str, MotionAction] = {}
        self.log: list[tuple[float, str, str, str]] = []   # (t, event, action id or "idle", detail)
        self.counters = {"posted": 0, "accepted": 0, "rate_limited": 0, "capacity": 0, "invalid": 0,
                         "upload_refused": 0, "store_full": 0, "uploads": 0, "deletes": 0,
                         "succeeded": 0, "failed": 0, "canceled": 0, "rejected": 0}
        self.store_peak = len(self.store)
        if self._idle_clip is not None and idle_at_start:
            self._schedule(self._t, "idle_start", IDLE)

    # ---------------------------------------------------------------- derived timings
    def compute_s(self, rows: int) -> float:
        """One validation pass over `rows` plan rows (see COMPUTE_*)."""
        return self.compute_base_s + self.compute_row_s * max(0, int(rows))

    @property
    def admission_delay_s(self) -> float:
        """POST to arbitration (the moment whatever was playing freezes) of a minimum move, free locks."""
        return self.http_s + self.pose_read_s + self.compute_s(MOVE_ROWS)

    def preroll_s_for(self, rows: int) -> float:
        """Freeze to the plan's first waypoint: sleep, read, re-plan over `rows`, read, one prepended frame."""
        return self.preroll_sleep_s + 2 * self.pose_read_s + self.compute_s(rows) + 1.0 / self.fps

    @property
    def preroll_s(self) -> float:
        """preroll_s_for a minimum move (61 rows)."""
        return self.preroll_s_for(MOVE_ROWS)

    def completion_bound_s(self, planned_duration_s: float) -> float:
        """Longest a lone accepted motion.move can take from its POST to its outcome, without pre-emption or
        a busy admission lock. Use it for a completion watchdog. A re-plan from a pose that moved during the
        pre-roll can make the executed plan a little longer than planned_duration_s for deltas over 144."""
        rows = math.ceil(float(planned_duration_s) * self.fps) + 1
        return (self.http_s + self.pose_read_s + self.compute_s(rows) + self.preroll_s_for(rows)
                + float(planned_duration_s) + self.settle_timeout_s)

    # ---------------------------------------------------------------- contract.MotionLike
    def submit_move(self, t: float, target: Units) -> ActionResult:
        """POST motion.move. The result is what the 202 (or the refusal) would say."""
        self._run_until(t)
        self.counters["posted"] += 1
        return self._admit(t, "move", target, None)

    def submit_clip(self, t: float, frames: list[tuple[float, Units]]) -> ActionResult:
        """POST /clips at t (a resource call: no rate slot), then POST clip.play as soon as the upload's reply
        is back (one action, one rate slot, taken at that later time).

        The upload holds the admission lock while the vendor validates every row, and the store lock while it
        saves with two fsyncs (sdk_gateway/research.py:61-70, clip_store.py:23-59); a full store refuses it
        after the validation. The ActionResult carries the stored clip's id (also when clip.play is refused
        afterwards: the file stays on the lamp until someone deletes it) and t_posted, the clip.play POST."""
        self._run_until(t)
        try:
            clip = self._validate_clip(frames)
        except PlanError as exc:
            self.counters["upload_refused"] += 1
            self.log.append((t, "upload_refused", "", str(exc)))
            return ActionResult(False, "", f"422 clip upload refused: {exc}")
        data = clip_csv(frames)
        rows = len(frames)
        arrive = t + self.http_s + len(data) * self.upload_s_per_byte
        validated = max(arrive, self._admission_free_at) + self.compute_s(rows)
        saved = max(validated, self._store_free_at) + self.fsync_s
        self._admission_free_at, self._store_free_at = saved, saved
        self.counters["uploads"] += 1
        stored = sum(self.store.values())
        if len(self.store) >= self.store_max_files or stored + 2 * len(data) > self.store_max_bytes:
            self.counters["upload_refused"] += 1
            self.counters["store_full"] += 1
            reason = "422 clip upload refused: the lamp's clip store is full (delete unused clips)"
            self.log.append((t, "upload_refused", "", reason))
            return ActionResult(False, "", reason)
        self._clips += 1
        clip_id = f"clip-{self._clips:04d}"
        # on disk: the JSON record holds the CSV as a string (each newline escaped: one more byte a row)
        self.store[clip_id] = len(data) + data.count(b"\n") + STORE_RECORD_OVERHEAD_B
        self.store_peak = max(self.store_peak, len(self.store))
        self.log.append((t, "uploaded", clip_id, f"{rows} rows, {len(data)} B, saved at {saved:.3f}"))
        t_play = saved + self.http_s                                  # the reply is back; clip.play goes out
        self.counters["posted"] += 1
        result = self._admit(t_play, "clip", {}, clip, clip_rows=rows)
        if result.action_id in self.actions:
            self.actions[result.action_id].clip_id = clip_id
            self.actions[result.action_id].t_upload = float(t)
        return ActionResult(result.accepted, result.action_id, result.reason, result.planned_duration_s,
                            clip_id=clip_id, t_posted=t_play)

    def delete_clip(self, t: float, clip_id: str) -> float:
        """DELETE /clips/{id} at t (a resource call: no rate slot). The file goes when the store lock is free
        (clip_store.py:95-97); returns that time. Deleting an id that is not there does nothing."""
        self._run_until(t)
        done = max(t + self.http_s, self._store_free_at) + self.delete_s
        self._store_free_at = done
        if self.store.pop(clip_id, None) is not None:
            self.counters["deletes"] += 1
            self.log.append((t, "deleted", clip_id, f"done at {done:.3f}"))
        return done

    def cancel(self, t: float, action_id: str) -> None:
        """POST /actions/{id}/cancel: not rate limited. It reaches the lamp http_s later; the arm then holds the
        goal it was last given and idle stays paused (vendor source sdk_gateway/service.py:524-565)."""
        self._run_until(t)
        if action_id in self.actions:
            self._schedule(t + self.http_s, "client_cancel", action_id)

    def step(self, t: float) -> MotionState:
        self._run_until(t)
        finished, self._finished = self._finished, []
        return MotionState(commanded=self._units(self._goal), measured=self._units(self._measure(t)),
                           active_action=self._open[-1] if self._open else None,
                           idle_playing=self._exec is not None and self._exec.owner == IDLE, finished=finished)

    # ---------------------------------------------------------------- reporting
    def stats(self, t: float | None = None) -> dict:
        """Request and outcome counts for a scorecard, and this session's rate window use at t."""
        t = self._clock if t is None else t
        return {**self.counters, "window_used": self.rate.used(t), "window_limit": self.rate.per_minute,
                "idle_source": self.idle_source, "store_files": len(self.store), "store_peak_files": self.store_peak,
                "store_bytes": sum(self.store.values())}

    # ---------------------------------------------------------------- admission
    def _admit(self, t: float, kind: str, target, clip, clip_rows: int = 0) -> ActionResult:
        """The gateway's order (vendor source sdk_gateway/service.py:307-449): rate limit, capacity, then an
        admission that validates, reads the pose, plans and collision-checks every waypoint. For clip.play
        the admission first validates the stored CSV again (sdk_gateway/research.py:107-117)."""
        # 1. Rate limit: refused requests take no slot and create no action record.
        if not self.rate.take(t):
            self.counters["rate_limited"] += 1
            self.log.append((t, "rate_limited", "", kind))
            return ActionResult(False, "", "429 rate_limited: too many SDK actions in the last 60 s")
        # 2. Capacity: running motion tasks plus admissions in progress (a pre-empted task ends quickly).
        if len(self._open) >= self.capacity:
            return self._refuse_record(t, kind, "capacity",
                                       "503 capability_unavailable: no free safe-motion slot")
        # Simplifications: the SDK reads the pose http_s + pose_read_s after the POST (and a clip.play POST
        # comes after its upload); submit_* returns at once, so the twin reads the pose at the call. Only
        # the reported duration can differ (above 144 units) and the admission's own collision pass: the
        # executed plan is planned and checked again from the pose measured at the re-plan.
        current = self._measure(min(t, self._clock))
        try:
            requested = self._validate_positions(target, complete=False) if kind == "move" else {}
            self._check_current(current)
            times, positions = self._plan(kind, current, requested, clip)
            self._check_collisions(positions)
        except PlanError as exc:
            return self._refuse_record(t, kind, "invalid", f"422 invalid_request: {exc}")
        # 3. Admission, one at a time (sdk_gateway/research.py:77), capped at 10 s: the stored clip's
        # validation (clip.play), a pose read, then the collision pass over every row of the plan.
        start = max(t + self.http_s, self._admission_free_at)
        done = start + (self.compute_s(clip_rows) if kind == "clip" else 0.0) + self.pose_read_s \
            + self.compute_s(len(times))
        if done - (t + self.http_s) > self.admission_timeout_s:
            return self._refuse_record(t, kind, "invalid", "422 invalid_request: admission timed out")
        self._admission_free_at = done
        action = self._new_action(t, kind, requested)
        action.planned_duration_s = float(times[-1])
        action.rows = len(times)
        action.clip = clip
        self._open.append(action.action_id)
        self.counters["accepted"] += 1
        self.log.append((t, "accepted", action.action_id, f"planned {times[-1]:.3f} s"))
        self._schedule(done, "arbitrate", action.action_id)
        return ActionResult(True, action.action_id, "", action.planned_duration_s)

    def _refuse_record(self, t, kind, counter, reason) -> ActionResult:
        # The gateway creates an action record and marks it REJECTED (service.py:422-425); the client learns it
        # from the response, so it is not reported again in MotionState.finished.
        self.counters[counter] += 1
        action = self._new_action(t, kind, {})
        action.state = action.outcome = "rejected"
        action.reason, action.t_finished = reason, t
        self.log.append((t, "refused", action.action_id, reason))
        return ActionResult(False, action.action_id, reason)

    def _new_action(self, t, kind, requested) -> MotionAction:
        self._count += 1
        action = MotionAction(f"{kind}-{self._count:04d}", kind, float(t), dict(requested))
        self.actions[action.action_id] = action
        return action

    # ---------------------------------------------------------------- validation and planning
    def _validate_positions(self, positions, *, complete: bool) -> dict:
        # vendor source control/safe_motion.py:18-46
        if not isinstance(positions, dict) or not positions:
            raise PlanError("positions are missing")
        if complete and set(positions) != set(JOINTS):
            raise PlanError("every joint needs a position")
        out = {}
        for name, raw in positions.items():
            if name not in JOINTS:
                raise PlanError(f"no joint named {name}")
            if isinstance(raw, bool):
                raise PlanError(f"{name} is not a finite number")
            try:
                value = float(raw)
            except (TypeError, ValueError):
                raise PlanError(f"{name} is not a finite number") from None
            if not math.isfinite(value):
                raise PlanError(f"{name} is not a finite number")
            lo, hi = self.limits[name]
            if not lo <= value <= hi:
                raise PlanError(f"{name} is out of its calibrated range")
            out[name] = value
        return out

    def _check_current(self, current: np.ndarray) -> None:
        lo, hi = self.calibrated_range
        if np.any(current < lo) or np.any(current > hi):
            raise PlanError("Joint is out of its calibrated range (measured pose)")

    def _validate_clip(self, frames) -> tuple[np.ndarray, np.ndarray]:
        # The upload checks (vendor source control/safe_motion.py:49-106): 2..18000 frames, first stamp
        # subtracted, strictly increasing, at most 600 s, all five joints in range, at most 300 units/s between
        # rows; then self-collision on every row (control/runtime.py:143-147). The 5 MiB CSV cap cannot bind:
        # 18000 rows of six numbers is well under it.
        if not isinstance(frames, (list, tuple)) or not 2 <= len(frames) <= MAX_FRAMES:
            raise PlanError("a clip needs 2 to 18000 frames")
        try:
            stamps = np.array([float(ts) for ts, _ in frames], dtype=float)
        except (TypeError, ValueError):
            raise PlanError("timestamp is not a finite number") from None
        if not np.all(np.isfinite(stamps)):
            raise PlanError("timestamp is not a finite number")
        times = stamps - stamps[0]
        positions = np.array([self._vec(self._validate_positions(p, complete=True)) for _, p in frames])
        dt = np.diff(times)
        if np.any(dt <= 0):
            raise PlanError("timestamps do not increase")
        if times[-1] > MAX_PLAN_S:
            raise PlanError("the clip is longer than 600 s")
        speed = np.abs(np.diff(positions, axis=0)) / dt[:, None]
        if np.any(speed > self.max_velocity + 1e-6):
            raise PlanError(f"{JOINTS[int(np.argmax(speed.max(axis=0)))]} would move faster than the velocity limit")
        self._check_collisions(positions)
        return times, positions

    def _plan(self, kind: str, current: np.ndarray, requested: dict, clip) -> tuple[np.ndarray, np.ndarray]:
        if kind == "move":
            target = current.copy()
            for name, value in requested.items():             # partial sets keep the MEASURED value
                target[JOINTS.index(name)] = value
            return _quintic(current, target, self.fps, self.max_velocity, self.speed_fraction)
        # clip.play: resample to the control rate, then an entry move from the measured pose to the first
        # frame, which lasts at least 2.0 s even when the clip starts where the arm is
        # (vendor source control/safe_motion.py:168-183).
        clip_t, clip_p = _resample(clip[0], clip[1], self.fps)
        entry_t, entry_p = _quintic(current, clip_p[0], self.fps, self.max_velocity, self.speed_fraction)
        return (np.concatenate([entry_t, entry_t[-1] + clip_t[1:]]),
                np.vstack([entry_p, clip_p[1:]]))

    def _check_collisions(self, positions: np.ndarray) -> None:
        if self.pose_ok is None:
            return
        for row in positions:
            if not self.pose_ok(self._units(row)):
                raise PlanError("Planned motion rejected: head could hit the base")

    # ---------------------------------------------------------------- the event loop
    def _schedule(self, t: float, kind: str, owner: str) -> None:
        self._seq += 1
        heapq.heappush(self._heap, (float(t), self._seq, kind, owner))

    def _run_until(self, t: float) -> None:
        t = float(t)
        if t < self._clock - 1e-12:
            raise ValueError(f"time went backwards: {t} < {self._clock}")
        self._clock = max(self._clock, t)
        while True:
            t_event = self._heap[0][0] if self._heap else math.inf
            t_write = self._exec.times[self._exec.next_i] if self._exec is not None else math.inf
            t_next = min(t_event, t_write)
            if t_next > t:
                return
            self._commit(t_next)
            if t_event <= t_write:                               # on a tie the pipeline acts first
                _, _, kind, owner = heapq.heappop(self._heap)
                getattr(self, "_on_" + kind)(owner)
            else:
                self._write()

    def _commit(self, t: float) -> None:
        if t > self._t:
            self._x = self._servo_at(t)
            self._t = t

    def _write(self) -> None:
        """The executor writes the next goal; goals are held between writes (vendor control/executor.py:143-215)."""
        pb = self._exec
        self._goal = pb.positions[pb.next_i].copy()
        pb.next_i += 1
        if pb.next_i < len(pb.times):
            return
        self._exec = None
        if pb.owner == IDLE:
            # The action queue plays idle again at once while it is not paused (vendor action_queue.py:234-252).
            self._schedule(self._t + 2 * self.pose_read_s, "idle_start", IDLE)
            return
        action = self.actions[pb.owner]
        action.t_last_write = self._t
        if action.kind == "clip":                                # clips have no settle check (runtime.py:229)
            self._finish(action, "succeeded")
            return
        action.state = "settling"
        action.next_check = self._t
        self._schedule(self._t + self.pose_read_s, "settle_poll", action.action_id)

    def _stop_executor(self) -> _Playback | None:
        """Stop writing: the servo keeps the last goal written; there is no slow-down ramp."""
        stopped, self._exec = self._exec, None
        return stopped

    # ---------------------------------------------------------------- the pipeline, one handler per stage
    def _on_arbitrate(self, action_id: str) -> None:
        # Behaviour arbitration (priority 85, lease sdk_research) calls stop_all(): whatever was playing, idle
        # or an earlier SDK move, freezes at its last written goal (vendor sdk_gateway/research.py:127-147;
        # motion_manager.py:122-128; runtime.py:898-905).
        action = self.actions[action_id]
        if action.state != "admitted":
            return                                               # cancelled by the client on the way
        action.t_arbitrated = self._t
        stopped = self._stop_executor()
        self._generation += 1
        previous = self.actions.get(self._foreground) if self._foreground else None
        if previous is not None and previous.state not in TERMINAL:
            self._supersede(previous, stopped)
        self._foreground = action_id
        self.log.append((self._t, "freeze", action_id, stopped.owner if stopped else ""))
        self._schedule(max(self._t, self._preroll_free_at), "preroll", action_id)

    def _supersede(self, old: MotionAction, stopped: _Playback | None) -> None:
        """The pre-empted action ends 'canceled' when its own loop next looks (vendor runtime.py:216-224,
        :276-277; service.py:712-714). Simplification: an action still waiting for the pre-roll lock is
        cancelled at once (in the vendor it may start and be stopped a moment later)."""
        old.superseded = True
        when = self._t
        if old.state == "preroll":
            when = max(self._t, old.t_preroll + self.preroll_sleep_s + self.pose_read_s + self.compute_s(old.rows))
        elif old.state == "running" and stopped is not None and stopped.owner == old.action_id:
            when = float(stopped.times[stopped.next_i])          # the executor loop wakes for its next write
        elif old.state == "settling":
            when = max(self._t, old.next_check)
        self._schedule(when, "superseded", old.action_id)

    def _on_superseded(self, action_id: str) -> None:
        self._finish(self.actions[action_id], "canceled", "cancelled_or_preempted")

    def _on_preroll(self, action_id: str) -> None:
        # Pause idle, stop playback, sleep two frames (vendor control/runtime.py:199-208).
        action = self.actions[action_id]
        if action.state != "admitted" or action.superseded:
            return
        action.state, action.t_preroll = "preroll", self._t
        self._idle_paused = True
        self._stop_executor()
        self._generation += 1
        action.generation = self._generation
        self._preroll_free_at = self._t + self.preroll_sleep_s + self.pose_read_s + self.compute_s(action.rows)
        self._schedule(self._t + self.preroll_sleep_s + self.pose_read_s, "replan", action_id)

    def _on_replan(self, action_id: str) -> None:
        # Read the pose again and plan again from it, at zero velocity, same absolute target; collision-check
        # again (vendor control/runtime.py:209-215). A refusal here ends the action "rejected" (service.py:715-729).
        action = self.actions[action_id]
        if action.state != "preroll" or action.superseded:
            return
        current = self._measure(self._t)
        action.t_replan, action.replan_from = self._t, self._units(current)
        try:
            self._check_current(current)
            times, positions = self._plan(action.kind, current, action.requested, action.clip)
            self._check_collisions(positions)
        except PlanError as exc:
            action.reason = f"422 invalid_request: {exc}"
            self._schedule(self._t + self.compute_s(action.rows), "rejected", action_id)
            return
        action.plan_rel = (times, positions)
        self._schedule(self._t + self.compute_s(action.rows) + self.pose_read_s, "live_frame", action_id)

    def _on_rejected(self, action_id: str) -> None:
        action = self.actions[action_id]
        self._finish(action, "rejected", action.reason)

    def _on_live_frame(self, action_id: str) -> None:
        # A third pose read is written first, and the plan follows one frame later (vendor runtime.py:431-468).
        action = self.actions[action_id]
        if action.state != "preroll" or action.superseded:
            return
        live = self._measure(self._t)
        rel_t, rel_p = action.plan_rel
        frame = 1.0 / self.fps
        action.plan_times = np.concatenate([[self._t], self._t + frame + rel_t])
        action.plan_positions = np.vstack([live, rel_p])
        action.t_live_frame, action.t_plan_start = self._t, self._t + frame
        action.executed_duration_s = float(rel_t[-1])
        action.target = self._units(rel_p[-1])
        action.state = "running"
        self._exec = _Playback(action_id, action.plan_times, action.plan_positions)
        self.log.append((self._t, "plan_start", action_id, f"{rel_t[-1]:.3f} s"))

    def _on_settle_poll(self, action_id: str) -> None:
        # Poll the measured pose every 50 ms for up to 1.5 s; succeed at the first sample with every joint in
        # tolerance, else fail with the arm holding the goal (vendor control/runtime.py:262-310).
        action = self.actions[action_id]
        if action.state != "settling" or action.superseded:
            return
        actual = self._measure(self._t)
        target = self._vec(action.target)
        errors = np.abs(actual - target)
        action.settle_errors = self._units(errors)
        action.settle_actual = self._units(actual)       # a failure reports it (control/runtime.py:43-67)
        tol = self._vec(self.tolerances)
        if np.all(errors <= tol):
            self._finish(action, "succeeded")
            return
        deadline = action.t_last_write + self.settle_timeout_s
        next_loop = self._t + self.settle_poll_s
        if next_loop + self.pose_read_s > deadline:
            outside = ", ".join(j for j, e, k in zip(JOINTS, errors, tol, strict=True) if e > k)
            action.reason = ("the arm stopped outside its settle tolerance: " + outside)
            action.next_check = deadline
            self._schedule(deadline, "settle_failed", action_id)
            return
        action.next_check = next_loop
        self._schedule(next_loop + self.pose_read_s, "settle_poll", action_id)

    def _on_settle_failed(self, action_id: str) -> None:
        action = self.actions[action_id]
        if action.state == "settling" and not action.superseded:
            self._finish(action, "failed", action.reason)

    def _on_client_cancel(self, action_id: str) -> None:
        action = self.actions[action_id]
        if action.state in TERMINAL:
            return
        if self._exec is not None and self._exec.owner == action_id:
            self._stop_executor()                                # the arm holds the last goal written
        if action.state == "preroll":
            self._preroll_free_at = min(self._preroll_free_at, self._t)   # the pre-roll lock is let go
        self._finish(action, "canceled", "cancelled_or_preempted")

    def _finish(self, action: MotionAction, outcome: str, reason: str = "") -> None:
        if action.state in TERMINAL:
            return
        action.state = action.outcome = outcome
        action.t_finished = self._t
        if reason:
            action.reason = reason
        if action.action_id in self._open:
            self._open.remove(action.action_id)
        if self._foreground == action.action_id:
            self._foreground = None
        self.counters[outcome] += 1
        self._finished.append((action.action_id, outcome))
        self.log.append((self._t, outcome, action.action_id, action.reason))
        # Idle resumes only after a completed move or clip, and only if nothing newer stopped playback since
        # (vendor control/runtime.py:255-260). canceled, failed and rejected leave idle paused.
        if outcome == "succeeded" and action.generation == self._generation:
            self._idle_paused = False
            self._schedule(self._t + 2 * self.pose_read_s, "idle_start", IDLE)

    # ---------------------------------------------------------------- idle
    def _pick_idle(self, idle, robot_dir):
        if idle is None or idle == "off":
            return None, "off (no idle: an owner-side lamp change nobody has approved)"
        if isinstance(idle, (tuple, list)) and len(idle) == 2:
            times, positions = np.asarray(idle[0], dtype=float), np.asarray(idle[1], dtype=float)
            return _resample(times - times[0], positions, self.fps), "custom"
        if idle in ("auto", "vendor"):
            clip = load_vendor_idle(robot_dir, self.fps)
            if clip is not None:
                return clip, "vendor idle.csv (read at run time from the robot package)"
            if idle == "vendor":
                raise FileNotFoundError("vendor idle.csv not readable: set FTM_ROBOT_DIR or robot_dir")
        if idle in ("auto", "synthetic"):
            return synthetic_idle(self.fps), "synthetic wander (ASSUMPTION: shaped by the vendor clip's ranges)"
        raise ValueError(f"unknown idle option: {idle!r}")

    def _on_idle_start(self, _owner: str) -> None:
        # The queue plays idle only when nothing else is playing and idle is not paused (action_queue.py:241-247).
        if self._idle_clip is None or self._idle_paused or self._exec is not None:
            return
        times, positions = self._idle_plan(self._measure(self._t))
        self._exec = _Playback(IDLE, self._t + times, positions)
        self.log.append((self._t, "idle_start", IDLE, ""))

    def _idle_plan(self, current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One pass of the idle clip, blended in from the measured pose, relative times, live frame first.

        The vendor 'fuse' (vendor motion/fusion.py:336-470; playback_pipeline.py:114-152) crossfades, over
        1.1 s, from a quintic transition toward the clip state at the end of a 1.5 s window that starts 1.5 s
        after the clip's first moving frame, into the clip itself over that window, then plays the rest.
        Simplifications (ASSUMPTION): the transition is rest-to-rest (the vendor also matches the current and
        the clip's velocities and accelerations), the window start ignores the vendor's velocity scan, and
        the vendor's smoothing and velocity limiting of recorded motion are left out."""
        clip_t, clip_p = self._idle_clip
        if np.max(np.abs(current - clip_p[0])) <= self.idle_skip_threshold:
            rel_t, rel_p = clip_t, clip_p
        else:
            moving = np.flatnonzero(np.any(np.abs(clip_p - clip_p[0]) > IDLE_MOVING_THRESHOLD, axis=1))
            first = clip_t[moving[0]] if len(moving) else 0.0
            w_start = min(first + self.idle_lookahead_s, clip_t[-1])
            w_end = min(w_start + self.idle_window_s, clip_t[-1])

            def at(ts):
                return np.column_stack([np.interp(ts, clip_t, clip_p[:, j]) for j in range(N)])

            steps = max(2, int(self.idle_blend_s * self.fps))      # vendor motion/fusion.py:410-412
            u = np.arange(steps) / (steps - 1)
            transition = current + np.outer(ease(u), at(np.array([w_end]))[0] - current)
            weight = ease(u)[:, None]
            fade = (1.0 - weight) * transition + weight * at(w_start + u * (w_end - w_start))
            rest = clip_t > w_end
            rel_t = np.concatenate([u * self.idle_blend_s, self.idle_blend_s + clip_t[rest] - w_end])
            rel_p = np.vstack([fade, clip_p[rest]])
        # The queued plan is also given a live first frame and shifted one frame (playback_pipeline.py:95-111).
        frame = 1.0 / self.fps
        return np.concatenate([[0.0], frame + rel_t]), np.vstack([current, rel_p])

    # ---------------------------------------------------------------- servo and units
    def _deflection(self, goal: np.ndarray) -> np.ndarray:
        """Where gravity holds each joint off `goal`, signed units (cached: the goal changes at 30 Hz)."""
        if self._sag_fn is None:
            return self._sag_const
        key = goal.tobytes()
        if self._sag_cache is None or self._sag_cache[0] != key:
            self._sag_cache = (key, np.asarray(self._sag_fn(goal.copy()), dtype=float))
        return self._sag_cache[1]

    def _setpoint(self, goal: np.ndarray) -> np.ndarray:
        # A loaded joint holds off its goal by the sag, never past the end of its calibrated range; the vendor
        # simulator clamps readings to the range the same way (robots/backends/feetech/simulation.py:227).
        lo, hi = self.calibrated_range
        return np.clip(goal + self._deflection(goal), lo, hi)

    def _servo_at(self, t: float) -> np.ndarray:
        return _servo(self._x, self._setpoint(self._goal), t - self._t, self.servo_tau_s, self.servo_max_velocity)

    def _measure(self, t: float) -> np.ndarray:
        """What /joints would report at t (no commit): the servo position, quantised."""
        x = self._servo_at(t) if t > self._t else self._x
        return np.round(x / self.quantum) * self.quantum if self.quantum > 0 else x.copy()

    @staticmethod
    def _vec(units: dict) -> np.ndarray:
        return np.array([float(units[j]) for j in JOINTS])

    @staticmethod
    def _units(vec) -> dict:
        return {j: float(v) for j, v in zip(JOINTS, vec, strict=True)}
