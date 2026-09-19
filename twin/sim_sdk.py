"""A simulated LeLamp SDK gateway: the lamp's own /api/sdk/v1 HTTP API, served by a simulated lamp.

The real lamp is borrowed and often away. This server speaks the same API as the vendor's gateway (same
routes, auth headers, JSON shapes, status codes, action states, timing, rate limit and error strings), so
the team's lamp code (lamp/sdk.py LampSDK, the follower, the light performer) runs against it unmodified.
Moving to the real lamp later changes only the base URL and the token.

    uv run --with numpy --with requests python -m twin.sim_sdk --scenario walk_in_sit --port 18081 --token sim-token
    # then, in the team's code:   LampSDK(token="sim-token", base="http://127.0.0.1:18081")

Options: --transition-ms 600 (light fade; a labelled what-if when changed), --rate-limit 120 (actions per
minute per session; the vendor default is 30, the lamp's .env says 120), --record out_dir (trace.jsonl,
light_log.json, actions.json), --idle auto|vendor|synthetic|off, --camera-fps,
--size 640x480. Port 18081 by default, NOT 8081, which the Mac conductor's guest server uses. The sim
token is not a secret. FTM_ROBOT_DIR (the vendor robot package, read at run time, never copied here) is
REQUIRED: it gives the self-collision model the lamp always applies, the poses, animations and idle clip.
Without it the sim refuses to start, unless --no-collision asks for a degraded sim that accepts every move
(and says self_collision_check false). FTM_CALIBRATION gives the calibration hash the real lamp reports.

What is simulated, and from where (vendor file:line citations are relative to the vendor's private runtime
repository; the rules are re-derived in our own words, no vendor code or data is copied):
  * the gateway: sessions, token, idempotency, policy, per-session rate limit, capacity, action records and
    their states, cancel, retention (sdk_gateway/service.py, policy.py, domain.py, auth.py; routes/sdk.py);
  * motion.move and clip.play through an injected contract.MotionLike: twin/motion.py SDKMotionModel when
    (plan from the measured pose, pre-emption, settle, idle, sag, compute cost, the clip store);
  * light.glow through twin/sim_light.py (600 ms blocking cross-fade behind one lock, effects);
  * animation.play fire-and-forget, torque release (the arm goes limp), system.stop (a real e-stop:
    motion stop, torque release, light off), attention.nudge and scenario.run refused by policy;
  * the head camera (snapshot and the multipart stream) from an injected renderer or a synthetic frame, with
    each person's head painted with a drawn face that MediaPipe's face detector finds (paint_people);
  * the runtime's unauthenticated GET /api/status and GET /api/animations/status (whose play fields belong
    to the dashboard: an SDK animation.play never shows there, as on the lamp);
  * the dashboard's idle selector, POST /api/animations/idle: {"name": "none"} switches idle off,
    {"name": "idle"} puts it back, and {"name": ""} or null answer ok but change NOTHING (traced through the
    vendor runtime: SimLamp.request_idle).

Not simulated, on purpose or not yet: audio (the capabilities say unavailable, like a lamp without a USB
audio board: speech.say and packaged sound.play answer 503, music.play/sound.play with asset_id fail when
polled, /sounds and /music answer 503); the microphone stream (503); audit-log failure modes and the bus
events; the other dashboard routes (POST /api/animations/play and /stop, which alone fill the play fields
of /api/animations/status); the light's ambient policy after 900 s of silence. twin/motion.py has no
torque: while the sim's arm is limp its motion model keeps running underneath, so after torque.enable the
pose comes back to the model's, where the real arm would stay where it fell. Animations are played as clips
through the motion model, so they get its >= 2 s entry move instead of the vendor's 1.1 s blend, and its
collision check. The painted faces are drawings, one skin tone, lit one way: a detector's confidence on
them says nothing about its confidence on real people.

Tests: tests/twin/test_sim_sdk.py (plain requests and the team's LampSDK, manual and real clocks):
    FTM_ROBOT_DIR=... FTM_CALIBRATION=... uv run --with mujoco --with numpy --with pytest --with requests \\
        python -m pytest tests/twin/test_sim_sdk.py -p no:cacheprovider
The face-detector test also needs MediaPipe (Python 3.12): add --python 3.12 --with 'mediapipe==0.10.14'.

Clock: "real" (a thread steps the lamp at 30 Hz in real time) or "manual" (tests call advance(dt); every
HTTP call that has to wait, such as a 600 ms light fade, waits in simulated time).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import importlib.util
import io
import json
import math
import os
import random
import re
import signal
import socket
import sys
import threading
import time
import uuid
import warnings
import zlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs

import numpy as np

from twin.contract import JOINTS, MotionState, Units
from twin.sim_light import EFFECTS, GlowCrash, GlowInvalid, SimLight

# ------------------------------------------------------------------ the gateway's constants (source per line)
PROTOCOL_VERSION = "lelamp.sdk.v1"          # sdk_gateway/domain.py:12
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18081                        # ours: 8081 is the real lamp's port (config/default.yaml:101) AND the
                                            # Mac conductor's guest server, so the sim stays off it
DEFAULT_TOKEN = "sim-token"                 # the simulator's token; not a secret
ROBOT_ID = "lelamp_v1_pi5_feetech_r1"       # robot package robot.yaml:2, reported by /joints (control/runtime.py:130)
UNITS = "normalized_m100_100"               # robot package actuation.yaml:3
MAX_VELOCITY_UNITS_S = 300.0                # control/safety_filter.py:16 (the class default the package enforces)
CALIBRATED = (-100.0, 100.0)                # every joint; the package refuses any other limit (calibration.py:159-163)
TARGET_TOLERANCE = 2.0                      # robot package safety.yaml:11
TARGET_TOLERANCES = {"elbow_pitch": 10.0, "wrist_pitch": 3.0}   # safety.yaml:12-16 ("measured on the loaded arm")
SAFE_COMMAND_TYPES = ("attention.nudge", "light.glow", "motion.move", "clip.play", "torque.enable",
                      "torque.release", "music.play", "sound.play", "speech.say", "scenario.run",
                      "animation.play", "system.stop", "status.read")          # domain.py:46-64, this order
FORBIDDEN_COMMAND_TYPES = ("motion.move_joint", "motion.play_animation", "animation.record", "animation.upload",
                           "animation.choreography", "animation.sequence", "animation.timeline", "animation.raw",
                           "animation.write", "animation.delete", "motion.torque", "motor.write", "camera.raw",
                           "microphone.raw", "event_bus.publish", "system.shell", "system.filesystem",
                           "system.update", "system.reboot")                   # domain.py:66-86
FORBIDDEN_PREFIXES = ("motion.", "motor.", "servo.", "torque.", "camera.", "microphone.", "filesystem.",
                      "shell.", "event_bus.")                                   # domain.py:88-98
EXEMPT = ("motion.move", "clip.play", "torque.enable", "torque.release")      # domain.py:326-332
RUNTIME_DISABLED = ("attention.nudge", "scenario.run")                         # policy.py:19-24
CAPABILITY_FORBIDDEN = ["motors", "per_motor_torque", "animation_record", "animation_upload",
                        "animation_choreography", "raw_motor_values", "shell", "filesystem", "update", "reboot",
                        "direct_event_bus_publish", "scenario.run", "scenario_handoff"]   # service.py:138-163
ANIMATION_FORBIDDEN = ["animation.record", "animation.upload", "animation.choreography", "animation.sequence",
                       "animation.timeline", "raw_motor_values"]                          # service.py:165-199
ANIMATION_KEYS = {"animation_id", "gentle", "id", "name", "priority", "transition", "transition_profile", "ttl"}
ANIMATION_BANNED_KEYS = {"animation_values", "animations", "choreography", "frames", "keyframes", "motor_values",
                         "motors", "record", "recording", "sequence", "steps", "timeline",
                         "upload"}  # policy.py:201-230

SESSION_TTL_S = 3600.0                      # config/default.yaml:448; floor 60 s (service.py:250-253)
MAX_SESSIONS = 16                           # config/default.yaml:450; policy.py:57
RATE_LIMIT_PER_MIN = 120                    # the lamp's .env (gateway-core.md section 5); vendor default is 30
                                            # (config/default.yaml:451): pass rate_limit=30 to model a lamp whose
                                            # runtime has not been restarted since the .env was written
RATE_WINDOW_S = 60.0                        # sliding, per session: policy.py:166-187
MAX_PAYLOAD_BYTES = 16384                   # config/default.yaml:452; policy.py:113-127
ACTION_TTL_S = 30.0                         # config/default.yaml:449; policy.py:157-164
SYNC_RECORD_MIN_S = 300.0                   # a record lives at least this long: service.py:348
ASYNC_RECORD_S = 900.0                      # accepted device and motion records: service.py:381, :427
MOTION_CAPACITY = 4                         # safe motions prepared or running: service.py:392
DEVICE_CAPACITY = 4                         # device actions in flight: service.py:375
RESOURCE_CAPACITY = 4                       # /joints, /clips in flight: service.py:646-649
ROUTE_TIMEOUT_S = (1.0, 30.0, 120.0)        # min, default, max of the body's `timeout`: routes/sdk.py:75, 128-132
SAFE_MOTION_CAP_S = 660.0                   # a motion task that runs longer fails (timed out): service.py:709
TORQUE_RAMP_S = 0.6                         # release ramps torque to limp over 600 ms: research.py:169; feetech 202-221
TORQUE_ENABLE_S = 0.05                      # ASSUMPTION: the servo config rewrite on enable is tens of ms of bus writes
LIMP_START_S = 0.3                          # ASSUMPTION: the arm starts to give halfway through the 6-step ramp
LIMP_FALL_S = 0.8                           # ASSUMPTION: time to sag onto the folded rest pose once limp
CLIP_MAX_BYTES = 5 * 1024 * 1024            # routes/sdk_research.py:57-78; safe_motion.py:74-75
CLIP_MAX_FRAMES = 18000                     # control/safe_motion.py:12
CLIP_MAX_SECONDS = 600.0                    # control/safe_motion.py:13
CLIP_STORE_MAX = 100                        # clip_store.py:26-32
CAMERA_DEFAULT_FPS, CAMERA_MAX_FPS = 10.0, 30.0   # routes/sdk_research.py:173-175; service.py:596-600
MAX_STREAMS = 4                             # media.py:9 (camera and microphone share them)
STREAM_IDLE_S = 15.0                        # media.py:83, 120-128
CAMERA_CAPTURE_FPS = 20.0                   # ASSUMPTION: configured 30 (config/default.yaml:27) but each capture loop
                                            # also sleeps 1/30 s, so 15-25 in practice (media-light-catalog.md
                                            # section 3)
FRAME_SIZE = (640, 480)                     # requested capture size: perception/usb_camera.py:19-20
JPEG_QUALITY = 80                           # perception/usb_camera.py:20, 125-126
MONOTONIC_OFFSET_S = 1000.0                 # ASSUMPTION: camera stamps are the Pi's perf_counter; any offset does
RENDER_TIMEOUT_S = 10.0                     # ours: a renderer slower than this is dropped for synthetic frames
STEP_HZ = 30.0                              # the lamp's control rate: config/default.yaml:104-105
ANIMATION_PRIORITY = 70                     # sdk_gateway/mapper.py:438
SDK_MOTION_PRIORITY = 85                    # sdk_gateway/research.py:134
NOD_DELTA, NOD_HALF_S = 5.0, 0.25           # "nod": base_pitch +5 then back, 250 ms each (motion_manager.py:596-606)
# ids that pick a random family member: behavior/motion_policy.py:17-20
ANIMATION_FAMILIES = {"dance": ("dance", "dance_2"), "talking": ("talking", "talking_1", "talking_2")}
# The animation catalogue is the robot package's animations/factory_v1/*.csv, read at run time from
# FTM_ROBOT_DIR (SimLamp._catalog). Without it, these few names of our own stand in (standin_animation).
STANDIN_ANIMATIONS = ("curious", "dance", "happy", "idle", "look_around", "nod")
# Poses used when the robot package is not readable. ASSUMPTION: rounded to the region the vendor's
# runtime_initial / sleep poses sit in (safe-motion.md section 3.3, commands.md section 3.8), not the values.
FALLBACK_START = {"base_yaw": 0.0, "base_pitch": -50.0, "elbow_pitch": -5.0, "wrist_roll": 0.0, "wrist_pitch": 35.0}
FALLBACK_REST = {"base_yaw": 0.0, "base_pitch": -90.0, "elbow_pitch": -95.0, "wrist_roll": 0.0, "wrist_pitch": 15.0}
NOT_REACHED = "the arm stopped outside its settle tolerance: "   # control/runtime.py:43-67
IDLE_REQUEST_PRIORITY = 20                  # the idle route's intent: legacy_routes/animations.py:187
IDLE_SWITCH_S = 0.02                        # ASSUMPTION: the background idle request runs on the runtime's next loop
                                            # turns after the route answers (behavior/runtime.py:224-240)
NO_COLLISION_MODEL = (
    "FTM_ROBOT_DIR is required for lamp fidelity: without the robot package the sim has no self-collision "
    "model, so it would accept moves, clips and clip uploads that the lamp refuses with 'Motion rejected "
    "because the head could hit the base'. Set FTM_ROBOT_DIR to the robot package, or pass --no-collision "
    "(SimLamp(collision='off')) to run a degraded sim that accepts every move and says so.")
NO_COLLISION_WARNING = (
    "self-collision check OFF (degraded sim, opted in): every motion.move, clip.play and clip upload is "
    "accepted, where the lamp refuses the ones whose head could hit the base; /joints and /capabilities "
    "report self_collision_check false. Not for judging whether a choreography will run on the lamp.")


class MissingCollisionModel(RuntimeError):
    """The sim cannot check self-collision the way the lamp does, and the degraded mode was not asked for."""


# ------------------------------------------------------------------ small helpers
def _optional_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _hex(n: int = 16) -> str:
    return uuid.uuid4().hex[:n]


def tolerances() -> dict:
    return {j: TARGET_TOLERANCES.get(j, TARGET_TOLERANCE) for j in JOINTS}


def is_forbidden(command_type: str) -> bool:
    value = str(command_type or "").strip()
    if value in EXEMPT:
        return False
    return value in FORBIDDEN_COMMAND_TYPES or any(value.startswith(p) for p in FORBIDDEN_PREFIXES)


def robot_dir(path: str | Path | None = None) -> Path | None:
    raw = path if path is not None else os.environ.get("FTM_ROBOT_DIR", "")
    p = Path(raw) if raw else None
    return p if p is not None and p.is_dir() else None


def read_poses(directory: Path | None) -> dict:
    """Named poses from the robot package's poses.yaml (read at run time; a tiny indentation parser)."""
    if directory is None or not (directory / "poses.yaml").exists():
        return {}
    poses, name = {}, None
    for line in (directory / "poses.yaml").read_text().splitlines():
        m = re.match(r"^  (\w+):\s*$", line)
        if m:
            name = m.group(1)
            poses[name] = {}
            continue
        m = re.match(r"^    (\w+):\s*(-?[\d.]+)\s*$", line)
        if m and name:
            poses[name][m.group(1)] = float(m.group(2))
    return {k: v for k, v in poses.items() if set(v) == set(JOINTS)}


def calibration_identity(path: str | Path | None = None) -> str:
    """sha256 of the device calibration file's bytes, as the runtime reports it (control/factory.py:86).
    Without the file: a hash of a fixed label, so the id is stable and clearly not a real lamp's."""
    raw = path if path is not None else os.environ.get("FTM_CALIBRATION", "")
    if raw and Path(raw).is_file():
        return hashlib.sha256(Path(raw).read_bytes()).hexdigest()
    return hashlib.sha256(b"feelthemusic twin: no device calibration file").hexdigest()


def safe_motion_info(device_calibration: str, self_collision_check: bool) -> dict:
    """What /joints reports besides positions (control/runtime.py:125-141). calibration_id is the sha256 of the
    sorted JSON of the identity, computed the vendor's way, so with the lamp's calibration file it matches."""
    identity = {"robot_id": ROBOT_ID,
                "joints": {j: {"range_min": CALIBRATED[0], "range_max": CALIBRATED[1]} for j in JOINTS},
                "max_velocity_units_s": MAX_VELOCITY_UNITS_S, "units": UNITS, "device_calibration": device_calibration}
    return {**identity, "calibration_id": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
            "target_tolerance": TARGET_TOLERANCE, "target_tolerances": dict(TARGET_TOLERANCES),
            "self_collision_check": bool(self_collision_check)}


# ------------------------------------------------------------------ the vendor self-collision rule
class HeadBaseCheck:
    """The only self-collision rule the SDK applies: the head shade's capsule must stay more than a clearance
    away from the base cylinder (robot package safety.yaml:17-31; kinematics/self_collision.py:31-60). The
    geometry comes from lamp/spatial.py's LampModel, which reads safety.yaml at run time. Callable as a
    pose_ok(units) -> bool, the form twin/motion.py expects. It knows nothing about the table.

    Kinematics: LampModel turns units into angles the way the vendor's configured model does (one scale for
    every joint, anchored at the vendor neutral, kinematics/configured_model.py:24-43), so this reproduces
    what the gateway itself accepts. twin/model.py LampTwin.sdk_head_base_clear() evaluates the same rule
    on the calibrated MuJoCo kinematics instead; the two disagree on about 5 % of random poses (measured
    here on 400 uniform samples, 2026-09-19), all far from neutral where the vendor model is known to drift
    (safe-motion.md section 3.3)."""

    def __init__(self, lamp_model):
        self.model = lamp_model

    def distance_m(self, units: Units) -> float:
        a, b = self.model.head(units)["shade"]
        pts = a[None, :] + (b - a)[None, :] * np.linspace(0.0, 1.0, 33)[:, None]
        base = self.model.base
        radial = np.maximum(0.0, np.hypot(pts[:, 0], pts[:, 1]) - base["radius"])
        axial = np.maximum(0.0, np.maximum(base["z_min"] - pts[:, 2], pts[:, 2] - base["z_max"]))
        return float(np.min(np.hypot(radial, axial)))

    def __call__(self, units: Units) -> bool:
        return self.distance_m(units) > self.model.shade["radius"] + self.model.base["clearance"]


def load_lamp_model(directory: Path | None):
    """lamp/spatial.py's LampModel (the team's forward kinematics), or None without the robot package."""
    if directory is None or not (directory / "robot.urdf").exists():
        return None
    try:
        from lamp.spatial import LampModel
        return LampModel(directory)
    except Exception:
        return None


# ------------------------------------------------------------------ recorded motion (animations and clips)
def _cells_to_numbers(cells: list[str], labels: list[str]) -> list[float]:
    """float() of each cell (float() accepts spaces around a value), refused unless finite."""
    out = []
    for text, label in zip(cells, labels, strict=True):
        try:
            value = float(text)
        except (TypeError, ValueError):
            raise ValueError(f"{label} is not a finite number") from None
        if not math.isfinite(value):
            raise ValueError(f"{label} is not a finite number")
        out.append(value)
    return out


def read_motion_csv(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Parse and validate a clip CSV with the rules the SDK applies at upload (control/safe_motion.py:48-106,
    restated): at most 5 MiB of UTF-8 (a BOM allowed) in strict CSV; a header of `timestamp` and exactly the
    five `<joint>.pos` columns, as written (a space after a comma is a different name); every cell a finite
    number and every joint inside its calibrated range; 2..18000 rows; stamps, counted from the first row,
    within 0..600 s and strictly increasing; no joint faster than 300 units/s between rows.

    Raises ValueError (the gateway answers 422 invalid_request; clients should rely on that status and code,
    not on the wording). Returns (times from the first stamp, positions[n, 5] in JOINTS order)."""
    if len(data) > CLIP_MAX_BYTES:
        raise ValueError("Clip exceeds 5 MiB")
    columns = ["timestamp", *(f"{j}.pos" for j in JOINTS)]
    try:
        table = list(csv.reader(io.StringIO(data.decode("utf-8-sig")), strict=True))
    except (UnicodeError, csv.Error):
        raise ValueError("the CSV cannot be parsed") from None
    if not table:
        raise ValueError("the CSV cannot be parsed")
    header, body = table[0], table[1:]
    if sorted(header) != sorted(columns):
        raise ValueError("the CSV header must be timestamp and the five <joint>.pos columns")
    if len(body) > CLIP_MAX_FRAMES or any(len(row) != len(header) for row in body):
        raise ValueError("a CSV row has the wrong number of cells, or there are too many rows")
    order = [header.index(c) for c in columns]                  # our column order, whatever the file's
    values = np.array([_cells_to_numbers([row[i] for i in order], ["timestamp", *JOINTS]) for row in body])
    if len(values) and np.any((values[:, 1:] < CALIBRATED[0]) | (values[:, 1:] > CALIBRATED[1])):
        j = JOINTS[int(np.argmax(np.any((values[:, 1:] < CALIBRATED[0]) | (values[:, 1:] > CALIBRATED[1]), axis=0)))]
        raise ValueError(f"{j} is out of its calibrated range")
    if not 2 <= len(values) <= CLIP_MAX_FRAMES:
        raise ValueError(f"a clip needs 2 to {CLIP_MAX_FRAMES} frames")
    times = values[:, 0] - values[0, 0]
    if np.any(times < 0) or np.any(times > CLIP_MAX_SECONDS):
        raise ValueError("the clip is longer than 600 s")
    dt = np.diff(times)
    if np.any(dt <= 0):
        raise ValueError("timestamps do not increase")
    speed = np.abs(np.diff(values[:, 1:], axis=0)) / dt[:, None]
    if np.any(speed > MAX_VELOCITY_UNITS_S + 1e-6):
        raise ValueError(f"{JOINTS[int(np.argmax(speed.max(axis=0)))]} would move faster than the velocity limit")
    return times, values[:, 1:]


def process_recorded(times: np.ndarray, positions: np.ndarray, fps: float = STEP_HZ) -> tuple[np.ndarray, np.ndarray]:
    """What the runtime does to a recorded animation before playing it (motion/trajectory_processing.py:191-289,
    in our words): resample to 30 fps, 5-tap moving average, and stretch time wherever a joint would exceed
    0.99 x 300 units/s. Outlier removal is left out (ASSUMPTION: the pack's CSVs are clean)."""
    times = np.asarray(times, float) - float(times[0])
    n = max(2, int(math.floor(times[-1] * fps)) + 1)
    grid = np.arange(n) / fps
    pos = np.column_stack([np.interp(grid, times, positions[:, k]) for k in range(positions.shape[1])])
    if n >= 5:
        kernel = np.ones(5) / 5.0
        padded = np.pad(pos, ((2, 2), (0, 0)), mode="edge")
        pos = np.column_stack([np.convolve(padded[:, k], kernel, mode="valid") for k in range(pos.shape[1])])
    dt = np.full(n - 1, 1.0 / fps)
    peak = np.max(np.abs(np.diff(pos, axis=0)), axis=1) / dt
    dt = dt * np.maximum(1.0, peak / (0.99 * MAX_VELOCITY_UNITS_S))
    return np.concatenate([[0.0], np.cumsum(dt)]), pos


def standin_animation(name: str, start: Units) -> tuple[np.ndarray, np.ndarray]:
    """ASSUMPTION, used only without the robot package: a gentle 5 s look-around around `start` so that
    animation.play has something to do. It is not the vendor's animation."""
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    t = np.arange(0.0, 5.0 + 1e-9, 1.0 / STEP_HZ)
    base = np.array([start[j] for j in JOINTS])
    swing = np.array([12.0, 4.0, 4.0, 3.0, 8.0]) * rng.uniform(0.6, 1.0, 5)       # units, gentle
    shape = np.sin(np.pi * t / t[-1])[:, None] * np.sin(2 * np.pi * t[:, None] / 2.5 + rng.uniform(0, 6, 5)[None, :])
    return t, np.clip(base[None, :] + swing[None, :] * shape, *CALIBRATED)


def default_motion(start: Units, *, pose_ok=None, idle="auto", directory: Path | None = None):
    """twin.motion.SDKMotionModel, with the gateway owning rate limit and capacity."""
    from twin.motion import SDKMotionModel
    return SDKMotionModel(start, pose_ok=pose_ok, idle=None if idle == "off" else idle, robot_dir=directory,
                          rate_limit_per_min=10 ** 9, capacity=10 ** 6)


def switch_idle(motion, t: float, clip, *, same: bool = False) -> bool:
    """Make a motion model loop `clip` ((times, positions[n, 5]); None = no idle) from time t, as the vendor's
    set_idle_animation does (control/runtime.py:672-705): an idle pass that is playing stops where it is, the
    arm holding its last goal (runtime.py:697-705, executor stop), unless the new idle is the same one; the new
    idle starts when nothing else plays and idle is not paused (the model's own rule). Returns False when the
    model cannot switch idle at all.

    A model with a public set_idle(t, clip) is asked directly. twin/motion.py SDKMotionModel has
    none yet, so this drives the few attributes it plays idle from. ASSUMPTION about that module's internals,
    kept to one place; tests/twin/test_sim_sdk.py fails if they change."""
    public = getattr(motion, "set_idle", None)
    if callable(public):
        public(t, clip)
        return True
    needed = ("_run_until", "_idle_clip", "_exec", "_stop_executor", "_schedule")
    if not all(hasattr(motion, name) for name in needed):
        return False
    motion._run_until(t)
    if not same:
        playing = getattr(motion._exec, "owner", None) == "idle"        # twin/motion.py IDLE
        if playing:
            motion._stop_executor()
        motion._idle_clip = None if clip is None else (np.asarray(clip[0], float), np.asarray(clip[1], float))
    if clip is not None:
        motion._schedule(t, "idle_start", "idle")                       # ignored while paused or busy
    return True


# ------------------------------------------------------------------ JPEG
def encode_jpeg(rgb: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    """Baseline JPEG through Pillow, which the simulator's camera route requires (the route answers 503
    when Pillow is not installed)."""
    from PIL import Image
    out = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)[:, :, :3])).save(out, "JPEG",
                                                                                         quality=int(quality))
    return out.getvalue()


def _pose_parts(pose):
    get = pose.__getitem__ if isinstance(pose, dict) else (lambda k: getattr(pose, k))
    return tuple(np.asarray(get(k), float) for k in ("position", "forward", "down", "right"))


def crude_head(units: Units):
    """ASSUMPTION, used only when neither twin/model.py nor lamp/spatial.py can load the robot: the camera
    30 cm above the base looking forward, turned by base_yaw at about 0.74 deg/unit (safe-motion.md
    section 2) and tilted by wrist_pitch around 35 at about 0.5 deg/unit. Good enough to move a face in
    the picture when the lamp turns, nothing more."""
    yaw = math.radians(-0.74 * float(units.get("base_yaw", 0.0)))
    tilt = math.radians(0.5 * (float(units.get("wrist_pitch", 35.0)) - 35.0))
    forward = np.array([math.sin(yaw) * math.cos(tilt), math.cos(yaw) * math.cos(tilt), -math.sin(tilt)])
    right = np.array([math.cos(yaw), -math.sin(yaw), 0.0])
    down = np.cross(forward, right)
    return {"position": np.array([0.0, 0.05, 0.30]), "forward": forward, "down": down, "right": right}


HFOV_DEG, VFOV_DEG = 61.0, 44.0     # the head camera's field of view, derived from the on-lamp measurement
                                    # (twin/perception.py HFOV/VFOV; twin/model.py renders with the same)
SKIN = (224.0, 176.0, 146.0)        # ASSUMPTION: one light skin tone, sRGB; the look only has to read as a face
HAIR = (52.0, 38.0, 30.0)           # ASSUMPTION: dark brown hair
HEAD_TALL = 1.15                    # ASSUMPTION: the painted head is 15 % taller than wide (a rounder head than
                                    # a real one, 23 x 16 cm; taller than a ball so it reads as a head)


def _smoothstep(e0: float, e1: float, x):
    u = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _blob(lon, lat, c_lon: float, c_lat, half_lon: float, half_lat: float, soft: float = 0.35):
    """1 inside an ellipse drawn on the head (degrees of longitude and latitude), fading out at its edge."""
    d = np.sqrt(((lon - c_lon) / half_lon) ** 2 + ((lat - c_lat) / half_lat) ** 2)
    return 1.0 - _smoothstep(1.0 - soft, 1.0 + 0.3 * soft, d)


def face_albedo(lon, lat) -> np.ndarray:
    """The colour of a head at longitude/latitude (degrees) in the head's own frame: 0, 0 is the middle of the
    face, lon > 0 toward the person's right, lat > 0 up. A drawn face (hair, brows, eyes, nose, lips) that a
    face detector recognises: MediaPipe BlazeFace (lamp/follow.py FaceTracker, model_selection=1) finds it at
    0.6-0.9 confidence up to about 1.5-2 m and 55-65 deg of head turn (measured here, 2026-09-19). Feature
    places are ASSUMPTIONS from average adult proportions on a 9 cm head: eyes 6.3 cm apart (lon +-19 deg),
    brows above them, mouth about 4 cm below the eyes. Returns (..., 3) sRGB 0..255."""
    albedo = np.broadcast_to(np.asarray(SKIN), lon.shape + (3,)).copy()
    side = np.abs(lon)

    def paint(mask, colour):
        m = np.clip(mask, 0.0, 1.0)[..., None]
        albedo[:] = albedo * (1.0 - m) + np.asarray(colour, float) * m

    def shadow(mask, k):
        albedo[:] = albedo * (1.0 - k * np.clip(mask, 0.0, 1.0)[..., None])

    paint(0.35 * _blob(side, lat, 26, -14, 12, 9, 0.6), (225, 140, 130))        # cheeks
    shadow(_blob(side, lat, 19, 5, 13, 8, 0.6), 0.32)                            # eye sockets
    shadow(_blob(lon, lat, 0, -19, 7, 2.5, 0.6), 0.25)                           # under the nose
    shadow(_blob(lon, lat, 0, -40, 22, 5, 0.8), 0.15)                            # under the chin
    paint(0.30 * _blob(lon, lat, 0, -4, 3.0, 12, 0.6), (245, 205, 180))         # the bridge of the nose, lit
    shadow(_blob(lon, lat, -6, -8, 3.0, 9, 0.6), 0.20)                           # and its shaded side
    paint(_blob(side, lat, 4.5, -15.5, 2.8, 1.6), (95, 55, 45))                  # nostrils
    for sign in (-1.0, 1.0):
        cx = 19.0 * sign
        white = _blob(lon, lat, cx, 6, 8.0, 3.6, 0.25)
        paint(white, (238, 232, 226))
        paint(white * _blob(lon, lat, cx, 6, 3.8, 3.8, 0.25), (60, 38, 26))      # iris
        paint(white * _blob(lon, lat, cx, 6, 1.5, 1.5, 0.3), (10, 8, 8))         # pupil
        paint(_blob(lon, lat, cx, 8.6, 8.2, 1.3, 0.4), (60, 40, 35))             # upper lid
        paint(_blob(lon, lat, cx + 1.5 * sign, 16.0 - 0.02 * (lon - cx) ** 2, 10.5, 2.4, 0.4), (55, 38, 30))
    paint(0.85 * _blob(lon, lat, 0, -26, 12, 3.8, 0.35), (178, 88, 88))         # lips
    paint(_blob(lon, lat, 0, -26, 11.5, 0.9, 0.5), (80, 30, 35))                 # between them
    # hair above a hairline that drops at the temples, and all over the back of the head
    hairline = 34.0 - 44.0 * np.clip((side - 38.0) / 57.0, 0.0, 1.0) - 25.0 * np.clip((side - 95.0) / 40.0, 0.0, 1.0)
    paint(_smoothstep(-2.0, 2.0, lat - hairline), HAIR)
    return albedo


def paint_people(img: np.ndarray, pose, people, *, fx: float, fy: float, light_rgb=None) -> np.ndarray:
    """Paint each person's head, a face on the side it faces, into a picture from the head camera `pose`
    (position, forward, down, right in the base frame; fx, fy in picture widths/heights per unit of tangent).
    Ray-traced per pixel, so the face turns, tilts and foreshortens with Person.facing and goes to hair at
    profile, and a head the camera sees from above or below looks it. Farthest first, so nearer heads cover
    farther ones. ASSUMPTION: nothing else covers a head (in the head camera's view the arm is behind the lens).
    Lit from behind the camera and above (ASSUMPTION: room light), tinted a little by the lamp's own light.
    img: H x W x 3, float or uint8, 0..255; a float copy is returned."""
    img = np.asarray(img, dtype=float).copy()
    if pose is None or not people:
        return img
    h, w = img.shape[:2]
    position, forward, down, right = _pose_parts(pose)
    light = -forward + np.array([0.0, 0.0, 0.8])
    light /= np.linalg.norm(light)
    tint = np.ones(3)
    if light_rgb is not None:
        tint = 0.9 + 0.2 * np.clip(np.asarray(light_rgb, float).reshape(-1, 3).mean(axis=0), 0, 1)
    squash = np.array([1.0, 1.0, 1.0 / HEAD_TALL])          # an ellipsoid is a sphere in squashed space
    seen = []
    for person in people:
        centre = np.asarray(person.head, float)
        depth = float((centre - position) @ forward)
        if depth > 0.05:
            seen.append((depth, person, centre))
    for depth, person, centre in sorted(seen, key=lambda s: -s[0]):
        r = float(getattr(person, "head_radius", 0.09))
        rel = centre - position
        cx = (0.5 + fx * float(rel @ right) / depth) * w
        cy = (0.5 + fy * float(rel @ down) / depth) * h
        near = max(depth - HEAD_TALL * r, 0.02)
        half_x = fx * HEAD_TALL * r / near * w * 1.2 + 2
        half_y = fy * HEAD_TALL * r / near * h * 1.2 + 2
        x0, x1 = max(0, int(cx - half_x)), min(w, int(cx + half_x) + 1)
        y0, y1 = max(0, int(cy - half_y)), min(h, int(cy + half_y) + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1]
        ray = (forward + (((xs + 0.5) / w - 0.5) / fx)[..., None] * right
               + (((ys + 0.5) / h - 0.5) / fy)[..., None] * down) * squash
        ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
        origin = (position - centre) * squash                 # the camera, relative to the head, squashed
        along = ray @ origin
        miss = np.sqrt(np.maximum(0.0, float(origin @ origin) - along ** 2))     # ray to head-centre distance
        cover = np.clip((r - miss) * fx * w / depth + 0.5, 0.0, 1.0)             # anti-aliased outline, pixels
        if not np.any(cover > 0):
            continue
        s = -along - np.sqrt(np.maximum(0.0, r * r - miss ** 2))
        normal = (origin + ray * s[..., None]) / r
        facing = np.asarray(person.facing, float)
        facing = facing / max(1e-9, float(np.linalg.norm(facing)))
        up = np.array([0.0, 0.0, 1.0]) - facing[2] * facing
        up = up / np.linalg.norm(up) if np.linalg.norm(up) > 1e-6 else np.array([0.0, 1.0, 0.0])
        their_right = np.cross(facing, up)
        lon = np.degrees(np.arctan2(normal @ their_right, normal @ facing))
        lat = np.degrees(np.arcsin(np.clip(normal @ up, -1.0, 1.0)))
        shade = 0.5 + 0.55 * np.clip(normal @ light, 0.0, 1.0)
        colour = face_albedo(lon, lat) * shade[..., None] * tint
        region = img[y0:y1, x0:x1]
        region[:] = region * (1.0 - cover[..., None]) + colour * cover[..., None]
    return img


def synthetic_frame(width: int, height: int, pose, people, *, hfov_deg: float = HFOV_DEG,
                    vfov_deg: float = VFOV_DEG, light_rgb=None) -> np.ndarray:
    """A grey room with each person's head painted where the head camera would see it, with a face a face
    detector finds (paint_people). Pinhole camera with the 61 x 44 deg field of view."""
    yy = np.mgrid[0:height, 0:width][0]
    grey = 118 + 20 * (yy / max(1, height - 1))                     # a faint gradient so JPEG has texture
    img = np.repeat(grey[:, :, None], 3, axis=2)
    if light_rgb is not None:
        tint = np.clip(np.asarray(light_rgb, float).reshape(-1, 3).mean(axis=0), 0, 1)
        img = img * (0.9 + 0.2 * tint[None, None, :])               # the lamp's own light tints the room a little
    fx = 0.5 / math.tan(math.radians(hfov_deg) / 2)
    fy = 0.5 / math.tan(math.radians(vfov_deg) / 2)
    img = paint_people(img, pose, people, fx=fx, fy=fy, light_rgb=light_rgb)
    return np.clip(img, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ the clock
class SimClock:
    """Simulation time in seconds from 0. mode "real" follows time.monotonic(); mode "manual" moves only when
    set() is called. wall(t) maps simulation time to epoch seconds for the SDK's time.time() fields."""

    def __init__(self, mode: str = "real"):
        if mode not in ("real", "manual"):
            raise ValueError("clock mode is 'real' or 'manual'")
        self.mode = mode
        self._t = 0.0
        self._cond = threading.Condition()
        self._t0 = time.monotonic()
        self.epoch0 = time.time()

    def now(self) -> float:
        return time.monotonic() - self._t0 if self.mode == "real" else self._t

    def wall(self, t: float | None = None) -> float:
        return self.epoch0 + (self.now() if t is None else float(t))

    def set(self, t: float) -> None:
        with self._cond:
            self._t = max(self._t, float(t))
            self._cond.notify_all()

    def wait_until(self, t: float, real_timeout: float = 1.0) -> bool:
        """Block until simulation time reaches t, at most real_timeout seconds of real time."""
        if self.mode == "real":
            time.sleep(max(0.0, min(real_timeout, t - self.now())))
            return self.now() >= t - 1e-9
        with self._cond:
            return self._cond.wait_for(lambda: self._t >= t - 1e-9, timeout=real_timeout)


# ------------------------------------------------------------------ the simulated lamp (the device)
class SimLamp:
    """The simulated device behind the gateway: a MotionLike for the arm, a SimLight for the panel, a clock,
    a scenario of people, the head camera, torque, the vendor animations, and a trace for evidence.

    motion: any contract.MotionLike. Default: twin/motion.py SDKMotionModel from the
        robot package's runtime_initial pose, with the vendor head-vs-base check.
    light: a SimLight. scenario: anything with people_at(t) -> [contract.Person] (twin/world.py).
    kinematics: anything with head(units) -> Pose or dict (twin/model.py LampTwin, lamp/spatial.py LampModel);
        default: LampModel from FTM_ROBOT_DIR, else crude_head().
    renderer: optional callable(units, people, light_rgb, width, height) -> HxWx3 uint8 picture; without it
        the camera sends synthetic_frame() pictures.
    faces: also paint the people's heads, with faces a face detector finds (paint_people), over the
        renderer's picture (its people are plain spheres), using kinematics.head() and its fx/fy. The
        synthetic frames always have them.
    clock: "real", "manual" or a SimClock.
    collision: "required" (default): the vendor's self-collision rule must be available, from pose_ok, else
        lamp/spatial.py LampModel (HeadBaseCheck), else kinematics.sdk_head_base_clear (twin/model.py
        LampTwin); without one the constructor raises MissingCollisionModel, because the lamp always checks
        (it refuses every move when it cannot: control/runtime.py:176-182) and a sim that does not would
        accept moves and clips the lamp refuses. "off": the degraded mode, opt in only: no check, every
        move and clip accepted, /joints and /capabilities say self_collision_check false.
    """

    def __init__(self, *, motion=None, light: SimLight | None = None, clock: SimClock | str = "real",
                 scenario=None, kinematics=None, renderer=None, pose_ok: Callable | None = None,
                 robot: str | Path | None = None, calibration: str | Path | None = None,
                 transition_ms: float = 600.0, camera_fps: float = CAMERA_CAPTURE_FPS,
                 frame_size: tuple[int, int] = FRAME_SIZE, idle: str = "auto",
                 step_hz: float = STEP_HZ, seed: int = 0, record_dir: str | Path | None = None,
                 trace_keep: int = 20000, collision: str = "required", faces: bool = True):
        self.clock = clock if isinstance(clock, SimClock) else SimClock(clock)
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.robot_dir = robot_dir(robot)
        self.poses = read_poses(self.robot_dir)
        self.start_pose = dict(self.poses.get("runtime_initial", FALLBACK_START))
        self.rest_pose = dict(self.poses.get("sleep", FALLBACK_REST))
        lamp_model = load_lamp_model(self.robot_dir)
        self.kinematics = kinematics if kinematics is not None else lamp_model
        self.pose_ok = self._collision_model(collision, pose_ok, lamp_model)
        self.faces = bool(faces)
        self.motion = motion if motion is not None else default_motion(
            self.start_pose, pose_ok=self.pose_ok, idle=idle, directory=self.robot_dir)
        self.light = light if light is not None else SimLight(transition_ms)
        self.scenario = scenario
        self.renderer = renderer
        self.camera_fps = float(camera_fps)
        self.frame_size = (int(frame_size[0]), int(frame_size[1]))
        self.step_hz = float(step_hz)
        self.device_calibration = calibration_identity(calibration)
        self.info = safe_motion_info(self.device_calibration, self.pose_ok is not None)
        self.catalog = self._catalog()
        self.rng = random.Random(seed)
        self.torque_enabled = True
        self._limp: dict | None = None            # {"t0", "from"} once the arm is limp
        self.animation: dict | None = None        # the vendor animation playing now (SDK animation.play)
        # Every animation outcome: SDK plays, refusals, the idle route. Only the sim's own record
        # (animations.json, the trace): the lamp logs these and reports none of them over HTTP.
        self.animation_log: list[dict] = []
        # What the dashboard's POST /api/animations/play and /stop would have set, the only writers of these
        # fields of GET /api/animations/status (legacy_routes/animations.py:86-88, 99, 144-145). The sim does
        # not serve those two routes, so they stay empty: an SDK animation.play never shows up there.
        self.dashboard_animation: dict | None = None
        self.dashboard_error: str | None = None
        idle_source = str(getattr(self.motion, "idle_source", "") or "")
        # The looped idle's name, as /api/animations/status current_idle reports it; the lamp starts with its
        # configured idle, "idle" (config/default.yaml:238; idle/lifecycle.py:86-105). None when the motion
        # model plays no idle.
        self.idle_name: str | None = "idle" if idle_source and not idle_source.startswith("off") else None
        self._serviced_t = self.clock.now()
        self.state: MotionState = self.motion.step(self._serviced_t)
        self.last_render_error: str | None = None
        self._jobs: list[tuple[float, int, Callable]] = []
        self._job_seq = 0
        self._frame_cache: tuple | None = None
        self._render_lock = threading.Lock()
        self._render_pool: ThreadPoolExecutor | None = None
        self.trace: deque = deque(maxlen=trace_keep)
        self.record_dir = Path(record_dir) if record_dir else None
        self._trace_file = None
        if self.record_dir is not None:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            self._trace_file = open(self.record_dir / "trace.jsonl", "w")
        self._tick_n = 0
        self.gateway = None                       # set by SimGateway
        self._thread: threading.Thread | None = None
        self._running = False

    def _collision_model(self, collision: str, pose_ok, lamp_model):
        if collision not in ("required", "off"):
            raise ValueError("collision is 'required' or 'off'")
        if collision == "off":
            warnings.warn(NO_COLLISION_WARNING, RuntimeWarning, stacklevel=3)
            return None
        if pose_ok is not None:
            return pose_ok
        if lamp_model is not None:
            return HeadBaseCheck(lamp_model)
        twin_rule = getattr(self.kinematics, "sdk_head_base_clear", None)
        twin_dir = getattr(self.kinematics, "robot_dir", None)
        # LampTwin lets every pose through when it cannot read safety.yaml, so only trust it when it can
        if callable(twin_rule) and twin_dir is not None and (Path(twin_dir) / "safety.yaml").is_file():
            return twin_rule
        raise MissingCollisionModel(NO_COLLISION_MODEL)

    # ---------------------------------------------------------------- time
    def now(self) -> float:
        return self.clock.now()

    def start(self) -> None:
        """Real-time mode: step the lamp at step_hz in a background thread."""
        if self.clock.mode != "real" or self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="simlamp-step", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._render_pool is not None:
            self._render_pool.shutdown(wait=False, cancel_futures=True)
            self._render_pool = None
        self.write_record()

    def _loop(self) -> None:
        period = 1.0 / self.step_hz
        next_t = self.clock.now()
        while self._running:
            with self.lock:
                self.tick(self.clock.now())
            next_t += period
            time.sleep(max(0.0, next_t - self.clock.now()))

    def advance(self, dt: float) -> None:
        """Manual mode: move simulated time forward by dt, stepping at step_hz."""
        if self.clock.mode != "manual":
            raise RuntimeError("advance() needs a manual clock")
        target = self.clock.now() + float(dt)
        while True:
            with self.lock:
                t = self.clock.now()
                if t >= target - 1e-12:
                    break
                nxt = min(target, t + 1.0 / self.step_hz)
                self.clock.set(nxt)
                self.tick(nxt)

    def run_until(self, predicate: Callable[[], bool], *, max_s: float = 30.0, dt: float = 0.05,
                  real_pause_s: float = 0.0) -> bool:
        """Manual mode helper: advance in dt steps until predicate() (checked under the lock) or max_s."""
        end = self.clock.now() + max_s
        while self.clock.now() < end:
            with self.lock:
                if predicate():
                    return True
            self.advance(dt)
            if real_pause_s:
                time.sleep(real_pause_s)
        with self.lock:
            return bool(predicate())

    def schedule(self, t: float, fn: Callable[[float], None]) -> None:
        self._job_seq += 1
        self._jobs.append((float(t), self._job_seq, fn))
        self._jobs.sort(key=lambda j: (j[0], j[1]))

    def next_due(self, t: float) -> float:
        """The next moment something is due (a job, a light event, a pending light reply), for waiters."""
        times = [t + 1.0 / self.step_hz]
        if self._jobs:
            times.append(self._jobs[0][0])
        light_next = self.light.next_event()
        if light_next is not None:
            times.append(light_next)
        times += [c.reply_at for c in self.light.log[-64:] if c.reply_at is not None and c.reply_at > t]
        return max(t, min(times))

    def service(self, t: float) -> None:
        """Bring the whole device to time t (idempotent for the same t). Call with the lock held. Jobs due
        before t run at their own time, after the arm and the light have been brought to that time."""
        while self._jobs and self._jobs[0][0] <= t + 1e-12:
            when, _, fn = self._jobs.pop(0)
            when = max(when, self._serviced_t)
            self._advance_to(when)
            fn(when)
        self._advance_to(t)
        self.changed.notify_all()

    def _advance_to(self, t: float) -> None:
        t = max(t, self._serviced_t)
        self._serviced_t = t
        self.light.update(t)
        self.state = self.motion.step(t)
        if self.gateway is not None:
            self.gateway.on_motion(t, self.state)
            self.gateway.on_time(t)
        for mid, outcome in list(self.state.finished):
            if self.animation is not None and mid == self.animation.get("motion_id"):
                self.animation_log.append({**self.animation, "outcome": outcome, "ended_at": t})
                self.animation = None

    def tick(self, t: float) -> None:
        self.service(t)
        self._record(t)

    # ---------------------------------------------------------------- what the arm reports
    def measured(self, t: float | None = None) -> dict:
        t = self.now() if t is None else t
        pose = dict(self.state.measured)
        if self._limp is not None:
            u = min(1.0, max(0.0, (t - self._limp["t0"]) / LIMP_FALL_S))
            fall = u * u                                  # gravity: slow to start, then it drops
            for j in ("base_pitch", "elbow_pitch", "wrist_pitch"):
                pose[j] = self._limp["from"][j] + (self.rest_pose[j] - self._limp["from"][j]) * fall
            for j in ("base_yaw", "wrist_roll"):
                pose[j] = self._limp["from"][j]
        return {j: round(float(pose[j]), 4) for j in JOINTS}

    def people(self, t: float) -> list:
        if self.scenario is None:
            return []
        try:
            return list(self.scenario.people_at(t))
        except Exception:
            return []

    # ---------------------------------------------------------------- torque (the SDK's whole-robot switch)
    def release_torque(self, t: float) -> None:
        """torque.release / system.stop: stop playback (moves end canceled), then a 600 ms torque ramp to limp
        from wherever the arm is. Nothing lowers it first (research.py:167-169; feetech driver 202-221)."""
        self.stop_playback(t)
        self.torque_enabled = False                       # the controller marks torque off at once (runtime.py:877-891)
        start = t + LIMP_START_S

        def go_limp(when: float) -> None:
            if not self.torque_enabled and self._limp is None:
                self._limp = {"t0": when, "from": dict(self.state.measured)}
        self.schedule(start, go_limp)

    def enable_torque(self, t: float) -> None:
        """torque.enable: servos hold where the arm now is. ASSUMPTION: the vendor never writes a goal before
        re-enabling, so what a real servo does first is unknown (commands.md section 3.7); here it holds."""
        where = self.measured(t)
        self.torque_enabled = True
        self._limp = None
        reset = getattr(self.motion, "reset_pose", None)
        if callable(reset):
            reset(t, where)

    def stop_playback(self, t: float) -> None:
        """stop_all / stop_playback: every SDK motion in flight ends canceled; a vendor animation stops."""
        if self.gateway is not None:
            self.gateway.cancel_motion_records(t, "cancelled_or_preempted")
        if self.animation is not None:
            self.motion.cancel(t, self.animation["motion_id"])
            self.animation_log.append({**self.animation, "outcome": "stopped", "ended_at": t})
            self.animation = None

    # ---------------------------------------------------------------- vendor animations
    def _catalog(self) -> dict:
        """animation id -> CSV path (None for the stand-in catalogue)."""
        if self.robot_dir is not None:
            pack = self.robot_dir / "animations" / "factory_v1"
            if pack.is_dir():
                found = {p.stem: p for p in sorted(pack.glob("*.csv"))}
                if found:
                    return found
        return {name: None for name in STANDIN_ANIMATIONS}

    def animation_frames(self, name: str, t: float) -> list[tuple[float, Units]]:
        measured = self.measured(t)
        if name == "nod":
            # not nod.csv: a hard-coded relative base_pitch flick from the measured pose (motion_manager.py:596-606)
            up = dict(measured, base_pitch=measured["base_pitch"] + NOD_DELTA)
            return [(0.0, measured), (NOD_HALF_S, up), (2 * NOD_HALF_S, dict(measured))]
        path = self.catalog.get(name)
        if path is not None:
            times, positions = read_motion_csv_unchecked(Path(path).read_bytes())
        else:
            times, positions = standin_animation(name, measured)
        times, positions = process_recorded(times, positions)
        return [(float(ts), {j: float(v) for j, v in zip(JOINTS, row, strict=True)})
                for ts, row in zip(times, positions, strict=True)]

    def play_animation(self, name: str, t: float, *, priority: int = ANIMATION_PRIORITY) -> tuple[bool, str]:
        """Fire and forget, like the vendor behaviour runtime (behavior/runtime.py:225-305). Played through the
        MotionLike as a clip. Differences from the lamp, stated: the model prepends its >= 2 s clip entry
        instead of the vendor's 1.1 s fuse, and checks self-collision, which the vendor skips for animations."""
        busy = self.gateway.motion_busy() if self.gateway is not None else False
        if busy and priority <= SDK_MOTION_PRIORITY:
            reason = "blocked_by_active_motion"           # 70 < 85: a safe move holds the joints
                                                          # (motion_manager.py:111-137)
        elif self.animation is not None and priority <= ANIMATION_PRIORITY:
            reason = "blocked_by_active_motion"           # equal priority cannot pre-empt (motion_manager.py:347-364)
        elif not self.torque_enabled:
            reason = "blocked_by_safety_state"
        else:
            if busy and self.gateway is not None:
                self.gateway.cancel_motion_records(t, "cancelled_or_preempted")   # priority > 85 pre-empts a move
            family = ANIMATION_FAMILIES.get(name)
            chosen = self.rng.choice([n for n in family if n in self.catalog]) if family else name
            try:
                frames = self.animation_frames(chosen, t)
                if name == "nod" and frames[1][1]["base_pitch"] > CALIBRATED[1]:
                    reason = "Motion target outside calibrated safety envelope for base_pitch"
                else:
                    result = self.motion.submit_clip(t, frames)
                    if result.accepted:
                        self.animation = {"name": chosen, "requested": name, "motion_id": result.action_id,
                                          "started_at": self.clock.wall(t), "t": t}
                        self.animation_log.append({"name": chosen, "requested": name, "outcome": "started", "t": t})
                        return True, "accepted_async"
                    reason = result.reason
            except Exception as exc:              # log-only, like every later animation failure on the lamp
                reason = f"{type(exc).__name__}: {exc}"
        # Only the lamp's log hears of this; no route reports it (not /api/animations/status either).
        self.animation_log.append({"name": name, "outcome": "refused", "reason": reason, "t": t})
        return False, reason

    # ---------------------------------------------------------------- the dashboard's idle selector
    def idle_clip(self, name: str):
        """(times, positions[n, 5]) of the animation `name` looped as idle: the model's own idle clip for
        "idle" (twin/motion.py: the vendor idle.csv, else its synthetic wander), else the pack's CSV."""
        if name == "idle":
            try:
                from twin.motion import FPS, load_vendor_idle, synthetic_idle
                clip = load_vendor_idle(self.robot_dir, FPS) if self.robot_dir else None
                return clip if clip is not None else synthetic_idle(FPS)
            except ImportError:
                pass
        path = self.catalog.get(name)
        if path is not None:
            times, positions = read_motion_csv_unchecked(Path(path).read_bytes())
        else:
            times, positions = standin_animation(name, self.start_pose)
        return process_recorded(times, positions)

    def request_idle(self, idle_name: str | None, t: float) -> None:
        """What POST /api/animations/idle sets in motion once the route has answered. The runtime turns it into
        a priority-20 idle request that runs in the background (behavior/runtime.py:224-240, 297-298;
        motion_policy.py:121-132), named after the animation, or "clear_idle_animation" when the body had no
        name (legacy_routes/animations.py:180-189). Traced through the vendor code, that means:
          * "none": the motion manager treats that name as "no idle" (motion_manager.py:443-444, 584-593):
            the idle pass stops where it is and the arm holds that pose. This is how to switch idle OFF.
          * the name of an animation in the pack: that animation becomes the looped idle (runtime.py:672-694).
          * no name ("", null, missing): "clear_idle_animation" is not an animation, so the request is refused
            "motion_not_available" (motion_manager.py:94-101, 440-447) and idle does NOT change, although
            the route said {"status": "ok", "idle": null}. The dashboard's own "None" choice sends null.
          * any other name: refused "motion_not_available" the same way; nothing changes.
          * while an animation or an SDK motion holds the joints, the request cannot pre-empt it and is
            refused "blocked_by_active_motion" (motion_manager.py:110-138, 342-364); nothing changes.
        Refusals are only logged on the lamp, never answered; here they go to the animation log."""
        request = idle_name or "clear_idle_animation"

        def apply(when: float) -> None:
            entry = {"name": request, "route": "POST /api/animations/idle", "t": when}
            busy = self.animation is not None or (self.gateway is not None and self.gateway.motion_busy())
            if request != "none" and request not in self.catalog:
                self.animation_log.append({**entry, "outcome": "idle unchanged", "reason": "motion_not_available"})
                return
            if busy:
                self.animation_log.append({**entry, "outcome": "idle unchanged", "reason": "blocked_by_active_motion"})
                return
            new = None if request == "none" else request
            switched = switch_idle(self.motion, when, None if new is None else self.idle_clip(new),
                                   same=new == self.idle_name)
            self.idle_name = new
            self.animation_log.append({**entry, "outcome": "idle off" if new is None else "idle set", "idle": new,
                                       "motion_model_switched": switched})
        self.schedule(t + IDLE_SWITCH_S, apply)

    # ---------------------------------------------------------------- camera
    def camera_frame(self) -> tuple[bytes, dict]:
        """The newest captured frame as (jpeg, metadata), rendered on demand at the capture rate."""
        with self.lock:
            t = self.now()
            k = int(math.floor(t * self.camera_fps + 1e-9))
            cached = self._frame_cache
            if cached is not None and cached[0] == k:
                return cached[1], dict(cached[2])
            t_capture = k / self.camera_fps
            units = self.measured(t)
            people = self.people(t_capture)
            light = self.light.linear(t)
        with self._render_lock:
            cached = self._frame_cache
            if cached is not None and cached[0] == k:
                return cached[1], dict(cached[2])
            width, height = self.frame_size
            picture = None
            if self.renderer is not None:
                try:
                    # One persistent thread renders everything: an OpenGL context (MuJoCo's Renderer) must be
                    # created and used on the same thread, and every HTTP request arrives on a new one.
                    if self._render_pool is None:
                        self._render_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="simlamp-render")
                    job = self._render_pool.submit(self.renderer, units, people, light, width, height)
                    picture = np.asarray(job.result(timeout=RENDER_TIMEOUT_S))
                    if picture.dtype != np.uint8:
                        picture = np.clip(picture * (255.0 if picture.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
                    if self.faces and people and self.kinematics is not None:
                        # the renderer's people are plain spheres no face detector fires on: paint the faces
                        # over them from the same camera (kinematics.head() and its focal lengths)
                        fx = float(getattr(self.kinematics, "fx", 0.5 / math.tan(math.radians(HFOV_DEG) / 2)))
                        fy = float(getattr(self.kinematics, "fy", 0.5 / math.tan(math.radians(VFOV_DEG) / 2)))
                        picture = np.clip(paint_people(picture, self.kinematics.head(units), people, fx=fx, fy=fy,
                                                       light_rgb=light), 0, 255).astype(np.uint8)
                except Exception as exc:          # a broken or stuck renderer must not take the camera down
                    self.renderer = None
                    picture = None
                    self.last_render_error = f"{type(exc).__name__}: {exc}"
            if picture is None:
                head = None
                try:
                    head = self.kinematics.head(units) if self.kinematics is not None else crude_head(units)
                except Exception:
                    head = crude_head(units)
                picture = synthetic_frame(width, height, head, people, light_rgb=light)
            data = encode_jpeg(picture)
            meta = {"sequence": k + 1, "timestamp": round(MONOTONIC_OFFSET_S + t_capture, 6), "clock": "monotonic",
                    "width": width, "height": height, "encoding": "jpeg"}      # media.py:35-42
            self._frame_cache = (k, data, meta)
            return data, dict(meta)

    # ---------------------------------------------------------------- evidence
    def _record(self, t: float) -> None:
        self._tick_n += 1
        people = self.people(t)
        row = {"t": round(t, 4), "measured": self.measured(t),
               "commanded": {j: round(float(v), 4) for j, v in self.state.commanded.items()},
               "active_action": self.state.active_action, "idle_playing": bool(self.state.idle_playing),
               "torque_enabled": self.torque_enabled, "limp": self._limp is not None,
               "animation": self.animation["name"] if self.animation else None,
               "light_mean": [round(float(v), 5) for v in self.light.linear(t).mean(axis=0)],
               "people": [{"id": p.id, "head": [round(float(v), 4) for v in p.head],
                           "facing": [round(float(v), 4) for v in p.facing]} for p in people]}
        if self._tick_n % 3 == 1:                  # the whole panel at a third of the step rate
            row["panel"] = self.light.sample(t).tobytes().hex()
        self.trace.append(row)
        if self._trace_file is not None:
            self._trace_file.write(json.dumps(row) + "\n")

    def write_record(self) -> None:
        if self.record_dir is None:
            return
        with self.lock:
            if self._trace_file is not None:
                self._trace_file.close()
                self._trace_file = None
            (self.record_dir / "light_log.json").write_text(
                json.dumps([c.log_row() for c in self.light.log], indent=1, default=str))
            if self.gateway is not None:
                (self.record_dir / "actions.json").write_text(json.dumps(
                    [r.to_dict(self.clock.wall()) for r in self.gateway.actions.values()], indent=1, default=str))
            (self.record_dir / "animations.json").write_text(json.dumps(self.animation_log, indent=1, default=str))


def read_motion_csv_unchecked(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """A vendor animation CSV (timestamp + the five .pos columns), without the SDK clip limits: the runtime
    time-stretches animations instead of refusing them (trajectory_processing.py:191-241)."""
    rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig"))))
    header = [h.strip() for h in rows[0]]
    col = {h: i for i, h in enumerate(header)}
    body = [r for r in rows[1:] if len(r) == len(header)]
    times = np.array([float(r[col["timestamp"]]) for r in body])
    positions = np.array([[float(r[col[f"{j}.pos"]]) for j in JOINTS] for r in body])
    keep = np.concatenate([[True], np.diff(times) > 0])
    return times[keep], np.clip(positions[keep], *CALIBRATED)


# ------------------------------------------------------------------ the SDK gateway
TERMINAL = ("succeeded", "failed", "rejected", "canceled")


@dataclass
class Session:
    session_id: str
    app_id: str
    created_at: float
    expires_at: float
    metadata: dict = field(default_factory=dict)

    def to_dict(self, wall: float) -> dict:
        return {"session_id": self.session_id, "app_id": self.app_id, "created_at": self.created_at,
                "expires_at": self.expires_at, "expired": wall >= self.expires_at, "metadata": dict(self.metadata)}


@dataclass
class Record:
    """One SDK action record (domain.py:243-277)."""
    action_id: str
    command_type: str
    session_id: str
    state: str
    created_at: float
    updated_at: float
    expires_at: float
    result: dict = field(default_factory=dict)
    lifecycle: dict = field(default_factory=dict)
    error: dict | None = None
    path: str = "sync"                      # sync | motion | device
    motion_id: str | None = None
    status_code: int = 200                  # the HTTP status of a failure (never echoed in the record)
    t_created: float = 0.0
    target: dict | None = None              # motion.move: the admission-time full target

    def to_dict(self, wall: float) -> dict:
        out = {"action_id": self.action_id, "command_type": self.command_type, "session_id": self.session_id,
               "state": self.state, "created_at": self.created_at, "updated_at": self.updated_at,
               "expires_at": self.expires_at, "expired": wall >= self.expires_at, "result": dict(self.result),
               "lifecycle": dict(self.lifecycle)}
        if self.error:
            out["error"] = dict(self.error)
        return out


def _err(status: int, code: str, message: str, details: dict | None = None) -> tuple[int, dict]:
    body = {"ok": False, "error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = dict(details)
    return status, body


_REASON = re.compile(r"^(\d{3}) ([a-z_]+): (.*)$", re.S)


def split_reason(reason: str) -> tuple[int, str, str]:
    """twin/motion.py reports refusals as '<status> <code>: <message>'; a bare message is a 422."""
    m = _REASON.match(reason or "")
    if m:
        return int(m.group(1)), m.group(2), m.group(3)
    return 422, "invalid_request", reason or ""


class SimGateway:
    """The vendor SDK gateway's semantics (sdk_gateway/service.py and friends) on top of a SimLamp.
    Every method returns (http_status, json_body); the HTTP server only moves bytes."""

    def __init__(self, lamp: SimLamp, *, token: str = DEFAULT_TOKEN, rate_limit: int = RATE_LIMIT_PER_MIN,
                 session_ttl_s: float = SESSION_TTL_S, max_sessions: int = MAX_SESSIONS,
                 max_payload_bytes: int = MAX_PAYLOAD_BYTES, allowed_commands=None, enabled: bool = True,
                 camera_available: bool = True):
        self.lamp = lamp
        lamp.gateway = self
        self.token = token
        self.rate_limit = max(1, int(rate_limit))
        self.session_ttl_s = max(60.0, float(session_ttl_s or SESSION_TTL_S))
        self.max_sessions = max(1, int(max_sessions))
        self.max_payload_bytes = max(1, int(max_payload_bytes))
        configured = allowed_commands or SAFE_COMMAND_TYPES
        self.allowed = {str(c).strip() for c in configured if str(c).strip() and str(c).strip() not in RUNTIME_DISABLED}
        self.enabled = enabled
        # the camera route encodes JPEG with Pillow; without it the simulated camera is unavailable (503)
        self.camera_available = camera_available and importlib.util.find_spec("PIL") is not None
        self.sessions: dict[str, Session] = {}
        self.actions: dict[str, Record] = {}
        self.idempotency: dict[tuple[str, str], str] = {}
        self.rate: dict[str, list[float]] = {}
        self.clips: dict[str, dict] = {}
        self.by_motion: dict[str, str] = {}
        self.pending_light: list[tuple[Record, object]] = []
        self.streams = 0
        self.resources = 0
        self.rejections: deque = deque(maxlen=2000)     # refusals before a record exists (policy, rate limit)
        self._open: set[str] = set()                     # accepted async records not yet terminal

    # ---------------------------------------------------------------- plumbing
    @property
    def clock(self) -> SimClock:
        return self.lamp.clock

    def preflight(self, token_header: str) -> tuple[int, dict] | None:
        """Gateway enabled, then token (service.py:942-949; auth.py:30-49)."""
        if not self.enabled:
            return _err(503, "disabled", "the SDK gateway is turned off")
        if not self.token:
            return _err(503, "unauthorized", "no SDK token is configured")
        supplied = str(token_header or "").strip()
        if supplied.lower().startswith("bearer "):
            supplied = supplied[7:].strip()
        if not hmac.compare_digest(supplied.encode(), self.token.encode()):
            return _err(401, "unauthorized", "the SDK token is missing or wrong")
        return None

    def prune(self, wall: float) -> None:
        """Expired sessions and records go at the start of every session/action call (service.py:1055-1073)."""
        for sid in [s for s, v in self.sessions.items() if wall >= v.expires_at]:
            del self.sessions[sid]
            self.rate.pop(sid, None)
        for aid in [a for a, r in self.actions.items() if wall >= r.expires_at]:
            del self.actions[aid]
        for key in [k for k, aid in self.idempotency.items() if aid not in self.actions]:
            del self.idempotency[key]

    def _complete(self, rec: Record, t: float, state: str, *, result: dict | None = None, error: dict | None = None,
                  status: int = 200) -> bool:
        if rec.state in TERMINAL:
            return False                      # a terminal record never changes (service.py:959-960)
        rec.state = state
        rec.updated_at = self.clock.wall(t)
        rec.result = dict(result or {})
        rec.error = dict(error) if error else None
        rec.status_code = status
        self._open.discard(rec.action_id)
        return True

    def _open_records(self) -> list[Record]:
        return [self.actions[a] for a in list(self._open)
                if a in self.actions and self.actions[a].state not in TERMINAL]

    def motion_busy(self) -> bool:
        return any(r.path == "motion" and r.state not in TERMINAL for r in self._open_records())

    def cancel_motion_records(self, t: float, reason: str) -> None:
        for rec in self._open_records():
            if rec.path == "motion" and rec.state not in TERMINAL:
                self.lamp.motion.cancel(t, rec.motion_id)
                self._complete(rec, t, "canceled", result={"reason": reason})

    # ---------------------------------------------------------------- sessions
    def create_session(self, token: str, body) -> tuple[int, dict]:
        with self.lamp.lock:
            t = self.lamp.now()
            wall = self.clock.wall(t)
            pre = self.preflight(token)
            if pre:
                return pre
            self.prune(wall)
            body = body if isinstance(body, dict) else {}
            app_id = str(body.get("app_id") or body.get("client_id") or "").strip()
            metadata = body.get("metadata", {}) or {}
            if not isinstance(metadata, dict):
                return 500, {"__html__": "Internal Server Error"}   # the vendor route has no try here (rt:15-27)
            if not app_id:
                return _err(400, "invalid_request", "a session needs an app_id")
            if len(self.sessions) >= self.max_sessions:
                return _err(429, "rate_limited", "the session limit is reached", {"max_sessions": self.max_sessions})
            session = Session("sdk_sess_" + _hex(16), app_id, wall, wall + self.session_ttl_s, dict(metadata))
            self.sessions[session.session_id] = session
            return 201, {"ok": True, "protocol_version": PROTOCOL_VERSION, "session": session.to_dict(wall)}

    # ---------------------------------------------------------------- catalogues
    def capability_list(self) -> list[dict]:
        animations = bool(self.lamp.catalog)
        table = {
            "attention.nudge": (False, "turned off by the runtime's policy"),
            "light.glow": (True, ""),
            "motion.move": (True, ""), "clip.play": (True, ""),
            "torque.enable": (True, ""), "torque.release": (True, ""),
            "music.play": (False, ""),                                  # no audio port in the sim (service.py:848-852)
            "sound.play": (False, "sound playback not available"),
            "speech.say": (False, "voiceover not available"),
            "scenario.run": (False, "turned off by the runtime's policy"),
            "animation.play": (animations, "" if animations else "no approved SDK animations configured"),
            "system.stop": (True, ""),
            "status.read": (True, ""),
        }
        out = []
        for name in SAFE_COMMAND_TYPES:
            available, reason = table[name]
            item = {"name": name, "available": available}
            if reason:
                item["reason"] = reason
            out.append(item)
        return out

    def capabilities(self, token: str) -> tuple[int, dict]:
        with self.lamp.lock:
            pre = self.preflight(token)
            if pre:
                return pre
            audio = {"available": False, "max_bytes": 20971520, "formats": ["wav", "mp3", "flac", "ogg"]}
            resources = {
                "camera": {"available": self.camera_available, "default_fps": 10, "max_fps": 30},
                "microphone": {"available": False, "playback": False},
                "joints": {"available": True, "units": UNITS, "target_planner": "runtime",
                           "self_collision_check": self.lamp.info["self_collision_check"]},
                "clips": {"available": True, "max_bytes": CLIP_MAX_BYTES, "max_frames": CLIP_MAX_FRAMES,
                          "max_seconds": int(CLIP_MAX_SECONDS)},
                "streams": {"transport": "http-multipart", "max_concurrent": MAX_STREAMS,
                            "idle_timeout_seconds": int(STREAM_IDLE_S)},
                "light": {"rgb": [0, 255], "luminance": [0, 1]},
                "torque": {"available": True, "operations": ["enable", "release"], "scope": "whole_robot"},
                "sounds": {**audio, "max_seconds": 60}, "music": {**audio, "max_seconds": 600},
            }                                                           # service.py:591-637
            return 200, {"ok": True, "protocol_version": PROTOCOL_VERSION, "capabilities": self.capability_list(),
                         "forbidden": list(CAPABILITY_FORBIDDEN), "resources": resources}

    def animations(self, token: str) -> tuple[int, dict]:
        with self.lamp.lock:
            pre = self.preflight(token)
            if pre:
                return pre
            items = [{"id": n, "name": n, "approved": True, "recordable": False, "uploadable": False}
                     for n in sorted(self.lamp.catalog)]
            return 200, {"ok": True, "protocol_version": PROTOCOL_VERSION, "animations": items,
                         "forbidden": list(ANIMATION_FORBIDDEN)}

    def scenarios(self, token: str) -> tuple[int, dict]:
        pre = self.preflight(token)
        return pre or _err(503, "capability_unavailable",
                           "saved-scenario handoff is turned off by the runtime's policy")

    # ---------------------------------------------------------------- policy
    def _policy(self, command: str, payload: dict, metadata: dict, ttl) -> tuple[int, dict] | None:
        """validate_action_request, in the vendor's order (policy.py:67-155)."""
        if not command:
            return _err(400, "invalid_command", "command_type is required.")
        if is_forbidden(command):
            return _err(403, "forbidden_command", f"Command is not exposed through LeLamp SDK: {command}")
        if command not in SAFE_COMMAND_TYPES:
            return _err(400, "unsupported_command", f"not an SDK command: {command}",
                        {"supported": list(SAFE_COMMAND_TYPES)})
        if command in RUNTIME_DISABLED:
            return _err(403, "forbidden_command", f"turned off by the runtime's policy: {command}")
        if command not in self.allowed:
            return _err(403, "forbidden_command", f"SDK command disabled by policy: {command}")
        if command in ("torque.enable", "torque.release") and payload:
            return _err(400, "invalid_request", "torque commands take no parameters")
        try:
            size = len(json.dumps({"command_type": command, "payload": payload, "metadata": metadata},
                                  default=str, separators=(",", ":")).encode())
        except Exception:
            size = 0
        if size > self.max_payload_bytes:
            return _err(413, "invalid_request", "the action payload is too big",
                        {"payload_bytes": size, "max_payload_bytes": self.max_payload_bytes})
        if ttl is not None and ttl <= 0:
            return _err(400, "invalid_request", "ttl_seconds has to be above zero")
        if command == "speech.say" and not str(payload.get("text") or "").strip():
            return _err(400, "invalid_request", "speech.say needs payload.text")
        if command == "animation.play":
            if any(k in ANIMATION_BANNED_KEYS or k not in ANIMATION_KEYS for k in payload):
                return _err(403, "forbidden_command", "animation.play plays approved animation ids only; no recording, "
                            "upload, choreography or raw motor values")
            if not self._animation_id(payload):
                return _err(400, "invalid_request", "animation.play needs payload.animation_id")
            approved = sorted(self.lamp.catalog)
            name = self._animation_id(payload)
            if name not in approved:                                    # service.py:1095-1110
                return _err(403, "forbidden_command", f"Animation is not approved for SDK playback: {name}",
                            {"approved_animation_ids": approved})
        return None

    @staticmethod
    def _animation_id(payload: dict) -> str:
        raw = str(payload.get("animation_id") or payload.get("id") or payload.get("name") or "").strip()
        if not raw or "/" in raw or "\\" in raw:
            return ""
        return (raw[:-4] if raw.lower().endswith(".csv") else raw).strip()

    def _rate_ok(self, sid: str, wall: float) -> tuple[int, dict] | None:
        window = self.rate.setdefault(sid, [])
        window[:] = [w for w in window if w >= wall - RATE_WINDOW_S]
        if len(window) >= self.rate_limit:
            return _err(429, "rate_limited", "too many SDK actions in the last 60 s",
                        {"max_actions_per_minute": self.rate_limit})
        window.append(wall)
        return None

    # ---------------------------------------------------------------- POST /actions
    def request_action(self, token: str, session_header: str, body) -> tuple[int, dict]:
        """Returns (status, body) at once for everything except the synchronous commands that have to wait
        (light.glow, system.stop): those return ("wait", record, deadline) for the HTTP layer to wait on."""
        with self.lamp.lock:
            t = self.lamp.now()
            self.lamp.service(t)
            wall = self.clock.wall(t)
            pre = self.preflight(token)
            if pre:
                return pre
            self.prune(wall)
            body = body if isinstance(body, dict) else {}
            try:
                command = str(body.get("command_type") or body.get("type") or "").strip()
                payload = dict(body.get("payload", {}) or {})
                metadata = dict(body.get("metadata", {}) or {})
            except Exception as exc:                                    # dict() of a string/list: the route's except
                return _err(500, "action_failed", str(exc) or type(exc).__name__)
            sid = str(body.get("session_id") or session_header or "").strip()
            key = str(body.get("idempotency_key") or "").strip()
            ttl = _optional_float(body.get("ttl_seconds", body.get("ttl")))
            try:
                asked = float(body.get("timeout") or ROUTE_TIMEOUT_S[1])
                route_timeout = max(ROUTE_TIMEOUT_S[0], min(ROUTE_TIMEOUT_S[2], asked))
            except (TypeError, ValueError):
                route_timeout = ROUTE_TIMEOUT_S[1]
            session = self.sessions.get(sid)
            if session is None or wall >= session.expires_at:
                return _err(401, "unauthorized", "a valid session_id is needed")
            if key and (sid, key) in self.idempotency and self.idempotency[(sid, key)] in self.actions:
                rec = self.actions[self.idempotency[(sid, key)]]   # replay: body not compared (service.py:297-306)
                return 200, {"ok": True, "action": rec.to_dict(wall), "idempotent": True}
            refusal = self._policy(command, payload, metadata, ttl) or self._rate_ok(sid, wall)
            if refusal:
                self.rejections.append({"t": t, "command_type": command, "session_id": sid,
                                        "error": refusal[1]["error"]})
                return refusal
            effective = ACTION_TTL_S if ttl is None else min(ACTION_TTL_S, max(1.0, ttl))
            rec = Record("sdk_act_" + _hex(16), command, sid, "running", wall, wall,
                         wall + max(effective, SYNC_RECORD_MIN_S), t_created=t)
            self.actions[rec.action_id] = rec
            if key:
                self.idempotency[(sid, key)] = rec.action_id
            try:
                return self._dispatch(rec, command, payload, metadata, t, wall, route_timeout, session)
            except Exception as exc:                                    # an unexpected error in a sync command
                self._complete(rec, t, "failed", error={"code": "action_failed", "message": str(exc)}, status=500)
                return _err(500, "action_failed", str(exc) or type(exc).__name__)

    def _dispatch(self, rec, command, payload, metadata, t, wall, route_timeout, session):
        device = command in ("torque.enable", "torque.release", "music.play") or (
            command == "sound.play" and "asset_id" in payload)
        if device:
            return self._device(rec, command, payload, t, wall)
        if command in ("motion.move", "clip.play"):
            return self._motion(rec, command, payload, t, wall)
        if command == "status.read":
            status = {"gateway": {"enabled": self.enabled, "active_sessions": len(self.sessions),
                                  "recent_actions": len(self.actions)},
                      "capabilities": self.capability_list()}
            self._complete(rec, t, "succeeded", result={"status": status, "sdk_action_id": rec.action_id})
            return 200, {"ok": True, "action": rec.to_dict(wall)}
        if command == "light.glow":
            try:
                cmd = self.lamp.light.glow(payload, t=t, session=rec.session_id)
            except GlowInvalid as exc:
                self._complete(rec, t, "rejected", error={"code": "invalid_request", "message": str(exc)}, status=400)
                return _err(400, "invalid_request", str(exc))
            except GlowCrash as exc:
                self._complete(rec, t, "failed", error={"code": "action_failed", "message": str(exc)}, status=500)
                return _err(500, "action_failed", str(exc))
            self.pending_light.append((rec, cmd))
            self.on_time(t)
            return ("wait", rec, t + route_timeout)
        if command == "animation.play":
            return self._animation(rec, payload, metadata, t, wall, session)
        if command == "system.stop":
            return self._system_stop(rec, payload, t, route_timeout)
        if command == "speech.say":
            self._complete(rec, t, "failed", error={"code": "capability_unavailable",
                                                    "message": "voiceover not available"}, status=503)
            return _err(503, "capability_unavailable", "voiceover not available")
        if command == "sound.play":
            self._complete(rec, t, "failed", error={"code": "capability_unavailable",
                                                    "message": "sound playback not available"}, status=503)
            return _err(503, "capability_unavailable", "sound playback not available")
        self._complete(rec, t, "failed", error={"code": "unsupported_command",
                                                "message": f"not an SDK command: {command}"}, status=400)
        return _err(400, "unsupported_command", f"not an SDK command: {command}")

    # device path: torque and uploaded audio (service.py:372-389, 773-818)
    def _device(self, rec, command, payload, t, wall):
        rec.path = "device"
        in_flight = sum(1 for r in self._open_records() if r.path == "device" and r is not rec)
        if in_flight >= DEVICE_CAPACITY:
            self._complete(rec, t, "rejected", error={"code": "capability_unavailable",
                                                      "message": "no free device-action slot"}, status=503)
            return _err(503, "capability_unavailable", "no free device-action slot")
        rec.state, rec.result, rec.expires_at = "accepted", {}, wall + ASYNC_RECORD_S
        self._open.add(rec.action_id)

        def fail(when, message):
            self._complete(rec, when, "failed", error={"code": "action_failed", "message": message}, status=500)

        if command in ("music.play", "sound.play"):
            self.lamp.schedule(t, lambda when: fail(when, "audio playback not available"))
        elif command == "torque.release":
            self.lamp.release_torque(t)
            self.lamp.schedule(t + TORQUE_RAMP_S, lambda when: self._complete(
                rec, when, "succeeded", result={"torque_enabled": False}))
        else:
            def enable(when):
                if rec.state in TERMINAL:
                    return
                self.lamp.enable_torque(when)
                self._complete(rec, when, "succeeded", result={"torque_enabled": True})
            self.lamp.schedule(t + TORQUE_ENABLE_S, enable)
        return 202, {"ok": True, "action": rec.to_dict(wall)}

    # safe-motion path: motion.move and clip.play (service.py:390-449; research.py:74-125)
    def _motion(self, rec, command, payload, t, wall):
        rec.path = "motion"
        in_flight = sum(1 for r in self._open_records() if r.path == "motion" and r is not rec)
        if in_flight >= MOTION_CAPACITY:
            self._complete(rec, t, "rejected", error={"code": "capability_unavailable",
                                                      "message": "no free safe-motion slot"}, status=503)
            return _err(503, "capability_unavailable", "no free safe-motion slot")

        def refuse(message: str, status: int = 422, code: str = "invalid_request"):
            details = {"reachable": False, "collision_checked": "head could hit the base" in message}
            if status != 422:
                details = None
            self._complete(rec, t, "rejected", error={"code": code, "message": message,
                                                      **({"details": details} if details else {})}, status=status)
            return _err(status, code, message, details)

        not_ready = "motion unavailable: the arm is disconnected, stopped or limp"
        if command == "motion.move":
            if "positions" not in payload or set(payload) - {"positions", "duration_seconds"}:
                return refuse("motion.move needs positions")
            legacy = payload.get("duration_seconds")
            if legacy is not None and (isinstance(legacy, bool) or not isinstance(legacy, (int, float))
                                       or not math.isfinite(float(legacy)) or float(legacy) != 0):
                return refuse("motion.move takes no duration: the runtime picks it")
            if not self.lamp.torque_enabled:
                return refuse(not_ready)
            positions = payload.get("positions")
            try:
                requested = self._validate_positions(positions)
            except ValueError as exc:
                return refuse(str(exc))
            measured = self.lamp.measured(t)
            rec.target = {**measured, **requested}
            result = self.lamp.motion.submit_move(t, requested)
        else:
            if set(payload) != {"clip_id"}:
                return refuse("clip.play takes clip_id and nothing else")
            clip_id = str(payload.get("clip_id"))
            if not re.fullmatch(r"[0-9a-f]{32}", clip_id):
                return refuse("Invalid clip ID")
            clip = self.clips.get(clip_id)
            if clip is None:
                return refuse("Clip not found")
            if clip["robot_id"] != ROBOT_ID or clip["calibration_id"] != self.lamp.info["calibration_id"]:
                return refuse("the clip was stored under another calibration: upload it again")
            if not self.lamp.torque_enabled:
                return refuse(not_ready)
            times, positions = clip["frames"]
            frames = [(float(ts), {j: float(v) for j, v in zip(JOINTS, row, strict=True)})
                      for ts, row in zip(times, positions, strict=True)]
            result = self.lamp.motion.submit_clip(t, frames)
        if not result.accepted:
            status, code, message = split_reason(result.reason)
            return refuse(message, status, code)
        rec.motion_id = result.action_id
        self.by_motion[result.action_id] = rec.action_id
        rec.state, rec.expires_at = "accepted", wall + ASYNC_RECORD_S
        self._open.add(rec.action_id)
        rec.result = {"estimated_duration_seconds": round(float(result.planned_duration_s), 6), "reachable": True,
                      "collision_checked": self.lamp.info["self_collision_check"]}
        if self.lamp.animation is not None:
            # a safe move (85) pre-empts a vendor animation (70); the model sees it as a newer clip
            self.lamp.animation_log.append({**self.lamp.animation, "outcome": "preempted", "ended_at": t})
            self.lamp.animation = None
        return 202, {"ok": True, "action": rec.to_dict(wall)}

    @staticmethod
    def _validate_positions(positions) -> dict:
        # control/safe_motion.py:18-46
        if not isinstance(positions, dict) or not positions:
            raise ValueError("positions are missing")
        out = {}
        for name, raw in positions.items():
            if name not in JOINTS:
                raise ValueError(f"no joint named {name}")
            if isinstance(raw, bool):
                raise ValueError(f"{name} is not a finite number")
            try:
                value = float(raw)
            except (TypeError, ValueError):
                raise ValueError(f"{name} is not a finite number") from None
            if not math.isfinite(value):
                raise ValueError(f"{name} is not a finite number")
            if not CALIBRATED[0] <= value <= CALIBRATED[1]:
                raise ValueError(f"{name} is out of its calibrated range")
            out[name] = value
        return out

    def _animation(self, rec, payload, metadata, t, wall, session):
        name = self._animation_id(payload)
        try:
            priority = int(payload.get("priority", ANIMATION_PRIORITY) or ANIMATION_PRIORITY)
        except (TypeError, ValueError) as exc:
            self._complete(rec, t, "failed", error={"code": "action_failed", "message": str(exc)}, status=500)
            return _err(500, "action_failed", str(exc))
        ttl = _optional_float(payload.get("ttl")) or 30.0
        accepted, reason = self.lamp.play_animation(name, t, priority=priority)
        if metadata:
            # On the lamp, top-level metadata is merged into the behaviour intent and the motion manager acts
            # on keys such as pose/positions, bypassing the planner (commands.md section 0, item 5). The sim
            # does not act on it; it records the hazard.
            self.lamp.animation_log.append({"name": name, "outcome": "warning", "t": t,
                                            "reason": "top-level metadata reaches the motion manager on the lamp",
                                            "metadata_keys": sorted(metadata)})
        intent = {"name": "sdk_animation_play", "source": "lelamp_sdk_gateway", "category": "direct_animation",
                  "priority": priority, "suggested_motion": name, "ttl": ttl,
                  "style": {"gentle": payload.get("gentle", True) is not False,
                            "transition": payload.get("transition", True) is not False,
                            "transition_profile": payload.get("transition_profile")},
                  "metadata": {"sdk_action_id": rec.action_id, "sdk_session_id": session.session_id,
                               "explicit_animation": True, "motion_action": True, "recordable": False,
                               "uploadable": False, **metadata}}
        behaviour = {"accepted": True, "request_name": "sdk_animation_play", "reason": "accepted_async",
                     "queued": True, "fallback_used": False, "action_id": "beh_" + _hex(12)}
        # fire and forget: success is "queued", whatever happens next (mapper.py:458-472). The real outcome
        # (played, refused "blocked_by_active_motion", ...) reaches no HTTP route at all: the lamp only logs
        # it, and GET /api/animations/status reports the dashboard's plays, never the SDK's
        # (legacy_routes/animations.py:86-88, 99). Here it is in the sim's animation log (animations.json).
        result = {"animation": {"id": name, "played": True, "recordable": False, "uploadable": False},
                  "intent": intent, "behavior_result": behaviour, "sdk_action_id": rec.action_id}
        self._complete(rec, t, "succeeded", result=result)
        return 200, {"ok": True, "action": rec.to_dict(wall)}

    def _system_stop(self, rec, payload, t, route_timeout):
        """Behaviour stop, motion stop, torque release over 600 ms, then light stop (600 ms fade): the arm goes
        limp and the light goes off (mapper.py:337-414). NOT a pause. Audio would not be stopped."""
        reason = str(payload.get("reason") or "sdk_request").strip()
        self.lamp.release_torque(t)                                    # also stops playback and animations
        stopped = {"behavior": {"accepted": True, "reason": f"sdk_system_stop:{reason}"},
                   "motion": {"stopped": True, "torque_released": True}}

        def light_off(when):
            cmd = self.lamp.light.stop(t=when)

            def finish(done_at):
                if cmd.status == "succeeded":
                    stopped["light"] = _lifecycle(cmd, rec.action_id)
                    self._complete(rec, done_at, "succeeded",
                                   result={"stopped": stopped, "sdk_action_id": rec.action_id})
                else:
                    self._complete(rec, done_at, "failed", status=503, error={
                        "code": "capability_unavailable", "message": "No lifecycle acknowledgment for light.command"})
            self.pending_light.append((rec, cmd))
            rec.lifecycle_finish = finish                              # type: ignore[attr-defined]
        self.lamp.schedule(t + TORQUE_RAMP_S, light_off)
        return ("wait", rec, t + route_timeout)

    # ---------------------------------------------------------------- time passing
    def on_time(self, t: float) -> None:
        """Answer light calls whose reply time has come (called under the lock by SimLamp.service)."""
        keep = []
        for rec, cmd in self.pending_light:
            if cmd.reply_at is not None and cmd.reply_at <= t + 1e-9 and cmd.done:
                finish = getattr(rec, "lifecycle_finish", None)
                if finish is not None:
                    finish(cmd.reply_at)
                elif cmd.status == "succeeded":
                    self._complete(rec, cmd.reply_at, "succeeded", result=_glow_result(cmd, rec.action_id))
                else:
                    self._complete(rec, cmd.reply_at, "failed", status=503, error={
                        "code": "capability_unavailable",
                        "message": "No lifecycle acknowledgment for light.command"})
            else:
                keep.append((rec, cmd))
        self.pending_light = keep

    def on_motion(self, t: float, state: MotionState) -> None:
        for rec in self._open_records():
            if rec.path in ("motion", "device") and rec.state == "accepted":
                rec.state = "running"          # the worker starts; updated_at does not change (service.py:708, 776)
            if rec.path == "motion" and rec.state == "running" and \
                    self.clock.wall(t) - rec.created_at > SAFE_MOTION_CAP_S:
                self.lamp.motion.cancel(t, rec.motion_id)
                self._complete(rec, t, "failed", status=504, error={
                    "code": "action_failed", "message": "Safe motion timed out", "details": {"reached": False}})
        for mid, outcome in state.finished:
            aid = self.by_motion.get(mid)
            rec = self.actions.get(aid) if aid else None
            if rec is None or rec.state in TERMINAL:
                continue
            self._motion_outcome(rec, mid, outcome, t)

    def _motion_outcome(self, rec: Record, mid: str, outcome: str, t: float) -> None:
        detail = getattr(self.lamp.motion, "actions", {}).get(mid)
        when = float(getattr(detail, "t_finished", None) or t)
        when = min(when, t)
        if outcome == "canceled":
            self._complete(rec, when, "canceled", result={"reason": "cancelled_or_preempted"})
            return
        reason = str(getattr(detail, "reason", "") or "")
        if outcome == "rejected":
            status, code, message = split_reason(reason or "calibration or readiness changed before the motion started")
            self._complete(rec, when, "rejected", status=422, error={
                "code": "invalid_request", "message": message,
                "details": {"reachable": False, "collision_checked": "head could hit the base" in message}})
            return
        target = getattr(detail, "target", None) or rec.target or dict(self.lamp.state.measured)
        tol = tolerances()
        measured = self.lamp.measured(when)
        errors = getattr(detail, "settle_errors", None) or {j: abs(measured[j] - target[j]) for j in JOINTS}
        errors = {j: round(float(errors[j]), 4) for j in JOINTS}
        if outcome == "succeeded":
            duration = getattr(detail, "executed_duration_s", None) or rec.result.get("estimated_duration_seconds", 0.0)
            result = {**rec.result, "completed": True, "duration_seconds": round(float(duration), 6),
                      "collision_checked": self.lamp.info["self_collision_check"]}   # always true on the lamp
            if rec.command_type == "motion.move":
                result.update(reached=True, position_errors=errors, position_tolerances=tol)
            self._complete(rec, when, "succeeded", result=result)
            return
        message = reason or "Safe motion failed"
        details = {"reached": False}
        if message.startswith(NOT_REACHED):
            outside = sorted(j for j in JOINTS if errors[j] > tol[j])
            details.update(target_positions={j: float(target[j]) for j in JOINTS}, actual_positions=measured,
                           position_errors=errors, outside_tolerance=outside, tolerance=TARGET_TOLERANCE,
                           position_tolerances=tol)
        self._complete(rec, when, "failed", status=500, error={"code": "action_failed", "message": message,
                                                               "details": details})

    # ---------------------------------------------------------------- GET /actions/<id> and cancel
    def action_status(self, token: str, action_id: str) -> tuple[int, dict]:
        with self.lamp.lock:
            t = self.lamp.now()
            self.lamp.service(t)
            wall = self.clock.wall(t)
            pre = self.preflight(token)
            if pre:
                return pre
            self.prune(wall)
            rec = self.actions.get(action_id)
            if rec is None:
                return _err(404, "action_not_found", "SDK action not found.")
            return 200, {"ok": True, "action": rec.to_dict(wall)}

    def cancel(self, token: str, action_id: str) -> tuple[int, dict]:
        with self.lamp.lock:
            t = self.lamp.now()
            self.lamp.service(t)
            wall = self.clock.wall(t)
            pre = self.preflight(token)
            if pre:
                return pre
            self.prune(wall)
            rec = self.actions.get(action_id)
            if rec is None:
                return _err(404, "action_not_found", "SDK action not found.")
            if rec.state in TERMINAL:
                return _err(409, "action_failed", f"SDK action is already terminal: {rec.state}.",
                            {"action_id": rec.action_id, "state": rec.state})
            if rec.path == "motion":
                self.lamp.motion.cancel(t, rec.motion_id)             # the arm holds the last goal (runtime.py:191-195)
                self._complete(rec, t, "canceled", result={"reason": "cancelled_or_preempted"})
            elif rec.path == "device":
                self._complete(rec, t, "canceled", result={})
            else:
                return _err(409, "action_failed", "this action cannot be cancelled", {"action_id": rec.action_id})
            return 200, {"ok": True, "action": rec.to_dict(wall)}

    # ---------------------------------------------------------------- resources: joints, clips, audio
    def _resource(self, token: str, fn: Callable[[float, float], tuple[int, dict]]) -> tuple[int, dict]:
        pre = self.preflight(token)
        if pre:
            return pre
        with self.lamp.lock:
            if self.resources >= RESOURCE_CAPACITY:
                return _err(429, "rate_limited", "too many resource requests in flight")
            self.resources += 1
            try:
                t = self.lamp.now()
                self.lamp.service(t)
                return fn(t, self.clock.wall(t))
            finally:
                self.resources -= 1

    def joints(self, token: str) -> tuple[int, dict]:
        return self._resource(token, lambda t, wall: (200, {"ok": True, **self.lamp.info,
                                                            "positions": self.lamp.measured(t)}))

    def clips_list(self, token: str) -> tuple[int, dict]:
        return self._resource(token, lambda t, wall: (200, {"ok": True, "clips": [
            {k: v for k, v in c.items() if k not in ("csv", "frames")} for _, c in sorted(self.clips.items())][:100]}))

    def clips_upload(self, token: str, data: bytes) -> tuple[int, dict]:
        def upload(t, wall):
            if len(self.clips) >= CLIP_STORE_MAX:
                return _err(422, "invalid_request", "clip storage is full: delete clips you no longer need")
            try:
                times, positions = read_motion_csv(data)
                if self.lamp.pose_ok is not None:      # self-collision on every uploaded row (runtime.py:143-147)
                    for ts, row in zip(times, positions, strict=True):
                        if not self.lamp.pose_ok({j: float(v) for j, v in zip(JOINTS, row, strict=True)}):
                            raise ValueError(f"Motion rejected because the head could hit the base at {ts:.2f} seconds")
            except ValueError as exc:
                return _err(422, "invalid_request", str(exc))
            text = data.decode("utf-8-sig")
            clip = {"id": uuid.uuid4().hex, "robot_id": ROBOT_ID, "calibration_id": self.lamp.info["calibration_id"],
                    "duration_seconds": float(times[-1]), "sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "csv": text, "frames": (times, positions)}
            self.clips[clip["id"]] = clip
            return 200, {"ok": True, "clip": {k: v for k, v in clip.items() if k not in ("csv", "frames")}}
        return self._resource(token, upload)

    def clips_delete(self, token: str, clip_id: str) -> tuple[int, dict]:
        def delete(t, wall):
            if not re.fullmatch(r"[0-9a-f]{32}", clip_id or ""):
                return _err(422, "invalid_request", "Invalid clip ID")
            self.clips.pop(clip_id, None)                         # idempotent (clip_store.py:95-97)
            return 200, {"ok": True, "deleted": clip_id}
        return self._resource(token, delete)

    def audio_resource(self, token: str) -> tuple[int, dict]:
        # research.audio is None on this sim, like a lamp without a supported USB audio board (research.py:36-38)
        return self._resource(token, lambda t, wall: _err(503, "capability_unavailable", "audio library not available"))

    # ---------------------------------------------------------------- the runtime's own status routes
    def runtime_status(self) -> tuple[int, dict]:
        with self.lamp.lock:
            return 200, {"servo": True, "light": True, "camera": self.camera_available, "is_sleeping": False,
                         "is_muted": False, "available_animations": sorted(self.lamp.catalog),
                         "available_light_animations": list(EFFECTS), "platform": "feelthemusic-twin-sim",
                         "animations_dir": None, "neutral_urdf_joints": {}}   # legacy_routes/logs_status.py:42-57

    def animation_status(self) -> tuple[int, dict]:
        """GET /api/animations/status (legacy_routes/animations.py:102-126). playing, current_animation,
        started_at, elapsed_seconds and last_error belong to the dashboard's own play/stop routes, which the
        sim does not serve: an SDK animation.play never changes them (it does not on the lamp either). The idle
        fields come from the motion layer: the idle's name and whether it plays now."""
        with self.lamp.lock:
            t = self.lamp.now()
            self.lamp.service(t)
            dash = self.lamp.dashboard_animation
            idle = self.lamp.idle_name
            return 200, {"playing": dash is not None, "current_animation": dash["name"] if dash else None,
                         "started_at": dash["started_at"] if dash else None,
                         "elapsed_seconds": round(self.clock.wall(t) - dash["started_at"], 1) if dash else None,
                         "current_idle": idle,
                         "idle_paused": idle is not None and not self.lamp.state.idle_playing,
                         "idle_playing": bool(self.lamp.state.idle_playing),
                         "last_error": self.lamp.dashboard_error}

    def set_idle(self, body) -> tuple[int, dict]:
        """POST /api/animations/idle, the dashboard's idle selector (legacy_routes/animations.py:175-215). It is
        not an SDK route: no token. The answer comes as soon as the runtime has queued the idle request, before
        anything happens, so it is 200 {"status": "ok", "idle": <name or null>} for ANY string name: whether
        the idle really changed is only in the lamp's log (see SimLamp.request_idle; here, the animation log).
        {"name": "none"} switches idle off; {"name": ""} or null does NOT, although it answers ok."""
        if body and not isinstance(body, dict):
            return 500, {"__html__": "Internal Server Error"}           # (a list).get: before the route's try
        name = (body or {}).get("name")
        try:
            idle_name = Path(name).stem if name else None               # "idle.csv" and "x/idle" -> "idle"
        except Exception as exc:                                         # a number, a list: the route's except
            return 500, {"error": str(exc)}
        with self.lamp.lock:
            t = self.lamp.now()
            self.lamp.service(t)
            self.lamp.request_idle(idle_name, t)
        request_id = "behavior_" + _hex(12)                             # event_lifecycle.py:123-128
        return 200, {"status": "ok", "idle": idle_name, "event_id": uuid.uuid4().hex[:8],   # event_bus.py:185
                     "command_id": request_id, "request_id": request_id}


def _lifecycle(cmd, action_id: str) -> dict:
    state = cmd.result.get("state", "SUCCEEDED") if cmd.result else "SUCCEEDED"
    return {"ok": True, "event_id": "evt_" + _hex(12), "command_id": "cmd_" + _hex(12),
            "request_id": "sdk_light_" + _hex(12), "action_id": action_id,
            "result": {"state": state, "success": True, "reason": "", "light_command_id": cmd.id},
            "error": None}                                           # mapper.py:521-531


def _glow_result(cmd, action_id: str) -> dict:
    """The mapper's result per branch (mapper.py:147-207)."""
    life = _lifecycle(cmd, action_id)
    if cmd.kind == "solid":
        return {"luminance": cmd.level, "light": "set_solid", "color": list(cmd.color), "lifecycle": life}
    if cmd.kind == "brightness" and not cmd.effect:
        return {"luminance": cmd.level, "lifecycle": life}
    name = str(cmd.effect or cmd.payload.get("animation") or cmd.payload.get("effect") or "breathing")
    return {"light": "play_animation", "animation": name, "lifecycle": life}


# ------------------------------------------------------------------ HTTP
_ROUTES = [
    ("POST", r"/api/sdk/v1/sessions", "sessions"),
    ("GET", r"/api/sdk/v1/capabilities", "capabilities"),
    ("GET", r"/api/sdk/v1/animations", "animations"),
    ("GET", r"/api/sdk/v1/scenarios", "scenarios"),
    ("DELETE", r"/api/sdk/v1/scenarios/(?P<ident>[^/]+)", "scenarios"),
    ("POST", r"/api/sdk/v1/actions", "actions"),
    ("GET", r"/api/sdk/v1/actions/(?P<ident>[^/]+)", "action_status"),
    ("POST", r"/api/sdk/v1/actions/(?P<ident>[^/]+)/cancel", "cancel"),
    ("GET", r"/api/sdk/v1/joints", "joints"),
    ("GET", r"/api/sdk/v1/clips", "clips_list"),
    ("POST", r"/api/sdk/v1/clips", "clips_upload"),
    ("DELETE", r"/api/sdk/v1/clips/(?P<ident>[^/]+)", "clips_delete"),
    ("GET", r"/api/sdk/v1/(?:sounds|music)", "audio"),
    ("POST", r"/api/sdk/v1/(?:sounds|music)", "audio"),
    ("DELETE", r"/api/sdk/v1/(?:sounds|music)/(?P<ident>[^/]+)", "audio"),
    ("GET", r"/api/sdk/v1/camera/snapshot", "snapshot"),
    ("GET", r"/api/sdk/v1/streams/(?P<ident>[^/]+)", "stream"),
    ("GET", r"/api/status", "runtime_status"),
    ("GET", r"/api/animations/status", "animation_status"),
    ("POST", r"/api/animations/idle", "animation_idle"),
]
_COMPILED = [(m, re.compile("^" + p + "$"), name) for m, p, name in _ROUTES]


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "LeLampTwinSim/1"
    timeout = 60

    def log_message(self, fmt, *args):
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_DELETE(self):
        self._dispatch("DELETE")

    def do_PUT(self):
        self._dispatch("PUT")

    # -------------------------------------------------------------------------------------------------
    def _token(self) -> str:
        # routes/sdk.py:107-112
        return self.headers.get("Authorization") or self.headers.get("X-LeLamp-SDK-Token") or ""

    def _body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length > 0 else b""

    def _json_value(self, raw: bytes):
        """request.get_json(silent=True): the parsed JSON, of any type, only with a JSON content type; None
        otherwise or when it does not parse."""
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json" and not ctype.endswith("+json"):
            return None
        try:
            return json.loads(raw.decode("utf-8")) if raw else None
        except (ValueError, UnicodeDecodeError):
            return None

    def _json_body(self, raw: bytes):
        """request.get_json(silent=True) or {}, as a dict (routes/sdk.py:119-121)."""
        value = self._json_value(raw)
        return value if isinstance(value, dict) else {}

    def _send(self, status: int, data: bytes, content_type: str, headers: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _json(self, status: int, payload: dict) -> None:
        if "__html__" in payload:
            self._send(status, b"<!doctype html><title>500 Internal Server Error</title>", "text/html; charset=utf-8")
            return
        self._send(status, (json.dumps(payload, sort_keys=True) + "\n").encode(), "application/json")

    def _dispatch(self, method: str) -> None:
        gateway: SimGateway = self.server.gateway
        path, _, query = self.path.partition("?")
        raw = self._body() if method in ("POST", "DELETE", "PUT") else b""
        methods_seen = False
        for m, pattern, name in _COMPILED:
            match = pattern.match(path)
            if not match:
                continue
            methods_seen = True
            if m != method:
                continue
            try:
                getattr(self, "_r_" + name)(gateway, match.groupdict().get("ident"), raw, parse_qs(query))
            except (BrokenPipeError, ConnectionResetError, socket.timeout):
                self.close_connection = True
            return
        if methods_seen:
            self._send(405, b"<!doctype html><title>405 Method Not Allowed</title>", "text/html; charset=utf-8")
        else:
            self._send(404, b"<!doctype html><title>404 Not Found</title>", "text/html; charset=utf-8")

    # routes ------------------------------------------------------------------------------------------
    def _r_sessions(self, gw, ident, raw, query):
        self._json(*gw.create_session(self._token(), self._json_body(raw)))

    def _r_capabilities(self, gw, ident, raw, query):
        self._json(*gw.capabilities(self._token()))

    def _r_animations(self, gw, ident, raw, query):
        self._json(*gw.animations(self._token()))

    def _r_scenarios(self, gw, ident, raw, query):
        self._json(*gw.scenarios(self._token()))

    def _r_actions(self, gw, ident, raw, query):
        answer = gw.request_action(self._token(), self.headers.get("X-LeLamp-SDK-Session") or "", self._json_body(raw))
        if answer[0] == "wait":
            _, rec, deadline = answer
            answer = self._wait_record(gw, rec, deadline)
        self._json(*answer)

    def _wait_record(self, gw: SimGateway, rec: Record, deadline: float):
        """The route waits for the synchronous command up to the body's timeout, in simulated time; past it,
        500 action_failed 'TimeoutError' while the lamp carries on (routes/sdk.py:69-79, 135-142)."""
        lamp = gw.lamp
        real_start = time.monotonic()
        while True:
            with lamp.lock:
                t = lamp.now()
                lamp.service(t)
                if rec.state in TERMINAL:
                    wall = lamp.clock.wall(t)
                    if rec.state == "succeeded":
                        return 200, {"ok": True, "action": rec.to_dict(wall)}
                    err = rec.error or {"code": "action_failed", "message": "action failed"}
                    return _err(rec.status_code or 500, err["code"], err["message"], err.get("details"))
                if t >= deadline - 1e-9 or time.monotonic() - real_start > 600:
                    return _err(500, "action_failed", "TimeoutError")
                wake = min(deadline, lamp.next_due(t))
            lamp.clock.wait_until(wake, real_timeout=0.25)

    def _r_action_status(self, gw, ident, raw, query):
        self._json(*gw.action_status(self._token(), ident))

    def _r_cancel(self, gw, ident, raw, query):
        self._json(*gw.cancel(self._token(), ident))

    def _r_joints(self, gw, ident, raw, query):
        self._json(*gw.joints(self._token()))

    def _r_clips_list(self, gw, ident, raw, query):
        self._json(*gw.clips_list(self._token()))

    def _r_clips_upload(self, gw, ident, raw, query):
        pre = gw.preflight(self._token())
        if pre:
            return self._json(*pre)
        if len(raw) > CLIP_MAX_BYTES:
            return self._json(*_err(413, "invalid_request", "Clip exceeds 5 MiB"))
        self._json(*gw.clips_upload(self._token(), raw))

    def _r_clips_delete(self, gw, ident, raw, query):
        self._json(*gw.clips_delete(self._token(), ident))

    def _r_audio(self, gw, ident, raw, query):
        self._json(*gw.audio_resource(self._token()))

    def _r_runtime_status(self, gw, ident, raw, query):
        self._json(*gw.runtime_status())

    def _r_animation_status(self, gw, ident, raw, query):
        self._json(*gw.animation_status())

    def _r_animation_idle(self, gw, ident, raw, query):
        self._json(*gw.set_idle(self._json_value(raw)))

    def _r_snapshot(self, gw, ident, raw, query):
        pre = gw.preflight(self._token())
        if pre:
            return self._json(*pre)
        if not gw.camera_available:
            return self._json(*_err(503, "capability_unavailable", "Camera unavailable"))
        data, meta = gw.lamp.camera_frame()
        self._send(200, data, "image/jpeg", {"Cache-Control": "no-store",
                                             "X-LeLamp-Metadata": json.dumps(meta, separators=(",", ":"))})

    def _r_stream(self, gw, kind, raw, query):
        """multipart/mixed; boundary=lelamp, no closing delimiter; each part paced at 1/fps and only when the
        camera has a newer frame (routes/sdk_research.py:155-213; media.py:51-113)."""
        token = self._token()
        pre = gw.preflight(token)
        if pre:
            return self._json(*pre)
        try:
            fps = float((query.get("fps") or ["10"])[0])
            if not math.isfinite(fps) or not 0.1 <= fps <= CAMERA_MAX_FPS:
                raise ValueError("fps must be between 0.1 and 30")
            with gw.lamp.lock:
                if gw.streams >= MAX_STREAMS:
                    raise ValueError("no free media stream")
                if kind == "camera":
                    if not gw.camera_available:
                        raise ValueError("no recent camera frame")
                elif kind == "microphone":
                    raise ValueError("no microphone")
                else:
                    raise ValueError("Unknown stream kind")
                gw.streams += 1
        except ValueError as exc:
            return self._json(*_err(503, "capability_unavailable", str(exc)))
        lamp = gw.lamp
        try:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/mixed; boundary=lelamp")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            self.connection.settimeout(STREAM_IDLE_S)    # a client that stops reading is dropped (media.py:120-128)
            previous = None
            while getattr(self.server, "serving", True):
                if gw.preflight(token):
                    break                                        # token re-checked around every frame
                data, meta = None, None
                while getattr(self.server, "serving", True):
                    target = lamp.now() + 1.0 / fps
                    real_start = time.monotonic()
                    while not lamp.clock.wait_until(target, real_timeout=0.5):
                        if time.monotonic() - real_start > STREAM_IDLE_S or not getattr(self.server, "serving", True):
                            return                               # simulated time stopped: end like a stale camera
                    data, meta = lamp.camera_frame()
                    if previous is None or meta["sequence"] != previous:
                        break
                if data is None:
                    break
                meta["gap"] = max(0, meta["sequence"] - previous - 1) if previous is not None else 0
                previous = meta["sequence"]
                head = (f"--lelamp\r\nContent-Type: image/jpeg\r\nContent-Length: {len(data)}\r\n"
                        f"X-LeLamp-Metadata: {json.dumps(meta, separators=(',', ':'))}\r\n\r\n").encode()
                self.wfile.write(head + data + b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError):
            pass
        finally:
            with gw.lamp.lock:
                gw.streams -= 1


class SimSDKServer(ThreadingHTTPServer):
    """The HTTP front of the simulated gateway (stdlib ThreadingHTTPServer). Port 0 picks a free port."""
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, gateway: SimGateway, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, *,
                 verbose: bool = False):
        super().__init__((host, port), _Handler)
        self.gateway = gateway
        self.verbose = verbose
        self.serving = True
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> "SimSDKServer":
        """Serve in a background thread (and start the lamp's real-time stepping if its clock is real)."""
        self.gateway.lamp.start()
        self._thread = threading.Thread(target=self.serve_forever, kwargs={"poll_interval": 0.1},
                                        name="sim-sdk-http", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self.serving = False
        self.shutdown()
        self.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.gateway.lamp.stop()


def build(*, scenario: str | None = "walk_in_sit", seed: int = 0, clock: str = "real", token: str = DEFAULT_TOKEN,
          host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, transition_ms: float = 600.0,
          rate_limit: int = RATE_LIMIT_PER_MIN, record: str | None = None, idle: str = "auto",
          render: str = "auto", camera_fps: float = CAMERA_CAPTURE_FPS,
          size: tuple[int, int] = FRAME_SIZE, verbose: bool = False, collision: str = "required") -> SimSDKServer:
    """Wire a SimLamp, a SimGateway and a SimSDKServer from the team's modules where they exist. Raises
    MissingCollisionModel without the robot package unless collision="off" (see SimLamp)."""
    people = None
    if scenario:
        from twin.world import scenario as make_scenario
        people = make_scenario(scenario, seed)
    kinematics, renderer = None, None
    if render in ("auto", "mujoco"):
        try:
            from twin.model import LampTwin
            twin = LampTwin()
            kinematics = twin

            def renderer(units, who, light, width, height, _twin=twin):
                return _twin.render(units, people=who, light_rgb=light, camera="head", width=width, height=height)
        except Exception:
            if render == "mujoco":
                raise
            kinematics, renderer = None, None
    lamp = SimLamp(clock=clock, scenario=people, kinematics=kinematics, renderer=renderer,
                   transition_ms=transition_ms, camera_fps=camera_fps, frame_size=size, idle=idle,
                   seed=seed, record_dir=record, collision=collision)
    gateway = SimGateway(lamp, token=token, rate_limit=rate_limit)
    return SimSDKServer(gateway, host, port, verbose=verbose)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m twin.sim_sdk", description=__doc__.split("\n\n")[0])
    ap.add_argument("--scenario", default="walk_in_sit", help="twin/world.py scenario name, or 'none'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--token", default=DEFAULT_TOKEN)
    ap.add_argument("--transition-ms", type=float, default=600.0, help="light fade; 600 is the lamp's setting")
    ap.add_argument("--rate-limit", type=int, default=RATE_LIMIT_PER_MIN, help="actions per minute per session")
    ap.add_argument("--record", default=None, help="directory for trace.jsonl, light_log.json, actions.json")
    ap.add_argument("--idle", default="auto", choices=["auto", "vendor", "synthetic", "off"])
    ap.add_argument("--render", default="auto", choices=["auto", "mujoco", "synthetic"])
    ap.add_argument("--camera-fps", type=float, default=CAMERA_CAPTURE_FPS)
    ap.add_argument("--size", default="640x480")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-collision", action="store_true",
                    help="run WITHOUT the self-collision check (degraded: accepts moves the lamp refuses); "
                         "only for when the robot package (FTM_ROBOT_DIR) is not available")
    args = ap.parse_args(argv)
    width, height = (int(v) for v in args.size.lower().split("x"))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # printed below, louder
            server = build(scenario=None if args.scenario == "none" else args.scenario, seed=args.seed,
                           token=args.token, host=args.host, port=args.port, transition_ms=args.transition_ms,
                           rate_limit=args.rate_limit, record=args.record, idle=args.idle,
                           render=args.render, camera_fps=args.camera_fps, size=(width, height),
                           verbose=args.verbose, collision="off" if args.no_collision else "required")
    except MissingCollisionModel as exc:
        print(f"refusing to start: {exc}", file=sys.stderr, flush=True)
        return 2
    lamp = server.gateway.lamp
    print(f"simulated LeLamp SDK gateway at {server.base_url}  (token: {args.token}; the sim token is not a secret)")
    print(f"  motion: {type(lamp.motion).__name__}, idle: {getattr(lamp.motion, 'idle_source', 'n/a')}")
    if lamp.pose_ok is None:
        banner = "!" * 100
        print(f"{banner}\n  DEGRADED: {NO_COLLISION_WARNING}\n{banner}", file=sys.stderr, flush=True)
    else:
        rule = "lamp/spatial.py LampModel" if isinstance(lamp.pose_ok, HeadBaseCheck) else "twin/model.py LampTwin"
        print(f"  self-collision check: the vendor's head-vs-base rule ({rule})")
    print(f"  camera: {'renderer' if lamp.renderer else 'synthetic frames'}, scenario: {args.scenario}, "
          f"light fade {args.transition_ms:.0f} ms, rate limit {args.rate_limit}/min")
    print(f"  client: LampSDK(token={args.token!r}, base={server.base_url!r})", flush=True)

    def _interrupt(signum, frame):
        raise KeyboardInterrupt

    # Ctrl-C and `kill` both stop cleanly and write the --record files, even when a script started us in the
    # background (which leaves SIGINT ignored).
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, _interrupt)
    server.start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("stopping", flush=True)
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)     # a second signal (uv forwards TERM too) must not cut
        signal.signal(signal.SIGTERM, signal.SIG_IGN)    # the shutdown that writes the record short
        server.stop()
        if args.record:
            print(f"recorded to {args.record}: trace.jsonl, light_log.json, actions.json, animations.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
