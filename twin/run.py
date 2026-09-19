"""Run the whole twin on one clock: a room, the lamp's head camera and face detector, the head tracker, the
vendor SDK's motion path, and the music's haptics and light. Writes the evidence of one run.

    python -m twin.run --scenario walk_in_sit --strategy settle --song synthetic --seconds 30 --out DIR
    python -m twin.run --scenario two_people --song runlog:/path/to/run.jsonl --out DIR --no-video
    python -m twin.run --matrix --out DIR          # every scenario x strategy, a summary table

Each step (1/30 s, the SDK's control rate):
  world.people_at(t) -> model.head(measured) = where the camera is -> perception observe/poll ->
  tracker.update -> its Command becomes motion.move / clip.play on SDKMotionModel -> motion.step(t) ->
  model.check(measured): the clearance of the pose the arm is REALLY in, every step.
Light and haptics run on the same clock. The recommended SDK light design (show.BestSdkDesign) is planned
before the run and its light.glow POSTs take their slots from the same SDK session budget as the motion
(one RateWindow), as on the lamp. After the run, show.run_show computes the panel for the ideal design,
the literal ("naive") SDK path and the recommended SDK path, with the real motion POST times.

What one run writes into its out dir:
  report.json                 every metric below, the light/haptic sync, the source of every constant
  trajectory.json             the scenario, the people, the measured joints and every command (hashed)
  replay/manifest.json        sha256 of every model file the run read (vendor files by relative name only)
  replay/search.report.json   the look-around until the first ACQUIRE, evidence mode "dance"
  replay/acquire.report.json  from the first ACQUIRE to the first LOCK (the turn to the face), mode "dance"
  replay/lock.report.json     from the first LOCK to the end, evidence mode "head-follow"
  video.mp4                   (unless --no-video) see twin/video.py

The replay reports use the exact schema of simulation/replay.py on branch codexfranklin/unified-app (15
fields, 6 per sample). Head-follow fails on any sample without valid tracking, so the look-around, the turn
to the face and the lock are separate reports (research report twin-spec/interfaces.md section 6). Together
they cover every step of the run, with no gap (replay_segments).

The robot description and calibration are read at run time (FTM_ROBOT_DIR, FTM_CALIBRATION). The twin is
design evidence, not hardware approval: nothing here stands in for a real SDK outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from twin import motion as M
from twin import perception as PER
from twin import show as S
from twin import world as W
from twin.contract import JOINTS
from twin.tracking import ACQUIRE, LOCK, LOST, STRATEGIES, HeadTracker, TrackerConfig

ROOT = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------ run constants (source per line)
DT = 1.0 / M.FPS            # one step per SDK control frame, 30 Hz (vendor source config/default.yaml:104-105)
# 120 s: two full 60 s rate windows. A shorter run cannot fill the SDK's window, so it never shows the steady
# state of the budget (a 90 s settle run stalled at 55 s); summary_row marks shorter runs "not steady state".
DEFAULT_SECONDS = 120.0
STEADY_STATE_S = 2 * M.RATE_WINDOW_S
# At the vendor default of 30 actions/min per session, settle-chained head tracking needs at most 26 motion
# calls a minute (research report twin-spec/motion.md section 8), which leaves 4 for light.glow on the same
# session. At the lamp's .env value of 120/min the tracker keeps its own default of 40 (tracking.py).
LIGHT_RESERVE_PER_MIN = 4
MOTION_CAP_MAX = TrackerConfig().max_commands_per_min
ON_AXIS_DEG = (5.0, 10.0)   # workflow brief: share of time the locked head is within 5 and 10 deg of the axis
# Replay thresholds, stated here so every run is judged the same way:
#   clearance: the smallest distance the twin's check allows anywhere (LampTwin self_margin 5 mm, an
#     ASSUMPTION from the workflow brief); table 3 cm and base 2 cm are checked separately in report.json.
#   head error: half the camera's vertical field of view (44 deg, measured on the lamp 2026-09-19): past it
#     the face can leave the picture, so head-follow has failed (research report twin-spec/interfaces.md).
REPLAY_MIN_CLEARANCE_M = 0.005
REPLAY_MAX_HEAD_ERROR_DEG = 22.0
# The replay schema needs a finite head error in every sample. With nobody in the room there is nothing to
# aim at; 180 deg (the worst possible) is the truthful value, and tracking_valid is false.
NO_PERSON_ERROR_DEG = 180.0
REPLAY_SIMULATOR = "feelthemusic twin (twin/run.py)"
# The twin's own modules a run executes (hashed into replay/manifest.json). Not sim_sdk.py (the HTTP simulator)
# nor video.py (a view of the run, no model); tests/twin/test_run.py checks this list against the imports.
RUN_MODULES = ("__init__", "contract", "model", "motion", "panel", "perception", "run", "show", "sim_light",
               "tracking", "world")
# ASSUMPTION: a clip left on the lamp by an earlier session is about the size of a lock clip (61 rows of CSV)
EARLIER_CLIP_BYTES = 3000
VIDEO_FPS = 15              # workflow brief: 15-30 fps
SONGS = ("synthetic", "runlog:<path>", "wav:<path>")


# ------------------------------------------------------------------ small helpers
def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two directions; atan2 is exact at 0 and at 180 deg (a cross-product residual is not)."""
    return math.degrees(math.atan2(float(np.linalg.norm(np.cross(a, b))), float(np.dot(a, b))))


def _picture_point(pose, point: np.ndarray, fx: float, fy: float) -> tuple[float, float] | None:
    """Where a base-frame point lands in the head picture (0..1 inside), or None behind the camera."""
    v = np.asarray(point, float) - np.asarray(pose.position, float)
    z = float(v @ np.asarray(pose.forward, float))
    if z <= 1e-6:
        return None
    return (0.5 + fx * float(v @ np.asarray(pose.right, float)) / z,
            0.5 + fy * float(v @ np.asarray(pose.down, float)) / z)


def _in_picture(xy) -> bool:
    return xy is not None and 0.0 <= xy[0] <= 1.0 and 0.0 <= xy[1] <= 1.0


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dump(obj) -> bytes:
    """Strict JSON bytes (no NaN), stable key order, so hashes are reproducible."""
    return json.dumps(S.json_safe(obj), allow_nan=False, sort_keys=True, separators=(",", ":")).encode()


def _write_json(path: Path, obj, *, pretty: bool = True) -> bytes:
    data = (json.dumps(S.json_safe(obj), allow_nan=False, sort_keys=True, indent=1).encode() if pretty
            else _dump(obj))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def _stats(x) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not x.size:
        return {"n": 0}
    return {"n": int(x.size), "mean": round(float(x.mean()), 3), "p50": round(float(np.median(x)), 3),
            "p95": round(float(np.quantile(x, 0.95)), 3), "max": round(float(x.max()), 3)}


def _max_in_60s(times) -> int:
    """Most POSTs in any sliding 60 s window (the SDK's rate window)."""
    ts = sorted(times)
    best, j = 0, 0
    for i, t in enumerate(ts):
        while ts[j] <= t - 60.0:
            j += 1
        best = max(best, i - j + 1)
    return best


def motion_cap_for(rate_limit: int) -> int:
    """How many motion calls a minute the tracker may use on a session with this rate limit."""
    return int(max(1, min(MOTION_CAP_MAX, rate_limit - LIGHT_RESERVE_PER_MIN)))


# ------------------------------------------------------------------ songs
def song_label(spec: str) -> str:
    """A song spec without the local path: 'runlog:<run folder>/<file>' keeps only the last two parts."""
    if ":" not in spec:
        return spec
    kind, path = spec.split(":", 1)
    p = Path(path)
    return f"{kind}:{p.parent.name}/{p.name}" if p.parent.name else f"{kind}:{p.name}"


def load_song(spec: str, seconds: float, seed: int = 0) -> S.Song:
    """'synthetic' (the Mac conductor's synthetic track, looped as the conductor loops it, FileSource.swift:4),
    'runlog:<run.jsonl>' (a real conductor run, first event at t = 2 s) or 'wav:<file>' (through the team's
    analysis module), cut to 0..seconds."""
    if spec == "synthetic":
        song = S.loop_song(S.synthetic_song(seed=seed), float(seconds))
    elif spec.startswith("runlog:"):
        song = S.from_runlog(spec.split(":", 1)[1])
    elif spec.startswith("wav:"):
        song = S.from_analysis(spec.split(":", 1)[1], seed=seed)
    else:
        raise ValueError(f"unknown song {spec!r}; use one of {', '.join(SONGS)}")
    song = song.window(0.0, float(seconds))
    if not song.events:
        raise ValueError(f"song {spec!r} has no events in 0..{seconds:g} s")
    return song


# ------------------------------------------------------------------ metrics (pure functions, tested without the robot)
def tracking_metrics(t: np.ndarray, state: list, pid: list, err_locked: np.ndarray, visible: list,
                     present: np.ndarray, dt: float) -> dict:
    """Head-tracking scores from GROUND TRUTH (world.py), never from detections.

    t, state, pid (ground-truth person id of the locked track, '' when none), err_locked (deg between the
    optical axis and that person's head, NaN when none), visible (per step: ids of people whose head is in
    the picture), present (per step: how many people are in the room)."""
    n = len(t)
    state = list(state)
    first = {s: next((float(t[k]) for k in range(n) if state[k] == s), None) for s in (ACQUIRE, LOCK)}
    t_present = next((float(t[k]) for k in range(n) if present[k] > 0), None)
    t_visible = next((float(t[k]) for k in range(n) if visible[k]), None)

    in_lock = np.array([s == LOCK and p != "" for s, p in zip(state, pid, strict=True)], dtype=bool)
    err = np.asarray(err_locked, dtype=float)
    lock_err = err[in_lock & np.isfinite(err)]
    share_lock = {f"within_{int(d)}_deg": (round(float(np.mean(lock_err <= d)), 4) if lock_err.size else None)
                  for d in ON_AXIS_DEG}
    # Of all the time someone is in the room: facing the person the tracker is on (ACQUIRE or LOCK).
    engaged = np.array([s in (ACQUIRE, LOCK) and p != "" for s, p in zip(state, pid, strict=True)], dtype=bool)
    with_people = present > 0
    share_presence = {}
    for d in ON_AXIS_DEG:
        hit = engaged & np.isfinite(err) & (np.nan_to_num(err, nan=1e9) <= d)
        share_presence[f"within_{int(d)}_deg"] = (round(float(np.sum(hit & with_people) / np.sum(with_people)), 4)
                                                  if with_people.any() else None)

    # Switches of the ground-truth person the tracker is on. A steal: the new person replaces one who is
    # still in the picture. The last person locked is remembered across SEARCH, so dropping someone and
    # locking a neighbour while the first is still in view counts too.
    switches = steals = 0
    last = ""
    for k in range(n):
        p = pid[k]
        if state[k] not in (ACQUIRE, LOCK) or p == "":
            continue
        if last and p != last:
            switches += 1
            if last in visible[k]:
                steals += 1
        last = p
    lost_entries = sum(1 for k in range(1, n) if state[k] == LOST and state[k - 1] != LOST)
    locked_s = float(np.sum(in_lock) * dt)
    by_person: dict[str, float] = {}
    for k in range(n):
        if in_lock[k]:
            by_person[pid[k]] = by_person.get(pid[k], 0.0) + dt
    return {
        "t_first_person_in_room_s": t_present,
        "t_first_face_in_picture_s": t_visible,
        "t_first_acquire_s": first[ACQUIRE],
        "t_first_lock_s": first[LOCK],
        "lock_after_arrival_s": (round(first[LOCK] - t_present, 3) if first[LOCK] is not None
                                 and t_present is not None else None),
        "locked_s": round(locked_s, 3),
        "locked_s_by_person": {k: round(v, 3) for k, v in sorted(by_person.items())},
        "lock_error_deg": _stats(lock_err),
        "share_of_lock_on_axis": share_lock,
        "share_of_presence_on_axis": share_presence,
        "person_switches": switches,
        "lock_steals": steals,
        "lost_episodes": lost_entries,
        "state_time_s": {s: round(sum(dt for x in state if x == s), 3) for s in sorted(set(state))},
    }


def replay_report(t: np.ndarray, clearance: np.ndarray, head_error: np.ndarray, collision: np.ndarray,
                  joint_violation: np.ndarray, tracking_valid: np.ndarray, *, mode: str, scene_id: str,
                  model_sha256: str, trajectory_sha256: str, simulator_version: str) -> dict:
    """One report in the exact schema of simulation/replay.py (branch codexfranklin/unified-app): 15 fields,
    6 per sample, integer nanoseconds, aggregates recomputed from the samples."""
    if mode not in ("head-follow", "dance"):
        raise ValueError("mode must be head-follow or dance")
    if len(t) < 2:
        raise ValueError("a replay report needs at least two samples")
    samples = []
    for k in range(len(t)):
        c, e = float(clearance[k]), float(head_error[k])
        if not (math.isfinite(c) and math.isfinite(e) and e >= 0):
            raise ValueError(f"sample {k}: clearance and head error must be finite, head error >= 0")
        samples.append({"time_ns": int(round(float(t[k]) * 1e9)), "clearance_m": c, "head_error_deg": e,
                        "collision": bool(collision[k]), "joint_limit_violation": bool(joint_violation[k]),
                        "tracking_valid": bool(tracking_valid[k])})
    return {
        "schema_version": 1,
        "scene_id": scene_id,
        "model_sha256": model_sha256,
        "trajectory_sha256": trajectory_sha256,
        "simulator_version": simulator_version,
        "mode": mode,
        "completed": True,
        "sample_count": len(samples),
        "duration_ns": samples[-1]["time_ns"] - samples[0]["time_ns"],
        "min_clearance_m": min(s["clearance_m"] for s in samples),
        "max_head_error_deg": max(s["head_error_deg"] for s in samples),
        "collision_count": sum(s["collision"] for s in samples),
        "joint_limit_violations": sum(s["joint_limit_violation"] for s in samples),
        "tracking_lost_count": sum(not s["tracking_valid"] for s in samples),
        "samples": samples,
    }


def replay_segments(n: int, k_acq: int | None, k_lock: int | None) -> list[tuple[str, int, int, str]]:
    """The replay reports of a run of n steps, as consecutive [start, stop) slices that cover every step:
    "search" (dance) until the first ACQUIRE, "acquire" (dance) until the first LOCK, "lock" (head-follow) to
    the end. A slice of fewer than two samples (the schema needs two) joins its neighbour."""
    k_acq = n if k_acq is None else k_acq
    k_lock = n if k_lock is None else max(k_lock, k_acq)
    segs = [(name, a, b, mode) for name, a, b, mode in (("search", 0, k_acq, "dance"),
                                                         ("acquire", k_acq, k_lock, "dance"),
                                                         ("lock", k_lock, n, "head-follow")) if b > a]
    out: list[tuple[str, int, int, str]] = []
    for seg in segs:
        if seg[2] - seg[1] < 2 and out:
            name, a, _, mode = out[-1]
            out[-1] = (name, a, seg[2], mode)
        else:
            out.append(seg)
    if len(out) > 1 and out[0][2] - out[0][1] < 2:
        name, _, b, mode = out[1]
        out[1:2] = [(name, out[0][1], b, mode)]
        out.pop(0)
    if sum(b - a for _, a, b, _ in out) != n:
        raise AssertionError("replay segments must cover every step")
    return out


def replay_args(report: dict, report_path: str, *, gap_ns: int) -> list[str]:
    """The replay CLI arguments that judge `report` with this module's thresholds."""
    mode = report["mode"]
    return ["--report", report_path, "--model-sha256", report["model_sha256"],
            "--trajectory-sha256", report["trajectory_sha256"],
            "--min-clearance", repr(REPLAY_MIN_CLEARANCE_M), "--max-head-error", repr(REPLAY_MAX_HEAD_ERROR_DEG),
            "--expected-mode", mode, "--expected-duration-ns", str(report["duration_ns"]),
            "--max-sample-gap-ns", str(gap_ns)]


def load_replay_validator():
    """The team's replay validator (branch codexfranklin/unified-app), loaded like show's team modules."""
    return S._load_team_module("simulation.replay", "codexfranklin/unified-app", "simulation/replay.py")


# ------------------------------------------------------------------ the model manifest
def model_manifest(kin, idle_source: str, calibration: Path | None) -> dict:
    """What the run read: sha256 of each vendor file (by its name inside the robot package, never its
    content or where it lives on this machine), of the calibration file and of the twin's own sources."""
    robot = Path(kin.robot_dir)
    names = ["robot.urdf", "joint_mapping.yaml", "safety.yaml", "simulation.yaml"]
    if "vendor" in idle_source:
        names.append("animations/factory_v1/idle.csv")
    try:
        meshes = sorted(set(re.findall(r'filename="([^"]+)"', (robot / "robot.urdf").read_text())))
    except OSError:
        meshes = []
    files = []
    for name in names + meshes:
        p = robot / name
        files.append({"path": name, "sha256": _sha256_file(p) if p.is_file() else "missing"})
    cal = None
    if calibration is not None and Path(calibration).is_file():
        cal = {"name": Path(calibration).name, "sha256": _sha256_file(Path(calibration))}
    twin_files = [{"path": f"twin/{name}.py", "sha256": _sha256_file(ROOT / "twin" / f"{name}.py")}
                  for name in RUN_MODULES]
    return {"what": "sha256 of every model file this run read; vendor files by name inside the robot package",
            "robot_package_files": files, "servo_calibration": cal, "twin_sources": twin_files,
            "joint_scale_source": kin.scale_source, "camera_source": getattr(kin, "camera_source", ""),
            "idle_source": idle_source, "flash_limiter": S.load_flash()[1]}


# ------------------------------------------------------------------ where every modelled number comes from
def constant_sources(kin, motion, cam, cfg: TrackerConfig, light: S.SdkLightParams, best: S.BestSdkDesign,
                     rate_limit: int, dt: float) -> list[dict]:
    """The value and the source of every constant the run used (the modules' comments, restated)."""
    out = []

    def add(name, value, source):
        out.append({"name": name, "value": S.json_safe(value), "source": source})

    add("sim_step_s", dt, "twin choice: the SDK control rate, 30 Hz (vendor source config/default.yaml:104-105)")
    # kinematics and safety envelope (twin/model.py)
    add("camera_fov_deg", [kin.hfov_deg, kin.vfov_deg], "measured on the lamp 2026-09-19 (two picture shifts at "
        "neutral, through the calibrated joint scales; twin/model.py)")
    add("joint_scale", kin.scale_source, "read at run time from FTM_CALIBRATION; reproduces both lamp shifts within 5%")
    add("table_plane_z_m", kin.table_z, "vendor robot description at run time: lowest vertex of the base plate")
    add("clearance_margins_m", {"table": kin.table_margin, "base": kin.base_margin, "self": kin.self_margin},
        "ASSUMPTION (workflow brief)")
    add("client_joint_envelope_units", kin.limits["base_yaw"], "ASSUMPTION: 6 units inside -100..100 (lamp/spatial.py)")
    # SDK motion (twin/motion.py)
    add("move_min_plan_s", M.plan_seconds(0.0), "vendor source control/safe_motion.py:115-117 (1.0 s baseline / 0.5)")
    add("waypoint_rate_hz", motion.fps, "vendor source config/default.yaml:104-105; control/runtime.py:37")
    add("max_velocity_units_s", motion.max_velocity, "vendor source robot safety.yaml:10; control/safety_filter.py:16")
    add("speed_fraction", motion.speed_fraction, "vendor source control/safe_motion.py:15")
    add("settle_tolerances_units", motion.tolerances,
        "vendor source robot safety.yaml:11-16; control/runtime.py:312-329")
    add("settle_timeout_s", motion.settle_timeout_s, "vendor source control/runtime.py:39")
    add("settle_poll_s", motion.settle_poll_s, "vendor source control/runtime.py:40")
    add("motion_capacity", motion.capacity, "vendor source sdk_gateway/service.py:392")
    add("sdk_rate_limit_per_min", rate_limit, "vendor source config/default.yaml:451, sdk_gateway/policy.py:166-187 "
        "(default 30); 120 is in the lamp's .env but not in effect until a runtime restart (research report "
        "pi-internet-deps.md:231)")
    add("http_s", motion.http_s, "research report pi-control-surface.md:201 (measured on the lamp)")
    add("pose_read_s", motion.pose_read_s, "research report pi-control-surface.md:201 (measured on the lamp)")
    add("compute_per_row_s", motion.compute_row_s, f"vendor collision check + CSV parse timed on this Mac "
        f"2026-09-19 ({M.MAC_ROW_S * 1e6:.0f} us/row) x ASSUMPTION Pi 5 slowdown {M.PI_SLOWDOWN:g}; every "
        "validation pass (upload, admission, re-plan) covers every row: vendor source control/runtime.py:143-163, "
        ":215; sdk_gateway/research.py:61-70, :107-117")
    add("compute_base_s", motion.compute_base_s, "ASSUMPTION (a pass's fixed cost)")
    add("move_start_delay_s", M.move_start_delay_s(), "motion.move POST -> first waypoint with free locks "
        "(vendor source control/runtime.py:149-215, :431-468; compute as above)")
    add("lock_clip_start_delay_s", cfg.clip_start_s, "clip upload -> first waypoint of a 2 s lock clip, free locks "
        "(upload, clip.play admission, re-plan; compute as above)")
    add("clip_upload", {"s_per_byte": motion.upload_s_per_byte, "fsync_s": motion.fsync_s,
                        "delete_s": motion.delete_s}, "body: research report pi-vision-readiness.md:158 (63 KB in "
        "4.4 ms); fsync and delete ASSUMPTION; locks: vendor source sdk_gateway/research.py:61-70, clip_store.py")
    add("clip_store_limits", {"files": motion.store_max_files, "bytes": motion.store_max_bytes},
        "vendor source sdk_gateway/clip_store.py:26-32 (persistent across sessions)")
    add("idle", motion.idle_source, "vendor data animations/factory_v1/idle.csv read at run time (else synthetic, "
        "ASSUMPTION); resumes only after a success (vendor source control/runtime.py:255-260)")
    add("idle_blend_s", motion.idle_blend_s, "vendor source config/default.yaml:113")
    add("servo_tau_s", motion.servo_tau_s, "ASSUMPTION (first-order lag; vendor simulator is bang-bang)")
    add("servo_max_velocity_units_s", motion.servo_max_velocity, "vendor source robot simulation.yaml:8")
    add("measurement_quantum_units", motion.quantum, "vendor source robot simulation.yaml:13")
    add("gravity_sag", getattr(motion._sag_fn, "describe", lambda: motion.sag)(), "ASSUMPTION: magnitudes at "
        "the reference lock pose (not measured; the vendor settle tolerances 'measured on the loaded arm', robot "
        "safety.yaml:12-16, bound them); pose dependence and direction from the vendor URDF's link masses at run "
        "time (twin/model.py GravitySag); swept by --sag-sweep")
    add("sdk_collision_rule", "head capsule vs base cylinder only", "vendor source kinematics/self_collision.py:31-60 "
        "(numbers read at run time from safety.yaml); the table is the client's job")
    # camera and detector (twin/perception.py)
    add("detector_fps", cam.fps, "ASSUMPTION: about half of the 19 fps live tracking measured on the lamp (AGENTS.md)")
    add("detector_timing_s", {"queue_age": cam.queue_age_s, "encode": cam.encode_s, "cache_wait": cam.cache_wait_s,
                               "client": cam.client_s, "clock_bias_this_run": round(cam.clock_bias_s, 4),
                               "clock_jitter": cam.clock_jitter_s,
                               "mean_exposure_to_delivery": round(cam.latency_s, 4)},
        "built from parts (twin/perception.py): queue age ASSUMPTION on research report pi-vision-readiness.md:161; "
        "encode from its measured inter-frame time; GET/decode/detect measured; the lamp stamps a frame after "
        "read + encode (vendor source perception/usb_camera.py:57-74); clock mapping ASSUMPTION; the "
        "end-to-end blink test has not been run")
    add("tracker_stamp_lag_s", cfg.stamp_lag_s, "ASSUMPTION: the client does not know the queue age (0)")
    add("tracker_reached_factor", cfg.reached_factor, "ASSUMPTION: a failed settle with every joint within this x "
        "its tolerance counts as reached (the SDK's failure carries position_errors and position_tolerances: vendor "
        "source control/runtime.py:43-67)")
    add("detector_noise", {"centre": cam.noise_frac, "size": cam.size_noise_frac}, "ASSUMPTION")
    add("detector_miss_rate", cam.miss_rate, "ASSUMPTION")
    add("detector_max_range_m", cam.max_range_m, "ASSUMPTION (research report pi-vision-readiness.md:242)")
    add("detector_profile_limit_deg", cam.max_face_yaw_deg, "ASSUMPTION (research report head-tracking.md:205)")
    add("motion_blur_deg_s", [cam.blur_start_deg_s, cam.blur_stop_deg_s], "ASSUMPTION built on research")
    # tracker (twin/tracking.py)
    add("tracker_min_confidence_and_debounce", [cfg.min_confidence, cfg.debounce_n, cfg.debounce_m],
        "research report head-tracking.md:250 (0.7, 3 detections); 3 of 5 frames ASSUMPTION")
    add("tracker_face_width_m", cfg.face_width_m, "lamp/follow.py:39 on branch karancodex/lamp-tracking "
        "(perception draws 0.18 m faces: depth under-estimated ~17%)")
    add("tracker_deadband_deg", cfg.deadband_deg,
        "ASSUMPTION inside fork sdk-spec safe-motion.md section 9 (4-5 units)")
    add("tracker_lock_enter_exit_deg", [cfg.lock_enter_deg, cfg.lock_exit_deg], "ASSUMPTION")
    add("tracker_settle_predict_s", cfg.settle_predict_s, "one 2.0 s plan + move_start_delay_s (client-side "
        "replica of the SDK cost model)")
    add("tracker_max_step", [cfg.max_step_units, cfg.max_step_deg], "lamp/follow.py --max-step 20 (branch "
        "karancodex/lamp-tracking); 15 deg ASSUMPTION")
    add("tracker_motion_calls_per_min", cfg.max_commands_per_min, f"run.py: min({MOTION_CAP_MAX}, rate limit - "
        f"{LIGHT_RESERVE_PER_MIN}); 26 at 30/min from research report twin-spec/motion.md section 8")
    add("tracker_backoff_s", [cfg.backoff_s, cfg.rate_limited_backoff_s], "fork sdk-spec safe-motion.md section 9")
    add("tracker_lost_s", [cfg.lost_after_s, cfg.lost_timeout_s], "ASSUMPTION / research report head-tracking.md 4.5")
    add("search", {"via": cfg.search_via, "dwell_s": cfg.search_dwell_s, "peak_speed_u_s": cfg.search_peak_speed_u_s,
                   "clip_s": cfg.search_clip_s}, "ASSUMPTION (motion spec section 14 asks for dwell periods)")
    # world (twin/world.py)
    add("people", {"seated_head_z_m": W.SEATED_HEAD_Z, "standing_head_z_m": W.STANDING_HEAD_Z,
                   "walk_m_s": W.WALK_SPEED_M_S}, "ASSUMPTION (workflow brief room model, typical adults)")
    # light (twin/show.py)
    add("light_transition_s", light.transition_s, "vendor source config/default.yaml:63; live lamp config 600 "
        "(research report pi-control-surface.md:156)")
    add("light_hardware_cap", light.hardware_cap, "vendor source config/profiles/lights/raspberry_pi_neopixel.yaml:29")
    add("light_admission_s", [*light.admission_s, light.admission_tail_s], "ASSUMPTION built on Mac->lamp ping "
        "(research report pi-network-remote-access.md:71-72)")
    add("light_effect_first_frame_s", light.effect_first_frame_s, "ASSUMPTION built on vendor code paths")
    add("light_effect_rate_hz", S.EFFECT_HZ, "vendor source rendering/effects/player.py:48")
    add("light_start_brightness", S.START_BRIGHTNESS,
        "measured on the lamp (research report pi-control-surface.md:158)")
    add("light_best_design", {"drive_luminance": best.drive_luminance, "pedestal_px": best.pedestal_px,
                              "admission_trim_s": round(best.trim_s(light), 4), "hold_s": best.hold_s,
                              "effect": best.effect_name, "grid": "cue sheet (what-if)" if best.use_cue_sheet
                              else "live: predicted from event arrivals"},
        "ASSUMPTION (design choices in research report twin-spec/light.md section 6.3); the trim is the ASSUMED "
        "admission latency's 95th percentile + the effect's first frame, so the light leads, never lags")
    add("event_lead_s", [S.LEAD_MEDIAN_S, S.LEAD_MIN_S], "measured on the Mac conductor run log "
        "2026-09-19T091049Z-wifi-music (median, min); the per-event shape is ASSUMPTION")
    add("room_latency_budget_s", S.LATENCY_S, "Mac conductor source Show/Show.swift (default latencyMs 300)")
    add("haptic_defaults", S.HAPTIC_DEFAULTS, "Mac conductor source Show/HapticAnalyzer.swift:33-38")
    add("phone_haptic_output_latency_s", 0.0, "ASSUMPTION: not measured (spread +-2 ms from acoustic clicks, "
        "Mac repo docs/field-notes-2026-09-15.md:23-24)")
    add("titan", {"reported_s": 0.1377, "trim_s": 0.100, "jitter_s": 0.001}, "reported by Core Audio in the "
        "2026-09-19 run log (not a board measurement); trim is that run's setting; real Bluetooth delay "
        "NOT measured (assumed = trim); jitter from a speaker stand-in (Mac repo AGENTS.md:79)")
    add("sync_window_ms", [S.SYNC_WINDOW_S * 1e3, S.SYNC_AIM_S * 1e3], "research report deaf-hoh-design.md:128-138")
    add("flash_governor", [S.GOVERNOR_FLASHES_PER_S, S.GOVERNOR_EDGE_SPACING_S], "research report "
        "deaf-hoh-design.md:149-166 (stricter than WCAG 2.3.1's 3/s)")
    add("replay_thresholds", {"min_clearance_m": REPLAY_MIN_CLEARANCE_M,
                              "max_head_error_deg": REPLAY_MAX_HEAD_ERROR_DEG},
        "run.py: the twin's smallest check margin (ASSUMPTION); half the measured vertical FOV")
    return out


# ------------------------------------------------------------------ one run
@dataclass
class RunResult:
    """Everything one run produced; the video is drawn from this."""
    report: dict
    out_dir: Path | None
    kin: object
    world: W.Scenario
    song: S.Song
    show: S.ShowResult
    steps: dict                      # per-step arrays (see simulate)
    detections: list                 # every Detection delivered, in delivery order
    commands: list                   # every tracker command and what the SDK said
    light_calls: list                # the recommended design's light.glow calls (after the run)
    felt: dict = field(default_factory=dict)    # lane name -> (times, kinds) of felt haptics
    paths: dict = field(default_factory=dict)


def simulate(scenario: str, strategy: str = "settle", song: str = "synthetic", seconds: float = DEFAULT_SECONDS,
             seed: int = 0, out_dir: str | Path | None = None, video: bool = True, *,
             rate_limit: int = M.RATE_LIMIT_PER_MIN, dt: float = DT, video_fps: int = VIDEO_FPS, kin=None,
             idle="auto", tracker_config: dict | None = None, sag="assumed", store_initial_files: int = 0,
             delete_clips: bool = True) -> RunResult:
    """One run of the twin. Deterministic for a given set of arguments.

    kin: a LampTwin to reuse (built from FTM_ROBOT_DIR / FTM_CALIBRATION when None).
    rate_limit: the SDK session's actions/min, shared by motion and light (vendor default 30).
    tracker_config: TrackerConfig overrides (the rate budget fields are set from rate_limit).
    sag: a level name of motion.SAG_LEVELS (magnitudes at the reference lock pose, scaled per pose by the
        gravity torque of the URDF masses: model.GravitySag) or a dict of such magnitudes.
    store_initial_files: clips already in the lamp's persistent clip store from earlier sessions.
    delete_clips: the client deletes each clip when its action ends (False: the what-if that fills the store)."""
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    from twin.model import GravitySag, LampTwin         # mujoco only when a run is made

    wall0 = time.perf_counter()
    kin = kin if kin is not None else LampTwin()
    world = W.scenario(scenario, seed=seed)
    tune = load_song(song, seconds, seed)
    motion_cap = motion_cap_for(rate_limit)
    sag_name = sag if isinstance(sag, str) else "custom"
    sag_model = GravitySag(kin, M.SAG_LEVELS[sag] if isinstance(sag, str) else dict(sag or {}))

    # One SDK session: motion and light take slots from the same sliding window.
    session = M.RateWindow(per_minute=rate_limit)
    motion = M.SDKMotionModel(kin.neutral, rate_window=session, pose_ok=kin.sdk_head_base_clear, idle=idle,
                              sag=sag_model, store_initial_files=store_initial_files,
                              store_initial_bytes=store_initial_files * EARLIER_CLIP_BYTES)
    cam = PER.FaceDetectorModel(seed=seed)
    cfg = TrackerConfig(**{**(tracker_config or {}), "seed": seed, "sdk_rate_limit_per_min": rate_limit,
                           "max_commands_per_min": motion_cap})
    tracker = HeadTracker(kin, strategy=strategy, config=cfg)

    # The recommended light design, planned from event ARRIVALS as the live conductor allows (no cue sheet),
    # with the light budget the session leaves after head tracking. Its POSTs take their session slots in
    # the loop.
    light_params = S.SdkLightParams(rate_limit_per_min=rate_limit, seed=seed)
    best = S.BestSdkDesign(motion_per_min=motion_cap)
    planned_light, _ = best.calls(tune, S.SdkLamp(light_params), S.motion_slots_for(tune, motion_cap))
    light_sends = sorted(c.t_send for c in planned_light)
    light_slots = {"taken": 0, "refused_429": 0}

    n_steps = int(math.floor(seconds / dt + 1e-9)) + 1
    t_arr = np.arange(n_steps) * dt
    measured = np.zeros((n_steps, len(JOINTS)))
    commanded = np.zeros((n_steps, len(JOINTS)))
    rec = {k: [] for k in ("state", "pid", "track", "why", "blocked", "visible", "people")}
    err_locked = np.full(n_steps, np.nan)
    err_nearest = np.full(n_steps, np.nan)
    present = np.zeros(n_steps, dtype=int)
    in_view_locked = np.zeros(n_steps, dtype=bool)
    clearance = np.zeros((n_steps, 3))                 # table, base, self (signed metres)
    check_ok = np.zeros(n_steps, dtype=bool)
    outside_envelope = np.zeros(n_steps, dtype=bool)
    idle_playing = np.zeros(n_steps, dtype=bool)
    busy_arr = np.zeros(n_steps, dtype=bool)
    session_used = np.zeros(n_steps, dtype=int)
    aim_track = np.full((n_steps, 3), np.nan)
    target_track = np.full((n_steps, 3), np.nan)
    reasons: dict[str, int] = {}
    detections, commands, ours, outcomes = [], [], {}, []
    clip_deletes = []                                  # (t, clip id): the adapter's clips.delete calls
    li = 0

    def delete(t_now: float, clip_id: str) -> None:
        if delete_clips and clip_id:
            clip_deletes.append((round(t_now, 4), clip_id, round(motion.delete_clip(t_now, clip_id), 4)))

    for k in range(n_steps):
        t = float(t_arr[k])
        # 1. light.glow POSTs due by now take their slot on the shared session
        while li < len(light_sends) and light_sends[li] <= t + 1e-9:
            light_slots["taken" if session.take(light_sends[li]) else "refused_429"] += 1
            li += 1
        # 2. the arm, then where the camera is and what it sees
        st = motion.step(t)
        for aid, outcome in st.finished:
            if aid in ours:
                ours[aid]["outcome"] = outcome
                a = motion.actions[aid]
                details = None
                if outcome == "failed" and a.target is not None and a.settle_actual is not None:
                    # what the SDK's failure carries (vendor source control/runtime.py:43-67)
                    details = {"target_positions": a.target, "actual_positions": a.settle_actual,
                               "position_errors": a.settle_errors, "position_tolerances": motion.tolerances}
                outcomes.append((outcome, details))
                delete(t, a.clip_id)                    # the clip ended: its stored file goes
        m = st.measured
        pose = kin.head(m)
        people = world.people_at(t)
        cam.observe(t, pose, people)
        delivered = cam.poll(t)
        detections.extend(delivered)
        busy = st.active_action is not None and st.active_action in ours
        # 3. the tracker decides; its command is POSTed at once (a clip is uploaded first)
        outcome, details = outcomes.pop(0) if outcomes else (None, None)
        for c in tracker.update(t, delivered, m, busy, outcome, details):
            r = motion.submit_move(t, c.target) if c.kind == "move" else motion.submit_clip(t, c.frames)
            if r.t_posted is not None:
                tracker.posted(t, r.t_posted)           # clip.play went out after its upload
            row = {"t": round(t, 4), "kind": c.kind, "why": c.why, "accepted": r.accepted, "reason": r.reason,
                   "action_id": r.action_id, "planned_s": round(r.planned_duration_s, 3), "outcome": None,
                   "frames": len(c.frames) if c.frames else 0, "clip_id": r.clip_id,
                   "t_posted": None if r.t_posted is None else round(r.t_posted, 4)}
            commands.append(row)
            if r.accepted:
                ours[r.action_id] = row
            else:
                row["outcome"] = "refused"
                outcomes.append(("rejected " + r.reason, None))   # the tracker learns a refused POST like an outcome
                delete(t, r.clip_id)                    # an upload whose clip.play was refused stays on the lamp
        tr = tracker.trace
        # 4. ground truth, for scoring only
        pid = (tr.get("scoring_person_id") or "") if tr["state"] in (ACQUIRE, LOCK, LOST) else ""
        vis, near = [], math.inf
        for p in people:
            to = np.asarray(p.head, float) - np.asarray(pose.position, float)
            e = _angle_deg(np.asarray(pose.forward, float), to)
            near = min(near, e)
            if _in_picture(_picture_point(pose, p.head, kin.fx, kin.fy)):
                vis.append(p.id)
            if p.id == pid:
                err_locked[k] = e
        err_nearest[k] = near if people else np.nan
        present[k] = len(people)
        in_view_locked[k] = pid != "" and pid in vis
        # 5. the pose the arm is really in: clearance and the client envelope
        chk = kin.check(m)
        clearance[k] = (chk.min_table_m, chk.min_base_m, chk.min_self_m)
        check_ok[k] = chk.ok
        for why in chk.reasons:                        # counted by kind: numbers replaced by '#'
            key = re.sub(r"[-+]?\d+(\.\d+)?", "#", why)[:80]
            reasons[key] = reasons.get(key, 0) + 1
        outside_envelope[k] = any(not kin.limits[j][0] <= m[j] <= kin.limits[j][1] for j in JOINTS)
        measured[k] = [m[j] for j in JOINTS]
        commanded[k] = [st.commanded[j] for j in JOINTS]
        idle_playing[k] = st.idle_playing
        busy_arr[k] = busy
        session_used[k] = session.used(t)
        if tr.get("aim_m") is not None:
            aim_track[k] = tr["aim_m"]
        if tr.get("target_m") is not None:
            target_track[k] = tr["target_m"]
        rec["state"].append(tr["state"])
        rec["pid"].append(pid)
        rec["track"].append(tr.get("locked_id") or "")
        rec["why"].append(tr.get("why", ""))
        rec["blocked"].append(tr.get("blocked", ""))
        rec["visible"].append(tuple(vis))
        rec["people"].append([(p.id, [round(float(v), 5) for v in p.head], [round(float(v), 5) for v in p.facing])
                              for p in people])
    loop_s = time.perf_counter() - wall0

    # ---- light and haptics, with the motion POST times that really happened
    # the SDK takes an action's rate slot when its POST goes out: a clip.play after its upload
    post_times = [c["t_posted"] if c.get("t_posted") is not None else c["t"] for c in commands]
    motion_posts = np.array(post_times, dtype=float)
    show = S.run_show(tune, sdk=light_params, best=best, motion_times=motion_posts, seed=seed, t0=0.0,
                      t1=float(t_arr[-1]))
    final_sends = sorted(c.t_send for c in show.best_run.calls)
    lanes = [S.PhoneLane(), S.TitanLane.from_song(tune)]
    rng = np.random.default_rng(seed)
    felt = {lane.name: (lane.felt(tune.events, rng), [e.kind for e in tune.events]) for lane in lanes}

    # ---- scores
    joint_speed = np.abs(np.diff(measured, axis=0)) / dt if n_steps > 1 else np.zeros((1, len(JOINTS)))
    fwd = np.array([kin.head({j: v for j, v in zip(JOINTS, row, strict=True)}).forward for row in measured])
    cam_speed = np.degrees(np.arccos(np.clip(np.sum(fwd[1:] * fwd[:-1], axis=1), -1.0, 1.0))) / dt
    tracking = tracking_metrics(t_arr, rec["state"], rec["pid"], err_locked, rec["visible"], present, dt)
    minutes = seconds / 60.0
    refused = {}
    for c in commands:
        if not c["accepted"]:
            key = c["reason"].split(":")[0]
            refused[key] = refused.get(key, 0) + 1
    outcome_counts = {}
    for c in commands:
        if c["accepted"]:
            key = motion.actions[c["action_id"]].outcome or "running at end"
            c["outcome"] = key
            outcome_counts[key] = outcome_counts.get(key, 0) + 1
    violations = int(np.sum(~check_ok))
    geo = clearance.min(axis=0)
    ev = show.evidence

    def hits(variant: str, share: str) -> dict:
        """Per lane: how many of the KICK/DROP hits felt in that lane have a light onset meeting `share`
        (share_in_sync: 0-30 ms BEFORE the felt time, the spec's lead-only rule; share_within_window: +-30 ms)."""
        out = {}
        for lane, st in ev[variant]["sync"]["lanes"].items():
            out[lane] = int(round(st.get(share, 0.0) * st.get("n", 0))) if st.get("events") else None
        return out

    light_summary = {}
    for variant in ("ideal", "sdk_naive", "sdk_best", "sdk_best_known_track"):
        if not ev.get(variant):
            continue
        sync = ev[variant]["sync"]
        light_summary[variant] = {
            "hits_scored": sync["events"],
            "hits_with_light": sync["events"] - sync["light_missed"],
            "hits_in_sync_lead_only": hits(variant, "share_in_sync"),
            "hits_within_30ms_either_side": hits(variant, "share_within_window"),
            "median_offset_ms": {k: v.get("median_ms") for k, v in sync["lanes"].items()},
            "p95_abs_offset_ms": {k: v.get("p95_abs_ms") for k, v in sync["lanes"].items()},
            "flashes_per_s": sync["flash"].get("general_flashes_per_s"),
            "governor_ok": sync["flash"].get("governor_ok"),
            "wcag_ok": sync["flash"].get("wcag_ok"),
            "colour_switches_per_min": sync["colour"]["per_minute"],
            "colour_switches_max_in_any_60s": sync["colour"]["max_in_any_60s"],
            "colour_steady_state": sync["colour"]["steady_state"],
        }
        if variant != "ideal":
            lamp = ev[variant]["lamp"]
            light_summary[variant] |= {"light_calls": lamp["calls"], "status": lamp["status"],
                                       "light_calls_max_in_60s": lamp["light_calls_max_in_60s"],
                                       "true_visible_lag_ms": lamp["visible_lag_ms"],
                                       "motion_moves_refused_429_on_shared_session": lamp["motion_moves_refused_429"]}

    scene_id = f"twin:{scenario}:{strategy}:seed{seed}:{song_label(song)}"
    report = {
        "honesty": "Simulation evidence only, not hardware approval. The robot is not driven; every SDK outcome "
                   "here is modelled. Numbers marked ASSUMPTION in 'sources' are not measured.",
        "run": {"scenario": scenario, "scenario_description": world.description, "strategy": strategy,
                "song": song_label(song),
                "song_origin": tune.origin, "seconds": seconds, "seed": seed, "step_s": dt, "steps": n_steps,
                "rate_limit_per_min": rate_limit, "tracker_motion_cap_per_min": motion_cap,
                "light_budget_per_min": ev["sdk_best"]["client"].get("budget_per_min"),
                "steady_state": bool(seconds >= STEADY_STATE_S), "sag": sag_name,
                "clip_store_initial_files": store_initial_files, "delete_clips": delete_clips,
                "idle": motion.idle_source, "loop_wall_s": round(loop_s, 2)},
        "tracking": tracking,
        "sdk": {
            "motion_posts": len(commands),
            "motion_posts_per_min": round(len(commands) / minutes, 2),
            "motion_posts_max_in_any_60s": _max_in_60s(post_times),
            "tracker_blocked_by_budget_s": round(tracker.stats.get("steps_blocked_budget", 0) * dt, 3),
            "motion_accepted": int(sum(c["accepted"] for c in commands)),
            "motion_refused": refused,
            "motion_outcomes": outcome_counts,
            "light_posts": len(light_sends),
            "light_posts_per_min": round(len(light_sends) / minutes, 2),
            "light_slots_in_loop": light_slots,
            "light_plan_unchanged_by_real_motion_times": final_sends == light_sends,
            "session_actions_per_min": round((len(commands) + len(light_sends)) / minutes, 2),
            "session_window_max_used": int(session_used.max()),
            "motion_model_counters": motion.stats(float(t_arr[-1])),
            "commands_by_kind": {k: sum(1 for c in commands if c["kind"] == k) for k in ("move", "clip")},
            "clip_store": {"uploads": motion.counters["uploads"], "refused_store_full": motion.counters["store_full"],
                           "deletes": len(clip_deletes), "files_at_start": store_initial_files,
                           "files_at_end": len(motion.store), "peak_files": motion.store_peak,
                           "max_files": motion.store_max_files,
                           "uploads_per_min": round(motion.counters["uploads"] / minutes, 2)},
            "clip_deletes": clip_deletes,
            "action_start_delay_s": _stats([motion.actions[c["action_id"]].t_plan_start - c["t"] for c in commands
                                            if c["accepted"] and motion.actions[c["action_id"]].t_plan_start
                                            is not None]),
        },
        "motion": {
            "max_joint_speed_units_s": {j: round(float(joint_speed[:, i].max()), 2) for i, j in enumerate(JOINTS)},
            "max_joint_speed_units_s_any": round(float(joint_speed.max()), 2),
            "t_at_max_joint_speed_s": round(float(t_arr[1 + int(np.argmax(joint_speed.max(axis=1)))]), 3)
            if n_steps > 1 else 0.0,
            "p99_joint_speed_units_s_any": round(float(np.quantile(joint_speed.max(axis=1), 0.99)), 2),
            "why_peaks": "a re-plan from the measured (sagged) pose steps the servo setpoint by the sag; the "
                         "first-order servo (ASSUMPTION, no acceleration limit) answers at up to sag/tau",
            "max_camera_turn_deg_s": round(float(cam_speed.max()), 2) if cam_speed.size else 0.0,
            "vendor_velocity_ceiling_units_s": motion.max_velocity,
            "idle_share": round(float(np.mean(idle_playing)), 4),
            "sag": sag_model.describe() | {"level": sag_name,
                                          "reference_torque_nm": {j: round(v, 4) for j, v in
                                                                  sag_model.reference_torque_nm.items()},
                                          "deflection_at_reference_units": {j: round(v, 3) for j, v in
                                                                            sag_model.at(sag_model.reference).items()
                                                                            if abs(v) > 1e-9}},
        },
        "safety": {
            "min_table_m": round(float(geo[0]), 5), "min_base_m": round(float(geo[1]), 5),
            "min_self_m": round(float(geo[2]), 5),
            "t_at_min_s": {name: round(float(t_arr[int(np.argmin(clearance[:, i]))]), 3)
                           for i, name in enumerate(("table", "base", "self"))},
            "margins_m": {"table": kin.table_margin, "base": kin.base_margin, "self": kin.self_margin},
            "steps_failing_check": violations,
            "steps_failing_check_during_idle": int(np.sum(~check_ok & idle_playing)),
            "check_reasons": reasons,
            "steps_outside_client_envelope": int(np.sum(outside_envelope)),
            "steps_outside_calibrated_range": int(np.sum(np.abs(measured) > 100.0 + 1e-9)),
            "steps_in_contact": int(np.sum(clearance.min(axis=1) < 0.0)),
            "what": "check() of the MEASURED pose every step: joint envelope (6 units inside the calibrated range) "
                    "and signed mesh distances to the table plane, the lamp's base and between links",
        },
        "light_haptics_summary": light_summary,
        "light_haptics": ev,
        "tracker_stats": {k: v for k, v in tracker.stats.items() if v},
        "perception_stats": cam.stats,
        "perception_timing": {"stamp_minus_exposure_ms": _stats(1e3 * np.asarray(cam.stamp_lags)),
                              "exposure_to_delivery_ms": _stats([1e3 * (d.t_delivered - d.t_capture)
                                                                 for d in detections]),
                              "tracker_stamp_lag_s": cfg.stamp_lag_s,
                              "what": "the tracker sees only t_stamp (the lamp stamps a frame after reading and "
                                      "encoding it, then the client maps the stamp onto its clock); t_capture "
                                      "is used for scoring only"},
        "commands": commands,
    }
    report["sources"] = constant_sources(kin, motion, cam, cfg, light_params, best, rate_limit, dt)

    steps = {"t": t_arr, "measured": measured, "commanded": commanded, "state": rec["state"], "pid": rec["pid"],
             "track": rec["track"], "why": rec["why"], "visible": rec["visible"], "people": rec["people"],
             "err_locked": err_locked, "err_nearest": err_nearest, "present": present,
             "in_view_locked": in_view_locked, "clearance": clearance, "check_ok": check_ok,
             "idle_playing": idle_playing, "busy": busy_arr, "session_used": session_used, "aim": aim_track,
             "target": target_track}
    result = RunResult(report=report, out_dir=Path(out_dir) if out_dir else None, kin=kin, world=world, song=tune,
                       show=show, steps=steps, detections=detections, commands=commands,
                       light_calls=list(show.best_run.calls), felt=felt)

    if out_dir is not None:
        _write_outputs(result, scenario, strategy, seed, scene_id, motion.idle_source, dt)
        if video:
            from twin import video as V
            t_vid = time.perf_counter()
            path = result.out_dir / "video.mp4"
            info = V.render_video(result, path, fps=video_fps)
            result.paths["video"] = str(path)
            report["video"] = info | {"file": "video.mp4", "wall_s": round(time.perf_counter() - t_vid, 1)}
        report["run"]["total_wall_s"] = round(time.perf_counter() - wall0, 2)
        _write_json(result.out_dir / "report.json", report)
    return result


def _write_outputs(result: RunResult, scenario: str, strategy: str, seed: int, scene_id: str, idle_source: str,
                   dt: float) -> None:
    """trajectory.json, the model manifest and the two replay reports; their hashes go into report.json."""
    out = result.out_dir
    out.mkdir(parents=True, exist_ok=True)
    s, report = result.steps, result.report
    kin = result.kin

    calibration = Path(os.environ.get("FTM_CALIBRATION") or Path.home() / "lelamp-hackathon-2026/lelamp.json")
    manifest_bytes = _write_json(out / "replay" / "manifest.json", model_manifest(kin, idle_source, calibration))
    model_sha = _sha256_bytes(manifest_bytes)
    trajectory = {"scene_id": scene_id, "scenario": scenario, "seed": seed, "strategy": strategy,
                  "song": report["run"]["song"], "step_s": dt, "joints": list(JOINTS),
                  "t": [round(float(x), 6) for x in s["t"]],
                  "measured": np.round(s["measured"], 4), "commanded": np.round(s["commanded"], 4),
                  "people": s["people"], "commands": result.commands}
    traj_bytes = _write_json(out / "trajectory.json", trajectory, pretty=False)
    traj_sha = _sha256_bytes(traj_bytes)

    # per-sample replay fields
    t = s["t"]
    clear = s["clearance"].min(axis=1)
    collision = clear < 0.0                                   # meshes overlap or a link is in the table
    joint_violation = np.any(np.abs(s["measured"]) > 100.0 + 1e-9, axis=1)   # outside the calibrated range
    import mujoco
    version = f"{REPLAY_SIMULATOR}; mujoco {mujoco.__version__}; numpy {np.__version__}"
    gap_ns = int(math.ceil(dt * 1e9))
    state = s["state"]
    k_acq = next((k for k, x in enumerate(state) if x in (ACQUIRE, LOCK)), None)
    k_lock = next((k for k, x in enumerate(state) if x == LOCK), None)
    replay = {"model_sha256": model_sha, "trajectory_sha256": traj_sha, "max_sample_gap_ns": gap_ns,
              "thresholds": {"min_clearance_m": REPLAY_MIN_CLEARANCE_M,
                             "max_head_error_deg": REPLAY_MAX_HEAD_ERROR_DEG}}
    validator, how = load_replay_validator()
    replay["validator"] = how

    def emit(name: str, sl: slice, mode: str, head_error: np.ndarray, valid: np.ndarray) -> None:
        if (sl.stop - sl.start) < 2:
            replay[name] = {"written": False, "why": "fewer than two samples"}
            return
        rep = replay_report(t[sl], clear[sl], head_error[sl], collision[sl], joint_violation[sl], valid[sl],
                            mode=mode, scene_id=f"{scene_id}:{name}", model_sha256=model_sha,
                            trajectory_sha256=traj_sha, simulator_version=version)
        rel = f"replay/{name}.report.json"
        _write_json(out / rel, rep, pretty=False)
        entry = {"written": True, "file": rel, "mode": mode, "samples": rep["sample_count"],
                 "duration_ns": rep["duration_ns"], "min_clearance_m": rep["min_clearance_m"],
                 "max_head_error_deg": rep["max_head_error_deg"], "tracking_lost_count": rep["tracking_lost_count"],
                 "cli_args": replay_args(rep, rel, gap_ns=gap_ns)}
        if validator is not None:
            verdict = validator.validate_report(
                validator.read_report(out / rel), model_sha256=model_sha, trajectory_sha256=traj_sha,
                min_clearance=REPLAY_MIN_CLEARANCE_M, max_head_error=REPLAY_MAX_HEAD_ERROR_DEG,
                expected_mode=mode, expected_duration_ns=rep["duration_ns"], max_sample_gap_ns=gap_ns)
            entry["verdict_in_process"] = {"status": verdict["status"], "reasons": verdict["reasons"][:6]}
        replay[name] = entry

    # Head error: to the person the tracker is on, else the nearest person, else the worst possible (empty
    # room). Tracking is valid only while ACQUIRE/LOCK holds a real person whose head is in the picture.
    # Every step of the run is in exactly one report (replay_segments): the look-around, the turn to the face
    # and, from the first LOCK to the end, the lock with its losses.
    near = np.where(np.isfinite(s["err_nearest"]), s["err_nearest"], NO_PERSON_ERROR_DEG)
    err = np.where(np.isfinite(s["err_locked"]), s["err_locked"], near)
    valid = np.array([st in (ACQUIRE, LOCK) for st in state]) & s["in_view_locked"]
    segments = replay_segments(len(t), k_acq, k_lock)
    for name, a, b, mode in segments:
        emit(name, slice(a, b), mode, err, valid)
    joined = "under two samples: joined the report before or after it"
    why = {"search": joined if k_acq else "the first step already acquired",
           "acquire": joined if k_acq is not None else "never acquired a face",
           "lock": joined if k_lock is not None else "never locked: no head-follow evidence"}
    for name in ("search", "acquire", "lock"):
        if name not in replay:
            replay[name] = {"written": False, "why": why[name]}
    samples = sum(replay[name].get("samples", 0) for name, *_ in segments)
    if samples != report["run"]["steps"]:
        raise AssertionError(f"replay reports cover {samples} samples, the run has {report['run']['steps']}")
    replay["coverage"] = {"steps": report["run"]["steps"], "samples": samples,
                          "segments": [{"name": n, "start": a, "stop": b, "mode": m} for n, a, b, m in segments]}
    report["replay"] = replay
    result.paths.update({"report": str(out / "report.json"), "trajectory": str(out / "trajectory.json")})


# ------------------------------------------------------------------ the matrix and its table
MATRIX_SCENARIOS = ("walk_in_sit", "sway_to_music", "cross_room", "leave_return", "two_people", "lean_close", "empty")


def summary_row(report: dict) -> dict:
    """One line of the results table from a report.json."""
    r, tr, sdk, sf = report["run"], report["tracking"], report["sdk"], report["safety"]
    ls = report["light_haptics_summary"]
    lock_share = tr["share_of_lock_on_axis"]
    pres = tr["share_of_presence_on_axis"]
    replay = report.get("replay", {})
    verdict = replay.get("lock", {}).get("verdict_in_process", {}).get("status") \
        if replay.get("lock", {}).get("written") else "no lock"
    best = ls["sdk_best"]
    known = ls.get("sdk_best_known_track")
    in_sync = best["hits_in_sync_lead_only"].get("phone")

    def pct(x):
        return "-" if x is None else f"{100 * x:.0f}%"
    return {
        "run": f"{r['scenario']}-{r['strategy']}" + ("" if r["song"] == "synthetic" else f" ({r['song']})")
               + ("" if r["seconds"] == DEFAULT_SECONDS else f" {r['seconds']:g} s")
               + ("" if r["rate_limit_per_min"] == 30 else f" @{r['rate_limit_per_min']}/min")
               + ("" if r.get("sag", "assumed") == "assumed" else f" sag={r['sag']}")
               + ("" if r.get("delete_clips", True) else " no deletes")
               + ("" if r.get("steady_state", r["seconds"] >= STEADY_STATE_S) else " (not steady state)"),
        "first lock s": "-" if tr["t_first_lock_s"] is None else f"{tr['t_first_lock_s']:.1f}",
        "lock err mean/p95 deg": "-" if not tr["lock_error_deg"].get("n") else
        f"{tr['lock_error_deg']['mean']:.1f}/{tr['lock_error_deg']['p95']:.1f}",
        "in lock <=5/<=10 deg": f"{pct(lock_share['within_5_deg'])}/{pct(lock_share['within_10_deg'])}",
        "of presence <=5/<=10": f"{pct(pres['within_5_deg'])}/{pct(pres['within_10_deg'])}",
        "steals": tr["lock_steals"], "lost": tr["lost_episodes"],
        "motion/min": sdk["motion_posts_per_min"], "refused": sum(sdk["motion_refused"].values()),
        "blocked s": sdk["tracker_blocked_by_budget_s"],
        "max u/s": report["motion"]["max_joint_speed_units_s_any"],
        "min table/base/self cm": "/".join(f"{100 * sf[k]:.1f}" for k in ("min_table_m", "min_base_m", "min_self_m")),
        "check fails": sf["steps_failing_check"],
        "light hits in sync (live)": f"{in_sync if in_sync is not None else '-'}/{best['hits_scored']}",
        "colour sw/min ideal/live/known": "/".join(
            [str(ls["ideal"]["colour_switches_per_min"]), str(best["colour_switches_per_min"]),
             str(known["colour_switches_per_min"]) if known else "-"]),
        "replay lock": verdict,
    }


def summary_table(reports: list[dict], row=None) -> str:
    """A markdown table of summary_row (or `row`) for each report."""
    rows = [(row or summary_row)(r) for r in reports]
    if not rows:
        return ""
    head = list(rows[0])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(str(row[h]) for h in head) + " |" for row in rows]
    return "\n".join(lines)


def run_matrix(out_root: str | Path, *, strategies=("settle", "preempt", "clip"), seconds: float = DEFAULT_SECONDS,
               seed: int = 0, song: str = "synthetic", videos: dict | None = None,
               rate_limit: int = M.RATE_LIMIT_PER_MIN) -> list[dict]:
    """Every scenario x strategy into out_root/<scenario>-<strategy>/. videos: {(scenario, strategy): True}."""
    from twin.model import LampTwin
    kin = LampTwin()
    reports = []
    for name in MATRIX_SCENARIOS:
        for strategy in strategies:
            want_video = bool((videos or {}).get((name, strategy)))
            res = simulate(name, strategy, song, seconds, seed, Path(out_root) / f"{name}-{strategy}",
                           video=want_video, rate_limit=rate_limit, kin=kin)
            reports.append(res.report)
            print(f"{name}-{strategy}: {json.dumps(summary_row(res.report))}", flush=True)
    table = summary_table(reports)
    (Path(out_root) / "matrix.md").write_text(table + "\n")
    return reports


# ------------------------------------------------------------------ the gravity sag sweep
SAG_SWEEP_SCENARIOS = ("two_people", "walk_in_sit", "sway_to_music")


def sag_row(report: dict) -> dict:
    """One line of the sag sweep: lock quality against the sag at the reference pose."""
    r, tr, sdk = report["run"], report["tracking"], report["sdk"]
    stats = report.get("tracker_stats", {})
    at_ref = report["motion"]["sag"]["deflection_at_reference_units"]
    err = tr["lock_error_deg"]
    replay = report.get("replay", {}).get("lock", {})
    return {
        "sag": r["sag"],
        "at reference b/e/w units": "/".join(f"{at_ref.get(j, 0.0):+.1f}"
                                             for j in ("base_pitch", "elbow_pitch", "wrist_pitch")),
        "run": f"{r['scenario']}-{r['strategy']}",
        "first lock s": "-" if tr["t_first_lock_s"] is None else f"{tr['t_first_lock_s']:.1f}",
        "locked s": tr["locked_s"],
        "lock err mean/p95/max deg": "-" if not err.get("n") else
        f"{err['mean']:.1f}/{err['p95']:.1f}/{err['max']:.1f}",
        "of presence <=10 deg": "-" if tr["share_of_presence_on_axis"]["within_10_deg"] is None else
        f"{100 * tr['share_of_presence_on_axis']['within_10_deg']:.0f}%",
        "settle failed": sdk["motion_outcomes"].get("failed", 0),
        "failed but reached": stats.get("failed_but_reached", 0),
        "failures": stats.get("failures", 0),
        "HOLD": stats.get("hold", 0),
        "replay lock": replay.get("verdict_in_process", {}).get("status") if replay.get("written") else "no lock",
    }


def run_sag_sweep(out_root: str | Path, *, scenarios=SAG_SWEEP_SCENARIOS, strategies=("settle", "clip"),
                  levels=tuple(M.SAG_LEVELS), seconds: float = DEFAULT_SECONDS, seed: int = 0,
                  song: str = "synthetic", rate_limit: int = M.RATE_LIMIT_PER_MIN) -> list[dict]:
    """Lock quality against the gravity sag (the review's blocker: every PASS rested on one unmeasured
    number): each level of motion.SAG_LEVELS x scenario x strategy into out_root/sag-<level>/, and a table
    in out_root/sag_sweep.md."""
    from twin.model import LampTwin
    kin = LampTwin()
    reports = []
    for level in levels:
        for name in scenarios:
            for strategy in strategies:
                out = Path(out_root) / f"sag-{level}" / f"{name}-{strategy}"
                res = simulate(name, strategy, song, seconds, seed, out,
                               video=False, rate_limit=rate_limit, kin=kin, sag=level)
                reports.append(res.report)
                print(f"sag={level} {name}-{strategy}: {json.dumps(sag_row(res.report))}", flush=True)
    (Path(out_root) / "sag_sweep.md").write_text(summary_table(reports, sag_row) + "\n")
    return reports


# ------------------------------------------------------------------ command line
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="walk_in_sit", choices=sorted(W.SCENARIOS))
    ap.add_argument("--strategy", default="settle", choices=STRATEGIES)
    ap.add_argument("--song", default="synthetic", help="synthetic | runlog:<run.jsonl> | wav:<file>")
    ap.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rate-limit", type=int, default=M.RATE_LIMIT_PER_MIN,
                    help="SDK actions/min per session (vendor default 30; 120 is a what-if until a restart)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--fps", type=int, default=VIDEO_FPS, help="video frames per second")
    ap.add_argument("--matrix", action="store_true", help="every scenario x strategy into <out>/<scenario>-<strategy>")
    ap.add_argument("--videos", default="", help="with --matrix: scenario:strategy pairs that get a video, "
                                                  "comma separated (e.g. two_people:clip,sway_to_music:settle)")
    ap.add_argument("--sag", default="assumed", choices=list(M.SAG_LEVELS),
                    help="gravity sag level at the reference lock pose (ASSUMPTION; see motion.SAG_LEVELS)")
    ap.add_argument("--sag-sweep", action="store_true", help="every sag level x (two_people, walk_in_sit, "
                                                             "sway_to_music) x (settle, clip) into <out>/sag-<level>")
    ap.add_argument("--keep-clips", action="store_true", help="what-if: the client never deletes its clips")
    ap.add_argument("--store-initial", type=int, default=0, help="clips already in the lamp's clip store")
    args = ap.parse_args(argv)
    if args.sag_sweep:
        reports = run_sag_sweep(args.out, seconds=args.seconds, seed=args.seed, song=args.song,
                                rate_limit=args.rate_limit)
        print(summary_table(reports, sag_row))
        return 0
    if args.matrix:
        videos = {tuple(pair.split(":", 1)): True for pair in args.videos.split(",") if ":" in pair}
        reports = run_matrix(args.out, seconds=args.seconds, seed=args.seed, song=args.song,
                             rate_limit=args.rate_limit, videos=videos)
        print(summary_table(reports))
        return 0
    res = simulate(args.scenario, args.strategy, args.song, args.seconds, args.seed, args.out,
                   video=not args.no_video, rate_limit=args.rate_limit, video_fps=args.fps, sag=args.sag,
                   store_initial_files=args.store_initial, delete_clips=not args.keep_clips)
    print(json.dumps(summary_row(res.report)))
    for name in ("search", "acquire", "lock"):
        entry = res.report["replay"].get(name, {})
        if entry.get("written"):
            print(f"replay {name}: {entry.get('verdict_in_process', {}).get('status', 'validator unavailable')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
