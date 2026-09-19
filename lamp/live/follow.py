#!/usr/bin/env python3
"""Make the LeLamp turn to face a hand (or a face). Runs on the lamp, needs no internet.

  see     camera frames from the lamp's SDK gateway, MediaPipe finds the palm or the face
  locate  picture position + apparent size -> a 3D point in the lamp's base frame (spatial.py)
  decide  inverse kinematics picks a whole-arm pose that faces that point from a comfortable
          distance, inside the allowed workspace: clear of the table, clear of the lamp's own base,
          inside joint limits. A pose that fails those checks is never sent.
  move    ONE motion.move through the SDK. The lamp's runtime plans it: reachability, self-collision,
          velocity limit, eased, at least 2 s, waits until the arm has settled.

So the lamp does not chase. It looks, thinks, and turns, every few seconds, like a person would. That
is the price of using only the vendor's safe motion path, and it is the right price for a robot that
is not ours. Every move names all five joints, so gravity-loaded joints are never left to sag.

On the lamp:
  ~/feelthemusic-lamp/.venv/bin/python follow.py --target hand --dry-run     # never moves
  ~/feelthemusic-lamp/.venv/bin/python follow.py --target hand
"""
from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from sdk import LampSDK, SDKError, read_token
from spatial import DEFAULT_ROBOT_DIR, JOINTS, LampModel

PALM_WIDTH_M = 0.08      # index knuckle to little-finger knuckle, adult
PALM_LENGTH_M = 0.10     # wrist to middle knuckle
FACE_WIDTH_M = 0.15


class Camera(threading.Thread):
    """Reads the SDK camera stream and keeps only the newest frame."""

    def __init__(self, sdk: LampSDK, fps: float):
        super().__init__(daemon=True)
        self.sdk, self.fps = sdk, fps
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stamp = 0.0
        self.error, self.running = "", True

    def run(self) -> None:
        while self.running:
            try:
                for jpeg in self.sdk.camera_frames(self.fps):
                    if not self.running:
                        return
                    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        with self._lock:
                            self._frame, self._stamp = img, time.monotonic()
            except Exception as exc:          # stream drops when the runtime restarts: reconnect
                self.error = str(exc)
            time.sleep(0.5)

    def newest(self, after: float, timeout: float = 2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._frame is not None and self._stamp > after:
                    return self._frame, self._stamp
            time.sleep(0.01)
        return None, 0.0


class HandTracker:
    def __init__(self):
        import mediapipe as mp
        self.hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=0,
                                              min_detection_confidence=0.6, min_tracking_confidence=0.5)

    def locate(self, bgr):
        """(x, y, size): palm centre in 0..1 picture coordinates and the distance that size implies."""
        result = self.hands.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not result.multi_hand_landmarks:
            return None
        lm = result.multi_hand_landmarks[0].landmark
        aspect = bgr.shape[0] / bgr.shape[1]

        def span(a, b):                       # in picture widths
            return math.hypot(lm[a].x - lm[b].x, (lm[a].y - lm[b].y) * aspect)

        palm = (0, 5, 9, 13, 17)
        x, y = float(np.mean([lm[i].x for i in palm])), float(np.mean([lm[i].y for i in palm]))
        # a tilted hand looks smaller along one axis: trust whichever span says "closer"
        return x, y, max(span(5, 17) / PALM_WIDTH_M, span(0, 9) / PALM_LENGTH_M)


class FaceTracker:
    """Keep the geometrically associated face, not whichever box is largest this frame.

    This is box continuity, not person recognition. A brief loss holds the old selection but
    returns no target; after expiry the follower must confirm the newly selected face afresh.
    """
    LOST_S = 0.6

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        import mediapipe as mp
        self.faces = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.35)
        self.clock = clock
        self._locked: tuple[float, float, float] | None = None
        self._last_seen = float("-inf")

    def locate(self, bgr: np.ndarray) -> tuple[float, float, float] | None:
        result = self.faces.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        now = self.clock()
        boxes: list[tuple[float, float, float]] = []
        for detection in result.detections or []:
            box = detection.location_data.relative_bounding_box
            candidate = (box.xmin + box.width / 2, box.ymin + box.height / 2, box.width)
            if all(math.isfinite(v) for v in candidate) and 0 < candidate[2] <= 1:
                boxes.append(candidate)
        if now - self._last_seen >= self.LOST_S:
            self._locked = None
        if not boxes:
            return None
        if self._locked is None:
            selected = max(boxes, key=lambda b: b[2])
        else:
            x, y, width = self._locked
            aspect = bgr.shape[0] / bgr.shape[1]

            def separation(box):
                return math.hypot(box[0] - x, (box[1] - y) * aspect)

            nearby = [b for b in boxes if 0.5 <= b[2] / width <= 2.0
                      and separation(b) <= max(0.06, 0.75 * width)]
            if not nearby:
                return None
            selected = min(nearby, key=separation)
        self._locked, self._last_seen = selected, now
        return selected[0], selected[1], selected[2] / FACE_WIDTH_M


class ObjectTracker:
    """Last resort when there is no face and no hand: look at a person's upper body, or at a thing.

    Google's EfficientDet-Lite0 (COCO classes) through MediaPipe. The model file is the copy already
    on the lamp, read in place: nothing to download, nothing copied. It costs about 59 ms a frame on
    the Pi 5, so it runs at most once a second, and only when the cheaper trackers found nothing.
    """
    MODEL = Path.home() / "lelamp-hackathon-2026/static/models/object_detection/efficientdet_lite0.tflite"
    # nominal real widths in metres, for a rough distance; anything not listed is ignored
    WIDTHS = {"person": 0.45, "cup": 0.09, "bottle": 0.08, "cell phone": 0.075, "book": 0.18, "laptop": 0.34,
              "remote": 0.05, "teddy bear": 0.25, "sports ball": 0.20, "mouse": 0.06, "keyboard": 0.40,
              "banana": 0.18, "apple": 0.08, "orange": 0.08, "wine glass": 0.08, "backpack": 0.35}

    def __init__(self, period_s: float = 1.0, min_score: float = 0.55):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self._mp = mp
        self.detector = vision.ObjectDetector.create_from_options(vision.ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(self.MODEL)), running_mode=vision.RunningMode.IMAGE,
            max_results=8, score_threshold=min_score))
        self.period_s, self._next, self.label = period_s, 0.0, "object"

    def locate(self, bgr):
        if time.monotonic() < self._next:
            return None
        self._next = time.monotonic() + self.period_s
        h, w = bgr.shape[:2]
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        best = None
        for det in self.detector.detect(image).detections:
            cat, box = det.categories[0], det.bounding_box
            name = cat.category_name
            if name not in self.WIDTHS or box.width > 0.9 * w:          # a box filling the frame is a false alarm
                continue
            rank = (name == "person", cat.score)                         # a person beats a thing
            if best is None or rank > best[0]:
                best = (rank, name, box)
        if best is None:
            return None
        _, name, box = best
        self.label = name
        x = (box.origin_x + box.width / 2) / w
        # for a person aim at the head end of the box, not the belly
        y = (box.origin_y + box.height * (0.15 if name == "person" else 0.5)) / h
        cut_off = box.origin_x <= 2 or box.origin_x + box.width >= w - 2   # box runs off the picture: width is a lie
        size = (box.width / w) / self.WIDTHS[name]
        return x, y, (self.NOMINAL if cut_off else size)

    NOMINAL = 0.85                              # picture widths per metre that mean "about 1 m away"; set from the camera model at start-up


# ----------------------------------------------------------------------------------------------
# The one thing the SDK cannot do: switch the lamp's looped idle animation off and on. The SDK
# resumes idle after every safe move, so with idle playing the head drifts off the person between
# moves. This is the same idle selector the lamp's own dashboard uses. It commands no motion: with
# idle "none" the arm simply holds the pose our last SDK move left it in. We put it back on exit.
def idle_off(base: str) -> str | None:
    import requests
    try:
        was = requests.get(f"{base}/api/animations/status", timeout=4).json().get("current_idle") or "idle"
        requests.post(f"{base}/api/animations/idle", json={"name": "none"}, timeout=8).raise_for_status()
        print(f"idle animation '{was}' switched off while tracking (restored on exit)", flush=True)
        return was
    except Exception as exc:
        print(f"could not switch idle off ({exc}); carrying on with idle playing", flush=True)
        return None


def idle_restore(base: str, name: str) -> None:
    import requests
    try:
        requests.post(f"{base}/api/animations/idle", json={"name": name}, timeout=8).raise_for_status()
        print(f"idle animation '{name}' restored", flush=True)
    except Exception as exc:
        print(f"COULD NOT RESTORE IDLE ({exc}). Run: curl -X POST -H 'Content-Type: application/json' "
              f"-d '{{\"name\":\"{name}\"}}' {base}/api/animations/idle", flush=True)


class Thermal:
    """Keeps our vision load from cooking the Pi. The Pi 5 throttles itself at about 80-85 C; we back
    off long before that, because the lamp's own runtime needs the headroom more than we do."""
    PATH = "/sys/class/thermal/thermal_zone0/temp"

    def __init__(self, warm_c: float, hot_c: float):
        self.warm_c, self.hot_c, self.peak_c, self._last = warm_c, hot_c, 0.0, 0.0
        self._temp: float | None = None

    def celsius(self) -> float | None:
        if time.monotonic() - self._last > 2.0:
            self._last = time.monotonic()
            try:
                self._temp = int(open(self.PATH).read()) / 1000.0
                self.peak_c = max(self.peak_c, self._temp)
            except (OSError, ValueError):
                self._temp = None
        return self._temp

    def frame_gap(self, tracking: bool) -> float:
        """Seconds between frames we bother to analyse: 10 a second while tracking someone, 4 a second
        while nobody is there, half of that when the SoC is warm."""
        gap = 0.10 if tracking else 0.25
        temp = self.celsius()
        return gap * 2 if temp is not None and temp >= self.warm_c else gap

    def too_hot(self) -> bool:
        temp = self.celsius()
        return temp is not None and temp >= self.hot_c


# ----------------------------------------------------------------------------------------------
# Live mode (--live): locked-on tracking through the runtime's own tracking route.
#
# The SDK's motion.move is a planned move of at least 2 s plus settle: right for a calm follower,
# useless for keeping a face in the middle of the picture. The runtime also has a direct position
# route, POST /api/motors/positions, that enters as category "tracking" (priority 85, idle
# suspended) and goes through the runtime's safety filter (clamped to the calibrated envelope,
# refused above 300 units/s per frame). It answers in ~5 ms, motion starts 140-250 ms later, and a
# timed move interpolates linearly. It also UNDER-DELIVERS: the arm lands 40-60 % of the commanded
# delta on base_yaw and 75-99 % on the other joints. So every command here is ONE SMALL STEP from
# the MEASURED pose (read at the start of each cycle) toward the IK pose, never from the last
# command, and the envelope below is enforced on every step before it leaves this process.
LIVE_BASE = "http://127.0.0.1:8081"
VENDOR_NEUTRAL = {"base_yaw": 4.4, "base_pitch": -49.3, "elbow_pitch": -22.5, "wrist_roll": 0.0, "wrist_pitch": 30.0}
SEARCH_POSTURE = {"base_pitch": -49.0, "elbow_pitch": -22.0, "wrist_roll": 0.0, "wrist_pitch": 30.0}
BASE_PITCH_MIN = -65.0        # the lamp tips backwards past this
FLIP_BASE_PITCH = -52.0       # shoulder further back than this AND
FLIP_ELBOW_MAX = -40.0        # ... elbow not lifted at least this far = the flip region
FLIP_ELBOW_CMD = -45.0        # what we COMMAND for the elbow there: the runtime lands 75-99 % of an
                              # elbow step, so aiming exactly at -40 leaves the arm measurably inside
                              # the region; a 5-unit margin lands it below -40
WRIST_PITCH_MAX = 60.0        # physical stop near +94 on this lamp; stay well clear
DEFAULT_LIMITS = {j: (-94.0, 94.0) for j in JOINTS}


def _clip(v: float, lo: float, hi: float) -> float:
    return float(min(max(float(v), lo), hi))


def envelope(pose: dict, limits: dict | None = None) -> dict:
    """The pose clamped into the envelope everything that moves the arm must respect."""
    limits = limits or DEFAULT_LIMITS
    p = {j: _clip(pose[j], *limits[j]) for j in JOINTS}
    p["base_pitch"] = max(p["base_pitch"], BASE_PITCH_MIN)
    p["wrist_pitch"] = min(p["wrist_pitch"], WRIST_PITCH_MAX)
    if p["base_pitch"] < FLIP_BASE_PITCH:
        p["elbow_pitch"] = min(p["elbow_pitch"], FLIP_ELBOW_CMD)   # with margin: lands <= -40
    return p


def _within_cap(measured: dict, pose: dict, cap: float, limits: dict) -> dict:
    """`pose` pulled back to within `cap` units of `measured` on every joint, still inside the
    envelope. The one rule a per-joint clamp can break is the flip rule (it couples two joints);
    it is repaired without ever exceeding the cap. Entry is gated on the MEASURED elbow, not the
    commanded one: the shoulder waits at the boundary until the arm's elbow has actually lifted
    past -40 (the runtime lands only 75-99 % of an elbow step). If the arm is already in the region
    with the elbow down, the step climbs out: shoulder forward, elbow up, each by up to `cap`.
    That step is BEST EFFORT: from deep inside (say base_pitch -64, elbow -20) one capped step
    cannot reach a legal pose, so the returned pose is still in the region, but closer to the
    boundary than the measured one on both joints. Moving out is safer than holding there."""
    p = envelope(pose, limits)
    p = {j: _clip(p[j], measured[j] - cap, measured[j] + cap) for j in JOINTS}
    if p["base_pitch"] < FLIP_BASE_PITCH:
        elbow_down = measured["elbow_pitch"] > FLIP_ELBOW_MAX or p["elbow_pitch"] > FLIP_ELBOW_CMD
        if measured["base_pitch"] >= FLIP_BASE_PITCH:
            if elbow_down:
                p["base_pitch"] = FLIP_BASE_PITCH                # wait at the boundary: within cap
        elif elbow_down:                                     # already in it: shoulder forward, elbow up
            p["base_pitch"] = _clip(min(measured["base_pitch"] + cap, FLIP_BASE_PITCH), *limits["base_pitch"])
            p["elbow_pitch"] = _clip(max(measured["elbow_pitch"] - cap, FLIP_ELBOW_CMD), *limits["elbow_pitch"])
    p["base_pitch"] = max(p["base_pitch"], BASE_PITCH_MIN)
    p["wrist_pitch"] = min(p["wrist_pitch"], WRIST_PITCH_MAX)
    return p


def step_towards(measured: dict, target: dict, cap: float, model: LampModel | None = None) -> dict:
    """One step from the MEASURED pose toward `target`: every joint moves at most `cap` units, the
    result is inside the envelope (joint limits, base_pitch >= -65, the flip rule, wrist_pitch <= 60)
    and, when a model is given, has no LampModel.problems(). If the step has a problem it is halved
    up to three times; if a quarter... an eighth step still has one, the arm holds (the measured
    pose comes back unchanged, and the caller sends nothing).

    One exception to "inside the envelope": a measured pose that is ALREADY in the flip region
    (shoulder back, elbow down: someone left it there, or a step under-delivered) is not held but
    stepped out of, best effort, up to `cap` per joint; see _within_cap."""
    limits = model.limits if model is not None else DEFAULT_LIMITS
    measured = {j: float(measured[j]) for j in JOINTS}
    goal = envelope({j: float(target[j]) for j in JOINTS}, limits)
    delta = {j: _clip(goal[j] - measured[j], -cap, cap) for j in JOINTS}
    scale = 1.0
    for _ in range(4):                                   # full step, then halved up to 3 times
        step = _within_cap(measured, {j: measured[j] + delta[j] * scale for j in JOINTS}, cap, limits)
        if model is None or not model.problems(step):
            return step
        scale *= 0.5
    return dict(measured)                                # hold


def moved_units(a: dict, b: dict) -> float:
    return max(abs(float(a[j]) - float(b[j])) for j in JOINTS)


class StallGuard:
    """Being PRESSED AGAINST something looks like under-delivery, and the runtime under-delivers
    anyway (40-60 % on base_yaw). Tell them apart by the FRACTION of the asked delta the arm made:
    below 25 % of an asked delta over 3 units, three commands in a row, is an obstruction. Then we
    stop commanding and hold until the target has moved by more than 15 deg of aim error."""

    def __init__(self, min_ask: float = 3.0, fraction: float = 0.25, strikes: int = 3, release_deg: float = 15.0):
        self.min_ask, self.fraction, self.strikes, self.release_deg = min_ask, fraction, strikes, release_deg
        self.misses, self.stalled = 0, False
        self.aim_at_stall: float | None = None
        self.last = ""                                   # commanded vs landed, for the log line

    def record(self, before: dict, commanded: dict, landed: dict, aim_error: float | None) -> bool:
        """`before` = measured when the command was sent, `landed` = measured after it played."""
        asked, got = moved_units(commanded, before), moved_units(landed, before)
        self.last = " ".join(f"{j} {commanded[j]:+.1f}->{landed[j]:+.1f}" for j in JOINTS)
        if asked > self.min_ask:
            self.misses = self.misses + 1 if got < self.fraction * asked else 0
            if self.misses >= self.strikes and not self.stalled:
                self.stalled, self.aim_at_stall = True, aim_error
        return self.stalled

    def blocked(self, aim_error: float | None) -> bool:
        """True while we must not command. A target that moved > release_deg of aim error since the
        stall (or since we first saw one after it) releases the guard."""
        if self.stalled and aim_error is not None:
            if self.aim_at_stall is None:
                self.aim_at_stall = aim_error
            elif abs(aim_error - self.aim_at_stall) > self.release_deg:
                self.reset()
        return self.stalled

    def reset(self) -> None:
        self.misses, self.stalled, self.aim_at_stall = 0, False, None


class Search:
    """Spontaneous acquisition. No face for HOLD_S: hold still (people look away for a moment).
    Still nothing: sweep base_yaw +-SWEEP units around the yaw where the face was last seen, as a
    slow triangle wave, in the vendor's neutral posture. A face ends the search at once. After
    PARK_S without a face: park at neutral and keep looking without moving."""
    HOLD_S, PARK_S, SWEEP, HZ = 3.0, 60.0, 30.0, 0.15

    def __init__(self, now: float, yaw: float, enabled: bool = True, limits: dict | None = None):
        self.last_seen, self.yaw, self.enabled = now, float(yaw), enabled
        self.limits = limits or DEFAULT_LIMITS

    def saw(self, now: float, yaw: float) -> None:
        self.last_seen, self.yaw = now, float(yaw)

    def want(self, now: float) -> tuple[str, dict | None]:
        """(state, pose to step toward or None to command nothing)."""
        gone = now - self.last_seen
        if gone < self.HOLD_S or not self.enabled:
            return "holding", None
        if gone >= self.PARK_S:
            return "parked", dict(VENDOR_NEUTRAL)
        phase = ((gone - self.HOLD_S) * self.HZ) % 1.0       # triangle wave that starts at 0, rising
        tri = 4 * phase if phase < 0.25 else (2 - 4 * phase if phase < 0.75 else 4 * phase - 4)
        yaw = _clip(self.yaw + self.SWEEP * tri, *self.limits["base_yaw"])
        return "searching", {"base_yaw": yaw, **SEARCH_POSTURE}


class LivePoster:
    """Runs each POST on a worker thread so the perception loop never blocks on the network.
    At most one command in flight, and at most one every `period` seconds."""

    def __init__(self, post, period: float, clock=time.monotonic):
        self._post, self.period, self.clock = post, period, clock
        self._lock, self._busy = threading.Lock(), False
        self._thread: threading.Thread | None = None
        self.last_sent, self.posts = float("-inf"), 0
        self.last_completed = float("-inf")
        self.rtt_ms: float | None = None
        self._error: Exception | None = None

    def ready(self, now: float | None = None) -> bool:
        now = self.clock() if now is None else now
        with self._lock:
            return not self._busy and now - self.last_sent >= self.period

    def submit(self, pose: dict, duration_ms: int, force: bool = False) -> bool:
        """True if the command was taken (it is posted in the background), False if not ready.
        `force` skips the period (for a settle command), never the one-in-flight rule."""
        now = self.clock()
        with self._lock:
            if self._busy or (not force and now - self.last_sent < self.period):
                return False
            self._busy, self.last_sent = True, now
        self._thread = threading.Thread(target=self._run, args=(pose, duration_ms), daemon=True)
        self._thread.start()
        return True

    def _run(self, pose: dict, duration_ms: int) -> None:
        t0 = time.perf_counter()
        try:
            self._post(pose, duration_ms)
            self.posts += 1
        except Exception as exc:                          # surfaced to the loop by take_error()
            self._error = exc
        finally:
            self.rtt_ms = (time.perf_counter() - t0) * 1000
            with self._lock:
                self.last_completed = self.clock()
                self._busy = False

    def take_error(self) -> Exception | None:
        exc, self._error = self._error, None
        return exc

    def wait_idle(self, timeout: float = 5.0) -> None:
        t = self._thread
        if t is not None:
            t.join(timeout)


class LiveRefused(RuntimeError):
    def __init__(self, status: int, text: str):
        super().__init__(f"HTTP {status}: {text}")
        self.status = status


class LiveLink:
    """The runtime's token-free motor routes. Two connections: the GET on the loop thread never
    queues behind a POST on the worker thread."""

    def __init__(self, base: str = LIVE_BASE, timeout: float = 2.0):
        import requests
        self.base, self.timeout = base.rstrip("/"), timeout
        self.http, self.post_http = requests.Session(), requests.Session()

    def positions(self) -> tuple[dict, bool]:
        r = self.http.get(f"{self.base}/api/motors/positions", timeout=self.timeout)
        r.raise_for_status()
        body = r.json()
        return {j: float(body["positions"][j]) for j in JOINTS}, bool(body.get("torque_enabled", True))

    def post(self, pose: dict, duration_ms: int) -> None:
        body = {"positions": {j: round(float(pose[j]), 1) for j in JOINTS}, "duration_ms": int(duration_ms)}
        r = self.post_http.post(f"{self.base}/api/motors/positions", json=body, timeout=self.timeout)
        if r.status_code >= 400:
            raise LiveRefused(r.status_code, r.text[:200])


class LiveConfig:
    def __init__(self, target: str = "face", deadband_deg: float = 4.0, prefer_distance: float = 0.45,
                 live_ms: int = 250, period: float = 0.25, step: float = 10.0, search: bool = True,
                 dry_run: bool = False, seconds: float = 0.0, land_settle: float | None = None):
        self.target, self.deadband_deg, self.prefer_distance = target, deadband_deg, prefer_distance
        self.live_ms, self.period, self.step, self.search = int(live_ms), period, step, search
        self.dry_run, self.seconds = dry_run, seconds
        # motion starts 140-250 ms after the POST and plays for live_ms: read "landed" after that
        self.land_settle = (0.30 + live_ms / 1000.0) if land_settle is None else land_settle


class LiveFollower:
    """The locked-on loop. Everything with a side effect is injected (camera, trackers, motor link,
    clock, sleep) so the whole loop runs against fakes on a laptop."""
    FACE_STICKY_S = 5.0            # with --target auto: once a face is seen, faces only for this long
    SIGHTINGS = 3

    def __init__(self, model: LampModel, link, frames, trackers, thermal, cfg: LiveConfig, *,
                 clock=time.monotonic, sleep=None, out=print, poster: LivePoster | None = None):
        self.model, self.link, self.frames, self.trackers, self.thermal, self.cfg = model, link, frames, trackers, thermal, cfg
        self.stop = threading.Event()                     # run() swaps in the caller's
        # pauses go through stop.wait so Ctrl-C ends them at once; tests inject a fake-clock sleep
        self.clock, self.sleep, self.out = clock, sleep or (lambda s: self.stop.wait(s)), out
        self.poster = poster or LivePoster(link.post, cfg.period, clock)
        self.guard = StallGuard()
        self.search: Search | None = None                 # made on the first measured pose
        self.sightings: list[tuple[float, np.ndarray, str]] = []
        self.history: list[tuple[float, dict]] = []
        self.pending: list[tuple[float, dict, dict]] = []
        self.face_hold_until, self.fresh_after, self.last_note = float("-inf"), float("-inf"), float("-inf")
        self.state = "holding"
        self.aim_error: float | None = None
        self.fatal: str | None = None
        self.commands, self.refusals, self.read_failures, self.post_failures, self.consecutive_refusals = 0, 0, 0, 0, 0
        self.settles = 0
        self.stalled_noted, self.parked_noted = False, False
        self.correcting = False
        self.confirmed_face_until = float("-inf")

    # -------------------------------------------------------------- helpers
    def _pose_at(self, stamp: float) -> dict:
        """The measured pose nearest the time a frame was taken (the arm moves between frames)."""
        return min(self.history, key=lambda h: abs(h[0] - stamp))[1]

    def _settle(self, measured: dict, why: str) -> bool:
        """One POST of the MEASURED pose (all five joints), so the runtime's servo goal becomes where
        the arm is. After a step that did not land, the runtime leaves the goal at the unreached
        target and the servos keep pushing: into an obstruction, that is how they cook. Used when the
        stall guard trips and when the loop ends. Not a step, so it feeds no guard sample."""
        if self.cfg.dry_run:
            return False
        self.poster.wait_idle()                           # never two in flight
        if not self.poster.submit({j: float(measured[j]) for j in JOINTS}, self.cfg.live_ms, force=True):
            self.out(f"     could not settle ({why}): a post is still in flight", flush=True)
            return False
        self.settles += 1
        self.pending = []                                 # earlier steps are superseded: no stale samples
        self.out(f"     settled at the measured pose ({why}): the servos stop pushing", flush=True)
        return True

    def _feed_guard(self, now: float, measured: dict) -> None:
        """Commands whose motion has had time to play: compare where the arm landed with what was asked."""
        if not self.poster.ready(now):
            return
        # A slow request must not spend the landing allowance before the runtime accepts it.
        completed = self.poster.last_completed + self.cfg.land_settle
        due = [p for p in self.pending if now >= max(p[0], completed)]
        self.pending = [p for p in self.pending if now < max(p[0], completed)]
        for _, before, commanded in due:
            tripped_before = self.guard.stalled
            self.guard.record(before, commanded, measured, self.aim_error)
            if self.guard.stalled and not tripped_before:
                self.out("stalled: something is in the way", flush=True)
                self.stalled_noted = True
                self._settle(measured, "stalled")
                break

    def _send(self, now: float, measured: dict, goal: dict) -> tuple[dict | None, str]:
        """Step from the measured pose toward `goal` and post it. Returns (step or None, note)."""
        if self.pending:
            return None, "waiting for the last step to land"
        step = step_towards(measured, goal, self.cfg.step, self.model)
        if moved_units(step, measured) < 0.05:
            return None, "hold"
        if self.cfg.dry_run:
            self.poster.last_sent = now                  # keep the cadence honest in a dry run
            return step, "would send " + " ".join(f"{j} {measured[j]:+.1f}->{step[j]:+.1f}" for j in JOINTS
                                                  if abs(step[j] - measured[j]) >= 0.05)
        if not self.poster.submit(step, self.cfg.live_ms):
            return None, "busy"
        self.commands += 1
        self.pending.append((self.poster.last_sent + self.cfg.land_settle, dict(measured), dict(step)))
        return step, "sent"

    def _post_errors(self) -> None:
        exc = self.poster.take_error()
        if exc is None:
            self.consecutive_refusals = self.post_failures = 0     # both counters mean "in a row"
            return
        self.pending.clear()                            # a failed POST is not a landed movement sample
        if isinstance(exc, LiveRefused):
            self.refusals += 1
            self.consecutive_refusals += 1
            self.out(f"     the runtime refused the step: {exc}", flush=True)
            if self.consecutive_refusals >= 3:
                self.fatal = "three refusals in a row: this needs a person to look at it"
            return
        self.post_failures += 1
        self.out(f"     post failed: {type(exc).__name__}: {exc}", flush=True)
        if self.post_failures >= 3:
            self.fatal = "three failed posts in a row: is the runtime up?"

    # -------------------------------------------------------------- one perception cycle
    def cycle(self) -> str:
        cfg, model = self.cfg, self.model
        if self.thermal is not None and self.thermal.too_hot():
            self.out(f"SoC is {self.thermal.celsius():.0f} C: pausing vision for 20 s to let it cool", flush=True)
            self.sightings.clear()
            self.sleep(20)
            self.fresh_after = self.clock()
            return "hot"
        frame, stamp = self.frames.newest(after=self.fresh_after, timeout=1.5)
        now = self.clock()
        if frame is None:
            if now - self.last_note > 3 and getattr(self.frames, "error", ""):
                self.out(f"no camera frames: {self.frames.error}", flush=True)
                self.last_note = now
            return "no-frame"
        gap = self.thermal.frame_gap(tracking=bool(self.sightings)) if self.thermal is not None else 0.1
        self.fresh_after = max(stamp, now - 0.02 + gap)
        frame_age_ms = (now - stamp) * 1000

        try:
            measured, torque = self.link.positions()
            self.read_failures = 0
        except Exception as exc:
            self.read_failures += 1
            self.out(f"cannot read joints: {type(exc).__name__}: {exc}", flush=True)
            if self.read_failures >= 10:
                self.fatal = "cannot read the motors: stopping"
            self.sleep(1)
            return "no-joints"
        if not torque:
            self.fatal = "torque is off: stopping (nothing here turns it on)"
            return "no-torque"
        self.history.append((now, measured))
        del self.history[:-30]
        if self.search is None:
            self.search = Search(now, measured["base_yaw"], enabled=cfg.search, limits=model.limits)
        self._post_errors()
        if not cfg.dry_run:
            self._feed_guard(now, measured)

        # see: a face first; in auto mode stay on faces for a while after one was seen
        t0 = time.perf_counter()
        seen, kind = None, cfg.target
        for name, tracker in self.trackers:
            if name != "face" and now < self.face_hold_until:
                continue
            seen = tracker.locate(frame)
            if seen is not None:
                kind = tracker.label if name == "object" else name
                break
        detect_ms = (time.perf_counter() - t0) * 1000
        now = self.clock()                               # inference time counts against sighting freshness
        if kind == "face" and seen is not None:
            self.face_hold_until = now + self.FACE_STICKY_S

        ik_ms, note, goal = 0.0, "", None
        if seen is not None:
            x, y, size = seen
            near, far = {"face": (0.30, 2.5), "hand": (0.20, 1.2)}.get(kind, (0.30, 3.0))
            distance = float(np.clip(model.distance_from_size(size, 1.0), near, far))
            point = model.target_point(self._pose_at(stamp), (x, y), distance)
            window = 0.6 if kind in ("face", "hand") else 3.5
            self.sightings = [s for s in self.sightings if now - s[0] < window and s[2] == kind]
            if not self.sightings:
                self.correcting, self.aim_error = False, None
            self.sightings.append((now, point, kind))
            del self.sightings[:-self.SIGHTINGS]
            # A confirmed face can blink out for a frame without starting a search. Only the
            # associated incumbent survives this grace period; new selection follows lock expiry.
            if kind == "face" and now < self.confirmed_face_until:
                self.search.saw(now, measured["base_yaw"])
                self.confirmed_face_until = now + FaceTracker.LOST_S
            if len(self.sightings) < self.SIGHTINGS:      # decide on a steady sighting, not on one frame
                self.state, note = "tracking", f"{kind} sighted ({len(self.sightings)}/{self.SIGHTINGS})"
            else:
                self.search.saw(now, measured["base_yaw"])
                self.parked_noted = False
                if kind == "face":
                    self.confirmed_face_until = now + FaceTracker.LOST_S
                target = np.median(np.array([s[1] for s in self.sightings]), axis=0)
                self.aim_error = model.aim_error_deg(measured, target)
                # Once centered, small detector noise must not restart a correction. A genuine
                # correction continues to the tighter stop band before this latch clears.
                if self.aim_error < cfg.deadband_deg:
                    self.correcting = False
                elif self.aim_error >= 1.5 * cfg.deadband_deg:
                    self.correcting = True
                if self.guard.blocked(self.aim_error):
                    self.state, note = "stalled", "holding until the target moves"
                elif target[1] < 0.05:
                    self.state, note = "holding", "behind me: no reachable pose"
                elif not self.correcting:
                    self.state, note = "tracking", "on target"
                elif self.pending or not self.poster.ready(now):
                    self.state, note = "tracking", "waiting for the last step"
                else:
                    t1 = time.perf_counter()
                    goal, report = model.look_at(target, prefer_distance=cfg.prefer_distance)
                    ik_ms = (time.perf_counter() - t1) * 1000
                    step, note = self._send(now, measured, goal)
                    self.state = "tracking"
                    if report["rejected"]:
                        note += " (whole-arm pose refused; yaw+tilt from neutral)"
        else:
            self.aim_error = None
            self.correcting = False
            if cfg.target in ("face", "hand") or (self.sightings and self.sightings[-1][2] in ("face", "hand")):
                self.sightings.clear()
            self.state, want = self.search.want(now)
            if self.state == "stalled" or self.guard.blocked(None):
                self.state, note = "stalled", "holding until the target moves"
            elif want is None:
                note = "no face"
            elif self.state == "parked" and moved_units(want, measured) < 1.0:
                note = "at neutral, still looking"
                if not self.parked_noted:
                    self.out("no face for a minute: parked at neutral, still looking", flush=True)
                    self.parked_noted = True
            elif not self.pending and self.poster.ready(now):
                step, note = self._send(now, measured, want)
            else:
                note = "waiting for the last step"

        rtt = self.poster.rtt_ms
        aim = f"{self.aim_error:5.1f}" if self.aim_error is not None else "  -- "
        self.out(f"{self.state:9s} frame {frame_age_ms:4.0f}ms det {detect_ms:3.0f}ms ik {ik_ms:3.0f}ms "
                 f"post {rtt if rtt is not None else float('nan'):4.0f}ms aim {aim}deg  {note}"
                 f"{'  prev ' + self.guard.last if self.guard.last else ''}", flush=True)
        return self.state

    def run(self, stop: threading.Event) -> None:
        self.stop = stop                                  # the pauses in cycle() wait on it
        started = self.clock()
        while not stop.is_set() and self.fatal is None and (not self.cfg.seconds or self.clock() - started < self.cfg.seconds):
            self.cycle()
        self.poster.wait_idle()
        self._post_errors()
        if self.fatal:
            self.out(f"     {self.fatal}", flush=True)
        # leave the servo goal where the arm is, not at the last unreached step (Ctrl-C, --seconds,
        # or a terminal error mid-move): the last cycle's step may still be under-delivering
        if self.commands and not self.cfg.dry_run:        # (a torque-off read below skips the settle)
            try:
                measured, torque = self.link.positions()
            except Exception as exc:
                measured, torque = None, False
                self.out(f"     could not settle on exit: cannot read joints ({type(exc).__name__}: {exc})", flush=True)
            if measured is not None and torque and self._settle(measured, "leaving"):
                self.poster.wait_idle()
                self._post_errors()


def settle_here(sdk: LampSDK) -> None:
    """After a move that did not complete, the runtime leaves the servo goal at the unreached target
    (so the servos keep pushing) and its idle loop paused. One planned hold at the measured pose
    completes at once: the goal becomes where the arm is, and the runtime tidies up."""
    try:
        here = {j: max(-100.0, min(100.0, float(v))) for j, v in sdk.joints()["positions"].items() if j in JOINTS}
        sdk.move(here)
    except SDKError as exc:
        print(f"     could not settle: {exc}", flush=True)


def describe(model: LampModel, units: dict) -> str:
    h = model.head(units)
    bearing = math.degrees(math.atan2(h["forward"][0], h["forward"][1]))
    pitch = math.degrees(math.asin(float(np.clip(h["forward"][2], -1, 1))))
    return (f"head {100 * (h['position'][2] - model.table_z):.0f} cm above the table, "
            f"facing {abs(bearing):.0f} deg {'right' if bearing > 0 else 'left'}, "
            f"{abs(pitch):.0f} deg {'up' if pitch > 0 else 'down'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=["auto", "face", "hand", "object"], default=None,
                    help="auto = a face if one is in view, otherwise a hand, otherwise a person or a thing "
                         "(default: auto; face with --live)")
    ap.add_argument("--dry-run", action="store_true", help="see, locate and decide, but never move")
    ap.add_argument("--seconds", type=float, default=0, help="stop after this long (0 = until Ctrl-C)")
    ap.add_argument("--deadband-deg", type=float, default=None,
                    help="do not move for aim errors smaller than this (default 10: the planner itself accepts "
                         "~7 deg of elbow settle error; 4 with --live)")
    live = ap.add_argument_group("live mode", "locked-on tracking through the runtime's tracking route instead of planned SDK moves")
    live.add_argument("--live", action="store_true",
                      help="small closed-loop steps through POST /api/motors/positions, several a second")
    live.add_argument("--live-ms", type=int, default=250, help="duration_ms of each timed step")
    live.add_argument("--live-period", type=float, default=0.25, help="seconds between commands, at least")
    live.add_argument("--live-step", type=float, default=10.0, help="largest per-joint change per command, units")
    live.add_argument("--live-base", default=LIVE_BASE, help="the runtime's dashboard (motor routes)")
    live.add_argument("--no-search", action="store_true",
                      help="with no face in view, hold still instead of the slow yaw sweep")
    ap.add_argument("--prefer-distance", type=float, default=0.45, help="viewing distance to aim for, metres")
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--min-interval", type=float, default=1.5, help="seconds to watch between moves")
    ap.add_argument("--max-step", type=float, default=20.0,
                    help="largest joint change per move, in units. Every SDK move takes about 2 s whatever its size, "
                         "so an unlimited move can peak near 70 deg/s. A big turn becomes several calm steps.")
    ap.add_argument("--warm-c", type=float, default=70.0, help="SoC temperature at which we halve our frame rate")
    ap.add_argument("--hot-c", type=float, default=77.0, help="SoC temperature at which we stop looking until it cools")
    ap.add_argument("--keep-idle", action="store_true",
                    help="leave the lamp's idle animation playing between our moves (the head will drift off target)")
    ap.add_argument("--robot-dir", default=str(DEFAULT_ROBOT_DIR))
    args = ap.parse_args()
    if args.target is None:
        args.target = "face" if args.live else "auto"
    if args.deadband_deg is None:
        args.deadband_deg = 4.0 if args.live else 10.0

    model = LampModel(args.robot_dir)
    try:
        sdk = LampSDK(read_token())
        caps = sdk.capabilities()
        info = sdk.joints()
    except SDKError as exc:
        sys.exit(f"the lamp's SDK gateway is not usable: {exc}\n"
                 "(it needs LELAMP_SDK_TOKEN in the lamp's .env and a runtime restart)")
    if not info.get("self_collision_check"):
        sys.exit("the runtime reports self_collision_check = false: refusing to move this robot")
    if info.get("units") != "normalized_m100_100" or set(info.get("joints", {})) != set(JOINTS):
        sys.exit(f"unexpected joint space from the SDK: {info.get('units')} {sorted(info.get('joints', {}))}")
    measured = {j: float(info["positions"][j]) for j in JOINTS}
    print(f"SDK ok ({caps.get('protocol_version', info.get('units'))}), planner self-collision check on, "
          f"velocity cap {info.get('max_velocity_units_s')} units/s", flush=True)
    print(f"joint angles from: {model.scale_source}", flush=True)
    print("now: " + describe(model, measured), flush=True)

    try:
        os.nice(10)                       # the lamp's own runtime always gets the CPU before we do
    except OSError:
        pass
    thermal = Thermal(args.warm_c, args.hot_c)
    print(f"SoC {thermal.celsius() or float('nan'):.0f} C now; we slow down at {args.warm_c:.0f} C and pause at {args.hot_c:.0f} C",
          flush=True)
    cam = Camera(sdk, args.fps)
    cam.start()
    trackers: list[tuple[str, FaceTracker | HandTracker | ObjectTracker]] = [
                (name, cls()) for name, cls in (("face", FaceTracker), ("hand", HandTracker))
                if args.target in ("auto", name)]
    if args.target in ("auto", "object"):
        if ObjectTracker.MODEL.exists():
            ObjectTracker.NOMINAL = model.fx / 1.0                # a cut-off box is treated as about 1 m away
            trackers.append(("object", ObjectTracker()))
        else:
            print(f"object tracking off: no model at {ObjectTracker.MODEL}", flush=True)
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    watching = {"auto": "a face, then a hand, then a person or a thing", "face": "a face", "hand": "a hand",
                "object": "a person or a thing"}[args.target]
    print(f"watching for {watching}{' (dry run: will not move)' if args.dry_run else ''}. Ctrl-C to stop.", flush=True)
    idle_was = None if (args.dry_run or args.keep_idle) else idle_off(sdk.base)

    if args.live:
        cfg = LiveConfig(target=args.target, deadband_deg=args.deadband_deg, prefer_distance=args.prefer_distance,
                         live_ms=args.live_ms, period=args.live_period, step=args.live_step,
                         search=not args.no_search, dry_run=args.dry_run, seconds=args.seconds)
        print(f"live: one step of <= {cfg.step:g} units every {cfg.period:g} s ({cfg.live_ms} ms moves) from the "
              f"measured pose, deadband {cfg.deadband_deg:g} deg, search {'on' if cfg.search else 'off'}", flush=True)
        try:
            link = LiveLink(args.live_base)
            link.positions()
        except Exception as exc:
            cam.running = False
            if idle_was:
                idle_restore(sdk.base, idle_was)
            sys.exit(f"the runtime's motor route is not usable at {args.live_base}: {type(exc).__name__}: {exc}")
        follower = LiveFollower(model, link, cam, trackers, thermal, cfg)
        follower.run(stop)
        cam.running = False
        if idle_was:
            idle_restore(sdk.base, idle_was)
        print(f"SoC peaked at {thermal.peak_c:.0f} C during this run", flush=True)
        print(f"stopped. {follower.commands} steps sent, {follower.refusals} refused by the runtime, "
              f"{follower.settles} settle{'s' if follower.settles != 1 else ''}"
              f"{', stalled' if follower.guard.stalled else ''}.", flush=True)
        return

    started = last_note = time.monotonic()
    fresh_after, moves, refused, failures = time.monotonic(), 0, 0, 0
    sightings: list[tuple[float, float, float, float, str]] = []
    not_reached = 0            # consecutive moves that ended short of target
    while not stop.is_set() and (not args.seconds or time.monotonic() - started < args.seconds):
        if thermal.too_hot():
            print(f"SoC is {thermal.celsius():.0f} C: pausing vision for 20 s to let it cool", flush=True)
            sightings.clear()
            stop.wait(20)
            fresh_after = time.monotonic()
            continue
        frame, stamp = cam.newest(after=fresh_after, timeout=1.5)
        now = time.monotonic()
        if frame is None:
            if now - last_note > 3 and cam.error:
                print(f"no camera frames: {cam.error}", flush=True)
                last_note = now
            continue
        fresh_after = max(stamp, now - 0.02 + thermal.frame_gap(tracking=bool(sightings)))
        seen, kind = None, args.target
        for name, tracker in trackers:                 # a face wins over a hand
            seen = tracker.locate(frame)
            if seen is not None:
                kind = tracker.label if isinstance(tracker, ObjectTracker) else name
                break
        now = time.monotonic()                           # do not blend pre-inference sightings after a slow frame
        sightings = [s for s in sightings if now - s[0] < (3.5 if kind not in ("face", "hand") else 0.6) and s[4] == kind]
        if seen is None:
            if args.target in ("face", "hand") or (sightings and sightings[-1][4] in ("face", "hand")):
                sightings.clear()
            if now - last_note > 2:
                print("nobody in view" if args.target != "hand" else "no hand in view", flush=True)
                last_note = now
            continue
        sightings.append((now, *seen, kind))
        if len(sightings) < 3:                 # decide on a steady sighting, not on one frame
            continue
        x, y, size = (float(np.median([s[k] for s in sightings])) for k in (1, 2, 3))

        try:
            measured = {j: float(v) for j, v in sdk.joints()["positions"].items() if j in JOINTS}
        except (SDKError, OSError) as exc:
            print(f"cannot read joints: {exc}", flush=True)
            time.sleep(1)
            continue
        near, far = {"face": (0.30, 2.5), "hand": (0.20, 1.2)}.get(kind, (0.30, 3.0))
        distance = float(np.clip(model.distance_from_size(size, 1.0), near, far))   # size = picture widths per metre
        point = model.target_point(measured, (x, y), distance)
        error = model.aim_error_deg(measured, point)
        # How hard to chase, from how far off-axis the target is. At the deadband edge this is 1 and
        # nothing changes; a target far to the side gets a proportionally bigger step and a shorter
        # pause before we look again. Capped at 3x so it stays a turn, not a lunge -- and every move
        # still goes through the SDK planner, so the velocity ceiling and easing are unchanged.
        urgency = float(min(max(error / (2.0 * max(args.deadband_deg, 1e-3)), 1.0), 3.0))
        allowed_step = args.max_step * urgency
        where = (f"{kind} {100 * distance:.0f} cm away at x {100 * point[0]:+.0f}, y {100 * point[1]:+.0f}, "
                 f"{100 * (point[2] - model.table_z):.0f} cm above the table; aim error {error:.0f} deg")
        if point[1] < 0.05:                                       # beside or behind the base: out of reach
            if now - last_note > 1.5:
                print(where + "  -> behind me: no reachable pose", flush=True)
                last_note = now
            continue
        if error < args.deadband_deg:
            if now - last_note > 1.5:
                print(where + "  -> already facing it", flush=True)
                last_note = now
            continue

        pose, report = model.look_at(point, prefer_distance=args.prefer_distance)
        blocked = model.problems(pose)
        if blocked:                                              # look_at never returns these; belt and braces
            print(where + f"  -> not moving: {blocked[0]}", flush=True)
            sightings.clear()
            continue
        biggest = max(abs(pose[j] - measured[j]) for j in JOINTS)
        if biggest > allowed_step:                               # one step along the way, sized by urgency
            part = allowed_step / biggest
            step = {j: measured[j] + (pose[j] - measured[j]) * part for j in JOINTS}
            step = {j: float(np.clip(v, *model.limits[j])) for j, v in step.items()}
            if model.problems(step):
                print(where + f"  -> not moving: the step on the way is not allowed ({model.problems(step)[0]})", flush=True)
                sightings.clear()
                fresh_after = time.monotonic() + 2.0
                continue
            pose = step
        plan = ", ".join(f"{j} {measured[j]:+.0f}->{pose[j]:+.0f}" for j in JOINTS if abs(pose[j] - measured[j]) >= 2)
        note = f" (whole-arm pose refused: {report['rejected'][0]}; turning from neutral instead)" if report["rejected"] else ""
        print(where, flush=True)
        print(f"  -> {plan or 'hold'}{note}\n     after: " + describe(model, pose), flush=True)
        last_note = now
        if args.dry_run:
            sightings.clear()
            fresh_after = time.monotonic() + 1.0
            continue
        try:
            action = sdk.move(pose)
            moves += 1
            took = action.get("result", {}).get("duration_seconds")
            print(f"     moved{f' in {took:.1f} s' if took else ''}, planner collision-checked", flush=True)
            failures = 0
            not_reached = 0
        except SDKError as exc:
            refused += 1
            text = exc.message.lower()
            if exc.status == 409:            # accepted, then it did not succeed: the arm may have moved
                print(f"     the move did not complete: {exc}", flush=True)
                errors = exc.details.get("position_errors") if isinstance(exc.details, dict) else None
                if errors:
                    print(f"     position errors: {errors}", flush=True)
                settle_here(sdk)
                if exc.code == "canceled":
                    print("     something else stopped the lamp. Not fighting it: stopping.", flush=True)
                    break
                if exc.code in ("timeout", "lost_track"):
                    print("     lost track of a move. Stopping rather than sending another on top of it.", flush=True)
                    break
                not_reached += 1
                # Ending short is normal on this lamp: wrist_pitch is calibrated to 65% of its range,
                # so the IK regularly asks for angles the servo cannot deliver. Being PRESSED AGAINST
                # something is different, and retrying into an obstruction is how servos cook. Tell
                # them apart by how much the arm moved as a FRACTION of what was asked -- a small
                # commanded move looks identical to a blocked one in absolute terms.
                stalled = False
                try:
                    landed = {j: float(v) for j, v in sdk.joints()["positions"].items() if j in JOINTS}
                    asked = max(abs(pose[j] - measured[j]) for j in JOINTS)
                    got = max(abs(landed[j] - measured[j]) for j in JOINTS)
                    stalled = asked > 3.0 and got < 0.25 * asked
                    print(f"     asked for {asked:.0f} units, arm moved {got:.0f}", flush=True)
                except (SDKError, OSError, KeyError, ValueError):
                    pass                                     # cannot tell: fall back to the counter
                if stalled:
                    print("     the arm barely moved: treating that as an obstruction, not a short "
                          "joint. Stopping before we push into it.", flush=True)
                    break
                if not_reached >= 3:
                    print("     three misses in a row: something is genuinely in the way. Stopping.", flush=True)
                    break
                print(f"     ended short of target ({not_reached}/3); still following.", flush=True)
                sightings.clear()
                stop.wait(1.0)                               # interruptible, and does not starve cam.newest
                fresh_after = time.monotonic()
                continue
            print(f"     the lamp's planner refused: {exc}", flush=True)     # nothing moved
            if "torque" in text or "too many active sdk sessions" in text:
                print("     cannot continue (torque is off, or too many SDK sessions this hour). Stopping.", flush=True)
                break
            failures += 1
            if failures >= 3:
                print("     three refusals in a row. Stopping: this needs a person to look at it.", flush=True)
                break
            # never hammer: a collision refusal will be refused again, a rate limit needs a full window
            stop.wait(60.0 if exc.status == 429 or exc.code == "rate_limited" else 10.0)
        sightings.clear()
        fresh_after = time.monotonic() + max(0.3, args.min_interval / urgency)   # settle; hurry when far off

    cam.running = False
    if idle_was:
        idle_restore(sdk.base, idle_was)
    print(f"SoC peaked at {thermal.peak_c:.0f} C during this run", flush=True)
    print(f"stopped. {moves} moves made, {refused} refused by the lamp's planner.", flush=True)


if __name__ == "__main__":
    main()
