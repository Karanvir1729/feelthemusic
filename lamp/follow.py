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
        self._frame, self._stamp = None, 0.0
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
    def __init__(self):
        import mediapipe as mp
        self.faces = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)

    def locate(self, bgr):
        result = self.faces.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not result.detections:
            return None
        box = max((d.location_data.relative_bounding_box for d in result.detections), key=lambda b: b.width)
        return box.xmin + box.width / 2, box.ymin + box.height / 2, box.width / FACE_WIDTH_M


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


def recent_sightings(sightings: list[tuple], now: float) -> list[tuple]:
    """Keep tracker samples for their own cadence.

    Face and hand trackers run on every analysed frame, so a short window rejects a transient
    detection. Object detection deliberately runs only once a second; its samples must survive the
    skipped frames or it can never collect the three independent detections required below.
    """
    return [sample for sample in sightings
            if now - sample[0] < (3.5 if sample[4] not in ("face", "hand") else 0.6)]


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
        self.warm_c, self.hot_c, self.peak_c, self._last, self._temp = warm_c, hot_c, 0.0, 0.0, None

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


def describe(model: LampModel, units: dict) -> str:
    h = model.head(units)
    bearing = math.degrees(math.atan2(h["forward"][0], h["forward"][1]))
    pitch = math.degrees(math.asin(float(np.clip(h["forward"][2], -1, 1))))
    return (f"head {100 * (h['position'][2] - model.table_z):.0f} cm above the table, "
            f"facing {abs(bearing):.0f} deg {'right' if bearing > 0 else 'left'}, "
            f"{abs(pitch):.0f} deg {'up' if pitch > 0 else 'down'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=["auto", "face", "hand", "object"], default="auto",
                    help="auto = a face if one is in view, otherwise a hand, otherwise a person or a thing")
    ap.add_argument("--dry-run", action="store_true", help="see, locate and decide, but never move")
    ap.add_argument("--seconds", type=float, default=0, help="stop after this long (0 = until Ctrl-C)")
    ap.add_argument("--deadband-deg", type=float, default=10.0,
                    help="do not move for aim errors smaller than this (the planner itself accepts ~7 deg of elbow settle error)")
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
    trackers = [(name, cls()) for name, cls in (("face", FaceTracker), ("hand", HandTracker))
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
    signal.signal(signal.SIGHUP, lambda *_: stop.set())
    def temperature_watch():
        last_note = 0.0
        while not stop.is_set():
            try:
                temp = int(Path(Thermal.PATH).read_text()) / 1000.0
                thermal.peak_c = max(thermal.peak_c, temp)
                if not math.isfinite(temp) or temp >= args.hot_c:
                    print(f"THERMAL STOP: Pi {temp:.1f} C; no more moves.", flush=True)
                    stop.set()
                    return
                if time.monotonic() - last_note >= 2:
                    print(f"Pi {temp:.1f} C | cutoff {args.hot_c:.1f} C | Ctrl-C stops tracking", flush=True)
                    last_note = time.monotonic()
            except (OSError, ValueError):
                print("THERMAL STOP: temperature unavailable; no more moves.", flush=True)
                stop.set()
                return
            stop.wait(0.25)
    guard = threading.Thread(target=temperature_watch, daemon=True)
    guard.start()
    watching = {"auto": "a face, then a hand, then a person or a thing", "face": "a face", "hand": "a hand",
                "object": "a person or a thing"}[args.target]
    print(f"watching for {watching}{' (dry run: will not move)' if args.dry_run else ''}. Ctrl-C to stop.", flush=True)
    idle_was = None if (args.dry_run or args.keep_idle) else idle_off(sdk.base)

    started = last_note = time.monotonic()
    fresh_after, sightings, moves, refused, failures = time.monotonic(), [], 0, 0, 0
    while not stop.is_set() and (not args.seconds or time.monotonic() - started < args.seconds):
        if thermal.too_hot():
            print("Thermal cutoff: ending tracking.", flush=True)
            break
        frame, stamp = cam.newest(after=fresh_after, timeout=1.5)
        now = time.monotonic()
        if frame is None:
            if now - last_note > 3:
                print(f"no camera frames... {cam.error}", flush=True)
                last_note = now
            continue
        fresh_after = max(stamp, now - 0.02 + thermal.frame_gap(tracking=bool(sightings)))
        seen, kind = None, args.target
        for name, tracker in trackers:                 # a face wins over a hand
            seen = tracker.locate(frame)
            if seen is not None:
                kind = tracker.label if name == "object" else name
                break
        sightings = recent_sightings(sightings, now)
        if seen is None:
            if now - last_note > 2:
                print("nobody in view" if args.target != "hand" else "no hand in view", flush=True)
                last_note = now
            continue
        sightings = [sample for sample in sightings if sample[4] == kind]
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
        where = (f"{kind} {100 * distance:.0f} cm away at x {100 * point[0]:+.0f}, y {100 * point[1]:+.0f}, "
                 f"{100 * (point[2] - model.table_z):.0f} cm above the table; aim error {error:.0f} deg")
        if point[1] < 0.05:                                       # beside or behind the base: out of reach of a calm turn
            if now - last_note > 1.5:
                print(where + "  -> behind me: not turning that far", flush=True)
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
        if biggest > args.max_step:                              # take one calm step along the way
            part = args.max_step / biggest
            step = {j: measured[j] + (pose[j] - measured[j]) * part for j in JOINTS}
            step = {j: float(np.clip(v, *model.limits[j])) for j, v in step.items()}
            if model.problems(step):
                print(where + f"  -> not moving: the step on the way is not allowed ({model.problems(step)[0]})", flush=True)
                sightings.clear()
                fresh_after = time.monotonic() + 2.0
                continue
            pose = step
        if max(abs(pose[j] - measured[j]) for j in JOINTS) > args.max_step + 0.001:
            print("Step exceeds configured bound after joint-limit adjustment; stopping.", flush=True)
            break
        # A measured joint can start beyond our extra margin while still inside the SDK range.
        # Interpolate toward the already-validated endpoint, checking geometry throughout.
        if any(any("outside" not in problem for problem in model.problems(
                {j: measured[j] + (pose[j] - measured[j]) * fraction / 20 for j in JOINTS}))
                for fraction in range(1, 21)):
            print("Intermediate pose outside our workspace; stopping.", flush=True)
            break
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
            if stop.is_set():
                break
            action = sdk.move(pose)
            moves += 1
            took = action.get("result", {}).get("duration_seconds")
            print(f"     moved{f' in {took:.1f} s' if took else ''}, planner collision-checked", flush=True)
            failures = 0
        except SDKError as exc:
            refused += 1
            text = exc.message.lower()
            if exc.status == 409:            # accepted, then it did not succeed: the arm may have moved
                print(f"     the move did not complete: {exc}", flush=True)
                errors = exc.details.get("position_errors") if isinstance(exc.details, dict) else None
                if errors:
                    print(f"     position errors: {errors}", flush=True)
                if exc.code == "canceled":
                    print("     something else stopped the lamp. Not fighting it: stopping.", flush=True)
                elif exc.code in ("timeout", "lost_track"):
                    print("     lost track of a move. Stopping rather than sending another on top of it.", flush=True)
                else:
                    print("     the arm did not reach its target, which usually means something is in the way. "
                          "Stopping: check the lamp before running again.", flush=True)
                break
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
        fresh_after = time.monotonic() + max(0.3, args.min_interval)   # settle, then watch before deciding again

    cam.running = False
    stop.set()
    if idle_was:
        idle_restore(sdk.base, idle_was)
    print(f"SoC peaked at {thermal.peak_c:.0f} C during this run", flush=True)
    print(f"stopped. {moves} moves made, {refused} refused by the lamp's planner.", flush=True)


if __name__ == "__main__":
    main()
