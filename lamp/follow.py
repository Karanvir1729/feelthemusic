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
    ap.add_argument("--target", choices=["auto", "face", "hand"], default="auto",
                    help="auto = a face if one is in view, otherwise a hand")
    ap.add_argument("--dry-run", action="store_true", help="see, locate and decide, but never move")
    ap.add_argument("--seconds", type=float, default=0, help="stop after this long (0 = until Ctrl-C)")
    ap.add_argument("--deadband-deg", type=float, default=10.0,
                    help="do not move for aim errors smaller than this (the planner itself accepts ~7 deg of elbow settle error)")
    ap.add_argument("--prefer-distance", type=float, default=0.45, help="viewing distance to aim for, metres")
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--min-interval", type=float, default=1.5, help="seconds to watch between moves")
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
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    watching = {"auto": "a face, or a hand if there is no face", "face": "a face", "hand": "a hand"}[args.target]
    print(f"watching for {watching}{' (dry run: will not move)' if args.dry_run else ''}. Ctrl-C to stop.", flush=True)
    idle_was = None if (args.dry_run or args.keep_idle) else idle_off(sdk.base)

    started = last_note = time.monotonic()
    fresh_after, sightings, moves, refused, failures = time.monotonic(), [], 0, 0, 0
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
            if now - last_note > 3:
                print(f"no camera frames... {cam.error}", flush=True)
                last_note = now
            continue
        fresh_after = max(stamp, now - 0.02 + thermal.frame_gap(tracking=bool(sightings)))
        seen, kind = None, args.target
        for name, tracker in trackers:                 # a face wins over a hand
            seen = tracker.locate(frame)
            if seen is not None:
                kind = name
                break
        sightings = [s for s in sightings if now - s[0] < 0.6 and s[4] == kind]
        if seen is None:
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
        near, far = (0.30, 2.5) if kind == "face" else (0.20, 1.2)
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
        except SDKError as exc:
            refused += 1
            if exc.status == 409:            # accepted, then failed / rejected / canceled / timed out: the arm may have moved
                print(f"     the move did not complete: {exc}", flush=True)
                settle_here(sdk)
                if exc.code == "canceled":
                    print("     something else stopped the lamp. Not fighting it: stopping.", flush=True)
                    break
                failures += 1
                if failures >= 3:
                    print("     three moves in a row did not complete. Stopping.", flush=True)
                    break
            else:                            # the gateway said no before anything moved
                print(f"     the lamp's planner refused: {exc}", flush=True)
                if "torque" in exc.message.lower() or "Too many active SDK sessions" in exc.message:
                    print("     cannot continue (torque is off, or too many SDK sessions this hour). Stopping.", flush=True)
                    break
            time.sleep(5.0 if exc.code == "rate_limited" else 2.0)
        sightings.clear()
        fresh_after = time.monotonic() + max(0.3, args.min_interval)   # settle, then watch before deciding again

    cam.running = False
    if idle_was:
        idle_restore(sdk.base, idle_was)
    print(f"SoC peaked at {thermal.peak_c:.0f} C during this run", flush=True)
    print(f"stopped. {moves} moves made, {refused} refused by the lamp's planner.", flush=True)


if __name__ == "__main__":
    main()
