"""Spatial awareness for the lamp: where its head is, where the target is, and where it may go.

Everything here is in metres in the lamp's base frame (z up, the table about 10 cm below the origin,
"forward" is +y when base_yaw is at neutral). Nothing in this file talks to the robot.

  LampModel.head(units)          forward kinematics: head position and the camera's axes
  LampModel.problems(units)      why a pose is not allowed: joint limits, table, the lamp's own base
  LampModel.target_point(...)    a picture position + apparent size -> a 3D point in the base frame
  LampModel.look_at(point)       inverse kinematics: a whole-arm pose that faces that point from a
                                 comfortable distance, inside the allowed workspace

The robot description (URDF and joint map) belongs to the lamp's vendor. We read it at run time from
the vendor checkout on the lamp and keep no copy of it in this repository.

Joint values are the runtime's units: -100..100 over each joint's calibrated range.
"""
from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

JOINTS = ("base_yaw", "base_pitch", "elbow_pitch", "wrist_roll", "wrist_pitch")
DEFAULT_ROBOT_DIR = Path.home() / "lelamp-hackathon-2026/static/robots/lelamp_v1/pi5_feetech_r1"

# The vendor's self-collision model (safety.yaml: head shade capsule against the base cylinder).
# Read from the file when present; these are the values it held on 2026-09-19.
SHADE = {"start": (-0.015, 0.0, 0.01875), "end": (0.11, 0.0, 0.01875), "radius": 0.06}
BASE = {"radius": 0.10, "z_min": -0.10, "z_max": 0.0114, "clearance": 0.005}


def _transform(xyz, rpy) -> np.ndarray:
    r, p, y = rpy
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = rz @ ry @ rx, xyz
    return m


def _rot_z(q: float) -> np.ndarray:
    m = np.eye(4)
    m[:2, :2] = [[math.cos(q), -math.sin(q)], [math.sin(q), math.cos(q)]]
    return m


class LampModel:
    def __init__(self, robot_dir: Path = DEFAULT_ROBOT_DIR, *, hfov_deg: float = 61.0, vfov_deg: float = 44.0,
                 table_margin: float = 0.06, base_margin: float = 0.02, limit_margin: float = 6.0):
        robot_dir = Path(robot_dir)
        mapping = json.loads((robot_dir / "joint_mapping.yaml").read_text())   # the file holds JSON
        self.neutral = {j: float(mapping["neutral"][j]) for j in JOINTS}
        self._scale = float(mapping["motion_scale"]) * float(mapping["degrees_to_radians"])
        urdf = {j.get("name"): j for j in ET.parse(robot_dir / "robot.urdf").getroot().findall("joint")}
        self._chain = []
        for motor in JOINTS:
            servo = mapping["motor_to_joint"][motor]
            origin = urdf[servo].find("origin")
            self._chain.append((motor, float(mapping["joint_offsets"][servo]),
                                _transform([float(v) for v in origin.get("xyz").split()],
                                           [float(v) for v in origin.get("rpy").split()])))
        self.shade, self.base = dict(SHADE), dict(BASE)
        self._read_safety(robot_dir / "safety.yaml")
        self.limits = {j: (-100.0 + limit_margin, 100.0 - limit_margin) for j in JOINTS}
        self.table_z = self.base["z_min"]                    # the base stands on the table
        self.table_margin, self.base_margin = table_margin, base_margin
        self.fx = 0.5 / math.tan(math.radians(hfov_deg) / 2)  # focal length in picture widths
        self.fy = 0.5 / math.tan(math.radians(vfov_deg) / 2)  # focal length in picture heights

    def _read_safety(self, path: Path) -> None:
        if not path.exists():
            return
        text = path.read_text()

        def numbers(key: str):
            found = re.search(rf"{key}:\s*\[?([-\d.,\s]+)\]?", text)
            return [float(v) for v in found.group(1).replace(",", " ").split()] if found else None

        for key, store, name in (("capsule_start_m", self.shade, "start"), ("capsule_end_m", self.shade, "end")):
            value = numbers(key)
            if value and len(value) == 3:
                store[name] = tuple(value)
        for key, store, name in (("capsule_radius_m", self.shade, "radius"), ("cylinder_radius_m", self.base, "radius"),
                                 ("cylinder_z_min_m", self.base, "z_min"), ("cylinder_z_max_m", self.base, "z_max"),
                                 ("clearance_m", self.base, "clearance")):
            value = numbers(key)
            if value:
                store[name] = value[0]

    # ------------------------------------------------------------------ forward kinematics
    def head(self, units: dict) -> dict:
        """Head position and camera axes in the base frame. The camera looks along the shade axis."""
        m = np.eye(4)
        for motor, offset, origin in self._chain:
            m = m @ origin @ _rot_z((float(units[motor]) - self.neutral[motor]) * self._scale + offset)
        forward, down = m[:3, 0], m[:3, 1]
        a = (m @ [*self.shade["start"], 1.0])[:3]
        b = (m @ [*self.shade["end"], 1.0])[:3]
        return {"position": m[:3, 3], "forward": forward, "down": down, "right": np.cross(down, forward),
                "shade": (a, b)}

    # ------------------------------------------------------------------ where the lamp may be
    def problems(self, units: dict) -> list[str]:
        """Reasons this pose is not allowed. Empty list = fine. The SDK planner checks again."""
        out = []
        for j in JOINTS:
            lo, hi = self.limits[j]
            if not lo <= float(units[j]) <= hi:
                out.append(f"{j} {float(units[j]):+.0f} is outside {lo:+.0f}..{hi:+.0f}")
        a, b = self.head(units)["shade"]
        radius = self.shade["radius"]
        for t in np.linspace(0.0, 1.0, 9):
            p = a + (b - a) * t
            if p[2] - radius < self.table_z + self.table_margin:
                out.append(f"head shade would be {100 * (p[2] - radius - self.table_z):.0f} cm above the table "
                           f"(minimum {100 * self.table_margin:.0f} cm)")
                break
        for t in np.linspace(0.0, 1.0, 9):
            p = a + (b - a) * t
            inside_height = self.base["z_min"] - radius <= p[2] <= self.base["z_max"] + radius + self.base_margin
            gap = math.hypot(p[0], p[1]) - self.base["radius"] - radius
            if inside_height and gap < self.base["clearance"] + self.base_margin:
                out.append(f"head shade would be {100 * gap:.0f} cm from the lamp's own base")
                break
        return out

    # ------------------------------------------------------------------ where the target is
    def ray(self, units: dict, where) -> np.ndarray:
        """Direction (base frame) of the picture position `where` = (x, y) in 0..1."""
        h = self.head(units)
        d = h["forward"] + h["right"] * ((where[0] - 0.5) / self.fx) + h["down"] * ((where[1] - 0.5) / self.fy)
        return d / np.linalg.norm(d)

    def project(self, units: dict, point) -> tuple[float, float] | None:
        """Where a base-frame point appears in the picture (0..1, 0..1), or None if behind the camera."""
        h = self.head(units)
        to = np.asarray(point, dtype=float) - h["position"]
        depth = float(to @ h["forward"])
        if depth <= 1e-6:
            return None
        return (0.5 + self.fx * float(to @ h["right"]) / depth, 0.5 + self.fy * float(to @ h["down"]) / depth)

    def distance_from_size(self, size_in_picture_widths: float, real_size_m: float) -> float:
        """Pinhole: an object of real_size_m that spans this much of the picture is this far away."""
        return float(np.clip(real_size_m * self.fx / max(size_in_picture_widths, 1e-3), 0.15, 2.0))

    def target_point(self, units: dict, where, distance_m: float) -> np.ndarray:
        return self.head(units)["position"] + self.ray(units, where) * distance_m

    def aim_error_deg(self, units: dict, point) -> float:
        h = self.head(units)
        to = np.asarray(point) - h["position"]
        return math.degrees(math.acos(float(np.clip(h["forward"] @ to / np.linalg.norm(to), -1, 1))))

    # ------------------------------------------------------------------ inverse kinematics
    def look_at(self, point, seed: dict | None = None, *, prefer_distance: float = 0.45,
                head_height: tuple[float, float] = (0.20, 0.36)) -> tuple[dict, dict]:
        """A whole-arm pose that faces `point`. Returns (pose, report).

        Solved over base_yaw, base_pitch, elbow_pitch, wrist_pitch (wrist_roll stays level) by damped
        least squares on: aim error, preferred viewing distance, a comfortable head height, and
        staying close to the neutral posture. Poses that fail problems() are never returned: if the
        solver cannot find an allowed pose it falls back to aiming with yaw and tilt from neutral.
        """
        point = np.asarray(point, dtype=float)
        free = ("base_yaw", "base_pitch", "elbow_pitch", "wrist_pitch")
        start = dict(self.neutral) if seed is None else {j: float(seed[j]) for j in JOINTS}
        start["wrist_roll"] = self.neutral["wrist_roll"]

        def pose_of(x) -> dict:
            pose = dict(start)
            for j, v in zip(free, x):
                pose[j] = float(np.clip(v, *self.limits[j]))
            return pose

        def residuals(x) -> np.ndarray:
            pose = pose_of(x)
            h = self.head(pose)
            to = point - h["position"]
            dist = float(np.linalg.norm(to))
            aim = np.cross(h["forward"], to / dist)                       # zero when facing the point
            z = h["position"][2]
            height = max(0.0, head_height[0] - z) + max(0.0, z - head_height[1])
            posture = [(pose[j] - self.neutral[j]) / 100.0 for j in free[1:]]   # yaw is free to turn
            return np.concatenate([aim * 4.0, [(dist - prefer_distance) * 0.6, height * 6.0], np.array(posture) * 0.35])

        x = np.array([start[j] for j in free], dtype=float)
        # a good first guess for yaw: face the point in plan view, from wherever the seed was
        h0 = self.head(pose_of(x))
        bearing_now = math.atan2(h0["forward"][0], h0["forward"][1])
        bearing_want = math.atan2(point[0] - h0["position"][0], point[1] - h0["position"][1])
        per_unit = self._yaw_bearing_per_unit(pose_of(x))
        if abs(per_unit) > 1e-6:
            x[0] = np.clip(x[0] + _wrap(bearing_want - bearing_now) / per_unit, *self.limits["base_yaw"])

        damping = 1e-2
        cost = float(residuals(x) @ residuals(x))
        for _ in range(60):
            r = residuals(x)
            jac = np.empty((len(r), len(x)))
            for k in range(len(x)):
                step = np.zeros(len(x))
                step[k] = 0.5
                jac[:, k] = (residuals(x + step) - r) / 0.5
            dx = np.linalg.solve(jac.T @ jac + damping * np.eye(len(x)), -jac.T @ r)
            dx = np.clip(dx, -15, 15)
            candidate = np.array([np.clip(v, *self.limits[j]) for j, v in zip(free, x + dx)])
            new_cost = float(residuals(candidate) @ residuals(candidate))
            if new_cost < cost:
                x, cost, damping = candidate, new_cost, max(damping * 0.5, 1e-4)
                if np.max(np.abs(dx)) < 0.05:
                    break
            else:
                damping *= 4.0
        pose = pose_of(x)
        why = self.problems(pose)
        if why:                                  # never hand back a pose we would not allow
            pose = dict(self.neutral)
            pose["base_yaw"] = float(x[0])
            why_fallback = self.problems(pose)
            if why_fallback:
                pose = dict(self.neutral)
        h = self.head(pose)
        report = {"aim_error_deg": self.aim_error_deg(pose, point), "head_height_m": float(h["position"][2]),
                  "distance_m": float(np.linalg.norm(point - h["position"])), "rejected": why}
        return pose, report

    def _yaw_bearing_per_unit(self, pose: dict) -> float:
        a = self.head(pose)["forward"]
        nudged = dict(pose)
        nudged["base_yaw"] = pose["base_yaw"] + 1.0
        b = self.head(nudged)["forward"]
        return _wrap(math.atan2(b[0], b[1]) - math.atan2(a[0], a[1]))


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi
