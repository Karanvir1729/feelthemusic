"""Offline kinematic replay of a URDF; no hardware or dynamic safety approval.

Written 2026-09-19. Loads external meshes without copying vendor assets into
this repository. Contact exclusions are NOT guessed from a baseline pose.
An assembled model with overlapping collision meshes fails rather than having
those contacts silently suppressed. Actuator/torque acceptance requires a twin.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def load_model(urdf_path):
    import mujoco

    path = Path(urdf_path).resolve(strict=True)
    root = ET.parse(path).getroot()
    if root.tag != "robot":
        raise ValueError("expected a URDF robot")
    assets = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()}
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename", "")
        if filename.startswith("package://"):
            filename = filename[len("package://"):]
        asset = (path.parent / filename).resolve(strict=True)
        mesh.set("filename", str(asset))
        assets[str(asset)] = hashlib.sha256(asset.read_bytes()).hexdigest()
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    if any(kind != mujoco.mjtJoint.mjJNT_HINGE for kind in model.jnt_type):
        raise ValueError("this replay supports revolute joints only")
    # Hash the bytes of every loaded mesh, not just the URDF reference strings.
    digest = hashlib.sha256(json.dumps(sorted(assets.values())).encode()).hexdigest()
    return model, digest


def replay(urdf_path, samples):
    """Replay ordered {time_s, positions_rad} samples with all joints specified.

    Reports sampled geometry only. No velocities, actuator tracking, calibration,
    between-sample clearance, SDK acceptance or synchronization is certified.
    """
    import mujoco

    model, digest = load_model(urdf_path)
    data = mujoco.MjData(model)
    names = [model.joint(i).name for i in range(model.njnt)]
    if not isinstance(samples, list) or not 2 <= len(samples) <= 100_000:
        raise ValueError("need 2..100000 samples")
    previous = -1.0
    max_gap = 0.0
    collision_samples = limit_samples = 0
    min_contact_distance = None
    pairs = set()
    for sample in samples:
        if not isinstance(sample, dict) or set(sample) != {"time_s", "positions_rad"}:
            raise ValueError("sample must contain time_s and positions_rad")
        stamp, pose = sample["time_s"], sample["positions_rad"]
        if type(stamp) not in (int, float) or not math.isfinite(stamp) or stamp < 0 or stamp <= previous:
            raise ValueError("time_s must be finite, nonnegative and strictly increasing")
        if previous >= 0:
            max_gap = max(max_gap, stamp - previous)
        previous = stamp
        if not isinstance(pose, dict) or set(pose) != set(names):
            raise ValueError("positions_rad must specify exactly every URDF joint")
        values = [pose[name] for name in names]
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError("joint positions must be finite numbers in radians")
        outside = False
        for i, value in enumerate(values):
            data.qpos[model.jnt_qposadr[i]] = value
            if model.jnt_limited[i] and not model.jnt_range[i, 0] <= value <= model.jnt_range[i, 1]:
                outside = True
        limit_samples += outside
        mujoco.mj_forward(model, data)
        penetrations = [c for c in data.contact if c.dist < -1e-6]
        collision_samples += bool(penetrations)
        for contact in penetrations:
            distance = float(contact.dist)
            min_contact_distance = distance if min_contact_distance is None else min(distance, min_contact_distance)
            pair = tuple(sorted(model.body(int(model.geom_bodyid[g])).name
                                for g in (contact.geom1, contact.geom2)))
            pairs.add(pair)
    return {
        "status": "FAIL" if collision_samples or limit_samples else "PASS_SAMPLED_KINEMATICS_ONLY",
        "hardware_approved": False,
        "dynamics_verified": False,
        "model_assets_sha256": digest,
        "mujoco_version": mujoco.__version__,
        "joint_names": names,
        "actuator_count": int(model.nu),
        "sample_count": len(samples),
        "duration_s": samples[-1]["time_s"] - samples[0]["time_s"],
        "max_sample_gap_s": max_gap,
        "collision_samples": collision_samples,
        "joint_limit_samples": limit_samples,
        "min_penetration_distance_m": min_contact_distance,
        "contact_body_pairs": sorted(pairs),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urdf", type=Path)
    parser.add_argument("trajectory", type=Path, help="JSON list of time_s / positions_rad samples")
    args = parser.parse_args()
    report = replay(args.urdf, json.loads(args.trajectory.read_text()))
    print(json.dumps(report, indent=2))
    return int(report["status"] == "FAIL")


if __name__ == "__main__":
    raise SystemExit(main())
