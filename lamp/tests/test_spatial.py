"""Spatial model checks. They need the vendor's robot description, which is not in this repository:
point FTM_ROBOT_DIR at .../static/robots/lelamp_v1/pi5_feetech_r1 (on the lamp it is found by default)."""
import math
import os
from pathlib import Path

import numpy as np
import pytest

from spatial import DEFAULT_ROBOT_DIR, JOINTS, LampModel

ROBOT_DIR = Path(os.environ.get("FTM_ROBOT_DIR", DEFAULT_ROBOT_DIR))
pytestmark = pytest.mark.skipif(not (ROBOT_DIR / "robot.urdf").exists(), reason="vendor robot description not found")

SLEEP = {"base_yaw": 1.6, "base_pitch": -92.7, "elbow_pitch": -98.0, "wrist_roll": -3.6, "wrist_pitch": 16.2}


@pytest.fixture(scope="module")
def lamp():
    return LampModel(ROBOT_DIR)


def test_neutral_head_is_up_and_looks_forward(lamp):
    h = lamp.head(lamp.neutral)
    assert 0.30 < h["position"][2] < 0.35                    # about 32 cm above the base origin
    assert h["forward"][1] > 0.99                            # facing +y, level
    assert h["down"][2] < -0.99 and h["right"][0] > 0.99     # picture down = world down, right = +x
    assert lamp.problems(lamp.neutral) == []


def test_picture_motion_matches_what_was_measured_on_the_real_lamp(lamp):
    """Measured 2026-09-19 at neutral: base_yaw +7.4 units moved the scene by -0.083 of the picture width,
    wrist_pitch +8.1 units moved it by -0.085 of the picture height. Signs must agree, sizes roughly."""
    far_point = lamp.target_point(lamp.neutral, (0.5, 0.5), 2.0)
    for joint, step, axis, measured in (("base_yaw", 7.4, 0, -0.083), ("wrist_pitch", 8.1, 1, -0.085)):
        nudged = dict(lamp.neutral)
        nudged[joint] += step
        shift = lamp.project(nudged, far_point)[axis] - 0.5
        assert np.sign(shift) == np.sign(measured), f"{joint}: model says {shift:+.3f}, lamp did {measured:+.3f}"
        assert 0.85 < shift / measured < 1.18, f"{joint}: model says {shift:+.3f}, lamp did {measured:+.3f}"


def test_uses_the_lamps_own_servo_calibration(lamp):
    assert lamp.scale_source.startswith("servo calibration"), lamp.scale_source
    # head tilt is the joint the vendor's single scale gets most wrong: 95.5 degrees over 200 units
    assert abs(math.degrees(lamp._scale["wrist_pitch"]) - 0.478) < 0.01


def test_a_sight_line_stops_at_the_table(lamp):
    looking_down = dict(lamp.neutral, wrist_pitch=90.0)
    point = lamp.target_point(looking_down, (0.5, 0.5), 3.0)          # "3 m away" straight into the table
    assert point[2] >= lamp.table_z + 0.049
    level = lamp.target_point(lamp.neutral, (0.5, 0.5), 3.0)           # level gaze: nothing to clip
    assert np.linalg.norm(level - lamp.head(lamp.neutral)["position"]) == pytest.approx(3.0)


def test_project_inverts_ray(lamp):
    for where in ((0.5, 0.5), (0.2, 0.7), (0.9, 0.1)):
        point = lamp.target_point(lamp.neutral, where, 0.6)
        assert np.allclose(lamp.project(lamp.neutral, point), where, atol=1e-6)


def test_poses_that_put_the_head_into_the_table_or_the_base_are_refused(lamp):
    """Found by sweeping the joint range: leaning the whole arm forward (base_pitch strongly positive)
    is what drives the head into the table or into the lamp's own base."""
    into_table = dict(lamp.neutral, base_pitch=90.0, elbow_pitch=-25.0, wrist_pitch=0.0)
    into_base = dict(lamp.neutral, base_pitch=90.0, elbow_pitch=-75.0, wrist_pitch=-25.0)
    assert any("table" in p for p in lamp.problems(into_table)), lamp.problems(into_table)
    assert any("base" in p for p in lamp.problems(into_base)), lamp.problems(into_base)
    assert any("outside" in p for p in lamp.problems(dict(lamp.neutral, base_yaw=99.0)))
    # the vendor's own folded sleep pose is geometrically fine (we only keep a wider margin on joint limits)
    assert not [p for p in lamp.problems(SLEEP) if "table" in p or "base" in p]


@pytest.mark.parametrize("point", [(0.25, 0.55, 0.30), (-0.30, 0.50, 0.15), (0.0, 0.60, 0.45), (0.35, 0.35, 0.05)])
def test_look_at_faces_the_point_from_an_allowed_pose(lamp, point):
    pose, report = lamp.look_at(point)
    assert set(pose) == set(JOINTS)
    assert lamp.problems(pose) == []
    assert report["aim_error_deg"] < 4.0, report
    assert 0.18 < report["head_height_m"] < 0.38, report


def test_look_at_never_returns_a_refused_pose(lamp):
    for point in ((0.0, 0.12, -0.08), (0.05, 0.05, -0.05), (0.0, -0.4, 0.0)):      # at the base, under the head, behind
        pose, _ = lamp.look_at(point)
        assert lamp.problems(pose) == []
