"""The lamp twin's body: kinematics, clearances, look-at and pictures, on the vendor's real robot description.

Every test here needs the vendor's robot description (FTM_ROBOT_DIR), the lamp's servo calibration
(FTM_CALIBRATION) and MuJoCo; neither file is in this repository, so without them the tests are skipped.
"""
import json
import math
import time

import numpy as np
import pytest

from conftest import CALIBRATION, ROBOT_DIR, needs_robot
from twin.contract import JOINTS, Check, Person, Pose

mujoco = pytest.importorskip("mujoco")

# The poses below were found with the lamp's own calibration; without it the scales differ.
pytestmark = [needs_robot, pytest.mark.skipif(not CALIBRATION.exists(),
                                              reason="lamp servo calibration not found (set FTM_CALIBRATION)")]

# A crouched pose far from neutral, used as a seed. A test input of ours, not vendor data.
FOLDED = {"base_yaw": 0.0, "base_pitch": -90.0, "elbow_pitch": -90.0, "wrist_roll": 0.0, "wrist_pitch": 15.0}


@pytest.fixture(scope="module")
def lamp():
    from twin.model import LampTwin
    twin = LampTwin(ROBOT_DIR, CALIBRATION)
    yield twin
    twin.close()


def moved(lamp, **changes):
    return dict(lamp.neutral, **changes)


# ------------------------------------------------------------------ the model is the vendor's, read right
def test_table_is_the_underside_of_the_base_plate(lamp):
    """z = 0 is the table: the lowest vertex of the base plate mesh sits at z = 0 in the base frame, and no
    part of the base (plate and turning base) goes below it. The vendor's base cylinder (z_min -0.10) is
    a collision envelope, not the table."""
    lamp.check(lamp.neutral)                                   # runs the kinematics
    plate = [g for g, link in lamp._geom_link.items() if link == 0]
    base = [g for g, link in lamp._geom_link.items() if link <= 1]
    lowest_plate = min(lamp._world_vertices(g, lamp.data)[:, 2].min() for g in plate)
    lowest_base = min(lamp._world_vertices(g, lamp.data)[:, 2].min() for g in base)
    assert abs(lowest_plate) < 1e-4, lowest_plate
    assert lowest_base >= -1e-4, lowest_base
    assert lamp.table_z == 0.0


def test_uses_the_lamps_own_servo_calibration(lamp):
    assert lamp.scale_source.startswith("servo calibration"), lamp.scale_source
    # head tilt is the joint the vendor's single approximate scale gets most wrong: it overstates it ~1.5x
    mapping = json.loads((ROBOT_DIR / "joint_mapping.yaml").read_text())
    approximate = mapping["motion_scale"] * mapping["degrees_to_radians"]
    assert lamp.scale["wrist_pitch"] < 0.8 * approximate
    assert lamp.radians(lamp.neutral)["wrist_pitch"] == pytest.approx(
        mapping["joint_offsets"][mapping["motor_to_joint"]["wrist_pitch"]])
    pose = dict(lamp.neutral, wrist_pitch=lamp.neutral["wrist_pitch"] + 10.0)
    assert lamp.radians(pose)["wrist_pitch"] - lamp.radians(lamp.neutral)["wrist_pitch"] == pytest.approx(
        10.0 * lamp.scale["wrist_pitch"])
    for j in JOINTS:                                           # units <-> radians round-trip
        assert lamp.units(lamp.radians(pose))[j] == pytest.approx(pose[j])


def test_falls_back_to_the_vendor_scale_without_a_calibration(tmp_path):
    from twin.model import LampTwin
    twin = LampTwin(ROBOT_DIR, tmp_path / "missing.json")
    assert twin.scale_source.startswith("vendor approximate"), twin.scale_source
    assert len(set(twin.scale.values())) == 1


def test_neutral_head_is_up_and_looks_forward_level(lamp):
    link = lamp.head_link(lamp.neutral)
    assert 0.30 < link[2] < 0.35, link                         # the wrist-pitch axis, about 32 cm up
    pose = lamp.head(lamp.neutral)
    assert isinstance(pose, Pose)
    assert pose.forward[1] > 0.99                              # facing +y
    assert abs(pose.forward[2]) < math.sin(math.radians(2))    # level within 2 deg
    assert pose.down[2] < -0.99 and pose.right[0] > 0.99       # picture down = world down, right = +x
    for axis in (pose.forward, pose.down, pose.right):
        assert np.linalg.norm(axis) == pytest.approx(1.0)
    assert np.cross(pose.right, pose.down) @ pose.forward == pytest.approx(1.0)
    # the lens sits on top of the shade, a few centimetres above and in front of the wrist axis
    assert 0.05 < pose.position[2] - link[2] < 0.10
    assert 0.04 < (pose.position - link) @ pose.forward < 0.10


def test_camera_is_the_front_of_the_camera_mesh(lamp):
    """The optical centre is the lens (front face) of the vendor's camera mesh, and the camera looks along
    the head link's +x: the mesh's square board lies across that axis and its lens protrudes along it."""
    assert lamp.camera_source.startswith("camera mesh")
    lamp.check(lamp.neutral)
    camera = [g for g, link in lamp._geom_link.items() if link == 5 and lamp._mesh_name[g] == "camera"]
    assert len(camera) == 1
    head = lamp.data.xpos[lamp._head_body]
    rot = lamp.data.xmat[lamp._head_body].reshape(3, 3)
    local = (lamp._world_vertices(camera[0], lamp.data) - head) @ rot
    extent = local.max(0) - local.min(0)
    assert extent[0] < extent[1] and extent[0] < extent[2]     # thin along x: the board faces +x
    assert abs(extent[1] - extent[2]) < 0.003                  # a square board, not rolled
    back = local[local[:, 0] < local[:, 0].min() + 0.003]
    front = local[local[:, 0] > local[:, 0].max() - 0.003]
    assert np.ptp(front[:, 1]) < 0.6 * np.ptp(back[:, 1])      # the small part (the lens) is at the front
    assert lamp.camera_local[0] == pytest.approx(local[:, 0].max())


# ------------------------------------------------------------------ check()
def test_neutral_is_allowed_with_room_to_spare(lamp):
    c = lamp.check(lamp.neutral)
    assert isinstance(c, Check) and c.ok and c.reasons == []
    assert c.min_table_m > lamp.table_margin
    assert c.min_base_m > lamp.base_margin
    assert c.min_self_m > lamp.self_margin


def test_leaning_into_the_table_is_refused_for_the_table(lamp):
    c = lamp.check(moved(lamp, base_pitch=90.0, elbow_pitch=-25.0, wrist_pitch=0.0))
    assert not c.ok and c.min_table_m < 0
    assert any("table" in r for r in c.reasons), c.reasons


def test_a_head_near_the_table_is_refused_even_where_the_sdk_rule_lets_it_through(lamp):
    """Found by sweeping the joints: leaning forward with the elbow folded puts the head into the table
    in front of the lamp. The vendor planner only checks the head against a cylinder on the base plate,
    so it would plan this; the twin's client-side check must refuse it."""
    pose = moved(lamp, base_pitch=0.0, elbow_pitch=-60.0, wrist_pitch=0.0)
    c = lamp.check(pose)
    assert [r for r in c.reasons if "table" in r] and c.min_table_m < 0
    assert lamp.sdk_head_base_clear(pose)


def test_folding_back_onto_the_base_is_refused_for_the_base(lamp):
    pose = moved(lamp, base_pitch=-75.0, elbow_pitch=-90.0, wrist_pitch=75.0)
    c = lamp.check(pose)
    assert not c.ok and c.min_base_m < lamp.base_margin
    assert any("base" in r for r in c.reasons) and not any("table" in r for r in c.reasons), c.reasons


def test_self_clearance_is_reported_between_non_adjacent_links(lamp):
    """Inside the calibrated range the links never touch (found by sweeping: the closest is the head to the
    upper arm with the head tilted fully down, a few millimetres). A stricter self margin must turn that
    into a readable reason."""
    from twin.model import LampTwin
    pose = moved(lamp, wrist_roll=-90.0, wrist_pitch=90.0)
    c = lamp.check(pose)
    assert 0.0 < c.min_self_m < 0.012 and c.ok
    strict = LampTwin(ROBOT_DIR, CALIBRATION, self_margin=c.min_self_m + 0.002)
    reasons = strict.check(pose).reasons
    assert any("apart" in r for r in reasons), reasons


def test_joint_limits_are_enforced_with_a_margin(lamp):
    c = lamp.check(moved(lamp, base_yaw=99.0))
    assert not c.ok and any("base_yaw" in r and "outside" in r for r in c.reasons)
    assert lamp.limits["base_yaw"] == (-94.0, 94.0)


def test_check_distances_are_exact(lamp):
    """The bounding-sphere pruning in check() must give the same minimum as every mesh pair."""
    rng = np.random.default_rng(3)
    fromto = np.zeros(6)
    for _ in range(25):
        pose = {j: float(rng.uniform(-94, 94)) for j in JOINTS}
        c = lamp.check(pose)
        for kind, got in (("base", c.min_base_m), ("self", c.min_self_m)):
            every = min(mujoco.mj_geomDistance(lamp.model, lamp.data, int(a), int(b), 10.0, fromto)
                        for a, b in lamp._pairs[kind])
            assert got == pytest.approx(every, abs=1e-9)


# ------------------------------------------------------------------ the head camera picture
def test_picture_shift_matches_what_was_measured_on_the_lamp(lamp):
    """Measured 2026-09-19 at neutral: base_yaw +7.4 units moved the scene -0.083 of the picture width,
    wrist_pitch +8.1 units moved it -0.085 of the picture height. The model must agree within 20%."""
    pose = lamp.head(lamp.neutral)
    far = pose.position + pose.forward * 2.5                   # a point across the room, picture centre
    for joint, step, axis, measured in (("base_yaw", 7.4, 0, -0.083), ("wrist_pitch", 8.1, 1, -0.085)):
        shift = lamp.project(moved(lamp, **{joint: lamp.neutral[joint] + step}), far)[axis] - 0.5
        assert 0.8 < shift / measured < 1.2, f"{joint}: model {shift:+.3f}, lamp {measured:+.3f}"


def test_project_inverts_ray(lamp):
    for where in ((0.5, 0.5), (0.2, 0.7), (0.9, 0.1)):
        point = lamp.head(lamp.neutral).position + lamp.ray(lamp.neutral, where) * 0.8
        assert np.allclose(lamp.project(lamp.neutral, point), where, atol=1e-9)
    behind = lamp.head(lamp.neutral).position - np.array([0.0, 1.0, 0.0])
    assert lamp.project(lamp.neutral, behind) is None


# ------------------------------------------------------------------ look_at()
REACHABLE = [(0.25, 0.55, 0.30), (-0.30, 0.50, 0.15), (0.0, 0.60, 0.45), (0.35, 0.35, 0.05),
             (0.0, 1.20, 0.46), (0.6, 1.0, 0.45), (-0.8, 1.5, 0.50), (0.8, 0.8, 1.0), (0.0, 2.2, 1.0),
             (-0.5, 0.6, 0.9), (0.9, 0.4, 0.45)]


@pytest.mark.parametrize("point", REACHABLE)
def test_look_at_faces_a_reachable_point_from_an_allowed_pose(lamp, point):
    pose, report = lamp.look_at(point)
    assert set(pose) == set(JOINTS)
    assert lamp.check(pose).ok, lamp.check(pose).reasons
    assert report["aim_error_deg"] < 4.0, report
    assert report["aim_error_deg"] == pytest.approx(lamp.aim_error_deg(pose, point))
    assert report["fallback"] == "none", report
    assert pose["wrist_roll"] == lamp.neutral["wrist_roll"]
    assert abs(pose["base_yaw"] - lamp.neutral["base_yaw"]) * math.degrees(lamp.scale["base_yaw"]) <= 70 + 1e-6


@pytest.mark.parametrize("point", [(0.0, 0.12, -0.08), (0.05, 0.05, -0.05), (0.0, -0.4, 0.0), (0.0, 0.0, 0.05),
                                   (0.0, -0.6, 0.4), (0.0, 0.2, -0.3), (0.02, 0.0, 0.9), (0.0, 0.0, 0.0)])
def test_look_at_never_returns_a_refused_pose_for_an_unreachable_point(lamp, point):
    """Behind the lamp, under the table, inside the base, straight above."""
    for seed in (None, lamp.neutral, FOLDED):
        pose, report = lamp.look_at(point, seed=seed)
        assert lamp.check(pose).ok, (point, seed, report)
        assert math.isfinite(report["aim_error_deg"])


def test_look_at_never_returns_a_refused_pose_anywhere(lamp):
    rng = np.random.default_rng(7)
    seed = None
    for _ in range(60):
        point = rng.uniform((-2.0, -1.5, -0.8), (2.0, 3.0, 2.0))
        pose, report = lamp.look_at(point, seed=seed)
        assert lamp.check(pose).ok, (point, report)
        seed = pose if rng.uniform() < 0.7 else {j: float(rng.uniform(-100, 100)) for j in JOINTS}


def test_look_at_is_continuous_for_a_moving_listener(lamp):
    """A listener walking across in front of the lamp in 2 cm steps: each pose starts from the last one and
    must stay close to it (no jumps between solution branches), and keep facing them."""
    seed, worst = None, 0.0
    for x in np.arange(-0.9, 0.9001, 0.02):
        point = (float(x), 1.0, 0.45)
        pose, report = lamp.look_at(point, seed=seed)
        assert report["aim_error_deg"] < 1.0 and report["fallback"] == "none", report
        if seed is not None:
            worst = max(worst, max(abs(pose[j] - seed[j]) for j in JOINTS))
        seed = pose
    assert worst < 3.0, worst


def test_look_at_reports_what_the_tracker_needs(lamp):
    pose, report = lamp.look_at((0.2, 0.9, 0.45))
    for key in ("aim_error_deg", "head_height_m", "camera_height_m", "distance_m", "fallback", "rejected"):
        assert key in report
    assert 0.20 - 0.01 <= report["head_height_m"] <= 0.36 + 0.01
    assert report["distance_m"] == pytest.approx(np.linalg.norm(np.array((0.2, 0.9, 0.45)) -
                                                                 lamp.head(pose).position))


# ------------------------------------------------------------------ speed
def test_head_and_check_are_fast_enough_for_30_hz(lamp):
    rng = np.random.default_rng(11)
    poses = [{j: float(rng.uniform(-94, 94)) for j in JOINTS} for _ in range(200)]
    start = time.perf_counter()
    for pose in poses:
        lamp.head(pose)
    head_ms = (time.perf_counter() - start) / len(poses) * 1e3
    start = time.perf_counter()
    for pose in poses:
        lamp.check(dict(pose, base_yaw=pose["base_yaw"] + 1e-6))   # defeat the same-pose cache
    check_ms = (time.perf_counter() - start) / len(poses) * 1e3
    print(f"\nhead() {head_ms:.3f} ms, check() {check_ms:.3f} ms per call")
    assert head_ms < 1.0 and check_ms < 5.0                    # a 30 Hz step has 33 ms


# ------------------------------------------------------------------ pictures
def test_render_orbit_has_the_right_shape_and_is_not_blank(lamp):
    img = lamp.render(lamp.neutral, width=320, height=240)
    assert img.shape == (240, 320, 3) and img.dtype == np.uint8
    assert img.std() > 10


def test_render_light_tints_the_shade(lamp):
    """The diffuser glows in the panel's colour: a blue panel makes the lamp's face bluer than a red one."""
    view = dict(orbit=(-90.0, -5.0, 0.8), width=320, height=240)
    blue = lamp.render(lamp.neutral, light_rgb=(0.05, 0.2, 0.9), **view).astype(float)
    red = lamp.render(lamp.neutral, light_rgb=np.tile((0.9, 0.1, 0.05), (93, 1)), **view).astype(float)
    off = lamp.render(lamp.neutral, light_rgb=None, **view).astype(float)
    changed = np.abs(blue - red).sum(axis=2) > 60
    assert changed.sum() > 500                                   # the shade's face, not one pixel
    assert (blue[changed, 2] - blue[changed, 0]).mean() > (red[changed, 2] - red[changed, 0]).mean() + 50
    assert np.abs(off - blue).sum(axis=2).max() > 60


def test_head_camera_picture_agrees_with_project(lamp):
    """A person drawn in the head camera's picture lands where project() says, whatever the picture size."""
    person = Person("p", head=np.array([0.25, 1.1, 0.55]), facing=np.array([0.0, -1.0, 0.0]))
    pose = moved(lamp, base_yaw=10.0, wrist_pitch=20.0)
    expected = lamp.project(pose, person.head)
    for width, height in ((640, 480), (400, 400)):
        img = lamp.render(pose, people=[person], camera="head", width=width, height=height)
        assert img.shape == (height, width, 3) and img.std() > 10
        renderer = mujoco.Renderer(lamp.model, height=height, width=width)
        renderer.enable_segmentation_rendering()
        option = mujoco.MjvOption()
        option.geomgroup[:] = [1, 1, 1, 0, 0, 0]
        renderer.update_scene(lamp._render_data, camera="head_camera", scene_option=option)
        ids = renderer.render()[:, :, 0]
        renderer.close()
        skull = mujoco.mj_name2id(lamp.model, mujoco.mjtObj.mjOBJ_GEOM, "person0_skull")
        face = mujoco.mj_name2id(lamp.model, mujoco.mjtObj.mjOBJ_GEOM, "person0_face")
        rows, cols = np.nonzero((ids == skull) | (ids == face))
        centre = ((cols.min() + cols.max() + 1) / 2 / width, (rows.min() + rows.max() + 1) / 2 / height)
        assert centre == pytest.approx(expected, abs=0.01), (centre, expected)
        # and its apparent width matches a pinhole of the same field of view
        size = (cols.max() - cols.min() + 1) / width
        depth = (person.head - lamp.head(pose).position) @ lamp.head(pose).forward
        assert size == pytest.approx(2 * person.head_radius * lamp.fx / depth, rel=0.1)


# ------------------------------------------------------------------ gravity sag
def test_gravity_sag_follows_the_urdf_masses(lamp):
    """GravitySag: the level at the reference lock pose, scaled per pose by the gravity torque of the vendor
    URDF's link masses, in the direction gravity pulls. At the lamp's lock poses that direction droops the
    camera on all three loaded joints (the old constant model's signs partly cancelled)."""
    from twin.model import GravitySag
    from twin.tracking import _az_el
    levels = {"base_pitch": 1.5, "elbow_pitch": 6.0, "wrist_pitch": 2.0}
    sag = GravitySag(lamp, levels)
    at_ref = sag.at(sag.reference)
    for j, level in levels.items():
        assert abs(at_ref[j]) == pytest.approx(level, rel=1e-9)
    assert at_ref["base_yaw"] == 0.0 and at_ref["wrist_roll"] == 0.0
    tau = lamp.gravity_torque(sag.reference)
    for j in levels:                                     # gravity pulls toward -sign(torque) units
        assert np.sign(at_ref[j]) == -np.sign(tau[j])
    drooped = {j: sag.reference[j] + at_ref[j] for j in sag.reference}
    assert _az_el(lamp.head(drooped).forward)[1] < _az_el(lamp.head(sag.reference).forward)[1] - 3.0
    standing, _ = lamp.look_at(np.array([0.0, 1.4, 1.0]), seed=dict(lamp.neutral))
    assert abs(sag.at(standing)["base_pitch"]) < abs(at_ref["base_pitch"])     # less reach, less load
    assert not np.any(GravitySag(lamp, {})(np.array([standing[j] for j in JOINTS])))
