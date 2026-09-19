"""Original tiny fixture, not the vendor's model. MuJoCo is an explicit test dependency."""
import shutil

import pytest

from dance.urdf_replay import load_model, replay

pytest.importorskip("mujoco", reason="install mujoco==3.13.0 for URDF replay checks")


@pytest.fixture
def robot(tmp_path):
    path = tmp_path / "robot.urdf"
    path.write_text('''<robot name="test">
      <link name="base"/>
      <link name="arm">
        <inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
        <collision><geometry><sphere radius="0.1"/></geometry></collision>
      </link>
      <joint name="hinge" type="revolute">
        <parent link="base"/><child link="arm"/><origin xyz="0 0 1"/>
        <axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="1" velocity="1"/>
      </joint>
    </robot>''')
    return path


def samples(value=0.2):
    return [{"time_s": 0, "positions_rad": {"hinge": 0}},
            {"time_s": 1, "positions_rad": {"hinge": value}}]


def test_explicitly_limited_to_kinematics(robot):
    report = replay(robot, samples())
    assert report["status"] == "PASS_SAMPLED_KINEMATICS_ONLY"
    assert report["hardware_approved"] is False
    assert report["dynamics_verified"] is False
    assert report["actuator_count"] == 0
    assert report["max_sample_gap_s"] == 1


def test_out_of_limit_trajectory_fails(robot):
    report = replay(robot, samples(2))
    assert report["status"] == "FAIL"
    assert report["joint_limit_samples"] == 1


def test_collision_free_model_is_not_a_validation_model(robot):
    robot.write_text(robot.read_text().replace(
        '<collision><geometry><sphere radius="0.1"/></geometry></collision>', '',
    ))
    with pytest.raises(ValueError, match="collision-enabled"):
        replay(robot, samples())


def test_fixed_model_is_not_an_articulated_robot(robot):
    robot.write_text('<robot name="empty"><link name="base"/></robot>')
    with pytest.raises(ValueError, match="articulated"):
        replay(robot, [{"time_s": 0, "positions_rad": {}},
                       {"time_s": 1, "positions_rad": {}}])


def test_penetration_is_reported_not_suppressed_as_baseline(robot):
    robot.write_text(robot.read_text().replace(
        '<link name="base"/>',
        '<link name="base"><collision><geometry><sphere radius="1.1"/></geometry></collision></link>',
    ))
    report = replay(robot, samples())
    assert report["status"] == "FAIL"
    assert report["collision_samples"] == 2
    assert report["min_penetration_distance_m"] < 0
    assert report["contact_body_pairs"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "0", pytest.param(10**1000, id="huge-int")])
def test_invalid_positions_fail_closed(robot, bad):
    with pytest.raises(ValueError):
        replay(robot, samples(bad))


def test_repeated_time_and_missing_joint_rejected(robot):
    trajectory = samples()
    trajectory[1]["time_s"] = 0
    with pytest.raises(ValueError):
        replay(robot, trajectory)
    trajectory = samples()
    trajectory[1]["positions_rad"] = {}
    with pytest.raises(ValueError):
        replay(robot, trajectory)


def test_unrepresentable_time_rejected(robot):
    trajectory = samples()
    trajectory[1]["time_s"] = 10**1000
    with pytest.raises(ValueError, match="time_s"):
        replay(robot, trajectory)


def test_mesh_digest_binds_bytes_to_their_references(robot):
    tetrahedron = "v 0 0 0\nv {size} 0 0\nv 0 {size} 0\nv 0 0 {size}\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
    first, second = robot.parent / "first.obj", robot.parent / "second.obj"
    small, large = tetrahedron.format(size="0.1"), tetrahedron.format(size="0.2")
    first.write_text(small)
    second.write_text(large)
    robot.write_text(robot.read_text().replace(
        '<collision><geometry><sphere radius="0.1"/></geometry></collision>',
        '<collision><geometry><mesh filename="first.obj"/></geometry></collision>'
        '<collision><origin xyz="1 0 0"/><geometry><mesh filename="second.obj"/></geometry></collision>',
    ))
    _, before = load_model(robot)
    first.write_text(large)
    second.write_text(small)
    _, after = load_model(robot)
    assert before != after, "swapping mesh contents changes geometry even when the hash multiset is unchanged"
    copied = robot.parent / "relocated"
    copied.mkdir()
    for asset in (robot, first, second):
        shutil.copyfile(asset, copied / asset.name)
    assert load_model(copied / robot.name)[1] == after
