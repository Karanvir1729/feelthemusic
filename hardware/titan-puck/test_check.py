"""Regression checks for the CAD validator; run with pytest in the CAD environment."""
import math
import os
import subprocess
import sys

import cadquery as cq
import pytest
import trimesh

import check


def test_failed_boolean_is_not_reported_as_zero_interference():
    class BrokenShape:
        def intersect(self, other):
            raise RuntimeError("CAD boolean failed")

    with pytest.raises(RuntimeError, match="CAD boolean failed"):
        check.vol(BrokenShape(), BrokenShape())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_invalid_boolean_volume_is_rejected(value):
    class InvalidIntersection:
        def intersect(self, other):
            return self

        def Volume(self):
            return value

        def isValid(self):
            return True

    with pytest.raises(check.CheckFailure, match="invalid intersection"):
        check.vol(InvalidIntersection(), InvalidIntersection())


def test_intersection_threshold_and_clear_separation():
    cube = cq.Workplane().box(1, 1, 1)
    assert check.interference("clear", cube, cube.translate((2, 0, 0))) == 0
    assert check.interference("small numerical touch", cube, cube.translate((0.996, 0, 0))) == pytest.approx(0.004)
    with pytest.raises(check.CheckFailure, match="exceeds"):
        check.interference("collision", cube, cube.translate((0.98, 0, 0)))


def test_disconnected_enclosure_fails():
    first = cq.Workplane().box(1, 1, 1).val()
    separated = cq.Compound.makeCompound([first, first.translate((2, 0, 0))])
    with pytest.raises(check.CheckFailure, match="one connected solid"):
        check.validate_solid("base", separated, single=True)


def test_clearance_requirement_is_enforced():
    cube = cq.Workplane().box(1, 1, 1)
    assert check.clearance("enough", cube, cube.translate((4, 0, 0)), 3) == pytest.approx(3)
    with pytest.raises(check.CheckFailure, match="below"):
        check.clearance("too close", cube, cube.translate((3.9, 0, 0)), 3)


@pytest.mark.parametrize("name,value", [("CHECK_MOTOR_D", "nan"), ("CHECK_MOTOR_D", "inf"),
                                        ("CHECK_MOTOR_D", "0"), ("CHECK_MOTOR_L", "-1"),
                                        ("CHECK_WAIST_D", "nan"), ("CHECK_WAIST_D", "0"),
                                        ("CHECK_WAIST_START", "0"), ("CHECK_WAIST_LEN", "-1")])
def test_invalid_motor_dimensions_fail(name, value):
    with pytest.raises(check.CheckFailure, match="finite and positive"):
        check.motor_dimensions({name: value})


@pytest.mark.parametrize("overrides", [
    {"CHECK_WAIST_D": "10.8", "CHECK_MOTOR_D": "10.8"},
    {"CHECK_WAIST_D": "11", "CHECK_MOTOR_D": "10.8"},
    {"CHECK_WAIST_START": "18", "CHECK_WAIST_LEN": "5.5", "CHECK_MOTOR_L": "23.5"},
    {"CHECK_WAIST_START": "19", "CHECK_WAIST_LEN": "5.5", "CHECK_MOTOR_L": "23.5"},
])
def test_invalid_waist_profile_fails(overrides):
    with pytest.raises(check.CheckFailure, match="waist"):
        check.motor_dimensions(overrides)


def test_candidate_dimensions_do_not_change_model():
    before = (check.M.MOTOR_D, check.M.MOTOR_L, check.M.MOTOR_Z, check.M.CRADLE_D,
              check.M.WAIST_D, check.M.WAIST_START, check.M.WAIST_LEN)
    small = check.motor_dimensions({"CHECK_MOTOR_D": "9.5", "CHECK_MOTOR_L": "22.5", "CHECK_WAIST_D": "8.1",
                                    "CHECK_WAIST_START": "8.6", "CHECK_WAIST_LEN": "5.5"})
    assert small == (9.5, 22.5, 8.1, 8.6, 5.5)
    assert check.motor_dimensions({}) == (check.M.MOTOR_D, check.M.MOTOR_L,
                                         check.M.WAIST_D, check.M.WAIST_START, check.M.WAIST_LEN)
    motors = check.motor_shapes(*small)
    volume = math.pi / 4 * (9.5 ** 2 * 22.5 - (9.5 ** 2 - 8.1 ** 2) * 5.5)
    for name, motor in motors.items():
        assert motor.val().Volume() == pytest.approx(volume)
        box = motor.val().BoundingBox()
        assert box.xlen == pytest.approx(9.5)
        assert (box.zlen if name == "LFi" else box.ylen) == pytest.approx(22.5)
        check.validate_solid(name, motor, single=True)
    assert before == (check.M.MOTOR_D, check.M.MOTOR_L, check.M.MOTOR_Z, check.M.CRADLE_D,
                      check.M.WAIST_D, check.M.WAIST_START, check.M.WAIST_LEN)


@pytest.fixture(scope="module")
def waist_capture_assembly():
    return check.M.base(), check.motor_shapes(*check.motor_dimensions({}))


def test_waist_shoulders_block_axial_travel_at_permitted_lift(waist_capture_assembly):
    base, motors = waist_capture_assembly
    for name in ("LF", "MF"):
        check.interference(f"{name} seated", base, motors[name])
    overlaps = check.axial_capture(base, motors)
    assert len(overlaps) == 4
    assert min(overlaps.values()) > check.INTERFERENCE_LIMIT


def test_axial_capture_fails_when_waist_collars_are_removed(waist_capture_assembly):
    base, motors = waist_capture_assembly
    for mx, my in (check.M.LF, check.M.MF):
        half_width = check.M.CRADLE_D / 2 + check.M.SADDLE_WALL + 0.1
        start = my + check.M.WAIST_START + check.M.COLLAR_END_GAP
        length = check.M.WAIST_LEN - 2 * check.M.COLLAR_END_GAP
        collar_region = cq.Solid.makeBox(2 * half_width, length, check.M.MOTOR_Z + 0.1,
                                        cq.Vector(mx - half_width, start, 0))
        base = base.cut(collar_region)
    with pytest.raises(check.CheckFailure, match="no shoulder/collar blocking"):
        check.axial_capture(base, motors)


def test_wrong_motor_for_selected_cradle_fails_cli(tmp_path):
    environ = dict(os.environ, MOTOR_D="9.5", CHECK_MOTOR_D="10.8", CHECK_MOTOR_L="23.5")
    environ.pop("TITAN_STEP", None)
    result = subprocess.run([sys.executable, check.__file__], cwd=tmp_path, env=environ,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "enclosure MOTOR_D=9.50 mm; motors checked as 10.80 mm" in result.stdout
    assert "FAIL: CheckFailure: LF x base:" in result.stdout
    assert not list(tmp_path.glob("*.stl"))


def test_open_and_disconnected_meshes_fail():
    cube = trimesh.creation.box()
    check.validate_mesh("cube", cube)
    opened = cube.copy()
    opened.update_faces(range(len(opened.faces) - 1))
    with pytest.raises(check.CheckFailure, match="watertight"):
        check.validate_mesh("open", opened)
    second = cube.copy()
    second.apply_translation((2, 0, 0))
    with pytest.raises(check.CheckFailure, match="one connected mesh"):
        check.validate_mesh("disconnected", trimesh.util.concatenate([cube, second]))


def test_printability_tessellates_current_part(tmp_path):
    # An unrelated/stale delivery STL must have no influence on this fresh check.
    (tmp_path / "puck_base.stl").write_text("stale file")
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    mesh = check.printability("puck_base", cq.Workplane().box(2, 3, 4), fresh)
    assert mesh.volume == pytest.approx(24)
    assert (tmp_path / "puck_base.stl").read_text() == "stale file"


def test_main_returns_failure_on_validation_error(monkeypatch, capsys):
    def failed_run():
        raise check.CheckFailure("interference")

    monkeypatch.setattr(check, "run", failed_run)
    assert check.main() == 1
    assert "FAIL: CheckFailure: interference" in capsys.readouterr().out


def test_main_returns_failure_on_unexpected_cad_error(monkeypatch, capsys):
    def failed_run():
        raise RuntimeError("boolean failed")

    monkeypatch.setattr(check, "run", failed_run)
    assert check.main() == 1
    assert "FAIL: RuntimeError: boolean failed" in capsys.readouterr().out
