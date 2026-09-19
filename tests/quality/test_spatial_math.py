"""Asset-independent geometry checks, written 2026-09-19; not robot validation."""

import math

import numpy as np
import pytest

from spatial import LampModel, _rot_z, _transform, _wrap


@pytest.mark.parametrize("angle", [-math.pi, -0.3, 0, 0.8, math.pi])
def test_rotation_preserves_lengths_and_orientation(angle):
    rotation = _rot_z(angle)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(4), atol=1e-14)
    assert np.linalg.det(rotation[:3, :3]) == pytest.approx(1)


def test_transform_translation_and_inverse():
    transform = _transform([0.2, -0.4, 0.6], [0.3, -0.2, 0.5])
    np.testing.assert_allclose(transform @ [0, 0, 0, 1], [0.2, -0.4, 0.6, 1])
    np.testing.assert_allclose(np.linalg.inv(transform) @ transform, np.eye(4), atol=1e-14)


@pytest.mark.parametrize("angle", [-20, -math.pi, 0, math.pi, 20])
def test_wrap_preserves_direction(angle):
    wrapped = _wrap(angle)
    assert -math.pi <= wrapped < math.pi
    assert math.sin(wrapped) == pytest.approx(math.sin(angle), abs=1e-14)
    assert math.cos(wrapped) == pytest.approx(math.cos(angle), abs=1e-14)


@pytest.fixture
def camera(monkeypatch):
    # A synthetic camera frame exercises projection only, without loading or
    # imitating private vendor geometry/calibration.
    model = LampModel.__new__(LampModel)
    model.fx, model.fy = 0.8, 1.1
    monkeypatch.setattr(model, "head", lambda _: {
        "position": np.array([0.1, 0.2, 0.3]),
        "forward": np.array([0, 1, 0]),
        "right": np.array([1, 0, 0]),
        "down": np.array([0, 0, -1]),
    })
    return model


@pytest.mark.parametrize("where", [(0, 0), (0.2, 0.7), (0.5, 0.5), (1, 1)])
def test_projection_round_trip_without_vendor_assets(camera, where):
    target = camera.target_point({}, where, 0.6)
    np.testing.assert_allclose(camera.project({}, target), where, atol=1e-14)


def test_points_behind_camera_are_not_projected(camera):
    assert camera.project({}, [0.1, -1, 0.3]) is None


@pytest.mark.parametrize("point,expected", [([0.1, 1.2, 0.3], 0), ([0.1, -0.8, 0.3], 180)])
def test_aim_error_distinguishes_forward_from_backward(camera, point, expected):
    assert camera.aim_error_deg({}, point) == pytest.approx(expected)
