from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

import follow
from follow import recent_sightings
from sdk import SDKError
from spatial import JOINTS


def test_object_sightings_survive_frames_where_slow_detector_does_not_run():
    samples = [(10.0, 0.5, 0.5, 1.0, "person")]

    assert recent_sightings(samples, 10.2) == samples
    assert recent_sightings(samples, 13.49) == samples
    assert recent_sightings(samples, 13.5) == []


@pytest.mark.parametrize("kind", ["face", "hand"])
def test_fast_tracker_sightings_use_short_window(kind):
    samples = [(10.0, 0.5, 0.5, 1.0, kind)]

    assert recent_sightings(samples, 10.59) == samples
    assert recent_sightings(samples, 10.61) == []


def run_follow(monkeypatch, *, interval=None, move_error=None, sightings=100,
               dry_run=False, hot=False):
    """Exercise the real entry point with a 10 Hz camera and a two-second SDK.

    All I/O, tracker inference, threads and the clock are injected. Synthetic
    geometry only exercises control flow; it does not validate a physical pose.
    """
    state = SimpleNamespace(now=100.0, frames=[], starts=[], ends=[], measured=[])

    def newest(after, timeout):
        state.now = max(state.now + 0.1, after + 0.001)
        return np.zeros((2, 2, 3), dtype=np.uint8), state.now

    camera = mock.Mock()
    camera.newest.side_effect = newest
    tracker = mock.Mock()

    def locate(_frame):
        state.frames.append(state.now)
        return (0.7, 0.5, 0.2) if len(state.frames) <= sightings else None

    tracker.locate.side_effect = locate
    model = mock.Mock()
    model.scale_source, model.table_z = "synthetic fixture", 0.0
    model.limits = {joint: (-94.0, 94.0) for joint in JOINTS}
    model.head.return_value = {"forward": np.array([0., 1., 0.]), "position": np.array([0., 0., .3])}
    model.distance_from_size.return_value = 1.0
    model.target_point.return_value = np.array([0.3, 1., .3])
    model.aim_error_deg.return_value = 20.0
    model.problems.return_value = []
    model.look_at.return_value = ({joint: 40.0 for joint in JOINTS}, {"rejected": []})

    sdk = mock.Mock()
    sdk.capabilities.return_value = {}

    def joints():
        state.measured.append(state.now)
        return {"self_collision_check": True, "units": "normalized_m100_100",
                "joints": dict.fromkeys(JOINTS, {}), "positions": dict.fromkeys(JOINTS, 0.0)}

    sdk.joints.side_effect = joints

    def move(pose):
        assert set(pose) == set(JOINTS)
        assert all(abs(value) <= 20.0 for value in pose.values())
        state.starts.append(state.now)
        state.now += 2.0
        state.ends.append(state.now)
        if move_error is not None:
            raise move_error
        return {"result": {"duration_seconds": 2.0, "reached": True}}

    sdk.move.side_effect = move
    thermal = mock.Mock()
    thermal.too_hot.return_value = hot
    thermal.celsius.return_value, thermal.peak_c = 40.0, 40.0
    thermal.frame_gap.side_effect = lambda tracking: .10 if tracking else .25

    for name, value in {"LampModel": model, "LampSDK": sdk, "Camera": camera,
                        "FaceTracker": tracker, "Thermal": thermal}.items():
        monkeypatch.setattr(follow, name, mock.Mock(return_value=value))
    monkeypatch.setattr(follow, "read_token", lambda: "test-token")
    monkeypatch.setattr(follow.threading, "Thread", mock.Mock())
    monkeypatch.setattr(follow.signal, "signal", mock.Mock())
    monkeypatch.setattr(follow.os, "nice", mock.Mock())
    monkeypatch.setattr(follow.time, "monotonic", lambda: state.now)
    argv = ["follow.py", "--target", "face", "--keep-idle", "--seconds", "8"]
    if interval is not None:
        argv += ["--min-interval", str(interval)]
    if dry_run:
        argv.append("--dry-run")
    monkeypatch.setattr(follow.sys, "argv", argv)
    follow.main()
    state.sdk, state.camera = sdk, camera
    return state


@pytest.mark.parametrize("interval", [None, 0, 0.3, 0.6, 1.5])
def test_fresh_sightings_overlap_the_post_move_admission_pause(monkeypatch, interval):
    state = run_follow(monkeypatch, interval=interval)
    pause = .3 if interval is None else max(.3, interval)
    assert len(state.starts) >= 2
    for ended, started in zip(state.ends, state.starts[1:]):
        # Three NEW detections, but not a pause followed by a second acquisition wait.
        assert sum(ended < stamp <= started for stamp in state.frames) >= 3
        assert pause <= started - ended <= max(pause, .45) + .11
        assert any(ended < stamp < ended + pause for stamp in state.frames)
    assert state.camera.running is False


@pytest.mark.parametrize("interval", [-1, float("nan"), float("inf")])
def test_invalid_pause_is_rejected_before_sdk_access(monkeypatch, interval):
    with pytest.raises(SystemExit) as exc:
        run_follow(monkeypatch, interval=interval)
    assert exc.value.code == 2
    follow.LampSDK.assert_not_called()


@pytest.mark.parametrize("code", ["not_reached", "lost_track", "timeout", "canceled"])
def test_timing_change_never_retries_an_uncertain_or_incomplete_move(monkeypatch, code):
    state = run_follow(monkeypatch, move_error=SDKError(409, code, "test failure"))
    assert len(state.starts) == 1
    assert state.camera.running is False


def test_two_detections_are_not_enough_to_move(monkeypatch):
    state = run_follow(monkeypatch, sightings=2)
    assert state.starts == []


@pytest.mark.parametrize("option", ["dry_run", "hot"])
def test_dry_run_and_thermal_cutoff_never_move(monkeypatch, option):
    state = run_follow(monkeypatch, **{option: True})
    assert state.starts == []


def test_stage_timings_are_reported_without_claiming_capture_latency(monkeypatch, capsys):
    run_follow(monkeypatch)
    output = capsys.readouterr().out
    for label in ("received_frame_age_ms=", "vision_ms=", "planning_ms=", "sdk_wait_ms=2000"):
        assert label in output
