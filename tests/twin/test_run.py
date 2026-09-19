"""twin/run.py: the whole twin on one clock, its scores, and its evidence in the team replay schema.

The pure parts (scores, replay reports, song and budget helpers) run without the robot description. The
end-to-end runs need FTM_ROBOT_DIR (and mujoco); the video test also needs Pillow and ffmpeg.
"""
import json
import math
import shutil

import numpy as np
import pytest

from conftest import needs_robot

from twin import run as R
from twin.tracking import ACQUIRE, LOCK, LOST, SEARCH

DT = 1.0 / 30


# ------------------------------------------------------------------ pure helpers
def test_motion_budget_leaves_room_for_light():
    assert R.motion_cap_for(30) == 26            # motion spec section 8: settle tracking needs <= 26 at 30/min
    assert R.motion_cap_for(120) == 40           # tracker default when the lamp's 120/min is in effect
    assert R.motion_cap_for(30) + R.LIGHT_RESERVE_PER_MIN <= 30


def test_song_label_keeps_no_local_path():
    assert R.song_label("synthetic") == "synthetic"
    label = R.song_label("runlog:/data/someone/runs/2026-09-19T0910Z/run.jsonl")
    assert label == "runlog:2026-09-19T0910Z/run.jsonl" and "someone" not in label


def test_synthetic_song_is_cut_to_the_run():
    song = R.load_song("synthetic", 20.0, seed=0)
    assert song.events and all(0.0 <= e.t < 20.0 for e in song.events)
    with pytest.raises(ValueError):
        R.load_song("mp3:nothing", 10.0)


def test_max_in_60s_matches_a_sliding_window():
    assert R._max_in_60s([]) == 0
    assert R._max_in_60s([0.0, 30.0, 59.9, 60.0, 61.0]) == 4    # 0.0 has left the window at 60.0
    assert R._max_in_60s(list(np.arange(0, 120, 2.0))) == 30


def _steps(pattern):
    """(state, pid, visible) per step from [(n, state, pid, visible ids)]."""
    state, pid, visible = [], [], []
    for n, s, p, v in pattern:
        state += [s] * n
        pid += [p] * n
        visible += [tuple(v)] * n
    return state, pid, visible


def test_tracking_metrics_from_ground_truth():
    state, pid, visible = _steps([(30, SEARCH, "", ()), (30, ACQUIRE, "p1", ("p1",)), (60, LOCK, "p1", ("p1", "p2")),
                                  (30, LOCK, "p2", ("p1", "p2")),          # switched while p1 still in the picture
                                  (30, LOST, "p2", ()), (30, SEARCH, "", ()), (30, LOCK, "p1", ("p1",))])
    n = len(state)
    t = np.arange(n) * DT
    err = np.full(n, np.nan)
    err[30:60] = 12.0
    err[60:90] = 3.0                                              # half the first lock within 5 deg ...
    err[90:120] = 7.0                                             # ... the other half within 10
    err[120:150] = 4.0
    err[210:240] = 20.0
    present = np.full(n, 2)
    m = R.tracking_metrics(t, state, pid, err, visible, present, DT)
    assert m["t_first_acquire_s"] == pytest.approx(30 * DT)
    assert m["t_first_lock_s"] == pytest.approx(60 * DT)
    assert m["person_switches"] == 2                               # p1 -> p2, then p2 -> p1 after SEARCH
    assert m["lock_steals"] == 1                                   # only the first: p2 was not visible at the second
    assert m["lost_episodes"] == 1
    lock_err = err[[k for k in range(n) if state[k] == LOCK]]
    assert m["share_of_lock_on_axis"]["within_5_deg"] == pytest.approx(round(float(np.mean(lock_err <= 5)), 4))
    assert m["share_of_lock_on_axis"]["within_10_deg"] == pytest.approx(round(float(np.mean(lock_err <= 10)), 4))
    assert m["locked_s_by_person"] == {"p1": pytest.approx(90 * DT, abs=1e-3), "p2": pytest.approx(30 * DT, abs=1e-3)}


def test_tracking_metrics_empty_room():
    n = 60
    m = R.tracking_metrics(np.arange(n) * DT, [SEARCH] * n, [""] * n, np.full(n, np.nan), [()] * n,
                           np.zeros(n, dtype=int), DT)
    assert m["t_first_lock_s"] is None and m["t_first_person_in_room_s"] is None
    assert m["share_of_presence_on_axis"]["within_5_deg"] is None
    assert m["lock_steals"] == 0


def _report(n=40, *, valid=True, mode="head-follow"):
    t = np.arange(n) * DT
    return R.replay_report(t, np.full(n, 0.05), np.linspace(1.0, 4.0, n), np.zeros(n, bool), np.zeros(n, bool),
                           np.full(n, valid), mode=mode, scene_id="twin:test", model_sha256="a" * 64,
                           trajectory_sha256="b" * 64, simulator_version="test")


def test_replay_report_has_the_exact_schema():
    rep = _report()
    assert set(rep) == {"schema_version", "scene_id", "model_sha256", "trajectory_sha256", "simulator_version",
                        "mode", "completed", "sample_count", "duration_ns", "min_clearance_m",
                        "max_head_error_deg", "collision_count", "joint_limit_violations", "tracking_lost_count",
                        "samples"}
    times = [s["time_ns"] for s in rep["samples"]]
    assert all(type(x) is int for x in times) and all(b > a for a, b in zip(times, times[1:], strict=False))
    assert all(set(s) == {"time_ns", "clearance_m", "head_error_deg", "collision", "joint_limit_violation",
                          "tracking_valid"} for s in rep["samples"])
    assert rep["duration_ns"] == times[-1] - times[0]
    assert rep["max_head_error_deg"] == max(s["head_error_deg"] for s in rep["samples"])
    json.dumps(rep, allow_nan=False)
    with pytest.raises(ValueError):
        R.replay_report(np.arange(3) * DT, [0.1, math.nan, 0.1], [1, 1, 1], [False] * 3, [False] * 3, [True] * 3,
                        mode="dance", scene_id="x", model_sha256="a" * 64, trajectory_sha256="b" * 64,
                        simulator_version="x")


def test_replay_reports_pass_and_fail_the_team_validator():
    validator, how = R.load_replay_validator()
    if validator is None:
        pytest.skip("team replay validator not loadable (no git or no origin/codexfranklin/unified-app)")
    kw = dict(model_sha256="a" * 64, trajectory_sha256="b" * 64, min_clearance=R.REPLAY_MIN_CLEARANCE_M,
              max_head_error=R.REPLAY_MAX_HEAD_ERROR_DEG, max_sample_gap_ns=math.ceil(DT * 1e9))
    good = _report()
    assert validator.validate_report(good, expected_mode="head-follow", expected_duration_ns=good["duration_ns"],
                                     **kw)["status"] == "PASS_SIMULATION_ONLY"
    lost = _report(valid=False)
    verdict = validator.validate_report(lost, expected_mode="head-follow", expected_duration_ns=lost["duration_ns"],
                                        **kw)
    assert verdict["status"] == "FAIL"
    # the look-around is judged as a dance: tracking is not gated there
    dance = _report(valid=False, mode="dance")
    assert validator.validate_report(dance, expected_mode="dance", expected_duration_ns=dance["duration_ns"],
                                     **kw)["status"] == "PASS_SIMULATION_ONLY"


@pytest.mark.parametrize("n, k_acq, k_lock", [(900, 60, 190), (900, None, None), (900, 60, None), (900, 0, 70),
                                              (900, 1, 70), (900, 60, 61), (900, 60, 899), (2, None, None)])
def test_replay_segments_cover_every_step_once(n, k_acq, k_lock):
    """The review found 6.4 s of acquire (and a whole negative control) in no report: never again."""
    segs = R.replay_segments(n, k_acq, k_lock)
    assert segs[0][1] == 0 and segs[-1][2] == n
    assert all(a[2] == b[1] for a, b in zip(segs, segs[1:], strict=False))                 # no gap, no overlap
    assert all(b - a >= 2 for _, a, b, _ in segs)                             # the schema needs two samples
    assert sum(b - a for _, a, b, _ in segs) == n
    if k_lock is not None and n - k_lock >= 2:
        assert segs[-1][0] == "lock" and segs[-1][3] == "head-follow" and segs[-1][1] == k_lock


def test_run_modules_are_what_a_run_imports():
    """replay/manifest.json hashes RUN_MODULES: every twin module the simulation imports, and not the HTTP
    simulator (the review found sim_sdk.py's hash in the evidence although a run never executes it)."""
    pytest.importorskip("mujoco")
    import subprocess
    import sys
    code = ("import sys; import twin.run, twin.model; "
            "print(sorted(m.split('.')[1] for m in sys.modules if m.startswith('twin.')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         cwd=str(R.ROOT)).stdout
    imported = set(eval(out))                                                  # a list of names, printed above
    assert imported <= set(R.RUN_MODULES) and "sim_sdk" not in R.RUN_MODULES and "video" not in R.RUN_MODULES


def test_short_runs_are_marked_not_steady_state():
    assert R.DEFAULT_SECONDS >= 2 * 60.0 == R.STEADY_STATE_S


# ------------------------------------------------------------------ end to end (needs the robot description)
@pytest.fixture(scope="module")
def kin():
    pytest.importorskip("mujoco")
    from twin.model import LampTwin
    return LampTwin()


def _strip_wall(report: dict) -> dict:
    out = json.loads(json.dumps(report))
    for k in ("loop_wall_s", "total_wall_s"):
        out["run"].pop(k, None)
    return out


@needs_robot
def test_walk_in_sit_locks_and_writes_valid_evidence(kin, tmp_path):
    # no gravity sag here: this checks the pipeline end to end; the sag's effect is the sweep's subject
    res = R.simulate("walk_in_sit", "settle", "synthetic", 20.0, 0, tmp_path / "run", video=False, kin=kin,
                     sag="none")
    rep = res.report
    tr = rep["tracking"]
    assert tr["t_first_lock_s"] is not None and tr["t_first_lock_s"] > tr["t_first_person_in_room_s"]
    assert tr["lock_error_deg"]["p95"] < R.REPLAY_MAX_HEAD_ERROR_DEG
    assert tr["lock_steals"] == 0
    # the pose the arm is really in stays clear of the table, the base and itself
    assert rep["safety"]["steps_in_contact"] == 0
    assert rep["safety"]["min_table_m"] >= 0.03
    # one SDK session: motion and light never exceed the vendor default together
    assert rep["run"]["rate_limit_per_min"] == 30
    assert rep["sdk"]["session_window_max_used"] <= 30
    assert rep["sdk"]["motion_refused"] == {}
    assert rep["sdk"]["light_plan_unchanged_by_real_motion_times"]
    # every modelled number says where it comes from
    assert rep["sources"] and all(s["source"] for s in rep["sources"])
    # evidence files
    out = tmp_path / "run"
    for name in ("report.json", "trajectory.json", "replay/manifest.json", "replay/search.report.json",
                 "replay/acquire.report.json", "replay/lock.report.json"):
        assert (out / name).is_file(), name
    manifest = (out / "replay" / "manifest.json").read_text()
    assert str(kin.robot_dir) not in manifest                # vendor files by relative name only
    # the manifest hashes the modules a run executes, not the HTTP simulator
    hashed = {f["path"] for f in json.loads(manifest)["twin_sources"]}
    assert "twin/sim_sdk.py" not in hashed and {"twin/run.py", "twin/motion.py", "twin/tracking.py"} <= hashed
    # the replay reports cover every step of the run, with no gap and no overlap
    cov = rep["replay"]["coverage"]
    assert cov["samples"] == cov["steps"] == rep["run"]["steps"]
    segs = cov["segments"]
    assert segs[0]["start"] == 0 and segs[-1]["stop"] == rep["run"]["steps"]
    assert all(a["stop"] == b["start"] for a, b in zip(segs, segs[1:], strict=False))
    lock = rep["replay"]["lock"]
    assert lock["written"] and lock["mode"] == "head-follow"
    assert R._sha256_file(out / "trajectory.json") == rep["replay"]["trajectory_sha256"]
    assert R._sha256_file(out / "replay" / "manifest.json") == rep["replay"]["model_sha256"]
    if "verdict_in_process" in lock:
        assert lock["verdict_in_process"]["status"] == "PASS_SIMULATION_ONLY"
        assert rep["replay"]["search"]["verdict_in_process"]["status"] == "PASS_SIMULATION_ONLY"
    # light: the ideal design pairs every hit, the SDK path cannot (the spec's key answer)
    ls = rep["light_haptics_summary"]
    assert ls["ideal"]["hits_with_light"] == ls["ideal"]["hits_scored"]
    assert ls["sdk_best"]["hits_with_light"] < ls["sdk_best"]["hits_scored"] / 4
    assert ls["ideal"]["governor_ok"] and ls["sdk_best"]["governor_ok"]


@needs_robot
def test_runs_are_deterministic(kin):
    a = R.simulate("two_people", "settle", "synthetic", 8.0, 3, None, video=False, kin=kin)
    b = R.simulate("two_people", "settle", "synthetic", 8.0, 3, None, video=False, kin=kin)
    assert _strip_wall(a.report) == _strip_wall(b.report)
    assert np.array_equal(a.steps["measured"], b.steps["measured"])


@needs_robot
def test_preempt_is_the_negative_control(kin):
    """Motion spec section 13: pre-empting a moving target restarts a 2 s plan from rest every time."""
    settle = R.simulate("sway_to_music", "settle", "synthetic", 16.0, 0, None, video=False, kin=kin).report
    preempt = R.simulate("sway_to_music", "preempt", "synthetic", 16.0, 0, None, video=False, kin=kin).report
    assert settle["tracking"]["t_first_lock_s"] is not None
    assert preempt["tracking"]["t_first_lock_s"] is None
    assert preempt["sdk"]["motion_posts"] > settle["sdk"]["motion_posts"]


@needs_robot
def test_empty_room_keeps_searching_and_never_locks(kin):
    rep = R.simulate("empty", "settle", "synthetic", 12.0, 0, None, video=False, kin=kin).report
    assert rep["tracking"]["t_first_acquire_s"] is None
    assert set(rep["tracking"]["state_time_s"]) == {"SEARCH"}
    assert rep["sdk"]["commands_by_kind"]["clip"] >= 1          # one long look-around clip, one rate slot


@needs_robot
def test_video_is_written(kin, tmp_path):
    pytest.importorskip("PIL")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")
    res = R.simulate("two_people", "settle", "synthetic", 3.0, 0, tmp_path / "v", video=True, video_fps=5, kin=kin)
    video = tmp_path / "v" / "video.mp4"
    assert video.is_file() and video.stat().st_size > 10_000
    assert res.report["video"]["frames"] == 16
