"""Checks of twin/motion.py, the model of the vendor SDK's motion path. Every expected number below comes from
the rules the model cites (vendor source file:line in twin/motion.py), not from invented data. Only the
vendor-idle test needs the vendor's robot package (FTM_ROBOT_DIR); the rest run anywhere."""
import math

import numpy as np
import pytest

from conftest import ROBOT_DIR, needs_robot
from twin.contract import JOINTS
from twin.motion import (COMPUTE_BASE_S, COMPUTE_ROW_S, MAX_VELOCITY, SAG_SIGN, RateWindow, SDKMotionModel,
                         clip_arbitration_delay_s, clip_csv, clip_start_delay_s, compute_s, ease, load_vendor_idle,
                         move_start_delay_s, plan_seconds, synthetic_idle, target_plan)

# A gaze-like pose for the tests: base_pitch and elbow in the region where the lamp looks at people, not the
# crouch. These are test inputs, not vendor data.
GAZE = {"base_yaw": 0.0, "base_pitch": -45.0, "elbow_pitch": -20.0, "wrist_roll": 0.0, "wrist_pitch": 30.0}
FRAME = 1.0 / 30.0


def model(**kw):
    kw.setdefault("idle", None)
    return SDKMotionModel(GAZE, **kw)


def run(m, t_from, t_to, dt=0.005):
    """Step the model; returns [(t, MotionState)] and every (action_id, outcome) seen."""
    states, finished = [], []
    for t in np.arange(t_from, t_to + 1e-9, dt):
        s = m.step(float(t))
        states.append((float(t), s))
        finished += s.finished
    return states, finished


def yaw_to(value):
    return {**GAZE, "base_yaw": value}


# ------------------------------------------------------------------ the plan
def test_plan_length_and_waypoints_follow_the_vendor_formula():
    for delta in (0.0, 10.0, 27.0, 100.0, 144.0):
        assert plan_seconds(delta) == pytest.approx(2.0)
    assert plan_seconds(200.0) == pytest.approx(200.0 / 72.0)
    times, positions = target_plan(GAZE, {"base_yaw": 20.0})
    assert len(times) == 61 and times[-1] == pytest.approx(2.0)
    u = np.arange(61) / 60
    assert np.allclose(positions[:, 0], 20.0 * ease(u))
    assert np.allclose(positions[:, 1:], [GAZE[j] for j in JOINTS[1:]])      # unnamed joints keep their value
    assert np.max(np.diff(positions[:, 0]) / np.diff(times)) <= 0.5 * 0.9 * MAX_VELOCITY + 1e-9
    assert float(ease(0.25)) == pytest.approx(0.1035, abs=1e-3) and float(ease(0.5)) == pytest.approx(0.5)


# ------------------------------------------------------------------ a single move
def test_a_20_unit_move_takes_at_least_2_s_and_follows_the_quintic():
    m = model(sag={})
    result = m.submit_move(0.0, yaw_to(20.0))
    assert result.accepted and result.planned_duration_s == pytest.approx(2.0)
    states, finished = run(m, 0.0, 4.0, dt=0.001)
    a = m.actions[result.action_id]
    assert finished == [(result.action_id, "succeeded")]

    # POST -> freeze = HTTP + pose read + a collision pass over the plan's 61 rows; freeze -> first waypoint =
    # 2 frames + read + the re-plan's pass + read + 1 frame. A pass costs a base + a cost per row.
    assert a.rows == 61 and compute_s(61) == pytest.approx(COMPUTE_BASE_S + 61 * COMPUTE_ROW_S)
    assert a.t_arbitrated == pytest.approx(0.015 + 0.008 + compute_s(61))
    assert a.t_plan_start - a.t_arbitrated == pytest.approx(m.preroll_s)
    assert m.preroll_s == pytest.approx(2 / 30 + 0.016 + compute_s(61) + 1 / 30)
    assert a.t_plan_start == pytest.approx(move_start_delay_s(), abs=1e-6)
    assert a.t_last_write - a.t_plan_start == pytest.approx(2.0)
    assert a.t_finished - a.t_post >= 2.0

    # Nothing is written before the live frame; then every goal is the quintic, held between 30 Hz writes.
    for t, s in states:
        if t < a.t_live_frame:
            assert s.commanded == GAZE
    plan_t = a.plan_times[1:]
    for i in (1, 10, 30, 45, 60):
        s = next(s for t, s in states if t >= plan_t[i] + 1e-6)
        assert s.commanded["base_yaw"] == pytest.approx(20.0 * float(ease(i / 60)), abs=1e-9)
    s = next(s for t, s in states if t >= plan_t[30] + 0.02)      # between writes the goal is held
    assert s.commanded["base_yaw"] == pytest.approx(10.0)
    assert abs(states[-1][1].measured["base_yaw"] - 20.0) <= 2.0


# ------------------------------------------------------------------ pre-emption
def test_preempting_at_half_a_second_freezes_the_goal_and_replans_from_the_measured_pose():
    m = model(sag={})
    first = m.submit_move(0.0, yaw_to(27.0))
    run(m, 0.0, 0.5, dt=0.001)
    second = m.submit_move(0.5, yaw_to(27.0))
    states, finished = run(m, 0.5, 3.5, dt=0.001)
    a, b = m.actions[first.action_id], m.actions[second.action_id]
    assert (first.action_id, "canceled") in finished and a.reason == "cancelled_or_preempted"
    assert (second.action_id, "succeeded") in finished

    # Progress when frozen: the first plan's waypoints written before the second move's arbitration.
    written = math.floor((b.t_arbitrated - a.t_plan_start) * 30 + 1e-9) + 1       # waypoints 0..k
    frozen = 27.0 * float(ease((written - 1) / 60))
    during = [s.commanded["base_yaw"] for t, s in states if b.t_arbitrated <= t < b.t_live_frame]
    assert during and np.allclose(during, frozen, atol=1e-9)                      # no ramp, no motion
    # The spec's continuous-time bound (motion spec section 6): below ease((T - dead time)/2).
    progress = frozen / 27.0
    dead = move_start_delay_s() - m.admission_delay_s                  # freeze -> first waypoint of a plan
    assert float(ease((0.5 - dead - FRAME) / 2)) - 1e-9 <= progress <= float(ease((0.5 - dead) / 2))

    # The new plan starts from the MEASURED pose (a read after the freeze), not from the commanded goal.
    assert b.plan_positions[1, 0] == b.replan_from["base_yaw"]
    assert abs(b.replan_from["base_yaw"] - frozen) < 0.15                         # the servo nearly caught up
    first_step = b.plan_positions[2, 0] - b.plan_positions[1, 0]                  # zero velocity at restart:
    assert first_step == pytest.approx((27.0 - b.replan_from["base_yaw"]) * float(ease(1 / 60)))   # a fresh quintic
    assert first_step < 0.01
    assert b.t_plan_start - b.t_arbitrated == pytest.approx(m.preroll_s)


def test_preempting_every_half_second_is_a_negative_control():
    m = model(sag={})
    ids = []
    for k in range(10):                                    # POST at 0, 0.5, ... 4.5 s toward the same target
        run(m, 0.5 * k, 0.5 * k, dt=1.0)
        ids.append(m.submit_move(0.5 * k, yaw_to(27.0)).action_id)
    states, finished = run(m, 4.5, 5.0, dt=0.001)
    for prev, nxt in zip(ids, ids[1:], strict=False):
        a, b = m.actions[prev], m.actions[nxt]
        assert a.outcome == "canceled"
        start = a.replan_from["base_yaw"]
        frozen = float(a.plan_positions[a.plan_times <= b.t_arbitrated][-1, 0])
        k = math.floor((b.t_arbitrated - a.t_plan_start) * 30 + 1e-9)
        assert (frozen - start) / (27.0 - start) == pytest.approx(float(ease(k / 60)), abs=1e-9)
        assert 0.03 < (frozen - start) / (27.0 - start) < 0.045                   # spec: about 4.5% per interval
    yaw = states[-1][1].measured["base_yaw"]
    assert 0.1 * 27.0 < yaw < 0.5 * 27.0                    # it moves, but after 5 s it is nowhere near
    assert m.stats()["posted"] == 10                        # and it spent 10 actions of the budget doing so


# ------------------------------------------------------------------ the rate limit and capacity
def test_rate_limit_refuses_the_next_post_within_60_s():
    m = model(sag={}, rate_limit_per_min=30)
    for k in range(30):
        run(m, 0.5 * k, 0.5 * k, dt=1.0)
        assert m.submit_move(0.5 * k, yaw_to(float(k % 2))).accepted
    run(m, 15.0, 15.0, dt=1.0)
    refused = m.submit_move(15.0, yaw_to(5.0))
    assert not refused.accepted and refused.reason.startswith("429") and refused.action_id == ""
    assert not m.submit_move(15.5, yaw_to(5.0)).accepted
    run(m, 15.5, 60.0, dt=0.1)
    assert not m.submit_move(60.0, yaw_to(5.0)).accepted   # the stamp at 0.0 is still inside [0, 60]
    # Refused requests took no slot: only the stamp at 0.0 has left the window at 60.25.
    assert m.submit_move(60.25, yaw_to(5.0)).accepted
    assert not m.submit_move(60.3, yaw_to(5.0)).accepted


def test_admission_refusals_still_use_a_rate_slot():
    m = model(sag={}, rate_limit_per_min=3)
    bad = m.submit_move(0.0, yaw_to(150.0))                # outside the calibrated range: 422 after the slot
    assert not bad.accepted and bad.reason.startswith("422") and bad.action_id
    assert not m.submit_move(0.1, {"no_such_joint": 1.0}).accepted
    assert m.submit_move(0.2, yaw_to(5.0)).accepted
    assert m.submit_move(0.3, yaw_to(5.0)).reason.startswith("429")


def test_capacity_refuses_a_fifth_task_in_flight():
    m = model(sag={})
    results = [m.submit_move(0.002 * k, yaw_to(float(k))) for k in range(5)]
    assert [r.accepted for r in results] == [True, True, True, True, False]
    assert results[4].reason.startswith("503")
    _, finished = run(m, 0.01, 5.0)
    assert [o for _, o in finished].count("canceled") == 3 and finished[-1][1] == "succeeded"
    assert m.submit_move(5.0, yaw_to(0.0)).accepted


# ------------------------------------------------------------------ idle
def idle_model(**kw):
    return SDKMotionModel(GAZE, idle="synthetic", **kw)


def test_idle_resumes_after_a_success_and_pulls_the_head_down():
    m = idle_model()
    assert m.step(0.3).idle_playing
    move = m.submit_move(0.3, GAZE)
    states, finished = run(m, 0.3, 6.0)
    a = m.actions[move.action_id]
    assert finished == [(move.action_id, "succeeded")]
    assert not any(s.idle_playing for t, s in states if a.t_arbitrated <= t < a.t_finished)
    resumed = [t for t, s in states if t >= a.t_finished and s.idle_playing]
    assert resumed and resumed[0] - a.t_finished < 0.05
    pitch = [s.commanded["base_pitch"] for t, s in states if t >= a.t_finished]
    assert min(pitch) < GAZE["base_pitch"] - 10.0          # the crouched idle drags the head toward the table


def test_idle_stays_paused_after_cancel_preemption_failure_and_rejection():
    # canceled by the client
    m = idle_model()
    move = m.submit_move(0.0, yaw_to(20.0))
    run(m, 0.0, 1.0)
    m.cancel(1.0, move.action_id)
    states, finished = run(m, 1.0, 6.0)
    assert finished == [(move.action_id, "canceled")]
    assert not any(s.idle_playing for t, s in states)

    # pre-empted: the first is canceled, idle comes back only after the second succeeds
    m = idle_model()
    first = m.submit_move(0.0, yaw_to(20.0))
    run(m, 0.0, 1.0)
    second = m.submit_move(1.0, yaw_to(-20.0))
    states, finished = run(m, 1.0, 5.0)
    assert finished == [(first.action_id, "canceled"), (second.action_id, "succeeded")]
    done = m.actions[second.action_id].t_finished
    assert not any(s.idle_playing for t, s in states if t < done)
    assert any(s.idle_playing for t, s in states if t > done)

    # failed: the elbow sags past its 10-unit tolerance, settle times out after 1.5 s
    m = idle_model(sag={"elbow_pitch": 12.0})
    move = m.submit_move(0.0, GAZE)
    states, finished = run(m, 0.0, 6.0)
    a = m.actions[move.action_id]
    assert finished == [(move.action_id, "failed")] and "elbow_pitch" in a.reason
    assert a.t_finished == pytest.approx(a.t_last_write + 1.5)
    assert not any(s.idle_playing for t, s in states if t > a.t_arbitrated)

    # rejected: the pose passed admission but the re-plan's collision check refuses it
    allowed = {"ok": True}
    m = idle_model(pose_ok=lambda pose: allowed["ok"])
    move = m.submit_move(0.0, yaw_to(20.0))
    assert move.accepted
    allowed["ok"] = False
    states, finished = run(m, 0.0, 4.0)
    assert finished == [(move.action_id, "rejected")] and "head could hit the base" in m.actions[move.action_id].reason
    assert not any(s.idle_playing for t, s in states if t > m.admission_delay_s + 0.01)


# ------------------------------------------------------------------ gravity sag
def test_sag_holds_each_loaded_joint_off_its_goal_the_way_gravity_pulls_and_the_move_succeeds():
    m = model()                                                    # default sag (ASSUMPTION values)
    target = {"base_yaw": 15.0, "base_pitch": -40.0, "elbow_pitch": -10.0, "wrist_roll": 5.0, "wrist_pitch": 40.0}
    move = m.submit_move(0.0, target)
    states, finished = run(m, 0.0, 4.0)
    a = m.actions[move.action_id]
    assert finished == [(move.action_id, "succeeded")]
    end = states[-1][1]
    assert end.commanded == pytest.approx(target)
    levels = {"base_pitch": 1.5, "elbow_pitch": 6.0, "wrist_pitch": 2.0}            # motion.SAG ("assumed")
    expected_sag = {j: -SAG_SIGN.get(j, -1.0) * levels.get(j, 0.0) for j in JOINTS}    # commanded - measured
    assert expected_sag["elbow_pitch"] > 0 > expected_sag["base_pitch"]              # the elbow folds down
    for j in JOINTS:
        assert end.commanded[j] - end.measured[j] == pytest.approx(expected_sag[j], abs=0.05 + 1e-9)
        assert a.settle_errors[j] <= m.tolerances[j]
    # Sag past a tolerance fails the same move: the tolerances bound what the sag can be.
    m = model(sag={"elbow_pitch": 10.5})
    move = m.submit_move(0.0, target)
    _, finished = run(m, 0.0, 5.0)
    assert finished == [(move.action_id, "failed")]


def test_partial_joint_sets_ratchet_the_loaded_joints_and_full_sets_do_not():
    m = model(sag={"elbow_pitch": 6.0})
    for k in range(4):                                     # name only base_yaw: the elbow goal = its sagged reading
        m.submit_move(3.0 * k, {"base_yaw": 5.0 * (k % 2)})
        run(m, 3.0 * k, 3.0 * k + 2.99, dt=0.01)
    assert m.step(12.0).commanded["elbow_pitch"] == pytest.approx(GAZE["elbow_pitch"] - 4 * 6.0, abs=0.2)
    m = model(sag={"elbow_pitch": 6.0})
    for k in range(4):
        m.submit_move(3.0 * k, yaw_to(5.0 * (k % 2)))
        run(m, 3.0 * k, 3.0 * k + 2.99, dt=0.01)
    assert m.step(12.0).commanded["elbow_pitch"] == pytest.approx(GAZE["elbow_pitch"])


# ------------------------------------------------------------------ clips
def sweep_clip(start_yaw, end_yaw, seconds, rate_hz=10.0):
    n = int(seconds * rate_hz)
    return [(k / rate_hz, yaw_to(start_yaw + (end_yaw - start_yaw) * k / n)) for k in range(n + 1)]


def test_clip_entry_blend_is_continuous_and_lasts_at_least_2_s():
    m = model(sag={})
    frames = sweep_clip(30.0, -30.0, 3.0)                  # starts 30 units from the arm, 20 units/s
    result = m.submit_clip(0.0, frames)
    assert result.accepted and result.planned_duration_s == pytest.approx(2.0 + 3.0)
    states, finished = run(m, 0.0, 6.0, dt=0.001)
    a = m.actions[result.action_id]
    assert finished == [(result.action_id, "succeeded")]
    assert a.t_finished == a.t_last_write                  # no settle check for clips

    goals = a.plan_positions[1:, 0]                        # the executed plan after the live frame
    entry = 61                                             # 2.0 s at 30 Hz: 61 waypoints
    assert goals[0] == a.replan_from["base_yaw"]           # starts at the measured pose
    assert goals[entry - 1] == pytest.approx(30.0)         # ends exactly on the clip's first frame
    steps = np.abs(np.diff(goals))                         # no jump anywhere: the entry peaks at 1.875*30/2 u/s,
    assert np.max(steps) <= 1.875 * 30.0 / 2.0 * FRAME + 1e-6      # the clip runs at 20 u/s
    assert steps[entry - 1] <= 20.0 * FRAME + 1e-6         # and the junction is one ordinary clip step
    assert a.plan_times[entry] - a.t_plan_start == pytest.approx(2.0)

    # The goal the servo sees, sampled at 30 Hz, never jumps either.
    sampled = np.array([s.commanded["base_yaw"] for t, s in states[::33]])
    assert np.max(np.abs(np.diff(sampled))) < 1.0

    # A clip that starts exactly where the arm is still gets a 2.0 s entry.
    m = model(sag={})
    result = m.submit_clip(0.0, sweep_clip(0.0, 10.0, 1.0))
    assert result.planned_duration_s == pytest.approx(2.0 + 1.0)


def test_clip_upload_refusals_take_no_rate_slot():
    m = model(sag={})
    too_fast = [(0.0, yaw_to(0.0)), (0.1, yaw_to(40.0))]  # 400 units/s
    result = m.submit_clip(0.0, too_fast)
    assert not result.accepted and "velocity" in result.reason and m.rate.used(0.0) == 0
    partial = [(0.0, {"base_yaw": 0.0}), (1.0, {"base_yaw": 1.0})]
    assert not m.submit_clip(0.0, partial).accepted and m.rate.used(0.0) == 0
    assert m.submit_clip(0.0, sweep_clip(0.0, 10.0, 1.0)).accepted and m.rate.used(0.0) == 1


# ------------------------------------------------------------------ housekeeping
def test_results_do_not_depend_on_how_finely_the_caller_steps():
    def scenario(dt):
        m = idle_model()
        out = []
        for t_post, target in ((0.2, yaw_to(20.0)), (1.0, yaw_to(-10.0)), (4.0, GAZE)):
            run(m, 0.0 if not out else out[-1][0], t_post, dt=dt)
            out.append((t_post, m.submit_move(t_post, target)))
        states, finished = run(m, 4.0, 9.0, dt=dt)
        return finished, {round(t, 6): s.measured for t, s in states}, m.log

    fine, coarse = scenario(0.001), scenario(0.05)
    assert fine[0] == coarse[0] and fine[2] == coarse[2]
    common = set(fine[1]) & set(coarse[1])
    assert common and all(fine[1][t] == coarse[1][t] for t in common)


def test_time_cannot_go_backwards_and_a_shared_window_is_shared():
    window = RateWindow(per_minute=2)
    motion = SDKMotionModel(GAZE, idle=None, rate_window=window)
    assert window.take(0.0)                                 # e.g. a light.glow on the same session
    assert motion.submit_move(0.1, GAZE).accepted
    assert not motion.submit_move(0.2, GAZE).accepted
    motion.step(1.0)
    with pytest.raises(ValueError):
        motion.step(0.5)
    lamp_env = RateWindow(per_minute=120)                  # the lamp's .env value, once a restart applies it
    assert all(lamp_env.take(0.1 * k) for k in range(120)) and not lamp_env.take(12.0)


def test_synthetic_idle_loops_and_stays_crouched():
    times, positions = synthetic_idle()
    assert times[-1] == pytest.approx(75.5) and np.allclose(positions[0], positions[-1])
    assert np.all(positions[:, 1] <= -53.0) and np.all(positions[:, 2] <= -60.0)
    assert SDKMotionModel(GAZE, idle="synthetic").idle_source.startswith("synthetic")
    assert SDKMotionModel(GAZE, idle=None).idle_source.startswith("off")
    assert load_vendor_idle("/nonexistent") is None


@needs_robot
def test_vendor_idle_is_read_at_run_time_when_present():
    clip = load_vendor_idle(ROBOT_DIR)
    assert clip is not None
    times, positions = clip
    assert times[-1] == pytest.approx(75.5, abs=0.1)       # summary statistic in the twin motion spec, section 8
    assert np.all(positions[:, 1] < -50.0)                 # crouched: base_pitch stays far below a gaze pose
    m = SDKMotionModel(GAZE, idle="auto", robot_dir=ROBOT_DIR)
    assert m.idle_source.startswith("vendor")
    assert m.step(2.0).idle_playing and m.step(2.0).commanded["base_pitch"] < -80.0


# ------------------------------------------------------------------ compute, uploads and the clip store
def test_every_validation_pass_costs_time_per_row():
    """The review: compute was a flat 20 ms whatever the plan's length, and the upload free. Now every pass
    (upload, clip.play admission, re-plan) costs a base plus a cost per row (vendor checker timed on a Mac,
    x an ASSUMED Pi slowdown), and a clip's freeze and first waypoint come later the longer it is."""
    short, long_ = sweep_clip(0.0, 10.0, 2.0, rate_hz=30.0), sweep_clip(0.0, 10.0, 20.0, rate_hz=30.0)
    delays = []
    for frames in (short, long_):
        m = model(sag={})
        r = m.submit_clip(0.0, frames)
        run(m, 0.0, 30.0, dt=0.01)
        a = m.actions[r.action_id]
        assert r.accepted and r.clip_id and a.clip_id == r.clip_id
        rows = len(frames)
        assert a.t_arbitrated == pytest.approx(clip_arbitration_delay_s(rows, clip_bytes=len(clip_csv(frames))),
                                               abs=1e-6)
        assert a.t_plan_start == pytest.approx(clip_start_delay_s(rows, clip_bytes=len(clip_csv(frames))), abs=1e-6)
        assert r.t_posted is not None and r.t_posted > 0.0                   # clip.play follows the upload's reply
        delays.append(a.t_plan_start)
    assert delays[1] - delays[0] > 2 * 540 * COMPUTE_ROW_S                    # three passes over 540 more rows


def test_an_upload_holds_the_admission_lock_and_clip_play_takes_its_slot_after_it():
    m = model(sag={}, rate_limit_per_min=30)
    clip = m.submit_clip(0.0, sweep_clip(0.0, 10.0, 20.0, rate_hz=30.0))
    move = m.submit_move(0.01, yaw_to(5.0))                                  # waits for the upload's validation
    assert clip.accepted and move.accepted
    run(m, 0.01, 3.0, dt=0.01)
    up = m.actions[clip.action_id]
    assert m.actions[move.action_id].t_arbitrated > up.t_arbitrated         # admitted after the clip.play
    assert up.t_post == pytest.approx(clip.t_posted) and clip.t_posted > 0.3
    assert sorted(m.rate.stamps) == pytest.approx(sorted([clip.t_posted, 0.01]))


def test_the_clip_store_fills_and_deleting_frees_it():
    """Vendor clip_store.py:26-32: at most 100 files, persistent. A client that never deletes is refused."""
    m = model(sag={}, rate_limit_per_min=10 ** 6, store_initial_files=98, store_initial_bytes=98 * 3000)
    frames = sweep_clip(0.0, 10.0, 1.0, rate_hz=30.0)
    first = m.submit_clip(0.0, frames)
    second = m.submit_clip(5.0, frames)
    third = m.submit_clip(10.0, frames)
    assert first.accepted and second.accepted and len(m.store) == 100
    assert not third.accepted and "clip store is full" in third.reason and m.counters["store_full"] == 1
    assert m.rate.used(10.0) == 2                                            # a refused upload takes no slot
    done = m.delete_clip(12.0, first.clip_id)
    assert done >= 12.0 + m.http_s and len(m.store) == 99
    assert m.submit_clip(15.0, frames).accepted
    # the byte cap: stored bytes plus twice the new CSV must stay under 50 MiB
    m = model(sag={}, store_max_bytes=2 * len(clip_csv(frames)) + 1000)
    assert m.submit_clip(0.0, frames).accepted
    assert not m.submit_clip(5.0, frames).accepted


def test_a_sag_function_gives_a_pose_dependent_deflection():
    def droop(goal):                                    # deeper when the elbow is further out (test input)
        return np.array([0.0, 0.0, -0.1 * abs(goal[2]), 0.0, 0.0])
    m = model(sag=droop)
    move = m.submit_move(0.0, {**GAZE, "elbow_pitch": -40.0})
    run(m, 0.0, 4.0)
    end = m.step(4.0)
    assert m.actions[move.action_id].outcome == "succeeded"
    assert end.commanded["elbow_pitch"] - end.measured["elbow_pitch"] == pytest.approx(4.0, abs=0.06)
