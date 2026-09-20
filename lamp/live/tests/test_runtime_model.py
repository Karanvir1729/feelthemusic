"""runtime_model: the animation route offline. The pure pieces (the min-jerk weight, the quintic and
its clamps, the processing steps and their quirks), the skip rule, the blend window, and -- the point
of the file -- the traces of ground_truth_2026-09-20.md: a 1.9 s look that streams as a 20-unit bump,
a turn whose right swing never leaves the runtime, and a 5 s dance that streams in full. The section
"the dropped window and the skip rule, pinned" runs the four clips exactly as installed on the lamp."""
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime_model as rm  # noqa: E402

HOME = np.array([0.0, -49.0, -22.0, 0.0, 30.0])       # beat_clips.START, where every trace tonight began
SCRATCH = Path("/private/tmp/claude-501/-Users-meharkhanna-feelthemusic/a2b4cfc6-081d-4df6-b3d6-864bf6cf8fa1/scratchpad")
CALIBRATION = Path(os.environ.get("LELAMP_CALIBRATION_PATH", SCRATCH / "robotdesc/lelamp-calibration.json"))
HYPE_124_C = SCRATCH / "beat_clips/beat_hype_124_c.csv"  # the +-68 yaw version the note describes ("excursion to about +68")
WAVE_HI = SCRATCH / "wave/wave_hi.csv"
INSTALLED = {name: SCRATCH / rel for name, rel in (            # the four clips as they were on the lamp tonight
    ("cha_look_right", "cha/cha_look_right.csv"), ("cha_turn", "cha/cha_turn.csv"),
    ("beat_hype_124_c", "bc5/beat_hype_124_c.csv"), ("wave_hi", "wave/wave_hi.csv"))}
REST = np.array([1.5, -48.5, -23.2, 0.0, 30.0])         # the measured rest the traces started from (1.5 off HOME on yaw)
needs_clips = pytest.mark.skipif(not all(p.exists() for p in INSTALLED.values()), reason="needs the installed clips")
FPS = 30
BEAT = 60.0 / 124.0                                    # cha_moves.TEMPO_BPM


def ease(u):
    """beat_clips.ease / cha_moves PROFILES["ease"]: the cosine ease the Cha clips are built from."""
    u = np.clip(u, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * u)


def keyframed(beats, keys):
    """A cha_moves.Call: (beats, pose) keyframes from HOME, cosine-eased between, one frame per
    1/30 s, n = round(beats * BEAT * FPS) + 1 rows -- the shape cha_moves.Call.rows() writes."""
    n = int(round(beats * BEAT * FPS)) + 1
    t = np.linspace(0.0, beats, n)
    U = np.tile(HOME, (n, 1))
    prev_t, prev_p = 0.0, HOME
    for kt, kp in keys:
        kp = np.asarray(kp, float)
        where = (t >= prev_t - 1e-12) & (t <= kt + 1e-12)
        U[where] = prev_p + (kp - prev_p) * ease((t[where] - prev_t) / max(kt - prev_t, 1e-12))[:, None]
        prev_t, prev_p = kt, kp
    return U


def yawed(yaw, pose=HOME):
    p = np.array(pose, float)
    p[0] = yaw
    return p


def cha_look_right():
    """cha_moves.cha_look_right: 4 beats -- 0.75 up to +55, hold 2.5, 0.75 home. 59 frames, 1.93 s."""
    return keyframed(4, [(0.75, yawed(55.0)), (3.25, yawed(55.0)), (4.0, HOME)])


def cha_turn(first=+65.0):
    """cha_moves.cha_turn: 8 beats -- 1.5 to +65 on TURN_REACH, hold 0.5, 3.0 across to -65, hold 0.5,
    2.0 home, hold 0.5. 117 frames, 3.87 s. `first` flips the order for the check below."""
    reach = np.array([0.0, -30.0, -12.0, 0.0, 24.0])   # cha_moves.TURN_REACH
    a, b = yawed(first, reach), yawed(-first, reach)
    return keyframed(8, [(1.5, a), (2.0, a), (5.0, b), (5.5, b), (7.5, HOME), (8.0, HOME)])


def hold_then_move(hold_s, rise_s, target_yaw, tail_s):
    """A beat_clips-style clip: hold at HOME, cosine rise on base_yaw, hold at the target."""
    n_hold, n_rise, n_tail = int(round(hold_s * FPS)), int(round(rise_s * FPS)), int(round(tail_s * FPS))
    rise = HOME + (yawed(target_yaw) - HOME) * ease(np.linspace(0, 1, n_rise + 1))[:, None]
    return np.vstack([np.tile(HOME, (n_hold, 1)), rise, np.tile(yawed(target_yaw), (n_tail, 1))])


def yaw(streamed):
    return streamed.positions[:, 0]


# ----------------------------------------------------------------------------- the pure pieces
def test_minimum_jerk_weight_endpoints_midpoint_and_clamp():
    """w(t) = 10t^3 - 15t^4 + 6t^5 is 0 at 0, 1 at 1, 1/2 at 1/2, flat at both ends, and clamped."""
    assert rm.minimum_jerk(0.0) == 0.0 and rm.minimum_jerk(1.0) == 1.0
    assert rm.minimum_jerk(0.5) == pytest.approx(0.5)
    assert rm.minimum_jerk(-3.0) == 0.0 and rm.minimum_jerk(7.0) == 1.0
    u = np.linspace(0, 1, 1001)
    w = np.array([rm.minimum_jerk(x) for x in u])
    assert (np.diff(w) >= -1e-12).all()
    assert w[1] < 1e-8 and 1 - w[-2] < 1e-8            # zero slope at the ends: no kink in or out
    assert rm.minimum_jerk(0.25) == pytest.approx(10 / 64 - 15 / 256 + 6 / 1024)


def test_quintic_transition_meets_its_boundary_conditions():
    """The polynomial hits p0/p1 exactly and its finite-difference slope matches v0/v1 when nothing is
    clamped; the sample grid is the vendor's exact i/(steps-1)*T, not the rounded timestamps."""
    p0, p1 = HOME, yawed(20.0)
    v0, v1 = np.array([10.0, 0, 0, 0, 0]), np.array([-30.0, 0, 0, 0, 0])
    ts, pos, clamps = rm.quintic_transition(p0, p1, v0, v1, np.zeros(5), np.zeros(5),
                                            duration_ms=1100.0, max_velocity=55.0, max_acceleration=45.0)
    assert len(ts) == 33 and ts[0] == 0.0 and ts[-1] == pytest.approx(1100.0, abs=1e-3)
    assert np.allclose(pos[0], p0, atol=1e-4) and np.allclose(pos[-1], p1, atol=1e-4)
    dt = 1.1 / 32
    assert (pos[1, 0] - pos[0, 0]) / dt == pytest.approx(10.0, abs=0.6)     # forward difference ~ v0
    assert (pos[-1, 0] - pos[-2, 0]) / dt == pytest.approx(-30.0, abs=1.0)  # backward difference ~ v1
    assert all(not clamps[k] for k in clamps)
    # against a direct solve of the vendor's 3x3 system on the same exact grid
    T = 1.1
    A = np.array([[T ** 3, T ** 4, T ** 5], [3 * T ** 2, 4 * T ** 3, 5 * T ** 4], [6 * T, 12 * T ** 2, 20 * T ** 3]])
    c3, c4, c5 = np.linalg.solve(A, [p1[0] - p0[0] - v0[0] * T, v1[0] - v0[0], 0.0])
    t = np.arange(33) / 32 * T
    assert np.allclose(pos[:, 0], np.round(p0[0] + v0[0] * t + c3 * t ** 3 + c4 * t ** 4 + c5 * t ** 5, 4), atol=1.5e-4)


def test_quintic_clamps_boundary_velocity_and_acceleration_to_the_config_limits():
    """max_velocity_dps / max_acceleration_dps2 clamp what the transition is asked to MATCH at its
    ends (fusion.make_transition_quintic clamps v and a before solving); they never limit the path."""
    v1 = np.array([200.0, -10.0, 0, 0, 0])
    a1 = np.array([1000.0, 0, 0, 0, 0])
    _, pos_c, clamps = rm.quintic_transition(HOME, HOME, np.zeros(5), v1, np.zeros(5), a1,
                                             duration_ms=1100.0, max_velocity=55.0, max_acceleration=45.0)
    assert clamps["v1"] == {"base_yaw": {"asked": 200.0, "clamped_to": 55.0}}
    assert clamps["a1"] == {"base_yaw": {"asked": 1000.0, "clamped_to": 45.0}}
    assert clamps["v0"] == {} and clamps["a0"] == {}
    dt = 1.1 / 32
    assert abs((pos_c[-1, 0] - pos_c[-2, 0]) / dt) < 60.0                 # the end slope is the clamped 55, not 200
    _, pos_u, _ = rm.quintic_transition(HOME, HOME, np.zeros(5), v1, np.zeros(5), a1,
                                        duration_ms=1100.0, max_velocity=1e9, max_acceleration=1e9)
    assert np.abs(pos_u[:, 0]).max() > np.abs(pos_c[:, 0]).max()       # unclamped, the same request bulges further


def test_transition_limits_are_the_yaml_numbers_in_units_per_second_unconverted():
    """transition_settings.fusion_kwargs reads max_velocity_dps / max_acceleration_dps2 with _float_value
    and passes them to fuse_into_motion_plan as max_velocity / max_acceleration; fusion.py clamps unit
    velocities against them. 55 is 55 units/s on every joint; there is no per-joint degree reading."""
    v, a = rm.LIVE_CONFIG.limits_units()
    assert np.all(v == 55.0) and np.all(a == 45.0) and v.shape == a.shape == (5,)
    assert not hasattr(rm.LIVE_CONFIG, "velocity_units") and not hasattr(rm.LIVE_CONFIG, "deg_per_unit")
    v, a = replace(rm.LIVE_CONFIG, max_velocity_dps=90.0, max_acceleration_dps2=300.0).limits_units()
    assert np.all(v == 90.0) and np.all(a == 300.0)


@pytest.mark.skipif(not CALIBRATION.exists(), reason="needs the servo calibration")
def test_deg_per_unit_from_the_calibration_is_the_measured_0_74_on_base_yaw():
    dpu = rm.deg_per_unit_from_calibration(CALIBRATION)
    assert dpu[0] == pytest.approx((2767 - 1074) / 4096 * 360 / 200)
    assert dpu[0] == pytest.approx(0.744, abs=0.001)
    assert len(dpu) == 5 and all(0.4 < d < 1.0 for d in dpu)


def test_clip_timestamps_are_the_csv_route_to_a_thousandth_of_a_millisecond():
    ts = rm.clip_timestamps_ms(59)
    assert ts[0] == 0.0 and ts[1] == 33.333 and ts[3] == 100.0 and ts[58] == 1933.333
    csv_ts, rows = rm.read_clip(WAVE_HI) if WAVE_HI.exists() else (rm.clip_timestamps_ms(138), None)
    assert np.array_equal(csv_ts, rm.clip_timestamps_ms(len(csv_ts)))   # beat_clips' 6-decimal seconds agree


def test_resample_drops_a_frame_when_the_duration_is_not_a_whole_number_of_frames():
    """resample_motion_plan: int(1933.333 / 1000 * 30) is 57, so a 59-frame clip comes back as 58
    samples over the same 1933 ms; a 118-frame clip (3900 ms exactly) keeps 118."""
    ts, pos = rm._resample(rm.clip_timestamps_ms(59), np.tile(HOME, (59, 1)), 30)
    assert len(ts) == 58 and ts[-1] == 1933.333
    ts, pos = rm._resample(rm.clip_timestamps_ms(118), np.tile(HOME, (118, 1)), 30)
    assert len(ts) == 118
    # the shape survives: a linear ramp interpolates back onto itself
    ramp = np.tile(HOME, (59, 1)); ramp[:, 0] = np.linspace(0, 58, 59)
    ts, pos = rm._resample(rm.clip_timestamps_ms(59), ramp, 30)
    assert np.allclose(pos[:, 0], ts / (1000 / 30), atol=1e-4)                # ts is rounded to a thousandth of a ms


def test_smoothing_keeps_the_ends_and_rounds_a_snap():
    step = np.tile(HOME, (20, 1)); step[10:, 0] = 55.0
    out = rm._smooth(step, 5, 1)
    assert np.array_equal(out[:3], step[:3]) and np.array_equal(out[-3:], step[-3:])
    assert 0 < out[9, 0] < 55 and 0 < out[10, 0] < 55                     # the corner is rounded
    assert np.array_equal(rm._smooth(step[:4], 5, 1), step[:4])            # shorter than the window: untouched


def test_velocity_limit_stretches_time_and_cascades_like_the_vendor_loop():
    """limit_motion_velocity stretches a pair over 297 units/s and, because dt is measured from the
    already-stretched previous timestamp, the frames after it see a shorter dt and stretch too."""
    ts = rm.clip_timestamps_ms(6)
    pos = np.tile(HOME, (6, 1)); pos[:, 0] = [0, 0, 20, 25, 30, 30]      # 20 units in 33 ms = 600 u/s
    new_ts, new_pos, count = rm._limit_velocity(ts, pos, 300.0)
    assert np.array_equal(new_pos, pos)                                    # positions untouched
    assert count >= 1 and new_ts[2] - new_ts[1] == pytest.approx(20 / 297 * 1000, abs=0.01)
    assert new_ts[-1] > ts[-1]
    v = np.abs(np.diff(new_pos[:, 0])) / (np.diff(new_ts) / 1000)
    assert v.max() <= 297.0 * (1 + 1e-5)                                    # the stretched timestamp is rounded to 3 decimals
    # the cascade: frame 3 is 5 units in (100 ms - stretched 100.7 ms) -> dt <= 0 -> one-frame fallback
    assert new_ts[3] - new_ts[2] == pytest.approx(1000 / 30, abs=0.01)
    calm_ts, _, calm = rm._limit_velocity(ts, np.tile(HOME, (6, 1)), 300.0)
    assert calm == 0 and np.array_equal(calm_ts, ts)


def test_outlier_repair_is_a_no_op_on_a_validated_clip_and_fixes_a_spike():
    clip = hold_then_move(0.3, 0.5, 55.0, 0.3)
    fixed, n = rm._remove_outliers(clip, 20.0)
    assert n == 0 and np.array_equal(fixed, clip)
    spiked = clip.copy(); spiked[5, 0] = 40.0                             # one bad frame in the hold
    fixed, n = rm._remove_outliers(spiked, 20.0)
    assert n == 1 and fixed[5, 0] == pytest.approx(0.0)


# ----------------------------------------------------------------------------- the skip rule and the window
def test_skip_rule_fires_for_a_clip_that_starts_stationary_at_the_current_pose():
    """beat_clips' 0.6 s hold at START: position within 2 units, arm still, clip still over its first
    three frames -> the blend is skipped and the processed clip streams as it is, behind the live
    baseline frame."""
    clip = hold_then_move(0.6, 1.0, 55.0, 0.6)
    out = rm.simulate_playback(clip, HOME)
    d = out.decisions
    assert d["skipped_blend"] is True and d["blend"] is None
    assert d["skip_rule"] == {"pos_close": True, "max_pos_diff": 0.0, "current_velocity_magnitude": 0.0,
                              "clip_entry_velocity_magnitude": 0.0, "threshold_deg": 2.0, "velocity_threshold": 5.0}
    assert d["live_baseline"] and out.t_ms[0] == 0.0 and out.t_ms[1] == pytest.approx(1000 / 30, abs=1e-3)
    assert np.array_equal(out.positions[0], HOME)                          # the baseline write: stay put
    assert yaw(out).max() == pytest.approx(55.0, abs=0.5)                  # the whole move goes out
    assert d["frames_out"] == d["process"]["resampled_frames"] + 1


def test_skip_rule_needs_position_and_both_velocities():
    clip = hold_then_move(0.6, 1.0, 55.0, 0.6)
    assert rm.simulate_playback(clip, HOME + [3.0, 0, 0, 0, 0]).decisions["skipped_blend"] is False     # 3 > threshold_deg 2
    assert rm.simulate_playback(clip, HOME + [1.5, 0, 0, 0, 0]).decisions["skipped_blend"] is True
    assert rm.simulate_playback(clip, HOME, current_velocity=[6.0, 0, 0, 0, 0]).decisions["skipped_blend"] is False
    assert rm.simulate_playback(clip, HOME, current_velocity=[3.0, 3.0, 0, 0, 0]).decisions["skipped_blend"] is True  # |v| 4.2 < 5


def test_a_clip_that_starts_moving_is_crossfaded_and_its_head_is_dropped():
    """cha_look_right moves off frame 0 at once (entry velocity 68 units/s), so even from the exact
    pose the blend runs: the window starts lookahead_ms after frame 1 and everything before is gone."""
    out = rm.simulate_playback(cha_look_right(), HOME)
    d, b = out.decisions, out.decisions["blend"]
    assert d["skipped_blend"] is False and d["skip_rule"]["pos_close"] is True
    assert d["skip_rule"]["clip_entry_velocity_magnitude"] > 5.0
    assert b["start_ms"] == pytest.approx(1500.0 + 1000 / 30, abs=40.0)   # nearest frame to first_move + 1500 ms
    assert b["dropped_frames"] == b["start_index"] >= 40
    assert b["end_index"] == d["process"]["resampled_frames"] - 1          # shorter than the window: ends at the clip's end
    assert b["transition_duration_ms"] == 1100.0 and b["fade_frames"] == 33
    assert b["remaining_frames"] == 0                                      # nothing of the clip survives the fade
    assert out.t_ms[-1] == pytest.approx(1100.0 + 1000 / 30, abs=1e-3)


def test_blend_window_skips_dead_frames_then_looks_ahead():
    """find_blend_window on a clip that holds 0.6 s, moves, holds: the window starts at the frame
    nearest first_move + 1500 ms, or the next moving frame within a second if the clip is still there."""
    clip = hold_then_move(0.6, 1.0, 55.0, 2.0)                             # 18 frames of hold, a 1 s rise, 2 s still
    ts = rm.clip_timestamps_ms(len(clip))
    first = rm._first_moving(clip)
    assert first == 19                                                     # frame 18 is ease(0) = HOME; frame 19 is 0.15 on
    start, end = rm._blend_window(ts, clip, lookahead_ms=1500.0, blend_window_ms=1500.0, min_velocity=5.0)
    # first + 45 frames (1500 ms) is in the tail hold; nothing moves within the next second -> it stays there
    assert start == first + 45 and end == len(clip) - 1
    start, end = rm._blend_window(ts, clip, lookahead_ms=300.0, blend_window_ms=300.0, min_velocity=5.0)
    assert start == first + 9 and end == start + 9                         # 9 frames = 300 ms, twice


# ----------------------------------------------------------------------------- what the arm saw tonight
def test_a_fast_rise_is_attenuated_and_a_slow_one_is_not():
    """cha_look_right (+55 in 0.36 s, held 1.2 s): the whole look is inside the 1.5 s lookahead and is
    dropped; what streams is an 1100 ms bump. The arm delivered a peak of +19.9 (and +19.0 on a second
    run) of the +55 commanded. A rise that takes 3 s streams in full, because the blend window lands
    while it is still under way and the transition is aimed at the moving clip."""
    fast = rm.simulate_playback(cha_look_right(), HOME)
    assert 15.0 < yaw(fast).max() < 30.0                                   # measured 19.9 / 19.0, before the servo's own lag
    assert yaw(fast)[-1] == pytest.approx(0.0, abs=0.5)
    assert fast.t_ms[-1] < 1200.0                                          # a 1.93 s clip streams for 1.13 s
    slow_clip = hold_then_move(0.0, 3.0, 55.0, 2.0)
    slow = rm.simulate_playback(slow_clip, HOME)
    assert slow.decisions["skipped_blend"] is True                         # 1 unit/s over its first 3 frames: under the 5 u/s rule
    assert yaw(slow).max() > 0.95 * 55.0 and yaw(slow)[-1] == pytest.approx(55.0, abs=0.5)
    # and from 3 units off the pose, where the blend has to run, the rise still arrives: the window
    # lands 1.5 s into the 3 s rise, on a moving clip, and the transition is aimed at it
    blended = rm.simulate_playback(slow_clip, HOME + [3.0, 0, 0, 0, 0])
    assert blended.decisions["skipped_blend"] is False
    assert yaw(blended).max() > 0.95 * 55.0 and yaw(blended)[-1] == pytest.approx(55.0, abs=0.5)


def test_fusion_py_defaults_would_have_delivered_the_look_so_they_are_not_what_runs():
    """The ground-truth note reasoned with fusion.py's own 500 / 300 / 90 / 300. Under those the look
    streams at its full +55 for over a second -- which the arm did not do. default.yaml's block does."""
    out = rm.simulate_playback(cha_look_right(), HOME, config=rm.FUSION_PY_DEFAULTS)
    held = out.t_ms[np.abs(yaw(out) - 55.0) < 1.0]
    assert yaw(out).max() > 54.0 and held[-1] - held[0] > 1000.0


def test_cha_turn_streams_only_its_left_swing():
    """cha_turn as cha_moves builds it (+65 first, 1.5 beats, then across to -65): the lookahead drops
    the first 1.53 s -- the whole right swing -- and the stream is the crossing to -65 and home.
    Measured: yaw -4.9..+1.5, "went left, hit the stop, never went right". Everything left of -4 is
    the stop, so the stream's -63 is what the stop turned into -5. The check below with the swings the
    other way round is what the Validate phase must run against the DEPLOYED csv: the note writes the
    order as 0 -> -65 -> +65, and if that is the file that played, this model predicts a right swing."""
    out = rm.simulate_playback(cha_turn(+65.0), HOME)
    b = out.decisions["blend"]
    assert b["dropped_frames"] >= 40 and b["start_ms"] == pytest.approx(1533.3, abs=1.0)
    assert yaw(out).max() < 2.0 and yaw(out).min() < -40.0
    assert yaw(out)[-1] == pytest.approx(0.0, abs=0.5)
    mirrored = rm.simulate_playback(cha_turn(-65.0), HOME)
    assert yaw(mirrored).min() > -2.0 and yaw(mirrored).max() > 40.0


@pytest.mark.skipif(not HYPE_124_C.exists(), reason="needs the generated beat clip")
def test_beat_hype_124_c_skips_the_blend_and_streams_its_full_swing():
    """152 frames, 5.07 s, 0.6 s hold at START: the skip rule fires and the clip streams whole,
    including the +68 right swing the arm delivered as +66.7 (the left swing met the stop)."""
    ts, rows = rm.read_clip(HYPE_124_C)
    out = rm.simulate_playback(rows, HOME, timestamps_ms=ts)
    assert out.decisions["skipped_blend"] is True
    assert yaw(out).max() > 0.97 * rows[:, 0].max() and yaw(out).min() < 0.97 * rows[:, 0].min()
    s = rm.delivered_summary(out)
    assert s["samples"][6:9] == pytest.approx([63.0, 57.0, 24.0], abs=3.0)   # measured +66 +57 +24 at 3.0, 3.5, 4.0 s
    assert out.decisions["frames_out"] == 152 and out.decisions["process"]["resampled_frames"] == 151


@pytest.mark.skipif(not WAVE_HI.exists(), reason="needs the wave clip")
def test_wave_hi_streams_in_full_after_the_skip():
    ts, rows = rm.read_clip(WAVE_HI)
    out = rm.simulate_playback(rows, HOME, timestamps_ms=ts)
    assert out.decisions["skipped_blend"] is True
    assert yaw(out).max() > 27.0 and yaw(out).min() < -27.0               # measured +25.8 right of the commanded 30


# ----------------------------------------------------------------------------- safety, baseline, determinism
def test_safety_refuses_out_of_envelope_and_over_speed_and_clamps_within_tolerance():
    clip = hold_then_move(0.6, 1.0, 55.0, 0.6)
    spike = clip.copy(); spike[30, 0] = 101.0                              # one frame: remove_position_outliers repairs it first
    assert rm.simulate_playback(spike, HOME).positions[:, 0].max() < 60.0
    bad = clip.copy(); bad[-6:, 0] = 101.0                                 # a plateau the outlier filter leaves alone
    with pytest.raises(rm.PlanRefused, match="envelope"):
        rm.simulate_playback(bad, HOME)
    grazing = hold_then_move(0.6, 1.0, 100.0005, 0.6)                      # inside the 0.001 tolerance: clamped, not refused
    out = rm.simulate_playback(grazing, HOME, config=replace(rm.LIVE_CONFIG, smooth_enabled=False))
    assert out.positions[:, 0].max() == 100.0 and out.decisions["safety"]["clamped_joints"] == ["base_yaw"]
    snap = np.tile(HOME, (10, 1)); snap[5:, 0] = 60.0                     # 60 units in one frame = 1800 u/s
    with pytest.raises(rm.PlanRefused, match="Velocity"):
        rm.simulate_playback(snap, HOME, config=replace(rm.LIVE_CONFIG, smooth_enabled=False))
    stretched = rm.simulate_playback(snap, HOME)                           # with processing on, the limiter stretches it first
    assert stretched.decisions["process"]["velocity_stretch_count"] >= 1
    assert stretched.decisions["safety"]["peak_velocity_units_s"] <= 300.0


def test_live_baseline_is_the_measured_pose_then_the_plan_one_frame_later():
    pose = HOME + [7.0, -3.0, 0, 0, 0]
    out = rm.simulate_playback(hold_then_move(0.6, 1.0, 55.0, 0.6), pose)
    assert np.array_equal(out.positions[0], pose) and out.t_ms[0] == 0.0
    off = rm.simulate_playback(hold_then_move(0.6, 1.0, 55.0, 0.6), pose, config=replace(rm.LIVE_CONFIG, live_baseline=False))
    assert off.decisions["frames_out"] == out.decisions["frames_out"] - 1
    assert np.allclose(off.positions, out.positions[1:]) and np.allclose(off.t_ms, out.t_ms[1:] - 1000 / 30, atol=1e-3)
    late = rm.simulate_playback(hold_then_move(0.6, 1.0, 55.0, 0.6), pose, config=replace(rm.LIVE_CONFIG, start_latency_ms=350.0))
    assert late.t_ms[0] == 350.0 and np.allclose(late.t_ms - 350.0, out.t_ms)


def test_the_model_is_deterministic_and_the_config_matters():
    clip = cha_look_right()
    a = rm.simulate_playback(clip, HOME, current_velocity=[2.0, 0, 0, 0, 0])
    b = rm.simulate_playback(clip, HOME, current_velocity=[2.0, 0, 0, 0, 0])
    assert np.array_equal(a.t_ms, b.t_ms) and np.array_equal(a.positions, b.positions)
    assert a.decisions == b.decisions
    c = rm.simulate_playback(clip, HOME, current_velocity=[2.0, 0, 0, 0, 0], config=replace(rm.LIVE_CONFIG, lookahead_ms=300.0))
    assert not np.array_equal(a.positions, c.positions)
    d = rm.simulate_playback(clip, HOME, current_velocity=[2.0, 0, 0, 0, 0], config=replace(rm.LIVE_CONFIG, max_velocity_dps=90.0))
    assert d.decisions["effective_max_velocity_units"][0] == 90.0
    assert d.decisions["blend"]["velocity_clamp"] != a.decisions["blend"]["velocity_clamp"]   # only the boundary clamp moved
    assert abs(yaw(d).max() - yaw(a).max()) < 3.0                          # ... and it is worth about a unit of peak on this clip


def test_bad_inputs_are_refused_early():
    with pytest.raises(ValueError):
        rm.simulate_playback(np.zeros((3, 4)), HOME)
    with pytest.raises(ValueError):
        rm.simulate_playback(np.zeros((0, 5)), HOME)
    with pytest.raises(ValueError):
        rm.simulate_playback(np.zeros((3, 5)), HOME, timestamps_ms=[0.0, 33.333])


# ----------------------------------------------------------------------------- the dropped window and the skip rule, pinned
def installed(name):
    ts, rows = rm.read_clip(INSTALLED[name])
    return rm.simulate_playback(rows, REST, timestamps_ms=ts)


def test_skip_rule_reads_the_clip_entry_velocity_over_frames_0_to_2():
    """blend_into_motion_plan: first_vel = estimate_velocity(plan, 0), window 2 -> (p[2] - p[0]) / 66.7 ms,
    magnitude over joints, against velocity_threshold 5. 0.3 units by frame 2 is 4.5 units/s and the
    clip streams raw; 0.4 units is 6.0 units/s and the blend runs, dropping the first 1.5 s. The first
    three frames survive processing untouched (smooth keeps half + 1 = 3 frames at each end, and a
    31-frame clip is exactly 1000 ms so the resample keeps every sample)."""
    for step, expect_skip in ((0.15, True), (0.20, False)):
        clip = np.tile(HOME, (31, 1))
        clip[1, 0] = HOME[0] + step
        clip[2:, 0] = HOME[0] + 2 * step
        out = rm.simulate_playback(clip, HOME)
        assert out.decisions["skip_rule"]["clip_entry_velocity_magnitude"] == pytest.approx(2 * step / 0.066667, abs=1e-3)   # ts[2] is 66.667 ms
        assert out.decisions["skipped_blend"] is expect_skip, step
    assert rm.simulate_playback(clip, HOME).decisions["blend"]["dropped_frames"] > 0


def test_max_velocity_dps_clamps_only_the_boundary_velocity_never_the_path():
    """make_transition_quintic clamps v0 / v1 (and a0 / a1) before solving: the quintic is asked to ARRIVE
    at 55 units/s, but nothing limits how fast the fused path moves between its ends. cha_turn's fade
    crosses 60 units in 1.1 s at up to 154 units/s with the 55 clamp in force on its end velocity."""
    clip = cha_turn(+65.0)
    out = rm.simulate_playback(clip, HOME)
    b = out.decisions["blend"]
    assert b["velocity_clamp"]["v1"]["base_yaw"]["clamped_to"] == 55.0
    assert abs(b["velocity_clamp"]["v1"]["base_yaw"]["asked"]) > 55.0
    speed = np.abs(np.diff(yaw(out))) / (np.diff(out.t_ms) / 1000.0)
    assert speed.max() > 100.0                                             # the path itself is well over the clamp
    wide = rm.simulate_playback(clip, HOME, config=replace(rm.LIVE_CONFIG, max_velocity_dps=200.0))
    assert wide.decisions["blend"]["velocity_clamp"]["v1"] == {}          # a wider clamp: the boundary is met ...
    assert wide.decisions["blend"]["dropped_frames"] == b["dropped_frames"]   # ... and nothing else changes
    assert yaw(wide).max() < 2.0                                           # the right swing is still gone


def test_lookahead_ms_alone_decides_whether_the_look_survives():
    """Same LIVE config, only lookahead_ms changed: at 1500 the window starts on the return and the +55
    look is dropped; at 300 (fusion.py's own default) it starts inside the rise and the look streams."""
    dropped = rm.simulate_playback(cha_look_right(), HOME)
    kept = rm.simulate_playback(cha_look_right(), HOME, config=replace(rm.LIVE_CONFIG, lookahead_ms=300.0, blend_window_ms=300.0))
    assert yaw(dropped).max() < 25.0 and yaw(kept).max() > 50.0
    assert kept.decisions["blend"]["dropped_frames"] < dropped.decisions["blend"]["dropped_frames"]


@needs_clips
def test_installed_cha_look_right_fails_the_skip_rule_and_loses_the_look():
    """The CSV as installed, from the measured rest (1.5 units off on yaw, inside threshold_deg): the
    clip moves off frame 0 at once (entry velocity ~68 units/s), so the blend runs; the window starts at
    the frame nearest first_move + 1500 ms (frame 45 of 58, 1526 ms, already on the return), everything
    before it is dropped, the window reaches the clip's end, and the stream is one 1100 ms fade plus
    the baseline frame: 1133 ms, peak +21.7 -- the +55 is never written."""
    out = installed("cha_look_right")
    d, b = out.decisions, out.decisions["blend"]
    assert d["skip_rule"]["pos_close"] is True and d["skip_rule"]["max_pos_diff"] == pytest.approx(1.5)
    assert d["skip_rule"]["clip_entry_velocity_magnitude"] > 50.0
    assert d["skipped_blend"] is False
    assert d["process"]["resampled_frames"] == 58 and d["frames_in"] == 59
    assert b["dropped_frames"] == b["start_index"] == 45
    assert b["start_ms"] == pytest.approx(45 * 1933.333 / 57, abs=0.01)   # 1526.3 ms: frame 45 of the 33.918 ms resample grid
    assert b["end_index"] == 57 and b["remaining_frames"] == 0
    assert d["duration_ms"] == pytest.approx(1100.0 + 1000 / 30, abs=0.01)
    assert d["frames_out"] == 34                                           # baseline + 33 fade frames
    assert 20.0 < yaw(out).max() < 23.0 and yaw(out)[-1] == pytest.approx(0.0, abs=0.01)


@needs_clips
def test_installed_cha_turn_drops_its_right_swing_and_slides_left():
    """cha_turn as installed goes RIGHT first (+65 at 0.73 s) then left. first_move is frame 1, the window
    starts at frame 46 (1533 ms, yaw +30 heading left) and ends at frame 91 (3033 ms, yaw -47); frames
    0..45 -- the whole right swing -- are dropped. What streams: the 1100 ms crossfade from the rest pose
    through the window (down to -63 as the clip's -65 hold passes through it), then the 25 frames after
    win_end back home. Nothing in the stream is right of the rest yaw."""
    out = installed("cha_turn")
    d, b = out.decisions, out.decisions["blend"]
    assert d["skipped_blend"] is False and d["process"]["resampled_frames"] == 117
    assert b["dropped_frames"] == 46 and b["start_ms"] == pytest.approx(1533.333, abs=0.01)
    assert b["end_index"] == 91 and b["end_ms"] == pytest.approx(3033.333, abs=0.01)
    assert b["remaining_frames"] == 25
    assert d["duration_ms"] == pytest.approx(1100.0 + 26 * 1000 / 30, abs=0.01)
    assert yaw(out).max() == pytest.approx(REST[0]) and yaw(out).min() < -60.0
    assert yaw(out)[-1] == pytest.approx(0.0, abs=0.01)


@needs_clips
def test_installed_beat_hype_124_c_passes_the_skip_rule_only_just():
    """0.63 s stationary at HOME before the first move: entry velocity 0, the rest pose within 2.0 of
    frame 0 (max diff 1.5 on yaw) -> raw playback, 151 processed frames, nothing dropped, 5.07 s. From
    0.6 units further off on yaw (2.1 > threshold_deg) the same clip would have been blended and its
    first 1.5 s of motion dropped."""
    out = installed("beat_hype_124_c")
    d = out.decisions
    assert d["skipped_blend"] is True and d["skip_rule"]["clip_entry_velocity_magnitude"] == 0.0
    assert d["skip_rule"]["max_pos_diff"] == pytest.approx(1.5)
    assert d["frames_in"] == 152 and d["process"]["resampled_frames"] == 151 and d["frames_out"] == 152
    assert d["duration_ms"] == pytest.approx(5033.333 + 1000 / 30, abs=0.01)
    assert yaw(out).max() > 67.0 and yaw(out).min() < -67.0
    ts, rows = rm.read_clip(INSTALLED["beat_hype_124_c"])
    off = rm.simulate_playback(rows, REST + [0.6, 0, 0, 0, 0], timestamps_ms=ts)
    assert off.decisions["skipped_blend"] is False and off.decisions["blend"]["dropped_frames"] > 40


@needs_clips
def test_installed_wave_hi_passes_the_skip_rule_and_streams_smoothed():
    """1.2 s stationary lead-in -> raw playback; the 5-frame box smooth takes the +-30 wave at 1.33 Hz to
    +-27.7 before it is streamed (what the servo is asked for is 27.7, not 30)."""
    out = installed("wave_hi")
    assert out.decisions["skipped_blend"] is True and out.decisions["blend"] is None
    assert out.decisions["duration_ms"] == pytest.approx(4566.667 + 1000 / 30, abs=0.01)
    assert yaw(out).max() == pytest.approx(27.7, abs=0.3) and yaw(out).min() == pytest.approx(-27.7, abs=0.3)
