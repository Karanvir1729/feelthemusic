"""simulate: the two stages chained, against the four clips of ground_truth_2026-09-20.md as installed
on the lamp, from the measured rest. Each case pins three things the note measured -- skipped or blended,
the streamed duration, the peak yaw -- and where the model does NOT reproduce a measurement the test says
so with a strict xfail (it flips the day a stated cause is modelled), instead of a tolerance wide enough
to hide it. Tolerances: peaks +-1.5 units where the two measured runs of the same clip differ by 0.9 and
the rest pose by 0.5; streamed durations to a frame where the mechanism fixes them exactly; journal
times to their own precision (~0.1 s)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime_model as rm  # noqa: E402
import servo_model as sm  # noqa: E402
import simulate as si  # noqa: E402

needs_clips = pytest.mark.skipif(not all(p.exists() for p in si.CASES.values()), reason="needs the installed clips")
FRAME_MS = 1000 / 30
M = si.MEASURED


def exec_samples(pred, every_s=0.5):
    return pred.sample(every_s=every_s, clock="executor")[1]


def peak_time_exec(pred):
    y = pred.measured[:, 0]
    return float(pred.t[np.argmax(y)] - pred.decision["start_latency_ms"] / 1000.0)


# ----------------------------------------------------------------------------- the defaults are the measured ranges
def test_fitted_parameters_sit_inside_their_measured_ranges():
    assert si.LAG_RANGE_S == (0.10, 0.15) and si.LAG_RANGE_S[0] <= si.LAG_S <= si.LAG_RANGE_S[1]
    assert si.LATENCY_RANGE_MS == (250.0, 450.0) and si.LATENCY_RANGE_MS[0] <= si.START_LATENCY_MS <= si.LATENCY_RANGE_MS[1]
    assert si.YAW_STOP == sm.STOP_YAW_MIN == -4.5
    assert np.allclose(si.REST, (1.5, -48.5, -23.2, 0.0, 30.0))


# ----------------------------------------------------------------------------- 1. cha_look_right
@needs_clips
def test_cha_look_right_is_blended_its_look_dropped_and_the_bump_matches():
    """Measured: peak +19.9 (+19.0 on a repeat), streamed ~1.2 s, samples +1 +10 +8 +1. Model: the skip
    rule fails on the entry velocity (frames 0->2 already moving), 45 of 58 frames dropped, the fade over
    the return streams as a 1133 ms bump the servo turns into +19.2, peaking 0.7 s after the first write."""
    p = si.run_case("cha_look_right")
    d = p.decision
    assert d["blended"] and not d["skipped_blend"]
    assert d["skip_rule"]["pos_close"] is True and d["skip_rule"]["clip_entry_velocity_magnitude"] > 5.0
    assert d["dropped_frames"] == 45 and d["frames_processed"] == 58
    assert d["streamed_ms"] == pytest.approx(1100.0 + FRAME_MS, abs=0.01)          # journal ~1.2 s incl. preparation
    assert abs(d["streamed_ms"] / 1000.0 + d["start_latency_ms"] / 1000.0 - M["cha_look_right"]["streamed_s"]) < 0.4
    lo, hi = p.extent()
    target = (M["cha_look_right"]["peak"] + M["cha_look_right"]["peak_repeat"]) / 2      # 19.45
    assert abs(hi - target) <= 1.5, hi                                                # 19.2 at lag 0.10
    assert hi < p.streamed.positions[:, 0].max()                                      # the servo delivers less than the 21.7 streamed
    assert 0.5 <= peak_time_exec(p) <= 0.9                                            # between the tracer's +10 (0.5 s) and +8 (1.0 s)
    s = exec_samples(p)
    assert 8.0 <= s[1] <= 15.0 and 7.0 <= s[2] <= 14.0 and s[3] < 3.0                # +10 +8 +1 measured; +-3 for the tracer's phase
    assert d["banging_s"]["base_yaw"] == 0.0 and lo >= 1.0                            # never near the stop


# ----------------------------------------------------------------------------- 2. cha_turn
@needs_clips
def test_cha_turn_never_goes_right_and_sits_on_the_stop():
    """Measured: +1 -5 -5 -5 -1 -1 ..., range -4.9..+1.5, never right. Model: 46 frames (the whole right
    swing) dropped, the fade slides from the rest pose to -63 (the clip's -65 hold passes through the
    window) and the stop turns that into -4.5 for 1.2 s; the max of the whole run is the rest yaw."""
    p = si.run_case("cha_turn")
    d = p.decision
    assert d["blended"] and d["dropped_frames"] == 46 and d["frames_processed"] == 117
    assert p.streamed.positions[:, 0].min() < -60.0                                   # what was asked of the servo ...
    lo, hi = p.extent()
    assert lo == si.YAW_STOP and abs(lo - M["cha_turn"]["min"]) <= 0.5                # ... and what the stop allowed (-4.9 measured)
    assert hi == pytest.approx(M["cha_turn"]["peak"], abs=0.1)                        # +1.5: the rest yaw, never right of it
    assert d["banging_s"]["base_yaw"] > 0.8
    s = exec_samples(p)
    assert np.allclose(s[:6], [1.5, -4.5, -4.5, -4.5, -1.2, -1.2], atol=0.6)          # measured +1 -5 -5 -5 -1 -1
    assert abs(p.measured[-1, 0] - (-1.0)) <= 0.5                                    # lands 1.2 short of 0, as the arm did (-1)
    assert d["streamed_ms"] == pytest.approx(1100.0 + 26 * FRAME_MS, abs=0.01)        # fade + 25 frames after win_end + baseline


@needs_clips
@pytest.mark.xfail(strict=True, reason="journal says incoming->accepted ~1.5 s; fusion.py as read cannot stream "
                                       "less than the 1100 ms fade plus the 25 frames after win_end (1.97 s)")
def test_cha_turn_streamed_duration_matches_the_journal():
    p = si.run_case("cha_turn")
    assert abs(p.decision["streamed_ms"] / 1000.0 - M["cha_turn"]["streamed_s"]) <= 0.25


# ----------------------------------------------------------------------------- 3. beat_hype_124_c
@needs_clips
def test_beat_hype_124_c_streams_raw_for_its_full_length():
    """Measured: -6.1..+66.7, streamed ~5.0 s, +66 +57 +24 at 3.0/3.5/4.0 s. Model: the skip rule fires
    (0.63 s stationary lead-in; rest pose 1.5 units off, inside threshold_deg 2.0), nothing dropped,
    5.07 s streamed, the left swing on the stop, the right swing delivered at +61.6 (short: see the
    xfail below), peaking 3.3 s after the first write."""
    p = si.run_case("beat_hype_124_c")
    d = p.decision
    assert d["skipped_blend"] and not d["blended"] and d["dropped_frames"] == 0
    assert d["frames_in"] == 152 and d["frames_processed"] == 151 and d["frames_streamed"] == 152
    assert d["streamed_ms"] == pytest.approx(5033.333 + FRAME_MS, abs=0.01)
    assert abs(d["streamed_ms"] / 1000.0 - M["beat_hype_124_c"]["streamed_s"]) <= 0.1  # journal ~5.0 s
    lo, hi = p.extent()
    assert lo == si.YAW_STOP and d["banging_s"]["base_yaw"] > 1.0                      # measured -6.1: the stop, hit at speed
    assert 60.0 <= hi <= 63.0                                                         # the model's number at lag 0.10, ceiling 140
    assert 2.9 <= peak_time_exec(p) <= 3.5                                            # measured peak at the 3.0 s sample
    s = exec_samples(p)
    assert s[7] == pytest.approx(M["beat_hype_124_c"]["yaw_every_0_5_s"][7], abs=6.0)  # +57 at 3.5 s
    assert (s[2:6] <= -4.0).all()                                                     # the left swing sits on the stop 1.0-2.5 s


@needs_clips
@pytest.mark.xfail(strict=True, reason="measured +66.7 of a +67.8 stream; the crossing is commanded at 240 units/s and "
                                       "servo_model keeps the vendor's 140 units/s ceiling (simulation.yaml), which the arm exceeded")
def test_beat_hype_124_c_right_swing_reaches_the_measured_peak():
    p = si.run_case("beat_hype_124_c")
    assert abs(p.extent()[1] - M["beat_hype_124_c"]["peak"]) <= 2.0


# ----------------------------------------------------------------------------- 4. wave_hi
@needs_clips
def test_wave_hi_streams_raw_left_on_the_stop_right_short():
    """Measured: -5.0..+25.8. Model: skip rule fires (1.2 s lead-in), 4.6 s streamed, the streamed wave is
    +-27.7 after the box smooth, the left side on the stop (-4.5 vs -5.0), the right side +18.5."""
    p = si.run_case("wave_hi")
    d = p.decision
    assert d["skipped_blend"] and d["dropped_frames"] == 0 and d["frames_streamed"] == 139
    assert d["streamed_ms"] == pytest.approx(4566.667 + FRAME_MS, abs=0.01)
    assert p.streamed.positions[:, 0].max() == pytest.approx(27.7, abs=0.3)
    lo, hi = p.extent()
    assert lo == si.YAW_STOP and abs(lo - M["wave_hi"]["min"]) <= 0.5
    assert 17.5 <= hi <= 19.5                                                         # the model's number at lag 0.10, ceiling 140
    assert d["banging_s"]["base_yaw"] > 0.2


@needs_clips
@pytest.mark.xfail(strict=True, reason="measured +25.8 = 93 % of the +-27.7 streamed at 1.33 Hz; a 100 ms first-order lag "
                                       "passes 77 % at most and the 140 units/s ceiling takes it to 67 %: the raised-pose yaw "
                                       "servo is faster than the folded-pose one beattrace measured")
def test_wave_hi_right_side_reaches_the_measured_peak():
    p = si.run_case("wave_hi")
    assert abs(p.extent()[1] - M["wave_hi"]["peak"]) <= 2.0


# ----------------------------------------------------------------------------- the fits
@needs_clips
def test_lag_fit_lands_on_the_lower_edge_because_every_peak_is_short():
    best, table = si.fit_lag()
    assert best == 0.10
    rms = [row["rms"] for row in table]
    assert all(a < b for a, b in zip(rms, rms[1:]))                                  # monotone: no interior optimum
    for row in table:
        assert all(r <= 0.05 for r in row["residuals"].values())                      # nothing is over-delivered at any lag


@needs_clips
def test_latency_is_not_identifiable_from_the_traces_which_run_on_the_executor_clock():
    """On the executor clock the sampled prediction does not contain the latency at all; on the POST
    clock every latency in range is a worse fit than the executor clock. The tracer started its clock
    at the first write, as cliptrace2.py does, so the latency stays a mid-range pass-through."""
    a = si.run_case("cha_look_right", start_latency_ms=250.0)
    b = si.run_case("cha_look_right", start_latency_ms=450.0)
    assert np.allclose(exec_samples(a), exec_samples(b))
    assert not np.allclose(a.sample(clock="post")[1][:4], b.sample(clock="post")[1][:4])
    for row in si.fit_latency():
        for case in ("cha_look_right", "beat_hype_124_c"):
            assert row["rms_post_clock"][case] > row["rms_executor_clock"][case]


# ----------------------------------------------------------------------------- inputs and plumbing
@needs_clips
def test_the_same_clip_by_path_rows_and_array_predicts_the_same_and_deterministically():
    path = si.CASES["cha_look_right"]
    ts, rows = rm.read_clip(path)
    by_path = si.simulate(path)
    by_array = si.simulate(rows, timestamps_ms=ts)
    with open(path) as f:
        header = [h.strip() for h in f.readline().split(",")]
        dicts = [dict(zip(header, line.split(","))) for line in f if line.strip()]
    by_rows = si.simulate(dicts)
    again = si.simulate(path)
    for other in (by_array, by_rows, again):
        assert np.array_equal(by_path.measured, other.measured) and np.array_equal(by_path.t, other.t)
        assert by_path.decision == other.decision


@needs_clips
def test_start_pose_and_velocity_feed_the_skip_rule():
    """beat_hype from 0.6 units further off on yaw (2.1 > threshold_deg 2.0) is blended and loses its first
    1.5 s; from the rest with a 6 units/s executor velocity likewise. That is how marginal tonight's raw
    playback was."""
    off = si.run_case("beat_hype_124_c", start_pose=np.array(si.REST) + [0.6, 0, 0, 0, 0])
    assert off.decision["blended"] and off.decision["dropped_frames"] > 40
    moving = si.run_case("beat_hype_124_c", start_velocity=6.0)
    assert moving.decision["blended"]
    assert si.run_case("beat_hype_124_c", start_velocity=[3.0, 3.0, 0, 0, 0]).decision["skipped_blend"]   # |v| 4.2 < 5


def test_refusals_and_clocks():
    clip = np.tile(si.REST, (40, 1))
    clip[-6:, 0] = 101.0                                                              # outside the +-100 envelope: the runtime raises
    with pytest.raises(rm.PlanRefused):
        si.simulate(clip)
    with pytest.raises(ValueError):
        si.simulate(clip[:-6], start_latency_ms=-1.0)
    p = si.simulate(clip[:-6], start_latency_ms=300.0)
    assert p.decision["first_write_ms"] == 300.0 and p.t[0] == pytest.approx(0.3)     # the POST clock
    t_post, v_post = p.sample(clock="post")
    t_exec, v_exec = p.sample(clock="executor")
    assert t_post[0] == 0.0 and v_post[0] == si.REST[0] and t_exec[0] == 0.0
    with pytest.raises(ValueError):
        p.sample(clock="wall")
