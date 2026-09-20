"""servo_model: the servos and joints against what the arm measured on 2026-09-20 (ground_truth_2026-09-20.md).
Offline, numpy only; the one test that needs the vendor description (simulation.yaml) is skipped without it."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import servo_model as sm  # noqa: E402

ROBOT_DIR = Path(os.environ.get("FTM_ROBOT_DIR")
                 or Path(os.environ.get("LAMP_ROBOTDESC", "/nonexistent")) / "pi5_feetech_r1")
SIMULATION_YAML = ROBOT_DIR / "simulation.yaml"

HOME = np.array([0.0, -49.0, -22.0, 0.0, 30.0])     # where tonight's three traces started
YAW, PITCH = sm.JOINTS.index("base_yaw"), sm.JOINTS.index("base_pitch")
FPS = 30                                            # the executor streams a clip at the pack's 30 fps


def stream(fn, seconds, fps=FPS):
    """(t_s, goals) as the executor would write them: one waypoint per frame."""
    t = np.arange(0.0, seconds + 1e-9, 1.0 / fps)
    return t, np.array([fn(x) for x in t], dtype=float)


def ramp(joint, units, seconds):
    """A tracking-route move: `units` at constant speed over `seconds`, then held."""
    step = np.zeros(5)
    step[joint] = 1.0
    return lambda x: HOME + step * units * min(1.0, x / seconds)


def cha_turn(x):
    """The shape of tonight's cha_turn: yaw 0 -> -65 in 0.5 s, hold, -> +65 in 1 s, hold, back; a
    pitch wiggle alongside so 'other joints unaffected' is checked on a moving joint."""
    if x < 0.5:
        yaw = -65.0 * x / 0.5
    elif x < 1.5:
        yaw = -65.0
    elif x < 2.5:
        yaw = -65.0 + 130.0 * (x - 1.5)
    elif x < 3.5:
        yaw = 65.0
    else:
        yaw = 65.0 - 65.0 * min(1.0, (x - 3.5) / 0.5)
    return HOME + np.array([yaw, 10.0 * np.sin(2 * np.pi * x), 0.0, 0.0, 0.0])


# ----------------------------------------------------------------------------- the constants
@pytest.mark.skipif(not SIMULATION_YAML.exists(), reason="vendor simulation.yaml not available")
def test_vendor_constants_are_the_files():
    """The numbers the module hard-codes are the vendor's simulation.yaml, read back from the file."""
    motor = sm.vendor_motor_params(SIMULATION_YAML)
    assert motor["update_hz"] == sm.UPDATE_HZ
    assert motor["command_latency_s"] == sm.COMMAND_LATENCY_S
    assert motor["read_latency_s"] == sm.READ_LATENCY_S
    assert motor["max_velocity_per_s"] == sm.VENDOR_MAX_VELOCITY
    assert motor["max_acceleration_per_s2"] == sm.VENDOR_MAX_ACCELERATION
    assert motor["damping_per_s"] == sm.VENDOR_DAMPING_PER_S
    assert motor["backlash"] == sm.VENDOR_BACKLASH
    assert motor["deadband"] == sm.VENDOR_DEADBAND
    assert motor["position_quantization"] == sm.VENDOR_QUANTISATION


def test_defaults_are_the_vendor_model_except_where_the_arm_disagrees():
    p = sm.ServoParams()
    assert np.all(p.vmax() == 140.0) and np.all(p.amax() == 420.0)
    assert p.backlash == 1.0 and p.deadband == 0.2 and p.quantisation == 0.05
    assert sm.MEASURED_LAG_S[0] <= p.tau()[YAW] <= sm.MEASURED_LAG_S[1]      # not the vendor's 42 ms
    assert sm.VENDOR_TAU_S == pytest.approx(1 / 24)
    lo, hi = p.stop_arrays()
    assert lo[YAW] == sm.STOP_YAW_MIN == -4.5 and hi[YAW] == 100.0
    assert sm.MEASURED_STOP_RANGE[0] <= sm.STOP_YAW_MIN <= sm.MEASURED_STOP_RANGE[1]
    assert all(lo[i] == -100.0 and hi[i] == 100.0 for i in range(5) if i != YAW)
    assert p.delivery is None                                                # the physics, unless overridden


# ----------------------------------------------------------------------------- slow moves
def test_slow_ramp_is_followed_within_a_few_percent():
    """40 units at 10 u/s on base_pitch, then held: the joint trails by the lag (v tau), the play and the
    deadband while moving, lands within 4 % once the goal stops, and never overshoots."""
    t, g = stream(ramp(PITCH, 40.0, 4.0), 4.0)
    tr = sm.simulate(t, g, start=HOME, tail_s=1.0)
    pos = tr.column("base_pitch")
    assert abs(pos[-1] - g[-1, PITCH]) <= 0.04 * 40
    assert np.abs(tr.tracking_error[:, PITCH]).max() <= 0.07 * 40
    assert (np.diff(tr.position[:, PITCH]) >= 0).all()
    assert tr.banging.sum() == 0


def test_the_yaw_ramp_lands_where_the_arm_did():
    """Ground truth: +40 over 4 s on base_yaw delivered +38.6. Rightward is free of the stop."""
    units, seconds, delivered = sm.MEASURED_SLOW_RAMP
    t, g = stream(ramp(YAW, units, seconds), seconds)
    tr = sm.simulate(t, g, start=HOME, tail_s=1.0)
    assert abs(tr.final()["base_yaw"] - delivered) <= 0.5
    assert tr.banging.sum() == 0


def test_direct_moves_land_an_absolute_deficit_not_a_proportional_one():
    """Tonight's 3 s direct moves of +11.4, +25.3 and +41.7 landed 1.2, 1.6 and 1.5 short: the same
    amount whatever the size, which is the play plus the deadband, not a percentage."""
    deficits = []
    for asked in (11.4, 25.3, 41.7):
        t, g = stream(ramp(YAW, asked, 3.0), 3.0)
        tr = sm.simulate(t, g, start=HOME, tail_s=1.0)
        deficits.append(asked - tr.final()["base_yaw"])
    lo, hi = sm.MEASURED_DIRECT_DEFICIT
    assert all(lo - 0.2 <= d <= hi + 0.1 for d in deficits), deficits
    assert max(deficits) - min(deficits) < 0.2


# ----------------------------------------------------------------------------- fast moves
def test_step_is_velocity_limited():
    """A 60-unit step on base_yaw: the joint moves at most max_velocity per tick, spins up at most
    max_acceleration per tick, and needs at least 60 * 0.9 / 140 s to get 90 % of the way."""
    t, g = stream(lambda x: HOME + np.array([60.0, 0, 0, 0, 0]), 2.0)
    tr = sm.simulate(t, g, start=HOME)
    pos, vel = tr.position[:, YAW], tr.velocity[:, YAW]
    assert np.abs(np.diff(pos)).max() <= sm.VENDOR_MAX_VELOCITY * sm.DT + 1e-9
    assert np.abs(vel).max() <= sm.VENDOR_MAX_VELOCITY + 1e-9
    growing = np.diff(np.abs(vel)) > 0
    assert np.diff(np.abs(vel))[growing].max() <= sm.VENDOR_MAX_ACCELERATION * sm.DT + 1e-9
    t90 = tr.t[np.argmax(pos >= 54.0)]
    assert t90 >= 0.9 * 60.0 / sm.VENDOR_MAX_VELOCITY
    assert pos[-1] >= 60.0 - sm.VENDOR_BACKLASH - sm.VENDOR_DEADBAND - 0.1     # it does arrive, play short
    assert (np.diff(pos) >= 0).all()                                            # and does not overshoot


def test_writes_take_effect_after_the_command_latency_and_hold_until_the_next():
    """One write at t = 0.1 s: the servo still holds the old goal at the 0.10 s tick, has the new one at
    0.11 s (4 ms later there is no tick in between), and keeps it with nothing further written."""
    t = np.array([0.0, 0.1])
    g = np.array([HOME, HOME + np.array([20.0, 0, 0, 0, 0])])
    tr = sm.simulate(t, g, start=HOME, tail_s=0.5)
    assert tr.goal[10, YAW] == 0.0 and tr.goal[11, YAW] == 20.0
    assert (tr.goal[11:, YAW] == 20.0).all()
    assert tr.t[0] == 0.0 and tr.t[-1] == pytest.approx(0.6)


def test_per_joint_tau_spans_the_measured_delivery_ratios():
    """The physics knob for 'yaw delivers 0.48, pitch 0.88 at beat rates' is a per-joint lag: the same
    +-20 wiggle at the 128 bpm back-and-forth rate through tau 0.27 s and 0.08 s, scored the way
    analysis/beattrace.py scored the arm."""
    t, g = stream(lambda x: HOME + 20.0 * np.sin(2 * np.pi * 1.07 * x) * np.ones(5), 6.0)
    p = sm.ServoParams(tau_s={"base_yaw": 0.27, "base_pitch": 0.08}, stops=sm.calibration_stops())
    tr = sm.simulate(t, g, start=HOME, params=p)
    yaw_ratio, yaw_lag = tr.beattrace_ratio("base_yaw")
    pitch_ratio, pitch_lag = tr.beattrace_ratio("base_pitch")
    assert abs(yaw_ratio - sm.MEASURED_DELIVERY["base_yaw"]) <= 0.08
    assert abs(pitch_ratio - sm.MEASURED_DELIVERY["base_pitch"]) <= 0.08
    assert 0.05 <= pitch_lag <= yaw_lag <= 0.25
    elbow_ratio, elbow_lag = tr.beattrace_ratio("elbow_pitch")             # the default tau, in between
    assert pitch_ratio > elbow_ratio > yaw_ratio
    assert sm.MEASURED_LAG_S[0] - 0.02 <= elbow_lag <= sm.MEASURED_LAG_S[1] + 0.02


def test_delivery_override_scales_fast_motion_only():
    """delivery={'base_yaw': r} attenuates a beat-rate wiggle on top of the physics and leaves a slow
    tracking move to land in full (later)."""
    t, g = stream(lambda x: HOME + np.array([30.0 * np.sin(2 * np.pi * 1.33 * x), 0, 0, 0, 0]), 4.0)
    plain = sm.simulate(t, g, start=HOME, params=sm.ServoParams(stops=sm.calibration_stops()))
    scaled = sm.simulate(t, g, start=HOME, params=sm.ServoParams(stops=sm.calibration_stops(),
                                                                 delivery={"base_yaw": 0.48}))
    plain_ratio, scaled_ratio = plain.beattrace_ratio("base_yaw")[0], scaled.beattrace_ratio("base_yaw")[0]
    # The override multiplies what the physics delivers (0.50 here: the lag and the spin-up ceiling at
    # 1.33 Hz); the spin-up limit bites less at the smaller amplitude, so the product is a little over
    # 0.48 x plain rather than exactly it.
    assert 0.48 * plain_ratio - 0.05 <= scaled_ratio <= 0.8 * plain_ratio
    assert np.array_equal(scaled.position[:, PITCH], plain.position[:, PITCH])   # only the named joint
    t, g = stream(ramp(YAW, 40.0, 4.0), 4.0)
    slow = sm.simulate(t, g, start=HOME, params=sm.ServoParams(delivery={"base_yaw": 0.48}), tail_s=4.0)
    assert slow.final()["base_yaw"] >= 0.9 * 40.0


# ----------------------------------------------------------------------------- the stop
def test_yaw_floor_is_honoured_and_reported():
    """cha_turn from home: the joint goes left, stops dead at STOP_YAW_MIN, is reported banging for as long
    as the goal is beyond the stop, keeps its (large) tracking error meanwhile, and is free again once
    the clip turns right."""
    t, g = stream(cha_turn, 4.0)
    tr = sm.simulate(t, g, start=HOME)
    yaw = tr.position[:, YAW]
    assert yaw.min() == sm.STOP_YAW_MIN
    assert tr.measured[:, YAW].min() == sm.STOP_YAW_MIN
    pressed = tr.banging[:, YAW]
    assert 1.0 <= pressed.sum() * sm.DT <= 2.5                      # the -65 leg lasts ~1.9 s at the servo
    assert (yaw[pressed] == sm.STOP_YAW_MIN).all()
    assert np.abs(tr.tracking_error[pressed, YAW]).max() > 55.0    # -65 asked, -4.5 delivered
    assert tr.any_banging.any() and not tr.banging[:, [i for i in range(5) if i != YAW]].any()
    assert yaw.max() >= 60.0                                        # the right leg is delivered
    assert not pressed[-1] and tr.banging_seconds()["base_pitch"] == 0.0


def test_other_joints_are_unaffected_by_a_banging_joint():
    t, g = stream(cha_turn, 4.0)
    g_still = g.copy()
    g_still[:, YAW] = HOME[YAW]
    banging = sm.simulate(t, g, start=HOME)
    still = sm.simulate(t, g_still, start=HOME)
    assert banging.banging[:, YAW].any() and not still.banging.any()
    for name in sm.JOINTS:
        if name != "base_yaw":
            assert np.array_equal(banging.column(name, "position"), still.column(name, "position"))


def test_a_joint_started_beyond_the_stop_drives_to_it_and_stays_pressed():
    """yaw_limits.py, seen tonight: a servo already past its limit parks at the limit."""
    beyond = HOME + np.array([-10.0, 0, 0, 0, 0])
    t, g = stream(lambda x: beyond, 1.0)
    tr = sm.simulate(t, g, start=beyond)
    assert (tr.position[:, YAW] == sm.STOP_YAW_MIN).all()
    assert tr.banging[:, YAW].all()


def test_setting_the_stop_to_minus_100_removes_the_floor():
    """Once the stop is fixed the calibration's -100 is the limit: the same clip goes left in full and
    nothing is reported banging."""
    assert sm.default_stops(sm.CALIBRATION_MIN) == sm.calibration_stops()
    t, g = stream(cha_turn, 4.0)
    tr = sm.simulate(t, g, start=HOME, params=sm.ServoParams(stops=sm.calibration_stops()))
    assert tr.position[:, YAW].min() <= -65.0 + 0.04 * 65
    assert not tr.banging.any()
    tr = sm.simulate(t, g, start=HOME, params=sm.ServoParams(stops={"base_yaw": (-100.0, 100.0)}))
    assert not tr.banging.any()


# ----------------------------------------------------------------------------- determinism, inputs
def test_model_is_deterministic():
    t, g = stream(cha_turn, 4.0)
    a = sm.simulate(t, g, start=HOME, params=sm.ServoParams(tau_s={"base_yaw": 0.2}))
    b = sm.simulate(t, g, start=HOME, params=sm.ServoParams(tau_s={"base_yaw": 0.2}))
    for name in ("t", "goal", "position", "measured", "velocity", "banging"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert a.summary() == b.summary()


def test_goals_accept_a_mapping_and_pack_rows():
    t, g = stream(ramp(YAW, 10.0, 1.0), 1.0)
    by_name = {j: g[:, i] for i, j in enumerate(sm.JOINTS)}
    assert np.array_equal(sm.simulate(t, by_name, start=HOME).measured, sm.simulate(t, g, start=HOME).measured)
    rows = [{"timestamp": 1000.0 + x, **{f"{j}.pos": g[k, i] for i, j in enumerate(sm.JOINTS)}}
            for k, x in enumerate(t)]
    t2, g2 = sm.goals_from_rows(rows)
    assert np.allclose(t2 - 1000.0, t) and np.array_equal(g2, g)
    assert sm.simulate(t2, g2).t[0] == 1000.0


def test_bad_parameters_are_refused():
    with pytest.raises(ValueError):
        sm.ServoParams(tau_s=0.005)                       # under one tick: the discrete loop would overshoot
    with pytest.raises(ValueError):
        sm.ServoParams(tau_s={"base_jaw": 0.1})
    with pytest.raises(ValueError):
        sm.ServoParams(stops={"base_yaw": (5.0, -5.0)})
    with pytest.raises(ValueError):
        sm.ServoParams(delivery={"base_yaw": 0.0})
    with pytest.raises(ValueError):
        sm.simulate([0.0, -0.1], np.zeros((2, 5)))
    with pytest.raises(ValueError):
        sm.simulate([0.0], np.zeros((1, 4)))
