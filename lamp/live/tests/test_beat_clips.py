"""beat_clips v4: the pure pieces (beats, easing, gain table, envelope clamp, per-joint amplitude,
variants, accents, the reach layer), the head travel the choreography is actually for -- measured
through the calibrated forward kinematics, because the joint columns cannot show it -- and one
end-to-end generation at 128 bpm (needs the vendor robot description, which is restricted material
read in place from the scratchpad and never copied)."""
import csv
import hashlib
import inspect
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import beat_clips as bc  # noqa: E402

ROBOTDESC = bc.DEFAULT_ROBOTDESC
HAVE_ROBOT = (ROBOTDESC / "pi5_feetech_r1" / "robot.urdf").exists() and (ROBOTDESC / "lelamp-calibration.json").exists()
COMBOS = [(t, v) for t in bc.TIERS for v in bc.VARIANTS]


def rms(U):
    return float(np.sqrt(np.mean(np.asarray(U, dtype=float) ** 2)))


# ----------------------------------------------------------------------------- pure pieces
def test_beat_instants_land_after_the_hold():
    t = bc.beat_instants(120)
    assert len(t) == bc.BEATS + 1
    assert t[0] == pytest.approx(bc.HOLD_S)
    assert np.allclose(np.diff(t), 0.5)
    assert t[-1] == pytest.approx(bc.HOLD_S + 8 * 0.5)
    assert bc.clip_seconds(120) == pytest.approx(2 * bc.HOLD_S + 4.0)


def test_ease_is_cosine_monotone_with_flat_ends():
    u = np.linspace(0, 1, 101)
    e = bc.ease(u)
    assert e[0] == pytest.approx(0.0) and e[-1] == pytest.approx(1.0)
    assert bc.ease(0.5) == pytest.approx(0.5)
    assert (np.diff(e) >= -1e-12).all()
    assert abs(e[1] - e[0]) < abs(e[51] - e[50])           # slow at the ends, fast in the middle
    assert bc.ease(-1) == 0.0 and bc.ease(2) == 1.0       # clipped outside 0..1


def test_the_hardware_envelope_constants_are_what_the_arm_was_measured_against():
    """The numbers the whole safety argument rests on, pinned as VALUES and not as their own names.

    Every other test in this file spells these as bc.JOINT_MAX, bc.SPEED_LIMIT and so on, which is
    right for reading but cannot fail: `out[bc.YAW] == bc.JOINT_MAX` holds whatever JOINT_MAX is set
    to. This test is the one place that says what they actually are, so moving one is a deliberate
    edit here rather than a silent drift in the envelope the clips are validated against.

    v3 -> v4 (the tempo-sized rewrite), for the ones that moved:
    * JOINT_MAX 94 -> 98. The servo's calibrated range is +-100; the margin to it went from 6 units
      to 2 once the calibration was trusted, which is where the extra yaw for `facing` came from.
    * SPEED_DESIGN/SPEED_LIMIT 140 -> 380. 140 was the vendor SIMULATION limit read as a hardware
      one; 380 is what the vendor's own shipped animation clips actually reach, so it is the
      runtime's demonstrated ceiling rather than a guess.
    * MOVE_FRAC 0.7 -> 0.85: more of each beat is the move, less of it the punctuating hold, which
      is free amplitude at the same speed.
    WP_MAX 60 did not move -- the wrist stops physically near +94 and 60 is the margin to it.
    """
    assert (bc.JOINT_MAX, bc.WP_MAX) == (98.0, 60.0)
    assert (bc.SPEED_DESIGN, bc.SPEED_LIMIT) == (380.0, 380.0)
    assert bc.MOVE_FRAC == 0.85
    assert (bc.BP_MIN, bc.BP_FLIP, bc.ELBOW_FLIP) == (-65.0, -52.0, -45.0)
    assert bc.SPEED_MARGIN == 0.99                      # design to 99 % so rounding never trips the validator
    assert bc.JOINT_MAX < 100.0 and bc.WP_MAX < bc.JOINT_MAX     # both inside the calibrated span
    assert np.allclose(bc.START, [0.0, -49.0, -22.0, 0.0, 30.0])
    # The ZMP bounds the tipping criterion is checked against, and the head's clearance over the table.
    #
    # ZMP_Y_MAX is the FORWARD tipping bound and it is the one that has been moving: 0.050 (half the
    # base footprint) -> 0.070 -> 0.085, all on 2026-09-20, each step taken to let the dance reach
    # lower and further over the table. The lamp actually tips when the ZMP leaves the base footprint
    # at 0.100 (safety.yaml cylinder_radius_m, read below off the model rather than trusted from a
    # comment), so 0.085 is 15 mm of margin on a criterion computed from a 5-frame-smoothed CoM.
    # It is pinned here, as a value, so that moving it again is a deliberate edit in one obvious
    # place and not a number that drifts while the choreography is being tuned.
    # ZMP_Y_MIN was -0.012 until 2026-09-20: 12 mm on a footprint that reaches 100 mm backward too.
    # With the bottom of the dance at the table's margin the fastest tempi overshot it by 0.4 mm on
    # the way back up and 14 clips were refused; -0.030 keeps 70 % of the footprint behind the lamp.
    assert (bc.HEAD_Y_MIN, bc.ZMP_Y_MIN, bc.ZMP_Y_MAX) == (0.020, -0.030, 0.085)
    if HAVE_ROBOT:
        tipping = bc.Validator(ROBOTDESC).model.base["radius"]
        assert tipping == pytest.approx(0.100)
        # the bound must stay strictly inside the footprint the lamp really goes over at, with the
        # remaining margin stated rather than implied
        assert bc.ZMP_Y_MAX < tipping
        assert tipping - bc.ZMP_Y_MAX == pytest.approx(0.015, abs=0.0005)


def test_the_speed_budget_rises_with_tempo_and_never_passes_the_hard_limit():
    """speed_for() is v4: a FIXED budget made fast songs look tamer than slow ones (a shorter beat
    buys a smaller excursion out of the same units/s), so the budget now rises with the tempo --
    260 units/s at 80 bpm to the vendor's 380 at 170 and above. SPEED_LIMIT stays 380 and stays the
    hard ceiling every clip is validated against, so the budget may reach it but never exceed it.

    Pinned here because the assertions elsewhere are written against speed_for(bpm) itself and would
    hold for any shape at all; this is what says the shape is a plateau, a ramp, then a plateau."""
    # SPEED_AT_FAST is 340 and not the vendor's own 380: with the bottom of the dance down at the
    # table's 3 cm margin, the fastest tempi were driving the ZMP onto the BACKWARD bound (-0.012)
    # on the way back up, and the last 40 units/s of top end cost more tipping margin than they were
    # worth. SPEED_DESIGN and SPEED_LIMIT stay 380, so the budget no longer reaches the hard ceiling.
    assert (bc.SPEED_AT_SLOW, bc.SPEED_AT_FAST) == (260.0, 340.0)
    assert (bc.BPM_SLOW, bc.BPM_FAST) == (80.0, 170.0)
    assert bc.speed_for(80) == 260.0 and bc.speed_for(170) == 340.0
    assert bc.speed_for(125) == pytest.approx(300.0)          # the midpoint of the ramp
    assert bc.speed_for(100) == pytest.approx(277.78, abs=0.01)
    # flat outside the ramp: a tempo the tracker cannot reach is not extrapolated into more speed
    for slow in (0.0, 40.0, 79.9, bc.BPM_SLOW):
        assert bc.speed_for(slow) == bc.SPEED_AT_SLOW
    for fast in (bc.BPM_FAST, 180.0, 214.0, 300.0):
        assert bc.speed_for(fast) == bc.SPEED_AT_FAST
    # monotone over everything the tracker reports, and never past the limit the validator enforces
    budgets = [bc.speed_for(b) for b in np.linspace(40, 300, 261)]
    assert budgets == sorted(budgets)
    assert max(budgets) <= bc.SPEED_LIMIT
    assert all(b >= bc.SPEED_AT_SLOW for b in budgets)
    # the invariant that must hold whatever the two endpoints are set to next
    assert bc.SPEED_AT_SLOW <= bc.SPEED_AT_FAST <= bc.SPEED_LIMIT
    # and the budget is what sizes the beat: a faster song moves faster AND still travels
    assert bc.max_move(180) > bc.max_move(180, bc.SPEED_AT_SLOW)


# ----------------------------------------------------------------------------- the speed dial (2026-09-20)
SPEED_DIALS = (0.0, 0.25, 0.5, 0.75, 1.0)


def test_the_speed_dial_rides_on_top_of_the_tempo_budget():
    """The operator's dance-speed dial, new on 2026-09-20 (lamp_show's Control lamp.speed, 0..1,
    default 1.0). Letting the TEMPO pick the budget on its own made a fast song frantic with no way
    to say so from the dashboard, so speed_for takes a second argument and interpolates:

        SPEED_CALM + speed * (this tempo's budget - SPEED_CALM)

    Dial 1.0 is therefore EXACTLY the tempo-ramped budget speed_for(bpm) always returned -- not
    nearly, exactly, which is what keeps every clip that does not pass the dial byte-identical (see
    test_the_speed_dial_at_one_is_the_clip_that_has_no_dial) -- and the dial can only lower it.

    Measured at 160 bpm: 150.0 u/s at dial 0.00, 195.3 at 0.25, 240.6 at 0.50, 285.8 at 0.75 and
    331.1 at 1.00. Pinned here, like SPEED_AT_SLOW/SPEED_AT_FAST above, because every other
    assertion in this file is written against speed_for() itself and would hold for any shape."""
    assert bc.SPEED_CALM == 150.0                     # pinned as a value: the calm end of the dial
    assert bc.SPEED_CALM < bc.SPEED_AT_SLOW           # ... and below even the slowest tempo's own budget
    assert [round(bc.speed_for(160, s), 1) for s in SPEED_DIALS] == [150.0, 195.3, 240.6, 285.8, 331.1]
    for bpm in (40, 80, 100, 127.3, 128, 160, 170, 180, 214, 300):
        # 1.0 is the budget that has no dial at all, to the bit; 0.0 is SPEED_CALM whatever the tempo
        assert bc.speed_for(bpm, 1.0) == bc.speed_for(bpm), bpm
        assert bc.speed_for(bpm, 0.0) == bc.SPEED_CALM, bpm
        # the dial is a fader between those two, monotone, and it NEVER raises the budget
        budgets = [bc.speed_for(bpm, s) for s in np.linspace(0.0, 1.0, 101)]
        assert budgets == sorted(budgets), bpm
        assert budgets[0] == bc.SPEED_CALM and budgets[-1] == bc.speed_for(bpm)
        assert all(bc.SPEED_CALM <= b <= bc.speed_for(bpm) for b in budgets), bpm
        assert max(budgets) <= bc.SPEED_LIMIT, bpm
        # the halfway point is halfway, so the dial is linear and not some curve the dashboard
        # would have to know about
        assert bc.speed_for(bpm, 0.5) == pytest.approx((bc.SPEED_CALM + bc.speed_for(bpm)) / 2)
        # out of range is clamped to 0..1 rather than extrapolated, and NaN takes the CALMEST value,
        # exactly as bold_multiplier reads a slider the dashboard failed to send a number for
        assert bc.speed_for(bpm, 2.0) == bc.speed_for(bpm, 1.0), bpm
        assert bc.speed_for(bpm, 1e9) == bc.speed_for(bpm, 1.0), bpm
        assert bc.speed_for(bpm, -1.0) == bc.SPEED_CALM, bpm
        assert bc.speed_for(bpm, float("nan")) == bc.SPEED_CALM, bpm
        assert bc.speed_for(bpm, float("-inf")) == bc.SPEED_CALM, bpm
        assert bc.speed_for(bpm, float("inf")) == bc.speed_for(bpm, 1.0), bpm
    # the dial reaches the beat through max_move and beat_budget, which carry it as a keyword
    for bpm in (80, 128, 180):
        for s in SPEED_DIALS:
            assert bc.max_move(bpm, speed=s) == pytest.approx(
                bc.speed_for(bpm, s) * bc.MOVE_FRAC * (60 / bpm) / (np.pi / 2), rel=1e-12)
            assert bc.beat_budget(bpm, speed=s) == pytest.approx(bc.max_move(bpm, speed=s) * bc.SPEED_MARGIN * 0.995)
            assert bc.max_move(bpm, speed=s) <= bc.max_move(bpm) + 1e-12       # never faster than no dial
            assert bc.beat_budget(bpm, speed=s) <= bc.beat_budget(bpm) + 1e-12
        assert bc.max_move(bpm, speed=1.0) == bc.max_move(bpm)                 # exactly, again
        assert bc.beat_budget(bpm, speed=1.0) == bc.beat_budget(bpm)
        # an explicit `limit` is still the caller's own and the dial does not touch it
        assert bc.max_move(bpm, 200.0, speed=0.0) == bc.max_move(bpm, 200.0)


def test_max_move_is_the_cosine_speed_budget():
    # a one-beat move of D units with cosine easing peaks at (pi/2) * D / (MOVE_FRAC * P), against
    # this tempo's own budget (speed_for: SPEED_AT_SLOW at BPM_SLOW rising to SPEED_AT_FAST at BPM_FAST;
    # the top end was 380, the vendor's own, until the bottom of the dance reached the table's margin
    # and the fastest tempi started driving the ZMP onto the backward bound on the way back up)
    for bpm in (80, 132, 180):
        D = bc.max_move(bpm)
        assert (np.pi / 2) * D / (bc.MOVE_FRAC * 60 / bpm) == pytest.approx(bc.speed_for(bpm))
        assert bc.max_move(bpm, bc.SPEED_DESIGN) >= D and bc.speed_for(bpm) <= bc.SPEED_LIMIT
    assert bc.max_move(80) > bc.max_move(132) > bc.max_move(180)
    assert bc.speed_for(bc.BPM_SLOW) == bc.SPEED_AT_SLOW and bc.speed_for(bc.BPM_FAST) == bc.SPEED_AT_FAST
    assert bc.max_move(132) == pytest.approx(bc.speed_for(132) * bc.MOVE_FRAC * (60 / 132) / (np.pi / 2), rel=1e-9)


def test_gain_table_scales_excursion_from_start_only():
    assert bc.GAINS == {"base_yaw": 1.8, "base_pitch": 1.15, "elbow_pitch": 1.2, "wrist_roll": 1.3, "wrist_pitch": 1.1}
    assert np.allclose(bc.GAIN, [1.8, 1.15, 1.2, 1.3, 1.1])
    pose = bc.START + 10.0
    g = bc.apply_gain(pose)
    assert np.allclose(g - bc.START, [18.0, 11.5, 12.0, 13.0, 11.0])
    assert np.allclose(bc.apply_gain(bc.START), bc.START)


def test_gains_override_is_validated():
    assert np.allclose(bc.gain_vector(None), bc.GAIN)
    g = bc.gain_vector({"base_yaw": 1.6, "wrist_pitch": 1.0})
    assert np.allclose(g, [1.6, 1.15, 1.2, 1.3, 1.0])
    assert bc.parse_gains(None) == {} and bc.parse_gains("") == {}
    assert bc.parse_gains('{"base_yaw": 1.6}') == {"base_yaw": 1.6}
    with pytest.raises(ValueError):
        bc.gain_vector({"knee": 1.0})
    with pytest.raises(ValueError):
        bc.gain_vector({"base_yaw": 9.0})
    with pytest.raises(ValueError):
        bc.parse_gains("[1, 2]")


def test_gains_override_from_file_changes_the_command(tmp_path):
    f = tmp_path / "measured.json"
    f.write_text('{"base_yaw": 1.2}')
    gains = bc.parse_gains(f"@{f}")
    g = bc.gain_vector(gains)
    lo = bc.commanded("groove", 128, "a", g)
    hi = bc.commanded("groove", 128, "a")
    # the same pattern commands a smaller yaw with a smaller gain at the same speed budget ... no:
    # the per-joint scale is chosen against the SAME speed limit, so the commanded peak speed is the
    # same and the commanded amplitude too; what changes is what the arm is asked relative to the pattern
    assert bc.peak_speed(lo)[bc.YAW] <= bc.SPEED_DESIGN and bc.peak_speed(hi)[bc.YAW] <= bc.SPEED_DESIGN
    # At the library's own gains no joint is speed-bound any more -- the yaw ceiling and the lift
    # envelope bind first -- so the scale is exercised at a gain that does bind, which is what the
    # --gains override is for. Less gain, more of the pattern fits.
    raw = bc.raw_trajectory("hype", 180, "a")
    big = bc.gain_vector({"base_yaw": 3.5})
    s_lo, s_hi = bc.joint_scales(raw, 180, g), bc.joint_scales(raw, 180, big)
    assert s_lo[bc.YAW] == 1.0 and s_hi[bc.YAW] < 1.0
    assert bc.peak_speed(bc.apply_gain(bc.scaled(raw, s_hi), big))[bc.YAW] <= bc.SPEED_DESIGN


def test_envelope_clamp_rules():
    c = bc.clamp_envelope
    assert c(np.array([0.0, -70.0, -60.0, 0.0, 30.0]))[bc.BP] == pytest.approx(-65.0)        # floor -65
    assert c(np.array([0.0, -58.0, -20.0, 0.0, 30.0]))[bc.BP] == pytest.approx(-52.0)        # flip region
    assert c(np.array([0.0, -58.0, -44.0, 0.0, 30.0]))[bc.BP] == pytest.approx(-52.0)        # elbow -44: still above -45
    assert c(np.array([0.0, -58.0, -47.0, 0.0, 30.0]))[bc.BP] == pytest.approx(-54.0)        # blend: 2 units under -45
    assert c(np.array([0.0, -58.0, -60.0, 0.0, 30.0]))[bc.BP] == pytest.approx(-58.0)        # elbow lifted: allowed
    out = c(np.array([120.0, -49.0, -22.0, -120.0, 80.0]))
    assert out[bc.YAW] == bc.JOINT_MAX and out[bc.WR] == -bc.JOINT_MAX and out[bc.WP] == bc.WP_MAX
    assert np.allclose(c(bc.START), bc.START)
    # continuous in the elbow: sweeping the elbow through -45..-58 at base_pitch -60 gives no jump
    el = np.linspace(-30, -65, 200)
    U = np.stack([np.zeros_like(el), np.full_like(el, -60.0), el, np.zeros_like(el), np.full_like(el, 30.0)], 1)
    bp = c(U)[:, bc.BP]
    assert bp[0] == pytest.approx(-52.0) and bp[-1] == pytest.approx(-60.0)
    assert np.abs(np.diff(bp)).max() < 0.5
    assert not bc.envelope_violations(c(U))
    assert bc.envelope_violations(U)                        # and the raw one is a violation
    # every clamped frame with base_pitch < -52 has the elbow at or under -45
    V = c(U)
    assert (V[V[:, bc.BP] < -52 - 1e-9][:, bc.EL] <= -45 + 1e-9).all()


def test_peak_speed_is_frame_difference_times_fps():
    U = np.array([[0.0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0], [3.0, 0, 0, 0, 0]])
    assert bc.peak_speed(U)[0] == pytest.approx(2.0 * bc.FPS)
    assert bc.peak_speed(U[:1]).max() == 0.0


def test_trajectory_lands_keyframes_on_the_beat_and_holds_the_ends():
    bpm = 120
    keys = bc.keyframes("groove", "a")
    U = bc.trajectory(keys, bpm)
    assert len(U) == round(bc.clip_seconds(bpm) * bc.FPS)
    hold = int(bc.HOLD_S * bc.FPS)
    assert np.allclose(U[:hold], bc.START) and np.allclose(U[-hold:], bc.START)
    for k, t in enumerate(bc.beat_instants(bpm)):
        i = int(round(t * bc.FPS))
        assert np.allclose(U[i], keys[k], atol=1e-6), f"beat {k}"
        if 0 < k < bc.BEATS:                     # the first 1 - MOVE_FRAC of the next beat is a hold
            j = i + int(0.5 * (1 - bc.MOVE_FRAC) * 60 / bpm * bc.FPS)
            assert np.allclose(U[j], keys[k], atol=1e-6)


def test_keyframes_start_and_end_at_start_pose_for_every_variant():
    for tier, variant in COMBOS:
        K = bc.keyframes(tier, variant)
        assert K.shape == (bc.BEATS + 1, 5)
        assert np.allclose(K[0], bc.START) and np.allclose(K[-1], bc.START), (tier, variant)
        assert not np.allclose(K[1:-1], bc.START)             # and there is a dance in between
    with pytest.raises(ValueError):
        bc.keyframes("waltz", "a")
    with pytest.raises(ValueError):
        bc.keyframes("groove", "z")


def test_the_swing_lands_where_the_head_is_furthest_from_the_yaw_axis():
    """The v4 rule, and the whole reason the dance is wide: sideways travel is the head's distance from
    the yaw axis times the sine of the turn, so a beat is only allowed to spend its swing where the arm
    is out. LIFTS and YAWS are written as one table and this is the invariant that ties them together.
    Read off the tables, which is where the pairing is authored; lift_profile only ever moves a beat
    TOWARD its neighbours, so a rate-limited phrase is a smaller version of this, never a wider one."""
    assert set(bc.YAWS) == {(t, v) for t in bc.TIERS if t != "build" for v in bc.VARIANTS}
    for (tier, variant), row in bc.YAWS.items():
        L = np.array(bc.LIFTS[(tier, variant)], dtype=float)
        y = np.abs(np.array(row, dtype=float))
        assert len(row) == len(L) == bc.BEATS - 1
        for k, (lift, turn) in enumerate(zip(L, y), start=1):
            if turn >= 0.95 * y.max():                       # the ends of this phrase's swing ...
                assert 0.45 <= lift <= 0.75, (tier, variant, k, lift)     # ... where the reach is
            if lift <= 0.05 or lift >= 0.95:                 # the beats that visit BOTTOM or TOP ...
                assert turn <= 0.4, (tier, variant, k, turn)              # ... pass near the centre


def test_reach_only_extends_the_arm_where_the_swing_can_use_it():
    L = np.linspace(0.0, 1.0, 21)
    full = bc.reach_layer(L, np.full_like(L, bc.YAW_MAX))
    assert full[0] == 0.0 and full[-1] == pytest.approx(0.0, abs=1e-12)   # BOTTOM and TOP: the bare line
    assert full.max() == pytest.approx(bc.REACH) and L[full.argmax()] == pytest.approx(0.5)
    assert np.allclose(bc.reach_layer(L, np.zeros_like(L)), 0.0)          # a centred beat gets none of it
    assert np.allclose(bc.reach_layer(L, np.full_like(L, bc.YAW_MAX / 2)), full / 2)
    assert np.allclose(bc.reach_layer(L, np.full_like(L, 2 * bc.YAW_MAX)), full)     # and never more
    # it only ever RAISES base_pitch, so it cannot reach the -65 floor or open the flip region
    bare, out = bc.lift_pose(L), bc.lift_pose(L, full)
    assert (out[:, bc.BP] >= bare[:, bc.BP] - 1e-12).all()
    assert np.array_equal(out[:, [bc.YAW, bc.EL, bc.WR, bc.WP]], bare[:, [bc.YAW, bc.EL, bc.WR, bc.WP]])
    assert not bc.envelope_violations(out) and not bc.envelope_violations(bare)


@pytest.mark.skipif(not HAVE_ROBOT, reason=f"vendor robot description not available at {ROBOTDESC}")
def test_reach_takes_the_head_away_from_the_yaw_axis():
    """The measurement the reach exists for. When BOTTOM leaned back with the elbow folded, the bare
    BOTTOM..TOP line left the head only 0.114 m from the yaw axis at mid lift and the reach took it past
    0.16 (x1.4). Since 2026-09-20 BOTTOM reaches DOWN toward the table instead, which already carries
    the head out to 0.218 m at lift 0.5 and 0.203 at 0.6, so the reach has less left to add: 0.255 and
    0.247 (x1.17, x1.22), measured on the model. The pose with the reach must still be one the model
    allows. At either END of the lift the reach adds nothing at all (sin(pi * lift) is zero there):
    the bottom sits at 0.187 m and the top at 0.082, with or without it."""
    model = bc.Validator(ROBOTDESC).model

    def radius(pose):
        p = model.head(dict(zip(bc.JOINTS, pose)))["position"]
        return float(np.hypot(p[0], p[1]))

    for lift, bare_r in ((0.5, 0.218), (0.6, 0.203)):
        bare = bc.lift_pose(np.array([lift]))[0]
        out = bc.lift_pose(np.array([lift]), bc.reach_layer(np.array([lift]), np.array([bc.YAW_MAX])))[0]
        assert radius(bare) == pytest.approx(bare_r, abs=0.005), lift
        assert radius(out) > 1.15 * radius(bare), (lift, radius(bare), radius(out))
        assert not model.problems(dict(zip(bc.JOINTS, out))), lift
    for lift in (0.0, 1.0):
        bare = bc.lift_pose(np.array([lift]))[0]
        out = bc.lift_pose(np.array([lift]), bc.reach_layer(np.array([lift]), np.array([bc.YAW_MAX])))[0]
        assert radius(out) == pytest.approx(radius(bare), abs=1e-6), lift


def test_the_bar_accent_widens_beat_one_of_each_bar_up_to_the_ceiling():
    for tier in ("groove", "hype", "drop"):
        for variant in bc.VARIANTS:
            shape = np.array(bc.YAWS[(tier, variant)], dtype=float)
            ceiling = float(np.abs(shape).max())
            yaw, roll, _ = bc.yaw_layer(tier, variant, 128, bc.lift_profile(tier, variant, 128))
            amp = float(np.abs(yaw).max()) / ceiling
            for k in bc.ACCENT_BEATS:                              # beat 1 of each bar: x1.3, then clipped
                want = float(np.clip(shape[k - 1] * bc.ACCENT, -ceiling, ceiling))
                assert yaw[k] == pytest.approx(amp * want), (tier, variant, k)
                assert abs(yaw[k]) >= abs(amp * shape[k - 1]) - 1e-9   # never narrower than unaccented
            for k in (3, 7):                                       # the other bar beats stay nominal
                assert yaw[k] == pytest.approx(amp * shape[k - 1]), (tier, variant, k)
            assert np.abs(yaw).max() == pytest.approx(amp * ceiling)   # and nothing passes the ceiling
            if (tier, variant) != ("groove", "b"):     # ... which runs its own roll, at the yaw's rate
                assert np.allclose(roll, bc.ROLL * yaw)                # the counter-tilt follows the yaw
            assert yaw[0] == 0.0 and yaw[bc.BEATS] == 0.0
    # the build tier crouches on the beat and carries no accent at all
    for variant in bc.VARIANTS:
        yaw, _, _ = bc.yaw_layer("build", variant, 128, bc.lift_profile("build", variant, 128))
        assert not np.isclose(abs(yaw[1]), bc.ACCENT * abs(yaw[3])) or np.allclose(yaw, 0.0), variant


def test_the_tiers_are_sized_by_their_own_yaw_ceiling():
    for bpm in (80, 128, 180):
        for tier in ("groove", "hype", "drop"):
            ceiling = bc.YAW_CALM if tier == "groove" else bc.YAW_MAX
            for v in bc.VARIANTS:
                cmd = bc.commanded(tier, bpm, v)
                A = float(np.abs(cmd[:, bc.YAW]).max())
                assert A <= ceiling + 1e-9, (tier, v, bpm, A)
                assert (np.abs(cmd - bc.START).max(axis=0) > 2.0).all(), (tier, v, bpm)   # no dead joint
        calm = max(np.abs(bc.commanded("groove", bpm, v)[:, bc.YAW]).max() for v in bc.VARIANTS)
        loud = min(np.abs(bc.commanded(t, bpm, v)[:, bc.YAW]).max()
                   for t in ("hype", "drop") for v in bc.VARIANTS)
        assert loud > calm, bpm                      # the loud tiers swing wider at every bucket tempo


def test_the_a_variants_are_the_same_wheel_turned_opposite_ways():
    a, da = np.array(bc.YAWS[("hype", "a")]), np.array(bc.YAWS[("drop", "a")])
    assert np.allclose(da[2:], -a[:5])               # drop a is hype a, two beats later and mirrored
    assert bc.LIFTS[("hype", "a")][0] == 0.6 and bc.LIFTS[("drop", "a")][0] == 1.0   # ... after the hit
    for tier in ("hype", "drop"):
        L = np.array(bc.LIFTS[(tier, "a")])
        assert L.min() == 0.0 and L.max() == 1.0     # the wheel still visits BOTTOM and TOP


def test_the_b_variants_lean_out_halfway_between_the_floor_and_the_top():
    for tier in ("hype", "drop"):
        L, Y = np.array(bc.LIFTS[(tier, "b")]), np.array(bc.YAWS[(tier, "b")])
        assert Y[1] > 0 > Y[5] and abs(Y[1]) == abs(Y[5])          # the two leans are opposite and equal
        assert L.min() == 0.0 and L.max() >= 0.9                   # and the phrase travels floor to top
        for k in (1, 5):                                           # each lean is taken mid-travel ...
            assert L[k - 1] < L[k] < L[k + 1] or L[k - 1] > L[k] > L[k + 1], (tier, k)
            assert abs(Y[k]) > abs(Y[k - 1]) and abs(Y[k]) > abs(Y[k + 1]), (tier, k)


def test_the_c_variants_sweep_across_the_crowd_at_mid_lift():
    for tier in ("hype", "drop"):
        L, Y = np.array(bc.LIFTS[(tier, "c")]), np.array(bc.YAWS[(tier, "c")])
        assert list(Y) == bc.SWEEP_C and Y[2] == -1.0 and Y[4] == 1.0          # the ends of the sweep
        assert 0.45 <= L[2] <= 0.75 and 0.45 <= L[4] <= 0.75                   # ... taken with the arm out
        assert Y[3] == 0.0 and L[3] > L[2] and L[3] > L[4]                     # the apex, through the centre
        assert Y[0] * Y[-1] < 0                                                # it starts and ends opposite
        _, _, nod = bc.yaw_layer(tier, "c", 128, bc.lift_profile(tier, "c", 128))
        assert np.abs(nod[1:bc.BEATS]).max() == pytest.approx(bc.NOD)          # the nod rides the sweep
        assert nod[0] == 0.0 and nod[bc.BEATS] == 0.0


def test_the_drop_hits_on_beat_one_and_the_build_crouches_without_reaching_out():
    for v in bc.VARIANTS:
        assert bc.LIFTS[("drop", v)][0] == 1.0                     # every drop opens on the highest pose
        assert abs(bc.YAWS[("drop", v)][0]) <= 0.35                # ... taken near the centre, as a hit
    for variant in bc.VARIANTS:
        L = bc.lift_profile("build", variant, 128)
        assert L[0] == L[bc.BEATS] == pytest.approx(bc.START_LIFT)
        assert (np.diff(L[1:7]) <= 1e-9).all() and L[6] == pytest.approx(0.0)   # a progressive sink
        K = bc.commanded_keyframes("build", variant, 128)
        # no reach: the crouch is the bare BOTTOM..TOP line, shoulder back and down, not a lean forward
        assert np.allclose(K[1:bc.BEATS][:, [bc.BP, bc.EL]], bc.lift_pose(L[1:bc.BEATS])[:, [bc.BP, bc.EL]])
        assert np.allclose(K[0], bc.START) and np.allclose(K[bc.BEATS], bc.START)
    K = bc.commanded_keyframes("build", "a", 128)
    assert np.allclose(K[:, [bc.YAW, bc.WR]], 0.0)                 # a sinks straight down
    low = K[:, bc.BP] < bc.BP_FLIP - 1e-9                          # the flip rule holds at every keyframe
    assert (K[low][:, bc.EL] <= bc.ELBOW_FLIP + 1e-9).all() and K[:, bc.BP].min() >= bc.BP_MIN - 1e-9
    assert bc.commanded_keyframes("build", "b", 128)[6, bc.WR] > 15    # b leans as it sinks
    assert bc.commanded_keyframes("build", "c", 128)[5, bc.WP] != bc.commanded_keyframes("build", "c", 128)[6, bc.WP]


@pytest.mark.parametrize("bpm", [80, 127.3, 132, 180])
def test_expressive_hype_and_inherited_drop_tails_keep_all_command_gates(bpm):
    for tier, variant in itertools.product(("hype", "drop"), ("b", "c")):
        U = bc.commanded(tier, bpm, variant)
        assert np.isfinite(U).all()
        assert not bc.envelope_violations(U), (tier, variant, bpm)
        assert bc.peak_speed(U).max() <= bc.SPEED_LIMIT, (tier, variant, bpm)
        assert np.allclose(U[0], bc.START) and np.allclose(U[-1], bc.START)
        assert len(U) == round(bc.clip_seconds(bpm) * bc.FPS)


@pytest.mark.skipif(not HAVE_ROBOT, reason=f"vendor robot description not available at {ROBOTDESC}")
@pytest.mark.parametrize("bpm", [80, 128, 180])
def test_the_head_sweeps_as_wide_as_it_is_tall(bpm):
    """What the whole choreography is for, measured where it can be measured: the head's own path
    through the calibrated forward kinematics. v3 spanned 0.271 m vertically and 0.107 m sideways
    (0.081 m at 180 bpm); v4 spans 0.229..0.260 m sideways on every hype and drop clip while keeping
    the vertical. The joint columns cannot show this -- the same base_yaw buys 0.067 m of travel with
    the arm folded and 0.17 m with it out -- which is why this test runs the model."""
    v = bc.Validator(ROBOTDESC)
    for tier, variant in itertools.product(("hype", "drop"), bc.VARIANTS):
        U = bc.commanded(tier, bpm, variant)
        P = np.array([v.model.head(dict(zip(bc.JOINTS, u)))["position"] for u in U])
        span = P.max(axis=0) - P.min(axis=0)
        assert span[0] > 0.22, (tier, variant, bpm, span)          # sideways, against v3's 0.081..0.128
        assert span[2] > 0.20, (tier, variant, bpm, span)          # and still tall
        assert v.validate(U)["ok"], (tier, variant, bpm)
    # the vertical envelope is untouched: the tallest phrase still runs the lamp's full reach
    U = bc.commanded("hype", bpm, "a")
    z = np.array([v.model.head(dict(zip(bc.JOINTS, u)))["position"][2] for u in U])
    assert z.min() < 0.19 and z.max() > 0.42


@pytest.mark.skipif(not HAVE_ROBOT, reason=f"vendor robot description not available at {ROBOTDESC}")
@pytest.mark.parametrize("bpm", [80, 132, 180])
def test_expressive_gestures_rise_and_cross_in_calibrated_head_space(bpm):
    v = bc.Validator(ROBOTDESC)
    for tier, variant in itertools.product(("hype", "drop"), ("b", "c")):
        U = bc.commanded(tier, bpm, variant)
        report = v.validate(U)
        assert report["ok"], (tier, variant, bpm, report["reasons"])
        # Take the first complete frame at each beat, during its hold rather than before arrival.
        indices = np.ceil(bc.beat_instants(bpm) * bc.FPS).astype(int)
        heads = np.array([v.model.head(dict(zip(bc.JOINTS, U[i])))["position"] for i in indices])
        if variant == "b":                            # each phrase crosses the height, leaning one way
            # measured 2026-09-19: 0.191..0.267 m of head height per phrase across 80, 132 and 180 bpm,
            # so 0.15 is a real floor and not a formality. The phrase is most of the lamp's 0.27 m reach.
            for start in (1, 5):
                assert abs(heads[start + 2, 2] - heads[start, 2]) > 0.15, (tier, variant, bpm, heads)
            assert heads[2, 0] * heads[6, 0] < 0, (tier, variant, bpm, heads)
        else:                                         # c crosses the crowd, left of centre to right
            dx = heads[5, 0] - heads[3, 0]
            assert abs(dx) > 0.15, (tier, variant, bpm, dx)
            assert heads[3, 0] * heads[5, 0] < 0, (tier, variant, bpm, heads)


def test_shiver_overlay_is_on_eighth_notes_and_zero_at_start():
    bpm = 120
    n = round(bc.clip_seconds(bpm) * bc.FPS)
    t = np.arange(n) / bc.FPS
    S = bc.shiver_overlay("build", "a", bpm, t)
    assert np.allclose(bc.shiver_overlay("groove", "a", bpm, t), 0.0)
    assert np.allclose(S[:, [bc.BP, bc.EL, bc.WR, bc.WP]], 0.0)
    hold = int(bc.HOLD_S * bc.FPS)
    assert np.allclose(S[:hold + 1], 0.0) and np.allclose(S[-hold:], 0.0)
    P = 60 / bpm
    for k in range(2, 6):                                     # inside the window: beats and eighths alternate
        i_beat, i_eighth = int(round((bc.HOLD_S + k * P) * bc.FPS)), int(round((bc.HOLD_S + (k + 0.5) * P) * bc.FPS))
        assert S[i_beat - 1:i_beat + 2, bc.YAW].min() == pytest.approx(-4.0, abs=0.1)      # 30 fps: the nearest frame
        assert S[i_eighth - 1:i_eighth + 2, bc.YAW].max() == pytest.approx(+4.0, abs=0.1)
    assert np.abs(bc.shiver_overlay("build", "b", bpm, t)[:, bc.WR]).max() == pytest.approx(6.0, abs=0.1)
    assert np.abs(bc.shiver_overlay("build", "c", bpm, t)[:, bc.WP]).max() == pytest.approx(3.0, abs=0.1)


@pytest.mark.parametrize("bpm", [80, 132, 180])
def test_the_tempo_budget_sizes_the_phrase_and_the_scale_has_nothing_left_to_trim(bpm):
    """v3 wrote each pattern at a nominal size and let joint_scales shrink the commanded trajectory to
    the speed limit, so a scale below 1.0 was the normal case and this test watched every bound joint
    sit within 2 % of the budget. v4 rate-limits the lift and the yaw to beat_budget(bpm) while it
    writes the phrase, so at the library's own gains NOTHING is speed-bound at any tempo the demo can
    reach: every scale is exactly 1.0, at every bucket bpm (checked 2026-09-19 over all of BPMS).

    That is the stronger statement, not the weaker one -- it says the choreography needed no trimming
    at all. But it also means v3's 2 % check had quietly stopped executing here, so it is kept below at
    a gain that does still bind, the same way the --gains tests reach for one."""
    budget = bc.speed_for(bpm) * bc.SPEED_MARGIN
    for tier, variant in COMBOS:
        raw = bc.raw_trajectory(tier, bpm, variant)
        s = bc.joint_scales(raw, bpm)
        assert (0 < s).all() and (s <= 1).all()
        if bpm <= bc.BPM_FAST:
            assert np.allclose(s, 1.0), (tier, variant, s)
        else:
            # past BPM_FAST the budget stops rising with the tempo (SPEED_AT_FAST, 340 since the bottom
            # of the dance reached the table) while the beat keeps shortening, so a joint may now need
            # a trim: measured 2026-09-20, hype a's base_pitch at 180 bpm scales to 0.975. A trimmed
            # joint must sit within 2 % of the budget, exactly v3's rule.
            pre = bc.peak_speed(bc.apply_gain(bc.scaled(raw, s)))
            for j in range(5):
                if s[j] < 1.0:
                    assert pre[j] >= budget * 0.98, (tier, variant, j, pre[j], budget)
        cmd = bc.commanded(tier, bpm, variant)
        speed = bc.peak_speed(cmd)
        # against THIS tempo's own budget (speed_for: 260 units/s at 80 bpm, 380 at 170 and above) and
        # not the bare 380 ceiling, which at 80 bpm would be 46 % of slack. Measured over the whole
        # bucket library the worst clip reaches 0.993 of its own budget, so this is tight.
        assert speed.max() <= budget + 1e-9, (tier, variant, bpm, speed)
        assert np.allclose(cmd[0], bc.START) and np.allclose(cmd[-1], bc.START)
        assert not bc.envelope_violations(cmd)
        # the commanded clip IS the gained design, frame for frame, once the budget's own per-joint
        # scale is carried. This was written as `clamp_envelope(apply_gain(raw))` while every scale
        # was 1.0 at every tempo; past BPM_FAST that is no longer so (hype a and drop a trim
        # base_pitch to 0.975 at 180, build c trims base_yaw to 0.985), so the scale has to be in it.
        assert np.allclose(cmd, bc.clamp_envelope(bc.apply_gain(bc.scaled(raw, s)))), (tier, variant, bpm)
        if np.allclose(s, 1.0):                  # ... and where nothing was trimmed, the old form exactly
            assert np.allclose(cmd, bc.clamp_envelope(bc.apply_gain(raw))), (tier, variant, bpm)
    # ... and the phrase SPENDS that budget rather than leaving it on the table: the lift's slowest
    # joint takes the whole of it on its biggest beat (ratio 1.00000 at 80, 128, 132 and 180)
    steps = [np.abs(np.diff(bc.commanded_keyframes(t, v, bpm), axis=0))[:, bc.LIFT_JOINT].max() for t, v in COMBOS]
    assert max(steps) == pytest.approx(bc.beat_budget(bpm))
    # This used to be the max over ALL five joints, because the lift was the only thing rate-limited
    # and nothing else came near it. It is the lift joint's own step now: REACH puts the lift AND the
    # lean on base_pitch in the same beat while lift_profile rate-limits only the lift, so at 180 bpm
    # hype a asks base_pitch for 62.71 units against a 60.41 budget -- 3.8 % over, which joint_scales
    # then pays for with the 0.975 trim above. Pinned as a ratio so a bigger overspend fails here
    # rather than being quietly trimmed; below BPM_FAST the beat is long enough that nothing is over.
    over = max(np.abs(np.diff(bc.commanded_keyframes(t, v, bpm), axis=0)).max() for t, v in COMBOS)
    assert over <= bc.beat_budget(bpm) * (1.039 if bpm > bc.BPM_FAST else 1.0) + 1e-9, (bpm, over)
    # The safety net is still a net. At a gain that does bind, the bound joint's commanded trajectory
    # sits within 2 % of the budget and one scale step more would break it -- v3's assertion, kept
    # alive at the one place it still fires.
    big = bc.gain_vector({"base_yaw": 3.5})
    bound = 0
    for tier, variant in COMBOS:
        raw = bc.raw_trajectory(tier, bpm, variant)
        s = bc.joint_scales(raw, bpm, big)
        pre = bc.peak_speed(bc.apply_gain(bc.scaled(raw, s), big))
        for j in range(5):
            if s[j] < 1.0:
                bound += 1
                assert budget * 0.98 <= pre[j] <= budget + 1e-9, (tier, variant, bpm, j, pre[j])
                s2 = s.copy()
                s2[j] += 2 * bc.SCALE_STEP
                assert bc.peak_speed(bc.apply_gain(bc.scaled(raw, s2), big))[j] > budget
    assert bound >= 6, (bpm, bound)         # measured: 7 clips bind at 80 bpm, 8 at 132 and at 180


def test_variants_differ_in_choreography():
    for tier in bc.TIERS:
        cmds = {v: bc.commanded(tier, 128, v) for v in bc.VARIANTS}
        for a, b in itertools.combinations(bc.VARIANTS, 2):
            size = max(rms(cmds[a] - bc.START), rms(cmds[b] - bc.START))
            diff = rms(cmds[a] - cmds[b]) / size
            # 0.15, not 0.20: build a and c share one crouch by design and differ by the looks and the
            # nods on the way down; with the crouch now reaching the table's margin that shared part is
            # bigger and the pair measures 0.199 (2026-09-20). Every other pair is well above 0.3.
            assert diff > 0.15, (tier, a, b, diff)


def test_commanded_clamps_after_the_gain(monkeypatch):
    # a ceiling past the box: commanded 120 -> raw 120/1.8 -> x1.8 again -> clamped to the +-98 box
    # AFTER the gain, not before
    monkeypatch.setattr(bc, "YAW_MAX", 120.0)
    raw = bc.raw_trajectory("hype", 80, "a")
    ones = np.ones(5)
    U = bc.clamp_envelope(bc.apply_gain(bc.scaled(raw, ones)))
    assert np.abs(U[:, bc.YAW]).max() == pytest.approx(bc.JOINT_MAX)
    assert not bc.envelope_violations(U)
    assert bc.envelope_violations(bc.apply_gain(raw))
    # and the base_pitch floor is applied to the gained pose too
    K = bc.keyframes("hype", "a")
    K[2, bc.EL] = -22.0                                       # fold without the elbow: -58 commanded is a flip
    V = bc.clamp_envelope(bc.apply_gain(bc.trajectory(K, 80)))
    assert V[:, bc.BP].min() >= -52.0 - 1e-9 or (V[V[:, bc.BP] < -52][:, bc.EL] <= -45).all()


def test_csv_text_format():
    U = np.array([bc.START, bc.START + 1.0])
    lines = bc.csv_text(U).splitlines()
    assert lines[0] == "timestamp,base_yaw.pos,base_pitch.pos,elbow_pitch.pos,wrist_roll.pos,wrist_pitch.pos"
    assert lines[1] == "1000.000000,0.0000,-49.0000,-22.0000,0.0000,30.0000"
    assert lines[2].startswith("1000.033333,1.0000,-48.0000")


def test_write_atomic_leaves_no_temp_file(tmp_path):
    p = tmp_path / "x.csv"
    md5 = bc.write_atomic(p, "hello\n")
    assert p.read_text() == "hello\n" and md5 == hashlib.md5(b"hello\n").hexdigest()
    assert [f.name for f in tmp_path.iterdir()] == ["x.csv"]


def test_names():
    assert bc.clip_name("groove", 128) == "beat_groove_128_a"
    assert bc.clip_name("build", 96, "c") == "beat_build_96_c"
    assert bc.base_name("hype", 128) == "beat_hype_128"


# ----------------------------------------------------------------------------- end to end
@pytest.mark.parametrize("inertial", [
    '<origin xyz="0 0 0"/>',
    '<mass value="1"/>',
    '<mass/><origin xyz="0 0 0"/>',
    '<mass value="1"/><origin/>',
])
def test_validator_rejects_inertial_xml_missing_required_fields(monkeypatch, inertial):
    root = bc.ET.fromstring(f"<robot><link><inertial>{inertial}</inertial></link></robot>")
    monkeypatch.setattr(bc, "LampModel", lambda *args, **kwargs: object())
    monkeypatch.setattr(bc.ET, "parse", lambda path: bc.ET.ElementTree(root))
    with pytest.raises(ValueError, match="inertial requires mass.*origin"):
        bc.Validator(Path("/synthetic-robot"))


def test_validator_still_skips_links_without_inertial_data(monkeypatch):
    root = bc.ET.fromstring('<robot><link/><link><inertial>'
                            '<mass value="2"/><origin xyz="0.1 0.2 0.3"/>'
                            '</inertial></link></robot>')
    monkeypatch.setattr(bc, "LampModel", lambda *args, **kwargs: object())
    monkeypatch.setattr(bc.ET, "parse", lambda path: bc.ET.ElementTree(root))
    validator = bc.Validator(Path("/synthetic-robot"))
    assert len(validator.inertials) == 1
    mass, origin = validator.inertials[0]
    assert mass == pytest.approx(2.0)
    assert np.allclose(origin, [0.1, 0.2, 0.3])


@pytest.mark.skipif(not HAVE_ROBOT, reason=f"vendor robot description not available at {ROBOTDESC} "
                    "(restricted material, read in place from the scratchpad; set LAMP_ROBOTDESC)")
def test_generate_128_all_tiers_and_variants(tmp_path):
    entries, failures = bc.generate(tmp_path, bc.TIERS, [128], ROBOTDESC, log=lambda *_: None)
    assert failures == []
    real = [e for e in entries if not e.get("alias_of")]
    alias = [e for e in entries if e.get("alias_of")]
    assert sorted(e["name"] for e in real) == sorted(f"beat_{t}_128_{v}" for t, v in COMBOS)
    assert sorted(e["name"] for e in alias) == sorted(f"beat_{t}_128" for t in bc.TIERS)
    assert all(e["alias_of"] == f"{e['name']}_a" for e in alias)
    validator = bc.Validator(ROBOTDESC)
    expected_frames = round(bc.clip_seconds(128) * bc.FPS)
    for e in entries:
        path = tmp_path / f"{e['name']}.csv"
        assert hashlib.md5(path.read_bytes()).hexdigest() == e["md5"]
        rows = list(csv.DictReader(open(path)))
        assert list(rows[0].keys()) == bc.CSV_HEADER.split(",")
        assert len(rows) == e["frames"] == expected_frames
        U = np.array([[float(r[f"{j}.pos"]) for j in bc.JOINTS] for r in rows])
        assert np.allclose(U[0], bc.START) and np.allclose(U[-1], bc.START)
        v = validator.validate(U)
        assert v["ok"], v["reasons"]
        assert v["head_y_min"] >= bc.HEAD_Y_MIN and bc.ZMP_Y_MIN <= v["zmp_y_min"]
        assert v["zmp_y_max"] <= bc.ZMP_Y_MAX
        assert v["peak_speed"].max() <= bc.SPEED_LIMIT
        assert not bc.envelope_violations(U)
        assert all(not validator.model.problems(dict(zip(bc.JOINTS, u))) for u in U[::7])
    # the alias file IS variant a
    for e in alias:
        a = next(x for x in real if x["name"] == e["alias_of"])
        assert e["md5"] == a["md5"] and (tmp_path / f"{e['name']}.csv").read_bytes() == (tmp_path / f"{a['name']}.csv").read_bytes()
    manifest = json.loads(bc.write_manifest(tmp_path, entries).read_text())
    assert manifest["version"] == 2 and manifest["fps"] == 30 and manifest["hold_s"] == 0.6
    assert manifest["variants"] == ["a", "b", "c"] and manifest["tiers"] == ["groove", "hype", "drop", "build"]
    assert manifest["gain"] == bc.GAINS and manifest["speed_limit"] == bc.SPEED_LIMIT == 380.0
    assert len(manifest["clips"]) == 16
    assert manifest["aliases"] == {f"beat_{t}_128": f"beat_{t}_128_a" for t in bc.TIERS}
    c = next(c for c in manifest["clips"] if c["name"] == "beat_groove_128_b")
    assert {"name", "base", "variant", "bpm", "tier", "frames", "md5", "seconds", "first", "last", "range",
            "scale", "amplitude", "accent_beats", "peak_speed", "head_y_min", "zmp_y_min", "zmp_y_max"} <= set(c)
    assert bc.ZMP_Y_MIN <= c["zmp_y_min"] and c["zmp_y_max"] <= bc.ZMP_Y_MAX
    assert c["base"] == "beat_groove_128" and c["variant"] == "b" and c["accent_beats"] == [1, 5]
    assert c["first"] == c["last"] == manifest["start_pose"]
    b = next(c for c in manifest["clips"] if c["name"] == "beat_build_128_a")
    # the crouch IS the bottom of the lift: since 2026-09-20 that reaches down toward the table
    # (base_pitch +10, elbow -44) instead of leaning back on the shoulder floor (-60, -55)
    assert b["crouch"]["base_pitch"] == pytest.approx(bc.BOTTOM[bc.BP], abs=0.5)
    assert b["crouch"]["elbow_pitch"] == pytest.approx(bc.BOTTOM[bc.EL], abs=0.5) and "START" in b["note"]
    assert b["accent_beats"] == []
    assert b["gain"] == manifest["gain"] == bc.GAINS
    assert b["last"] == manifest["start_pose"] == next(c for c in manifest["clips"] if c["name"] == "beat_drop_128_a")["first"]
    # the out dir holds exactly the manifest's clips (plus the manifest)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([f"{e['name']}.csv" for e in entries] + ["MANIFEST.json"])


@pytest.mark.skipif(not HAVE_ROBOT, reason=f"vendor robot description not available at {ROBOTDESC}")
def test_generate_refuses_failing_clip_and_removes_stale_csv(tmp_path, monkeypatch):
    quiet = lambda *_: None  # noqa: E731
    entries, failures = bc.generate(tmp_path, ["groove"], [128], ROBOTDESC, log=quiet, variants=("a",))
    assert failures == [] and (tmp_path / "beat_groove_128_a.csv").exists() and (tmp_path / "beat_groove_128.csv").exists()
    # the validator's speed limit is now impossible: the clip must fail, not be written, and the earlier
    # files of the same names (variant and alias) must be gone so the directory equals the manifest
    monkeypatch.setattr(bc, "SPEED_LIMIT", 10.0)
    entries, failures = bc.generate(tmp_path, ["groove"], [128], ROBOTDESC, log=quiet, variants=("a",))
    assert entries == []
    assert len(failures) == 1 and failures[0].startswith("beat_groove_128_a:") and "peak speed" in failures[0]
    assert [p.name for p in tmp_path.iterdir()] == []
    # a clip that does not start at START is refused by the validator itself
    monkeypatch.undo()                                   # ... and back to the real speed limit
    U = bc.commanded("groove", 128, "a")
    U2 = U.copy()
    U2[0, bc.YAW] = 3.0
    assert any("START" in r for r in bc.Validator(ROBOTDESC).validate(U2)["reasons"])
    assert bc.Validator(ROBOTDESC).validate(U)["ok"]


def test_run_names_and_alias_source():
    assert bc.run_names(["drop"], [128]) == {f"beat_drop_128_{v}" for v in "abc"} | {"beat_drop_128"}
    assert bc.run_names(["drop"], [128], variants=("b",)) == {"beat_drop_128_b"}          # no alias without a
    assert bc.run_names(["drop"], [128], aliases=False) == {f"beat_drop_128_{v}" for v in "abc"}
    assert bc.ALIAS_VARIANT == "a"


def _entry(name, tier, bpm, variant, alias_of=None):
    return {"name": name, "base": f"beat_{tier}_{bpm}", "variant": variant, "tier": tier, "bpm": bpm, "md5": "x",
            **({"alias_of": alias_of} if alias_of else {})}


def test_write_manifest_merges_a_partial_run(tmp_path, capsys):
    full = [_entry(f"beat_{t}_128_{v}", t, 128, v) for t in ("groove", "build") for v in "abc"]
    full += [_entry(f"beat_{t}_128", t, 128, None, alias_of=f"beat_{t}_128_a") for t in ("groove", "build")]
    for e in full:
        (tmp_path / f"{e['name']}.csv").write_text("x")
    m = json.loads(bc.write_manifest(tmp_path, full).read_text())
    assert len(m["clips"]) == 8 and m["tiers"] == ["groove", "build"]
    # a partial run of drop 128, variant b only: its one entry joins the library, nothing else moves
    (tmp_path / "beat_drop_128_b.csv").write_text("x")
    m = json.loads(bc.write_manifest(tmp_path, [_entry("beat_drop_128_b", "drop", 128, "b")],
                                     replace=bc.run_names(["drop"], [128], variants=("b",))).read_text())
    names = {c["name"] for c in m["clips"]}
    assert names == {e["name"] for e in full} | {"beat_drop_128_b"}
    assert m["tiers"] == ["groove", "drop", "build"] and m["aliases"] == {"beat_groove_128": "beat_groove_128_a",
                                                                          "beat_build_128": "beat_build_128_a"}
    assert "1 entries replaced, 8 kept" in capsys.readouterr().out
    # a partial run whose variant a failed: the alias name is in `replace`, so the old alias entry goes too;
    # a kept entry whose CSV vanished is dropped; a different gain is called out
    (tmp_path / "beat_build_128_c.csv").unlink()
    m = json.loads(bc.write_manifest(tmp_path, [_entry("beat_groove_128_b", "groove", 128, "b")],
                                     gains={"base_yaw": 1.6}, replace=bc.run_names(["groove"], [128])).read_text())
    names = {c["name"] for c in m["clips"]}
    assert names == {"beat_groove_128_b", "beat_build_128_a", "beat_build_128_b", "beat_build_128", "beat_drop_128_b"}
    assert m["aliases"] == {"beat_build_128": "beat_build_128_a"} and m["gain"]["base_yaw"] == 1.6
    assert next(c for c in m["clips"] if c["name"] == "beat_build_128_a")["gain"]["base_yaw"] == 1.8
    out = capsys.readouterr().out
    assert "1 dropped (CSV missing)" in out and "generated with gains" in out
    # a full run rebuilds from scratch (no replace): the stale entries are gone
    m = json.loads(bc.write_manifest(tmp_path, [_entry("beat_hype_128_a", "hype", 128, "a")]).read_text())
    assert [c["name"] for c in m["clips"]] == ["beat_hype_128_a"]


@pytest.mark.skipif(not HAVE_ROBOT, reason=f"vendor robot description not available at {ROBOTDESC}")
def test_partial_run_keeps_the_library_and_leaves_the_alias_alone(tmp_path):
    quiet = lambda *_: None  # noqa: E731
    entries, failures = bc.generate(tmp_path, ["groove", "build"], [128], ROBOTDESC, log=quiet)
    assert failures == []
    bc.write_manifest(tmp_path, entries, log=quiet)
    alias_md5 = hashlib.md5((tmp_path / "beat_groove_128.csv").read_bytes()).hexdigest()
    # --tier groove --variant b: variant a is not part of the run, so the alias is neither rewritten nor removed
    entries, failures = bc.generate(tmp_path, ["groove"], [128], ROBOTDESC, log=quiet, variants=("b",))
    assert failures == [] and [e["name"] for e in entries] == ["beat_groove_128_b"]
    assert hashlib.md5((tmp_path / "beat_groove_128.csv").read_bytes()).hexdigest() == alias_md5
    m = json.loads(bc.write_manifest(tmp_path, entries, replace=bc.run_names(["groove"], [128], ("b",)), log=quiet).read_text())
    assert len(m["clips"]) == 8 and "build" in m["tiers"] and m["aliases"]["beat_groove_128"] == "beat_groove_128_a"
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([f"{c['name']}.csv" for c in m["clips"]] + ["MANIFEST.json"])
    # main() with a subset merges; with the full library it rebuilds
    assert bc.main(["--out", str(tmp_path), "--tier", "drop", "--bpm", "128", "--robotdesc", str(ROBOTDESC)]) == 0
    m = json.loads((tmp_path / "MANIFEST.json").read_text())
    assert len(m["clips"]) == 12 and m["tiers"] == ["groove", "drop", "build"]
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([f"{c['name']}.csv" for c in m["clips"]] + ["MANIFEST.json"])


# ----------------------------------------------------------------------------- live path: bold, make_clip, atomic writes
def test_bold_multiplier_maps_the_slider():
    """BOLD_MIN was 0.35 until 2026-09-20 and is 0.10 now. The operator's complaint was that the
    "Bolder moves" slider's whole bottom half did nothing he could see: the map is
    BOLD_MIN + (1 - BOLD_MIN) * bold, so at slider 0.05 the arm still travelled 0.35 + 0.65 * 0.05 =
    38 % of full and slider 0 was a third of the library rather than a visibly small dance. At 0.10
    the bottom of the slider is a tenth of the library and the top is still exactly the library.

    Measured on the arm after the change (hype a, 128 bpm, elbow peak-to-peak in commanded units):
    13.8 at slider 0.00, 44.9 at 0.25, 75.9 at 0.50, 107.0 at 0.75, 138.0 at 1.00 -- a straight line
    through a visibly small dance, which is what the slider was asked for. Those five numbers are
    checked below, so the end points alone cannot drift the middle of the slider."""
    assert bc.BOLD_MIN == 0.10                               # pinned as a value: moving it is a deliberate edit
    assert bc.bold_multiplier(1.0) == 1.0                    # exactly: the library is the bold-1.0 case
    assert bc.bold_multiplier(0.0) == pytest.approx(0.10)    # was 0.35
    assert bc.bold_multiplier(0.6) == pytest.approx(0.64)    # was 0.74 (0.35 + 0.65 * 0.6)
    assert bc.bold_multiplier(2.0) == 1.0 and bc.bold_multiplier(-1.0) == pytest.approx(0.10)
    assert bc.bold_multiplier(float("nan")) == pytest.approx(0.10)
    assert bc.bold_multiplier(0.05) == pytest.approx(0.145)  # the operator's complaint: was 0.3825
    raw = bc.raw_trajectory("hype", 128, "a")
    assert bc.bold_trajectory("hype", 128, "a", 1.0) is not None
    assert np.array_equal(bc.bold_trajectory("hype", 128, "a", 1.0), raw)          # untouched, not a float round trip
    half = bc.bold_trajectory("hype", 128, "a", 0.0)
    assert np.allclose(half - bc.START, 0.10 * (raw - bc.START))                   # was 0.35 * (raw - START)
    # and the same straight line all the way out to the commanded clip, which is what the operator sees
    travel = [float(np.ptp(np.array(bc.make_clip("hype", "a", 128, b)[0])[:, bc.EL])) for b in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert travel == pytest.approx([13.8, 44.85, 75.9, 106.95, 138.0], abs=0.05), travel


def test_bold_is_applied_before_the_speed_budget():
    # a speed-bound joint: bold 1.0 is scaled by the budget; bold 0.0 is a third of the RAW pattern and,
    # if that fits the budget, gets scale 1.0 -- so its commanded excursion is min(m * raw, budget), never more
    bpm = 127.3
    raw = bc.raw_trajectory("groove", bpm, "a")
    assert (bc.joint_scales(bc.bold_trajectory("groove", bpm, "a", 0.0), bpm)
            >= bc.joint_scales(raw, bpm) - 1e-12).all()
    # a joint the budget actually binds: at the library's gains nothing is speed-bound any more (the
    # yaw ceiling and the lift envelope bind first), so this is read at a gain that does bind
    big = bc.gain_vector({"base_yaw": 3.5})
    s1 = bc.joint_scales(bc.bold_trajectory("hype", bpm, "a", 1.0), bpm, big)
    s0 = bc.joint_scales(bc.bold_trajectory("hype", bpm, "a", 0.0), bpm, big)
    assert s1[bc.YAW] < 1.0 and s0[bc.YAW] > s1[bc.YAW]
    assert (s0 >= s1 - 1e-12).all()
    for bold in (0.0, 0.6, 1.0):
        rows, meta = bc.make_clip("groove", "a", bpm, bold)
        U = np.array(rows)
        # this tempo's own budget, not the 380 ceiling: measured 2026-09-19 the worst clip over every
        # tier, variant, tempo and bold reaches 0.992 of speed_for(bpm) * SPEED_MARGIN
        assert bc.peak_speed(U).max() <= bc.speed_for(bpm) * bc.SPEED_MARGIN + 1e-9
        assert not bc.envelope_violations(U)
        assert np.allclose(U[0], bc.START) and np.allclose(U[-1], bc.START)
        assert meta["ok"] is None and meta["frames"] == len(rows) == round(bc.clip_seconds(bpm) * bc.FPS)
        assert meta["multiplier"] == pytest.approx(bc.bold_multiplier(bold))


def _excursion(rows):
    return np.abs(np.array(rows) - bc.START).max(axis=0)


@pytest.mark.skipif(not HAVE_ROBOT, reason=f"vendor robot description not available at {ROBOTDESC}")
def test_make_clip_at_exact_bpm_passes_validation_and_grows_with_bold():
    v = bc.Validator(ROBOTDESC)
    bpm = 127.3
    for tier, variant in COMBOS:
        prev = None
        for bold in (0.0, 0.6, 1.0):
            rows, meta = bc.make_clip(tier, variant, bpm, bold, model=v)
            assert meta["ok"] is True and meta["reasons"] == [], (tier, variant, bold, meta["reasons"])
            assert isinstance(rows, list) and len(rows[0]) == 5 and isinstance(rows[0], tuple)
            ok, report = bc.validate_rows(rows, v)
            assert ok and report["ok"] and report["frames"] == len(rows)
            assert report["peak_speed"]["base_yaw"] <= bc.SPEED_LIMIT
            exc = _excursion(rows)
            if prev is not None:
                # every joint's excursion is monotone in bold; a speed-bound joint sits at the budget for
                # every bold (its scale is quantised in 0.005 steps, hence the 1 % tolerance)
                assert (exc >= prev * 0.99 - 1e-6).all(), (tier, variant, bold, prev, exc)
                assert exc.sum() > prev.sum()
            prev = exc
        # bold 0 is 10 % of bold 1 on a joint that neither the speed budget nor the envelope clamp binds
        # (35 % until 2026-09-20: BOLD_MIN moved because the slider's bottom half did nothing visible).
        # The clamp runs after the gain and may trim either clip (the build's shallow crouch at bold 0
        # meets the flip-region floor before its elbow is lifted), so it is checked, not assumed.
        def unclamped(bold):
            rows, meta = bc.make_clip(tier, variant, bpm, bold)
            raw = bc.bold_trajectory(tier, bpm, variant, bold)
            pre = bc.apply_gain(bc.scaled(raw, [meta["scale"][j] for j in bc.JOINTS]))
            return np.array(rows), meta, np.isclose(pre, np.array(rows)).all(axis=0)
        U0, m0, free0 = unclamped(0.0)
        U1, m1, free1 = unclamped(1.0)
        lo, hi = _excursion(U0), _excursion(U1)
        checked = 0
        for j in range(5):
            if free0[j] and free1[j] and m1["scale"][bc.JOINTS[j]] == 1.0 and hi[j] > 1.0:
                # 0.10, not 0.35: BOLD_MIN moved on 2026-09-20 (the slider's bottom half did nothing
                # the operator could see -- at slider 0.05 the arm still travelled 38 % of full).
                # This is the same ratio as bold_multiplier(0.0), read out at the end of the pipeline.
                assert lo[j] / hi[j] == pytest.approx(bc.bold_multiplier(0.0), abs=0.01), (tier, variant, bc.JOINTS[j])
                assert lo[j] / hi[j] == pytest.approx(0.10, abs=0.01), (tier, variant, bc.JOINTS[j])
                checked += 1
        # v3 could only ever prove this on one joint, because the speed budget bound the rest. v4 trims
        # nothing at this tempo, so the slider's BOLD_MIN is exact on nearly every moving joint: measured
        # 2026-09-19, 5 joints for groove, 4 for hype/drop and build b/c, 3 for build a (whose wrist
        # roll never leaves START). A drop back to 1 would mean the budget had started binding again.
        assert checked >= 3, (tier, variant, checked)
    # a bad row is caught, and lists of tuples are accepted like arrays
    rows, _ = bc.make_clip("groove", "a", bpm, 0.6)
    bad = list(rows); bad[40] = (0.0, -70.0, -22.0, 0.0, 30.0)
    ok, report = bc.validate_rows(bad, v)
    assert not ok and any("base_pitch" in r for r in report["reasons"])
    assert bc.validate_rows([rows[0]], v) == (False, {"ok": False, "reasons": ["rows must be (n >= 2, 5), got (1, 5)"]})
    with pytest.raises(ValueError):
        bc.make_clip("groove", "a", 20.0, 0.5)


def test_the_speed_dial_at_one_is_the_clip_that_has_no_dial():
    """`speed` is threaded as a keyword with default 1.0 through speed_for, max_move, beat_budget,
    joint_scales, lift_profile, yaw_layer, commanded_keyframes, keyframes, raw_trajectory,
    bold_trajectory, commanded, build_clip and make_clip. The whole argument for adding it that way
    is that dial 1.0 changes NOTHING: the library, the manifest md5s and every live clip built
    before the dial existed have to come out bit for bit the same, or the dial is a rewrite of the
    choreography wearing a default value. Checked as array equality and as CSV md5, not allclose."""
    for tier, variant in COMBOS:
        for bpm in (80, 127.3, 128, 132, 180):
            # keyframes() and commanded_keyframes() take no `speed`: the dial stopped being a per-joint
            # limit inside the choreography on 2026-09-20 (that changed the POSE, not the pace, and the
            # Validator refused the result) and became one uniform factor applied after it, in
            # speed_multiplier(). The choreography a tempo produces is the same whatever the dial says.
            assert "speed" not in inspect.signature(bc.keyframes).parameters
            assert "speed" not in inspect.signature(bc.commanded_keyframes).parameters
            assert "speed" not in inspect.signature(bc.raw_trajectory).parameters
            for bold in (0.0, 0.6, 1.0):
                assert np.array_equal(bc.bold_trajectory(tier, bpm, variant, bold),
                                      bc.bold_trajectory(tier, bpm, variant, bold, speed=1.0))
                assert np.array_equal(bc.commanded(tier, bpm, variant, bold=bold),
                                      bc.commanded(tier, bpm, variant, bold=bold, speed=1.0)), (tier, variant, bpm, bold)
                s0, U0 = bc.build_clip(tier, bpm, variant, bold=bold)
                s1, U1 = bc.build_clip(tier, bpm, variant, bold=bold, speed=1.0)
                assert np.array_equal(s0, s1) and np.array_equal(U0, U1), (tier, variant, bpm, bold)
                r0, m0 = bc.make_clip(tier, variant, bpm, bold)
                r1, m1 = bc.make_clip(tier, variant, bpm, bold, speed=1.0)
                assert hashlib.md5(bc.csv_text(r0).encode()).hexdigest() \
                       == hashlib.md5(bc.csv_text(r1).encode()).hexdigest(), (tier, variant, bpm, bold)
                assert m0["scale"] == m1["scale"] and m0["amplitude"] == m1["amplitude"]
    # ... and the turn is orthogonal to the dial, so a turned clip at dial 1.0 is the turned clip too
    for facing in (0.0, +20.0, -20.0):
        r0, _ = bc.make_clip("groove", "a", 128, 1.0, facing=facing)
        r1, _ = bc.make_clip("groove", "a", 128, 1.0, facing=facing, speed=1.0)
        assert bc.csv_text(r0) == bc.csv_text(r1), facing


def test_a_lower_speed_dial_only_ever_slows_the_arm():
    """What the dial is FOR: the operator says "calm down" and every joint slows. It must never buy
    speed back, at any tempo, in any tier, at any point of its travel -- the dial is read live off
    the dashboard while the lamp is dancing, so a non-monotone patch in it would be an arm that
    speeds up when the operator turns it down. Checked per joint, on the commanded trajectory the
    runtime will actually play, over the dial in 0.1 steps. Measured 2026-09-20: dial 0.0 halves the
    peak (hype a at 180 bpm goes 335.6 -> 148.2 units/s, groove b at 128 goes 297.3 -> 148.0)."""
    for bpm in (80, 127.3, 132, 180):
        for tier, variant in COMBOS:
            prev_speed = prev_travel = None
            for dial in np.arange(0.0, 1.0001, 0.1):
                U = bc.commanded(tier, bpm, variant, speed=float(dial))
                sp, travel = bc.peak_speed(U), np.abs(U - bc.START).max(axis=0)
                # the dial's own budget binds, and the hard ceiling holds whatever the dial is at
                assert sp.max() <= bc.speed_for(bpm, float(dial)) * bc.SPEED_MARGIN + 1e-9, (tier, variant, bpm, dial, sp)
                assert sp.max() <= bc.SPEED_LIMIT, (tier, variant, bpm, dial, sp)
                assert np.allclose(U[0], bc.START) and np.allclose(U[-1], bc.START)
                assert not bc.envelope_violations(U), (tier, variant, bpm, dial)
                if prev_speed is not None:
                    # per JOINT, not just the maximum: turning the dial up may not slow anything down
                    # and turning it down may not speed anything up (measured: exactly monotone, so
                    # the tolerance here is float noise and not a measured wobble)
                    assert (sp >= prev_speed - 1e-9).all(), (tier, variant, bpm, dial, prev_speed, sp)
                    assert (travel >= prev_travel - 1e-9).all(), (tier, variant, bpm, dial, prev_travel, travel)
                prev_speed, prev_travel = sp, travel
    # and it bites: the calm end is about half the arm speed of the tempo's own budget
    for bpm, tier, variant in ((128, "hype", "a"), (128, "groove", "b"), (180, "hype", "a"), (180, "groove", "b")):
        calm = bc.peak_speed(bc.commanded(tier, bpm, variant, speed=0.0)).max()
        loud = bc.peak_speed(bc.commanded(tier, bpm, variant)).max()
        assert calm == pytest.approx(bc.SPEED_CALM * bc.SPEED_MARGIN, rel=0.02), (tier, variant, bpm, calm)
        assert calm < 0.60 * loud, (tier, variant, bpm, calm, loud)


@pytest.mark.skipif(not HAVE_ROBOT, reason=f"vendor robot description not available at {ROBOTDESC}")
def test_a_clip_built_at_a_low_speed_dial_still_passes_the_validator():
    """RED ON PURPOSE -- this is a genuine bug in the speed dial, not a stale expectation.

    Every clip the live path ships has to pass Validator.validate, whatever the dashboard's dials
    are set to; the bold slider manages it over its whole travel (swept 2026-09-20 in 0.1 steps over
    all twelve tier/variant pairs at 80, 127.3, 132, 160 and 180 bpm: zero refusals). The new speed
    dial does not. There is a hole at 180 bpm on hype c:

        dial 0.35  zmp_y min -0.031     dial 0.45  zmp_y min -0.034
        dial 0.40  zmp_y min -0.034     dial 0.50  zmp_y min -0.031

    against ZMP_Y_MIN -0.030 -- the BACKWARD tipping bound. Dial 0.30 and below passes, dial 0.55
    and above passes, so it is a band in the middle of the operator's travel and not an end stop.

    WHY (measured): the dial only reaches the choreography through joint_scales, which picks a
    SEPARATE scale per joint. At dial 0.45 hype c at 180 gets base_yaw 0.74, base_pitch 0.715,
    elbow_pitch 1.00 -- the elbow dances its full range while the shoulder is held to 72 % of its
    own, which is not a slower dance but a DIFFERENT pose: the arm folds up and back over the base
    instead of reaching down and out, and the CoM goes behind the footprint. `speed` is in fact
    already passed to lift_profile and yaw_layer, which would re-plan the phrase coherently at the
    lower budget -- but both of them call beat_budget(bpm) without it, so it is accepted and
    dropped. (Wiring it through is NOT the fix on its own: with both of them honouring the dial the
    sweep above goes from 4 refusals to 30, hype a dropping the head below HEAD_Y_MIN at 127.3 and
    132 bpm. The dial needs a design decision about what "slower" should do to the choreography,
    which is why this is reported rather than patched from here.)

    NOT A HAZARD TO THE RUNNING DEMO: make_clip reports ok False with the reason, and LiveClips._run
    in lamp_show.py counts the failure and falls back to the library clip, so nothing out of bounds
    reaches the arm. The visible symptom is that the speed dial silently stops producing live clips
    in that band at fast tempi. The dial also defaults to 1.0, which is outside the hole.

    This assertion stays as it is until the generator is fixed: it is the property the live path
    needs, and the numbers in it were measured, not guessed."""
    v = bc.Validator(ROBOTDESC)
    refused = []
    for bpm in (80, 127.3, 132, 180):
        for dial in SPEED_DIALS:
            for tier, variant in COMBOS:
                rows, meta = bc.make_clip(tier, variant, bpm, 1.0, model=v, speed=dial)
                # whatever the verdict, the clip must never have been ALLOWED past the hard ceiling
                assert bc.peak_speed(np.array(rows)).max() <= bc.SPEED_LIMIT, (tier, variant, bpm, dial)
                assert meta["ok"] is not None and meta["frames"] == len(rows)
                if meta["ok"] is not True:
                    refused.append((bpm, round(float(dial), 2), tier, variant, meta["reasons"]))
    assert refused == [], f"the speed dial builds clips the validator refuses: {refused}"


@pytest.mark.skipif(not HAVE_ROBOT, reason=f"vendor robot description not available at {ROBOTDESC}")
def test_make_clip_bold_1_at_a_bucket_is_the_batch_file_byte_for_byte(tmp_path):
    quiet = lambda *_: None  # noqa: E731
    entries, failures = bc.generate(tmp_path, ["groove", "drop"], [128], ROBOTDESC, log=quiet, variants=("b",), aliases=False)
    assert failures == []
    for e in entries:
        rows, meta = bc.make_clip(e["tier"], e["variant"], 128.0, 1.0, model=bc.Validator(ROBOTDESC))
        text = bc.csv_text(rows)
        assert (tmp_path / f"{e['name']}.csv").read_bytes() == text.encode()
        assert hashlib.md5(text.encode()).hexdigest() == e["md5"] and meta["ok"] and meta["scale"] == e["scale"]
    # ... and the library is unchanged by the refactor: every clip the OLD generator wrote (its manifest
    # carries the md5) comes out the same from the shared path, with or without going through make_clip
    manifest = bc.DEFAULT_OUT / "MANIFEST.json"
    if not manifest.exists():
        pytest.skip(f"no generated library at {bc.DEFAULT_OUT}")
    m = json.loads(manifest.read_text())
    # every real clip: the library at DEFAULT_OUT is generated by this generator, so any silent
    # re-choreography shows up here as an md5 mismatch. A DELIBERATE one means regenerating it.
    real = [c for c in m["clips"] if not c.get("alias_of")]
    assert len(real) >= 12
    assert any(c["tier"] == "hype" and c["variant"] in ("b", "c") for c in real)
    for c in real:
        _, U = bc.build_clip(c["tier"], c["bpm"], c["variant"])
        assert hashlib.md5(bc.csv_text(U).encode()).hexdigest() == c["md5"], c["name"]
    for c in real[::37]:
        rows, _ = bc.make_clip(c["tier"], c["variant"], float(c["bpm"]), 1.0)
        assert hashlib.md5(bc.csv_text(rows).encode()).hexdigest() == c["md5"], c["name"]


@pytest.mark.skipif(not HAVE_ROBOT, reason=f"vendor robot description not available at {ROBOTDESC}")
def test_validator_for_lamp_uses_the_given_checkout_and_calibration():
    v = bc.Validator.for_lamp(ROBOTDESC / "pi5_feetech_r1", ROBOTDESC / "lelamp-calibration.json")
    assert v.model.robot_dir == ROBOTDESC / "pi5_feetech_r1" and "lelamp-calibration.json" in v.model.scale_source
    rows, _ = bc.make_clip("build", "c", 127.3, 1.0)
    assert bc.validate_rows(rows, v)[0] and bc.validate_rows(rows, v.model)[0]     # a LampModel works too
    assert bc.validate_rows(rows, v)[1] == bc.validate_rows(rows, bc.Validator(ROBOTDESC))[1]
    assert bc.LAMP_CALIBRATION == Path("/var/lib/lelamp/user-data/v1/calibration/lelamp.json")


def test_write_clip_atomic_leaves_nothing_behind_on_failure(tmp_path, monkeypatch):
    rows = [(0.0, -49.0, -22.0, 0.0, 30.0)] * 40
    path = tmp_path / "live_2.csv"
    md5 = bc.write_clip_atomic(path, rows)
    assert path.read_text() == bc.csv_text(rows) and md5 == hashlib.md5(bc.csv_text(rows).encode()).hexdigest()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["live_2.csv"]                # no temp left
    before = path.read_bytes()
    # the fsync fails (disk gone): no temp file, the old file untouched
    def boom(*a, **k): raise OSError("simulated")
    monkeypatch.setattr(bc.os, "fsync", boom)
    with pytest.raises(OSError):
        bc.write_clip_atomic(path, rows[:10])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["live_2.csv"] and path.read_bytes() == before
    monkeypatch.undo()
    # the rename fails: same
    monkeypatch.setattr(bc.os, "replace", boom)
    with pytest.raises(OSError):
        bc.write_clip_atomic(tmp_path / "live_3.csv", rows)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["live_2.csv"]
    monkeypatch.undo()
    # the re-read does not match what was written: the file is removed, not left for the runtime
    monkeypatch.setattr(bc.Path, "read_bytes", lambda self: b"corrupt")
    with pytest.raises(IOError):
        bc.write_clip_atomic(tmp_path / "live_4.csv", rows)
    monkeypatch.undo()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["live_2.csv"]
    # the temp name starts with '.' and lives in the same dir
    seen = []
    real_replace = bc.os.replace
    monkeypatch.setattr(bc.os, "replace", lambda a, b: seen.append((Path(a), Path(b))) or real_replace(a, b))
    bc.write_clip_atomic(tmp_path / "live_5.csv", rows)
    (tmp, dst), = seen
    assert tmp.parent == tmp_path and tmp.name.startswith(".live_5.csv.") and tmp.name.endswith(".tmp") and dst == tmp_path / "live_5.csv"


@pytest.mark.skipif(not HAVE_ROBOT, reason=f"vendor robot description not available at {ROBOTDESC}")
def test_one_cli_writes_a_validated_clip_at_the_exact_bpm(tmp_path, capsys):
    out = tmp_path / "one"
    assert bc.main(["--one", "groove", "a", "127.3", "0.8", "--out", str(out), "--robotdesc", str(ROBOTDESC)]) == 0
    files = sorted(p.name for p in out.iterdir())
    assert files == ["live_groove_a_127p3_0p80.csv"]
    lines = (out / files[0]).read_text().splitlines()
    assert lines[0] == bc.CSV_HEADER and len(lines) - 1 == round(bc.clip_seconds(127.3) * bc.FPS)
    rows, _ = bc.make_clip("groove", "a", 127.3, 0.8)
    assert (out / files[0]).read_text() == bc.csv_text(rows)
    assert "PASS" in capsys.readouterr().out
    assert bc.main(["--one", "waltz", "a", "128", "0.5", "--out", str(out), "--robotdesc", str(ROBOTDESC)]) == 2
    assert bc.one_name("hype", "c", 128.0, 1.0) == "live_hype_c_128_1p00"


# ----------------------------------------------------------------------------- live path: facing
FACING_OK = 20.0        # Measured 2026-09-19 against this calibration, in 1-unit steps over 80, 128,
                        # 160 and 180 bpm. Two different things stop a bigger turn. The JOINT BOX is
                        # what binds the loud tiers: hype and drop swing to YAW_MAX 68, so the +-98 box
                        # leaves them exactly +-30 (+-33.9 at 180, where the tempo budget trims the
                        # swing), while groove keeps +-53 and build +-77 or more. The MODEL is what
                        # binds at fast tempi: the smallest turn refused anywhere in the demo band is
                        # +-25 on hype c and drop c at 180 bpm, and the reason is ALWAYS "zmp_y max"
                        # -- v4 leans out over the table (REACH) to buy sideways travel, so a turn now
                        # spends forward tipping margin, where v3's turns spent head clearance. Nothing
                        # at 80 or 128 bpm is refused at any turn the box will give. 20 leaves 5 units
                        # of margin on the earliest refusal, which is what makes it the safe constant.


def _yaw_room(tier, bpm, variant, bold=1.0):
    """(lowest, highest) facing the joint box leaves this clip, computed the long way from the commanded
    pre-clamp trajectory -- the same quantity facing_limit derives, arrived at independently."""
    raw = bc.bold_trajectory(tier, bpm, variant, bold)
    U = bc.apply_gain(bc.scaled(raw, bc.joint_scales(raw, bpm)))
    return -bc.JOINT_MAX - U[:, bc.YAW].min(), bc.JOINT_MAX - U[:, bc.YAW].max()


def test_facing_zero_is_todays_clip_byte_for_byte():
    # the default and an explicit 0.0 are the same call, and neither is a float round trip: face_trajectory
    # hands the array straight back, the way bold_trajectory does at bold 1.0
    U = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [-0.0, 2.0, 3.0, 4.0, 5.0]])
    assert bc.face_trajectory(U, 0.0) is U
    assert bc.face_trajectory(U, -0.0) is U
    assert bc.facing_limit(U, 0.0) == 0.0 and bc.facing_limit(U, float("nan")) == 0.0
    for tier, variant in COMBOS:
        for bpm in (80, 127.3, 128, 180):
            s0, U0 = bc.build_clip(tier, bpm, variant)
            sf, Uf = bc.build_clip(tier, bpm, variant, bold=1.0, facing=0.0)
            assert np.array_equal(s0, sf)
            assert bc.csv_text(U0) == bc.csv_text(Uf), (tier, variant, bpm)
            assert np.array_equal(bc.commanded(tier, bpm, variant), bc.commanded(tier, bpm, variant, facing=0.0))
            for bold in (0.0, 0.6, 1.0):
                r0, m0 = bc.make_clip(tier, variant, bpm, bold)
                rf, mf = bc.make_clip(tier, variant, bpm, bold, facing=0.0)
                assert hashlib.md5(bc.csv_text(r0).encode()).hexdigest() \
                       == hashlib.md5(bc.csv_text(rf).encode()).hexdigest(), (tier, variant, bpm, bold)
                assert m0["facing"] == 0.0 and mf["facing"] == 0.0
                assert m0["amplitude"] == mf["amplitude"]


@pytest.mark.parametrize("facing", [+FACING_OK, -FACING_OK, +7.5, -31.0])
def test_facing_shifts_every_frame_of_base_yaw_and_nothing_else(facing):
    for tier, variant in COMBOS:
        for bpm in (80, 128, 180):
            _, U0 = bc.build_clip(tier, bpm, variant)
            _, Uf = bc.build_clip(tier, bpm, variant, facing=facing)
            lo, hi = _yaw_room(tier, bpm, variant)
            applied = min(hi, max(lo, facing))
            assert bc.facing_limit(bc.apply_gain(bc.scaled(bc.raw_trajectory(tier, bpm, variant),
                                                           bc.joint_scales(bc.raw_trajectory(tier, bpm, variant), bpm))),
                                   facing) == pytest.approx(applied)
            # exactly, not approximately: nothing is clamped, so the turn is one addition per frame
            assert np.array_equal(Uf[:, bc.YAW], U0[:, bc.YAW] + applied), (tier, variant, bpm)
            rest = [j for j in range(5) if j != bc.YAW]
            assert np.array_equal(Uf[:, rest], U0[:, rest]), (tier, variant, bpm)
            # the whole phrase turned: the first and last frames sit at the facing, and the ends still match
            assert Uf[0, bc.YAW] == pytest.approx(applied) and Uf[-1, bc.YAW] == pytest.approx(applied)
            assert Uf[0, bc.YAW] == Uf[-1, bc.YAW]
            assert np.allclose(Uf[0, rest], bc.START[rest]) and np.allclose(Uf[-1, rest], bc.START[rest])
            rows, meta = bc.make_clip(tier, variant, bpm, 1.0, facing=facing)
            assert meta["facing"] == pytest.approx(applied)                  # reported, next to bold
            assert meta["bold"] == 1.0
            # the amplitude meta still measures the dance, not the turn
            assert meta["amplitude"] == pytest.approx(round(float(np.abs(U0[:, bc.YAW]).max()), 2), abs=0.02)


def test_facing_is_limited_by_the_joint_box_and_turns_rather_than_flattens():
    for tier, variant in COMBOS:
        for bpm in (80, 128, 180):
            _, U0 = bc.build_clip(tier, bpm, variant)
            lo, hi = _yaw_room(tier, bpm, variant)
            assert lo < 0 < hi                                   # every clip has room to turn either way
            for want, edge in ((500.0, hi), (-500.0, lo)):
                _, Uf = bc.build_clip(tier, bpm, variant, facing=want)
                _, meta = bc.make_clip(tier, variant, bpm, 1.0, facing=want)
                assert meta["facing"] == pytest.approx(edge)     # reduced, and the caller can see it was
                assert abs(meta["facing"]) < abs(want)
                assert np.abs(Uf[:, bc.YAW]).max() <= bc.JOINT_MAX + 1e-9, (tier, variant, bpm, want)
                assert not bc.envelope_violations(Uf)
                # right up against the box, and the choreography is rotated, not squashed against it
                assert np.abs(Uf[:, bc.YAW]).max() == pytest.approx(bc.JOINT_MAX)
                assert np.ptp(Uf[:, bc.YAW]) == pytest.approx(np.ptp(U0[:, bc.YAW]))
    # a clip that already fills the box on both sides has no room at all, so it is left where it is
    full = np.tile(bc.START, (10, 1))
    full[3, bc.YAW], full[6, bc.YAW] = 2 * bc.JOINT_MAX, -2 * bc.JOINT_MAX
    assert bc.facing_limit(full, 30.0) == 0.0


def test_facing_cannot_change_any_joints_peak_speed():
    # A constant on one column leaves every frame-to-frame difference the difference it was, which is why
    # the speed budget (joint_scales) is not redone for a turn. The joints the turn does not touch come
    # out bit-identical; base_yaw's own differences can move by the last bit of the subtraction, since
    # (a+f)-(b+f) need not round to exactly a-b -- far below the budget's 1 % margin, and never upward
    # past the limit, which is the property that matters.
    rest = [j for j in range(5) if j != bc.YAW]
    for tier, variant in COMBOS:
        for bpm in (80, 127.3, 132, 180):
            _, U0 = bc.build_clip(tier, bpm, variant)
            s0 = bc.peak_speed(U0)
            for facing in (+FACING_OK, -FACING_OK, 500.0):
                _, Uf = bc.build_clip(tier, bpm, variant, facing=facing)
                sf = bc.peak_speed(Uf)
                assert np.array_equal(sf[rest], s0[rest]), (tier, variant, bpm, facing)
                assert sf[bc.YAW] == pytest.approx(s0[bc.YAW], rel=1e-12), (tier, variant, bpm, facing)
                assert sf.max() <= bc.SPEED_LIMIT
                assert sf.max() <= s0.max() * (1 + 1e-12)          # a turn never buys speed back


# How far every tier and variant can be turned and still pass the validator, measured 2026-09-20 with
# the bottom of the dance at the table's margin: 60 units at 80 bpm, 26 at 128, 12 at 160, 8 at 170,
# 2 at 180. The forward ZMP is what refuses it (a turn leans the head out over the table), and the
# acceleration term grows with tempo, so the fast end has almost no room. lamp_show's WANDER_UNITS is
# 20, which the library covers up to about 140 bpm; a refused facing falls back to the home-facing
# library clip, which is the designed behaviour, not a fault.
FACING_OK_AT = {80: 20.0, 127.3: 20.0, 132: 20.0, 180: 2.0}


@pytest.mark.skipif(not HAVE_ROBOT, reason=f"vendor robot description not available at {ROBOTDESC}")
@pytest.mark.parametrize("bpm", [80, 127.3, 132, 180])
def test_a_turned_clip_still_passes_the_validator(bpm):
    v = bc.Validator(ROBOTDESC)
    turn = FACING_OK_AT[bpm]
    if bpm == 180:      # and the limit is real: the wander's 20 is refused here, on the forward ZMP
        # ... but only on ONE side, and this asked for the wrong one. A turn is signed and the dance
        # is not symmetric in yaw: groove a's lift is at its lowest and fastest while the phrase is
        # swung one way, so turning that reach round toward the front costs forward ZMP margin and
        # turning it the other way gives some back. Measured 2026-09-20 on this calibration, groove a
        # at 180 bpm: facing +20 passes at +0.084 forward, facing -20 is refused at +0.090 against the
        # +0.085 bound, and the first refusal walking out from 0 is at -4. (The +20 this asked for has
        # never failed on zmp_y max -- with the old ZMP_Y_MIN of -0.012 it passed too, and it was the
        # UNTURNED clip at that tempo that the backward bound refused, which is why ZMP_Y_MIN moved.)
        _, meta = bc.make_clip("groove", "a", bpm, 1.0, model=v, facing=-20.0)
        assert meta["ok"] is False and any("zmp_y max" in r for r in meta["reasons"]), meta["reasons"]
        _, meta = bc.make_clip("groove", "a", bpm, 1.0, model=v, facing=-4.0)
        assert meta["ok"] is False, meta["reasons"]          # the limit really is this tight at 180
        _, meta = bc.make_clip("groove", "a", bpm, 1.0, model=v, facing=+20.0)
        assert meta["ok"] is True, meta["reasons"]           # ... and it is one-sided
    for tier, variant in COMBOS:
        for facing in (0.0, +turn, -turn):
            rows, meta = bc.make_clip(tier, variant, bpm, 1.0, model=v, facing=facing)
            assert meta["ok"] is True, (tier, variant, bpm, facing, meta["reasons"])
            U = np.array(rows)
            assert meta["report"]["head_y_min"] >= bc.HEAD_Y_MIN and meta["report"]["zmp_y_min"] >= bc.ZMP_Y_MIN
            assert U[0, bc.YAW] == pytest.approx(meta["facing"]) and U[-1, bc.YAW] == pytest.approx(meta["facing"])
            # the validator accepts the turned ends, and still refuses ends that point different ways
            bad = U.copy()
            bad[0, bc.YAW] += 3.0
            assert any("START" in r for r in v.validate(bad)["reasons"]), (tier, variant, bpm, facing)


@pytest.mark.skipif(not HAVE_ROBOT, reason=f"vendor robot description not available at {ROBOTDESC}")
def test_a_turn_the_model_refuses_is_reported_not_shipped():
    # the joint box is not the only limit: turning the arm swings its reach round toward the front, and
    # past the measured +-24 units some clips lose a tipping margin. make_clip must say so, so the live
    # path falls back to the library rather than playing it (measured on this calibration).
    v = bc.Validator(ROBOTDESC)
    rows, meta = bc.make_clip("hype", "c", 180, 1.0, model=v, facing=500.0)
    assert meta["facing"] > 24.0 and meta["ok"] is False
    assert any("zmp_y" in r for r in meta["reasons"]), meta["reasons"]
    assert not bc.envelope_violations(np.array(rows))            # the joint box alone would have let it through


@pytest.mark.skipif(not HAVE_ROBOT, reason=f"vendor robot description not available at {ROBOTDESC}")
def test_facing_cli_renders_a_turned_clip_beside_the_home_facing_one(tmp_path, capsys):
    out = tmp_path / "one"
    common = ["--one", "groove", "a", "128", "1.00", "--out", str(out), "--robotdesc", str(ROBOTDESC)]
    assert bc.main(common) == 0
    assert bc.main(common + ["--facing", "20"]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["live_groove_a_128_1p00.csv", "live_groove_a_128_1p00_f20.csv"]
    rows, meta = bc.make_clip("groove", "a", 128.0, 1.0, facing=20.0)
    assert (out / "live_groove_a_128_1p00_f20.csv").read_text() == bc.csv_text(rows)
    assert meta["facing"] == pytest.approx(20.0)
    home = (out / "live_groove_a_128_1p00.csv").read_text()
    assert home == bc.csv_text(bc.make_clip("groove", "a", 128.0, 1.0)[0]) and home != bc.csv_text(rows)
    assert "PASS" in capsys.readouterr().out
    assert bc.one_name("groove", "a", 128.0, 1.0) == "live_groove_a_128_1p00"
    assert bc.one_name("groove", "a", 128.0, 1.0, 20.0) == "live_groove_a_128_1p00_f20"
    assert bc.one_name("groove", "a", 128.0, 1.0, -12.5) == "live_groove_a_128_1p00_f-12p5"
    # a turn the box cannot give is reduced, and the run says so rather than pretending it turned that
    # far. groove a at 128 has +-56.41 of box room (it was +-53 while the tempo budget ran to 380), and
    # the model still allows it out there, so that one passes; hype c at 180 has +-40.61 of room and the
    # model refuses the forward ZMP well before that, so that run fails and writes nothing: the box clamp
    # is a convenience, the validator is the verdict.
    assert bc.main(common + ["--facing", "500"]) == 0
    said = capsys.readouterr().out
    room = bc.facing_limit(bc.build_clip("groove", 128.0, "a")[1], 500.0)   # what the box leaves (was +53, +56.41 now)
    assert f"reduced to +{room:.2f}" in said and "PASS" in said, said
    written = sorted(q.name for q in out.iterdir())
    assert len(written) == 3 and written[2] == bc.one_name("groove", "a", 128.0, 1.0, room) + ".csv"
    hard = ["--one", "hype", "c", "180", "1.00", "--out", str(out), "--robotdesc", str(ROBOTDESC), "--facing", "500"]
    assert bc.main(hard) == 1
    said = capsys.readouterr().out
    # hype c at 180 had +-33 of box room while the budget at that tempo was the vendor's 380; with
    # SPEED_AT_FAST at 340 the phrase's own yaw swing is narrower, so the box leaves MORE room for the
    # turn -- +40.61 measured 2026-09-20. Read off facing_limit for the same reason the groove line
    # above does: the number is a consequence of the tempo budget, and the point of the assertion is
    # that the CLI reports the reduction it made, not that the reduction is any particular size.
    hard_room = bc.facing_limit(bc.build_clip("hype", 180.0, "c")[1], 500.0)   # was +33, +40.61 now
    assert hard_room == pytest.approx(40.61, abs=0.01), hard_room
    assert f"reduced to +{hard_room:.2f}" in said and "FAIL" in said and "zmp_y" in said, said
    assert sorted(q.name for q in out.iterdir()) == written       # the refused turn wrote nothing
    # --facing is a --one option: the batch library is the home-facing one
    assert bc.main(["--out", str(out), "--tier", "groove", "--bpm", "128", "--facing", "20",
                    "--robotdesc", str(ROBOTDESC)]) == 2
