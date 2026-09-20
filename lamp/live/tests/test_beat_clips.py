"""beat_clips v2: the pure pieces (beats, easing, gain table, envelope clamp, per-joint amplitude,
variants, accents), and one end-to-end generation at 128 bpm (needs the vendor robot description,
which is restricted material read in place from the scratchpad and never copied)."""
import csv
import hashlib
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


def test_max_move_is_the_cosine_speed_budget():
    # a one-beat move of D units with cosine easing peaks at (pi/2) * D / (0.7 * P)
    for bpm in (80, 132, 180):
        D = bc.max_move(bpm)
        assert (np.pi / 2) * D / (bc.MOVE_FRAC * 60 / bpm) == pytest.approx(bc.SPEED_DESIGN)
    assert bc.max_move(80) > bc.max_move(132) > bc.max_move(180)
    assert bc.max_move(132) == pytest.approx(28.3, abs=0.1)


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
    s_lo = bc.joint_scales(bc.raw_trajectory("groove", 128, "a"), 128, g)
    s_hi = bc.joint_scales(bc.raw_trajectory("groove", 128, "a"), 128)
    assert s_lo[bc.YAW] > s_hi[bc.YAW]                      # less gain -> more of the pattern fits


def test_envelope_clamp_rules():
    c = bc.clamp_envelope
    assert c(np.array([0.0, -70.0, -60.0, 0.0, 30.0]))[bc.BP] == pytest.approx(-65.0)        # floor -65
    assert c(np.array([0.0, -58.0, -20.0, 0.0, 30.0]))[bc.BP] == pytest.approx(-52.0)        # flip region
    assert c(np.array([0.0, -58.0, -44.0, 0.0, 30.0]))[bc.BP] == pytest.approx(-52.0)        # elbow -44: still above -45
    assert c(np.array([0.0, -58.0, -47.0, 0.0, 30.0]))[bc.BP] == pytest.approx(-54.0)        # blend: 2 units under -45
    assert c(np.array([0.0, -58.0, -60.0, 0.0, 30.0]))[bc.BP] == pytest.approx(-58.0)        # elbow lifted: allowed
    out = c(np.array([120.0, -49.0, -22.0, -120.0, 80.0]))
    assert out[bc.YAW] == 94.0 and out[bc.WR] == -94.0 and out[bc.WP] == 60.0
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
        if 0 < k < bc.BEATS:                                 # the first 30 % of the next beat is a hold
            j = i + int(0.25 * 60 / bpm * bc.FPS)
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


def test_accents_on_beat_one_of_each_bar_and_the_flick_on_beat_five():
    K = bc.keyframes("groove", "a") - bc.START
    # groove a: sway +Y alternating; beats 1 and 5 are x1.3 of beats 3 and 7
    assert K[3, bc.YAW] == pytest.approx(bc.Y) and K[7, bc.YAW] == pytest.approx(bc.Y)
    assert K[1, bc.YAW] == pytest.approx(bc.ACCENT * bc.Y) and K[5, bc.YAW] == pytest.approx(bc.ACCENT * bc.Y)
    assert K[2, bc.YAW] == pytest.approx(-bc.Y) and K[4, bc.YAW] == pytest.approx(-bc.Y)
    # beat 5 carries the head flick on top of the accented bob (wrist_pitch up, wrist_roll)
    assert np.allclose(K[5] - bc.ACCENT * K[3], bc.FLICK)
    assert not np.allclose(K[1], K[5])
    for tier in ("groove", "hype"):
        for variant in bc.VARIANTS:
            K = bc.keyframes(tier, variant) - bc.START
            plain = _unaccented(tier, variant)
            assert np.allclose(K[1], bc.ACCENT * plain[0])                       # beat 1: x1.3
            assert np.allclose(K[5], bc.ACCENT * plain[4] + bc.FLICK)            # beat 5: x1.3 + the flick
            assert np.allclose(K[3], plain[2]) and np.allclose(K[7], plain[6])   # the others: nominal
    # drop: beat 1 is the spec's spring, exactly (no x1.3), and the flick is still on beat 5
    D = bc.keyframes("drop", "a")
    assert np.allclose(D[1], bc.SPRING)
    assert D[5, bc.WP] - bc.START[bc.WP] == pytest.approx(bc.ACCENT * bc.hype_excursions("a")[4][bc.WP] + bc.FLICK[bc.WP])
    # build: a progressive crouch with neither the accent nor the beat-5 flick, on every variant
    for v in bc.VARIANTS:
        B = bc.keyframes("build", v)
        assert np.allclose(B[1:8], bc.START + np.array(bc.build_excursions(v))), v
    B = bc.keyframes("build", "a")
    assert B[1, bc.EL] > B[2, bc.EL] > B[3, bc.EL] > B[4, bc.EL] > B[5, bc.EL] == B[6, bc.EL]
    assert (np.diff(B[:7, bc.BP]) <= 1e-9).all()
    for v in ("a", "b"):                                    # wrist_pitch ramps 30 -> 15 over beats 1-6, no stutter
        wp = bc.keyframes("build", v)[:7, bc.WP]
        assert (np.diff(wp) < 0).all() and wp[0] == 30 and wp[6] == 15, (v, wp)
    assert np.allclose(B[:, bc.WR], 0.0) and np.allclose(B[:, bc.YAW], 0.0)   # a sinks straight down: no flick
    # the accent x1.3 and the flick only ever touch groove/hype/drop
    for v in bc.VARIANTS:                                   # c nods on alternate beats around the ramp, no flick on 5
        C = bc.keyframes("build", v)
        assert not np.allclose(C[5] - bc.START, np.array(bc.build_excursions(v))[4] + bc.FLICK)


def _unaccented(tier, variant):
    return {"groove": bc.groove_excursions, "hype": bc.hype_excursions}[tier](variant)


def test_hype_b_has_two_bottom_up_phrases_instead_of_a_head_circle():
    K = bc.keyframes("hype", "b")
    for start in (1, 5):
        phrase = K[start:start + 3]
        # Authored joint sequence only: actual head-height direction also needs calibrated FK.
        assert (np.diff(phrase[:, [bc.BP, bc.EL, bc.WP]], axis=0) > 0).all()
        assert phrase[0, bc.EL] < bc.START[bc.EL] < phrase[-1, bc.EL]
        assert np.sign(phrase[:, bc.YAW]).tolist() == [np.sign(phrase[0, bc.YAW])] * 3
    assert K[1, bc.YAW] * K[5, bc.YAW] < 0               # rises on each side, not repeated nods
    assert np.allclose(K[4], bc.START)


def test_hype_c_crosses_both_diagonals_through_the_centre():
    K = bc.keyframes("hype", "c")
    for start in (1, 5):
        phrase = K[start:start + 3]
        assert phrase[0, bc.YAW] * phrase[-1, bc.YAW] < 0
        assert phrase[1, bc.YAW] == pytest.approx(0.0)
        assert (np.diff(phrase[:, [bc.BP, bc.EL, bc.WP]], axis=0) > 0).all()
        assert phrase[0, bc.EL] < bc.START[bc.EL] < phrase[-1, bc.EL]
    assert (K[3, bc.YAW] - K[1, bc.YAW]) * (K[7, bc.YAW] - K[5, bc.YAW]) < 0


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
@pytest.mark.parametrize("bpm", [80, 132, 180])
def test_expressive_gestures_rise_and_cross_in_calibrated_head_space(bpm):
    validator = bc.Validator(ROBOTDESC)
    for tier, variant in itertools.product(("hype", "drop"), ("b", "c")):
        U = bc.commanded(tier, bpm, variant)
        report = validator.validate(U)
        assert report["ok"], (tier, variant, bpm, report["reasons"])
        # Take the first complete frame at each beat, during its hold rather than before arrival.
        indices = np.ceil(bc.beat_instants(bpm) * bc.FPS).astype(int)
        heads = np.array([validator.model.head(dict(zip(bc.JOINTS, U[i])))["position"] for i in indices])
        for start in ((1, 5) if tier == "hype" else (5,)):
            assert heads[start + 2, 2] - heads[start, 2] > 0.01, (tier, variant, bpm, heads)
        if tier == "hype" and variant == "c":
            dx_first = heads[3, 0] - heads[1, 0]
            dx_second = heads[7, 0] - heads[5, 0]
            assert dx_first * dx_second < 0, (bpm, heads)
            assert min(abs(dx_first), abs(dx_second)) > 0.01


def test_drop_and_build_poses_follow_the_spec():
    D = bc.keyframes("drop", "b")
    assert np.allclose(D[1], [0.0, -36.0, 2.0, 0.0, 50.0])                       # spring
    assert D[2, bc.YAW] == pytest.approx(bc.Y) and D[4, bc.YAW] == pytest.approx(-bc.Y)   # b sweeps right first
    assert D[3, bc.YAW] == 0 and D[3, bc.BP] < bc.START[bc.BP]                   # ... through a centre recoil on beat 3
    A = bc.keyframes("drop", "a")
    assert A[2, bc.YAW] == pytest.approx(-bc.Y) and A[3, bc.YAW] == 0 and A[4, bc.YAW] == pytest.approx(bc.Y)
    assert np.allclose(A[5:8], bc.keyframes("hype", "a")[5:8])                   # settles into hype
    h = [e * bc.MIRROR for e in bc.hype_excursions("b")[4:7]]                    # b: hype b mirrored, then the
    assert np.allclose(D[5:8] - bc.START, [bc.ACCENT * h[0] + bc.FLICK, h[1], h[2]])   # bar accent and the flick on top
    assert np.allclose(bc.keyframes("drop", "c")[5:8], bc.keyframes("hype", "c")[5:8])
    # no single beat carries more than 1.6 Y of yaw in any drop or groove c (the sweep and the look
    # reversals are spread over two beats, so the yaw budget is not spent on one move)
    for tier, v in (("drop", "a"), ("drop", "b"), ("drop", "c"), ("groove", "c")):
        K = bc.keyframes(tier, v)
        assert np.abs(np.diff(K[:, bc.YAW])).max() <= 1.6 * bc.Y + 1e-9, (tier, v)
    # the sweep's end and the hype's first beat point the same way (or the hype starts centred)
    for v in bc.VARIANTS:
        K = bc.keyframes("drop", v)
        assert K[4, bc.YAW] * K[5, bc.YAW] >= 0
    B = bc.keyframes("build", "a")
    assert np.allclose(B[6], bc.CROUCH)                                          # deepest on beat 6
    assert bc.keyframes("build", "b")[6, bc.WR] > 15 and bc.keyframes("build", "c")[5, bc.WP] != bc.keyframes("build", "c")[6, bc.WP]
    assert np.allclose(B[8], bc.START)                                           # the drop's first frame
    # at every crouch keyframe the flip rule holds after the gain: base_pitch < -52 only with elbow <= -45
    C = bc.apply_gain(B)
    low = C[:, bc.BP] < -52
    assert (C[low][:, bc.EL] <= -45 + 1e-9).all()
    assert C[:, bc.BP].min() >= -65


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
def test_per_joint_scales_fill_the_speed_budget(bpm):
    for tier, variant in COMBOS:
        raw = bc.raw_trajectory(tier, bpm, variant)
        s = bc.joint_scales(raw, bpm)
        assert (0 < s).all() and (s <= 1).all()
        cmd = bc.commanded(tier, bpm, variant)
        speed = bc.peak_speed(cmd)
        assert speed.max() <= bc.SPEED_DESIGN, (tier, variant, speed)
        # a joint that was scaled down sits within 2 % of the budget (the amplitude is as large as
        # the speed limit allows); one step more would break it
        pre = bc.peak_speed(bc.apply_gain(bc.scaled(raw, s)))
        for j in range(5):
            if s[j] < 1.0:
                assert pre[j] >= bc.SPEED_DESIGN * bc.SPEED_MARGIN * 0.98, (tier, variant, j, pre[j])
                s2 = s.copy()
                s2[j] += 2 * bc.SCALE_STEP
                assert bc.peak_speed(bc.apply_gain(bc.scaled(raw, s2)))[j] > bc.SPEED_DESIGN * bc.SPEED_MARGIN
        assert np.allclose(cmd[0], bc.START) and np.allclose(cmd[-1], bc.START)
        assert not bc.envelope_violations(cmd)


def test_v2_is_bigger_than_v1_where_the_choreography_allows():
    # the 4-beat and 8-beat sweeps buy amplitude at the same speed budget: at 132 bpm the figure-eight
    # sways about twice as far as the beat-by-beat sway, and hype's wide sway is wider than groove a's
    A = {v: np.abs(bc.commanded("groove", 132, v)[:, bc.YAW]).max() for v in bc.VARIANTS}
    assert A["b"] > 1.7 * A["a"]
    assert A["c"] > 1.6 * A["a"]                          # the nod-led look reversals no longer bind the yaw scale
    H = np.abs(bc.commanded("hype", 132, "a")[:, bc.YAW]).max()
    assert H > 1.5 * A["a"]
    # every drop swings as wide as hype at the demo tempo (the sweep no longer spends 2 Y on one beat)
    for v in bc.VARIANTS:
        assert np.abs(bc.commanded("drop", 132, v)[:, bc.YAW]).max() > 0.95 * H, v
    # every joint is alive in every hype/groove clip (no joint stays at START)
    for tier in ("groove", "hype"):
        for v in bc.VARIANTS:
            cmd = bc.commanded(tier, 132, v)
            assert (np.abs(cmd - bc.START).max(axis=0) > 2.0).all(), (tier, v)


def test_variants_differ_in_choreography():
    for tier in bc.TIERS:
        cmds = {v: bc.commanded(tier, 128, v) for v in bc.VARIANTS}
        for a, b in itertools.combinations(bc.VARIANTS, 2):
            size = max(rms(cmds[a] - bc.START), rms(cmds[b] - bc.START))
            diff = rms(cmds[a] - cmds[b]) / size
            assert diff > 0.20, (tier, a, b, diff)


def test_commanded_clamps_after_the_gain(monkeypatch):
    # a huge sway: raw 60 -> x1.8 = 108 -> clamped to the +-94 box after the gain, not before
    monkeypatch.setattr(bc, "Y", 60.0)
    raw = bc.raw_trajectory("groove", 80, "a")
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
        assert v["head_y_min"] >= bc.HEAD_Y_MIN and v["zmp_y_min"] >= bc.ZMP_Y_MIN
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
    assert manifest["gain"] == bc.GAINS and manifest["speed_limit"] == 140.0
    assert len(manifest["clips"]) == 16
    assert manifest["aliases"] == {f"beat_{t}_128": f"beat_{t}_128_a" for t in bc.TIERS}
    c = next(c for c in manifest["clips"] if c["name"] == "beat_groove_128_b")
    assert {"name", "base", "variant", "bpm", "tier", "frames", "md5", "seconds", "first", "last", "range",
            "scale", "amplitude", "accent_beats", "flick_beat", "peak_speed"} <= set(c)
    assert c["base"] == "beat_groove_128" and c["variant"] == "b" and c["accent_beats"] == [1, 5]
    assert c["first"] == c["last"] == manifest["start_pose"]
    b = next(c for c in manifest["clips"] if c["name"] == "beat_build_128_a")
    assert b["crouch"]["base_pitch"] < -58 and b["crouch"]["elbow_pitch"] < -50 and "START" in b["note"]
    assert b["flick_beat"] is None and b["accent_beats"] == [] and c["flick_beat"] == 5
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
    monkeypatch.setattr(bc, "SPEED_LIMIT", 140.0)
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
    assert bc.bold_multiplier(1.0) == 1.0                    # exactly: the library is the bold-1.0 case
    assert bc.bold_multiplier(0.0) == pytest.approx(0.35)
    assert bc.bold_multiplier(0.6) == pytest.approx(0.74)
    assert bc.bold_multiplier(2.0) == 1.0 and bc.bold_multiplier(-1.0) == pytest.approx(0.35)
    assert bc.bold_multiplier(float("nan")) == pytest.approx(0.35)
    raw = bc.raw_trajectory("hype", 128, "a")
    assert bc.bold_trajectory("hype", 128, "a", 1.0) is not None
    assert np.array_equal(bc.bold_trajectory("hype", 128, "a", 1.0), raw)          # untouched, not a float round trip
    half = bc.bold_trajectory("hype", 128, "a", 0.0)
    assert np.allclose(half - bc.START, 0.35 * (raw - bc.START))


def test_bold_is_applied_before_the_speed_budget():
    # a speed-bound joint: bold 1.0 is scaled by the budget; bold 0.0 is a third of the RAW pattern and,
    # if that fits the budget, gets scale 1.0 -- so its commanded excursion is min(m * raw, budget), never more
    bpm = 127.3
    raw = bc.raw_trajectory("groove", bpm, "a")
    s1 = bc.joint_scales(raw, bpm)
    s0 = bc.joint_scales(bc.bold_trajectory("groove", bpm, "a", 0.0), bpm)
    assert s1[bc.YAW] < 1.0 and s0[bc.YAW] > s1[bc.YAW]
    assert (s0 >= s1 - 1e-12).all()
    for bold in (0.0, 0.6, 1.0):
        rows, meta = bc.make_clip("groove", "a", bpm, bold)
        U = np.array(rows)
        assert bc.peak_speed(U).max() <= bc.SPEED_DESIGN and not bc.envelope_violations(U)
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
            assert ok and report["ok"] and report["frames"] == len(rows) and report["peak_speed"]["base_yaw"] <= 140
            exc = _excursion(rows)
            if prev is not None:
                # every joint's excursion is monotone in bold; a speed-bound joint sits at the budget for
                # every bold (its scale is quantised in 0.005 steps, hence the 1 % tolerance)
                assert (exc >= prev * 0.99 - 1e-6).all(), (tier, variant, bold, prev, exc)
                assert exc.sum() > prev.sum()
            prev = exc
        # bold 0 is 35 % of bold 1 on a joint that neither the speed budget nor the envelope clamp binds.
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
                assert lo[j] / hi[j] == pytest.approx(0.35, abs=0.01), (tier, variant, bc.JOINTS[j])
                checked += 1
        assert checked >= 1, (tier, variant)
    # a bad row is caught, and lists of tuples are accepted like arrays
    rows, _ = bc.make_clip("groove", "a", bpm, 0.6)
    bad = list(rows); bad[40] = (0.0, -70.0, -22.0, 0.0, 30.0)
    ok, report = bc.validate_rows(bad, v)
    assert not ok and any("base_pitch" in r for r in report["reasons"])
    assert bc.validate_rows([rows[0]], v) == (False, {"ok": False, "reasons": ["rows must be (n >= 2, 5), got (1, 5)"]})
    with pytest.raises(ValueError):
        bc.make_clip("groove", "a", 20.0, 0.5)


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
    # hype b/c were re-choreographed (bottom-up phrases, crossing diagonals; #31) and drop b/c reuse those
    # hype tails, so a library generated before that legitimately differs there: only the unchanged
    # choreography is held to the old md5s (they all match again once the library is regenerated).
    real = [c for c in m["clips"] if not c.get("alias_of")
            and not (c["tier"] in ("hype", "drop") and c["variant"] in ("b", "c"))]
    assert len(real) >= 12
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
