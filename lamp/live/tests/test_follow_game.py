"""Offline tests for follow_game.py (contract: docs/follow-game.md). Pure: no socket, no panel, no robot, no
vendor description; the forward-kinematics tests use a toy arm written in spatial.py's frame convention."""
import json
import math
import os
import sys
import threading
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import follow_game as fg  # noqa: E402

START = fg.START_POSE


def row(**excursion):
    """A COMMANDED frame in JOINTS order: START plus the given excursions."""
    return [START[j] + excursion.get(j, 0.0) for j in fg.JOINTS]


def wave(n, period=40, amp=0.8):
    return [amp * math.sin(2 * math.pi * i / period) for i in range(n)], [amp * math.cos(2 * math.pi * i / period) for i in range(n)]


# ---- the wire ---------------------------------------------------------------------------------------------
def test_lpath_messages_follow_the_contract():
    x, y = wave(200)
    msgs = fg.lpath_messages(17, 123_456_789_012_345, x, y, clip="beat_hype_120_a")
    assert len(msgs) == 3
    for k, m in enumerate(msgs):
        assert set(m) == {"t", "seq", "t0", "dt", "x", "y", "clip"}
        assert m["t"] == "lpath" and m["seq"] == 17 + k and m["dt"] == 50 and m["clip"] == "beat_hype_120_a"
        assert isinstance(m["t0"], int) and len(m["x"]) == len(m["y"]) and 2 <= len(m["x"]) <= 80
        assert all(isinstance(v, int) and -1000 <= v <= 1000 for v in m["x"] + m["y"])
        d = fg.encode_telemetry(m)
        assert d[0] == 4 and len(d) <= 1200 and b" " not in d
        assert json.loads(d[1:]) == m
    assert fg.lpath_messages(0, 0, [0.5], [0.5]) == [] and fg.lpath_messages(0, 0, [], []) == []
    assert "clip" not in fg.lpath_messages(0, 0, [0, 1], [0, 1])[0]
    assert fg.lpath_messages(0xFFFFFFFF, 0, x, y)[1]["seq"] == 0                    # u32 wrap
    assert fg.lpath_messages(0, 0, x, y, dt_ms=5)[0]["dt"] == 20 and fg.lpath_messages(0, 0, x, y, dt_ms=900)[0]["dt"] == 250


def test_the_worst_case_datagram_fits_even_with_the_keys_the_mac_adds():
    m = fg.lpath_messages(0xFFFFFFF0, 2**63 - 1 - 80 * 50_000_000, [-1.0] * 80, [-1.0] * 80, clip="c" * 200)[0]
    assert len(m["x"]) == 80 and len(m["clip"]) == fg.CLIP_NAME_MAX
    assert len(fg.encode_telemetry(m)) <= 1200
    assert len(fg.encode_telemetry({**m, "v": 1, "lamp": 255})) <= 1200           # the type-14 relay of the same JSON


def test_thousandths_are_rounded_clamped_and_nan_proof():
    assert [fg.thousandths(v) for v in (0.1234, -0.0006, 1.5, -2, 1, float("nan"), float("inf"))] == [123, -1, 1000, -1000, 1000, 0, 1000]


def test_a_long_path_becomes_consecutive_messages_that_share_their_boundary_point():
    x, y = wave(200)
    t0 = 5_000_000_000
    msgs = fg.lpath_messages(7, t0, x, y)
    assert [len(m["x"]) for m in msgs] == [80, 80, 42]
    assert [m["seq"] for m in msgs] == [7, 8, 9]
    assert [m["t0"] for m in msgs] == [t0, t0 + 79 * 50_000_000, t0 + 158 * 50_000_000]
    for a, b in zip(msgs, msgs[1:]):
        assert a["t0"] + (len(a["x"]) - 1) * a["dt"] * 1_000_000 == b["t0"]          # b starts when a ends ...
        assert (a["x"][-1], a["y"][-1]) == (b["x"][0], b["y"][0])                     # ... and where it ends
    joined = msgs[0]["x"] + msgs[1]["x"][1:] + msgs[2]["x"][1:]
    assert joined == [fg.thousandths(v) for v in x]
    # exactly one message more than fits: the second one still has two points
    assert [len(m["x"]) for m in fg.lpath_messages(0, 0, x[:81], y[:81])] == [80, 2]
    assert [len(m["x"]) for m in fg.lpath_messages(0, 0, x[:80], y[:80])] == [80]


# ---- the path: frame and signs ------------------------------------------------------------------------------
def test_without_fk_a_yaw_toward_the_lamps_right_is_positive_x_and_nothing_is_mirrored():
    """+base_yaw turns the head toward the lamp's OWN right (measured on the arm, see the module docstring):
    x must come out positive. The phone does the mirroring; a sign flip here would cancel it."""
    frames = [row(), row(base_yaw=20.0), row(base_yaw=-20.0)]
    x, y = fg.head_path([0.0, 0.1, 0.2], frames)
    assert x[0] == 0.0 and x[1] > 0.0 and x[2] < 0.0 and x[1] == pytest.approx(-x[2])
    assert y == [0.0, 0.0, 0.0]
    assert fg.head_path([0.0, 0.1, 0.2], frames, yaw_sign=-1.0)[0][1] < 0.0            # the one knob, if a rebuild flips the servo


def test_without_fk_up_is_positive_y_and_wrist_joints_are_ignored():
    up, crouch = row(base_pitch=5 * 1.15, elbow_pitch=14 * 1.2, wrist_pitch=12 * 1.1), row(base_pitch=-11 * 1.15, elbow_pitch=-33 * 1.2)
    x, y = fg.head_path([0, 1, 2, 3], [row(), up, crouch, row(wrist_pitch=25.0, wrist_roll=20.0)])
    assert y[1] == pytest.approx(19 / fg.Y_RANGE_UNITS) and y[2] == pytest.approx(-1.0)
    assert (x[3], y[3]) == (0.0, 0.0) and x[1] == 0.0


def test_commanded_values_are_scaled_to_what_the_arm_delivers_and_clamped():
    x, _ = fg.head_path([0, 1, 2], [row(), row(base_yaw=1.8 * fg.X_RANGE_UNITS), row(base_yaw=94.0)])
    assert x[1] == pytest.approx(1.0) and x[2] == 1.0
    x, _ = fg.head_path([0, 1], [row(), row(base_yaw=14.0)], delivery={})                  # already physical: no scaling
    assert x[1] == pytest.approx(0.5)
    x, y = fg.head_path([0, 1], [row(), [float("nan")] * 5])                              # a bad number is no excursion
    assert (x[1], y[1]) == (0.0, 0.0)
    assert fg.head_path([0, 1], [dict(START), {"base_yaw": 1.8 * 7.0}])[0][1] == pytest.approx(0.25)   # dict rows, joints missing


def test_head_path_checks_its_inputs():
    with pytest.raises(ValueError): fg.head_path([0, 1], [row()])
    with pytest.raises(ValueError): fg.head_path([0, 0], [row(), row()])
    with pytest.raises(ValueError): fg.head_path([0, 1], [row(), [1, 2, 3]])


class ToyArm:
    """A toy in spatial.py's base frame (metres, z up, forward = +y at neutral yaw, so the lamp's right = +x).
    +base_yaw turns the head toward +x, as measured on the real lamp; the head pivot is R in front of the yaw
    axis and rises with base_pitch + elbow_pitch. `flip` builds the opposite servo direction."""
    R, H, RAD, RISE = 0.08, 0.30, math.radians(0.9), 0.002

    def __init__(self, flip=False, broken=False):
        self.sign, self.broken, self.calls = (-1.0 if flip else 1.0), broken, 0

    def head(self, u):
        self.calls += 1
        if self.broken:
            raise RuntimeError("no model")
        th = self.sign * (u["base_yaw"] - START["base_yaw"]) * self.RAD
        z = self.H + self.RISE * ((u["base_pitch"] - START["base_pitch"]) + (u["elbow_pitch"] - START["elbow_pitch"]))
        return {"position": (self.R * math.sin(th), self.R * math.cos(th), z), "forward": (math.sin(th), math.cos(th), 0.0)}


def test_with_fk_x_is_the_base_frames_x_and_y_its_z():
    arm = ToyArm()
    frames = [row(), row(base_yaw=1.8 * 14.0), row(base_yaw=-1.8 * 14.0), row(base_pitch=1.15 * 13, elbow_pitch=1.2 * 24)]
    x, y = fg.head_path([0, 1, 2, 3], frames, arm)
    assert x[1] > 0.3 and x[2] == pytest.approx(-x[1]) and abs(y[1]) < 1e-9
    assert x[1] == pytest.approx(math.sin(14 * arm.RAD) / math.sin(28 * arm.RAD))      # +-1 is X_RANGE_UNITS through the model
    assert y[3] == pytest.approx(37 / 44) and abs(x[3]) < 1e-9                          # the spring, against the crouch's range
    xr, yr = fg.workspace_m(arm)
    assert xr == pytest.approx((arm.R + fg.LOOK_M) * math.sin(28 * arm.RAD)) and yr == pytest.approx(44 * arm.RISE)
    # explicit ranges in metres win over the defaults
    assert fg.head_path([0, 1], frames[:2], arm, x_range_m=1.0, y_range_m=1.0)[0][1] == pytest.approx((arm.R + fg.LOOK_M) * math.sin(14 * arm.RAD))
    # with a model the MODEL decides the sign, not yaw_sign: a lamp whose servo runs the other way comes out negative
    assert fg.head_path([0, 1], frames[:2], ToyArm(flip=True))[0][1] < -0.3
    # a callable that only returns a position works too (no face offset then)
    pos = fg.head_path([0, 1], frames[:2], lambda u: arm.head(u)["position"])[0][1]
    assert pos == pytest.approx(math.sin(14 * arm.RAD) / math.sin(28 * arm.RAD))


def test_a_model_that_fails_falls_back_to_joint_units():
    frames = [row(), row(base_yaw=1.8 * 14.0)]
    assert fg.head_path([0, 1], frames, ToyArm(broken=True)) == fg.head_path([0, 1], frames)
    assert fg.head_path([0, 1], frames, lambda u: (float("nan"), 0.0, 0.0)) == fg.head_path([0, 1], frames)


def test_resample_puts_30_fps_frames_on_the_50_ms_grid():
    times = [i / 30 for i in range(31)]                                                  # 1 s of clip
    x, y = fg.resample(times, [t * 0.5 for t in times], [-t for t in times])
    assert len(x) == 21 and x[0] == 0.0
    assert all(a == pytest.approx(0.5 * 0.05 * i) for i, a in enumerate(x)) and y[20] == pytest.approx(-1.0)
    assert fg.resample([0.0], [0.3], [0.1]) == ([0.3], [0.1])
    with pytest.raises(ValueError): fg.resample([0, 1], [0], [0, 1])


def test_trim_still_keeps_a_short_lead_in_and_one_point_after_the_last_move():
    x = [0.0] * 12 + [0.1, 0.2, 0.3] + [0.3] * 10
    i0, tx, ty = fg.trim_still(x, [0.0] * len(x))
    assert i0 == 6 and tx == [0.0] * 6 + [0.1, 0.2, 0.3, 0.3] and len(ty) == len(tx)       # 5 lead-in points + the last still one
    assert fg.trim_still(x, [0.0] * len(x), lead_in_s=0.0)[0] == 11
    assert fg.trim_still([0.0] * 3 + [0.5], [0.0] * 4)[0] == 0                             # never before the start
    assert fg.trim_still([0.2] * 30, [0.1] * 30) == (0, [], [])                            # nothing moves: nothing to follow


# ---- the room's status ---------------------------------------------------------------------------------------
def status_packet(**kw):
    j = {"v": 1, "seq": 41, "ts": 123456789012345, "n": 3, "nOk": 1, "best": 0.83, "ok": True, "level": 0.91, "rgb": [46, 255, 0]}
    j.update(kw)
    return b"\x0f" + json.dumps({k: v for k, v in j.items() if v is not None}).encode()


def test_level_rgb_is_the_contracts_colour():
    assert fg.level_rgb(0.91) == (46, 255, 0)                                               # the contract's own example
    assert [fg.level_rgb(l) for l in (0.0, 0.5, 1.0, -3, 7, float("nan"))] == [(255, 0, 0), (255, 255, 0), (0, 255, 0), (255, 0, 0), (0, 255, 0), (255, 0, 0)]


def test_parse_follow_status():
    st = fg.parse_follow_status(status_packet(), now=10.0)
    assert st == {"seq": 41, "ts": 123456789012345, "n": 3, "nOk": 1, "best": 0.83, "ok": True, "level": 0.91,
                  "rgb": (46, 255, 0), "idle": False, "rx": 10.0}
    assert fg.parse_follow_status(status_packet(idle=True, n=0))["idle"] is True
    assert fg.parse_follow_status(status_packet(n=0))["idle"] is True                       # nobody playing is idle, flag or not
    assert fg.parse_follow_status(status_packet(rgb=None, level=0.5))["rgb"] == (255, 255, 0)
    assert fg.parse_follow_status(status_packet(rgb=[300, -4, "x"], level=1.0))["rgb"] == (0, 255, 0)
    assert fg.parse_follow_status(status_packet(rgb=[300.4, -4, 12]))["rgb"] == (255, 0, 12)
    assert fg.parse_follow_status(status_packet(level=1.7))["level"] == 1.0
    assert fg.parse_follow_status(status_packet(ok="yes"))["ok"] is False
    assert fg.parse_follow_status(status_packet(unknown={"future": 1}))["seq"] == 41        # unknown keys are ignored
    for bad in (b"", b"\x0f", b"\x0fnot json", b"\x0f[1,2]", b"\x0d" + status_packet()[1:], status_packet(level=None),
                status_packet(level="0.5"), status_packet(level=True), b"\x0f{\"level\": NaN}", b"\x0f\xff\xfe"):
        assert fg.parse_follow_status(bad) is None


def run_light(light, status_at, music=(10, 10, 40), seconds=3.0, t0=100.0, limiter=None, fps=50):
    out = []
    for i in range(int(seconds * fps)):
        now = t0 + i / fps
        st = status_at(now)
        if st is not None:
            light.status = st
        out.append(light.rgb(music, now, limiter))
    return out


def test_status_colour_glides_in_is_dimmed_and_never_steps():
    light = fg.FollowLight()
    green = lambda now: fg.parse_follow_status(status_packet(level=1.0, rgb=[0, 255, 0]), now)
    out = run_light(light, green)
    assert out[0] != (0, int(round(255 * fg.FOLLOW_VALUE)), 0)                              # not a jump onto the colour
    assert out[-1] == (0, int(round(255 * fg.FOLLOW_VALUE)), 0) and light.engaged
    steps = [max(abs(a - b) for a, b in zip(p, q)) for p, q in zip(out, out[1:])]
    assert max(steps) <= math.ceil(fg.RATE_PER_S / 50) + 1 <= fg.MAX_STEP
    assert max(max(c) for c in out) <= round(255 * fg.FOLLOW_VALUE)
    # whatever the call rate, one call moves a channel by at most MAX_STEP
    slow = fg.FollowLight(); slow.status = green(0.0)
    a = slow.rgb((0, 0, 0), 0.0); slow.status = green(5.0); b = slow.rgb((0, 0, 0), 5.0)
    assert max(abs(p - q) for p, q in zip(a, b)) <= fg.MAX_STEP


def test_stale_or_idle_status_gives_the_music_colour_back():
    music = (10, 10, 40)
    light = fg.FollowLight()
    assert fg.status_rgb(None, music, 0.0, light=light) == music
    assert fg.status_rgb(fg.parse_follow_status(status_packet(idle=True, n=0), 0.0), music, 0.1, light=light) == music
    assert fg.status_rgb(fg.parse_follow_status(status_packet(), 0.0), music, 1.5, light=light) == music      # 1.5 s old
    assert not light.engaged
    # engaged, then the conductor goes quiet: glide back onto the music colour, then let go
    once = fg.parse_follow_status(status_packet(level=0.0, rgb=[255, 0, 0]), 100.0)
    out = run_light(light, lambda now: once if now < 100.5 else None, music, seconds=4.0)
    assert out[30][0] > 50 and out[-1] == music and not light.engaged
    back = out[50:]
    assert max(max(abs(a - b) for a, b in zip(p, q)) for p, q in zip(back, back[1:])) <= fg.MAX_STEP
    # a music colour that never settles cannot hold the panel for ever
    light2 = fg.FollowLight(); light2.status = fg.parse_follow_status(status_packet(), 0.0)
    for i in range(400):
        now = i / 50
        got = light2.rgb((255, 255, 255) if (i // 5) % 2 else (0, 0, 0), now)
    assert not light2.engaged and got in ((255, 255, 255), (0, 0, 0))


def test_engaging_goes_through_the_flash_limiter():
    class Limiter:                                      # lamp_show.FlashLimiter's rule
        def __init__(self): self.onsets = []
        def allow(self, now):
            self.onsets = [t for t in self.onsets if now - t <= 1.0]
            if len(self.onsets) >= 3: return False
            self.onsets.append(now); return True
    lim, light, music = Limiter(), fg.FollowLight(), (10, 10, 40)
    for t in (99.7, 99.8, 99.9):
        assert lim.allow(t)                             # the music show just used the second's three onsets
    out = run_light(light, lambda now: fg.parse_follow_status(status_packet(), now), music, seconds=2.0, limiter=lim)
    waited = [o == music for o in out]
    assert all(waited[:35]) and not all(waited[:45])    # held off until the oldest onset left the window (~0.7 s)
    assert light.engaged and len(lim.onsets) <= 3 and sum(1 for t in lim.onsets if t >= 100.0) == 1   # one onset, not one per frame


def flashes_per_second(values, fps=50, swing=0.2 * 255):
    """The most up-and-back-down luminance swings of >= 20 % of full scale that start in any one second."""
    onsets, lo, hi, rising = [], values[0], values[0], False
    for i, v in enumerate(values):
        if not rising:
            lo = min(lo, v)
            if v - lo >= swing: rising, hi = True, v
        else:
            hi = max(hi, v)
            if hi - v >= swing: onsets.append(i / fps); rising, lo = False, v
    return max((sum(1 for u in onsets if t <= u < t + 1.0) for t in onsets), default=0)


@pytest.mark.parametrize("period", [0.1, 0.25, 0.5, 1.0])
def test_no_status_stream_can_make_the_panel_flash(period):
    """The conductor's level glides, but the lamp does not rely on it: even a status that jumps between
    saturated red, black and green every `period` seconds stays under three flashes a second."""
    colours = ([255, 0, 0], [0, 0, 0], [0, 255, 0], [0, 0, 0])
    st = lambda now: fg.parse_follow_status(status_packet(rgb=colours[int((now - 100.0) / period) % 4]), now)
    out = run_light(fg.FollowLight(), st, (0, 0, 0), seconds=6.0)
    assert flashes_per_second([max(c) for c in out]) <= 2
    assert flashes_per_second([c[0] for c in out]) <= 2                                     # the red channel on its own
    assert flashes_per_second([0, 255] * 150) > 3                                           # the detector does detect


# ---- publishing ------------------------------------------------------------------------------------------------
def write_clip(path, yaw_at):
    """A 30 fps clip CSV like beat_clips writes: absolute timestamps, <joint>.pos columns."""
    lines = ["timestamp," + ",".join(j + ".pos" for j in fg.JOINTS)]
    for i in range(int(3.0 * 30) + 1):
        lines.append(f"{1000.0 + i / 30:.6f}," + ",".join(f"{v:.4f}" for v in row(base_yaw=yaw_at(i / 30))))
    path.write_text("\n".join(lines) + "\n")


def sway(t):
    """0.6 s hold, a 1.2 s commanded sway to the lamp's right and back, then still."""
    return 0.0 if t < 0.6 or t > 1.8 else 1.8 * 20.0 * math.sin(math.pi * (t - 0.6) / 1.2)


def test_load_clip_csv_reads_the_runtime_format(tmp_path):
    write_clip(tmp_path / "beat_test_120.csv", sway)
    (tmp_path / "broken.csv").write_text("timestamp,base_yaw.pos\n1,2\n")
    times, rows = fg.load_clip_csv("beat_test_120", [str(tmp_path / "nowhere"), str(tmp_path)])
    assert len(times) == 91 and times[0] == 0.0 and times[-1] == pytest.approx(3.0) and len(rows[0]) == 5
    assert rows[0] == [START[j] for j in fg.JOINTS] and max(r[0] for r in rows) == pytest.approx(36.0, abs=0.1)
    assert fg.load_clip_csv("broken", [str(tmp_path)]) is None and fg.load_clip_csv("missing", [str(tmp_path)]) is None


def publisher(tmp_path, sent, offset=lambda: 7_000_000_000, **kw):
    kw.setdefault("fk", None)
    return fg.PathPublisher(sent.append, offset, hold_ns=600_000_000, dirs=[str(tmp_path)], sync=True, seq0=500, log=lambda *_: None, **kw)


def test_publisher_stamps_the_instant_the_head_starts_moving_on_the_mac_clock(tmp_path):
    write_clip(tmp_path / "beat_test_120.csv", sway)
    sent = []
    pub = publisher(tmp_path, sent)
    beat_ns = fg.time.monotonic_ns() + 950_000_000                                          # the scheduler's beat instant, lamp clock
    msgs = pub.clip_posted("beat_test_120", beat_ns)
    assert len(msgs) == 1 and len(sent) == 2 and sent[0] == sent[1]                          # sent twice; de-duplicated by seq
    m = json.loads(sent[0][1:])
    assert sent[0][0] == 4 and m["t"] == "lpath" and m["seq"] == 500 and m["clip"] == "beat_test_120"
    # first frame at beat - hold; the clip is still for 0.6 s; 0.25 s of that is kept as the lead-in -> grid index 7;
    # + the servo lag; + offset_ns puts it on the Mac clock
    assert m["t0"] == beat_ns - 600_000_000 + 7 * 50_000_000 + fg.SERVO_LAG_NS + 7_000_000_000
    assert m["x"][:6] == [0] * 6 and max(m["x"]) == pytest.approx(1000 * 20 / fg.X_RANGE_UNITS, abs=2) and min(m["x"]) >= 0
    assert set(m["y"]) == {0}
    assert pub.seq == 501 and pub.sent == 1
    pub.clip_posted("beat_test_120", beat_ns + 4_000_000_000)                                # cached; the next seq
    assert json.loads(sent[-1][1:])["seq"] == 501


def test_publisher_adds_the_measured_first_frame_error_clamped(tmp_path):
    write_clip(tmp_path / "c.csv", sway)
    sent = []
    base = publisher(tmp_path, sent).clip_posted("c", 10_000_000_000)[0]["t0"]
    assert publisher(tmp_path, sent, phase=lambda: 40.0).clip_posted("c", 10_000_000_000)[0]["t0"] == base + 40_000_000
    assert publisher(tmp_path, sent, phase=lambda: -5000.0).clip_posted("c", 10_000_000_000)[0]["t0"] == base - 200_000_000
    assert publisher(tmp_path, sent, phase=lambda: 1 / 0).clip_posted("c", 10_000_000_000)[0]["t0"] == base


def test_publisher_sends_nothing_it_cannot_stand_behind(tmp_path):
    write_clip(tmp_path / "still.csv", lambda t: 0.0)
    write_clip(tmp_path / "c.csv", sway)
    sent = []
    assert publisher(tmp_path, sent, offset=lambda: None).clip_posted("c", 1) is None        # clock not synced: no timestamp
    pub = publisher(tmp_path, sent)
    assert pub.clip_posted("dance_fwd", 1) is None                                           # a vendor clip we know nothing about
    assert pub.clip_posted("still", 1) is None and sent == [] and pub.skipped == 2
    def boom(b): raise BlockingIOError
    assert fg.PathPublisher(boom, lambda: 0, dirs=[str(tmp_path)], fk=None, sync=True, log=lambda *_: None).clip_posted("c", 1) is None


def test_stop_ends_a_running_path_with_a_still_one(tmp_path):
    write_clip(tmp_path / "c.csv", sway)
    sent = []
    pub = publisher(tmp_path, sent)
    assert pub.stop() is None and sent == []                                                 # nothing running: nothing to end
    pub.clip_posted("c", fg.time.monotonic_ns() + 950_000_000)
    del sent[:]
    m = pub.stop()[0]
    now_mac = fg.time.monotonic_ns() + 7_000_000_000
    assert (m["x"], m["y"], m["clip"], m["seq"]) == ([0, 0], [0, 0], "home", 501) and 0 < m["t0"] - now_mac + 5_000_000 <= 105_000_000
    assert pub.stop() is None                                                                # once
    # sent home while a clip's path was still being worked out: that path is for a cancelled clip
    pub.offset = lambda: (pub.stop(), 7_000_000_000)[1]
    assert pub.clip_posted("c", fg.time.monotonic_ns() + 950_000_000) is None and sent[-1][1:] == fg.encode_telemetry(m)[1:]


def test_generate_clip_is_the_fallback_for_library_names():
    pytest.importorskip("numpy")
    times, rows = fg.generate_clip("beat_hype_120_a")
    assert len(times) == len(rows) == 156 and times[1] == pytest.approx(1 / 30) and rows[0] == [START[j] for j in fg.JOINTS]
    assert fg.generate_clip("beat_hype_120") == (times, rows)                               # the v1 alias is variant a
    assert fg.generate_clip("dance_fwd") is None and fg.generate_clip("beat_nope_120_a") is None
    x, y = fg.head_path(times, rows)
    assert max(x) > 0.4 and min(x) < -0.3 and max(y) > 0.3 and min(y) < -0.3                # the dance fills a good part of the range


# ---- attach(): the hook ------------------------------------------------------------------------------------------
class FakeScheduler:
    hold_ns, phase_ms = 600_000_000, []

    def __init__(self, answer):
        self.answer, self.posted = answer, []
        self.post_fn = self.runtime

    def runtime(self, name):
        self.posted.append(name)
        return dict(self.answer)

    def _post(self, name, beat_ns, late_ns):
        self.result = self.post_fn(name)


class FakePanel:
    def __init__(self): self.lock, self.dark, self.music, self.brightness = threading.Lock(), False, (10, 10, 40), 0.75
    def frame(self, now): return self.music, self.brightness


def fake_show(answer):
    sent, handled = [], []
    show = types.SimpleNamespace(scheduler=FakeScheduler(answer), panel=FakePanel(), limiter=None, offset_ns=7_000_000_000,
                                 send=sent.append, handle=lambda d, now: handled.append(d))
    return show, sent, handled


def test_attach_publishes_only_after_the_runtime_accepted_and_changes_nothing_else(tmp_path):
    write_clip(tmp_path / "c.csv", sway)
    show, sent, handled = fake_show({"status": "started"})
    pub, light = fg.attach(show, log=lambda *_: None, dirs=[str(tmp_path)], fk=None, sync=True)
    beat_ns = fg.time.monotonic_ns() + 950_000_000
    show.scheduler._post("c", beat_ns, 0)
    assert show.scheduler.posted == ["c"] and show.scheduler.result == {"status": "started"}   # the post itself is untouched
    assert len(sent) == 2 and json.loads(sent[0][1:])["t0"] == beat_ns - 600_000_000 + 7 * 50_000_000 + fg.SERVO_LAG_NS + 7_000_000_000
    show.scheduler.post_fn("home")                                                          # dance stop: the path ends on the phones
    assert json.loads(sent[-1][1:])["clip"] == "home"
    # refused by the runtime: nothing is published
    show2, sent2, _ = fake_show({"error": "busy"})
    fg.attach(show2, log=lambda *_: None, dirs=[str(tmp_path)], fk=None, sync=True)
    show2.scheduler._post("c", 5_000_000_000, 0)
    assert show2.scheduler.result == {"error": "busy"} and sent2 == []
    # a publisher that blows up never reaches the scheduler
    show3, _, _ = fake_show({"status": "started"})
    pub3, _ = fg.attach(show3, log=lambda *_: None, sync=True)
    pub3.clip_posted = lambda *a: 1 / 0
    show3.scheduler._post("c", 1, 0)
    assert show3.scheduler.result == {"status": "started"}


def test_attach_looks_for_clips_where_the_live_generator_writes_them(tmp_path):
    show, _, _ = fake_show({"status": "started"})
    show.scheduler.live = types.SimpleNamespace(pack_dir=str(tmp_path))
    pub, _ = fg.attach(show, log=lambda *_: None, sync=True)
    assert pub.dirs[0] == str(tmp_path) and fg.PACK_DIR in pub.dirs and fg.STAGE_DIR in pub.dirs


def test_attach_takes_type_15_and_passes_everything_else_on():
    show, _, handled = fake_show({"status": "started"})
    _, light = fg.attach(show, log=lambda *_: None, sync=True)
    show.handle(b"\x0d{}", 1.0); show.handle(status_packet(), 1.0); show.handle(b"\x0fgarbage", 1.0)
    assert handled == [b"\x0d{}"] and light.packets == 1 and light.status["seq"] == 41


def test_attach_colours_the_panel_unless_it_is_dark():
    show, _, _ = fake_show({"status": "started"})
    _, light = fg.attach(show, log=lambda *_: None, sync=True)
    assert show.panel.frame(100.0) == ((10, 10, 40), 0.75)                                  # no status: the music show, burst and all
    out = None
    for i in range(150):
        now = 100.0 + i / 50
        show.handle(status_packet(level=1.0, rgb=[0, 255, 0]), now)
        out = show.panel.frame(now)
    assert out == ((0, 153, 0), 0.6)                                                        # the status colour, without the DROP burst
    show.panel.dark = True
    assert show.panel.frame(103.0) == ((10, 10, 40), 0.75)                                  # lights off: not ours to override


def test_lamp_show_hook_is_off_unless_asked_for(monkeypatch):
    import lamp_show as L

    class Panel:
        def __init__(self): self.lock, self.model, self.dark, self.mode, self.audio_source, self.frames = threading.Lock(), L.ColourModel(), False, "fake", (lambda: None), 0
        def start(self): pass
        def frame(self, now): return (1, 2, 3), L.HW_BRIGHTNESS

    class Sock:
        def __init__(self): self.sent = []
        def setsockopt(self, *a): pass
        def setblocking(self, *a): pass
        def sendto(self, b, dst): self.sent.append(b)

    monkeypatch.setattr(L, "lamp", lambda path, body=None, timeout=4.0: {"status": "started"} if path.endswith("/play") else
                        {"current_animation": "c", "playing": True, "elapsed_seconds": 0.0})
    monkeypatch.setattr(L, "Panel", Panel)
    monkeypatch.setattr(L.socket, "socket", lambda *a, **k: Sock())
    monkeypatch.delenv("FOLLOW_GAME", raising=False)
    plain = L.Show("127.0.0.1", "light", False)
    assert not hasattr(plain, "follow_game") and plain.scheduler.post_fn("x") == {"status": "started"}
    monkeypatch.setenv("FOLLOW_GAME", "1")
    show = L.Show("127.0.0.1", "light", False)
    pub, light = show.follow_game
    show.handle(status_packet(), 5.0)
    assert light.packets == 1 and show.counts["control"] == 0
    show.handle(b"\x0d" + json.dumps({"lat": 250}).encode(), 5.0)                           # everything else still reaches Show.handle
    assert show.lat_ms == 250 and show.counts["control"] == 1
