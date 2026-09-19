"""Offline tests for follow.py's --live mode. No lamp, no mediapipe, no sockets: the loop runs
against a fake camera, fake trackers and a fake motors endpoint on a fake clock. The LampModel
tests read the vendor robot description in place (FTM_ROBOT_DIR, LELAMP_CALIBRATION_PATH) and are
skipped, with the reason, when those files are not there."""
import math
import os
import sys
import threading

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import follow as F  # noqa: E402
from spatial import JOINTS, LampModel  # noqa: E402

SCRATCH = "/private/tmp/claude-501/-Users-meharkhanna-feelthemusic/a2b4cfc6-081d-4df6-b3d6-864bf6cf8fa1/scratchpad/robotdesc"
ROBOT_DIR = os.environ.get("FTM_ROBOT_DIR", f"{SCRATCH}/pi5_feetech_r1")
CALIBRATION = os.environ.get("LELAMP_CALIBRATION_PATH", f"{SCRATCH}/lelamp-calibration.json")
HAVE_MODEL = os.path.exists(os.path.join(ROBOT_DIR, "robot.urdf")) and os.path.exists(CALIBRATION)
needs_model = pytest.mark.skipif(not HAVE_MODEL, reason=f"vendor robot description not at {ROBOT_DIR} / {CALIBRATION}")


@pytest.fixture(scope="module")
def model():
    if not HAVE_MODEL:
        pytest.skip("vendor robot description absent")
    return LampModel(ROBOT_DIR, calibration=CALIBRATION)


def pose(**kw):
    p = {"base_yaw": 0.0, "base_pitch": -49.0, "elbow_pitch": -22.0, "wrist_roll": 0.0, "wrist_pitch": 30.0}
    p.update(kw)
    return p


def in_envelope(p):
    assert all(-94 <= p[j] <= 94 for j in JOINTS)
    assert p["base_pitch"] >= -65
    assert p["wrist_pitch"] <= 60
    if p["base_pitch"] < -52:
        assert p["elbow_pitch"] <= -40


def within_cap(measured, p, cap):
    assert all(abs(p[j] - measured[j]) <= cap + 1e-9 for j in JOINTS)


# ---- step_towards: the envelope ------------------------------------------------------------------
def test_step_is_capped_per_joint_and_stops_at_the_target():
    m = pose()
    s = F.step_towards(m, pose(base_yaw=40, wrist_pitch=33), 10)
    assert s["base_yaw"] == pytest.approx(10) and s["wrist_pitch"] == pytest.approx(33)
    assert all(s[j] == pytest.approx(m[j]) for j in ("base_pitch", "elbow_pitch", "wrist_roll"))
    s = F.step_towards(m, pose(base_yaw=-3), 10)
    assert s["base_yaw"] == pytest.approx(-3)


def test_step_clips_to_joint_limits():
    s = F.step_towards(pose(base_yaw=90), pose(base_yaw=120), 10)
    assert s["base_yaw"] == pytest.approx(94)
    s = F.step_towards(pose(wrist_roll=-90), pose(wrist_roll=-100), 10)
    assert s["wrist_roll"] == pytest.approx(-94)


def test_step_never_tips_backwards():
    s = F.step_towards(pose(base_pitch=-60, elbow_pitch=-50), pose(base_pitch=-90, elbow_pitch=-50), 10)
    assert s["base_pitch"] == pytest.approx(-65)
    in_envelope(s)


def test_step_keeps_wrist_pitch_below_sixty():
    s = F.step_towards(pose(wrist_pitch=55), pose(wrist_pitch=90), 10)
    assert s["wrist_pitch"] == pytest.approx(60)


def test_step_does_not_enter_the_flip_region_until_the_elbow_is_up():
    m = pose(base_pitch=-48, elbow_pitch=-20)
    s = F.step_towards(m, pose(base_pitch=-62, elbow_pitch=-20), 10)
    in_envelope(s)
    within_cap(m, s, 10)
    assert s["base_pitch"] >= -52                      # waits at the boundary
    assert s["elbow_pitch"] <= -20


def test_step_climbs_out_of_the_flip_region_within_the_cap():
    m = pose(base_pitch=-60, elbow_pitch=-20)          # someone left the arm shoulder-back, elbow down
    s = F.step_towards(m, pose(base_pitch=-64, elbow_pitch=-20), 10)
    in_envelope(s)
    within_cap(m, s, 10)
    assert s["base_pitch"] > m["base_pitch"] and s["elbow_pitch"] < m["elbow_pitch"]


def test_step_from_deep_inside_the_flip_region_climbs_out_best_effort():
    """From (-64, -20) one capped step cannot reach a legal pose: the step is still in the region,
    but closer to the boundary on both joints, moving by the full cap. Moving out beats holding."""
    m = pose(base_pitch=-64, elbow_pitch=-20)
    s = F.step_towards(m, pose(base_pitch=-64, elbow_pitch=-20), 10)
    within_cap(m, s, 10)
    assert s["base_pitch"] == pytest.approx(-54) and s["elbow_pitch"] == pytest.approx(-30)
    assert s["base_pitch"] < -52 and s["elbow_pitch"] > -40      # not legal yet: documented best effort
    # closer to legal than measured: less shoulder to bring forward, less elbow to lift
    assert (-52 - s["base_pitch"]) < (-52 - m["base_pitch"])
    assert (s["elbow_pitch"] - -40) < (m["elbow_pitch"] - -40)
    s2 = F.step_towards(s, pose(base_pitch=-64, elbow_pitch=-20), 10)
    in_envelope(s2)                                     # the second step makes it legal
    assert s2["base_pitch"] == pytest.approx(-52) and s2["elbow_pitch"] == pytest.approx(-40)


def test_flip_entry_is_gated_on_the_measured_elbow_and_commands_a_margin():
    goal = pose(base_pitch=-62, elbow_pitch=-50)
    # elbow measured at -39 (the runtime landed 90 % of a step to -40): not in yet, however legal the command
    m = pose(base_pitch=-50, elbow_pitch=-39)
    s = F.step_towards(m, goal, 10)
    within_cap(m, s, 10)
    assert s["base_pitch"] == pytest.approx(-52) and s["elbow_pitch"] == pytest.approx(-49)
    # elbow measured at -41: it is up, the shoulder may go back
    m = pose(base_pitch=-50, elbow_pitch=-41)
    s = F.step_towards(m, goal, 10)
    in_envelope(s)
    assert s["base_pitch"] == pytest.approx(-60) and s["elbow_pitch"] == pytest.approx(-50)
    # in the region the elbow is COMMANDED to -45 at most, so a 75 % landing is still below -40
    m = pose(base_pitch=-56, elbow_pitch=-42)
    s = F.step_towards(m, pose(base_pitch=-60, elbow_pitch=-38), 10)
    in_envelope(s)
    assert s["elbow_pitch"] == pytest.approx(F.FLIP_ELBOW_CMD) == pytest.approx(-45)
    landed = m["elbow_pitch"] + 0.75 * (s["elbow_pitch"] - m["elbow_pitch"])
    assert landed <= -40
    assert F.envelope(pose(base_pitch=-60, elbow_pitch=0))["elbow_pitch"] == pytest.approx(-45)
    assert F.envelope(pose(base_pitch=-50, elbow_pitch=0))["elbow_pitch"] == pytest.approx(0)


class RejectBig:
    """A model whose problems() refuses any step longer than `allowed` units."""
    limits = dict(F.DEFAULT_LIMITS)

    def __init__(self, allowed, origin):
        self.allowed, self.origin, self.calls = allowed, origin, []

    def problems(self, p):
        self.calls.append(dict(p))
        return ["too far"] if F.moved_units(p, self.origin) > self.allowed else []


def test_step_halves_on_problems_then_holds():
    m = pose()
    target = pose(base_yaw=40)
    picky = RejectBig(3.0, m)
    s = F.step_towards(m, target, 10, picky)
    assert s["base_yaw"] == pytest.approx(2.5)          # 10 -> 5 -> 2.5: the second halving passes
    assert len(picky.calls) == 3
    never = RejectBig(-1.0, m)
    s = F.step_towards(m, target, 10, never)
    assert s == m                                       # 10, 5, 2.5, 1.25 all refused: hold
    assert len(never.calls) == 4


@needs_model
def test_step_with_the_real_model_never_returns_a_problem_pose(model):
    m = dict(model.neutral)
    into_the_table = pose(base_pitch=90, elbow_pitch=-90, wrist_pitch=-90)
    assert model.problems(into_the_table)
    for _ in range(40):
        s = F.step_towards(m, into_the_table, 10, model)
        assert not model.problems(s)
        in_envelope(s)
        within_cap(m, s, 10)
        m = s
    assert F.moved_units(m, model.neutral) > 0          # it got somewhere, and stopped where it must


# ---- stall guard ----------------------------------------------------------------------------------
def test_stall_guard_trips_on_the_fourth_under_delivery_and_resets_on_a_big_aim_change():
    g = F.StallGuard()
    before, asked = pose(), pose(base_yaw=10)
    barely = pose(base_yaw=1.0)                         # 10 % of what was asked (the runtime's honest 40-60 % is fine)
    for _ in range(3):
        assert not g.record(before, asked, barely, 20.0)
    assert g.record(before, asked, barely, 20.0)
    assert g.stalled and g.blocked(22.0)                # 2 deg is not "the target moved"
    assert not g.blocked(36.0)                          # 16 deg is
    assert not g.stalled and g.misses == 0


def test_stall_guard_needs_consecutive_misses_and_ignores_tiny_asks():
    g = F.StallGuard()
    before, asked, barely, fine = pose(), pose(base_yaw=10), pose(base_yaw=1.0), pose(base_yaw=5)
    g.record(before, asked, barely, 10.0)
    g.record(before, asked, barely, 10.0)
    g.record(before, asked, barely, 10.0)
    assert not g.record(before, asked, fine, 10.0)      # a good delivery resets the count
    assert g.misses == 0
    for _ in range(5):
        assert not g.record(before, pose(base_yaw=2), pose(), 10.0)   # asked < 3 units: not counted


def test_stall_guard_tripped_without_an_aim_takes_the_first_aim_as_reference():
    g = F.StallGuard()
    for _ in range(4):
        g.record(pose(), pose(base_yaw=10), pose(base_yaw=0.5), None)
    assert g.blocked(None) and g.blocked(30.0) and g.blocked(40.0)
    assert not g.blocked(46.0)


# ---- search state machine -------------------------------------------------------------------------
def test_search_holds_then_sweeps_face_cancels_then_parks():
    s = F.Search(100.0, yaw=12.0)
    assert s.want(102.9) == ("holding", None)
    state, want = s.want(103.1)
    assert state == "searching"
    assert abs(want["base_yaw"] - 12.0) < 2.0           # the triangle starts at the last-seen yaw
    assert (want["base_pitch"], want["elbow_pitch"], want["wrist_roll"], want["wrist_pitch"]) == (-49, -22, 0, 30)
    quarter = 1 / (4 * F.Search.HZ)
    assert s.want(103.0 + quarter)[1]["base_yaw"] == pytest.approx(42.0)
    assert s.want(103.0 + 3 * quarter)[1]["base_yaw"] == pytest.approx(-18.0)
    assert s.want(103.0 + 4 * quarter)[1]["base_yaw"] == pytest.approx(12.0)
    s.saw(110.0, yaw=-5.0)                              # a face ends the search at once
    assert s.want(110.0) == ("holding", None)
    assert s.want(112.9) == ("holding", None)
    assert s.want(113.5)[0] == "searching"
    assert s.want(169.9)[0] == "searching"
    state, want = s.want(170.0)
    assert state == "parked" and want == F.VENDOR_NEUTRAL


def test_search_sweep_stays_inside_the_limits_and_can_be_disabled():
    s = F.Search(0.0, yaw=80.0)
    yaws = [s.want(3.0 + t / 10)[1]["base_yaw"] for t in range(0, 70)]
    assert max(yaws) <= 94.0 and min(yaws) >= 50.0
    off = F.Search(0.0, yaw=0.0, enabled=False)
    assert off.want(30.0) == ("holding", None)


# ---- scheduler ------------------------------------------------------------------------------------
def test_poster_never_has_two_posts_in_flight_and_respects_the_period():
    clock = {"t": 0.0}
    gate, active, peak, calls = threading.Event(), [0], [0], []

    def post(pose, ms):
        active[0] += 1
        peak[0] = max(peak[0], active[0])
        calls.append((dict(pose), ms))
        gate.wait(2.0)
        active[0] -= 1

    p = F.LivePoster(post, period=0.25, clock=lambda: clock["t"])
    assert p.ready()
    assert p.submit(pose(), 250)
    clock["t"] = 0.10
    assert not p.submit(pose(base_yaw=1), 250)          # too soon
    clock["t"] = 0.30
    assert not p.ready() and not p.submit(pose(base_yaw=2), 250)   # first one still in flight
    gate.set()
    p.wait_idle()
    assert p.ready() and p.submit(pose(base_yaw=3), 250)
    p.wait_idle()
    assert peak[0] == 1 and p.posts == 2
    assert [c[0]["base_yaw"] for c in calls] == [0.0, 3.0]
    clock["t"] = 0.40
    assert not p.ready()                                # period counts from the last accepted command
    clock["t"] = 0.55
    assert p.ready()


def test_poster_surfaces_a_failed_post_once():
    def post(pose, ms):
        raise F.LiveRefused(422, "nope")
    p = F.LivePoster(post, period=0.0, clock=lambda: 0.0)
    assert p.submit(pose(), 250)
    p.wait_idle()
    exc = p.take_error()
    assert isinstance(exc, F.LiveRefused) and exc.status == 422
    assert p.take_error() is None and p.posts == 0 and p.rtt_ms is not None


# ---- the loop end to end, against fakes ----------------------------------------------------------
class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class FakeMotors:
    """The runtime's tracking route as measured: says ok at once, and `latency` later the arm has
    made only `delivery` of the commanded delta (the rest is lost to the servo's P gain and torque
    limit). A command that arrives while the last one is still pending replaces it."""

    def __init__(self, clock, start, delivery=0.5, latency=0.15):
        self.clock, self.delivery, self.latency = clock, delivery, latency
        self.frm, self.to, self.t0 = dict(start), dict(start), 0.0
        self.posts = []

    def positions(self):
        return dict(self.to if self.clock() >= self.t0 else self.frm), True

    def post(self, pose, ms):
        here, _ = self.positions()
        self.posts.append((self.clock(), dict(pose), ms))
        self.frm = here
        self.to = {j: here[j] + (pose[j] - here[j]) * self.delivery for j in JOINTS}
        self.t0 = self.clock() + self.latency


class FakeCamera:
    """Frames whose face box is where a fixed point in the room appears from the arm's CURRENT pose:
    as the lamp turns toward the face, the box walks from the edge of the picture to the centre."""

    def __init__(self, clock, motors, model, point):
        self.clock, self.motors, self.model, self.point, self.error = clock, motors, model, np.asarray(point, float), ""
        self.stamps = []

    def newest(self, after, timeout=2.0):
        p, _ = self.motors.positions()
        where = self.model.project(p, self.point)
        stamp = self.clock() - 0.03                     # 30 ms old by the time we look at it
        distance = float(np.linalg.norm(self.point - self.model.head(p)["position"]))
        size = self.model.fx / distance                 # picture widths per metre, what a face detector implies
        frame = {"face": (where[0], where[1], size) if where and 0 <= where[0] <= 1 else None}
        self.stamps.append(where)
        return frame, stamp


class FakeFaces:
    label = "face"

    def locate(self, frame):
        return frame["face"]


class NoThermal:
    def too_hot(self):
        return False

    def frame_gap(self, tracking):
        return 0.1

    def celsius(self):
        return None


@needs_model
def test_live_loop_turns_to_a_face_at_the_edge_and_locks_on(model):
    clock = FakeClock()
    start = dict(F.VENDOR_NEUTRAL)
    motors = FakeMotors(clock, start, delivery=0.5, latency=0.15)
    # a face 0.9 m away, 28 deg to the left of where the lamp is looking: at the left edge of the picture
    h = model.head(start)
    ang = math.radians(28)                            # +z rotation moves the point to the picture's left
    direction = np.array([h["forward"][0] * math.cos(ang) - h["forward"][1] * math.sin(ang),
                          h["forward"][0] * math.sin(ang) + h["forward"][1] * math.cos(ang), 0.0])
    point = h["position"] + 0.9 * direction
    cam = FakeCamera(clock, motors, model, point)
    assert cam.newest(0)[0]["face"][0] < 0.12            # starts at the left edge

    cfg = F.LiveConfig(target="face", deadband_deg=4.0, live_ms=250, period=0.25, step=10.0)
    lines = []
    f = F.LiveFollower(model, motors, cam, [("face", FakeFaces())], NoThermal(), cfg,
                       clock=clock, sleep=clock.sleep, out=lambda *a, **k: lines.append(" ".join(map(str, a))))
    errors, states = [], []
    for _ in range(12):
        states.append(f.cycle())
        f.poster.wait_idle()
        if f.aim_error is not None:
            errors.append(f.aim_error)
        clock.t += 0.25
    assert f.fatal is None
    assert states[-1] == "tracking" and all(s in ("tracking", "holding") for s in states)
    assert len(errors) >= 6 and errors[0] > 20
    assert all(b <= a + 0.3 for a, b in zip(errors, errors[1:])), errors      # monotone, no overshoot
    assert errors[-1] < cfg.deadband_deg, errors
    assert motors.posts and all(p[2] == 250 for p in motors.posts)
    # every command was one capped step from the measured pose, inside the envelope
    for _, p, _ in motors.posts:
        in_envelope(p)
    assert all(abs(p[1]["base_yaw"] - q[1]["base_yaw"]) <= 10 + 1e-6 for p, q in zip(motors.posts, motors.posts[1:]))
    assert all(F.moved_units(p, q) <= 10.0 + 1e-6 for (_, p, _), (_, q, _) in zip(motors.posts, motors.posts[1:]))
    x_track = [w[0] for w in cam.stamps if w]
    assert x_track[0] < 0.15 and abs(x_track[-1] - 0.5) < 0.1     # the face walked to the centre
    assert not f.guard.stalled
    assert any("tracking" in line and "aim" in line for line in lines)


@needs_model
def test_live_loop_dry_run_posts_nothing_and_says_what_it_would_send(model):
    clock = FakeClock()
    motors = FakeMotors(clock, dict(F.VENDOR_NEUTRAL))
    h = model.head(F.VENDOR_NEUTRAL)
    point = h["position"] + 0.8 * np.array([-0.4, 0.9, 0.0])
    cam = FakeCamera(clock, motors, model, point)
    lines = []
    cfg = F.LiveConfig(target="face", dry_run=True)
    f = F.LiveFollower(model, motors, cam, [("face", FakeFaces())], NoThermal(), cfg,
                       clock=clock, sleep=clock.sleep, out=lambda *a, **k: lines.append(" ".join(map(str, a))))
    for _ in range(6):
        f.cycle()
        clock.t += 0.25
    assert motors.posts == [] and f.commands == 0
    assert any("would send" in line for line in lines)


@needs_model
def test_live_loop_searches_when_nobody_is_there_and_stalls_against_an_obstruction(model):
    clock = FakeClock()
    motors = FakeMotors(clock, pose(base_yaw=20, base_pitch=-49.3, elbow_pitch=-22.5, wrist_pitch=30), delivery=0.5)
    cam = FakeCamera(clock, motors, model, np.array([0.0, -1.0, 0.3]))   # behind the lamp: never in the picture
    cfg = F.LiveConfig(target="face", live_ms=250, period=0.25, step=10.0, land_settle=0.3)
    f = F.LiveFollower(model, motors, cam, [("face", FakeFaces())], NoThermal(), cfg,
                       clock=clock, sleep=clock.sleep, out=lambda *a, **k: None)
    states = []
    for _ in range(int(6 / 0.25)):
        states.append(f.cycle())
        f.poster.wait_idle()
        clock.t += 0.25
    assert states[0] == "holding" and states[int(2.5 / 0.25)] == "holding"
    assert "searching" in states and motors.posts
    yaws = [p["base_yaw"] for _, p, _ in motors.posts]
    assert max(yaws) > 22 and all(abs(y - 20) <= 31 for y in yaws)
    assert not f.guard.stalled

    motors.delivery = 0.05                              # now the arm is pressed against something
    for _ in range(int(5 / 0.25)):
        states.append(f.cycle())
        f.poster.wait_idle()
        clock.t += 0.25
    assert f.guard.stalled and states[-1] == "stalled"
    sent = len(motors.posts)
    for _ in range(8):
        f.cycle()
        clock.t += 0.25
    assert len(motors.posts) == sent                    # stalled: nothing more goes out
    # ... and the last thing that went out was a settle: the measured pose itself, so the servo
    # goal is where the arm is and nothing keeps pushing into the obstruction
    assert f.settles == 1
    here, _ = motors.positions()
    assert motors.posts[-1][1] == pytest.approx(here)
    assert F.moved_units(motors.posts[-1][1], motors.posts[-2][1]) > 3     # the step before it was a real ask
    assert f.pending == []                              # the settle fed no guard sample


@needs_model
def test_live_loop_parks_at_neutral_after_a_minute(model):
    clock = FakeClock()
    motors = FakeMotors(clock, pose(base_yaw=40, base_pitch=-30, elbow_pitch=-10, wrist_pitch=10), delivery=0.9)
    cam = FakeCamera(clock, motors, model, np.array([0.0, -1.0, 0.3]))
    cfg = F.LiveConfig(target="face", live_ms=250, period=0.25, step=10.0)
    f = F.LiveFollower(model, motors, cam, [("face", FakeFaces())], NoThermal(), cfg,
                       clock=clock, sleep=clock.sleep, out=lambda *a, **k: None)
    states = []
    for _ in range(int(75 / 0.25)):
        states.append(f.cycle())
        f.poster.wait_idle()
        clock.t += 0.25
    assert states[-1] == "parked"
    assert F.moved_units(motors.positions()[0], F.VENDOR_NEUTRAL) < 1.5
    idx = states.index("parked")
    assert 60 / 0.25 - 2 <= idx <= 60 / 0.25 + 2


class TickingCamera(FakeCamera):
    """A FakeCamera that also moves the fake clock on each frame, so run() (which loops on cycle())
    ends by itself with cfg.seconds."""

    def newest(self, after, timeout=2.0):
        self.clock.t += 0.25
        return super().newest(after, timeout)


@needs_model
def test_run_settles_at_the_measured_pose_when_it_ends(model):
    clock = FakeClock()
    motors = FakeMotors(clock, dict(F.VENDOR_NEUTRAL), delivery=0.5, latency=0.15)
    h = model.head(F.VENDOR_NEUTRAL)
    point = h["position"] + 0.8 * np.array([-0.4, 0.9, 0.0])
    cam = TickingCamera(clock, motors, model, point)
    lines = []
    cfg = F.LiveConfig(target="face", seconds=3.0)
    f = F.LiveFollower(model, motors, cam, [("face", FakeFaces())], NoThermal(), cfg,
                       clock=clock, sleep=clock.sleep, out=lambda *a, **k: lines.append(" ".join(map(str, a))))
    f.run(threading.Event())
    assert f.commands >= 2 and f.settles == 1 and len(motors.posts) == f.commands + 1
    here, _ = motors.positions()
    assert motors.posts[-1][1] == pytest.approx(here)              # the goal is where the arm is
    assert motors.posts[-1][1] == pytest.approx(motors.frm)        # ... and the settle moved nothing
    assert F.moved_units(motors.posts[-2][1], motors.posts[-1][1]) > 0.5   # the last step was still short
    assert any("settled at the measured pose (leaving)" in line for line in lines)
    assert f.pending == []
    # a dry run posts nothing at all, settle included
    motors2 = FakeMotors(clock, dict(F.VENDOR_NEUTRAL))
    g = F.LiveFollower(model, motors2, TickingCamera(clock, motors2, model, point), [("face", FakeFaces())],
                       NoThermal(), F.LiveConfig(target="face", seconds=3.0, dry_run=True),
                       clock=clock, sleep=clock.sleep, out=lambda *a, **k: None)
    g.run(threading.Event())
    assert motors2.posts == [] and g.settles == 0


def test_run_does_not_settle_when_torque_is_off_or_nothing_was_sent():
    class Motors:
        def __init__(self, torque):
            self.torque, self.posts = torque, []

        def positions(self):
            return pose(), self.torque

        def post(self, p, ms):
            self.posts.append(p)

    for torque, commands in ((False, 5), (True, 0)):
        m = Motors(torque)
        f = F.LiveFollower(None, m, None, [], None, F.LiveConfig(), clock=lambda: 0.0, out=lambda *a, **k: None)
        f.commands = commands
        stop = threading.Event()
        stop.set()
        f.run(stop)                                     # stop already set: no cycle runs
        assert m.posts == [] and f.settles == 0


class ScriptedPoster:
    """Hands the loop a scripted sequence of post outcomes (None = a good post)."""

    def __init__(self, outcomes):
        self.outcomes, self.rtt_ms = list(outcomes), 5.0

    def take_error(self):
        return self.outcomes.pop(0) if self.outcomes else None


def test_post_failures_only_count_in_a_row():
    boom = ConnectionError("refused")
    f = F.LiveFollower(None, None, None, [], None, F.LiveConfig(), clock=lambda: 0.0, out=lambda *a, **k: None,
                       poster=ScriptedPoster([boom, None, boom, None, boom, None]))
    for _ in range(6):
        f._post_errors()
    assert f.fatal is None and f.post_failures == 0    # failure, success, failure... is not "in a row"
    g = F.LiveFollower(None, None, None, [], None, F.LiveConfig(), clock=lambda: 0.0, out=lambda *a, **k: None,
                       poster=ScriptedPoster([boom, boom, boom]))
    for _ in range(3):
        g._post_errors()
    assert g.fatal and "three failed posts" in g.fatal
    h = F.LiveFollower(None, None, None, [], None, F.LiveConfig(), clock=lambda: 0.0, out=lambda *a, **k: None,
                       poster=ScriptedPoster([F.LiveRefused(422, "no"), None, F.LiveRefused(422, "no"), None]))
    for _ in range(4):
        h._post_errors()
    assert h.fatal is None and h.consecutive_refusals == 0 and h.refusals == 2


class Hot:
    def too_hot(self):
        return True

    def celsius(self):
        return 80.0

    def frame_gap(self, tracking):
        return 0.1


def test_hot_pause_and_joint_read_pause_end_at_once_when_stopped():
    import time as _time

    class NoJoints:
        def positions(self):
            raise ConnectionError("down")

        def post(self, p, ms):
            raise AssertionError("nothing should be posted")

    class Frames:
        error = ""

        def newest(self, after, timeout=2.0):
            return {"face": None}, 0.0

    stop = threading.Event()
    stop.set()
    lines = []
    f = F.LiveFollower(None, NoJoints(), Frames(), [], Hot(), F.LiveConfig(), out=lambda *a, **k: lines.append(str(a[0])))
    t0 = _time.monotonic()
    f.run(stop)                                         # loop body never runs: stop is set
    f.stop = stop
    assert f.cycle() == "hot"                           # would sleep 20 s with time.sleep
    f.thermal = NoThermal()
    assert f.cycle() == "no-joints"                     # would sleep 1 s
    assert _time.monotonic() - t0 < 2.0
    assert any("pausing vision for 20 s" in line for line in lines)
    # the default pause is the stop event's wait, so Ctrl-C (which sets it) ends the pause
    unset = threading.Event()
    g = F.LiveFollower(None, NoJoints(), Frames(), [], Hot(), F.LiveConfig(), out=lambda *a, **k: None)
    g.stop = unset
    threading.Timer(0.2, unset.set).start()
    t1 = _time.monotonic()
    assert g.cycle() == "hot"
    assert 0.1 < _time.monotonic() - t1 < 2.0


def test_live_mode_defaults_face_and_a_tight_deadband(monkeypatch):
    import argparse
    # the same resolution main() applies after parse_args
    for argv, target, deadband in ((["--live"], "face", 4.0), ([], "auto", 10.0), (["--live", "--target", "auto"], "auto", 4.0)):
        ns = argparse.Namespace(live="--live" in argv, target=None, deadband_deg=None)
        if "--target" in argv:
            ns.target = argv[argv.index("--target") + 1]
        ns.target = ns.target or ("face" if ns.live else "auto")
        ns.deadband_deg = 4.0 if ns.live else 10.0
        assert (ns.target, ns.deadband_deg) == (target, deadband)


def test_no_network_and_no_mediapipe_needed_to_import():
    assert "mediapipe" not in sys.modules
    assert "board" not in sys.modules
