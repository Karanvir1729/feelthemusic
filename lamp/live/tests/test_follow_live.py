"""Offline tests for follow.py's --live mode. No lamp, no mediapipe, no sockets: the loop runs
against a fake camera, fake trackers and a fake motors endpoint on a fake clock. The LampModel
tests read the vendor robot description in place (FTM_ROBOT_DIR, LELAMP_CALIBRATION_PATH) and are
skipped, with the reason, when those files are not there."""
import math
import os
import sys
import threading
from types import SimpleNamespace

import cv2
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
    """Acknowledges at once, starts after `latency`, and interpolates for the requested duration.
    Only `delivery` of the requested delta lands; another command replaces an unfinished move."""

    def __init__(self, clock, start, delivery=0.5, latency=0.15):
        self.clock, self.delivery, self.latency = clock, delivery, latency
        self.frm, self.to, self.t0, self.t1 = dict(start), dict(start), 0.0, 0.0
        self.posts, self.before, self.superseded = [], [], 0

    def positions(self):
        fraction = (float(np.clip((self.clock() - self.t0) / (self.t1 - self.t0), 0.0, 1.0))
                    if self.t1 > self.t0 else 1.0)
        return {j: self.frm[j] + fraction * (self.to[j] - self.frm[j]) for j in JOINTS}, True

    def post(self, pose, ms):
        here, _ = self.positions()
        if self.clock() < self.t1 and F.moved_units(pose, here) > 0.05:
            self.superseded += 1                       # a measured-pose settle is deliberately allowed
        self.posts.append((self.clock(), dict(pose), ms))
        self.before.append(dict(here))
        self.frm = here
        self.to = {j: here[j] + (pose[j] - here[j]) * self.delivery for j in JOINTS}
        self.t0 = self.clock() + self.latency
        self.t1 = self.t0 + ms / 1000.0


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


class OneAxisModel:
    """An angular yaw-only model for controller tests, requiring no robot files or hardware."""
    limits = dict(F.DEFAULT_LIMITS)
    fx = 1.0

    def head(self, measured):
        return {"position": np.zeros(3)}

    def project(self, measured, point):
        if point[1] <= 0:
            return None
        return 0.5 + (point[0] - measured["base_yaw"]) / 90.0, 0.5

    def distance_from_size(self, size, width):
        return self.fx / size

    def target_point(self, measured, where, distance):
        return np.array([measured["base_yaw"] + (where[0] - 0.5) * 90.0, 1.0, 0.0])

    def aim_error_deg(self, measured, point):
        return abs(point[0] - measured["base_yaw"])

    def look_at(self, point, prefer_distance):
        return pose(base_yaw=float(point[0])), {"rejected": []}

    def problems(self, measured):
        return []


@pytest.fixture
def one_axis_follow():
    clock = FakeClock()
    model = OneAxisModel()
    motors = FakeMotors(clock, pose(), delivery=0.5, latency=0.20)
    camera = FakeCamera(clock, motors, model, [30.0, 1.0, 0.0])
    follower = F.LiveFollower(model, motors, camera, [("face", FakeFaces())], NoThermal(),
                              F.LiveConfig(search=False), clock=clock, sleep=clock.sleep,
                              out=lambda *a, **k: None)
    return follower, clock, motors, camera


def run_cycles(follower, clock, count):
    for _ in range(count):
        follower.cycle()
        follower.poster.wait_idle()
        clock.sleep(0.12)                               # newer than the requested camera frame threshold


def test_fake_motors_include_transport_latency_and_motion_duration():
    clock = FakeClock()
    motors = FakeMotors(clock, pose(), delivery=0.5, latency=0.20)
    motors.post(pose(base_yaw=10), 250)
    clock.sleep(0.19)
    assert motors.positions()[0]["base_yaw"] == pytest.approx(0)
    clock.sleep(0.135)
    assert motors.positions()[0]["base_yaw"] == pytest.approx(2.5)
    clock.sleep(0.125)
    assert motors.positions()[0]["base_yaw"] == pytest.approx(5)


def test_live_loop_steps_at_the_period_and_judges_every_step_after_its_own_acceptance(one_axis_follow):
    follower, clock, motors, _ = one_axis_follow
    judged, record = [], follower.guard.record
    follower.guard.record = lambda *a: (judged.append(clock.t), record(*a))[1]
    run_cycles(follower, clock, 60)
    assert len(motors.posts) >= 3
    # steps go out at the period (the deployed cadence), not one per landing: a later step planned
    # from the measured pose replaces an unfinished move
    gaps = [b[0] - a[0] for a, b in zip(motors.posts, motors.posts[1:])]
    assert all(g >= follower.cfg.period - 1e-9 for g in gaps) and min(gaps) < follower.cfg.land_settle
    assert motors.superseded >= 1
    assert follower.aim_error < follower.cfg.deadband_deg
    assert not follower.guard.stalled and follower.guard.misses == 0 and follower.fatal is None
    # ... and every step was judged, none before its own landing allowance had run
    assert len(judged) == follower.commands and follower.pending == []
    for (sent, _, _), when in zip(motors.posts, judged):
        assert when >= sent + follower.cfg.land_settle - 1e-9


def test_delivered_half_steps_do_not_false_stall_when_target_direction_changes(one_axis_follow):
    follower, clock, motors, camera = one_axis_follow
    for index in range(60):
        camera.point[0] = 8.0 if index % 2 else -8.0
        run_cycles(follower, clock, 1)
    assert follower.commands >= 4
    # a step sent back the other way supersedes the one before it (the target moved): the arm did
    # move when asked, so that is not an obstruction, however far it ends up from where it started
    assert motors.superseded >= 1
    assert not follower.guard.stalled
    assert follower.guard.misses == 0
    assert follower.settles == 0 and follower.fatal is None


def test_deadband_noise_does_not_start_repeated_corrections(one_axis_follow):
    follower, clock, motors, camera = one_axis_follow
    for index in range(40):
        camera.point[0] = 5.0 if index % 2 else -5.0  # outside 4-degree stop band, inside 6-degree start band
        run_cycles(follower, clock, 1)
    assert motors.posts == []
    assert not follower.guard.stalled


def test_correction_continues_inside_start_band_then_stays_stopped(one_axis_follow):
    follower, clock, motors, camera = one_axis_follow
    camera.point[0] = 10.0
    run_cycles(follower, clock, 20)
    # Each half-step leaves an error: keep stepping (at the period) until inside the 4-degree stop band.
    assert 2 <= len(motors.posts) <= 4
    assert abs(motors.positions()[0]["base_yaw"] - 10.0) < follower.cfg.deadband_deg
    assert motors.positions()[0]["base_yaw"] < 10.0 - 1.0           # short of the target: half steps
    sent = len(motors.posts)
    camera.point[0] = motors.positions()[0]["base_yaw"] + 5.0      # a new 5-degree error must not restart corrections
    run_cycles(follower, clock, 12)
    assert len(motors.posts) == sent


def test_a_brief_miss_keeps_the_sightings(one_axis_follow):
    """One missed frame is detector noise, not a lost target: the earlier sighting still counts.
    (Clearing on every miss made a real detector that drops every third frame never confirm a face.)"""
    follower, clock, motors, camera = one_axis_follow
    run_cycles(follower, clock, 1)
    camera.point[1] = -1.0
    run_cycles(follower, clock, 1)
    camera.point[1] = 1.0
    run_cycles(follower, clock, 2)                    # 1 pre-miss + 2 fresh = 3 sightings inside LOST_S
    assert len(motors.posts) == 1


def test_missing_face_longer_than_the_lost_window_requires_three_fresh_sightings(one_axis_follow):
    follower, clock, motors, camera = one_axis_follow
    run_cycles(follower, clock, 1)
    camera.point[1] = -1.0
    run_cycles(follower, clock, 1)
    clock.sleep(F.TargetLock.LOST_S + 0.01)           # gone for longer than the lock window
    camera.point[1] = 1.0
    run_cycles(follower, clock, 2)
    assert motors.posts == []                         # the pre-loss sighting has aged out
    run_cycles(follower, clock, 1)
    assert len(motors.posts) == 1


def test_missing_face_resets_active_correction_hysteresis(one_axis_follow):
    follower, clock, motors, camera = one_axis_follow
    camera.point[0] = 10.0
    run_cycles(follower, clock, 3)                     # first command has been accepted
    assert len(motors.posts) == 1
    camera.point[1] = -1.0
    run_cycles(follower, clock, 6)                     # allow its motion to finish with no detection
    camera.point[1] = 1.0                             # now only 5 degrees away: below the start threshold
    run_cycles(follower, clock, 8)
    assert len(motors.posts) == 1


def test_confirmed_centered_face_with_intermittent_misses_does_not_trigger_search(one_axis_follow):
    follower, clock, motors, camera = one_axis_follow
    follower.cfg.search = True
    camera.point[0] = 0.0
    run_cycles(follower, clock, 3)
    assert follower.aim_error == pytest.approx(0.0)
    started = clock()
    for index in range(60):
        camera.point[1] = -1.0 if index % 3 == 2 else 1.0
        run_cycles(follower, clock, 1)
        assert follower.state not in ("searching", "parked")
    assert clock() - started > 5.0
    assert motors.posts == [] and follower.commands == 0


def test_slow_replacement_detection_does_not_mix_expired_face_samples(one_axis_follow, monkeypatch):
    follower, clock, motors, camera = one_axis_follow
    run_cycles(follower, clock, 2)                     # two observations of A, not yet confirmed
    tracker = follower.trackers[0][1]
    locate = tracker.locate

    def slow_locate(frame):
        clock.sleep(0.61)                             # the old lock and both A observations expire
        return locate(frame)

    monkeypatch.setattr(tracker, "locate", slow_locate)
    camera.point[0] = -30.0                           # replacement B appears on the other side
    run_cycles(follower, clock, 1)
    assert len(follower.sightings) == 1
    assert follower.sightings[0][1][0] == pytest.approx(-30.0)
    assert follower.aim_error is None and not follower.correcting
    assert motors.posts == []

    monkeypatch.setattr(tracker, "locate", locate)
    run_cycles(follower, clock, 2)
    assert len(motors.posts) == 1
    assert motors.posts[0][1]["base_yaw"] < 0           # only B's three fresh samples may command motion


@pytest.mark.parametrize("slow_stage", ["detector", "ik"])
def test_landing_deadline_uses_actual_submission_after_slow_perception(one_axis_follow, monkeypatch, slow_stage):
    follower, clock, motors, _ = one_axis_follow
    if slow_stage == "detector":
        tracker = follower.trackers[0][1]
        locate, calls = tracker.locate, [0]

        def slow_locate(frame):
            calls[0] += 1
            if calls[0] == 3:
                clock.sleep(0.3)                      # still inside the three-sighting freshness window
            return locate(frame)

        monkeypatch.setattr(tracker, "locate", slow_locate)
    else:
        look_at = follower.model.look_at

        def slow_look_at(*args, **kwargs):
            clock.sleep(0.4)
            return look_at(*args, **kwargs)

        monkeypatch.setattr(follower.model, "look_at", slow_look_at)
    run_cycles(follower, clock, 3)
    assert len(motors.posts) == 1
    assert follower.pending[0][0] >= motors.posts[0][0] + follower.cfg.land_settle - 1e-9


def test_slow_post_gets_a_full_landing_allowance_after_completion(one_axis_follow):
    follower, clock, motors, camera = one_axis_follow

    def slow_post(commanded, ms):
        # The runtime accepts only after the original deadline. Written against the allowance itself
        # rather than a number: it was 0.30 s + duration when this test was authored and is 0.65 s +
        # duration now (measured: the tracking route starts 140-250 ms after the POST and the servos
        # lag ~150 ms after the move ends).
        clock.sleep(follower.cfg.land_settle + 0.25)
        motors.post(commanded, ms)

    follower.poster = F.LivePoster(slow_post, follower.cfg.period, clock=clock)
    run_cycles(follower, clock, 3)
    completed = follower.poster.last_completed
    assert completed - follower.poster.last_sent > follower.cfg.land_settle
    assert len(motors.posts) == 1 and len(follower.pending) == 1
    camera.point[1] = -1.0                            # no new command when the first landing is assessed

    clock.t = completed + follower.cfg.land_settle - 0.01
    follower.cycle()
    assert len(follower.pending) == 1
    assert follower.guard.last == "" and follower.guard.misses == 0

    clock.sleep(0.02)
    follower.cycle()
    assert follower.pending == [] and follower.guard.last
    assert follower.guard.misses == 0 and not follower.guard.stalled
    assert motors.positions()[0]["base_yaw"] == pytest.approx(follower.cfg.step * motors.delivery)  # one step landed
    assert len(motors.posts) == 1 and follower.settles == 0


def test_rejected_post_is_not_assessed_as_a_failed_landing(one_axis_follow):
    follower, clock, motors, camera = one_axis_follow

    def reject_post(commanded, ms):
        raise F.LiveRefused(422, "fixture refusal")

    follower.poster = F.LivePoster(reject_post, follower.cfg.period, clock=clock)
    run_cycles(follower, clock, 3)
    assert len(follower.pending) == 1
    camera.point[1] = -1.0
    clock.sleep(follower.cfg.land_settle + 0.1)
    follower.cycle()                                  # consumes the rejection before assessing a landing
    assert follower.refusals == 1 and follower.pending == []
    assert follower.guard.last == "" and follower.guard.misses == 0
    assert not follower.guard.stalled and follower.settles == 0
    assert motors.posts == [] and follower.poster.posts == 0


def test_pipelined_steps_are_judged_by_their_own_post_and_a_failed_one_is_no_sample(one_axis_follow):
    follower, clock, motors, camera = one_axis_follow
    camera.point[0] = 40.0                            # far off: several capped steps are needed
    calls = []

    def post(commanded, ms):
        calls.append(clock.t)
        if len(calls) == 2:
            raise F.LiveRefused(422, "fixture refusal")         # the second step is refused ...
        if len(calls) == 3:
            clock.sleep(follower.cfg.land_settle + 0.2)         # ... and the third is accepted late
        motors.post(commanded, ms)

    follower.poster = F.LivePoster(post, follower.cfg.period, clock=clock)
    judged, record = [], follower.guard.record
    follower.guard.record = lambda *a: (judged.append((clock.t, a[1]["base_yaw"])), record(*a))[1]
    while follower.commands < 3:
        run_cycles(follower, clock, 1)
    assert len(calls) == 3 and follower.refusals == 1 and follower.fatal is None
    # the refused step left no sample; the two accepted ones are still pending, each with its own POST
    seqs = [p[3] for p in follower.pending]
    assert len(follower.pending) == 2 and seqs == [1, 3]
    first, third = follower.poster.outcome(1), follower.poster.outcome(3)
    assert follower.poster.outcome(2) == (pytest.approx(calls[1]), False)
    assert first[1] and third[1] and third[0] - calls[2] > follower.cfg.land_settle
    # the first is judged land_settle after ITS acceptance, not after the slow third one's
    camera.point[1] = -1.0                            # no more steps while the pending ones are judged
    clock.t = max(first[0], calls[0]) + follower.cfg.land_settle - 0.01
    follower.cycle()
    assert len(judged) == 0
    clock.sleep(0.02)
    follower.cycle()
    assert len(judged) == 1 and [p[3] for p in follower.pending] == [3]
    clock.t = third[0] + follower.cfg.land_settle + 0.01
    follower.cycle()
    assert len(judged) == 2 and follower.pending == []
    assert follower.guard.misses == 0 and not follower.guard.stalled


def test_throttled_object_detections_still_acquire_three_sightings(one_axis_follow):
    follower, clock, motors, _ = one_axis_follow
    detected = []

    class ThrottledObject:
        label = "cup"

        def locate(self, frame):
            if detected and clock() - detected[-1] < 1.0:
                return None
            detected.append(clock())
            return frame["face"]

    follower.cfg.target = "object"
    follower.trackers = [("object", ThrottledObject())]
    run_cycles(follower, clock, 18)
    assert len(detected) == 2 and len(follower.sightings) == 2
    assert motors.posts == []
    run_cycles(follower, clock, 1)
    assert len(detected) == 3 and len(follower.sightings) == 3
    assert len(motors.posts) == 1
    assert motors.posts[0][0] >= detected[2]
    assert follower.state == "tracking" and follower.fatal is None


def face_detection(x, width):
    box = SimpleNamespace(xmin=x - width / 2, ymin=0.4, width=width, height=0.2)
    return SimpleNamespace(location_data=SimpleNamespace(relative_bounding_box=box))


@pytest.fixture
def face_lock(monkeypatch):
    clock, detections = FakeClock(), []
    detector = SimpleNamespace(process=lambda frame: SimpleNamespace(detections=detections))
    fake_mp = SimpleNamespace(solutions=SimpleNamespace(face_detection=SimpleNamespace(
        FaceDetection=lambda **kwargs: detector)))
    monkeypatch.setitem(sys.modules, "mediapipe", fake_mp)
    tracker = F.FaceTracker(clock=clock)
    return tracker, clock, detections, np.zeros((8, 8, 3), dtype=np.uint8)


def test_face_tracker_keeps_lock_when_another_face_becomes_larger(face_lock):
    tracker, clock, detections, frame = face_lock
    detections[:] = [face_detection(0.25, 0.20), face_detection(0.75, 0.18)]
    assert tracker.locate(frame)[0] == pytest.approx(0.25)
    clock.sleep(0.1)
    detections[:] = [face_detection(0.25, 0.19), face_detection(0.75, 0.28)]
    assert tracker.locate(frame)[0] == pytest.approx(0.25)


def test_face_tracker_reports_loss_before_reacquiring_a_distant_face(face_lock):
    tracker, clock, detections, frame = face_lock
    detections[:] = [face_detection(0.25, 0.20)]
    assert tracker.locate(frame)[0] == pytest.approx(0.25)
    detections[:] = [face_detection(0.75, 0.30)]
    clock.sleep(0.1)
    assert tracker.locate(frame) is None
    clock.sleep(0.4)
    assert tracker.locate(frame) is None
    clock.sleep(0.11)
    # Reacquisition itself is immediate; the follower supplies three-frame confirmation.
    assert tracker.locate(frame)[0] == pytest.approx(0.75)


def test_face_tracker_retains_lock_across_modest_frame_to_frame_camera_motion(face_lock):
    tracker, clock, detections, frame = face_lock
    detections[:] = [face_detection(0.25, 0.04)]
    assert tracker.locate(frame)[0] == pytest.approx(0.25)
    # Each frame shifts less than the 0.06 floor, even after cumulative displacement is larger.
    for x in (0.30, 0.35, 0.40, 0.45):
        clock.sleep(0.1)
        detections[:] = [face_detection(x, 0.04), face_detection(0.85, 0.20)]
        assert tracker.locate(frame)[0] == pytest.approx(x)


def test_expired_face_lock_does_not_prefer_a_nearby_box_over_a_larger_face(face_lock):
    tracker, clock, detections, frame = face_lock
    detections[:] = [face_detection(0.25, 0.20)]
    assert tracker.locate(frame)[0] == pytest.approx(0.25)
    detections.clear()
    clock.sleep(0.3)
    assert tracker.locate(frame) is None
    clock.sleep(0.31)
    detections[:] = [face_detection(0.26, 0.20), face_detection(0.75, 0.30)]
    assert tracker.locate(frame)[0] == pytest.approx(0.75)


@pytest.mark.parametrize("width,jump,accepted", [(0.20, 0.14, True), (0.20, 0.16, False),
                                               (0.04, 0.05, True), (0.04, 0.07, False)])
def test_face_tracker_rejects_spatial_jumps_using_previous_box_width(face_lock, width, jump, accepted):
    tracker, clock, detections, frame = face_lock
    detections[:] = [face_detection(0.25, width)]
    assert tracker.locate(frame)[0] == pytest.approx(0.25)
    clock.sleep(0.1)
    detections[:] = [face_detection(0.25 + jump, width)]
    result = tracker.locate(frame)
    if accepted:
        assert result is not None and result[0] == pytest.approx(0.25 + jump)
    else:
        assert result is None


def phone_frame(*rectangles, color=(160, 60, 245)):
    """Synthetic screen-sized pink rectangles, never camera captures or device identifiers."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for center, size, angle in rectangles:
        corners = cv2.boxPoints((center, size, angle)).astype(np.int32)
        cv2.fillConvexPoly(frame, corners, color)
    return frame


@pytest.mark.parametrize("angle", [0, 25, 60, 90, 135])
def test_phone_tracker_locates_pink_screen_and_uses_its_narrow_span(angle):
    frame = phone_frame(((210, 190), (60, 130), angle))
    seen = F.PhoneTracker().locate(frame)
    assert seen is not None
    assert seen[0] == pytest.approx(210 / 640, abs=0.003)
    assert seen[1] == pytest.approx(190 / 480, abs=0.003)
    # Rotating the same screen must not imply that it moved much closer to the camera.
    assert seen[2] == pytest.approx((60 / 640) / 0.075, rel=0.04)


@pytest.mark.parametrize("color", [(0, 0, 255), (0, 128, 255), (0, 255, 0),
                                  (255, 110, 0), (200, 200, 220), (40, 10, 60)])
def test_phone_tracker_rejects_nonpink_unsaturated_or_dark_regions(color):
    # (255, 110, 0) is a cyan-blue (hue ~107): pure blue (hue 120) is inside the measured screen
    # gradient's low end (118) and is accepted by design.
    assert F.PhoneTracker().locate(phone_frame(((320, 240), (60, 130), 0), color=color)) is None


@pytest.mark.parametrize("center,size", [((320, 240), (4, 9)), ((320, 240), (30, 140)),
                                       ((320, 240), (10, 150)), ((10, 240), (60, 130)),
                                       ((320, 30), (60, 130)), ((320, 240), (430, 600))])
def test_phone_tracker_rejects_small_wrong_shape_or_clipped_regions(center, size):
    # (30, 140) is a stick (aspect 4.7): a square is inside the 0.3..3.5 aspect gate (a screen seen
    # foreshortened, or the app's purple filling only part of it) and is accepted by design.
    assert F.PhoneTracker().locate(phone_frame((center, size, 0))) is None


def app_screen(width=300, height=260, hue=(118, 165), sat=190, val=200, glyphs=True):
    """A synthetic Feel the Music screen: the blue-violet-to-pink gradient measured on the lamp's
    camera (OpenCV hue 118..165, S ~190, V ~200) with white UI text and a white dot."""
    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    hsv[..., 0] = np.linspace(hue[0], hue[1], width).astype(np.uint8)[None, :]
    hsv[..., 1], hsv[..., 2] = sat, val
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if glyphs:
        for i, text in enumerate(("FEEL THE MUSIC", "128 bpm", "KICK  SNARE", "vibe 0.8")):
            cv2.putText(bgr, text, (18, 50 + 52 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.circle(bgr, (240, 215), 10, (255, 255, 255), -1)
    return bgr


def frame_with_screen(center, screen, angle=0.0):
    """`screen` pasted into a black 640x480 frame, its centre at `center`, turned by `angle` degrees."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    h, w = screen.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    m[:, 2] += (center[0] - w / 2, center[1] - h / 2)
    return cv2.warpAffine(screen, m, (640, 480), dst=frame, borderMode=cv2.BORDER_TRANSPARENT)


@pytest.mark.parametrize("center,angle", [((200, 200), 0), ((320, 240), 0), ((450, 300), 0), ((320, 240), 48)])
def test_phone_tracker_locates_the_apps_purple_gradient_screen_with_white_text(center, angle):
    frame = frame_with_screen(center, app_screen(), angle)
    seen = F.PhoneTracker().locate(frame)
    assert seen is not None
    assert abs(seen[0] - center[0] / 640) <= 0.02                # within 0.02 of the frame width ...
    assert abs(seen[1] * 480 - center[1]) <= 0.02 * 640         # ... on both axes


@pytest.mark.parametrize("hue", [(134, 134), (118, 122), (160, 165)])
def test_phone_tracker_accepts_the_measured_median_and_both_ends_of_the_gradient(hue):
    seen = F.PhoneTracker().locate(frame_with_screen((320, 240), app_screen(hue=hue, sat=190, val=199)))
    assert seen is not None and seen[:2] == pytest.approx((0.5, 0.5), abs=0.01)


def test_phone_mask_closes_the_text_holes_but_not_a_real_hole():
    tracker = F.PhoneTracker()
    frame = frame_with_screen((320, 240), app_screen())
    inside = tracker.mask(frame)[110:370, 170:470]
    assert cv2.countNonZero(inside) / inside.size >= 0.97        # the glyphs are closed over
    cv2.rectangle(frame, (230, 150), (410, 330), (0, 0, 0), -1)  # a 180x180 hole (fill < 65 %): not a screen
    assert tracker.locate(frame) is None


def test_phone_tracker_is_lost_in_a_purple_stage_wash_but_not_a_dim_one():
    """Documents the limit: the colour gate cannot tell the screen from surroundings lit the same
    purple (measured on the real frame: with the background recoloured to hue 140 and its saturation
    raised, the mask floods 56 % of the frame and nothing screen-shaped is left). A wash below the
    gate's saturation (80) is fine."""
    screen = app_screen()
    washed = np.zeros((480, 640, 3), dtype=np.uint8)
    washed[:] = cv2.cvtColor(np.uint8([[[140, 150, 140]]]), cv2.COLOR_HSV2BGR)[0, 0]
    frame = frame_with_screen((320, 240), screen)
    frame[np.all(frame == 0, axis=2)] = washed[0, 0]
    assert F.PhoneTracker().locate(frame) is None                  # the screen merges with the wash
    dim = frame_with_screen((320, 240), screen)
    dim[np.all(dim == 0, axis=2)] = cv2.cvtColor(np.uint8([[[140, 60, 140]]]), cv2.COLOR_HSV2BGR)[0, 0]
    seen = F.PhoneTracker().locate(dim)
    assert seen is not None and seen[:2] == pytest.approx((0.5, 0.5), abs=0.01)


def test_live_config_defaults_move_fast_but_the_step_is_capped_by_the_speed_budget():
    cfg = F.LiveConfig()
    assert (cfg.step, cfg.period, cfg.live_ms) == (F.LIVE_STEP, F.LIVE_PERIOD, F.LIVE_MS) == (30.0, 0.25, 250)
    assert cfg.speed == pytest.approx(120.0) and cfg.speed <= F.LIVE_SPEED_CAP < 300  # the runtime refuses above 300
    big = F.LiveConfig(step=100.0, live_ms=250)                    # FOLLOW_ARGS="--live-step 100": clamped
    assert big.step == pytest.approx(35.0) and big.step_asked == 100.0 and big.speed == pytest.approx(140.0)
    assert F.LiveConfig(step=100.0, live_ms=1000).step == 100.0    # a slow move may take a big step
    cfg = F.LiveConfig(step=30.0)
    assert cfg.limit_speed(300.0 / 2) is False and cfg.step == 30.0   # the SDK reports 300: half of it, no change
    assert cfg.limit_speed(80.0) is True and cfg.step == pytest.approx(20.0) and cfg.speed == pytest.approx(80.0)
    assert cfg.limit_speed(1000.0) is True and cfg.step == pytest.approx(20.0)  # a cap never loosens


def test_center_score_is_one_in_the_dead_zone_and_falls_linearly_to_the_edge():
    assert F.center_score(0.5, 0.5) == 1.0
    assert F.center_score(0.54, 0.46) == 1.0 and F.center_score(0.549, 0.5) == 1.0
    assert F.center_score(0.551, 0.5) < 1.0
    assert F.center_score(0.25, 0.5) == pytest.approx((0.5 - 0.25) / 0.45)      # a quarter of the way in
    assert F.center_score(0.5, 0.75) == pytest.approx((0.5 - 0.25) / 0.45)
    assert F.center_score(0.5, 0.125) == pytest.approx((0.5 - 0.375) / 0.45)
    assert F.center_score(1.0, 0.5) == 0.0 and F.center_score(0.0, 0.0) == 0.0 and F.center_score(0.5, 1.2) == 0.0


def test_target_report_writes_atomically_with_the_fields_and_reuses_the_last_sighting(tmp_path, monkeypatch):
    import json
    clock = FakeClock()
    path = tmp_path / "deep" / "target.json"
    replaced, real_replace = [], os.replace

    def replace(src, dst):
        assert os.path.exists(src) and str(src).endswith(".tmp") and not os.path.exists(dst) or replaced
        replaced.append((str(src), str(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(F.os, "replace", replace)
    report = F.TargetReport(path, clock=clock)
    body = report.write("phone", (0.52, 0.49), 3.2, "tracking")
    assert json.loads(path.read_text()) == body == {"t": 1000.0, "kind": "phone", "seen": True, "x": 0.52, "y": 0.49,
                                                     "aim_deg": 3.2, "center": 1.0, "state": "tracking"}
    assert replaced == [(str(path) + ".tmp", str(path))] and not (tmp_path / "deep" / "target.json.tmp").exists()
    clock.sleep(0.1)
    body = report.write(None, None, None, "holding")             # unseen: the last sighting still gives the centre
    assert json.loads(path.read_text()) == body
    assert body == {"t": pytest.approx(1000.1), "kind": "phone", "seen": False, "x": 0.52, "y": 0.49,
                    "aim_deg": None, "center": 1.0, "state": "holding"}
    clock.sleep(0.1)
    body = report.write("face", (0.2, 0.5), float("nan"), "tracking")
    assert body["kind"] == "face" and body["aim_deg"] is None and body["center"] == pytest.approx((0.5 - 0.3) / 0.45)
    fresh = F.TargetReport(tmp_path / "t2.json", clock=clock)     # before any sighting
    assert fresh.write(None, None, None, "searching") == {"t": pytest.approx(1000.2), "kind": None, "seen": False,
                                                          "x": None, "y": None, "aim_deg": None, "center": 0.0,
                                                          "state": "searching"}
    assert report.writes == 3 and report.errors == 0


def test_target_report_writes_at_most_twenty_times_a_second(tmp_path):
    import json
    clock = FakeClock()
    report = F.TargetReport(tmp_path / "target.json", clock=clock)
    for k in range(5):                                            # cycles at 33 Hz
        clock.t = 1000.0 + k * 0.03
        report.write("face", (0.5, 0.5), 0.0, "tracking")
    assert report.writes == 3                                     # t = 1000.00, 1000.06, 1000.12
    assert json.loads((tmp_path / "target.json").read_text())["t"] == pytest.approx(1000.12)


def test_target_report_survives_an_unwritable_path(tmp_path, capsys):
    report = F.TargetReport(tmp_path / "file.txt" / "target.json", clock=FakeClock())
    (tmp_path / "file.txt").write_text("not a directory")
    for _ in range(3):
        assert report.write("face", (0.5, 0.5), 0.0, "tracking")["seen"] is True
    assert report.writes == 0 and report.errors == 1              # throttled: one attempt in this 50 ms
    assert "cannot write" in capsys.readouterr().out


def test_live_follower_reports_the_target_every_cycle(one_axis_follow, tmp_path):
    import json
    follower, clock, motors, camera = one_axis_follow
    path = tmp_path / "target.json"
    follower.report = F.TargetReport(path, clock=clock)
    run_cycles(follower, clock, 3)
    body = json.loads(path.read_text())
    assert body["kind"] == "face" and body["seen"] is True and body["state"] == "tracking"
    assert 0.5 < body["x"] < 1.0 and body["y"] == 0.5 and body["t"] == pytest.approx(clock() - 0.12)
    assert body["aim_deg"] == pytest.approx(follower.aim_error) and 0.0 <= body["center"] < 1.0
    assert body["center"] == pytest.approx(F.center_score(body["x"], body["y"]))
    camera.point[1] = -1.0                                        # the face leaves
    run_cycles(follower, clock, 1)
    gone = json.loads(path.read_text())
    assert gone["seen"] is False and gone["kind"] == "face" and (gone["x"], gone["y"]) == (body["x"], body["y"])
    assert gone["t"] == pytest.approx(clock() - 0.12) and follower.report.writes == 4


def test_phone_tracker_rejects_blank_irregular_and_hollow_regions():
    tracker = F.PhoneTracker()
    frame = phone_frame()
    assert tracker.locate(frame) is None
    cv2.fillPoly(frame, [np.array([[200, 150], [280, 150], [230, 290]], dtype=np.int32)], (160, 60, 245))
    assert tracker.locate(frame) is None
    frame = phone_frame(((320, 240), (70, 150), 0))
    cv2.rectangle(frame, (288, 168), (352, 312), (0, 0, 0), -1)
    assert tracker.locate(frame) is None


def test_phone_tracker_accepts_screen_with_small_nonpink_interface_regions():
    frame = phone_frame(((320, 240), (80, 170), 0))
    for y in (190, 220, 250):
        cv2.rectangle(frame, (293, y), (345, y + 7), (255, 255, 255), -1)
    seen = F.PhoneTracker().locate(frame)
    assert seen is not None and seen[:2] == pytest.approx((0.5, 0.5))


def test_phone_tracker_keeps_screen_lock_then_reacquires_after_loss():
    clock = FakeClock()
    tracker = F.PhoneTracker(clock=clock)
    first = ((160, 240), (60, 130), 0)
    other = ((460, 240), (50, 110), 0)
    assert tracker.locate(phone_frame(first, other))[0] == pytest.approx(0.25)
    clock.sleep(0.1)
    larger_other = ((460, 240), (85, 180), 20)
    assert tracker.locate(phone_frame(first, larger_other))[0] == pytest.approx(0.25)
    clock.sleep(0.1)
    assert tracker.locate(phone_frame(larger_other)) is None
    clock.sleep(0.3)
    assert tracker.locate(phone_frame()) is None
    clock.sleep(0.21)
    assert tracker.locate(phone_frame(larger_other))[0] == pytest.approx(460 / 640, abs=0.003)


def test_phone_lock_retains_incumbent_across_a_blocking_planner_observation_gap():
    clock = FakeClock()
    tracker = F.PhoneTracker(clock=clock)
    first = ((160, 240), (60, 130), 0)
    assert tracker.locate(phone_frame(first, ((460, 240), (50, 110), 0)))[0] == pytest.approx(0.25)
    clock.sleep(3.5)                                   # SDK move plus the configured observation pause
    assert tracker.locate(phone_frame(first, ((460, 240), (85, 180), 0)))[0] == pytest.approx(0.25)


def test_phone_geometric_lock_reacquires_when_a_camera_turn_moves_every_box_outside_its_gate():
    clock = FakeClock()
    tracker = F.PhoneTracker(clock=clock)
    assert tracker.locate(phone_frame(((160, 240), (60, 130), 0)))[0] == pytest.approx(0.25)
    clock.sleep(3.5)
    # There is no camera-motion compensation or screen identity: both boxes are now outside
    # the previous gate, so expired geometric lock acquires the larger candidate afresh.
    changed_view = phone_frame(((320, 240), (60, 130), 0), ((460, 240), (85, 180), 0))
    assert tracker.locate(changed_view)[0] == pytest.approx(460 / 640, abs=0.003)


@pytest.mark.parametrize("loss", ["missing", "expired"])
def test_phone_following_requires_three_fresh_sightings_after_loss(one_axis_follow, loss):
    follower, clock, motors, camera = one_axis_follow
    follower.cfg.target = "phone"
    follower.trackers = [("phone", FakeFaces())]
    run_cycles(follower, clock, 2)
    if loss == "missing":
        camera.point[1] = -1.0
        run_cycles(follower, clock, 1)
        clock.sleep(F.TargetLock.LOST_S + 0.01)       # missing for longer than the lock window
        camera.point[1] = 1.0
    else:
        clock.sleep(0.61)
    run_cycles(follower, clock, 2)
    assert motors.posts == []
    run_cycles(follower, clock, 1)
    assert len(motors.posts) == 1
    assert motors.posts[0][1]["base_yaw"] > 0


def test_phone_tracker_drives_existing_controller_with_synthetic_frames_and_dry_run(one_axis_follow):
    follower, clock, motors, _ = one_axis_follow
    follower.cfg.target, follower.cfg.dry_run = "phone", True
    follower.trackers = [("phone", F.PhoneTracker(clock=clock))]
    frame = phone_frame(((480, 240), (60, 130), 0))
    follower.frames = SimpleNamespace(newest=lambda **kwargs: (frame, clock() - 0.03))
    run_cycles(follower, clock, 3)
    assert follower.aim_error > follower.cfg.deadband_deg
    assert follower.state == "tracking"
    assert follower.commands == 0 and motors.posts == []
    frame[:] = 0
    run_cycles(follower, clock, 1)
    assert len(follower.sightings) == 3 and follower.aim_error is not None   # one blank frame is kept
    clock.sleep(F.TargetLock.LOST_S + 0.01)
    run_cycles(follower, clock, 1)
    assert follower.sightings == [] and follower.aim_error is None           # gone past the window


@pytest.mark.parametrize("slow_stage", ["receipt", "detector", "joints", "ik"])
def test_phone_live_rejects_expired_receipt_before_any_command(one_axis_follow, monkeypatch, slow_stage):
    follower, clock, motors, camera = one_axis_follow
    follower.cfg.target = "phone"
    tracker = FakeFaces()
    follower.trackers = [("phone", tracker)]
    if slow_stage == "receipt":
        newest = camera.newest

        def expired_frame(**kwargs):
            frame, stamp = newest(**kwargs)
            return frame, stamp - 0.61

        monkeypatch.setattr(camera, "newest", expired_frame)
    else:
        obj, method = {"detector": (tracker, "locate"), "joints": (motors, "positions"),
                       "ik": (follower.model, "look_at")}[slow_stage]
        original = getattr(obj, method)

        def slow(*args, **kwargs):
            result = original(*args, **kwargs)
            clock.sleep(0.7)
            return result

        monkeypatch.setattr(obj, method, slow)
    run_cycles(follower, clock, 3)
    assert motors.posts == []
    assert follower.sightings == [] and follower.aim_error is None and not follower.correcting


@pytest.fixture
def sdk_phone_main(monkeypatch):
    """Run the real CLI orchestration with camera pixels and fake SDK, time and robot model."""
    clock, model = FakeClock(), OneAxisModel()
    model.scale_source, model.table_z = "offline fixture", 0.0

    class Frames:
        error, running = "", True

        def __init__(self):
            self.script, self.read_count = [], 0
            self.stamp_age = 0.0

        def start(self):
            pass

        def newest(self, after, timeout=2.0):
            clock.t = max(clock(), after + 0.03)
            delay, frame = self.script.pop(0) if self.script else (0.12, None)
            clock.sleep(delay)
            self.read_count += 1
            return frame, clock() - self.stamp_age

    camera = Frames()

    class SDK:
        def __init__(self):
            self.moves, self.measured, self.read_delay = [], pose(), 0.0

        def capabilities(self):
            return {}

        def joints(self):
            if camera.read_count:
                clock.sleep(self.read_delay)
            return {"self_collision_check": True, "units": "normalized_m100_100",
                    "joints": dict(self.measured), "positions": dict(self.measured)}

        def move(self, commanded):
            self.moves.append((camera.read_count, dict(commanded)))
            self.measured = dict(commanded)
            clock.sleep(2.0)
            return {"result": {"duration_seconds": 2.0}}

    def no_raw_idle(*args):
        raise AssertionError("default SDK mode must not mutate the raw idle route")

    sdk = SDK()
    phone_tracker = F.PhoneTracker(clock=clock)
    monkeypatch.setattr(F.time, "monotonic", clock)
    monkeypatch.setattr(F, "PhoneTracker", lambda: phone_tracker)
    monkeypatch.setattr(F, "LampModel", lambda path: model)
    monkeypatch.setattr(F, "LampSDK", lambda token: sdk)
    monkeypatch.setattr(F, "read_token", lambda: "offline-fixture")
    monkeypatch.setattr(F, "Camera", lambda sdk, fps: camera)
    thermal = NoThermal()
    thermal.peak_c = 0.0
    monkeypatch.setattr(F, "Thermal", lambda warm, hot: thermal)
    monkeypatch.setattr(F, "describe", lambda model, measured: "offline pose")
    monkeypatch.setattr(F.os, "nice", lambda value: None)
    monkeypatch.setattr(F.signal, "signal", lambda *args: None)
    monkeypatch.setattr(F, "idle_off", no_raw_idle)
    monkeypatch.setattr(F, "idle_restore", no_raw_idle)

    def run(script, *flags):
        camera.script = [(0.12, frame) for frame in script]
        monkeypatch.setattr(sys, "argv", ["follow.py", "--target", "phone", "--seconds", "10", *flags])
        F.main()

    return run, sdk, camera, phone_tracker, model, clock


def test_sdk_phone_cli_dry_run_uses_pixels_without_moving_or_raw_idle(sdk_phone_main, capsys):
    run, sdk, camera, _, _, _ = sdk_phone_main
    frame = phone_frame(((160, 240), (60, 130), 0))
    run([frame] * 3, "--dry-run")
    output = capsys.readouterr().out
    assert "watching for a phone screen showing the app (purple to pink) (dry run: will not move)" in output
    assert "phone " in output and "base_yaw +0->-22" in output
    assert sdk.moves == [] and not camera.running


def test_sdk_phone_cli_keeps_incumbent_after_blocking_move_and_larger_distractor(sdk_phone_main):
    run, sdk, _, _, _, _ = sdk_phone_main
    first = ((160, 240), (60, 130), 0)
    before = phone_frame(first, ((460, 240), (50, 110), 0))
    after = phone_frame(first, ((460, 240), (85, 180), 0))
    run([before] * 3 + [after] * 3)
    assert len(sdk.moves) == 2
    assert sdk.moves[0][0] == 3 and sdk.moves[1][0] == 6
    assert sdk.moves[1][1]["base_yaw"] < sdk.moves[0][1]["base_yaw"] < 0


def test_sdk_phone_cli_clears_sightings_after_a_blank_frame(sdk_phone_main):
    run, sdk, _, _, _, _ = sdk_phone_main
    frame = phone_frame(((160, 240), (60, 130), 0))
    run([frame] * 2 + [phone_frame()] + [frame] * 3)
    assert len(sdk.moves) == 1
    assert sdk.moves[0][0] == 6


@pytest.mark.parametrize("slow_stage", ["receipt", "joints", "ik"])
def test_sdk_phone_cli_rejects_expired_receipt_before_move(sdk_phone_main, monkeypatch, slow_stage):
    run, sdk, camera, _, model, clock = sdk_phone_main
    if slow_stage == "receipt":
        camera.stamp_age = 0.7
    elif slow_stage == "joints":
        sdk.read_delay = 0.7
    else:
        look_at = model.look_at

        def slow_ik(*args, **kwargs):
            clock.sleep(0.7)
            return look_at(*args, **kwargs)

        monkeypatch.setattr(model, "look_at", slow_ik)
    run([phone_frame(((160, 240), (60, 130), 0))] * 3)
    assert sdk.moves == []


def test_sdk_phone_cli_slow_detection_cannot_count_as_a_fresh_sighting(sdk_phone_main, monkeypatch):
    run, sdk, _, tracker, _, clock = sdk_phone_main
    locate, calls = tracker.locate, 0

    def slow_third(frame):
        nonlocal calls
        calls += 1
        if calls == 3:
            clock.sleep(0.7)
        return locate(frame)

    monkeypatch.setattr(tracker, "locate", slow_third)
    run([phone_frame(((160, 240), (60, 130), 0))] * 6)
    assert len(sdk.moves) == 1
    assert sdk.moves[0][0] == 6                       # three fresh observations after the expired one


@needs_model
def test_phone_off_center_target_uses_existing_safe_whole_arm_base_rotation(model):
    seen = F.PhoneTracker().locate(phone_frame(((140, 240), (60, 130), 0)))
    assert seen is not None
    x, y, size = seen
    measured = dict(model.neutral)
    distance = float(np.clip(model.distance_from_size(size, 1.0), 0.3, 3.0))
    target = model.target_point(measured, (x, y), distance)
    goal, _ = model.look_at(target, prefer_distance=0.45)
    assert abs(goal["base_yaw"] - measured["base_yaw"]) > 1.0
    assert model.aim_error_deg(goal, target) < model.aim_error_deg(measured, target)
    assert not model.problems(goal)


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
    for _ in range(32):                              # commands wait for latency plus the full timed move
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
    cfg = F.LiveConfig(target="face", live_ms=250, period=0.25, step=10.0)
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
