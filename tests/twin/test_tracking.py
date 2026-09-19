"""HeadTracker against small fakes: an analytic pan-tilt head, a goal-level stand-in for the SDK and a pinhole
face detector. Nothing here imports model.py or motion.py; the fakes keep only what the tracker relies on
(every move lasts 2 s after a 0.18 s pre-roll, a new move pre-empts the old one, which ends "canceled";
a clip starts with a 2 s entry move; detections arrive 0.15 s after capture at 10 Hz).
"""
import math
from collections import deque

import numpy as np
import pytest

from twin.contract import JOINTS, Check, Detection, Pose
from twin.tracking import ACQUIRE, HOLD, LOCK, LOST, SEARCH, HeadTracker, TrackerConfig

YAW_DEG, PITCH_DEG, ROLL_DEG = 0.74, 0.6, 0.5      # degrees per unit of the fake head (yaw about the real one)


# ------------------------------------------------------------------ fakes
class PanTilt:
    """A camera on a pan-tilt head: base_yaw turns it (+ = toward +x), wrist_pitch tilts it (+ = down, as on the
    lamp), wrist_roll rolls the picture. The camera sits 8 cm in front of a pivot 0.30 m above the table."""
    PIVOT = np.array([0.0, 0.0, 0.30])

    def __init__(self, forbidden=None, honest=True):
        self.neutral = {j: 0.0 for j in JOINTS}
        self.limits = {j: (-94.0, 94.0) for j in JOINTS}
        self.forbidden = forbidden          # units -> True for poses check() must refuse
        self.honest = honest                # False: look_at does not check its answer (the tracker must)
        self.checks = 0

    def head(self, u):
        az, el = math.radians(u["base_yaw"] * YAW_DEG), math.radians(-u["wrist_pitch"] * PITCH_DEG)
        f = np.array([math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)])
        r0 = np.array([math.cos(az), -math.sin(az), 0.0])
        d0 = np.cross(f, r0)
        roll = math.radians(u["wrist_roll"] * ROLL_DEG)
        r = r0 * math.cos(roll) + d0 * math.sin(roll)
        return Pose(position=self.PIVOT + 0.08 * f, forward=f, down=np.cross(f, r), right=r)

    def check(self, u):
        self.checks += 1
        reasons = [f"{j} out of range" for j in JOINTS if not self.limits[j][0] <= u[j] <= self.limits[j][1]]
        h = self.head(u)
        if math.degrees(math.asin(h.forward[2])) < -35.0:
            reasons.append("faces the table")
        if self.forbidden is not None and self.forbidden(u):
            reasons.append("forbidden")
        return Check(ok=not reasons, reasons=reasons, min_table_m=float(h.position[2]) - 0.05,
                     min_base_m=0.1, min_self_m=0.1)

    def look_at(self, point, seed=None, **kw):
        u = dict(self.neutral)
        for _ in range(4):                  # the camera moves with the aim: iterate
            to = np.asarray(point, float) - self.head(u).position
            u["base_yaw"] = float(np.clip(math.degrees(math.atan2(to[0], to[1])) / YAW_DEG, -94, 94))
            u["wrist_pitch"] = float(np.clip(-math.degrees(math.atan2(to[2], math.hypot(to[0], to[1]))) / PITCH_DEG,
                                             -94, 94))
        h = self.head(u)
        to = np.asarray(point, float) - h.position
        err = math.degrees(math.atan2(np.linalg.norm(np.cross(h.forward, to)), h.forward @ to))
        if self.honest and not self.check(u).ok:
            u = dict(seed) if seed is not None and self.check(seed).ok else dict(self.neutral)
        return u, {"aim_error_deg": err}


def vec(u):
    return np.array([u[j] for j in JOINTS], dtype=float)


def ease(x):
    x = min(1.0, max(0.0, x))
    return x ** 3 * (x * (6 * x - 15) + 10)


class FakeLamp:
    """Goal-level SDK stand-in. fail=True ends every action "failed". idle=True drifts the head down toward
    a crouch after a success, as the vendor idle does. sag (units per joint) holds the arm that far from its
    goal, and every plan starts from that sagged pose, as the SDK's re-plan from the measured pose does."""
    CROUCH = {"base_yaw": 0.0, "base_pitch": 0.0, "elbow_pitch": 0.0, "wrist_roll": 0.0, "wrist_pitch": 50.0}

    def __init__(self, start=None, *, fail=False, idle=False, sag=None):
        self.pose = vec(start or {j: 0.0 for j in JOINTS})
        self.plan, self.fail, self.idle, self.idle_on = None, fail, idle, idle
        self.sag = vec({j: 0.0 for j in JOINTS} | (sag or {}))
        self.outcomes, self.t = deque(), 0.0
        self.history = []                   # (t, pose) at every advance, for scoring

    @property
    def busy(self):
        return self.plan is not None

    def units(self):
        return {j: float(v) for j, v in zip(JOINTS, self.pose, strict=True)}

    def submit(self, t, cmd):
        self.advance(t)
        if self.plan is not None:
            self.outcomes.append("canceled")
        start = self.pose.copy()
        t0 = t + 0.18
        if cmd.kind == "move":
            goal = vec(cmd.target)
            dur = max(2.0, float(np.max(np.abs(goal - start))) / 72.0)
            self.plan = {"t0": t0, "start": start, "goal": goal, "dur": dur, "end": t0 + dur + 0.1, "clip": None}
        else:
            times = np.array([f[0] for f in cmd.frames]) - cmd.frames[0][0]
            poses = np.array([vec(f[1]) for f in cmd.frames])
            self.plan = {"t0": t0, "start": start, "goal": poses[0], "dur": 2.0, "clip": (times, poses),
                         "end": t0 + 2.0 + times[-1]}
        self.idle_on = False

    def advance(self, t):
        dt, self.t = t - self.t, t
        p = self.plan
        if p is not None:
            if t >= p["t0"]:
                u = (t - p["t0"]) / p["dur"]
                if u <= 1.0 or p["clip"] is None:
                    self.pose = p["start"] + (p["goal"] - p["start"]) * ease(u) + self.sag
                else:
                    times, poses = p["clip"]
                    s = t - p["t0"] - p["dur"]
                    self.pose = np.array([np.interp(s, times, poses[:, k]) for k in range(len(JOINTS))]) + self.sag
            if t >= p["end"]:
                self.outcomes.append("failed" if self.fail else "succeeded")
                self.plan = None
                self.idle_on = self.idle and not self.fail
        elif self.idle_on and dt > 0:
            crouch = vec(self.CROUCH)
            step = np.clip(crouch - self.pose, -15.0 * dt, 15.0 * dt)       # 15 units/s toward the crouch
            self.pose = self.pose + step
        self.history.append((t, self.units()))


class FakeCamera:
    """10 Hz pinhole face detector, 0.15 s latency, face width 0.15 m (the tracker's own assumption)."""
    FX = 0.5 / math.tan(math.radians(61.0) / 2)
    FY = 0.5 / math.tan(math.radians(44.0) / 2)

    def __init__(self, kin, noise=0.0, seed=1, relabel=None):
        self.kin, self.noise, self.relabel = kin, noise, relabel
        self.rng = np.random.default_rng(seed)
        self.queue, self.next_frame = [], 0.0

    def capture(self, t, units, people):
        if t + 1e-9 < self.next_frame:
            return
        self.next_frame = round(self.next_frame + 0.1, 6)
        h = self.kin.head(units)
        for pid, head in people:
            to = np.asarray(head, float) - h.position
            depth = float(to @ h.forward)
            if depth <= 0.05 or np.linalg.norm(to) > 2.5:
                continue
            x = 0.5 + self.FX * float(to @ h.right) / depth + self.noise * self.rng.standard_normal()
            y = 0.5 + self.FY * float(to @ h.down) / depth + self.noise * self.rng.standard_normal()
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                continue
            size = 0.15 * self.FX / depth * (1.0 + 5 * self.noise * self.rng.standard_normal())
            label = pid if self.relabel is None else self.relabel(pid)
            self.queue.append(Detection(t_capture=t, t_delivered=round(t + 0.15, 6), person_id=label,
                                        x=x, y=y, size=size, confidence=0.95))

    def poll(self, t):
        out = [d for d in self.queue if d.t_delivered <= t + 1e-9]
        self.queue = [d for d in self.queue if d.t_delivered > t + 1e-9]
        return out


def run(tracker, people, seconds, *, lamp=None, camera=None, dt=0.02):
    """The run loop: lamp, camera, tracker. people(t) -> [(id, head xyz)]."""
    lamp = lamp or FakeLamp()
    camera = camera or FakeCamera(tracker.kin)
    log, states = [], []
    t = 0.0
    while t <= seconds + 1e-9:
        lamp.advance(t)
        camera.capture(t, lamp.units(), people(t))
        outcome = lamp.outcomes.popleft() if lamp.outcomes else None
        for cmd in tracker.update(t, camera.poll(t), lamp.units(), lamp.busy, outcome):
            lamp.submit(t, cmd)
            log.append((t, cmd))
        states.append((t, tracker.state, tracker.locked_id))
        t = round(t + dt, 6)
    return log, states, lamp


def aim_error(kin, units, point):
    h = kin.head(units)
    to = np.asarray(point, float) - h.position
    return math.degrees(math.atan2(np.linalg.norm(np.cross(h.forward, to)), h.forward @ to))


def gaze(kin, units):
    f = kin.head(units).forward
    return math.degrees(math.atan2(f[0], f[1])), math.degrees(math.asin(f[2]))


def nobody(t):
    return []


def seated(x=0.3, y=0.85, z=0.46):
    return lambda t: [("p1", (x, y, z))]


def walker(t, period=8.0):
    """Walks side to side in front of the lamp, 1 m away."""
    return [("p1", (0.45 * math.sin(2 * math.pi * t / period), 1.0, 0.46))]


def first_time(states, state):
    return next((t for t, s, _ in states if s == state), None)


def max_in_window(times, window=60.0):
    times = sorted(times)
    best, j = 0, 0
    for i, t in enumerate(times):
        while times[j] <= t - window:
            j += 1
        best = max(best, i - j + 1)
    return best


def all_commanded_poses(log):
    for _, cmd in log:
        if cmd.kind == "move":
            yield cmd.target
        else:
            for _, pose in cmd.frames:
                yield pose


# ------------------------------------------------------------------ SEARCH
def test_search_moves_are_varied_allowed_and_never_a_sweep():
    kin = PanTilt()
    tracker = HeadTracker(kin, config=TrackerConfig(search_via="moves", seed=3))
    log, states, _ = run(tracker, nobody, 90.0)
    assert all(s == SEARCH for _, s, _ in states)
    looks = [cmd.target for _, cmd in log if cmd.why.startswith("search: look")]
    assert len(looks) >= 15
    views = [gaze(kin, u) for u in looks]
    azimuths = [round(az) for az, _ in views]
    elevations = [el for _, el in views]
    assert len(set(azimuths)) >= 8                                   # varied places
    assert max(azimuths) - min(azimuths) >= 50                       # both sides of the room
    assert max(elevations) - min(elevations) >= 10                   # seated and standing heights
    assert all(kin.check(u).ok for u in all_commanded_poses(log))
    assert all(-30.0 <= el and abs(az) <= 80.0 for az, el in views)  # never at the table, never behind
    # never a mechanical sweep: at most 2 jumps in a row the same way
    # (a clear jump is >= 8 deg sideways; anything smaller breaks a run)
    signs = [np.sign(b - a) if abs(b - a) >= 8.0 else 0 for (a, _), (b, _) in zip(views, views[1:], strict=False)]
    run_len, longest = 0, 0
    for i, s in enumerate(signs):
        run_len = run_len + 1 if s and i and s == signs[i - 1] else (1 if s else 0)
        longest = max(longest, run_len)
    assert longest <= 2
    # consecutive places are a gentle jump apart (largest step 45 deg plus jitter)
    steps = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(views, views[1:], strict=False)]
    assert max(steps) <= 52.0
    assert any(cmd.why.startswith("search: a small glance") for _, cmd in log)


def test_search_clip_has_pauses_glances_and_varied_places():
    kin = PanTilt()
    tracker = HeadTracker(kin, config=TrackerConfig(seed=5))
    log, _, _ = run(tracker, nobody, 5.0)
    # first a quick motion.move takes the arm from the vendor idle, then the look-around clip follows it
    assert log and log[0][1].kind == "move" and log[0][1].why.startswith("search: take the arm")
    assert log[1][1].kind == "clip" and log[1][0] - log[0][0] == pytest.approx(0.3, abs=0.03)
    assert vec(log[1][1].frames[0][1]) == pytest.approx(vec(log[0][1].target))      # it starts where the move went
    frames = log[1][1].frames
    times = np.array([f[0] for f in frames])
    poses = np.array([vec(f[1]) for f in frames])
    assert times[0] == 0.0 and np.all(np.diff(times) > 0)
    assert 18.0 <= times[-1] <= 600.0 and len(frames) <= 18000
    assert all(kin.check(u).ok for _, u in frames)
    speed = np.max(np.abs(np.diff(poses, axis=0)), axis=1) / np.diff(times)
    assert speed.max() <= 60.0 * 1.001                              # calm saccades
    # pauses: runs of still frames, of varied length
    still = speed < 1e-6
    holds, run_len = [], 0
    for s in still:
        if s:
            run_len += 1
        elif run_len:
            holds.append(run_len)
            run_len = 0
    long_holds = [h / 30.0 for h in holds if h / 30.0 >= 0.5]
    assert len(long_holds) >= 6 and np.std(long_holds) > 0.1
    places = {tuple(np.round(p, 0)) for p, s in zip(poses[1:], still, strict=True) if s}
    assert len(places) >= 5
    assert tracker.stats["search_clip"] == 1 and tracker.stats["search_move"] == 1


# ------------------------------------------------------------------ ACQUIRE and LOCK
@pytest.mark.parametrize("strategy", ["settle", "preempt", "clip"])
def test_steady_person_is_acquired_and_locked(strategy):
    kin = PanTilt()
    person = (-0.55, 0.75, 0.46)                                     # about 36 deg to the left
    tracker = HeadTracker(kin, strategy=strategy)
    log, states, lamp = run(tracker, seated(*person), 25.0, camera=FakeCamera(kin, noise=0.004))
    t_lock = first_time(states, LOCK)
    assert first_time(states, ACQUIRE) is not None and t_lock is not None and t_lock < 12.0
    assert states[-1][1] == LOCK
    assert tracker.locked_id.startswith("track-") and tracker.locked_id != "p1"
    assert aim_error(kin, lamp.units(), person) < 6.0
    assert tracker.trace["scoring_person_id"] == "p1"               # passed through for scoring only
    assert all(set(u) == set(JOINTS) and all(math.isfinite(v) for v in u.values())
               for u in all_commanded_poses(log))
    assert all(kin.check(u).ok for u in all_commanded_poses(log))


def test_big_turn_becomes_several_calm_steps():
    kin = PanTilt()
    config = TrackerConfig(search_via="moves", alive=False, max_step_deg=8.0, max_step_units=12.0)
    tracker = HeadTracker(kin, config=config)
    person = (-0.03, 0.9, 0.46)                                       # straight ahead
    start = {**{j: 0.0 for j in JOINTS}, "base_yaw": -40.0}           # looking about 30 deg left: at the edge
    log, states, lamp = run(tracker, seated(*person), 25.0, lamp=FakeLamp(start))
    turns = [cmd.target for _, cmd in log if cmd.why.startswith(("acquire", "track: turn"))]
    assert len(turns) >= 3
    for a, b in zip(turns, turns[1:], strict=False):
        assert max(abs(a[j] - b[j]) for j in JOINTS) <= 12.0 + 1e-6
        assert aim_error(kin, a, kin.head(b).position + kin.head(b).forward) <= 8.0 + 0.5
    assert aim_error(kin, lamp.units(), person) < 6.0


def test_second_person_does_not_steal_the_lock():
    kin = PanTilt()
    p1, p2 = (0.0, 0.9, 0.46), (0.3, 0.7, 0.50)                      # p2 is nearer (a bigger face), in view

    def people(t):
        out = [("p1", p1)]
        if t >= 8.0:
            out.append(("p2", p2))
        return out

    tracker = HeadTracker(kin)
    log, states, lamp = run(tracker, people, 30.0, camera=FakeCamera(kin, noise=0.004))
    t_lock = first_time(states, LOCK)
    assert t_lock is not None and t_lock < 8.0
    locked = {lid for t, s, lid in states if t >= t_lock}
    assert len(locked) == 1                                          # one track, the same from lock to end
    assert all(s in (LOCK, ACQUIRE) for t, s, _ in states if t >= t_lock)
    assert aim_error(kin, lamp.units(), p1) < 6.0 and aim_error(kin, lamp.units(), p2) > 15.0
    assert tracker.stats["tracks_created"] >= 2                       # p2 was seen, and ignored


def test_second_person_is_taken_only_after_the_first_is_lost():
    kin = PanTilt()
    p1, p2 = (-0.1, 0.9, 0.46), (0.3, 0.85, 0.50)                    # both in the picture at once

    def people(t):
        return ([("p1", p1)] if t < 12.0 else []) + ([("p2", p2)] if t >= 6.0 else [])

    tracker = HeadTracker(kin)
    _, states, lamp = run(tracker, people, 35.0)
    first = next(lid for t, s, lid in states if s == LOCK)
    assert all(lid == first for t, s, lid in states if 6.0 <= t < 12.0)
    t_lost = next(t for t, s, _ in states if t >= 12.0 and s == LOST)
    assert 11.9 + 1.5 <= t_lost <= 11.9 + 1.5 + 0.1                 # last capture 11.9 s, then 1.5 s
    later = [lid for t, s, lid in states if s == LOCK and t > t_lost]
    assert later and later[-1] != first
    assert aim_error(kin, lamp.units(), p2) < 6.0


def test_person_id_is_never_used():
    kin = PanTilt()
    rng = np.random.default_rng(7)

    def people(t):
        return [("p1", (0.2, 0.85, 0.46))] + ([("p2", (-0.4, 0.7, 0.9))] if t > 6.0 else [])

    plain, _, _ = run(HeadTracker(kin), people, 20.0, camera=FakeCamera(kin, noise=0.004, seed=2))
    scrambled, _, _ = run(HeadTracker(kin), people, 20.0,
                          camera=FakeCamera(kin, noise=0.004, seed=2, relabel=lambda pid: f"x{rng.integers(1e6)}"))
    assert [(t, c.kind, c.target, c.why) for t, c in plain] == [(t, c.kind, c.target, c.why) for t, c in scrambled]


# ------------------------------------------------------------------ LOST
def test_loss_goes_lost_then_peeks_then_searches():
    kin = PanTilt()

    def people(t):                                                    # still, walks right at 0.15 m/s, leaves
        return [("p1", (0.15 * max(t - 8.0, 0.0), 0.9, 0.46))] if t < 12.0 else []

    tracker = HeadTracker(kin)
    log, states, _ = run(tracker, people, 24.0)
    t_lost, t_search = None, None
    for t, s, _ in states:
        if t >= 12.0 and s == LOST and t_lost is None:
            t_lost = t
        if t_lost is not None and s == SEARCH:
            t_search = t
            break
    assert t_lost is not None and t_search is not None
    assert 3.9 <= t_search - t_lost <= 4.2
    peeks = [(t, c) for t, c in log if c.why.startswith("lost: peek")]
    assert len(peeks) == 1 and t_lost <= peeks[0][0] < t_search
    last_seen = np.array(people(11.9)[0][1])
    peek_az, _ = gaze(kin, peeks[0][1].target)
    seen_az = math.degrees(math.atan2(last_seen[0], last_seen[1]))
    assert peek_az > seen_az + 2.0                                    # it looks where they were heading
    assert any(c.why.startswith("search") for t, c in log if t >= t_search)


def test_a_hold_is_pre_empted_when_the_person_moves_a_lot():
    kin = PanTilt()

    def people(t):                                                    # seated, then stands up in 1 s at 10 s
        rise = min(max(t - 10.0, 0.0), 1.0)
        return [("p1", (0.1, 0.85 + 0.2 * rise, 0.46 + 0.35 * rise))]

    tracker = HeadTracker(kin, config=TrackerConfig(alive=False))
    log, states, lamp = run(tracker, people, 20.0)
    urgent = [(t, c) for t, c in log if "pre-empts a hold" in c.why]
    assert urgent and 10.0 < urgent[0][0] < 11.5                      # it did not wait for the hold to finish
    assert tracker.stats["urgent"] >= 1 and tracker.stats["failures"] == 0
    assert states[-1][1] == LOCK and aim_error(kin, lamp.units(), people(20.0)[0][1]) < 6.0
    off = HeadTracker(kin, config=TrackerConfig(alive=False, urgent_deg=None))
    log_off, _, _ = run(off, people, 20.0)
    assert not [c for _, c in log_off if "pre-empts a hold" in c.why]


def test_load_error_is_learnt_and_aimed_out():
    kin = PanTilt()
    person = (0.15, 0.9, 0.46)
    sag = {"wrist_pitch": 6.0}                                        # the camera holds 3.6 deg low
    results = {}
    for comp in (True, False):
        tracker = HeadTracker(kin, config=TrackerConfig(alive=False, sag_compensation=comp))
        log, _, lamp = run(tracker, seated(*person), 30.0, lamp=FakeLamp(sag=sag))
        # judged at each move's end, where the arm has settled (every re-plan also bobs by the sag)
        ends = [u for t, u in lamp.history if any(abs(t - (tc + 2.28)) < 0.011 for tc, _ in log if tc > 12.0)]
        results[comp] = (float(np.mean([aim_error(kin, u, person) for u in ends])), tracker.trace["aim_bias_deg"])
    assert results[False][0] > 3.0
    assert results[True][0] < 1.5
    assert results[True][1][1] == pytest.approx(6.0 * PITCH_DEG, abs=0.6)   # it learnt the sag, upward
    assert results[False][1] == [0.0, 0.0]


def test_brief_look_away_keeps_the_lock():
    kin = PanTilt()

    def people(t):
        return [] if 10.0 <= t < 11.0 else [("p1", (0.2, 0.85, 0.46))]

    tracker = HeadTracker(kin)
    _, states, _ = run(tracker, people, 20.0)
    t_lock = first_time(states, LOCK)
    assert all(s == LOCK for t, s, _ in states if t >= max(t_lock, 9.0))
    assert len({lid for t, s, lid in states if t >= t_lock}) == 1


# ------------------------------------------------------------------ budget, failures, strategies
def test_commands_per_minute_stay_within_budget():
    kin = PanTilt()
    for strategy, budget in (("preempt", 20), ("settle", 20), ("clip", 12)):
        tracker = HeadTracker(kin, strategy=strategy, config=TrackerConfig(max_commands_per_min=budget))
        log, _, _ = run(tracker, walker, 130.0)
        assert max_in_window([t for t, _ in log]) <= budget, strategy
        assert tracker.window_used(130.0) <= budget
        assert tracker.stats["steps_blocked_budget"] > 0, strategy     # the budget did bind
    with pytest.raises(ValueError):
        TrackerConfig(max_commands_per_min=120, sdk_rate_limit_per_min=120)


def test_failures_back_off_then_hold():
    kin = PanTilt()
    tracker = HeadTracker(kin, config=TrackerConfig(search_via="moves"))
    log, states, _ = run(tracker, nobody, 40.0, lamp=FakeLamp(fail=True))
    assert states[-1][1] == HOLD and tracker.locked_id is None
    assert len(log) == 3
    gaps = np.diff([t for t, _ in log])
    assert np.all(gaps >= 5.0 + 2.0)                                  # a 2 s move, then a 5 s back-off
    t_hold = first_time(states, HOLD)
    assert not [t for t, _ in log if t > t_hold]
    tracker.reset(40.0)
    assert tracker.state == SEARCH


def test_own_preemptions_are_not_failures():
    kin = PanTilt()
    tracker = HeadTracker(kin, strategy="preempt")
    _, states, _ = run(tracker, walker, 60.0)
    assert tracker.stats["expected_cancels"] >= 5
    assert tracker.stats["failures"] == 0 and all(s != HOLD for _, s, _ in states)


def test_an_unrequested_cancel_backs_off():
    kin = PanTilt()
    tracker = HeadTracker(kin, config=TrackerConfig(search_via="moves"))
    measured = {j: 0.0 for j in JOINTS}
    first = tracker.update(0.0, [], measured, False, None)
    assert first and first[0].kind == "move"
    assert tracker.update(1.0, [], measured, False, "canceled") == []   # the runtime took over
    assert tracker.trace["blocked"] == "backoff"
    assert tracker.update(4.0, [], measured, False, None) == []
    assert tracker.update(6.1, [], measured, False, None)               # 5 s later it tries again
    assert tracker.update(6.5, [], measured, True, "429 rate_limited") == []
    assert tracker.update(60.0, [], measured, False, None) == []        # a rate refusal pauses 60 s
    assert tracker.update(66.6, [], measured, False, None)


def test_preempt_issues_more_commands_than_settle():
    kin = PanTilt()
    counts = {}
    for strategy in ("settle", "preempt", "clip"):
        tracker = HeadTracker(kin, strategy=strategy)
        log, _, _ = run(tracker, walker, 60.0)
        counts[strategy] = sum(1 for _, c in log)
        if strategy == "clip":
            assert sum(1 for _, c in log if c.kind == "clip" and c.why.startswith("track")) >= 5
    assert counts["preempt"] > counts["settle"] * 1.3
    assert tracker.stats["commands"] == counts["clip"]


def test_keep_alive_holds_the_head_against_the_idle():
    kin = PanTilt()
    person = (0.1, 0.9, 0.46)
    on = HeadTracker(kin, config=TrackerConfig(alive=False))
    off = HeadTracker(kin, config=TrackerConfig(alive=False, keep_alive=False))
    log_on, _, lamp_on = run(on, seated(*person), 30.0, lamp=FakeLamp(idle=True))
    log_off, _, lamp_off = run(off, seated(*person), 30.0, lamp=FakeLamp(idle=True))

    def mean_error(lamp):
        return float(np.mean([aim_error(kin, u, person) for t, u in lamp.history if t >= 15.0]))

    assert on.stats["keep_alive"] >= 5
    assert np.diff([t for t, c in log_on if t > 10.0]).max() < 2.6   # moves back to back
    assert mean_error(lamp_on) < 2.0
    # without keep-alive the idle drags the head down between moves until the drift rule re-asserts it
    assert mean_error(lamp_off) > mean_error(lamp_on) + 1.0
    assert off.stats["keep_alive"] == 0 and any(c.why.startswith("track: turn") for t, c in log_off if t > 10.0)


def test_alive_gestures_are_small_bounded_and_optional():
    kin = PanTilt()
    person = (0.1, 0.9, 0.46)
    calm = HeadTracker(kin, config=TrackerConfig(alive=False))
    lively = HeadTracker(kin, config=TrackerConfig(alive=True, alive_every_s=(4.0, 6.0)))
    log_calm, states_calm, _ = run(calm, seated(*person), 40.0)
    log_live, _, _ = run(lively, seated(*person), 40.0)
    t_lock = first_time(states_calm, LOCK)
    held = [c.target for t, c in log_calm if t > t_lock + 3.0]
    assert held and all(u == held[0] for u in held)                   # nothing moves while locked and calm
    gestures = [c.target for _, c in log_live if c.why.startswith("alive: a small")]
    assert len(gestures) >= 4
    for u in gestures:
        assert aim_error(kin, u, person) < 5.0                         # still looking at the person
    assert calm.stats["alive"] == 0


def test_never_commands_a_pose_that_fails_check():
    # look_at does not check its own answers here; a band of yaw is forbidden (a cable, say)
    kin = PanTilt(forbidden=lambda u: 20.0 < u["base_yaw"] < 40.0, honest=False)
    person = (0.34, 0.85, 0.46)                                       # about 22 deg right: inside the band
    for strategy in ("settle", "preempt", "clip"):
        tracker = HeadTracker(kin, strategy=strategy)
        log, _, _ = run(tracker, seated(*person), 30.0)
        assert log
        assert all(kin.check(u).ok for u in all_commanded_poses(log)), strategy
        refused = tracker.stats["pose_refused"] + tracker.stats["path_refused"] + tracker.stats["unreachable"]
        assert refused > 0, strategy


# ------------------------------------------------------------------ seeing
def test_target_point_from_a_detection():
    kin = PanTilt()
    tracker = HeadTracker(kin)
    fx = 0.5 / math.tan(math.radians(61.0) / 2)
    neutral = {j: 0.0 for j in JOINTS}
    camera = kin.head(neutral).position

    def det(x, y, size):
        return Detection(t_capture=0.0, t_delivered=0.15, person_id="", x=x, y=y, size=size, confidence=0.9)

    p = tracker.target_point(det(0.5, 0.5, 0.15 * fx / 1.0), neutral)
    assert np.allclose(p, camera + [0.0, 1.0, 0.0], atol=1e-9)
    right = tracker.target_point(det(0.75, 0.5, 0.15 * fx / 1.0), neutral)
    # picture right is +x at neutral; 1 m deep along the optical axis, a quarter picture off-centre
    assert right == pytest.approx(camera + [0.25 / fx, 1.0, 0.0])
    near = tracker.target_point(det(0.5, 0.5, 5.0), neutral)
    far = tracker.target_point(det(0.5, 0.5, 0.001), neutral)
    assert np.linalg.norm(near - camera) == pytest.approx(0.30)
    assert np.linalg.norm(far - camera) == pytest.approx(2.5)
    down = {**neutral, "wrist_pitch": 55.0}                          # looking 33 deg down, a far "face"
    low = tracker.target_point(det(0.5, 0.9, 0.01), down)
    assert low[2] == pytest.approx(0.09)                             # stops above the table, never below


def test_trace_reports_every_step():
    kin = PanTilt()
    tracker = HeadTracker(kin)
    run(tracker, seated(), 8.0)
    trace = tracker.trace
    for key in ("t", "state", "strategy", "locked_id", "tracks", "target_m", "aim_error_deg", "busy",
                "command", "why", "blocked", "window_used", "failures", "scoring_person_id"):
        assert key in trace
    assert trace["state"] in (ACQUIRE, LOCK)


# ------------------------------------------------------------------ what the tracker may know
def test_the_tracker_times_a_frame_by_its_stamp_not_by_the_exposure():
    """Detection.t_capture is ground truth for scoring; the tracker must place a frame at its stamp."""
    kin = PanTilt()
    tracker = HeadTracker(kin)
    turned = {**kin.neutral, "base_yaw": 40.0}
    for k in range(40):                                      # the head sat still, then turned at t = 0.5
        tracker.update(k * 0.02, [], kin.neutral if k * 0.02 < 0.5 else turned, False)
    det = Detection(t_capture=0.3, t_delivered=0.8, person_id="", x=0.5, y=0.5, size=0.15, confidence=0.9,
                    t_stamp=0.6)                             # exposed at 0.3 (straight ahead), stamped at 0.6
    tracker.update(0.8, [det], turned, False)
    (track,) = tracker._tracks
    # placed with the pose at the stamp (turned 40 units = 29.6 deg to the right), not at the exposure
    assert math.degrees(math.atan2(track.pos[0], track.pos[1])) > 20.0


def test_a_failed_settle_close_to_its_target_counts_as_reached():
    """The review's blocker: sag just over a vendor tolerance made every settle 'fail', three failures put
    the tracker in HOLD and tracking stopped. The SDK's failure carries each joint's error and tolerance."""
    kin = PanTilt()
    tracker = HeadTracker(kin)
    tol = {"base_yaw": 2.0, "base_pitch": 2.0, "elbow_pitch": 10.0, "wrist_roll": 2.0, "wrist_pitch": 3.0}
    sag = {"base_yaw": 0.1, "base_pitch": 3.0, "elbow_pitch": 15.0, "wrist_roll": 0.0, "wrist_pitch": 4.5}  # 1.5 x
    for k in range(5):
        tracker._pending = 1
        tracker.update(1.0 + k, [], kin.neutral, False, "failed",
                       {"position_errors": sag, "position_tolerances": tol})
    assert tracker.state != HOLD and tracker.stats["failures"] == 0 and tracker.stats["failed_but_reached"] == 5
    stalled = {**sag, "elbow_pitch": 35.0}                   # 3.5 x its tolerance: not gravity
    for k in range(3):
        tracker._pending = 1
        tracker.update(10.0 + 6 * k, [], kin.neutral, False, "failed",
                       {"position_errors": stalled, "position_tolerances": tol})
    assert tracker.state == HOLD and tracker.stats["failures"] == 3
    # no details (an older gateway): a failure is a failure
    fresh = HeadTracker(kin)
    fresh._pending = 1
    fresh.update(1.0, [], kin.neutral, False, "failed", None)
    assert fresh.stats["failures"] == 1
