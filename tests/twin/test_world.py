"""The scripted room (twin/world.py). Pure numpy: no robot needed.

The bands checked here are the workflow brief's room model: table at z = 0, a seated head about
0.35-0.55 m above it, a standing head about 0.9-1.1 m, people about 0.3-1.5 m in front of the lamp.
"""
import math

import numpy as np
import pytest

from twin.contract import Person, Pose
from twin.perception import FaceDetectorModel
from twin.world import LAMP_LOOK_POINT, SCENARIOS, Scenario, scenario

NAMES = ("empty", "walk_in_sit", "sway_to_music", "cross_room", "leave_return", "two_people", "lean_close")
SEATED = (0.35, 0.55)
STANDING = (0.90, 1.10)


def samples(s: Scenario, dt: float = 0.05):
    """(t, people) from 0 to the scenario's duration, inclusive."""
    for k in range(int(round(s.duration_s / dt)) + 1):
        t = k * dt
        yield t, s.people_at(t)


def track(s: Scenario, pid: str, dt: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Times and head positions of one person, only where they are in the room."""
    ts, heads = [], []
    for t, people in samples(s, dt):
        for p in people:
            if p.id == pid:
                ts.append(t)
                heads.append(p.head)
    return np.array(ts), np.array(heads)


def face_angle_to_lamp_deg(p: Person) -> float:
    to = LAMP_LOOK_POINT - p.head
    return math.degrees(math.acos(float(np.clip(p.facing @ to / np.linalg.norm(to), -1.0, 1.0))))


# ------------------------------------------------------------------ every scenario
def test_the_registry_has_every_scenario():
    assert set(SCENARIOS) == set(NAMES)
    for name in NAMES:
        s = scenario(name)
        assert s.name == name and s.duration_s > 0 and s.description
    with pytest.raises(KeyError):
        scenario("dance_party")


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_each_scenario_runs_for_its_duration_with_heads_above_the_table(name, seed):
    s = scenario(name, seed=seed)
    seen = 0
    for t, people in samples(s, dt=0.02):
        assert len({p.id for p in people}) == len(people)
        for p in people:
            assert isinstance(p, Person)
            assert p.head.shape == (3,) and np.all(np.isfinite(p.head))
            assert SEATED[0] - 0.05 <= p.head[2] <= STANDING[1], f"{name} t={t:.2f}: head at z={p.head[2]:.3f}"
            assert p.head[1] > 0.25, f"{name} t={t:.2f}: head at y={p.head[1]:.3f} is not in front of the lamp"
            assert np.linalg.norm(p.facing) == pytest.approx(1.0, abs=1e-9)
            seen += 1
    assert (seen == 0) == (name == "empty")
    # asking beyond the end is allowed and keeps the last state
    s.people_at(s.duration_s + 5.0)


@pytest.mark.parametrize("name", NAMES)
def test_the_same_seed_gives_the_same_people(name):
    a, b, c = scenario(name, seed=4), scenario(name, seed=4), scenario(name, seed=5)
    times = np.linspace(0.0, a.duration_s, 97)
    def heads(s):
        return [(p.id, tuple(p.head), tuple(p.facing)) for t in times for p in s.people_at(t)]
    assert heads(a) == heads(b)
    if name != "empty":
        assert heads(a) != heads(c)
    # a pure function of t: asking out of order gives the same answer
    assert heads(a) == [x for x in heads(scenario(name, seed=4))]
    backwards = [(p.id, tuple(p.head)) for t in times[::-1] for p in a.people_at(t)]
    forwards = [(p.id, tuple(p.head)) for t in times for p in a.people_at(t)]
    assert sorted(backwards) == sorted(forwards)


def test_empty_room_is_empty():
    s = scenario("empty")
    assert s.duration_s == 20.0
    assert all(people == [] for _, people in samples(s))


# ------------------------------------------------------------------ each scene does what its name says
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_walk_in_sit_enters_from_the_side_and_sits_in_front(seed):
    s = scenario("walk_in_sit", seed=seed)
    ts, heads = track(s, "p1")
    assert ts[0] > 1.0                                              # nobody there at the start
    assert abs(heads[0][0]) > 2.5 and STANDING[0] <= heads[0][2] <= STANDING[1]
    seated = heads[ts > s.duration_s - 10.0]
    assert np.all((seated[:, 2] >= SEATED[0]) & (seated[:, 2] <= SEATED[1]))
    assert np.all(np.abs(seated[:, 0]) < 0.3) and np.all((seated[:, 1] > 0.5) & (seated[:, 1] < 1.5))
    # alive while seated: centimetre sway and a breathing bob, never frozen, never wandering off
    assert 0.001 < np.std(seated[:, 2]) < 0.02
    assert 0.002 < np.ptp(seated[:, 0]) < 0.06


def test_walk_in_sit_looks_around_sometimes():
    turned = 0
    for seed in range(5):
        s = scenario("walk_in_sit", seed=seed)
        for _, people in samples(s, dt=0.1):
            if people and people[0].head[2] < SEATED[1] and face_angle_to_lamp_deg(people[0]) > 25.0:
                turned += 1
    assert turned > 10                                              # at least a second of looking away


def test_sway_to_music_bobs_on_a_120_bpm_beat():
    s = scenario("sway_to_music", seed=0)
    beat = 60.0 / 120.0
    on = np.array([s.people_at(k * beat)[0].head for k in range(4, 60)])
    off = np.array([s.people_at(k * beat + beat / 2)[0].head for k in range(4, 60)])
    assert np.mean(off[:, 2]) - np.mean(on[:, 2]) > 0.01            # lowest on the beat, by a centimetre or more
    # one side-to-side sway every two beats
    one_side = np.array([s.people_at(k * 2 * beat + beat / 2)[0].head[0] for k in range(2, 30)])
    other_side = np.array([s.people_at(k * 2 * beat + 1.5 * beat)[0].head[0] for k in range(2, 30)])
    assert np.mean(one_side) - np.mean(other_side) > 0.04          # 3-5 cm each way
    ts, heads = track(s, "p1")
    assert ts[0] == 0.0 and ts[-1] == pytest.approx(s.duration_s)
    assert np.all((heads[:, 2] >= SEATED[0]) & (heads[:, 2] <= SEATED[1]))


@pytest.mark.parametrize("seed", [0, 1])
def test_cross_room_walks_left_to_right_at_walking_speed(seed):
    s = scenario("cross_room", seed=seed)
    ts, heads = track(s, "p1")
    assert heads[0][0] < -2.5 and heads[-1][0] > 2.5
    assert np.all(np.diff(heads[:, 0]) > 0)
    mid = (ts > 2.0) & (ts < 6.0)
    speed = np.polyfit(ts[mid], heads[mid, 0], 1)[0]
    assert speed == pytest.approx(0.8, abs=0.01)
    assert np.all(np.abs(heads[:, 1] - 1.2) < 0.08)
    assert np.all((heads[:, 2] >= STANDING[0]) & (heads[:, 2] <= STANDING[1]))
    assert s.people_at(s.duration_s) == []                          # gone by the end


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_leave_return_leaves_for_six_seconds_and_sits_somewhere_else(seed):
    s = scenario("leave_return", seed=seed)
    ts, heads = track(s, "p1", dt=0.02)
    gaps = np.diff(ts)
    assert gaps.max() >= 6.0 - 0.02 and np.sum(gaps > 0.03) == 1  # one absence, 6 s long
    before, after = heads[ts < 5.0], heads[ts > s.duration_s - 5.0]
    for part in (before, after):
        assert np.all((part[:, 2] >= SEATED[0]) & (part[:, 2] <= SEATED[1]))
    assert np.linalg.norm(before.mean(axis=0) - after.mean(axis=0)) > 0.2


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_two_people_second_walks_in_behind_and_stays(seed):
    s = scenario("two_people", seed=seed)
    assert [p.id for p in s.people_at(0.0)] == ["p1"]
    end = {p.id: p for p in s.people_at(s.duration_s)}
    assert set(end) == {"p1", "p2"}
    assert np.hypot(*end["p2"].head[:2]) > np.hypot(*end["p1"].head[:2]) + 0.3    # behind the first
    assert STANDING[0] <= end["p2"].head[2] <= STANDING[1]
    assert SEATED[0] <= end["p1"].head[2] <= SEATED[1]
    ts, _ = track(s, "p2")
    assert ts[-1] == pytest.approx(s.duration_s)                    # stays to the end


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lean_close_comes_to_thirty_centimetres_and_back(seed):
    s = scenario("lean_close", seed=seed)
    ts, heads = track(s, "p1")
    reach = np.hypot(heads[:, 0], heads[:, 1])                      # distance from the lamp's base axis
    assert reach.min() == pytest.approx(0.30, abs=0.025)            # postural sway still runs while leaning
    assert reach[0] > 0.7 and reach[-1] > 0.7
    closest = ts[np.argmin(reach)]
    assert 5.0 < closest < s.duration_s - 5.0
    assert np.all(heads[:, 2] >= SEATED[0])


# ------------------------------------------------------------------ with the detector
def test_a_camera_aimed_at_the_seated_listener_sees_them_most_of_the_time():
    """World and perception together: a camera at the lamp's neutral head spot, turned to face the seated
    person, detects them in most frames (misses, noise and look-arounds are allowed to cost a few)."""
    s = scenario("walk_in_sit", seed=0)
    detector = FaceDetectorModel(seed=0)
    camera = LAMP_LOOK_POINT
    frames = found = 0
    for k in range(int(s.duration_s * detector.fps)):
        t = k / detector.fps
        people = s.people_at(t)
        if not people or people[0].head[2] > SEATED[1]:
            continue                                                # score only the seated stretch
        forward = people[0].head - camera
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        pose = Pose(position=camera, forward=forward, down=np.cross(forward, right), right=right)
        detector.observe(t, pose, people)
        frames += 1
        found += len(detector.poll(t + detector.latency_s))
    assert frames > 100
    assert found / frames > 0.8
