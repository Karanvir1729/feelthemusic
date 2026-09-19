"""The room the lamp looks at: scripted people who walk in, sit, sway, lean close, leave and come back.

Pure numpy. A Scenario answers one question, people_at(t): who is in the room at time t, where is the
centre of each head, and which way does each face point. It is a pure function of t, so a run can ask
for any time in any order and two scenarios built with the same name and seed give identical people.

Frames and units are contract.py's: base frame in metres, x right, y forward (where the lamp faces at
base_yaw neutral), z up, z = 0 is the table top (the underside of the lamp's base plate). The lamp
stands at the origin. People sit or stand on the far side of the table, in front of the lamp (y > 0).

Every number that shapes a person is either the workflow brief's room model or a typical-adult
ASSUMPTION; none of it was measured on people at the event. The twin uses these scenes to exercise the
tracker, not to claim how real listeners move.

  scenario("walk_in_sit", seed=0).people_at(3.2)  -> [Person(id="p1", head=..., facing=...)]
  SCENARIOS                                      -> name -> Scenario subclass
"""
from __future__ import annotations

import math
import zlib

import numpy as np

from twin.contract import Person

# ------------------------------------------------------------------ the room (all ASSUMPTION unless cited)
# Head heights above the table. ASSUMPTION (workflow brief room model): a seated person's head is about
# 0.35-0.55 m above the table, a standing person's about 0.9-1.1 m. The ranges below are drawn per person
# and leave room for sway, breathing, gait bob and dancing to stay inside the brief's bands.
SEATED_HEAD_Z = (0.42, 0.50)
STANDING_HEAD_Z = (0.95, 1.05)

# Where people look when they look "at the lamp": the lamp's head in its neutral pose. lamp/spatial.py puts
# the neutral head camera about 8 cm forward of and 32 cm above the base origin, facing +y (computed from the
# vendor robot description at run time; lamp/tests/test_spatial.py checks 0.30-0.35 m). People look at the
# lamp, not at wherever the head happens to be, so this point does not follow the arm.
LAMP_LOOK_POINT = np.array([0.0, 0.08, 0.32])

# Walking. ASSUMPTION (typical adult gait, not measured here): 0.7-0.9 m/s indoors (the brief fixes
# cross_room at 0.8 m/s), about 1.8-2.0 steps per second, the head rising and falling about 4 cm peak to
# peak once per step and swaying about 3 cm side to side once per stride (two steps).
WALK_SPEED_M_S = (0.7, 0.9)
CROSS_ROOM_SPEED_M_S = 0.8                 # workflow brief
STEP_HZ = (1.8, 2.0)
GAIT_BOB_M = 0.02                          # amplitude, so about 4 cm peak to peak
GAIT_SWAY_M = 0.015
GAIT_RAMP_S = 0.5                          # ASSUMPTION: gait and gaze blend in/out over half a second
# ASSUMPTION: a walker turns the head up to 45-65 deg from the walking direction toward a moving, glowing
# lamp (a comfortable head turn without turning the shoulders). Once they are past the lamp it falls behind
# that angle, so from about the middle of a crossing on the face goes to profile and is lost, on purpose.
GLANCE_DEG = (45.0, 65.0)

# Sitting still. ASSUMPTION (typical seated posture): centimetre-scale postural sway at 0.1-0.3 Hz,
# breathing at 12-18 breaths per minute moving the head a few millimetres, a few degrees of idle head yaw.
SWAY_M = (0.008, 0.015)
SWAY_HZ = (0.10, 0.30)
BREATH_M = (0.002, 0.004)
BREATH_HZ = (0.20, 0.30)
IDLE_YAW_DEG = (2.0, 5.0)
IDLE_YAW_HZ = (0.08, 0.20)

# Looking around. ASSUMPTION: every 4-9 s a seated or standing person turns the head 30-80 deg for 1-2.5 s.
# Turns past the detector's profile limit (perception.py, 70 deg) briefly hide the face, on purpose.
LOOK_EVERY_S = (4.0, 9.0)
LOOK_HOLD_S = (1.0, 2.5)
LOOK_YAW_DEG = (30.0, 80.0)
LOOK_TURN_S = 0.3

# Getting up and down. ASSUMPTION: about 1-2 s to sit down or stand up, the head travelling from the
# standing spot about 25 cm behind the seat (away from the lamp) to the seat.
SIT_S = (1.3, 1.8)
STAND_S = (1.0, 1.4)
BEHIND_SEAT_M = 0.25

# Dancing in a chair. ASSUMPTION: the head dips once per beat by 1.5-2.5 cm and nods 4-8 deg, and the body
# sways 3-5 cm side to side once every two beats.
DANCE_BOB_M = (0.015, 0.025)
DANCE_SWAY_M = (0.03, 0.05)
DANCE_NOD_DEG = (4.0, 8.0)

# Out of the room. ASSUMPTION: a doorway 2.8 m to either side, which is beyond the detector's 2.5 m range.
EXIT_X_M = 2.8


# ------------------------------------------------------------------ small helpers
def _smootherstep(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * u * (u * (6.0 * u - 15.0) + 10.0)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _az_el(v: np.ndarray) -> tuple[float, float]:
    """Azimuth from +y toward +x, and elevation above the table plane, of a direction."""
    return math.atan2(v[0], v[1]), math.atan2(v[2], math.hypot(v[0], v[1]))


def _unit(az: float, el: float) -> np.ndarray:
    return np.array([math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)])


def _u(rng: np.random.Generator, band: tuple[float, float]) -> float:
    return float(rng.uniform(*band))


class _Leg:
    """One stretch of a person's route. kind: walk (constant speed), ease (sit/stand/lean, smootherstep),
    stay (still, seated or standing) or gone (out of the room)."""

    def __init__(self, t0: float, t1: float, kind: str, p0, p1, posture: str):
        self.t0, self.t1, self.kind, self.posture = t0, t1, kind, posture
        self.p0, self.p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)


class _Track:
    """One person's scripted route and the small motions layered on top of it. Built once, then read
    as a pure function of time."""

    def __init__(self, pid: str, rng: np.random.Generator, t: float, head, posture: str):
        self.pid = pid
        self.legs: list[_Leg] = []
        self.start = t
        self._t, self._p, self._posture = t, np.asarray(head, dtype=float), posture
        self.looks: list[tuple[float, float, float]] = []      # (start, hold, yaw offset in radians)
        self.dance: dict | None = None
        # this person's own small motions, drawn once
        self.m = {
            "sway": [(_u(rng, SWAY_M), _u(rng, SWAY_HZ), rng.uniform(0, 2 * math.pi)) for _ in range(2)],
            "breath": (_u(rng, BREATH_M), _u(rng, BREATH_HZ), rng.uniform(0, 2 * math.pi)),
            "yaw": (math.radians(_u(rng, IDLE_YAW_DEG)), _u(rng, IDLE_YAW_HZ), rng.uniform(0, 2 * math.pi)),
            "step_hz": _u(rng, STEP_HZ),
            "glance": math.radians(_u(rng, GLANCE_DEG)),
        }

    # ---------------------------------------------------------------- building the route
    @property
    def end(self) -> float:
        return self._t

    def walk(self, to, speed: float) -> "_Track":
        to = np.asarray(to, dtype=float)
        duration = max(float(np.linalg.norm(to - self._p)) / speed, 0.1)
        return self._add("walk", to, duration, "walking")

    def ease(self, to, duration: float, posture: str) -> "_Track":
        return self._add("ease", np.asarray(to, dtype=float), duration, posture)

    def stay(self, duration: float) -> "_Track":
        return self._add("stay", self._p, duration, self._posture)

    def gone(self, duration: float) -> "_Track":
        return self._add("gone", self._p, duration, "gone")

    def appear(self, head, posture: str) -> "_Track":
        """Come back into the room somewhere else (after gone())."""
        self._p, self._posture = np.asarray(head, dtype=float), posture
        return self

    def _add(self, kind: str, to: np.ndarray, duration: float, posture: str) -> "_Track":
        self.legs.append(_Leg(self._t, self._t + duration, kind, self._p, to, posture))
        self._t += duration
        self._p = to
        if kind != "walk":
            self._posture = posture
        return self

    def plan_looks(self, rng: np.random.Generator, until: float) -> None:
        """Scatter look-arounds over the route; keep only those that fit inside one still stretch."""
        t = self.start + _u(rng, LOOK_EVERY_S)
        while t < until:
            hold = _u(rng, LOOK_HOLD_S)
            yaw = math.radians(_u(rng, LOOK_YAW_DEG)) * (1.0 if rng.random() < 0.5 else -1.0)
            window = (t - LOOK_TURN_S, t + hold + 2 * LOOK_TURN_S)
            if any(leg.kind == "stay" and leg.t0 <= window[0] and window[1] <= leg.t1 for leg in self.legs):
                self.looks.append((t, hold, yaw))
            t += hold + _u(rng, LOOK_EVERY_S)

    # ---------------------------------------------------------------- reading it back
    def _leg_at(self, t: float) -> _Leg | None:
        if t < self.start or not self.legs:
            return None
        for leg in self.legs:
            if t < leg.t1:
                return leg
        return self.legs[-1]                               # after the script: keep the last state

    def person_at(self, t: float) -> Person | None:
        leg = self._leg_at(t)
        if leg is None or leg.kind == "gone":
            return None
        span = max(leg.t1 - leg.t0, 1e-9)
        u = min(1.0, max(0.0, (t - leg.t0) / span))
        if leg.kind == "walk":
            base = leg.p0 + (leg.p1 - leg.p0) * u
        elif leg.kind == "ease":
            base = leg.p0 + (leg.p1 - leg.p0) * _smootherstep(u)
        else:
            base = leg.p1
        # how much of the walking gait is on: ramps in and out so nothing jumps at a leg boundary
        gait = 0.0
        if leg.kind == "walk" and t < leg.t1:
            gait = min(1.0, max(0.0, min(t - leg.t0, leg.t1 - t) / GAIT_RAMP_S))
        head = base + self._still_motion(t) * (1.0 - gait) + self._gait_motion(t, leg) * gait
        return Person(id=self.pid, head=head, facing=self._facing(t, head, leg, gait))

    def _still_motion(self, t: float) -> np.ndarray:
        (ax, fx, px), (ay, fy, py) = self.m["sway"]
        ab, fb, pb = self.m["breath"]
        d = np.array([ax * math.sin(2 * math.pi * fx * t + px),
                      ay * math.sin(2 * math.pi * fy * t + py),
                      ab * math.sin(2 * math.pi * fb * t + pb)])
        if self.dance:
            beat = 60.0 / self.dance["bpm"]
            phase = (t - self.dance["beat0"]) / beat
            d[2] -= self.dance["bob"] * (0.5 + 0.5 * math.cos(2 * math.pi * phase))   # lowest on the beat
            d[0] += self.dance["sway"] * math.sin(math.pi * phase)                   # one sway per 2 beats
        return d

    def _gait_motion(self, t: float, leg: _Leg) -> np.ndarray:
        travel = leg.p1 - leg.p0
        travel = travel / max(float(np.linalg.norm(travel)), 1e-9)
        side = np.array([travel[1], -travel[0], 0.0])
        side = side / max(float(np.linalg.norm(side)), 1e-9)
        phase = 2 * math.pi * self.m["step_hz"] * (t - leg.t0)
        return np.array([0.0, 0.0, GAIT_BOB_M * math.cos(phase)]) + side * GAIT_SWAY_M * math.sin(phase / 2)

    def _facing(self, t: float, head: np.ndarray, leg: _Leg, gait: float) -> np.ndarray:
        az, el = _az_el(LAMP_LOOK_POINT - head)            # by default people look at the lamp
        if leg.kind == "walk":
            # walkers face where they go, glancing toward the lamp by up to their glance angle
            walk_az, _ = _az_el(leg.p1 - leg.p0)
            walk_az += float(np.clip(_wrap(az - walk_az), -self.m["glance"], self.m["glance"]))
            az += _wrap(walk_az - az) * gait
            el += (0.5 * el - el) * gait
        a, f, p = self.m["yaw"]
        az += a * math.sin(2 * math.pi * f * t + p)
        for start, hold, yaw in self.looks:
            turn = (_smootherstep((t - start) / LOOK_TURN_S)
                    - _smootherstep((t - start - hold - LOOK_TURN_S) / LOOK_TURN_S))
            az += yaw * turn
        if self.dance:
            phase = (t - self.dance["beat0"]) * self.dance["bpm"] / 60.0
            el -= self.dance["nod"] * (0.5 + 0.5 * math.cos(2 * math.pi * phase))    # nod down on the beat
        return _unit(az, el)


# ------------------------------------------------------------------ scenarios
# Inside each scene, the lengths, entry times, seat spots and hold times are script choices (ASSUMPTION),
# picked so each scene shows its one behaviour with time to spare for a 2 s-per-move lamp. Numbers the
# workflow brief fixes (0.8 m/s at 1.2 m, 6 s away, 0.3 m lean, 120 BPM) are marked where they are used.
class Scenario:
    """A scripted room. Subclasses set name, duration_s and description, and build their people."""
    name = ""
    duration_s = 0.0
    description = ""

    def __init__(self, seed: int = 0):
        self.seed = int(seed)
        # one stream per (scenario, seed); crc32 because Python's str hash changes between processes
        self._rng = np.random.default_rng([zlib.crc32(self.name.encode()), self.seed])
        self._tracks: list[_Track] = []
        self._build(self._rng)

    def _build(self, rng: np.random.Generator) -> None:
        pass

    @property
    def people_ids(self) -> list[str]:
        return [track.pid for track in self._tracks]

    def people_at(self, t: float) -> list[Person]:
        out = []
        for track in self._tracks:
            person = track.person_at(float(t))
            if person is not None:
                out.append(person)
        return out

    # ---------------------------------------------------------------- shared building blocks
    def _seat(self, rng, x=(-0.15, 0.15), y=(0.75, 0.95)) -> np.ndarray:
        return np.array([_u(rng, x), _u(rng, y), _u(rng, SEATED_HEAD_Z)])

    @staticmethod
    def _behind(seat: np.ndarray, standing_z: float) -> np.ndarray:
        return np.array([seat[0], seat[1] + BEHIND_SEAT_M, standing_z])


class Empty(Scenario):
    name, duration_s = "empty", 20.0
    description = "Nobody in the room. The lamp should keep looking around and never lock."


class WalkInSit(Scenario):
    name, duration_s = "walk_in_sit", 30.0
    description = "A person enters from the side, sits in front of the lamp, sways, breathes, looks around."

    def _build(self, rng):
        side = 1.0 if rng.random() < 0.5 else -1.0
        seat = self._seat(rng)
        standing_z = _u(rng, STANDING_HEAD_Z)
        spot = self._behind(seat, standing_z)
        entry = np.array([side * EXIT_X_M, spot[1] + _u(rng, (0.0, 0.4)), standing_z])
        track = _Track("p1", rng, _u(rng, (1.5, 3.0)), entry, "standing")
        track.walk(spot, _u(rng, WALK_SPEED_M_S)).stay(_u(rng, (0.3, 0.8)))
        track.ease(seat, _u(rng, SIT_S), "seated").stay(self.duration_s - track.end + 1.0)
        track.plan_looks(rng, self.duration_s)
        self._tracks.append(track)


class SwayToMusic(Scenario):
    """Seated all the way through, dancing to a 120 BPM beat that starts at t = 0: the tempo and length
    (16 bars, 32 s) of the Mac conductor's synthetic song (conductor source FileSource.swift:68-73). The
    dancer keeps the same energy all song long; they do not follow its build or drop."""
    name, duration_s = "sway_to_music", 32.0
    description = "A seated person bobs on every beat and sways side to side at 120 BPM."
    bpm, beat0 = 120.0, 0.0

    def _build(self, rng):
        track = _Track("p1", rng, 0.0, self._seat(rng, x=(-0.10, 0.10), y=(0.75, 0.90)), "seated")
        track.stay(self.duration_s + 1.0)
        track.dance = {"bpm": self.bpm, "beat0": self.beat0, "bob": _u(rng, DANCE_BOB_M),
                       "sway": _u(rng, DANCE_SWAY_M), "nod": math.radians(_u(rng, DANCE_NOD_DEG))}
        self._tracks.append(track)


class CrossRoom(Scenario):
    name, duration_s = "cross_room", 9.0
    description = "A person walks left to right at 0.8 m/s, 1.2 m in front of the lamp, and leaves."

    def _build(self, rng):
        y = 1.2 + _u(rng, (-0.05, 0.05))                 # workflow brief: at 1.2 m
        z = _u(rng, STANDING_HEAD_Z)
        track = _Track("p1", rng, 0.5, [-EXIT_X_M, y, z], "standing")
        track.walk([EXIT_X_M, y, z], CROSS_ROOM_SPEED_M_S).gone(self.duration_s)
        self._tracks.append(track)


class LeaveReturn(Scenario):
    name, duration_s = "leave_return", 34.0
    description = "Seated, gets up and leaves the room for 6 s, comes back from the other side to another seat."
    away_s = 6.0                                        # workflow brief

    def _build(self, rng):
        seat_a = self._seat(rng, x=(-0.25, -0.10), y=(0.78, 0.92))
        seat_b = self._seat(rng, x=(0.15, 0.30), y=(0.65, 0.80))
        standing_z = _u(rng, STANDING_HEAD_Z)
        track = _Track("p1", rng, 0.0, seat_a, "seated")
        track.stay(_u(rng, (7.0, 9.0)))
        track.ease(self._behind(seat_a, standing_z), _u(rng, STAND_S), "standing")
        track.walk([-EXIT_X_M, seat_a[1] + 0.4, standing_z], _u(rng, WALK_SPEED_M_S))
        track.gone(self.away_s)
        spot_b = self._behind(seat_b, standing_z)
        track.appear([EXIT_X_M, spot_b[1] + 0.1, standing_z], "standing")
        track.walk(spot_b, _u(rng, WALK_SPEED_M_S)).stay(_u(rng, (0.3, 0.8)))
        track.ease(seat_b, _u(rng, SIT_S), "seated").stay(self.duration_s - track.end + 1.0)
        track.plan_looks(rng, self.duration_s)
        self._tracks.append(track)


class TwoPeople(Scenario):
    name, duration_s = "two_people", 30.0
    description = "One person seated in front; a second walks in behind them, stands and stays."

    def _build(self, rng):
        seat = self._seat(rng, x=(-0.10, 0.10), y=(0.75, 0.90))
        first = _Track("p1", rng, 0.0, seat, "seated").stay(self.duration_s + 1.0)
        first.plan_looks(rng, self.duration_s)
        side = 1.0 if rng.random() < 0.5 else -1.0
        standing_z = _u(rng, STANDING_HEAD_Z)
        # behind the first person and a little to one side, so the second face is usually in the picture too
        spot = np.array([seat[0] + side * _u(rng, (0.35, 0.55)), seat[1] + _u(rng, (0.6, 0.8)), standing_z])
        second = _Track("p2", rng, _u(rng, (8.0, 10.0)), [side * EXIT_X_M, spot[1] + 0.2, standing_z], "standing")
        second.walk(spot, _u(rng, WALK_SPEED_M_S)).stay(self.duration_s - second.end + 1.0)
        second.plan_looks(rng, self.duration_s)
        self._tracks += [first, second]


class LeanClose(Scenario):
    """'0.3 m' is the head centre's horizontal distance from the lamp's base axis (the z axis)."""
    name, duration_s = "lean_close", 22.0
    description = "Seated, leans in to 0.3 m from the lamp, holds, then leans back."
    closest_m = 0.30                                     # workflow brief

    def _build(self, rng):
        seat = self._seat(rng, x=(-0.10, 0.10), y=(0.80, 0.90))
        # ASSUMPTION: leaning in lowers the head 4-7 cm; x shrinks with y so the horizontal distance is ~0.30 m
        close = np.array([seat[0] * self.closest_m / seat[1], self.closest_m, seat[2] - _u(rng, (0.04, 0.07))])
        close[:2] *= self.closest_m / math.hypot(close[0], close[1])
        track = _Track("p1", rng, 0.0, seat, "seated")
        track.stay(_u(rng, (5.0, 7.0)))
        track.ease(close, _u(rng, (1.8, 2.4)), "seated").stay(_u(rng, (3.0, 4.0)))
        track.ease(seat, _u(rng, (1.8, 2.4)), "seated").stay(self.duration_s - track.end + 1.0)
        # no look-arounds: this scene is about distance, and a turned face would hide the lean
        self._tracks.append(track)


SCENARIOS: dict[str, type[Scenario]] = {cls.name: cls for cls in
                                        (Empty, WalkInSit, SwayToMusic, CrossRoom, LeaveReturn, TwoPeople, LeanClose)}


def scenario(name: str, seed: int = 0) -> Scenario:
    """A fresh scenario by name. The same name and seed always give the same people."""
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; known: {', '.join(SCENARIOS)}")
    return SCENARIOS[name](seed)
