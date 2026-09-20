"""What the Feetech STS3215 servos and the lamp's joints do with the goal stream the runtime writes at them.

This is the last stage of an offline simulator of the lamp. Its INPUT is what the vendor runtime's
executor actually puts on the servo bus: one Goal_Position write per waypoint at the plan's frame rate
(runtime_ref modules/robot_base/control/executor.py, MotionExecutor.execute: `write_positions` per
waypoint, then sleep to the next waypoint's offset, and NOTHING after the last one, so the servo keeps
its last goal -- the README's "not chased after the plan ends"). Blending a clip in, the transition
clamp and the runtime's start latency happen BEFORE this stage (fusion.py) and are another module's
job; feeding this model a raw clip answers "what would the servos do with that stream", not "what does
the runtime do with that clip".

Its OUTPUT is the measured-position trajectory at the vendor's simulation rate, 100 Hz, plus a per-frame
flag saying whether a joint is pressed against a stop ("banging").

The model, per joint, per 10 ms tick (all units are the runtime's: -100..100 over the calibrated
range; 0.74 deg/unit on base_yaw from the calibration's 1693-tick span):

  1. goal in force   = the last write at or before (t - command_latency)            [simulation.yaml]
  2. tracking demand = 0 inside the deadband, else (goal - drive) / tau, clipped to +-max_velocity
                       -- a first-order lag. The vendor writes it as damping 24/s (tau 42 ms); the arm
                       measures 100-150 ms behind the clip timeline, so tau is a FREE parameter here.
  3. spin-up         = |velocity| may grow by at most max_acceleration * dt per tick; braking follows the
                       demand (the arm never overshot tonight: every direct move landed 1.2-1.6 units
                       SHORT, none long)
  4. drive           += velocity * dt
  5. joint           = drive through the gear train's play: the joint trails the drive by up to
                       `backlash` in the direction of travel and moves only when the drive reaches
                       the far edge of the play
  6. stops           = the joint is clamped to [lo, hi]; while its goal is beyond the stop it stays
                       pressed there with zero velocity and its tracking error persists; "banging" is
                       recorded for that tick; every other joint is untouched
  7. measured        = the joint position read_latency earlier, quantised to position_quantization

What is deliberately NOT here: pose-dependent inertia (the same yaw servo delivered 86 % of a +-30 wave
in the raised pose and 0.48 of beat wiggles with the arm out), the elbow's 20-29 unit sag under gravity
when extended (README), the runtime's stall when a joint cannot reach its target (the ground truth's
"a stalled joint drags the whole clip" is the executor waiting, not the servo -- read `banging` and
model it upstream), and the EEPROM park-at-limit on power-up (yaw_limits.py). No randomness anywhere:
two runs with the same input are bit-identical, so a fit in the Validate phase converges on the
parameters and not on noise.

Ground truth this must reproduce (scratchpad/ground_truth_2026-09-20.md, measured on the arm):
  * every leftward base_yaw command stopped at -4.0..-5.3 (direct POSTs -5.1, -5.0, -4.2; cha_turn
    -4.9; wave_hi -5.0; beat_hype -6.1 at speed), while +40 reaches +38.6 and dance clips reach +73:
    STOP_YAW_MIN, a hard floor, below the calibration's -100
  * a 40-unit tracking move over 4 s (10 u/s) delivers 96 % (+38.6); 3 s direct moves of +25.3, +11.4
    and +41.7 asked travelled 23.7, 10.2 and 40.2 -- an ABSOLUTE 1.2-1.6 unit deficit, not a
    proportional one, which is what backlash 1.0 + deadband 0.2 leave behind
  * at beat rates (128 bpm clips, P 24 / torque 700) base_pitch delivered 0.88, elbow and wrist_pitch
    0.75, base_yaw and wrist_roll ~0.48 of the commanded amplitude, 100-150 ms behind the clip
    (analysis/beattrace.py). A per-joint tau reproduces different ratios with one physics; if it
    cannot, `delivery` overrides the ratio per joint.
"""
from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spatial import JOINTS  # noqa: E402  (one definition of the joint order for the whole package)

NJ = len(JOINTS)
YAW = JOINTS.index("base_yaw")

# ----------------------------------------------------------------------------- the vendor's motor model
# static/robots/lelamp_v1/pi5_feetech_r1/simulation.yaml, block `motor:` (model feetech_sts3215.v1).
# The runtime ships no code for it; these numbers ARE the model. vendor_motor_params() reads the file
# back so a test can prove the constants below are the file's.
UPDATE_HZ = 100.0                    # motor.update_hz: the tick of this model
DT = 1.0 / UPDATE_HZ
COMMAND_LATENCY_S = 0.004            # motor.command_latency_s: a write takes effect 4 ms later
READ_LATENCY_S = 0.006               # motor.read_latency_s: a read returns the position 6 ms ago
VENDOR_MAX_VELOCITY = 140.0          # motor.max_velocity_per_s, units/s
VENDOR_MAX_ACCELERATION = 420.0      # motor.max_acceleration_per_s2, units/s^2. Applied to spin-up only
                                     # (step 3): as a symmetric limit it would cap a +-30 wiggle at 1.33 Hz
                                     # near 8 units, and wave_hi delivered 25.8 on its free side; as a
                                     # spin-up ceiling it costs that wave 0.1 unit (26.3 vs 26.4 peak).
VENDOR_DAMPING_PER_S = 24.0          # motor.damping_per_s: the vendor's tracking rate ...
VENDOR_TAU_S = 1.0 / VENDOR_DAMPING_PER_S   # ... i.e. a 42 ms first-order lag
VENDOR_BACKLASH = 1.0                # motor.backlash: how far the joint trails the drive in the direction of
                                     # travel (the play is twice that, edge to edge). Read as the OFFSET
                                     # because every completed move tonight landed 1.2-1.6 units short,
                                     # whatever its size (+25.3 -> 23.7, +11.4 -> 10.2, +41.7 -> 40.2):
                                     # offset 1.0 + deadband 0.2 = 1.2; as a total width it would be 0.7.
VENDOR_DEADBAND = 0.2                # motor.deadband: no drive inside this error
VENDOR_QUANTISATION = 0.05           # motor.position_quantization of a read

# ----------------------------------------------------------------------------- measured on the arm
TAU_DEFAULT_S = 0.12                 # the free parameter's starting point: the middle of the measured
                                     # 100-150 ms servo lag behind the clip timeline (beattrace). The
                                     # vendor's 42 ms is what THEIR simulator assumes, not what the arm did.
MEASURED_LAG_S = (0.10, 0.15)
STOP_YAW_MIN = -4.5                  # the left yaw stop: leftward commands stopped at -4.0..-5.3
MEASURED_STOP_RANGE = (-5.3, -4.0)   # (direct POSTs; the fastest clip hit went to -6.1)
CALIBRATION_MIN, CALIBRATION_MAX = -100.0, 100.0   # lelamp-calibration.json range_min..range_max map here
MEASURED_DELIVERY = {"base_yaw": 0.48, "base_pitch": 0.88, "elbow_pitch": 0.75,
                     "wrist_roll": 0.48, "wrist_pitch": 0.75}   # beat-rate ratios, for the fit; NOT applied
MEASURED_SLOW_RAMP = (40.0, 4.0, 38.6)              # units asked, seconds, units delivered (96 %)
MEASURED_DIRECT_DEFICIT = (1.2, 1.6)                # units short on 3 s direct moves, whatever the size


def default_stops(yaw_min: float = STOP_YAW_MIN) -> dict[str, tuple[float, float]]:
    """Joint stops as the arm has them tonight: the calibration's +-100 everywhere, except the unexplained
    floor on base_yaw. `default_stops(CALIBRATION_MIN)` is the arm once the stop is fixed."""
    stops = {j: (CALIBRATION_MIN, CALIBRATION_MAX) for j in JOINTS}
    stops["base_yaw"] = (float(yaw_min), CALIBRATION_MAX)
    return stops


def calibration_stops() -> dict[str, tuple[float, float]]:
    """The stops the calibration file promises (-100..+100 on every joint): the arm with the yaw stop fixed."""
    return default_stops(CALIBRATION_MIN)


def _per_joint(value, default: float) -> np.ndarray:
    """A scalar for every joint, or a mapping joint -> value with `default` for the joints it leaves out
    (so the Validate phase can fit one joint's tau without spelling out the other four)."""
    if isinstance(value, Mapping):
        unknown = sorted(set(value) - set(JOINTS))
        if unknown:
            raise ValueError(f"unknown joints {unknown}; joints are {JOINTS}")
        return np.array([float(value.get(j, default)) for j in JOINTS], dtype=float)
    return np.full(NJ, float(value), dtype=float)


@dataclass(frozen=True)
class ServoParams:
    """Every knob of the model. Defaults are the vendor's simulation.yaml where the arm agrees with it and
    the measurement where it does not (tau, the yaw stop). Scalars apply to every joint; a mapping sets
    some joints and leaves the rest at the default."""
    tau_s: float | Mapping[str, float] = TAU_DEFAULT_S            # FREE: first-order lag, to be fitted
    max_velocity: float | Mapping[str, float] = VENDOR_MAX_VELOCITY
    max_acceleration: float | Mapping[str, float] = VENDOR_MAX_ACCELERATION
    backlash: float = VENDOR_BACKLASH
    deadband: float = VENDOR_DEADBAND
    quantisation: float = VENDOR_QUANTISATION
    command_latency_s: float = COMMAND_LATENCY_S
    read_latency_s: float = READ_LATENCY_S
    stops: Mapping[str, tuple[float, float]] = field(default_factory=default_stops)
    delivery: Mapping[str, float] | None = None   # optional per-joint ratio override for fast motion
    delivery_slow_tau_s: float = 1.0              # motion slower than this is "slow" and delivers in full:
                                                  # the runtime's own moves take >= 1 s (safe_motion.py
                                                  # target_plan) and 3-4 s tracking moves landed 94-96 %
    play0: float = 0.0                            # where the drive sits in the play at the start, -1..+1 of
                                                  # `backlash`: 0 centred; -1 = the arm arrived from the right
                                                  # (the drive at the left edge), so the next rightward move
                                                  # spends 2 x backlash of drive travel before the joint moves.
                                                  # A transient only: the completed move lands backlash +
                                                  # deadband short whatever play0 was.

    def __post_init__(self):
        tau = self.tau()
        if (tau < DT).any():
            # x += (dt/tau)(g - x) is monotone only for tau >= dt: a smaller tau would make the discrete
            # loop overshoot, which is an artefact of the 100 Hz tick, not a property of the servo.
            raise ValueError(f"tau_s must be >= {DT} s (one tick); got {dict(zip(JOINTS, tau))}")
        if (self.vmax() <= 0).any() or (self.amax() <= 0).any():
            raise ValueError("max_velocity and max_acceleration must be positive")
        if self.backlash < 0 or self.deadband < 0 or self.quantisation <= 0:
            raise ValueError("backlash and deadband must be >= 0, quantisation > 0")
        if self.command_latency_s < 0 or self.read_latency_s < 0:
            raise ValueError("latencies must be >= 0")
        if not -1.0 <= self.play0 <= 1.0:
            raise ValueError("play0 is a fraction of half the backlash, -1..1")
        if self.delivery_slow_tau_s <= 0:
            raise ValueError("delivery_slow_tau_s must be positive")
        lo, hi = self.stop_arrays()
        if (lo >= hi).any():
            raise ValueError(f"every stop needs lo < hi; got {dict(self.stops)}")
        r = self.ratio()
        if ((r <= 0) | (r > 1)).any():
            raise ValueError("delivery ratios are in (0, 1]")

    def tau(self) -> np.ndarray:
        return _per_joint(self.tau_s, TAU_DEFAULT_S)

    def vmax(self) -> np.ndarray:
        return _per_joint(self.max_velocity, VENDOR_MAX_VELOCITY)

    def amax(self) -> np.ndarray:
        return _per_joint(self.max_acceleration, VENDOR_MAX_ACCELERATION)

    def ratio(self) -> np.ndarray:
        return _per_joint(self.delivery or {}, 1.0)

    def stop_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        base = default_stops()
        lo = np.array([float(self.stops.get(j, base[j])[0]) for j in JOINTS])
        hi = np.array([float(self.stops.get(j, base[j])[1]) for j in JOINTS])
        return lo, hi


def vendor_motor_params(path: Path) -> dict[str, float]:
    """The scalar lines of simulation.yaml's `motor:` block, as the runtime ships them (there is no PyYAML
    in the test environment and the block is flat `key: value` lines, so a line scanner is enough)."""
    out: dict[str, float] = {}
    in_motor = False
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0:
            in_motor = line.startswith("motor:")
            continue
        if not in_motor:
            continue
        m = re.match(r"\s*([A-Za-z_][\w]*):\s*([-+]?\d+(?:\.\d+)?)\s*$", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


# ----------------------------------------------------------------------------- input helpers
def goals_from_rows(rows: Sequence[Mapping[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    """(t_s, goals) from clip-style rows: `timestamp` in seconds and `<joint>.pos` columns (the pack's CSV
    layout, csv_transformer.py). Use it for what the executor streams, which for a clip is the plan
    AFTER the runtime's fusion, not the file."""
    t = np.array([float(r["timestamp"]) for r in rows], dtype=float)
    g = np.array([[float(r[f"{j}.pos"]) for j in JOINTS] for r in rows], dtype=float)
    return t, g


def _goal_matrix(goals, n: int) -> np.ndarray:
    if isinstance(goals, Mapping):
        g = np.column_stack([np.asarray(goals[j], dtype=float) for j in JOINTS])
    else:
        g = np.asarray(goals, dtype=float)
    if g.ndim == 1:
        g = g[None, :]
    if g.shape != (n, NJ):
        raise ValueError(f"goals must be ({n}, {NJ}) in the order {JOINTS}; got {g.shape}")
    if not np.isfinite(g).all():
        raise ValueError("goals must be finite")
    return g


# ----------------------------------------------------------------------------- the trace
@dataclass
class ServoTrace:
    """One simulated run. Rows are ticks at UPDATE_HZ starting at the first write; columns are JOINTS."""
    t: np.ndarray            # (K,) seconds
    goal: np.ndarray         # (K, J) the goal in force at the servo (after the command latency)
    position: np.ndarray     # (K, J) the joint's true position
    measured: np.ndarray     # (K, J) what a bus read at that tick returns
    velocity: np.ndarray     # (K, J) the drive's velocity, units/s
    banging: np.ndarray      # (K, J) bool: pressed against a stop with the goal beyond it
    params: ServoParams

    @property
    def any_banging(self) -> np.ndarray:
        """(K,) bool: some joint is pressed against a stop this tick."""
        return self.banging.any(axis=1)

    @property
    def tracking_error(self) -> np.ndarray:
        return self.goal - self.measured

    @staticmethod
    def joint(name: str) -> int:
        return JOINTS.index(name)

    def column(self, name: str, which: str = "measured") -> np.ndarray:
        return getattr(self, which)[:, self.joint(name)]

    def final(self) -> dict[str, float]:
        return {j: float(self.measured[-1, i]) for i, j in enumerate(JOINTS)}

    def banging_seconds(self) -> dict[str, float]:
        return {j: float(self.banging[:, i].sum() * DT) for i, j in enumerate(JOINTS)}

    def beattrace_ratio(self, name: str, t_cmd: np.ndarray | None = None,
                        cmd: np.ndarray | None = None) -> tuple[float, float]:
        """(ratio, lag_s) the way analysis/beattrace.py scores the real arm: mean-removed command and
        measurement, the lag in 0..1.2 s (10 ms steps) that minimises their RMS difference, then the ratio
        of standard deviations at that lag. Defaults to the trace's own goal stream as the command."""
        i = self.joint(name)
        t_cmd = self.t if t_cmd is None else np.asarray(t_cmd, dtype=float)
        cmd = self.goal[:, i] if cmd is None else np.asarray(cmd, dtype=float)
        c = cmd - cmd.mean()
        g = self.measured[:, i] - self.measured[:, i].mean()
        best = (float("inf"), 0.0)
        for lag in np.arange(0.0, 1.2, 0.01):
            ci = np.interp(self.t - lag, t_cmd, c, left=c[0], right=c[-1])
            e = float(np.sqrt(np.mean((g - ci) ** 2)))
            if e < best[0]:
                best = (e, float(lag))
        ci = np.interp(self.t - best[1], t_cmd, c, left=c[0], right=c[-1])
        return float(np.std(g) / max(1e-6, float(np.std(ci)))), best[1]

    def summary(self) -> dict[str, dict[str, float]]:
        out = {}
        for i, j in enumerate(JOINTS):
            out[j] = {"final": float(self.measured[-1, i]), "min": float(self.measured[:, i].min()),
                      "max": float(self.measured[:, i].max()),
                      "max_error": float(np.abs(self.goal[:, i] - self.measured[:, i]).max()),
                      "banging_s": float(self.banging[:, i].sum() * DT)}
        return out


# ----------------------------------------------------------------------------- the model
def simulate(t_s, goals, *, start=None, params: ServoParams | None = None, tail_s: float = 0.5) -> ServoTrace:
    """Run the goal stream through the servos.

    t_s     write times in seconds, non-decreasing (the executor's actual write instants; a 30 fps clip
            is every 33.3 ms, and a late frame is simply a later write)
    goals   (N, 5) in JOINTS order, or a mapping joint -> (N,) -- what each write asked for
    start   where the arm is when the first write lands (default: the first goal, i.e. no entry move)
    tail_s  how long to keep simulating after the last write while the servo converges on its last goal
            (the executor writes nothing more; 0.5 s is four default time constants)
    """
    p = params or ServoParams()
    t_s = np.asarray(t_s, dtype=float).ravel()
    if t_s.size == 0:
        raise ValueError("at least one write is needed")
    if (np.diff(t_s) < 0).any() or not np.isfinite(t_s).all():
        raise ValueError("write times must be finite and non-decreasing")
    g_writes = _goal_matrix(goals, t_s.size)
    start = g_writes[0].copy() if start is None else np.asarray(start, dtype=float).reshape(NJ)
    if tail_s < 0:
        raise ValueError("tail_s must be >= 0")

    t0 = float(t_s[0])
    K = int(math.floor((float(t_s[-1]) + tail_s - t0) / DT + 1e-9)) + 1
    t = t0 + np.arange(K) * DT

    # 1. the goal in force: the last write at or before (t - command_latency); before the first one has
    #    landed the servo holds where it is (torque is on; its previous goal is where it sits).
    idx = np.searchsorted(t_s, t - p.command_latency_s + 1e-9, side="right") - 1
    goal = np.where(idx[:, None] >= 0, g_writes[np.maximum(idx, 0)], start[None, :])

    tau, vmax, amax = p.tau(), p.vmax(), p.amax()
    ratio = p.ratio()
    override = bool((ratio != 1.0).any())
    lo, hi = p.stop_arrays()
    play = p.backlash                        # the joint trails the drive by up to this (see VENDOR_BACKLASH)
    dv_max = amax * DT

    # A servo switched on beyond its limit drives to the limit and parks (yaw_limits.py, seen 2026-09-20),
    # so the state starts inside the stops whatever `start` says.
    joint = np.clip(start, lo, hi)
    drive = joint + p.play0 * play           # play0 = -1: the drive sits at the left edge of the play
    vel = np.zeros(NJ)
    slow = joint.copy()                      # the slow part of the goal, for the delivery override

    position = np.empty((K, NJ))
    velocity = np.empty((K, NJ))
    banging = np.zeros((K, NJ), dtype=bool)

    for k in range(K):
        g = goal[k]
        if override:
            # 'delivery': the joint chases only `ratio` of the FAST part of its goal; the slow part (a
            # first-order low-pass of the goal) is chased in full, so a 4 s tracking move still lands.
            slow = slow + (DT / p.delivery_slow_tau_s) * (g - slow)
            g = slow + ratio * (g - slow)

        # 2. the loop's velocity demand: a first-order lag with deadband, clipped to the velocity ceiling
        err = g - drive
        want = np.where(np.abs(err) <= p.deadband, 0.0, err / tau)
        want = np.clip(want, -vmax, vmax)

        # 3. spin-up is torque-limited (max_acceleration); braking and stopping follow the demand at once,
        #    and a reversal brakes to zero first and then spins up from there.
        same = np.sign(want) == np.sign(vel)
        base = np.where(same, vel, 0.0)
        grown = base + np.clip(want - base, -dv_max, dv_max)
        vel = np.where(same & (np.abs(want) <= np.abs(vel)), want, grown)

        # 4. the drive moves ...
        drive = drive + vel * DT
        # 5. ... and the joint follows it through the play
        s = np.clip(drive - joint, -play, play)
        joint = drive - s

        # 6. the stops act on the joint (a mechanical stop, or the servo's own limit on its output encoder --
        #    either way the output stays put). The drive is left wound up against the play's edge, the
        #    momentum is gone, and the joint is "banging" while the goal it chases is still beyond the stop.
        below = joint < lo
        above = joint > hi
        joint = np.clip(joint, lo, hi)
        drive = np.where(below, lo - play, np.where(above, hi + play, drive))
        vel = np.where(below | above, 0.0, vel)
        pressed_lo = (joint <= lo) & (g < lo - p.deadband)
        pressed_hi = (joint >= hi) & (g > hi + p.deadband)
        banging[k] = pressed_lo | pressed_hi

        position[k] = joint
        velocity[k] = vel

    # 7. a read returns the position read_latency ago (linear between ticks), quantised
    frac = min(1.0, p.read_latency_s / DT)
    previous = np.vstack([position[:1], position[:-1]])
    delayed = position - frac * (position - previous)
    measured = np.round(delayed / p.quantisation) * p.quantisation

    return ServoTrace(t=t, goal=goal, position=position, measured=measured, velocity=velocity,
                      banging=banging, params=p)
