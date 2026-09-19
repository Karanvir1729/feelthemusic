"""The lamp's behaviour: look around like it is curious, find a face, turn to it, and stay with that person.

HeadTracker is the client that would run beside the vendor runtime. It sees only what a real client sees:
  * face detections from the head camera (contract.Detection), after the camera pipeline's latency;
  * the measured joint pose (what the SDK's /joints route reports);
  * whether its SDK action is still running, and the outcome of each action.
It answers with contract.Command objects, a motion.move or a clip.play; the run loop turns them into SDK
calls. It has no other way to move the arm, and it never reads Detection.person_id (ground truth, kept for
scoring): people are told apart by where they are in 3D, as a real tracker must.

States
  SEARCH   nobody locked. A lively look-around: fixations on places where a head could be (seated and
           standing heights, on an arc in front of the lamp), in a varied order, with pauses and small
           glances; never a mechanical sweep, never toward the table or behind the lamp.
  ACQUIRE  a face was seen in N of the last M frames: turn to it, in calm steps if the turn is large.
  LOCK     facing the person: stay on the same person (3D association, not identity), aim where they will
           be when the move ends, ignore small errors, keep the head up against the vendor idle, and
           optionally show a little life (tiny slow nods and tilts).
  LOST     the locked face has not been seen for a while: peek toward where it was heading, hold, then SEARCH.
  HOLD     three actions in a row failed: stop commanding until reset(). The lamp is borrowed.

Strategies for ACQUIRE and LOCK (twin motion spec, section 13):
  settle   one motion.move, wait for its outcome, look again. The recommended one. Its one exception: a
           keep-alive (a hold that makes no progress) is pre-empted when the person moves far (urgent_deg).
  preempt  send a newer move every preempt_interval_s even while one runs. The spec's negative control:
           every pre-emption restarts a 2 s plan from rest, so it moves less and costs more.
  clip     a short clip.play along the predicted path. Every clip starts with a >= 2 s entry move, so it
           is no faster than settle and costs an upload per clip; modelled to show that.

Load error: gravity holds the arm below its goals and no re-send of the same goal fixes that, so the
tracker learns, after each successful aiming move, the angle between where it aimed and where the camera
settled, and aims that much further (sag_compensation). It changes targets only, never adds a route.

Every pose the tracker commands comes from kin.look_at and passes kin.check (and the straight joint-space
path the vendor planner takes from the measured pose passes too; the vendor checks only head against base,
the table is the client's job). Clip frames between look_at knots are joint-space interpolation, as the
vendor's own 30 Hz resampling is, and every frame is checked.

The twin is design evidence, not hardware approval.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from twin import motion as M
from twin.contract import JOINTS, Command, Detection, Kinematics, Units

SEARCH, ACQUIRE, LOCK, LOST, HOLD = "SEARCH", "ACQUIRE", "LOCK", "LOST", "HOLD"
STATES = (SEARCH, ACQUIRE, LOCK, LOST, HOLD)
STRATEGIES = ("settle", "preempt", "clip")
_EPS = 1e-6
_Z = np.array([0.0, 0.0, 1.0])


@dataclass
class TrackerConfig:
    """Every number says where it comes from. Seconds, metres, degrees, and SDK joint units."""

    # ---------------------------------------------------------------- camera and detector
    # Field of view: measured on the lamp 2026-09-19 (derived from two on-lamp steps at neutral: base_yaw
    # +7.4 units moved the picture -0.083 widths, wrist_pitch +8.1 units moved it -0.085 heights).
    hfov_deg: float = 61.0
    vfov_deg: float = 44.0
    # ASSUMPTION: perception.py's detector rate. Only used to turn "N of the last M frames" into a window.
    camera_fps: float = 10.0
    # lamp/follow.py:39 on branch karancodex/lamp-tracking (the follower that ran on the lamp). Real faces
    # vary; perception.py draws a box 2 x 0.09 m wide, so this tracker under-estimates depth by about 17%.
    face_width_m: float = 0.15
    # Distance clamp for a face: lamp/follow.py (branch karancodex/lamp-tracking) and the workflow brief.
    min_range_m: float = 0.30
    max_range_m: float = 2.5
    # A head resting on the table has its centre one head radius above it (contract.Person default 0.09 m),
    # so no ray is followed below that height.
    min_target_z_m: float = 0.09
    # research report head-tracking.md:250: accept score >= 0.7, acquire only after 3 detections.
    min_confidence: float = 0.7
    debounce_n: int = 3
    debounce_m: int = 5                 # ASSUMPTION: 3 of the last 5 frames tolerates one missed or blurred frame

    # ---------------------------------------------------------------- tracks (all ASSUMPTION unless cited)
    gate_m: float = 0.25                # a detection joins a track within this distance of its prediction
    gate_speed_m_s: float = 1.0         # ... plus this much per second unseen (world.py walks at 0.7-0.9 m/s)
    gate_max_m: float = 0.6             # ... never more: world.py's second person stands 0.7+ m from the first
    alpha: float = 0.5                  # position gain per detection: research report head-tracking.md:249 (EMA 0.5)
    beta: float = 0.05                  # velocity gain per detection: ASSUMPTION (about a 2 s velocity memory at 10 Hz)
    max_speed_m_s: float = 2.0          # faster than anyone walks at a table
    still_speed_m_s: float = 0.12       # below this, do not extrapolate: seated sway is about 0.03 m/s (world.py)
    reset_gap_s: float = 0.6            # unseen longer than this: forget the velocity when the face returns
    max_predict_s: float = 2.5          # never extrapolate further ahead than this ...
    predict_max_m: float = 0.6          # ... or further away than this
    fresh_s: float = 0.5                # a track counts as "in view" if seen this recently
    track_drop_s: float = 1.0           # unlocked tracks unseen this long are forgotten
    max_tracks: int = 8                 # bounds the work; a table has a few faces, not a crowd

    # ---------------------------------------------------------------- aiming
    # Deadband: fork sdk-spec safe-motion.md section 9 says 4-5 units (the runtime's own yaw tolerance is 2),
    # about 3-3.7 deg at about 0.74 deg per yaw unit; 4 deg is an ASSUMPTION inside that.
    deadband_deg: float = 4.0
    lock_enter_deg: float = 8.0         # ASSUMPTION: ACQUIRE becomes LOCK once the requested pose is this close
    lock_exit_deg: float = 20.0         # ASSUMPTION: LOCK becomes ACQUIRE past this (half the vertical FOV: 22 deg)
    # Step limit: a big turn becomes several calm steps. lamp/follow.py --max-step default 20 units (branch
    # karancodex/lamp-tracking); the motion spec asks for a sweep over 15-40. 15 deg is about 20 yaw units.
    max_step_units: float = 20.0
    max_step_deg: float = 15.0
    max_aim_error_deg: float = 10.0     # ASSUMPTION: a look_at result must face its point this well to be used
    drift_units: float = 12.0           # re-assert a pose the arm left by more than this (vendor's widest settle
                                        # tolerance is 10 on the elbow: vendor source safety.yaml:12-16, motion spec 7)
    # Load error. Gravity holds the loaded joints below their goals (motion spec 11; the size of the sag is an
    # ASSUMPTION in the twin and unmeasured on the lamp), and re-sending the same goal cannot fix it: the
    # planner re-plans to the same goal. So after each successful aiming move the tracker measures the angle
    # between where it aimed and where the camera settled, low-passes it, and aims that much further next time.
    # Only the targets of moves it sends anyway change: no extra moves, nothing outside the SDK. ASSUMPTION values.
    sag_compensation: bool = True
    sag_gain: float = 0.5
    # ASSUMPTION: 1.5 x the camera droop the vendor tolerances allow. With every loaded joint sagging the way
    # gravity pulls it (twin/model.py GravitySag) the head droops about 6.7 deg at the "assumed" sag and
    # 10.6 deg at the vendor tolerances; an 8 deg cap could not follow the second.
    max_bias_deg: float = 16.0
    bias_refresh_deg: float = 1.0       # re-solve the held pose once the learnt bias has moved this much
    # A settle "failed" whose arm stopped close to its target is gravity, not a fault. The SDK's failure
    # carries every joint's error and tolerance (position_errors, position_tolerances: vendor source
    # control/runtime.py:43-67, passed on by sdk_gateway/service.py:740-753). When every error is within
    # reached_factor x its tolerance, the move counts as reached: no failure toward HOLD, no back-off, and its
    # load error is learnt as after a success. ASSUMPTION: 2 x, above the 1.5 x the sag sweep goes to; a stall
    # against an obstacle or a person leaves more.
    reached_factor: float = 2.0
    # Keep-alive: with the vendor idle on (the default, and the SDK cannot turn it off), a successful move is
    # followed by a crouched idle within about 0.2 s, so the next move goes out at once (motion spec 8, 14).
    keep_alive: bool = True
    # POST to the first waypoint of a new plan. None: the client-side replica of the SDK's cost model
    # (twin/motion.py move_start_delay_s: HTTP, pose reads, two collision passes over 61 rows at an ASSUMED Pi
    # speed, the 2-frame pre-roll), about 0.23 s. Measure it on the lamp.
    sdk_preroll_s: float | None = None
    # Prediction horizon for settle: one plan (2.0 s) plus the POST-to-plan overhead. None: 2.0 + sdk_preroll_s.
    settle_predict_s: float | None = None
    # Upload start to the first waypoint of a lock clip's entry move. None: twin/motion.py clip_start_delay_s for
    # a clip of clip_path_s (upload, two validations, the re-plan over entry + clip), about 0.45 s.
    clip_start_s: float | None = None
    # What the client subtracts from a frame's stamp to get its exposure time. 0: the client does not know the
    # camera queue's age (the blink test that would measure it has not been run), so it takes the stamp as
    # the exposure. ASSUMPTION.
    stamp_lag_s: float = 0.0
    # Pre-empt: 1 s is the least bad interval in the motion spec's table (section 6: 37% per interval, 60/min).
    preempt_interval_s: float = 1.0
    preempt_min_change_deg: float = 2.0     # ASSUMPTION: only pre-empt for a target that really moved
    # Settle, urgent re-aim: a keep-alive only re-holds the same pose, so pre-empting it loses nothing but
    # its bob (motion spec 6). When the target has moved this far while a hold runs (someone stands up), the
    # turn goes out at once instead of up to 2.2 s later. Turns are never pre-empted. None turns it off.
    urgent_deg: float | None = 10.0     # ASSUMPTION
    # Clip strategy: every clip.play starts with an entry move of at least 2.0 s (vendor source
    # safe_motion.py:116-117 via motion spec 9). Knot spacing and path length are ASSUMPTION.
    clip_entry_s: float = 2.0
    clip_knot_s: float = 0.5
    clip_path_s: float = 2.0
    clip_fps: float = 30.0              # vendor resamples clips to 30 Hz (config/default.yaml:104-105, motion spec 9)
    # ASSUMPTION: calm. A motion.move peaks at <= 135 units/s, a clip may reach 300 (motion spec 4, 9).
    clip_max_speed_u_s: float = 60.0

    # ---------------------------------------------------------------- alive while locked (optional)
    alive: bool = True
    alive_every_s: tuple = (6.0, 10.0)  # ASSUMPTION: a small gesture every 6-10 s. Not on beats: every SDK move lasts
                                        # >= 2 s, so a nod cannot land on a 120 BPM beat (motion spec 4).
    alive_offset_m: float = 0.04        # ASSUMPTION: 4 cm at the face, about 2.5 deg at 0.9 m: a nod, not a look away

    # ---------------------------------------------------------------- loss (all ASSUMPTION unless cited)
    lost_after_s: float = 1.5           # ASSUMPTION: world.py's listeners look away for 1-2.5 s; research 4.5 holds 1 s
    lost_timeout_s: float = 4.0         # research report head-tracking.md section 4.5: scan again after 4 s
    lost_peek_after_s: float = 0.0      # ASSUMPTION: peek at once; any gap lets the vendor idle pull the head down
    lost_peek_ahead_s: float = 1.5      # ASSUMPTION: look where a moving person would be this much later
    lost_peek_max_m: float = 0.4        # ASSUMPTION: never peek further than this from the last sighting
    lost_peek_side_m: float = 0.2       # ASSUMPTION: a still person who vanished: glance this far to one side

    # ---------------------------------------------------------------- search (all ASSUMPTION unless cited)
    # One long clip.play covers up to 600 s for one rate slot and keeps the vendor idle out (motion spec 9, 14).
    # "moves" chains motion.move fixations instead (about 26 per minute).
    search_via: str = "clip"
    # Yaw within about +-60 units, about +-44 deg (fork sdk-spec safe-motion.md section 9 item 3). The uneven
    # spacing is an ASSUMPTION, so no two visits look like steps of a sweep.
    search_azimuths_deg: tuple = (-45.0, -35.0, -24.0, -13.0, -4.0, 5.0, 14.0, 25.0, 36.0, 45.0)
    # (head height above the table, distance from the lamp): seated and standing heads in world.py's room
    # model (seated 0.42-0.50 m at 0.75-0.95 m, standing 0.95-1.05 m about 0.25 m further back). ASSUMPTION.
    search_rows: tuple = ((0.46, 0.95), (1.00, 1.40))
    search_jitter_deg: float = 3.0      # ASSUMPTION: each visit lands a little differently
    search_jitter_m: float = 0.04
    search_dwell_s: tuple = (0.6, 1.5)  # fixation length: motion spec 14 (ASSUMPTION values there too)
    search_peak_speed_u_s: tuple = (35.0, 60.0)  # ASSUMPTION: calm saccades, far below the 135 a move may reach
    search_min_saccade_s: float = 0.5   # ASSUMPTION: even a small jump takes half a second
    search_step_deg: tuple = (6.0, 22.0, 45.0)   # ASSUMPTION: smallest, preferred and largest jump between fixations
    search_step_spread_deg: float = 14.0    # ASSUMPTION: how strongly the preferred jump is preferred
    search_max_same_direction: int = 2  # ASSUMPTION: a third jump the same way is refused: no sweeps
    search_memory: int = 3              # ASSUMPTION: do not revisit the last 3 places
    search_novel_k: int = 4             # ASSUMPTION: choose among the 4 reachable places looked at least recently
    search_row_switch_weight: float = 0.7   # ASSUMPTION: switching between seated and standing height a bit less often
    search_glance_p: float = 0.35       # ASSUMPTION: a small curious glance during about a third of fixations
    search_glance_m: tuple = (0.05, 0.12)       # ASSUMPTION: 3-7 deg aside at about 1 m
    search_glance_hold_s: tuple = (0.25, 0.6)   # ASSUMPTION
    search_glance_move_s: tuple = (0.35, 0.5)   # ASSUMPTION
    # ASSUMPTION; the vendor caps a clip at 600 s (safe_motion.py:13, motion spec 2). Short enough that its
    # upload and admission (every row validated twice, about 1.4 s at the ASSUMED Pi speed) end while the
    # take-over move still runs: a 40 s clip takes about 3 s, long enough for the vendor idle to crouch the arm
    # into its base.
    search_clip_s: float = 20.0
    # SEARCH with nothing of ours running (the start, or after a loss): take the arm from the vendor idle with
    # one quick motion.move (admitted in about 70 ms) to the first place to look, then upload the look-around
    # clip behind it; its clip.play pre-empts the move. Without it the idle plays until the clip is admitted.
    search_take_over: bool = True
    # A running clip (look-around or lock) is replaced this long before it ends, by a clip uploaded early
    # enough to be admitted then: between a clip's end and the next admission the vendor idle would pull the
    # head down (it resumes 16 ms after every success). ASSUMPTION: covers jitter of the admission lock.
    pipeline_margin_s: float = 0.3

    # ---------------------------------------------------------------- budget and failures
    # The SDK counts every action per session over a sliding 60 s window. The vendor default is 30/min
    # (config/default.yaml:451); the lamp's .env says 120/min, not in effect until the runtime restarts
    # (research report pi-internet-deps.md:231). Light shares the session. 40/min for motion is an ASSUMPTION
    # that leaves 80/min for light.glow at 120/min; at 30/min use <= 26 (settle-chained tracking alone).
    sdk_rate_limit_per_min: int = 120
    max_commands_per_min: int = 40
    min_command_gap_s: float = 0.3      # ASSUMPTION
    busy_grace_s: float = 0.5           # ASSUMPTION: no outcome and not busy this long after a POST: nothing runs
    backoff_s: float = 5.0              # fork sdk-spec safe-motion.md section 9: back off at least 5 s
    rate_limited_backoff_s: float = 60.0    # same section: "429: pause 60 s"
    max_failures: int = 3               # workflow brief: three failures in a row -> HOLD
    hold_release_s: float | None = None     # None: HOLD until reset(); a number releases it after that long
    retry_s: float = 0.25               # ASSUMPTION: after a refused pose, do not re-solve IK every step

    # ---------------------------------------------------------------- where the head may point
    min_facing_elev_deg: float = -30.0  # ASSUMPTION: never face the table
    max_facing_az_deg: float = 80.0     # ASSUMPTION: never face beside or behind the lamp
    max_target_az_deg: float = 75.0     # ASSUMPTION: targets further round are clamped to this bearing
    path_samples: int = 6               # ASSUMPTION: points checked on the straight joint-space line the vendor plans
                                        # from the measured pose (vendor source safe_motion.py:109-135, motion spec 4)
    pose_history_s: float = 3.0         # poses kept to find the camera at capture (latency 0.15 s: perception.py)
    look_at_kw: dict = field(default_factory=dict)  # passed through to kin.look_at
    seed: int = 0                       # the search's random order and the alive gestures are seeded

    def __post_init__(self):
        if self.sdk_preroll_s is None:
            self.sdk_preroll_s = M.move_start_delay_s()
        if self.settle_predict_s is None:
            self.settle_predict_s = M.plan_seconds(0.0) + self.sdk_preroll_s
        if self.clip_start_s is None:
            self.clip_start_s = M.clip_start_delay_s(int(round(self.clip_path_s * self.clip_fps)) + 1)
        if self.search_via not in ("clip", "moves"):
            raise ValueError(f"search_via must be 'clip' or 'moves', not {self.search_via!r}")
        if not 0 < self.max_commands_per_min < self.sdk_rate_limit_per_min:
            raise ValueError("max_commands_per_min must stay below the SDK rate limit")
        if not 1 <= self.debounce_n <= self.debounce_m:
            raise ValueError("debounce needs 1 <= N <= M")


# ------------------------------------------------------------------ small geometry helpers
def _ease(u: float) -> float:
    """The vendor planner's easing, 10u^3 - 15u^4 + 6u^5 (vendor source safe_motion.py:114, 124)."""
    u = min(1.0, max(0.0, u))
    return u * u * u * (u * (6.0 * u - 15.0) + 10.0)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two directions. atan2(|a x b|, a.b) is exact at 0 and at 180 deg; a residual built from the
    cross product alone is zero at 180 deg (the known bug in the lamp-sdk branch's spatial.py)."""
    return math.degrees(math.atan2(float(np.linalg.norm(np.cross(a, b))), float(np.dot(a, b))))


def _az_el(v: np.ndarray) -> tuple[float, float]:
    """Bearing from +y toward +x and elevation above the table plane, in degrees."""
    return math.degrees(math.atan2(v[0], v[1])), math.degrees(math.atan2(v[2], math.hypot(v[0], v[1])))


def _direction(az_deg: float, el_deg: float) -> np.ndarray:
    az, el = math.radians(az_deg), math.radians(el_deg)
    return np.array([math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)])


def _slerp(a: np.ndarray, b: np.ndarray, frac: float) -> np.ndarray:
    a, b = _unit(a), _unit(b)
    omega = math.acos(float(np.clip(a @ b, -1.0, 1.0)))
    if omega < 1e-6:
        return b
    if omega > math.pi - 1e-3:                       # opposite: turn about the vertical
        c, s = math.cos(math.pi * frac), math.sin(math.pi * frac)
        return np.array([c * a[0] - s * a[1], s * a[0] + c * a[1], a[2]])
    return (math.sin((1 - frac) * omega) * a + math.sin(frac * omega) * b) / math.sin(omega)


@dataclass
class _Track:
    """One face in 3D, filtered with a constant-velocity alpha-beta filter on capture times."""
    id: str
    pos: np.ndarray
    vel: np.ndarray
    t: float                                  # capture time of the detection that last updated it
    hits: deque = field(default_factory=lambda: deque(maxlen=32))
    n: int = 1
    scoring_person_id: str = ""               # ground truth of the last detection, passed through for the report
                                              # only; no decision reads it


class HeadTracker:
    """The lamp's head behaviour. Call update() every simulation step; it returns at most one Command.

    update(t, detections, measured_units, motion_busy, last_outcome):
      detections     the Detections delivered since the last call (perception's poll(t))
      measured_units the measured joint pose now (all five joints, SDK units)
      motion_busy    an SDK action the tracker started is admitted and not finished
      last_outcome   the outcome of one action that finished since the last call: "succeeded", "failed",
                     "canceled", "rejected" (also for a refused POST), or None. A refusal mentioning a
                     rate limit ("429", "rate") pauses for rate_limited_backoff_s.

    Read-outs: state, locked_id (the tracker's own track id), trace (this step, for the report), stats
    (counters), window_used(t) (commands in the last 60 s), target_point(detection, units). reset(t) leaves
    HOLD; nothing else does unless hold_release_s is set.
    """

    def __init__(self, kin: Kinematics, *, strategy: str = "settle", config: TrackerConfig | None = None):
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}, not {strategy!r}")
        self.kin, self.strategy = kin, strategy
        self.cfg = config if config is not None else TrackerConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self._fx = 0.5 / math.tan(math.radians(self.cfg.hfov_deg) / 2)   # focal length in picture widths
        self._fy = 0.5 / math.tan(math.radians(self.cfg.vfov_deg) / 2)   # focal length in picture heights

        self._state = SEARCH
        self._tracks: list[_Track] = []
        self._track_count = 0
        self._locked: _Track | None = None
        self._poses: deque = deque()               # (t, measured pose as a vector), for the pose at capture time

        # what has been sent and what came back
        self._issued: deque = deque()              # times of our commands in the last 60 s
        self._last_issue_t = -math.inf
        self._last_issue_state = SEARCH
        self._pending = 0                          # commands whose outcome has not come back
        self._expected_cancels = 0                 # outcomes "canceled" we caused by pre-empting our own action
        self._failures = 0                         # unexpected non-successes in a row
        self._backoff_until = -math.inf
        self._next_try_t = -math.inf
        self._hold_since = 0.0
        self._must_preempt = False                 # the first ACQUIRE move pre-empts a running search action

        # lock bookkeeping
        self._sent_base: Units | None = None       # the last aiming pose asked for (without an alive gesture)
        self._sent_point: np.ndarray | None = None # the point it was meant to face (before the load bias)
        self._base_point: np.ndarray | None = None # the point look_at was given for it (after the bias)
        self._toward: np.ndarray | None = None     # the point the last successful _pose_toward looked at
        self._cmd_point: np.ndarray | None = None  # the look_at point of the command being built this step
        self._last_point: np.ndarray | None = None # ... and of the last command sent
        self._cmd_kind = "turn"                    # what the command being built does: turn | hold | search
        self._running: str | None = None           # ... and what the last command sent does
        self._bias = np.zeros(2)                   # load bias: (bearing, elevation) degrees to aim beyond a point
        self._base_bias = np.zeros(2)              # the bias the held pose was solved with
        self._alive_out = False                    # the last command was an alive gesture
        self._next_alive_t = math.inf
        self._acquire_successes = 0
        self._lost_since = 0.0
        self._lost_seen_t = 0.0
        self._peeked = False

        # search bookkeeping
        self._anchors: list[dict] | None = None
        self._visits: dict[int, int] = {}
        self._visit_count = 0
        self._search_dir = 0
        self._search_run = 0
        self._recent: deque = deque(maxlen=max(1, self.cfg.search_memory))
        self._fixation: dict | None = None         # moves mode: where the search is looking now
        self._glance_back: Units | None = None
        self._search_last: dict | None = None      # the fixation the running look-around ends on
        self._search_from_last = False             # ... and the next clip starts ON it (after a take-over move)
        self._replace_at = math.inf                # when the running clip (or take-over move) is replaced
        self._clip_done_for = -math.inf            # issue time of the lock clip last credited as reached

        self._ok_cache: dict[tuple, str] = {}
        self._frame_cache: dict[tuple, bool] = {}
        self._blocked = ""
        self._aim_err: float | None = None
        self._aim_point: np.ndarray | None = None
        self._trace: dict = {}
        self.stats = {k: 0 for k in (
            "commands", "move", "clip", "search_clip", "search_move", "acquire", "lock", "lost", "reacquire",
            "search_again", "peek", "keep_alive", "alive", "turn", "preempt", "hold", "failures",
            "expected_cancels", "succeeded", "steps_blocked_budget", "steps_blocked_backoff", "pose_refused",
            "path_refused", "step_refused", "unreachable", "tracks_created", "urgent", "failed_but_reached")}

    # ------------------------------------------------------------------ public read-outs
    @property
    def state(self) -> str:
        return self._state

    @property
    def locked_id(self) -> str | None:
        """The tracker's own id of the track it is on (ACQUIRE, LOCK, LOST), never a ground-truth person id."""
        if self._state in (ACQUIRE, LOCK, LOST) and self._locked is not None:
            return self._locked.id
        return None

    @property
    def trace(self) -> dict:
        """What happened at the last update, for the report."""
        return dict(self._trace)

    def window_used(self, t: float) -> int:
        """Commands this tracker sent in the 60 s before t (the SDK's sliding window)."""
        self._prune_issued(t)
        return len(self._issued)

    def reset(self, t: float) -> None:
        """Leave HOLD (an operator decision) and start searching again."""
        self._failures = 0
        self._backoff_until = -math.inf
        self._release(t)

    # ------------------------------------------------------------------ the step
    def update(self, t: float, detections: list[Detection], measured_units: Units, motion_busy: bool,
               last_outcome: str | None = None, outcome_details: dict | None = None) -> list[Command]:
        """outcome_details: what the SDK's result says with last_outcome; for a failed settle its
        position_errors and position_tolerances (see reached_factor)."""
        t = float(t)
        measured = {j: float(measured_units[j]) for j in JOINTS}
        self._blocked, self._aim_err, self._aim_point, self._cmd_point = "", None, None, None
        self._cmd_kind = "turn"
        self._remember_pose(t, measured)
        if last_outcome == "failed" and self._reached(outcome_details):
            self.stats["failed_but_reached"] += 1
            last_outcome = "succeeded"                  # the arm is where gravity lets it be: see reached_deg
        self._note_outcome(t, last_outcome, measured)
        self._ingest(detections or [])
        self._prune_tracks(t)
        busy = self._busy(t, bool(motion_busy))

        if self._state == HOLD and self.cfg.hold_release_s is not None \
                and t - self._hold_since >= self.cfg.hold_release_s:
            self.reset(t)
        command = None
        if self._state == SEARCH:
            command = self._search_step(t, measured, busy)
        elif self._state in (ACQUIRE, LOCK):
            command = self._follow_step(t, measured, busy)
        elif self._state == LOST:
            command = self._lost_step(t, measured, busy)

        if command is not None:
            self._issue(t, command, bool(motion_busy))
        self._write_trace(t, busy, command)
        return [command] if command is not None else []

    def posted(self, t_call: float, t_posted: float) -> None:
        """The adapter reports that the action POST of the command issued at t_call went out at t_posted (a
        clip.play follows its upload): the SDK counts it in its rate window from then."""
        for k in range(len(self._issued) - 1, -1, -1):
            if abs(self._issued[k] - t_call) < 1e-9:
                self._issued[k] = max(float(t_posted), t_call)
                break

    # ------------------------------------------------------------------ outcomes, budget, busy
    def _reached(self, details: dict | None) -> bool:
        """A failed settle whose every joint stopped within reached_factor x its tolerance."""
        try:
            errors = {j: float(v) for j, v in details["position_errors"].items()}
            tolerances = {j: float(v) for j, v in details["position_tolerances"].items()}
            return bool(errors) and all(abs(e) <= self.cfg.reached_factor * tolerances[j] for j, e in errors.items())
        except (KeyError, TypeError, ValueError, AttributeError):
            return False

    def _note_outcome(self, t: float, outcome: str | None, measured: Units) -> None:
        if outcome is None:
            return
        outcome = str(outcome)
        self._pending = max(0, self._pending - 1)
        if outcome == "succeeded":
            self._failures = 0
            self.stats["succeeded"] += 1
            if self._pending == 0 and self._last_issue_state in (ACQUIRE, LOCK):
                self._acquire_successes += 1
            if self._pending == 0 and self._last_point is not None and self.cfg.sag_compensation:
                self._learn_bias(measured)
        elif outcome == "canceled" and self._expected_cancels > 0:
            self._expected_cancels -= 1                 # we pre-empted our own action: not a failure
            self.stats["expected_cancels"] += 1
        else:
            # failed settle, a cancel we did not ask for (the runtime took over), a refusal: back off, and
            # after max_failures in a row stop (fork sdk-spec safe-motion.md section 9)
            self._failures += 1
            self.stats["failures"] += 1
            rate = "429" in outcome or "rate" in outcome.lower()
            self._backoff_until = t + (self.cfg.rate_limited_backoff_s if rate else self.cfg.backoff_s)
            if self._failures >= self.cfg.max_failures and self._state != HOLD:
                self._state, self._hold_since = HOLD, t
                self.stats["hold"] += 1
        if self._pending == 0:
            self._expected_cancels = 0

    def _busy(self, t: float, motion_busy: bool) -> bool:
        if self._pending and not motion_busy and t - self._last_issue_t > self.cfg.busy_grace_s:
            # nothing is running and no outcome came back: the run loop did not report one; do not wait forever
            self._pending, self._expected_cancels = 0, 0
        return motion_busy or self._pending > 0

    def _prune_issued(self, t: float) -> None:
        # The SDK keeps a stamp exactly 60 s old in its window (vendor source sdk_gateway/policy.py:172-179;
        # twin/motion.py RateWindow keeps s >= t - 60). Forget ours only once strictly older, with a little
        # slack for float time, or a POST on the boundary is let through here and refused 429 by the SDK
        # (found by twin/run.py: a 90 s run at 30/min, tracking 26 + light 4 on one session).
        while self._issued and t - self._issued[0] > 60.0 + 1e-6:
            self._issued.popleft()

    def _gate(self, t: float) -> str:
        """Why no command may go out now, or ''."""
        if self._state == HOLD:
            return "hold"
        if t < self._backoff_until:
            self.stats["steps_blocked_backoff"] += 1
            return "backoff"
        if t - self._last_issue_t < self.cfg.min_command_gap_s:
            return "gap"
        if t < self._next_try_t:
            return "retry_wait"
        self._prune_issued(t)
        if len(self._issued) >= self.cfg.max_commands_per_min:
            self.stats["steps_blocked_budget"] += 1
            return "budget"
        return ""

    def _learn_bias(self, measured: Units) -> None:
        """After a successful aiming move: the angle between where look_at was asked to face and where the
        camera settled is the load error (plus the IK's own residual). Low-passed, bounded."""
        h = self.kin.head(measured)
        want_az, want_el = _az_el(self._last_point - np.asarray(h.position, float))
        got_az, got_el = _az_el(np.asarray(h.forward, float))
        gap = np.array([(want_az - got_az + 180.0) % 360.0 - 180.0, want_el - got_el])
        if np.all(np.abs(gap) < 3 * self.cfg.max_bias_deg):     # a wild reading is not a load error
            g = self.cfg.sag_gain
            self._bias = np.clip((1 - g) * self._bias + g * gap, -self.cfg.max_bias_deg, self.cfg.max_bias_deg)

    def _biased(self, point: np.ndarray, origin: np.ndarray) -> np.ndarray:
        """The point to give look_at so that, after the learnt load error, the camera faces `point`."""
        if not self.cfg.sag_compensation or not np.any(self._bias):
            return np.asarray(point, float)
        v = np.asarray(point, float) - origin
        az, el = _az_el(v)
        return origin + _direction(az + self._bias[0], el + self._bias[1]) * float(np.linalg.norm(v))

    def _issue(self, t: float, command: Command, motion_busy: bool) -> None:
        # Anything still running will be pre-empted by this command and end "canceled": that is ours.
        self._expected_cancels += max(self._pending, 1 if motion_busy else 0)
        self._pending += 1
        self._last_issue_t = t
        self._last_issue_state = self._state
        self._last_point = self._cmd_point            # None for search: nothing to learn from those
        self._running = self._cmd_kind
        self._issued.append(t)
        self.stats["commands"] += 1
        self.stats[command.kind] += 1

    # ------------------------------------------------------------------ seeing: detections -> 3D tracks
    def target_point(self, detection: Detection, units: Units) -> np.ndarray:
        """A detection -> the head's centre in the base frame, seen from the camera at `units`.

        Picture position -> a ray through the pinhole; distance from the face's apparent width
        (face_width_m), clamped to min_range_m..max_range_m; a ray going down stops at min_target_z_m above
        the table, because no head is below the table."""
        h = self.kin.head(units)
        ray = (np.asarray(h.forward, float) + np.asarray(h.right, float) * ((detection.x - 0.5) / self._fx)
               + np.asarray(h.down, float) * ((detection.y - 0.5) / self._fy))
        depth = self.cfg.face_width_m * self._fx / max(float(detection.size), 1e-4)   # along the optical axis
        length = float(np.linalg.norm(ray))
        rng = float(np.clip(depth * length, self.cfg.min_range_m, self.cfg.max_range_m))
        position = np.asarray(h.position, float)
        point = position + ray * (rng / length)
        z_min = self.cfg.min_target_z_m
        if point[2] < z_min:
            if ray[2] < -1e-9 and position[2] > z_min:
                point = position + ray * ((z_min - position[2]) / ray[2])
            point[2] = max(point[2], z_min)
        return point

    def _remember_pose(self, t: float, measured: Units) -> None:
        self._poses.append((t, np.array([measured[j] for j in JOINTS])))
        while len(self._poses) > 2 and t - self._poses[0][0] > self.cfg.pose_history_s:
            self._poses.popleft()

    def _pose_at(self, t: float) -> Units:
        """The measured pose at a past time (linear between samples): where the camera was at capture."""
        hist = self._poses
        if t <= hist[0][0]:
            vec = hist[0][1]
        elif t >= hist[-1][0]:
            vec = hist[-1][1]
        else:
            k = len(hist) - 1
            while hist[k - 1][0] > t:
                k -= 1
            (t0, a), (t1, b) = hist[k - 1], hist[k]
            vec = a + (b - a) * ((t - t0) / max(t1 - t0, 1e-9))
        return {j: float(v) for j, v in zip(JOINTS, vec, strict=True)}

    def _predict(self, tr: _Track, t: float) -> np.ndarray:
        """Constant-velocity prediction, with a still person kept still and the extrapolation clamped."""
        dt = min(max(t - tr.t, 0.0), self.cfg.max_predict_s)
        vel = tr.vel if float(np.linalg.norm(tr.vel)) >= self.cfg.still_speed_m_s else np.zeros(3)
        step = vel * dt
        size = float(np.linalg.norm(step))
        if size > self.cfg.predict_max_m:
            step *= self.cfg.predict_max_m / size
        return tr.pos + step

    def _ingest(self, detections: list[Detection]) -> None:
        # The time of a frame is its stamp as the client sees it (t_stamp), less what the client believes the
        # stamp lags the exposure by. t_capture, the true exposure time, is ground truth: never used here.
        frames: dict[float, list[Detection]] = {}
        for det in detections:
            if float(det.confidence) >= self.cfg.min_confidence:
                stamp = det.t_capture if det.t_stamp is None else det.t_stamp
                frames.setdefault(round(float(stamp) - self.cfg.stamp_lag_s, 6), []).append(det)
        for tc in sorted(frames):
            units = self._pose_at(tc)
            self._associate(tc, [(det, self.target_point(det, units)) for det in frames[tc]])

    def _associate(self, tc: float, points: list[tuple[Detection, np.ndarray]]) -> None:
        """Nearest pairs first over every (track, detection) pair inside the track's gate, each track and each
        detection used once. So a second person's detection stays with the second person's own track even
        when it falls inside the locked track's widening gate. Identity is never used: two people are two
        tracks because they are in two places."""
        pairs = []
        for k, tr in enumerate(self._tracks):
            pred = self._predict(tr, tc)
            gate = min(self.cfg.gate_m + self.cfg.gate_speed_m_s * max(0.0, tc - tr.t), self.cfg.gate_max_m)
            for i, (_, point) in enumerate(points):
                d = float(np.linalg.norm(point - pred))
                if d <= gate:
                    pairs.append((d, k, i))
        used_tracks, free = set(), set(range(len(points)))
        for _, k, i in sorted(pairs):
            if k in used_tracks or i not in free:
                continue
            self._update_track(self._tracks[k], tc, *points[i])
            used_tracks.add(k)
            free.discard(i)
        for i in sorted(free):
            if len(self._tracks) >= self.cfg.max_tracks:
                break
            det, point = points[i]
            self._track_count += 1
            tr = _Track(f"track-{self._track_count}", point.copy(), np.zeros(3), tc, scoring_person_id=det.person_id)
            tr.hits.append(tc)
            self._tracks.append(tr)
            self.stats["tracks_created"] += 1

    def _update_track(self, tr: _Track, tc: float, det: Detection, z: np.ndarray) -> None:
        dt = tc - tr.t
        if dt > self.cfg.reset_gap_s:                   # a long gap: start again from this sighting
            tr.pos, tr.vel = z.copy(), np.zeros(3)
        elif dt <= 1e-6:                                # same frame or an older one: position only
            tr.pos = tr.pos + self.cfg.alpha * (z - tr.pos)
        else:
            pred = tr.pos + tr.vel * dt
            r = z - pred
            tr.pos = pred + self.cfg.alpha * r
            tr.vel = tr.vel + (self.cfg.beta / dt) * r
            speed = float(np.linalg.norm(tr.vel))
            if speed > self.cfg.max_speed_m_s:
                tr.vel *= self.cfg.max_speed_m_s / speed
        tr.t = max(tr.t, tc)
        if not tr.hits or tr.hits[-1] != tc:
            tr.hits.append(tc)
        tr.n += 1
        tr.scoring_person_id = det.person_id

    def _confirmed(self, tr: _Track) -> bool:
        """Seen in at least N of the last M frames (a time window of M frames at camera_fps)."""
        window = self.cfg.debounce_m / self.cfg.camera_fps
        return sum(1 for h in tr.hits if h > tr.t - window + _EPS) >= self.cfg.debounce_n

    def _prune_tracks(self, t: float) -> None:
        keep = []
        for tr in self._tracks:
            if tr is self._locked or t - tr.t <= self.cfg.track_drop_s:
                keep.append(tr)
        self._tracks = keep

    def _best_confirmed(self, t: float) -> _Track | None:
        """A confirmed face in view; the nearest one if there are several (likely the listener)."""
        ready = [tr for tr in self._tracks if t - tr.t <= self.cfg.fresh_s and self._confirmed(tr)]
        return min(ready, key=lambda tr: float(np.linalg.norm(tr.pos))) if ready else None

    # ------------------------------------------------------------------ poses the tracker may command
    def _key(self, units: Units) -> tuple:
        if len(self._ok_cache) + len(self._frame_cache) > 50000:   # measured poses never repeat: keep it bounded
            self._ok_cache.clear()
            self._frame_cache.clear()
        return tuple(round(float(units[j]), 3) for j in JOINTS)

    def _look(self, point: np.ndarray, seed: Units | None) -> Units | None:
        try:
            pose, _ = self.kin.look_at(np.asarray(point, float), seed=seed, **self.cfg.look_at_kw)
        except Exception:                                # an IK failure is a refusal, never a crash
            return None
        try:
            pose = {j: round(float(pose[j]), 2) for j in JOINTS}   # all five joints, absolute
        except (KeyError, TypeError, ValueError):
            return None
        return pose if all(math.isfinite(v) for v in pose.values()) else None

    def _frame_ok(self, units: Units) -> bool:
        """Limits and kin.check: the rule for every commanded pose and every clip frame."""
        key = self._key(units)
        if key not in self._frame_cache:
            ok = all(self.kin.limits[j][0] <= units[j] <= self.kin.limits[j][1] for j in JOINTS)
            self._frame_cache[key] = bool(ok and self.kin.check(units).ok)
        return self._frame_cache[key]

    def _pose_ok(self, units: Units) -> str:
        """'' when the tracker may aim with this pose, else why not: limits, kin.check, and facing the room."""
        key = self._key(units)
        if key not in self._ok_cache:
            why = ""
            if not self._frame_ok(units):
                why = "check"
            else:
                az, el = _az_el(np.asarray(self.kin.head(units).forward, float))
                if el < self.cfg.min_facing_elev_deg:
                    why = "faces the table"
                elif abs(az) > self.cfg.max_facing_az_deg:
                    why = "faces away from the room"
            self._ok_cache[key] = why
        return self._ok_cache[key]

    def _clearance(self, units: Units) -> float:
        c = self.kin.check(units)
        return min(c.min_table_m, c.min_base_m, c.min_self_m)

    def _path_ok(self, start: Units, end: Units) -> bool:
        """The vendor plans a straight joint-space line from the measured pose and checks only head against
        base, so the client checks the rest along it. From a pose that already fails (say the vendor idle
        crouched the arm), the way out may only get better until it is clear, then must stay clear."""
        a = np.array([start[j] for j in JOINTS])
        b = np.array([end[j] for j in JOINTS])
        escaping = not self._frame_ok(start)
        worst = self._clearance(start) if escaping else 0.0
        n = max(1, self.cfg.path_samples)
        for k in range(1, n + 1):
            p = {j: float(v) for j, v in zip(JOINTS, a + (b - a) * (k / n), strict=True)}
            if self._frame_ok(p):
                escaping = False
                continue
            if not escaping:
                return False
            c = self._clearance(p)
            if c < worst - 1e-4:
                return False
            worst = c
        return True

    def _aim_error(self, units: Units, point: np.ndarray) -> float:
        h = self.kin.head(units)
        return _angle_deg(np.asarray(h.forward, float), np.asarray(point, float) - np.asarray(h.position, float))

    def _clamp_point(self, p: np.ndarray) -> np.ndarray:
        """Keep a target in front of the lamp and above the table."""
        p = np.array(p, dtype=float)
        p[2] = max(p[2], self.cfg.min_target_z_m)
        az = math.atan2(p[0], p[1])
        lim = math.radians(self.cfg.max_target_az_deg)
        if abs(az) > lim:
            r = math.hypot(p[0], p[1])
            az = math.copysign(lim, az)
            p[0], p[1] = r * math.sin(az), r * math.cos(az)
        return p

    def _pose_toward(self, t: float, point: np.ndarray, ref: Units, measured: Units) -> Units | None:
        """A pose from kin.look_at facing `point`, or part of the way there when the turn is large.

        The turn is limited in angle (max_step_deg) from where `ref` faces, and in joint units
        (max_step_units) from `ref`; ref is the pose last asked for, or the measured one before any. The
        joint limit is skipped when ref itself does not face the room (for example the vendor idle crouched
        the arm): getting back up is one move. Refused poses are counted and not retried for retry_s."""
        point = self._clamp_point(point)
        h = self.kin.head(ref)
        origin, forward = np.asarray(h.position, float), _unit(np.asarray(h.forward, float))
        ref_faces_room = self._pose_ok(ref) == ""
        if not ref_faces_room:                            # start the turn from level, not from the table
            forward = _unit(np.array([forward[0], forward[1], 0.0])) if math.hypot(forward[0], forward[1]) > 1e-6 \
                else np.array([0.0, 1.0, 0.0])
        to = point - origin
        dist = max(float(np.linalg.norm(to)), 1e-3)
        want = to / dist
        angle = _angle_deg(forward, want)
        first = 1.0 if angle <= self.cfg.max_step_deg else self.cfg.max_step_deg / angle
        # Seed the IK with the pose last asked for, never with a reading: a seed taken from the sagged
        # measured pose would ratchet the loaded joints down (motion spec 11).
        seed = self._sent_base if self._sent_base is not None else dict(self.kin.neutral)
        reason = "unreachable"
        for frac in (first, first * 0.66, first * 0.4, first * 0.2):
            aim = origin + _slerp(forward, want, frac) * dist
            pose = self._look(aim, seed)
            if pose is None:
                continue
            if self._pose_ok(pose) or self._aim_error(pose, aim) > self.cfg.max_aim_error_deg:
                reason = "pose_refused" if self._pose_ok(pose) else "unreachable"
                continue
            if ref_faces_room and max(abs(pose[j] - ref[j]) for j in JOINTS) > self.cfg.max_step_units + 1e-6:
                reason = "step_refused"
                continue
            if not self._path_ok(measured, pose):
                reason = "path_refused"
                continue
            self._toward = self._cmd_point = aim
            return pose
        self.stats[reason] += 1
        self._blocked = reason
        self._next_try_t = t + self.cfg.retry_s
        return None

    def _move(self, pose: Units, why: str) -> Command:
        return Command(kind="move", target={j: float(pose[j]) for j in JOINTS}, why=why)

    def _horizon(self) -> float:
        """How far ahead to aim: until the commanded motion will have arrived."""
        if self.strategy == "preempt":
            return self.cfg.preempt_interval_s + self.cfg.sdk_preroll_s
        if self.strategy == "clip":
            return self.cfg.clip_entry_s + self.cfg.clip_start_s
        return self.cfg.settle_predict_s

    # ------------------------------------------------------------------ states
    def _set_state(self, state: str, t: float) -> None:
        if state != self._state:
            self._state = state
            if state == LOCK:
                self.stats["lock"] += 1
                lo, hi = self.cfg.alive_every_s
                self._next_alive_t = t + float(self._rng.uniform(lo, hi))

    def _acquire(self, t: float, tr: _Track, busy: bool) -> None:
        self._locked = tr
        self._set_state(ACQUIRE, t)
        self.stats["acquire"] += 1
        self._sent_base, self._sent_point, self._alive_out = None, None, False
        self._acquire_successes = 0
        self._must_preempt = busy
        self._next_try_t = -math.inf
        self._fixation, self._glance_back = None, None
        self._replace_at = math.inf

    def _release(self, t: float) -> None:
        """Back to SEARCH: forget the lock and the pose that went with it."""
        if self._locked is not None and self._locked in self._tracks:
            self._tracks.remove(self._locked)
        self._locked = None
        self._state = SEARCH
        self._sent_base, self._sent_point, self._alive_out = None, None, False
        self._must_preempt = False
        self._fixation, self._glance_back = None, None
        self._search_dir, self._search_run = 0, 0
        self._search_last, self._search_from_last, self._replace_at = None, False, math.inf

    def _enter_lost(self, t: float) -> None:
        self._set_state(LOST, t)
        self.stats["lost"] += 1
        self._lost_since = t
        self._lost_seen_t = self._locked.t if self._locked is not None else t
        self._peeked = False
        self._alive_out = False

    # ---------------------------------------------------------------- SEARCH
    def _search_step(self, t: float, measured: Units, busy: bool) -> Command | None:
        found = self._best_confirmed(t)
        if found is not None:
            self._acquire(t, found, busy)
            return self._follow_step(t, measured, busy)
        # a hold makes no progress, so a new search may replace it; so may the next look-around, just before
        # the running one (or the take-over move) ends (pipeline_margin_s)
        replace = self._running in ("search", "search_move") and t >= self._replace_at
        if busy and self._running != "hold" and not replace:
            return None
        self._cmd_kind = "search"
        self._blocked = self._gate(t)
        if self._blocked:
            return None
        if self.cfg.search_via == "clip":
            ours = busy and self._running in ("search", "search_move")
            if not ours and self.cfg.search_take_over:
                pose, why = self._search_move(measured)
                if pose is not None:
                    self._search_last, self._search_from_last = self._fixation, True
                    self._cmd_kind = "search_move"
                    self._replace_at = t + self.cfg.min_command_gap_s     # the look-around follows at once
                    self.stats["search_move"] += 1
                    return self._move(pose, "search: take the arm from the vendor idle, " + why.split(": ", 1)[-1])
            origin = self._search_last if ours else None
            frames = self._search_clip(measured, origin=origin, first=origin if self._search_from_last else None)
            if frames is None:
                self._blocked = "no_search_path"
                self._next_try_t = t + self.cfg.retry_s
                return None
            self._search_from_last = False
            self._replace_at = t + self._clip_replace_after(len(frames))
            self.stats["search_clip"] += 1
            return Command(kind="clip", frames=frames,
                           why=f"search: look around, {len(frames) / self.cfg.clip_fps:.0f} s clip")
        pose, why = self._search_move(measured)
        if pose is None:
            self._blocked = "no_search_path"
            self._next_try_t = t + self.cfg.retry_s
            return None
        self.stats["search_move"] += 1
        return self._move(pose, why)

    def _anchors_list(self) -> list[dict]:
        """Places a head could be: an arc of bearings at seated and standing height, each checked once."""
        if self._anchors is None:
            self._anchors = []
            for row, (height, dist) in enumerate(self.cfg.search_rows):
                for az in self.cfg.search_azimuths_deg:
                    a = math.radians(az)
                    point = np.array([dist * math.sin(a), dist * math.cos(a), height])
                    pose = self._look(point, dict(self.kin.neutral))
                    if pose is None or self._pose_ok(pose) or self._aim_error(pose, point) > self.cfg.max_aim_error_deg:
                        continue
                    gaze_az, gaze_el = _az_el(np.asarray(self.kin.head(pose).forward, float))
                    self._anchors.append({"i": len(self._anchors), "row": row, "az": gaze_az, "el": gaze_el,
                                          "point": point, "pose": pose})
        return self._anchors

    def _pick_fixation(self, cur_az: float, cur_el: float) -> dict | None:
        """The next place to look: not one just visited, a moderate jump, and never a third jump the same way.
        Weighted at random (seeded), so the order varies from clip to clip."""
        anchors = self._anchors_list()
        if not anchors:
            return None
        lo, pref, hi = self.cfg.search_step_deg
        options: list[tuple[dict, float, int]] = []
        for strict in (2, 1, 0):          # relax the rules if nothing fits: no direction rule, then anything
            for k in self._rng.permutation(len(anchors)):    # shuffled, so ties in novelty break at random
                a = anchors[int(k)]
                if strict and a["i"] in self._recent:
                    continue
                d_az, d_el = a["az"] - cur_az, a["el"] - cur_el
                step = math.hypot(d_az, d_el)
                direction = 0 if abs(d_az) < 2.0 else (1 if d_az > 0 else -1)
                if strict and not lo <= step <= hi:
                    continue
                if strict == 2 and direction and direction == self._search_dir \
                        and self._search_run >= self.cfg.search_max_same_direction:
                    continue
                # a moderate jump is likeliest, a long one still possible ("what's over there?")
                weight = max(math.exp(-((step - pref) / self.cfg.search_step_spread_deg) ** 2), 0.2)
                if strict and abs(d_el) > 5.0:
                    weight *= self.cfg.search_row_switch_weight
                options.append((a, weight, direction))
            if options:
                break
        # Curiosity: only the places least recently looked at are candidates, so the whole room is covered
        # (otherwise a run of short jumps can keep the lamp on one side for many seconds).
        options.sort(key=lambda o: self._visits.get(o[0]["i"], -1))
        options = options[:max(1, self.cfg.search_novel_k)]
        weights = np.array([w for _, w, _ in options])
        a, _, direction = options[int(self._rng.choice(len(options), p=weights / weights.sum()))]
        if direction and direction == self._search_dir:
            self._search_run += 1
        elif direction:
            self._search_dir, self._search_run = direction, 1
        self._visit_count += 1
        self._visits[a["i"]] = self._visit_count
        self._recent.append(a["i"])
        return self._jittered(a)

    def _jittered(self, anchor: dict) -> dict:
        """Each visit lands a little differently; falls back to the anchor if the variant is not allowed."""
        x, y, z = anchor["point"]
        a = math.atan2(x, y) + math.radians(self._rng.uniform(-1, 1) * self.cfg.search_jitter_deg)
        r = math.hypot(x, y)
        point = np.array([r * math.sin(a), r * math.cos(a), z + self._rng.uniform(-1, 1) * self.cfg.search_jitter_m])
        pose = self._look(point, anchor["pose"])
        if pose is None or self._pose_ok(pose) or self._aim_error(pose, point) > self.cfg.max_aim_error_deg:
            point, pose = anchor["point"], anchor["pose"]
        az, el = _az_el(np.asarray(self.kin.head(pose).forward, float))
        return {"pose": pose, "point": point, "az": az, "el": el, "row": anchor["row"]}

    def _glance(self, fix: dict) -> Units | None:
        """A small curious look beside or above/below the fixation point."""
        lo, hi = self.cfg.search_glance_m
        view = _unit(np.asarray(fix["point"], float))
        side = _unit(np.cross(view, _Z))
        offset = side * self._rng.choice([-1.0, 1.0]) * self._rng.uniform(lo, hi) \
            + _Z * self._rng.uniform(-0.5, 0.5) * lo
        point = np.asarray(fix["point"], float) + offset
        pose = self._look(point, fix["pose"])
        if pose is None or self._pose_ok(pose) or self._aim_error(pose, point) > self.cfg.max_aim_error_deg:
            return None
        return pose

    def _segment(self, a: Units, b: Units, seconds: float) -> list[Units] | None:
        """Frames (after a) easing from a to b, at clip_fps; None if any frame fails the checks. Stretched when
        needed so the quintic's peak speed (1.875 x the mean) stays below clip_max_speed_u_s."""
        delta = max(abs(a[j] - b[j]) for j in JOINTS)
        seconds = max(seconds, 1.875 * delta / self.cfg.clip_max_speed_u_s)
        n = max(1, int(math.ceil(seconds * self.cfg.clip_fps - 1e-9)))
        va = np.array([a[j] for j in JOINTS])
        vb = np.array([b[j] for j in JOINTS])
        out = []
        for k in range(1, n + 1):
            pose = {j: round(float(v), 3) for j, v in zip(JOINTS, va + (vb - va) * _ease(k / n), strict=True)}
            if not self._frame_ok(pose):
                return None
            out.append(pose)
        return out

    def _saccade_s(self, a: Units, b: Units) -> float:
        delta = max(abs(a[j] - b[j]) for j in JOINTS)
        peak = float(self._rng.uniform(*self.cfg.search_peak_speed_u_s))
        # a quintic's peak speed is 1.875 x its mean speed (vendor source safe_motion.py:114)
        return max(self.cfg.search_min_saccade_s, 1.875 * delta / peak)

    def _hold(self, pose: Units, seconds: float) -> list[Units]:
        return [dict(pose) for _ in range(max(1, int(round(seconds * self.cfg.clip_fps))))]

    def _clip_replace_after(self, rows: int) -> float:
        """From a clip's upload to the moment the next clip should be uploaded so that it is admitted
        pipeline_margin_s before this one ends (client-side replica of the SDK's costs, twin/motion.py)."""
        entry = M.plan_seconds(0.0)                     # at least 2.0 s; longer only for turns over 144 units
        end = M.clip_start_delay_s(rows) + entry + (rows - 1) / self.cfg.clip_fps
        return max(self.cfg.min_command_gap_s, end - self.cfg.pipeline_margin_s - M.clip_arbitration_delay_s(rows))

    def _search_clip(self, measured: Units, origin: dict | None = None,
                     first: dict | None = None) -> list[tuple[float, Units]] | None:
        """One look-around clip: fixations with pauses and glances, in a varied order.

        origin: the fixation the arm will be on when this clip is admitted (the end of the running
        look-around, or the take-over move's target); None = the measured pose. first: start ON that
        fixation instead of choosing a new one. The vendor adds its own >= 2 s entry move from wherever
        the arm is to the first frame; that path is checked here too, from origin. Every frame is checked,
        and speeds stay below clip_max_speed_u_s."""
        start = origin["pose"] if origin is not None else measured
        if origin is not None:
            cur_az, cur_el = origin["az"], origin["el"]
        else:
            cur_az, cur_el = _az_el(np.asarray(self.kin.head(measured).forward, float))
        if first is None:
            for _ in range(6):
                cand = self._pick_fixation(cur_az, cur_el)
                if cand is None:
                    return None
                if self._path_ok(start, cand["pose"]):
                    first = cand
                    break
        if first is None:
            return None
        poses = [first["pose"]] + self._hold(first["pose"], self._rng.uniform(*self.cfg.search_dwell_s))
        cur = first
        while len(poses) / self.cfg.clip_fps < self.cfg.search_clip_s:
            if self._rng.random() < self.cfg.search_glance_p:
                glance = self._glance(cur)
                if glance is not None:
                    there = self._segment(cur["pose"], glance, self._rng.uniform(*self.cfg.search_glance_move_s))
                    back = self._segment(glance, cur["pose"], self._rng.uniform(*self.cfg.search_glance_move_s))
                    if there and back:
                        poses += there + self._hold(glance, self._rng.uniform(*self.cfg.search_glance_hold_s))
                        poses += back + self._hold(cur["pose"], 0.2)
            nxt, seg = None, None
            for _ in range(4):
                cand = self._pick_fixation(cur["az"], cur["el"])
                if cand is None:
                    break
                seg = self._segment(cur["pose"], cand["pose"], self._saccade_s(cur["pose"], cand["pose"]))
                if seg:
                    nxt = cand
                    break
            if nxt is None:
                break
            poses += seg + self._hold(nxt["pose"], self._rng.uniform(*self.cfg.search_dwell_s))
            cur = nxt
        if len(poses) < 2:
            return None
        frames = [(k / self.cfg.clip_fps, pose) for k, pose in enumerate(poses)]
        if not self._clip_speed_ok(frames):
            return None
        self._search_last = cur                          # where this look-around ends: the next one starts there
        return frames

    def _clip_speed_ok(self, frames: list[tuple[float, Units]]) -> bool:
        for (ta, a), (tb, b) in zip(frames, frames[1:], strict=False):
            if max(abs(a[j] - b[j]) for j in JOINTS) / (tb - ta) > self.cfg.clip_max_speed_u_s * 1.001:
                return False
        return True

    def _search_move(self, measured: Units) -> tuple[Units | None, str]:
        """Moves mode: one motion.move per fixation or glance, sent as soon as the last one succeeded (the
        plan's own slow start and end are the pauses; waiting would let the vendor idle pull the head down)."""
        if self._glance_back is not None:
            pose, self._glance_back = self._glance_back, None
            if self._path_ok(measured, pose):
                return pose, "search: back from a glance"
        if self._fixation is not None and self._rng.random() < self.cfg.search_glance_p:
            glance = self._glance(self._fixation)
            if glance is not None and self._path_ok(measured, glance):
                self._glance_back = self._fixation["pose"] if self._rng.random() < 0.5 else None
                return glance, "search: a small glance"
        if self._fixation is not None:
            cur_az, cur_el = self._fixation["az"], self._fixation["el"]
        else:
            cur_az, cur_el = _az_el(np.asarray(self.kin.head(measured).forward, float))
        for _ in range(6):
            cand = self._pick_fixation(cur_az, cur_el)
            if cand is None:
                return None, ""
            if self._path_ok(measured, cand["pose"]):
                self._fixation = cand
                row = "seated" if cand["row"] == 0 else "standing"
                return cand["pose"], f"search: look {cand['az']:+.0f} deg at {row} height"
        return None, ""

    # ---------------------------------------------------------------- ACQUIRE and LOCK
    def _reference(self, measured: Units) -> Units:
        return self._sent_base if self._sent_base is not None else measured

    def _drifted(self, measured: Units) -> bool:
        return self._sent_base is not None and \
            max(abs(measured[j] - self._sent_base[j]) for j in JOINTS) > self.cfg.drift_units

    def _follow_step(self, t: float, measured: Units, busy: bool) -> Command | None:
        tr = self._locked
        if tr is None or t - tr.t > self.cfg.lost_after_s:
            self._enter_lost(t)
            return self._lost_step(t, measured, busy)
        ref = self._reference(measured)
        origin = np.asarray(self.kin.head(ref).position, float)
        now = self._clamp_point(self._predict(tr, t))
        aim = self._clamp_point(self._predict(tr, t + self._horizon()))
        self._aim_point = aim
        # A pose we asked for faces the biased point it was built for; a measured pose faces what it faces.
        if self._sent_base is not None:
            err = self._aim_error(ref, self._biased(now, origin))
            err_aim = self._aim_error(ref, self._biased(aim, origin))
        else:
            err, err_aim = self._aim_error(ref, now), self._aim_error(ref, aim)
        self._aim_err = err
        # clip: the next clip replaces the running one just before it ends (pipeline_margin_s). By then the
        # running clip is on its last knot, which is its success (its own outcome will be "canceled").
        replace = busy and self.strategy == "clip" and self._running == "turn" and t >= self._replace_at
        if replace and self._clip_done_for != self._last_issue_t:
            self._clip_done_for = self._last_issue_t
            if self._last_issue_state in (ACQUIRE, LOCK):
                self._acquire_successes += 1
            if self._last_point is not None and self.cfg.sag_compensation:
                self._learn_bias(measured)
        if self._state == ACQUIRE and (not busy or replace) and self._acquire_successes > 0 \
                and err <= self.cfg.lock_enter_deg:
            self._set_state(LOCK, t)
        elif self._state == LOCK and err > self.cfg.lock_exit_deg:
            self._set_state(ACQUIRE, t)
            self._acquire_successes = 0
        # the load bias moved since the held pose was solved: re-solve it (replaces a keep-alive, same cost)
        reaim = not busy and self._sent_base is not None and self.cfg.sag_compensation \
            and float(np.max(np.abs(self._bias - self._base_bias))) > self.cfg.bias_refresh_deg
        need = self._sent_base is None or err_aim > self.cfg.deadband_deg or reaim or \
            (not busy and self._drifted(measured))

        urgent = busy and self.strategy == "settle" and self._running == "hold" \
            and self.cfg.urgent_deg is not None and err_aim > self.cfg.urgent_deg
        if busy and not self._must_preempt and not urgent and not replace:
            if self.strategy == "preempt":
                return self._preempt(t, measured, ref, origin, aim)
            return None
        self._blocked = self._gate(t)
        if self._blocked:
            return None
        if self.strategy == "clip" and not self._must_preempt:
            return self._lock_clip(t, measured, ref, origin, tr, need)
        if need:
            pose = self._pose_toward(t, self._biased(aim, origin), ref, measured)
            if pose is None:
                return None
            why = "acquire: turn to the face" if self._must_preempt else "track: turn to the face"
            if reaim and err_aim <= self.cfg.deadband_deg:
                why = "track: re-aim for the learnt load error"
            if urgent and not self._must_preempt:
                why += " (pre-empts a hold: the target moved)"
                self.stats["urgent"] += 1
            self._must_preempt = False
            self._set_base(pose, aim)
            self.stats["turn"] += 1
            return self._move(pose, why)
        if self._state == LOCK and self.cfg.alive and t >= self._next_alive_t and not self._alive_out:
            command = self._alive(t, measured, ref, origin)
            if command is not None:
                return command
        if self._alive_out or self.cfg.keep_alive:
            why = "alive: back to centre" if self._alive_out else "keep-alive: hold the gaze"
            return self._resend_base(t, measured, why)
        return None

    def _set_base(self, pose: Units, point: np.ndarray) -> None:
        """Remember an aiming pose: the pose, the point it is for, and the (biased) point look_at was given."""
        self._sent_base, self._sent_point, self._base_point = pose, np.asarray(point, float), self._cmd_point
        self._base_bias = self._bias.copy()
        self._alive_out = False

    def _resend_base(self, t: float, measured: Units, why: str) -> Command | None:
        """The same aim again: holds the head up against the vendor idle (or ends an alive gesture)."""
        if not self._path_ok(measured, self._sent_base):
            self.stats["path_refused"] += 1
            self._blocked = "path_refused"
            self._next_try_t = t + self.cfg.retry_s
            return None
        self._alive_out = False
        self._cmd_point, self._cmd_kind = self._base_point, "hold"
        self.stats["keep_alive"] += 1
        return self._move(self._sent_base, why)

    def _alive(self, t: float, measured: Units, ref: Units, origin: np.ndarray) -> Command | None:
        """A tiny, slow, bounded gesture at the person: a nod (aim a little above or below the face) or a
        curious tilt of the gaze to one side. Everything still comes from look_at and passes check."""
        lo, hi = self.cfg.alive_every_s
        self._next_alive_t = t + float(self._rng.uniform(lo, hi))
        centre = self._sent_point
        if centre is None:
            return None
        nod = bool(self._rng.random() < 0.5)
        sign = float(self._rng.choice([-1.0, 1.0]))
        offset = _Z if nod else _unit(np.cross(centre - origin, _Z))
        pose = self._pose_toward(t, self._biased(centre + offset * sign * self.cfg.alive_offset_m, origin), ref,
                                 measured)
        if pose is None:
            return None
        self._alive_out, self._cmd_kind = True, "hold"
        self.stats["alive"] += 1
        return self._move(pose, f"alive: a small {'nod' if nod else 'tilt'}")

    def _preempt(self, t: float, measured: Units, ref: Units, origin: np.ndarray, aim: np.ndarray) -> Command | None:
        """Negative control: a newer move while one runs, every preempt_interval_s, if the target moved."""
        if t - self._last_issue_t < self.cfg.preempt_interval_s:
            return None
        self._blocked = self._gate(t)
        if self._blocked:
            return None
        if self._sent_point is not None:
            if _angle_deg(self._sent_point - origin, aim - origin) < self.cfg.preempt_min_change_deg:
                self._blocked = "target_unchanged"
                return None
        pose = self._pose_toward(t, self._biased(aim, origin), ref, measured)
        if pose is None:
            return None
        self._set_base(pose, aim)
        self.stats["preempt"] += 1
        return self._move(pose, "preempt: newer target while moving")

    def _lock_clip(self, t: float, measured: Units, ref: Units, origin: np.ndarray, tr: _Track,
                   need: bool) -> Command | None:
        """A short clip along the predicted path: knots from look_at every clip_knot_s after the vendor's
        entry move, a cubic through them that starts and ends at rest, every frame checked."""
        if not need and not self.cfg.keep_alive:
            return None
        start = t + self.cfg.clip_entry_s + self.cfg.clip_start_s
        first = self._pose_toward(t, self._biased(self._clamp_point(self._predict(tr, start)), origin), ref, measured)
        if first is None:
            return None
        knots, last_point, last_aim = [first], self._clamp_point(self._predict(tr, start)), self._cmd_point
        steps = int(round(self.cfg.clip_path_s / self.cfg.clip_knot_s))
        # a cubic through the knots can run about 1.5 x faster than knot to knot: leave that headroom
        limit = self.cfg.clip_max_speed_u_s * self.cfg.clip_knot_s / 1.5
        for k in range(1, steps + 1):
            point = self._clamp_point(self._predict(tr, start + k * self.cfg.clip_knot_s))
            biased = self._biased(point, origin)
            pose = self._look(biased, knots[-1])
            if pose is None or self._pose_ok(pose) or self._aim_error(pose, biased) > self.cfg.max_aim_error_deg \
                    or max(abs(pose[j] - knots[-1][j]) for j in JOINTS) > limit:
                break                                     # the path ends where it stops being allowed
            knots.append(pose)
            last_point, last_aim = point, biased
        frames = self._hermite(knots)
        if frames is None:
            self._blocked = "clip_refused"
            self._next_try_t = t + self.cfg.retry_s
            return None
        self._cmd_point = last_aim                        # the clip ends at rest on its last knot
        self._set_base(knots[-1], last_point)
        self._replace_at = t + self._clip_replace_after(len(frames))
        self.stats["turn"] += 1
        return Command(kind="clip", frames=frames, why=f"track: clip along the predicted path ({len(knots)} knots)")

    def _hermite(self, knots: list[Units]) -> list[tuple[float, Units]] | None:
        dt = self.cfg.clip_knot_s
        q = np.array([[k[j] for j in JOINTS] for k in knots])
        if len(q) == 1:                                   # nothing to follow: hold for the path length
            q = np.vstack([q, q])
            dt = self.cfg.clip_path_s
        m = np.zeros_like(q)                              # tangents: at rest at both ends
        for i in range(1, len(q) - 1):
            m[i] = (q[i + 1] - q[i - 1]) / (2 * dt)
        n = int(round(dt * self.cfg.clip_fps))
        frames = [(0.0, {j: float(v) for j, v in zip(JOINTS, q[0], strict=True)})]
        for i in range(len(q) - 1):
            for k in range(1, n + 1):
                s = k / n
                h00, h10 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s
                h01, h11 = -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
                v = h00 * q[i] + h10 * dt * m[i] + h01 * q[i + 1] + h11 * dt * m[i + 1]
                pose = {j: round(float(x), 3) for j, x in zip(JOINTS, v, strict=True)}
                if not self._frame_ok(pose):
                    return None
                frames.append(((i * n + k) / self.cfg.clip_fps, pose))
        return frames if self._clip_speed_ok(frames) else None

    # ---------------------------------------------------------------- LOST
    def _lost_step(self, t: float, measured: Units, busy: bool) -> Command | None:
        tr = self._locked
        if tr is not None and tr.t > self._lost_seen_t and t - tr.t <= self.cfg.fresh_s and self._confirmed(tr):
            # the same track is back (by 3D proximity to where it was going): carry on with it
            self.stats["reacquire"] += 1
            self._set_state(ACQUIRE, t)
            self._acquire_successes = 0
            return self._follow_step(t, measured, busy)
        if t - self._lost_since >= self.cfg.lost_timeout_s:
            self.stats["search_again"] += 1
            self._release(t)
            return self._search_step(t, measured, busy)
        # the peek may replace a running hold; anything else, and any hold after the peek, waits for the outcome
        if (busy and (self._running != "hold" or self._peeked)) or t - self._lost_since < self.cfg.lost_peek_after_s:
            return None
        self._blocked = self._gate(t)
        if self._blocked:
            return None
        if not self._peeked and tr is not None:
            ref = self._reference(measured)
            point = self._peek_point(tr, ref)
            pose = self._pose_toward(t, self._biased(point, np.asarray(self.kin.head(ref).position, float)), ref,
                                     measured)
            self._peeked = True                       # one peek per loss, whether or not it could be made
            if pose is not None:
                self._set_base(pose, point)
                self.stats["peek"] += 1
                return self._move(pose, "lost: peek toward where they were heading")
            return None
        if self.cfg.keep_alive and self._sent_base is not None:
            return self._resend_base(t, measured, "lost: hold and wait")
        return None

    def _peek_point(self, tr: _Track, ref: Units) -> np.ndarray:
        """Where the person was heading; for someone who was still, a little to the side they were on."""
        speed = float(np.linalg.norm(tr.vel))
        if speed >= self.cfg.still_speed_m_s:
            step = tr.vel * self.cfg.lost_peek_ahead_s
            size = float(np.linalg.norm(step))
            if size > self.cfg.lost_peek_max_m:
                step *= self.cfg.lost_peek_max_m / size
            return self._clamp_point(tr.pos + step)
        h = self.kin.head(ref)
        to = tr.pos - np.asarray(h.position, float)
        right = _unit(np.cross(to, _Z))                   # to the right of the line of sight, horizontal
        sign = 1.0 if float(np.dot(to, np.asarray(h.right, float))) >= 0 else -1.0
        return self._clamp_point(tr.pos + right * sign * self.cfg.lost_peek_side_m)

    # ---------------------------------------------------------------- the report
    def _write_trace(self, t: float, busy: bool, command: Command | None) -> None:
        self._prune_issued(t)
        tr = self._locked if self._state in (ACQUIRE, LOCK, LOST) else None
        self._trace = {
            "t": t, "state": self._state, "strategy": self.strategy, "locked_id": self.locked_id,
            "tracks": len(self._tracks), "confirmed": sum(1 for x in self._tracks if self._confirmed(x)),
            "target_m": None if tr is None else [float(v) for v in tr.pos],
            "target_vel_m_s": None if tr is None else [float(v) for v in tr.vel],
            "aim_m": None if self._aim_point is None else [float(v) for v in self._aim_point],
            "aim_error_deg": self._aim_err, "busy": busy, "pending": self._pending,
            "command": None if command is None else command.kind, "why": "" if command is None else command.why,
            "blocked": self._blocked if command is None else "", "window_used": len(self._issued),
            "failures": self._failures, "aim_bias_deg": [float(v) for v in self._bias],
            # ground truth of the detections on the locked track, for scoring only (tracking_valid)
            "scoring_person_id": None if tr is None else tr.scoring_person_id,
        }
