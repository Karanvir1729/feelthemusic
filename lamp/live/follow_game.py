#!/usr/bin/env python3
"""Follow the lamp: the lamp's side of the phone game. Contract: docs/follow-game.md.

While the lamp dances, every phone shows the lamp's head movement as a ball with a trail and people move
their phones along with it, like a dance game. This file does the two things the lamp owes that game and
nothing else. It never talks to the runtime, never posts a clip, never moves a joint:

  1. PUBLISH THE PATH. When a clip is posted, tell the conductor where the head is going to be:
     {"t": "lpath", ...} telemetry (type 4) with the head's path in the LAMP'S OWN frame, stamped on the Mac
     clock with the instant the head is physically at point 0. The Mac relays it to the phones.
  2. SHOW THE ROOM'S STATUS. The conductor sends followStatus (type 15) four times a second: green when any
     phone is on track, sliding through yellow and orange to red as the best score falls. status_rgb() /
     FollowLight turn that into a panel colour that glides and can never flash.

INTEGRATION (for whoever owns lamp_show.py's movement). One guarded hook at the end of Show.__init__, off
unless FOLLOW_GAME=1 is in the environment:

    if os.environ.get("FOLLOW_GAME") == "1":
        try: import follow_game; follow_game.attach(self)
        except Exception as exc: print(f"follow game: disabled ({exc})", flush=True)

attach() is the whole integration and leaves the choreography alone. What it does, so you can do it by hand
instead if you change how clips are posted:

    ClipScheduler  a clip `name` was accepted by the runtime for beat instant `beat_ns` (lamp clock; the first
                   frame reaches the servos at beat_ns - hold_ns, the pattern's pose 0 lands on beat_ns):
                       publisher.clip_posted(name, beat_ns)          # AFTER the POST returned, on its own thread
                   -> lpath with t0 = (beat_ns - hold_ns) + [still lead-in trimmed] + servo lag + offset_ns,
                      i.e. the instant the head physically starts the movement, on the Mac clock.
                   and, when the runtime says the clip is running (ClipScheduler._measure, 300-500 ms later):
                       publisher.clip_started(name, beat_ns, first_ns)   # first_ns = its TRUE first servo frame
                   -> the same path again with a corrected t0 (higher seq, so it replaces the guess) when the
                      guess was out by more than CORRECTION_NS. See TIMING below.

TIMING. The POST is planned for beat - start_latency - hold, but the runtime starts the clip 250-450 ms
after the POST (lamp/live/README.md), so the first servo frame lands anywhere in a +-100 ms window around
the plan. The mean of the last 10 measured errors (ClipScheduler.phase_ms) removes the systematic part,
not the per-clip jitter: simulated over a minute at 132 bpm the ball was out of step with the real head by
-91..+89 ms, rms 58 ms, i.e. past the 50 ms budget on about half the clips. _measure knows this clip's true
first frame 300-500 ms after the POST, while point 0 of its path is still 420-720 ms away, so the corrected
path arrives with plenty of lead and the error falls to the clock-sync floor. The guess is published
immediately all the same: a clip whose runtime never answers still gets a path, with the full lead.
    Show.handle    elif t == 15: light.on_packet(d, now)             # store the room's status
    Panel.frame    rgb = light.rgb(rgb, now, self.limiter)           # after safe_output(), unless dark

Any other choreography (not a CSV clip) publishes the same way with three calls:
    x, y = head_path(times_s, joint_frames, fk, neutral); x, y = resample(times_s, x, y)
    for m in lpath_messages(seq, t0_mac_ns, x, y, clip="my_move"): show.send(encode_telemetry(m))

FRAME AND SIGNS (read this before changing a sign; the phone mirrors, the lamp NEVER does).
  The path is in the lamp's own frame: x + = toward the lamp's own right, y + = up, (0, 0) = the neutral
  dance pose (beat_clips.START), +-1 = the edge of the choreography's range. "The lamp's own right" is the
  right-hand side of the lamp as it looks out at the audience (its camera's right, a dancer's own right). A
  person facing the lamp sees that on THEIR left; turning it into "the person's right" is the phone's job
  (PathTimeline in the Mac/iOS code negates x, the one and only place the mirror happens).
  * spatial.py's base frame is metres, z up, forward = +y when base_yaw is at neutral. It is right-handed
    (URDF), so the lamp's own right = forward x up = (+y) x (+z) = +x. Check: LampModel.head() returns
    right = cross(down, forward), and lamp/tests/test_spatial.py asserts right[0] > 0.99 at neutral.
    So with forward kinematics: x_lamp = (base-frame x) - (its value at neutral), y_lamp = (base-frame z) - (...).
  * Without FK (tests, CI, no vendor description) x comes from base_yaw, and +base_yaw = the head turns
    toward the lamp's own right. That is MEASURED on the real lamp (2026-09-19, recorded in
    lamp/tests/test_spatial.py): base_yaw +7.4 units moved the scene by -0.083 picture widths. The scene
    sliding to the picture's left means the camera turned to its own right. beat_clips.py's comments agree
    (yaw = +Y is "lower-right", "looks right"). y comes from elbow_pitch MINUS base_pitch: the two joints
    move the head in OPPOSITE vertical directions. Through the vendor description (and confirmed by an
    independent MuJoCo twin of the same lamp) +10 units of base_pitch move the head DOWN 2.27 cm while +10
    units of elbow_pitch move it UP 2.47 cm, so the crouch (bp -11, el -33) drops the head and the drop's
    spring (bp +13, el +24) raises it. Adding the two terms instead of subtracting them -- which is what
    this file did until 2026-09-19 -- inverts the drop, whose two joints move against each other: the
    spring's ball dived while the head rose. Against the FK path the fallback's y now correlates p50 0.995
    / worst 0.928 over the 312-clip library (it was p50 0.889 / worst 0.649 on the drop).
  * x follows the head's FACE CENTRE (LOOK_M in front of the wrist pivot along the camera axis): a yaw sway is
    seen as the head swinging to its side, and the pivot alone sits only a few cm off the yaw axis. y follows the
    pivot's HEIGHT. wrist_pitch is deliberately not in y: every UP pose tilts the head down by about as much
    (wp +12) to keep facing the listener, which would cancel the rise people actually see, and the phones do not
    score pitch rotation anyway.
  * Clip CSVs hold COMMANDED values (beat_clips applies a gain per joint because the runtime under-delivers).
    The path is where the head physically goes, so every excursion from neutral is multiplied by DELIVERY
    (= 1 / beat_clips.GAINS) first, and t0 includes SERVO_LAG_NS (the arm trails the commanded frames by
    100-150 ms, lamp/live/README.md).

RANGES. +-1 is the edge of the CHOREOGRAPHY's range, not of the arm's workspace (base_yaw alone can take
the head three times further and still pass the safety envelope). It is measured on the whole 312-clip
library (4 tiers x 3 variants x 26 tempos, beat_clips.commanded -> delivered): the boldest delivered yaw
excursion is 28.1 units and the deepest lift |d elbow_pitch - d base_pitch| is 24.2 (the build's crouch at
80 bpm). X_RANGE_UNITS / Y_RANGE_UNITS sit 3 % above those. With FK the same yaw excursion and the
reference poses (Y_UP_REF = the drop's spring, Y_DOWN_REF = the build's crouch) are pushed through the
model, and y is widened by Y_HEADROOM because the real crouch goes past the reference pose. Both modes
then land within about 1 % of each other, so a lamp with no vendor description draws the same size ball.

  measured 2026-09-19 through the vendor description + this lamp's servo calibration, dt 40, all 312 clips:
  x 1.0 = 6.64 cm of head travel, y 1.0 = 7.36 cm.  ZERO points clipped anywhere in the library.
      tier    |x| p50 / max     |y| p50 / max     radial p50 / max
      groove  0.53 / 0.99       0.27 / 0.44       0.58 / 1.00
      hype    0.57 / 0.97       0.58 / 0.63       0.79 / 1.14
      drop    0.59 / 0.97       0.36 / 0.63       0.63 / 1.10
      build   0.26 / 0.46       0.92 / 0.97       0.98 / 1.03
  So a typical clip fills 0.58-0.79 of the range and the boldest (the build's crouch, the hype's accent)
  touches the edge without ever clipping. Before this scale the build spent up to a third of every clip
  welded to y = -1.000 (1639 clipped points library-wide, runs of up to 1.9 s): a dead horizontal line
  along the bottom of the screen during the build, the moment right before the drop.
  `python3 follow_game.py <clip> --fk` prints the ranges in metres where there is a vendor description.

CONTINUITY. The scheduler rests 1-3 beats between clips (ClipScheduler.plan: the next beat must be at
least HOLD + START_JITTER after the boundary) and each clip holds START for 0.6 s at each end, so the
head really is still for 0.5-1.4 s every 8 beats. A path that stops at the last movement leaves that time
uncovered, and the phone drops the ball off the screen 0.4 s later (PathTimeline.holdNs) -- the trail
collapsed to a dot, vanished and popped back at centre 13 times a minute. So every clip's path is followed
by a REST PATH: still points at the clip's own last position (the START pose, i.e. the centre) on the
contract's coarsest grid (REST_DT_MS, about ten points and 150 bytes), covering from the end of the clip's
movement past the earliest instant the next clip's path can start. Overshooting is free:
the next clip's path has a higher seq and owns the time from its own t0. Same for `home`, which now covers
the whole remainder of the path it cancels (the phone gives the ball to the newest path that has started,
so a 2-point home used to end and take the ball with it) and ramps to centre over HOME_RAMP_S instead of
teleporting there.

LIGHT SAFETY. The status colour never pulses: it moves at most RATE_PER_S (and MAX_STEP per call), so a 20 %
swing and back takes half a second, under the 3-per-second rule whatever the conductor sends, and red is only
ever reached by a glide. Taking the panel over from the music colours is one large change, so it spends one
of the FlashLimiter's three onsets a second and waits when there is none left. The status colour is shown at
FOLLOW_VALUE of full scale and the DROP burst's extra brightness is not applied to it.

Standard library only (numpy arrays are accepted, numpy is never imported here).

    python3 follow_game.py beat_hype_120_a          # the lpath messages for one library clip (needs beat_clips)
    python3 follow_game.py beat_hype_120_a --fk     # the same through the vendor model, and the ranges in metres
"""
from __future__ import annotations

import json, math, os, sys, threading, time

T_TELEMETRY, T_LAMP_PATH, T_FOLLOW_STATUS = 4, 14, 15
JOINTS = ("base_yaw", "base_pitch", "elbow_pitch", "wrist_roll", "wrist_pitch")        # spatial.JOINTS = the CSV's column order
START_POSE = {"base_yaw": 0.0, "base_pitch": -49.0, "elbow_pitch": -22.0, "wrist_roll": 0.0, "wrist_pitch": 30.0}   # beat_clips.START
DELIVERY = {"base_yaw": 1 / 1.8, "base_pitch": 1 / 1.15, "elbow_pitch": 1 / 1.2, "wrist_roll": 1 / 1.3, "wrist_pitch": 1 / 1.1}   # 1 / beat_clips.GAINS
X_RANGE_UNITS = 29.0                                   # delivered base_yaw units = x 1.0 (the library's boldest sway is 28.1)
Y_UP_REF = {"base_pitch": 13.0, "elbow_pitch": 24.0}     # the drop's spring, from START (delivered units)
Y_DOWN_REF = {"base_pitch": -11.0, "elbow_pitch": -33.0}  # the build's crouch
Y_RANGE_UNITS = 25.0                                   # |d elbow_pitch - d base_pitch| = y 1.0 (the crouch reaches 24.2)
Y_HEADROOM = 1.10                                      # with FK: the crouch goes this far past Y_DOWN_REF through the model
LOOK_M = 0.10                                          # x is taken this far in front of the wrist pivot (the head's face)
DT_MS, MAX_POINTS, MAX_DATAGRAM, CLIP_NAME_MAX = 40, 80, 1200, 32
SERVO_LAG_NS = 120_000_000                             # the arm trails the commanded frames by 100-150 ms (measured)
CORRECTION_NS = 15_000_000                             # re-publish a clip's path when the measured start moves t0 by more
STILL_SPEED, LEAD_IN_S = 0.08, 0.25                    # units/s that counts as moving (the scorer's own floor), and the
                                                       # stillness kept before the first move so the ball is on screen first
REST_DT_MS, REST_BEATS, REST_MIN_S, REST_MAX_S = 250, 2, 1.5, 4.0   # the still path published for the rest between clips
HOME_RAMP_S, HOME_LEAD_S = 0.6, 0.1                    # `home` glides to centre over this, starting this far ahead
FRESH_S, FOLLOW_VALUE, RATE_PER_S, MAX_STEP, RELEASE_S = 1.0, 0.6, 210.0, 12.0, 3.0
PACK_DIR = os.path.expanduser("~/lelamp-hackathon-2026/static/robots/lelamp_v1/pi5_feetech_r1/animations/factory_v1")
STAGE_DIR = os.path.expanduser("~/feelthemusic-lamp/beat_clips")


# ---- the head's path (pure) -----------------------------------------------------------------------------
def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and v not in (math.inf, -math.inf)


def _clamp1(v: float) -> float:
    return 0.0 if v != v else -1.0 if v < -1.0 else 1.0 if v > 1.0 else v


def _pose(p) -> dict:
    """A pose as {joint: units}: a dict (missing joints = START) or five numbers in JOINTS order."""
    if p is None:
        return dict(START_POSE)
    if isinstance(p, dict):
        return {j: float(p.get(j, START_POSE[j])) for j in JOINTS}
    vals = [float(v) for v in p]
    if len(vals) != len(JOINTS):
        raise ValueError(f"a pose needs {len(JOINTS)} joints in the order {JOINTS}, got {len(vals)}")
    return dict(zip(JOINTS, vals))


def delivered(row, neutral: dict, delivery: dict | None = None) -> dict:
    """Commanded joint values -> where the arm really goes: neutral + (commanded - neutral) * delivery.
    A NaN or infinite value counts as neutral (no excursion): a bad number must not become a wild path."""
    cmd = {j: float(row.get(j, neutral[j])) for j in JOINTS} if isinstance(row, dict) else _pose(row)
    d = DELIVERY if delivery is None else delivery
    return {j: neutral[j] + ((cmd[j] - neutral[j]) if math.isfinite(cmd[j]) else 0.0) * float(d.get(j, 1.0)) for j in JOINTS}


def _fk_point(fk, units: dict, look_m: float) -> tuple[float, float]:
    """(base-frame x of the head's face centre, base-frame z of the wrist pivot). `fk` is a spatial.LampModel
    (anything with .head(units)) or a callable units -> that dict, or -> an (x, y, z) position."""
    h = fk.head(units) if hasattr(fk, "head") else fk(units)
    p, f = (h["position"], h.get("forward")) if isinstance(h, dict) else (h, None)
    x, z = float(p[0]) + (look_m * float(f[0]) if f is not None and look_m else 0.0), float(p[2])
    if not (math.isfinite(x) and math.isfinite(z)):
        raise ValueError("forward kinematics returned a non-finite head position")
    return x, z


def workspace_m(fk, neutral=None, look_m: float = LOOK_M) -> tuple[float, float]:
    """(x range, y range) in metres for +-1: the reference excursions pushed through the model at the
    neutral pose. X_RANGE_UNITS already sits above the library's boldest sway, but the reference POSES do
    not bound y -- through the model the build's crouch goes 6.5 % deeper than Y_DOWN_REF and the library's
    deepest clip 10 % -- so y is widened by Y_HEADROOM. Without it the build spends up to a third of every
    clip clamped at y = -1.000 (1639 clipped points library-wide). See RANGES."""
    n = _pose(neutral)
    x0, z0 = _fk_point(fk, n, look_m)
    xr = abs(_fk_point(fk, {**n, "base_yaw": n["base_yaw"] + X_RANGE_UNITS}, look_m)[0] - x0)
    yr = max(abs(_fk_point(fk, {**n, **{j: n[j] + d for j, d in ref.items()}}, look_m)[1] - z0) for ref in (Y_UP_REF, Y_DOWN_REF))
    return xr, yr * Y_HEADROOM


def head_path(times_s, joint_frames, fk=None, neutral=None, *, delivery: dict | None = None, look_m: float = LOOK_M,
              x_range_m: float | None = None, y_range_m: float | None = None,
              x_range_units: float = X_RANGE_UNITS, y_range_units: float = Y_RANGE_UNITS, yaw_sign: float = 1.0):
    """The head's path in the LAMP'S OWN frame, one point per frame: (x[], y[]), x + = toward the lamp's own
    right, y + = up, (0, 0) = `neutral` (default beat_clips.START), clamped to -1...1. See the module docstring
    for the frame, the signs and the ranges. NOT mirrored: the phone does that.

    times_s        seconds, strictly increasing, one per frame (only checked here; resample() uses them)
    joint_frames   per frame, COMMANDED joint units: five numbers in JOINTS order (a clip's CSV row, a
                   make_clip row, a numpy row) or a {joint: units} dict
    fk             a spatial.LampModel (or a callable, see _fk_point). None, or any failure inside it, uses the
                   joint-unit mapping: x = yaw_sign * d base_yaw / x_range_units,
                   y = (d elbow_pitch - d base_pitch) / y_range_units
    delivery       fraction of each joint's commanded excursion the arm really makes (default 1 / GAINS);
                   pass {} for frames that are already physical (measured poses)
    x_range_m, y_range_m   metres for +-1 with FK; default workspace_m(fk, neutral)"""
    times, frames = [float(t) for t in times_s], list(joint_frames)
    if len(times) != len(frames):
        raise ValueError(f"{len(times)} times for {len(frames)} frames")
    if any(not b > a for a, b in zip(times, times[1:])):
        raise ValueError("times_s must be strictly increasing")
    n = _pose(neutral)
    units = [delivered(f, n, delivery) for f in frames]
    xy = None
    if fk is not None:
        try:
            x0, z0 = _fk_point(fk, n, look_m)
            auto = workspace_m(fk, n, look_m) if x_range_m is None or y_range_m is None else (0.0, 0.0)
            xr, yr = float(x_range_m or auto[0]), float(y_range_m or auto[1])
            if xr > 1e-4 and yr > 1e-4:
                pts = [_fk_point(fk, u, look_m) for u in units]
                xy = [((px - x0) / xr, (pz - z0) / yr) for px, pz in pts]
        except Exception:
            xy = None                                   # no model, a broken model: the unit mapping below
    if xy is None:
        # y: elbow_pitch MINUS base_pitch. The two joints move the head in opposite vertical directions
        # (+10 units of base_pitch = 2.27 cm DOWN, +10 of elbow_pitch = 2.47 cm UP, vendor description and
        # MuJoCo twin agreeing); adding them inverts the drop, where they move against each other.
        xy = [(yaw_sign * (u["base_yaw"] - n["base_yaw"]) / x_range_units,
               ((u["elbow_pitch"] - n["elbow_pitch"]) - (u["base_pitch"] - n["base_pitch"])) / y_range_units) for u in units]
    return [_clamp1(a) for a, _ in xy], [_clamp1(b) for _, b in xy]


def resample(times_s, x, y, dt_ms: int | None = None):
    """Linear interpolation onto a uniform grid of dt_ms starting at times_s[0]: (x[], y[]).
    dt_ms None = the module's DT_MS *at call time* (never bound into this default: t0's arithmetic reads
    the global, so a default bound at import would silently stamp one grid's times onto another's points)."""
    dt_ms = DT_MS if dt_ms is None else dt_ms
    times = [float(t) for t in times_s]
    if not (len(times) == len(x) == len(y)):
        raise ValueError("times_s, x and y must have the same length")
    if len(times) < 2:
        return list(x), list(y)
    dt, t0 = dt_ms / 1000.0, times[0]
    count = int(math.floor((times[-1] - t0) / dt + 1e-6)) + 1
    ox, oy, k = [], [], 0
    for i in range(count):
        t = t0 + i * dt
        while k < len(times) - 2 and times[k + 1] < t:
            k += 1
        span = times[k + 1] - times[k]
        u = min(1.0, max(0.0, (t - times[k]) / span)) if span > 0 else 0.0
        ox.append(x[k] + (x[k + 1] - x[k]) * u); oy.append(y[k] + (y[k + 1] - y[k]) * u)
    return ox, oy


def trim_still(x, y, dt_ms: int | None = None, speed: float = STILL_SPEED, lead_in_s: float = LEAD_IN_S):
    """Drop the still head and tail of a path (a clip holds START for 0.6 s at both ends, and for the first
    30 % of every beat): (first index kept, x[], y[]). Keeps `lead_in_s` of stillness before the first move,
    so the ball is on screen just before it starts, and one point after the last move. The rest of the
    stillness comes back as the rest path (see CONTINUITY), so nothing is lost by trimming it here.
    A path that never moves comes back empty: nothing to follow.

    `speed` is units per SECOND, not per step: it is the scorer's own floor (docs/follow-game.md: only
    instants where the ball moves faster than 0.08 units/s are scored), so the same movement is kept at
    every dt. As a per-step distance it used to mean 0.08 units/s at dt 50 by coincidence and 0.12 at
    dt 33 -- changing dt silently changed what counted as movement."""
    dt_ms = DT_MS if dt_ms is None else dt_ms
    eps = float(speed) * float(dt_ms) / 1000.0
    n = min(len(x), len(y))
    moving = [i for i in range(1, n) if abs(x[i] - x[i - 1]) + abs(y[i] - y[i - 1]) > eps]
    if not moving:
        return 0, [], []
    a = max(0, moving[0] - 1 - int(round(lead_in_s * 1000.0 / dt_ms)))
    b = min(n, moving[-1] + 2)
    return a, list(x[a:b]), list(y[a:b])


# ---- the wire (pure) ------------------------------------------------------------------------------------
def thousandths(v) -> int:
    """-1...1 -> the contract's integer thousandths, clamped to +-1000; NaN -> 0."""
    v = float(v)
    return 0 if v != v else int(round(max(-1.0, min(1.0, v)) * 1000.0))


def lpath_messages(seq0: int, t0_mac_ns: int, x, y, clip: str | None = None, dt_ms: int | None = None,
                   max_points: int = MAX_POINTS) -> list[dict]:
    """The contract's {"t": "lpath"} messages for one path whose point 0 is at t0_mac_ns (Mac host ns, the
    instant the head is PHYSICALLY there). At most max_points per message (80 x 50 ms keeps a datagram under
    1200 bytes); a longer path becomes consecutive messages seq0, seq0 + 1, ... that SHARE their boundary
    point: message k + 1 starts exactly where and when message k ends, so the phone (a newer path replaces
    older ones from its own t0 on) sees one continuous movement. Fewer than two points: no message.
    dt_ms None = the module's DT_MS at call time (see resample)."""
    dt_ms = DT_MS if dt_ms is None else dt_ms
    dt_ms, max_points = int(min(250, max(20, dt_ms))), int(min(256, max(2, max_points)))
    n, out, s = min(len(x), len(y)), [], 0
    while s < n - 1:
        e = min(n, s + max_points)
        m = {"t": "lpath", "seq": (int(seq0) + len(out)) & 0xFFFFFFFF, "t0": int(t0_mac_ns) + s * dt_ms * 1_000_000,
             "dt": dt_ms, "x": [thousandths(v) for v in x[s:e]], "y": [thousandths(v) for v in y[s:e]]}
        if clip:
            m["clip"] = str(clip)[:CLIP_NAME_MAX]
        out.append(m)
        s = e - 1
    return out


def encode_telemetry(obj: dict) -> bytes:
    """Type byte 4 + compact JSON: what the conductor's telemetry handler reads."""
    return bytes([T_TELEMETRY]) + json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


# ---- the room's status (pure) ---------------------------------------------------------------------------
def level_rgb(level: float) -> tuple[int, int, int]:
    """The contract's colour: hue = level x 120 degrees at full saturation. 0 = red, 0.5 = yellow, 1 = green.
    The same arithmetic as FollowColor.rgb255 on the Mac and the phones."""
    l = float(level)
    h = 2.0 * (0.0 if l != l else min(1.0, max(0.0, l)))
    r, g = (1.0, h) if h <= 1.0 else (2.0 - h, 1.0)
    return int(round(r * 255)), int(round(g * 255)), 0


def parse_follow_status(packet: bytes, now: float | None = None) -> dict | None:
    """A followStatus packet (type 15 + JSON) -> {"seq", "ts", "n", "nOk", "best", "ok", "level", "rgb",
    "idle", "rx"}, or None for anything else or anything malformed. `rx` is when it arrived (time.monotonic()
    unless given): freshness is judged against it. `rgb` falls back to level_rgb(level) when absent or
    unusable; `idle` is the conductor's flag, or no phone playing (n = 0)."""
    if not packet or packet[0] != T_FOLLOW_STATUS:
        return None
    try:
        j = json.loads(bytes(packet[1:]).decode("utf-8"))
    except Exception:
        return None
    if not isinstance(j, dict) or not _finite(j.get("level")):
        return None
    level = min(1.0, max(0.0, float(j["level"])))
    num = lambda k, default=0: j[k] if _finite(j.get(k)) else default
    rgb = j.get("rgb")
    if isinstance(rgb, (list, tuple)) and len(rgb) == 3 and all(_finite(c) for c in rgb):
        rgb = tuple(int(min(255, max(0, round(c)))) for c in rgb)
    else:
        rgb = level_rgb(level)
    n = int(num("n"))
    return {"seq": int(num("seq")), "ts": int(num("ts")), "n": n, "nOk": int(num("nOk")),
            "best": min(1.0, max(0.0, float(num("best", 0.0)))), "ok": j.get("ok") is True, "level": level, "rgb": rgb,
            "idle": j.get("idle") is True or n <= 0, "rx": time.monotonic() if now is None else float(now)}


def status_active(status: dict | None, now: float) -> bool:
    """Fresh (under FRESH_S old) and somebody is playing."""
    return bool(status) and not status.get("idle") and 0.0 <= now - status.get("rx", -1e9) < FRESH_S


class FollowLight:
    """The panel's colour while the game runs. rgb() is asked once per rendered frame with the music colour
    (what safe_output() produced) and answers with what to show: the music colour untouched while there is
    no fresh, non-idle status; otherwise the status colour at FOLLOW_VALUE, reached and left by a glide of at
    most RATE_PER_S (0-255 units per second, and MAX_STEP per call whatever the call rate). Engaging spends
    one FlashLimiter onset (it is one large change) and waits while the limiter refuses. After the status
    goes stale or idle the colour glides back onto the music colour and lets go (at the latest RELEASE_S
    later). on_packet() is called from the network thread, rgb() from the render thread: the only shared
    state is one dict reference."""
    def __init__(self):
        self.status, self.engaged = None, False
        self.out, self.last, self.release_since = [0.0, 0.0, 0.0], None, None
        self.packets = 0

    def on_packet(self, d: bytes, now: float | None = None) -> bool:
        st = parse_follow_status(d, now)
        if st is not None:
            self.status, self.packets = st, self.packets + 1
        return st is not None

    def rgb(self, music_rgb, now: float | None = None, limiter=None) -> tuple[int, int, int]:
        now = time.monotonic() if now is None else float(now)
        music = [min(255.0, max(0.0, float(c))) for c in music_rgb]
        st = self.status                                          # one read: the network thread may replace it
        active = status_active(st, now)
        dt = 0.02 if self.last is None else min(0.1, max(0.0, now - self.last))
        self.last = now
        if active and not self.engaged and (limiter is None or limiter.allow(now)):
            self.engaged, self.out, self.release_since = True, list(music), None
        if not self.engaged:
            return tuple(int(c) for c in music)
        target = [c * FOLLOW_VALUE for c in st["rgb"]] if active else music
        step = min(MAX_STEP, RATE_PER_S * dt)
        self.out = [o + max(-step, min(step, t - o)) for o, t in zip(self.out, target)]
        if active:
            self.release_since = None
        else:
            self.release_since = now if self.release_since is None else self.release_since
            if max(abs(t - o) for o, t in zip(self.out, target)) < 1.0 or now - self.release_since > RELEASE_S:
                self.engaged, self.release_since = False, None
                return tuple(int(c) for c in music)
        return tuple(int(round(c)) for c in self.out)


_LIGHT = FollowLight()


def status_rgb(status: dict | None, music_rgb, now: float | None = None, limiter=None, light: FollowLight | None = None):
    """What the panel should show: the contract's rgb at the panel's safe level while `status` (from
    parse_follow_status) is fresh and not idle, else the music colour. The change per call is clamped (see
    FollowLight, which holds the last colour; one module-level instance unless `light` is given), and it is
    meant to be called AFTER safe_output() with the show's FlashLimiter, never instead of them."""
    light = _LIGHT if light is None else light
    light.status = status
    return light.rgb(music_rgb, now, limiter)


# ---- publishing a clip's path (files, a socket; still no robot) -----------------------------------------------
def load_clip_csv(name: str, dirs) -> tuple[list[float], list[list[float]]] | None:
    """(times from the first frame, commanded joint rows) of the clip CSV the runtime plays, or None. The
    header names the columns (timestamp, <joint>.pos); rows that do not parse are skipped."""
    for d in dirs:
        path = os.path.join(d, name + ".csv")
        try:
            with open(path) as f:
                lines = f.read().splitlines()
        except OSError:
            continue
        head = [h.strip() for h in lines[0].split(",")] if lines else []
        try:
            cols = [head.index("timestamp")] + [head.index(j + ".pos") for j in JOINTS]
        except ValueError:
            continue
        times, rows = [], []
        for line in lines[1:]:
            parts = line.split(",")
            try:
                vals = [float(parts[c]) for c in cols]
            except (ValueError, IndexError):
                continue
            if times and vals[0] <= times[-1]:
                continue
            times.append(vals[0]); rows.append(vals[1:])
        if len(times) >= 2:
            return [t - times[0] for t in times], rows
    return None


def clip_bpm(name: str) -> float | None:
    """The tempo a beat_<tier>_<bpm>[_<variant>] clip was generated for, or None. The rest between clips is
    a whole number of beats, so the publisher needs it to know how long to keep the ball on screen."""
    p = str(name).split("_")
    if len(p) not in (3, 4) or p[0] != "beat":
        return None
    try:
        bpm = float(p[2])
    except ValueError:
        return None
    return bpm if 20.0 <= bpm <= 400.0 else None


def generate_clip(name: str) -> tuple[list[float], list[list[float]]] | None:
    """beat_<tier>_<bpm>[_<variant>] regenerated with beat_clips (pure maths, needs numpy): the fallback when
    the CSV is not where we look. None for any other name or any failure."""
    p = name.split("_")
    if len(p) not in (3, 4) or p[0] != "beat":
        return None
    try:
        import beat_clips as bc
        U = bc.commanded(p[1], float(p[2]), p[3] if len(p) == 4 else bc.ALIAS_VARIANT)
        return [i / bc.FPS for i in range(len(U))], [[float(v) for v in u] for u in U]
    except Exception:
        return None


class PathPublisher:
    """Turns "clip `name` was accepted for beat `beat_ns`" into lpath telemetry. Everything runs on a short
    daemon thread started AFTER the runtime accepted the clip, so nothing here can delay a post. `send` takes
    bytes (Show.send); `offset` returns conductor ns - lamp monotonic ns (Show.offset_ns) or None; `phase`
    optionally returns the scheduler's mean measured first-frame error in ms (added to t0, clamped +-200)."""
    def __init__(self, send, offset, hold_ns: int = 600_000_000, fk="auto", neutral=None, dirs=None, lag_ns: int = SERVO_LAG_NS,
                 phase=None, repeats: int = 2, sync: bool = False, log=print, seq0: int | None = None, extra_dirs=(),
                 dt_ms: int | None = None, clip_beats: int = 8):
        self.send, self.offset, self.phase, self.log = send, offset, phase, log
        self.hold_ns, self.lag_ns, self.repeats, self.sync = int(hold_ns), int(lag_ns), max(1, int(repeats)), sync
        self.dt_ms = int(min(250, max(20, DT_MS if dt_ms is None else dt_ms)))
        self.clip_beats = int(clip_beats)                          # lamp_show.CLIP_BEATS: how long a clip's pattern is
        self.fk, self.neutral = fk, neutral
        env = [d for d in os.environ.get("FOLLOW_GAME_CLIP_DIRS", "").split(os.pathsep) if d]
        self.dirs = list(dirs) if dirs is not None else env + [d for d in extra_dirs if d] + [
            PACK_DIR, STAGE_DIR, os.path.join(os.path.dirname(os.path.abspath(__file__)), "beat_clips")]
        # seq only has to grow within a conductor session; starting from the wall clock keeps it growing across a
        # restart of this process too (about one message every 2 s, so it never catches up with the clock)
        self.seq = (int(time.time()) if seq0 is None else int(seq0)) & 0x7FFFFFFF
        self.cache: dict = {}
        self.sent, self.skipped, self.end_mac_ns, self.gen = 0, 0, 0, 0
        self.last_path = None                                      # (t0_mac_ns, dt_ms, x[], y[]) of the last CLIP path
        self.pending = None                                        # (name, beat_ns, t0) of the path published from the POST
        self.corrected = 0
        self.lock = threading.Lock()

    def model(self):
        """The vendor model, read at run time from the lamp's own checkout (never copied); None = unit mapping."""
        if isinstance(self.fk, str):                               # "auto": not tried yet
            try:
                from spatial import LampModel
                self.fk = LampModel()
                self.log(f"follow game: head path through the lamp's model ({self.fk.scale_source})")
            except Exception as exc:
                self.fk = None
                # Loud on purpose: the joint-unit path is an approximation of the same movement (y correlates
                # 0.99 with the model's over the library, worst clip 0.93), not a failure the operator can see
                # on the phones. If this line is in the log, the vendor description was not readable.
                self.log(f"follow game: WARNING no forward kinematics ({type(exc).__name__}: {exc}) -- "
                         f"falling back to the joint-unit path, which is approximate on the drop")
        return self.fk

    def path_for(self, name: str):
        """(first kept index on the DT_MS grid from the clip's first frame, x[], y[]) or None; cached by file."""
        key = None
        for d in self.dirs:
            try:
                st = os.stat(os.path.join(d, name + ".csv")); key = (d, st.st_mtime_ns, st.st_size); break
            except OSError:
                continue
        hit = self.cache.get(name)
        if hit is not None and hit[0] == key:
            return hit[1]
        clip = load_clip_csv(name, self.dirs) or generate_clip(name)
        path = None
        if clip is not None:
            x, y = head_path(clip[0], clip[1], self.model(), self.neutral)
            path = trim_still(*resample(clip[0], x, y, self.dt_ms), dt_ms=self.dt_ms)
            path = path if len(path[1]) >= 2 else None
        if len(self.cache) > 64:
            self.cache.clear()
        self.cache[name] = (key, path)
        return path

    def clip_posted(self, name: str, beat_ns: int):
        """The runtime accepted `name`; its first frame reaches the servos at beat_ns - hold_ns (lamp clock)."""
        if self.sync:
            return self._publish(name, int(beat_ns))
        threading.Thread(target=self._publish, args=(name, int(beat_ns)), name="follow-path", daemon=True).start()

    def clip_started(self, name: str, beat_ns: int, first_ns: int):
        """The runtime reported `name` running: `first_ns` is when its first frame REALLY reached the servos
        (lamp clock), which the POST could only guess at within about +-90 ms. Re-publishes the same path
        with the corrected t0 when that moves the ball by more than CORRECTION_NS and point 0 is still
        ahead; a newer seq replaces the guess on the phones. See TIMING."""
        if self.sync:
            return self._correct(name, int(beat_ns), int(first_ns))
        threading.Thread(target=self._correct, args=(name, int(beat_ns), int(first_ns)), name="follow-path", daemon=True).start()

    def _correct(self, name: str, beat_ns: int, first_ns: int):
        try:
            pending, offset = self.pending, self.offset()
            if pending is None or offset is None or pending[0] != name or pending[1] != beat_ns:
                return None                                        # not the clip we published (home, a skipped post)
            path = self.path_for(name)
            if path is None:
                return None
            t0 = first_ns + path[0] * self.dt_ms * 1_000_000 + self.lag_ns + int(offset)
            now_mac = time.monotonic_ns() + int(offset)
            if abs(t0 - pending[2]) < CORRECTION_NS or t0 < now_mac + 50_000_000:
                return None                                        # close enough, or the ball is already rolling
            self.corrected += 1
            return self._publish(name, beat_ns, first_ns=first_ns)
        except Exception as exc:
            self.log(f"follow game: correction for {name} not sent ({type(exc).__name__}: {exc})")
            return None

    def _publish(self, name: str, beat_ns: int, first_ns: int | None = None):
        try:
            gen, offset = self.gen, self.offset()
            path = self.path_for(name) if offset is not None else None
            if path is None or gen != self.gen:                    # unknown clip, no clock, or sent home meanwhile
                self.skipped += 1
                return None
            i0, x, y = path
            if first_ns is None:                                   # the POST's guess: the planned first frame, plus
                phase_ns = 0                                       # the mean of the last measured start errors
                if self.phase is not None:
                    try: phase_ns = int(max(-200.0, min(200.0, float(self.phase() or 0.0))) * 1e6)
                    except Exception: phase_ns = 0
                first_ns = beat_ns - self.hold_ns + phase_ns
            t0 = int(first_ns) + i0 * self.dt_ms * 1_000_000 + self.lag_ns + int(offset)
            end = t0 + (len(x) - 1) * self.dt_ms * 1_000_000
            segments = [(t0, self.dt_ms, x, y, name)]
            rest = self._rest_segment(name, end, x[-1], y[-1], i0 + len(x) - 1)
            if rest is not None:
                segments.append(rest)
            self.last_path, self.pending = (t0, self.dt_ms, list(x), list(y)), (name, beat_ns, t0)
            return self._send(self._messages(segments), name, t0 - (time.monotonic_ns() + int(offset)), gen)
        except Exception as exc:
            self.log(f"follow game: path for {name} not sent ({type(exc).__name__}: {exc})")
            return None

    def _rest_segment(self, name: str, end_ns: int, x_last: float, y_last: float, last_index: int):
        """The still path that keeps the ball on screen while the arm rests between clips: it holds the
        clip's last point (the head really is parked at START) from the end of the movement until past the
        earliest moment the next clip's path can start. See CONTINUITY.

        ClipScheduler.plan() will not start the next clip before boundary + hold + START_JITTER, snapped up
        to a beat, so from this clip's frame 0 the next path's point 0 is at the earliest
            (boundary + hold) + [0 .. 1 beat] - hold + [its own lead-in] ,
        and covering boundary + hold + REST_BEATS beats is past that at every tempo in the library (checked
        at 80-180 bpm: 0.7-1.4 s of slack). Overshooting costs nothing -- the next path's higher seq takes
        the ball from its own t0 -- so a refused post or a lost beat lock also keeps the ball for a while
        instead of dropping it."""
        bpm = clip_bpm(name)
        if bpm is None:
            seconds = REST_MIN_S
        else:
            period = 60.0 / bpm
            covered = last_index * self.dt_ms / 1000.0             # seconds of the clip the path reaches
            seconds = (self.clip_beats + REST_BEATS) * period + 2 * self.hold_ns / 1e9 - covered
        seconds = min(REST_MAX_S, max(REST_MIN_S, seconds))
        count = int(math.ceil(seconds * 1000.0 / REST_DT_MS)) + 1
        return (end_ns, REST_DT_MS, [x_last] * count, [y_last] * count, "rest")

    def position_at(self, mac_ns: int) -> tuple[float, float]:
        """Where the ball is at `mac_ns` according to the last clip path published, clamped to its ends
        ((0, 0) before anything). `home` ramps from here instead of teleporting the ball to the centre."""
        last = self.last_path
        if not last:
            return 0.0, 0.0
        t0, dt, x, y = last
        u = (int(mac_ns) - t0) / (dt * 1e6)
        if u <= 0:
            return x[0], y[0]
        if u >= len(x) - 1:
            return x[-1], y[-1]
        i = int(u); f = u - i
        return x[i] + (x[i + 1] - x[i]) * f, y[i] + (y[i + 1] - y[i]) * f

    def _messages(self, segments) -> list[dict]:
        """[(t0, dt_ms, x, y, clip)] -> lpath messages with consecutive seqs, allocated in one go so a
        stop on another thread can never take a seq in the middle of a path."""
        out: list[dict] = []
        with self.lock:                                            # clip and stop threads share the counter
            for t0, dt, x, y, clip in segments:
                msgs = lpath_messages(self.seq, t0, x, y, clip=clip, dt_ms=dt)
                self.seq = (self.seq + len(msgs)) & 0xFFFFFFFF
                out += msgs
        return out

    def _send(self, msgs: list[dict], name: str, lead_ns: int, gen: int | None = None):
        packets = [encode_telemetry(m) for m in msgs]
        if any(len(p) > MAX_DATAGRAM for p in packets):
            raise ValueError("an lpath datagram is over the size limit")
        for r in range(self.repeats):                              # the Mac and the phones de-duplicate by seq
            if r: time.sleep(0.02)
            for p in packets:
                if gen is not None and gen != self.gen:            # sent home while these were going out
                    return msgs
                self.send(p)
        if msgs:
            last = msgs[-1]
            self.end_mac_ns = max(self.end_mac_ns, last["t0"] + (len(last["x"]) - 1) * last["dt"] * 1_000_000)
            self.sent += len(msgs)
            self.log(f"{time.strftime('%H:%M:%S')} follow game: lpath {name} seq {msgs[0]['seq']}..{last['seq']} "
                     f"{sum(len(m['x']) for m in msgs)} pts, starts in {lead_ns / 1e6:+.0f} ms, "
                     f"covers {(self.end_mac_ns - msgs[0]['t0']) / 1e9:.1f} s")
        return msgs

    def stop(self):
        """The arm was sent home: a still path replaces whatever the phones still hold."""
        self.gen += 1                                              # a path still being worked out is for a cancelled clip
        if self.sync:
            return self._stop()
        threading.Thread(target=self._stop, name="follow-path", daemon=True).start()

    def _stop(self):
        """Glide the ball to the centre and hold it there for the remainder of the path being cancelled
        (up to REST_MAX_S: the dance has stopped, so after that the game simply goes quiet). The phone
        gives the ball to the newest path that has STARTED and never hands it back, so a 2-point home
        ended after 40 ms and took the ball off the screen with it while the arm was still on its way;
        and the arm reaches home over a second or so, so the ball ramps there instead of jumping."""
        try:
            offset = self.offset()
            if offset is None:
                return None
            now_mac = time.monotonic_ns() + int(offset)
            t0 = now_mac + int(HOME_LEAD_S * 1e9)
            if now_mac >= self.end_mac_ns:
                return None
            x0, y0 = self.position_at(t0)
            segments, end = [], t0
            if abs(x0) + abs(y0) > 0.01:                           # ramp: cosine ease from here to the centre
                steps = max(2, int(round(HOME_RAMP_S * 1000.0 / self.dt_ms)) + 1)
                ease = [0.5 * (1.0 - math.cos(math.pi * i / (steps - 1))) for i in range(steps)]
                segments.append((t0, self.dt_ms, [x0 * (1 - e) for e in ease], [y0 * (1 - e) for e in ease], "home"))
                end = t0 + (steps - 1) * self.dt_ms * 1_000_000
            hold_ns = max(int(0.5e9), min(int(REST_MAX_S * 1e9), self.end_mac_ns - end))
            count = int(math.ceil(hold_ns / (REST_DT_MS * 1e6))) + 1
            segments.append((end, REST_DT_MS, [0.0] * count, [0.0] * count, "home"))
            msgs = self._send(self._messages(segments), "home", t0 - now_mac)
            self.last_path, self.pending = None, None
            self.end_mac_ns = 0                                     # ended: a second home has nothing left to end
            return msgs
        except Exception as exc:
            self.log(f"follow game: stop not sent ({type(exc).__name__}: {exc})")
            return None


def attach(show, log=print, **publisher_args):
    """Wire the game into a running lamp_show.Show from the outside, without editing its scheduler:
      * ClipScheduler._post is wrapped only to remember the beat instant of the clip being posted, and
        ClipScheduler.post_fn (the injection seam its tests use) to publish AFTER the runtime's answer: the
        original call runs first and its result is returned untouched, so no post is delayed or changed.
        `home` (dance stop, mode change) ends the path on the phones.
      * ClipScheduler._measure is wrapped to publish the same path again with the true first-frame instant
        once the runtime reports the clip running (it already measures it; the wrapper only reads the error
        it appended to phase_ms). The original runs first and its result is returned untouched.
      * Show.handle additionally takes type 15 (the original ignores it).
      * Panel.frame's colour goes through FollowLight with the show's FlashLimiter, unless the panel is dark;
        the DROP burst's extra brightness is not applied to the status colour.
    Every added step is inside try/except: a failure here logs one line and the show goes on as before.
    Returns (publisher, light)."""
    sched, panel = show.scheduler, show.panel
    hw = getattr(sys.modules.get(type(show).__module__), "HW_BRIGHTNESS", 0.6)
    mean_phase = lambda: (sum(sched.phase_ms[-10:]) / len(sched.phase_ms[-10:])) if getattr(sched, "phase_ms", None) else 0.0
    publisher_args.setdefault("phase", mean_phase)
    publisher_args.setdefault("extra_dirs", [getattr(getattr(sched, "live", None), "pack_dir", None)])   # --pack-dir of the live clips
    publisher_args.setdefault("clip_beats", getattr(sys.modules.get(type(sched).__module__), "CLIP_BEATS", 8))
    pub = PathPublisher(show.send, lambda: show.offset_ns, hold_ns=sched.hold_ns, log=log, **publisher_args)
    light, pending = FollowLight(), {}
    post, post_fn, handle, frame = sched._post, sched.post_fn, show.handle, getattr(panel, "frame", None)
    measure = getattr(sched, "_measure", None)

    def _post(name, beat_ns, *rest):
        pending[name] = beat_ns
        return post(name, beat_ns, *rest)

    def _measure(name, beat_ns, t0, *rest):
        before = getattr(sched, "phase_ms", None)
        r = measure(name, beat_ns, t0, *rest)
        try:                                                       # _measure replaces the list when it measured one
            after = getattr(sched, "phase_ms", None)
            if after and after is not before:
                pub.clip_started(name, int(beat_ns), int(beat_ns) - pub.hold_ns + int(float(after[-1]) * 1e6))
        except Exception as exc:
            log(f"follow game: {type(exc).__name__}: {exc}")
        return r

    def _post_fn(name):
        r = post_fn(name)
        try:
            beat_ns = pending.pop(name, None)
            if name == "home":
                pub.stop()
            elif beat_ns is not None and isinstance(r, dict) and r.get("status") == "started":
                pub.clip_posted(name, beat_ns)
        except Exception as exc:
            log(f"follow game: {type(exc).__name__}: {exc}")
        return r

    def _handle(d, now):
        if d and d[0] == T_FOLLOW_STATUS:
            try: light.on_packet(d, now)
            except Exception: pass
            return None
        return handle(d, now)

    def _frame(now):
        rgb, brightness = frame(now)
        try:
            with panel.lock:                                       # the limiter is only ever touched under this lock
                if panel.dark:
                    return rgb, brightness
                out = light.rgb(rgb, now, getattr(show, "limiter", None))
            return out, (min(brightness, hw) if light.engaged else brightness)
        except Exception:
            return rgb, brightness

    sched._post, sched.post_fn, show.handle = _post, _post_fn, _handle
    if measure is not None:
        sched._measure = _measure
    if frame is not None:
        panel.frame = _frame
    show.follow_game = (pub, light)
    log("follow game: on (lpath after every accepted clip, followStatus on the panel)")
    return pub, light


def record(out_path: str, tempos, tiers, variants, pub: "PathPublisher") -> dict:
    """Write the paths of a slice of the clip library as one JSON file: what the phones would be sent for
    that dance, with no clip CSV, no vendor description and no robot anywhere in it. This is what
    apple/tools/dance_replay.py replays into a conductor so the owner can see the real dance on the phones.
    Everything the replay needs to chain clips the way ClipScheduler does is in the header."""
    clips = []
    for bpm in tempos:
        for tier in tiers:
            for v in variants:
                name = f"beat_{tier}_{bpm}_{v}"
                found = pub.path_for(name)
                if found is None:
                    continue
                i0, xs, ys = found
                clips.append({"name": name, "tier": tier, "bpm": int(bpm), "variant": v, "i0": int(i0),
                              "x": [thousandths(a) for a in xs], "y": [thousandths(b) for b in ys]})
    doc = {"v": 1, "made": time.strftime("%Y-%m-%d"), "dt": pub.dt_ms, "hold_ms": int(pub.hold_ns / 1e6),
           "beats": pub.clip_beats, "lag_ms": int(pub.lag_ns / 1e6), "start_jitter_ms": 100,
           "rest": {"dt": REST_DT_MS, "beats": REST_BEATS, "min_s": REST_MIN_S, "max_s": REST_MAX_S},
           "note": ("Head paths of real beat_clips choreography, generated by lamp/live/follow_game.py --record. "
                    "Lamp's own frame, thousandths, x + = the lamp's own right, y + = up, point n is at "
                    "t0 + n*dt. Generated output: no vendor robot description, no clip CSV, no calibration."),
           "clips": clips}
    with open(out_path, "w") as f:
        json.dump(doc, f, separators=(",", ":"), sort_keys=True)
        f.write("\n")
    return doc


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Print the lpath messages for one library clip. Sends nothing, moves nothing.")
    ap.add_argument("clip", nargs="?", help="beat_<tier>_<bpm>[_<variant>], or any CSV name found in --dir")
    ap.add_argument("--dir", action="append", default=[], help="where to look for <clip>.csv (repeatable)")
    ap.add_argument("--fk", action="store_true", help="use the vendor model (spatial.LampModel) and print the ranges in metres")
    ap.add_argument("--record", metavar="OUT.json", help="write a slice of the library's paths for apple/tools/dance_replay.py")
    ap.add_argument("--tempos", default="88,120,152,180", help="--record: bpm buckets, comma separated")
    ap.add_argument("--tiers", default="groove,hype,drop,build", help="--record: tiers, comma separated")
    ap.add_argument("--variants", default="a,b,c", help="--record: variants, comma separated")
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    p = PathPublisher(lambda b: None, lambda: 0, fk="auto" if a.fk else None, dirs=a.dir or None, sync=True, seq0=0)
    if a.record:
        doc = record(a.record, [int(t) for t in a.tempos.split(",")], a.tiers.split(","), a.variants.split(","), p)
        pts = sum(len(c["x"]) for c in doc["clips"])
        print(f"{a.record}: {len(doc['clips'])} clips, {pts} points, {os.path.getsize(a.record)} bytes, dt {doc['dt']} ms"
              f"{'' if p.model() is not None else ' (JOINT-UNIT path: no vendor model on this machine)'}")
        sys.exit(0)
    if not a.clip:
        sys.exit("a clip name, or --record OUT.json")
    if a.fk and p.model() is not None:
        print("workspace (m): x %.4f  y %.4f  at look %.2f m" % (*workspace_m(p.model()), LOOK_M))
    found = p.path_for(a.clip)
    if found is None:
        sys.exit(f"{a.clip}: no CSV in {p.dirs} and not a beat_<tier>_<bpm>[_<variant>] name beat_clips can make")
    i0, xs, ys = found
    msgs = lpath_messages(0, 0, xs, ys, clip=a.clip)
    rest = p._rest_segment(a.clip, 0, xs[-1], ys[-1], i0 + len(xs) - 1)
    print(f"{a.clip}: starts {i0 * DT_MS} ms into the clip, {len(xs)} points at dt {DT_MS} ms, "
          f"x {min(xs):+.2f}..{max(xs):+.2f} (+ = the lamp's own right), y {min(ys):+.2f}..{max(ys):+.2f}, "
          f"{len(msgs)} message(s) of {[len(encode_telemetry(m)) for m in msgs]} bytes, "
          f"then {len(rest[2])} still points covering {(len(rest[2]) - 1) * REST_DT_MS / 1000:.2f} s of rest")
    print(json.dumps(msgs[0], separators=(",", ":")))
