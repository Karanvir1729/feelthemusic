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
    (yaw = +Y is "lower-right", "looks right"). y comes from base_pitch + elbow_pitch: about START,
    +base_pitch brings the leaned-back shoulder toward upright and +elbow_pitch lifts the elbow, and both
    raise the head (beat_clips: UP = bp +5, el +14 "head up and out"; CROUCH = bp -11, el -33).
  * x follows the head's FACE CENTRE (LOOK_M in front of the wrist pivot along the camera axis): a yaw sway is
    seen as the head swinging to its side, and the pivot alone sits only a few cm off the yaw axis. y follows the
    pivot's HEIGHT. wrist_pitch is deliberately not in y: every UP pose tilts the head down by about as much
    (wp +12) to keep facing the listener, which would cancel the rise people actually see, and the phones do not
    score pitch rotation anyway.
  * Clip CSVs hold COMMANDED values (beat_clips applies a gain per joint because the runtime under-delivers).
    The path is where the head physically goes, so every excursion from neutral is multiplied by DELIVERY
    (= 1 / beat_clips.GAINS) first, and t0 includes SERVO_LAG_NS (the arm trails the commanded frames by
    100-150 ms, lamp/live/README.md).

RANGES. +-1 is the edge of the choreography's range, measured on the 312-clip library's manifest
(MANIFEST.example.json, commanded range / gain): the largest delivered yaw excursion is 28.1 units
(median 15), the highest pose is the drop's spring (bp +13, el +24) and the lowest the build's crouch
(bp -11, el -33). With FK the same three poses are pushed through the model to get the range in metres, so
both modes share one set of defaults. Not measured: the ranges in metres on the real arm (no vendor
description on the machine this was written on); `python3 follow_game.py --fk` prints them where there is one.

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
X_RANGE_UNITS = 28.0                                   # delivered base_yaw units = x 1.0 (the library's largest sway)
Y_UP_REF = {"base_pitch": 13.0, "elbow_pitch": 24.0}     # the drop's spring, from START (delivered units)
Y_DOWN_REF = {"base_pitch": -11.0, "elbow_pitch": -33.0}  # the build's crouch
Y_RANGE_UNITS = 44.0                                   # |d base_pitch + d elbow_pitch| = y 1.0 (the crouch)
LOOK_M = 0.10                                          # x is taken this far in front of the wrist pivot (the head's face)
DT_MS, MAX_POINTS, MAX_DATAGRAM, CLIP_NAME_MAX = 50, 80, 1200, 32
SERVO_LAG_NS = 120_000_000                             # the arm trails the commanded frames by 100-150 ms (measured)
STILL_EPS, LEAD_IN_S = 0.004, 0.25                     # trim the clip's holds, keep this much stillness before the first move
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
    """(x range, y range) in metres: the default unit ranges pushed through the model at the neutral pose."""
    n = _pose(neutral)
    x0, z0 = _fk_point(fk, n, look_m)
    xr = abs(_fk_point(fk, {**n, "base_yaw": n["base_yaw"] + X_RANGE_UNITS}, look_m)[0] - x0)
    yr = max(abs(_fk_point(fk, {**n, **{j: n[j] + d for j, d in ref.items()}}, look_m)[1] - z0) for ref in (Y_UP_REF, Y_DOWN_REF))
    return xr, yr


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
                   y = (d base_pitch + d elbow_pitch) / y_range_units
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
        xy = [(yaw_sign * (u["base_yaw"] - n["base_yaw"]) / x_range_units,
               ((u["base_pitch"] - n["base_pitch"]) + (u["elbow_pitch"] - n["elbow_pitch"])) / y_range_units) for u in units]
    return [_clamp1(a) for a, _ in xy], [_clamp1(b) for _, b in xy]


def resample(times_s, x, y, dt_ms: int = DT_MS):
    """Linear interpolation onto a uniform grid of dt_ms starting at times_s[0]: (x[], y[])."""
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


def trim_still(x, y, dt_ms: int = DT_MS, eps: float = STILL_EPS, lead_in_s: float = LEAD_IN_S):
    """Drop the still head and tail of a path (a clip holds START for 0.6 s at both ends, and for the first
    30 % of every beat): (first index kept, x[], y[]). Keeps `lead_in_s` of stillness before the first move,
    so the ball is on screen just before it starts, and one point after the last move (the phone holds a
    path's last point by itself). A path that never moves comes back empty: nothing to follow."""
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


def lpath_messages(seq0: int, t0_mac_ns: int, x, y, clip: str | None = None, dt_ms: int = DT_MS,
                   max_points: int = MAX_POINTS) -> list[dict]:
    """The contract's {"t": "lpath"} messages for one path whose point 0 is at t0_mac_ns (Mac host ns, the
    instant the head is PHYSICALLY there). At most max_points per message (80 x 50 ms keeps a datagram under
    1200 bytes); a longer path becomes consecutive messages seq0, seq0 + 1, ... that SHARE their boundary
    point: message k + 1 starts exactly where and when message k ends, so the phone (a newer path replaces
    older ones from its own t0 on) sees one continuous movement. Fewer than two points: no message."""
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
                 phase=None, repeats: int = 2, sync: bool = False, log=print, seq0: int | None = None, extra_dirs=()):
        self.send, self.offset, self.phase, self.log = send, offset, phase, log
        self.hold_ns, self.lag_ns, self.repeats, self.sync = int(hold_ns), int(lag_ns), max(1, int(repeats)), sync
        self.fk, self.neutral = fk, neutral
        env = [d for d in os.environ.get("FOLLOW_GAME_CLIP_DIRS", "").split(os.pathsep) if d]
        self.dirs = list(dirs) if dirs is not None else env + [d for d in extra_dirs if d] + [
            PACK_DIR, STAGE_DIR, os.path.join(os.path.dirname(os.path.abspath(__file__)), "beat_clips")]
        # seq only has to grow within a conductor session; starting from the wall clock keeps it growing across a
        # restart of this process too (about one message every 2 s, so it never catches up with the clock)
        self.seq = (int(time.time()) if seq0 is None else int(seq0)) & 0x7FFFFFFF
        self.cache: dict = {}
        self.sent, self.skipped, self.end_mac_ns, self.gen = 0, 0, 0, 0
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
                self.log(f"follow game: no forward kinematics ({type(exc).__name__}: {exc}); joint-unit path")
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
            path = trim_still(*resample(clip[0], x, y))
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

    def _publish(self, name: str, beat_ns: int):
        try:
            gen, offset = self.gen, self.offset()
            path = self.path_for(name) if offset is not None else None
            if path is None or gen != self.gen:                    # unknown clip, no clock, or sent home meanwhile
                self.skipped += 1
                return None
            i0, x, y = path
            phase_ns = 0
            if self.phase is not None:
                try: phase_ns = int(max(-200.0, min(200.0, float(self.phase() or 0.0))) * 1e6)
                except Exception: phase_ns = 0
            t0 = beat_ns - self.hold_ns + i0 * DT_MS * 1_000_000 + self.lag_ns + phase_ns + int(offset)
            return self._send(self._messages(t0, x, y, name), name, t0 - (time.monotonic_ns() + int(offset)))
        except Exception as exc:
            self.log(f"follow game: path for {name} not sent ({type(exc).__name__}: {exc})")
            return None

    def _messages(self, t0: int, x, y, clip: str) -> list[dict]:
        with self.lock:                                            # clip and stop threads share the counter
            msgs = lpath_messages(self.seq, t0, x, y, clip=clip)
            self.seq = (self.seq + len(msgs)) & 0xFFFFFFFF
        return msgs

    def _send(self, msgs: list[dict], name: str, lead_ns: int):
        packets = [encode_telemetry(m) for m in msgs]
        if any(len(p) > MAX_DATAGRAM for p in packets):
            raise ValueError("an lpath datagram is over the size limit")
        for r in range(self.repeats):                              # the Mac and the phones de-duplicate by seq
            if r: time.sleep(0.02)
            for p in packets:
                self.send(p)
        if msgs:
            last = msgs[-1]
            self.end_mac_ns = max(self.end_mac_ns, last["t0"] + (len(last["x"]) - 1) * last["dt"] * 1_000_000)
            self.sent += len(msgs)
            self.log(f"{time.strftime('%H:%M:%S')} follow game: lpath {name} seq {msgs[0]['seq']}..{last['seq']} "
                     f"{sum(len(m['x']) for m in msgs)} pts, starts in {lead_ns / 1e6:+.0f} ms")
        return msgs

    def stop(self):
        """The arm was sent home: a two-point still path from now replaces whatever the phones still hold."""
        self.gen += 1                                              # a path still being worked out is for a cancelled clip
        if self.sync:
            return self._stop()
        threading.Thread(target=self._stop, name="follow-path", daemon=True).start()

    def _stop(self):
        try:
            offset = self.offset()
            if offset is None:
                return None
            now_mac = time.monotonic_ns() + int(offset)
            if now_mac >= self.end_mac_ns:
                return None
            msgs = self._send(self._messages(now_mac + 100_000_000, [0.0, 0.0], [0.0, 0.0], "home"), "home", 100_000_000)
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
    pub = PathPublisher(show.send, lambda: show.offset_ns, hold_ns=sched.hold_ns, log=log, **publisher_args)
    light, pending = FollowLight(), {}
    post, post_fn, handle, frame = sched._post, sched.post_fn, show.handle, getattr(panel, "frame", None)

    def _post(name, beat_ns, *rest):
        pending[name] = beat_ns
        return post(name, beat_ns, *rest)

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
    if frame is not None:
        panel.frame = _frame
    show.follow_game = (pub, light)
    log("follow game: on (lpath after every accepted clip, followStatus on the panel)")
    return pub, light


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Print the lpath messages for one library clip. Sends nothing, moves nothing.")
    ap.add_argument("clip", help="beat_<tier>_<bpm>[_<variant>], or any CSV name found in --dir")
    ap.add_argument("--dir", action="append", default=[], help="where to look for <clip>.csv (repeatable)")
    ap.add_argument("--fk", action="store_true", help="use the vendor model (spatial.LampModel) and print the ranges in metres")
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    p = PathPublisher(lambda b: None, lambda: 0, fk="auto" if a.fk else None, dirs=a.dir or None, sync=True, seq0=0)
    if a.fk and p.model() is not None:
        print("workspace (m): x %.4f  y %.4f  at look %.2f m" % (*workspace_m(p.model()), LOOK_M))
    found = p.path_for(a.clip)
    if found is None:
        sys.exit(f"{a.clip}: no CSV in {p.dirs} and not a beat_<tier>_<bpm>[_<variant>] name beat_clips can make")
    i0, xs, ys = found
    msgs = lpath_messages(0, 0, xs, ys, clip=a.clip)
    print(f"{a.clip}: starts {i0 * DT_MS} ms into the clip, {len(xs)} points, x {min(xs):+.2f}..{max(xs):+.2f} (+ = the lamp's own right), "
          f"y {min(ys):+.2f}..{max(ys):+.2f}, {len(msgs)} message(s) of {[len(encode_telemetry(m)) for m in msgs]} bytes")
    print(json.dumps(msgs[0], separators=(",", ":")))
