#!/usr/bin/env python3
"""Where the lamp's head is about to be, told to everyone: telemetry {"t": "lpath"}.

The phones draw a ball with a trail and people copy it; the Mac validates the message and relays it to
them (docs/follow-game.md, apple/Shared/FollowGame.swift struct LampPath, apple/Mac/.../Show.swift
lampPath). This file is the lamp's half: turn the clip the scheduler is posting into head positions and
send them, timed on the conductor's clock.

    {"t":"lpath","v":1,"seq":<uint32>,"t0":<uint64 ns>,"dt":<20..250 ms>,"x":[ints],"y":[ints],"clip":"..."}

  t0    MAC HOST ns, a PRESENTATION time: the instant the head is at point 0.
  dt    whole ms between points, 20..250.
  x, y  THOUSANDTHS of a unit clamped to -1000..1000 (1000 = 1.0), 2..256 of each, equal lengths.
        Anything else the Mac refuses outright, and it refuses a relay over 1400 bytes as well.

Frame: the LAMP's own, x + = toward the lamp's right, y + = UP, (0, 0) = the neutral dance pose. The
mirroring for the person facing the lamp happens on the phone (PathTimeline.point negates x) and nowhere
else. The y sign is worth being sure of, because a flipped one asks the room to dance upside down: the
renderer plots `cy - y * sy` (ShowView.swift, FollowFX.draw) so a positive y is drawn ABOVE centre, and
FollowScorer correlates the path's vy with the phone's acceleration along UP, so a flipped y would score
an honest follower at 0. Both say y + = up, which is what the contract's own comment says too.

The path is a nicety and the dance is the product: every failure here (no model, no clock offset, a
missing or corrupt CSV, anything else) is logged ONCE and the clip is posted regardless.
"""
from __future__ import annotations

import json
import math
import os
import threading
from collections import OrderedDict

# How far the head reaches, in metres, for x = y = 1. ONE number for both axes, so the shape the phones
# draw is the shape the head makes. Measured with spatial.LampModel on this lamp's servo calibration over
# the whole of beat_clips' choreography (4 tiers x 3 variants x 80..180 bpm, facings 0 and +-24), as the
# head's displacement from the clip's START pose -- which is what the contract calls (0, 0). Displacement
# in metres: bold 1.0 (the pre-generated library is bold 1.0 and nothing else) dx -0.251..+0.223,
# dz -0.240..+0.114; bold 0.6 (the dashboard's default) dx -0.166..+0.146, dz -0.146..+0.089.
#
# At 0.24 m that comes out on the wire as, per tier (x, then y, in thousandths, and the share of frames
# that hit the clamp):
#     bold 1.0   groove  -825..+735   -692..+325   0.00 %      bold 0.6   -525..+460   -389..+258   0 %
#                hype   -1000..+928   -999..+473   1.28 %                 -690..+609   -607..+369   0 %
#                drop   -1000..+928   -999..+391   1.07 %                 -690..+609   -607..+307   0 %
#                build   -534..+395   -999..  +0   0.00 %                 -430..+325   -607..  +0   0 %
# The loudest tiers fill the range and clip about one frame in eighty, and only at full boldness turned a
# full +-24 units at someone; groove, which plays most, still reaches three quarters of it. A smaller span
# would clip the bars people actually watch (0.20 m clips 14 % of bold-1.0 frames in height alone); a
# larger one would shrink the whole dance on the screen. The dance is not symmetric about the neutral pose
# -- it crouches much further than it rises -- so y runs about -1.0..+0.45, and that is the head telling
# the truth rather than a mapping to correct.
REACH_M = 0.24

# At most this many points per message. The Mac REFUSES a relay over 1400 bytes (it would fragment on a
# venue network, twice per phone), and its relay is our JSON plus "lamp":<index>. Swept over the whole
# choreography at every half-bpm from 80 to 180, the largest datagram 96 points produces is 913 bytes
# (92.5 bpm, 192 frames, dt 67) with a full-width clip, the longest library name and the largest possible
# seq and t0 -- well under the limit, and docs/follow-game.md's "about 100 points" agrees.
MAX_POINTS = 96
MAX_RELAY_BYTES = 1400              # the Mac's own limit; ours must already be under it (see above)
DT_MIN_MS, DT_MAX_MS = 20, 250      # the contract's range for `dt`
CLIP_NAME_CHARS = 32                # what of the clip name is worth a log line on the Mac (it allows 48)

# Finished paths kept by clip name, for the LIBRARY clips only (see LampPathSender.points_for). An entry
# is two lists of at most MAX_POINTS small ints: ~7 kB, so 32 of them is ~0.2 MB, and 32 covers two or
# three bpm buckets of all 4 tiers x 3 variants at once -- far more than one song moves through.
CACHE_MAX = 32


def clamp_thousandths(v: float) -> int:
    """A unit value -> the wire's thousandths, clamped to -1000..1000. Not finite -> 0, as the Mac and the
    phone both do: a NaN that reached a phone would be a ball at a corner, not a ball that is missing."""
    if not math.isfinite(v):
        return 0
    return max(-1000, min(1000, int(round(v * 1000.0))))


def read_clip_rows(path: str, joints) -> list[tuple]:
    """The runtime's animation CSV -> one tuple of joint values per frame, in `joints` order.

    The columns are found by NAME ("<joint>.pos", as beat_clips.CSV_HEADER writes them, or bare
    "<joint>"), never by position: the vendor's own clips sit in the same pack dir and a column order
    assumed rather than read would silently map pitch onto yaw. A file that does not have every joint,
    or that holds something that is not a number, raises -- the caller turns that into one log line."""
    with open(path) as f:
        header = [h.strip() for h in f.readline().split(",")]
        want = []
        for j in joints:
            if f"{j}.pos" in header:
                want.append(header.index(f"{j}.pos"))
            elif j in header:
                want.append(header.index(j))
            else:
                raise ValueError(f"{path}: no column for {j}")
        rows = []
        for line in f:
            if not line.strip():
                continue
            cells = line.split(",")
            rows.append(tuple(float(cells[i]) for i in want))
    if len(rows) < 2:
        raise ValueError(f"{path}: {len(rows)} frame(s), a path needs at least 2 points")
    return rows


class LampPathSender:
    """Sends one {"t": "lpath"} per posted clip. Built by Show, called from the scheduler's post thread.

    `send(bytes)` is the show's UDP send, `clock()` returns Show.offset_ns (conductor host ns MINUS lamp
    monotonic ns) or None while the clock is not shared yet. Nothing here ever raises at the caller."""

    def __init__(self, send, clock, pack_dir: str, model_fn=None, log=print):
        self.send, self.clock, self.pack_dir, self.log = send, clock, pack_dir, log
        self.model_fn = model_fn                  # tests inject a model; on the lamp it is built below
        self.model, self.joints, self.fps, self.origin = None, None, 30.0, None
        self.off = False                          # the model could not be built: no path this run
        self.seq = 0                              # the NEXT path's seq; wraps at 32 bits
        self.sent, self.dropped = 0, 0
        self.cache: "OrderedDict[str, tuple]" = OrderedDict()
        self.said: set[str] = set()               # what has already cost a log line
        self.lock = threading.Lock()              # posts are serialised by the scheduler, but the cache
                                                  # and seq are also read by the panel/status line

    # ---- logging: once each, whatever happens ------------------------------------------------
    def once(self, key: str, message: str):
        with self.lock:
            if key in self.said:
                return
            self.said.add(key)
        self.log(f"follow path: {message}")

    # ---- the model ---------------------------------------------------------------------------
    def load(self) -> bool:
        """The lamp's kinematic model, the joint order and the neutral dance pose, once (~10 ms, plus the
        numpy import when the live-clip generator has not already paid for it -- so the very first path
        of a library-only run may go out after its own t0, which the Mac accepts and the phones join
        from wherever "now" is). beat_clips is where the clips' own START pose and frame rate are
        defined, so they are taken from there rather than copied: a generator that re-poses the dance
        must not leave this file mapping the head around a pose the clips no longer start from."""
        if self.model is not None:
            return True
        if self.off:
            return False
        try:
            from spatial import JOINTS
            import beat_clips
            model = self.model_fn() if self.model_fn is not None else self.build_model()
            self.joints = tuple(JOINTS)
            self.fps = float(beat_clips.FPS)
            start = {j: float(v) for j, v in zip(self.joints, beat_clips.START)}
            self.origin = tuple(float(c) for c in model.head(start)["position"])
            self.model = model
        except Exception as exc:
            self.off = True
            self.once("model", f"no model ({type(exc).__name__}: {exc}); the dance goes on without a path")
            return False
        return True

    @staticmethod
    def build_model():
        """spatial.LampModel on the lamp's own vendor checkout. The servo calibration comes from
        LampModel's own search (LELAMP_CALIBRATION_PATH, then /etc/lelamp/runtime.env, then the
        checkout), which is the same calibration the clip generator validates against, so the metres
        here are the metres the arm was planned in."""
        from spatial import LampModel
        return LampModel()

    # ---- rows -> points ----------------------------------------------------------------------
    def sample_frames(self, frames: int) -> tuple[int, list[int]]:
        """(dt_ms, frame index per point) for a clip of `frames` frames at 30 fps.

        A clip is 0.6 s of hold, 8 beats, 0.6 s of hold: 116 frames (3.9 s) at 180 bpm to 216 frames
        (7.2 s) at 80 bpm. Take every `stride`-th frame, stride = ceil(frames / MAX_POINTS), so the
        count fits one datagram; dt = round(1000 * stride / 30) ms, i.e. 67 ms for stride 2 and 100 ms
        for stride 3, both well inside 20..250. So a 4.9 s clip (147 frames, about 130 bpm) is stride
        2 and 73 points; 80 bpm is stride 3 and 72; 180 bpm is stride 2 and 58.

        The index is the frame nearest k * dt, not k * stride, because dt has to be a whole number of
        ms and 67 is 0.33 ms more than two frames. Indexing by k * stride would advertise every point
        0.33k ms later than the head is really there -- 24 ms by the end of that 147-frame clip, more
        on a longer one -- whereas dt is the contract's spacing in TIME and this is what makes the
        points evenly spaced in it. The cost is the odd last point: 73 rather than the 74 that every
        2nd frame would give, the difference being the last 67 ms of the clip.

        That tail (under one stride, 67-100 ms) is dropped rather than appended at a short dt: it is
        the end of the 0.6 s hold at the START pose, where the head is already still, and the phones
        hold the last point for 0.4 s anyway (PathTimeline.holdNs)."""
        stride = max(1, -(-frames // MAX_POINTS))
        dt_ms = int(round(1000.0 * stride / self.fps))
        if not (DT_MIN_MS <= dt_ms <= DT_MAX_MS):
            raise ValueError(f"{frames} frames at {self.fps:g} fps wants dt {dt_ms} ms, outside {DT_MIN_MS}..{DT_MAX_MS}")
        idx, last = [], frames - 1
        for k in range(MAX_POINTS):
            i = int(round(k * dt_ms * self.fps / 1000.0))
            if i > last:
                break
            idx.append(i)
        if len(idx) < 2:
            raise ValueError(f"{frames} frames gave {len(idx)} point(s)")
        return dt_ms, idx

    def points(self, rows) -> tuple[int, list[int], list[int]]:
        """(dt_ms, x, y) in thousandths for a clip's rows. x from the head's left-right axis and y from
        its height, both as the displacement from the neutral dance pose, over REACH_M, clamped."""
        dt_ms, idx = self.sample_frames(len(rows))
        head, joints = self.model.head, self.joints
        ox, oz = self.origin[0], self.origin[2]
        xs, ys = [], []
        for i in idx:
            p = head({j: float(v) for j, v in zip(joints, rows[i])})["position"]
            # Base frame: z is up and +y is forward at neutral yaw, so +x is the lamp's own right --
            # exactly the contract's x -- and its height is z.
            xs.append(clamp_thousandths((float(p[0]) - ox) / REACH_M))
            ys.append(clamp_thousandths((float(p[2]) - oz) / REACH_M))
        return dt_ms, xs, ys

    def points_for(self, name: str, rows=None) -> tuple[int, list[int], list[int]]:
        """The points for the clip the scheduler just posted.

        A LIVE clip hands its rows over (LiveClip.rows): it was generated in this process seconds ago and
        is converted every time, which costs one forward-kinematics pass (~4 ms for 74 points on the Pi 5)
        against the ~120 ms the generator already spent. It is never cached, because live clips reuse six
        pooled names (live_0..live_5) with different contents -- a cache by name would hand the phones the
        clip before last. A LIBRARY clip is a CSV in the pack dir, ~2 ms to read and parse on top of that
        pass, and the same name is posted over and over, so it IS cached (see CACHE_MAX)."""
        if rows is not None:
            return self.points(rows)
        with self.lock:
            hit = self.cache.get(name)
            if hit is not None:
                self.cache.move_to_end(name)
                return hit
        built = self.points(read_clip_rows(os.path.join(self.pack_dir, name + ".csv"), self.joints))
        with self.lock:
            self.cache[name] = built
            while len(self.cache) > CACHE_MAX:
                self.cache.popitem(last=False)
        return built

    # ---- the one thing the scheduler calls ---------------------------------------------------
    def clip_posted(self, name: str, first_frame_ns: int, rows=None) -> bool:
        """The scheduler has posted `name` and its first frame reaches the servos at `first_frame_ns` on
        the LAMP's monotonic clock. Returns whether a path went out; never raises."""
        try:
            if not self.load():
                return False
            offset_ns = self.clock()
            if offset_ns is None:
                # No shared clock: t0 would be a guess on the wrong clock, the Mac would refuse it as
                # "already ended" or 60 s ahead, and a path drawn at the wrong time is worse than none.
                self.once("clock", "no clock offset from the conductor yet; nothing to send until there is")
                self.dropped += 1
                return False
            dt_ms, xs, ys = self.points_for(name, rows)
            return self.emit(name, int(first_frame_ns) + int(offset_ns), dt_ms, xs, ys)
        except Exception as exc:
            self.dropped += 1
            self.once(f"{name}:{type(exc).__name__}", f"{name}: {type(exc).__name__}: {exc}")
            return False

    def emit(self, name: str, t0_host_ns: int, dt_ms: int, xs: list[int], ys: list[int]) -> bool:
        if t0_host_ns <= 0 or not (2 <= len(xs) <= 256) or len(xs) != len(ys):
            self.dropped += 1
            self.once("shape", f"{name}: {len(xs)} point(s), t0 {t0_host_ns}: not a path the Mac would take")
            return False
        with self.lock:
            seq, self.seq = self.seq, (self.seq + 1) & 0xFFFFFFFF
        msg = {"t": "lpath", "v": 1, "seq": seq, "t0": t0_host_ns, "dt": dt_ms,
               "x": xs, "y": ys, "clip": name[:CLIP_NAME_CHARS]}
        blob = b"\x04" + json.dumps(msg, sort_keys=True, separators=(",", ":")).encode()
        if len(blob) > MAX_RELAY_BYTES:
            # MAX_POINTS is chosen so this cannot happen; if it ever does, the Mac would refuse the relay
            # and nothing would reach the phones anyway, so drop it here where it can be seen.
            self.dropped += 1
            self.once("bytes", f"{name}: {len(blob)} bytes for {len(xs)} points, over the Mac's {MAX_RELAY_BYTES}")
            return False
        self.send(blob)
        self.sent += 1
        return True

    def state(self) -> str:
        """One field for the operator's status line."""
        if self.off:
            return "off"
        return f"{self.sent}/{self.sent + self.dropped}"
