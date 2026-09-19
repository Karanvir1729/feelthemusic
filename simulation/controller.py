"""Simulation-only scheduling and motion ownership. Written 2026-09-19.

This is NOT the FTM network protocol. An upstream clock adapter supplies due_ns
already converted to local monotonic time, including room latency and output
trim. This module never adds latency or estimates synchronization from arrival.

The caller serializes all calls on one thread. The output callback is a fast,
nonblocking simulator sink; it must validate geometry before animating a motion
and report completion explicitly. No SDK, sockets, model assets or hardware are
accessed here. Passing these tests proves scheduling, not robot collision safety.
"""
from collections import Counter
from copy import deepcopy
import heapq
import math
import time

JOINTS = ("base_yaw", "base_pitch", "elbow_pitch", "wrist_roll", "wrist_pitch")
MODES = ("hold", "follow", "dance")
COMMON_FIELDS = {"session_id", "generation", "seq", "due_ns", "kind"}
KIND_FIELDS = {
    "head_target": {"target_id", "frame", "position_m", "capture_ns", "valid_until_ns"},
    "dance": {"positions", "duration_ns"},
    "light": {"rgb"},
}


def _integer(value):
    return type(value) is int and 0 <= value <= 2**63 - 1


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


class Controller:
    """Bounded deadline queue with one motion owner and an independent light lane.

    Sequence numbers strictly increase across this session, including mode changes.
    A transition clears queued commands, increments generation and releases the
    target lock. A running action remains owned until completion; changing mode
    does not pretend it stopped. Late tolerance defaults to zero nanoseconds.

    head_target position_m is [x right, y forward, z up] in metres in frame
    "lamp_base". Depth is supplied by the caller, never inferred from image size.
    capture_ns and valid_until_ns are local monotonic times. A target must still
    be fresh at dispatch, even when general deadline lateness is tolerated.
    The default 250 ms maximum target age is a simulation design assumption,
    not a measured tracking tolerance; configure it for the intended experiment.
    Target loss must call release_target(); there is no implicit reacquisition.
    dance positions name the SDK's five joints in normalized [-100,100] units;
    duration_ns is a simulation segment duration, at least two seconds.

    Light input is sRGB in [0,1]; this scheduler performs value validation only.
    The final lamp renderer MUST apply the existing safety.flash.FlashLimiter
    to actual emitted output. A Unity preview must remain low brightness. This
    module does not certify flashes, physical LEDs, or vendor fade behavior.
    """

    def __init__(self, session_id, output, *, clock=time.monotonic_ns,
                 capacity=128, late_tolerance_ns=0, max_future_ns=30_000_000_000,
                 max_target_age_ns=250_000_000):
        if not isinstance(session_id, str) or not session_id or len(session_id) > 128:
            raise ValueError("session_id must be a nonempty ephemeral string")
        if not callable(output) or not callable(clock):
            raise ValueError("output and clock must be callable")
        if not _integer(capacity) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        if not all(_integer(v) for v in (late_tolerance_ns, max_future_ns, max_target_age_ns)):
            raise ValueError("timing bounds must be nonnegative integer nanoseconds")
        self.session_id, self.output, self.clock = session_id, output, clock
        self.capacity, self.late_tolerance_ns = capacity, late_tolerance_ns
        self.max_future_ns = max_future_ns
        self.max_target_age_ns = max_target_age_ns
        self.mode, self.armed, self.synced = "hold", False, False
        self.generation, self.last_seq = 0, -1
        self.target_id, self.inflight = None, None
        self.motion_latched = False
        self.stats = Counter()
        self._queue = []

    @property
    def queued(self):
        return len(self._queue)

    def _invalidate(self):
        self.stats["invalidated"] += len(self._queue)
        self._queue.clear()
        self.generation += 1

    def set_mode(self, mode, *, armed=False):
        if mode not in MODES or type(armed) is not bool:
            raise ValueError("invalid mode or arm state")
        if armed and (mode == "hold" or not self.synced or self.motion_latched):
            raise ValueError("cannot arm hold, unsynchronized, or latched motion")
        self._invalidate()
        self.mode, self.armed, self.target_id = mode, armed, None
        return self.generation

    def set_synced(self, synced):
        if type(synced) is not bool:
            raise ValueError("sync state must be boolean")
        if not synced:
            self._invalidate()
            self.armed = False
            self.target_id = None
        self.synced = synced

    def lock_target(self, target_id):
        if (self.mode != "follow" or not self.armed or not self.synced
                or not isinstance(target_id, str) or not target_id or len(target_id) > 128):
            return False
        if self.target_id is not None and self.target_id != target_id:
            return False
        self.target_id = target_id
        return True

    def release_target(self):
        """Call on target loss or explicit release; no automatic person switching."""
        self.target_id = None
        self._invalidate()

    def _reject(self, reason):
        self.stats[reason] += 1
        return False

    def submit(self, command):
        """Validate and copy a command. Rejection never consumes sequence numbers."""
        if not isinstance(command, dict):
            return self._reject("invalid")
        c = command
        kind = c.get("kind")
        if (not isinstance(kind, str) or kind not in KIND_FIELDS
                or set(c) != COMMON_FIELDS | KIND_FIELDS[kind]):
            return self._reject("invalid")
        if (c.get("session_id") != self.session_id
                or not all(_integer(c.get(k)) for k in ("generation", "seq", "due_ns"))
                or c["generation"] != self.generation):
            return self._reject("invalid")
        if not self.synced:
            return self._reject("unsynced")
        if c["seq"] <= self.last_seq:
            return self._reject("reordered")
        now = self.clock()
        if now - c["due_ns"] > self.late_tolerance_ns:
            return self._reject("late")
        if c["due_ns"] - now > self.max_future_ns:
            return self._reject("too_early")
        if kind == "head_target":
            capture, expiry = c["capture_ns"], c["valid_until_ns"]
            if not _integer(capture) or not _integer(expiry):
                return self._reject("invalid_target_time")
            if (capture > now or not capture <= c["due_ns"] <= expiry
                    or now - capture > self.max_target_age_ns
                    or expiry > capture + self.max_target_age_ns or now > expiry):
                return self._reject("stale_target")
            p = c.get("position_m")
            if (self.mode != "follow" or not self.armed or self.motion_latched
                    or self.target_id is None or c.get("target_id") != self.target_id
                    or c.get("frame") != "lamp_base"
                    or not isinstance(p, (list, tuple)) or len(p) != 3
                    or not all(_number(v) for v in p)):
                return self._reject("invalid_motion")
        elif kind == "dance":
            p = c.get("positions")
            if (self.mode != "dance" or not self.armed or self.motion_latched
                    or not isinstance(p, dict) or set(p) != set(JOINTS)
                    or not all(_number(v) and -100 <= v <= 100 for v in p.values())
                    or not _integer(c.get("duration_ns")) or c["duration_ns"] < 2_000_000_000):
                return self._reject("invalid_motion")
        elif kind == "light":
            rgb = c.get("rgb")
            if (self.mode == "hold" or not isinstance(rgb, (list, tuple)) or len(rgb) != 3
                    or not all(_number(v) and 0 <= v <= 1 for v in rgb)):
                return self._reject("invalid_light")
        else:
            return self._reject("invalid")
        if len(self._queue) >= self.capacity:
            return self._reject("capacity")
        self.last_seq = c["seq"]
        heapq.heappush(self._queue, (c["due_ns"], c["seq"], deepcopy(c)))
        self.stats["accepted"] += 1
        return True

    def tick(self):
        """Dispatch only due commands. Busy motion drops, never blocks the light lane."""
        if not self.synced:
            return 0
        fired = 0
        while self._queue:
            now = self.clock()
            if self._queue[0][0] > now:
                break
            due, seq, c = heapq.heappop(self._queue)
            if c["kind"] == "head_target" and (
                    now > c["valid_until_ns"] or now - c["capture_ns"] > self.max_target_age_ns):
                self.stats["stale_target"] += 1
                continue
            if now - due > self.late_tolerance_ns:
                self.stats["late"] += 1
                continue
            motion = c["kind"] != "light"
            if motion and (not self.armed or self.motion_latched or self.inflight is not None):
                self.stats["motion_blocked"] += 1
                continue
            action_id = (self.session_id, c["generation"], seq)
            c["action_id"] = action_id
            if motion:
                self.inflight = action_id
            try:
                self.output(c)
            except Exception:
                self.stats["output_failed"] += 1
                if motion:
                    self.complete(action_id, "unknown")
                    self.motion_latched, self.armed = True, False
                continue
            fired += 1
            self.stats["fired"] += 1
        return fired

    def complete(self, action_id, outcome):
        """Resolve the owned motion; any failed/unknown outcome requires a new controller.

        No automatic unlatch exists. An operator must inspect the simulator state
        and create a fresh disarmed session before trying motion again.
        """
        if outcome not in ("succeeded", "failed", "unknown"):
            raise ValueError("invalid completion outcome")
        if self.inflight is None or action_id != self.inflight:
            return self._reject("invalid_completion")
        self.inflight = None
        self.stats["motion_" + outcome] += 1
        if outcome != "succeeded":
            self.motion_latched, self.armed = True, False
        return True
