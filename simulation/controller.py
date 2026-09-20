"""Simulation-only scheduling and motion ownership. Written 2026-09-19.

This is NOT the FTM network protocol. An upstream clock adapter supplies due_ns
already converted to local monotonic time, including room latency and output
trim. This module never adds latency or estimates synchronization from arrival.

The caller serializes all calls on one thread. The output callback is a fast,
nonblocking simulator sink; it must validate geometry before animating a motion,
raise on output failure (or call safety_latch), and report completion explicitly.
No SDK, sockets, model assets or hardware are
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
    does not pretend it stopped. Due motion waits for that completion, only while
    its ORIGINAL deadline tolerance and target freshness remain valid. Late
    tolerance defaults to the project's 80 ms event-drop budget, not a new room
    latency. A zero tolerance remains available for exact-clock replay tests.

    Follow updates coalesce to one latest captured target; capture time cannot
    regress within a lock. At capacity a fresh head target may evict the farthest
    future light, so preloaded lighting cannot starve current tracking.
    Sequence jumps are bounded (4096 by default); larger discontinuities require
    a fresh session rather than permanently advancing the replay high-water mark.

    safety_latch() blocks ALL new output, including lights, even when idle. It
    clears pending work but preserves in-flight ownership; only completion can
    resolve that ownership. No automatic reset exists. A tick-driven watchdog
    latches if completion has not arrived by dispatch + duration + margin. The
    head duration (2 s) and margin (1 s) defaults are simulation policies, not
    hardware timing measurements. The caller must keep ticking while unsynced.
    Constructor policy bounds cap queued commands at 4096, lookahead and motion
    durations at 30 s, watchdog margin and target age at 5 s, and late tolerance
    at 1 s. These prevent unbounded resource use/stalls, not prove physical safety.
    Configuration is trusted caller state and must not be mutated from packets.

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
    The final lamp renderer MUST apply safety.flash.FlashLimiter (separate PR #5)
    to actual emitted output. A Unity preview must remain low brightness. This
    module does not certify flashes, physical LEDs, or vendor fade behavior.
    """

    def __init__(self, session_id, output, *, clock=time.monotonic_ns,
                 capacity=128, late_tolerance_ns=80_000_000, max_future_ns=30_000_000_000,
                 max_target_age_ns=250_000_000, head_motion_duration_ns=2_000_000_000,
                 max_motion_duration_ns=30_000_000_000, watchdog_margin_ns=1_000_000_000,
                 max_sequence_jump=4096):
        if not isinstance(session_id, str) or not session_id or len(session_id) > 128:
            raise ValueError("session_id must be a nonempty ephemeral string")
        if not callable(output) or not callable(clock):
            raise ValueError("output and clock must be callable")
        bounds = ((capacity, 1, 4096), (late_tolerance_ns, 0, 1_000_000_000),
                  (max_future_ns, 0, 30_000_000_000), (max_target_age_ns, 0, 5_000_000_000),
                  (max_motion_duration_ns, 2_000_000_000, 30_000_000_000),
                  (head_motion_duration_ns, 2_000_000_000, 30_000_000_000),
                  (watchdog_margin_ns, 0, 5_000_000_000), (max_sequence_jump, 1, 65536))
        if any(not _integer(v) or not lo <= v <= hi for v, lo, hi in bounds):
            raise ValueError("capacity or timing policy outside bounded simulation limits")
        if head_motion_duration_ns > max_motion_duration_ns:
            raise ValueError("head duration exceeds maximum motion duration")
        self.session_id, self.output, self.clock = session_id, output, clock
        self.capacity, self.late_tolerance_ns = capacity, late_tolerance_ns
        self.max_future_ns = max_future_ns
        self.max_target_age_ns = max_target_age_ns
        self.head_motion_duration_ns = head_motion_duration_ns
        self.max_motion_duration_ns = max_motion_duration_ns
        self.watchdog_margin_ns = watchdog_margin_ns
        self.max_sequence_jump = max_sequence_jump
        self.mode, self.armed, self.synced = "hold", False, False
        self.generation, self.last_seq = 0, -1
        self.target_id, self.inflight = None, None
        self.motion_latched = self.safety_latched = False
        self.latch_reason = None
        self._inflight_deadline_ns = None
        self._last_capture_ns = -1
        self.stats = Counter()
        self._queue = []

    def safety_latch(self, reason="external"):
        """Latch all output for a thermal/torque/action/renderer fault, even idle.

        This only stops dispatch. It sends no cancellation, dark cut, servo stop
        or hardware command. A running action remains explicitly owned until an
        outcome is reported. Create a new controller only after operator review.
        """
        if self.safety_latched:
            return
        self.latch_reason = reason[:128] if isinstance(reason, str) else "external"
        self.motion_latched = self.safety_latched = True
        self.armed, self.target_id = False, None
        self._invalidate()
        self.stats["safety_latched"] += 1

    def _watchdog(self):
        if (not self.safety_latched and self.inflight is not None
                and self._inflight_deadline_ns is not None
                and self.clock() > self._inflight_deadline_ns):
            self.stats["motion_timeout"] += 1
            self.safety_latch("motion_completion_timeout")

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
        self._last_capture_ns = -1
        return self.generation

    def set_synced(self, synced):
        if type(synced) is not bool:
            raise ValueError("sync state must be boolean")
        if not synced:
            self._invalidate()
            self.armed = False
            self.target_id = None
            self._last_capture_ns = -1
        self.synced = synced

    def lock_target(self, target_id):
        if (self.mode != "follow" or not self.armed or not self.synced
                or self.safety_latched
                or not isinstance(target_id, str) or not target_id or len(target_id) > 128):
            return False
        if self.target_id is not None and self.target_id != target_id:
            return False
        self.target_id = target_id
        return True

    def release_target(self):
        """Call on target loss or explicit release; no automatic person switching."""
        self.target_id = None
        self._last_capture_ns = -1
        self._invalidate()

    def _reject(self, reason):
        self.stats[reason] += 1
        return False

    def submit(self, command):
        """Validate and copy a command. Rejection never consumes sequence numbers."""
        if not isinstance(command, dict):
            return self._reject("invalid")
        self._watchdog()
        if self.safety_latched:
            return self._reject("latched")
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
        if c["seq"] - self.last_seq > self.max_sequence_jump:
            return self._reject("sequence_jump")
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
            if capture < self._last_capture_ns:
                return self._reject("capture_reordered")
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
                    or not _integer(c.get("duration_ns"))
                    or not 2_000_000_000 <= c["duration_ns"] <= self.max_motion_duration_ns):
                return self._reject("invalid_motion")
        elif kind == "light":
            rgb = c.get("rgb")
            if (self.mode == "hold" or not isinstance(rgb, (list, tuple)) or len(rgb) != 3
                    or not all(_number(v) and 0 <= v <= 1 for v in rgb)):
                return self._reject("invalid_light")
        else:
            return self._reject("invalid")
        if kind == "head_target":
            old_heads = [entry for entry in self._queue if entry[2]["kind"] == "head_target"]
            if old_heads:
                self._queue = [entry for entry in self._queue if entry[2]["kind"] != "head_target"]
                heapq.heapify(self._queue)
                self.stats["target_coalesced"] += len(old_heads)
            if len(self._queue) >= self.capacity:
                future_lights = [entry for entry in self._queue
                                 if entry[2]["kind"] == "light" and entry[0] > now]
                if future_lights:
                    self._queue.remove(max(future_lights, key=lambda entry: entry[:2]))
                    heapq.heapify(self._queue)
                    self.stats["future_light_evicted"] += 1
        if len(self._queue) >= self.capacity:
            return self._reject("capacity")
        self.last_seq = c["seq"]
        if kind == "head_target":
            self._last_capture_ns = c["capture_ns"]
        heapq.heappush(self._queue, (c["due_ns"], c["seq"], deepcopy(c)))
        self.stats["accepted"] += 1
        return True

    def tick(self):
        """Dispatch due commands; await busy motion without blocking due lights.

        Waiting never changes due_ns. Expired/too-late commands are discarded on
        the next tick, even if the previous action has just completed.
        """
        self._watchdog()
        if not self.synced or self.safety_latched:
            return 0
        fired = 0
        while self._queue:
            self._watchdog()
            if self.safety_latched:
                break
            now = self.clock()
            if self._queue[0][0] > now:
                break
            # Leave waiting actions IN the bounded queue. This preserves capacity
            # and coalescing if the output callback submits another observation.
            selected = None
            for entry in sorted(self._queue, key=lambda item: item[:2]):
                due, seq, c = entry
                if due > now:
                    break
                expired = (c["kind"] == "head_target" and (
                    now > c["valid_until_ns"] or now - c["capture_ns"] > self.max_target_age_ns))
                if expired or now - due > self.late_tolerance_ns:
                    self._queue.remove(entry)
                    self.stats["stale_target" if expired else "late"] += 1
                    continue
                if c["kind"] != "light" and self.inflight is not None:
                    self.stats["motion_waits"] += 1
                    continue
                selected = entry
                break
            heapq.heapify(self._queue)
            if selected is None:
                break
            self._queue.remove(selected)
            heapq.heapify(self._queue)
            due, seq, c = selected
            motion = c["kind"] != "light"
            if motion and (not self.armed or self.motion_latched):
                self.stats["motion_blocked"] += 1
                continue
            action_id = (self.session_id, c["generation"], seq)
            c["action_id"] = action_id
            if motion:
                self.inflight = action_id
                duration = c.get("duration_ns", self.head_motion_duration_ns)
                self._inflight_deadline_ns = now + duration + self.watchdog_margin_ns
            try:
                self.output(c)
            except BaseException as exc:
                self.stats["output_failed"] += 1
                self.safety_latch("output_failed")
                if not isinstance(exc, Exception):
                    raise
                break
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
        self._watchdog()
        if self.inflight is None or action_id != self.inflight:
            return self._reject("invalid_completion")
        self.inflight = None
        self._inflight_deadline_ns = None
        self.stats["motion_" + outcome] += 1
        if outcome != "succeeded":
            self.safety_latch("motion_" + outcome)
        return True
