"""Single-owner, bounded SDK dispatch, written 2026-09-19.

Deadlines are LOCAL monotonic ns, already including room latency/output trim.
Callbacks block on one SDK action and its terminal polls, outside the state lock.
Each callback MUST issue exactly one action POST, with no internal POST retries;
sessions must already exist. This makes the shared admission budget meaningful.
Use independent SDK HTTP clients for the two lanes. No hardware API is imported.

Admission linearizes at the worker's final state/rate/deadline check under the
lock. Off/close cannot cancel an action already admitted, even if its callback
has not yet entered HTTP. They clear pending work and remain responsive. Never
send raw motor commands, system.stop, or cancellation from these callbacks.

The application applies its shared FlashLimiter before submission. This module
additionally caps emitted RGB channels at .3, rejects red-dominant output, and
spaces ALL local admissions by at least 500 ms. HTTP arrival and visible fade
timing can differ; this is not a measured transition-rate guarantee.
Vendor fades/other light owners still need physical verification; this is not
an LED safety certification. Timeouts latch admission, not hardware motion.
"""
from collections import Counter, deque
from collections.abc import Mapping
from dataclasses import dataclass, replace
import math
import threading
import time
from types import MappingProxyType

JOINTS = ("base_yaw", "base_pitch", "elbow_pitch", "wrist_roll", "wrist_pitch")


@dataclass(frozen=True)
class ClipIntent:
    clip_id: str
    start_due_ns: int
    duration_ns: int
    epoch: object
    gen: int


@dataclass(frozen=True)
class MoveIntent:
    positions: Mapping
    start_due_ns: int
    duration_ns: int
    epoch: object
    gen: int

    def __post_init__(self):
        if isinstance(self.positions, Mapping):
            object.__setattr__(self, "positions", MappingProxyType(dict(self.positions)))


@dataclass(frozen=True)
class LightIntent:
    rgb: tuple
    due_ns: int
    epoch: object
    gen: int

    def __post_init__(self):
        if isinstance(self.rgb, (list, tuple)):
            object.__setattr__(self, "rgb", tuple(self.rgb))


def _uint(value):
    return type(value) is int and 0 <= value <= 2**63 - 1


def _epoch(value):
    return _uint(value) or (type(value) is str and 0 < len(value) <= 128)


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


class Dispatcher:
    """Two daemon workers: one motion owner and one independently polled light.

    One latest pending intent per lane; no per-envelope backlog. Call tick()
    regularly, even while unsynchronized, for watchdogs and worker wakeups.
    The application owns a bounded presentation timeline and releases its newest
    ELIGIBLE light sample here (due <= now). Continuously replacing one future
    light with a later future sample would postpone it forever; do not do that.
    Explicit tick time must share the injected clock domain; workers re-read
    that clock before admission. No automatic fault reset or hidden retry.
    clip_allowlist maps locally verified clip IDs to maximum duration in ns.
    These bounds and watchdog margins are design policies, not measurements.
    """

    def __init__(self, run_motion, run_light, *, clock=time.monotonic_ns,
                 clip_allowlist=None, late_tolerance_ns=80_000_000,
                 max_future_ns=30_000_000_000, timeout_margin_ns=1_000_000_000,
                 light_timeout_ns=8_000_000_000, min_post_interval_ns=500_000_000,
                 max_posts_per_minute=120):
        if not all(callable(x) for x in (run_motion, run_light, clock)):
            raise ValueError("callbacks and clock must be callable")
        bounds = ((late_tolerance_ns, 0, 1_000_000_000), (max_future_ns, 0, 30_000_000_000),
                  (timeout_margin_ns, 0, 5_000_000_000), (light_timeout_ns, 1, 30_000_000_000),
                  (min_post_interval_ns, 500_000_000, 60_000_000_000), (max_posts_per_minute, 1, 120))
        if any(not _uint(v) or not lo <= v <= hi for v, lo, hi in bounds):
            raise ValueError("dispatch policy outside supported bounds")
        clips = {} if clip_allowlist is None else dict(clip_allowlist)
        if any(type(k) is not str or not 0 < len(k) <= 128 or not _uint(v)
               or not 1 <= v <= 30_000_000_000 for k, v in clips.items()):
            raise ValueError("clip allowlist needs IDs and bounded verified durations in ns")
        self._callbacks = {"motion": run_motion, "light": run_light}
        self._clock, self._clips = clock, MappingProxyType(clips)
        self._late, self._horizon, self._margin = late_tolerance_ns, max_future_ns, timeout_margin_ns
        self._light_timeout, self._spacing, self._budget = light_timeout_ns, min_post_interval_ns, max_posts_per_minute
        self._condition = threading.Condition()
        self._config = None
        self._valid_until = None
        self._revision = 0
        self._closed = self._latched = False
        self._reason = None
        self._pending = {"motion": None, "light": None}
        self._active = {"motion": None, "light": None}
        self._ready = {"motion": False, "light": False}
        self._posts, self._stats = deque(), Counter()
        self._workers = []
        for lane in self._pending:
            worker = threading.Thread(target=self._worker, args=(lane,), daemon=True, name="ftm-" + lane)
            self._workers.append(worker)
            worker.start()

    def _clear_pending(self):
        self._stats["invalidated"] += sum(item is not None for item in self._pending.values())
        for lane in self._pending:
            self._pending[lane], self._ready[lane] = None, False

    def _latch(self, reason):
        if not self._latched:
            self._latched, self._reason = True, reason
            self._clear_pending()
            self._stats["latched"] += 1

    def latch(self, reason="external_fault"):
        """Unconditional local admission latch; never cancels an active action."""
        with self._condition:
            self._latch(reason[:128] if type(reason) is str else "external_fault")

    def configure(self, epoch, gen, mode, lights, synced, armed=False, valid_until_ns=None):
        """Return False for invalid/stale state. An identical config is a no-op.

        A new epoch permits generation reset. Same epoch generations never move
        backwards. Changed state invalidates pending intents but retains actual
        active callbacks. A latched dispatcher cannot rearm through configure.
        Network applications supply a conservative valid_until_ns readiness
        lease; refreshing the lease does not invalidate queued work.
        """
        if valid_until_ns is not None and not _uint(valid_until_ns):
            return False
        if (not _epoch(epoch) or not _uint(gen) or mode not in ("off", "light", "follow", "dance")
                or any(type(v) is not bool for v in (lights, synced, armed))):
            return False
        config = (epoch, gen, mode, lights, synced, armed)
        with self._condition:
            if self._closed or self._latched:
                return False
            if self._config is not None and epoch == self._config[0] and gen < self._config[1]:
                return False
            if config != self._config:
                self._clear_pending()
                self._revision += 1
                self._config = config
            self._valid_until = valid_until_ns
            return True

    def _allowed(self, intent):
        if self._closed or self._latched or self._config is None:
            return False
        epoch, gen, mode, lights, synced, armed = self._config
        if not synced or not armed or mode == "off" or intent.epoch != epoch or intent.gen != gen:
            return False
        if isinstance(intent, LightIntent):
            return lights
        return mode == ("dance" if isinstance(intent, ClipIntent) else "follow")

    def _submit(self, intent, lane):
        due = intent.due_ns if lane == "light" else intent.start_due_ns
        if not _epoch(intent.epoch) or not _uint(intent.gen) or not _uint(due):
            return False
        with self._condition:
            now = self._clock()
            self._watchdog(now)
            if not self._allowed(intent):
                return False
            if now - due > self._late or due - now > self._horizon:
                self._stats["deadline_rejected"] += 1
                return False
            if self._pending[lane] is not None:
                self._stats[lane + "_coalesced"] += 1
            self._pending[lane] = (intent, self._revision)
            self._stats[lane + "_accepted"] += 1
            return True

    def submit_clip(self, intent):
        if (type(intent) is not ClipIntent or type(intent.clip_id) is not str
                or intent.clip_id not in self._clips or not _uint(intent.duration_ns)
                or not 1 <= intent.duration_ns <= self._clips[intent.clip_id]):
            return False
        return self._submit(intent, "motion")

    def submit_move(self, intent):
        if (type(intent) is not MoveIntent or not isinstance(intent.positions, Mapping)
                or set(intent.positions) != set(JOINTS)
                or not all(_finite(v) and -100 <= v <= 100 for v in intent.positions.values())
                or not _uint(intent.duration_ns) or not 2_000_000_000 <= intent.duration_ns <= 30_000_000_000):
            return False
        return self._submit(intent, "motion")

    def submit_light(self, intent):
        if (type(intent) is not LightIntent or not isinstance(intent.rgb, tuple) or len(intent.rgb) != 3
                or not all(_finite(v) and 0 <= v <= 1 for v in intent.rgb)):
            return False
        rgb = tuple(min(float(v), .3) for v in intent.rgb)
        if rgb[0] > rgb[1] + rgb[2]:
            return False
        return self._submit(replace(intent, rgb=rgb), "light")

    def _watchdog(self, now):
        for lane, active in self._active.items():
            if active is not None and now > active["deadline_ns"] and active["status"] == "running":
                active["status"] = "timed_out"
                self._stats[lane + "_timeout"] += 1
                self._latch(lane + "_timeout")

    def tick(self, now_ns=None):
        now = self._clock() if now_ns is None else now_ns
        if not _uint(now):
            raise ValueError("tick requires local monotonic integer ns")
        with self._condition:
            self._watchdog(now)
            if not self._closed and not self._latched:
                for lane in self._pending:
                    if self._pending[lane] is not None:
                        self._ready[lane] = True
                self._condition.notify_all()

    def _admit(self, lane):
        entry = self._pending[lane]
        if entry is None or self._active[lane] is not None:
            return None
        intent, revision = entry
        now = self._clock()                  # admission's authoritative fresh clock
        self._watchdog(now)
        if self._valid_until is not None and now > self._valid_until:
            self._pending[lane] = None
            self._stats["readiness_expired"] += 1
            return None
        if revision != self._revision or not self._allowed(intent):
            self._pending[lane] = None
            return None
        due = intent.due_ns if lane == "light" else intent.start_due_ns
        if now < due:
            self._stats[lane + "_early_wait"] += 1
            return None
        if now - due > self._late:
            self._pending[lane] = None
            self._stats[lane + "_late"] += 1
            return None
        while self._posts and self._posts[0] <= now - 60_000_000_000:
            self._posts.popleft()
        if len(self._posts) >= self._budget or (self._posts and now - self._posts[-1] < self._spacing):
            self._stats["rate_deferred"] += 1
            return None
        self._posts.append(now)
        self._pending[lane] = None
        duration = self._light_timeout if lane == "light" else intent.duration_ns + self._margin
        self._active[lane] = {"intent": intent, "started_ns": now, "deadline_ns": now + duration, "status": "running"}
        self._stats[lane + "_started"] += 1
        return intent

    @staticmethod
    def _success(intent, record):
        if not isinstance(record, dict) or record.get("state") != "succeeded":
            return False
        if isinstance(intent, MoveIntent):
            result = record.get("result")
            return isinstance(result, dict) and result.get("completed") is True and result.get("reached") is True
        return True

    def _worker(self, lane):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._ready[lane])
                if self._closed:
                    return
                self._ready[lane] = False
                intent = self._admit(lane)
            if intent is None:
                continue
            try:
                record = self._callbacks[lane](intent)   # NEVER hold state lock across HTTP/poll
            except BaseException:
                with self._condition:
                    self._active[lane]["status"] = "uncertain"
                    self._latch(lane + "_callback_error")
                continue
            with self._condition:
                self._watchdog(self._clock())
                if self._success(intent, record):
                    self._active[lane] = None
                    self._stats[lane + "_completed"] += 1
                else:
                    if isinstance(record, dict) and record.get("state") in ("failed", "rejected", "canceled"):
                        self._active[lane] = None    # a known terminal failure still latches every lane
                        self._stats[lane + "_failed"] += 1
                    else:
                        self._active[lane]["status"] = "uncertain"
                    self._latch(lane + "_terminal_failure")

    def snapshot(self):
        with self._condition:
            return {"closed": self._closed, "latched": self._latched, "reason": self._reason,
                    "config": self._config, "pending_motion": self._pending["motion"] is not None,
                    "pending_light": self._pending["light"] is not None,
                    "motion_inflight": self._active["motion"] is not None,
                    "light_inflight": self._active["light"] is not None,
                    "motion_status": None if self._active["motion"] is None else self._active["motion"]["status"],
                    "light_status": None if self._active["light"] is None else self._active["light"]["status"],
                    "stats": dict(self._stats)}

    def close(self):
        """Nonblocking local close; admitted callbacks finish on their daemon worker."""
        with self._condition:
            self._closed = True
            self._clear_pending()
            self._condition.notify_all()
