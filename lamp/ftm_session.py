"""Sans-IO join/session state machine: the lamp joins the FTM Conductor the way a phone does (task #19).

PURE: no sockets, no threads, no wall clock, no sleeping. Every call takes ``now_ns``, a LOCAL MONOTONIC
integer nanosecond count supplied by the caller; nothing here reads a clock. ftm_udp.py is the thin socket
wrapper. This module adds NO timing logic of its own beyond send intervals: offsets, room budget L and
due times come only from ftm_clock.ClockEstimator / ftm_events.Normalizer (no L is added anywhere).

What is verified
----------------
NOTHING here was checked against the real conductor or a capture. Source of the join protocol: a report
by Tempo (the host's agent) who read the conductor's Swift source and run logs; the agents cannot read
that source. Reported:
  * Discovery is Bonjour ``_feelthemusic._udp``, UDP port 47300 (NOT implemented, see ftm_udp.py).
  * Clock: the client sends type 1 SyncReq, the conductor answers type 2 SyncResp, replies delayed 0-15 ms.
  * The conductor sends EventPacket (type 3) and BassEnvelope (type 12); ``startTs = startPts + L`` so bass
    start times ALREADY include the 300 ms room budget L. This session therefore opts in to
    ``bass_time_policy="presentation"`` (start_ts treated like masterTs, no L added). If the report is wrong
    every bass sample is 300 ms early; only the Normalizer policy can change that.
  * Hello: client to conductor telemetry, type 4, JSON ``{"t":"hi","role":"lamp","name":<str>,"v":1}``.
    The conductor answers a hello with Assign (type 5) and Control (type 13, UTF-8 JSON that includes a
    reported latency ``lat`` = 300 and a session u32).
  * Audio frames exist; their type numbers are NOT known. Every unknown type byte is counted and ignored.
  * Peers are keyed by ip:port and dropped after 6 s without traffic, so the client sends something at
    least every ``keepalive_ns`` (default 2 s, well under 3 s).
  * New Control key (extend, never rename): ``"lamp": {"mode": "follow"|"dance"|"light"|"off",
    "lights": bool, "gen": int}`` where gen bumps on every change. Lamp status goes back as type 4
    ``{"t":"lamp","state":...,"mode":...,"locked":bool,"piC":...,"sdk":"ok"|<error>,"moves":n,"refused":n}``.

INFERRED (not given, do not rely on):
  * Type 4 framing = one type byte then UTF-8 JSON, like type 13 (ftm_client's Hello already assumes it).
  * The SyncReq/SyncResp u16 is a request id echoed back.
  * Assign's first u8 is unknown and ignored; Assign's u32 is used as the session id (reported meaning).
  * The Control key that carries the session u32 is called ``"session"`` here (the report says "a session
    u32", not its key name). The exact nesting of the ``lamp`` key (top level of the Control object) is as
    stated above, unverified.
  * The keepalive telemetry ``{"t":"ka","role":..,"v":1}`` is our own invention; the conductor is only
    reported to key peers on any traffic. If it dislikes an unknown ``t``, change the keepalive, not the
    interval.
  * That the conductor answers a repeated hello: we re-send hello only while no Assign/Control has arrived
    (or after a sync timeout).

Behaviour
---------
Join: ``poll`` sends hello (every ``hello_interval_ns``) until a valid Assign/Control arrives; then
SyncReq every ``sync_interval_fast_ns`` until the estimator has an estimate and every
``sync_interval_slow_ns`` afterwards; plus a keepalive whenever nothing else was sent for
``keepalive_ns``. ``poll`` assumes the caller sends what it returns.

Requests: each SyncReq records t0 (local ns) by its u16 id. A SyncResp adds one sample (t0, t1, t2,
t3=now_ns) only for an outstanding, unexpired id; unknown, duplicate, expired or non-physical responses
are counted and never add a sample. Outstanding ids are capped (oldest evicted) and expire after
``sync_timeout_ns``.

Sessions: the first Assign/Control adopts its session id (epoch unchanged: nothing to invalidate). A
DIFFERENT session id, or any Assign/Control after a sync timeout (a request expired that no later response
answered), starts a new session: ``normalizer.new_epoch()`` exactly once, outstanding ids and the mode
are cleared, ``SessionOut`` is emitted. While timed out, hello is re-sent.

Mode (SAFE DEFAULT is 'off': the lamp does nothing until told): a ``ModeOut`` is emitted only for a valid
``lamp`` key. Unknown mode strings (``bad_mode``) never map to a default; ``lights`` must be a real bool;
``gen`` must be an int strictly greater than the last accepted gen of this session, else ``stale_mode``
and the Control's mode is ignored. ``current_mode(now_ns)`` is 'off' until a valid mode arrives and again
after ``mode_max_age_ns``. NOTE the consequence of "equal gen is stale": the age is only refreshed by a new
gen, so the conductor has to bump gen (or the lamp falls back to 'off' after ``mode_max_age_ns``).

Control JSON is hostile input: size cap, strict JSON (NaN/Infinity rejected, duplicate keys rejected,
top level must be an object, nesting depth capped). Anything malformed is counted ``bad_control``.
``on_datagram`` never raises for any bytes; a bad ``now_ns`` is a caller bug and raises.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import ftm_client
from ftm_client import (
    Assign, BassEnvelope, Control, EventPacket, ParseError, SyncReq, SyncResp,
)
from ftm_clock import INT63_MAX, ClockEstimator
from ftm_events import BassSample, Event, Normalizer

NS_PER_S = 1_000_000_000
MODES = ("follow", "dance", "light", "off")
MAX_CONTROL_BYTES = 8192
MAX_CONTROL_DEPTH = 16
MAX_STATUS_BYTES = 1200
MAX_NAME_BYTES = 64
U32_MAX = 2**32 - 1

_KNOWN_TYPES = frozenset({
    ftm_client.TYPE_SYNC_REQ, ftm_client.TYPE_SYNC_RESP, ftm_client.TYPE_EVENT, ftm_client.TYPE_HELLO,
    ftm_client.TYPE_ASSIGN, ftm_client.TYPE_BASS_ENVELOPE, ftm_client.TYPE_CONTROL,
})

_COUNTERS = (
    "datagrams", "parse_errors", "unknown_type", "unexpected_packet", "bad_control", "bad_mode",
    "stale_mode", "mode_accepted", "sync_resp_unknown", "sync_resp_duplicate", "sync_resp_expired",
    "sync_evicted", "bad_sample", "sync_samples", "sync_timeouts", "sync_id_collision", "sessions",
    "new_epochs", "events_out", "bass_out", "sent_hello", "sent_sync", "sent_keepalive", "sent_status",
)


@dataclass(frozen=True)
class EventOut:
    event: Event


@dataclass(frozen=True)
class BassOut:
    samples: list[BassSample]


@dataclass(frozen=True)
class ModeOut:
    mode: str
    lights: bool
    gen: int


@dataclass(frozen=True)
class SessionOut:
    session_id: int  # a session (re)started; consumers drop work from older epochs


Output = EventOut | BassOut | ModeOut | SessionOut


def _check_now(now_ns: object) -> int:
    if isinstance(now_ns, bool) or not isinstance(now_ns, int) or not 0 <= now_ns <= INT63_MAX:
        raise ValueError(f"now_ns must be an int in 0..2**63-1, got {now_ns!r}")
    return now_ns


def _positive(name: str, v: object) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
        raise ValueError(f"{name} must be a positive int of nanoseconds")
    return v


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _json_depth_ok(text: str, cap: int) -> bool:
    """True if brackets nest at most ``cap`` deep (strings skipped). Cheap guard before json.loads."""
    depth = 0
    in_str = esc = False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "[{":
            depth += 1
            if depth > cap:
                return False
        elif ch in "]}":
            depth -= 1
    return True


def _reject_constant(name: str):
    raise ValueError(f"non-finite JSON constant {name}")


def _reject_float(text: str) -> float:
    v = float(text)
    if v != v or v in (float("inf"), float("-inf")):
        raise ValueError("non-finite JSON number")
    return v


def _object_hook(pairs: list) -> dict:
    out: dict = {}
    for k, v in pairs:
        if not isinstance(k, str) or k in out:
            raise ValueError("non-str or duplicate key")
        out[k] = v
    return out


def parse_control_json(text: str) -> dict | None:
    """Strictly parse a Control document to a dict, or None if it is hostile/malformed. Never raises."""
    try:
        if len(text) > MAX_CONTROL_BYTES or len(text.encode("utf-8")) > MAX_CONTROL_BYTES:
            return None
        if not _json_depth_ok(text, MAX_CONTROL_DEPTH):
            return None
        doc = json.loads(text, parse_constant=_reject_constant, parse_float=_reject_float,
                         object_pairs_hook=_object_hook)
    except (ValueError, RecursionError, TypeError, MemoryError):
        return None
    return doc if isinstance(doc, dict) else None


def _default_id_source() -> Callable[[], int]:
    state = {"n": 0}

    def next_id() -> int:
        state["n"] = (state["n"] + 1) & 0xFFFF
        return state["n"]

    return next_id


class Session:
    def __init__(
        self,
        name: str,
        role: str = "lamp",
        *,
        hello_interval_ns: int = 1 * NS_PER_S,
        sync_interval_fast_ns: int = 250_000_000,
        sync_interval_slow_ns: int = 5 * NS_PER_S,
        keepalive_ns: int = 2 * NS_PER_S,  # peers are dropped after 6 s of silence (reported)
        sync_timeout_ns: int = 3 * NS_PER_S,
        mode_max_age_ns: int = 30 * NS_PER_S,
        id_source: Callable[[], int] | None = None,
        max_outstanding: int = 16,
        estimator: ClockEstimator | None = None,
    ) -> None:
        if not isinstance(name, str) or not name or len(name.encode("utf-8")) > MAX_NAME_BYTES:
            raise ValueError(f"name must be a non-empty str of at most {MAX_NAME_BYTES} UTF-8 bytes")
        if not isinstance(role, str) or not role or len(role.encode("utf-8")) > MAX_NAME_BYTES:
            raise ValueError("role must be a non-empty str")
        self.name = name
        self.role = role
        self.hello_interval_ns = _positive("hello_interval_ns", hello_interval_ns)
        self.sync_interval_fast_ns = _positive("sync_interval_fast_ns", sync_interval_fast_ns)
        self.sync_interval_slow_ns = _positive("sync_interval_slow_ns", sync_interval_slow_ns)
        self.keepalive_ns = _positive("keepalive_ns", keepalive_ns)
        self.sync_timeout_ns = _positive("sync_timeout_ns", sync_timeout_ns)
        self.mode_max_age_ns = _positive("mode_max_age_ns", mode_max_age_ns)
        self.max_outstanding = _positive("max_outstanding", max_outstanding)
        self._id_source = id_source or _default_id_source()
        self.estimator = estimator if estimator is not None else ClockEstimator()
        self._last_now = 0
        # bass_time_policy='presentation' is opted in EXPLICITLY on Tempo's report that the conductor sets
        # BassEnvelope.startTs = startPts + L, i.e. it already includes the room budget L (unverified).
        self.normalizer = Normalizer(self.estimator, bass_time_policy="presentation", clock=lambda: self._last_now)
        self._c = dict.fromkeys(_COUNTERS, 0)
        self._outstanding: OrderedDict[int, int] = OrderedDict()  # req id -> t0 local ns
        self._settled: OrderedDict[int, str] = OrderedDict()  # recent ids -> "answered" | "expired"
        self._last_resp_t0: int | None = None
        self._timed_out = False
        self._joined = False
        self._session_id: int | None = None
        self._last_hello: int | None = None
        self._last_sync: int | None = None
        self._last_send: int | None = None
        self._mode: tuple[str, bool, int, int] | None = None  # mode, lights, gen, received_ns

    # ---- introspection -------------------------------------------------------------------------------
    @property
    def epoch(self) -> int:
        return self.normalizer.epoch

    @property
    def session_id(self) -> int | None:
        return self._session_id

    @property
    def joined(self) -> bool:
        return self._joined

    @property
    def outstanding(self) -> int:
        return len(self._outstanding)

    def current_mode(self, now_ns: int) -> str:
        """'off' until a valid mode arrived, and again once it is older than mode_max_age_ns."""
        _check_now(now_ns)
        if self._mode is None or now_ns - self._mode[3] > self.mode_max_age_ns:
            return "off"
        return self._mode[0]

    def current_lights(self, now_ns: int) -> bool:
        """False under exactly the same conditions as current_mode is 'off' by default."""
        _check_now(now_ns)
        if self._mode is None or now_ns - self._mode[3] > self.mode_max_age_ns:
            return False
        return self._mode[1]

    def stats(self) -> dict:
        n = self.normalizer
        out = dict(self._c)
        out.update(
            epoch=n.epoch, outstanding=len(self._outstanding), not_synced=n.not_synced,
            unknown_kind=n.unknown_kind, bad_time=n.bad_time, bass_policy_unset=n.bass_policy_unset,
            normalized=n.normalized, bass_normalized=n.bass_normalized,
        )
        return out

    # ---- sending -------------------------------------------------------------------------------------
    def _telemetry(self, doc: dict) -> bytes:
        body = json.dumps(doc, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
        return bytes([ftm_client.TYPE_HELLO]) + body

    def hello_datagram(self) -> bytes:
        return self._telemetry({"t": "hi", "role": self.role, "name": self.name, "v": 1})

    def status_datagram(self, status: dict, now_ns: int) -> bytes:
        """Lamp telemetry (type 4, ``t`` = "lamp"). The caller sends it; it does not count as keepalive."""
        _check_now(now_ns)
        if not isinstance(status, dict):
            raise ValueError("status must be a dict")
        doc = {"t": "lamp"}
        doc.update({k: v for k, v in status.items() if k != "t"})
        data = self._telemetry(doc)
        if len(data) > MAX_STATUS_BYTES:
            raise ValueError(f"status datagram is {len(data)} bytes, cap {MAX_STATUS_BYTES}")
        self._c["sent_status"] += 1
        return data

    def _new_id(self) -> int | None:
        for _ in range(4):
            rid = self._id_source()
            if isinstance(rid, bool) or not isinstance(rid, int) or not 0 <= rid <= 0xFFFF:
                raise ValueError(f"id_source must return a u16 int, got {rid!r}")
            if rid not in self._outstanding:
                return rid
        self._c["sync_id_collision"] += 1
        return None

    def _expire(self, now_ns: int) -> None:
        for rid in [r for r, t0 in self._outstanding.items() if now_ns - t0 > self.sync_timeout_ns]:
            t0 = self._outstanding.pop(rid)
            self._remember(rid, "expired")
            if self._last_resp_t0 is None or t0 > self._last_resp_t0:
                if not self._timed_out:
                    self._c["sync_timeouts"] += 1
                self._timed_out = True

    def _remember(self, rid: int, why: str) -> None:
        self._settled.pop(rid, None)
        self._settled[rid] = why
        while len(self._settled) > 64:
            self._settled.popitem(last=False)

    def poll(self, now_ns: int) -> list[bytes]:
        """Datagrams to send now. The caller is assumed to send them all."""
        now_ns = _check_now(now_ns)
        self._last_now = now_ns
        self._expire(now_ns)
        out: list[bytes] = []
        if (not self._joined or self._timed_out) and (
                self._last_hello is None or now_ns - self._last_hello >= self.hello_interval_ns):
            out.append(self.hello_datagram())
            self._last_hello = now_ns
            self._c["sent_hello"] += 1
        if self._joined:
            interval = (self.sync_interval_slow_ns if self.estimator.estimate(now_ns) is not None
                        else self.sync_interval_fast_ns)
            if self._last_sync is None or now_ns - self._last_sync >= interval:
                rid = self._new_id()
                if rid is not None:
                    while len(self._outstanding) >= self.max_outstanding:
                        self._outstanding.popitem(last=False)
                        self._c["sync_evicted"] += 1
                    self._outstanding[rid] = now_ns
                    out.append(SyncReq(rid).encode())
                    self._c["sent_sync"] += 1
                self._last_sync = now_ns
        if not out and (self._last_send is None or now_ns - self._last_send >= self.keepalive_ns):
            out.append(self._telemetry({"t": "ka", "role": self.role, "v": 1}))
            self._c["sent_keepalive"] += 1
        if out:
            self._last_send = now_ns
        return out

    # ---- receiving -----------------------------------------------------------------------------------
    def on_datagram(self, data: bytes, now_ns: int) -> list[Output]:
        now_ns = _check_now(now_ns)
        self._last_now = now_ns
        self._c["datagrams"] += 1
        try:
            return self._handle(data, now_ns)
        except Exception:  # defence in depth: hostile bytes must never take the lamp loop down
            self._c["parse_errors"] += 1
            return []

    def _handle(self, data: bytes, now_ns: int) -> list[Output]:
        if isinstance(data, bytes) and data and data[0] not in _KNOWN_TYPES:
            self._c["unknown_type"] += 1
            return []
        if isinstance(data, bytes) and data[:1] == bytes([ftm_client.TYPE_CONTROL]) and len(data) - 1 > MAX_CONTROL_BYTES:
            self._c["bad_control"] += 1
            return []
        try:
            pkt = ftm_client.decode(data)
        except ParseError:
            self._c["parse_errors"] += 1
            return []
        if isinstance(pkt, SyncResp):
            self._on_sync_resp(pkt, now_ns)
            return []
        if isinstance(pkt, EventPacket):
            ev = self.normalizer.normalize_event(pkt, now_ns)
            if ev is None:
                return []
            self._c["events_out"] += 1
            return [EventOut(ev)]
        if isinstance(pkt, BassEnvelope):
            samples = self.normalizer.normalize_bass(pkt, now_ns)
            if not samples:
                return []
            self._c["bass_out"] += 1
            return [BassOut(samples)]
        if isinstance(pkt, Assign):
            return self._on_session(pkt.session, now_ns)
        if isinstance(pkt, Control):
            return self._on_control(pkt, now_ns)
        # SyncReq / Hello are client-to-conductor; the lamp is a client and never answers them.
        self._c["unexpected_packet"] += 1
        return []

    def _on_sync_resp(self, pkt: SyncResp, now_ns: int) -> None:
        t0 = self._outstanding.pop(pkt.req_id, None)
        if t0 is None:
            why = self._settled.get(pkt.req_id)
            key = {"answered": "sync_resp_duplicate", "expired": "sync_resp_expired"}.get(why, "sync_resp_unknown")
            self._c[key] += 1
            return
        if now_ns - t0 > self.sync_timeout_ns:
            self._remember(pkt.req_id, "expired")
            self._c["sync_resp_expired"] += 1
            return
        self._remember(pkt.req_id, "answered")
        try:
            self.estimator.add_sample(t0, pkt.t1, pkt.t2, now_ns)
        except ValueError:
            self._c["bad_sample"] += 1
            return
        self._c["sync_samples"] += 1
        self._last_resp_t0 = t0 if self._last_resp_t0 is None else max(self._last_resp_t0, t0)
        self._timed_out = False

    def _on_session(self, sid: int, now_ns: int) -> list[Output]:
        restart = False
        if self._session_id is None:
            restart = True  # first session: adopt, nothing to invalidate
        elif sid != self._session_id or self._timed_out:
            restart = True
            self.normalizer.new_epoch()
            self._c["new_epochs"] += 1
            self._outstanding.clear()
            self._settled.clear()
            self._mode = None
            self._last_resp_t0 = None
            self._timed_out = False
            self._last_sync = None
        self._joined = True
        self._session_id = sid
        if not restart:
            return []
        self._c["sessions"] += 1
        return [SessionOut(sid)]

    def _on_control(self, pkt: Control, now_ns: int) -> list[Output]:
        doc = parse_control_json(pkt.text)
        if doc is None:
            self._c["bad_control"] += 1
            return []
        out: list[Output] = []
        if "session" in doc:
            sid = doc["session"]
            if not _is_int(sid) or not 0 <= sid <= U32_MAX:
                self._c["bad_control"] += 1
                return []
            out += self._on_session(sid, now_ns)
        else:
            self._joined = True  # a valid Control ends the hello phase even without a session id
        if "lamp" in doc:
            mode_out = self._on_lamp(doc["lamp"], now_ns)
            if mode_out is not None:
                out.append(mode_out)
        return out

    def _on_lamp(self, lamp: object, now_ns: int) -> ModeOut | None:
        if not isinstance(lamp, dict):
            self._c["bad_mode"] += 1
            return None
        mode, lights, gen = lamp.get("mode"), lamp.get("lights"), lamp.get("gen")
        if not isinstance(mode, str) or mode not in MODES:
            self._c["bad_mode"] += 1  # unknown modes are ignored, never mapped to a default
            return None
        if not isinstance(lights, bool) or not _is_int(gen) or not 0 <= gen <= INT63_MAX:
            self._c["bad_mode"] += 1
            return None
        if self._mode is not None and gen <= self._mode[2]:
            self._c["stale_mode"] += 1
            return None
        self._mode = (mode, lights, gen, now_ns)
        self._c["mode_accepted"] += 1
        return ModeOut(mode, lights, gen)
