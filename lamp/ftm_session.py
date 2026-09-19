"""Sans-IO join/session state machine: the lamp joins the FTM Conductor the way a phone does (task #19).

PURE: no sockets, no threads, no wall clock, no sleeping. Every call takes ``now_ns``, a LOCAL MONOTONIC
integer nanosecond count supplied by the caller; nothing here reads a clock. ftm_udp.py is the thin socket
wrapper. This module adds NO timing logic of its own beyond send intervals: offsets, room budget L and
due times come only from ftm_clock.ClockEstimator / ftm_events.Normalizer (no L is added anywhere).

What is verified
----------------
NOTHING here was checked against the real conductor or a capture. We claim no interop. The join
protocol below comes from docs/ftm-protocol.md (Tempo, PR #19), which its author says was written from
the conductor's Swift source and checked with a probe that joins as a lamp; that is a reported source,
not a check made by this code or its tests (the tests drive a FakeConductor written from the same
document).

From docs/ftm-protocol.md (each item is document-reported, not verified here):
  * Discovery is Bonjour ``_feelthemusic._udp``, UDP port 47300 (NOT implemented, see ftm_udp.py).
  * Hello: the first datagram, type 4 = type byte + UTF-8 JSON, exactly
    ``{"t":"hi","role":"lamp","name":<name>,"v":1,"audio":false}``. The conductor creates the peer on the
    first datagram, answers the hello with Assign (type 5: u8 ``index``, u32 ``session``) and Control
    (type 13: UTF-8 JSON), and re-sends Control whenever a setting changes and after every hello.
  * Sync cadence: SyncReq (type 1, u16 ``seq``) every 50 ms for the first 40 requests, then every 250 ms
    plus 0..20 ms jitter, forever. The SyncReq loop doubles as the keepalive: a peer silent for 6 s is
    dropped, so there is NO other keepalive datagram. If nothing at all has been heard from the conductor
    for 2 s the hello is re-sent once a second (a restarted conductor picks it up).
  * Clock: SyncResp (type 2) carries the echoed seq, t1 (Mac receive ns) and t2 (Mac send ns).
  * EventPacket (type 3) and BassEnvelope (type 12): ``masterTs`` and ``startTs`` already include the
    room budget L, so this session opts in to ``bass_time_policy="presentation"`` (no L added; if the
    document is wrong every time is 300 ms early). Critical events are sent 3 times 12 ms apart and
    BassEnvelope twice 8 ms apart: de-duplicate by ``seq`` (done here, separate windows).
  * EventPacket ``target`` is 255 for everyone, otherwise one phone's index; the lamp's own index is
    Assign's ``index``.
  * Types 6..10 are audio: ignored, counted ``audio_ignored``, never parsed. A lamp is sent no audio.
  * Control's ``lamp`` key is at the top level of the Control object:
    ``"lamp": {"mode": "follow"|"dance"|"light"|"off", "lights": bool, "gen": int}``; ``gen`` bumps on
    every dashboard change (drop work queued for an older gen). Unknown keys are ignored. ``session`` is
    also a Control key.
  * Lamp status, sent about once a second as type 4 (see ``set_status``):
    ``{"t":"lamp","state":<searching|locked|dancing|light|idle|error>,"mode":<str>,"locked":<bool>,
    "piC":<number>,"sdk":<"ok"|short error>,"moves":<int>,"refused":<int>}``.

Still NOT checked against a real conductor or capture, and ours rather than the document's:
  * The safe-default rule for a silent conductor (``conductor_lost_ns``, default 4 s, under the 6 s peer
    drop; see below). The document says nothing about what a lamp should do when the conductor vanishes.
  * The exact reading of "every 50 ms for the first 40": here the gap after each of the first 40 requests
    is 50 ms (the 41st request is 50 ms after the 40th), then 250 ms + jitter after each later request.
  * Whether the conductor answers a repeated hello (the document says it re-sends Control after every
    hello, which is what we rely on to refresh the mode).
  * That an Assign/Control ``session`` disagreement is anything other than a conductor restart; we cannot
    tell them apart, so the newest datagram wins and ``session_mismatch`` counts an Assign/Control
    disagreement (a restart announced by the other message type counts too).
  * Whether ``gen`` may restart from a lower number in a new session (we assume yes, keyed by session).

Behaviour
---------
Join: ``poll`` sends the hello (every ``hello_interval_ns``, default 1 s) until a valid Assign/Control
arrives; then the SyncReq loop above, jitter drawn from the injected ``jitter_source`` (a callable
returning integer ns in 0..20 ms; the default uses ``random``); and the hello once a second whenever
nothing has been received for ``hello_silence_ns`` (default 2 s). ``poll`` assumes the caller sends
what it returns. Once ``set_status`` has been given a status dict, ``poll`` also emits it every 1 s.

Requests: each SyncReq records t0 (local ns) by its u16 seq. A SyncResp adds one sample (t0, t1, t2,
t3=now_ns) only for an outstanding, unexpired seq; unknown, duplicate, expired or non-physical responses
are counted and never add a sample. Outstanding seqs are capped (oldest evicted) and expire after
``sync_timeout_ns``.

Sessions: the first Assign/Control adopts its session id (epoch unchanged: nothing to invalidate). A
DIFFERENT session id (from either message), or any Assign/Control after a sync timeout (a request
expired that no later response answered), starts a new session: ``normalizer.new_epoch()`` exactly once,
outstanding seqs, the mode, the lamp's index and both de-duplication windows are cleared, the fast sync
burst restarts, and ``SessionOut`` is emitted. gen and modes are keyed by session.

Filtering: an EventPacket is accepted only if ``target`` is 255 or equals the index from the latest
Assign; otherwise it is counted ``not_for_me``. Before any Assign only 255 is accepted. BassEnvelope has no
target. A repeat of a seq still in the (bounded, exact-u32-keyed, age-limited) window is counted
``dup_event`` / ``dup_bass`` and produces no output; a packet that produced no output (not synced, unknown
kind) is not remembered, so a later copy can still be used.

Mode (SAFE DEFAULT is 'off': the lamp does nothing until told): a ``ModeOut`` is emitted only for a valid
``lamp`` key. Unknown mode strings (``bad_mode``) never map to a default; ``lights`` must be a real bool;
``gen`` must be an int. A LOWER gen within a session is stale (``stale_mode``, ignored); an EQUAL gen is a
harmless refresh (``mode_refresh``: no output, nothing changes). The mode does NOT expire by age: the
conductor only re-sends Control on a change or after a hello, so a steady ``dance`` keeps dancing.
Instead ``current_mode(now_ns)`` is 'off' when nothing at all has been received from the conductor for
more than ``conductor_lost_ns`` (default 4 s), and stays 'off' after a loss until a fresh Assign, or a
Control with a valid ``lamp`` key (new gen, equal-gen refresh or new session), arrives; then it returns to
the last valid mode of the session. Before any valid ``lamp`` key it is 'off'.

Control JSON is hostile input: size cap, strict JSON (NaN/Infinity rejected, duplicate keys rejected,
top level must be an object, nesting depth capped). Anything malformed is counted ``bad_control``.
``on_datagram`` never raises for any bytes; a bad ``now_ns`` is a caller bug and raises.
"""
from __future__ import annotations

import json
import math
import random
import unicodedata
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
NS_PER_MS = 1_000_000
MODES = ("follow", "dance", "light", "off")
MAX_CONTROL_BYTES = 8192
MAX_CONTROL_DEPTH = 16
MAX_NAME_CHARS = 64
U32_MAX = 2**32 - 1

# Sync cadence from docs/ftm-protocol.md: 50 ms for the first 40 requests, then 250 ms + 0..20 ms jitter.
FAST_SYNC_COUNT = 40
FAST_SYNC_NS = 50 * NS_PER_MS
SLOW_SYNC_NS = 250 * NS_PER_MS
JITTER_MAX_NS = 20 * NS_PER_MS
STATUS_INTERVAL_NS = 1 * NS_PER_S
# Recent-seq windows. The conductor repeats an event 3x 12 ms apart and a bass envelope 2x 8 ms apart, so a
# repeat always lands within a few tens of ms; the bounds are generous and keep memory constant.
SEQ_WINDOW_MAX = 128
SEQ_WINDOW_AGE_NS = 1 * NS_PER_S

STATUS_STATES = ("searching", "locked", "dancing", "light", "idle", "error")
STATUS_KEYS = ("state", "mode", "locked", "piC", "sdk", "moves", "refused")
STATUS_MODE_MAX = 32
STATUS_SDK_MAX = 64

_KNOWN_TYPES = frozenset({
    ftm_client.TYPE_SYNC_REQ, ftm_client.TYPE_SYNC_RESP, ftm_client.TYPE_EVENT, ftm_client.TYPE_HELLO,
    ftm_client.TYPE_ASSIGN, ftm_client.TYPE_BASS_ENVELOPE, ftm_client.TYPE_CONTROL,
})

_COUNTERS = (
    "datagrams", "parse_errors", "unknown_type", "audio_ignored", "unexpected_packet", "bad_control",
    "bad_mode", "stale_mode", "mode_accepted", "mode_refresh", "conductor_lost", "session_mismatch",
    "sync_resp_unknown", "sync_resp_duplicate", "sync_resp_expired", "sync_evicted", "bad_sample",
    "sync_samples", "sync_timeouts", "sync_id_collision", "sessions", "new_epochs", "events_out",
    "bass_out", "dup_event", "dup_bass", "not_for_me", "sent_hello", "sent_sync", "sent_status",
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


def _clean_text(what: str, v: object, max_chars: int) -> str:
    """A str of 1..max_chars characters with no control or surrogate characters, else ValueError."""
    if not isinstance(v, str) or not 1 <= len(v) <= max_chars:
        raise ValueError(f"{what} must be a str of 1..{max_chars} characters")
    if any(unicodedata.category(ch) in ("Cc", "Cs") for ch in v):
        raise ValueError(f"{what} must not contain control characters")
    return v


def _validate_status(status: object) -> dict:
    """Check a lamp status dict against the shape in docs/ftm-protocol.md; return it in wire key order."""
    if not isinstance(status, dict) or set(status) != set(STATUS_KEYS):
        raise ValueError(f"status must be a dict with exactly the keys {STATUS_KEYS}")
    if not isinstance(status["state"], str) or status["state"] not in STATUS_STATES:
        raise ValueError(f"state must be one of {STATUS_STATES}")
    if not isinstance(status["locked"], bool):
        raise ValueError("locked must be a bool")
    pic = status["piC"]
    if isinstance(pic, bool) or not isinstance(pic, (int, float)) or not math.isfinite(pic):
        raise ValueError("piC must be a finite number")
    for k in ("moves", "refused"):
        v = status[k]
        if not _is_int(v) or not 0 <= v <= U32_MAX:
            raise ValueError(f"{k} must be an int in 0..2**32-1")
    return {
        "state": status["state"],
        "mode": _clean_text("mode", status["mode"], STATUS_MODE_MAX),
        "locked": status["locked"],
        "piC": pic,
        "sdk": _clean_text("sdk", status["sdk"], STATUS_SDK_MAX),
        "moves": status["moves"],
        "refused": status["refused"],
    }


class _SeqWindow:
    """Bounded memory of recently seen u32 seqs, keyed by exact value so u32 wrap needs no special case
    (2**32-1 then 0 are two different packets). Entries fall out by count and by age."""

    def __init__(self, capacity: int, max_age_ns: int) -> None:
        self._capacity = capacity
        self._max_age_ns = max_age_ns
        self._seen: OrderedDict[int, int] = OrderedDict()

    def __len__(self) -> int:
        return len(self._seen)

    def _purge(self, now_ns: int) -> None:
        while self._seen:
            oldest = next(iter(self._seen.values()))
            if now_ns - oldest <= self._max_age_ns:
                break
            self._seen.popitem(last=False)

    def seen(self, seq: int, now_ns: int) -> bool:
        self._purge(now_ns)
        return seq in self._seen

    def add(self, seq: int, now_ns: int) -> None:
        self._purge(now_ns)
        self._seen.pop(seq, None)
        self._seen[seq] = now_ns
        while len(self._seen) > self._capacity:
            self._seen.popitem(last=False)

    def clear(self) -> None:
        self._seen.clear()


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
        *,
        hello_interval_ns: int = 1 * NS_PER_S,
        hello_silence_ns: int = 2 * NS_PER_S,
        conductor_lost_ns: int = 4 * NS_PER_S,  # under the conductor's 6 s peer drop; see the module docstring
        content_silence_ns: int = 5 * NS_PER_S,  # joined, SyncResp arrives, but no Assign/Control/event/bass: re-hello
        sync_timeout_ns: int = 3 * NS_PER_S,
        jitter_source: Callable[[], int] | None = None,
        id_source: Callable[[], int] | None = None,
        max_outstanding: int = 16,
        estimator: ClockEstimator | None = None,
    ) -> None:
        self.name = _clean_text("name", name, MAX_NAME_CHARS)
        self.hello_interval_ns = _positive("hello_interval_ns", hello_interval_ns)
        self.hello_silence_ns = _positive("hello_silence_ns", hello_silence_ns)
        self.conductor_lost_ns = _positive("conductor_lost_ns", conductor_lost_ns)
        self.content_silence_ns = _positive("content_silence_ns", content_silence_ns)
        self.sync_timeout_ns = _positive("sync_timeout_ns", sync_timeout_ns)
        self.max_outstanding = _positive("max_outstanding", max_outstanding)
        if jitter_source is not None and not callable(jitter_source):
            raise ValueError("jitter_source must be a callable returning ns in 0..20 ms")
        self._jitter_source = jitter_source or (lambda: random.randint(0, JITTER_MAX_NS))
        self._id_source = id_source or _default_id_source()
        self.estimator = estimator if estimator is not None else ClockEstimator()
        self._last_now = 0
        # bass_time_policy='presentation' is opted in EXPLICITLY on docs/ftm-protocol.md saying the conductor
        # sets BassEnvelope.startTs = startPts + L, i.e. it already includes the room budget L (unverified).
        self.normalizer = Normalizer(self.estimator, bass_time_policy="presentation", clock=lambda: self._last_now)
        self._c = dict.fromkeys(_COUNTERS, 0)
        self._outstanding: OrderedDict[int, int] = OrderedDict()  # seq -> t0 local ns
        self._settled: OrderedDict[int, str] = OrderedDict()  # recent seqs -> "answered" | "expired"
        self._last_resp_t0: int | None = None
        self._timed_out = False
        self._joined = False
        self._session_id: int | None = None
        self._session_src: str | None = None  # "assign" | "control": source of the last session id seen
        self._index: int | None = None  # Assign.index of the current session
        self._last_hello: int | None = None
        self._next_sync: int | None = None
        self._sync_sent = 0  # SyncReqs sent in the current session (drives the 50 ms burst)
        self._last_rx: int | None = None  # local ns of the last datagram received from the conductor
        self._last_content: int | None = None  # ... of the last non-SyncResp one (Assign/Control/event/bass)
        self._suspended = False  # conductor was lost: mode stays 'off' until a fresh Assign/Control
        self._mode: tuple[str, bool, int] | None = None  # mode, lights, gen (of the current session)
        self._status: dict | None = None
        self._last_status: int | None = None
        self._event_seen = _SeqWindow(SEQ_WINDOW_MAX, SEQ_WINDOW_AGE_NS)
        self._bass_seen = _SeqWindow(SEQ_WINDOW_MAX, SEQ_WINDOW_AGE_NS)

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

    def _mode_live(self, now_ns: int) -> bool:
        return (self._mode is not None and not self._suspended and self._last_rx is not None
                and now_ns - self._last_rx <= self.conductor_lost_ns)

    def current_mode(self, now_ns: int) -> str:
        """The last valid mode, or 'off': before any valid ``lamp`` key, while the conductor has been silent
        for more than ``conductor_lost_ns``, and after such a loss until a fresh Assign/Control arrives.
        A steady mode never expires by age."""
        _check_now(now_ns)
        return self._mode[0] if self._mode_live(now_ns) else "off"

    def current_lights(self, now_ns: int) -> bool:
        """False under exactly the same conditions as current_mode is 'off' by default."""
        _check_now(now_ns)
        return self._mode[1] if self._mode_live(now_ns) else False

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
        return self._telemetry({"t": "hi", "role": "lamp", "name": self.name, "v": 1, "audio": False})

    def status_datagram(self, status: dict) -> bytes:
        """Lamp telemetry (type 4, ``t`` = "lamp") in exactly the shape of docs/ftm-protocol.md. ValueError
        if the dict does not match that shape."""
        doc = {"t": "lamp"}
        doc.update(_validate_status(status))
        return self._telemetry(doc)

    def set_status(self, status: dict) -> None:
        """Hand the session the current lamp status; poll() sends it about once a second. ValueError (and the
        previous status is kept) if it is not exactly the spec shape."""
        self._status = _validate_status(status)

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

    def _draw_jitter(self) -> int:
        j = self._jitter_source()
        if isinstance(j, bool) or not isinstance(j, int) or not 0 <= j <= JITTER_MAX_NS:
            raise ValueError(f"jitter_source must return an int of ns in 0..{JITTER_MAX_NS}, got {j!r}")
        return j

    def poll(self, now_ns: int) -> list[bytes]:
        """Datagrams to send now. The caller is assumed to send them all."""
        now_ns = _check_now(now_ns)
        self._last_now = now_ns
        self._expire(now_ns)
        out: list[bytes] = []
        silent = self._last_rx is None or now_ns - self._last_rx >= self.hello_silence_ns
        # After a conductor restart the peer table is empty: SyncReq alone re-creates a peer that is answered
        # (SyncResp) but never sent Assign, Control, events or bass until it says hello again. Silence of the
        # whole conductor is therefore not enough: silence of everything except SyncResp also means hello.
        orphaned = self._joined and (self._last_content is None or now_ns - self._last_content >= self.content_silence_ns)
        if (not self._joined or silent or orphaned) and (
                self._last_hello is None or now_ns - self._last_hello >= self.hello_interval_ns):
            out.append(self.hello_datagram())
            self._last_hello = now_ns
            self._c["sent_hello"] += 1
        if self._joined and (self._next_sync is None or now_ns >= self._next_sync):
            # the SyncReq loop is also the keepalive: the conductor drops a peer silent for 6 s
            gap = FAST_SYNC_NS if self._sync_sent + 1 <= FAST_SYNC_COUNT else SLOW_SYNC_NS + self._draw_jitter()
            rid = self._new_id()
            if rid is not None:
                while len(self._outstanding) >= self.max_outstanding:
                    self._outstanding.popitem(last=False)
                    self._c["sync_evicted"] += 1
                self._outstanding[rid] = now_ns
                out.append(SyncReq(rid).encode())
                self._c["sent_sync"] += 1
                self._sync_sent += 1
            self._next_sync = now_ns + gap
        if self._status is not None and (
                self._last_status is None or now_ns - self._last_status >= STATUS_INTERVAL_NS):
            doc = {"t": "lamp"}
            doc.update(self._status)
            out.append(self._telemetry(doc))
            self._last_status = now_ns
            self._c["sent_status"] += 1
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

    def _note_rx(self, now_ns: int) -> None:
        """Any datagram from the conductor is proof of life. If it was silent for too long the mode is
        suspended (stays 'off') until a fresh Assign/Control brings it back."""
        if self._last_rx is not None and now_ns - self._last_rx > self.conductor_lost_ns:
            self._suspended = True
            self._c["conductor_lost"] += 1
        self._last_rx = now_ns

    def _handle(self, data: bytes, now_ns: int) -> list[Output]:
        if isinstance(data, bytes) and data:
            self._note_rx(now_ns)
            if data[0] in ftm_client.AUDIO_TYPES:  # 6..10: audio, a lamp ignores it and never parses it
                self._c["audio_ignored"] += 1
                return []
            if data[0] not in _KNOWN_TYPES:
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
        self._last_content = now_ns  # a parsed Assign, Control, event or bass envelope (even a duplicate)
        if isinstance(pkt, EventPacket):
            if pkt.target != ftm_client.TARGET_ALL and pkt.target != self._index:
                self._c["not_for_me"] += 1
                return []
            if self._event_seen.seen(pkt.seq, now_ns):
                self._c["dup_event"] += 1
                return []
            ev = self.normalizer.normalize_event(pkt, now_ns)
            if ev is None:
                return []  # not remembered: a later copy may still be usable
            self._event_seen.add(pkt.seq, now_ns)
            self._c["events_out"] += 1
            return [EventOut(ev)]
        if isinstance(pkt, BassEnvelope):
            if self._bass_seen.seen(pkt.seq, now_ns):
                self._c["dup_bass"] += 1
                return []
            samples = self.normalizer.normalize_bass(pkt, now_ns)
            if not samples:
                return []
            self._bass_seen.add(pkt.seq, now_ns)
            self._c["bass_out"] += 1
            return [BassOut(samples)]
        if isinstance(pkt, Assign):
            out = self._on_session(pkt.session, "assign")
            self._index = pkt.index
            self._suspended = False
            return out
        if isinstance(pkt, Control):
            return self._on_control(pkt, now_ns)
        # SyncReq / Hello are client-to-conductor; the lamp is a client and never answers them.
        self._c["unexpected_packet"] += 1
        return []

    def _on_sync_resp(self, pkt: SyncResp, now_ns: int) -> None:
        t0 = self._outstanding.pop(pkt.seq, None)
        if t0 is None:
            why = self._settled.get(pkt.seq)
            key = {"answered": "sync_resp_duplicate", "expired": "sync_resp_expired"}.get(why, "sync_resp_unknown")
            self._c[key] += 1
            return
        if now_ns - t0 > self.sync_timeout_ns:
            self._remember(pkt.seq, "expired")
            self._c["sync_resp_expired"] += 1
            return
        self._remember(pkt.seq, "answered")
        try:
            self.estimator.add_sample(t0, pkt.t1, pkt.t2, now_ns)
        except ValueError:
            self._c["bad_sample"] += 1
            return
        self._c["sync_samples"] += 1
        self._last_resp_t0 = t0 if self._last_resp_t0 is None else max(self._last_resp_t0, t0)
        self._timed_out = False

    def _on_session(self, sid: int, src: str) -> list[Output]:
        restart = False
        if self._session_id is None:
            restart = True  # first session: adopt, nothing to invalidate
        elif sid != self._session_id or self._timed_out:
            restart = True
            if sid != self._session_id and src != self._session_src:
                self._c["session_mismatch"] += 1  # Assign and Control disagree; the newest datagram wins
            self.normalizer.new_epoch()
            self._c["new_epochs"] += 1
            self._outstanding.clear()
            self._settled.clear()
            self._mode = None
            self._index = None
            self._event_seen.clear()
            self._bass_seen.clear()
            self._last_resp_t0 = None
            self._timed_out = False
            self._next_sync = None
            self._sync_sent = 0
        self._joined = True
        self._session_id = sid
        self._session_src = src
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
            out += self._on_session(sid, "control")
        else:
            self._joined = True  # a valid Control ends the hello phase even without a session id
        if "lamp" in doc:  # top level of the Control object; other keys are ignored
            mode_out = self._on_lamp(doc["lamp"])
            if mode_out is not None:
                out.append(mode_out)
        return out

    def _on_lamp(self, lamp: object) -> ModeOut | None:
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
        if self._mode is not None and gen < self._mode[2]:
            self._c["stale_mode"] += 1
            return None
        if self._mode is not None and gen == self._mode[2]:
            # the conductor re-sends Control after every hello: an equal gen is a harmless refresh
            self._c["mode_refresh"] += 1
            self._suspended = False
            return None
        self._mode = (mode, lights, gen)
        self._suspended = False
        self._c["mode_accepted"] += 1
        return ModeOut(mode, lights, gen)
