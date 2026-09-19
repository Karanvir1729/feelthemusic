"""Structure-only parser for the native FTM Conductor datagrams (task #19).

Parsing only: no sockets, no discovery, no clock estimator, no behaviour. (Reusable estimator logic
that differences the four sync stamps lives in conductor/clock.py on origin/claude/6-conductor-core.)

What is verified
----------------
NOTHING in this module has been checked against the real conductor or against any real packet
capture. The layout below comes from one source only: a teammate's hub message (text of task #6).
The shipped Swift source was not readable. Golden vectors in the tests are self-consistent with this
reported layout and do not prove interop.

Reported (from the teammate's message), all little-endian, first byte of every datagram = type:
  * Port 47300, Bonjour service ``_feelthemusic._udp``.
  * type 1 SyncReq   : u8 type, u16 id                                   =  3 bytes
  * type 2 SyncResp  : u8 type, u16 id, u64, u64                         = 19 bytes
  * type 3 EventPacket: u8 type, u32 seq, u8 kind, u8 flags, u8 intensity (x255), u8 sharpness (x255),
                        u16 durationMs, u16 freqHz, u8 target, u64 masterTs, u32 leadUs = 26 bytes
    (the teammate says 26 bytes, not the 27 the source comment claims; the struct here sums to 26).
  * type 4 Hello     : a client "telemetry hello" carrying JSON with t = "hi" (nothing else reported).
  * type 5 Assign    : u8 type, u8, u32                                  =  6 bytes
    The u32 is reported to be a session id (the conductor answers a client hello with Assign plus
    Control carrying a session u32 and lat 300). The meaning of the u8 is NOT given.
  * type 12 BassEnvelope: u8 type, u32 seq, u64 startTs, u8 stepMs, u8 n, then n bytes (u8)
                        = 15 + n bytes, n in 0..255
  * type 13 Control  : u8 type followed by a UTF-8 JSON document (variable length)
  * Event kinds (names): click=0, kick=1, snare=2, bass=3, build=4, drop=5.
    Flag bits: audio=1, haptic=2, measure=4. Target 0xFF means all.
  * Conductor behaviour, context only and NOT a parser concern: peers are keyed by ip:port and
    dropped after 6 s without traffic; sync replies are delayed randomly 0 to 15 ms.
  * This text was extracted from the hub UI by a teammate and was written by the teammate Nyquist,
    who reports having verified the encode calls. It is still source-reported, not tested here.
  * Reported semantics, NOT used or relied on here: the conductor already adds the 300 ms room
    budget L into masterTs (a native presentation timestamp), so a consumer must not add L again.
  * Reported: every t1, t2 and masterTs comes from the conductor's clock and the client's estimator
    only differences the four stamps.

Inferred (not given), new in this revision:
  * Hello (type 4) as "u8 type + UTF-8 text" is INFERRED from the similar type 13 description. The
    source only says it carries JSON with t = "hi"; the exact layout is not reported.
  * Field names unknown_u8/session (Assign), seq/start_ts/step_ms/samples (BassEnvelope) and text
    (Control, Hello) are ours; ``session`` is the reported meaning of the u32, ``unknown_u8`` is
    deliberately unnamed because the source names nothing.

Inferred (not given):
  * The two u64 fields of SyncResp are most likely the conductor receive/send stamps t1 and t2.
    The field NAMES ``t1`` and ``t2`` are inferred. The u16 is named ``req_id`` by inference from
    "a request id"; the source did not name it.
  * Field names of EventPacket are the reported names converted to snake_case (duration_ms,
    freq_hz, master_ts, lead_us).

Unknown (deliberately not modelled, must not be invented):
  * The meaning of Assign's u8.
  * The exact layout of type 4 beyond "JSON with t = hi".
  * The audio anchor packet and audio access unit formats, and any type not listed above.
  * The meaning of any flag bit above 4 or any kind above 5 (unknown values still parse).
  * What exactly the u16 in SyncReq/SyncResp is.
  * The referenced file base-protocol-for-new-clients.md, which we have not read.

Fields are stored as the raw decoded integers: no scaling, no enums, no unit conversion. The reported
names (KIND_NAMES, FLAG_*, TARGET_ALL) are lookup helpers OUTSIDE the dataclasses, source-reported and
not verified; unknown kinds/flag bits are never rejected. Control/Hello text is decoded strictly as
UTF-8 and is NOT parsed as JSON here.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

PORT = 47300
BONJOUR_SERVICE = "_feelthemusic._udp"

TYPE_SYNC_REQ = 1
TYPE_SYNC_RESP = 2
TYPE_EVENT = 3
TYPE_HELLO = 4  # layout inferred, see docstring
TYPE_ASSIGN = 5
TYPE_BASS_ENVELOPE = 12
TYPE_CONTROL = 13

SYNC_REQ_FORMAT = "<BH"
SYNC_RESP_FORMAT = "<BHQQ"
EVENT_FORMAT = "<BIBBBBHHBQI"
ASSIGN_FORMAT = "<BBI"
BASS_ENVELOPE_BASE_FORMAT = "<BIQBB"

SYNC_REQ_SIZE = struct.calcsize(SYNC_REQ_FORMAT)
SYNC_RESP_SIZE = struct.calcsize(SYNC_RESP_FORMAT)
EVENT_SIZE = struct.calcsize(EVENT_FORMAT)
ASSIGN_SIZE = struct.calcsize(ASSIGN_FORMAT)
BASS_ENVELOPE_BASE_SIZE = struct.calcsize(BASS_ENVELOPE_BASE_FORMAT)

assert SYNC_REQ_SIZE == 3
assert SYNC_RESP_SIZE == 19
assert EVENT_SIZE == 26
assert ASSIGN_SIZE == 6
assert BASS_ENVELOPE_BASE_SIZE == 15

# Source-reported names, NOT verified against the real conductor. Lookups only; never used to reject.
KIND_NAMES = {0: "click", 1: "kick", 2: "snare", 3: "bass", 4: "build", 5: "drop"}
FLAG_AUDIO = 1
FLAG_HAPTIC = 2
FLAG_MEASURE = 4
_FLAG_NAMES = ((FLAG_AUDIO, "audio"), (FLAG_HAPTIC, "haptic"), (FLAG_MEASURE, "measure"))
TARGET_ALL = 0xFF


def kind_name(kind: int) -> str | None:
    """Reported name of an event kind, or None if unknown (unknown kinds still parse)."""
    return KIND_NAMES.get(kind)


def flag_names(flags: int) -> list[str]:
    """Reported names of the set flag bits, in bit order; unknown bits are ignored."""
    return [name for bit, name in _FLAG_NAMES if flags & bit]


class ParseError(ValueError):
    """The datagram is not a well-formed packet of the reported layout."""


@dataclass(frozen=True)
class SyncReq:
    req_id: int  # u16, name inferred

    def encode(self) -> bytes:
        """Produce the layout described in the module docstring; not checked against the real app."""
        return struct.pack(SYNC_REQ_FORMAT, TYPE_SYNC_REQ, self.req_id)


@dataclass(frozen=True)
class SyncResp:
    req_id: int  # u16, name inferred
    t1: int  # u64, name inferred (likely conductor receive stamp)
    t2: int  # u64, name inferred (likely conductor send stamp)

    def encode(self) -> bytes:
        """Produce the layout described in the module docstring; not checked against the real app."""
        return struct.pack(SYNC_RESP_FORMAT, TYPE_SYNC_RESP, self.req_id, self.t1, self.t2)


@dataclass(frozen=True)
class EventPacket:
    seq: int
    kind: int
    flags: int
    intensity: int  # raw 0..255, not scaled
    sharpness: int  # raw 0..255, not scaled
    duration_ms: int
    freq_hz: int
    target: int
    master_ts: int  # raw u64, not converted to seconds
    lead_us: int

    def encode(self) -> bytes:
        """Produce the layout described in the module docstring; not checked against the real app."""
        return struct.pack(
            EVENT_FORMAT, TYPE_EVENT, self.seq, self.kind, self.flags, self.intensity, self.sharpness,
            self.duration_ms, self.freq_hz, self.target, self.master_ts, self.lead_us,
        )


@dataclass(frozen=True)
class Assign:
    unknown_u8: int  # raw u8; unnamed in the source, meaning NOT given
    session: int  # raw u32; reported to be a session id

    def encode(self) -> bytes:
        """Produce the layout described in the module docstring; not checked against the real app."""
        return struct.pack(ASSIGN_FORMAT, TYPE_ASSIGN, self.unknown_u8, self.session)


@dataclass(frozen=True)
class BassEnvelope:
    seq: int
    start_ts: int  # raw u64
    step_ms: int
    samples: tuple[int, ...]  # raw 0..255, length n

    def encode(self) -> bytes:
        """Produce the layout described in the module docstring; not checked against the real app."""
        return struct.pack(
            BASS_ENVELOPE_BASE_FORMAT, TYPE_BASS_ENVELOPE, self.seq, self.start_ts, self.step_ms, len(self.samples)
        ) + bytes(self.samples)


@dataclass(frozen=True)
class Control:
    text: str  # strict UTF-8; reported to be a JSON document, NOT parsed here (may be empty)

    def encode(self) -> bytes:
        """Produce the layout described in the module docstring; not checked against the real app."""
        return bytes([TYPE_CONTROL]) + self.text.encode("utf-8")


@dataclass(frozen=True)
class Hello:
    text: str  # layout INFERRED (u8 type + UTF-8), reported content: JSON with t = "hi"

    def encode(self) -> bytes:
        """Inferred layout; not checked against the real app."""
        return bytes([TYPE_HELLO]) + self.text.encode("utf-8")


Packet = SyncReq | SyncResp | EventPacket | Assign | BassEnvelope | Control | Hello

_TABLE = {
    TYPE_SYNC_REQ: (SYNC_REQ_SIZE, SYNC_REQ_FORMAT, SyncReq),
    TYPE_SYNC_RESP: (SYNC_RESP_SIZE, SYNC_RESP_FORMAT, SyncResp),
    TYPE_EVENT: (EVENT_SIZE, EVENT_FORMAT, EventPacket),
    TYPE_ASSIGN: (ASSIGN_SIZE, ASSIGN_FORMAT, Assign),
}


def _decode_text(datagram: bytes, cls: type) -> Packet:
    try:
        return cls(datagram[1:].decode("utf-8", errors="strict"))
    except UnicodeDecodeError as exc:
        raise ParseError(f"type {datagram[0]} text is not valid UTF-8: {exc}") from exc


def _decode_bass(datagram: bytes) -> BassEnvelope:
    if len(datagram) < BASS_ENVELOPE_BASE_SIZE:
        raise ParseError(f"type 12 needs at least {BASS_ENVELOPE_BASE_SIZE} bytes, got {len(datagram)}")
    _, seq, start_ts, step_ms, n = struct.unpack_from(BASS_ENVELOPE_BASE_FORMAT, datagram)
    if len(datagram) != BASS_ENVELOPE_BASE_SIZE + n:
        raise ParseError(f"type 12 with n={n} needs exactly {BASS_ENVELOPE_BASE_SIZE + n} bytes, got {len(datagram)}")
    return BassEnvelope(seq, start_ts, step_ms, tuple(datagram[BASS_ENVELOPE_BASE_SIZE:]))


def decode(datagram: bytes) -> Packet:
    """Decode one datagram. Raises ParseError (only) on anything that is not exactly one packet."""
    if not isinstance(datagram, bytes):
        raise ParseError(f"datagram must be bytes, got {type(datagram).__name__}")
    if not datagram:
        raise ParseError("empty datagram")
    if datagram[0] == TYPE_BASS_ENVELOPE:
        return _decode_bass(datagram)
    if datagram[0] == TYPE_CONTROL:
        return _decode_text(datagram, Control)
    if datagram[0] == TYPE_HELLO:
        return _decode_text(datagram, Hello)
    entry = _TABLE.get(datagram[0])
    if entry is None:
        raise ParseError(f"unknown message type {datagram[0]}")
    size, fmt, cls = entry
    if len(datagram) != size:
        raise ParseError(f"type {datagram[0]} needs exactly {size} bytes, got {len(datagram)}")
    try:
        fields = struct.unpack(fmt, datagram)
    except struct.error as exc:  # unreachable given the length check; kept so nothing else can escape
        raise ParseError(str(exc)) from exc
    return cls(*fields[1:])
