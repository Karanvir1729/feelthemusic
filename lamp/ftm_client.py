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
  * Reported semantics, NOT used or relied on here: the conductor already adds the 300 ms room
    budget L into masterTs (a native presentation timestamp), so a consumer must not add L again.
  * Reported: every t1, t2 and masterTs comes from the conductor's clock and the client's estimator
    only differences the four stamps.

Inferred (not given):
  * The two u64 fields of SyncResp are most likely the conductor receive/send stamps t1 and t2.
    The field NAMES ``t1`` and ``t2`` are inferred. The u16 is named ``req_id`` by inference from
    "a request id"; the source did not name it.
  * Field names of EventPacket are the reported names converted to snake_case (duration_ms,
    freq_hz, master_ts, lead_us).

Unknown (deliberately not modelled, must not be invented):
  * The meaning of kind, flags and target values.
  * The audio anchor packet and audio access unit formats, and any other message types.
  * What exactly the u16 in SyncReq/SyncResp is.

Fields are stored as the raw decoded integers: no scaling, no enums, no unit conversion.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

PORT = 47300
BONJOUR_SERVICE = "_feelthemusic._udp"

TYPE_SYNC_REQ = 1
TYPE_SYNC_RESP = 2
TYPE_EVENT = 3

SYNC_REQ_FORMAT = "<BH"
SYNC_RESP_FORMAT = "<BHQQ"
EVENT_FORMAT = "<BIBBBBHHBQI"

SYNC_REQ_SIZE = struct.calcsize(SYNC_REQ_FORMAT)
SYNC_RESP_SIZE = struct.calcsize(SYNC_RESP_FORMAT)
EVENT_SIZE = struct.calcsize(EVENT_FORMAT)

assert SYNC_REQ_SIZE == 3
assert SYNC_RESP_SIZE == 19
assert EVENT_SIZE == 26


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


Packet = SyncReq | SyncResp | EventPacket

_TABLE = {
    TYPE_SYNC_REQ: (SYNC_REQ_SIZE, SYNC_REQ_FORMAT, SyncReq),
    TYPE_SYNC_RESP: (SYNC_RESP_SIZE, SYNC_RESP_FORMAT, SyncResp),
    TYPE_EVENT: (EVENT_SIZE, EVENT_FORMAT, EventPacket),
}


def decode(datagram: bytes) -> Packet:
    """Decode one datagram. Raises ParseError (only) on anything that is not exactly one packet."""
    if not isinstance(datagram, bytes):
        raise ParseError(f"datagram must be bytes, got {type(datagram).__name__}")
    if not datagram:
        raise ParseError("empty datagram")
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
