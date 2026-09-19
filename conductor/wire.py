"""Strict, versioned wire codec: one canonical JSON object per UDP datagram.

Design (a proposal; docs/sync-protocol.md did not exist when this was written, see
conductor/PROTOCOL.md). Limits match lamp/bridge.py: 2048 bytes, nesting depth <= 4.
``decode`` returns a message or raises ``WireError`` -- nothing else, ever.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, Tuple

from . import PROTOCOL_VERSION

MAX_DATAGRAM_BYTES = 2048
MAX_NESTING = 4
INT_MIN = -(2**63)
INT_MAX = 2**63 - 1
MAX_STR = 256
MAX_KEY = 32
MAX_ITEMS = 64

_ID_RE = re.compile(r"\A[A-Za-z0-9_.:-]{1,64}\Z")
_ROLE_RE = re.compile(r"\A[a-z0-9_-]{1,16}\Z")
_KIND_RE = re.compile(r"\A[a-z0-9_.]{1,32}\Z")


class WireError(ValueError):
    """The only exception the codec raises."""


# ---- field validators (shared by encode and decode) -------------------------------

def _int_in(lo: int, hi: int) -> Callable[[Any], int]:
    def check(v: Any) -> int:
        if type(v) is not int:  # rejects bool and float
            raise WireError("expected an integer")
        if not lo <= v <= hi:
            raise WireError("integer out of range")
        return v
    return check


def _pattern(rx: "re.Pattern[str]") -> Callable[[Any], str]:
    def check(v: Any) -> str:
        if type(v) is not str or not rx.match(v):
            raise WireError("bad identifier")
        return v
    return check


def _text(v: Any, limit: int) -> str:
    if type(v) is not str or len(v) > limit:
        raise WireError("bad string")
    try:
        v.encode("utf-8")  # rejects lone surrogates
    except UnicodeEncodeError:
        raise WireError("string is not valid unicode") from None
    return v


def _value(v: Any, depth: int) -> Any:
    """Payload value: null, bool, int, str, list, object. No floats. ``depth`` is the
    nesting level a container here would occupy (top-level object = 1)."""
    if v is None or type(v) is bool:
        return v
    if type(v) is int:
        return _int_in(INT_MIN, INT_MAX)(v)
    if type(v) is str:
        return _text(v, MAX_STR)
    if type(v) in (list, dict):
        if depth > MAX_NESTING:
            raise WireError("nesting too deep")
        if len(v) > MAX_ITEMS:
            raise WireError("container too large")
        if type(v) is list:
            return [_value(x, depth + 1) for x in v]
        out = {}
        for k, x in v.items():
            out[_text(k, MAX_KEY)] = _value(x, depth + 1)
        return out
    raise WireError("unsupported payload value")


def _payload(v: Any) -> Dict[str, Any]:
    if type(v) is not dict:
        raise WireError("payload must be an object")
    return _value(v, 2)


_ns = _int_in(0, INT_MAX)


# ---- messages ---------------------------------------------------------------------

@dataclass(frozen=True)
class Hello:
    T = "hello"
    cid: str
    role: str


@dataclass(frozen=True)
class Welcome:
    T = "welcome"
    cid: str
    hub: str


@dataclass(frozen=True)
class Probe:
    T = "probe"
    id: int
    t0: int


@dataclass(frozen=True)
class ProbeReply:
    T = "probe_reply"
    id: int
    t0: int
    t1: int
    t2: int


@dataclass(frozen=True)
class Event:
    T = "event"
    seq: int
    kind: str
    pts: int
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Bye:
    T = "bye"
    cid: str


_ident = _pattern(_ID_RE)
_SPEC: Dict[str, Tuple[type, Dict[str, Callable[[Any], Any]]]] = {
    "hello": (Hello, {"cid": _ident, "role": _pattern(_ROLE_RE)}),
    "welcome": (Welcome, {"cid": _ident, "hub": _ident}),
    "probe": (Probe, {"id": _int_in(0, INT_MAX), "t0": _ns}),
    "probe_reply": (ProbeReply, {"id": _int_in(0, INT_MAX), "t0": _ns, "t1": _ns, "t2": _ns}),
    "event": (Event, {"seq": _int_in(0, INT_MAX), "kind": _pattern(_KIND_RE), "pts": _ns,
                      "payload": _payload}),
    "bye": (Bye, {"cid": _ident}),
}

Message = Any  # one of the dataclasses above


def _from_fields(t: str, raw: Dict[str, Any]) -> Message:
    cls, validators = _SPEC[t]
    if set(raw) != set(validators):
        raise WireError("wrong field set for " + t)
    return cls(**{name: check(raw[name]) for name, check in validators.items()})


# ---- encode -----------------------------------------------------------------------

def encode(msg: Message) -> bytes:
    """Canonical bytes: UTF-8, sorted keys, no whitespace, integers only."""
    t = getattr(type(msg), "T", None)
    if t not in _SPEC or type(msg) is not _SPEC[t][0]:
        raise WireError("not a wire message")
    raw = {f.name: getattr(msg, f.name) for f in fields(msg)}
    valid = _from_fields(t, raw)  # validates every field
    obj = {f.name: getattr(valid, f.name) for f in fields(valid)}
    obj["t"] = t
    obj["v"] = PROTOCOL_VERSION
    try:
        data = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, RecursionError) as exc:
        raise WireError("cannot encode: %s" % exc) from None
    if len(data) > MAX_DATAGRAM_BYTES:
        raise WireError("message exceeds %d bytes" % MAX_DATAGRAM_BYTES)
    return data


# ---- decode -----------------------------------------------------------------------

def _nesting_depth(data: bytes) -> int:
    """Deepest [ / { nesting, ignoring brackets in strings (same scan as lamp/bridge.py)."""
    depth = deepest = 0
    in_string = escaped = False
    for byte in data:
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:
                escaped = True
            elif byte == 0x22:
                in_string = False
        elif byte == 0x22:
            in_string = True
        elif byte in (0x5B, 0x7B):
            depth += 1
            deepest = max(deepest, depth)
        elif byte in (0x5D, 0x7D):
            depth = max(0, depth - 1)
    return deepest


def _no_dupes(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise WireError("duplicate key")
        out[k] = v
    return out


def _no_constant(name):
    raise WireError("non-finite number")


def _no_float(text):
    raise WireError("floats are not allowed")


def decode(data: bytes) -> Message:
    """Parse and validate one datagram. Returns a message or raises WireError."""
    try:
        return _decode(data)
    except WireError:
        raise
    except Exception:  # RecursionError, MemoryError, anything unforeseen
        raise WireError("undecodable datagram") from None


def _decode(data: Any) -> Message:
    if not isinstance(data, (bytes, bytearray)):
        raise WireError("datagram must be bytes")
    data = bytes(data)
    if not data:
        raise WireError("empty datagram")
    if len(data) > MAX_DATAGRAM_BYTES:
        raise WireError("datagram too large")
    if _nesting_depth(data) > MAX_NESTING:
        raise WireError("nesting too deep")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise WireError("not UTF-8") from None
    try:
        obj = json.loads(text, object_pairs_hook=_no_dupes,
                         parse_constant=_no_constant, parse_float=_no_float)
    except json.JSONDecodeError:
        raise WireError("not JSON") from None
    if type(obj) is not dict:
        raise WireError("top level must be an object")
    v = obj.get("v")
    if type(v) is not int:
        raise WireError("missing or non-integer version")
    if v != PROTOCOL_VERSION:
        raise WireError("unsupported protocol version")
    t = obj.get("t")
    if type(t) is not str or t not in _SPEC:
        raise WireError("unknown message type")
    body = {k: x for k, x in obj.items() if k not in ("t", "v")}
    return _from_fields(t, body)
