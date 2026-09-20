import json
import random

import pytest

from conductor import wire
from conductor.wire import (Bye, Event, Hello, Probe, ProbeReply, Welcome, WireError,
                            decode, encode)

GOLDEN = [
    (Hello("phone-1", "haptic"),
     b'{"cid":"phone-1","role":"haptic","t":"hello","v":1}'),
    (Welcome("phone-1", "hub-1"),
     b'{"cid":"phone-1","hub":"hub-1","t":"welcome","v":1}'),
    (Probe(7, 1000000000),
     b'{"id":7,"t":"probe","t0":1000000000,"v":1}'),
    (ProbeReply(7, 1000000000, 5000000000, 5000000500),
     b'{"id":7,"t":"probe_reply","t0":1000000000,"t1":5000000000,"t2":5000000500,"v":1}'),
    (Event(42, "kick", 123456789012, {"amp": 800, "band": [50, 100]}),
     b'{"kind":"kick","payload":{"amp":800,"band":[50,100]},"pts":123456789012,"seq":42,'
     b'"t":"event","v":1}'),
    (Event(1, "note", 5, {"name": "é"}),
     b'{"kind":"note","payload":{"name":"\xc3\xa9"},"pts":5,"seq":1,"t":"event","v":1}'),
    (Bye("phone-1"), b'{"cid":"phone-1","t":"bye","v":1}'),
]


@pytest.mark.parametrize("msg,data", GOLDEN)
def test_golden_bytes(msg, data):
    assert encode(msg) == data
    assert decode(data) == msg


def test_encoding_is_deterministic_regardless_of_dict_order():
    a = Event(1, "k", 2, {"b": 1, "a": 2})
    b = Event(1, "k", 2, {"a": 2, "b": 1})
    assert encode(a) == encode(b)


def test_decode_accepts_reordered_and_spaced_input_from_other_implementations():
    data = b' { "v" : 1 , "t" : "probe" , "t0" : 5 , "id" : 1 } '
    assert decode(data) == Probe(1, 5)


def bad(data):
    with pytest.raises(WireError):
        decode(data)


def test_reject_non_object_top_level():
    for d in (b"[]", b"1", b'"x"', b"null", b"true"):
        bad(d)


def test_reject_size_and_encoding():
    ok = encode(Probe(1, 1))
    assert len(ok) < wire.MAX_DATAGRAM_BYTES
    pad = b" " * (wire.MAX_DATAGRAM_BYTES - len(ok))
    assert decode(ok + pad) == Probe(1, 1)          # exactly 2048 is allowed
    bad(ok + pad + b" ")                             # 2049 is not
    bad(b"")
    bad(b"\xff\xfe{}")
    bad(b'{"v":1,"t":"bye","cid":"\xc3("}')          # invalid UTF-8 inside a string
    bad(b"\xef\xbb\xbf" + ok)                        # BOM


def test_reject_versions():
    for v in (b"0", b"2", b"1.0", b'"1"', b"true", b"null"):
        bad(b'{"v":' + v + b',"t":"bye","cid":"a"}')
    bad(b'{"t":"bye","cid":"a"}')


def test_reject_unknown_type_and_field_sets():
    bad(b'{"v":1,"t":"nope"}')
    bad(b'{"v":1,"t":5}')
    bad(b'{"v":1,"t":"bye"}')
    bad(b'{"v":1,"t":"bye","cid":"a","extra":1}')


def test_reject_non_integer_times():
    for pts in (b"1.0", b"1e3", b"1E3", b"true", b'"5"', b"null", b"-1", b"9223372036854775808"):
        bad(b'{"v":1,"t":"event","seq":1,"kind":"k","pts":' + pts + b',"payload":{}}')
    for t0 in (b"1.5", b"false"):
        bad(b'{"v":1,"t":"probe","id":1,"t0":' + t0 + b'}')


def test_bool_is_not_int_on_encode_too():
    with pytest.raises(WireError):
        encode(Probe(True, 1))
    with pytest.raises(WireError):
        encode(Event(1, "k", False, {}))
    with pytest.raises(WireError):
        encode(Event(1, "k", 1.0, {}))


def test_reject_nan_infinity():
    for c in (b"NaN", b"Infinity", b"-Infinity"):
        bad(b'{"v":1,"t":"event","seq":1,"kind":"k","pts":1,"payload":{"a":' + c + b"}}")
    with pytest.raises(WireError):
        encode(Event(1, "k", 1, {"a": float("nan")}))


def test_payload_floats_rejected_everywhere():
    bad(b'{"v":1,"t":"event","seq":1,"kind":"k","pts":1,"payload":{"a":0.5}}')
    with pytest.raises(WireError):
        encode(Event(1, "k", 1, {"a": 0.5}))


def test_reject_duplicate_keys():
    bad(b'{"v":1,"t":"bye","cid":"a","cid":"b"}')
    bad(b'{"v":1,"v":1,"t":"bye","cid":"a"}')
    bad(b'{"v":1,"t":"event","seq":1,"kind":"k","pts":1,"payload":{"a":1,"a":2}}')


def test_nesting_limit_matches_lamp_bridge():
    # depth counts the top-level object as 1; 4 is the maximum, like lamp/bridge.py
    def ev(payload):
        return b'{"v":1,"t":"event","seq":1,"kind":"k","pts":1,"payload":' + payload + b"}"
    assert decode(ev(b'{"a":{"b":1}}'))          # depth 3
    assert decode(ev(b'{"a":[1]}'))              # depth 3
    assert decode(ev(b'{"a":{"b":[]}}'))         # depth 4
    bad(ev(b'{"a":{"b":[[]]}}'))                 # depth 5
    with pytest.raises(WireError):
        encode(Event(1, "k", 1, {"a": {"b": [[]]}}))
    assert wire._nesting_depth(b'{"a":"[[[[[[["}') == 1  # brackets inside strings ignored
    bad(b"[" * 3000)
    bad(b'{"a":' * 500 + b"1" + b"}" * 500)


def test_string_and_identifier_rules():
    bad(b'{"v":1,"t":"bye","cid":""}')
    bad(b'{"v":1,"t":"bye","cid":"has space"}')
    bad(b'{"v":1,"t":"bye","cid":"line\\n"}')
    bad(b'{"v":1,"t":"bye","cid":"' + b"a" * 65 + b'"}')
    bad(b'{"v":1,"t":"event","seq":1,"kind":"Bad Kind","pts":1,"payload":{}}')
    bad(b'{"v":1,"t":"event","seq":1,"kind":"k","pts":1,"payload":{"a":"\\ud800"}}')  # lone surrogate
    bad(b'{"v":1,"t":"event","seq":1,"kind":"k","pts":1,"payload":[]}')
    bad(b'{"v":1,"t":"event","seq":1,"kind":"k","pts":1,"payload":{"a":"' + b"x" * 300 + b'"}}')


def test_encode_rejects_oversize_and_foreign_objects():
    big = {"k%d" % i: "x" * 60 for i in range(40)}
    with pytest.raises(WireError):
        encode(Event(1, "k", 1, big))
    for junk in (None, {}, "x", 5, object()):
        with pytest.raises(WireError):
            encode(junk)


def test_decode_rejects_non_bytes():
    for junk in (None, "str", 5, [], {}):
        bad(junk)


def test_int_extremes_round_trip():
    e = Event(2**63 - 1, "k", 2**63 - 1, {"n": -(2**63)})
    assert decode(encode(e)) == e


def test_time_fields_are_exact_integers_not_floats():
    # 2**53 + 1 is not representable as a float; it must survive intact
    t = 2**53 + 1
    assert decode(encode(Probe(1, t))).t0 == t


def _valid_corpus():
    return [encode(m) for m, _ in GOLDEN]


def test_fuzz_random_bytes_only_wire_error():
    rng = random.Random(1)
    for _ in range(20000):
        n = rng.choice([0, 1, 2, 5, 20, 100, 2048, 2049, 4000])
        data = bytes(rng.getrandbits(8) for _ in range(n))
        try:
            decode(data)
        except WireError:
            pass


def test_fuzz_mutated_valid_messages():
    rng = random.Random(2)
    corpus = _valid_corpus()
    accepted = 0
    for _ in range(30000):
        b = bytearray(rng.choice(corpus))
        for _ in range(rng.randint(1, 4)):
            op = rng.random()
            i = rng.randrange(len(b)) if b else 0
            if op < 0.4 and b:
                b[i] = rng.getrandbits(8)
            elif op < 0.6 and b:
                del b[i]
            elif op < 0.8:
                b.insert(i, rng.choice(b'{}[]",:0123456789eE.-+ntfaNI'))
            else:
                b[i:i] = rng.choice(corpus)[: rng.randint(0, 30)]
        try:
            m = decode(bytes(b))
        except WireError:
            continue
        accepted += 1
        assert decode(encode(m)) == m  # anything accepted re-encodes canonically
    assert accepted > 0


def test_fuzz_deep_and_hostile_json_terminates_with_wire_error():
    cases = [b"[" * n for n in (1, 5, 100, 2048)] + [b'{"a":' * n for n in (1, 3, 5, 400)]
    cases += [b'{"v":1,"t":"bye","cid":"a"}' + b"\x00" * 100, b"9" * 2048, b'{"v":' + b"9" * 2000 + b"}",
              b'"' * 2048, b"\\" * 2048, b'{"v":1,"t":"event","payload":' + b"[" * 10 + b"}"]
    for c in cases:
        try:
            decode(c)
        except WireError:
            pass


def test_fuzz_structured_json_values():
    rng = random.Random(3)

    def rnd(depth=0):
        r = rng.random()
        if depth > 6 or r < 0.3:
            return rng.choice([None, True, False, 0, -1, 2**70, 1.5, "s", "", "é", float("inf")])
        if r < 0.65:
            return [rnd(depth + 1) for _ in range(rng.randint(0, 3))]
        return {rng.choice(["v", "t", "cid", "pts", "seq", "kind", "payload", "x"]): rnd(depth + 1)
                for _ in range(rng.randint(0, 5))}
    for _ in range(5000):
        try:
            data = json.dumps(rnd(), allow_nan=True).encode()
        except (ValueError, RecursionError):
            continue
        try:
            decode(data)
        except WireError:
            pass


def test_unforeseen_internal_errors_still_surface_as_wire_error(monkeypatch):
    for exc in (RuntimeError("boom"), RecursionError(), MemoryError(), ZeroDivisionError()):
        def explode(*a, **k):
            raise exc
        monkeypatch.setattr(wire.json, "loads", explode)
        with pytest.raises(WireError):
            decode(b'{"v":1,"t":"bye","cid":"a"}')
