import random
import struct

import pytest

import ftm_client
from ftm_client import EventPacket, ParseError, SyncReq, SyncResp, decode

U64 = 2**64 - 1
U32 = 2**32 - 1
LENGTHS = {1: 3, 2: 19, 3: 26}

# NOTE: golden vectors are built independently with struct.pack from the REPORTED layout. They are
# self-consistent with that report only and do not prove interop with the real conductor.
GOLD_REQ = struct.pack("<BH", 1, 0x1234)
GOLD_RESP = struct.pack("<BHQQ", 2, 0xBEEF, 0x0102030405060708, 0x1112131415161718)
GOLD_EVENT = struct.pack("<BIBBBBHHBQI", 3, 0xA1B2C3D4, 5, 6, 200, 100, 0x0203, 0x0405, 7, 0x1122334455667788, 0x99AABBCC)

ALL_VALID = [
    SyncReq(0x1234), SyncReq(0), SyncReq(0xFFFF),
    SyncResp(1, 2, 3), SyncResp(0xFFFF, U64, U64),
    EventPacket(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    EventPacket(U32, 255, 255, 255, 255, 65535, 65535, 255, U64, U32),
]


def _pad(t, n):
    """n bytes long, first byte t (empty for n == 0)."""
    return bytes([t]) + b"\x00" * (n - 1) if n else b""


def test_sizes():
    assert ftm_client.SYNC_REQ_SIZE == 3
    assert ftm_client.SYNC_RESP_SIZE == 19
    assert ftm_client.EVENT_SIZE == 26
    assert len(GOLD_REQ) == 3 and len(GOLD_RESP) == 19 and len(GOLD_EVENT) == 26


def test_golden_bytes_decode_little_endian():
    assert GOLD_REQ == bytes([1, 0x34, 0x12])
    assert decode(GOLD_REQ) == SyncReq(0x1234)
    assert decode(GOLD_RESP) == SyncResp(0xBEEF, 0x0102030405060708, 0x1112131415161718)
    assert decode(GOLD_EVENT) == EventPacket(0xA1B2C3D4, 5, 6, 200, 100, 0x0203, 0x0405, 7, 0x1122334455667788, 0x99AABBCC)


def test_golden_bytes_encode():
    assert SyncReq(0x1234).encode() == GOLD_REQ
    assert SyncResp(0xBEEF, 0x0102030405060708, 0x1112131415161718).encode() == GOLD_RESP
    assert decode(GOLD_EVENT).encode() == GOLD_EVENT
    assert GOLD_EVENT[1:5] == bytes([0xD4, 0xC3, 0xB2, 0xA1])
    assert GOLD_EVENT[9:11] == bytes([0x03, 0x02])  # durationMs position, little-endian
    assert GOLD_EVENT[22:26] == bytes([0xCC, 0xBB, 0xAA, 0x99])  # leadUs is last


@pytest.mark.parametrize("pkt", ALL_VALID)
def test_round_trip(pkt):
    assert decode(pkt.encode()) == pkt
    assert type(decode(pkt.encode())) is type(pkt)


@pytest.mark.parametrize("t", [1, 2, 3])
def test_every_wrong_length_rejected(t):
    for n in range(0, 41):
        if n == LENGTHS[t]:
            assert type(decode(_pad(t, n))) is {1: SyncReq, 2: SyncResp, 3: EventPacket}[t]
        else:
            with pytest.raises(ParseError):
                decode(_pad(t, n))


@pytest.mark.parametrize("t", [1, 2, 3])
def test_trailing_and_truncated_valid(t):
    valid = {1: GOLD_REQ, 2: GOLD_RESP, 3: GOLD_EVENT}[t]
    for extra in (b"\x00", b"\xff", b"\x00" * 100):
        with pytest.raises(ParseError):
            decode(valid + extra)
    for cut in range(len(valid)):
        with pytest.raises(ParseError):
            decode(valid[:cut])


def test_unknown_type_bytes():
    for t in range(256):
        if t in LENGTHS:
            continue
        for n in (1, 3, 19, 26):
            with pytest.raises(ParseError):
                decode(_pad(t, n))


def test_type_zero_rejected():
    with pytest.raises(ParseError):
        decode(b"\x00\x00\x00")


@pytest.mark.parametrize("bad", [None, "abc", 5, [1, 0, 0], bytearray(GOLD_REQ), memoryview(GOLD_REQ), (1, 2, 3)])
def test_non_bytes_rejected(bad):
    with pytest.raises(ParseError):
        decode(bad)


def test_parse_error_is_value_error():
    assert issubclass(ParseError, ValueError)


def test_boundaries():
    assert decode(b"\x01\x00\x00") == SyncReq(0)
    assert decode(b"\x01\xff\xff") == SyncReq(0xFFFF)
    assert decode(b"\x02" + b"\x00" * 18) == SyncResp(0, 0, 0)
    assert decode(b"\x02" + b"\xff" * 18) == SyncResp(0xFFFF, U64, U64)
    assert decode(b"\x03" + b"\x00" * 25) == EventPacket(*[0] * 10)
    assert decode(b"\x03" + b"\xff" * 25) == EventPacket(U32, 255, 255, 255, 255, 65535, 65535, 255, U64, U32)


def test_fields_are_raw_no_interpretation():
    p = decode(GOLD_EVENT)
    q = decode(b"\x03" + b"\xff" * 25)
    assert q.intensity == 255 and type(q.intensity) is int
    assert q.sharpness == 255 and type(q.sharpness) is int
    assert p.intensity == 200 and p.sharpness == 100
    assert q.master_ts == U64 and type(q.master_ts) is int
    for pkt in (decode(GOLD_REQ), decode(GOLD_RESP), p, q):
        assert all(type(v) is int for v in vars(pkt).values())


def test_frozen():
    with pytest.raises(Exception):
        decode(GOLD_REQ).req_id = 1


def _only_valid_or_parse_error(data):
    try:
        r = decode(data)
    except ParseError:
        return
    assert isinstance(r, (SyncReq, SyncResp, EventPacket))


def test_fuzz_random():
    rng = random.Random(1234)
    for _ in range(20000):
        data = bytes(rng.getrandbits(8) for _ in range(rng.randint(0, 200)))
        if data and rng.random() < 0.5:
            data = bytes([rng.choice((1, 2, 3))]) + data[1:]
        _only_valid_or_parse_error(data)


def test_fuzz_mutations():
    rng = random.Random(99)
    seeds = [p.encode() for p in ALL_VALID]
    for _ in range(2000):
        d = bytearray(rng.choice(seeds))
        op = rng.choice(["flip", "trunc", "ext"])
        if op == "flip":
            d[rng.randrange(len(d))] ^= 1 << rng.randrange(8)
        elif op == "trunc":
            del d[rng.randrange(len(d)):]
        else:
            d += bytes(rng.getrandbits(8) for _ in range(rng.randint(1, 10)))
        _only_valid_or_parse_error(bytes(d))


def test_large_input_bounded():
    for n in (0, 1, 65535, 65536):
        for t in (0, 1, 2, 3, 255):
            with pytest.raises(ParseError):
                decode(_pad(t, n))
