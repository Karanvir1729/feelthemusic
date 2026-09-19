import random
import struct

import pytest

import ftm_client
from ftm_client import Assign, BassEnvelope, Control, EventPacket, Hello, ParseError, SyncReq, SyncResp, decode

U64 = 2**64 - 1
U32 = 2**32 - 1
LENGTHS = {1: 3, 2: 19, 3: 26}
KNOWN_TYPES = {1, 2, 3, 4, 5, 12, 13}
ALL_CLASSES = (SyncReq, SyncResp, EventPacket, Hello, Assign, BassEnvelope, Control)

# NOTE: golden vectors are built independently with struct.pack from the REPORTED layout. They are
# self-consistent with that report only and do not prove interop with the real conductor.
GOLD_REQ = struct.pack("<BH", 1, 0x1234)
GOLD_RESP = struct.pack("<BHQQ", 2, 0xBEEF, 0x0102030405060708, 0x1112131415161718)
GOLD_EVENT = struct.pack("<BIBBBBHHBQI", 3, 0xA1B2C3D4, 5, 6, 200, 100, 0x0203, 0x0405, 7, 0x1122334455667788, 0x99AABBCC)

GOLD_ASSIGN = struct.pack("<BBI", 5, 0x7E, 0xA1B2C3D4)


def _gold_bass(n):
    body = bytes((i * 37 + 5) % 256 for i in range(n))
    return struct.pack("<BIQBB", 12, 0xCAFEBABE, 0x0102030405060708, 20, n) + body, body


GOLD_BASS0, _ = _gold_bass(0)
GOLD_BASS1, _ = _gold_bass(1)
GOLD_BASS255, _ = _gold_bass(255)

ALL_VALID = [
    Assign(0, 0), Assign(255, U32), Assign(7, 300),
    BassEnvelope(0, 0, 0, ()), BassEnvelope(U32, U64, 255, (0,)), BassEnvelope(1, 2, 3, (0, 1, 254, 255)),
    BassEnvelope(9, 9, 9, tuple(range(256))[:255]),
    Control(""), Control('{"t":"ctl","lat":300}'), Hello('{"t":"hi"}'), Hello(""), Control("h\u00e9llo \u4e16\u754c \U0001f3b5"),

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
        if t in KNOWN_TYPES:
            continue
        for n in (1, 3, 6, 15, 19, 26):
            with pytest.raises(ParseError):
                decode(_pad(t, n))


def test_types_6_to_11_rejected():
    for t in range(6, 12):
        for n in (1, 2, 6, 15, 19, 26):
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
        decode(GOLD_REQ).seq = 1


def _only_valid_or_parse_error(data):
    try:
        r = decode(data)
    except ParseError:
        return
    assert isinstance(r, ALL_CLASSES)


def test_fuzz_random():
    rng = random.Random(1234)
    for _ in range(20000):
        data = bytes(rng.getrandbits(8) for _ in range(rng.randint(0, 200)))
        if data and rng.random() < 0.5:
            data = bytes([rng.choice(tuple(KNOWN_TYPES))]) + data[1:]
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
        for t in (0, 1, 2, 3, 5, 6, 11, 14, 255):
            with pytest.raises(ParseError):
                decode(_pad(t, n))
    for n in (0, 1, 14, 65535, 65536):  # type 12 base needs 15; padding gives n=0 in byte 14
        with pytest.raises(ParseError):
            decode(_pad(12, n))
    assert decode(_pad(12, 15)) == BassEnvelope(0, 0, 0, ())
    assert type(decode(_pad(13, 65536))) is Control and type(decode(_pad(4, 65536))) is Hello


# ---- new packet types (all source-reported, none verified against the real conductor) ----

def test_assign_golden_and_wrong_lengths():
    assert len(GOLD_ASSIGN) == ftm_client.ASSIGN_SIZE == 6
    assert GOLD_ASSIGN[2:] == bytes([0xD4, 0xC3, 0xB2, 0xA1])  # u32 little-endian
    assert decode(GOLD_ASSIGN) == Assign(0x7E, 0xA1B2C3D4)
    assert Assign(0x7E, 0xA1B2C3D4).encode() == GOLD_ASSIGN
    for n in range(0, 21):
        if n == 6:
            assert type(decode(_pad(5, n))) is Assign
        else:
            with pytest.raises(ParseError):
                decode(_pad(5, n))
    for cut in range(6):
        with pytest.raises(ParseError):
            decode(GOLD_ASSIGN[:cut])
    with pytest.raises(ParseError):
        decode(GOLD_ASSIGN + b"\x00")


def test_assign_raw_fields():
    a = decode(b"\x05\xff\xff\xff\xff\xff")
    assert a == Assign(255, U32) and type(a.index) is int and type(a.session) is int
    assert decode(bytes([5, 0, 1, 0, 0, 0])).session == 1  # not big-endian


@pytest.mark.parametrize("n", [0, 1, 255])
def test_bass_golden(n):
    data, body = _gold_bass(n)
    assert len(data) == 15 + n
    assert data[1:5] == bytes([0xBE, 0xBA, 0xFE, 0xCA])
    assert data[5:13] == bytes([8, 7, 6, 5, 4, 3, 2, 1])
    assert data[13] == 20 and data[14] == n
    got = decode(data)
    assert got == BassEnvelope(0xCAFEBABE, 0x0102030405060708, 20, tuple(body))
    assert got.encode() == data


def test_bass_length_must_match_n():
    assert ftm_client.BASS_ENVELOPE_BASE_SIZE == 15
    for n in (0, 1, 2, 10, 255):
        good = struct.pack("<BIQBB", 12, 1, 2, 3, n) + bytes(n)
        assert len(decode(good).samples) == n
        for delta in (-2, -1, 1, 2, 255):  # missing / trailing bytes
            cand = good + b"\x07" * delta if delta > 0 else good[:len(good) + delta]
            if len(cand) == len(good):
                continue
            with pytest.raises(ParseError):
                decode(cand)
    # n disagrees with actual body in both directions
    with pytest.raises(ParseError):
        decode(struct.pack("<BIQBB", 12, 1, 2, 3, 5) + bytes(4))  # claims 5, has 4
    with pytest.raises(ParseError):
        decode(struct.pack("<BIQBB", 12, 1, 2, 3, 5) + bytes(6))  # claims 5, has 6
    with pytest.raises(ParseError):
        decode(struct.pack("<BIQBB", 12, 1, 2, 3, 0) + bytes(1))  # claims 0, has 1
    with pytest.raises(ParseError):
        decode(struct.pack("<BIQBB", 12, 1, 2, 3, 255) + bytes(254))
    for cut in range(15):
        with pytest.raises(ParseError):
            decode(_gold_bass(1)[0][:cut])


def test_bass_samples_raw_unsigned_ints():
    got = decode(struct.pack("<BIQBB", 12, 0, 0, 255, 3) + bytes([0, 128, 255]))
    assert got.samples == (0, 128, 255)  # not signed, not scaled
    assert all(type(v) is int for v in got.samples) and type(got.samples) is tuple
    assert got.step_ms == 255 and type(got.step_ms) is int


@pytest.mark.parametrize("cls,t", [(Control, 13), (Hello, 4)])
def test_text_packets(cls, t):
    for text in ("", "{}", '{"t":"hi"}', "caf\u00e9", "\u4e16\u754c", "\U0001f3b5", "a\x00b", " {} \n", "\t", "\r\n"):
        data = bytes([t]) + text.encode("utf-8")
        assert decode(data) == cls(text)
        assert cls(text).encode() == data
    assert decode(bytes([t])) == cls("")  # empty text allowed
    assert decode(bytes([t]) + b"\xc3\xa9") == cls("\u00e9")  # multi-byte, independently built
    assert type(decode(bytes([t]) + b"{}").text) is str


@pytest.mark.parametrize("cls,t", [(Control, 13), (Hello, 4)])
def test_text_packets_invalid_utf8_rejected(cls, t):
    for bad in (b"\xff", b"\xc3", b"\xc3\x28", b"\xe4\xb8", b"\xf0\x9f\x8e", b"\x80", b"ok\xfe",
                b"\xed\xa0\x80", b"\xc0\xaf"):  # lone surrogate and overlong are invalid strict UTF-8
        with pytest.raises(ParseError):
            decode(bytes([t]) + bad)


def test_control_is_not_json_parsed():
    assert decode(b"\x0dnot json at all") == Control("not json at all")
    assert decode(b"\x0d[") == Control("[")


def test_reported_names():
    assert ftm_client.KIND_NAMES == {0: "click", 1: "kick", 2: "snare", 3: "bass", 4: "build", 5: "drop"}
    assert [ftm_client.kind_name(k) for k in range(6)] == ["click", "kick", "snare", "bass", "build", "drop"]
    for k in (6, 7, 200, 255, -1):
        assert ftm_client.kind_name(k) is None
    assert (ftm_client.FLAG_AUDIO, ftm_client.FLAG_HAPTIC, ftm_client.FLAG_MEASURE) == (1, 2, 4)
    assert ftm_client.TARGET_ALL == 0xFF
    assert ftm_client.flag_names(0) == []
    assert ftm_client.flag_names(1) == ["audio"]
    assert ftm_client.flag_names(2) == ["haptic"]
    assert ftm_client.flag_names(4) == ["measure"]
    assert ftm_client.flag_names(7) == ["audio", "haptic", "measure"]
    assert ftm_client.flag_names(0x80) == []
    assert ftm_client.flag_names(0x80 | 2) == ["haptic"]
    assert ftm_client.flag_names(0xFF) == ["audio", "haptic", "measure"]


def test_unknown_kind_flags_target_still_parse():
    for kind in (6, 200, 255):
        pkt = decode(struct.pack("<BIBBBBHHBQI", 3, 1, kind, 0xFF, 255, 255, 1, 2, 0xFF, 3, 4))
        assert pkt.kind == kind and pkt.flags == 0xFF and pkt.target == 0xFF
        assert ftm_client.kind_name(pkt.kind) is None


def test_event_intensity_stays_raw():
    pkt = decode(struct.pack("<BIBBBBHHBQI", 3, 1, 1, 3, 255, 128, 1, 2, 255, 3, 4))
    assert pkt.intensity == 255 and pkt.sharpness == 128 and type(pkt.intensity) is int


def test_fuzz_all_seven_types():
    rng = random.Random(4242)
    for _ in range(20000):
        data = bytes(rng.getrandbits(8) for _ in range(rng.randint(0, 300)))
        if data and rng.random() < 0.7:
            data = bytes([rng.choice(sorted(KNOWN_TYPES))]) + data[1:]
        _only_valid_or_parse_error(data)
    seeds = [p.encode() for p in ALL_VALID]
    assert {s[0] for s in seeds} == KNOWN_TYPES
    for _ in range(2000):
        d = bytearray(rng.choice(seeds))
        op = rng.choice(["flip", "trunc", "ext", "flip"])
        if op == "flip" and d:
            d[rng.randrange(len(d))] ^= 1 << rng.randrange(8)
        elif op == "trunc" and d:
            del d[rng.randrange(len(d)):]
        else:
            d += bytes(rng.getrandbits(8) for _ in range(rng.randint(1, 10)))
        _only_valid_or_parse_error(bytes(d))


# ---- fields renamed to the names in docs/ftm-protocol.md (Tempo, PR #19); still no real-conductor check ----

def test_syncreq_and_syncresp_use_the_spec_name_seq():
    assert decode(GOLD_REQ).seq == 0x1234 and decode(GOLD_RESP).seq == 0xBEEF
    assert SyncReq(seq=5).encode() == struct.pack("<BH", 1, 5)
    assert SyncResp(seq=5, t1=6, t2=7).encode() == struct.pack("<BHQQ", 2, 5, 6, 7)
    assert not hasattr(decode(GOLD_REQ), "req_id") and not hasattr(decode(GOLD_RESP), "req_id")


def test_assign_first_byte_is_named_index():
    a = decode(GOLD_ASSIGN)
    assert (a.index, a.session) == (0x7E, 0xA1B2C3D4)
    assert Assign(index=3, session=9).encode() == struct.pack("<BBI", 5, 3, 9)
    assert not hasattr(a, "unknown_u8")


def test_audio_type_set_is_6_to_10():
    assert ftm_client.AUDIO_TYPES == frozenset({6, 7, 8, 9, 10})
