"""Golden bytes: datagrams recorded from a REAL shipped FTM Conductor (not from docs/ftm-protocol.md).

Source: a 12 s loopback capture by @meharsclaude (hub general seq 1003, 2026-09-19): conductor
`-mode music -source tap -iface any` with test audio, one probe peer on 127.0.0.1, 1950 datagrams.
Only the first datagram of each lamp-relevant type is kept here, verbatim, with its local arrival time
(t_local_ns, the capture host's monotonic clock; conductor and probe share that clock, so the true
offset is 0 by construction). The Control datagram's hex was truncated in the report; its JSON is
verbatim and the leading type byte 0x0d is the documented one.

What this does NOT prove: anything about a lamp on Wi-Fi (offset, jitter), a mode change, a conductor
restart, or a shipped conductor with a `lamp` Control key (this one has none: see test_control_*).
Expected values are read off the recorded bytes, not computed by the code under test.
"""
import json

import pytest

import ftm_client
from ftm_client import Assign, BassEnvelope, EventPacket, Hello, SyncReq, SyncResp, decode
from ftm_session import BassOut, EventOut, Session, SessionOut

MS = 1_000_000
SESSION = 3423683759  # Control JSON "session", and Assign bytes af4411cc little-endian

HELLO = bytes.fromhex(
    "047b22617070223a2263617074757265222c22686170223a66616c73652c226d223a2270726f6265222c2274223a226869227d"
)
SYNCREQ = bytes.fromhex("010100")
ASSIGN = bytes.fromhex("0501af4411cc")
SYNCRESP = bytes.fromhex("020100365c77fdf857000088c595fdf8570000")
CONTROL = b"\x0d" + (
    b'{"asr":48000,"bassGain":1,"codec":1,"cookie":"A4CAgCQAAAAEgICAFkAUABgAAAF3AAABdwAFgICABPjmMAAGgICABPjmMAAGgICAAQI=",'
    b'"fpp":480,"hapticGain":1,"lat":300,"mode":"music","session":3423683759,"v":2}'
)
# Event seq 107 (snare) arrived three times; seqs 108 and 109 are kicks.
EV107 = bytes.fromhex("036b0000000202d8d828000000ff4cb78f78f9570000e0130400")
EV107_ARRIVALS = (96728714189500, 96728726469041, 96728738785500)
EV108 = bytes.fromhex("036c0000000102ff3846000000ffc022f777f957000061c80300")
EV108_ARRIVAL = 96728723492416
EV109 = bytes.fromhex("036d0000000102fd3846000000ffd9217bb1f957000088b20300")
BASS6265 = bytes.fromhex("0c79180000c0fafa0cf95700000a05ffffffffff")
BASS6265_ARRIVALS = (96726944669916, 96726953597791)


def synced_session():
    """A Session whose clock estimate is offset 0 (true for this loopback capture), from symmetric samples."""
    s = Session("golden")
    base = 96_726_000_000_000
    for i in range(3):
        t0 = base + i * 250 * MS
        s.estimator.add_sample(t0, t0 + 2 * MS, t0 + 3 * MS, t0 + 5 * MS)  # 2 ms each way, offset 0
    return s


def test_hello_and_syncreq_sent_by_the_probe_parse_as_specified():
    assert decode(HELLO) == Hello(text='{"app":"capture","hap":false,"m":"probe","t":"hi"}')
    assert decode(SYNCREQ) == SyncReq(seq=1)
    assert len(SYNCREQ) == 3
    assert SyncReq(seq=1).encode() == SYNCREQ  # our encoder produces the bytes a real peer sent


def test_assign_is_six_bytes_and_its_last_four_are_the_control_session():
    assert len(ASSIGN) == 6
    assert decode(ASSIGN) == Assign(index=1, session=SESSION)
    assert Assign(index=1, session=SESSION).encode() == ASSIGN
    assert json.loads(decode(CONTROL).text)["session"] == SESSION  # same number, from the other packet


def test_syncresp_layout_and_plausible_timestamps():
    pkt = decode(SYNCRESP)
    assert len(SYNCRESP) == 19
    assert pkt == SyncResp(seq=1, t1=96726915963958, t2=96726917957000)
    assert 0 < pkt.t2 - pkt.t1 < 10 * MS  # the conductor's hold time is about 2 ms


@pytest.mark.parametrize("raw,seq,kind,flags,inten,sharp,dur,master_ts,lead_us", [
    (EV107, 107, 2, 2, 216, 216, 40, 96728981157708, 267232),
    (EV108, 108, 1, 2, 255, 56, 70, 96728971158208, 247905),
    (EV109, 109, 1, 2, 253, 56, 70, 96729936110041, 242312),
])
def test_event_packets_decode_to_recorded_values(raw, seq, kind, flags, inten, sharp, dur, master_ts, lead_us):
    assert len(raw) == 26
    assert decode(raw) == EventPacket(seq=seq, kind=kind, flags=flags, intensity=inten, sharpness=sharp,
                                      duration_ms=dur, freq_hz=0, target=255, master_ts=master_ts, lead_us=lead_us)
    assert ftm_client.kind_name(kind) in ("kick", "snare")
    assert ftm_client.flag_names(flags) == ["haptic"]  # a haptic-only event: no audio flag, no measure flag


def test_bass_envelope_decodes_to_recorded_values():
    pkt = decode(BASS6265)
    assert len(BASS6265) == 15 + 5
    assert pkt == BassEnvelope(seq=6265, start_ts=96727176248000, step_ms=10, samples=(255,) * 5)


def test_control_from_the_shipped_conductor_has_no_lamp_key_and_no_gen():
    """docs/ftm-protocol.md describes a `lamp` key with `gen`; the recorded conductor sends neither."""
    doc = json.loads(decode(CONTROL).text)
    assert doc["v"] == 2 and doc["session"] == SESSION and doc["lat"] == 300 and doc["mode"] == "music"
    assert "lamp" not in doc and "gen" not in doc and "role" not in doc


def test_session_adopts_assign_then_control_and_stays_off_without_a_lamp_key():
    s = Session("golden")
    now = 96726916006875
    assert s.on_datagram(ASSIGN, now) == [SessionOut(SESSION)]
    assert s.on_datagram(CONTROL, now + 36 * 1000) == []  # same session: not a restart
    assert s.session_id == SESSION
    assert s.current_mode(now + 1 * MS) == "off"  # no `lamp` key: the safe default holds, nothing dances
    assert s.stats()["bad_control"] == 0 and s.stats()["parse_errors"] == 0


def test_event_sent_three_times_yields_one_event_and_no_second_L():
    s = synced_session()
    outs = [s.on_datagram(EV107, t) for t in EV107_ARRIVALS]
    assert [len(o) for o in outs] == [1, 0, 0]
    ev = outs[0][0].event
    assert isinstance(outs[0][0], EventOut) and ev.seq == 107 and ev.kind == "snare"
    assert s.stats()["dup_event"] == 2 and s.stats()["events_out"] == 1
    # masterTs = conductor send time + lead: the event is due lead_us after it was sent. With offset 0 that
    # is arrival + about 267 ms (minus the send delay). A second +L (300 ms) would put it ~567 ms out.
    lead_ns = 267232 * 1000
    assert ev.due_ns == 96728981157708  # offset 0: due is the recorded masterTs, unchanged
    assert abs((ev.due_ns - EV107_ARRIVALS[0]) - lead_ns) < 1 * MS
    assert (ev.due_ns - EV107_ARRIVALS[0]) < 300 * MS  # L is already inside masterTs


def test_distinct_event_seqs_are_not_deduplicated_against_each_other():
    s = synced_session()
    assert len(s.on_datagram(EV108, EV108_ARRIVAL)) == 1
    assert len(s.on_datagram(EV109, EV108_ARRIVAL + 10 * MS)) == 1
    assert s.stats()["dup_event"] == 0


def test_bass_sent_twice_yields_one_batch_starting_at_startts():
    s = synced_session()
    outs = [s.on_datagram(BASS6265, t) for t in BASS6265_ARRIVALS]
    assert [len(o) for o in outs] == [1, 0]
    batch = outs[0][0]
    assert isinstance(batch, BassOut) and len(batch.samples) == 5
    assert [x.due_ns for x in batch.samples] == [96727176248000 + i * 10 * MS for i in range(5)]
    assert all(x.level == 1.0 for x in batch.samples)
    lead = batch.samples[0].due_ns - BASS6265_ARRIVALS[0]
    assert 0 < lead < 300 * MS  # startTs already includes L: it is in the future, but by less than L
    assert s.stats()["dup_bass"] == 1


def test_event_and_bass_seq_spaces_are_separate():
    """Recorded: events ran 107-130 and bass 6265-6504. A bass seq equal to an event seq is not a duplicate."""
    s = synced_session()
    assert len(s.on_datagram(EV107, EV107_ARRIVALS[0])) == 1
    same = BassEnvelope(seq=107, start_ts=96727176248000, step_ms=10, samples=(255,) * 5).encode()
    assert len(s.on_datagram(same, EV107_ARRIVALS[0] + MS)) == 1
