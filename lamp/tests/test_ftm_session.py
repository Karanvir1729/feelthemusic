"""Tests for ftm_session.Session against a FakeConductor built from docs/ftm-protocol.md (Tempo, PR #19).

Nothing here is checked against the real conductor. The FakeConductor encodes that document (see the
ftm_session docstring) with its own struct.pack calls, its own known clock offset and asymmetric delays;
expected values are computed from the fake's constants, never from the code under test.
"""
import heapq
import json
import random
import struct

import pytest

from ftm_session import (
    BassOut, EventOut, MAX_CONTROL_BYTES, ModeOut, Session, SessionOut, parse_control_json,
)

MS = 1_000_000
S = 1_000_000_000
T0 = 10**12
OFFSET = 5_000_000_000_000 + 123_456_789  # conductor_time = local_time + OFFSET
FWD, HOLD, RET = 3 * MS, 5 * MS, 9 * MS  # asymmetric: estimator error is (FWD - RET) / 2 = -3 ms
LEAD = 250 * MS  # remaining lead; the conductor's masterTs already includes L (reported)
TOL = 5 * MS


def ev_bytes(seq=1, kind=1, flags=0, inten=255, sharp=0, dur=10, freq=100, target=0xFF, master_ts=0, lead_us=0):
    return struct.pack("<BIBBBBHHBQI", 3, seq, kind, flags, inten, sharp, dur, freq, target, master_ts, lead_us)


def assign_bytes(session, index=2):
    return struct.pack("<BBI", 5, index, session)


def control_bytes(obj):
    text = obj if isinstance(obj, str) else json.dumps(obj)
    return bytes([13]) + text.encode("utf-8")


def resp_bytes(rid, t1, t2):
    return struct.pack("<BHQQ", 2, rid, t1, t2)


def bass_bytes(seq, start, step, samples):
    return struct.pack("<BIQBB", 12, seq, start, step, len(samples)) + bytes(samples)


class FakeConductor:
    def __init__(self, session=777, lamp=None, answer_hello=True, answer_sync=True):
        self.session = session
        self.lamp = lamp
        self.answer_hello = answer_hello
        self.answer_sync = answer_sync
        self.received = []  # (type byte, data)
        self.hellos = 0
        self.syncreqs = []  # (rid, local send time)

    def control(self):
        doc = {"lat": 300, "session": self.session}
        if self.lamp is not None:
            doc["lamp"] = self.lamp
        return control_bytes(doc)

    def on_send(self, data, sent_local):
        self.received.append((data[0], data))
        out = []
        if data[0] == 4 and json.loads(data[1:]).get("t") == "hi":
            self.hellos += 1
            if self.answer_hello:
                out.append((sent_local + MS, assign_bytes(self.session)))
                out.append((sent_local + MS, self.control()))
        elif data[0] == 1 and self.answer_sync:
            rid = struct.unpack("<H", data[1:3])[0]
            self.syncreqs.append((rid, sent_local))
            t1 = sent_local + FWD + OFFSET
            out.append((sent_local + FWD + HOLD + RET, resp_bytes(rid, t1, t1 + HOLD)))
        return out

    def event_at(self, local_due, **kw):
        return ev_bytes(master_ts=local_due + OFFSET, **kw)


class Rig:
    """Fake clock + wire between a Session and a FakeConductor, 1 ms ticks."""

    def __init__(self, sess, cond, now=T0):
        self.s, self.c, self.now = sess, cond, now
        self.heap = []
        self.n = 0
        self.outputs = []
        self.sent = []  # (time, data)

    def inject(self, at, data):
        self.n += 1
        heapq.heappush(self.heap, (at, self.n, data))

    def run(self, duration, tick=MS, poll=True):
        end = self.now + duration
        while self.now < end:
            while self.heap and self.heap[0][0] <= self.now:
                _, _, data = heapq.heappop(self.heap)
                self.outputs += self.s.on_datagram(data, self.now)
            if poll:
                for d in self.s.poll(self.now):
                    self.sent.append((self.now, d))
                    for at, resp in self.c.on_send(d, self.now):
                        self.inject(at, resp)
            self.now += tick

    def kinds(self, cls):
        return [o for o in self.outputs if isinstance(o, cls)]


JITTER = 10 * MS  # the deterministic injected jitter used by make()


def make(**kw):
    kw.setdefault("jitter_source", lambda: JITTER)
    return Session("lamp-1", **kw)


def synced_rig(**ckw):
    cond = FakeConductor(**ckw)
    rig = Rig(make(), cond)
    rig.run(2 * S)
    assert rig.s.estimator.estimate(rig.now) is not None
    return rig


def feed(s, data, now):
    return s.on_datagram(data, now)


def valid_lamp(mode="follow", lights=True, gen=1):
    return {"mode": mode, "lights": lights, "gen": gen}


# ---------------------------------------------------------------------------------------- hello / join
def test_first_poll_is_the_hello_telemetry():
    s = make()
    out = s.poll(T0)
    assert len(out) == 1 and out[0][0] == 4
    assert json.loads(out[0][1:].decode("utf-8")) == {"t": "hi", "role": "lamp", "name": "lamp-1", "v": 1,
                                                       "audio": False}


def test_hello_bytes_are_exactly_the_spec_shape():
    assert make().poll(T0) == [b'\x04{"t":"hi","role":"lamp","name":"lamp-1","v":1,"audio":false}']


@pytest.mark.parametrize("name", ["", None, 5, b"x", "x" * 65, "a\nb", "\x00", "a\x7fb", "tab\there", "\ud800"])
def test_bad_names_are_rejected(name):
    with pytest.raises(ValueError):
        Session(name)


@pytest.mark.parametrize("name", ["x", "x" * 64, "\u00e9" * 64, "Le Lamp \U0001f3b5"])
def test_good_names_are_accepted_and_sent_verbatim_as_utf8(name):
    d = Session(name).poll(T0)[0]
    assert json.loads(d[1:].decode("utf-8"))["name"] == name


def test_hello_repeats_until_assign_or_control_then_stops():
    s = make()
    assert s.poll(T0)[0][0] == 4
    assert [d for d in s.poll(T0 + 500 * MS) if d[0] == 4 and b'"hi"' in d] == []
    assert any(b'"hi"' in d for d in s.poll(T0 + 1 * S))  # default hello interval is 1 s
    feed(s, assign_bytes(9), T0 + 1 * S + MS)
    later = [d for k in range(1, 19) for d in s.poll(T0 + S + k * 100 * MS)]  # < 2 s of silence
    assert not any(d[0] == 4 and b'"hi"' in d for d in later)


def test_a_valid_control_alone_also_ends_the_hello_phase():
    s = make()
    s.poll(T0)
    feed(s, control_bytes({"lat": 300}), T0 + MS)
    assert s.joined
    assert any(d[0] == 1 for d in s.poll(T0 + 2 * MS))


def test_a_bad_control_does_not_end_the_hello_phase():
    s = make()
    s.poll(T0)
    feed(s, control_bytes("{nope"), T0 + MS)
    assert not s.joined and s.stats()["bad_control"] == 1
    assert not any(d[0] == 1 for d in s.poll(T0 + 2 * S))


def test_end_to_end_join_offset_and_intervals():
    cond = FakeConductor()
    rig = Rig(make(), cond)
    rig.run(5 * S)
    s = rig.s
    assert cond.hellos == 1 and s.joined and s.session_id == 777
    assert [o.session_id for o in rig.kinds(SessionOut)] == [777]
    est = s.estimator.estimate(rig.now)
    assert est is not None and abs(est.offset_ns - OFFSET) <= est.bound_ns
    assert abs(est.offset_ns - OFFSET) <= TOL
    times = [t for _, t in cond.syncreqs]
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert gaps[:40] == [50 * MS] * 40 and gaps[40:] and all(g == 250 * MS + JITTER for g in gaps[40:])


# ---------------------------------------------------------------------------------------- cadence
def request_times(s, start, end, tick=MS):
    out, now = [], start
    while now < end:
        for d in s.poll(now):
            if d[0] == 1:
                out.append(now)
        now += tick
    return out


def cadence_session(**kw):
    s = make(sync_timeout_ns=10**15, max_outstanding=1000, **kw)
    s.poll(T0)
    feed(s, assign_bytes(1), T0 + MS)
    return s


def test_sync_cadence_is_50ms_for_40_requests_then_250ms_plus_injected_jitter():
    jit = [0, 20 * MS, 7 * MS, 13 * MS, 20 * MS, 1 * MS]
    calls = []

    def src():
        calls.append(1)
        return jit[len(calls) - 1] if len(calls) <= len(jit) else 0

    s = cadence_session(jitter_source=src)
    times = request_times(s, T0 + 2 * MS, T0 + 2 * MS + 5 * S)
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert times[0] == T0 + 2 * MS  # the first request goes out on the first poll after joining
    assert gaps[:40] == [50 * MS] * 40  # 40 gaps of 50 ms: one after each of the first 40 requests
    assert gaps[40:46] == [250 * MS + j for j in jit]


def test_jitter_is_not_drawn_during_the_fast_phase():
    calls = []
    s = cadence_session(jitter_source=lambda: calls.append(1) or 0)
    request_times(s, T0 + 2 * MS, T0 + 2 * MS + 40 * 50 * MS)
    assert calls == []


def test_default_jitter_source_stays_within_0_to_20ms_and_varies():
    s = Session("lamp-1", max_outstanding=1000, sync_timeout_ns=10**15)
    s.poll(T0)
    feed(s, assign_bytes(1), T0 + MS)
    times = request_times(s, T0 + 2 * MS, T0 + 2 * MS + 60 * S)
    slow = [b - a for a, b in zip(times, times[1:])][40:]
    assert len(slow) > 100 and all(250 * MS <= g <= 270 * MS for g in slow) and len(set(slow)) > 5


@pytest.mark.parametrize("bad", [-1, 20 * MS + 1, 1.5, True, None, "1"])
def test_out_of_range_jitter_is_a_config_error(bad):
    s = cadence_session(jitter_source=lambda: bad)
    with pytest.raises(ValueError):
        request_times(s, T0 + 2 * MS, T0 + 4 * S)


def test_jitter_of_exactly_0_and_20ms_is_accepted():
    for j in (0, 20 * MS):
        s = cadence_session(jitter_source=lambda j=j: j)
        assert len(request_times(s, T0 + 2 * MS, T0 + 4 * S)) > 41


def test_a_new_session_restarts_the_fast_burst():
    s = cadence_session()
    times = request_times(s, T0 + 2 * MS, T0 + 4 * S)
    assert times[45] - times[44] == 250 * MS + JITTER
    feed(s, assign_bytes(2), T0 + 4 * S)
    again = request_times(s, T0 + 4 * S + MS, T0 + 4 * S + 3 * S)
    assert [b - a for a, b in zip(again, again[1:])][:40] == [50 * MS] * 40


@pytest.mark.parametrize("kw", [{"jitter_source": 5}, {"jitter_source": "x"}])
def test_jitter_source_must_be_callable(kw):
    with pytest.raises(ValueError):
        Session("x", **kw)


# ---------------------------------------------------------------------------------------- hello resend
def test_hello_is_resent_once_a_second_after_2s_of_conductor_silence():
    s = joined()
    last_rx = T0 + MS
    hellos = [t for t in range(0, 6 * S, 10 * MS) if any(b'"hi"' in d for d in s.poll(last_rx + t))]
    assert hellos == [2 * S, 3 * S, 4 * S, 5 * S]


def test_any_datagram_from_the_conductor_defers_the_hello_resend():
    s = joined()
    now = T0 + MS
    for _ in range(2):  # a datagram every 1.5 s keeps the silence under 2 s (and content silence under 5 s)
        now += 1500 * MS
        feed(s, resp_bytes(1, 0, 0), now)
        assert not any(b'"hi"' in d for d in s.poll(now + MS))
    assert not any(b'"hi"' in d for d in s.poll(now + 1999 * MS))
    assert any(b'"hi"' in d for d in s.poll(now + 2 * S + MS))


def test_hello_resend_needs_a_full_second_since_the_last_hello_even_when_silent_long():
    s = joined()
    assert any(b'"hi"' in d for d in s.poll(T0 + 10 * S))
    assert not any(b'"hi"' in d for d in s.poll(T0 + 10 * S + 999 * MS))


# ---------------------------------------------------------------------------------------- events
def test_nothing_is_emitted_before_the_estimator_is_ready():
    s = make()
    s.poll(T0)
    feed(s, assign_bytes(5), T0 + MS)
    now = T0 + 10 * MS
    for i in range(2):  # two samples: still below min_samples = 3
        d = s.poll(now)
        rid = struct.unpack("<H", d[0][1:3])[0]
        t1 = now + OFFSET + FWD
        feed(s, resp_bytes(rid, t1, t1 + HOLD), now + FWD + HOLD + RET)
        now += 200 * MS
        assert feed(s, ev_bytes(master_ts=now + OFFSET + LEAD), now) == []
        assert feed(s, bass_bytes(1, now + OFFSET + LEAD, 20, [255, 128]), now) == []
    assert s.stats()["not_synced"] == 4 and s.stats()["events_out"] == 0
    assert s.stats()["sync_samples"] == 2
    d = s.poll(now)
    rid = struct.unpack("<H", d[0][1:3])[0]
    t1 = now + OFFSET + FWD
    feed(s, resp_bytes(rid, t1, t1 + HOLD), now + FWD + HOLD + RET)
    assert s.stats()["sync_samples"] == 3
    out = feed(s, ev_bytes(master_ts=now + OFFSET + LEAD), now + 20 * MS)
    assert len(out) == 1 and isinstance(out[0], EventOut)


def test_event_due_is_local_now_plus_remaining_lead_with_no_extra_L():
    rig = synced_rig()
    now = rig.now
    out = feed(rig.s, rig.c.event_at(now + LEAD, seq=42, kind=2, inten=255, sharp=51, flags=3), now)
    assert len(out) == 1 and isinstance(out[0], EventOut)
    ev = out[0].event
    assert abs(ev.due_ns - (now + LEAD)) <= TOL
    assert abs(ev.due_ns - (now + LEAD + 300 * MS)) > 250 * MS  # never +L
    assert (ev.seq, ev.kind, ev.intensity, ev.sharpness, ev.flags) == (42, "snare", 1.0, 0.2, frozenset({"audio", "haptic"}))
    assert ev.epoch == rig.s.epoch


def test_bass_samples_start_at_presentation_time_without_L():
    rig = synced_rig()
    now = rig.now
    out = feed(rig.s, bass_bytes(9, now + OFFSET + LEAD, 20, [0, 255, 51]), now)
    assert len(out) == 1 and isinstance(out[0], BassOut)
    got = out[0].samples
    assert [round(g.level, 3) for g in got] == [0.0, 1.0, 0.2]
    for i, g in enumerate(got):
        assert abs(g.due_ns - (now + LEAD + i * 20 * MS)) <= TOL


def test_unknown_event_kind_is_counted_and_never_defaulted():
    rig = synced_rig()
    assert feed(rig.s, rig.c.event_at(rig.now + LEAD, kind=77), rig.now) == []
    assert rig.s.stats()["unknown_kind"] == 1


def test_event_with_a_time_before_zero_is_counted_not_raised():
    rig = synced_rig()
    assert feed(rig.s, ev_bytes(master_ts=1), rig.now) == []
    assert rig.s.stats()["bad_time"] == 1


# ---------------------------------------------------------------------------------------- target filter
def test_event_for_my_index_and_for_everyone_is_accepted_others_are_not_for_me():
    rig = synced_rig()  # the fake conductor assigns index 2
    s, now = rig.s, rig.now
    assert len(feed(s, rig.c.event_at(now + LEAD, seq=1, target=255), now)) == 1
    assert len(feed(s, rig.c.event_at(now + LEAD, seq=2, target=2), now)) == 1
    for seq, target in ((3, 0), (4, 1), (5, 3), (6, 254)):
        assert feed(s, rig.c.event_at(now + LEAD, seq=seq, target=target), now) == []
    st = s.stats()
    assert st["not_for_me"] == 4 and st["events_out"] == 2 and st["normalized"] == 2


def test_before_an_assign_only_target_255_is_accepted():
    rig = synced_rig()
    s, now = rig.s, rig.now
    feed(s, control_bytes({"session": 900}), now)  # a session change by Control only: no Assign yet
    rig.c.session = 900
    rig.c.answer_hello = False  # keep the fake from sending the Assign on a re-hello
    rig.run(2 * S)
    now = rig.now
    assert len(feed(s, rig.c.event_at(now + LEAD, seq=1, target=255), now)) == 1
    assert feed(s, rig.c.event_at(now + LEAD, seq=2, target=2), now) == []  # the OLD index is forgotten
    assert feed(s, rig.c.event_at(now + LEAD, seq=3, target=0), now) == []
    assert s.stats()["not_for_me"] == 2
    feed(s, assign_bytes(900, index=7), now)
    assert len(feed(s, rig.c.event_at(now + LEAD, seq=4, target=7), now)) == 1


def test_a_first_assign_is_needed_and_index_0_is_a_real_index():
    s = make()
    s.poll(T0)
    feed(s, control_bytes({"lat": 300}), T0 + MS)  # joined without an Assign
    assert s._index is None
    feed(s, assign_bytes(4, index=0), T0 + 2 * MS)
    assert s._index == 0


def test_a_new_assign_in_the_same_session_updates_the_index():
    rig = synced_rig()
    s, now = rig.s, rig.now
    feed(s, assign_bytes(777, index=9), now)  # same session id as the fake's
    assert len(feed(s, rig.c.event_at(now + LEAD, seq=1, target=9), now)) == 1
    assert feed(s, rig.c.event_at(now + LEAD, seq=2, target=2), now) == []


def test_bass_envelopes_have_no_target_and_are_never_filtered():
    rig = synced_rig()
    assert len(feed(rig.s, bass_bytes(1, rig.now + OFFSET + LEAD, 20, [1, 2]), rig.now)) == 1
    assert rig.s.stats()["not_for_me"] == 0


# ---------------------------------------------------------------------------------------- de-duplication
def ev(rig, seq, **kw):
    return feed(rig.s, rig.c.event_at(rig.now + LEAD, seq=seq, **kw), rig.now)


def test_a_critical_event_sent_3_times_12ms_apart_produces_one_output():
    rig = synced_rig()
    outs = []
    for i in range(3):
        outs += feed(rig.s, rig.c.event_at(rig.now + LEAD - i * 12 * MS, seq=77), rig.now + i * 12 * MS)
    assert len(outs) == 1 and isinstance(outs[0], EventOut) and outs[0].event.seq == 77
    st = rig.s.stats()
    assert st["dup_event"] == 2 and st["events_out"] == 1 and st["normalized"] == 1


def test_a_bass_envelope_sent_twice_8ms_apart_produces_one_output():
    rig = synced_rig()
    outs = []
    for i in range(2):
        outs += feed(rig.s, bass_bytes(5, rig.now + OFFSET + LEAD, 10, [1, 2, 3, 4, 5]), rig.now + i * 8 * MS)
    assert len(outs) == 1 and isinstance(outs[0], BassOut)
    st = rig.s.stats()
    assert st["dup_bass"] == 1 and st["bass_out"] == 1 and st["dup_event"] == 0


def test_distinct_seqs_are_all_accepted_and_event_and_bass_windows_are_separate():
    rig = synced_rig()
    for seq in range(1, 6):
        assert len(ev(rig, seq)) == 1
    assert rig.s.stats()["dup_event"] == 0
    assert len(feed(rig.s, bass_bytes(3, rig.now + OFFSET + LEAD, 10, [1]), rig.now)) == 1  # same seq value 3
    assert len(feed(rig.s, bass_bytes(4, rig.now + OFFSET + LEAD, 10, [1]), rig.now)) == 1
    assert ev(rig, 3) == [] and rig.s.stats()["dup_event"] == 1 and rig.s.stats()["dup_bass"] == 0
    assert feed(rig.s, bass_bytes(4, rig.now + OFFSET + LEAD, 10, [1]), rig.now) == []
    assert rig.s.stats()["dup_bass"] == 1


def test_seq_window_is_wrap_aware_across_u32():
    rig = synced_rig()
    assert len(ev(rig, 2**32 - 1)) == 1
    assert len(ev(rig, 0)) == 1  # 0 follows 2**32-1: a NEW packet, not a repeat
    assert len(ev(rig, 1)) == 1
    assert ev(rig, 2**32 - 1) == [] and ev(rig, 0) == [] and rig.s.stats()["dup_event"] == 2
    b = rig.now + OFFSET + LEAD
    assert len(feed(rig.s, bass_bytes(2**32 - 1, b, 10, [1]), rig.now)) == 1
    assert len(feed(rig.s, bass_bytes(0, b, 10, [1]), rig.now)) == 1
    assert feed(rig.s, bass_bytes(0, b, 10, [1]), rig.now) == [] and rig.s.stats()["dup_bass"] == 1


@pytest.mark.parametrize("seq", [0, 2**32 - 1, 2**31, 1])
def test_a_lone_repeat_of_an_edge_seq_is_a_duplicate_and_does_not_shadow_its_wrap_neighbour(seq):
    rig = synced_rig()
    assert len(ev(rig, seq)) == 1 and ev(rig, seq) == [] and rig.s.stats()["dup_event"] == 1
    other = 0 if seq == 2**32 - 1 else (2**32 - 1 if seq == 0 else seq + 1)
    assert len(ev(rig, other)) == 1  # a different value is never a repeat of it


def test_seq_window_stays_bounded():
    rig = synced_rig()
    for seq in range(5000):
        ev(rig, seq)
        feed(rig.s, bass_bytes(seq, rig.now + OFFSET + LEAD, 10, [1]), rig.now)
    assert len(rig.s._event_seen) <= 256 and len(rig.s._bass_seen) <= 256
    assert len(rig.s._event_seen) > 0 and len(rig.s._bass_seen) > 0


def test_a_recycled_seq_far_outside_the_window_is_a_new_packet():
    rig = synced_rig()
    assert len(ev(rig, 10)) == 1
    for seq in range(1000, 1300):  # 300 newer packets push seq 10 out of the count-bounded window
        ev(rig, seq)
    assert len(ev(rig, 10)) == 1
    assert rig.s.stats()["dup_event"] == 0


def test_a_recycled_seq_after_the_age_limit_is_a_new_packet():
    rig = synced_rig()
    assert len(ev(rig, 10)) == 1
    rig.now += 500 * MS
    assert ev(rig, 10) == []  # still a repeat well inside the age window
    rig.now += 2 * S
    assert len(ev(rig, 10)) == 1  # the same seq 2.5 s later is not a copy


def test_the_age_limit_is_inclusive_of_exactly_one_second():
    rig = synced_rig()
    assert len(ev(rig, 10)) == 1
    rig.now += S  # exactly SEQ_WINDOW_AGE_NS later: still inside the window
    assert ev(rig, 10) == []
    rig.now += 1  # one ns past the limit (a repeat never refreshes the entry's age)
    assert len(ev(rig, 10)) == 1


def test_a_copy_that_arrived_before_sync_does_not_block_a_later_copy():
    rig = synced_rig()
    s = rig.s
    feed(s, assign_bytes(999, index=2), rig.now)  # new session: estimator reset, not synced
    assert ev(rig, 5) == [] and s.stats()["not_synced"] == 1 and s.stats()["dup_event"] == 0
    rig.c.session = 999
    rig.run(2 * S)
    assert len(ev(rig, 5)) == 1  # the first copy produced no output, so it did not enter the window


def test_both_windows_are_emptied_the_moment_a_new_session_starts():
    rig = synced_rig()
    assert len(ev(rig, 9)) == 1
    feed(rig.s, bass_bytes(9, rig.now + OFFSET + LEAD, 10, [1]), rig.now)
    assert len(rig.s._event_seen) == 1 and len(rig.s._bass_seen) == 1
    feed(rig.s, assign_bytes(31337, index=2), rig.now + MS)
    assert len(rig.s._event_seen) == 0 and len(rig.s._bass_seen) == 0


def test_the_window_is_cleared_on_a_new_session_epoch():
    rig = synced_rig()
    assert len(ev(rig, 9)) == 1
    feed(rig.s, assign_bytes(4242, index=2), rig.now + MS)
    rig.c.session = 4242
    rig.run(2 * S)
    assert len(ev(rig, 9)) == 1  # the new conductor session may reuse seq 9
    assert rig.s.stats()["dup_event"] == 0


def test_an_event_dropped_for_an_unknown_kind_does_not_poison_the_window():
    rig = synced_rig()
    assert ev(rig, 6, kind=77) == [] and rig.s.stats()["unknown_kind"] == 1
    assert len(ev(rig, 6, kind=1)) == 1


# ---------------------------------------------------------------------------------------- audio types
def test_audio_types_6_to_10_are_ignored_and_counted_never_an_error_or_parsed():
    s = make()
    for t in (6, 7, 8, 9, 10):
        for body in (b"", b"\x00" * 4, b"\xff" * 300):
            assert feed(s, bytes([t]) + body, T0) == []
    st = s.stats()
    assert st["audio_ignored"] == 15 and st["unknown_type"] == 0 and st["parse_errors"] == 0
    assert st["datagrams"] == 15 and not s.joined


# ---------------------------------------------------------------------------------------- sync ids
def one_request(s, now):
    d = [x for x in s.poll(now) if x[0] == 1]
    assert len(d) == 1
    return struct.unpack("<H", d[0][1:3])[0]


def joined(**kw):
    s = make(**kw)
    s.poll(T0)
    feed(s, assign_bytes(5), T0 + MS)
    return s


def test_valid_resp_adds_exactly_one_sample_and_a_duplicate_adds_none():
    s = joined()
    now = T0 + 10 * MS
    rid = one_request(s, now)
    t1 = now + OFFSET + FWD
    feed(s, resp_bytes(rid, t1, t1 + HOLD), now + 20 * MS)
    assert s.stats()["sync_samples"] == 1 and len(s.estimator._samples) == 1
    feed(s, resp_bytes(rid, t1, t1 + HOLD), now + 25 * MS)
    st = s.stats()
    assert st["sync_samples"] == 1 and st["sync_resp_duplicate"] == 1 and len(s.estimator._samples) == 1


def test_forged_unknown_id_adds_no_sample():
    s = joined()
    now = T0 + 10 * MS
    rid = one_request(s, now)
    forged = (rid + 1) & 0xFFFF
    feed(s, resp_bytes(forged, now + OFFSET, now + OFFSET), now + 5 * MS)
    st = s.stats()
    assert st["sync_resp_unknown"] == 1 and st["sync_samples"] == 0 and len(s.estimator._samples) == 0
    assert s.outstanding == 1  # the real request is still waiting


def test_resp_for_an_expired_id_is_ignored_and_counted():
    s = joined()
    now = T0 + 10 * MS
    rid = one_request(s, now)
    late = now + 3 * S + 1  # default sync_timeout_ns is 3 s
    s.poll(late)  # the poll expires it
    feed(s, resp_bytes(rid, now + OFFSET, now + OFFSET), late + MS)
    st = s.stats()
    assert st["sync_resp_expired"] == 1 and st["sync_samples"] == 0 and len(s.estimator._samples) == 0


def test_resp_arriving_after_timeout_without_a_poll_is_still_expired():
    s = joined()
    now = T0 + 10 * MS
    rid = one_request(s, now)
    feed(s, resp_bytes(rid, now + OFFSET, now + OFFSET), now + 3 * S + 1)
    st = s.stats()
    assert st["sync_resp_expired"] == 1 and st["sync_samples"] == 0


def test_non_physical_resp_is_counted_bad_sample():
    s = joined()
    now = T0 + 10 * MS
    rid = one_request(s, now)
    feed(s, resp_bytes(rid, OFFSET + 100, OFFSET + 50), now + 20 * MS)  # t2 < t1
    st = s.stats()
    assert st["bad_sample"] == 1 and st["sync_samples"] == 0 and s.outstanding == 0


def test_outstanding_ids_are_capped_and_the_oldest_is_evicted():
    ids = iter(range(100, 1000))
    s = joined(max_outstanding=4, id_source=lambda: next(ids), sync_timeout_ns=10**15)
    sent = []
    for i in range(10):
        sent.append(one_request(s, T0 + 10 * MS + i * 50 * MS))
        assert s.outstanding <= 4
    assert s.outstanding == 4 and s.stats()["sync_evicted"] == 6
    now = T0 + 600 * MS
    feed(s, resp_bytes(sent[0], now + OFFSET, now + OFFSET), now)  # evicted, so unknown
    assert s.stats()["sync_resp_unknown"] == 1 and s.stats()["sync_samples"] == 0
    feed(s, resp_bytes(sent[-1], now + OFFSET, now + OFFSET), now + MS)
    assert s.stats()["sync_samples"] == 1


def test_id_collision_with_an_outstanding_id_is_never_reused():
    s = joined(id_source=lambda: 7)
    assert one_request(s, T0 + 10 * MS) == 7
    assert [d for d in s.poll(T0 + 60 * MS) if d[0] == 1] == []
    assert s.stats()["sync_id_collision"] == 1 and s.outstanding == 1


def test_bad_id_source_is_a_config_error():
    s = joined(id_source=lambda: 70000)
    with pytest.raises(ValueError):
        s.poll(T0 + 10 * MS)


# ---------------------------------------------------------------------------------------- sessions
def test_session_change_bumps_epoch_once_and_invalidates_old_ids_and_samples():
    rig = synced_rig()
    s = rig.s
    e0 = s.epoch
    t = rig.now + 6 * S  # past the slow sync interval, so a request goes out
    rid = one_request(s, t)
    assert s.outstanding == 1 and s.estimator.estimate(t) is not None
    out = feed(s, assign_bytes(888), t + 20 * MS)
    assert out == [SessionOut(888)] and s.epoch == e0 + 1 and s.session_id == 888
    assert s.outstanding == 0 and s.estimator.estimate(t + 21 * MS) is None
    feed(s, resp_bytes(rid, t + OFFSET, t + OFFSET), t + 22 * MS)
    assert s.stats()["sync_resp_unknown"] == 1 and len(s.estimator._samples) == 0
    # the same Assign again, and a Control with the same id: no further bump
    assert feed(s, assign_bytes(888), t + 30 * MS) == []
    assert feed(s, control_bytes({"session": 888}), t + 31 * MS) == []
    assert s.epoch == e0 + 1 and s.stats()["new_epochs"] == 1
    # events from the new epoch are not synced yet, and nothing fires
    assert feed(s, rig.c.event_at(t + LEAD), t + 40 * MS) == []


def test_session_change_via_control_bumps_epoch_and_first_session_does_not():
    s = make()
    s.poll(T0)
    assert feed(s, control_bytes({"session": 1}), T0 + MS) == [SessionOut(1)]
    assert s.epoch == 0  # first adoption has nothing to invalidate
    assert feed(s, control_bytes({"session": 2}), T0 + 2 * MS) == [SessionOut(2)]
    assert s.epoch == 1


def test_event_epoch_tags_follow_the_session():
    rig = synced_rig()
    ev1 = feed(rig.s, rig.c.event_at(rig.now + LEAD), rig.now)[0].event
    assert ev1.epoch == 0
    feed(rig.s, assign_bytes(4242), rig.now + MS)
    rig.c.session = 4242
    rig.run(2 * S)
    ev2 = feed(rig.s, rig.c.event_at(rig.now + LEAD), rig.now)[0].event
    assert ev2.epoch == 1


def test_sync_timeout_then_assign_starts_a_new_session_and_hello_is_resent():
    cond = FakeConductor()
    rig = Rig(make(), cond)
    rig.run(2 * S)
    e0, hellos = rig.s.epoch, cond.hellos
    cond.answer_sync = False  # the conductor stops answering (hello included)
    cond.answer_hello = False
    rig.run(8 * S)
    assert rig.s.stats()["sync_timeouts"] >= 1
    assert cond.hellos > hellos  # hello is re-sent while timed out
    assert rig.s.epoch == e0  # no new epoch until a new Assign arrives
    out = feed(rig.s, assign_bytes(777), rig.now)  # same session id, but after a sync timeout
    assert out == [SessionOut(777)] and rig.s.epoch == e0 + 1
    assert feed(rig.s, assign_bytes(777), rig.now + MS) == [] and rig.s.epoch == e0 + 1


def test_one_lost_request_followed_by_an_answered_one_is_not_a_timeout():
    s = joined()
    r1 = one_request(s, T0 + 10 * MS)  # lost
    now = T0 + 110 * MS
    r2 = one_request(s, now)
    t1 = now + OFFSET + FWD
    feed(s, resp_bytes(r2, t1, t1 + HOLD), now + 20 * MS)
    s.poll(T0 + 10 * MS + 3 * S + 1)
    assert s.stats()["sync_timeouts"] == 0
    e0 = s.epoch
    assert feed(s, assign_bytes(5), T0 + 4 * S) == [] and s.epoch == e0
    assert r1 != r2


# ---------------------------------------------------------------------------------------- mode
def test_mode_is_off_until_a_valid_lamp_key_arrives():
    s = make()
    assert s.current_mode(T0) == "off" and s.current_lights(T0) is False
    feed(s, control_bytes({"session": 1, "lat": 300}), T0)
    assert s.current_mode(T0 + MS) == "off"  # a Control without a lamp key changes nothing


@pytest.mark.parametrize("mode", ["follow", "dance", "light", "off"])
def test_every_reported_mode_is_accepted(mode):
    s = make()
    out = feed(s, control_bytes({"session": 1, "lamp": valid_lamp(mode, True, 5)}), T0)
    assert out == [SessionOut(1), ModeOut(mode, True, 5)]
    assert s.current_mode(T0 + MS) == mode and s.current_lights(T0 + MS) is True


def test_a_steady_mode_never_expires_by_age_while_the_conductor_is_heard():
    s = make()
    feed(s, control_bytes({"session": 1, "lamp": valid_lamp("dance", True, 3)}), T0)
    now = T0
    for _ in range(3600):  # an hour of ordinary traffic, one datagram per second, no new Control
        now += S
        feed(s, resp_bytes(1, 0, 0), now)
        assert s.current_mode(now) == "dance" and s.current_lights(now) is True
    assert not hasattr(s, "mode_max_age_ns")


def test_conductor_silence_over_conductor_lost_ns_forces_off_with_the_4s_default():
    s = make()
    assert s.conductor_lost_ns == 4 * S
    feed(s, control_bytes({"session": 1, "lamp": valid_lamp("dance", True, 3)}), T0)
    assert s.current_mode(T0 + 4 * S) == "dance" and s.current_lights(T0 + 4 * S) is True
    assert s.current_mode(T0 + 4 * S + 1) == "off" and s.current_lights(T0 + 4 * S + 1) is False


def test_conductor_lost_ns_is_configurable():
    s = make(conductor_lost_ns=500 * MS)
    feed(s, control_bytes({"session": 1, "lamp": valid_lamp("light", False, 1)}), T0)
    assert s.current_mode(T0 + 500 * MS) == "light" and s.current_mode(T0 + 500 * MS + 1) == "off"


def test_any_datagram_from_the_conductor_counts_as_life():
    s = make(conductor_lost_ns=S)
    feed(s, control_bytes({"session": 1, "lamp": valid_lamp("follow", True, 1)}), T0)
    feed(s, resp_bytes(1, 0, 0), T0 + 900 * MS)  # an unknown-id SyncResp is still a datagram
    assert s.current_mode(T0 + 1800 * MS) == "follow"
    feed(s, bytes([200]), T0 + 1800 * MS)  # so is an unknown type
    assert s.current_mode(T0 + 2700 * MS) == "follow"


def test_after_a_loss_the_mode_stays_off_until_a_fresh_control_or_assign():
    s = make()
    feed(s, control_bytes({"session": 1, "lamp": valid_lamp("dance", True, 5)}), T0)
    late = T0 + 10 * S  # silent for 10 s
    feed(s, resp_bytes(1, 0, 0), late)  # the conductor is back on the wire, but no Control yet
    assert s.current_mode(late + MS) == "off" and s.current_lights(late + MS) is False
    feed(s, bytes([3]) + bytes(25), late + 2 * MS)  # events do not revive the mode either
    assert s.current_mode(late + 3 * MS) == "off"
    # a stale (lower gen) Control does not revive it
    feed(s, control_bytes({"lamp": valid_lamp("follow", False, 4)}), late + 4 * MS)
    assert s.current_mode(late + 5 * MS) == "off"
    # an equal-gen Control is a refresh: it brings the LAST valid mode back, unchanged
    assert feed(s, control_bytes({"lamp": valid_lamp("dance", True, 5)}), late + 6 * MS) == []
    assert s.current_mode(late + 7 * MS) == "dance" and s.current_lights(late + 7 * MS) is True
    # and it stays alive afterwards
    feed(s, resp_bytes(1, 0, 0), late + 3 * S)
    assert s.current_mode(late + 3 * S + MS) == "dance"


def test_after_a_loss_an_assign_for_the_same_session_revives_the_last_mode():
    s = make()
    feed(s, control_bytes({"session": 1, "lamp": valid_lamp("follow", True, 5)}), T0)
    feed(s, assign_bytes(1), T0 + 10 * S)
    assert s.current_mode(T0 + 10 * S + MS) == "follow"


def test_after_a_loss_a_control_with_a_new_gen_or_session_applies_that_mode():
    s = make()
    feed(s, control_bytes({"session": 1, "lamp": valid_lamp("dance", True, 5)}), T0)
    out = feed(s, control_bytes({"lamp": valid_lamp("light", False, 6)}), T0 + 10 * S)
    assert out == [ModeOut("light", False, 6)] and s.current_mode(T0 + 10 * S + MS) == "light"
    s2 = make()
    feed(s2, control_bytes({"session": 1, "lamp": valid_lamp("dance", True, 5)}), T0)
    out = feed(s2, control_bytes({"session": 2, "lamp": valid_lamp("follow", True, 1)}), T0 + 10 * S)
    assert out == [SessionOut(2), ModeOut("follow", True, 1)]
    assert s2.current_mode(T0 + 10 * S + MS) == "follow"


def test_a_second_loss_after_a_revival_is_a_new_loss():
    s = make()
    feed(s, control_bytes({"session": 1, "lamp": valid_lamp("dance", True, 5)}), T0)
    feed(s, control_bytes({"lamp": valid_lamp("dance", True, 5)}), T0 + 10 * S)
    assert s.current_mode(T0 + 11 * S) == "dance"
    assert s.current_mode(T0 + 15 * S) == "off"
    feed(s, resp_bytes(1, 0, 0), T0 + 20 * S)
    assert s.current_mode(T0 + 20 * S + MS) == "off"


def test_mode_is_off_before_any_datagram_and_after_a_fresh_valid_key_only():
    s = make()
    assert s.current_mode(0) == "off" and s.current_lights(0) is False
    feed(s, resp_bytes(1, 0, 0), T0)  # traffic alone never makes a mode
    assert s.current_mode(T0 + MS) == "off"


def test_unknown_control_keys_are_ignored_and_lamp_must_be_top_level():
    s = make()
    doc = {"v": 2, "session": 1, "lat": 300, "mode": "music", "hapticGain": 1.0, "future": {"a": [1]},
           "lamp": {**valid_lamp("dance", True, 1), "extra": "ignored"}}
    assert feed(s, control_bytes(doc), T0) == [SessionOut(1), ModeOut("dance", True, 1)]
    s2 = make()  # the conductor's own top-level ``mode`` key is NOT the lamp mode, and nesting does not count
    assert feed(s2, control_bytes({"mode": "dance", "x": {"lamp": valid_lamp("dance", True, 1)}}), T0) == []
    assert s2.current_mode(T0 + MS) == "off" and s2.stats()["bad_mode"] == 0 and s2.stats()["parse_errors"] == 0


@pytest.mark.parametrize("bad", ["FOLLOW", "Dance", "", "on", "auto", "follow ", None, 1, True, ["follow"], {}])
def test_unknown_or_non_string_modes_are_ignored_keep_the_last_valid_and_count(bad):
    s = make()
    feed(s, control_bytes({"session": 1, "lamp": valid_lamp("dance", True, 1)}), T0)
    out = feed(s, control_bytes({"lamp": {"mode": bad, "lights": True, "gen": 2}}), T0 + MS)
    assert out == [] and s.stats()["bad_mode"] == 1
    assert s.current_mode(T0 + 2 * MS) == "dance"


def test_a_bad_mode_with_no_prior_mode_leaves_it_off_not_defaulted():
    s = make()
    feed(s, control_bytes({"lamp": {"mode": "party", "lights": True, "gen": 1}}), T0)
    assert s.current_mode(T0 + MS) == "off" and s.stats()["bad_mode"] == 1


@pytest.mark.parametrize("bad", [1, 0, "true", None, 1.0, [], {}])
def test_lights_must_be_a_real_bool(bad):
    s = make()
    out = feed(s, control_bytes({"lamp": {"mode": "follow", "lights": bad, "gen": 1}}), T0)
    assert out == [] and s.stats()["bad_mode"] == 1 and s.current_mode(T0 + MS) == "off"


@pytest.mark.parametrize("bad", [True, "1", None, 1.5, -1, [], 2**63])
def test_gen_must_be_a_real_nonnegative_int(bad):
    s = make()
    out = feed(s, control_bytes({"lamp": {"mode": "follow", "lights": True, "gen": bad}}), T0)
    assert out == [] and s.stats()["bad_mode"] == 1 and s.current_mode(T0 + MS) == "off"


@pytest.mark.parametrize("missing", ["mode", "lights", "gen"])
def test_missing_lamp_fields_are_bad_mode(missing):
    lamp = valid_lamp()
    del lamp[missing]
    s = make()
    assert feed(s, control_bytes({"lamp": lamp}), T0) == [] and s.stats()["bad_mode"] == 1


@pytest.mark.parametrize("lamp", [None, 5, "follow", [1, 2], True])
def test_lamp_key_that_is_not_an_object_is_bad_mode(lamp):
    s = make()
    assert feed(s, control_bytes({"lamp": lamp}), T0) == [] and s.stats()["bad_mode"] == 1


def test_gen_may_not_go_backwards_but_an_equal_gen_is_a_harmless_refresh():
    s = make()
    feed(s, control_bytes({"session": 1, "lamp": valid_lamp("follow", True, 10)}), T0)
    for gen in (9, 0):
        assert feed(s, control_bytes({"lamp": valid_lamp("dance", False, gen)}), T0 + MS) == []
    st = s.stats()
    assert st["stale_mode"] == 2 and st["mode_refresh"] == 0
    # equal gen: not stale, no output, and the mode does NOT change even if the content differs
    assert feed(s, control_bytes({"lamp": valid_lamp("dance", False, 10)}), T0 + 2 * MS) == []
    assert feed(s, control_bytes({"lamp": valid_lamp("follow", True, 10)}), T0 + 3 * MS) == []
    st = s.stats()
    assert st["stale_mode"] == 2 and st["mode_refresh"] == 2 and st["mode_accepted"] == 1
    assert s.current_mode(T0 + 4 * MS) == "follow" and s.current_lights(T0 + 4 * MS) is True
    assert feed(s, control_bytes({"lamp": valid_lamp("dance", False, 11)}), T0 + 5 * MS) == [ModeOut("dance", False, 11)]


def test_gen_tracking_is_per_session():
    s = make()
    feed(s, control_bytes({"session": 1, "lamp": valid_lamp("follow", True, 500)}), T0)
    assert feed(s, control_bytes({"session": 2, "lamp": valid_lamp("light", True, 1)}), T0 + MS) == [
        SessionOut(2), ModeOut("light", True, 1)]
    assert feed(s, control_bytes({"lamp": valid_lamp("dance", True, 1)}), T0 + 2 * MS) == []  # equal: refresh
    assert s.stats()["stale_mode"] == 0
    assert feed(s, control_bytes({"lamp": valid_lamp("dance", True, 0)}), T0 + 3 * MS) == []  # lower: stale
    assert s.stats()["stale_mode"] == 1


# ---------------------------------------------------------------------------------------- session ids
def test_assign_and_control_session_ids_may_disagree_the_newest_wins_and_it_is_counted_not_raised():
    s = make()
    assert feed(s, assign_bytes(5), T0) == [SessionOut(5)] and s.stats()["session_mismatch"] == 0
    assert feed(s, control_bytes({"session": 6, "lamp": valid_lamp("light", True, 1)}), T0 + 2 * MS) == [
        SessionOut(6), ModeOut("light", True, 1)]
    assert s.session_id == 6 and s.stats()["session_mismatch"] == 1
    assert feed(s, assign_bytes(7), T0 + 3 * MS) == [SessionOut(7)]
    assert s.session_id == 7 and s.stats()["session_mismatch"] == 2


def test_agreeing_assign_and_control_are_not_a_mismatch():
    s = make()
    feed(s, assign_bytes(5), T0)
    assert feed(s, control_bytes({"session": 5, "lat": 300}), T0 + MS) == []
    assert feed(s, assign_bytes(5), T0 + 2 * MS) == [] and s.stats()["session_mismatch"] == 0


def test_two_assigns_with_different_sessions_is_a_change_not_a_mismatch():
    s = make()
    feed(s, assign_bytes(5), T0)
    assert feed(s, assign_bytes(6), T0 + MS) == [SessionOut(6)]
    assert s.stats()["session_mismatch"] == 0 and s.session_id == 6
    feed(s, control_bytes({"session": 6}), T0 + 2 * MS)
    assert feed(s, control_bytes({"session": 9}), T0 + 3 * MS) == [SessionOut(9)]
    assert s.stats()["session_mismatch"] == 0


def test_the_session_from_either_source_alone_is_accepted():
    a = make()
    assert feed(a, assign_bytes(11), T0) == [SessionOut(11)] and a.session_id == 11
    c = make()
    assert feed(c, control_bytes({"session": 12}), T0) == [SessionOut(12)] and c.session_id == 12


def test_gen_may_restart_after_a_session_change_and_the_old_mode_is_dropped():
    s = make()
    feed(s, control_bytes({"session": 1, "lamp": valid_lamp("follow", True, 500)}), T0)
    feed(s, assign_bytes(2), T0 + MS)
    assert s.current_mode(T0 + 2 * MS) == "off"  # the old session's mode does not carry over
    out = feed(s, control_bytes({"lamp": valid_lamp("light", True, 1)}), T0 + 3 * MS)
    assert out == [ModeOut("light", True, 1)] and s.current_mode(T0 + 4 * MS) == "light"
    # a Control that changes the session and carries gen 1 in one document is accepted too
    out = feed(s, control_bytes({"session": 3, "lamp": valid_lamp("dance", False, 1)}), T0 + 5 * MS)
    assert out == [SessionOut(3), ModeOut("dance", False, 1)]


def test_bad_session_in_control_ignores_the_whole_control():
    s = make()
    for bad in (True, "7", -1, 2**32, 1.5, None):
        assert feed(s, control_bytes({"session": bad, "lamp": valid_lamp()}), T0) == []
    assert s.stats()["bad_control"] == 6 and s.current_mode(T0 + MS) == "off" and not s.joined


# ---------------------------------------------------------------------------------------- hostile control
def deep(n):
    return "[" * n + "]" * n


@pytest.mark.parametrize("text", [
    "x" * (MAX_CONTROL_BYTES + 1),
    json.dumps({"lamp": valid_lamp(), "pad": "y" * MAX_CONTROL_BYTES}),  # valid JSON, too big
    '{"lamp": {"mode": "follow", "lights": true, "gen": NaN}}',
    '{"lamp": {"mode": "follow", "lights": true, "gen": Infinity}}',
    '{"lamp": {"mode": "follow", "lights": true, "gen": -Infinity}}',
    '{"lamp": {"mode": "follow", "lights": true, "gen": 1e999}}',
    "[1, 2, 3]",
    '"just a string"',
    "42",
    "null",
    "",
    "{",
    '{"a": ' + deep(10000) + "}",
    '{"a": ' + deep(17) + "}",
    '{"lamp": {"mode": "follow", "mode": "off", "lights": true, "gen": 1}}',  # duplicate key
    "{'lamp': 1}",
])
def test_hostile_control_json_is_counted_and_never_raises_or_changes_state(text):
    s = make()
    assert feed(s, control_bytes(text), T0) == []
    assert s.stats()["bad_control"] == 1 and s.current_mode(T0 + MS) == "off" and not s.joined


def test_control_at_depth_cap_is_accepted_and_just_over_is_not():
    ok = '{"a": ' + deep(15) + "}"  # depth 16 in total
    assert parse_control_json(ok) is not None
    assert parse_control_json('{"a": ' + deep(16) + "}") is None
    assert parse_control_json("{" + '"k": "' + "]" * 100 + '"}') == {"k": "]" * 100}  # brackets in strings


def test_control_at_exactly_the_size_cap_is_accepted_over_is_rejected():
    body = '{"p": "' + "a" * (MAX_CONTROL_BYTES - 9) + '"}'
    assert len(body.encode()) == MAX_CONTROL_BYTES and parse_control_json(body) is not None
    assert parse_control_json(body[:-2] + "a" + '"}') is None


def test_size_cap_counts_utf8_bytes_not_characters():
    body = '{"p": "' + "é" * (MAX_CONTROL_BYTES // 2) + '"}'  # 2 bytes per char, over the cap
    assert len(body) < MAX_CONTROL_BYTES < len(body.encode())
    s = make()
    assert feed(s, control_bytes(body), T0) == [] and s.stats()["bad_control"] == 1


def test_control_that_is_not_utf8_is_a_parse_error_and_never_raises():
    s = make()
    assert feed(s, bytes([13]) + b'{"a": "\xff\xfe"}', T0) == []
    assert s.stats()["parse_errors"] == 1 and not s.joined


def test_parse_control_json_accepts_ordinary_documents():
    assert parse_control_json('{"lat": 300, "session": 5, "f": 1.5}') == {"lat": 300, "session": 5, "f": 1.5}


# ---------------------------------------------------------------------------------------- send side
def test_lamp_never_sends_a_syncresp_or_answers_a_syncreq():
    s = joined()
    assert feed(s, struct.pack("<BH", 1, 9), T0 + 5 * MS) == []
    assert s.stats()["unexpected_packet"] == 1
    sent = []
    for i in range(300):
        sent += s.poll(T0 + 10 * MS + i * 50 * MS)
    assert sent and all(d[0] in (1, 4) for d in sent)  # only SyncReq and telemetry, never type 2
    assert not any(d[0] == 2 for d in sent)
    assert [d for d in s.poll(T0 + 20 * S) if d[0] == 2] == []


def test_a_hello_received_from_the_peer_is_counted_and_ignored():
    s = make()
    assert feed(s, bytes([4]) + b'{"t":"hi"}', T0) == [] and s.stats()["unexpected_packet"] == 1


def test_the_sync_loop_is_the_keepalive_and_no_other_telemetry_is_invented():
    for scenario in ("silent_conductor", "joined_and_synced"):
        cond = FakeConductor(answer_hello=(scenario == "joined_and_synced"),
                             answer_sync=(scenario == "joined_and_synced"))
        rig = Rig(Session("lamp-1"), cond)  # every default
        rig.run(60 * S, tick=10 * MS)
        times = [t for t, _ in rig.sent]
        gaps = [b - a for a, b in zip(times, times[1:])]
        assert times and times[0] == T0
        if scenario == "joined_and_synced":
            assert max(gaps) <= 270 * MS  # the SyncReq loop alone keeps the peer alive (6 s drop)
        else:
            assert max(gaps) <= 1 * S + 10 * MS  # hello once a second while unanswered
        for _, d in rig.sent:
            assert d[0] in (1, 4)
            if d[0] == 4:
                assert json.loads(d[1:])["t"] == "hi"  # never a "ka" or any other invented type
        assert "sent_keepalive" not in rig.s.stats()


def test_no_ka_telemetry_exists_anywhere_in_a_long_run():
    s = joined()
    seen = set()
    for i in range(2000):
        for d in s.poll(T0 + 10 * MS + i * 50 * MS):
            seen.add(d[0] if d[0] != 4 else json.loads(d[1:])["t"])
    assert seen <= {1, "hi"}


GOOD_STATUS = {"state": "idle", "mode": "follow", "locked": True, "piC": 12.5, "sdk": "ok", "moves": 3,
               "refused": 1}


def status_docs(s, start, end, tick=10 * MS):
    out, now = [], start
    while now < end:
        for d in s.poll(now):
            if d[0] == 4:
                doc = json.loads(d[1:])
                if doc["t"] == "lamp":
                    out.append((now, doc))
        now += tick
    return out


def test_no_status_is_sent_until_the_caller_provides_one():
    s = joined()
    assert status_docs(s, T0 + 10 * MS, T0 + 5 * S) == []
    assert s.stats()["sent_status"] == 0


def test_status_is_emitted_from_poll_about_once_a_second_with_exactly_the_spec_shape():
    s = joined()
    s.set_status(GOOD_STATUS)
    got = status_docs(s, T0 + 10 * MS, T0 + 10 * MS + 5 * S + 5 * MS, tick=MS)
    assert [t - (T0 + 10 * MS) for t, _ in got] == [0, S, 2 * S, 3 * S, 4 * S, 5 * S]
    assert all(doc == {"t": "lamp", **GOOD_STATUS} for _, doc in got)
    assert list(got[0][1]) == ["t", "state", "mode", "locked", "piC", "sdk", "moves", "refused"]
    assert s.stats()["sent_status"] == 6


def test_status_is_never_sent_faster_than_1s_and_uses_the_latest_status():
    s = joined()
    s.set_status(GOOD_STATUS)
    assert len(status_docs(s, T0 + 10 * MS, T0 + 11 * MS)) == 1
    s.set_status({**GOOD_STATUS, "state": "dancing", "moves": 4})
    assert status_docs(s, T0 + 11 * MS, T0 + 10 * MS + 999 * MS) == []
    got = status_docs(s, T0 + 10 * MS + 999 * MS, T0 + 10 * MS + S + 20 * MS)
    assert len(got) == 1 and got[0][1]["state"] == "dancing" and got[0][1]["moves"] == 4


@pytest.mark.parametrize("state", ["searching", "locked", "dancing", "light", "idle", "error"])
def test_every_spec_state_is_accepted(state):
    s = make()
    s.set_status({**GOOD_STATUS, "state": state})
    assert json.loads(s.poll(T0)[-1][1:])["state"] == state


@pytest.mark.parametrize("patch", [
    {"state": "bogus"}, {"state": "Idle"}, {"state": ""}, {"state": None}, {"state": 1},
    {"mode": 5}, {"mode": ""}, {"mode": "x" * 33}, {"mode": "a\nb"},
    {"locked": 1}, {"locked": "true"}, {"locked": None},
    {"piC": "61"}, {"piC": None}, {"piC": True}, {"piC": float("nan")}, {"piC": float("inf")},
    {"sdk": 5}, {"sdk": ""}, {"sdk": "e" * 65}, {"sdk": "a\x00b"},
    {"moves": True}, {"moves": -1}, {"moves": 1.5}, {"moves": "1"}, {"moves": 2**40},
    {"refused": True}, {"refused": -1}, {"refused": 0.5}, {"refused": None},
])
def test_bad_status_values_are_rejected_at_set_status_and_the_old_status_stays(patch):
    s = make()
    s.set_status(GOOD_STATUS)
    with pytest.raises(ValueError):
        s.set_status({**GOOD_STATUS, **patch})
    assert json.loads(s.poll(T0)[-1][1:]) == {"t": "lamp", **GOOD_STATUS}


@pytest.mark.parametrize("bad", [[], None, "x", {}, {"state": "idle"}, {**GOOD_STATUS, "extra": 1},
                                 {**GOOD_STATUS, "t": "evil"}])
def test_status_must_be_a_dict_with_exactly_the_spec_keys(bad):
    with pytest.raises(ValueError):
        make().set_status(bad)


def test_status_accepts_int_pic_and_the_limits():
    s = make()
    s.set_status({**GOOD_STATUS, "piC": 61, "sdk": "e" * 64, "mode": "m" * 32, "moves": 0, "refused": 0})
    doc = json.loads(s.poll(T0)[-1][1:])
    assert doc["piC"] == 61 and doc["sdk"] == "e" * 64


def test_status_datagram_is_lamp_telemetry_type_4():
    d = make().status_datagram(GOOD_STATUS)
    assert d[0] == 4 and json.loads(d[1:].decode()) == {"t": "lamp", **GOOD_STATUS}
    with pytest.raises(ValueError):
        make().status_datagram({**GOOD_STATUS, "state": "nope"})


def test_config_validation():
    for kw in ({"hello_interval_ns": -1}, {"sync_timeout_ns": True}, {"conductor_lost_ns": 0},
               {"conductor_lost_ns": 1.5}, {"hello_silence_ns": 0}, {"max_outstanding": 0}):
        with pytest.raises(ValueError):
            Session("x", **kw)
    for gone in ("keepalive_ns", "mode_max_age_ns", "sync_interval_fast_ns", "sync_interval_slow_ns"):
        with pytest.raises(TypeError):
            Session("x", **{gone: 1})
    s = make()
    for bad in (-1, True, 1.0, None, 2**63):
        with pytest.raises(ValueError):
            s.poll(bad)
        with pytest.raises(ValueError):
            s.on_datagram(b"\x03", bad)
        with pytest.raises(ValueError):
            s.current_mode(bad)


def test_bass_policy_is_opted_in_explicitly():
    assert make().normalizer.bass_time_policy == "presentation"


def test_stats_is_a_plain_dict_of_ints():
    st = make().stats()
    assert type(st) is dict and all(type(k) is str and type(v) is int for k, v in st.items())
    for key in ("unknown_type", "parse_errors", "bad_control", "bad_mode", "stale_mode", "not_synced"):
        assert st[key] == 0


# ---------------------------------------------------------------------------------------- never raises
def test_unknown_type_bytes_are_counted_and_ignored():
    s = make()
    known = {1, 2, 3, 4, 5, 12, 13}
    audio = {6, 7, 8, 9, 10}
    n = 0
    for t in range(256):
        if t in known or t in audio:
            continue
        assert feed(s, bytes([t]) + b"\x00" * 40, T0) == []
        n += 1
    assert s.stats()["unknown_type"] == n == 244 and s.stats()["parse_errors"] == 0
    assert s.stats()["audio_ignored"] == 0


def test_empty_and_non_bytes_never_raise():
    s = make()
    for bad in (b"", None, "abc", 5, bytearray(b"\x03"), memoryview(b"\x03"), [1, 2]):
        assert feed(s, bad, T0) == []
    assert s.stats()["parse_errors"] == 7


def test_fuzz_random_datagrams_never_raise():
    rig = synced_rig()
    before = rig.s.stats()["datagrams"]
    rng = random.Random(1234)
    for i in range(20000):
        n = rng.choice([0, 1, 2, 3, 5, 6, 8, 19, 25, 26, 27, 40, 100, 300])
        data = bytes(rng.randrange(256) for _ in range(n))
        if data and rng.random() < 0.6:
            data = bytes([rng.choice([1, 2, 3, 4, 5, 12, 13, 14, 200])]) + data[1:]
        out = rig.s.on_datagram(data, rig.now + i)
        assert isinstance(out, list)
    assert rig.s.stats()["datagrams"] - before == 20000


def test_fuzz_mutations_of_valid_datagrams_never_raise():
    rig = synced_rig()
    t = rig.now
    corpus = [
        rig.c.event_at(t + LEAD, seq=3, kind=1), assign_bytes(9), struct.pack("<BH", 1, 4),
        resp_bytes(1, t + OFFSET, t + OFFSET), bass_bytes(4, t + OFFSET, 20, [1, 2, 3]),
        control_bytes({"session": 9, "lat": 300, "lamp": valid_lamp("follow", True, 77)}),
        bytes([4]) + b'{"t":"hi","role":"lamp","name":"x","v":1}',
    ]
    rng = random.Random(99)
    for i in range(2000):
        b = bytearray(rng.choice(corpus))
        op = rng.randrange(4)
        if op == 0 and b:
            b[rng.randrange(len(b))] = rng.randrange(256)
        elif op == 1 and b:
            del b[rng.randrange(len(b)):]
        elif op == 2:
            b += bytes(rng.randrange(256) for _ in range(rng.randrange(1, 20)))
        else:
            for _ in range(3):
                if b:
                    b[rng.randrange(len(b))] ^= 1 << rng.randrange(8)
        out = rig.s.on_datagram(bytes(b), t + i)
        assert all(isinstance(o, (EventOut, BassOut, ModeOut, SessionOut)) for o in out)


def test_fuzz_random_json_in_control_never_raises():
    s = make()
    rng = random.Random(5)
    atoms = ["1", "-1", "true", "null", '"follow"', '"x"', "[]", "{}", "1.5", "NaN", "2e999"]
    for i in range(2000):
        txt = "{" + ",".join('"%s": %s' % (rng.choice(["session", "lamp", "mode", "lights", "gen", "lat"]),
                                          rng.choice(atoms)) for _ in range(rng.randrange(4))) + "}"
        assert isinstance(s.on_datagram(control_bytes(txt), T0 + i), list)
    assert s.current_mode(T0 + 3000) in ("follow", "dance", "light", "off")


def test_parse_control_json_size_cap_counts_utf8_bytes_directly():
    body = '{"p": "' + "\u00e9" * (MAX_CONTROL_BYTES // 2) + '"}'
    assert len(body) < MAX_CONTROL_BYTES < len(body.encode())
    assert parse_control_json(body) is None


def test_oversized_control_datagram_is_rejected_before_it_is_decoded(monkeypatch):
    import ftm_client
    calls = []
    real = ftm_client.decode
    monkeypatch.setattr(ftm_client, "decode", lambda d: calls.append(len(d)) or real(d))
    s = make()
    big = bytes([13]) + b'{"p": "' + b"a" * (MAX_CONTROL_BYTES + 100) + b'"}'
    assert feed(s, big, T0) == [] and s.stats()["bad_control"] == 1 and calls == []


def test_an_unexpected_exception_inside_handling_is_contained_and_counted(monkeypatch):
    import ftm_client

    def boom(_):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(ftm_client, "decode", boom)
    s = make()
    assert feed(s, bytes([3]) + bytes(25), T0) == [] and s.stats()["parse_errors"] == 1


def test_parse_error_from_decode_is_counted(monkeypatch):
    import ftm_client

    def bad(_):
        raise ftm_client.ParseError("x")

    monkeypatch.setattr(ftm_client, "decode", bad)
    s = make()
    assert feed(s, bytes([3]) + bytes(25), T0) == [] and s.stats()["parse_errors"] == 1


# ---------------------------------------------------------------------------------------- orphaned peer
def _hellos(datagrams):
    return [d for d in datagrams if d[:1] == b"\x04" and b'"hi"' in d]


def test_rehello_when_only_syncresp_keeps_arriving_after_a_conductor_restart():
    """Reported by meharsclaude (hub general seq 1065, on a real lamp): after a conductor restart the session changes
    and a peer that said hello once and then only sends SyncReq looks joined but receives nothing. The conductor still
    answers SyncReq, so 'the conductor is silent' (hello_silence) never fires. Content silence (no Assign, Control,
    event or bass) while joined must bring the hello back."""
    s = Session("lamp", jitter_source=lambda: 0)
    now = T0
    s.poll(now)
    assert s.on_datagram(assign_bytes(777), now + MS) == [SessionOut(777)]
    assert s.joined
    hello_count = 0
    for i in range(1, 40):  # 20 s at 0.5 s steps: SyncReqs are answered, nothing else ever arrives
        now = T0 + i * 500 * MS
        out = s.poll(now)
        hello_count += len(_hellos(out))
        for d in out:
            if d[:1] == b"\x01":
                rid = struct.unpack("<BH", d)[1]
                s.on_datagram(resp_bytes(rid, 5 * S + now, 5 * S + now + MS), now + 2 * MS)
    assert hello_count >= 3, "peer stayed silent to the conductor: an orphaned session is never re-hello'd"


def test_no_rehello_while_content_keeps_arriving():
    s = Session("lamp", jitter_source=lambda: 0)
    s.poll(T0)
    s.on_datagram(assign_bytes(777), T0 + MS)
    hellos = 0
    for i in range(1, 40):
        now = T0 + i * 500 * MS
        s.on_datagram(control_bytes({"lat": 300, "session": 777}), now)  # periodic Control counts as content
        out = s.poll(now)
        hellos += len(_hellos(out))
        for d in out:
            if d[:1] == b"\x01":
                rid = struct.unpack("<BH", d)[1]
                s.on_datagram(resp_bytes(rid, 5 * S + now, 5 * S + now + MS), now + 2 * MS)
    assert hellos == 0
