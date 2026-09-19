"""Tests for ftm_session.Session against a FakeConductor built from the REPORTED join spec.

Nothing here is checked against the real conductor. The FakeConductor encodes Tempo's report (see the
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


def assign_bytes(session, u8=0):
    return struct.pack("<BBI", 5, u8, session)


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


def make(**kw):
    kw.setdefault("sync_interval_fast_ns", 100 * MS)
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
    assert json.loads(out[0][1:].decode("utf-8")) == {"t": "hi", "role": "lamp", "name": "lamp-1", "v": 1}


def test_hello_repeats_until_assign_or_control_then_stops():
    s = make()
    assert s.poll(T0)[0][0] == 4
    assert [d for d in s.poll(T0 + 500 * MS) if d[0] == 4 and b'"hi"' in d] == []
    assert any(b'"hi"' in d for d in s.poll(T0 + 1 * S))  # default hello interval is 1 s
    feed(s, assign_bytes(9), T0 + 1 * S + MS)
    later = [d for k in range(1, 25) for d in s.poll(T0 + S + k * 100 * MS)]  # < 3 s: no sync timeout yet
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
    rig.run(3 * S)
    s = rig.s
    assert cond.hellos == 1 and s.joined and s.session_id == 777
    assert [o.session_id for o in rig.kinds(SessionOut)] == [777]
    est = s.estimator.estimate(rig.now)
    assert est is not None and abs(est.offset_ns - OFFSET) <= est.bound_ns
    assert abs(est.offset_ns - OFFSET) <= TOL
    times = [t for _, t in cond.syncreqs]
    fast = [b - a for a, b in zip(times, times[1:]) if b - T0 < 800 * MS]
    assert fast and all(g == 100 * MS for g in fast)  # fast interval while not ready
    slow = [b - a for a, b in zip(times, times[1:]) if a - T0 > 1500 * MS]
    assert not slow or all(g >= 5 * S for g in slow)  # slow (default 5 s) once ready


def test_slow_interval_is_used_once_ready_and_fast_before():
    s = make(sync_interval_slow_ns=2 * S)  # fast = 100 ms
    s.poll(T0)
    feed(s, assign_bytes(1), T0 + MS)
    now = T0 + 10 * MS
    for _ in range(3):
        d = s.poll(now)
        assert [x[0] for x in d] == [1]  # one SyncReq per fast interval while not ready
        rid = struct.unpack("<H", d[0][1:3])[0]
        t1 = now + OFFSET + 2 * MS
        feed(s, resp_bytes(rid, t1, t1), now + 4 * MS)
        now += 100 * MS
    assert s.estimator.estimate(now) is not None
    last = now - 100 * MS
    assert all(x[0] != 1 for x in s.poll(last + 1 * S))  # ready: the slow 2 s interval is not reached
    assert [x[0] for x in s.poll(last + 2 * S)] == [1]


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
    s = joined(max_outstanding=4, id_source=lambda: next(ids), sync_interval_fast_ns=MS,
               sync_timeout_ns=10**15)
    sent = []
    for i in range(10):
        sent.append(one_request(s, T0 + 10 * MS + i * 2 * MS))
        assert s.outstanding <= 4
    assert s.outstanding == 4 and s.stats()["sync_evicted"] == 6
    now = T0 + 100 * MS
    feed(s, resp_bytes(sent[0], now + OFFSET, now + OFFSET), now)  # evicted, so unknown
    assert s.stats()["sync_resp_unknown"] == 1 and s.stats()["sync_samples"] == 0
    feed(s, resp_bytes(sent[-1], now + OFFSET, now + OFFSET), now + MS)
    assert s.stats()["sync_samples"] == 1


def test_id_collision_with_an_outstanding_id_is_never_reused():
    s = joined(id_source=lambda: 7, sync_interval_fast_ns=MS)
    assert one_request(s, T0 + 10 * MS) == 7
    assert [d for d in s.poll(T0 + 20 * MS) if d[0] == 1] == []
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
    s = joined(sync_interval_fast_ns=100 * MS)
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


def test_mode_returns_to_off_after_the_max_age_and_only_then():
    s = make(mode_max_age_ns=10 * S)
    feed(s, control_bytes({"session": 1, "lamp": valid_lamp("dance", True, 3)}), T0)
    assert s.current_mode(T0 + 10 * S) == "dance"
    assert s.current_mode(T0 + 10 * S + 1) == "off" and s.current_lights(T0 + 10 * S + 1) is False
    feed(s, control_bytes({"lamp": valid_lamp("light", False, 4)}), T0 + 20 * S)
    assert s.current_mode(T0 + 21 * S) == "light"  # a fresh gen revives it


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


def test_gen_may_not_go_backwards_or_repeat():
    s = make()
    feed(s, control_bytes({"session": 1, "lamp": valid_lamp("follow", True, 10)}), T0)
    for gen in (9, 10, 0):
        out = feed(s, control_bytes({"lamp": valid_lamp("dance", False, gen)}), T0 + MS)
        assert out == []
    assert s.stats()["stale_mode"] == 3
    assert s.current_mode(T0 + 2 * MS) == "follow" and s.current_lights(T0 + 2 * MS) is True
    assert feed(s, control_bytes({"lamp": valid_lamp("dance", False, 11)}), T0 + 3 * MS) == [ModeOut("dance", False, 11)]


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


def test_peer_drop_keepalive_default_interval_never_leaves_6s_of_silence():
    for scenario in ("silent_conductor", "joined_and_synced"):
        cond = FakeConductor(answer_hello=(scenario == "joined_and_synced"),
                             answer_sync=(scenario == "joined_and_synced"))
        rig = Rig(Session("lamp-1"), cond)  # every default
        rig.run(60 * S, tick=50 * MS)
        times = [t for t, _ in rig.sent]
        gaps = [b - a for a, b in zip(times, times[1:])]
        assert times and max(gaps) < 3 * S, (scenario, max(gaps))
        if scenario == "joined_and_synced":
            assert rig.s.stats()["sent_keepalive"] > 0  # the 5 s slow sync alone would leave gaps
    assert times[0] == T0


def test_keepalive_is_sent_only_when_nothing_else_went_out():
    s = joined(sync_interval_slow_ns=100 * S, sync_interval_fast_ns=100 * S)
    now = T0 + 10 * MS
    one_request(s, now)  # the only sync for a long time
    assert s.poll(now + 1 * S) == []
    ka = s.poll(now + 2 * S)
    assert len(ka) == 1 and ka[0][0] == 4 and json.loads(ka[0][1:])["t"] == "ka"
    assert s.poll(now + 2 * S + MS) == []
    assert len(s.poll(now + 4 * S)) == 1


def test_keepalive_interval_is_configurable():
    s = joined(keepalive_ns=300 * MS, sync_interval_slow_ns=100 * S, sync_interval_fast_ns=100 * S)
    now = T0 + 10 * MS
    one_request(s, now)
    assert s.poll(now + 200 * MS) == [] and len(s.poll(now + 300 * MS)) == 1


def test_status_datagram_is_lamp_telemetry_and_is_capped():
    s = make()
    st = {"state": "idle", "mode": "follow", "locked": True, "piC": 12.5, "sdk": "ok", "moves": 3, "refused": 1}
    d = s.status_datagram(st, T0)
    assert d[0] == 4
    assert json.loads(d[1:].decode()) == {"t": "lamp", **st}
    assert json.loads(s.status_datagram({"t": "evil", "sdk": "x"}, T0)[1:])["t"] == "lamp"
    with pytest.raises(ValueError):
        s.status_datagram({"sdk": "e" * 5000}, T0)
    with pytest.raises(ValueError):
        s.status_datagram({"piC": float("nan")}, T0)
    with pytest.raises(ValueError):
        s.status_datagram([], T0)


def test_config_validation():
    for kw in ({"keepalive_ns": 0}, {"hello_interval_ns": -1}, {"sync_timeout_ns": True},
               {"mode_max_age_ns": 1.5}, {"max_outstanding": 0}):
        with pytest.raises(ValueError):
            Session("x", **kw)
    for name in ("", None, "x" * 65):
        with pytest.raises(ValueError):
            Session(name)
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
    n = 0
    for t in range(256):
        if t in known:
            continue
        assert feed(s, bytes([t]) + b"\x00" * 40, T0) == []
        n += 1
    assert s.stats()["unknown_type"] == n == 249 and s.stats()["parse_errors"] == 0


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
