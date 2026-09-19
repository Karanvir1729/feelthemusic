"""UdpClient tests with a fake socket (plus one loopback smoke test). Nothing touches the real conductor."""
import json
import socket
import struct
import time

import pytest

from ftm_session import EventOut, Session, SessionOut
from ftm_udp import MAX_DATAGRAM, MAX_RECV_PER_STEP, UdpClient

MS = 1_000_000
T0 = 10**12
PEER = ("192.168.1.50", 47300)
STRANGER = ("192.168.1.66", 47300)
OFFSET = 4_000_000_000_000


class FakeSock:
    def __init__(self, incoming=()):
        self.incoming = list(incoming)  # (data, addr) or an exception instance to raise
        self.sent = []
        self.recv_calls = 0
        self.bound = None
        self.send_exc = None
        self.bufsizes = []

    def recvfrom(self, bufsize):
        self.recv_calls += 1
        self.bufsizes.append(bufsize)
        if not self.incoming:
            raise BlockingIOError()
        item = self.incoming.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def sendto(self, data, addr):
        if self.send_exc is not None:
            raise self.send_exc
        self.sent.append((data, addr))

    def bind(self, *a):  # the client must never bind
        self.bound = a


def client(sock, **kw):
    return UdpClient(Session("lamp-1"), sock, PEER, **kw)


def assign(sid):
    return struct.pack("<BBI", 5, 0, sid)


def test_step_sends_hello_to_the_pinned_peer_only_and_never_binds():
    sock = FakeSock()
    c = client(sock)
    assert c.step(T0) == []
    assert len(sock.sent) == 1 and sock.sent[0][1] == PEER
    assert json.loads(sock.sent[0][0][1:])["t"] == "hi"
    c.step(T0 + 1 * 10**9)
    assert len(sock.sent) == 2 and all(addr == PEER for _, addr in sock.sent)
    assert sock.bound is None


def test_datagram_from_the_peer_reaches_the_session():
    sock = FakeSock([(assign(9), PEER)])
    c = client(sock)
    assert c.step(T0) == [SessionOut(9)]
    assert c.session.session_id == 9 and c.stats()["recv"] == 1


def test_foreign_source_is_counted_and_dropped_not_fed_to_the_session():
    ctl = b"\x0d" + json.dumps({"session": 5, "lamp": {"mode": "follow", "lights": True, "gen": 1}}).encode()
    sock = FakeSock([(assign(1), STRANGER), (ctl, ("192.168.1.50", 47301)), (ctl, ("10.0.0.1", 47300))])
    c = client(sock)
    assert c.step(T0) == []
    st = c.stats()
    assert st["foreign_source"] == 3 and st["session"]["datagrams"] == 0
    assert not c.session.joined and c.session.current_mode(T0 + MS) == "off"


def test_ipv6_four_tuple_source_matches_on_host_and_port():
    sock = FakeSock([(assign(3), ("fe80::1", 47300, 0, 2))])
    c = UdpClient(Session("l"), sock, ("fe80::1", 47300))
    assert c.step(T0) == [SessionOut(3)]


def test_per_step_recv_cap_bounds_the_loop_and_the_rest_waits_for_the_next_step():
    sock = FakeSock([(bytes([200]), PEER)] * 1000)
    c = client(sock)
    c.step(T0)
    assert sock.recv_calls == MAX_RECV_PER_STEP == 64
    assert c.session.stats()["unknown_type"] == 64
    c.step(T0 + MS)
    assert sock.recv_calls == 128


def test_recv_cap_is_configurable_and_counts_foreign_datagrams_too():
    sock = FakeSock([(b"x", STRANGER)] * 50)
    c = client(sock, max_recv_per_step=5)
    c.step(T0)
    assert sock.recv_calls == 5 and c.stats()["foreign_source"] == 5


def test_oversize_datagram_is_dropped_and_the_read_size_is_capped():
    sock = FakeSock([(bytes([200]) * 101, PEER), (bytes([200]) * 100, PEER)])
    c = client(sock, max_datagram=100)
    c.step(T0)
    assert set(sock.bufsizes) == {101}
    assert c.stats()["oversize"] == 1 and c.session.stats()["unknown_type"] == 1
    assert MAX_DATAGRAM == 65535


def test_recv_errors_are_counted_and_do_not_escape():
    sock = FakeSock([ConnectionResetError(), (assign(2), PEER)])
    c = client(sock)
    assert c.step(T0) == [SessionOut(2)] and c.stats()["recv_reset"] == 1
    c2 = client(FakeSock([OSError("boom"), (assign(2), PEER)]))
    assert c2.step(T0) == [] and c2.stats()["recv_error"] == 1


def test_send_errors_are_counted_and_do_not_escape():
    sock = FakeSock()
    sock.send_exc = BlockingIOError()
    c = client(sock)
    c.step(T0)
    assert c.stats()["send_blocked"] == 1 and c.stats()["sent"] == 0
    sock.send_exc = OSError("down")
    c.step(T0 + 1 * 10**9)
    assert c.stats()["send_error"] == 1


def test_the_lamp_never_replies_to_a_syncreq_from_the_peer():
    sock = FakeSock([(struct.pack("<BH", 1, 77), PEER)])
    c = client(sock)
    for i in range(100):
        c.step(T0 + i * 100 * MS)
    assert sock.sent and not any(d[0] == 2 for d, _ in sock.sent)
    assert c.session.stats()["unexpected_packet"] == 1


def test_end_to_end_events_through_the_wrapper():
    sock = FakeSock()
    c = client(sock)
    c.step(T0)
    sock.incoming.append((assign(7), PEER))
    now = T0 + 5 * MS
    for _ in range(3):
        c.step(now)
        req = [d for d, _ in sock.sent if d[0] == 1][-1]
        rid = struct.unpack("<H", req[1:3])[0]
        t1 = now + OFFSET + 2 * MS
        sock.incoming.append((struct.pack("<BHQQ", 2, rid, t1, t1), PEER))
        now += 4 * MS
        c.step(now)  # the response is read here, 4 ms after the request (symmetric 2 ms legs)
        now += 96 * MS
    ev = struct.pack("<BIBBBBHHBQI", 3, 1, 1, 0, 255, 0, 10, 100, 255, now + OFFSET + 250 * MS, 0)
    sock.incoming.append((ev, PEER))
    out = c.step(now)
    assert len(out) == 1 and isinstance(out[0], EventOut)
    assert abs(out[0].event.due_ns - (now + 250 * MS)) <= 5 * MS


def test_set_status_is_sent_by_the_session_to_the_peer_about_once_a_second():
    sock = FakeSock()
    c = client(sock)
    c.set_status({"state": "idle", "mode": "follow", "locked": False, "piC": 1.5, "sdk": "ok", "moves": 0,
                  "refused": 0})
    c.step(T0)
    lamp = [(d, a) for d, a in sock.sent if d[0] == 4 and json.loads(d[1:])["t"] == "lamp"]
    assert len(lamp) == 1 and lamp[0][1] == PEER
    c.step(T0 + 500 * MS)
    assert len([1 for d, _ in sock.sent if d[0] == 4 and json.loads(d[1:])["t"] == "lamp"]) == 1
    c.step(T0 + 1 * 10**9)
    assert len([1 for d, _ in sock.sent if d[0] == 4 and json.loads(d[1:])["t"] == "lamp"]) == 2
    with pytest.raises(ValueError):
        c.set_status({"state": "bogus"})
    assert not hasattr(c, "send_status")


@pytest.mark.parametrize("addr", [None, "1.2.3.4", ("1.2.3.4",), (1, 2), ("1.2.3.4", 0), ("1.2.3.4", 70000),
                                  ("1.2.3.4", True), ["1.2.3.4", 47300]])
def test_bad_peer_addr_is_rejected(addr):
    with pytest.raises(ValueError):
        UdpClient(Session("l"), FakeSock(), addr)


def test_bad_caps_are_rejected():
    for kw in ({"max_datagram": 0}, {"max_recv_per_step": 0}, {"max_recv_per_step": True}):
        with pytest.raises(ValueError):
            UdpClient(Session("l"), FakeSock(), PEER, **kw)


def test_loopback_smoke_with_a_real_nonblocking_socket():
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    server.settimeout(2)
    cli = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cli.setblocking(False)
    try:
        c = UdpClient(Session("lamp-1"), cli, server.getsockname())
        assert c.step(T0) == []
        data, src = server.recvfrom(2048)
        assert data[0] == 4 and json.loads(data[1:])["t"] == "hi"
        server.sendto(assign(31), src)
        out = []
        for _ in range(300):
            out = c.step(T0 + 1 * 10**9)
            if out:
                break
            time.sleep(0.01)
        assert out == [SessionOut(31)]
    finally:
        server.close()
        cli.close()
