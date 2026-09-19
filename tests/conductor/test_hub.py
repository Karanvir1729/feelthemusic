import socket
import threading
import time

import pytest

from conductor import discovery, wire
from conductor.clock import FakeClock, OffsetEstimator
from conductor.hub import DEFAULT_PORT, Hub


class StepClock(FakeClock):
    """Fake clock that ticks 500 ns on every read so t2 > t1 is observable."""

    def __call__(self):
        with_lock = self._now
        self._now += 500
        return with_lock


def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout  # test-harness timeout only, not protocol time
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.005)
    return False


class Client:
    def __init__(self, hub):
        self.hub_addr = hub.address
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.settimeout(3.0)

    def send(self, data):
        self.sock.sendto(data if isinstance(data, bytes) else wire.encode(data), self.hub_addr)

    def recv(self, timeout=3.0):
        self.sock.settimeout(timeout)
        try:
            return wire.decode(self.sock.recvfrom(4096)[0])
        except socket.timeout:
            return None

    def hello(self, cid="c1"):
        self.send(wire.Hello(cid, "haptic"))
        return self.recv()

    def close(self):
        self.sock.close()


@pytest.fixture
def hub():
    clock = StepClock(5_000_000_000)
    h = Hub(clock=clock, port=0, hub_id="hub-t", max_clients=3, client_ttl_ns=10_000)
    h.clock = clock
    h.start()
    yield h
    h.stop()


def test_default_port_constant():
    assert DEFAULT_PORT == 47300


def test_hello_registers_and_welcomes(hub):
    c = Client(hub)
    assert c.hello("c1") == wire.Welcome("c1", "hub-t")
    assert hub.client_count() == 1 and hub.counters()["hellos"] == 1
    c.close()


def test_probe_reply_stamps_t1_t2_from_injected_clock(hub):
    c = Client(hub)
    c.hello()
    c.send(wire.Probe(9, 123))
    r = c.recv()
    assert isinstance(r, wire.ProbeReply)
    assert (r.id, r.t0) == (9, 123)
    assert r.t1 >= 5_000_000_000 and r.t2 > r.t1
    assert hub.counters()["probes_answered"] == 1
    c.close()


def test_estimator_over_loopback_with_hub_clock_offset(hub):
    c = Client(hub)
    c.hello()
    est = OffsetEstimator()
    client_clock = FakeClock(1_000)  # a different clock domain: offset should be ~ +5e9
    for i in range(5):
        t0 = client_clock.advance(1_000)
        c.send(wire.Probe(i, t0))
        r = c.recv()
        t3 = client_clock.advance(2_000)
        assert est.add(r.t0, r.t1, r.t2, t3)
    e = est.estimate()
    assert abs(e.offset_ns - (5_000_000_000 - 2_000)) < 100_000
    c.close()


def test_no_reply_to_clients_without_hello(hub):
    c = Client(hub)
    c.send(wire.Probe(1, 1))
    assert c.recv(0.3) is None
    c.send(wire.Bye("c1"))
    assert c.recv(0.3) is None
    assert wait_for(lambda: hub.counters()["rx_unregistered"] == 1)
    assert hub.counters()["probes_answered"] == 0
    c.close()


def test_invalid_hello_is_not_registered_and_gets_nothing(hub):
    c = Client(hub)
    c.send(b'{"v":1,"t":"hello","cid":"a b","role":"x"}')
    assert c.recv(0.3) is None
    assert hub.client_count() == 0
    c.close()


def test_hostile_datagrams_are_counted_and_hub_survives(hub):
    c = Client(hub)
    c.send(b"\xff\xfe\x00garbage")
    c.send(b"[" * 1000)
    c.send(b"x" * 5000)                       # oversize
    c.send(b'{"v":2,"t":"probe","id":1,"t0":1}')
    c.send(b'{"v":1,"t":"probe","id":1,"t0":1.5}')
    c.send(wire.Welcome("a", "b"))            # valid but hub->client type
    assert wait_for(lambda: hub.counters()["rx_datagrams"] == 6)
    n = hub.counters()
    assert n["rx_oversize"] == 1 and n["rx_invalid"] == 4 and n["rx_unexpected"] == 1
    assert n["internal_errors"] == 0
    assert c.hello("still-ok") == wire.Welcome("still-ok", "hub-t")  # still serving
    c.close()


def test_client_table_is_bounded_and_refusals_are_silent(hub):
    cs = [Client(hub) for _ in range(4)]
    for i, c in enumerate(cs[:3]):
        assert c.hello("c%d" % i) is not None
    assert cs[3].hello("c3") is None
    assert hub.client_count() == 3 and hub.counters()["table_full"] == 1
    cs[3].send(wire.Probe(1, 1))
    assert cs[3].recv(0.3) is None
    for c in cs:
        c.close()


def test_stale_clients_are_evicted_to_make_room(hub):
    cs = [Client(hub) for _ in range(4)]
    for i, c in enumerate(cs[:3]):
        c.hello("c%d" % i)
    hub.clock.advance(1_000_000)   # far beyond the 10 microsecond TTL
    assert cs[3].hello("c3") == wire.Welcome("c3", "hub-t")
    assert hub.counters()["evicted"] == 3 and hub.client_count() == 1
    for c in cs:
        c.close()


def test_re_hello_from_same_address_does_not_consume_a_slot(hub):
    c = Client(hub)
    for _ in range(5):
        assert c.hello("c1") is not None
    assert hub.client_count() == 1
    c.close()


def test_bye_removes_only_the_matching_registered_client(hub):
    a, b = Client(hub), Client(hub)
    a.hello("a")
    b.hello("b")
    b.send(wire.Bye("a"))                     # wrong cid for b's address
    a.send(wire.Bye("a"))
    assert wait_for(lambda: hub.client_count() == 1)
    assert hub.counters()["byes"] == 1
    a.close()
    b.close()


def test_broadcast_reaches_registered_clients_only(hub):
    a, b, stranger = Client(hub), Client(hub), Client(hub)
    a.hello("a")
    b.hello("b")
    ev = wire.Event(1, "kick", 9_999, {"amp": 5})
    assert hub.broadcast(ev) == 2
    assert a.recv() == ev and b.recv() == ev
    assert stranger.recv(0.3) is None
    assert hub.counters()["tx_events"] == 2
    for c in (a, b, stranger):
        c.close()


def test_broadcast_survives_a_dead_client_and_rejects_bad_events(hub):
    a, dead = Client(hub), Client(hub)
    a.hello("a")
    dead.hello("d")
    dead.close()
    ev = wire.Event(1, "k", 1, {})
    assert hub.broadcast(ev) >= 1
    assert a.recv() == ev
    with pytest.raises(wire.WireError):
        hub.broadcast(wire.Event(1, "k", 1.5, {}))
    assert a.recv(0.3) is None
    a.close()


def test_client_events_to_hub_are_rejected_not_relayed(hub):
    a, b = Client(hub), Client(hub)
    a.hello("a")
    b.hello("b")
    a.send(wire.Event(1, "spoof", 1, {}))
    assert b.recv(0.3) is None
    assert wait_for(lambda: hub.counters()["rx_unexpected"] == 1)
    a.close()
    b.close()


def test_concurrent_clients_do_not_interfere(hub):
    results = []

    def run(i):
        c = Client(hub)
        ok = c.hello("c%d" % i) is not None
        if ok:
            c.send(wire.Probe(i, i))
            r = c.recv()
            ok = isinstance(r, wire.ProbeReply) and r.id == i
        results.append(ok)
        c.close()
    ts = [threading.Thread(target=run, args=(i,)) for i in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert results == [True] * 3


# ---- discovery (pure parts; zeroconf itself is NOT tested) ---------------------

def test_service_record():
    r = discovery.build_service_record()
    assert r.service_type == "_feelmusic._udp.local."
    assert r.name == "feelthemusic._feelmusic._udp.local."
    assert r.port == 47300
    assert r.txt == {"v": "1", "role": "conductor"}
    assert discovery.build_service_record(port=5000, instance="room 2").port == 5000


@pytest.mark.parametrize("kw", [{"port": 0}, {"port": 65536}, {"port": True}, {"port": "1"},
                                {"instance": ""}, {"instance": "a.b"}, {"instance": "x" * 64},
                                {"instance": "a\nb"}, {"version": 0}])
def test_service_record_rejects_bad_input(kw):
    with pytest.raises(discovery.DiscoveryError):
        discovery.build_service_record(**kw)


@pytest.mark.parametrize("text,expected", [
    ("192.168.1.5", ("192.168.1.5", 47300)),
    ("192.168.1.5:5000", ("192.168.1.5", 5000)),
    ("conductor.local", ("conductor.local", 47300)),
    ("conductor.local:1", ("conductor.local", 1)),
    ("  10.0.0.1:65535 ", ("10.0.0.1", 65535)),
    ("[::1]:5000", ("::1", 5000)),
    ("[::1]", ("::1", 47300)),
    ("fe80::1", ("fe80::1", 47300)),
])
def test_manual_address_ok(text, expected):
    assert discovery.parse_manual_address(text) == expected


@pytest.mark.parametrize("text", [
    "", "   ", "host name", "a b:5", "host:", "host:0", "host:65536", "host:-1", "host:80x",
    "host:٥", "1.2.3", "300.1.1.1", "1.2.3.4.5", "-bad.local", "bad_.local", "a..b",
    "[::1", "[::1]x", "[1.2.3.4]:5", "::zz", "ho\nst", "ho\tst", None, 5,
])
def test_manual_address_rejects(text):
    with pytest.raises(discovery.DiscoveryError):
        discovery.parse_manual_address(text)


def test_custom_default_port():
    assert discovery.parse_manual_address("h", default_port=1234) == ("h", 1234)


def test_advertise_without_zeroconf_raises_clear_error(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "zeroconf":
            raise ImportError("no zeroconf")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(discovery.DiscoveryUnavailable, match="zeroconf"):
        discovery.advertise(discovery.build_service_record(), addresses=["10.0.0.2"])


def test_hub_keeps_serving_after_an_unexpected_handler_error(hub, monkeypatch):
    real = hub._handle
    calls = {"n": 0}

    def flaky(sock, data, addr, t1):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated bug")
        return real(sock, data, addr, t1)
    monkeypatch.setattr(hub, "_handle", flaky)
    c = Client(hub)
    c.send(b"first datagram triggers the bug")
    assert wait_for(lambda: hub.counters()["internal_errors"] == 1)
    assert c.hello("after") == wire.Welcome("after", "hub-t")
    c.close()
