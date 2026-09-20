"""UDP hub: answers clock probes and fans events out to registered clients.

Plain socket + one thread. It schedules nothing: the caller stamps ``pts`` and calls
``broadcast``. Hostile input is dropped and counted; a client that has not sent a valid
hello never receives a datagram.
"""
from __future__ import annotations

import socket
import threading
from typing import Dict, Optional, Tuple

from . import wire
from .clock import ClockSource, monotonic_ns

DEFAULT_PORT = 47300
DEFAULT_MAX_CLIENTS = 64
DEFAULT_CLIENT_TTL_NS = 60_000_000_000
_RECV_BUFFER = 65535  # big enough to see (and count) oversize datagrams instead of erroring

Addr = Tuple[str, int]

_COUNTERS = (
    "rx_datagrams", "rx_oversize", "rx_invalid", "rx_unregistered", "rx_unexpected",
    "hellos", "byes", "probes_answered", "table_full", "evicted",
    "tx_events", "tx_errors", "internal_errors",
)


class Hub:
    def __init__(self, clock: ClockSource = monotonic_ns, host: str = "127.0.0.1",
                 port: int = DEFAULT_PORT, hub_id: str = "hub",
                 max_clients: int = DEFAULT_MAX_CLIENTS,
                 client_ttl_ns: int = DEFAULT_CLIENT_TTL_NS) -> None:
        self._clock = clock
        self._host = host
        self._port = port
        self._hub_id = hub_id
        self._max_clients = max_clients
        self._ttl = client_ttl_ns
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._clients: Dict[Addr, Tuple[str, int]] = {}  # addr -> (cid, last_seen_ns)
        self._counters = {name: 0 for name in _COUNTERS}

    # -- lifecycle --------------------------------------------------------------
    def start(self) -> Addr:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind((self._host, self._port))
            sock.settimeout(0.05)
            if hasattr(socket, "SIO_UDP_CONNRESET"):  # Windows: ICMP unreachable must not kill recv
                try:
                    sock.ioctl(socket.SIO_UDP_CONNRESET, False)
                except OSError:
                    pass
        except OSError:
            sock.close()
            raise
        self._sock = sock
        self._stop.clear()
        self._thread = threading.Thread(target=self._serve, name="conductor-hub", daemon=True)
        self._thread.start()
        return sock.getsockname()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._sock is not None:
            self._sock.close()
        self._thread = None
        self._sock = None

    def __enter__(self) -> "Hub":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    @property
    def address(self) -> Addr:
        assert self._sock is not None, "hub not started"
        return self._sock.getsockname()

    # -- observability ----------------------------------------------------------
    def counters(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._counters)

    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)

    def _bump(self, name: str) -> None:
        with self._lock:
            self._counters[name] += 1

    # -- serving ----------------------------------------------------------------
    def _serve(self) -> None:
        sock = self._sock
        while not self._stop.is_set():
            try:
                data, addr = sock.recvfrom(_RECV_BUFFER)
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                continue
            t1 = self._clock()  # stamped before any parsing: the honest receive time
            try:
                self._handle(sock, data, addr, t1)
            except Exception:  # never let one datagram take the hub down
                self._bump("internal_errors")

    def _handle(self, sock: socket.socket, data: bytes, addr: Addr, t1: int) -> None:
        self._bump("rx_datagrams")
        if len(data) > wire.MAX_DATAGRAM_BYTES:
            self._bump("rx_oversize")
            return
        try:
            msg = wire.decode(data)
        except wire.WireError:
            self._bump("rx_invalid")
            return
        if isinstance(msg, wire.Hello):
            self._on_hello(sock, msg, addr, t1)
        elif isinstance(msg, wire.Probe):
            self._on_probe(sock, msg, addr, t1)
        elif isinstance(msg, wire.Bye):
            with self._lock:
                entry = self._clients.get(addr)
                if entry is not None and entry[0] == msg.cid:
                    del self._clients[addr]
                    self._counters["byes"] += 1
        else:
            self._bump("rx_unexpected")

    def _on_hello(self, sock: socket.socket, msg: wire.Hello, addr: Addr, now: int) -> None:
        with self._lock:
            if addr not in self._clients and len(self._clients) >= self._max_clients:
                stale = [a for a, (_, seen) in self._clients.items() if now - seen > self._ttl]
                for a in stale:
                    del self._clients[a]
                self._counters["evicted"] += len(stale)
                if len(self._clients) >= self._max_clients:
                    self._counters["table_full"] += 1
                    return
            self._clients[addr] = (msg.cid, now)
            self._counters["hellos"] += 1
        self._send(sock, addr, wire.Welcome(cid=msg.cid, hub=self._hub_id))

    def _on_probe(self, sock: socket.socket, msg: wire.Probe, addr: Addr, t1: int) -> None:
        with self._lock:
            entry = self._clients.get(addr)
            if entry is None:
                self._counters["rx_unregistered"] += 1
                return
            self._clients[addr] = (entry[0], t1)
        reply = wire.ProbeReply(id=msg.id, t0=msg.t0, t1=t1, t2=self._clock())
        if self._send(sock, addr, reply):
            self._bump("probes_answered")

    def _send(self, sock: socket.socket, addr: Addr, msg) -> bool:
        try:
            sock.sendto(wire.encode(msg), addr)
            return True
        except (OSError, wire.WireError):
            self._bump("tx_errors")
            return False

    # -- outbound events --------------------------------------------------------
    def broadcast(self, event: wire.Event) -> int:
        """Send one event to every registered client. Returns how many sends succeeded.

        Raises WireError only if the event itself is invalid (nothing is sent then)."""
        data = wire.encode(event)
        with self._lock:
            targets = list(self._clients)
        sock = self._sock
        assert sock is not None, "hub not started"
        sent = 0
        for addr in targets:
            try:
                sock.sendto(data, addr)
                sent += 1
            except OSError:
                self._bump("tx_errors")
        with self._lock:
            self._counters["tx_events"] += sent
        return sent
