"""Thin UDP wrapper around ftm_session.Session (task #19). One ``step`` = one bounded, non-blocking pass.

The socket is injected: real code passes a non-blocking ``socket.socket`` (SOCK_DGRAM) that is used only
as a client (it is never bound to the conductor's port 47300 and never listens for peers); tests pass a
fake with ``recvfrom``/``sendto``. Nothing here has been run against the real conductor.

# Bonjour discovery is not implemented: browsing ``_feelthemusic._udp`` needs an mDNS stack (zeroconf or
# the OS resolver) which is a separate, platform-specific piece of work with its own failure modes, and a
# hard-coded pinned address keeps this layer small and testable. The conductor address is passed in.

Rules kept here: only datagrams whose source (ip, port) equals the pinned conductor address are fed to the
session (others are counted ``foreign_source`` and dropped, so a stranger on the LAN cannot inject events
or modes); all sends go to the pinned address only; the lamp never replies to a SyncReq (the session has
no path that does); one step reads at most ``max_recv_per_step`` datagrams of at most ``max_datagram``
bytes, so it cannot loop forever or buffer without bound.
"""
from __future__ import annotations

from ftm_session import Output, Session

MAX_DATAGRAM = 65535  # the largest possible UDP payload is 65507; anything bigger cannot be genuine
MAX_RECV_PER_STEP = 64

_COUNTERS = ("recv", "foreign_source", "oversize", "recv_reset", "recv_error", "sent", "send_blocked", "send_error")


class UdpClient:
    def __init__(self, session: Session, sock, peer_addr: tuple, *, max_datagram: int = MAX_DATAGRAM,
                 max_recv_per_step: int = MAX_RECV_PER_STEP) -> None:
        if (not isinstance(peer_addr, tuple) or len(peer_addr) < 2 or not isinstance(peer_addr[0], str)
                or isinstance(peer_addr[1], bool) or not isinstance(peer_addr[1], int)
                or not 1 <= peer_addr[1] <= 65535):
            raise ValueError("peer_addr must be a (numeric ip str, port) tuple")
        for name, v in (("max_datagram", max_datagram), ("max_recv_per_step", max_recv_per_step)):
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"{name} must be a positive int")
        self.session = session
        self.sock = sock
        self.peer_addr = (peer_addr[0], peer_addr[1])
        self.max_datagram = max_datagram
        self.max_recv_per_step = max_recv_per_step
        self._c = dict.fromkeys(_COUNTERS, 0)

    def stats(self) -> dict:
        out = dict(self._c)
        out["session"] = self.session.stats()
        return out

    def _is_peer(self, addr) -> bool:
        # an IPv6 source is a 4-tuple (host, port, flow, scope); compare host and port only
        return isinstance(addr, tuple) and len(addr) >= 2 and (addr[0], addr[1]) == self.peer_addr

    def _send(self, data: bytes) -> None:
        try:
            self.sock.sendto(data, self.peer_addr)
        except (BlockingIOError, InterruptedError):
            self._c["send_blocked"] += 1
        except OSError:
            self._c["send_error"] += 1
        else:
            self._c["sent"] += 1

    def set_status(self, status: dict) -> None:
        """Give the session the current lamp status (spec shape; ValueError otherwise). The session sends it
        about once a second from ``step``; there is no other telemetry."""
        self.session.set_status(status)

    def step(self, now_ns: int) -> list[Output]:
        """Drain up to max_recv_per_step datagrams into the session, then send what the session wants."""
        outputs: list[Output] = []
        for _ in range(self.max_recv_per_step):
            try:
                data, addr = self.sock.recvfrom(self.max_datagram + 1)
            except (BlockingIOError, InterruptedError):
                break
            except ConnectionResetError:  # Windows reports an ICMP port-unreachable on the next recv
                self._c["recv_reset"] += 1
                continue
            except OSError:
                self._c["recv_error"] += 1
                break
            self._c["recv"] += 1
            if not self._is_peer(addr):
                self._c["foreign_source"] += 1
                continue
            if len(data) > self.max_datagram:
                self._c["oversize"] += 1
                continue
            outputs.extend(self.session.on_datagram(bytes(data), now_ns))
        for data in self.session.poll(now_ns):
            self._send(data)
        return outputs
