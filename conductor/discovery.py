"""Bonjour / mDNS advertisement and the manual-address fallback.

``zeroconf`` is optional and imported lazily inside ``advertise``. Everything else here
is pure and unit-tested without it.
"""
from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from . import PROTOCOL_VERSION
from .hub import DEFAULT_PORT

SERVICE_TYPE = "_feelthemusic._udp.local."
DEFAULT_INSTANCE = "feelthemusic"
SERVER_LABEL = "feelthemusic-conductor.local."

_LABEL_RE = re.compile(r"\A[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?\Z")


class DiscoveryError(ValueError):
    """Bad address or record input."""


class DiscoveryUnavailable(RuntimeError):
    """The optional zeroconf package is not installed."""


@dataclass(frozen=True)
class ServiceRecord:
    service_type: str
    name: str                      # full instance name, "<instance>._feelthemusic._udp.local."
    port: int
    server: str
    txt: Dict[str, str]


def build_service_record(port: int = DEFAULT_PORT, instance: str = DEFAULT_INSTANCE,
                         version: int = PROTOCOL_VERSION) -> ServiceRecord:
    """Pure: what we advertise. TXT carries the protocol version and the role."""
    if type(port) is not int or not 1 <= port <= 65535:
        raise DiscoveryError("port out of range")
    if (not isinstance(instance, str) or not instance or len(instance.encode("utf-8")) > 63
            or any(ord(c) < 32 or c == "\x7f" or c == "." for c in instance)):
        raise DiscoveryError("bad instance name")
    if type(version) is not int or version < 1:
        raise DiscoveryError("bad protocol version")
    return ServiceRecord(SERVICE_TYPE, instance + "." + SERVICE_TYPE, port, SERVER_LABEL,
                         {"v": str(version), "role": "conductor"})


def parse_manual_address(text: str, default_port: int = DEFAULT_PORT) -> Tuple[str, int]:
    """Parse "host", "host:port", "1.2.3.4:5", "[::1]:5" or a bare IPv6 into (host, port)."""
    if not isinstance(text, str):
        raise DiscoveryError("address must be text")
    s = text.strip()
    if not s or any(c.isspace() or ord(c) < 32 for c in s):
        raise DiscoveryError("address must not contain spaces")
    port_text: Optional[str] = None
    if s.startswith("["):
        end = s.find("]")
        if end < 0:
            raise DiscoveryError("unterminated [ in address")
        host, rest = s[1:end], s[end + 1:]
        if rest:
            if not rest.startswith(":"):
                raise DiscoveryError("junk after ]")
            port_text = rest[1:]
        _require_ip(host, 6)
    elif s.count(":") > 1:
        host = s
        _require_ip(host, 6)
    else:
        host, sep, ptext = s.partition(":")
        if sep:
            port_text = ptext
        _require_host(host)
    port = default_port
    if port_text is not None:
        if not (port_text.isascii() and port_text.isdigit()):
            raise DiscoveryError("port must be digits")
        port = int(port_text)
    if not 1 <= port <= 65535:
        raise DiscoveryError("port out of range")
    return host, port


def _require_ip(host: str, family: int) -> None:
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        raise DiscoveryError("bad IP address") from None
    if ip.version != family:
        raise DiscoveryError("bad IP address")


def _require_host(host: str) -> None:
    if not host or len(host) > 253:
        raise DiscoveryError("bad host")
    if re.fullmatch(r"[0-9.]+", host):  # looks numeric: must be a real IPv4
        _require_ip(host, 4)
        return
    for label in host.rstrip(".").split("."):
        if not _LABEL_RE.match(label):
            raise DiscoveryError("bad host label")


def local_ipv4_addresses() -> List[str]:
    """Best-effort non-loopback IPv4 addresses of this machine (no network traffic)."""
    found: List[str] = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127.") and ip not in found:
                found.append(ip)
    except OSError:
        pass
    return found


class Advertisement:
    def __init__(self, zc, info) -> None:
        self._zc, self._info = zc, info

    def close(self) -> None:
        try:
            self._zc.unregister_service(self._info)
        finally:
            self._zc.close()


def advertise(record: ServiceRecord, addresses: Optional[Sequence[str]] = None) -> Advertisement:
    """Publish ``record`` over mDNS. Needs the optional ``zeroconf`` package.

    NOT exercised by the unit tests (zeroconf is not a dependency); see PROTOCOL.md."""
    try:
        from zeroconf import ServiceInfo, Zeroconf  # lazy: optional dependency
    except ImportError:
        raise DiscoveryUnavailable(
            "mDNS advertising needs the optional 'zeroconf' package (pip install zeroconf); "
            "clients can still connect by manual address") from None
    addrs = list(addresses) if addresses else local_ipv4_addresses()
    if not addrs:
        raise DiscoveryError("no local IPv4 address to advertise; pass addresses=")
    info = ServiceInfo(record.service_type, record.name,
                       addresses=[socket.inet_aton(a) for a in addrs],
                       port=record.port, properties=dict(record.txt), server=record.server)
    zc = Zeroconf()
    zc.register_service(info)
    return Advertisement(zc, info)
