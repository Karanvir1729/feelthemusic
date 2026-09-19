#!/usr/bin/env python3
"""Find the Feel the Music conductor on the LAN with plain mDNS/DNS-SD (no zeroconf, no avahi).

The conductor advertises `_feelthemusic._udp.local` (port 47300). The venue hotspot may hand the
Mac a different address every day, so the lamp must not depend on a fixed IP. Three unicast-reply
("QU") queries: PTR for the service -> instance, SRV for the instance -> port + host, A for the host.
Responders usually put SRV/A in the additional section of the first reply, so one round trip is
typical. The last good answer is cached so a boot without the conductor still has a sensible target.

    python3 ftm_discover.py            # prints ip:port, exit 1 if nothing found and nothing cached
    from ftm_discover import discover  # -> (ip, port) or None
"""
from __future__ import annotations

import json, os, socket, struct, sys, time

SERVICE = "_feelthemusic._udp.local"
MDNS = ("224.0.0.251", 5353)
CACHE = os.path.expanduser("~/.cache/ftm-conductor.json")
PTR, A, TXT, SRV = 12, 1, 16, 33


def _qname(name: str) -> bytes:
    return b"".join(bytes([len(p)]) + p.encode() for p in name.rstrip(".").split(".")) + b"\0"


def _query(name: str, qtype: int) -> bytes:
    return struct.pack(">HHHHHH", 0, 0, 1, 0, 0, 0) + _qname(name) + struct.pack(">HH", qtype, 1 | 0x8000)


def _name(buf: bytes, off: int) -> tuple[str, int]:
    parts, jumped, end = [], False, off
    for _ in range(64):
        n = buf[off]
        if n == 0:
            off += 1; break
        if n & 0xC0 == 0xC0:
            ptr = struct.unpack(">H", buf[off:off + 2])[0] & 0x3FFF
            if not jumped: end = off + 2
            off, jumped = ptr, True; continue
        parts.append(buf[off + 1:off + 1 + n].decode(errors="replace")); off += 1 + n
    if not jumped: end = off
    return ".".join(parts), end


def _records(buf: bytes):
    qd, an, ns, ar = struct.unpack(">HHHH", buf[4:12])
    off = 12
    for _ in range(qd):
        _, off = _name(buf, off); off += 4
    for _ in range(an + ns + ar):
        name, off = _name(buf, off)
        rtype, rclass, ttl, rdlen = struct.unpack(">HHIH", buf[off:off + 10]); off += 10
        rdata = buf[off:off + rdlen]
        if rtype == PTR:   yield name, rtype, _name(buf, off)[0]
        elif rtype == SRV: yield name, rtype, (struct.unpack(">HHH", rdata[:6]), _name(buf, off + 6)[0])
        elif rtype == A:   yield name, rtype, socket.inet_ntoa(rdata[:4])
        off += rdlen


def discover(timeout: float = 2.0, attempts: int = 3) -> tuple[str, int] | None:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
    s.bind(("", 0)); s.settimeout(0.4)
    instance = host = ip = None; port = 47300
    try:
        for _ in range(attempts):
            if instance is None: s.sendto(_query(SERVICE, PTR), MDNS)
            elif host is None:   s.sendto(_query(instance, SRV), MDNS)
            elif ip is None:     s.sendto(_query(host, A), MDNS)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline and ip is None:
                try: buf, _ = s.recvfrom(4096)
                except socket.timeout: continue
                try:
                    for name, rtype, val in _records(buf):
                        if rtype == PTR and name.lower() == SERVICE: instance = val
                        elif rtype == SRV and instance and name.lower() == instance.lower():
                            port, host = val[0][2], val[1]
                        elif rtype == A and host and name.lower() == host.lower(): ip = val
                except Exception:
                    continue
            if ip: break
    finally:
        s.close()
    if ip:
        try:
            os.makedirs(os.path.dirname(CACHE), exist_ok=True)
            json.dump({"ip": ip, "port": port, "instance": instance, "at": time.time()}, open(CACHE, "w"))
        except OSError:
            pass
        return ip, port
    return None


def cached() -> tuple[str, int] | None:
    try:
        c = json.load(open(CACHE)); return c["ip"], int(c.get("port", 47300))
    except Exception:
        return None


if __name__ == "__main__":
    found = discover()
    if found is None:
        found = cached()
        if found is None: print("no conductor found and nothing cached", file=sys.stderr); sys.exit(1)
        print(f"{found[0]}:{found[1]}", end=""); print("  (cached)", file=sys.stderr); sys.exit(0)
    print(f"{found[0]}:{found[1]}")
