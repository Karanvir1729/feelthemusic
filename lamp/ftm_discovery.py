"""Conductor discovery selection for the lamp (task #19).

Why this exists: two conductors were measured advertising ``_feelthemusic._udp`` on port 47300 at
the same time on the same Wi-Fi ("Feel the Music on <name>'s MacBook Pro" and "... MacBook Air").
The shipped phone joins the first result, which is wrong. The lamp never takes "the first": it joins
a conductor chosen by a configured instance name or a fixed address, and if several are found and
none is configured it reports "ambiguous conductor" and joins NOBODY.

``choose`` is pure: no network, no wall clock, never raises. ``browse_once`` is a thin wrapper that
imports ``zeroconf`` lazily; see its docstring for what is and is not verified.

Rules, in order
---------------
a. ``configured_addr`` ("ip" or "ip:port"; "[v6]:port" for IPv6 with a port; IPv4/IPv6 literals only,
   no hostnames, no ``%zone``, port 1..65535, default 47300) wins over everything, needs no
   discovery, and selects even with zero candidates. A configured but invalid address gives state
   "none" and does NOT fall back to discovery.
b. else ``configured_name`` (exact, case-sensitive, after stripping a trailing
   ``._feelthemusic._udp.local.`` from both sides) selects the one candidate with that name;
   "not_found" if none; "ambiguous" if several distinct candidates share it.
c. else exactly one fresh distinct candidate -> "selected"; none -> "none"; two or more ->
   "ambiguous". Never first, never newest, never sorted-and-take-one.
d. candidates with now_ns - seen_ns > max_age_ns are ignored.
e. the same (name, sorted addresses, port) counts once (newest seen_ns kept).
f. address preference for a selected candidate: routable IPv4, then link-local IPv4, then
   non-link-local IPv6, then link-local IPv6, then loopback. ``Choice.address_class`` and
   ``Choice.addresses`` expose what was picked so the caller can decide (IPv6 is never silently
   preferred over IPv4). Invalid candidates (bad shapes, empty/invalid address list, port outside
   1..65535, non-str/over-long name) are counted in ``Choice.rejected``.
g. hostile input never raises; it yields state "none" with a reason.
"""
from __future__ import annotations

import importlib
import ipaddress
import time
from dataclasses import dataclass
from typing import Any, Optional

SERVICE_TYPE = "_feelthemusic._udp.local."
SERVICE_SUFFIX = "." + SERVICE_TYPE
DEFAULT_PORT = 47300

MAX_NAME_LEN = 255
MAX_ADDRESSES = 16
MAX_CANDIDATES = 512

STATES = ("selected", "none", "ambiguous", "not_found")


class DiscoveryUnavailable(RuntimeError):
    """The zeroconf package is missing or could not start."""


@dataclass(frozen=True)
class Candidate:
    name: str  # mDNS instance name, with or without the service suffix
    host: str  # advertised server host name (informational only)
    port: int
    addresses: tuple  # tuple[str, ...] of IP literals
    seen_ns: int  # caller's monotonic clock, same clock as choose(now_ns=)


@dataclass(frozen=True)
class Choice:
    state: str  # one of STATES
    selected: Optional[tuple]  # (ip, port) only when state == "selected"
    names: tuple  # names of the fresh distinct candidates considered
    reason: str
    rejected: int = 0
    addresses: tuple = ()  # every address of the selected candidate
    address_class: str = ""  # "ipv4", "ipv4_link_local", "ipv6", "ipv6_link_local", "loopback"
    wanted: str = ""  # configured name, for status_text


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _strip_suffix(name: str) -> str:
    if name.endswith(SERVICE_SUFFIX) and len(name) > len(SERVICE_SUFFIX):
        return name[: -len(SERVICE_SUFFIX)]
    return name


def _port_of(port_s: str) -> Optional[int]:
    if port_s.isascii() and port_s.isdigit() and len(port_s) <= 5:
        return int(port_s)
    return None


def parse_addr(text: Any) -> Optional[tuple]:
    """Parse "ip", "ip:port" or "[v6]:port" into (ip_str, port); None if invalid."""
    if not isinstance(text, str) or not text or len(text) > 64 or text != text.strip():
        return None
    if "%" in text or any(ch.isspace() for ch in text):
        return None
    host, port = text, DEFAULT_PORT
    if text.startswith("["):
        end = text.find("]")
        if end < 0:
            return None
        host, rest = text[1:end], text[end + 1:]
        if rest:
            if not rest.startswith(":"):
                return None
            port = _port_of(rest[1:])
            if port is None:
                return None
        try:
            if ipaddress.ip_address(host).version != 6:
                return None
        except ValueError:
            return None
    elif text.count(":") == 1:
        host, port_s = text.split(":")
        port = _port_of(port_s)
        if port is None:
            return None
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return None
    if not 1 <= port <= 65535:
        return None
    return (str(ip), port)


_PREFERENCE = ("ipv4", "ipv4_link_local", "ipv6", "ipv6_link_local", "loopback")


def _classify(ip_s: str) -> str:
    ip = ipaddress.ip_address(ip_s)
    if ip.is_loopback:
        return "loopback"
    if ip.version == 4:
        return "ipv4_link_local" if ip.is_link_local else "ipv4"
    return "ipv6_link_local" if ip.is_link_local else "ipv6"


def _valid_candidate(c: Any) -> Optional[Candidate]:
    """Return a normalised Candidate (addresses canonical, sorted, unique) or None."""
    if not isinstance(c, Candidate):
        return None
    if not isinstance(c.name, str) or not 0 < len(c.name) <= MAX_NAME_LEN:
        return None
    if not isinstance(c.host, str) or len(c.host) > MAX_NAME_LEN:
        return None
    if not _is_int(c.port) or not 1 <= c.port <= 65535:
        return None
    if not _is_int(c.seen_ns):
        return None
    if not isinstance(c.addresses, (tuple, list)) or not 0 < len(c.addresses) <= MAX_ADDRESSES:
        return None
    addrs = []
    for a in c.addresses:
        if not isinstance(a, str) or len(a) > 64 or "%" in a:
            return None
        try:
            addrs.append(str(ipaddress.ip_address(a)))
        except ValueError:
            return None
    return Candidate(c.name, c.host, c.port, tuple(sorted(set(addrs))), c.seen_ns)


def _none(reason: str, rejected: int = 0, wanted: str = "") -> Choice:
    return Choice("none", None, (), reason, rejected, wanted=wanted)


def _select(c: Candidate, names: tuple, rejected: int, wanted: str) -> Choice:
    best_rank, best = len(_PREFERENCE), None
    for a in c.addresses:  # already sorted, so ties are deterministic
        rank = _PREFERENCE.index(_classify(a))
        if rank < best_rank:
            best_rank, best = rank, a
    cls = _PREFERENCE[best_rank]
    note = "" if cls == "ipv4" else f" (only {cls} address)"
    return Choice("selected", (best, c.port), names, f"selected {_strip_suffix(c.name)}{note}",
                  rejected, c.addresses, cls, wanted)


def choose(candidates: Any, *, configured_name: Any = None, configured_addr: Any = None,
           now_ns: Any, max_age_ns: Any) -> Choice:
    """Pick a conductor or refuse to. Pure, never raises. See module docstring for the rules."""
    try:
        return _choose(candidates, configured_name, configured_addr, now_ns, max_age_ns)
    except Exception as exc:  # last-resort guard: hostile input must not take the lamp down
        return _none(f"internal error on hostile input: {type(exc).__name__}")


def _choose(candidates, configured_name, configured_addr, now_ns, max_age_ns) -> Choice:
    if configured_addr is not None:
        parsed = parse_addr(configured_addr)
        if parsed is None:
            return _none("configured address is not an IP literal (ip, ip:port, [v6]:port)")
        return Choice("selected", parsed, (), "configured address", 0, (parsed[0],),
                      _classify(parsed[0]))

    wanted = ""
    if configured_name is not None:
        if (not isinstance(configured_name, str) or len(configured_name) > MAX_NAME_LEN
                or not _strip_suffix(configured_name)):
            return _none("configured name is invalid")
        wanted = _strip_suffix(configured_name)

    if not _is_int(now_ns) or not _is_int(max_age_ns) or max_age_ns < 0:
        return _none("invalid clock or max age", wanted=wanted)
    if not isinstance(candidates, (list, tuple)):
        return _none("candidates is not a list", wanted=wanted)
    if len(candidates) > MAX_CANDIDATES:
        return _none("too many candidates", len(candidates), wanted)

    rejected = 0
    newest: dict = {}
    for raw in candidates:
        c = _valid_candidate(raw)
        if c is None:
            rejected += 1
            continue
        if now_ns - c.seen_ns > max_age_ns:
            continue
        key = (_strip_suffix(c.name), c.addresses, c.port)
        prev = newest.get(key)
        if prev is None or c.seen_ns > prev.seen_ns:
            newest[key] = c

    fresh = list(newest.values())
    names = tuple(sorted({_strip_suffix(c.name) for c in fresh}))

    if wanted:
        matches = [c for c in fresh if _strip_suffix(c.name) == wanted]
        if not matches:
            return Choice("not_found", None, names, f"no conductor named {wanted!r}", rejected,
                          wanted=wanted)
        if len(matches) > 1:
            return Choice("ambiguous", None, names,
                          f"{len(matches)} conductors share the name {wanted!r}", rejected,
                          wanted=wanted)
        return _select(matches[0], names, rejected, wanted)

    if not fresh:
        return Choice("none", None, names, "no conductor found", rejected)
    if len(fresh) > 1:
        return Choice("ambiguous", None, names,
                      f"{len(fresh)} conductors found and none configured", rejected)
    return _select(fresh[0], names, rejected, "")


def status_text(choice: Choice) -> str:
    """Short string for the lamp's telemetry state field."""
    state = getattr(choice, "state", None)
    if state == "selected":
        return "ok"
    if state == "ambiguous":
        return "ambiguous conductor"
    if state == "not_found":
        w = "".join(ch for ch in str(getattr(choice, "wanted", "")) if ch.isprintable())[:40]
        return f"no conductor {w}".rstrip()
    return "no conductor"


def browse_once(timeout_s: float, *, zeroconf_module: Any = None) -> list:
    """Browse ``_feelthemusic._udp.local.`` for about ``timeout_s`` seconds.

    Returns a list of Candidate stamped with ``time.monotonic_ns()`` (pass the same clock as
    ``now_ns`` to ``choose``). It waits most of the timeout even if one conductor appears early,
    because the point is to notice a second one. ``zeroconf`` is imported here, never at module
    import. Raises DiscoveryUnavailable if it is not installed or cannot start.

    NOT VERIFIED: this wrapper is untested against real mDNS on this machine; it is exercised only
    against an injected fake. The Pi has python zeroconf 0.151.3 in a venv, so the calls used
    (Zeroconf(), ServiceBrowser(zc, type, listener=), zc.get_service_info(type, name, timeout=ms),
    info.parsed_addresses()/.port/.server, zc.close()) are the classic API, assumed unchanged there.
    """
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) \
            or not 0 < timeout_s <= 60:
        raise ValueError("timeout_s must be a number in (0, 60]")
    if zeroconf_module is None:
        try:
            zeroconf_module = importlib.import_module("zeroconf")
        except ImportError as exc:
            raise DiscoveryUnavailable("python package 'zeroconf' is not installed") from exc

    start = time.monotonic()
    deadline = start + float(timeout_s)
    browse_until = start + 0.6 * float(timeout_s)
    found: list = []

    class _Listener:
        def add_service(self, zc, type_, name):
            if name not in found and len(found) < 64:
                found.append(name)

        def update_service(self, zc, type_, name):
            self.add_service(zc, type_, name)

        def remove_service(self, zc, type_, name):
            pass

    try:
        zc = zeroconf_module.Zeroconf()
    except Exception as exc:
        raise DiscoveryUnavailable(f"zeroconf failed to start: {exc}") from exc
    out: list = []
    try:
        browser = zeroconf_module.ServiceBrowser(zc, SERVICE_TYPE, listener=_Listener())
        try:
            while time.monotonic() < browse_until:
                time.sleep(min(0.02, max(0.0, browse_until - time.monotonic())))
            names = list(found)
            for i, name in enumerate(names):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                ms = max(1, int(remaining * 1000 / (len(names) - i)))
                info = zc.get_service_info(SERVICE_TYPE, name, timeout=ms)
                if info is None:
                    continue
                out.append(Candidate(str(name), str(getattr(info, "server", "") or ""),
                                     info.port, tuple(info.parsed_addresses()),
                                     time.monotonic_ns()))
        finally:
            cancel = getattr(browser, "cancel", None)
            if cancel:
                cancel()
    except Exception as exc:
        raise DiscoveryUnavailable(f"zeroconf browse failed: {exc}") from exc
    finally:
        try:
            zc.close()
        except Exception:
            pass
    return out
