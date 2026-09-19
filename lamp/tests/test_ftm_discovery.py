"""Tests for lamp/ftm_discovery.py. No real network; zeroconf is faked or blocked."""
import importlib
import random
import subprocess
import sys
import time
from pathlib import Path

import pytest

import ftm_discovery as d
from ftm_discovery import Candidate, choose, status_text

SFX = "._feelthemusic._udp.local."
AIR = "Feel the Music on Ann's MacBook Air"
PRO = "Feel the Music on Ann's MacBook Pro"
NOW = 10_000_000_000
AGE = 5_000_000_000


def cand(name=AIR, addrs=("192.168.1.20",), port=47300, seen=NOW, host="air.local."):
    return Candidate(name, host, port, tuple(addrs), seen)


def run(cands, **kw):
    kw.setdefault("now_ns", NOW)
    kw.setdefault("max_age_ns", AGE)
    return choose(cands, **kw)


# ---- (c) no config -------------------------------------------------------------------------
def test_two_conductors_no_config_is_ambiguous_and_selects_nobody():
    r = run([cand(AIR + SFX, ("192.168.1.20",)), cand(PRO + SFX, ("192.168.1.21",))])
    assert r.state == "ambiguous"
    assert r.selected is None
    assert r.names == tuple(sorted([AIR, PRO]))
    assert status_text(r) == "ambiguous conductor"


def test_ambiguous_does_not_depend_on_order_or_recency():
    a, b = cand(AIR, ("192.168.1.20",), seen=NOW - 10), cand(PRO, ("192.168.1.21",), seen=NOW)
    for lst in ([a, b], [b, a]):
        assert run(lst).selected is None
        assert run(lst).state == "ambiguous"


def test_single_fresh_selected():
    r = run([cand(AIR + SFX)])
    assert r.state == "selected" and r.selected == ("192.168.1.20", 47300)
    assert r.names == (AIR,)
    assert status_text(r) == "ok"


def test_zero_candidates_none():
    r = run([])
    assert r.state == "none" and r.selected is None
    assert status_text(r) == "no conductor"


def test_three_way_ambiguous():
    r = run([cand("a", ("10.0.0.1",)), cand("b", ("10.0.0.2",)), cand("c", ("10.0.0.3",))])
    assert r.state == "ambiguous" and r.selected is None


# ---- (b) configured name -------------------------------------------------------------------
def test_config_name_picks_right_one_of_two():
    cs = [cand(AIR + SFX, ("192.168.1.20",)), cand(PRO + SFX, ("192.168.1.21",), port=47301)]
    r = run(cs, configured_name=PRO)
    assert r.state == "selected" and r.selected == ("192.168.1.21", 47301)
    r = run(cs, configured_name=AIR)
    assert r.selected == ("192.168.1.20", 47300)
    assert run(list(reversed(cs)), configured_name=AIR).selected == ("192.168.1.20", 47300)


def test_config_name_suffix_stripped_on_both_sides():
    cs = [cand(AIR + SFX), cand(PRO, ("192.168.1.21",))]
    assert run(cs, configured_name=AIR + SFX).selected == ("192.168.1.20", 47300)
    assert run(cs, configured_name=PRO + SFX).selected == ("192.168.1.21", 47300)
    assert run(cs, configured_name=PRO).selected == ("192.168.1.21", 47300)


def test_config_name_is_case_sensitive():
    r = run([cand(AIR)], configured_name=AIR.lower())
    assert r.state == "not_found" and r.selected is None


def test_config_name_is_exact_not_prefix_or_substring():
    assert run([cand(AIR)], configured_name="Feel the Music").state == "not_found"
    assert run([cand(AIR)], configured_name=AIR + "x").state == "not_found"
    assert run([cand(AIR)], configured_name=" " + AIR).state == "not_found"


def test_config_name_not_found_text_has_name():
    r = run([cand(AIR)], configured_name="Studio")
    assert r.state == "not_found" and r.selected is None
    assert status_text(r) == "no conductor Studio"


def test_config_name_not_found_with_zero_candidates():
    assert run([], configured_name="Studio").state == "not_found"


def test_same_name_different_addresses_is_ambiguous():
    r = run([cand(AIR, ("192.168.1.20",)), cand(AIR + SFX, ("192.168.1.99",))], configured_name=AIR)
    assert r.state == "ambiguous" and r.selected is None
    assert status_text(r) == "ambiguous conductor"


def test_same_name_different_port_is_ambiguous():
    r = run([cand(AIR, port=47300), cand(AIR, port=47301)], configured_name=AIR)
    assert r.state == "ambiguous"


def test_config_name_ignores_stale_match():
    stale = cand(AIR, seen=NOW - AGE - 1)
    assert run([stale], configured_name=AIR).state == "not_found"


def test_suffix_only_middle_is_not_stripped():
    odd = "x" + SFX + "y"
    assert run([cand(odd)], configured_name=odd).state == "selected"
    assert run([cand(odd)], configured_name="x").state == "not_found"


def test_suffix_alone_is_a_plain_name():
    assert run([cand(AIR)], configured_name=SFX).state == "not_found"  # not stripped to empty
    assert run([cand(SFX)], configured_name=SFX).state == "selected"


# ---- (a) configured address ----------------------------------------------------------------
def test_config_addr_no_candidates_selected():
    r = run([], configured_addr="192.168.1.50")
    assert r.state == "selected" and r.selected == ("192.168.1.50", 47300)
    assert status_text(r) == "ok"


def test_config_addr_needs_no_valid_clock_or_candidates():
    r = choose(None, configured_addr="10.0.0.5:9000", now_ns=None, max_age_ns="x")
    assert r.selected == ("10.0.0.5", 9000)


def test_config_addr_wins_over_name_and_ambiguity():
    cs = [cand(AIR, ("192.168.1.20",)), cand(PRO, ("192.168.1.21",))]
    r = run(cs, configured_addr="10.1.1.1", configured_name=PRO)
    assert r.selected == ("10.1.1.1", 47300)


@pytest.mark.parametrize("text,expected", [
    ("1.2.3.4", ("1.2.3.4", 47300)),
    ("1.2.3.4:1", ("1.2.3.4", 1)),
    ("1.2.3.4:65535", ("1.2.3.4", 65535)),
    ("::1", ("::1", 47300)),
    ("2001:db8::1", ("2001:db8::1", 47300)),
    ("[2001:db8::1]:5000", ("2001:db8::1", 5000)),
    ("[2001:db8::1]", ("2001:db8::1", 47300)),
])
def test_parse_addr_ok(text, expected):
    assert d.parse_addr(text) == expected
    assert run([], configured_addr=text).selected == expected


@pytest.mark.parametrize("text", [
    "conductor.local", "localhost", "example.com:47300", "1.2.3.4:0", "1.2.3.4:65536",
    "1.2.3.4:-1", "1.2.3.4:", "1.2.3.4:x", "1.2.3.4:123456", "", " ", " 1.2.3.4", "1.2.3.4 ",
    "1.2.3", "256.1.1.1", "fe80::1%eth0", "[fe80::1%eth0]:5", "[1.2.3.4]:5", "[::1", "[::1]x",
    "::1:5000x", "1.2.3.4:5:6", "1.2.3.4:٣", "0x7f.0.0.1", 12345, None.__class__, b"1.2.3.4", ["1.2.3.4"],
])
def test_bad_config_addr_is_none_and_does_not_fall_back(text):
    r = run([cand(AIR)], configured_addr=text)
    assert r.state == "none" and r.selected is None
    assert "address" in r.reason


# ---- (d) stale, (e) dedup ------------------------------------------------------------------
def test_stale_ignored_and_boundary():
    assert run([cand(seen=NOW - AGE - 1)]).state == "none"
    assert run([cand(seen=NOW - AGE)]).state == "selected"  # exactly max age is still fresh
    assert run([cand(seen=NOW - AGE - 1), cand(PRO, ("192.168.1.21",))]).selected == ("192.168.1.21", 47300)


def test_stale_second_conductor_does_not_cause_ambiguity():
    r = run([cand(AIR), cand(PRO, ("192.168.1.21",), seen=NOW - 10 * AGE)])
    assert r.state == "selected" and r.selected == ("192.168.1.20", 47300)


def test_future_seen_is_fresh():
    assert run([cand(seen=NOW + 1_000_000)]).state == "selected"


def test_duplicates_collapse():
    r = run([cand(), cand(seen=NOW - 5), cand(AIR + SFX)])
    assert r.state == "selected"


def test_dedup_ignores_address_order_and_repeats():
    r = run([cand(addrs=("192.168.1.20", "10.0.0.2")), cand(addrs=("10.0.0.2", "192.168.1.20", "10.0.0.2"))])
    assert r.state == "selected"


def test_stale_duplicate_does_not_hide_fresh_one():
    old = cand(seen=NOW - AGE - 1)
    new = cand(seen=NOW)
    assert run([old, new]).state == "selected"
    assert run([new, old]).state == "selected"


def test_different_ports_or_addresses_are_distinct():
    assert run([cand(), cand(port=47301)]).state == "ambiguous"
    assert run([cand(), cand(addrs=("192.168.1.21",))]).state == "ambiguous"


# ---- (f) addresses -------------------------------------------------------------------------
def test_prefers_routable_ipv4_over_ipv6_and_linklocal_and_loopback():
    r = run([cand(addrs=("fe80::1", "2001:db8::5", "169.254.3.3", "127.0.0.1", "192.168.1.20"))])
    assert r.selected == ("192.168.1.20", 47300) and r.address_class == "ipv4"
    assert r.addresses == tuple(sorted(("fe80::1", "2001:db8::5", "169.254.3.3", "127.0.0.1", "192.168.1.20")))


def test_ipv6_only_selected_but_exposed():
    r = run([cand(addrs=("2001:db8::5",))])
    assert r.state == "selected" and r.selected == ("2001:db8::5", 47300)
    assert r.address_class == "ipv6" and r.addresses == ("2001:db8::5",)
    assert "ipv6" in r.reason


def test_linklocal_only_selected_but_exposed():
    r = run([cand(addrs=("169.254.9.9",))])
    assert r.selected == ("169.254.9.9", 47300) and r.address_class == "ipv4_link_local"
    r = run([cand(addrs=("fe80::1", "169.254.9.9"))])
    assert r.selected[0] == "169.254.9.9"
    r = run([cand(addrs=("fe80::1",))])
    assert r.selected == ("fe80::1", 47300) and r.address_class == "ipv6_link_local"


def test_ipv4_linklocal_beats_ipv6_global_is_not_assumed():
    # documented order: any IPv4 (even link-local) before IPv6
    r = run([cand(addrs=("2001:db8::5", "169.254.9.9"))])
    assert r.address_class == "ipv4_link_local"


def test_ipv6_global_beats_ipv6_linklocal_and_loopback_last():
    r = run([cand(addrs=("fe80::1", "2001:db8::5"))])
    assert r.selected[0] == "2001:db8::5"
    r = run([cand(addrs=("127.0.0.1", "fe80::1"))])
    assert r.selected[0] == "fe80::1"
    r = run([cand(addrs=("127.0.0.1",))])
    assert r.selected[0] == "127.0.0.1" and r.address_class == "loopback"


def test_config_addr_class_exposed():
    assert run([], configured_addr="fe80::1").address_class == "ipv6_link_local"
    assert run([], configured_addr="::1").address_class == "loopback"
    assert run([], configured_addr="2001:db8::1").address_class == "ipv6"
    assert run([], configured_addr="8.8.8.8").address_class == "ipv4"


def test_rejected_candidates_counted_and_do_not_participate():
    bad = [
        cand(addrs=()), cand(port=0), cand(port=65536), cand(port=-1), cand(port="47300"),
        cand(port=True), cand(name=5), cand(name=None), cand(name=""), cand(name="x" * 256),
        cand(addrs=("not-an-ip",)), cand(addrs=("host.local",)), cand(addrs=("fe80::1%eth0",)),
        cand(addrs=(5,)), cand(addrs="192.168.1.20"), cand(seen=1.5), cand(seen=float("nan")),
        cand(seen=None), cand(host=None), Candidate(AIR, "h", 47300, ("10.0.0.%d" % i for i in range(3)), NOW),
        "not a candidate", None, 3, {"name": "x"},
    ]
    r = run(bad + [cand(AIR)])
    assert r.rejected == len(bad)
    assert r.state == "selected" and r.selected == ("192.168.1.20", 47300)
    r = run(bad)
    assert r.state == "none" and r.rejected == len(bad)


def test_too_many_addresses_rejected():
    r = run([cand(addrs=tuple("10.0.0.%d" % i for i in range(17)))])
    assert r.state == "none" and r.rejected == 1
    r = run([cand(addrs=tuple("10.0.0.%d" % i for i in range(16)))])
    assert r.state == "selected"


def test_name_length_boundary():
    assert run([cand(name="x" * 255)]).state == "selected"
    assert run([cand(name="x" * 256)]).rejected == 1


def test_port_boundaries():
    assert run([cand(port=1)]).state == "selected"
    assert run([cand(port=65535)]).state == "selected"


def test_accepts_list_addresses():
    assert run([Candidate(AIR, "h", 47300, ["192.168.1.20"], NOW)]).state == "selected"


# ---- (g) hostile ---------------------------------------------------------------------------
def test_hostile_top_level_values_never_raise():
    for cs in (None, 5, "abc", {"a": 1}, object(), b"x", 1.5, float("nan")):
        r = run(cs)
        assert r.state == "none" and r.selected is None and r.reason
    for now in (None, "1", 1.5, float("nan"), True, object()):
        assert run([cand()], now_ns=now).state == "none"
    for age in (None, "1", 1.5, float("nan"), True, -1, object()):
        assert run([cand()], max_age_ns=age).state == "none"
    for name in (5, [], b"x", "", "x" * 256, float("nan")):
        assert run([cand()], configured_name=name).state == "none"
    # None means unset, "" does not
    assert run([cand()], configured_name=None).state == "selected"


def test_huge_candidate_list_is_none_with_reason():
    r = run([cand()] * (d.MAX_CANDIDATES + 1))
    assert r.state == "none" and "too many" in r.reason and r.rejected == d.MAX_CANDIDATES + 1
    assert run([cand()] * d.MAX_CANDIDATES).state == "selected"  # dedup collapses them


def test_tuple_of_candidates_accepted_generator_not():
    assert run((cand(),)).state == "selected"
    assert run(c for c in [cand()]).state == "none"


def test_fuzz_never_raises():
    rng = random.Random(1234)
    junk = [None, 0, -1, 1, 47300, 65535, 65536, 2**70, 1.5, float("nan"), float("inf"), True, "", "x",
            "192.168.1.5", "::1", "fe80::1%1", "a" * 400, b"b", (), [], {}, object(), AIR, AIR + SFX,
            SFX, "1.2.3.4:99999", ("192.168.1.5",), ["::1", "1.1.1.1"], (None,), "[::1]:5"]
    good_names = [AIR, PRO, AIR + SFX, PRO + SFX, "x"]
    good_addrs = ["192.168.1.20", "192.168.1.21", "169.254.1.1", "fe80::1", "2001:db8::1", "127.0.0.1"]

    def rnd_cand():
        kind = rng.random()
        if kind < 0.4:
            return Candidate(rng.choice(good_names), "h.local.", rng.choice([47300, 47301]),
                             tuple(rng.sample(good_addrs, rng.randint(1, 3))),
                             NOW - rng.randint(0, 2 * AGE))
        if kind < 0.9:
            return Candidate(*(rng.choice(junk) for _ in range(5)))
        return rng.choice(junk)

    for _ in range(5000):
        cs = rng.choice([
            [rnd_cand() for _ in range(rng.randint(0, 6))],
            tuple(rnd_cand() for _ in range(rng.randint(0, 6))),
            rng.choice(junk),
        ])
        r = choose(cs, configured_name=rng.choice([None, None] + junk + good_names),
                   configured_addr=rng.choice([None, None, None] + junk + good_addrs),
                   now_ns=rng.choice([NOW, NOW, NOW] + junk), max_age_ns=rng.choice([AGE, AGE, AGE] + junk))
        assert r.state in d.STATES
        assert (r.selected is not None) == (r.state == "selected")
        if r.selected is not None:
            ip, port = r.selected
            assert isinstance(ip, str) and isinstance(port, int) and 1 <= port <= 65535
        assert isinstance(status_text(r), str) and status_text(r)
        assert r.reason


def test_fuzz_ambiguity_invariant():
    """With no config, two distinct fresh valid candidates can never yield a selection."""
    rng = random.Random(99)
    for _ in range(2000):
        n = rng.randint(2, 5)
        cs = [cand(f"n{i}", (f"10.0.{i}.{rng.randint(1, 250)}",), seen=NOW - rng.randint(0, AGE)) for i in range(n)]
        rng.shuffle(cs)
        r = run(cs)
        assert r.state == "ambiguous" and r.selected is None


# ---- status text ---------------------------------------------------------------------------
def test_status_text_values():
    assert status_text(run([cand()])) == "ok"
    assert status_text(run([])) == "no conductor"
    assert status_text(run([cand(AIR), cand(PRO, ("10.0.0.9",))])) == "ambiguous conductor"
    assert status_text(run([], configured_name="Studio")) == "no conductor Studio"
    assert status_text(run([], configured_name="Studio" + SFX)) == "no conductor Studio"
    assert status_text(run([], configured_name="x" * 100)) == "no conductor " + "x" * 40
    assert status_text(run([], configured_name="a\nb\x00")) == "no conductor ab"
    assert status_text(d.Choice("none", None, (), "r")) == "no conductor"
    assert status_text(None) == "no conductor"


# ---- lazy import ---------------------------------------------------------------------------
def test_module_imports_and_chooses_with_zeroconf_blocked():
    code = (
        "import sys; sys.modules['zeroconf'] = None; sys.path.insert(0, %r)\n"
        "import ftm_discovery as d\n"
        "assert 'zeroconf' in sys.modules and sys.modules['zeroconf'] is None\n"
        "r = d.choose([], configured_addr='1.2.3.4', now_ns=0, max_age_ns=1)\n"
        "assert r.state == 'selected'\n"
        "try:\n    d.browse_once(0.05)\nexcept d.DiscoveryUnavailable as e:\n    print('unavailable', e)\n"
        "else:\n    raise SystemExit('expected DiscoveryUnavailable')\n"
    ) % str(Path(__file__).resolve().parents[1])
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert "unavailable" in out.stdout


def test_importing_does_not_import_zeroconf():
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import ftm_discovery\n"
        "assert 'zeroconf' not in sys.modules\n"
    ) % str(Path(__file__).resolve().parents[1])
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr


def test_browse_once_missing_zeroconf_raises_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "zeroconf", None)
    with pytest.raises(d.DiscoveryUnavailable, match="zeroconf"):
        d.browse_once(0.05)


# ---- browse_once with a fake zeroconf ------------------------------------------------------
class FakeInfo:
    def __init__(self, server, port, addrs):
        self.server, self.port, self._addrs = server, port, addrs

    def parsed_addresses(self):
        return list(self._addrs)


class FakeZeroconfModule:
    """Minimal stand-in for the few zeroconf calls browse_once uses."""

    def __init__(self, services=None, boot_error=None):
        self.services = services or {}
        self.boot_error = boot_error
        self.closed = 0
        self.cancelled = 0
        self.info_timeouts = []
        mod = self

        class Zeroconf:
            def __init__(self):
                if mod.boot_error:
                    raise mod.boot_error

            def get_service_info(self, type_, name, timeout=3000):
                assert type_ == "_feelthemusic._udp.local."
                mod.info_timeouts.append(timeout)
                return mod.services.get(name)

            def close(self):
                mod.closed += 1

        class ServiceBrowser:
            def __init__(self, zc, type_, listener=None):
                assert type_ == "_feelthemusic._udp.local."
                for name in list(mod.services) + ["ghost" + SFX]:
                    listener.add_service(zc, type_, name)
                    listener.add_service(zc, type_, name)  # duplicate announcements

            def cancel(self):
                mod.cancelled += 1

        self.Zeroconf, self.ServiceBrowser = Zeroconf, ServiceBrowser


def test_browse_once_with_fake_returns_candidates_and_choose_refuses():
    fake = FakeZeroconfModule({
        AIR + SFX: FakeInfo("air.local.", 47300, ["192.168.1.20"]),
        PRO + SFX: FakeInfo("pro.local.", 47300, ["192.168.1.21", "fe80::2"]),
    })
    t0 = time.monotonic_ns()
    cs = d.browse_once(0.2, zeroconf_module=fake)
    assert {c.name for c in cs} == {AIR + SFX, PRO + SFX}  # ghost (no info) dropped, dupes collapsed
    assert all(isinstance(c, Candidate) and c.seen_ns >= t0 for c in cs)
    assert [c for c in cs if c.name.startswith("Feel the Music on Ann's MacBook Pro")][0].addresses \
        == ("192.168.1.21", "fe80::2")
    assert fake.closed == 1 and fake.cancelled == 1
    r = choose(cs, now_ns=time.monotonic_ns(), max_age_ns=10**10)
    assert r.state == "ambiguous" and r.selected is None
    r = choose(cs, configured_name=PRO, now_ns=time.monotonic_ns(), max_age_ns=10**10)
    assert r.selected == ("192.168.1.21", 47300)


def test_browse_once_respects_timeout():
    fake = FakeZeroconfModule({AIR + SFX: FakeInfo("air.local.", 47300, ["192.168.1.20"])})
    t = time.monotonic()
    d.browse_once(0.3, zeroconf_module=fake)
    elapsed = time.monotonic() - t
    assert 0.15 <= elapsed <= 0.6  # waits to notice a second conductor, but not past the timeout
    assert all(ms <= 300 for ms in fake.info_timeouts)


def test_browse_once_empty_and_closes():
    fake = FakeZeroconfModule({})
    assert d.browse_once(0.05, zeroconf_module=fake) == []
    assert fake.closed == 1


def test_browse_once_boot_failure_is_unavailable():
    fake = FakeZeroconfModule(boot_error=OSError("no interface"))
    with pytest.raises(d.DiscoveryUnavailable, match="no interface"):
        d.browse_once(0.05, zeroconf_module=fake)


def test_browse_once_error_midway_closes_and_wraps():
    fake = FakeZeroconfModule({AIR + SFX: FakeInfo("a", 47300, ["192.168.1.20"])})

    def boom(self, *a, **k):
        raise RuntimeError("boom")

    fake.Zeroconf.get_service_info = boom
    with pytest.raises(d.DiscoveryUnavailable, match="boom"):
        d.browse_once(0.05, zeroconf_module=fake)
    assert fake.closed == 1


@pytest.mark.parametrize("bad", [0, -1, 61, None, "1", True, float("nan")])
def test_browse_once_rejects_bad_timeout(bad):
    with pytest.raises(ValueError):
        d.browse_once(bad, zeroconf_module=FakeZeroconfModule())


def test_reimport_module_is_clean():
    importlib.reload(d)
