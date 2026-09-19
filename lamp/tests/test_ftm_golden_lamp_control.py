"""Golden bytes: the Control a role=lamp peer receives from Mac conductor build 483ded3.

Source: a loopback capture by @meharsclaude (hub general seq 1310, 2026-09-19): hello
{"t":"hi","role":"lamp","name":"ctl-capture","v":1,"audio":false} plus a SyncReq; 7 identical Control receipts
in 6 s and no audio frames. The JSON is verbatim. It is the first real message with a `lamp` key; the earlier
builds (before 1edd442) sent none (see test_ftm_golden.py). `bold` is new (0..1, 2 decimals) and is not part of
docs/ftm-protocol.md. This proves the client reads `mode`, `lights` and `gen` from a real conductor and ignores
`bold`; it does not prove what the lamp should DO in a mode.
"""
import pytest

from ftm_session import ModeOut, Session, SessionOut

MS = 1_000_000
REAL = (
    '{"asr":48000,"bassGain":1,"codec":2,"cookie":"AAAIAAAAu4AAAAHg///8GAAAAAEAAAAAAAAAAA==","fpp":480,'
    '"hapticGain":1,"lamp":{"bold":1,"gen":2,"lights":true,"mode":"dance"},"lat":300,"mode":"music",'
    '"session":3500669897,"v":2}'
)
T0 = 10**12


def control(text):
    return b"\x0d" + text.encode("utf-8")


def test_real_lamp_control_selects_dance_with_lights_and_the_conductors_gen():
    s = Session("golden")
    out = s.on_datagram(control(REAL), T0)
    assert out == [SessionOut(3500669897), ModeOut(mode="dance", lights=True, gen=2)]
    assert s.current_mode(T0 + MS) == "dance" and s.current_lights(T0 + MS) is True
    assert s.stats()["bad_control"] == 0 and s.stats()["bad_mode"] == 0


@pytest.mark.parametrize("bold", ["0", "0.75", "0.05", "1.0"])
def test_a_fractional_or_integral_bold_never_makes_the_control_invalid(bold):
    s = Session("golden")
    out = s.on_datagram(control(REAL.replace('"bold":1', f'"bold":{bold}')), T0)
    assert ModeOut("dance", True, 2) in out
    assert s.stats()["bad_control"] == 0


def test_the_conductor_repeats_the_same_control_and_repeats_are_harmless():
    """7 identical receipts in 6 s: the equal gen is a refresh, not a new mode or a restart."""
    s = Session("golden")
    first = s.on_datagram(control(REAL), T0)
    repeats = [s.on_datagram(control(REAL), T0 + i * 800 * MS) for i in range(1, 7)]
    assert len(first) == 2 and all(r == [] for r in repeats)
    assert s.stats()["mode_refresh"] == 6 and s.stats()["stale_mode"] == 0


def test_a_higher_gen_changes_the_mode_and_a_lower_gen_is_ignored():
    s = Session("golden")
    s.on_datagram(control(REAL), T0)
    follow = REAL.replace('"gen":2', '"gen":3').replace('"mode":"dance"', '"mode":"follow"')
    assert s.on_datagram(control(follow), T0 + MS) == [ModeOut("follow", True, 3)]
    assert s.on_datagram(control(REAL), T0 + 2 * MS) == []     # gen 2 after gen 3: stale
    assert s.current_mode(T0 + 3 * MS) == "follow"
