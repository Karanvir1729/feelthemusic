"""Hostile-input and refusal-policy tests for lamp/bridge.py. None of these need the robot.

test_bridge.py covers the normal path, pinning and the 409 stop. These cover what it does not:
datagrams that used to escape the receive loop, and how the bridge treats repeated refusals (it
used to retry on every packet for as long as it ran).
"""
import json
import random
from unittest import mock

import numpy as np
import pytest

import bridge
from sdk import REFUSAL_LIMIT, SDKError, refusal_policy
from spatial import JOINTS
from test_bridge import FakeSocket, NoThermalThread, packet

PEER = ("192.0.2.9", 5000)


class FakeTime:
    """Stand-in for the `time` name inside bridge only. The clock is frozen except when a datagram
    arrives (see TimedSocket), so a test says exactly how many seconds apart packets are."""

    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def sleep(self, _seconds):
        pass


class TimedSocket(FakeSocket):
    def __init__(self, datagrams, clock, gap):
        super().__init__(datagrams)
        self.clock, self.gap = clock, gap

    def recvfrom(self, size):
        self.clock.now += self.gap
        return super().recvfrom(size)


def run(datagrams, move_effect=None, gap=20.0, extra_argv=()):
    """Run bridge.main() with `gap` seconds between datagrams. Returns (exit_code, sdk, sock).
    exit_code is None when the datagrams ran out first (FakeSocket raises KeyboardInterrupt)."""
    clock = FakeTime()
    sock = TimedSocket(datagrams, clock, gap)
    model = mock.MagicMock()
    model.limits = {joint: (-100.0, 100.0) for joint in JOINTS}
    model.problems.return_value = []
    model.distance_from_size.return_value = 0.5
    model.target_point.return_value = np.array([0.3, 0.2, 0.3])
    model.look_at.return_value = ({joint: 1.0 for joint in JOINTS}, {"aim_error_deg": 1.0})
    sdk = mock.MagicMock()
    sdk.base = "http://lamp"
    sdk.capabilities.return_value = {}
    sdk.joints.return_value = {
        "self_collision_check": True, "units": "normalized_m100_100",
        "joints": {joint: {} for joint in JOINTS}, "positions": {joint: 0.0 for joint in JOINTS}}
    sdk.move.side_effect = move_effect or (lambda _pose: {"action_id": "a", "result": {"reached": True}})
    argv = ["bridge.py", "--min-interval", "0", "--keep-idle", *extra_argv]
    code = None
    with mock.patch.object(bridge.sys, "argv", argv), \
            mock.patch.object(bridge, "LampModel", return_value=model), \
            mock.patch.object(bridge, "LampSDK", return_value=sdk), \
            mock.patch.object(bridge, "read_token", return_value="test-token"), \
            mock.patch.object(bridge, "temperature_c", return_value=40.0), \
            mock.patch.object(bridge.socket, "socket", return_value=sock), \
            mock.patch.object(bridge.threading, "Thread", NoThermalThread), \
            mock.patch.object(bridge, "time", clock):
        try:
            code = bridge.main()
        except KeyboardInterrupt:
            pass
    return code, sdk, sock


def deep(n, kind="array"):
    return (b"[" * n + b"]" * n) if kind == "array" else (b'{"a":' * n + b"1" + b"}" * n)


def refused(_pose):
    raise SDKError(400, "bad_request", "the planner would collide")


# --- datagrams that used to escape the receive loop -------------------------------------------------

@pytest.mark.parametrize("data", [
    pytest.param(deep(2000), id="array-2000"),
    pytest.param(deep(4000), id="array-4000"),
    pytest.param(deep(1500, "object"), id="object-1500"),
    pytest.param(deep(5), id="array-5"),
])
def test_deeply_nested_json_is_discarded_not_fatal(data):
    """One datagram of 2000 nested '[' used to raise RecursionError (a RuntimeError, so not in the
    bridge's except list) and kill it; 4000 fit inside the old 8 KB receive buffer."""
    assert bridge.parse_datagram(data, 5.0) is None
    _code, sdk, _sock = run([(data, PEER), (packet(1), PEER)])
    assert sdk.move.call_count == 1, "the bridge must survive and still serve the next good packet"


def test_brackets_inside_strings_do_not_count_as_nesting():
    assert bridge._nesting_depth(b'{"a": "[[[[[[[[[[[[", "b": "\\"[[[[["}') == 1
    assert bridge._nesting_depth(b"[[[]]]") == 3


def test_oversize_datagram_is_discarded():
    padded = packet(1)[:-1] + b',"pad":"' + b"x" * bridge.MAX_DATAGRAM_BYTES + b'"}'
    assert bridge.parse_datagram(padded, 5.0) is None
    assert bridge.parse_datagram(packet(1), 5.0) is not None


@pytest.mark.parametrize("data", [
    b"\xff\xfe\x00{", b"nan",
    b'{"t":"lamp_target","v":1,"session":"\\ud800","seq":1}',
    b'{"t":"lamp_target","v":1,"seq":' + b"9" * 5000 + b"}",
    b'{"t":"lamp_target","v":1,"session":"s","seq":1,"kind":"face","x":NaN,"y":Infinity,"size":1,"confidence":1}',
    b'{"t":"lamp_target","v":1,"session":{"a":1},"seq":1}',
])
def test_parse_datagram_never_raises_on_hostile_bytes(data):
    result = bridge.parse_datagram(data, 5.0)     # must not raise; a valid "no target" packet may be a dict
    assert result is None or isinstance(result, dict)


@pytest.mark.parametrize("data", [b"", b"{", b'{"t":"lamp_target","v":1,"seq":1}', b"[]", b"null"])
def test_clearly_invalid_datagrams_are_rejected(data):
    assert bridge.parse_datagram(data, 5.0) is None


def test_parse_datagram_survives_random_hostile_payloads():
    """Seeded fuzz: 4000 random byte strings, randomly corrupted valid packets and truncated packets.
    The only acceptable outcomes are a validated dict or None; an exception fails the test."""
    rng = random.Random(0)
    valid = packet(3)
    fields = ["t", "v", "seq", "session", "kind", "x", "y", "size", "confidence", "ttlMs", "replyPort", "maxStep"]
    junk = [None, [], {}, "x", "", -1, 2 ** 70, 1e400, True, "1", 1.5, float("nan")]
    accepted = 0
    for _ in range(4000):
        roll = rng.random()
        if roll < 0.35:
            data = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 200)))
        elif roll < 0.7:
            obj = json.loads(valid)
            for key in rng.sample(fields, rng.randrange(1, 5)):
                obj[key] = rng.choice(junk)
            data = json.dumps(obj, allow_nan=True).encode()
        else:
            data = valid[: rng.randrange(0, len(valid))]                 # cut off in flight
        result = bridge.parse_datagram(data, 5.0)
        assert result is None or isinstance(result, dict)
        accepted += result is not None
    assert accepted < 4000


# --- refusals: back off, count, stop -------------------------------------------------------------

def test_refusal_policy_matches_what_follow_py_does():
    """The numbers follow.py applies inline: 10 s back-off, 60 s for a rate limit, stop after three
    in a row, stop at once if torque is off or the SDK session limit is hit."""
    assert refusal_policy(SDKError(400, "bad_request", "would collide"), 1) == (False, 10.0, "refused")
    assert refusal_policy(SDKError(429, "rate_limited", "slow down"), 1)[:2] == (False, 60.0)
    assert refusal_policy(SDKError(429, "x", "slow down"), 2)[:2] == (False, 60.0)
    assert refusal_policy(SDKError(400, "bad_request", "would collide"), REFUSAL_LIMIT)[0] is True
    assert refusal_policy(SDKError(403, "forbidden", "Torque is off"), 1)[0] is True
    assert refusal_policy(SDKError(429, "rate_limited", "Too many active SDK sessions"), 1)[0] is True


def test_repeated_refusals_stop_the_bridge_after_three():
    """Used to be unbounded: every packet retried the SDK for as long as the bridge ran."""
    code, sdk, _ = run([(packet(i), PEER) for i in range(1, 9)], move_effect=refused, gap=20.0)
    assert sdk.move.call_count == REFUSAL_LIMIT
    assert code == 2


def test_a_refusal_backs_off_instead_of_retrying_on_the_next_packet():
    # packets 1 s apart: all eight arrive inside the 10 s back-off window
    code, sdk, _ = run([(packet(i), PEER) for i in range(1, 9)], move_effect=refused, gap=1.0)
    assert sdk.move.call_count == 1
    assert code is None


def test_a_rate_limit_backs_off_longer_than_a_plain_refusal():
    def limited(_pose):
        raise SDKError(429, "rate_limited", "slow down")
    three = [(packet(i), PEER) for i in range(1, 4)]
    # packets 20 s apart: past a 10 s back-off, inside a 60 s one
    _c, limited_sdk, _ = run(three, move_effect=limited, gap=20.0)
    _c, plain_sdk, _ = run(three, move_effect=refused, gap=20.0)
    assert limited_sdk.move.call_count == 1, "a rate limit must wait a full window, not 10 s"
    assert plain_sdk.move.call_count == 3, "a plain refusal only waits 10 s"


@pytest.mark.parametrize("message", ["torque is off", "Too many active SDK sessions this hour"])
def test_torque_off_or_session_limit_stops_immediately(message):
    def blocked(_pose):
        raise SDKError(429, "rate_limited", message)
    code, sdk, _ = run([(packet(i), PEER) for i in range(1, 6)], move_effect=blocked)
    assert sdk.move.call_count == 1
    assert code == 2


def test_a_success_resets_the_refusal_streak():
    calls = {"n": 0}

    def flaky(_pose):
        calls["n"] += 1
        if calls["n"] % 2 == 1:
            raise SDKError(400, "bad_request", "the planner would collide")
        return {"action_id": "a", "result": {"reached": True}}
    code, sdk, _ = run([(packet(i), PEER) for i in range(1, 9)], move_effect=flaky, gap=20.0)
    assert sdk.move.call_count == 8, "alternating failure and success must never reach the stop limit"
    assert code is None


# --- exit status --------------------------------------------------------------------------------------

@pytest.mark.parametrize("failure", [SDKError(409, "lost_track", "unknown"), SDKError(409, "not_reached", "blocked"),
                                     SDKError(409, "canceled", "someone else took over")])
def test_uncertain_or_incomplete_moves_exit_non_zero(failure):
    """A supervisor that restarts a bridge on exit 0 must not restart one that stopped because the
    arm may be obstructed or in an unknown state."""
    def boom(_pose):
        raise failure
    code, sdk, _ = run([(packet(i), PEER) for i in range(1, 4)], move_effect=boom)
    assert sdk.move.call_count == 1
    assert code == 2


# --- sender validation ----------------------------------------------------------------------------------

def test_allow_ip_restricts_senders_before_anything_is_pinned():
    """Without --allow-ip the first valid sender is pinned, so whoever speaks first wins. With it,
    a stranger cannot claim the bridge even if it speaks first."""
    _code, sdk, sock = run([(packet(1), ("192.0.2.66", 5000)), (packet(2), ("192.0.2.9", 5000)),
                            (packet(3), ("192.0.2.66", 5000))], extra_argv=["--allow-ip", "192.0.2.9"])
    assert sdk.move.call_count == 1
    assert sock.sent and all(addr[0] == "192.0.2.9" for _data, addr in sock.sent)


def test_a_stranger_speaking_first_can_pin_the_bridge_without_allow_ip():
    """Documents the known limit: pinning is not authentication. UDP source addresses can also be
    spoofed on the LAN, so --allow-ip narrows the exposure but does not remove it."""
    _code, sdk, sock = run([(packet(1), ("192.0.2.66", 5000)), (packet(2), ("192.0.2.9", 5000))])
    assert sdk.move.call_count == 1
    assert all(addr[0] == "192.0.2.66" for _data, addr in sock.sent), "the stranger got the lamp"
