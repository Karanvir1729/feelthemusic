import json
from unittest import mock

import numpy as np

import bridge
from sdk import SDKError
from spatial import JOINTS


def packet(seq=1, *, reply_port=47401, session="s"):
    return json.dumps({
        "v": 1,
        "t": "lamp_target",
        "session": session,
        "seq": seq,
        "kind": "face",
        "x": 0.7,
        "y": 0.4,
        "size": 0.2,
        "confidence": 0.9,
        "ttlMs": 1500,
        "maxStep": 5,
        "replyPort": reply_port,
    }).encode()


class FakeSocket:
    def __init__(self, datagrams):
        self.datagrams = list(datagrams)
        self.sent = []

    def bind(self, *_args):
        pass

    def settimeout(self, *_args):
        pass

    def close(self):
        pass

    def sendto(self, data, addr):
        self.sent.append((data, addr))

    def recvfrom(self, _size):
        if not self.datagrams:
            raise KeyboardInterrupt
        return self.datagrams.pop(0)


class NoThermalThread:
    def __init__(self, *_args, **_kwargs):
        pass

    def start(self):
        pass


def run_bridge(datagrams, move_effect=None):
    sock = FakeSocket(datagrams)
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
        "self_collision_check": True,
        "units": "normalized_m100_100",
        "joints": {joint: {} for joint in JOINTS},
        "positions": {joint: 0.0 for joint in JOINTS},
    }
    sdk.move.side_effect = move_effect or (lambda _pose: {"action_id": "a", "result": {"reached": True}})

    argv = ["bridge.py", "--min-interval", "0", "--keep-idle"]
    with mock.patch.object(bridge.sys, "argv", argv), \
            mock.patch.object(bridge, "LampModel", return_value=model), \
            mock.patch.object(bridge, "LampSDK", return_value=sdk), \
            mock.patch.object(bridge, "read_token", return_value="test-token"), \
            mock.patch.object(bridge, "temperature_c", return_value=40.0), \
            mock.patch.object(bridge.socket, "socket", return_value=sock), \
            mock.patch.object(bridge.threading, "Thread", NoThermalThread), \
            mock.patch.object(bridge.time, "sleep", lambda _seconds: None):
        try:
            bridge.main()
        except KeyboardInterrupt:
            pass
    return sock, sdk


def test_normalize_packet_rejects_non_objects_and_bad_numbers():
    for value in ([], "hello", 42, None, True):
        assert bridge.normalize_packet(value, 5.0) is None
    assert bridge.normalize_packet({"v": 1, "t": "lamp_target", "session": "s", "seq": 1,
                                    "kind": "face", "x": float("nan"), "y": 0.5,
                                    "size": 0.2, "confidence": 0.9}, 5.0) is None
    assert bridge.normalize_packet({"v": 1, "t": "lamp_target", "session": "s", "seq": 1,
                                    "kind": "face", "x": 0.5, "y": 0.5,
                                    "size": 0.2, "confidence": 0.9, "ttlMs": 999999}, 5.0) is None


def test_malformed_json_objects_are_ignored_without_killing_the_bridge():
    sock, sdk = run_bridge([(b"[]", ("192.0.2.9", 5000)),
                            (b'"hello"', ("192.0.2.9", 5000)),
                            (packet(), ("192.0.2.9", 5000))])

    assert sdk.move.call_count == 1
    assert sock.sent


def test_first_valid_sender_is_pinned_and_other_senders_cannot_move_the_lamp():
    sock, sdk = run_bridge([(packet(1), ("192.0.2.9", 5000)),
                            (packet(2), ("192.0.2.10", 5000))])

    assert sdk.move.call_count == 1
    assert all(addr[0] == "192.0.2.9" for _data, addr in sock.sent)


def test_uncertain_move_failure_stops_before_the_next_target():
    def lost_track(_pose):
        raise SDKError(409, "lost_track", "the action outcome is unknown")

    _sock, sdk = run_bridge([(packet(1), ("192.0.2.9", 5000)),
                             (packet(2), ("192.0.2.9", 5000)),
                             (packet(3), ("192.0.2.9", 5000))], move_effect=lost_track)

    assert sdk.move.call_count == 1


def test_terminal_motion_failures_are_stopping_conditions():
    for code in ("lost_track", "timeout", "not_reached", "canceled"):
        assert bridge.terminal_motion_failure(SDKError(409, code, "failed"))
    assert not bridge.terminal_motion_failure(SDKError(400, "bad_request", "failed"))
