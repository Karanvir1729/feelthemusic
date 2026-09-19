"""Offline SDK contract checks, written 2026-09-19. No sockets or robot access."""

import io
from unittest.mock import Mock

import pytest
import requests

from sdk import LampSDK, SDKError, read_token


@pytest.fixture
def client(monkeypatch):
    # Replace the session constructor before creating the client. A regression
    # must not accidentally send a command to an available local robot.
    session = Mock(headers={})
    monkeypatch.setattr("sdk.requests.Session", lambda: session)
    return LampSDK("test-only-token")


def response(body, status=200):
    return Mock(status_code=status, text="test response", json=Mock(return_value=body))


def test_environment_token_wins_over_file(monkeypatch, tmp_path):
    token_file = tmp_path / ".env"
    token_file.write_text('LELAMP_SDK_TOKEN="file-token"\n')
    monkeypatch.setenv("LELAMP_SDK_TOKEN", " environment-token ")
    assert read_token(token_file) == "environment-token"


def test_quoted_file_token_and_missing_token(monkeypatch, tmp_path):
    monkeypatch.delenv("LELAMP_SDK_TOKEN", raising=False)
    token_file = tmp_path / ".env"
    token_file.write_text('OTHER=value\nLELAMP_SDK_TOKEN="file-token"\n')
    assert read_token(token_file) == "file-token"
    with pytest.raises(SDKError, match="no_token"):
        read_token(tmp_path / "absent")


@pytest.mark.parametrize("method,path", [
    ("capabilities", "/api/sdk/v1/capabilities"),
    ("joints", "/api/sdk/v1/joints"),
])
def test_reads_stay_on_sdk_gateway(client, method, path):
    body = {"ok": True, "data": {}}
    client.http.request.return_value = response(body)
    assert getattr(client, method)() == body
    client.http.request.assert_called_once_with(
        "GET", "http://127.0.0.1:8081" + path, json=None, timeout=20,
    )


@pytest.mark.parametrize("status,body", [
    (403, {"error": {"code": "refused", "message": "denied"}}),
    (200, {"ok": False, "error": {"code": "refused", "message": "denied"}}),
])
def test_gateway_refusal_never_becomes_success(client, status, body):
    client.http.request.return_value = response(body, status)
    with pytest.raises(SDKError) as failure:
        client.capabilities()
    assert failure.value.status == status
    assert failure.value.code == "refused"


def test_terminal_action_failure_is_not_retried(client, monkeypatch):
    monkeypatch.setattr(client, "_ensure_session", Mock())
    client.http.request.return_value = response({
        "action": {"action_id": "a", "state": "failed",
                   "error": {"code": "collision", "message": "refused"}},
    })
    with pytest.raises(SDKError, match="collision"):
        client.action("motion.move", {"positions": {}})
    assert client.http.request.call_count == 1


def test_unauthorized_action_renews_once(client, monkeypatch):
    renew = Mock()
    monkeypatch.setattr(client, "_ensure_session", renew)
    client.http.request.side_effect = [
        response({"error": {"code": "expired"}}, 401),
        response({"action": {"action_id": "a", "state": "succeeded"}}),
    ]
    assert client.action("light.glow", {"color": [0, 0, 0]})["state"] == "succeeded"
    assert renew.call_count == 2
    assert client.http.request.call_count == 2


def test_repeated_unauthorized_does_not_loop(client, monkeypatch):
    monkeypatch.setattr(client, "_ensure_session", Mock())
    client.http.request.return_value = response({"error": {"code": "expired"}}, 401)
    with pytest.raises(SDKError):
        client.action("light.glow", {"color": [0, 0, 0]})
    assert client.http.request.call_count == 2


@pytest.mark.parametrize("body", [None, [], "bad", 3, {"ok": "yes"}])
def test_malformed_response_is_a_controlled_sdk_error(client, body):
    client.http.request.return_value = response(body)
    with pytest.raises(SDKError, match="invalid_response"):
        client.capabilities()


def test_invalid_json_does_not_appear_successful(client):
    client.http.request.return_value = response(None)
    client.http.request.return_value.json.side_effect = ValueError("invalid JSON")
    with pytest.raises(SDKError, match="invalid_response"):
        client.capabilities()


@pytest.mark.parametrize("bad_response", [None, [], "bad"])
def test_ambiguous_action_reply_is_terminal_not_a_retryable_refusal(client, monkeypatch, bad_response):
    monkeypatch.setattr(client, "_ensure_session", Mock())
    client.http.request.return_value = response(bad_response)
    with pytest.raises(SDKError) as failure:
        client.action("motion.move", {})
    assert failure.value.status == 409
    assert failure.value.code == "lost_track"
    assert client.http.request.call_count == 1


def test_transport_failure_after_submission_is_terminal_and_not_retried(client, monkeypatch):
    monkeypatch.setattr(client, "_ensure_session", Mock())
    client.http.request.side_effect = requests.ConnectionError("untrusted response text")
    with pytest.raises(SDKError) as failure:
        client.action("motion.move", {})
    assert failure.value.status == 409
    assert failure.value.code == "lost_track"
    assert "untrusted response text" not in str(failure.value)
    assert client.http.request.call_count == 1


@pytest.mark.parametrize("session", [None, {}, {"session_id": "a", "expires_at": float("nan")},
                                     {"session_id": [], "expires_at": 1000}])
def test_malformed_session_is_rejected_before_header_mutation(client, session):
    client.http.request.return_value = response({"session": session})
    with pytest.raises(SDKError, match="invalid_response"):
        client._ensure_session()
    assert "X-LeLamp-SDK-Session" not in client.http.headers


@pytest.mark.parametrize("action", [None, {}, {"state": "running"}, {"state": [], "action_id": "a"}])
def test_malformed_action_fails_without_polling(client, monkeypatch, action):
    monkeypatch.setattr(client, "_ensure_session", Mock())
    client.http.request.return_value = response({"action": action})
    with pytest.raises(SDKError, match="invalid_response"):
        client.action("motion.move", {}, wait_s=0.01)
    assert client.http.request.call_count == 1


@pytest.mark.parametrize("wait", [0, -1, True, float("nan"), float("inf"), pytest.param(10**1000, id="huge-int")])
def test_invalid_wait_budget_cannot_submit_an_action(client, monkeypatch, wait):
    monkeypatch.setattr(client, "_ensure_session", Mock(side_effect=AssertionError("validate before session")))
    with pytest.raises(ValueError, match="wait_s"):
        client.action("motion.move", {}, wait_s=wait)
    client.http.request.assert_not_called()


def test_polling_respects_remaining_budget(client, monkeypatch):
    now = [100.0]
    monkeypatch.setattr("sdk.time.monotonic", lambda: now[0])
    monkeypatch.setattr("sdk.time.sleep", lambda duration: now.__setitem__(0, now[0] + duration))
    monkeypatch.setattr(client, "_ensure_session", Mock())

    def request(method, url, **kwargs):
        if method == "GET":
            assert now[0] < 100.15, "no poll can begin after the deadline"
            assert 0 < kwargs["timeout"] <= 100.15 - now[0]
        return response({"action": {"action_id": "a", "state": "running"}})

    client.http.request.side_effect = request
    with pytest.raises(SDKError, match="timeout"):
        client.action("motion.move", {}, wait_s=0.15)
    assert now[0] == pytest.approx(100.15)


def stream_response(client, payload):
    stream = response({})
    stream.raw = io.BytesIO(payload)
    client.http.get.return_value = stream
    return stream


def test_camera_generator_close_releases_response(client):
    stream = stream_response(client, b"--lelamp\r\nContent-Length: 3\r\n\r\nabc")
    frames = client.camera_frames()
    assert next(frames) == b"abc"
    frames.close()
    stream.close.assert_called_once()


@pytest.mark.parametrize("length", [b"-1", b"0", b"invalid", b"4194305", b"999999999999"])
def test_camera_invalid_lengths_rejected_without_payload_read(client, length):
    stream = stream_response(client, b"--lelamp\r\nContent-Length: " + length + b"\r\n\r\n")
    with pytest.raises(SDKError, match="camera"):
        list(client.camera_frames())
    stream.close.assert_called_once()


@pytest.mark.parametrize("payload", [
    b"--lelamp\r\nContent-Length: 10\r\n\r\nshort",
    b"--lelamp\r\nContent-Length: 3\r\nContent-Length: 3\r\n\r\nabc",
    b"x" * 9000 + b"\r\n",
    b"--lelamp\r\n" + b"X-Header: 1\r\n" * 6000,
], ids=["truncated", "duplicate-length", "long-line", "many-headers"])
def test_camera_truncation_duplicate_length_and_unbounded_headers_fail(client, payload):
    stream = stream_response(client, payload)
    with pytest.raises(SDKError, match="camera"):
        list(client.camera_frames())
    stream.close.assert_called_once()


def test_poll_cannot_report_success_for_a_different_action(client, monkeypatch):
    monkeypatch.setattr(client, "_ensure_session", Mock())
    monkeypatch.setattr("sdk.time.sleep", lambda _: None)
    client.http.request.side_effect = [
        response({"action": {"action_id": "original", "state": "running"}}),
        response({"action": {"action_id": "different", "state": "succeeded"}}),
    ]
    with pytest.raises(SDKError, match="lost_track"):
        client.action("motion.move", {})


def test_missing_session_record_is_a_controlled_error(client):
    client.http.request.return_value = response({})
    with pytest.raises(SDKError, match="invalid_response"):
        client._ensure_session()


def test_deeply_nested_json_is_a_controlled_error(client):
    client.http.request.return_value = response(None)
    client.http.request.return_value.json.side_effect = RecursionError("too deep")
    with pytest.raises(SDKError, match="invalid_response"):
        client.capabilities()
