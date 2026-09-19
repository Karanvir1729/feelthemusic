import pytest

from sdk import LampSDK, SDKError


class FakeSDK(LampSDK):
    def __init__(self, responses):
        self.responses = iter(responses)
        self.posts = []

    def _ensure_session(self):
        pass

    def _call(self, method, path, *, json=None, timeout=20):
        if method == "POST" and path == "/api/sdk/v1/actions":
            self.posts.append(json)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def test_lost_post_response_retries_with_the_same_idempotency_key(monkeypatch):
    monkeypatch.setattr("sdk.time.sleep", lambda _: None)
    sdk = FakeSDK([
        SDKError(0, "network", "connection dropped"),
        {"action": {"state": "succeeded", "result": {"reached": True}}},
    ])

    result = sdk.action("motion.move", {"positions": {"base_yaw": 0}})

    assert result["state"] == "succeeded"
    assert len(sdk.posts) == 2
    assert sdk.posts[0]["idempotency_key"] == sdk.posts[1]["idempotency_key"]


def test_second_uncertain_post_failure_escalates_to_lost_track(monkeypatch):
    monkeypatch.setattr("sdk.time.sleep", lambda _: None)
    sdk = FakeSDK([
        SDKError(0, "network", "connection dropped"),
        SDKError(500, "gateway", "response lost again"),
    ])

    with pytest.raises(SDKError, match="lost_track") as raised:
        sdk.action("motion.move", {"positions": {"base_yaw": 0}})

    assert raised.value.status == 409
    assert len(sdk.posts) == 2
    assert sdk.posts[0]["idempotency_key"] == sdk.posts[1]["idempotency_key"]


def test_client_error_is_not_retried():
    sdk = FakeSDK([SDKError(400, "bad_request", "invalid payload")])

    with pytest.raises(SDKError, match="bad_request"):
        sdk.action("motion.move", {"positions": {"base_yaw": 0}})

    assert len(sdk.posts) == 1


@pytest.mark.parametrize("result", [{}, {"reached": False}])
def test_move_requires_an_explicit_reached_confirmation(result):
    sdk = object.__new__(LampSDK)
    sdk.action = lambda *_args, **_kwargs: {"state": "succeeded", "result": result}

    with pytest.raises(SDKError, match="not_reached"):
        sdk.move({"base_yaw": 1.23})


def test_move_returns_only_after_explicit_reached_confirmation():
    sdk = object.__new__(LampSDK)
    sdk.action = lambda *_args, **_kwargs: {"state": "succeeded", "result": {"reached": True}}

    assert sdk.move({"base_yaw": 1.23})["result"]["reached"] is True


def test_play_animation_dispatches_animation_play():
    sdk = object.__new__(LampSDK)
    recorded = []
    sdk.action = lambda cmd, payload, **kwargs: recorded.append((cmd, payload)) or {"state": "succeeded"}
    sdk.play_animation("nod")
    assert recorded == [("animation.play", {"name": "nod"})]


def test_play_clip_dispatches_clip_play():
    sdk = object.__new__(LampSDK)
    recorded = []
    sdk.action = lambda cmd, payload, **kwargs: recorded.append((cmd, payload)) or {"state": "succeeded"}
    sdk.play_clip("clip_42")
    assert recorded == [("clip.play", {"clip_id": "clip_42"})]

