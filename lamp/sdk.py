"""Client for the LeLamp SDK gateway: the lamp's own, vendor-supported control API.

Why the SDK and nothing else. Every motion.move sent here is planned by the lamp's runtime: it checks
reachability and head-against-base self-collision, enforces the velocity limit, eases the move
(minimum 2 s) and waits for the arm to settle. The runtime also has an open, unauthenticated joint
route that skips all of that. We do not use it: it let an early version of our follower sink the arm.

The gateway needs LELAMP_SDK_TOKEN in the runtime's environment (the lamp's .env) and a runtime
restart. The token is a local shared secret: this module reads it and never prints it.
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path
from urllib.parse import quote

import requests

DEFAULT_BASE = "http://127.0.0.1:8081"
DEFAULT_ENV_FILE = Path.home() / "lelamp-hackathon-2026" / ".env"
DONE = {"succeeded", "failed", "rejected", "canceled"}
# Resource ceilings, not claims about image resolution or robot safety.
MAX_FRAME_BYTES = 4 * 1024 * 1024
MAX_HEADER_LINE_BYTES = 8192
MAX_FRAME_HEADER_BYTES = 64 * 1024


class SDKError(RuntimeError):
    def __init__(self, status: int, code: str, message: str, details: dict | None = None):
        super().__init__(f"{code or status}: {message}")
        self.status, self.code, self.message, self.details = status, code, message, details or {}


def read_token(env_file: Path = DEFAULT_ENV_FILE) -> str:
    token = os.environ.get("LELAMP_SDK_TOKEN", "").strip()
    if not token and Path(env_file).exists():
        for line in Path(env_file).read_text().splitlines():
            if line.startswith("LELAMP_SDK_TOKEN="):
                token = line.split("=", 1)[1].strip().strip("\"'")
    if not token:
        raise SDKError(0, "no_token", f"LELAMP_SDK_TOKEN is not set in the environment or in {env_file}")
    return token


class LampSDK:
    def __init__(self, token: str, base: str = DEFAULT_BASE, app_id: str = "feel-the-music"):
        self.base, self.app_id = base.rstrip("/"), app_id
        self.http = requests.Session()
        self.http.headers["Authorization"] = f"Bearer {token}"
        self._session_expires = 0.0

    # ------------------------------------------------------------------ plumbing
    def _call(self, method: str, path: str, *, json: dict | None = None, timeout: float = 20) -> dict:
        try:
            r = self.http.request(method, self.base + path, json=json, timeout=timeout)
        except requests.RequestException as exc:
            raise SDKError(0, "network", type(exc).__name__) from None
        try:
            body = r.json()
        except (ValueError, RecursionError):
            raise SDKError(r.status_code, "invalid_response", "gateway returned invalid JSON") from None
        if not isinstance(body, dict) or ("ok" in body and type(body["ok"]) is not bool):
            raise SDKError(r.status_code, "invalid_response", "gateway response must be an object with boolean ok")
        if r.status_code >= 400 or body.get("ok") is False:
            error = body.get("error")
            err = error if isinstance(error, dict) else {}
            raise SDKError(r.status_code, str(err.get("code", "")), str(err.get("message", r.text[:200])),
                           err.get("details"))
        return body

    def _ensure_session(self) -> None:
        # Sessions expire on the lamp's wall clock (1 h). That clock can jump when the internet
        # comes back, so renew early and also renew on any 401 (see action()).
        if time.time() < self._session_expires - 120:
            return
        session = self._call("POST", "/api/sdk/v1/sessions", json={"app_id": self.app_id}).get("session")
        if not isinstance(session, dict) or not isinstance(session.get("session_id"), str) or not session["session_id"]:
            raise SDKError(0, "invalid_response", "gateway returned an invalid session")
        try:
            expiry = float(session["expires_at"])
        except (KeyError, TypeError, ValueError, OverflowError):
            raise SDKError(0, "invalid_response", "gateway returned an invalid session expiry") from None
        if isinstance(session["expires_at"], bool) or not math.isfinite(expiry) or expiry <= 0:
            raise SDKError(0, "invalid_response", "gateway returned an invalid session expiry")
        self.http.headers["X-LeLamp-SDK-Session"] = session["session_id"]
        self._session_expires = expiry

    # ------------------------------------------------------------------ reads
    def capabilities(self) -> dict:
        return self._call("GET", "/api/sdk/v1/capabilities")

    def joints(self) -> dict:
        """Limits, units, velocity cap, whether self-collision checking is on, and measured positions."""
        return self._call("GET", "/api/sdk/v1/joints")

    def snapshot(self) -> bytes:
        r = self.http.get(f"{self.base}/api/sdk/v1/camera/snapshot", timeout=8)
        if r.status_code >= 400:
            raise SDKError(r.status_code, "camera", r.text[:200])
        return r.content

    def camera_frames(self, fps: float = 10):
        """Yield bounded JPEG frames; close the generator when abandoning a stream."""
        r = self.http.get(f"{self.base}/api/sdk/v1/streams/camera", params={"fps": fps}, stream=True,
                          timeout=(5, 20))
        try:
            if r.status_code >= 400:
                raise SDKError(r.status_code, "camera", "camera stream request failed")
            raw = r.raw
            header_bytes = 0

            def read_line():
                nonlocal header_bytes
                line = raw.readline(MAX_HEADER_LINE_BYTES + 1)
                header_bytes += len(line)
                if len(line) > MAX_HEADER_LINE_BYTES or header_bytes > MAX_FRAME_HEADER_BYTES:
                    raise SDKError(0, "camera", "camera header exceeds resource limit")
                return line

            while True:
                line = read_line()
                if not line or line.strip() == b"--lelamp--":
                    return
                if line.strip() != b"--lelamp":
                    continue
                length = None
                while True:
                    header = read_line()
                    if not header:
                        raise SDKError(0, "camera", "truncated camera headers")
                    if header in (b"\r\n", b"\n"):
                        break
                    if header.lower().startswith(b"content-length:"):
                        value = header.split(b":", 1)[1].strip()
                        if length is not None or not value.isdigit() or len(value) > 10:
                            raise SDKError(0, "camera", "invalid camera content length")
                        length = int(value)
                        if not 0 < length <= MAX_FRAME_BYTES:
                            raise SDKError(0, "camera", "camera frame exceeds resource limit")
                if length is None:
                    raise SDKError(0, "camera", "missing camera content length")
                data = raw.read(length)
                if len(data) != length:
                    raise SDKError(0, "camera", "truncated camera frame")
                header_bytes = 0
                yield data
        finally:
            r.close()

    # ------------------------------------------------------------------ actions
    def action(self, command_type: str, payload: dict, wait_s: float = 30.0) -> dict:
        """Run one SDK action; wait_s bounds polling after submission, not HTTP total time.

        Each poll uses at most the remaining budget as its socket timeout.
        Expiry means the outcome is unknown, not that the robot stopped.
        """
        try:
            valid_wait = type(wait_s) in (int, float) and math.isfinite(wait_s) and wait_s > 0
        except OverflowError:
            valid_wait = False
        if not valid_wait:
            raise ValueError("wait_s must be a finite positive number")
        self._ensure_session()
        request = {"type": command_type, "payload": payload}
        try:
            action = self._call_action("POST", "/api/sdk/v1/actions", json=request)
        except SDKError as exc:
            if exc.status != 401:
                raise
            self._session_expires = 0.0           # session expired or the lamp's clock jumped
            self._ensure_session()
            action = self._call_action("POST", "/api/sdk/v1/actions", json=request)
        deadline = time.monotonic() + wait_s
        while action["state"] not in DONE:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SDKError(409, "timeout", "action outcome unknown after polling deadline")
            time.sleep(min(0.1, remaining))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SDKError(409, "timeout", "action outcome unknown after polling deadline")
            action_id = action["action_id"]
            action = self._call_action("GET", f"/api/sdk/v1/actions/{quote(action_id, safe='')}",
                                       timeout=min(20, remaining))
            if action["action_id"] != action_id:
                raise SDKError(409, "lost_track", "gateway returned a different action while polling")
        if action.get("state") != "succeeded":
            err = action.get("error") or {}
            raise SDKError(409, str(err.get("code", action.get("state", "timeout"))),
                           str(err.get("message", "action did not succeed")), err.get("details"))
        return action

    def _call_action(self, method: str, path: str, *, json: dict | None = None, timeout: float = 20) -> dict:
        try:
            body = self._call(method, path, json=json, timeout=timeout)
        except SDKError as exc:
            if exc.status == 0 or (exc.code == "invalid_response" and exc.status < 400):
                # A malformed/lost response is not proof the action was refused.
                # Preserve the existing caller contract: 409 means stop issuing moves.
                raise SDKError(409, "lost_track", "action outcome unknown: invalid or lost gateway response") from None
            raise
        action = body.get("action")
        if (not isinstance(action, dict)
                or not isinstance(action.get("state"), str)
                or not action["state"]
                or not isinstance(action.get("action_id"), str) or not action["action_id"]
                or (action.get("error") is not None and not isinstance(action["error"], dict))):
            raise SDKError(409, "invalid_response", "action outcome unknown: malformed gateway record")
        return action

    def move(self, positions: dict[str, float]) -> dict:
        """A safe, planned move. Always pass ALL joints: a joint left out is held at its *measured*
        position, and on a gravity-loaded joint that is a little lower every time."""
        return self.action("motion.move", {"positions": {k: round(float(v), 2) for k, v in positions.items()}})

    def glow(self, rgb: tuple[int, int, int], luminance: float | None = None) -> dict:
        payload: dict = {"color": [int(c) for c in rgb]}
        if luminance is not None:
            payload["luminance"] = float(luminance)
        return self.action("light.glow", payload, wait_s=8)

    def stop(self) -> dict:
        return self.action("system.stop", {"reason": "feel-the-music"}, wait_s=8)
