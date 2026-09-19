"""Client for the LeLamp SDK gateway: the lamp's own, vendor-supported control API.

Why the SDK and nothing else. Every motion.move sent here is planned by the lamp's runtime: it checks
reachability and head-against-base self-collision, enforces the velocity limit, eases the move
(minimum 2 s) and waits for the arm to settle. The runtime also has an open, unauthenticated joint
route that skips all of that. We do not use it: it let an early version of our follower sink the arm.

The gateway needs LELAMP_SDK_TOKEN in the runtime's environment (the lamp's .env) and a runtime
restart. The token is a local shared secret: this module reads it and never prints it.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import requests

DEFAULT_BASE = "http://127.0.0.1:8081"
DEFAULT_ENV_FILE = Path.home() / "lelamp-hackathon-2026" / ".env"
DONE = {"succeeded", "failed", "rejected", "canceled"}


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
        r = self.http.request(method, self.base + path, json=json, timeout=timeout)
        try:
            body = r.json()
        except ValueError:
            body = {}
        if r.status_code >= 400 or body.get("ok") is False:
            err = body.get("error") if isinstance(body.get("error"), dict) else {}
            raise SDKError(r.status_code, str(err.get("code", "")), str(err.get("message", r.text[:200])),
                           err.get("details"))
        return body

    def _ensure_session(self) -> None:
        # Sessions expire on the lamp's wall clock (1 h). That clock can jump when the internet
        # comes back, so renew early and also renew on any 401 (see action()).
        if time.time() < self._session_expires - 120:
            return
        session = self._call("POST", "/api/sdk/v1/sessions", json={"app_id": self.app_id})["session"]
        self.http.headers["X-LeLamp-SDK-Session"] = session["session_id"]
        self._session_expires = float(session["expires_at"])

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
        """Yield JPEG frames from the runtime's multipart camera stream."""
        r = self.http.get(f"{self.base}/api/sdk/v1/streams/camera", params={"fps": fps}, stream=True,
                          timeout=(5, 20))
        if r.status_code >= 400:
            raise SDKError(r.status_code, "camera", r.text[:200])
        raw = r.raw
        while True:
            line = raw.readline()
            if not line:
                return
            if not line.startswith(b"--lelamp"):
                continue
            length = 0
            while True:
                header = raw.readline()
                if header in (b"\r\n", b"\n", b""):
                    break
                if header.lower().startswith(b"content-length:"):
                    length = int(header.split(b":", 1)[1])
            data = raw.read(length) if length else b""
            if length and len(data) == length:
                yield data

    # ------------------------------------------------------------------ actions
    def action(self, command_type: str, payload: dict, wait_s: float = 30.0) -> dict:
        """Run one SDK action to completion. Raises SDKError unless it succeeded."""
        self._ensure_session()
        request = {"type": command_type, "payload": payload}
        try:
            action = self._call("POST", "/api/sdk/v1/actions", json=request)["action"]
        except SDKError as exc:
            if exc.status != 401:
                raise
            self._session_expires = 0.0           # session expired or the lamp's clock jumped
            self._ensure_session()
            action = self._call("POST", "/api/sdk/v1/actions", json=request)["action"]
        deadline = time.monotonic() + wait_s
        while action.get("state") not in DONE and time.monotonic() < deadline:
            time.sleep(0.1)
            action = self._call("GET", f"/api/sdk/v1/actions/{action['action_id']}")["action"]
        if action.get("state") != "succeeded":
            err = action.get("error") or {}
            raise SDKError(409, str(err.get("code", action.get("state", "timeout"))),
                           str(err.get("message", "action did not succeed")), err.get("details"))
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
