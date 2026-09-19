"""Client for the LeLamp SDK gateway: the lamp's own, vendor-supported control API.

Why the SDK and nothing else. Every motion.move sent here is planned by the lamp's runtime: it checks
reachability and head-against-base self-collision, enforces the velocity limit, eases the move
(minimum 2 s) and waits for the arm to settle. The runtime also has an open, unauthenticated joint
route that skips all of that. We do not use it: it let an early version of our follower sink the arm.

The gateway needs LELAMP_SDK_TOKEN in the runtime's environment (the lamp's .env) and a runtime
restart. The token is a local shared secret: this module reads it and never prints it.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import requests

DEFAULT_BASE = "http://127.0.0.1:8081"
DEFAULT_ENV_FILE = Path.home() / "lelamp-hackathon-2026" / ".env"
DONE = {"succeeded", "failed", "rejected", "canceled"}
SESSION_CACHE = Path.home() / ".cache" / "feelthemusic-sdk-session.json"


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
        auth = {"Authorization": f"Bearer {token}"}
        self.http = requests.Session()
        self.http.headers.update(auth)
        self.stream_http = requests.Session()          # the camera thread streams on its own connection
        self.stream_http.headers.update(auth)
        self._session_expires = 0.0

    # ------------------------------------------------------------------ plumbing
    def _call(self, method: str, path: str, *, json: dict | None = None, timeout: float = 20) -> dict:
        try:
            r = self.http.request(method, self.base + path, json=json, timeout=timeout)
        except requests.RequestException as exc:      # runtime restarting, Wi-Fi blip, read timeout
            raise SDKError(0, "network", f"{type(exc).__name__}: {exc}") from None
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
        # The gateway allows 16 sessions, keeps each for an hour and has no way to delete one, so
        # a new session per run would lock us out on the 17th run. Reuse one across runs.
        if not self._session_expires:
            try:
                cached = json.loads(SESSION_CACHE.read_text())
                if cached.get("base") == self.base and time.time() < float(cached["expires_at"]) - 120:
                    self.http.headers["X-LeLamp-SDK-Session"] = cached["session_id"]
                    self._session_expires = float(cached["expires_at"])
                    return
            except (OSError, ValueError, KeyError, TypeError):
                pass
        session = self._call("POST", "/api/sdk/v1/sessions", json={"app_id": self.app_id})["session"]
        self.http.headers["X-LeLamp-SDK-Session"] = session["session_id"]
        self._session_expires = float(session["expires_at"])
        try:
            SESSION_CACHE.parent.mkdir(parents=True, exist_ok=True)
            SESSION_CACHE.write_text(json.dumps({"base": self.base, "session_id": session["session_id"],
                                                 "expires_at": session["expires_at"]}))
            SESSION_CACHE.chmod(0o600)
        except OSError:
            pass

    # ------------------------------------------------------------------ reads
    def capabilities(self) -> dict:
        return self._call("GET", "/api/sdk/v1/capabilities")

    def joints(self) -> dict:
        """Limits, units, velocity cap, whether self-collision checking is on, and measured positions."""
        return self._call("GET", "/api/sdk/v1/joints")

    def snapshot(self) -> bytes:
        try:
            r = self.http.get(f"{self.base}/api/sdk/v1/camera/snapshot", timeout=8)
        except requests.RequestException as exc:
            raise SDKError(0, "network", f"{type(exc).__name__}: {exc}") from None
        if r.status_code >= 400:
            raise SDKError(r.status_code, "camera", r.text[:200])
        return r.content

    def camera_frames(self, fps: float = 10):
        """Yield JPEG frames from the runtime's multipart camera stream."""
        try:
            r = self.stream_http.get(f"{self.base}/api/sdk/v1/streams/camera", params={"fps": fps}, stream=True,
                                     timeout=(5, 20))
        except requests.RequestException as exc:
            raise SDKError(0, "network", f"{type(exc).__name__}: {exc}") from None
        try:
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
        finally:
            r.close()

    # ------------------------------------------------------------------ actions
    def action(self, command_type: str, payload: dict, wait_s: float = 30.0) -> dict:
        """Run one SDK action to completion. Raises SDKError unless it succeeded."""
        self._ensure_session()
        # The key makes a re-sent POST (our 401 retry, a flaky connection) return the first action
        # instead of running it twice, and lets a failed POST be matched to its record.
        request = {"type": command_type, "payload": payload, "idempotency_key": uuid.uuid4().hex}
        try:
            action = self._call("POST", "/api/sdk/v1/actions", json=request)["action"]
        except SDKError as exc:
            if exc.status == 401:
                self._session_expires = 0.0           # session expired, runtime restarted, or the lamp's clock jumped
                SESSION_CACHE.unlink(missing_ok=True)  # the cached session is the one that just failed
                self.http.headers.pop("X-LeLamp-SDK-Session", None)
                self._ensure_session()
                action = self._call("POST", "/api/sdk/v1/actions", json=request)["action"]
            elif exc.status in (0, 500):
                # The answer was lost (timeout, connection drop, a gateway hiccup), but the lamp may have
                # accepted the command. Error replies never carry the action id, so the only safe recovery
                # is the same idempotency key: it returns the first record and cannot run the move twice.
                time.sleep(1.0)
                try:
                    action = self._call("POST", "/api/sdk/v1/actions", json=request)["action"]
                except SDKError as again:
                    raise SDKError(409, "lost_track", f"do not know whether the lamp accepted this: {again}") from None
            else:
                raise
        estimate = (action.get("result") or {}).get("estimated_duration_seconds")
        if isinstance(estimate, (int, float)):
            wait_s = min(wait_s, float(estimate) + 1.5 + 4.0)      # planned duration + settle + margin
        deadline = time.monotonic() + wait_s
        hiccups = 0
        while action.get("state") not in DONE and time.monotonic() < deadline:
            time.sleep(0.2)
            try:
                action = self._call("GET", f"/api/sdk/v1/actions/{action['action_id']}")["action"]
                hiccups = 0
            except SDKError as exc:
                # A status poll can fail while the move is still running (the gateway's bookkeeping is
                # not thread-safe, the network can blip). That is NOT a failed move: keep polling.
                hiccups += 1
                if exc.status in (401, 404) or hiccups > 10:
                    raise SDKError(409, "lost_track", f"could not follow the action: {exc}") from None
        if action.get("state") not in DONE:             # still running after wait_s: do not abandon it
            try:
                self._call("POST", f"/api/sdk/v1/actions/{action['action_id']}/cancel", timeout=8)
            except SDKError:
                pass
            raise SDKError(409, "timeout", f"action still {action.get('state')} after {wait_s:.0f} s; cancel requested")
        if action.get("state") != "succeeded":          # status 409 = the gateway accepted it, then it did not succeed
            err = action.get("error") or {}
            raise SDKError(409, str(err.get("code", action.get("state", "timeout"))),
                           str(err.get("message", "action did not succeed")), err.get("details"))
        return action

    def move(self, positions: dict[str, float]) -> dict:
        """A safe, planned move. Always pass ALL joints: a joint left out is held at its *measured*
        position, and on a gravity-loaded joint that is a little lower every time."""
        action = self.action("motion.move", {"positions": {k: round(float(v), 1) for k, v in positions.items()}})
        result = action.get("result") or {}
        if result.get("reached") is not True:
            raise SDKError(409, "not_reached", "the move ended without reaching its target", result)
        return action

    def glow(self, rgb: tuple[int, int, int], luminance: float | None = None) -> dict:
        payload: dict = {"color": [int(c) for c in rgb]}
        if luminance is not None:
            payload["luminance"] = float(luminance)
        return self.action("light.glow", payload, wait_s=8)

    # There is deliberately no stop(). The vendor's system.stop stops motion AND releases torque over
    # 0.6 s: the arm goes limp and the head falls onto the table. To stop following, stop sending
    # moves: the runtime finishes the current planned move on its own.
