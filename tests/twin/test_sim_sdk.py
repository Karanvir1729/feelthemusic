"""The simulated SDK gateway (twin/sim_sdk.py, twin/sim_light.py), exercised over HTTP with plain requests and
with the team's real client class, lamp/sdk.py LampSDK from origin/gemini38/lamp-follower-music (loaded with
git show into a temp dir; those tests skip when git or the ref is unavailable).

Most tests use a manual clock: simulated time moves only when the test advances it, so "the move took at
least 2 s" and "the glow blocked for 600 ms" are exact statements about simulated time, not wall time.
"""
from __future__ import annotations

import importlib.util
import io
import json
import math
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest

try:                                  # only the HTTP tests need requests: they skip without it, the rest run
    import requests
except ImportError:
    requests = None

from twin.contract import JOINTS, Person  # noqa: E402
from twin.sim_light import SimLight  # noqa: E402
from twin.sim_sdk import (HAIR, SKIN, MissingCollisionModel, SimGateway, SimLamp, SimSDKServer,  # noqa: E402
                          build, main, read_motion_csv, robot_dir, split_reason,
                          synthetic_frame)

ROOT = Path(__file__).resolve().parents[2]
TOKEN = "sim-token"
TEAM_REF = os.environ.get("FTM_TEAM_SDK_REF", "origin/gemini38/lamp-follower-music")
# The sim needs the robot package for its self-collision check; without it these tests run the opted-in
# degraded mode and skip what depends on the check.
COLLISION = "required" if robot_dir() is not None else "off"
HEADER = "timestamp," + ",".join(f"{j}.pos" for j in JOINTS)


# ------------------------------------------------------------------ fixtures and helpers
class Sim:
    def __init__(self, lamp: SimLamp, gw: SimGateway, server: SimSDKServer):
        self.lamp, self.gw, self.server = lamp, gw, server
        self.base = server.base_url
        self.auth = {"Authorization": f"Bearer {TOKEN}"}
        self.headers = dict(self.auth)

    def url(self, path: str) -> str:
        return self.base + path

    def session(self) -> str:
        r = requests.post(self.url("/api/sdk/v1/sessions"), json={"app_id": "test"}, headers=self.auth, timeout=5)
        assert r.status_code == 201, r.text
        sid = r.json()["session"]["session_id"]
        self.headers["X-LeLamp-SDK-Session"] = sid
        return sid

    def post(self, command: str, payload: dict | None = None, **extra):
        body = {"command_type": command, "payload": payload or {}, **extra}
        return requests.post(self.url("/api/sdk/v1/actions"), json=body, headers=self.headers, timeout=30)

    def action(self, action_id: str) -> dict:
        r = requests.get(self.url(f"/api/sdk/v1/actions/{action_id}"), headers=self.headers, timeout=5)
        assert r.status_code == 200, r.text
        return r.json()["action"]

    def joints(self) -> dict:
        r = requests.get(self.url("/api/sdk/v1/joints"), headers=self.auth, timeout=5)
        assert r.status_code == 200, r.text
        return r.json()

    def run_until_state(self, action_id: str, states, max_s: float = 10.0) -> dict:
        states = {states} if isinstance(states, str) else set(states)
        self.lamp.run_until(lambda: self.gw.actions[action_id].state in states, max_s=max_s)
        return self.action(action_id)


@pytest.fixture
def make_sim():
    if requests is None:
        pytest.skip("requests is not installed (uv run --with requests): the HTTP tests need it")
    made = []

    def make(*, clock: str = "manual", rate_limit: int = 120, motion=None, scenario=None,
             transition_ms: float = 600.0, **kw) -> Sim:
        kw.setdefault("collision", COLLISION)
        kw.setdefault("idle", "off")                     # the gateway tests want an arm that stays put
        lamp = SimLamp(clock=clock, motion=motion, scenario=scenario,
                       transition_ms=transition_ms, **kw)
        gw = SimGateway(lamp, token=TOKEN, rate_limit=rate_limit)
        server = SimSDKServer(gw, port=0).start()
        made.append(server)
        return Sim(lamp, gw, server)

    yield make
    for server in made:
        server.stop()


@pytest.fixture
def sim(make_sim) -> Sim:
    s = make_sim()
    s.session()
    return s


class Advancer:
    """Moves a manual clock forward in the background while a real-time client (LampSDK polls every 0.2 s of
    real time) waits: dt simulated seconds every `pause` real seconds."""

    def __init__(self, lamp: SimLamp, dt: float = 0.05, pause: float = 0.004):
        self.lamp, self.dt, self.pause = lamp, dt, pause
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.lamp.advance(self.dt)
            time.sleep(self.pause)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)


@pytest.fixture(scope="module")
def team_sdk(tmp_path_factory):
    """The team's LampSDK module, from git, written to a temp dir (never into the repo)."""
    if requests is None:
        pytest.skip("requests is not installed (uv run --with requests): the team client needs it")
    if shutil.which("git") is None:
        pytest.skip("git not available")
    try:
        source = subprocess.run(["git", "-C", str(ROOT), "show", f"{TEAM_REF}:lamp/sdk.py"],
                                capture_output=True, check=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        pytest.skip(f"{TEAM_REF}:lamp/sdk.py is not available")
    directory = tmp_path_factory.mktemp("team_sdk")
    path = directory / "team_lamp_sdk.py"
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location("team_lamp_sdk", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SESSION_CACHE = directory / "session-cache.json"     # never touch the user's ~/.cache
    return module


def client_for(team_sdk, sim: Sim, token: str = TOKEN):
    team_sdk.SESSION_CACHE.unlink(missing_ok=True)
    return team_sdk.LampSDK(token, base=sim.base, app_id="twin-test")


def all_joints(pose: dict, **changes) -> dict:
    out = {j: round(float(pose[j]), 2) for j in JOINTS}
    out.update(changes)
    return out


def clip_csv(start: dict, *, seconds: float = 2.0, yaw_swing: float = 15.0, fps: float = 30.0) -> bytes:
    rows = ["timestamp," + ",".join(f"{j}.pos" for j in JOINTS)]
    n = int(seconds * fps) + 1
    for i in range(n):
        t = i / fps
        pose = dict(start)
        pose["base_yaw"] = start["base_yaw"] + yaw_swing * math.sin(math.pi * t / seconds)
        rows.append(f"{100.0 + t:.6f}," + ",".join(f"{pose[j]:.4f}" for j in JOINTS))   # absolute stamps work too
    return ("\n".join(rows) + "\n").encode()


def decode_jpeg(data: bytes, tmp_path: Path) -> np.ndarray:
    """An independent decoder: Pillow when installed, else macOS sips to BMP."""
    try:
        from PIL import Image
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    except ImportError:
        pass
    if shutil.which("sips") is None:
        pytest.skip("no independent JPEG decoder (Pillow or macOS sips)")
    src, dst = tmp_path / "frame.jpg", tmp_path / "frame.bmp"
    src.write_bytes(data)
    subprocess.run(["sips", "-s", "format", "bmp", str(src), "--out", str(dst)], check=True, capture_output=True)
    b = dst.read_bytes()
    offset = int.from_bytes(b[10:14], "little")
    width = int.from_bytes(b[18:22], "little", signed=True)
    height = int.from_bytes(b[22:26], "little", signed=True)
    bpp = int.from_bytes(b[28:30], "little")
    row = ((bpp * width + 31) // 32) * 4
    rows = np.frombuffer(b[offset:offset + row * abs(height)], np.uint8).reshape(abs(height), row)
    pixels = rows[:, :width * (bpp // 8)].reshape(abs(height), width, bpp // 8)[:, :, 2::-1]
    return pixels[::-1] if height > 0 else pixels


class OnePerson:
    """A scenario with one seated person straight in front of the lamp, facing it."""

    def people_at(self, t):
        return [Person("p1", np.array([0.0, 0.75, 0.42]), np.array([0.0, -1.0, 0.0]))]


# ------------------------------------------------------------------ session, capabilities, joints, auth
def test_session_capabilities_and_joints(sim):
    r = requests.post(sim.url("/api/sdk/v1/sessions"), json={"app_id": "feel-the-music", "metadata": {"k": 1}},
                      headers=sim.auth, timeout=5)
    assert r.status_code == 201
    session = r.json()["session"]
    assert r.json()["protocol_version"] == "lelamp.sdk.v1"
    assert session["session_id"].startswith("sdk_sess_") and len(session["session_id"]) == 25
    assert session["expires_at"] - session["created_at"] == pytest.approx(3600.0)
    assert session["metadata"] == {"k": 1} and session["expired"] is False
    bad = requests.post(sim.url("/api/sdk/v1/sessions"), json={}, headers=sim.auth, timeout=5)
    assert bad.status_code == 400 and bad.json()["error"]["message"] == "a session needs an app_id"

    caps = requests.get(sim.url("/api/sdk/v1/capabilities"), headers=sim.auth, timeout=5).json()
    names = [c["name"] for c in caps["capabilities"]]
    assert names == ["attention.nudge", "light.glow", "motion.move", "clip.play", "torque.enable", "torque.release",
                     "music.play", "sound.play", "speech.say", "scenario.run", "animation.play", "system.stop",
                     "status.read"]
    by_name = {c["name"]: c for c in caps["capabilities"]}
    assert by_name["attention.nudge"] == {"name": "attention.nudge", "available": False,
                                          "reason": "turned off by the runtime's policy"}
    assert by_name["motion.move"]["available"] is True and "reason" not in by_name["motion.move"]
    assert caps["resources"]["joints"]["units"] == "normalized_m100_100"
    assert caps["resources"]["streams"] == {"transport": "http-multipart", "max_concurrent": 4,
                                            "idle_timeout_seconds": 15}
    assert "research" not in caps and "scenario_handoff" in caps["forbidden"]

    joints = sim.joints()
    assert joints["ok"] is True and joints["robot_id"] == "lelamp_v1_pi5_feetech_r1"
    assert joints["units"] == "normalized_m100_100" and joints["max_velocity_units_s"] == 300.0
    assert joints["joints"] == {j: {"range_min": -100.0, "range_max": 100.0} for j in JOINTS}
    assert len(joints["calibration_id"]) == 64 and len(joints["device_calibration"]) == 64
    assert joints["target_tolerance"] == 2.0
    assert joints["target_tolerances"] == {"elbow_pitch": 10.0, "wrist_pitch": 3.0}
    assert joints["self_collision_check"] is (sim.lamp.pose_ok is not None)
    assert set(joints["positions"]) == set(JOINTS)
    with sim.lamp.lock:
        assert joints["positions"] == sim.lamp.measured()                  # MEASURED, not commanded


def test_token_and_session_errors(sim, team_sdk):
    path = sim.url("/api/sdk/v1/capabilities")
    assert requests.get(path, timeout=5).status_code == 401
    wrong = requests.get(path, headers={"Authorization": "Bearer nope"}, timeout=5)
    assert wrong.status_code == 401
    assert wrong.json() == {"ok": False,
                            "error": {"code": "unauthorized", "message": "the SDK token is missing or wrong"}}
    assert requests.get(path, headers={"X-LeLamp-SDK-Token": TOKEN}, timeout=5).status_code == 200
    assert requests.get(path, headers={"Authorization": f"bearer {TOKEN}"}, timeout=5).status_code == 200
    # Authorization wins over X-LeLamp-SDK-Token, even when it is the wrong one (routes/sdk.py:107-112)
    both = requests.get(path, headers={"Authorization": "x", "X-LeLamp-SDK-Token": TOKEN}, timeout=5)
    assert both.status_code == 401
    no_session = requests.post(sim.url("/api/sdk/v1/actions"), json={"command_type": "status.read"},
                               headers=sim.auth, timeout=5)
    assert no_session.status_code == 401
    assert no_session.json()["error"]["message"] == "a valid session_id is needed"
    sdk = client_for(team_sdk, sim, token="wrong-token")
    with pytest.raises(team_sdk.SDKError) as exc:
        sdk.capabilities()
    assert exc.value.status == 401 and exc.value.code == "unauthorized"


def test_team_client_reads(sim, team_sdk):
    sdk = client_for(team_sdk, sim)
    caps = sdk.capabilities()
    assert caps["resources"]["joints"]["self_collision_check"] is (sim.lamp.pose_ok is not None)
    info = sdk.joints()
    assert set(info["positions"]) == set(JOINTS)
    status = sdk.action("status.read", {})
    assert status["state"] == "succeeded"
    assert status["result"]["status"]["gateway"]["enabled"] is True
    assert status["result"]["sdk_action_id"] == status["action_id"]


# ------------------------------------------------------------------ motion
def test_move_succeeds_after_two_seconds_with_team_client(sim, team_sdk):
    sdk = client_for(team_sdk, sim)
    start = sim.joints()["positions"]
    target = all_joints(start, base_yaw=round(start["base_yaw"] + 20.0, 2))
    with Advancer(sim.lamp):
        action = sdk.move(target)
    assert action["state"] == "succeeded"
    result = action["result"]
    assert result["completed"] is True and result["reached"] is True and result["reachable"] is True
    assert result["estimated_duration_seconds"] == pytest.approx(2.0)
    assert action["updated_at"] - action["created_at"] >= 2.0          # simulated seconds
    assert set(result["position_errors"]) == set(JOINTS)
    assert all(result["position_errors"][j] <= result["position_tolerances"][j] for j in JOINTS)
    assert sim.joints()["positions"]["base_yaw"] == pytest.approx(target["base_yaw"], abs=2.0)


def test_move_states_accepted_running_succeeded(sim):
    start = sim.joints()["positions"]
    t0 = sim.lamp.now()
    r = sim.post("motion.move", {"positions": all_joints(start, base_yaw=start["base_yaw"] - 30)})
    assert r.status_code == 202
    action = r.json()["action"]
    assert action["state"] == "accepted" and action["expires_at"] - action["created_at"] == pytest.approx(900.0)
    assert action["result"] == {"estimated_duration_seconds": 2.0, "reachable": True,
                                "collision_checked": sim.lamp.pose_ok is not None}
    sim.lamp.advance(1.5)
    mid = sim.action(action["action_id"])
    assert mid["state"] == "running" and mid["updated_at"] == action["updated_at"]   # updated only when terminal
    done = sim.run_until_state(action["action_id"], "succeeded")
    with sim.lamp.lock:
        finished_at = sim.lamp.motion.actions[sim.gw.actions[action["action_id"]].motion_id].t_finished
    assert finished_at - t0 >= 2.0
    assert done["result"]["duration_seconds"] == pytest.approx(2.0)


def test_preemption_cancels_and_replans_from_measured(sim):
    start = sim.joints()["positions"]
    first = sim.post("motion.move", {"positions": all_joints(start, base_yaw=start["base_yaw"] + 40)}).json()["action"]
    sim.lamp.advance(1.0)                                                 # half way along the first move
    second = sim.post("motion.move", {"positions": all_joints(start, base_yaw=start["base_yaw"] - 10)})
    assert second.status_code == 202
    second = second.json()["action"]
    sim.lamp.advance(0.3)
    old = sim.action(first["action_id"])
    assert old["state"] == "canceled" and old["result"] == {"reason": "cancelled_or_preempted"}
    with sim.lamp.lock:
        detail = sim.lamp.motion.actions[sim.gw.actions[second["action_id"]].motion_id]
        replan_yaw = detail.replan_from["base_yaw"]
        first_waypoint = detail.plan_positions[0]
    # the new plan starts where the arm WAS (measured, mid-path), not at the old target and not at the start
    assert start["base_yaw"] + 5 < replan_yaw < start["base_yaw"] + 35
    # twin/motion.py writes a live frame (a third pose read) first; the plan proper starts at the re-plan read
    assert detail.plan_positions[1][JOINTS.index("base_yaw")] == pytest.approx(replan_yaw, abs=1e-9)
    assert first_waypoint[JOINTS.index("base_yaw")] == pytest.approx(replan_yaw, abs=0.5)
    done = sim.run_until_state(second["action_id"], "succeeded")
    assert done["result"]["reached"] is True


def test_cancel(sim):
    start = sim.joints()["positions"]
    action = sim.post("motion.move", {"positions": all_joints(start, base_yaw=start["base_yaw"] + 30)}).json()["action"]
    sim.lamp.advance(0.8)
    url = sim.url(f"/api/sdk/v1/actions/{action['action_id']}/cancel")
    r = requests.post(url, headers=sim.auth, timeout=5)
    assert r.status_code == 200
    assert r.json()["action"]["state"] == "canceled"
    assert r.json()["action"]["result"] == {"reason": "cancelled_or_preempted"}
    held = sim.joints()["positions"]["base_yaw"]
    sim.lamp.advance(2.0)
    # holds the last goal written, no return home (the servo catches up with that goal first)
    assert sim.joints()["positions"]["base_yaw"] == pytest.approx(held, abs=1.5)
    again = requests.post(url, headers=sim.auth, timeout=5)
    assert again.status_code == 409
    assert again.json()["error"] == {"code": "action_failed", "message": "SDK action is already terminal: canceled.",
                                     "details": {"action_id": action["action_id"], "state": "canceled"}}
    missing = requests.post(sim.url("/api/sdk/v1/actions/sdk_act_0000000000000000/cancel"), headers=sim.auth, timeout=5)
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "action_not_found"
    assert requests.get(sim.url("/api/sdk/v1/actions/nope"), headers=sim.auth, timeout=5).status_code == 404


def test_motion_refusals_on_the_post(sim):
    start = sim.joints()["positions"]
    extra = sim.post("motion.move", {"positions": all_joints(start), "speed": 2})
    assert extra.status_code == 422
    assert extra.json()["error"] == {"code": "invalid_request", "message": "motion.move needs positions",
                                     "details": {"reachable": False, "collision_checked": False}}
    duration = sim.post("motion.move", {"positions": all_joints(start), "duration_seconds": 1.5})
    assert duration.json()["error"]["message"] == "motion.move takes no duration: the runtime picks it"
    far = sim.post("motion.move", {"positions": {"base_yaw": 100.5}})
    assert far.status_code == 422
    assert far.json()["error"]["message"] == "base_yaw is out of its calibrated range"
    unknown = sim.post("motion.move", {"positions": {"neck": 1}})
    assert unknown.json()["error"]["message"] == "no joint named neck"
    if sim.lamp.pose_ok is not None:
        # arm folded forward and down: the head capsule would touch the base (safe-motion.md section 3.3)
        folded = sim.post("motion.move", {"positions": all_joints(start, base_pitch=95.0, elbow_pitch=-20.0)})
        assert folded.status_code == 422
        error = folded.json()["error"]
        assert "head could hit the base" in error["message"]
        assert error["details"] == {"reachable": False, "collision_checked": True}


def test_idempotent_replay(sim):
    start = sim.joints()["positions"]
    body = {"positions": all_joints(start, base_yaw=start["base_yaw"] + 10)}
    first = sim.post("motion.move", body, idempotency_key="k-1")
    assert first.status_code == 202
    replay = sim.post("motion.move", {"positions": all_joints(start, base_yaw=-50)}, idempotency_key="k-1")
    assert replay.status_code == 200 and replay.json()["idempotent"] is True
    assert replay.json()["action"]["action_id"] == first.json()["action"]["action_id"]
    assert sum(1 for r in sim.gw.actions.values() if r.command_type == "motion.move") == 1


# ------------------------------------------------------------------ clips
def test_clip_upload_and_play_with_team_client(sim, team_sdk):
    sdk = client_for(team_sdk, sim)
    start = sim.joints()["positions"]
    csv_bytes = clip_csv(start)
    up = requests.post(sim.url("/api/sdk/v1/clips"), data=csv_bytes, headers=sim.auth, timeout=10)
    assert up.status_code == 200, up.text
    clip = up.json()["clip"]
    assert len(clip["id"]) == 32 and set(clip) == {"id", "robot_id", "calibration_id", "duration_seconds", "sha256"}
    assert clip["duration_seconds"] == pytest.approx(2.0)
    assert clip["calibration_id"] == sim.joints()["calibration_id"]
    listed = requests.get(sim.url("/api/sdk/v1/clips"), headers=sim.auth, timeout=5).json()["clips"]
    assert [c["id"] for c in listed] == [clip["id"]]
    with Advancer(sim.lamp):
        if hasattr(sdk, "play_clip"):
            action = sdk.play_clip(clip["id"])
        else:                                                              # origin/main's older client
            action = sdk.action("clip.play", {"clip_id": clip["id"]})
    assert action["state"] == "succeeded"
    result = action["result"]
    assert result["completed"] is True and "reached" not in result          # clips have no settle check
    assert result["estimated_duration_seconds"] >= 4.0                       # >= 2 s entry + the 2 s clip
    gone = requests.delete(sim.url(f"/api/sdk/v1/clips/{clip['id']}"), headers=sim.auth, timeout=5)
    assert gone.json() == {"ok": True, "deleted": clip["id"]}
    replay = sim.post("clip.play", {"clip_id": clip["id"]})
    assert replay.status_code == 422 and replay.json()["error"]["message"] == "Clip not found"
    bad_id = requests.delete(sim.url("/api/sdk/v1/clips/NOT-AN-ID"), headers=sim.auth, timeout=5)
    assert bad_id.status_code == 422 and bad_id.json()["error"]["message"] == "Invalid clip ID"


def test_clip_refused_over_speed_cap_and_bad_csv(sim):
    start = sim.joints()["positions"]
    header = "timestamp," + ",".join(f"{j}.pos" for j in JOINTS)
    def row(t, yaw):
        return f"{t}," + ",".join(f"{(yaw if j == 'base_yaw' else start[j]):.3f}" for j in JOINTS)

    fast = f"{header}\n{row(0.0, 0.0)}\n{row(1 / 30, 20.0)}\n{row(2 / 30, 20.0)}\n".encode()   # 600 units/s
    r = requests.post(sim.url("/api/sdk/v1/clips"), data=fast, headers=sim.auth, timeout=5)
    assert r.status_code == 422
    assert r.json()["error"] == {"code": "invalid_request",
                                 "message": "base_yaw would move faster than the velocity limit"}
    cases = {
        b"timestamp,base_yaw.pos\n0,0\n1,0\n": "the CSV header must be timestamp and the five <joint>.pos columns",
        f"{header}\n{row(0.0, 0.0)}\n".encode(): "a clip needs 2 to 18000 frames",
        f"{header}\n{row(0.0, 0.0)}\n{row(0.0, 1.0)}\n".encode(): "timestamps do not increase",
        f"{header}\n{row(0.0, 0.0)}\n\n{row(1.0, 1.0)}\n".encode():
            "a CSV row has the wrong number of cells, or there are too many rows",
    }
    for body, message in cases.items():
        r = requests.post(sim.url("/api/sdk/v1/clips"), data=body, headers=sim.auth, timeout=5)
        assert (r.status_code, r.json()["error"]["message"]) == (422, message)
    with pytest.raises(ValueError, match="out of its calibrated range"):
        read_motion_csv(f"{header}\n{row(0.0, 0.0)}\n{row(1.0, 101.0)}\n".encode())


def test_clip_header_is_compared_as_written(sim):
    """Conformance finding: the sim stripped the header cells, so 'timestamp, base_yaw.pos, ...' (a space
    after each comma, as hand-written and many CSV writers produce) got a clip id here and a 422 on the lamp,
    whose parser compares the raw cells (control/safe_motion.py:73-78)."""
    start = sim.joints()["positions"]
    good = clip_csv(start).decode()
    spaced = good.replace(",", ", ", HEADER.count(","))                  # only the header line gets spaces
    assert spaced.splitlines()[0] == "timestamp, " + ", ".join(f"{j}.pos" for j in JOINTS)
    r = requests.post(sim.url("/api/sdk/v1/clips"), data=spaced.encode(), headers=sim.auth, timeout=5)
    assert r.status_code == 422
    assert r.json() == {"ok": False, "error": {"code": "invalid_request", "message":
                                               "the CSV header must be timestamp and the five <joint>.pos columns"}}
    assert requests.get(sim.url("/api/sdk/v1/clips"), headers=sim.auth, timeout=5).json()["clips"] == []
    trailing = good.replace(HEADER, HEADER + " ", 1)                       # a stray space at the end is refused too
    assert requests.post(sim.url("/api/sdk/v1/clips"), data=trailing.encode(), headers=sim.auth,
                         timeout=5).status_code == 422
    # spaces inside a VALUE are fine: float() takes them, on the lamp too (safe_motion.py:18-27)
    lines = good.splitlines()
    lines[1] = " " + lines[1].replace(",", " , ")
    ok = requests.post(sim.url("/api/sdk/v1/clips"), data=("\n".join(lines) + "\n").encode(), headers=sim.auth,
                       timeout=5)
    assert ok.status_code == 200, ok.text


def test_clip_errors_come_in_the_vendors_order():
    """The first problem the lamp names is the one the sim names (safe_motion.py:48-106): rows are range
    checked while parsing, before any timing check; a stamp before the first one is a duration error."""
    base = {j: 0.0 for j in JOINTS}

    def row(t, **pose):
        p = {**base, **pose}
        return f"{t}," + ",".join(str(p[j]) for j in JOINTS)

    earlier = f"{HEADER}\n{row(1.0)}\n{row(0.5)}\n".encode()               # goes back in time
    with pytest.raises(ValueError, match="^the clip is longer than 600 s$"):
        read_motion_csv(earlier)
    both = f"{HEADER}\n{row(0.0)}\n{row(0.0)}\n{row(1.0, base_yaw=150.0)}\n".encode()   # repeat, then range
    with pytest.raises(ValueError, match="^base_yaw is out of its calibrated range$"):
        read_motion_csv(both)
    with pytest.raises(ValueError, match="^the CSV cannot be parsed$"):
        read_motion_csv(b"")
    with pytest.raises(ValueError, match="^elbow_pitch is not a finite number$"):
        read_motion_csv(f"{HEADER}\n{row(0.0)}\n{row(1.0, elbow_pitch='nan')}\n".encode())
    times, positions = read_motion_csv(f"{HEADER}\n{row(5.0)}\n{row(5.5, base_yaw=10.0)}\n".encode())
    assert times.tolist() == [0.0, 0.5] and positions[1, 0] == 10.0


# ------------------------------------------------------------------ light
def _post_in_thread(sim: Sim, payload: dict, out: dict, key: str):
    def run():
        out[key] = sim.post("light.glow", payload)
    th = threading.Thread(target=run, daemon=True)
    th.start()
    return th


def _wait_light_log(sim: Sim, n: int):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with sim.lamp.lock:
            if len(sim.lamp.light.log) >= n:
                return
        time.sleep(0.005)
    raise AssertionError("light requests did not arrive")


def test_light_glow_blocks_for_the_fade_and_serialises(sim):
    t0 = sim.lamp.now()
    out = {}
    blue = _post_in_thread(sim, {"color": [0, 120, 255], "luminance": 0.6}, out, "blue")
    _wait_light_log(sim, 1)
    red = _post_in_thread(sim, {"color": "#FF0000", "luminance": 0.5}, out, "red")
    _wait_light_log(sim, 2)
    sim.lamp.advance(0.5)
    time.sleep(0.05)
    assert blue.is_alive() and red.is_alive()                           # nobody answers during the first fade
    sim.lamp.advance(0.15)
    blue.join(5)
    assert not blue.is_alive() and red.is_alive()
    sim.lamp.advance(0.6)
    red.join(5)
    assert not red.is_alive()
    for key, color in (("blue", [0, 120, 255]), ("red", [255, 0, 0])):
        assert out[key].status_code == 200
        action = out[key].json()["action"]
        assert action["state"] == "succeeded"
        assert action["result"]["light"] == "set_solid" and action["result"]["color"] == color
    first, second = sim.lamp.light.log[:2]
    assert first.started_at == pytest.approx(t0) and first.reply_at == pytest.approx(t0 + 0.6)
    assert second.requested_at == pytest.approx(t0)
    assert second.started_at == pytest.approx(first.reply_at)            # waited for the lock
    assert second.reply_at == pytest.approx(t0 + 1.2)
    with sim.lamp.lock:
        panel = sim.lamp.light.sample(sim.lamp.now())
    assert tuple(panel[0]) == (128, 0, 0)                               # red x 0.5


def test_light_glow_blocks_in_real_time_with_team_client(make_sim, team_sdk):
    sim = make_sim(clock="real")
    sdk = client_for(team_sdk, sim)
    sdk.joints()
    t = time.monotonic()
    action = sdk.glow((0, 200, 120), 0.4)
    took = time.monotonic() - t
    assert action["state"] == "succeeded" and action["result"]["luminance"] == 0.4
    assert 0.55 <= took < 2.5                                           # the 600 ms fade, plus HTTP


def test_light_variants_and_errors(sim):
    bad = sim.post("light.glow", {"color": [0, 120, 255], "luminance": 2})
    assert bad.status_code == 400
    assert bad.json()["error"] == {"code": "invalid_request", "message": "luminance must be a number between 0 and 1"}
    floats = sim.post("light.glow", {"color": [0.5, 0, 0], "luminance": 0.2})
    assert floats.json()["error"]["message"] == "color must be three whole numbers from 0 to 255"
    effect = sim.post("light.glow", {"effect": "flowing"})                 # answers at RUNNING, no fade
    assert effect.status_code == 200 and effect.json()["action"]["result"]["animation"] == "flowing"
    unknown = sim.post("light.glow", {"effect": "disco"})                  # an unknown name still "succeeds"
    assert unknown.status_code == 200 and "unknown effect" in sim.lamp.light.log[-1].detail
    # a luminance-only call cancels a running colour fade; that colour call never gets an answer: 503 after 4 s
    out = {}
    colour = _post_in_thread(sim, {"color": [0, 255, 0], "luminance": 0.8}, out, "colour")
    _wait_light_log(sim, 3)
    sim.lamp.advance(0.2)
    dim = _post_in_thread(sim, {"luminance": 0.1}, out, "dim")
    _wait_light_log(sim, 4)
    sim.lamp.advance(0.7)
    dim.join(5)
    assert out["dim"].status_code == 200 and colour.is_alive()
    sim.lamp.advance(3.5)
    colour.join(5)
    assert out["colour"].status_code == 503
    assert out["colour"].json()["error"]["message"] == "No lifecycle acknowledgment for light.command"


def test_sim_light_unit():
    light = SimLight(transition_ms=80)                                   # the owner's what-if: a short fade
    cmd = light.glow({"color": [10, 20, 30], "luminance": 1.0}, t=0.0)
    light.update(0.1)
    assert cmd.status == "succeeded" and cmd.reply_at == pytest.approx(0.08)
    assert tuple(light.sample(0.1)[92]) == (10, 20, 30)
    effect = light.glow({"animation": "breathing", "duration": 1.0}, t=1.0)
    assert effect.reply_at == 1.0
    light.update(3.0)
    assert light.state(3.0)["effect"] is None                           # the timed effect ended, last frame stays


# ------------------------------------------------------------------ rate limit, policy
def test_rate_limit_429_on_the_n_plus_first(make_sim):
    sim = make_sim(rate_limit=3)
    sim.session()
    assert sim.post("attention.nudge").status_code == 403                # refused by policy: takes no slot
    codes = [sim.post("status.read").status_code for _ in range(3)]
    assert codes == [200, 200, 200]
    limited = sim.post("status.read")
    assert limited.status_code == 429
    assert limited.json() == {"ok": False, "error": {"code": "rate_limited",
                                                     "message": "too many SDK actions in the last 60 s",
                                                     "details": {"max_actions_per_minute": 3}}}
    assert "Retry-After" not in limited.headers
    sim.lamp.advance(61.0)                                               # the 60 s window slides
    assert sim.post("status.read").status_code == 200


def test_policy_refusals(sim):
    expect = {
        "attention.nudge": (403, "turned off by the runtime's policy: attention.nudge"),
        "scenario.run": (403, "turned off by the runtime's policy: scenario.run"),
        "motor.write": (403, "Command is not exposed through LeLamp SDK: motor.write"),
        "dance.now": (400, "not an SDK command: dance.now"),
        "": (400, "command_type is required."),
    }
    for command, (status, message) in expect.items():
        r = sim.post(command)
        assert (r.status_code, r.json()["error"]["message"]) == (status, message)
    torque = sim.post("torque.enable", {"now": True})
    assert (torque.status_code, torque.json()["error"]["message"]) == (400, "torque commands take no parameters")
    speech = sim.post("speech.say", {"text": "hi"})
    assert (speech.status_code, speech.json()["error"]["message"]) == (503, "voiceover not available")
    big = sim.post("status.read", {"x": "y" * 17000})
    assert big.status_code == 413 and big.json()["error"]["details"]["max_payload_bytes"] == 16384
    scen = requests.get(sim.url("/api/sdk/v1/scenarios"), headers=sim.auth, timeout=5)
    assert scen.status_code == 503
    assert scen.json()["error"]["message"] == "saved-scenario handoff is turned off by the runtime's policy"


# ------------------------------------------------------------------ the self-collision model is not optional
def test_no_collision_model_refuses_to_start(monkeypatch, capsys):
    """Conformance finding: without FTM_ROBOT_DIR the sim silently accepted every move, clip and upload
    (and said self_collision_check false) where the lamp checks and refuses some. Now it will not start
    without a collision model unless the degraded mode is asked for by name."""
    monkeypatch.delenv("FTM_ROBOT_DIR", raising=False)
    with pytest.raises(MissingCollisionModel, match="FTM_ROBOT_DIR is required"):
        SimLamp(clock="manual")
    with pytest.raises(MissingCollisionModel):
        build(scenario=None, clock="manual", port=0, render="synthetic")
    assert main(["--scenario", "none", "--port", "0", "--render", "synthetic"]) == 2
    err = capsys.readouterr().err
    assert "refusing to start" in err and "--no-collision" in err


def test_degraded_mode_is_opt_in_and_says_so(monkeypatch, make_sim):
    monkeypatch.delenv("FTM_ROBOT_DIR", raising=False)
    with pytest.warns(RuntimeWarning, match="self-collision check OFF"):
        sim = make_sim(collision="off")
    sim.session()
    assert sim.joints()["self_collision_check"] is False
    caps = requests.get(sim.url("/api/sdk/v1/capabilities"), headers=sim.auth, timeout=5).json()
    assert caps["resources"]["joints"]["self_collision_check"] is False
    folded = sim.post("motion.move", {"positions": {"base_pitch": 95.0, "elbow_pitch": -20.0}})
    assert folded.status_code == 202                                   # the lamp would refuse this one
    assert folded.json()["action"]["result"]["collision_checked"] is False
    done = sim.run_until_state(folded.json()["action"]["action_id"], "succeeded")
    assert done["result"]["collision_checked"] is False                # never claims a check it did not do


def test_twin_model_rule_stands_in_for_lamp_spatial(make_sim):
    """With the robot package but without lamp/spatial.py's model, twin/model.py LampTwin's copy of the vendor
    rule is used, so the sim still refuses what the lamp refuses."""
    model_mod = pytest.importorskip("twin.model")
    if robot_dir() is None:
        pytest.skip("FTM_ROBOT_DIR is not set")
    twin = model_mod.LampTwin()
    sim = make_sim(robot="", kinematics=twin)                            # robot="" -> no LampModel
    sim.session()
    assert sim.lamp.pose_ok == twin.sdk_head_base_clear
    assert sim.joints()["self_collision_check"] is True
    start = sim.joints()["positions"]
    folded = sim.post("motion.move", {"positions": all_joints(start, base_pitch=95.0, elbow_pitch=-20.0)})
    assert folded.status_code == 422 and "head could hit the base" in folded.json()["error"]["message"]


# ------------------------------------------------------------------ animations, torque, system.stop, runtime routes
def test_animation_play_is_fire_and_forget(sim):
    listing = requests.get(sim.url("/api/sdk/v1/animations"), headers=sim.auth, timeout=5).json()
    ids = [a["id"] for a in listing["animations"]]
    assert "curious" in ids and listing["forbidden"][0] == "animation.record"
    r = sim.post("animation.play", {"animation_id": "curious"})
    assert r.status_code == 200
    action = r.json()["action"]
    assert action["state"] == "succeeded" and action["result"]["animation"]["played"] is True
    assert action["result"]["behavior_result"]["reason"] == "accepted_async"
    with sim.lamp.lock:
        assert sim.lamp.animation["name"] == "curious"                    # it plays (the sim's own record)
    banned = sim.post("animation.play", {"animation_id": "curious", "frames": []})
    assert banned.status_code == 403
    unknown = sim.post("animation.play", {"animation_id": "moonwalk"})
    assert unknown.status_code == 403 and "approved_animation_ids" in unknown.json()["error"]["details"]
    # a safe move pre-empts the animation; an animation during a safe move is refused, silently to the SDK
    start = sim.joints()["positions"]
    assert sim.post("motion.move", {"positions": all_joints(start, base_yaw=0.0)}).status_code == 202
    again = sim.post("animation.play", {"animation_id": "nod"})
    assert again.status_code == 200 and again.json()["action"]["state"] == "succeeded"
    with sim.lamp.lock:
        assert [e["outcome"] for e in sim.lamp.animation_log if e.get("name") == "curious"][-1] == "preempted"
        refused = sim.lamp.animation_log[-1]
    assert (refused["name"], refused["outcome"], refused["reason"]) == ("nod", "refused", "blocked_by_active_motion")


def test_torque_release_goes_limp_and_blocks_moves(sim):
    start = sim.joints()["positions"]
    released = sim.post("torque.release")
    assert released.status_code == 202 and released.json()["action"]["state"] == "accepted"
    done = sim.run_until_state(released.json()["action"]["action_id"], "succeeded")
    assert done["result"] == {"torque_enabled": False}
    sim.lamp.advance(1.5)
    fallen = sim.joints()["positions"]
    assert fallen["base_pitch"] < start["base_pitch"] - 20                # sagged toward the folded rest pose
    refused = sim.post("motion.move", {"positions": all_joints(start)})
    assert refused.status_code == 422
    assert refused.json()["error"]["message"] == "motion unavailable: the arm is disconnected, stopped or limp"
    enabled = sim.post("torque.enable")
    assert sim.run_until_state(enabled.json()["action"]["action_id"], "succeeded")["result"] == {"torque_enabled": True}
    assert sim.post("motion.move", {"positions": all_joints(start)}).status_code == 202


def test_system_stop_is_an_estop(sim):
    out = {}
    th = threading.Thread(target=lambda: out.update(r=sim.post("system.stop", {"reason": "test"})), daemon=True)
    th.start()
    deadline = time.monotonic() + 5
    while not any(r.command_type == "system.stop" for r in list(sim.gw.actions.values())):
        assert time.monotonic() < deadline
        time.sleep(0.005)
    sim.lamp.advance(1.0)
    time.sleep(0.05)
    assert th.is_alive()                                                 # torque ramp 0.6 s + light fade 0.6 s
    sim.lamp.advance(0.4)
    th.join(5)
    body = out["r"].json()
    assert out["r"].status_code == 200 and body["action"]["state"] == "succeeded"
    assert body["action"]["result"]["stopped"]["motion"] == {"stopped": True, "torque_released": True}
    with sim.lamp.lock:
        assert sim.lamp.torque_enabled is False
        assert int(sim.lamp.light.sample(sim.lamp.now()).max()) == 0      # the light is off


def test_runtime_status_routes(sim):
    status = requests.get(sim.url("/api/status"), timeout=5).json()       # unauthenticated, not /api/sdk/v1
    assert status["is_sleeping"] is False and "flowing" in status["available_light_animations"]
    anim = requests.get(sim.url("/api/animations/status"), timeout=5).json()
    assert set(anim) == {"playing", "current_animation", "started_at", "elapsed_seconds", "current_idle",
                         "idle_paused", "idle_playing", "last_error"}
    assert requests.get(sim.url("/api/nothing"), timeout=5).status_code == 404
    assert requests.delete(sim.url("/api/sdk/v1/joints"), timeout=5).status_code == 405


PLAY_FIELDS = ("playing", "current_animation", "started_at", "elapsed_seconds", "last_error")


def test_sdk_animation_play_leaves_animation_status_alone(sim, team_sdk):
    """Conformance finding: the sim reported SDK animation.play in GET /api/animations/status (playing,
    current_animation, last_error). On the lamp only the dashboard's own play/stop routes write those fields
    (legacy_routes/animations.py:86-88, 99, 144-145), so code that watched them would work here and get
    nothing on the lamp."""
    url = sim.url("/api/animations/status")
    before = requests.get(url, timeout=5).json()
    assert {k: before[k] for k in PLAY_FIELDS} == dict.fromkeys(PLAY_FIELDS) | {"playing": False}
    sdk = client_for(team_sdk, sim)
    if hasattr(sdk, "play_animation"):                                     # the team's client, as in the finding
        played = sdk.play_animation("dance")
    else:                                                                  # origin/main's older client
        played = sdk.action("animation.play", {"name": "dance"})
    assert played["state"] == "succeeded"
    sim.lamp.advance(0.5)
    during = requests.get(url, timeout=5).json()
    with sim.lamp.lock:
        assert sim.lamp.animation is not None                              # it IS playing, in the sim's record
    assert {k: during[k] for k in PLAY_FIELDS} == {k: before[k] for k in PLAY_FIELDS}
    start = sim.joints()["positions"]                                      # now one that is refused
    assert sim.post("motion.move", {"positions": all_joints(start, base_yaw=start["base_yaw"] + 5)}).status_code == 202
    assert sim.post("animation.play", {"animation_id": "happy"}).json()["action"]["state"] == "succeeded"
    after = requests.get(url, timeout=5).json()
    assert {k: after[k] for k in PLAY_FIELDS} == {k: before[k] for k in PLAY_FIELDS}
    with sim.lamp.lock:
        assert sim.lamp.animation_log[-1]["reason"] == "blocked_by_active_motion"   # animations.json only


def _idle(sim: Sim, body=None, *, data=None, headers=None):
    return requests.post(sim.url("/api/animations/idle"), json=body, data=data, headers=headers, timeout=5)


def test_idle_route_answers_like_the_lamp(make_sim):
    """Conformance finding: POST /api/animations/idle was missing (404). It is the dashboard's idle selector
    (legacy_routes/animations.py:175-215), unauthenticated, and it answers once the idle request is queued:
    200 ok for any string name, whatever then happens."""
    sim = make_sim(idle="off")
    for body, idle in (({"name": "none"}, "none"), ({"name": ""}, None), ({}, None), ({"name": None}, None),
                       ({"name": "curious.csv"}, "curious"), ({"name": "moonwalk"}, "moonwalk")):
        r = _idle(sim, body)
        assert r.status_code == 200, (body, r.text)
        answer = r.json()
        assert answer["status"] == "ok" and answer["idle"] == idle
        assert answer["command_id"] == answer["request_id"] and answer["request_id"].startswith("behavior_")
        assert len(answer["event_id"]) == 8
    bad = _idle(sim, {"name": 5})                                          # Path(5): the route's except
    assert bad.status_code == 500 and set(bad.json()) == {"error"}
    assert _idle(sim, [1]).status_code == 500                              # (a list).get before the try
    assert _idle(sim, data=b"not json", headers={"Content-Type": "text/plain"}).json()["idle"] is None


def test_idle_route_effects_follow_the_vendor_runtime(make_sim):
    """What each body really does, traced through the vendor runtime (SimLamp.request_idle): "none" is the
    one name that switches idle off; "" / null answer ok but change nothing ("clear_idle_animation" is not
    an animation); an unknown name changes nothing; while a move holds the joints nothing changes.
    lamp/follow.py and lamp/bridge.py send {"name": "none"}: that is the right body for the lamp."""
    sim = make_sim()
    sim.session()
    status = sim.url("/api/animations/status")

    def current_idle():
        sim.lamp.advance(0.1)                                              # the request runs in the background
        return requests.get(status, timeout=5).json()["current_idle"]

    with sim.lamp.lock:
        sim.lamp.idle_name = "idle"                                        # the stub plays none; say it does
    # follow.py idle_off(): read the idle, then switch it off (lamp/follow.py:183-184)
    was = requests.get(status, timeout=5).json().get("current_idle") or "idle"
    assert _idle(sim, {"name": "none"}).status_code == 200
    assert current_idle() is None and was == "idle"
    assert _idle(sim, {"name": was}).status_code == 200                    # follow.py idle_restore()
    assert current_idle() == "idle"
    for body in ({"name": ""}, {"name": None}, {}, {"name": "moonwalk"}):
        assert _idle(sim, body).json()["status"] == "ok"
        assert current_idle() == "idle", body                              # answered ok, changed nothing
    with sim.lamp.lock:
        reasons = [e.get("reason") for e in sim.lamp.animation_log if e.get("route")]
    assert reasons[-4:] == ["motion_not_available"] * 4
    start = sim.joints()["positions"]
    assert sim.post("motion.move", {"positions": all_joints(start, base_yaw=start["base_yaw"] + 20)}).status_code == 202
    assert _idle(sim, {"name": "none"}).status_code == 200
    assert current_idle() == "idle"                                        # the move holds the joints
    with sim.lamp.lock:
        assert sim.lamp.animation_log[-1]["reason"] == "blocked_by_active_motion"
    assert _idle(sim, {"name": "curious"}).status_code == 200
    sim.lamp.advance(3.0)                                                  # after the move: an animation stem
    assert _idle(sim, {"name": "curious"}).status_code == 200
    assert current_idle() == "curious"


# ------------------------------------------------------------------ camera
def test_camera_snapshot_decodes(make_sim, team_sdk, tmp_path):
    pytest.importorskip("PIL", reason="the simulated camera encodes JPEG with Pillow (503 without it)")
    sim = make_sim(scenario=OnePerson())
    r = requests.get(sim.url("/api/sdk/v1/camera/snapshot"), headers=sim.auth, timeout=10)
    assert r.status_code == 200 and r.headers["Content-Type"] == "image/jpeg"
    assert r.headers["Cache-Control"] == "no-store"
    meta = json.loads(r.headers["X-LeLamp-Metadata"])
    assert set(meta) == {"sequence", "timestamp", "clock", "width", "height", "encoding"}
    assert meta["clock"] == "monotonic" and meta["encoding"] == "jpeg" and meta["sequence"] >= 1
    picture = decode_jpeg(r.content, tmp_path)
    assert picture.shape == (meta["height"], meta["width"], 3) == (480, 640, 3)
    skin = np.all(np.abs(picture.astype(int) - (224, 176, 146)) < 25, axis=2)
    assert skin.sum() > 200                                              # the person in front is in the picture
    assert team_sdk.LampSDK(TOKEN, base=sim.base).snapshot()[:2] == b"\xff\xd8"
    assert requests.get(sim.url("/api/sdk/v1/camera/snapshot"), timeout=5).status_code == 401


def test_painted_head_shows_a_face_toward_the_camera_and_hair_away():
    """The head camera sees a face (skin, dark eyes and brows) when the person faces it and hair from behind."""
    pose = {"position": np.array([0.0, 0.0, 0.30]), "forward": np.array([0.0, 1.0, 0.0]),
            "down": np.array([0.0, 0.0, -1.0]), "right": np.array([1.0, 0.0, 0.0])}

    def middle_of_head(facing) -> np.ndarray:
        person = Person("p", np.array([0.0, 0.8, 0.30]), np.array(facing, float))
        return synthetic_frame(640, 480, pose, [person])[200:280, 280:360].reshape(-1, 3).astype(int)

    front = middle_of_head([0.0, -1.0, 0.0])
    skin = (front[:, 0] > 150) & (front[:, 0] - front[:, 2] > 40)
    dark = front.max(axis=1) < 90
    assert skin.mean() > 0.5 and 0.01 < dark.mean() < 0.3                 # a face: mostly skin, some eyes/brows
    back = middle_of_head([0.0, 1.0, 0.0])
    assert (back.max(axis=1) < 90).mean() > 0.9                           # the back of the head: hair
    assert np.abs(back.mean(axis=0) - np.array(HAIR)).max() < 30
    assert np.abs(front[skin].mean(axis=0) - np.array(SKIN)).max() < 45


def _face_detector():
    """MediaPipe's BlazeFace, set up exactly as lamp/follow.py FaceTracker does (model_selection=1, 0.6)."""
    mp = pytest.importorskip("mediapipe")
    try:
        return mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)
    except AttributeError:
        pytest.skip("this MediaPipe has no solutions.face_detection (lamp/follow.py uses 0.10.x)")


def _look_at_seated_person_and_detect(sim: Sim, kinematics, detector, tmp_path) -> tuple[list, tuple]:
    """walk_in_sit: wait until the person is seated, turn the head to them with one SDK move, take a
    snapshot and run the detector. Returns (detections as (score, x, y), projected head (x, y))."""
    sim.lamp.advance(14.0 - sim.lamp.now())                                # seated by now (twin/world.py)
    person = sim.lamp.people(sim.lamp.now())[0]
    pose, _ = kinematics.look_at(person.head, prefer_distance=0.45)
    moved = sim.post("motion.move", {"positions": {j: round(float(pose[j]), 2) for j in JOINTS}})
    assert moved.status_code == 202, moved.text
    assert sim.run_until_state(moved.json()["action"]["action_id"], "succeeded")["state"] == "succeeded"
    r = requests.get(sim.url("/api/sdk/v1/camera/snapshot"), headers=sim.auth, timeout=30)
    assert r.status_code == 200
    rgb = np.ascontiguousarray(decode_jpeg(r.content, tmp_path))
    found = detector.process(rgb).detections or []
    boxes = [d.location_data.relative_bounding_box for d in found]
    with sim.lamp.lock:
        units = sim.lamp.measured()
        head = sim.lamp.people(sim.lamp.now())[0].head
    return ([(d.score[0], b.xmin + b.width / 2, b.ymin + b.height / 2) for d, b in zip(found, boxes, strict=True)],
            kinematics.project(units, head))


def test_face_detector_finds_the_seated_person(make_sim, tmp_path):
    """Conformance finding: the follower (lamp/follow.py) saw 'nobody in view' for 40 s because the sim's
    people had no face a detector could find. Now the team's own detector finds the seated person once the
    head is turned to them, where the person's head is in the picture."""
    detector = _face_detector()
    world = pytest.importorskip("twin.world")
    if robot_dir() is None:
        pytest.skip("FTM_ROBOT_DIR is not set (the aim comes from lamp/spatial.py)")
    sim = make_sim(scenario=world.scenario("walk_in_sit", 0))
    sim.session()
    faces, projected = _look_at_seated_person_and_detect(sim, sim.lamp.kinematics, detector, tmp_path)
    assert faces, "no face detected in the sim's camera picture"
    score, x, y = max(faces)
    assert score >= 0.6 and abs(x - projected[0]) < 0.08 and abs(y - projected[1]) < 0.12


def test_multipart_stream(make_sim, team_sdk):
    pytest.importorskip("PIL", reason="the simulated camera encodes JPEG with Pillow (503 without it)")
    sim = make_sim(clock="real", scenario=OnePerson())
    url = sim.url("/api/sdk/v1/streams/camera")
    assert requests.get(url, params={"fps": 0}, headers=sim.auth, timeout=5).json()["error"]["message"] == \
        "fps must be between 0.1 and 30"
    assert requests.get(sim.url("/api/sdk/v1/streams/microphone"), headers=sim.auth, timeout=5).status_code == 503
    r = requests.get(url, params={"fps": 10}, headers=sim.auth, stream=True, timeout=(5, 10))
    try:
        assert r.status_code == 200
        assert r.headers["Content-Type"] == "multipart/mixed; boundary=lelamp"
        parts = []
        raw = r.raw
        while len(parts) < 3:
            line = raw.readline()
            assert line, "stream ended early"
            if line.strip() != b"--lelamp":
                continue
            headers = {}
            while (h := raw.readline()) not in (b"\r\n", b""):
                k, _, v = h.decode("latin-1").partition(":")
                headers[k.strip().lower()] = v.strip()
            body = raw.read(int(headers["content-length"]))
            assert raw.read(2) == b"\r\n"
            parts.append((headers, body))
    finally:
        r.close()
    metas = [json.loads(h["x-lelamp-metadata"]) for h, _ in parts]
    assert all(h["content-type"] == "image/jpeg" for h, _ in parts)
    assert all(b[:2] == b"\xff\xd8" and b[-2:] == b"\xff\xd9" for _, b in parts)
    assert metas[0]["gap"] == 0 and all(m["gap"] >= 0 for m in metas)
    assert [m["sequence"] for m in metas] == sorted({m["sequence"] for m in metas})   # strictly increasing
    sdk = client_for(team_sdk, sim)
    frames = []
    for jpeg in sdk.camera_frames(fps=10):
        frames.append(jpeg)
        if len(frames) == 3:
            break
    assert len(frames) == 3 and all(f[:2] == b"\xff\xd8" for f in frames)


# ------------------------------------------------------------------ helpers
def test_split_reason():
    assert split_reason("422 invalid_request: no joint named x") == (422, "invalid_request", "no joint named x")
    assert split_reason("503 capability_unavailable: no free safe-motion slot")[0] == 503
    assert split_reason("plain words") == (422, "invalid_request", "plain words")


# ------------------------------------------------------------------ integration with the twin's real modules
def test_integration_with_twin_model_head_camera(make_sim, tmp_path):
    """Camera frames rendered by twin/model.py LampTwin (MuJoCo) from the head camera, through the gateway."""
    model_mod = pytest.importorskip("twin.model")
    directory = os.environ.get("FTM_ROBOT_DIR", "")
    if not directory or not (Path(directory) / "robot.urdf").exists():
        pytest.skip("FTM_ROBOT_DIR is not set")
    twin = model_mod.LampTwin()

    def render(units, people, light, width, height):
        return twin.render(units, people=people, light_rgb=light, camera="head", width=width, height=height)

    sim = make_sim(scenario=OnePerson(), kinematics=twin, renderer=render, frame_size=(320, 240))
    shots = []
    for _ in range(2):                                   # two requests arrive on two HTTP threads
        sim.lamp.advance(0.1)
        r = requests.get(sim.url("/api/sdk/v1/camera/snapshot"), headers=sim.auth, timeout=30)
        assert r.status_code == 200
        shots.append(r.content)
    if sim.lamp.last_render_error:
        pytest.skip(f"MuJoCo rendering unavailable here: {sim.lamp.last_render_error}")
    picture = decode_jpeg(shots[-1], tmp_path)
    assert picture.shape == (240, 320, 3)
    assert picture.std() > 5                             # a rendered scene, not a flat fill


def test_integration_face_detector_on_mujoco_frames(make_sim, tmp_path):
    """The same as test_face_detector_finds_the_seated_person with twin/model.py LampTwin's MuJoCo picture
    (what `python -m twin.sim_sdk` serves by default): its people are plain spheres, so the sim paints the
    faces over them from the same camera."""
    detector = _face_detector()
    model_mod = pytest.importorskip("twin.model")
    world = pytest.importorskip("twin.world")
    if robot_dir() is None:
        pytest.skip("FTM_ROBOT_DIR is not set")
    twin = model_mod.LampTwin()

    def render(units, people, light, width, height):
        return twin.render(units, people=people, light_rgb=light, camera="head", width=width, height=height)

    sim = make_sim(scenario=world.scenario("walk_in_sit", 0), kinematics=twin, renderer=render)
    sim.session()
    faces, projected = _look_at_seated_person_and_detect(sim, twin, detector, tmp_path)
    if sim.lamp.last_render_error:
        pytest.skip(f"MuJoCo rendering unavailable here: {sim.lamp.last_render_error}")
    assert faces, "no face detected in the MuJoCo camera picture"
    score, x, y = max(faces)
    assert score >= 0.6 and abs(x - projected[0]) < 0.08 and abs(y - projected[1]) < 0.12


def test_integration_idle_off_with_twin_motion_model(make_sim):
    """POST /api/animations/idle {"name": "none"} stops twin/motion.py's idle where it is and the arm holds;
    {"name": "idle"} brings it back. Guards switch_idle(), which drives that model's idle attributes."""
    motion_mod = pytest.importorskip("twin.motion")
    if robot_dir() is None:
        pytest.skip("FTM_ROBOT_DIR is not set")
    probe = SimLamp(clock="manual", idle="off")
    model = motion_mod.SDKMotionModel(probe.start_pose, pose_ok=probe.pose_ok, idle="auto", robot_dir=robot_dir(),
                                      rate_limit_per_min=10 ** 9, capacity=10 ** 6)
    sim = make_sim(motion=model)
    status = sim.url("/api/animations/status")
    sim.lamp.advance(3.0)
    assert requests.get(status, timeout=5).json()["idle_playing"] is True
    assert requests.get(status, timeout=5).json()["current_idle"] == "idle"
    assert _idle(sim, {"name": "none"}).json()["idle"] == "none"
    sim.lamp.advance(0.5)
    now = requests.get(status, timeout=5).json()
    assert now["current_idle"] is None and now["idle_playing"] is False
    held = sim.joints()["positions"]
    sim.lamp.advance(4.0)
    later = sim.joints()["positions"]
    assert max(abs(later[j] - held[j]) for j in JOINTS) < 0.5              # it holds where idle left it
    assert _idle(sim, {"name": "idle"}).status_code == 200
    sim.lamp.advance(0.5)
    assert requests.get(status, timeout=5).json()["idle_playing"] is True
    sim.lamp.advance(4.0)
    moved = sim.joints()["positions"]
    assert max(abs(moved[j] - later[j]) for j in JOINTS) > 1.0             # idle moves the arm again


def test_integration_with_twin_motion_model(make_sim):
    """The gateway on top of twin/motion.py SDKMotionModel (idle off so the arm stays where it is put)."""
    motion_mod = pytest.importorskip("twin.motion")
    directory = os.environ.get("FTM_ROBOT_DIR", "")
    if not directory or not Path(directory).is_dir():
        pytest.skip("FTM_ROBOT_DIR is not set")
    probe = SimLamp(clock="manual", idle="off")                          # borrow its poses and collision check
    model = motion_mod.SDKMotionModel(probe.start_pose, pose_ok=probe.pose_ok, idle=None,
                                      rate_limit_per_min=10 ** 9, capacity=10 ** 6)
    sim = make_sim(motion=model)
    sim.session()
    start = sim.joints()["positions"]
    t0 = sim.lamp.now()
    first = sim.post("motion.move", {"positions": all_joints(start, base_yaw=start["base_yaw"] + 30)}).json()["action"]
    sim.lamp.advance(1.0)
    second = sim.post("motion.move", {"positions": all_joints(start, base_yaw=start["base_yaw"] - 10)}).json()["action"]
    old = sim.run_until_state(first["action_id"], "canceled")
    assert old["result"] == {"reason": "cancelled_or_preempted"}
    done = sim.run_until_state(second["action_id"], ("succeeded", "failed"))
    assert done["state"] == "succeeded", done
    assert done["result"]["reached"] is True
    with sim.lamp.lock:
        detail = model.actions[sim.gw.actions[second["action_id"]].motion_id]
        assert start["base_yaw"] < detail.replan_from["base_yaw"] < start["base_yaw"] + 30
        assert detail.t_finished - t0 >= 3.0                               # 1 s of the first, then >= 2 s of the second
