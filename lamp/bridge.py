#!/usr/bin/env python3
"""Receive Mac camera targets and turn the lamp through the vendor SDK.

The Mac sends small UDP JSON packets to port 47400. This process is the only bridge
between that add-on and the lamp: it reads the SDK token locally, applies the joint
step/thermal limits, and sends whole-arm ``motion.move`` actions through ``/api/sdk/v1``.
No camera frames or secrets leave the devices.

Run on the Pi (beside sdk.py and spatial.py)::

    .venv/bin/python bridge.py --cutoff-c 70
"""
from __future__ import annotations

import argparse
import json
import math
import signal
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

from sdk import LampSDK, SDKError, read_token
from spatial import DEFAULT_ROBOT_DIR, JOINTS, LampModel

LISTEN_PORT = 47400
THERMAL_PATH = "/sys/class/thermal/thermal_zone0/temp"


def temperature_c() -> float | None:
    try:
        return int(Path(THERMAL_PATH).read_text()) / 1000.0
    except (OSError, ValueError):
        return None


def idle_off(base: str) -> str | None:
    """Pause the runtime's dashboard idle selector so it cannot undo a tracking move.

    This is the same non-motion selector used by follow.py. All arm movement still goes
    through the token-gated SDK gateway.
    """
    import requests
    try:
        previous = requests.get(f"{base}/api/animations/status", timeout=4).json().get("current_idle") or "idle"
        requests.post(f"{base}/api/animations/idle", json={"name": "none"}, timeout=8).raise_for_status()
        return previous
    except Exception:
        return None


def idle_restore(base: str, previous: str | None) -> None:
    if not previous:
        return
    import requests
    try:
        requests.post(f"{base}/api/animations/idle", json={"name": previous}, timeout=8).raise_for_status()
    except Exception:
        print("warning: could not restore the lamp idle selector", flush=True)


def telemetry(sock: socket.socket, addr: tuple[str, int] | None, reply_port: int, state: str,
              temp: float | None, **extra) -> None:
    if addr is None or not 1024 <= int(reply_port) <= 65535:
        return
    obj = {"v": 1, "t": "lamp_telemetry", "state": state, "piC": temp}
    obj.update(extra)
    try:
        sock.sendto(json.dumps(obj, separators=(",", ":")).encode(), (addr[0], int(reply_port)))
    except OSError:
        pass


def bounded_step(model: LampModel, measured: dict[str, float], target: dict[str, float], max_step: float) -> dict[str, float] | None:
    """Take one safe, bounded step and check geometry between measured and endpoint."""
    biggest = max(abs(target[j] - measured[j]) for j in JOINTS)
    if biggest > max_step:
        f = max_step / biggest
        target = {j: measured[j] + (target[j] - measured[j]) * f for j in JOINTS}
    target = {j: float(np.clip(target[j], *model.limits[j])) for j in JOINTS}
    if model.problems(target):
        return None
    for n in range(1, 21):
        intermediate = {j: measured[j] + (target[j] - measured[j]) * n / 20 for j in JOINTS}
        if model.problems(intermediate):
            return None
    return target


def target_point(model: LampModel, measured: dict[str, float], packet: dict) -> tuple[np.ndarray, str] | None:
    kind = str(packet.get("kind", "none"))
    if kind == "none":
        return None
    try:
        x, y = float(packet["x"]), float(packet["y"])
        size = float(packet.get("size", 0.2))
        confidence = float(packet.get("confidence", 0))
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x, y, size, confidence)) or confidence < 0.35:
        return None
    x, y, size = np.clip([x, y, size], [0, 0, 0.05], [1, 1, 1])
    real_width = 0.15 if kind == "face" else 0.45
    near, far = (0.30, 2.5) if kind == "face" else (0.45, 3.0)
    distance = float(np.clip(model.distance_from_size(float(size), real_width), near, far))
    point = model.target_point(measured, (float(x), float(y)), distance)
    # A target beside/behind the base would require an abrupt, uncomfortable turn.
    if point[1] < 0.05:
        return None
    return point, kind


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--listen", default="0.0.0.0:47400", help="UDP bind address")
    ap.add_argument("--cutoff-c", type=float, default=70.0, help="stop motion at this Pi temperature")
    ap.add_argument("--max-step", type=float, default=5.0, help="maximum change to any joint per move")
    ap.add_argument("--min-interval", type=float, default=2.5, help="seconds between SDK moves")
    ap.add_argument("--keep-idle", action="store_true", help="leave the lamp's idle animation running between moves")
    ap.add_argument("--robot-dir", default=str(DEFAULT_ROBOT_DIR))
    args = ap.parse_args()
    host, _, port_text = args.listen.rpartition(":")
    host = host or "0.0.0.0"
    port = int(port_text or LISTEN_PORT)

    try:
        model = LampModel(args.robot_dir)
        sdk = LampSDK(read_token())
        caps = sdk.capabilities()
        info = sdk.joints()
    except (SDKError, OSError, ValueError) as exc:
        sys.exit(f"lamp SDK/model is not usable: {exc}")
    if not info.get("self_collision_check"):
        sys.exit("runtime reports self_collision_check = false; refusing to move")
    if info.get("units") != "normalized_m100_100" or set(info.get("joints", {})) != set(JOINTS):
        sys.exit(f"unexpected joint space: {info.get('units')} {sorted(info.get('joints', {}))}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port)); sock.settimeout(0.25)
    stop = threading.Event()

    def request_stop(*_):
        stop.set()

    def thermal_watch() -> None:
        while not stop.is_set():
            temp = temperature_c()
            if temp is None:
                print("THERMAL STOP: Pi temperature unavailable", flush=True)
                stop.set(); return
            if temp >= args.cutoff_c:
                print(f"THERMAL STOP: Pi {temp:.1f} C >= {args.cutoff_c:.1f} C", flush=True)
                stop.set(); return
            stop.wait(0.25)

    signal.signal(signal.SIGINT, request_stop); signal.signal(signal.SIGTERM, request_stop)
    print(f"SDK {caps.get('protocol_version', info.get('units'))} ready; UDP {host}:{port}; cutoff {args.cutoff_c:.1f} C",
          flush=True)
    print("waiting for FeelTheMusic camera targets (Ctrl-C stops after the current action)", flush=True)

    latest: dict | None = None
    latest_addr: tuple[str, int] | None = None
    latest_reply_port = 47401
    last_session = None
    last_seq = -1
    last_packet_at = 0.0
    last_move = 0.0
    last_telemetry = 0.0
    measured = {j: float(info["positions"][j]) for j in JOINTS}
    moves = 0
    previous_idle = None if args.keep_idle else idle_off(sdk.base)
    threading.Thread(target=thermal_watch, name="pi-thermal-guard", daemon=True).start()

    try:
        while not stop.is_set():
            now = time.monotonic()
            temp = temperature_c()
            if temp is None:
                telemetry(sock, latest_addr, latest_reply_port, "thermal_unavailable", None)
                print("THERMAL STOP: Pi temperature unavailable", flush=True); break
            if temp >= args.cutoff_c:
                telemetry(sock, latest_addr, latest_reply_port, "thermal_cutoff", temp)
                print(f"THERMAL STOP: Pi {temp:.1f} C >= {args.cutoff_c:.1f} C", flush=True); break
            try:
                data, addr = sock.recvfrom(8192)
                packet = json.loads(data.decode("utf-8"))
                if packet.get("t") != "lamp_target" or int(packet.get("v", 0)) != 1:
                    continue
                session = packet.get("session")
                seq = int(packet.get("seq", -1))
                if session != last_session:
                    last_session, last_seq = session, -1
                if seq <= last_seq:
                    continue
                last_seq, last_packet_at = seq, time.monotonic()
                latest, latest_addr = packet, addr
                latest_reply_port = int(packet.get("replyPort", 47401))
            except socket.timeout:
                pass
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
                continue

            if now - last_telemetry >= 1.0:
                telemetry(sock, latest_addr, latest_reply_port, "tracking" if latest else "waiting", temp,
                          moves=moves, reached=True)
                last_telemetry = now
            if latest is None or latest.get("kind") == "none":
                continue
            if now - last_packet_at > float(latest.get("ttlMs", 1500)) / 1000.0:
                latest = None; continue
            if now - last_move < args.min_interval:
                continue

            try:
                measured = {j: float(v) for j, v in sdk.joints()["positions"].items() if j in JOINTS}
                result = target_point(model, measured, latest)
                if result is None:
                    telemetry(sock, latest_addr, latest_reply_port, "holding", temp, reached=True); continue
                point, kind = result
                pose, report = model.look_at(point, seed=measured, prefer_distance=0.45)
                step = bounded_step(model, measured, pose, min(args.max_step, float(latest.get("maxStep", args.max_step))))
                if step is None:
                    telemetry(sock, latest_addr, latest_reply_port, "target_rejected", temp, reached=False)
                    print(f"target rejected by local workspace checks ({kind})", flush=True); latest = None; continue
                telemetry(sock, latest_addr, latest_reply_port, "moving", temp, reached=False)
                action = sdk.move(step)
                measured = step; last_move = time.monotonic(); moves += 1
                telemetry(sock, latest_addr, latest_reply_port, "reached", temperature_c(), reached=True,
                          action=action.get("action_id"))
                print(f"move {moves}: {kind}, Pi {temp:.1f} C, aim error {report['aim_error_deg']:.1f} deg", flush=True)
            except SDKError as exc:
                telemetry(sock, latest_addr, latest_reply_port, "move_failed", temp, reached=False, error=exc.code)
                print(f"SDK move failed; holding: {exc}", flush=True)
                latest = None
                time.sleep(0.5)
    finally:
        sock.close()
        idle_restore(sdk.base, previous_idle)
        print(f"stopped; {moves} SDK moves completed", flush=True)


if __name__ == "__main__":
    main()
