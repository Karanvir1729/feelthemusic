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

from sdk import LampSDK, SDKError, read_token, refusal_policy
from spatial import DEFAULT_ROBOT_DIR, JOINTS, LampModel

LISTEN_PORT = 47400
DEFAULT_REPLY_PORT = 47401
DEFAULT_TTL_MS = 1500
# A real target packet is about 200 bytes and a flat object. Anything bigger or deeper is hostile.
# The limits keep json.loads away from deep nesting, which raises RecursionError (a RuntimeError,
# not a ValueError) and used to kill the bridge from one datagram.
MAX_DATAGRAM_BYTES = 2048
MAX_NESTING = 4
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


def _bounded_int(value: object, *, minimum: int, maximum: int) -> int | None:
    """Parse an integer without accepting booleans or silently truncating floats."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.strip().lstrip("+-").isdigit():
        try:
            parsed = int(value)
        except ValueError:
            return None
    else:
        return None
    return parsed if minimum <= parsed <= maximum else None


def _bounded_float(value: object, *, minimum: float, maximum: float) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        return None
    return parsed


def normalize_packet(packet: object, max_step: float) -> dict | None:
    """Validate and normalize one untrusted UDP target packet.

    The Mac sender is on the local network, so malformed input must be treated as data and
    discarded. In particular, never call ``.get`` until JSON has been confirmed to be an object,
    and never let packet-controlled timing values escape their safe bounds.
    """
    if (not isinstance(packet, dict) or packet.get("t") != "lamp_target"
            or not isinstance(packet.get("v"), int) or isinstance(packet.get("v"), bool)
            or packet.get("v") != 1):
        return None
    session = packet.get("session")
    if isinstance(session, bool) or not isinstance(session, (int, str)) or not str(session) or len(str(session)) > 128:
        return None
    seq = _bounded_int(packet.get("seq"), minimum=0, maximum=2**63 - 1)
    reply_port = _bounded_int(packet.get("replyPort", DEFAULT_REPLY_PORT), minimum=1024, maximum=65535)
    ttl_ms = _bounded_int(packet.get("ttlMs", DEFAULT_TTL_MS), minimum=1, maximum=60_000)
    if seq is None or reply_port is None or ttl_ms is None:
        return None
    ttl_ms = min(ttl_ms, DEFAULT_TTL_MS)
    kind = packet.get("kind", "none")
    if not isinstance(kind, str) or not kind or len(kind) > 32:
        return None
    if kind != "none":
        for name in ("x", "y", "size", "confidence"):
            if _bounded_float(packet.get(name), minimum=-1e9, maximum=1e9) is None:
                return None
    packet = dict(packet)
    packet.update(seq=seq, replyPort=reply_port, ttlMs=ttl_ms, kind=kind)
    requested_step = _bounded_float(packet.get("maxStep", max_step), minimum=0.01, maximum=max_step)
    if requested_step is None:
        return None
    packet["maxStep"] = requested_step
    return packet


def _nesting_depth(data: bytes) -> int:
    """Deepest [ / { nesting in a JSON byte string, ignoring brackets inside strings."""
    depth = deepest = 0
    in_string = escaped = False
    for byte in data:
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:            # backslash
                escaped = True
            elif byte == 0x22:            # closing quote
                in_string = False
        elif byte == 0x22:
            in_string = True
        elif byte in (0x5B, 0x7B):        # [ {
            depth += 1
            deepest = max(deepest, depth)
        elif byte in (0x5D, 0x7D):        # ] }
            depth = max(0, depth - 1)
    return deepest


def parse_datagram(data: bytes, max_step: float) -> dict | None:
    """One untrusted UDP datagram to a validated packet, or None. Never raises."""
    if not isinstance(data, (bytes, bytearray)) or not data or len(data) > MAX_DATAGRAM_BYTES:
        return None
    if _nesting_depth(bytes(data)) > MAX_NESTING:
        return None
    try:
        return normalize_packet(json.loads(bytes(data).decode("utf-8")), max_step)
    except (UnicodeDecodeError, ValueError, TypeError, RecursionError):
        return None


def terminal_motion_failure(exc: SDKError) -> bool:
    """Whether it is unsafe for the bridge to send another move after this error."""
    return exc.status == 409 or exc.code in {"lost_track", "timeout", "not_reached", "canceled"}


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
    ap.add_argument("--allow-ip", action="append", default=[], metavar="IP",
                    help="only accept targets from this source IP (repeatable); without it, pin the first valid sender")
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
    latest_reply_port = DEFAULT_REPLY_PORT
    pinned_ip: str | None = None
    last_session = None
    last_seq = -1
    last_packet_at = 0.0
    last_move = 0.0
    last_telemetry = 0.0
    measured = {j: float(info["positions"][j]) for j in JOINTS}
    moves = 0
    failures, retry_after, exit_code = 0, 0.0, 0
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
                data, addr = sock.recvfrom(MAX_DATAGRAM_BYTES + 1)   # +1 so an oversize datagram is detectable
                packet = parse_datagram(data, args.max_step)
                if packet is None:
                    continue
                peer_ip = str(addr[0])
                if args.allow_ip and peer_ip not in set(args.allow_ip):
                    continue
                if pinned_ip is None:
                    pinned_ip = peer_ip
                    print(f"pinned target sender to {pinned_ip}", flush=True)
                elif peer_ip != pinned_ip:
                    continue
                # Do not let a later packet redirect telemetry to an arbitrary UDP endpoint.
                if latest_addr is not None and packet["replyPort"] != latest_reply_port:
                    continue
                session = packet.get("session")
                seq = packet["seq"]
                if session != last_session:
                    last_session, last_seq = session, -1
                if seq <= last_seq:
                    continue
                last_seq, last_packet_at = seq, time.monotonic()
                latest, latest_addr = packet, addr
                latest_reply_port = packet["replyPort"]
            except socket.timeout:
                pass
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
                continue

            now = time.monotonic()      # re-read: recvfrom blocks for up to its 0.25 s timeout
            if now - last_telemetry >= 1.0:
                telemetry(sock, latest_addr, latest_reply_port, "tracking" if latest else "waiting", temp,
                          moves=moves, reached=False if latest else True)
                last_telemetry = now
            if latest is None or latest.get("kind") == "none":
                continue
            if now - last_packet_at > float(latest.get("ttlMs", 1500)) / 1000.0:
                latest = None; continue
            if now - last_move < args.min_interval or now < retry_after:
                continue

            try:
                measured = {j: float(v) for j, v in sdk.joints()["positions"].items() if j in JOINTS}
                result = target_point(model, measured, latest)
                if result is None:
                    telemetry(sock, latest_addr, latest_reply_port, "holding", temp, reached=True); continue
                point, kind = result
                pose, report = model.look_at(point, seed=measured, prefer_distance=0.45)
                step = bounded_step(model, measured, pose, min(args.max_step, latest["maxStep"]))
                if step is None:
                    telemetry(sock, latest_addr, latest_reply_port, "target_rejected", temp, reached=False)
                    print(f"target rejected by local workspace checks ({kind})", flush=True); latest = None; continue
                telemetry(sock, latest_addr, latest_reply_port, "moving", temp, reached=False)
                action = sdk.move(step)
                measured = step; last_move = time.monotonic(); moves += 1; failures = 0
                telemetry(sock, latest_addr, latest_reply_port, "reached", temperature_c(), reached=True,
                          action=action.get("action_id"))
                print(f"move {moves}: {kind}, Pi {temp:.1f} C, aim error {report['aim_error_deg']:.1f} deg", flush=True)
            except SDKError as exc:
                telemetry(sock, latest_addr, latest_reply_port, "move_failed", temp, reached=False, error=exc.code)
                print(f"SDK move failed; holding: {exc}", flush=True)
                latest = None
                if terminal_motion_failure(exc):
                    print("stopping after an uncertain or incomplete move; no move will be sent on top of it", flush=True)
                    exit_code = 2
                    stop.set()
                else:
                    failures += 1
                    stop_now, wait_s, why = refusal_policy(exc, failures)
                    if stop_now:
                        print(f"stopping: {why}", flush=True)
                        exit_code = 2
                        stop.set()
                    else:
                        retry_after = time.monotonic() + wait_s
                        print(f"the lamp refused the move; no new move for {wait_s:.0f} s", flush=True)
    finally:
        sock.close()
        idle_restore(sdk.base, previous_idle)
        print(f"stopped; {moves} SDK moves completed", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
