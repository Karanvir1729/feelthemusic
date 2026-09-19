"""One lamp process, written 2026-09-19.

Run ``python lamp/app.py --conductor <numeric-IP>`` after installing PR #16.
The default is an observation-only client: it never creates an SDK session.
``--enable-output`` enables the conservative light renderer through the SDK.
Follow and dance need injected, independently validated planners; this module
does not silently substitute the old standalone follower or builtin animations.

Times are local monotonic submission deadlines from the shared FTM client.
No second room latency is added. Light output latency is an explicit measured
trim; its default zero is uncalibrated, not a claim of visible synchronization.
The dim cyan renderer is a baseline integration path, not the final light show.
"""
from __future__ import annotations

import argparse
import heapq
import ipaddress
import math
import re
import socket
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lamp.dispatch import ClipIntent, Dispatcher, LightIntent, MoveIntent

NS = 1_000_000_000
ADMISSION_LEASE_NS = 20_000_000


def finite_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


@dataclass(frozen=True)
class OutputTypes:
    event: type
    bass: type
    mode: type
    session: type

    @classmethod
    def native(cls):
        from ftm_session import BassOut, EventOut, ModeOut, SessionOut
        return cls(EventOut, BassOut, ModeOut, SessionOut)


class LampApp:
    """Single-threaded client/planner loop; only Dispatcher owns SDK workers.

    Optional planners implement reset(epoch, gen), on_event(event, now_ns,
    epoch, gen)->iterable[ClipIntent], and follow(now_ns, epoch, gen)->MoveIntent
    or None. Methods may be omitted. They must be bounded and nonblocking.
    Planner intents retain their original epoch/gen; app never relabels them.
    """

    def __init__(self, client, dispatcher, *, output_types=None, planner=None,
                 armed=False, light_trim_ns=0, temperature=None):
        if type(armed) is not bool:
            raise ValueError("armed must be bool")
        if type(light_trim_ns) is not int or not 0 <= light_trim_ns <= NS:
            raise ValueError("light trim must be 0..1 second")
        self.client, self.dispatcher = client, dispatcher
        self.types = output_types or OutputTypes.native()
        self.planner = planner
        self.armed, self.light_trim_ns = armed, light_trim_ns
        self.temperature = temperature
        self.gen = 0
        self._epoch = client.session.epoch
        self._key = None
        self._lights = []
        self._order = 0
        self.dropped = 0
        self.error = None

    def _configure(self, now_ns):
        s = self.client.session
        mode, lights = s.current_mode(now_ns), s.current_lights(now_ns)
        valid_until = now_ns + ADMISSION_LEASE_NS
        # Prove readiness through a short future lease, not just this loop's
        # instant. A delayed worker rechecks the lease at actual admission.
        synced = (s.estimator.estimate(now_ns=valid_until) is not None and
                  s.current_mode(valid_until) == mode and
                  s.current_lights(valid_until) == lights)
        key = (s.epoch, self.gen, mode, lights, synced, self.armed)
        if key != self._key:
            self._lights.clear()
            if self.planner is not None and hasattr(self.planner, "reset"):
                self.planner.reset(s.epoch, self.gen)
            self._key = key
        self.dispatcher.configure(epoch=s.epoch, gen=self.gen, mode=mode,
                                  lights=lights, synced=synced, armed=self.armed,
                                  valid_until_ns=valid_until)
        return mode, lights, synced

    def _queue_light(self, due_ns, level, epoch, now_ns):
        if epoch != self.client.session.epoch or type(due_ns) is not int:
            self.dropped += 1
            return
        if not finite_number(level):
            self.dropped += 1
            return
        due_ns -= self.light_trim_ns
        if not now_ns <= due_ns <= now_ns + 30 * NS:
            self.dropped += 1
            return
        # No red, and every channel <= 0.3. Dispatcher repeats these bounds at
        # the actual output boundary; the SDK payload also caps luminance.
        level = max(0.0, min(1.0, level))
        intent = LightIntent((0.0, .3 * level, .15 * level), due_ns, epoch, self.gen)
        if len(self._lights) >= 256:
            self.dropped += 1
            return  # bounded memory; do not evict the imminent deadline
        self._order += 1
        heapq.heappush(self._lights, (due_ns, self._order, intent))

    def step(self, now_ns):
        """One bounded receive/plan/dispatch pass; no sockets or hardware in tests."""
        outputs = self.client.step(now_ns)
        s = self.client.session
        # A socket pass can contain multiple mode/session changes. Discard all
        # earlier outputs before the last barrier; old events carry no gen.
        barrier = -1
        if s.epoch != self._epoch:
            self.gen = 0
            self._epoch = s.epoch
        for i, out in enumerate(outputs):
            if isinstance(out, self.types.session):
                self.gen = 0
                barrier = i
            elif isinstance(out, self.types.mode):
                self.gen = out.gen
                barrier = i
        try:
            mode, lights, synced = self._configure(now_ns)
            for out in outputs[barrier + 1:]:
                if isinstance(out, self.types.event):
                    event = out.event
                    if event.epoch != s.epoch or not synced or mode == "off":
                        continue
                    if lights:
                        self._queue_light(event.due_ns, event.intensity, event.epoch, now_ns)
                    if mode == "dance" and self.planner is not None and hasattr(self.planner, "on_event"):
                        for index, intent in enumerate(self.planner.on_event(event, now_ns, s.epoch, self.gen)):
                            if index >= 32:
                                raise ValueError("planner output limit")
                            self.dispatcher.submit_clip(intent)
                elif isinstance(out, self.types.bass) and synced and lights and mode != "off":
                    for sample in out.samples:
                        self._queue_light(sample.due_ns, sample.level, sample.epoch, now_ns)
            if mode == "follow" and synced and self.planner is not None and hasattr(self.planner, "follow"):
                intent = self.planner.follow(now_ns, s.epoch, self.gen)
                if intent is not None:
                    self.dispatcher.submit_move(intent)
            latest = None
            while self._lights and self._lights[0][0] <= now_ns:
                latest = heapq.heappop(self._lights)[2]
            if latest is not None:
                self.dispatcher.submit_light(latest)
            self.dispatcher.tick(now_ns)
        except Exception:
            # Do not echo SDK/network errors: they may contain addresses/tokens.
            self.error = "application failure"
            self.armed = False
            self._lights.clear()
            self.dispatcher.configure(epoch=s.epoch, gen=self.gen, mode="off",
                                      lights=False, synced=False, armed=False)
            self.dispatcher.tick(now_ns)
            raise
        self.client.set_status(self.status(now_ns))

    def status(self, now_ns):
        mode = self.client.session.current_mode(now_ns)
        snap = self.dispatcher.snapshot()
        missing = ((mode == "follow" and not hasattr(self.planner, "follow")) or
                   (mode == "dance" and not hasattr(self.planner, "on_event")))
        error = self.error or ("planner unavailable" if missing else None)
        if self.armed and mode != "off" and not (self._key and self._key[4]):
            error = "clock unsynchronized"
        if snap.get("latched") or snap.get("safety_latched"):
            error = "output latched"
        state = "error" if error else ("idle" if not self.armed or mode == "off" else
                                      "light" if mode == "light" else
                                      "searching" if mode == "follow" else
                                      "dancing" if snap.get("motion_inflight") else "idle")
        # piC is required by the current wire schema. Zero means unavailable in
        # this adapter and is accompanied by sdk text, never claimed measured.
        temp = self.temperature() if self.temperature else None
        known = finite_number(temp)
        stats = snap.get("stats", {})
        return {"state": state, "mode": mode, "locked": False,
                "piC": float(temp) if known else 0.0,
                "sdk": error or ("observation only" if not self.armed else
                                  "ok" if known else "temperature unavailable"),
                "moves": min(int(stats.get("motion_completed", 0)), 2**32 - 1),
                "refused": min(int(stats.get("latched", 0)), 2**32 - 1)}

    def close(self):
        self.armed = False
        self._lights.clear()
        self.dispatcher.close()


class SDKActions:
    """Use existing SDK plumbing, without its retry/cancel/polling action helper.

    Supply two independently prepared LampSDK clients sharing one SDK session.
    Exactly one action POST per callback, no replay after ambiguous failure.
    Polls are reads; motion and light use separate HTTP connection pools.
    Timeout never issues cancel or system.stop; the dispatcher retains ownership
    and latches. Session creation/refresh occurs before arming, not in a worker.
    """

    def __init__(self, motion_sdk, light_sdk, *, clock=time.monotonic_ns, sleep=time.sleep, limiter=None):
        self.motion_sdk, self.light_sdk = motion_sdk, light_sdk
        self.clock, self.sleep = clock, sleep
        self.limiter = limiter

    def _run(self, sdk, kind, payload, duration_ns):
        deadline = self.clock() + duration_ns + NS

        def call(method, path, body=None):
            remaining = (deadline - self.clock()) / NS
            if remaining <= 0:
                raise RuntimeError("SDK outcome unconfirmed")
            result = sdk._call(method, path, json=body, timeout=min(2.0, remaining))
            if not isinstance(result, dict) or not isinstance(result.get("action"), dict):
                raise RuntimeError("invalid SDK action response")
            return result["action"]

        action = call("POST", "/api/sdk/v1/actions",
                      {"type": kind, "payload": payload, "idempotency_key": uuid.uuid4().hex})
        action_id = action.get("action_id")
        if not isinstance(action_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", action_id):
            raise RuntimeError("invalid SDK action identity")
        while action.get("state") in {"accepted", "running"}:
            remaining = (deadline - self.clock()) / NS
            if remaining <= 0:
                raise RuntimeError("SDK outcome unconfirmed")
            self.sleep(min(.05, remaining))
            action = call("GET", "/api/sdk/v1/actions/" + action_id)
            if action.get("action_id") != action_id:
                raise RuntimeError("SDK action identity changed")
        if action.get("state") not in {"succeeded", "failed", "rejected", "canceled"}:
            raise RuntimeError("invalid SDK action state")
        return action

    def motion(self, intent):
        if isinstance(intent, ClipIntent):
            return self._run(self.motion_sdk, "clip.play", {"clip_id": intent.clip_id}, intent.duration_ns)
        if isinstance(intent, MoveIntent):
            return self._run(self.motion_sdk, "motion.move", {"positions": dict(intent.positions)}, intent.duration_ns)
        raise ValueError("unsupported motion intent")

    def light(self, intent):
        rgb = tuple(intent.rgb)
        if self.limiter is not None:
            rgb = self.limiter.limit(self.clock() / NS, rgb)
        # Preserve the current cap even if an injected limiter is faulty.
        if len(rgb) != 3 or not all(finite_number(v) and 0 <= v <= .3 for v in rgb):
            raise ValueError("invalid limited light")
        return self._run(self.light_sdk, "light.glow",
                         {"color": [round(v * 255) for v in rgb], "luminance": .3}, 7 * NS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conductor", required=True, help="Explicit numeric conductor IP; never first-discovered")
    parser.add_argument("--port", type=int, default=47300)
    parser.add_argument("--sdk-base", default="http://127.0.0.1:8081")
    parser.add_argument("--enable-output", action="store_true")
    parser.add_argument("--light-trim-ms", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        address = ipaddress.ip_address(args.conductor)
        if not 1 <= args.port <= 65535 or not 0 <= args.light_trim_ms <= 1000:
            raise ValueError
    except ValueError:
        parser.error("numeric conductor IP, valid port and trim 0..1000 ms required")
    from ftm_session import Session
    from ftm_udp import UdpClient
    sock = socket.socket(socket.AF_INET6 if address.version == 6 else socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    clients = []
    app = None
    try:
        def forbidden(_intent):
            raise RuntimeError("observation-only output")
        motion = light = forbidden
        if args.enable_output:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from safety.flash import FlashLimiter
            from sdk import LampSDK, read_token
            token = read_token()
            first = LampSDK(token, base=args.sdk_base)
            clients.append(first)
            first._ensure_session()
            second = LampSDK(token, base=args.sdk_base)
            clients.append(second)
            # One session and budget, separate HTTP pools. Do not create two
            # sessions or share a requests.Session concurrently across lanes.
            second.http.headers.update(first.http.headers)
            second._session_expires = first._session_expires
            actions = SDKActions(first, second, limiter=FlashLimiter())
            motion, light = actions.motion, actions.light
        dispatcher = Dispatcher(motion, light)
        client = UdpClient(Session("LeLamp"), sock, (str(address), args.port))
        app = LampApp(client, dispatcher, armed=args.enable_output,
                      light_trim_ns=args.light_trim_ms * 1_000_000)
        while True:
            app.step(time.monotonic_ns())
            time.sleep(.01)
    except KeyboardInterrupt:
        return 0
    except Exception:
        print("Lamp app stopped: inspect local configuration or sanitized telemetry.")
        return 1
    finally:
        if app is not None:
            app.close()
        sock.close()
        # Workers may still be observing an admitted action. Closing their HTTP
        # pools here would abandon that observation; process exit is operator-owned.


if __name__ == "__main__":
    raise SystemExit(main())
