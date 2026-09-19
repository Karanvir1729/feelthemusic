#!/usr/bin/env python3
"""Take the arm off idle and keep it off, by owning the motion lease.

Why this instead of /api/animations/idle: that route sends a BehaviorIntent with ttl=1.0 and the
runtime reinstates idle a second later (_resume_idle_after_tracking). And config cannot disable it
either -- boot.py does `default_idle_animation or getattr(config, ..., "idle")`, so an empty string
is falsy and falls straight back to "idle".

What does work is priority. A direct position command enters as category "tracking", priority 85,
with stop_before_execute=True; idle is priority 20. So idle can never take the arm back while
somebody keeps refreshing that lease. This refreshes it faster than its own ttl (duration_ms/1000 + 1s).

This holds the pose the arm is ALREADY in -- it is an idle suppressor, not a mover. Point your own
controller at hold_pose() to drive it somewhere.

  python3 hold.py            # hold current pose, suppress idle, report drift
  python3 hold.py 60         # for 60 seconds
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8081"
JOINTS = ("base_yaw", "base_pitch", "elbow_pitch", "wrist_roll", "wrist_pitch")
REFRESH_S = 0.8           # must stay under the lease ttl, which is duration_ms/1000 + 1 s
DURATION_MS = 1000


def call(path: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        BASE + path, data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(request, timeout=6) as response:
        return json.loads(response.read())


def measured() -> dict[str, float]:
    return {k: v for k, v in call("/api/motors/positions")["positions"].items() if k in JOINTS}


def hold_pose(pose: dict[str, float]) -> dict:
    # Every joint, every time: a joint left out is held at its MEASURED position, and on a
    # gravity-loaded joint that is a little lower each round.
    return call("/api/motors/positions", {"positions": pose, "duration_ms": DURATION_MS})


def main() -> int:
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
    anchor = measured()
    print("holding:", {k: round(v, 1) for k, v in anchor.items()})

    deadline = time.monotonic() + seconds
    worst = 0.0
    rounds = 0
    while time.monotonic() < deadline:
        try:
            hold_pose(anchor)
        except Exception as exc:
            print(f"  lease refresh failed: {exc}")
        rounds += 1
        time.sleep(REFRESH_S)
        now = measured()
        drift = max(abs(now[j] - anchor[j]) for j in anchor)
        worst = max(worst, drift)
        print(f"  t={time.monotonic():.0f} drift={drift:5.2f} worst={worst:5.2f}")

    print(f"\n{rounds} lease refreshes, worst drift {worst:.2f} units")
    print("under ~2 units = idle is suppressed; tens of units = idle is still winning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
