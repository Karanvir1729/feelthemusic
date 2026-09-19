"""Is the 40-60 % shortfall yaw-only? Same direct 5..8-unit writes on the unloaded and the loaded joints."""
import json, time, urllib.request
B = "http://127.0.0.1:8081"
J = ("base_yaw", "base_pitch", "elbow_pitch", "wrist_roll", "wrist_pitch")
def call(p, body=None, timeout=8):
    d = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(B + p, data=d, headers={"Content-Type": "application/json"} if d else {}, method="POST" if d else "GET")
    with urllib.request.urlopen(r, timeout=timeout) as f: return json.loads(f.read())
pos = lambda: {k: v for k, v in call("/api/motors/positions")["positions"].items() if k in J}
start = pos(); print("start", {k: round(v, 1) for k, v in start.items()}, flush=True)
for joint, deltas in (("wrist_roll", (8, 0, -8, 0)), ("wrist_pitch", (8, 0, -8, 0)), ("base_pitch", (-6, 0, 6, 0))):
    for delta in deltas:
        target = dict(start); target[joint] = start[joint] + delta
        p0 = pos()[joint]; t0 = time.monotonic()
        r = call("/api/motors/positions", {"positions": target, "duration_ms": 300})
        time.sleep(1.0); p1 = pos()[joint]
        want = target[joint] - p0
        print(f"{joint:12s} {p0:+6.1f} -> {target[joint]:+6.1f}: landed {p1:+6.1f} = {100*(p1-p0)/want if abs(want) > 0.5 else 100:4.0f}%  {r.get('status') or r.get('error')}", flush=True)
