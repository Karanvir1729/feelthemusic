"""Same wiggle, but WITHOUT hammering the servo bus with position reads while the plan runs."""
import json, time, urllib.request
B = "http://127.0.0.1:8081"
J = ("base_yaw", "base_pitch", "elbow_pitch", "wrist_roll", "wrist_pitch")
def call(p, body=None, timeout=8):
    d = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(B + p, data=d, headers={"Content-Type": "application/json"} if d else {}, method="POST" if d else "GET")
    with urllib.request.urlopen(r, timeout=timeout) as f: return json.loads(f.read())
pos = lambda: {k: v for k, v in call("/api/motors/positions")["positions"].items() if k in J}
start = pos(); print("start", {k: round(v, 1) for k, v in start.items()}, flush=True)
for delta, dur, reads in ((12, 400, 0), (0, 400, 0), (25, 500, 0), (0, 500, 0), (12, 400, 5), (0, 400, 5)):
    target = dict(start); target["base_yaw"] = start["base_yaw"] + delta
    y0 = pos()["base_yaw"]; t0 = time.monotonic()
    r = call("/api/motors/positions", {"positions": target, "duration_ms": dur})
    trace = []
    for i in range(reads):                      # 0 = do not touch the bus during the move
        time.sleep(0.1); trace.append((time.monotonic() - t0, pos()["base_yaw"]))
    time.sleep(1.2); y1 = pos()["base_yaw"]
    print(f"yaw {y0:+.1f} -> {target['base_yaw']:+.1f} in {dur} ms, {reads} reads during: landed {y1:+.1f} "
          f"({100*(y1-y0)/(target['base_yaw']-y0) if abs(target['base_yaw']-y0)>0.5 else 100:.0f}% of the way) {r.get('status') or r.get('error')}"
          + ("  trace " + " ".join(f"{1000*t:.0f}:{y:+.1f}" for t, y in trace) if trace else ""), flush=True)
