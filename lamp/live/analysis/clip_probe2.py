import hashlib, json, os, time, urllib.request
B = "http://127.0.0.1:8081"
J = ("base_yaw", "base_pitch", "elbow_pitch", "wrist_roll", "wrist_pitch")
PACK = os.path.expanduser("~/lelamp-hackathon-2026/static/robots/lelamp_v1/pi5_feetech_r1/animations/factory_v1")
def call(p, body=None, timeout=8):
    d = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(B + p, data=d, headers={"Content-Type": "application/json"} if d else {}, method="POST" if d else "GET")
    with urllib.request.urlopen(r, timeout=timeout) as f: return json.loads(f.read())
pos = lambda: {k: v for k, v in call("/api/motors/positions")["positions"].items() if k in J}
start = pos()
rows, t = [], 0.0
def seg(secs, f):
    global t
    n = int(round(secs * 30))
    for i in range(n): rows.append((t, f(i / max(1, n - 1)))); t += 1 / 30
# hold 0.6 | +14 over 0.5 | hold 0.5 | -14 over 0.5 | hold 0.5 | +14 over 0.5 | hold 0.5 | back over 0.5 | hold 0.6  (4.7 s)
seg(0.6, lambda u: 0.0); seg(0.5, lambda u: 14 * u); seg(0.5, lambda u: 14.0); seg(0.5, lambda u: 14 * (1 - u)); seg(0.5, lambda u: 0.0)
seg(0.5, lambda u: 14 * u); seg(0.5, lambda u: 14.0); seg(0.5, lambda u: 14 * (1 - u)); seg(0.6, lambda u: 0.0)
path, tmp = os.path.join(PACK, "beat_test.csv"), os.path.join(PACK, ".beat_test.csv.tmp")
with open(tmp, "w") as f:
    f.write("timestamp," + ",".join(f"{j}.pos" for j in J) + "\n")
    for ts, dy in rows:
        p = dict(start); p["base_yaw"] += dy
        f.write(f"{1000 + ts:.9f}," + ",".join(f"{p[j]:.4f}" for j in J) + "\n")
    f.flush(); os.fsync(f.fileno())
os.replace(tmp, path); os.sync()
print(f"clip {len(rows)} frames {t:.1f} s; arm at yaw {start['base_yaw']:+.1f}", flush=True)
for trial in range(3):
    y0 = pos()["base_yaw"]; t0 = time.monotonic()
    r = call("/api/animations/play", {"name": "beat_test"})
    tr, st_log = [], []
    while time.monotonic() - t0 < 5.6:
        tr.append((time.monotonic() - t0, pos()["base_yaw"]))
        if len(tr) % 10 == 1:
            st = call("/api/animations/status"); st_log.append((round(time.monotonic() - t0, 1), st.get("current_animation"), st.get("playing"), st.get("elapsed_seconds")))
        time.sleep(0.02)
    print(f"trial {trial}: {r.get('status') or r.get('error')} | yaw trace (ms:+d): " + " ".join(f"{1000*ts:.0f}:{y-y0:+.0f}" for ts, y in tr[::8]), flush=True)
    print(f"   status: " + " ".join(f"{a}s:{b}/{'P' if c else '-'}/{d}" for a, b, c, d in st_log[::3]), flush=True)
    time.sleep(2.0)
