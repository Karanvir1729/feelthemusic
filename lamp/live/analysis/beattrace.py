"""Measure what the arm really does during a beat clip: per-joint achieved/commanded amplitude ratio and the
lag of the servos behind the clip's own timeline (cross-correlation), so the generator's gains and the
scheduler's start latency come from the hardware, not from guesses."""
import csv, json, sys, time, urllib.request
import numpy as np
B = "http://127.0.0.1:8081"; J = ("base_yaw", "base_pitch", "elbow_pitch", "wrist_roll", "wrist_pitch")
PACK = "/home/lelamp/lelamp-hackathon-2026/static/robots/lelamp_v1/pi5_feetech_r1/animations/factory_v1"
def get(p):
    with urllib.request.urlopen(B + p, timeout=3) as f: return json.load(f)
want = int(sys.argv[1]) if len(sys.argv) > 1 else 2
print(time.strftime("%H:%M:%S"), f"waiting for {want} beat clips (up to 4 min)", flush=True)
seen, last_start, t_end = 0, None, time.time() + 240
while seen < want and time.time() < t_end:
    try: st = get("/api/animations/status")
    except Exception: time.sleep(0.2); continue
    name, started = st.get("current_animation") or "", st.get("started_at")
    if not name.startswith("beat_") or started == last_start:
        time.sleep(0.1); continue
    last_start = started; seen += 1
    t0 = time.time(); rows = list(csv.DictReader(open(f"{PACK}/{name}.csv")))
    cmd_t = np.array([float(r["timestamp"]) for r in rows]); cmd_t -= cmd_t[0]
    cmd = {j: np.array([float(r[f"{j}.pos"]) for r in rows]) for j in J}
    dur = cmd_t[-1] + 1.0
    ts, pos = [], {j: [] for j in J}
    while time.time() - t0 < dur:
        try:
            p = get("/api/motors/positions")["positions"]; ts.append(time.time() - t0)
            for j in J: pos[j].append(float(p[j]))
        except Exception: pass
        time.sleep(0.02)
    ts = np.array(ts)
    print(f"\n{time.strftime('%H:%M:%S')} clip {name}: {len(rows)} frames {cmd_t[-1]:.1f}s, {len(ts)} samples ({len(ts)/dur:.0f} Hz), started_at->first sample {(t0 - started)*1000:+.0f} ms")
    print(f"  {'joint':12} {'cmd p-p':>8} {'got p-p':>8} {'ratio':>6} {'lag ms':>7} {'rms err':>8}")
    for j in J:
        c = cmd[j] - cmd[j].mean(); g = np.array(pos[j]) - np.mean(pos[j])
        if np.ptp(cmd[j]) < 2: print(f"  {j:12} {np.ptp(cmd[j]):8.1f} {np.ptp(pos[j]):8.1f}      -       -        -"); continue
        best = (1e9, 0)
        for lag in np.arange(0.0, 1.2, 0.01):
            ci = np.interp(ts - lag, cmd_t, c, left=c[0], right=c[-1]); e = float(np.sqrt(np.mean((g - ci) ** 2)))
            if e < best[0]: best = (e, lag)
        ci = np.interp(ts - best[1], cmd_t, c, left=c[0], right=c[-1])
        ratio = float(np.std(g) / max(1e-6, np.std(ci)))
        print(f"  {j:12} {np.ptp(cmd[j]):8.1f} {np.ptp(pos[j]):8.1f} {ratio:6.2f} {best[1]*1000:7.0f} {best[0]:8.1f}")
print(time.strftime("%H:%M:%S"), "done", flush=True)
