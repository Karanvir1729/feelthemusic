import json, time, urllib.request
B = "http://127.0.0.1:8081"
def get(p):
    with urllib.request.urlopen(B + p, timeout=4) as f: return json.loads(f.read())
print(time.strftime("%H:%M:%S"), "tracer v2: floor measured only after the 3 s blend-in; waiting up to 6 min", flush=True)
t_end = time.time() + 360; seen = 0
while time.time() < t_end and seen < 3:
    try: st = get("/api/animations/status")
    except Exception: time.sleep(1); continue
    name = st.get("current_animation")
    if name in ("dance_fwd", "robot_dance_fwd") and st.get("playing") and (st.get("elapsed_seconds") or 0) < 1.0:
        seen += 1; t0 = time.time()
        try: start = get("/api/motors/positions")["positions"]
        except Exception: start = {}
        lo = {}
        while True:
            try:
                p = get("/api/motors/positions")["positions"]
                if time.time() - t0 >= 3.0:
                    for k, v in p.items(): lo[k] = min(lo.get(k, v), v)
                st = get("/api/animations/status")
            except Exception: pass
            if not st.get("playing") or st.get("current_animation") != name or time.time() - t0 > 16: break
            time.sleep(0.4)
        floor = -65 if name == "dance_fwd" else -100
        bp = lo.get("base_pitch")
        print(f"{time.strftime('%H:%M:%S')} clip #{seen} {name}: started from base_pitch {start.get('base_pitch', float('nan')):+.0f}; "
              f"post-blend base_pitch min {bp:+.0f} vs floor {floor} -> {'FLOOR HELD' if bp is not None and bp >= floor - 8 else 'PAST THE FLOOR'}; "
              f"elbow {lo.get('elbow_pitch', 0):+.0f} wrist_pitch {lo.get('wrist_pitch', 0):+.0f}", flush=True)
    time.sleep(0.3)
print(time.strftime("%H:%M:%S"), "tracer v2 done", flush=True)
