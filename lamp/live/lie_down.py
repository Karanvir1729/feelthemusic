import json, time, urllib.request
B = "http://127.0.0.1:8081"
def pos():
    return {k: float(v) for k, v in json.load(urllib.request.urlopen(B + "/api/motors/positions", timeout=4))["positions"].items()}
# the pose the arm boots into and rests in with torque off: folded down, head on its cradle (measured after reboot)
REST = {"base_yaw": 5.0, "base_pitch": -91.0, "elbow_pitch": -95.0, "wrist_roll": 0.0, "wrist_pitch": 63.0}
d = json.dumps({"positions": REST, "duration_ms": 4000}).encode()
urllib.request.urlopen(urllib.request.Request(B + "/api/motors/positions", data=d, headers={"Content-Type": "application/json"}), timeout=6).read()
time.sleep(4.8)
p = pos(); print("lying down: " + " ".join("%s=%+.1f" % (k, v) for k, v in sorted(p.items())), flush=True)
