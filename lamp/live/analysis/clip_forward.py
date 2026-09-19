"""Re-level a dance clip so it never leans behind the base.

The vendor's `dance` pins base_pitch at -100 around t=2.9 s, which swings the head behind the base
with momentum behind it. Leaning back is the joint going MORE negative, so the fix is a floor on
base_pitch plus a small forward bias, chosen as the mildest pair that keeps every frame's head in
front of the base and the zero-moment point clear of the back edge. Timing is untouched (same
frames, 30 fps) so it stays on the music; every frame is re-checked with the URDF's own limit,
table and self-collision rules before it is written.
"""
import csv, sys, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
sys.path.insert(0, "lamp")
from spatial import JOINTS, LampModel, _rot_z

SP = Path(sys.argv[1]); ROBOT = SP / "robotdesc/pi5_feetech_r1"; OUT = SP / "clips_fwd"; OUT.mkdir(exist_ok=True)
model = LampModel(ROBOT, calibration=SP / "robotdesc/lelamp-calibration.json")
# Geometry checks (table, own base) use the same model, but joint limits are judged at the runtime's
# real +/-100, not the follower's +/-94 cushion: the vendor's own clips reach -100 and are legal.
checker = LampModel(ROBOT, calibration=SP / "robotdesc/lelamp-calibration.json", limit_margin=0.0)
root = ET.parse(ROBOT / "robot.urdf").getroot()
inertials = [(float(l.find("inertial/mass").get("value")),
              np.array([float(v) for v in l.find("inertial/origin").get("xyz").split()]))
             for l in root.findall("link") if l.find("inertial") is not None]
G, FPS = 9.81, 30.0
HEAD_MIN_Y, ZMP_MIN_Y = 0.020, -0.012          # head at least 2 cm in front; ZMP never >1.2 cm behind

def fk(units):
    m, acc, tot, fs = np.eye(4), np.zeros(3), 0.0, [np.eye(4)]
    for motor, offset, origin in model._chain:
        m = m @ origin @ _rot_z((float(units[motor]) - model.neutral[motor]) * model._scale[motor] + offset)
        fs.append(m.copy())
    for (mass, local), f in zip(inertials, fs):
        acc += mass * (f @ np.append(local, 1.0))[:3]; tot += mass
    return acc / tot, fs[-1][:3, 3]

def evaluate(U):
    coms, heads = zip(*(fk(dict(zip(JOINTS, u))) for u in U)); coms, heads = np.array(coms), np.array(heads)
    k = np.ones(5) / 5
    sm = np.stack([np.convolve(coms[:, i], k, mode="same") for i in range(3)], 1)
    acc = np.gradient(np.gradient(sm, 1 / FPS, axis=0), 1 / FPS, axis=0)
    zmp_y = sm[:, 1] - (sm[:, 2] / G) * acc[:, 1]
    bad = [p for u in U for p in checker.problems(dict(zip(JOINTS, u)))]
    return heads[:, 1].min(), zmp_y.min(), len(bad), (bad[0] if bad else "")

bp = JOINTS.index("base_pitch")
for clip in ("dance", "robot_dance"):
    src = ROBOT / f"animations/factory_v1/{clip}.csv"
    rows = list(csv.DictReader(open(src))); fields = list(rows[0].keys())
    U0 = np.array([[float(r[f"{j}.pos"]) for j in JOINTS] for r in rows])
    h0, z0, _, _ = evaluate(U0)
    print(f"=== {clip}: original head_y min {h0:+.3f}  zmp_y min {z0:+.3f}")
    chosen = None
    for bias in (0, 4, 8, 12):
        for floor in (-100, -85, -80, -75, -70, -65, -60):
            U = U0.copy()
            U[:, bp] = np.maximum(U[:, bp] + bias, floor)
            U[:, bp] = np.clip(U[:, bp], *model.limits["base_pitch"])
            hy, zy, nbad, why = evaluate(U)
            ok = hy >= HEAD_MIN_Y and zy >= ZMP_MIN_Y and nbad == 0
            if clip == "dance" and floor in (-100, -75, -65) and bias in (0, 8):
                print(f"     try floor {floor:>4} bias +{bias:<2}: head_y {hy:+.3f}  zmp_y {zy:+.3f}  urdf-violations {nbad}"
                      + (f"  e.g. {why}" if nbad else ""))
            if ok and chosen is None:
                chosen = (bias, floor, U, hy, zy)
                print(f"  -> mildest fix: base_pitch floor {floor}, forward bias +{bias}: head_y min {hy:+.3f}, zmp_y min {zy:+.3f}, URDF checks clean")
    if chosen is None:
        print("  no floor/bias combination satisfied the margins; leaving clip unchanged"); continue
    bias, floor, U, hy, zy = chosen
    changed = int((np.abs(U[:, bp] - U0[:, bp]) > 0.5).sum())
    with open(OUT / f"{clip}_fwd.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r, u in zip(rows, U):
            r = dict(r)
            for n, j in enumerate(JOINTS): r[f"{j}.pos"] = f"{u[n]:.4f}"
            w.writerow(r)
    print(f"  wrote {clip}_fwd.csv: {changed}/{len(rows)} frames moved, timing unchanged")
