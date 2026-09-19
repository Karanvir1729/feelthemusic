import csv, sys, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
sys.path.insert(0, "lamp")
from spatial import JOINTS, LampModel, _rot_z
SP = Path(sys.argv[1]); ROBOT = SP / "robotdesc/pi5_feetech_r1"
model = LampModel(ROBOT, calibration=SP / "robotdesc/lelamp-calibration.json")
checker = LampModel(ROBOT, calibration=SP / "robotdesc/lelamp-calibration.json", limit_margin=0.0)
root = ET.parse(ROBOT / "robot.urdf").getroot()
inertials = [(float(l.find("inertial/mass").get("value")), np.array([float(v) for v in l.find("inertial/origin").get("xyz").split()]))
             for l in root.findall("link") if l.find("inertial") is not None]
G, FPS = 9.81, 30.0
def fk(units):
    m, acc, tot, fs = np.eye(4), np.zeros(3), 0.0, [np.eye(4)]
    for motor, offset, origin in model._chain:
        m = m @ origin @ _rot_z((float(units[motor]) - model.neutral[motor]) * model._scale[motor] + offset); fs.append(m.copy())
    for (mass, local), f in zip(inertials, fs):
        acc += mass * (f @ np.append(local, 1.0))[:3]; tot += mass
    return acc / tot, fs[-1][:3, 3]
def evaluate(U):
    coms, heads = zip(*(fk(dict(zip(JOINTS, u))) for u in U)); coms, heads = np.array(coms), np.array(heads)
    k = np.ones(5) / 5; sm = np.stack([np.convolve(coms[:, i], k, mode="same") for i in range(3)], 1)
    acc = np.gradient(np.gradient(sm, 1 / FPS, axis=0), 1 / FPS, axis=0)
    zmp_y = sm[:, 1] - (sm[:, 2] / G) * acc[:, 1]
    bad = sum(1 for u in U if checker.problems(dict(zip(JOINTS, u))))
    jv = np.abs(np.gradient(U, 1 / FPS, axis=0)).max()
    return heads[:, 1].min(), zmp_y.min(), bad, jv
def smooth_where(x, mask, width=7):
    """Moving-average only across corrected frames (+/-3), so clamps do not leave corners."""
    y = x.copy(); k = np.ones(width) / width; s = np.convolve(x, k, mode="same")
    grow = np.convolve(mask.astype(float), np.ones(width), mode="same") > 0
    y[grow] = s[grow]; return y

bp_i, el_i = JOINTS.index("base_pitch"), JOINTS.index("elbow_pitch")
rows = list(csv.DictReader(open(ROBOT / "animations/factory_v1/dance.csv"))); fields = list(rows[0].keys())
U0 = np.array([[float(r[f"{j}.pos"]) for j in JOINTS] for r in rows])
print(f"  {'bias':>5}{'BACK':>6}{'CAP':>6}{'touched':>9}{'head_y':>9}{'zmp_y':>8}{'peak u/s':>10}   verdict")
best = None
for bias in (0, 4, 8):
    for back in (-80, -75, -70):
        for cap in (-60, -50, -40):
            U = U0.copy(); U[:, bp_i] = np.clip(U[:, bp_i] + bias, -100, 100)
            mask = U[:, bp_i] < back
            U[mask, bp_i] = back; U[mask, el_i] = np.minimum(U[mask, el_i], cap)
            U[:, bp_i] = smooth_where(U[:, bp_i], mask); U[:, el_i] = smooth_where(U[:, el_i], mask)
            hy, zy, bad, jv = evaluate(U)
            ok = hy >= 0.020 and zy >= -0.012 and bad == 0
            touched = int((np.abs(U - U0).max(axis=1) > 0.5).sum())
            if ok or (bias == 8 and cap == -40):
                print(f"  {bias:>+5}{back:>6}{cap:>6}{touched:>9}{hy:>+9.3f}{zy:>+8.3f}{jv:>10.0f}   {'OK' if ok else '--'}")
            if ok and (best is None or touched < best[0]): best = (touched, bias, back, cap, U, hy, zy)
if best:
    touched, bias, back, cap, U, hy, zy = best
    with open(SP / "clips_fwd/dance_fwd2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r, u in zip(rows, U):
            r = dict(r); [r.__setitem__(f"{j}.pos", f"{u[k]:.4f}") for k, j in enumerate(JOINTS)]; w.writerow(r)
    print(f"\nchosen: bias +{bias}, when leaning back past {back}: hold shoulder at {back} and elbow <= {cap}, smoothed."
          f"\n  {touched}/371 frames touched (flat version: 371/371)  head_y min {hy:+.3f}  zmp_y min {zy:+.3f}  -> dance_fwd2.csv")
else:
    print("\nno combination met the margins; the flat floor version (already on the lamp) stays")
