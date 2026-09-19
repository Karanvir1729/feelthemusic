"""Where does each frame of a dance clip put the lamp's weight -- including momentum?

Static centre of mass is not the whole story: a clip swings ~0.5 kg at the end of the arm at
30 fps, and the acceleration of that mass adds an overturning moment. The zero-moment point
approximates the point where the ground reaction must act: ZMP = CoM_xy - (z_com / g) * a_xy.
While the ZMP stays inside the foot the lamp stays down; behind the base (negative y) is the
direction the operator can feel it wanting to go.
"""
import csv, math, sys, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
sys.path.insert(0, "lamp")
from spatial import JOINTS, LampModel, _rot_z

SP = Path(sys.argv[1]); ROBOT = SP / "robotdesc/pi5_feetech_r1"
model = LampModel(ROBOT, calibration=SP / "robotdesc/lelamp-calibration.json")
root = ET.parse(ROBOT / "robot.urdf").getroot()
inertials = []
for link in root.findall("link"):
    it = link.find("inertial")
    if it is None: continue
    o = it.find("origin")
    inertials.append((float(it.find("mass").get("value")), np.array([float(v) for v in o.get("xyz").split()])))
G, FPS = 9.81, 30.0
FOOT = {"safety.yaml cylinder": 0.10, "conservative": 0.06}

def com_and_head(units):
    m, acc, tot, fs = np.eye(4), np.zeros(3), 0.0, [np.eye(4)]
    for motor, offset, origin in model._chain:
        m = m @ origin @ _rot_z((float(units[motor]) - model.neutral[motor]) * model._scale[motor] + offset)
        fs.append(m.copy())
    for (mass, local), f in zip(inertials, fs):
        acc += mass * (f @ np.append(local, 1.0))[:3]; tot += mass
    return acc / tot, fs[-1][:3, 3]

for clip in ("dance", "robot_dance"):
    rows = list(csv.DictReader(open(ROBOT / f"animations/factory_v1/{clip}.csv")))
    U = np.array([[float(r[f"{j}.pos"]) for j in JOINTS] for r in rows])
    coms, heads = [], []
    for u in U:
        c, h = com_and_head(dict(zip(JOINTS, u))); coms.append(c); heads.append(h)
    coms, heads = np.array(coms), np.array(heads)
    # smooth then differentiate twice for acceleration
    k = np.ones(5) / 5
    sm = np.stack([np.convolve(coms[:, i], k, mode="same") for i in range(3)], 1)
    vel = np.gradient(sm, 1 / FPS, axis=0); acc = np.gradient(vel, 1 / FPS, axis=0)
    zmp = sm[:, :2] - (sm[:, 2:3] / G) * acc[:, :2]
    jv = np.abs(np.gradient(U, 1 / FPS, axis=0))
    print(f"=== {clip}: {len(rows)} frames, {len(rows)/FPS:.1f} s ===")
    print(f"  head y (forward +):   min {heads[:,1].min():+.3f}  max {heads[:,1].max():+.3f} m   "
          f"frames with head BEHIND base (y<0): {(heads[:,1] < 0).sum()}")
    print(f"  CoM y  static:        min {coms[:,1].min():+.3f}  max {coms[:,1].max():+.3f} m")
    print(f"  ZMP y  with momentum: min {zmp[:,1].min():+.3f}  max {zmp[:,1].max():+.3f} m   <- backward push is the negative side")
    r = np.hypot(zmp[:, 0], zmp[:, 1])
    for name, foot in FOOT.items():
        print(f"  ZMP outside a {foot*100:.0f} cm foot ({name}): {(r > foot).sum()} frames   worst margin {foot - r.max():+.3f} m")
    worst = np.argsort(zmp[:, 1])[:3]
    print("  three most backward frames (t, ZMP y, head y, joints):")
    for i in worst:
        print(f"    t={i/FPS:5.2f}s  zmp_y={zmp[i,1]:+.3f}  head_y={heads[i,1]:+.3f}  " +
              "  ".join(f"{j[:5]}={U[i,n]:+.0f}" for n, j in enumerate(JOINTS)))
    print(f"  peak joint speed (units/s): " + "  ".join(f"{j[:5]}={jv[:,n].max():.0f}" for n, j in enumerate(JOINTS)) +
          "   (design 140, SDK refuses >300)")
    print()
