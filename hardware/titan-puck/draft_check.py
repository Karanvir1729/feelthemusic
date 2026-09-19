"""Quick independent fit check of the draft puck: motors with feet at extreme sizes, board keep-out, lid closed."""
import itertools
import json

import cadquery as cq

import model as M

base = M.base()
lid = M.lid()
boxes = json.load(open("titan_core_keepout.json"))["boxes"]
board = cq.Workplane().add(cq.Compound.makeCompound(
    [cq.Solid.makeBox(x1 - x0, y1 - y0, z1 - z0, cq.Vector(x0, y0, z0)) for x0, x1, y0, y1, z0, z1 in boxes])).translate((0, 0, M.PCB_TOP))


def vol(a, b):
    try:
        r = a.intersect(b)
        return r.val().Volume(), r
    except Exception as e:  # fail loudly: an error is not "0"
        return float("nan"), repr(e)


def lying(mx, myc, D, prot, fw, fstart, L=23.1):
    y0 = myc - L / 2
    zc = prot + D / 2
    body = cq.Workplane("XZ").center(mx, zc).circle(D / 2).extrude(-L).translate((0, y0, 0))
    for s in (fstart, L - fstart - M.FOOT_LEN):
        foot = cq.Workplane("XY").box(fw, M.FOOT_LEN, prot + 1.0, centered=(True, False, False)).translate((mx, y0 + s, 0.0))
        body = body.union(foot)
    return body


def standing(x, y, D, prot, fw, fstart, L=23.1):
    body = cq.Workplane("XY").circle(D / 2).extrude(L).translate((x, y, 0))
    for s in (fstart, L - fstart - M.FOOT_LEN):
        foot = cq.Workplane("XY").box(prot + 1.0, fw, M.FOOT_LEN, centered=(False, True, False)).translate((x + D / 2 - 1.0, y, s))
        body = body.union(foot)
    return body


print(f"board x base {vol(board, base)[0]:.3f}  board x lid {vol(board, lid)[0]:.3f}  lid x base {vol(lid, base)[0]:.3f}", flush=True)
for D, prot, fw, fs in itertools.product((9.5, 10.2), (0.8, 1.5), (3.0, 8.0), (M.FOOT_START - 0.5, M.FOOT_START + 0.5)):
    ms = {"LF": lying(*M.LF, D, prot, fw, fs), "MF": lying(*M.MF, D, prot, fw, fs), "LFi": standing(*M.LFI, D, prot, fw, fs)}
    row = []
    for n, m in ms.items():
        vb, _ = vol(m, base)
        vl, rl = vol(m, lid)
        vbd, _ = vol(m, board)
        zl = ""
        if hasattr(rl, "val") and vl > 0.001:
            bb = rl.val().BoundingBox()
            zl = f"[lid z {bb.zmin:.1f}-{bb.zmax:.1f} x {bb.xmin:.1f}-{bb.xmax:.1f} y {bb.ymin:.1f}-{bb.ymax:.1f}]"
        row.append(f"{n}: base {vb:.3f} lid {vl:.3f}{zl} board {vbd:.3f}")
    print(f"cap {D} foot {prot} w {fw} start {fs:.1f} | " + " | ".join(row), flush=True)
print("spring posts free at z", M.SPRING_FREE_Z, "max deflection", M.SPRING_MAX_DEFLECTION, "LF", M.LF, "MF", M.MF, "LFI", M.LFI)
