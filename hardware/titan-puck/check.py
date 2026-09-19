"""Fit check against the board, printability numbers, and the assembly for rendering.

  uv run --python 3.12 --with cadquery --with trimesh --with numpy python check.py

The board is the vendor STEP when TITAN_STEP points at it (not redistributable), otherwise the keep-out envelope in
titan_core_keepout.json (made by gen_keepout.py; at least as big as the real board everywhere, so 0 against it
means 0 against the board). MOTOR_D / MOTOR_L override the motor size checked (for example the end caps measured
from a photo): MOTOR_D=10.4 python check.py
"""
import json, os
import numpy as np, trimesh, cadquery as cq
import model as M

STEP = os.environ.get("TITAN_STEP", "")
base, lid = M.base(), M.lid()
if STEP and os.path.exists(STEP):
    board = cq.importers.importStep(STEP).translate((0, 0, M.PCB_TOP)); print(f"board: vendor STEP {STEP}")
else:
    boxes = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "titan_core_keepout.json")))["boxes"]
    board = cq.Workplane().add(cq.Compound.makeCompound(
        [cq.Solid.makeBox(x1 - x0, y1 - y0, z1 - z0, cq.Vector(x0, y0, z0)) for x0, x1, y0, y1, z0, z1 in boxes])).translate((0, 0, M.PCB_TOP))
    print(f"board: keep-out envelope ({len(boxes)} boxes)")
M.MOTOR_D = float(os.environ.get("MOTOR_D", M.MOTOR_D)); M.MOTOR_L = float(os.environ.get("MOTOR_L", M.MOTOR_L))
print(f"motors checked as {M.MOTOR_D} mm x {M.MOTOR_L} mm")

def motor_lying(mx, my):
    return cq.Workplane("XZ").center(mx, M.MOTOR_Z).circle(M.MOTOR_D / 2).extrude(-M.MOTOR_L).translate((0, my, 0))
lx, ly = M.LFI
lfi = cq.Workplane("XY").circle(M.MOTOR_D / 2).extrude(M.MOTOR_L).translate((lx, ly, 0.0))
motors = {"LF": motor_lying(*M.LF), "MF": motor_lying(*M.MF), "LFi": lfi}
# USB-C plug: 8.3 x 2.5 tongue into the receptacle + 12.5 x 6.5 overmould from the receptacle mouth outward
plug = (cq.Workplane("XY").box(12.5, 20, 6.5).translate((M.USB_X, -1.34 - 0.2 - 10, M.PCB_TOP + 1.6)))

def vol(a, b):
    try:
        return a.intersect(b).val().Volume()
    except Exception:
        return 0.0

print("== interference volumes (mm3), must be 0")
for pname, part in (("base", base), ("lid", lid)):
    print(f"  board x {pname}: {vol(board, part):.3f}")
    for n, m in motors.items():
        print(f"  {n} x {pname}: {vol(m, part):.3f}")
    print(f"  USB plug overmould x {pname}: {vol(plug, part):.3f}")
print(f"  lid x base (closed): {vol(lid, base):.3f}")
print(f"  board x motors: {sum(vol(board, m) for m in motors.values()):.3f}")

def dist(a, b):
    return a.val().distance(b.val())[0] if hasattr(a.val(), 'distance') else float('nan')
print("== minimum clearances (mm)")
import OCP
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
def mind(a, b):
    d = BRepExtrema_DistShapeShape(a.val().wrapped, b.val().wrapped); d.Perform(); return d.Value()
print(f"  board to base: {mind(board, base):.2f}   board to lid: {mind(board, lid):.2f}")
for n, m in motors.items():
    print(f"  {n}: to base {mind(m, base):.2f}, to lid {mind(m, lid):.2f}, to board {mind(m, board):.2f}")
print(f"  LFi top end to lid underside: {M.INNER_H - M.MOTOR_L:.2f}")
print(f"  LF/MF ends to -Y wall: {M.LF[1] - M.IY0:.2f}; MF end to LFi sleeve: {np.hypot(M.LFI[0]-M.MF[0], M.LFI[1]-(M.MF[1]+M.MOTOR_L)) - (M.CRADLE_D/2+1.6):.2f}")

print("== printability")
for f in ("puck_base.stl", "puck_lid_print.stl"):
    m = trimesh.load(f)
    zmin = m.bounds[0][2]
    n = m.face_normals; c = m.triangles_center
    down = (n[:, 2] < -np.cos(np.radians(45))) & (c[:, 2] > zmin + 0.3)
    areas = m.area_faces
    print(f"  {f}: watertight {m.is_watertight}, volume {m.volume/1000:.1f} cm3 (~{m.volume/1000*1.24:.0f} g PLA solid), "
          f"size {np.ptp(m.bounds, axis=0).round(1)}, bed contact {areas[(n[:,2] < -0.999) & (c[:,2] < zmin + 0.05)].sum():.0f} mm2, "
          f"down-facing >45deg not on bed: {areas[down].sum():.0f} mm2")
    zs = c[down][:, 2]; 
    if len(zs):
        hist = np.histogram(zs, bins=[zmin, zmin+2, zmin+6, zmin+10, zmin+14, zmin+20, zmin+30])
        print("     where (z bands above bed):", [(round(float(hist[1][i]-zmin),0), round(float(areas[down][(zs>=hist[1][i])&(zs<hist[1][i+1])].sum()),0)) for i in range(len(hist[0]))])

# assembly meshes for rendering
cq.exporters.export(board, "asm_board.stl", tolerance=0.1)
for n, m in motors.items(): cq.exporters.export(m, f"asm_{n}.stl", tolerance=0.05)
cq.exporters.export(lid, "asm_lid.stl", tolerance=0.05)
