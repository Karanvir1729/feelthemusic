"""Motor fit gauge: a 10-minute print that measures the real DRAKE motors before the puck is printed.

Push each motor END CAP (the widest part) into the holes from the smallest up. The first hole it enters with a light
push is its diameter; the puck's cradle should be that hole + 0.4 mm. The slot at the end checks the length: the
motor must drop in lengthwise (slot 23.6 mm) and must NOT fit the 22.4 mm notch beside it.

  uv run --python 3.12 --with cadquery python gauge.py      -> motor_gauge.stl (print flat, no supports)
"""
import cadquery as cq

HOLES = (9.6, 9.9, 10.2, 10.5, 10.8, 11.1, 11.4)
T, PITCH, W = 5.0, 13.5, 16.0
L = PITCH * len(HOLES) + 3
SLOT_L, SLOT_W, NOTCH_L = 23.6, 12.0, 22.4

g = cq.Workplane("XY").box(L, W, T, centered=(False, True, False))
for i, d in enumerate(HOLES):
    x = 8 + i * PITCH
    g = g.cut(cq.Workplane("XY").circle(d / 2).extrude(T + 2).translate((x, 1.5, -1)))
    g = g.union(cq.Workplane("XY").text(f"{d:.1f}", 3.2, 0.6, halign="center", valign="center").translate((x, -6.0, T)))
# length strip: a slot the motor must lie in, and a shorter notch it must not
strip = cq.Workplane("XY").box(SLOT_L + NOTCH_L + 12, 18, T, centered=False).translate((0, W / 2 - 0.5, 0))
strip = strip.cut(cq.Workplane("XY").box(SLOT_L, SLOT_W, T, centered=False).translate((3, W / 2 + 3.5, 1.2)))
strip = strip.cut(cq.Workplane("XY").box(NOTCH_L, SLOT_W, T, centered=False).translate((SLOT_L + 9, W / 2 + 3.5, 1.2)))
strip = strip.union(cq.Workplane("XY").text("23.6 fits", 2.6, 0.6, halign="left", valign="center").translate((3, W / 2 + 1.5, T)))
strip = strip.union(cq.Workplane("XY").text("22.4 must not", 2.6, 0.6, halign="left", valign="center").translate((SLOT_L + 9, W / 2 + 1.5, T)))
g = g.union(strip)
cq.exporters.export(g, "motor_gauge.stl", tolerance=0.02, angularTolerance=0.1)
bb = g.val().BoundingBox()
print(f"motor_gauge.stl {bb.xlen:.1f} x {bb.ylen:.1f} x {bb.zlen:.1f} mm")
