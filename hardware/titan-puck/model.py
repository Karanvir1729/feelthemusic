"""TITAN haptic puck, quick version (v1): USB-C powered, TITAN Core + DRAKE LF / LFi / MF.

Frame (mm): floor top z = 0 (the skin face is the underside, z = -FLOOR). Board coordinates are the vendor STEP's:
board X 0..20.32, Y 0..43.31, PCB top at z = 0 in the STEP, USB-C overhangs to Y = -1.34 on the top side.
Here the board is lifted so its PCB top sits at z = PCB_TOP.

Measured from the vendor STEP (height maps, 0.5 mm cells):
- underside is bare PCB along both long edges for Y 0..13.5 (USB half) -> two ledges carry it there;
- header spacers (plastic) X 0.05..2.45 and 17.85..20.25, Y 20.45..43.25, 4.3 mm under the PCB top;
  pins go 10.18 mm under the PCB top -> PCB bottom at 10.1 leaves 1.5 mm for the pin tips and a jumper;
- bare underside at (5.0, 38.5) and (15.3, 38.5) -> two pillars carry the ESP32 half;
- USB-C receptacle X 9.0..17.5, 0..3.2 above the PCB top; terminal blocks Y 14.5..20.5, 8.5 tall, wires enter
  from the USB side; snap-off line at Y ~13.7.
"""
import cadquery as cq

# ---------------------------------------------------------------- parameters
FLOOR = 1.6            # skin-face wall under the motors (TITAN guide: rigid plastic transmits well)
WALL = 2.0
PCB_BOTTOM = 10.1      # pin tips (8.58 under the PCB bottom) + 1.5 mm for a jumper
PCB_T = 1.6
PCB_TOP = PCB_BOTTOM + PCB_T
INNER_H = 25.5         # LFi (23 mm) standing on the floor + 2.5 mm air at its top end
LID = 1.6
LIP_H, LIP_T, LIP_GAP = 4.0, 1.2, 0.2
BOARD_CL = 0.3

MOTOR_D, MOTOR_L = 9.5, 23.0
CRADLE_D = 9.9          # 0.2 mm radial clearance; snap opening narrower than the motor
SADDLE_H = 7.0
LF = (29.0, 4.0)        # axis X, start Y (lying, axis along Y)
MF = (42.6, 4.0)
LFI = (35.8, 36.0)      # standing, axis vertical
MOTOR_Z = 5.15          # axis height for the lying motors (0.4 mm above the floor)

IX0, IX1 = -1.5, 50.5   # interior (0.5 mm extra at the board side for the small inner corner radius)
IY0, IY1 = -1.34 - BOARD_CL, 43.31 + BOARD_CL
R_OUT, R_IN = 6.0, 2.0   # small inner radius: the board's rectangular corners must fit

USB_X, USB_W, USB_H = 13.25, 14.0, 8.0     # opening for a 12.5 x 6.5 mm plug overmould
USB_Z = PCB_TOP + 1.6

STRAP_W, STRAP_SLOT = 27.0, 3.6            # 25 mm velcro
FLANGE_OUT, FLANGE_T = 9.0, 3.5

TONGUE_LEN, TONGUE_W = 30.0, 16.0          # cable strain relief
CABLE_D = 4.0


def rbox(x0, x1, y0, y1, z0, z1, r):
    return (cq.Workplane("XY").box(x1 - x0, y1 - y0, z1 - z0, centered=False)
            .translate((x0, y0, z0)).edges("|Z").fillet(r))


def base():
    ox0, ox1, oy0, oy1 = IX0 - WALL, IX1 + WALL, IY0 - WALL, IY1 + WALL
    b = rbox(ox0, ox1, oy0, oy1, -FLOOR, INNER_H, R_OUT)
    b = b.cut(rbox(IX0, IX1, IY0, IY1, 0, INNER_H + 1, R_IN))
    b = b.edges("<Z").fillet(2.5)                              # soft skin-face edge
    # lid snap groove on the inner wall (0.5 mm), 2 mm below the rim
    groove = rbox(IX0 - 0.5, IX1 + 0.5, IY0 - 0.5, IY1 + 0.5, INNER_H - 2.6, INNER_H - 1.6, R_IN + 0.5).cut(
        rbox(IX0, IX1, IY0, IY1, INNER_H - 3, INNER_H, R_IN))
    b = b.cut(groove)

    # board: ledges + side guides (USB half), pillars (ESP32 half), end stops
    for x0, x1, gx0, gx1 in ((-0.9, 1.5, -0.9, -0.4), (18.8, 21.2, 20.72, 21.2)):
        b = b.union(rbox(x0, x1, 0.5, 13.5, 0, PCB_BOTTOM, 0.3))
        b = b.union(rbox(gx0, gx1, 0.5, 13.5, PCB_BOTTOM, PCB_TOP + 1.3, 0.1))
    for x, y in ((5.0, 38.5), (15.3, 38.5)):
        b = b.union(cq.Workplane("XY").circle(1.5).extrude(PCB_BOTTOM).translate((x, y, 0)))
    b = b.union(rbox(1.0, 8.0, IY0, -BOARD_CL, 0, PCB_TOP - 0.1, 0.2))   # stop in front of the PCB edge (solid to the floor: no overhang)

    # lying motors: two snap saddles each, 2.5 mm in from each end
    for mx, my in (LF, MF):
        for s0 in (my + 2.5, my + 15.5):
            saddle = rbox(mx - 6.15, mx + 6.15, s0, s0 + 5.0, 0, SADDLE_H, 0.5)
            cut = cq.Workplane("XZ").center(mx, MOTOR_Z).circle(CRADLE_D / 2).extrude(-6).translate((0, s0 - 0.5, 0))
            b = b.union(saddle.cut(cut))
    # standing LFi: sleeve with a wire slot facing -X
    lx, ly = LFI
    sleeve = cq.Workplane("XY").circle(CRADLE_D / 2 + 1.6).circle(CRADLE_D / 2).extrude(12.0).translate((lx, ly, 0))
    sleeve = sleeve.cut(rbox(lx - 8, lx - 3, ly - 1.6, ly + 1.6, 0, 13, 0.1))
    b = b.union(sleeve)

    # raised labels on the floor, clear of every motor: in front of the lying ones, vertical beside the LFi sleeve
    for text, x, y, ang in (("LF > L", LF[0], 1.6, 0), ("MF > M", MF[0], 1.6, 0), ("LFi > R", 46.8, LFI[1], 90)):
        t = cq.Workplane("XY").text(text, 3.0, 0.6, halign="center", valign="center").rotate((0, 0, 0), (0, 0, 1), ang).translate((x, y, 0))
        b = b.union(t)

    # USB-C opening
    usb = rbox(USB_X - USB_W / 2, USB_X + USB_W / 2, IY0 - WALL - 1, IY0 + 0.1, USB_Z - USB_H / 2, USB_Z + USB_H / 2, 1.0)
    b = b.cut(usb)

    # strap flanges (vertical slots: strap goes up through one, over the top, down through the other)
    ymid = (IY0 + IY1) / 2
    for side in (-1, 1):
        wx = IX0 - WALL if side < 0 else IX1 + WALL
        fl = rbox(min(wx, wx + side * FLANGE_OUT), max(wx, wx + side * FLANGE_OUT), ymid - STRAP_W / 2 - 4, ymid + STRAP_W / 2 + 4,
                  -FLOOR, -FLOOR + FLANGE_T, 2.0)
        slot_x0 = wx + side * 1.0
        slot = rbox(min(slot_x0, slot_x0 + side * STRAP_SLOT), max(slot_x0, slot_x0 + side * STRAP_SLOT),
                    ymid - STRAP_W / 2, ymid + STRAP_W / 2, -FLOOR - 1, 10, 0.8)
        b = b.union(fl).cut(slot)

    # cable tongue + saddle with a zip-tie tunnel: a pull loads this, not the USB socket
    ty1 = IY0 - WALL
    ty0 = ty1 - TONGUE_LEN
    tongue = rbox(USB_X - TONGUE_W / 2, USB_X + TONGUE_W / 2, ty0, ty1 + 1, -FLOOR, -FLOOR + 2.0, 2.0)
    cable_bottom = USB_Z - CABLE_D / 2
    saddle = rbox(USB_X - 6, USB_X + 6, ty0, ty0 + 7, -FLOOR, cable_bottom, 1.5)
    groove = cq.Workplane("XZ").center(USB_X, cable_bottom + CABLE_D / 2 - 0.3).circle(CABLE_D / 2 + 0.2).extrude(-9).translate((0, ty0 - 1, 0))
    tunnel = rbox(USB_X - 8, USB_X + 8, ty0 + 2.0, ty0 + 5.2, cable_bottom - 3.8, cable_bottom - 2.0, 0.1)   # 3.2 x 1.8 for a 2.5 mm tie
    b = b.union(tongue).union(saddle.cut(groove)).cut(tunnel)
    return b


def lid():
    ox0, ox1, oy0, oy1 = IX0 - WALL, IX1 + WALL, IY0 - WALL, IY1 + WALL
    plate = rbox(ox0, ox1, oy0, oy1, INNER_H, INNER_H + LID, R_OUT)
    g = LIP_GAP
    lip = rbox(IX0 + g, IX1 - g, IY0 + g, IY1 - g, INNER_H - LIP_H, INNER_H, R_IN - g).cut(
        rbox(IX0 + g + LIP_T, IX1 - g - LIP_T, IY0 + g + LIP_T, IY1 - g - LIP_T, INNER_H - LIP_H - 1, INNER_H + 1, max(R_IN - g - LIP_T, 0.5)))
    # USB notch in the lip
    lip = lip.cut(rbox(USB_X - USB_W / 2 - 1, USB_X + USB_W / 2 + 1, IY0 - 1, IY0 + LIP_T + 1, INNER_H - LIP_H - 1, INNER_H, 0.1))
    l = plate.union(lip)
    # 0.45 mm snap bumps on the lip, matching the base groove
    zb = INNER_H - 2.1
    for x, y, dx, dy in (((IX0 + IX1) / 2, IY0 + g, 0, -1), ((IX0 + IX1) / 2, IY1 - g, 0, 1),
                         (IX0 + g, (IY0 + IY1) / 2 + 8, -1, 0), (IX1 - g, (IY0 + IY1) / 2 + 8, 1, 0)):
        bump = cq.Workplane("XY").box(10 if dx == 0 else 0.9, 0.9 if dx == 0 else 10, 0.8).translate((x + dx * 0.0, y + dy * 0.0, zb))
        l = l.union(bump)
    # hold-down post over the ESP32 module (top measured at 2.55 mm above the PCB), 0.35 mm gap
    l = l.union(rbox(8.0, 12.0, 30.0, 34.0, PCB_TOP + 2.9, INNER_H, 0.5))   # the module top measures 2.55 mm
    # vents over the lying motors
    for mx, my in (LF, MF):
        for k in (-1, 1):
            l = l.cut(rbox(mx + k * 2.6 - 0.9, mx + k * 2.6 + 0.9, my + 4, my + 19, INNER_H - 1, INNER_H + LID + 1, 0.8))
    # soft top edge
    l = l.faces(">Z").edges().fillet(1.5)
    return l


if __name__ == "__main__":
    import sys
    b, l = base(), lid()
    cq.exporters.export(b, "puck_base.step"); cq.exporters.export(l, "puck_lid.step")
    cq.exporters.export(b, "puck_base.stl", tolerance=0.02, angularTolerance=0.1)
    # lid prints upside down (outer face on the bed)
    cq.exporters.export(l.rotate((0, 0, 0), (1, 0, 0), 180), "puck_lid_print.stl", tolerance=0.02, angularTolerance=0.1)
    bb, lb = b.val().BoundingBox(), l.val().BoundingBox()
    print(f"base {bb.xlen:.1f} x {bb.ylen:.1f} x {bb.zlen:.1f} mm; lid {lb.xlen:.1f} x {lb.ylen:.1f} x {lb.zlen:.1f} mm")
    print(f"closed puck body {IX1 - IX0 + 2 * WALL:.1f} x {IY1 - IY0 + 2 * WALL:.1f} x {INNER_H + FLOOR + LID:.1f} mm (plus strap flanges and cable tongue)")
