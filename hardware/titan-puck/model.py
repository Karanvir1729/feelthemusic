"""TITAN haptic puck v2.1: feet-down DRAKE holders, sprung lid retention, thin haptic skin face.

Frame (mm): floor top z = 0 (the skin face is the underside, z = -FLOOR). Board coordinates are the vendor STEP's:
board X 0..20.32, Y 0..43.31, PCB top at z = 0 in the STEP, USB-C overhangs to Y = -1.34 on the top side.
Here the board is lifted so its PCB top sits at z = PCB_TOP.

Measured from the vendor STEP (height maps, 0.5 mm cells):
- underside is bare PCB along both long edges for Y 0..13.5 (USB half) -> two ledges carry it there;
- header spacers (plastic) X 0.05..2.45 and 17.85..20.25, Y 20.45..43.25, 4.3 mm under the PCB top;
  pins go 10.18 mm under the PCB top -> PCB bottom at 10.1 leaves 1.5 mm for the pin tips and a jumper;
- bare underside at (5.0, 38.5) and (15.3, 38.5) -> two pillars carry the ESP32 half;
- USB-C receptacle X 9.0..17.5, 0..3.2 above the PCB top; terminal blocks Y 14.5..20.5, 8.5 tall, wires enter
  from the USB side (-Y) 2.3 mm above the PCB top; the PCB is flat (no parts) over Y 8..13.5.

The DRAKE motor (FIT.md, measured 2026-09-19 from the host's photo with a perspective camera model):
end caps 9.8 +- 0.2 mm, centre band about cap / 1.054, length 23.0-23.2 mm, and two flat mounting feet on one side,
at the inner ends of the caps, about 5.7-8.8 mm from each end, protruding about 1.03 mm (a lower bound) past the cap
outline. Foot width and any notch are not measured. TITAN mounts the motor feet-down on a flat plate; so do we.

Haptics first: the puck exists to carry the motors' vibration into the skin. The skin face under the motors is a
1.0 mm floor with the motor feet standing directly on it; walls and lid are 1.2 mm; only the strap flanges, cable
tongue, board supports, snap rim and spring roots are thicker. A lighter shell moves more for the same motor force.
"""

import math
import os

import cadquery as cq


def _env(name, default):
    value = float(os.environ.get(name, default))
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


# ---------------------------------------------------------------- shell (haptic-thin)
FLOOR = 1.0  # skin face: 5 layers at 0.20 mm; the motor feet stand directly on it
WALL = 1.2  # side walls: 3 lines at 0.4 mm
RIM_WALL = 2.0  # top band only: carries the lid snap groove (leaves 1.6 mm)
RIM_H = 4.4  # height of the thick top band (its lower 0.8 mm is a 45-degree chamfer, no overhang)
LID = 1.2  # lid plate; the spring tongues are cut from it
PCB_BOTTOM = 10.1  # pin tips (8.58 under the PCB bottom) + 1.5 mm for a jumper
PCB_T = 1.6
PCB_TOP = PCB_BOTTOM + PCB_T
INNER_H = 25.5  # longest LFi envelope (23.5 mm) + 2 mm air at its top end
LIP_H, LIP_T, LIP_GAP = 4.0, 1.6, 0.2
GROOVE_DEPTH = 0.4  # leaves 1.6 mm of the 2.0 mm rim
BOARD_CL = 0.3
SKIN_FILLET = 2.5  # rounded skin-face edge
INNER_COVE = 1.5  # inside fillet at the floor/wall joint keeps the rounded corner >= 1.1 mm thick

# ---------------------------------------------------------------- motor (TITAN DRAKE), measured
CAP_D_NOM = 9.8  # measured end cap, +-0.2 (yellow 9.83, plain 9.70, red >= 10.0)
CAP_D_MIN, CAP_D_MAX = 9.5, 10.2  # range every holder is designed for
BAND_RATIO = 1.054  # cap / band, measured on the yellow motor (band 9.33 mm)
MOTOR_L_NOM = _env("MOTOR_L_NOM", 23.1)  # measured 23.0-23.2 (datasheet 23)
MOTOR_L = 23.5  # design envelope for end air and the lid height
FOOT_PROTRUSION = _env("FOOT_PROTRUSION", 1.03)  # past the cap outline; a lower bound from the photo
FOOT_PROT_MIN, FOOT_PROT_MAX = 0.8, 1.5
FOOT_START = _env("FOOT_START", 5.7)  # end of the motor to the foot's outer face (both ends, mirrored)
FOOT_START_TOL = 0.5
FOOT_LEN = _env("FOOT_LEN", 3.1)  # along the axis
FOOT_W = _env("FOOT_W", 6.0)  # across the motor: NOT measured (render suggests narrower than the cap)
FOOT_W_NARROW, FOOT_W_WIDE = 3.0, 8.0

# One holder size for every motor: the locator width takes the largest cap with clearance.
MOTOR_D = _env("MOTOR_D", CAP_D_MAX)  # largest end cap the holders accept
if not 9.5 <= MOTOR_D <= 11.4:
    raise ValueError("MOTOR_D must be the largest accepted end cap, from 9.5 to 11.4 mm")
CAP_CLEARANCE = 0.2  # per side at MOTOR_D; smaller caps get more lateral play (the lid springs clamp them)
LOCATOR_W = MOTOR_D + 2 * CAP_CLEARANCE
if not (0 < FOOT_LEN and 0 < FOOT_START and 2 * (FOOT_START + FOOT_LEN) < MOTOR_L_NOM - 1.5):
    raise ValueError("feet must lie inside the motor length and leave a gap between them")
if not 0.3 <= FOOT_PROTRUSION <= 3.0:
    raise ValueError("FOOT_PROTRUSION must be 0.3-3.0 mm")

# ---------------------------------------------------------------- lying holders (LF, MF), feet down
FOOT_GAP_NOM = MOTOR_L_NOM - 2 * (FOOT_START + FOOT_LEN)  # between the two feet's inner faces (5.5 mm)
RIB_PLAY = 0.3  # per end, rib to foot, at the nominal foot position
RIB_L = FOOT_GAP_NOM - 2 * RIB_PLAY  # the feet straddle it: the motor cannot walk along its axis
RIB_W = 4.0  # centred and wider than any plausible central foot notch; a narrow foot still meets it
RIB_H = 0.6  # 3 layers; under the lowest band (foot 0.8 + (9.5 - 9.01) / 2 = 1.04 mm) with 0.44 mm to spare
CAP_WALL_T = 1.2
CAP_WALL_H = 6.5  # above the equator of the smallest motor on its feet (5.55 mm)
CAP_WALL_IN = FOOT_START - FOOT_START_TOL - 0.4  # walls stop short of the feet, even if they sit 0.5 mm further out
CAP_WALL_OUT = -0.1  # and run past the end of the cap
MOTOR_SEPARATION = 3.3  # between the caps of LF and MF (brief: >= 3)
NEAR_MOTOR_X = 29.0  # MF, next to the board (M is the far-left terminal: shortest MF lead)
FAR_MOTOR_X = NEAR_MOTOR_X + MOTOR_D + MOTOR_SEPARATION  # LF, against the outer wall
LABEL_H, LABEL_STROKE, LABEL_Z = 3.6, 0.85, 0.6  # raised floor labels: strokes >= 0.8 mm slice with any wall setting
MOTOR_FRONT = 3.9  # front end of the lying motors (labels in front of them, >= 2 mm air at both ends)
MOTOR_Y_C = MOTOR_FRONT + MOTOR_L_NOM / 2  # centre of the feet gap = centre of the rib
LF = (FAR_MOTOR_X, MOTOR_Y_C)
MF = (NEAR_MOTOR_X, MOTOR_Y_C)

# ---------------------------------------------------------------- lid springs over the lying motors
# Motor top on its feet ranges 0.8 + 9.5 = 10.3 to 1.5 + 10.2 = 11.7 mm: a rigid post cannot hold all of them without
# a gap that lets the feet ride over the rib. A post on a tongue cut from the lid plate presses every motor down.
PRELOAD_MIN = 0.3  # deflection at the lowest motor top
SPRING_FREE_Z = FOOT_PROT_MIN + CAP_D_MIN - PRELOAD_MIN  # post bottom, lid closed, tongue unloaded (10.0)
SPRING_MAX_DEFLECTION = 2.2  # design limit (strain below)
TONGUE_L = 16.0  # root to post centre; tongues root over the board (-X)
TONGUE_W = 3.5
SLIT = 1.0
POST_W = 3.0
POST_DY = {"MF": 9.0, "LF": 4.5}  # staggered so the two motors' tongues interleave; both on the cap tops
SPRINGS = {
    f"{name}{end}": (mx, my + sign * POST_DY[name])
    for name, (mx, my) in (("MF", MF), ("LF", LF))
    for end, sign in (("-front", -1), ("-rear", 1))
}


def tongue_strain(deflection, length=TONGUE_L, thickness=LID):
    """Peak bending strain of a cantilever tongue deflected at its post."""
    return 1.5 * thickness * deflection / length**2


# Board hold-down: a short spring over the ESP32 module (module top 2.55 mm above the PCB top).
BOARD_POST = (10.0, 32.0)
BOARD_POST_W = 4.0
BOARD_TONGUE_ROOT = 43.31 + 0.3 - 0.2 - 1.6  # rooted toward +Y where the lid lip stiffens the plate (IY1 - gap - lip)
BOARD_PRELOAD = 0.3
BOARD_FREE_Z = PCB_TOP + 2.55 - BOARD_PRELOAD

# ---------------------------------------------------------------- standing LFi
LFI = ((NEAR_MOTOR_X + FAR_MOTOR_X) / 2, 36.5)
LFI_BORE = LOCATOR_W
LFI_WALL = 1.2
LFI_SLEEVE_H = 4.6  # below the lower foot even if it starts 0.5 mm nearer the end (5.2 - 0.6)
LFI_RING_Z0 = 18.6  # lid ring bottom: above the upper foot of a 23.5 mm motor with feet 0.5 mm further out (18.3)
LFI_RING_CHAMFER = 1.0  # lead-in at the ring mouth

# ---------------------------------------------------------------- enclosure outline
IX0 = -1.5
IX1 = FAR_MOTOR_X + LOCATOR_W / 2  # the outer wall is LF's outboard cap locator
IY0, IY1 = -1.34 - BOARD_CL, 43.31 + BOARD_CL
R_OUT, R_IN = 6.0, 2.0  # small inner radius: the board's rectangular corners must fit

USB_X, USB_W, USB_H = 13.25, 14.0, 8.0  # opening for a 12.5 x 6.5 mm plug overmould
USB_Z = PCB_TOP + 1.6
USB_BRIDGE = 10.0
USB_SHOULDER = (USB_W - USB_BRIDGE) / 2

STRAP_W, STRAP_SLOT = 27.0, 3.6  # 25 mm velcro
FLANGE_OUT, FLANGE_T = 9.0, 3.5
SLOT_LIGAMENT = 1.6

TONGUE_LEN, TONGUE_WIDTH = 30.0, 16.0  # cable strain relief
CABLE_D = 4.0

# Open comb on the board's +X edge: one lane per motor pair, all in front of the terminal blocks (Y < 13.6) where the
# PCB is flat, so each pair runs west and turns north into its terminal. Front to back: MF -> M, LF -> L, LFi -> R.
WIRE_GUIDE_X0, WIRE_GUIDE_X1 = 20.72, 22.32
WIRE_SLOT_Y = {"MF": 8.2, "LF": 10.4, "LFi": 12.6}
WIRE_SLOT_W = 1.4  # a pair of approximately 0.8 mm leads stacked in the lane
WIRE_GUIDE_Y0 = min(WIRE_SLOT_Y.values()) - WIRE_SLOT_W / 2 - 0.9
WIRE_GUIDE_Y1 = max(WIRE_SLOT_Y.values()) + WIRE_SLOT_W / 2 + 0.6
WIRE_SLOT_Z = PCB_TOP + 1.9  # lane floor
WIRE_GUIDE_Z = PCB_TOP + 4.0


def rbox(x0, x1, y0, y1, z0, z1, r):
    b = cq.Workplane("XY").box(x1 - x0, y1 - y0, z1 - z0, centered=False).translate((x0, y0, z0))
    return b.edges("|Z").fillet(r) if r > 0 else b


# ---------------------------------------------------------------- stroke font (labels with a known stroke width)
_GLYPHS = {
    "L": [[(0, 1), (0, 0), (0.6, 0)]],
    "F": [[(0.62, 1), (0, 1), (0, 0)], [(0, 0.52), (0.46, 0.52)]],
    "M": [[(0, 0), (0, 1), (0.42, 0.42), (0.84, 1), (0.84, 0)]],
    "i": [[(0, 0), (0, 0.5)], [(0, 0.94), (0, 0.95)]],
    "R": [[(0, 0), (0, 1), (0.42, 1), (0.62, 0.86), (0.62, 0.66), (0.42, 0.52), (0, 0.52)], [(0.3, 0.52), (0.64, 0)]],
    "G": [[(0.66, 0.84), (0.5, 1), (0.16, 1), (0, 0.82), (0, 0.18), (0.16, 0), (0.5, 0), (0.66, 0.16),
           (0.66, 0.44), (0.36, 0.44)]],
    "O": [[(0.16, 0), (0.5, 0), (0.66, 0.18), (0.66, 0.82), (0.5, 1), (0.16, 1), (0, 0.82), (0, 0.18), (0.16, 0)]],
    "N": [[(0, 0), (0, 1), (0.66, 0), (0.66, 1)]],
    "T": [[(0, 1), (0.7, 1)], [(0.35, 1), (0.35, 0)]],
    "P": [[(0, 0), (0, 1), (0.44, 1), (0.64, 0.86), (0.64, 0.64), (0.44, 0.5), (0, 0.5)]],
    "U": [[(0, 1), (0, 0.18), (0.16, 0), (0.5, 0), (0.66, 0.18), (0.66, 1)]],
    "S": [[(0.64, 0.86), (0.48, 1), (0.16, 1), (0, 0.84), (0, 0.66), (0.16, 0.52), (0.48, 0.48), (0.64, 0.34),
           (0.64, 0.16), (0.48, 0), (0.16, 0), (0, 0.14)]],
    "1": [[(0.1, 0.8), (0.35, 1), (0.35, 0)]],
    "2": [[(0, 0.84), (0.16, 1), (0.48, 1), (0.64, 0.84), (0.64, 0.62), (0, 0), (0.66, 0)]],
    "3": [[(0, 0.86), (0.16, 1), (0.48, 1), (0.64, 0.84), (0.64, 0.66), (0.46, 0.52), (0.2, 0.52)],
          [(0.46, 0.52), (0.66, 0.36), (0.66, 0.16), (0.5, 0), (0.16, 0), (0, 0.14)]],
    "-": [[(0.05, 0.5), (0.45, 0.5)]],
    "+": [[(0, 0.5), (0.56, 0.5)], [(0.28, 0.22), (0.28, 0.78)]],
    ">": [[(0, 0.9), (0.55, 0.5), (0, 0.1)]],
}


def stroke_text(text, height, stroke, depth, spacing=0.34):
    """Raised text made of round-ended strokes of a known width, centred on the origin, bottom at z = 0."""
    s = height - stroke  # glyph box inside the stroke's own half-width margin
    pieces, x = [], 0.0
    for ch in text:
        if ch == " ":
            x += 0.5 * s
            continue
        strokes = _GLYPHS[ch]
        width = max(px for line in strokes for px, _ in line)
        for line in strokes:
            pts = [(x + px * s, py * s) for px, py in line]
            for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
                length = math.hypot(x1 - x0, y1 - y0)
                seg = cq.Workplane("XY").box(max(length, 1e-3), stroke, depth, centered=(True, True, False))
                seg = seg.rotate((0, 0, 0), (0, 0, 1), math.degrees(math.atan2(y1 - y0, x1 - x0)))
                pieces.append(seg.translate(((x0 + x1) / 2, (y0 + y1) / 2, 0)))
            for px, py in pts:
                pieces.append(cq.Workplane("XY").circle(stroke / 2).extrude(depth).translate((px, py, 0)))
        x += width * s + spacing * s + stroke
    total = x - spacing * s - stroke
    out = pieces[0]
    for piece in pieces[1:]:
        out = out.union(piece)
    return out.translate((-total / 2, -s / 2, 0))


def text_width(text, height=LABEL_H, stroke=LABEL_STROKE, spacing=0.34):
    s = height - stroke
    x = 0.0
    for ch in text:
        if ch == " ":
            x += 0.5 * s
            continue
        x += max(px for line in _GLYPHS[ch] for px, _ in line) * s + spacing * s + stroke
    return x - spacing * s - stroke + stroke


# ---------------------------------------------------------------- motor holders
def lying_holder(mx, my, outboard_wall=True):
    """Feet-down holder: a low rib between the feet and side walls beside the outer part of each end cap."""
    parts = [rib(mx, my)]
    half = MOTOR_L_NOM / 2
    for sign in (-1, 1):
        y_a, y_b = my + sign * (half - CAP_WALL_IN), my + sign * (half - CAP_WALL_OUT)
        y0, y1 = min(y_a, y_b), max(y_a, y_b)
        for side in (-1, 1):
            if side > 0 and not outboard_wall:
                continue
            x_in = mx + side * LOCATOR_W / 2
            x_out = x_in + side * CAP_WALL_T
            parts.append(rbox(min(x_in, x_out), max(x_in, x_out), y0, y1, 0, CAP_WALL_H, 0.3))
    out = parts[0]
    for part in parts[1:]:
        out = out.union(part)
    return out


def rib(mx, my):
    return rbox(mx - RIB_W / 2, mx + RIB_W / 2, my - RIB_L / 2, my + RIB_L / 2, 0, RIB_H, 0.3)


def ribs():
    """The axial-capture ribs alone (for the checker)."""
    return rib(*MF).union(rib(*LF))


def lfi_sleeve():
    lx, ly = LFI
    return (
        cq.Workplane("XY")
        .circle(LFI_BORE / 2 + LFI_WALL)
        .circle(LFI_BORE / 2)
        .extrude(LFI_SLEEVE_H)
        .translate((lx, ly, 0))
    )


def lfi_ring(z_top=INNER_H):
    """Ring hanging from the lid around the LFi's top cap, with a lead-in chamfer at its mouth."""
    lx, ly = LFI
    r_in, r_out = LFI_BORE / 2, LFI_BORE / 2 + LFI_WALL
    h = z_top - LFI_RING_Z0
    c = LFI_RING_CHAMFER * 0.8
    ring = cq.Workplane("XY").circle(r_out).circle(r_in).extrude(h)
    # Mouth chamfer: the wall thins toward the open end (no overhang when the lid prints upside down).
    cone = cq.Solid.makeCone(r_in + c, r_in, c, cq.Vector(0, 0, -1e-3), cq.Vector(0, 0, 1))
    ring = ring.cut(cq.Workplane("XY").add(cone))
    return ring.translate((lx, ly, LFI_RING_Z0))


# ---------------------------------------------------------------- base
def shell():
    rim = (IX0 - RIM_WALL, IX1 + RIM_WALL, IY0 - RIM_WALL, IY1 + RIM_WALL)
    step = RIM_WALL - WALL
    z_rim = INNER_H - RIM_H
    lower = rbox(IX0 - WALL, IX1 + WALL, IY0 - WALL, IY1 + WALL, -FLOOR, z_rim + step, R_OUT - step)
    band = rbox(*rim, z_rim, INNER_H, R_OUT).faces("<Z").edges().chamfer(step * 0.999)
    body = lower.union(band)
    cavity = rbox(IX0, IX1, IY0, IY1, 0, INNER_H + 1, R_IN).faces("<Z").edges().fillet(INNER_COVE)
    body = body.cut(cavity)
    return body.edges("<Z").fillet(SKIN_FILLET)


def base():
    b = shell()
    # Lid snap groove in the thick rim.
    groove = rbox(
        IX0 - GROOVE_DEPTH, IX1 + GROOVE_DEPTH, IY0 - GROOVE_DEPTH, IY1 + GROOVE_DEPTH,
        INNER_H - 2.6, INNER_H - 1.6, R_IN + GROOVE_DEPTH,
    ).cut(rbox(IX0, IX1, IY0, IY1, INNER_H - 3, INNER_H, R_IN))
    b = b.cut(groove)

    # board: ledges + side guides (USB half), pillars (ESP32 half), end stop
    for x0, x1, gx0, gx1 in ((-2.0, 1.5, -2.0, -0.4), (18.8, 22.32, 20.72, 22.32)):
        b = b.union(rbox(x0, x1, 0.5, 13.5, 0, PCB_BOTTOM, 0.3))
        b = b.union(rbox(gx0, gx1, 0.5, 13.5, PCB_BOTTOM, PCB_TOP + 1.3, 0.1))
    for x, y in ((5.0, 38.5), (15.3, 38.5)):
        b = b.union(cq.Workplane("XY").circle(1.5).extrude(PCB_BOTTOM).translate((x, y, 0)))
    b = b.union(rbox(1.0, 8.0, IY0, -BOARD_CL, 0, PCB_TOP - 0.1, 0.2))  # stop in front of the PCB edge

    # feet-down motor holders; LF's outboard side is the enclosure wall itself
    b = b.union(lying_holder(*MF)).union(lying_holder(*LF, outboard_wall=False))
    b = b.union(lfi_sleeve())

    # raised labels (stroke >= 0.8 mm), in front of the lying motors and beside the LFi
    label_y = MOTOR_FRONT - 0.3 - LABEL_H / 2
    for text, (x, _) in (("MF", MF), ("LF", LF)):
        b = b.union(stroke_text(text, LABEL_H, LABEL_STROKE, LABEL_Z).translate((x, label_y, 0)))
    lfi_label = stroke_text("LFi", LABEL_H, LABEL_STROKE, LABEL_Z).rotate((0, 0, 0), (0, 0, 1), 90)
    b = b.union(lfi_label.translate((LFI[0] + LFI_BORE / 2 + LFI_WALL + 0.4 + LABEL_H / 2, LFI[1], 0)))

    # open comb on the board's +X edge
    guide = rbox(WIRE_GUIDE_X0, WIRE_GUIDE_X1, WIRE_GUIDE_Y0, WIRE_GUIDE_Y1, 0, WIRE_GUIDE_Z, 0.3)
    for wire_y in WIRE_SLOT_Y.values():
        lane = rbox(
            WIRE_GUIDE_X0 - 0.5, WIRE_GUIDE_X1 + 0.5,
            wire_y - WIRE_SLOT_W / 2, wire_y + WIRE_SLOT_W / 2,
            WIRE_SLOT_Z, WIRE_GUIDE_Z + 1, 0,
        )
        guide = guide.cut(lane)
    b = b.union(guide)

    # USB-C opening; 45-degree roof shoulders shorten the bridge to USB_BRIDGE
    ux0, ux1 = USB_X - USB_W / 2, USB_X + USB_W / 2
    uz0, uz1 = USB_Z - USB_H / 2, USB_Z + USB_H / 2
    usb = (
        cq.Workplane("XZ")
        .polyline([(ux0, uz0), (ux1, uz0), (ux1, uz1), (ux1 - USB_SHOULDER, uz1 + USB_SHOULDER),
                   (ux0 + USB_SHOULDER, uz1 + USB_SHOULDER), (ux0, uz1)])
        .close()
        .extrude(RIM_WALL + 1.1)
        .translate((0, IY0 + 0.1, 0))
    )
    b = b.cut(usb)

    # strap flanges (vertical slots: strap goes up through one, over the top, down through the other)
    ymid = (IY0 + IY1) / 2
    for side in (-1, 1):
        wx = IX0 - WALL if side < 0 else IX1 + WALL
        fl = rbox(
            min(wx, wx + side * FLANGE_OUT), max(wx, wx + side * FLANGE_OUT),
            ymid - STRAP_W / 2 - 4, ymid + STRAP_W / 2 + 4, -FLOOR, -FLOOR + FLANGE_T, 2.0,
        )
        slot_x0 = wx + side * SLOT_LIGAMENT
        slot = rbox(
            min(slot_x0, slot_x0 + side * STRAP_SLOT), max(slot_x0, slot_x0 + side * STRAP_SLOT),
            ymid - STRAP_W / 2, ymid + STRAP_W / 2, -FLOOR - 1, 10, 0.8,
        )
        b = b.union(fl).cut(slot)

    # cable tongue + saddle with a zip-tie tunnel: a pull loads this, not the USB socket
    ty1 = IY0 - WALL
    ty0 = ty1 - TONGUE_LEN
    tongue = rbox(USB_X - TONGUE_WIDTH / 2, USB_X + TONGUE_WIDTH / 2, ty0, ty1 + 1, -FLOOR, -FLOOR + 2.0, 2.0)
    cable_bottom = USB_Z - CABLE_D / 2
    saddle = rbox(USB_X - 6, USB_X + 6, ty0, ty0 + 7, -FLOOR, cable_bottom, 1.5)
    groove = (
        cq.Workplane("XZ")
        .center(USB_X, cable_bottom + CABLE_D / 2 - 0.3)
        .circle(CABLE_D / 2 + 0.2)
        .extrude(-9)
        .translate((0, ty0 - 1, 0))
    )
    tunnel = rbox(USB_X - 8, USB_X + 8, ty0 + 2.0, ty0 + 5.2, cable_bottom - 3.8, cable_bottom - 2.0, 0)
    b = b.union(tongue).union(saddle.cut(groove)).cut(tunnel)
    return b


# ---------------------------------------------------------------- lid
def _tongue_cut(root, tip, band0, band1, along):
    """Slit (through the lid plate only) around a tongue spanning root..tip along X or Y, band0..band1 across."""
    z0, z1 = INNER_H - 0.01, INNER_H + LID + 1
    d = 1 if tip > root else -1
    a0, a1 = sorted((root, tip + d * SLIT))
    t0, t1 = sorted((root - d * 1.0, tip))
    if along == "x":
        outer = rbox(a0, a1, band0 - SLIT, band1 + SLIT, z0, z1, 0)
        inner = rbox(t0, t1, band0, band1, z0 - 1, z1 + 1, 0)
    else:
        outer = rbox(band0 - SLIT, band1 + SLIT, a0, a1, z0, z1, 0)
        inner = rbox(band0, band1, t0, t1, z0 - 1, z1 + 1, 0)
    return outer.cut(inner)


def spring_post(name, lift=0.0):
    """A lying-motor retainer post, hanging from its tongue tip; lift = tongue deflection when installed."""
    px, py = SPRINGS[name]
    return rbox(px - POST_W / 2, px + POST_W / 2, py - POST_W / 2, py + POST_W / 2,
                SPRING_FREE_Z + lift, INNER_H + 0.01 + lift, 0.4)


def board_post(lift=0.0):
    bx, by = BOARD_POST
    h = BOARD_POST_W / 2
    return rbox(bx - h, bx + h, by - h, by + h, BOARD_FREE_Z + lift, INNER_H + 0.01 + lift, 0.4)


def lid(lifts=None):
    """The lid, assembly orientation. lifts: {spring name or 'board': installed deflection} for fit checks;
    None gives the as-printed (unloaded) lid."""
    lifts = lifts or {}
    ox0, ox1, oy0, oy1 = IX0 - RIM_WALL, IX1 + RIM_WALL, IY0 - RIM_WALL, IY1 + RIM_WALL
    plate = rbox(ox0, ox1, oy0, oy1, INNER_H, INNER_H + LID, R_OUT)
    plate = plate.faces(">Z").edges().fillet(1.0)
    g = LIP_GAP
    lip = rbox(IX0 + g, IX1 - g, IY0 + g, IY1 - g, INNER_H - LIP_H, INNER_H, R_IN - g).cut(
        rbox(IX0 + g + LIP_T, IX1 - g - LIP_T, IY0 + g + LIP_T, IY1 - g - LIP_T,
             INNER_H - LIP_H - 1, INNER_H + 1, max(R_IN - g - LIP_T, 0.5))
    )
    lip = lip.cut(rbox(USB_X - USB_W / 2 - 1, USB_X + USB_W / 2 + 1, IY0 - 1, IY0 + LIP_T + 1,
                       INNER_H - LIP_H - 1, INNER_H, 0))  # USB notch
    lid_part = plate.union(lip)
    # snap bumps on the lip, matching the base groove; the -Y bump stays clear of the USB notch
    zb = INNER_H - 2.1
    usb_notch_x1 = USB_X + USB_W / 2 + 1
    for x, y, dx in (
        ((usb_notch_x1 + IX1) / 2, IY0 + g, 0),
        ((IX0 + IX1) / 2, IY1 - g, 0),
        (IX0 + g, (IY0 + IY1) / 2 + 8, 1),
        (IX1 - g, (IY0 + IY1) / 2 + 8, 1),
    ):
        bump = cq.Workplane("XY").box(10 if dx == 0 else 0.9, 0.9 if dx == 0 else 10, 0.8).translate((x, y, zb))
        lid_part = lid_part.union(bump)
    # spring tongues: slit the plate around each, rooted toward -X (over the board)
    for name, (px, py) in SPRINGS.items():
        tip = px + POST_W / 2 + 0.5
        lid_part = lid_part.cut(_tongue_cut(px - TONGUE_L, tip, py - TONGUE_W / 2, py + TONGUE_W / 2, "x"))
    bx, by = BOARD_POST
    lid_part = lid_part.cut(_tongue_cut(BOARD_TONGUE_ROOT, by - BOARD_POST_W / 2 - 0.5,
                                        bx - BOARD_POST_W / 2, bx + BOARD_POST_W / 2, "y"))
    for name in SPRINGS:
        lid_part = lid_part.union(spring_post(name, lifts.get(name, 0.0)))
    lid_part = lid_part.union(board_post(lifts.get("board", 0.0)))
    lid_part = lid_part.union(lfi_ring())
    return lid_part


def lid_print():
    """The lid in print orientation: outer face on the bed."""
    return lid().rotate((0, 0, 0), (1, 0, 0), 180)


if __name__ == "__main__":
    base_part, lid_part = base(), lid()
    cq.exporters.export(base_part, "puck_base.step")
    cq.exporters.export(lid_part, "puck_lid.step")
    cq.exporters.export(base_part, "puck_base.stl", tolerance=0.02, angularTolerance=0.1)
    cq.exporters.export(lid_part.rotate((0, 0, 0), (1, 0, 0), 180), "puck_lid_print.stl",
                        tolerance=0.02, angularTolerance=0.1)
    bb, lb = base_part.val().BoundingBox(), lid_part.val().BoundingBox()
    print(f"base {bb.xlen:.1f} x {bb.ylen:.1f} x {bb.zlen:.1f} mm; lid {lb.xlen:.1f} x {lb.ylen:.1f} x {lb.zlen:.1f} mm")
    print(f"closed puck body {IX1 - IX0 + 2 * RIM_WALL:.1f} x {IY1 - IY0 + 2 * RIM_WALL:.1f} x "
          f"{INNER_H + FLOOR + LID:.1f} mm (plus strap flanges and cable tongue)")
    print(f"holders: locator {LOCATOR_W:.2f} mm for caps {CAP_D_MIN}-{MOTOR_D} mm; rib {RIB_L:.2f} x {RIB_W} x {RIB_H} "
          f"in a {FOOT_GAP_NOM:.2f} mm foot gap; spring posts free at z {SPRING_FREE_Z:.2f}")
    print(f"volumes: base {base_part.val().Volume() / 1000:.2f} cm3, lid {lid_part.val().Volume() / 1000:.2f} cm3")
