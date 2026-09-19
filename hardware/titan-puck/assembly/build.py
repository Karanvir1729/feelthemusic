"""Assembly diagram for the TITAN haptic puck draft (the one printed on 2026-09-19, commit 30e2ae1).

Renders step01.png .. step09.png, overview_exploded.png and sheet.png from the real geometry: the base and lid come
from model.py, the TITAN Core is drawn from titan_core_keepout.json (never the vendor mesh), and the motors are drawn
the way draft_check.py models them (end caps, centre band with the lead ring, two feet).

Run from hardware/titan-puck:
  uv run --quiet --python 3.12 --with-requirements requirements.txt --with pillow python assembly/build.py [names]
"""

import json
import math
import os
import shutil
import sys
import tempfile

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
PUCK = os.path.dirname(HERE)
sys.path.insert(0, PUCK)
import model as M  # noqa: E402

TMP = tempfile.mkdtemp(prefix="puck_asm_", dir=os.environ.get("ASM_TMP") or None)
W = 1400

# ---------------------------------------------------------------- colours (3D, 0..1) and overlay (0..255)
COL = {
    "base": (0.97, 0.56, 0.18, 1),
    "base_cut": (0.72, 0.33, 0.05, 1),
    "lid": (0.90, 0.93, 0.98, 0.42),
    "lid_solid": (0.90, 0.92, 0.96, 1),
    "lid_cut": (0.55, 0.60, 0.70, 1),
    "pcb": (0.10, 0.46, 0.22, 1),
    "term": (0.36, 0.76, 0.42, 1),
    "silver": (0.80, 0.81, 0.83, 1),
    "can": (0.70, 0.72, 0.74, 1),
    "module": (0.20, 0.24, 0.28, 1),
    "comp": (0.22, 0.22, 0.24, 1),
    "jst": (0.95, 0.94, 0.88, 1),
    "pins": (0.12, 0.12, 0.13, 1),
    "pin_metal": (0.78, 0.66, 0.28, 1),
    "hole": (0.05, 0.05, 0.05, 1),
    "LF": (0.18, 0.44, 0.95, 1),
    "MF": (0.58, 0.30, 0.86, 1),
    "LFi": (0.90, 0.18, 0.20, 1),
    "ring": (0.07, 0.07, 0.07, 1),
    "red": (0.88, 0.06, 0.06, 1),
    "black": (0.06, 0.06, 0.06, 1),
    "jumper": (0.10, 0.10, 0.12, 1),
    "cable": (0.16, 0.16, 0.18, 1),
    "plug": (0.25, 0.25, 0.28, 1),
    "tie": (0.62, 0.63, 0.66, 1),
    "strap": (0.24, 0.30, 0.38, 1),
    "strap_hook": (0.34, 0.40, 0.50, 1),
}
INK = (30, 30, 36)
RGB = {
    "LF": (36, 100, 225),
    "MF": (128, 60, 190),
    "LFi": (210, 30, 40),
    "board": (20, 120, 60),
    "lid": (70, 90, 130),
    "base": (215, 110, 20),
    "act": (0, 150, 136),
    "red": (215, 20, 20),
    "warn": (225, 30, 30),
}

FONT_DIR = "/System/Library/Fonts/Supplemental/"


def font(size, bold=True):
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    try:
        return ImageFont.truetype(FONT_DIR + name, size)
    except OSError:
        return ImageFont.load_default(size)


# ---------------------------------------------------------------- geometry constants
CAP_D = M.CAP_D_NOM
CAP_R = CAP_D / 2
BAND_R = CAP_D / M.BAND_RATIO / 2
L_MOT = M.MOTOR_L_NOM
PROT = M.FOOT_PROTRUSION
CAP_LEN = M.FOOT_START + M.FOOT_LEN  # the feet sit at the inner end of each cap
ZC_LYING = PROT + CAP_R  # motor axis height when it stands on its feet
RING_TOP = ZC_LYING + CAP_R - 0.15
HOLE_X = {"M+": 1.75, "M-": 4.35, "L+": 8.85, "L-": 11.45, "R+": 16.05, "R-": 18.55}
HOLE_REL_Z = 2.3  # above the PCB top
HOLE_FACE_Y = 13.58
SCREW_Y = 17.1
PIN_ROWS_X = (1.27, 19.05)
LIFT_SPRING = PROT + CAP_D - M.SPRING_FREE_Z  # installed deflection of every spring post
LIFT_BOARD = M.BOARD_PRELOAD

KO = json.load(open(os.path.join(PUCK, "titan_core_keepout.json")))


# ---------------------------------------------------------------- parts
def rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    x, y, z = axis
    return np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])


class Part:
    def __init__(self, group):
        self.group = group
        self.items = []

    def box(self, x0, x1, y0, y1, z0, z1, col):
        c = np.array([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
        h = np.array([abs(x1 - x0) / 2, abs(y1 - y0) / 2, abs(z1 - z0) / 2])
        self.items.append(("box", c, h, col))
        return self

    def cyl(self, p0, p1, r, col):
        self.items.append(("cyl", np.array(p0, float), np.array(p1, float), r, col))
        return self

    def cap(self, p0, p1, r, col):
        self.items.append(("cap", np.array(p0, float), np.array(p1, float), r, col))
        return self

    def sphere(self, p, r, col):
        self.items.append(("sph", np.array(p, float), r, col))
        return self

    def tube(self, pts, r, col):
        pts = [np.array(p, float) for p in pts]
        for a, b in zip(pts, pts[1:]):
            if np.linalg.norm(b - a) > 1e-3:
                self.cap(a, b, r, col)
        return self

    def mesh(self, key, col):
        self.items.append(("mesh", key, col))
        return self

    def plank(self, a, b, width, thick, col, across=(0, 1, 0)):
        a, b = np.array(a, float), np.array(b, float)
        ex = b - a
        ln = np.linalg.norm(ex)
        ex /= ln
        ey = np.array(across, float)
        ey = ey - ex * (ey @ ex)
        ey /= np.linalg.norm(ey)
        ez = np.cross(ex, ey)
        self.items.append(("obox", (a + b) / 2, np.array([ln / 2, width / 2, thick / 2]), np.column_stack([ex, ey, ez]),
                           col))
        return self


MESH = {}


def cq_mesh(key, shape, tol=0.04):
    """Tessellate a CadQuery Workplane once and keep (V, F)."""
    if key not in MESH:
        verts, tris = shape.val().tessellate(tol, 0.2)
        MESH[key] = (np.array([[v.x, v.y, v.z] for v in verts]), np.array(tris, dtype=np.int32))
    return key


def write_stl(path, V, F):
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 0)
    rec = np.zeros(len(F), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, tri
    with open(path, "wb") as fh:
        fh.write(b"\0" * 80 + np.uint32(len(F)).tobytes() + rec.tobytes())


def smooth(points, step=0.5):
    """Catmull-Rom through the control points, sampled about every `step` mm."""
    P = [np.array(p, float) for p in points]
    P = [P[0]] + P + [P[-1]]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        n = max(2, int(np.linalg.norm(p2 - p1) / step))
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-2])
    return out


# ---------------------------------------------------------------- the TITAN Core, from the keep-out height map
def _merge(cells):
    """Greedy rectangles over {(ix, iy): value} with equal values."""
    todo = dict(cells)
    rects = []
    for (ix, iy) in sorted(cells, key=lambda k: (k[1], k[0])):
        if (ix, iy) not in todo:
            continue
        val = todo[(ix, iy)]
        ix1 = ix
        while todo.get((ix1 + 1, iy)) == val:
            ix1 += 1
        iy1 = iy
        while all(todo.get((k, iy1 + 1)) == val for k in range(ix, ix1 + 1)):
            iy1 += 1
        for a in range(ix, ix1 + 1):
            for b in range(iy, iy1 + 1):
                todo.pop((a, b))
        rects.append((ix, ix1, iy, iy1, val))
    return rects


def board_part(jumper=None, hl_pins=()):
    """TITAN Core in its own frame (PCB top z = 0), drawn from the keep-out boxes."""
    xe, ye, zt, zb = KO["x_edges"], KO["y_edges"], KO["z_top"], KO["z_bottom"]
    nx, ny = len(xe) - 1, len(ye) - 1
    top, bot = {}, {}
    for ix in range(nx):
        for iy in range(ny):
            t, b = zt[ix][iy], zb[ix][iy]
            if not math.isfinite(t):
                continue
            x, y = (xe[ix] + xe[ix + 1]) / 2, (ye[iy] + ye[iy + 1]) / 2
            if t > 0.05:
                if 13.3 <= y <= 20.7 and t >= 4.0:
                    cls = "term"
                elif y < 6.2 and x < 6.8:
                    cls = "jst"
                elif y < 6.2 and x >= 9.0:
                    cls = "silver"
                elif 26.0 <= y <= 37.0 and 3.4 <= x <= 16.9 and t >= 1.5:
                    cls = "can"
                elif y >= 25.9 and 3.4 <= x <= 17.2:
                    cls = "module"
                elif (x < 2.6 or x > 17.7) and y > 20.4:
                    cls = "pins"
                else:
                    cls = "comp"
                top.setdefault(cls, {})[(ix, iy)] = (0.0, round(t, 2))
            if b < -1.65 and not ((x < 2.6 or x > 17.7) and y > 20.3):
                bot.setdefault("comp", {})[(ix, iy)] = (round(b, 2), -1.6)
    p = Part("board")
    p.box(0, 20.32, 0, 43.31, -1.6, 0, COL["pcb"])
    for cls, cells in list(top.items()) + list(bot.items()):
        for ix0, ix1, iy0, iy1, (z0, z1) in _merge(cells):
            p.box(xe[ix0], xe[ix1 + 1], ye[iy0], ye[iy1 + 1], z0, z1, COL[cls])
    # header: a dark plastic band under each long edge, pins below it (from the keep-out's pin cells)
    pin_y = [21.40 + 2.54 * k for k in range(9)]
    for px, (x0, x1) in zip(PIN_ROWS_X, ((0.05, 2.45), (17.85, 20.25))):
        p.box(x0, x1, 20.45, 43.25, -4.11, -1.6, COL["pins"])
        for py in pin_y:
            hl = any(abs(px - a) < 0.1 and abs(py - b) < 0.1 for a, b in hl_pins)
            p.box(px - 0.32, px + 0.32, py - 0.32, py + 0.32, -10.18, -4.11, (0.0, 0.62, 0.55, 1) if hl else COL["pins"])
    # screw heads (top) and wire holes (-Y face) of the three terminal blocks
    for name, hx in HOLE_X.items():
        p.cyl((hx, SCREW_Y, 7.0), (hx, SCREW_Y, 7.9), 1.15, COL["silver"])
        p.box(hx - 0.95, hx + 0.95, SCREW_Y - 0.18, SCREW_Y + 0.18, 7.85, 7.95, COL["hole"])
        p.box(hx - 0.85, hx + 0.85, HOLE_FACE_Y - 0.12, HOLE_FACE_Y, HOLE_REL_Z - 0.85, HOLE_REL_Z + 0.85, COL["hole"])
    if jumper is not None:
        px, py0 = jumper
        p.items.extend(jumper_part(px, py0, -4.4).items)
    return p


JUMPER_PINS = (PIN_ROWS_X[1], 21.40 + 2.54 * 4, 21.40 + 2.54 * 5)  # an example pair only (read the board's labels)


def jumper_part(px, py_mid, z_top):
    """A 2.54 mm shunt, open end at z_top, over two pins in a row along Y (board frame, pins hang down)."""
    p = Part("jumper")
    p.box(px - 1.25, px + 1.25, py_mid - 2.5, py_mid + 2.5, z_top - 6.0, z_top, COL["jumper"])
    p.box(px - 0.9, px + 0.9, py_mid - 2.2, py_mid + 2.2, z_top - 6.05, z_top - 5.95, (0.35, 0.35, 0.4, 1))
    return p


# ---------------------------------------------------------------- motors
def motor_part(name, lead_dir=None, stub=0.0):
    """DRAKE motor: axis along +Y from y 0..L, on the axis x = z = 0, feet toward -Z, lead ring on top (+Z)."""
    c = COL[name]
    dark = tuple(v * 0.72 for v in c[:3]) + (c[3],)
    light = tuple(min(1, v * 1.12 + 0.08) for v in c[:3]) + (c[3],)
    p = Part(name)
    p.cyl((0, 0, 0), (0, CAP_LEN, 0), CAP_R, c)
    p.cyl((0, L_MOT - CAP_LEN, 0), (0, L_MOT, 0), CAP_R, c)
    p.cyl((0, CAP_LEN, 0), (0, L_MOT - CAP_LEN, 0), BAND_R, light)
    p.cyl((0, L_MOT / 2 - 0.8, 0), (0, L_MOT / 2 + 0.8, 0), CAP_R - 0.1, COL["ring"])
    for y0 in (M.FOOT_START, L_MOT - M.FOOT_START - M.FOOT_LEN):
        p.box(-M.FOOT_W / 2, M.FOOT_W / 2, y0, y0 + M.FOOT_LEN, -CAP_R - PROT, -CAP_R + 1.5, dark)
    if stub:
        for dx, col in ((-0.5, "red"), (0.5, "black")):
            p.tube(smooth([(dx, L_MOT / 2, CAP_R - 0.3), (dx, L_MOT / 2, CAP_R + stub * 0.5),
                           (dx * 1.5, L_MOT / 2 + 1.5, CAP_R + stub)]), 0.45, COL[col])
    return p


def motor_lying(mx, y_front, z_feet=0.0):
    """Transform for a feet-down motor along +Y, front end at y_front."""
    return np.eye(3), np.array([mx, y_front, z_feet + ZC_LYING])


R_STAND = np.column_stack([(0, -1, 0), (0, 0, 1), (-1, 0, 0)]).astype(float)  # axis up, feet +X, leads -X


def lfi_standing(x, y, z0=0.0):
    return R_STAND, np.array([x, y, z0])


def ring_exit(R, t, side):
    """World point where a lead leaves the lead ring; side -1 red, +1 black."""
    return R @ np.array([side * 0.5, L_MOT / 2, CAP_R - 0.25]) + t


# ---------------------------------------------------------------- puck parts
def base_mesh():
    return cq_mesh("base", M.base())


def lid_mesh(installed):
    if installed:
        lifts = {n: LIFT_SPRING for n in M.SPRINGS}
        lifts["board"] = LIFT_BOARD
        return cq_mesh("lid_in", M.lid(lifts))
    return cq_mesh("lid_free", M.lid())


def section(key, shape, box):
    """shape cut to the box (x0, x1, y0, y1, z0, z1) and the thin skin of it on one face for a darker section."""
    import cadquery as cq
    x0, x1, y0, y1, z0, z1 = box
    keep = cq.Workplane("XY").box(x1 - x0, y1 - y0, z1 - z0, centered=False).translate((x0, y0, z0))
    return cq_mesh(f"{key}@cut{box}", shape.intersect(keep))


def section_face(key, shape, axis, at, toward, thick=0.06):
    """A thin slab of `shape` at the cut plane, slightly on the camera side, drawn darker."""
    import cadquery as cq
    big = 400
    lo = at - thick if toward < 0 else at
    dims = {"x": (thick, big, big), "y": (big, thick, big)}[axis]
    org = {"x": (lo, -big / 2, -big / 2), "y": (-big / 2, lo, -big / 2)}[axis]
    slab = cq.Workplane("XY").box(*dims, centered=False).translate(org)
    try:
        return cq_mesh(f"{key}@{axis}{at}{toward}", shape.intersect(slab))
    except Exception:
        return None


# ---------------------------------------------------------------- cables and straps
def usb_cable(tip_y, length=60.0, bend=None, x=M.USB_X, z=M.USB_Z):
    """USB-C plug pointing +Y with its metal tip at tip_y; the cable runs -Y."""
    p = Part("cable")
    p.box(x - 4.1, x + 4.1, tip_y - 6.6, tip_y, z - 1.25, z + 1.25, COL["silver"])
    p.box(x - 6.25, x + 6.25, tip_y - 26.6, tip_y - 6.6, z - 3.25, z + 3.25, COL["plug"])
    y0 = tip_y - 26.6
    pts = [(x, y0 + 1, z), (x, y0 - 8, z)]
    if bend is None:
        pts += [(x, y0 - length, z)]
    else:
        pts += bend
    p.tube(smooth(pts, 1.0), 2.0, COL["cable"])
    return p


def zip_tie_installed():
    """Through the saddle's tunnel and over the cable at the end of the cable tongue."""
    ty0 = M.IY0 - M.WALL - M.TONGUE_LEN
    y = ty0 + 3.6
    cb = M.USB_Z - M.CABLE_D / 2
    zb = cb - 2.9
    zt = M.USB_Z + M.CABLE_D / 2 + 0.6
    xa, xb = M.USB_X - 6.6, M.USB_X + 6.6
    p = Part("tie")
    w, t = 2.4, 0.5
    p.box(xa - 0.6, xb + 0.6, y - w / 2, y + w / 2, zb - t / 2, zb + t / 2, COL["tie"])
    for xx in (xa, xb):
        p.box(xx - t / 2 - 0.6 * (xx < M.USB_X) + 0.6 * (xx > M.USB_X), xx + t / 2 - 0.6 * (xx < M.USB_X)
              + 0.6 * (xx > M.USB_X), y - w / 2, y + w / 2, zb, zt - 1.5, COL["tie"])
    pts = [(xa - 0.6, y, zt - 1.5), (xa + 1.0, y, zt), (M.USB_X, y, zt + 0.4), (xb - 1.0, y, zt), (xb + 0.6, y, zt - 1.5)]
    for a, b in zip(pts, pts[1:]):
        p.cap(a, b, 0.35, COL["tie"])
        p.cap((a[0], y - 0.8, a[2]), (b[0], y - 0.8, b[2]), 0.35, COL["tie"])
        p.cap((a[0], y + 0.8, a[2]), (b[0], y + 0.8, b[2]), 0.35, COL["tie"])
    p.box(xb - 1.2, xb + 2.4, y - 2.0, y + 2.0, zt - 2.4, zt + 0.6, COL["tie"])  # head
    p.box(xb + 2.4, xb + 5.0, y - w / 2, y + w / 2, zt - 1.2, zt - 0.7, COL["tie"])  # tail, trimmed
    return p


STRAP_T = 1.8


def strap_installed(tail=26.0):
    """25 mm velcro: up through one flange slot, over the lid, down through the other (model.py's strap path)."""
    ymid = (M.IY0 + M.IY1) / 2
    top = M.INNER_H + M.LID + 0.2 + STRAP_T / 2
    rim = (M.IX0 - M.RIM_WALL, M.IX1 + M.RIM_WALL)
    xs = [M.IX0 - M.WALL - M.SLOT_LIGAMENT - M.STRAP_SLOT / 2, M.IX1 + M.WALL + M.SLOT_LIGAMENT + M.STRAP_SLOT / 2]
    fl_top = -M.FLOOR + M.FLANGE_T
    pts = [(xs[0], -M.FLOOR - tail), (xs[0], fl_top + 0.5), (rim[0] - 1.2, M.INNER_H - 3), (rim[0] - 0.6, top - 1.0),
           (rim[0] + 1.5, top), (rim[1] - 1.5, top), (rim[1] + 0.6, top - 1.0), (rim[1] + 1.2, M.INNER_H - 3),
           (xs[1], fl_top + 0.5), (xs[1], -M.FLOOR - tail)]
    p = Part("strap")
    for (x0, z0), (x1, z1) in zip(pts, pts[1:]):
        p.plank((x0, ymid, z0), (x1, ymid, z1), 25.0, STRAP_T, COL["strap"])
    for x, z in pts[1:-1]:
        p.cyl((x, ymid - 12.5, z), (x, ymid + 12.5, z), STRAP_T / 2, COL["strap"])
    return p


# ---------------------------------------------------------------- scene and rendering
class Cam:
    def __init__(self, target, az=0.0, el=35.0, dist=200.0, fovy=30.0, ortho=None, H=800, up=None, Wd=W):
        self.t = np.array(target, float)
        a, e = math.radians(az), math.radians(el)
        d = np.array([math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)])
        self.eye = self.t + dist * d
        f = -d
        up = np.array(up if up is not None else ((0, 1, 0) if abs(el) > 89 else (0, 0, 1)), float)
        r = np.cross(f, up)
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        self.f, self.r, self.u = f, r, u
        self.fovy, self.ortho, self.H, self.W = fovy, ortho, H, Wd

    def xml(self):
        e, r, u = self.eye, self.r, self.u
        s = (f'<camera name="c" pos="{e[0]:.4f} {e[1]:.4f} {e[2]:.4f}" '
             f'xyaxes="{r[0]:.6f} {r[1]:.6f} {r[2]:.6f} {u[0]:.6f} {u[1]:.6f} {u[2]:.6f}" ')
        if self.ortho:
            return s + f'orthographic="true" fovy="{self.ortho}"/>'
        return s + f'fovy="{self.fovy}"/>'

    def px(self, p):
        q = np.array(p, float) - self.eye
        x, y, z = q @ self.r, q @ self.u, q @ self.f
        if self.ortho:
            s = self.H / self.ortho
            return (self.W / 2 + x * s, self.H / 2 - y * s)
        fy = (self.H / 2) / math.tan(math.radians(self.fovy) / 2)
        return (self.W / 2 + x * fy / z, self.H / 2 - y * fy / z)

    def pix_mm(self, depth):
        if self.ortho:
            return self.ortho / self.H
        return depth * 2 * math.tan(math.radians(self.fovy) / 2) / self.H


def _q(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, dtype=float).flatten())
    return q


class Scene:
    def __init__(self):
        self.parts = []  # (part, R, t, alpha, layer)

    def add(self, part, R=None, t=(0, 0, 0), alpha=None, layer=0):
        self.parts.append((part, np.eye(3) if R is None else np.array(R, float), np.array(t, float), alpha, layer))
        return self

    def build(self, cam):
        mats, assets, geoms = {}, [], []

        def mat(col):
            key = tuple(round(c, 4) for c in col)
            if key not in mats:
                mats[key] = f"mat{len(mats)}"
            return mats[key]

        def f3(v):
            return f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f}"

        n = 0
        for part, R, t, alpha, layer in self.parts:
            for it in part.items:
                kind = it[0]
                col = it[-1]
                if alpha is not None:
                    col = tuple(col[:3]) + (alpha,)
                name = f"{part.group}__{n}"
                n += 1
                common = f'name="{name}" material="{mat(col)}" group="{layer}"'
                if kind == "box":
                    c, h = R @ it[1] + t, it[2]
                    q = _q(R)
                    geoms.append(f'<geom type="box" {common} pos="{f3(c)}" size="{f3(h)}" quat="{q[0]:.6f} {q[1]:.6f} '
                                 f'{q[2]:.6f} {q[3]:.6f}"/>')
                elif kind in ("cyl", "cap"):
                    a, b = R @ it[1] + t, R @ it[2] + t
                    typ = "cylinder" if kind == "cyl" else "capsule"
                    geoms.append(f'<geom type="{typ}" {common} fromto="{f3(a)} {f3(b)}" size="{it[3]:.4f}"/>')
                elif kind == "sph":
                    geoms.append(f'<geom type="sphere" {common} pos="{f3(R @ it[1] + t)}" size="{it[2]:.4f}"/>')
                elif kind == "obox":
                    c, h = R @ it[1] + t, it[2]
                    q = _q(R @ it[3])
                    geoms.append(f'<geom type="box" {common} pos="{f3(c)}" size="{f3(h)}" quat="{q[0]:.6f} {q[1]:.6f} '
                                 f'{q[2]:.6f} {q[3]:.6f}"/>')
                elif kind == "mesh":
                    V, F = MESH[it[1]]
                    fn = f"m{len(assets)}.stl"
                    with np.errstate(all="ignore"):
                        Vt = V @ R.T + t
                    write_stl(os.path.join(TMP, fn), Vt, F)
                    assets.append(f'<mesh name="mesh{len(assets)}" file="{fn}" inertia="shell"/>')
                    geoms.append(f'<geom type="mesh" {common} mesh="mesh{len(assets) - 1}"/>')
        mat_xml = "".join(f'<material name="{v}" rgba="{k[0]} {k[1]} {k[2]} {k[3]}" specular="0.18" shininess="0.12"/>'
                          for k, v in mats.items())
        return f"""<mujoco><compiler meshdir="{TMP}"/>
<visual><global offwidth="{cam.W}" offheight="{cam.H}"/><quality offsamples="8"/>
<headlight ambient=".44 .44 .44" diffuse=".50 .50 .50" specular=".06 .06 .06"/>
<map znear="0.002" zfar="40"/><rgba haze="1 1 1 1"/></visual>
<statistic extent="120" center="20 10 10"/>
<asset><texture type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="8" height="8"/>{mat_xml}{"".join(assets)}</asset>
<worldbody><light directional="true" pos="0 0 300" dir="0.35 0.55 -1" diffuse=".32 .32 .32" specular=".05 .05 .05"
castshadow="false"/>{"".join(geoms)}{cam.xml()}</worldbody></mujoco>"""


def render(scene, cam, outline=True):
    xml = scene.build(cam)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, cam.H, cam.W)
    opt = mujoco.MjvOption()
    r.update_scene(d, camera="c", scene_option=opt)
    rgb = r.render().astype(np.float32)
    groups = np.array([m.geom(i).name.split("__")[0] for i in range(m.ngeom)] + ["<bg>"])
    edges = np.zeros(rgb.shape[:2], bool)
    soft = np.zeros(rgb.shape[:2], bool)
    if outline:
        for layers, target in (((1, 0, 0, 0, 0, 0), edges), ((0, 1, 0, 0, 0, 0), soft)):
            opt2 = mujoco.MjvOption()
            opt2.geomgroup[:] = layers
            if not any(m.geom_group[i] == layers.index(1) for i in range(m.ngeom)):
                continue
            r.update_scene(d, camera="c", scene_option=opt2)
            r.enable_segmentation_rendering()
            seg = r.render()[:, :, 0].copy()
            r.disable_segmentation_rendering()
            r.enable_depth_rendering()
            dep = r.render().copy()
            r.disable_depth_rendering()
            gid = np.searchsorted(np.unique(groups), groups)
            lab = np.where(seg >= 0, gid[np.clip(seg, 0, m.ngeom - 1)], -1)
            e = np.zeros_like(target)
            e[:, 1:] |= lab[:, 1:] != lab[:, :-1]
            e[1:, :] |= lab[1:, :] != lab[:-1, :]
            thr = (6.0 * (cam.ortho / cam.H) if cam.ortho else
                   6.0 * dep * 2 * math.tan(math.radians(cam.fovy) / 2) / cam.H)
            dz_x = np.abs(dep[:, 1:] - dep[:, :-1])
            dz_y = np.abs(dep[1:, :] - dep[:-1, :])
            thr_x = thr[:, 1:] if np.ndim(thr) else thr
            thr_y = thr[1:, :] if np.ndim(thr) else thr
            e[:, 1:] |= (dz_x > thr_x) & (lab[:, 1:] >= 0) & (lab[:, :-1] >= 0)
            e[1:, :] |= (dz_y > thr_y) & (lab[1:, :] >= 0) & (lab[:-1, :] >= 0)
            target |= e
        img = rgb
        for mask, colr, a in ((soft, (95, 110, 140), 0.75), (edges, (38, 38, 44), 0.85)):
            k = mask.copy()
            k[1:, :] |= mask[:-1, :]
            k[:, 1:] |= mask[:, :-1]
            img[k] = img[k] * (1 - a) + np.array(colr, np.float32) * a
        rgb = img
    r.close()
    for f in os.listdir(TMP):
        os.remove(os.path.join(TMP, f))
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------- 2D annotation
class Ink:
    def __init__(self, img, cam=None, oy=0, ox=0):
        self.img, self.cam, self.oy, self.ox = img, cam, oy, ox
        self.d = ImageDraw.Draw(img)

    def P(self, p):
        if len(p) == 2:
            return (p[0], p[1])
        x, y = self.cam.px(p)
        return (x + self.ox, y + self.oy)

    def arrow(self, a, b, col, w=11, head=34, outline=True):
        a, b = np.array(self.P(a)), np.array(self.P(b))
        v = b - a
        ln = np.linalg.norm(v)
        if ln < 1:
            return
        u = v / ln
        nrm = np.array([-u[1], u[0]])
        tip = b
        base = b - u * head
        tri = [tuple(tip), tuple(base + nrm * head * 0.55), tuple(base - nrm * head * 0.55)]
        if outline:
            self.d.line([tuple(a), tuple(base)], fill=(255, 255, 255), width=w + 8)
            tri_o = [tuple(tip + u * 5), tuple(base + nrm * (head * 0.55 + 6) - u * 3),
                     tuple(base - nrm * (head * 0.55 + 6) - u * 3)]
            self.d.polygon(tri_o, fill=(255, 255, 255))
        self.d.line([tuple(a), tuple(base + u * 2)], fill=col, width=w)
        self.d.polygon(tri, fill=col)

    def curve_arrow(self, pts, col, w=9, head=28):
        pts = [np.array(self.P(p)) for p in pts]
        self.d.line([tuple(p) for p in pts[:-1]], fill=(255, 255, 255), width=w + 8, joint="curve")
        self.d.line([tuple(p) for p in pts[:-1]], fill=col, width=w, joint="curve")
        self.arrow(tuple(pts[-2]), tuple(pts[-1]), col, w=w, head=head)

    def text(self, xy, s, size=34, col=INK, bold=True, anchor="la", box=False, pad=8):
        f = font(size, bold)
        if box:
            bb = self.d.multiline_textbbox(xy, s, font=f, anchor=anchor, spacing=6)
            self.d.rounded_rectangle((bb[0] - pad, bb[1] - pad, bb[2] + pad, bb[3] + pad), radius=10,
                                     fill=(255, 255, 255), outline=(200, 200, 205), width=2)
        self.d.multiline_text(xy, s, font=f, fill=col, anchor=anchor, spacing=6)
        return self.d.multiline_textbbox(xy, s, font=f, anchor=anchor, spacing=6)

    def label(self, anchor, s, at, col=INK, size=34, dot=True):
        """Text box at `at` (pixel, centre) with a leader line to the 3D (or 2D) anchor."""
        ap = self.P(anchor)
        f = font(size)
        bb = self.d.multiline_textbbox(at, s, font=f, anchor="mm", spacing=6)
        pad = 9
        box = (bb[0] - pad, bb[1] - pad, bb[2] + pad, bb[3] + pad)
        cx = min(max(ap[0], box[0]), box[2])
        cy = min(max(ap[1], box[1]), box[3])
        self.d.line([ap, (cx, cy)], fill=(255, 255, 255), width=7)
        self.d.line([ap, (cx, cy)], fill=col, width=3)
        if dot:
            self.d.ellipse((ap[0] - 7, ap[1] - 7, ap[0] + 7, ap[1] + 7), fill=col, outline=(255, 255, 255), width=2)
        self.d.rounded_rectangle(box, radius=10, fill=(255, 255, 255), outline=col, width=3)
        self.d.multiline_text(at, s, font=f, fill=col, anchor="mm", spacing=6, align="center")


def wrap(s, f, width):
    words, lines, cur = s.split(), [], ""
    for w_ in words:
        t = (cur + " " + w_).strip()
        if f.getlength(t) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w_
    if cur:
        lines.append(cur)
    return lines


def page(step, title, panels, notes, accent=(30, 30, 36)):
    """Stack header, panels (PIL images) and numbered notes into one 1400-wide page."""
    fh = font(52)
    fb = font(34, bold=False)
    fbb = font(34)
    head_h = 118
    body = []
    for n in notes:
        body.append(wrap(n, fb, W - 150))
    body_h = sum(len(b) * 44 + 16 for b in body) + 36
    h = head_h + sum(p.height for p in panels) + body_h
    img = Image.new("RGB", (W, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, W, head_h - 10), fill=(245, 246, 248))
    if step is not None:
        d.ellipse((28, 16, 28 + 82, 16 + 82), fill=accent)
        d.text((28 + 41, 16 + 42), str(step), font=font(52), fill=(255, 255, 255), anchor="mm")
        d.text((136, 56), title, font=fh, fill=INK, anchor="lm")
    else:
        d.text((40, 56), title, font=fh, fill=INK, anchor="lm")
    y = head_h
    for p in panels:
        img.paste(p, (0, y))
        y += p.height
    y += 16
    for i, lines in enumerate(body):
        tag = "•"
        d.text((44, y), tag, font=fbb, fill=accent)
        for ln in lines:
            d.text((80, y), ln, font=fb, fill=INK)
            y += 44
        y += 16
    return img


def save(img, name):
    path = os.path.join(HERE, name)
    img.save(path, optimize=True)
    print("wrote", name, img.size)
    return path


# ---------------------------------------------------------------- installed state
def installed_leads():
    """Final lead routing (world mm): {motor: {'red': pts, 'black': pts}} ending inside the terminal holes."""
    hz = M.PCB_TOP + HOLE_REL_Z
    lane = M.WIRE_SLOT_Y
    zc0, zc1 = M.WIRE_GUIDE_X0, M.WIRE_GUIDE_X1
    zl = M.WIRE_SLOT_Z + 0.45  # lower lead in the lane, the other stacked on it
    R0, t_mf = motor_lying(M.MF[0], M.MOTOR_FRONT)
    _, t_lf = motor_lying(M.LF[0], M.MOTOR_FRONT)
    Rs, t_lfi = lfi_standing(*M.LFI)
    out = {}
    # MF -> M (front lane): south, west through the comb, north into M+ / M-
    ex_r, ex_b = ring_exit(R0, t_mf, -1), ring_exit(R0, t_mf, 1)
    yl = lane["MF"]
    out["MF"] = {
        "red": [ex_r, ex_r + (0, 0, 1.2), (27.2, 13.6, 13.4), (25.0, 10.4, 14.3), (23.6, yl - 0.25, zl + 0.8),
                (zc1, yl - 0.25, zl + 0.8), (zc0, yl - 0.25, zl + 0.8), (18.0, 7.8, hz + 0.3), (6.0, 7.8, hz),
                (3.0, 8.4, hz), (1.85, 10.0, hz), (HOLE_X["M+"], 12.2, hz), (HOLE_X["M+"], 16.3, hz)],
        "black": [ex_b, ex_b + (0, 0, 1.2), (27.9, 13.6, 13.2), (25.6, 10.6, 13.9), (24.0, yl + 0.25, zl),
                  (zc1, yl + 0.25, zl), (zc0, yl + 0.25, zl), (18.0, 8.35, hz), (7.5, 8.4, hz), (5.2, 8.9, hz),
                  (4.4, 10.4, hz), (HOLE_X["M-"], 12.2, hz), (HOLE_X["M-"], 16.3, hz)],
    }
    # LF -> L (middle lane): over MF, west through the comb, north into L+ / L-
    ex_r, ex_b = ring_exit(R0, t_lf, -1), ring_exit(R0, t_lf, 1)
    yl = lane["LF"]
    out["LF"] = {
        "red": [ex_r, ex_r + (0, 0, 1.5), (38.0, 14.2, 13.4), (31.5, 12.6, 12.6), (26.0, 11.3, 13.9),
                (23.6, yl - 0.25, zl + 0.8), (zc1, yl - 0.25, zl + 0.8), (zc0, yl - 0.25, zl + 0.8),
                (18.5, 9.4, hz + 0.3), (13.0, 9.3, hz), (10.0, 9.4, hz), (8.95, 10.6, hz), (HOLE_X["L+"], 12.2, hz),
                (HOLE_X["L+"], 16.3, hz)],
        "black": [ex_b, ex_b + (0, 0, 1.5), (38.6, 14.6, 13.2), (31.8, 13.0, 12.4), (26.4, 11.6, 13.6),
                  (24.0, yl + 0.25, zl), (zc1, yl + 0.25, zl), (zc0, yl + 0.25, zl), (18.5, 9.95, hz),
                  (13.4, 9.9, hz), (11.9, 10.4, hz), (11.5, 11.6, hz), (HOLE_X["L-"], 12.4, hz),
                  (HOLE_X["L-"], 16.3, hz)],
    }
    # LFi -> R (back lane): down past MF's inboard side, through the comb, a U-loop into R+ / R-
    ex_r, ex_b = ring_exit(Rs, t_lfi, -1), ring_exit(Rs, t_lfi, 1)
    yl = lane["LFi"]
    out["LFi"] = {
        "red": [ex_r, ex_r + (-1.2, 0, 0), (28.6, 34.2, 12.6), (26.0, 29.0, 13.4), (25.0, 22.0, 13.8),
                (24.6, 16.0, 14.2), (23.6, yl - 0.25, zl + 0.8), (zc1, yl - 0.25, zl + 0.8),
                (zc0, yl - 0.25, zl + 0.8), (19.4, 12.0, hz + 0.3), (17.6, 11.1, hz), (16.4, 11.4, hz),
                (16.08, 12.3, hz), (HOLE_X["R+"], 13.2, hz), (HOLE_X["R+"], 16.3, hz)],
        "black": [ex_b, ex_b + (-1.2, 0, 0), (29.0, 34.9, 12.4), (26.5, 29.2, 13.1), (25.5, 22.0, 13.5),
                  (25.1, 16.0, 13.9), (24.0, yl + 0.25, zl), (zc1, yl + 0.25, zl), (zc0, yl + 0.25, zl),
                  (19.6, 12.75, hz), (18.8, 12.55, hz), (HOLE_X["R-"], 13.1, hz), (HOLE_X["R-"], 13.6, hz),
                  (HOLE_X["R-"], 16.3, hz)],
    }
    return out


def add_leads(sc, leads, r=0.42, layer=0):
    for mot, pair in leads.items():
        for colname, pts in pair.items():
            p = Part(f"lead_{mot}_{colname}")
            p.tube(smooth(pts, 0.5), r, COL[colname])
            sc.add(p, layer=layer)


def add_installed(sc, board=True, motors=True, leads=True, lid=False, lid_alpha=None, jumper=True, base=True,
                  board_dz=0.0):
    if base:
        sc.add(Part("base").mesh(base_mesh(), COL["base"]))
    if motors:
        R0, t = motor_lying(M.MF[0], M.MOTOR_FRONT)
        sc.add(motor_part("MF"), R0, t)
        R0, t = motor_lying(M.LF[0], M.MOTOR_FRONT)
        sc.add(motor_part("LF"), R0, t)
        Rs, t = lfi_standing(*M.LFI)
        sc.add(motor_part("LFi"), Rs, t)
    if board:
        sc.add(board_part(JUMPER_PINS[:1] + ((JUMPER_PINS[1] + JUMPER_PINS[2]) / 2,) if jumper else None),
               t=(0, 0, M.PCB_TOP + board_dz))
    if leads:
        add_leads(sc, installed_leads())
    if lid:
        col = COL["lid"] if lid_alpha is None else COL["lid"][:3] + (lid_alpha,)
        sc.add(Part("lid").mesh(lid_mesh(True), col), layer=1)




def motor_solid(name, lying_at=None, standing_at=None):
    """CadQuery motor (body + feet) and its lead ring, for cut-away views."""
    import cadquery as cq
    V = cq.Vector
    if lying_at is not None:
        mx, y0 = lying_at
        zc, ax = ZC_LYING, V(0, 1, 0)

        def P(s, r=0.0):
            return V(mx, y0 + s, zc)
        feet = [cq.Solid.makeBox(M.FOOT_W, M.FOOT_LEN, PROT + 1.5, V(mx - M.FOOT_W / 2, y0 + s, 0))
                for s in (M.FOOT_START, L_MOT - M.FOOT_START - M.FOOT_LEN)]
    else:
        lx, ly = standing_at
        ax = V(0, 0, 1)

        def P(s, r=0.0):
            return V(lx, ly, s)
        feet = [cq.Solid.makeBox(PROT + 1.5, M.FOOT_W, M.FOOT_LEN, V(lx + CAP_R - 1.5, ly - M.FOOT_W / 2, s))
                for s in (M.FOOT_START, L_MOT - M.FOOT_START - M.FOOT_LEN)]
    body = cq.Solid.makeCylinder(CAP_R, CAP_LEN, P(0), ax)
    for part in [cq.Solid.makeCylinder(BAND_R, L_MOT - 2 * CAP_LEN, P(CAP_LEN), ax),
                 cq.Solid.makeCylinder(CAP_R, CAP_LEN, P(L_MOT - CAP_LEN), ax)] + feet:
        body = body.fuse(part)
    ring = cq.Solid.makeCylinder(CAP_R - 0.1, 1.6, P(L_MOT / 2 - 0.8), ax)
    return cq.Workplane().add(body.clean()), cq.Workplane().add(ring)


def add_cut(sc, key, shape, box, col, cut_col, axis, at, toward, alpha=None, layer=0):
    sc.add(Part(key).mesh(section(key, shape, box), col), alpha=alpha, layer=layer)
    face = section_face(key + "_face", shape, axis, at, toward)
    if face:
        sc.add(Part(key).mesh(face, cut_col), layer=layer)


def darker(c, k=0.62):
    return tuple(v * k for v in c[:3]) + (1,)


def lead_stubs(sc, R, t, mot, rise=14.0, lean=(0, 0, 0)):
    """Loose leads from a motor's ring (for steps before the board is in)."""
    for side, colname in ((-1, "red"), (1, "black")):
        a = ring_exit(R, t, side)
        up = R @ np.array([0, 0, 1.0])
        pts = [a, a + up * 3, a + up * rise * 0.6 + np.array(lean) * 0.5 + (side * 0.6, 0, 0),
               a + up * rise + np.array(lean) + (side * 0.9, 0, 0)]
        sc.add(Part(f"lead_{mot}_{colname}").tube(smooth(pts, 0.6), 0.45, COL[colname]))


# ---------------------------------------------------------------- step 1: parts
def step01():
    sc = Scene()
    sc.add(Part("base").mesh(base_mesh(), COL["base"]), t=(-88.3, 34.0, M.FLOOR))
    sc.add(Part("lid").mesh(lid_mesh(False), COL["lid"][:3] + (0.7,)), R=rot((1, 0, 0), 180),
           t=(-1.5, 47.0, M.INNER_H + M.LID), layer=1)
    strap = Part("strap")
    strap.plank((70, -2, 0.9), (70, 88, 0.9), 25.0, STRAP_T, COL["strap"], across=(1, 0, 0))
    sc.add(strap)
    sc.add(board_part(), t=(-106, -78, 10.18))
    Rx = rot((0, 0, 1), -90)  # lying along +X
    x0 = -76.0
    for nm, yc in (("MF", -24.0), ("LF", -44.0), ("LFi", -64.0)):
        t = np.array([x0, yc, ZC_LYING])
        sc.add(motor_part(nm), Rx, t)
        for side, colname in ((-1, "red"), (1, "black")):
            a = ring_exit(Rx, t, side)
            pts = [a, a + (0, 0, 2.5), (x0 + L_MOT + 3, a[1] - side * 0.2, 9.0), (x0 + L_MOT + 9, a[1] - side * 0.5, 1.0),
                   (x0 + L_MOT + 30, a[1] - side * 0.8, 0.5)]
            sc.add(Part(f"lead_{nm}_{colname}").tube(smooth(pts, 0.6), 0.45, COL[colname]))
    sc.add(jumper_part(0, 0, 0), R=rot((1, 0, 0), 180), t=(-18.0, -52.0, 0.0))
    sc.add(usb_cable(-18.0, x=6.0, z=3.25, bend=[(6.0, -60, 2.0), (14, -72, 2.0), (30, -75, 2.0), (42, -66, 2.0)]))
    tie = Part("tie")
    tie.plank((54.0, -78, 0.5), (54.0, -18, 0.5), 2.5, 1.0, COL["tie"], across=(1, 0, 0))
    tie.box(52.0, 56.0, -18, -13.5, 0, 3.0, COL["tie"])
    sc.add(tie)
    cam = Cam((-16, -2, 0), az=0, el=62, dist=345, fovy=30, H=1000)
    img = render(sc, cam)
    ink = Ink(img, cam)
    ink.label((-58, 60, 22), "base", (180, 150), RGB["base"])
    ink.label((22, 22, 12), "lid (upside\ndown here)", (770, 150), RGB["lid"])
    ink.label((70, 60, 2), "25 mm velcro\nstrap", (1220, 160), INK)
    ink.label((-96, -55, 20), "TITAN Core", (120, 560), RGB["board"])
    for nm, yc in (("MF", -24.0), ("LF", -44.0), ("LFi", -64.0)):
        ink.label((x0 + 17, yc + 3.0, 10), nm, {"MF": (590, 668), "LF": (590, 783), "LFi": (590, 955)}[nm], RGB[nm])
    ink.label((-18.0, -52.0, 5.5), "jumper", (738, 955), INK)
    ink.label((6, -30, 6.5), "USB-C cable", (720, 560), INK)
    ink.label((54, -45, 1), "zip tie\n(2.5 mm max)", (1225, 600), INK)
    notes = ["Check you have every part before you start.",
             "The motor colours in these pictures are codes only: read the kit's labels to tell LF, MF and LFi apart."]
    return save(page(1, "Parts", [img], notes), "step01.png")


# ---------------------------------------------------------------- step 2: jumper
def step02():
    sc = Scene()
    Rf = np.diag([-1.0, 1.0, -1.0])
    tf = np.array([20.32, 0.0, 0.0])
    jx, jy = JUMPER_PINS[0], (JUMPER_PINS[1] + JUMPER_PINS[2]) / 2
    sc.add(board_part(hl_pins=((jx, JUMPER_PINS[1]), (jx, JUMPER_PINS[2]))), Rf, tf)
    lift = np.array([0, 0, 8.0])
    sc.add(jumper_part(jx, jy, -4.4), Rf, tf + lift)
    cam = Cam((10.5, 24, 4), az=-40, el=42, dist=112, fovy=30, H=800)
    img = render(sc, cam)
    ink = Ink(img, cam)
    tip = Rf @ np.array([jx, jy, -10.18]) + tf
    ink.arrow(tip + lift + (0, 0, 16), tip + lift + (0, 0, 1.6), RGB["act"], w=12)
    ink.label(Rf @ np.array([jx, jy + 2.5, -7.0]) + tf + lift, "jumper", (230, 180), INK)
    ink.label(Rf @ np.array([jx, JUMPER_PINS[2], -10.18]) + tf, "IO19 + IO22\n(example pins)", (1130, 330),
              RGB["act"])
    ink.label(Rf @ np.array([PIN_ROWS_X[0], 41.72, -10.18]) + tf, "header pins", (900, 110), INK)
    ink.label(Rf @ np.array([13.25, -1.3, 1.6]) + tf, "USB-C end", (1100, 700), INK)
    ink.label(Rf @ np.array([19.8, 15.0, 6.0]) + tf, "screw terminals\n(face down now)", (250, 690), RGB["board"])
    notes = ["Turn the board over. Push the jumper fully onto the two pins labelled IO19 and IO22: this sets "
             "Bluetooth mode.",
             "Use the pin labels on your board: the pair drawn here is only an example. The jumper cannot be reached "
             "once the board is in the puck."]
    return save(page(2, "Bluetooth jumper on IO19 + IO22", [img], notes), "step02.png")


# ---------------------------------------------------------------- step 3: wire the motors to the terminals
BENCH_Z = 10.18  # the board standing on its header pins on the bench


def bench_leads(mots, hz):
    out = {}
    pairs = {"MF": ("M+", "M-"), "LF": ("L+", "L-"), "LFi": ("R+", "R-")}
    for nm, (R, t) in mots.items():
        out[nm] = {}
        for side, colname, hole in ((-1, "red", pairs[nm][0]), (1, "black", pairs[nm][1])):
            a = ring_exit(R, t, side)
            hx = HOLE_X[hole]
            out[nm][colname] = [a, a + (0, 1.0, 4.0), ((a[0] + hx) / 2, (a[1] - 4) / 2, hz + 6.5),
                                (hx, 3.0, hz + 6.0), (hx, 9.0, hz + 1.2), (hx, 12.0, hz), (hx, 16.3, hz)]
    return out


def screwdriver(x, y, z0, col=(0.0, 0.59, 0.53, 1)):
    p = Part("driver")
    p.box(x - 1.0, x + 1.0, y - 0.2, y + 0.2, z0 - 0.4, z0 + 1.5, COL["silver"])
    p.cyl((x, y, z0 + 1.0), (x, y, z0 + 16), 0.9, COL["silver"])
    p.cyl((x, y, z0 + 16), (x, y, z0 + 40), 3.2, col)
    return p


def step03():
    mots = {nm: motor_lying(x, -44.0) for nm, x in (("MF", -8.0), ("LF", 10.0), ("LFi", 28.0))}
    hz = BENCH_Z + HOLE_REL_Z

    def scene(close):
        sc = Scene()
        sc.add(board_part(), t=(0, 0, BENCH_Z))
        for nm, (R, t) in mots.items():
            sc.add(motor_part(nm), R, t)
        add_leads(sc, bench_leads(mots, hz))
        if close:
            sc.add(screwdriver(HOLE_X["L-"], SCREW_Y, BENCH_Z + 7.9))
        return sc

    camA = Cam((10, -12, 7), az=-18, el=42, dist=132, fovy=30, H=620)
    A = render(scene(False), camA)
    ink = Ink(A, camA)
    ink.label((-8, -40, 11.5), "MF", (170, 400), RGB["MF"])
    ink.label((10, -40, 11.5), "LF", (760, 590), RGB["LF"])
    ink.label((28, -40, 11.5), "LFi", (1180, 460), RGB["LFi"])
    ink.label((3, 17.5, BENCH_Z + 8.5), "MF  to  M", (250, 70), RGB["MF"])
    ink.label((10.15, 17.5, BENCH_Z + 8.5), "LF  to  L", (700, 50), RGB["LF"])
    ink.label((17.3, 17.5, BENCH_Z + 8.5), "LFi  to  R", (1130, 90), RGB["LFi"])
    camB = Cam((10.15, 14, BENCH_Z + 5.0), az=0, el=26, ortho=19.0, H=560)
    B = render(scene(True), camB)
    ink = Ink(B, camB)
    for hole, hx in HOLE_X.items():
        col = RGB["red"] if hole.endswith("+") else INK
        x, y = ink.P((hx, HOLE_FACE_Y, hz + 1.55))
        ink.text((x, y), hole, size=32, col=col, anchor="mb", box=True, pad=4)
    sx, sy = ink.P((HOLE_X["L-"], SCREW_Y, BENCH_Z + 12.5))
    ink.d.arc((sx - 64, sy - 22, sx + 64, sy + 22), 200, 340, fill=RGB["act"], width=9)
    ink.arrow((sx + 44, sy - 21), (sx + 66, sy - 3), RGB["act"], w=8, head=24)
    ink.text((1010, 30), "1  loosen (screw on top)\n3  tighten", size=32, col=RGB["act"], box=True)
    ink.text((24, 24), "Close-up, front view", size=30, col=INK, box=True)
    notes = ["Before the board goes in: MF to M, LF to L, LFi to R. Red lead to +, black lead to -.",
             "For each hole: 1 loosen the screw from the top, 2 push the bare end into the front hole, 3 tighten, "
             "4 tug the lead gently: it must not come out."]
    return save(page(3, "Wire the motors to the screw terminals", [A, B], notes), "step03.png")


# ---------------------------------------------------------------- step 4: lying motors
def step04():
    sc = Scene()
    sc.add(Part("base").mesh(base_mesh(), COL["base"]))
    for nm, (mx, _) in (("MF", M.MF), ("LF", M.LF)):
        R, t = motor_lying(mx, M.MOTOR_FRONT)
        sc.add(motor_part(nm), R, t)
        lead_stubs(sc, R, t, nm, rise=12, lean=(-4, 0, 0))
    camA = Cam((30, 18, 5), az=18, el=60, dist=125, fovy=30, H=600)
    A = render(sc, camA)
    ink = Ink(A, camA)
    for nm, (mx, my) in (("MF", M.MF), ("LF", M.LF)):
        for dy in (-8.0, 8.0):
            ink.arrow((mx, my + dy, ZC_LYING + CAP_R + 16), (mx, my + dy, ZC_LYING + CAP_R + 1.5), RGB[nm], w=10,
                      head=28)
    ink.label((M.MF[0] - 3, M.MF[1] - 6, ZC_LYING + 2), "MF: the cradle\nnext to the board", (220, 470), RGB["MF"])
    ink.label((M.LF[0] + 4.5, M.LF[1] - 4, ZC_LYING + 2), "LF: outboard\ncradle", (1200, 520), RGB["LF"])
    ink.label(ring_exit(*motor_lying(M.LF[0], M.MOTOR_FRONT), 0) + (0, 0, 9), "leads up", (1170, 170), INK)

    # close-up: cut along MF's axis, seen from +X
    sc = Scene()
    x0, x1 = M.WIRE_GUIDE_X1 + 0.1, M.MF[0]  # MF's inboard cap walls, not the comb and ledge beside them
    add_cut(sc, "base", M.base(), (x0, x1, -5, 32, -3, 14), COL["base"], darker(COL["base"]), "x", M.MF[0], 1)
    R, t = motor_lying(M.MF[0], M.MOTOR_FRONT)
    ghost = motor_part("MF")
    ghost.items = [it for it in ghost.items if it[0] != "box"]
    ghost.items = [it[:-1] + ((0.86, 0.80, 0.98, 1),) if it[-1] != COL["ring"] else it for it in ghost.items]
    sc.add(ghost, R, t, alpha=0.38, layer=0)
    feet = Part("MF_feet")
    feet.items = [it for it in motor_part("MF").items if it[0] == "box"]
    sc.add(feet, R, t)
    lead_stubs(sc, R, t, "MF", rise=3.2)
    camB = Cam((M.MF[0], M.MOTOR_Y_C, 4.6), az=90, el=14, ortho=17.0, H=560)
    B = render(sc, camB)
    ink = Ink(B, camB)
    y_f1 = M.MOTOR_FRONT + M.FOOT_START + M.FOOT_LEN / 2
    y_f2 = M.MOTOR_FRONT + L_MOT - M.FOOT_START - M.FOOT_LEN / 2
    ink.label((M.MF[0], y_f1, 0.5), "foot", (330, 490), RGB["MF"])
    ink.label((M.MF[0], y_f2, 0.5), "foot", (1070, 490), RGB["MF"])
    ink.label((M.MF[0], M.MOTOR_Y_C, M.RIB_H / 2), "rib 0.6 mm between the feet", (700, 505), RGB["base"])
    ink.label((M.MF[0], 1.2, -M.FLOOR / 2), "skin floor 1.0 mm", (170, 390), RGB["base"])
    ink.label((x0 + 0.6, M.MOTOR_FRONT + 2.0, M.CAP_WALL_H - 0.3), "cap wall", (140, 110), RGB["base"])
    ink.label((x0 + 0.6, M.MOTOR_FRONT + L_MOT - 2.0, M.CAP_WALL_H - 0.3), "cap wall", (1270, 110), RGB["base"])
    ink.label(ring_exit(R, t, 0) + (0, 0, 2.5), "leads up", (700, 60), INK)
    ink.text((24, 24), "Cut-away along the MF axis", size=30, col=INK, box=True)
    notes = ["Lay MF in the cradle next to the board and LF in the outboard cradle, feet down, leads pointing up.",
             "The two feet must straddle the floor rib. If a motor rocks, shave the rib ends with a knife."]
    return save(page(4, "Lay LF and MF in their cradles, feet down", [A, B], notes), "step04.png")


# ---------------------------------------------------------------- step 5: LFi
def step05():
    sc = Scene()
    sc.add(Part("base").mesh(base_mesh(), COL["base"]))
    for nm, (mx, _) in (("MF", M.MF), ("LF", M.LF)):
        R, t = motor_lying(mx, M.MOTOR_FRONT)
        sc.add(motor_part(nm), R, t)
        lead_stubs(sc, R, t, nm, rise=12, lean=(-4, 0, 0))
    lift = 26.0
    R, t = lfi_standing(*M.LFI)
    t = t + (0, 0, lift)
    sc.add(motor_part("LFi"), R, t)
    lead_stubs(sc, R, t, "LFi", rise=10, lean=(0, -3, 3))
    cam = Cam((31, 24, 14), az=-32, el=52, dist=150, fovy=30, H=820)
    img = render(sc, cam)
    ink = Ink(img, cam)
    lx, ly = M.LFI
    ink.arrow((lx - 9, ly - 3, lift + 8), (lx - 9, ly - 3, 7), RGB["LFi"], w=12)
    ink.label((lx, ly - M.LFI_BORE / 2 - 0.6, M.LFI_SLEEVE_H), "short sleeve\n(4.6 mm)", (1180, 620), RGB["base"])
    ink.label((lx + 3, ly, lift + L_MOT), "LFi: stands up,\neither end down", (1130, 110), RGB["LFi"])
    ink.label(ring_exit(R, t, 0) + (-2.5, 0, 0.5), "lead leaves the middle band:\npoint it toward the board",
              (330, 110), INK)
    notes = ["Stand LFi upright in the short sleeve behind the two lying motors. Either end can go down.",
             "Its lead comes out of the middle band, above the sleeve (the sleeve has no slot in this print). "
             "Point it toward the board."]
    return save(page(5, "Stand LFi in its sleeve", [img], notes), "step05.png")


# ---------------------------------------------------------------- step 6: board in
def step06():
    # A: where the board lands
    sc = Scene()
    add_installed(sc, board=False, leads=False)
    pads = Part("pads")
    z = M.PCB_BOTTOM
    pads.box(-0.4, 1.5, 0.5, 13.5, z, z + 0.25, (0.15, 0.85, 0.35, 1))
    pads.box(18.8, M.WIRE_GUIDE_X0, 0.5, 13.5, z, z + 0.25, (0.15, 0.85, 0.35, 1))
    for x, y in ((5.0, 38.5), (15.3, 38.5)):
        pads.cyl((x, y, z), (x, y, z + 0.25), 1.5, (0.15, 0.85, 0.35, 1))
    sc.add(pads)
    camA = Cam((15, 17, 5), az=18, el=63, dist=140, fovy=30, H=560)
    A = render(sc, camA)
    ink = Ink(A, camA)
    ink.label((0.5, 7.0, M.PCB_BOTTOM), "ledge", (220, 420), RGB["base"])
    ink.label((19.7, 7.0, M.PCB_BOTTOM), "ledge", (900, 640), RGB["base"])
    ink.label((5.0, 38.5, M.PCB_BOTTOM), "pillar", (220, 140), RGB["base"])
    ink.label((15.3, 38.5, M.PCB_BOTTOM), "pillar", (1180, 120), RGB["base"])
    ink.label((M.USB_X, M.IY0 - 0.6, M.USB_Z - 3), "USB-C opening", (330, 525), INK)
    ink.text((24, 24), "Where the board sits (green)", size=30, col=INK, box=True)
    # B: lowering it, leads already on the terminals
    sc = Scene()
    add_installed(sc, board=False, leads=False)
    lift = 30.0
    sc.add(board_part(JUMPER_PINS[:1] + ((JUMPER_PINS[1] + JUMPER_PINS[2]) / 2,)), t=(0, 0, M.PCB_TOP + lift))
    mots = {"MF": motor_lying(M.MF[0], M.MOTOR_FRONT), "LF": motor_lying(M.LF[0], M.MOTOR_FRONT),
            "LFi": lfi_standing(*M.LFI)}
    hz = M.PCB_TOP + lift + HOLE_REL_Z
    pairs = {"MF": ("M+", "M-"), "LF": ("L+", "L-"), "LFi": ("R+", "R-")}
    for nm, (R, t) in mots.items():
        for side, colname, hole in ((-1, "red", pairs[nm][0]), (1, "black", pairs[nm][1])):
            a = ring_exit(R, t, side)
            hx = HOLE_X[hole]
            up = R @ np.array([0, 0, 1.0])
            pts = [a, a + up * 3, (a[0] - 3, min(a[1], 20) - 6, hz - 16), ((hx + 22) / 2, 2.0, hz - 3),
                   (hx, 8.0, hz + 0.6), (hx, 12.0, hz), (hx, 16.3, hz)]
            sc.add(Part(f"lead_{nm}_{colname}").tube(smooth(pts, 0.6), 0.45, COL[colname]))
    camB = Cam((18, 14, 24), az=-26, el=22, dist=150, fovy=30, H=600)
    B = render(sc, camB)
    ink = Ink(B, camB)
    for (x, y) in ((-3.0, 3), (-3.0, 40)):
        ink.arrow((x, y, M.PCB_TOP + lift + 2), (x, y, M.INNER_H + 3), RGB["board"], w=11, head=30)
    ink.label((10, 30, M.PCB_TOP + lift + 3), "TITAN Core, jumper on:\nkeep it level", (320, 70), RGB["board"])
    ink.label((M.USB_X, -1.3, M.PCB_TOP + lift + 1.6), "USB-C socket\ngoes to the opening", (230, 420), INK)
    notes = ["Lower the board level onto its two ledges (front) and two pillars (back), USB-C socket to the wall "
             "opening.",
             "Do not trap a lead under the board."]
    return save(page(6, "Lower the board in, level", [A, B], notes), "step06.png")


# ---------------------------------------------------------------- step 7: route the leads
def keep_clear(ink):
    for name, (px, py) in list(M.SPRINGS.items()) + [("board", M.BOARD_POST)]:
        h = (M.POST_W if name != "board" else M.BOARD_POST_W) / 2 + 0.4
        z = M.SPRING_FREE_Z + 1 if name != "board" else M.BOARD_FREE_Z
        c = [ink.P(p) for p in ((px - h, py - h, z), (px + h, py - h, z), (px + h, py + h, z), (px - h, py + h, z))]
        ink.d.polygon(c, outline=RGB["warn"], width=6)
        ink.d.line((c[0], c[2]), fill=RGB["warn"], width=4)
        ink.d.line((c[1], c[3]), fill=RGB["warn"], width=4)


def step07():
    sc = Scene()
    add_installed(sc, lid=False)
    cam = Cam((24.0, 18.0, 12), az=0, el=90, ortho=58.0, H=840)
    A = render(sc, cam)
    ink = Ink(A, cam)
    keep_clear(ink)
    lx, ly = M.LFI
    r_ring = M.LFI_BORE / 2 + M.LFI_WALL
    a, b = ink.P((lx - r_ring, ly + r_ring, 25)), ink.P((lx + r_ring, ly - r_ring, 25))
    ink.d.ellipse((a[0], a[1], b[0], b[1]), outline=RGB["warn"], width=6)
    ink.label((M.SPRINGS["LF-front"][0] + 1.9, M.SPRINGS["LF-front"][1] - 1.9, 25),
              "red marks: the lid's posts\nand ring press here.\nKeep leads clear.", (1110, 760), RGB["warn"])
    ink.label((21.5, M.WIRE_SLOT_Y["MF"] - 1.2, 16), "comb", (150, 700), RGB["base"])
    ink.label((M.MF[0] + 3, M.MF[1] - 8, 11), "MF", (720, 790), RGB["MF"])
    ink.label((M.LF[0] + 3, M.LF[1] + 7, 11), "LF", (1320, 470), RGB["LF"])
    ink.label((lx + 3, ly + 3, 23), "LFi", (1300, 110), RGB["LFi"])
    for blk, hx, nm in (("M", 3.05, "MF"), ("L", 10.15, "LF"), ("R", 17.25, "LFi")):
        x, y = ink.P((hx, 21.3, 20))
        ink.text((x, y - 6), blk, size=40, col=RGB[nm], anchor="mb", box=True, pad=5)

    camB = Cam((12.5, 11.6, 14), az=0, el=90, ortho=13.0, H=560)
    B = render(sc, camB)
    ink = Ink(B, camB)
    for hole, hx in HOLE_X.items():
        col = RGB["red"] if hole.endswith("+") else INK
        x, y = ink.P((hx, HOLE_FACE_Y + 2.2, 20))
        ink.text((x, y), hole, size=32, col=col, anchor="mm", box=True, pad=4)
    for nm in ("MF", "LF", "LFi"):
        ink.label((M.WIRE_GUIDE_X1 - 0.2, M.WIRE_SLOT_Y[nm], M.WIRE_GUIDE_Z),
                  f"{nm} lane", (1290, {"MF": 470, "LF": 330, "LFi": 190}[nm]), RGB[nm])
    ink.label((17.2, 9.3, M.PCB_TOP + HOLE_REL_Z + 0.5), "U-loop: each lead goes\nstraight into its hole",
              (260, 470), INK)
    ink.text((560, 520), "Close-up from above", size=30, col=INK, box=True)
    notes = ["Run each pair down to its lane in the comb (front to back: MF, LF, LFi), through it, and across to "
             "its block, so each lead goes straight into its hole (a U-loop).",
             "Tuck spare lead flat over the motors, away from the red marks."]
    return save(page(7, "Route the leads through the comb", [A, B], notes), "step07.png")


# ---------------------------------------------------------------- step 8: lid
def step08():
    sc = Scene()
    add_installed(sc, lid=False)
    lift = 24.0
    sc.add(Part("lid").mesh(lid_mesh(False), COL["lid"][:3] + (0.5,)), t=(0, 0, lift), layer=1)
    camA = Cam((22, 17, 22), az=22, el=32, dist=192, fovy=30, H=660)
    A = render(sc, camA)
    ink = Ink(A, camA)
    for (x, y) in ((M.IX0 + 2, M.IY0 + 2), (M.IX1 - 2, M.IY0 + 2), (M.IX1 - 2, M.IY1 - 2), (M.IX0 + 2, M.IY1 - 2)):
        ink.arrow((x, y, M.INNER_H + lift + 12), (x, y, M.INNER_H + lift + 2.5), RGB["lid"], w=10, head=30)
    ink.label((M.IX0 + 10, M.IY0 + 4, M.INNER_H + lift + 1.2), "lid: press down\nuntil it clicks", (200, 140),
              RGB["lid"])
    ink.label((M.SPRINGS["MF-front"][0], M.SPRINGS["MF-front"][1], M.SPRING_FREE_Z + lift + 1), "4 spring posts",
              (1180, 380), RGB["lid"])
    ink.label((M.LFI[0] + 3, M.LFI[1] + 5, M.LFI_RING_Z0 + lift + 2), "ring over LFi", (1150, 110), RGB["lid"])

    lid_in = M.lid({**{n: LIFT_SPRING for n in M.SPRINGS}, "board": LIFT_BOARD})
    base = M.base()
    # B: cut along MF's axis (the two posts on MF's caps), seen from +X
    sc = Scene()
    x0, x1 = M.MF[0] - 12, M.MF[0]
    box = (x0, x1, -4, 31, -3, 29)
    add_cut(sc, "base", base, box, COL["base"], darker(COL["base"]), "x", x1, 1)
    add_cut(sc, "lid", lid_in, box, COL["lid_solid"], COL["lid_cut"], "x", x1, 1)
    body, ring = motor_solid("MF", lying_at=(M.MF[0], M.MOTOR_FRONT))
    add_cut(sc, "MF", body, box, COL["MF"], darker(COL["MF"], 0.7), "x", x1, 1)
    add_cut(sc, "MFring", ring, box, COL["ring"], COL["ring"], "x", x1, 1)
    camB = Cam((M.MF[0], 13.5, 12.6), az=90, el=8, ortho=31.0, H=580, Wd=W // 2)
    B = render(sc, camB)
    ink = Ink(B, camB)
    px, py = M.SPRINGS["MF-front"]
    ink.label((px, py - 1.0, ZC_LYING + CAP_R + 0.8), "post on\nthe cap", (110, 300), RGB["lid"])
    ink.label((M.MF[0], M.SPRINGS["MF-rear"][1] - 5.0, M.INNER_H + 0.6), "spring tongue", (470, 70), RGB["lid"])
    ink.label((M.MF[0], M.MF[1], 5), "MF", (560, 330), RGB["MF"])
    ink.text((16, 522), "MF, cut along its axis", size=30, col=INK, box=True)
    # C: cut through LFi, seen from -Y
    sc = Scene()
    lx, ly = M.LFI
    box = (lx - 13, lx + 13, ly, ly + 12, -3, 29)
    add_cut(sc, "base", base, box, COL["base"], darker(COL["base"]), "y", ly, -1)
    add_cut(sc, "lid", lid_in, box, COL["lid_solid"], COL["lid_cut"], "y", ly, -1)
    body, ring = motor_solid("LFi", standing_at=(lx, ly))
    add_cut(sc, "LFi", body, box, COL["LFi"], darker(COL["LFi"], 0.7), "y", ly, -1)
    add_cut(sc, "LFiring", ring, box, COL["ring"], COL["ring"], "y", ly, -1)
    camC = Cam((lx, ly, 12.6), az=0, el=8, ortho=31.0, H=580, Wd=W // 2)
    C = render(sc, camC)
    ink = Ink(C, camC)
    ink.label((lx - M.LFI_BORE / 2 - 0.6, ly, M.LFI_RING_Z0 + 2.5), "ring holds\nthe top", (130, 150), RGB["lid"])
    ink.label((lx - M.LFI_BORE / 2 - 0.6, ly, 2.3), "sleeve holds\nthe bottom", (130, 450), RGB["base"])
    ink.label((lx, ly, 11), "LFi", (560, 330), RGB["LFi"])
    ink.text((16, 522), "LFi, cut through its axis", size=30, col=INK, box=True)
    BC = Image.new("RGB", (W, 580), (255, 255, 255))
    BC.paste(B, (0, 0))
    BC.paste(C, (W // 2, 0))
    ImageDraw.Draw(BC).line((W // 2, 20, W // 2, 560), fill=(210, 210, 215), width=3)
    notes = ["Put the lid on (its USB notch over the USB opening) and press all round until it clicks. The four "
             "spring posts press the lying motors' end caps; the ring holds the top of LFi.",
             "If the lid will not close, a lead is under a post: open it and re-route."]
    return save(page(8, "Snap the lid on", [A, BC], notes), "step08.png")


# ---------------------------------------------------------------- step 9: cable, zip tie, strap
def step09():
    sc = Scene()
    add_installed(sc, lid=True, lid_alpha=0.55)
    ty0 = M.IY0 - M.WALL - M.TONGUE_LEN
    sc.add(usb_cable(-1.34 + 6.6, bend=[(M.USB_X, ty0 - 6, M.USB_Z), (M.USB_X - 3, ty0 - 22, M.USB_Z - 5),
                                        (M.USB_X - 12, ty0 - 32, M.USB_Z - 12)]))
    sc.add(zip_tie_installed())
    sc.add(strap_installed())
    cam = Cam((20, -8, 4), az=62, el=28, dist=215, fovy=30, H=860)
    img = render(sc, cam)
    ink = Ink(img, cam)
    ink.arrow((M.USB_X + 8, -30, M.USB_Z + 9), (M.USB_X + 8, -12, M.USB_Z + 9), RGB["act"], w=11)
    ink.label((M.USB_X + 6.2, -8, M.USB_Z + 3.3), "1  USB-C in", (560, 140), RGB["act"])
    ink.label((M.USB_X + 6.8, ty0 + 3.6, M.USB_Z + 2.5), "2  zip tie through the\ntunnel, over the cable",
              (300, 790), RGB["act"])
    ymid = (M.IY0 + M.IY1) / 2
    xs_l = M.IX0 - M.WALL - M.SLOT_LIGAMENT - M.STRAP_SLOT / 2
    xs_r = M.IX1 + M.WALL + M.SLOT_LIGAMENT + M.STRAP_SLOT / 2
    ink.arrow((xs_l - 5, ymid - 13, -24), (xs_l - 5, ymid - 13, -4), RGB["act"], w=10, head=28)
    ink.arrow((xs_r + 5, ymid - 13, -2), (xs_r + 5, ymid - 13, -22), RGB["act"], w=10, head=28)
    ink.label((xs_r + 1.0, ymid - 12.5, 16), "3  strap: up through one\nside slot, over the lid,\ndown the other",
              (1130, 130), RGB["act"])
    notes = ["Plug the USB-C cable in. Lay it along the cable tongue and zip-tie it through the tunnel at the end "
             "(2.5 mm tie at most), so a pull loads the tongue, not the socket.",
             "Thread the 25 mm velcro strap through both side slots and over the lid."]
    return save(page(9, "Cable, zip tie and strap", [img], notes), "step09.png")


# ---------------------------------------------------------------- overview (exploded)
def overview():
    sc = Scene()
    sc.add(Part("base").mesh(base_mesh(), COL["base"]))
    dz_m, dz_b, dz_l = 26.0, 56.0, 96.0
    for nm, (mx, _) in (("MF", M.MF), ("LF", M.LF)):
        R, t = motor_lying(mx, M.MOTOR_FRONT)
        sc.add(motor_part(nm), R, t + (0, 0, dz_m))
        lead_stubs(sc, R, t + (0, 0, dz_m), nm, rise=9, lean=(-3, 0, 0))
    R, t = lfi_standing(*M.LFI)
    sc.add(motor_part("LFi"), R, t + (0, 0, dz_m + 8))
    lead_stubs(sc, R, t + (0, 0, dz_m + 8), "LFi", rise=8, lean=(0, -3, 2))
    sc.add(board_part(), t=(0, 0, M.PCB_TOP + dz_b))
    jx, jy = JUMPER_PINS[0], (JUMPER_PINS[1] + JUMPER_PINS[2]) / 2
    sc.add(jumper_part(jx, jy, -4.4), t=(0, 0, M.PCB_TOP + dz_b - 9))
    sc.add(Part("lid").mesh(lid_mesh(False), COL["lid"][:3] + (0.55,)), t=(0, 0, dz_l), layer=1)
    ty0 = M.IY0 - M.WALL - M.TONGUE_LEN
    sc.add(usb_cable(-1.34 - 34, bend=[(M.USB_X, ty0 - 40, M.USB_Z), (M.USB_X - 6, ty0 - 58, M.USB_Z - 4)]))
    tie = Part("tie")
    tie.plank((36.0, -48, 0.5), (36.0, -8, 0.5), 2.5, 1.0, COL["tie"], across=(1, 0, 0))
    tie.box(34.0, 38.0, -8, -3.5, 0, 3.0, COL["tie"])
    sc.add(tie)
    strap = Part("strap")
    strap.plank((78, -30, 0.9), (78, 60, 0.9), 25.0, STRAP_T, COL["strap"], across=(1, 0, 0))
    sc.add(strap)
    cam = Cam((24, 4, 48), az=30, el=24, dist=335, fovy=30, H=1150)
    img = render(sc, cam)
    ink = Ink(img, cam)
    for p0, p1 in (((M.MF[0], M.MF[1], dz_m), (M.MF[0], M.MF[1], 12)), ((M.LF[0], M.LF[1], dz_m), (M.LF[0], M.LF[1], 12)),
                   ((M.LFI[0], M.LFI[1], dz_m + 8), (M.LFI[0], M.LFI[1], 6)),
                   ((10, 20, M.PCB_TOP + dz_b - 11), (10, 20, M.PCB_TOP + 1))):
        a, b = ink.P(p0), ink.P(p1)
        n = int(math.dist(a, b) / 18)
        for k in range(n):
            if k % 2 == 0:
                u0, u1 = k / n, (k + 1) / n
                ink.d.line((a[0] + (b[0] - a[0]) * u0, a[1] + (b[1] - a[1]) * u0,
                            a[0] + (b[0] - a[0]) * u1, a[1] + (b[1] - a[1]) * u1), fill=(150, 150, 160), width=3)
    ink.label((M.IX0 + 6, M.IY0 + 3, M.INNER_H + dz_l + 1.2), "lid", (220, 110), RGB["lid"])
    ink.label((3, 30, M.PCB_TOP + dz_b + 2.5), "TITAN Core", (170, 320), RGB["board"])
    ink.label((jx, jy - 2.5, M.PCB_TOP + dz_b - 9 - 10.4), "jumper", (1150, 470), INK)
    ink.label((M.MF[0], M.MF[1] - 9, dz_m + 10), "MF", (330, 700), RGB["MF"])
    ink.label((M.LF[0], M.LF[1] - 9, dz_m + 10), "LF", (1260, 640), RGB["LF"])
    ink.label((M.LFI[0], M.LFI[1], dz_m + 8 + L_MOT), "LFi", (1180, 330), RGB["LFi"])
    ink.label((30, M.IY0 - 1.2, 4), "base", (1000, 980), RGB["base"])
    ink.label((M.USB_X, -60, M.USB_Z), "USB-C cable", (300, 1090), INK)
    ink.label((36, -30, 1), "zip tie", (700, 1100), INK)
    ink.label((78, 20, 2), "velcro strap", (1230, 850), INK)
    notes = ["Everything that goes into one puck, pulled apart along the direction it goes in."]
    return save(page(None, "TITAN haptic puck: exploded view", [img], notes), "overview_exploded.png")


# ---------------------------------------------------------------- sheet
def sheet():
    names = ["overview_exploded.png"] + [f"step{k:02d}.png" for k in range(1, 10)]
    ims = [Image.open(os.path.join(HERE, n)).convert("RGB") for n in names]
    head = 150
    h = head + sum(i.height for i in ims) + 20 * len(ims)
    out = Image.new("RGB", (W, h), (255, 255, 255))
    d = ImageDraw.Draw(out)
    d.rectangle((0, 0, W, head - 20), fill=(35, 38, 48))
    d.text((40, 40), "TITAN haptic puck: assembly", font=font(56), fill=(255, 255, 255))
    d.text((42, 100), "Feel the Music, draft printed 2026-09-19. Steps 1 to 9, top to bottom.", font=font(30, False),
           fill=(215, 218, 228))
    y = head
    for im in ims:
        out.paste(im, (0, y))
        y += im.height
        d.rectangle((0, y + 6, W, y + 12), fill=(225, 226, 230))
        y += 20
    out = out.quantize(colors=192, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    return save(out, "sheet.png")


STEPS = {"overview": overview, "step01": step01, "step02": step02, "step03": step03, "step04": step04,
         "step05": step05, "step06": step06, "step07": step07, "step08": step08, "step09": step09, "sheet": sheet}

if __name__ == "__main__":
    try:
        for nm in sys.argv[1:] or list(STEPS):
            STEPS[nm]()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
