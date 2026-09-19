"""Check current enclosure geometry and make safe assembly meshes for rendering.

Run with the same CadQuery/trimesh environment as model.py.
MOTOR_D configures the enclosure in model.py.
CHECK_MOTOR_D / CHECK_MOTOR_L independently select the hardware envelope checked.
CHECK_WAIST_D / CHECK_WAIST_START / CHECK_WAIST_LEN select the recessed waist profile.
Without those overrides, the checked dimensions come from the current model.
TITAN_STEP optionally selects a private vendor board for additional validation.
The exported asm_board.stl always comes from our distributable keep-out envelope.
"""
import json
import math
import os
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import cadquery as cq
import numpy as np
import trimesh
from OCP.BRepExtrema import BRepExtrema_DistShapeShape

import model as M

INTERFERENCE_LIMIT = 0.01  # mm3; includes the approximately 0.004 mm3 USB corner touch
MIN_MOTOR_GAP = 3.0       # mm, the mechanical brief's conservative separation
MIN_WALL = 1.6            # mm, structural wall/floor/roof
HERE = Path(__file__).resolve().parent


class CheckFailure(ValueError):
    """A geometry or fit condition was not proved safe by this check."""


def shape(part):
    return part.val() if isinstance(part, cq.Workplane) else part


def validate_solid(name, part, single=False):
    solid = shape(part)
    volume = solid.Volume()
    if not solid.isValid() or not math.isfinite(volume) or volume <= 0:
        raise CheckFailure(f"{name}: invalid or empty CAD solid")
    count = len(solid.Solids())
    if not count or (single and count != 1):
        raise CheckFailure(f"{name}: expected {'one connected solid' if single else 'solids'}, found {count}")
    return solid


def vol(a, b):
    # Failed booleans must never look like a clean, zero-volume intersection.
    intersection = shape(a).intersect(shape(b))
    value = intersection.Volume()
    if not intersection.isValid() or not math.isfinite(value) or value < 0:
        raise CheckFailure("invalid intersection result")
    return value


def interference(name, a, b):
    value = vol(a, b)
    print(f"  {name}: {value:.6f} mm3")
    if value > INTERFERENCE_LIMIT:
        raise CheckFailure(f"{name}: {value:.6f} mm3 exceeds {INTERFERENCE_LIMIT:.2f} mm3")
    return value


def mind(a, b):
    distance = BRepExtrema_DistShapeShape(shape(a).wrapped, shape(b).wrapped)
    distance.Perform()
    if not distance.IsDone():
        raise CheckFailure("minimum-distance calculation failed")
    value = distance.Value()
    if not math.isfinite(value) or value < 0:
        raise CheckFailure("invalid minimum-distance result")
    return value


def clearance(name, a, b, minimum=0.0):
    value = mind(a, b)
    print(f"  {name}: {value:.3f} mm (minimum {minimum:.3f})")
    if value + 1e-6 < minimum:
        raise CheckFailure(f"{name}: {value:.3f} mm is below {minimum:.3f} mm")
    return value


def motor_dimensions(environ):
    diameter = float(environ.get("CHECK_MOTOR_D", M.MOTOR_D))
    length = float(environ.get("CHECK_MOTOR_L", M.MOTOR_L))
    waist_diameter = float(environ.get("CHECK_WAIST_D", M.WAIST_D))
    waist_start = float(environ.get("CHECK_WAIST_START", M.WAIST_START))
    waist_length = float(environ.get("CHECK_WAIST_LEN", M.WAIST_LEN))
    for name, value in (("diameter", diameter), ("length", length), ("waist diameter", waist_diameter),
                        ("waist start", waist_start), ("waist length", waist_length)):
        if not math.isfinite(value) or value <= 0:
            raise CheckFailure(f"motor {name} must be finite and positive; got {value}")
    if waist_diameter >= diameter:
        raise CheckFailure("waist diameter must be smaller than the end-cap diameter")
    if waist_start + waist_length >= length:
        raise CheckFailure("waist must end before the far motor end cap")
    return diameter, length, waist_diameter, waist_start, waist_length


def motor_profile(diameter, length, waist_diameter, waist_start, waist_length):
    """One stepped profile along +Z, shared by every motor orientation."""
    cylinder = cq.Workplane("XY").circle(diameter / 2).extrude(length)
    recess = (cq.Workplane("XY").circle(diameter / 2 + 1).circle(waist_diameter / 2)
              .extrude(waist_length).translate((0, 0, waist_start)))
    return cylinder.cut(recess)


def motor_shapes(diameter, length, waist_diameter, waist_start, waist_length):
    profile = motor_profile(diameter, length, waist_diameter, waist_start, waist_length)
    lying = profile.rotate((0, 0, 0), (1, 0, 0), -90)
    motors = {
        name: lying.translate((mx, my, M.MOTOR_Z))
        for name, (mx, my) in (("LF", M.LF), ("MF", M.MF))
    }
    motors["LFi"] = profile.translate((*M.LFI, M.LFI_Z))
    return motors


def axial_capture(base, motors):
    """Sample shoulder blocking in both directions at the nominal permitted lift."""
    print(f"== sampled axial capture: +/-1 mm travel at {M.RETAINER_GAP:.3f} mm upward lift")
    overlaps = {}
    for name in ("LF", "MF"):
        for travel in (-1.0, 1.0):
            moved = motors[name].translate((0, travel, M.RETAINER_GAP))
            overlap = vol(base, moved)
            print(f"  {name}, Y {travel:+.1f} mm: blocking overlap {overlap:.6f} mm3")
            if overlap <= INTERFERENCE_LIMIT:
                raise CheckFailure(f"{name}: no shoulder/collar blocking at Y {travel:+.1f} mm and lift {M.RETAINER_GAP:.3f} mm")
            overlaps[name, travel] = overlap
    print("  Static sampled configurations only: not a motion-path, strength, or dynamic-retention proof; measure the real profile.")
    return overlaps


def keepout_board():
    with (HERE / "titan_core_keepout.json").open() as source:
        boxes = json.load(source)["boxes"]
    board = cq.Compound.makeCompound([
        cq.Solid.makeBox(x1 - x0, y1 - y0, z1 - z0, cq.Vector(x0, y0, z0))
        for x0, x1, y0, y1, z0, z1 in boxes
    ])
    return board.translate((0, 0, M.PCB_TOP))


def validate_mesh(name, mesh):
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise CheckFailure(f"{name}: missing triangle mesh")
    if not np.isfinite(mesh.vertices).all() or not mesh.is_watertight or not mesh.is_volume:
        raise CheckFailure(f"{name}: mesh is not a finite, watertight, outward-facing volume")
    components = trimesh.graph.connected_components(mesh.face_adjacency, nodes=np.arange(len(mesh.faces)), engine="networkx")
    if len(components) != 1:
        raise CheckFailure(f"{name}: expected one connected mesh, found {len(components)}")


def printability(name, part, directory):
    # Re-tessellate this run's solid; old delivery STLs cannot make this check pass.
    path = Path(directory) / f"{name}.stl"
    cq.exporters.export(part, str(path), tolerance=0.02, angularTolerance=0.1)
    mesh = trimesh.load_mesh(path)
    validate_mesh(name, mesh)
    zmin = mesh.bounds[0, 2]
    normals, centers, areas = mesh.face_normals, mesh.triangles_center, mesh.area_faces
    down = (normals[:, 2] < -np.cos(np.radians(45))) & (centers[:, 2] > zmin + 0.3)
    bed = areas[(normals[:, 2] < -0.999) & (centers[:, 2] < zmin + 0.05)].sum()
    if bed <= 0:
        raise CheckFailure(f"{name}: no planar bed contact in the print orientation")
    print(f"  {name}: watertight, one body, {mesh.volume / 1000:.2f} cm3; "
          f"size {mesh.extents.round(2)} mm; bed contact {bed:.1f} mm2; "
          f"down-facing >45deg above bed {areas[down].sum():.1f} mm2")
    return mesh


def wall_checks(base, lid):
    for name, value in (("floor", M.FLOOR), ("lid roof", M.LID), ("lid lip", M.LIP_T),
                        ("motor cradle wall", M.SADDLE_WALL), ("side wall at groove", M.WALL - M.GROOVE_DEPTH)):
        if value < MIN_WALL:
            raise CheckFailure(f"{name}: {value} mm is below {MIN_WALL} mm")
    # Representative material probes complement nominal parameters; this is not a global thickness solver.
    probes = (
        ("floor", base, cq.Solid.makeBox(2, 2, MIN_WALL, cq.Vector(9, 6, -M.FLOOR))),
        ("lid roof", lid, cq.Solid.makeBox(2, 2, MIN_WALL, cq.Vector(9, 37, M.INNER_H))),
        ("side wall at snap groove", base, cq.Solid.makeBox(MIN_WALL, 2, 0.4,
            cq.Vector(M.IX0 - M.WALL, (M.IY0 + M.IY1) / 2 - 1, M.INNER_H - 2.3))),
    )
    for name, part, probe in probes:
        missing = probe.Volume() - vol(part, probe)
        if missing > INTERFERENCE_LIMIT:
            raise CheckFailure(f"{name}: {missing:.6f} mm3 missing from the {MIN_WALL} mm wall probe")
    print(f"  nominal structural walls and three material probes >= {MIN_WALL} mm; "
          "global minimum wall thickness requires slicer review")


def run():
    dimensions = motor_dimensions(os.environ)
    diameter, length, waist_diameter, waist_start, waist_length = dimensions
    print(f"enclosure MOTOR_D={M.MOTOR_D:.2f} mm; motors checked as {diameter:.2f} mm x {length:.2f} mm")
    print(f"checked waist: diameter {waist_diameter:.2f} mm, start {waist_start:.2f} mm, length {waist_length:.2f} mm")
    base, lid = M.base(), M.lid()
    keepout = keepout_board()
    motors = motor_shapes(*dimensions)
    plug = cq.Workplane("XY").box(12.5, 20, 6.5).translate((M.USB_X, -1.34 - 0.2 - 10, M.PCB_TOP + 1.6))
    for name, part in (("base", base), ("lid", lid)):
        validate_solid(name, part, single=True)
    for name, part in (("keep-out board", keepout), ("USB plug", plug), *motors.items()):
        validate_solid(name, part)
    boards = {"keep-out board": keepout}
    if os.environ.get("TITAN_STEP"):
        private_board = cq.importers.importStep(os.environ["TITAN_STEP"]).translate((0, 0, M.PCB_TOP))
        validate_solid("private vendor board", private_board)
        boards["private vendor board"] = private_board

    print(f"== interference volumes: pass at <= {INTERFERENCE_LIMIT:.2f} mm3")
    for pname, part in (("base", base), ("lid", lid)):
        for name, obj in (*boards.items(), *motors.items(), ("USB plug overmould", plug)):
            interference(f"{name} x {pname}", obj, part)
    interference("lid x base (closed)", lid, base)
    for bname, board in boards.items():
        for name, motor in motors.items():
            interference(f"{bname} x {name}", board, motor)
    for name, motor in motors.items():
        interference(f"USB plug overmould x {name}", plug, motor)
    for (aname, a), (bname, b) in combinations(motors.items(), 2):
        interference(f"{aname} x {bname}", a, b)

    print("== minimum clearances")
    for (aname, a), (bname, b) in combinations(motors.items(), 2):
        clearance(f"{aname} to {bname}", a, b, MIN_MOTOR_GAP)
    for name, motor in motors.items():
        clearance(f"{name} to board", motor, keepout)
        clearance(f"{name} to lid", motor, lid)
    # Use the sleeve's full outside cylinder, so the wire slot cannot inflate the end clearance.
    sleeve = cq.Workplane("XY").circle(M.CRADLE_D / 2 + M.SADDLE_WALL).extrude(M.LFI_SLEEVE_H).translate((*M.LFI, M.LFI_Z))
    for name, (_, y) in (("LF", M.LF), ("MF", M.MF)):
        for end, gap in (("front", y - M.IY0), ("rear", M.IY1 - y - length)):
            if gap < 2.0:
                raise CheckFailure(f"{name} {end} end to inner wall: {gap:.3f} mm is below 2 mm")
            print(f"  {name} {end} end to inner wall: {gap:.3f} mm")
        clearance(f"{name} to LFi sleeve outside envelope", motors[name], sleeve, 2.0)
    top_gap = M.INNER_H - M.LFI_Z - length
    if top_gap < 2.0:
        raise CheckFailure(f"LFi top end: {top_gap:.3f} mm is below 2 mm")
    print(f"  LFi top end to lid underside: {top_gap:.3f} mm")
    print("  The LFi lower end contacts the floor; end protection/foam and motor retention require physical checks.")
    axial_capture(base, motors)

    print("== structural and fresh-mesh checks")
    wall_checks(base, lid)
    with TemporaryDirectory(prefix="titan-puck-check-") as directory:
        printability("puck_base", base, directory)
        printability("puck_lid_print", lid.rotate((0, 0, 0), (1, 0, 0), 180), directory)
    print("  Down-facing area is diagnostic only: it does not prove bridge length or support-free printability.")

    # Only distributable derived envelopes leave this script, including when TITAN_STEP is supplied.
    cq.exporters.export(keepout, "asm_board.stl", tolerance=0.1)
    for name, motor in motors.items():
        cq.exporters.export(motor, f"asm_{name}.stl", tolerance=0.05)
    cq.exporters.export(lid, "asm_lid.stl", tolerance=0.05)
    print("PASS: geometry and fit checks passed; slicer and physical checks remain necessary.")


def main():
    try:
        run()
    except Exception as error:
        print(f"FAIL: {type(error).__name__}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
