"""Put validated enclosure STLs in print orientation on a millimeter 3MF plate."""

from pathlib import Path

import trimesh
from trimesh.exchange.threemf import export_3MF

from check import validate_mesh

base, lid = trimesh.load_mesh("puck_base.stl"), trimesh.load_mesh("puck_lid_print.stl")
validate_mesh("print base", base)
validate_mesh("print lid", lid)
base.apply_translation(-base.bounds[0])
lid.apply_translation(-lid.bounds[0] + [base.extents[0] + 6.0, 0, 0])
base.units = lid.units = "mm"
scene = trimesh.Scene({"puck_base": base, "puck_lid": lid})
Path("puck_plate.3mf").write_bytes(export_3MF(scene))
print(
    f"puck_plate.3mf: plate {base.extents[0] + 6 + lid.extents[0]:.0f} x {max(base.extents[1], lid.extents[1]):.0f} mm"
)
