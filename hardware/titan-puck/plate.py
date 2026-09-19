"""Put puck_base.stl and puck_lid_print.stl side by side on one plate: puck_plate.3mf (open it in Bambu Studio)."""
import trimesh
from trimesh.exchange.threemf import export_3MF

base, lid = trimesh.load("puck_base.stl"), trimesh.load("puck_lid_print.stl")
base.apply_translation(-base.bounds[0])
lid.apply_translation(-lid.bounds[0] + [base.extents[0] + 6.0, 0, 0])
scene = trimesh.Scene({"puck_base": base, "puck_lid": lid})
open("puck_plate.3mf", "wb").write(export_3MF(scene))
print(f"puck_plate.3mf: plate {base.extents[0] + 6 + lid.extents[0]:.0f} x {max(base.extents[1], lid.extents[1]):.0f} mm")
