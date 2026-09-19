"""Board keep-out envelope for the TITAN Core, from the vendor model, for fit checks without the vendor file.

A grid of columns (about 0.5 mm cells over the board outline). Each column spans the lowest to the highest point of
the real board inside that cell: PCB, parts on top, pin headers and spacers underneath. It is at least as big as the
board everywhere (a part that only covers a corner of a cell fills the whole cell), so zero overlap with it means zero
overlap with the real board. Frame: the vendor STEP's, PCB top at z = 0, X 0..20.32, Y -1.34..43.31 (USB-C end at -Y).

  uv run --python 3.12 --with trimesh --with numpy --with cadquery python gen_keepout.py <titan_core.stl>
writes titan_core_keepout.stl and titan_core_keepout.json (the height map and the boxes; check.py builds the
solid from the boxes).
"""
import json, sys
import numpy as np, trimesh

src = sys.argv[1] if len(sys.argv) > 1 else "../titan_core.stl"
m = trimesh.load(src)
(x0, y0, _), (x1, y1, _) = m.bounds
NX = 40
xs = np.linspace(x0, x1, NX + 1)
ys = np.concatenate([[y0], np.linspace(0.0, y1, 87)])    # one row for the USB-C overhang (no PCB there), then the PCB
NY = len(ys) - 1

# dense surface samples + every vertex; triangles are small (72k faces) so this finds every part's extent
pts = np.vstack([m.vertices, trimesh.sample.sample_surface(m, 1_500_000, seed=1)[0]])
ix = np.clip(np.searchsorted(xs, pts[:, 0], side="right") - 1, 0, NX - 1)
iy = np.clip(np.searchsorted(ys, pts[:, 1], side="right") - 1, 0, NY - 1)
zmax = np.full((NX, NY), -np.inf); zmin = np.full((NX, NY), np.inf)
np.maximum.at(zmax, (ix, iy), pts[:, 2]); np.minimum.at(zmin, (ix, iy), pts[:, 2])
# every cell over the PCB (y >= 0) holds at least the PCB (the outline is a full rectangle); the overhang row only
# what is really there (the USB-C receptacle)
pcb = ys[:-1] >= 0.0
empty = ~np.isfinite(zmax)
zmax[:, pcb] = np.where(empty[:, pcb], 0.0, np.maximum(zmax[:, pcb], 0.0))
zmin[:, pcb] = np.where(empty[:, pcb], -1.6, np.minimum(zmin[:, pcb], -1.6))
zmax = np.ceil(zmax * 100) / 100; zmin = np.floor(zmin * 100) / 100

# merge runs along X with the same span into boxes
boxes = []
for j in range(NY):
    i = 0
    while i < NX:
        if not np.isfinite(zmax[i, j]):
            i += 1; continue
        k = i
        while k + 1 < NX and zmax[k + 1, j] == zmax[i, j] and zmin[k + 1, j] == zmin[i, j]:
            k += 1
        boxes.append((xs[i], xs[k + 1], ys[j], ys[j + 1], zmin[i, j], zmax[i, j]))
        i = k + 1

meshes = [trimesh.creation.box(extents=(bx1 - bx0, by1 - by0, bz1 - bz0),
                               transform=trimesh.transformations.translation_matrix(((bx0 + bx1) / 2, (by0 + by1) / 2, (bz0 + bz1) / 2)))
          for bx0, bx1, by0, by1, bz0, bz1 in boxes]
trimesh.util.concatenate(meshes).export("titan_core_keepout.stl")
json.dump({"frame": "vendor STEP, PCB top z=0", "x_edges": xs.round(3).tolist(), "y_edges": ys.round(3).tolist(),
           "z_top": zmax.round(2).tolist(), "z_bottom": zmin.round(2).tolist(), "boxes": [list(map(lambda v: round(float(v), 3), b)) for b in boxes]},
          open("titan_core_keepout.json", "w"))
print(f"{len(boxes)} boxes; z top max {zmax.max():.2f}, z bottom min {zmin.min():.2f}")
