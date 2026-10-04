"""Reduce a heavy generated mesh (Tripo's are ~2M triangles) with MeshLab's quadric edge collapse, keeping its texture
coordinates so the texture still maps. (Maya's polyReduce refuses generated meshes' non-manifold bits.)

    tools/.venv/Scripts/python tools/world/reduce_mesh.py 3d/tripo/model.obj 200000

Writes <name>_<N>k.obj with its .mtl and texture next to the input.
"""

import sys
from pathlib import Path

import pymeshlab

src = Path(sys.argv[1]).resolve()
target = int(sys.argv[2]) if len(sys.argv) > 2 else 200000
ms = pymeshlab.MeshSet()
ms.load_new_mesh(str(src))
n0 = ms.current_mesh().face_number()
ms.meshing_remove_duplicate_vertices()
ms.meshing_remove_null_faces()
ms.meshing_repair_non_manifold_edges()
ms.meshing_decimation_quadric_edge_collapse_with_texture(targetfacenum=target, qualitythr=0.5, preserveboundary=True,
                                                          preservenormal=True, planarquadric=True)
n1 = ms.current_mesh().face_number()
out = src.with_name(f"{src.stem}_{round(target / 1000)}k.obj")
ms.save_current_mesh(str(out), save_textures=True, save_wedge_texcoord=True)
print(f"{n0} -> {n1} triangles; wrote {out}")
