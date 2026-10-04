"""Export a mesh aligned to a scene's photo (align_mesh.py) for Maya: the mesh plus the matched camera, as Alembic, and a
Maya scene with the photo as the camera's image plane (Alembic can't carry image planes).

    "C:/Program Files/Autodesk/Maya2027/bin/mayapy.exe" tools/world/export_maya.py scenes/canyon 3d/tripo/model.obj tripo

Writes <mesh folder>/<name>_matched.abc and <name>_matched.ma. Units are the mesh's own (Maya reads them as cm).
The camera: world-to-camera from <scene>/world/<name>/camera.json (OpenCV axes) turned into Maya's (looks down -Z,
Y up); vertical field of view matched exactly with a 36 x 20.25 mm film back (16:9) and vertical film fit.
"""

import json
import math
import sys
from pathlib import Path

import maya.standalone

maya.standalone.initialize(name="python")
import maya.cmds as cmds  # noqa: E402

repo = Path(__file__).resolve().parents[2]
scene, mesh, name = repo / sys.argv[1], repo / sys.argv[2], sys.argv[3]
info = json.loads((scene / "scene.json").read_text())
cam = json.loads((scene / "world" / name / "camera.json").read_text())
V = cam["world_to_camera"]

# Camera to world: invert the rigid world-to-camera.
R = [[V[r][c] for c in range(3)] for r in range(3)]
t = [V[r][3] for r in range(3)]
Rt = [[R[c][r] for c in range(3)] for r in range(3)]  # inverse rotation
pos = [-sum(Rt[r][k] * t[k] for k in range(3)) for r in range(3)]
# OpenCV camera axes (x right, y down, z forward) -> Maya's (x right, y up, looking down -z): flip y and z.
M = [[Rt[r][0], -Rt[r][1], -Rt[r][2]] for r in range(3)]

cmds.file(new=True, force=True)
for plugin in ("objExport", "AbcExport"):
    cmds.loadPlugin(plugin, quiet=True)
before = set(cmds.ls(assemblies=True))
cmds.file(str(mesh), i=True, type="OBJ", ignoreVersion=True, options="mo=1", namespace="geo")
geo = [n for n in cmds.ls(assemblies=True) if n not in before]
group = cmds.group(geo, name=f"{name}_geo")

cam_t, cam_s = cmds.camera(name=f"{name}_cam")
# Maya's xform matrix is row-vector form: rows are the local x, y, z axes in world space, then the position.
cmds.xform(cam_t, worldSpace=True, matrix=[M[0][0], M[1][0], M[2][0], 0, M[0][1], M[1][1], M[2][1], 0, M[0][2], M[1][2], M[2][2], 0, *pos, 1])
h_mm, v_mm = 36.0, 36.0 * info["height"] / info["width"]
focal = (v_mm / 2) / math.tan(math.radians(cam["fov_y_deg"]) / 2)
cmds.setAttr(f"{cam_s}.horizontalFilmAperture", h_mm / 25.4)
cmds.setAttr(f"{cam_s}.verticalFilmAperture", v_mm / 25.4)
cmds.setAttr(f"{cam_s}.focalLength", focal)
cmds.setAttr(f"{cam_s}.filmFit", 2)  # vertical
cmds.setAttr(f"{cam_s}.nearClipPlane", 0.001)
cmds.setAttr(f"{cam_s}.farClipPlane", 1000)

out = mesh.parent
abc = out / f"{name}_matched.abc"
cmds.AbcExport(j=f"-frameRange 1 1 -uvWrite -worldSpace -writeVisibility -dataFormat ogawa -root |{group} -root |{cam_t} -file {abc.as_posix()}")

photo = (scene / info["image"]).resolve()
ip = cmds.imagePlane(camera=cam_s, fileName=photo.as_posix())
cmds.setAttr(f"{ip[1]}.depth", 50)
ma = out / f"{name}_matched.ma"
cmds.file(rename=str(ma))
cmds.file(save=True, type="mayaAscii")
print(f"camera at {[round(p, 4) for p in pos]}, focal {focal:.2f} mm on 36 x {v_mm:.2f} mm")
print(f"wrote {abc}\nwrote {ma}")
maya.standalone.uninitialize()
