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
# Optional 4th argument: scale to real size, e.g. feet per model unit (then 1 Maya unit = 1 foot). The mesh and the
# camera's position scale together, so the view through the camera is unchanged.
S = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
# Optional 5th argument: output name (default <name>_matched).
stem = sys.argv[5] if len(sys.argv) > 5 else f"{name}_matched"
# Optional 6th argument "level": turn mesh and camera together about the camera so the scene's true vertical (scene.json
# "up", known from the photo) is Maya's +Y. A generated model is built as if the camera were level; the real camera
# looks down (~23 degrees here), so without this the canyon's ground is tilted in Maya. The view is unchanged.
level = len(sys.argv) > 6 and sys.argv[6] == "level"
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
pos = [p * S for p in pos]
if S != 1.0:
    cmds.scale(S, S, S, group, pivot=(0, 0, 0), absolute=True)
    cmds.makeIdentity(group, apply=True, scale=True)

cam_t, cam_s = cmds.camera(name=f"{name}_cam")
# Maya's xform matrix is row-vector form: rows are the local x, y, z axes in world space, then the position.
cmds.xform(cam_t, worldSpace=True, matrix=[M[0][0], M[1][0], M[2][0], 0, M[0][1], M[1][1], M[2][1], 0, M[0][2], M[1][2], M[2][2], 0, *pos, 1])
h_mm, v_mm = 36.0, 36.0 * info["height"] / info["width"]
focal = (v_mm / 2) / math.tan(math.radians(cam["fov_y_deg"]) / 2)
cmds.setAttr(f"{cam_s}.horizontalFilmAperture", h_mm / 25.4)
cmds.setAttr(f"{cam_s}.verticalFilmAperture", v_mm / 25.4)
cmds.setAttr(f"{cam_s}.focalLength", focal)
cmds.setAttr(f"{cam_s}.filmFit", 2)  # vertical
cmds.setAttr(f"{cam_s}.nearClipPlane", 0.001 * S)
cmds.setAttr(f"{cam_s}.farClipPlane", 1000 * S)

roots = [f"|{group}", f"|{cam_t}"]
if level:
    # True up in the photo's camera: scene space is x right, y up, z forward; OpenCV's y points down.
    us = info.get("up", [0, 1, 0])
    up_cv = [us[0], -us[1], us[2]]
    a = [sum(Rt[r][k] * up_cv[k] for k in range(3)) for r in range(3)]  # into the model's world
    n = math.sqrt(sum(x * x for x in a))
    a = [x / n for x in a]
    # Smallest rotation taking a onto +Y (Rodrigues).
    v = [-a[2], 0.0, a[0]]  # a x (0, 1, 0)
    c = a[1]
    vx = [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]
    vx2 = [[sum(vx[r][k] * vx[k][s] for k in range(3)) for s in range(3)] for r in range(3)]
    f = 1 / (1 + c)
    Rl = [[(1 if r == s else 0) + vx[r][s] + vx2[r][s] * f for s in range(3)] for r in range(3)]
    # Euler angles for rotate order xyz (R = Rz Ry Rx).
    ry = math.degrees(math.asin(max(-1, min(1, -Rl[2][0]))))
    rx = math.degrees(math.atan2(Rl[2][1], Rl[2][2]))
    rz = math.degrees(math.atan2(Rl[1][0], Rl[0][0]))
    lev = cmds.group(empty=True, name=f"{name}_level")
    cmds.parent(group, lev)
    cmds.parent(cam_t, lev)
    cmds.xform(lev, worldSpace=True, rotatePivot=pos, scalePivot=pos)
    cmds.xform(lev, rotation=(rx, ry, rz))
    roots = [f"|{lev}"]
    print(f"levelled: true vertical was {math.degrees(math.acos(max(-1, min(1, c)))):.1f} degrees off the model's Y")

out = mesh.parent
abc = out / f"{stem}.abc"
cmds.AbcExport(j=f"-frameRange 1 1 -uvWrite -worldSpace -writeVisibility -dataFormat ogawa {' '.join('-root ' + r for r in roots)} -file {abc.as_posix()}")

photo = (scene / info["image"]).resolve()
ip = cmds.imagePlane(camera=cam_s, fileName=photo.as_posix())
cmds.setAttr(f"{ip[1]}.depth", 50 * S)
ma = out / f"{stem}.ma"
cmds.file(rename=str(ma))
cmds.file(save=True, type="mayaAscii")
print(f"camera at {[round(p, 4) for p in pos]}, focal {focal:.2f} mm on 36 x {v_mm:.2f} mm")
print(f"wrote {abc}\nwrote {ma}")
maya.standalone.uninitialize()
