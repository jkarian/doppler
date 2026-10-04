"""Export only the matched camera (and its image plane) from a Maya scene made by export_maya.py. Runs under mayapy.

    "C:/Program Files/Autodesk/Maya2027/bin/mayapy.exe" tools/world/export_camera.py 3d/tripo/tripo_matched.ma tripo_cam1

Writes <scene>_cam.abc (camera only) and <scene>_cam.ma (camera + image plane) next to the scene.
"""

import sys
from pathlib import Path

import maya.standalone

maya.standalone.initialize(name="python")
import maya.cmds as cmds  # noqa: E402

repo = Path(__file__).resolve().parents[2]
scene, cam = repo / sys.argv[1], sys.argv[2]
cmds.loadPlugin("AbcExport", quiet=True)
cmds.file(str(scene), open=True, force=True)
out = scene.with_name(scene.stem + "_cam")
# The camera may sit under a levelling group: work on a world-space copy at the top level, so the file stands alone.
cam = cmds.parent(cmds.duplicate(cam, name=cam + "_export", returnRootsOnly=True)[0], world=True)[0] if cmds.listRelatives(cam, parent=True) else cam
cmds.AbcExport(j=f"-frameRange 1 1 -worldSpace -dataFormat ogawa -root |{cam} -file {out.with_suffix('.abc').as_posix()}")
shape = cmds.listRelatives(cam, shapes=True, type="camera")[0]
planes = cmds.listConnections(f"{shape}.imagePlane", source=True) or []
cmds.select([cam] + planes)
cmds.file(str(out.with_suffix(".ma")), exportSelected=True, type="mayaAscii", force=True, constructionHistory=False)
print(f"wrote {out.with_suffix('.abc')}\nwrote {out.with_suffix('.ma')}")
maya.standalone.uninitialize()
