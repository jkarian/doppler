"""Maya side of camera_map.py export (runs under mayapy): builds the projection scene from <folder>/manifest.json and
the OBJs, writes <name>.abc (meshes + cameras, world space) and <name>.ma (with the projections hooked up).

    "C:/Program Files/Autodesk/Maya2027/bin/mayapy.exe" tools/camera_map_maya.py scenes/ice-cave/maya

    projCam     the photo's camera at its base position, never moves. Its film gate covers the whole painted
                canvas (the photo plus its border), so each layer's image projects 1:1 onto its mesh.
    renderCam   the photo's own lens and framing, with the sway animated; on frame 1 it sits exactly on projCam.
    <layer>_geo one mesh per layer (and a sky card), real size in cm. Shader: the layer's image through a
                perspective projection node linked to projCam, its alpha as transparency. The meshes carry the
                same mapping as UVs, for packages that read the .abc without the projection.
"""

import json
import sys
from pathlib import Path

import maya.standalone

maya.standalone.initialize(name="python")
import maya.cmds as cmds  # noqa: E402

folder = Path(sys.argv[1])
m = json.loads((folder / "manifest.json").read_text())
name = m["name"].replace("-", "_")
W, H = m["width"], m["height"]
Wc, Hc = m["canvas"]

cmds.file(new=True, force=True)
for plugin in ("objExport", "AbcExport"):
    cmds.loadPlugin(plugin, quiet=True)
frames = len(m["camera"]["x"])
cmds.currentUnit(linear="cm", time={24: "film", 25: "pal", 30: "ntsc"}.get(m["fps"], "film"))
cmds.playbackOptions(minTime=1, maxTime=frames, animationStartTime=1, animationEndTime=frames)

# Everything is built in the photo camera's space, under one group turned so that true vertical is +Y.
root = cmds.group(empty=True, name=f"{name}_levelled")
R = m["level"]
cmds.xform(root, matrix=[R[0][0], R[1][0], R[2][0], 0, R[0][1], R[1][1], R[2][1], 0, R[0][2], R[1][2], R[2][2], 0, 0, 0, 0, 1])

# Cameras. Film back: 24 mm tall for the photo's frame; focal length from the photo's field of view.
vfa = 0.945  # inches
focal = vfa * 25.4 / (2 * m["tanHalfFov"])


def camera(cam_name, w, h):
    cam, shape = cmds.camera()
    cam = cmds.rename(cmds.parent(cam, root)[0], cam_name)
    shape = cmds.listRelatives(cam, shapes=True)[0]
    cmds.setAttr(f"{shape}.verticalFilmAperture", vfa * h / H)
    cmds.setAttr(f"{shape}.horizontalFilmAperture", vfa * w / H)
    cmds.setAttr(f"{shape}.filmFit", 2)  # vertical
    cmds.setAttr(f"{shape}.focalLength", focal)
    cmds.setAttr(f"{shape}.nearClipPlane", 1)
    cmds.setAttr(f"{shape}.farClipPlane", 1e7)
    return cam, shape


proj_cam, proj_shape = camera("projCam", Wc, Hc)
# The projection node fits the image to the gate horizontally whatever the camera's fit says (measured: vertical fit
# misplaces the projection by up to ~65 px at the frame's top). The gate has the canvas's own proportions, so
# horizontal fit frames projCam identically.
cmds.setAttr(f"{proj_shape}.filmFit", 1)
for attr in ("translate", "rotate", "scale"):
    cmds.setAttr(f"{proj_cam}.{attr}", lock=True)
render_cam, render_shape = camera("renderCam", W, H)
for i, (x, y) in enumerate(zip(m["camera"]["x"], m["camera"]["y"])):
    cmds.setKeyframe(render_cam, attribute="translateX", time=i + 1, value=x)
    cmds.setKeyframe(render_cam, attribute="translateY", time=i + 1, value=y)
cmds.setAttr("defaultResolution.width", W)
cmds.setAttr("defaultResolution.height", H)
cmds.setAttr("defaultResolution.deviceAspectRatio", W / H)
cmds.setAttr(f"{render_shape}.renderable", True)
cmds.setAttr("perspShape.renderable", False)

geos = []
for layer in m["layers"]:
    before = set(cmds.ls(assemblies=True))
    cmds.file(str(folder / layer["obj"]), i=True, type="OBJ", ignoreVersion=True, options="mo=0")
    new = [n for n in cmds.ls(assemblies=True) if n not in before]
    geo = cmds.rename(new[0], f"{layer['name']}_geo")
    cmds.parent(geo, root)
    cmds.polySoftEdge(geo, angle=180, constructionHistory=False)
    geos.append(geo)
    shape = cmds.listRelatives(geo, shapes=True)[0]
    if not cmds.attributeQuery("aiOpaque", node=shape, exists=True):
        cmds.addAttr(shape, longName="aiOpaque", attributeType="bool", defaultValue=True)
    cmds.setAttr(f"{shape}.aiOpaque", False)  # Arnold: honour the projected alpha (cut-out layers)
    tex = cmds.shadingNode("file", asTexture=True, isColorManaged=True, name=f"{layer['name']}_image")
    cmds.setAttr(f"{tex}.fileTextureName", str((folder / layer["texture"]).resolve()), type="string")
    cmds.setAttr(f"{tex}.filterType", 0)  # off: the image 1:1, no blur
    proj = cmds.shadingNode("projection", asUtility=True, name=f"{layer['name']}_projection")
    place = cmds.shadingNode("place3dTexture", asUtility=True, name=f"{layer['name']}_place3d")
    cmds.connectAttr(f"{place}.worldInverseMatrix[0]", f"{proj}.placementMatrix")
    cmds.setAttr(f"{proj}.projType", 8)  # perspective
    cmds.connectAttr(f"{proj_shape}.message", f"{proj}.linkedCamera")
    cmds.setAttr(f"{proj}.fitType", 1)  # match the camera's film gate
    cmds.connectAttr(f"{tex}.outColor", f"{proj}.image")
    cmds.connectAttr(f"{tex}.outTransparency", f"{proj}.transparency")
    shader = cmds.shadingNode("surfaceShader", asShader=True, name=f"{layer['name']}_projected")
    cmds.connectAttr(f"{proj}.outColor", f"{shader}.outColor")
    if layer["name"] != "sky":
        cmds.connectAttr(f"{proj}.outTransparency", f"{shader}.outTransparency")
    sg = cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name=f"{shader}SG")
    cmds.connectAttr(f"{shader}.outColor", f"{sg}.surfaceShader")
    cmds.sets(geo, edit=True, forceElement=sg)

stem = m.get("file", name)
abc = folder / f"{stem}.abc"
cmds.AbcExport(j=f"-frameRange 1 {frames} -uvWrite -worldSpace -writeVisibility -dataFormat ogawa -root |{root} -file {abc.as_posix()}")
ma = folder / f"{stem}.ma"
cmds.file(rename=str(ma))
cmds.file(save=True, type="mayaAscii", force=True)
print(f"wrote {abc}\nwrote {ma}")
maya.standalone.uninitialize()
