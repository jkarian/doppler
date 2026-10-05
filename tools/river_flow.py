"""Flow directions for the water in a vista scene: which way the river runs at every water pixel.

    python tools/river_flow.py scenes/canyon-vista

The water mask is the vista layer's albedo alpha (255 = not water; see vista_plate.py). At each water pixel the river's
course is the average of its two banks' directions: the banks' orientation (from the gradient of the distance to the
nearest bank) smoothed across the river's width, turned along the banks. That picture direction is carried onto the
ground (the scene's level plane, from the vista depth) and stored as an angle in the vista layer's normal alpha:
0..255 = 0..180 degrees in the ground frame (sideways, into the scene). The shader points the water's streaks that
way and runs them downstream toward the viewer. Run after vista_plate.py (which rewrites those files).
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def main() -> None:
    scene = Path(sys.argv[1])
    info = json.loads((scene / "scene.json").read_text())
    W, H = info["width"], info["height"]
    layers = info["background"] if isinstance(info["background"], list) else [info["background"]]
    vista = layers[-1]
    alb = np.asarray(Image.open(scene / vista["albedo"]).convert("RGBA"))
    water = 1 - alb[..., 3] / 255.0
    wet = water > 0.5
    if wet.sum() < 100:
        raise SystemExit("no water in this scene")
    z = np.fromfile(scene / vista["depth"], dtype="<f4").reshape(H, W).astype(np.float64)

    # Bank orientation: gradient of the distance to the nearest bank (perpendicular to the banks), as a structure
    # tensor smoothed over about the river's width, so both banks are averaged. The course is perpendicular to it.
    dist = ndi.distance_transform_edt(wet)
    width = max(4.0, 2 * np.median(dist[wet & (dist >= ndi.maximum_filter(dist, 5))]))
    gy, gx = np.gradient(ndi.gaussian_filter(dist, 2))
    sig = width * 1.5
    w = ndi.gaussian_filter(wet.astype(float), sig) + 1e-6
    jxx = ndi.gaussian_filter(gx * gx * wet, sig) / w
    jxy = ndi.gaussian_filter(gx * gy * wet, sig) / w
    jyy = ndi.gaussian_filter(gy * gy * wet, sig) / w
    across = 0.5 * np.arctan2(2 * jxy, jxx - jyy)  # picture angle of the banks' normal
    tx, ty = -np.sin(across), np.cos(across)       # along the banks (picture x right, y down)

    # Onto the ground: picture steps -> ground coordinates (sideways, into the scene) through the vista depth.
    up = np.array(info["up"], float)
    up /= np.linalg.norm(up)
    fwd = np.array([0.0, 0.0, 1.0]) - up * up[2]
    fwd /= np.linalg.norm(fwd)
    side = np.cross(up, fwd)
    t = np.tan(np.radians(info["fovDeg"]) / 2)
    ys, xs = np.mgrid[0:H, 0:W]
    zs = ndi.gaussian_filter(np.where(wet, z, 0), 3) / np.maximum(ndi.gaussian_filter(wet.astype(float), 3), 1e-6)
    zs = np.where(wet, zs, z)
    P = np.stack([((xs + 0.5) / W * 2 - 1) * t * (W / H) * zs, (1 - (ys + 0.5) / H * 2) * t * zs, zs], -1)
    gs, gf = P @ side, P @ fwd
    dsy, dsx = np.gradient(gs)
    dfy, dfx = np.gradient(gf)
    vs = dsx * tx + dsy * ty
    vf = dfx * tx + dfy * ty
    ang = np.mod(np.arctan2(vf, vs), np.pi)  # orientation only: the shader makes it run toward the viewer

    nrm_path = scene / vista["normal"]
    nrm = np.asarray(Image.open(nrm_path).convert("RGB"))
    a = np.where(wet | (water > 0.01), np.round(ang / np.pi * 255), 128).astype(np.uint8)
    Image.fromarray(np.dstack([nrm, a])).save(nrm_path)
    print(f"river flow: {wet.mean() * 100:.1f}% of the picture, about {width:.0f} px wide; stored in {nrm_path.name} alpha")


if __name__ == "__main__":
    main()
