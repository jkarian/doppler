"""Flow directions for the water in a vista scene: which way the river runs at every water pixel.

    python tools/river_flow.py scenes/canyon-vista [images/river_mask.png]

The water mask is the vista layer's albedo alpha (255 = not water; see vista_plate.py). At each water pixel the river's
course is the average of its two banks' directions: the banks' orientation (from the gradient of the distance to the
nearest bank) smoothed across the river's width, turned along the banks. That picture direction is carried onto the
ground (the scene's level plane, from the vista depth) and stored as an angle in the vista layer's normal alpha:
0..255 = 0..360 degrees in the ground frame (sideways, into the scene), pointing downstream: toward where the river
leaves the picture (the lowest-right end of the water), so it bends toward its exit. The water mask is first grown
through connected low-saturation blue-grey pixels (shaded and bright water both), then saved back. Run after vista_plate.py (which rewrites those files).
"""

import heapq
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi


NEAR_SEED_PX = 60  # dim blue-grey pixels count as water only this close to the river found so far (per step)
GROW_STEPS = 8


def main() -> None:
    scene = Path(sys.argv[1])
    info = json.loads((scene / "scene.json").read_text())
    W, H = info["width"], info["height"]
    layers = info["background"] if isinstance(info["background"], list) else [info["background"]]
    vista = layers[-1]
    alb = np.asarray(Image.open(scene / vista["albedo"]).convert("RGBA"))
    # Start from the segmentation model's water on the vista plate (not the stored mask: repeatable runs).
    sys.path.insert(0, str(Path(__file__).parent))
    from scene_prep import SKY_MODEL, class_probability
    water = class_probability(Image.open(scene / vista["image"]).convert("RGB"), SKY_MODEL, ["water", "river", "sea", "lake", "waterfall"])
    water = (water > 0.5).astype(np.float64)
    # Grow the water through connected pixels with the water's tint (low saturation, slightly blue: brown banks are
    # warm), shaded or bright, out in the vista (not sky).
    photo = np.asarray(Image.open(scene / vista["image"]).convert("RGB"))
    lab = cv2.cvtColor(photo, cv2.COLOR_RGB2LAB).astype(np.float64)
    zv = np.fromfile(scene / vista["depth"], dtype="<f4").reshape(H, W)
    # Bright white water may grow anywhere it connects (it's the river, sunlit or far up the gorge); dim blue-grey
    # pixels (shaded water, but also shaded rock) only close to the river already found. Heights from the depth were
    # tried and are too loose this far out (the far river came out 100 m below the near one).
    open_air = (zv > 100) & (zv < info["far"] * 0.9)
    seed = water > 0.5
    foam = (lab[..., 0] > 215) & (lab[..., 1] < 129) & (lab[..., 2] < 146) & open_air  # sunlit white water is a little warm
    dim_all = (lab[..., 1] < 131) & (lab[..., 2] < 133) & open_air
    # Grow in steps: each pass accepts dim water only near what is water so far, so the mask walks up and down the
    # river through dim stretches without jumping across to shaded rock further off.
    grown = seed
    for _ in range(GROW_STEPS):
        near = ndi.binary_dilation(grown, iterations=NEAR_SEED_PX)
        tint = ndi.binary_closing(foam | (dim_all & near) | grown, iterations=4)  # foam is speckled: close the gaps
        comp, _ = ndi.label(tint)
        keep = np.unique(comp[grown])
        new = np.isin(comp, keep[keep > 0])
        if new.sum() == grown.sum():
            break
        grown = new
    water = np.maximum(water, cv2.GaussianBlur(grown.astype(np.float32), (0, 0), 2))
    # A painted mask wins (white = water, the plate's framing, any size): images/river_mask.png or the second argument.
    painted = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("images/river_mask.png")
    if painted.exists():
        im = Image.open(painted)
        alpha = np.asarray(im.convert("RGBA").resize((W, H), Image.LANCZOS))[..., 3].astype(np.float64)
        rgb = np.asarray(im.convert("RGB").resize((W, H), Image.LANCZOS)).astype(np.float64)
        if alpha.min() < 128 and alpha.max() > 128:
            m = alpha / 255.0  # white strokes on transparency
        elif (rgb.std(-1) > 12).mean() > 0.05:
            # Painted over a picture (white strokes on the photo), not a black-and-white mask: water is where it's
            # near-pure white and differs from the scene's own photo.
            ref = np.asarray(Image.open(scene / info["image"]).convert("RGB")).astype(np.float64)
            white = (rgb.min(-1) > 235) & (rgb.max(-1) - rgb.min(-1) < 12)
            changed = np.abs(rgb - ref).mean(-1) > 18
            m = ndi.binary_closing(white & changed, iterations=2).astype(np.float64)
        else:
            m = rgb.mean(-1) / 255.0
        water = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.5).astype(np.float64)
        print(f"  water from the painted mask {painted}")
    Image.fromarray(np.dstack([alb[..., :3], np.round(255 * (1 - water)).astype(np.uint8)])).save(scene / vista["albedo"])
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
    ang = np.arctan2(vf, vs)

    # Downstream: the water heads for where the river leaves the picture (the lowest-right end of the water, behind the
    # near ground), along the river. Geodesic distance to that outlet inside the water (Dijkstra at half size); the
    # course points the way that distance falls, so the river bends toward its exit instead of running straight on.
    sm = wet[::2, ::2]
    hs, ws = sm.shape
    rows = np.nonzero(sm.any(1))[0]
    low = sm.copy()
    low[: rows.max() - max(3, (rows.max() - rows.min()) // 30)] = False
    oy, ox = np.nonzero(low)
    k = np.argmax(ox + oy)
    dist_out = np.full((hs, ws), np.inf)
    dist_out[oy[k], ox[k]] = 0.0
    heap = [(0.0, int(oy[k]), int(ox[k]))]
    steps = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
    while heap:
        d, y, x = heapq.heappop(heap)
        if d > dist_out[y, x]:
            continue
        for dy, dx, c in steps:
            yy, xx = y + dy, x + dx
            if 0 <= yy < hs and 0 <= xx < ws and sm[yy, xx] and d + c < dist_out[yy, xx]:
                dist_out[yy, xx] = d + c
                heapq.heappush(heap, (d + c, yy, xx))
    reach = np.isfinite(dist_out)
    dfull = np.repeat(np.repeat(np.where(reach, dist_out, dist_out[reach].max()), 2, 0), 2, 1)[:H, :W] * 2
    dfull = ndi.gaussian_filter(dfull, width * 0.5)
    dgy, dgx = np.gradient(dfull)
    # Downstream in the picture is down the distance to the outlet: give the banks' course that sign.
    sign = np.where(-(dgx * tx + dgy * ty) < 0, -1.0, 1.0)
    ang = np.where(sign < 0, ang + np.pi, ang)
    # Near the outlet the banks stop (the river goes out of sight): follow the way to the outlet there.
    near = np.clip(1 - dfull / (width * 6), 0, 1)
    ds_ = -(dsx * dgx + dsy * dgy)
    df_ = -(dfx * dgx + dfy * dgy)
    ang_out = np.arctan2(df_, ds_)
    ang = np.arctan2((1 - near) * np.sin(ang) + near * np.sin(ang_out), (1 - near) * np.cos(ang) + near * np.cos(ang_out))
    ang = np.mod(ang, 2 * np.pi)  # full direction, downstream

    nrm_path = scene / vista["normal"]
    nrm = np.asarray(Image.open(nrm_path).convert("RGB"))
    a = np.where(wet | (water > 0.01), np.round(ang / (2 * np.pi) * 255) % 256, 128).astype(np.uint8)
    Image.fromarray(np.dstack([nrm, a])).save(nrm_path)
    print(f"river flow: {wet.mean() * 100:.1f}% of the picture, about {width:.0f} px wide; stored in {nrm_path.name} alpha")


if __name__ == "__main__":
    main()
