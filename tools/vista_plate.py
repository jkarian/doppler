"""Matte-painting vista: a painted plate behind the photo's near layers (cave frame, pillars, spire).

    python tools/vista_plate.py scenes/canyon-layers images/canyon_midground.png scenes/canyon-vista \
        [--middle images/canyon_middle.png]

The plate is the photo with everything beyond the near rock repainted as an open vista (Photoshop: the real
canyon through the opening kept, the rest filled). It must have the photo's framing (any size, same aspect).

    1. Kept vs painted: where the plate equals the photo is the real canyon the user kept; where it differs,
       the near rock was painted over. That split decides which pixels are near (photo, the scene's depth) and
       which are vista (the plate's depth).
    2. Plate depth: MoGe-2 on the plate alone. Without the near cave in the picture, the depth model spends its
       range on the canyon, so the far gorge recedes instead of flattening into a card (tested on a crop of
       the photo: iso-depth lines follow the walls and step back along the river).
    3. Real distances: MoGe's log depth mapped linearly (scale and stretch) onto the source scene's depth over
       the kept canyon, so the markup's distances hold (near canyon ~0.3 km, far ~3 km, mountains ~30 km) and
       the painted parts extrapolate from that. Vista stays behind the pillars (at least VISTA_MIN_M).
    4. Main layer: the photo, near rock at the source depth, vista at the plate depth, normals to match.
       Background layer: the whole plate at its own depth, so near rock sliding aside uncovers painted vista.
       Albedo: the photo's de-lit albedo where kept; the plate times the photo's albedo/photo gain elsewhere.
    5. Middle layer (optional, --middle): the pillars and spire cut out of the photo with what the cave walls hide
       of them painted in (PNG with transparency). It becomes a layer of its own between the cave and the vista,
       so the cave sliding aside uncovers pillar, and the pillars sliding aside uncover vista. Its depth: MoGe-2 on
       the middle over the plate, mapped onto the source's pillar distances where the pillars are visible.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
from scene_prep import linear_to_srgb, save_normals, srgb_to_linear  # noqa: E402

MODEL = "Ruicheng/moge-2-vitl-normal"
VISTA_MIN_M = 100.0  # pillars and spire reach to 80 m in the markup
VISTA_KNEE_M = 300.0  # plate depths below this are squeezed so the nearest painted rock lands at VISTA_MIN_M
KEPT_DIFF = 10.0  # mean abs difference (0-255, 5 px box) below which plate and photo are the same pixels
PILLARS_RGB = (0, 80, 140)  # layers_from_markup.py: "pillars and spire"
BEHIND_PX = 120  # how far behind the cave frame the pillars carry on (beyond any camera move)
KEPT_ERODE_PX = 3  # the kept strip loses this much (scene pixels) to the near rock: its rim moves with it


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", type=Path, help="source scene (true scale: canyon-layers)")
    ap.add_argument("plate", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--middle", type=Path, help="middle layer: pillars and spire with their hidden parts painted (PNG with transparency)")
    ap.add_argument("--pillars", type=Path, help="cut-out of the pillars and spire (PNG with transparency, the photo's framing)")
    ap.add_argument("--markup", type=Path, default=Path("images/markup_layers.webp"), help="layer markup (frame vs pillars)")
    ap.add_argument("--resolution-level", type=int, default=9)
    args = ap.parse_args()

    info = json.loads((args.scene / "scene.json").read_text())
    W, H, far = info["width"], info["height"], info["far"]
    src = lambda name: args.scene / name
    z_src = np.fromfile(src(info["depth"]), dtype="<f4").reshape(H, W).astype(np.float64)
    sky_src = z_src >= far * 0.98
    photo = np.asarray(Image.open(src(info["image"])).convert("RGB"))
    albedo = np.asarray(Image.open(src(info["albedo"])).convert("RGB"))
    normal_src = np.asarray(Image.open(src(info["normal"])).convert("RGB"))
    plate = np.asarray(Image.open(args.plate).convert("RGB").resize((W, H), Image.LANCZOS))

    # 1. Kept (real canyon) vs painted.
    diff = ndi.uniform_filter(np.abs(photo.astype(np.float32) - plate.astype(np.float32)).mean(-1), 5)
    kept = ndi.binary_opening(diff < KEPT_DIFF, iterations=2)
    kept = ndi.binary_fill_holes(kept)
    lab, n = ndi.label(kept)
    kept = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1) if n else kept
    # The kept strip carries a few pixels of the near rock's dark rim (the cut-out isn't exact). As vista they'd stay
    # behind as a dark outline when the rock slides; shrink it so the rim moves with the rock.
    vista = ndi.binary_erosion(kept, iterations=KEPT_ERODE_PX)
    # Pillars and spire: the user's cut-out (alpha) when given, else the coarse markup.
    mid = mid_rgb = M = None
    if args.middle:
        mid = np.asarray(Image.open(args.middle).convert("RGBA").resize((W, H), Image.LANCZOS))
        mid_rgb, M = mid[..., :3], mid[..., 3] > 127
        # Visible pillar: where the middle layer is the photo's own pixels; the rest of it is painted (hidden).
        same = ndi.uniform_filter(np.abs(photo.astype(np.float32) - mid_rgb.astype(np.float32)).mean(-1), 5) < KEPT_DIFF
        pillars = ndi.binary_opening(M & same, iterations=2)
        vista &= ~pillars
    elif args.pillars:
        a = np.asarray(Image.open(args.pillars).convert("RGBA").resize((W, H), Image.LANCZOS))[..., 3]
        pillars = a > 127
        vista &= ~pillars
    else:
        m = np.asarray(Image.open(args.markup).convert("RGB").resize((W, H), Image.NEAREST)).astype(np.float64)
        pillars = (np.linalg.norm(m - np.array(PILLARS_RGB, np.float64), axis=-1) < 60) & ~vista
    near = ~vista
    print(f"kept canyon {kept.mean() * 100:.1f}% of the picture, vista {vista.mean() * 100:.1f}%, pillars {pillars.mean() * 100:.1f}%")

    # Near depth: the source's, except where it disagrees with the split (rim pixels the source put far away,
    # pixels the cut-out calls pillar but the markup didn't): those take the nearest trustworthy pixel's depth.
    z_near = z_src.copy()
    frame = near & ~pillars
    for part, lo, hi in ((pillars, 20.0, VISTA_MIN_M), (frame, 0.0, VISTA_MIN_M)):
        good = part & (z_src >= lo) & (z_src < hi)
        if good.any():
            _, (iy, ix) = ndi.distance_transform_edt(~good, return_indices=True)
            bad = part & ~good
            z_near[bad] = z_src[iy, ix][bad]

    # 2. Plate depth.
    print(f"MoGe-2 on the plate at {W}x{H} ...")
    zm, valid, n_plate = moge(plate, args.resolution_level)
    sky = ~valid
    sky[kept] = sky_src[kept]  # trust the source's sky where the pixels are the same

    # 3. Real distances: log-linear fit over the kept land.
    fit = kept & ~sky & valid & ~sky_src
    q = np.linspace(0.02, 0.98, 97)
    a, b = np.polyfit(np.quantile(np.log(zm[fit]), q), np.quantile(np.log(z_src[fit]), q), 1)
    z_plate = np.full((H, W), far)
    land = ~sky & valid
    L = a * np.log(zm[land]) + b
    # The painted near walls extrapolate closer than the kept canyon reaches, often in front of the pillars.
    # Below the kept canyon's nearest, squeeze log depth so the plate's nearest lands at VISTA_MIN_M: still
    # ordered and shaped (no flat clamp), just shallower.
    L0, Lmin = np.log(VISTA_KNEE_M), np.quantile(L, 0.005)
    if Lmin < np.log(VISTA_MIN_M):
        lo = L < L0
        L[lo] = L0 - (L0 - L[lo]) * (L0 - np.log(VISTA_MIN_M)) / (L0 - Lmin)
    z_plate[land] = np.clip(np.exp(L), VISTA_MIN_M, far * 0.95)
    print(f"  log depth stretch {a:.2f}; MoGe far/near {np.exp(np.quantile(np.log(zm[land]), 0.98) - np.quantile(np.log(zm[land]), 0.02)):.0f}x, "
          f"mapped {np.quantile(z_plate[land], 0.02):.0f}-{np.quantile(z_plate[land], 0.98):.0f} m")

    # 4. Layers.
    z_main = np.where(near, z_near, z_plate)
    n_plate[sky] = (128, 128, 255)
    n_main = np.where(near[..., None], normal_src, n_plate)

    lin = lambda c: srgb_to_linear(c.astype(np.float64) / 255.0)
    ref = kept & ~sky
    gain = np.median(lin(albedo)[ref], 0) / np.maximum(np.median(lin(photo)[ref], 0), 1e-4)
    plate_albedo = np.round(linear_to_srgb(np.clip(lin(plate) * gain, 0, 1)) * 255).astype(np.uint8)
    bg_albedo = np.where(kept[..., None], albedo, plate_albedo)
    print(f"  plate albedo gain {np.round(gain, 3)}")

    layers = []  # (name, photo, albedo, normal, depth), front to back, behind the main layer
    if mid is not None:
        # 5. Middle layer depth: MoGe on the middle over the plate, fitted onto the visible pillars' distances.
        comp = np.where(M[..., None], mid_rgb, plate)
        print("MoGe-2 on the middle layer over the plate ...")
        zc, valid_c, n_comp = moge(comp, args.resolution_level)
        fitp = pillars & valid_c & (z_src >= 20) & (z_src < VISTA_MIN_M)
        a2, b2 = np.polyfit(np.quantile(np.log(zc[fitp]), q), np.quantile(np.log(z_src[fitp]), q), 1)
        z_mid = np.clip(np.exp(a2 * np.log(np.where(valid_c, zc, 1.0)) + b2), 20.0, VISTA_MIN_M * 0.95)
        z_mid = np.where(frame, np.maximum(z_mid, z_near * 1.15), z_mid)  # stays behind the cave where it's hidden
        z_main = np.where(pillars, z_mid, z_main)  # visible pillars: the same depth in both layers
        gain_p = np.median(lin(albedo)[pillars], 0) / np.maximum(np.median(lin(photo)[pillars], 0), 1e-4)
        mid_albedo = np.round(linear_to_srgb(np.clip(lin(mid_rgb) * gain_p, 0, 1)) * 255).astype(np.uint8)
        mid_albedo = np.where(pillars[..., None], albedo, mid_albedo)
        print(f"  middle: log depth stretch {a2:.2f}, {np.quantile(z_mid[M], 0.02):.0f}-{np.quantile(z_mid[M], 0.98):.0f} m; "
              f"{M.mean() * 100:.1f}% of the picture ({(M & ~pillars).mean() * 100:.1f}% painted)")
        layers.append(("mid", comp, np.where(M[..., None], mid_albedo, bg_albedo), np.where(M[..., None], n_comp, n_plate),
                       np.where(M, z_mid, z_plate)))
        layers.append(("bg", plate, bg_albedo, n_plate, z_plate))
    else:
        # Near rock behind near rock (cave wall over a pillar, the spire against the right wall): what's hidden
        # there is more rock, not vista. Behind the frame, each pixel takes the nearest pillar-or-vista pixel; where
        # that is a pillar, the pillar carries on (its colour, normal and depth, softened like background_layer's fill).
        bg_photo, bg_normal = plate.copy(), n_plate.copy()
        if pillars.any():
            _, (iy, ix) = ndi.distance_transform_edt(frame, return_indices=True)
            dist = ndi.distance_transform_edt(frame)
            rock = frame & pillars[iy, ix] & (dist < BEHIND_PX)
            soft = lambda a: cv2.GaussianBlur(a, (0, 0), 3)
            for dst, img in ((bg_photo, photo), (bg_albedo, albedo), (bg_normal, normal_src)):
                dst[rock] = soft(img[iy, ix])[rock]
            z_plate = np.where(rock, np.maximum(z_near[iy, ix], z_near * 1.15), z_plate)
            print(f"  pillars behind the cave frame: {rock.mean() * 100:.2f}% of the picture")
        layers.append(("bg", bg_photo, bg_albedo, bg_normal, z_plate))

    args.out.mkdir(parents=True, exist_ok=True)
    for old in list(args.out.glob("mid_*")) + list(args.out.glob("bg_*")):
        old.unlink()
    z_main.astype("<f4").tofile(args.out / "depth.bin")
    Image.fromarray(n_main).save(args.out / "normal.png")
    Image.fromarray((kept.astype(np.uint8) * 255)).save(args.out / "kept.png")
    background = []
    for name, rgb, alb, nrm, z in layers:
        z.astype("<f4").tofile(args.out / f"{name}_depth.bin")
        Image.fromarray(rgb).save(args.out / f"{name}_photo.png")
        Image.fromarray(alb).save(args.out / f"{name}_albedo.png")
        Image.fromarray(nrm).save(args.out / f"{name}_normal.png")
        background.append({"depth": f"{name}_depth.bin", "image": f"{name}_photo.png", "albedo": f"{name}_albedo.png", "normal": f"{name}_normal.png"})

    rel = lambda name: os.path.normpath(Path("..") / args.scene.name / name).replace("\\", "/")
    scene = {**info, "image": rel(info["image"]),
             "albedo": rel(info["albedo"]), "normal": "normal.png", "depth": "depth.bin",
             "depthModel": f"near rock from {args.scene.name}, vista: MoGe-2 on {args.plate.name}",
             "background": background if len(background) > 1 else background[0]}
    (args.out / "scene.json").write_text(json.dumps(scene, indent=2))
    print(f"wrote {args.out}")


def moge(img: np.ndarray, resolution_level: int):
    """MoGe-2 on an image: planar depth, valid mask, normals as standard-colour PNG pixels."""
    from moge.model.v2 import MoGeModel

    model = MoGeModel.from_pretrained(MODEL).cuda().eval()
    with torch.no_grad():
        out = model.infer(torch.tensor(img / 255.0, dtype=torch.float32, device="cuda").permute(2, 0, 1),
                          resolution_level=resolution_level, use_fp16=True)
    z = out["points"][..., 2].float().cpu().numpy().astype(np.float64)
    valid = out["mask"].cpu().numpy() & np.isfinite(z) & (z > 0)
    nm = out["normal"].float().cpu().numpy()
    del model, out
    torch.cuda.empty_cache()
    n = np.round((np.stack([nm[..., 0], -nm[..., 1], -nm[..., 2]], -1) * 0.5 + 0.5) * 255)  # MoGe y down -> standard colours
    return z, valid, n.clip(0, 255).astype(np.uint8)


if __name__ == "__main__":
    main()
