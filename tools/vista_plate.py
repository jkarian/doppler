"""Matte-painting vista: a painted plate behind the photo's near layers (cave frame, pillars, spire).

    python tools/vista_plate.py scenes/canyon-layers images/canyon_midground.png scenes/canyon-vista

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
BEHIND_PX = 200  # how far behind the cave frame the pillars carry on (beyond any camera move)
FRINGE_PX = 10  # Photoshop selection expand into the kept canyon (in scene pixels)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", type=Path, help="source scene (true scale: canyon-layers)")
    ap.add_argument("plate", type=Path)
    ap.add_argument("out", type=Path)
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
    # The fill overlapped the kept canyon by the selection expand: those photo pixels are still canyon unless
    # the markup calls them near rock.
    fringe = ndi.binary_dilation(kept, iterations=FRINGE_PX) & ~kept & (z_src >= VISTA_MIN_M)
    vista = kept | fringe
    print(f"kept canyon {kept.mean() * 100:.1f}% of the picture (+{fringe.mean() * 100:.1f}% fringe)")

    # 2. Plate depth.
    print(f"MoGe-2 on the plate at {W}x{H} ...")
    model = load_moge()
    with torch.no_grad():
        out = model.infer(torch.tensor(plate / 255.0, dtype=torch.float32, device="cuda").permute(2, 0, 1),
                          resolution_level=args.resolution_level, use_fp16=True)
    zm = out["points"][..., 2].float().cpu().numpy().astype(np.float64)
    valid = out["mask"].cpu().numpy() & np.isfinite(zm) & (zm > 0)
    nm = out["normal"].float().cpu().numpy()
    del model, out
    torch.cuda.empty_cache()
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
    near = ~vista
    z_main = np.where(near, z_src, z_plate)
    n_plate = np.round((np.stack([nm[..., 0], -nm[..., 1], -nm[..., 2]], -1) * 0.5 + 0.5) * 255)  # MoGe y down -> standard colours
    n_plate[sky] = (128, 128, 255)
    n_plate = n_plate.clip(0, 255).astype(np.uint8)
    n_main = np.where(near[..., None], normal_src, n_plate)

    lin = lambda c: srgb_to_linear(c.astype(np.float64) / 255.0)
    ref = kept & ~sky
    gain = np.median(lin(albedo)[ref], 0) / np.maximum(np.median(lin(photo)[ref], 0), 1e-4)
    plate_albedo = np.round(linear_to_srgb(np.clip(lin(plate) * gain, 0, 1)) * 255).astype(np.uint8)
    bg_albedo = np.where(kept[..., None], albedo, plate_albedo)
    print(f"  plate albedo gain {np.round(gain, 3)}")

    # Near rock behind near rock (cave wall over a pillar, the spire against the right wall): what's hidden
    # there is more rock, not vista. The markup says which near rock is the cave frame and which the pillars:
    # behind the frame, each pixel takes the nearest pillar-or-vista pixel; where that is a pillar, the pillar
    # carries on (its colour, normal and depth, softened like background_layer's quick fill).
    bg_photo, bg_normal = plate.copy(), n_plate.copy()
    if args.markup:
        m = np.asarray(Image.open(args.markup).convert("RGB").resize((W, H), Image.NEAREST)).astype(np.float64)
        pillars = (np.linalg.norm(m - np.array(PILLARS_RGB, np.float64), axis=-1) < 60) & near
        frame = near & ~pillars
        _, (iy, ix) = ndi.distance_transform_edt(frame, return_indices=True)
        dist = ndi.distance_transform_edt(frame)
        rock = frame & pillars[iy, ix] & (dist < BEHIND_PX)
        soft = lambda a: cv2.GaussianBlur(a, (0, 0), 3)
        for dst, img in ((bg_photo, photo), (bg_albedo, albedo), (bg_normal, normal_src)):
            dst[rock] = soft(img[iy, ix])[rock]
        z_plate = np.where(rock, np.maximum(z_src[iy, ix], z_src * 1.15), z_plate)
        print(f"  pillars behind the cave frame: {rock.mean() * 100:.2f}% of the picture")

    args.out.mkdir(parents=True, exist_ok=True)
    z_main.astype("<f4").tofile(args.out / "depth.bin")
    z_plate.astype("<f4").tofile(args.out / "bg_depth.bin")
    Image.fromarray(n_main).save(args.out / "normal.png")
    Image.fromarray(bg_normal).save(args.out / "bg_normal.png")
    Image.fromarray(bg_photo).save(args.out / "bg_photo.png")
    Image.fromarray(bg_albedo).save(args.out / "bg_albedo.png")
    Image.fromarray(near.astype(np.uint8) * 255).save(args.out / "bg_mask.png")
    Image.fromarray((kept.astype(np.uint8) * 255)).save(args.out / "kept.png")

    rel = lambda name: os.path.normpath(Path("..") / args.scene.name / name).replace("\\", "/")
    scene = {**info, "image": rel(info["image"]),
             "albedo": rel(info["albedo"]), "normal": "normal.png", "depth": "depth.bin",
             "depthModel": f"near rock from {args.scene.name}, vista: MoGe-2 on {args.plate.name}",
             "background": {"depth": "bg_depth.bin", "image": "bg_photo.png", "albedo": "bg_albedo.png",
                            "normal": "bg_normal.png", "mask": "bg_mask.png", "fill": f"plate {args.plate.name}"}}
    (args.out / "scene.json").write_text(json.dumps(scene, indent=2))
    print(f"wrote {args.out}")


def load_moge():
    from moge.model.v2 import MoGeModel
    return MoGeModel.from_pretrained(MODEL).cuda().eval()


if __name__ == "__main__":
    main()
