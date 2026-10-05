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
from scene_prep import SKY_MODEL, linear_to_srgb, sky_probability, srgb_to_linear  # noqa: E402

MODEL = "Ruicheng/moge-2-vitl-normal"
VISTA_MIN_M = 100.0  # pillars and spire reach to 80 m in the markup
VISTA_KNEE_M = 300.0  # plate depths below this are squeezed so the nearest painted rock lands at VISTA_MIN_M
KEPT_DIFF = 10.0  # mean abs difference (0-255, 5 px box) below which plate and photo are the same pixels
FRAME_RGB = (139, 0, 0)  # layers_from_markup.py: "frame (cave walls)"
PILLARS_RGB = (0, 80, 140)  # layers_from_markup.py: "pillars and spire"
BEHIND_PX = 120  # how far behind the cave frame the pillars carry on (beyond any camera move)
HAZE_VISIBILITY_M = 150_000.0  # how far the air lets you see: sets the haze on far land
EDGE_PX = 2  # cut-outs lose this much of their outline (scene pixels): the blended edge pixels made a halo
NORMAL_BAND_PX = 4  # and their normals come from this much further in (edge normals turn sideways: a lit rim)
MARKUP_HOLE_PX = 20_000  # unpainted spots in the cave markup smaller than this (scene pixels) count as cave
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
        # The cave is only what the markup paints as cave. Anything else outside the kept canyon and the pillars
        # (the gap between a loose first cut and the tight pillar cut-out, around the spire) is vista: the plate
        # painted it. As cave it floated at cave distance as a lit halo around the spire.
        m = np.asarray(Image.open(args.markup).convert("RGB").resize((W, H), Image.NEAREST)).astype(np.float64)
        cave = np.linalg.norm(m - np.array(FRAME_RGB, np.float64), axis=-1) < 60
        # Small spots the markup left unpainted inside the cave are cave too (as vista they were holes: flecks of
        # whatever lies behind, showing through the rock).
        lab, n = ndi.label(~cave)
        sizes = np.bincount(lab.ravel())
        cave |= np.isin(lab, np.nonzero(sizes < MARKUP_HOLE_PX)[0]) & (lab > 0)
        # Where the markup paints cave, it's cave even if the middle layer repeats the photo there (the painted wall
        # reuses real cave rock in places): as pillar those spots sat at pillar distance and slid as flecks.
        pillars &= ~cave
        vista |= ~cave & ~pillars
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
    # Sky: SegFormer on the plate, the one region of it that reaches the top. MoGe's own mask took the snow on the
    # far peaks for sky (pink holes in the mountains under the sky gradient).
    sky = sky_probability(Image.fromarray(plate), SKY_MODEL) > 0.5
    lab, n = ndi.label(sky)
    top = np.unique(lab[0][lab[0] > 0])
    sky = np.isin(lab, top) if len(top) else sky

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
    # Land MoGe has no depth for (bright snow it took for sky): the nearest land pixel's depth.
    hole = ~sky & ~valid
    if hole.any():
        _, (iy, ix) = ndi.distance_transform_edt(~land, return_indices=True)
        z_plate[hole] = z_plate[iy, ix][hole]
    print(f"  log depth stretch {a:.2f}; MoGe far/near {np.exp(np.quantile(np.log(zm[land]), 0.98) - np.quantile(np.log(zm[land]), 0.02)):.0f}x, "
          f"mapped {np.quantile(z_plate[land], 0.02):.0f}-{np.quantile(z_plate[land], 0.98):.0f} m")

    # 4. Layers.
    z_main = np.where(near, z_near, z_plate)
    n_plate[sky] = (128, 128, 255)
    n_main = np.where(near[..., None], normal_src, n_plate)

    lin = lambda c: srgb_to_linear(c.astype(np.float64) / 255.0)
    ref = kept & ~sky
    # The vista's rock colour without the photo's own haze: the renderer adds aerial haze itself (Sky visibility), so
    # baked-in haze made the far canyon washed out twice over next to the saturated near rock. Dehaze the plate with its
    # depth (airlight: the sky just above the land; extinction: the value that makes rock colour steadiest with distance),
    # then use the near rock's photo-to-albedo gain (the pillars: near, so barely hazed) for all of it.
    Lp = lin(plate)
    rows = np.nonzero(sky.any(1))[0]
    edge = sky & ~ndi.binary_erosion(sky, iterations=6)
    airlight = np.median(Lp[edge & (ndi.binary_dilation(land, iterations=12))], 0)
    far_land = land & (z_plate > 150)
    bins = np.quantile(np.log(z_plate[far_land]), np.linspace(0, 1, 9))
    which = np.digitize(np.log(z_plate[far_land]), bins[1:-1])
    lum = (Lp[far_land] * np.array([0.2126, 0.7152, 0.0722])).sum(-1)
    best = (np.inf, 0.0)
    for beta in np.geomspace(1e-5, 2e-3, 60):
        t = np.exp(-beta * z_plate[far_land])
        j = (lum - airlight @ np.array([0.2126, 0.7152, 0.0722]) * (1 - t)) / np.maximum(t, 0.15)
        meds = np.array([np.median(j[which == k]) for k in range(8)])
        if (meds <= 0).any():
            continue
        spread = np.std(np.log(meds))
        if spread < best[0]:
            best = (spread, beta)
    beta = best[1]
    t = np.exp(-beta * np.where(land, z_plate, 0))[..., None]
    plate_clear = np.clip((Lp - airlight * (1 - t)) / np.maximum(t, 0.15), 0, 1)
    pil = pillars if pillars.any() else ref
    # Brightness only (one number for all channels): the painting keeps its own hues (blue water, green brush); a
    # per-channel gain from the red pillars took the blue out of the river.
    lw = np.array([0.2126, 0.7152, 0.0722])
    gain = np.median(lin(albedo)[pil] @ lw) / max(np.median(lin(photo)[pil] @ lw), 1e-4)
    plate_albedo = np.round(linear_to_srgb(np.clip(plate_clear * gain, 0, 1)) * 255).astype(np.uint8)
    plate_albedo[sky] = np.round(linear_to_srgb(np.clip(Lp * gain, 0, 1)) * 255).astype(np.uint8)[sky]
    bg_albedo = plate_albedo.copy()
    print(f"  plate dehazed: airlight {np.round(airlight, 3)}, extinction {beta:.2e}/m (visibility {3.9 / beta / 1000:.0f} km in the photo)")

    rel = lambda name: os.path.normpath(Path("..") / args.scene.name / name).replace("\\", "/")
    layers = []  # (name, photo, albedo, normal, depth), front to back, behind the main layer
    if mid is not None:
        # 5. Middle layer depth: MoGe on the middle over the plate (it needs the surroundings to judge distance),
        #    fitted onto the visible pillars' distances.
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
        # A cut-out: see-through outside the outline (alpha), and everything else carried a little past it from the
        # nearest pixel inside, so filtering at the edge doesn't pick up other colours and the depth has no step there
        # (a step would make the renderer stretch the edge into spikes).
        A, ext, ext_n = cutout(M)
        mid_photo = np.dstack([ext(mid_rgb), A.astype(np.uint8) * 255])
        layers.append(("mid", mid_photo, ext(mid_albedo), ext_n(np.where(M[..., None], n_comp, n_plate)), ext(z_mid)))
        # Behind the cave where the middle layer doesn't reach, the plate holds whatever the painting put there (sky
        # where the cave ceiling was): uncovered by the camera it showed as pale bands along the rim. The cave carries
        # on behind itself instead, its own rock (softened) a little deeper.
        behind = frame & ~M
        soft = lambda a: cv2.GaussianBlur(a, (0, 0), 4)
        bg_photo, bg_alb, bg_nrm = plate.copy(), bg_albedo.copy(), n_plate.copy()
        for dst, img in ((bg_photo, photo), (bg_alb, albedo), (bg_nrm, normal_src)):
            dst[behind] = soft(img)[behind]
        z_bg = np.where(behind, z_near * 1.3, z_plate)
        print(f"  cave behind the cave (no middle layer there): {behind.mean() * 100:.1f}% of the picture")
        layers.append(("bg", bg_photo, bg_alb, bg_nrm, z_bg))
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

    main_files = {"image": rel(info["image"]), "albedo": rel(info["albedo"])}
    if mid is not None:
        # The main layer becomes a cut-out of the cave alone (the pillars are in the middle layer, the vista behind),
        # carried past its outline from the nearest cave pixel like the middle layer, so it has no depth step there.
        A, ext, ext_n = cutout(frame)
        z_mesh, n_main = ext(z_main), ext_n(n_main)
        main_files = {"image": "photo.png", "albedo": "albedo.png", "meshDepth": "mesh_depth.bin"}
        cave_photo = np.dstack([ext(photo), A.astype(np.uint8) * 255])
        cave_albedo = ext(albedo)

    args.out.mkdir(parents=True, exist_ok=True)
    if mid is not None:
        Image.fromarray(cave_photo).save(args.out / "photo.png")
        Image.fromarray(cave_albedo).save(args.out / "albedo.png")
        z_mesh.astype("<f4").tofile(args.out / "mesh_depth.bin")
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

    if info.get("metersPerUnit") == 1.0:
        # Clear desert air: about 150 km visibility (Koschmieder: extinction = 3.9 / visibility).
        info = {**info, "haze": {**info.get("haze", {}), "beta": 3.9 / HAZE_VISIBILITY_M}}
    scene = {**info, **main_files, "normal": "normal.png", "depth": "depth.bin", "graph": "vista",
             "depthModel": f"near rock from {args.scene.name}, vista: MoGe-2 on {args.plate.name}",
             "background": background if len(background) > 1 else background[0]}
    (args.out / "scene.json").write_text(json.dumps(scene, indent=2))
    print(f"wrote {args.out}")


def cutout(mask: np.ndarray):
    """A cut-out layer's outline and how to fill past it. The outline is pulled in EDGE_PX: the outermost pixels of a
    cut blend the rock with what was behind it and showed as a thin halo. Colour, albedo and depth carry on past the
    outline from the nearest pixel inside, so filtering at the edge picks up no other colour and the mesh has no depth
    step there (a step would stretch into spikes). Normals carry on from NORMAL_BAND_PX further in: at the very edge
    they turn sideways, and the lights caught them as a glowing rim."""
    A = ndi.binary_erosion(mask, iterations=EDGE_PX)
    _, (iy, ix) = ndi.distance_transform_edt(~A, return_indices=True)
    _, (ny, nx) = ndi.distance_transform_edt(~ndi.binary_erosion(A, iterations=NORMAL_BAND_PX), return_indices=True)
    return A, (lambda img: img[iy, ix]), (lambda img: img[ny, nx])


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
