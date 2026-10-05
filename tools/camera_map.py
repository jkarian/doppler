"""Camera mapping: any single image -> projection layers (foreground, middle, background...) for a matte-painting
camera projection. Each layer is a cut-out image, painted where nearer layers hide it and past the frame's edges,
with real-size geometry of its own; a projection camera lines them all up.

    python tools/camera_map.py measure images/ice_cave.webp scenes/ice-cave
        Depth, normals, lens and sky from the image (MoGe-2, metric). Writes the scene and dimensions.png:
        the picture with a few distances and sizes on it for a person to check.
    python tools/camera_map.py scale scenes/ice-cave --set opening=6
        Confirm or correct the dimensions: one known size (any name from dimensions.json, in metres) rescales
        the whole scene. Or --factor 1.5. Run it again any time; it always starts from MoGe's own estimate.
    parts.json (written from the picture by the LLM): the parts, a few points on each, near-to-far order, the
        layer each belongs to, the layer that takes everything unclaimed ("rest") and a prompt per layer. Then
        tools/segment_parts.py traces each part's outline with SAM 2.
    python tools/camera_map.py layers scenes/ice-cave
        Parts -> layer masks (layers/<name>_mask.png; layers_overlay.png to check). Stray pixels go to the nearby
        layer whose depth matches.
    python tools/camera_map.py paint scenes/ice-cave
        Each layer as a cut-out image on a canvas 12% bigger each side: its own pixels kept exactly, painted
        (FLUX.1 Fill) behind the nearer layers and past the frame's edges; depth for it (MoGe-2 on the painted
        layer, fitted onto the scene's depth where the layer shows, and always behind the layers in front).
    python tools/camera_map.py export scenes/ice-cave
        For Maya: one real-size mesh (cm) per layer plus a sky card; projCam at the photo's position projecting
        each layer's image onto its mesh (projection nodes; matching UVs on the meshes too); renderCam = projCam
        plus a small animated sway (at rest on frame 1); levelled (Y = true up).
        Writes <scene>/maya/vNNN/<name>_vNNN.abc (meshes + cameras) and .ma (with the projections hooked up): a new
        version every time, never overwriting an earlier one.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
from scene_prep import estimate_up, save_normals  # noqa: E402

MOGE = "Ruicheng/moge-2-vitl-normal"
# FLUX.1 Fill, from the model library. The Hub cache copy is WSL symlinks (Windows can't follow them), so this
# folder hard-links the same files (no extra disk).
FLUX_FILL = r"C:\AI_Models\huggingface\FLUX.1-Fill-dev"
MAYAPY = r"C:\Program Files\Autodesk\Maya2027\bin\mayapy.exe"
SKY_M = 10000.0  # sky depth, metres
HF_HUB = r"C:\AI_Models\huggingface\hub"  # the model library (HF_HOME)


# --- measure -----------------------------------------------------------------------------------------------

def moge(photo: np.ndarray, fov_x: float | None = None):
    """MoGe-2 (metric): planar depth in metres, sky mask, normals (MoGe axes) and the vertical focal length in px.
    fov_x (degrees) fixes the lens instead of estimating it."""
    import torch
    from moge.model.v2 import MoGeModel

    H, W = photo.shape[:2]
    print(f"MoGe-2 on {W}x{H} ...")
    model = MoGeModel.from_pretrained(MOGE).cuda().eval()
    with torch.no_grad():
        o = model.infer(torch.tensor(photo / 255.0, dtype=torch.float32, device="cuda").permute(2, 0, 1), resolution_level=9, use_fp16=True, fov_x=fov_x)
    z = o["points"][..., 2].float().cpu().numpy().astype(np.float64)
    sky = ~(o["mask"].cpu().numpy() & np.isfinite(z) & (z > 0))
    normal = o["normal"].float().cpu().numpy()
    fy = float(o["intrinsics"][1, 1]) * H
    del model, o
    torch.cuda.empty_cache()
    sky = cv2.morphologyEx(sky.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    sky |= ~np.isfinite(z) | (z <= 0)  # the opening must not leave invalid depths behind
    z[sky] = SKY_M
    return z, sky, normal, fy


def measure(image: Path, out: Path) -> None:
    photo = np.asarray(Image.open(image).convert("RGB"))
    H, W = photo.shape[:2]
    z, sky, normal, fy = moge(photo)

    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(photo).save(out / "photo.png")
    z.astype("<f4").tofile(out / "depth_moge.bin")  # MoGe's own scale; `scale` always starts from this
    save_normals(normal * np.array([1.0, -1.0, 1.0]), sky, out / "normal.png")
    Image.fromarray(sky.astype(np.uint8) * 255).save(out / "sky.png")
    fov = float(np.degrees(2 * np.arctan(H / 2 / fy)))
    info = {"source": str(image).replace("\\", "/"), "image": "photo.png", "albedo": "photo.png", "width": W, "height": H,
            "depth": "depth.bin", "normal": "normal.png", "normalEncoding": "standard", "fovDeg": round(fov, 2),
            "relief": 0.6, "haze": {"airlight": [0.8, 0.82, 0.85], "beta": 0.0}, "metersPerUnit": 1.0,
            "depthModel": "MoGe-2 (metric), camera_map.py", "scale": 1.0}
    info["up"] = estimate_up(out / "normal.png", z, sky)
    (out / "scene.json").write_text(json.dumps(info, indent=2))
    write_scaled(out, 1.0)


def dimensions(z: np.ndarray, sky: np.ndarray, fy: float) -> list[dict]:
    """A few sizes a person can judge from the picture: how far the nearest surface is, how far the middle of
    the scene is, how wide the frame is there, and the width of each opening (sky or far view framed by
    near surfaces) at the distance of its rim."""
    H, W = z.shape
    land = ~sky
    L = np.where(land, z, np.nan)
    dims = []
    # Nearest surface (2nd percentile, ignoring the outer 3% so a corner doesn't win).
    m = int(0.03 * min(H, W))
    inner = np.zeros_like(land); inner[m:-m, m:-m] = True
    zn = np.nanpercentile(L[inner & land], 2)
    yx = np.argwhere(inner & land & (np.abs(z - zn) < zn * 0.03))
    p = yx[len(yx) // 2]
    dims.append({"name": "nearest", "what": "distance to the nearest surface", "m": zn, "at": [int(p[1]), int(p[0])]})
    zm = float(np.nanmedian(L))
    dims.append({"name": "middle", "what": "distance to the middle of the scene (median)", "m": zm})
    dims.append({"name": "frame_width", "what": f"width the frame covers at {zm:.1f} m", "m": W * zm / fy, "line": [[0, H // 2], [W - 1, H // 2]]})
    # Openings: regions much farther than the near surface around them (sky, or land > 3x the rim's depth).
    rim_z = cv2.erode(np.where(land, z, 1e9).astype(np.float32), np.ones((25, 25), np.uint8))
    far_mask = sky | (z > 3 * cv2.blur(rim_z, (61, 61)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(far_mask.astype(np.uint8))
    k = 0
    for i in np.argsort(-st[1:, cv2.CC_STAT_AREA]) + 1:
        if st[i, cv2.CC_STAT_AREA] < 0.01 * H * W or k >= 2:
            break
        x, y, w, h = st[i, :4]
        if x == 0 or x + w >= W:  # touches a side: not framed, no width to read
            continue
        reg = lab == i
        ring = cv2.dilate(reg.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~reg & land
        if not ring.any():
            continue
        zr = float(np.median(z[ring]))
        # widest row of the opening
        widths = reg.sum(1); row = int(np.argmax(widths)); xs = np.nonzero(reg[row])[0]
        k += 1
        dims.append({"name": "opening" if k == 1 else f"opening{k}", "what": f"width of the opening at its widest, at the rim ({zr:.1f} m away)",
                     "m": (xs[-1] - xs[0]) * zr / fy, "line": [[int(xs[0]), row], [int(xs[-1]), row]]})
        dims.append({"name": "rim" if k == 1 else f"rim{k}", "what": "distance to the opening's rim", "m": zr})
    far_land = np.nanpercentile(L, 98)
    yx = np.argwhere(land & (np.abs(z - far_land) < far_land * 0.05))
    if len(yx):
        p = yx[len(yx) // 2]
        dims.append({"name": "farthest", "what": "distance to the farthest land", "m": far_land, "at": [int(p[1]), int(p[0])]})
    for d in dims:
        d["m"] = round(float(d["m"]), 2)
    return dims


def write_scaled(out: Path, factor: float) -> None:
    info = json.loads((out / "scene.json").read_text())
    W, H = info["width"], info["height"]
    z0 = np.fromfile(out / "depth_moge.bin", dtype="<f4").reshape(H, W).astype(np.float64)
    sky = np.asarray(Image.open(out / "sky.png")) > 127
    z = np.where(sky, SKY_M, z0 * factor)
    z.astype("<f4").tofile(out / "depth.bin")
    land = z[~sky]
    info.update(scale=factor, near=float(land.min()), far=SKY_M)
    if "border" in info:
        info.pop("border", None); info.pop("canvas", None)
        print("  (the painted layers were for the old size: run paint again)")
    (out / "scene.json").write_text(json.dumps(info, indent=2))
    fy = H / (2 * np.tan(np.radians(info["fovDeg"]) / 2))
    dims = dimensions(z, sky, fy)
    (out / "dimensions.json").write_text(json.dumps(dims, indent=2))
    draw_dimensions(out, dims)
    print(f"scale x{factor:g}; field of view {info['fovDeg']:.0f} deg vertical")
    for d in dims:
        print(f"  {d['name']:12s} {d['m']:8.2f} m   {d['what']}")
    print(f"Check {out / 'dimensions.png'}; correct with: camera_map.py scale {out} --set <name>=<metres>")


def draw_dimensions(out: Path, dims: list[dict]) -> None:
    img = Image.open(out / "photo.png").convert("RGB")
    W, H = img.size
    s = max(W, H) / 1000
    try:
        font = ImageFont.truetype("arial.ttf", int(22 * s))
    except OSError:
        font = ImageFont.load_default()
    d = ImageDraw.Draw(img)
    col = (255, 210, 0)

    def label(x, y, text):
        x = min(max(x, 4), W - d.textlength(text, font=font) - 4)
        d.text((x, y), text, font=font, fill=col, stroke_width=max(2, int(3 * s)), stroke_fill=(0, 0, 0))

    for dm in dims:
        txt = f"{dm['name']}: {dm['m']:.1f} m"
        if "line" in dm:
            (x0, y0), (x1, y1) = dm["line"]
            d.line([(x0, y0), (x1, y1)], fill=col, width=max(2, int(3 * s)))
            for x in (x0, x1):
                d.line([(x, y0 - 12 * s), (x, y0 + 12 * s)], fill=col, width=max(2, int(3 * s)))
            label((x0 + x1) / 2 - 80 * s, y0 - 34 * s if dm["name"] != "frame_width" else y0 + 10 * s, txt)
        elif "at" in dm:
            x, y = dm["at"]
            r = 10 * s
            d.ellipse([x - r, y - r, x + r, y + r], outline=col, width=max(2, int(3 * s)))
            label(x + 14 * s, y - 12 * s, txt)
    y = 8 * s
    for dm in dims:
        if "line" not in dm and "at" not in dm:
            label(8 * s, y, f"{dm['name']}: {dm['m']:.1f} m ({dm['what']})"); y += 28 * s
    img.save(out / "dimensions.png")


def scale(out: Path, sets: list[str], factor: float | None) -> None:
    if factor is None:
        factor = 1.0
        if sets:
            info = json.loads((out / "scene.json").read_text())
            dims = {d["name"]: d["m"] for d in json.loads((out / "dimensions.json").read_text())}
            ratios = []
            for s in sets:
                name, val = s.split("=")
                if name not in dims:
                    sys.exit(f"unknown dimension {name!r}; have {', '.join(dims)}")
                ratios.append(float(val) / dims[name])
            # dimensions.json is at the current scale: compose with it
            factor = info.get("scale", 1.0) * float(np.exp(np.mean(np.log(ratios))))
    write_scaled(out, factor)



def birefnet_cut(photo: np.ndarray, rough: np.ndarray, margin: int = 24) -> np.ndarray:
    """An object's exact cut-out (sharp outline, real holes) from BiRefNet (MIT), run on a crop round its rough
    outline and kept within `margin` px of it, so it can't take a neighbouring object."""
    import torch
    from transformers import AutoModelForImageSegmentation

    H, W = rough.shape
    ys, xs = np.nonzero(rough)
    pad = int(0.08 * max(np.ptp(ys), np.ptp(xs))) + margin
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad + 1)
    crop = photo[y0:y1, x0:x1].astype(np.float32) / 255
    model = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True, cache_dir=HF_HUB).to("cuda").eval().half()
    x = cv2.resize(crop, (1024, 1024), interpolation=cv2.INTER_AREA)
    x = (x - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    with torch.no_grad():
        pred = model(torch.tensor(x.transpose(2, 0, 1)[None], dtype=torch.float16, device="cuda"))[-1].sigmoid()
    alpha = cv2.resize(pred[0, 0].float().cpu().numpy(), (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    del model
    torch.cuda.empty_cache()
    cut = np.zeros_like(rough)
    cut[y0:y1, x0:x1] = alpha > 0.5
    near = cv2.dilate(rough.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1,) * 2)).astype(bool)
    return cut & near


# --- layers ------------------------------------------------------------------------------------------------

def layers(out: Path) -> None:
    info = json.loads((out / "scene.json").read_text())
    W, H = info["width"], info["height"]
    desc = json.loads((out / "parts.json").read_text())
    L = np.log(np.fromfile(out / "depth.bin", dtype="<f4").reshape(H, W).astype(np.float64))
    names, order, masks, standalone, sharp, hair, opn = [], {}, {}, {}, {}, {}, {}
    for part in desc["parts"]:
        lay = part["layer"]
        m = np.asarray(Image.open(out / "parts" / f"{part['name'].replace(' ', '_')}.png").convert("L")) > 127
        if lay not in masks:
            names.append(lay); masks[lay] = np.zeros((H, W), bool); order[lay] = part["order"]; standalone[lay] = True; sharp[lay] = True; hair[lay] = False; opn[lay] = True
        masks[lay] |= m
        order[lay] = min(order[lay], part["order"])
        standalone[lay] &= bool(part.get("freestanding", False))
        # "outline": "sam" for see-through objects (glass, a translucent fin): BiRefNet cuts what's seen through them.
        sharp[lay] &= part.get("outline", "birefnet") != "sam"
        # "hair": true: fine see-through detail round the outline (hairs, fibres, wisps) goes with this layer as soft
        # alpha. "open": true: open water or sky, no surface to stand on: the layer goes on the far card.
        hair[lay] |= bool(part.get("hair", False))
        opn[lay] &= bool(part.get("open", False))
    names.sort(key=lambda n: order[n])
    rest = names.index(desc.get("rest", names[0]))
    lab = np.full((H, W), -1)
    for i in reversed(range(len(names))):  # nearer layers win overlaps
        lab[masks[names[i]]] = i
    # Specks (islands under 0.05% of the picture) are decided again like unclaimed pixels.
    for i in range(len(names)):
        k, cc, st, _ = cv2.connectedComponentsWithStats((lab == i).astype(np.uint8))
        for c in range(1, k):
            if st[c, cv2.CC_STAT_AREA] < 5e-4 * H * W:
                lab[cc == c] = -1
    # Unclaimed pixels: of the layers within 25 px, the one whose median depth is closest, if it fits; else the rest
    # layer. Land is compared with land (a layer's sky doesn't count in its typical depth); sky with sky.
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    sky_px = L >= np.log(SKY_M * 0.98)
    cost = np.full((len(names), H, W), np.inf)
    for i in range(len(names)):
        if standalone[names[i]] and not sharp[names[i]]:
            continue  # a see-through object's SAM outline is final: depth (blurred across its edge) mustn't grow it
        if (lab == i).any():
            near = cv2.dilate((lab == i).astype(np.uint8), disk).astype(bool)
            land_i = (lab == i) & ~sky_px
            if land_i.any():
                c = near & ~sky_px
                cost[i][c] = np.abs(L - np.median(L[land_i]))[c]
            if ((lab == i) & sky_px).any():
                cost[i][near & sky_px] = 0.0
    free = lab < 0
    fits = cost.min(0) < np.log(1.6)  # a nearby layer at about this depth; else the rest layer
    lab[free] = np.where(fits, cost.argmin(0), rest)[free]
    # Freestanding objects (a car, a boulder) get their final outline from BiRefNet, a model made for exact cut-outs.
    # SAM's outlines are loose and keep holes (sky through a window), and the gap SAM's sky leaves round an object
    # gets filled above by depth, which MoGe blurs across the edge: a band of sky round the car. SAM still says which
    # object is which (BiRefNet only works inside a margin round the outline); if BiRefNet keeps under half of it (a
    # blurry boulder at the frame's edge isn't "the object" to it), SAM's outline stands. Pixels the object gives
    # up go to the nearest other layer.
    photo_rgb = np.asarray(Image.open(out / info["image"]).convert("RGB"))
    for i, n in enumerate(names[:-1]):
        m = lab == i
        if not standalone[n] or not sharp[n] or m.sum() < 500:
            continue
        cut = birefnet_cut(photo_rgb, m)
        if (cut & m).sum() < 0.5 * m.sum():
            print(f"  {n}: BiRefNet disagrees (keeps {(cut & m).sum() / m.sum() * 100:.0f}%), SAM's outline stands")
            continue
        give = m & ~cut
        take = cut & ~m & ((lab < 0) | (lab > i))  # only from farther layers or unclaimed
        _, (iy, ix) = ndi.distance_transform_edt((lab == i) | (lab < 0), return_indices=True)
        lab[give] = lab[iy[give], ix[give]]
        lab[take] = i
        print(f"  {n}: BiRefNet outline; gave up {give.sum()} px, took {take.sum()} px")
    # The one line of edge pixels that is a mix of both sides (anti-aliasing) belongs to the layer in front, or the
    # far layer keeps a line of near colour along its edge that stays behind when the near layer slides away.
    for i in range(1, len(names)):
        front = np.where((lab >= 0) & (lab < i), lab, 99).astype(np.uint8)
        nearest_front = cv2.erode(front, np.ones((3, 3), np.uint8))  # min label within 1 px
        edge = (lab == i) & (nearest_front < i)
        lab[edge] = nearest_front[edge]
    # Last, slivers (under ~5 px wide) and islands (under 0.05% of the picture) join the layer around them: no
    # layer keeps hairlines or floating specks.
    for _ in range(2):
        for i in range(len(names)):
            m = (lab == i).astype(np.uint8)
            thin = m.astype(bool) & ~cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)
            if thin.any():
                other = np.where(lab == i, 255, lab).astype(np.uint8)
                # each sliver pixel: the most common other layer within 3 px
                votes = np.stack([cv2.blur((other == j).astype(np.float32), (7, 7)) for j in range(len(names))])
                votes[i] = -1
                lab[thin] = votes.argmax(0)[thin]
        for i in range(len(names)):
            k, cc, st, _ = cv2.connectedComponentsWithStats((lab == i).astype(np.uint8))
            for c in range(1, k):
                if st[c, cv2.CC_STAT_AREA] < 5e-4 * H * W:
                    isl = cc == c
                    ring = cv2.dilate(isl.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~isl
                    around = lab[ring]
                    if around.size:
                        lab[isl] = np.bincount(around, minlength=len(names)).argmax()

    (out / "layers").mkdir(exist_ok=True)
    photo = np.asarray(Image.open(out / info["image"]).convert("RGB")).astype(np.float32)
    cols = np.array([[230, 90, 70], [80, 200, 110], [80, 140, 255], [240, 200, 60], [200, 90, 220]], np.float32)
    over = photo * 0.4
    meta = []
    for i, n in enumerate(names):
        m = lab == i
        Image.fromarray(m.astype(np.uint8) * 255).save(out / "layers" / f"{n}_mask.png")
        over[m] += cols[i % len(cols)] * 0.6
        zm = np.exp(L[m & (L < np.log(SKY_M * 0.98))])
        rng = [round(float(np.percentile(zm, 5)), 1), round(float(np.percentile(zm, 95)), 1)] if zm.size else None
        meta.append({"name": n, "prompt": desc.get("prompts", {}).get(n, ""), "freestanding": standalone[n] and i < len(names) - 1,
                     "area": round(float(m.mean()), 4), "depth_m": rng, "hair": hair[n], "open": opn[n] and i == len(names) - 1})
        print(f"  {n:6s} {m.mean() * 100:5.1f}% of the picture" + (f", {rng[0]}-{rng[1]} m" if rng else ""))
    img = Image.fromarray(over.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", int(max(W, H) / 40))
    except OSError:
        font = ImageFont.load_default()
    for i, n in enumerate(names):
        ys, xs = np.nonzero(lab == i)
        if len(xs):
            j = int(np.argmin((xs - np.median(xs)) ** 2 + (ys - np.median(ys)) ** 2))
            d.text((int(xs[j]), int(ys[j])), n, font=font, fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))
    img.save(out / "layers_overlay.png")
    (out / "layers.json").write_text(json.dumps({"layers": meta}, indent=2))
    print(f"Check {out / 'layers_overlay.png'}")


# --- paint -------------------------------------------------------------------------------------------------

_flux = None


def flux_fill(img: np.ndarray, mask: np.ndarray, prompt: str, seed: int = 7) -> np.ndarray:
    """FLUX.1 Fill over the whole canvas, at a size the model likes (multiples of 16, at most ~2.5 MP); only the
    masked pixels are taken back, at the canvas's own resolution."""
    import torch
    from diffusers import FluxFillPipeline

    global _flux
    H, W = mask.shape
    k = min(1.0, (2.5e6 / (H * W)) ** 0.5)
    w, h = int(W * k) // 16 * 16, int(H * k) // 16 * 16
    if _flux is None:
        _flux = FluxFillPipeline.from_pretrained(FLUX_FILL, torch_dtype=torch.bfloat16)
        # The transformer is 23.8 GB in bf16: just over the 4090's free memory, and Windows then spills it into
        # system RAM (20x slower). Stored as fp8 (computed in bf16) it is ~12 GB and fits.
        _flux.transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
        _flux.enable_model_cpu_offload()
        _flux.set_progress_bar_config(mininterval=10)
    res = _flux(prompt=prompt, image=Image.fromarray(img).resize((w, h), Image.LANCZOS),
                mask_image=Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.NEAREST),
                height=h, width=w, guidance_scale=30, num_inference_steps=40, max_sequence_length=512,
                generator=torch.Generator("cpu").manual_seed(seed)).images[0]
    res = np.asarray(res.resize((W, H), Image.LANCZOS)).astype(np.float32)
    # Flux returns the whole picture through its autoencoder, so even untouched pixels come back with a slight
    # tone shift, and its painting matches that shifted version. Measured on the same pixels just outside the hole
    # (original minus Flux's version), smoothed, and carried smoothly across the hole, the shift is taken back out
    # of the painting: the join to the real pixels disappears.
    ring = cv2.dilate(mask.astype(np.uint8), np.ones((97, 97), np.uint8)).astype(bool) & ~mask
    corr = membrane(img.astype(np.float32) - res, ring, mask)
    out = res + corr
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.5)[..., None]
    return (out * soft + img * (1 - soft)).clip(0, 255).astype(np.uint8)


def membrane(diff: np.ndarray, known: np.ndarray, fill: np.ndarray) -> np.ndarray:
    """A smooth field that equals the (low-pass) difference `diff` where it is `known` and continues it smoothly
    over `fill` (OpenCV inpainting of the difference at 1/4 size). Zero elsewhere."""
    H, W = known.shape
    w = cv2.GaussianBlur(known.astype(np.float32), (0, 0), 6)[..., None]
    lp = cv2.GaussianBlur(diff * known[..., None], (0, 0), 6) / np.maximum(w, 1e-4)
    q = 4
    small = cv2.resize(lp, (W // q, H // q), interpolation=cv2.INTER_AREA)
    kn = cv2.resize((w[..., 0] > 0.3).astype(np.uint8), (W // q, H // q), interpolation=cv2.INTER_NEAREST).astype(bool)
    if not kn.any():
        return np.zeros_like(diff)
    filled = np.zeros_like(small)
    for c in range(3):  # 8-bit inpainting of the difference, offset to 128, quarter-level steps
        ch = np.clip(small[..., c] * 4 + 128, 0, 255).astype(np.uint8)
        filled[..., c] = (cv2.inpaint(ch, (~kn).astype(np.uint8), 15, cv2.INPAINT_TELEA).astype(np.float32) - 128) / 4
    full = cv2.GaussianBlur(cv2.resize(filled, (W, H), interpolation=cv2.INTER_LINEAR), (0, 0), 4)
    return np.where((fill | known)[..., None], full, 0)


def sam_object(img: np.ndarray, own: np.ndarray, not_this: np.ndarray, n_pos: int = 8, n_neg: int = 8) -> np.ndarray:
    """SAM 2's outline of a layer in its painted image, seeded on the layer's real pixels (spread-out points
    inside them), with "not this" points on `not_this` near it: real pixels of the farther layers, never on the
    painted parts (they are what's being judged)."""
    import torch
    from transformers import Sam2Model, Sam2Processor

    H, W = own.shape
    rng = np.random.default_rng(0)

    def spread(mask, n):  # farthest-point sampling: points spread over the whole region
        ys, xs = np.nonzero(mask)
        if not len(xs):
            return []
        pts = [int(rng.integers(len(xs)))]
        d = np.full(len(xs), np.inf)
        for _ in range(n - 1):
            d = np.minimum(d, (xs - xs[pts[-1]]) ** 2 + (ys - ys[pts[-1]]) ** 2)
            pts.append(int(np.argmax(d)))
        return [[float(xs[p]), float(ys[p])] for p in pts]

    inside = cv2.erode(own.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    outside = cv2.dilate(own.astype(np.uint8), np.ones((121, 121), np.uint8)).astype(bool) & not_this
    pos, neg = spread(inside if inside.any() else own, n_pos), spread(outside, n_neg)
    k = min(1.0, 1920 / W)
    small = Image.fromarray(img).resize((int(W * k), int(H * k)), Image.LANCZOS)
    proc = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")
    model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").to("cuda").eval()
    points = [[x * k, y * k] for x, y in pos + neg]
    inputs = proc(images=small, input_points=[[points]], input_labels=[[[1] * len(pos) + [0] * len(neg)]], return_tensors="pt").to("cuda")
    with torch.no_grad():
        o = model(**inputs, multimask_output=True)
    masks = proc.post_process_masks(o.pred_masks.cpu(), inputs["original_sizes"])[0][0]
    best = int(np.argmax(o.iou_scores[0, 0].float().cpu().numpy()))
    del model
    torch.cuda.empty_cache()
    m = cv2.resize(masks[best].numpy().astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    return m | own


def fit_depth(ze: np.ndarray, known: np.ndarray, zk: np.ndarray, blur: float) -> np.ndarray:
    """MoGe's depth for a painted image, fitted onto the scene's depth where that is known: the log ratio between
    the two, carried out from the known pixels to the rest and smoothed, so painted parts continue the real
    surfaces' distances without a step. Known pixels get the scene's depth exactly."""
    lr = np.zeros(ze.shape)
    lr[known] = np.log(zk[known] / ze[known])
    _, (iy, ix) = ndi.distance_transform_edt(~known, return_indices=True)
    lr = cv2.GaussianBlur(lr[iy, ix].astype(np.float32), (0, 0), blur).astype(np.float64)
    zl = ze * np.exp(lr)
    zl[known] = zk[known]
    return zl


def smooth_backdrop(img: np.ndarray, known: np.ndarray, q: int = 8, scales=(30, 80, 250)) -> np.ndarray:
    """Open water / clear sky over the whole canvas from its `known` pixels: normalized blurs at several scales,
    the finer ones taking over wherever they have enough known pixels, so the fill follows the real water near it
    and is only the broad gradient deep inside a hole (no outline of what stood in front). The real pixels go
    through the same smooth field too (no seam where they meet the fill); only their fine detail (bokeh specks) is
    added back. For soft, featureless backdrops: Flux paints an object into an object-shaped hole."""
    Hc, Wc = known.shape
    src = img.astype(np.float32)
    k = cv2.resize(known.astype(np.float32), (Wc // q, Hc // q), interpolation=cv2.INTER_AREA)
    c = cv2.resize(src * known[..., None], (Wc // q, Hc // q), interpolation=cv2.INTER_AREA)
    f = None
    for s in reversed(scales):  # coarse first
        w = cv2.GaussianBlur(k, (0, 0), s / q, borderType=cv2.BORDER_REPLICATE)
        n = cv2.GaussianBlur(c, (0, 0), s / q, borderType=cv2.BORDER_REPLICATE) / np.maximum(w, 1e-5)[..., None]
        conf = np.clip(w / 0.3, 0, 1)[..., None]
        f = n if f is None else n * conf + f * (1 - conf)
    low = cv2.GaussianBlur(cv2.resize(f, (Wc, Hc), interpolation=cv2.INTER_CUBIC), (0, 0), q)
    kn = known.astype(np.float32)
    local = cv2.GaussianBlur(src * kn[..., None], (0, 0), 6) / np.maximum(cv2.GaussianBlur(kn, (0, 0), 6), 1e-4)[..., None]
    fine = (src - local) * kn[..., None]
    wk = cv2.GaussianBlur(cv2.erode(known.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(np.float32), (0, 0), 3)[..., None]
    return (low + fine * wk).clip(0, 255).astype(np.uint8)


def lift_hair(out: Path, lay: dict, photo: np.ndarray, clean: np.ndarray, band: np.ndarray) -> None:
    """Difference key: what the photo has over the clean plate in `band` (brighter: glowing fibres, sparkles)
    becomes soft alpha on the layer's image, its colour un-mixed from the plate; depth from the layer's nearest
    real pixel, so the fibres recede with the body."""
    rgba = np.asarray(Image.open(out / lay["image"])).copy()
    Hc, Wc = band.shape
    z = np.fromfile(out / lay["depth"], dtype="<f4").reshape(Hc, Wc)
    p, c = photo.astype(np.float32), clean.astype(np.float32)
    lum = lambda a: a @ np.array([0.299, 0.587, 0.114], np.float32)
    lp, lc = lum(p), lum(c)
    top = float(np.percentile(lp[band], 99.7))  # a fibre's full brightness
    a = np.clip((lp - lc) / np.maximum(top - lc, 8.0), 0, 1)
    a = np.where(a < 0.04, 0, (a - 0.04) / 0.96) * band  # below the plate's own noise: nothing
    sel = (a > 0) & (rgba[..., 3] == 0)
    col = np.clip(c + (p - c) / np.maximum(a, 0.04)[..., None], 0, 255)
    rgba[sel, :3] = col[sel].astype(np.uint8)
    rgba[sel, 3] = np.maximum(1, (a[sel] * 255).round()).astype(np.uint8)
    solid = rgba[..., 3] == 255
    # Depth of the nearest real pixel, smoothed: nearest alone jumps where the nearest part changes (belly to tail)
    # and one strand would be cut into pieces at different depths that drift apart as the camera moves.
    _, (iy, ix) = ndi.distance_transform_edt(~solid, return_indices=True)
    zs = cv2.GaussianBlur(z[iy, ix].astype(np.float32), (0, 0), 25)
    z[sel] = zs[sel]
    Image.fromarray(rgba, "RGBA").save(out / lay["image"])
    z.astype("<f4").tofile(out / lay["depth"])
    print(f"  {lay['name']}: lifted {sel.sum()} px of fibres (mean alpha {a[sel].mean():.2f}) from the clean plate")


def paint(out: Path, border: float, reach: float, move: float, margin: float) -> None:
    info = json.loads((out / "scene.json").read_text())
    W, H = info["width"], info["height"]
    f = H / (2 * np.tan(np.radians(info["fovDeg"]) / 2))
    meta = json.loads((out / "layers.json").read_text())["layers"]
    photo = np.asarray(Image.open(out / info["image"]).convert("RGB"))
    z = np.fromfile(out / "depth.bin", dtype="<f4").reshape(H, W).astype(np.float64)
    m = int(round(border * max(W, H)))
    R_max = int(round(reach * max(W, H)))
    pad = lambda a, mode=cv2.BORDER_CONSTANT: cv2.copyMakeBorder(a, m, m, m, m, mode)
    canvas = pad(photo, cv2.BORDER_REPLICATE)
    zc = pad(z.astype(np.float32), cv2.BORDER_REPLICATE).astype(np.float64)
    ring = ~pad(np.ones((H, W), np.uint8)).astype(bool)  # past the frame's edges
    land_c = zc < SKY_M * 0.98
    fov_x = float(np.degrees(2 * np.arctan((W + 2 * m) / 2 / f)))
    owns = [pad((np.asarray(Image.open(out / "layers" / f"{lay['name']}_mask.png")) > 127).astype(np.uint8)).astype(bool) for lay in meta]
    ellipse = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    # Hair: a band round a "hair" layer's outline, over the back layer's own pixels. The back layer is painted clean
    # there (no fibres); at the end the fibres are pulled out of the photo against that clean plate (difference
    # key) as soft alpha on the hair layer, at the depth of its nearest real pixel: they move with it.
    hair_r = int(round(0.08 * max(W, H)))
    hairy = [i for i, lay in enumerate(meta) if lay.get("hair") and i < len(meta) - 1]
    dist = {i: ndi.distance_transform_edt(~owns[i]) for i in hairy}
    bands = {}
    for i in hairy:  # each fibre goes with the hair layer it is closest to (the nearer layer on a tie)
        b = (dist[i] <= hair_r) & ~owns[i] & owns[-1] & ~ring
        for j in hairy:
            if j != i:
                b &= (dist[i] < dist[j]) | ((dist[i] == dist[j]) & (i < j))
        for j in range(i):  # nearer layers' pixels never take another layer's hair
            b &= ~cv2.dilate(owns[j].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        bands[i] = b
    hair_px = np.zeros(ring.shape, bool)
    for b in bands.values():
        hair_px |= b
    nearer = np.zeros(ring.shape, bool)  # own pixels of the layers in front
    front_z = np.zeros(ring.shape)       # depth of the nearest layer in front, wherever one covers
    print(f"canvas {W + 2 * m}x{H + 2 * m} ({m} px border); camera move {move} m")
    for i, lay in enumerate(meta):
        n = lay["name"]
        own = owns[i]
        if i > 0:
            # Choke: the 2 px of this layer next to a layer in front are a mix of both (anti-aliasing); kept, they
            # leave a thin outline of the front object behind when it slides away. They are painted with the hole.
            own = own & ~cv2.dilate(nearer.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        if i == len(meta) - 1:
            # The back layer's small islands enclosed by nearer layers (sky through a car's windows) are repainted
            # with the rest: kept, they leave the window's outline standing in the painted sky.
            k, cc, st, _ = cv2.connectedComponentsWithStats(own.astype(np.uint8))
            if k > 2:
                main = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
                isl = (cc > 0) & (cc != main) & np.isin(cc, [c for c in range(1, k) if st[c, cv2.CC_STAT_AREA] < 0.005 * H * W])
                own = own & ~isl
        back = i == len(meta) - 1
        if back:
            own = own & ~hair_px  # painted clean: its fibres go with the layer they grow from
        given_up = owns[i] & ~own  # choked edge and enclosed islands: this layer's, but painted
        hidden = nearer | ring | given_up
        # How far behind each layer in front the camera can see: f * move * (1/near - 1/far), with a safety margin.
        behind = np.zeros_like(own)
        radii = []
        for j in range(i):
            zf = zc[owns[j] & land_c & ~ring]
            # this layer's depth just next to that layer (what is actually behind its edge)
            zb = zc[own & land_c & cv2.dilate(owns[j].astype(np.uint8), ellipse(30)).astype(bool)]
            if not (zf.size and zb.size):
                continue
            Rj = int(np.clip(f * move * margin * (1 / np.percentile(zf, 5) - 1 / np.median(zb)), 24, R_max))
            radii.append(f"{meta[j]['name']} {Rj}")
            behind |= cv2.dilate(own.astype(np.uint8), ellipse(Rj)).astype(bool) & owns[j] & ~ring
        # The layer: its own pixels, plus what's hidden behind the layers in front up to R px from them, plus its
        # continuation past the frame's edges (the whole border). The back layer: everything.
        # Past the frame only from where the layer reaches the frame's edge (an object wholly in frame has nothing
        # out there to continue).
        at_edge = own & cv2.dilate(ring.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        past = cv2.dilate(at_edge.astype(np.uint8), ellipse(m)).astype(bool) & ring
        keep = np.ones_like(own) if back else own | behind | past | given_up
        paint_px = keep & hidden & ~own
        print(f"{n}: {own[~ring].mean() * 100:.1f}% of the frame is its own; painting {paint_px.mean() * 100:.1f}% of the canvas"
              + ("" if back or not radii else f" (px behind: {', '.join(radii)})"))
        img = canvas.copy()
        if paint_px.any() and back and lay.get("open"):
            # Open water / sky is out of focus and featureless: a smooth continuation of what's round the hole.
            # Flux paints an object into an object-shaped hole (a ghost of what's in front).
            img = smooth_backdrop(canvas, own & ~ring)
        elif paint_px.any():
            # Exactly the hole (growing it into the real pixels and pasting them back leaves a hard join on the
            # outline). An object Flux paints into an object-shaped hole is caught by the SAM check below.
            img = flux_fill(canvas, hidden & ~own, lay["prompt"] or "natural photo, sharp detail, matching surroundings", seed=11 + i)
        if not (back and lay.get("open")):  # open water is all one smooth field (see smooth_backdrop)
            img[own] = canvas[own]
        ze, skye, _, _ = moge(img, fov_x=fov_x)
        if not back and paint_px.any():
            # A front layer keeps only painted pixels that continue it. Past the frame: anything but painted sky.
            # Behind the layers in front: what SAM, seeded on the layer's real pixels, sees as more of the layer
            # (Flux sometimes paints an object there, the ghost of what's in front).
            drop = paint_px & skye
            judged = behind | (paint_px & ring if lay.get("freestanding") else np.zeros_like(own))
            if judged.any():
                farther = np.zeros_like(own)
                for j in range(i + 1, len(meta)):
                    farther |= owns[j] & ~ring
                rejected = judged & ~sam_object(img, own, farther)
                # past the frame, a check that throws away most of the extension is the check failing (SAM doesn't
                # see a blurry boulder as an object either): the extension stays there, minus sky
                past_j = judged & ring
                if past_j.any() and (rejected & past_j).sum() > 0.5 * past_j.sum():
                    rejected &= ~ring
                drop |= rejected
            keep &= ~drop
            # What the camera needs but was dropped: inpainted from the layer's own kept colours (OpenCV Telea, a
            # smooth continuation; fine for a narrow strip). The rest of the picture is masked out of it.
            need = behind & ~keep
            if need.any():
                ys, xs = np.nonzero(cv2.dilate(need.astype(np.uint8), ellipse(40)))
                y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                sub = img[y0:y1, x0:x1].copy()
                hole = (~keep[y0:y1, x0:x1]).astype(np.uint8)
                filled = cv2.inpaint(sub, hole, 9, cv2.INPAINT_TELEA)
                nd = need[y0:y1, x0:x1]
                sub[nd] = filled[nd]
                img[y0:y1, x0:x1] = sub
                keep |= need
            print(f"  kept {(keep & hidden & ~own & ~need).mean() * 100:.1f}% of the canvas as painted {n}; inpainted {need.mean() * 100:.2f}%")
        known = own & ~ring & land_c & ~skye
        zl = fit_depth(ze, known, zc, blur=max(W, H) / 30)
        if not back:
            # A front layer's own pixels are all surface, even the mixed edge pixels it took over from the sky
            # behind it (sky depth there): they take the depth of the nearest real surface pixel of the layer.
            odd = (own | (keep & skye)) & ~known
            _, (iy, ix) = ndi.distance_transform_edt(~known, return_indices=True)
            zl[odd] = zl[iy[odd], ix[odd]]
        covered = keep & (front_z > 0)
        zl[covered] = np.maximum(zl[covered], front_z[covered] * 1.08)  # always behind what's in front
        skyl = (skye & ~known) | (own & ~land_c) if back else np.zeros_like(own)
        if back and lay.get("open"):
            skyl = np.ones_like(own)  # open water / sky: all of it on the far card
        zl[skyl] = SKY_M
        Image.fromarray(np.dstack([img, keep.astype(np.uint8) * 255]), "RGBA").save(out / "layers" / f"{n}.png")
        zl.astype("<f4").tofile(out / "layers" / f"{n}_depth.bin")
        if back:
            for j, band in bands.items():
                lift_hair(out, meta[j], canvas, img, band)
        Image.fromarray(skyl.astype(np.uint8) * 255).save(out / "layers" / f"{n}_sky.png")
        lay["image"], lay["depth"], lay["sky"] = f"layers/{n}.png", f"layers/{n}_depth.bin", f"layers/{n}_sky.png"
        front_z = np.where((front_z == 0) & keep & ~skyl, zl, front_z)
        nearer |= owns[i]  # the whole layer, choked edge included, covers what is behind it
    info["border"] = m
    info["canvas"] = [W + 2 * m, H + 2 * m]
    (out / "scene.json").write_text(json.dumps(info, indent=2))
    (out / "layers.json").write_text(json.dumps({"layers": meta}, indent=2))
    sheets = []
    for lay in meta:  # check sheet: each layer on grey, frame outlined
        im = np.asarray(Image.open(out / lay["image"])).astype(np.float32)
        a = im[..., 3:] / 255
        s = (im[..., :3] * a + 90 * (1 - a)).astype(np.uint8)
        cv2.rectangle(s, (m, m), (m + W, m + H), (255, 210, 0), 3)
        sheets.append(s)
    sheet = np.concatenate(sheets, 1)
    Image.fromarray(sheet).resize((sheet.shape[1] // 2, sheet.shape[0] // 2), Image.LANCZOS).save(out / "layers_painted.png")
    print(f"Wrote {len(meta)} layers into {out / 'layers'}; check {out / 'layers_painted.png'}")


# --- export ------------------------------------------------------------------------------------------------

def grid_mesh(path: Path, z: np.ndarray, keep: np.ndarray, f: float, centre: tuple[float, float], step: int,
              cut: float | None = None) -> int:
    """OBJ of a depth map as a camera-space grid (Maya axes: x right, y up, looking down -z), in cm, with UVs that
    project the picture from the camera. `centre`: the optical centre in this map's pixels. Triangles whose
    corners differ in depth by more than `cut` (ratio) are dropped. Returns the triangle count."""
    H, W = z.shape
    ys = np.unique(np.r_[np.arange(0, H, step), H - 1]); xs = np.unique(np.r_[np.arange(0, W, step), W - 1])
    gx, gy = np.meshgrid(xs, ys)
    zz = z[gy, gx]; ok = keep[gy, gx]
    P = np.stack([(gx + 0.5 - centre[0]) / f * zz, -(gy + 0.5 - centre[1]) / f * zz, -zz], -1) * 100.0
    UV = np.stack([(gx + 0.5) / W, 1 - (gy + 0.5) / H], -1)
    idx = np.arange(gx.size).reshape(gx.shape)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    tris = np.concatenate([np.stack([a, c, b], -1).reshape(-1, 3), np.stack([b, c, d], -1).reshape(-1, 3)])
    zf, okf = zz.ravel(), ok.ravel()
    good = okf[tris].all(1)
    if cut:
        zt = zf[tris]
        good &= zt.max(1) / zt.min(1) < cut
    tris = tris[good]
    used = np.unique(tris); remap = np.full(zf.size, -1); remap[used] = np.arange(len(used))
    tris = remap[tris] + 1
    P = P.reshape(-1, 3)[used]; UV = UV.reshape(-1, 2)[used]
    with open(path, "w") as fh:
        fh.write("".join(f"v {x:.3f} {y:.3f} {zv:.3f}\n" for x, y, zv in P))
        fh.write("".join(f"vt {u:.6f} {v:.6f}\n" for u, v in UV))
        fh.write("".join(f"f {i}/{i} {j}/{j} {k}/{k}\n" for i, j, k in tris))
    return len(tris)


def export(out: Path, step: int, move: float, frames: int, period: int, fps: int) -> None:
    import subprocess

    info = json.loads((out / "scene.json").read_text())
    W, H, m = info["width"], info["height"], info["border"]
    Wc, Hc = info["canvas"]
    tan_half = float(np.tan(np.radians(info["fovDeg"]) / 2))
    f = H / (2 * tan_half)
    centre = (m + W / 2, m + H / 2)
    meta = json.loads((out / "layers.json").read_text())["layers"]
    # Every export is a new version (maya/v001, v002, ...): nothing is ever overwritten.
    versions = sorted(int(d.name[1:]) for d in (out / "maya").glob("v[0-9][0-9][0-9]") if d.is_dir())
    version = (versions[-1] + 1) if versions else 1
    dst = out / "maya" / f"v{version:03d}"
    dst.mkdir(parents=True)
    print(f"export version v{version:03d} -> {dst}")
    layers_out, far_land = [], 0.0
    for lay in meta:
        n = lay["name"]
        z = np.fromfile(out / lay["depth"], dtype="<f4").reshape(Hc, Wc).astype(np.float64)
        alpha = np.asarray(Image.open(out / lay["image"]))[..., 3] > 0
        skyl = np.asarray(Image.open(out / lay["sky"])) > 127
        keep = alpha & ~skyl
        if not keep.any():
            if lay is meta[-1]:  # all open water / sky: no mesh, but the sky card wears its image
                Image.open(out / lay["image"]).convert("RGB").save(dst / f"{n}.png")
            continue
        far_land = max(far_land, float(np.percentile(z[keep], 99.5)))
        # A little geometry past the outline (the projected alpha draws the exact edge), at the depth of the
        # layer's nearest real pixel: MoGe's depth there is what's behind, and triangles reaching back to it
        # fan out as streaks as soon as the camera moves.
        _, (iy, ix) = ndi.distance_transform_edt(~keep, return_indices=True)
        z = z[iy, ix]
        keep = cv2.dilate(keep.astype(np.uint8), np.ones((2 * step + 1,) * 2, np.uint8)).astype(bool) & ~skyl
        k = grid_mesh(dst / f"{n}.obj", z, keep, f, centre, step)  # whole: real occlusions are layer edges
        # Bleed: the colours under the transparent margin are the layer's own edge colours pushed outward, so the
        # geometry past the outline and texture filtering at the edge pick up the object, not the sky behind it.
        rgba = np.asarray(Image.open(out / lay["image"])).copy()
        solid = rgba[..., 3] > 0
        if not solid.all():
            _, (by, bx) = ndi.distance_transform_edt(~solid, return_indices=True)
            rgba[..., :3] = rgba[by, bx, :3]
        Image.fromarray(rgba, "RGBA").save(dst / f"{n}.png")
        layers_out.append({"name": n, "obj": f"{n}.obj", "texture": f"{n}.png"})
        print(f"  {n}: {k} triangles")
    sky_d = far_land * 3  # sky card behind everything, textured with the back layer (its sky is painted too)
    k = grid_mesh(dst / "sky.obj", np.full((Hc, Wc), sky_d), np.ones((Hc, Wc), bool), f, centre, 16)
    layers_out.append({"name": "sky", "obj": "sky.obj", "texture": f"{meta[-1]['name']}.png"})
    print(f"  sky card at {sky_d:.0f} m: {k} triangles")
    # Level: turn everything about the camera so the scene's true vertical is Maya's +Y (camera-space up, Maya axes).
    u = np.array(info.get("up", [0, 1, 0]), float) * np.array([1, 1, -1]); u /= np.linalg.norm(u)
    y = np.array([0, 1, 0.0]); v = np.cross(u, y); s, c = np.linalg.norm(v), float(u @ y)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / s**2) if s > 1e-8 else np.eye(3)
    mv = move * 100
    t = np.arange(frames)
    cam = {"x": (mv * np.sin(2 * np.pi * t / period)).round(3).tolist(),
           "y": (mv / 2 * np.sin(2 * np.pi * t / (2 * period))).round(3).tolist()}
    manifest = {"name": out.name, "file": f"{out.name.replace('-', '_')}_v{version:03d}", "width": W, "height": H, "canvas": [Wc, Hc], "tanHalfFov": tan_half, "level": R.tolist(),
                "fps": fps, "layers": layers_out, "camera": cam}
    (dst / "manifest.json").write_text(json.dumps(manifest))
    print(f"renderCam sway: {frames} frames at {fps} fps, +/-{mv:.0f} cm sideways, +/-{mv / 2:.0f} cm up/down, at rest on frame 1")
    subprocess.run([MAYAPY, str(Path(__file__).with_name("camera_map_maya.py")), str(dst.resolve())], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("measure"); a.add_argument("image", type=Path); a.add_argument("out", type=Path)
    a = sub.add_parser("scale"); a.add_argument("out", type=Path); a.add_argument("--set", action="append", default=[]); a.add_argument("--factor", type=float)
    a = sub.add_parser("layers"); a.add_argument("out", type=Path)
    a = sub.add_parser("paint"); a.add_argument("out", type=Path)
    a.add_argument("--border", type=float, default=0.12, help="canvas border each side, fraction of the longer side")
    a.add_argument("--reach", type=float, default=0.1, help="the most a layer is painted on behind nearer ones, fraction of the longer side")
    a.add_argument("--move", type=float, default=0.3, help="the camera's largest move, metres: how much it can see behind things")
    a.add_argument("--margin", type=float, default=2.0, help="safety factor on what the move can reveal")
    a = sub.add_parser("export"); a.add_argument("out", type=Path); a.add_argument("--step", type=int, default=2, help="mesh grid spacing, px")
    a.add_argument("--move", type=float, default=0.3, help="renderCam sway, metres sideways (half that up/down)")
    a.add_argument("--frames", type=int, default=240); a.add_argument("--period", type=int, default=120, help="frames per sideways sway")
    a.add_argument("--fps", type=int, default=24)
    args = ap.parse_args()
    if args.cmd == "measure":
        measure(args.image, args.out)
    elif args.cmd == "scale":
        scale(args.out, args.set, args.factor)
    elif args.cmd == "layers":
        layers(args.out)
    elif args.cmd == "paint":
        paint(args.out, args.border, args.reach, args.move, args.margin)
    else:
        export(args.out, args.step, args.move, args.frames, args.period, args.fps)


if __name__ == "__main__":
    main()
