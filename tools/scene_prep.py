"""Scene prep: one landscape image -> the photo, albedo, depth and normal maps the display page loads.

Runs ahead of time, never inside the app.

    python tools/scene_prep.py images/canyon.png --out scenes/canyon --upscale 1
    python tools/scene_prep.py images/canyon.png --out scenes/canyon --from-depth   # after hand-editing depth.png

Outputs, in --out (default: next to the image), all at the working size:
    photo.png         the image (upscaled with Real-ESRGAN if --upscale > 1, capped at --max-width): the dim base.
    albedo.png        photo.png with its baked lighting removed, for lights to multiply.
    shading.png       the baked lighting that was removed (for reference).
    dehazed.png       photo.png with the atmospheric haze removed (what delighting works from).
    delight_raw.png   Marigold's albedo at up to --delight-res (cached; --redo-delight to recompute).
    sky.png           sky mask (white = sky).
    distant.png       distant-land mask (mountains), pushed back by --distant-push. Paint your own and pass --hand-distant.
    depth.png         16-bit inverse depth, white = near, black = far/sky. Edit by hand to fix wobbly ridges.
    depth.bin         float32 view-space distance per pixel (what the app loads), row-major.
    normals_raw.npy   cached Marigold normals (--redo-normals to recompute).
    normal.png        view-space normals, standard tangent-space colours: x right, y up, z toward the camera
                      (a surface facing the camera is (128, 128, 255)).
    scene.json        sizes and camera settings shared with the app, including "up": true vertical measured
                      from the flat ground (the photo's camera is rarely level).

Depth is fused from three models:
    Depth Anything V2   shape and surface detail (relative depth)
    Depth Pro           metric scale, lens angle, and the far distance where Depth Anything flattens out
    SegFormer (ADE20K)  sky mask: neither depth model handles sky reliably
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter, zoom

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Large-hf"
METRIC_MODEL = "apple/DepthPro-hf"
SKY_MODEL = "nvidia/segformer-b5-finetuned-ade-640-640"
DELIGHT_MODEL = "prs-eth/marigold-iid-lighting-v1-1"
NORMALS_MODEL = "prs-eth/marigold-normals-v1-1"
UPSCALE_MODEL = Path(__file__).parent / "models" / "RealESRGAN_x4plus.pth"


# --- helpers ---------------------------------------------------------------------------------


def device_and_dtype():
    import torch

    return ("cuda", torch.float16) if torch.cuda.is_available() else ("cpu", torch.float32)


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def to_float(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0


def to_image(c: np.ndarray) -> Image.Image:
    return Image.fromarray(np.round(np.clip(c, 0.0, 1.0) * 255).astype(np.uint8))


def luminance(c: np.ndarray) -> np.ndarray:
    return c @ np.array([0.2126, 0.7152, 0.0722])


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Edge-aware smoothing of src that follows the edges in guide (He et al. 2010)."""
    box = lambda a: uniform_filter(a, 2 * radius + 1, mode="reflect")
    mean_g, mean_s = box(guide), box(src)
    a = (box(guide * src) - mean_g * mean_s) / (box(guide * guide) - mean_g**2 + eps)
    b = mean_s - a * mean_g
    return box(a) * guide + box(b)


def resize_to(a: np.ndarray, h: int, w: int, order: int = 1) -> np.ndarray:
    return zoom(a, (h / a.shape[0], w / a.shape[1]), order=order, grid_mode=True, mode="nearest")[:h, :w]


# --- photo -----------------------------------------------------------------------------------


def upscale(img: Image.Image, factor: float, model_path: Path, tile: int = 256, pad: int = 16) -> Image.Image:
    """Real-ESRGAN x4 in overlapping tiles, then Lanczos down to the requested factor."""
    import torch
    from spandrel import ModelLoader

    device, _ = device_and_dtype()
    model = ModelLoader().load_from_file(str(model_path)).eval().to(device)
    s = model.scale
    x = torch.from_numpy(to_float(img).astype(np.float32)).permute(2, 0, 1)[None].to(device)
    _, _, h, w = x.shape
    out = torch.zeros((1, 3, h * s, w * s), device=device)
    with torch.no_grad():
        for y0 in range(0, h, tile):
            for x0 in range(0, w, tile):
                y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
                py0, px0 = max(y0 - pad, 0), max(x0 - pad, 0)
                py1, px1 = min(y1 + pad, h), min(x1 + pad, w)
                part = model(x[:, :, py0:py1, px0:px1])
                out[:, :, y0 * s : y1 * s, x0 * s : x1 * s] = part[
                    :, :, (y0 - py0) * s : (y1 - py0) * s, (x0 - px0) * s : (x1 - px0) * s
                ]
    big = to_image(out[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy())
    size = (round(img.width * factor), round(img.height * factor))
    return big if size == big.size else big.resize(size, Image.LANCZOS)


# --- delighting ------------------------------------------------------------------------------


def marigold_albedo(img: Image.Image, model_id: str, steps: int, ensemble: int) -> Image.Image:
    """Albedo from Marigold intrinsic decomposition, at the given image's size."""
    import torch
    from diffusers import MarigoldIntrinsicsPipeline

    device, dtype = device_and_dtype()
    pipe = MarigoldIntrinsicsPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
    out = pipe(
        img,
        num_inference_steps=steps,
        ensemble_size=ensemble,
        processing_resolution=max(img.size) // 8 * 8,  # Marigold needs a multiple of 8
        generator=torch.Generator(device=device).manual_seed(0),  # same image -> same albedo
    )
    vis = pipe.image_processor.visualize_intrinsics(out.prediction, pipe.target_properties)[0]
    return vis["albedo"].resize(img.size, Image.LANCZOS)


def transfer_albedo(source: Image.Image, raw_albedo: Image.Image, photo: Image.Image, z: np.ndarray):
    """Divide the sharp photo by the baked lighting Marigold found at reduced size.

    Marigold's albedo smears fine texture, but the lighting it removes (photo / albedo) is smooth,
    so upsampling that ratio and dividing it out of the full-size photo keeps the photo's detail.
    """
    src = srgb_to_linear(to_float(source))
    alb = srgb_to_linear(to_float(raw_albedo))
    pho = srgb_to_linear(to_float(photo))

    def up(a: np.ndarray, order: int = 3) -> np.ndarray:
        f = (photo.height / a.shape[0], photo.width / a.shape[1]) + (1,) * (a.ndim - 2)
        return zoom(a, f, order=order, grid_mode=True, mode="nearest")[: photo.height, : photo.width]

    # Brightness of the baked light, in log space. Per-channel ratios amplify JPEG noise into colour speckle.
    log_s = np.log(np.maximum(luminance(src), 0.01)) - np.log(np.maximum(luminance(alb), 0.01))
    # Snap its strong edges (cast shadows) to the sharp photo, or misaligned shadow boundaries leave dark lines.
    # Wide radius and high eps: only big edges transfer. With a tight filter the photo's fine detail leaks
    # into the shading, and dividing it out strips the crevice shadows that make rock look crisp.
    radius = max(2, round(16 * photo.width / 3840))
    log_s = guided_filter(np.log(np.maximum(luminance(pho), 0.01)), up(log_s), radius=radius, eps=0.3)
    # Colour of the baked light (warm sun, blue sky fill) changes slowly: take it from a heavily blurred ratio.
    # Smoothed with depth as the guide, so it never crosses a silhouette: a plain blur across the
    # opening tinted the far canyon blue-grey from the near rock and the near rims orange from the far canyon.
    zs = resize_to(z, source.height, source.width)
    depth_guide = np.log(np.maximum(zs, 1e-3))
    ratio = src / np.maximum(alb, 1e-3)
    tint = np.stack([guided_filter(depth_guide, ratio[..., c], radius=12, eps=1e-3) for c in range(3)], axis=-1)
    tint = np.maximum(tint, 1e-3)
    tint /= np.maximum(luminance(tint), 1e-3)[..., None]
    shading = np.exp(log_s)[..., None] * np.clip(up(tint, order=1), 0.3, 3.0)
    albedo = pho / np.clip(shading, 0.02, 20.0)
    return to_image(linear_to_srgb(albedo)), to_image(linear_to_srgb(shading / np.percentile(shading, 99.5)))


# --- depth -----------------------------------------------------------------------------------


def relative_disparity(img: Image.Image, model_id: str, res: int) -> np.ndarray:
    """Depth Anything V2 inverse depth (relative, larger = nearer) at the image's size.

    `res` is the processing height. Around 1000 keeps surface detail and smooth edges; the default 518
    gives staircase edges, and much higher starts to break up large shapes.
    """
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    device, _ = device_and_dtype()
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).eval().to(device)
    h = res // 14 * 14
    w = round(res * img.width / img.height / 14) * 14
    x = processor(images=img, return_tensors="pt", size={"height": h, "width": w}, keep_aspect_ratio=False)
    with torch.no_grad():
        out = model(pixel_values=x["pixel_values"].to(device)).predicted_depth.float()
        out = torch.nn.functional.interpolate(out[:, None], size=(img.height, img.width), mode="bicubic")
    return out[0, 0].cpu().numpy().astype(np.float64)


def metric_depth(img: Image.Image, model_id: str) -> tuple[np.ndarray, float]:
    """Depth Pro: metric depth in metres and the horizontal field of view in degrees."""
    import torch
    from transformers import DepthProForDepthEstimation, DepthProImageProcessor

    device, dtype = device_and_dtype()
    processor = DepthProImageProcessor.from_pretrained(model_id)
    model = DepthProForDepthEstimation.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    with torch.no_grad():
        out = model(**processor(images=img, return_tensors="pt").to(device, dtype))
    post = processor.post_process_depth_estimation(out, target_sizes=[(img.height, img.width)])[0]
    return post["predicted_depth"].float().cpu().numpy().astype(np.float64), float(post["field_of_view"])


def sky_probability(img: Image.Image, model_id: str) -> np.ndarray:
    """SegFormer (ADE20K) probability that each pixel is sky."""
    import torch
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

    device, _ = device_and_dtype()
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(device).eval()
    sky_id = model.config.label2id["sky"]
    with torch.no_grad():
        logits = model(**processor(images=img, return_tensors="pt").to(device)).logits
        logits = torch.nn.functional.interpolate(logits, size=(img.height, img.width), mode="bilinear")
        prob = logits.softmax(dim=1)[0, sky_id]
    return prob.cpu().numpy().astype(np.float64)


def fuse_depth(rel: np.ndarray, metric: np.ndarray, sky_prob: np.ndarray, photo: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Relative depth's shape and detail, put on Depth Pro's metric scale.

    Returns (inverse depth, sky mask, distant-land mask). Distant land is what Depth Anything puts as far
    away as the sky (distant mountains), where Depth Pro tends to flatten it onto the plateau in front.
    """
    guide = np.log(np.maximum(luminance(photo), 0.005))
    # Sky edges follow the photo's edges.
    sky = guided_filter(guide, sky_prob, radius=4, eps=0.01) > 0.5
    land = ~sky

    rel_n = (rel - rel.min()) / max(np.ptp(rel), 1e-9)
    inv_metric = 1.0 / np.maximum(metric, 1e-3)
    # Fit rel_n -> 1/metric on land where the relative model hasn't flattened out (it bottoms out far away).
    fit = land & (rel_n > 0.09)
    idx = np.flatnonzero(fit)
    idx = idx[np.linspace(0, len(idx) - 1, min(len(idx), 300_000)).astype(int)]
    x, y = rel_n.ravel()[idx], inv_metric.ravel()[idx]
    keep = np.ones_like(x, dtype=bool)
    for _ in range(3):  # trim the worst fits: the two models disagree badly in places
        a, b = np.polyfit(x[keep], y[keep], 1)
        r = np.abs(a * x + b - y)
        keep = r <= np.percentile(r, 80)
    inv = a * rel_n + b
    # Where the relative model flattens out (far canyon, mountains), use Depth Pro.
    w = np.clip((rel_n - 0.045) / (0.09 - 0.045), 0, 1)
    w = w * w * (3 - 2 * w)
    inv = w * inv + (1 - w) * inv_metric
    # Snap depth edges to the photo's edges.
    inv = guided_filter(guide, inv, radius=6, eps=0.02 * np.var(guide))
    land_far = np.percentile(1.0 / np.maximum(inv[land], 1e-6), 99.5)
    inv = np.maximum(inv, 1.0 / (land_far * 1.2))
    # Distant land: rel at the sky's level, in regions that touch the sky.
    from scipy.ndimage import binary_dilation, label

    sky_level = float(np.median(rel_n[sky])) if sky.any() else 0.0
    candidate = land & (rel_n < sky_level * 1.1 + 1e-4)
    regions, count = label(candidate)
    touching = np.unique(regions[binary_dilation(sky, iterations=3) & candidate])
    distant = np.isin(regions, touching[touching > 0])
    return inv, sky, distant


def estimate_haze(photo: np.ndarray, z: np.ndarray, sky: np.ndarray) -> tuple[np.ndarray, float]:
    """Atmospheric haze from the photo and depth: airlight colour, and extinction per scene unit.

    Haze adds airlight in proportion to (1 - t), with transmission t = exp(-beta * z). Dark channel
    prior: haze-free rock almost always has some channel near zero in a small patch, so the darkest
    channel (relative to the airlight) at each distance tells how much haze is in front of it.
    """
    from scipy.ndimage import minimum_filter

    land = ~sky
    size = max(3, round(15 * photo.shape[1] / 3840))
    # Airlight: the most haze-dominated pixels are those whose darkest channel is brightest (He et al.).
    # Taken from land only, so a blue sky doesn't tint it.
    dark0 = minimum_filter(photo.min(axis=-1), size=size)
    hazy = land & (dark0 >= np.percentile(dark0[land], 99.9))
    airlight = photo[hazy].mean(axis=0)
    dark = minimum_filter((photo / np.maximum(airlight, 1e-3)).min(axis=-1), size=size)
    edges = np.geomspace(max(z[land].min(), 1e-3), z[land].max(), 25)
    zs, ts, ws = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = land & (z >= lo) & (z < hi)
        if m.sum() < 500:
            continue
        zs.append(np.sqrt(lo * hi))
        ts.append(1 - 0.95 * np.percentile(dark[m], 10))
        ws.append(np.sqrt(m.sum()))
    if not zs:
        return airlight, 0.0
    zs, ts, ws = map(np.array, (zs, ts, ws))
    betas = np.geomspace(1e-4, 2.0, 400)
    err = [(ws * (ts - np.exp(-b * zs)) ** 2).sum() for b in betas]
    return airlight, float(betas[int(np.argmin(err))])


def dehaze(photo: np.ndarray, z: np.ndarray, sky: np.ndarray, airlight: np.ndarray, beta: float) -> np.ndarray:
    t = np.exp(-beta * z)[..., None]
    clear = (photo - airlight * (1 - t)) / np.maximum(t, 0.15)
    clear = np.clip(clear, 0, 1)
    clear[sky] = photo[sky]
    return clear


def compute_normals(z: np.ndarray, tan_half_fov: float, relief: float) -> np.ndarray:
    """View-space normals from depth. x right, y up, z into the scene (facing the camera = negative z).

    At depth edges, each pixel takes the one-sided difference on its own side of the edge,
    so silhouettes don't turn into bright bands of sideways-facing surface.
    """
    h, w = z.shape
    aspect = w / h
    xs = (np.arange(w) + 0.5) / w * 2.0 - 1.0
    ys = 1.0 - (np.arange(h) + 0.5) / h * 2.0
    ndc_x, ndc_y = np.meshgrid(xs, ys)
    p = np.stack([ndc_x * tan_half_fov * aspect * z, ndc_y * tan_half_fov * z, z], axis=-1)

    def one_sided(axis: int) -> np.ndarray:
        fwd = np.roll(p, -1, axis=axis) - p
        bwd = p - np.roll(p, 1, axis=axis)
        use_fwd = np.abs(fwd[..., 2]) < np.abs(bwd[..., 2])
        return np.where(use_fwd[..., None], fwd, bwd)

    dpdx, dpdy = one_sided(1), one_sided(0)
    # Single-image depth tends to tilt cliff faces back; relief < 1 flattens depth slopes toward the camera.
    dpdx[..., 2] *= relief
    dpdy[..., 2] *= relief
    n = np.cross(dpdx, dpdy)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    n *= np.where(n[..., 2:3] > 0, -1.0, 1.0)
    return n


def marigold_normals(img: Image.Image, model_id: str, res: int, tile: int, steps: int, ensemble: int) -> np.ndarray:
    """Normals estimated from the photo itself (Marigold), in view space: x right, y up, z into the scene.

    Normals from depth slopes amplify every depth error into speckle; a model that looks at the
    photo's shading and texture gives clean surface detail and no bands at silhouettes.

    The model is trained on ~768 px images, so one pass over a 4K image only has ~1536 px of detail.
    Two passes: the whole image at `res` for large-scale orientation (tiles can't agree on which way
    a wall faces, and each tile sees itself as a centred photo), plus overlapping `tile`-sized tiles
    at full resolution for fine form. Result = low frequencies of the whole + high frequencies of the tiles.
    """
    import torch
    from diffusers import MarigoldNormalsPipeline

    device, dtype = device_and_dtype()
    pipe = MarigoldNormalsPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=True)

    def run(im: Image.Image, proc_res: int) -> np.ndarray:
        out = pipe(
            im,
            num_inference_steps=steps,
            ensemble_size=ensemble,
            processing_resolution=proc_res // 8 * 8,
            generator=torch.Generator(device=device).manual_seed(0),  # deterministic
        )
        n = np.asarray(out.prediction[0], dtype=np.float64)  # x right, y up, z toward the camera
        return np.stack([resize_to(n[..., c], im.height, im.width, order=3) for c in range(3)], axis=-1)

    whole = run(img, min(res, max(img.size)))
    if tile <= 0 or max(img.size) <= res:
        n = whole
    else:
        tile = min(tile, img.width, img.height)
        overlap = tile // 4
        stride = tile - overlap

        def starts(length: int) -> list[int]:
            xs = list(range(0, max(length - tile, 0) + 1, stride))
            if xs[-1] + tile < length:
                xs.append(length - tile)
            return xs

        # Feathered weights so tile seams blend.
        ramp = np.minimum(np.arange(tile) + 1, overlap) / overlap
        ramp = np.minimum(ramp, ramp[::-1])
        weight = np.outer(ramp, ramp)[..., None]
        acc = np.zeros((img.height, img.width, 3))
        wsum = np.zeros((img.height, img.width, 1))
        boxes = [(x, y) for y in starts(img.height) for x in starts(img.width)]
        for i, (x, y) in enumerate(boxes):
            print(f"  normals tile {i + 1}/{len(boxes)}", end="\r")
            acc[y : y + tile, x : x + tile] += weight * run(img.crop((x, y, x + tile, y + tile)), tile)
            wsum[y : y + tile, x : x + tile] += weight
        print()
        tiles = acc / np.maximum(wsum, 1e-9)
        # Split at about a quarter tile: above that, trust the whole-image pass.
        sigma = tile / 24
        low = lambda a: np.stack([gaussian_filter(a[..., c], sigma) for c in range(3)], axis=-1)
        n = low(whole) + (tiles - low(tiles))
    # A whisker of blur takes off the model's brush-stroke grain without losing form.
    n = np.stack([gaussian_filter(n[..., c], 1.0 * img.width / 3840) for c in range(3)], axis=-1)
    n[..., 2] *= -1  # to scene space: z into the scene
    return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12)


def estimate_up(normal_png: Path, z: np.ndarray, sky: np.ndarray) -> list[float]:
    """True vertical in scene space (x right, y up, z into the scene), from the flat ground in the picture:
    canyon floors and plateau tops are level in reality, so their average normal is "up". A camera looking
    down tilts it toward the camera (negative z). Only ground beyond the median distance counts: the floor
    right around the camera (a cave, a ledge) is often not level."""
    n = np.asarray(Image.open(normal_png).convert("RGB")).astype(np.float32) / 127.5 - 1
    n[..., 2] *= -1  # standard colours have z toward the camera
    land = ~sky
    flat = land & (n[..., 1] > 0.75) & (z > np.median(z[land]))
    if flat.sum() < 1000:
        return [0.0, 1.0, 0.0]
    up = n[flat].mean(0)
    # Keep only the pitch: sloping plateaus skew the sideways part, and photos are nearly always level
    # side to side.
    up[0] = 0
    up /= np.linalg.norm(up)
    return [round(float(c), 4) for c in up]


def save_normals(n: np.ndarray, sky: np.ndarray, path: Path) -> None:
    # Standard colours: flip z so a surface facing the camera is blue (128, 128, 255).
    std = np.stack([n[..., 0], n[..., 1], -n[..., 2]], axis=-1)
    std[sky] = (0.0, 0.0, 1.0)
    Image.fromarray(np.round((std * 0.5 + 0.5) * 255).astype(np.uint8)).save(path)


# --- main ------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", type=Path)
    ap.add_argument("--out", type=Path, help="scene folder (default: the image's folder)")
    ap.add_argument("--upscale", type=float, default=2.0, help="Real-ESRGAN upscale factor (1 = none, e.g. after Topaz)")
    ap.add_argument("--max-width", type=int, default=3840, help="working width cap: 3840 = 1:1 on a 4K TV")
    ap.add_argument("--redo-photo", action="store_true", help="recompute photo.png even if it exists")
    ap.add_argument("--depth-res", type=int, default=1022, help="Depth Anything processing height")
    ap.add_argument("--distant-push", type=float, default=4.0, help="how much further back distant mountains go (1 = as estimated)")
    ap.add_argument("--hand-distant", action="store_true", help="use a hand-painted distant.png (white = distant land) instead of the automatic mask")
    ap.add_argument("--fov", type=float, help="vertical field of view in degrees (default: Depth Pro's estimate)")
    ap.add_argument("--normals", choices=["model", "depth"], default="model", help="Marigold normals, or slopes of the depth map")
    ap.add_argument("--normals-res", type=int, default=1536, help="longest side the normals model sees")
    ap.add_argument("--normals-tile", type=int, default=1024, help="tile size for full-resolution normals (0 = one pass)")
    ap.add_argument("--redo-normals", action="store_true", help="recompute normals_raw.npy even if it exists")
    ap.add_argument("--relief", type=float, default=0.6, help="with --normals depth: depth slope strength (1 = as estimated)")
    ap.add_argument("--delight-res", type=int, default=1536, help="longest side the delight model sees")
    ap.add_argument("--delight-steps", type=int, default=8)
    ap.add_argument("--delight-ensemble", type=int, default=5, help="more = steadier albedo, slower")
    ap.add_argument("--redo-delight", action="store_true", help="recompute delight_raw.png even if it exists")
    ap.add_argument("--from-depth", action="store_true", help="skip the depth models and reuse an edited depth.png")
    args = ap.parse_args()

    out_dir = args.out or args.image.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    source = Image.open(args.image).convert("RGB")
    scene_json = out_dir / "scene.json"
    previous = json.loads(scene_json.read_text()) if scene_json.exists() else {}

    # Photo: optional upscale, then cap the working width.
    photo_png = out_dir / "photo.png"
    width = min(round(source.width * args.upscale), args.max_width)
    expected = (width, round(width * source.height / source.width))
    photo = Image.open(photo_png).convert("RGB") if photo_png.exists() else None
    if args.redo_photo or photo is None or photo.size != expected or previous.get("source") != args.image.name:
        photo = source
        if args.upscale > 1:
            print(f"Upscaling x{args.upscale} with {UPSCALE_MODEL.name} ...")
            photo = upscale(source, args.upscale, UPSCALE_MODEL)
        if photo.size != expected:
            photo = photo.resize(expected, Image.LANCZOS)
        photo.save(photo_png)
    w, h = photo.size

    # Depth.
    depth_png, sky_png = out_dir / "depth.png", out_dir / "sky.png"
    if args.from_depth:
        disp = np.asarray(Image.open(depth_png), dtype=np.float64) / 65535.0
        if disp.shape != (h, w):
            raise SystemExit(f"depth.png is {disp.shape[::-1]}, photo.png is {(w, h)}")
        sky = np.asarray(Image.open(sky_png).convert("L")) > 127
        near, far, fov = previous["near"], previous["far"], previous["fovDeg"]
        inv = disp * (1 / near - 1 / far) + 1 / far
    else:
        print(f"Estimating depth: {DEPTH_MODEL} at {args.depth_res}, {METRIC_MODEL}, sky from {SKY_MODEL} ...")
        rel = relative_disparity(photo, DEPTH_MODEL, args.depth_res)
        metric, fov_x = metric_depth(photo, METRIC_MODEL)
        sky_prob = sky_probability(photo, SKY_MODEL)
        inv, sky, distant = fuse_depth(rel, metric, sky_prob, srgb_to_linear(to_float(photo)))
        far_png = out_dir / "distant.png"
        if far_png.exists() and args.hand_distant:
            distant = np.asarray(Image.open(far_png).convert("L").resize((w, h))) > 127  # hand-painted override
        else:
            Image.fromarray(distant.astype(np.uint8) * 255).save(far_png)
        if distant.any() and args.distant_push != 1:
            # Push distant mountains back, with a soft edge so the mesh doesn't tear at the boundary.
            soft = guided_filter(np.log(np.maximum(luminance(srgb_to_linear(to_float(photo))), 0.005)), distant.astype(np.float64), radius=4, eps=0.01)
            soft = np.clip(soft, 0, 1)
            inv = inv / (1 + (args.distant_push - 1) * soft)
            print(f"  distant land: {distant.mean() * 100:.1f}% of the picture, pushed back x{args.distant_push}")
        fov = 2 * np.degrees(np.arctan(np.tan(np.radians(fov_x) / 2) * h / w))
        near = float(1.0 / inv[~sky].max())
        far = float(1.0 / inv[~sky].min()) * 1.5  # sky sits beyond the farthest land
        disp = np.clip((inv - 1 / far) / (1 / near - 1 / far), 0, 1)
        disp[sky] = 0
        Image.fromarray(np.round(disp * 65535).astype(np.uint16)).save(depth_png)
        Image.fromarray(sky.astype(np.uint8) * 255).save(sky_png)
    if args.fov:
        fov = args.fov

    z = 1.0 / np.maximum(inv, 1 / far)
    z[sky] = far
    tan_half_fov = float(np.tan(np.radians(fov) / 2))
    if args.normals == "model":
        raw = out_dir / "normals_raw.npy"
        if args.redo_normals or not raw.exists() or np.load(raw, mmap_mode="r").shape[:2] != (h, w):
            print(f"Estimating normals with {NORMALS_MODEL} ...")
            np.save(raw, marigold_normals(photo, NORMALS_MODEL, args.normals_res, args.normals_tile, args.delight_steps, args.delight_ensemble).astype(np.float32))
        n = np.load(raw).astype(np.float64)
    else:
        n = compute_normals(z, tan_half_fov, args.relief)
    save_normals(n, sky, out_dir / "normal.png")
    up = estimate_up(out_dir / "normal.png", z, sky)
    print(f"  up (from flat ground): {up}, camera pitch {np.degrees(np.arctan2(-up[2], up[1])):+.1f} deg (+ = looking down)")
    z.astype("<f4").tofile(out_dir / "depth.bin")

    # Haze: estimated from depth, removed before delighting (the renderer adds it back to lit areas).
    photo_lin = srgb_to_linear(to_float(photo))
    airlight, beta = estimate_haze(photo_lin, z, sky)
    clear = dehaze(photo_lin, z, sky, airlight, beta)
    to_image(linear_to_srgb(clear)).save(out_dir / "dehazed.png")
    clear_img = Image.open(out_dir / "dehazed.png")
    haze = {"airlight": [round(float(c), 4) for c in airlight], "beta": round(float(beta), 5)}
    print(f"  haze: airlight {haze['airlight']}, beta {haze['beta']} per unit (half-visibility at {np.log(2) / max(beta, 1e-9):.1f})")

    # Albedo, from the dehazed photo.
    shrink = min(1.0, args.delight_res / max(w, h))
    delight_src = clear_img.resize((round(w * shrink), round(h * shrink)), Image.LANCZOS)
    raw_png = out_dir / "delight_raw.png"
    stale = previous.get("haze") != haze or previous.get("source") != args.image.name
    if args.redo_delight or stale or not raw_png.exists() or Image.open(raw_png).size != delight_src.size:
        print(f"Removing baked lighting with {DELIGHT_MODEL} at {delight_src.size} ...")
        marigold_albedo(delight_src, DELIGHT_MODEL, args.delight_steps, args.delight_ensemble).save(raw_png)
    albedo, shading = transfer_albedo(delight_src, Image.open(raw_png), clear_img, z)
    albedo.save(out_dir / "albedo.png")
    shading.save(out_dir / "shading.png")

    scene_json.write_text(
        json.dumps(
            {
                "source": args.image.name,
                "image": photo_png.name,
                "albedo": "albedo.png",
                "width": w,
                "height": h,
                "depth": "depth.bin",
                "normal": "normal.png",
                "normalEncoding": "standard",
                "near": round(near, 4),
                "far": round(far, 4),
                "fovDeg": round(float(fov), 2),
                "relief": args.relief,
                "up": up,
                "haze": haze,
            },
            indent=2,
        )
    )
    print(f"Wrote {out_dir}  ({w}x{h}, fov {fov:.1f} deg, depth {near:.2f}..{far / 1.5:.2f} + sky at {far:.2f})")


if __name__ == "__main__":
    main()
