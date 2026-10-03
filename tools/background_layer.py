"""Background layer for a scene: what's hidden behind near rock, so camera moves don't open dark slivers.

    python tools/background_layer.py scenes/canyon [--fill opencv|sdxl] [--sway 0.05] [--margin 2.5]

A single depth map holds one surface per pixel. When the camera sways or pushes in, near rock slides
over the far rock behind it, and the strip it uncovers was never in the picture: the mesh stretches
across it and it shows as a dark seam. This builds a second layer (a "layered depth image"), only along
those edges:

    1. Depth edges: near rock in front of something much further away (at least 10% further, like the renderer's silhouettes).
    2. The band behind each edge that the largest camera move can uncover. Its width follows the parallax
       between the near and far side: f * baseline * (1/near - 1/far), times a safety margin.
    3. Depth there: the far side's depth, carried in behind the edge.
    4. Colour there (photo and albedo): filled in from the far side. opencv copies the nearest far colour
       (quick, smeary); sdxl uses Stable Diffusion
       XL inpainting for real rock texture (a 7 GB download the first time).
    5. Normals there: from the background depth.

Writes bg_depth.bin, bg_photo.png, bg_albedo.png, bg_normal.png and bg_mask.png into the scene folder and
adds "background" to scene.json. The display draws the layer behind the main one.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
from scene_prep import compute_normals, save_normals  # noqa: E402

WORK_W = 1920  # band finding and filling happen at this width; results are scaled to the scene's size


def find_band(z: np.ndarray, sky: np.ndarray, tan_half_fov: float, aspect: float, baseline: float, margin: float):
    """The band behind near silhouettes, and the background depth to put there."""
    h, w = z.shape
    L = np.log(z)
    f_px = h / (2 * tan_half_fov)
    # Farthest depth around each pixel, out to the widest band we could need.
    reach = int(min(0.12 * w, f_px * baseline * margin * (1 / z[~sky].min()) + 4))
    Lfar = ndi.maximum_filter(L, size=2 * max(reach, 3) + 1)
    # Near side of a depth step the renderer treats as a silhouette (it tears squares spanning more than 8%;
    # 10% here, a little margin). Every such edge needs a band, or it still opens a seam.
    edge = (ndi.maximum_filter(L, size=5) - L) > np.log(1.1)
    edge = ndi.binary_opening(edge, iterations=1)
    dist = ndi.distance_transform_edt(~edge)
    need = f_px * baseline * margin * (1 / z - 1 / np.exp(Lfar))
    band = (dist < np.maximum(need, 3)) & (L < Lfar - np.log(1.1)) & ~sky
    band = ndi.binary_closing(band, iterations=3) & ~sky
    # Background depth: the far side carried in, smoothed so the hidden surface is plausible, not blocky.
    Lbg = cv2.GaussianBlur(Lfar.astype(np.float32), (0, 0), max(2.0, reach / 6))
    bg = np.where(band, np.exp(Lbg), z)
    # Everything nearer than the background around it (the near rock itself) is unknown for the fill, so
    # the band fills from the far side only: what gets uncovered is the far canyon, not the near wall.
    # "Near" is judged locally (about 60 px): in front of what's right beside it. Judged over the whole
    # reach, the canyon walls would count as near next to the sky, and the fill would pull in sky colours.
    Lfar_local = ndi.maximum_filter(L, size=2 * min(reach, 60) + 1)
    near = (L < Lfar_local - np.log(1.25)) & ~sky
    return band, bg, reach, near | band


def fill_opencv(img: np.ndarray, unknown: np.ndarray) -> np.ndarray:
    """Quick fill: each unknown pixel takes the colour of the nearest known (far) pixel, softened. Smeary,
    but the right colours: canyon rock behind canyon edges, sky only where sky is behind."""
    _, (iy, ix) = ndi.distance_transform_edt(unknown, return_indices=True)
    out = img[iy, ix]
    soft = cv2.GaussianBlur(out, (0, 0), 3)
    return np.where(unknown[..., None], soft, img)


def fill_sdxl(img: np.ndarray, band: np.ndarray, unknown: np.ndarray, prompt: str) -> np.ndarray:
    """SDXL inpainting, tile by tile around the band (1024 px windows), blended back. Besides the band, the
    near rock right next to it is hidden from the model too, so it paints what's behind (the far side)
    rather than more of the near rock."""
    import torch
    from diffusers import AutoPipelineForInpainting

    pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to("cuda")
    out = img.copy()
    h, w = band.shape
    T = 1024
    hide = band | (unknown & ndi.binary_dilation(band, iterations=40))
    ys, xs = np.nonzero(band)
    done = np.zeros_like(band)
    gen = torch.Generator("cuda").manual_seed(7)
    while True:
        todo = band & ~done
        if not todo.any():
            break
        y, x = np.argwhere(todo)[0]
        y0 = int(np.clip(y - T // 4, 0, max(0, h - T)))
        x0 = int(np.clip(x - T // 2, 0, max(0, w - T)))
        tile = out[y0:y0 + T, x0:x0 + T]
        tmask = hide[y0:y0 + T, x0:x0 + T]
        res = pipe(prompt=prompt, negative_prompt="people, text, blurry", image=Image.fromarray(tile), mask_image=Image.fromarray(tmask.astype(np.uint8) * 255),
                   height=tile.shape[0], width=tile.shape[1], strength=0.99, num_inference_steps=30, guidance_scale=6.0, generator=gen).images[0]
        res = np.asarray(res.resize((tile.shape[1], tile.shape[0])))
        soft = cv2.GaussianBlur(tmask.astype(np.float32), (0, 0), 3)[..., None]
        out[y0:y0 + T, x0:x0 + T] = (res * soft + tile * (1 - soft)).astype(np.uint8)
        done[y0:y0 + T, x0:x0 + T] = True
    del pipe
    torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", type=Path)
    ap.add_argument("--fill", choices=["opencv", "sdxl"], default="opencv")
    ap.add_argument("--sway", type=float, default=0.05, help="the display's sway amount (largest parallax shift, fraction of half-width)")
    ap.add_argument("--margin", type=float, default=1.5, help="band width safety factor (covers push-ins and the drop's camera moves)")
    ap.add_argument("--prompt", default="sandstone canyon rock wall, natural texture, photo")
    args = ap.parse_args()

    info = json.loads((args.scene / "scene.json").read_text())
    W, H, far = info["width"], info["height"], info["far"]
    z_full = np.fromfile(args.scene / info["depth"], dtype="<f4").reshape(H, W)
    tan_half_fov = float(np.tan(np.radians(info["fovDeg"]) / 2))
    aspect = W / H

    # The camera's largest move (same as the display: sway amount over the parallax span).
    land = np.sort(z_full[z_full < far * 0.98].ravel())
    pct = lambda f: float(land[int((len(land) - 1) * f)])
    pivot = pct(0.5)
    span = max(1 / pct(0.02) - 1 / pivot, 1 / pivot - 1 / far)
    baseline = args.sway * tan_half_fov * aspect / span

    h = round(H * WORK_W / W)
    z = cv2.resize(z_full, (WORK_W, h), interpolation=cv2.INTER_NEAREST).astype(np.float64)
    sky = z >= far * 0.98
    band, bg, reach, unknown = find_band(z, sky, tan_half_fov, aspect, baseline, args.margin)
    print(f"camera move up to {baseline:.4f} units; band covers {band.mean() * 100:.1f}% of the picture (reach {reach}px at {WORK_W})")

    def load(name: str) -> np.ndarray:
        return np.asarray(Image.open(args.scene / name).convert("RGB").resize((WORK_W, h), Image.LANCZOS))

    images = {}
    for key, name in (("photo", info["image"]), ("albedo", info.get("albedo", info["image"]))):
        print(f"filling the {key} ({args.fill}) ...")
        img = load(name)
        if args.fill == "sdxl":
            prompt = args.prompt if key == "photo" else args.prompt + ", flat even lighting, no shadows"
            images[key] = fill_sdxl(img, band, unknown, prompt)
        else:
            images[key] = fill_opencv(img, unknown)

    # Full resolution: the original everywhere but the band.
    band_full = cv2.resize(band.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    bg_full = np.where(band_full, cv2.resize(bg.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR), z_full)
    bg_full.astype("<f4").tofile(args.scene / "bg_depth.bin")
    Image.fromarray(band_full.astype(np.uint8) * 255).save(args.scene / "bg_mask.png")
    for key, name in (("photo", info["image"]), ("albedo", info.get("albedo", info["image"]))):
        orig = np.asarray(Image.open(args.scene / name).convert("RGB"))
        up = np.asarray(Image.fromarray(images[key]).resize((W, H), Image.LANCZOS))
        Image.fromarray(np.where(band_full[..., None], up, orig)).save(args.scene / f"bg_{key}.png")
    n = compute_normals(bg_full.astype(np.float64), tan_half_fov, 1.0)
    save_normals(n, bg_full >= far * 0.98, args.scene / "bg_normal.png")

    info["background"] = {"depth": "bg_depth.bin", "image": "bg_photo.png", "albedo": "bg_albedo.png", "normal": "bg_normal.png", "mask": "bg_mask.png", "fill": args.fill}
    (args.scene / "scene.json").write_text(json.dumps(info, indent=2))
    print(f"Wrote the background layer into {args.scene}")


if __name__ == "__main__":
    main()
