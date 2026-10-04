"""Make a scene from a baked splat (bake_splat.py): its depth, and its background layer from what the splat has
behind the near rock instead of a painted guess.

    python tools/world/make_splat_scene.py scenes/canyon scenes/canyon-splat

Depth: the splat's visible-surface depth, mapped onto the source scene's distribution of depths over the land
(like depth_moge.py): the splat decides which rock is in front of which and how surfaces are shaped, the source
scene decides how far away things are, so placed lights, rigs and the sun keep working. Sky stays as before.
Background layer: the same band along silhouettes as background_layer.py, filled from the splat's "behind"
render (what's actually behind the spire and the rims, as the dreamed views saw it) where it has coverage, and
by the nearest far colour elsewhere. The new scene reuses the source's photo, albedo and normals by relative path.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from background_layer import WORK_W, fill_opencv, find_band  # noqa: E402
from scene_prep import compute_normals, save_normals  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", type=Path, help="source scene (with world/bake from bake_splat.py)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--sway", type=float, default=0.05)
    ap.add_argument("--margin", type=float, default=1.5)
    ap.add_argument("--bake", type=Path, default=None, help="folder with front/behind renders (default <scene>/world/bake)")
    ap.add_argument("--flow-align", action="store_true",
                    help="warp the renders onto the photo by optical flow first (for a model whose shapes are close but "
                         "not exact, e.g. from Tripo: its edges land on the photo's edges)")
    args = ap.parse_args()

    info = json.loads((args.scene / "scene.json").read_text())
    W, H, far = info["width"], info["height"], info["far"]
    old = np.fromfile(args.scene / info["depth"], dtype="<f4").reshape(H, W).astype(np.float64)
    sky = old >= far * 0.98
    bake = args.bake or args.scene / "world" / "bake"
    fr, bh = np.load(bake / "front.npz"), np.load(bake / "behind.npz")
    zf, af = fr["depth"].astype(np.float64), fr["alpha"]
    zb, ab = bh["depth"].astype(np.float64), bh["alpha"]
    behind_rgb = np.asarray(Image.open(bake / "behind.png").convert("RGB")).astype(np.float32)
    if args.flow_align:
        # Where each photo pixel's content sits in the render (dense optical flow, photo -> render), then pull every
        # render layer back onto the photo's pixels. Nearest for depth and coverage (no blending across edges).
        g1 = cv2.cvtColor(np.asarray(Image.open(args.scene / info["image"]).convert("RGB")), cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(np.asarray(Image.open(bake / "front.png").convert("RGB")), cv2.COLOR_RGB2GRAY)
        s = 1920 / W
        a1, a2 = cv2.resize(g1, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.resize(g2, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        clahe = cv2.createCLAHE(3.0, (8, 8))
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        flow = dis.calc(clahe.apply(a1), clahe.apply(a2), None)
        flow = cv2.resize(flow, (W, H), interpolation=cv2.INTER_LINEAR) / s
        gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        mx, my = gx + flow[..., 0], gy + flow[..., 1]
        warp = lambda a, how=cv2.INTER_NEAREST: cv2.remap(a.astype(np.float32), mx, my, how, borderMode=cv2.BORDER_REPLICATE)
        zf, af, zb, ab = warp(zf).astype(np.float64), warp(af), warp(zb).astype(np.float64), warp(ab)
        behind_rgb = warp(behind_rgb, cv2.INTER_LINEAR)
        print(f"flow-aligned: median shift {np.median(np.hypot(flow[..., 0], flow[..., 1])):.1f} px, 95th percentile "
              f"{np.percentile(np.hypot(flow[..., 0], flow[..., 1]), 95):.1f} px")

    # The splat's depths on the old scene's range: match the distributions of log depth over the land.
    ok = ~sky & (af > 0.5) & (zf > 0)
    q = np.linspace(0, 1, 2001)
    src = np.quantile(np.log(zf[ok]), q)
    dst = np.quantile(np.log(old[ok]), q)

    def remap(z):
        return np.exp(np.interp(np.log(np.maximum(z, 1e-9)), src, dst))

    z = np.where(ok, remap(zf), old)
    z = np.where(sky, old, z).astype(np.float32)

    args.out.mkdir(parents=True, exist_ok=True)
    rel = Path("..") / args.scene.name
    new = dict(info)
    new["image"] = str(rel / info["image"]).replace("\\", "/")
    new["albedo"] = str(rel / info.get("albedo", info["image"])).replace("\\", "/")
    new["normal"] = str(rel / info["normal"]).replace("\\", "/")
    new["depth"] = "depth.bin"
    new["depthModel"] = f"{bake.name} renders (mapped onto {args.scene.name})" + (", flow-aligned" if args.flow_align else "")
    new.pop("background", None)
    z.tofile(args.out / "depth.bin")

    # Background layer: the band along silhouettes the camera's sway can open up.
    tan_half_fov = float(np.tan(np.radians(info["fovDeg"]) / 2))
    aspect = W / H
    # The band comes from the source scene's edges: the splat's depth is too noisy to find silhouettes in.
    land = np.sort(old[~sky].ravel())
    pct = lambda f: float(land[int((len(land) - 1) * f)])
    pivot = pct(0.5)
    span = max(1 / pct(0.02) - 1 / pivot, 1 / pivot - 1 / far)
    baseline = args.sway * tan_half_fov * aspect / span
    h = round(H * WORK_W / W)
    zw = cv2.resize(old.astype(np.float32), (WORK_W, h), interpolation=cv2.INTER_NEAREST).astype(np.float64)
    band, bg, reach, unknown = find_band(zw, zw >= far * 0.98, tan_half_fov, aspect, baseline, args.margin)
    band_full = cv2.resize(band.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    bg_guess = cv2.resize(bg.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)

    # Where the splat saw behind the near rock: its depth (mapped the same way) and colour.
    zb_m = remap(zb)
    seen = band_full & (ab > 0.6) & (zb > 0) & (zb_m > z * 1.02)
    bg_full = np.where(band_full, np.where(seen, zb_m, bg_guess), z).astype(np.float32)
    bg_full.astype("<f4").tofile(args.out / "bg_depth.bin")
    Image.fromarray(band_full.astype(np.uint8) * 255).save(args.out / "bg_mask.png")
    print(f"band covers {band_full.mean() * 100:.1f}% of the picture; the splat saw behind {seen.sum() / max(band_full.sum(), 1) * 100:.0f}% of it")

    behind = behind_rgb
    photo = np.asarray(Image.open(args.scene / info["image"]).convert("RGB")).astype(np.float32)
    albedo = np.asarray(Image.open(args.scene / info.get("albedo", info["image"])).convert("RGB")).astype(np.float32)
    ratio = (albedo[~sky].mean(0) + 1) / (photo[~sky].mean(0) + 1)  # photo -> albedo, roughly
    unknown_full = band_full & ~seen
    for key, img, scale in (("photo", photo, np.ones(3)), ("albedo", albedo, ratio)):
        small = cv2.resize(img, (WORK_W, h), interpolation=cv2.INTER_AREA).astype(np.uint8)
        guess = cv2.resize(fill_opencv(small, cv2.resize(unknown_full.astype(np.uint8), (WORK_W, h), interpolation=cv2.INTER_NEAREST).astype(bool)),
                           (W, H), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
        fill = np.where(seen[..., None], np.clip(behind * scale, 0, 255), guess)
        Image.fromarray(np.where(band_full[..., None], fill, img).astype(np.uint8)).save(args.out / f"bg_{key}.png")
    n = compute_normals(bg_full.astype(np.float64), tan_half_fov, 1.0)
    save_normals(n, bg_full >= far * 0.98, args.out / "bg_normal.png")

    new["background"] = {"depth": "bg_depth.bin", "image": "bg_photo.png", "albedo": "bg_albedo.png", "normal": "bg_normal.png",
                         "mask": "bg_mask.png", "fill": "splat"}
    (args.out / "scene.json").write_text(json.dumps(new, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
