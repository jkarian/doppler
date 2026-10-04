"""Frame a generated vista splat (e.g. Matrix-3D's) as the background behind a photo's near layers, and preview the
composite. The vista doesn't have to match the photo pixel for pixel: it's art-directed, so this renders a grid of view
directions to choose from.

    (WSL, venv: gsplat)  bash tools/world/run.sh tools/world/frame_vista.py scenes/canyon \
        scenes/canyon/world/matrix_vista_hr/generated_3dgs_opt.ply images/markup_layers.webp

Near layers: the markup's frame (dark red) and pillars + spire (dark blue) keep the photo; everything else is the vista,
colour-matched to the photo's own vista (mean and spread per channel in Lab), so the two read as one place.
Writes <scene>/world/vista_frames/: frame_<yaw>_<pitch>.jpg (composite previews) and grid.jpg.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from view_splat import load_ply  # noqa: E402

NEAR = [(139, 0, 0), (0, 80, 140)]  # markup colours kept from the photo: frame, pillars and spire
FAR = [(0, 170, 150), (0, 200, 30), (85, 85, 85)]  # the photo's own vista (for the colour match)


def labels(markup: Path, W: int, H: int, cols) -> np.ndarray:
    m = np.asarray(Image.open(markup).convert("RGB").resize((W, H), Image.NEAREST)).astype(np.float32)
    d = np.min([np.linalg.norm(m - np.array(c, np.float32), axis=-1) for c in cols], axis=0)
    return d < 60


def match_colour(src: np.ndarray, ref: np.ndarray, src_mask: np.ndarray, ref_mask: np.ndarray) -> np.ndarray:
    a = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype(np.float32)
    b = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB).astype(np.float32)
    for c in range(3):
        sm, ss = a[..., c][src_mask].mean(), a[..., c][src_mask].std() + 1e-3
        rm, rs = b[..., c][ref_mask].mean(), b[..., c][ref_mask].std()
        a[..., c] = (a[..., c] - sm) * (rs / ss) + rm
    return cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", type=Path)
    ap.add_argument("splat", type=Path)
    ap.add_argument("markup", type=Path)
    ap.add_argument("--yaws", default="-25,0,25")
    ap.add_argument("--pitches", default="-20,-8")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--at", default="", help="camera position in the splat's world as x,y,z (e.g. along its training path); overrides rise/forward")
    ap.add_argument("--min-dist", type=float, default=0.0, help="leave out Gaussians nearer to the camera than this fraction of the splat's median distance (blurry near rock: our photo covers that range)")
    ap.add_argument("--rise", type=float, default=0.0, help="lift the camera by this fraction of the splat's median distance (to see over a near rim)")
    ap.add_argument("--forward", type=float, default=0.0, help="move the camera forward along its view by this fraction")
    args = ap.parse_args()

    info = json.loads((args.scene / "scene.json").read_text())
    W = args.width
    H = round(W * info["height"] / info["width"])
    photo = np.asarray(Image.open(args.scene / info["image"]).convert("RGB").resize((W, H), Image.LANCZOS))
    near = labels(args.markup, W, H, NEAR)
    far = labels(args.markup, W, H, FAR)
    soft = cv2.GaussianBlur(near.astype(np.float32), (0, 0), 1.5)[..., None]

    s = load_ply(args.splat)
    dev = "cuda"
    means, quats = s["means"].to(dev), F.normalize(s["quats"].to(dev), dim=-1)
    scales, opac = torch.exp(s["scales"].to(dev)), torch.sigmoid(s["opacities"].to(dev))
    colors = torch.cat([s["sh0"], s["shN"]], 1).to(dev)
    shd = int(round(math.sqrt(colors.shape[1]))) - 1
    f = H / 2 / math.tan(math.radians(info["fovDeg"]) / 2)  # the scene's own lens
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=torch.float32, device=dev)

    med = float(means.norm(dim=1).median())
    out = args.scene / "world" / "vista_frames"
    out.mkdir(parents=True, exist_ok=True)
    tiles = []
    for pitch in [float(p) for p in args.pitches.split(",")]:
        for yaw in [float(y) for y in args.yaws.split(",")]:
            a, t = math.radians(yaw), math.radians(pitch)
            Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
            # OpenCV axes (y down): looking down is a positive turn about x.
            Rx = np.array([[1, 0, 0], [0, math.cos(-t), -math.sin(-t)], [0, math.sin(-t), math.cos(-t)]])
            V = np.eye(4, dtype=np.float32)
            V[:3, :3] = (Rx @ Ry).astype(np.float32)
            # Camera position in the splat's world (OpenCV: up is -y), then V = [R | -R c].
            fwd = np.linalg.inv(V[:3, :3]) @ np.array([0, 0, 1.0])
            c = np.array([0, -args.rise * med, 0]) + args.forward * med * np.array([fwd[0], 0, fwd[2]])
            if args.at:
                c = np.array([float(v) for v in args.at.split(",")])
            V[:3, 3] = (-V[:3, :3] @ c).astype(np.float32)
            keep = torch.ones(len(means), dtype=torch.bool, device=dev)
            if args.min_dist > 0:
                keep = (means - torch.tensor(c, dtype=torch.float32, device=dev)).norm(dim=1) > args.min_dist * med
            with torch.no_grad():
                img, _, _ = rasterization(means[keep], quats[keep], scales[keep], opac[keep], colors[keep], torch.tensor(V, device=dev)[None], K[None], W, H, sh_degree=shd)
            vista = (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            vista = match_colour(vista, photo, ~near, far)
            comp = (photo * soft + vista * (1 - soft)).astype(np.uint8)
            at = "_at" + args.at.replace(",", "_") if args.at else ""
            name = f"frame_{int(yaw):+d}_{int(pitch):+d}{at}"
            Image.fromarray(comp).save(out / f"{name}.jpg", quality=92)
            tile = cv2.resize(comp, (W // 3, H // 3))
            cv2.putText(tile, f"yaw {yaw:+.0f}  pitch {pitch:+.0f}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            tiles.append(tile)
    cols = len(args.yaws.split(","))
    rows = [np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)]
    Image.fromarray(np.concatenate(rows, 0)).save(out / "grid.jpg", quality=90)
    print(f"wrote {len(tiles)} framings to {out}")


if __name__ == "__main__":
    main()
