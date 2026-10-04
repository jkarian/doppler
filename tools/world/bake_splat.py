"""Render a trained splat from the photo's camera at full size: the visible surface and what's behind it.

    bash tools/world/run.sh tools/world/bake_splat.py scenes/canyon

Writes <scene>/world/bake/:
  front.npz    depth (planar, splat units) and coverage of the visible surface
  front.png    its colour
  behind.npz   depth and coverage of what's behind it: only Gaussians clearly behind the visible surface at their
               own pixel are drawn, which peels off the near layer (the spire, the fin, the cave rims)
  behind.png   its colour
make_splat_scene.py (Windows side) turns these into a scene.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", type=Path)
    ap.add_argument("--behind", type=float, default=1.12, help="a Gaussian counts as behind if this much further than the surface")
    args = ap.parse_args()

    info = json.loads((args.scene / "scene.json").read_text())
    W, H = info["width"], info["height"]
    s = torch.load(args.scene / "world" / "splat" / "splat.pt", weights_only=False)
    dev = "cuda"
    means, quats = s["means"].to(dev), F.normalize(s["quats"].to(dev), dim=-1)
    scales, opac = torch.exp(s["scales"].to(dev)), torch.sigmoid(s["opacities"].to(dev))
    colors = torch.cat([s["sh0"], s["shN"]], 1).to(dev)
    pw, ph = s["size"]
    K = torch.tensor(s["K"], dtype=torch.float32, device=dev) * torch.tensor([[W / pw], [H / ph], [1.0]], device=dev)
    V = torch.eye(4, device=dev)

    def render(keep):
        with torch.no_grad():
            out, alpha, _ = rasterization(means[keep], quats[keep], scales[keep], opac[keep], colors[keep], V[None], K[None],
                                          W, H, sh_degree=3, render_mode="RGB+ED", packed=False)
        rgb = out[0, ..., :3].clamp(0, 1).cpu().numpy()
        return rgb, out[0, ..., 3].cpu().numpy(), alpha[0, ..., 0].cpu().numpy()

    dst = args.scene / "world" / "bake"
    dst.mkdir(exist_ok=True)
    every = torch.ones(len(means), dtype=torch.bool, device=dev)
    rgb, z, a = render(every)
    np.savez_compressed(dst / "front.npz", depth=z, alpha=a)
    Image.fromarray((rgb * 255).astype(np.uint8)).save(dst / "front.png")

    # Peel: keep Gaussians whose depth is clearly beyond the visible surface at the pixel they land on.
    q = means @ K.T
    zc = q[:, 2]
    u = (q[:, 0] / zc.clamp_min(1e-6)).long()
    v = (q[:, 1] / zc.clamp_min(1e-6)).long()
    inside = (zc > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    surf = torch.tensor(z, device=dev)
    zs = torch.full_like(zc, float("inf"))
    zs[inside] = surf[v[inside], u[inside]]
    behind = ~inside | (zc > zs * args.behind)
    rgb_b, z_b, a_b = render(behind)
    np.savez_compressed(dst / "behind.npz", depth=z_b, alpha=a_b)
    Image.fromarray((rgb_b * 255).astype(np.uint8)).save(dst / "behind.png")
    print(f"wrote {dst}: {int(behind.sum())} of {len(means)} Gaussians are behind the visible surface")


if __name__ == "__main__":
    main()
