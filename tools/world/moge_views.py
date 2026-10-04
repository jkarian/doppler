"""Depth for the photo and the dreamed frames with MoGe, all with the dreamed frames' lens.

    (inside WSL, MoGe venv)  bash tools/world/run_moge.sh tools/world/moge_views.py scenes/canyon spiral [--stride 3]

The splat needs geometry made for the same lens as the dreamed frames. Our scene's depth was built for the
renderer's wider lens (61 degrees vertical against the dreamed frames' 32), which stretches the canyon about 2x
in depth when seen through the dreamed lens. So MoGe runs again here with its field of view fixed to theirs.

Writes <scene>/world/moge_photo.npz (depth at 1920x1080, mask) and <scene>/world/<move>/moge/<k>.npz per frame.
Depth is planar (camera z), in MoGe's units; the photo's depth defines the splat's world units.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODEL = "Ruicheng/moge-2-vitl-normal"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", type=Path)
    ap.add_argument("moves", nargs="+")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--photo-width", type=int, default=1920)
    args = ap.parse_args()

    from moge.model.v2 import MoGeModel

    model = MoGeModel.from_pretrained(MODEL).cuda().eval()

    def depth(im, fov_x):
        t = torch.tensor(np.asarray(im.convert("RGB")) / 255.0, dtype=torch.float32, device="cuda").permute(2, 0, 1)
        with torch.no_grad():
            o = model.infer(t, fov_x=fov_x, resolution_level=9, use_fp16=True)
        return o["depth"].float().cpu().numpy(), o["mask"].cpu().numpy()

    info = json.loads((args.scene / "scene.json").read_text())
    for n, move in enumerate(args.moves):
        d = args.scene / "world" / move
        tf = json.loads((d / "transforms.json").read_text())["frames"]
        f0 = tf[0]
        fov_x = math.degrees(2 * math.atan(f0["cx"] / f0["fl_x"]))
        if n == 0:
            pw = args.photo_width
            photo = Image.open(args.scene / info["image"]).resize((pw, round(pw * info["height"] / info["width"])), Image.LANCZOS)
            z, m = depth(photo, fov_x)
            np.savez_compressed(args.scene / "world" / "moge_photo.npz", depth=z, mask=m, fov_x=fov_x)
            print(f"photo: lens {fov_x:.1f} deg across, depth {np.nanmin(z[m]):.2f} to {np.nanmax(z[m]):.2f}")
        (d / "moge").mkdir(exist_ok=True)
        for k in range(args.stride, len(tf), args.stride):
            z, m = depth(Image.open(d / "samples-rgb" / f"{k:03d}.png"), fov_x)
            np.savez_compressed(d / "moge" / f"{k:03d}.npz", depth=z, mask=m)
        print(f"{move}: {len(range(args.stride, len(tf), args.stride))} frames")


if __name__ == "__main__":
    main()
