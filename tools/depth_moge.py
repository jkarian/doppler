"""Rebuild a scene's depth (and optionally normals) with MoGe-2, keeping everything else.

    python tools/depth_moge.py scenes/canyon scenes/canyon-moge [--normals moge|keep]

MoGe-2 (Microsoft, MIT licence) gives crisper silhouettes and cleaner layering than the Depth Anything +
Depth Pro fusion in scene_prep.py, without its radial streaks (see docs/research-camera-mapping.md). It
squashes the far range, though (far/near about 36 against our 83), so its depths are mapped onto the
existing scene's depth distribution: MoGe decides which rock is in front of which and how surfaces are
shaped; the old scene decides how far away the far canyon and the mountains are. Sky stays as before.

The new scene folder reuses the old one's photo and albedo (by relative path) and keeps its camera
(field of view), so placed lights and rigs land in the same spots. Its "up" is measured again.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from scene_prep import estimate_up, save_normals  # noqa: E402

MODEL = "Ruicheng/moge-2-vitl-normal"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", type=Path, help="existing scene folder")
    ap.add_argument("out", type=Path, help="new scene folder")
    ap.add_argument("--normals", choices=["moge", "keep"], default="keep", help="MoGe's normals, or keep the scene's")
    ap.add_argument("--resolution-level", type=int, default=9, help="MoGe detail level 0-9")
    args = ap.parse_args()

    info = json.loads((args.scene / "scene.json").read_text())
    w, h, far = info["width"], info["height"], info["far"]
    old = np.fromfile(args.scene / info["depth"], dtype="<f4").reshape(h, w).astype(np.float64)
    sky = old >= far * 0.98
    photo = np.asarray(Image.open(args.scene / info["image"]).convert("RGB"))

    from moge.model.v2 import MoGeModel

    print(f"MoGe-2 on {w}x{h} ...")
    model = MoGeModel.from_pretrained(MODEL).cuda().eval()
    with torch.no_grad():
        out = model.infer(torch.tensor(photo / 255.0, dtype=torch.float32, device="cuda").permute(2, 0, 1), resolution_level=args.resolution_level, use_fp16=True)
    z = out["points"][..., 2].float().cpu().numpy().astype(np.float64)  # planar depth, like depth.bin
    valid = out["mask"].cpu().numpy() & np.isfinite(z) & (z > 0)
    normal = out["normal"].float().cpu().numpy() if "normal" in out else None
    del model, out
    torch.cuda.empty_cache()

    # MoGe's shape on the old scene's range: match the distributions of log depth over the land.
    land = ~sky & valid
    q = np.linspace(0, 1, 2001)
    src = np.quantile(np.log(z[land]), q)
    dst = np.quantile(np.log(old[land]), q)
    new = old.copy()
    new[land] = np.exp(np.interp(np.log(z[land]), src, dst))
    new[sky] = far
    print(f"  land {land.mean() * 100:.1f}% of the picture; MoGe far/near {np.exp(src[-21] - src[20]):.0f}, mapped {np.exp(dst[-21] - dst[20]):.0f}")
    print(f"  {(~sky & ~valid).mean() * 100:.2f}% of land MoGe left out: kept the old depth there")

    args.out.mkdir(parents=True, exist_ok=True)
    new.astype("<f4").tofile(args.out / "depth.bin")
    rel = lambda name: str(Path("..") / args.scene.name / name).replace("\\", "/")
    if args.normals == "moge" and normal is not None:
        # MoGe: x right, y down, z forward. Scene space: x right, y up, z forward.
        save_normals(normal * np.array([1.0, -1.0, 1.0]), sky, args.out / "normal.png")
        normal_name = "normal.png"
    else:
        normal_name = rel(info["normal"])
    up = estimate_up(args.out / normal_name if args.normals == "moge" else args.scene / info["normal"], new, sky)
    scene = {**info, "image": rel(info["image"]), "albedo": rel(info.get("albedo", info["image"])), "normal": normal_name,
             "depth": "depth.bin", "up": up, "depthModel": "MoGe-2 (mapped onto " + args.scene.name + ")"}
    (args.out / "scene.json").write_text(json.dumps(scene, indent=2))
    print(f"Wrote {args.out} (normals: {args.normals}, up {up})")


if __name__ == "__main__":
    main()
