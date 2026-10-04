"""Export a trained splat (splat.pt from splat_train.py) as a standard 3D Gaussian Splatting .ply, for viewers
such as SuperSplat (superspl.at/editor) or any 3DGS viewer.

    bash tools/world/run.sh tools/world/export_ply.py scenes/canyon/world/splat

Coordinates are the photo's camera (x right, y down, z forward) with y and z flipped to y up, so viewers show it
upright, looking down -z from the origin like the photo. Writes splat.ply next to splat.pt.
"""

import argparse
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--max-distance", type=float, default=0, help="drop Gaussians further than this (0 = keep all)")
    args = ap.parse_args()

    s = torch.load(args.folder / "splat.pt", weights_only=False)
    xyz = s["means"].numpy().astype(np.float32)
    q = torch.nn.functional.normalize(s["quats"], dim=-1).numpy().astype(np.float32)  # w, x, y, z
    keep = np.ones(len(xyz), bool)
    if args.max_distance > 0:
        keep = np.linalg.norm(xyz, axis=1) < args.max_distance

    # Flip y and z (a 180-degree turn about x): positions, and the rotations (q' = (0,1,0,0) * q).
    xyz = xyz * np.array([1, -1, -1], np.float32)
    w, x, y, z = q.T
    q = np.stack([-x, w, -z, y], 1)
    # The view-dependent colour bands would need rotating with the turn; keep the base colour only.
    dc = s["sh0"][:, 0, :].numpy().astype(np.float32)
    rest = np.zeros((len(xyz), 45), np.float32)
    op = s["opacities"].numpy().astype(np.float32)[:, None]
    sc = s["scales"].numpy().astype(np.float32)
    normals = np.zeros_like(xyz)

    data = np.concatenate([xyz, normals, dc, rest, op, sc, q], 1)[keep]
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(45)] + \
            ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {len(data)}\n" + \
             "".join(f"property float {n}\n" for n in names) + "end_header\n"
    out = args.folder / "splat.ply"
    with open(out, "wb") as f:
        f.write(header.encode())
        f.write(data.astype("<f4").tobytes())
    print(f"wrote {out}: {len(data)} Gaussians, {out.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
