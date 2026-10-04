"""Fly around a trained splat in the browser, rendered by gsplat (the renderer that trained it).

    bash tools/world/run.sh tools/world/view_splat.py scenes/canyon/world/splat      then open http://localhost:8080

Starts at the photo's camera. Drag to orbit, right-drag to pan, scroll to move. viser serves the page.
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import viser
import viser.transforms as vtf
from gsplat import rasterization


def load_ply(path: Path) -> dict:
    """A standard 3D Gaussian Splatting .ply (x y z, f_dc_*, f_rest_*, opacity, scale_*, rot_*), e.g. from Matrix-3D."""
    with open(path, "rb") as f:
        names, n = [], 0
        while (line := f.readline().decode().strip()) != "end_header":
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                names.append(line.split()[-1])
        data = np.frombuffer(f.read(n * 4 * len(names)), dtype="<f4").reshape(n, len(names))
    col = {k: data[:, i] for i, k in enumerate(names)}
    t = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32)
    rest = sorted((k for k in names if k.startswith("f_rest_")), key=lambda k: int(k.split("_")[-1]))
    sh0 = t(np.stack([col[f"f_dc_{i}"] for i in range(3)], 1))[:, None, :]
    shN = t(np.stack([col[k] for k in rest], 1)).reshape(n, 3, -1).transpose(1, 2) if rest else torch.zeros(n, 0, 3)
    return {
        "means": t(np.stack([col["x"], col["y"], col["z"]], 1)), "quats": t(np.stack([col[f"rot_{i}"] for i in range(4)], 1)),
        "scales": t(np.stack([col[f"scale_{i}"] for i in range(3)], 1)), "opacities": t(col["opacity"]), "sh0": sh0, "shN": shN,
        "K": np.array([[600, 0, 640], [0, 600, 360], [0, 0, 1]], np.float64), "size": (1280, 720),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="a splat folder (splat.pt from splat_train.py) or a .ply file")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--snap", type=int, default=0, help="instead of serving: save this many views turning around the origin, then exit")
    args = ap.parse_args()

    s = load_ply(args.folder) if args.folder.suffix == ".ply" else torch.load(args.folder / "splat.pt", weights_only=False)
    dev = "cuda"
    means, quats = s["means"].to(dev), F.normalize(s["quats"].to(dev), dim=-1)
    scales, opac = torch.exp(s["scales"].to(dev)), torch.sigmoid(s["opacities"].to(dev))
    colors = torch.cat([s["sh0"], s["shN"]], 1).to(dev)
    sh_degree = int(round(math.sqrt(colors.shape[1]))) - 1
    K0 = s["K"]
    pw, ph = s["size"]
    fov_y = 2 * np.arctan(ph / 2 / K0[1, 1])

    if args.snap:
        # Views from the origin, turning around the vertical (OpenCV axes: y down), slightly downward.
        out = (args.folder.parent if args.folder.suffix == ".ply" else args.folder) / "snapshots"
        out.mkdir(exist_ok=True)
        Kt = torch.tensor(K0, dtype=torch.float32, device=dev)
        tilt = math.radians(-10)
        Rx = np.array([[1, 0, 0], [0, math.cos(tilt), -math.sin(tilt)], [0, math.sin(tilt), math.cos(tilt)]])
        for k in range(args.snap):
            a = 2 * math.pi * k / args.snap
            Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
            V = np.eye(4, dtype=np.float32)
            V[:3, :3] = (Rx @ Ry).astype(np.float32)
            with torch.no_grad():
                img, _, _ = rasterization(means, quats, scales, opac, colors, torch.tensor(V, device=dev)[None], Kt[None], pw, ph, sh_degree=sh_degree)
            from PIL import Image

            Image.fromarray((img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(out / f"view_{k:02d}.png")
        print(f"wrote {args.snap} views to {out}")
        return

    server = viser.ViserServer(port=args.port)
    # World is the photo's camera in OpenCV axes (y down); tell viser so "up" is -y.
    server.scene.set_up_direction("-y")

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = (0.0, 0.0, 0.0)
        client.camera.look_at = (0.0, 0.0, 10.0)
        client.camera.up_direction = (0.0, -1.0, 0.0)
        client.camera.fov = float(fov_y)

    last = {}
    while True:
        for cid, client in server.get_clients().items():
            cam = client.camera
            key = (tuple(np.round(cam.position, 5)), tuple(np.round(cam.wxyz, 5)), cam.aspect)
            if last.get(cid) == key:
                continue
            last[cid] = key
            h = 720
            w = int(h * cam.aspect)
            c2w = np.eye(4)
            c2w[:3, :3] = vtf.SO3(cam.wxyz).as_matrix()
            c2w[:3, 3] = cam.position
            V = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=dev)
            f = h / 2 / np.tan(cam.fov / 2)
            K = torch.tensor([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=torch.float32, device=dev)
            with torch.no_grad():
                img, _, _ = rasterization(means, quats, scales, opac, colors, V[None], K[None], w, h, sh_degree=sh_degree)
            client.scene.set_background_image((img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8), format="jpeg")
        time.sleep(1 / 60)


if __name__ == "__main__":
    main()
