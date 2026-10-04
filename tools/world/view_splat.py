"""Fly around a trained splat in the browser, rendered by gsplat (the renderer that trained it).

    bash tools/world/run.sh tools/world/view_splat.py scenes/canyon/world/splat      then open http://localhost:8080

Starts at the photo's camera. Drag to orbit, right-drag to pan, scroll to move. viser serves the page.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import viser
import viser.transforms as vtf
from gsplat import rasterization


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    s = torch.load(args.folder / "splat.pt", weights_only=False)
    dev = "cuda"
    means, quats = s["means"].to(dev), F.normalize(s["quats"].to(dev), dim=-1)
    scales, opac = torch.exp(s["scales"].to(dev)), torch.sigmoid(s["opacities"].to(dev))
    colors = torch.cat([s["sh0"], s["shN"]], 1).to(dev)
    K0 = s["K"]
    pw, ph = s["size"]
    fov_y = 2 * np.arctan(ph / 2 / K0[1, 1])

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
                img, _, _ = rasterization(means, quats, scales, opac, colors, V[None], K[None], w, h, sh_degree=3)
            client.scene.set_background_image((img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8), format="jpeg")
        time.sleep(1 / 60)


if __name__ == "__main__":
    main()
