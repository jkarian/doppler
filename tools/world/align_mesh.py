"""Find the camera that sees a 3D model (e.g. from Tripo) the way the scene's photo sees the place, then render what
a scene needs from it: the visible surface and the layer behind it (depth peeling).

    (WSL, Matrix-3D venv: nvdiffrast)  python tools/world/align_mesh.py scenes/canyon 3d/tripo/model.obj [--out tripo]

1. Coarse: render the model's texture from a ring of directions and heights around it; the view whose features match
   the photo best (SIFT, kept by a fundamental-matrix check) is the starting guess.
2. Fine: matched photo pixels get the model's 3D points under them; PnP solves the camera, for a few lens widths,
   keeping the one with the smallest error. Render again from there and repeat.
3. Bake at the photo's size: front.npz (planar depth, coverage), behind.npz and behind.png (the second surface: what's
   behind the near rock, coloured by the model's texture), and preview images. Same format as bake_splat.py, so
   make_splat_scene.py --bake <scene>/world/<out> builds a scene from it.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch
import trimesh
from PIL import Image

dev = "cuda"


def look_at(eye, target, up=(0, 1, 0)):
    """World-to-camera, OpenCV axes (x right, y down, z forward)."""
    eye, target, up = map(np.asarray, (eye, target, up))
    z = target - eye
    z = z / np.linalg.norm(z)
    x = np.cross(z, up)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    V = np.eye(4)
    V[:3, :3] = R
    V[:3, 3] = -R @ eye
    return V


class Renderer:
    def __init__(self, mesh_path):
        m = trimesh.load(mesh_path, process=False, force="mesh")
        self.pos = torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device=dev)
        self.tri = torch.tensor(np.asarray(m.faces), dtype=torch.int32, device=dev)
        uv = np.asarray(m.visual.uv, dtype=np.float32)
        uv[:, 1] = 1 - uv[:, 1]
        self.uv = torch.tensor(uv, device=dev)
        tex = m.visual.material.baseColorTexture if hasattr(m.visual.material, "baseColorTexture") else m.visual.material.image
        self.tex = torch.tensor(np.asarray(tex.convert("RGB")), dtype=torch.float32, device=dev)[None] / 255.0
        self.ctx = dr.RasterizeCudaContext()
        self.center = m.bounds.mean(0)
        self.radius = float(np.linalg.norm(m.extents)) / 2

    def render(self, V, K, w, h, layers=1):
        """Colour, world position and planar depth for up to `layers` surfaces along each pixel (depth peeling)."""
        V = torch.tensor(V, dtype=torch.float32, device=dev)
        near, far = 1e-3, 100.0
        P = torch.zeros(4, 4, device=dev)
        # OpenCV pixels straight to clip space: nvdiffrast's image row 0 is at ndc y = -1, so pixel row v maps to
        # ndc 2v/h - 1 with no flip.
        P[0, 0] = 2 * K[0, 0] / w
        P[1, 1] = 2 * K[1, 1] / h
        P[0, 2] = 2 * K[0, 2] / w - 1
        P[1, 2] = 2 * K[1, 2] / h - 1
        P[2, 2] = (far + near) / (far - near)
        P[2, 3] = -2 * far * near / (far - near)
        P[3, 2] = 1
        ph = torch.cat([self.pos, torch.ones_like(self.pos[:, :1])], 1)
        cam = ph @ V.T
        clip = cam @ P.T
        out = []
        with dr.DepthPeeler(self.ctx, clip[None].contiguous(), self.tri, (h, w)) as peeler:
            for _ in range(layers):
                rast, _ = peeler.rasterize_next_layer()
                uvi, _ = dr.interpolate(self.uv[None], rast, self.tri)
                col = dr.texture(self.tex, uvi, filter_mode="linear")
                xyz, _ = dr.interpolate(self.pos[None], rast, self.tri)
                zc, _ = dr.interpolate(cam[None, :, 2:3].contiguous(), rast, self.tri)
                cov = (rast[..., 3:] > 0).float()
                out.append(((col * cov)[0].cpu().numpy(), xyz[0].cpu().numpy(), (zc * cov)[0, ..., 0].cpu().numpy(), cov[0, ..., 0].cpu().numpy()))
        return out


def K_for(fov_y_deg, w, h):
    f = h / 2 / math.tan(math.radians(fov_y_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)


def match(photo_gray, rgb, sift, mask=None):
    g = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    kp1, d1 = sift.detectAndCompute(photo_gray, None)
    kp2, d2 = sift.detectAndCompute(g, mask)
    if d1 is None or d2 is None or len(kp2) < 10:
        return np.zeros((0, 2)), np.zeros((0, 2))
    good = [a for a, b in cv2.BFMatcher().knnMatch(d1, d2, k=2) if a.distance < 0.8 * b.distance]
    if len(good) < 12:
        return np.zeros((0, 2)), np.zeros((0, 2))
    p1 = np.float64([kp1[a.queryIdx].pt for a in good])
    p2 = np.float64([kp2[a.trainIdx].pt for a in good])
    _, fm = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 2.0, 0.999)
    if fm is None:
        return np.zeros((0, 2)), np.zeros((0, 2))
    fm = fm.ravel().astype(bool)
    return p1[fm], p2[fm]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", type=Path)
    ap.add_argument("mesh", type=Path)
    ap.add_argument("--out", default="tripo")
    ap.add_argument("--work", type=int, default=1024, help="width used for matching")
    args = ap.parse_args()

    info = json.loads((args.scene / "scene.json").read_text())
    W, H = info["width"], info["height"]
    w = args.work
    h = round(w * H / W)
    photo = np.asarray(Image.open(args.scene / info["image"]).convert("RGB").resize((w, h), Image.LANCZOS))
    clahe = cv2.createCLAHE(3.0, (8, 8))
    pg = clahe.apply(cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY))
    sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.01)
    R = Renderer(args.mesh)
    c, rad = R.center, R.radius
    print(f"model: {len(R.tri)} triangles, centre {np.round(c, 3)}, radius {rad:.3f}")

    # 1. Coarse search around the model.
    best = None
    for elev in (0, 15, 30, 50):
        for az in range(0, 360, 20):
            for dist in (1.2, 2.0):
                a, e = math.radians(az), math.radians(elev)
                eye = c + dist * rad * np.array([math.sin(a) * math.cos(e), math.sin(e), math.cos(a) * math.cos(e)])
                V = look_at(eye, c)
                K = K_for(60, w, h)
                rgb, _, _, cov = R.render(V, K, w, h)[0]
                p1, _ = match(pg, rgb, sift)
                if best is None or len(p1) > best[0]:
                    best = (len(p1), V, az, elev, dist)
    print(f"coarse: best view az {best[2]} elev {best[3]} dist {best[4]} with {best[0]} matches")

    # 2. Fine: PnP from matched points, a few lens widths, iterate.
    V = best[1]
    fov = 60.0
    for it in range(4):
        rgb, xyz, _, cov = R.render(V, K_for(fov, w, h), w, h)[0]
        p1, p2 = match(pg, rgb, sift)
        ok = cov[p2[:, 1].astype(int), p2[:, 0].astype(int)] > 0
        X, x = xyz[p2[ok, 1].astype(int), p2[ok, 0].astype(int)].astype(np.float64), p1[ok]
        if len(X) < 12:
            print("  too few matches to solve the camera")
            break
        res = None
        for f in (35, 45, 55, 61.26, 70, 80):
            K = K_for(f, w, h)
            good, rv, tv, inl = cv2.solvePnPRansac(X, x, K, None, reprojectionError=6, iterationsCount=3000)
            if not good or inl is None:
                continue
            pr, _ = cv2.projectPoints(X[inl.ravel()], rv, tv, K, None)
            err = float(np.median(np.linalg.norm(pr[:, 0] - x[inl.ravel()], axis=1)))
            score = len(inl) / (1 + err)
            if res is None or score > res[0]:
                res = (score, f, rv, tv, len(inl), err)
        if res is None:
            break
        _, fov, rv, tv, n_in, err = res
        V = np.eye(4)
        V[:3, :3] = cv2.Rodrigues(rv)[0]
        V[:3, 3] = tv.ravel()
        print(f"  pass {it + 1}: {len(X)} matches, {n_in} agree, median error {err:.1f} px, lens {fov:.1f} deg vertical")

    # 3. Bake at the photo's size: the visible surface and the one behind it.
    out = args.scene / "world" / args.out
    out.mkdir(parents=True, exist_ok=True)
    Kf = K_for(fov, W, H)
    layers = R.render(V, Kf, W, H, layers=4)
    front_rgb, _, front_z, front_a = layers[0]
    # Behind: the first surface clearly further back than the front one. (The very next layer is often the back of
    # the same rock: a spire is a closed solid.)
    back_rgb = np.zeros_like(front_rgb)
    back_z = np.zeros_like(front_z)
    back_a = np.zeros_like(front_a)
    for rgb_k, _, z_k, a_k in layers[1:]:
        take = (back_a == 0) & (a_k > 0) & (z_k > front_z * 1.12)
        back_rgb[take], back_z[take], back_a[take] = rgb_k[take], z_k[take], 1
    # Where nothing's behind (open sky beyond), leave it uncovered.
    np.savez_compressed(out / "front.npz", depth=front_z, alpha=front_a)
    np.savez_compressed(out / "behind.npz", depth=back_z, alpha=back_a)
    Image.fromarray((front_rgb * 255).astype(np.uint8)).save(out / "front.png")
    Image.fromarray((back_rgb * 255).astype(np.uint8)).save(out / "behind.png")
    small = np.asarray(Image.fromarray((front_rgb * 255).astype(np.uint8)).resize((w, h)))
    Image.fromarray(np.concatenate([photo, small, (0.5 * photo + 0.5 * small).astype(np.uint8)], 1)).save(out / "align_check.png")
    (out / "camera.json").write_text(json.dumps({"world_to_camera": V.tolist(), "fov_y_deg": fov}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
