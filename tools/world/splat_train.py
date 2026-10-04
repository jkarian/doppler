"""Train a Gaussian splat of a scene from its photo plus dreamed camera moves, then bake depth from the photo's camera.

    (inside WSL)  bash tools/world/run.sh tools/world/splat_train.py scenes/canyon spiral [--stride 3] [--steps 15000]

Inputs:
  <scene>/world/<move>/transforms.json   the dreamed frames' lens (frame 0 = the photo's view)     dream_views.sh
  <scene>/world/<move>/samples-rgb/*.png the frames as dreamed (used to measure the cameras)        dream_views.sh
  <scene>/world/<move>/upscaled/*.png    the frames upscaled (trained on, if present)               upscale_views.sh
  <scene>/world/moge_photo.npz, <move>/moge/*.npz   depth with the dreamed lens                     moge_views.py

The dreamed frames "boil": fine detail is re-invented in every frame. One 3D splat can't hold two versions of a
crack, so what the frames agree on (the shapes) is kept and what they don't (the boil) averages out. The real
photo is shown on half the steps, so it wins every disagreement it can see.

Cameras: the video model doesn't follow its own camera path exactly (a frame saved as a 6.7 degree turn shows
about 3.5), so each frame's camera is measured from its pixels: points matched to the photo get the photo's depth,
and PnP solves where the camera was. World = the photo's camera, in the units of the photo's MoGe depth. The
cameras keep being refined a little during training; the photo's is fixed.

Few views let a splat cheat: blobs that only look right from the training cameras. Two standard cures:
- start points in the hidden areas: every dreamed frame adds points where it sees something the photo can't;
- depth guidance: each view's rendered depth is pulled toward its MoGe depth (scaled to the world per frame).

Writes <scene>/world/splat/: splat.pt (Gaussians, cameras), depth_z.npy (planar depth from the photo's camera),
preview_photo_view.png, check_views.png (renders at training cameras next to their frames) and spin.mp4.
"""

import argparse
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy
from PIL import Image

dev = "cuda"


def load_rgb(path, size=None):
    im = Image.open(path).convert("RGB")
    if size and im.size != size:
        im = im.resize(size, Image.LANCZOS)
    return torch.tensor(np.asarray(im), dtype=torch.float32, device=dev) / 255.0


def intrinsics(frame, w, h):
    """The transforms give focal and centre in pixels of a base size of 2*cx by 2*cy; rescale to w x h."""
    sx, sy = w / (2 * frame["cx"]), h / (2 * frame["cy"])
    return np.array([[frame["fl_x"] * sx, 0, w / 2], [0, frame["fl_y"] * sy, h / 2], [0, 0, 1]], np.float64)


def unproject(z, K, step=1):
    h, w = z.shape
    vv, uu = np.mgrid[0:h:step, 0:w:step]
    rays = np.stack([uu + 0.5, vv + 0.5, np.ones_like(uu)], -1) @ np.linalg.inv(K).T
    return rays * z[vv, uu][..., None], vv, uu


def measure_cameras(move_dir, ks, frame0, z, land):
    """World-to-camera (OpenCV) of dreamed frames ks, measured by PnP against frame 0 = the photo's view, plus the
    inlier matches (world points, pixels in frame k). World = the photo's camera, in the units of z."""
    sift = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.01)
    clahe = cv2.createCLAHE(3.0, (8, 8))

    def feats(k):
        g = cv2.imread(str(move_dir / "samples-rgb" / f"{k:03d}.png"), 0)
        return g.shape, *sift.detectAndCompute(clahe.apply(g), None)

    (h, w), k0, d0 = feats(0)
    K = intrinsics(frame0, w, h)
    zs = cv2.resize(z, (w, h), interpolation=cv2.INTER_NEAREST)
    ls = cv2.resize(land.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    matcher = cv2.BFMatcher()
    out = {}
    for k in ks:
        _, kk, dd = feats(k)
        m = [a for a, b in matcher.knnMatch(d0, dd, k=2) if a.distance < 0.75 * b.distance]
        p0 = np.float64([k0[a.queryIdx].pt for a in m])
        p1 = np.float64([kk[a.trainIdx].pt for a in m])
        iu, iv = p0[:, 0].astype(int), p0[:, 1].astype(int)
        ok = ls[iv, iu]
        p0, p1, iu, iv = p0[ok], p1[ok], iu[ok], iv[ok]
        if len(p0) < 30:
            continue
        X = (np.c_[p0 + 0.5, np.ones(len(p0))] @ np.linalg.inv(K).T) * zs[iv, iu][:, None]
        good, rv, tv, inl = cv2.solvePnPRansac(X, p1, K, None, reprojectionError=4, iterationsCount=3000)
        if not good or inl is None or len(inl) < 20:
            continue
        V = np.eye(4)
        V[:3, :3] = cv2.Rodrigues(rv)[0]
        V[:3, 3] = tv.ravel()
        inl = inl.ravel()
        out[k] = (V.astype(np.float32), X[inl], p1[inl], (w, h))
    return out


def ssim(a, b):
    a, b = a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]
    w = torch.ones(3, 1, 11, 11, device=dev) / 121
    mu_a, mu_b = F.conv2d(a, w, groups=3), F.conv2d(b, w, groups=3)
    va = F.conv2d(a * a, w, groups=3) - mu_a**2
    vb = F.conv2d(b * b, w, groups=3) - mu_b**2
    cab = F.conv2d(a * b, w, groups=3) - mu_a * mu_b
    c1, c2 = 0.01**2, 0.03**2
    return (((2 * mu_a * mu_b + c1) * (2 * cab + c2)) / ((mu_a**2 + mu_b**2 + c1) * (va + vb + c2))).mean()


def small_rotation(v):
    """Rotation matrices from axis-angle vectors (n, 3)."""
    th = v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = v / th
    Kx = torch.zeros(len(v), 3, 3, device=v.device)
    Kx[:, 0, 1], Kx[:, 0, 2], Kx[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    Kx = Kx - Kx.transpose(1, 2)
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    return torch.eye(3, device=v.device) + s * Kx + (1 - c) * (Kx @ Kx)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", type=Path)
    ap.add_argument("moves", nargs="+")
    ap.add_argument("--stride", type=int, default=3, help="use every Nth dreamed frame")
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--photo-share", type=float, default=0.5, help="share of steps that show the real photo")
    ap.add_argument("--depth-weight", type=float, default=0.3, help="pull of the MoGe depths")
    ap.add_argument("--seed-step", type=int, default=4, help="pixel step when seeding hidden areas from dreamed frames")
    args = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)

    info = json.loads((args.scene / "scene.json").read_text())
    world = args.scene / "world"
    out = world / "splat"
    out.mkdir(parents=True, exist_ok=True)

    mp = np.load(world / "moge_photo.npz")
    zph, land = mp["depth"].astype(np.float64), mp["mask"].astype(bool)
    ph, pw = zph.shape
    zph[~land] = np.nan
    zmax = float(np.nanmax(zph))
    sky_z = zmax * 3  # sky sits well behind the land

    photo = load_rgb(args.scene / info["image"], (pw, ph))
    imgs, Ks, w2cs, dgts = [photo], [], [np.eye(4, dtype=np.float32)], [torch.tensor(np.nan_to_num(zph, nan=0), dtype=torch.float32, device=dev)]
    seeds_xyz, seeds_rgb = [], []
    frame0 = None
    for move in args.moves:
        d = world / move
        tf = json.loads((d / "transforms.json").read_text())["frames"]
        frame0 = frame0 or tf[0]
        src = d / "upscaled" if (d / "upscaled").exists() else d / "samples-rgb"
        ks = list(range(args.stride, len(tf), args.stride))
        cams = measure_cameras(d, ks, tf[0], np.nan_to_num(zph, nan=sky_z), land)
        print(f"  {move}: measured {len(cams)} of {len(ks)} cameras")
        Kph = intrinsics(tf[0], pw, ph)
        for k, (V, Xw, p1, (fw, fh)) in cams.items():
            im = load_rgb(src / f"{k:03d}.png")
            mk = np.load(d / "moge" / f"{k:03d}.npz")
            zk, mkm = mk["depth"].astype(np.float64), mk["mask"].astype(bool)
            # MoGe's depth for this frame, scaled into world units by the matched points' depths in this camera.
            Xc = Xw @ V[:3, :3].T + V[:3, 3]
            zz = cv2.resize(zk, (fw, fh), interpolation=cv2.INTER_NEAREST)[p1[:, 1].astype(int).clip(0, fh - 1), p1[:, 0].astype(int).clip(0, fw - 1)]
            ok = np.isfinite(zz) & (zz > 0)
            s = float(np.median(Xc[ok, 2] / zz[ok]))
            zk = np.where(mkm, zk * s, 0)
            Kk = intrinsics(tf[k], im.shape[1], im.shape[0])
            imgs.append(im)
            w2cs.append(V)
            Ks.append(Kk)
            zk_full = cv2.resize(zk, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
            dgts.append(torch.tensor(zk_full, dtype=torch.float32, device=dev))
            # Seed points where this frame sees something the photo doesn't: outside the photo, or behind its surface.
            Kd = intrinsics(tf[k], zk.shape[1], zk.shape[0])
            P, vv, uu = unproject(np.where(mkm, zk, np.nan), Kd, args.seed_step)
            P = P.reshape(-1, 3)
            cols = np.asarray(Image.open(d / "samples-rgb" / f"{k:03d}.png").convert("RGB").resize((zk.shape[1], zk.shape[0])))[vv, uu].reshape(-1, 3)
            keep = np.isfinite(P).all(1)
            Rinv = V[:3, :3].T
            Pw = (P[keep] - V[:3, 3]) @ Rinv.T
            q = Pw @ Kph.T
            u, v = q[:, 0] / q[:, 2], q[:, 1] / q[:, 2]
            inside = (q[:, 2] > 0) & (u >= 0) & (u < pw) & (v >= 0) & (v < ph)
            zseen = np.full(len(Pw), np.inf)
            zseen[inside] = np.nan_to_num(zph, nan=sky_z)[v[inside].astype(int), u[inside].astype(int)]
            hidden = ~inside | (Pw[:, 2] > zseen * 1.08)
            seeds_xyz.append(Pw[hidden])
            seeds_rgb.append(cols[keep][hidden] / 255.0)
    Ks.insert(0, intrinsics(frame0, pw, ph))
    nv = len(imgs)
    print(f"{nv} views: the photo at {pw}x{ph} and {nv - 1} dreamed frames at {imgs[1].shape[1]}x{imgs[1].shape[0]}")

    # Start points: the photo's depth (sky far behind), plus the hidden-area seeds.
    zstart = np.where(land, zph, sky_z)
    P, vv, uu = unproject(zstart, Ks[0], 2)
    means = np.concatenate([P.reshape(-1, 3)] + seeds_xyz)
    cols = np.concatenate([photo[vv, uu].reshape(-1, 3).cpu().numpy()] + seeds_rgb)
    print(f"  {P.shape[0] * P.shape[1]} points from the photo, {sum(len(s) for s in seeds_xyz)} seeded in hidden areas")
    n = len(means)
    size = np.linalg.norm(means, axis=1) / Ks[0][0, 0] * 2  # about two photo pixels wide at its distance

    C0 = 0.28209479177387814  # SH band 0
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.tensor(means, dtype=torch.float32, device=dev)),
        "scales": torch.nn.Parameter(torch.log(torch.tensor(size, dtype=torch.float32, device=dev))[:, None].repeat(1, 3)),
        "quats": torch.nn.Parameter(F.normalize(torch.randn(n, 4, device=dev), dim=-1)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), 0.5, device=dev))),
        "sh0": torch.nn.Parameter(((torch.tensor(cols, dtype=torch.float32, device=dev) - 0.5) / C0)[:, None, :]),
        "shN": torch.nn.Parameter(torch.zeros(n, 15, 3, device=dev)),
    })
    extent = float(np.nanpercentile(zph, 90))
    lrs = {"means": 1.6e-4 * extent, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    opts = {k: torch.optim.Adam([params[k]], lr=lr, eps=1e-15) for k, lr in lrs.items()}
    sched = torch.optim.lr_scheduler.ExponentialLR(opts["means"], gamma=0.01 ** (1.0 / args.steps))
    strategy = DefaultStrategy(refine_stop_iter=int(args.steps * 0.6), reset_every=3000, verbose=False)
    strategy.check_sanity(params, opts)
    state = strategy.initialize_state(scene_scale=extent)

    # Camera refinement: a small turn and shift per dreamed view (view 0, the photo, stays put).
    dturn = torch.zeros(nv, 3, device=dev, requires_grad=True)
    dshift = torch.zeros(nv, 3, device=dev, requires_grad=True)
    cam_opt = torch.optim.Adam([{"params": [dturn], "lr": 3e-5}, {"params": [dshift], "lr": 3e-5 * extent}])

    Kt = torch.tensor(np.stack(Ks), dtype=torch.float32, device=dev)
    Vt = torch.tensor(np.stack(w2cs), device=dev)

    def view(i):
        if i == 0:
            return Vt[0]
        V = Vt[i]
        R = small_rotation(dturn[i : i + 1])[0]
        return torch.cat([torch.cat([R @ V[:3, :3], (R @ V[:3, 3:]) + dshift[i, :, None]], 1), V[3:]], 0)

    def render(V, K, w, h, sh_degree=3, mode="RGB+ED"):
        colors = torch.cat([params["sh0"], params["shN"]], 1)
        return rasterization(params["means"], F.normalize(params["quats"], dim=-1), torch.exp(params["scales"]),
                             torch.sigmoid(params["opacities"]), colors, V[None], K[None], w, h,
                             sh_degree=sh_degree, render_mode=mode, packed=False, absgrad=False)

    seen = {"photo": [], "dreamed": []}
    for it in range(args.steps):
        i = 0 if np.random.rand() < args.photo_share else np.random.randint(1, nv)
        gt, dg = imgs[i], dgts[i]
        out_, alpha, meta = render(view(i), Kt[i], gt.shape[1], gt.shape[0], sh_degree=min(3, it // 1000))
        strategy.step_pre_backward(params, opts, state, it, meta)
        rgb, dep = out_[0, ..., :3], out_[0, ..., 3]
        loss_c = 0.8 * (rgb - gt).abs().mean() + 0.2 * (1 - ssim(rgb, gt))
        m = dg > 0
        loss_d = (torch.log(dep[m].clamp_min(1e-4)) - torch.log(dg[m])).abs().mean() if m.any() else torch.zeros((), device=dev)
        loss = loss_c + args.depth_weight * loss_d
        loss.backward()
        strategy.step_post_backward(params, opts, state, it, meta, packed=False)
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        if it > 1000:
            cam_opt.step()
        cam_opt.zero_grad(set_to_none=True)
        sched.step()
        seen["photo" if i == 0 else "dreamed"].append(loss_c.item())
        if (it + 1) % 1000 == 0:
            moved = math.degrees(dturn.norm(dim=1).max().item())
            print(f"  step {it + 1:5d}  colour error photo {np.mean(seen['photo']):.4f}  dreamed {np.mean(seen['dreamed']):.4f}"
                  f"  gaussians {len(params['means'])}  largest camera fix {moved:.2f} deg", flush=True)
            seen = {"photo": [], "dreamed": []}

    with torch.no_grad():
        Vfinal = torch.stack([view(i) for i in range(nv)]).cpu()
    torch.save({k: v.detach().cpu() for k, v in params.items()} | {"K": Ks[0], "size": (pw, ph), "views": Vfinal, "Ks": Kt.cpu()}, out / "splat.pt")
    with torch.no_grad():
        rgbd, _, _ = render(Vt[0], Kt[0], pw, ph)
        Image.fromarray((rgbd[0, ..., :3].clamp(0, 1) * 255).byte().cpu().numpy()).save(out / "preview_photo_view.png")
        np.save(out / "depth_z.npy", rgbd[0, ..., 3].cpu().numpy())
        # Renders at a few training cameras, next to their frames: bad 3D shows here; swinging past the dreams doesn't.
        rows = []
        for i in np.linspace(1, nv - 1, 4).astype(int):
            gt = imgs[i]
            r, _, _ = render(Vfinal[i].to(dev), Kt[i], gt.shape[1], gt.shape[0])
            a = (gt.cpu().numpy() * 255).astype(np.uint8)
            b = (r[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            rows.append(np.concatenate([a, b], 1))
        Image.fromarray(np.concatenate(rows, 0)).resize((2048, 1152 * len(rows) // 2)).save(out / "check_views.png")
        # A slow sideways swing about a point at mid depth, to see the 3D.
        tmp = out / "spin"
        tmp.mkdir(exist_ok=True)
        pivot = float(np.nanmedian(zph))
        for k in range(72):
            a = math.radians(10) * math.sin(2 * math.pi * k / 72)
            R = np.array([[math.cos(a), 0, -math.sin(a)], [0, 1, 0], [math.sin(a), 0, math.cos(a)]], np.float32)
            V = np.eye(4, dtype=np.float32)
            V[:3, :3] = R
            V[:3, 3] = (np.eye(3) - R) @ np.array([0, 0, pivot], np.float32)
            r, _, _ = render(torch.tensor(V, device=dev), Kt[0], pw, ph, mode="RGB")
            Image.fromarray((r[0].clamp(0, 1) * 255).byte().cpu().numpy()).save(tmp / f"{k:03d}.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "24", "-i", str(tmp / "%03d.png"), "-c:v", "libx264",
                    "-crf", "16", "-pix_fmt", "yuv420p", str(out / "spin.mp4")], check=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
