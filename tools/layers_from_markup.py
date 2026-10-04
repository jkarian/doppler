"""Rebuild a scene's depth from a layer markup: the user paints the picture's layers in flat colours and gives each a
real distance range, and every layer's depth is set into its range while keeping its own shape from the source
depth (what's nearer inside a layer stays nearer). A single photo's depth estimate squashes far distances (here the
pillars and the far canyon came out ~27x apart; the user's markup says ~100x, the mountains ~1000x).

    python tools/layers_from_markup.py scenes/canyon images/markup_layers.webp scenes/canyon-layers

The layer table below (colour -> metres) is this scene's markup; units of the new scene are metres
(scene.json metersPerUnit 1). Pixels in no layer keep the source's sky. Haze is rescaled to the new units.
Then run background_layer.py on the new scene.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

# colour (RGB, as painted) -> (name, nearest m, farthest m)
LAYERS = [
    ((139, 0, 0), "frame (cave walls)", 3, 30),
    ((0, 80, 140), "pillars and spire", 25, 80),
    ((0, 170, 150), "near canyon", 80, 400),
    ((0, 200, 30), "far canyon", 400, 3000),
    ((85, 85, 85), "mountains", 20000, 30000),
]
SKY_M = 60000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", type=Path)
    ap.add_argument("markup", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()

    info = json.loads((args.scene / "scene.json").read_text())
    W, H, far = info["width"], info["height"], info["far"]
    z = np.fromfile(args.scene / info["depth"], dtype="<f4").reshape(H, W).astype(np.float64)
    sky = z >= far * 0.98
    m = np.asarray(Image.open(args.markup).convert("RGB").resize((W, H), Image.NEAREST)).astype(np.float64)

    # Each pixel to the nearest painted colour (if close enough); everything else is sky.
    cols = np.array([c for c, *_ in LAYERS], np.float64)
    d = np.linalg.norm(m[:, :, None, :] - cols[None, None], axis=-1)
    label = np.where(d.min(-1) < 60, d.argmin(-1), -1)

    new = np.full((H, W), SKY_M, np.float64)
    for k, (_, name, lo, hi) in enumerate(LAYERS):
        sel = (label == k) & ~sky
        if not sel.any():
            continue
        # Rank within the layer, onto the layer's range in log distance.
        order = np.argsort(np.argsort(z[sel]))
        r = order / max(1, sel.sum() - 1)
        new[sel] = np.exp(np.log(lo) + r * (np.log(hi) - np.log(lo)))
        print(f"  {name:20s} {sel.mean() * 100:5.1f}% of the picture -> {lo}-{hi} m")
    # Painted as a layer but the source called it sky: trust the paint, at the layer's far end.
    for k, (_, name, lo, hi) in enumerate(LAYERS):
        new[(label == k) & sky] = hi

    args.out.mkdir(parents=True, exist_ok=True)
    rel = Path("..") / args.scene.name
    out = dict(info)
    for key in ("image", "albedo", "normal"):
        if key in info:
            out[key] = str(rel / info[key]).replace("\\", "/")
    out.pop("background", None)
    out["depth"] = "depth.bin"
    out["far"] = float(SKY_M)
    out["near"] = float(new.min())
    out["metersPerUnit"] = 1.0
    out["depthModel"] = f"layer markup ({args.markup.name}) over {args.scene.name}'s depth"
    if "haze" in out:
        # Same air per metre as before: the old scene's units were ~383 m (far land assumed ~3 miles at its 95th percentile).
        land = np.sort(z[~sky])
        old_m_per_unit = 15840 * 0.3048 / land[int((len(land) - 1) * 0.95)]
        out["haze"] = dict(out["haze"], beta=out["haze"]["beta"] / old_m_per_unit)
    new.astype("<f4").tofile(args.out / "depth.bin")
    (args.out / "scene.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out} (metres; far {SKY_M} m)")


if __name__ == "__main__":
    main()
