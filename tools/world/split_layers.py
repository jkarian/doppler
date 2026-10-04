"""Split a scene's photo into layers for separate 3D generation (e.g. one Tripo mesh each), so each comes out a simple
object and the layers can be placed at their real distances (a single image makes a squashed diorama).

    (WSL, Matrix-3D venv: diffusers + FLUX Fill)  python tools/world/split_layers.py scenes/canyon

Uses parts.json and parts/*.png (segment_parts.py). Writes <scene>/world/layers/:
  frame.png   the nearest parts (order 1: cave walls, floor) cut out, transparent elsewhere
  mid.png     the freestanding parts in front of the vista (spire, fin, tree) cut out
  vista.png   everything beyond, with the frame and the mid-ground parts painted out by FLUX Fill so the view opens
              up behind them (prompt from --prompt)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

PROMPT = ("a vast open red rock canyon vista seen from high above, layered sandstone mesas and buttes, a river winding far "
          "below, distant canyon rims fading into haze, snowy mountains on the horizon, warm late afternoon light, photo")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", type=Path)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    info = json.loads((args.scene / "scene.json").read_text())
    photo = Image.open(args.scene / info["image"]).convert("RGB")
    W, H = photo.size
    parts = json.loads((args.scene / "parts.json").read_text())["parts"]

    def mask_of(sel):
        m = np.zeros((H, W), bool)
        for p in parts:
            f = args.scene / "parts" / f"{p['name'].replace(' ', '_')}.png"
            if sel(p) and f.exists():
                m |= np.asarray(Image.open(f).convert("L").resize((W, H), Image.NEAREST)) > 127
        return m

    near = min(p["order"] for p in parts)
    frame = mask_of(lambda p: p["order"] == near)
    mid = mask_of(lambda p: p.get("freestanding") and p["order"] > near and p["name"] not in ("left mesa",))
    out = args.scene / "world" / "layers"
    out.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(photo)
    for name, m in (("frame", frame), ("mid", mid)):
        a = Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))
        Image.fromarray(np.dstack([rgb, np.asarray(a)])).save(out / f"{name}.png")
        print(f"{name}: {m.mean() * 100:.1f}% of the picture")

    # Vista: paint out the frame and the mid-ground (a little wider, so no rim of them survives).
    hole = Image.fromarray(((frame | mid) * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(25)).filter(ImageFilter.GaussianBlur(6))
    from diffusers import FluxFillPipeline

    pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16)
    # fp8 storage (12 GB) so the transformer sits on the 24 GB card instead of spilling into shared memory.
    pipe.transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()
    w, h = 1536, 864
    img = pipe(prompt=args.prompt, image=photo.resize((w, h), Image.LANCZOS), mask_image=hole.resize((w, h), Image.LANCZOS),
               height=h, width=w, guidance_scale=30, num_inference_steps=40,
               generator=torch.Generator("cpu").manual_seed(args.seed)).images[0]
    img.save(out / "vista.png")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
