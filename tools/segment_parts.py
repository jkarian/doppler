"""Segment a scene into the parts named in its parts.json, with SAM 2.

    python tools/segment_parts.py scenes/canyon

parts.json is the scene description: each part's name, a few points on it, its order (nearest first) and
whether it stands free of what's behind it. A vision-language model understands what the parts are and
how they relate; SAM 2 (Meta, Apache licence, through transformers) traces each one's exact outline from
those points. Points on the other parts are given as "not this", so neighbouring parts don't merge.

Writes parts/<name>.png (white = the part) and parts_overlay.png (all parts coloured) into the scene folder.
Where parts overlap, the nearer one (lower order) wins.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODEL = "facebook/sam2.1-hiera-large"
WORK_W = 1920


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", type=Path)
    args = ap.parse_args()

    info = json.loads((args.scene / "scene.json").read_text())
    parts = json.loads((args.scene / "parts.json").read_text())["parts"]
    photo = Image.open(args.scene / info["image"]).convert("RGB")
    W, H = photo.size
    h = round(H * WORK_W / W)
    img = photo.resize((WORK_W, h), Image.LANCZOS)

    from transformers import Sam2Model, Sam2Processor

    proc = Sam2Processor.from_pretrained(MODEL)
    model = Sam2Model.from_pretrained(MODEL).to("cuda").eval()

    masks = []
    for i, part in enumerate(parts):
        pos = [[u * WORK_W, v * h] for u, v in part["points"]]
        neg = [[u * WORK_W, v * h] for j, other in enumerate(parts) if j != i for u, v in other["points"][:1]]
        points = pos + neg
        labels = [1] * len(pos) + [0] * len(neg)
        inputs = proc(images=img, input_points=[[points]], input_labels=[[labels]], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(**inputs, multimask_output=True)
        m = proc.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0][0]  # (3, h, w)
        scores = out.iou_scores[0, 0].float().cpu().numpy()
        best = int(np.argmax(scores))
        masks.append(m[best].numpy().astype(bool))
        print(f"  {part['name']:24s} {masks[-1].mean() * 100:5.1f}% of the picture  (score {scores[best]:.2f})")

    # Nearer parts win overlaps.
    order = sorted(range(len(parts)), key=lambda k: parts[k]["order"])
    taken = np.zeros((h, WORK_W), bool)
    out_dir = args.scene / "parts"
    out_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(3)
    overlay = np.asarray(img).astype(np.float32) * 0.35
    for k in order:
        m = masks[k] & ~taken
        taken |= m
        Image.fromarray(m.astype(np.uint8) * 255).resize((W, H), Image.NEAREST).save(out_dir / f"{parts[k]['name'].replace(' ', '_')}.png")
        overlay[m] += rng.uniform(80, 255, 3) * 0.65
    Image.fromarray(overlay.clip(0, 255).astype(np.uint8)).save(args.scene / "parts_overlay.png")
    print(f"unassigned: {(~taken).mean() * 100:.1f}% of the picture. Wrote {out_dir} and parts_overlay.png")


if __name__ == "__main__":
    main()
