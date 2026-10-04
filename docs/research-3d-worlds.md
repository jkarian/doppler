# How image-to-3D-world systems make their splats (research, 2026-10-03)

Background research after our own dreamed-views-to-splat test (tools/world/, see STATUS.md). Facts are as
reported by the sources below; speculation is marked. Not re-verified by hand.

## World Labs Marble (official)

- **Inputs:** text, image(s), video, 360° panoramas, coarse 3D layouts ("Chisel": block out structure, prompt
  the style).
- **Exports:**
  - Splats: .spz/.ply, about 2M (or 500k) splats.
  - Panorama: 2560×1280.
  - Collider mesh: GLB, 100–200k triangles.
  - High-quality mesh: about 600k textured triangles or about 1M vertex-coloured triangles, up to an hour to make.
  - Conventions: OpenCV axes; rendered with their open-source three.js renderer, Spark.
- **Growing a world:** "Expand" grows a selected region; worlds can be composed. Marble 1.1 (Apr 2026)
  auto-expands ("dynamic cubes").
- **Not public:** the architecture. A panorama-first stage is likely (it exports a panorama) but unconfirmed.

## Open systems

| System | Key idea | Output | 24 GB? | Licence |
|---|---|---|---|---|
| HunyuanWorld 1.0 | Flux 360° panorama -> semantic layers (sky/bg/objects) -> layered mesh | mesh | "lite" fits a 4090 | Tencent community (no EU/UK/KR) |
| HunyuanWorld-Voyager | RGB+depth video from a reprojected "world cache" of points | RGB-D video, points | no (60-80 GB) | Tencent community |
| HY-World 2.0/2.1 | Pano (80B) -> WorldNav paths aimed at unseen areas -> WorldStereo keyframes with global point memory -> WorldMirror 2.0 feed-forward recon -> 3DGS with depth/normal losses, sky separate | 3DGS, mesh | full: unclear; WorldMirror 2.0 (1.2B) reportedly yes | Tencent community; outputs can't train other models |
| Matrix-3D (Skywork) | Panorama -> panoramic video along a path -> optimised 3DGS (or PanoLRM, 80 GB) | .ply | yes (5B pano video 12-19 GB; optimisation route) | **MIT** |
| FlashWorld | Gaussians generated directly inside multi-view diffusion | .spz/.ply in seconds | yes with offload flags | **Apache-2.0** |
| NVIDIA Lyra 1 | Video model distilled into a 3DGS decoder on its latents | 3DGS | probably not | NVIDIA Open Model (commercial OK) |
| NVIDIA Lyra 2.0 | Wan 2.1 14B walkthrough + per-frame geometry memory, anti-drift training, DA3 depth, feed-forward 3DGS | 3DGS | no (80 GB) | non-commercial |
| GEN3C | 3D cache rendered along the path, video model fills holes | video | unclear | NVIDIA Open Model |
| WonderWorld | Layered surfels (fg/bg/sky), guided depth diffusion | surfels | probably | unverified |
| SEVA (ours) | multi-view diffusion along a preset path | frames | yes | non-commercial |

## The common recipe (what separates good results from ours)

1. **Panorama first:** the whole 360° is fixed in one consistent image, so views are cut from it rather than
   dreamed separately (removes most boil).
2. **A 3D cache drives generation:** render what's built so far into the new view, dream only the holes.
   SEVA has no such memory, which is why our frames drift.
3. **RGB and depth together,** or feed-forward depth aligned to one global reference.
4. **Feed-forward poses and depth** (WorldMirror 2.0, DA3 + VIPE, VGGT-style) instead of PnP + bundle adjustment.
5. **Camera paths aimed at under-observed areas,** not one fixed spiral.
6. **Sky as its own far layer;** pruning, depth and normal losses in 3DGS training.

## For our canyon, by likely impact

1. Panorama from the photo (Matrix-3D's or HY-Pano), then train on views cut from it.
2. Cache-conditioned dreaming: render our splat into the new pose, inpaint only the holes.
3. The sky as a separate far layer.
4. WorldMirror 2.0 or DA3 for poses and depth.

**Worth trying on the 4090:**
1. Matrix-3D (MIT; 5B pano-video model plus the optimisation route).
2. FlashWorld (Apache; offload flags; a quick baseline).
3. HY-World 2.0's WorldMirror 2.0 (pose/depth stage; check the licence first).

**Skip:** Voyager, Lyra 2 and WorldPlay (too much memory and/or non-commercial).

## Sources
- https://docs.worldlabs.ai/marble/export/specs
- https://www.worldlabs.ai/blog/marble-world-model
- https://rits.shanghai.nyu.edu/ai/world-labs-releases-marble-1-1-auto-expanding-3d-world-generation
- https://radiancefields.com/world-labs-formally-launches-marble-a-generative-world-model
- https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0
- https://github.com/Tencent-Hunyuan/HunyuanWorld-Voyager
- https://github.com/Tencent-Hunyuan/HY-WorldPlay
- https://github.com/Tencent-Hunyuan/HY-World-2.0
- https://hyper.ai/en/papers/2604.14268
- https://docs.clore.ai/guides/guides_v2-hi/3d-generation/hunyuan-world-2 (third-party)
- https://arxiv.org/html/2510.10726v1 (WorldMirror)
- https://github.com/SkyworkAI/Matrix-3D
- https://github.com/imlixinyang/FlashWorld
- https://arxiv.org/html/2510.13678v1
- https://github.com/nv-tlabs/lyra
- https://huggingface.co/nvidia/Lyra-2.0
- https://arxiv.org/html/2509.19296v1
- https://arxiv.org/abs/2503.03751 (GEN3C)
- https://arxiv.org/abs/2406.09394v4 (WonderWorld)
