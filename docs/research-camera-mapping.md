# Automatic camera mapping: one image to a 3D scene

Research and hands-on tests on the canyon image, October 2026.

**Confidence labels:**

- **[V]** checked against a primary page (repo, README, licence file or official docs).
- **[S]** secondary source (news, aggregator).
- **[U]** unverified or from memory.
- **[T]** tested on the canyon image on the RTX 4090.

**Where the test images are:** the comparison images and test scripts are kept locally (not in git, because they show the source image) in `captures/research/camera-mapping/`. The raw outputs (point maps, Gaussian splat files, a scratch Python environment) stay in the session scratchpad and can be regenerated with the scripts in `captures/research/camera-mapping/scripts/`.

---

## Executive summary

- **Apple SHARP is the best disocclusion fix for sway and head tracking, but it can't be shipped.** [T]
  - SHARP (ICLR 2026) turned the canyon into a 2-layer Gaussian splat in about 1 s.
  - Rendered with the camera moved 0.08 and 0.15 scene units sideways, and pushed in 0.3, it showed 0% holes.
  - Our single-layer mesh showed 2–3% holes inside the frame, plus large gaps at the frame edges.
  - **Licence:** the weights are research only, with no commercial use. [V]
  - Splats also give no geometry for lasers or shadows.
- **MoGe-2 and MoGe-3 (MIT licence) give clearly better geometry than our current Depth Anything V2 + Depth Pro fusion on this image.** [T]
  - Silhouettes are crisper.
  - Depth increases steadily into the canyon.
  - Our depth has radial "sunburst" streak artefacts; MoGe has none.
  - Good normals that agree with the geometry.
  - Fast: 0.85 s (MoGe-2) and 1.6 s (MoGe-3) at 1920×1080.
  - Both squash the far range: the ratio of the 99th to the 1st depth percentile is about 36–39, against 83 for ours. So the far depth still needs an artistic remap.
  - Depth Anything 3 (Apache licence) was worse on the far canyon: flat and blobby. [T]
- **A home-made 2-layer layered depth image (LDI) is the automatic fix that fits our pipeline best, and its licences are commercially clean.** [T]
  - Steps: find depth edges, mark a band behind each edge sized to the expected parallax, inpaint that band's colour with LaMa, extend the background depth into it.
  - Result: 1.7 s, and holes inside the frame dropped from 2.9% to 0.8% at a 0.08 sway.
  - LaMa smears wide bands. Good quality needs a diffusion inpainter (FLUX Fill or SDXL).
  - This is essentially "3D Photo Inpainting" (Shih et al., MIT licence), rebuilt with 2026 parts.
- **For real geometry with filled-in occlusions, the strongest automatic option is World Labs Marble** (single image in; a 2M Gaussian splat, a collision mesh and a high-quality mesh of about 600k triangles out). [V]
  - **Cost:** API about $1.26 per world plus $2.80 per high-quality mesh. The Pro app is $35/month, and Pro is needed for commercial rights. [V/S]
  - **Fidelity:** it reinterprets the image rather than matching it exactly.
  - **Vendor risk:** AMD announced it is acquiring World Labs on 28 September 2026. [S]
- **Open-weight "world generators" mostly can't run locally here, or have restrictive licences.**
  - **HY-World 2.0/2.1:** Tencent's licence excludes the EU, UK and South Korea and has a monthly-user cap; the models are 17B–80B parameters.
  - **Lyra 2.0:** NVIDIA internal-research licence; built for H100-class GPUs.
  - **Voyager and WonderWorld:** need 60 GB and 48 GB of GPU memory.
  - **The realistic open candidates:** FlashWorld (Apache, fits 24 GB with offloading) and Matrix-3D (MIT, 16–19 GB, about 1 hour per scene). Both need Linux or WSL2 with the CUDA build tools, which this machine doesn't have, so they weren't tested.
- **Gaussian splats can be rendered in WebGPU, but we'd write our own pass.**
  - **Reference code:** PlayCanvas (MIT) has a WebGPU compute splat renderer with GPU sorting and relighting through a proxy mesh. web-splat (Apache) has a WGSL radix sort that would port straight to Rust/wgpu.
  - **Lighting:** splats can be lit through a proxy mesh, but not properly on their own.
- **Recommendation in short:**
  1. Swap in MoGe-2/3 for geometry now, keeping the far-depth remap.
  2. Build the automatic 2-layer LDI with diffusion inpainting, behind the same file format.
  3. Try Marble once on the canyon to judge whether a splat-plus-mesh scene is worth changing the renderer for.
  4. Keep Maya projection as a 1–2 hour fallback for hero shots.
  5. Use SHARP only as a quality reference.
- **Licence warning for the current pipeline:** Depth Anything V2 Base and Large are **non-commercial** (CC BY-NC); only Small is Apache. [V/S] Depth Pro's licence is a permissive Apple licence. [V]

---

## 1. Hands-on tests on the canyon image

**Setup.**
- A separate Python environment (Python 3.12, torch 2.11 with CUDA 12.8). The project's own environment wasn't touched.
- Input: the Topaz 7850×4416 image, downscaled to 1920×1080.

### 1.1 Depth and geometry models

| Model | Licence | Time (1920×1080, 4090) | GPU memory peak | Estimated field of view | Far range (99th/1st percentile; ours = 83) | Observations |
|---|---|---|---|---|---|---|
| **Ours** (Depth Anything V2 + Depth Pro fused) | DA-V2-Large is non-commercial | – | – | 61.3° vertical | 83 | Radial "sunburst" streaks on the upper cave walls. The mountains are separated, but a patch of mid-canyon sits at mountain depth, probably wrongly. Soft edges. |
| **MoGe-2 ViT-L** | MIT [V] | **0.85 s** | 2.4 GB | 63.8° vertical / 95.7° horizontal | 36 | Crisp silhouettes. Mesas form separate layers, in order. No streaks. Sky masked automatically. Good normals with visible strata. Mountains merge into the far mesas. |
| **MoGe-3 ViT-G** (August 2026) | MIT [V] | 1.6 s (14.9 s on first load) | 7.2 GB | 59.9° vertical | 39 | Like MoGe-2, with slightly better wall shapes and fine detail. Same squashed far range. |
| **Depth Anything 3 Metric Large** | Apache-2.0 [V] | 0.6 s | – | – | 22 | Far canyon flat and blobby; noisy mountain edge. |
| **Depth Anything 3 Mono Large** | Apache-2.0 [V] | 0.5 s | – | – | 67 | Better range, but a smooth, blobby far field. |

**Notes for running MoGe:**
- The pip `moge` command-line tool says the MoGe-3 weights aren't available yet, but they are. Load them through the Python API: `moge.model.v3.MoGeModel.from_pretrained("Ruicheng/moge-3-vitg")`.
- Its EXR writer fails on Windows unless `OPENCV_IO_ENABLE_OPENEXR=1` is set.
- Its output is camera-space points. Our `depth.bin` holds distance, so take the length of each point.

**Comparison images** (in `captures/research/camera-mapping/`):
- `compare_logdepth_full.png` (ours | MoGe-2)
- `compare_full_ours_moge2_moge3.png`
- `compare_far_ours_moge2_moge3.png` (far-canyon close-up)
- `compare_far_da3metric_da3mono.png`
- `compare_normals_ours_moge2.png`

**Verdict on the blobby far depth:** MoGe gives cleaner far *structure*, with ordered layers and sharp edges, which suits tearing and layering. Every single-image model still underestimates how deep a real canyon is, so keep an artistic remap of the far range.

### 1.2 Apple SHARP (single image to 3D Gaussians)

**Install and runtime:**
- It installs and predicts without compiler tools. Its built-in renderer needs CUDA compilation, which this machine can't do.
- About 1 s inference, 16 s including loading the model.
- Output: about 1.18M Gaussians, two layers at 768×768.

**Watch out for the lens setting.**
- SHARP reads the lens from the image's EXIF data and defaults to a 30 mm equivalent (about 64° horizontal). That's wrong for this image, which is about 96°.
- Setting EXIF `FocalLengthIn35mmFilm=17` fixed it.
- Rule for the pipeline: always pass the field of view from `scene.json` or MoGe.

**The two layers:**
- The second layer sits on top of the first almost everywhere. Only 0.6% of pixels have layer 2 more than 5% behind layer 1.
- Those pixels are thin bands along depth edges, which is exactly where filling is needed.
- So SHARP is effectively a learned 2-layer LDI with stretched Gaussians at the edges.

**Holes in rendered new views** (magenta marks a hole; images in `captures/research/camera-mapping/`: `cmp3_dx15_ours_ldi_sharp.png`, `cmp3_push30_ours_ldi_sharp.png`, `crop_dx15_sharp_vs_ours.png`):

| Hole fraction | Sway 0.08 | Sway 0.15 | Push in 0.3 |
|---|---|---|---|
| Ours (single-layer mesh, torn at >4% depth jumps) | 16.6% total / **2.9% inside the frame** | 27.7% / 3.1% | 13.4% / 2.2% |
| MoGe-2, single layer | 13.9% / 1.4% | 24.1% / 1.1% | – |
| Home-made 2-layer LDI (ours + LaMa background) | 12.3% / **0.8%** | 21.3% / 0.8% | 1.6% / 1.5% |
| SHARP | **0% / 0%** | 0% / 0% | 0% / 0% |

"Total" includes the area beyond the original frame edge, which SHARP covers with smeared edge Gaussians. Each model's scale differs slightly, so the camera moves are comparable but not identical.

**What SHARP's fills look like:** at a 0.15 sway, the strips revealed behind the cave lip and the rock pillars are filled with believable rock and canyon, with no visible tearing. The right frame edge becomes a dark smear.

**Verdict:** the best disocclusion result, and fast. But it's research-only, gives splats only, and is built for nearby views. Use it as the quality bar, not the production path.

### 1.3 Home-made layered depth image (script: `make_ldi.py`)

**How it works** (1.7 s at 1920×1080):
1. Find depth edges (jumps of more than 15%).
2. Mark the foreground pixels near each edge that the camera's movement could uncover. The band's width is set by the expected parallax: wide behind the near cave lip, a few pixels behind distant mesas, capped at 250 px.
3. Inpaint those pixels' colour with **LaMa** (Apache licence, about 1 s).
4. Make the background depth the blurred depth of the farthest neighbour.
5. Draw layer 2 only in the marked band, behind layer 1.

**Result:**
- Most holes behind the cave lip and pillars are gone.
- LaMa makes mushy rock over wide bands.
- The remaining slivers come from threshold tuning.

**Fixes for production:**
- Use a diffusion inpainter instead of LaMa.
- Re-run the depth model on the inpainted background.
- Use MoGe depth to find the edges, since ours has soft, streaky edges.

Preview: `captures/research/camera-mapping/ldi_preview.png`, background plate: `ldi_bg_rgb.png`.

### 1.4 What couldn't be tested

- **FlashWorld, HunyuanWorld 1.0, WonderWorld, Matrix-3D, and the gsplat renderer:** all need CUDA extensions compiled, or Linux. Installing WSL2 with the CUDA toolkit would unblock FlashWorld and Matrix-3D.
- **Marble:** export needs a paid account.

---

## 2. The landscape of methods

### 2.1 Fast single-image splats and layered depth (small camera moves)

| Method | Output | Licence | Notes | Link |
|---|---|---|---|---|
| **Apple SHARP** (ICLR 2026) | 3D Gaussian splat file, 2 layers per pixel, real-world scale | Weights: **research only, no commercial use** [V] | Best quality and speed tested. Set the lens via EXIF. | https://github.com/apple/ml-sharp |
| **InfiniSplat** (SIGGRAPH Asia 2026) | Single-image Gaussian splats, aimed at larger camera moves | [U] | Newest; worth watching | https://arxiv.org/abs/2608.02437 |
| **Multi-Layer Gaussian Splatting** (ACM MM 2025) | Layered splats with inpainting of hidden areas | [U] | Closest in concept to what we want | https://dl.acm.org/doi/10.1145/3746027.3755176 |
| **ORCA** (September 2026) | Repairs small revealed areas and inpaints large ones | [U] | Targets exactly this problem; very new | https://arxiv.org/abs/2609.17450 |
| **3D Photo Inpainting** (CVPR 2020) | Layered depth mesh with inpainted colour and depth | MIT [V] | Old code; rebuild with modern parts (§1.3) instead | https://github.com/vt-vl-lab/3d-photo-inpainting |
| Flash3D, Splatter Image (2024) | Single-image splats | Mixed | Superseded by SHARP | https://github.com/eldar/flash3d |

### 2.2 Generative world models (bigger camera moves, invented content)

| Method | Output | Licence (commercial?) | Hardware and runtime | Link |
|---|---|---|---|---|
| **World Labs Marble 1.1** | Splats (SPZ/PLY), collision mesh, high-quality mesh, 360° panorama | Commercial from the Pro tier [V] | Cloud; about $1.26 per world through the API | https://docs.worldlabs.ai/marble/export/gaussian-splat/index |
| **Tencent HY-World 2.0/2.1** | Panorama → navigation → stereo → splats or mesh | Tencent licence: **not EU, UK or South Korea**; monthly-user cap [V] | 24 GB is borderline | https://github.com/Tencent-Hunyuan/HY-World-2.0 |
| **HunyuanWorld 1.0** | **Layered meshes** (sky, background, foreground) built from a generated 360° panorama | Tencent licence, same limits [V] | The lite version runs on a 4090 | https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0 |
| **FlashWorld** (ICLR 2026) | Splats in seconds | **Apache-2.0** [V] | 24 GB with offloading; needs Linux or WSL2 | https://github.com/imlixinyang/FlashWorld |
| **Matrix-3D** | Panoramic video → splats | **MIT** [V] | 12–19 GB; about 1 hour per scene | https://github.com/SkyworkAI/Matrix-3D |
| **NVIDIA Lyra 2.0** | Splats via video diffusion | **Internal research licence, no commercial use** [V] | Built for H100s | https://huggingface.co/nvidia/Lyra-2.0 |
| **WonderWorld** (CVPR 2025) | Layered Gaussian surfels in under 10 s | Licence not stated | **48 GB** | https://github.com/KovenYu/WonderWorld |
| **HunyuanWorld-Voyager** | Video plus depth → point cloud | Tencent licence | **60 GB minimum** | https://github.com/Tencent-Hunyuan/HunyuanWorld-Voyager |
| Genie 3, Odyssey, Runway world models | Interactive video | No 3D export | – | – |

### 2.3 Depth and geometry models (faithful to the image)

| Model | Licence | Notes | Link |
|---|---|---|---|
| **MoGe-2 / MoGe-3** | MIT [V] | Real-world-scale points, normals, sky mask and lens. **Recommended.** [T] | https://github.com/microsoft/MoGe |
| Depth Pro | Apple licence, commercial use allowed [V] | Sharp boundaries | https://github.com/apple/ml-depth-pro |
| Depth Anything V2 | Small: Apache. **Base and Large: non-commercial** | Licence risk in the current pipeline | https://github.com/DepthAnything/Depth-Anything-V2 |
| Depth Anything 3 | Some sizes Apache, some non-commercial [V] | Weaker than MoGe here [T] | https://github.com/ByteDance-Seed/Depth-Anything-3 |
| VGGT, MapAnything, DUSt3R/MASt3R | Mixed | Built mainly for multiple views | – |

**Takeaway:** no open, commercially usable method that fits in 24 GB yet produces layered meshes faithful to the input image end to end. The practical automatic route is strong single-image geometry (MoGe) plus inpainting of a background layer, which is what SHARP learned implicitly. For bigger camera moves, generative world models invent the whole scene.

---

## 3. Gaussian splats in our WebGPU renderer

| Renderer | Backend | Fit for us | Link |
|---|---|---|---|
| Spark (World Labs) | WebGL2 only | Good for prototyping; can't share our WebGPU device | https://github.com/sparkjsdev/spark |
| PlayCanvas / SuperSplat 3 | WebGPU compute renderer, GPU radix sort, relighting through a proxy mesh | Most complete WebGPU reference (MIT) | https://developer.playcanvas.com/user-manual/gaussian-splatting/building/relighting/ |
| three.js r186 | Native WebGPU splat mesh | Easy if we used three.js | – |
| **web-splat** | **Rust + wgpu** (also WebAssembly), radix sort in WGSL | **Best code to borrow for us, now and for a future Rust port** (Apache) | https://github.com/KeKsBoTer/web-splat |

**Capacity:** 1–2M splats (SHARP is 1.2M, Marble about 2M) is comfortable on a desktop GPU with GPU sorting.

**Drawing order:**
1. Opaque meshes first, writing depth.
2. Sorted splats, depth-tested but not writing depth.
3. Lasers and particles.

**Lighting splats:**
- **Proxy mesh (cheapest):** light a mesh (Marble's mesh, or our depth mesh) and multiply the splat colour by that lighting.
- **Depth and normal buffer:** write the splats' depth and normals into a buffer and light that.
- **Baked lighting:** splat colours include the photo's lighting, so our delighting would need projecting back onto them.

**Implication:** keep a mesh for lighting, shadows and lasers either way. That makes the layered mesh the main representation, with splats optional.

---

## 4. Commercial apps and services

| Tool | Single image? | Output | Fit for landscapes | Cost |
|---|---|---|---|---|
| **World Labs Marble** | Yes | Splats, collision mesh, high-quality mesh, panorama | **Best** (reinterprets the image) | Pro about $35/month; API about $1.26 per world + $2.80 per high-quality mesh |
| HY-World web product | Yes | Splats, mesh | Good [S] | Free trial |
| Blockade Skybox AI | Mostly text | Panorama, depth, mesh | Backdrop or sky dome only | $24–140/month |
| Immersity AI (Leia) | Yes | Depth maps, parallax video | Depth only, one layer | $5–100/month |
| DepthFlow | Image + depth | Parallax video | Shader reference | Free, AGPL licence |
| Meshy, Tripo, Rodin, Hunyuan3D, CSM, Kaedim | – | Object meshes | **Not suited to landscapes** | – |
| Runway, Genie, Odyssey, Luma | – | Video only | – | – |

---

## 5. Maya and other 3D software (if done by hand)

**There's no ready-made "depth to projected mesh" plugin for Maya.** [V] The fastest route is a short Python script:
1. Read `depth.bin` and `scene.json`.
2. Turn a grid into points along camera rays, with each vertex's UV set to its pixel position, so the projection is built in.
3. Drop squares whose depth jumps more than about 5–10%, or that cross the sky mask. This tears the cave lip away from the walls behind it.
4. Group the pieces into layers by depth.

Displacing a plane along its normal is the wrong approach, because that's an orthographic projection.

**Then:**
- **Artist pass (30–60 minutes):** extend hidden geometry behind torn edges and fix depth errors.
- **Clean plates (20–40 minutes):** Photoshop Generative Fill, one layer at a time.
- **Passes per layer from Arnold:** render the **position (P) pass as 32-bit float with the "closest" filter** and compute distance from it. That avoids ambiguity about how Z is measured. Render normals the same way and rotate them into camera space. Use one render layer per depth layer, hiding the nearer layers; a deep EXR does *not* reveal hidden surfaces.
- **GLB export:** Maya has no built-in glTF exporter. Use Maya GLB I/O or glTFMaya, or go through FBX/USD to Blender. Alternatively, project the texture in our shader directly.

**Camera matching:** fSpy can be imported into Maya (`maya_fspy`), but it's mostly unnecessary because we already know the lens. KeenTools has no Maya version.

**Other software:**
- **Blender:** the best ecosystem (Depth Map Batch, Depth Mesh Pro, KIRI 3DGS Render).
- **Houdini 21:** ONNX inference and depth-to-points nodes.
- **Nuke 17.x:** a 3D system, Gaussian splat support and DepthToPoints.

---

## 6. Recommendation

Current scene files: `photo.png`, `albedo.png`, `depth.bin` (float32 distance), `normal.png`, `sky.png`, `scene.json`.

### Path 1 (first, about 1 day): swap in MoGe geometry

- **What:**
  - Run MoGe-2 or MoGe-3 on the 4K image, passing the known lens so the field of view stays consistent.
  - Write `depth.bin` as now, using the length of each point.
  - Keep the far-range remap and the sky mask.
  - Optionally blend: MoGe for structure and edges, Depth Pro for near detail.
- **Gains:**
  - No more sunburst streaks.
  - Cleaner, ordered far layers.
  - Crisp silhouettes for tearing.
  - Normals that match the geometry (it could replace Marigold for normals).
  - The licence becomes MIT.
- **Doesn't fix:** disocclusion.

### Path 2 (the main fix, 2–4 days): automatic 2-layer LDI with diffusion inpainting

- **Steps:**
  1. MoGe depth.
  2. Find edges and mark the band each one could reveal.
  3. Inpaint the background colour with a diffusion model, in 4K tiles.
  4. Re-run MoGe on the inpainted background and align it.
  5. Delight the background too.
- **New files:**
  - `photo_bg.png`, `albedo_bg.png`, `depth_bg.bin`, `normal_bg.png`
  - `layer_mask.png` (where layer 2 is valid)
  - a `layers` list in `scene.json`
- **Renderer:**
  - Tear the front mesh at depth jumps instead of stretching it.
  - Draw the layer-2 mesh behind it.
  - Both layers are real geometry, so lighting, shadows and lasers keep working.
- **Measured:** holes inside the frame dropped from 2.9% to 0.8% even with rough LaMa inpainting. Expect close to zero with diffusion inpainting and MoGe edges.
- **Frame edges:** outpaint the photo about 10% on each side first, to cover the gaps at the edges during sway and push-ins.

### Path 3 (try this week, 1–2 hours, about $35): Marble for a full scene

- **Try it:** upload the canyon to Marble on the Pro tier. Export the splats and both meshes, and look at them in a splat viewer to judge fidelity and how far the camera can go.
- **Option A:** render the high-quality mesh from our camera to make `depth.bin`, `normal.png` and the background layers (a "virtual Maya").
- **Option B:** add a splat pass to the renderer, using the mesh for lighting, lasers and shadows.
- **Risks:** not exactly faithful to the image, a cloud dependency, and the AMD acquisition.
- **Open alternative:** FlashWorld or Matrix-3D under WSL2 on the 4090.

### Path 4 (fallback for hero scenes, 1–2 hours of artist time): Maya-assisted layered projection

Auto-build the torn mesh (§5), fix it up by hand, make clean plates with generative fill, then render passes per layer or export a mesh. This produces the same files as Path 2, but art-directed.

### SHARP

Use it as the benchmark for a good 2-layer result. Don't ship it unless Apple changes the licence or the use is private.

### This week

1. **Day 1:** MoGe swap (Path 1), compared side by side in the real renderer.
2. **Days 2–3:** make the LDI script production-ready, with MoGe edges and a diffusion inpainter, and add layer support plus tearing to the renderer.
3. **Alongside:** one month of Marble Pro to evaluate the canyon.
4. **Optional:** set up WSL2 with CUDA to try FlashWorld.
