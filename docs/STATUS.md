# Doppler Canyon: status and handoff

Last updated 2026-10-05 (late evening). Read this first when picking the project back up. Two threads: the
**camera-mapping utility** (start with **Whale shark 2026-10-05** and **Handoff 2026-10-05** below) and the canyon visualizer (**Handoff 2026-10-04
night**; its real sunrise is still its next step).

## Whale shark 2026-10-05 (camera-mapping, third test image; the "hard one")

`images/whale_shark.webp` -> `scenes/whale-shark` (scale x2: the user set the shark to 10 m nose to tail tip; MoGe
said ~5 m). Export to open: `scenes/whale-shark/maya/v002/whale_shark_v002.ma` (v001 is missing the sky card's
texture: export bug, fixed; v001 kept, ask before deleting). Not yet reviewed by the user in Maya.
Layers: fin (near pectoral), shark (body + head + dorsal + tail, with fibres as soft alpha), farfin, water (open,
on the far card at 22 m). Hard because it's see-through, has hair-like glowing fibres and an out-of-focus backdrop.

New per-part options in parts.json (general, for any image):
- `"outline": "sam"`: see-through objects (glassy fins). BiRefNet cuts what shows through them (holes, ragged
  edges); SAM's outline stands and is final: depth (blurred across edges) no longer grows it into unclaimed pixels
  (that was a dark bleed round the fin tip, which the user spotted).
- `"hair": true`: fine see-through detail round the outline (fibres, fur, wisps). A band (8% of the frame) round the
  layer is painted clean in the back layer, then the fibres are pulled out of the photo against that clean plate
  (difference key, `lift_hair`) as soft alpha on the layer, at the depth of its nearest real pixel: they recede with
  the body (the user's requirement). Maya takes the soft alpha through the projection's transparency.
- `"open": true` (back layer): open water / clear sky. No mesh; all of it goes on the far card. Filled by
  `smooth_backdrop` (multi-scale normalized blur of the real water, fine detail added back) instead of Flux: Flux
  painted a sparkle ghost of the shark into the shark-shaped hole, and a plain inpaint left the shark's outline
  (its glow). The real water goes through the same field, so there is no seam anywhere.
- Layers are extended past the frame only from where they touch the frame's edge (the shark got blobs of water
  painted above it otherwise).

Open: parallax is the default +/-30 cm sway (gentle for a 10 m animal at 3-7 m; `export --move 1.0` if the user
wants more); shark mesh ~460k triangles (NEXT 4); bokeh specks near the shark ride with it (lifted with the fibres).
Commands, for redoing it: `measure`, `scale --factor 2`, parts.json (in scenes/, git-ignored), `segment_parts.py`,
`layers`, `paint` (~6 min), `export`.

## Handoff 2026-10-05: camera-mapping utility (base)

**Goal (the user's words):** any image in -> a matte-painting camera projection: separate cut-out layers (fg/mg/bg...),
each extended by inpainting behind what's in front and past the frame, real-size geometry per layer, a projection
camera that lines them up, a small animated camera for parallax, Maya out (.abc + .ma). End to end, the only user
interaction being the dimension check. Later a web app (Claude API judges the image; users don't connect a claude.ai
subscription). A person may optionally supply layer cut-outs and/or painted plates; the tool then only fills gaps.

**Where we are.** Works end to end on two test images; the user's verdict on the car: "the general scene looks right
and the parallax is right", artifacts left are small. Latest exports (open in Maya, look through renderCam):
`scenes/car-beach/maya/v001/car_beach_v001.ma` (images/car_beach.webp, scale x1) and `scenes/ice-cave/maya/ice_cave.ma`
(images/ice_cave.webp, x1.5; exported before versioning and before the seam/choke/BiRefNet fixes: re-run `layers`,
`paint`, `export` to bring it up to date, it will become its v001). `scenes/car-beach/car_beach_saved.ma` is the
user's accidental save, kept aside; the unversioned files directly in `scenes/car-beach/maya/` duplicate v001
(ask before deleting). The two scenes' `parts.json` (Claude's judging) live in `scenes/` (git-ignored): don't lose them.

**Tool:** `tools/camera_map.py` (+ `tools/camera_map_maya.py` under mayapy, Maya 2027). Steps:
1. `measure <image> <scene>`: MoGe-2 metric depth, lens, normals, sky; dimensions.png with labelled sizes.
2. `scale <scene> --factor F` or `--set name=metres`: the user confirms the size (the one interactive step).
3. `parts.json` (written by Claude by hand so far: parts, points (u, v), near-to-far order, layer per part, a "rest"
   layer, one prompt per layer), then `tools/segment_parts.py <scene>` (SAM 2 traces each part).
4. `layers <scene>`: parts -> layer masks. Freestanding objects get BiRefNet's outline (sharp, real holes) as the final
   word; strays/slivers/islands cleaned; check `layers_overlay.png`.
5. `paint <scene>`: per layer, FLUX.1 Fill behind the nearer layers (only as far as the move reveals) and past the frame
   (12% border); SAM check of what's painted; MoGe depth per painted layer fitted to the real depth; check
   `layers_painted.png`. ~13 min on the 4090 for 4 layers.
6. `export <scene>`: `maya/vNNN/` (a new version every time, never overwritten): projCam (fixed, gate = whole canvas,
   projection nodes), renderCam (photo lens, sway, at rest on frame 1), one mesh per layer + sky card, levelled.
Check renders (Arnold, watermark; Maya's default view transform makes them darker): scratchpad script
`render_maya.py <scene> <frames>`; recreate it if the scratchpad is gone (open latest vNNN .ma, arnoldRender renderCam).

**Lessons (each cost a round; keep them):**
- One depth sheet with patches is wrong; separate layers are right. MoGe depth is too smooth to find layers: SAM does.
- Cut-outs were the weak link (the user spotted it): SAM's outlines are loose, and the gap SAM's sky leaves round an
  object got filled by depth, which MoGe blurs across edges: a band of sky moved with the car. BiRefNet fixed it.
- Inpainting must not leave a seam (the user's point): mask exactly the hole (growing it into real pixels and pasting
  them back left a hard join); Flux's autoencoder shifts tone slightly, so measure the shift on the same pixels just
  outside the hole and carry it smoothly across (`membrane`). Choke the far layer 2 px next to a nearer one and bleed
  the near layer's colours into its transparent margin: no outlines or halos left behind.
- Flux Fill: paints an object into an object-shaped hole, duplicates whatever the prompt names, "cinematic" gives
  letterbox bars. Prompts describe what's really there (clouds, not "misty haze"), say "empty", name nothing in view.
- Mesh past a layer's outline must take the layer's own depth (else streaks as the camera moves); Arnold needs
  aiOpaque off; Maya's projection node fits horizontally (projCam filmFit horizontal), measured.
- FLUX.1 Fill loads fp8-stored (23.8 GB bf16 spills to system RAM on the 4090) from
  C:\AI_Models\huggingface\FLUX.1-Fill-dev (hard links; the hub copy is WSL symlinks Windows can't read).
  BiRefNet: ZhengPeng7/BiRefNet in C:\AI_Models\huggingface (needs timm, kornia: installed in tools/.venv).

**NEXT (in the order discussed):**
1. Gen-Fill style painting (the user's Photoshop habit): only the band the camera can reveal, on crops at full
   resolution, 3 variants with the best picked automatically (edge match + SAM check), retry instead of the smear
   fallback (the last pale sliver under the car is that fallback).
2. The Claude API step that writes parts.json and checks layers_overlay/layers_painted (prompt rules above).
3. Input path for the user's own cut-outs (PSD or PNGs, named near-to-far) and painted plates; anything painted beyond
   the visible part counts as real. Plus automatic handling of contours inside a layer (strips where the move
   reveals more than a few px), reported to the user.
4. Adaptive / quad meshes (now 2 px grids, ~1M triangles a layer: too dense to edit in Maya).
5. Speed: keep models loaded, 28 Flux steps, crops (estimates: H100 ~1.5-2 min a scene after that).
**Product notes:** FLUX.1-dev licence is non-commercial: compare models when it becomes a product (BFL licence or
FLUX Fill [pro] API vs Apache Qwen-Image-Edit / Flux 2 Klein 4B). On-demand serverless GPUs (Modal / RunPod /
Replicate), weights (~40 GB) cached on the provider's volume. A server can't run mayapy: .abc via PyAlembic, .ma as
text. Rough cost per image: Claude ~$0.25-0.30 (Opus 5.5), GPU ~$0.20-0.50.

## Handoff 2026-10-04 night

**Where we are.** The main scene is `scenes/canyon-vista` with its own graph `graphs/vista.json` (loaded automatically):
http://localhost:5173/?scene=canyon-vista&track=doomsday_clock-thomas_barrandon.m4a . The user is happy with it
("this is great"). Everything is committed and pushed.

**NEXT, first thing: a real sunrise (the user's words: "it's a hard ball and as soon as you see the edge the top sliver
starts flaring and the flare gets bigger till it's full size and it moves to the centre of the ball till the centre has
cleared the mountains").** Right now a soft glow ball appears first and the flare then switches on, which reads fake.
Plan (not started):
1. Hard sun disc drawn in the shader only on sky pixels (fs, where `sky`): angle between the pixel's view ray
   `normalize(viewPos(uv, 1))` and the sun direction `normalize(u.lightPos)` under a disc radius (new uniform, e.g.
   sunD.y, ~0.6 deg for drama; the real sun is 0.27). Crisp edge, so the ridge cuts it exactly. Add it after the
   brightness cap, next to `lensFlare` (scene.wgsl, `if (u.flare > 0.0) { color += lensFlare(...) }`).
2. In `display.ts` `sunScreen()`: instead of 9 samples in a ring, sample a grid over the disc (in image uv, through
   `depthAt` against `info.far * 0.98` = open sky). flareVisible = visible fraction (try fraction^0.6 so a sliver
   already flares); return the CENTROID of the visible samples as the flare centre (not the disc centre), so the flare
   starts on the top sliver and slides to the middle as the sun clears the ridge.
3. Remove the soft cores: the bloom's `exp(-r*r*900)*6` "disc, burnt out" term in `lensFlare` and shrink the base
   `exp(-r*r*60)*1.2` core; the disc is real now. Keep halo, glow, streak, rays (Sun `bloom`, uniform sunD.x at
   uniforms[88]).
4. Check frame by frame around drop 1 (1:30-1:42): edge appears, sliver flares, flare grows and moves down to centre.
Note: the user moved the sun sliders live before saving: vista graph Sun maxArc 0, minArc -33 (min off). Ask what they
intend before changing them; with maxArc 0 the sun never goes above the skimming height.

**Song layout (Doomsday Clock, vista graph).** Drops at 1:36, 2:53, 3:31 (`Setup.drops` picks drops by number).
- Nook lights on the tick-tock from the start. Sun rises over 3 s into each drop, holds 5 s, sets (sunRise); height per
  drop from sunHeight; drop 1 dimmer. Sunset bloom flare with it. Sun direction fixed straight into the gap
  (a sideways swing looked like a UFO).
- Drop 1: ground rigs' fans open in a wave from the nearest rig to the farthest over 4 s (SkyLaser `wave`).
- Drop 2: only the blue-purple mega fan scan (two rim rigs, MRI-like lines with trails, 1/3 speed until past the spire,
  then 2x, camera-hit laser flare on confident drops). Ground lasers sit it out.
- Drop 3: ground rigs with full fans (Peaks lead 0.2 s: fans open over ~6 frames into the hit, hold 1 s).
- Rigs turn smoothly to new aims (no one-frame jumps). Beams thin, depth through brightness, world-space haze texture.

**Scene build (in order):**
1. `tools/.venv/Scripts/python tools/layers_from_markup.py scenes/canyon-moge images/markup_layers.webp scenes/canyon-layers-moge`
2. `tools/.venv/Scripts/python tools/vista_plate.py scenes/canyon-layers-moge images/canyon_midground.png scenes/canyon-vista --middle images/canyon_middle.png --mountains 0.33`
3. `tools/.venv/Scripts/python tools/river_flow.py scenes/canyon-vista` (always after 2: it writes the river mask and flow
   directions into the vista layer's albedo/normal alpha; uses the user's painted `images/river_mask.png`).

**New this session (all nodes/settings live in the editor, E, and the music monitor, G):**
- Three cut-out layers (cave, the user's pillar/spire layer, vista plate) with depth pre-pass and 4x MSAA alpha to
  coverage (anti-aliased outlines). Vista plate dehazed (capped), mountains sunk to a third (--mountains).
- Sky node: haze by hand (haze, hazeNear, hazeFar in metres) plus sliders on the monitor.
- Water node ("River"): flow-map foam along the river's own course (banks averaged, heads for its exit), curl-noise
  swirl, sheen (Fresnel sky reflection), Blinn glints; features scale with distance.
- Sun node: minArc/maxArc (+ monitor sliders), rise, bloom (sunset flare). Peaks: hold, lead. Setup: drops, lead.
  SkyLaser: wave, sweepOneWay, clearAt/nearSpeed/farSpeed, onVista, planeRoll, spin, camHit, flare. DropHit: number.
- Performance: ~5 ms/frame at 1080p on the 4090 in busy moments.

**Open items:** the drop-beat brightness jump from the sun's on/off gate ("ember" 0.2 -> 1 at once; easing it is a
choice); the orange rock column between the spire and the right cave wall is in no cut-out (the plate's repainted
version shows; the user could add it to the middle layer); uncovered strips of layers behind get no image-space shadow;
drop 2 not yet reviewed in a clip since the sun changes. Ideas raised: a general camera-mapping tool (split into
layers, find where each needs back information, inpaint, depth per layer) = the one-command image-to-scene goal;
log this session's plain-language requests and the mechanisms they became in the Doppler brief's "Lessons for
Doppler" (offered, not done yet).

## What it is

A realtime music visualizer: an AI landscape (a canyon seen from inside a cave) turned into a 2.5D scene, lit
by a music-driven sun, sun shafts, lens flare, lasers and an MRI-style scan. Shown full screen on a TV, mainly
from the user's desktop RTX 4090. The original brief (`landscape-music-visualizer-brief.docx`, not in the
repo) asked for TypeScript + WebGPU in the browser; the user later asked for everything to become procedural
nodes, built "while eating our lunch", with a possible move to native Rust + wgpu later.

## Name

Repo renamed from `doppler` to `doppler-canyon` (github.com/jkarian/doppler-canyon) on 2026-10-03. The name
"Doppler" is reserved for the planned plain-language node tool ([brief](brief-plain-language-nodes.md)); this
visualizer is meant to become its first scene. The local folder is `D:\Projects\doppler-canyon`.

## Running it

```bash
npm run dev
```

- Display: http://localhost:5173/?track=doomsday_clock-thomas_barrandon.m4a (the user's current favourite track)
- Editor: press `E` on the display, or http://localhost:5173/editor.html (same browser; live link)
- `H` toggles the overlay with all keys. `K` records a 12 s clip into `captures/`, then
  `tools/.venv/Scripts/python tools/encode_clip.py` makes an mp4 with audio. `P` saves a frame, `B` benchmarks.
- Not in git (see `.gitignore`): `images/` (source image), `scenes/` (generated maps), `music/` and `tracks/`
  (the user's tracks), `captures/`, `tools/.venv` (Python with CUDA torch), `tools/models`.

## How it's built

- `tools/scene_prep.py`: one image to photo, albedo (Marigold delighting, detail transferred back), haze,
  depth (Depth Anything V2 at 1022 for shape, Depth Pro for scale and lens, SegFormer sky mask, distant
  mountains pushed back x4), tiled Marigold normals. Current scene: `images/canyon Topaz Gigapixel 4x scale.png`
  (Photoshop generative expand to 16:9, Topaz 4x) -> `scenes/canyon` at 3840x2160.
- `tools/audio_analysis.py` + `tools/music_structure.py` (analysis v2, 2026-10-03): HTDemucs stems, Beat This!
  beats fitted to constant-tempo pieces and refitted to the kick (test track within 1 ms), 8-bar phrase grid,
  sections intro/build/drop/breakdown/normal/outro from stem novelty + per-track energy, drops with confidence,
  tension/energy/build curves, grid confidence (Bliss: no reliable grid). Plus loudness, bass, kick (sub),
  intensity, hats. About 5 s per track. `tools/shims/torchaudio` stands in for torchaudio (no build for torch
  2.14). v1 files are kept in `captures/analysis-v1/`. Tuned on our 4 tracks only; All-In-One could be added
  as an extra boundary voter if new tracks come out wrong.
- `src/display.ts` + `src/scene.wgsl`: WebGPU renderer. Depth-displaced mesh, camera with parallax sway,
  image-space shadows (enclosed "cave" rules for the sun, thinning with distance), shafts, flare, lasers
  (occluded by rock in front), scan, nook lights (16 lit, 7 shadowed), sky gradient, brightness cap.
  ~2 ms/frame at 1080p on the 4090 with every light on.
- `src/graph/*`: node graph runtime, node library, expression language, plain-language help for every
  setting (`help.ts`). `graphs/default.json` drives everything. Contract: `docs/graph-format.md`.
- `src/editor.ts`: node editor with live values (BroadcastChannel), save via the dev server.
- `src/monitor.ts` (G on the display): music monitor. Track timeline (sections, drops, phrases, energy,
  tension, sun height, a lane per setup), equalizer meters, master sliders (sun, direction, bounce, floor
  light, brightness cap, nook lights, sky, visual lead), a Setups table (on/off, section chips, each setup's
  sliders), pop-up explanations, Save. The display picks up the graph file when it changes on disk and warns
  when it has unsaved edits instead.
- Placement (N on the display, Tab switches module): nook lights (drag, scroll for area, shift+scroll for
  brightness; blue rings = near lights) and laser rigs (drag to move, aim handle for turn/tilt, scroll for the
  search cone). Banner at the top lists the controls.

## Real-world angles

The photo's camera looks down about 23 degrees. scene.json `up` (measured by scene_prep from the flat
ground) gives true vertical; sun height and direction, the gap's horizon, rig turn/tilt, the sky gradient,
flat-ground detection and bounce all use it. Sun heights are true degrees above the horizon.

## Current look (default graph)

Everything is a **setup** (a module) switched by the song's sections through a Setup node: master on/off,
which section kinds it plays in, fades. The user wants setups combined in drops, more intense ones there.

- Nook lights (first setup, on except in the outro): hand-placed amber pools in nooks of the rock, real size
  (so far ones look smaller), cut off by rock in between. Mid/far lights fire on the tick-tock (named sounds:
  tick on the left half, tock on the right, fade 2 s); near lights (under 8000 ft) only on drops (fade 4.5 s).
- Sun (always): rests just above the gap's horizon, spurts on strong bars and drops (heavy, eased); colour by
  height (SunTint: red-orange low to near white high); lights walls and high ground, not the floor
  (floorLight 0, floor = flat ground below -4300 ft real height, or the cave floor); bounce light into shaded faces; gain 3,
  brightness cap 2.5; ceiling (maxArc) set by the user. Sky: gradient by sun height (dusk from the user's
  reference, golden, day) with the photo's clouds as texture.
- Ground laser rigs (builds, drops, normal): six hand-placed rigs re-aiming within 30 degree cones every 2 bars.
- Off for now (the user is isolating setups): god lasers, sky curtains, contour sheet, MRI scan.

## User preferences learned

See the memory notes; in short: heavy, inertial motion (no per-beat twitching), real-world scale where it
makes sense, crisp laser cores that pop from their glow, additive laser light, plausible physical setups
(rigs on the ground), iterate by eye with frames and clips.

## Known issues

- Dark slivers behind rock edges when the camera moves (single-layer depth). Fix in progress: the automatic
  background layer (tools/background_layer.py).
- Far canyon depth is smooth/approximate; contour and scan lines get blobby there (hence `reach`). The
  distant-mountain mask leaks a strip down the canyon centre; trim it when depth is next touched.
- Bliss (first track) loses its first drop's sun with the intensity-based gate, and has no reliable beat grid
  (no drums); beat-locked effects should multiply by `Beat.confidence`. Low priority.
- Old spotlight mode isn't graph-driven.
- Saving from a page that loaded an older graph overwrote newer file changes (fixed: the display now syncs
  with the file). If a node seems to have vanished, check git history for graphs/default.json.

## Research (2026-10-03, overnight)

- [research-camera-mapping.md](research-camera-mapping.md): single image to layered 3D. Tested on the canyon:
  MoGe-2/3 beats our depth (and is MIT); Apple SHARP fills disocclusion perfectly but is research-only;
  a home-made 2-layer LDI with inpainting is the recommended fix; World Labs Marble for full scenes.
  Licence warning: Depth Anything V2 Base/Large are non-commercial.
- [research-music-understanding.md](research-music-understanding.md): tested on our tracks. Our downbeats
  are fine; our beat grid wobbles (up to about ±130 ms) and section boundaries miss the bar lines. Recommended:
  HTDemucs stems -> Beat This! grid -> All-In-One/SongFormer boundaries voted onto an 8-bar phrase grid ->
  our own stem-based drop/build labels -> tension, energy and anticipation curves. Bliss needs a no-grid mode.
- Test scripts, outputs and Windows workarounds for both are kept locally in `captures/research/`.

## Guiding principle

Image in, living scene out, automatically. Manual 3D work (Maya) dilutes the value and is out of the plan.
Every scene guess (depth, normals, pitch, background, shadow thickness) is estimated automatically, with a
slider to correct by eye. Goal: one command from image to scene. The user's own work is taste (placing and
tuning), ideally starting from automatic suggestions (lights in nooks, rigs on ledges).

## Plan (agreed 2026-10-03)

1. Lasers: done for now (hand-placed rigs, rotate rings, real-world angles, sky reach).
2. Camera mapping (in progress):
   - `tools/depth_moge.py`: MoGe-2 depth (mapped onto the old depth range) and normals into a new scene folder
     (`scenes/canyon-moge` = MoGe depth + our normals, `scenes/canyon-moge-n` = MoGe depth + MoGe normals;
     view with `&scene=...`). MoGe-3 needs flex_gemm, which needs a compiler: MoGe-2 used.
   - MoGe normals are clearly better (crisper strata, sharper terminators): now in `scenes/canyon` too
     (old ones kept as `normal_marigold.png`).
   - MoGe depth: crisper silhouettes, but more seams when the camera moves, so it needs the background layer.
   - `tools/background_layer.py`: second layer behind every silhouette the renderer tears, band sized to the
     camera's largest move, filled from the far side (opencv quick fill, or SDXL inpainting: --fill sdxl);
     the display draws it behind the main layer (`?bg=off` to compare). Seams gone; fill quality next.
   - Camera pitch is uncertain (5-23 deg depending on the measure); kept at 23 so tuned angles hold.
   - Not done yet: switching the main scene to MoGe depth; "Regenerate from depth" (fallback, optional).
   - **Dreamed views -> splat (2026-10-03, `tools/world/`, runs in WSL2 Ubuntu 24.04 on the 4090).** One photo
     -> Stable Virtual Camera dreams a camera move (`dream_views.sh`, non-commercial licence) -> SeedVR2 2x
     upscale as one clip (`upscale_views.sh`, Apache) -> MoGe with the dreamed lens (`moge_views.py`, own venv)
     -> gsplat training (`splat_train.py`): cameras by PnP + bundle adjustment (the video model doesn't follow
     its own camera path), MoGe depth corrected by distance to match the dreams, hidden-area seed points, depth
     guidance, the real photo on half the steps -> `bake_splat.py` (visible surface + peeled "behind" layer at
     4K) -> `make_splat_scene.py` builds `scenes/canyon-splat` (view `&scene=canyon-splat`). Fly-around:
     `view_splat.py` (localhost:8080); `.ply` export: `export_ply.py`. Setup: `setup_wsl.sh`, `setup_moge.sh`.
   - User verdict: spire much better, left near wall better, still some artifacts; less than hoped from that
     much work. The splat is good only along the dreamed path (one spiral); further moves (left/right/up/
     forward, `dream_and_upscale.sh`) were started and stopped, not yet run. Splat look is soft (boil averages
     out). The user is now researching how 3D-world systems (Marble, HunyuanWorld/Voyager, Lyra...) do it:
     their key trick seems to be memory (render the 3D so far, dream only the holes, add, repeat).
   - Models live in C:\AI_Models (Hugging Face cache in C:\AI_Models\huggingface; Wan 2.2 already there).
   - **Tripo model -> scene (works best so far for the spire):** the user's Tripo mesh (3d/tripo/model.obj) ->
     `align_mesh.py` (finds the photo's camera on the mesh, bakes front + behind layers) -> `make_splat_scene.py
     scenes/canyon scenes/canyon-tripo --bake scenes/canyon/world/tripo --flow-align --parts` (optical flow onto
     the photo, then each freestanding part of parts.json snapped onto its outline by template matching).
     Spire and left mesa snap; fin and tree too uncertain. User's idea for later: camera-project the photo onto
     the mesh, bake into its UV texture, repair stretched texels (a real textured mesh: archetype 3 renderer).
   - Maya hand-off: `3d/tripo/tripo_clean_unlevelled.ma`/`.abc` (200k mesh `canyon_mesh` + `tripo_cam1`, real size in
     cm, 1400 ft per Tripo unit, no groups) is the one the user works in. Levelling to our 23-degree pitch made the
     spires lean back: the user judged Tripo's near-level camera (-2.4 deg) right. Our scene's pitch (scene.json "up")
     may be too steep (earlier estimates ranged 5-23 degrees); revisit, as it sets the sun's and rigs' real angles.
   - **Layer markup (2026-10-04):** the user painted the layers (images/markup_layers.webp: frame, pillars+spire, near
     canyon, far canyon, mountains) with distances (images/markup_distances.webp: 30 m, 0.3 km, 3 km, 30 km, river to
     plateau 500 m). `tools/layers_from_markup.py` -> `scenes/canyon-layers` (true scale in metres, metersPerUnit 1;
     display uses it for feet). Renders, but the graph was tuned on the old squashed scale (nook pools 800 ft swamp the
     3-30 m cave; GapHorizon reads -43): needs a retune before it looks right. images/ is git-ignored (local only).
   - **Matrix-3D open vista (overnight, works):** fed only the view through the opening (crop with the markup:
     scenes/canyon/world/matrix_vista/vista_input.jpg) so the panorama opens up instead of rebuilding the cave.
     Results in scenes/canyon/world/matrix_vista/: pano_img.jpg (a wide-open cliff-top canyon), pano_video.mp4
     (81 frames, 1440x720), generated_3dgs_opt.ply (694 MB splat), snapshots/ (6 views from the centre; view with
     `bash tools/world/run.sh tools/world/view_splat.py scenes/canyon/world/matrix_vista/generated_3dgs_opt.ply` on
     :8080). Soft: its StableSR step fails (`No module named 'taming'`), so it trained on un-upscaled views. Fixes made
     on the way: swap 32 GB (C:\Users\johnk\.wslconfig), protobuf 6.31.1, StableSR checkpoints.
     Next idea: our near layers (cave, pillars, spire from the photo + markup) in front of this splat as the vista.
   - **Sharper vista (2026-10-04):** `tools/world/matrix3d_hires.sh scenes/canyon/world/matrix_vista` -> SeedVR2 2x of
     the panoramic video (2880x1440, one clip), Matrix-3D's 3D step at that size (`matrix3d_recon.py`: no StableSR,
     15,000 iterations, densify only to 1,501 or the Gaussians fill the card) -> scenes/canyon/world/matrix_vista_hr/
     generated_3dgs_opt.ply (728 MB). Clearly sharper (peaks, cliff edges, rock texture), not photo-sharp up close.
     Training ~18 min; the depth step ~25 min (writes big per-keyframe meshes to D:).
   - **Vista framing (2026-10-04):** `tools/world/frame_vista.py` composites a vista splat behind the photo's near
     layers (markup frame + pillars/spire), colour-matched in Lab. Cleanest from points on the generator's own path
     (`--at 0,0,1.48`, yaw 0 or -12, pitch +2); off-path views turn to confetti. Verdict: the combination reads as one
     place, but the splat is soft (mid-distance rock blurs) and straight ahead Matrix's world is a gorge, not vast.
     **User: "I need things sharp."** => splats from a 720p world model are the wrong tool for the background
     (~8 px/degree vs the photo's ~35).
   - **NEXT (handoff 2026-10-04):** matte-painting background: generate the vista as a high-res 2D plate from our
     camera's view (FLUX Fill here, or the user's Qwen-Image / Flux 2 Klein / Z-Image in C:\AI_Models), upscale 4-8K
     (SeedVR2 or the user's Topaz), MoGe depth mapped to the markup's distances (3 km, 30 km) for subtle parallax;
     the photo's near/mid layers (cave, pillars, spire) in front, 4K and real. **Open question for the user:** (a) keep
     the photo's real canyon through the opening and paint only what's hidden/beyond the frame, or (b) repaint all
     beyond the pillars as the vast open vista. Then build the scene from canyon-layers (true scale) and retune the
     graph for metres (nook radii, near threshold, GapHorizon read -43 there).
   - **Matte-painted vista, three layers (2026-10-04, works; user: "def an improvement"):** the user painted
     images/canyon_midground.png (real canyon through the opening kept, the rest repainted as an open vista) and
     images/canyon_middle.png (pillar + spire cut out, with what the cave walls hide of them painted, RGBA).
     `tools/vista_plate.py scenes/canyon-layers images/canyon_midground.png scenes/canyon-vista --middle
     images/canyon_middle.png` -> scenes/canyon-vista: main = photo (cave), then middle layer, then vista plate.
     Depth on each plate on its own (MoGe-2), mapped onto the markup distances: fixes the far gorge flattening at the
     river bend (the scan now recedes along it). The renderer takes a list of layers behind the main one.
     All three are cut-outs that stack (lesson learned the hard way): the cave and the middle layer are see-through
     outside their outlines (alpha, discarded in the shader), with their mesh depth carried past the outline so no
     triangle stretches across a depth step (scene.json `meshDepth` for the cave; `depth` stays the whole picture's,
     for shadows/shafts/sky). Pushing stretched edges back or dropping them by a depth threshold both failed
     (spikes; cracks in the cave). The cave is only what the markup paints as cave.
     View: `?scene=canyon-vista&graph=vista-test&cam=sweep` (vista-test: MRI scan always on, over 100 m-3 km;
     `cam=sweep` / `cam=x,y` move or hold the camera for parallax checks). Lighting retuned for true scale in
     graphs/vista.json (the scene's own graph via scene.json "graph"): nook pools rescaled per light, near split
     820 ft, sun floor -1100 ft; aerial perspective toward the sky gradient (haze from ~150 km visibility); sky from
     SegFormer. Build now: `layers_from_markup.py scenes/canyon-moge images/markup_layers.webp scenes/canyon-layers-moge`
     then `vista_plate.py scenes/canyon-layers-moge images/canyon_midground.png scenes/canyon-vista --middle
     images/canyon_middle.png` (MoGe cave depth: the old fused depth's radial streaks cast straight shadow lines).
     Lasers have real width on true-scale scenes (angle at 500 m); rig fans hold 1 s (Peaks hold). Depth pre-pass
     keeps three layers near single-layer cost.
     **Lasers (2026-10-04 evening, user happy):** beams thin and crisp, depth through brightness (laserNear), haze
     texture in world space (airDensity). Rig fans ramp open over ~10 frames before the hit (Peaks lead), hold 1 s.
     MRI scan replaced by two blue-purple **mega fan lasers** (megaLeft/megaRight in graphs/vista.json) on the canyon
     rims (SkyLaser onVista, planeRoll 90, sweepOneWay): upright fans whose 2-inch line + 4 m trail crawls from just
     short of the camera to the end of the gorge, 1/3 speed until it passes the spire, then 2x (clearAt). Drop 2 only
     (Setup `drops`: "2"); the ground lasers sit drop 2 out ("-2"). On confident drops the right fan sweeps through
     the camera: laser lens flare (user: "whoa that laser flare is cool"). Waiting on the user: a cut-out of the orange rock column between the
     spire and the right cave wall (in no cut-out, so the plate's repainted version shows). Open items: uncovered strips of layers behind get no image-space shadow (better than the wrong
     one; per-layer shadows would be the real fix); the river floor looks bright at the drop; the dark nook at the left mesa's base is in the painting; mountains map to ~13 km
     (markup says 20-30). User's idea, agreed direction: a general camera-mapping tool (split into layers, find
     where each needs "back information", inpaint it, depth per layer) = the one-command image-to-scene goal.
   - **Matrix-3D (parked, run last):** environment built (`setup_matrix3d.sh`, venv-matrix), models in
     C:\AI_Models\matrix3d, panorama done (scenes/canyon/world/matrix/pano_img.jpg). The video step is killed for
     out of memory (24 GB WSL RAM, 8 GB swap). Plan agreed with the user: after a PC restart, with nothing else
     running, `wsl -d Ubuntu-24.04 -- bash /mnt/d/Projects/doppler-canyon/tools/world/matrix3d_scene.sh
     scenes/canyon unused` (resumes at the video step). If still killed, more swap in C:\Users\johnk\.wslconfig
     (ask first). Caveat: the panorama gives the canyon beyond the opening only ~100 px of 1600.
3. Patterns. Structure: Scene -> Arrangement (when: which pattern on which setup in which song part) ->
   Pattern x Setup (how x what; patterns reusable across setups) -> Objects -> Elements -> Nodes -> Code.
   Foundation: a shared Houdini-style point/attribute model (every element has position, colour, brightness,
   group, index...), a spreadsheet view, per-element expressions (wrangle-like), groups by rule. Start with
   nook lights and rigs sharing it, plus 2-3 patterns (chase, alternate sides, ripple).

## Backlog / ideas

- Open from the session: should the rigs' brightness stop depending on the sun's gate (the user hasn't
  answered); a "what drives what" matrix in the monitor (sounds x setups); more named sounds / instrument
  nodes (by example); bring the other setups back one by one and decide which combine in drops.
- Use the tension/phrase signals in the look (wind-up before drops, blackout in gap bars, laser pattern
  changes on phrase lines).
- Camera mapping next: MoGe depth first; if not good enough, a separate "Regenerate from depth" step/node
  (depth-conditioned image generation, optional, not the default path).

0. The plain-language node brief has a paper exercise for the user to do before any code.

1. Head tracking: webcam face/eye tracking (e.g. MediaPipe in the browser) moving the camera for real
   parallax as the viewer moves. Would be a Head node feeding Camera.
2. Temporal palette: a Palette node that shifts colours over the track for all lights.
3. Background inpainting layer (fixes edge slivers, allows bigger camera moves).
4. Particles and glow (milestone 3), moving clouds and cloud shadows, more laser types (tunnels/cones,
   chasers), raise the 4-fixture laser limit.
5. One command, image to scene: scene_prep + MoGe + background layer + up estimate, good defaults.
6. Possible port to Rust + wgpu: shaders and graph files carry over; runtime and editor need rewriting.
