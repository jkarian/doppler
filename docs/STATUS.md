# Doppler Canyon: status and handoff

Last updated 2026-10-03 (evening). Read this first when picking the project back up.

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
