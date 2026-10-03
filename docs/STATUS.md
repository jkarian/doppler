# Doppler Canyon: status and handoff

Last updated 2026-10-03. Read this first when picking the project back up.

## What it is

A realtime music visualizer: an AI landscape (a canyon seen from inside a cave) turned into a 2.5D scene, lit
by a music-driven sun, sun shafts, lens flare, lasers and an MRI-style scan. Shown full screen on a TV, mainly
from the user's desktop RTX 4090. The original brief (`landscape-music-visualizer-brief.docx`, not in the
repo) asked for TypeScript + WebGPU in the browser; the user later asked for everything to become procedural
nodes, built "while eating our lunch", with a possible move to native Rust + wgpu later.

## Name

Repo renamed from `doppler` to `doppler-canyon` (github.com/jkarian/doppler-canyon) on 2026-10-03. The name
"Doppler" is reserved for the planned plain-language node tool ([brief](brief-plain-language-nodes.md)); this
visualizer is meant to become its first scene. The local folder is still `D:\Projects\doppler`.

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
- `tools/audio_analysis.py`: beats (aligned to bass hits, ~5 ms), bars, loudness, bass, kick (sub), intensity
  (loudness + brightness, for flat-mastered tracks), hats, sections (quiet/build/drop/normal).
- `src/display.ts` + `src/scene.wgsl`: WebGPU renderer. Depth-displaced mesh, camera with parallax sway,
  image-space shadows (enclosed "cave" rules for the sun), shafts, flare, lasers, scan, brightness cap,
  mipmaps. ~4.4 ms/frame at 1080p on the 4090.
- `src/graph/*`: node graph runtime, node library, expression language. `graphs/default.json` drives the
  sun, camera, tone, lasers and scan. Contract: `docs/graph-format.md`.
- `src/editor.ts`: node editor with live values (BroadcastChannel), save via the dev server.

## Current look (default graph)

- Sun: rests just above the horizon seen through the gap, spurts up on strong bass bars and drops (heavy:
  eased rise, hold, long S-curve back, spring follower), on/off gate from kick + intensity (fast on, slow off,
  20% ember when off), hats flicker, kick pump on shafts and sky.
- Lasers (4 fixtures max): red ground rigs (fixed spots on floor/ledge tops, re-aim every 2 bars, drift);
  sky-down "god lasers" in drops only; magenta sky curtains (contours on the rock); pink sheet from the far
  plateau rim. Beam profile: hard core 100%, glow starts at 50% and falls off. Light on rock fades with
  distance to 30%; `reach` stops rock light before the unreliable far canyon.
- Scan: single depth slice sweeping every 2 bars in drops, beam-style core with an 80 ft trail.

## User preferences learned

See the memory notes; in short: heavy, inertial motion (no per-beat twitching), real-world scale where it
makes sense, crisp laser cores that pop from their glow, additive laser light, plausible physical setups
(rigs on the ground), iterate by eye with frames and clips.

## Known issues

- Dark slivers behind rock edges when the camera moves (single-layer depth). Real fix: inpainted background
  layer, or camera-mapped geometry from Maya (the user can do this).
- Far canyon depth is smooth/approximate; contour and scan lines get blobby there (hence `reach`). The
  distant-mountain mask leaks a strip down the canyon centre; trim it when depth is next touched.
- Bliss (first track) loses its first drop's sun with the intensity-based gate. Low priority.
- Old spotlight mode isn't graph-driven.

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

## Backlog / ideas

0. The plain-language node brief has a paper exercise for the user to do before any code.

1. Head tracking: webcam face/eye tracking (e.g. MediaPipe in the browser) moving the camera for real
   parallax as the viewer moves. Would be a Head node feeding Camera.
2. Temporal palette: a Palette node that shifts colours over the track for all lights.
3. Background inpainting layer (fixes edge slivers, allows bigger camera moves).
4. Particles and glow (milestone 3), moving clouds and cloud shadows, more laser types (tunnels/cones,
   chasers), raise the 4-fixture laser limit.
5. Maya camera-map pipeline (layered depth/normal EXRs or full geometry) when the user wants it.
6. Possible port to Rust + wgpu: shaders and graph files carry over; runtime and editor need rewriting.
