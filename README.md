# Doppler

Landscape music visualizer. See the v1 brief for scope. Currently at milestone 1 (look test).

## Run

```bash
npm run dev
```

Open http://localhost:5173/ (canyon) or http://localhost:5173/?scene=butte. Use Chrome or Edge (WebGPU).
The dev server has no dependencies: it uses Node 24's built-in TypeScript stripping.

Controls: drag aims the light · right-drag moves the source (shift: nearer/farther) · wheel cone · shift+wheel intensity ·
arrows pan · `=` `-` zoom · `M` camera sway (parallax) · `;` `'` sway amount · `O` fill screen / fit whole picture ·
`K` record a 12 s 1080p clip to `captures/` (shift+K: 30 s), then `tools/.venv/Scripts/python tools/encode_clip.py` makes the mp4 ·
`T` sun / spotlight (sun: drag places the sun where the pointer is) · `5` `6` sun shafts · `7` `8` how much of the photo's own lighting shows ·
`9` `0` beam edge softness · `B` GPU benchmark · `[` `]` base dim · `,` `.` brightness cap · `S` shadows · `V` debug views · `P` save 1080p frame to `captures/` (shift+P: 4K) · `C` copy values · `R` reset · `F` full screen · `H` hide values.

## Music

```bash
tools/.venv/Scripts/pip install librosa soundfile imageio-ffmpeg
tools/.venv/Scripts/python tools/audio_analysis.py tracks/song.mp3
```

Writes `tracks/song.mp3.analysis.json` (beats, bar starts, loudness and bass curves, quiet/build/drop sections).
Open http://localhost:5173/?track=song.mp3. Space plays. The camera sway follows the bars (one sideways cycle per 4 bars),
grows with the section and loudness, nudges in on beats and pushes in on drops; the searchlight drifts to a new spot on
the rock every bar and flashes on beats. `J` `L` seek · `{` `}` audio/video offset for speaker and TV lag (kept per browser) · `A` auto light.
`tools/make_test_track.py` synthesizes a 120 BPM test track with known sections. Clips recorded with `K` include the audio.

## Scene prep (ahead of time)

```bash
python -m venv tools/.venv
tools/.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu
tools/.venv/Scripts/pip install transformers diffusers accelerate spandrel pillow numpy scipy
curl -L -o tools/models/RealESRGAN_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
tools/.venv/Scripts/python tools/scene_prep.py scenes/canyon/image.jpg
```

Upscales 2x (Real-ESRGAN), removes the baked lighting (Marigold IID, detail transferred back onto the sharp photo), and writes
`photo.png`, `albedo.png`, `shading.png`, `depth.png` (16-bit, editable), `depth.bin`, `normal.png` and `scene.json` next to the image.
After hand-fixing `depth.png`, rebuild with `--from-depth`. `--relief` (default 0.35) sets how steep the normals are.
