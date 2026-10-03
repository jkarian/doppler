// Display page, milestone 1 (look test): one scene, dim base, one draggable spotlight,
// seen through a camera that fills the screen, pans, zooms and sways for parallax.
//
// No buttons. Mouse and keys:
//   drag            aim the light at a point on the rock
//   right-drag      move the light source (shift: move it nearer / farther)
//   wheel           cone width        shift+wheel  intensity
//   [ ]             base dim          , .          brightness cap
//   arrows          pan (shift: faster)              = -          zoom
//   M camera sway on/off   ; '  sway amount   O fill / fit whole picture
//   with ?track=<file in tracks/>:
//   space play/pause   J L seek 5 s   { } audio/video offset (10 ms; alt: 1 ms)   A auto light on/off
//   S shadows   V cycle debug view   H show values   P save 1080p frame (shift: 4K)   C copy values   R reset   F full screen

import { Music, SunGate, SunMotion, seeded, type Analysis, type Spurt } from "./music.ts";

type Vec3 = [number, number, number];

interface SceneInfo {
  image: string;
  albedo?: string; // de-lit image; falls back to the photo
  width: number;
  height: number;
  depth: string;
  normal: string;
  near: number;
  far: number;
  fovDeg: number;
  haze?: { airlight: number[]; beta: number };
}

interface Look {
  baseDim: number;
  intensity: number;
  coneDeg: number;
  coneSoft: number;
  color: Vec3;
  cap: number;
  shadows: boolean;
  target: Vec3;
  // Where the light comes from, relative to its target, in units of the target's distance from the camera.
  // Relative, so the beam covers the same share of the screen on near rock and far canyon alike.
  source: Vec3;
  baked: number; // 0..1: how much of the photo's own lighting shows in the base
  sun: boolean; // directional sun instead of the spotlight
  sunAzimuth: number; // degrees: 0 = straight into the scene (behind the canyon), negative = left
  sunElevation: number; // degrees above the horizon
  rays: number; // visible sun shafts
  flare: number; // lens flare strength
  sunArc?: number; // music-driven: degrees along the arc (0 front horizon, 90 overhead, 180 behind us)
}

const VIEWS = ["final", "albedo", "depth", "normals", "light only", "shadow mask", "photo", "sun shafts"];

const params = new URLSearchParams(location.search);
const sceneName = params.get("scene") ?? "canyon";
const sceneUrl = `scenes/${sceneName}/`;
const trackName = params.get("track");

const canvas = document.querySelector("canvas")!;
const hud = document.querySelector<HTMLElement>("#hud")!;

function fail(msg: string): never {
  document.body.insertAdjacentHTML("beforeend", `<pre class="error">${msg}</pre>`);
  throw new Error(msg);
}

async function main() {
  if (!navigator.gpu) fail("WebGPU is not available in this browser.");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) fail("No WebGPU adapter found.");
  const device = await adapter.requestDevice();
  device.lost.then((info) => fail(`GPU device lost: ${info.message}`));

  const info: SceneInfo = await (await fetch(sceneUrl + "scene.json")).json();
  const loadBitmap = (name: string) =>
    fetch(sceneUrl + name).then((r) => r.blob()).then((b) => createImageBitmap(b, { colorSpaceConversion: "none" }));
  const [photoBmp, albedoBmp, normalBmp, depthBuf] = await Promise.all([
    loadBitmap(info.image),
    loadBitmap((params.get("delight") !== "off" && info.albedo) || info.image),
    loadBitmap(info.normal),
    fetch(sceneUrl + info.depth).then((r) => r.arrayBuffer()),
  ]);
  const depth = new Float32Array(depthBuf);

  // --- Music ---------------------------------------------------------------------
  let music: Music | null = null;
  const audio = new Audio();
  if (trackName) {
    const res = await fetch(`tracks/${trackName}.analysis.json`);
    if (!res.ok) fail(`No analysis for ${trackName}. Run: python tools/audio_analysis.py tracks/${trackName}`);
    music = new Music((await res.json()) as Analysis);
    audio.src = `tracks/${trackName}`;
    audio.preload = "auto";
  }
  // Audio/video offset for this setup (speaker and TV lag): visuals run this far ahead of the audio.
  let avOffset = Number(safeGet("doppler.avOffset") ?? 0) || 0;
  const { width: W, height: H } = info;
  if (depth.length !== W * H) fail(`depth.bin has ${depth.length} values, expected ${W * H}`);

  const tanHalfFov = Math.tan((info.fovDeg * Math.PI) / 360);
  const aspect = W / H;

  // --- GPU resources -------------------------------------------------------
  // Image textures carry a full mip chain: a 4K scene shown at 1080p (or zoomed out) samples
  // pre-averaged levels instead of skipping texels, which would soften and shimmer fine rock detail.
  const mipModule = device.createShaderModule({
    code: `
      @group(0) @binding(0) var src: texture_2d<f32>;
      @group(0) @binding(1) var samp: sampler;
      struct V { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
      @vertex fn vs(@builtin(vertex_index) i: u32) -> V {
        let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
        return V(vec4f(p * 2.0 - 1.0, 0.0, 1.0), vec2f(p.x, 1.0 - p.y));
      }
      @fragment fn fs(v: V) -> @location(0) vec4f { return textureSampleLevel(src, samp, v.uv, 0.0); }`,
  });
  const mipPipelines = new Map<GPUTextureFormat, GPURenderPipeline>();
  const mipSampler = device.createSampler({ minFilter: "linear", magFilter: "linear" });
  const generateMips = (tex: GPUTexture) => {
    let pipe = mipPipelines.get(tex.format);
    if (!pipe) {
      pipe = device.createRenderPipeline({
        layout: "auto",
        vertex: { module: mipModule, entryPoint: "vs" },
        fragment: { module: mipModule, entryPoint: "fs", targets: [{ format: tex.format }] },
      });
      mipPipelines.set(tex.format, pipe);
    }
    const enc = device.createCommandEncoder();
    for (let level = 1; level < tex.mipLevelCount; level++) {
      const bind = device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: tex.createView({ baseMipLevel: level - 1, mipLevelCount: 1 }) },
          { binding: 1, resource: mipSampler },
        ],
      });
      const pass = enc.beginRenderPass({
        colorAttachments: [{ view: tex.createView({ baseMipLevel: level, mipLevelCount: 1 }), loadOp: "clear", storeOp: "store" }],
      });
      pass.setPipeline(pipe);
      pass.setBindGroup(0, bind);
      pass.draw(3);
      pass.end();
    }
    device.queue.submit([enc.finish()]);
  };
  const imageTexture = (bmp: ImageBitmap, format: GPUTextureFormat) => {
    const tex = device.createTexture({
      size: [bmp.width, bmp.height],
      format,
      mipLevelCount: Math.floor(Math.log2(Math.max(bmp.width, bmp.height))) + 1,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    device.queue.copyExternalImageToTexture({ source: bmp }, { texture: tex }, [bmp.width, bmp.height]);
    generateMips(tex);
    return tex;
  };
  const photoTex = imageTexture(photoBmp, "rgba8unorm-srgb");
  const albedoTex = imageTexture(albedoBmp, "rgba8unorm-srgb");
  const normalTex = imageTexture(normalBmp, "rgba8unorm");
  const depthTex = device.createTexture({
    size: [W, H],
    format: "r32float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  device.queue.writeTexture({ texture: depthTex }, depth, { bytesPerRow: W * 4 }, [W, H]);

  const format = navigator.gpu.getPreferredCanvasFormat();
  const ctx = canvas.getContext("webgpu")!;
  ctx.configure({ device, format, alphaMode: "opaque" });

  const code = await (await fetch(new URL("./scene.wgsl", import.meta.url))).text();
  const module = device.createShaderModule({ code });
  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module, entryPoint: "vs" },
    fragment: { module, entryPoint: "fs", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
    depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
  });
  let zBuffer: GPUTexture | null = null;

  // Shadow passes: half-res visibility in image space, march then depth-aware blur (see scene.wgsl).
  const shadowSize = [Math.ceil(W / 2), Math.ceil(H / 2)];
  const shadowTexture = () =>
    device.createTexture({
      size: shadowSize,
      format: "r16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
  const shadowA = shadowTexture();
  const shadowB = shadowTexture();
  const fullscreen = (entryPoint: string, constants?: Record<string, number>) =>
    device.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vs_full" },
      fragment: { module, entryPoint, constants, targets: [{ format: "r16float" }] },
    });
  const shadowPipeline = fullscreen("fs_shadow");
  const raysPipeline = fullscreen("fs_rays");
  const raysTex = shadowTexture();
  const blurH = fullscreen("fs_blur", { HORIZONTAL: 1 });
  const blurV = fullscreen("fs_blur", { HORIZONTAL: 0 });

  // Mesh density: at most ~1M quads (one per 2x2 texels for a 1472x2208 scene, one per ~3x3 at 4K).
  const step = Math.max(2, Math.ceil(Math.sqrt((W * H) / 1_000_000)));
  const grid = [Math.ceil(W / step), Math.ceil(H / step)];

  const UNIFORM_FLOATS = 44;
  const uniforms = new Float32Array(UNIFORM_FLOATS);
  const uniformBuf = device.createBuffer({ size: UNIFORM_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      {
        binding: 1,
        resource: device.createSampler({ magFilter: "linear", minFilter: "linear", mipmapFilter: "linear", maxAnisotropy: 8 }),
      },
      { binding: 2, resource: albedoTex.createView() },
      { binding: 3, resource: normalTex.createView() },
      { binding: 4, resource: depthTex.createView() },
      { binding: 5, resource: photoTex.createView() },
      { binding: 6, resource: shadowA.createView() },
      { binding: 7, resource: raysTex.createView() },
    ],
  });
  const raysBind = device.createBindGroup({
    layout: raysPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: device.createSampler({ magFilter: "linear", minFilter: "linear", mipmapFilter: "linear" }) },
      { binding: 4, resource: depthTex.createView() },
      { binding: 5, resource: photoTex.createView() },
    ],
  });
  const shadowBind = device.createBindGroup({
    layout: shadowPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 4, resource: depthTex.createView() },
    ],
  });
  const blurBind = (pipe: GPURenderPipeline, input: GPUTexture) =>
    device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuf } },
        { binding: 4, resource: depthTex.createView() },
        { binding: 6, resource: input.createView() },
      ],
    });
  const blurHBind = blurBind(blurH, shadowA);
  const blurVBind = blurBind(blurV, shadowB);

  // --- Scene geometry on the CPU (for picking) ------------------------------
  const depthAt = (u: number, v: number) => {
    const x = Math.min(W - 1, Math.max(0, Math.round(u * W - 0.5)));
    const y = Math.min(H - 1, Math.max(0, Math.round(v * H - 0.5)));
    return depth[y * W + x];
  };
  const viewPos = (u: number, v: number, z = depthAt(u, v)): Vec3 => [
    (u * 2 - 1) * tanHalfFov * aspect * z,
    (1 - v * 2) * tanHalfFov * z,
    z,
  ];

  // --- Look state ------------------------------------------------------------
  const defaults = (): Look => {
    const target = viewPos(0.45, 0.45);
    return {
      baseDim: 0.14,
      intensity: 2.2,
      coneDeg: 12,
      coneSoft: 0.08, // hard-edged beam
      color: [1.0, 0.82, 0.62],
      cap: 1.4,
      shadows: true,
      target,
      // Up and to the left, a little in front of the target: rakes across the rock face.
      // Up-left and well toward the camera: from further back, beams on far rock get blocked by the near walls.
      source: [-0.35, 0.3, -0.65],
      baked: 0.1,
      // Sun low behind the mountains, seen in the gap: backlights the canyon toward the camera.
      // In sun mode, drag to put the sun where the pointer is.
      sun: true,
      sunAzimuth: 3,
      sunElevation: 24,
      rays: 0.6,
      flare: 0.8,
    };
  };
  let look = defaults();
  let view = 0;

  // --- Camera ------------------------------------------------------------------
  // Pivot: median depth of the land (sky excluded). It holds still while the camera sways.
  const land = depth.filter((z) => z < info.far * 0.98).sort();
  const pct = (f: number) => (land.length ? land[Math.floor((land.length - 1) * f)] : info.far / 2);
  const pivotZ = pct(0.5);
  // Depth of the rock around the camera: the sun's shadows treat anything this near as part of the cave.
  const caveDepth = pct(0.7) * 1.5;
  // Parallax range: how much more a near rock moves than the pivot, per unit of camera travel.
  const parallaxSpan = Math.max(1 / pct(0.02) - 1 / pivotZ, 1 / pivotZ - 1 / info.far);
  // swayAmount: largest parallax shift, as a fraction of the picture's half-width.
  const cam = { zoom: 1, center: [0, 0], cover: true, sway: true, swayAmount: 0.05 };
  const camPos = (t: number): Vec3 => {
    if (!cam.sway) return [0, 0, 0];
    const a = (cam.swayAmount * tanHalfFov * aspect) / parallaxSpan;
    if (!music) {
      // Slow Lissajous drift. Deterministic in time.
      // Depth motion only pushes in: pulling back shrinks near rock and exposes the picture's edges.
      return [a * Math.sin(t * 0.3), a * 0.5 * Math.sin(t * 0.21 + 1.3), a * (1 + Math.sin(t * 0.11))];
    }
    // With music: the same drift, but its phase is the bar count (one sideways cycle per 4 bars),
    // its size follows the section and the slow loudness, the camera nudges in on beats,
    // and pushes in hard on a drop.
    const sec = music.section(t);
    const gain = { quiet: 0.45, build: 0.6 + 0.5 * sec.progress, drop: 1.15, normal: 0.85 }[sec.kind];
    const amp = a * gain * (0.5 + 0.5 * music.energy(t));
    const bar = music.bar(t);
    const tau = 2 * Math.PI;
    // One gentle push per bar in drops, nothing per beat: per-beat motion reads as jitter.
    const sinceBar = (bar - Math.floor(bar)) * 4 * (60 / Math.max(music.a.tempo, 1));
    const kick = sec.kind === "drop" ? a * 0.5 * Math.exp(-sinceBar / 0.6) : 0;
    const dropHit = a * 3 * Math.exp(-music.sinceDrop(t) / 0.8);
    return [
      amp * Math.sin((tau * bar) / 4),
      amp * 0.5 * Math.sin((tau * bar) / 8 + 1.3),
      amp * (1 + Math.sin((tau * bar) / 16)) + kick + dropHit, // never behind the rest position
    ];
  };

  // --- Music-driven searchlight ----------------------------------------------------------
  // Waypoints on the rock inside the 16:9 window, one per bar, picked by a seeded RNG.
  let autoLight = !!music;
  const waypoint = (i: number): Vec3 => {
    const rand = seeded(1000 + i);
    const k = aspect / (16 / 9);
    const fill = cam.cover ? Math.max(1, 1 / k) : Math.min(1, 1 / k);
    const half = [1 / (k * fill * cam.zoom), 1 / (fill * cam.zoom)];
    for (let tries = 0; tries < 30; tries++) {
      const qx = cam.center[0] + (rand() * 2 - 1) * half[0] * 0.7;
      const qy = cam.center[1] + (rand() * 2 - 1) * half[1] * 0.7;
      const u = qx * 0.5 + 0.5;
      const v = 0.5 - qy * 0.5;
      if (depthAt(u, v) < info.far * 0.9) return viewPos(u, v);
    }
    return look.target;
  };
  // The sun on its arc. Rest level follows the section (eased over 2 s); spurts kick it up and it crawls
  // back. U kicks it by hand, to try the feel without music.
  const musicSpurts: Spurt[] = music ? music.findSpurts() : [];
  const manualSpurts: Spurt[] = [];
  let sunMotion = new SunMotion(musicSpurts);
  const sunGate = music ? new SunGate(music) : null;

  // Where the sun is on screen (ndc), and how much of it is visible: the fraction of a small disc
  // around it that is open sky. The flare uses both; behind the rock rim it dims.
  let flareVisible = 0;
  const sunScreen = (l: Look): [number, number] => {
    flareVisible = 0;
    if (!l.sun) return [0, 0];
    const d = sunDirection(l);
    if (d[2] < 0.05) return [0, 0];
    const qx = d[0] / (d[2] * tanHalfFov * aspect);
    const qy = d[1] / (d[2] * tanHalfFov);
    const c = camPos(time);
    let sky = 0;
    for (let k = 0; k < 9; k++) {
      const a = (k / 9) * 2 * Math.PI;
      const r = k === 0 ? 0 : 0.012;
      const u = qx * 0.5 + 0.5 + r * Math.cos(a);
      const v = 0.5 - qy * 0.5 + r * Math.sin(a) * aspect;
      if (u >= 0 && u <= 1 && v >= 0 && v <= 1 && depthAt(u, v) >= info.far * 0.98) sky++;
    }
    flareVisible = sky / 9;
    // Same transform as the mesh (pan, zoom, parallax pivot) so the flare sits on the drawn sun.
    const sx = (qx + c[0] / (pivotZ * tanHalfFov * aspect) - cam.center[0]) * viewScale[0];
    const sy = (qy + c[1] / (pivotZ * tanHalfFov) - cam.center[1]) * viewScale[1];
    return [sx, sy];
  };
  const restArc = (t: number) => {
    if (!music) return look.sunElevation;
    let sum = 0;
    for (let k = 0; k < 8; k++) {
      const sec = music.section(t - k * 0.25);
      sum += { quiet: -4, build: 6 + 14 * sec.progress, drop: 20, normal: 12 }[sec.kind];
    }
    return sum / 8;
  };
  const sunLook = (t: number): Look => {
    const lifted = sunMotion.lift(t);
    const h = lifted / 60; // 0 at rest, ~1 a long way up
    const arc = restArc(t) + lifted;
    const up = clamp(arc / 70, 0, 1); // 0 at the horizon, 1 high in the sky
    const above = clamp((arc + 6) / 10, 0, 1); // fades out as it sets
    const dawn: Vec3 = [1.0, 0.55, 0.3];
    const noon: Vec3 = [1.0, 0.95, 0.88];
    const loud = music ? 0.6 + 0.4 * music.energy(t, 2) : 1;
    // Flicker from the hats and shakers: quick dips, like light through moving cloud.
    const flicker = music ? 1 - 0.35 * music.flicker(t) : 1;
    // On while the low end is in: snaps on, fades off.
    const gate = sunGate ? sunGate.value(t) : 1;
    return {
      ...look,
      sunArc: arc,
      color: dawn.map((c, i) => c + (noon[i] - c) * up) as Vec3,
      intensity: look.intensity * above * (0.8 + 0.8 * Math.min(1, Math.max(0, h))) * loud * flicker * gate,
      // Flares most when the sun is low in the frame; scaled by visibility at draw time.
      flare: look.flare * above * gate * flicker * (1 - 0.5 * up),
      // Shafts are a low-sun thing: strongest at dawn, fading as it climbs.
      rays: look.rays * above * (1 - 0.7 * up) * loud * gate,
    };
  };

  const animatedLook = (t: number): Look => {
    if (look.sun && (music || manualSpurts.length)) return sunLook(t);
    if (!music || !autoLight) return look;
    const bar = music.bar(t);
    const i = Math.floor(bar);
    const f = bar - i;
    const ease = f * f * (3 - 2 * f); // drift across the whole bar, easing in and out
    const a = waypoint(i);
    const b = waypoint(i + 1);
    const target = a.map((x, j) => x + (b[j] - x) * ease) as Vec3;
    const sec = music.section(t);
    const sectionGain = { quiet: 0.6, build: 0.8 + 0.4 * sec.progress, drop: 1.2, normal: 1 }[sec.kind];
    const flash = music.beatPulse(t, 0.18) * (sec.kind === "drop" ? 1.2 : 0.5);
    return {
      ...look,
      target,
      intensity: look.intensity * (sectionGain * (0.35 + 0.65 * music.loudness(t)) + flash),
      // Shafts breathe with the music too: they swell with loudness and flash on beats.
      rays: look.rays * (sectionGain * (0.4 + 0.6 * music.energy(t, 1)) + 0.5 * flash),
      coneDeg: look.coneDeg * (1 + 0.3 * music.bass(t)),
    };
  };

  // --- Layout: image ndc -> screen ndc ---------------------------------------------
  let viewScale = [1, 1];
  const updateView = () => {
    const k = aspect / (canvas.width / canvas.height);
    const fill = cam.cover ? Math.max(1, 1 / k) : Math.min(1, 1 / k);
    // Overscan by the sway so the picture's edges never slide into view.
    const over = cam.sway ? 1 + 1.2 * cam.swayAmount : 1;
    viewScale = [k * fill * cam.zoom * over, fill * cam.zoom * over];
    // Keep the window inside the picture.
    for (const i of [0, 1]) {
      const room = Math.max(0, 1 - 1 / viewScale[i]);
      cam.center[i] = clamp(cam.center[i], -room, room);
    }
  };
  const setSize = (w: number, h: number) => {
    canvas.width = Math.max(1, w);
    canvas.height = Math.max(1, h);
    updateView();
  };
  const resize = () => {
    const dpr = window.devicePixelRatio || 1;
    setSize(Math.round(innerWidth * dpr), Math.round(innerHeight * dpr));
  };
  addEventListener("resize", resize);
  resize();

  // Exact at the pivot depth; off by a little parallax elsewhere, which is fine for aiming.
  const toImageUv = (e: PointerEvent | WheelEvent) => {
    const c = camPos(time);
    const qx = ((e.clientX / innerWidth) * 2 - 1) / viewScale[0] + cam.center[0] - c[0] / (pivotZ * tanHalfFov * aspect);
    const qy = (1 - (e.clientY / innerHeight) * 2) / viewScale[1] + cam.center[1] - c[1] / (pivotZ * tanHalfFov);
    return [qx * 0.5 + 0.5, 0.5 - qy * 0.5];
  };

  // --- Input ------------------------------------------------------------------
  let dragging: "target" | "source" | null = null;
  let last = [0, 0];
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId);
    dragging = e.button === 2 ? "source" : "target";
    last = [e.clientX, e.clientY];
    if (dragging === "target") aimAt(e);
  });
  canvas.addEventListener("pointerup", () => (dragging = null));
  canvas.addEventListener("pointermove", (e) => {
    if (dragging === "target") aimAt(e);
    if (dragging === "source") {
      const dx = e.clientX - last[0];
      const dy = e.clientY - last[1];
      const step = 0.004;
      if (look.sun) {
        look.sunAzimuth = clamp(look.sunAzimuth + dx * 0.2, -170, 170);
        look.sunElevation = clamp(look.sunElevation - dy * 0.2, 1, 89);
      } else if (e.shiftKey) look.source[2] = Math.max(-0.95, look.source[2] - dy * step);
      else {
        look.source[0] += dx * step;
        look.source[1] -= dy * step;
      }
      last = [e.clientX, e.clientY];
    }
  });
  const aimAt = (e: PointerEvent) => {
    const [u, v] = toImageUv(e);
    if (look.sun) {
      // The sun sits where the pointer is: its direction is the view ray through that point.
      const dir = [(u * 2 - 1) * tanHalfFov * aspect, (1 - v * 2) * tanHalfFov, 1];
      const len = Math.hypot(...dir);
      look.sunAzimuth = (Math.atan2(dir[0], dir[2]) * 180) / Math.PI;
      look.sunElevation = clamp((Math.asin(dir[1] / len) * 180) / Math.PI, 1, 89);
      return;
    }
    if (u < 0 || u > 1 || v < 0 || v > 1) return;
    if (depthAt(u, v) >= info.far * 0.98) return; // sky: nothing to hit
    look.target = viewPos(u, v);
  };
  canvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      const k = Math.exp(-e.deltaY * 0.001);
      if (e.shiftKey) look.intensity = clamp(look.intensity * k, 0.05, 50);
      else look.coneDeg = clamp(look.coneDeg * k, 0.5, 60);
    },
    { passive: false },
  );
  addEventListener("keydown", (e) => {
    const key = e.key.toLowerCase();
    if (key === "s") look.shadows = !look.shadows;
    else if (key === "v") view = (view + (e.shiftKey ? VIEWS.length - 1 : 1)) % VIEWS.length;
    else if (key === "h") hud.hidden = !hud.hidden;
    else if (key === "r") look = defaults();
    else if (key === "f") document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
    else if (key === "[") look.baseDim = clamp(look.baseDim / 1.15, 0, 1);
    else if (key === "]") look.baseDim = clamp(look.baseDim * 1.15 || 0.01, 0, 1);
    else if (key === ",") look.cap = clamp(look.cap / 1.1, 0.2, 10);
    else if (key === ".") look.cap = clamp(look.cap * 1.1, 0.2, 10);
    else if (e.key.startsWith("Arrow")) {
      const step = (e.shiftKey ? 0.1 : 0.025) / cam.zoom;
      const d = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, step], ArrowDown: [0, -step] }[e.key]!;
      cam.center[0] += d[0];
      cam.center[1] += d[1];
      updateView();
    } else if (key === "=" || key === "+") (cam.zoom = clamp(cam.zoom * 1.1, 1, 6)), updateView();
    else if (key === "-") (cam.zoom = clamp(cam.zoom / 1.1, 1, 6)), updateView();
    else if (key === "m") (cam.sway = !cam.sway), updateView();
    else if (key === ";") (cam.swayAmount = clamp(cam.swayAmount / 1.25, 0.005, 0.3)), updateView();
    else if (key === "'") (cam.swayAmount = clamp(cam.swayAmount * 1.25, 0.005, 0.3)), updateView();
    else if (key === "k") recordClip(e.shiftKey ? 30 : 12);
    else if (key === "o") (cam.cover = !cam.cover), updateView();
    else if (key === "b") benchmark();
    else if (key === "t") look.sun = !look.sun;
    else if (key === "u") {
      manualSpurts.push({ t: time, deg: e.shiftKey ? 40 : 20 });
      sunMotion = new SunMotion(musicSpurts.concat(manualSpurts));
    }
    else if (key === "1") look.flare = clamp(look.flare / 1.25, 0, 5);
    else if (key === "2") look.flare = clamp(look.flare * 1.25 || 0.05, 0, 5);
    else if (key === "5") look.rays = clamp(look.rays / 1.25, 0, 5);
    else if (key === "6") look.rays = clamp(look.rays * 1.25 || 0.05, 0, 5);
    else if (key === "7") look.baked = clamp(look.baked - 0.05, 0, 1);
    else if (key === "8") look.baked = clamp(look.baked + 0.05, 0, 1);
    else if (key === "9") look.coneSoft = clamp(look.coneSoft / 1.4, 0.01, 1);
    else if (key === "0") look.coneSoft = clamp(look.coneSoft * 1.4, 0.01, 1);
    else if (key === "p") capture(e.shiftKey ? [3840, 2160] : [1920, 1080]);
    else if (key === "c") navigator.clipboard?.writeText(JSON.stringify(rounded(look), null, 2));
    else if (music && key === " ") (e.preventDefault(), audio.paused ? audio.play() : audio.pause());
    else if (music && key === "j") audio.currentTime = Math.max(0, audio.currentTime - 5);
    else if (music && key === "l") audio.currentTime = Math.min(music.a.duration, audio.currentTime + 5);
    else if (music && key === "a") autoLight = !autoLight;
    else if (key === "{" || key === "}") {
      avOffset += (key === "}" ? 1 : -1) * (e.altKey ? 0.001 : 0.01);
      avOffset = Math.round(avOffset * 1000) / 1000;
      safeSet("doppler.avOffset", String(avOffset));
    }
  });

  // --- Frame ----------------------------------------------------------------
  let time = 0;
  const draw = () => {
    const look = animatedLook(time);
    if (!zBuffer || zBuffer.width !== canvas.width || zBuffer.height !== canvas.height) {
      zBuffer?.destroy();
      zBuffer = device.createTexture({ size: [canvas.width, canvas.height], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
    }
    uniforms.set([
      canvas.width, canvas.height, W, H,
      viewScale[0], viewScale[1], cam.center[0], cam.center[1],
      tanHalfFov, aspect, info.far, look.baseDim,
      ...(look.sun ? sunDirection(look) : (look.target.map((t, i) => t + look.source[i] * look.target[2]) as Vec3)), look.intensity,
      ...look.target, Math.cos((look.coneDeg * Math.PI) / 180),
      ...look.color, look.coneSoft,
      look.cap, look.shadows ? 1 : 0, view, 0.98,
      ...camPos(time), pivotZ,
      grid[0], grid[1], info.haze?.beta ?? 0, look.baked,
      look.sun ? 1 : 0, look.rays, caveDepth, 0,
      ...sunScreen(look), 0, 0,
    ]);
    uniforms[39] = look.sun ? look.flare * flareVisible : 0; // after sunScreen() measured visibility
    device.queue.writeBuffer(uniformBuf, 0, uniforms);

    const enc = device.createCommandEncoder();
    // march -> A, blur A -> B (horizontal), blur B -> A (vertical); the main pass reads A.
    for (const [pipe, bind, target] of [
      [shadowPipeline, shadowBind, shadowA],
      [blurH, blurHBind, shadowB],
      [blurV, blurVBind, shadowA],
      [raysPipeline, raysBind, raysTex],
    ] as const) {
      const sp = enc.beginRenderPass({ colorAttachments: [{ view: target.createView(), loadOp: "clear", storeOp: "store", clearValue: [1, 0, 0, 0] }] });
      sp.setPipeline(pipe);
      sp.setBindGroup(0, bind);
      sp.draw(3);
      sp.end();
    }
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 1] }],
      depthStencilAttachment: { view: zBuffer.createView(), depthLoadOp: "clear", depthClearValue: 1, depthStoreOp: "discard" },
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(grid[0] * grid[1] * 6);
    pass.end();
    device.queue.submit([enc.finish()]);
  };

  // B: render 120 frames at 1080p back to back and log the average GPU time per frame.
  const benchmark = async () => {
    setSize(1920, 1080);
    draw();
    await device.queue.onSubmittedWorkDone();
    const n = 120;
    const t0 = performance.now();
    for (let i = 0; i < n; i++) draw();
    await device.queue.onSubmittedWorkDone();
    const ms = (performance.now() - t0) / n;
    resize();
    console.log(`benchmark 1080p: ${ms.toFixed(2)} ms/frame (${(1000 / ms).toFixed(0)} fps), mesh ${grid[0]}x${grid[1]}, shadows ${look.shadows ? "on" : "off"}`);
  };

  // Save exactly what the TV would get (P: 1080p, shift+P: 4K) into captures/ via the dev server.
  const capture = async ([w, h]: number[]) => {
    setSize(w, h);
    draw();
    const blob = await new Promise<Blob | null>((done) => canvas.toBlob(done, "image/png"));
    resize();
    if (!blob) return console.error("capture failed");
    const name = `${sceneName}-${VIEWS[view].replace(" ", "-")}-${h}p-${Date.now()}.png`;
    const res = await fetch(`/capture?name=${encodeURIComponent(name)}`, { method: "POST", body: blob });
    console.log(res.ok ? `saved captures/${name}` : `capture upload failed: ${res.status}`);
  };

  // K: render a 12 s clip (shift: 30 s) at 30 fps and 1080p, frame by frame at fixed time steps,
  // into captures/<name>/. Encode it with tools/encode_clip.py. The live view pauses meanwhile.
  let recording = false;
  const recordClip = async (seconds: number) => {
    if (recording) return;
    recording = true;
    const dir = `${sceneName}-clip-${Date.now()}`;
    // In a file there is no speaker or TV lag: frames use the audio time directly.
    if (music) audio.pause();
    const t0 = music ? audio.currentTime : time;
    if (music) {
      const meta = JSON.stringify({ audio: `tracks/${trackName}`, start: t0, fps: 30 });
      await fetch(`/capture?dir=${dir}&name=clip.json`, { method: "POST", body: meta });
    }
    const fps = 30;
    for (let i = 0; i < seconds * fps; i++) {
      time = t0 + i / fps;
      setSize(1920, 1080);
      draw();
      const blob = await new Promise<Blob | null>((done) => canvas.toBlob(done, "image/jpeg", 0.95));
      if (blob) await fetch(`/capture?dir=${dir}&name=${String(i).padStart(4, "0")}.jpg`, { method: "POST", body: blob });
      hud.textContent = `recording ${dir}: frame ${i + 1} / ${seconds * fps}`;
    }
    resize();
    recording = false;
    console.log(`saved captures/${dir}`);
  };

  const frame = (now: number) => {
    if (recording) return requestAnimationFrame(frame);
    time = music ? audio.currentTime + avOffset : now / 1000;
    draw();
    if (!hud.hidden && !recording) {
      hud.textContent =
        `${sceneName}  ·  view: ${VIEWS[view]}  ·  shadows ${look.shadows ? "on" : "off"}\n` +
        `base ${look.baseDim.toFixed(3)}  intensity ${look.intensity.toFixed(2)}  cone ${look.coneDeg.toFixed(1)}°  edge softness ${look.coneSoft.toFixed(2)}  cap ${look.cap.toFixed(2)}\n` +
        `target ${fmt(look.target)}  source ${fmt(look.source)}\n` +
        musicLine() +
        (look.sun ? `sun arc ${(animatedLook(time).sunArc ?? look.sunElevation).toFixed(0)}°  U push sun (shift: big)\n` : "") +
        `zoom ${cam.zoom.toFixed(2)}  ${cam.cover ? "fill" : "fit"}  sway ${cam.sway ? (cam.swayAmount * 100).toFixed(1) + "%" : "off"}\n` +
        `arrows pan · = - zoom · M sway · ; ' sway amount · O fill/fit\n` +
        `drag aim · right-drag move source (shift: depth) · wheel cone · shift+wheel intensity\n` +
        `9 0 beam edge · [ ] base · , . cap · S shadows · V view · C copy · R reset · F full screen · H hide`;
    }
    requestAnimationFrame(frame);
  };
  const musicLine = () => {
    if (!music) return "";
    const sec = music.section(time);
    return (
      `${trackName}  ${audio.paused ? "paused" : "playing"}  ${time.toFixed(2)} s  bar ${music.bar(time).toFixed(2)}  ` +
      `${sec.kind} ${(sec.progress * 100).toFixed(0)}%  offset ${(avOffset * 1000).toFixed(0)} ms  auto light ${autoLight ? "on" : "off"}\n` +
      `space play · J L seek · { } offset · A auto light\n`
    );
  };
  requestAnimationFrame(frame);
}

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));
/**
 * Unit vector toward the sun, in scene space (x right, y up, z into the scene). With an arc angle, the sun
 * sits on the vertical half-circle through its azimuth: 0 front horizon, 90 overhead, 180 behind us.
 */
const sunDirection = (l: { sunAzimuth: number; sunElevation: number; sunArc?: number }): Vec3 => {
  const az = (l.sunAzimuth * Math.PI) / 180;
  const arc = ((l.sunArc ?? l.sunElevation) * Math.PI) / 180;
  return [Math.sin(az) * Math.cos(arc), Math.sin(arc), Math.cos(az) * Math.cos(arc)];
};
// Storage can be unavailable (private windows, blocked site data): the offset just falls back to 0.
const safeGet = (k: string) => {
  try {
    return localStorage.getItem(k);
  } catch {
    return null;
  }
};
const safeSet = (k: string, v: string) => {
  try {
    localStorage.setItem(k, v);
  } catch {}
};
const fmt = (v: Vec3) => v.map((x) => x.toFixed(2)).join(", ");
const rounded = (look: Look) =>
  JSON.parse(JSON.stringify(look, (_k, v) => (typeof v === "number" ? Math.round(v * 1000) / 1000 : v)));

main().catch((err) => {
  console.error(err);
  if (!document.querySelector(".error")) document.body.insertAdjacentHTML("beforeend", `<pre class="error">${err}</pre>`);
});
