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

import { Music, seeded, type Analysis } from "./music.ts";
import { GraphRuntime, type Graph } from "./graph/runtime.ts";
import type { RenderOut } from "./graph/nodes.ts";

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
const graphName = params.get("graph") ?? "default";

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

  const UNIFORM_FLOATS = 56;
  // Lasers: header (count) + 4 fixtures x 24 floats + 96 beams x 4 floats. Layout matches `Lasers` in scene.wgsl.
  const LASER_FLOATS = 4 + 4 * 24 + 192 * 4;
  const laserData = new Float32Array(LASER_FLOATS);
  const laserCount = new Uint32Array(laserData.buffer, 0, 1);
  const laserBuf = device.createBuffer({ size: LASER_FLOATS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
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
      { binding: 8, resource: { buffer: laserBuf } },
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
  // Camera motion comes from the graph's Camera node, in units of the sway amount.
  let graphOut: RenderOut = {};
  const camPos = (_t?: number): Vec3 => {
    if (!cam.sway || !graphOut.camera) return [0, 0, 0];
    const a = (cam.swayAmount * tanHalfFov * aspect) / parallaxSpan;
    const c = graphOut.camera;
    return [c.swayX * a, c.swayY * a, c.pushZ * a];
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
  // The lowest open sky in the sun's direction, seen through the gap (degrees above the camera's horizon):
  // the Gap horizon node, so graphs can place the sun relative to the picture.
  const gapHorizon = (() => {
    const u = 0.5 + Math.tan((look.sunAzimuth * Math.PI) / 180) / (tanHalfFov * aspect) / 2;
    let lowest = -1;
    for (let v = 0; v <= 1; v += 0.002) {
      for (const du of [-0.01, 0, 0.01]) if (depthAt(u + du, v) >= info.far * 0.98) lowest = Math.max(lowest, v);
    }
    if (lowest < 0) return 10; // no sky above the sun: assume a low horizon
    return (Math.atan((1 - 2 * lowest) * tanHalfFov) * 180) / Math.PI;
  })();

  // --- Node graph ------------------------------------------------------------------------------------
  // The graph drives the sun, camera and tone. The editor (editor.html) talks to this page over a
  // BroadcastChannel: it sends graph edits, this page sends back live node values.
  const graphRes = await fetch(`graphs/${graphName}.json`);
  if (!graphRes.ok) fail(`No graph graphs/${graphName}.json`);
  const runtime = new GraphRuntime((await graphRes.json()) as Graph, { music, scene: { gapHorizon } });
  const channel = new BroadcastChannel("doppler");
  channel.onmessage = (e: MessageEvent) => {
    const msg = e.data;
    if (msg?.type === "graph") runtime.load(msg.graph as Graph);
    else if (msg?.type === "hello") channel.postMessage({ type: "graph", graph: runtime.graph, name: graphName, from: "display" });
  };
  let lastSent = 0;
  const sendValues = (now: number) => {
    if (now - lastSent < 66) return; // ~15 updates a second is plenty for the editor
    lastSent = now;
    channel.postMessage({
      type: "values",
      t: time,
      track: trackName,
      playing: !audio.paused,
      values: Object.fromEntries(runtime.values),
      errors: Object.fromEntries(runtime.errors),
    });
  };
  let skyBoost = 0;
  const graphLook = (base: Look): Look => {
    const l = { ...base };
    const tone = graphOut.tone;
    if (tone) Object.assign(l, { baseDim: tone.baseDim, baked: tone.baked, cap: tone.cap });
    const sun = graphOut.sun;
    skyBoost = 0;
    if (sun && l.sun) {
      Object.assign(l, { sunArc: sun.arc, sunAzimuth: sun.azimuth, intensity: sun.intensity, color: sun.color as Vec3, rays: sun.rays, flare: sun.flare });
      skyBoost = sun.skyBoost;
    }
    if (sun && !sun.on) l.sun = false;
    return l;
  };

  const animatedLook = (t: number): Look => {
    if (look.sun) return graphLook(look);
    if (!music || !autoLight) return graphLook(look);
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
    else if (key === "e") window.open(`editor.html?graph=${graphName}`, "doppler-editor");
    else if (key === "b") benchmark();
    else if (key === "t") look.sun = !look.sun;
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
    graphOut = runtime.evaluate(time);
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
      ...sunScreen(look), look.sun ? skyBoost : 0, 0,
      ...scanUniforms(graphOut.scan),
    ]);
    uniforms[39] = look.sun ? look.flare * flareVisible : 0; // after sunScreen() measured visibility
    device.queue.writeBuffer(uniformBuf, 0, uniforms);
    writeLasers(graphOut.lasers ?? [], graphOut.skyLasers ?? []);

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

  // Scan: map the node's 0..1 position and spacing onto the land's range on its axis
  // (log distance for depth, so the sweep moves evenly from the cave mouth to the mountains).
  const landRange = (() => {
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    for (let y = 0; y < H; y += 8) {
      for (let x = 0; x < W; x += 8) {
        const z = depth[y * W + x];
        if (z >= info.far * 0.98) continue;
        const pnt = viewPos((x + 0.5) / W, (y + 0.5) / H, z);
        const c = [Math.log(Math.max(z, 1e-3)), pnt[1], pnt[0]];
        for (let i = 0; i < 3; i++) {
          lo[i] = Math.min(lo[i], c[i]);
          hi[i] = Math.max(hi[i], c[i]);
        }
      }
    }
    return { lo, hi };
  })();
  const scanUniforms = (s: RenderOut["scan"]): number[] => {
    if (!s) return new Array(12).fill(0);
    const span = landRange.hi[s.axis] - landRange.lo[s.axis];
    const front = landRange.lo[s.axis] + span * s.position;
    // Thickness: for depth it's in log units (constant on screen), otherwise it's scaled in the shader.
    const thick = s.axis === 0 ? 0.006 * s.thickness : s.thickness;
    return [s.axis, front, s.spacing * span, thick, ...(s.color as Vec3), s.intensity, s.lines, 0, 0, 0];
  };

  // Lasers: origin, fan basis, and each beam's length to the first rock it hits (marched through the
  // depth map, the same "solid behind the surface" rule as the sun's shadows).
  const rad = Math.PI / 180;
  const beamLength = (o: Vec3, d: Vec3, maxLen: number): number => {
    const steps = 160;
    let prev = 0;
    for (let k = 1; k <= steps; k++) {
      const s = maxLen * (k / steps) ** 2;
      const q: Vec3 = [o[0] + d[0] * s, o[1] + d[1] * s, o[2] + d[2] * s];
      if (q[2] <= 0.02) return s; // past the camera
      const u = q[0] / (q[2] * tanHalfFov * aspect) * 0.5 + 0.5;
      const v = 0.5 - q[1] / (q[2] * tanHalfFov) * 0.5;
      if (u < 0 || u > 1 || v < 0 || v > 1) return maxLen; // leaves the picture: draw it to full length
      const z = depthAt(u, v);
      if (z < info.far * 0.98 && q[2] > z * 1.01 && q[2] < z * 3) {
        // Refine between the last free step and this one.
        let lo = prev;
        let hi = s;
        for (let r = 0; r < 8; r++) {
          const mid = (lo + hi) / 2;
          const m: Vec3 = [o[0] + d[0] * mid, o[1] + d[1] * mid, o[2] + d[2] * mid];
          const mu = m[0] / (m[2] * tanHalfFov * aspect) * 0.5 + 0.5;
          const mv = 0.5 - m[1] / (m[2] * tanHalfFov) * 0.5;
          if (m[2] > depthAt(mu, mv) * 1.01) hi = mid;
          else lo = mid;
        }
        return hi;
      }
      prev = s;
    }
    return maxLen;
  };
  const beamSlot = (li: number, bi: number) => 4 + 4 * 24 + (li * 24 + bi) * 8;
  const writeLasers = (list: NonNullable<RenderOut["lasers"]>, sky: NonNullable<RenderOut["skyLasers"]>) => {
    laserData.fill(0);
    const n = Math.min(4, list.length);
    const maxLen = info.far * 1.5;
    for (let li = 0; li < n; li++) {
      const L = list[li];
      const z = L.originDepth > 0 ? L.originDepth : depthAt(L.originU, L.originV) * 0.98;
      const o = viewPos(L.originU, L.originV, Math.min(z, info.far * 0.9));
      const az = L.azimuth * rad;
      const el = L.elevation * rad;
      const aim: Vec3 = [Math.sin(az) * Math.cos(el), Math.sin(el), Math.cos(az) * Math.cos(el)];
      // Sideways axis of the fan (horizontal at roll 0), then rolled around the aim.
      let right = normalize(cross([0, 1, 0], aim));
      if (!Number.isFinite(right[0])) right = [1, 0, 0];
      const up = cross(aim, right);
      const roll = L.roll * rad;
      right = normalize(right.map((x, i) => x * Math.cos(roll) + up[i] * Math.sin(roll)) as Vec3);
      const normal = normalize(cross(aim, right));
      const half = (L.spread / 2) * rad;
      const base = 4 + li * 24;
      laserData.set([...o, L.count, ...aim, L.sheet, ...right, half, ...normal, L.intensity, ...(L.color as Vec3), L.width * rad, L.hit, maxLen, 0, 0], base);
      for (let bi = 0; bi < L.count; bi++) {
        const a = L.count > 1 ? -half + (2 * half * bi) / (L.count - 1) : 0;
        const d = normalize(aim.map((x, i) => x * Math.cos(a) + right[i] * Math.sin(a)) as Vec3);
        laserData.set([...o, beamLength(o, d, maxLen), ...d, 0], beamSlot(li, bi));
      }
    }
    // Sky lasers: beams from high above onto seeded random spots on the rock, new spots per trigger step.
    let used = n;
    for (const S of sky) {
      if (used >= 4) break;
      const li = used++;
      const step = Math.floor(S.trigger);
      const brightness = S.intensity * Math.exp(-(S.trigger - step) * S.fade);
      const height = info.far * 0.6;
      laserData.set([0, 0, 0, S.count, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1, brightness, ...(S.color as Vec3), S.width * rad, S.hit, maxLen, 0, 0], 4 + li * 24);
      for (let bi = 0; bi < S.count; bi++) {
        const rand = seeded(S.seed * 100003 + step * 101 + bi);
        let u = 0.5;
        let v = 0.5;
        for (let tries = 0; tries < 40; tries++) {
          u = S.uMin + (S.uMax - S.uMin) * rand();
          v = S.vMin + (S.vMax - S.vMin) * rand();
          const z = depthAt(u, v);
          if (z < info.far * 0.9 && z >= S.minDepth) break; // on rock, out in the canyon, not sky
        }
        const target = viewPos(u, v);
        const lean = S.tilt * rad * rand();
        const turn = 2 * Math.PI * rand();
        const o: Vec3 = [target[0] + Math.sin(lean) * Math.cos(turn) * height, target[1] + Math.cos(lean) * height, target[2] + Math.sin(lean) * Math.sin(turn) * height];
        const len = Math.hypot(target[0] - o[0], target[1] - o[1], target[2] - o[2]);
        const d = normalize([target[0] - o[0], target[1] - o[1], target[2] - o[2]]);
        laserData.set([...o, len, ...d, S.sheet > 0.5 ? (S.sheetWidth / 2) * rad : 0], beamSlot(li, bi));
      }
    }
    laserCount[0] = used;
    device.queue.writeBuffer(laserBuf, 0, laserData);
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
    sendValues(now);
    if (!hud.hidden && !recording) {
      const look = animatedLook(time); // what's actually on screen, after the graph
      hud.textContent =
        `${sceneName}  ·  view: ${VIEWS[view]}  ·  shadows ${look.shadows ? "on" : "off"}\n` +
        `base ${look.baseDim.toFixed(3)}  intensity ${look.intensity.toFixed(2)}  cone ${look.coneDeg.toFixed(1)}°  edge softness ${look.coneSoft.toFixed(2)}  cap ${look.cap.toFixed(2)}\n` +
        `target ${fmt(look.target)}  source ${fmt(look.source)}\n` +
        musicLine() +
        (look.sun ? `sun arc ${(animatedLook(time).sunArc ?? look.sunElevation).toFixed(0)}°  graph ${graphName}${runtime.errors.size ? "  (" + runtime.errors.size + " node errors)" : ""}\n` : "") +
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
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const normalize = (a: Vec3): Vec3 => {
  const l = Math.hypot(...a);
  return [a[0] / l, a[1] / l, a[2] / l];
};
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
