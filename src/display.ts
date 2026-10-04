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
//   N place nook lights (click add, shift/right-click remove; shows them all)
//   G music monitor: the track's sections, drops, energy and tension, live meters, and response sliders
//   S shadows   V cycle debug view   H show values   P save 1080p frame (shift: 4K)   C copy values   R reset   F full screen

import { Music, seeded, type Analysis } from "./music.ts";
import { GraphRuntime, type Graph } from "./graph/runtime.ts";
import { createMonitor } from "./monitor.ts";
import type { RenderOut } from "./graph/nodes.ts";

type Vec3 = [number, number, number];

interface SceneInfo {
  image: string;
  albedo?: string; // de-lit image; falls back to the photo
  width: number;
  height: number;
  depth: string;
  meshDepth?: string; // a cut-out main layer's mesh depth, carried past its outline (tools/vista_plate.py)
  normal: string;
  near: number;
  far: number;
  graph?: string; // the graph this scene is tuned with (graphs/<name>.json), when not the default
  metersPerUnit?: number; // real scale, when known (e.g. scenes built from a distance markup)
  fovDeg: number;
  haze?: { airlight: number[]; beta: number };
  up?: number[]; // true vertical in scene space, from the flat ground (the photo's camera looks down a little)
  // What's hidden behind near rock (tools/background_layer.py, tools/vista_plate.py): drawn behind the main layer so
  // camera moves uncover rock instead of dark seams. One layer, or several front to back (e.g. pillars, then vista).
  background?: BgLayer | BgLayer[];
}

interface BgLayer { depth: string; image: string; albedo: string; normal: string; mask?: string }

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
  // The graph: ?graph=name, else the scene's own (tuned to its scale), else the default.
  const graphName = params.get("graph") ?? info.graph ?? "default";
  const loadBitmap = (name: string) =>
    fetch(sceneUrl + name).then((r) => r.blob()).then((b) => createImageBitmap(b, { colorSpaceConversion: "none", premultiplyAlpha: "none" }));
  const [photoBmp, albedoBmp, normalBmp, depthBuf] = await Promise.all([
    loadBitmap(info.image),
    loadBitmap((params.get("delight") !== "off" && info.albedo) || info.image),
    loadBitmap(info.normal),
    fetch(sceneUrl + info.depth).then((r) => r.arrayBuffer()),
  ]);
  const meshDepthBuf = info.meshDepth ? await fetch(sceneUrl + info.meshDepth).then((r) => r.arrayBuffer()) : null;
  const depth = new Float32Array(depthBuf);
  // ?bg=off: no layers behind; ?bg=1: only the first.
  const bgInfos = params.get("bg") === "off" || !info.background ? [] : [info.background].flat().slice(0, params.get("bg") === "1" ? 1 : undefined);
  const bgFilesList = await Promise.all(
    bgInfos.map((b) =>
      Promise.all([
        loadBitmap(b.image),
        loadBitmap((params.get("delight") !== "off" && b.albedo) || b.image),
        loadBitmap(b.normal),
        fetch(sceneUrl + b.depth).then((r) => r.arrayBuffer()),
      ]),
    ),
  );

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
  // Visuals vs audio: the graph's Sync node (lead, ms; positive = visuals earlier). Set once the graph loads.
  let visualLead = () => 0;
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
  const meshDepthTex = meshDepthBuf
    ? (() => {
        const d = device.createTexture({ size: [W, H], format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
        device.queue.writeTexture({ texture: d }, new Float32Array(meshDepthBuf), { bytesPerRow: W * 4 }, [W, H]);
        return d;
      })()
    : depthTex;
  // Layers behind: textures (same sizes and formats as the main layer's).
  const bgTexs = bgFilesList.map((bgFiles) => {
    const d = device.createTexture({ size: [W, H], format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    device.queue.writeTexture({ texture: d }, new Float32Array(bgFiles[3]), { bytesPerRow: W * 4 }, [W, H]);
    return { photo: imageTexture(bgFiles[0], "rgba8unorm-srgb"), albedo: imageTexture(bgFiles[1], "rgba8unorm-srgb"), normal: imageTexture(bgFiles[2], "rgba8unorm"), depth: d };
  });

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
      format: "rgba16float", // r: sun/spot, g b a: up to 3 nook lights
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
  const shadowA = shadowTexture();
  const shadowB = shadowTexture();
  // Nook lights 4-7's shadows: marched alongside (second target), blurred the same way.
  const shadowA2 = shadowTexture();
  const shadowB2 = shadowTexture();
  const fullscreen = (entryPoint: string, constants?: Record<string, number>) =>
    device.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vs_full" },
      fragment: { module, entryPoint, constants, targets: [{ format: "rgba16float" }] },
    });
  const shadowPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module, entryPoint: "vs_full" },
    fragment: { module, entryPoint: "fs_shadow", targets: [{ format: "rgba16float" }, { format: "rgba16float" }] },
  });
  const raysPipeline = fullscreen("fs_rays");
  const raysTex = shadowTexture();
  const blurH = fullscreen("fs_blur", { HORIZONTAL: 1 });
  const blurV = fullscreen("fs_blur", { HORIZONTAL: 0 });

  // Mesh density: at most ~1M quads (one per 2x2 texels for a 1472x2208 scene, one per ~3x3 at 4K).
  const step = Math.max(2, Math.ceil(Math.sqrt((W * H) / 1_000_000)));
  const grid = [Math.ceil(W / step), Math.ceil(H / step)];

  const UNIFORM_FLOATS = 76;
  // Lasers: header (count) + 4 fixtures x 24 floats + 96 beams x 4 floats. Layout matches `Lasers` in scene.wgsl.
  const LASER_FIXTURES = 16; // matches `Lasers` in scene.wgsl: scanning rigs take one fixture each
  const LASER_FLOATS = 4 + LASER_FIXTURES * 24 + LASER_FIXTURES * 24 * 2 * 4;
  const laserData = new Float32Array(LASER_FLOATS);
  const laserCount = new Uint32Array(laserData.buffer, 0, 1);
  const laserBuf = device.createBuffer({ size: LASER_FLOATS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  // Nook lights: count + 16 lights x 8 floats. Layout matches `Nooks` in scene.wgsl.
  const NOOK_FLOATS = 4 + 16 * 8;
  const nookData = new Float32Array(NOOK_FLOATS);
  const nookCount = new Uint32Array(nookData.buffer, 0, 1);
  const nookBuf = device.createBuffer({ size: NOOK_FLOATS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
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
      { binding: 9, resource: { buffer: nookBuf } },
      { binding: 10, resource: shadowA2.createView() },
      { binding: 11, resource: meshDepthTex.createView() },
      { binding: 12, resource: depthTex.createView() },
    ],
  });
  // The layers behind draw with the same pipeline, each with its own depth, colour and normals.
  const bgBindGroups = bgTexs.map((bgTex) =>
    device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuf } },
        { binding: 1, resource: device.createSampler({ magFilter: "linear", minFilter: "linear", mipmapFilter: "linear", maxAnisotropy: 8 }) },
        { binding: 2, resource: bgTex.albedo.createView() },
        { binding: 3, resource: bgTex.normal.createView() },
        { binding: 4, resource: bgTex.depth.createView() },
        { binding: 5, resource: bgTex.photo.createView() },
        { binding: 6, resource: shadowA.createView() },
        { binding: 7, resource: raysTex.createView() },
        { binding: 8, resource: { buffer: laserBuf } },
        { binding: 9, resource: { buffer: nookBuf } },
        { binding: 10, resource: shadowA2.createView() },
        { binding: 11, resource: bgTex.depth.createView() },
        { binding: 12, resource: depthTex.createView() },
      ],
    }),
  );
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
      { binding: 9, resource: { buffer: nookBuf } },
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
  const blurHBind2 = blurBind(blurH, shadowA2);
  const blurVBind2 = blurBind(blurV, shadowB2);

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

  // --- Real-world frame ------------------------------------------------------
  // The photo's camera looks down a little (about 23 degrees on the canyon), so its own "up" leans into the
  // scene. True vertical comes from the flat ground (scene.json "up"). Sun heights, rig aims, the sky
  // gradient and "flat ground" are all measured against it.
  const UP = normalize((info.up ?? [0, 1, 0]) as Vec3);
  const FWD = normalize([-UP[0] * UP[2], -UP[1] * UP[2], 1 - UP[2] * UP[2]] as Vec3); // into the scene, level
  const RIGHT = cross(UP, FWD);
  /** A direction from real-world turn (0 = into the scene, + right) and tilt (up from level), radians. */
  const worldDir = (turnRad: number, tiltRad: number): Vec3 => {
    const h = Math.cos(tiltRad);
    return [0, 1, 2].map((j) => RIGHT[j] * Math.sin(turnRad) * h + UP[j] * Math.sin(tiltRad) + FWD[j] * Math.cos(turnRad) * h) as Vec3;
  };
  /** Real-world turn and tilt (degrees) of a scene direction. */
  const worldAngles = (d: Vec3): [number, number] => {
    const n = normalize(d);
    const dot = (a: Vec3) => n[0] * a[0] + n[1] * a[1] + n[2] * a[2];
    return [(Math.atan2(dot(RIGHT), dot(FWD)) * 180) / Math.PI, (Math.asin(clamp(dot(UP), -1, 1)) * 180) / Math.PI];
  };
  // The sun along its arc (0 front horizon, 90 overhead, 180 behind), in the real-world frame.
  const sunDirection = (l: { sunAzimuth: number; sunElevation: number; sunArc?: number }): Vec3 =>
    worldDir((l.sunAzimuth * Math.PI) / 180, ((l.sunArc ?? l.sunElevation) * Math.PI) / 180);

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
  // For checking parallax, instead of the graph's camera: ?cam=x,y[,z] holds a fixed offset (in sway units),
  // ?cam=sweep[,size] sweeps slowly left-right and up-down (default size 2).
  const camArg = params.get("cam")?.split(",");
  const camSweep = camArg?.[0] === "sweep" ? Number(camArg[1] ?? 2) : 0;
  const camHold = camArg && !camSweep ? camArg.map(Number) : undefined;
  const camPos = (_t?: number): Vec3 => {
    const a = (cam.swayAmount * tanHalfFov * aspect) / parallaxSpan;
    if (camSweep) {
      const s = _t ?? performance.now() / 1000; // music time when given, so recorded clips sweep too
      return [camSweep * Math.sin(s * 0.5) * a, 0.4 * camSweep * Math.sin(s * 0.31) * a, 0];
    }
    if (camHold) return [camHold[0] * a, (camHold[1] ?? 0) * a, (camHold[2] ?? 0) * a];
    if (!cam.sway || !graphOut.camera) return [0, 0, 0];
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
  // The lowest open sky in the sun's direction, seen through the gap (degrees above the real horizon):
  // the Gap horizon node, so graphs can place the sun relative to the picture.
  const gapHorizon = (() => {
    const d = worldDir((look.sunAzimuth * Math.PI) / 180, 0);
    const u = d[2] > 0.05 ? 0.5 + d[0] / (d[2] * tanHalfFov * aspect) / 2 : 0.5;
    let lowest = -1;
    for (let v = 0; v <= 1; v += 0.002) {
      for (const du of [-0.01, 0, 0.01]) if (depthAt(u + du, v) >= info.far * 0.98) lowest = Math.max(lowest, v);
    }
    if (lowest < 0) return 0; // no sky above the sun: assume the real horizon
    return worldAngles(viewPos(u, lowest, 1))[1];
  })();

  // --- Node graph ------------------------------------------------------------------------------------
  // The graph drives the sun, camera and tone. The editor (editor.html) talks to this page over a
  // BroadcastChannel: it sends graph edits, this page sends back live node values.
  const graphRes = await fetch(`graphs/${graphName}.json`);
  if (!graphRes.ok) fail(`No graph graphs/${graphName}.json`);
  // Real-world distance of a picture point, for nodes that treat near and far differently.
  // Real scale: scene.json metersPerUnit when the scene knows it (e.g. from a distance markup); otherwise assume the far land is about 3 miles (15,840 ft) out.
  const sceneFeetPerUnit = info.metersPerUnit ? info.metersPerUnit * 3.28084 : 15840 / pct(0.95);
  const depthFeet = (u: number, v: number) => depthAt(u, v) * sceneFeetPerUnit;
  let diskText = await graphRes.text();
  const runtime = new GraphRuntime(JSON.parse(diskText) as Graph, { music, scene: { gapHorizon, depthFeet } });
  // Unsaved edits made here (monitor, placement, keys) or in the editor. While there are none, a change
  // to the graph file on disk (another window's Save, or an edit to the file) is picked up live; while
  // there are, the monitor warns instead of overwriting them.
  let graphDirty = false;
  const markEdited = () => (graphDirty = true);
  visualLead = () => Number(runtime.graph.nodes.find((n) => n.type === "Sync")?.params?.lead ?? 0) / 1000;
  const channel = new BroadcastChannel("doppler");
  channel.onmessage = (e: MessageEvent) => {
    const msg = e.data;
    if (msg?.type === "graph" && msg.from !== "display") runtime.load(msg.graph as Graph), monitor?.graphChanged(), markEdited();
    else if (msg?.type === "hello") channel.postMessage({ type: "graph", graph: runtime.graph, name: graphName, from: "display" });
  };
  const monitor = music
    ? createMonitor({
        music,
        runtime,
        time: () => time,
        seek: (t) => (audio.currentTime = Math.max(0, t - visualLead())),
        edited: (graph) => (markEdited(), channel.postMessage({ type: "graph", graph, name: graphName, from: "display", edit: true })),
        save: async () => {
          const body = JSON.stringify(runtime.graph, null, 1);
          const res = await fetch(`/graph?name=${encodeURIComponent(graphName)}`, { method: "POST", body });
          if (res.ok) (diskText = body), (graphDirty = false);
          return res.ok ? `saved graphs/${graphName}.json` : `save failed: ${res.status}`;
        },
      })
    : null;
  setInterval(async () => {
    const res = await fetch(`graphs/${graphName}.json`, { cache: "no-store" }).catch(() => null);
    if (!res?.ok) return;
    const text = await res.text();
    if (text === diskText) return;
    if (text === JSON.stringify(runtime.graph, null, 1)) return void ((diskText = text), (graphDirty = false)); // saved from the editor
    if (graphDirty) return monitor?.setStatus("the graph file changed on disk: Save keeps your edits here, reload (F5) takes the file's");
    diskText = text;
    runtime.load(JSON.parse(text) as Graph);
    monitor?.graphChanged();
    channel.postMessage({ type: "graph", graph: runtime.graph, name: graphName, from: "display", edit: true });
  }, 2000);
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
  let sunFloor = 1;
  const graphLook = (base: Look): Look => {
    const l = { ...base };
    const tone = graphOut.tone;
    if (tone) Object.assign(l, { baseDim: tone.baseDim, baked: tone.baked, cap: tone.cap });
    const sun = graphOut.sun;
    skyBoost = 0;
    sunFloor = sun?.floor ?? 1;
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
    const sectionGain = { quiet: 0.6, build: 0.8 + 0.4 * sec.progress, drop: 1.2, normal: 1 }[sec.mood];
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
  // Placement (N): put each module's fixtures on the rock by hand. Tab switches module.
  //   Nook lights: drag a ring to move it, scroll over it for its area, shift+scroll for its brightness.
  //   Laser rigs: drag the numbered ring to move the rig; rotate it like Maya's rotate tool: drag along the
  //   green ring (lying flat) to turn, along the red ring (upright, through the beam) to tilt; shift for
  //   fine. Scroll over the numbered ring for the search cone it re-aims within.
  //   Both: click empty rock to add, shift- or right-click to remove the nearest, Delete clears them all.
  // Edits go into the graph's nodes (and reach the editor); save with the monitor (G). Everything in the
  // module shows while placing.
  type Module = "nooks" | "rigs";
  const MODULES: Module[] = ["nooks", "rigs"];
  const MODULE_NAME: Record<Module, string> = { nooks: "NOOK LIGHTS", rigs: "LASER RIGS" };
  let placing: Module | null = null;
  let lastModule: Module = "nooks";
  const moduleNode = (m: Module) =>
    m === "nooks"
      ? runtime.graph.nodes.find((n) => n.type === "NookLights")
      : runtime.graph.nodes.find((n) => n.type === "SkyLaser" && Number(n.params?.from ?? 0) >= 0.5);
  const PARAM: Record<Module, string> = { nooks: "positions", rigs: "rigs" };
  // Points: nook lights [u, v, area, brightness]; rigs [u, v, turn, tilt, cone] (degrees).
  const readPoints = (m: Module): number[][] =>
    String(moduleNode(m)?.params?.[PARAM[m]] ?? "")
      .split(";")
      .map((q) => q.split(",").map(Number))
      .filter((q) => q.length >= 2 && q.every(Number.isFinite))
      .map((q) => (m === "nooks" ? [q[0], q[1], q[2] ?? 1, q[3] ?? 1] : [q[0], q[1], q[2] ?? 0, q[3] ?? 50, q[4] ?? 30]));
  const fmtPoint = (m: Module, q: number[]) =>
    m === "nooks"
      ? [q[0].toFixed(4), q[1].toFixed(4), ...(q[2] !== 1 || q[3] !== 1 ? [q[2].toFixed(2), q[3].toFixed(2)] : [])].join(",")
      : [q[0].toFixed(4), q[1].toFixed(4), ...q.slice(2, 5).map((x) => x.toFixed(1))].join(",");
  // final = false while dragging or scrolling: update the picture only; tell the editor and monitor after.
  const writePoints = (m: Module, pts: number[][], final = true) => {
    const node = moduleNode(m);
    if (!node) return;
    node.params = { ...node.params, [PARAM[m]]: pts.map((q) => fmtPoint(m, q)).join("; ") };
    runtime.load(runtime.graph);
    if (!final) return;
    markEdited();
    channel.postMessage({ type: "graph", graph: runtime.graph, name: graphName, from: "display", edit: true });
    monitor?.graphChanged();
  };
  const uvToScreen = (u: number, v: number) => {
    const c = camPos(time);
    return [
      (((u * 2 - 1 - cam.center[0] + c[0] / (pivotZ * tanHalfFov * aspect)) * viewScale[0] + 1) / 2) * innerWidth,
      ((1 - (1 - v * 2 - cam.center[1] + c[1] / (pivotZ * tanHalfFov)) * viewScale[1]) / 2) * innerHeight,
    ];
  };
  const project3 = (p: Vec3): [number, number] => [(p[0] / (p[2] * tanHalfFov * aspect)) * 0.5 + 0.5, 0.5 - (p[1] / (p[2] * tanHalfFov)) * 0.5];
  // Rigs aim by real-world angles: turn around true vertical (0 = into the scene, + right) and tilt up from
  // level ground (see worldDir), so tilt 90 is straight up into the sky.
  // A rig stands just in front of the rock at its spot.
  const rigOrigin = (u: number, v: number) => viewPos(u, v, depthAt(u, v) * 0.985);
  const rigAim = (q: number[]) => {
    const o = rigOrigin(q[0], q[1]);
    const az = q[2] * rad;
    const el = q[3] * rad;
    const d = worldDir(az, el);
    // How far out to draw the cone: a good way out, but not past the rock the beam would hit.
    return { o, d, len: Math.min(beamLength(o, d, info.far * 1.5), o[2] * 0.5) };
  };
  // Rotate rings, a fixed size on screen, centred on the rig: the turn ring lies flat (its angle is the
  // beam's turn), the tilt ring stands upright through the beam (its angle is the beam's tilt).
  const RING_PX = 70;
  type RingHandle = "turn" | "tilt";
  const ringPoints = (q: number[], which: RingHandle) => {
    const o = rigOrigin(q[0], q[1]);
    const R = (RING_PX * 2 * tanHalfFov * o[2]) / (innerHeight * viewScale[1]);
    const az = q[2] * rad;
    const out: { a: number; x: number; y: number }[] = [];
    for (let k = 0; k <= 96; k++) {
      const ang = -Math.PI + (2 * Math.PI * k) / 96;
      const dir = which === "turn" ? worldDir(ang, 0) : worldDir(az, ang);
      const pnt: Vec3 = [o[0] + dir[0] * R, o[1] + dir[1] * R, o[2] + dir[2] * R];
      if (pnt[2] > 0.01) {
        const [x, y] = uvToScreen(...project3(pnt));
        out.push({ a: ang, x, y });
      }
    }
    return { pts: out, R, o };
  };
  const nearestOnRing = (ring: { a: number; x: number; y: number }[], x: number, y: number) => {
    let best = { a: 0, d: Infinity };
    for (const r of ring) {
      const d = Math.hypot(r.x - x, r.y - y);
      if (d < best.d) best = { a: r.a, d };
    }
    return best;
  };
  const wrapAngle = (x: number) => Math.atan2(Math.sin(x), Math.cos(x));
  const coneBasis = (d: Vec3): [Vec3, Vec3] => {
    let e1 = normalize(cross([0, 1, 0], d));
    if (!Number.isFinite(e1[0])) e1 = [1, 0, 0];
    return [e1, cross(d, e1)];
  };

  // Ring drags. A ring that looks round enough is followed around: each move takes the nearest point on the
  // ring within 60 degrees of the last one (so it never jumps to the far side), and the angle travelled is
  // the rotation, all the way round. A ring seen nearly edge-on (front and back overlap on screen) works
  // like Maya's: only movement along its direction at the grab counts, RING_PX pixels = one radian.
  type Grab = {
    i: number;
    handle: "point" | RingHandle;
    a?: number;
    edgeOn?: boolean;
    sx?: number;
    sy?: number;
    tx?: number;
    ty?: number;
    v0?: number;
    turn0?: number; // turn at the grab, for tilting over the top
  };
  // Tilt past straight up (or down) carries on over the top, like a moving head: the turn swaps by 180
  // and the tilt comes back down the other side. Returns [turn, tilt] in degrees, and whether it flipped.
  const overTheTop = (turn: number, tilt: number): [number, number, boolean] => {
    const t = (wrapAngle(tilt * rad) * 180) / Math.PI;
    if (t > 90) return [(wrapAngle((turn + 180) * rad) * 180) / Math.PI, 180 - t, true];
    if (t < -90) return [(wrapAngle((turn + 180) * rad) * 180) / Math.PI, -180 - t, true];
    return [turn, t, false];
  };
  // How flat a ring looks: the short axis of its on-screen outline over the long one.
  const ringRoundness = (ring: { x: number; y: number }[]) => {
    const n = ring.length;
    const mx = ring.reduce((t, r) => t + r.x, 0) / n;
    const my = ring.reduce((t, r) => t + r.y, 0) / n;
    let sxx = 0;
    let syy = 0;
    let sxy = 0;
    for (const r of ring) (sxx += (r.x - mx) ** 2), (syy += (r.y - my) ** 2), (sxy += (r.x - mx) * (r.y - my));
    const tr = sxx + syy;
    const det = sxx * syy - sxy * sxy;
    const disc = Math.sqrt(Math.max(0, (tr * tr) / 4 - det));
    const big = tr / 2 + disc;
    const small = Math.max(0, tr / 2 - disc);
    return big > 0 ? Math.sqrt(small / big) : 0;
  };
  let grab: Grab | null = null;
  let hover: Grab | null = null;
  const hitTest = (m: Module, x: number, y: number, within = 18): Grab | null => {
    let best: Grab | null = null;
    let bd = within;
    readPoints(m).forEach((q, i) => {
      const handles: [Grab["handle"], number, number][] = [["point", ...(uvToScreen(q[0], q[1]) as [number, number])]];
      for (const [handle, sx, sy] of handles) {
        const d = Math.hypot(sx - x, sy - y);
        if (d < bd) (bd = d), (best = { i, handle });
      }
    });
    if (best || m !== "rigs") return best;
    // Rings: grab one within a few pixels of its line.
    readPoints(m).forEach((q, i) => {
      for (const which of ["turn", "tilt"] as const) {
        const near = nearestOnRing(ringPoints(q, which).pts, x, y);
        if (near.d < Math.min(bd, 9)) (bd = near.d), (best = { i, handle: which, a: near.a });
      }
    });
    return best;
  };
  let wheelTimer = 0;

  const markers = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  markers.style.cssText = "position:fixed;inset:0;width:100%;height:100%;pointer-events:none;font:600 12px system-ui,sans-serif";
  document.body.append(markers);
  const drawMarkers = () => {
    if (!placing) return void (markers.innerHTML = "");
    const m = placing;
    const pts = readPoints(m);
    const isOn = (i: number, h: Grab["handle"]) => (grab?.i === i && grab.handle === h) || (hover?.i === i && hover.handle === h);
    let svg = "";
    const label = (x: number, y: number, text: string, color: string) =>
      `<text x="${x}" y="${y}" fill="${color}" font-size="11" paint-order="stroke" stroke="#000" stroke-width="3">${text}</text>`;
    const ring = (x: number, y: number, n: number, color: string) =>
      `<circle cx="${x}" cy="${y}" r="9" fill="rgba(0,0,0,0.35)" stroke="${color}" stroke-width="2"/>` +
      `<text x="${x}" y="${y + 4}" fill="${color}" text-anchor="middle" paint-order="stroke" stroke="#000" stroke-width="3">${n}</text>`;
    if (m === "nooks") {
      const feet = Number(moduleNode(m)?.params?.radius ?? 800) || 800;
      const nearFeet = Number(moduleNode(m)?.params?.near ?? 8000);
      pts.forEach(([u, v, r, b], i) => {
        const [x, y] = uvToScreen(u, v);
        // Near lights (kept for big moments) in blue, mid and far ones (on the sounds) in amber.
        const color = isOn(i, "point") ? "#fff" : depthFeet(u, v) < nearFeet ? "#8fc8ff" : "#ffd27a";
        // The area of effect as it appears on screen: the pool's real radius at the light's distance.
        const px = (((feet / feetPerUnit) * r) / (depthAt(u, v) * tanHalfFov)) * viewScale[1] * (innerHeight / 2);
        svg += `<circle cx="${x}" cy="${y}" r="${px}" fill="none" stroke="${color}" stroke-dasharray="4 4" opacity="0.45"/>`;
        svg += ring(x, y, i + 1, color);
        if (r !== 1 || b !== 1) svg += label(x + 13, y + 4, `area ${Math.round(r * 100)}% · bright ${Math.round(b * 100)}%`, color);
      });
    } else {
      pts.forEach((q, i) => {
        const { o, d, len } = rigAim(q);
        const [e1, e2] = coneBasis(d);
        const half = (q[4] / 2) * rad;
        const color = isOn(i, "point") ? "#fff" : "#ff7a6b";
        const [x, y] = uvToScreen(q[0], q[1]);
        // The search cone: its rim at the aim distance, and lines out to it.
        const rim: number[][] = [];
        for (let k = 0; k <= 32; k++) {
          const ph = (2 * Math.PI * k) / 32;
          const dir = d.map((x0, j) => x0 * Math.cos(half) + (e1[j] * Math.cos(ph) + e2[j] * Math.sin(ph)) * Math.sin(half));
          const p: Vec3 = [o[0] + dir[0] * len, o[1] + dir[1] * len, o[2] + dir[2] * len];
          if (p[2] > 0.05) rim.push(uvToScreen(...project3(p)));
        }
        if (rim.length > 2) {
          svg += `<polyline points="${rim.map((p) => p.join(",")).join(" ")}" fill="rgba(255,122,107,0.08)" stroke="${color}" stroke-dasharray="4 4" opacity="0.6"/>`;
          for (const p of [rim[0], rim[Math.floor(rim.length / 2)]]) {
            svg += `<line x1="${x}" y1="${y}" x2="${p[0]}" y2="${p[1]}" stroke="${color}" stroke-dasharray="4 4" opacity="0.5"/>`;
          }
        }
        // Rotate rings: green lies flat (turn), red stands upright through the beam (tilt).
        for (const [which, ringColor] of [["turn", "#5fd35f"], ["tilt", "#ff5f5f"]] as const) {
          const { pts: rp } = ringPoints(q, which);
          const hot = isOn(i, which);
          if (rp.length > 2) {
            svg += `<polyline points="${rp.map((r) => r.x + "," + r.y).join(" ")}" fill="none" stroke="${hot ? "#fff" : ringColor}" stroke-width="${hot ? 4 : 2.5}" opacity="0.9"/>`;
          }
        }
        // The beam's direction, out past the rings, with a dot where it meets them.
        const { R } = ringPoints(q, "turn");
        const tip: Vec3 = [o[0] + d[0] * R * 1.5, o[1] + d[1] * R * 1.5, o[2] + d[2] * R * 1.5];
        const mid: Vec3 = [o[0] + d[0] * R, o[1] + d[1] * R, o[2] + d[2] * R];
        if (tip[2] > 0.01 && mid[2] > 0.01) {
          const [tx, ty] = uvToScreen(...project3(tip));
          const [mx, my] = uvToScreen(...project3(mid));
          svg += `<line x1="${x}" y1="${y}" x2="${tx}" y2="${ty}" stroke="#ffd27a" stroke-width="2.5"/>`;
          svg += `<circle cx="${mx}" cy="${my}" r="4" fill="#ffd27a"/>`;
        }
        svg += ring(x, y, i + 1, color);
        svg += label(x + 13, y + 18, `turn ${Math.round(q[2])}° · tilt ${Math.round(q[3])}° · cone ${Math.round(q[4])}°`, color);
      });
    }
    const how =
      m === "nooks"
        ? "drag a ring to move · scroll: area · shift+scroll: brightness"
        : "drag the number to move · drag the GREEN ring to turn, the RED ring to tilt (shift: fine) · scroll over the number: search cone";
    svg +=
      `<text x="50%" y="28" text-anchor="middle" fill="#fff" font-size="14" paint-order="stroke" stroke="#000" stroke-width="4">` +
      `PLACING ${MODULE_NAME[m]} (Tab: switch) · ${how} · click: add · shift/right-click: remove · Delete: clear · N: done · save in G</text>`;
    markers.innerHTML = svg;
  };
  const placeAt = (e: PointerEvent) => {
    const m = placing!;
    if (!moduleNode(m)) return;
    const pts = readPoints(m);
    if (!e.shiftKey && e.button === 0) {
      const hit = hitTest(m, e.clientX, e.clientY);
      if (hit) {
        grab = hit;
        if (hit.handle === "turn" || hit.handle === "tilt") {
          const q = pts[hit.i];
          const ring = ringPoints(q, hit.handle).pts;
          const k = ring.findIndex((r) => r.a === hit.a);
          const prev = ring[Math.max(0, k - 2)];
          const next = ring[Math.min(ring.length - 1, k + 2)];
          let tx = next.x - prev.x;
          let ty = next.y - prev.y;
          const l = Math.hypot(tx, ty);
          if (l < 1e-3) (tx = 1), (ty = 0);
          else (tx /= l), (ty /= l);
          Object.assign(grab, { sx: e.clientX, sy: e.clientY, tx, ty, v0: hit.handle === "turn" ? q[2] : q[3], turn0: q[2], edgeOn: ringRoundness(ring) < 0.3 });
        }
        canvas.setPointerCapture(e.pointerId);
        return;
      }
    }
    const [u, v] = toImageUv(e);
    if (e.shiftKey || e.button === 2) {
      const hit = hitTest(m, e.clientX, e.clientY, 60);
      if (hit) pts.splice(hit.i, 1);
    } else pts.push(m === "nooks" ? [u, v, 1, 1] : [u, v, 0, 50, 30]);
    writePoints(m, pts);
  };

  let dragging: "target" | "source" | null = null;
  let last = [0, 0];
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("pointerdown", (e) => {
    if (placing) return placeAt(e);
    canvas.setPointerCapture(e.pointerId);
    dragging = e.button === 2 ? "source" : "target";
    last = [e.clientX, e.clientY];
    if (dragging === "target") aimAt(e);
  });
  canvas.addEventListener("pointerup", () => {
    dragging = null;
    if (grab && placing) (grab = null), writePoints(placing, readPoints(placing));
  });
  canvas.addEventListener("pointermove", (e) => {
    if (placing) hover = hitTest(placing, e.clientX, e.clientY);
    if (grab && placing) {
      const pts = readPoints(placing);
      const [u, v] = toImageUv(e);
      const q = pts[grab.i];
      if (grab.handle === "turn" || grab.handle === "tilt") {
        let deg: number;
        if (grab.edgeOn) {
          // Edge-on: movement along the ring's direction at the grab.
          const along = (e.clientX - grab.sx!) * grab.tx! + (e.clientY - grab.sy!) * grab.ty!;
          deg = ((along / RING_PX) * 180) / Math.PI;
        } else {
          // Round enough: follow the cursor around the ring, step by step.
          const ring = ringPoints(q, grab.handle).pts.filter((r) => Math.abs(wrapAngle(r.a - grab!.a!)) < Math.PI / 3);
          const near = nearestOnRing(ring, e.clientX, e.clientY);
          const step = Number.isFinite(near.d) ? wrapAngle(near.a - grab.a!) : 0;
          if (Number.isFinite(near.d)) grab.a = near.a;
          deg = (step * 180) / Math.PI;
          grab.v0 = grab.handle === "turn" ? q[2] : q[3]; // step from where it is now
        }
        deg *= e.shiftKey ? 0.25 : 1;
        if (grab.handle === "turn") q[2] = (wrapAngle((grab.v0 + deg) * rad) * 180) / Math.PI;
        else {
          const [turn, tilt, flipped] = overTheTop(grab.edgeOn ? grab.turn0! : q[2], grab.v0 + deg);
          q[2] = turn;
          q[3] = tilt;
          // Over the top, the ring's angles run the other way round: keep following the same point on it.
          if (flipped && !grab.edgeOn) grab.a = wrapAngle(Math.PI - grab.a!);
        }
      } else (q[0] = u), (q[1] = v);
      return writePoints(placing, pts, false);
    }
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
      const [turn, tilt] = worldAngles(viewPos(u, v, 1));
      look.sunAzimuth = turn;
      look.sunElevation = clamp(tilt, 1, 89);
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
      // Shift+wheel arrives as sideways scrolling in some browsers (Chrome on Windows).
      const k = Math.exp(-(e.deltaY || e.deltaX) * 0.001);
      // Placing: nook lights' area (shift: brightness), or a rig's search cone.
      if (placing) {
        const m = placing;
        const hit = hitTest(m, e.clientX, e.clientY, 40);
        if (!hit) return;
        const pts = readPoints(m);
        const q = pts[hit.i];
        if (m === "nooks") q[e.shiftKey ? 3 : 2] = clamp(q[e.shiftKey ? 3 : 2] * k, 0.1, 10);
        else q[4] = clamp(q[4] * k, 2, 120);
        writePoints(m, pts, false);
        clearTimeout(wheelTimer);
        wheelTimer = window.setTimeout(() => writePoints(m, readPoints(m)), 300);
        return;
      }
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
    else if (key === "g") monitor?.toggle();
    else if (key === "n") placing = placing ? null : lastModule;
    else if (placing && e.key === "Tab") {
      e.preventDefault();
      placing = lastModule = MODULES[(MODULES.indexOf(placing) + 1) % MODULES.length];
    } else if (placing && (e.key === "Delete" || e.key === "Backspace")) writePoints(placing, []);
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
      // Adjusts the Sync node (made if the graph has none), so the offset is saved with the graph.
      let node = runtime.graph.nodes.find((n) => n.type === "Sync");
      if (!node) runtime.graph.nodes.push((node = { id: "sync", type: "Sync", name: "Sync (visual lead)", params: { lead: 0 }, pos: [40, 40] }));
      node.params = { ...node.params, lead: Math.round(Number(node.params?.lead ?? 0) + (key === "}" ? 1 : -1) * (e.altKey ? 1 : 10)) };
      runtime.load(runtime.graph);
      markEdited();
    channel.postMessage({ type: "graph", graph: runtime.graph, name: graphName, from: "display", edit: true });
      monitor?.graphChanged();
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
      ...sunScreen(look), look.sun ? skyBoost : 0, sunFloor,
      ...scanUniforms(graphOut.scan),
      ...skyUniforms(),
      look.sun ? graphOut.sun?.bounce ?? 0 : 0, graphOut.sun?.shadowSoftness ?? 0.1, graphOut.sun?.shadowDepth ?? 0.6, graphOut.sun?.caveDepth ?? 15,
      ...UP, bgBindGroups.length ? 1 : 0,
      graphOut.sun?.terminator ?? 0, 0, 0, 0,
    ]);
    uniforms[39] = look.sun ? look.flare * flareVisible : 0; // after sunScreen() measured visibility
    uniforms[55] = graphOut.sun?.detailBump ?? 0; // scanLines.w: fine rock relief from the photo's texture
    device.queue.writeBuffer(uniformBuf, 0, uniforms);
    writeLasers(graphOut.lasers ?? [], graphOut.skyLasers ?? []);
    writeNooks(graphOut.nooks);

    const enc = device.createCommandEncoder();
    // march -> A (and A2), blur A -> B (horizontal), blur B -> A (vertical), same for A2; the main pass reads A and A2.
    const march = enc.beginRenderPass({
      colorAttachments: [shadowA, shadowA2].map((t) => ({ view: t.createView(), loadOp: "clear" as const, storeOp: "store" as const, clearValue: [1, 1, 1, 1] })),
    });
    march.setPipeline(shadowPipeline);
    march.setBindGroup(0, shadowBind);
    march.draw(3);
    march.end();
    for (const [pipe, bind, target] of [
      [blurH, blurHBind, shadowB],
      [blurV, blurVBind, shadowA],
      [blurH, blurHBind2, shadowB2],
      [blurV, blurVBind2, shadowA2],
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
    // Behind it, the layers behind (instance 1, 2...): each only shows where the layers in front moved off it.
    bgBindGroups.forEach((g, k) => {
      pass.setBindGroup(0, g);
      pass.draw(grid[0] * grid[1] * 6, 1, 0, k + 1);
    });
    pass.end();
    device.queue.submit([enc.finish()]);
  };

  // Sky gradient: follows the sun's height above the lowest open sky (the gap's horizon), even while the
  // sun itself is switched off, so the sky keeps its colour.
  const skyUniforms = (): number[] => {
    const S = graphOut.sky;
    if (!S) return [0, 0, 0, 0, 0, 0, 1, (graphOut.sun?.floorBelow ?? -1500) / feetPerUnit];
    const arc = graphOut.sun?.arc ?? look.sunElevation;
    const elev = arc <= 90 ? arc : 180 - arc;
    return [S.mix, S.brightness, S.clouds, S.glow, elev - gapHorizon, (gapHorizon * Math.PI) / 180, (S.span * Math.PI) / 180, (graphOut.sun?.floorBelow ?? -1500) / feetPerUnit];
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
  // Real-world scale: the far land is about 3 miles (15,840 ft) out.
  const feetPerUnit = sceneFeetPerUnit;
  const scanUniforms = (s: RenderOut["scan"]): number[] => {
    if (!s) return new Array(12).fill(0);
    const span = landRange.hi[s.axis] - landRange.lo[s.axis];
    let front: number;
    let spacing: number;
    if (s.axis === 0) {
      // Depth sweeps evenly in log distance, but the shader works in plain distance.
      front = Math.exp(landRange.lo[0] + span * s.position);
      spacing = front * (1 - Math.exp(-s.spacing * span));
    } else {
      front = landRange.lo[s.axis] + span * s.position;
      spacing = s.spacing * span;
    }
    return [s.axis, front, s.trail / feetPerUnit, s.thickness / feetPerUnit, ...(s.color as Vec3), s.intensity, s.lines, spacing, s.reach, 0];
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
      // Behind a surface counts as inside rock only for so deep: thick for the cave walls around us, thin for
      // far ridges (like the sun's shadows), so beams aimed up clear distant rims and reach the sky.
      const f = Math.min(1, Math.max(0, (z - caveDepth) / (2 * caveDepth)));
      const solid = 2 - 1.75 * f * f * (3 - 2 * f);
      if (z < info.far * 0.98 && q[2] > z * 1.01 && q[2] < z * (1 + solid)) {
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
  // Ground rigs: fixed spots on the rock (seeded by the node's seed only, so they never move), each
  // firing a beam in a seeded random direction that changes per trigger step and drifts in between.
  // Aims that run into the rock right away are re-rolled. A negative sheet field marks the beam's
  // start as a visible rig for the shader.
  // A hand-placed rig's beam: a random direction inside its cone (even over its area), new each step, swaying a
  // little while lit. While placing rigs it points straight down the cone's axis.
  const rigBeam = (S: NonNullable<RenderOut["skyLasers"]>[number], q: number[], bi: number, step: number) => {
    const { o, d: aim } = rigAim(q);
    if (placing === "rigs") return { o, d: aim };
    const r = seeded(S.seed * 104729 + step * 131 + bi * 17);
    const phase = 2 * Math.PI * (S.driftSpeed * S.t) + bi * 1.7;
    const half = (q[4] / 2) * rad;
    const th = clamp(half * Math.sqrt(r()) + S.drift * rad * 0.5 * Math.sin(phase), 0, half);
    const ph = 2 * Math.PI * r() + 0.3 * Math.sin(phase * 0.7 + 1.1);
    const [e1, e2] = coneBasis(aim);
    return { o, d: normalize(aim.map((x, j) => x * Math.cos(th) + (e1[j] * Math.cos(ph) + e2[j] * Math.sin(ph)) * Math.sin(th)) as Vec3) };
  };
  // Scanning rigs: each beam becomes a scanner sweeping fast across a fan, read by the eye as a triangular plane of
  // light. One fixture per rig (its own plane): scanLines beams across the fan give the plane its length at each
  // angle (cut where it meets rock); only the two edge ones are drawn as lines. The plane's tilt around the beam is
  // seeded per step. Flicker: each frame the plane dims a little at random.
  const scanRigs = (S: NonNullable<RenderOut["skyLasers"]>[number], first: number, step: number, brightness: number, maxLen: number) => {
    let li = first;
    for (const [bi, q] of (S.rigs ?? []).entries()) {
      if (li >= LASER_FIXTURES) break;
      const beam = rigBeam(S, q, bi, step);
      const o = beam.o;
      let d = beam.d;
      const r = seeded(S.seed * 7907 + step * 53 + bi * 29);
      const roll = Math.PI * r();
      let right = normalize(cross([0, 1, 0], d));
      if (!Number.isFinite(right[0])) right = [1, 0, 0];
      const up = cross(d, right);
      right = normalize(right.map((x, j) => x * Math.cos(roll) + up[j] * Math.sin(roll)) as Vec3);
      // The plane swings across itself (around its sideways axis), each rig out of step: wider and faster on peaks.
      const swing = S.sweep * S.sweepBoost * rad * Math.sin(2 * Math.PI * S.sweepPhase + bi * 1.9);
      const across = normalize(cross(d, right));
      d = normalize(d.map((x, j) => x * Math.cos(swing) + across[j] * Math.sin(swing)) as Vec3);
      const normal = normalize(cross(d, right));
      const half = (S.scanSpread / 2) * rad;
      const n = S.scanLines;
      const flick = 1 - S.flicker * Math.random();
      // sheet > 1 marks a scanned plane for the shader: 1 + the plane's brightness.
      laserData.set([...o, n, ...d, 1 + S.scanBright, ...right, half, ...normal, brightness * flick, ...(S.color as Vec3), S.width * rad * pivotZ, S.hit, maxLen, S.glow, S.reach], 4 + li * 24);
      for (let k = 0; k < n; k++) {
        const a = -half + (2 * half * k) / (n - 1);
        const dk = normalize(d.map((x, j) => x * Math.cos(a) + right[j] * Math.sin(a)) as Vec3);
        // The middle line also marks the rig itself (a small bright source).
        laserData.set([...o, beamLength(o, dk, maxLen), ...dk, k === n >> 1 ? -1 : 0], beamSlot(li, k));
      }
      li++;
    }
    return li;
  };
  const groundRigs = (S: NonNullable<RenderOut["skyLasers"]>[number], li: number, step: number, maxLen: number) => {
    if (S.rigs?.length) {
      S.rigs.slice(0, 24).forEach((q, bi) => {
        const { o, d } = rigBeam(S, q, bi, step);
        laserData.set([...o, beamLength(o, d, maxLen), ...d, -1], beamSlot(li, bi));
      });
      return;
    }
    for (let bi = 0; bi < S.count; bi++) {
      const place = seeded(S.seed * 7919 + bi * 31);
      let u = 0.5;
      let v = 0.8;
      // Ground you could stand a rig on: rock that recedes going up the picture (a floor or ledge top),
      // not a cliff face turned toward the camera.
      for (let tries = 0; tries < 120; tries++) {
        u = S.uMin + (S.uMax - S.uMin) * place();
        v = S.vMin + (S.vMax - S.vMin) * place();
        const z = depthAt(u, v);
        if (z >= info.far * 0.9 || z < S.minDepth) continue;
        const above = depthAt(u, v - 0.01);
        const below = depthAt(u, v + 0.01);
        if (above > z * 1.04 && below < z * 0.99 && above < info.far * 0.9) break;
      }
      const o = viewPos(u, v, depthAt(u, v) * 0.985); // just in front of the rock
      const aimRand = seeded(S.seed * 104729 + step * 131 + bi * 17);
      const phase = 2 * Math.PI * (S.driftSpeed * S.t) + bi * 1.7;
      let d: Vec3 = [0, 1, 0];
      let len = 0;
      for (let tries = 0; tries < 8; tries++) {
        const az = 2 * Math.PI * aimRand() + S.drift * rad * Math.sin(phase);
        const el = (S.elevMin + (S.elevMax - S.elevMin) * aimRand()) * rad + S.drift * rad * 0.5 * Math.sin(phase * 0.7 + 1.1);
        d = normalize([Math.sin(az) * Math.cos(el), Math.sin(el), Math.cos(az) * Math.cos(el)]);
        if (d[2] < -0.2 && tries < 7) continue; // don't fire back at the camera: up close it's a smear
        len = beamLength(o, d, maxLen);
        if (len > o[2] * 0.4) break; // clear of the rock it stands on
      }
      laserData.set([...o, len, ...d, -1], beamSlot(li, bi));
    }
  };
  const beamSlot = (li: number, bi: number) => 4 + LASER_FIXTURES * 24 + (li * 24 + bi) * 8;
  const writeLasers = (list: NonNullable<RenderOut["lasers"]>, sky: NonNullable<RenderOut["skyLasers"]>) => {
    laserData.fill(0);
    const n = Math.min(LASER_FIXTURES, list.length);
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
      // Width: degrees as seen from the middle distance of the scene, turned into a physical thickness.
      laserData.set([...o, L.count, ...aim, L.sheet, ...right, half, ...normal, L.intensity, ...(L.color as Vec3), L.width * rad * pivotZ, L.hit, maxLen, L.glow, L.reach], base);
      for (let bi = 0; bi < L.count; bi++) {
        const a = L.count > 1 ? -half + (2 * half * bi) / (L.count - 1) : 0;
        const d = normalize(aim.map((x, i) => x * Math.cos(a) + right[i] * Math.sin(a)) as Vec3);
        laserData.set([...o, beamLength(o, d, maxLen), ...d, 0], beamSlot(li, bi));
      }
    }
    // Sky lasers: beams from high above onto seeded random spots on the rock, new spots per trigger step.
    let used = n;
    for (const S of sky) {
      const showRigs = placing === "rigs" && S.from === 1;
      if (S.intensity <= 0 && !showRigs) continue; // switched off (its setup is out)
      if (used >= LASER_FIXTURES) break;
      const step = Math.floor(S.trigger);
      const brightness = showRigs ? Math.max(1, S.intensity) : S.intensity * Math.exp(-(S.trigger - step) * S.fade);
      if (S.from === 1 && S.rigs?.length && S.scan > 0.5 && !showRigs) {
        used = scanRigs(S, used, step, brightness, maxLen);
        continue;
      }
      const li = used++;
      const height = info.far * 0.6;
      const count = S.from === 1 && S.rigs?.length ? Math.min(24, S.rigs.length) : S.count;
      laserData.set([0, 0, 0, count, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1, brightness, ...(S.color as Vec3), S.width * rad * pivotZ, S.hit, maxLen, S.glow, S.reach], 4 + li * 24);
      if (S.from === 1) {
        groundRigs(S, li, step, maxLen);
        continue;
      }
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
        // Sway while lit: the sky end swings, the landing spot stays put. Each beam out of step.
        const phase = 2 * Math.PI * (S.driftSpeed * S.t) + bi * 1.7;
        const lean = S.tilt * rad * rand() + S.drift * rad * Math.sin(phase);
        const turn = 2 * Math.PI * rand() + 0.6 * Math.sin(phase * 0.7 + 1.1);
        const o: Vec3 = [target[0] + Math.sin(lean) * Math.cos(turn) * height, target[1] + Math.cos(lean) * height, target[2] + Math.sin(lean) * Math.sin(turn) * height];
        const len = Math.hypot(target[0] - o[0], target[1] - o[1], target[2] - o[2]);
        const d = normalize([target[0] - o[0], target[1] - o[1], target[2] - o[2]]);
        laserData.set([...o, len, ...d, S.sheet > 0.5 ? (S.sheetWidth / 2) * rad : 0], beamSlot(li, bi));
      }
    }
    laserCount[0] = used;
    device.queue.writeBuffer(laserBuf, 0, laserData);
  };

  // Nook lights: each sits a little in front of the rock at its picture position (toward the camera, so
  // it's in the nook's open air). Pools have a real size (feet), so far ones look smaller. Up to 16 render,
  // brightest first; the first 7 get shadows.
  // Placement mode (N) shows them all.
  const writeNooks = (N: RenderOut["nooks"]) => {
    nookData.fill(0);
    const all = (N?.lights ?? []).map((l) => (placing === "nooks" ? { ...l, level: 0.6 } : l));
    const lit = all.filter((l) => l.level > 0.002).sort((a, b) => b.level * b.b - a.level * a.b).slice(0, 16);
    lit.forEach((l, i) => {
      const z = depthAt(l.u, l.v);
      const radius = (N!.radius / feetPerUnit) * l.r;
      const pos = viewPos(l.u, l.v, Math.max(z * 0.05, z - N!.standoff * radius));
      const k = N!.intensity * l.level * l.b;
      nookData.set([...pos, radius, ...(N!.color.map((c) => c * k) as Vec3), 0], 4 + i * 8);
    });
    nookCount[0] = lit.length;
    device.queue.writeBuffer(nookBuf, 0, nookData);
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
    time = music ? audio.currentTime + visualLead() : now / 1000;
    draw();
    sendValues(now);
    monitor?.update(time);
    drawMarkers();
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
`${sec.kind} ${(sec.progress * 100).toFixed(0)}%  phrase bar ${music.phrase(time).bar.toFixed(1)}  tension ${music.tension(time).toFixed(2)}  visual lead ${(visualLead() * 1000).toFixed(0)} ms  auto light ${autoLight ? "on" : "off"}\n` +
      `space play · J L seek · { } offset · A auto light · G music monitor · N place nook lights\n` +
      (placing
        ? `PLACING ${MODULE_NAME[placing]} (Tab: switch) · ` +
          (placing === "nooks" ? `drag a ring to move · scroll: area · shift+scroll: brightness` : `drag the ring to move · drag the diamond to aim · scroll: search cone`) +
          ` · click to add · shift/right-click to remove · Delete clears all · N when done · Save in the monitor (G)\n`
        : "")
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
