// The scene as a depth-displaced mesh seen by a movable camera (pan, zoom, parallax),
// lit by a dim base and one spotlight from the depth and normal maps.
// Scene space: the photo's own camera at the origin, x right, y up, z = distance into the scene.
// Units match scene.json near/far. Lighting always happens in scene space, so moving the
// viewing camera never changes where light lands on the rock.

struct Uniforms {
  screen: vec2f,
  imgSize: vec2f,
  viewScale: vec2f,    // image ndc -> screen ndc, after subtracting center (fit, aspect and zoom)
  center: vec2f,       // image ndc at the middle of the screen (pan)
  tanHalfFov: f32,
  aspect: f32,         // image width / height
  far: f32,
  baseDim: f32,
  lightPos: vec3f,
  intensity: f32,
  lightTarget: vec3f,
  coneCos: f32,        // cosine of the cone's outer half-angle
  lightColor: vec3f,
  coneSoft: f32,       // 0 = hard edge, 1 = fades all the way to the centre
  cap: f32,            // soft ceiling on total brightness
  shadows: f32,
  view: f32,           // 0 final, 1 albedo, 2 depth, 3 normals, 4 light only, 5 shadow mask, 6 photo, 7 sun shafts
  skyCut: f32,         // fraction of far beyond which a pixel counts as sky
  camPos: vec3f,       // viewing camera offset from the photo's camera (parallax)
  pivotZ: f32,         // depth that stays put while the camera moves
  grid: vec2f,         // mesh quads across, down
  hazeBeta: f32,       // atmospheric extinction per scene unit: light on far rock is veiled by the air
  baked: f32,          // how much of the original photo's own lighting shows in the base (0 = none)
  sun: f32,            // 1: directional sun, lightPos holds the direction toward it. 0: spotlight
  rays: f32,           // strength of visible sun shafts in the air
  caveDepth: f32,      // sun shadows: rock nearer than this is the cave around the camera
  flare: f32,          // lens flare strength (already scaled by how much of the sun is visible)
  sunScreen: vec2f,    // the sun's position on screen, ndc
};

// Lens flare, in screen space: a glow and starburst at the sun, and coloured ghosts along the line
// from the sun through the screen centre, the way reflections inside a camera lens line up.
fn lensFlare(frag: vec2f) -> vec3f {
  let p = vec2f(frag.x / u.screen.x * 2.0 - 1.0, 1.0 - frag.y / u.screen.y * 2.0);
  let a = vec2f(u.screen.x / u.screen.y, 1.0);
  let d = (p - u.sunScreen) * a;
  let r = length(d);
  let ang = atan2(d.y, d.x);
  var c = vec3f(1.0, 0.85, 0.6) * (exp(-r * r * 60.0) * 1.2 + exp(-r * 3.5) * 0.12);
  // Starburst: thin streaks, fading out from the sun.
  let streak = pow(abs(cos(ang * 3.0)), 60.0) * 0.5 + pow(abs(cos(ang * 4.0 + 0.6)), 90.0) * 0.3;
  c += vec3f(1.0, 0.9, 0.75) * streak * exp(-r * 2.5) * 0.5;
  // Halo ring around the sun.
  c += vec3f(0.6, 0.8, 1.0) * exp(-pow((r - 0.42) * 22.0, 2.0)) * 0.05;
  // Ghosts: positions along the sun-centre axis, radius, colour.
  let ghosts = array(
    vec4f(0.45, 0.05, 1.0, 0.7), vec4f(-0.25, 0.08, 0.6, 0.9), vec4f(-0.55, 0.035, 0.9, 1.0),
    vec4f(-0.9, 0.14, 0.5, 0.6), vec4f(-1.3, 0.06, 1.0, 0.5));
  let tints = array(vec3f(1.0, 0.6, 0.3), vec3f(0.4, 0.8, 1.0), vec3f(0.9, 0.5, 1.0), vec3f(0.5, 1.0, 0.6), vec3f(1.0, 0.8, 0.4));
  for (var i = 0; i < 5; i++) {
    let g = ghosts[i];
    let gd = length((p - u.sunScreen * g.x) * a);
    let disc = smoothstep(g.y, g.y * 0.6, gd) * g.z * 0.06;
    c += tints[i] * disc;
  }
  return c * u.flare;
}

// The light at a surface point: direction toward it, distance, and how much of it arrives (cone x falloff).
struct LightHit {
  l: vec3f,
  dist: f32,
  amount: f32,
};

fn lightAt(p: vec3f) -> LightHit {
  if (u.sun > 0.5) {
    // Parallel rays, no falloff. March far enough to cross the whole scene.
    return LightHit(normalize(u.lightPos), u.far * 1.5, 1.0);
  }
  let toL = u.lightPos - p;
  let dist = length(toL);
  let l = toL / dist;
  let cone = smoothstep(u.coneCos, mix(u.coneCos, 1.0, max(u.coneSoft, 0.001)), dot(-l, normalize(u.lightTarget - u.lightPos)));
  // Intensity is calibrated at the target: falloff relative to the target's distance.
  let dTarget = distance(u.lightTarget, u.lightPos);
  return LightHit(l, dist, cone * (dTarget * dTarget) / (dist * dist));
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var albedoTex: texture_2d<f32>;
@group(0) @binding(3) var normalTex: texture_2d<f32>;
@group(0) @binding(4) var depthTex: texture_2d<f32>;   // r32float, not filterable: bilinear by hand
@group(0) @binding(5) var photoTex: texture_2d<f32>;   // original image, baked lighting and all: the dim base
@group(0) @binding(6) var shadowTex: texture_2d<f32>;  // half-res light visibility, in image space (see fs_shadow)
@group(0) @binding(7) var raysTex: texture_2d<f32>;    // half-res sun shafts, in image space (see fs_rays)


fn depthAt(uv: vec2f) -> f32 {
  let size = vec2i(u.imgSize);
  let t = uv * u.imgSize - 0.5;
  let i = vec2i(floor(t));
  let f = fract(t);
  let a = textureLoad(depthTex, clamp(i, vec2i(0), size - 1), 0).r;
  let b = textureLoad(depthTex, clamp(i + vec2i(1, 0), vec2i(0), size - 1), 0).r;
  let c = textureLoad(depthTex, clamp(i + vec2i(0, 1), vec2i(0), size - 1), 0).r;
  let d = textureLoad(depthTex, clamp(i + vec2i(1, 1), vec2i(0), size - 1), 0).r;
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// Unblended depth: bilinear blending across silhouettes invents geometry that casts thin false shadows.
fn depthNearest(uv: vec2f) -> f32 {
  let i = clamp(vec2i(uv * u.imgSize), vec2i(0), vec2i(u.imgSize) - 1);
  return textureLoad(depthTex, i, 0).r;
}

fn viewPos(uv: vec2f, z: f32) -> vec3f {
  let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  return vec3f(ndc.x * u.tanHalfFov * u.aspect * z, ndc.y * u.tanHalfFov * z, z);
}

fn project(p: vec3f) -> vec2f {
  let ndc = p.xy / (p.z * vec2f(u.tanHalfFov * u.aspect, u.tanHalfFov));
  return vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

struct VsOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
};

// One vertex per mesh corner, generated from the index: no vertex buffers.
@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
  let cols = u32(u.grid.x);
  let quad = i / 6u;
  let corner = array(vec2u(0, 0), vec2u(1, 0), vec2u(0, 1), vec2u(0, 1), vec2u(1, 0), vec2u(1, 1))[i % 6u];
  let g0 = vec2u(quad % cols, quad / cols);
  let g = g0 + corner;
  let uv = vec2f(g) / u.grid;

  // A square whose corners sit at very different depths spans a silhouette (near rock against far
  // canyon), not a surface: stretched by parallax it smears into hairs. Put the whole square at the
  // far side's depth so the near edge stays crisp and nothing stretches across the gap.
  let d00 = depthAt(vec2f(g0) / u.grid);
  let d10 = depthAt(vec2f(g0 + vec2u(1, 0)) / u.grid);
  let d01 = depthAt(vec2f(g0 + vec2u(0, 1)) / u.grid);
  let d11 = depthAt(vec2f(g0 + vec2u(1, 1)) / u.grid);
  let zmin = min(min(d00, d10), min(d01, d11));
  let zmax = max(max(d00, d10), max(d01, d11));
  let z = select(depthAt(uv), zmax, zmax > zmin * 1.08);

  let pc = viewPos(uv, z) - u.camPos;
  let tanXY = vec2f(u.tanHalfFov * u.aspect, u.tanHalfFov);
  // Where this point lands in the photo's frame when seen from the moved camera,
  // shifted so the pivot depth holds still: near rock and far canyon slide in opposite directions.
  let q = pc.xy / (pc.z * tanXY) + u.camPos.xy / (u.pivotZ * tanXY);
  var out: VsOut;
  out.pos = vec4f((q - u.center) * u.viewScale, clamp(pc.z / (u.far * 2.0), 0.0, 1.0), 1.0);
  out.uv = uv;
  return out;
}

// Shadows, in three passes over a half-resolution image-space texture:
//   fs_shadow   march from each surface point toward the light through the depth map
//   fs_blur     depth-aware blur, horizontal then vertical: turns the march's jitter into soft shadow
//               without bleeding across silhouettes
//   fs          samples the result
// Image space, not screen space, so shadows stay put on the rock while the camera moves.

@vertex
fn vs_full(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

fn shadowSize() -> vec2f {
  return ceil(u.imgSize * 0.5);
}

@fragment
fn fs_shadow(@builtin(position) frag: vec4f) -> @location(0) vec4f {
  let uv = frag.xy / shadowSize();
  let z = depthNearest(uv);
  if (u.shadows < 0.5 || z >= u.far * u.skyCut) { return vec4f(1.0); }
  let p = viewPos(uv, z);
  let hit = lightAt(p);
  if (hit.amount <= 0.0) { return vec4f(1.0); }  // outside the beam: nothing to shadow
  return vec4f(shadowAt(p, hit.l, hit.dist, frag.xy));
}

// Sun shafts: light scattered by the air toward the camera, wherever the line from a pixel toward the
// sun crosses open sky. A radial gather in image space (Mitchell, GPU Gems 3 ch. 13), using the sky
// mask as the light source, so rock edges around the gap cut the light into streaks.
fn sunUv() -> vec2f {
  let d = normalize(u.lightPos);
  return project(d * 1000.0);
}

@fragment
fn fs_rays(@builtin(position) frag: vec4f) -> @location(0) vec4f {
  if (u.sun < 0.5 || u.rays <= 0.0 || normalize(u.lightPos).z <= 0.05) { return vec4f(0.0); }
  const N = 96;
  let uv = frag.xy / shadowSize();
  let sun = sunUv();
  let jitter = fract(52.9829189 * fract(dot(frag.xy, vec2f(0.06711056, 0.00583715))));
  // Gather all the way to the sun; the per-step decay makes shafts fade with distance from the gap.
  let span = sun - uv;
  var sum = 0.0;
  var weight = 1.0;
  for (var i = 0; i < N; i++) {
    let s = uv + span * ((f32(i) + jitter) / f32(N));
    if (all(s >= vec2f(0.0)) && all(s <= vec2f(1.0)) && depthNearest(s) >= u.far * u.skyCut) {
      // Bright sky (clouds, the horizon glow) sends more light: that structure becomes the streaks.
      let sky = textureSampleLevel(photoTex, samp, s, 2.0).rgb;
      let glow = 0.3 + 0.7 * exp(-dot(s - sun, s - sun) * 40.0);
      sum += glow * dot(sky, vec3f(0.2126, 0.7152, 0.0722)) * 2.0 * weight;
    }
    weight *= 0.985;
  }
  return vec4f(sum / f32(N));
}

override HORIZONTAL: bool = true;

@fragment
fn fs_blur(@builtin(position) frag: vec4f) -> @location(0) vec4f {
  const R = 3;  // small: just enough to dissolve the march jitter, so shadow edges stay crisp
  let size = shadowSize();
  let texel = vec2i(frag.xy);
  let uv = frag.xy / size;
  let z0 = depthNearest(uv);
  let step = select(vec2i(0, 1), vec2i(1, 0), HORIZONTAL);
  var sum = 0.0;
  var wsum = 0.0;
  for (var k = -R; k <= R; k++) {
    let t = clamp(texel + step * k, vec2i(0), vec2i(size) - 1);
    let zk = depthNearest((vec2f(t) + 0.5) / size);
    // Gaussian in distance, and nearly zero across a depth jump of more than a few percent.
    let dz = (zk - z0) / (0.03 * z0);
    let w = exp(-f32(k * k) / 4.5) * exp(-dz * dz);
    sum += w * textureLoad(shadowTex, t, 0).r;
    wsum += w;
  }
  return vec4f(sum / wsum);
}

// March from the surface toward the light and look for depth in the way.
fn shadowAt(p: vec3f, toLight: vec3f, dist: f32, frag: vec2f) -> f32 {
  const STEPS = 64;  // more steps, less jitter to blur away
  // Fixed per-pixel jitter (interleaved gradient noise) breaks step aliasing into fine grain. No time term: same frame every run.
  let jitter = fract(52.9829189 * fract(dot(frag, vec2f(0.06711056, 0.00583715))));
  // March all the way to the light: the beam must stop at the first surface it meets, near or far.
  let maxLen = dist;
  let enclosed = u.sun > 0.5;
  var lit = 1.0;
  for (var s = 1; s <= STEPS; s++) {
    let t = (f32(s) - jitter) / f32(STEPS);
    let q = p + toLight * (maxLen * t * t);
    if (q.z <= 0.05) {
      // Sun behind us: the cave is behind us too.
      if (enclosed) { return 0.0; }
      break;
    }
    let uv = project(q);
    if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) {
      // Sun: we look out from inside rock. A ray leaving the frame while still near is going into the
      // cave around us; one leaving far out is in open air above the canyon.
      if (enclosed && q.z < u.caveDepth) { return 0.0; }
      break;
    }
    let d = depthNearest(uv);
    // Reached open sky in front of the ray: nothing more can block it.
    if (enclosed && d >= u.far * u.skyCut) { return lit; }
    let gap = q.z - d;
    // Ignore small depth noise near the start of the march: it reads as dark cracks on lit rock.
    let bias = q.z * mix(0.06, 0.02, t);
    // Spotlight: surfaces are thin shells, so a ray passing well behind one isn't blocked.
    // Sun: everything behind a surface is solid rock.
    // Sun: rock is solid for a while behind its surface (twice its distance), so the cave walls block
    // light, but far canyon rays passing well behind the cave mouth aren't blocked by it.
    let thickness = select(q.z * 0.35, d * 2.0, enclosed);
    if (gap > bias && gap < thickness) {
      lit = min(lit, 1.0 - smoothstep(bias, bias * 1.5, gap));
      if (enclosed && lit <= 0.0) { return 0.0; }
    }
  }
  return lit;
}

fn toSrgb(c: vec3f) -> vec3f {
  let lo = c * 12.92;
  let hi = 1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055;
  return select(hi, lo, c <= vec3f(0.0031308));
}

@fragment
fn fs(@builtin(position) frag: vec4f, @location(0) uv: vec2f) -> @location(0) vec4f {
  // Implicit-level sampling: the mip level follows how small the picture is drawn on screen.
  let albedo = textureSample(albedoTex, samp, uv).rgb;   // de-lit, linear (srgb texture)
  let photo = textureSample(photoTex, samp, uv).rgb;
  // normal.png uses standard colours (z toward the camera); scene space has z into the scene.
  let n = normalize((textureSample(normalTex, samp, uv).rgb * 2.0 - 1.0) * vec3f(1.0, 1.0, -1.0));
  let z = depthAt(uv);
  let p = viewPos(uv, z);
  let sky = z >= u.far * u.skyCut;

  let hit = lightAt(p);
  // Slight wrap so rough AI-derived normals don't go hard black at the terminator.
  let ndl = clamp((dot(n, hit.l) + 0.15) / 1.15, 0.0, 1.0);
  let shadow = select(textureSampleLevel(shadowTex, samp, uv, 0.0).r, 1.0, u.shadows < 0.5);
  let haze = exp(-u.hazeBeta * z);
  let light = select(u.lightColor * (u.intensity * ndl * hit.amount * shadow * haze), vec3f(0.0), sky);

  // Base: the de-lit rock under soft night-sky light from above (cool) and bounce from below (warm),
  // so the photo's own daylight, haze and sun shafts don't show. `baked` mixes the photo back in.
  // The sky itself keeps the photo.
  let ambient = mix(vec3f(0.18, 0.14, 0.12), vec3f(0.42, 0.50, 0.65), 0.5 + 0.5 * n.y);
  let base = select(mix(albedo * ambient, photo, u.baked), photo, sky);
  var color = base * u.baseDim + albedo * light;

  // Sun shafts: only in the open air beyond the cave mouth, so they never veil the near walls.
  let air = smoothstep(u.far * 0.02, u.far * 0.12, z);
  color += u.lightColor * (u.rays * air * textureSampleLevel(raysTex, samp, uv, 0.0).r);
  if (u.flare > 0.0) { color += lensFlare(frag.xy) * u.lightColor; }

  // Brightness cap on luminance, keeping hue, so stacked lights roll off instead of clipping to white.
  let lum = dot(color, vec3f(0.2126, 0.7152, 0.0722));
  color *= (u.cap * (1.0 - exp(-lum / u.cap))) / max(lum, 1e-5);

  let view = i32(u.view + 0.5);
  if (view == 1) { color = albedo; }
  else if (view == 6) { color = photo; }
  else if (view == 2) { color = vec3f(1.0 - z / u.far); }
  else if (view == 3) { return vec4f(n * vec3f(0.5, 0.5, -0.5) + 0.5, 1.0); }
  else if (view == 4) { color = light * 0.25; }
  else if (view == 5) { color = vec3f(shadow); }
  else if (view == 7) { color = vec3f(textureSampleLevel(raysTex, samp, uv, 0.0).r); }
  return vec4f(toSrgb(color), 1.0);
}
