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
  skyBoost: f32,       // extra sky glow, pumping with the kick
  sunFloor: f32,       // how much sun flat, upward-facing ground gets (1 = all; 0 = the sun lights only walls)
  scan: vec4f,         // MRI scan: axis (0 depth, 1 height, 2 sideways), front position, spacing, thickness (in axis units)
  scanColor: vec4f,    // rgb, intensity
  scanLines: vec4f,    // number of slices, spacing (scene units), reach (0..1 into the vista), detail bump strength
  skyA: vec4f,         // sky gradient: mix with the photo, brightness, clouds, glow around the sun
  skyB: vec4f,         // sun height above the lowest open sky (degrees), lowest open sky (radians), span (radians), floor height (scene units)
  sunB: vec4f,         // bounce light strength, sun shadow softness (0 hard..1), far-ridge solid depth, cave rock solid depth (fractions of distance)
  up: vec4f,           // true vertical in scene space (the photo's camera looks down a little); w: layers behind the main one
  sunC: vec4f,         // sun terminator hardness (0 = soft Lambert falloff, 1 = hard), laser reference distance, laser line width, time (s)
  hazeC: vec4f,        // aerial haze by hand: opacity at and beyond far, near, far (scene units), on (1/0); off: hazeBeta
  waterC: vec4f,       // flowing water: speed (scene units / s), ripple size (scene units), sheen (sky reflection), glints (Blinn)
  waterD: vec4f,       // foam (rushing brightness), shininess (Blinn exponent), ripple strength (normal tilt), -
};

// How much of a surface (or laser) at distance d shows through the air (1 = all). By hand (Sky node haze, near,
// far): none before near, ramping evenly in log distance to `opacity` at far. Otherwise physical (hazeBeta).
fn airTransmit(d: f32) -> f32 {
  if (u.hazeC.w > 0.5) {
    return 1.0 - u.hazeC.x * smoothstep(log(max(u.hazeC.y, 1e-3)), log(max(u.hazeC.z, u.hazeC.y * 1.01)), log(max(d, 1e-3)));
  }
  return exp(-u.hazeBeta * d);
}

// The sky by the sun's height. Three looks, each a gradient up from the lowest open sky (t 0) to the
// top (t 1): dusk (the sun at the horizon), golden (about 14 degrees up), day (35 and higher).
const SKY_T = array(0.0, 0.08, 0.16, 0.25, 0.45, 1.0);
const SKY_DUSK = array(vec3f(0.79, 0.24, 0.35), vec3f(0.90, 0.48, 0.53), vec3f(0.58, 0.49, 0.68), vec3f(0.22, 0.30, 0.62), vec3f(0.013, 0.036, 0.25), vec3f(0.0012, 0.0022, 0.018));
const SKY_GOLD = array(vec3f(1.0, 0.55, 0.22), vec3f(0.95, 0.62, 0.38), vec3f(0.70, 0.60, 0.58), vec3f(0.40, 0.45, 0.66), vec3f(0.12, 0.20, 0.50), vec3f(0.03, 0.07, 0.25));
const SKY_DAY = array(vec3f(0.75, 0.82, 0.90), vec3f(0.62, 0.74, 0.90), vec3f(0.48, 0.64, 0.88), vec3f(0.36, 0.55, 0.86), vec3f(0.20, 0.40, 0.80), vec3f(0.08, 0.22, 0.62));

fn skyRamp(t: f32, look: i32) -> vec3f {
  var c = vec3f(0.0);
  for (var k = 0; k < 5; k++) {
    if (t >= SKY_T[k] && t <= SKY_T[k + 1]) {
      let f = (t - SKY_T[k]) / (SKY_T[k + 1] - SKY_T[k]);
      if (look == 0) { c = mix(SKY_DUSK[k], SKY_DUSK[k + 1], f); }
      else if (look == 1) { c = mix(SKY_GOLD[k], SKY_GOLD[k + 1], f); }
      else { c = mix(SKY_DAY[k], SKY_DAY[k + 1], f); }
    }
  }
  return c;
}

fn skyGradient(uv: vec2f) -> vec3f {
  let dir = normalize(viewPos(uv, 1.0));
  let t = clamp((asin(dot(dir, u.up.xyz)) - u.skyB.y) / u.skyB.z, 0.0, 1.0);
  let s = u.skyB.x;
  let toGold = smoothstep(2.0, 14.0, s);
  let toDay = smoothstep(14.0, 35.0, s);
  return mix(mix(skyRamp(t, 0), skyRamp(t, 1), toGold), skyRamp(t, 2), toDay);
}

// MRI-style scan: slices of constant depth (log distance), height or sideways position, sweeping
// through the scene. Each slice lights a thin line where it cuts the rock; trailing slices fade.
fn scanLight(p: vec3f) -> f32 {
  let axis = i32(u.scan.x + 0.5);
  var c = p.z;
  if (axis == 1) { c = p.y; }
  if (axis == 2) { c = p.x; }
  // Sizes are real-world (scene units here), but never thinner than about a pixel on screen.
  let dist = distance(p, u.camPos);
  let pixel = 2.0 * u.tanHalfFov / (u.screen.y * u.viewScale.y) * dist;
  let thick = max(u.scan.w, pixel);
  let trail = max(u.scan.z, thick);
  var g = 0.0;
  let n = i32(u.scanLines.x + 0.5);
  for (var k = 0; k < n; k++) {
    let d = c - (u.scan.y - f32(k) * u.scanLines.y);
    // Same core/glow relationship as a laser beam: a solid hard-edged core, and behind the moving
    // line a trail that starts at half the core and falls off (quick drop, long faint tail).
    let core = 1.0 - smoothstep(thick * 0.45, thick * 0.55, abs(d));
    let y = max(0.0, -d - thick * 0.5) / trail;
    let behind = select(0.0, 0.5 * (0.75 * exp(-y / 0.04) + 0.25 * exp(-y / 0.25)), d < 0.0);
    let fade = 1.0 - f32(k) / f32(n);
    g += (core + (1.0 - core) * behind) * fade * fade;
  }
  return u.scanColor.a * g * reachFade(dist, u.scanLines.z);
}

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

// A laser plane sweeping through the camera: the source blazes, a starburst and a wash of its colour over the
// picture, as when a club laser crosses your eyes. Its strength (lasers.flare) is worked out on the CPU.
fn laserFlare(frag: vec2f) -> vec3f {
  let k = lasers.flare;
  if (k <= 0.0) { return vec3f(0.0); }
  let L = lasers.l[lasers.flareFixture];
  let pc = L.origin - u.camPos;
  let tanXY = vec2f(u.tanHalfFov * u.aspect, u.tanHalfFov);
  let q = pc.xy / (pc.z * tanXY) + u.camPos.xy / (u.pivotZ * tanXY);
  let src = (q - u.center) * u.viewScale;
  let p = vec2f(frag.x / u.screen.x * 2.0 - 1.0, 1.0 - frag.y / u.screen.y * 2.0);
  let a = vec2f(u.screen.x / u.screen.y, 1.0);
  let d = (p - src) * a;
  let r = length(d);
  let ang = atan2(d.y, d.x);
  let white = mix(L.color, vec3f(1.0), 0.7);
  var c = white * exp(-r * r * 400.0) * 3.0 + L.color * exp(-r * 6.0) * 0.6;
  // Long horizontal streak (anamorphic) and a starburst.
  c += mix(L.color, vec3f(1.0), 0.4) * exp(-abs(d.y) * 120.0) * exp(-abs(d.x) * 1.2) * 0.8;
  c += L.color * (pow(abs(cos(ang * 3.0)), 80.0) + pow(abs(cos(ang * 3.0 + 0.52)), 120.0) * 0.6) * exp(-r * 2.0) * 0.7;
  // The whole picture washes with the laser's colour.
  c += L.color * 0.12;
  return c * k;
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

// Nook lights: up to 16 small lights tucked into the rock, each washing a pool of light over the walls
// around it. The display sorts them brightest first; the first 7 get shadows (rock jutting out between
// the light and a wall cuts the pool off): 3 in the shadow texture's g, b, a, 4 more in a second texture.
struct Nook {
  pos: vec3f,
  radius: f32,   // the pool's reach, scene units: light falls to nothing there
  color: vec3f,  // colour x intensity x level
  pad: f32,
};
struct Nooks {
  count: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
  l: array<Nook, 16>,
};
@group(0) @binding(9) var<storage, read> nooks: Nooks;
@group(0) @binding(10) var shadowTex2: texture_2d<f32>;  // nook lights 4-7's visibility (see fs_shadow)
// The depth the mesh is built from. Usually depthTex itself; a cut-out layer (the cave in front of painted layers)
// has its own, carried past its outline so the mesh has no step there to stretch (depthTex stays the whole
// picture's depth, for shadows, shafts and everything else that reads the scene).
@group(0) @binding(11) var meshDepthTex: texture_2d<f32>;
// The whole picture's depth: what the image-space shadows were worked out for. A layer behind compares its own
// surface with it, so a strip the camera uncovers doesn't borrow the shadow of the rock in front of it.
@group(0) @binding(12) var sceneDepthTex: texture_2d<f32>;
const NOOK_SHADOWS = 7u;

// Bright at the light, easing to nothing at the radius.
fn nookFall(d: f32, r: f32) -> f32 {
  let x = d / r;
  let w = max(0.0, 1.0 - x * x);
  return w * w / (1.0 + 2.0 * x * x);
}

// Lasers: up to 16 fixtures of up to 24 beams. Beam directions and lengths (to the first rock they hit)
// are worked out on the CPU each frame; this only draws them.
struct Laser {
  origin: vec3f, count: f32,
  aim: vec3f, sheet: f32,          // 0..1: plane of light between the beams; > 1: a scanning laser's plane, 1 + its brightness
  right: vec3f, halfSpread: f32,   // radians
  normal: vec3f, intensity: f32,   // normal of the fan's plane
  color: vec3f, width: f32,        // width: beam thickness in scene units (thins with distance)
  hit: f32, maxLen: f32, glow: f32, reach: f32,  // glow: in beam widths; reach: 0..1 into the vista
};
struct Lasers {
  count: u32, flareFixture: u32, flare: f32, pad2: u32,  // flare: a scanning plane sweeping through the camera
  l: array<Laser, 16>,
  beams: array<vec4f, 768>,        // (laser * 24 + beam) * 2: [origin, length], [direction, curtain half-angle or -1 = ground rig]
};
@group(0) @binding(8) var<storage, read> lasers: Lasers;

// Length of a laser's fan at an angle inside it, interpolated between its beams.
fn fanLength(li: u32, n: u32, a: f32, halfSpread: f32) -> f32 {
  if (n < 2u) { return lasers.beams[(li * 24u) * 2u].w; }
  let f = clamp((a / max(halfSpread, 1e-4)) * 0.5 + 0.5, 0.0, 1.0) * f32(n - 1u);
  let i0 = u32(floor(f));
  let i1 = min(i0 + 1u, n - 1u);
  return mix(lasers.beams[(li * 24u + i0) * 2u].w, lasers.beams[(li * 24u + i1) * 2u].w, fract(f));
}

// Light from the lasers reaching the camera along the view ray to surface point p (or the sky):
// glowing beams in the air, a translucent sheet, hot spots where beams land, and contour lines where
// a sheet's plane cuts the rock.
// Laser light, split two ways: `air` is glow in the air (beams, sheet planes), added on top after the
// brightness cap like real additive laser light; `rock` lights the rock (hit spots, contour lines and
// their spill), multiplying its colour like any light.
struct LaserLight {
  air: vec3f,
  rock: vec3f,
};

// Brightness across a beam at angle `ang` from its centre: a solid hard-edged core (100%) as wide as
// the beam; right outside it the glow starts at half that (so the core pops out of it) and falls to 0
// over `glow` beam widths: quickly at first, then a long faint tail into the haze.
fn beamProfile(ang: f32, width: f32, glow: f32) -> f32 {
  let core = 1.0 - smoothstep(width * 0.45, width * 0.55, ang);
  let y = max(0.0, ang - width * 0.5) / (width * glow);
  let tail = 0.5 * (0.75 * exp(-y / 0.04) + 0.25 * exp(-y / 0.25));
  return core + (1.0 - core) * tail;
}

// How far into the vista a distance is: 0 at the camera, 1 at the farthest land (log distance).
fn vistaDepth(d: f32) -> f32 {
  return clamp(log(max(d, 1.0)) / log(max(u.far / 1.5, 2.0)), 0.0, 1.0);
}

// Light dissipates before `reach` (0..1 into the vista): it never gets to the far canyon, where the
// depth map is least reliable.
fn reachFade(d: f32, reach: f32) -> f32 {
  return 1.0 - smoothstep(reach - 0.15, reach, vistaDepth(d));
}

// Laser and scan light fade with distance across the vista: full at the cave mouth, 30% at the far end
// of the land, spread evenly in log distance (gentler than real life, so far rock still lights up).
fn distanceFade(d: f32) -> f32 {
  return mix(1.0, 0.3, vistaDepth(d));
}

// Beams read their depth through brightness, like a laser show seen in haze: the part of a beam or fan near you is
// bright, a source far away is faint, brightening as the beam comes toward you. Relative to the scene's middle
// reference distance (sunC.y, about where the rigs stand): full as tuned there, up to x2.2 nearer, x0.4 at five
// times as far, faint beyond.
fn laserNear(t: f32) -> f32 {
  if (u.sunC.y <= 0.0) { return 1.0; }
  return clamp(pow(u.sunC.y / max(t, 1e-3), 0.6), 0.15, 2.2);
}

// Uneven air: laser light in the air shows the haze it passes through, wisps of denser and thinner air drifting
// slowly. Value noise in world space, two octaves (scaled to the scene: ~2 m and ~10 m wisps on true-scale scenes).
// Evaluated only where a beam or sheet actually shows on a pixel.
fn hash3(p: vec3f) -> f32 {
  // Integer hash of the lattice point (stable at canyon-scale coordinates, where sin-based hashes lose precision
  // and shimmer).
  var h = bitcast<vec3u>(vec3i(p));
  h = h * vec3u(1664525u, 22695477u, 1103515245u) + vec3u(1013904223u, 1u, 12345u);
  h.x += h.y * h.z; h.y += h.z * h.x; h.z += h.x * h.y;
  h = h ^ (h >> vec3u(16u));
  h.x += h.y * h.z;
  return f32(h.x & 0xffffffu) / 16777216.0;
}
fn vnoise(p: vec3f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let w = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(hash3(i), hash3(i + vec3f(1, 0, 0)), w.x), mix(hash3(i + vec3f(0, 1, 0)), hash3(i + vec3f(1, 1, 0)), w.x), w.y),
             mix(mix(hash3(i + vec3f(0, 0, 1)), hash3(i + vec3f(1, 0, 1)), w.x), mix(hash3(i + vec3f(0, 1, 1)), hash3(i + vec3f(1, 1, 1)), w.x), w.y), w.z);
}
fn airDensity(q: vec3f) -> f32 {
  let s = select(u.pivotZ * 0.03, 1.0, u.sunC.z > 0.0);  // metres on true-scale scenes
  let drift = vec3f(0.25, 0.04, 0.12) * s * u.sunC.w;     // a breeze: slow drift through the canyon
  let n = 0.65 * vnoise((q + drift) / (10.0 * s)) + 0.35 * vnoise((q + drift * 1.7) / (2.2 * s) + 17.0);
  return mix(0.35, 1.55, smoothstep(0.2, 0.8, n));
}

// Flowing water (the river in the vista plate): a ripple height field in world space, streaked along the flow and
// moving downstream (toward the camera here), whose slopes make the water's normal map.
fn waterLayers(q: vec2f) -> f32 {
  // Ripples in fixed world axes (never rotated, so no swirling): warped so they curl like white water; a fine chop
  // layer and a broad one for foam patches.
  let sz = u.waterC.y;
  let w = vec2f(vnoise(vec3f(q / (sz * 2.5), 11.0)), vnoise(vec3f(q / (sz * 2.5) + 5.2, 19.0))) - 0.5;
  let r = q + w * sz * 2.0;
  return 0.45 * vnoise(vec3f(r / sz, 0.5)) + 0.25 * vnoise(vec3f(r / (sz * 0.35), 7.3)) + 0.3 * vnoise(vec3f(r / (sz * 3.0), 3.1));
}

fn waterStreak(q: vec2f, dir: vec2f) -> f32 {
  // Streaks along the flow: the ripples blurred along it.
  let sz = u.waterC.y;
  var h = 0.0;
  for (var k = 0; k < 4; k++) { h += waterLayers(q + dir * (f32(k) - 1.5) * sz * 0.9); }
  return h * 0.25;
}

// Flowing water (the river in the vista plate): a flow map. The ripple field slides along the river's course (dir, on
// the ground) at speed; two copies half a cycle apart cross-fade so the sliding never stretches it over time.
fn waterHeight(q: vec2f, dir: vec2f) -> f32 {
  let period = 1.6;
  let ph = u.sunC.w / period;
  let fa = fract(ph);
  let fb = fract(ph + 0.5);
  let wa = 1.0 - abs(2.0 * fa - 1.0);
  let travel = period * u.waterC.x;
  let ha = waterStreak(q - dir * fa * travel, dir);
  let hb = waterStreak(q - dir * fb * travel + vec2f(13.7, 5.1) * u.waterC.y, dir);
  return mix(hb, ha, wa);
}

// Where a laser plane cuts the rock: a line of real width (sunC.z, scene units: about 3 inches on true-scale scenes),
// so it thins with distance, but never drawn thinner than about a pixel; below that it dims instead. Returns
// (width to draw, brightness factor). Without a real width (old scenes): `fallback`, full brightness.
fn lineWidth(dist: f32, fallback: f32) -> vec2f {
  if (u.sunC.z <= 0.0) { return vec2f(fallback, 1.0); }
  let pixel = 0.6 * 2.0 * u.tanHalfFov / (u.screen.y * u.viewScale.y) * dist;
  let w = max(u.sunC.z, pixel);
  return vec2f(w, u.sunC.z / w);
}

fn laserLight(p: vec3f, sky: bool) -> LaserLight {
  let cam = u.camPos;
  let toP = p - cam;
  let viewLen = select(length(toP), 1e7, sky);
  let v = normalize(toP);
  // Smallest thickness worth drawing: about a pixel. Thinner beams dim instead of shrinking further.
  let minAng = 1.5 * 2.0 * u.tanHalfFov / (u.screen.y * u.viewScale.y);
  var air = vec3f(0.0);
  var rock = vec3f(0.0);
  for (var li = 0u; li < lasers.count; li++) {
    let L = lasers.l[li];
    let n = u32(L.count);
    var g = 0.0;  // glow in the air
    var r = 0.0;  // light on the rock
    for (var bi = 0u; bi < n; bi++) {
      let ob = lasers.beams[(li * 24u + bi) * 2u];
      let db = lasers.beams[(li * 24u + bi) * 2u + 1u];
      let o = ob.xyz;
      let d = db.xyz;
      let b = vec4f(d, ob.w);
      // Closest approach between the view ray (cam + t v) and the beam (o + s d, 0 <= s <= len).
      let w0 = cam - o;
      let bb = dot(v, d);
      let dd = dot(v, w0);
      let ee = dot(d, w0);
      let denom = max(1.0 - bb * bb, 1e-6);
      let s = clamp((ee - bb * dd) / denom, 0.0, b.w);
      let tRaw = dot(o + s * d - cam, v);
      let t = clamp(tRaw, 0.0, viewLen);
      // Rock in front hides the beam: no glow where it passes behind the surface this pixel sees, so a
      // beam going behind a wall is cut off at the wall's edge (2% slack for where it lands on the rock).
      let seen = 1.0 - smoothstep(viewLen, viewLen * 1.02, tRaw);
      let gap = distance(cam + t * v, o + s * d);
      let ang = gap / max(t, 0.05);
      // Physical thickness: thinner on screen the further away the closest point is.
      let beamAng = L.width / max(t, 0.05);
      let drawAng = max(beamAng, minAng);
      let energy = beamAng / drawAng;
      // Curtain: a vertical sheet around this beam (sky lasers). Glow where the view ray crosses its
      // plane inside the wedge, and a line where the wedge's plane cuts the rock.
      if (db.w > 0.0) {
        let side = normalize(cross(d, vec3f(0.0, 0.0, 1.0)));
        let cn = normalize(cross(d, side));
        let vn = dot(v, cn);
        let tp = dot(o - cam, cn) / select(vn, 1e-6, abs(vn) < 1e-6);
        if (tp > 0.0 && tp < viewLen) {
          let rel = cam + tp * v - o;
          let a = atan2(dot(rel, side), dot(rel, d));
          if (abs(a) <= db.w && length(rel) < b.w / max(cos(a), 0.2)) {
            g += 0.012 * min(6.0, 1.0 / max(abs(vn), 0.03)) * distanceFade(tp) * reachFade(tp, L.reach) * laserNear(tp); // faint haze
          }
        }
        if (!sky) {
          let rel = p - o;
          let a = atan2(dot(rel, side), dot(rel, d));
          if (abs(a) <= db.w && length(rel) <= b.w / max(cos(a), 0.2) * 1.05) {
            let thick = 0.0025 * distance(p, cam) + 0.002;
            let off = dot(rel, cn);
            r += L.hit * (2.0 * exp(-pow(off / thick, 2.0)) + 0.25 * exp(-pow(off / (thick * 8.0), 2.0)));
          }
        }
      }
      // A sheet reads as a plane: its individual beams fade back.
      // Level set so the core lands near full white and the glow stays below it: if both clipped to
      // white they'd merge into one soft band.
      // Beams in the air don't depend on the depth map, so `reach` doesn't cut them; they only dim with distance.
      // A scanning laser's plane shows only its two edge lines; the beams inside only shape the plane (where rock
      // cuts it off) and aren't drawn.
      let scanned = L.sheet > 1.0;
      let edgeLine = bi == 0u || bi + 1u == n;
      let drawn = !scanned || edgeLine;
      // The air veils the far parts of a beam like the far land (aerial perspective).
      let gb = select(0.0, 6.0, drawn) * beamProfile(ang, drawAng, L.glow) * energy * mix(1.0, select(0.12, 0.5, scanned), min(L.sheet, 1.0)) * distanceFade(t) * seen * airTransmit(t) * laserNear(t);
      if (gb > 1e-4) { g += gb * airDensity(cam + t * v); }
      // A ground rig: a small bright source where the beam starts, if it's in front of what we see.
      if (db.w < 0.0) {
        let to = dot(o - cam, v);
        if (to > 0.0 && to < viewLen * 1.02) {
          let angO = length(cross(v, o - cam)) / max(to, 0.05);
          g += 3.0 * beamProfile(angO, drawAng * 1.5, 3.0) * distanceFade(to);
        }
      }
      // Hot spot where the beam lands on the rock (not for scanning planes: their line on the rock already shows it,
      // and the edge rays' landing glow sat on the river as a blob).
      if (drawn && !scanned && b.w < L.maxLen * 0.999) {
        let end = o + b.w * d;
        // Additive: a glowing point where the beam strikes, with the beam's profile...
        let te = dot(end - cam, v);
        if (te > 0.0 && te < viewLen * 1.02) {
          let angE = length(cross(v, end - cam)) / max(te, 0.05);
          g += L.hit * 2.0 * beamProfile(angE, drawAng * 1.5, 4.0) * distanceFade(te);
        }
        // ...and a soft spill lighting the rock around it.
        if (!sky) {
          let rad = 0.004 * distance(end, cam) + 0.002;
          r += L.hit * 0.3 * exp(-pow(distance(p, end) / (rad * 6.0), 2.0));
        }
      }
    }
    if (L.sheet > 0.0) {
      // The sheet in the air: where the view ray crosses the fan's plane, inside the fan.
      let vn = dot(v, L.normal);
      let tp = dot(L.origin - cam, L.normal) / select(vn, 1e-6, abs(vn) < 1e-6);
      if (tp > 0.0 && tp < viewLen) {
        let rel = cam + tp * v - L.origin;
        let a = atan2(dot(rel, L.right), dot(rel, L.aim));
        if (abs(a) <= L.halfSpread && length(rel) < fanLength(li, n, a, L.halfSpread)) {
          let through = min(6.0, 1.0 / max(abs(vn), 0.03));  // longer path through the plane seen edge-on
          if (L.sheet > 1.0) {
            // A scanning laser's plane: the beam sweeps across it, dwelling longest where it turns round, so the
            // edges are brightest (1 / sqrt(1 - x^2), the time a swing spends at each angle). Same falloff with
            // distance as the beams.
            let e = abs(a) / max(L.halfSpread, 1e-4);
            let dwell = 0.5 + 0.5 * min(4.0, inverseSqrt(max(1.0 - e * e, 1e-3)));
            g += (L.sheet - 1.0) * 0.06 * dwell * through * distanceFade(tp) * laserNear(tp) * airDensity(cam + tp * v);
          } else {
            // Faint: light scattered by haze. The contour lines and edges carry the shape.
            g += L.sheet * 0.012 * through * exp(-length(rel) / (u.far * 0.08)) * distanceFade(tp) * reachFade(tp, L.reach) * laserNear(tp) * airDensity(cam + tp * v);
          }
        }
      }
      // Contour line where the plane cuts the rock it reaches.
      if (!sky) {
        let rel = p - L.origin;
        let a = atan2(dot(rel, L.right), dot(rel, L.aim));
        let scannedPlane = L.sheet > 1.0;
        if (abs(a) <= L.halfSpread && length(rel) <= fanLength(li, n, a, L.halfSpread) * select(1.03, 1.4, scannedPlane) + 0.02) {
          let off = dot(rel, L.normal);
          let lw = lineWidth(distance(p, cam), 0.0025 * distance(p, cam) + 0.002);
          let line = min(L.sheet, 2.0) * L.hit * lw.y * (2.5 * exp(-pow(off / lw.x, 2.0)) + 0.3 * exp(-pow(off / (lw.x * 4.0), 2.0)));
          r += line;
          // A scanning plane's line also adds on top as light (after the brightness cap), so the fan's veil in the air
          // can't wash it out.
          if (scannedPlane) { g += line * 0.35 * distanceFade(length(toP)); }
          // A scanning plane leaves a fading trail on the rock behind the line, like the MRI scan's (the plane moves
          // toward +normal): it shows which way the scan travels. A real-world length behind the line (about 4 m on
          // true-scale scenes), so it shrinks with distance; the MRI scan's shape: a quick drop and a short faint tail.
          if (scannedPlane && off < 0.0) {
            let trail = select(0.03 * distance(p, cam), u.sunC.z * 80.0, u.sunC.z > 0.0);
            let y = -off / trail;
            r += L.hit * lw.y * 0.5 * (0.75 * exp(-y / 0.04) + 0.25 * exp(-y / 0.25));
          }
        }
      }
    }
    air += L.color * (L.intensity * g);
    rock += L.color * (L.intensity * r * distanceFade(length(toP)) * reachFade(length(toP), L.reach));
  }
  return LaserLight(air, rock);
}


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

fn meshDepthGrid(g: vec2u) -> f32 {
  let size = vec2i(u.imgSize);
  let t = vec2i(round(vec2f(g) / u.grid * u.imgSize - 0.5));
  return textureLoad(meshDepthTex, clamp(t, vec2i(0), size - 1), 0).r;
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
  @builtin(position) @invariant pos: vec4f,  // invariant: the depth pre-pass and the shading pass must agree exactly
  @location(0) uv: vec2f,
  @location(1) @interpolate(flat) layer: u32,  // 0 main, 1 background layer
};

// One vertex per mesh corner, generated from the index: no vertex buffers.
@vertex
fn vs(@builtin(vertex_index) i: u32, @builtin(instance_index) layer: u32) -> VsOut {
  let cols = u32(u.grid.x);
  let quad = i / 6u;
  let corner = array(vec2u(0, 0), vec2u(1, 0), vec2u(0, 1), vec2u(0, 1), vec2u(1, 0), vec2u(1, 1))[i % 6u];
  let g0 = vec2u(quad % cols, quad / cols);
  let g = g0 + corner;
  let uv = vec2f(g) / u.grid;

  // A square whose corners sit at very different depths spans a silhouette (near rock against far
  // canyon), not a surface: stretched by parallax it smears into hairs. Put the whole square at the
  // far side's depth so the near edge stays crisp and nothing stretches across the gap.
  // Mesh corners sit on the depth map's grid: one read each (a bilinear blend per corner cost 16 reads per vertex,
  // and with several layers drawn twice, depth pre-pass and shading, that was most of the frame).
  let d00 = meshDepthGrid(g0);
  let d10 = meshDepthGrid(g0 + vec2u(1, 0));
  let d01 = meshDepthGrid(g0 + vec2u(0, 1));
  let d11 = meshDepthGrid(g0 + vec2u(1, 1));
  let zmin = min(min(d00, d10), min(d01, d11));
  let zmax = max(max(d00, d10), max(d01, d11));
  let silhouette = zmax > zmin * 1.08;
  // Without a background layer, the square moves to the far side's depth: crisp edges, but it detaches from
  // its near neighbours and cracks open (dark dots) when the camera moves. With one, the square stays a
  // continuous stretched surface (no cracks), pushed back behind the hidden rock below.
  let dHere = select(select(d00, d10, corner.x == 1u), select(d01, d11, corner.x == 1u), corner.y == 1u);
  let z = select(dHere, zmax, silhouette && u.up.w < 0.5);

  let pc = viewPos(uv, z) - u.camPos;
  let tanXY = vec2f(u.tanHalfFov * u.aspect, u.tanHalfFov);
  // Where this point lands in the photo's frame when seen from the moved camera,
  // shifted so the pivot depth holds still: near rock and far canyon slide in opposite directions.
  let q = pc.xy / (pc.z * tanXY) + u.camPos.xy / (u.pivotZ * tanXY);
  var out: VsOut;
  out.pos = vec4f((q - u.center) * u.viewScale, clamp(pc.z / (u.far * 2.0), 0.0, 1.0), 1.0);
  // With a background layer, squares that span a silhouette (stretched, they show as seams when the camera
  // moves) are pushed back in the depth test instead of removed, so nothing ever opens a black hole:
  //   real surfaces (both layers) > the main layer's stretched edges > the background layer's stretched edges.
  // Where hidden rock exists, it covers the main layer's stretched edge; where it doesn't (the camera moved
  // further than the band reaches, or an edge has no band), the stretched edge shows as before. The
  // background layer's own seam (where its band meets the object it sits behind) stays behind everything,
  // so it can't slide out as a ghost outline.
  if (silhouette && u.up.w > 0.5) {
    out.pos.z = clamp(out.pos.z * (1.25 + 0.35 * f32(layer)), 0.0, 1.0);  // 1.25 main, 1.6, 1.95... behind
  }
  out.uv = uv;
  out.layer = layer;
  return out;
}

// Depth pre-pass: every layer's surface, cut-outs see-through outside their outline (alpha in the photo texture).
// The shading pass then lights only the nearest surface per pixel: with three full-screen layers, shading each
// before the depth test (which a discard in the shading shader forces) cost about 3x.
@fragment
fn fs_depth(@location(0) uv: vec2f) -> @location(0) vec4f {
  // Alpha to coverage (4x multisampling): the cut-out's filtered alpha decides how many samples of the pixel it covers,
  // so outlines come out anti-aliased instead of stair-stepped. Fully clear: no samples at all.
  let a = textureSample(photoTex, samp, uv).a;
  if (a < 0.02) { discard; }
  return vec4f(0.0, 0.0, 0.0, smoothstep(0.25, 0.75, a));
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

struct ShadowOut {
  @location(0) a: vec4f,  // r: sun or spotlight, g b a: nook lights 1-3
  @location(1) b: vec4f,  // nook lights 4-7
};

@fragment
fn fs_shadow(@builtin(position) frag: vec4f) -> ShadowOut {
  let uv = frag.xy / shadowSize();
  let z = depthNearest(uv);
  var out = ShadowOut(vec4f(1.0), vec4f(1.0));
  if (u.shadows < 0.5 || z >= u.far * u.skyCut) { return out; }
  let p = viewPos(uv, z);
  let hit = lightAt(p);
  if (hit.amount > 0.0) { out.a.x = shadowAt(p, hit.l, hit.dist, frag.xy, u.sun > 0.5, select(1.0, u.sunB.y, u.sun > 0.5)); }  // outside the beam: nothing to shadow
  for (var i = 0u; i < min(nooks.count, NOOK_SHADOWS); i++) {
    let toL = nooks.l[i].pos - p;
    let d = length(toL);
    if (d < nooks.l[i].radius && d > 1e-4) {
      let sh = shadowAt(p, toL / d, d, frag.xy, false, 1.0);
      if (i < 3u) { out.a[i + 1u] = sh; } else { out.b[i - 3u] = sh; }
    }
  }
  return out;
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
  const R = 4;  // small: just enough to dissolve the march jitter, so shadow edges stay crisp
  let size = shadowSize();
  let texel = vec2i(frag.xy);
  let uv = frag.xy / size;
  let z0 = depthNearest(uv);
  let step = select(vec2i(0, 1), vec2i(1, 0), HORIZONTAL);
  var sum = vec4f(0.0);
  var wsum = 0.0;
  // The sun's channel (r) blurs only as much as its softness asks: a hard sun keeps crisp edges.
  var sunSum = 0.0;
  var sunW = 0.0;
  // Never below about a texel: a hard sun barely blurred, and on grazing rock near the camera (the cave floor in
  // bright moments) the march's jitter stayed as a visible dot pattern.
  let sunSigma = mix(1.0, 1.8, select(1.0, u.sunB.y, u.sun > 0.5));
  let sunSigma2 = 2.0 * sunSigma * sunSigma;
  for (var k = -R; k <= R; k++) {
    let t = clamp(texel + step * k, vec2i(0), vec2i(size) - 1);
    let zk = depthNearest((vec2f(t) + 0.5) / size);
    // Gaussian in distance, and nearly zero across a depth jump of more than a few percent.
    let dz = (zk - z0) / (0.03 * z0);
    let edge = exp(-dz * dz);
    let v = textureLoad(shadowTex, t, 0);
    let w = exp(-f32(k * k) / 4.5) * edge;
    sum += w * v;
    wsum += w;
    let ws = exp(-f32(k * k) / sunSigma2) * edge;
    sunSum += ws * v.r;
    sunW += ws;
  }
  var out = sum / wsum;
  out.r = sunSum / sunW;
  return out;
}

// March from the surface toward the light and look for depth in the way.
fn shadowAt(p: vec3f, toLight: vec3f, dist: f32, frag: vec2f, enclosed: bool, soft: f32) -> f32 {
  const STEPS = 128;  // more steps, less jitter to blur away (at true scale the rays are long)
  // Fixed per-pixel jitter (interleaved gradient noise) breaks step aliasing into fine grain that the small blur
  // dissolves. (A white-noise hash instead left visible grain on near rock; the dot grid the gradient noise showed on
  // the long true-scale rays at 64 steps is gone at 128.) No time term: same frame every run.
  let jitter = fract(52.9829189 * fract(dot(frag, vec2f(0.06711056, 0.00583715))));
  // March all the way to the light: the beam must stop at the first surface it meets, near or far.
  let maxLen = dist;
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
    // Far terrain is thinner: beyond the cave, a ray passing a little behind a distant ridge is over it, not
    // inside it, so plateau tops and upper walls catch a high sun.
    let thickness = select(q.z * 0.35, d * mix(u.sunB.w, u.sunB.z, smoothstep(u.caveDepth, u.caveDepth * 3.0, d)), enclosed);
    if (gap > bias && gap < thickness) {
      // How abruptly a grazing ray goes from lit to blocked: the penumbra (soft = 1 is the old look).
      lit = min(lit, 1.0 - smoothstep(bias, bias * (1.0 + 0.5 * soft + 0.02), gap));
      if (enclosed && lit <= 0.0) { return 0.0; }
    }
  }
  return lit;
}

// Read a half-res visibility texture at full res without bleeding across depth jumps: plain bilinear mixes a
// shadowed far pixel into the near rock in front of it, which draws dark stair-stepped outlines along every
// silhouette. Each of the 4 nearest texels counts only as much as its depth matches this pixel's.
fn visUp(tex: texture_2d<f32>, uv: vec2f, z: f32) -> vec4f {
  let size = shadowSize();
  let f = uv * size - 0.5;
  let b = floor(f);
  let fr = f - b;
  var sum = vec4f(0.0);
  var wsum = 0.0;
  var seen = 0.0;
  for (var k = 0; k < 4; k++) {
    let o = vec2i(k & 1, k >> 1);
    let t = clamp(vec2i(b) + o, vec2i(0), vec2i(size) - 1);
    let zt = textureLoad(sceneDepthTex, clamp(vec2i((vec2f(t) + 0.5) / size * u.imgSize), vec2i(0), vec2i(u.imgSize) - 1), 0).r;
    let dz = (zt - z) / (0.03 * z);
    let bw = select(1.0 - fr.x, fr.x, o.x == 1) * select(1.0 - fr.y, fr.y, o.y == 1);
    let m = exp(-dz * dz);
    let w = bw * (m + 1e-3);
    sum += w * textureLoad(tex, t, 0);
    wsum += w;
    seen = max(seen, m);
  }
  // None of the shadow texels saw this surface (a layer behind, uncovered by the camera): no image-space shadow
  // is known for it. It was hidden behind nearer rock, so it counts as in that rock's shadow. (Unshadowed, the
  // uncovered strips lit up as pale bands along the cave rim at sunrise; borrowing the front rock's own shadow
  // instead gave dark seams.)
  if (seen < 0.05) { return vec4f(0.0); }
  return sum / max(wsum, 1e-6);
}

fn toSrgb(c: vec3f) -> vec3f {
  let lo = c * 12.92;
  let hi = 1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055;
  return select(hi, lo, c <= vec3f(0.0031308));
}

@fragment
fn fs(@builtin(position) frag: vec4f, @location(0) uv: vec2f, @location(1) @interpolate(flat) layer: u32) -> @location(0) vec4f {
  // Implicit-level sampling: the mip level follows how small the picture is drawn on screen.
  let albedo = textureSample(albedoTex, samp, uv).rgb;   // de-lit, linear (srgb texture)
  let photoA = textureSample(photoTex, samp, uv);
  let photo = photoA.rgb;
  // Cut-outs (the cave, the pillars) are see-through outside their outline. The depth pre-pass (fs_depth) already
  // left them out, and this pass only shades the surface that won there (depth test "equal"): no discard here,
  // so the GPU can skip hidden fragments before shading them.
  // normal.png uses standard colours (z toward the camera); scene space has z into the scene.
  var n = normalize((textureSample(normalTex, samp, uv).rgb * 2.0 - 1.0) * vec3f(1.0, 1.0, -1.0));
  let z = depthAt(uv);
  let p = viewPos(uv, z);
  let sky = z >= u.far * u.skyCut;
  // Detail bump: the depth-based normals hold ledges and strata, not the grain of the rock. Treat the de-lit
  // rock's fine brightness as height (cracks darker, so lower) and tilt the normal down its slope, so a raking
  // light picks out the texture. Fine scale only: the slope is measured two texels either side.
  let bump = u.scanLines.w;
  // The detail's scale follows how small the picture is drawn: read from the mip level that matches a screen pixel,
  // two of its texels apart. At level 0 on a 4K picture shown at 1080p, the slope skipped texels and the rock grain
  // came out as moire.
  let texPerPx = max(length(dpdx(uv) * u.imgSize), length(dpdy(uv) * u.imgSize));
  let lod = max(0.0, log2(max(texPerPx, 1e-4)));
  if (bump > 0.0 && !sky) {
    let px = 2.0 * exp2(lod) / u.imgSize;
    let lw = vec3f(0.2126, 0.7152, 0.0722);
    let hx = dot(textureSampleLevel(albedoTex, samp, uv + vec2f(px.x, 0.0), lod).rgb - textureSampleLevel(albedoTex, samp, uv - vec2f(px.x, 0.0), lod).rgb, lw);
    let hy = dot(textureSampleLevel(albedoTex, samp, uv + vec2f(0.0, px.y), lod).rgb - textureSampleLevel(albedoTex, samp, uv - vec2f(0.0, px.y), lod).rgb, lw);
    // Picture right is +x; picture down is -y in scene space.
    n = normalize(n - bump * vec3f(hx, -hy, 0.0));
  }

  let hit = lightAt(p);
  // Slight wrap so rough AI-derived normals don't go hard black at the terminator, except for the sun: a real sun
  // leaves faces turned away dark, and at high gain the wrap washed everything into general brightness.
  var ndl = select(clamp((dot(n, hit.l) + 0.15) / 1.15, 0.0, 1.0), clamp(dot(n, hit.l), 0.0, 1.0), u.sun > 0.5);
  // Hard terminator: our normals come from a depth estimate and are smooth, so light fades gradually across a face and
  // reads as a wash. Real rock has crisp ledge lips: faces toward the sun are lit solid, and the turn into shade is
  // narrow. Hardness narrows that turn (lit faces go full, the falloff squeezes into a thin band near grazing).
  if (u.sun > 0.5 && u.sunC.x > 0.0) {
    let k = u.sunC.x;
    ndl = mix(ndl, smoothstep(0.02, mix(0.6, 0.12, k), ndl), k);
  }
  var vis = select(visUp(shadowTex, uv, z), vec4f(1.0), u.shadows < 0.5);
  var vis2 = select(visUp(shadowTex2, uv, z), vec4f(1.0), u.shadows < 0.5);
  let shadow = vis.r;
  let haze = airTransmit(z);
  // Sun on the floor can be held back so it lights the walls and high ground, however high it climbs.
  // Floor: flat ground (normal pointing up) that is low (the canyon floor, the river) or part of the cave
  // around us. Plateau tops and high ledges are flat too, but stay lit.
  let nUp = dot(n, u.up.xyz);
  let flat = smoothstep(0.55, 0.85, nUp);
  let low = max(1.0 - smoothstep(u.skyB.w - 0.02 * z, u.skyB.w + 0.02 * z, dot(p, u.up.xyz)), 1.0 - smoothstep(u.caveDepth * 0.8, u.caveDepth, z));
  let floorK = select(1.0, mix(1.0, u.sunFloor, flat * low), u.sun > 0.5);
  var nook = vec3f(0.0);
  for (var i = 0u; i < min(nooks.count, 16u); i++) {
    let toL = nooks.l[i].pos - p;
    let d = length(toL);
    if (d < nooks.l[i].radius) {
      let nl = clamp((dot(n, toL / max(d, 1e-4)) + 0.15) / 1.15, 0.0, 1.0);
      var sh = 1.0;
      if (i < 3u) { sh = vis[i + 1u]; } else if (i < NOOK_SHADOWS) { sh = vis2[i - 3u]; }
      nook += nooks.l[i].color * (nookFall(d, nooks.l[i].radius) * nl * sh);
    }
  }
  // Bounce: sunlit rock and the bright sky throw warm light into faces turned away from the sun (and into
  // its shadows), more the higher the sun is. Faces turned up catch more of it.
  var bounce = 0.0;
  if (u.sun > 0.5 && u.sunB.x > 0.0) {
    let up = smoothstep(-5.0, 30.0, u.skyB.x);
    bounce = u.sunB.x * up * (1.0 - ndl * shadow) * (0.6 + 0.4 * max(nUp, 0.0));
  }
  let light = select(u.lightColor * (u.intensity * (ndl * hit.amount * shadow + 0.25 * bounce) * haze * floorK) + nook * haze, vec3f(0.0), sky);

  // Base: the de-lit rock under soft night-sky light from above (cool) and bounce from below (warm),
  // so the photo's own daylight, haze and sun shafts don't show. `baked` mixes the photo back in.
  // The sky itself keeps the photo.
  let ambient = mix(vec3f(0.18, 0.14, 0.12), vec3f(0.42, 0.50, 0.65), 0.5 + 0.5 * dot(n, u.up.xyz));
  var skyColor = photo;
  if (sky && u.skyA.x > 0.0) {
    // The gradient, with the photo's clouds kept as texture (its brightness against its own blurred self).
    let lum = dot(photo, vec3f(0.2126, 0.7152, 0.0722));
    let soft = dot(textureSampleLevel(photoTex, samp, uv, 5.0).rgb, vec3f(0.2126, 0.7152, 0.0722));
    // Cloud texture: the photo against its blurred self. Not on the background layer, whose "photo" there is
    // painted fill, not sky (it gave a bright fringe along rims).
    let clouds = select(u.skyA.z, 0.0, layer >= 1u);
    var g = skyGradient(uv) * mix(1.0, clamp(lum / max(soft, 1e-4), 0.3, 2.5), clouds) * u.skyA.y;
    // Warm glow around the sun, wider and stronger when it's low.
    if (u.sun > 0.5) {
      let c = dot(normalize(viewPos(uv, 1.0)), normalize(u.lightPos));
      let low = 1.0 - smoothstep(0.0, 30.0, u.skyB.x);
      g += u.lightColor * u.skyA.w * (exp((c - 1.0) * 60.0) + 0.5 * low * exp((c - 1.0) * 8.0));
    }
    skyColor = mix(photo, g, u.skyA.x);
  }
  let base = select(mix(albedo * ambient, photo, u.baked), skyColor * (1.0 + u.skyBoost), sky);
  var color = base * u.baseDim + albedo * light;

  // Flowing water: where the vista plate marks water (its albedo's alpha), ripples moving downstream make a normal map;
  // foam brightens the crests (rushing), the surface reflects the sky (sheen, stronger at grazing angles) and the sun
  // glints off it (Blinn-Phong).
  let water = 1.0 - textureSample(albedoTex, samp, uv).a;
  if (water > 0.01 && !sky && u.waterC.y > 0.0) {
    let up = u.up.xyz;
    let fwd = normalize(vec3f(0.0, 0.0, 1.0) - up * up.z);  // level, into the scene: the river flows toward us (-fwd)
    let side = cross(up, fwd);
    // The river's course here (tools/river_flow.py: the banks' directions averaged), an angle on the ground in the
    // vista layer's normal alpha; pointed downstream, toward us. Streaks run along it, across is sideways to it.
    let th = textureSampleLevel(normalTex, samp, uv, 0.0).a * 6.2831853;  // downstream, full circle
    let g = vec2f(dot(p, side), dot(p, fwd));
    let dirG = vec2f(cos(th), sin(th));
    let e = u.waterC.y * 0.15;
    let h0 = waterHeight(g, dirG);
    let hx = waterHeight(g + vec2f(e, 0.0), dirG) - h0;
    let hy = waterHeight(g + vec2f(0.0, e), dirG) - h0;
    let nw = normalize(up - (side * hx + fwd * hy) * (u.waterD.z / 0.15));
    let vdir = normalize(u.camPos - p);
    // Rushing: white foam on the crests, added (the water never goes darker than the photo: no debossed band).
    let crest = smoothstep(0.47, 0.62, h0);  // the streak blur narrows the range
    let foamLight = u.baseDim * 1.5 + dot(light, vec3f(0.3333)) * 0.6;
    color += water * u.waterD.x * crest * vec3f(0.88, 0.93, 0.96) * foamLight;
    // Sheen: the sky reflected, Fresnel (Schlick, water ~2%) so it grows toward grazing angles.
    let r = reflect(-vdir, nw);
    let fres = 0.02 + 0.98 * pow(1.0 - max(dot(nw, vdir), 0.0), 5.0);
    let skyR = skyGradient(project(u.camPos + r * u.far)) * u.skyA.y * (1.0 + u.skyBoost);
    color += water * u.waterC.z * fres * skyR * u.baseDim * 3.0;
    // Glints: Blinn-Phong highlight of the sun (or spot) on the rippled surface, in its shadow only where lit.
    let hv = normalize(hit.l + vdir);
    let spec = pow(max(dot(nw, hv), 0.0), u.waterD.y) * hit.amount * shadow;
    color += water * u.waterC.w * spec * u.lightColor * min(u.intensity, 1.0) * 0.6;
  }
  // Aerial perspective: far land fades toward the sky's own colour at its height in the picture (the air between
  // glows like the sky behind it), so distant ridges sit back instead of reading as black cut-outs against the
  // gradient. Follows the gradient, so dusk mountains go rosy grey and day ones blue.
  if (!sky && u.skyA.x > 0.0) {
    let air = skyGradient(uv) * u.skyA.y * (1.0 + u.skyBoost) * u.baseDim;
    color = mix(air, color, haze);
  }

  // Sun shafts: only in the open air beyond the cave mouth, so they never veil the near walls.
  let air = smoothstep(u.far * 0.02, u.far * 0.12, z);
  color += u.lightColor * (u.rays * air * textureSampleLevel(raysTex, samp, uv, 0.0).r);
  if (u.flare > 0.0) { color += lensFlare(frag.xy) * u.lightColor; }
  color += laserFlare(frag.xy);
  // Lasers and the scan light the rock: they multiply its colour (with a little floor so dark rock
  // still shows the line), then go through the cap with everything else.
  var las = LaserLight(vec3f(0.0), vec3f(0.0));
  if (lasers.count > 0u) { las = laserLight(select(p, viewPos(uv, u.far * 4.0), sky), sky); }
  var scanRock = vec3f(0.0);
  // The scan sweeps the vista: with layers, only the last one (the vista plate) takes it. On the near layers its
  // trailing slices lit whole smooth painted surfaces at once, as pale bands where the camera uncovered them.
  let vistaLayer = f32(layer) > u.up.w - 0.5;
  if (u.scanColor.a > 0.0 && !sky && vistaLayer) { scanRock = u.scanColor.rgb * scanLight(p) * distanceFade(distance(p, u.camPos)); }
  let rockLight = las.rock + scanRock;
  color += (albedo + 0.08) * rockLight * 2.0;

  // Brightness cap on luminance, keeping hue, so stacked lights roll off instead of clipping to white.
  let lum = dot(color, vec3f(0.2126, 0.7152, 0.0722));
  color *= (u.cap * (1.0 - exp(-lum / u.cap))) / max(lum, 1e-5);
  // Additive on top, after the cap: laser glow in the air, plus a hot core where light hits the rock.
  color += las.air + rockLight * 0.15;

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
