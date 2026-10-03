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
  scanLines: vec4f,    // number of slices, spacing (scene units), reach (0..1 into the vista), -
  skyA: vec4f,         // sky gradient: mix with the photo, brightness, clouds, glow around the sun
  skyB: vec4f,         // sun height above the lowest open sky (degrees), lowest open sky (radians), span (radians), floor height (scene units)
  sunB: vec4f,         // bounce light strength, sun shadow softness (0 hard..1), far-ridge solid depth, cave rock solid depth (fractions of distance)
  up: vec4f,           // true vertical in scene space (the photo's camera looks down a little), background layer present (1/0)
};

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
const NOOK_SHADOWS = 7u;

// Bright at the light, easing to nothing at the radius.
fn nookFall(d: f32, r: f32) -> f32 {
  let x = d / r;
  let w = max(0.0, 1.0 - x * x);
  return w * w / (1.0 + 2.0 * x * x);
}

// Lasers: up to 4 fixtures of up to 24 beams. Beam directions and lengths (to the first rock they hit)
// are worked out on the CPU each frame; this only draws them.
struct Laser {
  origin: vec3f, count: f32,
  aim: vec3f, sheet: f32,
  right: vec3f, halfSpread: f32,   // radians
  normal: vec3f, intensity: f32,   // normal of the fan's plane
  color: vec3f, width: f32,        // width: beam thickness in scene units (thins with distance)
  hit: f32, maxLen: f32, glow: f32, reach: f32,  // glow: in beam widths; reach: 0..1 into the vista
};
struct Lasers {
  count: u32, pad0: u32, pad1: u32, pad2: u32,
  l: array<Laser, 4>,
  beams: array<vec4f, 192>,        // (laser * 24 + beam) * 2: [origin, length], [direction, curtain half-angle or -1 = ground rig]
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
            g += 0.012 * min(6.0, 1.0 / max(abs(vn), 0.03)) * distanceFade(tp) * reachFade(tp, L.reach); // faint haze
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
      g += 6.0 * beamProfile(ang, drawAng, L.glow) * energy * mix(1.0, 0.12, L.sheet) * distanceFade(t) * seen;
      // A ground rig: a small bright source where the beam starts, if it's in front of what we see.
      if (db.w < 0.0) {
        let to = dot(o - cam, v);
        if (to > 0.0 && to < viewLen * 1.02) {
          let angO = length(cross(v, o - cam)) / max(to, 0.05);
          g += 3.0 * beamProfile(angO, drawAng * 1.5, 3.0) * distanceFade(to);
        }
      }
      // Hot spot where the beam lands on the rock.
      if (b.w < L.maxLen * 0.999) {
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
          // Faint: light scattered by haze. The contour lines and edges carry the shape.
          g += L.sheet * 0.012 * min(6.0, 1.0 / max(abs(vn), 0.03)) * exp(-length(rel) / (u.far * 0.08)) * distanceFade(tp) * reachFade(tp, L.reach);
        }
      }
      // Contour line where the plane cuts the rock it reaches.
      if (!sky) {
        let rel = p - L.origin;
        let a = atan2(dot(rel, L.right), dot(rel, L.aim));
        if (abs(a) <= L.halfSpread && length(rel) <= fanLength(li, n, a, L.halfSpread) * 1.03 + 0.02) {
          let off = dot(rel, L.normal);
          let thick = 0.0025 * distance(p, cam) + 0.002;
          r += L.sheet * L.hit * (2.5 * exp(-pow(off / thick, 2.0)) + 0.3 * exp(-pow(off / (thick * 8.0), 2.0)));
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
  let d00 = depthAt(vec2f(g0) / u.grid);
  let d10 = depthAt(vec2f(g0 + vec2u(1, 0)) / u.grid);
  let d01 = depthAt(vec2f(g0 + vec2u(0, 1)) / u.grid);
  let d11 = depthAt(vec2f(g0 + vec2u(1, 1)) / u.grid);
  let zmin = min(min(d00, d10), min(d01, d11));
  let zmax = max(max(d00, d10), max(d01, d11));
  let silhouette = zmax > zmin * 1.08;
  let z = select(depthAt(uv), zmax, silhouette);

  let pc = viewPos(uv, z) - u.camPos;
  let tanXY = vec2f(u.tanHalfFov * u.aspect, u.tanHalfFov);
  // Where this point lands in the photo's frame when seen from the moved camera,
  // shifted so the pivot depth holds still: near rock and far canyon slide in opposite directions.
  let q = pc.xy / (pc.z * tanXY) + u.camPos.xy / (u.pivotZ * tanXY);
  var out: VsOut;
  out.pos = vec4f((q - u.center) * u.viewScale, clamp(pc.z / (u.far * 2.0), 0.0, 1.0), 1.0);
  // With a background layer, both layers drop the squares that span a silhouette (stretched, they show as
  // dark seams when the camera moves). The main layer's gaps are filled by the background layer's hidden
  // rock; the background layer's own silhouettes (where its band of hidden rock meets the near object it
  // sits behind) would otherwise slide out from behind that object as a ghost outline.
  if (silhouette && u.up.w > 0.5) { out.pos = vec4f(0.0, 0.0, -1.0, 1.0); }
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
  const R = 3;  // small: just enough to dissolve the march jitter, so shadow edges stay crisp
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
  let sunSigma2 = 2.0 * mix(0.3, 1.5, select(1.0, u.sunB.y, u.sun > 0.5)) * mix(0.3, 1.5, select(1.0, u.sunB.y, u.sun > 0.5));
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
  const STEPS = 64;  // more steps, less jitter to blur away
  // Fixed per-pixel jitter (interleaved gradient noise) breaks step aliasing into fine grain. No time term: same frame every run.
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
  var vis = select(textureSampleLevel(shadowTex, samp, uv, 0.0), vec4f(1.0), u.shadows < 0.5);
  var vis2 = select(textureSampleLevel(shadowTex2, samp, uv, 0.0), vec4f(1.0), u.shadows < 0.5);
  let shadow = vis.r;
  let haze = exp(-u.hazeBeta * z);
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
    var g = skyGradient(uv) * mix(1.0, clamp(lum / max(soft, 1e-4), 0.3, 2.5), u.skyA.z) * u.skyA.y;
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

  // Sun shafts: only in the open air beyond the cave mouth, so they never veil the near walls.
  let air = smoothstep(u.far * 0.02, u.far * 0.12, z);
  color += u.lightColor * (u.rays * air * textureSampleLevel(raysTex, samp, uv, 0.0).r);
  if (u.flare > 0.0) { color += lensFlare(frag.xy) * u.lightColor; }
  // Lasers and the scan light the rock: they multiply its colour (with a little floor so dark rock
  // still shows the line), then go through the cap with everything else.
  var las = LaserLight(vec3f(0.0), vec3f(0.0));
  if (lasers.count > 0u) { las = laserLight(select(p, viewPos(uv, u.far * 4.0), sky), sky); }
  var scanRock = vec3f(0.0);
  if (u.scanColor.a > 0.0 && !sky) { scanRock = u.scanColor.rgb * scanLight(p) * distanceFade(distance(p, u.camPos)); }
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
