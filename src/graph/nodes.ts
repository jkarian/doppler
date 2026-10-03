// Node library. Every node is a pure function of its inputs, the time t and the track analysis, so a
// frame at a given time always renders the same (seeking and recording stay exact). Nodes that need the
// whole track (spurts, the sun gate) precompute from their constant inputs in init().
//
// Inputs are "signal" (wireable, evaluated each frame) or "const" (set in the node, fixed: they may
// trigger a precompute). Values are numbers or 3-vectors (colours). Output nodes write the renderer's
// parameters into ctx.out.

import { Music, SunGate, SunMotion, type GateOptions, type MotionOptions, type SpurtOptions } from "../music.ts";
import { compile } from "./expr.ts";

export type Value = number | number[] | string;

export interface InputDef {
  name: string;
  default: Value;
  kind?: "signal" | "const"; // default "signal"
  min?: number;
  max?: number;
  step?: number;
  doc?: string;
}

export interface SceneConsts {
  gapHorizon: number; // degrees: lowest open sky above the sun, seen through the gap
}

/** What the output nodes produce for the renderer. Anything left out keeps the display's own setting. */
/** Lasers from the sky: count beams (or vertical sheets) coming straight down onto random spots on the rock. */
export interface SkyLaserOut {
  count: number;
  trigger: number; // new random spots each time floor(trigger) changes (wire a Beat or Bar count)
  seed: number;
  uMin: number; // area of the picture the beams land in
  uMax: number;
  vMin: number;
  vMax: number;
  minDepth: number; // only land on rock at least this far away (keeps them out in the canyon)
  tilt: number; // degrees of random lean away from vertical
  sheet: number; // 0: beams; 1: vertical curtains of light
  sheetWidth: number; // degrees wide, for curtains
  fade: number; // brightness falls with the trigger's phase: exp(-phase * fade)
  width: number;
  color: number[];
  intensity: number;
  hit: number;
}

export interface LaserOut {
  originU: number; // where it starts, as a point in the picture (0..1)
  originV: number;
  originDepth: number; // 0: on the rock at that pixel; > 0: in the air, this far from the camera
  azimuth: number; // degrees: 0 into the scene, 180 toward the camera, positive = right
  elevation: number; // degrees up
  roll: number; // degrees: tilts the fan's plane around the aim (0 = fan spreads sideways)
  spread: number; // degrees across the fan
  count: number; // beams
  width: number; // beam thickness, degrees as seen from the camera
  color: number[];
  intensity: number;
  sheet: number; // 0..1: fill between the beams with a plane of light
  hit: number; // how brightly beams and sheets mark the rock they strike
}

export interface RenderOut {
  lasers?: LaserOut[];
  skyLasers?: SkyLaserOut[];
  sun?: { on: boolean; arc: number; azimuth: number; intensity: number; color: number[]; rays: number; flare: number; skyBoost: number };
  camera?: { swayX: number; swayY: number; pushZ: number };
  tone?: { baseDim: number; baked: number; cap: number };
}

export interface EvalContext {
  t: number;
  music: Music | null;
  scene: SceneConsts;
  out: RenderOut;
}

export interface InitContext {
  music: Music | null;
  scene: SceneConsts;
}

export interface NodeDef {
  type: string;
  category: "Music" | "Shape" | "Value" | "Scene" | "Output";
  doc: string;
  inputs: InputDef[];
  outputs: string[];
  init?(consts: Record<string, Value>, ctx: InitContext): unknown;
  eval(inputs: Record<string, Value>, ctx: EvalContext, state: unknown): Record<string, Value>;
}

const num = (v: Value): number => (typeof v === "number" ? v : Array.isArray(v) ? v[0] ?? 0 : Number(v) || 0);
const vec = (v: Value): number[] => (Array.isArray(v) ? v : [num(v), num(v), num(v)]);
const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));

const defs: NodeDef[] = [
  // --- Music ------------------------------------------------------------------------------
  {
    type: "Time",
    category: "Music",
    doc: "Seconds into the track (or since the page opened, without a track).",
    inputs: [],
    outputs: ["t"],
    eval: (_i, ctx) => ({ t: ctx.t }),
  },
  {
    type: "Beat",
    category: "Music",
    doc: "Position in the beat grid: count (fractional), phase 0..1 within the beat, and a pulse that hits 1 on each beat and decays (bass-weighted).",
    inputs: [{ name: "decay", default: 0.12, kind: "const", min: 0.01, max: 2, step: 0.01, doc: "pulse decay, seconds" }],
    outputs: ["count", "phase", "pulse"],
    eval: (i, ctx) => {
      if (!ctx.music) return { count: ctx.t * 2, phase: (ctx.t * 2) % 1, pulse: 0 };
      const c = ctx.music.beat(ctx.t);
      return { count: c, phase: c - Math.floor(c), pulse: ctx.music.beatPulse(ctx.t, num(i.decay)) };
    },
  },
  {
    type: "Bar",
    category: "Music",
    doc: "Position in bars (4 beats): count (fractional) and phase 0..1 within the bar.",
    inputs: [],
    outputs: ["count", "phase"],
    eval: (_i, ctx) => {
      const c = ctx.music ? ctx.music.bar(ctx.t) : ctx.t / 2;
      return { count: c, phase: c - Math.floor(c) };
    },
  },
  {
    type: "Intensity",
    category: "Music",
    doc: "How intense the music feels, 0..1 (loudness plus brightness): now, and averaged over the last `window` seconds.",
    inputs: [{ name: "window", default: 2, kind: "const", min: 0.1, max: 16, step: 0.1 }],
    outputs: ["now", "slow"],
    eval: (i, ctx) => (ctx.music ? { now: ctx.music.intensity(ctx.t), slow: ctx.music.energy(ctx.t, num(i.window)) } : { now: 1, slow: 1 }),
  },
  {
    type: "Loudness",
    category: "Music",
    doc: "Loudness 0..1.",
    inputs: [],
    outputs: ["value"],
    eval: (_i, ctx) => ({ value: ctx.music ? ctx.music.loudness(ctx.t) : 1 }),
  },
  {
    type: "Bass",
    category: "Music",
    doc: "Low end: bass (under 150 Hz) and kick (sub-bass, 25-65 Hz), 0..1.",
    inputs: [],
    outputs: ["bass", "kick"],
    eval: (_i, ctx) => (ctx.music ? { bass: ctx.music.bass(ctx.t), kick: ctx.music.sub(ctx.t) } : { bass: 0, kick: 0 }),
  },
  {
    type: "Pump",
    category: "Music",
    doc: "Kick pump, like sidechain compression: a quick swell on each beat and a slower release, weighted by the kick.",
    inputs: [
      { name: "attack", default: 0.05, kind: "const", min: 0.005, max: 1, step: 0.005 },
      { name: "release", default: 0.4, kind: "const", min: 0.02, max: 4, step: 0.02 },
    ],
    outputs: ["value"],
    eval: (i, ctx) => ({ value: ctx.music ? ctx.music.pump(ctx.t, num(i.attack), num(i.release)) : 0 }),
  },
  {
    type: "Hats",
    category: "Music",
    doc: "High-frequency hits (hats, shakers): ~1 right on a strong hit, gone within ~0.2 s.",
    inputs: [],
    outputs: ["value"],
    eval: (_i, ctx) => ({ value: ctx.music ? ctx.music.flicker(ctx.t) : 0 }),
  },
  {
    type: "Section",
    category: "Music",
    doc: "A value per kind of section (builds go from start to end), eased over `ease` seconds so changes glide. Also the section's progress 0..1.",
    inputs: [
      { name: "quiet", default: 0, kind: "const", step: 0.1 },
      { name: "buildStart", default: 0, kind: "const", step: 0.1 },
      { name: "buildEnd", default: 1, kind: "const", step: 0.1 },
      { name: "drop", default: 1, kind: "const", step: 0.1 },
      { name: "normal", default: 0.5, kind: "const", step: 0.1 },
      { name: "ease", default: 2, kind: "const", min: 0, max: 16, step: 0.1 },
    ],
    outputs: ["value", "progress"],
    eval: (i, ctx) => {
      const m = ctx.music;
      if (!m) return { value: num(i.normal), progress: 0 };
      const at = (t: number) => {
        const s = m.section(t);
        if (s.kind === "build") return num(i.buildStart) + (num(i.buildEnd) - num(i.buildStart)) * s.progress;
        return num(i[s.kind]);
      };
      const n = 8;
      const span = num(i.ease);
      let sum = 0;
      for (let k = 0; k < n; k++) sum += at(ctx.t - (span * k) / n);
      return { value: sum / n, progress: m.section(ctx.t).progress };
    },
  },
  {
    type: "DropHit",
    category: "Music",
    doc: "Since the last drop: seconds, and a hit that is 1 at the drop and decays.",
    inputs: [{ name: "decay", default: 0.8, kind: "const", min: 0.05, max: 10, step: 0.05 }],
    outputs: ["seconds", "hit"],
    eval: (i, ctx) => {
      const s = ctx.music ? ctx.music.sinceDrop(ctx.t) : Infinity;
      return { seconds: Number.isFinite(s) ? s : 1e6, hit: Math.exp(-s / num(i.decay)) };
    },
  },
  {
    type: "Spurts",
    category: "Music",
    doc:
      "Heavy elastic lift (degrees). Strong bass bars kick it up by a seeded random amount scaled by intensity; drops kick harder. " +
      "Each kick eases up, holds, and returns along a long S-curve; the result follows like a mass on a spring.",
    inputs: [
      { name: "minGap", default: 6, kind: "const", min: 0, max: 30, step: 0.5, doc: "seconds between bar spurts" },
      { name: "sizeMin", default: 10, kind: "const", min: 0, max: 90, step: 1 },
      { name: "sizeMax", default: 30, kind: "const", min: 0, max: 90, step: 1 },
      { name: "dropMin", default: 35, kind: "const", min: 0, max: 90, step: 1 },
      { name: "dropMax", default: 45, kind: "const", min: 0, max: 90, step: 1 },
      { name: "rise", default: 1.2, kind: "const", min: 0.05, max: 10, step: 0.05 },
      { name: "hold", default: 1, kind: "const", min: 0, max: 10, step: 0.1 },
      { name: "returnBase", default: 5, kind: "const", min: 0.1, max: 30, step: 0.1 },
      { name: "returnPerDeg", default: 0.125, kind: "const", min: 0, max: 1, step: 0.005 },
      { name: "max", default: 75, kind: "const", min: 1, max: 180, step: 1 },
      { name: "heaviness", default: 2.2, kind: "const", min: 0.3, max: 20, step: 0.1, doc: "spring frequency: lower = heavier" },
    ],
    outputs: ["lift"],
    init: (c, ctx) => {
      const spurts: SpurtOptions = { minGap: num(c.minGap), sizeMin: num(c.sizeMin), sizeMax: num(c.sizeMax), dropMin: num(c.dropMin), dropMax: num(c.dropMax) };
      const motion: MotionOptions = {
        rise: num(c.rise), hold: num(c.hold), returnBase: num(c.returnBase), returnPerDeg: num(c.returnPerDeg), max: num(c.max), omega: num(c.heaviness),
      };
      return new SunMotion(ctx.music ? ctx.music.findSpurts(spurts) : [], motion);
    },
    eval: (_i, ctx, state) => ({ lift: (state as SunMotion).lift(ctx.t) }),
  },
  {
    type: "Gate",
    category: "Music",
    doc: "On/off from kick presence and intensity, with hysteresis: snaps on (attack), fades off (release). 1 without a track.",
    inputs: [
      { name: "on", default: 0.5, kind: "const", min: 0, max: 1, step: 0.01, doc: "fraction of the track's range" },
      { name: "off", default: 0.35, kind: "const", min: 0, max: 1, step: 0.01 },
      { name: "attack", default: 0.08, kind: "const", min: 0.001, max: 5, step: 0.01 },
      { name: "release", default: 1.2, kind: "const", min: 0.001, max: 10, step: 0.05 },
    ],
    outputs: ["value"],
    init: (c, ctx) => {
      const o: GateOptions = { on: num(c.on), off: num(c.off), attack: num(c.attack), release: num(c.release) };
      return ctx.music ? new SunGate(ctx.music, o) : null;
    },
    eval: (_i, ctx, state) => ({ value: state ? (state as SunGate).value(ctx.t) : 1 }),
  },

  // --- Scene ------------------------------------------------------------------------------
  {
    type: "GapHorizon",
    category: "Scene",
    doc: "Degrees above the camera's horizon of the lowest open sky in the sun's direction: where the sun appears in the gap.",
    inputs: [],
    outputs: ["degrees"],
    eval: (_i, ctx) => ({ degrees: ctx.scene.gapHorizon }),
  },

  // --- Values -----------------------------------------------------------------------------
  {
    type: "Number",
    category: "Value",
    doc: "A constant.",
    inputs: [{ name: "value", default: 1, kind: "const", step: 0.01 }],
    outputs: ["value"],
    eval: (i) => ({ value: num(i.value) }),
  },
  {
    type: "Color",
    category: "Value",
    doc: "A colour (linear RGB).",
    inputs: [
      { name: "r", default: 1, kind: "const", min: 0, max: 4, step: 0.01 },
      { name: "g", default: 1, kind: "const", min: 0, max: 4, step: 0.01 },
      { name: "b", default: 1, kind: "const", min: 0, max: 4, step: 0.01 },
    ],
    outputs: ["color"],
    eval: (i) => ({ color: [num(i.r), num(i.g), num(i.b)] }),
  },

  // --- Shaping ----------------------------------------------------------------------------
  {
    type: "Expression",
    category: "Shape",
    doc: "A formula of inputs a-f and t. Functions: sin cos tan abs sqrt exp log pow min max floor fract sign clamp mix step smoothstep; constants pi, tau.",
    inputs: [
      { name: "expr", default: "a", kind: "const" },
      { name: "a", default: 0 },
      { name: "b", default: 0 },
      { name: "c", default: 0 },
      { name: "d", default: 0 },
      { name: "e", default: 0 },
      { name: "f", default: 0 },
    ],
    outputs: ["value"],
    init: (c) => compile(String(c.expr)),
    eval: (i, ctx, state) => {
      const fn = state as (env: Record<string, number>) => number;
      const v = fn({ a: num(i.a), b: num(i.b), c: num(i.c), d: num(i.d), e: num(i.e), f: num(i.f), t: ctx.t });
      return { value: Number.isFinite(v) ? v : 0 };
    },
  },
  {
    type: "Add",
    category: "Shape",
    doc: "a + b.",
    inputs: [{ name: "a", default: 0 }, { name: "b", default: 0 }],
    outputs: ["value"],
    eval: (i) => ({ value: num(i.a) + num(i.b) }),
  },
  {
    type: "Multiply",
    category: "Shape",
    doc: "a × b.",
    inputs: [{ name: "a", default: 1 }, { name: "b", default: 1 }],
    outputs: ["value"],
    eval: (i) => ({ value: num(i.a) * num(i.b) }),
  },
  {
    type: "Remap",
    category: "Shape",
    doc: "Maps value from [inLo, inHi] to [outLo, outHi], clamped.",
    inputs: [
      { name: "value", default: 0 },
      { name: "inLo", default: 0, kind: "const", step: 0.1 },
      { name: "inHi", default: 1, kind: "const", step: 0.1 },
      { name: "outLo", default: 0, kind: "const", step: 0.1 },
      { name: "outHi", default: 1, kind: "const", step: 0.1 },
    ],
    outputs: ["value"],
    eval: (i) => {
      const k = clamp((num(i.value) - num(i.inLo)) / (num(i.inHi) - num(i.inLo) || 1e-9), 0, 1);
      return { value: num(i.outLo) + (num(i.outHi) - num(i.outLo)) * k };
    },
  },
  {
    type: "Oscillator",
    category: "Shape",
    doc: "offset + amplitude × sin(2π × (phase × cycles + shift)). Wire a Bar count into phase to lock it to the music.",
    inputs: [
      { name: "phase", default: 0 },
      { name: "cycles", default: 1, kind: "const", step: 0.01, doc: "per unit of phase" },
      { name: "shift", default: 0, kind: "const", step: 0.01 },
      { name: "amplitude", default: 1 },
      { name: "offset", default: 0 },
    ],
    outputs: ["value"],
    eval: (i) => ({ value: num(i.offset) + num(i.amplitude) * Math.sin(2 * Math.PI * (num(i.phase) * num(i.cycles) + num(i.shift))) }),
  },
  {
    type: "MixColor",
    category: "Shape",
    doc: "Blend from colour a to colour b by k (0..1).",
    inputs: [{ name: "a", default: [1, 0.55, 0.3] }, { name: "b", default: [1, 0.95, 0.88] }, { name: "k", default: 0 }],
    outputs: ["color"],
    eval: (i) => {
      const a = vec(i.a);
      const b = vec(i.b);
      const k = clamp(num(i.k), 0, 1);
      return { color: a.map((x, j) => x + (b[j] - x) * k) };
    },
  },

  // --- Outputs ----------------------------------------------------------------------------
  {
    type: "Sun",
    category: "Output",
    doc:
      "The directional sun. arc: degrees along its half-circle (0 front horizon, 90 overhead, 180 behind us). " +
      "intensity, colour, visible shafts, lens flare, and extra sky glow.",
    inputs: [
      { name: "arc", default: 24 },
      { name: "azimuth", default: 3 },
      { name: "intensity", default: 2.2 },
      { name: "color", default: [1, 0.82, 0.62] },
      { name: "rays", default: 0.6 },
      { name: "flare", default: 0.8 },
      { name: "skyBoost", default: 0 },
      { name: "enabled", default: 1, kind: "const", min: 0, max: 1, step: 1 },
    ],
    outputs: [],
    eval: (i, ctx) => {
      ctx.out.sun = {
        on: num(i.enabled) > 0.5, arc: num(i.arc), azimuth: num(i.azimuth), intensity: Math.max(0, num(i.intensity)),
        color: vec(i.color), rays: Math.max(0, num(i.rays)), flare: Math.max(0, num(i.flare)), skyBoost: num(i.skyBoost),
      };
      return {};
    },
  },
  {
    type: "Camera",
    category: "Output",
    doc: "Camera motion in units of the sway amount (1 = the largest parallax shift the display allows): sideways, up, and push in (never back).",
    inputs: [{ name: "swayX", default: 0 }, { name: "swayY", default: 0 }, { name: "pushZ", default: 0 }],
    outputs: [],
    eval: (i, ctx) => {
      ctx.out.camera = { swayX: num(i.swayX), swayY: num(i.swayY), pushZ: Math.max(0, num(i.pushZ)) };
      return {};
    },
  },
  {
    type: "Laser",
    category: "Output",
    doc:
      "A laser fixture: count beams fanned over spread degrees from an origin in the picture (originDepth 0 = on the rock " +
      "there). sheet fills between the beams with a plane of light; hit sets how brightly beams and sheets mark the rock " +
      "they strike. Up to 4 lasers render.",
    inputs: [
      { name: "originU", default: 0.5, min: 0, max: 1, step: 0.005 },
      { name: "originV", default: 0.7, min: 0, max: 1, step: 0.005 },
      { name: "originDepth", default: 0, min: 0, step: 0.05 },
      { name: "azimuth", default: 0, step: 0.5 },
      { name: "elevation", default: 20, step: 0.5 },
      { name: "roll", default: 0, step: 0.5 },
      { name: "spread", default: 40, min: 0, max: 180, step: 0.5 },
      { name: "count", default: 7, min: 1, max: 24, step: 1 },
      { name: "width", default: 0.08, min: 0.005, max: 2, step: 0.005 },
      { name: "color", default: [0.1, 1, 0.25] },
      { name: "intensity", default: 1, min: 0, step: 0.05 },
      { name: "sheet", default: 0, min: 0, max: 1, step: 0.05 },
      { name: "hit", default: 1, min: 0, step: 0.05 },
    ],
    outputs: [],
    eval: (i, ctx) => {
      if (num(i.intensity) <= 0) return {};
      (ctx.out.lasers ??= []).push({
        originU: num(i.originU), originV: num(i.originV), originDepth: Math.max(0, num(i.originDepth)),
        azimuth: num(i.azimuth), elevation: num(i.elevation), roll: num(i.roll), spread: clamp(num(i.spread), 0, 180),
        count: Math.round(clamp(num(i.count), 1, 24)), width: Math.max(0.005, num(i.width)), color: vec(i.color),
        intensity: num(i.intensity), sheet: clamp(num(i.sheet), 0, 1), hit: Math.max(0, num(i.hit)),
      });
      return {};
    },
  },
  {
    type: "SkyLaser",
    category: "Output",
    doc:
      "Lasers from the sky: count straight beams coming down onto random spots on the rock inside an area of the picture. " +
      "New spots each time floor(trigger) changes (wire a Beat or Bar count); the same track always picks the same spots. " +
      "sheet 1 makes them vertical curtains of light. Shares the 4-fixture limit with Laser.",
    inputs: [
      { name: "trigger", default: 0, step: 1 },
      { name: "count", default: 4, min: 1, max: 24, step: 1 },
      { name: "seed", default: 1, kind: "const", step: 1 },
      { name: "uMin", default: 0.2, min: 0, max: 1, step: 0.01 },
      { name: "uMax", default: 0.8, min: 0, max: 1, step: 0.01 },
      { name: "vMin", default: 0.3, min: 0, max: 1, step: 0.01 },
      { name: "vMax", default: 0.9, min: 0, max: 1, step: 0.01 },
      { name: "minDepth", default: 3, min: 0, step: 0.1, doc: "only land on rock at least this far away" },
      { name: "tilt", default: 12, min: 0, max: 60, step: 0.5 },
      { name: "sheet", default: 0, min: 0, max: 1, step: 0.05 },
      { name: "sheetWidth", default: 6, min: 0.5, max: 60, step: 0.5 },
      { name: "fade", default: 2, min: 0, max: 20, step: 0.1 },
      { name: "width", default: 0.07, min: 0.005, max: 2, step: 0.005 },
      { name: "color", default: [0.2, 0.6, 1] },
      { name: "intensity", default: 1, min: 0, step: 0.05 },
      { name: "hit", default: 1.5, min: 0, step: 0.05 },
    ],
    outputs: [],
    eval: (i, ctx) => {
      if (num(i.intensity) <= 0) return {};
      (ctx.out.skyLasers ??= []).push({
        count: Math.round(clamp(num(i.count), 1, 24)), trigger: num(i.trigger), seed: num(i.seed),
        uMin: num(i.uMin), uMax: num(i.uMax), vMin: num(i.vMin), vMax: num(i.vMax), minDepth: Math.max(0, num(i.minDepth)), tilt: Math.max(0, num(i.tilt)),
        sheet: clamp(num(i.sheet), 0, 1), sheetWidth: Math.max(0.5, num(i.sheetWidth)), fade: Math.max(0, num(i.fade)),
        width: Math.max(0.005, num(i.width)), color: vec(i.color), intensity: num(i.intensity), hit: Math.max(0, num(i.hit)),
      });
      return {};
    },
  },
  {
    type: "Tone",
    category: "Output",
    doc: "Overall look: base brightness of the night scene, how much of the photo's own lighting shows, brightness cap.",
    inputs: [{ name: "baseDim", default: 0.14 }, { name: "baked", default: 0.1 }, { name: "cap", default: 1.4 }],
    outputs: [],
    eval: (i, ctx) => {
      ctx.out.tone = { baseDim: Math.max(0, num(i.baseDim)), baked: clamp(num(i.baked), 0, 1), cap: Math.max(0.05, num(i.cap)) };
      return {};
    },
  },
];

export const NODE_TYPES: Record<string, NodeDef> = Object.fromEntries(defs.map((d) => [d.type, d]));
