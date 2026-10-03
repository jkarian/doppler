// Node library. Every node is a pure function of its inputs, the time t and the track analysis, so a
// frame at a given time always renders the same (seeking and recording stay exact). Nodes that need the
// whole track (spurts, the sun gate) precompute from their constant inputs in init().
//
// Inputs are "signal" (wireable, evaluated each frame) or "const" (set in the node, fixed: they may
// trigger a precompute). Values are numbers or 3-vectors (colours). Output nodes write the renderer's
// parameters into ctx.out.

import { Music, seeded, SunGate, SunMotion, type GateOptions, type MotionOptions, type SpurtOptions } from "../music.ts";
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
  depthFeet?: (u: number, v: number) => number; // real-world distance of a picture point
}

/** What the output nodes produce for the renderer. Anything left out keeps the display's own setting. */
/**
 * Scattered lasers: from the sky, count beams (or vertical sheets) come straight down onto random spots on
 * the rock; from the ground, count fixed rigs on the rock each fire a beam in a random direction.
 */
export interface SkyLaserOut {
  from: number; // 0: from the sky, 1: from rigs on the ground
  elevMin: number; // ground: aim elevation range, degrees
  elevMax: number;
  count: number;
  trigger: number; // new random spots each time floor(trigger) changes (wire a Beat or Bar count)
  seed: number;
  uMin: number; // area of the picture the beams land in
  uMax: number;
  vMin: number;
  vMax: number;
  minDepth: number; // only land on rock at least this far away (keeps them out in the canyon)
  tilt: number; // degrees of random lean away from vertical
  drift: number; // degrees each beam sways while lit (pivoting on its landing spot)
  driftSpeed: number; // sways per second
  t: number; // time, for the sway
  sheet: number; // 0: beams; 1: vertical curtains of light
  sheetWidth: number; // degrees wide, for curtains
  fade: number; // brightness falls with the trigger's phase: exp(-phase * fade)
  width: number;
  glow: number;
  reach: number;
  color: number[];
  intensity: number;
  hit: number;
  rigs?: number[][]; // ground rigs placed by hand: [u, v, turn, tilt, cone] (degrees); empty = seeded random spots
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
  width: number; // beam thickness, degrees as seen from the middle distance
  glow: number; // how far the glow reaches, in beam widths
  reach: number; // 0..1 into the vista (cave mouth to farthest land) before the light has dissipated
  color: number[];
  intensity: number;
  sheet: number; // 0..1: fill between the beams with a plane of light
  hit: number; // how brightly beams and sheets mark the rock they strike
}

/** MRI-style scan: a stack of parallel slices sweeping through the scene, lighting the rock where they cut it. */
export interface ScanOut {
  axis: number; // 0 depth (log distance from the camera), 1 height, 2 sideways
  position: number; // 0..1 across the scene's range on that axis
  lines: number; // slices in the stack: the front one plus trailing ones, fading
  spacing: number; // between slices, as a fraction of the range
  thickness: number; // feet, real-world scale (at least about a pixel on screen)
  trail: number; // feet: the glow left behind the moving line, falling off
  reach: number; // 0..1 into the vista before it dissipates
  color: number[];
  intensity: number;
}

/** Nook lights: small lights tucked into the rock, each at a point in the picture, at its own level. */
export interface NookOut {
  lights: { u: number; v: number; level: number; r: number; b: number }[]; // r, b: this light's area and brightness multipliers
  radius: number; // feet: how far each pool of light reaches (real size, so far pools look smaller)
  standoff: number; // how far in front of the rock (toward the camera), as a fraction of the pool's radius
  color: number[];
  intensity: number;
}

export interface RenderOut {
  nooks?: NookOut;
  scan?: ScanOut;
  lasers?: LaserOut[];
  skyLasers?: SkyLaserOut[];
  sun?: { on: boolean; arc: number; azimuth: number; intensity: number; color: number[]; rays: number; flare: number; skyBoost: number; floor: number };
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
  category: "Music" | "Setup" | "Shape" | "Value" | "Scene" | "Output";
  doc: string;
  inputs: InputDef[];
  outputs: string[];
  init?(consts: Record<string, Value>, ctx: InitContext): unknown;
  eval(inputs: Record<string, Value>, ctx: EvalContext, state: unknown): Record<string, Value>;
}

const num = (v: Value): number => (typeof v === "number" ? v : Array.isArray(v) ? v[0] ?? 0 : Number(v) || 0);
const vec = (v: Value): number[] => (Array.isArray(v) ? v : [num(v), num(v), num(v)]);
const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));
/** Like min(x, ceiling), but x slows and eases into the ceiling over the last `knee` instead of stopping dead. */
const softCeiling = (x: number, ceiling: number, knee: number) => {
  const start = ceiling - knee;
  return x <= start ? x : ceiling - knee * Math.exp(-(x - start) / knee);
};

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
    doc:
      "Position in the beat grid: count (fractional), phase 0..1 within the beat, a pulse that hits 1 on each beat and decays (bass-weighted), " +
      "and confidence: 1 when the grid can be trusted, 0 on drumless tracks (multiply beat-locked effects by it).",
    inputs: [{ name: "decay", default: 0.12, kind: "const", min: 0.01, max: 2, step: 0.01, doc: "pulse decay, seconds" }],
    outputs: ["count", "phase", "pulse", "confidence"],
    eval: (i, ctx) => {
      if (!ctx.music) return { count: ctx.t * 2, phase: (ctx.t * 2) % 1, pulse: 0, confidence: 1 };
      const c = ctx.music.beat(ctx.t);
      return { count: c, phase: c - Math.floor(c), pulse: ctx.music.beatPulse(ctx.t, num(i.decay)), confidence: ctx.music.gridConfidence() };
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
    doc:
      "A value per kind of section (builds go from start to end; intros, breakdowns and outros count as quiet), eased over `ease` seconds " +
      "so changes glide. Also the section's progress 0..1.",
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
        if (s.mood === "build") return num(i.buildStart) + (num(i.buildEnd) - num(i.buildStart)) * s.progress;
        return num(i[s.mood]);
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
    doc:
      "Drops: seconds since the last one, a hit that is 1 at the drop and decays, the hit scaled by how sure the analysis is (strength), " +
      "seconds until the next one (ahead, 1e6 if none), and a wind-up that eases from 0 to 1 over the last `window` bars before it.",
    inputs: [
      { name: "decay", default: 0.8, kind: "const", min: 0.05, max: 10, step: 0.05 },
      { name: "window", default: 4, kind: "const", min: 0.5, max: 32, step: 0.5, doc: "bars of wind-up before a drop" },
    ],
    outputs: ["seconds", "hit", "strength", "ahead", "windup"],
    eval: (i, ctx) => {
      const m = ctx.music;
      if (!m) return { seconds: 1e6, hit: 0, strength: 0, ahead: 1e6, windup: 0 };
      const last = m.lastDrop(ctx.t);
      const next = m.nextDrop(ctx.t);
      const hit = Math.exp(-last.since / num(i.decay));
      const span = num(i.window) * m.phrase(ctx.t).length / ((m.a.phrases?.bars_per_phrase ?? 8));
      const x = Number.isFinite(next.until) ? clamp(1 - next.until / span, 0, 1) : 0;
      return {
        seconds: Number.isFinite(last.since) ? last.since : 1e6,
        hit,
        strength: hit * (last.drop?.confidence ?? 0),
        ahead: Number.isFinite(next.until) ? next.until : 1e6,
        windup: x * x * (3 - 2 * x),
      };
    },
  },
  {
    type: "Sound",
    category: "Music",
    doc:
      "A sound named by example in the analysis (e.g. tick, tock: tools/audio_analysis.py --sound tick@0.615): count of hits so far, " +
      "a hit that is 1 on each and decays, and seconds since the last.",
    inputs: [
      { name: "name", default: "tick", kind: "const" },
      { name: "decay", default: 0.3, kind: "const", min: 0.02, max: 5, step: 0.01 },
    ],
    outputs: ["count", "hit", "since"],
    eval: (i, ctx) => {
      const hits = ctx.music?.sound(String(i.name)) ?? [];
      let lo = 0;
      let hi = hits.length;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (hits[mid][0] <= ctx.t) lo = mid + 1;
        else hi = mid;
      }
      if (lo === 0) return { count: 0, hit: 0, since: 1e6 };
      const since = ctx.t - hits[lo - 1][0];
      return { count: lo, hit: hits[lo - 1][1] * Math.exp(-since / num(i.decay)), since };
    },
  },
  {
    type: "Phrase",
    category: "Music",
    doc:
      "Position in the phrase grid (usually 8 bars; sections change on phrase lines): count (fractional), bar 0..8 within the phrase, " +
      "progress 0..1, and a pulse that is 1 at each phrase start and decays.",
    inputs: [{ name: "decay", default: 1.5, kind: "const", min: 0.05, max: 10, step: 0.05, doc: "pulse decay, seconds" }],
    outputs: ["count", "bar", "progress", "pulse"],
    eval: (i, ctx) => {
      if (!ctx.music) {
        const c = ctx.t / 16;
        return { count: c, bar: (c % 1) * 8, progress: c % 1, pulse: Math.exp(-((c % 1) * 16) / num(i.decay)) };
      }
      const p = ctx.music.phrase(ctx.t);
      return { count: p.count, bar: p.bar, progress: p.since / p.length, pulse: Math.exp(-p.since / num(i.decay)) };
    },
  },
  {
    type: "Tension",
    category: "Music",
    doc:
      "Anticipation: tension 0..1 (rises through builds and the last bars before a drop, releases over ~2.5 s after it), " +
      "build progress 0..1, and energy 0..1 (how big each bar is, held per bar so it never leaks before a drop).",
    inputs: [],
    outputs: ["tension", "build", "energy"],
    eval: (_i, ctx) => {
      const m = ctx.music;
      if (!m) return { tension: 0, build: 0, energy: 1 };
      return { tension: m.tension(ctx.t), build: m.buildProgress(ctx.t), energy: m.barEnergy(ctx.t) };
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

  // --- Setups -----------------------------------------------------------------------------
  {
    type: "Setup",
    category: "Setup",
    doc:
      "A setup (the sun, the ground laser rigs, the scan, ...) switched on and off by the song: on in the kinds of section " +
      "ticked here, off in the rest. level fades up over fadeIn seconds when a chosen section starts (keep it short so " +
      "drops hit) and down over fadeOut when it ends. on = 0 turns the setup off everywhere. Wire level into the setup's " +
      "output node (its level input).",
    inputs: [
      { name: "on", default: 1, kind: "const", min: 0, max: 1, step: 1, doc: "master switch" },
      { name: "intro", default: 1, kind: "const", min: 0, max: 1, step: 1 },
      { name: "build", default: 1, kind: "const", min: 0, max: 1, step: 1 },
      { name: "drop", default: 1, kind: "const", min: 0, max: 1, step: 1 },
      { name: "breakdown", default: 1, kind: "const", min: 0, max: 1, step: 1 },
      { name: "normal", default: 1, kind: "const", min: 0, max: 1, step: 1 },
      { name: "outro", default: 1, kind: "const", min: 0, max: 1, step: 1 },
      { name: "fadeIn", default: 0.05, kind: "const", min: 0, max: 8, step: 0.05, doc: "seconds" },
      { name: "fadeOut", default: 1.5, kind: "const", min: 0, max: 8, step: 0.05, doc: "seconds" },
    ],
    outputs: ["level"],
    init: (c, ctx) => {
      if (num(c.on) < 0.5) return { off: true, runs: [] as [number, number][] };
      const want = (kind: string) => {
        // Version 1 files only say "quiet": it counts when breakdowns do.
        const k = kind === "quiet" ? "breakdown" : kind;
        return num(c[k] ?? 0) > 0.5;
      };
      const runs: [number, number][] = [];
      for (const s of ctx.music?.a.sections ?? []) {
        if (!want(s.kind)) continue;
        const last = runs.at(-1);
        if (last && Math.abs(last[1] - s.start) < 1e-3) last[1] = s.end;
        else runs.push([s.start, s.end]);
      }
      if (!ctx.music) runs.push([-Infinity, Infinity]);
      return { off: false, runs, fadeIn: num(c.fadeIn), fadeOut: num(c.fadeOut) };
    },
    eval: (_i, ctx, state) => {
      const s = state as { off: boolean; runs: [number, number][]; fadeIn: number; fadeOut: number };
      if (s.off) return { level: 0 };
      const ease = (x: number) => (x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x));
      let level = 0;
      for (const [a, b] of s.runs) {
        if (ctx.t >= a && ctx.t < b) level = Math.max(level, s.fadeIn > 0 ? ease((ctx.t - a) / s.fadeIn) : 1);
        else if (ctx.t >= b) level = Math.max(level, s.fadeOut > 0 ? 1 - ease((ctx.t - b) / s.fadeOut) : 0);
      }
      return { level };
    },
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
      { name: "gain", default: 1, min: 0, max: 10, step: 0.05, doc: "multiplies the sun's brightness (on top of whatever drives intensity)" },
      { name: "floorLight", default: 1, min: 0, max: 1, step: 0.05, doc: "how much sun flat ground (floor, river, ledge tops) gets: 0 = none, the sun lights only the walls" },
      { name: "maxArc", default: 180, min: 0, max: 180, step: 0.5, doc: "degrees: the sun never goes higher (low enough, it lights only the walls). It eases into this ceiling over the last 6 degrees instead of stopping dead." },
      { name: "level", default: 1, min: 0, max: 1, step: 0.01, doc: "0..1: wire a Setup node's level here to switch this on and off with the song" },
    ],
    outputs: [],
    eval: (i, ctx) => {
      const k = clamp(num(i.level), 0, 1);
      ctx.out.sun = {
        on: num(i.enabled) > 0.5 && k > 0.001, arc: softCeiling(num(i.arc), num(i.maxArc), 6), azimuth: num(i.azimuth),
        intensity: Math.max(0, num(i.intensity)) * Math.max(0, num(i.gain)) * k, floor: clamp(num(i.floorLight), 0, 1),
        color: vec(i.color), rays: Math.max(0, num(i.rays)) * k, flare: Math.max(0, num(i.flare)) * k, skyBoost: num(i.skyBoost) * k,
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
      { name: "reach", default: 0.72, min: 0.05, max: 1, step: 0.01, doc: "how far into the vista the light gets before it dissipates (0 cave mouth, 1 farthest land)" },
      { name: "glow", default: 12, min: 1, max: 80, step: 0.5, doc: "how far the glow reaches, in beam widths" },
      { name: "color", default: [0.1, 1, 0.25] },
      { name: "intensity", default: 1, min: 0, step: 0.05 },
      { name: "sheet", default: 0, min: 0, max: 1, step: 0.05 },
      { name: "hit", default: 1, min: 0, step: 0.05 },
      { name: "level", default: 1, min: 0, max: 1, step: 0.01, doc: "0..1: wire a Setup node's level here to switch this on and off with the song" },
    ],
    outputs: [],
    eval: (i, ctx) => {
      const intensity = num(i.intensity) * clamp(num(i.level), 0, 1);
      if (intensity <= 0) return {};
      (ctx.out.lasers ??= []).push({
        originU: num(i.originU), originV: num(i.originV), originDepth: Math.max(0, num(i.originDepth)),
        azimuth: num(i.azimuth), elevation: num(i.elevation), roll: num(i.roll), spread: clamp(num(i.spread), 0, 180),
        count: Math.round(clamp(num(i.count), 1, 24)), width: Math.max(0.005, num(i.width)), glow: Math.max(1, num(i.glow)), reach: clamp(num(i.reach), 0.05, 1), color: vec(i.color),
        intensity, sheet: clamp(num(i.sheet), 0, 1), hit: Math.max(0, num(i.hit)),
      });
      return {};
    },
  },
  {
    type: "SkyLaser",
    category: "Output",
    doc:
      "Scattered lasers. from 0 (sky): count straight beams come down onto random spots on the rock, new spots each " +
      "time floor(trigger) changes; sheet 1 makes them vertical curtains. from 1 (ground): count laser rigs sit at fixed " +
      "random spots on the rock and each fires a beam in a random direction (elevation elevMin-elevMax), re-aimed each " +
      "trigger step. The same track always gives the same choices. Shares the 4-fixture limit with Laser. " +
      "rigs (ground): rigs placed by hand (N on the display, Tab to laser rigs), u,v,turn,tilt,cone; ... in degrees (turn 0 = " +
      "into the scene, positive right; tilt up from level). Each re-aims at random within its cone every trigger step. " +
      "Empty: rigs at seeded random spots.",
    inputs: [
      { name: "from", default: 0, kind: "const", min: 0, max: 1, step: 1, doc: "0 sky, 1 ground rigs" },
      { name: "elevMin", default: 15, min: -30, max: 89, step: 1, doc: "ground: lowest aim, degrees up" },
      { name: "elevMax", default: 75, min: -30, max: 89, step: 1, doc: "ground: highest aim" },
      { name: "trigger", default: 0, step: 1 },
      { name: "count", default: 4, min: 1, max: 24, step: 1 },
      { name: "seed", default: 1, kind: "const", step: 1 },
      { name: "rigs", default: "", kind: "const" },
      { name: "uMin", default: 0.2, min: 0, max: 1, step: 0.01 },
      { name: "uMax", default: 0.8, min: 0, max: 1, step: 0.01 },
      { name: "vMin", default: 0.3, min: 0, max: 1, step: 0.01 },
      { name: "vMax", default: 0.9, min: 0, max: 1, step: 0.01 },
      { name: "minDepth", default: 3, min: 0, step: 0.1, doc: "only land on rock at least this far away" },
      { name: "tilt", default: 12, min: 0, max: 60, step: 0.5 },
      { name: "drift", default: 3, min: 0, max: 30, step: 0.1, doc: "degrees each beam sways, pivoting on its landing spot" },
      { name: "driftSpeed", default: 0.2, min: 0, max: 5, step: 0.01, doc: "sways per second" },
      { name: "sheet", default: 0, min: 0, max: 1, step: 0.05 },
      { name: "sheetWidth", default: 6, min: 0.5, max: 60, step: 0.5 },
      { name: "fade", default: 2, min: 0, max: 20, step: 0.1 },
      { name: "width", default: 0.07, min: 0.005, max: 2, step: 0.005 },
      { name: "reach", default: 0.72, min: 0.05, max: 1, step: 0.01, doc: "how far into the vista the light gets before it dissipates (0 cave mouth, 1 farthest land)" },
      { name: "glow", default: 12, min: 1, max: 80, step: 0.5, doc: "how far the glow reaches, in beam widths" },
      { name: "color", default: [0.2, 0.6, 1] },
      { name: "intensity", default: 1, min: 0, step: 0.05 },
      { name: "hit", default: 1.5, min: 0, step: 0.05 },
      { name: "level", default: 1, min: 0, max: 1, step: 0.01, doc: "0..1: wire a Setup node's level here to switch this on and off with the song" },
    ],
    outputs: [],
    eval: (i, ctx) => {
      // Pushed even when off (intensity 0): the display skips it, except to show rigs while placing them.
      const intensity = Math.max(0, num(i.intensity) * clamp(num(i.level), 0, 1));
      const rigs = String(i.rigs ?? "")
        .split(";")
        .map((q) => q.split(",").map(Number))
        .filter((q) => q.length >= 2 && q.every(Number.isFinite))
        .map(([u, v, turn = 0, tilt = 50, cone = 30]) => [u, v, turn, tilt, cone]);
      (ctx.out.skyLasers ??= []).push({
        rigs,
        from: num(i.from) > 0.5 ? 1 : 0, elevMin: num(i.elevMin), elevMax: Math.max(num(i.elevMin), num(i.elevMax)),
        count: Math.round(clamp(num(i.count), 1, 24)), trigger: num(i.trigger), seed: num(i.seed),
        uMin: num(i.uMin), uMax: num(i.uMax), vMin: num(i.vMin), vMax: num(i.vMax), minDepth: Math.max(0, num(i.minDepth)), tilt: Math.max(0, num(i.tilt)),
        drift: Math.max(0, num(i.drift)), driftSpeed: Math.max(0, num(i.driftSpeed)), t: ctx.t,
        sheet: clamp(num(i.sheet), 0, 1), sheetWidth: Math.max(0.5, num(i.sheetWidth)), fade: Math.max(0, num(i.fade)),
        width: Math.max(0.005, num(i.width)), glow: Math.max(1, num(i.glow)), reach: clamp(num(i.reach), 0.05, 1), color: vec(i.color), intensity, hit: Math.max(0, num(i.hit)),
      });
      return {};
    },
  },
  {
    type: "Scan",
    category: "Output",
    doc:
      "MRI-style scan: a slice (or stack of slices) sweeping through the scene, lighting the rock where it cuts, with a trail behind. Sizes in real-world feet. " +
      "axis 0: depth (contours of equal distance, near to far), 1: height, 2: sideways. position 0..1 across the scene " +
      "(wire a Bar count through fract() for a sweep locked to the music).",
    inputs: [
      { name: "axis", default: 0, kind: "const", min: 0, max: 2, step: 1 },
      { name: "position", default: 0.5, min: 0, max: 1, step: 0.005 },
      { name: "lines", default: 1, kind: "const", min: 1, max: 32, step: 1 },
      { name: "spacing", default: 0.035, min: 0.001, max: 0.5, step: 0.001 },
      { name: "thickness", default: 0.5, min: 0.01, max: 500, step: 0.1, doc: "feet, real-world scale" },
      { name: "trail", default: 80, min: 0, max: 5000, step: 1, doc: "feet: glow left behind the moving line" },
      { name: "reach", default: 0.72, min: 0.05, max: 1, step: 0.01 },
      { name: "color", default: [0.4, 0.9, 1] },
      { name: "intensity", default: 1, min: 0, step: 0.05 },
      { name: "level", default: 1, min: 0, max: 1, step: 0.01, doc: "0..1: wire a Setup node's level here to switch this on and off with the song" },
    ],
    outputs: [],
    eval: (i, ctx) => {
      const intensity = num(i.intensity) * clamp(num(i.level), 0, 1);
      if (intensity <= 0) return {};
      ctx.out.scan = {
        axis: Math.round(clamp(num(i.axis), 0, 2)), position: num(i.position), lines: Math.round(clamp(num(i.lines), 1, 32)),
        spacing: Math.max(0.001, num(i.spacing)), thickness: Math.max(0.01, num(i.thickness)), trail: Math.max(0, num(i.trail)),
        reach: clamp(num(i.reach), 0.05, 1), color: vec(i.color), intensity,
      };
      return {};
    },
  },
  {
    type: "NookLights",
    category: "Output",
    doc:
      "Small lights tucked into nooks in the rock, each washing a pool of light over the walls around it (cut off where rock " +
      "juts out in between). Each hit of the named sounds switches one on: it rises over attack seconds and eases out over " +
      "decay. positions: u,v[,area,brightness] picture points separated by ; (place them on the display with N: drag, scroll for " +
      "area, shift+scroll for brightness). Lights further than near feet (mid and far ground) are fired by the sounds " +
      "(names separated by |): pattern 0 walks through them in order, 1 sends the first sound to the left half and the second " +
      "to the right, 2 is seeded random. Nearer lights are kept for bigger moments (moments: drops and/or phrases, separated " +
      "by |): all of them come on together and fade over nearDecay. Up to 16 show at once, the brightest 7 with shadows.",
    inputs: [
      { name: "positions", default: "0.5,0.7", kind: "const" },
      { name: "sounds", default: "tick|tock", kind: "const" },
      { name: "pattern", default: 1, kind: "const", min: 0, max: 2, step: 1 },
      { name: "seed", default: 1, kind: "const", step: 1 },
      { name: "attack", default: 0.04, kind: "const", min: 0, max: 2, step: 0.01, doc: "seconds to come on" },
      { name: "decay", default: 2, kind: "const", min: 0.05, max: 10, step: 0.05, doc: "seconds to go out" },
      { name: "near", default: 8000, kind: "const", min: 0, max: 60000, step: 100, doc: "feet: lights nearer than this are kept for bigger moments" },
      { name: "moments", default: "drops|phrases", kind: "const", doc: "what fires the near lights: drops, phrases (8-bar lines)" },
      { name: "nearDecay", default: 4.5, kind: "const", min: 0.05, max: 20, step: 0.05, doc: "seconds for near lights to go out" },
      { name: "radius", default: 800, min: 20, max: 10000, step: 10, doc: "feet: how far each pool reaches (each light can scale it: scroll over its ring with N)" },
      { name: "standoff", default: 0.3, min: 0, max: 2, step: 0.01, doc: "distance in front of the rock, as a fraction of the pool's radius" },
      { name: "color", default: [1, 0.55, 0.22] },
      { name: "intensity", default: 3, min: 0, step: 0.05 },
      { name: "level", default: 1, min: 0, max: 1, step: 0.01, doc: "0..1: wire a Setup node's level here to switch this on and off with the song" },
    ],
    outputs: [],
    init: (c, ctx) => {
      const pts = String(c.positions)
        .split(";")
        .map((p) => p.split(",").map(Number))
        .filter((p) => p.length >= 2 && p.length <= 4 && p.every(Number.isFinite))
        .map(([u, v, r = 1, b = 1]) => [u, v, r, b]) as [number, number, number, number][];
      // Near lights (in front) wait for big moments; mid and far ones answer the sounds.
      const nearFeet = num(c.near);
      const isNear = pts.map(([u, v]) => (ctx.scene.depthFeet ? ctx.scene.depthFeet(u, v) < nearFeet : false));
      const far = pts.map((_, k) => k).filter((k) => !isNear[k]);
      const near = pts.map((_, k) => k).filter((k) => isNear[k]);
      const n = far.length;
      const order = [...far].sort((a, b) => pts[a][0] - pts[b][0]); // left to right
      const groups = String(c.sounds).split("|").map((s) => s.trim()).filter(Boolean);
      const pattern = Math.round(num(c.pattern));
      const rand = seeded(num(c.seed) * 7919 + 13);
      // Every hit of every named sound, in time order, with the light it fires.
      const hits: [number, number, number, number][] = []; // time, strength, light, decay
      groups.forEach((g, gi) => (ctx.music?.sound(g) ?? []).forEach(([t, s]) => hits.push([t, s, gi, num(c.decay)])));
      hits.sort((a, b) => a[0] - b[0]);
      const next = groups.map(() => 0);
      let walk = 0;
      let last = -1;
      for (const h of hits) {
        let light = 0;
        if (n === 0) break;
        if (pattern === 1 && groups.length > 1) {
          // Split left to right among the sounds: tick lights the left side, tock the right.
          const per = Math.max(1, Math.floor(n / groups.length));
          const lo = h[2] * per;
          const hi = h[2] === groups.length - 1 ? n : Math.min(n, lo + per);
          light = order[lo + (next[h[2]]++ % Math.max(1, hi - lo))];
        } else if (pattern === 2) {
          do light = far[Math.floor(rand() * n)];
          while (n > 1 && light === last);
        } else light = order[walk++ % n];
        last = light;
        h[2] = light;
      }
      if (n === 0) hits.length = 0;
      // Big moments: drops (as sure as the analysis is) and phrase starts (softer), all near lights at once.
      const kinds = String(c.moments).split("|").map((s) => s.trim());
      const moments: [number, number][] = [];
      const drops = ctx.music?.drops() ?? [];
      if (kinds.includes("drops")) drops.forEach((d) => moments.push([d.t, d.confidence]));
      if (kinds.includes("phrases")) {
        const bar = ctx.music ? (60 / Math.max(1, ctx.music.a.tempo)) * 4 : 2;
        for (const t of ctx.music?.a.phrases?.starts ?? []) {
          if (t > 0.5 && !drops.some((d) => Math.abs(d.t - t) < bar)) moments.push([t, 0.6]);
        }
      }
      for (const [t, s] of moments) for (const k of near) hits.push([t, s, k, num(c.nearDecay)]);
      hits.sort((a, b) => a[0] - b[0]);
      const longest = Math.max(num(c.decay), num(c.nearDecay));
      return { pts, hits, attack: num(c.attack), longest };
    },
    eval: (i, ctx, state) => {
      const s = state as { pts: [number, number, number, number][]; hits: [number, number, number, number][]; attack: number; longest: number };
      const level = clamp(num(i.level), 0, 1);
      const lv = s.pts.map(() => 0);
      if (level > 0) {
        const ease = (x: number) => (x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x));
        let k = s.hits.length - 1;
        while (k >= 0 && s.hits[k][0] > ctx.t) k--;
        for (; k >= 0 && ctx.t - s.hits[k][0] < s.attack + s.longest; k--) {
          const [t0, strength, light, decay] = s.hits[k];
          const x = ctx.t - t0;
          if (x >= s.attack + decay) continue;
          const env = x < s.attack ? ease(x / Math.max(s.attack, 1e-3)) : 1 - ease((x - s.attack) / decay);
          lv[light] = Math.max(lv[light], env * (0.6 + 0.4 * strength));
        }
      }
      ctx.out.nooks = {
        lights: s.pts.map(([u, v, r, b], k) => ({ u, v, r, b, level: lv[k] * level })),
        radius: Math.max(1, num(i.radius)), standoff: Math.max(0, num(i.standoff)), color: vec(i.color), intensity: Math.max(0, num(i.intensity)),
      };
      return {};
    },
  },
  {
    type: "Sync",
    category: "Output",
    doc:
      "Shifts all the visuals against the audio: lead in milliseconds, positive = visuals earlier. Use it if lights land a " +
      "little after (or before) what you hear: speakers, TVs and Bluetooth add delay. Keys { } on the display change it " +
      "(10 ms; alt: 1 ms).",
    inputs: [{ name: "lead", default: 0, kind: "const", min: -500, max: 500, step: 1, doc: "ms: positive = visuals earlier" }],
    outputs: [],
    eval: () => ({}),
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
