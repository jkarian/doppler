// Music clock: turns a track's analysis file into smooth, deterministic signals at any time t.
// Everything is a pure function of t, so a frame at a given time always renders the same.

export interface Analysis {
  duration: number;
  tempo: number;
  beats: number[];
  downbeats: number[];
  loudness: { rate: number; values: number[] };
  bass: { rate: number; values: number[] };
  sub?: { rate: number; values: number[] }; // kick and sub-bass, 0..1 over the track's range
  intensity?: { rate: number; values: number[] }; // loudness plus brightness, 0..1
  sections: { start: number; end: number; kind: SectionKind }[];
  hats?: [number, number][]; // high-frequency hits: [time, strength]
}

export type SectionKind = "quiet" | "build" | "drop" | "normal";

export interface Section {
  kind: SectionKind;
  start: number;
  end: number;
  progress: number; // 0..1 through the section
}

/** A kick to the sun: at time t, jump up by deg degrees along its arc. */
export interface Spurt {
  t: number;
  deg: number;
}

export class Music {
  readonly a: Analysis;
  private readonly period: number;

  constructor(analysis: Analysis) {
    this.a = analysis;
    this.period = 60 / Math.max(analysis.tempo, 1);

  }

  /**
   * Spurts for the sun: on a bar start where the kick and bass hit hard, a jump of 10-30 degrees
   * scaled by how intense the music is; drops jump 35-45. At least MIN_GAP seconds apart (drops
   * excepted), so there's time to see it crawl back. The amount is "random" from a fixed seed per bar,
   * so the same track always moves the sun the same way.
   */
  findSpurts(): Spurt[] {
    const MIN_GAP = 6;
    const drops = this.a.sections.filter((s) => s.kind === "drop").map((s) => s.start);
    // Drops always get a big spurt.
    const candidates: (Spurt & { drop: boolean })[] = drops.map((t, i) => ({ t, deg: 35 + 10 * seeded(9000 + i)(), drop: true }));
    for (const [i, t] of this.a.downbeats.entries()) {
      // A hit: the low end punching in, or the music getting noticeably more intense than a bar ago.
      const hit = Math.max(
        this.bass(t + 0.05) - Math.min(this.bass(t - 0.15), this.bass(t - 0.3)),
        this.sub(t + 0.05) - Math.min(this.sub(t - 0.3), this.sub(t - 0.6)),
        (this.energy(t + 1, 1) - this.energy(t - 1, 2)) * 0.8,
      );
      const sec = this.section(t);
      if (sec.kind === "quiet" || !(hit > 0.05 || (sec.kind === "drop" && this.bass(t + 0.05) > 0.6))) continue;
      const intensity = 0.4 + 0.6 * this.energy(t, 1);
      candidates.push({ t, deg: (10 + 20 * seeded(5000 + i)()) * intensity, drop: false });
    }
    candidates.sort((a, b) => a.t - b.t);
    const spurts: Spurt[] = [];
    let last = -Infinity;
    for (const c of candidates) {
      // Bar spurts keep their distance from the last spurt and leave room before a drop.
      if (!c.drop && (c.t - last < MIN_GAP || drops.some((d) => d > c.t && d - c.t < 2))) continue;
      spurts.push({ t: c.t, deg: c.deg });
      last = c.t;
    }
    return spurts;
  }

  /** Flicker from high-frequency hits: 0 = none, up to ~1 right on a strong hit, gone within ~0.2 s. */
  flicker(t: number): number {
    const hats = this.a.hats;
    if (!hats?.length) return 0;
    let lo = 0;
    let hi = hats.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (hats[mid][0] <= t) lo = mid + 1;
      else hi = mid;
    }
    let f = 0;
    for (let i = lo - 1; i >= 0 && t - hats[i][0] < 0.25; i--) f = Math.max(f, hats[i][1] * Math.exp(-(t - hats[i][0]) / 0.05));
    return f;
  }

  /** Fractional position along a sorted list of event times, extrapolated at the track tempo outside it. */
  private position(times: number[], t: number, spacing: number): number {
    const n = times.length;
    if (n === 0) return t / spacing;
    if (t <= times[0]) return (t - times[0]) / spacing;
    if (t >= times[n - 1]) return n - 1 + (t - times[n - 1]) / spacing;
    const i = upperBound(times, t) - 1;
    return i + (t - times[i]) / (times[i + 1] - times[i]);
  }

  /** Beats since the first beat, fractional. */
  beat(t: number): number {
    return this.position(this.a.beats, t, this.period);
  }

  /** Bars since the first downbeat, fractional. */
  bar(t: number): number {
    return this.position(this.a.downbeats, t, this.period * 4);
  }

  /** 1 on each beat, decaying with the given time constant. Bass-heavy beats hit harder. */
  beatPulse(t: number, decay = 0.12): number {
    const i = upperBound(this.a.beats, t) - 1;
    if (i < 0) return 0;
    const since = t - this.a.beats[i];
    return Math.exp(-since / decay) * (0.3 + 0.7 * this.bass(this.a.beats[i] + 0.03));
  }

  loudness(t: number): number {
    return sampleCurve(this.a.loudness, t);
  }

  bass(t: number): number {
    return sampleCurve(this.a.bass, t);
  }

  /** Kick and sub-bass presence (falls back to bass for older analysis files). */
  sub(t: number): number {
    return sampleCurve(this.a.sub ?? this.a.bass, t);
  }

  /** Loudness plus brightness: how intense the music feels, even when it's mastered flat. */
  intensity(t: number): number {
    return this.a.intensity ? sampleCurve(this.a.intensity, t) : this.loudness(t);
  }

  /** Intensity averaged over the last `window` seconds: for slow moves that shouldn't jitter. */
  energy(t: number, window = 2): number {
    let sum = 0;
    const steps = 8;
    for (let k = 0; k < steps; k++) sum += this.intensity(t - (window * k) / steps);
    return sum / steps;
  }

  section(t: number): Section {
    const s = this.a.sections.find((s) => t >= s.start && t < s.end) ?? this.a.sections.at(-1);
    if (!s) return { kind: "normal", start: 0, end: this.a.duration, progress: 0 };
    return { ...s, progress: clamp01((t - s.start) / Math.max(s.end - s.start, 1e-3)) };
  }

  /** Seconds since the most recent drop started (Infinity if none yet). */
  sinceDrop(t: number): number {
    let since = Infinity;
    for (const s of this.a.sections) if (s.kind === "drop" && s.start <= t) since = t - s.start;
    return since;
  }
}

/**
 * The sun's lift above its resting height, in degrees along its arc, from a list of spurts.
 * Each spurt eases it up (an S-curve over JUMP seconds: accelerates out of where it was, settles in
 * at the top), it holds for HOLD seconds, then returns to rest along a long S-curve: slow to leave
 * the top, quicker in the middle, gently into rest. Higher lifts take longer to come down.
 * A spurt starts from wherever the sun is. Precomputed, so any time can be evaluated directly.
 */
export class SunMotion {
  static readonly JUMP = 1.2;
  static readonly HOLD = 1.0;
  static readonly MAX = 75;
  private readonly t: number[] = [];
  private readonly from: number[] = [];
  private readonly to: number[] = [];

  // The sun itself follows the eased target like a heavy mass on a critically damped spring:
  // no abrupt starts, stops or reversals. Simulated once at RATE Hz; lift() interpolates.
  static readonly RATE = 60;
  static readonly OMEGA = 2.2; // lower = heavier
  private readonly samples: Float32Array;

  constructor(spurts: Spurt[]) {
    for (const s of [...spurts].sort((a, b) => a.t - b.t)) {
      const level = this.target(s.t);
      this.t.push(s.t);
      this.from.push(level);
      // Diminishing kicks: the higher it already is, the less a spurt adds.
      this.to.push(Math.min(SunMotion.MAX, level + s.deg * (1 - level / SunMotion.MAX) ** 1.5));
    }
    const end = (this.t.at(-1) ?? 0) + 30;
    const n = Math.ceil(end * SunMotion.RATE) + 1;
    this.samples = new Float32Array(n);
    const dt = 1 / SunMotion.RATE;
    const w = SunMotion.OMEGA;
    let x = 0;
    let v = 0;
    for (let k = 0; k < n; k++) {
      this.samples[k] = x;
      // Semi-implicit Euler, a few substeps for stability.
      for (let sub = 0; sub < 4; sub++) {
        const h = dt / 4;
        v += (w * w * (this.target(k * dt) - x) - 2 * w * v) * h;
        x += v * h;
      }
    }
  }

  lift(t: number): number {
    const x = t * SunMotion.RATE;
    if (x <= 0) return 0;
    const k = Math.floor(x);
    if (k >= this.samples.length - 1) return 0;
    return this.samples[k] + (this.samples[k + 1] - this.samples[k]) * (x - k);
  }

  /** Seconds to come back down from a given height. */
  private static returnTime(peak: number): number {
    return 5 + peak / 8;
  }

  /** Where the spurts are pulling the sun: eased up, hold, long S back down. */
  private target(t: number): number {
    const i = upperBound(this.t, t) - 1;
    if (i < 0) return 0;
    const x = t - this.t[i];
    const peak = this.to[i];
    if (x < SunMotion.JUMP) return this.from[i] + (peak - this.from[i]) * ease(x / SunMotion.JUMP);
    const y = x - SunMotion.JUMP - SunMotion.HOLD;
    if (y < 0) return peak;
    return peak * (1 - ease(Math.min(1, y / SunMotion.returnTime(peak))));
  }
}

/**
 * Sun on/off from the low end: on while the kick and bass are in, off in breakdowns and the cut before
 * a drop. Thresholds are relative to the track's own bass range (on above 60%, off below 45%, so a
 * borderline moment doesn't flicker). Switches on fast (ATTACK) and fades off slowly (RELEASE).
 * Precomputed at RATE Hz, so any time can be evaluated directly.
 */
export class SunGate {
  static readonly RATE = 60;
  static readonly ATTACK = 0.08; // seconds: snaps on
  static readonly RELEASE = 1.2; // seconds: fades off over a couple of seconds
  private readonly samples: Float32Array;

  constructor(music: Music) {
    // Kick presence and intensity together: the sun is out when the kick is in and the music is
    // intense; it goes dark in breakdowns, intros, and when the kick cuts.
    const signal = (t: number) => 0.3 * music.sub(t) + 0.7 * music.intensity(t);
    const values: number[] = [];
    for (let t = 0; t < music.a.duration; t += 0.05) values.push(signal(t));
    // Range from the playing part only: silent intros and outros would drag the thresholds down.
    const sorted = values.filter((v) => v > 0.25).sort((a, b) => a - b);
    const pct = (f: number) => sorted[Math.floor((sorted.length - 1) * f)] ?? 0;
    const lo = pct(0.15);
    const hi = pct(0.95);
    const on = lo + 0.6 * (hi - lo);
    const off = lo + 0.45 * (hi - lo);
    const n = Math.ceil(music.a.duration * SunGate.RATE) + 1;
    this.samples = new Float32Array(n);
    const dt = 1 / SunGate.RATE;
    let lit = false;
    let level = 0;
    for (let k = 0; k < n; k++) {
      const t = k * dt;
      let bass = 0; // half a second of bass, so single kicks don't toggle it
      for (let j = 0; j < 10; j++) bass += signal(t - j * 0.05);
      bass /= 10;
      if (!lit && bass > on) lit = true;
      else if (lit && bass < off) lit = false;
      const tau = lit ? SunGate.ATTACK : SunGate.RELEASE;
      level += ((lit ? 1 : 0) - level) * (1 - Math.exp(-dt / tau));
      this.samples[k] = level;
    }
  }

  value(t: number): number {
    const x = t * SunGate.RATE;
    if (x <= 0) return this.samples[0] ?? 0;
    const k = Math.floor(x);
    if (k >= this.samples.length - 1) return this.samples.at(-1) ?? 0;
    return this.samples[k] + (this.samples[k + 1] - this.samples[k]) * (x - k);
  }
}

/** Smootherstep: an S-curve from 0 to 1 with zero speed and acceleration at both ends. */
const ease = (x: number) => {
  const k = Math.min(1, Math.max(0, x));
  return k * k * k * (k * (6 * k - 15) + 10);
};

function sampleCurve(c: { rate: number; values: number[] }, t: number): number {
  const x = t * c.rate;
  const i = Math.floor(x);
  if (i < 0) return c.values[0] ?? 0;
  if (i >= c.values.length - 1) return c.values.at(-1) ?? 0;
  return c.values[i] + (c.values[i + 1] - c.values[i]) * (x - i);
}

function upperBound(a: number[], x: number): number {
  let lo = 0;
  let hi = a.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (a[mid] <= x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

const clamp01 = (x: number) => Math.min(1, Math.max(0, x));

/** Small seeded PRNG (mulberry32): the same seed always gives the same sequence. */
export function seeded(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let r = Math.imul(s ^ (s >>> 15), 1 | s);
    r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}
