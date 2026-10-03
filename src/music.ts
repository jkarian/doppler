// Music clock: turns a track's analysis file into smooth, deterministic signals at any time t.
// Everything is a pure function of t, so a frame at a given time always renders the same.

export interface Analysis {
  duration: number;
  tempo: number;
  beats: number[];
  downbeats: number[];
  loudness: { rate: number; values: number[] };
  bass: { rate: number; values: number[] };
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
      const hit = this.bass(t + 0.05) - Math.min(this.bass(t - 0.15), this.bass(t - 0.3));
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

  /** Loudness averaged over the last `window` seconds: for slow moves that shouldn't jitter. */
  energy(t: number, window = 2): number {
    let sum = 0;
    const steps = 8;
    for (let k = 0; k < steps; k++) sum += this.loudness(t - (window * k) / steps);
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
  static readonly JUMP = 0.6;
  static readonly HOLD = 0.6;
  static readonly MAX = 75;
  private readonly t: number[] = [];
  private readonly from: number[] = [];
  private readonly to: number[] = [];

  constructor(spurts: Spurt[]) {
    for (const s of [...spurts].sort((a, b) => a.t - b.t)) {
      const level = this.lift(s.t);
      this.t.push(s.t);
      this.from.push(level);
      // Diminishing kicks: the higher it already is, the less a spurt adds.
      this.to.push(Math.min(SunMotion.MAX, level + s.deg * (1 - level / SunMotion.MAX) ** 1.5));
    }
  }

  /** Seconds to come back down from a given height. */
  private static returnTime(peak: number): number {
    return 3 + peak / 12;
  }

  lift(t: number): number {
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
