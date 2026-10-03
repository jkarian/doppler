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
}

export type SectionKind = "quiet" | "build" | "drop" | "normal";

export interface Section {
  kind: SectionKind;
  start: number;
  end: number;
  progress: number; // 0..1 through the section
}

/** A kick to the sun: at time t, with strength (1 = up to overhead). */
export interface Push {
  t: number;
  strength: number;
}

export class Music {
  readonly a: Analysis;
  readonly pushes: Push[];
  private readonly period: number;

  constructor(analysis: Analysis) {
    this.a = analysis;
    this.period = 60 / Math.max(analysis.tempo, 1);
    this.pushes = this.findPushes();
  }

  /**
   * Big moments that kick the sun up. Drop starts always push hard; otherwise a bar start pushes when
   * the track is much louder than over the few seconds before it. After each push there's a cooldown,
   * so a long loud stretch makes the sun pump rather than stay up. Fixed rules on the analysis:
   * the same track always gives the same pushes.
   */
  private findPushes(): Push[] {
    const pushes: Push[] = [];
    const drops = this.a.sections.filter((s) => s.kind === "drop").map((s) => s.start);
    const DROP = 1.3;
    const candidates: Push[] = drops.map((t) => ({ t, strength: DROP }));
    for (const t of this.a.downbeats) {
      if (t < 4) continue; // the "before" window needs four seconds of track behind it
      let before = 0;
      for (let k = 1; k <= 16; k++) before += this.loudness(t - k * 0.25);
      before /= 16;
      const now = Math.max(this.loudness(t + 0.05), this.loudness(t + 0.15));
      const sec = this.section(t);
      const bonus = { drop: 0.35, build: 0.15 * sec.progress, normal: 0, quiet: 0 }[sec.kind];
      const lift = ((now - before) * 5 + bonus) * (sec.kind === "quiet" ? 0.4 : 1);
      if (lift > 0.35) candidates.push({ t, strength: Math.min(1.1, lift) });
    }
    candidates.sort((a, b) => a.t - b.t || b.strength - a.strength);
    const cooldown = 3.5;
    for (const c of candidates) {
      const last = pushes.at(-1);
      if (last && c.t - last.t < cooldown) {
        // A drop always gets its push: it replaces a weaker push that came shortly before.
        if (c.strength >= DROP && last.strength < DROP) pushes[pushes.length - 1] = c;
        continue;
      }
      pushes.push(c);
    }
    return pushes;
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
 * Elastic response to pushes: how far the sun is lifted above its rest level (0 = rest, 1 = overhead).
 * Fast rise, a hold, then a damped spring back that dips a little below rest before settling.
 */
export function lift(pushes: Push[], t: number): number {
  const ATTACK = 0.25;
  const HOLD = 1.0;
  let best: number | null = null;
  for (const p of pushes) {
    const x = t - p.t;
    if (x < 0 || x > ATTACK + HOLD + 5) continue;
    let k: number;
    if (x < ATTACK) k = 1 - (1 - x / ATTACK) ** 3;
    else if (x < ATTACK + HOLD) k = 1;
    else {
      const y = x - ATTACK - HOLD;
      k = Math.exp(-y / 0.8) * Math.cos(y * 2.3);
    }
    const v = k * p.strength;
    if (best === null || v > best) best = v;
  }
  return best ?? 0;
}

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
