// Music monitor: a panel over the display (G) for seeing and tuning how the visuals respond to the music.
//
//   Timeline   the whole track: sections (coloured, named), drops (triangles, bigger = surer), phrase
//              lines, the music's energy and tension, the sun's height as the graph drives it, and one
//              lane per setup showing when it's on. Click or drag to seek; hover for details.
//   Meters     live levels, like an equalizer: what the music is doing now, and each setup's level.
//   Setups     one row per Setup node: master on/off, which kinds of section it plays in, and its
//              settings (the sliders of the nodes that feed it), opened one setup at a time.
//              Changes apply at once, redraw the timeline, reach the editor, and are saved with Save.

import { mood, type Music, type SectionKind } from "./music.ts";
import { NODE_TYPES, type Value } from "./graph/nodes.ts";
import { inputHelp, nodeHelp } from "./graph/help.ts";
import type { Graph, GraphNode, GraphRuntime } from "./graph/runtime.ts";

export interface MonitorOptions {
  music: Music;
  runtime: GraphRuntime;
  time: () => number; // current track time, seconds
  seek: (t: number) => void;
  edited: (graph: Graph) => void; // graph changed here: pass it on (editor)
  save: () => Promise<string>; // returns a status line
}

const MOOD_COLOR: Record<string, string> = { quiet: "#3987e5", build: "#d95926", drop: "#e66767", normal: "#6f6e69" };
const KINDS = ["intro", "build", "drop", "breakdown", "normal", "outro"] as const;
const TENSION = "#e87ba4";
const SUN = "#ffd27a";
const ENERGY = "rgba(255,255,255,0.22)";
const SKIP_TYPES = new Set(["Expression", "Color", "MixColor", "Time", "Setup", "Number", "GapHorizon"]);
const SKIP_INPUTS = new Set(["level", "trigger", "seed", "t"]);

const css = `
#monitor { position: fixed; left: 0; right: 0; bottom: 0; padding: 10px 14px 12px; background: rgba(8,8,10,0.95);
  color: #ddd; font: 12px/1.35 system-ui, sans-serif; display: grid; grid-template-columns: 1fr auto; gap: 8px 16px;
  border-top: 1px solid rgba(255,255,255,0.08); max-height: 60vh; overflow: auto; }
#monitor[hidden] { display: none; }
#monitor .head { grid-column: 1 / -1; grid-row: 1; display: flex; gap: 14px; align-items: center; color: #aaa; flex-wrap: wrap; }
#monitor .head b { color: #fff; }
#monitor .head .save { margin-left: auto; background: #ffd27a; color: #000; border: 0; border-radius: 3px; padding: 3px 12px; cursor: pointer; font-weight: 600; }
#monitor .key { display: inline-flex; align-items: center; gap: 4px; }
#monitor .key i { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
#monitor .master { grid-column: 1 / -1; grid-row: 2; display: flex; gap: 18px; flex-wrap: wrap; }
#monitor .master label { display: flex; align-items: center; gap: 6px; color: #aaa; }
#monitor .master input { width: 140px; accent-color: #ffd27a; }
#monitor .master output { color: #fff; min-width: 36px; font-variant-numeric: tabular-nums; }
#monitor .tl { grid-column: 1; grid-row: 3; position: relative; align-self: end; }
#monitor .tl canvas { width: 100%; display: block; cursor: pointer; }
#monitor .tip { position: absolute; top: 0; pointer-events: none; background: rgba(0,0,0,0.85); padding: 3px 6px;
  border-radius: 3px; white-space: nowrap; font-size: 11px; transform: translateX(-50%); }
#monitor .meters { grid-column: 2; grid-row: 3; display: flex; gap: 6px; align-items: flex-end; }
#monitor .meter { width: 36px; display: flex; flex-direction: column; align-items: center; gap: 4px; }
#monitor .bar { position: relative; width: 16px; height: 118px; background: rgba(255,255,255,0.07); border-radius: 3px; overflow: hidden; }
#monitor .fill { position: absolute; left: 0; right: 0; bottom: 0; border-radius: 3px 3px 0 0; }
#monitor .peak { position: absolute; left: 0; right: 0; height: 2px; background: #fff; opacity: 0.7; }
#monitor .meter span { font-size: 10px; color: #aaa; text-align: center; line-height: 1.1; }
#monitor .sep { width: 1px; height: 130px; background: rgba(255,255,255,0.12); margin: 0 4px; }
#monitor .setups { grid-column: 1 / -1; grid-row: 4; display: flex; flex-direction: column; gap: 4px; }
#monitor .setup { display: grid; grid-template-columns: 14px 190px 64px 1fr auto; gap: 10px; align-items: center;
  background: rgba(255,255,255,0.04); border-radius: 4px; padding: 5px 8px; }
#monitor .setup.off { opacity: 0.55; }
#monitor .swatch { width: 12px; height: 12px; border-radius: 3px; }
#monitor .setup .name { color: #fff; font-weight: 600; }
#monitor button.tog { border: 1px solid rgba(255,255,255,0.25); background: none; color: #aaa; border-radius: 3px; padding: 2px 0; cursor: pointer; font-size: 11px; }
#monitor button.tog.on { background: #ffd27a; border-color: #ffd27a; color: #000; font-weight: 600; }
#monitor .chips { display: flex; gap: 4px; flex-wrap: wrap; }
#monitor .chip { border: 1px solid rgba(255,255,255,0.18); background: none; color: #888; border-radius: 10px; padding: 1px 9px; cursor: pointer; font-size: 11px; }
#monitor .chip.on { color: #fff; }
#monitor .more { background: none; border: 0; color: #aaa; cursor: pointer; font-size: 11px; }
#monitor .settings { display: flex; gap: 12px; flex-wrap: wrap; padding: 4px 0 8px 32px; }
#monitor .node { background: rgba(255,255,255,0.04); border-radius: 4px; padding: 6px 8px; min-width: 220px; }
#monitor .node h4 { margin: 0 0 4px; font-size: 12px; font-weight: 600; color: #fff; display: flex; justify-content: space-between; gap: 8px; }
#monitor .node h4 small { font-weight: 400; color: #888; }
#monitor .node button { background: none; border: 0; color: #888; cursor: pointer; font-size: 11px; padding: 0; }
#monitor .row { display: grid; grid-template-columns: 92px 1fr 48px; gap: 6px; align-items: center; }
#monitor .row label { color: #aaa; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
#monitor .row output { text-align: right; color: #fff; font-variant-numeric: tabular-nums; }
#monitor .row input { width: 100%; accent-color: #ffd27a; }
#monitor [data-tip] { cursor: help; }
#monitor .row label[data-tip]::after, #monitor .master span[data-tip]::after { content: " ?"; color: #666; font-size: 10px; }
.monitor-tip { position: fixed; z-index: 10; max-width: 320px; padding: 7px 10px; border-radius: 5px; background: #1d1d20;
  border: 1px solid rgba(255,255,255,0.18); color: #eee; font: 12px/1.4 system-ui, sans-serif; pointer-events: none;
  box-shadow: 0 4px 16px rgba(0,0,0,0.5); }
.monitor-tip b { color: #ffd27a; }
`;

const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
const KIND_HELP: Record<string, string> = {
  intro: "the quiet opening",
  build: "the climb into a drop",
  drop: "the big hits",
  breakdown: "quiet stretches between drops",
  normal: "everything else that's playing",
  outro: "the ending",
};
const fmt = (t: number) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
const num = (v: Value | undefined, d = 0) => (typeof v === "number" ? v : Array.isArray(v) ? v[0] ?? d : d);

/** A display colour for a linear-RGB light colour: gamma-encoded, brightest channel at full. */
function lightCss(c: number[]): string {
  const m = Math.max(...c, 1e-6);
  return `rgb(${c.map((x) => Math.round(255 * Math.pow(Math.max(0, x / m), 1 / 2.2))).join(",")})`;
}

interface SetupInfo {
  node: GraphNode;
  targets: GraphNode[];
  color: string;
  label: string;
  levels: Float32Array; // sampled over the track at SAMPLE_DT
}
const SAMPLE_DT = 0.1;

export function createMonitor(o: MonitorOptions) {
  const { music, runtime } = o;
  const dur = music.a.duration;
  document.head.append(Object.assign(document.createElement("style"), { textContent: css }));
  const root = Object.assign(document.createElement("div"), { id: "monitor", hidden: true });
  document.body.append(root);

  // Pop-up explanations: hover anything with data-tip.
  const tipBox = Object.assign(document.createElement("div"), { className: "monitor-tip", hidden: true });
  document.body.append(tipBox);
  root.addEventListener("mouseover", (e) => {
    const el = (e.target as HTMLElement).closest<HTMLElement>("[data-tip]");
    if (!el || !el.dataset.tip) return void (tipBox.hidden = true);
    tipBox.innerHTML = `<b>${esc(el.dataset.tipTitle ?? el.textContent ?? "")}</b><br>${esc(el.dataset.tip)}`;
    tipBox.hidden = false;
    const r = el.getBoundingClientRect();
    const top = r.top - tipBox.offsetHeight - 6;
    tipBox.style.left = `${Math.min(innerWidth - tipBox.offsetWidth - 8, Math.max(8, r.left))}px`;
    tipBox.style.top = `${top > 8 ? top : r.bottom + 6}px`;
  });
  root.addEventListener("mouseleave", () => (tipBox.hidden = true));

  // --- Graph helpers ---------------------------------------------------------------------------
  const graph = () => runtime.graph;
  const nodeById = (id: string) => graph().nodes.find((n) => n.id === id);
  const upstream = (ids: string[]): Set<string> => {
    const seen = new Set<string>();
    const stack = [...ids];
    while (stack.length) {
      const id = stack.pop()!;
      if (seen.has(id)) continue;
      seen.add(id);
      for (const w of graph().wires) if (w.to[0] === id) stack.push(w.from[0]);
    }
    return seen;
  };
  const colorOf = (target: GraphNode): string => {
    if (target.type === "Sun") return SUN;
    // The colour actually wired in (as last evaluated), else the node's own.
    const w = graph().wires.find((w) => w.to[0] === target.id && w.to[1] === "color");
    const wired = w ? runtime.values.get(w.from[0])?.[w.from[1]] : undefined;
    const c = (wired ?? target.params?.color ?? NODE_TYPES[target.type]?.inputs.find((i) => i.name === "color")?.default) as number[] | undefined;
    return Array.isArray(c) ? lightCss(c) : "#cccccc";
  };
  let setups: SetupInfo[] = [];
  const findSetups = () => {
    setups = graph()
      .nodes.filter((n) => n.type === "Setup")
      .map((node) => {
        const targets = graph().wires.filter((w) => w.from[0] === node.id).map((w) => nodeById(w.to[0])).filter((n): n is GraphNode => !!n);
        return {
          node,
          targets,
          color: targets[0] ? colorOf(targets[0]) : "#ccc",
          label: (node.name ?? node.id).replace(/^Setup:\s*/i, ""),
          levels: new Float32Array(0),
        };
      });
  };

  // --- Header ----------------------------------------------------------------------------------
  const head = document.createElement("div");
  head.className = "head";
  const keys = Object.entries(MOOD_COLOR).map(([k, c]) => `<span class="key"><i style="background:${c}"></i>${k}</span>`).join("");
  head.innerHTML =
    `<b>Music monitor</b>${keys}<span class="key"><i style="background:#fff"></i>energy</span>` +
    `<span class="key"><i style="background:${TENSION}"></i>tension</span><span class="key"><i style="background:${SUN}"></i>sun height</span>` +
    `<span>▼ drop</span><span class="status"></span><button class="save">Save</button>`;
  const status = head.querySelector<HTMLElement>(".status")!;
  head.querySelector<HTMLButtonElement>(".save")!.onclick = async () => (status.textContent = await o.save());
  root.append(head);

  // Master sliders: the few settings you reach for most, always in view.
  const MASTER: { type: string; param: string; label: string; min: number; max: number; step: number }[] = [
    { type: "Sun", param: "gain", label: "sun", min: 0, max: 10, step: 0.05 },
    { type: "Sun", param: "azimuth", label: "sun direction", min: -60, max: 60, step: 0.5 },
    { type: "Sun", param: "bounce", label: "bounce", min: 0, max: 2, step: 0.05 },
    { type: "Sun", param: "floorLight", label: "floor light", min: 0, max: 1, step: 0.05 },
    { type: "Tone", param: "cap", label: "brightness cap", min: 0.2, max: 10, step: 0.05 },
    { type: "NookLights", param: "intensity", label: "nook lights", min: 0, max: 20, step: 0.05 },
    { type: "Sky", param: "brightness", label: "sky", min: 0, max: 20, step: 0.1 },
    { type: "Sync", param: "lead", label: "visual lead ms", min: -200, max: 200, step: 1 },
  ];
  const master = document.createElement("div");
  master.className = "master";
  root.append(master);
  const buildMaster = () => {
    master.innerHTML = "";
    for (const m of MASTER) {
      const node = runtime.graph.nodes.find((n) => n.type === m.type);
      if (!node || runtime.graph.wires.some((w) => w.to[0] === node.id && w.to[1] === m.param)) continue;
      const def = NODE_TYPES[m.type].inputs.find((i) => i.name === m.param);
      const value = () => num(node.params?.[m.param], (def?.default as number) ?? 0);
      const row = document.createElement("label");
      row.innerHTML = `<span data-tip="${esc(inputHelp(m.type, m.param))}">${m.label}</span><input type="range" min="${m.min}" max="${m.max}" step="${m.step}"><output></output>`;
      const r = row.querySelector("input")!;
      const out = row.querySelector("output")!;
      r.value = String(value());
      out.textContent = String(Math.round(value() * 100) / 100);
      r.oninput = () => {
        node.params = { ...(node.params ?? {}), [m.param]: Number(r.value) };
        out.textContent = String(Math.round(Number(r.value) * 100) / 100);
        runtime.load(runtime.graph);
        o.edited(runtime.graph);
      };
      master.append(row);
    }
  };

  // --- Timeline --------------------------------------------------------------------------------
  const tl = document.createElement("div");
  tl.className = "tl";
  const canvas = document.createElement("canvas");
  const tip = Object.assign(document.createElement("div"), { className: "tip", hidden: true });
  tl.append(canvas, tip);
  root.append(tl);
  const ctx = canvas.getContext("2d")!;
  let bg: HTMLCanvasElement | null = null;
  let sun: { h: number; on: boolean }[] = [];

  // Run the graph over the whole track: the sun's height and every setup's level.
  const sample = () => {
    const n = Math.ceil(dur / SAMPLE_DT);
    const heights: { h: number; on: boolean }[] = [];
    for (const s of setups) s.levels = new Float32Array(n);
    for (let k = 0; k < n; k++) {
      const s = runtime.evaluate(k * SAMPLE_DT).sun;
      heights.push({ h: s ? s.arc : 0, on: !!s?.on });
      for (const st of setups) st.levels[k] = num(runtime.values.get(st.node.id)?.level);
    }
    runtime.evaluate(o.time()); // leave live values at the current time
    const lo = Math.min(...heights.map((s) => s.h));
    const hi = Math.max(...heights.map((s) => s.h));
    sun = heights.map((s) => ({ ...s, h: hi > lo ? (s.h - lo) / (hi - lo) : 0 }));
    bg = null;
  };

  const L = { band: 20, top: 26, curves: 110, lane: 13, axis: 14 };
  const height = () => L.top + L.curves + setups.length * L.lane + L.axis + 4;
  const drawBackground = () => {
    const dpr = devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = height();
    canvas.style.height = `${h}px`;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    bg = document.createElement("canvas");
    bg.width = canvas.width;
    bg.height = canvas.height;
    const g = bg.getContext("2d")!;
    g.scale(dpr, dpr);
    const x = (t: number) => (t / dur) * w;
    const y0 = L.top;
    const y1 = L.top + L.curves;
    const yv = (v: number) => y1 - v * (y1 - y0 - 4);
    const lanes0 = y1 + 4;

    // Sections: a named band on top and a faint wash down through everything.
    g.font = "11px system-ui, sans-serif";
    for (const s of music.a.sections) {
      const a = x(s.start);
      const b = x(s.end);
      g.fillStyle = MOOD_COLOR[mood(s.kind as SectionKind)];
      g.globalAlpha = 0.85;
      g.fillRect(a + 1, 0, Math.max(1, b - a - 2), L.band);
      g.globalAlpha = 0.1;
      g.fillRect(a + 1, y0, Math.max(1, b - a - 2), h - L.axis - y0);
      g.globalAlpha = 1;
      if (b - a > 40) {
        g.fillStyle = "#fff";
        g.fillText(s.kind, a + 5, 14, b - a - 8);
      }
    }
    // Phrase lines.
    g.strokeStyle = "rgba(255,255,255,0.10)";
    g.lineWidth = 1;
    for (const p of music.a.phrases?.starts ?? []) {
      g.beginPath();
      g.moveTo(Math.round(x(p)) + 0.5, y0);
      g.lineTo(Math.round(x(p)) + 0.5, h - L.axis);
      g.stroke();
    }
    // Energy (area), tension (line), sun height (line, faded where the sun is off).
    const step = dur / w;
    g.fillStyle = ENERGY;
    g.beginPath();
    g.moveTo(0, y1);
    for (let px = 0; px <= w; px++) g.lineTo(px, yv(music.barEnergy(px * step)));
    g.lineTo(w, y1);
    g.fill();
    g.strokeStyle = TENSION;
    g.lineWidth = 2;
    g.beginPath();
    for (let px = 0; px <= w; px++) g.lineTo(px, yv(music.tension(px * step)));
    g.stroke();
    g.strokeStyle = SUN;
    for (let i = 1; i < sun.length; i++) {
      g.globalAlpha = sun[i].on ? 1 : 0.25;
      g.beginPath();
      g.moveTo(x((i - 1) * SAMPLE_DT), yv(sun[i - 1].h));
      g.lineTo(x(i * SAMPLE_DT), yv(sun[i].h));
      g.stroke();
    }
    g.globalAlpha = 1;
    // Drops.
    for (const d of music.drops()) {
      const px = x(d.t);
      const s = 4 + 5 * d.confidence;
      g.fillStyle = "#fff";
      g.beginPath();
      g.moveTo(px - s, y0);
      g.lineTo(px + s, y0);
      g.lineTo(px, y0 + s * 1.4);
      g.fill();
    }
    // Setup lanes: filled where the setup is on (brighter = higher level).
    setups.forEach((s, i) => {
      const y = lanes0 + i * L.lane;
      g.fillStyle = "rgba(255,255,255,0.04)";
      g.fillRect(0, y, w, L.lane - 3);
      g.fillStyle = s.color;
      const n = s.levels.length;
      for (let px = 0; px < w; px++) {
        const v = s.levels[Math.min(n - 1, Math.floor((px * step) / SAMPLE_DT))] ?? 0;
        if (v <= 0.01) continue;
        g.globalAlpha = 0.25 + 0.75 * v;
        g.fillRect(px, y, 1, L.lane - 3);
      }
      g.globalAlpha = 1;
      g.fillStyle = "#000";
      g.globalAlpha = 0.55;
      const label = s.label + (num(s.node.params?.on, 1) < 0.5 ? " (off)" : "");
      g.fillRect(2, y, g.measureText(label).width + 8, L.lane - 3);
      g.globalAlpha = 1;
      g.fillStyle = "#ddd";
      g.font = "10px system-ui, sans-serif";
      g.fillText(label, 6, y + L.lane - 5);
    });
    // Time axis.
    g.fillStyle = "#888";
    g.font = "10px system-ui, sans-serif";
    for (let t = 0; t < dur; t += 30) g.fillText(fmt(t), x(t) + 2, h - 2);
  };

  const drawFrame = (t: number) => {
    if (!bg || canvas.clientWidth * (devicePixelRatio || 1) !== canvas.width) drawBackground();
    const dpr = devicePixelRatio || 1;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(bg!, 0, 0);
    ctx.scale(dpr, dpr);
    const px = (t / dur) * canvas.clientWidth;
    ctx.fillStyle = "#fff";
    ctx.fillRect(px - 1, 0, 2, height() - L.axis);
    ctx.font = "600 11px system-ui, sans-serif";
    ctx.fillText(fmt(t), Math.min(px + 4, canvas.clientWidth - 30), L.top + 12);
  };

  const tAt = (e: MouseEvent) => Math.min(dur, Math.max(0, ((e.clientX - canvas.getBoundingClientRect().left) / canvas.clientWidth) * dur));
  let dragging = false;
  canvas.onpointerdown = (e) => ((dragging = true), canvas.setPointerCapture(e.pointerId), o.seek(tAt(e)));
  canvas.onpointerup = () => (dragging = false);
  canvas.onpointermove = (e) => {
    const t = tAt(e);
    if (dragging) o.seek(t);
    const on = setups.filter((s) => (s.levels[Math.floor(t / SAMPLE_DT)] ?? 0) > 0.01).map((s) => s.label);
    tip.hidden = false;
    tip.style.left = `${e.clientX - canvas.getBoundingClientRect().left}px`;
    tip.textContent = `${fmt(t)} · ${music.section(t).kind} · on: ${on.join(", ") || "nothing"}`;
  };
  canvas.onpointerleave = () => (tip.hidden = true);
  addEventListener("resize", () => (bg = null));

  // --- Meters ----------------------------------------------------------------------------------
  const metersEl = document.createElement("div");
  metersEl.className = "meters";
  root.append(metersEl);
  type Meter = { label: string; color: string; value: (t: number) => number; el?: HTMLElement; peakEl?: HTMLElement; peak?: number };
  let meters: Meter[] = [];
  const spurtLift = () => {
    const n = graph().nodes.find((n) => n.type === "Spurts");
    return n ? num(runtime.values.get(n.id)?.lift) / (num(n.params?.max, 75) || 75) : 0;
  };
  const buildMeters = () => {
    meters = [
      { label: "kick", color: "#3987e5", value: (t) => music.sub(t) },
      { label: "bass", color: "#3987e5", value: (t) => music.bass(t) },
      { label: "hats", color: "#3987e5", value: (t) => music.flicker(t) },
      { label: "pump", color: "#3987e5", value: (t) => music.pump(t) },
      { label: "energy", color: "#cfcfcf", value: (t) => music.barEnergy(t) },
      { label: "tension", color: TENSION, value: (t) => music.tension(t) },
      { label: "", color: "", value: () => 0 },
      { label: "sun\nlift", color: SUN, value: spurtLift },
      ...setups.map((s) => ({ label: s.label.split(" ").slice(-1)[0], color: s.color, value: () => num(runtime.values.get(s.node.id)?.level) })),
    ];
    metersEl.innerHTML = "";
    for (const m of meters) {
      if (!m.label) {
        metersEl.append(Object.assign(document.createElement("div"), { className: "sep" }));
        continue;
      }
      const wrap = document.createElement("div");
      wrap.className = "meter";
      wrap.title = m.label;
      wrap.innerHTML = `<div class="bar"><div class="fill" style="background:${m.color}"></div><div class="peak"></div></div><span>${m.label.replace("\n", "<br>")}</span>`;
      m.el = wrap.querySelector<HTMLElement>(".fill")!;
      m.peakEl = wrap.querySelector<HTMLElement>(".peak")!;
      m.peak = 0;
      metersEl.append(wrap);
    }
  };
  const drawMeters = (t: number) => {
    for (const m of meters) {
      if (!m.el) continue;
      const v = Math.min(1, Math.max(0, m.value(t)));
      m.el.style.height = `${v * 100}%`;
      m.peak = Math.max(v, (m.peak ?? 0) - 0.008);
      m.peakEl!.style.bottom = `calc(${m.peak * 100}% - 2px)`;
    }
  };

  // --- Setups ----------------------------------------------------------------------------------
  const list = document.createElement("div");
  list.className = "setups";
  root.append(list);
  let open: string | null = null;
  let resampleTimer = 0;
  const changed = (resampleNow = false) => {
    runtime.load(graph());
    o.edited(graph());
    clearTimeout(resampleTimer);
    resampleTimer = window.setTimeout(sample, resampleNow ? 0 : 150);
  };
  const setParam = (node: GraphNode, name: string, v: number) => (node.params = { ...(node.params ?? {}), [name]: v });

  const sliders = (node: GraphNode, sharedWith: string[]): HTMLElement | null => {
    const def = NODE_TYPES[node.type];
    if (!def || SKIP_TYPES.has(node.type)) return null;
    const wired = new Set(graph().wires.filter((w) => w.to[0] === node.id).map((w) => w.to[1]));
    const inputs = def.inputs.filter((i) => typeof i.default === "number" && !SKIP_INPUTS.has(i.name) && !wired.has(i.name));
    if (!inputs.length) return null;
    const box = document.createElement("div");
    box.className = "node";
    const h = document.createElement("h4");
    h.innerHTML = `<span data-tip="${esc(nodeHelp(node.type))}" data-tip-title="${esc(node.name ?? node.type)}">${node.name ?? node.type} <small>${sharedWith.length ? "also in " + sharedWith.join(", ") : ""}</small></span>`;
    const reset = Object.assign(document.createElement("button"), { textContent: "reset", title: "back to the node type's defaults" });
    h.append(reset);
    box.append(h);
    const shows: (() => void)[] = [];
    for (const inp of inputs) {
      const value = () => num(node.params?.[inp.name], inp.default as number);
      const v0 = value();
      const min = inp.min ?? Math.min(0, v0 * 3);
      const max = inp.max ?? Math.max(1, Math.abs(v0) * 3);
      const row = document.createElement("div");
      row.className = "row";
      row.innerHTML = `<label data-tip="${esc(inputHelp(node.type, inp.name))}">${inp.name}</label><input type="range" min="${min}" max="${max}" step="${inp.step ?? 0.01}"><output></output>`;
      const range = row.querySelector("input")!;
      const out = row.querySelector("output")!;
      const show = () => ((range.value = String(value())), (out.textContent = String(Math.round(value() * 1000) / 1000)));
      show();
      shows.push(show);
      range.oninput = () => {
        setParam(node, inp.name, Number(range.value));
        out.textContent = String(Math.round(Number(range.value) * 1000) / 1000);
        changed();
      };
      box.append(row);
    }
    reset.onclick = () => {
      for (const inp of inputs) if (node.params) delete node.params[inp.name];
      shows.forEach((f) => f());
      changed();
    };
    return box;
  };

  const buildList = () => {
    list.innerHTML = "";
    if (!setups.length) {
      list.textContent = "No Setup nodes in this graph.";
      return;
    }
    // Which setups each node feeds, for the "also in" note.
    const feeds = new Map<string, string[]>();
    for (const s of setups) for (const id of upstream(s.targets.map((t) => t.id))) feeds.set(id, [...(feeds.get(id) ?? []), s.label]);
    for (const s of setups) {
      const p = s.node.params ?? {};
      const isOn = num(p.on, 1) > 0.5;
      const row = document.createElement("div");
      row.className = `setup${isOn ? "" : " off"}`;
      row.innerHTML = `<i class="swatch" style="background:${s.color}"></i><span class="name">${s.label}</span>`;
      const tog = Object.assign(document.createElement("button"), { className: `tog${isOn ? " on" : ""}`, textContent: isOn ? "ON" : "OFF" });
      tog.onclick = () => (setParam(s.node, "on", isOn ? 0 : 1), changed(true), buildList(), buildMeters());
      const chips = document.createElement("div");
      chips.className = "chips";
      for (const k of KINDS) {
        const active = num(p[k], 1) > 0.5;
        const chip = Object.assign(document.createElement("button"), { className: `chip${active ? " on" : ""}`, textContent: k });
        if (active) chip.style.background = MOOD_COLOR[mood(k)];
        chip.dataset.tip = `Click to ${active ? "stop" : "start"} playing this setup in ${k} sections (${KIND_HELP[k]}).`;
        chip.onclick = () => (setParam(s.node, k, active ? 0 : 1), changed(true), buildList());
        chips.append(chip);
      }
      const more = Object.assign(document.createElement("button"), { className: "more", textContent: open === s.node.id ? "settings ▾" : "settings ▸" });
      more.onclick = () => ((open = open === s.node.id ? null : s.node.id), buildList());
      row.append(tog, chips, more);
      list.append(row);
      if (open === s.node.id) {
        const panel = document.createElement("div");
        panel.className = "settings";
        // The setup's own fades first, then everything that feeds its output, outputs first.
        const fades = document.createElement("div");
        fades.className = "node";
        fades.innerHTML = `<h4><span>Fades</span></h4>`;
        for (const name of ["fadeIn", "fadeOut"]) {
          const inp = NODE_TYPES.Setup.inputs.find((i) => i.name === name)!;
          const row2 = document.createElement("div");
          row2.className = "row";
          const value = () => num(s.node.params?.[name], inp.default as number);
          row2.innerHTML = `<label data-tip="${esc(inputHelp("Setup", name))}">${name}</label><input type="range" min="${inp.min}" max="${inp.max}" step="${inp.step}"><output>${value()}</output>`;
          const r = row2.querySelector("input")!;
          r.value = String(value());
          r.oninput = () => (setParam(s.node, name, Number(r.value)), (row2.querySelector("output")!.textContent = r.value), changed());
          fades.append(row2);
        }
        panel.append(fades);
        const ids = [...upstream(s.targets.map((t) => t.id))];
        ids.sort((a, b) => Number(!s.targets.some((t) => t.id === a)) - Number(!s.targets.some((t) => t.id === b)));
        for (const id of ids) {
          const node = nodeById(id);
          if (!node) continue;
          const box = sliders(node, (feeds.get(id) ?? []).filter((l) => l !== s.label));
          if (box) panel.append(box);
        }
        list.append(panel);
      }
    }
  };

  // --- Public ----------------------------------------------------------------------------------
  let built = false;
  let hudWasHidden = true;
  const rebuild = () => {
    buildMaster();
    findSetups();
    sample();
    buildMeters();
    buildList();
  };
  const show = (on: boolean) => {
    root.hidden = !on;
    const hud = document.querySelector<HTMLElement>("#hud");
    if (hud && on) (hudWasHidden = hud.hidden), (hud.hidden = true);
    else if (hud) hud.hidden = hudWasHidden;
    if (on && !built) (built = true), rebuild();
  };
  return {
    toggle: () => show(root.hidden),
    get visible() {
      return !root.hidden;
    },
    /** Call every frame. */
    update(t: number) {
      if (root.hidden) return;
      drawFrame(t);
      drawMeters(t);
    },
    /** The graph was replaced from elsewhere (the editor). */
    graphChanged() {
      if (built) rebuild();
    },
    /** A message in the header (e.g. the graph changed on disk). */
    setStatus(text: string) {
      status.textContent = text;
    },
  };
}
