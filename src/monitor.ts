// Music monitor: a panel over the display (G) for seeing and tuning how the visuals respond to the music.
//
//   Timeline   the whole track: sections (coloured, named), drops (triangles, bigger = surer), phrase
//              lines, the music's energy and tension, and what the graph makes the sun do (its height,
//              faded where the sun is off). Click or drag to seek; hover for details.
//   Meters     live levels, like an equalizer: what the music is doing now, and the sun's response.
//   Response   sliders for every setting of the graph's music nodes (Spurts, Gate, DropHit, ...).
//              Changes apply at once, update the timeline and the editor, and are saved with Save.

import { mood, type Music, type SectionKind } from "./music.ts";
import { NODE_TYPES, type Value } from "./graph/nodes.ts";
import type { Graph, GraphRuntime } from "./graph/runtime.ts";

export interface MonitorOptions {
  music: Music;
  runtime: GraphRuntime;
  time: () => number; // current track time, seconds
  seek: (t: number) => void;
  edited: (graph: Graph) => void; // graph changed here: pass it on (editor)
  save: () => Promise<string>; // returns a status line
}

const MOOD_COLOR: Record<string, string> = { quiet: "#3987e5", build: "#d95926", drop: "#e66767", normal: "#6f6e69" };
const TENSION = "#e87ba4";
const SUN = "#ffd27a";
const ENERGY = "rgba(255,255,255,0.22)";

const css = `
#monitor { position: fixed; left: 0; right: 0; bottom: 0; padding: 10px 14px 12px; background: rgba(8,8,10,0.95);
  color: #ddd; font: 12px/1.35 system-ui, sans-serif; display: grid; grid-template-columns: 1fr auto; gap: 8px 16px;
  border-top: 1px solid rgba(255,255,255,0.08); max-height: 52vh; overflow: auto; }
#monitor[hidden] { display: none; }
#monitor .tl { grid-column: 1; grid-row: 2; position: relative; align-self: end; }
#monitor .tl canvas { width: 100%; height: 150px; display: block; cursor: pointer; }
#monitor .tip { position: absolute; top: 0; pointer-events: none; background: rgba(0,0,0,0.85); padding: 3px 6px;
  border-radius: 3px; white-space: nowrap; font-size: 11px; transform: translateX(-50%); }
#monitor .meters { grid-column: 2; grid-row: 2; display: flex; gap: 6px; align-items: flex-end; }
#monitor .meter { width: 34px; display: flex; flex-direction: column; align-items: center; gap: 4px; }
#monitor .bar { position: relative; width: 16px; height: 118px; background: rgba(255,255,255,0.07); border-radius: 3px; overflow: hidden; }
#monitor .fill { position: absolute; left: 0; right: 0; bottom: 0; border-radius: 3px 3px 0 0; }
#monitor .peak { position: absolute; left: 0; right: 0; height: 2px; background: #fff; opacity: 0.7; }
#monitor .meter span { font-size: 10px; color: #aaa; text-align: center; line-height: 1.1; }
#monitor .sep { width: 1px; height: 130px; background: rgba(255,255,255,0.12); margin: 0 4px; }
#monitor .resp { grid-column: 1 / -1; grid-row: 3; display: flex; gap: 14px; flex-wrap: wrap; align-items: flex-start; }
#monitor .node { background: rgba(255,255,255,0.04); border-radius: 4px; padding: 6px 8px; min-width: 210px; }
#monitor .node h4 { margin: 0 0 4px; font-size: 12px; font-weight: 600; color: #fff; display: flex; justify-content: space-between; gap: 8px; }
#monitor .node h4 small { font-weight: 400; color: #888; }
#monitor .node button { background: none; border: 0; color: #888; cursor: pointer; font-size: 11px; padding: 0; }
#monitor .row { display: grid; grid-template-columns: 92px 1fr 44px; gap: 6px; align-items: center; }
#monitor .row label { color: #aaa; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
#monitor .row output { text-align: right; color: #fff; font-variant-numeric: tabular-nums; }
#monitor .row input { width: 100%; accent-color: #ffd27a; }
#monitor .head { grid-column: 1 / -1; grid-row: 1; display: flex; gap: 14px; align-items: center; color: #aaa; }
#monitor .head b { color: #fff; }
#monitor .head .save { margin-left: auto; background: #ffd27a; color: #000; border: 0; border-radius: 3px; padding: 3px 12px; cursor: pointer; font-weight: 600; }
#monitor .key { display: inline-flex; align-items: center; gap: 4px; }
#monitor .key i { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
`;

const fmt = (t: number) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;

export function createMonitor(o: MonitorOptions) {
  const { music, runtime } = o;
  const dur = music.a.duration;
  document.head.append(Object.assign(document.createElement("style"), { textContent: css }));
  const root = Object.assign(document.createElement("div"), { id: "monitor", hidden: true });
  document.body.append(root);

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

  // --- Timeline --------------------------------------------------------------------------------
  const tl = document.createElement("div");
  tl.className = "tl";
  const canvas = document.createElement("canvas");
  const tip = Object.assign(document.createElement("div"), { className: "tip", hidden: true });
  tl.append(canvas, tip);
  root.append(tl);
  const ctx = canvas.getContext("2d")!;
  let bg: HTMLCanvasElement | null = null;
  let sun: { t: number; h: number; on: boolean }[] = [];

  const sampleSun = () => {
    const out: typeof sun = [];
    for (let t = 0; t < dur; t += 0.1) {
      const s = runtime.evaluate(t).sun;
      out.push({ t, h: s ? s.arc : 0, on: !!s?.on });
    }
    runtime.evaluate(o.time()); // leave live values at the current time
    const lo = Math.min(...out.map((s) => s.h));
    const hi = Math.max(...out.map((s) => s.h));
    sun = out.map((s) => ({ ...s, h: hi > lo ? (s.h - lo) / (hi - lo) : 0 }));
  };

  const layout = { band: 20, top: 26, axis: 14 };
  const drawBackground = () => {
    const dpr = devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    bg = document.createElement("canvas");
    bg.width = canvas.width;
    bg.height = canvas.height;
    const g = bg.getContext("2d")!;
    g.scale(dpr, dpr);
    const x = (t: number) => (t / dur) * w;
    const y0 = layout.top;
    const y1 = h - layout.axis;
    const yv = (v: number) => y1 - v * (y1 - y0 - 4);

    // Sections.
    g.font = "11px system-ui, sans-serif";
    for (const s of music.a.sections) {
      const a = x(s.start);
      const b = x(s.end);
      g.fillStyle = MOOD_COLOR[mood(s.kind as SectionKind)];
      g.globalAlpha = 0.85;
      g.fillRect(a + 1, 0, Math.max(1, b - a - 2), layout.band);
      g.globalAlpha = 0.1;
      g.fillRect(a + 1, y0, Math.max(1, b - a - 2), y1 - y0);
      g.globalAlpha = 1;
      if (b - a > 40) {
        g.fillStyle = "#fff";
        g.fillText(s.kind, a + 5, 14, b - a - 8);
      }
    }
    // Phrase lines.
    g.strokeStyle = "rgba(255,255,255,0.10)";
    g.lineWidth = 1;
    const starts = music.a.phrases?.starts ?? [];
    for (const p of starts) {
      g.beginPath();
      g.moveTo(Math.round(x(p)) + 0.5, y0);
      g.lineTo(Math.round(x(p)) + 0.5, y1);
      g.stroke();
    }
    // Energy (area), tension (line), sun height (line, faded where off).
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
    for (let i = 1; i < sun.length; i++) {
      g.strokeStyle = SUN;
      g.globalAlpha = sun[i].on ? 1 : 0.25;
      g.beginPath();
      g.moveTo(x(sun[i - 1].t), yv(sun[i - 1].h));
      g.lineTo(x(sun[i].t), yv(sun[i].h));
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
    ctx.fillRect(px - 1, 0, 2, canvas.clientHeight - layout.axis);
    ctx.font = "600 11px system-ui, sans-serif";
    const label = fmt(t);
    const lx = Math.min(px + 4, canvas.clientWidth - 30);
    ctx.fillText(label, lx, layout.top + 12);
  };

  const tAt = (e: MouseEvent) => Math.min(dur, Math.max(0, ((e.clientX - canvas.getBoundingClientRect().left) / canvas.clientWidth) * dur));
  let dragging = false;
  canvas.onpointerdown = (e) => ((dragging = true), canvas.setPointerCapture(e.pointerId), o.seek(tAt(e)));
  canvas.onpointerup = () => (dragging = false);
  canvas.onpointermove = (e) => {
    const t = tAt(e);
    if (dragging) o.seek(t);
    const s = music.section(t);
    tip.hidden = false;
    tip.style.left = `${e.clientX - canvas.getBoundingClientRect().left}px`;
    tip.textContent = `${fmt(t)} · ${s.kind} · energy ${music.barEnergy(t).toFixed(2)} · tension ${music.tension(t).toFixed(2)}`;
  };
  canvas.onpointerleave = () => (tip.hidden = true);
  addEventListener("resize", () => (bg = null));

  // --- Meters ----------------------------------------------------------------------------------
  const metersEl = document.createElement("div");
  metersEl.className = "meters";
  root.append(metersEl);
  const firstOfType = (type: string) => runtime.graph.nodes.find((n) => n.type === type)?.id;
  const nodeOut = (type: string, out: string): number => {
    const id = firstOfType(type);
    const v = id ? runtime.values.get(id)?.[out] : undefined;
    return typeof v === "number" ? v : 0;
  };
  const spurtMax = () => {
    const n = runtime.graph.nodes.find((n) => n.type === "Spurts");
    return Number(n?.params?.max ?? 75) || 75;
  };
  const meters: { label: string; color: string; value: (t: number) => number; el?: HTMLElement; peakEl?: HTMLElement; peak?: number }[] = [
    { label: "kick", color: "#3987e5", value: (t) => music.sub(t) },
    { label: "bass", color: "#3987e5", value: (t) => music.bass(t) },
    { label: "hats", color: "#3987e5", value: (t) => music.flicker(t) },
    { label: "pump", color: "#3987e5", value: (t) => music.pump(t) },
    { label: "inten-\nsity", color: "#cfcfcf", value: (t) => music.intensity(t) },
    { label: "energy", color: "#cfcfcf", value: (t) => music.barEnergy(t) },
    { label: "tension", color: TENSION, value: (t) => music.tension(t) },
    { label: "drop\nhit", color: "#e66767", value: (t) => Math.exp(-music.sinceDrop(t) / 0.8) },
    { label: "", color: "", value: () => 0 },
    { label: "sun\nlift", color: SUN, value: () => nodeOut("Spurts", "lift") / spurtMax() },
    { label: "sun\non", color: SUN, value: () => nodeOut("Gate", "value") },
  ];
  for (const m of meters) {
    if (!m.label) {
      metersEl.append(Object.assign(document.createElement("div"), { className: "sep" }));
      continue;
    }
    const wrap = document.createElement("div");
    wrap.className = "meter";
    wrap.innerHTML = `<div class="bar"><div class="fill" style="background:${m.color}"></div><div class="peak"></div></div><span>${m.label.replace("\n", "<br>")}</span>`;
    m.el = wrap.querySelector<HTMLElement>(".fill")!;
    m.peakEl = wrap.querySelector<HTMLElement>(".peak")!;
    m.peak = 0;
    metersEl.append(wrap);
  }
  const drawMeters = (t: number) => {
    for (const m of meters) {
      if (!m.el) continue;
      const v = Math.min(1, Math.max(0, m.value(t)));
      m.el.style.height = `${v * 100}%`;
      m.peak = Math.max(v, (m.peak ?? 0) - 0.008);
      m.peakEl!.style.bottom = `calc(${m.peak * 100}% - 2px)`;
    }
  };

  // --- Response sliders ------------------------------------------------------------------------
  const resp = document.createElement("div");
  resp.className = "resp";
  root.append(resp);
  const ORDER = ["Spurts", "Gate", "DropHit", "Section", "Pump", "Beat", "Phrase", "Intensity"];
  let resampleTimer = 0;
  const changed = () => {
    o.edited(runtime.graph);
    clearTimeout(resampleTimer);
    resampleTimer = window.setTimeout(() => (sampleSun(), (bg = null)), 120);
  };
  const buildSliders = () => {
    resp.innerHTML = "";
    const nodes = runtime.graph.nodes
      .filter((n) => NODE_TYPES[n.type]?.category === "Music" && NODE_TYPES[n.type].inputs.some((i) => i.kind === "const"))
      .sort((a, b) => (ORDER.indexOf(a.type) + 99) % 99 - (ORDER.indexOf(b.type) + 99) % 99);
    for (const node of nodes) {
      const def = NODE_TYPES[node.type];
      const box = document.createElement("div");
      box.className = "node";
      const h = document.createElement("h4");
      h.innerHTML = `<span>${node.name ?? node.type} <small>${node.name && node.name !== node.type ? node.type : ""}</small></span>`;
      const reset = Object.assign(document.createElement("button"), { textContent: "reset", title: "back to the node type's defaults" });
      h.append(reset);
      box.append(h);
      const inputs: (() => void)[] = [];
      for (const inp of def.inputs.filter((i) => i.kind === "const")) {
        if (typeof inp.default !== "number") continue;
        const row = document.createElement("div");
        row.className = "row";
        const value = () => Number(node.params?.[inp.name] ?? inp.default);
        const min = inp.min ?? Math.min(0, value() * 2);
        const max = inp.max ?? Math.max(1, value() * 3);
        row.innerHTML = `<label title="${inp.doc ?? inp.name}">${inp.name}</label><input type="range" min="${min}" max="${max}" step="${inp.step ?? 0.01}"><output></output>`;
        const range = row.querySelector("input")!;
        const out = row.querySelector("output")!;
        const show = () => {
          range.value = String(value());
          out.textContent = String(Math.round(value() * 1000) / 1000);
        };
        show();
        inputs.push(show);
        range.oninput = () => {
          node.params = { ...(node.params ?? {}), [inp.name]: Number(range.value) as Value };
          out.textContent = String(Math.round(Number(range.value) * 1000) / 1000);
          runtime.load(runtime.graph);
          changed();
        };
        box.append(row);
      }
      reset.onclick = () => {
        for (const inp of def.inputs) if (inp.kind === "const" && node.params) delete node.params[inp.name];
        runtime.load(runtime.graph);
        inputs.forEach((f) => f());
        changed();
      };
      resp.append(box);
    }
  };

  // --- Public ----------------------------------------------------------------------------------
  let built = false;
  let hudWasHidden = true;
  const show = (on: boolean) => {
    root.hidden = !on;
    const hud = document.querySelector<HTMLElement>("#hud");
    if (hud && on) (hudWasHidden = hud.hidden), (hud.hidden = true);
    else if (hud) hud.hidden = hudWasHidden;
    if (on && !built) {
      built = true;
      sampleSun();
      buildSliders();
      bg = null;
    }
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
    /** The graph was replaced from elsewhere (the editor): rebuild sliders and the sun curve. */
    graphChanged() {
      if (!built) return;
      sampleSun();
      buildSliders();
      bg = null;
    },
  };
}
