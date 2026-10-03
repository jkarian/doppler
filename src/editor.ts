// Node graph editor. Edits apply live to the display page (same browser) over a BroadcastChannel;
// the display sends back each node's current values, shown on outputs and wired inputs.
// Save writes graphs/<name>.json through the dev server.

import { NODE_TYPES, type InputDef, type NodeDef, type Value } from "./graph/nodes.ts";
import type { Graph, GraphNode } from "./graph/runtime.ts";

const view = document.getElementById("view")!;
const world = document.getElementById("world")!;
const wiresSvg = document.getElementById("wires") as unknown as SVGSVGElement;
const menu = document.getElementById("menu")!;
const status = document.getElementById("status")!;
const nameInput = document.getElementById("name") as HTMLInputElement;

const params = new URLSearchParams(location.search);
nameInput.value = params.get("graph") ?? "default";

let graph: Graph = { version: 1, nodes: [], wires: [] };
let pan = { x: 40, y: 20 };
let zoom = 0.85;
let selectedNode: string | null = null;
let selectedWire: number | null = null;
let lastValues: Record<string, Record<string, Value>> = {};
let lastErrors: Record<string, string> = {};
let lastHeard = 0;
const elements = new Map<string, HTMLElement>(); // node id -> element

const channel = new BroadcastChannel("doppler");

// --- Sync ---------------------------------------------------------------------------------------
let sendTimer = 0;
const changed = () => {
  clearTimeout(sendTimer);
  sendTimer = window.setTimeout(() => channel.postMessage({ type: "graph", graph }), 30);
};

channel.onmessage = (e: MessageEvent) => {
  const msg = e.data;
  if (msg?.type === "values") {
    lastHeard = performance.now();
    lastValues = msg.values;
    lastErrors = msg.errors;
    status.innerHTML =
      `<span class="live">● live</span> ${msg.track ?? "no track"} · ${msg.playing ? "playing" : "paused"} · ${Number(msg.t).toFixed(2)} s`;
    updateLive();
  } else if (msg?.type === "graph" && msg.from === "display" && (msg.edit || !graph.nodes.length)) {
    // From the display: its graph when we open, or an edit made in its music monitor.
    setGraph(msg.graph);
  }
};
setInterval(() => {
  if (performance.now() - lastHeard > 1500) status.textContent = "display not open: edits are saved only when you press Save";
}, 1000);

async function loadFromDisk(): Promise<void> {
  const res = await fetch(`graphs/${nameInput.value}.json`, { cache: "no-store" });
  if (!res.ok) return alert(`graphs/${nameInput.value}.json not found`);
  setGraph(await res.json());
  changed();
}

function setGraph(g: Graph): void {
  graph = g;
  selectedNode = null;
  selectedWire = null;
  render();
}

// Ask an open display for its live graph; fall back to the file.
channel.postMessage({ type: "hello" });
setTimeout(() => {
  if (!graph.nodes.length) loadFromDisk();
}, 400);

document.getElementById("save")!.onclick = async () => {
  const name = nameInput.value.trim();
  const res = await fetch(`/graph?name=${encodeURIComponent(name)}`, { method: "POST", body: JSON.stringify(graph, null, 1) });
  status.textContent = res.ok ? `saved graphs/${name}.json` : `save failed: ${res.status}`;
};
document.getElementById("load")!.onclick = () => loadFromDisk();
document.getElementById("fit")!.onclick = () => fit();

// --- Rendering ----------------------------------------------------------------------------------
const def = (n: GraphNode): NodeDef | undefined => NODE_TYPES[n.type];
const wiredInto = (id: string, input: string) => graph.wires.findIndex((w) => w.to[0] === id && w.to[1] === input);
const fmt = (v: Value | undefined): string => {
  if (v === undefined) return "";
  if (typeof v === "number") return Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(3);
  if (Array.isArray(v)) return v.map((x) => x.toFixed(2)).join(" ");
  return String(v);
};

function render(): void {
  for (const el of elements.values()) el.remove();
  elements.clear();
  for (const n of graph.nodes) elements.set(n.id, renderNode(n));
  applyTransform();
  drawWires();
  updateLive();
}

function renderNode(n: GraphNode): HTMLElement {
  const d = def(n);
  const el = document.createElement("div");
  el.className = `node ${d?.category ?? ""}${n.id === selectedNode ? " selected" : ""}`;
  el.style.left = `${n.pos?.[0] ?? 0}px`;
  el.style.top = `${n.pos?.[1] ?? 0}px`;
  el.dataset.id = n.id;

  const head = document.createElement("div");
  head.className = "head";
  head.innerHTML = `<span class="title"></span><span class="type"></span><span class="lock" title="lock: approved, no edits">🔒</span>`;
  (head.querySelector(".title") as HTMLElement).textContent = n.name ?? n.type;
  (head.querySelector(".type") as HTMLElement).textContent = n.name && n.name !== n.type ? n.type : "";
  const lock = head.querySelector(".lock") as HTMLElement;
  lock.classList.toggle("on", !!n.locked);
  lock.onclick = (e) => {
    e.stopPropagation();
    n.locked = !n.locked;
    render();
    changed();
  };
  head.title = d?.doc ?? "";
  head.ondblclick = () => {
    if (n.locked) return;
    const name = prompt("Node name", n.name ?? n.type);
    if (name !== null) {
      n.name = name.trim() || undefined;
      render();
      changed();
    }
  };
  head.onpointerdown = (e) => startNodeDrag(e, n, el);
  el.appendChild(head);

  if (!d) {
    el.insertAdjacentHTML("beforeend", `<div class="err">unknown node type</div>`);
    world.appendChild(el);
    return el;
  }
  for (const inp of visibleInputs(n, d)) el.appendChild(renderInput(n, inp));
  for (const out of d.outputs) {
    const row = document.createElement("div");
    row.className = "row out";
    row.innerHTML = `<span class="live" data-out="${out}"></span><span class="name">${out}</span><span class="port out" data-port="out" data-name="${out}"></span>`;
    (row.querySelector(".port") as HTMLElement).onpointerdown = (e) => startWire(e, n.id, out);
    el.appendChild(row);
  }
  const err = document.createElement("div");
  err.className = "err";
  err.hidden = true;
  el.appendChild(err);
  el.onpointerdown = (e) => {
    if ((e.target as HTMLElement).closest("input,.port")) return;
    select(n.id, null);
  };
  world.appendChild(el);
  return el;
}

// Expression nodes show only the letters in use, plus the next free one.
function visibleInputs(n: GraphNode, d: NodeDef): InputDef[] {
  if (n.type !== "Expression") return d.inputs;
  const expr = String(n.params?.expr ?? "");
  const letters = "abcdef";
  const used = (k: string) => wiredInto(n.id, k) >= 0 || new RegExp(`\\b${k}\\b`).test(expr);
  let last = -1;
  for (let i = 0; i < letters.length; i++) if (used(letters[i])) last = i;
  const show = new Set(letters.slice(0, Math.min(letters.length, last + 2)));
  return d.inputs.filter((inp) => !letters.includes(inp.name) || show.has(inp.name));
}

function renderInput(n: GraphNode, inp: InputDef): HTMLElement {
  const row = document.createElement("div");
  const isConst = inp.kind === "const";
  row.className = `row ${isConst ? "const" : "in"}`;
  row.title = inp.doc ?? "";
  const value = n.params?.[inp.name] ?? inp.default;
  const wired = !isConst && wiredInto(n.id, inp.name) >= 0;
  if (!isConst) {
    const port = document.createElement("span");
    port.className = `port in${wired ? " wired" : ""}`;
    port.dataset.port = "in";
    port.dataset.name = inp.name;
    row.appendChild(port);
  }
  const label = document.createElement("span");
  label.className = "name";
  label.textContent = inp.name;
  row.appendChild(label);

  const set = (v: Value) => {
    if (n.locked) return;
    n.params = { ...(n.params ?? {}), [inp.name]: v };
    changed();
  };
  if (wired) {
    const live = document.createElement("span");
    live.className = "live";
    live.dataset.in = inp.name;
    row.appendChild(live);
  } else if (typeof inp.default === "string") {
    const field = document.createElement("input");
    field.type = "text";
    field.className = inp.name === "expr" ? "expr" : "";
    field.value = String(value);
    field.disabled = !!n.locked;
    field.onchange = () => {
      set(field.value);
      render();
    };
    row.appendChild(field);
  } else if (Array.isArray(inp.default)) {
    const sw = document.createElement("span");
    sw.className = "swatch";
    const c = (Array.isArray(value) ? value : inp.default).map((x) => Math.round(Math.min(1, x) ** (1 / 2.2) * 255));
    sw.style.background = `rgb(${c.join(",")})`;
    row.appendChild(sw);
  } else {
    const field = document.createElement("input");
    field.type = "number";
    field.step = String(inp.step ?? 0.01);
    field.value = String(value);
    field.disabled = !!n.locked;
    field.oninput = () => field.value !== "" && set(Number(field.value));
    row.appendChild(field);
    // Scrub: drag the name sideways.
    label.onpointerdown = (e) => {
      if (n.locked) return;
      e.preventDefault();
      label.setPointerCapture(e.pointerId);
      const x0 = e.clientX;
      const v0 = Number(field.value) || 0;
      const step = inp.step ?? 0.01;
      label.onpointermove = (m) => {
        let v = v0 + Math.round((m.clientX - x0) / 4) * step * (m.shiftKey ? 10 : 1);
        if (inp.min !== undefined) v = Math.max(inp.min, v);
        if (inp.max !== undefined) v = Math.min(inp.max, v);
        v = Math.round(v / step) * step;
        field.value = String(Number(v.toFixed(6)));
        set(Number(field.value));
      };
      label.onpointerup = () => (label.onpointermove = null);
    };
  }
  return row;
}

function updateLive(): void {
  for (const n of graph.nodes) {
    const el = elements.get(n.id);
    if (!el) continue;
    const vals = lastValues[n.id] ?? {};
    for (const span of el.querySelectorAll<HTMLElement>("[data-out]")) span.textContent = fmt(vals[span.dataset.out!]);
    for (const span of el.querySelectorAll<HTMLElement>("[data-in]")) {
      const w = graph.wires[wiredInto(n.id, span.dataset.in!)];
      span.textContent = w ? fmt(lastValues[w.from[0]]?.[w.from[1]]) : "";
    }
    const err = lastErrors[n.id];
    el.classList.toggle("error", !!err);
    const errEl = el.querySelector<HTMLElement>(".err");
    if (errEl) {
      errEl.hidden = !err;
      errEl.textContent = err ?? "";
    }
  }
}

// --- Wires --------------------------------------------------------------------------------------
function portPos(id: string, kind: "in" | "out", name: string): [number, number] | null {
  const el = elements.get(id);
  const port = el?.querySelector<HTMLElement>(`.port.${kind}[data-name="${name}"]`);
  if (!el || !port) return null;
  const n = graph.nodes.find((x) => x.id === id)!;
  return [(n.pos?.[0] ?? 0) + port.offsetLeft + 6, (n.pos?.[1] ?? 0) + port.offsetTop + port.parentElement!.offsetTop + 2];
}

const curve = (a: [number, number], b: [number, number]) => {
  const dx = Math.max(40, Math.abs(b[0] - a[0]) * 0.5);
  return `M${a[0]},${a[1]} C${a[0] + dx},${a[1]} ${b[0] - dx},${b[1]} ${b[0]},${b[1]}`;
};

function drawWires(pending?: { from: [number, number]; to: [number, number] }): void {
  let html = "";
  graph.wires.forEach((w, i) => {
    const a = portPos(w.from[0], "out", w.from[1]);
    const b = portPos(w.to[0], "in", w.to[1]);
    if (a && b) html += `<path d="${curve(a, b)}" data-wire="${i}" class="${i === selectedWire ? "selected" : ""}"></path>`;
  });
  if (pending) html += `<path class="pending" d="${curve(pending.from, pending.to)}"></path>`;
  wiresSvg.innerHTML = html;
  for (const p of wiresSvg.querySelectorAll<SVGPathElement>("path[data-wire]")) {
    p.onpointerdown = (e) => {
      e.stopPropagation();
      select(null, Number(p.dataset.wire));
    };
  }
}

const toWorld = (clientX: number, clientY: number): [number, number] => {
  const r = view.getBoundingClientRect();
  return [(clientX - r.left - pan.x) / zoom, (clientY - r.top - pan.y) / zoom];
};

function startWire(e: PointerEvent, fromId: string, out: string): void {
  e.stopPropagation();
  e.preventDefault();
  const from = portPos(fromId, "out", out)!;
  const move = (m: PointerEvent) => drawWires({ from, to: toWorld(m.clientX, m.clientY) });
  const up = (u: PointerEvent) => {
    removeEventListener("pointermove", move);
    removeEventListener("pointerup", up);
    const target = (document.elementFromPoint(u.clientX, u.clientY) as HTMLElement | null)?.closest<HTMLElement>(".port.in");
    const toId = target?.closest<HTMLElement>(".node")?.dataset.id;
    if (target && toId && toId !== fromId) {
      const toNode = graph.nodes.find((n) => n.id === toId)!;
      if (!toNode.locked) {
        graph.wires = graph.wires.filter((w) => !(w.to[0] === toId && w.to[1] === target.dataset.name));
        graph.wires.push({ from: [fromId, out], to: [toId, target.dataset.name!] });
        render();
        changed();
        return;
      }
    }
    drawWires();
  };
  addEventListener("pointermove", move);
  addEventListener("pointerup", up);
}

// --- Selection, dragging, panning ---------------------------------------------------------------
function select(node: string | null, wire: number | null): void {
  selectedNode = node;
  selectedWire = wire;
  for (const [id, el] of elements) el.classList.toggle("selected", id === node);
  drawWires();
}

function startNodeDrag(e: PointerEvent, n: GraphNode, el: HTMLElement): void {
  e.stopPropagation();
  select(n.id, null);
  if (n.locked) return;
  const start = toWorld(e.clientX, e.clientY);
  const p0 = [...(n.pos ?? [0, 0])];
  const move = (m: PointerEvent) => {
    const p = toWorld(m.clientX, m.clientY);
    n.pos = [Math.round(p0[0] + p[0] - start[0]), Math.round(p0[1] + p[1] - start[1])];
    el.style.left = `${n.pos[0]}px`;
    el.style.top = `${n.pos[1]}px`;
    drawWires();
  };
  const up = () => {
    removeEventListener("pointermove", move);
    removeEventListener("pointerup", up);
  };
  addEventListener("pointermove", move);
  addEventListener("pointerup", up);
}

function applyTransform(): void {
  world.style.transform = `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`;
  view.style.backgroundSize = `${24 * zoom}px ${24 * zoom}px`;
  view.style.backgroundPosition = `${pan.x}px ${pan.y}px`;
}

view.onpointerdown = (e) => {
  if (e.button !== 0 || e.target !== view) return;
  select(null, null);
  menu.style.display = "none";
  view.classList.add("panning");
  const x0 = e.clientX - pan.x;
  const y0 = e.clientY - pan.y;
  const move = (m: PointerEvent) => {
    pan = { x: m.clientX - x0, y: m.clientY - y0 };
    applyTransform();
  };
  const up = () => {
    view.classList.remove("panning");
    removeEventListener("pointermove", move);
    removeEventListener("pointerup", up);
  };
  addEventListener("pointermove", move);
  addEventListener("pointerup", up);
};
view.addEventListener(
  "wheel",
  (e) => {
    e.preventDefault();
    const [wx, wy] = toWorld(e.clientX, e.clientY);
    zoom = Math.min(2, Math.max(0.25, zoom * Math.exp(-e.deltaY * 0.0015)));
    const r = view.getBoundingClientRect();
    pan = { x: e.clientX - r.left - wx * zoom, y: e.clientY - r.top - wy * zoom };
    applyTransform();
  },
  { passive: false },
);

function fit(): void {
  if (!graph.nodes.length) return;
  const xs = graph.nodes.map((n) => n.pos?.[0] ?? 0);
  const ys = graph.nodes.map((n) => n.pos?.[1] ?? 0);
  const w = Math.max(...xs) + 280 - Math.min(...xs);
  const h = Math.max(...ys) + 260 - Math.min(...ys);
  zoom = Math.min(1.2, Math.max(0.25, Math.min(view.clientWidth / w, view.clientHeight / h)));
  pan = { x: -Math.min(...xs) * zoom + 20, y: -Math.min(...ys) * zoom + 20 };
  applyTransform();
}

// --- Add / delete -------------------------------------------------------------------------------
view.oncontextmenu = (e) => {
  e.preventDefault();
  const at = toWorld(e.clientX, e.clientY);
  const cats = new Map<string, NodeDef[]>();
  for (const d of Object.values(NODE_TYPES)) cats.set(d.category, [...(cats.get(d.category) ?? []), d]);
  menu.innerHTML = "";
  for (const [cat, list] of cats) {
    menu.insertAdjacentHTML("beforeend", `<div class="cat">${cat}</div>`);
    for (const d of list) {
      const item = document.createElement("div");
      item.className = "item";
      item.textContent = d.type;
      item.title = d.doc;
      item.onclick = () => {
        menu.style.display = "none";
        let i = 1;
        const base = d.type.charAt(0).toLowerCase() + d.type.slice(1);
        while (graph.nodes.some((n) => n.id === `${base}${i}`)) i++;
        graph.nodes.push({ id: `${base}${i}`, type: d.type, pos: [Math.round(at[0]), Math.round(at[1])] });
        render();
        changed();
      };
      menu.appendChild(item);
    }
  }
  menu.style.left = `${e.clientX}px`;
  menu.style.top = `${e.clientY}px`;
  menu.style.display = "block";
};
addEventListener("pointerdown", (e) => {
  if (!(e.target as HTMLElement).closest("#menu")) menu.style.display = "none";
});

addEventListener("keydown", (e) => {
  if ((e.target as HTMLElement).closest("input")) return;
  if (e.key !== "Delete" && e.key !== "Backspace") return;
  if (selectedWire !== null) {
    const w = graph.wires[selectedWire];
    const target = graph.nodes.find((n) => n.id === w?.to[0]);
    if (w && !target?.locked) graph.wires.splice(selectedWire, 1);
  } else if (selectedNode) {
    const n = graph.nodes.find((x) => x.id === selectedNode);
    if (!n || n.locked) return;
    graph.nodes = graph.nodes.filter((x) => x.id !== selectedNode);
    graph.wires = graph.wires.filter((w) => w.from[0] !== selectedNode && w.to[0] !== selectedNode);
  } else return;
  select(null, null);
  render();
  changed();
});
