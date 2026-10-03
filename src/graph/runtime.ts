// Graph runtime: loads a graph (see docs/graph-format.md), orders it so every node runs after the nodes
// feeding it, and evaluates it once per frame. Output nodes fill a RenderOut for the display.

import { NODE_TYPES, type EvalContext, type InitContext, type NodeDef, type RenderOut, type Value } from "./nodes.ts";

export interface GraphNode {
  id: string; // unique, stable: wires refer to it
  type: string;
  name?: string; // label shown in the editor (defaults to the type)
  locked?: boolean; // approved: the editor won't change or delete it
  params?: Record<string, Value>; // values of unwired inputs
  pos?: [number, number]; // editor position
}

export interface GraphWire {
  from: [string, string]; // [node id, output name]
  to: [string, string]; // [node id, input name]
}

export interface Graph {
  version: 1;
  nodes: GraphNode[];
  wires: GraphWire[];
}

interface Compiled {
  node: GraphNode;
  def: NodeDef;
  state: unknown;
  constKey: string;
  error?: string;
}

export class GraphRuntime {
  graph: Graph;
  private order: Compiled[] = [];
  private byId = new Map<string, Compiled>();
  private incoming = new Map<string, Map<string, [string, string]>>(); // node -> input -> source
  /** Last evaluated outputs of every node: for the editor's live values. */
  values = new Map<string, Record<string, Value>>();
  errors = new Map<string, string>();

  private initCtx: InitContext;

  constructor(graph: Graph, initCtx: InitContext) {
    this.initCtx = initCtx;
    this.graph = graph;
    this.load(graph);
  }

  /** Replace the graph. Precomputed state is kept for nodes whose constants didn't change. */
  load(graph: Graph): void {
    const previous = this.byId;
    this.graph = graph;
    this.byId = new Map();
    this.incoming = new Map();
    this.errors = new Map();
    for (const w of graph.wires) {
      if (!this.incoming.has(w.to[0])) this.incoming.set(w.to[0], new Map());
      this.incoming.get(w.to[0])!.set(w.to[1], w.from);
    }
    for (const node of graph.nodes) {
      const def = NODE_TYPES[node.type];
      if (!def) {
        this.errors.set(node.id, `unknown node type "${node.type}"`);
        continue;
      }
      const consts: Record<string, Value> = {};
      for (const inp of def.inputs) if (inp.kind === "const") consts[inp.name] = node.params?.[inp.name] ?? inp.default;
      const constKey = node.type + JSON.stringify(consts);
      const old = previous.get(node.id);
      const c: Compiled = { node, def, state: undefined, constKey };
      if (old && old.constKey === constKey && !old.error) c.state = old.state;
      else {
        try {
          c.state = def.init?.(consts, this.initCtx);
        } catch (err) {
          c.error = String((err as Error).message ?? err);
          this.errors.set(node.id, c.error);
        }
      }
      this.byId.set(node.id, c);
    }
    this.order = this.sort();
  }

  /** Depth-first topological order. A cycle is reported on the node that closes it, which is skipped. */
  private sort(): Compiled[] {
    const order: Compiled[] = [];
    const mark = new Map<string, 1 | 2>();
    const visit = (id: string): void => {
      const c = this.byId.get(id);
      if (!c || mark.get(id) === 2) return;
      if (mark.get(id) === 1) {
        this.errors.set(id, "part of a loop: a node can't feed itself");
        return;
      }
      mark.set(id, 1);
      for (const [, src] of this.incoming.get(id) ?? []) visit(src[0]);
      mark.set(id, 2);
      order.push(c);
    };
    for (const id of this.byId.keys()) visit(id);
    return order;
  }

  evaluate(t: number): RenderOut {
    const out: RenderOut = {};
    const ctx: EvalContext = { t, music: this.initCtx.music, scene: this.initCtx.scene, out };
    for (const c of this.order) {
      if (c.error || this.errors.has(c.node.id)) continue;
      const inputs: Record<string, Value> = {};
      const wired = this.incoming.get(c.node.id);
      for (const inp of c.def.inputs) {
        const src = inp.kind === "const" ? undefined : wired?.get(inp.name);
        const fromValue = src ? this.values.get(src[0])?.[src[1]] : undefined;
        inputs[inp.name] = fromValue ?? c.node.params?.[inp.name] ?? inp.default;
      }
      try {
        this.values.set(c.node.id, c.def.eval(inputs, ctx, c.state));
      } catch (err) {
        this.errors.set(c.node.id, String((err as Error).message ?? err));
      }
    }
    return out;
  }
}
