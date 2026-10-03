// A tiny expression language for Expression nodes: numbers, inputs a-f and t, + - * / ^, unary minus,
// parentheses, comparisons (return 0 or 1), and a fixed set of functions. Parsed once into a closure.
// Deliberately small and self-contained (no eval) so the same grammar is easy to implement in Rust.
//
//   expr    := compare
//   compare := sum (("<" | ">" | "<=" | ">=") sum)?
//   sum     := product (("+" | "-") product)*
//   product := power (("*" | "/") power)*
//   power   := unary ("^" power)?
//   unary   := "-" unary | atom
//   atom    := number | name | name "(" args ")" | "(" expr ")"

export type Env = Record<string, number>;
type Fn = (env: Env) => number;

const FUNCTIONS: Record<string, (...x: number[]) => number> = {
  sin: Math.sin,
  cos: Math.cos,
  tan: Math.tan,
  abs: Math.abs,
  sqrt: (x) => Math.sqrt(Math.max(0, x)),
  exp: Math.exp,
  log: (x) => Math.log(Math.max(1e-9, x)),
  pow: Math.pow,
  min: Math.min,
  max: Math.max,
  floor: Math.floor,
  fract: (x) => x - Math.floor(x),
  sign: Math.sign,
  clamp: (x, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, x)),
  mix: (a, b, k) => a + (b - a) * k,
  step: (edge, x) => (x >= edge ? 1 : 0),
  smoothstep: (e0, e1, x) => {
    const k = Math.min(1, Math.max(0, (x - e0) / (e1 - e0 || 1e-9)));
    return k * k * (3 - 2 * k);
  },
};
const CONSTANTS: Record<string, number> = { pi: Math.PI, tau: 2 * Math.PI };

export class ExprError extends Error {}

export function compile(source: string): Fn {
  const tokens = source.match(/\d+\.?\d*(?:e[+-]?\d+)?|\.\d+|[A-Za-z_]\w*|<=|>=|[-+*/^(),<>]|\S/g) ?? [];
  let i = 0;
  const peek = () => tokens[i];
  const take = (expected?: string) => {
    const tok = tokens[i++];
    if (expected !== undefined && tok !== expected) throw new ExprError(`expected "${expected}" but found "${tok ?? "end"}" in: ${source}`);
    return tok;
  };

  const expr = (): Fn => compare();
  const compare = (): Fn => {
    const left = sum();
    const op = peek();
    if (op === "<" || op === ">" || op === "<=" || op === ">=") {
      take();
      const right = sum();
      if (op === "<") return (e) => (left(e) < right(e) ? 1 : 0);
      if (op === ">") return (e) => (left(e) > right(e) ? 1 : 0);
      if (op === "<=") return (e) => (left(e) <= right(e) ? 1 : 0);
      return (e) => (left(e) >= right(e) ? 1 : 0);
    }
    return left;
  };
  const sum = (): Fn => {
    let left = product();
    while (peek() === "+" || peek() === "-") {
      const op = take();
      const l = left;
      const r = product();
      left = op === "+" ? (e) => l(e) + r(e) : (e) => l(e) - r(e);
    }
    return left;
  };
  const product = (): Fn => {
    let left = power();
    while (peek() === "*" || peek() === "/") {
      const op = take();
      const l = left;
      const r = power();
      left = op === "*" ? (e) => l(e) * r(e) : (e) => l(e) / (r(e) || 1e-9);
    }
    return left;
  };
  const power = (): Fn => {
    const base = unary();
    if (peek() === "^") {
      take();
      const ex = power();
      return (e) => Math.pow(base(e), ex(e));
    }
    return base;
  };
  const unary = (): Fn => {
    if (peek() === "-") {
      take();
      const inner = unary();
      return (e) => -inner(e);
    }
    return atom();
  };
  const atom = (): Fn => {
    const tok = take();
    if (tok === undefined) throw new ExprError(`unexpected end of: ${source}`);
    if (tok === "(") {
      const inner = expr();
      take(")");
      return inner;
    }
    if (/^[\d.]/.test(tok)) {
      const v = Number(tok);
      return () => v;
    }
    if (/^[A-Za-z_]/.test(tok)) {
      if (peek() === "(") {
        const fn = FUNCTIONS[tok];
        if (!fn) throw new ExprError(`unknown function "${tok}" in: ${source}`);
        take("(");
        const args: Fn[] = [];
        if (peek() !== ")") {
          args.push(expr());
          while (peek() === ",") {
            take();
            args.push(expr());
          }
        }
        take(")");
        return (e) => fn(...args.map((a) => a(e)));
      }
      if (tok in CONSTANTS) {
        const v = CONSTANTS[tok];
        return () => v;
      }
      return (e) => e[tok] ?? 0;
    }
    throw new ExprError(`unexpected "${tok}" in: ${source}`);
  };

  const fn = expr();
  if (i < tokens.length) throw new ExprError(`unexpected "${tokens[i]}" in: ${source}`);
  return fn;
}
