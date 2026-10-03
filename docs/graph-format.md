# Graph format

The visuals are driven by a node graph saved as JSON in `graphs/<name>.json`. The display loads
`graphs/default.json` (or `?graph=<name>`), evaluates it once per frame, and feeds the result to the
renderer. The editor (`editor.html`, or press `E` on the display) edits it live.

This document is the contract: a runtime in another language (e.g. Rust + wgpu) that follows it can
load the same files and produce the same frames.

## File

```json
{
  "version": 1,
  "nodes": [
    { "id": "arc", "type": "Expression", "name": "Sun arc", "locked": false,
      "params": { "expr": "a + b + c" }, "pos": [300, 20] }
  ],
  "wires": [
    { "from": ["horizon", "degrees"], "to": ["arc", "a"] }
  ]
}
```

- `id`: unique and stable; wires refer to it.
- `type`: a node type (below).
- `name`: label in the editor. Optional.
- `locked`: approved; the editor won't change, move or delete it. Optional.
- `params`: values for inputs that aren't wired. Missing inputs use the type's default.
- `pos`: editor position. Ignored by the runtime.
- A wire connects one output to one input. An input has at most one wire. Only `signal` inputs take wires.

## Values

A value is a number, a list of 3 numbers (a linear-RGB colour), or a string (only for `expr`).
Where a number is expected and a colour arrives, its first component is used; where a colour is
expected and a number arrives, it's repeated three times.

## Evaluation

Nodes are evaluated in dependency order (a depth-first topological sort). A node that is part of a
loop is reported as an error and skipped. For each node, each input takes, in order: the wired
output's value, the node's `params` value, the type's default.

Every node is a pure function of its inputs, the time `t` (seconds into the track) and the track's
analysis file. There is no per-frame state, so any time can be evaluated directly: seeking and
offline recording give identical frames. Nodes that need the whole track (`Spurts`, `Gate`)
precompute from their `const` inputs when the graph loads or those inputs change.

Input kinds:

- `signal`: wireable, read every frame.
- `const`: set in the node only; may trigger a precompute.

## Node types

Defined in `src/graph/nodes.ts` (each with a one-line doc shown in the editor). Categories:

| Category | Types |
|---|---|
| Music | Time, Beat, Bar, Phrase, Sound, Intensity, Loudness, Bass, Pump, Hats, Section, DropHit, Tension, Spurts, Gate |
| Scene | GapHorizon |
| Value | Number, Color |
| Setup | Setup |
| Shape | Expression, Add, Multiply, Remap, Oscillator, MixColor |
| Output | Sun, Camera, Tone |

Version 2 analysis files (`tools/audio_analysis.py`) add the phrase grid, drops with confidence, and
tension/energy curves (`Phrase`, `Tension`, `DropHit`'s `strength`/`ahead`/`windup`, `Beat`'s `confidence`).
With a version 1 file those fall back: phrases every 8 bars from the first downbeat, tension 0, every drop
section's start a drop with confidence 1. Section kinds intro, breakdown and outro count as quiet.

Without a track, music nodes return neutral values (intensity 1, pump 0, gate 1, section "normal",
bars advance at 2 s per bar), so graphs still run.

### Expression grammar

`Expression` evaluates a formula over inputs `a`–`f` and `t`:

```
expr    := compare
compare := sum (("<" | ">" | "<=" | ">=") sum)?      comparisons give 0 or 1
sum     := product (("+" | "-") product)*
product := power (("*" | "/") power)*              division by 0 divides by 1e-9
power   := unary ("^" power)?                      right-associative
unary   := "-" unary | atom
atom    := number | name | name "(" args ")" | "(" expr ")"
```

Functions: `sin cos tan abs sqrt exp log pow min max floor fract sign clamp(x, lo=0, hi=1)
mix(a, b, k) step(edge, x) smoothstep(e0, e1, x)`. Constants: `pi`, `tau`. Unknown names read as 0.
A non-finite result outputs 0.

## Outputs

Output nodes write the renderer's parameters. Anything a graph doesn't set keeps the display's own value.

| Node | Inputs |
|---|---|
| Sun | `arc` degrees along the sun's half-circle (0 front horizon, 90 overhead, 180 behind the camera), `azimuth` degrees, `intensity`, `color`, `rays` (visible shafts), `flare`, `skyBoost`, `enabled` |
| Camera | `swayX`, `swayY`, `pushZ` in units of the display's sway amount (pushZ is clamped to ≥ 0) |
| Tone | `baseDim`, `baked`, `cap` |
| NookLights | `positions` (u,v[,area,brightness]; ...), `sounds` (names, `|`-separated, fire the mid and far lights), `pattern`, `seed`, `attack`, `decay`, `near` (feet: nearer lights wait for big moments), `moments` (drops and/or phrases), `nearDecay`, `radius` (feet; each light can scale its area and brightness: `u,v,area,brightness`), `standoff`, `color`, `intensity`, `level` |

Sun, Laser, SkyLaser and Scan also take `level` (0..1, default 1), which scales them. A `Setup` node's `level`
output goes there: a setup (the sun, the ground laser rigs, the scan, ...) plays in the kinds of section ticked
on its Setup node (intro, build, drop, breakdown, normal, outro), fading in over `fadeIn` and out over
`fadeOut` seconds; `on` = 0 turns it off everywhere. The music monitor (G on the display) lists the setups.
