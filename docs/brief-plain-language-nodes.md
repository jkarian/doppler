# Brief: plain-language scripting with nodes

Status: idea, not started. Written 2026-10-03 from a design chat. Intended to seed its own project.

## The vision

Anyone can make 3D, music-driven scenes by typing. You write what you want in plain words; the system turns
it into a small, ordered scripting language; the script builds and connects nodes. Beginners work with
words and a few parameters. Experts can open any node, see the parts inside, and make new ones that become
new words for everyone.

> "make a geodesic sphere and give it a particle trail"

becomes

```
make geodesic sphere "orb"
add particle emitter "sparks" on orb's vertices
sparks leave trails
```

and those three lines create and wire the nodes.

## Principles

1. **The text drives creation.** The script is what you author. Nodes are what it produces.
2. **Two stages, kept separate.**
   - Prompt → script: AI, fuzzy. It rewrites loose language into the proper vocabulary.
   - Script → nodes: a plain parser, exact. The same lines always give the same graph. No AI, offline, instant.
3. **Always show the rewrite before applying it.** You see the tidied script (and what it will add or change),
   then accept. Nothing is applied silently.
4. **The rewrite teaches.** Seeing your words come back in the proper form teaches you the vocabulary, so
   next time you type it directly.
5. **Everything is reusable.** Every node is built from smaller nodes, down to a set of basic building blocks.
6. **Words for making, sliders and curves for tuning.** Sentences create things; params on the node refine
   them. Some params stay drawn curves, because no sentence describes a falloff as well as a sketch.

## Where you type

- **On the canvas** (click empty space, or the prompt dialog): a scene-level addition. Creates something new.
- **Inside a node, on one of its script lines**: edits that line. For example
  `sparks leave trails` → typed as "sparks leave trails that last between 1.5 and 3 seconds" → rewritten to
  `sparks leave trails (life 1.5–3 s)`.

## How nodes get grouped and connected

### Every word is a recipe

Each verb or phrase in the vocabulary is a **recipe**: a small, prebuilt group of basic nodes with a few
params brought to the front.

| Phrase | Recipe (basic nodes inside) | Front params |
|---|---|---|
| `make geodesic sphere` | sphere generator, position, material, render | detail, radius, colour |
| `add particle emitter … on <thing>'s vertices` | vertex points, emitter, particle simulation | rate, speed, life |
| `<thing> leave trails` | trail history, ribbon renderer, fade | length, width, fade |

**The vocabulary and the node library are the same thing.** Adding a word means adding a recipe.

### The grammar does the wiring

The subject of a sentence is what the recipe plugs into. `sparks leave trails` attaches the trail recipe to
the output of `sparks`. Beginners never wire basic nodes; the sentence shape already says where things connect.

A few sentence shapes should cover most needs:

```
make <type> "<name>" (params)
add <type> "<name>" on/from <thing>
<thing> <verb> (params)
<thing> <param> follows / pulses with <signal> (params)
place <thing> <preposition> <place>, <distance>
```

Binding things to the music in words ("orb size pulses with kick, heavy"; "emit rate follows intensity,
doubles in drops") is the domain-specific win over general node tools.

### Names decide the grouping

Whatever the user names ("orb", "sparks") becomes one **thing**: the unit a beginner sees.

### Three levels of view

1. **Scene**: a card per thing (orb, sparks, sun, lasers) plus music signals. The only wires shown are the
   relations the script stated: *on, from, follows, with*.
2. **Thing**: open a card to see its stack of sentences, each with its front params (sliders, colours,
   curves). Like Unity components, Blender's modifier stack or After Effects' effect stack: a list, not a graph.
3. **Parts** (advanced): open a sentence to see the basic nodes of its recipe. Edit and save as a new
   recipe = **make a new word**. This is how the library grows and how expert work becomes beginner vocabulary.

### Places and prepositions

"Place orb above the canyon gap, 40 m up" splits into two reusable parts:

- **Places**: named spots in the scene, derived from the scene's depth map and sky mask (gap, floor, ledge
  tops, rim, sky). Scene-specific. Making a new one could be as simple as clicking the image and saying
  "call this the ledge". (Doppler's `GapHorizon` node is an early example.)
- **Prepositions**: a small fixed set of generic operators (*above, on, around, between, facing, along*)
  taking a place and a distance. They work in every scene.

## Lines nothing can fulfil

When a script line has no recipe, the line and its node are flagged ("no recipe for *orbit in a figure
eight*"), with two ways forward:

1. **Compose it** from existing words: the AI proposes sentences that do it, which become a new recipe.
2. **Request a new building block**: genuinely new code (a shader, a simulation). This goes in as a draft
   change to be reviewed and tested, never generated silently into the live editor.

## Known hard parts

- **New data types.** Today Doppler's graph only carries numbers and colours over time. Geometry needs
  meshes, point clouds and particle buffers on the wires.
- **State.** Particles and trails remember previous frames. Doppler's nodes are pure functions of time, which
  is what makes seeking and offline recording exact. Simulations need checkpoints or caching to keep that.
- **GPU execution.** A per-frame CPU graph is fine for dozens of signals, not for 100k particles. The graph
  has to compile into GPU passes.
- **Recipe versioning.** If a recipe changes, do existing uses update, or only new ones?
- **Ambiguity.** "It" and "the sphere" don't scale; the rewrite should give every thing an explicit name.

## Prior art to study

Houdini (digital assets, VEX), TouchDesigner (components; channels vs geometry), Blender geometry nodes and
node groups, Unreal Blueprints, ComfyUI, Unity components, and Inform 7 (a programming language written as
English sentences, the closest model for the controlled vocabulary).

## Open questions

- **Name.** "Doppler" is taken by the current visualizer repo, and it's a good name. One option: the new tool
  becomes Doppler, and the current repo becomes its first scene (e.g. renamed to the canyon scene).
- Is the script file the main saved format (good for git diffs and AI context), with the node JSON generated?
- How much of the vocabulary ships built in, and how much is grown by users?
- How are curves and other drawn params written in the script (by reference, e.g. `fade (curve "soft")`)?

## Action item (John): paper exercise

Before any code, write 4–5 real scenes in the script language by hand, then list what they need. The
grouping, the vocabulary and the gaps fall out of it.

For each scene:

1. Write it the loose way, as you'd say it.
2. Rewrite it as ordered script lines, using and inventing vocabulary as you go.
3. For each line, list the recipe it needs and the basic nodes inside.
4. Mark which basic nodes appear in several scenes (the core library) and which lines have no obvious recipe.

Suggested scenes:

- Today's canyon: sun spurts on bass, ground laser rigs re-aiming every 2 bars, scan in drops.
- The orb: geodesic sphere above the gap, sparks from its vertices with trails, size pulsing on the kick.
- A laser tunnel along the canyon floor, chasing in time with the hats.
- Clouds drifting over the rim, casting moving shadows, thickening in quiet sections.
- One scene of your own choosing that you couldn't build in any tool today.

## Archetype scenes

Doppler's nodes and workflow are being reverse-engineered from three archetype scenes, each built for real:

1. **Image to world** (doppler-canyon): one image becomes a 2.5D scene, lit by music-driven setups.
2. **Particles**: simulation, state over time, very large counts on the GPU.
3. **Straight 3D**: real geometry and procedural generation (idea: a car driving on a procedural road).

What all three need is the core; what only one needs is a library.

## Lessons for Doppler

From archetype 1, the canyon (October 2026):

- **Setups are the unit people think in.** Not nodes: "the nook lights", "the ground rigs", "the sun". Each is
  switched on and off by the song (Setup node: section kinds, fades) and has its own handful of settings.
- **Separate what, how and when.** Setup = the objects (what); pattern = how they animate (chase, alternate
  sides, hit); arrangement = which pattern plays on which setup in which part of the song (when). Patterns
  should work on any setup.
- **Objects need hands-on placement, with the right handles.** Clicking a point on the picture places a
  light; rigs need Maya-style rotate rings (turn and tilt), not abstract numbers. Each kind of object
  needs its own direct-manipulation mode, shown on the picture, with numbers as readouts.
- **Elements need attributes.** Each light has its own area and brightness; each rig its own aim and cone.
  This points to a Houdini-style point/attribute model with groups and per-element rules.
- **People hear instruments, not frequencies.** Name sounds by example ("this is the tick") and drive things
  from them. Stems give instrument groups; naming gives the instruments.
- **Song structure is the timeline.** Sections, drops, phrases and tension drive everything; people tune
  against a visual of the whole track (the monitor), not a list of numbers.
- **Every setting needs a plain explanation on hover.** Unexplained knobs (floorLight) stall people.
- **A few master sliders matter most.** Sun strength, brightness cap, audio sync. Keep them always in view.
- **Real-world units and frames.** Feet for sizes, true vertical for angles (a photo's camera is never
  level), real degrees for the sun. Picture-relative numbers confuse.
- **Physical plausibility reads as quality.** Shadows with hard edges, light that stays off the floor when
  it should, beams stopped by rock, cave rock that blocks a low sun.
- **Audio/visual sync is per setup of screens and speakers.** A TV needed 135 ms; it must be easy to set.
- **The pipeline must be automatic.** Image in, scene out; every guess estimated, with a slider to correct
  it by eye. Manual 3D steps dilute the value.
- **Show, don't describe.** Iterating by eye with frames and short clips, and sketches drawn on screenshots,
  is how decisions get made. The tool should make both easy.
- **Automatic first, markups second.** The image-to-scene pass gets most of the way; the rest is fixed with
  quick markups on the picture inside the app, not a 3D package: brush an area and say what it is or what
  to do ("repaint behind this: red sandstone", "this is sky", "push this back", "this is one object"). Each
  markup is a node (mask, action, prompt), saved with the scene, editable and undoable.
