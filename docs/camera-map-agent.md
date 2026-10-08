# Camera-map agent: design

Agreed with the user 2026-10-08. Nothing built yet. Goal: drop in an image, an agent does what Claude did by hand on
the car, ice cave and whale shark: judge the image, run the pipeline, look at every check image with its own eyes,
fix, and learn from every encounter. The tool stays as general as possible, but some images need their own method.

## Where the learning lives

The model's weights don't change and it remembers nothing between sessions. All learning is in files the agent reads
at the start of every run: the **method library**, the **casebook** and the **tag vocabulary**. Every image adds a
case (often a new method or failure example); the next image starts from all of them. The casebook doubles as the
eval set (and later fine-tuning data, if ever worth it).

## Tags

Two kinds, kept apart:
- **Trait tags** describe the input, visible in the image before anything runs. They pick the methods.
- **Failure tags** describe what went wrong, recorded afterwards, each linked to the method that fixed it.

A small controlled vocabulary in flat facets (no category tree: images don't fit trees; the shark is three branches
at once). Each tag gets a one-line definition and an example crop from a real case, so tagging stays consistent.
Free tags drift into synonyms (glassy / translucent / see-through) and break search.

| Facet | Trait tags (start) |
|---|---|
| Subject edge | `solid-edge`, `see-through`, `hair`, `soft-volume` (smoke, cloud), `reflective` |
| Backdrop | `open-sky`, `open-water`, `ground-surface`, `enclosure` (cave, interior) |
| Depth structure | `single-subject`, `layered-landscape`, `close-up`, `crowd` |
| Optics | `deep-focus`, `shallow-dof`, `bokeh`, `glow` |
| Frame | `subject-inside-frame`, `subject-cropped` |
| Origin | `photo`, `ai-render`, `painting` |

Failure tags (start): `halo`, `edge-bleed`, `ghost-fill` (object painted into a hole), `outline-in-fill`,
`blobs-past-frame`, `depth-jump`, `seam`, `missing-texture`.

The vocabulary grows like methods do: the agent proposes a tag when nothing fits; it joins when the user approves or
a second case needs it.

Tags carry the reasoning; they miss style. Each case also stores an image embedding (DINOv2 or CLIP). Retrieval for a
new image = tag overlap + visual similarity; the agent reads the closest 2-3 cases in full.

## Method library

Each method is a named module:
- **When:** the trait tags / visual signs that call for it.
- **What:** what it does and where it plugs into the pipeline (cut-out, fill, depth, lift).
- **Code and parameters.**
- **Known failures**, with real example images (e.g. hair-lift: the fibre bulge of whale-shark v002).
- **Cases** that use it.

A method only changes what it is switched on for, so a new method is tested on the cases that use it, not on
every image. First methods (this week's options, code stays in `tools/camera_map.py`, now named and documented):

| Method | Called by | parts.json today |
|---|---|---|
| `birefnet-outline` | `solid-edge` | default for freestanding parts |
| `see-through-outline` | `see-through` | `"outline": "sam"` |
| `hair-lift` (soft-alpha plane for fibres, difference key on a clean plate, smoothed body depth) | `hair` | `"hair": true` |
| `open-backdrop` (far card, smooth multi-scale fill, no Flux) | `open-sky`, `open-water` | `"open": true` |
| `flux-fill-behind` | everything not open | default |

A trait tag with no method is the agent's signal that this image may need a new one.

## Promotion ladder

| Level | Lives | Used |
|---|---|---|
| One-off | in that case's record only | a trick for one troublesome image; allowed, marked as such |
| Method | the library, opt-in | once built general enough to reuse |
| Core | the default pipeline | once proven on several cases; must then pass all cases |

The agent writes one-offs and methods freely; anything moving up a level goes to the user with before/after images.
Regression runs only cover the cases that share the changed code (all cases for core changes).

## Casebook (one record per image)

Image; trait tags; embedding; methods chosen; parts.json; size confirmed; check images per stage (incl. motion);
every failure with who caught it (agent or user) and its failure tag; the fix; the user's verdict.

First three cases:

| Case | Trait tags | Failure tags |
|---|---|---|
| car-beach | `solid-edge`, `ground-surface`, `open-sky`, `single-subject`, `deep-focus`, `subject-cropped`, `ai-render` | `halo` |
| ice-cave | `enclosure`, `layered-landscape`, `deep-focus`, `subject-cropped`, `photo` | `seam` |
| whale-shark | `see-through`, `hair`, `open-water`, `single-subject`, `shallow-dof`, `bokeh`, `glow`, `subject-inside-frame`, `ai-render` | `edge-bleed`, `ghost-fill`, `outline-in-fill`, `blobs-past-frame`, `depth-jump`, `missing-texture` |

## The run

1. Tag the image; retrieve similar cases; pick methods; write parts.json.
2. `measure`; propose a real size (from the subject: "whale shark ~10 m"); the user confirms (the only human step).
3. Segment, layers, paint, export, each followed by the agent looking at the check image against the failure
   examples of the chosen methods and similar cases; fix parts.json and retry (bounded).
4. Motion check: the layers rendered from the camera's extreme positions side by side (numpy, no Maya). Both faults
   the user caught in Maya (fin bleed, fibre bulge) only showed in motion; the agent needs this eye.
5. When no method copes: diagnose, write a one-off or a new method, test it on the cases it touches, record it.
6. Write the case record; propose promotions and new tags.

## Build order

1. Motion check (the agent's most important missing eye).
2. Case records for the three scenes + the tag vocabulary with example crops + the method docs.
3. The skill (`.claude/skills/camera-map/SKILL.md`): the run above, retrieval, promotion rules, retry limits.
4. Later: the same as the system prompt of an Agent SDK / API web app (each step a tool returning numbers + its
   check image; size check as a UI screen; serverless GPU; the casebook as evals). FLUX.1 Fill [dev] is
   non-commercial: swap or license before selling.
