# Hearing music structure the way a listener does

Research and hands-on tests for the visualizer, October 2026.

**About this document:**

- Every tool test was run on our four tracks on the RTX 4090, in a separate scratch environment. The project and its Python environment weren't touched.
- Anything that couldn't be checked is marked **[unverified]**.
- The test scripts, outputs and Windows workarounds are kept locally (not in git) in `captures/research/music/`. The separated stems aren't kept; they're regenerated in seconds.

---

## Executive summary

- **On these tracks the problems are a wobbly beat grid and section boundaries that don't sit on bar lines, not the downbeat phase.**
  - On Doomsday Clock (0–305 s) and Ever Now (0.8–94 s), every one of our downbeats lands on the "1" in both Beat This! and All-In-One.
  - **Our librosa beat grid wobbles by up to about ±130 ms in busy passages.** In Ever Now, bar lengths jump to 2.36 s and then 2.09 s at 27.5, 34.3, 67.5 and 69.9 s. The real bar is 2.22 s.
  - **Our section boundaries don't land on bar lines or phrases.** On Doomsday ours fall at 36.03, 135.67, 153.08 and 209.43 s (build start); the models agree on bar lines at 38.42, 134.42, 153.62 and 208.82 s. That's one beat to one bar off, which is exactly the "a beat late" feeling.
- **Beat This! (with its DBN smoothing step) or All-In-One gives a perfectly regular grid.** Neither had a single irregular beat on these tracks.
  - On the synthetic test track, where the true beat times are known, Beat This! is exactly on the beat in the drop (0 ms error). We're 16–20 ms late.
  - Beat This! only works in 20 ms steps, so keep our bass-onset alignment as a fine-tuning step afterwards.
- **The structure models agree with each other and sit on an 8-bar phrase grid.**
  - For Doomsday (bar 2.4 s, phrase 19.2 s), All-In-One, SongFormer and EDMFormer all put boundaries at 19.2, 38.4, 57.6, 134.4, 153.6, 172.8, **211.2** and 268.8 s.
  - The big drop at 211.2 s is found by all three models (211.20, 211.09, 211.21) and by our pipeline (211.23).
  - The separated stems show the build clearly: the drums fade from 201.6 s, and the bar at 208.8 s is a "gap bar" where the bass and sub are at zero.
- **Take the boundaries from the models, but not their labels.**
  - All-In-One calls almost everything "solo" or "intro".
  - EDMFormer (an EDM model from 2026) calls almost every section of the synthwave track and of Ever Now a "drop".
  - SongFormer's "chorus" lands on the drops, which is the most useful of the three.
  - Plan: boundaries from the models by vote, snapped to downbeats; drop, build and breakdown labels computed by us from the stems' energy.
- **Bliss (KLSR) has essentially no drums**: the drums stem is about 0.0009 loud against 0.12 for the bass. Every tracker fails on it:
  - All-In-One finds 1 beat.
  - Beat This! finds random beats.
  - Our 121.6 BPM is also a guess.
  - We need an explicit "no reliable grid" mode for tracks like this.
- **Audio-language AI models can't place events precisely enough to time visuals yet.**
  - Gemini 2.5 Pro scores 0.42 at ±0.5 s on the SongFormBench benchmark, against SongFormer's 0.70.
  - On the 2026 MusTBench benchmark, the best models find onsets within 3 s only about 60% of the time, and get worse after about 120 s into a track.
  - Use them only for rough descriptions, given our grid and candidate boundaries, and snap whatever they say to the grid.
- **For anticipation, compute two curves, not one.**
  - Listener studies show *tension* rises through the build and releases about 2–3 s into the drop, while *energy* peaks in the drop.
  - So drive the wind-up from tension and the sun "spurt" from the jump in energy.
  - Since our analysis is offline, anticipation is mostly look-ahead: seconds or bars until the next drop or phrase.
  - Build cues confirm and shape it: drums or kick removed, bass dropping out in the last bar, a riser, a snare roll speeding up, a drum fill in the last bar.
- **Recommended setup, under about a minute per track on the 4090:**
  1. HTDemucs stems (3.3 s for a 5-minute track).
  2. Beat This! with smoothing (under 1 s, plus 1–8 s).
  3. All-In-One and SongFormer (plus EDMFormer) boundaries (about 15 s, 2 s and 10 s).
  4. Vote and snap to the bar grid.
  5. Fit the 8-bar phrase grid.
  6. Label sections from the stems, then compute the tension, energy and anticipation curves.
  - A prototype of this gets the synthetic test track exactly right: build at 16.02 s, drop at 32.00 s, outro at 52.00 s.
  - On Doomsday it finds drops at 96.02, 172.82 and 211.22 s, with 211.22 as the most confident (0.81).
- **Licences:**
  - MIT: Beat This!, All-In-One, Demucs.
  - madmom: the code is BSD, its bundled models are non-commercial. All-In-One only uses its DBN *code*.
  - SongFormer and EDMFormer include MuQ, whose weights are **non-commercial**.
  - Non-commercial: MERT, Music Flamingo, Essentia's emotion models.
  - Apache-2.0: Qwen3-Omni.

---

## 0. Results from testing on our tracks

### 0.1 Setup (what it took on Windows)

- **Environment:** Python 3.12 with torch 2.11 (CUDA 12.8).
  - Careful: installing `muq` or `x-transformers` silently replaced the CUDA version of torch with a CPU-only one. Pin torch.
- **Audio:** decoded the m4a files to 44.1 kHz WAV with the imageio-ffmpeg binary.
- **madmom** has no Windows install and needs Microsoft's C++ compiler, which isn't installed. Its one compiled file that matters was rewritten in pure Python, with the Viterbi step running on the GPU in PyTorch. That's enough for the smoothing step used by Beat This! and All-In-One. (Saved in `captures/research/music/windows-shims/`.)
- **NATTEN** (needed by All-In-One) also needs compiling on Windows. A small pure-PyTorch replacement for the four functions All-In-One uses lets it run unchanged. Its beats on the synthetic track land within 10–20 ms of the truth.
- **SongFormer** ran through Hugging Face (with `muq`, `loguru` and `msaf` installed).
- **EDMFormer** ran from the `25ohms/EDM-98` repo.

### 0.2 Speed on the RTX 4090

| Step | Time | Notes |
|---|---|---|
| HTDemucs (`htdemucs`), 315 s track | **3.3 s** (+1 s model load) | 1.3 GB GPU memory |
| `htdemucs_ft` (four fine-tuned models) | 11.6 s (+10.8 s load) | 3.1 GB |
| Beat This! `final0`, no smoothing | **0.1–1 s per track** (+7.5 s first load) | 20 ms steps |
| Beat This! with smoothing (our GPU workaround) | 1.5–7.5 s per track | The compiled original is likely faster **[unverified]** |
| All-In-One, stems reused | **about 15 s for a 315 s track** | |
| SongFormer | 0.5–2 s per track after a **25 s load** | Downloads about 2.8 GB of weights |
| EDMFormer | 11–34 s per call (reloads its models each time) | About 2–3 s per track if kept loaded **[estimate]** |

### 0.3 Beats and downbeats

Compared with Beat This! with smoothing; "matched" means within 70 ms:

| Track | System | Beats | BPM | Matched | Median offset |
|---|---|---|---|---|---|
| Doomsday Clock | ours | 525 | 100.33 | 0.95 | **+31 ms** |
| | Beat This! (no smoothing) | 518 | 100.00 | 0.99 | 0 |
| | All-In-One | 448 | 100.00 | 1.00 | −10 ms |
| Ever Now | ours | 174 | 107.72 | 0.88 | **+26 ms** |
| | Beat This! (no smoothing) | 177 | 107.14 | 0.97 | 0 |
| | All-In-One | 155 | 107.14 | 1.00 | 0 |
| Bliss | ours | 298 | "121.7" | 0.14 | – |
| | Beat This! | 128 / 161 | 187.5 / 76.9 | – | **all fail** |
| | All-In-One | 1 | – | – | **fails** |

**Downbeats:**
- Doomsday: on the "1" from 0 to 304.8 s against Beat This!, and from 2.4 to 268.8 s against All-In-One.
- Ever Now: on the "1" from 0.8 to 94.4 s in all three.

So the downbeats agree. The problems are the grid's local wobble and where the sections are placed.

**Synthetic test track against the true beat times** (a beat exactly every 0.5 s):

| Section | Ours | Beat This! | Beat This! + smoothing | All-In-One |
|---|---|---|---|---|
| Quiet intro (hats on the off-beats only) | −215 ms (locked onto the hats) | −202 ms | −197 ms | +165 ms |
| Build (kick on every beat) | +20 ms | +18 ms | +17 ms | −10 ms |
| Drop | **+16 ms, worst beat 109 ms** | **0 ms, worst 0** | **0 ms** | −11 ms |

- **Drumless intros:** every system, ours included, puts the intro beats on the off-beat hats. Only the kick fixes it. Since we know the whole track, **fit the grid where it's confident and extend it backwards** into drumless intros.
- **Doubled tempo:** Beat This! without smoothing can double the tempo during fast synth passages. It did on Ever Now at 28.0–29.4 s and 62.5–67.7 s. Smoothing removes this, and All-In-One never does it.
- **Steady tempo:** Doomsday is exactly 99.999 BPM all the way through (11.9 ms average deviation, mostly Beat This!'s 20 ms steps). Ever Now is 107.94 BPM until its drumless outro. A **constant tempo (or a few constant pieces)** is a strong, safe assumption for electronic music.

### 0.4 Sections and boundaries

| Track | System | Boundaries (s) and labels |
|---|---|---|
| **Doomsday** | ours | 36.03 quiet · 57.06 normal · 87.66 quiet · 89.44 normal · **135.67** quiet · **153.08** normal · **209.43 build** · **211.23 drop** · 303.04 · 307.09 |
| | All-In-One | 19.20 · 38.40 · 57.60 · 76.80 · 96.00 · 115.20 · 134.40 break · 153.61 · 172.81 · 194.38 · **211.20** · 230.40 · 249.60 · 268.80 · 283.21 outro · 313.60 |
| | SongFormer | 19.20 · 38.28 · 57.60 · 95.88 chorus · 115.08 · 134.41 bridge · 153.61 · 172.69 · **211.09 chorus** · 230.29 · 249.25 · 268.81 · 292.57 outro |
| | EDMFormer | 19.20 buildup · 38.28 · 57.60 drop · 86.40 · 115.32 · 134.29 breakdown · 153.97 · 172.81 · **211.21 drop** · 225.49 · 239.89 · 268.69 outro · 292.69 |
| **Ever Now** | ours | 0 normal · 89.69 quiet (a single section) |
| | All-In-One | 23.03 · **36.35** · **58.58** · 71.91 · 97.02 (all "solo") |
| | SongFormer | 4.80 · **36.36** · 54.12 · 71.88 · 96.36 (all "inst") |
| | EDMFormer | 23.16 · **62.88** · 89.28 (all "drop" except the outro) |
| **Bliss** | ours | 10.66 · 62.93 build · 72.00 drop · 133.57 |
| | All-In-One | 24.96 · 41.61 · 60.97 · 95.98 · 113.35 (barely any confidence) |
| | SongFormer | 5.28 · 25.08 · 43.80 · **70.68** · 95.88 · 113.28 · 131.89 |
| **Test track** (truth: 16 / 32 / 52) | ours | 5.77 build · 16.02 · 32.02 drop · 51.89 |
| | All-In-One | 15.99 · 31.98 · 51.98 |
| | SongFormer | 15.84 · **31.92 chorus** · 51.84 |
| | EDMFormer | **15.96 buildup** · **32.04 drop** · 51.96 (but it calls the quiet intro a "drop") |

**What the stems show at the key moments** (loudness per bar; bars from Beat This!):

- **Doomsday, the big drop at 211.22 s.**
  - The drums fade over the phrase before it: 0.33 → 0.15 (201.6 s) → 0.13 → 0.11 → **0.06 (208.8 s)**.
  - Bar 208.8 is a gap bar: bass 0.00, sub 0.003, while the brightness jumps, probably a riser or filter sweep.
  - At 211.22 s the high band jumps from 0.14 to **0.40** and the "other" stem to 0.62.
  - This drop is a **brightness and lead event, not a kick event**, so a kick- or bass-only detector underrates it.
- **Doomsday at 96.02 s.** Bar 93.62 is a **drum fill** (drums 0.32, against about 0.03 before), and at 96.02 the high band goes 0.13 → 0.37. The fill is a one-bar warning.
- **Ever Now** has no drums at all. What you hear as intensity is brightness: averaged per 4 bars it's about 1040 Hz at 27.5 s, drops to 711–774 Hz from 36 to 54 s, and rises to **1518 Hz at 63 s**.
  - Your sense of its shape (peaks around 28–38 s and 64–88 s, breakdown around 40–60 s) matches boundaries at **36.34 and 60.82 s**.
  - The AI models disagree around 54–63 s, but a simple bar-by-bar change detector on the stem features finds 36.34 and 60.82 exactly.
- **Bliss:** the bass drops out from about 60 s to 72 s and comes back at about 72 s. That's its build and drop, and our pipeline actually catches it reasonably. The structure models see almost nothing in this track.

### 0.5 Prototype that combines everything (`scripts/fuse.py`)

**Method:**

1. Bars come from Beat This! with smoothing.
2. Each structure model, plus a stem-change detector, votes for the nearest downbeat: 1 vote, plus 0.5 for each neighbouring bar.
3. Keep peaks with at least 2 votes.
4. Choose the 8-bar phrase alignment that explains the most boundaries.
5. Snap boundaries that are one bar off onto the 4-bar grid.
6. Score each bar's energy from the stems.
7. Label sections:
   - **build**: energy rises by more than 0.15 inside the section;
   - **drop**: the section's start contrasts strongly with the quietest of the 2 bars before it, *and* the section is above median energy.
8. Count the gap bars before each drop.

**Results:**

- **Test track:** intro 0 · **build 16.02** · **drop 32.00** · outro 52.00. All correct.
- **Doomsday:**
  - 92% of boundaries land on the 4-bar grid.
  - Drops at **96.02** (confidence 0.65), **172.82** (0.60) and **211.22** (**0.81**, with one gap bar before).
  - Build 153.62–172.82; breakdown 134.42–153.62.
  - Plausible, but 172.82 is debatable.
- **Ever Now:** boundaries at 23.04, 36.34, 60.82 and 71.92 are right, but the labels are wrong: the breakdown comes out as "high" and the peak as "build", because the prototype's energy score under-weights brightness on a drumless, flat-mastered track. Our existing "intensity" curve (loudness plus brightness) is the better energy measure here, weighted per track by how much each feature varies.
- **Sample signals** (Doomsday):
  - Seconds to the next drop counts down 10.2 → 0 over 201–211.2 s.
  - Tension: 0.06 at 204 s, 0.44 at 208 s, 0.96 at 211 s, then 0 after the hit.
  - Lesson learned: smoothly blending per-bar energy between bars made it start rising *before* the drop (0.65 at 210 s). **Hold bar values as steps, or use frame-level values at boundaries. Never blend across a drop.**

---

## 1. Beat, downbeat and tempo tracking: the state of the art

| Tool | What it is | Accuracy | Licence | Ease of use | For electronic music |
|---|---|---|---|---|---|
| **[Beat This!](https://github.com/CPJKU/beat_this)** ([paper, ISMIR 2024](https://arxiv.org/abs/2407.21658)) | Transformer model; beats and downbeats without needing the smoothing step | State of the art without smoothing (e.g. Ballroom beat/downbeat F1 97.5/95.3) | **MIT** (code and weights) | `pip install beat-this`, GPU, very fast; optional smoothing needs madmom | Excellent with smoothing; without it, can double the tempo in fast passages |
| **[madmom](https://github.com/CPJKU/madmom)** | The classic RNN beat and downbeat tracker plus smoothing (DBN) | Long-time reference; **downbeat F1 only 0.669 on EDM**, against 0.965 for All-In-One trained on EDM ([Raveform](https://transactions.ismir.net/articles/10.5334/tismir.288)) | Code BSD; **models non-commercial** | No Windows install; needs compiling | Its smoothing code is useful |
| **[BeatNet](https://github.com/mjhydri/BeatNet)** | Real-time tracker | Below offline state of the art | **[unverified]** | Python | Only useful for live input |
| **[All-In-One](https://github.com/mir-aidj/all-in-one)** (`allin1`) | Separates stems, then a transformer: beats, downbeats, tempo, boundaries, labels | Trained on pop. A 2026 version trained on EDM reaches beat F1 0.991, downbeat F1 0.965. | **MIT** | Needs NATTEN and madmom, both of which need compiling on Windows (workarounds saved locally) | Very good grid; pop labels don't fit EDM. **The EDM-trained version is announced but not released.** |
| **[Masked diffusion beat tracking](https://arxiv.org/abs/2608.04624)** (ISMIR 2026, by the Beat This! authors) | Fixes "consecutive downbeats and erratic tempo changes" without any smoothing step | Beats earlier approaches | Code **[not found]** | – | Likely the next upgrade |
| BeatFM, [BeatFCOS](https://arxiv.org/abs/2510.14391) (2025) | Newer trackers | Competitive | No code / unknown | – | Watch list |
| **[Essentia](https://essentia.upf.edu/)** | Music analysis toolkit, many models | Good for tempo | **AGPL**; models non-commercial | Awkward on Windows | Mainly useful for its emotion models (§4) |
| **[Demucs / HTDemucs](https://github.com/facebookresearch/demucs)** | Splits a track into drums, bass, other and vocals | Strong (newer RoFormer models separate better) | **MIT** (maintained fork: [adefossez/demucs](https://github.com/adefossez/demucs)) | `pip install demucs`, GPU, 3.3 s per 5-minute track | Separation artefacts don't matter for loudness curves |

**Common failure modes** ([2026 failure analysis](https://arxiv.org/abs/2605.12287)):
- half or double tempo;
- losing continuity;
- confident but wrong;
- madmom's default minimum of 55 BPM forces double tempo on slow tracks.

**Recommendation:**
1. Beat This! (`final0`) with smoothing, 4 beats per bar, tempo range 70–180.
2. Fit a constant-tempo grid (or a few constant pieces).
3. Fine-tune each beat to the nearby kick or bass hit, within ±30 ms.
4. All-In-One as a second opinion, flagging where they disagree.

---

## 2. Structure, sections, drops and builds

### Models

- **All-In-One** ([repo](https://github.com/mir-aidj/all-in-one), [paper](https://arxiv.org/abs/2307.16425)): boundaries, pop labels, and frame-by-frame confidence curves (useful for voting). Boundaries are good and phrase-aligned on Doomsday; the labels don't suit EDM.
- **SongFormer** ([paper](https://arxiv.org/abs/2510.02797), [code](https://github.com/ASLP-lab/SongFormer), [Hugging Face](https://huggingface.co/ASLP-lab/SongFormer)):
  - Accuracy on SongFormBench (HarmonixSet subset):

    | Model | Accuracy | Boundaries within ±0.5 s | Boundaries within ±3 s |
    |---|---|---|---|
    | SongFormer | **0.807** | **0.696** | 0.780 |
    | All-In-One | 0.740 | 0.596 | 0.730 |
    | Gemini 2.5 Pro | 0.748 | 0.423 | **0.813** |

  - Code CC-BY-4.0, **but its MuQ component is non-commercial.**
  - Fast. Its "chorus" lined up best with the drops in our tracks.
- **EDMFormer / EDM-98** ([paper, March 2026](https://arxiv.org/abs/2603.08759), [code](https://github.com/25ohms/EDM-98)):
  - SongFormer retrained on 98 EDM tracks, with EDM labels: intro, buildup, drop, breakdown, outro.
  - Also uses MuQ, so non-commercial.
  - It found the test track's build and drop correctly but calls too many things "drop" on other styles.
- **Raveform** ([TISMIR, April 2026](https://transactions.ismir.net/articles/10.5334/tismir.288), [site](https://mir-aidj.github.io/raveform/)):
  - 1,423 EDM tracks annotated by experts, with an EDM vocabulary.
  - Key findings:
    - boundaries fall on multiples of 4 or 8 bars;
    - the most common section length is **16 bars**, then 8 and 32;
    - the typical shape is intro → buildup → drop → breakdown → drop → cooldown → outro.
  - **The trained model isn't released yet.**
- **[MSAF](https://github.com/urinieto/msaf):** classic unsupervised structure analysis. Its "novelty" idea, applied to bar-by-bar stem features, is what found Ever Now's 36.34 and 60.82 s boundaries.
- **[CUE-DETR](https://github.com/ETH-DISCO/cue-detr)** ([ISMIR 2024](https://arxiv.org/abs/2407.06823)): predicts DJ cue points, which are phrase-aligned. Trained on 21k cue points from 4,710 EDM tracks; code and checkpoints are public. Could be another phrase-start voter. **Not tested.**
- **Drop detection:**
  - [Yadati et al., 2014](https://archives.ismir.net/ismir2014/paper/000297.pdf) is the classic approach.
  - [drop-detector](https://github.com/felixmeyer6/drop-detector) (bass envelope plus features plus XGBoost) claims F1 0.95 **[unverified; GPL]**.

### Phrase grid

Electronic music is built in 4-, 8-, 16- and 32-bar phrases. The method used in the prototype:
1. Collect boundary candidates.
2. Snap them to downbeats.
3. Choose the 8-bar alignment that most of them agree with.
4. Pull in candidates that are one bar off when the grid bar also has support. This fixed Doomsday's 93.62 → 96.02, where 93.62 is the drum fill.
5. Output signals: phrase number, bar within phrase, and position through the phrase (0–1).

### Build cues to detect (all easy with stems)

| Cue | How to measure it | Seen in our tracks |
|---|---|---|
| Drums or kick removed | Drums stem loudness falls over the phrase | Doomsday 201.6–208.8 s |
| Bass/sub dropout ("the gap") | Sub-bass near zero for 1–2 bars before the hit | Doomsday 208.8 s |
| Riser / white-noise sweep | High-band noisiness and brightness rising | – |
| Snare roll speeding up | Drum hits per beat doubling (1 → 2 → 4) | The test track |
| Filter sweep | Brightness of the "other" stem steadily rising | – |
| Drum fill in a phrase's last bar | Spike in drum activity the bar before a phrase start | Doomsday 93.62 s |
| Pitch rise | Pitch tracking on the "other" stem | Not tested |

**No 2025–26 open model was found specifically for detecting risers or snare rolls.** Hand-built stem features are the practical route.

---

## 3. AI models that "understand" music

| Model | Type | Gives timestamped structure? | Licence | Notes |
|---|---|---|---|---|
| [MERT](https://huggingface.co/m-a-p/MERT-v1-330M) | Music encoder (features only) | No | **Non-commercial** | Good features for change detection |
| [MusicFM](https://github.com/minzwon/musicfm) | Music encoder | No | Code MIT/Apache **[weights unverified]** | Used inside SongFormer and EDMFormer |
| [MuQ / MuQ-MuLan](https://github.com/tencent-ailab/MuQ) | Music encoder; music-text matching | No | Weights **non-commercial** | |
| [CLAP (LAION)](https://github.com/LAION-AI/CLAP) | Audio-text matching | Only in multi-second windows | **[unverified]** | Coarse |
| [Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner) | General audio LLM (30B) | Weak: within 3 s about 62% of the time on [MusTBench](https://arxiv.org/abs/2605.29300); worse after 120 s | **Apache-2.0** | Recommends clips of 30 s or less |
| [Gemini 2.5 / 3](https://ai.google.dev/gemini-api/docs/audio) | API | Accepts MM:SS timestamps; 0.42 within ±0.5 s, 0.81 within ±3 s | Commercial API | Best general LLM at ±3 s; useless at ±0.5 s. Users report timing got worse in Gemini 3. |
| GPT audio models | API | Within 3 s only 14–22% of the time | Commercial | Poor at timing |
| [Music Flamingo](https://huggingface.co/nvidia/music-flamingo-hf) (NVIDIA) | Music LLM, full tracks up to about 10 minutes | Better at structure reasoning | **Non-commercial** | Tested on 80 GB GPUs; 4090 **[unverified]** |
| [LLark](https://github.com/spotify-research/llark) | Music LLM | – | **No released weights** | |
| MusT / [TEMPO (EMNLP 2026)](https://arxiv.org/abs/2608.29999) | Tuned for timestamps | Better, but still accurate to seconds, not beats | **[unverified]** | Improving |

**Short answer:** none of them reliably gives beat-accurate drops, builds or tension curves for a whole track. Errors are measured in seconds, and they get worse late in long tracks.

How to use them anyway:
1. **Never ask for free-form timestamps.** Give the model our bar grid and the candidate boundaries ("bars 88–96 start at 211.22 s…"). Ask it to *classify* each one, rate energy and tension from 1 to 10, and describe what changes. Get JSON back.
2. **Work phrase by phrase:** 16–32-bar windows with one phrase of overlap, stating the absolute times.
3. **Snap everything** to our downbeats.
4. **Or use their features instead of their words:** music-encoder features feeding change detection, or a small classifier trained on a few dozen hand-labelled tracks. This is the most promising "learned" route for our mix of genres.

---

## 4. Tension, energy and anticipation

- **Tension and energy are different curves.** In [Turrell et al., *Music Perception* 2025](https://kar.kent.ac.uk/106400/), 34 listeners continuously rated tension through EDM breakdowns.
  - Tension *rises during builds* and *falls after the drop*, with a **delay of about 2–3 s**.
  - Higher tension went with stronger arousal and emotion, and with less pleasant feelings.
  - [Solberg's studies](https://eprints.whiterose.ac.uk/145911/) show crowds react strongly and consistently to breakdown → build → drop.
  - For the visuals: the wind-up follows tension (peaking on the last beat before the drop), and the spurt follows the jump in energy on the drop's downbeat. Let tension fade over about 2–4 s after the hit instead of snapping to zero.
- **A practical tension formula for EDM**, after [Farbood 2012](https://bpb-us-e1.wpmucdn.com/wp.nyu.edu/dist/f/11865/files/2020/08/Farbood_2012_Musical_Tension.pdf): `tension = w1·rising brightness or riser + w2·rising hit density / snare roll + w3·kick or bass removed while the phrase continues + w4·position in the phrase before the expected drop`. Normalise it per track and release it at the drop.
- **Emotion-recognition models** (valence and arousal over time) are mostly trained on pop and classical, not EDM, and are driven by loudness, brightness and tempo, which we already compute. They probably wouldn't add much over stem-based energy **[judgement, not tested]**.
- **"Energy" in products:**
  - Spotify's audio analysis has been **closed to new apps since November 2024**.
  - Mixed In Key's energy level listens for hi-hat patterns and white-noise risers, not just loudness.
  - Our "intensity" curve is the right idea; weight its parts per track by how much each varies.
- **Anticipation** (we know the future):
  - Signals: seconds to the next drop, bars to the next drop, seconds to the next phrase, bar within the phrase, build progress (0–1).
  - The wind-up starts at the build's start, or 4–8 bars before the drop if there's no build.
  - The prototype uses `max(build_progress^1.5, (1 − t/4 bars)^2)`.

---

## 5. What real products and professionals do

- **rekordbox phrase analysis:**
  - Each track gets a mood (High, Mid or Low) and phrase types (Intro, Up, Down, Chorus, Outro, and so on).
  - Phrases are stored *as beat numbers*, with a **fill-in flag and the beat where the fill starts** ([format spec](https://github.com/Deep-Symmetry/crate-digger/blob/main/src/main/kaitai/rekordbox_anlz.ksy)). That's effectively grid + phrase + fill = an anticipation cue.
  - If you own rekordbox, its analysis files could be another voter (via [pyrekordbox](https://github.com/dylanljones/pyrekordbox), **not tested**).
- **SoundSwitch:**
  - Detects intros and outros, main and bridge sections, and **drops, breakdowns and build-ups**.
  - Assigns lighting roles (washes, spots, strobes) and has 32 beat-synced loops for unscripted tracks.
  - **Built into rekordbox as of 24 September 2026.**
- **MaestroDMX** claims lighting driven by musical structure rather than volume. Its methods aren't documented **[marketing claims]**.
- **Resolume** has no reliable automatic beat detection beyond simple 4/4 kicks ([forum](https://resolume.com/forum/viewtopic.php?p=70304)).
- **[Synesthesia's](https://app.synesthesia.live/docs/ssf/audio_uniforms.html) signals are worth copying as node types:**
  - *Intensity*: slowly building energy.
  - *Presence* and *bass presence*: rising and falling activity that ignores single hits.
  - *Hits*: spikes per frequency band.
  - *On-beat*: 1.0 on the beat, then fading.
  - *Beat twitcher*: a beat clock that eases into each beat.
- **TouchDesigner's [audioAnalysis](https://derivative.ca/UserGuide/Palette:audioAnalysis):** low, mid and high levels, kick, snare, rhythm, brightness.
- **How lighting designers think** ([on drop cues](https://www.ticketfairy.com/blog/kinetic-festival-lighting-without-chaos-movement-cues-for-drops), [EDC lighting designer](https://www.westword.com/music/meet-steve-lieberman-the-man-behind-the-lights-at-edc-las-vegas-6830660/)):
  - **black out or pull back in the gap before the drop**, so the hit "feels twice as loud";
  - keep some contrast so strobes punch;
  - change looks on phrase boundaries, not randomly;
  - hits must be exactly on time, because early or late looks like a miss.
- **How precise sync needs to be:** visuals arriving *after* the sound look worse than visuals arriving slightly before. The broadcast standard ITU-R BT.1359 puts the limits around +45 ms (audio early) and −125 ms (audio late) **[unverified]**.
  - In the browser, time flashes against the audio's actual output time, and lead by 0–20 ms.
  - Our current +16–30 ms late beat bias works against us.

---

## 6. Recommended pipeline (offline, per track)

### Steps

1. **Decode** the audio exactly as the browser plays it. AAC/m4a files start with a built-in delay of about 48 ms (2112 samples). Check once with a click track that the analysis and the browser agree.
2. **Separate stems** with HTDemucs: 3.3 s per 5-minute track, MIT. Keep them and reuse them for All-In-One.
3. **Beat grid:**
   1. Beat This! with smoothing (4/4, 70–180 BPM), and All-In-One as a second grid.
   2. Fit constant-tempo pieces, splitting where the error exceeds about 40 ms.
   3. Extend the grid into drumless intros and outros.
   4. Fine-tune each beat to the kick or bass hit within ±30 ms.
   - Outputs: beats, downbeats, tempo pieces, and a grid confidence (how well the two agree, and how much drum energy there is).
   - **If confidence is low** (tracks like Bliss), switch to a *no-grid mode*: use bass and "other" hits instead of beats, sections by time, and no beat-locked effects.
4. **Boundary candidates:**
   - All-In-One (with its confidence curve), SongFormer, and EDMFormer if the non-commercial licence is acceptable.
   - Bar-by-bar change detection on the stem features (loudness, sub, drums, bass, other, high band, brightness, noisiness).
   - Optionally CUE-DETR, rekordbox or MSAF.
5. **Vote and snap:** each source adds 1 vote to the nearest downbeat and 0.5 to its neighbours. Keep peaks with at least 2 votes.
6. **Phrase grid:**
   - Choose the 8-bar alignment that gets the most votes, and pull in boundaries that are one bar off.
   - Outputs: phrase starts, bar within phrase, position through the phrase.
   - Allow 4-bar and odd-length phrases; Ever Now had a 10-bar section.
7. **Label sections from the stems** (bar features, weighted per track):
   - **drop:** a strong jump from the quietest of the 2 bars before it, then sustained high energy;
   - **build:** energy rising inside the section, *or* build cues present;
   - **breakdown:** low energy between highs;
   - **intro / outro.**
   - Optionally, ask an AI model to classify the *given* sections.
   - Each drop gets: time (a downbeat), confidence, contrast, gap bars before it, and whether a fill comes first.
8. **Continuous curves**, 20–60 per second:
   - energy (our intensity curve, held per bar, never blended across boundaries);
   - tension (build cues plus build progress, fading over 2–3 s after the drop);
   - build progress;
   - seconds to the next drop, phrase and boundary;
   - per-stem curves and hits: kick from the drums stem, sub, hats.
9. **Checking view** in the editor: overlay all sources and flag where they disagree ("models disagree around 54–63 s").

### Suggested additions to `analysis.json`

```json
{
  "grid": {"beats": [...], "downbeats": [...], "tempo_segments": [{"start":0,"end":305,"bpm":100.0,"t0":0.011}], "confidence": 0.97},
  "phrases": {"bars_per_phrase": 8, "starts": [0.0, 19.22, 38.42, ...]},
  "sections": [{"start": 201.6, "end": 211.22, "label": "build", "energy": 0.55, "cues": ["drum_fade", "gap_bar"]},
               {"start": 211.22, "end": 230.40, "label": "drop", "energy": 0.95}],
  "drops": [{"t": 211.22, "confidence": 0.81, "contrast": 0.42, "gap_bars_before": 1, "fill_before": false}],
  "curves": {"rate": 20, "energy": [...], "tension": [...], "build_progress": [...],
             "seconds_to_next_drop": [...], "seconds_to_next_phrase": [...]},
  "stems": {"rate": 50, "kick": [...], "bass": [...], "sub": [...], "other_brightness": [...]}
}
```

### How it maps onto our visuals (as new node types)

- **Sun spurt:** a push at each drop's time, scaled by its confidence and contrast, optionally 0–20 ms early, followed by our inertia. Smaller pushes on phrase starts with a big enough energy jump.
- **Wind-up:** hold the sun back or dim it as tension rises, speeding up over the last 2 bars, and black out during the gap bars.
- **Lasers:** gated on beats and downbeats. Double the rate when the build's hit rate doubles; change pattern at phrase starts.
- **MRI scan:** sweep once per bar or per phrase, so each sweep lands on "the 1".

### Speed and licences

| Component | Time per 5-minute track (4090) | Licence |
|---|---|---|
| HTDemucs | 3–4 s | MIT |
| Beat This! (+ smoothing) | about 1 s (+2–8 s) | MIT (madmom's smoothing code is BSD) |
| All-In-One | about 15 s with stems reused | MIT (Windows workarounds saved locally) |
| SongFormer | about 2 s (+25 s first load) | CC-BY-4.0, **plus non-commercial MuQ** |
| EDMFormer | about 3 s if kept loaded **[estimate]** | **Plus non-commercial MuQ** |
| Stem features, voting, curves | about 5 s | ours |
| **Total** | **about 30–60 s per track** | |

If this could ever be commercial, drop SongFormer and EDMFormer and keep All-In-One, stem change detection and our own labelling.

### Pitfalls

- **Half or double tempo:** use smoothing with a tempo range, and cross-check against All-In-One.
- **Tempo changes:** use constant pieces, not one forced line.
- **Drumless intros:** every tracker locks onto off-beat hats. Extend the confident grid backwards.
- **Beatless tracks** (like Bliss): detect them from drum energy and grid confidence, and switch modes.
- **Pop labels:** use the models' boundaries with our own labels.
- **Drops that aren't kick events:** Doomsday's 211 s drop is driven by brightness and the lead. Include the high band and the "other" stem.
- **Boundaries a bar early or late** between models: use the phrase grid, and treat a fill bar as a lead-in.
- **Blending curves across a boundary** gives the drop away early. Hold values as steps.
- **Delays to account for:** the m4a start delay, Beat This!'s 20 ms steps, our 16–30 ms late bias, and the browser's audio output delay.
- **Windows environments:** installing other packages can silently swap CUDA torch for a CPU version. Pin it.

---

## 7. What couldn't be verified

- **Audio LLMs:** none were run. Their numbers come from benchmarks and model cards.
- **Unchecked details:**
  - the claim that madmom's smoothing is patented;
  - several model licences;
  - how SoundSwitch, MaestroDMX and Mixed In Key work inside;
  - the broadcast sync limits.
- **The prototype's labels:** tuned by hand on these four tracks only. They illustrate the method rather than validate it, and the Ever Now labels came out wrong.
- **The Windows workarounds:** All-In-One ran on them, not the official compiled libraries. Outputs look right but weren't compared bit for bit against a Linux install.
