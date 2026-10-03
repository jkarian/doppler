"""Music structure for the analysis file: stems, a steady beat grid, phrases, sections, drops, and
anticipation curves. Used by audio_analysis.py. See docs/research-music-understanding.md for the why.

Pipeline (all offline, about 10 s per 5-minute track on the 4090):
    1. HTDemucs splits the track into drums, bass, other and vocals.
    2. Beat This! finds beats and downbeats. We fit constant-tempo pieces to them (electronic music
       keeps a steady tempo), which removes the tracker's wobble and doubled-tempo stretches, extend
       the grid into drumless intros and outros, and line it up with the kick in the drums stem.
    3. Bar-by-bar features from the stems give a novelty curve (how much the music changes at each
       bar). Boundaries are its peaks, with a strong preference for the 8-bar phrase grid.
    4. Each section gets a label from its energy (weighted per track by how much each feature varies)
       and its build cues; drops get a time, confidence, contrast, gap bars and fill flag.
    5. Curves at CURVE_RATE: energy (held per bar, never blended across a boundary), tension (rises
       through a build, releases over ~2.5 s after the drop), build progress, seconds to the next
       drop and to the next phrase.

If the grid can't be trusted (no drums, like Bliss) `grid.confident` is false: beat-locked effects
should back off. Sections still come from the features, on a 4-beat grid at the estimated tempo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "shims"))  # torchaudio stand-in for beat_this

STEM_SR = 44100
FEAT_SR = 22050
FEAT_HOP = 441  # 50 feature frames per second
FPS = FEAT_SR / FEAT_HOP
CURVE_RATE = 20
BARS_PER_PHRASE = 8
# The kick detector (onset strength on the drums stem) peaks this long after the attack: measured on
# tracks/test-120bpm.wav, where the kicks are exactly on 0.5 s steps.
ONSET_LAG = 0.006
OFF_GRID = 1.0  # novelty (in units of the track's 90th percentile) a boundary off the 4-bar grid needs


# --- 1. Stems ---------------------------------------------------------------------------------

def separate(stereo44: np.ndarray, device: str) -> dict[str, np.ndarray]:
    """HTDemucs stems as mono float32 at FEAT_SR: drums, bass, other, vocals."""
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    model = get_model("htdemucs").to(device).eval()
    wav = torch.tensor(stereo44.T, dtype=torch.float32, device=device)  # (2, T)
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        out = apply_model(model, ((wav - mean) / std)[None], device=device, split=True, overlap=0.25, progress=False)[0]
    out = out * std + mean
    stems = {}
    for name, src in zip(model.sources, out):
        mono = src.mean(0).cpu().numpy()
        stems[name] = librosa.resample(mono, orig_sr=STEM_SR, target_sr=FEAT_SR).astype(np.float32)
    del model, out, wav
    if device == "cuda":
        torch.cuda.empty_cache()
    return stems


# --- 2. Beat grid -----------------------------------------------------------------------------

def track_beats(mono22: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    from beat_this.inference import Audio2Beats

    beats, downbeats = Audio2Beats("final0", device=device)(mono22, FEAT_SR)
    return np.asarray(beats, float), np.asarray(downbeats, float)


def _period(beats: np.ndarray) -> float:
    """The beat period: the most common spacing, refined by the median of spacings near it."""
    ioi = np.diff(beats)
    ioi = ioi[(ioi > 60 / 200) & (ioi < 60 / 60)]
    if len(ioi) == 0:
        return 0.5
    hist, edges = np.histogram(ioi, bins=np.arange(0.3, 1.0, 0.005))
    mode = edges[np.argmax(hist)] + 0.0025
    near = ioi[np.abs(ioi - mode) < 0.04 * mode]
    return float(np.median(near)) if len(near) else float(mode)


def _fit_piece(beats: np.ndarray, period: float) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Fit t = t0 + k * p to beats, robust to doubled and missing beats.
    Returns (t0, p, beat index per beat, inlier mask)."""
    anchor = beats[len(beats) // 2]
    t0, p = anchor, period
    inl = np.ones(len(beats), bool)
    for _ in range(6):
        k = np.round((beats - t0) / p)
        res = beats - (t0 + k * p)
        inl = np.abs(res) < 0.2 * p  # half-beat (doubled) and stray beats drop out
        if inl.sum() < 8:
            break
        p, t0 = np.polyfit(k[inl], beats[inl], 1)
    k = np.round((beats - t0) / p)
    return float(t0), float(p), k.astype(int), inl


def fit_grid(beats: np.ndarray, duration: float, max_pieces: int = 4, tol: float = 0.035) -> list[dict]:
    """Constant-tempo pieces {start, end, t0, period}. Splits a piece in two where one line can't
    fit it within `tol` seconds for a sustained stretch (a real tempo change), up to max_pieces."""
    period = _period(beats)

    def worst_run(b: np.ndarray) -> float:
        t0, p, k, inl = _fit_piece(b, period)
        res = np.abs(b - (t0 + k * p))
        res[~inl] = 0
        # Median over 16-beat windows: sustained drift, not single sloppy beats.
        if len(res) < 16:
            return 0.0
        return float(max(np.median(res[i:i + 16]) for i in range(len(res) - 15)))

    pieces = [beats]
    while len(pieces) < max_pieces:
        scores = [worst_run(b) for b in pieces]
        i = int(np.argmax(scores))
        if scores[i] < tol or len(pieces[i]) < 64:
            break
        b = pieces[i]
        # Best split point: least total residual, each side at least 32 beats.
        best, best_cost = None, np.inf
        for s in range(32, len(b) - 32, 4):
            cost = 0.0
            for part in (b[:s], b[s:]):
                t0, p, k, inl = _fit_piece(part, period)
                cost += float(np.sum(np.minimum(np.abs(part - (t0 + k * p)), 0.2 * p) ** 2))
            if cost < best_cost:
                best, best_cost = s, cost
        if best is None:
            break
        pieces[i:i + 1] = [b[:best], b[best:]]

    out = []
    for n, b in enumerate(pieces):
        t0, p, _, _ = _fit_piece(b, period)
        start = 0.0 if n == 0 else float(b[0])
        end = duration if n == len(pieces) - 1 else float(pieces[n + 1][0])
        out.append({"start": start, "end": end, "t0": t0, "period": p})
    return out


def grid_beats(pieces: list[dict]) -> np.ndarray:
    """All grid beats from the pieces, from the track start to its end."""
    beats = []
    for pc in pieces:
        k0 = int(np.ceil((pc["start"] - pc["t0"]) / pc["period"] - 0.5))
        t = pc["t0"] + k0 * pc["period"]
        while t < pc["end"] - 0.5 * pc["period"]:
            if t >= -0.01 and (not beats or t - beats[-1] > 0.5 * pc["period"]):
                beats.append(max(0.0, t))
            t += pc["period"]
    return np.array(beats)


def kick_refit(pieces: list[dict], drums: np.ndarray) -> tuple[list[dict], int]:
    """Refit each tempo piece to the kick attacks in the drums stem. The tracker's beats carry small
    local biases (about +18 ms in builds, early on hats), which tilt the fitted tempo; kicks are exact.
    Uses beats that have a strong low-end attack within +-40 ms of the grid. Returns the pieces and how
    many kicks were used."""
    hop = 64  # 2.9 ms
    onset = librosa.onset.onset_strength(y=drums, sr=FEAT_SR, hop_length=hop, fmax=200, n_mels=16)
    if onset.max() <= 0:
        return pieces, 0
    t = librosa.frames_to_time(np.arange(len(onset)), sr=FEAT_SR, hop_length=hop)
    used = 0
    out = []
    for pc in pieces:
        pc = dict(pc)
        kicks, strengths = [], []
        for b in grid_beats([pc]):
            m = (t > b - 0.04) & (t < b + 0.04)
            if m.any():
                j = int(np.argmax(onset[m]))
                kicks.append(t[m][j])
                strengths.append(onset[m][j])
        if len(kicks) < 24:
            out.append(pc)
            continue
        kicks, strengths = np.array(kicks), np.array(strengths)
        keep = strengths > max(np.percentile(strengths, 50), 0.2 * strengths.max())
        kt = kicks[keep]
        t0, p = pc["t0"], pc["period"]
        inl = np.ones(len(kt), bool)
        for _ in range(4):
            k = np.round((kt - t0) / p)
            inl = np.abs(kt - (t0 + k * p)) < 0.012
            if inl.sum() < 24:
                break
            p, t0 = np.polyfit(k[inl], kt[inl], 1)
        if inl.sum() >= 24:
            pc["t0"], pc["period"] = float(t0) - ONSET_LAG, float(p)
            used += int(inl.sum())
        out.append(pc)
    return out, used


def downbeat_phase(grid: np.ndarray, downbeats: np.ndarray) -> int:
    """Which of every 4 grid beats starts a bar, by vote of the tracker's downbeats."""
    votes = np.zeros(4)
    for d in downbeats:
        i = int(np.argmin(np.abs(grid - d)))
        if abs(grid[i] - d) < 0.07:
            votes[i % 4] += 1
    return int(np.argmax(votes))


# --- 3. Bar features and boundaries -----------------------------------------------------------

def frame_features(mix: np.ndarray, stems: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Feature curves at FPS, raw scale."""
    f = {}
    for name in ("drums", "bass", "other", "vocals"):
        f[name] = librosa.feature.rms(y=stems[name], frame_length=2048, hop_length=FEAT_HOP)[0]
    S = np.abs(librosa.stft(mix, n_fft=2048, hop_length=FEAT_HOP))
    fr = librosa.fft_frequencies(sr=FEAT_SR, n_fft=2048)
    f["loud"] = librosa.feature.rms(S=S)[0]
    f["sub"] = S[(fr >= 25) & (fr < 65)].mean(0)
    f["high"] = S[fr > 4000].mean(0)
    f["centroid"] = librosa.feature.spectral_centroid(S=S, sr=FEAT_SR)[0]
    f["noise"] = librosa.feature.spectral_flatness(S=S[fr > 2000])[0]  # risers, white-noise sweeps
    # Drum hits per frame (for fills and snare rolls).
    f["hits"] = librosa.onset.onset_strength(y=stems["drums"], sr=FEAT_SR, hop_length=FEAT_HOP)
    n = min(len(v) for v in f.values())
    return {k: v[:n] for k, v in f.items()}


def per_bar(feat: dict[str, np.ndarray], bars: np.ndarray, duration: float) -> dict[str, np.ndarray]:
    edges = np.r_[bars, duration]
    n = len(next(iter(feat.values())))
    out = {}
    for k, v in feat.items():
        vals = []
        for a, b in zip(edges[:-1], edges[1:]):
            i, j = int(a * FPS), max(int(a * FPS) + 1, int(b * FPS))
            vals.append(float(v[min(i, n - 1):min(j, n)].mean()) if i < n else 0.0)
        out[k] = np.array(vals)
    return out


def _spread(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a, [5, 95])
    return np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)


def bar_energy(B: dict[str, np.ndarray]) -> np.ndarray:
    """How big each bar feels, 0..1. Each feature is scaled to the track's own range and weighted by
    how much it varies: on a flat-mastered drumless track brightness carries the shape, on a techno
    track the kick and loudness do."""
    keys = {"loud": 1.0, "sub": 1.0, "drums": 0.8, "high": 1.2, "centroid": 1.0, "other": 0.6}
    parts = {k: _spread(B[k]) for k in keys}
    w = {k: keys[k] * (np.std(parts[k]) + 0.05) for k in keys}
    e = sum(w[k] * parts[k] for k in keys) / sum(w.values())
    return _spread(e) if np.ptp(e) > 0 else e


def playing(B: dict[str, np.ndarray]) -> np.ndarray:
    """Bars where music is playing (not the silence before or after it)."""
    return B["loud"] > 0.1 * np.percentile(B["loud"], 95)


def novelty(B: dict[str, np.ndarray], width: int) -> np.ndarray:
    """How different the `width` bars after each bar are from the `width` bars before it.
    Features are standardised over the playing bars only: silence would swamp the differences."""
    keys = ["loud", "sub", "drums", "bass", "other", "vocals", "high", "centroid", "noise"]
    X = np.stack([B[k] for k in keys], 1)
    on = playing(B)
    ref = X[on] if on.sum() > 4 else X
    X = (X - ref.mean(0)) / (ref.std(0) + 1e-9)
    n = len(X)
    nov = np.zeros(n)
    for i in range(1, n):
        a, b = X[max(0, i - width):i], X[i:i + width]
        if len(a) and len(b):
            nov[i] = np.linalg.norm(b.mean(0) - a.mean(0))
    return nov


def find_boundaries(B: dict[str, np.ndarray], nbars: int, E: np.ndarray) -> tuple[list[int], int, np.ndarray]:
    """Section starts (bar indices), the phrase phase (0..7), and the novelty curve.
    Boundaries on the 4-bar grid need a moderate novelty peak; off-grid ones need a strong one
    (odd-length sections exist, e.g. Ever Now's 10 bars). A peak one bar off the grid next to a grid
    bar with support snaps onto the grid (fills and lead-in bars)."""
    nov = 0.6 * novelty(B, 4) + 0.4 * novelty(B, 2)
    nov[0] = 0
    # Scale from the body of the track: full windows, music on both sides. The edges into and out of
    # silence are always huge and would push everything else under the threshold.
    on = playing(B)
    body = [i for i in range(4, nbars - 4) if on[i - 4:i + 4].all()]
    scale = np.percentile(nov[body], 90) if len(body) > 4 else np.percentile(nov[1:], 90) if nbars > 2 else 1.0
    z = nov / max(scale, 1e-9)
    # Phrase phase: the 8-bar alignment that the clear changes (strong novelty peaks and energy jumps)
    # fall on; the half-phrase lines count half.
    jump = np.abs(np.diff(E, prepend=E[0]))
    strength = np.maximum(0, z - 0.5) + jump
    phase = max(range(BARS_PER_PHRASE), key=lambda o: strength[o::BARS_PER_PHRASE].sum() + 0.5 * strength[(o + 4) % 8::BARS_PER_PHRASE].sum())

    def is_peak(i: int) -> bool:
        return z[i] >= z[max(0, i - 1)] and z[i] >= z[min(nbars - 1, i + 1)]

    chosen = set()
    for i in range(1, nbars):
        on_grid = (i - phase) % 4 == 0
        if on_grid and z[i] > 0.55 and (is_peak(i) or z[i] > 0.8):
            chosen.add(i)
    for i in range(1, nbars):
        if (i - phase) % 4 == 0 or not is_peak(i) or z[i] < OFF_GRID:
            continue
        near = [j for j in (i - 1, i + 1) if (j - phase) % 4 == 0 and 0 < j < nbars]
        if any(j in chosen for j in near):
            continue  # the grid bar already has it
        snap = [j for j in near if z[j] > 0.5 * z[i]]
        chosen.add(snap[0] if snap else i)
    # At least 2 bars apart: keep the stronger (grid bars win ties).
    out: list[int] = []
    for i in sorted(chosen):
        if out and i - out[-1] < 2:
            rank = lambda j: ((j - phase) % 4 == 0, z[j])
            if rank(i) > rank(out[-1]):
                out[-1] = i
            continue
        out.append(i)
    return [0] + out, phase, z


# --- 4. Sections and drops --------------------------------------------------------------------

def label_sections(starts: list[int], E: np.ndarray, B: dict[str, np.ndarray], bars: np.ndarray, duration: float) -> tuple[list[dict], list[dict]]:
    nb = len(bars)
    bounds = starts + [nb]
    secs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        e = E[a:b]
        slope = float(np.polyfit(np.arange(len(e)), e, 1)[0]) if len(e) > 2 else 0.0
        secs.append({"bar0": a, "bar1": b, "E": float(e.mean()), "rise": slope * len(e)})
    Es = np.array([s["E"] for s in secs])
    med = float(np.median(np.repeat(Es, [s["bar1"] - s["bar0"] for s in secs])))  # bar-weighted
    sub, drums, hits = _spread(B["sub"]), _spread(B["drums"]), B["hits"]

    def contrast(i: int) -> float:
        return float(E[i:i + 2].mean() - E[max(0, i - 2):i].min()) if i > 0 else 0.0

    drops = []
    for k, s in enumerate(secs):
        i = s["bar0"]
        s["contrast"] = contrast(i)
        if s["contrast"] > 0.18 and s["E"] >= med and s["rise"] < 0.25:
            s["label"] = "drop"
        elif s["rise"] > 0.2 and k + 1 < len(secs):
            s["label"] = "build"
        elif s["E"] < med - 0.12:
            s["label"] = "breakdown"
        else:
            s["label"] = "normal"
    # A "drop" that the music keeps climbing out of within 8 bars was a stage of the build:
    # the real drop is where the climb lands.
    for k in range(len(secs) - 2, -1, -1):
        s = secs[k]
        if s["label"] != "drop":
            continue
        for nxt in secs[k + 1:]:
            if nxt["bar0"] - s["bar0"] > 8 or nxt["label"] not in ("build", "drop"):
                break
            if nxt["label"] == "drop" and nxt["E"] > s["E"] + 0.05:
                s["label"] = "build"
                break
    # Low stretches at the very start and end are the intro and outro.
    for run, name in ((secs, "intro"), (secs[::-1], "outro")):
        for s in run:
            if s["label"] not in ("breakdown", "normal") or s["E"] >= med - 0.12:
                break
            s["label"] = name
    # A section right after a drop at the drop's level is the drop carrying on.
    for k in range(1, len(secs)):
        if secs[k - 1]["label"] == "drop" and secs[k]["label"] == "normal" and secs[k]["E"] >= secs[k - 1]["E"] - 0.1:
            secs[k]["label"] = "drop+"
    # The stretch before a drop that ramps up (or strips down: drums fading, the bass cut) is a build,
    # from where it starts changing: the last phrase-aligned bar inside the previous section.
    for k in range(1, len(secs)):
        if secs[k]["label"] != "drop":
            continue
        prev = secs[k - 1]
        i = secs[k]["bar0"]
        cut = any(sub[j] < 0.15 or drums[j] < 0.25 for j in range(max(prev["bar0"], i - 2), i))
        fading = drums[max(prev["bar0"], i - 4):i].mean() < drums[max(prev["bar0"], i - 8):max(prev["bar0"], i - 4)].mean() - 0.1 if i - prev["bar0"] >= 8 else False
        if prev["label"] != "drop" and (cut or fading or prev["rise"] > 0.1):
            if prev["bar1"] - prev["bar0"] > 8 and prev["label"] != "build":
                # Split: only the last up-to-8 bars are the build.
                b0 = prev["bar1"] - 8
                secs.insert(k, {"bar0": b0, "bar1": prev["bar1"], "E": float(E[b0:prev["bar1"]].mean()), "rise": 0.0, "contrast": 0.0, "label": "build"})
                prev["bar1"] = b0
            else:
                prev["label"] = "build"
    # Merge neighbours with the same label (except drops: back-to-back drops are separate hits).
    merged: list[dict] = []
    for s in secs:
        if merged and (s["label"] == "drop+" or (merged[-1]["label"] == s["label"] and s["label"] != "drop")):
            merged[-1]["bar1"] = s["bar1"]
            continue
        merged.append(dict(s))
    for s in merged:
        s["start"] = 0.0 if s["bar0"] == 0 else float(bars[s["bar0"]])
        s["end"] = float(bars[s["bar1"]]) if s["bar1"] < nb else duration
        s["energy"] = float(E[s["bar0"]:s["bar1"]].mean())
        if s["label"] == "drop":
            i = s["bar0"]
            gap = int(sum(sub[j] < 0.15 for j in range(max(0, i - 2), i)))
            fill = bool(i >= 2 and hits[i - 1] > 1.6 * max(1e-9, np.median(hits[max(0, i - 9):i - 1])))
            conf = float(np.clip(0.25 + 1.4 * s["contrast"] + 0.1 * gap + 0.1 * fill + (0.1 if i % 4 == 0 else 0), 0, 1))
            drops.append({"t": round(s["start"], 3), "bar": int(i), "confidence": round(conf, 2), "contrast": round(s["contrast"], 3),
                          "gap_bars_before": gap, "fill_before": fill})
    sections = [{"start": round(s["start"], 3), "end": round(s["end"], 3), "kind": s["label"], "energy": round(s["energy"], 3)} for s in merged]
    return sections, drops


# --- 5. Curves --------------------------------------------------------------------------------

def curves(duration: float, bars: np.ndarray, E: np.ndarray, sections: list[dict], drops: list[dict], phrase_starts: np.ndarray) -> dict:
    t = np.arange(0, duration, 1 / CURVE_RATE)
    nb = len(bars)
    bar_i = np.clip(np.searchsorted(bars, t, side="right") - 1, 0, nb - 1)
    energy = E[bar_i]  # held per bar: a drop's energy never leaks into the bar before it
    energy[t < bars[0]] = E[0]

    build = np.zeros_like(t)
    for s in sections:
        if s["kind"] == "build":
            m = (t >= s["start"]) & (t < s["end"])
            build[m] = (t[m] - s["start"]) / max(s["end"] - s["start"], 1e-3)

    drop_t = np.array([d["t"] for d in drops])
    nxt = np.searchsorted(drop_t, t, side="right")
    until = np.where(nxt < len(drop_t), drop_t[np.minimum(nxt, len(drop_t) - 1)] - t, -1.0)
    bar_len = float(np.median(np.diff(bars))) if nb > 1 else 2.0
    win = 4 * bar_len
    wind = np.where((until >= 0) & (until < win), (1 - until / win) ** 2, 0.0)
    tension = np.maximum(build ** 1.5, wind)
    # Release: tension falls over ~2.5 s after each drop (listeners' tension lags the hit).
    for d in drop_t:
        m = (t >= d) & (t < d + 2.5)
        x = (t[m] - d) / 2.5
        tension[m] = np.maximum(tension[m], 1 - x * x * (3 - 2 * x))

    pn = np.searchsorted(phrase_starts, t, side="right")
    until_phrase = np.where(pn < len(phrase_starts), phrase_starts[np.minimum(pn, len(phrase_starts) - 1)] - t, -1.0)

    r = lambda a, d=3: [round(float(x), d) for x in a]
    return {
        "rate": CURVE_RATE,
        "energy": r(energy),
        "tension": r(tension),
        "build": r(build),
        "until_drop": r(until, 2),
        "until_phrase": r(until_phrase, 2),
    }


# --- Entry point ------------------------------------------------------------------------------

def analyse_structure(mono22: np.ndarray, stereo44: np.ndarray, duration: float, log=print) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stems = separate(stereo44, device)
    raw_beats, raw_down = track_beats(mono22, device)

    pieces = fit_grid(raw_beats, duration) if len(raw_beats) >= 16 else [{"start": 0.0, "end": duration, "t0": 0.0, "period": 0.5}]
    grid = grid_beats(pieces)
    period = float(np.median([p["period"] for p in pieces]))
    drum_share = float(np.sqrt(np.mean(stems["drums"] ** 2)) / (np.sqrt(np.mean(mono22 ** 2)) + 1e-9))
    before = grid
    kicks = 0
    if drum_share > 0.08:
        pieces, kicks = kick_refit(pieces, stems["drums"])
        grid = grid_beats(pieces)
        period = float(np.median([p["period"] for p in pieces]))
    n = min(len(before), len(grid))
    shift = float(np.median(grid[:n] - before[:n])) if n else 0.0

    # Confidence: does the tracker agree with the grid, and are there drums to hear it by?
    near = np.abs(raw_beats[:, None] - grid[None]).min(1) if len(raw_beats) and len(grid) else np.array([1.0])
    agree = float((near < 0.07).mean())
    # Agreement is loose because every tracker puts drumless intros on the off-beat hats; the fit
    # ignores those, so the grid is still right there. No drums is the real failure (Bliss).
    confident = agree > 0.75 or (agree > 0.5 and drum_share > 0.08)

    phase = downbeat_phase(grid, raw_down) if len(raw_down) else 0
    bars = grid[phase::4]
    feat = frame_features(mono22, stems)
    B = per_bar(feat, bars, duration)
    E = bar_energy(B)
    starts, phrase_phase, nov = find_boundaries(B, len(bars), E)
    sections, drops = label_sections(starts, E, B, bars, duration)
    phrase_starts = bars[phrase_phase::BARS_PER_PHRASE]

    log(f"  grid: {len(pieces)} piece(s), {60 / period:.2f} BPM, {kicks} kicks, shift {shift * 1000:+.0f} ms, agreement {agree:.2f}, "
        f"drums {drum_share:.3f} -> {'confident' if confident else 'NOT confident'}")
    return {
        "beats": grid,
        "downbeats": bars,
        "tempo": 60 / period,
        "grid": {
            "pieces": [{k: round(v, 5) for k, v in p.items()} for p in pieces],
            "confident": confident,
            "agreement": round(agree, 3),
            "drum_share": round(drum_share, 3),
        },
        "phrases": {"bars_per_phrase": BARS_PER_PHRASE, "starts": [round(float(x), 3) for x in phrase_starts]},
        "sections": sections,
        "drops": drops,
        "curves": curves(duration, bars, E, sections, drops, phrase_starts),
        "debug": {"bar_energy": [round(float(x), 3) for x in E], "novelty": [round(float(x), 3) for x in nov]},
    }
