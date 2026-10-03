"""Audio analysis: one track -> <track>.analysis.json next to it. Run once per track, ahead of time.

    python tools/audio_analysis.py tracks/song.mp3    (any audio, or a video's audio track: decoded with ffmpeg)

The file holds:
    beats       beat times, seconds
    downbeats   bar starts (assumes 4/4; phase picked by where the low end hits hardest)
    loudness    0..1 perceived loudness at `rate` values per second
    bass        0..1 energy under ~150 Hz, same rate
    hats        [time, strength] of high-frequency hits (hats, shakers), for flicker
    sections    [{start, end, kind}] with kind = quiet | build | drop | normal
"""

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage import uniform_filter1d

SR = 22050
HOP = 512
BEAT_HOP = 128  # beat periods are whole frames: 128 samples keeps tempo error under 0.5%
LOUDNESS_RATE = 20  # values per second in the output


def load_audio(path: Path) -> np.ndarray:
    """Decode any audio or video file to mono float32 at SR, through the ffmpeg bundled with imageio-ffmpeg."""
    import subprocess

    import imageio_ffmpeg

    raw = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
        check=True,
        capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def analyse(path: Path) -> dict:
    y, sr = load_audio(path), SR
    duration = len(y) / sr
    frame_rate = sr / HOP

    # Beats from the onset envelope.
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=BEAT_HOP)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=BEAT_HOP, trim=False)
    beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=BEAT_HOP)

    # Bass energy (under ~150 Hz): kicks and bass lines. Drops are mostly the low end coming back.
    mel = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=HOP, fmax=150, n_mels=8)
    bass_db = librosa.power_to_db(mel.sum(axis=0), ref=np.max)
    bass = np.clip((bass_db + 50) / 50, 0, 1)

    bass_onset = np.maximum(0, np.diff(bass, prepend=bass[0]))

    # The tracker can lock onto off-beat hats. Shift all beats by the offset (within half a beat)
    # that best lines them up with bass hits.
    if len(beats) >= 8:
        period = float(np.median(np.diff(beats)))
        bass_t = librosa.frames_to_time(np.arange(len(bass_onset)), sr=sr, hop_length=HOP)
        onset_t = librosa.frames_to_time(np.arange(len(onset)), sr=sr, hop_length=BEAT_HOP)
        norm = lambda a: a / (a.max() + 1e-9)
        offsets = np.arange(-period / 2, period / 2, 0.005)
        bass_score = np.array([np.interp(beats + o, bass_t, norm(bass_onset)).sum() for o in offsets])
        # Bass decides when it has a clear peak; otherwise fall back to all onsets (hats would outvote kicks).
        if bass_score.max() > 1.5 * np.median(bass_score):
            score = bass_score
        else:
            score = np.array([np.interp(beats + o, onset_t, norm(onset)).sum() for o in offsets])
        beats = np.clip(beats + offsets[int(np.argmax(score))], 0, duration)

        # Fine timing: the bass search works on 23 ms frames. Shift every beat by the median distance
        # to the nearest onset peak (5.8 ms frames), so beats land on the attack, not just near it.
        to_peak = []
        for b in beats:
            m = (onset_t > b - 0.06) & (onset_t < b + 0.06)
            if m.any():
                to_peak.append(onset_t[m][np.argmax(onset[m])] - b)
        if to_peak:
            beats = np.clip(beats + float(np.median(to_peak)), 0, duration)

    # Downbeats: the beat phase (of 4) that best starts bars. Bars start with a bass hit and a harmony
    # change; snares (beats 2 and 4) carry low end too, so broadband hits count against a phase.
    from scipy.ndimage import maximum_filter1d

    norm = lambda a: a / (np.max(a) + 1e-9)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    harmony = norm(np.r_[0, np.linalg.norm(np.diff(uniform_filter1d(chroma, 9, axis=1), axis=1), axis=0)])
    high = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP, fmax=None, feature=librosa.feature.melspectrogram, n_mels=32, fmin=2000)
    beat_idx = librosa.time_to_frames(beats, sr=sr, hop_length=HOP).clip(0, len(bass_onset) - 1)
    near = lambda a: maximum_filter1d(a, 5)[beat_idx[: len(beat_idx)]]  # strongest within +-2 frames
    n = min(len(bass_onset), len(harmony), len(high))
    per_beat = near(norm(bass_onset[:n])) + near(harmony[:n]) - 0.5 * near(norm(high[:n]))
    phase = int(np.argmax([per_beat[p::4].mean() for p in range(4)])) if len(beats) >= 8 else 0
    downbeats = beats[phase::4]

    # Loudness: RMS in dB over a 40 dB window below the peak, mapped to 0..1.
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    db = librosa.amplitude_to_db(rms, ref=np.max)
    loud = np.clip((db + 40) / 40, 0, 1)
    loud = uniform_filter1d(loud, size=max(1, int(frame_rate * 0.1)))  # 100 ms smoothing
    times = np.arange(0, duration, 1 / LOUDNESS_RATE)
    frame_times = librosa.frames_to_time(np.arange(len(loud)), sr=sr, hop_length=HOP)
    loudness = np.interp(times, frame_times, loud)
    bass_curve = np.interp(times, frame_times, uniform_filter1d(bass, size=max(1, int(frame_rate * 0.1))))

    hats = find_hats(y, sr)
    sections = find_sections(loudness, bass_curve, LOUDNESS_RATE, duration)
    # A drop starts on its first big hit: move each drop to the beat with the strongest bass attack
    # within a beat of where the loudness curves put it (smoothing makes those land a little late).
    attack = maximum_filter1d(norm(bass_onset), 5)
    beat_period = float(np.median(np.diff(beats))) if len(beats) > 1 else 0.5
    for sec in sections:
        if sec["kind"] == "drop" and len(beats):
            near_beats = beats[np.abs(beats - sec["start"]) <= 1.01 * beat_period]
            if len(near_beats):
                idx = librosa.time_to_frames(near_beats, sr=sr, hop_length=HOP).clip(0, len(attack) - 1)
                sec["start"] = round(float(near_beats[np.argmax(attack[idx])]), 3)

    # Snap section boundaries to the nearest beat (beats are accurate; bar starts are a guess).
    for sec in sections[1:]:
        if len(beats):
            nearest = float(beats[np.argmin(np.abs(beats - sec["start"]))])
            if abs(nearest - sec["start"]) <= 0.6 * np.median(np.diff(beats)):
                sec["start"] = round(nearest, 3)
    for a, b in zip(sections, sections[1:]):
        a["end"] = b["start"]
    return {
        "version": 1,
        "track": path.name,
        "duration": round(duration, 3),
        # From the actual beat spacing: the tracker's own tempo figure is rounded to its frame grid.
        "tempo": round(60 / float(np.median(np.diff(beats))), 2) if len(beats) > 1 else round(float(np.atleast_1d(tempo)[0]), 2),
        "beats": [round(float(b), 3) for b in beats],
        "downbeats": [round(float(b), 3) for b in downbeats],
        "loudness": {"rate": LOUDNESS_RATE, "values": [round(float(v), 3) for v in loudness]},
        "bass": {"rate": LOUDNESS_RATE, "values": [round(float(v), 3) for v in bass_curve]},
        "sections": sections,
        "hats": hats,
    }


def find_hats(y: np.ndarray, sr: int) -> list[list[float]]:
    """High-frequency hits (hats, shakers, cymbal ticks): [time, strength 0..1], for flicker effects."""
    high = librosa.onset.onset_strength(y=y, sr=sr, hop_length=BEAT_HOP, fmin=5000, n_mels=32)
    if high.max() <= 0:
        return []
    high = high / np.percentile(high[high > 0], 99)
    # At least ~40 ms apart, and clearly above the local average.
    peaks = librosa.util.peak_pick(high, pre_max=3, post_max=3, pre_avg=12, post_avg=12, delta=0.15, wait=7)
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=BEAT_HOP)
    return [[round(float(t), 3), round(float(min(1.0, high[p])), 2)] for t, p in zip(times, peaks) if high[p] > 0.2]


def find_sections(loudness: np.ndarray, bass: np.ndarray, rate: int, duration: float) -> list[dict]:
    """Label quiet, build and drop stretches from the slow loudness and bass envelopes."""
    energy = uniform_filter1d(loudness, size=2 * rate)  # 2 s window
    lo, hi = np.percentile(energy, [10, 90])
    span = max(hi - lo, 1e-3)
    labels = np.full(len(energy), "normal", dtype=object)
    labels[energy < lo + 0.3 * span] = "quiet"

    # Drops: the bass jumps within about a second and the track lands loud.
    low = uniform_filter1d(bass, size=rate)
    after = np.roll(low, -rate)
    before = np.maximum.reduce([np.roll(low, k) for k in range(rate, 3 * rate, rate // 2)])
    jump = after - before
    jump[: 3 * rate] = jump[-rate:] = 0
    candidates = np.where((jump > 0.15) & (np.roll(energy, -rate) > lo + 0.6 * span))[0]
    drops = []
    for i in candidates:
        if not drops or i - drops[-1] > 8 * rate:
            # The drop lands where the bass rises most over a quarter second in this stretch.
            q = max(1, rate // 4)
            window = np.arange(max(0, i - rate), min(len(bass) - q, i + 2 * rate))
            drops.append(int(window[np.argmax(bass[window + q] - bass[window])]) + 1)
    for d in drops:
        # Drop lasts while energy stays in the upper half. Start checking a second in:
        # the 2 s energy window still remembers any silence just before the drop.
        end = min(len(energy), d + rate)
        while end < len(energy) and energy[end] > lo + 0.5 * span:
            end += 1
        labels[d:end] = "drop"
        # Build: back from the drop to the last low point, up to 16 s. Skip the last 2 s:
        # producers often cut to near-silence right before a drop, which is part of the build.
        start, stop = max(0, d - 16 * rate), max(0, d - 2 * rate)
        low_point = start + int(np.argmin(energy[start:stop])) if stop > start else d
        labels[low_point:d] = "build"

    # Run-length encode, then fold stretches under 2 s into the previous one.
    sections: list[dict] = []
    for i, kind in enumerate(labels):
        if sections and sections[-1]["kind"] == kind:
            continue
        sections.append({"start": round(i / rate, 2), "kind": kind})
    merged: list[dict] = []
    for s in sections:
        if merged and (s["start"] - merged[-1]["start"] < 2 and merged[-1]["kind"] not in ("drop", "build")):
            merged[-1]["kind"] = s["kind"]
            continue
        if merged and merged[-1]["kind"] == s["kind"]:
            continue
        merged.append(s)
    for a, b in zip(merged, merged[1:] + [{"start": round(duration, 2)}]):
        a["end"] = b["start"]
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tracks", type=Path, nargs="+")
    args = ap.parse_args()
    for track in args.tracks:
        result = analyse(track)
        out = track.with_name(track.name + ".analysis.json")
        out.write_text(json.dumps(result, separators=(",", ":")))
        summary = ", ".join(f"{s['kind']} {s['start']:.1f}-{s['end']:.1f}" for s in result["sections"])
        print(f"{track.name}: {result['tempo']} BPM, {len(result['beats'])} beats -> {out.name}\n  {summary}")


if __name__ == "__main__":
    main()
