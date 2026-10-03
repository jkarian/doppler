"""Audio analysis: one track -> <track>.analysis.json next to it. Run once per track, ahead of time.

    python tools/audio_analysis.py tracks/song.mp3    (any audio, or a video's audio track: decoded with ffmpeg)

Version 2. The file holds:
    tempo       BPM of the beat grid
    beats       beat times, seconds: a steady grid (Beat This! fitted to constant-tempo pieces, on the kick)
    downbeats   bar starts
    grid        {pieces, confident, agreement, drum_share}: confident is false for drumless tracks
                like Bliss, where beat-locked effects should back off
    phrases     {bars_per_phrase, starts}: the 8-bar phrase grid
    sections    [{start, end, kind, energy}], kind = intro | build | drop | breakdown | normal | outro
    drops       [{t, bar, confidence, contrast, gap_bars_before, fill_before}]
    curves      {rate, energy (held per bar), tension, build, until_drop, until_phrase}: -1 = none ahead
    loudness    0..1 perceived loudness at `rate` values per second
    intensity   0..1 loudness plus brightness (filters opening, high layers): structure of flat-mastered tracks
    sub         0..1 kick and sub-bass (25-65 Hz): drops out before techno drops
    bass        0..1 energy under ~150 Hz, same rate
    hats        [time, strength] of high-frequency hits (hats, shakers), for flicker
    sounds      {name: [[time, strength], ...]}: sounds named by example (see --sound), found wherever they play

Named sounds: point at one instance and name it; it's remembered in tracks/<track>.sounds.json.
    python tools/audio_analysis.py tracks/song.m4a --sound tick@0.615 --sound tock@1.817
    (name@seconds, optionally @stem: drums, bass, other or vocals; default drums)

Structure (grid, phrases, sections, drops, curves) comes from music_structure.py.
"""

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage import uniform_filter1d

from music_structure import STEM_SR, analyse_structure

SR = 22050
HOP = 512
BEAT_HOP = 128  # beat periods are whole frames: 128 samples keeps tempo error under 0.5%
LOUDNESS_RATE = 20  # values per second in the output


def load_audio(path: Path, sr: int = SR, channels: int = 1) -> np.ndarray:
    """Decode any audio or video file to float32 (interleaved if stereo), through the ffmpeg bundled with imageio-ffmpeg."""
    import subprocess

    import imageio_ffmpeg

    raw = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(path), "-vn", "-ac", str(channels), "-ar", str(sr), "-f", "f32le", "-"],
        check=True,
        capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def analyse(path: Path, sounds: list[dict] = ()) -> dict:
    y, sr = load_audio(path), SR
    duration = len(y) / sr
    frame_rate = sr / HOP

    # Bass energy (under ~150 Hz): kicks and bass lines.
    mel = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=HOP, fmax=150, n_mels=8)
    bass_db = librosa.power_to_db(mel.sum(axis=0), ref=np.max)
    bass = np.clip((bass_db + 50) / 50, 0, 1)

    # Loudness: RMS in dB over a 40 dB window below the peak, mapped to 0..1.
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    db = librosa.amplitude_to_db(rms, ref=np.max)
    loud = np.clip((db + 40) / 40, 0, 1)
    loud = uniform_filter1d(loud, size=max(1, int(frame_rate * 0.1)))  # 100 ms smoothing
    times = np.arange(0, duration, 1 / LOUDNESS_RATE)
    frame_times = librosa.frames_to_time(np.arange(len(loud)), sr=sr, hop_length=HOP)
    loudness = np.interp(times, frame_times, loud)
    bass_curve = np.interp(times, frame_times, uniform_filter1d(bass, size=max(1, int(frame_rate * 0.1))))

    # Heavily mastered tracks are flat in loudness; their structure is in brightness (filters opening,
    # high layers coming in) and in the kick cutting out and slamming back. Two more curves for that.
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    band_db = lambda lo, hi: librosa.amplitude_to_db(S[(freqs >= lo) & (freqs < hi)].mean(axis=0) + 1e-10, ref=np.max)

    def spread(a: np.ndarray) -> np.ndarray:  # to 0..1 over this track's own range
        lo, hi = np.percentile(a, [5, 95])
        return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)

    sub = spread(uniform_filter1d(band_db(25, 65), size=max(1, int(frame_rate * 0.1))))
    intensity = 0.35 * spread(db) + 0.4 * spread(band_db(2000, 6000)) + 0.25 * spread(band_db(6000, 11000))
    intensity = uniform_filter1d(intensity, size=max(1, int(frame_rate * 0.5)))
    sub_curve = np.interp(times, frame_times[: len(sub)], sub[: len(frame_times)])
    intensity_curve = np.interp(times, frame_times[: len(intensity)], intensity[: len(frame_times)])

    hats = find_hats(y, sr)
    structure = analyse_structure(y, load_audio(path, sr=STEM_SR, channels=2).reshape(-1, 2), duration, sounds)
    r3 = lambda a: [round(float(b), 3) for b in a]
    return {
        "version": 2,
        "track": path.name,
        "duration": round(duration, 3),
        "tempo": round(float(structure["tempo"]), 3),
        "beats": r3(structure["beats"]),
        "downbeats": r3(structure["downbeats"]),
        "grid": structure["grid"],
        "phrases": structure["phrases"],
        "sections": structure["sections"],
        "drops": structure["drops"],
        "curves": structure["curves"],
        "loudness": {"rate": LOUDNESS_RATE, "values": [round(float(v), 3) for v in loudness]},
        "bass": {"rate": LOUDNESS_RATE, "values": [round(float(v), 3) for v in bass_curve]},
        "sub": {"rate": LOUDNESS_RATE, "values": [round(float(v), 3) for v in sub_curve]},
        "intensity": {"rate": LOUDNESS_RATE, "values": [round(float(v), 3) for v in intensity_curve]},
        "hats": hats,
        "sounds": structure["sounds"],
        "debug": structure["debug"],
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tracks", type=Path, nargs="+")
    ap.add_argument("--sound", action="append", default=[], metavar="NAME@SECONDS[@STEM]", help="name a sound by one example of it")
    args = ap.parse_args()
    for track in args.tracks:
        side = track.with_name(track.name + ".sounds.json")
        sounds = json.loads(side.read_text()) if side.exists() else []
        for spec in args.sound:
            name, at, *stem = spec.split("@")
            sounds = [s for s in sounds if s["name"] != name] + [{"name": name, "at": float(at), "stem": stem[0] if stem else "drums"}]
        if args.sound:
            side.write_text(json.dumps(sounds, indent=1))
        result = analyse(track, sounds)
        out = track.with_name(track.name + ".analysis.json")
        out.write_text(json.dumps(result, separators=(",", ":")))
        summary = ", ".join(f"{s['kind']} {s['start']:.1f}-{s['end']:.1f}" for s in result["sections"])
        drops = ", ".join(f"{d['t']:.2f} ({d['confidence']:.2f})" for d in result["drops"])
        grid = "steady grid" if result["grid"]["confident"] else "NO reliable grid"
        named = ", ".join(f"{k} x{len(v)}" for k, v in result["sounds"].items())
        print(f"{track.name}: {result['tempo']} BPM ({grid}), {len(result['beats'])} beats -> {out.name}\n  {summary}\n  drops: {drops}"
              + (f"\n  sounds: {named}" if named else ""))


if __name__ == "__main__":
    main()
