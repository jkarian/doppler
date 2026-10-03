"""Synthesize a 64-second, 120 BPM test track with known sections, for checking the analysis and sync.

    python tools/make_test_track.py   ->  tracks/test-120bpm.wav

    0-16 s   quiet: soft pad and hats
    16-32 s  build: kicks every beat, rising noise sweep and snare roll
    32-52 s  drop: loud kick, bass and claps
    52-64 s  quiet outro
"""

from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100
BPM = 120
LENGTH = 64.0
rng = np.random.default_rng(7)
n = int(SR * LENGTH)
t = np.arange(n) / SR
out = np.zeros(n)
beat = 60 / BPM


def add(at: float, sound: np.ndarray, gain: float) -> None:
    i = int(at * SR)
    j = min(n, i + len(sound))
    out[i:j] += gain * sound[: j - i]


def env(length: float, decay: float) -> np.ndarray:
    return np.exp(-np.arange(int(length * SR)) / SR / decay)


kick_t = np.arange(int(0.35 * SR)) / SR
# Pitch falls from 150 Hz to 45 Hz: the phase is the integral of 45 + 105 * exp(-t / 0.04).
kick = np.sin(2 * np.pi * (45 * kick_t + 105 * 0.04 * (1 - np.exp(-kick_t / 0.04)))) * env(0.35, 0.09)
hat = rng.standard_normal(int(0.05 * SR)) * env(0.05, 0.01)
hat = np.diff(hat, prepend=0)  # brighter
snare = rng.standard_normal(int(0.2 * SR)) * env(0.2, 0.05)

# Pad over the whole track.
for f in (110, 164.8, 220, 277.2):
    out += 0.03 * np.sin(2 * np.pi * f * t) * (0.85 + 0.15 * np.sin(2 * np.pi * t / 8))

for b in range(int(LENGTH / beat)):
    at = b * beat
    if at < 16 or at >= 52:
        add(at + beat / 2, hat, 0.08)
    elif at < 32:
        rise = (at - 16) / 16
        add(at, kick, 0.35 + 0.3 * rise)
        add(at + beat / 2, hat, 0.12)
        rolls = 1 if rise < 0.5 else 2 if rise < 0.75 else 4
        for k in range(rolls):
            add(at + k * beat / rolls, snare, 0.05 + 0.25 * rise)
    else:
        add(at, kick, 0.9)
        add(at + beat / 2, hat, 0.2)
        if b % 2 == 1:
            add(at, snare, 0.45)
        bass_t = np.arange(int(beat * SR)) / SR
        root = 55 if (b // 4) % 2 == 0 else 49
        add(at, np.sign(np.sin(2 * np.pi * root * bass_t)) * env(beat, 0.25), 0.18)

# Build sweep: rising filtered noise.
build = (t >= 16) & (t < 32)
noise = np.convolve(rng.standard_normal(n), np.ones(8) / 8, mode="same")
out += build * noise * 0.12 * np.clip((t - 16) / 16, 0, 1) ** 2

# Half a bar of near-silence right before the drop.
out[(t >= 31) & (t < 32)] *= 0.05

out /= np.max(np.abs(out)) * 1.05
path = Path(__file__).resolve().parent.parent / "tracks" / "test-120bpm.wav"
path.parent.mkdir(exist_ok=True)
sf.write(path, out.astype(np.float32), SR)
print(f"wrote {path}")
