#!/usr/bin/env python3
"""
E8 Triality Sonification — Fusion-QEC Data Audio Renderer
---------------------------------------------------------

Maps Quantum Error Correction benchmark data into sound,
following the E8 Triality relation:

    Φ = π / 2  SCL DIAG + [1, −2, 1]

Each QEC column becomes a harmonic voice:
  - Steane → root tone (stability)
  - Surface → mid layer (coherence)
  - Reed-Muller → modulation (fault tolerance)
  - Fusion-QEC (Photonic) → brightness / amplitude

If playback fails, automatically exports to e8_triality.wav.
"""

import math
import numpy as np
import pandas as pd
import wave
import sys
from pathlib import Path

# Optional audio backend
try:
    import simpleaudio as sa
    AUDIO_BACKEND = True
except ImportError:
    AUDIO_BACKEND = False

# Load dataset
csv_path = Path("qec_data_prepared.csv")
if not csv_path.exists():
    sys.exit("❌ Missing qec_data_prepared.csv — run extraction first.")

df = pd.read_csv(csv_path)
cols = [c for c in df.columns if c != "error_rate"]

# Parameters
sample_rate = 44100
duration = 0.5  # seconds per frame
phase_shift = math.pi / 2  # Φ = π/2
ternary = [1, -2, 1]  # [Signal, Loss, Coherence]
base_freq = 220.0  # Hz
amplitude = 0.3

# Normalize helper
def normalize(arr):
    arr = np.array(arr, dtype=float)
    return np.zeros_like(arr) if arr.ptp() == 0 else (arr - arr.min()) / arr.ptp()

# Generate tone
def make_tone(freq, amp=0.3, phase=0):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    tone = np.sin(2 * np.pi * freq * t + phase)
    return (tone * (amp * 32767)).astype(np.int16)

# Prepare frequency bands
freq_map = {
    col: base_freq * factor for col, factor in zip(cols, [1.0, 1.5, 2.0, 2.5])
}
# Build sound frames
audio_frames = []
for i, row in df.iterrows():
    error_rate = row["error_rate"]
    freqs = []
    amps = []
    for j, col in enumerate(cols):
        value = float(row[col])
        freq = freq_map[col] * (1 + abs(math.log10(value + 1e-12)))
        amp = amplitude * abs(ternary[j % len(ternary)])
        freqs.append(freq)
        amps.append(amp)

    # Sum and normalize mixed frame
    tone_mix = sum(make_tone(f, a, phase_shift) for f, a in zip(freqs, amps))
    tone_mix = tone_mix.astype(np.float32) / len(freqs)
    tone_mix = np.clip(tone_mix, -32767, 32767).astype(np.int16)
    tone_mix = np.clip(tone_mix / len(freqs), -32767, 32767).astype(np.int16)
    audio_frames.append(tone_mix)

# Concatenate all frames
audio_data = np.concatenate(audio_frames)

# Fallback writer
def write_wav(filename, data):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())
    print(f"💾 Wrote fallback audio to {filename}")

# Playback or export
try:
    if AUDIO_BACKEND:
        print("🔊 Playing E8 Triality sonification...")
        play_obj = sa.play_buffer(audio_data, 1, 2, sample_rate)
        play_obj.wait_done()
    else:
        raise RuntimeError("Audio backend not available")
except Exception as e:
    print(f"⚠️  Playback failed ({e}). Exporting to WAV instead.")
    write_wav("e8_triality.wav", audio_data)
    sys.exit(0)

print("✅ Sonification complete.")
