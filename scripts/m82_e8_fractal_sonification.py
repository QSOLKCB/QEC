#!/usr/bin/env python3
"""
M82 E8 Fractal Sonification
Creates a .wav audio file by sonifying Chandra and Xtend images of M82
using E8-inspired mathematics: golden ratio (phi) for spectra/rhythms (quasicrystal projection proxy),
Fibonacci for fractal self-similar timing/onsets, 432Hz cosmic tuning,
triality elements (3-fold permutations in voices/phases), and 8D echoes via layered partials.
Brightness from images drives amplitude, triggers, and timbral complexity.
Deterministic, precise, replay-safe generation aligned with QEC philosophy.
"""

import numpy as np
from PIL import Image
from scipy.io import wavfile
from scipy.interpolate import interp1d
import math

# === Constants ===
SR = 48000          # High quality sample rate
DUR = 75.0          # seconds - enough to explore structure without fatigue
N = int(SR * DUR)
t = np.arange(N) / float(SR)

PHI = (1.0 + np.sqrt(5.0)) / 2.0  # Golden ratio - key to E8 projections & quasicrystals
F_BASE = 432.0      # Cosmic tuning preference (user context)

# E8 / Golden partials generator (inharmonic spectrum evoking quasicrystal interference & φFM like Chowning/Stria)
def golden_e8_partials(fundamental, num_partials=8, spread=1.0):
    """Generate frequencies using powers of PHI. Evokes E8->4D/2D quasicrystal projections
    and golden ratio spectra. Spread controls inharmonicity."""
    freqs = []
    for k in range(num_partials):
        # Mix integer and phi powers for richer E8-like relations (some near harmonic, some inharmonic)
        f = fundamental * (PHI ** (k * spread * 0.7)) * (1.0 + 0.02 * (k % 3 - 1))  # slight triality perturbation
        freqs.append(f)
    return freqs

# Fibonacci sequence for fractal rhythms/onsets/durations (self-similar, aperiodic like quasicrystals)
def fibonacci_sequence(max_val):
    fib = [1, 1]
    while True:
        next_f = fib[-1] + fib[-2]
        if next_f > max_val:
            break
        fib.append(next_f)
    return fib

# === Load & Process Images ===
print("Loading M82 images...")
chandra_img = Image.open('/home/workdir/attachments/m82_chandra_2000eV_10000eV_1x1kpc.jpg').convert('L')
chandra = np.array(chandra_img, dtype=np.float64) / 255.0

xtend_img = Image.open('/home/workdir/attachments/M82_Xtend.png').convert('L')
xtend = np.array(xtend_img, dtype=np.float64) / 255.0

# Resize for manageable feature extraction if very large
if chandra.shape[0] > 400:
    chandra = np.array(Image.fromarray((chandra*255).astype(np.uint8)).resize((400, int(400*chandra.shape[0]/chandra.shape[1])), Image.LANCZOS)) / 255.0
if xtend.shape[0] > 400:
    xtend = np.array(Image.fromarray((xtend*255).astype(np.uint8)).resize((400, int(400*xtend.shape[0]/xtend.shape[1])), Image.LANCZOS)) / 255.0

h_c, w_c = chandra.shape
h_x, w_x = xtend.shape

# Brightness profiles (column averages) - map galactic structure to time evolution
# Assume horizontal elongation in Xtend corresponds to major axis or outflow projection
profile_xtend = np.mean(xtend, axis=0)
profile_chandra = np.mean(chandra, axis=0)

# Normalize profiles robustly
def norm_profile(p):
    p = p - np.min(p)
    ptp = np.ptp(p)
    return p / (ptp + 1e-9)

profile_xtend = norm_profile(profile_xtend)
profile_chandra = norm_profile(profile_chandra)

# Interpolate profiles to full audio length for continuous modulation (core bright -> higher energy)
x_xt = np.linspace(0, 1, len(profile_xtend))
mod_xtend = interp1d(x_xt, profile_xtend, kind='cubic', fill_value='extrapolate')(np.linspace(0, 1, N))

x_ch = np.linspace(0, 1, len(profile_chandra))
mod_chandra = interp1d(x_ch, profile_chandra, kind='cubic', fill_value='extrapolate')(np.linspace(0, 1, N))

# Global intensity from core (Chandra bright center for starburst energy)
global_intensity = np.clip(0.4 + 0.6 * mod_chandra, 0.2, 1.0)

# Discrete bright "stars/knots" detection in Chandra (point sources -> percussive events)
# Simple threshold + local max for deterministic "star" triggers
bright_mask = chandra > 0.75
# Sample positions of bright pixels for events (downsample to ~20-30 events)
bright_coords = np.argwhere(bright_mask)
if len(bright_coords) > 40:
    step = len(bright_coords) // 35
    bright_coords = bright_coords[::step]

print(f"Detected {len(bright_coords)} bright X-ray knots for percussive events.")

# === Audio Synthesis ===
print("Synthesizing E8-fractal sonification...")

audio = np.zeros(N, dtype=np.float64)

# --- Layer 1: Foundational E8-Golden Drone/Pad (quasicrystal-like interference) ---
# 8 partials echoing E8 dimension. Amplitude + complexity driven by Xtend extended structure (outflows/wind)
freqs_pad = golden_e8_partials(F_BASE * 0.5, num_partials=8, spread=0.85)  # Lower register for cosmic foundation
for i, f in enumerate(freqs_pad):
    # Slow evolving phase with triality-ish 3-phase offset
    phase = (2 * np.pi * f * t + 
             0.15 * np.sin(2 * np.pi * 0.007 * t + i * 2 * np.pi / 3) +  # triality phase shift
             0.08 * np.sin(2 * np.pi * 0.023 * t * PHI))  # golden modulation
    partial = np.sin(phase)
    # Amplitude envelope from Xtend profile + global intensity, slight fractal breathing
    amp_env = (0.15 + 0.85 * mod_xtend) * global_intensity * (0.7 + 0.3 * np.sin(2 * np.pi * 0.011 * t + i))
    # Gentle inharmonicity evolution (brighter core -> slightly more detuned for tension)
    detune = 1.0 + 0.008 * mod_chandra * np.sin(2 * np.pi * 0.031 * t)
    audio += partial * amp_env * (0.65 ** i) * detune * 0.22

# --- Layer 2: Fractal Fibonacci Burst/Melody (self-similar star formation bursts) ---
# Onsets from Fibonacci sequence scaled to duration. Pitches & timbres from image brightness.
fib = fibonacci_sequence(DUR * 8)  # generous for density control
onset_times = np.cumsum([f_val / 9.5 for f_val in fib])  # scale factor tuned for ~40-60 events in 75s
onset_times = onset_times[onset_times < DUR - 1.5]

for onset in onset_times:
    idx = int(onset * SR)
    if idx >= N - 100: continue
    
    # Local brightness from Chandra (core energy) and Xtend (context)
    local_bright = float(mod_chandra[min(idx, N-1)])
    struct_bright = float(mod_xtend[min(idx, N-1)])
    
    # Burst duration and density fractal: brighter -> longer/more complex (more partials active)
    burst_dur = 0.18 + 0.55 * local_bright + 0.15 * struct_bright
    burst_n = int(burst_dur * SR)
    if burst_n < 50: continue
    burst_t = np.arange(burst_n) / float(SR)
    
    # Number of partials scales with brightness (fractal depth)
    n_part = max(3, min(9, int(4 + 5 * local_bright)))
    freqs_burst = golden_e8_partials(F_BASE * (0.6 + 1.8 * struct_bright), num_partials=n_part, spread=1.1)
    
    burst = np.zeros(burst_n)
    for k, f in enumerate(freqs_burst):
        # Envelope: fast attack, exponential decay (starburst flash)
        env = np.exp(-burst_t / (0.06 + 0.18 * local_bright)) * (0.4 + 0.6 * local_bright)
        # Slight triality rotation in phase (3 different phase offsets cycling)
        phase_off = (k % 3) * 2.0 * np.pi / 3.0
        tone = np.sin(2 * np.pi * f * burst_t + phase_off + 0.3 * k)
        # Additional golden-ratio FM-ish sideband for richer fractal texture
        if k > 2:
            mod_idx = 0.4 * local_bright
            tone *= (1.0 + mod_idx * np.sin(2 * np.pi * f * PHI * burst_t))
        burst += tone * env * (0.55 ** k)
    
    # Add to main audio with some panning prep (will split to stereo later)
    end_idx = min(idx + burst_n, N)
    audio[idx:end_idx] += burst[:end_idx - idx] * 0.65

# --- Layer 3: X-ray Point Source "Pings" (discrete Chandra knots as mathematical stars) ---
# Each bright knot triggers a short, high-register golden ping. Pitch varies deterministically with position.
for ii, (y, x) in enumerate(bright_coords):
    # Map image position to time within the piece (scan order or distributed)
    # Distribute across duration proportionally to x-position for left-to-right galactic scan feel
    event_time = 3.0 + (DUR - 6.0) * (x / float(w_c)) * (0.6 + 0.4 * (ii % 5) / 4.0)  # slight deterministic jitter via ii
    idx = int(event_time * SR)
    if idx < 200 or idx > N - 200: continue
    
    ping_dur = 0.04 + 0.03 * (chandra[y, x] - 0.75) * 4  # brighter = slightly longer
    pn = int(ping_dur * SR)
    pt = np.arange(pn) / float(SR)
    
    # Pitch from vertical position + Fibonacci-ish offset for variety (E8 triality groups)
    base_pitch_mult = 6.0 + 4.0 * (y / float(h_c))  # higher in image = higher pitch (top of galaxy?)
    fib_offset = (fib[ii % len(fib)] % 7) * 0.07
    ping_f = F_BASE * base_pitch_mult * (1.0 + fib_offset)
    
    # Golden partials for the ping (short burst of E8 spectrum)
    ping_partials = golden_e8_partials(ping_f, num_partials=5, spread=1.3)
    ping = np.zeros(pn)
    for k, f in enumerate(ping_partials):
        env = np.exp(-pt / 0.012) * (chandra[y, x] ** 1.5)
        tone = np.sin(2 * np.pi * f * pt + k * np.pi * 0.6)
        ping += tone * env * (0.7 ** k) * 0.8
    
    end_idx = min(idx + pn, N)
    audio[idx:end_idx] += ping[:end_idx-idx] * 0.55

# --- Layer 4: Outflow / Superwind Whoosh (low frequency modulated noise from extended structure) ---
# Brownian-like motion but deterministic via integrated low-freq oscillators + Xtend profile
whoosh = np.zeros(N)
# Multiple slow oscillators with golden freq ratios for organic wind feel
whoosh_freqs = [12.0, 12.0*PHI*0.5, 19.5, 31.0]
for wf in whoosh_freqs:
    whoosh += np.sin(2 * np.pi * wf * t + 0.4 * np.sin(2 * np.pi * 0.003 * t)) * 0.6
whoosh = whoosh / len(whoosh_freqs)
# Modulate by Xtend profile (extended emission = stronger wind) and low-pass-ish feel via slow env
whoosh *= (0.08 + 0.25 * mod_xtend) * global_intensity * 0.6
# Add gentle high-freq shimmer on bright core for starburst "sparkle"
shimmer = np.sin(2 * np.pi * 1800 * t) * np.clip(mod_chandra - 0.3, 0, 1) * 0.015 * global_intensity
whoosh += shimmer

audio += whoosh

# === Post-processing: Stereo spatialization, normalization, gentle limiting ===
print("Finalizing stereo image and levels...")

# Stereo: slight width from phase difference + panning based on slow LFO + structure
# Left channel slightly "earlier" galactic scan, right has more outflow whoosh
stereo_l = audio.copy()
stereo_r = audio.copy()

# Add micro-delay and filtering difference for width (E8 symmetry breaking to stereo)
delay_samples = int(0.0008 * SR)  # ~0.8ms
stereo_r = np.roll(stereo_r, delay_samples) * 0.97 + audio * 0.03

# Gentle auto-panning following the Xtend profile peaks (core centered, edges wider)
pan_lfo = 0.5 + 0.4 * np.sin(2 * np.pi * 0.009 * t) * mod_xtend
stereo_l *= (1.0 - pan_lfo * 0.3)
stereo_r *= (0.7 + pan_lfo * 0.3)

# Final mix
audio_stereo = np.column_stack((stereo_l, stereo_r))

# Normalize with headroom
peak = np.max(np.abs(audio_stereo))
if peak > 0:
    audio_stereo = audio_stereo / peak * 0.92

# Soft limiting to avoid clipping while preserving dynamics
audio_stereo = np.tanh(audio_stereo * 1.05) * 0.95

# Convert to 24-bit int (high quality) or 16-bit
audio_int = np.int16(audio_stereo * 32767)

# === Write WAV ===
out_path = '/home/workdir/artifacts/M82_E8_Fractal_Sonification.wav'
wavfile.write(out_path, SR, audio_int)
print(f"\n✓ Created: {out_path}")
print(f"  Duration: {DUR:.1f} s | SR: {SR} Hz | Stereo")
print(f"  Features: E8-golden partials (8D echo), Fibonacci fractal onsets, 432Hz base,")
print(f"            Chandra X-ray knots as pings, Xtend outflows as whoosh,")
print(f"            triality phase shifts, image brightness as amplitude/timbre driver.")
print("  Aligned with deterministic QEC principles and your E8 fractal cosmology explorations.")