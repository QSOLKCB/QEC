#!/usr/bin/env python3
"""
NGC 3603 JWST NIRSpec Stacked Nebular Spectrum
E8 Fractal Sonification Script
----------------------------------------------
Sonifies the emission line spectrum from Rogers et al. (2024) A&A 688, A111
using additive synthesis with quasi-periodic modulations inspired by the
golden ratio (φ) connected to E8/icosian structures and fractal-like
self-similar interference patterns in the evolving texture.

The strong recombination and s-process lines are mapped across ~5 octaves
of pitch space (starting near 432-tuning A1). Amplitudes from log-compressed
fluxes. Each partial has its own golden-ratio-derived LFO for evolving,
non-repeating "fractal" beating and timbre shifts over time.

Suitable as a cosmic pad/drone element for music tracks.
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os

# =============================================================================
# LOAD AND PROCESS SPECTRUM DATA
# =============================================================================
data_path = '/home/workdir/attachments/spec.dat'
data = np.loadtxt(data_path)
lambda_um = data[:, 0]
flux = data[:, 1]

print(f"Loaded spectrum: {len(lambda_um)} points")
print(f"Lambda range: {lambda_um.min():.4f} - {lambda_um.max():.4f} μm")
print(f"Flux range: {flux.min():.2f} - {flux.max():.2f} (mean {flux.mean():.2f})")

# Find prominent emission lines (recombination, s-process, H2 etc.)
# Adjust height/distance to capture key features without too many weak lines
peak_height = 2.5
peak_distance = 10
peaks_idx, properties = find_peaks(flux, height=peak_height, distance=peak_distance)
print(f"\nFound {len(peaks_idx)} peaks above height={peak_height}")

# Take the strongest N lines (log-compressed later anyway)
N_LINES = 16
if len(peaks_idx) > N_LINES:
    top_indices = np.argsort(properties['peak_heights'])[-N_LINES:][::-1]
    selected_idx = peaks_idx[top_indices]
else:
    selected_idx = peaks_idx

selected_lambda = lambda_um[selected_idx]
selected_flux = flux[selected_idx]

# Sort by increasing wavelength (IR progression)
sort_order = np.argsort(selected_lambda)
selected_lambda = selected_lambda[sort_order]
selected_flux = selected_flux[sort_order]

print(f"Using top {len(selected_lambda)} emission lines for sonification:")
for i, (lam, flx) in enumerate(zip(selected_lambda, selected_flux)):
    print(f"  {i+1:2d}. {lam:7.4f} μm   flux={flx:8.2f}")

# =============================================================================
# SONIFICATION PARAMETERS
# =============================================================================
SR = 48000                    # Sample rate (CD quality)
DURATION = 28.0               # Seconds - nice length for a track element
FADE_TIME = 0.8               # Seconds fade in/out

t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)

# Wavelength -> Frequency mapping
# Spread the IR lines across ~5 octaves starting near 432 Hz tuning A1 (54 Hz)
LAMBDA_MIN = lambda_um.min()
LAMBDA_MAX = lambda_um.max()
BASE_FREQ = 54.0              # A1 in 432 Hz tuning (A4=432)

def lambda_to_freq(lam):
    """Map wavelength position to frequency with exponential spread."""
    norm = (lam - LAMBDA_MIN) / (LAMBDA_MAX - LAMBDA_MIN)
    # 5 octaves span: 54 Hz → 54 * 32 ≈ 1728 Hz
    return BASE_FREQ * (2 ** (norm * 5.0))

freqs = np.array([lambda_to_freq(lam) for lam in selected_lambda])

# Amplitude from log-compressed flux (handles huge dynamic range of lines)
log_flux = np.log10(selected_flux + 1.0)
amps = (log_flux / log_flux.max()) * 0.55   # Headroom for layering

# =============================================================================
# E8 / FRACTAL INSPIRED MODULATION
# Golden ratio φ appears in E8 projections and icosian ring.
# Using powers of φ for LFO rates creates quasi-periodic, self-similar
# interference patterns that evolve without exact repetition → "fractal" feel.
# =============================================================================
PHI = (1 + np.sqrt(5)) / 2   # Golden ratio ≈ 1.6180339887

# LFO rates: base ~0.012 Hz, scaled by φ^k for different partials
# This gives a hierarchy of slow modulations (tremolo / timbre shift)
lfo_rates = 0.012 * np.array([PHI ** (k % 7) for k in range(len(freqs))])

print(f"\nE8/φ modulation LFO rates (Hz): {np.round(lfo_rates, 4)}")

# =============================================================================
# SYNTHESIS
# =============================================================================
print("\nSynthesizing audio...")

signal = np.zeros_like(t, dtype=np.float64)

for i in range(len(freqs)):
    f = freqs[i]
    a = amps[i]
    lfo_r = lfo_rates[i]
    
    # Random starting phase for organic feel
    phase = np.random.uniform(0, 2 * np.pi)
    
    # Primary tone (main emission line contribution)
    tone = a * np.sin(2 * np.pi * f * t + phase)
    
    # Evolving amplitude modulation (tremolo) with golden LFO
    # Creates slow beating and "breathing" texture
    amp_mod = 0.65 + 0.35 * np.sin(2 * np.pi * lfo_r * t)
    
    # Very slight frequency modulation (detune) for rich beating
    # Second LFO at incommensurate rate (φ related)
    fm_depth = 0.0008 * f
    fm_lfo = np.sin(2 * np.pi * (lfo_r * 0.618) * t)
    inst_freq = f + fm_depth * fm_lfo
    
    # Add a higher harmonic (≈3rd) with its own slight detune for spectral richness
    harm3 = (a * 0.18) * np.sin(2 * np.pi * (inst_freq * 1.498) * t)
    
    signal += tone * amp_mod + harm3

# =============================================================================
# ADDITIONAL E8-INSPIRED LAYERS (sub-bass drone + mid pad)
# =============================================================================
# Low sub-bass near 432-tuning A0 (27 Hz), modulated slowly
sub_f = 27.0
sub_lfo_rate = 0.007 * PHI
sub_mod = 0.75 + 0.25 * np.sin(2 * np.pi * sub_lfo_rate * t)
signal += 0.22 * np.sin(2 * np.pi * sub_f * t) * sub_mod

# Mid drone with φ-detuned frequency for "E8 dimension" feel
drone_f = 108.0 * PHI * 0.5   # ~87.4 Hz
drone_lfo = 0.011 * PHI**2
drone_mod = 0.82 + 0.18 * np.sin(2 * np.pi * drone_lfo * t)
signal += 0.15 * np.sin(2 * np.pi * drone_f * t) * drone_mod

# Very high ethereal layer (faint "nebula halo")
high_f = 1728.0 * 0.97
high_mod = 0.4 + 0.6 * np.sin(2 * np.pi * 0.009 * t)
signal += 0.04 * np.sin(2 * np.pi * high_f * t) * high_mod

# =============================================================================
# POST-PROCESSING
# =============================================================================
# Peak normalize
max_val = np.max(np.abs(signal))
if max_val > 0:
    signal = signal / max_val * 0.92

# Gentle fade in/out
fade_samples = int(FADE_TIME * SR)
if fade_samples > 0:
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    signal[:fade_samples] *= fade_in
    signal[-fade_samples:] *= fade_out

# Soft clip / safety
signal = np.clip(signal, -0.98, 0.98)

# Convert to 16-bit PCM
signal_int16 = np.int16(signal * 32767)

# =============================================================================
# SAVE WAV
# =============================================================================
output_wav = '/home/workdir/artifacts/NGC3603_E8_Fractal_Sonification.wav'
wavfile.write(output_wav, SR, signal_int16)
print(f"\n✓ WAV saved: {output_wav}")
print(f"  Duration: {DURATION:.1f} s | Sample rate: {SR} Hz | Channels: mono")

# =============================================================================
# DIAGNOSTIC PLOT (spectrum + selected lines)
# =============================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})

# Full spectrum
ax1.plot(lambda_um, flux, color='#1a1a2e', linewidth=0.6, alpha=0.85, label='Stacked nebular spectrum')
ax1.plot(selected_lambda, selected_flux, 'o', color='#e94560', markersize=7, 
         markeredgecolor='white', markeredgewidth=0.5, label=f'Selected emission lines (N={len(selected_lambda)})')
ax1.set_yscale('log')
ax1.set_xlabel('Wavelength (μm)', fontsize=11)
ax1.set_ylabel('Flux (arbitrary units, log scale)', fontsize=11)
ax1.set_title('NGC 3603 — JWST NIRSpec Stacked Nebular Spectrum\n'
              'E8 Fractal Sonification: Emission lines → Additive synthesis with φ-modulated evolution',
              fontsize=13, pad=10)
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.25, which='both')
ax1.set_xlim(lambda_um.min(), lambda_um.max())

# Zoomed view of selected region with line labels
ax2.plot(lambda_um, flux, color='#16213e', linewidth=0.7, alpha=0.9)
ax2.plot(selected_lambda, selected_flux, 'o', color='#e94560', markersize=6)
for lam, flx in zip(selected_lambda, selected_flux):
    ax2.annotate(f'{lam:.3f}', xy=(lam, flx), xytext=(0, 8), 
                 textcoords='offset points', fontsize=7, ha='center', color='#e94560',
                 rotation=45)
ax2.set_xlabel('Wavelength (μm)', fontsize=10)
ax2.set_ylabel('Flux', fontsize=10)
ax2.set_title('Zoom: Selected lines used for sonification (sorted by λ → freq mapping)', fontsize=10)
ax2.grid(True, alpha=0.2)
ax2.set_xlim(1.65, 3.20)

plt.tight_layout()
plot_path = '/home/workdir/artifacts/NGC3603_spectrum_E8_sonification.png'
plt.savefig(plot_path, dpi=160, facecolor='white', edgecolor='none')
plt.close()
print(f"✓ Diagnostic plot saved: {plot_path}")

print("\n" + "="*70)
print("E8 FRACTAL SONIFICATION COMPLETE")
print("="*70)
print("The .wav contains an evolving cosmic pad/drone whose partials")
print("correspond to the actual JWST-observed emission lines in NGC 3603.")
print("Modulations use the golden ratio φ (E8/icosian connection) to create")
print("quasi-periodic, fractal-like interference that slowly shifts timbre")
print("and amplitude over the 28-second duration — ideal for layering in tracks.")
print("="*70)