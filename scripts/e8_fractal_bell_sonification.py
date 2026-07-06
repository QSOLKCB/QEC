#!/usr/bin/env python3
"""
E8 Fractal Bell States Sonification Script
==========================================

Generates a high-quality .wav file that sonifies the four Bell states
with |Φ⁺⟩ (Phi Plus) as the dominant state.

Key Design Principles (aligned with deterministic, validation-focused ethos):
- Purely mathematical / deterministic construction (no heuristics, no probabilistic logic at runtime).
- Reproducible via fixed mathematical constants and structure (np.random.seed set for any future extensions).
- Self-similar fractal spectrum using golden ratio (φ) scaling — inspired by E8 lattice projections,
  quasicrystals, and the golden ratio's deep appearance in E8 affine Weyl group and icosahedral symmetries.
- 8 additive layers explicitly reference the rank/dimension of E8.
- 432 Hz base tuning (A4=432) as preferred in related mathematical music work.
- Classical simulation of quantum entanglement via audio channel correlation (stereo L/R identical
  for dominant Phi+ → perfect positive correlation: "measuring" one channel perfectly predicts the other).
- All four Bell states are present in the mix (superposition of sonic "states"), with Phi+ carrying
  the majority energy/weight so it dominates the perceptual character while the others add
  textural nuance, phase interference, and timbral contrast.

Bell State Mapping to Sound:
- |Φ⁺⟩ = 1/√2 (|00⟩ + |11⟩)     → In-phase superposition of two frequencies (f and f·φ). Positive correlation.
- |Φ⁻⟩ = 1/√2 (|00⟩ - |11⟩)     → Same frequencies but relative minus sign → phase opposition, spectral cancellations.
- |Ψ⁺⟩ = 1/√2 (|01⟩ + |10⟩)     → "Odd parity" frequency pair (different base tones representing bit flip).
- |Ψ⁻⟩ = 1/√2 (|01⟩ - |10⟩)     → Odd parity pair + relative phase flip.

Fractal Construction:
Each state's contribution is built from 8 layers (E8 rank):
  layer_k = amp_k · ( sin(2π·f0·φ^k ·t)  ±  sin(2π·f1·φ^k ·t) )
where amp_k decreases geometrically with persistence ≈ 1/φ, lacunarity = φ (irrational → dense inharmonic
but mathematically coherent partials, fractal power spectrum).

The final mix is peak-normalized, faded, and rendered as 16-bit stereo WAV with L ≡ R channels
to emphasize the dominant Phi+ entanglement correlation.

Usage:
    python3 e8_fractal_bell_sonification.py

Output:
    /home/workdir/artifacts/e8_fractal_bell_sonification_phi_plus_dominant.wav

This script is self-contained, uses only numpy + scipy (standard scientific stack), and runs
deterministically on any machine with the same floating-point behavior.
"""

import numpy as np
from scipy.io import wavfile
import os


def generate_e8_bell_sonification(
    output_path: str = "/home/workdir/artifacts/e8_fractal_bell_sonification_phi_plus_dominant.wav",
    duration: float = 60.0,
    sr: int = 44100,
    base_freq: float = 432.0,
    e8_seed: int = 240,  # Number of roots in E8 root system — used for reproducibility marker
) -> str:
    """
    Core generation function. All parameters fixed for determinism.
    """
    np.random.seed(e8_seed)  # Set for full reproducibility. Currently used only as marker;
                             # all synthesis is closed-form mathematical. Can be extended
                             # with seeded micro-variations (e.g. tiny phase dither) without
                             # breaking determinism.

    phi = (1.0 + np.sqrt(5.0)) / 2.0  # Golden ratio — central to Phi Plus, Fibonacci, and E8

    # Time vector
    n_samples = int(sr * duration)
    t = np.linspace(0.0, duration, n_samples, endpoint=False)

    # Perceptual mix weights — Phi Plus dominant
    weights = {
        "phi_plus": 0.55,
        "phi_minus": 0.15,
        "psi_plus": 0.15,
        "psi_minus": 0.15,
    }

    # Frequency pairs chosen for musical/mathematical coherence
    # Phi states: "even parity" — correlated bits (|00> and |11>)
    f_phi_0 = base_freq
    f_phi_1 = base_freq * phi

    # Psi states: "odd parity" — anti-correlated bits (|01> and |10>)
    f_psi_0 = base_freq * (phi - 1.0)      # ≈ 266.96 Hz  (φ⁻¹ scaling)
    f_psi_1 = base_freq * (phi + 1.0)      # ≈ 1131.0 Hz  (φ² scaling)

    state_params = {
        "phi_plus":  {"f0": f_phi_0, "f1": f_phi_1, "sign": +1.0},
        "phi_minus": {"f0": f_phi_0, "f1": f_phi_1, "sign": -1.0},
        "psi_plus":  {"f0": f_psi_0, "f1": f_psi_1, "sign": +1.0},
        "psi_minus": {"f0": f_psi_0, "f1": f_psi_1, "sign": -1.0},
    }

    master_signal = np.zeros_like(t, dtype=np.float64)

    for state_name, w in weights.items():
        p = state_params[state_name]
        f0, f1, sgn = p["f0"], p["f1"], p["sign"]

        state_signal = np.zeros_like(t)

        # === E8 Fractal Additive Synthesis (8 layers = rank of E8) ===
        num_layers = 8
        for k in range(num_layers):
            scale = phi ** k                    # lacunarity = φ (self-similar frequency scaling)
            amp = (1.0 / np.sqrt(2.0)) / (phi ** k)   # persistence ≈ 1/φ, Bell normalization

            # Core Bell superposition at this fractal scale
            layer = amp * (
                np.sin(2.0 * np.pi * f0 * scale * t) +
                sgn * np.sin(2.0 * np.pi * f1 * scale * t)
            )
            state_signal += layer

        master_signal += w * state_signal

    # === Post-processing: peak normalize + gentle fades (musical presentation) ===
    peak = np.max(np.abs(master_signal))
    if peak > 1e-12:
        master_signal = master_signal / peak * 0.92   # -0.7 dBFS headroom

    # 2.5 second fade in / out for artifact-free listening
    fade_dur = 2.5
    fade_n = int(fade_dur * sr)
    if fade_n > 0 and fade_n < n_samples:
        fade_in = np.linspace(0.0, 1.0, fade_n)
        fade_out = np.linspace(1.0, 0.0, fade_n)
        master_signal[:fade_n] *= fade_in
        master_signal[-fade_n:] *= fade_out

    # Convert to 16-bit PCM
    audio_int16 = np.int16(np.clip(master_signal, -1.0, 1.0) * 32767.0)

    # === Stereo rendering: identical L/R channels ===
    # This is the key classical simulation of Phi+ entanglement:
    # Perfect positive correlation between channels. "Measuring" left ear
    # instantly tells you the state of the right ear with certainty.
    # The small contributions from other Bell states add subtle decorrelation
    # and timbral complexity without destroying the dominant correlation.
    stereo = np.column_stack((audio_int16, audio_int16))

    # Write WAV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    wavfile.write(output_path, sr, stereo)

    # === Metadata / reproducibility report ===
    print("=" * 70)
    print("E8 FRACTAL BELL STATES SONIFICATION — GENERATION COMPLETE")
    print("=" * 70)
    print(f"Output file      : {output_path}")
    print(f"Duration         : {duration:.1f} s")
    print(f"Sample rate      : {sr} Hz")
    print(f"Channels         : 2 (stereo, L ≡ R for Phi+ correlation)")
    print(f"Bit depth        : 16-bit PCM")
    print(f"Base frequency   : {base_freq} Hz (A4 = 432 tuning)")
    print(f"Golden ratio φ   : {phi:.10f}")
    print(f"E8 layers        : {num_layers} (rank of E8 root system)")
    print(f"E8 seed          : {e8_seed} (root count marker for determinism)")
    print(f"Bell weights     : Phi+ = {weights['phi_plus']:.2f} (dominant), "
          f"others = {weights['phi_minus']:.2f} each")
    print("-" * 70)
    print("Quantum → Audio Mapping:")
    print("  • |Φ⁺⟩ (dominant) : In-phase f & f·φ superposition + fractal copies")
    print("  • |Φ⁻⟩            : Anti-phase (minus sign) → cancellations & roughness")
    print("  • |Ψ⁺⟩ / |Ψ⁻⟩     : Odd-parity frequency pair (φ⁻¹ and φ² scalings)")
    print("  • Entanglement sim: Identical stereo channels (perfect correlation)")
    print("-" * 70)
    print("Mathematical notes:")
    print("  • Spectrum is self-similar (fractal) under multiplication by φ")
    print("  • All partials are deterministic sines — no noise, no RNG in synthesis")
    print("  • Classical simulation only: true QM non-locality cannot be reproduced")
    print("    in a local classical wave equation, but correlation is faithfully")
    print("    demonstrated via identical channels.")
    print("=" * 70)

    return output_path


if __name__ == "__main__":
    generate_e8_bell_sonification()