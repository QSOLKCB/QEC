# QEC v1.5 — Ququart Stabilizer Code + High-Density Geometry Layer  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17742258.svg)](https://doi.org/10.5281/zenodo.17742258)
![GitHub release (latest by tag)](https://img.shields.io/github/v/release/QSOLKCB/QEC?label=release)


# QEC v1.5 — Ququart Stabilizer Code + High-Density Geometry Layer

## 🧬 What's New in v1.5

### Ququart Stabilizer Code (d = 4)

A proper ℤ₄ stabilizer code implemented in `src/qec_ququart.py`.

**Codewords:**
```
|jₗ⟩ = |j, j, j⟩   for j ∈ {0, 1, 2, 3}
```

**Stabilizers:**
```
S₁ = Z₁ · Z₂⁻¹
S₂ = Z₂ · Z₃⁻¹
```

**Logical Operators:**
```
Xₗ = X₁ · X₂ · X₃
Zₗ = Z₁
```

This is the first QSOL-IMC demonstration of higher-dimensional QEC (d = 4) integrated directly into the existing stack.

### High-Density Lattice Geometry Layer

**Implementation:**
- `src/ququart_lattice_prior.py`

The geometry layer projects logical-amplitude vectors in ℝ⁴ onto:
- **ℤ⁴** — baseline cubic lattice
- **D₄** — dense lattice (E8-surrogate)

This projection acts as a geometric pre-decoder that:
- Reduces effective noise
- Sharpens logical amplitudes
- Increases threshold performance
- Creates lattice-stabilized logical states

This is the first demonstration of lattice geometry stabilizing a ququart code.

### Threshold Benchmarks (Baseline vs Geometry)

New figures added:
- `ququart_threshold.png`
- `ququart_lattice_prior_threshold.png`

These compare:
- Raw [[3,1]]₄ ququart stabilizer performance
- Geometry-enhanced performance using D₄

**Result:** Across the entire range of physical error rates, D₄ consistently reduces the logical error rate.

## 🎧 Track Summaries

| Track | QEC Model / Concept | Sonic Behavior |
|-------|---------------------|----------------|
| QSOL Triplet Polymeter | Steane [[7,1,3]] baseline | Clean E-minor triad; tight, stable, low-noise harmonic field |
| e8_triality | Fusion-QEC / photonic triality | Lattice-stable overtone network; coherent reference tone |
| QEC Fault Lines | Threshold turbulence | Compression + stereo widening scale with p_phys |
| Spectral Algebraics Live | Post-threshold collapse | Dense overtone cloud; diffuse reverb; decoherence aesthetic |

## 🧩 QEC ↔ Audio Analogues

| Regime | Physical Error Rate | Sonic Condition |
|--------|---------------------|-----------------|
| Stable | < 1×10⁻⁵ | Clean, coherent, narrow-band |
| Transition | 1×10⁻⁵ → 1×10⁻³ | Rising density; thickening spectra; harmonic pressure |
| Critical | > 1×10⁻³ | Saturated, diffuse, wide; motifs collapse completely |

The new ququart geometry layer introduces novel sonic behaviors tied to D₄-stabilized states.

## ⚙️ Simulation Engine

### Core Components

**Steane Fast Simulator:**
- `src/steane_numpy_fast.py`

**Ququart QEC Stack:**
- `src/qec_ququart.py`
- `src/qudit_stabilizer.py`
- `src/ququart_lattice_prior.py`

**Example Scripts:**
- `examples/ququart_threshold_demo.py`
- `examples/ququart_threshold_with_prior.py`

### Features

- Deterministic Monte Carlo simulation
- Unified stabilizer formalism (arbitrary d)
- Geometry-prior augmentation
- Full ququart benchmarking

## 🧾 License

[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

## 🔖 Citation
```bibtex
@software{slade_2025_qsolkcb,
  author       = {Slade, T.},
  title        = {QSOLKCB/QEC: QEC v1.5 — Ququart Stabilizer Code + High-Density Geometry Layer},
  year         = {2025},
  version      = {v1.5-ququart-geometry},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17742258},
  url          = {https://doi.org/10.5281/zenodo.17742258}
}
```

## 🏷️ Keywords

quantum error correction · ququart · qudit stabilizer · D4 lattice · geometry layer · spectral algebraics · sonification · QSOL-IMC · E8-inspired · threshold curves
