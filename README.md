---

# 🎛️ Spectral QEC Benchmark Set v1

### Sonified Thresholds of Quantum Error Correction

Author:** Trent Slade (QSOL-IMC)
Version:** 1.1 — November 2025
License:** CC BY 4.0
DOI:** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17742258.svg)](https://doi.org/10.5281/zenodo.17742258)

---

## 🧠 Overview

**Spectral QEC Benchmark Set v1** merges *quantum error correction* with *spectral audio design*, converting stability curves into sonic behavior.
Each audio track corresponds to a QEC model and uses real logical-error estimates to drive mix parameters such as compression, reverb, saturation, and stereo width.

The dataset integrates:

* High-speed Steane [[7,1,3]] Monte Carlo (pure NumPy)
* Threshold-walk sonification (p_phys → mix automation)
* Spectral analyses and overlays
* Full benchmark report + figures

This continues the QSOL-IMC *Spectral Algebraics* program, embedding physical structure directly into sound.

---

## 📂 Contents

```
Spectral-QEC-Benchmark-Set-v1/
│
├── audio/
│   ├── QSOL_Triplet_Polymeter – Producer Bounce.mp3
│   ├── e8_triality.wav
│   ├── QEC_Fault_Lines_Sonification.mp3
│   └── Spectral_Algebraics_Live – Quantum Nostalgia Ambient.wav
│
├── spectra/
│   ├── *_spectrum.png
│   └── all_tracks_overlay.png
│
├── analysis/
│   ├── threshold_walk_automation.csv
│   ├── automation_plot.png
│   ├── QEC_vs_Audio_table.csv
│   └── QEC_threshold_curves.png
│
├── qc_benchmark_data/
│   ├── QEC_Benchmark_Report.pdf
│   └── benchmark_table.csv
│
├── src/
│   └── steane_numpy_fast.py
│
├── LICENSE.txt
└── README.md
```

---

## 🎧 Track Summaries

| Track                            | QEC Model / Concept            | Sonic Behavior                                                              |
| -------------------------------- | ------------------------------ | --------------------------------------------------------------------------- |
| **QSOL Triplet Polymeter**       | Steane [[7,1,3]] baseline      | Clean triad in E-minor (E≈165 Hz); tight, stable, low-noise harmonic field. |
| **e8_triality**                  | Fusion-QEC / photonic triality | Lattice-stable overtone network; highly coherent reference tone.            |
| **QEC Fault Lines Sonification** | Pseudo-threshold turbulence    | Compression + stereo widening scale with p_phys; rising spectral fog.       |
| **Spectral Algebraics Live**     | Post-threshold collapse        | Dense overtone cloud; diffuse reverb; intentional decoherence aesthetic.    |

---

## ⚙️ Analysis Files

| File                              | Purpose                                                     |
| --------------------------------- | ----------------------------------------------------------- |
| **threshold_walk_automation.csv** | Mapping of p_phys → compression, reverb, tape saturation.   |
| **automation_plot.png**           | Visualization of threshold-walk automation curves.          |
| **QEC_vs_Audio_table.csv**        | Cross-correlation of spectral features vs logical error.    |
| **QEC_Benchmark_Report.pdf**      | Full analytic write-up of stability and threshold behavior. |

---

## 🔢 Threshold-Walk Mapping

| p_phys | Comp Ratio | Reverb Wet | Tape Drive |
| ------ | ---------- | ---------- | ---------- |
| 1e-6   | 1.2        | 0.12       | 0.0        |
| 1e-5   | 1.3        | 0.16       | 0.05       |
| 1e-4   | 1.6        | 0.30       | 0.20       |
| 5e-4   | 1.7        | 0.38       | 0.35       |
| 1e-3   | 1.8        | 0.45       | 0.50       |
| 5e-3   | 2.0        | 0.62       | 0.70       |
| 1e-2   | 2.5        | 0.80       | 1.00       |

These traces approximate an auditory walk from *stable error correction* → *threshold turbulence* → *full decoherence*.

---

## 🧩 QEC ↔ Audio Analogues

| Regime         | Physical Error Rate | Sonic Condition                                                  |
| -------------- | ------------------- | ---------------------------------------------------------------- |
| **Stable**     | < 1e-5              | Dry, precise, coherent; minimal spectral smear.                  |
| **Transition** | 1e-5 → 1e-3         | Increasing density, compression, and harmonic pressure.          |
| **Critical**   | > 1e-3              | Saturated, diffuse, stereo-wide; breakdown of structured motifs. |

---

## 🧬 Simulation Engine (New)

The repository now includes a **pure-NumPy**, fully vectorized Steane [[7,1,3]] simulator:

```
src/steane_numpy_fast.py
```

Features:

* deterministic RNG
* auto-derived Hamming decoder
* no external quantum libraries
* chunked or full-array Monte Carlo
* baseline for all threshold-walk mappings

This replaces previous scripts and provides a clean foundation for further QEC sonification work.

---

## 🧾 License

Creative Commons Attribution 4.0 International (CC BY 4.0)

You are free to share or adapt this material with attribution.

---

## 🔖 Citation

> Slade, T. (2025). *Spectral QEC Benchmark Set v1 — Sonified Thresholds of Quantum Error Correction.* Zenodo. DOI to be assigned.

---

## 🏷️ Keywords

`quantum error correction` · `spectral algebraics` · `sonification` · `industrial electronic`
`audio dataset` · `E-minor` · `QEC stability` · `physics-in-sound` · `QSOL-IMC`

---
