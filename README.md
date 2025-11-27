🎛️ Spectral QEC Benchmark Set v1.5 — Ququart Geometry Expansion
Sonified & Geometric Thresholds of Quantum Error Correction

Author: Trent Slade (QSOL-IMC)
Version: 1.5 — November 2025 (Ququart Geometry Release)
License: CC BY 4.0
DOI:

🧠 Overview

Version 1.5 introduces a major architectural upgrade to the QSOL-IMC Spectral QEC Benchmark Set:

✔ Ququart (d = 4) Stabilizer Code

A full [[3,1]]₄ code using generalized Pauli operators 
𝑋
4
X
4
	​

 and 
𝑍
4
Z
4
	​

, embedded into a dimension-agnostic stabilizer engine.

✔ High-Density Geometry Layer (D₄ / E8-inspired)

Logical-amplitude vectors in ℝ⁴ undergo lattice projection (Z⁴ or D₄).
This acts as a geometric pre-decoder, compressing noise before stabilizer decoding.

✔ Geometry-Augmented Threshold Curves

Monte Carlo simulations now support “baseline vs geometry-prior” comparative studies.

✔ Full Qudit Engine (arbitrary d)

The new module qudit_stabilizer.py supports qutrits, ququarts, and higher qudits.

This expands the QSOL-IMC Spectral Algebraics program into the higher-dimensional qudit regime, where geometry and stabilizers co-operate as information-protective structures.
</pre>
📂 Contents
Spectral-QEC-Benchmark-Set-v1.5/
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
│   ├── QEC_threshold_curves.png
│   ├── ququart_threshold.png
│   └── ququart_lattice_prior_threshold.png
│
├── qc_benchmark_data/
│   ├── QEC_Benchmark_Report.pdf
│   └── benchmark_table.csv
│
├── src/
│   ├── steane_numpy_fast.py
│   ├── qec_ququart.py
│   ├── qudit_stabilizer.py
│   └── ququart_lattice_prior.py
│
├── examples/
│   ├── ququart_threshold_demo.py
│   └── ququart_threshold_with_prior.py
│
├── LICENSE.txt
└── README.md
</pre>
🧬 New in v1.5 — Ququart + Geometry Layer
1. Ququart Stabilizer Code (d = 4)

A true ℤ₄ stabilizer code:

Codewords: 
∣
𝑗
𝐿
⟩
=
∣
𝑗
,
𝑗
,
𝑗
⟩
∣j
L
	​

⟩=∣j,j,j⟩

Stabilizers:

𝑆
1
=
𝑍
1
𝑍
2
−
1
S
1
	​

=Z
1
	​

Z
2
−1
	​


𝑆
2
=
𝑍
2
𝑍
3
−
1
S
2
	​

=Z
2
	​

Z
3
−1
	​


Logical operators:

𝑋
𝐿
=
𝑋
1
𝑋
2
𝑋
3
X
L
	​

=X
1
	​

X
2
	​

X
3
	​


𝑍
𝐿
=
𝑍
1
Z
L
	​

=Z
1
	​


This demonstrates higher-dimensional QEC inside the QSOL-IMC framework.

2. High-Density Lattice Geometry Layer

The new module:

src/ququart_lattice_prior.py


projects logical amplitudes onto:

Z⁴ (baseline)

D₄ (dense — E8-surrogate)

This geometric “snap-to-structure” prior:

reduces effective noise

sharpens logical amplitudes

raises the effective QEC threshold

acts as a geometry-driven pre-decoder

This is the first demonstration of lattice geometry stabilizing a ququart code.

3. Threshold Benchmarks (Baseline vs Geometry)

New figures:

ququart_threshold.png

ququart_lattice_prior_threshold.png

These compare:

Raw ququart stabilizer performance

Geometry-enhanced performance

The D₄ prior exhibits lower logical error rates across the entire range.

🎧 Track Summaries (unchanged from v1.1)
Track	QEC Model / Concept	Sonic Behavior
QSOL Triplet Polymeter	Steane [[7,1,3]] baseline	Clean triad in E-minor; tight and stable.
e8_triality	Fusion-QEC / photonic triality	Lattice-stable overtone network; coherent reference tone.
QEC Fault Lines Sonification	Pseudo-threshold turbulence	Rising noise mapped to compression & stereo width.
Spectral Algebraics Live	Post-threshold collapse	Intentional decoherence aesthetic; thick spectral fog.
🧩 QEC ↔ Audio Analogues
Regime	Physical Error Rate	Sonic Condition
Stable	< 1e-5	Coherent, clean, narrow-band.
Transition	1e-5 → 1e-3	Pressure increase; spectral thickening; harmonic instability.
Critical	> 1e-3	Saturated + diffuse; stereo blows open; motifs collapse.

The ququart geometry layer allows exploration of new sonic QEC artifacts.

⚙️ Simulation Engine
Steane Fast Simulator
src/steane_numpy_fast.py

New: Ququart QEC Stack
src/qec_ququart.py
src/qudit_stabilizer.py
src/ququart_lattice_prior.py

New Example Scripts
examples/ququart_threshold_demo.py
examples/ququart_threshold_with_prior.py


These provide:

deterministic Monte Carlo

unified stabilizer formalism

geometry-prior augmentation

ququart benchmarking

🧾 License

Creative Commons Attribution 4.0 International (CC BY 4.0)

🔖 Citation

Slade, T. (2025). QSOLKCB/QEC: QEC v1.5 — Ququart Stabilizer Code + High-Density Geometry Layer (v1.5-ququart-geometry). Zenodo. https://doi.org/10.5281/zenodo.17742258

🏷️ Keywords

quantum error correction · ququart · qudit stabilizer ·
D4 lattice · geometry layer · spectral algebraics ·
sonification · QSOL-IMC · E8-inspired · threshold curves
