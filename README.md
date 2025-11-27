🧬 New in v1.5 — Ququart + Geometry Layer
Ququart Stabilizer Code (d = 4)

A proper ℤ₄ stabilizer code implemented in src/qec_ququart.py.

Codewords:

|j_L⟩ = |j, j, j⟩   for j ∈ {0,1,2,3}


Stabilizers:

S1 = Z1 · Z2^{-1}
S2 = Z2 · Z3^{-1}


Logical Operators:

XL = X1 · X2 · X3
ZL = Z1


This is the first QSOL-IMC demonstration of higher-dimensional QEC (d = 4) integrated directly into the existing stack.

High-Density Lattice Geometry Layer

Implemented in:

src/ququart_lattice_prior.py


The geometry layer projects logical-amplitude vectors in ℝ⁴ onto:

Z⁴ — baseline cubic lattice

D₄ — dense lattice (E8-surrogate)

This projection acts as a geometric pre-decoder:

reduces effective noise

sharpens logical amplitudes

increases threshold performance

creates lattice-stabilized logical states

This is the first demonstration of lattice geometry stabilizing a ququart code.

Threshold Benchmarks (Baseline vs Geometry)

New figures added:

ququart_threshold.png

ququart_lattice_prior_threshold.png

These compare:

raw [[3,1]]₄ ququart stabilizer performance

geometry-enhanced performance using D₄

Across the entire range of physical error rates,
D₄ consistently reduces the logical error rate.

🎧 Track Summaries (unchanged from v1.1)
Track	QEC Model / Concept	Sonic Behavior
QSOL Triplet Polymeter	Steane [[7,1,3]] baseline	Clean E-minor triad; tight, stable, low-noise harmonic field.
e8_triality	Fusion-QEC / photonic triality	Lattice-stable overtone network; coherent reference tone.
QEC Fault Lines	Threshold turbulence	Compression + stereo widening scale with p_phys.
Spectral Algebraics Live	Post-threshold collapse	Dense overtone cloud; diffuse reverb; decoherence aesthetic.
🧩 QEC ↔ Audio Analogues
Regime	Physical Error Rate	Sonic Condition
Stable	< 1e-5	Clean, coherent, narrow-band.
Transition	1e-5 → 1e-3	Rising density; thickening spectra; harmonic pressure.
Critical	> 1e-3	Saturated, diffuse, wide; motifs collapse completely.

The new ququart geometry layer introduces novel sonic behaviors tied to D₄-stabilized states.

⚙️ Simulation Engine
Steane Fast Simulator
src/steane_numpy_fast.py

New: Ququart QEC Stack
src/qec_ququart.py
src/qudit_stabilizer.py
src/ququart_lattice_prior.py

Example Scripts
examples/ququart_threshold_demo.py
examples/ququart_threshold_with_prior.py


Provides:

deterministic Monte Carlo

unified stabilizer formalism (arbitrary d)

geometry-prior augmentation

full ququart benchmarking

🧾 License

Creative Commons Attribution 4.0 International (CC BY 4.0)

🔖 Citation

Slade, T. (2025). QSOLKCB/QEC: QEC v1.5 — Ququart Stabilizer Code + High-Density Geometry Layer (v1.5-ququart-geometry). Zenodo. https://doi.org/10.5281/zenodo.17742258

🏷️ Keywords

quantum error correction · ququart · qudit stabilizer · D4 lattice ·
geometry layer · spectral algebraics · sonification ·
QSOL-IMC · E8-inspired · threshold curves
