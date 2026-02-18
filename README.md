QSOLKCB / QEC — Quantum Error Correction (QLDPC CSS Toolkit)

[![v2.3.0](https://img.shields.io/badge/version-v2.3.0-blue)](https://github.com/QSOLKCB/QEC/releases/tag/v2.3.0)
[![DOI v2.3.0](https://zenodo.org/badge/DOI/10.5281/zenodo.18679878.svg)](https://doi.org/10.5281/zenodo.18679878)
&nbsp;&nbsp;
[![v2.2.0](https://img.shields.io/badge/version-v2.2.0-blue)](https://github.com/QSOLKCB/QEC/releases/tag/v2.2.0)
[![DOI v2.2.0](https://zenodo.org/badge/DOI/10.5281/zenodo.18679203.svg)](https://doi.org/10.5281/zenodo.18679203)
&nbsp;&nbsp;
![License](https://img.shields.io/badge/license-CC--BY--4.0-lightgrey)

Deterministic quantum error correction framework for QLDPC CSS codes with algebraic construction guarantees, numerically stable belief propagation, and modular decoder utilities.

Release Lineage
v2.3.0 — Decoder Utility Formalization and Stability Refinement

DOI: https://doi.org/10.5281/zenodo.18679878

This release formalizes the decoder layer into standalone utilities while preserving full backward compatibility with JointSPDecoder.

Highlights

Explicit detection → inference → correction separation

Standalone bp_decode with per-variable LLR input

Pauli-frame update abstraction (update_pauli_frame)

Channel LLR modeling with optional scalar or vector bias

Enforced input validation (p ∈ (0, 1))

Reduced per-iteration overhead in BP early-stop logic

101 / 101 tests passing

Construction layer remains algebraically guaranteed (v2.1.0).

v2.2.0 — Belief Propagation Stability Hardening

DOI: https://doi.org/10.5281/zenodo.18679203

Numerical stability refinement of the sum-product decoder.

Highlights

Correct handling of degree-1 check nodes

Eliminated artificial LLR amplification from atanh(≈1)

Removed false confidence injection in sparse Tanner graphs

Stabilized belief-propagation behavior under irregular parity structures

No architectural changes. Decoder logic stabilization only.

v2.1.0 — Additive Lift Invariant Hardening

DOI: https://doi.org/10.5281/zenodo.18660270

Transition from empirically stable lifting to algebraically guaranteed construction.

Additive lift structure:

s(i, j) = (r_i + c_j) mod L


Highlights

Algebraic guarantee of lifted CSS orthogonality

Sparse-safe GF(2) rank computation

Deterministic seeded construction

89 / 89 invariant tests passing

Construction layer transitioned from probabilistic behavior → structural invariance.

v2.0.0 — Architectural Expansion

Initial multidimensional QLDPC CSS stack:

Protograph-based construction

GF(2^e) lifting

Ternary Golay [[11,1,5]]₃

Ququart stabilizer + D4 lattice prior

Deterministic construction framework

Current System State (v2.3.0)

Construction layer is algebraically enforced

Decoder layer is numerically stable under sparse edge cases

Detection, inference, and correction are modular and independently test-covered

Fully deterministic seeded workflow

101 total tests passing

Architecture Overview
Channel Model      → channel_llr
Detection          → syndrome / detect
Inference          → bp_decode / infer
Correction         → update_pauli_frame
Construction Layer → Additive invariant QLDPC CSS lift


The framework separates algebraic construction guarantees from numerically stable belief-propagation decoding, enabling deterministic, test-covered workflows from channel modeling to Pauli-frame correction.
