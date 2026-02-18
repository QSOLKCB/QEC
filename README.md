QSOLKCB / QEC

Quantum LDPC CSS Construction and Decoder Toolkit

[![v2.3.0](https://img.shields.io/badge/version-v2.3.0-blue)](https://github.com/QSOLKCB/QEC/releases/tag/v2.3.0)
[![DOI v2.3.0](https://zenodo.org/badge/DOI/10.5281/zenodo.18679878.svg)](https://doi.org/10.5281/zenodo.18679878)
&nbsp;&nbsp;
[![v2.2.0](https://img.shields.io/badge/version-v2.2.0-blue)](https://github.com/QSOLKCB/QEC/releases/tag/v2.2.0)
[![DOI v2.2.0](https://zenodo.org/badge/DOI/10.5281/zenodo.18679203.svg)](https://doi.org/10.5281/zenodo.18679203)
&nbsp;&nbsp;
![License](https://img.shields.io/badge/license-CC--BY--4.0-lightgrey)

Deterministic construction and decoding framework for QLDPC CSS codes with algebraic invariant guarantees and numerically stable belief propagation.

Release Lineage
v2.3.0 — Decoder Utility Formalization and Stability Refinement

DOI: https://doi.org/10.5281/zenodo.18679878

This release formalizes the decoder layer into standalone utilities while preserving full backward compatibility.

Highlights

Explicit detection → inference → correction separation

Standalone bp_decode operating on per-variable LLR vectors

Pauli-frame update abstraction (update_pauli_frame)

Channel LLR modeling with optional bias weighting

Input validation enforcing p ∈ (0, 1)

Micro-optimization of BP early-stop casting

101 / 101 tests passing

Construction layer remains algebraically guaranteed (v2.1.0).

v2.2.0 — Belief Propagation Stability Hardening

DOI: https://doi.org/10.5281/zenodo.18679203

Released in parallel with the v2.3.0 refinement cycle.

Highlights

Correct handling of degree-1 check nodes

Prevented artificial LLR amplification from atanh(≈1)

Eliminated false confidence injection in sparse Tanner graphs

Numerical stabilization of sum-product decoding

No architectural changes. Decoder logic stability hardening only.

v2.1.0 — Additive Lift Invariant Hardening

DOI: https://doi.org/10.5281/zenodo.18660270

Highlights

Additive lift structure:

s(i, j) = (r_i + c_j) mod L


Algebraic guarantee of lifted CSS orthogonality

Sparse-safe GF(2) rank computation

Deterministic seeded construction

89 / 89 invariant tests passing

Construction layer transitioned from empirically stable → structurally guaranteed.

v2.0.0 — Architectural Expansion

Initial multidimensional QLDPC CSS stack:

Protograph-based construction

GF(2^e) lifting

Ternary Golay [[11,1,5]]₃

Ququart stabilizer + D4 lattice prior

Deterministic construction framework

Current System State

With v2.3.0:

Construction layer is algebraically enforced

Decoder layer is numerically stable under sparse edge cases

Detection, inference, and correction are modular and test-covered

Fully deterministic seeded workflow

101 total tests passing
