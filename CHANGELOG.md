Changelog

All notable changes to this project will be documented in this file.

The project follows semantic versioning.
Each release reflects structural, numerical, or architectural maturity improvements in the QLDPC CSS construction and decoding stack.

[2.3.0] — 2026-02-18

Decoder Utility Formalization and Stability Refinement

Added

Standalone decoder utility layer formalizing detection–inference–correction separation:

update_pauli_frame(frame, correction) — pure GF(2) Pauli-frame XOR update (non-mutating, validated).

syndrome(H, e) — standalone binary syndrome computation.

bp_decode(H, llr, max_iter, syndrome_vec) — standalone belief-propagation decoder operating on per-variable LLR vectors.

detect(H, e) — thin wrapper over syndrome.

infer(H, llr, max_iter, syndrome_vec) — thin wrapper over bp_decode.

channel_llr(e, p, bias) — channel LLR computation with optional scalar or per-variable bias weighting.

36 new unit tests covering:

Pauli-frame algebra

Syndrome equivalence

BP determinism and convergence

Channel LLR validation and bias behavior

Integration with decoding workflow

Changed

channel_llr now enforces p ∈ (0, 1) to prevent undefined or numerically unstable boundary behavior.

bp_decode now precomputes integer-casted parity-check matrix and syndrome vectors for early-stopping checks, eliminating repeated per-iteration casting.

Decoder workflow is now explicitly modular while remaining backward compatible.

Notes

No changes to construction layer.

No changes to additive lift invariants.

No changes to CSS orthogonality guarantees.

No changes to JointSPDecoder public API.

Fully backward compatible.

All tests passing (101 / 101).

[2.2.0] — 2026-02-18

Belief-Propagation Stability Hardening

Added

Explicit handling of degree-1 check nodes in the JointSPDecoder belief-propagation loop.

Zero extrinsic message returned for single-neighbor check nodes.

Changed

Corrected check-to-variable update rule in _bp_component:

Degree-1 check nodes now return 0.0 (no extrinsic information)
instead of falling through to the general tanh-product rule.

This prevents artificial LLR amplification from:

atanh(≈1) → ∞


when the product over an empty neighbor set numerically approaches unity.

Fixed

Eliminated false confidence injection in BP decoding for sparse parity structures.

Resolved numerical instability in extremely sparse or irregular Tanner graphs.

Notes

No changes to construction layer.

No changes to additive lift invariants.

No changes to CSS orthogonality logic.

All tests passing (65 / 65).

Decoder stability hardening release.

[2.1.0] — 2026-02-16

Additive Lift Invariant Hardening

Added

Additive lift invariant formalization for shared-circulant QLDPC CSS constructions.

Deterministic structured shift mapping:

s(i, j) = (r_i + c_j) mod L


Algebraic guarantee of lifted CSS orthogonality.

Sparse-safe orthogonality verification.

Binary GF(2) rank computation without dense float conversion.

Expanded invariant test coverage (89 / 89 passing).

Changed

Replaced per-edge random lift tables with additive invariant lift structure.

Lift implementation is now deterministic, process-independent, and order-independent.

Orthogonality now follows structurally from base-matrix commutation.

Removed

Probabilistic orthogonality edge-case behavior from prior lift implementation.

Notes

No architectural changes from v2.0.0.

Structural invariant hardening release.

[2.0.0] — 2026-02-??

Architectural Expansion of QLDPC CSS Stack

Added

Multidimensional stabilizer stack.

Protograph-based QLDPC CSS constructions.

GF(2^e) finite-field lifting framework.

Ternary Golay [[11,1,5]]₃ implementation.

Ququart stabilizer and D4 lattice prior layer.

Deterministic seeded construction framework.

Integrated simulation and hashing bound tooling.

Notes

Major architectural rewrite establishing the construction and decoding foundation for subsequent invariant hardening and stability refinement releases.
