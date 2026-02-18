Changelog

All notable changes to this project will be documented in this file.

[2.2.0] — 2026-02-18
Added

Degree-1 check node handling in JointSPDecoder belief-propagation loop

Explicit zero extrinsic message for single-neighbor check nodes

Improved numerical stability of sum-product decoding on sparse Tanner graphs

Changed

Corrected check-to-variable message update rule in _bp_component:

Degree-1 check nodes now return 0.0 (no extrinsic information)
instead of falling through to the general tanh-product rule

Prevents artificial LLR amplification from:
arctanh(≈1) when product over empty neighbor set occurs

Fixed

Eliminated false confidence injection in BP decoding for degree-1 parity checks

Resolved potential instability under very sparse or irregular parity structures

Notes

No changes to construction layer

No changes to additive lift invariants

No changes to CSS orthogonality logic

All tests passing (65/65)

Decoder stability hardening release

[2.1.0] — 2026-02-16
Added

Additive lift invariant formalization for shared-circulant QLDPC CSS constructions

Deterministic structured shift mapping
s(i, j) = (r_i + c_j) mod L

Algebraic guarantee of lifted CSS orthogonality

Invariant enforcement via sparse-safe orthogonality checks

Binary GF(2) rank computation without dense float conversion

Expanded invariant test coverage (89/89 passing)

Changed

Replaced per-edge random lift tables with additive invariant lift structure

Lift implementation is now deterministic, process-independent, and order-independent

Orthogonality now follows structurally from base-matrix commutation

Removed

Probabilistic orthogonality edge-case behavior from prior lift implementation

Notes

No architectural changes from v2.0.0

Structural invariant hardening release

[2.0.0] — 2026-??-??
Added

Multidimensional stabilizer stack

Protograph-based QLDPC CSS codes

GF(2^e) finite-field lifting

Ternary Golay [[11,1,5]]₃ implementation

Ququart stabilizer and D4 lattice prior layer

Deterministic seeded construction framework
