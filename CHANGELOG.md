Changelog

All notable changes to this project will be documented in this file.

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
