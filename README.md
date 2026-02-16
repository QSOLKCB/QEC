QEC v2.1.0 — Invariant-Hardened Stabilizer Stack + QLDPC + Golay-Class Logic

![Version](https://img.shields.io/badge/version-v2.1.0-blue)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18660270.svg)](https://doi.org/10.5281/zenodo.18660270)
![License](https://img.shields.io/badge/license-CC--BY--4.0-lightgrey)

Overview

QEC is a research-grade quantum error correction toolkit exploring:

Non-binary stabilizer codes

Lattice-informed decoding

Modern quantum LDPC constructions across multiple local dimensions

Version v2.1.0 is an invariant-hardening release.

It formalizes additive lift invariants for shared-circulant CSS constructions, providing algebraic guarantees of lifted orthogonality and eliminating prior probabilistic edge-case failures.

All constructions are deterministic, seeded, and invariant-checked by design.

Existing qutrit Golay and ququart lattice systems remain fully supported.

What’s New in v2.1.0
Additive Lift Invariants (Hardening Update)

The QLDPC lifting layer now uses structured additive shifts:

s(i, j) = (r_i + c_j) mod L


This guarantees:

H_X · H_Z^T = 0 (mod 2)


whenever base protograph matrices satisfy orthogonality.

Why This Matters

No per-edge random lift tables

No post hoc orthogonality repair

No probabilistic failure modes

CSS orthogonality follows algebraically

Deterministic across processes

This release moves the construction from empirically stable to mathematically enforced.

No architectural changes from v2.0.0 — this is a structural invariant hardening release.

Protograph-Based Quantum LDPC CSS Codes

Module: src/qec_qldpc_codes.py

Implements quantum LDPC CSS codes built from orthogonal protograph pairs over GF(2^e), following the Komoto–Kasai (2025) construction paradigm.

Key Properties

Protograph-based LDPC ensembles (column weight 2)

Shared circulant lifting per protograph edge

Additive invariant lift structure

Deterministic, seeded construction

Binary expansion via GF(2^e) lifting

Sparse-safe invariant enforcement

Honest benchmarking against the hashing bound (asymptotic reference)

Supported Predefined Code Rates

0.50

0.60

0.75

Minimal Usage Example
from src.qec_qldpc_codes import QuantumLDPCCode

code = QuantumLDPCCode.from_predefined(rate=0.50, e=8, P=128, seed=42)
print(code.n, code.k)


Hashing-bound comparisons are benchmarks, not finite-length guarantees.

Ternary Golay Qutrit Code ([[11,1,5]]₃)

Module: src/qec_golay.py

Full implementation of the unique perfect ternary Golay code.

Classical parameters: [11, 6, 5]₃

Quantum CSS lift: [[11,1,5]]₃

Corrects any single-qutrit error

Encodes one logical qutrit into eleven physical qutrits

Parity-Check Matrix over GF(3)
H =
[1 0 0 0 0 1 1 1 2 2 0]
[0 1 0 0 0 1 1 2 1 0 2]
[0 0 1 0 0 1 2 1 0 1 2]
[0 0 0 1 0 1 2 0 1 2 1]
[0 0 0 0 1 1 0 2 2 1 1]


Self-orthogonal over GF(3)

Nullspace generates 729 exact codewords

Fully CSS-compatible for qutrit stabilizers

Ququart Stabilizer Code (d = 4)

Module: src/qec_ququart.py

Encodes a logical ququart using repetition-style stabilizers.

Logical Basis States
|j_L> = |j, j, j>     for j ∈ {0,1,2,3}

Stabilizers
S1 = Z1 Z2^-1
S2 = Z2 Z3^-1

Logical Operators
X_L = X1 X2 X3
Z_L = Z1
