QEC v2.0.0 — Multidimensional Stabilizer Stack + QLDPC + Golay-Class Logic




Overview

QEC is a research-grade quantum error correction toolkit exploring:

Non-binary stabilizer codes

Lattice-informed decoding

Modern quantum LDPC constructions across multiple local dimensions

Version v2.0.0 marks a major architectural milestone. QEC now includes state-of-the-art protograph-based quantum LDPC CSS codes following the Komoto–Kasai (2025) construction paradigm.

CSS orthogonality is enforced by construction rather than repaired post hoc, and deterministic, invariant-safe code generation is a core guarantee. Existing qutrit Golay and ququart lattice systems remain fully supported.

This release unifies finite-field QLDPC theory, non-binary stabilizers, and geometric decoding priors in a single framework.

What’s New in v2.0.0
Protograph-Based Quantum LDPC CSS Codes (NEW)

Module: src/qec_qldpc_codes.py

This release adds a full implementation of quantum LDPC CSS codes built from orthogonal protograph pairs over GF(2^e), following the Komoto–Kasai construction style.

Key Properties

Protograph-based LDPC ensembles with column weight 2

Shared circulant lifting per protograph edge

CSS condition H_X · H_Z^T = 0 enforced structurally

Deterministic, seeded construction

Binary expansion via GF(2^e) lifting

Honest benchmarking against the hashing bound as an asymptotic reference

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

QEC includes a full implementation of the ternary Golay code, the unique perfect linear code over GF(3).

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


The matrix is self-orthogonal over GF(3). Its nullspace generates 729 exact codewords and is fully CSS-compatible for qutrit stabilizers.

Ququart Stabilizer Code (d = 4)

Module: src/qec_ququart.py

Encodes a logical ququart using repetition-style stabilizers.

Logical Basis States
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
for 
𝑗
∈
{
0
,
1
,
2
,
3
}
∣j
L
	​

⟩=∣j,j,j⟩for j∈{0,1,2,3}
Stabilizers

𝑆
1
=
𝑍
1
⋅
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

⋅Z
2
−1
	​


𝑆
2
=
𝑍
2
⋅
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

⋅Z
3
−1
	​


Logical Operators

𝑋
𝐿
=
𝑋
1
⋅
𝑋
2
⋅
𝑋
3
X
L
	​

=X
1
	​

⋅X
2
	​

⋅X
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


High-Density Geometry Layer (D4 Prior)

Module: src/ququart_lattice_prior.py

Projects logical amplitudes into 
𝑍
4
Z
4
 (baseline) and the dense D4 lattice (an E8-surrogate geometry).

This layer acts as a geometric pre-decoder that:

Compresses noise

Sharpens amplitudes

Lowers logical error rates

Produces lattice-stabilized logical states

Threshold and Benchmarking

LDPC simulations include frame error rate versus physical error probability studies.

The hashing bound is used as an asymptotic reference, not a finite-length prediction.

The D4 lattice prior strictly improves ququart logical error rates across tested regimes.

Core Simulation Stack

src/qec_qldpc_codes.py

src/qec_golay.py

src/qec_ququart.py

src/qudit_stabilizer.py

src/ququart_lattice_prior.py

src/steane_numpy_fast.py

License

Creative Commons Attribution 4.0 International (CC BY 4.0)
https://creativecommons.org/licenses/by/4.0/

Citation (v2.0.0)
@software{slade_2026_qsolkcb_qec,
  author    = {Slade, T.},
  title     = {QSOLKCB/QEC: Quantum Error Correction Toolkit v2.0.0},
  year      = {2026},
  version   = {v2.0.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.17742258},
  url       = {https://doi.org/10.5281/zenodo.17742258}
}

Keywords

quantum error correction · QLDPC · CSS codes · protograph LDPC · qutrit · ququart · Golay code · non-binary stabilizer · D4 lattice · finite-field lifting · hashing bound · QSOL-IMC
