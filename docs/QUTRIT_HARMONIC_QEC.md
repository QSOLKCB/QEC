# Exact qutrit QEC with harmonic syndrome observation

This slice adds real qutrit error correction without changing the legacy
three-message decoder. The correction kernel is finite-field stabilizer
algebra. Sonification is a redundant classical observation channel around the
syndrome, not the mechanism that makes the quantum code correct.

## Mathematical core

For a qutrit, let

\[
X|j\rangle=|j+1\bmod 3\rangle,\qquad
Z|j\rangle=\omega^j|j\rangle,\qquad
\omega=e^{2\pi i/3}.
\]

An \(n\)-qutrit Pauli error is represented by
\((x\mid z)\in\mathbb F_3^{2n}\). Its syndrome against stabilizer row
\((a\mid b)\) is the exact symplectic product

\[
\sigma=(a\cdot z-b\cdot x)\bmod 3.
\]

Every stabilizer pair must satisfy the same product equal to zero. The decoder
enumerates all Pauli errors through the certified radius
\(t=\lfloor(d-1)/2\rfloor\), stores exact syndrome coset leaders, and rejects a
bounded set if a collision violates the quantum correction conditions. There
are no learned weights, scores, or tie-breaking heuristics.

| Code | Certified set | Non-identity errors exhaustively corrected |
|---|---:|---:|
| Cyclic \([[5,1,3]]_3\) | weight \(\leq 1\) | 40 |
| Shor \([[9,1,3]]_3\) | weight \(\leq 1\) | 72 |
| Ternary Golay \([[11,1,5]]_3\) | weight \(\leq 2\) | 3,608 |

The Golay construction uses the self-orthogonal parity checks of the classical
perfect \([11,6,5]_3\) code in both CSS sectors. The tests independently
enumerate its 729 classical codewords and the 243-word stabilizer subcode to
certify logical distance five, then run every weight-one and weight-two qutrit
Pauli error through correction.

## Harmonic theorem

Encode syndrome symbol \(s\in\mathbb F_3\) at harmonic order \(h\) by

\[
r_h(s)=\omega^{hs}.
\]

- If \(3\nmid h\), multiplication by \(h\) is invertible in \(\mathbb F_3\);
  the harmonic preserves the symbol.
- If \(3\mid h\), \(r_h(s)=1\) for every \(s\); that harmonic cannot identify
  state.
- Orders 1 and 2 therefore provide conjugate redundant reads. Disagreement is
  an exact reason to reject the observation.
- Order 3 is a state-dark reference. Departure from its constant baseline
  exposes distortion but cannot say which qutrit state was present.

The QEC cycle is fail-closed: no correction is issued from an incomplete,
discordant, ambiguous, or dark-invariant-breaking harmonic observation.

For collective-mode analysis, the syndrome phasor field is transformed with a
unitary DFT. Parseval's identity fixes total spectral power, so changes in mode
shape can be studied without inventing or losing energy in the receiver.

## ETQ, Spectral Algebraics, UFT-ID, and E8

ETQ-303 supplies a typed address for observations:

\[
(c,s)\mapsto 3c+s,\qquad 0\leq c<101,\quad s\in\mathbb F_3.
\]

This is a bijective software bridge between a check label and its syndrome
fibre. It is not evidence that ETQ is a physical stabilizer code.

Spectral Algebraics and Collective Modes enter through exact phase characters,
the unitary mode transform, and invariant-breaking detection. UFT-ID 2.0 is
appropriate as a typed cross-field constraint vocabulary: algebraic facts,
receiver choices, hardware assumptions, and empirical results remain separate
types.

E8 is deliberately outside the correction path in this first slice. An E8
feature map may enter only after it either preserves the symplectic syndrome
algebra by construction or beats the exact/reference decoder on a declared,
held-out noise benchmark. Visual appeal alone is not a decoder result.

## What the “800×” result means

The Microsoft–Quantinuum experiment reported error rates from 4.7× to 800×
below selected physical Bell-circuit baselines using fault-tolerant protocols
with the qubit \([[7,1,3]]\) and \([[12,2,4]]\) codes. It did not establish a
universal 800× property of one code. The transferable lesson here is repeated
checked operation, exact small-code validation, and rejection/detection
alongside correction.

## Run

```bash
qec-qutrit
npm test
```

Open `viz/index.html` directly for the dependency-free visual lab.

## Research lineage

User work:

- [ETQ-303 v3.0.1](https://doi.org/10.5281/zenodo.21494678)
- [Spectral Algebraics](https://doi.org/10.5281/zenodo.21308248)
- [Collective Modes v1.2](https://doi.org/10.5281/zenodo.21293821)
- [UFT-ID 2.0: Constraint-First Information Dynamics and Deterministic Recovery](https://doi.org/10.22541/au.176790865.55905239)
- [Trent Slade ORCID](https://orcid.org/0009-0002-4515-9237)

External primary sources:

- [Prakash, *Magic State Distillation with the Ternary Golay Code*](https://arxiv.org/abs/2003.02717)
- [Paetznick et al., *Logical qubits and repeated error correction with better-than-physical rates*](https://arxiv.org/abs/2404.02280)
- [Brock et al., *Quantum error correction of qudits beyond break-even*](https://www.nature.com/articles/s41586-025-08899-y)
- [Google Quantum AI, *Quantum error correction below the surface-code threshold*](https://www.nature.com/articles/s41586-024-08449-y)
- [Spencer et al., *Qudit low-density parity-check codes*](https://arxiv.org/abs/2510.06495)
- [Abiad and Castriota, spectral methods for lifted-product QLDPC failure structures](https://arxiv.org/abs/2607.13666)
