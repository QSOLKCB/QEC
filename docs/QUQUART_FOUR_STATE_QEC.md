# Exact four-state QEC: native ququart gates plus packed-qubit correction

This module adds an **opt-in four-state path** without deleting or weakening the
existing qutrit decoder. It deliberately separates two meanings of “ququart”
that are often blurred together:

1. **Native ququart:** one four-level system with basis
   \(\{|0\rangle,|1\rangle,|2\rangle,|3\rangle\}\) and generalized gates
   \(X_4\), \(Z_4\), and \(H_4\).
2. **Encoded pair of qubits:** the same four basis states interpreted through
   \(|q_0q_1\rangle\leftrightarrow|2q_0+q_1\rangle\).

The gate layer exposes both interpretations. The correction layer uses the
second interpretation because it gives an exact additive stabilizer
construction over \(GF(2)\times GF(2)\), avoiding the mathematically invalid
shortcut of treating ordinary integers modulo four as a field.

## Basis and gates

The packed encoding is

\[
|00\rangle\leftrightarrow|0\rangle,\quad
|01\rangle\leftrightarrow|1\rangle,\quad
|10\rangle\leftrightarrow|2\rangle,\quad
|11\rangle\leftrightarrow|3\rangle.
\]

The native ququart Weyl and Fourier gates are

\[
X_4|j\rangle=|j+1\bmod 4\rangle,\qquad
Z_4|j\rangle=i^j|j\rangle,
\]

\[
H_4=\frac12
\begin{bmatrix}
1&1&1&1\\
1&i&-1&-i\\
1&-1&1&-1\\
1&-i&-1&i
\end{bmatrix}.
\]

`qec.decoder.ququart.gates` also provides the packed operations
\(X\otimes I\), \(I\otimes X\), \(X\otimes X\), and the internal SWAP that
exchanges \(|1\rangle\leftrightarrow|2\rangle\). Native `X4` is intentionally
not aliased to either encoded-qubit X operation.

## Exact packed \([[5,1,3]]_4\) construction

Each physical ququart contains two binary Pauli lanes. Each lane independently
uses the perfect five-qubit \([[5,1,3]]_2\) code with generators

```text
X Z Z X I
I X Z Z X
X I X Z Z
Z X I X Z
```

Across five physical ququarts this gives eight independent stabilizer
generators in a ten-qubit Hilbert space. The codespace dimension is therefore
\(2^{10-8}=4\): one logical ququart. The exact decoder enumerates the identity
and all 75 non-identity one-ququart Pauli-basis errors
\((4^2-1)\times5\), builds a collision-checked syndrome table, and accepts a
correction only when the residual belongs to the stabilizer.

Because the 16 two-qubit Pauli products form an operator basis on one
four-dimensional site, correcting all 15 non-identity basis elements at every
site corrects an arbitrary error supported on one packed ququart. The included
exact search independently finds physical-ququart distance three.

## Four-state harmonic syndrome receiver

The binary syndrome has four checks per lane. Corresponding lane bits are
packed into four-state symbols with

\[
(s_0,s_1)\mapsto 2s_0+s_1.
\]

A syndrome symbol \(s\in\{0,1,2,3\}\) is observed at harmonic order \(h\) as

\[
r_h(s)=i^{hs}.
\]

The fail-closed receiver assigns exact roles:

| Harmonic residue mod 4 | Role |
|---|---|
| 1 and 3 | full four-state reads; conjugate redundant channels |
| 2 | parity-only check; distinguishes even from odd symbols |
| 0 | state-dark reference; detects distortion but cannot identify state |

The default receiver requires H1, H3, H2, and H4. Missing roles, H1/H3
disagreement, parity disagreement, ambiguity, excessive residual, or a broken
H4 invariant cause rejection before correction.

## Run

```bash
qec-ququart
python -m qec.decoder.ququart
```

The command emits canonical JSON with a SHA-256-bound certificate covering all
75 one-ququart Pauli-basis errors.

## Hardware and claim boundary

This is an exact finite software model and decoder oracle. It does not claim a
hardware threshold, transmon pulse fidelity, photonic coincidence rate, or
break-even experiment. Hardware backends must separately declare leakage,
state-preparation-and-measurement error, coherence by level, gate duration, and
whether readout is joint or decoded.

The separation is important experimentally: work on superconducting transmons
explicitly distinguishes a native ququart Clifford model from an encoded
pair-of-qubits Clifford model, and reports four-state readout as a significant
SPAM challenge. Mixed-radix compilation work likewise treats internal,
qubit–ququart, and ququart–ququart operations as different cost classes rather
than assuming all four-state gates are equally cheap.

## Primary references

- L. M. Seifert et al., *Exploring Ququart Computation on a Transmon using
  Optimal Control*, arXiv:2304.11159.
- A. Litteken et al., *Qompress: Efficient Compilation for Ququarts Exploiting
  Partial and Mixed Radix Operations for Communication Reduction*, ASPLOS 2023,
  DOI: 10.1145/3575693.3575726.
- Y. Chi et al., *Multi-qubit entanglement and quantum processing within
  ququart*, Communications Physics, 2026, DOI: 10.1038/s42005-026-02734-0.
- M.-X. Luo et al., *Photonic ququart logic assisted by the cavity-QED system*,
  Scientific Reports 5, 13255 (2015), DOI: 10.1038/srep13255.
