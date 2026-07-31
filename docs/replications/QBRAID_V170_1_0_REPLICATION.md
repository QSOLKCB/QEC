# Independent qBraid Replication of QEC v170.1.0

## Status

**Parameter-bound independent software replication.**

This document preserves the useful evidence from the supplied qBraid report
while correcting terminology and arithmetic that would otherwise overstate the
result.

## Environment and parameters

| Field | Value |
|---|---|
| Platform | qBraid Lab |
| Operating system | Ubuntu 24.04 |
| Python | 3.12 |
| Date | 2026-07-31 |
| Seed | 1701001 |
| Monte Carlo trials per cell | 1,000 |
| Harmonic trials per cell | 500 |
| Source document SHA-256 | `22f9d219d88f6dfe923d0d14d8a1ef387b24c60649541fd912e408a846e20ccf` |
| Reported run manifest | `16a40ff502aafff7896c1102a1f92af8c744a5f0cfdcd9c361eb3a1e0a4b9426` |

The canonical v170.1.0 release workflow used 10,000 Monte Carlo trials and
4,000 harmonic trials per cell. Sampled artifact hashes are therefore expected
to differ.

## Cross-environment evidence

The qBraid report displayed eight-character hash prefixes. The following
exact deterministic outputs are prefix-consistent with the canonical release:

| Artifact | qBraid prefix | Canonical release prefix | Status |
|---|---:|---:|---|
| `exact_weight_enumerator.csv` | `f273cd3e` | `f273cd3e` | prefix consistent |
| `exact_fer_curve.csv` | `c93dc243` | `c93dc243` | prefix consistent |
| `harmonic_fault_matrix.csv` | `c6f744ba` | `c6f744ba` | prefix consistent |

This is meaningful independent evidence, but an eight-character prefix is not
promoted to a complete SHA-256 equality claim. Full cross-environment equality
requires the complete qBraid artifact hashes or the original files.

## Corrected evidence interpretation

- The exact finite oracle enumerated all `16^5 = 1,048,576` packed-Pauli
  patterns.
- The full packed channel has low-error behaviour
  `FER(p) = 10p^2 + O(p^3)`.
- This is a **low-error quadratic regime**, not a measured hardware threshold.
- The deterministic harmonic matrix contains 525 total evaluations:
  - 150 expected-accept evaluations;
  - 375 adversarial expected-reject evaluations.
- The deterministic adversarial matrix contains zero false accepts.
- The end-to-end sweep has `4 × 7 = 28` parameter cells, not 112.
- Accepted logical residuals caused by uncorrectable physical patterns must be
  distinguished from receiver false-trust events.

## Exact-channel implication

The qBraid Monte Carlo tables showed small lane-0/lane-1 differences because
only 1,000 samples were used per cell. v170.1.1 now enumerates both restricted
lane channels exactly and certifies their equality under lane exchange.

## Receipt

The machine-readable receipt is stored at:

```text
docs/replications/qbraid_v170_1_0_receipt.json
```

Its deterministic cross-environment status is `prefix_consistent`, and its
sampled-artifact status is `parameter_variant_expected`.

## Scientific boundary

This replication concerns a finite software model under declared code-capacity
Pauli channels and a classical harmonic receiver simulation. It does not make
hardware, circuit-threshold, break-even, leakage, SPAM, or universal quantum
advantage claims.
