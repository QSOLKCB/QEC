---
title: "QEC v170.3.0 Independent Verification Package"
subtitle: "Exact Ququart Frame-Error Rate and Harmonic Receiver Evidence"
author:
  - "Trent Slade - software author and package curator"
  - "Monica AI - autonomous independent executor"
  - "OpenAI GPT-5.6 Thinking - independent evidence reviewer and formalization"
date: "2026-08-03"
lang: en
papersize: a4
geometry: margin=24mm
fontsize: 10pt
toc: true
toc-depth: 3
numbersections: true
colorlinks: true
linkcolor: black
urlcolor: blue
header-includes:
  - |
    \usepackage{booktabs}
    \usepackage{longtable}
    \usepackage{microtype}
    \usepackage{siunitx}
    \usepackage{fancyhdr}
    \pagestyle{fancy}
    \fancyhf{}
    \fancyhead[L]{QEC v170.3.0 Independent FER Verification}
    \fancyhead[R]{Package v1.0.0}
    \fancyfoot[C]{\thepage}
---

<!-- SPDX-License-Identifier: MPL-2.0 -->

# Abstract

This report formalizes an independent execution and evidence review of the QSOLKCB/QEC
v170.3.0 ququart frame-error-rate (FER) battery. The tested release was pinned to commit
`dada8b7a20a75753db43acc01a6a9e723ebaa6b6` and checked out with full Git history.
The complete repository test suite passed with **19,172 tests passed, 0 failed, 4 skipped,
and 4 warnings**. The focused ququart FER battery passed **18 of 18** tests.

The evidence includes exhaustive classification of all
$16^5 = 1,048,576$ packed Pauli patterns for the packed $[[5,1,3]]_4$ code,
deterministic Monte Carlo sampling over four declared channels, a deterministic harmonic
fault matrix, a 28-cell end-to-end harmonic receiver battery, claim-validation artifacts,
and cryptographic manifests. Exact and Monte Carlo results are statistically compatible
overall: **38 of 40** Wilson 95% intervals contain the exact FER. The two misses are
consistent with nominal interval noncoverage and low-count fluctuation, and no systematic
bias is evident.

The package does **not** claim a hardware threshold, circuit-level fault tolerance,
quantum advantage, pulse fidelity, leakage performance, SPAM performance, or physical
device behavior. Its scope is exact finite-code capacity analysis plus deterministic
classical simulation of a fail-closed harmonic receiver.

# Record identity and roles

| Field | Value |
|---|---|
| Software release | QEC v170.3.0 |
| Pinned and verified commit | `dada8b7a20a75753db43acc01a6a9e723ebaa6b6` |
| Parent commit resolved | `6e7dc3fb33d6b668012f5327ad58b9268f309d3b` |
| Package version | 1.0.0 |
| Execution date | 2026-08-03 |
| Package publication date | 2026-08-03 |
| Creator and curator | Trent Slade |
| Autonomous executor | Monica AI |
| Evidence reviewer and formalization | OpenAI GPT-5.6 Thinking |
| License | Mozilla Public License 2.0 |
| Repository | `https://github.com/QSOLKCB/QEC` |
| Release | `https://github.com/QSOLKCB/QEC/releases/tag/v170.3.0` |

The AI systems are recorded as computational contributors, not as human creators.
The citation creator for the Zenodo record is Trent Slade.

# Provenance and reproducibility

The autonomous executor performed a full-depth clone, checked out the pinned release
commit, verified that `HEAD^` resolved, and recorded a clean working tree. The full test
suite was then run without an artificial per-test timeout:

```text
python -m pytest -q
19172 passed, 4 skipped, 4 warnings in 198.84s
EXIT CODE: 0
```

The focused battery was executed separately:

```text
18 passed, 0 failed, 0 skipped in 2.21s
```

The canonical benchmark command was:

```bash
qec-ququart-bench \
  --output artifacts/ququart-fer-v170.3.0 \
  --trials 10000 \
  --harmonic-trials 4000 \
  --seed 1701001
```

The claim validator returned `passed: true` with threshold claims prohibited and the
hardware-claim contract enforced.

## Integrity model

Let $B_i$ denote the bytes of artifact $i$. Its file identity is

$$
h_i = \operatorname{SHA256}(B_i).
$$

The benchmark also records an internal canonical bundle identity. That internal value is
distinct from the raw SHA-256 digest of `benchmark_manifest.json`; both identities are
preserved and labeled separately.

Hashes support provenance and make post-generation alteration detectable. They do not,
by themselves, prove that no human influenced an earlier stage of the workflow. The
provenance case therefore combines the pinned public commit, full-depth checkout, clean
tree, raw terminal log, deterministic parameters, evidence manifests, and independent
review receipt.

# Mathematical model

## Packed ququart Pauli space

The code under test is declared as the packed $[[5,1,3]]_4$ stabilizer code:

- $n=5$ physical ququarts;
- $k=1$ logical ququart;
- distance $d=3$;
- local Pauli basis size $4^2=16$;
- $15$ nonidentity local Pauli operators.

The exhaustive basis therefore contains

$$
16^5 = 1,048,576
$$

packed Pauli patterns.

For a declared channel with $m$ admissible nonidentity local operators and physical
error rate $p$, a pattern $E$ of weight $w(E)$ has probability

$$
\Pr(E\mid p,m)
  = \left(\frac{p}{m}\right)^{w(E)}
    (1-p)^{n-w(E)}.
$$

Here $m=15$ for the full packed depolarizing channel and $m=3$ for each restricted
single-lane or same-Pauli-correlated channel.

## Exact frame-error rate

Let $F(E)$ be one when the exact decoder classification is either
`detected_uncorrectable` or `accepted_nonstabilizer_residual`, and zero otherwise. The
exact frame-error rate is

$$
\operatorname{FER}(p)
  = \sum_{E\in\mathcal{P}_4^{\otimes 5}}
    F(E)\Pr(E\mid p,m).
$$

The declared small-$p$ behavior for the full packed channel is

$$
\operatorname{FER}_{\mathrm{full}}(p)
  = 10p^2 + O(p^3).
$$

This is a finite-code expansion, not a threshold theorem.

## Monte Carlo estimator and Wilson interval

For $N=10,000$ deterministic trials per cell and observed frame-error count $X$, the
Monte Carlo estimate is

$$
\widehat{q} = \frac{X}{N}.
$$

With $z=1.9599639845$, the Wilson 95% interval is

$$
\frac{
\widehat{q} + \frac{z^2}{2N}
\pm
z\sqrt{\frac{\widehat{q}(1-\widehat{q})}{N}+
\frac{z^2}{4N^2}}
}{1+\frac{z^2}{N}}.
$$

Each cell seed is deterministically derived as

$$
\operatorname{SHA256}(
\text{seed}\Vert\text{battery}\Vert\text{channel}\Vert\text{parameter cell}
).
$$

## Harmonic receiver model

The evidence defines independent complex Gaussian perturbations applied to harmonic
samples, a declared tolerance of $0.35$, and a fail-closed receiver policy. Its roles are:

- $H_1$ and $H_3$: redundant full-state identification;
- $H_2$: parity validation;
- $H_4$: state-dark distortion reference.

The report does not infer an unrecorded normalization convention for the complex
Gaussian distribution. It evaluates only the emitted deterministic artifacts and their
declared parameters.

# Execution results

## Repository test suite

| Metric | Result |
|---|---:|
| Passed | 19,172 |
| Failed | 0 |
| Skipped | 4 |
| Warnings | 4 |
| Runtime | 198.84 s |
| Exit code | 0 |

The four skips required optional external backends (`stim`, `pymatching`, and
`qiskit-aer`) that are not installed by the default development extras. The warnings were
deprecation notices and did not affect the test result.

## Focused FER battery

All 18 ququart-specific unit and integration tests passed. The battery verified, among
other properties:

- exhaustive classification of all packed Pauli patterns;
- exact probability normalization;
- inclusion of all 15 local nonidentity operators in the full channel;
- deterministic Monte Carlo cells;
- fail-closed harmonic faults;
- code parameters and distance;
- correction of every weight-one packed Pauli error;
- rejection of unknown syndromes;
- parity and cross-harmonic disagreement rejection.

# Exact finite-code evidence

## Weight enumerator

| Weight | Patterns | Corrected | Detected uncorrectable | Logical failure | Frame failures |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 1 | 0 | 0 | 0 |
| 1 | 75 | 75 | 0 | 0 | 0 |
| 2 | 2,250 | 0 | 1,800 | 450 | 2,250 |
| 3 | 33,750 | 300 | 23,400 | 10,050 | 33,450 |
| 4 | 253,125 | 5,175 | 178,200 | 69,750 | 247,950 |
| 5 | 759,375 | 13,905 | 533,880 | 211,590 | 745,470 |

All 75 weight-one patterns are corrected. All 2,250 weight-two patterns are frame
failures, partitioned into 1,800 detected-uncorrectable outcomes and 450 logical
failures. This is consistent with distance three.

> Figures and the rendered mathematical PDF are included in the Zenodo upload package identified by `ZENODO_UPLOAD_SHA256.txt`.

## Full packed depolarizing channel

| $p$ | Exact FER | Detected uncorrectable | Logical failure |
|---:|---:|---:|---:|
| 1e-05 | 9.9998e-10 | 7.99983e-10 | 1.99997e-10 |
| 3e-05 | 8.99946e-09 | 7.19954e-09 | 1.79992e-09 |
| 0.0001 | 9.99799e-08 | 7.99829e-08 | 1.9997e-08 |
| 0.0003 | 8.99458e-07 | 7.19539e-07 | 1.79918e-07 |
| 0.001 | 9.97993e-06 | 7.98295e-06 | 1.99698e-06 |
| 0.003 | 8.94588e-05 | 7.15403e-05 | 1.79185e-05 |
| 0.01 | 0.000980061 | 0.000783069 | 0.000196992 |
| 0.03 | 0.00846971 | 0.00675016 | 0.00171955 |
| 0.1 | 0.0813786 | 0.0642598 | 0.0171188 |
| 0.2 | 0.262128 | 0.204069 | 0.0580589 |

The exact curve follows the declared $10p^2$ leading order at small $p$. Higher-order
terms become visible as $p$ increases. This finite-code crossover must not be described
as a hardware or fault-tolerance threshold.

# Exact versus Monte Carlo evidence

Across 40 cells:

| Cell class | Count | Exact FER inside Wilson 95% interval |
|---|---:|---:|
| Zero observed frame errors | 20 | 20 |
| One or more observed frame errors | 20 | 18 |
| Total | 40 | 38 |

The correct partition is therefore **20 zero-event cells**, **20 nonzero-event cells**,
and **38 of 40** intervals containing the exact FER.

The two noncovering cells are:

| Channel | $p$ | Errors | MC FER | Wilson 95% interval | Exact FER |
|---|---:|---:|---:|---:|---:|
| `full_packed_depolarizing` | 0.2 | 2518 | 0.2518 | [0.243389, 0.260401] | 0.262128 |
| `same_pauli_correlated` | 0.001 | 1 | 0.0001 | [1.76527e-05, 0.000566269] | 9.9778e-06 |

For the full packed channel at $p=0.2$, the exact rate lies slightly above the interval.
A 95% interval has a nominal 5% noncoverage probability, so isolated misses are expected
across a family of 40 intervals. For the same-Pauli-correlated channel at $p=0.001$, one
event was observed where the exact expectation is approximately $0.1$ event per 10,000
trials. This is a low-count fluctuation. The two misses do not exhibit a common direction
or a systematic channel-wide bias.

# Harmonic receiver evidence

## Deterministic fault matrix

The battery contains seven fault scenarios:

- two expected-accept cases: `clean` and `bounded_complex_noise`;
- five adversarial cases: `missing_h3`, `h1_h3_disagreement`, `h2_parity_flip`,
  `h4_dark_distortion`, and `ambiguous_h1`.

The evaluation count is

$$
5\times 75 + 2\times 75 = 375 + 150 = 525.
$$

| Quantity | Result |
|---|---:|
| Adversarial evaluations | 375 |
| Expected-accept evaluations | 150 |
| Total evaluations | 525 |
| Adversarial false accepts | 0 |
| Receiver false-trust events | 0 |

Every adversarial case was rejected and every expected-accept case was accepted and
corrected in the declared deterministic battery.

## End-to-end receiver battery

The end-to-end battery combines four physical error rates

$$
p\in\{0.001,0.003,0.01,0.03\}
$$

with seven harmonic noise levels

$$
\sigma\in\{0,0.02,0.05,0.10,0.20,0.35,0.50\},
$$

producing $4\times7=28$ cells with 4,000 trials per cell.

At low noise, acceptance remains high and observed frame errors are dominated by the
underlying finite-code behavior. At $\sigma=0.10$, receiver rejection begins to appear.
At $\sigma=0.20$, the receiver rejects most trials, and at or above the declared
tolerance it rejects essentially all trials. No `receiver_false_trust` event occurs in
the 28-cell evidence.

# Symmetry, claim validation, and replication

The lane-symmetry certificate states that `lane0_only` and `lane1_only` have identical
weight enumerators across all six weight classes. Its internal certificate identity is

```text
a9073805c2c3d85c5b37824b3d2462b400d8e71be723739fcc96cf92539bfca7
```

The claim-validation artifact reports `passed: true`. It confirms:

- full SHA-256 is required for artifact-match claims;
- numeric claims match the evidence;
- false-accept claims match the fault matrix;
- the lane-symmetry claim matches the certificate;
- hardware declarations are enforced;
- `threshold_claim_permitted` is false.

The included qBraid receipt records a prior reduced-trial cross-environment replication.
It distinguishes deterministic artifacts from parameter-bound artifacts and explicitly
does not promote eight-character prefix consistency to a full-hash match.

# Scope and limitations

## Supported claims

This package supports the following bounded statements:

1. The pinned QEC v170.3.0 checkout passed the complete available test suite in the
   declared environment.
2. The exact oracle exhaustively classified the declared packed Pauli basis.
3. Every declared weight-one packed Pauli error is corrected by the exact finite-code
   model.
4. The deterministic fault matrix is fail-closed for the tested adversarial harmonic
   cases.
5. The Monte Carlo artifacts are reproducible under their deterministic seed policy.
6. The exact and sampled results show no evidence of systematic disagreement in the
   declared battery.
7. The provided evidence files match their recorded SHA-256 digests.

## Unsupported claims

This package does not establish:

- a physical fault-tolerance threshold;
- a hardware break-even point;
- quantum advantage;
- circuit-level performance;
- leakage resistance;
- SPAM performance;
- timing or pulse-fidelity performance;
- universal decoder superiority;
- independent human laboratory replication.

The harmonic receiver is a deterministic classical simulation and evidence-separation
mechanism, not physical ququart readout hardware.

## Statistical limits

Ten thousand Monte Carlo trials are insufficient to resolve FER values near
$10^-6$ or below. In those cells the exact oracle is authoritative. Wilson intervals
are individual 95% intervals, not a simultaneous 95% confidence band over all 40 cells.

# Evidence and package identities

The evidence directory contains the 15 files emitted by the canonical benchmark. Their
raw SHA-256 values are recorded in `evidence/SHA256SUMS.txt` and in the package-wide
`SHA256SUMS.txt`.

Important identity distinction:

| Identity | Meaning |
|---|---|
| `1daec95984d2c914ede2b207ce8f6e2402ddbd5889c8095969838ce8700d66af` | Internal canonical bundle identity recorded inside `benchmark_manifest.json` |
| `3dbd0bbb068c379ffbb402fe1a5863a5c5cc2797a2d078c55706e88839dd4a64` | Raw file SHA-256 of `benchmark_manifest.json` |
| `7211dba1b71ddca9bcbd47e9599344270882f4851dbd4d507603924573005822` | Internal claim-validation identity |
| `a9073805c2c3d85c5b37824b3d2462b400d8e71be723739fcc96cf92539bfca7` | Internal lane-symmetry certificate identity |

The package-wide ZIP hash is published outside the ZIP in
`ZENODO_UPLOAD_SHA256.txt`, avoiding self-reference.

# Reproduction

```bash
git clone https://github.com/QSOLKCB/QEC.git
cd QEC
git checkout dada8b7a20a75753db43acc01a6a9e723ebaa6b6
python -m pip install -e ".[dev]"
python -m pytest -q

qec-ququart-bench \
  --output artifacts/ququart-fer-v170.3.0 \
  --trials 10000 \
  --harmonic-trials 4000 \
  --seed 1701001

qec-ququart-validate-report \
  --claims artifacts/ququart-fer-v170.3.0/report_claims.json \
  --evidence artifacts/ququart-fer-v170.3.0
```

The report PDF is generated from the canonical Markdown source included in the Zenodo
package.

# Conclusion

The corrected independent verification package supports a strong but bounded result:
the QEC v170.3.0 exact ququart FER and harmonic receiver evidence is internally coherent,
reproducible under the declared software environment, hash-verifiable, and guarded
against threshold and hardware overclaiming. The complete test suite passes in a
full-depth checkout, the exact oracle covers the full declared basis, the deterministic
harmonic adversarial battery fails closed, and the Monte Carlo evidence is statistically
compatible with the exact oracle overall.

The formal result is a software-evidence verification record. It is not a physical
quantum-hardware result.

# Acknowledgment of corrections

The original executor report v2 corrected the shallow-clone and timeout issues from the
first run. This formal package additionally corrects the interval-partition arithmetic:
the evidence contains 20 zero-event cells and 20 nonzero-event cells, with 18 of the
20 nonzero intervals and all 20 zero-event intervals covering the exact FER.
