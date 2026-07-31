# QEC v170.1.0 — Exact Ququart FER Oracle and Harmonic Fault Battery

## Purpose

v170.1.0 adds a reproducible performance laboratory around the exact packed
ququart decoder introduced in v170.0.0. It does **not** alter the
`[[5,1,3]]_4` construction or its bounded coset decoder. It adds exact
enumeration, correctly parameterized physical-noise channels, receiver fault
injection, deterministic Monte Carlo cross-checks, canonical evidence, and a
dependency-free GitHub Pages report.

The implementation replaces a useful qBraid prototype whose reporting
architecture was sound but whose physical-noise sampler applied an entire
random frame with probability `p` and exercised only lane 0. In v170.1.0,
`p` means an independent physical error probability **per ququart**, and the
full channel samples all 15 nonidentity two-lane Pauli products.

## Exact finite oracle

Five packed ququarts have a local Pauli basis of size 16, so the complete
finite pattern space contains

\[
16^5 = 1,048,576.
\]

The oracle classifies every pattern by physical-ququart weight and decoder
outcome:

| Weight | Patterns | Corrected | Detected uncorrectable | Logical failure |
|---:|---:|---:|---:|---:|
| 0 | 1 | 1 | 0 | 0 |
| 1 | 75 | 75 | 0 | 0 |
| 2 | 2,250 | 0 | 1,800 | 450 |
| 3 | 33,750 | 300 | 23,400 | 10,050 |
| 4 | 253,125 | 5,175 | 178,200 | 69,750 |
| 5 | 759,375 | 13,905 | 533,880 | 211,590 |

For the iid packed-depolarizing channel, every nonidentity local operator has
probability `p/15`. The exact frame-error rate is therefore

\[
\mathrm{FER}(p)
=
\sum_{w=0}^{5}
N_{\mathrm{fail}}(w)
\left(\frac{p}{15}\right)^w
(1-p)^{5-w}.
\]

The leading small-`p` behavior is

\[
\mathrm{FER}(p)=10p^2+O(p^3),
\]

consistent with a distance-three code correcting every one-ququart Pauli-basis
error.

The oracle uses a complete 1024-state table for each binary Pauli lane and
combines the lanes exactly. This avoids random sampling while remaining fast
enough for CI and Pages generation.

## Corrected Monte Carlo channels

The Monte Carlo battery cross-checks the oracle and explores declared
nonuniform channels. Each physical site is sampled independently.

- `full_packed_depolarizing`: all 15 nonidentity `(lane0, lane1)` products.
- `lane0_only`: `XI`, `YI`, and `ZI`.
- `lane1_only`: `IX`, `IY`, and `IZ`.
- `same_pauli_correlated`: `XX`, `YY`, and `ZZ`.

Every cell has a seed derived from SHA-256 over the global seed and the full
cell identity. Adding or reordering other cells therefore does not silently
change an existing result. Frame-error estimates include 95% Wilson intervals.

## Harmonic fault battery

The v170.0.0 receiver is tested as an end-to-end fail-closed measurement path:

- H1 and H3 provide redundant full four-state identification;
- H2 validates parity;
- H4 is a state-dark distortion reference.

Every one of the 75 certified one-ququart errors is exercised under:

- clean readout;
- bounded complex perturbation inside tolerance;
- missing H3;
- H1/H3 disagreement;
- H2 parity corruption;
- H4 dark-reference distortion;
- an exactly ambiguous H1 sample.

Clean and bounded cases must be accepted and corrected. Adversarial cases must
be rejected. Any accepted-but-incorrect result is counted as a false accept.

A second battery combines the full iid physical channel with independent
complex Gaussian noise on every harmonic sample. It reports receiver
rejections, incorrect trusted syndromes, false accepts, total frame errors,
and Wilson intervals.

## Deterministic artifacts

Run:

```bash
qec-ququart-bench
```

or:

```bash
python -m qec.benchmark.ququart_battery
```

Default output:

```text
benchmarks/ququart_fer_v170_1_0/
├── benchmark_manifest.json
├── methodology.json
├── report.js
├── exact_weight_enumerator.csv
├── exact_fer_curve.csv
├── monte_carlo_fer.csv
├── harmonic_fault_matrix.csv
└── harmonic_end_to_end.csv
```

All CSV files use stable column order and LF line endings. `methodology.json`
and `benchmark_manifest.json` use QEC canonical JSON and SHA-256 identities.
Floating-point measurements are represented as decimal strings at the
canonical boundary.

Example quick run:

```bash
qec-ququart-bench \
  --output /tmp/ququart-fer \
  --trials 1000 \
  --harmonic-trials 500 \
  --seed 1701001
```

## GitHub Pages

The `ququart-fer-pages.yml` workflow:

1. runs the focused v170.1.0 test modules;
2. installs the QEC package;
3. generates the exact and deterministic benchmark artifacts;
4. publishes the complete `viz/` directory through GitHub Pages.

The browser report is available under:

```text
https://qsolkcb.github.io/QEC/ququart-fer/
```

The site is static HTML, CSS, and JavaScript. It has no Node build, CDN,
framework, telemetry, or external runtime dependency.

## Claim boundary

This battery is an exact finite code-capacity Pauli analysis plus a classical
harmonic-readout simulation. It does not establish a hardware threshold,
fault-tolerant circuit threshold, break-even experiment, transmon pulse
fidelity, photonic coincidence rate, leakage rate, SPAM rate, or universal
quantum advantage.

The phrase “exact FER” applies only to the declared finite packed-Pauli channel
and decoder model. Hardware claims require separately declared device,
circuit, leakage, timing, and readout assumptions.
