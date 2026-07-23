# Qutrit Decoder Benchmark Battery

## Purpose

This battery tests the opt-in exact qutrit decoder without modifying the frozen
historical decoder. It answers three different questions with three different
artifacts:

1. What is the exact block failure probability of the declared bounded decoder
   under a common finite code-capacity model?
2. What logical-failure upper bound follows only from a code's guaranteed
   correction radius?
3. How does the decoder classify a deterministic corpus at every physical
   error weight, including silent miscorrections?

Those questions are not merged into one leaderboard.

## Common simulation model

Each physical site is independently:

- identity with probability \(1-p\);
- one of \(q^2-1\) non-identity generalized Paulis with probability
  \(p/(q^2-1)\) each.

Syndrome extraction is perfect. This is an iid depolarizing code-capacity
model, not a circuit-level or hardware model.

The decoder is the same non-heuristic rule for every exact curve: build the
syndrome table from every error of weight at most
\(t=\lfloor(d-1)/2\rfloor\), correct with that exact coset leader, reject an
unknown syndrome, and count an accepted result as successful only when the
residual is a stabilizer.

## Exact logical-error calculation

For tractable codes, the benchmark enumerates the stabilizer coset associated
with every decoder-table entry. If \(A_w\) is the exact number of successful
Pauli errors of weight \(w\), then

\[
P_{\mathrm{fail}}(p)
=
1-\sum_{w=0}^{n}
A_w
\left(\frac{p}{q^2-1}\right)^w
(1-p)^{n-w}.
\]

The sum is evaluated with 80-digit decimal arithmetic. No Monte Carlo sampling,
fitted curve, decoder score, or confidence interval is used. The exact track
contains:

| Code | Alphabet | Parameters | Result |
|---|---:|---:|---|
| Qutrit cyclic | 3 | \([[5,1,3]]_3\) | Exact curve |
| Qutrit Shor | 3 | \([[9,1,3]]_3\) | Exact curve |
| Five-qubit | 2 | \([[5,1,3]]\) | Exact curve |
| Steane | 2 | \([[7,1,3]]\) | Exact curve |
| Rotated surface | 2 | \([[9,1,3]]\) | Exact curve |
| Quantum Reed–Muller | 2 | \([[15,1,3]]\) | Exact curve |
| Ternary Golay | 3 | \([[11,1,5]]_3\) | Rigorous bound + stress corpus |

The Golay success-coset expansion would require 213,107,841 pair operations.
The battery refuses to replace that calculation with a convenient estimate.

## Guaranteed-radius comparison

Every declared \([[n,k,d]]_q\) code corrects every error of weight at most
\(t=\lfloor(d-1)/2\rfloor\). Therefore, under independent per-site error
probability \(p\),

\[
P_{\mathrm{logical\ failure}}
\le
P(W>t)
=
1-\sum_{w=0}^{t}
\binom{n}{w}p^w(1-p)^{n-w}.
\]

This bound is computed for the finite qutrit codes and the comparison catalog:
five-qubit, Steane, Reed–Muller, rotated surface distances 3/5/7,
Quantinuum's \([[12,2,4]]\) C4/C6-derived code, the
\([[16,4,4]]\) tesseract subsystem code, and the
\([[144,12,12]]\) bivariate-bicycle code.

The bound is a weight-only code-capacity statement. It is not an achieved
logical error rate, a threshold, or evidence that one physical platform beats
another.

## Deterministic false-positive stress

For every implemented decoder and every weight \(0,\ldots,n\), the stress
corpus is either exhaustive or an evenly spaced canonical ordinal corpus.
Each row records:

- corrected;
- rejected;
- miscorrected: accepted but with a non-stabilizer residual;
- corpus SHA-256;
- exact versus deterministic-corpus coverage.

The corpus makes no statistical claim. Its job is falsification and regression:
all errors in the certified radius must be corrected, while failures outside
the radius remain visible rather than being reported as successes.

`harmonic_fault_injection.csv` separately exhausts the 40, 72, and 3,608
certified non-identity error sets. Clean H1/H2/H3 observations must correct
every error; an H2 disagreement or a broken state-dark H3 invariant must reject
every error before correction.

## Immutable v3 lineage

`qec_data_prepared.csv` is treated as read-only historical material.

| Field | Value |
|---|---|
| Verified tags | `v3.0.0` through `v3.9.1` |
| Git blob | `b4a3d4a9f9bb8de9b2ba391406f269b27c6715dc` |
| SHA-256 | `80f1f74ad02c2ac7fdaf5e6a6df1611f55df3f4294cc11edfc889f5d7fe41b0a` |

The loader fails closed if one byte changes. The overlay and numeric ratios
retain the repo's original boundary: those values were illustrative /
research-aligned and were not device-calibrated. Ratios are therefore numeric
lineage observations, not performance claims.

## External evidence remains separate

`published_evidence.csv` records source-bound results without inserting them
into the simulated curves:

- Google's distance-7 surface-code memory:
  [Nature (2025)](https://www.nature.com/articles/s41586-024-08449-y);
- Quantinuum/Microsoft logical error-reduction ranges:
  [arXiv:2404.02280](https://arxiv.org/abs/2404.02280);
- experimental GKP qutrit gain:
  [Nature (2025)](https://www.nature.com/articles/s41586-025-08899-y);
- tesseract subsystem color code:
  [arXiv:2409.04628](https://arxiv.org/abs/2409.04628);
- fusion-based photonic thresholds:
  [Nature Communications (2023)](https://www.nature.com/articles/s41467-023-36493-1);
- bivariate-bicycle qLDPC reference:
  [arXiv:2308.07915](https://arxiv.org/abs/2308.07915);
- qudit qLDPC research target:
  [arXiv:2510.06495](https://arxiv.org/abs/2510.06495).
- spectral cycle and absorbing-set diagnostics for lifted-product qLDPC codes:
  [arXiv:2607.13666](https://arxiv.org/abs/2607.13666).

The `research_watch.csv` file also records why current neural, search, or other
approximate decoders are not silently substituted for the exact bounded rule.

## Run and outputs

```bash
qec-qutrit-bench
```

The default output is `benchmarks/qutrit_decoder_v1/`:

- `decoded_logical_error_long.csv`
- `decoded_logical_error_wide.csv`
- `guaranteed_radius_tail_long.csv`
- `guaranteed_radius_tail_wide.csv`
- `deterministic_stress_corpus.csv`
- `harmonic_fault_injection.csv`
- `historical_v3_baseline.csv`
- `v3_overlay.csv`
- `v3_numeric_deltas.csv`
- `published_evidence.csv`
- `research_watch.csv`
- `methodology.json`
- `benchmark_manifest.json`

Every JSON identity uses sorted-key compact canonical JSON and SHA-256. Every
CSV has stable ordering and Unix line endings. No wall-clock value enters the
artifacts.

## Claim boundary

The battery supports finite algebraic correctness, deterministic regression,
code-capacity simulation, and source-bound comparison. It does not establish a
circuit-level threshold, account for leakage/crosstalk/measurement faults,
model harmonic transducers as quantum hardware, or prove universal QEC
advantage.
