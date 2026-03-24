# QEC Industry Comparison — v3.1.4 Metric Expansion

**Project Version:** v3.1.4
**Git Commit:** `eae3dbd1d988ccbb8ad09d6da34090473a3334b9`
**Execution Environment:** Claude Code Compute
**Report Artifact:** `bench/industry_comparison_v3.1.4_metrics.md`

**Layer 1 decoder logic unchanged.**

---

## 1. Metric Definitions

### 1.1 Logical Fidelity

Logical Fidelity measures the probability that the decoder successfully corrects the error:

```
FER = logical_failures / trials
Fidelity = 1.0 - FER
```

Fidelity is the logical success probability per sweep point. A Fidelity of 1.0 means all trials were decoded correctly; 0.0 means all trials failed.

### 1.2 Syndrome Consistency Rate

Syndrome Consistency Rate measures how often the decoder's correction output satisfies the observed syndrome constraints, independent of logical correctness:

```
For each trial:
  H = parity-check matrix
  e = actual error vector
  s = syndrome(H, e)
  c = decoder correction output

  Trial is syndrome-consistent if: syndrome(H, c) == s

Syndrome Consistency Rate = successful_syndrome_matches / trials
```

A correction can be syndrome-consistent without being logically correct (if it differs from the true error by a stabilizer element). Conversely, a correction that fails syndrome consistency has not even satisfied the parity constraints.

---

## 2. Benchmark Configurations

### Config A — Operating-Region Baseline

```json
{
  "schema_version": "3.0.1",
  "seed": 20260226,
  "distances": [7, 9, 11],
  "p_values": [0.01, 0.02, 0.05],
  "trials": 200,
  "max_iters": 50,
  "decoders": [
    {"adapter": "bp", "params": {"mode": "min_sum", "schedule": "flooding"}},
    {"adapter": "bp", "params": {"mode": "min_sum", "schedule": "residual"}},
    {"adapter": "bp", "params": {"mode": "min_sum", "schedule": "adaptive"}}
  ],
  "runtime_mode": "off",
  "deterministic_metadata": true,
  "channel_model": "oracle"
}
```

### Config B — Transition-Band Probe

```json
{
  "schema_version": "3.0.1",
  "seed": 20260226,
  "distances": [7, 9, 11],
  "p_values": [0.49, 0.499, 0.50],
  "trials": 200,
  "max_iters": 50,
  "decoders": [
    {"adapter": "bp", "params": {"mode": "min_sum", "schedule": "flooding"}},
    {"adapter": "bp", "params": {"mode": "min_sum", "schedule": "residual"}},
    {"adapter": "bp", "params": {"mode": "min_sum", "schedule": "adaptive"}}
  ],
  "runtime_mode": "off",
  "deterministic_metadata": true,
  "channel_model": "oracle"
}
```

---

## 3. Environment Snapshot

| Key | Value |
|-----|-------|
| Python version | 3.11.14 (main, Oct 10 2025, 08:54:04) [GCC 13.3.0] |
| NumPy version | 2.4.2 |
| Platform | Linux-4.4.0-x86_64-with-glibc2.39 |
| Git commit | `eae3dbd1d988ccbb8ad09d6da34090473a3334b9` |

---

## 4. Config A — FER Tables (Operating Region)

All 27 sweep points (3 decoders x 3 distances x 3 error rates):

**FER = 0.000000** for all combinations at p in {0.01, 0.02, 0.05}.

All schedules converge in exactly 1 iteration. The oracle channel model provides perfect LLR sign information for any p < 0.50.

---

## 5. Config A — Fidelity Tables

| Schedule | d=7 (all p) | d=9 (all p) | d=11 (all p) |
|----------|-------------|-------------|--------------|
| flooding | 1.000000 | 1.000000 | 1.000000 |
| residual | 1.000000 | 1.000000 | 1.000000 |
| adaptive | 1.000000 | 1.000000 | 1.000000 |

Fidelity = 1.0 across all operating-region sweep points.

---

## 6. Config A — Syndrome Consistency Tables

| Schedule | d=7 (all p) | d=9 (all p) | d=11 (all p) |
|----------|-------------|-------------|--------------|
| flooding | 1.000000 | 1.000000 | 1.000000 |
| residual | 1.000000 | 1.000000 | 1.000000 |
| adaptive | 1.000000 | 1.000000 | 1.000000 |

Syndrome Consistency Rate = 1.0 across all operating-region sweep points.

---

## 7. Config B — FER and Fidelity (Transition Band)

| Schedule | d | p=0.49 FER / Fidelity | p=0.499 FER / Fidelity | p=0.50 FER / Fidelity |
|----------|---|----------------------|------------------------|----------------------|
| flooding | 7 | 0.000 / 1.000 | 0.000 / 1.000 | 1.000 / 0.000 |
| flooding | 9 | 0.000 / 1.000 | 0.000 / 1.000 | 1.000 / 0.000 |
| flooding | 11 | 0.000 / 1.000 | 0.000 / 1.000 | 1.000 / 0.000 |
| residual | 7 | 0.000 / 1.000 | 0.000 / 1.000 | 1.000 / 0.000 |
| residual | 9 | 0.000 / 1.000 | 0.000 / 1.000 | 1.000 / 0.000 |
| residual | 11 | 0.000 / 1.000 | 0.000 / 1.000 | 1.000 / 0.000 |
| adaptive | 7 | 0.000 / 1.000 | 0.000 / 1.000 | 1.000 / 0.000 |
| adaptive | 9 | 0.000 / 1.000 | 0.000 / 1.000 | 1.000 / 0.000 |
| adaptive | 11 | 0.000 / 1.000 | 0.000 / 1.000 | 1.000 / 0.000 |

The FER transition is a degenerate step function at p=0.50. For p < 0.50, Fidelity = 1.0. At p = 0.50, Fidelity = 0.0.

---

## 8. Config B — Syndrome Consistency (Transition Band)

| Schedule | d | p=0.49 SCR | p=0.499 SCR | p=0.50 SCR |
|----------|---|-----------|-------------|-----------|
| flooding | 7 | 1.000000 | 1.000000 | 0.000000 |
| flooding | 9 | 1.000000 | 1.000000 | 0.000000 |
| flooding | 11 | 1.000000 | 1.000000 | 0.000000 |
| residual | 7 | 1.000000 | 1.000000 | 0.000000 |
| residual | 9 | 1.000000 | 1.000000 | 0.000000 |
| residual | 11 | 1.000000 | 1.000000 | 0.000000 |
| adaptive | 7 | 1.000000 | 1.000000 | 0.000000 |
| adaptive | 9 | 1.000000 | 1.000000 | 0.000000 |
| adaptive | 11 | 1.000000 | 1.000000 | 0.000000 |

At p=0.50, the decoder receives zero channel information (base_llr = 0). No schedule produces syndrome-consistent corrections under these conditions — the decoder output fails to satisfy even the parity constraints.

---

## 9. Config B — Mean Iterations (Transition Band)

| Schedule | d | p=0.49 | p=0.499 | p=0.50 |
|----------|---|--------|---------|--------|
| flooding | 7 | 1.0 | 1.0 | 50.0 |
| flooding | 9 | 1.0 | 1.0 | 50.0 |
| flooding | 11 | 1.0 | 1.0 | 50.0 |
| residual | 7 | 1.0 | 1.0 | 50.0 |
| residual | 9 | 1.0 | 1.0 | 50.0 |
| residual | 11 | 1.0 | 1.0 | 50.0 |
| adaptive | 7 | 1.0 | 1.0 | 12.0 |
| adaptive | 9 | 1.0 | 1.0 | 12.0 |
| adaptive | 11 | 1.0 | 1.0 | 12.0 |

The adaptive schedule terminates after 12 iterations (phase-1 budget) at p=0.50. Flooding and residual run to max_iters=50.

---

## 10. Runtime Tables

Runtime mode was set to `"off"` for deterministic artifact generation. No runtime measurements were collected for these configurations. See the prior industry comparison report (`industry_comparison_report_feb-26-2026.md`) for runtime data at these operating points.

---

## 11. Determinism Verification

### Config A

| Metric | Value |
|--------|-------|
| Artifact size | 8,539 bytes |
| SHA-256 hash | `0cc2e247c04852e1681bae007cda55dad26573d09ce912c7507d59fbf8392a4a` |
| Re-run comparison | **Byte-identical** |

### Config B

| Metric | Value |
|--------|-------|
| Artifact size | 8,548 bytes |
| SHA-256 hash | `43848a15f36d483e0ae49d0e16ef6ba44e89101eedcac1ccdcc72e60e96496c2` |
| Re-run comparison | **Byte-identical** |

Both configs use `deterministic_metadata=true` and `runtime_mode="off"`. Canonical JSON serialization with `sort_keys=True, separators=(",",":")`.

New fields (`fidelity`, `syndrome_success_rate`) participate fully in artifact hashing and do not break determinism guarantees.

---

## 12. Technical Interpretation

**Operating region (p <= 0.05):** Under the oracle channel model, the decoder has perfect access to error positions via LLR signs. Both new metrics are trivially 1.0 — the decoder produces exact corrections that satisfy all parity constraints.

**Transition band (p = 0.49 to 0.50):** The Fidelity and Syndrome Consistency Rate exhibit the same degenerate step-function behavior as FER:
- At p < 0.50: Fidelity = 1.0, SCR = 1.0 (perfect decoding).
- At p = 0.50: Fidelity = 0.0, SCR = 0.0 (no channel information, decoder outputs fail parity constraints).

**Syndrome Consistency vs Fidelity at p=0.50:** Both metrics collapse to 0.0, indicating the decoder not only fails logically but also fails to find any parity-consistent correction. This is expected: with zero LLR, BP messages remain at zero, and the hard-decision output is the all-zero vector, which does not match the syndrome of a non-trivial error pattern.

**Not comparable to Stim/PyMatching baselines.** The reference baselines from the prior report use different code families, noise models, and decoders. The new metrics are defined only for the QEC benchmark pipeline.

**Layer 1 decoder logic unchanged.**
