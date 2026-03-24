# QEC Benchmark Baseline — v3.1.4 Metric Expansion

**Project Version:** v3.1.4
**Git Commit:** `eae3dbd1d988ccbb8ad09d6da34090473a3334b9`
**Execution Environment:** Claude Code Compute
**Report Artifact:** `bench/benchmark_report_v3.1.4_metrics.md`

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

## 2. Benchmark Configuration

```json
{
  "schema_version": "3.0.1",
  "seed": 20260226,
  "distances": [3, 5, 7, 9],
  "p_values": [0.001, 0.002, 0.005, 0.01],
  "trials": 200,
  "max_iters": 50,
  "decoders": [
    {"adapter": "bp", "params": {"mode": "min_sum", "schedule": "flooding"}},
    {"adapter": "bp", "params": {"mode": "min_sum", "schedule": "layered"}},
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

## 4. FER Tables

All 64 sweep points (4 decoders x 4 distances x 4 physical error rates):

**FER = 0.000000** for all (decoder, distance, p) combinations.

All schedules converge in exactly 1 iteration at these operating points (p <= 0.01), well below the oracle channel threshold of p=0.50.

---

## 5. Fidelity Tables

| Schedule | d=3 | d=5 | d=7 | d=9 |
|----------|-----|-----|-----|-----|
| flooding | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| layered | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| residual | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| adaptive | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

Fidelity = 1.0 at all points for all p in {0.001, 0.002, 0.005, 0.01}.

---

## 6. Syndrome Consistency Tables

| Schedule | d=3 | d=5 | d=7 | d=9 |
|----------|-----|-----|-----|-----|
| flooding | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| layered | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| residual | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| adaptive | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

Syndrome Consistency Rate = 1.0 at all points for all p in {0.001, 0.002, 0.005, 0.01}.

---

## 7. Runtime Tables

Runtime mode was set to `"off"` for deterministic artifact generation. No runtime measurements were collected for this suite.

---

## 8. Determinism Verification

| Metric | Value |
|--------|-------|
| Artifact size | 18,760 bytes |
| SHA-256 hash | `174cbf9093bea70fca07c0e68a0cfb00f1a53f5a73eae05a5bc11306bf1602fb` |
| Re-run hash | `174cbf9093bea70fca07c0e68a0cfb00f1a53f5a73eae05a5bc11306bf1602fb` |
| Comparison | **Byte-identical** |
| Canonical JSON | `sort_keys=True, separators=(",",":")` |

The `deterministic_metadata=true` flag sets `created_utc` to `"1970-01-01T00:00:00+00:00"`, eliminating timestamp non-determinism. Combined with SHA-256-based sub-seed derivation (independent of sweep ordering), this guarantees reproducible results.

New fields (`fidelity`, `syndrome_success_rate`) participate fully in artifact hashing and do not break determinism guarantees.

---

## 9. Technical Interpretation

At physical error rates p <= 0.01, the oracle channel model provides perfect side information via LLR signs (`base_llr * (1 - 2*e[i])`). The min-sum decoder converges in a single iteration with zero frame errors across all schedules and distances. Consequently:

- **Fidelity = 1.0** universally, confirming no logical errors at these operating points.
- **Syndrome Consistency Rate = 1.0** universally, confirming that every decoder correction satisfies the parity-check constraints exactly.

These metrics become non-trivial (and differentiating) at or above the p=0.50 threshold boundary, where the decoder receives zero or inverted channel information.

This baseline establishes the deterministic baseline for both new metrics under v3.1.4 with channel architecture hardened.

**Layer 1 decoder logic unchanged.**
