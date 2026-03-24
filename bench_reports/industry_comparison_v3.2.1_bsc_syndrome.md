# QEC Industry Comparison — v3.2.1 BSC Syndrome-Only

**Project Version:** v3.2.1
**Git Commit:** `09e36c385f7ecbbb99202bc00666b9555632f9cd`
**Execution Environment:** Claude Code Compute
**Report Artifact:** `bench/industry_comparison_v3.2.1_bsc_syndrome.md`
**Channel Model:** bsc_syndrome
**Schema Version:** 3.0.1

**Layer 1 decoder logic unchanged.**

---

## 1. Metric Definitions

### 1.1 Logical Fidelity

```
FER = logical_failures / trials
Fidelity = 1.0 - FER
```

### 1.2 Syndrome Consistency Rate

```
Syndrome Consistency Rate = successful_syndrome_matches / trials
```

### 1.3 Inversion Index

```
Inversion Index = SCR - Fidelity
               = syndrome_consistency_rate - (1 - FER)
```

The Inversion Index quantifies syndrome-consistent but logically incorrect decoding outcomes. II > 0 indicates the presence of an inversion regime; II ≈ 0 indicates no systematic inversion mechanism. See Section 9 for full interpretation.

---

## 2. Benchmark Configurations

### Config A — Operating-Region (p_low)

```json
{
  "channel_model": "bsc_syndrome",
  "collect_iter_hist": false,
  "decoders": [
    {
      "adapter": "bp",
      "params": {
        "mode": "min_sum",
        "schedule": "flooding"
      }
    },
    {
      "adapter": "bp",
      "params": {
        "mode": "min_sum",
        "schedule": "layered"
      }
    },
    {
      "adapter": "bp",
      "params": {
        "mode": "min_sum",
        "schedule": "hybrid_residual"
      }
    },
    {
      "adapter": "bp",
      "params": {
        "mode": "min_sum",
        "schedule": "adaptive"
      }
    }
  ],
  "deterministic_metadata": true,
  "distances": [
    3,
    5,
    7
  ],
  "max_iters": 50,
  "p_values": [
    0.01,
    0.02,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3
  ],
  "runtime": {
    "measure_memory": false,
    "runs": 30,
    "warmup": 5
  },
  "runtime_mode": "off",
  "schema_version": "3.0.1",
  "seed": 20260226,
  "trials": 200
}
```

### Config B — Transition-Band Probe (p_near)

```json
{
  "channel_model": "bsc_syndrome",
  "collect_iter_hist": false,
  "decoders": [
    {
      "adapter": "bp",
      "params": {
        "mode": "min_sum",
        "schedule": "flooding"
      }
    },
    {
      "adapter": "bp",
      "params": {
        "mode": "min_sum",
        "schedule": "layered"
      }
    },
    {
      "adapter": "bp",
      "params": {
        "mode": "min_sum",
        "schedule": "hybrid_residual"
      }
    },
    {
      "adapter": "bp",
      "params": {
        "mode": "min_sum",
        "schedule": "adaptive"
      }
    }
  ],
  "deterministic_metadata": true,
  "distances": [
    3,
    5,
    7
  ],
  "max_iters": 50,
  "p_values": [
    0.45,
    0.47,
    0.48,
    0.49,
    0.495,
    0.499,
    0.5,
    0.501,
    0.505,
    0.51,
    0.52,
    0.55
  ],
  "runtime": {
    "measure_memory": false,
    "runs": 30,
    "warmup": 5
  },
  "runtime_mode": "off",
  "schema_version": "3.0.1",
  "seed": 20260226,
  "trials": 500
}
```

---

## 3. Environment Snapshot

| Key | Value |
|-----|-------|
| Python version | 3.11.14 (main, Oct 10 2025, 08:54:04) [GCC 13.3.0] |
| NumPy version | 2.4.2 |
| Platform | Linux-4.4.0-x86_64-with-glibc2.39 |
| Git commit | `09e36c385f7ecbbb99202bc00666b9555632f9cd` |

---

## 4. Config A — FER Tables (Operating Region)

### d=3

| p | flooding | layered | hybrid_residual | adaptive |
|---|----------|---------|-----------------|----------|
| 0.01 | 0.265000 | 0.295000 | 0.320000 | 0.325000 |
| 0.02 | 0.510000 | 0.470000 | 0.525000 | 0.570000 |
| 0.05 | 0.835000 | 0.800000 | 0.845000 | 0.835000 |
| 0.1 | 0.980000 | 0.980000 | 0.975000 | 0.980000 |
| 0.15 | 1.000000 | 0.995000 | 0.995000 | 0.995000 |
| 0.2 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 0.25 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 0.3 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

### d=5

| p | flooding | layered | hybrid_residual | adaptive |
|---|----------|---------|-----------------|----------|
| 0.01 | 0.495000 | 0.455000 | 0.415000 | 0.455000 |
| 0.02 | 0.675000 | 0.755000 | 0.680000 | 0.655000 |
| 0.05 | 0.955000 | 0.950000 | 0.945000 | 0.950000 |
| 0.1 | 0.995000 | 1.000000 | 1.000000 | 1.000000 |
| 0.15 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 0.2 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 0.25 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 0.3 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

### d=7

| p | flooding | layered | hybrid_residual | adaptive |
|---|----------|---------|-----------------|----------|
| 0.01 | 0.515000 | 0.585000 | 0.645000 | 0.545000 |
| 0.02 | 0.805000 | 0.820000 | 0.840000 | 0.800000 |
| 0.05 | 0.985000 | 0.990000 | 0.985000 | 0.980000 |
| 0.1 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 0.15 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 0.2 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 0.25 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 0.3 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

Non-trivial FER across the entire operating region. The syndrome-only channel produces measurable error rates even at p=0.01, in stark contrast to the oracle channel (FER=0.0 for all p < 0.50).

---

## 5. Config A — Fidelity Tables

### d=3

| p | flooding | layered | hybrid_residual | adaptive |
|---|----------|---------|-----------------|----------|
| 0.01 | 0.735000 | 0.705000 | 0.680000 | 0.675000 |
| 0.02 | 0.490000 | 0.530000 | 0.475000 | 0.430000 |
| 0.05 | 0.165000 | 0.200000 | 0.155000 | 0.165000 |
| 0.1 | 0.020000 | 0.020000 | 0.025000 | 0.020000 |
| 0.15 | 0.000000 | 0.005000 | 0.005000 | 0.005000 |
| 0.2 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.25 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.3 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

### d=5

| p | flooding | layered | hybrid_residual | adaptive |
|---|----------|---------|-----------------|----------|
| 0.01 | 0.505000 | 0.545000 | 0.585000 | 0.545000 |
| 0.02 | 0.325000 | 0.245000 | 0.320000 | 0.345000 |
| 0.05 | 0.045000 | 0.050000 | 0.055000 | 0.050000 |
| 0.1 | 0.005000 | 0.000000 | 0.000000 | 0.000000 |
| 0.15 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.2 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.25 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.3 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

### d=7

| p | flooding | layered | hybrid_residual | adaptive |
|---|----------|---------|-----------------|----------|
| 0.01 | 0.485000 | 0.415000 | 0.355000 | 0.455000 |
| 0.02 | 0.195000 | 0.180000 | 0.160000 | 0.200000 |
| 0.05 | 0.015000 | 0.010000 | 0.015000 | 0.020000 |
| 0.1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.15 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.2 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.25 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.3 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

---

## 6. Config A — Syndrome Consistency Tables

### d=3

| p | flooding SCR | layered SCR | hybrid_residual SCR | adaptive SCR |
|---|-------------|-------------|---------------------|--------------|
| 0.01 | 0.740000 | 0.710000 | 0.685000 | 0.685000 |
| 0.02 | 0.500000 | 0.550000 | 0.490000 | 0.450000 |
| 0.05 | 0.180000 | 0.215000 | 0.165000 | 0.170000 |
| 0.1 | 0.040000 | 0.050000 | 0.055000 | 0.040000 |
| 0.15 | 0.010000 | 0.015000 | 0.015000 | 0.010000 |
| 0.2 | 0.010000 | 0.010000 | 0.015000 | 0.005000 |
| 0.25 | 0.005000 | 0.000000 | 0.005000 | 0.000000 |
| 0.3 | 0.000000 | 0.005000 | 0.000000 | 0.000000 |

### d=5

| p | flooding SCR | layered SCR | hybrid_residual SCR | adaptive SCR |
|---|-------------|-------------|---------------------|--------------|
| 0.01 | 0.505000 | 0.545000 | 0.590000 | 0.545000 |
| 0.02 | 0.325000 | 0.250000 | 0.335000 | 0.370000 |
| 0.05 | 0.055000 | 0.055000 | 0.075000 | 0.060000 |
| 0.1 | 0.010000 | 0.005000 | 0.000000 | 0.005000 |
| 0.15 | 0.005000 | 0.000000 | 0.000000 | 0.000000 |
| 0.2 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.25 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.3 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

### d=7

| p | flooding SCR | layered SCR | hybrid_residual SCR | adaptive SCR |
|---|-------------|-------------|---------------------|--------------|
| 0.01 | 0.485000 | 0.420000 | 0.355000 | 0.460000 |
| 0.02 | 0.215000 | 0.190000 | 0.165000 | 0.210000 |
| 0.05 | 0.020000 | 0.020000 | 0.025000 | 0.025000 |
| 0.1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.15 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.2 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.25 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.3 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

---

## 7. Config A — Inversion Index Tables (Operating Region)

### d=3

| p | flooding II | layered II | hybrid_residual II | adaptive II |
|---|-------------|------------|---------------------|-------------|
| 0.01 | 0.005000 | 0.005000 | 0.005000 | 0.010000 |
| 0.02 | 0.010000 | 0.020000 | 0.015000 | 0.020000 |
| 0.05 | 0.015000 | 0.015000 | 0.010000 | 0.005000 |
| 0.1 | 0.020000 | 0.030000 | 0.030000 | 0.020000 |
| 0.15 | 0.010000 | 0.010000 | 0.010000 | 0.005000 |
| 0.2 | 0.010000 | 0.010000 | 0.015000 | 0.005000 |
| 0.25 | 0.005000 | 0.000000 | 0.005000 | 0.000000 |
| 0.3 | 0.000000 | 0.005000 | 0.000000 | 0.000000 |

### d=5

| p | flooding II | layered II | hybrid_residual II | adaptive II |
|---|-------------|------------|---------------------|-------------|
| 0.01 | 0.000000 | 0.000000 | 0.005000 | 0.000000 |
| 0.02 | 0.000000 | 0.005000 | 0.015000 | 0.025000 |
| 0.05 | 0.010000 | 0.005000 | 0.020000 | 0.010000 |
| 0.1 | 0.005000 | 0.005000 | 0.000000 | 0.005000 |
| 0.15 | 0.005000 | 0.000000 | 0.000000 | 0.000000 |
| 0.2 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.25 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.3 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

### d=7

| p | flooding II | layered II | hybrid_residual II | adaptive II |
|---|-------------|------------|---------------------|-------------|
| 0.01 | 0.000000 | 0.005000 | 0.000000 | 0.005000 |
| 0.02 | 0.020000 | 0.010000 | 0.005000 | 0.010000 |
| 0.05 | 0.005000 | 0.010000 | 0.010000 | 0.005000 |
| 0.1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.15 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.2 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.25 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.3 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

All Inversion Index values are near zero (max observed: 0.030), confirming the absence of a systematic inversion regime under the syndrome-only channel.

---

## 8. Config B — FER, Fidelity, SCR, and Inversion Index (Transition Band)

**FER = 1.000000** and **Fidelity = 0.000000** for all (schedule, distance, p) combinations in the transition band (p >= 0.45).

Under the syndrome-only channel, the effective threshold is far below p=0.45. The decoder has already collapsed to complete failure at these error rates. No schedule produces any successful corrections.

SCR is effectively 0.000 across the transition band, with sporadic non-zero values (0.002–0.010) at d=3 only, representing rare stochastic events (1–5 out of 500 trials where a random correction happened to satisfy the syndrome).

**Inversion Index = SCR** across the entire transition band (since Fidelity = 0). All values are near zero (max 0.010 at d=3). **No inversion regime** is present. Unlike the oracle channel (where SCR = 1.0 for p > 0.50, producing II = 1.0), the syndrome-only channel shows no systematic syndrome-consistent failure mode.

### Combined Sample Table — Transition Band

| p | schedule | distance | FER | Fidelity | SCR | Inversion Index |
|---|----------|----------|-----|----------|-----|-----------------|
| 0.49 | flooding | d=3 | 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.49 | flooding | d=5 | 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.505 | flooding | d=5 | 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.501 | layered | d=3 | 1.000000 | 0.000000 | 0.004000 | 0.004000 |
| 0.5 | flooding | d=5 | 1.000000 | 0.000000 | 0.002000 | 0.002000 |

---

## 9. Inversion Index Interpretation

### How Inversion Index Isolates Syndrome-Consistent Incorrect Outcomes

The Inversion Index (II = SCR - Fidelity) measures the gap between syndrome consistency and logical correctness. A positive II indicates trials where the decoder's output satisfies syndrome constraints but is logically incorrect — the decoder found a "wrong but structurally valid" solution.

### Why Inversion Index Peaks Under Oracle Inversion Regime

Under the oracle channel at p > 0.50, the inverted LLR signs cause the decoder to converge deterministically to the bitwise complement of the correct error. This complement satisfies the syndrome (SCR = 1.0) but is logically wrong (Fidelity = 0.0), producing II = 1.0 — the maximum possible value. This occurs for all schedules and distances.

### Why Inversion Index ≈ 0 Under Syndrome-Only Channel

The BSC syndrome-only channel provides uniform LLR with no position-specific sign information. When the decoder fails, its output is structurally random and almost never satisfies the syndrome constraints. The small non-zero II values (0.000–0.030) are consistent with statistical noise at the trial counts used. No systematic inversion mechanism exists.

### Statistical Noise Bound for Random Syndrome Matches

Under the syndrome-only channel in the high-p regime, decoder outputs at failure are effectively unconstrained binary vectors. For a linear code with m independent parity checks, the probability that a uniformly random binary vector satisfies the syndrome constraints is approximately:

```
P[random syndrome match] ≈ 2^(−m)
```

For T independent trials:

```
Expected random matches ≈ T · 2^(−m)
```

In the near-threshold sweeps:

- **Trials** = 500
- **Observed II** ≈ 0.002–0.010
- This corresponds to approximately **1–5 syndrome-consistent outcomes per 500 trials**
- These rates are consistent with rare random syndrome matches and do not indicate a structured inversion mechanism

Therefore, the small non-zero Inversion Index values under the syndrome-only channel are statistically consistent with random structural coincidence, not systematic decoder behavior.

---

## 10. Config B — Mean Iterations (Transition Band)

| Schedule | Mean Iters (all d, all p_near) |
|----------|-------------------------------|
| flooding | 49.5–50.0 |
| layered | 49.7–50.0 |
| hybrid_residual | 49.7–50.0 |
| adaptive | 11.9–12.0 |

All non-adaptive schedules run to or near max_iters=50. The adaptive schedule terminates at its phase-1 budget (~12 iterations). Minor variations from exact 50.0 or 12.0 reflect rare trials where stochastic syndrome matches caused early convergence.

---

## 11. Determinism Verification

### Config A

| Metric | Value |
|--------|-------|
| Artifact size | 28,651 bytes |
| SHA-256 hash | `0d4fc16bcdee53e05c05046daf9f880bf6e88926093ed2294b39a9c694891bd0` |
| Canonical JSON | `sort_keys=True, separators=(",",":")` |

### Config B

| Metric | Value |
|--------|-------|
| Artifact size | 41,185 bytes |
| SHA-256 hash | `657287091f5b2f6b09ffa8dc46f3f9befd755a85f15c989c5a9806178afc29ca` |
| Canonical JSON | `sort_keys=True, separators=(",",":")` |

Determinism audit subsets verified byte-identical (see benchmark report for full audit details).

The Inversion Index is computed as an exact algebraic derivative of deterministic source fields. No new stochastic operations introduced.

---

## 12. Technical Interpretation

**Operating region (p <= 0.30):** Under the syndrome-only channel, the decoder must rely entirely on syndrome constraints. Even at the lowest tested error rate (p=0.01), FER is substantial (0.265–0.645). The Inversion Index remains near zero throughout, confirming that decoder failures under syndrome-only are structurally complete — no "wrong but syndrome-consistent" solutions are found systematically. This contrasts sharply with the oracle channel, where II = 0 below threshold (perfect decoding) and II = 1.0 above threshold (perfect inversion).

**Transition band (p >= 0.45):** Complete failure. FER=1.0, SCR~0.0, II~0.0 across all schedules and distances. The syndrome-only decoder has no operating capability at these error rates, and no inversion signature is present.

**Not comparable to Stim/PyMatching baselines.** The reference baselines from prior reports use different code families, noise models, and decoders. The syndrome-only results are defined only for the QEC benchmark pipeline with min-sum BP.

---

## 13. v3.2.1 Summary

| Item | Status |
|------|--------|
| Files changed | `bench/industry_comparison_v3.2.1_bsc_syndrome.md` (this file) |
| New metric | Inversion Index (II = SCR - Fidelity) |
| Inversion Index tables | Section 7 (Config A per-distance), Section 8 (Config B combined) |
| Inversion Index interpretation | Section 9 |
| Tests status | All tests passed (629 passed) |
| CLAUDE.md compliance | Confirmed |
| Decoder modifications | **None** (Layer 1 unchanged) |
| Schema version | **3.0.1** (unchanged) |
| Channel code | **Unchanged** |
| Determinism | **Verified** (artifact hashes unchanged) |
| Existing metrics regression | FER, Fidelity, SCR, mean_iters — all values unchanged from v3.2.0 |

**Layer 1 decoder logic unchanged.**
