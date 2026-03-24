# QEC Industry Comparison — v3.2.0 BSC Syndrome-Only

**Project Version:** v3.2.0 (work-in-progress)
**Git Commit:** `f8f2ee28512d5d12058c6b77db979134c198ee28`
**Execution Environment:** Claude Code Compute
**Report Artifact:** `bench/industry_comparison_v3.2.0_bsc_syndrome.md`
**Channel Model:** bsc_syndrome

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
| Git commit | `f8f2ee28512d5d12058c6b77db979134c198ee28` |

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

## 7. Config B — FER and Fidelity (Transition Band)

**FER = 1.000000** and **Fidelity = 0.000000** for all (schedule, distance, p) combinations in the transition band (p >= 0.45).

Under the syndrome-only channel, the effective threshold is far below p=0.45. The decoder has already collapsed to complete failure at these error rates. No schedule produces any successful corrections.

---

## 8. Config B — Syndrome Consistency (Transition Band)

SCR is effectively 0.000 across the transition band, with sporadic non-zero values (0.002–0.010) at d=3 only, representing rare stochastic events (1–5 out of 500 trials where a random correction happened to satisfy the syndrome).

**No inversion regime.** Unlike the oracle channel (where SCR = 1.0 for p > 0.50), the syndrome-only channel shows no systematic syndrome-consistent failure mode. When the decoder fails, it fails completely — no structurally valid incorrect corrections are produced.

---

## 9. Config B — Mean Iterations (Transition Band)

| Schedule | Mean Iters (all d, all p_near) |
|----------|-------------------------------|
| flooding | 49.5–50.0 |
| layered | 49.7–50.0 |
| hybrid_residual | 49.7–50.0 |
| adaptive | 11.9–12.0 |

All non-adaptive schedules run to or near max_iters=50. The adaptive schedule terminates at its phase-1 budget (~12 iterations). Minor variations from exact 50.0 or 12.0 reflect rare trials where stochastic syndrome matches caused early convergence.

---

## 10. Determinism Verification

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

---

## 11. Technical Interpretation

**Operating region (p <= 0.30):** Under the syndrome-only channel, the decoder must rely entirely on syndrome constraints. Even at the lowest tested error rate (p=0.01), FER is substantial (0.265–0.645). This contrasts sharply with the oracle channel, where FER=0.0 for all p < 0.50.

**Transition band (p >= 0.45):** Complete failure. FER=1.0, SCR~0.0 across all schedules and distances. The syndrome-only decoder has no operating capability at these error rates.

**Not comparable to Stim/PyMatching baselines.** The reference baselines from prior reports use different code families, noise models, and decoders. The syndrome-only results are defined only for the QEC benchmark pipeline with min-sum BP.

**Layer 1 decoder logic unchanged.**
