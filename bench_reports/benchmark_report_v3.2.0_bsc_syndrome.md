# QEC Benchmark Baseline — v3.2.0 BSC Syndrome-Only

**Project Version:** v3.2.0 (work-in-progress)
**Git Commit:** `f8f2ee28512d5d12058c6b77db979134c198ee28`
**Execution Environment:** Claude Code Compute
**Report Artifact:** `bench/benchmark_report_v3.2.0_bsc_syndrome.md`
**Channel Model:** bsc_syndrome

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

### Config — Below-Threshold Region (p_low)

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

### Config — Near-Critical Region (p_near)

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

**Trial counts:** 200 trials for p_low, 500 trials for p_near.

---

## 3. Environment Snapshot

| Key | Value |
|-----|-------|
| Python version | 3.11.14 (main, Oct 10 2025, 08:54:04) [GCC 13.3.0] |
| NumPy version | 2.4.2 |
| Platform | Linux-4.4.0-x86_64-with-glibc2.39 |
| Git commit | `f8f2ee28512d5d12058c6b77db979134c198ee28` |

---

## 4. FER Tables (Below-Threshold Region)

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

### Near-Critical Region (p_near) — FER Summary

**FER = 1.000000** for all (schedule, distance, p) combinations in the near-critical region (p >= 0.45).

All schedules exhaust their iteration budgets without successful decoding. The syndrome-only channel provides no position-specific information, and at these high error rates the decoder cannot recover from the noise.

---

## 5. Fidelity Tables (Below-Threshold Region)

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

### Key Observation — Fidelity Under Syndrome-Only Channel

Unlike the oracle channel (where Fidelity = 1.0 for all p < 0.50), the syndrome-only channel produces non-trivial FER even at the lowest tested error rate (p=0.01). Fidelity degrades rapidly with increasing p and increasing code distance. This is the expected behavior: without position-specific LLR information, the BP decoder must rely solely on syndrome constraints, which becomes insufficient as the error density increases.

---

## 6. Syndrome Consistency Tables (Below-Threshold Region)

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

### Key Observation — SCR Tracks FER Under Syndrome-Only Channel

Under the syndrome-only channel, SCR closely tracks Fidelity (1 - FER). This is structurally different from the oracle channel, where SCR = 1.0 in the inverted regime (p > 0.50) despite FER = 1.0. Under bsc_syndrome, when the decoder fails logically, it also typically fails to satisfy the syndrome constraints — there is no "syndrome-consistent but logically wrong" regime. This is because the uniform LLR provides no directional bias, so the decoder cannot systematically converge to an incorrect-but-syndrome-consistent correction.

---

## 7. Mean Iteration Tables (Below-Threshold Region)

### d=3

| p | flooding | layered | hybrid_residual | adaptive |
|---|----------|---------|-----------------|----------|
| 0.01 | 13.74 | 15.21 | 16.435 | 4.465 |
| 0.02 | 25.5 | 23.05 | 25.99 | 7.05 |
| 0.05 | 41.18 | 39.465 | 41.915 | 10.13 |
| 0.1 | 48.04 | 47.55 | 47.305 | 11.56 |
| 0.15 | 49.51 | 49.265 | 49.265 | 11.89 |
| 0.2 | 49.51 | 49.51 | 49.265 | 11.945 |
| 0.25 | 49.755 | 50.0 | 49.755 | 12.0 |
| 0.3 | 50.0 | 49.755 | 50.0 | 12.0 |

### d=5

| p | flooding | layered | hybrid_residual | adaptive |
|---|----------|---------|-----------------|----------|
| 0.01 | 25.255 | 23.295 | 21.09 | 6.005 |
| 0.02 | 34.075 | 37.75 | 33.585 | 7.93 |
| 0.05 | 47.305 | 47.305 | 46.325 | 11.34 |
| 0.1 | 49.51 | 49.755 | 50.0 | 11.945 |
| 0.15 | 49.755 | 50.0 | 50.0 | 12.0 |
| 0.2 | 50.0 | 50.0 | 50.0 | 12.0 |
| 0.25 | 50.0 | 50.0 | 50.0 | 12.0 |
| 0.3 | 50.0 | 50.0 | 50.0 | 12.0 |

### d=7

| p | flooding | layered | hybrid_residual | adaptive |
|---|----------|---------|-----------------|----------|
| 0.01 | 26.235 | 29.42 | 32.605 | 6.94 |
| 0.02 | 39.465 | 40.69 | 41.915 | 9.69 |
| 0.05 | 49.02 | 49.02 | 48.775 | 11.725 |
| 0.1 | 50.0 | 50.0 | 50.0 | 12.0 |
| 0.15 | 50.0 | 50.0 | 50.0 | 12.0 |
| 0.2 | 50.0 | 50.0 | 50.0 | 12.0 |
| 0.25 | 50.0 | 50.0 | 50.0 | 12.0 |
| 0.3 | 50.0 | 50.0 | 50.0 | 12.0 |

---

## 8. Schedule Differentiation Summary

Unlike the oracle channel (where all schedules produce identical results for p < 0.50), the syndrome-only channel reveals genuine schedule differentiation across the entire operating range:

**At p=0.01 (best tested operating point):**

| Schedule | d=3 FER | d=5 FER | d=7 FER |
|----------|---------|---------|---------|
| flooding | 0.265000 | 0.495000 | 0.515000 |
| layered | 0.295000 | 0.455000 | 0.585000 |
| hybrid_residual | 0.320000 | 0.415000 | 0.645000 |
| adaptive | 0.325000 | 0.455000 | 0.545000 |

**Observations:**

1. **Flooding** tends to achieve the best FER at low p, likely due to its full parallel message update providing more information propagation per iteration.
2. **Layered** performs comparably to flooding, with slightly different FER at some operating points.
3. **Hybrid_residual** shows mixed performance — better than flooding at d=5/p=0.01 but worse at d=7.
4. **Adaptive** terminates early (12 iterations vs 50 max), which limits its performance under the harder syndrome-only decoding problem.
5. **Higher distance degrades performance** — the inverse of the expected oracle behavior. Without position information, larger codes present a harder combinatorial decoding problem.

---

## 9. Regime Interpretation — Syndrome-Only Channel

The bsc_syndrome channel model produces a fundamentally different regime structure compared to the oracle channel:

### Regime A — Partial Recovery (p <= ~0.02)

| Property | Value |
|----------|-------|
| FER | 0.265–0.840 (depends on schedule, distance) |
| Fidelity | 0.160–0.735 |
| SCR | Approximately equal to Fidelity |
| mean_iters | 4–42 (schedule-dependent) |
| Distance dependence | **Negative** — higher d = higher FER |
| Schedule dependence | **Yes** — meaningful differentiation |

The decoder achieves partial success using only syndrome constraints. Smaller codes (d=3) are easier to decode without position information. Schedule choice matters because the decoder is working near its information-theoretic limit.

### Regime B — Effective Failure (p >= ~0.05)

| Property | Value |
|----------|-------|
| FER | 0.800–1.000 |
| Fidelity | 0.000–0.200 |
| SCR | 0.000–0.215 |
| mean_iters | 39–50 (near max_iters) |
| Distance dependence | Minimal (all near failure) |
| Schedule dependence | Minimal |

The error density is too high for syndrome-only decoding. The decoder exhausts its iteration budget without finding valid corrections.

### Regime C — Complete Failure (p >= ~0.15)

| Property | Value |
|----------|-------|
| FER | 1.000 |
| Fidelity | 0.000 |
| SCR | 0.000–0.015 (stochastic noise) |
| mean_iters | 12 (adaptive) / 50 (others) |

**No inversion regime.** Unlike the oracle channel (where p > 0.50 produces SCR = 1.0 with FER = 1.0), the syndrome-only channel shows no recovery above any tested p value. This is expected: the BSC syndrome channel LLR magnitude decreases monotonically as p approaches 0.50 and the LLR does not carry position-specific sign information, so there is no sign-inversion mechanism to produce the "wrong but consistent" behavior observed under oracle.

---

## 10. Runtime Tables

Runtime mode was set to `"off"` for deterministic artifact generation. No runtime measurements were collected for this suite.

---

## 11. Determinism Verification

### Suite — Below-Threshold (p_low)

| Metric | Value |
|--------|-------|
| Artifact size | 28,651 bytes |
| SHA-256 hash | `0d4fc16bcdee53e05c05046daf9f880bf6e88926093ed2294b39a9c694891bd0` |
| Canonical JSON | `sort_keys=True, separators=(",",":")` |

### Suite — Near-Critical (p_near)

| Metric | Value |
|--------|-------|
| Artifact size | 41,185 bytes |
| SHA-256 hash | `657287091f5b2f6b09ffa8dc46f3f9befd755a85f15c989c5a9806178afc29ca` |
| Canonical JSON | `sort_keys=True, separators=(",",":")` |

### Determinism Audit Subset

**p_low subset** (p=[0.05, 0.2], schedules=['flooding', 'layered'], d=5, trials=200):

| Metric | Value |
|--------|-------|
| Size | 2,166 bytes |
| SHA-256 (run 1) | `0dcff6e17888d26c4d802f54fdd868c576bdb18c76f9a10e9818f582f6800534` |
| SHA-256 (run 2) | `0dcff6e17888d26c4d802f54fdd868c576bdb18c76f9a10e9818f582f6800534` |
| Comparison | **Byte-identical** |

**p_near subset** (p=[0.49, 0.5, 0.505], schedules=['flooding', 'layered'], d=5, trials=500):

| Metric | Value |
|--------|-------|
| Size | 2,671 bytes |
| SHA-256 (run 1) | `b6ebc00ad2eb5a656505a20d04bf954d165031bf3545a327fc30c69cdf45d31e` |
| SHA-256 (run 2) | `b6ebc00ad2eb5a656505a20d04bf954d165031bf3545a327fc30c69cdf45d31e` |
| Comparison | **Byte-identical** |

Determinism confirmed: `deterministic_metadata=true` + `runtime_mode="off"` produces byte-identical canonical JSON across runs. SHA-256 sub-seed derivation ensures order-independent reproducibility.

---

## 12. Technical Interpretation

The BSC syndrome-only channel (`bsc_syndrome`) produces a uniform LLR vector: `LLR_i = log((1-p)/p)` for all i, regardless of the actual error vector. The decoder receives no information about which specific bits are in error — it must rely entirely on the syndrome to locate errors.

This produces dramatically different behavior compared to the oracle channel:

1. **No trivial decoding regime.** Even at p=0.01, FER is non-zero (0.265–0.645 depending on schedule and distance). The decoder must perform genuine iterative inference rather than reading the answer from the LLR signs.

2. **Rapid FER degradation.** By p=0.05, FER exceeds 0.80 for all configurations. By p=0.10, FER is effectively 1.0. The effective operating range of the syndrome-only decoder is extremely narrow (p < ~0.02).

3. **Inverse distance scaling.** Under oracle, higher distance always improves or maintains performance. Under syndrome-only, higher distance makes decoding harder because the code has more variables to resolve without position-specific information.

4. **No inversion regime.** The oracle channel's sign inversion at p > 0.50 (producing SCR=1, FER=1) has no analogue under syndrome-only. The uniform LLR magnitude simply approaches zero as p approaches 0.50, and there is no mechanism to produce "wrong but syndrome-consistent" corrections.

5. **Schedule differentiation.** Unlike oracle mode (where all schedules are equivalent for p < 0.50), syndrome-only mode reveals genuine differences in schedule effectiveness. This is expected: when the decoder must perform real iterative inference, the message-passing order affects convergence behavior.

This baseline establishes the syndrome-only channel behavior against which future decoder improvements (e.g., post-processing, multi-stage decoding) can be measured.

**Layer 1 decoder logic unchanged.**
