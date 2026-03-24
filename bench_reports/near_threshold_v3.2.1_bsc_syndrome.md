# QEC Near-Threshold Comparative Benchmark — v3.2.1 BSC Syndrome-Only

**Project Version:** v3.2.1
**Git Commit:** `09e36c385f7ecbbb99202bc00666b9555632f9cd`
**Execution Environment:** Claude Code Compute
**Report Artifact:** `bench/near_threshold_v3.2.1_bsc_syndrome.md`
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

The Inversion Index quantifies syndrome-consistent but logically incorrect decoding outcomes. II > 0 indicates the presence of an inversion regime; II ≈ 0 indicates no systematic inversion mechanism. See Section 8 for full interpretation.

---

## 2. Benchmark Configuration

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

**Trials:** 500 per sweep point.

---

## 3. Environment Snapshot

| Key | Value |
|-----|-------|
| Python version | 3.11.14 (main, Oct 10 2025, 08:54:04) [GCC 13.3.0] |
| NumPy version | 2.4.2 |
| Platform | Linux-4.4.0-x86_64-with-glibc2.39 |
| Git commit | `09e36c385f7ecbbb99202bc00666b9555632f9cd` |

---

## 4. FER Tables

**FER = 1.000000** for all (schedule, distance, p) combinations.

All 144 sweep points (4 schedules x 3 distances x 12 p_near values) produce FER = 1.0. Under the syndrome-only channel, the effective threshold is far below the p_near range (p >= 0.45). The decoder has no decoding capability in this region.

---

## 5. Fidelity Tables

**Fidelity = 0.000000** for all (schedule, distance, p) combinations.

---

## 6. Syndrome Consistency Tables

### d=3

| p | flooding SCR | layered SCR | hybrid_residual SCR | adaptive SCR |
|---|-------------|-------------|---------------------|--------------|
| 0.45 | 0.002000 | 0.000000 | 0.006000 | 0.000000 |
| 0.47 | 0.006000 | 0.002000 | 0.006000 | 0.000000 |
| 0.48 | 0.010000 | 0.002000 | 0.002000 | 0.000000 |
| 0.49 | 0.000000 | 0.002000 | 0.004000 | 0.002000 |
| 0.495 | 0.002000 | 0.004000 | 0.000000 | 0.000000 |
| 0.499 | 0.002000 | 0.002000 | 0.002000 | 0.000000 |
| 0.5 | 0.002000 | 0.002000 | 0.000000 | 0.004000 |
| 0.501 | 0.002000 | 0.004000 | 0.004000 | 0.004000 |
| 0.505 | 0.004000 | 0.000000 | 0.004000 | 0.000000 |
| 0.51 | 0.004000 | 0.004000 | 0.000000 | 0.006000 |
| 0.52 | 0.002000 | 0.002000 | 0.002000 | 0.006000 |
| 0.55 | 0.000000 | 0.006000 | 0.002000 | 0.006000 |

### d=5, d=7

SCR = 0.000000 for all schedules and p values, with the following exceptions:

- d=5, flooding, p=0.5: SCR=0.002000
- d=5, hybrid_residual, p=0.5: SCR=0.002000

The sporadic non-zero SCR values at d=3 and the rare exceptions at d=5 represent stochastic events where a random decoder output happened to satisfy the syndrome constraints by chance. These events occur at rates consistent with the probability of a random binary vector having the correct syndrome (approximately `2^(-m)` where m is the number of checks).

---

## 7. Inversion Index Tables

Since Fidelity = 0.000000 across all near-threshold sweep points, the Inversion Index equals the Syndrome Consistency Rate: **II = SCR - 0 = SCR**.

### d=3

| p | flooding II | layered II | hybrid_residual II | adaptive II |
|---|-------------|------------|---------------------|-------------|
| 0.45 | 0.002000 | 0.000000 | 0.006000 | 0.000000 |
| 0.47 | 0.006000 | 0.002000 | 0.006000 | 0.000000 |
| 0.48 | 0.010000 | 0.002000 | 0.002000 | 0.000000 |
| 0.49 | 0.000000 | 0.002000 | 0.004000 | 0.002000 |
| 0.495 | 0.002000 | 0.004000 | 0.000000 | 0.000000 |
| 0.499 | 0.002000 | 0.002000 | 0.002000 | 0.000000 |
| 0.5 | 0.002000 | 0.002000 | 0.000000 | 0.004000 |
| 0.501 | 0.002000 | 0.004000 | 0.004000 | 0.004000 |
| 0.505 | 0.004000 | 0.000000 | 0.004000 | 0.000000 |
| 0.51 | 0.004000 | 0.004000 | 0.000000 | 0.006000 |
| 0.52 | 0.002000 | 0.002000 | 0.002000 | 0.006000 |
| 0.55 | 0.000000 | 0.006000 | 0.002000 | 0.006000 |

### d=5, d=7

**II = 0.000000** for all schedules and p values, with the same rare exceptions as SCR:

- d=5, flooding, p=0.5: II=0.002000
- d=5, hybrid_residual, p=0.5: II=0.002000

All other points: II = 0.000000.

### Combined Sample Table — Inversion Index Derivation

| p | schedule | distance | FER | Fidelity | SCR | Inversion Index |
|---|----------|----------|-----|----------|-----|-----------------|
| 0.49 | flooding | d=3 | 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.49 | flooding | d=5 | 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.5 | flooding | d=5 | 1.000000 | 0.000000 | 0.002000 | 0.002000 |
| 0.505 | flooding | d=5 | 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.501 | layered | d=3 | 1.000000 | 0.000000 | 0.004000 | 0.004000 |

All values verified: II = SCR - Fidelity exactly. Sporadic non-zero values represent stochastic noise, not a systematic inversion mechanism.

---

## 8. Inversion Index Interpretation

### How Inversion Index Isolates Syndrome-Consistent Incorrect Outcomes

The Inversion Index (II = SCR - Fidelity) captures the fraction of trials where the decoder produces a correction that satisfies the syndrome constraints but is logically incorrect. In the near-threshold regime under the syndrome-only channel, II is uniformly near zero because the decoder cannot find syndrome-consistent solutions at all — when it fails logically, it also fails structurally.

### Why Inversion Index Peaks Under Oracle Inversion Regime

Under the oracle channel at p > 0.50, the LLR sign inversion causes the decoder to converge to the bitwise complement of the correct error vector. This complement satisfies the syndrome constraints (SCR = 1.0) while being logically wrong (Fidelity = 0.0), yielding II = 1.0. This is the maximum possible Inversion Index and occurs deterministically for all (schedule, distance) combinations in the oracle inverted regime.

By contrast, the near-threshold syndrome-only data shows **no such phenomenon**: the maximum observed II is 0.010 (at d=3, flooding, p=0.48), consistent with 5 out of 500 trials where a random decoder output happened to satisfy the syndrome by chance.

### Why Inversion Index ≈ 0 Under Syndrome-Only Channel

The BSC syndrome-only channel provides a uniform LLR magnitude with no position-specific sign information. At p >= 0.45, the LLR magnitude is small (`|log((1-p)/p)| <= 0.20`), providing minimal directional prior. The decoder cannot converge to any specific solution — correct or incorrect but syndrome-consistent. Without the oracle's LLR sign to guide it toward a specific (wrong) answer, the decoder's output at failure is structurally random and almost never satisfies the syndrome constraints.

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

### Contrast with Oracle Near-Threshold Behavior

```
Oracle channel (p > 0.50):
  FER = 1.0, Fidelity = 0.0, SCR = 1.0 → II = 1.0
  → Decoder finds wrong-but-valid solution (deterministic inversion)

Syndrome-only channel (p >= 0.45):
  FER = 1.0, Fidelity = 0.0, SCR ≈ 0.0 → II ≈ 0.0
  → Decoder finds no valid solution at all (complete structural failure)
```

The Inversion Index cleanly separates these two failure modes: "wrong but structurally valid" (II >> 0) vs "structurally invalid failure" (II ≈ 0).

---

## 9. Mean Iteration Tables

### Summary Across All Distances

| Schedule | Typical mean_iters |
|----------|-------------------|
| flooding | 49.8–50.0 |
| layered | 49.7–50.0 |
| hybrid_residual | 49.7–50.0 |
| adaptive | 11.9–12.0 |

The flooding, layered, and hybrid_residual schedules run to or near max_iters=50 at all sweep points. The adaptive schedule terminates at its phase-1 budget (~12 iterations). Minor variations from the maximums correspond to rare successful-syndrome trials that triggered early convergence.

---

## 10. Key Observations — Syndrome-Only Near-Threshold Behavior

### 10.1 No Threshold Transition

Under the oracle channel, the near-threshold region (p = 0.49–0.505) exhibits a sharp phase transition at p=0.50. Under the syndrome-only channel, no such transition exists in this region because the decoder has already fully collapsed by p ~0.05. The p_near sweep is entirely within the complete-failure regime.

### 10.2 No SCR/FER Divergence

Under the oracle channel at p > 0.50, SCR = 1.0 while FER = 1.0 (the decoder finds syndrome-consistent but logically wrong corrections, producing II = 1.0). Under the syndrome-only channel, no such divergence is observed: both SCR and FER indicate total failure, and II ≈ 0.0. The uniform LLR provides no directional bias that could steer the decoder to a syndrome-consistent wrong answer.

### 10.3 Depressed Effective Threshold

The effective threshold of the syndrome-only decoder (defined as the max p where FER < 0.50) is approximately:

| Schedule | d=3 p* | d=5 p* | d=7 p* |
|----------|--------|--------|--------|
| flooding | ~0.01 | ~0.01 | < 0.01 |
| layered | ~0.02 | ~0.01 | < 0.01 |
| hybrid_residual | ~0.01 | ~0.01 | < 0.01 |
| adaptive | ~0.01 | ~0.01 | < 0.01 |

Compared to the oracle channel effective threshold of p ~0.4999, the syndrome-only threshold is displaced by approximately Δp = 0.48–0.49+. This massive threshold depression quantifies the information advantage provided by the oracle channel's position-specific LLR signs.

### 10.4 Regime Structure — Syndrome-Only vs Oracle

```
Oracle channel:
p:     0          0.50          1.0
       |          |             |
       | Stable   |Degen| Inverted   |
FER:   0.000      1.000  1.000
SCR:   1.000      ~0.00  1.000
II:    0.000      ~0.00  1.000

Syndrome-only channel:
p:     0    ~0.02  ~0.05         1.0
       |     |      |             |
       |Part.|Trans.| Complete Failure  |
FER:   0.3+   0.5+   0.8+    1.000
SCR:   ~FER   ~FER   ~0.00   ~0.000
II:    ~0.00  ~0.00  ~0.00   ~0.000
```

The syndrome-only channel eliminates both the stable regime (Regime I) and the inverted regime (Regime III) observed under oracle. The Inversion Index confirms this: II remains near zero across the entire p range, with no inversion signature.

---

## 11. Determinism Verification

| Metric | Value |
|--------|-------|
| Artifact size | 41,185 bytes |
| SHA-256 hash | `657287091f5b2f6b09ffa8dc46f3f9befd755a85f15c989c5a9806178afc29ca` |
| Canonical JSON | `sort_keys=True, separators=(",",":")` |

**Determinism audit subset** (p=[0.49, 0.5, 0.505], schedules=['flooding', 'layered'], d=5):

| Metric | Value |
|--------|-------|
| SHA-256 (run 1) | `b6ebc00ad2eb5a656505a20d04bf954d165031bf3545a327fc30c69cdf45d31e` |
| SHA-256 (run 2) | `b6ebc00ad2eb5a656505a20d04bf954d165031bf3545a327fc30c69cdf45d31e` |
| Comparison | **Byte-identical** |

The Inversion Index is computed as an exact algebraic derivative of deterministic source fields. No new stochastic operations introduced.

---

## 12. Technical Interpretation

The near-threshold sweep under the syndrome-only channel reveals that the entire p_near region (p = 0.45–0.55) is in complete decoder failure. This is the expected consequence of removing oracle side-information:

**Below oracle threshold (p < 0.50, syndrome-only):** The decoder has already failed. The uniform LLR `log((1-p)/p)` provides only a weak prior that "most bits are likely correct" but no information about *which* bits are in error. At p >= 0.05, this prior is too weak for BP to converge to valid corrections.

**At oracle threshold (p = 0.50, syndrome-only):** LLR = 0 for all bits. The decoder receives zero channel information, identical to the oracle case. The difference is that under oracle, this is the only failure point; under syndrome-only, it is merely one point in a wide failure band.

**Above oracle threshold (p > 0.50, syndrome-only):** LLR magnitude increases but with uniform (non-inverted) sign. The decoder gets a prior that "most bits are likely in error" but still cannot identify specific positions. There is no sign-inversion mechanism to produce syndrome-consistent wrong corrections, so both SCR and II remain near 0.

The Inversion Index provides the sharpest diagnostic for this distinction: under oracle at p > 0.50, II = 1.0 (maximum inversion); under syndrome-only at all p, II ≈ 0.0 (no inversion). This confirms that the oracle channel's Regime III is an artifact of its position-specific LLR sign structure, not an intrinsic property of BP decoding.

---

## 13. v3.2.1 Summary

| Item | Status |
|------|--------|
| Files changed | `bench/near_threshold_v3.2.1_bsc_syndrome.md` (this file) |
| New metric | Inversion Index (II = SCR - Fidelity) |
| Inversion Index tables | Section 7 (per-distance tables + combined sample) |
| Inversion Index interpretation | Section 8 |
| Tests status | All tests passed (629 passed) |
| CLAUDE.md compliance | Confirmed |
| Decoder modifications | **None** (Layer 1 unchanged) |
| Schema version | **3.0.1** (unchanged) |
| Channel code | **Unchanged** |
| Determinism | **Verified** (artifact hashes unchanged) |
| Existing metrics regression | FER, Fidelity, SCR, mean_iters — all values unchanged from v3.2.0 |

**Layer 1 decoder logic unchanged.**
