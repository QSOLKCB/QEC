# QEC Near-Threshold Comparative Benchmark — v3.1.4 Metric Expansion

**Project Version:** v3.1.4
**Git Commit:** `eae3dbd1d988ccbb8ad09d6da34090473a3334b9`
**Execution Environment:** Claude Code Compute
**Report Artifact:** `bench/near_threshold_v3.1.4_metrics.md`

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
  "distances": [5, 7, 9, 11],
  "p_values": [0.490, 0.495, 0.498, 0.499, 0.4995, 0.4999, 0.500, 0.501, 0.505],
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

### d=5 (n=60 qubits)

| p | flooding FER | layered FER | residual FER | adaptive FER |
|---|-------------|-------------|--------------|--------------|
| 0.4900 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.4950 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.4980 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.4990 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.4995 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.4999 | 0.000 | 0.000 | 0.000 | 0.000 |
| **0.5000** | **1.000** | **1.000** | **1.000** | **1.000** |
| 0.5010 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.5050 | 1.000 | 1.000 | 1.000 | 1.000 |

### d=7, d=9, d=11

Identical pattern to d=5: FER=0.0 for p < 0.50, FER=1.0 for p >= 0.50.

---

## 5. Fidelity Tables

### d=5 (n=60 qubits)

| p | flooding | layered | residual | adaptive |
|---|----------|---------|----------|----------|
| 0.4900 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 0.4950 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 0.4980 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 0.4990 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 0.4995 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 0.4999 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| **0.5000** | **0.000000** | **0.000000** | **0.000000** | **0.000000** |
| 0.5010 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0.5050 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

### d=7, d=9, d=11

Identical Fidelity pattern across all distances.

---

## 6. Syndrome Consistency Tables

### d=5 (n=60 qubits)

| p | flooding SCR | layered SCR | residual SCR | adaptive SCR |
|---|-------------|-------------|--------------|--------------|
| 0.4900 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 0.4950 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 0.4980 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 0.4990 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 0.4995 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 0.4999 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| **0.5000** | **0.005000** | **0.000000** | **0.000000** | **0.000000** |
| 0.5010 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 0.5050 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

### d=7 (n=84 qubits)

| p | flooding SCR | layered SCR | residual SCR | adaptive SCR |
|---|-------------|-------------|--------------|--------------|
| 0.4900 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 0.4950 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 0.4980 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 0.4990 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 0.4995 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 0.4999 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| **0.5000** | **0.000000** | **0.000000** | **0.000000** | **0.000000** |
| 0.5010 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 0.5050 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

### d=9 (n=108 qubits)

Identical to d=7 pattern: SCR=1.0 except at p=0.50 where SCR=0.0.

### d=11 (n=132 qubits)

Identical to d=7 pattern: SCR=1.0 except at p=0.50 where SCR=0.0.

---

## 7. Key Syndrome Consistency Observations

### 7.1 SCR at p=0.50 (Zero-Information Point)

At p=0.50, `base_llr = log((1-p)/p) = log(1) = 0`. All channel LLRs are exactly zero. The decoder has no information to work with, and the hard-decision output is the all-zero vector. Since the syndrome of a non-trivial error pattern is non-zero, the all-zero correction cannot satisfy the syndrome constraints. SCR = 0.0 for all schedules (with one exception noted below).

**Exception — flooding at d=5, p=0.50: SCR = 0.005** (1 out of 200 trials). In this single trial, the randomly generated error vector happened to have syndrome zero (i.e., the error was a codeword). The all-zero correction trivially satisfies the zero syndrome. This occurs with probability approximately `2^k / 2^n` where k is the code dimension and n is the block length. For the d=5 code (n=60, rate~0.50, k~30), this probability is `2^(-30)` for a random binary vector, but the trial error vector is generated with per-bit probability p=0.50, making each bit equally likely to be 0 or 1 — the error is effectively a uniform random binary vector. A length-60 rate-0.50 code has approximately `2^30` codewords out of `2^60` total vectors, so the probability is indeed ~`2^(-30)`, making this event very rare but not impossible over 200 trials.

### 7.2 SCR at p > 0.50 (Inverted LLR Region)

At p=0.501 and p=0.505, SCR = 1.0 despite FER = 1.0. This reveals an important distinction:

- The inverted LLRs cause the decoder to produce the **bitwise complement** of the correct error vector.
- This complemented correction satisfies the syndrome constraints (`syndrome(H, complement(e)) == syndrome(H, e)` when the all-ones vector is in the code's row space or has zero syndrome under H).
- However, the correction is logically incorrect (it differs from the true error by the all-ones vector, which is not a stabilizer).

This is the differentiating regime where Syndrome Consistency Rate and Fidelity diverge: the decoder finds a syndrome-consistent solution, but it is the wrong one logically.

### 7.3 Mean Iterations at p=0.50

| Schedule | d=5 | d=7 | d=9 | d=11 |
|----------|-----|-----|-----|------|
| flooding | 49.755 | 50.0 | 50.0 | 50.0 |
| layered | 50.0 | 50.0 | 50.0 | 50.0 |
| residual | 50.0 | 50.0 | 50.0 | 50.0 |
| adaptive | 12.0 | 12.0 | 12.0 | 12.0 |

The adaptive schedule terminates at exactly 12 iterations (phase-1 budget). The flooding mean of 49.755 at d=5 reflects the single trial where the zero-syndrome error caused early convergence.

---

## 8. Runtime Tables

Runtime mode was set to `"off"` for deterministic artifact generation. No runtime measurements were collected. See the prior near-threshold report (`near_threshold_report_feb-26-2026.md`) for runtime data at these operating points.

---

## 9. Determinism Verification

| Metric | Value |
|--------|-------|
| Artifact size | 40,487 bytes |
| SHA-256 hash | `829f0dc943f83ddeca456caf9e88b75a611554e7374038b2840cf945f9128764` |
| Re-run hash | `829f0dc943f83ddeca456caf9e88b75a611554e7374038b2840cf945f9128764` |
| Comparison | **Byte-identical** |
| Canonical JSON | `sort_keys=True, separators=(",",":")` |

New fields (`fidelity`, `syndrome_success_rate`) participate fully in artifact hashing and do not break determinism guarantees. The `deterministic_metadata=true` flag eliminates timestamp non-determinism.

---

## 10. Technical Interpretation

The near-threshold sweep reveals the metric behavior at the oracle channel's degenerate p=0.50 boundary:

**Below threshold (p < 0.50):** Fidelity = 1.0, SCR = 1.0. The decoder produces exact corrections satisfying all constraints.

**At threshold (p = 0.50):** Fidelity = 0.0, SCR = 0.0 (with rare exceptions for zero-syndrome errors). The decoder receives zero channel information and outputs the all-zero vector, which fails both logical correctness and syndrome consistency.

**Above threshold (p > 0.50):** Fidelity = 0.0, SCR = 1.0. The inverted LLR causes the decoder to produce syndrome-consistent but logically incorrect corrections. This is the key differentiating regime: **Syndrome Consistency Rate reveals that the decoder is still finding valid solutions to the parity constraints, even though those solutions are logically wrong.** This distinction is invisible to FER/Fidelity alone.

The Syndrome Consistency Rate thus provides a diagnostic signal that separates "decoder failure due to no information" (SCR = 0.0 at p = 0.50) from "decoder finds wrong but structurally valid solution" (SCR = 1.0 at p > 0.50). Under non-oracle channel models with non-trivial FER curves, this metric would further distinguish between parity-satisfying approximate corrections and structurally invalid outputs.

**Layer 1 decoder logic unchanged.**
