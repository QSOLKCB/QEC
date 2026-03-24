# Channel-Model Comparative Analysis — v3.2.0

**Project Version:** v3.2.0 (work-in-progress)
**Git Commit:** `f8f2ee28512d5d12058c6b77db979134c198ee28`
**Execution Environment:** Claude Code Compute
**Report Artifact:** `bench/channel_model_comparative_analysis_v3.2.0.md`
**Schema Version:** 3.0.1
**Seed:** 20260226

**Layer 1 decoder logic unchanged.**
**No decoder modifications. No schema changes. No channel code changes.**

---

## 0. Data Sources

This comparative analysis uses benchmark artifacts from two channel models:

### Oracle Baseline (v3.1.4)

| Source Report | Channel | Distances | p Region |
|-------------|---------|-----------|----------|
| `benchmark_report_v3.1.4_metrics.md` | oracle | d={3,5,7,9} | p={0.001–0.01} |
| `near_threshold_v3.1.4_metrics.md` | oracle | d={5,7,9,11} | p={0.49–0.505} |
| `industry_comparison_v3.1.4_metrics.md` | oracle | d={7,9,11} | p={0.01–0.05, 0.49–0.50} |
| `v3.1.4_oracle_diagnostic_analysis.md` | oracle | (diagnostic) | (all above) |

### Syndrome-Only (v3.2.0)

| Source Report | Channel | Distances | p Region | Trials |
|-------------|---------|-----------|----------|--------|
| `benchmark_report_v3.2.0_bsc_syndrome.md` | bsc_syndrome | d={3,5,7} | p_low={0.01–0.30} | 200 |
| `near_threshold_v3.2.0_bsc_syndrome.md` | bsc_syndrome | d={3,5,7} | p_near={0.45–0.55} | 500 |
| `industry_comparison_v3.2.0_bsc_syndrome.md` | bsc_syndrome | d={3,5,7} | p_low + p_near | 200/500 |

### Artifact Integrity

| Artifact | SHA-256 | Byte-Identical |
|----------|---------|----------------|
| Oracle baseline | `174cbf9093bea70fca07c0e68a0cfb00f1a53f5a73eae05a5bc11306bf1602fb` | Yes (v3.1.4 verified) |
| Oracle near-threshold | `829f0dc943f83ddeca456caf9e88b75a611554e7374038b2840cf945f9128764` | Yes (v3.1.4 verified) |
| Syndrome-only (p_low) | `0d4fc16bcdee53e05c05046daf9f880bf6e88926093ed2294b39a9c694891bd0` | Yes |
| Syndrome-only (p_near) | `657287091f5b2f6b09ffa8dc46f3f9befd755a85f15c989c5a9806178afc29ca` | Yes |

---

## 1. Threshold Displacement Estimate

### Definition

The effective threshold p* is defined as the maximum p value at which FER < 0.50 (majority-success decoding), computed per (schedule, distance).

**Oracle channel:** p*_oracle is effectively 0.4999+ for all (schedule, distance) combinations tested in v3.1.4. FER = 0.0 for all p < 0.50 due to oracle side-information providing exact error positions. The oracle "threshold" at p=0.50 is a degenerate singularity, not a true error-correction threshold.

**Syndrome-only channel:** p*_syndrome varies by schedule and distance:

| Schedule | d=3 p* | d=5 p* | d=7 p* |
|----------|--------|--------|--------|
| flooding | <= 0.01 | <= 0.01 | < 0.01 |
| layered | <= 0.02 | <= 0.01 | < 0.01 |
| hybrid_residual | <= 0.01 | <= 0.01 | < 0.01 |
| adaptive | <= 0.01 | <= 0.01 | < 0.01 |

### Threshold Displacement Δp

| Schedule | d=3 Δp | d=5 Δp | d=7 Δp |
|----------|--------|--------|--------|
| flooding | ~0.490 | ~0.490 | > 0.490 |
| layered | ~0.480 | ~0.490 | > 0.490 |
| hybrid_residual | ~0.490 | ~0.490 | > 0.490 |
| adaptive | ~0.490 | ~0.490 | > 0.490 |

The threshold displacement is approximately **Δp ≈ 0.48–0.49+** for all configurations. This quantifies the massive information advantage provided by oracle side-information: knowing which specific bits are in error (via LLR sign) is worth approximately 0.48 in threshold improvement.

**Important caveat:** The oracle "threshold" is not a true error-correction threshold but a degenerate artifact of the channel model providing exact error positions. The syndrome-only threshold represents the genuine decoding capability of the BP decoder operating on syndrome information alone. The Δp value measures the gap between "perfect side information" and "no side information," not between two genuine thresholds.

---

## 2. Comparative FER at Key Operating Points

### p = 0.01 (Common Operating Point)

| Schedule | d=3 Oracle FER | d=3 Syndrome FER | d=5 Oracle FER | d=5 Syndrome FER |
|----------|----------------|------------------|----------------|------------------|
| flooding | 0.000000 | 0.265000 | 0.000000 | 0.495000 |
| layered | 0.000000 | 0.295000 | 0.000000 | 0.455000 |
| adaptive | 0.000000 | 0.325000 | 0.000000 | 0.455000 |

At p=0.01, the oracle channel produces FER=0.0 (perfect decoding) while the syndrome-only channel produces FER ranging from 0.265 to 0.545 depending on schedule and distance.

### p = 0.50 (Oracle Critical Point)

| Metric | Oracle | Syndrome-Only |
|--------|--------|---------------|
| FER | 1.000 | 1.000 |
| Fidelity | 0.000 | 0.000 |
| SCR | ~0.000 | ~0.000 |
| mean_iters (flooding) | 50.0 | 50.0 |
| mean_iters (adaptive) | 12.0 | 12.0 |

At p=0.50, both channel models produce identical metrics: LLR is exactly zero in both cases (`log(1) = 0` for oracle; `log(1) = 0` uniform for syndrome-only). This is the sole point of convergence between the two channel models.

### p = 0.505 (Oracle Inverted Region)

| Metric | Oracle | Syndrome-Only |
|--------|--------|---------------|
| FER | 1.000 | 1.000 |
| Fidelity | 0.000 | 0.000 |
| SCR | **1.000** | **~0.000** |
| mean_iters (flooding) | 1.0 | 50.0 |
| mean_iters (adaptive) | 1.0 | 12.0 |

At p > 0.50, the channel models diverge dramatically:
- **Oracle:** SCR=1.0, mean_iters=1.0 — the decoder converges instantly to a syndrome-consistent but logically wrong correction (sign-inverted LLR provides strong but incorrect information).
- **Syndrome-only:** SCR~0.0, mean_iters=50 — the decoder exhausts iterations without finding any syndrome-consistent correction (uniform LLR provides no position information, only a weakening magnitude prior).

---

## 3. What Changed Structurally?

### 3.1 Oracle Channel — Regime Summary (from v3.1.4)

| Regime | p Range | FER | SCR | mean_iters | Character |
|--------|---------|-----|-----|------------|-----------|
| I — Stable | p < 0.50 | 0.000 | 1.000 | 1.0 | Perfect decoding via oracle LLR signs |
| II — Degenerate | p = 0.50 | 1.000 | ~0.000 | 12–50 | Zero information, no convergence |
| III — Inverted | p > 0.50 | 1.000 | 1.000 | 1.0 | Wrong but syndrome-consistent (LLR sign inversion) |

The oracle channel has three sharply delineated regimes with step-function transitions at p = 0.50.

### 3.2 Syndrome-Only Channel — Regime Summary (v3.2.0)

| Regime | p Range | FER | SCR | mean_iters | Character |
|--------|---------|-----|-----|------------|-----------|
| A — Partial Recovery | p <= ~0.02 | 0.26–0.84 | ~Fidelity | 4–42 | Genuine syndrome-based decoding |
| B — Transition | ~0.02 < p < ~0.10 | 0.80–1.00 | ~Fidelity | 39–50 | Degrading capability |
| C — Complete Failure | p >= ~0.10 | 1.000 | ~0.000 | 12/50 | No decoding capability |

The syndrome-only channel has a gradual degradation curve rather than sharp transitions. There is no inverted regime.

### 3.3 Key Structural Differences

| Property | Oracle | Syndrome-Only |
|----------|--------|---------------|
| Effective threshold (FER < 0.50) | p ~0.4999 | p ~0.01–0.02 |
| Threshold displacement Δp | (baseline) | ~0.48–0.49 |
| Regime count | 3 (Stable / Degen. / Inverted) | 3 (Partial / Transition / Failure) |
| Transition sharpness | Step function | Gradual curve |
| SCR/FER divergence | Yes (Regime III: SCR=1, FER=1) | **No** (SCR tracks FER) |
| Distance scaling | Positive (higher d = better or equal) | **Negative** (higher d = worse) |
| Schedule differentiation | None below threshold | **Meaningful** in Regime A |
| Inversion regime | Yes (p > 0.50) | **None** |
| Mean iters at p < threshold | 1.0 (single-pass) | 4–42 (genuine iteration) |
| FER at p=0.01 | 0.000 | 0.265–0.645 |

### 3.4 Interpretation

The comparative analysis reveals that the oracle channel model fundamentally obscures the decoder's genuine capability by providing an unrealistically strong side channel:

1. **The oracle's "threshold" is not a decoding threshold.** It is the point where the channel model stops providing the answer directly. The syndrome-only channel reveals the decoder's true threshold (~0.01–0.02), which is vastly lower.

2. **The oracle's inversion regime (Regime III) is an artifact of the channel model.** It does not reflect decoder capability — it reflects the mathematical structure of the oracle LLR (sign inversion produces syndrome-consistent complements for codes with even-weight checks). Under syndrome-only, this regime vanishes entirely.

3. **Schedule differentiation is real but masked by oracle.** Under oracle, all schedules produce identical results because the decoder converges in a single iteration. Under syndrome-only, schedule choice affects FER by 5–15 percentage points at low p, revealing genuine differences in iterative convergence behavior.

4. **Distance scaling inverts without oracle.** The oracle channel's distance-invariant perfect decoding hides the fact that larger QLDPC codes are harder to decode with min-sum BP when position information is unavailable. This is a genuine property of the decoder-code system, not an artifact of the channel model.

5. **SCR as a diagnostic tool.** Under oracle, SCR diverges from FER in Regime III, providing a useful diagnostic for "wrong but structurally valid" corrections. Under syndrome-only, SCR tracks FER closely, confirming that decoder failures are structurally complete (not finding syndrome-consistent wrong answers). Both behaviors are informative.

---

## 4. Determinism Confirmation

### Invariant Preservation Statement

| Invariant | Status |
|-----------|--------|
| Runner logic modified | **No** |
| Schema version changed | **No** (remains 3.0.1) |
| New metrics added | **No** |
| Decoder logic modified | **No** |
| Channel logic modified | **No** |
| New dependencies introduced | **No** |
| Canonicalization logic modified | **No** |
| Hashing logic modified | **No** |

### Determinism Audit Results

| Subset | p Values | Schedules | d | Trials | SHA-256 | Byte-Identical |
|--------|----------|-----------|---|--------|---------|----------------|
| p_low | [0.05, 0.2] | ['flooding', 'layered'] | 5 | 200 | `0dcff6e17888d26c4d802f54fdd868c5...` | **Yes** |
| p_near | [0.49, 0.5, 0.505] | ['flooding', 'layered'] | 5 | 500 | `b6ebc00ad2eb5a656505a20d04bf954d...` | **Yes** |

All determinism audits passed. Canonical JSON output is byte-identical across double runs with `runtime_mode="off"` and `deterministic_metadata=true`.

### Compliance

This report complies with all CLAUDE.md governance constraints:

- **Architectural layering:** Report-layer only (Layer 3). No modifications to Layer 1 (decoder) or Layer 2 (channel).
- **Determinism:** All stochastic operations seeded. Byte-identical reproduction verified.
- **Minimal diff:** New report files only. No existing files modified.
- **Decoder core protection:** No decoder files modified.
- **Schema governance:** Schema version 3.0.1 unchanged.
- **Dependency policy:** No new dependencies.

**Layer 1 decoder logic unchanged.**
