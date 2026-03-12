# QEC Repository Pre-v12.0.0 Structural Audit Report

**Date:** 2026-03-12
**Scope:** Full structural audit of QSOLKCB/QEC before v12.0.0 release
**Current version:** 9.1.0
**Auditor:** Claude Code (read-only review)

---

## 1. Repository Health Summary

The QEC repository is a substantial codebase (~73,000 lines across 321 Python files) implementing deterministic Tanner graph generation and QLDPC structure discovery. The architecture is generally well-structured with clear layer separation, consistent deterministic seeding via SHA-256 sub-seed derivation, and comprehensive test coverage (136 test files, ~42,000 lines).

**Overall status: HEALTHY with targeted issues to resolve.**

Key strengths:
- Core `src/qec/` packages use `np.random.RandomState(seed)` consistently (38 occurrences across 17 files)
- No use of `import random` (stdlib) in `src/qec/`
- No use of Python `hash()` for determinism-sensitive operations (SHA-256 used instead)
- Decoder core (Layer 1) has zero imports from higher layers
- Input immutability is consistently maintained via `.copy()` patterns
- Deterministic sorting is applied throughout analysis modules

Critical issues requiring attention before v12:
- 5 files construct dense non-backtracking matrices (violates CLAUDE.md Rule 9)
- Missing QLDPC commutativity constraint validation in all 18 mutation operators
- ACE repair operator has edge-count preservation bug
- 1 architectural layer violation (diagnostics → experiments import)
- 1 unseeded RNG in `src/qec_qldpc_codes.py`

---

## 2. Potential Bugs

### BUG-1: ACE Repair Operator — Edge Count Not Preserved (CRITICAL)

**File:** `src/qec/discovery/ace_repair.py:39-46`

```python
for i in range(m):
    for j in range(n):
        if H_new[i, j] == 1:
            degree = int(np.sum(H_new[:, j]))
            if degree < 2:
                H_new[i, j] = 0.0
                new_j = (j + 1) % n
                H_new[i, new_j] = 1.0
```

**Problem:** Three failure modes:
1. **Duplicate edge creation:** If `H_new[i, (j+1)%n]` is already `1.0`, setting it to `1.0` is a no-op but the original edge at `(i, j)` is removed — net edge count decreases.
2. **In-place mutation during iteration:** The loop mutates `H_new` while iterating, causing previously-rewired edges to be encountered and potentially rewired again.
3. **No target validation:** The `(j+1) % n` wrapping does not check if the target position already holds an edge.

**Impact:** Violates the invariant "edge count preserved by mutations."

### BUG-2: Silent Mutation Failures

**Files:** `src/qec/discovery/mutation_operators.py:149-152`, `src/qec/discovery/guided_mutations.py` (multiple locations)

Multiple mutation operators silently return unmodified matrices when rewiring is impossible (e.g., all positions in a check row are occupied). There is no diagnostic signal that a mutation was skipped, which could lead to stalled evolution in the population engine.

**Affected operators:** `edge_swap`, `spectral_edge_pressure_mutation`, `cycle_pressure_mutation`, `expansion_driven_rewire`, and others.

### BUG-3: Architectural Layer Violation — Diagnostics Imports Experiments

**File:** `src/qec/diagnostics/sensitivity_map.py:221-223`

```python
from src.qec.experiments.spectral_instability_phase_map import (
    compute_spectral_instability_score,
)
```

Diagnostics (Layer 3) imports from experiments (Layer 5). This violates the architectural layer model: "Lower layers must never import higher layers." The import is inside a function (lazy import), but it still creates a runtime dependency from a lower layer to a higher one.

---

## 3. Determinism Risks

### DET-1: Unseeded RNG in `src/qec_qldpc_codes.py` (HIGH)

**File:** `src/qec_qldpc_codes.py:604`

```python
rng = np.random.default_rng()  # NO SEED
```

This creates a non-deterministic RNG when the optional `rng` parameter is `None`. While this file appears to be outside the core `src/qec/` package, it could produce non-reproducible results.

### DET-2: `default_rng` Usage Outside Core (LOW)

**Files:** `src/bench/runner.py:145,200`, `src/bench/interop/runners.py:68`, `src/qldpc/protograph.py:103`, `src/simulation/fer.py:181`

These all use `np.random.default_rng(seed)` with explicit seeds — correctly seeded. The `src/qec/` core uses `np.random.RandomState(seed)` consistently. The mixed API usage (`default_rng` vs `RandomState`) is not a bug but a consistency concern. Both are deterministic with seeds.

### DET-3: Dense Eigensolver Determinism

**Files:** All 5 files using `np.linalg.eig` / `np.linalg.eigvals` on NB matrices

Dense eigensolvers are deterministic for the same input, so there is no correctness issue. However, the non-backtracking matrix is non-symmetric, and `np.linalg.eig` may produce eigenvector sign flips across platforms. The code uses lexicographic sorting by magnitude, real part, and imaginary part (deterministic tie-breaking), which mitigates this.

### DET-4: Core `src/qec/` Determinism — EXCELLENT

All 38 uses of `np.random.RandomState` in `src/qec/` are properly seeded via SHA-256 sub-seed derivation. No use of `import random`, no use of `hash()`, and sorted iteration is applied consistently in analysis modules.

---

## 4. Sparse Safety Risks

### SPARSE-1: Dense Non-Backtracking Matrix Construction (CRITICAL — Rule 9 Violation)

Five files construct the full NB matrix as a dense `|E|² × |E|²` array using `np.zeros((num_directed, num_directed))`, then compute eigenvalues via `np.linalg.eig(B)` or `np.linalg.eigvals(B)`:

| File | Lines | Operation |
|------|-------|-----------|
| `src/qec/diagnostics/non_backtracking_spectrum.py` | 85-96 | `B = np.zeros(...)` + `np.linalg.eigvals(B)` |
| `src/qec/diagnostics/nb_localization.py` | 147-155 | `B = np.zeros(...)` + `np.linalg.eig(B)` |
| `src/qec/diagnostics/sensitivity_map.py` | 93-100 | `B = np.zeros(...)` + `np.linalg.eig(B)` |
| `src/qec/experiments/spectral_graph_optimizer.py` | 96-103 | `B = np.zeros(...)` + `np.linalg.eig(B)` |
| `src/qec/experiments/tanner_graph_repair.py` | 746-792 | Multiple dense NB constructions |

**Memory impact:**
- 1,000 edges → 8 MB
- 10,000 edges → 800 MB
- 100,000 edges → 80 GB (catastrophic)

**CLAUDE.md Rule 9 states:** "Forbidden: dense Hashimoto matrix construction" and "Forbidden: numpy.linalg.eig on NB matrices."

**Note:** A correct sparse LinearOperator implementation already exists in `src/qec/diagnostics/_spectral_utils.py:90-146` (`build_nb_operator()` + `compute_nb_dominant_eigenpair()`). The 5 violating files should migrate to this pattern.

### SPARSE-2: Dense Bethe Hessian Construction (HIGH)

**File:** `src/qec/analysis/bethe_hessian.py:73-90`

```python
HtH = H_arr.T @ H_arr           # Dense n×n matrix
A = (HtH > 0).astype(np.float64)
H_B = r2_minus_1 * np.eye(n) - r * A + np.diag(degrees)
```

Constructs a dense `n × n` variable-node adjacency matrix via `H^T @ H`. For large QLDPC codes (n > 10,000), this creates an n² dense matrix before converting to sparse for `eigsh`.

### SPARSE-3: Dense Tanner Bipartite Adjacency (HIGH)

**Files:**
- `src/qec/diagnostics/tanner_spectral_analysis.py:49-51`
- `src/qec/diagnostics/bethe_hessian.py:67-69` (diagnostics copy)

These construct dense `(n+m) × (n+m)` bipartite adjacency matrices via block construction `[[0, H^T], [H, 0]]`.

### SPARSE-4: Safe Patterns (Positive Finding)

- `src/qec/diagnostics/_spectral_utils.py` — correct LinearOperator pattern
- `src/qec/diagnostics/spectral_metrics.py` — `safe_eigsh()` and `safe_eigs()` with controlled dense fallback for small matrices only
- All `.toarray()` calls in test files — acceptable for test verification

---

## 5. Mutation Safety Audit

### 5.1 Operator Inventory

**Total mutation operators: 18** across two registries plus local optimizer.

**Registry 1 — `mutation_operators.py:39-47` (7 operators):**
1. `edge_swap`
2. `local_rewire`
3. `cycle_break`
4. `degree_preserving_rotation`
5. `seeded_reconstruction`
6. `cycle_guided_mutation`
7. `spectral_pressure_guided_mutation`

**Registry 2 — `guided_mutations.py:57-69` (11 operators):**
1. `spectral_edge_pressure`
2. `cycle_pressure`
3. `ace_repair`
4. `girth_preserving_rewire`
5. `expansion_driven_rewire`
6. `ipr_trapping_pressure`
7. `trapping_set_pressure`
8. `residual_guided`
9. `absorbing_set_pressure`
10. `residual_cluster`
11. `spectral_localization`

**Local optimizer — `local_optimizer.py:140-146` (5 operators):**
1. `_absorbing_set_repair`
2. `_residual_hotspot_smoothing`
3. `_cycle_irregularity_reduction`
4. `_bethe_hessian_improvement`
5. `_residual_cluster_smoothing`

### 5.2 Invariant Verification

| Invariant | Status | Notes |
|-----------|--------|-------|
| Matrix shape preserved | PASS | All operators use `.copy()` and preserve `(m, n)` |
| Binary entries | PASS | All operators assign only `0.0` or `1.0` |
| No all-zero rows | PASS | Checked before edge removal |
| No all-zero columns | PASS | Checked before edge removal |
| Edge count preserved | **FAIL** | `ace_repair.py` can lose edges (see BUG-1) |
| Variable degree preserved | PARTIAL | `degree_preserving_rotation` is correct; other operators may alter degrees |
| Check degree preserved | PARTIAL | Same as above |
| No duplicate edges | **FAIL** | `ace_repair.py` can create duplicate edges |
| Input immutability | PASS | All operators call `.copy()` at entry |
| Deterministic execution | PASS | All use `RandomState(seed)` |
| **QLDPC commutativity (H_X @ H_Z^T = 0)** | **FAIL** | **NOT CHECKED IN ANY OPERATOR** |

### 5.3 Missing QLDPC Commutativity Validation (CRITICAL)

**CLAUDE.md Section 10 states:**
> "QLDPC graphs obey strict commutativity constraints: H_X H_Z^T = 0"
> "Edge swaps must be rejected if they break stabilizer commutativity."

**Current state:** None of the 18 mutation operators, the repair pipeline (`repair_operators.py:168-209`), or the validation function (`validate_tanner_graph()`) checks this constraint. This is the most significant structural gap in the mutation subsystem.

---

## 6. Operator Registry Check

### 6.1 Dual Registry Issue

Two separate `_OPERATORS` lists exist:
- `mutation_operators.py:39-47` — 7 operators (used by `discovery_engine.py`)
- `guided_mutations.py:57-69` — 11 operators (used by `population_engine.py`)

The `population_engine.py` imports `_OPERATORS` from `guided_mutations.py` (11 items) and schedules via `operator_idx = (self._generation + i) % len(operators)` (modulo 11).

The `discovery_engine.py` uses `get_operator_for_generation()` from `mutation_operators.py` (modulo 7).

Both derive scheduling dynamically from `len(operators)` — no hardcoded counts. This is correct behavior per the audit requirements.

### 6.2 Operator Function Mapping Verification

**`mutation_operators.py` _OPERATOR_FUNCTIONS dict:**
All 7 string names map to implemented functions. No dead operators.

**`guided_mutations.py` _OPERATOR_FUNCTIONS dict:**
All 11 string names map to implemented functions. No dead operators.

**`guided_mutations.py` OPERATORS list (function references):**
All 11 function objects match their string counterparts.

### 6.3 Unreachable Functions

No functions were found that are defined but unreachable from either registry. All operator functions are registered.

### 6.4 Final Operator Lists

**mutation_operators.py (7):**
```
edge_swap, local_rewire, cycle_break, degree_preserving_rotation,
seeded_reconstruction, cycle_guided_mutation, spectral_pressure_guided_mutation
```

**guided_mutations.py (11):**
```
spectral_edge_pressure, cycle_pressure, ace_repair, girth_preserving_rewire,
expansion_driven_rewire, ipr_trapping_pressure, trapping_set_pressure,
residual_guided, absorbing_set_pressure, residual_cluster, spectral_localization
```

---

## 7. Test Coverage Gaps

### 7.1 Existing Coverage (Strong)

- Mutation determinism: `test_mutation_operators.py`, `test_guided_mutations.py` — verify same seed → same result
- Mutation invariants: `test_mutation_operators.py` — verifies shape, edge count, degree preservation
- Repair operators: `test_repair_operators.py` — validates repair pipeline
- ACE repair: `test_ace_repair.py` — tests healthy graph unchanged
- Local optimizer: `test_local_optimizer.py` — tests determinism and shape preservation
- Discovery engine: `test_discovery_engine.py`, `test_population_discovery_engine.py`
- Analysis modules: individual test files for each analyzer

### 7.2 Missing Coverage (Gaps)

| Gap | Impact | Priority |
|-----|--------|----------|
| No test for ACE repair with degree-1 columns that would trigger the edge-count bug | Masks BUG-1 | HIGH |
| No test for QLDPC commutativity preservation after mutation | Masks missing validation | HIGH |
| No test for mutation operators with fully-connected check rows (silent failure case) | Masks no-op mutations | MEDIUM |
| No integration test verifying population engine + discovery engine use consistent operator schedules | Masks registry drift | MEDIUM |
| No large-graph memory tests for NB spectrum computation | Would catch sparse safety violations | MEDIUM |
| No test for `sensitivity_map.py`'s lazy import from experiments layer | Would catch layer violation | LOW |

---

## 8. Performance Risks

### PERF-1: Dense NB Matrix — O(|E|²) Memory and O(|E|³) Compute (CRITICAL)

**Files:** 5 files listed in Section 4 (SPARSE-1)

The dense `np.linalg.eig(B)` on the NB matrix has O(|E|³) time complexity. For codes with 10,000+ edges, this becomes impractical. The sparse LinearOperator + `scipy.sparse.linalg.eigs(k=1)` approach in `_spectral_utils.py` has O(|E|) memory and O(k·|E|) time per Krylov iteration.

### PERF-2: Dense Bethe Hessian — O(n²) Memory

**File:** `src/qec/analysis/bethe_hessian.py:73`

The `H^T @ H` construction creates an `n × n` dense matrix. For sparse LDPC codes, the adjacency can be constructed in O(|E|) by iterating over edges directly.

### PERF-3: Quadratic Edge Iteration in Directed Edge Construction

**Files:** `non_backtracking_spectrum.py:54-60`, `nb_localization.py:107-113`

```python
for ci in range(m):
    for vi in range(n):
        if H[ci, vi] != 0:
```

This iterates over all `m × n` entries of H even when H is sparse. For large sparse codes, this should iterate over nonzero entries only (e.g., using `np.argwhere` or sparse matrix `.nonzero()`).

### PERF-4: Power Iteration in Guided Mutations

**File:** `src/qec/discovery/guided_mutations.py:158-173`

The `spectral_edge_pressure_mutation` uses 50 iterations of power iteration on `H^T @ H` to approximate the dominant eigenvector. This is O(50 · |E|) per mutation, which is acceptable but could accumulate over many generations.

---

## 9. Recommended Fixes

Priority ordering for pre-v12.0.0:

### P0 — Must Fix

1. **Fix ACE repair edge-count bug** (`ace_repair.py:39-46`): Check if target position `(i, new_j)` already has an edge before rewiring. Track and validate edge count after repair.

2. **Migrate 5 dense NB matrix files to sparse LinearOperator pattern**: Use the existing `_spectral_utils.build_nb_operator()` + `compute_nb_dominant_eigenpair()` as reference. Files: `non_backtracking_spectrum.py`, `nb_localization.py`, `sensitivity_map.py`, `spectral_graph_optimizer.py`, `tanner_graph_repair.py`.

3. **Add QLDPC commutativity validation to `validate_tanner_graph()`**: Add an optional `H_z` parameter to check `H_x @ H_z.T == 0` when QLDPC pairs are available.

### P1 — Should Fix

4. **Fix architectural layer violation**: Move `compute_spectral_instability_score` from `experiments/spectral_instability_phase_map.py` to a lower layer (diagnostics or analysis), or refactor `sensitivity_map.py` to not depend on it.

5. **Seed the unseeded RNG** in `src/qec_qldpc_codes.py:604`: Add `seed` parameter or require explicit RNG.

6. **Refactor Bethe Hessian to sparse construction** (`analysis/bethe_hessian.py:73`): Build adjacency from edge enumeration instead of dense `H^T @ H`.

### P2 — Nice to Have

7. **Add mutation failure diagnostics**: Return a signal (e.g., `mutation_applied: bool`) when operators fall back to no-op.

8. **Add edge-iteration optimization**: Replace `m × n` loops with `.nonzero()` iteration in NB matrix construction.

9. **Add integration test**: Verify population engine and discovery engine operator registries are aligned or intentionally different.

10. **Add test for ACE repair on graphs with degree-1 columns**: Exercise the edge-count bug fix.

---

## Appendix: Invariant Checklist

| # | Invariant | Status | Evidence |
|---|-----------|--------|----------|
| 1 | Parity-check matrix shape preserved | PASS | All operators copy and return same shape |
| 2 | Edge count preserved by mutations | **FAIL** | `ace_repair.py` can lose edges |
| 3 | Degree constraints preserved | PARTIAL | Only `degree_preserving_rotation` guarantees this |
| 4 | No duplicate edges introduced | **FAIL** | `ace_repair.py` can create duplicates |
| 5 | Mutation operators do not mutate inputs in-place | PASS | All use `.copy()` |
| 6 | Same seed → identical results | PASS | `RandomState(seed)` used throughout |
| 7 | QLDPC commutativity (H_X H_Z^T = 0) | **NOT CHECKED** | No operator validates this |
| 8 | No dense Hashimoto construction | **FAIL** | 5 files construct dense NB matrices |
| 9 | Memory scales with \|E\|, not \|E\|² | **FAIL** | Dense NB and Bethe Hessian constructions |
| 10 | Decoder core untouched by diagnostics | PASS | No imports from higher layers into decoder |
| 11 | Diagnostics are side-effect free | PASS | All analysis modules preserve inputs |
| 12 | Architectural layering respected | **FAIL** | `sensitivity_map.py` imports from experiments |
