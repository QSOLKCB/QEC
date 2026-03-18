# QEC Invariant Registry

## QSOL-BP-INV-001: Sign Vector Purity

**Type:** Data (functional purity)

**Statement:**
`_sign(v)` is a pure, deterministic, side-effect-free function.
For any given float64 vector `v`, `_sign(v)` always returns the same
int-valued array where each element is -1 (if `v[i] < 0`) or +1 (otherwise).
The output depends only on the input vector's element values.

Consequently, `crc32(_sign(v).astype(int8).tobytes())` is also a pure
deterministic function of `v`.

**Exact validity condition:**
- The input LLR trace has been normalized to immutable 1-D float64 arrays
  via `_normalize_llr_trace`.
- No mutation occurs to the normalized trace between pre-computation and
  metric evaluation (enforced by setting `ndarray.flags.writeable = False`).
- The pre-computed sign and CRC arrays are indexed by absolute trace
  position, so each metric function takes its own window slice from the
  full-trace arrays. Window parameters (tail_window, gos_window,
  tsl_window, bti_window) may differ freely.

**What it allows:**
Pre-compute `_sign()` and `crc32(sign_bytes)` once per trace element in
`compute_bp_dynamics_metrics`, then pass the arrays internally to
`_compute_msi`, `_compute_cpi`, `_compute_tsl`, `_compute_gos`, and
`_compute_bti`. This eliminates redundant recomputation of identical
sign vectors across 5 metric functions.

**What would break it:**
- Mutation of the normalized LLR trace between pre-computation and use.
- A change to `_sign()` that introduces dependence on external state
  (e.g., iteration index, RNG, global config).
- A change to `_normalize_llr_vector` that makes its output non-deterministic.
- Passing pre-computed data from a different trace to a metric function
  (length assertion guards against this).

**Safety guards implemented:**
- `signs[i].flags.writeable = False` — prevents accidental mutation.
- `assert len(_crc_sigs) == n_iters` in CPI and BTI — ensures cached
  array matches the trace being processed.
- Fallback: all metric functions accept `_signs=None` / `_crc_sigs=None`
  and revert to inline computation when pre-computed data is absent.
