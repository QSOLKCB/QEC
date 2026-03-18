# QEC Invariant Registry

## QSOL-BP-INV-001: Sign Vector Purity

**Type:** Data (functional purity)

**Version:** v68.5.1 (formally validated)

### Formal Definition

Let `T = [v_0, v_1, ..., v_{N-1}]` be a normalized LLR trace where each
`v_i` is a 1-D float64 array produced by `_normalize_llr_trace`.

Define:
```
sign(v) := np.where(v < 0, -1, 1)
crc(v)  := zlib.crc32(sign(v).astype(int8).tobytes()) & 0xFFFFFFFF
```

**Invariant statement:** For any fixed `v`, `sign(v)` and `crc(v)` are
pure deterministic functions: they depend only on the element values of
`v`, produce no side effects, read no external state, and return
identical results on every invocation.

**Consequence:** Pre-computing `signs[i] = sign(v_i)` and
`crc_sigs[i] = crc(v_i)` once for all `i in [0, N)` yields arrays that
can be shared by any metric function that would otherwise compute
`sign(v_i)` or `crc(v_i)` inline, without any change in output.

### Validity Conditions

All of the following must hold:

1. **Trace immutability.** The normalized LLR trace `T` is not mutated
   between pre-computation and metric evaluation.
   *Enforced by:* `signs[i].flags.writeable = False`.

2. **Index alignment.** Pre-computed arrays are indexed by absolute trace
   position `[0, N)`. Each metric slices its own window from these arrays
   independently.
   *No assumption of equal window sizes is required.*

3. **Length consistency.** `len(signs) == len(crc_sigs) == len(T) == N`.
   *Enforced by:* assertions in `_compute_cpi` and `_compute_bti`.

4. **Purity of `_sign`.** The function `_sign(x) = np.where(x < 0, -1, 1)`
   depends only on its argument `x`.
   *Verified by:* TestPurityProof test suite.

### Proof Sketch

**1. Purity.**
`_sign(x) = np.where(x < 0, -1, 1)`. The comparison `x < 0` is
element-wise on float64, deterministic. `np.where` selects from
constants `-1` and `1`. No global state is read. No mutation occurs.
Therefore `_sign` is a pure function.

`zlib.crc32(bytes)` is a deterministic hash function per CPython/zlib
specification, producing identical output for identical byte input.

`ndarray.astype(int8).tobytes()` produces bytes in C-contiguous order,
deterministic for identical array values.

**2. Slice equivalence.**
Each metric's original code slices `llr_trace[-w:]` and computes
`_sign(vec)` inline. The optimized code indexes `signs[abs_idx]` where
`abs_idx = N - w + t` for the same iteration variable `t`. Since
`signs[i] = _sign(normed_llr[i])` by construction, and `normed_llr[-w:][j]
== normed_llr[N - w + j]`, the values are identical.

Proven for all 5 metrics (MSI, CPI, TSL, GOS, BTI) — see
TestSliceLevelEquivalence.

**3. Byte equivalence.**
`signs[i].astype(int8).tobytes() == _sign(normed_llr[i]).astype(int8).tobytes()`
because `signs[i]` and `_sign(normed_llr[i])` have identical values, dtype,
and shape. Therefore CRC32 outputs are identical.

Proven — see TestByteLevelEquivalence.

**4. Zero mutation.**
`signs[i].flags.writeable = False` prevents any write. CRC signature
list elements are Python `int` (immutable). Metric functions only read
via comparison operators and list slicing.

Proven — see TestZeroMutationGuarantee.

### Measured Redundancy Elimination

With default parameters (`tail_window=gos_window=tsl_window=bti_window=12`)
and N=20 trace elements:

| Path      | _sign() calls | Breakdown                              |
|-----------|---------------|----------------------------------------|
| Uncached  | 81            | MSI=22, CPI=12, TSL=13, GOS=22, BTI=12 |
| Cached    | 20            | precompute=20, metrics=0               |

**Reduction: 81 → 20 calls (75.3%)**

Verified via deterministic monkeypatch counting — see
TestRedundancyElimination.

### What Would Break This Invariant

- Mutation of `normed_llr[i]` between pre-computation and metric use.
- A change to `_sign()` that introduces state dependence (e.g., iteration
  index, RNG, global config, or non-float64 type handling).
- A change to `_normalize_llr_vector` that makes output non-deterministic.
- Passing pre-computed arrays from a different trace (guarded by length
  assertions).
- A change to `np.where` semantics for float64 comparison (would require
  numpy API break).

### Limitations

- Optimization scope is per-call: no cross-call caching between separate
  invocations of `compute_bp_dynamics_metrics`.
- Memory overhead: O(N * n_vars) for sign arrays + O(N) for CRC ints.
  Negligible relative to the LLR trace itself.
- The invariant applies only to the 5 metrics that use `_sign()` or CRC32.
  LEC, CVNE, and EDS are unaffected (they do not use sign vectors).
