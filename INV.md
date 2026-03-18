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
  a NumPy API break).

### Limitations

- Optimization scope is per-call: no cross-call caching between separate
  invocations of `compute_bp_dynamics_metrics`.
- Memory overhead: O(N * n_vars) for sign arrays + O(N) for CRC ints.
  Negligible relative to the LLR trace itself.
- The invariant applies only to the 5 metrics that use `_sign()` or CRC32.
  LEC, CVNE, and EDS are unaffected (they do not use sign vectors).

---

## QSOL-BP-INV-003: Cross-Call Deterministic Reuse

**Type:** Computational (cross-call memoization)

**Version:** v68.6.0 (formally validated)

**Depends on:** INV-001 (sign vector purity)

### Formal Definition

Let `F = compute_bp_dynamics_metrics` with inputs
`(llr_trace, energy_trace, correction_vectors, params)`.

`F` is a **pure deterministic function**: it reads no global state,
produces no side effects, and its output depends only on the content
of its arguments.

**Invariant statement:** For any two calls `F(A)` and `F(B)`, if
`content(A) == content(B)` (byte-identical input content), then
`F(A) == F(B)` (identical output). Therefore, the result of a prior
call can be reused for any subsequent call with content-identical
inputs, without any change in behavior or output.

**Consequence:** A module-level content-addressed cache can store
results keyed by a deterministic hash of input content. Cache hits
return the stored result, eliminating redundant computation across
calls within the same process.

### Validity Conditions

All of the following must hold for reuse to be safe:

1. **Input immutability.** Inputs must not be mutated between the
   original computation and cache lookup. The cache key is derived
   from input *content* at call time, so mutation after keying would
   break the invariant.
   *Enforced by:* content-based keying (bytes snapshot at call time).

2. **Function purity.** `compute_bp_dynamics_metrics` must remain a
   pure function: no global state reads, no RNG, no `hash()`, no
   system calls that vary across invocations.
   *Verified by:* TestDeterminism, TestCrossCallReuse.

3. **No mutation of cached outputs.** Callers must not mutate returned
   dicts in ways that corrupt cached references. The cache returns
   the stored dict directly; if callers mutate it, subsequent cache
   hits would return corrupted data.
   *Mitigated by:* mutation safety test (TestCacheMutationSafety).
   The output is a plain dict of Python scalars (float/int/str/None)
   and nested dicts, which are rarely mutated in practice. Deep copy
   is available but not applied by default to avoid overhead.

4. **No dependency on test ordering.** Cache correctness must not
   depend on which test runs first or in what order.
   *Enforced by:* content-addressed keys (order-independent).

5. **No hidden global state.** The function must not read module-level
   mutable state (other than the cache itself) that could change
   between calls.
   *Verified by:* code inspection — only `DEFAULT_PARAMS` and
   `DEFAULT_THRESHOLDS` are read, both are module constants.

### Cache Key Construction

The cache key is a deterministic content hash:

```
key = (
    llr_bytes,           # concatenated .tobytes() of normalized LLR arrays
    energy_bytes,        # packed float64 bytes of energy trace
    cv_bytes_or_None,    # correction vector bytes or None sentinel
    frozen_params,       # tuple(sorted(effective_params.items()))
)
```

All components are derived from input *content*, not object identity.
No use of Python `hash()` (salted per process, forbidden by CLAUDE.md).
No use of `id()` (memory-layout dependent).

### What Would Break This Invariant

- Making `compute_bp_dynamics_metrics` read global mutable state.
- Introducing RNG or non-deterministic behavior in any metric.
- Mutating cached output dicts (corrupts future cache hits).
- Changing `_normalize_llr_vector` to be non-deterministic.
- Using Python `hash()` for cache keys (salted, non-deterministic).

### Limitations

- Cache is module-level, not cross-process. Cleared on import.
- Memory grows with number of distinct input patterns. Suitable for
  test suites with bounded distinct inputs, not unbounded production.
- Cache does not persist across pytest worker processes (no issue for
  single-process test runs).
- Correction vectors with mutable numpy arrays are snapshotted at
  call time; subsequent mutation of the original arrays does not
  invalidate the cache entry (by design: key was correct at call time).
