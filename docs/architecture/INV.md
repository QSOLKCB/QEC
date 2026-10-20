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

**Version:** v68.6.1 (hardened, formally proven)

**Depends on:** INV-001 (sign vector purity)

### Formal Definition

Let `F = compute_bp_dynamics_metrics` be the public API function with
input tuple `x = (llr_trace, energy_trace, correction_vectors, params)`.

Define `cache_key(x)` as the content-based key constructed by
`_make_cache_key(x)`: the tuple of concatenated raw float64 bytes of
all input LLR arrays, `struct.pack`-ed energy bytes, correction vector
bytes (or `None`), and the sorted effective parameter tuple.  See
**Cache Key Construction** below for the precise definition.

**Invariant statement (INV-003):**

```
For any inputs x and y, if the cache-key byte encoding of x equals
that of y, then F(x) = F(y) under the current implementation.
```

More precisely: if `cache_key(x) == cache_key(y)` then
`json.dumps(F(x), sort_keys=True) == json.dumps(F(y), sort_keys=True)`.

**Consequence:** A module-level content-addressed cache can store
results keyed by `cache_key(x)`. Cache hits return a deep copy of the
stored result, eliminating redundant computation across calls within
the same process. Caller mutation of returned dicts cannot corrupt
the cache because both storage and retrieval use `copy.deepcopy`.

### Scope

This invariant applies under the following assumptions:

1. **Current implementation only.** The invariant holds for the
   implementation in `src/qec/diagnostics/bp_dynamics.py` as of
   v68.6.1. Changes to internal metric functions may require
   re-validation.

2. **Function purity.** `F` is composed entirely of pure deterministic
   transformations (proven below). No RNG, no `hash()`, no global
   mutable state reads.

3. **IEEE 754 float64 determinism.** All arithmetic uses NumPy float64
   operations on the same platform. IEEE 754 guarantees bit-identical
   results for identical inputs under identical rounding mode.
   Cross-platform reproducibility is not claimed.

### Validity Conditions

All of the following must hold for reuse to be safe:

1. **Input immutability.** Inputs must not be mutated between the
   original computation and cache lookup. The cache key is derived
   from input *content* at call time via byte snapshot.
   *Enforced by:* content-based keying (bytes snapshot at call time).

2. **Function purity.** `F` must remain a pure function: no global
   state reads, no RNG, no `hash()`, no system calls that vary
   across invocations.
   *Verified by:* TestDeterminism, TestCrossCallReuse, formal proof.

3. **Cache immutability.** Cached results must not be corrupted by
   caller mutation.
   *Enforced by:* `copy.deepcopy` on both cache store and cache
   retrieval. The cached object is never exposed to callers.
   *Verified by:* TestCacheMutationSafety.

4. **No dependency on call ordering.** Cache correctness must not
   depend on which call occurs first.
   *Enforced by:* content-addressed keys (order-independent).

5. **No hidden global state.** The function must not read module-level
   mutable state (other than the cache itself) that could change
   between calls.
   *Verified by:* code inspection — only `DEFAULT_PARAMS` and
   `DEFAULT_THRESHOLDS` are read, both are module constants.

### Cache Key Construction

The cache key is a deterministic content tuple:

```
key = (
    llr_bytes,           # concatenated .tobytes() of input LLR arrays
    energy_bytes,        # struct.pack(">Nd", *energy) — big-endian float64
    cv_bytes_or_None,    # correction vector bytes or None sentinel
    frozen_params,       # tuple(sorted(effective_params.items()))
)
```

All components are derived from input *content*, not object identity.
No use of Python `hash()` (salted per process; forbidden by CLAUDE.md).
No use of `id()` (memory-layout dependent).

### Proof of Correctness

**Theorem.** For any inputs `x, y` to `compute_bp_dynamics_metrics`:
if `cache_key(x) == cache_key(y)` then `F(x) == F(y)`.

**Proof.**

**Step 1. Determinism — no non-deterministic primitives.**

Enumerate all operations in `F`:
- `np.asarray(x, dtype=float64)` — deterministic type coercion
- `np.squeeze`, `reshape`, indexing — deterministic array ops
- `np.where(x < 0, -1, 1)` — element-wise comparison on float64,
  deterministic per IEEE 754 (INV-001, proven by TestPurityProof)
- `zlib.crc32(bytes)` — deterministic per zlib spec
- `np.diff`, `np.mean`, `np.var`, `np.sum`, `np.clip`, `np.median`,
  `np.abs`, `np.log`, `np.linalg.norm` — all deterministic for
  identical float64 inputs under fixed rounding mode
- `struct.pack` — deterministic byte encoding
- `float()`, `int()` — deterministic coercion
- `sorted()`, `set()`, `dict()` — deterministic for hashable keys
- No `random`, no `hash()`, no `os`/`sys` calls, no `datetime`

Therefore `F` uses only deterministic primitives. ∎

**Step 2. Byte-level equivalence — identical bytes → identical
normalized trace.**

If `cache_key(x) == cache_key(y)`, then by construction of
`_make_cache_key` each component is identical:
- `llr_bytes(x) == llr_bytes(y)` — identical raw float64 bytes
- `energy_bytes(x) == energy_bytes(y)` — identical packed floats
- `cv_bytes(x) == cv_bytes(y)` — identical correction vector bytes
- `params(x) == params(y)` — identical parameter tuples

Since `_normalize_llr_vector` converts inputs to float64 via
`np.asarray(x, dtype=float64)`, and `_make_cache_key` reads the
same raw bytes via `np.asarray(x, dtype=float64).ravel().tobytes()`,
byte-identical keys guarantee that `_normalize_llr_trace` produces
identical normalized arrays.

Therefore: `cache_key(x) == cache_key(y)` → identical internal state. ∎

**Step 3. Functional composition — pure steps compose purely.**

`F` is composed as:
```
F(x) = classify(metrics(normalize(x), precompute(normalize(x)), params(x)))
```

Where:
- `normalize` = `_normalize_llr_trace` (pure: array ops only)
- `precompute` = `_precompute_signs_and_sigs` (pure: INV-001)
- `metrics` = composition of `_compute_{msi,cpi,tsl,lec,cvne,gos,eds,bti}`
  (each pure: arithmetic + array ops only)
- `classify` = `classify_bp_regime` (pure: comparisons + dict construction)

A composition of pure deterministic functions is pure and deterministic.
Therefore: identical internal state → identical output. ∎

**Step 4. Cache correctness — stored value equals computed value.**

On cache miss: `F(x)` is computed, `copy.deepcopy(result)` is stored
in `_CROSS_CALL_CACHE[cache_key(x)]`. The caller receives the original
`result` object.

On cache hit for `cache_key(y) == cache_key(x)`: `copy.deepcopy` of the stored
value is returned. By Steps 1–3, the stored value equals `F(x)`, and
`F(x) == F(y)`. The deep copy is structurally identical to the stored
value. Therefore the returned value equals `F(y)`. ∎

**Step 5. Mutation safety — cached value cannot be corrupted.**

- On store: `copy.deepcopy(result)` creates an independent copy.
  The caller's reference to `result` cannot reach the cached copy.
- On retrieval: `copy.deepcopy(cached)` creates an independent copy.
  The caller's reference cannot reach the cached copy.

Therefore: no external mutation path to cached data exists. ∎

**Conclusion:** The cache correctly returns `F(y)` for any `y` where
`cache_key(y) == cache_key(x)` for some previously computed `x`, and no
external mutation can corrupt cached values. Reuse is mathematically
safe. ∎

### Instrumentation

Module-level counters track cache performance:

```
_CACHE_HITS: int    # incremented on each cache hit
_CACHE_MISSES: int  # incremented on each cache store (miss → compute)
```

These are informational only and do not affect behavior.

### What Would Break This Invariant

- Making `F` read global mutable state (breaks Step 1).
- Introducing RNG or non-deterministic behavior in any metric (breaks Step 1).
- Removing `copy.deepcopy` from cache store or retrieval (breaks Step 5).
- Changing `_normalize_llr_vector` to be non-deterministic (breaks Step 2).
- Using Python `hash()` for cache keys (salted, non-deterministic per process).
- Platform change affecting IEEE 754 rounding (breaks Step 1 assumption).

### Limitations

- **Per-process only.** Cache is module-level, not cross-process.
  Cleared on import. No persistence.
- **Unbounded memory growth.** Memory grows with number of distinct
  input patterns. Suitable for test suites with bounded distinct
  inputs, not unbounded production workloads.
- **No cross-worker sharing.** Cache does not persist across pytest
  worker processes (no issue for single-process test runs).
- **Deep copy overhead.** Each cache hit incurs `copy.deepcopy` cost.
  Acceptable because the output is a small nested dict of scalars
  (~21 float/int/str/None values + regime classification dict).
- **IEEE 754 scope.** Cross-platform bit-identical results are not
  guaranteed if floating-point rounding modes differ.

---

## QSOL-PHI-INV-004: PHI_SCALE_NODE

**Type:** Structural (golden-ratio shell quantization)

**Version:** v137.0.13

**Invariant statement:** All raster depth spans must quantize to the
canonical φ-shell progression `(1.0, 1.618, 2.618, 4.236, 6.854)`.
Linear z-bands are forbidden. Shell boundaries are monotonically
increasing and each successive shell equals the sum of the two
preceding values (golden recurrence). Quantization is deterministic:
identical depth inputs always map to the same shell.

---

## QSOL-E8-INV-005: E8_TRIALITY_LOCK

**Type:** Structural (E8 triality constraint)

**Version:** v137.0.13

**Invariant statement:** The visibility classification system must
enforce exactly three primary shell classes (NEAR_SHELL, MID_SHELL,
OUTER_SHELL) plus two boundary classes (RESONANCE_NODE, WIGGLE_ZONE).
The triality of near/mid/outer mirrors the E8 triality structure from
the theory corpus. Classification boundaries are fixed and
deterministic.

---

## QSOL-OURO-INV-006: OUROBOROS_FEEDBACK_LOOP

**Type:** Computational (self-consistent restore operator)

**Version:** v137.0.13

**Invariant statement:** The UFF restore operator
`∇²T + (φ + ψ)² T = 0` is implemented as a deterministic
span-energy correction via `compute_phi_restore_term`. The restore
term is a pure function of span_energy and phase_offset with no
hidden state. For fixed inputs, the output is byte-identical across
all invocations.

---

## QSOL-SIS2-INV-007: SIS2_STABILITY_RING

**Type:** Structural (ledger stability)

**Version:** v137.0.13

**Invariant statement:** The raster ledger is a frozen, immutable
artifact. Once constructed, no field may be mutated. The ledger
stable_hash is computed from canonical JSON of all constituent
decision hashes. 100-run replay of identical inputs must produce
byte-identical ledger exports (JSON + SHA-256).
