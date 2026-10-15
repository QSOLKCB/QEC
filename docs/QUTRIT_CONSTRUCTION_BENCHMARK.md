# Exact qutrit constructor benchmark

This opt-in, standard-library benchmark compares the live `ExactDecoder`
constructor with an independent copy of its original scalar constructor. It
measures construction overhead, including preparation, without changing code
parameters, error sets, syndrome ordering, coset-leader selection, or correction
semantics. It is not a simulation of a physical device.

## Run

From the repository root, using Python 3.10 or newer:

```sh
python scripts/benchmark_qutrit_construction.py > qutrit-construction.json
```

The script also works from another directory when invoked by its full path; it
uses this checkout's `src` rather than an installed `qec` package. The default is
three latency samples and one separate memory sample per workload, with two
repeats of each correction batch. Expect seconds to tens of seconds on a typical
host, rather than a large simulation campaign. Increase sampling for a local
performance investigation:

```sh
python scripts/benchmark_qutrit_construction.py \
  --samples 9 --memory-samples 3 --correction-repeats 10 > qutrit-construction.json
```

All counts must be positive integers. JSON is the only stdout output. A parity
failure raises an exception before any timing begins; there are no timing
thresholds or speedup requirements. This is a manual benchmark, not a timing test
in the normal test suite. Save new results separately; do not replace historical
fixtures with machine-dependent benchmark observations.

## Reference and correctness gate

`ScalarReferenceDecoder` is a frozen dataclass subclass of `ExactDecoder`. It
replaces only `__post_init__` with the original scalar builder, including the
original negative-weight check, candidate enumeration, collision validation,
first-representative policy, and `MappingProxyType` table. It does not invoke the
live constructor or its optimized helper. It inherits the unchanged `decode`
and `correct` methods, with the same code, max-weight, result and immutable-table
API. This deliberately isolates constructor differences; it is not a second
independent implementation of all stabilizer algebra.

The original source was the pinned pre-optimization
`qec/decoder/qutrit/exact.py`, with SHA-256:

```text
11b5cad38e88fad36dc37a9ab916496b17b75eb3d56df765ce90dd2e4153bfce
```

The embedded scalar constructor's AST was checked against that original. The
script is standalone and does not require that historical checkout at runtime.
Every report includes the historical source hash, a hash of the embedded builder
source, and hashes of the actual benchmark, live decoder, code factories,
stabilizer algebra, and GF(3) source files used. These hashes identify source
bytes, not a claim that measurements repeat bit-for-bit across machines.

Before **any** measurements, the script builds all code objects and verifies:

- Identical ordered `(syndrome, representative)` table items, not merely equal
  unordered mappings.
- A read-only live table.
- Identical complete `DecodeResult` values and successful correction for every
  error in each case's bounded-weight set.

Cases and the number of verified errors are:

| Case | n | k | max_weight | Errors |
| --- | ---: | ---: | ---: | ---: |
| Cyclic five | 5 | 1 | 1 | 41 |
| Shor nine | 9 | 1 | 1 | 73 |
| Ternary Golay eleven | 11 | 1 | 2 | 3,609 |
| Two-qutrit Z-generator control | 2 | 0 | 2 | 81 |
| Cyclic five, identity-only control | 5 | 1 | 0 | 1 |

The Shor and k=0 cases exercise degenerate errors and collision checks. The
zero-weight case verifies the path that does not prepare a plan. These gates do
not replace dedicated tests of invalid inputs, over-radius errors, or custom
subclass behavior.

## What each measurement includes

### Full constructor

Each sample constructs a fresh live or scalar decoder with an existing, already
validated code object. The live full-constructor sample includes its entire
one-site syndrome preparation, candidate creation/enumeration, sparse syndrome
calculation, collision validation and table creation. The original scalar
constructor includes the same work except that it calculates each syndrome via
`code.syndrome(candidate)` without a prepared plan.

Construction of code objects, error generation, correctness verification,
imports, and explicit pre-sample garbage collection are excluded. Normal Python
cyclic garbage collection remains enabled according to the reported environment.
The returned decoder remains alive until after the timer stops, so decoder
teardown is excluded. Implementation order alternates between samples to reduce
systematic first/second bias.

“Cold” means a fresh instance and fresh plan on every call. It does **not** mean a
fresh Python process, empty CPU caches, or cold imports: correctness gates have
already exercised both implementations. There is no cross-construction cache.

### Cold plan preparation

For each positive-radius case the script separately measures
`_one_site_syndromes(code)` with a fresh result each call. This is a component of
the live full-constructor measurement, not an additional cost to add to it.
The scalar builder has no corresponding preparation phase. For the zero-weight
case this field is `null`, because its constructor skips preparation.

### Reused `correct()` lifecycle

Both implementations reuse their already constructed decoder for exactly the
same fixed batch: up to 16 evenly spaced errors from the verified enumeration,
including the first and last errors, repeated the configured number of times.
The JSON records the exact error vectors and calls per sample. Each observation
is the time for the whole batch; divide by `calls_per_sample` only when an
average per-call figure is needed.

`correct()` still uses the unchanged scalar syndrome method. This comparison is
a lifecycle control, not a claim that a constructor optimization accelerates
steady-state correction. It excludes construction and does not model a channel
or estimate a frame-error rate.

### Peak traced allocation

Construction, preparation and reused correction batches are each run separately
under `tracemalloc`. These instrumented runs never contribute to latency samples.
Each trace begins after explicit garbage collection, so it reports peak traced
allocation during that operation, including its live result where applicable.
Existing code objects and reused decoder tables are outside the trace.

These are Python-traced allocation peaks, not process RSS, retained table size,
or complete native allocation totals. Instrumentation overhead is substantial;
do not use these passes to estimate latency.

## Complexity and unchanged public API

For n sites, r stabilizer generators and an error of weight w, the live helper
scans the complete x/z vectors to find nonzero sites, then combines the selected
contributions. Per-candidate syndrome work is O(n + r*w), not O(r*w). One-time
preparation costs O(n*r) time and space, with nine constant-size local power
choices per site. The scalar syndrome calculation costs O(n*r) per candidate.
Candidate allocation, enumeration, table storage and degeneracy/collision
validation remain additional costs. No public API changes are introduced.

## Reading the report and limits

The report includes Python/platform details, configuration, source hashes, raw
samples, medians, parity counts and explicit scope statements. Its
`constructor_scalar_over_live_median_ratio` is the scalar median divided by the
live median for full construction; values greater than one mean the live
constructor was faster in that particular run. Small ratios near one can easily
reflect measurement noise. Use raw samples and multiple independent runs on an
otherwise quiet host before drawing a performance conclusion.

The optimization is limited to construction for exact
`QutritStabilizerCode` instances. Code subclasses use the scalar syndrome path
to preserve overridden semantics. `max_weight` behavior is unchanged: there is
no new coercion, clamping, automatic correction-radius selection or relaxed
exact-correction check. Requests whose bounded set violates exact correction
conditions still fail; negative values still fail; over-length ranges retain
the existing enumeration behavior. Frozen decoder fields and read-only tables
remain in place.

No result here supports claims about FER, channel quality, quantum hardware,
physical error thresholds, larger decoding radii, BP convergence or general
end-to-end simulation speed. Host/load effects, Python version, allocation
behavior, and the proportion of collision-check work can materially change the
observed construction ratio.

## Retained observation

`benchmarks/qutrit_construction/2026-10-15-single-host.json` retains one run
with nine latency samples, three memory samples, and ten correction-batch
repeats. Repository regression suites were also running on the shared host;
these are source-bound observations under shared load, not an isolated-host
performance guarantee or second-host reproduction. The JSON contains all raw
measurements and source hashes. Historical decoder fixtures are unchanged.
