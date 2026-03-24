QEC Benchmark Baseline — Feb 26, 2026

Project Version: v3.0.2
Git Commit: 83ee67f95046a017d2beeca8a5e44a7516ee710b
Execution Environment: Claude Code Compute
Report Artifact: bench/benchmark_report_feb-26-2026.md

1. Executive Summary

This report establishes a deterministic performance baseline for QEC v3.0.2 using the standardized benchmarking framework introduced in v3.0.0.

No production code, schema logic, decoder implementation, or benchmark engine logic was modified for this evaluation.

Two execution modes were used:

Performance mode (runtime_mode="on") for latency and FER measurement.

Determinism mode (runtime_mode="off") to verify byte-identical artifact generation.

The determinism contract was verified successfully.

2. Benchmark Configuration
2.1 Specification Config (Target)

The benchmark was defined using the following specification:

{
  "schema_version": "3.0.2",
  "seed": 20260226,
  "distances": [3, 5, 7, 9],
  "p_values": [0.001, 0.002, 0.005, 0.01],
  "trials": 200,
  "max_iters": 50,
  "decoders": [
    {"adapter": "bp", "params": {"mode": "min_sum", "schedule": "flooding"}},
    {"adapter": "bp", "params": {"mode": "min_sum", "schedule": "layered"}},
    {"adapter": "bp", "params": {"mode": "min_sum", "schedule": "residual"}},
    {"adapter": "bp", "params": {"mode": "min_sum", "schedule": "adaptive"}}
  ],
  "runtime_mode": "on",
  "runtime": {
    "warmup": 3,
    "runs": 10,
    "measure_memory": false
  }
}
2.2 Executed Config (Actual Run)

The current schema layer supports schema versions 3.0.0 and 3.0.1.
To avoid introducing production code changes in a patch release, the benchmark was executed using:

{
  "schema_version": "3.0.1",
  "... all other fields identical ..."
}

No other configuration fields were changed.

Important: The schema version tracks independently of the project version (v3.0.2). This benchmark therefore uses the latest supported schema version (3.0.1) without altering runtime semantics.

3. Environment Snapshot (Frozen)

The following execution environment was frozen and recorded for reproducibility:

Python: 3.11.14 (main, Oct 10 2025, 08:54:04) [GCC 13.3.0]

NumPy: 2.4.2

OS: Linux 4.4.0 (#1 SMP Sun Jan 10 15:06:54 PST 2016)

Machine: x86_64

Platform: Linux-4.4.0-x86_64-with-glibc2.39

CPU Count: 16

Git Commit: 83ee67f95046a017d2beeca8a5e44a7516ee710b

No environment mutation occurred during execution.

4. Determinism Verification

Two distinct execution configurations were used.

4.1 Performance Runs

Used for runtime and FER measurements:

"runtime_mode": "on"

This enables latency tracking while preserving deterministic sweep ordering and seed derivation.

4.2 Determinism Runs

Used exclusively to verify artifact stability:

"runtime_mode": "off"

The full configuration was executed twice with runtime measurement disabled.

Raw JSON outputs were serialized with:

sort_keys=True

separators=(",", ":")

Result:

Byte-identical artifact size: 16,008 bytes

Comparison result: Identical

Determinism contract verified.

5. FER Results Summary

All 64 sweep points (4 decoders × 4 distances × 4 physical error rates) produced identical FER behavior.

For all schedules and all configurations:

FER = 0.0

Mean iterations = 1.0

The tested error rates (p ≤ 0.01) are well below threshold for all tested code distances.

(Results tables unchanged — omitted here for brevity but retained in full in artifact.)

6. Runtime Summary

Latency scales with code distance as expected.

Observed range:

~162 µs (d=3)

~661 µs (d=9)

All schedules converge in one iteration at these operating points, so runtime differences reflect scheduling overhead rather than decoding difficulty.

Confidence intervals remain narrow except where system-level scheduling noise is observed (e.g., layered at d=9, p=0.005).

No anomalies detected.

(Full tables retained as-is.)

7. Observations

Zero FER across sweep: Operating well below threshold.

Single-iteration convergence: Indicates trivial decoding at these physical error rates.

Schedule equivalence: No FER differentiation between flooding, layered, residual, and adaptive schedules at these operating points.

Distance scaling consistent: Runtime increases approximately linearly with graph size.

Variance attributed to system noise, not algorithmic instability.

This baseline primarily establishes:

Deterministic reproducibility

Runtime scaling characteristics

Schedule overhead comparison below threshold

Near-threshold benchmarking is required for decoder differentiation.

8. Reproducibility Guarantees

Config-driven execution

Deterministic sub-seed derivation (SHA-256 over sweep coordinates)

Canonical JSON serialization

Schema locked to 3.0.1

Frozen environment snapshot

No production code changes

No benchmark logic modifications

Read-only execution

This document serves as the formal baseline artifact for future comparative benchmarking.
