# QEC Industry Comparison Harness — Feb 26, 2026

## 1. Summary

**What ran:**

- **QEC (this repo):** BP min-sum decoder with three schedules (flooding, residual, adaptive) across distances d={7,9,11}.
  - **Config A — Operating-region baseline:** p={0.01, 0.02, 0.05}, 1000 trials, runtime measurement enabled.
  - **Config B — Transition-band probe:** p={0.49, 0.499, 0.50}, 200 trials, runtime measurement enabled.
  - **Determinism check:** d=9, p=0.499, 20 trials, runtime_mode=off, deterministic_metadata=true. Two runs, byte-identical comparison.
- **Stim 1.15.0 + PyMatching 2.3.1 (reference baselines):**
  - **R1 — Repetition code:** d={7,9,11}, p={0.01,0.02,0.05,0.10}, 10,000 shots, MWPM decoder.
  - **R2 — Surface code (rotated memory Z):** d={7,9,11}, p={0.005,0.01,0.02}, 10,000 shots, MWPM decoder.

**What is NOT comparable (critical disclaimer):**

QEC's QLDPC CSS codes with an oracle-aided LLR channel model are fundamentally different from Stim's circuit-level noise models on repetition and surface codes decoded by MWPM. These baselines provide **reference context only** — they are not head-to-head competitors on the same problem.

**Key takeaways (neutral):**

1. QEC achieves FER=0 for all p < 0.50 under its oracle-aided channel model. The transition is a degenerate step function at p=0.50 (explained by `channel_llr(e, p)` encoding exact error positions via LLR signs).
2. Stim+PyMatching baselines show realistic, non-trivial logical error rates that decrease with distance (as expected for standard code families under circuit-level depolarizing noise).
3. QEC's deterministic benchmark pipeline produces byte-identical results across runs (PASS).
4. QEC's per-decode latency ranges from ~350–630 µs (pure Python, 1 BP iteration below threshold). Stim+PyMatching (C++/Rust internals) achieves ~1–210 µs/shot including sampling + decoding.
5. Runtime comparisons between the two stacks are not meaningful due to fundamentally different implementations (pure Python BP vs. compiled C++/Rust MWPM) and different workloads (single-shot decode vs. batch sample+decode).

---

## 2. Environment Snapshot

| Key | Value |
|-----|-------|
| Python version | 3.11.14 (main, Oct 10 2025, 08:54:04) [GCC 13.3.0] |
| NumPy version | 2.4.2 |
| Platform | Linux-4.4.0-x86_64-with-glibc2.39 |
| CPU count | 16 |
| Git commit | `856c1d98f4423d9949161d816b3dfa7e633ad06d` |
| QEC version | 3.0.2 |
| Stim version | 1.15.0 |
| PyMatching version | 2.3.1 |

---

## 3. QEC Results (This Repo)

### 3.1 Operating-Region Baseline (p = 0.01, 0.02, 0.05)

**Config A:** 1000 trials, max_iters=50, runtime enabled (warmup=3, runs=10).

All three schedules produce **FER = 0.000** and **mean iters = 1.0** at every (distance, p) point. This is expected: for any p < 0.50, the oracle channel model (`channel_llr`) provides perfect sign information, and BP converges in a single iteration.

**Runtime (µs) — flooding schedule**

| d (n qubits) | p=0.01 | p=0.02 | p=0.05 |
|---|---|---|---|
| 7 (n=84) | 362 [358–366] | 377 [363–391] | 363 [348–378] |
| 9 (n=108) | 448 [441–454] | 453 [441–466] | 461 [458–463] |
| 11 (n=132) | 555 [551–560] | 554 [552–557] | 536 [531–540] |

**Runtime (µs) — residual schedule**

| d (n qubits) | p=0.01 | p=0.02 | p=0.05 |
|---|---|---|---|
| 7 (n=84) | 354 [348–360] | 357 [352–363] | 347 [341–353] |
| 9 (n=108) | 480 [436–525] | 450 [446–454] | 443 [435–452] |
| 11 (n=132) | 558 [551–566] | 645 [577–712] | 594 [575–612] |

**Runtime (µs) — adaptive schedule**

| d (n qubits) | p=0.01 | p=0.02 | p=0.05 |
|---|---|---|---|
| 7 (n=84) | 378 [369–386] | 395 [387–403] | 373 [364–383] |
| 9 (n=108) | 499 [492–505] | 469 [463–475] | 627 [558–695] |
| 11 (n=132) | 582 [569–595] | 586 [582–591] | 629 [585–672] |

**Runtime scaling (log-log slope, averaged across p values):**

| Schedule | Slope | Interpretation |
|---|---|---|
| flooding | 0.88 | Sub-linear scaling |
| residual | 1.17 | Slightly super-linear (driven by variance spikes) |
| adaptive | 1.01 | Approximately linear |

### 3.2 Transition-Band Probe (p = 0.49, 0.499, 0.50)

**Config B:** 200 trials, max_iters=50, runtime enabled (warmup=3, runs=10).

**FER and mean iterations:**

| d | p | flooding FER / iters | residual FER / iters | adaptive FER / iters |
|---|---|---|---|---|
| 7 | 0.490 | 0.000 / 1.0 | 0.000 / 1.0 | 0.000 / 1.0 |
| 7 | 0.499 | 0.000 / 1.0 | 0.000 / 1.0 | 0.000 / 1.0 |
| 7 | **0.500** | **1.000 / 50.0** | **1.000 / 50.0** | **1.000 / 12.0** |
| 9 | 0.490 | 0.000 / 1.0 | 0.000 / 1.0 | 0.000 / 1.0 |
| 9 | 0.499 | 0.000 / 1.0 | 0.000 / 1.0 | 0.000 / 1.0 |
| 9 | **0.500** | **1.000 / 50.0** | **1.000 / 50.0** | **1.000 / 12.0** |
| 11 | 0.490 | 0.000 / 1.0 | 0.000 / 1.0 | 0.000 / 1.0 |
| 11 | 0.499 | 0.000 / 1.0 | 0.000 / 1.0 | 0.000 / 1.0 |
| 11 | **0.500** | **1.000 / 50.0** | **1.000 / 50.0** | **1.000 / 12.0** |

At p=0.50, `base_llr = log(1) = 0`, providing zero channel information. All schedules fail (FER=1.0). The adaptive schedule terminates early at 12 iterations (phase-1 budget), while flooding and residual run to max_iters=50.

**Runtime at p=0.50 (µs):**

| d | flooding | residual | adaptive |
|---|---|---|---|
| 7 | 14,951 [14535–15367] | 14,622 [14570–14675] | 15,695 [15334–16056] |
| 9 | 19,668 [19204–20132] | 19,079 [18901–19257] | 19,413 [19202–19624] |
| 11 | 24,612 [23046–26178] | 23,100 [22677–23524] | 22,591 [22359–22824] |

Note: Adaptive runs only 12 iterations but takes comparable time to 50-iteration schedules, indicating ~4x higher per-iteration overhead from convergence monitoring.

---

## 4. Determinism Verification (QEC)

**Configuration:**
- d=9, p=0.499, 20 trials
- `runtime_mode="off"`, `deterministic_metadata=true`
- Three decoders: flooding, residual, adaptive

**Result: PASS** — byte-identical output across two independent runs.

| Metric | Value |
|---|---|
| File size | 1906 bytes (each) |
| MD5 | `32d4c8890388607be8fa030491671950` (both) |
| Canonical JSON | `sort_keys=True, separators=(",",":")`  |

The `deterministic_metadata=true` flag sets `created_utc` to `"1970-01-01T00:00:00+00:00"`, eliminating timestamp non-determinism. Combined with SHA-256-based sub-seed derivation (independent of sweep ordering), this guarantees reproducible results.

---

## 5. Reference Baselines (Stim + PyMatching)

> **Disclaimer:** These baselines use different code families (repetition code, rotated surface code), different noise models (circuit-level depolarizing noise), and a different decoder (minimum-weight perfect matching). They are **not directly comparable** to QEC's QLDPC CSS codes under the oracle-aided LLR channel model. They are provided as **reference context** for understanding typical QEC performance numbers in the literature.

### 5.1 Repetition Code (Reference Baseline R1)

**Setup:** `stim.Circuit.generated("repetition_code:memory")`, rounds=d, distance=d, `after_clifford_depolarization=p`. Decoded with PyMatching MWPM. 10,000 shots per point.

| d | p=0.01 | p=0.02 | p=0.05 | p=0.10 |
|---|---|---|---|---|
| 7 | 0.0000 | 0.0003 | 0.0071 | 0.0773 |
| 9 | 0.0000 | 0.0000 | 0.0035 | 0.0686 |
| 11 | 0.0000 | 0.0001 | 0.0016 | 0.0621 |

**Runtime (µs per shot, including sampling + decoding):**

| d | p=0.01 | p=0.02 | p=0.05 | p=0.10 |
|---|---|---|---|---|
| 7 | 0.7 | 0.8 | 2.0 | 4.9 |
| 9 | 0.9 | 1.4 | 3.6 | 9.2 |
| 11 | 1.4 | 2.2 | 5.6 | 15.3 |

**Observations:**
- Logical error rate decreases with distance at fixed p, confirming expected scaling behavior.
- At p=0.10, LER is substantial (~6–8%) even at d=11, consistent with the repetition code's threshold being near p≈0.11 for this noise model.
- Runtime is sub-microsecond for low-p points due to Stim's compiled sampling and PyMatching's C++/Rust decoder.

### 5.2 Surface Code — Rotated Memory Z (Reference Baseline R2)

**Setup:** `stim.Circuit.generated("surface_code:rotated_memory_z")`, rounds=d, distance=d, `after_clifford_depolarization=p`. Decoded with PyMatching MWPM. 10,000 shots per point.

| d | p=0.005 | p=0.01 | p=0.02 |
|---|---|---|---|
| 7 | 0.0017 | 0.0250 | 0.1984 |
| 9 | 0.0010 | 0.0249 | 0.2552 |
| 11 | 0.0002 | 0.0231 | 0.3174 |

**Runtime (µs per shot, including sampling + decoding):**

| d | p=0.005 | p=0.01 | p=0.02 |
|---|---|---|---|
| 7 | 7.7 | 16.1 | 35.4 |
| 9 | 17.8 | 40.1 | 97.9 |
| 11 | 35.2 | 83.4 | 210.9 |

**Observations:**
- At p=0.005, LER decreases strongly with distance (0.0017 → 0.0002), indicating operation below the surface code threshold (~1%).
- At p=0.01 (near threshold), LER is roughly flat across distances (~2.3–2.5%).
- At p=0.02 (above threshold), LER increases with distance, confirming the expected above-threshold behavior.
- Runtime scales roughly as O(d³) due to the surface code's O(d²) data qubits × O(d) rounds.

---

## 6. Interpretation (Strict)

### 6.1 What Is Comparable vs. Not Comparable

| Aspect | Comparable? | Notes |
|---|---|---|
| Code family | **No** | QEC: QLDPC CSS codes. Baselines: repetition & surface codes. |
| Channel/noise model | **No** | QEC: oracle-aided LLR (uses actual error vector). Baselines: circuit-level depolarizing noise. |
| Decoder | **No** | QEC: BP (min-sum, various schedules). Baselines: MWPM (PyMatching). |
| Distances | Partially | Both use d={7,9,11}. |
| FER/LER metric | Structurally similar | Both measure logical failure rate, but under incompatible models. |
| Runtime | **No** | QEC: pure Python single-decode latency. Baselines: compiled C++/Rust batch (sample+decode). |

### 6.2 Determinism and Reproducibility

- **QEC:** Fully deterministic. SHA-256-derived sub-seeds, canonical JSON serialization, `deterministic_metadata` flag. Byte-identical across runs (verified).
- **Stim+PyMatching:** Stim supports seeded sampling for reproducibility. PyMatching MWPM is deterministic given fixed input. However, the batch workflow (sample + decode) makes per-shot timing non-deterministic.

### 6.3 Scaling Notes

- **QEC runtime** scales sub-linearly to linearly with distance (log-log slopes 0.88–1.17) for a single BP iteration. At the degenerate threshold (p=0.50, multi-iteration), runtime scales approximately linearly with distance.
- **Stim+PyMatching surface code** runtime scales roughly as O(d³) per shot, consistent with the quadratic growth in data qubits and linear growth in rounds.
- **Repetition code** runtime scales roughly as O(d) per shot.

### 6.4 No Superiority Claims

This report makes no claims about which system is "better." The QEC oracle channel model produces a degenerate step-function threshold that does not reflect realistic noise conditions. A syndrome-only decoding model would produce non-trivial FER at much lower p values. The Stim+PyMatching baselines operate under realistic circuit-level noise and are included solely to provide numerical reference points from a well-established industry toolchain.

---

## 7. Reproducibility Notes

- **Evaluation-only artifact.** No production code (src/, tests/, schema, decoders, bench runner) was modified.
- **No dependencies added** to pyproject.toml. Stim and PyMatching were installed ephemerally.
- **No scripts committed.** All Stim/PyMatching code was executed via inline Python (stdin).
- **All configurations** are embedded in the appendices below.
- **Exact software versions** are recorded in Section 2.
- **Seed:** 20260226 (QEC configs). Stim baselines used default random seeds (Stim's internal PRNG).

---

## Appendix A: QEC Config A — Operating-Region Baseline

```json
{
  "schema_version": "3.0.1",
  "seed": 20260226,
  "distances": [7, 9, 11],
  "p_values": [0.01, 0.02, 0.05],
  "trials": 1000,
  "max_iters": 50,
  "decoders": [
    {"adapter": "bp", "params": {"mode": "min_sum", "schedule": "flooding"}},
    {"adapter": "bp", "params": {"mode": "min_sum", "schedule": "residual"}},
    {"adapter": "bp", "params": {"mode": "min_sum", "schedule": "adaptive"}}
  ],
  "runtime_mode": "on",
  "runtime": {"warmup": 3, "runs": 10, "measure_memory": false}
}
```

## Appendix B: QEC Config B — Transition-Band Probe

```json
{
  "schema_version": "3.0.1",
  "seed": 20260226,
  "distances": [7, 9, 11],
  "p_values": [0.49, 0.499, 0.50],
  "trials": 200,
  "max_iters": 50,
  "decoders": [
    {"adapter": "bp", "params": {"mode": "min_sum", "schedule": "flooding"}},
    {"adapter": "bp", "params": {"mode": "min_sum", "schedule": "residual"}},
    {"adapter": "bp", "params": {"mode": "min_sum", "schedule": "adaptive"}}
  ],
  "runtime_mode": "on",
  "runtime": {"warmup": 3, "runs": 10, "measure_memory": false}
}
```

## Appendix C: QEC Determinism Check Config

```json
{
  "schema_version": "3.0.1",
  "seed": 20260226,
  "distances": [9],
  "p_values": [0.499],
  "trials": 20,
  "max_iters": 50,
  "decoders": [
    {"adapter": "bp", "params": {"mode": "min_sum", "schedule": "flooding"}},
    {"adapter": "bp", "params": {"mode": "min_sum", "schedule": "residual"}},
    {"adapter": "bp", "params": {"mode": "min_sum", "schedule": "adaptive"}}
  ],
  "runtime_mode": "off",
  "deterministic_metadata": true
}
```

## Appendix D: Stim + PyMatching Inline Code (Reference Baselines)

The following Python code was executed via `python3 -` (stdin) to generate the reference baselines. No script files were written to the repository.

```python
import stim
import pymatching
import numpy as np
import time

# ============================================================
# Baseline R1: Repetition Code
# ============================================================
for d in [7, 9, 11]:
    for p in [0.01, 0.02, 0.05, 0.10]:
        circuit = stim.Circuit.generated(
            "repetition_code:memory",
            rounds=d,
            distance=d,
            after_clifford_depolarization=p,
            after_reset_flip_probability=0,
            before_measure_flip_probability=0,
            before_round_data_depolarization=0,
        )
        dem = circuit.detector_error_model(decompose_errors=True)
        matcher = pymatching.Matching.from_detector_error_model(dem)

        shots = 10000
        t0 = time.perf_counter()
        sampler = circuit.compile_detector_sampler()
        detection_events, observable_flips = sampler.sample(
            shots, separate_observables=True
        )
        predictions = matcher.decode_batch(detection_events)
        t1 = time.perf_counter()

        num_errors = np.sum(np.any(predictions != observable_flips, axis=1))
        ler = num_errors / shots
        elapsed = t1 - t0

        print(f"d={d}, p={p}: LER={ler:.4f}, "
              f"{elapsed/shots*1e6:.1f} us/shot ({shots} shots)")

# ============================================================
# Baseline R2: Surface Code (Rotated Memory Z)
# ============================================================
for d in [7, 9, 11]:
    for p in [0.005, 0.01, 0.02]:
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=d,
            distance=d,
            after_clifford_depolarization=p,
            after_reset_flip_probability=0,
            before_measure_flip_probability=0,
            before_round_data_depolarization=0,
        )
        dem = circuit.detector_error_model(decompose_errors=True)
        matcher = pymatching.Matching.from_detector_error_model(dem)

        shots = 10000
        t0 = time.perf_counter()
        sampler = circuit.compile_detector_sampler()
        detection_events, observable_flips = sampler.sample(
            shots, separate_observables=True
        )
        predictions = matcher.decode_batch(detection_events)
        t1 = time.perf_counter()

        num_errors = np.sum(np.any(predictions != observable_flips, axis=1))
        ler = num_errors / shots
        elapsed = t1 - t0

        print(f"d={d}, p={p}: LER={ler:.4f}, "
              f"{elapsed/shots*1e6:.1f} us/shot ({shots} shots)")
```

## Appendix E: Commands Used

```bash
# Environment
pip install numpy stim pymatching  # ephemeral install, not in pyproject.toml

# QEC benchmarks (Configs A, B, determinism) — run via inline Python:
python3 - <<'PY'
import json, sys
sys.path.insert(0, '.')
from src.bench.config import BenchmarkConfig
from src.bench.runner import run_benchmark
config = BenchmarkConfig.from_dict({...})  # see appendices A/B/C
result = run_benchmark(config)
print(json.dumps(result, indent=2))
PY

# Stim/PyMatching baselines — run via inline Python:
python3 - <<'PY'
# see Appendix D
PY
```
