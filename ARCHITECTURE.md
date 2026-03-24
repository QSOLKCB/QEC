# Architecture

## Overview

QEC is a deterministic structural analysis and adaptive control system for LDPC / QLDPC Tanner graphs. It operates as a closed-loop control system:

```
metrics -> attractor -> strategy -> evaluation -> adaptation -> memory
```

All operations are deterministic, reproducible, and externally controlled — the decoder core is never modified by the control loop.

## Layer Model

Dependencies flow strictly downward. Lower layers must never import higher layers.

| Layer | Path | Role |
|-------|------|------|
| 1 | `src/qec/decoder/` | Decoder core (protected, never modified) |
| 2 | `src/qec/channel/` | Channel models (BSC, oracle, syndrome) |
| 3 | `src/qec/diagnostics/` | Observational diagnostics (BP dynamics, spectral analysis) |
| 4 | `src/qec/predictors/` | Instability predictors (pre-decode risk signals) |
| 5 | `src/qec/analysis/` | Core pipeline (metrics, attractors, strategies, memory) |
| 6 | `src/qec/experiments/` | Experimental orchestration |
| 7 | `src/bench/` | Benchmark harness |

### Decoder Core (Layer 1) — Protected

The decoder (`src/qec/decoder/`) implements belief propagation and is treated as a fixed experimental object. It is:

- Never modified by the adaptive system
- Bit-stable across minor releases
- Protected by architectural invariant

The BP reference implementation, energy computation, and structural configuration live here. All external interaction with the decoder is through its public interface — no hooks, no injection, no modifications.

### Channel Models (Layer 2)

Channel models (`src/qec/channel/`) provide syndrome generation, LLR computation, and noise modeling. These are the input interface to the decoder.

### Diagnostics (Layer 3)

Diagnostics (`src/qec/diagnostics/`) are observational only. They compute metrics from decoder outputs without modifying decoder inputs or behavior. Key modules:

- `bp_dynamics.py` — BP trajectory metrics
- `spectral_nb.py` — Non-backtracking spectrum analysis
- `stability_*.py` — Stability prediction and classification

Diagnostics operate on copies and are side-effect free.

### Predictors (Layer 4)

Predictors (`src/qec/predictors/`) estimate instability before decoding. They consume diagnostics outputs and produce signals only — they never modify the decoder or its inputs.

### Analysis (Layer 5) — Core Pipeline

The analysis layer (`src/qec/analysis/`) contains the core adaptive pipeline with ~130 modules. The six pipeline stages live here:

**Metrics** — `field_metrics.py`, `multiscale_metrics.py`
- Phi alignment, curvature, resonance, complexity
- Scale consistency, scale divergence

**Attractor analysis** — `attractor_analysis.py`
- Regime classification (stable, transitional, oscillatory, unstable, mixed)
- Basin score computation
- Transition detection

**Strategy selection** — `strategy_transition.py`
- Deterministic regime-action scoring table
- Metric-driven adjustments
- Deterministic tie-breaking (score, confidence, simplicity, lexicographic ID)

**Evaluation** — `strategy_evaluation.py`
- Before/after state comparison
- Improvement scoring with weighted deltas
- Outcome classification (stabilized, recovered, damped, regressed, neutral)

**Adaptation** — `strategy_adaptation.py`
- Global bias from evaluation history
- Trajectory scoring with recency weighting
- Regime-aware evaluation weight overrides

**Memory** — three layers:
- `strategy_memory.py` — Per-strategy and regime-indexed memory (v99.1–99.2)
- `strategy_transition_learning.py` — Transition outcome recording and bias (v99.3)
- `multi_step_evaluation.py` — Two-step lookahead scoring (v99.4)

### Experiments (Layer 6)

Experiments (`src/qec/experiments/`) orchestrate runs using the analysis pipeline. Key module:

- `metrics_probe.py` — Full pipeline probe with deterministic test inputs

### Benchmarks (Layer 7)

The benchmark harness (`src/bench/`) is external to the core package and must never be imported by it.

## Determinism Guarantees

The system enforces bitwise reproducibility:

- No hidden randomness
- Explicit seed injection where RNG is needed
- Deterministic iteration ordering (sorted keys)
- Canonical serialization (JSON-safe outputs)
- No use of Python `hash()` (unstable across runs)
- No unordered floating-point reductions
- Stable multi-key ranking for tie-breaking

With the same inputs and configuration, outputs are byte-identical across runs.

## Memory Architecture

All memory is:

- **Bounded** — capped at 10 events per key (configurable)
- **Immutable** — updates return new dicts; inputs are never mutated
- **Deterministic** — no stochastic learning, no randomness
- **Indexed** — regime-aware keys for targeted adaptation

### Per-Strategy Memory (v99.1)

Maps `strategy_id -> [events]`. Tracks score and outcome per strategy to compute local bias.

### Regime-Indexed Memory (v99.2)

Maps `(regime_key, strategy_id) -> [events]` with fallback to global aggregation. Stability-weighted scoring reduces noise from volatile strategies.

### Transition Learning (v99.3)

Maps `(regime_before, attractor_before, strategy_id, regime_after, attractor_after) -> stats`. Records transition outcomes and computes multiplicative bias in [0.8, 1.2] based on historical success rate.

### Multi-Step Evaluation (v99.4)

Uses transition memory to estimate two-step expected value. For each candidate strategy, estimates the combined value of applying it now and the best follow-up, producing a multiplicative factor in [0.8, 1.2]. Fixed horizon of 2 — no recursion.

### Combined Scoring Formula (v99.4)

```
final_score = base_score * stability_weight * transition_bias * multi_step_factor
```

All factors are clamped to [0, 1] for the final score.

## Why the Decoder is Untouched

The decoder is a fixed experimental object — like an instrument being measured, not a variable being optimized. The adaptive system operates entirely outside the decoder:

- Metrics measure decoder behavior
- Strategies adjust decoder inputs (LLR, schedules, graph structure)
- The decoder itself is never modified

This separation ensures that experimental results are reproducible and that the decoder remains bit-stable across releases. Control happens at the boundary, not inside the system.
