# System Definition — v101.0.0

Formal definition of the QEC deterministic adaptive control system.

---

## Pipeline Definition

The system implements a deterministic state-transition pipeline:

```
S₀ → metrics → attractor → strategy → transition → evaluation → adaptation → S₁
```

Where:
- `S₀` = initial system state (LLR values, graph structure, memory)
- `S₁` = updated system state (new memory, updated biases)

Each stage is a pure function. The pipeline is compositional: stages can be composed, and each stage preserves all system invariants (see [INVARIANTS.md](INVARIANTS.md)).

---

## Components

### 1. Metrics (`field_metrics`, `multiscale_metrics`)

Extracts deterministic scalar signals from input data.

- **Inputs**: raw values (LLR traces, energy sequences)
- **Outputs**: phi_alignment, curvature, resonance, complexity, symmetry, scale_consistency
- **Properties**: pure, bounded [0,1], deterministic

### 2. Attractor Analysis (`attractor_analysis`)

Classifies the current system state into a regime and computes basin stability.

- **Inputs**: metrics dict
- **Outputs**: regime label (str), basin_score (float), attractor_id (str)
- **Regimes**: stable, transitional, oscillatory, unstable, mixed, degenerate
- **Properties**: deterministic classification, no side effects

### 3. Strategy Selection (`strategy_transition`)

Selects the next strategy based on current metrics, history, and memory.

- **Inputs**: full_metrics, strategy catalog, previous strategy/state, history, memory, transition_memory
- **Outputs**: selected strategy (id, score), adaptation info, transition info
- **Properties**: deterministic scoring, regime-aware, memory-biased

### 4. Transition Learning (`strategy_transition_learning`)

Records outcomes of strategy transitions for future decision-making.

- **Inputs**: transition_memory, before/after regime+attractor, strategy_id, eval_score
- **Outputs**: updated transition_memory
- **Properties**: deterministic, bounded memory, append-only within bounds

### 5. Multi-Step Evaluation (`multi_step_evaluation`)

Two-step lookahead using transition memory to estimate future value.

- **Inputs**: current regime/attractor, strategy_id, transition_memory
- **Outputs**: multi_step_factor ∈ [0.8, 1.2]
- **Formula**: `factor = 1 + α × normalized_two_step_value`, α = 0.2
- **Properties**: deterministic, bounded, no recursion, horizon fixed at 2

### 6. Strategy Evaluation (`strategy_evaluation`)

Compares before/after metrics to assess strategy effectiveness.

- **Inputs**: prev_metrics, current_metrics, history
- **Outputs**: evaluation score, direction (IMPROVED/DEGRADED/NEUTRAL), outcome classification
- **Properties**: deterministic, history-aware

### 7. Physics Signal Layer (`physics_signal`)

Extracts physics-informed signals for adaptation modulation.

- **Signals**: oscillation_strength, phase_stability, multiscale_coherence, system_energy, control_alignment
- **All outputs**: bounded [0, 1], deterministic, side-effect free

### 8. Adaptation Modulation (`strategy_memory`)

Computes a modulation factor from physics signals.

- **Formula**: geometric mean of `(1-energy) × phase × coherence × alignment`, shifted by 0.5
- **Output**: adaptation_modulation ∈ [0.5, 1.5]
- **Properties**: deterministic, regime-sensitive damping for oscillatory states

### 9. Policy Constraints & Cycle Detection (`policy_signal_robustness`)

Prevents oscillatory traps and ensures scoring robustness.

- **Cycle detection**: scans recent history for period-2 and period-3 patterns
- **Cycle penalty**: ∈ [0.8, 1.0], scales with number of distinct labels in cycle
- **Signal decorrelation**: normalizes redundant signals
- **Properties**: deterministic, bounded

### 10. Trajectory Validation (`trajectory_validation`)

Validates transition quality and enforces monotonic improvement.

- **Inputs**: before/after metrics (score, energy, coherence)
- **Output**: trajectory_score ∈ [0.7, 1.1]
- **Properties**: deterministic, penalizes degradation, rewards improvement

### 11. Strategy Memory (`strategy_memory`)

Bounded per-strategy performance records indexed by regime.

- **Schema**: `Dict[(regime_key, strategy_id), List[event]]`
- **Bound**: max 10 events per key
- **Properties**: deterministic, regime-aware, recency-weighted

---

## Final Scoring Function

```
final_score = base_score
            × stability_weight
            × transition_bias
            × multi_step_factor
            × adaptation_modulation
            × cycle_penalty
            × trajectory_score
```

Result clamped to [0.0, 1.0].

All factors default to 1.0 when data is unavailable, ensuring graceful degradation.

---

## Properties

| Property | Guarantee |
|----------|-----------|
| Deterministic | Identical inputs → identical outputs, always |
| Bounded | All signals ∈ [0,1], all factors bounded, final score ∈ [0,1] |
| Compositional | Each stage is a pure function; stages compose without side effects |
| No randomness | No stochastic exploration, no ML, no RL |
| No mutation | All functions operate on copies; inputs are never modified |
| Memory-bounded | Strategy memory capped at 10 events per key |
| Horizon-bounded | Multi-step lookahead fixed at horizon = 2 |

---

## Benchmarking Layer (v101)

### Baseline Strategies (`analysis/baseline_strategies`)

Deterministic baselines for comparison against the adaptive pipeline:

- **random_strategy_deterministic**: SHA-256 seeded pseudo-random selection. Identical seed+step → identical choice.
- **fixed_strategy**: Always returns the same strategy.
- **round_robin_strategy**: Cycles through strategies by step index.

### Performance Metrics (`analysis/performance_metrics`)

- **compute_cumulative_score**: Running cumulative average of scores.
- **compute_convergence_rate**: Mean absolute step-to-step change (lower = faster convergence).
- **compute_stability_variance**: Population variance of scores (lower = more stable).
- **compute_final_performance**: Mean of trailing window of scores.

### Convergence Analysis (`analysis/convergence_analysis`)

- **detect_convergence**: Finds the first step where scores stabilize within a window threshold.
- **compute_convergence_signal**: [0, 1] measure of tail stability (1.0 = perfectly converged).

### Benchmark Comparison (`analysis/benchmark_comparison`)

- **compare_strategies**: Computes relative performance ratios, convergence differences, and stability differences between QEC and each baseline.

### Benchmark Runner (`experiments/benchmark_runner`)

- **run_benchmark**: Executes QEC pipeline and all baselines on identical inputs, returns structured results for comparison.

All benchmarking components are deterministic, bounded, and side-effect free.
