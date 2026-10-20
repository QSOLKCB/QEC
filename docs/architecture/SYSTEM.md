# System Definition — v101.3.0

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
            × confidence_modulation
            × trust_modulation
            × regime_trust_modulation
```

Result clamped to [0.0, 1.0].

All factors default to 1.0 when data is unavailable, ensuring graceful degradation.

---

## Benchmark-Aware Self-Evaluation (v101.1.0)

Compares QEC performance against deterministic baselines to derive a bounded confidence signal.

- **Relative advantage**: `max(0, qec - baseline) / max(|qec|, |baseline|, 1e-12)`, bounded [0, 1]
- **Benchmark confidence**: Mean relative advantage across all baselines, bounded [0, 1]
- **Confidence modulation**: `0.9 + 0.2 × benchmark_confidence`, range [0.9, 1.1]

Confidence modulation is optional. When no benchmark data is available, it defaults to 1.0 (neutral). It is applied at the outermost scoring layer only. No existing factor formulas are altered.

**Properties**: deterministic, bounded, side-effect free, opt-in.

---

## Temporal Confidence (v101.2.0)

Tracks confidence over time to derive a bounded trust signal.

### Confidence History

A bounded FIFO list (max 10 entries) of confidence values, updated each step.

### Stability

Measures confidence volatility: `stability = 1 / (1 + variance)`, bounded [0, 1]. High stability means consistent confidence across steps.

### Trend

Directional signal: `trend = last - first`, clamped to [-1, 1]. Positive trend indicates improving confidence over time.

### Trust Signal

Combines stability and trend:

```
trend_factor = 0.5 + 0.5 × trend       # maps [-1,1] → [0,1]
trust = stability × trend_factor        # bounded [0,1]
```

Interpretation:
- High stability + positive trend → high trust
- Unstable or declining → low trust

### Trust Modulation

Optional multiplicative factor for the scoring layer:

```
trust_modulation = 0.9 + 0.2 × trust   # range [0.9, 1.1]
```

Neutral (1.0) when trust = 0.5. Defaults to 1.0 when no temporal data is available.

**Properties**: deterministic, bounded, side-effect free, opt-in.

---

## Regime-Aware Trust (v101.3.0)

Tracks confidence history per regime, computes local trust, and blends it with global trust.

### Regime Confidence Memory

A per-regime bounded FIFO dict mapping `regime_key → list[float]` (max 10 entries per regime). Updated each step with the current confidence for the active regime.

### Local Trust

Computed from the regime's own confidence history using the same stability/trend/trust pipeline as global trust. Represents how confident the system is *within a specific regime*.

### Global vs Local

- **Global trust**: derived from overall temporal confidence history (all regimes combined)
- **Local trust**: derived from a single regime's confidence history
- Systems may trust themselves in one regime but not another

### Blending

```
blended_trust = alpha × local_trust + (1 - alpha) × global_trust
```

Default alpha = 0.5 (equal weight). Result clamped to [0, 1].

### Regime Trust Modulation

```
regime_trust_modulation = 0.9 + 0.2 × blended_trust   # range [0.9, 1.1]
```

Neutral (1.0) when blended_trust = 0.5. Defaults to 1.0 when no regime data is available.

**Properties**: deterministic, bounded, side-effect free, opt-in, per-regime isolation (no cross-regime interactions).

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
