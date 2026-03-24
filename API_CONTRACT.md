# API Contract — v100.0.0

Stable public interfaces for the QEC analysis layer.

These interfaces are the supported entry points for external use.
Internal structures are not exposed and may change without notice.

---

## Entry Points

### 1. Metrics Evaluation

```python
from qec.analysis.field_metrics import compute_field_metrics

result = compute_field_metrics(values: list[float]) -> dict[str, float]
```

**Input**: list of float values (LLR trace, energy sequence, etc.)

**Output**:
```python
{
    "phi_alignment": float,   # [0, 1]
    "curvature": float,       # [0, 1]
    "resonance": float,       # [0, 1]
    "complexity": float,      # [0, 1]
    "symmetry": float,        # [0, 1]
}
```

**Guarantees**: deterministic, no mutation of input, all outputs bounded [0, 1].

---

### 2. Attractor Analysis

```python
from qec.analysis.attractor_analysis import analyze_attractors

result = analyze_attractors(metrics: dict[str, float]) -> dict[str, Any]
```

**Input**: metrics dict (output of field_metrics or evaluate_metrics)

**Output**:
```python
{
    "regime": str,         # one of: stable, transitional, oscillatory, unstable, mixed, degenerate
    "basin_score": float,  # [0, 1]
    "attractors": list,    # attractor details
}
```

**Guarantees**: deterministic, pure function, no side effects.

---

### 3. Strategy Selection

```python
from qec.analysis.strategy_transition import select_next_strategy

result = select_next_strategy(
    metrics: dict,
    strategies: dict[str, dict],
    prev_strategy: dict | None,
    prev_state: dict | None,
    history: list | None = None,
    memory: dict | None = None,
    transition_memory: dict | None = None,
) -> dict[str, Any]
```

**Output**:
```python
{
    "strategy": {"id": str, "score": float, ...},
    "state": dict,          # opaque state for next call
    "adaptation": {
        "bias": float,              # [-0.2, 0.2]
        "transition_bias": float,   # bounded
        "multi_step_factor": float, # [0.8, 1.2]
    } | None,
    "transition": {
        "from": str,
        "to": str,
        "change": str,
    } | None,
}
```

**Guarantees**: deterministic, does not modify any input arguments.

---

### 4. Strategy Evaluation

```python
from qec.analysis.strategy_evaluation import evaluate_strategy

result = evaluate_strategy(
    prev_metrics: dict,
    current_metrics: dict,
    history: list | None = None,
) -> dict[str, Any]
```

**Output**:
```python
{
    "evaluation": {
        "score": float,       # evaluation delta
        "direction": str,     # "improved", "degraded", or "neutral"
    },
    "outcome": str,           # outcome classification
    "history": list,          # updated history
}
```

**Guarantees**: deterministic, pure function.

---

### 5. Transition Learning

```python
from qec.analysis.strategy_transition_learning import record_transition_outcome

result = record_transition_outcome(
    transition_memory: dict,
    regime_before: str,
    attractor_before: str,
    strategy_id: str,
    regime_after: str,
    attractor_after: str,
    eval_score: float,
) -> dict
```

**Output**: updated transition_memory dict.

**Guarantees**: deterministic, bounded memory, returns new dict.

---

### 6. Multi-Step Evaluation

```python
from qec.analysis.multi_step_evaluation import compute_multi_step_factor

result = compute_multi_step_factor(
    regime: str,
    attractor: str,
    strategy_id: str,
    transition_memory: dict,
) -> float  # [0.8, 1.2]
```

**Guarantees**: deterministic, bounded, no recursion.

---

### 7. Physics Signals

```python
from qec.analysis.physics_signal import compute_physics_signals

result = compute_physics_signals(
    history: list[float],
    metrics: dict | None = None,
) -> dict[str, float]
```

**Output**:
```python
{
    "oscillation_strength": float,    # [0, 1]
    "phase_stability": float,         # [0, 1]
    "multiscale_coherence": float,    # [0, 1]
    "system_energy": float,           # [0, 1]
    "control_alignment": float,       # [0, 1]
}
```

**Guarantees**: deterministic, all outputs bounded [0, 1].

---

### 8. Cycle Detection

```python
from qec.analysis.policy_signal_robustness import detect_cycle, compute_cycle_penalty

detected = detect_cycle(history: list[str], window: int = 5) -> bool
penalty = compute_cycle_penalty(history: list[str], window: int = 5) -> float  # [0.8, 1.0]
```

**Guarantees**: deterministic, bounded.

---

### 9. Integrated Scoring

```python
from qec.analysis.policy_signal_robustness import compute_robust_score

score = compute_robust_score(
    base_score: float,
    stability_weight: float = 1.0,
    transition_bias: float = 1.0,
    multi_step_factor: float = 1.0,
    adaptation_modulation: float = 1.0,
    cycle_penalty: float = 1.0,
    trajectory_score: float = 1.0,
) -> float  # [0.0, 1.0]
```

**Guarantees**: deterministic, clamped output, all-neutral defaults.

---

## Global Guarantees

| Property | Contract |
|----------|----------|
| Determinism | All API functions produce identical outputs for identical inputs |
| No mutation | No API function modifies its input arguments |
| Bounded outputs | All numeric outputs are within documented ranges |
| No side effects | No API function writes to disk, network, or global state |
| Stable schema | Output dict keys are fixed per version; new keys may be added but existing keys are not removed |
