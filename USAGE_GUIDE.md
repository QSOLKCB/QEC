# Usage Guide

## Core Workflow

QEC operates as a deterministic closed-loop control system:

```
metrics -> attractor -> strategy -> evaluation -> adaptation -> memory
```

### 1. Metrics (Sensing)

Compute field and multiscale metrics from a value sequence:

```python
from qec.analysis.field_metrics import compute_field_metrics
from qec.analysis.multiscale_metrics import compute_multiscale_summary

field = compute_field_metrics(values)    # phi, curvature, resonance, complexity, ...
multi = compute_multiscale_summary(values)  # scale_consistency, scale_divergence
```

Or use the combined helper from the experiment layer:

```python
from qec.experiments.metrics_probe import evaluate_metrics

metrics = evaluate_metrics(values)  # {"field": ..., "multiscale": ...}
```

### 2. Attractor Analysis (Classification)

Classify the regime and compute a basin score:

```python
from qec.analysis.attractor_analysis import analyze_attractors

attractor = analyze_attractors(metrics)
# {"signals": {...}, "regime": "unstable", "basin_score": 0.32}
```

Regimes (deterministic priority order):
- **stable** — phi > 0.8 AND consistency > 0.8 AND curvature < 0.2
- **transitional** — divergence > 0.4 AND 0.4 <= consistency <= 0.8
- **oscillatory** — resonance > 0.5 AND curvature_var > 0.2
- **unstable** — curvature > 0.4 OR complexity > 0.6
- **mixed** — default fallback

### 3. Strategy Selection (Decision)

Select a strategy based on the current state:

```python
from qec.analysis.strategy_transition import select_next_strategy

full_metrics = {**metrics, "attractor": attractor}
decision = select_next_strategy(full_metrics, strategies)
# {"strategy": {"id": "s1", "score": 0.9, ...}, "state": {...}, "transition": ...}
```

Strategies are scored by a deterministic regime-action table with metric-driven adjustments.

### 4. Evaluation (Feedback)

Compare before/after metrics to measure improvement:

```python
from qec.analysis.strategy_evaluation import evaluate_strategy

result = evaluate_strategy(prev_metrics, curr_metrics, history=eval_history)
# {"delta": {...}, "evaluation": {"improved": True, "score": 0.15, "direction": "improved"},
#  "outcome": "stabilized", "history": [...]}
```

Outcomes: stabilized, recovered, damped, regressed, neutral.

### 5. Adaptation (Global Bias)

Feed evaluation history back into strategy scoring:

```python
decision = select_next_strategy(
    full_metrics, strategies,
    prev_strategy=prev_decision["strategy"],
    prev_state=prev_decision["state"],
    history=eval_history,
    memory=strategy_memory,
    transition_memory=transition_memory,
)
```

When history and memory are provided, the system uses:
- Global adaptation bias from evaluation history
- Per-strategy local bias from strategy memory
- Transition bias from transition learning (v99.3.0)
- Multi-step lookahead factor (v99.4.0)

### 6. Memory

Three memory layers accumulate deterministic performance records:

**Per-strategy memory** (`strategy_memory.py`):
```python
from qec.analysis.strategy_memory import update_strategy_memory, update_regime_memory
```

**Transition learning** (`strategy_transition_learning.py`):
```python
from qec.analysis.strategy_transition_learning import record_transition_outcome
```

**Multi-step evaluation** (`multi_step_evaluation.py`):
```python
from qec.analysis.multi_step_evaluation import compute_two_step_value, compute_multi_step_factor
```

All memory is bounded (capped at 10 events per key), deterministic, and immutable (updates return new dicts).

## Entry Points

### Demo Script

```bash
python scripts/qec_demo.py
```

Runs the full adaptive loop on fixed deterministic inputs. See [QUICKSTART.md](QUICKSTART.md).

### Metrics Probe Experiment

```python
from qec.experiments.metrics_probe import run_experiments, print_experiment_report

results = run_experiments()
print_experiment_report(results)
```

Runs 16 input patterns through the full pipeline with strategy topology analysis.

### Trajectory Analysis

```python
from qec.experiments.metrics_probe import run_trajectory_experiments

trajectories = run_trajectory_experiments()
```

Analyzes fixed metric sequences (stable convergence, regime transition, oscillatory switching, unstable escalation, mixed plateau).

### CLI — Spectral Threshold Search

```bash
qec-exp spectral-search --iterations 10 --seed 0
```

Key flags:

| Flag | Description |
|------|-------------|
| `--iterations N` | Number of search iterations |
| `--seed S` | Deterministic seed |
| `--output-dir DIR` | Output directory (default: `experiments/threshold_search`) |
| `--population N` | Population size |
| `--enable-bp-diagnostics` | Enable BP diagnostic metrics |
| `--enable-pareto` | Enable Pareto front tracking |
| `--enable-learning` | Enable learning mode |

Run `qec-exp spectral-search --help` for the full list.

## Interpreting Outputs

### Regime

The regime tells you the system's current dynamical state. Stable means metrics are consistent and curvature is low. Unstable means high curvature or complexity. The system uses regime classification to select appropriate strategies.

### Basin Score

A scalar in [0, 1] summarizing overall stability. Higher is more stable. Computed as a weighted combination of phi alignment, consistency, curvature, and divergence.

### Strategy Score

How well a strategy matches the current regime. Ranges from 0 to 1. Includes contributions from base regime-action scoring, global adaptation bias, local memory bias, transition learning bias, and multi-step lookahead factor.

### Evaluation Score

Measures improvement between consecutive steps. Positive means the system improved; negative means regression. The outcome classifier maps this to human-readable labels (stabilized, recovered, damped, regressed, neutral).
