# Quick Start

## One-Command Demo

After installing (see [INSTALL.md](INSTALL.md)):

```bash
python scripts/qec_demo.py
```

This runs the full deterministic adaptive pipeline on 16 fixed input patterns:

```
metrics -> attractor -> strategy -> evaluation -> adaptation -> memory
```

Each step processes a deterministic signal pattern (constant, ramp, oscillation, step change, etc.), classifies the regime, selects a strategy, evaluates improvement, adapts bias, and records to memory — including transition learning and multi-step lookahead.

### What to expect

The output shows one block per input pattern, including:

- **regime** — classified state (stable, unstable, oscillatory, transitional, mixed)
- **attractor** — basin ID and basin score
- **strategy** — selected strategy and its score
- **adaptation** — global bias and trajectory score
- **transition bias** — learned bias from prior regime transitions
- **multi-step factor** — two-step lookahead influence
- **evaluation** — improvement score and outcome classification

At the end, a summary shows regime distribution and memory statistics.

### Determinism

Running the demo twice produces identical output. The system uses no randomness — all scoring, selection, and adaptation are fully deterministic.

## Extended Run

For a more detailed report including per-metric breakdowns and strategy topology analysis:

```bash
python -c "from qec.experiments.metrics_probe import run_experiments, print_experiment_report; print_experiment_report(run_experiments())"
```

This exercises the same pipeline with additional topology and calibration summary output.
