# Reproducible Experiments — v101.1.0

Formal experiment definitions for validating system properties.

All experiments use existing scripts and produce deterministic outputs.

---

## Experiment 1 — Determinism Verification

**Objective**: Confirm that two identical runs produce byte-identical output.

**Command**:
```bash
python scripts/qec_demo.py > /tmp/run1.txt 2>&1
python scripts/qec_demo.py > /tmp/run2.txt 2>&1
diff /tmp/run1.txt /tmp/run2.txt
```

**Expected result**: No differences. `diff` produces no output.

**What this verifies**: INV-DET-1 (identical inputs → identical outputs), INV-DET-3 (stable iteration ordering), INV-REP-1 (full reproducibility).

---

## Experiment 2 — Stability / Energy Behavior

**Objective**: Show that the adaptive pipeline produces stable or improving trajectories.

**Command**:
```bash
python scripts/qec_demo.py
```

**Analysis**: Inspect the `eval` lines in output. Count occurrences of IMPROVED vs DEGRADED directions. In a well-functioning system, improvements should outnumber degradations over the full trajectory.

**What this verifies**: INV-MON-1 (trajectory validation discourages degradation), system stability under diverse input regimes.

---

## Experiment 3 — Cycle Suppression

**Objective**: Demonstrate that oscillatory patterns are detected and penalized.

**Command**:
```bash
python -c "
from qec.analysis.policy_signal_robustness import detect_cycle, compute_cycle_penalty

# Oscillatory pattern
history = ['stable', 'unstable', 'stable', 'unstable', 'stable']
print(f'Cycle detected: {detect_cycle(history)}')
print(f'Cycle penalty:  {compute_cycle_penalty(history)}')

# Stable pattern (not a cycle)
stable = ['stable', 'stable', 'stable', 'stable', 'stable']
print(f'Stable detected: {detect_cycle(stable)}')
print(f'Stable penalty:  {compute_cycle_penalty(stable)}')
"
```

**Expected result**:
- Oscillatory pattern: cycle detected = True, penalty < 1.0
- Stable pattern: cycle detected = False, penalty = 1.0

**What this verifies**: INV-MON-2 (cycle detection prevents oscillatory traps).

---

## Experiment 4 — Full Pipeline Validation

**Objective**: Verify end-to-end pipeline runs without exceptions, produces bounded outputs, and is deterministic.

**Command**:
```bash
python -m pytest tests/test_end_to_end_system.py -v
```

**Expected result**: All tests pass. Outputs are within documented bounds. Two runs produce identical results.

**What this verifies**: All system invariants (INV-DET-*, INV-BND-*, INV-LAY-*, INV-MON-*, INV-REP-*).

---

## Experiment 5 — Bounded Scoring Verification

**Objective**: Confirm that the integrated scoring function respects all bounds.

**Command**:
```bash
python -c "
from qec.analysis.policy_signal_robustness import compute_robust_score

# Extreme inputs
print('All neutral:', compute_robust_score(0.5))
print('Max factors:', compute_robust_score(1.0, 1.0, 1.0, 1.2, 1.5, 1.0, 1.1))
print('Min factors:', compute_robust_score(0.1, 0.5, 0.5, 0.8, 0.5, 0.8, 0.7))
print('Zero base:',   compute_robust_score(0.0))
"
```

**Expected result**: All outputs in [0.0, 1.0].

**What this verifies**: INV-BND-3 (final scores clamped), INV-CMP-1 (compositional scoring).

---

## Experiment 6 — Benchmark-Aware Self-Evaluation (v101.1.0)

**Objective**: Verify that the self-evaluation layer produces bounded, deterministic confidence signals.

**Command**:
```bash
python scripts/qec_demo.py --self-eval
```

**Expected result**:
- Benchmark confidence in [0.0, 1.0]
- Relative advantage in [0.0, 1.0]
- Confidence modulation in [0.9, 1.1]
- Repeated runs produce identical output

**What this verifies**: Self-evaluation is deterministic, bounded, and does not alter existing pipeline behavior. Confidence modulation defaults to neutral (1.0) when no benchmark data is available.

---

## Notes

- All experiments require only: `pip install -e .`
- No external data, network access, or GPU required.
- Results are fully reproducible on any platform with the same Python/NumPy versions.
