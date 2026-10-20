# System Invariants — v100.0.0

Formal invariants governing the QEC deterministic adaptive control system.

Every component must preserve these invariants at all times.
Violation of any invariant constitutes a system-level defect.

---

## 1. Determinism

**INV-DET-1**: Identical inputs produce identical outputs across all runs.

**INV-DET-2**: No hidden randomness. All stochastic sources require explicit seed injection via `numpy.random.RandomState(seed)`.

**INV-DET-3**: Stable iteration ordering. All dict iterations use `sorted()` keys. No dependence on `hash()` or insertion order for correctness.

**INV-DET-4**: Canonical serialization. All artifact outputs use deterministic JSON with sorted keys and consistent float precision.

**INV-DET-5**: When `runtime_mode = "off"`, outputs are byte-identical across runs on the same platform.

---

## 2. Boundedness

**INV-BND-1**: All physics signals are bounded to [0, 1].
- `system_energy ∈ [0, 1]`
- `phase_stability ∈ [0, 1]`
- `multiscale_coherence ∈ [0, 1]`
- `control_alignment ∈ [0, 1]`
- `oscillation_strength ∈ [0, 1]`

**INV-BND-2**: All modulation factors are bounded.
- `adaptation_modulation ∈ [0.5, 1.5]`
- `multi_step_factor ∈ [0.8, 1.2]`
- `cycle_penalty ∈ [0.8, 1.0]`
- `trajectory_score ∈ [0.7, 1.1]`

**INV-BND-3**: All final scores are clamped to [0.0, 1.0].

**INV-BND-4**: Strategy memory is bounded. Maximum 10 events per (regime_key, strategy) pair.

**INV-BND-5**: Global bias is bounded to [-0.2, +0.2]. Local bias is bounded to [-0.2, +0.2].

---

## 3. Layering

**INV-LAY-1**: Decoder core (`src/qec/decoder/`) is isolated. No analysis, experiment, or bench module may modify decoder internals.

**INV-LAY-2**: Analysis layer is non-intrusive. All analysis functions are pure, side-effect free, and operate on copies only.

**INV-LAY-3**: No upward dependencies. Lower layers never import higher layers:
- decoder (L1) → channel (L2) → diagnostics (L3) → predictors (L4) → analysis (L5) → experiments (L6) → bench (L7)

**INV-LAY-4**: Diagnostics are observational only. They do not modify decoder inputs, BP messages, or any mutable state.

**INV-LAY-5**: Predictors produce signals only. They do not modify decoder or input state.

---

## 4. Monotonic Safety

**INV-MON-1**: Trajectory validation discourages degradation. Transitions that decrease score, increase energy, or reduce coherence receive validation scores < 1.0.

**INV-MON-2**: Cycle detection prevents oscillatory traps. Repeating patterns of length 2 or 3 in regime/strategy history trigger a multiplicative penalty ∈ [0.8, 1.0).

**INV-MON-3**: Bad-transition guardrail. Degrading transitions receive additional penalty via the trajectory validation layer.

**INV-MON-4**: Strategy consistency constraint penalizes rapid flipping between strategies.

---

## 5. Reproducibility

**INV-REP-1**: Runs are fully reproducible given the same inputs, seed, and software versions.

**INV-REP-2**: Artifacts are immutable after hashing. No mutation of experiment records post-creation.

**INV-REP-3**: Identity is independent of memory layout. Identical configurations produce identical outputs regardless of execution environment.

**INV-REP-4**: All experiment outputs include version, seed, and environment metadata sufficient for exact reproduction.

---

## 6. Compositional Safety

**INV-CMP-1**: The final scoring function is a product of bounded factors:

```
score = base * stability * transition * multi_step * modulation * cycle_penalty * trajectory_score
```

Each factor is individually bounded, therefore the product is bounded.

**INV-CMP-2**: No factor can drive the score negative. All multiplicative factors are non-negative.

**INV-CMP-3**: The neutral value for all optional factors is 1.0. Missing signals default to neutral, preserving existing behavior.

---

## Verification

Each invariant is enforced by at least one of:
- Unit tests (in `tests/`)
- Structural constraints (architecture enforcement)
- Runtime assertions (bounded checks)
- Determinism regression tests (byte-identical output verification)
