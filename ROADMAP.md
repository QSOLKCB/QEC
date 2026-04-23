# QSOLKCB / QEC — EXECUTION ROADMAP (Post-v143.5)

---

## Core Law (Global)

```text
same input
→ same ordering
→ same canonical JSON
→ same stable hash
→ same bytes
```

All modules MUST:

- be deterministic
- be replay-safe
- be bounded
- fail fast on invalid input
- produce canonical artifacts
- avoid randomness, wall-clock dependence, and implicit async
- preserve decoder immutability
- remain analysis-layer only unless explicitly proven otherwise

---

## Current State

QEC has completed the full SPHAERA arc.

```text
QEC
→ distributed proof system
→ bounded hardware feedback mesh
→ autonomous recovery
→ IRIS invariant runtime
→ SPHAERA formal invariant geometry runtime
→ proof artifact
```

### v143.5 — SPHAERA Proof Artifact (COMPLETE)

```text
end-to-end execution
→ cross-domain demonstration
→ canonical outputs
→ reproducible evidence
```

This closes the original roadmap.

The next roadmap is not a restart of core architecture.
It is a research-to-application roadmap built on top of the completed deterministic runtime.

---

## Strategic Position

QEC is now a:

```text
deterministic execution system
+ invariant-driven reduction engine
+ geometric structure runtime
+ ensemble-consistent decision system
+ proof-carrying receipt architecture
```

The next direction is:

```text
proof
→ demonstration
→ research
→ application
```

This roadmap therefore focuses on a new **analysis-only companion layer**:

# v144.x — SCOL (State-Conditioned Orchestration Layer)

---

## Purpose

Extend QEC with a deterministic orchestration and benchmarking layer that:

```text
consumes SPHAERA receipts
→ classifies current operating state
→ scores admissible filter/control orderings
→ detects recurrence and oscillation
→ applies bounded refinement
→ validates outcomes through deterministic stress coverage
→ emits canonical orchestration receipts
```

This layer is **not** a decoder change.
It is **not** a stochastic controller.
It is **not** a probabilistic runtime.

It begins as:

```text
bounded application layer
+ benchmarking layer
+ observability layer
```

Only after benchmark proof may it be promoted to an optional orchestration kernel.

---

## SCOL Law

All orchestration modules must satisfy:

```text
state
→ deterministic classification
→ admissible ordering generation
→ deterministic score ranking
→ canonical tie-breaking
→ bounded correction
→ receipt generation
→ stable-hash lineage
```

### Imported Concept Translation

- **Markov matrices** → deterministic transition score matrix (TSM)
- **Newton refinement** → bounded deterministic correction loop (BDCL)
- **Hash-gated replay** → native canonical lineage and replay identity
- **Kasiski spacing analysis** → recurrence / periodicity detector over quantized traces
- **Monte Carlo** → rejected in live form, replaced by deterministic low-discrepancy stress lattices

---

## Formal Runtime State

The orchestration state is derived from completed QEC/SPHAERA artifacts:

```text
S_t = (
  invariant_signature,
  geometry_class,
  ensemble_consistency_regime,
  spectral_regime,
  hardware_consensus_class
)
```

The state must be:

- finite
- canonical
- receipt-derived
- replay-safe
- bounded in representation size

---

## Admissible Ordering Domain

SCOL does not invent arbitrary actions.
It ranks only admissible permutations over validated control/filter families such as:

- thermal control
- latency stabilization
- timing mesh alignment
- power modulation
- hardware consensus bridging
- SPHAERA-derived routing constraints

---

## Implementation Principle

The research converges on the following staged strategy:

```text
start simple
→ prove determinism
→ benchmark aggressively
→ add recurrence analysis
→ add bounded refinement
→ integrate consensus
→ publish evidence
→ only then consider optional promotion
```

That means:

- **SDDS** class ideas first (dwell-aware deterministic scheduling)
- **DLDSL** infrastructure early (deterministic stress coverage)
- **PASKA** next (periodicity-aware orchestration)
- **NRSS** after evidence (bounded Newton-style refinement)
- **HSAOM** only after all prior stages are proven and stable

---

# v144.x — SCOL FOUNDATION ARC

---

## v144.0 — Periodicity Structure Kernel

### Purpose

Establish deterministic recurrence and oscillation detection over quantized hardware traces.

### Model

```text
quantized trace
→ motif detection
→ spacing analysis
→ GCD period extraction
→ oscillation / resonance classification
→ canonical periodicity receipt
```

### Primary Module

- `src/qec/analysis/periodicity_structure_kernel.py`

### Inputs

- bounded thermal trace window
- bounded latency trace window
- bounded timing trace window
- bounded power trace window
- optional consensus trace window
- policy thresholds

### Outputs

- `PeriodicityStructureReceipt`

### Metrics [0,1]

- recurrence_strength
- periodicity_confidence
- oscillation_risk
- motif_stability

### Classification

- aperiodic
- weakly_periodic
- strongly_periodic
- oscillatory
- resonant

### Role

Provides the first deterministic warning system for repeated instability motifs.

---

## v144.1 — Stress Lattice Engine

### Purpose

Replace Monte Carlo-style exploration with deterministic low-discrepancy stress coverage.

### Model

```text
bounded input domain
→ Sobol / Halton lattice generation
→ deterministic stress envelopes
→ coverage validation
→ canonical stress case receipts
```

### Primary Modules

- `src/qec/analysis/stress_lattice_engine.py`
- `src/qec/analysis/low_discrepancy_sequence_tools.py`

### Inputs

- thermal bounds
- latency bounds
- timing bounds
- power bounds
- lattice policy

### Outputs

- `StressLatticeReceipt`
- deterministic stress case packs

### Metrics [0,1]

- coverage_uniformity
- boundary_pressure
- scenario_diversity
- replay_stability

### Role

Creates the canonical deterministic benchmark generator for all later SCOL work.

---

## v144.2 — State-Conditioned Filter Mesh

### Purpose

Introduce deterministic state-conditioned ordering proposals over admissible control/filter sequences.

### Model

```text
state signature
→ admissible ordering set
→ transition score matrix
→ deterministic ranking
→ canonical ordering proposal
```

### Primary Module

- `src/qec/analysis/state_conditioned_filter_mesh.py`

### Inputs

- invariant / geometry / ensemble / spectral / hardware state signature
- admissible filter/control set
- static published weights
- optional dwell counters

### Outputs

- `StateConditionedFilterMeshReceipt`

### Metrics [0,1]

- invariant_alignment
- ensemble_stability
- spectral_coherence
- hardware_demand_alignment
- ordering_confidence

### Classification

- thermal_dominant
- latency_critical
- timing_skewed
- power_unbalanced
- consensus_fragile
- balanced

### Role

Creates deterministic ordering proposals without stochastic selection.

---

## v144.3 — Deterministic Transition Policy

### Purpose

Harden transition ranking into a formal policy layer with canonical tie-breaking and dwell-aware penalties.

### Model

```text
state
+ candidate orderings
+ dwell history
+ periodicity signals
→ deterministic policy score
→ argmax selection
→ canonical transition receipt
```

### Primary Module

- `src/qec/analysis/deterministic_transition_policy.py`

### Inputs

- filter mesh receipt
- periodicity receipt
- dwell state
- policy configuration

### Outputs

- `DeterministicTransitionPolicyReceipt`

### Metrics [0,1]

- transition_score
- dwell_penalty
- recurrence_penalty
- route_stability
- policy_confidence

### Role

Implements the first full **SDDS-style** scheduler in lawful QEC form.

---

## v144.4 — Bounded Refinement Kernel

### Purpose

Add bounded Newton-style local correction after coarse deterministic selection.

### Model

```text
selected ordering
+ residual estimate
+ bounded sensitivity model
→ fixed-iteration correction loop
→ clamped refinement
→ canonical refinement receipt
```

### Primary Module

- `src/qec/analysis/bounded_refinement_kernel.py`

### Inputs

- deterministic transition policy receipt
- bounded residual vector
- bounded sensitivity estimate
- damping policy

### Outputs

- `BoundedRefinementReceipt`

### Metrics [0,1]

- residual_pressure
- correction_strength
- convergence_score
- fallback_safety

### Classification

- no_refinement
- corrected
- saturated
- fallback

### Role

Introduces **NRSS-style** correction while preserving fixed iteration ceilings and fail-fast rollback.

---

## v144.5 — Filter Order Consensus Bridge

### Purpose

Bridge orchestration proposals across redundant nodes using canonical lineage and deterministic aggregation.

### Model

```text
node orchestration receipts
→ replay identity verification
→ deterministic aggregation
→ conflict accounting
→ consensus ordering receipt
```

### Primary Module

- `src/qec/analysis/filter_order_consensus_bridge.py`

### Inputs

- transition policy receipts
- refinement receipts
- lineage hashes
- hardware consensus context

### Outputs

- `FilterOrderConsensusBridgeReceipt`

### Metrics [0,1]

- consensus_confidence
- cross_node_agreement
- conflict_pressure
- replay_integrity

### Role

Locks SCOL into multi-node deterministic agreement without mutating SPHAERA core behavior.

---

## v144.6 — Deterministic Benchmark Battery

### Purpose

Prove that SCOL is lawful, bounded, and useful under deterministic replay.

### Benchmark Families

- thermal boundary scenarios
- latency drift and jitter scenarios
- timing alignment scenarios
- power imbalance scenarios
- cross-node consensus stress scenarios
- synthetic oscillation / recurrence scenarios

### Required Assertions

- byte-identical replay
- stable ordering rankings
- deterministic tie-breaking
- bounded correction behavior
- zero stochasticity leakage
- reproducible stress lattice generation
- recurrence detector correctness
- consensus stability under identical inputs

### Primary Modules

- `tests/test_periodicity_structure_kernel.py`
- `tests/test_stress_lattice_engine.py`
- `tests/test_state_conditioned_filter_mesh.py`
- `tests/test_deterministic_transition_policy.py`
- `tests/test_bounded_refinement_kernel.py`
- `tests/test_filter_order_consensus_bridge.py`

### Role

This release is the proof wall.
SCOL does not advance without passing it.

---

## v144.7 — SCOL Evaluation Pack

### Purpose

Package the first complete research-grade evaluation of SCOL as a bounded companion layer.

### Deliverables

- comparative design summary: PDMSM vs SDDS vs PASKA vs NRSS
- benchmark result tables
- replay certification summary
- stress lattice coverage report
- recurrence detection quality report
- integration notes for optional promotion path

### Output

- `SCOLEvaluationPackReceipt`
- markdown research summary
- paper-ready figures / tables (if desired)

### Role

Marks the end of the foundation arc and the decision point for promotion.

---

# v145.x — OPTIONAL PROMOTION ARC

**This arc is conditional.**
It only proceeds if v144.x proves that SCOL improves stability or orchestration quality without any violation of determinism, boundedness, or replay identity.

---

## v145.0 — PASKA Integration

### Purpose

Promote recurrence-aware scheduling into the live companion policy.

### Result

```text
state-conditioned policy
+ periodicity penalties
→ oscillation-aware deterministic routing
```

---

## v145.1 — NRSS Comparative Promotion

### Purpose

Compare coarse-only policy vs bounded refinement policy and formalize acceptance thresholds.

### Result

```text
baseline policy
vs
refined policy
→ deterministic promotion decision
```

---

## v145.2 — Companion Orchestration Runtime

### Purpose

Bundle validated SCOL modules into a stable analysis companion runtime for hardware-control recommendation.

### Constraint

Still analysis-only.
Still optional.
Still no decoder mutation.

---

## v145.3 — HSAOM Feasibility Review

### Purpose

Decide whether the full Hybrid SPHAERA-Aware Orchestration Model is justified.

### Decision Rule

Proceed only if:

- all v144.x replay proofs hold
- benchmark gains are material
- periodicity false positives remain bounded
- refinement divergence remains zero under policy limits
- interface boundaries with SPHAERA stay clean

If not, stop here and preserve SCOL as a bounded companion layer.

---

## v145.4 — Research Publication / Documentation Release

### Purpose

Finalize the first SCOL publication and public-facing documentation.

### Possible Outputs

- formal paper draft
- README updates
- docs / diagrams
- benchmark appendix
- deterministic replay certification summary

---

# Guardrails for the Entire New Roadmap

## Never Allowed

- stochastic sampling in live execution
- PRNG-driven control routing
- wall-clock dependent behavior
- floating-point dependent consensus decisions
- implicit memory or hidden state mutation
- decoder-core modifications
- architectural overlap that mutates SPHAERA semantics

## Required Mitigations

- quantization rules must be explicit and canonical
- tie-breaking must be canonical and published
- refinement loops must be bounded and rollback-safe
- recurrence detection must resist false periodicity aliasing
- state cardinality must remain bounded and reviewable
- all receipts must serialize canonically and hash stably

---

# Final Strategic Direction

```text
v143.5
→ completed deterministic runtime proof

v144.x
→ deterministic orchestration research + benchmarking foundation

v145.x
→ optional promotion to companion runtime and publication
```

This roadmap keeps QEC faithful to its law:

```text
prove first
→ benchmark second
→ promote only if lawful
```

That is the post-SPHAERA path.
