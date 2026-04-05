# QSOL QEC — Canonical Linux-Style Roadmap
Branch: long-running canonical deterministic systems line  
Current Stable Release: `v137.3.0`

This roadmap fully replaces all prior roadmap versions.

Versioning follows long-line Linux-style progression.

Major line remains **v137.x.x**

Future architecture evolves through **minor series**

Example:

- `v137.4.x`
- `v137.5.x`
- `v137.6.x`

This preserves release continuity and replay lineage.

---

# CORE LAW

Determinism is architecture.

**same input = same bytes**

Mandatory invariants:

- frozen dataclasses
- canonical JSON / bytes export
- stable SHA-256 replay identity
- deterministic ordering
- deterministic tie-breaking
- fail-fast validation
- append-only provenance
- schema migration contracts
- immutable event history

`src/qec/decoder/` remains sacred infrastructure.

No roadmap arc may violate this.

---

# ACTIVE CANONICAL ARC

---

## v137.4.x — Sovereignty Kernel + Cryptographic Audit

Mandatory governance substrate.

Focus:

- append-only event history
- cryptographic audit
- capability boundaries
- signed provenance
- deterministic replay law

Planned line:

### v137.4.0
Merkle-linked event history kernel

### v137.4.1
capability + privilege boundary engine

### v137.4.2
signed provenance artifact chain

### v137.4.3
policy decision artifact + evidence receipts

### v137.4.4
replay-law certification battery

---

## v137.5.x — Autonomous Planning + Control Synthesis

Focus:

- deterministic search
- route planning
- world-state control
- bounded planning IR

Planned line:

### v137.5.0
plan IR + deterministic search kernel

### v137.5.1
route graph execution runtime

### v137.5.2
dead-end pruning + stable tie-breaking

### v137.5.3
policy-constrained planner

### v137.5.4
planning replay battery

---

## v137.6.x — Hierarchical Memory + State Compaction

Mandatory bridge release.

Focus:

- context compression
- memory hierarchy
- evidence linkage
- bounded retrieval

Planned line:

### v137.6.0
raw → episode hierarchy

### v137.6.1
semantic → theme memory compaction

### v137.6.2
hash-preserving compression chain

### v137.6.3
uncertainty-gated retrieval

### v137.6.4
memory fidelity benchmark battery

---

## v137.7.x — Deterministic NLP Governance v2

Evolution of the completed NLP layer.

Focus:

natural language → intent lattice → governed planner → replay-safe output

Planned line:

### v137.7.0
intent lattice v2

### v137.7.1
semantic graph compiler

### v137.7.2
privilege-aware response planner

### v137.7.3
semantic risk classifier

### v137.7.4
governance evidence battery

---

## v137.8.x — Unified World Model Kernel

Focus:

- persistent graph
- time-series mutation log
- causal state evolution
- bounded traversal

Planned line:

### v137.8.0
typed world graph

### v137.8.1
time-series mutation kernel

### v137.8.2
bounded traversal API

### v137.8.3
future-state projection engine

### v137.8.4
world-state replay certification

---

## v137.9.x — Multimodal Physics-Aware Reasoning

Focus:

- multimodal ingestion
- geometry kernels
- observational photonic fast path
- deterministic fixed-point wrappers

Planned line:

### v137.9.0
multimodal feature schema

### v137.9.1
geometry + polytope reasoning layer

### v137.9.2
fixed-point compensation wrappers

### v137.9.3
observational photonic fast path

### v137.9.4
cross-hardware divergence battery

---

# NEXT LONG ARC

---

## v137.10.x — Scientific Reasoning Kernel

Focus:

- hypothesis lattice
- experiment DSL
- theorem bridge
- proof-carrying artifacts

---

## v137.11.x — Distributed Governance Kernel

Focus:

- multi-node orchestration
- signed replication
- transparency logs
- federated trust chain

---

## v137.12.x — Unified Intelligence Substrate

Focus:

- planning
- memory
- language
- world modeling
- scientific reasoning
- distributed governance

This becomes the journal / preprint platform.

---

# FORMAL METHODS LINE

Parallel milestone series.

---

## v137.20.x — TLA+ + model checking
- replay law verification
- state transition proofs
- distributed convergence

---

## v137.21.x — Lean 4 invariant pack
- hash chain proofs
- schema migration proofs
- lattice monotonicity

---

# BENCHMARK LAW

Every release must pass:

- byte identity battery
- replay battery
- schema migration battery
- governance adversary suite
- planning determinism suite
- memory fidelity suite
- cross-hardware divergence suite
