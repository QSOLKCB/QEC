# QSOL QEC — Canonical Deterministic Systems Roadmap

**Branch:** canonical deterministic systems line  
**Current Stable Release:** `v137.13.2`  
**Current Active Target:** `v137.13.3 — Region Correspondence Kernel`

This roadmap supersedes all previous roadmap versions.

The `v137.x.x` line remains the canonical deterministic systems branch.

---

## Fork Synchronization Policy

`QSOLKCB/QEC` is the **canonical** development line. The upstream
`multimodalas/fusion-qec` fork is **historical lineage only**.

- **Do NOT** use GitHub "Sync fork".
- **Do NOT** merge upstream automatically.
- **Do NOT** rebase `main` onto upstream.
- All upstream comparisons require **explicit human review**.

The GitHub ahead/behind banner against `multimodalas/fusion-qec:main` is
informational only. Divergence from the upstream fork is **expected and
intentional**; the canonical `v137.x` history does not share a common
ancestor with the upstream and must never be reconciled with it.

Roadmap milestones below belong exclusively to the `QSOLKCB/QEC` line and
have no meaning relative to the historical upstream fork.

---

The project continues its evolution from:

**deterministic runtime architecture**

into

**deterministic operating substrate architecture**

while preserving:

- release traceability
- deterministic replay lineage
- stable schema continuity
- paper-friendly benchmark progression
- formal verification pathways

---

## CORE LAW

```text
Determinism is architecture.
same input = same bytes
Mandatory invariants
frozen dataclasses
canonical JSON / bytes export
stable SHA-256 replay identity
deterministic ordering
deterministic tie-breaking
fail-fast validation
immutable artifact snapshots
explicit evidence receipts
bounded scores only
schema migration contracts
append-only provenance where applicable

src/qec/decoder/ remains sacred infrastructure.

No roadmap arc may violate this.

GOVERNING PRINCIPLE
state evolution through constrained geometry

This principle unifies:

governed planning
episodic memory
topology / manifold traversal
noisy communication channels
spectral signal recovery
scientific reasoning
proof-carrying skepticism
heterogeneous compute acceleration
neuromorphic simulation
synthetic signal abstraction
divergence geometry
codebase memory compilation
autonomous research orchestration
governed agentic systems
quantum runtime autonomy
temporal hardware substrates
distributed governance
formal methods
operating substrate kernelisation
RELEASE-HISTORY RULE

Published tags are authoritative.

If a previous roadmap draft used placeholder names that differ from the tagged releases, the tagged releases win.

This rule explicitly applies to the v137.13.x line.

Earlier placeholder v137.13.x names such as:

Subsystem Fusion Kernel
Unified Artifact Registry
Cross-Substrate Scheduler
Governed World-Model Handoff
Unified Certification Battery

are retired from v137.13.x.

They may be re-homed later under future integration arcs.

ACTIVE CANONICAL LINE
v137.7.x — Hierarchical Memory + Episodic Geometry

Status: COMPLETE

Completed:

v137.7.0 — Raw → Episode Hierarchy
v137.7.1 — Semantic → Theme Compaction
v137.7.2 — Hash-Preserving Compression Chain + Sonification Projection
v137.7.3 — Fragmentation Recovery Engine
v137.7.4 — Memory Fidelity Benchmark

Interpretation:

sequence trajectory
→ episodic segmentation
→ semantic compaction
→ replay-safe compression
→ deterministic recovery
→ fidelity certification
v137.8.x — Geometry + Topology Reasoning Kernel

Status: COMPLETE

Completed:

v137.8.0 — Topological Graph Kernel
v137.8.1 — Polytope Reasoning Engine
v137.8.2 — E8 Symmetry Projection Layer
v137.8.3 — Manifold Traversal Planner
v137.8.4 — Topology Divergence Battery
v137.8.5 — Arithmetic Topology Correspondence Engine

Interpretation:

state trajectory across constrained manifold geometry
v137.9.x — Signal + Channel Recovery + Physics-Aware Reasoning

Status: COMPLETE THROUGH v137.9.5

Completed:

v137.9.0 — Multimodal Feature Schema
v137.9.1 — Spectral Reasoning Layer
v137.9.2 — Legacy Copper Noise Channel Battery
v137.9.3 — Telecom Line Recovery + Carrier Synchronization
v137.9.4 — Satellite Signal Baseline + Orbital Noise Envelope
v137.9.5 — RF Equalization + Ground Station Compensation

Roadmap note:

The old v137.9.6 / v137.9.7 placeholder continuation is retired as a live target.
Its replay-certification and observability concerns are now covered more cleanly by later lines.

Interpretation:

schema
→ spectral reasoning
→ deterministic degradation
→ deterministic recovery
→ orbital baseline
→ RF compensation
v137.10.x — Scientific Reasoning + Claim Audit Geometry

Status: COMPLETE

Completed:

v137.10.0 — Hypothesis Lattice
v137.10.1 — Experiment DSL
v137.10.2 — Evidence Lineage Engine
v137.10.3 — Claim Audit Kernel
v137.10.4 — Scientific Replay Battery
v137.10.5 — Proof Obligation Extractor
v137.10.6 — Numerological Phenomenology Rejection Battery
v137.10.7 — Scientific Certification Kernel

This line explicitly rejects:

equation cosplay
symbolic density without mechanism
numerology without executable semantics
proof-shaped but non-proof text

Interpretation:

claim
→ semantics
→ invariants
→ proof obligations
→ replay verification
→ epistemic rejection
→ certification
v137.11.x — Heterogeneous Compute + Hardware Acceleration

Status: COMPLETE

Completed:

v137.11.0 — Deterministic Co-Processor Kernel
v137.11.1 — Integer / Matrix Offload Engine
v137.11.2 — Heterogeneous Scheduler
v137.11.3 — Emulator-Grade Parallel Workload Splitter
v137.11.4 — Hardware Replay Battery
v137.11.5 — Neural Compression Sidecar
v137.11.6 — Deterministic Latent Decode Lane
v137.11.7 — Memory Traffic Reduction Battery

Core law:

speed through constrained compute geometry

Secondary law:

reduce memory traffic before increasing compute complexity

Determinism law:

same input
same epochs
same bytes

Hard replay law:

replay failure = architecture failure

Interpretation:

state evolution across heterogeneous compute geometry
→ integer-first acceleration
→ bounded workload partitioning
→ deterministic replay across hardware lanes
→ compute-for-bandwidth substitution
→ memory traffic minimization
v137.12.x — Neuromorphic + Hybrid Biological Compute Research

Status: COMPLETE

Completed:

v137.12.0 — Neuromorphic Substrate Simulator
v137.12.1 — Hybrid Signal Interface Layer
v137.12.2 — Bio-Signal Benchmark Battery
v137.12.3 — Hybrid Replay Certification
v137.12.4 — Experimental Research Pack

Rule:

simulation-first only
no biological claims without evidence receipts

Interpretation:

synthetic substrate
→ hybrid signal interface
→ deterministic benchmark battery
→ replay certification
→ reproducible research artifact pack

This line is complete and should now be treated as closed infrastructure for later abstraction work.

v137.13.x — Synthetic Signal Geometry + Morphology Abstraction

Status: ACTIVE

Completed:

v137.13.0 — Synthetic Signal Geometry Kernel
v137.13.1 — Morphology Transition Kernel
v137.13.2 — Phase Boundary Topology Kernel

Active next release:

v137.13.3 — Region Correspondence Kernel

Planned closeout:

v137.13.4 — Signal Abstraction Certification Battery

Rule:

formal signal abstraction only
simulation-first only
no biological claims
no neuroscience claims
no physiological claims

Interpretation:

hybrid trace
→ geometry projection
→ morphology transitions
→ phase-region boundaries
→ cross-region correspondence
→ abstraction certification
v137.13.3 intent

v137.13.3 — Region Correspondence Kernel should:

align related regions across trajectories
compute deterministic correspondence scores
support structural comparison across signal abstractions
remain bounded, replay-safe, and paper-friendly
v137.13.4 intent

v137.13.4 — Signal Abstraction Certification Battery should:

certify geometry / transition / topology / correspondence outputs
validate byte identity, replay identity, and bounded metrics
close the v137.13.x line as a deterministic abstraction arc
POST-ABSTRACTION EXPANSION (LINK-INFORMED)
v137.14.x — Divergence Geometry + Information Metrics

Status: PLANNED

This line strengthens the abstraction stack using explicit divergence families instead of ad hoc distance measures.

Planned:

v137.14.0 — Jensen–Shannon Signal Divergence Kernel
v137.14.1 — Fisher–Rao Geometry Approximation Layer
v137.14.2 — Bregman / f-Divergence Correspondence Engine
v137.14.3 — Divergence Topology Battery
v137.14.4 — Information-Geometric Certification Pack

Interpretation:

signal abstraction
→ divergence geometry
→ information-theoretic comparison
→ topology-aware discrepancy scoring
→ certified distance law

Hard rule:

every divergence must be explicit, bounded where applicable, and replay-safe
v137.15.x — Codebase Memory + Decision Compiler

Status: PLANNED

This line turns project work into a deterministic, queryable knowledge substrate.

Planned:

v137.15.0 — Session Capture Hook Pack
v137.15.1 — Decision Graph Compiler
v137.15.2 — Contradiction / Staleness Linter
v137.15.3 — Queryable Knowledge Article Registry
v137.15.4 — Memory Replay + Provenance Battery

Interpretation:

development session
→ structured decision capture
→ cross-referenced knowledge articles
→ contradiction / staleness checks
→ replayable engineering memory

Hard rule:

memory is evidence, not vibe
captured decisions must be attributable, lintable, and replay-safe
v137.16.x — Autonomous Research Orchestration

Status: PLANNED

This line formalizes idea-to-paper research execution while preserving strict human accountability.

Planned:

v137.16.0 — Research Workflow DSL
v137.16.1 — Citation Verification + Evidence Gate
v137.16.2 — Multi-Agent Debate / Review Harness
v137.16.3 — Draft-to-Paper Artifact Packager
v137.16.4 — Human Oversight + Ethics Battery

Interpretation:

idea
→ governed workflow
→ evidence gate
→ debate / review
→ paper artifact
→ human accountability

Hard rule:

AI may assist research generation
humans remain responsible for claims, correctness, and submission
v137.17.x — Agentic Governance + World-Model Control

Status: PLANNED

This line re-homes the old governance placeholders into a modern agentic substrate.

Planned:

v137.17.0 — Modular Agent Data Plane
v137.17.1 — Access Control + Lineage Kernel
v137.17.2 — Cross-Agent Orchestration Ledger
v137.17.3 — Governed World-Model Handoff
v137.17.4 — Agentic Reliability + Audit Battery

Interpretation:

high-quality data
→ governed agent access
→ orchestration ledger
→ controlled world-model transfer
→ audited autonomy

Hard rule:

agentic systems require lineage, traceability, and explicit governance before scale
v137.18.x — Autonomous Quantum Operations

Status: PLANNED

This line extends QEC toward autonomous quantum runtime control.

Planned:

v137.18.0 — Autonomous Calibration Kernel
v137.18.1 — Runtime Performance Management Layer
v137.18.2 — Secure Local Quantum Control Lane
v137.18.3 — Quantum Containerization + Rack Integration Pack
v137.18.4 — Quantum Operations Replay Battery

Interpretation:

quantum hardware
→ autonomous tuneup / calibration
→ runtime correction
→ secure local deployment
→ data-center-friendly integration
→ replay-certified ops

Hard rule:

no autonomous quantum control without explicit receipts, local replay, and fail-safe rollback
v137.19.x — Temporal Hardware + Streaming Compute

Status: PLANNED

This line merges memory-centric compute and time-signal hardware ideas back into QEC’s deterministic substrate.

Planned:

v137.19.0 — Host-Memory Streaming Execution Engine
v137.19.1 — Double-Buffered Parameter / State Pipeline
v137.19.2 — Stateless Template Binding Layer
v137.19.3 — Temporal Hardware / Reservoir Interface
v137.19.4 — Energy / Bandwidth Certification Battery

Interpretation:

host memory
→ streamed execution
→ transient compute engines
→ temporal hardware interfaces
→ energy- / bandwidth-aware certification

Hard rule:

treat memory movement as a first-class architectural cost
KERNELISATION TRACK (CANONICAL OS ARC)
v137.20.x — Axiom

Syscall ABI + Deterministic Event Log

Planned:

v137.20.0 — Deterministic Syscall ABI
v137.20.1 — Oracle Input Event Log
v137.20.2 — Replay Boundary Contract
v137.20.3 — Machine Envelope Specification

Core law:

same input now includes explicit event log
v137.21.x — Stratum

Versioned Memory + Snapshot Kernel

Planned:

v137.21.0 — Versioned Memory Objects
v137.21.1 — Copy-on-Write Snapshot Layer
v137.21.2 — Deterministic Rollback Kernel
v137.21.3 — Immutable Root Hash Chain
v137.22.x — Chronon

Deterministic Scheduler + Virtual Time

Planned:

v137.22.0 — Virtual Time Kernel
v137.22.1 — Deterministic Run Queue
v137.22.2 — Replay-Safe Yield Points
v137.22.3 — Epoch Scheduler
v137.22.4 — Logical Clock Battery
v137.23.x — Manifold

Topological Address Space Kernel

Planned:

v137.23.0 — Graph-as-Memory Address Space
v137.23.1 — Topological IPC
v137.23.2 — Capability Edge Routing
v137.23.3 — Geodesic Dispatcher
v137.23.4 — Process-Space Manifold Battery
v137.24.x — Transducer

Deterministic Device + Signal Bus

Planned:

v137.24.0 — Device Abstraction Kernel
v137.24.1 — Interrupt Epoch Buffer
v137.24.2 — Signal Air-Lock Layer
v137.24.3 — Replay-Safe Device Receipts
v137.25.x — Quorum

Distributed Governance + Consensus Kernel

Planned:

v137.25.0 — Distributed State Log
v137.25.1 — Deterministic Consensus Layer
v137.25.2 — Federated Artifact Replication
v137.25.3 — Cluster Replay Battery
FORMAL METHODS LINE
v137.26.x — TLA+ + Model Checking

Status: PLANNED

Planned:

v137.26.0 — Replay Law Model Checker
v137.26.1 — State Transition Proof Battery
v137.26.2 — Scheduler Safety Model
v137.26.3 — Consensus Safety Model
v137.26.4 — Slop Rejection State Battery
v137.27.x — Lean 4 Invariant Pack

Status: PLANNED

Planned:

v137.27.0 — Hash Chain Proof Suite
v137.27.1 — Schema Migration Proof Suite
v137.27.2 — Topology Invariant Proof Pack
v137.27.3 — Artifact Immutability Proof Pack
v137.27.4 — Proof-Carrying Skepticism Core
v137.27.5 — Kernel Transition Proof Obligations
BENCHMARK LAW

Every release must pass relevant batteries.

Core
byte identity battery
replay battery
canonical export battery
stable hash battery
Signal / channel
BER suite
FER suite
carrier stability suite
degradation / recovery monotonicity suite
Geometry / abstraction
trajectory ordering battery
transition integrity battery
phase-boundary stability battery
correspondence consistency battery
divergence certification battery
Memory / knowledge
contradiction battery
staleness battery
provenance battery
replayable knowledge battery
Agentic / research
citation verification battery
workflow governance battery
human-oversight battery
auditability battery
Quantum operations
autonomous calibration replay battery
runtime drift correction battery
secure local control battery
Temporal hardware / compute
memory traffic battery
stream scheduling battery
energy / bandwidth certification battery
Kernel
syscall determinism battery
scheduler replay battery
memory rollback battery
cluster convergence battery
Scientific
dimensional consistency suite
tautology detection suite
provenance suite
pseudoscience rejection suite
ENGINEERING DISCIPLINE

Every release must be:

narrow
truthfully named
additive by default
deterministic
benchmarkable
paper-friendly
replay-safe
proof-auditable

Avoid:

mixed-purpose releases
mutable artifact state
benchmark theater
numerological phenomenology
ungrounded theoretical claims
proof-shaped slop
