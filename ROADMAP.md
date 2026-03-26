QEC Roadmap

Deterministic QLDPC Stability, Adaptive Control & Theory Engine

Author: Trent Slade — QSOL-IMC
ORCID: 0009-0002-4515-9237

0. Purpose

This document defines the architectural direction and research trajectory of the QEC framework.

QEC has evolved beyond a decoder toolkit.

It is now a:

Deterministic Structural Analysis, Adaptive Control, and Theory-Building System for QLDPC Decoding

1. Governing Philosophy

QEC evolves under a strict hierarchy:

Determinism → Architecture → Measurement → Control → Adaptation → Theory

Rules:

Capability may expand
Determinism must never regress
The decoder remains sacred
All intelligence is externalized
2. Core Architectural Invariants (Non-Negotiable)
2.1 Determinism is Architecture

All outputs must be byte-identical under fixed configuration.

Required:

no hidden randomness
explicit seed control
deterministic ordering everywhere
canonical JSON serialization
SHA-256 artifact hashing
stable sweep ordering
runtime_mode = "off" ⇒ identical outputs across runs

Determinism is not a feature.
It is a constraint.

2.2 Decoder Core Protection

Location:

src/qec/decoder/

Rules:

no algorithm changes without explicit approval
no adaptive logic inside decoder
no stochastic behavior
no experimental leakage

The decoder is:

a fixed physical system under study

2.3 Layer Separation

Strict directional dependency:

decoder → channel → diagnostics → predictors → control → adaptation → theory

Constraints:

no upward imports
no circular dependencies
no cross-layer mutation
2.4 Minimalism

Prefer:

stdlib
NumPy / SciPy
sparse operators

Avoid:

frameworks
heavy abstractions
unnecessary dependencies
3. System Architecture (v105 State)

QEC now operates as a closed-loop reasoning and intervention system:

metrics
→ trajectory
→ geometry
→ diagnostics
→ differential diagnosis
→ provocation
→ revised diagnosis
→ treatment planning
→ invariant extraction
→ invariant registry
→ law formation
→ refinement → control
Layer 1 — Decoder Core
BP decoding (all variants)
deterministic scheduling
OSD / decimation

Status: frozen / protected

Layer 2 — Channels
deterministic LLR construction
pluggable noise models
Layer 3 — Diagnostics (Observation)

QEC as a measurement instrument

Includes:

attractors / basins
spectral metrics
instability measures
Layer 4 — Geometry & Trajectory Analysis
angular velocity
curvature
spiral dynamics
axis locking
coupling metrics

New direction:

trajectory = motion through an error landscape

Layer 5 — Differential Diagnosis

System explains its own behavior:

oscillatory traps
metastable plateaus
basin switching
control overshoot
Layer 6 — Provocation (Active Testing)

Controlled interventions to test hypotheses:

baseline treatments
response classification
diagnosis revision
Layer 7 — Treatment Planning

Deterministic intervention selection:

candidate generation
scoring
geometry-aware control
law-aware constraints
Layer 8 — Invariant Extraction

System identifies structural truths:

sign invariants
ordering invariants
topology invariants
geometry invariants
Layer 9 — Invariant Registry (Cross-Run Memory)

Tracks invariants across runs:

frequency
strength
streaks
break events
Layer 10 — Law Engine

Builds emergent laws:

stable laws
fragile laws
emerging laws
decaying laws

Includes:

stability scoring
lifecycle tracking
drift detection
Layer 11 — Theory Feedback

System uses learned laws to guide:

control decisions
diagnosis thresholds
strategy selection
4. New Research Directions (v105+)
4.1 Active Probe Diagnostics (Knowledge Landscape Mapping)

Inspired by learning-landscape research.

Goal:

infer system structure using minimal probes

Features:

short diagnostic runs
targeted perturbations
landscape reconstruction
4.2 Trajectory Singularity Analysis

Inspired by phase singularity physics.

Goal:

detect instability events as topological defects

Signals:

pre-switch acceleration
singularity pair formation
annihilation events
4.3 Treatment Transport Model

Inspired by quantum transfer physics.

Goal:

model intervention strength as landscape transport distance

Concepts:

local vs non-local transitions
driving-force control
basin skipping
4.4 Structural Stress & Defect Analysis

Goal:

detect hidden structural strain causing failure

Features:

stress hotspots
off-target behavior
defect propagation
4.5 Automata-Based Control System

Goal:

formalize system behavior as a deterministic state machine

States:

diagnose → provoke → revise → treat → verify → archive

Benefits:

provable transitions
clean TUI integration
safer control flow
4.6 Hardware Modality Profiles

Goal:

align QEC behavior with hardware constraints

Profiles:

superconducting (time/depth optimized)
neutral atom (connectivity/space optimized)
5. Evolution Strategy

QEC evolves in this order:

Measure → Understand → Diagnose → Test → Control → Adapt → Generalize → Formalize
6. Future Roadmap
v105.2 — Active Probe Engine
minimal diagnostic probes
landscape inference
adaptive probing
v105.3 — Trajectory Singularity Engine
instability event detection
pre-transition signals
v105.4 — Transport-Based Control
intervention distance modeling
basin-jump optimization
v105.5 — Control Automaton
formal state machine
provable control transitions
v105.6 — Hardware-Aware QEC
modality profiles
constraint-aware control
v106.0 — Human Interface Layer (Rust TUI)
full system navigation
zero-logic UI
CLI/JSON bridge
7. Anti-Patterns (Strictly Forbidden)
modifying decoder core
stochastic optimization
ML replacing structural reasoning
hidden state
non-deterministic control
8. Success Criteria

The system succeeds when it can:

diagnose its own failures
test its hypotheses
select optimal interventions
learn invariant structure
identify stable laws
adapt behavior deterministically
9. Final Principle

Truth is what survives repeated attempts to break it.

Author

Trent Slade
QSOL-IMC

ORCID: https://orcid.org/0009-0002-4515-9237
