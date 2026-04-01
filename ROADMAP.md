# QEC ROADMAP
## Deterministic Supervisory Control, Formal Verification & Theory Engine
### Canonical Architecture Roadmap — v132.x Era

**Author:** Trent Slade — QSOL-IMC  
**ORCID:** 0009-0002-4515-9237

---

# 0. Purpose

This document defines the canonical architectural direction of the QEC framework.

QEC has evolved beyond a decoder toolkit and beyond a diagnostic-only system.

It is now a:

**Deterministic Supervisory Control, Structural Analysis, Adaptive Control, and Formal Verification Framework for QLDPC Decoding**

The framework now operates as a closed-loop control and reasoning engine.

Primary mission:

```text
measure → diagnose → verify → control → supervise → prove → explain
1. Governing Philosophy

QEC evolves under strict architectural law:

Determinism
→ Safety
→ Architecture
→ Verification
→ Control
→ Adaptation
→ Theory
→ Explainability

Rules:

capability may expand
determinism must never regress
safety always dominates performance
decoder remains sacred
all intelligence externalized
all decisions auditable
all transitions formally verifiable
2. Core Architectural Invariants (NON-NEGOTIABLE)
2.1 Determinism is Architecture

All outputs must be byte-identical under fixed configuration.

Required:

no hidden randomness
explicit seed control
deterministic ordering everywhere
canonical JSON serialization
SHA-256 artifact hashing
stable sweep ordering
replay-identical outputs
no async nondeterminism
frozen dataclasses preferred

Determinism is not a feature.

It is a hard architectural constraint.

2.2 Decoder Core Protection

Protected location:

src/qec/decoder/

Rules:

no algorithm changes without explicit approval
no adaptive logic inside decoder
no stochastic behavior
no supervisory logic leakage
no theory-engine mutation

Decoder is:

fixed physical system under study
2.3 Layer Separation

Strict dependency order:

decoder
→ channel
→ diagnostics
→ control
→ supervisory
→ verification
→ explainability
→ theory

Constraints:

no upward imports
no circular dependencies
no cross-layer mutation
no hidden state sharing
2.4 Minimalism

Prefer:

stdlib
NumPy
SciPy
sparse operators
frozen dataclasses
TypedDict schemas

Avoid:

heavy frameworks
unnecessary dependencies
hidden runtime systems
magic abstractions
3. System Architecture (v132.x Canonical State)

QEC now operates as a deterministic closed-loop supervisory framework:

metrics
→ diagnosis
→ control
→ supervisory state machine
→ temporal verification
→ formal proof
→ policy memory
→ feedback
→ explanation
→ law formation
Layer 1 — Decoder Core

Protected and frozen.

Includes:

BP decoding
OSD / decimation
deterministic scheduling
fixed physical decoder semantics

Status:

sacred / protected
Layer 2 — Diagnostics

QEC as a measurement instrument.

Includes:

attractors
basins
topology
graph controllability
instability metrics
invariant fusion
Layer 3 — Control Layer

Deterministic intervention systems.

Includes:

hysteresis controller
policy controller
adaptive policy orchestrator
memory-aware feedback
safety automata
Layer 4 — Supervisory Control

Formal supervisory control system.

Includes:

adaptive supervisory controller
temporal transition verifier
supervisory mode switching
escalation locks
fail-safe latching
dwell-time guards
Layer 5 — Formal Verification

Machine-verifiable control correctness.

Includes:

theorem-proof generation
Coq / Lean integration
TLA+ supervisory models
temporal proof checks
transition legality proofs
Layer 6 — Explainability

Every decision must be inspectable.

Includes:

immutable audit trails
policy memory history
anomaly explanation
forensic control tracing
decision replay
Layer 7 — Theory Engine

System extracts laws from stable supervisory behavior.

Includes:

invariant registry
temporal laws
topology laws
stability laws
supervisory behavior laws
4. Active Roadmap (v132.x)
v132.1.0 — DFA Controller Synthesis

Goal:

deterministic Ramadge-Wonham controller synthesis

Includes:

DFA synthesis
controllability proofs
nonblocking verification
deterministic synthesis gate
v132.2.0 — DPDA Supervisory Memory

Goal:

nested supervisory memory systems

Includes:

stack-based recovery
rollback-safe control memory
bounded stack depth
deterministic pushdown control
v132.3.0 — Dwell-Time & Switching Control

Goal:

prevent supervisory chatter and oscillation

Includes:

dwell-time guards
hysteresis filters
safe switching
bounded-time fail-safe
v132.4.0 — Formal Proof Integration

Goal:

machine-verifiable supervisory correctness

Includes:

Coq proof generation
Lean theorem wrappers
proof registries
verification caches
v132.5.0 — Explainable Supervisory Control

Goal:

world-class operator transparency

Includes:

immutable audit trails
forensic anomaly detection
decision explainability
operator replay tools
5. World-Class Extensions (v133+)
v133.x — Hybrid Automata

continuous + discrete supervisory systems

v134.x — Adversarial Supervisory Games

robust control under disturbance

v135.x — Symbolic Execution / Model Checking

bounded trace verification
SAT / SMT integration
counterexample synthesis

6. Evolution Strategy

QEC now evolves under:

Measure
→ Diagnose
→ Control
→ Supervise
→ Verify
→ Explain
→ Formalize
→ Generalize
7. Strict Anti-Patterns

## Rust TUI Status (Post-v132.4)

The Rust TUI is now considered **feature-stable infrastructure**.

Future modifications are restricted to:

* critical bug fixes
* installer compatibility
* terminal rendering corrections
* release version synchronization
* security / dependency maintenance

All future intelligence, supervisory logic, simulation, proof systems, and research functionality remain Python-engine side.

The UI must remain a zero-logic render surface.

Forbidden:

decoder mutation
stochastic optimization
hidden mutable state
nondeterministic control
ML replacing formal reasoning
unverifiable supervisory transitions
8. Success Criteria

The system succeeds when it can:

diagnose failures
select deterministic interventions
supervise safely
prove transitions
explain decisions
replay behavior identically
learn stable laws
scale supervisory logic formally
9. Final Principle
Truth is what survives repeated attempts to break it.
Safety is what survives repeated attempts to destabilize it.
Determinism is what survives repeated attempts to replay it.

Trent Slade
QSOL-IMC
ORCID: https://orcid.org/0009-0002-4515-9237
