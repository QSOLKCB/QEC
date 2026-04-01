# QSOLKCB / QEC — Deterministic Supervisory Control Framework
## Current Project State Snapshot (v132.x)

**Author:** Trent Slade  
**Organization:** QSOL-IMC  
**ORCID:** 0009-0002-4515-9237  

**Philosophy:**  
Determinism-first engineering, formal supervisory control, structural QLDPC reasoning

---

# Purpose of This File

This document provides the **current authoritative system-state snapshot** of the QEC framework.

It exists to help:

- new contributors
- research collaborators
- automated agents (ChatGPT / Claude / Codex)
- future versions of the author

quickly understand:

- what the system currently is
- what modules already exist
- what architectural invariants must not be broken
- what the active control stack currently does
- what the protected boundaries are

---

# Authoritative Project Documents

- `README.md` — overview and entry point
- `PROJECT_STATE.md` — **what exists now**
- `ROADMAP.md` — future direction
- `CHANGELOG.md` — historical release evolution

Interpretation:

```text
PROJECT_STATE = what currently exists
ROADMAP = what comes next
CHANGELOG = what already happened
Current System Identity (v132.x)

QEC is no longer a decoder toolkit.

It is no longer only an adaptive controller.

It is now a:

Deterministic Supervisory Control, Verification, and Theory Framework for QLDPC Decoding

This is now a closed-loop supervisory reasoning system.

Primary loop:

sense
→ diagnose
→ control
→ supervise
→ verify
→ explain
→ learn laws

Expanded:

metrics
→ topology
→ control decision
→ supervisory transition
→ temporal verification
→ policy memory
→ feedback
→ explainability
→ law formation
Core System Properties

The system is:

fully deterministic
fully reproducible
byte-identical under fixed configuration
formally auditable
memory-aware
supervisory-state driven
fail-safe capable
non-ML adaptive

Strict exclusions:

no randomness
no async
no black-box learning
no hidden mutable state
System Architecture (Current)

The system is layered and strictly constrained.

Layer	Path	Role
1	src/qec/decoder/	Protected BP decoder
2	src/qec/channel/	Channel / LLR models
3	src/qec/diagnostics/	Observational metrics
4	src/qec/analysis/	Supervisory intelligence
5	src/qec/experiments/	Controlled experiments
6	src/bench/	deterministic harness
Layer Responsibilities
Layer 1 — Decoder Core (Protected)

Location:

src/qec/decoder/

This remains the protected physical decoding system.

Includes:

belief propagation
deterministic scheduling
OSD / decimation
protected decode semantics

This layer is sacred.

Must remain bit-stable.

No supervisory logic may enter this layer.

Layer 2 — Channels

Includes:

deterministic LLR generation
pluggable noise models
modality-specific signal profiles

Fully deterministic.

Layer 3 — Diagnostics

QEC as a measurement system.

Includes:

field metrics
multiscale metrics
attractor analysis
topology analysis
graph controllability
invariant extraction
risk fusion

This layer converts raw system behavior into stable signals.

Layer 4 — Supervisory Intelligence (Core)

This is where the modern QEC control framework lives.

This is now the dominant system layer.

Existing Supervisory Modules

Current implemented modules include:

invariant proving engine
law engine
hybrid automata
graph controllability
safety state automata
threshold hysteresis controller
NetworkX topology analysis
invariant fusion engine
adaptive policy orchestrator
policy memory engine
policy feedback controller
adaptive supervisory controller
temporal transition verifier

This is the current production supervisory stack.

Current Control Stack

The control stack now operates as:

diagnostics
→ fused risk
→ policy selection
→ supervisory transition
→ temporal legality check
→ memory feedback
→ state evolution

This is now a deterministic supervisory control system.

Current Supervisory Capabilities

The system can currently:

detect risk conditions
fuse multi-layer invariants
select control policies
escalate safely
trigger fail-safe states
enforce absorbing safe states
verify temporal legality
detect oscillatory supervisory patterns
track bounded policy memory
adapt future decisions deterministically
provide audit-safe replay

This is a major evolution beyond the v99 adaptive layer.

Current Safety Guarantees

Implemented guarantees include:

fail-safe precedence
absorbing safe_mode
absorbing escalation_lock
temporal transition legality
deterministic hysteresis anti-flap
bounded policy memory
explicit recovery precedence
no hidden stochastic drift

These are now core system guarantees.

Current Research Position

QEC has transitioned from:

decoder diagnostics

to:

formal supervisory control framework

Current research direction:

control synthesis
temporal verification
hybrid automata
formal theorem integration
explainable supervision
Current Known Limits

Not yet implemented:

DFA controller synthesis
DPDA stack supervisors
theorem prover bridges
symbolic execution
SAT / SMT control proofs
hybrid continuous-time control

These are active roadmap items.

Current Test Status

Current expectation:

all supervisory modules fully covered by deterministic pytest suite

Must maintain:

zero nondeterministic failures
stable schemas
replay-identical results
Architectural Invariants

The following must never change without major roadmap approval:

decoder semantics
deterministic outputs
stable schemas
supervisory transition contracts
fail-safe precedence
temporal legality semantics
Determinism Anchor

Canonical guarantee:

same input
→ same supervisory trace
→ same output
→ same bytes

This is absolute.

Project Philosophy
Small is beautiful.
Determinism is holy.
Safety is sacred.
Supervision must be provable.

No randomness.

No excuses.

Negative results are data.

Final System Identity

QEC is now:

a deterministic supervisory control framework for QLDPC systems

not merely a decoder toolkit.
