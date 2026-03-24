# QSOLKCB / QEC — Deterministic Adaptive QLDPC System
## Current Project State Snapshot (v99.1)

Author: Trent Slade  
Organization: QSOL-IMC  
Philosophy: Determinism-first engineering and structural decoding research  

---

# Purpose of This File

This document provides a **current architectural snapshot** of the QEC system.

It exists to help:

- new contributors  
- research collaborators  
- automated agents (ChatGPT / Claude)  
- future versions of the author  

quickly understand:

- what the system currently is  
- what problems it solves  
- what invariants must not be broken  
- where the research is heading  

---

# Authoritative Project Documents

- README.md — overview and entry point  
- PROJECT_STATE.md — **current system state (this file)**  
- ROADMAP.md — future direction  
- CHANGELOG.md — version history  

Interpretation:

- PROJECT_STATE = what exists now  
- ROADMAP = what is coming  
- CHANGELOG = what already happened  

---

# Current System Identity (v99+)

QEC is no longer just a decoder toolkit.

It is now a:

# **Deterministic Adaptive Decision System for QLDPC Decoding**

---

## Core Loop

The system operates as a fully deterministic control loop:

```text
metrics → attractor → strategy → evaluation → adaptation → memory

Expanded:

sense → decide → evaluate → adapt → specialize
Key Properties
fully deterministic (no randomness)
fully reproducible (byte-identical runs)
fully interpretable (no black-box learning)
adaptive without machine learning
memory-driven strategy selection
System Architecture (v99)

The system is layered and strictly constrained.

Layer	Path	Role
1	src/qec/decoder/	Protected BP decoder
2	src/qec/channel/	Channel / LLR models
3	src/qec/diagnostics/	Observational signals
4	src/qec/predictors/	Pre-decode risk estimation
5	src/qec/analysis/	Metrics, attractors, strategy logic
6	src/qec/experiments/	Experiment orchestration
7	src/bench/	Benchmark harness
Layer Responsibilities
Decoder Layer (Protected)
belief propagation (BP)
deterministic post-processing
MUST remain bit-stable
Analysis Layer (Core Intelligence)

This is where most of the v98–v99 system lives.

Subsystems:
1. Field Metrics (v98.5)
phi alignment
symmetry / triality
curvature
resonance
complexity
2. Multiscale Metrics (v98.6)
scale consistency
scale divergence
multi-resolution structure
3. Attractor Analysis (v98.7)
regime classification:
stable
oscillatory
unstable
transitional
basin score
transition detection
4. Strategy Selection (v98.8)
deterministic scoring
regime-aware decisions
transition triggering
5. Strategy Evaluation (v98.9)
before/after comparison
improvement scoring
outcome classification:
stabilized
recovered
damped
regressed
6. Adaptation Layer (v99.0)
global feedback bias
trajectory scoring
recency-weighted performance
7. Strategy Memory (v99.1)
per-strategy history
bounded memory (cap = 10)
local biasing of strategy selection
What the System Now Does

The system can:

detect system state (via metrics)
classify behavior (via attractors)
choose actions (strategies)
evaluate outcomes
adapt future decisions
remember which strategies work
Crucially

It does all of this:

WITHOUT randomness
WITHOUT ML
WITHOUT hidden state
Research Position

The system has transitioned from:

decoder diagnostics

to:

deterministic adaptive control of decoding behavior
Original Research Insight (v3–v7)

Decoding failure is driven by:

Tanner graph structure
↓
spectral localization
↓
instability modes
↓
BP failure
Current Extension (v98–v99)

The system now explores:

state → intervention → feedback → adaptation

Meaning:

decoding is treated as a dynamical system
strategies act as control inputs
evaluation provides feedback signals
memory enables specialization
Current Capabilities
Deterministic Strategy System
multiple strategy types
composable actions
deterministic selection
Adaptive Feedback
evaluates improvement
biases future decisions
tracks trajectory performance
Memory System
strategy-specific performance
bounded historical tracking
specialization over time
Experiment Harness
deterministic input generation
structured reports
full trace visibility
Current Limitations

The system is intentionally constrained.

Not Yet Implemented
regime-specific memory
multi-step trajectory optimization
learned weight tuning
attractor-aware adaptation
Important

These are intentional omissions, not gaps.

The system prioritizes:

clarity > complexity
determinism > performance
structure > heuristics
Research Direction
Immediate (v99.x)
refine adaptation behavior
introduce regime-aware memory
improve transition intelligence
Near-Term
attractor-conditioned strategy selection
transition success modeling
structured decision policies
Long-Term
deterministic stability oracle
Tanner graph optimization via control signals
full decoding phase-space mapping
Test Suite Status
~6000+ tests passing
0 failures
deterministic validation across system

Legacy v3 tests have been archived and excluded.

Architectural Invariants

The following must never change without a major version bump:

decoder semantics
BP scheduling
deterministic outputs
artifact identity
schema compatibility

All advanced behavior must remain:

opt-in
externally applied
non-invasive
Determinism Anchor

Example configuration:

runtime_mode = "off"
seed = fixed
deterministic_metadata = True

Guarantee:

same input → same output → byte-identical
Project Philosophy

Small is beautiful.
Determinism is holy.
Stability is engineered.

No randomness.
No excuses.

Negative results are data.

Author

Trent Slade
QSOL-IMC

ORCID: https://orcid.org/0009-0002-4515-9237


👉 a system
