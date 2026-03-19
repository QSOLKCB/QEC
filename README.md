QSOLKCB / QEC
Deterministic Invariant-Driven Discovery & Phase Analysis Engine for LDPC / QLDPC Tanner Graphs

[![Release](https://img.shields.io/github/v/release/QSOLKCB/QEC?label=release)](https://github.com/QSOLKCB/QEC/releases/tag/v84.0.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19099503.svg)](https://doi.org/10.5281/zenodo.19099503)
[![Authorea](https://img.shields.io/badge/Authorea-10.22541%2Fau.177376131.17346095%2Fv1-blue)](https://doi.org/10.22541/au.177376131.17346095/v1)

[![Type](https://img.shields.io/badge/type-deterministic%20phase%20analysis%20framework-blue)]()
[![Engine](https://img.shields.io/badge/engine-invariant--driven-lightblue)]()
[![Determinism](https://img.shields.io/badge/determinism-bitwise%20reproducible-success)]()
[![Architecture](https://img.shields.io/badge/architecture-inverse%20design%20%E2%86%92%20phase%20maps-purple)]()
[![Mode](https://img.shields.io/badge/mode-no%20stochastic%20search-critical)]()

[![License](https://img.shields.io/badge/license-CC--BY--4.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

QEC is a deterministic research framework for studying belief-propagation (BP) dynamics, spectral structure, and Tanner-graph geometry in LDPC and QLDPC codes.

It functions as both:

🧠 A deterministic discovery engine
🌌 A phase-space analysis system for invariant behavior

The system now supports end-to-end invariant analysis:

target → candidate structures → ranking → transitions → regimes

All results are bitwise reproducible.

🧠 What Makes QEC Different

QEC is not just a simulator — it is a:

Deterministic Invariant Discovery + Phase Analysis System

Unlike traditional LDPC toolkits, QEC emphasizes:

determinism over stochastic search

invariants over heuristics

interpretability over optimization

phase structure over raw metrics

🚀 Core Capabilities (v84)

QEC now operates as a complete analysis stack:

1. Hybrid Co-Design Engine (v82)

evaluates joint (theta, sequence) configurations

extracts invariant structure from decoding behavior

supports deterministic hybrid scoring

2. Inverse Design Engine (v82.8)
target_behavior → (theta, sequence)

bounded candidate space

deterministic ranking

invariant-aligned scoring

3. Parametric Target System (v83.0)

Targets are now programmable:

{
  "desired_class": "stable",
  "min_stability": 0.8,
  "weight_phase": 0.3
}

→ transforms inverse design into an

invariant query language

4. Target Sweep Engine (v83.1)

evaluates multiple targets in a single run

enables exploration of intent space

produces structured sweep datasets

5. Phase Boundary Detection (v83.2)

detects where dominant structures change

identifies regime transitions

6. Transition Metrics (v83.3)

measures boundary strength

distinguishes:

class-driven transitions

phase-driven transitions

degenerate transitions

7. Phase Map Summary (v83.4)

compresses sweep into a phase signature:

number of transitions

strength of boundaries

transition types

8. Regime Extraction (v84.0)

segments sweep into stable regions

extracts:

dominant structures

dominant class/phase

mean behavior

→ completes the pipeline:

🧠 Discrete Phase Diagram Construction (Deterministic)

🌌 Phase-Space Analysis (Modern QEC)

QEC now reconstructs invariant structure across intent space:

Phase Boundaries

Where dominant structures change

Regimes

Stable regions of consistent behavior

Transition Strength

Sharp vs smooth changes

Degeneracy

Multiple structures with identical score

Phase Signatures

Compact summaries of system behavior

⚙️ System Architecture (v84)
Target Specification (Parametric)
↓
Candidate Generation (Bounded)
↓
Hybrid Co-Design Evaluation
↓
Invariant Scoring
↓
Deterministic Ranking
↓
Target Sweep
↓
Transition Detection
↓
Transition Metrics
↓
Transition Summary
↓
Regime Extraction
↓
Phase Structure
🔁 Determinism Guarantees

QEC enforces strict reproducibility:

no hidden randomness

deterministic ordering everywhere

deterministic ranking (multi-key)

invariant-preserving transformations

identical outputs across runs

Randomness must be explicit:

np.random.RandomState(seed)
🔬 Invariant Framework

QEC includes formally validated invariants:

QSOL-BP-INV-001 — Algebraic Identity

URW(min-sum, ρ = 1.0) ≡ baseline min-sum

QSOL-BP-INV-002 — Trace-Indexed Data Reuse

sign(vᵢ), CRC(vᵢ) are pure functions of trace index

Properties:

analytically justified

empirically validated

test-saturated

bitwise exact

📊 Research Applications

QEC now enables:

phase diagram reconstruction of BP dynamics

regime analysis in Tanner graph space

invariant-driven code discovery

decoding stability prediction

trapping-set and attractor analysis

deterministic inverse design

reproducible computational experiments

⚡ Quick Start
Install
pip install -e .
Minimal diagnostic
from qec.diagnostics.bp_dynamics import compute_bp_dynamics_metrics

out = compute_bp_dynamics_metrics(llr_trace, energy)
print(out["metrics"])
Determinism
import numpy as np
np.random.RandomState(0)
🧠 Design Philosophy

Small is beautiful.
Determinism is essential.
Invariants over heuristics.
Structure over noise.

If it cannot be reproduced, it is not a result.

📚 Citation

Trent Slade
QSOL-IMC

QEC: Deterministic Invariant-Driven Discovery & Phase Analysis Framework

ORCID
https://orcid.org/0009-0002-4515-9237

👤 Author

Trent Slade
QSOL-IMC
