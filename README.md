# QSOLKCB / QEC

### Deterministic Invariant-Driven Discovery Engine for LDPC & QLDPC Tanner Graphs

[![Release](https://img.shields.io/github/v/release/QSOLKCB/QEC?label=release)](https://github.com/QSOLKCB/QEC/releases/tag/v68.5.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19099503.svg)](https://doi.org/10.5281/zenodo.19099503)
[![DOI](https://img.shields.io/badge/Authorea-10.22541%2Fau.177376131.17346095%2Fv1-blue)](https://doi.org/10.22541/au.177376131.17346095/v1)
[![Type](https://img.shields.io/badge/type-deterministic%20research%20framework-blue)]()
[![License](https://img.shields.io/badge/license-CC--BY--4.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

QEC is a deterministic research framework for studying belief-propagation (BP)
dynamics, spectral structure, and Tanner-graph geometry in LDPC and QLDPC
codes.

The system functions as a **deterministic discovery engine and experimental
laboratory**, combining spectral diagnostics, decoding dynamics, and formally
validated invariants to explore how graph structure governs decoding behavior.

Unlike traditional LDPC simulation toolkits, QEC emphasizes:

* **determinism over stochastic search**
* **invariants over heuristics**
* **reproducibility over approximation**

Every experiment, discovery trajectory, and diagnostic result is **bitwise
reproducible**.

---

# 🧠 What Makes QEC Different

QEC is not just a simulator — it is a:

> **Deterministic Tanner-Graph Discovery + Invariant Validation System**

The framework integrates three core capabilities:

### 1. Spectral Discovery Engine

* explores Tanner-graph space via deterministic mutation
* detects spectral basins and phase boundaries
* reconstructs phase diagrams of decoding behavior

### 2. BP Dynamics Diagnostics

* analyzes belief-propagation stability and attractor structure
* measures oscillation, convergence, and instability regimes
* links spectral signals to decoding outcomes

### 3. Invariant-Driven Optimization (NEW)

* discovers and validates structural invariants
* eliminates redundant computation safely
* preserves **bitwise identity** and deterministic execution

Recent releases introduce **formally proven invariants** that transform runtime
behavior without altering results.

---

# 🔬 Invariant Framework (v68 Series)

QEC now includes a growing registry of formally validated invariants:

### QSOL-BP-INV-001 — Algebraic Identity (v68.4.1)

URW(min-sum, ρ = 1.0) ≡ baseline min-sum

→ eliminates redundant decoder execution paths

### QSOL-BP-INV-002 — Trace-Indexed Data Reuse (v68.5.0)

sign(vᵢ), CRC(vᵢ) are pure functions of trace index

→ eliminates redundant per-metric computation (75% reduction)

These invariants are:

* analytically justified
* empirically validated
* test-saturated
* bitwise exact

📄 Formal documents and DOIs are linked in Releases.

---

# ⚙️ Discovery Engine Architecture

The system operates as a layered pipeline:

Tanner Graph Generation
↓
Structural Diagnostics
↓
Spectral Diagnostics
↓
BP Dynamics Analysis
↓
Invariant Detection (NEW)
↓
Mutation Plugin Registry
↓
Mutation Operators
↓
Local Graph Optimization
↓
Discovery Archive
↓
Spectral Basin Detection
↓
Spectral Ridge Detection
↓
Phase Map Reconstruction
↓
Phase-Guided Exploration
↓
Phase Novelty Discovery
↓
Phase Characterization
↓
Spectral Theory Synthesis

This architecture enables QEC to:

* analyze structure
* explore graph space
* extract patterns
* synthesize theory

---

# 🌌 Spectral Phase-Space Analysis

QEC reconstructs the geometry of Tanner-graph space through:

### Spectral Basins

Regions of similar decoding behavior

### Spectral Ridges

Phase boundaries separating decoding regimes

### Phase Maps

Global structure of decoding stability

### Discovery Trajectories

Paths taken through spectral space

### Phase-Guided Exploration

Directed search toward under-explored regimes

### Phase Novelty Detection

Identification of previously unseen graph structures

### Phase Characterization

Automatic classification of decoding regimes

### Spectral Conjecture Synthesis

Extraction of candidate theoretical relationships

---

# 🔁 Deterministic Experiment Design

QEC enforces strict determinism:

* no hidden randomness
* deterministic mutation ordering
* deterministic decoder scheduling
* reproducible experiment artifacts
* identical outputs across runs

Randomness must be explicit:

```python
np.random.RandomState(seed)
```

Same seed → identical results.

---

# 📊 Research Applications

QEC enables research into:

* belief-propagation attractor geometry
* trapping-set dynamics
* spectral fragility of Tanner graphs
* decoding stability prediction
* LDPC / QLDPC code discovery
* phase-space structure of decoding
* invariant-driven optimization
* reproducible computational systems

The framework acts as a:

> **Deterministic experimental lab for inference dynamics in sparse graphical models**

---

# 📁 Key Project Files

CLAUDE.md — Development guardrails
CHANGELOG.md — Release history
INV.md — Invariant registry
PROJECT_STATE.md — Architecture snapshot
ROADMAP.md — Research direction

⚡ Quick Start
### 1) Install (editable)
```bash
pip install -e .
2) Run a minimal diagnostic
from qec.diagnostics.bp_dynamics import compute_bp_dynamics_metrics

llr_trace = [...]  # list/array of float64 vectors
energy = [...]     # optional, same length as trace

out = compute_bp_dynamics_metrics(llr_trace, energy)
print(out["metrics"])  # MSI, CPI, TSL, GOS, BTI, ...
3) Determinism (required)
import numpy as np
np.random.RandomState(0)  # explicit seed if randomness is used
4) Reproduce release behavior
git checkout v68.5.0
pytest -q
5) Invariants

See INV.md for the invariant registry:

INV-001 (v68.4.1): algebraic identity (URW ρ=1.0)

INV-002 (v68.5.0): trace-indexed sign/CRC reuse

Notes

Bitwise deterministic under float64

No hidden randomness

Same input → identical output


---

# 🧭 Design Philosophy

Small is beautiful.
Determinism is essential.
Invariants over heuristics.
Transparent systems over opaque ones.

Negative results are data.

---

# 📚 Citation

If you use QEC in research, please cite:

Trent Slade
QSOL-IMC

**QEC: Deterministic Invariant-Driven Discovery Framework for Tanner Graph Dynamics**

ORCID
[https://orcid.org/0009-0002-4515-9237](https://orcid.org/0009-0002-4515-9237)

---

# 👤 Author

Trent Slade
QSOL-IMC
