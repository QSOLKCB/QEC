# QSOLKCB / QEC
Deterministic Structural Analysis & Adaptive Control System  
for LDPC / QLDPC Tanner Graphs

[![Release](https://img.shields.io/github/v/release/QSOLKCB/QEC)](https://github.com/QSOLKCB/QEC/releases)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19099503.svg)](https://doi.org/10.5281/zenodo.19099503)
[![Authorea](https://img.shields.io/badge/Authorea-10.22541%2Fau.177376131.17346095%2Fv1-blue)](https://doi.org/10.22541/au.177376131.17346095/v1)

[![Type](https://img.shields.io/badge/type-deterministic%20analysis%20%2B%20adaptive%20control-blue)]()
[![Engine](https://img.shields.io/badge/engine-structure--driven-lightblue)]()
[![Determinism](https://img.shields.io/badge/determinism-bitwise%20reproducible-success)]()
[![Mode](https://img.shields.io/badge/mode-no%20stochastic%20search-critical)]()
[![Architecture](https://img.shields.io/badge/architecture-measure%20%E2%86%92%20control%20%E2%86%92%20adapt-purple)]()

[![License](https://img.shields.io/badge/license-CC--BY--4.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

---

QEC is a deterministic research framework for studying:

- belief propagation (BP) dynamics  
- Tanner graph structure  
- spectral instability  
- phase behavior  
- adaptive control of decoding strategies  

It functions as:

- 🧠 **A deterministic analysis system**
- 🌌 **A phase-space reconstruction engine**
- ⚙️ **An adaptive control loop (v98+)**

---

# 🧠 What QEC Actually Is (Now)

QEC is not just a simulator.

It is a:

> **Deterministic Structural + Adaptive System for Decoding Dynamics**

The system operates as a closed loop:


metrics → attractor → strategy → evaluation → adaptation → memory


Everything is:

- deterministic  
- explainable  
- reproducible  
- externally controlled (decoder untouched)  

---

# 🚀 Core Capabilities (v99)

---

## 1. Structural Diagnostics (Foundation)

- BP trajectory analysis  
- attractor / basin detection  
- oscillation & metastability metrics  
- free-energy landscape analysis  

---

## 2. Spectral Analysis (Graph Physics Layer)

- non-backtracking spectrum  
- eigenvector localization (IPR)  
- trapping-set candidate detection  
- spectral instability metrics  

---

## 3. Phase & Regime Analysis

- deterministic phase diagrams  
- transition detection  
- regime segmentation  
- phase boundary metrics  

---

## 4. Strategy System (Decision Layer)

- deterministic strategy scoring  
- regime-aware selection  
- structured transition logic  

---

## 5. Evaluation Framework

- before/after comparison  
- outcome classification  
- improvement scoring  

---

## 6. Adaptive Layer (v99)

- trajectory-based feedback  
- global bias adjustment  
- recency-weighted performance  

---

## 7. Strategy Memory (v99.1)

- bounded per-strategy memory  
- specialization via historical performance  
- deterministic biasing (no randomness)  

---

# 🌌 Phase-Space + Control System

QEC reconstructs both:

### Phase Structure
- regimes  
- boundaries  
- degeneracy  
- transitions  

### Behavioral Dynamics
- strategy effectiveness  
- adaptation patterns  
- system response  

---

# ⚙️ System Architecture


Tanner Graph
↓
Diagnostics (metrics)
↓
Attractor Classification
↓
Strategy Selection
↓
Evaluation
↓
Adaptation
↓
Memory
↓
System Behavior


---

# 🔁 Determinism Guarantees

QEC enforces strict reproducibility:

- no hidden randomness  
- deterministic ordering everywhere  
- canonical JSON outputs  
- stable multi-key ranking  
- explicit seeded RNG only  

```python
import numpy as np
np.random.RandomState(seed)

If it cannot be reproduced byte-for-byte, it is not a result.

🔬 Invariant Framework

QEC is built on explicit, testable invariants.

Example:

QSOL-BP-INV-001
URW(min-sum, ρ = 1.0) ≡ baseline min-sum

Properties:

analytically justified
empirically validated
bitwise exact
📊 Research Applications

QEC enables:

decoding phase diagram reconstruction
spectral instability analysis
trapping-set identification
deterministic inverse design
strategy optimization without randomness
reproducible computational experiments
📖 Documentation

- [INSTALL.md](INSTALL.md) — Setup and installation
- [QUICKSTART.md](QUICKSTART.md) — One-command demo
- [USAGE_GUIDE.md](USAGE_GUIDE.md) — Workflow and entry points
- [ARCHITECTURE.md](ARCHITECTURE.md) — System architecture and design

⚡ Quick Start
Install
pip install -e .
Run the demo
python scripts/qec_demo.py
Minimal diagnostic
from qec.diagnostics.bp_dynamics import compute_bp_dynamics_metrics

out = compute_bp_dynamics_metrics(llr_trace, energy)
print(out["metrics"])
🧠 Design Philosophy

Small is beautiful.
Determinism is essential.
Structure over heuristics.
Measurement before control.

📚 Citation

Trent Slade — QSOL-IMC
QEC: Deterministic Structural Analysis & Adaptive Control Framework

ORCID: https://orcid.org/0009-0002-4515-9237

👤 Author

Trent Slade
QSOL-IMC


---
