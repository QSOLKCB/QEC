# QEC Roadmap

Deterministic QLDPC Stability & Adaptive Control Platform

Author: Trent Slade — QSOL-IMC  
ORCID: 0009-0002-4515-9237  

---

## 0. Purpose

This document defines the **architectural direction and research trajectory** of the QEC framework.

QEC is no longer just a decoder toolkit.

It is evolving into a:

> **Deterministic Structural Analysis + Adaptive Control System for QLDPC Decoding**

---

## 1. Governing Philosophy

QEC evolves under a strict hierarchy:


Determinism → Architecture → Measurement → Control → Adaptation


Rules:

- Capability may expand  
- Determinism must never regress  
- The decoder remains sacred  
- All intelligence is externalized  

---

## 2. Core Architectural Invariants (Non-Negotiable)

### 2.1 Determinism is Architecture

All outputs must be **byte-identical** under fixed configuration.

Required:

- no hidden randomness  
- explicit seed control  
- deterministic ordering everywhere  
- canonical JSON serialization  
- SHA-256 artifact hashing  
- stable sweep ordering  


runtime_mode = "off" ⇒ identical outputs across runs


Determinism is not a feature.  
It is a constraint.

---

### 2.2 Decoder Core Protection

Location:


src/qec/decoder/


Rules:

- no algorithm changes without explicit approval  
- no adaptive logic inside decoder  
- no stochastic behavior  
- no experimental leakage  

The decoder is:

> **a fixed physical system under study**

---

### 2.3 Layer Separation

Strict directional dependency:


decoder → channel → diagnostics → predictors → control → adaptation


Constraints:

- no upward imports  
- no circular dependencies  
- no cross-layer mutation  

---

### 2.4 Minimalism

Prefer:

- stdlib  
- NumPy / SciPy  
- sparse operators  

Avoid:

- frameworks  
- heavy abstractions  
- unnecessary dependencies  

---

## 3. System Architecture (v99 State)

QEC now operates as a **deterministic control loop**:


metrics → attractor → strategy → evaluation → adaptation → memory


---

### Layer 1 — Decoder Core

- BP decoding (all variants)
- deterministic scheduling
- OSD / decimation
- invariant-safe execution

Status: **frozen / protected**

---

### Layer 2 — Channels

- deterministic LLR construction  
- syndrome-only inference  
- pluggable models  

Future:

- AWGN  
- erasure  
- stim-compatible noise  

---

### Layer 3 — Diagnostics (Observation)

Transforms QEC into a **measurement instrument**

Includes:

- iteration dynamics  
- basin / attractor detection  
- spectral analysis (NB, Bethe Hessian)  
- localization (IPR)  
- instability metrics  

Rules:

- observational only  
- no side effects  

---

### Layer 4 — Predictors (Pre-Decode Intelligence)

Goal:

> predict failure before decoding

Inputs:

- spectral signals  
- structural metrics  
- attractor features  

Outputs:

- failure risk  
- instability classification  

---

### Layer 5 — Strategy System (Decision Layer)

Goal:

> choose what to do next

Components:

- strategy scoring  
- deterministic selection  
- regime-aware policies  

Examples:

- damping  
- scheduling changes  
- perturbation strategies  

---

### Layer 6 — Evaluation (Feedback Layer)

Goal:

> measure impact of actions

Includes:

- before/after comparison  
- outcome classification  
- improvement scoring  

---

### Layer 7 — Adaptation (Global Learning)

Goal:

> adjust future behavior

Mechanisms:

- trajectory scoring  
- recency weighting  
- global bias updates  

Constraint:

- deterministic  
- no stochastic learning  

---

### Layer 8 — Memory (v99+)

Goal:

> local specialization per strategy

Features:

- bounded memory per strategy  
- performance tracking  
- bias generation  

Effect:

> system becomes **experience-aware without randomness**

---

## 4. Spectral Research Foundation (Preserved)

Core hypothesis:


cycle structure
→ eigenvector localization
→ instability modes
→ BP failure


Key signals:

- non-backtracking spectral radius  
- dominant eigenvector  
- inverse participation ratio (IPR)  
- edge sensitivity  

Edge proxy:


s(e) ≈ |v_i|² · |v_j|²


These remain the **physical grounding layer** of the system.

---

## 5. Current System Identity

QEC is now:

> A deterministic, explainable, adaptive system  
> built on top of a fixed decoding substrate

Not:

- a heuristic optimizer  
- a stochastic search system  
- a black-box ML pipeline  

---

## 6. Near-Term Roadmap (v100+)

### v100 — Strategy Ecology

- strategy taxonomy formation  
- cross-regime dominance mapping  
- specialization patterns  

Outcome:

> system develops **behavioral structure**

---

### v101 — Memory Refinement

- decay functions  
- long vs short horizon memory  
- stability-aware weighting  

---

### v102 — Predictive Control

- pre-selection of strategies before execution  
- multi-step planning  
- failure avoidance policies  

---

### v103 — System-Level Phase Diagrams

Axes:

- instability  
- strategy selection  
- adaptation response  

Goal:

> map full **system dynamics**, not just decoder dynamics  

---

### v104 — Meta-Stability Detection

- detect false convergence  
- distinguish:
  - true stability  
  - metastability  
  - oscillation  

---

### v105 — Structural + Adaptive Fusion

Combine:

- spectral signals  
- adaptive behavior  

Goal:

> unified structural + behavioral model  

---

## 7. Anti-Patterns (Strictly Forbidden)

- modifying decoder core  
- stochastic optimization  
- ML replacing structural reasoning  
- dense NB matrix construction  
- breaking QLDPC constraints  
- hidden state or implicit randomness  

---

## 8. Evolution Strategy

QEC evolves in this order:


Measure → Understand → Control → Adapt → Specialize


Never:


Guess → Optimize → Hope


---

## 9. Success Criteria

The system succeeds when it can:

- predict failure before decoding  
- explain *why* failure occurs  
- select corrective strategies deterministically  
- improve outcomes reproducibly  
- maintain full auditability  

---

## 10. Final Principle

> If it cannot be reproduced byte-for-byte, it is not a baseline.

---

## Author

Trent Slade  
QSOL-IMC  

ORCID: https://orcid.org/0009-0002-4515-9237
