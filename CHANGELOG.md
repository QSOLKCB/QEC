# Changelog

All notable changes to this project are documented in this file.

This project follows Semantic Versioning (SemVer).

---

## Latest Development State (v98–v99)

The project has evolved from a spectral diagnostics toolkit into a:

**Deterministic Adaptive Decision System for QLDPC Decoding**

Recent development introduces:

- deterministic field metrics and multiscale analysis  
- attractor classification and regime detection  
- strategy selection and evaluation pipeline  
- adaptive feedback and trajectory scoring  
- bounded strategy memory and specialization  

The system now operates as a deterministic control loop:


metrics → attractor → strategy → evaluation → adaptation → memory


All features remain:

- deterministic  
- opt-in  
- non-invasive to the decoder core  

---


👉 This gives instant context without rewriting history.

🚀 2. New Entry (v99.x series — ADD THIS ABOVE v50)

Drop this right after the header:

[99.1.0] — Deterministic Adaptive Strategy Memory

Added

- Introduces per-strategy bounded memory (cap=10) for tracking recent performance.
- Enables local specialization: strategy selection now biased by historical success.
- Deterministic memory updates with recency weighting.
- Memory remains fully observable and JSON-serializable.

---

[99.0.0] — Deterministic Adaptation Layer

Added

- Introduces global adaptation layer for feedback-driven strategy adjustment.
- Adds trajectory scoring and recency-weighted performance tracking.
- Enables deterministic biasing of future decisions based on past outcomes.
- Fully opt-in and non-invasive to baseline behavior.

---

[98.9.0] — Strategy Evaluation Framework

Added

- Deterministic before/after strategy evaluation pipeline.
- Outcome classification: stabilized, recovered, damped, regressed.
- Improvement scoring and structured evaluation artifacts.

---

[98.8.0] — Strategy Selection System

Added

- Deterministic strategy scoring and selection engine.
- Regime-aware decision logic driven by attractor classification.
- Transition-triggered strategy switching.

---

[98.7.0] — Attractor Analysis System

Added

- Deterministic attractor classification:
  - stable
  - oscillatory
  - unstable
  - transitional
- Basin scoring and transition detection.
- Structured attractor-state representation for downstream control.

---

[98.6.0] — Multiscale Field Metrics

Added

- Multiscale analysis of system state across resolutions.
- Metrics for scale consistency and divergence.
- Enables detection of cross-scale instability patterns.

---

[98.5.0] — Field Metrics Framework

Added

- Deterministic field metrics:
  - phi alignment
  - symmetry / triality
  - curvature
  - resonance
  - complexity
- Provides foundational signal layer for system state detection.

---
