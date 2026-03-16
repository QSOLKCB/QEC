# QSOLKCB / QEC
### Deterministic Spectral Discovery Engine for LDPC and QLDPC Tanner Graphs

QEC is a deterministic research framework for studying belief-propagation dynamics,
spectral stability, and Tanner-graph structure in sparse graphical codes.

The system functions as a **spectral discovery engine** capable of evolving LDPC
and QLDPC parity-check graphs using structural diagnostics, spectral signals,
and deterministic mutation operators.

Rather than relying on stochastic evolutionary search, QEC performs **fully
deterministic graph exploration**, allowing Tanner-graph discovery experiments
to be reproduced exactly across machines and runs.

---

## Project Overview

QEC has evolved into a **spectral phase-space exploration and theory generation
system for Tanner graphs**.

Recent versions introduce a layered discovery architecture capable of:

- detecting **spectral basins** in Tanner-graph space
- identifying **phase boundaries via ridge detection**
- reconstructing **spectral phase diagrams**
- tracing **discovery trajectories across phase space**
- guiding exploration toward **under-explored phases**
- actively searching for **entirely new spectral phases**
- automatically **characterizing newly discovered phases**
- synthesizing **analytic conjectures from phase data**

This transforms the discovery engine from a mutation-based search system into a
**deterministic experimental laboratory for Tanner-graph phase structure and
decoding theory discovery**.

The framework can now:

- map decoding stability landscapes
- analyze belief-propagation attractor geometry
- reconstruct spectral phase diagrams
- steer discovery toward unexplored structural regimes
- detect candidate **new decoding phases**
- characterize spectral regimes automatically
- generate candidate **theoretical relationships between spectral metrics and decoding behavior**

All experiments remain **fully deterministic**, ensuring exact reproducibility
of discovery trajectories and experimental artifacts.

[![Release v62.0.0](https://img.shields.io/badge/release-v62.0.0-blue)](https://github.com/QSOLKCB/QEC/releases/tag/v62.0.0)
[![Research Framework](https://img.shields.io/badge/type-research%20framework-blue)]
[![License: CC BY 4.0](https://img.shields.io/badge/license-CC--BY--4.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

---

# QEC
### Deterministic Spectral Discovery for LDPC / QLDPC Tanner Graphs

QEC is a deterministic research framework for studying belief-propagation
dynamics and Tanner graph structure in classical LDPC and quantum QLDPC codes.

The framework combines:

- spectral diagnostics
- decoding dynamics analysis
- deterministic mutation operators
- spectral phase-space analysis
- automated phase discovery
- automated phase characterization
- automated spectral conjecture synthesis

to explore how parity-check graph structure influences decoding stability.

Unlike traditional LDPC simulation toolkits, QEC acts as a **deterministic
Tanner-graph discovery engine** capable of evolving graph structures guided by
spectral signals and predicted decoding failures.

Modern versions extend this idea further by reconstructing the **phase geometry
of Tanner-graph space**, allowing discovery runs to automatically produce
spectral phase diagrams and structural maps of decoding stability.

---

# Discovery Engine Architecture

The QEC discovery system now operates as a **layered spectral exploration and
theory synthesis engine**.


Tanner Graph Generation
↓
Structural Diagnostics
↓
Spectral Diagnostics
↓
Failure Structure Prediction
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


This layered design allows the system to **first analyze the structure of
Tanner-graph spectral space, then explore it, and finally synthesize candidate
theoretical explanations for the observed behavior**.

---

# Spectral Phase-Space Analysis

Recent releases introduce tools for analyzing the geometry of Tanner-graph
spectral space.

These include:

### Spectral Basins
Regions of spectral space where belief-propagation dynamics behave similarly.

### Spectral Ridges
High-curvature regions separating decoding regimes and phase boundaries.

### Phase Map Reconstruction
Automatic reconstruction of phase diagrams describing decoding behavior across
spectral space.

### Discovery Trajectories
Tracking the path taken by the discovery engine through the spectral landscape.

### Phase-Guided Discovery
Steering graph mutations toward under-explored decoding phases.

### Phase Novelty Discovery
Searching for candidate Tanner-graph structures that lie outside previously
observed spectral phases.

### Phase Characterization
Automatic classification of discovered phases based on decoding behavior and
spectral diagnostics.

### Spectral Conjecture Synthesis
Generation of candidate analytic relationships between spectral metrics and
decoding performance across discovered phases.

Together these components allow QEC to function as a **deterministic discovery
system for Tanner-graph phase structure and decoding theory**.

---

# Deterministic Experiment Design

All experiments in QEC are strictly deterministic.

The framework guarantees:

- no hidden randomness
- deterministic mutation ordering
- deterministic decoder scheduling
- reproducible experiment artifacts
- identical results across repeated runs

All randomness must be explicit:


np.random.RandomState(seed)


Same seed → identical results.

This property allows Tanner-graph discovery experiments to be reproduced exactly.

---

# Research Applications

QEC enables research into:

- belief-propagation attractor geometry
- trapping-set dynamics
- spectral fragility of Tanner graphs
- decoding stability prediction
- LDPC / QLDPC code discovery
- phase-space structure of decoding dynamics
- automated discovery of new Tanner-graph regimes
- automated extraction of decoding-phase theory

The system acts as a **deterministic experimental laboratory for studying
inference dynamics in sparse graphical models**.

---

# Project Documents

Important project documents:


CLAUDE.md Development guardrails
CHANGELOG.md Release history
PROJECT_STATE.md Architecture snapshot
ROADMAP.md Long-term research direction


---

# Design Philosophy

The project follows several guiding principles.

Small is beautiful.  
Determinism is essential.  
Transparent algorithms beat opaque heuristics.

Negative results are data.

---

# Citation

If you use this framework in research, please cite:

Trent Slade  
QSOL-IMC

**QEC: Deterministic Spectral Discovery Framework for Tanner Graph Dynamics**

ORCID  
https://orcid.org/0009-0002-4515-9237

---

# Author

Trent Slade  
QSOL-IMC
