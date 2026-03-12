# QSOLKCB / QEC  
### Deterministic Quantum Error Correction Research Framework

[![Release v12.9.0](https://img.shields.io/badge/release-v12.9.0-blue)](https://github.com/QSOLKCB/QEC/releases/tag/v12.9.0)
[![Research Framework](https://img.shields.io/badge/type-research%20framework-blue)]
[![License: CC BY 4.0](https://img.shields.io/badge/license-CC--BY--4.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

QEC is a deterministic research framework for studying belief propagation dynamics, spectral structure, and Tanner graph evolution in LDPC / QLDPC codes.

The system combines spectral diagnostics, failure-structure prediction, and graph mutation to explore how parity-check graphs influence decoding stability.

Unlike typical simulation toolkits, QEC functions as a deterministic discovery engine and experimental laboratory for Tanner graph dynamics.

Overview

Belief propagation decoding on sparse parity-check graphs exhibits complex nonlinear dynamics, including:

trapping sets

absorbing sets

oscillatory convergence

metastable states

incorrect fixed points

spectral fragility

Understanding these phenomena requires controlled experiments where decoder behavior and graph structure can be observed under reproducible conditions.

The QEC framework provides this deterministic infrastructure.

Recent versions extend the system into a graph discovery engine capable of evolving Tanner graphs using spectral signals and predicted decoding failure structures.

Key Capabilities
Deterministic Tanner Graph Discovery

The discovery engine evolves parity-check matrices using structural and spectral diagnostics.

Signals guiding graph mutation include:

non-backtracking spectral radius

eigenvector localization (IPR)

trapping-set prediction

decoder residual dynamics

cycle topology diagnostics

This enables physics-informed graph evolution rather than stochastic search.

LDPC / QLDPC Code Construction

The framework provides deterministic tools for constructing parity-check matrices:

PEG-style Tanner graph generation

deterministic graph lifting

parity-check validation

CSS commutation verification

reproducible graph transformations

These tools support both classical LDPC and quantum CSS code research.

Belief Propagation Decoding

Deterministic BP implementations include:

sum-product

min-sum

normalized min-sum

offset min-sum

Supported scheduling strategies:

flooding

layered

residual

hybrid residual

adaptive scheduling

All implementations avoid hidden randomness.

Spectral Tanner Graph Diagnostics

The framework includes deterministic spectral analysis tools for Tanner graphs:

non-backtracking spectral radius

Bethe-Hessian stability

eigenvector localization

inverse participation ratio (IPR)

These diagnostics identify graph structures associated with decoding instability.

Trapping-Set Prediction

Recent versions introduce NB eigenvector trapping-set prediction.

The predictor identifies graph regions likely to produce decoding failures using:

eigenvector localization

non-backtracking flow

structural clustering

These signals enable mutation steering and instability detection before decoding runs.

Spectral-Guided Mutation

The discovery engine includes mutation operators guided by structural diagnostics.

Mutation signals include:

cycle pressure

ACE repair

spectral localization

residual cluster analysis

non-backtracking flow

trapping-set pressure

These operators evolve Tanner graphs while preserving structural invariants.

Spectral Basin Analysis

The framework now supports visualization of spectral phase space during graph mutation.

Metrics tracked include:

spectral radius

eigenvector localization

trapping-set risk

mutation trajectory

This allows researchers to study instability basins in Tanner graph space.

Benchmarking Framework

The repository includes deterministic benchmarking tools for measuring decoder performance.

Experiments evaluate:

Frame Error Rate (FER)

decoding iteration counts

spectral stability metrics

mutation effectiveness

All benchmark runs reuse identical deterministic error instances.

This ensures fair comparisons across decoding strategies.

System Architecture

The framework follows a layered experimental architecture.

Tanner Graph Generation
↓
Structural Diagnostics
↓
Spectral Diagnostics
↓
Failure Structure Prediction
↓
Mutation Operators
↓
Memetic Local Optimization
↓
Evolutionary Search
↓
Discovery Archive

Each layer observes or steers the layer below it without violating its invariants.

Determinism Guarantees

The system enforces strict deterministic execution.

Key guarantees:

no hidden randomness

deterministic scheduling

deterministic experiments

reproducible JSON artifacts

identical results across repeated runs

All randomness must use:

np.random.RandomState(seed)

Same seed → identical outputs.

Installation

Install the repository in editable mode:

pip install -e .

For development:

pip install -e .[dev]
Running Experiments

Example experiment:

PYTHONPATH=. python experiments/trapping_risk_correlation.py

Example discovery experiment:

PYTHONPATH=. python experiments/basin_steering_experiment.py

Experiments produce deterministic JSON artifacts for analysis.

Documentation

Important project documents:

CLAUDE.md — Codex development guardrails

CHANGELOG.md — release history

PROJECT_STATE.md — architecture snapshot

ROADMAP.md — long-term research direction

These documents describe the current system state and research trajectory.

Research Applications

The QEC framework enables research into:

belief propagation attractor geometry

trapping-set dynamics

spectral fragility of Tanner graphs

decoding stability prediction

LDPC / QLDPC code discovery

structure-aware decoding strategies

The system acts as a deterministic laboratory for inference dynamics in sparse graphical models.

Design Philosophy

The project follows several guiding principles.

Small is beautiful.
Determinism is essential.
Transparent algorithms beat opaque heuristics.

Negative results are data.

Author

Trent Slade
QSOL-IMC

ORCID
https://orcid.org/0009-0002-4515-9237
