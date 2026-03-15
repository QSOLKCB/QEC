# QSOLKCB / QEC  
### Deterministic Spectral Discovery Framework for LDPC / QLDPC Tanner Graphs

[![Release v23.1.0](https://img.shields.io/badge/release-v23.1.0-blue)](https://github.com/QSOLKCB/QEC/releases/tag/v23.1.0)
[![Research Framework](https://img.shields.io/badge/type-research%20framework-blue)]
[![License: CC BY 4.0](https://img.shields.io/badge/license-CC--BY--4.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

QEC
Deterministic Spectral Discovery for LDPC / QLDPC Tanner Graphs

QEC is a deterministic research framework for studying belief-propagation dynamics and Tanner graph structure in classical LDPC and quantum QLDPC codes.

The system combines spectral diagnostics, decoding dynamics analysis, and deterministic graph mutation to explore how parity-check graph structure influences decoding stability.

Unlike conventional LDPC simulation toolkits, QEC functions as a deterministic discovery engine capable of evolving Tanner graphs using spectral signals and predicted failure structures.

Why This Project Exists

Belief propagation on sparse graphical models exhibits complex nonlinear behavior:

trapping sets

absorbing sets

oscillatory convergence

metastable decoding states

incorrect fixed points

spectral fragility

These phenomena determine the error-floor behavior of LDPC and QLDPC codes, yet they remain difficult to analyze with traditional design methods.

QEC provides a deterministic experimental laboratory where researchers can:

observe decoder dynamics

analyze spectral fragility

detect trapping structures

evolve Tanner graphs guided by physics-inspired signals

Discovery Engine Overview

The core of QEC is a deterministic Tanner graph discovery engine.

Initialize Tanner Graph
        ↓
Structural Diagnostics
        ↓
Spectral Diagnostics
        ↓
Failure Structure Prediction
        ↓
Mutation Candidate Generation
        ↓
Spectral Scoring
        ↓
Apply Best Mutation
        ↓
Repeat

Over time the system evolves Tanner graphs toward improved decoding stability.

System Architecture

The framework follows a layered architecture:

Tanner Graph Generation
        ↓
Structural Diagnostics
        ↓
Spectral Diagnostics
        ↓
Failure Structure Prediction
        ↓
Spectral Mutation Operators
        ↓
Adaptive Mutation Controller
        ↓
Local Graph Optimization
        ↓
Discovery Archive

Each layer analyzes or steers the layer below it while preserving its invariants.

Spectral Signals Used for Discovery

The discovery engine uses multiple structural and spectral diagnostics.

Spectral Structure

non-backtracking spectrum

Bethe–Hessian stability

eigenvector localization (IPR)

spectral entropy

spectral diversity

Structural Signals

trapping-set prediction

cycle topology diagnostics

ACE constraints

residual decoding dynamics

Flow Diagnostics

non-backtracking flow fields

cycle pressure

spectral basin detection

These signals allow physics-informed Tanner graph evolution.

Exploration Control

Modern versions include mechanisms preventing the optimizer from repeatedly rediscovering similar structures.

Exploration tools include:

trap memory

trap subspace memory

spectral diversity memory

entropy-guided exploration

temperature-annealed exploration

These mechanisms encourage the discovery engine to explore new regions of Tanner graph space.

Deterministic Experiment Design

All experiments in QEC are strictly deterministic.

The framework guarantees:

no hidden randomness

deterministic mutation ordering

deterministic decoder scheduling

reproducible experiment artifacts

identical results across repeated runs

All randomness must be explicit:

np.random.RandomState(seed)

Same seed → identical results.

Running Experiments

Install the project:

pip install -e .

For development tools:

pip install -e .[dev]
CLI Experiment System

Experiments are executed through the deterministic CLI:

qec-exp
Run a registered experiment
qec-exp run bp-threshold
Generate a phase diagram
qec-exp phase-diagram bp-threshold
Estimate BP threshold
qec-exp estimate-threshold bp-threshold
Run spectral Tanner graph search
qec-exp spectral-search --iterations 10
Enable BP convergence diagnostics
qec-exp spectral-search --iterations 10 --enable-bp-diagnostics

Experiments produce deterministic JSON artifacts for analysis.

Research Applications

QEC enables research into:

belief propagation attractor geometry

trapping-set dynamics

spectral fragility of Tanner graphs

decoding stability prediction

LDPC / QLDPC code discovery

structure-aware decoding strategies

The system acts as a deterministic laboratory for inference dynamics in sparse graphical models.

Project Documents

Important project documents:

CLAUDE.md        Development guardrails
CHANGELOG.md     Release history
PROJECT_STATE.md Architecture snapshot
ROADMAP.md       Long-term research direction
Design Philosophy

The project follows several guiding principles.

Small is beautiful.
Determinism is essential.
Transparent algorithms beat opaque heuristics.

Negative results are data.
Citation

If you use this framework in research, please cite:

Trent Slade
QSOL-IMC

QEC: Deterministic Spectral Discovery Framework for Tanner Graph Dynamics

ORCID
https://orcid.org/0009-0002-4515-9237

Author

Trent Slade
QSOL-IMC
