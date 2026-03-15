# QSOLKCB / QEC
### Deterministic Spectral Discovery Engine for LDPC and QLDPC Tanner Graphs

QEC is a deterministic research framework for studying belief-propagation dynamics,
spectral stability, and Tanner-graph structure in sparse graphical codes.

The system functions as a **spectral discovery engine** capable of evolving LDPC
and QLDPC parity-check graphs using structural diagnostics, spectral signals,
and deterministic mutation operators.

Rather than relying on stochastic evolutionary search, QEC performs **fully
deterministic graph exploration**, allowing Tanner graph discovery experiments
to be reproduced exactly across machines and runs.

[![Release v34.0.0](https://img.shields.io/badge/release-v34.0.0-blue)](https://github.com/QSOLKCB/QEC/releases/tag/v34.0.0)
[![Research Framework](https://img.shields.io/badge/type-research%20framework-blue)]
[![License: CC BY 4.0](https://img.shields.io/badge/license-CC--BY--4.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

## QEC
### Deterministic Spectral Discovery for LDPC / QLDPC Tanner Graphs

QEC is a deterministic research framework for studying belief-propagation
dynamics and Tanner graph structure in classical LDPC and quantum QLDPC codes.

The framework combines:

- spectral diagnostics
- decoding dynamics analysis
- deterministic mutation operators
- failure-structure prediction

to explore how parity-check graph structure influences decoding stability.

Unlike traditional LDPC simulation toolkits, QEC acts as a **deterministic
Tanner-graph discovery engine** capable of evolving graph structures guided by
spectral signals and predicted decoding failures.

Recent versions introduce a **mutation plugin architecture** that allows new
graph-evolution strategies to be added without modifying the core search
pipeline.

This architecture enables deterministic exploration of Tanner-graph design
space using physics-inspired signals derived from belief-propagation dynamics.

---

## Why This Project Exists

Belief propagation on sparse graphical models exhibits complex nonlinear
behavior including:

- trapping sets
- absorbing sets
- oscillatory convergence
- metastable decoding states
- incorrect fixed points
- spectral fragility

These phenomena strongly influence the **error-floor behavior of LDPC and QLDPC
codes**, yet they remain difficult to analyze with traditional code-design
approaches.

QEC provides a deterministic experimental laboratory where researchers can:

- observe decoder dynamics
- analyze spectral fragility
- detect trapping structures
- evolve Tanner graphs using spectral feedback signals
- study the geometry of belief-propagation attractors

The system enables controlled experiments on how Tanner graph structure affects
inference stability.

---

## How QEC Differs from Traditional LDPC Simulators

Most LDPC software packages focus on **simulating decoder performance** for fixed code constructions.

Typical toolkits provide:

- Monte Carlo BER simulations  
- decoder implementations  
- code construction utilities  
- performance benchmarking

QEC serves a different purpose.

Instead of only simulating existing codes, QEC acts as a **deterministic discovery engine** that explores Tanner graph structure space.

The system analyzes belief-propagation dynamics and uses spectral signals to guide deterministic mutations of parity-check graphs.

This allows researchers to study:

- how Tanner graph structure influences decoding stability  
- how trapping structures emerge  
- how spectral fragility predicts decoding failure  
- how graph mutations can improve decoder behavior

In this sense, QEC functions more like a **laboratory for Tanner graph evolution** than a traditional simulation toolkit.

---

## Discovery Engine Overview

The core of QEC is a deterministic Tanner-graph discovery engine.

System Architecture

QEC follows a layered architecture where each stage analyzes or steers the layer below it while preserving deterministic invariants.

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
Adaptive Mutation Controller
        ↓
Local Graph Optimization
        ↓
Discovery Archive

Recent versions introduce a mutation plugin registry, allowing new Tanner-graph mutation algorithms to be added without modifying the core discovery engine.

All mutation operators implement a shared interface and are executed in deterministic order based on spectral scoring signals.

Spectral Signals Used for Discovery

The discovery engine relies on several structural and spectral diagnostics.

Spectral Structure

non-backtracking spectrum

Bethe–Hessian stability

eigenvector localization (IPR)

spectral entropy

spectral diversity

These quantities describe the stability landscape of belief propagation on the Tanner graph.

Structural Signals

trapping-set prediction

cycle topology diagnostics

ACE constraints

residual decoding dynamics

These signals characterize graph structures that destabilize belief propagation.

Flow Diagnostics

non-backtracking flow fields

cycle pressure

spectral basin detection

These diagnostics help guide mutation operators toward structurally meaningful graph modifications.

Mutation System

Modern versions of QEC implement Tanner-graph evolution through a deterministic mutation framework.

The mutation system now consists of:

MutationContext
        ↓
MutationRegistry
        ↓
MutationOperators
        ↓
Deterministic Spectral Scoring

Mutation operators can include:

non-backtracking eigenvector flow mutations

trapping-set repair mutations

spectral defect atlas reuse

future structural mutation operators

Operators are executed in deterministic order based on spectral signals such as the non-backtracking spectral radius.

This allows multiple mutation strategies to compete within the same discovery pipeline.

Exploration Control

To prevent the discovery engine from repeatedly rediscovering the same structures, QEC maintains deterministic exploration controls.

These include:

spectral mutation memory

spectral defect atlas reuse

diversity-preserving mutation ordering

deterministic scoring of mutation operators

These mechanisms encourage exploration of new regions of Tanner-graph space while maintaining reproducibility.

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

This property allows Tanner-graph discovery experiments to be reproduced exactly.

Running Experiments

Install the project:

pip install -e .

For development tools:

pip install -e .[dev]
CLI Experiment System

Experiments are executed through the deterministic CLI.

Run a registered experiment:

qec-exp run bp-threshold

Generate a phase diagram:

qec-exp phase-diagram bp-threshold

Estimate BP threshold:

qec-exp estimate-threshold bp-threshold

Run spectral Tanner graph discovery:

qec-exp spectral-search --iterations 10

Enable BP convergence diagnostics:

qec-exp spectral-search --iterations 10 --enable-bp-diagnostics

Experiments produce deterministic JSON artifacts for analysis.

Research Applications

QEC enables research into:

belief-propagation attractor geometry

trapping-set dynamics

spectral fragility of Tanner graphs

decoding stability prediction

LDPC / QLDPC code discovery

structure-aware decoding strategies

The system acts as a deterministic laboratory for studying inference dynamics in sparse graphical models.

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
