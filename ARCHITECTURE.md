# QSOLKCB / QEC — ARCHITECTURE.md

## Deterministic System Architecture (Canonical)

---

## 🧠 System Identity

> QEC is a deterministic, replay-safe reasoning system operating as a decoder over system evolution.

The architecture enforces strict separation between:

* **decoder (measured system)**
* **analysis (reasoning system)**

Control is external. The decoder is never modified.

---

## 🔁 System Model

QEC operates as a closed-loop deterministic reasoning pipeline:

```
metrics
→ structure
→ decision
→ evaluation
→ adaptation
→ memory
```

All transitions are:

* deterministic
* replay-safe
* invariant-preserving

---

## 🧱 Layer Architecture (Strict Dependency Model)

Dependencies flow downward only.

Violation → architecture invalid

| Layer | Path                   | Role                                      |
| ----- | ---------------------- | ----------------------------------------- |
| L1    | `src/qec/decoder/`     | Decoder core (protected, immutable)       |
| L2    | `src/qec/channel/`     | Input generation (noise, syndromes, LLRs) |
| L3    | `src/qec/diagnostics/` | Observation (decoder state measurement)   |
| L4    | `src/qec/predictors/`  | Pre-decode signals (risk / instability)   |
| L5    | `src/qec/analysis/`    | Deterministic reasoning pipeline          |
| L6    | `src/qec/experiments/` | Orchestration (controlled execution)      |
| L7    | `src/bench/`           | External benchmarking (non-importable)    |

Rules:

* lower layers MUST NOT import higher layers
* decoder MUST NOT depend on analysis
* benchmark layer MUST NOT be imported internally

---

## 🔒 Decoder Core (L1) — Protected System

The decoder is a fixed experimental object.

Properties:

* immutable
* bit-stable across releases
* externally controlled only

Constraints:

* no modification
* no injection
* no hooks
* no adaptive logic inside decoder

Interaction boundary:

* inputs: LLRs, schedules, graph structure
* outputs: beliefs, energies, syndromes

Violation → system invalid

---

## ⚡ Channel Layer (L2)

Provides deterministic input generation:

* syndrome generation
* noise modeling
* LLR computation

Constraint:

* outputs must be canonical and reproducible

---

## 🔍 Diagnostics Layer (L3)

Pure observation layer.

Properties:

* side-effect free
* operates on copies
* no mutation of inputs

Functions:

* BP dynamics tracking
* spectral analysis
* stability characterization

Constraint:

* diagnostics MUST NOT influence decoder execution

---

## 🧠 Predictor Layer (L4)

Pre-decode signal generation.

Properties:

* consumes diagnostic outputs
* produces signals only
* no control authority

Constraint:

* predictors MUST NOT modify decoder inputs directly

---

## 🧩 Analysis Layer (L5) — Deterministic Reasoning Core

This is the core reasoning system.

Pipeline stages:

### 1. Metrics

* alignment
* curvature
* resonance
* complexity
* scale consistency

### 2. Structure (Attractor Analysis)

* regime classification
* basin identification
* transition detection

### 3. Decision (Strategy Selection)

Deterministic selection based on:

* score
* confidence
* simplicity
* lexicographic ordering

Tie-breaking is fully deterministic.

### 4. Evaluation

* before/after comparison
* improvement scoring
* outcome classification

### 5. Adaptation

* global bias adjustment
* trajectory weighting
* regime-aware scaling

### 6. Memory

Stores deterministic historical signals for future decisions.

Constraint:

* no randomness
* no mutation of prior state

---

## 🧠 Memory Architecture

All memory is:

* bounded
* immutable
* deterministic
* indexed

### Memory Types

**Strategy Memory**

* per-strategy outcome tracking

**Regime Memory**

* context-aware aggregation

**Transition Memory**

* state transition recording
* bias computation

**Multi-Step Evaluation**

* fixed horizon lookahead (depth = 2)

Constraint:

* no recursion
* no stochastic updates

---

## ⚙️ Scoring Model

```
final_score = base_score
            × stability_weight
            × transition_bias
            × multi_step_factor
```

Constraints:

* all factors bounded
* all factors deterministic
* final score clamped

---

## 🧪 Experiment Layer (L6)

Responsible for controlled execution.

Properties:

* deterministic orchestration
* fixed inputs
* reproducible runs

Constraint:

* no modification of analysis logic

---

## 📊 Benchmark Layer (L7)

External evaluation only.

Constraint:

* MUST NOT be imported by core system

---

## 🔒 Determinism Guarantees

The system enforces:

* no hidden randomness
* explicit seeded RNG only (if used)
* deterministic ordering (sorted keys)
* canonical serialization
* no unstable hashing
* stable floating-point handling
* deterministic tie-breaking

Result:

```
same input → same bytes
```

---

## 🚫 Forbidden Operations

* randomness (implicit or hidden)
* wall-clock dependence
* nondeterministic async behavior
* decoder mutation
* unordered iteration
* non-canonical serialization

Violation → system invalid

---

## 🧠 Architectural Principle

The decoder is not optimized.

It is measured.

The system operates as:

```
external deterministic reasoning
→ controlled input modification
→ invariant observation
→ proof of behavior
```

Control occurs at the boundary, never inside the decoder.

---

## 🧠 Final Statement

QEC architecture enforces:

* strict separation of execution and reasoning
* deterministic control over stochastic domains
* invariant-preserving system evolution

The result is a system that does not approximate behavior.

It explains, constrains, and proves it.

---
