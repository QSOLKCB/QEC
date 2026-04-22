# QSOLKCB / QEC

**Deterministic Runtime • Invariant-Driven Compute • Replay-Safe Systems • Proof-Carrying Execution**

---

[![Release](https://img.shields.io/github/v/release/QSOLKCB/QEC)](https://github.com/QSOLKCB/QEC/releases)
[![Latest](https://img.shields.io/badge/stable-v142.4.2-success)](https://github.com/QSOLKCB/QEC/releases/tag/v142.4.2)

[![OSF Registration](https://img.shields.io/badge/OSF-Registration-blue)](https://osf.io/sjk7b)

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19062692-blue)](https://doi.org/10.5281/zenodo.19062692)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19102390-blue)](https://doi.org/10.5281/zenodo.19102390)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19099503-blue)](https://doi.org/10.5281/zenodo.19099503)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19104208-blue)](https://doi.org/10.5281/zenodo.19104208)

[![Branch](https://img.shields.io/badge/branch-v142%20canonical-purple)]()
[![Architecture](https://img.shields.io/badge/architecture-deterministic%20runtime-blueviolet)]()
[![Determinism](https://img.shields.io/badge/determinism-byte--identical-success)]()
[![Replay](https://img.shields.io/badge/replay-hash--stable-green)]()
[![Governance](https://img.shields.io/badge/governance-proof--carrying-orange)]()
[![License](https://img.shields.io/badge/license-CC--BY--4.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

---

# 🚀 What QEC Is (Now)

QEC is a **deterministic runtime + invariant-driven compute reduction engine**.

It does not just execute systems.

It determines:

```text
what must run
what can be skipped
what is already known
```

---

# 🧠 Core Law

```text
same input
→ same ordering
→ same canonical JSON
→ same stable hash
→ same bytes
```

---

# 🔥 The Big Idea

Most systems:

```text
iterate → recompute → waste work
```

QEC / IRIS:

```text
iterate
→ detect invariants
→ detect convergence structure
→ eliminate redundant computation
→ produce deterministic proof
```

---

# ⚙️ What It Actually Does

Across any iterative system:

* detects fixed points, plateaus, oscillations
* classifies convergence behavior
* builds deterministic execution plans
* quantifies redundant computation
* produces replay-safe, canonical proof artifacts

---

# 🧠 System Evolution

```text
QEC (error correction)
→ deterministic runtime
→ invariant detection
→ convergence classification
→ execution planning
→ benchmarking
→ proof of compute reduction
```

---

# ⚡ Quickstart — Run the Proof

This is the fastest way to understand the system.

---

## 1. Install

```bash
git clone https://github.com/QSOLKCB/QEC.git
cd QEC
pip install -e .
```

---

## 2. Run the Cross-Domain Demo

```bash
python scripts/demo_cross_domain_benchmarks.py
```

---

## 3. What You’ll See

The script runs the full pipeline:

```text
trace
→ invariants
→ convergence
→ execution plan
→ structural redundancy
→ benchmark receipt
```

Across domains:

```text
transformers
diffusion
gnn
physics
```

---

## 4. Output (Simplified)

```text
SUMMARY | transformers | total=8 | effective=5 | cutoff=4 | redundancy=0.375 | efficiency=0.558 | label=high
```

```text
SUMMARY | gnn | total=7 | effective=3 | cutoff=2 | redundancy=0.571 | efficiency=0.720 | label=high
```

---

## 5. What This Proves

```text
redundant iterations exist
→ they can be detected deterministically
→ they can be quantified
→ they can be eliminated
```

---

# 📊 Benchmark Output (Interpretation)

QEC does not guess.

It produces:

```text
structural redundancy
effective iterations
convergence classification
execution posture
efficiency gain
```

All:

* deterministic
* replayable
* hash-stable

---

# 🧠 What “Redundancy” Means Here

```text
redundant iterations =
steps executed AFTER convergence structure was already established
```

This is:

* not heuristic
* not probabilistic
* derived from the execution trace

---

# 🔬 Why This Matters

At scale:

```text
compute cost ≈ iterations × redundancy
```

If redundancy is:

```text
30–80%
```

then:

```text
compute can be reduced proportionally
```

---

# 🧩 Deterministic Runtime Pipeline

```text
INPUT
 ↓
[Iterative System]
 ↓
[Invariant Detection]
 ↓
[Convergence Engine]
 ↓
[Execution Wrapper]
 ↓
[Redundancy Measurement]
 ↓
[Benchmark / Proof]
 ↓
OUTPUT
```

---

# 🔗 Artifact Lineage

```text
state_hash
→ replay_hash
→ convergence_hash
→ execution_hash
→ benchmark_hash
```

---

# 🔒 Determinism Guarantees

* no randomness
* canonical JSON
* SHA-256 hashing
* replay-safe artifacts
* identical outputs across runs

---

# ⚙️ Covenant Runtime Model

```text
state_t + action → next_state + proof
```

---

# 🧠 Engineering Laws

* Same input = same bytes
* Replay is law
* Proof > intuition
* Determinism over heuristics
* Eliminate work, don’t accelerate waste

---

# 🖥 Rust TUI Operator Console

Keyboard-first deterministic control surface.

### Features

* live diagnostics
* replay inspection
* invariant health
* proof audit inspection
* orchestration tracing
* simulation monitoring

---

### Install

```bash
curl -fsSL https://raw.githubusercontent.com/QSOLKCB/QEC/main/tui/install.sh | sh
```

---

### Run

```bash
qec-tui
```

---

## 🔹 Operator Walkthrough (TUI Demo Flow)

### 1. Launch TUI

```bash
qec-tui
```

---

### 2. Navigate Modes

Use:

* ↑ / ↓ → move
* Enter → select
* Q → quit

---

### 3. Run a Deterministic Replay

Go to:

```
Diagnostics → Replay Inspector
```

You’ll see:

* input state
* canonical ordering
* stable hash
* artifact lineage

---

### 4. Inspect Proof-Carrying Execution

Go to:

```
Governance → Proof Audit
```

Observe:

* skip vs execute decisions
* validation state
* proof receipt integrity

---

### 5. View Runtime Elimination

Go to:

```
Simulation → Execution Fabric
```

Watch:

* regions being skipped
* deterministic routing
* elimination decisions

---

### 6. Inspect Benchmark Projection

Go to:

```
Diagnostics → Benchmark View
```

Shows:

* baseline cost
* projected cost
* normalized reduction score

---

### 7. Verify Determinism

Re-run same input:

* ✔ hashes remain identical
* ✔ decisions remain identical

---

## 🧠 Key Insight

The TUI is not visualization.

It is:

```text
deterministic state introspection
```

---

## 📚 References

* Deterministic Runtime Optimization and Formal Invariant Validation
  https://doi.org/10.5281/zenodo.19062692

* Invariant-Driven Computation Elimination
  https://doi.org/10.5281/zenodo.19102390

* Deterministic Redundancy Elimination
  https://doi.org/10.5281/zenodo.19099503

* Dark-State Invariants
  https://doi.org/10.5281/zenodo.19104208

---

## 📄 Registered Research Artifact

Deterministic Proof-Carrying Runtime Elimination (v138.6.x)
https://osf.io/sjk7b

---

## 📄 Papers

https://github.com/QSOLKCB/QEC/tree/main/papers

---

## 👤 Author

Trent Slade
ORCID: https://orcid.org/0009-0002-4515-9237
