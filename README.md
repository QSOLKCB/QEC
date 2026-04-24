# QSOLKCB / QEC

**Deterministic Autonomous Control • Memory • Governance • Validation • Replay-Safe Systems**

---

## 📦 Release & Research

[![Release](https://img.shields.io/github/v/release/QSOLKCB/QEC)](https://github.com/QSOLKCB/QEC/releases)  
[![Latest](https://img.shields.io/badge/stable-v148.0-success)](https://github.com/QSOLKCB/QEC/releases/tag/v148.0)

[![OSF Registration](https://img.shields.io/badge/OSF-Registration-blue)](https://osf.io/sjk7b)

---

## 📚 DOIs

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19697907-blue)](https://doi.org/10.5281/zenodo.19697907)  
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19062692-blue)](https://doi.org/10.5281/zenodo.19062692)  
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19102390-blue)](https://doi.org/10.5281/zenodo.19102390)  
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19099503-blue)](https://doi.org/10.5281/zenodo.19099503)  
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19104208-blue)](https://doi.org/10.5281/zenodo.19104208)

---

## 🧩 System Properties

[![Branch](https://img.shields.io/badge/branch-v148%20canonical-purple)]()  
[![Architecture](https://img.shields.io/badge/architecture-autonomous%20control%20system-blueviolet)]()  
[![Determinism](https://img.shields.io/badge/determinism-byte--identical-success)]()  
[![Replay](https://img.shields.io/badge/replay-hash--stable-green)]()  
[![Governance](https://img.shields.io/badge/governance-adaptive%20validated-orange)]()  
[![Validation](https://img.shields.io/badge/validation-replay--proven-yellow)]()  
[![License](https://img.shields.io/badge/license-CC--BY--4.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

---

# 🚀 What QEC Is

QEC is a **deterministic autonomous control and governance system**.

It is not just a runtime.

It is a system that:

- observes state  
- predicts evolution  
- selects actions  
- accumulates memory  
- governs future behavior  
- validates its own decisions  

It determines:

```text
what must run
what can be skipped
what is already known
what is structurally constrained
what is admissible next
what policy should evolve
what decisions are provably stable
🧠 Core Law
same input
→ same ordering
→ same canonical JSON
→ same stable hash
→ same bytes
🔥 The Big Idea
Most systems
iterate → recompute → waste work
QEC (IRIS + SPHAERA → Control → Governance → Validation)
iterate
→ detect invariants
→ detect convergence structure
→ eliminate redundant computation
→ embed structure (geometry)
→ enforce agreement (ensemble)
→ characterize dynamics (spectral)
→ select deterministically
→ produce proof-carrying artifacts
→ accumulate memory
→ generate governance decisions
→ validate decision stability
⚙️ System Capability (v148 State)

QEC now performs:

invariant detection
convergence classification
redundancy elimination
deterministic execution planning
geometric invariant embedding
ensemble consistency enforcement
spectral structure characterization
deterministic transition selection
unified runtime state construction
policy memory accumulation
adaptive governance recommendation
governance validation (v148.0)

---

```
⚡ Quickstart — Run the Proof Artifact

## 1. Install

```bash
git clone https://github.com/QSOLKCB/QEC.git
cd QEC
pip install -e .
```

## 2. Run the SPHAERA Proof

```bash
python scripts/sphaera_proof_demo.py
```

## 3. What It Executes

```text
trace
→ invariants
→ convergence
→ execution structure
→ geometry
→ ensemble validation
→ spectral structure
→ deterministic decision
→ runtime receipt
```

## 4. Example Output (v143.5)

```text
=== SPHAERA TABLE ===

| Domain       | Invariants | Geometry Classes | Ensemble Consistency | Spectral Dynamics | Selected Transition       | Runtime Coherence | Global State           |
|--------------|-----------:|-----------------:|---------------------:|-------------------|---------------------------|------------------:|------------------------|
| transformers |          4 |                4 |                0.982 | structured        | maintain_invariant_anchor |             0.931 | structured_equilibrium |
| diffusion    |          5 |                5 |                0.954 | coupled           | adaptive_balance          |             0.882 | adaptive_state         |
| gnn          |          3 |                3 |                0.721 | dynamic           | coupling_probe            |             0.671 | dynamic_state          |
| physics      |          2 |                2 |                1.000 | rigid             | safe_hold                 |             1.000 | stable_equilibrium     |
```

---

## 🔹 Operator Walkthrough — Rust TUI Control Surface

The QEC Rust TUI provides a **deterministic operator console** for observing, validating, and interacting with the runtime.

### ⚡ 0. Quick Start (One-Line Bootstrap)

```bash
curl -fsSL https://raw.githubusercontent.com/QSOLKCB/QEC/main/tui/install.sh | sh
```

This will:

- build the Rust TUI in release mode
- launch the operator console

### 🟢 1. Launch the Console (Manual)

```bash
cd tui
cargo run --release
```

Production binary:

```bash
./target/release/qec-tui
```

### 🧭 Layout

```text
Left   → navigation
Center → workspace
Right  → system / invariant status
Bottom → hotkeys
```

### 🔍 2. Diagnostics Mode (`D`)

Press:

```text
D
```

View:

- system state
- invariant metrics
- convergence signals
- anomaly indicators

Purpose:  
Understand system condition.

### 📜 3. History Mode (`H`)

Press:

```text
H
```

View:

- execution timeline
- state transitions
- replay checkpoints

Purpose:  
Trace deterministic evolution.

### 🧠 4. Phase Workstation (`P`)

Press:

```text
P
```

View:

- phase-space structure
- attractor behavior
- convergence regions

Purpose:  
Analyze structural dynamics.

### ⚙️ 5. Actions Mode (`A`)

Press:

```text
A
```

View:

- invariant-guided actions
- recovery pathways
- control sequences

Purpose:  
Apply deterministic interventions.

### 🔁 6. Replay Mode (`R`)

Press:

```text
R
```

View:

- replay execution
- validate reconstruction
- verify hash stability

Purpose:  
Prove reproducibility.

### 📊 7. System Status (`S`)

Press:

```text
S
```

View:

- invariant health
- convergence classification
- runtime integrity

Purpose:  
Confirm global correctness.

### 🔒 Operator Model

- read-only by default
- deterministic rendering
- invariant-aligned
- no hidden state

### ⚡ Core Flow

```text
Observe → Validate → Decide → Act → Replay → Confirm
```

### 🧠 Design Intent

```text
deterministic control surface
for invariant-governed systems
```

---

## 📚 Attribution

Marc Brendecke  
ORCID: https://orcid.org/0009-0009-4034-598X

Quantum Sphaera Companion v3.30.0  
https://doi.org/10.5281/zenodo.19682951

License: CC-BY-4.0

---

## 📚 References

- IRIS: Deterministic Invariant-Driven Reduction of Redundant Computation  
  https://doi.org/10.5281/zenodo.19697907

- Deterministic Runtime Optimization and Formal Invariant Validation  
  https://doi.org/10.5281/zenodo.19062692

- Invariant-Driven Computation Elimination  
  https://doi.org/10.5281/zenodo.19102390

- Deterministic Redundancy Elimination  
  https://doi.org/10.5281/zenodo.19099503

- Dark-State Invariants  
  https://doi.org/10.5281/zenodo.19104208

---

## 👤 Author

Trent Slade  
https://orcid.org/0009-0002-4515-9237
