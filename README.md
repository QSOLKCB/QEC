# QSOLKCB / QEC
### Deterministic Quantum Error Correction • Replay-Safe Systems Architecture • Supervisory Information Geometry
### Rust TUI Operator Console • Heterogeneous Compute Lanes • Formal Replay Lineage • Control-Ready Architecture

[![Release](https://img.shields.io/github/v/release/QSOLKCB/QEC)](https://github.com/QSOLKCB/QEC/releases)
[![DOI](https://zenodo.org/badge/latestdoi/19099502.svg)](https://doi.org/10.5281/zenodo.19099502)
[![Authorea](https://img.shields.io/badge/Authorea-10.22541%2Fau.177376131.17346095%2Fv1-blue)](https://doi.org/10.22541/au.177376131.17346095/v1)
[![Branch](https://img.shields.io/badge/branch-v137%20canonical-purple)]()
[![Architecture](https://img.shields.io/badge/architecture-deterministic%20systems-blueviolet)]()
[![Determinism](https://img.shields.io/badge/determinism-byte--identical-success)]()
[![Replay](https://img.shields.io/badge/replay-hash--stable-green)]()
[![Layer 4](https://img.shields.io/badge/layer-4%20analysis-orange)]()
[![Consensus](https://img.shields.io/badge/information%20geometry-consensus-blue)]()
[![Rust TUI](https://img.shields.io/badge/operator%20console-rust%20tui-red)]()
[![Decoder Safety](https://img.shields.io/badge/decoder-sacred-critical)]()
[![Proof Ready](https://img.shields.io/badge/formal%20methods-proof%20ready-lightgrey)]()
[![License](https://img.shields.io/badge/license-CC--BY--4.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

---

# Current Stable State

**Current stable release:** `v137.14.4`  
**Current active arc:** `v137.14.x — Information Geometry + Consensus`

Completed in the current arc:

- `v137.14.0` — Jensen–Shannon Signal Divergence Kernel
- `v137.14.1` — Fisher–Rao Geometry Approximation Layer
- `v137.14.2` — Bregman / f-Divergence Correspondence Engine
- `v137.14.3` — Optimal Transport Geometry Kernel
- `v137.14.4` — Information Geometry Consensus Kernel

Current next target:

- `v137.14.5` — Geometry Drift Forecast Kernel

---

# What QEC Is

QEC is a **deterministic systems architecture** that began in quantum error correction and has evolved into a broader **replay-safe computational substrate**.

It now spans:

- quantum error correction
- deterministic orchestration
- scientific reasoning + certification
- topology-aware diagnostics
- heterogeneous compute lanes
- hardware replay verification
- synthetic signal abstraction
- morphology + topology correspondence
- information geometry kernels
- consensus manifold analysis
- drift forecasting
- Rust operator tooling
- proof-ready replay lineage

The governing law remains:

```text
continuous state
→ quantized symbolic state
→ observability
→ topology
→ governed action
→ compute lanes
→ replay verification
→ stable replay identity
→ information geometry
→ consensus manifold
→ deterministic forecast
```

> If the same input does not produce the same bytes, it is not a valid result.

---

# Current Canonical State

**Current stable release:** `v137.14.4`

The canonical development line is:

```text
v137.x.x
```

---

# Completed Canonical Arcs

## v137.11.x — Heterogeneous Compute Substrate

Completed:

- `v137.11.0` — Deterministic Co-Processor Kernel
- `v137.11.1` — Integer / Matrix Offload Engine
- `v137.11.2` — Heterogeneous Scheduler
- `v137.11.3` — Emulator-Grade Parallel Workload Splitter
- `v137.11.4` — Hardware Replay Battery
- `v137.11.5` — Neural Compression Sidecar
- `v137.11.6` — Deterministic Latent Decode Lane
- `v137.11.7` — Memory Traffic Reduction Battery

This line established:

- fixed-function compute lanes
- deterministic epoch scheduling
- workload sharding
- hardware replay validation

---

## v137.12.x — Neuromorphic + Hybrid Compute Research

Completed:

- `v137.12.0` — Neuromorphic Substrate Simulator
- `v137.12.1` — Hybrid Signal Interface Layer
- `v137.12.2` — Bio-Signal Benchmark Battery
- `v137.12.3` — Hybrid Replay Certification
- `v137.12.4` — Experimental Research Pack

This line is **simulation-first only**.

No biological claims are made without evidence receipts.

---

## v137.13.x — Signal Abstraction Certification Arc

Completed:

- `v137.13.0` — Synthetic Signal Geometry Kernel
- `v137.13.1` — Morphology Transition Kernel
- `v137.13.2` — Phase Boundary Topology Kernel
- `v137.13.3` — Region Correspondence Kernel
- `v137.13.4` — Signal Abstraction Certification Battery

This arc formalizes:

```text
geometry
→ morphology
→ topology
→ correspondence
→ certification
```

All outputs are deterministic, replay-safe, and bounded.

---

## v137.14.x — Information Geometry Arc

Completed:

- `v137.14.0` — Jensen–Shannon divergence
- `v137.14.1` — Fisher–Rao geometry
- `v137.14.2` — divergence correspondence
- `v137.14.3` — transport geometry
- `v137.14.4` — consensus manifold

This upgrades comparison from heuristic similarity into explicit bounded information geometry.

Current architecture:

```text
signal abstraction
→ divergence geometry
→ geodesic manifold
→ transport geometry
→ consensus manifold
→ drift forecasting
```

---

# Core Architecture

```text
decoder substrate
→ symbolic quantization
→ observability
→ topology graph kernel
→ compute lanes
→ replay battery
→ morphology abstraction
→ correspondence mapping
→ certification battery
→ divergence geometry
→ consensus manifold
→ replay-safe artifacts
→ operator console
```

---

# Determinism Guarantees

QEC enforces strict reproducibility:

- no hidden randomness
- deterministic ordering
- deterministic tie-breaking
- canonical serialization
- stable SHA-256 hashes
- explicit seeded RNG only
- 12dp quantization where required
- receipt-chain continuity

Example:

```python
import numpy as np
rng = np.random.RandomState(seed)
```

> Determinism is architecture.

---

# Engineering Laws

## Determinism is architecture
Same input = same bytes.

## Replay is law
Same artifacts = same stable hash.

## Hardware replay is mandatory
Divergence = failure.

## Decoder core is sacred

Do not modify:

```text
src/qec/decoder/
```

without explicit need.

## Layering is law
Lower layers do not import higher layers.

## Canonical identity is mandatory
Canonical JSON + stable SHA-256 required.

---

# Installation

## Python / Core Runtime

```bash
git clone https://github.com/QSOLKCB/QEC.git
cd QEC
pip install -e .
```

Development:

```bash
pip install -r requirements-dev.txt
pytest -q
```

---

# Rust TUI Operator Console

QEC includes a Rust TUI operator workstation for fast keyboard-first workflows.

Supports:

- live diagnostics
- topology visualization
- replay inspection
- invariant health
- compute-lane inspection
- scheduler receipts
- divergence summary inspection
- consensus manifold inspection

Install latest release:

```bash
curl -fsSL https://raw.githubusercontent.com/QSOLKCB/QEC/main/tui/install.sh | sh
```

Run:

```bash
qec-tui
```

---

# Design Philosophy

Small is beautiful.  
Determinism is architecture.  
Replay identity is law.  
Proofs beat vibes.  
Operator clarity beats hidden automation.

---

# Author

**Trent Slade**  
QSOL-IMC  
ORCID: https://orcid.org/0009-0002-4515-9237  
DOI: https://doi.org/10.5281/zenodo.19099502
