# QSOLKCB / QEC
### Deterministic Quantum Error Correction • Geometry-Driven Runtime • Replay-Safe Systems Architecture
### Rust TUI Operator Console • Topological Diagnostics • Supervisory Control • Formal Replay Lineage

[![Release](https://img.shields.io/github/v/release/QSOLKCB/QEC)](https://github.com/QSOLKCB/QEC/releases)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19099503.svg)](https://doi.org/10.5281/zenodo.19099503)
[![Authorea](https://img.shields.io/badge/Authorea-10.22541%2Fau.177376131.17346095%2Fv1-blue)](https://doi.org/10.22541/au.177376131.17346095/v1)
[![Branch](https://img.shields.io/badge/branch-v137%20canonical-purple)]()
[![Architecture](https://img.shields.io/badge/architecture-deterministic%20systems-blueviolet)]()
[![Determinism](https://img.shields.io/badge/determinism-byte--identical-success)]()
[![Replay](https://img.shields.io/badge/replay-stable%20hash%20chain-green)]()
[![Decoder Safety](https://img.shields.io/badge/decoder-sacred-critical)]()
[![Layer 4](https://img.shields.io/badge/layer-4%20analysis-orange)]()
[![Rust TUI](https://img.shields.io/badge/operator%20console-rust%20tui-red)]()
[![Topology](https://img.shields.io/badge/topology-graph%20kernel-blue)]()
[![Proof Ready](https://img.shields.io/badge/formal%20methods-proof%20ready-lightgrey)]()
[![License](https://img.shields.io/badge/license-CC--BY--4.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

---

# What QEC Is

QEC is a **deterministic systems architecture** built around quantum error correction and extended into:

- topology-aware reasoning
- replay-safe artifact chains
- temporal supervisory memory
- multimodal synchronization
- signal recovery
- deterministic orchestration
- formal verification pathways
- Rust operator tooling

The governing law is:

```text
continuous state
→ quantized symbolic state
→ observability
→ memory
→ governed action
→ stable replay identity
→ topology kernel
→ deterministic orchestration
```

If the same input does not produce the same bytes, it is not a valid result.

---

# Core Architecture

```text
decoder substrate
→ symbolic quantization
→ observability
→ supervisory memory
→ fragmentation recovery
→ fidelity benchmarking
→ topological graph kernel
→ simulation / signal runtime
→ replay-safe artifacts
→ operator console
```

Active canonical branch:

`v137.x.x`

---

# Core Capabilities

## Structural Diagnostics
- BP trajectory analysis
- attractor / basin detection
- metastability metrics
- topology-aware diagnostics
- deterministic observability artifacts
- graph continuity scoring

## Supervisory Control
- governed steering
- escalation paths
- hysteresis / timeout control
- temporal policy memory
- deterministic decisions

## Replay-Safe Artifact System
All major outputs are:

- frozen dataclasses
- canonical JSON
- canonical bytes
- stable SHA-256 hashes
- immutable lineage receipts
- byte-identical on replay

## Geometry + Topology
The `v137.8.x` line introduces:

- deterministic graph kernels
- topology divergence batteries
- manifold traversal prep
- E8 / polytope future substrate

## Formal Methods (Roadmap)
Planned verification lines:

- TLA+ model checking
- Lean 4 invariant proofs
- replay law proofs
- schema migration proofs
- topology invariant packs

---

# Determinism Guarantees

QEC enforces strict reproducibility:

- no hidden randomness
- stable ordering
- deterministic tie-breaking
- canonical serialization
- replay-safe export
- explicit seeded RNG only

Example:

```python
import numpy as np
rng = np.random.RandomState(seed)
```

Determinism is architecture.

---

# Installation

---

## Python / Core Runtime

```bash
git clone https://github.com/QSOLKCB/QEC.git
cd QEC
pip install -e .
```

Optional development tools:

```bash
pip install -r requirements-dev.txt
pytest -q
```

---

# Rust TUI Operator Console

QEC includes a **Rust TUI operator workstation** for fast keyboard-first workflows.

Supports:

- live diagnostics
- topology visualization
- replay inspection
- invariant health
- session export
- control workflows

---

## Install Latest Release (Recommended)

Install the latest tagged Rust TUI release binary with:

```bash
curl -fsSL https://raw.githubusercontent.com/QSOLKCB/QEC/main/tui/install.sh | sh
```

After install:

```bash
qec-tui
```

This resolves the latest released Rust TUI binary from GitHub releases.

---

## Windows PowerShell (Build From Source)

Open **PowerShell** inside the repository root.

Build:

```powershell
cd .\tui
cargo build --release
```

Run:

```powershell
cargo run --release
```

Run compiled binary from the local build output:

```powershell
.\target\release\qec-tui.exe
```

Important:
This `.exe` is **generated locally by Cargo** after build.
It is **not stored in the repository**.

---

## Linux / macOS (Build From Source)

```bash
cd tui
cargo build --release
cargo run --release
./target/release/qec-tui
```

---

## Windows + WSL (Optional)

For Linux-style workflow on Windows:

```powershell
wsl
cd /mnt/c/path/to/QEC/tui
cargo run --release
```

---

## Cargo Install (Optional)

Install globally from local source:

```bash
cargo install --path ./tui
```

Then run:

```bash
qec-tui
```

---

# TUI Layout

```text
Left   → navigation / mode selection
Center → diagnostics / graph / history
Right  → invariant health / topology status
Bottom → hotkeys / command legend
```

---

# Key Controls

```text
↑ / ↓   move selection
← / →   switch panes
Enter   activate
Esc     back
Q       quit
```

---

# Mode Shortcuts

```text
D   diagnostics
T   topology graph
M   memory
R   replay
I   invariants
L   law engine
X   action console
H   health
S   sessions
```

---

# Session / Replay

```text
E   export session
P   replay last
V   diff session
S   scan saved
```

---

# Engineering Laws

## Determinism is architecture
Same input = same bytes.

## Decoder core is sacred
Do not modify:

```text
src/qec/decoder/
```

without explicit need.

## Layering is law
Lower layers do not import higher layers.

## Replay identity is mandatory
Canonical JSON + stable SHA-256 required.

## Topology is deterministic
Node ordering and edge ordering must be stable.

## Formal methods are roadmap law
Future releases must remain proof-compatible.

---

# Design Philosophy

Small is beautiful.  
Determinism is architecture.  
Replay identity is law.  
Topology is structure.  
Proofs beat vibes.  
Operator clarity beats hidden automation.

---

# Author

**Trent Slade**  
**QSOL-IMC**  
ORCID: https://orcid.org/0009-0002-4515-9237
