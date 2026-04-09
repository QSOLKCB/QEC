# QSOLKCB / QEC
### Deterministic Quantum Error Correction • Heterogeneous Compute Substrate • Replay-Safe Systems Architecture
### Rust TUI Operator Console • Topological Diagnostics • Fixed-Function Compute Lanes • Formal Replay Lineage

[![Release](https://img.shields.io/github/v/release/QSOLKCB/QEC)](https://github.com/QSOLKCB/QEC/releases)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19099503.svg)](https://doi.org/10.5281/zenodo.19099503)
[![Authorea](https://img.shields.io/badge/Authorea-10.22541%2Fau.177376131.17346095%2Fv1-blue)](https://doi.org/10.22541/au.177376131.17346095/v1)
[![Branch](https://img.shields.io/badge/branch-v137%20canonical-purple)]()
[![Architecture](https://img.shields.io/badge/architecture-deterministic%20systems-blueviolet)]()
[![Determinism](https://img.shields.io/badge/determinism-byte--identical-success)]()
[![Replay](https://img.shields.io/badge/replay-hardware%20verified-green)]()
[![Compute](https://img.shields.io/badge/compute-heterogeneous%20lanes-orange)]()
[![Decoder Safety](https://img.shields.io/badge/decoder-sacred-critical)]()
[![Layer 4](https://img.shields.io/badge/layer-4%20analysis-orange)]()
[![Rust TUI](https://img.shields.io/badge/operator%20console-rust%20tui-red)]()
[![Topology](https://img.shields.io/badge/topology-graph%20kernel-blue)]()
[![Proof Ready](https://img.shields.io/badge/formal%20methods-proof%20ready-lightgrey)]()
[![License](https://img.shields.io/badge/license-CC--BY--4.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

---

# What QEC Is

QEC is a **deterministic systems architecture** rooted in quantum error correction and extended into a replay-safe heterogeneous compute substrate.

The system now spans:

- quantum error correction
- deterministic orchestration
- scientific reasoning + certification
- topology-aware diagnostics
- fixed-function compute lanes
- integer / matrix offload
- epoch scheduling
- parallel workload partitioning
- hardware replay verification
- Rust operator tooling

The governing law is:

```text
continuous state
→ quantized symbolic state
→ observability
→ topology
→ governed action
→ heterogeneous compute lanes
→ replay verification
→ stable replay identity

If the same input does not produce the same bytes, it is not a valid result.

Current Canonical State

Current stable release: v137.11.4

The v137.11.x line establishes a deterministic heterogeneous compute substrate.

Completed core arc:

v137.11.0 — Deterministic Co-Processor Kernel
v137.11.1 — Integer / Matrix Offload Engine
v137.11.2 — Heterogeneous Scheduler
v137.11.3 — Emulator-Grade Parallel Workload Splitter
v137.11.4 — Hardware Replay Battery

This line is inspired by:

retro fixed-function co-processors
integer-first acceleration
deterministic scheduling epochs
replay-safe workload partitioning
hardware divergence detection
Core Architecture
decoder substrate
→ symbolic quantization
→ observability
→ topology graph kernel
→ deterministic co-processor lanes
→ integer / matrix offload
→ epoch scheduler
→ workload splitter
→ hardware replay battery
→ replay-safe artifacts
→ operator console

Active canonical branch:

v137.x.x

Core Capabilities
Quantum + Structural Diagnostics
BP trajectory analysis
attractor / basin detection
metastability metrics
topology-aware diagnostics
graph continuity scoring
deterministic observability artifacts
Scientific Reasoning + Certification

Completed v137.10.x reasoning stack:

hypothesis lattice
experiment DSL
evidence lineage
claim audit kernel
proof obligation extractor
numerological rejection battery
scientific certification kernel

All outputs are replay-safe and hash-stable.

Heterogeneous Compute (v137.11.x)
Deterministic Co-Processor Kernel

Fixed-function compute contract:

cpu → descriptor → co-processor → receipt
Integer / Matrix Offload Engine

Integer-first acceleration lane:

deterministic matrix transforms
fixed-point multiply-accumulate
saturating arithmetic
explicit scaling metadata
stable cycle counts

Hard rule:

no floating point unless mathematically unavoidable
Heterogeneous Scheduler

Epoch-based deterministic scheduling:

stable dependency ordering
merge barriers
lexicographic tie-break
replay-safe dispatch receipts

Hard rule:

same epoch = same execution order
Parallel Workload Splitter

Tile / chunk partitioning:

fixed tiles
row chunks
column chunks
scanline split
deterministic merge receipts

Hard rule:

same input = same shards = same merge bytes
Hardware Replay Battery

Replay verification across all compute lanes.

Validates:

input hash
output hash
epoch identity
shard identity
byte equality

Hard law:

replay failure = architecture failure
Determinism Guarantees

QEC enforces strict reproducibility:

no hidden randomness
stable ordering
deterministic tie-breaking
canonical serialization
stable SHA-256 hashes
replay-safe export
explicit seeded RNG only

Example:

import numpy as np
rng = np.random.RandomState(seed)

Determinism is architecture.

Installation
Python / Core Runtime
git clone https://github.com/QSOLKCB/QEC.git
cd QEC
pip install -e .

Development tools:

pip install -r requirements-dev.txt
pytest -q
Rust TUI Operator Console

QEC includes a Rust TUI operator workstation for fast keyboard-first workflows.

Supports:

live diagnostics
topology visualization
replay inspection
invariant health
session export
compute-lane inspection
scheduler receipts
hardware replay inspection
Install Latest Release
curl -fsSL https://raw.githubusercontent.com/QSOLKCB/QEC/main/tui/install.sh | sh

Run:

qec-tui
Build From Source
Linux / macOS
cd tui
cargo build --release
cargo run --release
./target/release/qec-tui
Windows PowerShell
cd .\tui
cargo build --release
cargo run --release
.\target\release\qec-tui.exe
TUI Layout
Left   → navigation / mode selection
Center → diagnostics / graph / scheduler / replay
Right  → invariant health / topology / compute lanes
Bottom → hotkeys / command legend
Engineering Laws
Determinism is architecture

Same input = same bytes.

Replay is law

Same artifacts = same stable hash.

Hardware replay is mandatory

Divergence = failure.

Decoder core is sacred

Do not modify:

src/qec/decoder/

without explicit need.

Layering is law

Lower layers do not import higher layers.

Canonical identity is mandatory

Canonical JSON + stable SHA-256 required.

Design Philosophy

Small is beautiful.
Determinism is architecture.
Replay identity is law.
Hardware replay proves truth.
Fixed-function beats ambiguity.
Proofs beat vibes.
Operator clarity beats hidden automation.

Author

Trent Slade
QSOL-IMC
ORCID: https://orcid.org/0009-0002-4515-9237
