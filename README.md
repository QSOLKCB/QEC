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

## Fork Synchronization Policy

**`QSOLKCB/QEC` is the canonical development line.**

This repository is a historical fork of `multimodalas/fusion-qec`, but it is
now the authoritative upstream for all QEC work. The `multimodalas/fusion-qec`
repository is retained **for historical lineage only** and must not be treated
as a source of truth.

GitHub will display a banner similar to:

> This branch is N commits ahead of and M commits behind multimodalas/fusion-qec:main

**This divergence is intentional and expected.** The ahead/behind counter does
**not** imply any required synchronization. The canonical `v137.x` history does
not share a common ancestor with the historical upstream and must never be
rebased, merged, or reconciled against it.

### Hard Rules

- **Do NOT** click "Sync fork" on GitHub.
- **Do NOT** merge `multimodalas/fusion-qec` (or any other upstream) into
  `QSOLKCB/QEC:main`.
- **Do NOT** rebase `QSOLKCB/QEC:main` onto any upstream branch.
- **Do NOT** configure automation (GitHub Actions, bots, Dependabot-style
  upstream trackers) to perform any of the above.
- All upstream comparisons are **informational only** and require **explicit
  human review** before any ref is touched.

### Why

A previous accidental "Sync fork" operation rolled `origin/main` back to a
307-commit historical fork snapshot, destroying the visible v137.x line at the
branch tip. Recovery was possible because the canonical history was preserved
in a local clone; it will not always be. Treat the upstream fork pointer as
read-only lineage metadata, nothing more.

See `AUDIT_CHECKLIST.md` § 11 and `PROJECT_STATE.md` § Disaster Recovery for
the governance controls and the restoration record.

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

If the same input does not produce the same bytes, it is not a valid result.

Current Canonical State

Current stable release: v137.14.0

The current canonical branch is:

v137.x.x

QEC now includes three major completed arcs beyond the original compute substrate.

Completed Canonical Arcs
v137.11.x — Heterogeneous Compute Substrate

Completed:

v137.11.0 — Deterministic Co-Processor Kernel
v137.11.1 — Integer / Matrix Offload Engine
v137.11.2 — Heterogeneous Scheduler
v137.11.3 — Emulator-Grade Parallel Workload Splitter
v137.11.4 — Hardware Replay Battery

This line established:

fixed-function compute lanes
deterministic epoch scheduling
workload sharding
hardware replay validation
v137.12.x — Neuromorphic + Hybrid Compute Research

Completed:

v137.12.0 — Neuromorphic Substrate Simulator
v137.12.1 — Hybrid Signal Interface Layer
v137.12.2 — Bio-Signal Benchmark Battery
v137.12.3 — Hybrid Replay Certification
v137.12.4 — Experimental Research Pack

This line is simulation-first only.

No biological claims are made without evidence receipts.

v137.13.x — Signal Abstraction Certification Arc

Completed:

v137.13.0 — Synthetic Signal Geometry Kernel
v137.13.1 — Morphology Transition Kernel
v137.13.2 — Phase Boundary Topology Kernel
v137.13.3 — Region Correspondence Kernel
v137.13.4 — Signal Abstraction Certification Battery

This arc formalizes:

geometry
→ morphology
→ topology
→ correspondence
→ certification

All outputs are deterministic, replay-safe, and bounded.

v137.14.x — Information Geometry Arc

Current:

v137.14.0 — Jensen–Shannon Signal Divergence Kernel

This introduces explicit information-theoretic divergence into the analysis stack.

Implemented:

deterministic distribution builder
Jensen–Shannon divergence
entropy alignment metrics
canonical divergence reports
receipt-chain continuity

This upgrades comparison from heuristic similarity into explicit bounded information geometry.

Core Architecture
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
→ replay-safe artifacts
→ operator console
Determinism Guarantees

QEC enforces strict reproducibility:

no hidden randomness
deterministic ordering
deterministic tie-breaking
canonical serialization
stable SHA-256 hashes
explicit seeded RNG only
12dp quantization where required
receipt-chain continuity

Example:

import numpy as np
rng = np.random.RandomState(seed)

Determinism is architecture.

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

Installation
Python / Core Runtime
git clone https://github.com/QSOLKCB/QEC.git
cd QEC
pip install -e .

Development:

pip install -r requirements-dev.txt
pytest -q
Rust TUI Operator Console

QEC includes a Rust TUI operator workstation for fast keyboard-first workflows.

Supports:

live diagnostics
topology visualization
replay inspection
invariant health
compute-lane inspection
scheduler receipts
hardware replay inspection
divergence summary inspection
Install Latest Release
curl -fsSL https://raw.githubusercontent.com/QSOLKCB/QEC/main/tui/install.sh | sh

Run:

qec-tui
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
