# QSOLKCB / QEC

QEC is a deterministic, replay-safe reasoning system for quantum error correction,
invariant-driven computation, canonical proof artifacts, and receipt-bound
simulation analysis.

It turns computation, evidence, semantic alignment, governance, replay
validation, lattice topology, router/readout proof, multi-scale invariance,
entropy drift analysis, GameWorld interaction boundaries,
perturbation/stress contracts, substrate constraints, bounded recursive loops,
reality-loop composition, global replay proofs, heavy-dependency optimization,
optimized simulation specifications, backend replay equivalence, benchmark
receipts, telemetry receipts, and simulation reports into canonical proof
artifacts.

The repository began with a Layer 1 QEC decoder and now extends that foundation
into a broader deterministic proof stack.

In this README, QEC means the QSOLKCB/QEC software system and release lineage.

**Deterministic Reasoning • Canonical Identity • RES/RAG Semantics • Governance • Distributed Proof • Atomic Lattices • Router Paths • Readout Projections • SearchMask64 • Hilber/Hilbert Shift • Functional Readout Shells • Replay Validation • Multi-Scale Invariance • Entropy Drift • Decay-Resistance Proofs • GameWorld Interaction Reports • Perturbation Contracts • Substrate Constraints • Recursive Proof Loops • Reality Loop Proofs • Global Replay Proofs • Heavy-Dependency Invariants • Optimization Contracts • Adapter Specs • Canonical Kernel Receipts • Fast-Path Equivalence • Dependency Reduction • Optimized Simulation Specs • Backend Replay Receipts • Benchmark Receipts • Telemetry Receipts • Simulation Reports**

## 📦 Release & Research

[![Release](https://img.shields.io/github/v/release/QSOLKCB/QEC)](https://github.com/QSOLKCB/QEC/releases)
[![Latest](https://img.shields.io/badge/stable-v165.4-success)](https://github.com/QSOLKCB/QEC/releases/tag/v165.4)
[![Branch](https://img.shields.io/badge/branch-v165.4%20canonical-purple)]()

Current release line: **v165.4**  
Current frontier: **v165.5.0 — DataframeBackendManifest**  
Active arc: **v165.5.x — Deterministic Dataframe / Columnar Backend Receipts**  
Completed arc: **v165.x — Optimized QEC Simulation Backends**

Repository status is current through **v165.4 → OptimizedSimulationReport**.

Recent preprint: **QSOLKCB/QEC v155.x: Deterministic Entropy and Decay
Signatures — Hash-Bound Checkpoints, Subsystem Drift Receipts, and
Replay-Resistant Proof Chains**.

## 📚 DOIs

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19697907-blue)](https://doi.org/10.5281/zenodo.19697907)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19062692-blue)](https://doi.org/10.5281/zenodo.19062692)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19102390-blue)](https://doi.org/10.5281/zenodo.19102390)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19099503-blue)](https://doi.org/10.5281/zenodo.19099503)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19104208-blue)](https://doi.org/10.5281/zenodo.19104208)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19840786-blue)](https://doi.org/10.5281/zenodo.19840786)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20039913-blue)](https://doi.org/10.5281/zenodo.20039913)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20045771-blue)](https://doi.org/10.5281/zenodo.20045771)
[![OSF Registration](https://img.shields.io/badge/OSF-Registration-blue)](https://osf.io/sjk7b)

# ✅ System Properties

| Property | Status |
|---|---|
| Deterministic canonical hashing | Stable |
| Frozen proof artifacts | Stable |
| Replay-safe validation | Stable |
| Self-hash exclusion | Stable |
| Recomputed-hash validation | Stable |
| GameWorld interaction receipts | Complete through v156.6 |
| Perturbation/stress contracts | Complete through v157.3 |
| Substrate constraint receipts | Complete through v158.3 |
| Bounded recursive proof loops | Complete through v159.3 |
| Reality-loop composition | Complete through v160.2 |
| Global validation/truth/replay receipts | Complete through v161.2 |
| IRC operator replay receipts | Complete through v162.2 |
| Heavy-dependency invariant discovery | Complete through v163.4 |
| Invariant-based optimization receipts | Complete through v164.5 |
| Optimized simulation backend receipts | Complete through v165.4 |
| Release metadata safety tooling | Hardened through v165.3.3 |

# 🧠 What QEC Is

QEC is a deterministic proof system.

It does three things:

1. Receives structured inputs.
2. Converts them into canonical, hash-bound proof artifacts.
3. Revalidates those artifacts by recomputing their canonical hashes.

QEC is useful when you need proof objects that are:

- replay-safe
- tamper-evident
- deterministic
- canonical JSON based
- explicit about failure modes
- bounded rather than open-ended
- lineage-preserving across release arcs
- safe around external backends and heavy dependencies
- strict about simulation, benchmark, telemetry, and report boundaries

QEC does not make the world deterministic.

QEC makes the boundary deterministic.

Pipeline:

```text
input
→ canonical JSON
→ stable SHA-256 hash
→ frozen receipt
→ validator recomputes hash
→ replay-safe proof result
```

## ⚡ Quickstart

```bash
git clone https://github.com/QSOLKCB/QEC.git
cd QEC
python -m pip install -e ".[dev,science]"
```

For full setup, environment, and troubleshooting details, see
[`INSTALL.md`](INSTALL.md).

### Testing

```bash
python -m pip install -e ".[dev,science]"
pytest -q -ra
```

For full local validation, install the developer/science extras, then run
`pytest -q -ra`.

Rust TUI Control Surface: see [`USAGE.md`](USAGE.md).

## Commands

| Command | Purpose |
|---|---|
| `python scripts/sphaera_proof_demo.py` | Runs the SPHAERA proof demo |
| `qec-exp --help` | Shows packaged experiment CLI commands |
| `pytest -q` | Runs the deterministic test suite |
| `pytest -q -ra` | Shows skip/warning diagnostics |
| `curl -fsSL https://raw.githubusercontent.com/QSOLKCB/QEC/main/tui/install.sh \| sh` | Installs the Rust `qec-tui` launcher |


## IRC Operator Surface

- v162.x adds a local-only IRC operator surface.
- v162.0 provides the local IRC server core.
- v162.1 adds deterministic read-only commands.
- v162.2 adds replay audit receipts for local IRC command streams.
- Default bind is `127.0.0.1`.
- No command execution authority is granted.
- No LLM/API integration is enabled.

```bash
python scripts/qec_irc_server.py --host 127.0.0.1 --port 6667
```

Example IRC usage:

```text
/join #qec
!help
!corelaw
!hashchain
```

## Why This Matters

QEC matters because deterministic proof artifacts make it possible to:

- replay validation instead of trusting claims
- detect tampering through stable hashes
- represent messy external systems without executing them
- compare expected and observed proof chains
- bind long research arcs into a single canonical replay chain
- separate symbolic language from deterministic artifacts
- wrap heavy scientific dependencies as adapters rather than authorities
- prove optimization boundaries before accepting fast paths
- require replay equivalence before benchmark claims
- require benchmark receipts before telemetry/report claims
- produce simulation reports without executing simulations

The world may be chaotic.
The receipt must not be.

## Capability Summary

- v163.0 → HeavyDependencyDiscoveryManifest
- v163.1 → DependencyImportAndHotPathReceipt
- v163.2 → BackendInvariantCandidateReceipt
- v163.3 → CrossBackendEquivalenceReceipt
- v163.3.1 → Optional Scientific Dependency Test Normalization
- v163.4 → OptimizationOpportunityIndex
- v164.0 → OptimizationContract
- v164.1 → LightweightAdapterSpec
- v164.2 → CachedCanonicalKernelReceipt
- v164.3 → FastPathEquivalenceReceipt
- v164.4 → OptimizationImplementationReceipt
- v164.5 → DependencyReductionReceipt
- v165.0 → OptimizedSimulationSpec
- v165.1 → BackendEquivalenceReplayReceipt
- v165.2 → OptimizedQECBenchmarkReceipt
- v165.2.1 → Release Documentation Automation + README Boundary Enforcement
- v165.3 → OptimizedTelemetryReceipt
- v165.3.1 → Canonical Release History Reconstruction
- v165.3.2 → Release Metadata Infrastructure Hardening
- v165.3.3 → Canonical Release Manifest Infrastructure
- v165.4 → OptimizedSimulationReport

QEC now supports deterministic heavy-dependency discovery, static import /
hot-path receipts, backend invariant candidates, cross-backend equivalence,
optimization opportunity indexing, optimization contracts, lightweight adapter
specs, cached canonical kernel receipts, fast-path equivalence receipts,
implementation receipts, dependency-reduction receipts, optimized simulation
specs, backend replay receipts, benchmark receipts, telemetry receipts, and
final optimized simulation reports.

The v163.x arc discovered deterministic invariants in heavy dependencies.

The v164.x arc turned those invariants into replay-safe optimization contracts,
adapter/cache receipts, fast-path equivalence proofs, implementation receipts,
and dependency-reduction receipts.

The v165.x arc applies those contracts to optimized QEC simulation backends
through declarative specs, backend replay equivalence, bounded benchmark
receipts, telemetry receipts, and final report receipts.

v165.4 completes the optimized simulation backend reporting arc.

- External dependencies are adapters, never authorities.
- Simulation output is not truth.
- Simulation speed is not proof.
- Benchmarks require replay equivalence.
- Telemetry requires benchmark and replay lineage.
- Reports aggregate receipts; they do not execute systems.
- The decoder remains untouched.

## Proof Artifacts

QEC emits canonical JSON proof artifacts with stable SHA-256 identity and
validator-backed replay checks.

For full artifact families, schemas, and failure semantics, see
[`DOCUMENTATION.md`](DOCUMENTATION.md).

# 🧠 Core Law

```text
same input
→ same ordering
→ same canonical JSON
→ same stable SHA-256 hash
→ same bytes
→ same proof artifact
→ same outcome
```

Violation:

```text
SYSTEM INVALID
```

## 🧾 Attribution

If you build on QEC in publications, software, or research artifacts,
please cite both the repository and DOI-backed records above.

### Marc Brendecke — QAM / Quantum Sphaera Lineage

Selected QEC releases include informational attribution to work by **Marc Brendecke**.

**ORCID:** https://orcid.org/0009-0009-4034-598X

Relevant attributed works:

- **Quantum Sphaera Companion v3.30.0**  
  DOI: https://doi.org/10.5281/zenodo.19682951  
  License: CC-BY-4.0  
  Applied where relevant to the v143.x release lineage, including v143.5.

- **QAM Version v4.1.0**  
  Informational router/readout architecture lineage attribution.  
  Applied where relevant across the QEC release lineage from **v143.5 through v154.3**.

This attribution is informational only and does not influence canonical identity, hashing, proof semantics, receipt semantics, validation behavior, or release identity within QEC.

## References

- Roadmap and historical progression: [`ROADMAP.md`](ROADMAP.md)
- Install and test instructions: [`INSTALL.md`](INSTALL.md)
- Full system documentation: [`DOCUMENTATION.md`](DOCUMENTATION.md)
- Rust TUI installer/usage: [`USAGE.md`](USAGE.md)

## Author

**Author / Maintainer: Trent Slade, QSOL-IMC**

- ORCID: https://orcid.org/0009-0006-5966-1243
- GitHub: https://github.com/QSOLKCB
