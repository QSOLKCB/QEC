# QSOLKCB / QEC

QEC is a deterministic, replay-safe reasoning system for quantum error correction
and invariant-driven computation.

It turns computation, evidence, semantic alignment, governance, replay
validation, lattice topology, router/readout proof, multi-scale invariance,
entropy drift analysis, GameWorld interaction boundaries,
perturbation/stress contracts, substrate constraints, bounded recursive loops,
reality-loop composition, and global replay proofs into canonical proof
artifacts.

The repository began with a Layer 1 QEC decoder and now extends that foundation
into a broader deterministic proof stack.

In this README, QEC means the QSOLKCB/QEC software system and release lineage.

**Deterministic Reasoning • Canonical Identity • RES/RAG Semantics • Governance • Distributed Proof • Atomic Lattices • Router Paths • Readout Projections • SearchMask64 • Hilber/Hilbert Shift • Functional Readout Shells • Replay Validation • Multi-Scale Invariance • Entropy Drift • Decay-Resistance Proofs • GameWorld Interaction Reports • Perturbation Contracts • Substrate Constraints • Recursive Proof Loops • Reality Loop Proofs • Global Replay Proofs**

## 📦 Release & Research

[![Release](https://img.shields.io/github/v/release/QSOLKCB/QEC)](https://github.com/QSOLKCB/QEC/releases)
[![Latest](https://img.shields.io/badge/stable-v165.2.1-success)](https://github.com/QSOLKCB/QEC/releases/tag/v164.2)
[![Branch](https://img.shields.io/badge/branch-v164.2%20canonical-purple)]()

Current release line: **v165.2.1**  
Current frontier: **v165.3 — ReleaseDocumentationFrontier**  
Active arc: **v164.x — Invariant-Based Heavy Dependency Optimization**  
Completed arc: **v164.x — Invariant-Based Heavy Dependency Optimization**

Repository status is current through **v165.2.1**.

Recent preprint: **QSOLKCB/QEC v155.x: Deterministic Entropy and Decay
Signatures — Hash-Bound Checkpoints, Subsystem Drift Receipts, and
Replay-Resistant Proof Chains**.

## 📚 DOIs

[OSF Registration](https://osf.io/sjk7b)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19697907-blue)](https://doi.org/10.5281/zenodo.19697907)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19062692-blue)](https://doi.org/10.5281/zenodo.19062692)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19102390-blue)](https://doi.org/10.5281/zenodo.19102390)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19099503-blue)](https://doi.org/10.5281/zenodo.19099503)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19104208-blue)](https://doi.org/10.5281/zenodo.19104208)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19840786-blue)](https://doi.org/10.5281/zenodo.19840786)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20039913-blue)](https://doi.org/10.5281/zenodo.20039913)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20045771-blue)](https://doi.org/10.5281/zenodo.20045771)

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

The world may be chaotic.
The receipt must not be.

## v163.x → v164.x Capability Summary

- v163.0 → HeavyDependencyDiscoveryManifest
- v163.1 → DependencyImportAndHotPathReceipt
- v163.2 → BackendInvariantCandidateReceipt
- v163.3 → CrossBackendEquivalenceReceipt
- v163.3.1 → Optional Scientific Dependency Test Normalization
- v163.4 → OptimizationOpportunityIndex
- v164.0 → OptimizationContract
- v164.1 → LightweightAdapterSpec
- v164.2 → CachedCanonicalKernelReceipt

QEC now supports deterministic heavy-dependency discovery, static import /
hot-path receipts, backend invariant candidate receipts, cross-backend
equivalence receipts, optimization opportunity indexing, optimization
contracts, lightweight adapter specs, and cached canonical kernel receipts.

The v164.x arc turns discovered invariants into deterministic optimization
contracts and adapter/cache receipts. These artifacts define replay-safe
optimization boundaries before any runtime fast path exists.

v164.2 does **not** implement runtime caches, fast paths, backend execution,
benchmarking, dependency reduction, or optimized simulation execution.

- External dependencies are adapters, never authorities.
- Speedups require benchmark receipts.
- Optimizations require equivalence receipts.
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
