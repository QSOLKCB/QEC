# QSOLKCB / QEC

Deterministic quantum error-correction software, replayable evidence, and
classical syndrome-transport experiments. QEC combines exact finite-code models
with canonical JSON / SHA-256 receipts and validators that reconstruct results.

[![Latest](https://img.shields.io/badge/stable-v177.1-success)](https://github.com/QSOLKCB/QEC/releases/tag/v177.1)
[![Tests](https://github.com/QSOLKCB/QEC/actions/workflows/build-test.yml/badge.svg?branch=main)](https://github.com/QSOLKCB/QEC/actions/workflows/build-test.yml)
[![License: MPL-2.0](https://img.shields.io/badge/license-MPL--2.0-blue)](LICENSE)

## Release

<!-- README-RELEASE:COMPACT -->
Current release line: **v177.1**

Includes sparse qutrit construction and opt-in CSS / syndrome-delivery research.
See the [release history](https://github.com/QSOLKCB/QEC/releases) for changes and
verification records. Release labels and contract milestones are separate from
Python package metadata, which currently remains `173.0.0`.

## Try the labs

Open the browser-native [Qutrit Harmonic Lab](https://qsolkcb.github.io/QEC/) or
[Ququart Evidence Lab](https://qsolkcb.github.io/QEC/ququart-fer/).
For local/offline use, see [Qutrit](viz/index.html), [Ququart](viz/ququart-fer/index.html)
and the [Acoustic Lab guide](docs/ACOUSTIC_OFFLINE_LAB.md).

## Quick start

Requires Python 3.10+ and Git. From a terminal on Linux or macOS:

```sh
git clone https://github.com/QSOLKCB/QEC.git
cd QEC
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
qec-qutrit
python scripts/generate_qec_research_evidence.py --check
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell.
`qec-qutrit` prints an exact bounded-error certificate; the research check verifies
the retained synthetic fixture. See [installation and troubleshooting](INSTALL.md)
for additional dependencies, and [Rust TUI usage](USAGE.md) for the operator UI.

For the optional Rust TUI launcher, review the [installer](tui/install.sh) before running:

```sh
curl -fsSL https://raw.githubusercontent.com/QSOLKCB/QEC/main/tui/install.sh | sh
```

### Tests

```sh
python -m pytest -q
```

Optional scientific/external backends may be skipped unless their extras are
installed. Use `python -m pytest -q -ra` to inspect skips; see
[validation boundaries](docs/CI_VALIDATION_BOUNDARIES.md).

## What is included

- **Exact qutrit and ququart models:** finite-code correction and FER evidence.
  [Qutrit](docs/QUTRIT_HARMONIC_QEC.md) · [Ququart](docs/QUQUART_FOUR_STATE_QEC.md)
- **Replayable receipts:** canonical artifacts, source-bound claims, replication
  checks and decoder-governance boundaries. [System documentation](DOCUMENTATION.md)
- **Six classical switching models:** Strowger, Panel, Crossbar, ESS, digital TDM
  and packet routing, with cross-era replay. [Contracts](docs/CROSS_ERA_MIGRATION_AND_SELECTION.md)
- **Acoustic syndrome transport:** sampled classical waveforms, receiver checks
  and exact-model correction. [Bridge and limits](docs/ACOUSTIC_SYNDROME_BRIDGE.md)
- **Opt-in research:** binary CSS presentation witnesses and synthetic delivery
  replay using an unchanged reference decoder. [Scope and reproduction](docs/research/CSS_AND_DELIVERY_EXPERIMENTS.md)

These are bounded software models and experiments. They do not establish physical
fault tolerance, hardware thresholds, quantum-state transfer, or quantum advantage.
Hashes bind declared bytes; they do not by themselves prove authenticity or
scientific correctness. Governance receipts do not activate a decoder. QEC OS
runtime/benchmark work remains deferred to v193.x; see the [roadmap](ROADMAP.md).

## Documentation

- [Architecture and safety boundaries](ARCHITECTURE.md) · [Contributor instructions](CLAUDE.md)
- [Commands and capability metadata](docs/CAPABILITIES.md) · [IRC operator guide](docs/IRC_BOT_GUIDE.md)
- [Qutrit construction benchmark and limits](docs/QUTRIT_CONSTRUCTION_BENCHMARK.md)
- [Roadmap](ROADMAP.md) · [Detailed historical overview](docs/archive/README_BEFORE_V177_1_CLEANUP.md)
- [Citation, DOI records and attribution](docs/CITATION_AND_ATTRIBUTION.md)

Maintained by **Trent Slade, QSOL-IMC**. Licensed under [MPL-2.0](LICENSE).
