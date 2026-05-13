# INSTALL

## Overview

This guide covers environment setup, installation, testing, and common
QEC developer workflows for the v161.2 release line.

## Python requirements

- Python 3.10+
- pip (latest recommended)
- virtual environment tooling (`venv`)

## Create a virtual environment

```bash
git clone https://github.com/QSOLKCB/QEC.git
cd QEC
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## Install package and dependencies

Install QEC with deterministic developer extras:

```bash
python -m pip install -e ".[dev]"
```

For optional scientific dependencies (`qutip`, `qiskit`, `pandas`, etc.), use
authoritative upstream releases pinned to reviewed versions before enabling
tests that require them. Avoid unrestricted `pip install .[science]` resolution
for deterministic validation/proof workflows.

## Run tests

```bash
pytest -q
pytest -q -ra
```

## Run targeted tests

```bash
pytest -q tests/test_tui_installer_flow.py
pytest -q tests/test_global_replay_proof.py
pytest -q tests/test_global_truth_receipt.py
pytest -q tests/test_global_validation_index.py
```

## Run the SPHAERA proof demo

```bash
python scripts/sphaera_proof_demo.py
```

## Rust TUI Control Surface

Rust `qec-tui` installation and usage are documented in [`USAGE.md`](USAGE.md).

Canonical installer command:

```bash
curl -fsSL https://raw.githubusercontent.com/QSOLKCB/QEC/main/tui/install.sh | sh
```

## Troubleshooting

- **`qec-tui` test binary source**: tests use a deterministic local stub by
  default. To run against a real install, set `QEC_TUI_USE_SYSTEM_BIN=1` and
  ensure `qec-tui` is on `PATH`.
- **`gh` mirror tests**: tests inject a deterministic `gh` stub and do not
  require a host-installed GitHub CLI binary.
- **Scientific warning from `scipy.linalg.logm`**: a known singular-matrix
  warning can appear in one entropy test path; it is not caused by missing
  SciPy.
- **`qldpc` CSS construction skip**: `tests/test_css_construction.py` is
  intentionally optional until `qldpc` is normalized into the repository's
  approved upstream-source dependency workflow. QEC intentionally avoids
  unrestricted PyPI dependency resolution; external scientific dependencies are
  expected to come from authoritative upstream repositories and pass explicit
  deterministic review before inclusion.
