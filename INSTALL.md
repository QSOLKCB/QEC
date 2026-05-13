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

Install QEC plus developer and scientific extras:

```bash
python -m pip install -e ".[dev,science]"
```

Scientific extras may include packages used by test and analysis paths,
including `scipy`, `pandas`, `matplotlib`, `qutip`, and `qiskit`.

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
curl -fsSL https://raw.githubusercontent.com/QSOLKCB/qec-tui/main/install.sh | sh
```

## Troubleshooting

- **`qec-tui` tests skip**: local binary may be missing. Install via `USAGE.md`
  if you need runtime launch tests.
- **`gh` mirror tests skip**: GitHub CLI (`gh`) is optional and required only for
  mirror-tool tests.
- **Scientific warning from `scipy.linalg.logm`**: a known singular-matrix
  warning can appear in one entropy test path; it is not caused by missing
  SciPy.
- **Missing optional backends**: some tests intentionally skip when optional
  backend packages are absent.
