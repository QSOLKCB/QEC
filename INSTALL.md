# Installation Guide

## Requirements

- Python >= 3.10
- numpy >= 1.24
- scipy >= 1.10

Optional (development):

- pytest >= 7.0
- pytest-repeat >= 0.9
- matplotlib >= 3.5

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install in editable mode

```bash
pip install -e .
```

This installs the `qec` package and the `qec-exp` CLI entry point.
Dependencies (numpy, scipy) are installed automatically from `pyproject.toml`.

### 3. Install development dependencies (optional)

```bash
pip install -e ".[dev]"
```

## Verification

Run the test suite:

```bash
pytest tests/
```

Run the demo script:

```bash
python scripts/qec_demo.py
```

Verify the CLI entry point:

```bash
qec-exp --help
```

## Troubleshooting

- If `import qec` fails, ensure you ran `pip install -e .` from the repository root.
- If tests fail with import errors, confirm your virtual environment is activated and the editable install completed without errors.
- The project has no binary or platform-specific dependencies beyond numpy and scipy.
