"""
Attribution:
This demo traverses modules incorporating concepts from:
Marc Brendecke (2026)
Quantum Sphaera Companion v3.30.0
DOI: https://doi.org/10.5281/zenodo.19682951
License: CC-BY-4.0

Aligned with:
Slade, T. (2026)
IRIS: Deterministic Invariant-Driven Reduction of Redundant Computation
DOI: https://doi.org/10.5281/zenodo.19697907
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


DEMO_TIMEOUT_SECONDS = 60


def _run_script() -> bytes:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "sphaera_proof_demo.py"
    try:
        completed = subprocess.run(
            [sys.executable, str(script)],
            check=True,
            capture_output=True,
            timeout=DEMO_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        raise AssertionError(
            f"sphaera_proof_demo.py timed out after {DEMO_TIMEOUT_SECONDS}s.\n"
            f"stderr:\n{stderr}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        raise AssertionError(
            f"sphaera_proof_demo.py failed with exit code {exc.returncode}.\n"
            f"stderr:\n{stderr}"
        ) from exc
    return completed.stdout


@pytest.fixture(scope="module")
def sphaera_proof_demo_output() -> bytes:
    """Run the demo script once and share its output across related tests."""
    return _run_script()


def test_sphaera_proof_demo_is_byte_identical_across_runs(
    sphaera_proof_demo_output: bytes,
) -> None:
    first = sphaera_proof_demo_output
    second = _run_script()
    assert first == second


def test_sphaera_proof_demo_sections_present(
    sphaera_proof_demo_output: bytes,
) -> None:
    output = sphaera_proof_demo_output.decode("utf-8")
    assert "SPHAERA Proof Artifact Demo (v143.5)" in output
    assert "=== DOMAIN: transformers ===" in output
    assert "=== DOMAIN: diffusion ===" in output
    assert "=== DOMAIN: gnn ===" in output
    assert "=== DOMAIN: physics ===" in output
    assert "=== SPHAERA TABLE ===" in output
    assert "=== DETERMINISM CHECK ===" in output
    assert "=== INTERPRETATION ===" in output
