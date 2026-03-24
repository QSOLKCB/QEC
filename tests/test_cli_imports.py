"""
Regression test: bench_reports/dps_v381_eval.py resolves imports from repo root.

Verifies the path-setup logic at the top of the harness allows
importing qec modules without setting PYTHONPATH manually.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestCLIImports:
    """Verify harness can import qec modules from repo root."""

    def test_harness_importable(self):
        """Import the harness module without PYTHONPATH set."""
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, %r); sys.path.insert(0, %r); "
             "from bench_reports.dps_v381_eval import _parse_args"
             % (_repo_root, os.path.join(_repo_root, "src"))],
            capture_output=True, text=True,
            env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        )
        assert result.returncode == 0, result.stderr

    def test_harness_help_runs(self):
        """Harness --help exits cleanly without PYTHONPATH."""
        result = subprocess.run(
            [sys.executable, os.path.join(_repo_root, "bench_reports", "dps_v381_eval.py"),
             "--help"],
            capture_output=True, text=True,
            env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        )
        assert result.returncode == 0, result.stderr
        assert "spectral-optimizer-sanity" in result.stdout
