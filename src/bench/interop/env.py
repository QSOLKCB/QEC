"""
Deterministic environment capture for interop benchmarks.

Captures tool versions, platform info, and git commit when available.
When ``deterministic=True``, machine-dependent fields are replaced with
fixed placeholders to enable byte-identical artifact comparison.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any

import numpy as np

from .imports import tool_versions


def _git_commit() -> str | None:
    """Return the current git HEAD commit hash, or None."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def capture_environment(*, deterministic: bool = False) -> dict[str, Any]:
    """Capture a snapshot of the current environment.

    Parameters
    ----------
    deterministic:
        If True, replace machine-dependent fields with fixed placeholders
        so that the resulting dict is identical across machines.
    """
    try:
        from importlib.metadata import version as _pkg_version
        qec_version = _pkg_version("qec")
    except Exception:
        qec_version = "unknown"

    if deterministic:
        return {
            "platform": "DETERMINISTIC",
            "python_version": "DETERMINISTIC",
            "numpy_version": "DETERMINISTIC",
            "qec_version": str(qec_version),
            "git_commit": "DETERMINISTIC",
            "tool_versions": {k: "DETERMINISTIC" for k in tool_versions()},
        }

    git = _git_commit()
    return {
        "platform": platform.platform(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "qec_version": str(qec_version),
        "git_commit": git,
        "tool_versions": tool_versions(),
    }
