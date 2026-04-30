"""
v9.2.0 — Reproducibility Metadata.

Collects environment metadata for artifact reproducibility.
Records repo version, git commit, Python version, and dependency versions.

Does not modify any experiment state.  Pure informational utility.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

import numpy
import scipy


def _get_git_commit() -> str:
    """Return the current git HEAD commit hash, or 'unknown'."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def collect_environment_metadata(
    spec: dict[str, Any] | None = None,
    generation_count: int | None = None,
    population_size: int | None = None,
) -> dict[str, Any]:
    """Collect environment metadata for reproducibility.

    Parameters
    ----------
    spec : dict or None
        Optional generation specification to embed.
    generation_count : int or None
        Optional generation count to embed.
    population_size : int or None
        Optional population size to embed.

    Returns
    -------
    dict[str, Any]
        Metadata dictionary with repo_version, git_commit,
        python_version, numpy_version, scipy_version, timestamp,
        and optional spec/generation_count/population_size.
    """
    metadata: dict[str, Any] = {
        "repo_version": "9.2.0",
        "git_commit": _get_git_commit(),
        "python_version": sys.version.split()[0],
        "numpy_version": numpy.__version__,
        "scipy_version": scipy.__version__,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    if spec is not None:
        metadata["spec"] = spec

    if generation_count is not None:
        metadata["generation_count"] = generation_count

    if population_size is not None:
        metadata["population_size"] = population_size

    return metadata
