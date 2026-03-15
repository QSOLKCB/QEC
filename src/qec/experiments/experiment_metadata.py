"""
v18.0.0 — Deterministic Experiment Metadata.

Collects deterministic environment metadata for experiment artifacts.
Layer 5 (experiments): informational only, no decoder behavior changes.
"""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy
import scipy


def git_commit() -> str:
    """Return git commit hash for HEAD, or ``"unknown"`` when unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except Exception:
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    commit = result.stdout.strip()
    return commit if commit else "unknown"


def _repo_version_from_pyproject() -> str | None:
    pyproject_path = Path(__file__).resolve().parents[3] / "pyproject.toml"
    try:
        text = pyproject_path.read_text(encoding="utf-8")
    except OSError:
        return None

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("version"):
            parts = stripped.split("=", maxsplit=1)
            if len(parts) != 2:
                continue
            value = parts[1].strip().strip('"').strip("'")
            if value:
                return value
    return None


def repo_version() -> str:
    """Return repository version from pyproject, then ``qec.__version__``, then unknown."""
    version = _repo_version_from_pyproject()
    if version is not None:
        return version

    try:
        import src.qec as qec_package

        package_version = getattr(qec_package, "__version__", None)
        if isinstance(package_version, str) and package_version:
            return package_version
    except Exception:
        pass
    return "unknown"


class ExperimentMetadata:
    """Collector for deterministic experiment metadata."""

    def __init__(self, seed: int, timestamp: str | None = None) -> None:
        self.seed = int(seed)
        self._timestamp = timestamp

    def collect(self) -> dict[str, Any]:
        timestamp = self._timestamp
        if timestamp is None:
            timestamp = (
                datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            )

        return {
            "repo_version": repo_version(),
            "git_commit": git_commit(),
            "timestamp": timestamp,
            "python_version": sys.version.split()[0],
            "numpy_version": numpy.__version__,
            "scipy_version": scipy.__version__,
            "seed": self.seed,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
        }

    @staticmethod
    def deterministic_json(data: dict[str, Any]) -> str:
        """Serialize metadata deterministically."""
        return json.dumps(data, sort_keys=True, indent=2)
