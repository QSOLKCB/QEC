"""Deterministic experiment hashing and cache-aware execution utilities."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import scipy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentEnvironmentMetadata:
    """Environment fields included in the deterministic experiment hash."""

    git_commit: str
    repo_version: str
    python_version: str
    numpy_version: str
    scipy_version: str
    dirty_repo: bool | str


class ExperimentHash:
    """Build deterministic hashes for experiment configurations."""

    @staticmethod
    def canonical_json(value: Any) -> str:
        """Serialize as canonical JSON with deterministic ordering."""
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    @classmethod
    def collect_environment_metadata(cls) -> ExperimentEnvironmentMetadata:
        """Collect deterministic environment metadata fields for hashing."""
        return ExperimentEnvironmentMetadata(
            git_commit=cls._get_git_commit(),
            repo_version=cls._get_repo_version(),
            python_version=platform.python_version(),
            numpy_version=np.__version__,
            scipy_version=scipy.__version__,
            dirty_repo=cls._get_dirty_repo(),
        )

    @classmethod
    def build_payload(
        cls,
        config: dict[str, Any],
        experiment_callable: str,
        metadata: ExperimentEnvironmentMetadata | None = None,
    ) -> dict[str, Any]:
        """Build the canonical payload used for hash computation."""
        env = metadata or cls.collect_environment_metadata()
        return {
            "experiment_callable": experiment_callable,
            "config": config,
            "git_commit": env.git_commit,
            "repo_version": env.repo_version,
            "python_version": env.python_version,
            "numpy_version": env.numpy_version,
            "scipy_version": env.scipy_version,
            "dirty_repo": env.dirty_repo,
        }

    @classmethod
    def compute(
        cls,
        config: dict[str, Any],
        experiment_callable: str = "unknown",
        metadata: ExperimentEnvironmentMetadata | None = None,
        length: int = 16,
    ) -> str:
        """Compute a deterministic short SHA256 hash identifier."""
        payload = cls.build_payload(
            config=config,
            experiment_callable=experiment_callable,
            metadata=metadata,
        )
        canonical = cls.canonical_json(payload)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return digest[:length]

    @staticmethod
    def _get_git_commit() -> str:
        try:
            return (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    stderr=subprocess.DEVNULL,
                )
                .decode("utf-8")
                .strip()
            )
        except Exception:
            return "unknown"

    @staticmethod
    def _get_repo_version() -> str:
        try:
            if sys.version_info >= (3, 11):
                import tomllib
            else:  # pragma: no cover - supported runtime is >=3.11
                import tomli as tomllib
            repo_root = Path(__file__).resolve().parents[3]
            pyproject_path = repo_root / "pyproject.toml"
            if pyproject_path.exists():
                with pyproject_path.open("rb") as f:
                    pyproject = tomllib.load(f)
                project = pyproject.get("project", {})
                version = project.get("version")
                if isinstance(version, str) and version:
                    return version
        except Exception:
            pass
        return "unknown"

    @staticmethod
    def _get_dirty_repo() -> bool | str:
        try:
            output = subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
            ).decode("utf-8")
            return bool(output.strip())
        except Exception:
            return "unknown"


class ExperimentRunner:
    """Deterministic experiment runner with hash-based result cache."""

    def __init__(self, artifacts_root: str | Path = "experiments") -> None:
        self.artifacts_root = Path(artifacts_root)

    def run(
        self,
        config: dict[str, Any],
        execute_fn: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        """Run an experiment or reuse cached results for identical inputs."""
        callable_name = f"{execute_fn.__module__}:{execute_fn.__name__}"
        environment_metadata = ExperimentHash.collect_environment_metadata()
        experiment_hash = ExperimentHash.compute(
            config,
            experiment_callable=callable_name,
            metadata=environment_metadata,
        )
        logger.info("Experiment hash: %s", experiment_hash)

        experiment_dir = self.artifacts_root / experiment_hash
        metadata_path = experiment_dir / "metadata.json"
        config_path = experiment_dir / "config.json"
        results_path = experiment_dir / "results.json"

        if self._has_cache(metadata_path, config_path, results_path):
            logger.info("Cached results found — skipping execution")
            with results_path.open("r", encoding="utf-8") as f:
                return json.load(f)

        logger.info("No cache found — running experiment")
        results = execute_fn(config)

        experiment_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "experiment_name": config.get("experiment_name", "unknown"),
            "experiment_hash": experiment_hash,
            "experiment_callable": callable_name,
            "git_commit": environment_metadata.git_commit,
            "repo_version": environment_metadata.repo_version,
            "dirty_repo": environment_metadata.dirty_repo,
            "python": environment_metadata.python_version,
            "numpy": environment_metadata.numpy_version,
            "scipy": environment_metadata.scipy_version,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

        self._write_json(config_path, config)
        self._write_json(results_path, results)
        self._write_json(metadata_path, metadata)
        return results

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _has_cache(metadata_path: Path, config_path: Path, results_path: Path) -> bool:
        return metadata_path.exists() and config_path.exists() and results_path.exists()
