"""
v18.0.0 — Deterministic Experiment Runner.

Opt-in wrapper that records metadata/config/results for reproducible runs.
Does not modify discovery experiment behavior.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .experiment_metadata import ExperimentMetadata


class ExperimentRunner:
    """Run an experiment function and save deterministic artifacts."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config)

    def run(self, discovery_function: Callable[..., Any]) -> dict[str, Any]:
        seed = int(self.config.get("seed", 0))
        np.random.seed(seed)

        metadata = ExperimentMetadata(seed=seed).collect()
        runtime_parameters = dict(self.config)

        kwargs = self.config.get("function_kwargs", {})
        if not isinstance(kwargs, dict):
            raise TypeError("function_kwargs must be a dictionary")

        results = discovery_function(**kwargs)

        run_name = str(self.config.get("run_name", "run"))
        base_dir = Path(self.config.get("output_root", "experiments"))
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        artifact_dir = base_dir / f"{day}_{run_name}"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        metadata_payload = {
            "runtime_parameters": runtime_parameters,
            **metadata,
        }
        self._write_json(artifact_dir / "metadata.json", metadata_payload)
        self._write_json(artifact_dir / "config.json", runtime_parameters)
        self._write_json(artifact_dir / "results.json", results)

        return {
            "artifact_dir": str(artifact_dir),
            "metadata": metadata_payload,
            "config": runtime_parameters,
            "results": results,
        }

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        serialized = json.dumps(payload, sort_keys=True, indent=2)
        path.write_text(f"{serialized}\n", encoding="utf-8")
