#!/usr/bin/env python
"""Run deterministic experiment artifacts with optional parameter sweeps."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from typing import Any

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.qec.experiments.experiment_hash import ExperimentHash, ExperimentRunner
from src.qec.experiments.parameter_sweep import run_parameter_sweep


def _load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()

    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".json":
        return json.loads(text)

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "YAML config requested but PyYAML is not installed; use JSON or install PyYAML",
            ) from exc
        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise ValueError("YAML config must contain a top-level mapping")
        return loaded

    raise ValueError("Unsupported config file extension; use .json, .yaml, or .yml")


def _resolve_function(path: str):
    if ":" not in path:
        raise ValueError("function path must be in the format module.submodule:function_name")
    module_name, function_name = path.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    fn = getattr(module, function_name)
    if not callable(fn):
        raise TypeError(f"Resolved object is not callable: {path}")
    return fn


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic experiment artifact generation")
    parser.add_argument("config", help="Path to JSON/YAML config")
    args = parser.parse_args()

    config = _load_config(args.config)
    if isinstance(config.get("sweep"), dict):
        run_parameter_sweep(config)
        return 0

    callable_path = config.get("callable", config.get("function"))
    if not isinstance(callable_path, str):
        raise ValueError("Config must include string field 'callable' or 'function'")

    fn = _resolve_function(callable_path)
    artifacts_root = config.get("artifacts_root", config.get("output_root", "experiments"))
    runner = ExperimentRunner(artifacts_root=artifacts_root)
    run_config = {
        key: value
        for key, value in config.items()
        if key not in {"callable", "function", "artifacts_root", "output_root"}
    }
    result = runner.run(run_config, fn)
    exp_hash = ExperimentHash.compute(
        run_config,
        experiment_callable=f"{fn.__module__}:{fn.__name__}",
    )
    print(f"hash={exp_hash}")
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
