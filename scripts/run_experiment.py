#!/usr/bin/env python
"""Run an opt-in deterministic experiment via ExperimentRunner."""

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

from src.qec.experiments.experiment_runner import ExperimentRunner


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
    function_path = config.get("function")
    if not isinstance(function_path, str):
        raise ValueError("Config must include string field 'function'")

    fn = _resolve_function(function_path)
    runner = ExperimentRunner(config)
    result = runner.run(fn)
    print(result["artifact_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
