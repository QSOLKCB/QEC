"""Deterministic parameter sweep generation and execution."""

from __future__ import annotations

import importlib
import itertools
from typing import Any

from .experiment_hash import ExperimentRunner


def _resolve_callable(path: str):
    if ":" not in path:
        raise ValueError("callable must be in the format module.submodule:function_name")
    module_name, function_name = path.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    fn = getattr(module, function_name)
    if not callable(fn):
        raise TypeError(f"Resolved object is not callable: {path}")
    return fn


def generate_parameter_grid(sweep_dict: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate deterministic cartesian product configs from a sweep mapping."""
    sorted_items = sorted(sweep_dict.items(), key=lambda item: item[0])
    keys = [key for key, _ in sorted_items]
    value_lists = [list(values) for _, values in sorted_items]
    return [dict(zip(keys, values)) for values in itertools.product(*value_lists)]


def run_parameter_sweep(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Run each deterministic parameter combination through the experiment cache runner."""
    callable_path = config.get("callable", config.get("function"))
    if not isinstance(callable_path, str):
        raise ValueError("Config must include string field 'callable' or 'function'")
    sweep = config.get("sweep")
    if not isinstance(sweep, dict):
        raise ValueError("Config must include mapping field 'sweep'")

    base_config = {
        key: value
        for key, value in config.items()
        if key not in {"callable", "function", "sweep", "artifacts_root", "output_root"}
    }
    combinations = generate_parameter_grid(sweep)
    execute_fn = _resolve_callable(callable_path)
    artifacts_root = config.get("artifacts_root", config.get("output_root", "experiments"))
    runner = ExperimentRunner(artifacts_root=artifacts_root)

    print(f"Running sweep with {len(combinations)} configurations")
    results: list[dict[str, Any]] = []
    for index, combination in enumerate(combinations, start=1):
        run_config = dict(base_config)
        run_config.update(combination)
        result = runner.run(run_config, execute_fn)
        print(f"[{index}/{len(combinations)}] completed")
        results.append(result)
    return results
