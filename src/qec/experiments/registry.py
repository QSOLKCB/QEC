"""Deterministic experiment registry and discovery helpers."""

from __future__ import annotations

import importlib
from collections.abc import Callable


class ExperimentRegistry:
    """Registry mapping deterministic experiment names to callables."""

    def __init__(self) -> None:
        self._registry: dict[str, Callable] = {}

    def register(self, name: str, callable_obj: Callable) -> None:
        key = str(name)
        if key in self._registry:
            raise ValueError(f"experiment '{key}' is already registered")
        self._registry[key] = callable_obj

    def get(self, name: str) -> Callable:
        key = str(name)
        if key not in self._registry:
            raise KeyError(f"unknown experiment '{key}'")
        return self._registry[key]

    def list(self) -> list[str]:
        return sorted(self._registry.keys())


DEFAULT_REGISTRY = ExperimentRegistry()


def register_experiment(name: str):
    """Register an experiment function at import time."""

    def _decorator(func: Callable) -> Callable:
        DEFAULT_REGISTRY.register(name, func)
        return func

    return _decorator


_KNOWN_EXPERIMENT_MODULES = (
    "src.qec.experiments.bp_threshold",
    "src.qec.experiments.spectral_heatmap",
    "src.qec.experiments.ldpc_search",
)


def discover_experiments() -> None:
    """Import known experiment modules so decorators execute deterministically."""
    for module_name in _KNOWN_EXPERIMENT_MODULES:
        importlib.import_module(module_name)
