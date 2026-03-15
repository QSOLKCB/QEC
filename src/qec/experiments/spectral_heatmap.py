"""Minimal deterministic spectral-heatmap experiment."""

from __future__ import annotations

from typing import Any

from .registry import register_experiment


@register_experiment("spectral-heatmap")
def run(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 0))
    return {
        "experiment": "spectral-heatmap",
        "seed": seed,
        "status": "ok",
    }
