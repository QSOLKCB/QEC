"""Minimal deterministic bp-threshold experiment."""

from __future__ import annotations

from typing import Any

from .registry import register_experiment


@register_experiment("bp-threshold")
def run(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 0))
    return {
        "experiment": "bp-threshold",
        "seed": seed,
        "status": "ok",
    }
