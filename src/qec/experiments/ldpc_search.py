"""Minimal deterministic ldpc-search experiment."""

from __future__ import annotations

from typing import Any

from .registry import register_experiment


@register_experiment("ldpc-search")
def run(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 0))
    return {
        "experiment": "ldpc-search",
        "seed": seed,
        "status": "ok",
    }
