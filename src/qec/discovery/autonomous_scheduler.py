"""Deterministic autonomous candidate scheduler with curriculum strategy."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.qec.analysis.spectral_difficulty import estimate_spectral_difficulty


def schedule_candidate(
    candidates: list[dict[str, Any]],
    *,
    strategy: str = "default",
    curriculum_controller: Any = None,
    region_tiers: dict[str, list[int]] | None = None,
    memory: Any = None,
    model: Any = None,
) -> dict[str, Any] | None:
    """Select one deterministic candidate according to strategy."""
    if not candidates:
        return None

    pool = list(candidates)
    if (
        strategy == "curriculum_exploration"
        and curriculum_controller is not None
        and region_tiers is not None
    ):
        filtered = curriculum_controller.filter_candidates(pool, region_tiers)
        if filtered:
            pool = filtered

    scored: list[tuple[float, float, str, dict[str, Any]]] = []
    for c in pool:
        spectrum = np.asarray(c.get("spectrum", []), dtype=np.float64)
        difficulty = float(estimate_spectral_difficulty(memory, spectrum))
        uncertainty = float(np.float64(c.get("bayesian_uncertainty", 0.0)))
        information_gain = uncertainty / (1.0 + difficulty)
        if strategy == "curriculum_exploration":
            rank_primary = -information_gain
        else:
            rank_primary = float(c.get("objectives", {}).get("composite_score", float("inf")))
        rank_secondary = difficulty
        candidate_id = str(c.get("candidate_id", ""))
        scored.append((rank_primary, rank_secondary, candidate_id, c))

    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    return scored[0][3]
