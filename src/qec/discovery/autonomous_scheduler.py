"""Deterministic autonomous scheduler with optional information-gain strategy."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.qec.analysis.information_gain import rank_candidates_by_information_gain


def schedule_autonomous_target(
    memory: object,
    frontier_spectra: list[np.ndarray | list[float]],
    *,
    strategy: str = "frontier",
    novelty_weight: float = 0.5,
    uncertainty_weight: float = 0.5,
) -> dict[str, Any]:
    """Select a deterministic target spectrum from candidate frontier spectra."""
    candidates = [np.asarray(s, dtype=np.float64) for s in frontier_spectra]
    if not candidates:
        return {
            "target_spectrum": None,
            "information_gain_score": 0.0,
            "novelty_score": 0.0,
            "spectral_uncertainty": 0.0,
            "strategy": str(strategy),
        }

    if strategy == "information_gain":
        ranked = rank_candidates_by_information_gain(
            candidates,
            memory,
            novelty_weight=novelty_weight,
            uncertainty_weight=uncertainty_weight,
        )
        best = ranked[0]
        return {
            "target_spectrum": best["target_spectrum"],
            "information_gain_score": float(best["information_gain_score"]),
            "novelty_score": float(best["novelty_score"]),
            "spectral_uncertainty": float(best["spectral_uncertainty"]),
            "strategy": "information_gain",
        }

    first = candidates[0]
    return {
        "target_spectrum": first,
        "information_gain_score": 0.0,
        "novelty_score": 0.0,
        "spectral_uncertainty": 0.0,
        "strategy": str(strategy),
    }
