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


def schedule_next_experiment(
    memory: object,
    *,
    gap_radius: float = 0.3,
    max_gaps: int = 16,
) -> dict[str, Any]:
    """Backward-compatible deterministic scheduler entry point."""
    centers_fn = getattr(memory, "centers", None)
    spectra = []
    if callable(centers_fn):
        centers = np.asarray(centers_fn(), dtype=np.float64)
        if centers.ndim == 2 and centers.shape[0] > 0:
            spectra = [centers[i] for i in range(centers.shape[0])]
    scheduled = schedule_autonomous_target(memory, spectra, strategy="frontier")
    target = scheduled.get("target_spectrum")
    return {
        "target_spectrum": target,
        "strategy": "landscape_exploration",
        "gap_count": int(min(max_gaps, len(spectra))),
        "gap_radius": float(np.float64(gap_radius)),
    }
