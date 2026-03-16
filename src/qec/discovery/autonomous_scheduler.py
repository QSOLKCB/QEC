"""Deterministic autonomous scheduler with optional information-gain strategy."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.qec.analysis.information_gain import rank_candidates_by_information_gain
from src.qec.analysis.landscape_gaps import detect_landscape_gaps
from src.qec.discovery.experiment_targets import choose_experiment_target


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
    archive,
    landscape_model=None,
    exploration_state=None,
    gap_radius=None,
    **kwargs,
):
    """Deterministically schedule a landscape exploration target."""
    radius = 0.3 if gap_radius is None else float(gap_radius)
    max_gaps = int(kwargs.get("max_gaps", 16))

    gap_candidates = detect_landscape_gaps(archive, gap_radius=radius, max_gaps=max_gaps)
    target = choose_experiment_target(gap_candidates)

    return {
        "target_spectrum": (
            None
            if target is None
            else np.asarray(target, dtype=np.float64)
        ),
        "gap_count": int(len(gap_candidates)),
        "strategy": "landscape_exploration",
    }
