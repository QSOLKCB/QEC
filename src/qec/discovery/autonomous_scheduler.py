"""Deterministic autonomous scheduler strategies for discovery targets."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.qec.analysis.information_gain import rank_candidates_by_information_gain
from src.qec.analysis.landscape_gaps import detect_landscape_gaps
from src.qec.discovery.experiment_targets import choose_experiment_target


def schedule_next_experiment(
    memory: object,
    *,
    gap_radius: float = 0.3,
    max_gaps: int = 16,
) -> dict[str, Any]:
    """Schedule deterministic target from spectral landscape gaps."""
    gaps = detect_landscape_gaps(memory, gap_radius=float(gap_radius), max_gaps=int(max_gaps))
    target = choose_experiment_target(gaps)
    return {
        "target_spectrum": None if target is None else np.asarray(target, dtype=np.float64),
        "gap_count": int(len(gaps)),
        "strategy": "landscape_exploration",
    }


def schedule_autonomous_target(
    memory: object,
    frontier_spectra: list[np.ndarray | list[float]],
    *,
    strategy: str = "frontier",
    novelty_weight: float = 0.5,
    uncertainty_weight: float = 0.5,
    phase_uncertainty: np.ndarray | None = None,
    candidate_novelty: list[float] | np.ndarray | None = None,
    exploration_weight: float = 0.5,
) -> dict[str, Any]:
    """Select a deterministic target spectrum from candidate frontier spectra."""
    candidates = [np.asarray(s, dtype=np.float64) for s in frontier_spectra]
    if not candidates:
        return {
            "target_spectrum": None,
            "information_gain_score": 0.0,
            "novelty_score": 0.0,
            "spectral_uncertainty": 0.0,
            "phase_uncertainty": 0.0,
            "combined_score": 0.0,
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
            "phase_uncertainty": 0.0,
            "combined_score": float(best["information_gain_score"]),
            "strategy": "information_gain",
        }

    if strategy == "experiment_planning":
        n = len(candidates)
        phase_scores = np.zeros(n, dtype=np.float64)
        novelty_scores = np.zeros(n, dtype=np.float64)
        if phase_uncertainty is not None:
            arr = np.asarray(phase_uncertainty, dtype=np.float64).reshape(-1)
            if arr.size > 0:
                upto = min(n, int(arr.size))
                phase_scores[:upto] = arr[:upto]
        if candidate_novelty is not None:
            nov = np.asarray(candidate_novelty, dtype=np.float64).reshape(-1)
            if nov.size > 0:
                upto = min(n, int(nov.size))
                novelty_scores[:upto] = nov[:upto]

        combined_score = float(uncertainty_weight) * phase_scores + float(exploration_weight) * novelty_scores
        indices = np.arange(n, dtype=np.int64)
        order = np.lexsort((indices, -combined_score))
        best_idx = int(order[0])
        return {
            "target_spectrum": candidates[best_idx],
            "information_gain_score": 0.0,
            "novelty_score": float(novelty_scores[best_idx]),
            "spectral_uncertainty": 0.0,
            "phase_uncertainty": float(phase_scores[best_idx]),
            "combined_score": float(combined_score[best_idx]),
            "strategy": "experiment_planning",
        }

    first = candidates[0]
    return {
        "target_spectrum": first,
        "information_gain_score": 0.0,
        "novelty_score": 0.0,
        "spectral_uncertainty": 0.0,
        "phase_uncertainty": 0.0,
        "combined_score": 0.0,
        "strategy": str(strategy),
    }
