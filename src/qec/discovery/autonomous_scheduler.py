"""Deterministic autonomous scheduler strategies for discovery targets."""

from __future__ import annotations

from typing import Any

import numpy as np

from qec.analysis.information_gain import rank_candidates_by_information_gain
from qec.analysis.landscape_gaps import detect_landscape_gaps
from qec.discovery.experiment_targets import choose_experiment_target


def schedule_next_experiment(
    memory: object | None,
    *,
    gap_radius: float = 0.3,
    max_gaps: int = 16,
) -> dict[str, Any]:
    """Schedule next deterministic target from landscape-gap exploration.

    Compatible with older callers that pass ``None`` or plain dictionaries.
    """
    if memory is None:
        gaps: list[np.ndarray] = []
    else:
        try:
            gaps = detect_landscape_gaps(memory, gap_radius=float(gap_radius), max_gaps=int(max_gaps))
        except (AttributeError, KeyError, TypeError, ValueError):
            gaps = []
    target = choose_experiment_target(gaps)
    return {
        "target_spectrum": None if target is None else np.asarray(target, dtype=np.float64),
        "strategy": "landscape_exploration",
        "gap_count": int(len(gaps)),
    }

def compute_combined_score(
    exploration_score: float,
    hypothesis_bias: float = 0.0,
    *,
    strategy: str = "default",
    hypothesis_weight: float = 0.5,
) -> float:
    """Compute deterministic combined candidate score."""
    explore = np.float64(exploration_score)
    bias = np.float64(hypothesis_bias)
    weight = np.float64(min(max(float(np.float64(hypothesis_weight)), 0.0), 1.0))
    if strategy == "hypothesis_guided":
        return float(np.float64((np.float64(1.0) - weight) * explore + weight * bias))
    return float(np.float64(explore))


def select_best_candidate(
    candidates: list[dict[str, Any]],
    *,
    strategy: str = "default",
    hypothesis_weight: float = 0.5,
) -> int:
    """Select the best candidate index with deterministic tie-breaking."""
    if not candidates:
        return -1

    scores = np.asarray(
        [
            compute_combined_score(
                item.get("exploration_score", 0.0),
                item.get("hypothesis_bias", 0.0),
                strategy=strategy,
                hypothesis_weight=hypothesis_weight,
            )
            for item in candidates
        ],
        dtype=np.float64,
    )
    indices = np.arange(scores.shape[0], dtype=np.int64)
    order = np.lexsort((indices, -scores))
    return int(order[0])


def schedule_autonomous_target(
    memory: object | None,
    candidate_spectra: list[np.ndarray] | None = None,
    *,
    frontier_spectra: list[np.ndarray] | None = None,
    strategy: str = "landscape_exploration",
    gap_radius: float = 0.3,
    max_gaps: int = 16,
    novelty_weight: float = 0.5,
    uncertainty_weight: float = 0.5,
    exploration_weight: float = 0.5,
    phase_uncertainty: np.ndarray | None = None,
    candidate_novelty: np.ndarray | None = None,
) -> dict[str, Any]:
    """Backward-compatible scheduler entry point."""
    spectra = frontier_spectra if frontier_spectra is not None else candidate_spectra

    if strategy == "information_gain" and spectra:
        ranked = rank_candidates_by_information_gain(
            spectra,
            memory,
            novelty_weight=float(novelty_weight),
            uncertainty_weight=float(uncertainty_weight),
        )
        best = ranked[0] if ranked else None
        return {
            "target_spectrum": None if best is None else np.asarray(best.get("target_spectrum"), dtype=np.float64),
            "strategy": "information_gain",
            "information_gain_score": 0.0 if best is None else float(np.float64(best.get("information_gain_score", 0.0))),
            "selected_novelty": 0.0 if best is None else float(np.float64(best.get("novelty_score", 0.0))),
            "selected_uncertainty": 0.0 if best is None else float(np.float64(best.get("spectral_uncertainty", 0.0))),
            "gap_count": 0,
        }

    if strategy == "experiment_planning" and spectra:
        unc = np.asarray(
            phase_uncertainty if phase_uncertainty is not None else np.zeros((len(spectra),), dtype=np.float64),
            dtype=np.float64,
        )
        nov = np.asarray(
            candidate_novelty if candidate_novelty is not None else np.zeros((len(spectra),), dtype=np.float64),
            dtype=np.float64,
        )
        n = min(len(spectra), int(unc.size), int(nov.size))
        if n == 0:
            return {"target_spectrum": None, "strategy": "experiment_planning", "combined_score": 0.0, "gap_count": 0}
        idx = np.arange(n, dtype=np.int64)
        combined = np.float64(uncertainty_weight) * unc[:n] + np.float64(exploration_weight) * nov[:n]
        order = np.lexsort((idx, -combined))
        best_i = int(order[0])
        return {
            "target_spectrum": np.asarray(spectra[best_i], dtype=np.float64),
            "strategy": "experiment_planning",
            "combined_score": float(np.float64(combined[best_i])),
            "gap_count": 0,
        }

    return schedule_next_experiment(memory, gap_radius=gap_radius, max_gaps=max_gaps)
