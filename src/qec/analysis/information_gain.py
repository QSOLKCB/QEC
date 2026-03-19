"""Deterministic information-gain scoring for spectral exploration."""

from __future__ import annotations

from typing import Any

import numpy as np

from qec.analysis.landscape_metrics import novelty_score as spectral_novelty_score
from qec.analysis.spectral_uncertainty import estimate_spectral_uncertainty


def information_gain_score(
    memory: object,
    candidate_spectrum: np.ndarray | list[float],
    *,
    novelty_weight: float = 0.5,
    uncertainty_weight: float = 0.5,
) -> dict[str, float]:
    """Compute weighted information gain for one candidate spectrum."""
    spec = np.asarray(candidate_spectrum, dtype=np.float64)
    if spec.ndim != 1:
        raise ValueError("candidate_spectrum must be 1D")

    nw = np.float64(novelty_weight)
    uw = np.float64(uncertainty_weight)

    novelty = np.float64(spectral_novelty_score(spec, memory))
    uncertainty = np.float64(estimate_spectral_uncertainty(memory, spec))
    gain = nw * novelty + uw * uncertainty
    return {
        "information_gain_score": float(np.float64(gain)),
        "novelty_score": float(novelty),
        "spectral_uncertainty": float(uncertainty),
    }


def rank_candidates_by_information_gain(
    candidates: list[np.ndarray | list[float]],
    memory: Any,
    *,
    novelty_weight: float = 0.5,
    uncertainty_weight: float = 0.5,
) -> list[dict[str, Any]]:
    """Rank candidates by descending information gain with stable index tiebreak."""
    scored: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates):
        info = information_gain_score(
            memory,
            candidate,
            novelty_weight=novelty_weight,
            uncertainty_weight=uncertainty_weight,
        )
        scored.append(
            {
                "candidate_index": int(idx),
                "target_spectrum": np.asarray(candidate, dtype=np.float64),
                **info,
            }
        )

    if not scored:
        return []

    idx = np.asarray([item["candidate_index"] for item in scored], dtype=np.int64)
    gain = np.asarray([item["information_gain_score"] for item in scored], dtype=np.float64)
    order = np.lexsort((idx, -gain))
    return [scored[int(i)] for i in order.tolist()]
