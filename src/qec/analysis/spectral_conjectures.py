"""Deterministic spectral conjecture generation and ranking."""

from __future__ import annotations

from typing import Any

import numpy as np


def generate_conjectures(fitted_models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Generate deterministic conjectures from fitted model records."""
    conjectures: list[dict[str, Any]] = []
    for i, model in enumerate(fitted_models):
        ranking_score = np.float64(model.get("ranking_score", 0.0))
        dataset_size = int(model.get("dataset_size", 0))
        confidence = np.float64(1.0 - np.exp(-np.float64(max(dataset_size, 0)) / np.float64(10.0)))
        conjectures.append(
            {
                "conjecture_id": f"conj_{i:04d}",
                "equation_string": str(model.get("equation_string", "")),
                "model_type": str(model.get("model_type", "")),
                "features_used": [str(f) for f in model.get("features_used", [])],
                "dataset_size": dataset_size,
                "fit_metrics": dict(model.get("fit_metrics", {})),
                "ranking_score": float(ranking_score),
                "confidence_score": float(confidence),
            }
        )
    return conjectures


def rank_conjectures(conjectures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministically rank conjectures via lexsort on id and descending score."""
    if not conjectures:
        return []
    ids = np.asarray([str(c.get("conjecture_id", "")) for c in conjectures], dtype=object)
    scores = np.asarray([np.float64(c.get("ranking_score", 0.0)) for c in conjectures], dtype=np.float64)
    order = np.lexsort((ids, -scores))
    return [dict(conjectures[int(i)]) for i in order.tolist()]
