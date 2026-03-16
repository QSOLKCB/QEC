"""Deterministic dataset extraction for spectral theory conjecture fitting."""

from __future__ import annotations

from typing import Any

import numpy as np


_FEATURE_NAMES = [
    "spectral_radius",
    "bethe_min_eigenvalue",
    "bp_stability",
    "cycle_density",
    "motif_frequency",
    "mean_degree",
]


def _extract_feature_vector(objectives: dict[str, Any]) -> np.ndarray:
    """Extract the fixed-order theory feature vector as float64."""
    return np.asarray(
        [float(objectives.get(name, 0.0)) for name in _FEATURE_NAMES],
        dtype=np.float64,
    )


def _extract_target(objectives: dict[str, Any]) -> float:
    """Extract deterministic target preferring threshold when available."""
    if "decoding_threshold" in objectives:
        return float(objectives.get("decoding_threshold", 0.0))
    if "threshold" in objectives:
        return float(objectives.get("threshold", 0.0))
    if "threshold_estimate" in objectives:
        return float(objectives.get("threshold_estimate", 0.0))
    return float(objectives.get("composite_score", 0.0))


def build_theory_dataset(archive: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Build deterministic theory dataset from archive categories."""
    categories = archive.get("categories", {})

    by_candidate: dict[str, dict[str, Any]] = {}
    for category_name in sorted(categories.keys()):
        category_entries = categories.get(category_name, [])
        for entry in category_entries:
            candidate_id = str(entry.get("candidate_id", ""))
            if candidate_id:
                by_candidate[candidate_id] = entry

    if not by_candidate:
        return (
            np.empty((0, len(_FEATURE_NAMES)), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )

    rows: list[np.ndarray] = []
    targets: list[float] = []
    for candidate_id in sorted(by_candidate.keys()):
        objectives = by_candidate[candidate_id].get("objectives", {})
        rows.append(_extract_feature_vector(objectives))
        targets.append(_extract_target(objectives))

    X = np.vstack(rows).astype(np.float64, copy=False)
    y = np.asarray(targets, dtype=np.float64)
    return X, y
