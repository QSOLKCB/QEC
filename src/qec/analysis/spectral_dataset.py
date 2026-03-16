"""Deterministic spectral dataset extraction for Bayesian landscape modeling."""

from __future__ import annotations

from typing import Any

import numpy as np


def _objective_spectrum(objectives: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            float(objectives.get("spectral_radius", 0.0)),
            float(objectives.get("bethe_margin", 0.0)),
            float(objectives.get("ipr_localization", 0.0)),
            float(objectives.get("entropy", 0.0)),
        ],
        dtype=np.float64,
    )


def build_spectral_dataset(archive: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Build deterministic training dataset from archive entries.

    Each row is derived from a unique candidate id and contains:
    - spectral_vector
    - composite_score
    - threshold_estimate

    The target metric y is an expected-code-quality proxy:
    ``threshold_estimate - composite_score`` (larger is better).
    """
    categories = archive.get("categories", {})

    seen: dict[str, dict[str, Any]] = {}
    for category_entries in categories.values():
        for entry in category_entries:
            candidate_id = str(entry.get("candidate_id", ""))
            if candidate_id:
                seen[candidate_id] = entry

    dim = 4
    if not seen:
        return (
            np.empty((0, dim), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )

    rows: list[np.ndarray] = []
    targets: list[float] = []
    for candidate_id in sorted(seen.keys()):
        entry = seen[candidate_id]
        objectives = entry.get("objectives", {})
        spectrum = _objective_spectrum(objectives)
        composite_score = float(objectives.get("composite_score", 0.0))
        threshold_estimate = float(
            objectives.get("threshold_estimate", objectives.get("threshold", 0.0))
        )
        rows.append(spectrum)
        targets.append(threshold_estimate - composite_score)

    if len(rows) == 0:
        return (
            np.empty((0, dim), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )

    X = np.vstack(rows).astype(np.float64, copy=False)
    y = np.asarray(targets, dtype=np.float64)
    return X, y
