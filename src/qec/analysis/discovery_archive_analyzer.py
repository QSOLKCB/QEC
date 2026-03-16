"""Deterministic archive analysis for self-reflective discovery."""

from __future__ import annotations

from typing import Any

import numpy as np


_FEATURES: tuple[str, ...] = (
    "spectral_radius",
    "bethe_margin",
    "bp_stability",
    "motif_frequency",
    "cycle_density",
    "mean_degree",
)


def _iter_unique_archive_entries(archive: dict[str, Any]) -> list[dict[str, Any]]:
    categories = archive.get("categories", {})
    seen: dict[str, dict[str, Any]] = {}
    for cat_name in sorted(categories.keys()):
        for entry in categories.get(cat_name, []):
            cid = str(entry.get("candidate_id", ""))
            if cid not in seen:
                seen[cid] = entry
    return [seen[cid] for cid in sorted(seen.keys())]


def _safe_corr(x: np.ndarray, y: np.ndarray) -> np.float64:
    if x.size < 2 or y.size < 2:
        return np.float64(0.0)
    x0 = np.asarray(x, dtype=np.float64)
    y0 = np.asarray(y, dtype=np.float64)
    x_std = np.float64(np.std(x0, dtype=np.float64))
    y_std = np.float64(np.std(y0, dtype=np.float64))
    if float(x_std) <= 0.0 or float(y_std) <= 0.0:
        return np.float64(0.0)
    cov = np.float64(np.mean((x0 - np.mean(x0)) * (y0 - np.mean(y0)), dtype=np.float64))
    return np.float64(cov / (x_std * y_std))


def analyze_discovery_archive(archive: dict[str, Any]) -> dict[str, Any]:
    """Compute deterministic float64 feature correlations with discovery success."""
    entries = _iter_unique_archive_entries(archive)
    if not entries:
        return {"feature_correlations": {name: 0.0 for name in _FEATURES}}

    success = np.zeros((len(entries),), dtype=np.float64)
    values = np.zeros((len(entries), len(_FEATURES)), dtype=np.float64)

    for i, entry in enumerate(entries):
        obj = entry.get("objectives", {})
        success[i] = -np.float64(obj.get("composite_score", 0.0))
        values[i, 0] = np.float64(obj.get("spectral_radius", 0.0))
        values[i, 1] = np.float64(obj.get("bethe_margin", 0.0))
        values[i, 2] = np.float64(obj.get("bp_stability", values[i, 1]))
        values[i, 3] = np.float64(obj.get("motif_frequency", 0.0))
        values[i, 4] = np.float64(obj.get("cycle_density", 0.0))
        values[i, 5] = np.float64(obj.get("mean_degree", obj.get("check_degree_mean", 0.0)))

    correlations: dict[str, float] = {}
    for idx, feature_name in enumerate(_FEATURES):
        corr = _safe_corr(success, values[:, idx])
        correlations[feature_name] = float(np.float64(corr))

    return {"feature_correlations": correlations}
