"""Deterministic spectral-theory dataset extraction from discovery archives."""

from __future__ import annotations

from typing import Any

import numpy as np

_FEATURE_ORDER: tuple[str, ...] = (
    "spectral_radius",
    "bethe_min_eigenvalue",
    "bp_stability",
    "cycle_density",
    "motif_frequency",
    "mean_degree",
)
_TARGET_ORDER: tuple[str, ...] = (
    "decoding_threshold",
    "threshold",
    "threshold_estimate",
    "composite_score",
)


def _entry_value(entry: dict[str, Any], name: str) -> np.float64:
    obj = entry.get("objectives", {})
    metrics = entry.get("metrics", {})
    if name == "bethe_min_eigenvalue":
        val = obj.get(name, metrics.get(name, obj.get("bethe_margin", metrics.get("bethe_margin", 0.0))))
    elif name == "mean_degree":
        val = obj.get(name, metrics.get(name, obj.get("check_degree_mean", metrics.get("check_degree_mean", 0.0))))
    else:
        val = obj.get(name, metrics.get(name, 0.0))
    return np.float64(val)


def _entry_has_target(entry: dict[str, Any], name: str) -> bool:
    obj = entry.get("objectives", {})
    metrics = entry.get("metrics", {})
    return name in obj or name in metrics


def build_theory_dataset(archive: dict[str, Any]) -> dict[str, Any]:
    """Build a deterministic float64 theory-fitting dataset from archive entries."""
    categories = archive.get("categories", {})
    rows: list[dict[str, Any]] = []
    x_rows: list[list[float]] = []
    y_rows: list[float] = []
    target_name = "composite_score"

    sorted_categories = sorted(categories.keys())
    for category in sorted_categories:
        entries = sorted(
            categories.get(category, []),
            key=lambda e: str(e.get("candidate_id", "")),
        )
        for entry in entries:
            candidate_id = str(entry.get("candidate_id", ""))
            chosen_target = target_name
            target_value = np.float64(0.0)
            for candidate_target in _TARGET_ORDER:
                val = _entry_value(entry, candidate_target)
                if candidate_target == "composite_score" or _entry_has_target(entry, candidate_target):
                    chosen_target = candidate_target
                    target_value = val
                    break

            features = [_entry_value(entry, name) for name in _FEATURE_ORDER]
            x_rows.append([float(np.float64(v)) for v in features])
            y_rows.append(float(target_value))
            rows.append(
                {
                    "archive_category": category,
                    "candidate_id": candidate_id,
                    "target_name": chosen_target,
                    "target_value": float(target_value),
                    "features": {name: float(np.float64(value)) for name, value in zip(_FEATURE_ORDER, features)},
                }
            )
            target_name = chosen_target

    X = np.asarray(x_rows, dtype=np.float64) if x_rows else np.zeros((0, len(_FEATURE_ORDER)), dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.float64) if y_rows else np.zeros((0,), dtype=np.float64)

    return {
        "feature_names": list(_FEATURE_ORDER),
        "target_name": target_name,
        "X": X,
        "y": y,
        "rows": rows,
        "dataset_size": int(X.shape[0]),
    }
