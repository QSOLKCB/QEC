"""Deterministic validation metrics for spectral conjectures."""

from __future__ import annotations

from typing import Any

import numpy as np


def _predict(conjecture: dict[str, Any], dataset: dict[str, Any]) -> np.ndarray:
    X = np.asarray(dataset.get("X", np.zeros((0, 0), dtype=np.float64)), dtype=np.float64)
    y = np.asarray(dataset.get("y", np.zeros((0,), dtype=np.float64)), dtype=np.float64)
    n = int(y.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.float64)

    if "predictions" in conjecture:
        pred = np.asarray(conjecture.get("predictions", []), dtype=np.float64)
        if pred.shape[0] == n:
            return pred

    if "coefficients" in conjecture:
        coeffs = np.asarray(conjecture.get("coefficients", []), dtype=np.float64)
        intercept = np.float64(conjecture.get("intercept", 0.0))
        if coeffs.ndim == 1 and coeffs.shape[0] == X.shape[1]:
            return np.asarray(intercept + X @ coeffs, dtype=np.float64)

    if "target_value" in conjecture:
        return np.full((n,), np.float64(conjecture.get("target_value", 0.0)), dtype=np.float64)

    return np.zeros((n,), dtype=np.float64)


def validate_conjectures(conjectures: list[dict[str, Any]], dataset: dict[str, Any], tolerance: float = 0.15) -> list[dict[str, Any]]:
    """Compute deterministic float64 validation records sorted by conjecture id."""
    y = np.asarray(dataset.get("y", np.zeros((0,), dtype=np.float64)), dtype=np.float64)
    dataset_size = int(y.shape[0])
    out: list[dict[str, Any]] = []
    for conjecture in conjectures:
        cid = str(conjecture.get("conjecture_id", ""))
        pred = _predict(conjecture, dataset)
        if pred.shape[0] != dataset_size:
            pred = np.zeros((dataset_size,), dtype=np.float64)
        resid = np.asarray(pred - y, dtype=np.float64)
        mse = np.float64(np.mean(resid * resid)) if dataset_size else np.float64(0.0)
        mae = np.float64(np.mean(np.abs(resid))) if dataset_size else np.float64(0.0)
        max_abs_error = np.float64(np.max(np.abs(resid))) if dataset_size else np.float64(0.0)
        support_score = np.float64(1.0) / (np.float64(1.0) + mse + mae)
        out.append(
            {
                "conjecture_id": cid,
                "dataset_size": dataset_size,
                "mse": float(mse),
                "mae": float(mae),
                "max_abs_error": float(max_abs_error),
                "support_score": float(support_score),
                "passes_tolerance": bool(max_abs_error <= np.float64(tolerance)),
            }
        )
    return sorted(out, key=lambda r: r["conjecture_id"])
