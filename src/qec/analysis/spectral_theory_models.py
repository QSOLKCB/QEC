"""Deterministic analytic model candidates for spectral theory extraction."""

from __future__ import annotations

from typing import Any

import numpy as np


_ROUND = 12
_EPS = np.float64(1e-12)
_FEATURE_NAMES = [
    "spectral_radius",
    "bethe_min_eigenvalue",
    "bp_stability",
    "cycle_density",
    "motif_frequency",
    "mean_degree",
]


def _safe_log(v: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(np.asarray(v, dtype=np.float64), _EPS))


def _fit_linear_design(design: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coeffs, _, _, _ = np.linalg.lstsq(design.astype(np.float64), y.astype(np.float64), rcond=None)
    pred = np.dot(design, coeffs)
    return coeffs.astype(np.float64), pred.astype(np.float64)


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    yv = np.asarray(y, dtype=np.float64)
    pv = np.asarray(pred, dtype=np.float64)
    if yv.size == 0:
        return {"r2": 0.0, "mse": 0.0, "correlation": 0.0}

    resid = yv - pv
    mse = float(np.mean(resid * resid, dtype=np.float64))
    y_mean = float(np.mean(yv, dtype=np.float64))
    sst = float(np.sum((yv - y_mean) ** 2, dtype=np.float64))
    ssr = float(np.sum(resid ** 2, dtype=np.float64))
    r2 = 1.0 - (ssr / sst) if sst > 0.0 else 0.0

    y_std = float(np.std(yv, dtype=np.float64))
    p_std = float(np.std(pv, dtype=np.float64))
    corr = 0.0 if (y_std <= 0.0 or p_std <= 0.0) else float(np.corrcoef(yv, pv)[0, 1])

    return {
        "r2": float(np.float64(r2)),
        "mse": float(np.float64(mse)),
        "correlation": float(np.float64(corr)),
    }


def _feature_index_subsets(dim: int, max_size: int = 3) -> list[tuple[int, ...]]:
    subsets: list[tuple[int, ...]] = []
    for size in (1, 2, 3):
        if size > max_size:
            break
        if size == 1:
            for i in range(dim):
                subsets.append((i,))
        elif size == 2:
            for i in range(dim):
                for j in range(i + 1, dim):
                    subsets.append((i, j))
        else:
            for i in range(dim):
                for j in range(i + 1, dim):
                    for k in range(j + 1, dim):
                        subsets.append((i, j, k))
    return subsets


def _fit_candidate(
    *,
    model_name: str,
    model_type: str,
    features_used: list[str],
    expression: str,
    design: np.ndarray,
    y: np.ndarray,
    fit_in_log_space: bool = False,
) -> dict[str, Any]:
    y64 = np.asarray(y, dtype=np.float64)
    if fit_in_log_space:
        y_transformed = _safe_log(y64 + _EPS)
        params, pred_log = _fit_linear_design(design, y_transformed)
        predictions = np.exp(pred_log)
    else:
        params, predictions = _fit_linear_design(design, y64)

    model_metrics = _metrics(y64, predictions)
    return {
        "model_name": str(model_name),
        "model_type": str(model_type),
        "features_used": list(features_used),
        "dataset_size": int(y64.shape[0]),
        "analytic_expression": str(expression),
        "fit_parameters": [round(float(p), _ROUND) for p in params.tolist()],
        "predictions": np.asarray(predictions, dtype=np.float64),
        "r2": float(np.float64(model_metrics["r2"])),
        "mse": float(np.float64(model_metrics["mse"])),
        "correlation": float(np.float64(model_metrics["correlation"])),
        "fit_metrics": {
            "r2": float(np.float64(model_metrics["r2"])),
            "mse": float(np.float64(model_metrics["mse"])),
            "correlation": float(np.float64(model_metrics["correlation"])),
        },
    }


def fit_theory_models(X: np.ndarray, y: np.ndarray) -> list[dict[str, Any]]:
    """Fit deterministic candidate analytic models and return scored results."""
    X64 = np.asarray(X, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)

    if X64.ndim != 2 or X64.shape[1] < len(_FEATURE_NAMES) or y64.ndim != 1 or X64.shape[0] != y64.shape[0]:
        return []

    fitted: list[dict[str, Any]] = []
    subsets = _feature_index_subsets(len(_FEATURE_NAMES), max_size=3)
    for subset in subsets:
        features_used = [_FEATURE_NAMES[i] for i in subset]
        subset_name = "_".join(str(i) for i in subset)
        x0 = X64[:, subset[0]]
        ones = np.ones((X64.shape[0],), dtype=np.float64)

        fitted.append(
            _fit_candidate(
                model_name=f"linear_{subset_name}",
                model_type="linear",
                features_used=features_used,
                expression="y ≈ a0 + " + " + ".join([f"a{i+1}*{name}" for i, name in enumerate(features_used)]),
                design=np.column_stack([ones] + [X64[:, i] for i in subset]).astype(np.float64),
                y=y64,
            )
        )

        fitted.append(
            _fit_candidate(
                model_name=f"poly2_{subset_name}",
                model_type="polynomial2",
                features_used=features_used,
                expression="y ≈ a0 + a1*x + a2*x^2",
                design=np.column_stack((ones, x0, x0 * x0)).astype(np.float64),
                y=y64,
            )
        )

        fitted.append(
            _fit_candidate(
                model_name=f"poly3_{subset_name}",
                model_type="polynomial3",
                features_used=features_used,
                expression="y ≈ a0 + a1*x + a2*x^2 + a3*x^3",
                design=np.column_stack((ones, x0, x0 * x0, x0 * x0 * x0)).astype(np.float64),
                y=y64,
            )
        )

        fitted.append(
            _fit_candidate(
                model_name=f"log_linear_{subset_name}",
                model_type="log_linear",
                features_used=features_used,
                expression="y ≈ a0 + " + " + ".join([f"a{i+1}*log(1+{name})" for i, name in enumerate(features_used)]),
                design=np.column_stack([ones] + [_safe_log(1.0 + X64[:, i]) for i in subset]).astype(np.float64),
                y=y64,
            )
        )

        ratio_denom = np.abs(X64[:, subset[-1]]) + _EPS
        ratio_feature = x0 / ratio_denom
        fitted.append(
            _fit_candidate(
                model_name=f"ratio_{subset_name}",
                model_type="ratio",
                features_used=features_used,
                expression=f"y ≈ a0 + a1*{features_used[0]}/(|{features_used[-1]}|+eps)",
                design=np.column_stack((ones, ratio_feature)).astype(np.float64),
                y=y64,
            )
        )

        fitted.append(
            _fit_candidate(
                model_name=f"power_law_{subset_name}",
                model_type="power_law",
                features_used=features_used,
                expression="y ≈ exp(a0) * " + " * ".join([f"({name}+eps)^a{i+1}" for i, name in enumerate(features_used)]),
                design=np.column_stack([ones] + [_safe_log(X64[:, i] + _EPS) for i in subset]).astype(np.float64),
                y=y64,
                fit_in_log_space=True,
            )
        )

    return fitted
