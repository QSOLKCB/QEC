"""Deterministic fitting of interpretable spectral-theory candidate models."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np


def _fit_linear(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return np.asarray(coeffs, dtype=np.float64)


def _metrics(y: np.ndarray, y_hat: np.ndarray) -> dict[str, float]:
    resid = np.asarray(y - y_hat, dtype=np.float64)
    mse = np.float64(np.mean(resid * resid)) if y.size else np.float64(0.0)
    rmse = np.float64(np.sqrt(mse))
    mae = np.float64(np.mean(np.abs(resid))) if y.size else np.float64(0.0)
    y_var = np.float64(np.var(y))
    r2 = np.float64(1.0) if float(y_var) == 0.0 else np.float64(1.0 - mse / y_var)
    return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}


def _equation_string(model_type: str, features: list[str], coeffs: np.ndarray) -> str:
    c = np.asarray(coeffs, dtype=np.float64)
    if model_type == "linear":
        terms = [f"({float(c[i + 1]):.12g})*{name}" for i, name in enumerate(features)]
    elif model_type == "polynomial_degree_2":
        terms = [f"({float(c[i + 1]):.12g})*{name}" for i, name in enumerate(features)]
        terms += [f"({float(c[len(features) + i + 1]):.12g})*{name}^2" for i, name in enumerate(features)]
    elif model_type == "polynomial_degree_3":
        n = len(features)
        terms = [f"({float(c[i + 1]):.12g})*{name}" for i, name in enumerate(features)]
        terms += [f"({float(c[n + i + 1]):.12g})*{name}^2" for i, name in enumerate(features)]
        terms += [f"({float(c[2 * n + i + 1]):.12g})*{name}^3" for i, name in enumerate(features)]
    elif model_type == "log_linear":
        terms = [f"({float(c[i + 1]):.12g})*log1p(|{name}|)" for i, name in enumerate(features)]
    elif model_type == "ratio":
        if len(features) >= 2:
            terms = [f"({float(c[1]):.12g})*({features[0]}/(1+|{features[1]}|))"]
        else:
            terms = [f"({float(c[1]):.12g})*{features[0]}"]
    else:  # power_law
        terms = [f"({float(c[i + 1]):.12g})*|{name}|^0.5" for i, name in enumerate(features)]
    return f"y = {float(c[0]):.12g} + " + " + ".join(terms)


def _fit_candidate(X: np.ndarray, y: np.ndarray, features: list[str], subset: tuple[int, ...], model_type: str) -> dict[str, Any]:
    Xs = np.asarray(X[:, subset], dtype=np.float64)
    subset_features = [features[i] for i in subset]
    if model_type == "linear":
        design = np.column_stack([np.ones((Xs.shape[0],), dtype=np.float64), Xs])
    elif model_type == "polynomial_degree_2":
        design = np.column_stack([np.ones((Xs.shape[0],), dtype=np.float64), Xs, Xs * Xs])
    elif model_type == "polynomial_degree_3":
        design = np.column_stack([np.ones((Xs.shape[0],), dtype=np.float64), Xs, Xs * Xs, Xs * Xs * Xs])
    elif model_type == "log_linear":
        design = np.column_stack([np.ones((Xs.shape[0],), dtype=np.float64), np.log1p(np.abs(Xs))])
    elif model_type == "ratio":
        if Xs.shape[1] >= 2:
            ratio_term = Xs[:, 0] / (np.float64(1.0) + np.abs(Xs[:, 1]))
            ratio_term = ratio_term.reshape(-1, 1)
            design = np.column_stack([np.ones((Xs.shape[0],), dtype=np.float64), ratio_term])
        else:
            design = np.column_stack([np.ones((Xs.shape[0],), dtype=np.float64), Xs])
    else:  # power-law
        design = np.column_stack([np.ones((Xs.shape[0],), dtype=np.float64), np.sqrt(np.abs(Xs) + np.float64(1e-12))])

    coeffs = _fit_linear(design, y)
    y_hat = np.asarray(design @ coeffs, dtype=np.float64)
    fit_metrics = _metrics(y, y_hat)
    ranking_score = np.float64(fit_metrics["r2"] - fit_metrics["rmse"])
    return {
        "model_type": model_type,
        "features_used": subset_features,
        "equation_string": _equation_string(model_type, subset_features, coeffs),
        "dataset_size": int(Xs.shape[0]),
        "fit_metrics": fit_metrics,
        "ranking_score": float(ranking_score),
    }


def fit_theory_models(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    """Fit deterministic model families over feature subsets of size 1..3."""
    X = np.asarray(dataset.get("X", np.zeros((0, 0), dtype=np.float64)), dtype=np.float64)
    y = np.asarray(dataset.get("y", np.zeros((0,), dtype=np.float64)), dtype=np.float64)
    features = [str(f) for f in dataset.get("feature_names", [])]
    if X.size == 0 or y.size == 0 or X.shape[0] != y.shape[0]:
        return []

    families = (
        "linear",
        "polynomial_degree_2",
        "polynomial_degree_3",
        "log_linear",
        "ratio",
        "power_law",
    )
    fitted: list[dict[str, Any]] = []
    model_counter = 0
    max_subset = min(3, X.shape[1])
    for subset_size in (1, 2, 3):
        if subset_size > max_subset:
            continue
        for subset in combinations(range(X.shape[1]), subset_size):
            for model_type in families:
                record = _fit_candidate(X, y, features, subset, model_type)
                record["model_id"] = f"model_{model_counter:04d}"
                model_counter += 1
                fitted.append(record)
    return fitted
