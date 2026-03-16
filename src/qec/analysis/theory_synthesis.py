"""Deterministic synthesis of spectral conjectures from phase profiles."""

from __future__ import annotations

from typing import Any

import numpy as np

_FEATURE_NAMES = (
    "spectral_radius",
    "bethe_hessian_min_eigenvalue",
    "bp_stability_score",
    "trapping_density",
)

_TARGET_NAME = "estimated_threshold"


def _stable_phase_records(phase_profiles: Any) -> list[dict[str, Any]]:
    records = [dict(rec) for rec in (phase_profiles or []) if isinstance(rec, dict)]
    indexed: list[tuple[int, int, dict[str, Any]]] = []
    for idx, rec in enumerate(records):
        phase_id = int(rec.get("phase_id", idx))
        indexed.append((phase_id, idx, rec))
    indexed.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in indexed]


def _feature_value(record: dict[str, Any], feature_name: str) -> np.float64:
    if feature_name == "bethe_hessian_min_eigenvalue":
        raw = record.get("bethe_hessian_min_eigenvalue", record.get("bethe_min_eigenvalue", 0.0))
    else:
        raw = record.get(feature_name, 0.0)
    return np.float64(raw)


def build_phase_dataset(phase_profiles: Any) -> dict[str, Any]:
    """Build deterministic float64 design matrix from phase profile records."""
    ordered = _stable_phase_records(phase_profiles)
    if not ordered:
        return {
            "X": np.zeros((0, len(_FEATURE_NAMES)), dtype=np.float64),
            "y": np.zeros((0,), dtype=np.float64),
            "feature_names": list(_FEATURE_NAMES),
        }

    X = np.zeros((len(ordered), len(_FEATURE_NAMES)), dtype=np.float64)
    y = np.zeros((len(ordered),), dtype=np.float64)
    for row_idx, record in enumerate(ordered):
        for col_idx, feature in enumerate(_FEATURE_NAMES):
            X[row_idx, col_idx] = _feature_value(record, feature)
        y[row_idx] = np.float64(record.get(_TARGET_NAME, 0.0))

    return {
        "X": np.asarray(X, dtype=np.float64),
        "y": np.asarray(y, dtype=np.float64),
        "feature_names": list(_FEATURE_NAMES),
    }


def _fit_model(design: np.ndarray, y: np.ndarray, model_type: str) -> dict[str, Any]:
    coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    y_hat = np.asarray(design @ coeffs, dtype=np.float64)
    residual = np.asarray(y - y_hat, dtype=np.float64)
    mse = np.float64(np.mean(residual * residual)) if y.size > 0 else np.float64(0.0)
    y_var = np.float64(np.var(y)) if y.size > 0 else np.float64(0.0)
    r2_score = np.float64(1.0) if float(y_var) == 0.0 else np.float64(1.0 - mse / y_var)
    return {
        "model_type": str(model_type),
        "coefficients": coeffs,
        "mse": float(mse),
        "r2_score": float(r2_score),
    }


def fit_spectral_models(X: Any, y: Any) -> list[dict[str, Any]]:
    """Fit deterministic candidate analytic forms with float64 least squares."""
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if X_arr.ndim != 2 or X_arr.shape[0] == 0 or X_arr.shape[0] != y_arr.shape[0]:
        return []

    ones = np.ones((X_arr.shape[0], 1), dtype=np.float64)
    linear = np.column_stack((ones, X_arr))
    quadratic = np.column_stack((ones, X_arr, X_arr * X_arr))

    ratio_features = np.zeros((X_arr.shape[0], 1), dtype=np.float64)
    ratio_features[:, 0] = X_arr[:, 0] / (np.float64(1.0) + np.abs(X_arr[:, 3]))
    ratio = np.column_stack((ones, ratio_features))

    log_linear = np.column_stack((ones, np.log(np.maximum(np.float64(1e-12), X_arr[:, [0]]))))

    log_target = np.log(np.maximum(np.float64(1e-12), np.abs(y_arr)))
    power_law = np.column_stack((ones, np.log(np.maximum(np.float64(1e-12), np.abs(X_arr[:, [0]])))))

    models = [
        _fit_model(linear, y_arr, "linear"),
        _fit_model(quadratic, y_arr, "quadratic"),
        _fit_model(ratio, y_arr, "ratio"),
        _fit_model(log_linear, y_arr, "log_linear"),
        _fit_model(power_law, log_target, "power_law"),
    ]

    return models


def _format_term(value: np.float64) -> str:
    return f"{float(np.float64(value)):.12g}"


def _equation_for_model(model: dict[str, Any], feature_names: list[str]) -> str:
    model_type = str(model.get("model_type", ""))
    coeffs = np.asarray(model.get("coefficients", []), dtype=np.float64)
    if coeffs.size == 0:
        return "estimated_threshold ≈ 0"

    b0 = _format_term(coeffs[0])
    if model_type == "linear":
        terms = [f"{_format_term(coeffs[i + 1])} * {feature_names[i]}" for i in range(len(feature_names))]
        return f"estimated_threshold ≈ {b0} + " + " + ".join(terms)
    if model_type == "quadratic":
        n = len(feature_names)
        linear_terms = [f"{_format_term(coeffs[i + 1])} * {feature_names[i]}" for i in range(n)]
        quad_terms = [f"{_format_term(coeffs[n + i + 1])} * {feature_names[i]}^2" for i in range(n)]
        return f"estimated_threshold ≈ {b0} + " + " + ".join(linear_terms + quad_terms)
    if model_type == "ratio":
        return (
            f"estimated_threshold ≈ {b0} + {_format_term(coeffs[1])} * "
            f"({feature_names[0]} / (1 + |{feature_names[3]}|))"
        )
    if model_type == "log_linear":
        return (
            f"estimated_threshold ≈ {b0} + {_format_term(coeffs[1])} * "
            f"log(max(1e-12, {feature_names[0]}))"
        )
    return (
        f"log(estimated_threshold) ≈ {b0} + {_format_term(coeffs[1])} * "
        f"log(max(1e-12, |{feature_names[0]}|))"
    )


def generate_spectral_conjectures(models: list[dict[str, Any]], feature_names: list[str]) -> list[dict[str, Any]]:
    """Generate deterministic symbolic conjectures ranked by fit quality."""
    records: list[dict[str, Any]] = []
    for idx, model in enumerate(models):
        r2 = float(np.float64(model.get("r2_score", 0.0)))
        mse = float(np.float64(model.get("mse", 0.0)))
        fit_quality = float(np.float64(r2 - mse))
        rec = {
            "conjecture_id": f"spectral_conjecture_{idx:03d}",
            "equation": _equation_for_model(model, feature_names),
            "model_type": str(model.get("model_type", "")),
            "fit_quality": fit_quality,
        }
        records.append(rec)

    records.sort(key=lambda rec: (-float(np.float64(rec.get("fit_quality", 0.0))), str(rec.get("conjecture_id", ""))))
    return records
