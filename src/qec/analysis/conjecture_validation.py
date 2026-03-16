"""Deterministic conjecture validation experiments for theory falsification.

Evaluates spectral conjectures against observed phase profiles,
detects counterexamples, and designs targeted validation experiments.

Layer 3 — Analysis.
Does not modify decoder internals.  Fully deterministic.
"""

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


def _feature_value(record: dict[str, Any], feature_name: str) -> np.float64:
    """Extract a single feature from a phase profile record as float64."""
    if feature_name == "bethe_hessian_min_eigenvalue":
        raw = record.get(
            "bethe_hessian_min_eigenvalue",
            record.get("bethe_min_eigenvalue", 0.0),
        )
    else:
        raw = record.get(feature_name, 0.0)
    return np.float64(raw)


def _predict_from_conjecture(
    conjecture: dict[str, Any],
    feature_vector: np.ndarray,
) -> np.float64:
    """Predict target value from a conjecture and a single feature vector.

    Supports conjectures with coefficients (linear/quadratic models)
    and conjectures with a constant target_value fallback.
    """
    model_type = str(conjecture.get("model_type", ""))
    coeffs = conjecture.get("coefficients")

    if coeffs is not None:
        c = np.asarray(coeffs, dtype=np.float64).reshape(-1)
        x = np.asarray(feature_vector, dtype=np.float64).reshape(-1)

        if model_type == "linear" and c.shape[0] == x.shape[0] + 1:
            return np.float64(c[0] + np.dot(c[1:], x))

        if model_type == "quadratic" and c.shape[0] == 2 * x.shape[0] + 1:
            n = x.shape[0]
            linear_part = np.dot(c[1 : n + 1], x)
            quad_part = np.dot(c[n + 1 : 2 * n + 1], x * x)
            return np.float64(c[0] + linear_part + quad_part)

        if model_type == "ratio" and c.shape[0] == 2 and x.shape[0] >= 4:
            ratio_val = x[0] / (np.float64(1.0) + np.abs(x[3]))
            return np.float64(c[0] + c[1] * ratio_val)

        if model_type == "log_linear" and c.shape[0] == 2 and x.shape[0] >= 1:
            log_val = np.log(np.maximum(np.float64(1e-12), x[0]))
            return np.float64(c[0] + c[1] * log_val)

        if model_type == "power_law" and c.shape[0] == 2 and x.shape[0] >= 1:
            log_pred = c[0] + c[1] * np.log(
                np.maximum(np.float64(1e-12), np.abs(x[0]))
            )
            return np.float64(np.exp(log_pred))

        if c.shape[0] == x.shape[0] + 1:
            return np.float64(c[0] + np.dot(c[1:], x))
        if c.shape[0] == x.shape[0]:
            return np.float64(np.dot(c, x))

    if "target_value" in conjecture:
        return np.float64(conjecture["target_value"])

    return np.float64(0.0)


def _extract_feature_vector(profile: dict[str, Any]) -> np.ndarray:
    """Extract the standard feature vector from a phase profile."""
    return np.array(
        [_feature_value(profile, name) for name in _FEATURE_NAMES],
        dtype=np.float64,
    )


def evaluate_conjecture(
    conjecture: dict[str, Any],
    phase_profile: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate how well a spectral conjecture predicts an observed metric.

    Parameters
    ----------
    conjecture : dict[str, Any]
        A spectral conjecture with model_type, coefficients, etc.
    phase_profile : dict[str, Any]
        An observed phase profile with feature values and target.

    Returns
    -------
    dict[str, Any]
        Keys: predicted_value (float), observed_value (float),
        absolute_error (float), squared_error (float).
    """
    feature_vector = _extract_feature_vector(phase_profile)
    predicted = _predict_from_conjecture(conjecture, feature_vector)
    observed = np.float64(phase_profile.get(_TARGET_NAME, 0.0))
    abs_error = np.float64(np.abs(predicted - observed))
    sq_error = np.float64(abs_error * abs_error)

    return {
        "predicted_value": float(predicted),
        "observed_value": float(observed),
        "absolute_error": float(abs_error),
        "squared_error": float(sq_error),
    }


def find_conjecture_counterexamples(
    conjecture: dict[str, Any],
    phase_profiles: list[dict[str, Any]],
    error_threshold: float = 0.1,
    max_counterexamples: int = 128,
) -> list[dict[str, Any]]:
    """Search phase profiles for cases that violate the conjecture.

    Parameters
    ----------
    conjecture : dict[str, Any]
        A spectral conjecture to test.
    phase_profiles : list[dict[str, Any]]
        Known phase profiles to evaluate against.
    error_threshold : float
        Minimum absolute error to count as a counterexample.
    max_counterexamples : int
        Maximum number of counterexamples to return.

    Returns
    -------
    list[dict[str, Any]]
        Records sorted by descending absolute error. Each has keys:
        phase_id (int), predicted (float), observed (float), error (float).
    """
    ordered = _stable_sort_profiles(phase_profiles)
    thresh = np.float64(error_threshold)
    records: list[tuple[np.float64, int, dict[str, Any]]] = []

    for profile in ordered:
        phase_id = int(profile.get("phase_id", 0))
        evaluation = evaluate_conjecture(conjecture, profile)
        abs_error = np.float64(evaluation["absolute_error"])
        if abs_error > thresh:
            records.append(
                (
                    abs_error,
                    phase_id,
                    {
                        "phase_id": phase_id,
                        "predicted": evaluation["predicted_value"],
                        "observed": evaluation["observed_value"],
                        "error": float(abs_error),
                    },
                )
            )

    records.sort(key=lambda item: (-float(item[0]), item[1]))
    limit = int(max(0, max_counterexamples))
    return [item[2] for item in records[:limit]]


def design_validation_experiment(
    conjecture: dict[str, Any],
    archive: list[dict[str, Any]],
    num_targets: int = 8,
) -> list[dict[str, Any]]:
    """Generate targeted experiment targets to test the conjecture.

    Strategy:
    1. Evaluate conjecture against all archive profiles.
    2. Identify spectral regions where prediction error is highest.
    3. Generate candidate spectral vectors near those regions.

    Parameters
    ----------
    conjecture : dict[str, Any]
        A spectral conjecture to test.
    archive : list[dict[str, Any]]
        Phase profiles from the discovery archive.
    num_targets : int
        Maximum number of experiment targets to generate.

    Returns
    -------
    list[dict[str, Any]]
        Deterministic experiment targets. Each has keys:
        target_vector (list[float]), source_phase_id (int),
        prediction_error (float), priority (float).
    """
    ordered = _stable_sort_profiles(archive)
    if not ordered:
        return []

    evaluations: list[tuple[np.float64, int, np.ndarray]] = []
    for profile in ordered:
        phase_id = int(profile.get("phase_id", 0))
        result = evaluate_conjecture(conjecture, profile)
        abs_error = np.float64(result["absolute_error"])
        feature_vector = _extract_feature_vector(profile)
        evaluations.append((abs_error, phase_id, feature_vector))

    evaluations.sort(key=lambda item: (-float(item[0]), item[1]))

    if not evaluations:
        return []

    errors = np.array([float(e[0]) for e in evaluations], dtype=np.float64)
    max_error = np.float64(np.max(errors)) if errors.size > 0 else np.float64(1.0)
    if float(max_error) == 0.0:
        max_error = np.float64(1.0)

    targets: list[dict[str, Any]] = []
    limit = int(min(num_targets, len(evaluations)))

    for rank in range(limit):
        abs_error, phase_id, feature_vector = evaluations[rank]
        priority = np.float64(abs_error / max_error)

        perturbation = _deterministic_perturbation(
            feature_vector, rank, phase_id
        )
        target_vector = np.asarray(
            feature_vector + perturbation, dtype=np.float64
        )

        targets.append(
            {
                "target_vector": [float(np.float64(v)) for v in target_vector],
                "source_phase_id": phase_id,
                "prediction_error": float(abs_error),
                "priority": float(priority),
            }
        )

    return targets


def _deterministic_perturbation(
    base_vector: np.ndarray,
    rank: int,
    phase_id: int,
) -> np.ndarray:
    """Generate a small deterministic perturbation vector.

    Uses a deterministic formula based on rank and phase_id
    to avoid any randomness.
    """
    n = base_vector.shape[0]
    perturbation = np.zeros(n, dtype=np.float64)
    scale = np.float64(0.05)

    for i in range(n):
        sign = np.float64(1.0) if ((rank + phase_id + i) % 2 == 0) else np.float64(-1.0)
        magnitude = scale * np.float64(1.0 + rank) / np.float64(1.0 + rank + i)
        base_val = np.abs(base_vector[i])
        perturbation[i] = sign * magnitude * np.maximum(base_val, np.float64(0.01))

    return perturbation


def _stable_sort_profiles(
    phase_profiles: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Return phase profiles in deterministic order by (phase_id, index)."""
    records = [dict(rec) for rec in (phase_profiles or []) if isinstance(rec, dict)]
    indexed: list[tuple[int, int, dict[str, Any]]] = []
    for idx, rec in enumerate(records):
        phase_id = int(rec.get("phase_id", idx))
        indexed.append((phase_id, idx, rec))
    indexed.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in indexed]
