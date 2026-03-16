"""Deterministic spectral conjecture generation and ranking."""

from __future__ import annotations

import numpy as np


_ROUND = 12


def _clamp01(x: float) -> float:
    return float(min(max(float(x), 0.0), 1.0))


def _equation_string(model: dict[str, object]) -> str:
    model_type = str(model.get("model_type", ""))
    features_used = [str(f) for f in list(model.get("features_used", []))]

    if model_type == "polynomial2":
        name = features_used[0] if features_used else "x"
        return f"threshold ≈ a0 + a1*{name} + a2*{name}^2"
    if model_type == "polynomial3":
        name = features_used[0] if features_used else "x"
        return f"threshold ≈ a0 + a1*{name} + a2*{name}^2 + a3*{name}^3"
    if model_type == "ratio":
        num = features_used[0] if features_used else "x"
        den = features_used[-1] if features_used else "x"
        return f"threshold ≈ a0 + a1*{num}/(|{den}|+eps)"
    if model_type == "log_linear":
        if not features_used:
            return "threshold ≈ a0"
        return "threshold ≈ a0 + " + " + ".join(
            [f"a{i+1}*log(1+{name})" for i, name in enumerate(features_used)]
        )
    if model_type == "power_law":
        if not features_used:
            return "threshold ≈ exp(a0)"
        factors = " * ".join([f"({name}+eps)^a{i+1}" for i, name in enumerate(features_used)])
        return f"threshold ≈ exp(a0) * {factors}"
    if not features_used:
        return "threshold ≈ a0"
    return "threshold ≈ a0 + " + " + ".join(
        [f"a{i+1}*{name}" for i, name in enumerate(features_used)]
    )


def generate_conjectures(fitted_models: list[dict[str, object]]) -> list[dict[str, object]]:
    """Generate conjectures from fitted analytic model summaries."""
    conjectures: list[dict[str, object]] = []
    ordered_models = sorted(fitted_models, key=lambda m: str(m.get("model_name", "")))
    for idx, model in enumerate(ordered_models, start=1):
        fit_metrics = dict(model.get("fit_metrics", {}))
        r2 = float(np.float64(fit_metrics.get("r2", model.get("r2", 0.0))))
        mse = float(np.float64(fit_metrics.get("mse", model.get("mse", 0.0))))
        corr = float(np.float64(fit_metrics.get("correlation", model.get("correlation", 0.0))))

        fit_score = _clamp01(max(0.0, r2))
        mse_factor = 1.0 / (1.0 + max(0.0, mse))
        confidence = _clamp01(0.5 * abs(corr) + 0.5 * mse_factor)

        conjectures.append(
            {
                "conjecture_id": int(idx),
                "model_name": str(model.get("model_name", "")),
                "model_type": str(model.get("model_type", "")),
                "features_used": [str(v) for v in list(model.get("features_used", []))],
                "dataset_size": int(model.get("dataset_size", 0)),
                "analytic_expression": str(model.get("analytic_expression", "")),
                "equation_string": _equation_string(model),
                "fit_parameters": [
                    round(float(np.float64(p)), _ROUND)
                    for p in list(model.get("fit_parameters", []))
                ],
                "fit_metrics": {
                    "r2": float(np.float64(r2)),
                    "mse": float(np.float64(mse)),
                    "correlation": float(np.float64(corr)),
                },
                "fit_score": float(np.float64(fit_score)),
                "confidence_score": float(np.float64(confidence)),
            }
        )

    return conjectures


def rank_conjectures(conjectures: list[dict[str, object]]) -> list[dict[str, object]]:
    """Rank conjectures by score with deterministic tie-breaking."""
    if not conjectures:
        return []

    ranking_scores = np.asarray(
        [
            float(np.float64(c.get("fit_score", 0.0)))
            * float(np.float64(c.get("confidence_score", 0.0)))
            for c in conjectures
        ],
        dtype=np.float64,
    )
    ids = np.asarray([int(c.get("conjecture_id", 0)) for c in conjectures], dtype=np.int64)
    order = np.lexsort((ids, -ranking_scores))

    ranked: list[dict[str, object]] = []
    for idx in order.tolist():
        c = dict(conjectures[int(idx)])
        c["ranking_score"] = float(np.float64(ranking_scores[int(idx)]))
        ranked.append(c)
    return ranked
