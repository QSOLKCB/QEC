"""Deterministic linear model mapping spectral metrics to thresholds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.canonicalize import canonicalize


_ROUND = 12
_DEFAULT_FEATURE_ORDER = [
    "nb_spectral_radius",
    "bethe_negative_mass",
    "flow_ipr",
    "spectral_entropy",
    "trap_similarity",
]


class SpectralThresholdModel:
    def __init__(self, feature_order: list[str] | None = None) -> None:
        self.feature_order = list(feature_order or _DEFAULT_FEATURE_ORDER)
        self.coefficients = np.zeros(len(self.feature_order) + 1, dtype=np.float64)
        self.fitted = False

    def fit(self, dataset: list[dict[str, Any]]) -> dict[str, Any]:
        if not dataset:
            self.coefficients = np.zeros(len(self.feature_order) + 1, dtype=np.float64)
            self.fitted = False
            return {"samples": 0, "fitted": False}

        rows: list[np.ndarray] = []
        targets: list[float] = []
        for item in dataset:
            x = np.array([float(item.get(name, 0.0)) for name in self.feature_order], dtype=np.float64)
            rows.append(np.concatenate([x, np.array([1.0], dtype=np.float64)]))
            targets.append(float(item["threshold"]))

        X = np.vstack(rows).astype(np.float64)
        y = np.asarray(targets, dtype=np.float64)
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        self.coefficients = np.round(coeffs.astype(np.float64), _ROUND)
        self.fitted = True
        return {"samples": int(len(dataset)), "fitted": True}

    def predict(self, features: dict[str, Any]) -> float:
        x = np.array([float(features.get(name, 0.0)) for name in self.feature_order] + [1.0], dtype=np.float64)
        value = float(np.dot(self.coefficients, x))
        return round(value, _ROUND)

    def save_model(self, path: str | Path) -> None:
        payload = {
            "feature_order": self.feature_order,
            "coefficients": [round(float(c), _ROUND) for c in self.coefficients.tolist()],
            "fitted": bool(self.fitted),
        }
        text = json.dumps(canonicalize(payload), indent=2, sort_keys=True)
        Path(path).write_text(f"{text}\n", encoding="utf-8")

    @classmethod
    def load_model(cls, path: str | Path) -> "SpectralThresholdModel":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(feature_order=[str(name) for name in data.get("feature_order", _DEFAULT_FEATURE_ORDER)])
        coeffs = np.asarray(data.get("coefficients", []), dtype=np.float64)
        if coeffs.size == len(model.feature_order) + 1:
            model.coefficients = np.round(coeffs, _ROUND)
        model.fitted = bool(data.get("fitted", False))
        return model


def load_training_dataset(experiments_root: str | Path = "experiments") -> list[dict[str, Any]]:
    root = Path(experiments_root)
    if not root.exists():
        return []

    dataset: list[dict[str, Any]] = []
    for result_path in sorted(root.glob("*/results.json")):
        data = json.loads(result_path.read_text(encoding="utf-8"))
        spectral = data.get("spectral_metrics", {})
        threshold = data.get("threshold")
        if threshold is None:
            continue
        row = {
            "nb_spectral_radius": float(spectral.get("nb_spectral_radius", 0.0)),
            "bethe_negative_mass": float(spectral.get("bethe_negative_mass", spectral.get("bethe_hessian_negative_modes", 0.0))),
            "flow_ipr": float(spectral.get("flow_ipr", spectral.get("ipr_localization", 0.0))),
            "spectral_entropy": float(spectral.get("spectral_entropy", 0.0)),
            "trap_similarity": float(spectral.get("trap_similarity", 0.0)),
            "threshold": float(threshold),
        }
        dataset.append(row)
    return dataset
