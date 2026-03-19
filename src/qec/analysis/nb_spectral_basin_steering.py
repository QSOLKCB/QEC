"""v12.9.0 — NB spectral basin steering metric."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse

from qec.analysis.nb_trapping_set_predictor import NBTrappingSetPredictor


_ROUND = 12


class NBSpectralBasinSteering:
    """Compute deterministic spectral-basin steering score for Tanner graphs."""

    def __init__(self, *, precision: int = _ROUND) -> None:
        self.precision = precision
        self._predictor = NBTrappingSetPredictor(precision=precision)

    def compute_steering(
        self,
        H: np.ndarray | scipy.sparse.spmatrix,
        prediction: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if prediction is None:
            prediction = self._predictor.predict_trapping_regions(H)
        spectral_radius = round(float(prediction.get("spectral_radius", 0.0)), self.precision)
        ipr = round(float(prediction.get("ipr", 0.0)), self.precision)
        risk_score = round(float(prediction.get("risk_score", 0.0)), self.precision)
        steering_score = spectral_radius + ipr + (2.0 * risk_score)

        return {
            "spectral_radius": spectral_radius,
            "ipr": ipr,
            "risk_score": risk_score,
            "trapping_risk": risk_score,
            "steering_score": round(float(steering_score), self.precision),
        }
