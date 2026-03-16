"""Deterministic structural hypothesis generation from archive correlations."""

from __future__ import annotations

from typing import Any

import numpy as np


def generate_structural_hypotheses(correlation_map: dict[str, Any]) -> list[dict[str, Any]]:
    """Produce deterministic hypothesis objects from a correlation map."""
    if "feature_correlations" in correlation_map:
        corr_map = correlation_map.get("feature_correlations", {})
    else:
        corr_map = correlation_map

    hypotheses: list[dict[str, Any]] = []
    for idx, feature_name in enumerate(sorted(corr_map.keys())):
        corr = float(np.float64(corr_map.get(feature_name, 0.0)))
        confidence = float(np.float64(min(1.0, abs(corr))))
        hypotheses.append(
            {
                "hypothesis_id": int(idx),
                "feature_name": str(feature_name),
                "correlation_strength": corr,
                "confidence_score": confidence,
            }
        )

    hypotheses = sorted(
        hypotheses,
        key=lambda item: (str(item.get("feature_name", "")), int(item.get("hypothesis_id", 0))),
    )
    return hypotheses
