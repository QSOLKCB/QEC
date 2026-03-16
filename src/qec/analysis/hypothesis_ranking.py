"""Deterministic ranking of structural hypotheses."""

from __future__ import annotations

from typing import Any

import numpy as np


def rank_hypotheses(hypotheses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank hypotheses by |correlation_strength| * confidence_score using lexsort."""
    if not hypotheses:
        return []

    n = len(hypotheses)
    score = np.zeros((n,), dtype=np.float64)
    hypothesis_ids = np.zeros((n,), dtype=np.int64)
    for i, item in enumerate(hypotheses):
        corr = np.float64(item.get("correlation_strength", 0.0))
        conf = np.float64(item.get("confidence_score", 0.0))
        score[i] = np.float64(np.abs(corr) * conf)
        hypothesis_ids[i] = int(item.get("hypothesis_id", 0))

    order = np.lexsort((hypothesis_ids, -score))
    ranked: list[dict[str, Any]] = []
    for i in order.tolist():
        enriched = dict(hypotheses[i])
        enriched["score"] = float(np.float64(score[i]))
        ranked.append(enriched)
    return ranked
