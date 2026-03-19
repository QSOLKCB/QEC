"""Deterministic counterexample extraction for spectral conjectures."""

from __future__ import annotations

from typing import Any

import numpy as np

from qec.analysis.spectral_conjecture_validation import _predict


_MAX_COUNTEREXAMPLES_PER_CONJECTURE = 128


def extract_counterexamples(
    conjecture: dict[str, Any],
    dataset: dict[str, Any],
    error_threshold: float,
    max_counterexamples: int = _MAX_COUNTEREXAMPLES_PER_CONJECTURE,
) -> list[dict[str, Any]]:
    """Return deterministic high-error rows sorted by error desc then row index."""
    y = np.asarray(dataset.get("y", np.zeros((0,), dtype=np.float64)), dtype=np.float64)
    pred = _predict(conjecture, dataset)
    n = int(y.shape[0])
    if pred.shape[0] != n:
        pred = np.zeros((n,), dtype=np.float64)
    abs_error = np.asarray(np.abs(pred - y), dtype=np.float64)
    thresh = np.float64(error_threshold)

    idx = np.flatnonzero(abs_error > thresh)
    ordered = sorted(idx.tolist(), key=lambda i: (-float(abs_error[i]), int(i)))
    limit = int(max(0, max_counterexamples))

    conjecture_id = str(conjecture.get("conjecture_id", ""))
    out: list[dict[str, Any]] = []
    for rank, i in enumerate(ordered[:limit]):
        out.append(
            {
                "counterexample_id": f"{conjecture_id}_ce_{rank:04d}",
                "conjecture_id": conjecture_id,
                "row_index": int(i),
                "absolute_error": float(np.float64(abs_error[i])),
                "predicted_value": float(np.float64(pred[i])),
                "observed_value": float(np.float64(y[i])),
            }
        )
    return out
