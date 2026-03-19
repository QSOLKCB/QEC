"""
v8.4.0 — Generated Graph Export.

Exports a generated parity-check matrix and its metadata to a JSON
file for reuse in later experiments.

Layer 3 — Generation.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from qec.generation.candidate_evaluation import evaluate_tanner_graph_candidate
from utils.canonicalize import canonicalize


def export_generated_graph(
    H: np.ndarray,
    path: str,
) -> dict[str, Any]:
    """Export a generated graph with metrics to a JSON file.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    path : str
        Output file path for the JSON artifact.

    Returns
    -------
    dict[str, Any]
        The exported artifact dictionary.
    """
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape

    metrics = evaluate_tanner_graph_candidate(H_arr)

    artifact = {
        "num_variables": int(n),
        "num_checks": int(m),
        "H": H_arr.tolist(),
        "metrics": metrics,
        "stability_score": -metrics["instability_score"],
    }

    artifact = canonicalize(artifact)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(artifact, f, sort_keys=True, separators=(",", ":"))
        f.write("\n")

    return artifact
