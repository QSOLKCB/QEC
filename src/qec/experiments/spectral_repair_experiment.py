"""
v7.8.0 — Spectral Repair Experiment.

Runs one deterministic single-step graph repair and produces a
canonical JSON artifact recording before/after spectral diagnostics.

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from qec.diagnostics.spectral_repair import (
    propose_repair_candidates,
    select_best_repair,
)


_REPAIR_SCHEMA_VERSION = 1


def run_spectral_repair_experiment(
    H: np.ndarray,
    *,
    top_k_edges: int = 10,
    max_candidates: int = 50,
) -> dict[str, Any]:
    """Run a single-step deterministic repair experiment.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    top_k_edges : int
        Number of top hot edges to use as repair anchors.
    max_candidates : int
        Maximum number of candidates to generate.

    Returns
    -------
    dict[str, Any]
        JSON-serializable repair artifact.
    """
    H_arr = np.asarray(H, dtype=np.float64)

    # Generate candidates
    candidates = propose_repair_candidates(
        H_arr, top_k_edges=top_k_edges, max_candidates=max_candidates,
    )

    # Select best repair
    result = select_best_repair(H_arr, candidates)

    before = result["before_metrics"]
    after = result["after_metrics"]

    artifact = {
        "schema_version": _REPAIR_SCHEMA_VERSION,
        "original_spectral_radius": before["spectral_radius"],
        "original_ipr": before["ipr"],
        "original_eeec": before["eeec"],
        "original_sis": before["sis"],
        "repaired_spectral_radius": after["spectral_radius"],
        "repaired_ipr": after["ipr"],
        "repaired_eeec": after["eeec"],
        "repaired_sis": after["sis"],
        "delta_spectral_radius": round(
            after["spectral_radius"] - before["spectral_radius"], 12,
        ),
        "delta_ipr": round(after["ipr"] - before["ipr"], 12),
        "delta_eeec": round(after["eeec"] - before["eeec"], 12),
        "delta_sis": round(after["sis"] - before["sis"], 12),
        "selected_candidate": result["selected_candidate"],
        "num_candidates_considered": len(candidates),
        "num_valid_candidates": result["num_candidates_scored"],
        "improved": result["improved"],
    }

    return artifact


def serialize_repair_artifact(artifact: dict[str, Any]) -> str:
    """Serialize repair artifact to canonical JSON."""
    return json.dumps(artifact, sort_keys=True)
