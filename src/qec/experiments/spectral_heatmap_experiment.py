"""
v7.7.0 — Spectral Heatmap Artifact Experiment.

Runs spectral trapping-set heatmap diagnostics and produces a
deterministic JSON artifact recording the top variable nodes,
check nodes, and edges by spectral heat.

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from qec.diagnostics.spectral_heatmaps import (
    compute_spectral_heatmaps,
    rank_variable_nodes_by_heat,
    rank_check_nodes_by_heat,
    rank_edges_by_heat,
)


_HEATMAP_SCHEMA_VERSION = 1


def run_spectral_heatmap_experiment(
    H: np.ndarray,
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    """Run spectral heatmap experiment and produce artifact.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    top_k : int
        Number of top items to include in artifact.

    Returns
    -------
    dict[str, Any]
        JSON-serializable heatmap artifact.
    """
    H_arr = np.asarray(H, dtype=np.float64)

    heatmaps = compute_spectral_heatmaps(H_arr)

    variable_ranking = rank_variable_nodes_by_heat(H_arr)
    check_ranking = rank_check_nodes_by_heat(H_arr)
    edge_ranking = rank_edges_by_heat(H_arr)

    # Top-k items
    top_variable = variable_ranking[:top_k]
    top_check = check_ranking[:top_k]
    top_edges = edge_ranking[:top_k]

    # Max heat values
    var_heat = heatmaps["variable_node_heat"]
    chk_heat = heatmaps["check_node_heat"]

    max_variable_heat = float(np.max(var_heat)) if len(var_heat) > 0 else 0.0
    max_check_heat = float(np.max(chk_heat)) if len(chk_heat) > 0 else 0.0

    artifact = {
        "schema_version": _HEATMAP_SCHEMA_VERSION,
        "spectral_radius": heatmaps["spectral_radius"],
        "ipr": heatmaps["ipr"],
        "eeec": heatmaps["eeec"],
        "sis": heatmaps["sis"],
        "max_variable_heat": round(max_variable_heat, 12),
        "max_check_heat": round(max_check_heat, 12),
        "num_variable_nodes": len(var_heat),
        "num_check_nodes": len(chk_heat),
        "num_undirected_edges": len(heatmaps["undirected_edge_heat"]),
        "top_variable_nodes": [
            {"index": idx, "heat": val} for idx, val in top_variable
        ],
        "top_check_nodes": [
            {"index": idx, "heat": val} for idx, val in top_check
        ],
        "top_edges": [
            {"index": idx, "heat": val} for idx, val in top_edges
        ],
    }

    return artifact


def serialize_heatmap_artifact(artifact: dict[str, Any]) -> str:
    """Serialize heatmap artifact to canonical JSON."""
    return json.dumps(artifact, sort_keys=True)
