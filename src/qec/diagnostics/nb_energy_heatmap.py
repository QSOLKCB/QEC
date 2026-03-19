"""
v8.0.0 — NB Eigenvector Energy Heatmap.

Computes per-node energy heatmaps from the dominant non-backtracking
eigenvector.  Aggregates directed edge energy onto variable and check
nodes to produce node-level instability heat scores.

Layer 3 — Diagnostics.
Does not import or modify the decoder (Layer 1).
Does not import from experiments (Layer 5) or bench (Layer 6).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qec.diagnostics._spectral_utils import build_directed_edges
from qec.diagnostics.spectral_nb import _TannerGraph, compute_nb_spectrum


_ROUND = 12


def compute_nb_energy_heatmap(
    H: np.ndarray,
) -> dict[str, Any]:
    """Compute per-node NB eigenvector energy heatmap.

    Aggregates edge energy |v_e|^2 from the dominant NB eigenvector
    onto variable and check nodes.  Each node's heat is the sum of
    edge energies on its incident directed edges.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:

        - ``variable_node_heat`` : list[float] — length n
        - ``check_node_heat`` : list[float] — length m
        - ``max_variable_heat`` : float
        - ``max_check_heat`` : float
        - ``hottest_variable_node`` : int
        - ``hottest_check_node`` : int
        - ``total_energy`` : float
    """
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape

    spectrum = compute_nb_spectrum(H_arr)
    edge_energy = spectrum["edge_energy"]

    graph = _TannerGraph(H_arr)
    directed_edges = build_directed_edges(graph)

    # Aggregate edge energy onto nodes
    var_heat = np.zeros(n, dtype=np.float64)
    chk_heat = np.zeros(m, dtype=np.float64)

    for idx, (u, v) in enumerate(directed_edges):
        energy = edge_energy[idx] if idx < len(edge_energy) else 0.0
        # u and v are Tanner graph node indices
        # variable nodes: 0..n-1, check nodes: n..n+m-1
        if u < n:
            var_heat[u] += energy
        else:
            chk_heat[u - n] += energy
        if v < n:
            var_heat[v] += energy
        else:
            chk_heat[v - n] += energy

    total_energy = float(edge_energy.sum()) if len(edge_energy) > 0 else 0.0

    max_var_heat = float(np.max(var_heat)) if n > 0 else 0.0
    max_chk_heat = float(np.max(chk_heat)) if m > 0 else 0.0
    hottest_var = int(np.argmax(var_heat)) if n > 0 else 0
    hottest_chk = int(np.argmax(chk_heat)) if m > 0 else 0

    return {
        "variable_node_heat": [round(float(h), _ROUND) for h in var_heat],
        "check_node_heat": [round(float(h), _ROUND) for h in chk_heat],
        "max_variable_heat": round(max_var_heat, _ROUND),
        "max_check_heat": round(max_chk_heat, _ROUND),
        "hottest_variable_node": hottest_var,
        "hottest_check_node": hottest_chk,
        "total_energy": round(total_energy, _ROUND),
    }
