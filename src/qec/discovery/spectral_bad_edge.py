"""
v9.0.0 — Spectral Bad-Edge Detector.

Identifies edges contributing most to spectral instability by scoring
each edge using the dominant NB eigenvector components.

edge_score(e) = |v_i * v_j|

where v_i and v_j are eigenvector components at the directed edge
endpoints.

Layer 3 — Discovery.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qec.diagnostics._spectral_utils import (
    build_directed_edges,
    compute_nb_dominant_eigenpair,
)
from qec.diagnostics.spectral_nb import _TannerGraph


_ROUND = 12


def detect_bad_edges(H: np.ndarray) -> dict[str, Any]:
    """Identify edges contributing most to spectral instability.

    Procedure:
    1. Compute dominant NB eigenvector.
    2. For each undirected edge (ci, vi), compute
       edge_score = |v_{(vi,ci+n)} * v_{(ci+n,vi)}|
       using the directed edge components.
    3. Return edges sorted by descending score.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - ``bad_edges`` : list of (ci, vi) sorted by descending score
        - ``edge_scores`` : list of (ci, vi, score)
        - ``max_score`` : float
        - ``mean_score`` : float
    """
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape

    graph = _TannerGraph(H_arr)
    _, eigenvector, directed_edges = compute_nb_dominant_eigenpair(graph)

    # Normalize and canonicalize sign
    norm = np.linalg.norm(eigenvector)
    if norm > 0:
        eigenvector = eigenvector / norm
    max_idx = int(np.argmax(np.abs(eigenvector)))
    if eigenvector[max_idx] < 0:
        eigenvector = -eigenvector

    # Build edge-to-index mapping
    edge_index = {e: i for i, e in enumerate(directed_edges)}

    # Score each undirected edge (ci, vi)
    edge_scores: list[tuple[int, int, float]] = []
    for ci in range(m):
        for vi in range(n):
            if H_arr[ci, vi] == 0:
                continue

            cnode = n + ci  # Check node index in graph

            # Directed edges: (vi, cnode) and (cnode, vi)
            idx_vc = edge_index.get((vi, cnode))
            idx_cv = edge_index.get((cnode, vi))

            if idx_vc is not None and idx_cv is not None:
                score = abs(float(eigenvector[idx_vc] * eigenvector[idx_cv]))
            elif idx_vc is not None:
                score = abs(float(eigenvector[idx_vc])) ** 2
            elif idx_cv is not None:
                score = abs(float(eigenvector[idx_cv])) ** 2
            else:
                score = 0.0

            edge_scores.append((ci, vi, round(score, _ROUND)))

    # Sort by score descending, then (ci, vi) for determinism
    edge_scores.sort(key=lambda e: (-e[2], e[0], e[1]))

    scores_only = [s for _, _, s in edge_scores]
    max_s = max(scores_only) if scores_only else 0.0
    mean_s = sum(scores_only) / len(scores_only) if scores_only else 0.0

    return {
        "bad_edges": [(ci, vi) for ci, vi, _ in edge_scores],
        "edge_scores": edge_scores,
        "max_score": round(max_s, _ROUND),
        "mean_score": round(mean_s, _ROUND),
    }
