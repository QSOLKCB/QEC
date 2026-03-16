"""
v12.6.0 — NB Instability Gradient Analyzer.

Builds a deterministic instability gradient field from the dominant
non-backtracking eigenvector flow on Tanner graph edges.

Layer 3 — Analysis.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse

from src.qec.analysis.nonbacktracking_flow import NonBacktrackingFlowAnalyzer


_ROUND = 12


class NBInstabilityGradientAnalyzer:
    """Compute edge/node instability and directed instability gradients."""

    def __init__(self) -> None:
        self._flow_analyzer = NonBacktrackingFlowAnalyzer()

    def compute_gradient(
        self,
        H: np.ndarray | scipy.sparse.spmatrix,
    ) -> dict[str, Any]:
        """Compute deterministic NB instability gradient diagnostics.

        Parameters
        ----------
        H : np.ndarray or scipy.sparse.spmatrix
            Binary parity-check matrix of shape (m, n).

        Returns
        -------
        dict
            edge_scores : dict[(ci, vi), float]
                score(ci,vi) = |v_(u→v)| + |v_(v→u)| for each present edge.
            node_instability : dict[int, float]
                Aggregated incident edge instability for all check and variable
                nodes. Variable node ids are offset by m.
            gradient_direction : dict[(ci, vi), float]
                gradient(ci,vi) = node_score(ci) - node_score(vi).
                Here ``vi`` uses the variable-node id with +m offset.
        """
        if scipy.sparse.issparse(H):
            H_arr = np.asarray(H.todense(), dtype=np.float64)
        else:
            H_arr = np.asarray(H, dtype=np.float64)

        m, n = H_arr.shape
        flow = self._flow_analyzer.compute_flow(H_arr)

        directed_edges = flow.get("directed_edges", [])
        directed_edge_flow = np.asarray(
            flow.get("directed_edge_flow", np.zeros(0, dtype=np.float64)),
            dtype=np.float64,
        )
        if len(directed_edges) != len(directed_edge_flow):
            raise ValueError("NB flow vector mismatch")

        edge_scores = self._compute_edge_scores(
            H_arr,
            directed_edges,
            directed_edge_flow,
        )

        node_instability: dict[int, float] = {
            node_id: 0.0 for node_id in range(m + n)
        }
        for (ci, vi), score in edge_scores.items():
            node_instability[ci] = round(node_instability[ci] + score, _ROUND)
            var_node = m + vi
            node_instability[var_node] = round(
                node_instability[var_node] + score,
                _ROUND,
            )

        gradient_direction: dict[tuple[int, int], float] = {}
        for ci, vi in sorted(edge_scores):
            var_node = m + vi
            grad = node_instability[ci] - node_instability[var_node]
            gradient_direction[(ci, vi)] = round(float(grad), _ROUND)

        return {
            "edge_scores": edge_scores,
            "node_instability": node_instability,
            "gradient_direction": gradient_direction,
        }

    @staticmethod
    def _collect_edges(H: np.ndarray) -> list[tuple[int, int]]:
        coords = np.argwhere(H != 0)
        return [(int(row), int(col)) for row, col in coords]

    @staticmethod
    def _compute_edge_scores(
        H: np.ndarray,
        directed_edges: list[tuple[int, int]],
        directed_edge_flow: np.ndarray,
    ) -> dict[tuple[int, int], float]:
        m, n = H.shape
        de_index: dict[tuple[int, int], int] = {
            edge: idx for idx, edge in enumerate(directed_edges)
        }

        edge_scores: dict[tuple[int, int], float] = {}
        for ci, vi in NBInstabilityGradientAnalyzer._collect_edges(H):
            j_forward = de_index.get((vi, n + ci))
            j_reverse = de_index.get((n + ci, vi))

            score = 0.0
            if j_forward is not None:
                score += abs(float(directed_edge_flow[j_forward]))
            if j_reverse is not None:
                score += abs(float(directed_edge_flow[j_reverse]))

            edge_scores[(ci, vi)] = round(float(score), _ROUND)

        return edge_scores
