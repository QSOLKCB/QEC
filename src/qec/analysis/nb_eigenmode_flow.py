"""
v13.0.0 — Non-Backtracking Eigenmode Flow Analyzer.

Computes a compact deterministic eigenmode signature and edge-pressure map
from the dominant non-backtracking eigenmode.

Layer 3 — Analysis.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse

from src.qec.analysis.nonbacktracking_flow import NonBacktrackingFlowAnalyzer
from src.qec.fitness.spectral_metrics import compute_nbt_spectral_radius


_ROUND = 12


class NBEigenmodeFlowAnalyzer:
    """Compute deterministic NB eigenmode edge pressure and signature."""

    def __init__(
        self,
        *,
        power_iterations: int = 50,
        support_threshold_ratio: float = 0.5,
    ) -> None:
        self._flow = NonBacktrackingFlowAnalyzer(power_iterations=power_iterations)
        self.support_threshold_ratio = support_threshold_ratio

    def analyze(
        self,
        H: np.ndarray | scipy.sparse.spmatrix,
    ) -> dict[str, Any]:
        """Return eigenmode pressure diagnostics and compact signature.

        Directed NB-edge magnitudes are aggregated into undirected Tanner edges
        by summing absolute magnitudes of both directed orientations:
        |vi->(n+ci)| + |(n+ci)->vi|.
        """
        H_arr = self._to_dense(H)
        m, n = H_arr.shape

        if m == 0 or n == 0:
            return self._empty_signature(m, n)

        flow = self._flow.compute_flow(H_arr)
        directed_edges = flow.get("directed_edges", [])
        directed_edge_flow = np.asarray(
            flow.get("directed_edge_flow", np.zeros(0, dtype=np.float64)),
            dtype=np.float64,
        )

        edges = self._collect_edges(H_arr)
        edge_scores = self._compute_edge_scores(
            n=n,
            edges=edges,
            directed_edges=directed_edges,
            directed_edge_flow=directed_edge_flow,
        )
        ranked_edges = self._ranked_edges(edge_scores)

        max_score = max(edge_scores.values()) if edge_scores else 0.0
        total_mass = float(sum(edge_scores.values()))
        threshold = self.support_threshold_ratio * max_score
        support_count = sum(1 for value in edge_scores.values() if value >= threshold)

        k = max(1, int(np.ceil(np.sqrt(len(edges))))) if edges else 1
        topk_mass = sum(score for _, score in ranked_edges[:k])

        if max_score <= 1e-15 or total_mass <= 1e-15:
            support_fraction = 0.0
            topk_mass_fraction = 0.0
        else:
            support_fraction = (
                float(support_count) / float(len(edges)) if edges else 0.0
            )
            topk_mass_fraction = topk_mass / total_mass

        mode_ipr = float(flow.get("flow_localization", 0.0))
        spectral_radius = float(compute_nbt_spectral_radius(H_arr))

        signature = {
            "spectral_radius": round(spectral_radius, _ROUND),
            "mode_ipr": round(mode_ipr, _ROUND),
            "support_fraction": round(float(support_fraction), _ROUND),
            "topk_mass_fraction": round(float(topk_mass_fraction), _ROUND),
        }

        return {
            **signature,
            "edge_scores": edge_scores,
            "hot_edges": [edge for edge, _ in ranked_edges],
            "signature": signature,
        }

    @staticmethod
    def _to_dense(H: np.ndarray | scipy.sparse.spmatrix) -> np.ndarray:
        if scipy.sparse.issparse(H):
            return np.asarray(H.todense(), dtype=np.float64)
        return np.asarray(H, dtype=np.float64).copy()

    @staticmethod
    def _collect_edges(H: np.ndarray) -> list[tuple[int, int]]:
        coords = np.argwhere(H != 0)
        return [(int(ci), int(vi)) for ci, vi in coords]

    @staticmethod
    def _compute_edge_scores(
        *,
        n: int,
        edges: list[tuple[int, int]],
        directed_edges: list[tuple[int, int]],
        directed_edge_flow: np.ndarray,
    ) -> dict[tuple[int, int], float]:
        de_index = {edge: i for i, edge in enumerate(directed_edges)}
        scores: dict[tuple[int, int], float] = {}
        for ci, vi in edges:
            j1 = de_index.get((vi, n + ci))
            j2 = de_index.get((n + ci, vi))
            score = 0.0
            if j1 is not None and j1 < len(directed_edge_flow):
                score += abs(float(directed_edge_flow[j1]))
            if j2 is not None and j2 < len(directed_edge_flow):
                score += abs(float(directed_edge_flow[j2]))
            scores[(ci, vi)] = round(score, _ROUND)
        return scores

    @staticmethod
    def _ranked_edges(
        edge_scores: dict[tuple[int, int], float],
    ) -> list[tuple[tuple[int, int], float]]:
        return sorted(
            edge_scores.items(),
            key=lambda kv: (-kv[1], kv[0][0], kv[0][1]),
        )

    @staticmethod
    def _empty_signature(m: int, n: int) -> dict[str, Any]:
        del m, n
        signature = {
            "spectral_radius": 0.0,
            "mode_ipr": 0.0,
            "support_fraction": 0.0,
            "topk_mass_fraction": 0.0,
        }
        return {
            **signature,
            "edge_scores": {},
            "hot_edges": [],
            "signature": signature,
        }
