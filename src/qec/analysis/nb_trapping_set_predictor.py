"""
v12.8.0 — NB Eigenvector Trapping-Set Predictor.

Predicts unstable Tanner-graph regions before decoding runs by
analysing localization of the dominant non-backtracking eigenvector.

Layer 3 — Analysis.
Does not import or modify decoder internals.
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse

from src.qec.analysis.eigenvector_localization import EigenvectorLocalizationAnalyzer
from src.qec.analysis.nonbacktracking_flow import NonBacktrackingFlowAnalyzer


_ROUND = 12


class NBTrappingSetPredictor:
    """Predict trapping-set candidate regions from NB eigenvector localization."""

    def __init__(self, *, precision: int = _ROUND) -> None:
        self.precision = precision
        self._flow_analyzer = NonBacktrackingFlowAnalyzer()

    def predict_trapping_regions(
        self,
        H: np.ndarray | scipy.sparse.spmatrix,
    ) -> dict[str, Any]:
        """Predict unstable candidate trapping regions.

        Returns rounded deterministic outputs with keys:
          node_scores, edge_scores, candidate_sets, ipr,
          spectral_radius, risk_score.
        """
        H_arr = self._to_dense_copy(H)
        m, n = H_arr.shape

        if m == 0 or n == 0:
            return {
                "node_scores": {},
                "edge_scores": {},
                "candidate_sets": [],
                "ipr": 0.0,
                "spectral_radius": 0.0,
                "risk_score": 0.0,
            }

        flow = self._flow_analyzer.compute_flow(H_arr)
        directed_edges = flow.get("directed_edges", [])
        directed_edge_flow = np.asarray(
            flow.get("directed_edge_flow", np.zeros(0, dtype=np.float64)),
            dtype=np.float64,
        )

        dir_index = {edge: i for i, edge in enumerate(directed_edges)}

        edges = self._collect_edges(H_arr)
        edge_scores: dict[tuple[int, int], float] = {}
        node_scores = np.zeros(n, dtype=np.float64)

        for ci, vi in edges:
            idx_fwd = dir_index.get((vi, n + ci))
            idx_rev = dir_index.get((n + ci, vi))
            fwd = abs(float(directed_edge_flow[idx_fwd])) if idx_fwd is not None else 0.0
            rev = abs(float(directed_edge_flow[idx_rev])) if idx_rev is not None else 0.0
            edge_score = fwd + rev
            edge_scores[(ci, vi)] = round(edge_score, self.precision)
            node_scores[vi] += edge_score

        rounded_node_scores: dict[int, float] = {
            vi: round(float(node_scores[vi]), self.precision)
            for vi in range(n)
        }

        ipr = EigenvectorLocalizationAnalyzer.compute_ipr(
            flow.get("variable_flow", np.zeros(n, dtype=np.float64)),
        )["ipr"]

        spectral_radius = round(float(flow.get("max_flow", 0.0)), self.precision)

        mean_score = float(np.mean(node_scores)) if n > 0 else 0.0
        std_score = float(np.std(node_scores)) if n > 0 else 0.0
        threshold = mean_score + 2.0 * std_score

        candidate_nodes = sorted(
            vi for vi in range(n)
            if float(node_scores[vi]) > threshold
        )
        candidate_sets = self._cluster_candidate_variables(H_arr, candidate_nodes)

        if candidate_nodes:
            risk_raw = float(np.mean([node_scores[vi] for vi in candidate_nodes])) * float(ipr)
            risk_score = round(risk_raw, self.precision)
        else:
            risk_score = 0.0

        rounded_edge_scores = {
            edge: round(float(score), self.precision)
            for edge, score in sorted(edge_scores.items(), key=lambda x: (x[0][0], x[0][1]))
        }

        return {
            "node_scores": rounded_node_scores,
            "edge_scores": rounded_edge_scores,
            "candidate_sets": candidate_sets,
            "ipr": round(float(ipr), self.precision),
            "spectral_radius": spectral_radius,
            "risk_score": risk_score,
        }

    @staticmethod
    def _to_dense_copy(H: np.ndarray | scipy.sparse.spmatrix) -> np.ndarray:
        if scipy.sparse.issparse(H):
            return np.asarray(H.todense(), dtype=np.float64)
        return np.asarray(H, dtype=np.float64).copy()

    @staticmethod
    def _collect_edges(H: np.ndarray) -> list[tuple[int, int]]:
        m, n = H.shape
        edges: list[tuple[int, int]] = []
        for ci in range(m):
            for vi in range(n):
                if H[ci, vi] != 0:
                    edges.append((ci, vi))
        edges.sort()
        return edges

    @staticmethod
    def _cluster_candidate_variables(
        H: np.ndarray,
        candidate_nodes: list[int],
    ) -> list[list[int]]:
        if not candidate_nodes:
            return []

        m, _ = H.shape
        cand_set = set(candidate_nodes)

        adj: dict[int, set[int]] = {vi: set() for vi in candidate_nodes}
        for ci in range(m):
            # derive variable indices from non-zeros in H[ci] and intersect with candidate set
            row_nz = np.flatnonzero(H[ci])
            vars_on_check = [vi for vi in row_nz if vi in cand_set]
            if len(vars_on_check) < 2:
                continue
            vars_on_check.sort()
            for i, va in enumerate(vars_on_check):
                for vb in vars_on_check[i + 1:]:
                    adj[va].add(vb)
                    adj[vb].add(va)

        visited: set[int] = set()
        components: list[list[int]] = []
        for root in sorted(cand_set):
            if root in visited:
                continue
            stack = [root]
            visited.add(root)
            comp: list[int] = []
            while stack:
                cur = stack.pop()
                comp.append(cur)
                for nxt in sorted(adj[cur]):
                    if nxt not in visited:
                        visited.add(nxt)
                        stack.append(nxt)
            comp.sort()
            components.append(comp)

        components.sort(key=lambda comp: (-len(comp), comp[0]))
        return components
