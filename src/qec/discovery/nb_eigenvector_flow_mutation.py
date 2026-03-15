"""Deterministic non-backtracking eigenvector-flow mutation operator."""

from __future__ import annotations

from typing import Any

import numpy as np

_ROUND = 12


def compute_multi_mode_flow(eigenvectors: np.ndarray) -> np.ndarray:
    """Combine multiple eigenmodes into one deterministic flow vector."""
    v = np.asarray(eigenvectors, dtype=np.complex128)
    if v.ndim == 1:
        v = v.reshape((-1, 1))
    if v.size == 0:
        return np.zeros((0,), dtype=np.float64)

    flow = np.sum(np.abs(v), axis=1, dtype=np.float64)
    norm = float(np.linalg.norm(flow))
    if norm == 0.0:
        return np.asarray(flow, dtype=np.float64)

    flow = flow / norm
    return np.round(np.asarray(flow, dtype=np.float64), _ROUND)


class NBEigenvectorFlowMutator:
    """Deterministic mutation operator guided by NB eigenvector flow."""

    def __init__(self) -> None:
        pass

    def compute_flow(self, eigenvector: np.ndarray) -> np.ndarray:
        """Compute normalized edge-flow magnitude from NB eigenvector."""
        vec = np.asarray(eigenvector, dtype=np.float64)
        if vec.ndim != 1:
            vec = compute_multi_mode_flow(vec)
        flow = np.abs(vec).astype(np.float64)
        total = float(np.sum(flow, dtype=np.float64))
        if total <= 0.0:
            if flow.size == 0:
                return flow
            flow = np.full(flow.shape, 1.0 / float(flow.size), dtype=np.float64)
        else:
            flow = flow / total
        return flow.astype(np.float64)

    def select_edge(self, flow: np.ndarray) -> int:
        """Deterministically choose the directed-edge index with highest flow."""
        if flow.size == 0:
            return -1
        return int(np.argmax(flow))

    def mutate(self, graph: np.ndarray, nb_eigenvector: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Mutate a binary parity-check matrix using the dominant NB eigenvector."""
        flow = self.compute_flow(nb_eigenvector)
        edge_index = self.select_edge(flow)
        mutated = self._mutate_edge(graph, edge_index)
        flow_strength = 0.0
        if flow.size > 0 and edge_index >= 0:
            flow_strength = round(float(flow[edge_index]), _ROUND)
        return mutated, {
            "flow_edge_index": int(edge_index),
            "flow_strength": flow_strength,
        }

    def _mutate_edge(self, graph: np.ndarray, edge_index: int) -> np.ndarray:
        """Deterministic edge rewiring for a binary parity-check matrix."""
        H = np.asarray(graph, dtype=np.float64)
        g = H.copy()
        m, n = g.shape
        if m == 0 or n == 0:
            return g

        edges = np.argwhere(g == 1.0)
        if edges.size == 0 or edge_index < 0:
            return g

        idx = int(edge_index) % int(edges.shape[0])
        u = int(edges[idx, 0])
        v = int(edges[idx, 1])

        new_v = (v + 1) % n
        for offset in range(n):
            cand_v = (new_v + offset) % n
            if cand_v == v:
                continue
            if g[u, cand_v] == 1.0:
                continue

            swap_row = -1
            for r in range(m):
                if r == u:
                    continue
                if g[r, cand_v] == 1.0 and g[r, v] == 0.0:
                    swap_row = r
                    break

            if swap_row >= 0:
                g[u, v] = 0.0
                g[u, cand_v] = 1.0
                g[swap_row, cand_v] = 0.0
                g[swap_row, v] = 1.0
                return g

        return g
