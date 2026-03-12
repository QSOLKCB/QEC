"""
v12.0.0 — Non-Backtracking Flow Analyzer.

Detects instability propagation channels in Tanner graphs using the
dominant eigenvector of the non-backtracking matrix.  The eigenvector
represents the dominant structural shear direction in the graph.

Layer 3 — Analysis.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse


_POWER_ITER = 50
_ROUND = 12


class NonBacktrackingFlowAnalyzer:
    """Compute instability flow via non-backtracking eigenvector.

    Parameters
    ----------
    power_iterations : int
        Number of power iterations for the eigensolver (default 50).
    """

    def __init__(self, power_iterations: int = _POWER_ITER) -> None:
        self.power_iterations = power_iterations

    def compute_flow(self, H: np.ndarray | scipy.sparse.spmatrix) -> dict[str, Any]:
        """Compute non-backtracking flow for variables and edges.

        Parameters
        ----------
        H : np.ndarray or scipy.sparse.spmatrix
            Binary parity-check matrix, shape (m, n).

        Returns
        -------
        dict
            directed_edge_flow : np.ndarray, shape (2*num_edges,)
            edge_flow : np.ndarray, shape (num_edges,)
            variable_flow : np.ndarray, shape (n,)
            check_flow : np.ndarray, shape (m,)
            max_flow : float
            mean_flow : float
            flow_localization : float
        """
        if scipy.sparse.issparse(H):
            H_csr = H.tocsr()
            m, n = H_csr.shape
        else:
            H_arr = np.asarray(H, dtype=np.float64)
            m, n = H_arr.shape

        empty = {
            "directed_edge_flow": np.zeros(0, dtype=np.float64),
            "edge_flow": np.zeros(0, dtype=np.float64),
            "variable_flow": np.zeros(n, dtype=np.float64),
            "check_flow": np.zeros(m, dtype=np.float64),
            "max_flow": 0.0,
            "mean_flow": 0.0,
            "flow_localization": 0.0,
        }

        if m == 0 or n == 0:
            return empty

        # Collect undirected edges (ci, vi) sorted deterministically.
        edges: list[tuple[int, int]] = []
        if scipy.sparse.issparse(H):
            for ci in range(m):
                for vi in H_csr.indices[
                    H_csr.indptr[ci] : H_csr.indptr[ci + 1]
                ]:
                    edges.append((ci, int(vi)))
        else:
            for ci in range(m):
                for vi in range(n):
                    if H_arr[ci, vi] != 0:
                        edges.append((ci, vi))
        edges.sort()

        num_edges = len(edges)
        if num_edges == 0:
            return empty

        # Build directed edge list for non-backtracking operator.
        # Each undirected edge (ci, vi) produces two directed edges:
        #   vi -> (n + ci)  and  (n + ci) -> vi
        directed_edges: list[tuple[int, int]] = []
        adj: dict[int, list[int]] = {}
        for ci, vi in edges:
            directed_edges.append((vi, n + ci))
            directed_edges.append((n + ci, vi))
            adj.setdefault(vi, []).append(n + ci)
            adj.setdefault(n + ci, []).append(vi)

        for node in adj:
            adj[node] = sorted(adj[node])
        directed_edges.sort()

        num_de = len(directed_edges)
        edge_index = {e: i for i, e in enumerate(directed_edges)}

        # Build sparse non-backtracking matrix B.
        # B[j, i] = 1  iff  directed_edges[i] = (u, v) and
        #                    directed_edges[j] = (v, w) with w != u
        rows: list[int] = []
        cols: list[int] = []
        for i, (u, v) in enumerate(directed_edges):
            for w in adj.get(v, []):
                if w == u:
                    continue
                j = edge_index.get((v, w))
                if j is not None:
                    rows.append(j)
                    cols.append(i)

        if not rows:
            return {
                "directed_edge_flow": np.zeros(num_de, dtype=np.float64),
                "edge_flow": np.zeros(num_edges, dtype=np.float64),
                "variable_flow": np.zeros(n, dtype=np.float64),
                "check_flow": np.zeros(m, dtype=np.float64),
                "max_flow": 0.0,
                "mean_flow": 0.0,
                "flow_localization": 0.0,
            }

        data = np.ones(len(rows), dtype=np.float64)
        B = scipy.sparse.csr_matrix(
            (data, (rows, cols)), shape=(num_de, num_de),
        )

        # Compute leading eigenvector via deterministic power iteration.
        ev = self._power_iteration(B, num_de)

        # Normalize eigenvector
        ev_norm = np.linalg.norm(ev)
        if ev_norm < 1e-15:
            return {
                "directed_edge_flow": np.zeros(num_de, dtype=np.float64),
                "edge_flow": np.zeros(num_edges, dtype=np.float64),
                "variable_flow": np.zeros(n, dtype=np.float64),
                "check_flow": np.zeros(m, dtype=np.float64),
                "max_flow": 0.0,
                "mean_flow": 0.0,
                "flow_localization": 0.0,
            }
        ev = ev / ev_norm

        directed_edge_flow = np.array(
            [round(float(ev[i]), _ROUND) for i in range(num_de)],
            dtype=np.float64,
        )

        # Aggregate directed-edge flow into undirected edge flow.
        edge_flow = np.zeros(num_edges, dtype=np.float64)
        for idx, (ci, vi) in enumerate(edges):
            j1 = edge_index.get((vi, n + ci))
            j2 = edge_index.get((n + ci, vi))
            f = 0.0
            if j1 is not None:
                f += ev[j1]
            if j2 is not None:
                f += ev[j2]
            edge_flow[idx] = round(f, _ROUND)

        # Aggregate edge flow into variable flow and check flow.
        variable_flow = np.zeros(n, dtype=np.float64)
        check_flow = np.zeros(m, dtype=np.float64)
        for idx, (ci, vi) in enumerate(edges):
            variable_flow[vi] += edge_flow[idx]
            check_flow[ci] += edge_flow[idx]

        # Normalize variable_flow to [0, 1].
        vf_max = variable_flow.max()
        if vf_max > 1e-15:
            variable_flow = variable_flow / vf_max

        # Round for determinism.
        variable_flow = np.array(
            [round(v, _ROUND) for v in variable_flow],
            dtype=np.float64,
        )
        check_flow = np.array(
            [round(v, _ROUND) for v in check_flow],
            dtype=np.float64,
        )

        max_flow = float(round(variable_flow.max(), _ROUND))
        mean_flow = float(round(variable_flow.mean(), _ROUND))

        # Flow localization: IPR of variable_flow.
        vf_sum = variable_flow.sum()
        if vf_sum > 1e-15:
            p = variable_flow / vf_sum
            ipr = float(np.sum(p ** 2))
        else:
            ipr = 0.0
        flow_localization = float(round(ipr, _ROUND))

        return {
            "directed_edge_flow": directed_edge_flow,
            "edge_flow": edge_flow,
            "variable_flow": variable_flow,
            "check_flow": check_flow,
            "max_flow": max_flow,
            "mean_flow": mean_flow,
            "flow_localization": flow_localization,
        }

    def _power_iteration(
        self, B: scipy.sparse.csr_matrix, size: int,
    ) -> np.ndarray:
        """Deterministic power iteration for leading eigenvector magnitude."""
        x = np.ones(size, dtype=np.float64)
        x /= np.linalg.norm(x)
        for _ in range(self.power_iterations):
            y = B.dot(x)
            norm_y = np.linalg.norm(y)
            if norm_y < 1e-15:
                return np.abs(x)
            x = y / norm_y
        return np.abs(x)
