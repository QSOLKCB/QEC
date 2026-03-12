"""
v11.6.0 — Spectral Localization Analyzer.

Computes spectral pressure scores for Tanner graph variables and edges
using the non-backtracking eigenvector.  High pressure indicates
structurally fragile regions where belief-propagation instability
concentrates.

Layer 3 — Analysis.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse
import scipy.sparse.linalg


_POWER_ITER = 50
_ROUND = 12


class SpectralLocalizationAnalyzer:
    """Compute spectral pressure via non-backtracking eigenvector localization.

    Parameters
    ----------
    power_iterations : int
        Number of power iterations for the sparse eigensolver fallback
        (default 50).
    """

    def __init__(self, power_iterations: int = _POWER_ITER) -> None:
        self.power_iterations = power_iterations

    def compute_pressure(self, H: np.ndarray | scipy.sparse.spmatrix) -> dict[str, Any]:
        """Compute spectral pressure scores for variables and edges.

        Parameters
        ----------
        H : np.ndarray or scipy.sparse.spmatrix
            Binary parity-check matrix, shape (m, n).

        Returns
        -------
        dict
            variable_pressure : np.ndarray, shape (n,)
            edge_pressure : np.ndarray, shape (num_edges,)
            max_pressure : float
            mean_pressure : float
        """
        if scipy.sparse.issparse(H):
            H_csr = H.tocsr()
            m, n = H_csr.shape
        else:
            H_arr = np.asarray(H, dtype=np.float64)
            m, n = H_arr.shape

        if m == 0 or n == 0:
            return {
                "variable_pressure": np.zeros(n, dtype=np.float64),
                "edge_pressure": np.zeros(0, dtype=np.float64),
                "max_pressure": 0.0,
                "mean_pressure": 0.0,
            }

        # Collect undirected edges (ci, vi) sorted deterministically
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
            return {
                "variable_pressure": np.zeros(n, dtype=np.float64),
                "edge_pressure": np.zeros(0, dtype=np.float64),
                "max_pressure": 0.0,
                "mean_pressure": 0.0,
            }

        # Build directed edge list for non-backtracking operator.
        # Each undirected edge (ci, vi) produces two directed edges:
        #   vi -> (n + ci)  and  (n + ci) -> vi
        # Total nodes in bipartite graph: n variable nodes + m check nodes.
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
                "variable_pressure": np.zeros(n, dtype=np.float64),
                "edge_pressure": np.zeros(num_edges, dtype=np.float64),
                "max_pressure": 0.0,
                "mean_pressure": 0.0,
            }

        data = np.ones(len(rows), dtype=np.float64)
        B = scipy.sparse.csr_matrix(
            (data, (rows, cols)), shape=(num_de, num_de),
        )

        # Compute leading eigenvector via deterministic power iteration.
        # scipy.sparse.linalg.eigs uses random initialization and is
        # non-deterministic across calls; power iteration with a fixed
        # initial vector guarantees byte-identical results.
        ev = self._power_iteration(B, num_de)

        # Normalize eigenvector
        ev_norm = np.linalg.norm(ev)
        if ev_norm < 1e-15:
            return {
                "variable_pressure": np.zeros(n, dtype=np.float64),
                "edge_pressure": np.zeros(num_edges, dtype=np.float64),
                "max_pressure": 0.0,
                "mean_pressure": 0.0,
            }
        ev = ev / ev_norm

        # Aggregate directed-edge weights into undirected edge pressure.
        # Each undirected edge (ci, vi) has two directed edges:
        #   (vi, n+ci) and (n+ci, vi).
        edge_pressure = np.zeros(num_edges, dtype=np.float64)
        for idx, (ci, vi) in enumerate(edges):
            j1 = edge_index.get((vi, n + ci))
            j2 = edge_index.get((n + ci, vi))
            p = 0.0
            if j1 is not None:
                p += ev[j1]
            if j2 is not None:
                p += ev[j2]
            edge_pressure[idx] = round(p, _ROUND)

        # Aggregate edge pressure into variable pressure
        variable_pressure = np.zeros(n, dtype=np.float64)
        for idx, (ci, vi) in enumerate(edges):
            variable_pressure[vi] += edge_pressure[idx]

        # Normalize variable pressure
        vp_max = variable_pressure.max()
        if vp_max > 1e-15:
            variable_pressure = variable_pressure / vp_max

        # Round for determinism
        variable_pressure = np.array(
            [round(v, _ROUND) for v in variable_pressure],
            dtype=np.float64,
        )

        max_pressure = float(round(variable_pressure.max(), _ROUND))
        mean_pressure = float(round(variable_pressure.mean(), _ROUND))

        return {
            "variable_pressure": variable_pressure,
            "edge_pressure": edge_pressure,
            "max_pressure": max_pressure,
            "mean_pressure": mean_pressure,
        }

    def _power_iteration(
        self, B: scipy.sparse.csr_matrix, size: int,
    ) -> np.ndarray:
        """Fallback power iteration for leading eigenvector magnitude."""
        x = np.ones(size, dtype=np.float64)
        x /= np.linalg.norm(x)
        for _ in range(self.power_iterations):
            y = B.dot(x)
            norm_y = np.linalg.norm(y)
            if norm_y < 1e-15:
                return np.abs(x)
            x = y / norm_y
        return np.abs(x)
