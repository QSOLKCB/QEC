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

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from src.qec.analysis.eigenvector_localization import EigenvectorLocalizationAnalyzer
from src.qec.analysis.flow_alignment import FlowAlignmentAnalyzer


_POWER_ITER = 50
_ROUND = 12


@dataclass(frozen=True)
class NBFlowConfig:
    """Configuration for deterministic v14.2.0 NB eigenvector flow."""

    num_nb_eigenvalues: int = 16
    mode_weight_beta: float = 1.0
    alpha_loc: float = 0.1
    canonical_phase: bool = True
    use_left_right_pairing: bool = True
    bulk_radius_mode: str = "auto"
    precision: int = _ROUND


@dataclass(frozen=True)
class NBMode:
    """Selected non-backtracking mode."""

    eigenvalue: complex
    right: np.ndarray
    left: np.ndarray | None
    ipr: float
    weight: float


@dataclass(frozen=True)
class EdgeFlowField:
    """Deterministic directed and undirected NB flow field."""

    directed_edges: tuple[tuple[int, int], ...]
    undirected_edges: tuple[tuple[int, int], ...]
    directed_pressure: np.ndarray
    edge_pressure: np.ndarray
    edge_pressure_map: dict[tuple[int, int], float]
    bulk_radius: float
    selected_modes: tuple[NBMode, ...]


class NonBacktrackingFlowAnalyzer:
    """Compute instability flow via non-backtracking eigenvector.

    Parameters
    ----------
    power_iterations : int
        Number of power iterations for the eigensolver (default 50).
    """

    def __init__(self, power_iterations: int = _POWER_ITER) -> None:
        self.power_iterations = power_iterations

    def compute_flow(
        self,
        H: np.ndarray | scipy.sparse.spmatrix,
        residual_map: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Compute non-backtracking flow for variables and edges.

        Parameters
        ----------
        H : np.ndarray or scipy.sparse.spmatrix
            Binary parity-check matrix, shape (m, n).
        residual_map : np.ndarray or None
            Optional BP residual magnitude per variable node, shape (n,).
            When provided, flow alignment diagnostics are included.

        Returns
        -------
        dict
            directed_edges : list[tuple[int, int]], sorted directed edge list
            directed_edge_flow : np.ndarray, shape (2*num_edges,)
            edge_flow : np.ndarray, shape (num_edges,)
            variable_flow : np.ndarray, shape (n,)
            check_flow : np.ndarray, shape (m,)
            max_flow : float
            mean_flow : float
            flow_localization : float
            flow_alignment : dict (only when residual_map is provided)
        """
        if scipy.sparse.issparse(H):
            H_csr = H.tocsr()
            m, n = H_csr.shape
        else:
            H_arr = np.asarray(H, dtype=np.float64)
            m, n = H_arr.shape

        empty = {
            "directed_edges": [],
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
                "directed_edges": directed_edges,
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

        nb_localization = EigenvectorLocalizationAnalyzer.compute_ipr(
            variable_flow,
        )

        result = {
            "directed_edges": directed_edges,
            "directed_edge_flow": directed_edge_flow,
            "edge_flow": edge_flow,
            "variable_flow": variable_flow,
            "check_flow": check_flow,
            "max_flow": max_flow,
            "mean_flow": mean_flow,
            "flow_localization": flow_localization,
            "nb_localization": nb_localization,
        }

        if residual_map is not None:
            alignment_analyzer = FlowAlignmentAnalyzer()
            result["flow_alignment"] = alignment_analyzer.compute_alignment(
                residual_map=residual_map,
                variable_flow=variable_flow,
            )

        return result

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



def canonical_directed_edges(
    H: np.ndarray | scipy.sparse.spmatrix,
) -> tuple[
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
    dict[tuple[int, int], int],
    dict[int, tuple[int, ...]],
    int,
    int,
]:
    """Return deterministic undirected/directed Tanner edge ordering."""
    H_csr = scipy.sparse.csr_matrix(H, dtype=np.float64)
    m, n = H_csr.shape
    undirected: list[tuple[int, int]] = []
    for ci in range(m):
        row = H_csr.indices[H_csr.indptr[ci]:H_csr.indptr[ci + 1]]
        for vi in row:
            undirected.append((ci, int(vi)))
    undirected.sort()

    directed: list[tuple[int, int]] = []
    adj: dict[int, list[int]] = {}
    for ci, vi in undirected:
        u = int(vi)
        c = int(n + ci)
        directed.append((u, c))
        directed.append((c, u))
        adj.setdefault(u, []).append(c)
        adj.setdefault(c, []).append(u)

    for node, nbrs in list(adj.items()):
        adj[node] = sorted(nbrs)
    directed.sort()
    index = {edge: i for i, edge in enumerate(directed)}
    return tuple(undirected), tuple(directed), index, {k: tuple(v) for k, v in adj.items()}, m, n


def normalize_mode_phase(vec: np.ndarray) -> np.ndarray:
    """Deterministically fix sign/phase by dominant component."""
    out = np.asarray(vec, dtype=np.complex128).copy()
    if out.size == 0:
        return out
    magnitudes = np.abs(out)
    max_mag = float(np.max(magnitudes))
    if max_mag <= 1e-15:
        return out
    idxs = np.flatnonzero(np.isclose(magnitudes, max_mag, atol=0.0, rtol=0.0))
    pivot = int(idxs[0])
    ref = out[pivot]
    if abs(ref) <= 1e-15:
        return out
    phase = np.angle(ref)
    out = out * np.exp(-1j * phase)
    if out[pivot].real < 0.0:
        out = -out
    return out




def project_directed_pressure_to_undirected(
    *,
    undirected_edges: tuple[tuple[int, int], ...],
    directed_index: dict[tuple[int, int], int],
    n: int,
    directed_pressure: np.ndarray,
    precision: int = _ROUND,
) -> tuple[np.ndarray, dict[tuple[int, int], float]]:
    """Project directed-edge pressure p_(u->v) to Tanner undirected edges."""
    edge_pressure = np.zeros(len(undirected_edges), dtype=np.float64)
    edge_pressure_map: dict[tuple[int, int], float] = {}
    for idx, (ci, vi) in enumerate(undirected_edges):
        a = directed_index[(vi, n + ci)]
        b = directed_index[(n + ci, vi)]
        val = round(float(directed_pressure[a] + directed_pressure[b]), precision)
        edge_pressure[idx] = val
        edge_pressure_map[(ci, vi)] = val
    return edge_pressure, edge_pressure_map


class NonBacktrackingEigenvectorFlowAnalyzer:
    """v14.2.0 deterministic NB eigenvector flow on directed Tanner edges."""

    def __init__(self, config: NBFlowConfig | None = None) -> None:
        self.config = config or NBFlowConfig()

    def build_flow_field(self, H: np.ndarray | scipy.sparse.spmatrix) -> EdgeFlowField:
        undirected, directed, edge_index, adj, m, n = canonical_directed_edges(H)
        de = len(directed)
        if de == 0:
            return EdgeFlowField(
                directed_edges=directed,
                undirected_edges=undirected,
                directed_pressure=np.zeros(0, dtype=np.float64),
                edge_pressure=np.zeros(0, dtype=np.float64),
                edge_pressure_map={},
                bulk_radius=0.0,
                selected_modes=(),
            )

        rows: list[int] = []
        cols: list[int] = []
        for i, (u, v) in enumerate(directed):
            for w in adj.get(v, ()):  # deterministic order
                if w == u:
                    continue
                j = edge_index.get((v, w))
                if j is not None:
                    rows.append(j)
                    cols.append(i)
        if not rows:
            return EdgeFlowField(
                directed_edges=directed,
                undirected_edges=undirected,
                directed_pressure=np.zeros(de, dtype=np.float64),
                edge_pressure=np.zeros(len(undirected), dtype=np.float64),
                edge_pressure_map={edge: 0.0 for edge in undirected},
                bulk_radius=0.0,
                selected_modes=(),
            )

        B = scipy.sparse.csr_matrix((np.ones(len(rows), dtype=np.float64), (rows, cols)), shape=(de, de))

        k = int(min(max(1, self.config.num_nb_eigenvalues), max(1, de - 1)))
        eigvals = np.zeros(0, dtype=np.complex128)
        right = np.zeros((de, 0), dtype=np.complex128)
        left = np.zeros((de, 0), dtype=np.complex128)
        use_dense = de <= 2 or k >= de - 1
        try:
            if use_dense:
                raise ValueError('dense_fallback')
            eigvals, right = scipy.sparse.linalg.eigs(
                B,
                k=k,
                which='LR',
                v0=np.ones(de, dtype=np.float64),
                maxiter=max(100, 5 * de),
                tol=0.0,
            )
            if self.config.use_left_right_pairing:
                _, left = scipy.sparse.linalg.eigs(
                    B.transpose().tocsr(),
                    k=k,
                    which='LR',
                    v0=np.ones(de, dtype=np.float64),
                    maxiter=max(100, 5 * de),
                    tol=0.0,
                )
        except Exception:
            dense = B.toarray()
            vals, vecs = np.linalg.eig(dense)
            order = np.argsort(-np.abs(vals), kind='stable')
            order = order[:k]
            eigvals = vals[order]
            right = vecs[:, order]
            if self.config.use_left_right_pairing:
                _, left_all = np.linalg.eig(dense.T)
                left = left_all[:, order]

        order = np.argsort(-np.abs(eigvals), kind='stable')
        eigvals = eigvals[order]
        right = right[:, order]
        if left.shape[1] == right.shape[1]:
            left = left[:, order]

        bulk_radius = self._bulk_radius(H)

        directed_pressure = np.zeros(de, dtype=np.float64)
        modes: list[NBMode] = []
        for idx in range(right.shape[1]):
            lam = eigvals[idx]
            rv = right[:, idx]
            lv = left[:, idx] if left.shape[1] > idx else None
            if self.config.canonical_phase:
                rv = normalize_mode_phase(rv)
                if lv is not None:
                    lv = normalize_mode_phase(lv)
            norm = float(np.linalg.norm(rv))
            if norm <= 1e-15:
                continue
            rv = rv / norm
            ipr = float(np.sum(np.abs(rv) ** 4))
            outlier = max(0.0, float(abs(lam) - bulk_radius))
            w = (outlier ** self.config.mode_weight_beta) / (1.0 + self.config.alpha_loc * ipr)
            if w <= 0.0:
                continue

            if self.config.use_left_right_pairing and lv is not None and lv.shape == rv.shape:
                denom = np.vdot(lv, rv)
                if abs(denom) > 1e-15:
                    lv = lv / denom
                p_dir = np.real(np.conjugate(lv) * rv)
                p_dir = np.abs(p_dir)
            else:
                p_dir = np.abs(rv) ** 2
            directed_pressure += float(w) * np.asarray(p_dir, dtype=np.float64)
            modes.append(NBMode(eigenvalue=lam, right=rv, left=lv, ipr=round(ipr, self.config.precision), weight=round(float(w), self.config.precision)))

        edge_pressure, edge_pressure_map = project_directed_pressure_to_undirected(
            undirected_edges=undirected,
            directed_index=edge_index,
            n=n,
            directed_pressure=directed_pressure,
            precision=self.config.precision,
        )

        return EdgeFlowField(
            directed_edges=directed,
            undirected_edges=undirected,
            directed_pressure=np.asarray([round(float(x), self.config.precision) for x in directed_pressure], dtype=np.float64),
            edge_pressure=edge_pressure,
            edge_pressure_map=edge_pressure_map,
            bulk_radius=round(float(bulk_radius), self.config.precision),
            selected_modes=tuple(modes),
        )

    @staticmethod
    def _bulk_radius(H: np.ndarray | scipy.sparse.spmatrix) -> float:
        H_csr = scipy.sparse.csr_matrix(H, dtype=np.float64)
        if H_csr.nnz == 0:
            return 0.0
        row_deg = np.diff(H_csr.indptr).astype(np.float64)
        col_deg = np.asarray(H_csr.sum(axis=0), dtype=np.float64).ravel()
        c = float(np.mean(np.maximum(row_deg - 1.0, 0.0))) if row_deg.size else 0.0
        v = float(np.mean(np.maximum(col_deg - 1.0, 0.0))) if col_deg.size else 0.0
        return float(np.sqrt(max(c * v, 0.0)))
