"""
v13.1.0 — First-order NB perturbation scorer.

Deterministically estimates the dominant non-backtracking eigenvalue change
under a valid degree-preserving 2-edge Tanner swap.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse


_ROUND = 12
_EPS = 1e-12


class NBPerturbationScorer:
    """First-order Hashimoto perturbation scorer for candidate rewires."""

    def baseline_state(self, H: np.ndarray | scipy.sparse.spmatrix) -> dict[str, Any]:
        H_arr = self._to_dense_copy(H)
        B, directed_edges, out_map = self._build_nb_operator(H_arr)
        n_de = len(directed_edges)
        if n_de == 0 or B.nnz == 0:
            return {
                "tracked_eigenvalue": 0.0,
                "right_eigenvector": np.zeros(n_de, dtype=np.float64),
                "left_eigenvector": np.zeros(n_de, dtype=np.float64),
                "directed_edges": directed_edges,
                "edge_index": {},
                "out_map": out_map,
                "valid_first_order": False,
            }

        u_vec = self._power_iteration(B, n_de)
        v_vec = self._power_iteration(B.transpose().tocsr(), n_de)

        norm_u = float(np.linalg.norm(u_vec))
        norm_v = float(np.linalg.norm(v_vec))
        if norm_u <= _EPS or norm_v <= _EPS:
            return {
                "tracked_eigenvalue": 0.0,
                "right_eigenvector": np.asarray(u_vec, dtype=np.float64),
                "left_eigenvector": np.asarray(v_vec, dtype=np.float64),
                "directed_edges": directed_edges,
                "edge_index": {e: i for i, e in enumerate(directed_edges)},
                "out_map": out_map,
                "valid_first_order": False,
            }

        # L2 normalization required for stable first-order FOHPE scaling.
        u_vec = u_vec / norm_u
        v_vec = v_vec / norm_v

        Bu = B.dot(u_vec)
        denom_r = float(np.dot(u_vec, u_vec))
        lambda_right = float(np.dot(u_vec, Bu) / denom_r) if denom_r > _EPS else 0.0
        Btv = B.transpose().tocsr().dot(v_vec)
        denom_l = float(np.dot(v_vec, v_vec))
        lambda_left = float(np.dot(v_vec, Btv) / denom_l) if denom_l > _EPS else 0.0

        if abs(lambda_right - lambda_left) > 1e-6 * max(1.0, abs(lambda_right)):
            return {
                "tracked_eigenvalue": round(float(lambda_right), _ROUND),
                "right_eigenvector": np.asarray(u_vec, dtype=np.complex128),
                "left_eigenvector": np.asarray(v_vec, dtype=np.complex128),
                "directed_edges": directed_edges,
                "edge_index": {e: i for i, e in enumerate(directed_edges)},
                "out_map": out_map,
                "valid_first_order": False,
            }

        u_vec = np.asarray(u_vec, dtype=np.complex128)
        v_vec = np.asarray(v_vec, dtype=np.complex128)
        valid = bool(np.linalg.norm(u_vec) > _EPS and np.linalg.norm(v_vec) > _EPS)

        return {
            "tracked_eigenvalue": round(float(lambda_right), _ROUND),
            "right_eigenvector": u_vec,
            "left_eigenvector": v_vec,
            "directed_edges": directed_edges,
            "edge_index": {e: i for i, e in enumerate(directed_edges)},
            "out_map": out_map,
            "valid_first_order": bool(valid),
        }

    def score_swap(
        self,
        H: np.ndarray | scipy.sparse.spmatrix,
        swap: tuple[int, int, int, int],
        baseline_state: dict[str, Any],
    ) -> dict[str, Any] | None:
        H_arr = self._to_dense_copy(H)
        ci, vi, cj, vj = swap

        if not self._is_valid_swap(H_arr, ci, vi, cj, vj):
            return None

        if not bool(baseline_state.get("valid_first_order", False)):
            return None

        edge_index = baseline_state.get("edge_index")
        out_map_before = baseline_state.get("out_map")
        directed_edges = baseline_state.get("directed_edges", [])
        u_vec = baseline_state.get("right_eigenvector")
        v_vec = baseline_state.get("left_eigenvector")

        if (
            not isinstance(edge_index, dict)
            or not isinstance(out_map_before, dict)
            or u_vec is None
            or v_vec is None
        ):
            return None

        H_mut = H_arr.copy()
        H_mut[ci, vi] = 0.0
        H_mut[cj, vj] = 0.0
        H_mut[ci, vj] = 1.0
        H_mut[cj, vi] = 1.0
        _, _, out_map_after = self._build_nb_operator(H_mut)

        affected = {
            (vi, H_arr.shape[1] + ci),
            (H_arr.shape[1] + ci, vi),
            (vj, H_arr.shape[1] + cj),
            (H_arr.shape[1] + cj, vj),
            (vj, H_arr.shape[1] + ci),
            (H_arr.shape[1] + ci, vj),
            (vi, H_arr.shape[1] + cj),
            (H_arr.shape[1] + cj, vi),
        }

        delta = 0.0 + 0.0j
        nnz = 0
        for edge in sorted(affected):
            col_idx = edge_index.get(edge)
            if col_idx is None:
                continue
            before_rows = out_map_before.get(edge, ())
            after_rows = out_map_after.get(edge, ())
            if not isinstance(before_rows, tuple) or not isinstance(after_rows, tuple):
                return None
            removed = sorted(set(before_rows) - set(after_rows))
            added = sorted(set(after_rows) - set(before_rows))
            for row_edge in removed:
                row_idx = edge_index.get(row_edge)
                if row_idx is None:
                    continue
                delta -= v_vec[row_idx] * u_vec[col_idx]
                nnz += 1
            for row_edge in added:
                row_idx = edge_index.get(row_edge)
                if row_idx is None:
                    continue
                delta += v_vec[row_idx] * u_vec[col_idx]
                nnz += 1

        if nnz <= 0:
            return None

        tracked = float(baseline_state.get("tracked_eigenvalue", 0.0))
        predicted_delta = round(float(np.real(delta)), _ROUND)

        i = edge_index.get((vi, H_arr.shape[1] + ci))
        j = edge_index.get((H_arr.shape[1] + cj, vj))
        pressure_weight = 0.0
        if i is not None and j is not None:
            pressure_weight = round(float(abs(u_vec[i]) * abs(v_vec[j])), _ROUND)
        weighted_delta = round(predicted_delta * (1.0 + pressure_weight), _ROUND)

        return {
            "tracked_eigenvalue": round(tracked, _ROUND),
            "predicted_delta": predicted_delta,
            "weighted_delta": weighted_delta,
            "pressure_weight": pressure_weight,
            "predicted_new_value": round(tracked + predicted_delta, _ROUND),
            "perturbation_nnz": int(nnz),
            "valid_first_order": True,
            "swap": swap,
            "directed_edges_count": len(directed_edges),
        }

    @staticmethod
    def _to_dense_copy(H: np.ndarray | scipy.sparse.spmatrix) -> np.ndarray:
        if scipy.sparse.issparse(H):
            return np.asarray(H.todense(), dtype=np.float64)
        return np.asarray(H, dtype=np.float64).copy()

    @staticmethod
    def _is_valid_swap(H: np.ndarray, ci: int, vi: int, cj: int, vj: int) -> bool:
        if ci == cj or vi == vj:
            return False
        if H[ci, vi] == 0.0 or H[cj, vj] == 0.0:
            return False
        if H[ci, vj] != 0.0 or H[cj, vi] != 0.0:
            return False
        return True

    @staticmethod
    def _build_nb_operator(
        H: np.ndarray,
    ) -> tuple[scipy.sparse.csr_matrix, list[tuple[int, int]], dict[tuple[int, int], tuple[tuple[int, int], ...]]]:
        m, n = H.shape
        edges: list[tuple[int, int]] = []
        for ci in range(m):
            for vi in range(n):
                if H[ci, vi] != 0.0:
                    edges.append((ci, vi))
        edges.sort()

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
        rows: list[int] = []
        cols: list[int] = []
        out_map: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
        for edge in directed_edges:
            u, v = edge
            next_edges: list[tuple[int, int]] = []
            for w in adj.get(v, []):
                if w == u:
                    continue
                e2 = (v, w)
                if e2 in edge_index:
                    next_edges.append(e2)
            next_edges.sort()
            out_map[edge] = tuple(next_edges)
            i = edge_index[edge]
            for e2 in next_edges:
                j = edge_index[e2]
                rows.append(j)
                cols.append(i)

        B = scipy.sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.float64), (rows, cols)),
            shape=(num_de, num_de),
        )
        return B, directed_edges, out_map

    @staticmethod
    def _power_iteration(B: scipy.sparse.csr_matrix, size: int) -> np.ndarray:
        x = np.ones(size, dtype=np.float64)
        norm = float(np.linalg.norm(x))
        if norm <= _EPS:
            return x
        x = x / norm
        for _ in range(50):
            y = B.dot(x)
            ny = float(np.linalg.norm(y))
            if ny <= _EPS:
                return np.abs(x)
            x = y / ny
        return np.abs(x)
