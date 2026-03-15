"""Deterministic fast Bethe-Hessian builders for repeated inertia evaluation."""

from __future__ import annotations

import numpy as np
import scipy.sparse


class BetheHessianBuilder:
    """Cache Bethe-Hessian structural terms for deterministic rebuilds."""

    def __init__(self, A: np.ndarray | scipy.sparse.spmatrix, r: float):
        if isinstance(A, np.ndarray):
            A_arr = np.asarray(A, dtype=np.float64)
        else:
            A_arr = np.asarray(scipy.sparse.csr_matrix(A, dtype=np.float64).toarray(), dtype=np.float64)
        if A_arr.ndim != 2 or A_arr.shape[0] != A_arr.shape[1]:
            raise ValueError("A must be a square adjacency matrix")
        self.A = A_arr
        self.r = float(r)
        self.r2 = float(self.r * self.r)
        self.I = np.eye(A_arr.shape[0], dtype=np.float64)
        self.deg = np.sum(self.A, axis=1, dtype=np.float64).astype(np.float64, copy=False)
        self._A_trial_buffer = self.A.copy()

    def degree_matrix(self, deg: np.ndarray | None = None) -> np.ndarray:
        deg_eff = self.deg if deg is None else np.asarray(deg, dtype=np.float64)
        return np.diag(deg_eff).astype(np.float64, copy=False)

    def build(self) -> np.ndarray:
        H = self.I.copy()
        H *= (self.r2 - 1.0)
        H -= self.r * self.A
        H += self.degree_matrix()
        return H

    def build_after_swap(self, ci: int, vi: int, cj: int, vj: int) -> np.ndarray:
        A_trial = self._A_trial_buffer
        A_trial[:, :] = self.A

        ci_i = int(ci)
        vi_i = int(vi)
        cj_i = int(cj)
        vj_i = int(vj)

        A_trial[ci_i, vi_i] = 0.0
        A_trial[vi_i, ci_i] = 0.0
        A_trial[cj_i, vj_i] = 0.0
        A_trial[vj_i, cj_i] = 0.0

        A_trial[ci_i, vj_i] = 1.0
        A_trial[vj_i, ci_i] = 1.0
        A_trial[cj_i, vi_i] = 1.0
        A_trial[vi_i, cj_i] = 1.0

        deg_trial = self.deg.copy()
        for u, v in ((ci_i, vi_i), (cj_i, vj_i), (ci_i, vj_i), (cj_i, vi_i)):
            old_val = float(self.A[u, v])
            new_val = float(A_trial[u, v])
            delta = new_val - old_val
            if delta == 0.0:
                continue
            deg_trial[u] += delta
            if u != v:
                deg_trial[v] += delta

        H = self.I.copy()
        H *= (self.r2 - 1.0)
        H -= self.r * A_trial
        H += self.degree_matrix(deg_trial)
        return H
