"""Deterministic fast Bethe-Hessian builders for repeated inertia evaluation."""

from __future__ import annotations

import numpy as np
import scipy.sparse


class BetheHessianBuilder:
    """Cache Bethe-Hessian structural terms for deterministic rebuilds."""

    def __init__(self, A: np.ndarray | scipy.sparse.spmatrix, r: float):
        A_arr = np.asarray(scipy.sparse.csr_matrix(A, dtype=np.float64).toarray(), dtype=np.float64)
        if A_arr.ndim != 2 or A_arr.shape[0] != A_arr.shape[1]:
            raise ValueError("A must be a square adjacency matrix")
        self.A = A_arr
        self.r = float(r)
        self.r2 = float(self.r * self.r)
        self.I = np.eye(A_arr.shape[0], dtype=np.float64)

    def degree_matrix(self, A: np.ndarray | None = None) -> np.ndarray:
        A_eff = self.A if A is None else np.asarray(A, dtype=np.float64)
        return np.diag(np.sum(A_eff, axis=1, dtype=np.float64)).astype(np.float64, copy=False)

    def build(self) -> np.ndarray:
        return (self.r2 - 1.0) * self.I - self.r * self.A + self.degree_matrix()

    def build_after_swap(self, ci: int, vi: int, cj: int, vj: int) -> np.ndarray:
        A_trial = self.A.copy()

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

        return (self.r2 - 1.0) * self.I - self.r * A_trial + self.degree_matrix(A_trial)
