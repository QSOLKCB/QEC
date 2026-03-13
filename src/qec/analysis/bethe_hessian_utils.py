"""Utilities for deterministic Bethe-Hessian construction and caching."""

from __future__ import annotations

import numpy as np
import scipy.sparse


def _as_float64_adjacency(A: np.ndarray | scipy.sparse.spmatrix) -> np.ndarray | scipy.sparse.spmatrix:
    if scipy.sparse.issparse(A):
        return scipy.sparse.csr_matrix(A, dtype=np.float64)
    return np.asarray(A, dtype=np.float64)


def _degree_diagonal(A: np.ndarray | scipy.sparse.spmatrix) -> np.ndarray | scipy.sparse.spmatrix:
    degrees = np.asarray(A.sum(axis=1), dtype=np.float64).ravel()
    n = int(A.shape[0])
    if scipy.sparse.issparse(A):
        return scipy.sparse.diags(degrees, dtype=np.float64, format="csr")
    return np.diag(degrees).reshape((n, n))


def _identity_like(A: np.ndarray | scipy.sparse.spmatrix) -> np.ndarray | scipy.sparse.spmatrix:
    n = int(A.shape[0])
    if scipy.sparse.issparse(A):
        return scipy.sparse.eye(n, dtype=np.float64, format="csr")
    return np.eye(n, dtype=np.float64)


def build_bethe_hessian(
    A: np.ndarray | scipy.sparse.spmatrix,
    r: float,
) -> tuple[np.ndarray | scipy.sparse.spmatrix, np.ndarray | scipy.sparse.spmatrix, np.ndarray | scipy.sparse.spmatrix]:
    """Construct Bethe-Hessian matrix: H(r) = (r^2 - 1)I - rA + D."""
    A64 = _as_float64_adjacency(A)
    I = _identity_like(A64)
    D = _degree_diagonal(A64)
    r64 = float(r)
    H = ((r64 * r64 - 1.0) * I - r64 * A64 + D)
    if scipy.sparse.issparse(H):
        H = H.tocsr()
    return H, D, A64


class BetheHessianCache:
    """Cache reusable Bethe-Hessian construction components."""

    def __init__(self, A: np.ndarray | scipy.sparse.spmatrix, r: float):
        self.A = _as_float64_adjacency(A)
        self.r = float(r)
        self.r2 = self.r * self.r
        self.I = _identity_like(self.A)
        self.D = _degree_diagonal(self.A)

    def build(self) -> np.ndarray | scipy.sparse.spmatrix:
        H = ((self.r2 - 1.0) * self.I - self.r * self.A + self.D)
        if scipy.sparse.issparse(H):
            return H.tocsr()
        return H

    def update_for_swap(self, ci: int, vi: int, cj: int, vj: int) -> np.ndarray | scipy.sparse.spmatrix:
        """Apply a deterministic swap update and rebuild H from cached components."""
        if scipy.sparse.issparse(self.A):
            A_work = self.A.tolil(copy=True)
        else:
            A_work = np.array(self.A, dtype=np.float64, copy=True)

        self._set_undirected_edge(A_work, int(ci), int(vi), 0.0)
        self._set_undirected_edge(A_work, int(cj), int(vj), 0.0)
        self._set_undirected_edge(A_work, int(ci), int(vj), 1.0)
        self._set_undirected_edge(A_work, int(cj), int(vi), 1.0)

        if scipy.sparse.issparse(A_work):
            self.A = A_work.tocsr()
        else:
            self.A = np.asarray(A_work, dtype=np.float64)

        self.D = _degree_diagonal(self.A)
        return self.build()

    @staticmethod
    def _set_undirected_edge(A: np.ndarray | scipy.sparse.spmatrix, i: int, j: int, value: float) -> None:
        n = int(A.shape[0])
        if i < 0 or i >= n or j < 0 or j >= n:
            return
        if i == j:
            if scipy.sparse.issparse(A):
                A[i, j] = 0.0
            else:
                A[i, j] = 0.0
            return
        if scipy.sparse.issparse(A):
            A[i, j] = value
            A[j, i] = value
        else:
            A[i, j] = value
            A[j, i] = value
