"""
v13.2.0 — Bethe Hessian Stability Analyzer.

Deterministic Bethe Hessian diagnostics for Tanner-graph instability
boundaries and Nishimori-temperature root estimation.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse
import scipy.sparse.linalg


_ROUND = 12


class BetheHessianAnalyzer:
    """Estimate BP stability via the Bethe Hessian spectrum."""

    @staticmethod
    def _build_variable_adjacency(H: np.ndarray) -> scipy.sparse.csr_matrix:
        H_sparse = scipy.sparse.csr_matrix(np.asarray(H, dtype=np.float64))
        HtH_sparse = H_sparse.T.dot(H_sparse).tocsr()
        HtH_sparse.setdiag(0.0)
        HtH_sparse.eliminate_zeros()
        A_sparse = HtH_sparse.copy()
        if A_sparse.nnz > 0:
            A_sparse.data[:] = np.where(A_sparse.data > 0.0, 1.0, 0.0)
            A_sparse.eliminate_zeros()
        return A_sparse

    @staticmethod
    def _lambda_min_from_adjacency(A_sparse: scipy.sparse.csr_matrix, r: float) -> float:
        n = int(A_sparse.shape[0])
        if n == 0:
            return 0.0
        degrees = np.asarray(A_sparse.sum(axis=1), dtype=np.float64).ravel()
        I_sparse = scipy.sparse.eye(n, dtype=np.float64, format="csr")
        D_sparse = scipy.sparse.diags(degrees, dtype=np.float64, format="csr")
        H_B_sparse = ((r * r - 1.0) * I_sparse - r * A_sparse + D_sparse).tocsr()

        if n == 1:
            return round(float(H_B_sparse[0, 0]), _ROUND)

        try:
            eigvals = scipy.sparse.linalg.eigsh(
                H_B_sparse,
                k=1,
                which="SA",
                return_eigenvectors=False,
                v0=np.ones(n, dtype=np.float64),
                tol=0.0,
                maxiter=max(5 * n, 100),
            )
            val = float(eigvals[0])
        except (scipy.sparse.linalg.ArpackNoConvergence, RuntimeError, ValueError):
            val = float(np.linalg.eigvalsh(H_B_sparse.toarray())[0])
        return round(val, _ROUND)

    def smallest_eigenvalue(self, H: np.ndarray, r: float) -> float:
        A_sparse = self._build_variable_adjacency(H)
        return self._lambda_min_from_adjacency(A_sparse=A_sparse, r=float(r))

    def compute_bethe_hessian_stability(self, H: np.ndarray) -> dict[str, float]:
        H_arr = np.asarray(H, dtype=np.float64)
        if H_arr.size == 0 or float(np.sum(H_arr)) == 0.0:
            return {
                "bethe_hessian_min_eigenvalue": 0.0,
                "bethe_hessian_stability_score": 0.0,
            }

        A_sparse = self._build_variable_adjacency(H_arr)
        if A_sparse.shape[0] == 0:
            return {
                "bethe_hessian_min_eigenvalue": 0.0,
                "bethe_hessian_stability_score": 0.0,
            }

        degrees = np.asarray(A_sparse.sum(axis=1), dtype=np.float64).ravel()
        avg_degree = float(degrees.mean()) if degrees.size else 0.0
        r = 1.0 if avg_degree <= 1.0 else float(np.sqrt(avg_degree - 1.0))

        min_eigenvalue = self._lambda_min_from_adjacency(A_sparse=A_sparse, r=r)
        stability_score = round(min_eigenvalue / r, _ROUND) if r > 0.0 else min_eigenvalue

        return {
            "bethe_hessian_min_eigenvalue": min_eigenvalue,
            "bethe_hessian_stability_score": stability_score,
        }


def estimate_nishimori_temperature(H: np.ndarray) -> float:
    """Estimate r such that lambda_min(H_B(r)) ~= 0 deterministically.

    Uses quadratic interpolation on three deterministic support points,
    followed by Newton refinement with finite-difference slope.
    """
    analyzer = BetheHessianAnalyzer()
    A_sparse = analyzer._build_variable_adjacency(np.asarray(H, dtype=np.float64))
    n = int(A_sparse.shape[0])
    if n == 0:
        return 1.0

    degrees = np.asarray(A_sparse.sum(axis=1), dtype=np.float64).ravel()
    mean_degree = float(degrees.mean()) if degrees.size else 0.0
    base = 1.0 if mean_degree <= 1.0 else float(np.sqrt(mean_degree - 1.0))

    r0 = max(0.5, 0.80 * base)
    r1 = max(0.6, 1.00 * base)
    r2 = max(0.7, 1.20 * base)

    f0 = analyzer._lambda_min_from_adjacency(A_sparse, r0)
    f1 = analyzer._lambda_min_from_adjacency(A_sparse, r1)
    f2 = analyzer._lambda_min_from_adjacency(A_sparse, r2)

    coeff = np.polyfit(np.array([r0, r1, r2], dtype=np.float64), np.array([f0, f1, f2], dtype=np.float64), 2)
    roots = np.roots(coeff)
    roots_real = [float(x.real) for x in roots if abs(x.imag) < 1e-9 and x.real > 0.0]

    if roots_real:
        r = min(roots_real, key=lambda x: abs(x - r1))
    else:
        r = r1

    for _ in range(4):
        f = analyzer._lambda_min_from_adjacency(A_sparse, r)
        h = max(1e-3, 1e-3 * r)
        fp = analyzer._lambda_min_from_adjacency(A_sparse, r + h)
        fm = analyzer._lambda_min_from_adjacency(A_sparse, max(1e-6, r - h))
        slope = (fp - fm) / (2.0 * h)
        if abs(slope) < 1e-12:
            break
        r_next = r - f / slope
        if not np.isfinite(r_next) or r_next <= 0.0:
            break
        r = float(r_next)

    return round(float(r), _ROUND)
