from __future__ import annotations

import numpy as np
import scipy.sparse
import scipy.sparse.linalg


_ROUND = 12


def _to_csr(H_pc: np.ndarray | scipy.sparse.spmatrix) -> scipy.sparse.csr_matrix:
    if scipy.sparse.issparse(H_pc):
        return H_pc.tocsr().astype(np.float64)
    return scipy.sparse.csr_matrix(np.asarray(H_pc, dtype=np.float64))


def _build_tanner_adjacency(H_pc: np.ndarray | scipy.sparse.spmatrix) -> tuple[scipy.sparse.csr_matrix, np.ndarray]:
    H = _to_csr(H_pc)
    m, n = H.shape
    z_nn = scipy.sparse.csr_matrix((n, n), dtype=np.float64)
    z_mm = scipy.sparse.csr_matrix((m, m), dtype=np.float64)
    top = scipy.sparse.hstack([z_nn, H.T], format="csr")
    bottom = scipy.sparse.hstack([H, z_mm], format="csr")
    A = scipy.sparse.vstack([top, bottom], format="csr")
    deg = np.asarray(A.sum(axis=1), dtype=np.float64).ravel()
    return A, deg


def build_bethe_hessian(
    H_pc: np.ndarray | scipy.sparse.spmatrix,
    r: float | None = None,
) -> scipy.sparse.csr_matrix:
    """Build sparse Bethe-Hessian matrix for Tanner graph bipartite adjacency."""
    A, degrees = _build_tanner_adjacency(H_pc)
    total = A.shape[0]
    if total == 0:
        return scipy.sparse.csr_matrix((0, 0), dtype=np.float64)

    if r is None:
        d_avg = float(np.mean(degrees)) if degrees.size > 0 else 0.0
        r_used = float(np.sqrt(max(d_avg - 1.0, 0.0))) if d_avg > 1.0 else 1.0
    else:
        r_used = float(r)

    I = scipy.sparse.eye(total, format="csr", dtype=np.float64)
    D = scipy.sparse.diags(degrees, format="csr")
    H_bh = (r_used * r_used - 1.0) * I - r_used * A + D
    return H_bh.tocsr()


def extract_negative_modes(
    H_BH: np.ndarray | scipy.sparse.spmatrix,
    k_max: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract negative Bethe-Hessian eigenmodes using deterministic SA solve."""
    if scipy.sparse.issparse(H_BH):
        M = H_BH.tocsr().astype(np.float64)
    else:
        M = scipy.sparse.csr_matrix(np.asarray(H_BH, dtype=np.float64))

    n = M.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64), np.zeros((0, 0), dtype=np.float64)

    if n == 1:
        val = float(M.toarray()[0, 0])
        if val < 0.0:
            return np.array([round(val, _ROUND)], dtype=np.float64), np.array([[1.0]], dtype=np.float64)
        return np.zeros(0, dtype=np.float64), np.zeros((1, 0), dtype=np.float64)

    k = max(1, min(int(k_max), n - 1))
    v0 = np.ones(n, dtype=np.float64)
    try:
        evals, evecs = scipy.sparse.linalg.eigsh(M, k=k, which="SA", v0=v0)
    except (scipy.sparse.linalg.ArpackNoConvergence, RuntimeError):
        dense = np.asarray(M.toarray(), dtype=np.float64)
        all_vals, all_vecs = np.linalg.eigh(dense)
        idx = np.argsort(all_vals, kind="mergesort")
        evals = all_vals[idx][:k]
        evecs = all_vecs[:, idx][:, :k]

    order = np.argsort(evals, kind="mergesort")
    evals = np.asarray(evals[order], dtype=np.float64)
    evecs = np.asarray(evecs[:, order], dtype=np.float64)

    neg_mask = evals < 0.0
    neg_vals = np.round(evals[neg_mask], _ROUND).astype(np.float64)
    neg_vecs = np.asarray(evecs[:, neg_mask], dtype=np.float64)
    return neg_vals, neg_vecs


class BetheHessianAnalyzer:
    """Backward-compatible v11.2 analyzer API."""

    def compute_bethe_hessian_stability(self, H: np.ndarray) -> dict[str, float]:
        H_arr = np.asarray(H, dtype=np.float64)
        m, n = H_arr.shape
        if m == 0 or n == 0 or H_arr.sum() == 0:
            return {"bethe_hessian_min_eigenvalue": 0.0, "bethe_hessian_stability_score": 0.0}

        H_bh = build_bethe_hessian(H_arr)
        evals, _ = extract_negative_modes(H_bh, k_max=min(6, max(1, H_bh.shape[0] - 1)))
        if evals.size > 0:
            min_eigenvalue = float(evals[0])
        else:
            # need smallest algebraic even if non-negative
            try:
                val = scipy.sparse.linalg.eigsh(H_bh, k=1, which="SA", return_eigenvectors=False, v0=np.ones(H_bh.shape[0], dtype=np.float64))
                min_eigenvalue = float(np.asarray(val, dtype=np.float64)[0])
            except (scipy.sparse.linalg.ArpackNoConvergence, RuntimeError):
                min_eigenvalue = float(np.min(np.linalg.eigvalsh(H_bh.toarray())))
            min_eigenvalue = round(min_eigenvalue, _ROUND)

        _, degrees = _build_tanner_adjacency(H_arr)
        avg_degree = float(np.mean(degrees)) if degrees.size else 0.0
        r = float(np.sqrt(avg_degree - 1.0)) if avg_degree > 1.0 else 1.0
        stability_score = round(min_eigenvalue / r, _ROUND) if r > 0 else round(min_eigenvalue, _ROUND)
        return {
            "bethe_hessian_min_eigenvalue": round(float(min_eigenvalue), _ROUND),
            "bethe_hessian_stability_score": stability_score,
        }
