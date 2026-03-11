"""
v11.2.0 — Bethe Hessian Stability Analyzer.

Estimates belief-propagation stability directly from Tanner graph structure
using the Bethe Hessian operator.

For graph adjacency matrix A and degree matrix D:

    H_B(r) = (r^2 - 1) I - r A + D

where r = sqrt(average_degree - 1).

Interpretation of the smallest eigenvalue:
    positive  → stable BP
    near zero → marginal
    negative  → unstable region

Layer 3 — Analysis.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse
import scipy.sparse.linalg


_ROUND = 12


class BetheHessianAnalyzer:
    """Estimate BP stability via the Bethe Hessian spectrum.

    The Bethe Hessian is constructed from the variable-node adjacency
    matrix of the Tanner graph.  Its smallest eigenvalue indicates
    whether BP fixed points are locally stable.
    """

    def compute_bethe_hessian_stability(
        self,
        H: np.ndarray,
    ) -> dict[str, float]:
        """Compute Bethe Hessian stability metrics.

        Parameters
        ----------
        H : np.ndarray
            Binary parity-check matrix, shape (m, n).

        Returns
        -------
        dict[str, float]
            Dictionary with keys:
            - ``bethe_hessian_min_eigenvalue`` : float
            - ``bethe_hessian_stability_score`` : float
        """
        H_arr = np.asarray(H, dtype=np.float64)
        m, n = H_arr.shape

        if m == 0 or n == 0 or H_arr.sum() == 0:
            return {
                "bethe_hessian_min_eigenvalue": 0.0,
                "bethe_hessian_stability_score": 0.0,
            }

        # Build variable-node adjacency matrix A (n x n).
        # Two variable nodes are adjacent if they share at least one check.
        # A = H^T H with diagonal zeroed gives the variable-node adjacency
        # (with weights = number of shared checks).  We binarize for the
        # standard Bethe Hessian formulation.
        HtH = H_arr.T @ H_arr
        np.fill_diagonal(HtH, 0.0)
        # Binarize: adjacency is 0 or 1
        A = (HtH > 0).astype(np.float64)

        # Node degree matrix D (diagonal)
        degrees = A.sum(axis=1)

        # Estimate r = sqrt(average_degree - 1)
        avg_degree = float(degrees.mean())
        if avg_degree <= 1.0:
            r = 1.0
        else:
            r = float(np.sqrt(avg_degree - 1.0))

        # Construct Bethe Hessian: H_B(r) = (r^2 - 1) I - r A + D
        r2_minus_1 = r * r - 1.0
        H_B = r2_minus_1 * np.eye(n, dtype=np.float64) - r * A + np.diag(degrees)

        # Compute smallest eigenvalues using sparse eigensolver
        H_B_sparse = scipy.sparse.csr_matrix(H_B)
        k = min(6, n - 1) if n > 2 else 1

        if k < 1:
            return {
                "bethe_hessian_min_eigenvalue": 0.0,
                "bethe_hessian_stability_score": 0.0,
            }

        try:
            eigenvalues = scipy.sparse.linalg.eigsh(
                H_B_sparse,
                k=k,
                which="SA",  # smallest algebraic
                return_eigenvectors=False,
            )
        except (scipy.sparse.linalg.ArpackNoConvergence, RuntimeError):
            # Fallback: dense eigensolver for small matrices
            eigenvalues = np.linalg.eigvalsh(H_B)

        min_eigenvalue = round(float(np.min(eigenvalues)), _ROUND)

        # Stability score: positive = stable, negative = unstable
        # Normalize by r to make the score scale-invariant
        if r > 0:
            stability_score = round(min_eigenvalue / r, _ROUND)
        else:
            stability_score = round(min_eigenvalue, _ROUND)

        return {
            "bethe_hessian_min_eigenvalue": min_eigenvalue,
            "bethe_hessian_stability_score": stability_score,
        }
