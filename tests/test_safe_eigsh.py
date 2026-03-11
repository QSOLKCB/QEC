"""
Tests for safe_eigsh and safe_eigs wrappers (v11.2.1 Spectral Stability Patch).

Verifies that ARPACK fallback handling works correctly for:
- small matrices (2x2, 3x3, identity)
- singular matrices (zero, rank-deficient)
- sparse matrices
- determinism across repeated runs
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from src.qec.diagnostics.spectral_metrics import safe_eigsh, safe_eigs


# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------


def _dense_eigvalsh_sorted(A, k, which="LM"):
    """Reference eigenvalues via dense symmetric solver."""
    vals = np.linalg.eigvalsh(np.asarray(A.toarray() if sp.issparse(A) else A, dtype=float))
    if which == "LM":
        order = np.argsort(np.abs(vals))[::-1]
    elif which == "SM":
        order = np.argsort(np.abs(vals))
    elif which == "LA":
        order = np.argsort(vals)[::-1]
    elif which == "SA":
        order = np.argsort(vals)
    else:
        order = np.argsort(np.abs(vals))[::-1]
    return vals[order[:k]]


# -----------------------------------------------------------
# Small matrix tests — safe_eigsh
# -----------------------------------------------------------


class TestSafeEigshSmallMatrices:
    """safe_eigsh must handle small matrices without raising."""

    def test_2x2_identity(self):
        A = sp.csr_matrix(np.eye(2))
        vals = safe_eigsh(A, k=1, which="LM")
        assert len(vals) == 1
        np.testing.assert_allclose(np.abs(vals[0]), 1.0, atol=1e-10)

    def test_2x2_diagonal(self):
        A = sp.csr_matrix(np.diag([3.0, 1.0]))
        vals = safe_eigsh(A, k=1, which="LM")
        assert len(vals) == 1
        np.testing.assert_allclose(np.abs(vals[0]), 3.0, atol=1e-10)

    def test_3x3_identity(self):
        A = sp.csr_matrix(np.eye(3))
        vals = safe_eigsh(A, k=1, which="LM")
        assert len(vals) == 1
        np.testing.assert_allclose(np.abs(vals[0]), 1.0, atol=1e-10)

    def test_3x3_symmetric(self):
        M = np.array([[2.0, 1.0, 0.0],
                       [1.0, 3.0, 1.0],
                       [0.0, 1.0, 2.0]])
        A = sp.csr_matrix(M)
        vals = safe_eigsh(A, k=2, which="LM")
        ref = _dense_eigvalsh_sorted(A, k=2, which="LM")
        np.testing.assert_allclose(np.sort(np.abs(vals)), np.sort(np.abs(ref)), atol=1e-8)

    def test_identity_4x4(self):
        A = sp.csr_matrix(np.eye(4))
        vals = safe_eigsh(A, k=2, which="LM")
        assert len(vals) == 2
        np.testing.assert_allclose(np.abs(vals), [1.0, 1.0], atol=1e-10)


# -----------------------------------------------------------
# Singular matrix tests — safe_eigsh
# -----------------------------------------------------------


class TestSafeEigshSingular:
    """safe_eigsh must not raise on singular or degenerate matrices."""

    def test_zero_matrix_2x2(self):
        A = sp.csr_matrix((2, 2))
        vals = safe_eigsh(A, k=1, which="LM")
        assert len(vals) == 1
        np.testing.assert_allclose(np.abs(vals[0]), 0.0, atol=1e-10)

    def test_zero_matrix_5x5(self):
        A = sp.csr_matrix((5, 5))
        vals = safe_eigsh(A, k=2, which="LM")
        assert len(vals) == 2
        np.testing.assert_allclose(np.abs(vals), [0.0, 0.0], atol=1e-10)

    def test_rank_deficient(self):
        """Rank-1 symmetric matrix."""
        v = np.array([1.0, 2.0, 3.0, 4.0])
        M = np.outer(v, v)  # rank 1, symmetric
        A = sp.csr_matrix(M)
        vals = safe_eigsh(A, k=2, which="LM")
        assert len(vals) == 2
        # One eigenvalue should be ~30, rest ~0
        assert np.max(np.abs(vals)) > 1.0

    def test_near_singular(self):
        """Nearly singular symmetric matrix."""
        M = np.eye(4) * 1e-15
        M[0, 0] = 1.0
        A = sp.csr_matrix(M)
        vals = safe_eigsh(A, k=1, which="LM")
        assert len(vals) == 1
        np.testing.assert_allclose(np.abs(vals[0]), 1.0, atol=1e-8)


# -----------------------------------------------------------
# Sparse matrix tests — safe_eigsh
# -----------------------------------------------------------


class TestSafeEigshSparse:
    """safe_eigsh on larger sparse matrices."""

    def test_sparse_adjacency(self):
        """Random sparse symmetric adjacency matrix."""
        rng = np.random.default_rng(42)
        n = 20
        density = 0.3
        M = sp.random(n, n, density=density, random_state=rng, format="csr")
        # Make symmetric
        M = (M + M.T) / 2.0
        vals = safe_eigsh(M, k=3, which="LM")
        assert len(vals) == 3
        # Verify against dense
        ref = _dense_eigvalsh_sorted(M, k=3, which="LM")
        np.testing.assert_allclose(
            np.sort(np.abs(vals))[::-1],
            np.sort(np.abs(ref))[::-1],
            atol=1e-6,
        )

    def test_sparse_laplacian(self):
        """Graph Laplacian — always symmetric positive semi-definite."""
        n = 10
        rng = np.random.default_rng(99)
        A = sp.random(n, n, density=0.4, random_state=rng, format="csr")
        A = (A + A.T) / 2.0
        A.data[:] = 1.0  # unweighted
        D = sp.diags(np.array(A.sum(axis=1)).ravel())
        L = D - A
        vals = safe_eigsh(L, k=2, which="SM")
        assert len(vals) == 2
        # Smallest eigenvalue of Laplacian is 0
        np.testing.assert_allclose(np.min(np.abs(vals)), 0.0, atol=1e-8)


# -----------------------------------------------------------
# safe_eigs tests (non-symmetric)
# -----------------------------------------------------------


class TestSafeEigs:
    """safe_eigs must handle small and degenerate non-symmetric matrices."""

    def test_2x2(self):
        A = sp.csr_matrix(np.array([[0.0, 1.0], [-1.0, 0.0]]))
        vals, vecs = safe_eigs(A, k=1, which="LM")
        assert len(vals) == 1
        # Eigenvalues of [[0,1],[-1,0]] are ±i, magnitude 1
        np.testing.assert_allclose(np.abs(vals[0]), 1.0, atol=1e-10)

    def test_3x3_non_symmetric(self):
        M = np.array([[1.0, 2.0, 0.0],
                       [0.0, 3.0, 1.0],
                       [0.0, 0.0, 2.0]])
        A = sp.csr_matrix(M)
        vals, vecs = safe_eigs(A, k=1, which="LM")
        assert len(vals) == 1
        np.testing.assert_allclose(np.abs(vals[0]), 3.0, atol=1e-10)

    def test_zero_matrix(self):
        A = sp.csr_matrix((3, 3))
        vals, vecs = safe_eigs(A, k=1, which="LM")
        assert len(vals) == 1
        np.testing.assert_allclose(np.abs(vals[0]), 0.0, atol=1e-10)


# -----------------------------------------------------------
# Determinism tests
# -----------------------------------------------------------


class TestDeterminism:
    """Results must be identical across repeated calls."""

    def test_safe_eigsh_deterministic(self):
        rng = np.random.default_rng(7)
        M = sp.random(15, 15, density=0.3, random_state=rng, format="csr")
        M = (M + M.T) / 2.0

        vals1 = safe_eigsh(M, k=2, which="LM")
        vals2 = safe_eigsh(M, k=2, which="LM")
        # ARPACK iterative solvers may have ULP-level variation;
        # confirm results are within machine epsilon.
        np.testing.assert_allclose(vals1, vals2, atol=1e-12, rtol=1e-12)

    def test_safe_eigs_deterministic(self):
        rng = np.random.default_rng(13)
        M = sp.random(15, 15, density=0.3, random_state=rng, format="csr")

        vals1, _ = safe_eigs(M, k=2, which="LM")
        vals2, _ = safe_eigs(M, k=2, which="LM")
        np.testing.assert_allclose(vals1, vals2, atol=1e-12, rtol=1e-12)

    def test_safe_eigsh_repeated_10x(self):
        """Run 10 times and verify all results identical."""
        A = sp.csr_matrix(np.diag([5.0, 3.0, 1.0]))
        results = [safe_eigsh(A, k=2, which="LM") for _ in range(10)]
        for r in results[1:]:
            np.testing.assert_array_equal(results[0], r)
