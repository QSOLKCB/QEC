"""Construction invariant tests for the lifted CSS code.

Every test here validates a structural invariant that must hold for
any correctly-constructed QLDPC CSS code.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from qldpc.css_code import CSSCode
from qldpc.invariants import (
    ConstructionInvariantError,
    check_css_orthogonality,
    check_no_zero_rows_or_cols,
)
from qldpc.lift import SharedLiftTable, lift_matrix


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
SMALL_PARAMS = dict(r=3, c=3, L=7, seed=42)


@pytest.fixture
def code():
    return CSSCode(**SMALL_PARAMS)


# ---------------------------------------------------------------
# 1. Deterministic seed test
# ---------------------------------------------------------------
class TestDeterminism:
    def test_same_seed_gives_identical_matrices(self):
        c1 = CSSCode(**SMALL_PARAMS)
        c2 = CSSCode(**SMALL_PARAMS)
        # HX and HZ must be bit-for-bit identical.
        assert (c1.HX - c2.HX).nnz == 0
        assert (c1.HZ - c2.HZ).nnz == 0

    def test_different_seed_gives_different_matrices(self):
        c1 = CSSCode(r=3, c=3, L=7, seed=42)
        c2 = CSSCode(r=3, c=3, L=7, seed=99)
        # With overwhelming probability the matrices differ.
        assert (c1.HX - c2.HX).nnz > 0 or (c1.HZ - c2.HZ).nnz > 0


# ---------------------------------------------------------------
# 2. Shared circulant reuse test
# ---------------------------------------------------------------
class TestSharedLift:
    def test_lift_table_is_shared(self, code: CSSCode):
        """Both H_X and H_Z use the *same* SharedLiftTable instance."""
        # By construction, code.lift_table is used for both.
        # We verify by re-lifting with the same table and comparing.
        HX2 = lift_matrix(code.base_X, code.lift_table)
        HZ2 = lift_matrix(code.base_Z, code.lift_table)
        assert (code.HX - HX2).nnz == 0
        assert (code.HZ - HZ2).nnz == 0

    def test_single_lift_table_object(self, code: CSSCode):
        """The code has exactly one lift table, not two."""
        # This is a design invariant: there is only one lift_table attribute.
        assert hasattr(code, "lift_table")
        assert isinstance(code.lift_table, SharedLiftTable)


# ---------------------------------------------------------------
# 3. CSS orthogonality test
# ---------------------------------------------------------------
class TestCSSOrthogonality:
    def test_orthogonality_holds(self, code: CSSCode):
        """H_X @ H_Z^T == 0 mod 2."""
        check_css_orthogonality(code.HX, code.HZ)

    def test_orthogonality_across_seeds(self):
        """Orthogonality must hold for every valid seed."""
        for seed in range(20):
            c = CSSCode(r=3, c=3, L=5, seed=seed)
            check_css_orthogonality(c.HX, c.HZ)

    def test_orthogonality_various_sizes(self):
        """Orthogonality holds across different protograph sizes."""
        for r in [2, 3, 4, 5]:
            for L in [3, 5, 7]:
                c = CSSCode(r=r, c=r, L=L, seed=17)
                check_css_orthogonality(c.HX, c.HZ)


# ---------------------------------------------------------------
# 4. No all-zero rows or columns
# ---------------------------------------------------------------
class TestNoZeroRowsCols:
    def test_HX_no_zero_rows_or_cols(self, code: CSSCode):
        check_no_zero_rows_or_cols(code.HX, name="H_X")

    def test_HZ_no_zero_rows_or_cols(self, code: CSSCode):
        check_no_zero_rows_or_cols(code.HZ, name="H_Z")


# ---------------------------------------------------------------
# 5. Matrix shapes are consistent
# ---------------------------------------------------------------
class TestShapes:
    def test_column_count_matches(self, code: CSSCode):
        """H_X and H_Z must have the same number of columns (= n qubits)."""
        assert code.HX.shape[1] == code.HZ.shape[1]

    def test_expected_dimensions(self, code: CSSCode):
        r, L = code.r, code.L
        expected_cols = 2 * r * L
        assert code.HX.shape == (r * L, expected_cols)
        assert code.HZ.shape == (r * L, expected_cols)


# ---------------------------------------------------------------
# 6. Negative test: breaking shared lift → orthogonality must fail
# ---------------------------------------------------------------
class TestNegativeOrthogonality:
    def test_independent_lifts_break_orthogonality(self):
        """Using *different* lift tables for H_X and H_Z should
        (with high probability) break CSS orthogonality.
        """
        from qldpc.css_code import _circulant_matrix

        r, L = 4, 11
        rng = np.random.RandomState(42)

        # Build circulant base matrices (same as CSSCode does).
        row_a = rng.randint(0, 2, size=r).astype(np.int8)
        row_a[0] = 1  # ensure non-zero
        row_b = rng.randint(0, 2, size=r).astype(np.int8)
        row_b[0] = 1

        A = _circulant_matrix(row_a)
        B = _circulant_matrix(row_b)

        base_X = np.hstack([A, B]).astype(np.int8)
        base_Z = np.hstack([B.T, A.T]).astype(np.int8)

        # Base orthogonality holds because circulants commute.
        assert not np.any((base_X @ base_Z.T) % 2)

        # Use TWO INDEPENDENT lift tables (different seeds).
        lift_X = SharedLiftTable(rows=r, cols=2 * r, L=L, seed=100)
        lift_Z = SharedLiftTable(rows=r, cols=2 * r, L=L, seed=200)

        HX = lift_matrix(base_X, lift_X)
        HZ = lift_matrix(base_Z, lift_Z)

        product = (HX @ HZ.T).tocsr()
        product.data %= 2
        product.eliminate_zeros()

        # With independent lifts, orthogonality almost certainly fails.
        assert product.nnz > 0, (
            "Independent lift tables did NOT break orthogonality — "
            "the test may need a different seed/size to trigger failure."
        )

    def test_invariant_check_raises_on_broken_code(self):
        """check_css_orthogonality must raise when given non-orthogonal matrices."""
        # Construct two random sparse binary matrices that are NOT orthogonal.
        rng = np.random.RandomState(999)
        n = 20
        HX = sp.random(10, n, density=0.3, random_state=rng, format="csr")
        HX.data[:] = 1
        HX = HX.astype(np.int8)
        HZ = sp.random(10, n, density=0.3, random_state=rng, format="csr")
        HZ.data[:] = 1
        HZ = HZ.astype(np.int8)

        # Almost certainly not orthogonal — verify before testing.
        product = (HX @ HZ.T).tocsr()
        product.data %= 2
        product.eliminate_zeros()
        if product.nnz > 0:
            with pytest.raises(ConstructionInvariantError, match="orthogonality"):
                check_css_orthogonality(HX, HZ)


# ---------------------------------------------------------------
# 7. Sparse format check
# ---------------------------------------------------------------
class TestSparseFormat:
    def test_HX_is_csr(self, code: CSSCode):
        assert isinstance(code.HX, sp.csr_matrix)

    def test_HZ_is_csr(self, code: CSSCode):
        assert isinstance(code.HZ, sp.csr_matrix)

    def test_entries_are_binary(self, code: CSSCode):
        """All entries must be 0 or 1."""
        assert set(code.HX.data.tolist()).issubset({0, 1})
        assert set(code.HZ.data.tolist()).issubset({0, 1})


# ---------------------------------------------------------------
# 8. Lift-table determinism
# ---------------------------------------------------------------
class TestLiftTableDeterminism:
    def test_same_seed_same_table(self):
        t1 = SharedLiftTable(3, 6, 7, seed=42)
        t2 = SharedLiftTable(3, 6, 7, seed=42)
        for key in t1._table:
            assert t1[key] == t2[key]

    def test_different_seed_different_table(self):
        t1 = SharedLiftTable(3, 6, 7, seed=42)
        t2 = SharedLiftTable(3, 6, 7, seed=99)
        differs = any(t1[key] != t2[key] for key in t1._table)
        assert differs


# ---------------------------------------------------------------
# 9. Circulant structure test
# ---------------------------------------------------------------
class TestCirculantStructure:
    def test_each_block_is_circulant_permutation(self, code: CSSCode):
        """Each L×L non-zero block of H_X should be a circulant permutation matrix."""
        L = code.L
        dense = code.HX.toarray()
        r_blocks = dense.shape[0] // L
        c_blocks = dense.shape[1] // L
        for bi in range(r_blocks):
            for bj in range(c_blocks):
                block = dense[bi * L:(bi + 1) * L, bj * L:(bj + 1) * L]
                if block.sum() == 0:
                    continue
                # Must be a permutation matrix: exactly one 1 per row & col.
                assert np.all(block.sum(axis=0) == 1), f"Block ({bi},{bj}) not perm"
                assert np.all(block.sum(axis=1) == 1), f"Block ({bi},{bj}) not perm"
                # Must be circulant: row i is a cyclic shift of row 0.
                row0 = block[0]
                shift = int(np.argmax(row0))
                for row_idx in range(L):
                    expected_col = (row_idx + shift) % L
                    assert block[row_idx, expected_col] == 1, (
                        f"Block ({bi},{bj}) not circulant at row {row_idx}"
                    )


# ---------------------------------------------------------------
# 10. Base-matrix commutation (algebraic guarantee)
# ---------------------------------------------------------------
class TestBaseMatrixCommutation:
    def test_base_matrix_commutation(self):
        """(A @ B.T + B @ A.T) % 2 == 0 for circulant A, B across seeds."""
        from qldpc.css_code import _circulant_matrix

        for seed in range(30):
            rng = np.random.RandomState(seed)
            r = 4
            row_a = rng.randint(0, 2, size=r).astype(np.int8)
            if not row_a.any():
                row_a[0] = 1
            row_b = rng.randint(0, 2, size=r).astype(np.int8)
            if not row_b.any():
                row_b[0] = 1

            A = _circulant_matrix(row_a)
            B = _circulant_matrix(row_b)

            # AB^T + BA^T must vanish mod 2 because circulants commute.
            product = (A @ B.T + B @ A.T) % 2
            assert not np.any(product), (
                f"seed={seed}: circulant commutation failed"
            )
