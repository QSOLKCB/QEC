"""
Tests for v3.8.0 RPC (Redundant Parity Check) augmentation builder.

Covers:
- Determinism: identical inputs → identical outputs across two calls
- Feasible-set integrity: augmented rows obey weight bounds
- Disabled identity: config.enabled=False → original (H, s) returned
- Lexicographic enumeration order
- max_rows cap enforcement
- Edge cases: empty matrix, no feasible pairs
"""

import numpy as np
import pytest

from qec.decoder.rpc import (
    RPCConfig,
    StructuralConfig,
    build_rpc_augmented_system,
)


# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────

@pytest.fixture
def small_H():
    """4×8 binary parity-check matrix with known structure."""
    return np.array([
        [1, 1, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 0, 1],
    ], dtype=np.uint8)


@pytest.fixture
def small_s():
    """Syndrome vector matching small_H (length 4)."""
    return np.array([1, 0, 1, 0], dtype=np.uint8)


# ──────────────────────────────────────────────────────────────────
# Disabled identity
# ──────────────────────────────────────────────────────────────────

class TestDisabledIdentity:

    def test_disabled_returns_original(self, small_H, small_s):
        """When disabled, build_rpc_augmented_system returns (H, s) unchanged."""
        cfg = RPCConfig(enabled=False)
        H_out, s_out = build_rpc_augmented_system(small_H, small_s, cfg)
        assert H_out is small_H
        assert s_out is small_s

    def test_disabled_default_config(self, small_H, small_s):
        """Default RPCConfig has enabled=False."""
        cfg = RPCConfig()
        assert cfg.enabled is False
        H_out, s_out = build_rpc_augmented_system(small_H, small_s, cfg)
        assert H_out is small_H
        assert s_out is small_s

    def test_structural_config_default_disabled(self, small_H, small_s):
        """Default StructuralConfig has rpc.enabled=False."""
        scfg = StructuralConfig()
        assert scfg.rpc.enabled is False
        H_out, s_out = build_rpc_augmented_system(
            small_H, small_s, scfg.rpc
        )
        assert H_out is small_H
        assert s_out is small_s


# ──────────────────────────────────────────────────────────────────
# Determinism
# ──────────────────────────────────────────────────────────────────

class TestDeterminism:

    def test_two_calls_identical(self, small_H, small_s):
        """Two calls with the same inputs produce byte-identical outputs."""
        cfg = RPCConfig(enabled=True, max_rows=10, w_min=2, w_max=6)
        H1, s1 = build_rpc_augmented_system(small_H, small_s, cfg)
        H2, s2 = build_rpc_augmented_system(small_H, small_s, cfg)
        np.testing.assert_array_equal(H1, H2)
        np.testing.assert_array_equal(s1, s2)

    def test_determinism_large_random(self):
        """Determinism holds on a larger random matrix."""
        rng = np.random.default_rng(12345)
        H = (rng.random((20, 40)) < 0.15).astype(np.uint8)
        s = rng.integers(0, 2, size=20, dtype=np.uint8)
        cfg = RPCConfig(enabled=True, max_rows=50, w_min=2, w_max=15)
        H1, s1 = build_rpc_augmented_system(H, s, cfg)
        H2, s2 = build_rpc_augmented_system(H, s, cfg)
        np.testing.assert_array_equal(H1, H2)
        np.testing.assert_array_equal(s1, s2)


# ──────────────────────────────────────────────────────────────────
# Feasible-set integrity
# ──────────────────────────────────────────────────────────────────

class TestFeasibleSetIntegrity:

    def test_augmented_rows_obey_weight_bounds(self, small_H, small_s):
        """Every appended row has Hamming weight in [w_min, w_max]."""
        cfg = RPCConfig(enabled=True, max_rows=100, w_min=2, w_max=4)
        H_aug, s_aug = build_rpc_augmented_system(small_H, small_s, cfg)
        m_orig = small_H.shape[0]
        new_rows = H_aug[m_orig:]
        for row in new_rows:
            w = int(np.sum(row))
            assert cfg.w_min <= w <= cfg.w_max, (
                f"Row weight {w} outside [{cfg.w_min}, {cfg.w_max}]"
            )

    def test_augmented_syndrome_consistency(self, small_H, small_s):
        """Each augmented syndrome bit equals s[i] ^ s[j] for the source pair."""
        cfg = RPCConfig(enabled=True, max_rows=100, w_min=1, w_max=100)
        H_aug, s_aug = build_rpc_augmented_system(small_H, small_s, cfg)
        m_orig = small_H.shape[0]
        # Recompute expected augmented rows/syndrome
        idx = 0
        for i in range(m_orig):
            for j in range(i + 1, m_orig):
                combined = small_H[i] ^ small_H[j]
                w = int(np.sum(combined))
                if 1 <= w <= 100:
                    np.testing.assert_array_equal(
                        H_aug[m_orig + idx], combined,
                        err_msg=f"Row mismatch at augmented index {idx} (pair {i},{j})"
                    )
                    assert s_aug[m_orig + idx] == (small_s[i] ^ small_s[j]), (
                        f"Syndrome mismatch at augmented index {idx}"
                    )
                    idx += 1

    def test_original_rows_preserved(self, small_H, small_s):
        """Original rows in H and s are not modified."""
        cfg = RPCConfig(enabled=True, max_rows=10, w_min=2, w_max=6)
        H_orig = small_H.copy()
        s_orig = small_s.copy()
        H_aug, s_aug = build_rpc_augmented_system(small_H, small_s, cfg)
        np.testing.assert_array_equal(H_aug[:4], H_orig)
        np.testing.assert_array_equal(s_aug[:4], s_orig)

    def test_no_in_place_mutation(self, small_H, small_s):
        """Input arrays are not mutated."""
        H_before = small_H.copy()
        s_before = small_s.copy()
        cfg = RPCConfig(enabled=True, max_rows=10, w_min=2, w_max=6)
        build_rpc_augmented_system(small_H, small_s, cfg)
        np.testing.assert_array_equal(small_H, H_before)
        np.testing.assert_array_equal(small_s, s_before)


# ──────────────────────────────────────────────────────────────────
# Lexicographic enumeration order
# ──────────────────────────────────────────────────────────────────

class TestLexicographicOrder:

    def test_pairs_enumerated_lexicographically(self):
        """Augmented rows appear in (i,j) lexicographic order."""
        H = np.eye(4, dtype=np.uint8)
        s = np.zeros(4, dtype=np.uint8)
        # All pairs of identity rows XOR to weight=2 vectors
        cfg = RPCConfig(enabled=True, max_rows=100, w_min=1, w_max=10)
        H_aug, _ = build_rpc_augmented_system(H, s, cfg)
        new_rows = H_aug[4:]
        # Pairs: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
        expected_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        for k, (i, j) in enumerate(expected_pairs):
            expected_row = H[i] ^ H[j]
            np.testing.assert_array_equal(
                new_rows[k], expected_row,
                err_msg=f"Pair index {k}: expected pair ({i},{j})"
            )


# ──────────────────────────────────────────────────────────────────
# max_rows cap
# ──────────────────────────────────────────────────────────────────

class TestMaxRowsCap:

    def test_max_rows_respected(self):
        """Number of augmented rows never exceeds max_rows."""
        rng = np.random.default_rng(99)
        H = (rng.random((30, 50)) < 0.2).astype(np.uint8)
        s = rng.integers(0, 2, size=30, dtype=np.uint8)
        cfg = RPCConfig(enabled=True, max_rows=5, w_min=1, w_max=100)
        H_aug, s_aug = build_rpc_augmented_system(H, s, cfg)
        assert H_aug.shape[0] == 30 + 5
        assert s_aug.shape[0] == 30 + 5

    def test_max_rows_zero(self, small_H, small_s):
        """max_rows=0 produces no augmented rows."""
        cfg = RPCConfig(enabled=True, max_rows=0, w_min=1, w_max=100)
        H_aug, s_aug = build_rpc_augmented_system(small_H, small_s, cfg)
        assert H_aug.shape[0] == small_H.shape[0]
        assert s_aug.shape[0] == small_s.shape[0]


# ──────────────────────────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_single_row_matrix(self):
        """Matrix with 1 row has no pairs → no augmentation."""
        H = np.array([[1, 0, 1, 0]], dtype=np.uint8)
        s = np.array([1], dtype=np.uint8)
        cfg = RPCConfig(enabled=True, max_rows=10, w_min=1, w_max=10)
        H_aug, s_aug = build_rpc_augmented_system(H, s, cfg)
        assert H_aug.shape[0] == 1
        assert s_aug.shape[0] == 1

    def test_no_feasible_pairs(self, small_H, small_s):
        """If w_min > max possible weight, no rows are added."""
        cfg = RPCConfig(enabled=True, max_rows=10, w_min=100, w_max=200)
        H_aug, s_aug = build_rpc_augmented_system(small_H, small_s, cfg)
        assert H_aug.shape[0] == small_H.shape[0]

    def test_dtype_preserved(self, small_H, small_s):
        """Output arrays have dtype uint8."""
        cfg = RPCConfig(enabled=True, max_rows=10, w_min=1, w_max=100)
        H_aug, s_aug = build_rpc_augmented_system(small_H, small_s, cfg)
        assert H_aug.dtype == np.uint8
        assert s_aug.dtype == np.uint8

    def test_immutable_config(self):
        """RPCConfig is frozen (immutable)."""
        cfg = RPCConfig(enabled=True)
        with pytest.raises(AttributeError):
            cfg.enabled = False

    def test_structural_config_immutable(self):
        """StructuralConfig is frozen (immutable)."""
        scfg = StructuralConfig()
        with pytest.raises(AttributeError):
            scfg.rpc = RPCConfig(enabled=True)
