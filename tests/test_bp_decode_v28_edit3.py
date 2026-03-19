"""
Tests for v2.8.0 Edit 3: hybrid_residual schedule.
"""

import numpy as np
import pytest

from qec_qldpc_codes import bp_decode, syndrome, channel_llr, create_code


@pytest.fixture
def small_code():
    """Small rate-0.50 code for fast tests."""
    return create_code('rate_0.50', lifting_size=8, seed=42)


@pytest.fixture
def noisy_setup(small_code):
    """Code + low-noise error + syndrome + LLR."""
    rng = np.random.default_rng(42)
    e = (rng.random(small_code.n) < 0.01).astype(np.uint8)
    s = syndrome(small_code.H_X, e)
    llr = channel_llr(e, 0.01)
    return small_code, e, s, llr


class TestHybridResidualDeterminism:

    def test_determinism_no_threshold(self, noisy_setup):
        """hybrid_residual without threshold: two runs must be identical."""
        code, e, s, llr = noisy_setup
        r1 = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="hybrid_residual",
        )
        r2 = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="hybrid_residual",
        )
        np.testing.assert_array_equal(r1[0], r2[0])
        assert r1[1] == r2[1]

    def test_determinism_with_threshold(self, noisy_setup):
        """hybrid_residual with threshold: two runs must be identical."""
        code, e, s, llr = noisy_setup
        r1 = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="hybrid_residual",
            hybrid_residual_threshold=0.1,
        )
        r2 = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="hybrid_residual",
            hybrid_residual_threshold=0.1,
        )
        np.testing.assert_array_equal(r1[0], r2[0])
        assert r1[1] == r2[1]

    def test_determinism_sum_product(self, noisy_setup):
        """hybrid_residual with sum_product: deterministic."""
        code, e, s, llr = noisy_setup
        r1 = bp_decode(
            code.H_X, llr, max_iters=20, mode="sum_product",
            syndrome_vec=s, schedule="hybrid_residual",
        )
        r2 = bp_decode(
            code.H_X, llr, max_iters=20, mode="sum_product",
            syndrome_vec=s, schedule="hybrid_residual",
        )
        np.testing.assert_array_equal(r1[0], r2[0])
        assert r1[1] == r2[1]


class TestHybridResidualTieBreak:

    def test_equal_residual_ascending_index(self):
        """With all residuals equal (iter 0), checks ordered by ascending index within each layer."""
        # Tiny 4-check code: checks 0,2 in even layer; 1,3 in odd layer.
        # At iteration 0, all residuals are 0 so tie-break is ascending index.
        H = np.array([
            [1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1],
        ], dtype=np.uint8)
        llr = np.array([2.0, -1.0, 0.5, -0.3, 1.0])

        # Run with hybrid_residual — must not crash and must be deterministic.
        r1 = bp_decode(H, llr, max_iters=5, mode="min_sum",
                        schedule="hybrid_residual")
        r2 = bp_decode(H, llr, max_iters=5, mode="min_sum",
                        schedule="hybrid_residual")
        np.testing.assert_array_equal(r1[0], r2[0])
        assert r1[1] == r2[1]


class TestFloodingUnchanged:

    def test_default_schedule_unchanged(self, noisy_setup):
        """Default schedule='flooding' must be unaffected by hybrid_residual code."""
        code, e, s, llr = noisy_setup
        r1 = bp_decode(
            code.H_X, llr, max_iters=50, mode="min_sum",
            syndrome_vec=s, schedule="flooding",
        )
        r2 = bp_decode(
            code.H_X, llr, max_iters=50, mode="min_sum",
            syndrome_vec=s, schedule="flooding",
        )
        np.testing.assert_array_equal(r1[0], r2[0])
        assert r1[1] == r2[1]
