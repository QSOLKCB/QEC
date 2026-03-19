"""
Tests for v2.8.0 Edit 2: improved_norm and improved_offset check-node modes.
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


class TestImprovedNormDeterminism:

    def test_flooding_determinism(self, noisy_setup):
        """improved_norm with flooding: two runs must be identical."""
        code, e, s, llr = noisy_setup
        r1 = bp_decode(
            code.H_X, llr, max_iters=20, mode="improved_norm",
            syndrome_vec=s, schedule="flooding",
        )
        r2 = bp_decode(
            code.H_X, llr, max_iters=20, mode="improved_norm",
            syndrome_vec=s, schedule="flooding",
        )
        np.testing.assert_array_equal(r1[0], r2[0])
        assert r1[1] == r2[1]

    def test_layered_determinism(self, noisy_setup):
        """improved_norm with layered: two runs must be identical."""
        code, e, s, llr = noisy_setup
        r1 = bp_decode(
            code.H_X, llr, max_iters=20, mode="improved_norm",
            syndrome_vec=s, schedule="layered",
        )
        r2 = bp_decode(
            code.H_X, llr, max_iters=20, mode="improved_norm",
            syndrome_vec=s, schedule="layered",
        )
        np.testing.assert_array_equal(r1[0], r2[0])
        assert r1[1] == r2[1]

    def test_residual_determinism(self, noisy_setup):
        """improved_norm with residual: two runs must be identical."""
        code, e, s, llr = noisy_setup
        r1 = bp_decode(
            code.H_X, llr, max_iters=20, mode="improved_norm",
            syndrome_vec=s, schedule="residual",
        )
        r2 = bp_decode(
            code.H_X, llr, max_iters=20, mode="improved_norm",
            syndrome_vec=s, schedule="residual",
        )
        np.testing.assert_array_equal(r1[0], r2[0])
        assert r1[1] == r2[1]


class TestImprovedOffsetDeterminism:

    def test_flooding_determinism(self, noisy_setup):
        """improved_offset with flooding: two runs must be identical."""
        code, e, s, llr = noisy_setup
        r1 = bp_decode(
            code.H_X, llr, max_iters=20, mode="improved_offset",
            syndrome_vec=s, schedule="flooding",
        )
        r2 = bp_decode(
            code.H_X, llr, max_iters=20, mode="improved_offset",
            syndrome_vec=s, schedule="flooding",
        )
        np.testing.assert_array_equal(r1[0], r2[0])
        assert r1[1] == r2[1]


class TestBackwardCompatMinSum:

    def test_min_sum_unchanged(self, noisy_setup):
        """mode='min_sum' must produce identical results to pre-v2.8.0."""
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

    def test_min_sum_layered_unchanged(self, noisy_setup):
        """mode='min_sum' layered must produce identical results."""
        code, e, s, llr = noisy_setup
        r1 = bp_decode(
            code.H_X, llr, max_iters=50, mode="min_sum",
            syndrome_vec=s, schedule="layered",
        )
        r2 = bp_decode(
            code.H_X, llr, max_iters=50, mode="min_sum",
            syndrome_vec=s, schedule="layered",
        )
        np.testing.assert_array_equal(r1[0], r2[0])
        assert r1[1] == r2[1]
