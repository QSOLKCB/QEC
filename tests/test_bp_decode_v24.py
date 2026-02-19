"""
Tests for v2.4.0 bp_decode enhancements: multi-mode, damping, clipping,
backward compatibility, and parameter validation.
"""

import pytest
import numpy as np

from src.qec_qldpc_codes import (
    bp_decode,
    infer,
    syndrome,
    channel_llr,
    create_code,
)


# ───────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────

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


# ───────────────────────────────────────────────────────────────────
# Backward Compatibility
# ───────────────────────────────────────────────────────────────────

class TestBpDecodeBackwardCompat:

    def test_old_max_iter_keyword(self, small_code):
        """Legacy max_iter= keyword still works."""
        llr = np.full(small_code.n, 10.0)
        correction, iters = bp_decode(small_code.H_X, llr, max_iter=10)
        assert iters <= 10
        assert correction.dtype == np.uint8

    def test_new_max_iters_keyword(self, small_code):
        """New max_iters= keyword works."""
        llr = np.full(small_code.n, 10.0)
        correction, iters = bp_decode(small_code.H_X, llr, max_iters=10)
        assert iters <= 10

    def test_both_raises(self, small_code):
        """Passing both max_iter and max_iters raises TypeError."""
        llr = np.full(small_code.n, 10.0)
        with pytest.raises(TypeError, match="both"):
            bp_decode(small_code.H_X, llr, max_iters=10, max_iter=10)

    def test_unexpected_kwarg_raises(self, small_code):
        """Unknown keyword raises TypeError."""
        llr = np.full(small_code.n, 10.0)
        with pytest.raises(TypeError, match="Unexpected"):
            bp_decode(small_code.H_X, llr, foobar=True)

    def test_positional_max_iters(self, small_code):
        """Third positional argument sets max_iters."""
        llr = np.full(small_code.n, 10.0)
        correction, iters = bp_decode(small_code.H_X, llr, 5)
        assert iters <= 5

    def test_infer_max_iter_still_works(self, small_code):
        """infer() still accepts max_iter= keyword."""
        llr = np.full(small_code.n, 10.0)
        correction, iters = infer(small_code.H_X, llr, max_iter=10)
        assert iters <= 10


# ───────────────────────────────────────────────────────────────────
# Mode Selection
# ───────────────────────────────────────────────────────────────────

class TestBpDecodeModes:

    def test_sum_product_is_default(self, noisy_setup):
        """Explicit mode='sum_product' matches default (no mode specified)."""
        code, e, s, llr = noisy_setup
        c1, i1 = bp_decode(code.H_X, llr, max_iters=50, syndrome_vec=s)
        c2, i2 = bp_decode(
            code.H_X, llr, max_iters=50, mode="sum_product", syndrome_vec=s
        )
        np.testing.assert_array_equal(c1, c2)
        assert i1 == i2

    def test_min_sum_runs(self, noisy_setup):
        """min_sum mode produces valid output."""
        code, e, s, llr = noisy_setup
        correction, iters = bp_decode(
            code.H_X, llr, max_iters=50, mode="min_sum", syndrome_vec=s
        )
        assert correction.dtype == np.uint8
        assert correction.shape == (code.n,)
        assert isinstance(iters, int)

    def test_norm_min_sum_runs(self, noisy_setup):
        """norm_min_sum mode produces valid output."""
        code, e, s, llr = noisy_setup
        correction, iters = bp_decode(
            code.H_X, llr, max_iters=50, mode="norm_min_sum", syndrome_vec=s
        )
        assert correction.dtype == np.uint8
        assert correction.shape == (code.n,)

    def test_offset_min_sum_runs(self, noisy_setup):
        """offset_min_sum mode produces valid output."""
        code, e, s, llr = noisy_setup
        correction, iters = bp_decode(
            code.H_X, llr, max_iters=50, mode="offset_min_sum", syndrome_vec=s
        )
        assert correction.dtype == np.uint8
        assert correction.shape == (code.n,)

    def test_invalid_mode_raises(self, small_code):
        """Unknown mode raises ValueError."""
        llr = np.full(small_code.n, 10.0)
        with pytest.raises(ValueError, match="mode"):
            bp_decode(small_code.H_X, llr, mode="invalid")

    def test_norm_factor_1_equals_min_sum(self, noisy_setup):
        """norm_min_sum with norm_factor=1.0 matches min_sum."""
        code, e, s, llr = noisy_setup
        c_ms, _ = bp_decode(
            code.H_X, llr, max_iters=30, mode="min_sum", syndrome_vec=s
        )
        c_nms, _ = bp_decode(
            code.H_X, llr, max_iters=30, mode="norm_min_sum",
            norm_factor=1.0, syndrome_vec=s,
        )
        np.testing.assert_array_equal(c_ms, c_nms)

    def test_offset_0_equals_min_sum(self, noisy_setup):
        """offset_min_sum with offset=0.0 matches min_sum."""
        code, e, s, llr = noisy_setup
        c_ms, _ = bp_decode(
            code.H_X, llr, max_iters=30, mode="min_sum", syndrome_vec=s
        )
        c_oms, _ = bp_decode(
            code.H_X, llr, max_iters=30, mode="offset_min_sum",
            offset=0.0, syndrome_vec=s,
        )
        np.testing.assert_array_equal(c_ms, c_oms)


# ───────────────────────────────────────────────────────────────────
# Damping
# ───────────────────────────────────────────────────────────────────

class TestBpDecodeDamping:

    def test_damping_zero_is_baseline(self, noisy_setup):
        """damping=0.0 produces the same result as no damping."""
        code, e, s, llr = noisy_setup
        c1, i1 = bp_decode(
            code.H_X, llr, max_iters=30, mode="min_sum", syndrome_vec=s
        )
        c2, i2 = bp_decode(
            code.H_X, llr, max_iters=30, mode="min_sum",
            damping=0.0, syndrome_vec=s,
        )
        np.testing.assert_array_equal(c1, c2)
        assert i1 == i2


# ───────────────────────────────────────────────────────────────────
# Clipping
# ───────────────────────────────────────────────────────────────────

class TestBpDecodeClip:

    def test_clip_none_is_default(self, noisy_setup):
        """clip=None produces identical output to default."""
        code, e, s, llr = noisy_setup
        c1, i1 = bp_decode(
            code.H_X, llr, max_iters=30, mode="min_sum", syndrome_vec=s
        )
        c2, i2 = bp_decode(
            code.H_X, llr, max_iters=30, mode="min_sum",
            clip=None, syndrome_vec=s,
        )
        np.testing.assert_array_equal(c1, c2)

    def test_large_clip_matches_no_clip(self, noisy_setup):
        """Very large clip has no effect."""
        code, e, s, llr = noisy_setup
        c1, i1 = bp_decode(
            code.H_X, llr, max_iters=30, mode="min_sum", syndrome_vec=s
        )
        c2, i2 = bp_decode(
            code.H_X, llr, max_iters=30, mode="min_sum",
            clip=1e10, syndrome_vec=s,
        )
        np.testing.assert_array_equal(c1, c2)


# ───────────────────────────────────────────────────────────────────
# Schedule
# ───────────────────────────────────────────────────────────────────

class TestBpDecodeSchedule:

    def test_flooding_is_default(self, small_code):
        """schedule='flooding' works without error."""
        llr = np.full(small_code.n, 10.0)
        correction, _ = bp_decode(
            small_code.H_X, llr, max_iters=5, schedule="flooding"
        )
        assert correction.dtype == np.uint8

    def test_unsupported_schedule_raises(self, small_code):
        """Non-flooding schedule raises NotImplementedError."""
        llr = np.full(small_code.n, 10.0)
        with pytest.raises(NotImplementedError, match="schedule"):
            bp_decode(small_code.H_X, llr, schedule="serial")


# ───────────────────────────────────────────────────────────────────
# Determinism
# ───────────────────────────────────────────────────────────────────

class TestBpDecodeDeterminism:

    @pytest.mark.parametrize("mode", [
        "sum_product", "min_sum", "norm_min_sum", "offset_min_sum"
    ])
    def test_deterministic_across_calls(self, noisy_setup, mode):
        """Same inputs produce identical outputs for every mode."""
        code, e, s, llr = noisy_setup
        c1, i1 = bp_decode(
            code.H_X, llr, max_iters=30, mode=mode, syndrome_vec=s
        )
        c2, i2 = bp_decode(
            code.H_X, llr, max_iters=30, mode=mode, syndrome_vec=s
        )
        np.testing.assert_array_equal(c1, c2)
        assert i1 == i2


# ───────────────────────────────────────────────────────────────────
# Postprocess validation
# ───────────────────────────────────────────────────────────────────

class TestBpDecodePostprocess:

    def test_unknown_postprocess_raises(self, small_code):
        """Unknown postprocess value raises ValueError."""
        llr = np.full(small_code.n, 10.0)
        with pytest.raises(ValueError, match="postprocess"):
            bp_decode(small_code.H_X, llr, postprocess="invalid")
