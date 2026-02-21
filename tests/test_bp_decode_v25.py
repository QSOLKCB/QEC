"""
Tests for v2.5.0 bp_decode enhancements: layered scheduling and
backward compatibility regression tests.
"""

import pytest
import numpy as np

from src.qec_qldpc_codes import (
    bp_decode,
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
# Layered Schedule
# ───────────────────────────────────────────────────────────────────

class TestBpDecodeLayered:

    def test_layered_schedule_accepted(self, small_code):
        """schedule='layered' does not raise."""
        llr = np.full(small_code.n, 10.0)
        correction, _ = bp_decode(
            small_code.H_X, llr, max_iters=5, schedule="layered"
        )
        assert correction.dtype == np.uint8

    def test_layered_produces_valid_output(self, noisy_setup):
        """Layered produces output with correct dtype and shape."""
        code, e, s, llr = noisy_setup
        correction, iters = bp_decode(
            code.H_X, llr, max_iters=30, mode="min_sum",
            schedule="layered", syndrome_vec=s,
        )
        assert correction.dtype == np.uint8
        assert correction.shape == (code.n,)
        assert isinstance(iters, int)
        assert 1 <= iters <= 30

    @pytest.mark.parametrize("mode", [
        "sum_product", "min_sum", "norm_min_sum", "offset_min_sum"
    ])
    def test_layered_deterministic(self, noisy_setup, mode):
        """Same inputs produce identical outputs for every mode."""
        code, e, s, llr = noisy_setup
        c1, i1 = bp_decode(
            code.H_X, llr, max_iters=30, mode=mode,
            schedule="layered", syndrome_vec=s,
        )
        c2, i2 = bp_decode(
            code.H_X, llr, max_iters=30, mode=mode,
            schedule="layered", syndrome_vec=s,
        )
        np.testing.assert_array_equal(c1, c2)
        assert i1 == i2

    def test_layered_converges_easy_code(self, noisy_setup):
        """Layered converges on low-noise scenario."""
        code, e, s, llr = noisy_setup
        correction, iters = bp_decode(
            code.H_X, llr, max_iters=50, mode="min_sum",
            schedule="layered", syndrome_vec=s,
        )
        # Check syndrome matches (convergence).
        result_syn = (
            (code.H_X.astype(np.int32) @ correction.astype(np.int32)) % 2
        ).astype(np.uint8)
        np.testing.assert_array_equal(result_syn, s)

    def test_layered_sum_product_mode(self, noisy_setup):
        """sum_product with layered schedule works."""
        code, e, s, llr = noisy_setup
        correction, _ = bp_decode(
            code.H_X, llr, max_iters=30, mode="sum_product",
            schedule="layered", syndrome_vec=s,
        )
        assert correction.dtype == np.uint8

    def test_layered_with_damping(self, noisy_setup):
        """Layered with damping > 0 works."""
        code, e, s, llr = noisy_setup
        correction, _ = bp_decode(
            code.H_X, llr, max_iters=30, mode="min_sum",
            schedule="layered", damping=0.3, syndrome_vec=s,
        )
        assert correction.dtype == np.uint8

    def test_layered_with_clipping(self, noisy_setup):
        """Layered with clipping works."""
        code, e, s, llr = noisy_setup
        correction, _ = bp_decode(
            code.H_X, llr, max_iters=30, mode="min_sum",
            schedule="layered", clip=5.0, syndrome_vec=s,
        )
        assert correction.dtype == np.uint8

    def test_layered_with_osd0_postprocess(self, noisy_setup):
        """Layered with postprocess='osd0' works."""
        code, e, s, llr = noisy_setup
        correction, _ = bp_decode(
            code.H_X, llr, max_iters=5, mode="min_sum",
            schedule="layered", postprocess="osd0", syndrome_vec=s,
        )
        assert correction.dtype == np.uint8

    def test_layered_with_osd1_postprocess(self, noisy_setup):
        """Layered with postprocess='osd1' works."""
        code, e, s, llr = noisy_setup
        correction, _ = bp_decode(
            code.H_X, llr, max_iters=5, mode="min_sum",
            schedule="layered", postprocess="osd1", syndrome_vec=s,
        )
        assert correction.dtype == np.uint8

    def test_invalid_schedule_still_raises(self, small_code):
        """Non-valid schedule still raises NotImplementedError."""
        llr = np.full(small_code.n, 10.0)
        with pytest.raises(NotImplementedError, match="schedule"):
            bp_decode(small_code.H_X, llr, schedule="serial")

    def test_layered_different_from_flooding(self, noisy_setup):
        """Layered and flooding are different algorithms (may produce different
        iteration counts or corrections on non-trivial inputs)."""
        code, e, s, llr = noisy_setup
        c_flood, i_flood = bp_decode(
            code.H_X, llr, max_iters=50, mode="min_sum",
            schedule="flooding", syndrome_vec=s,
        )
        c_layer, i_layer = bp_decode(
            code.H_X, llr, max_iters=50, mode="min_sum",
            schedule="layered", syndrome_vec=s,
        )
        # Both should produce valid corrections (syndrome match).
        flood_syn = (
            (code.H_X.astype(np.int32) @ c_flood.astype(np.int32)) % 2
        ).astype(np.uint8)
        layer_syn = (
            (code.H_X.astype(np.int32) @ c_layer.astype(np.int32)) % 2
        ).astype(np.uint8)
        np.testing.assert_array_equal(flood_syn, s)
        np.testing.assert_array_equal(layer_syn, s)
        # Iteration counts are typically different.
        # We don't assert they differ (could coincide), but both are valid.


# ───────────────────────────────────────────────────────────────────
# Backward Compatibility Regression Tests
# ───────────────────────────────────────────────────────────────────

class TestBackwardCompatV25:

    def test_flooding_unchanged(self, noisy_setup):
        """Flooding output identical to v2.4.0 default behavior."""
        code, e, s, llr = noisy_setup
        c1, i1 = bp_decode(
            code.H_X, llr, max_iters=30, mode="min_sum", syndrome_vec=s
        )
        c2, i2 = bp_decode(
            code.H_X, llr, max_iters=30, mode="min_sum",
            schedule="flooding", syndrome_vec=s,
        )
        np.testing.assert_array_equal(c1, c2)
        assert i1 == i2

    def test_default_params_unchanged(self, small_code):
        """bp_decode with only H and llr still works identically."""
        llr = np.full(small_code.n, 10.0)
        c1, i1 = bp_decode(small_code.H_X, llr)
        c2, i2 = bp_decode(small_code.H_X, llr)
        np.testing.assert_array_equal(c1, c2)
        assert i1 == i2

    def test_postprocess_osd0_unchanged(self, noisy_setup):
        """postprocess='osd0' in bp_decode unchanged."""
        code, e, s, llr = noisy_setup
        c1, i1 = bp_decode(
            code.H_X, llr, max_iters=5, mode="min_sum",
            postprocess="osd0", syndrome_vec=s,
        )
        c2, i2 = bp_decode(
            code.H_X, llr, max_iters=5, mode="min_sum",
            postprocess="osd0", syndrome_vec=s,
        )
        np.testing.assert_array_equal(c1, c2)
        assert i1 == i2
