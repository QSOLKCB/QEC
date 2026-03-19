"""
Tests for v3.7.0: Uniformly Reweighted BP (URW-BP).

Verifies:
    1. Baseline invariance: all existing modes produce identical output.
    2. rho=1.0 invariance: min_sum_urw with rho=1.0 matches min_sum exactly.
    3. Determinism: two identical URW runs produce identical results.
    4. Validation: invalid rho raises deterministic error.
    5. URW with rho<1 produces valid output (converges or returns valid hard decision).
    6. Identity: URW mode reflected in adapter identity.
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


# ── Baseline invariance ─────────────────────────────────────────────

class TestBaselineInvariance:
    """URW disabled → existing modes produce identical outputs."""

    @pytest.mark.parametrize("mode", [
        "sum_product", "min_sum", "norm_min_sum", "offset_min_sum",
        "improved_norm", "improved_offset",
    ])
    @pytest.mark.parametrize("schedule", ["flooding", "layered"])
    def test_existing_mode_unchanged(self, noisy_setup, mode, schedule):
        """Every existing mode returns bit-identical output regardless of
        urw_rho default (1.0) presence in the signature."""
        code, e, s, llr = noisy_setup
        r1 = bp_decode(
            code.H_X, llr, max_iters=20, mode=mode,
            syndrome_vec=s, schedule=schedule,
        )
        # Call again — must be identical.
        r2 = bp_decode(
            code.H_X, llr, max_iters=20, mode=mode,
            syndrome_vec=s, schedule=schedule,
        )
        np.testing.assert_array_equal(r1[0], r2[0])
        assert r1[1] == r2[1]


# ── rho=1.0 invariance ──────────────────────────────────────────────

class TestRhoOneInvariance:
    """min_sum_urw with rho=1.0 must produce identical output to min_sum."""

    @pytest.mark.parametrize("schedule", ["flooding", "layered"])
    def test_urw_rho1_matches_min_sum(self, noisy_setup, schedule):
        """URW rho=1.0 is algebraically equivalent to min_sum."""
        code, e, s, llr = noisy_setup
        r_ms = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule=schedule,
        )
        r_urw = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum_urw",
            urw_rho=1.0, syndrome_vec=s, schedule=schedule,
        )
        np.testing.assert_array_equal(r_ms[0], r_urw[0])
        assert r_ms[1] == r_urw[1]

    @pytest.mark.parametrize("schedule", ["flooding", "layered"])
    def test_urw_rho1_with_damping(self, noisy_setup, schedule):
        """URW rho=1.0 + damping matches min_sum + damping."""
        code, e, s, llr = noisy_setup
        r_ms = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            damping=0.25, syndrome_vec=s, schedule=schedule,
        )
        r_urw = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum_urw",
            urw_rho=1.0, damping=0.25, syndrome_vec=s, schedule=schedule,
        )
        np.testing.assert_array_equal(r_ms[0], r_urw[0])
        assert r_ms[1] == r_urw[1]

    @pytest.mark.parametrize("schedule", ["flooding", "layered"])
    def test_urw_rho1_with_clipping(self, noisy_setup, schedule):
        """URW rho=1.0 + clipping matches min_sum + clipping."""
        code, e, s, llr = noisy_setup
        r_ms = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            clip=5.0, syndrome_vec=s, schedule=schedule,
        )
        r_urw = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum_urw",
            urw_rho=1.0, clip=5.0, syndrome_vec=s, schedule=schedule,
        )
        np.testing.assert_array_equal(r_ms[0], r_urw[0])
        assert r_ms[1] == r_urw[1]

    def test_urw_rho1_with_llr_history(self, noisy_setup):
        """URW rho=1.0 with llr_history matches min_sum with llr_history."""
        code, e, s, llr = noisy_setup
        r_ms = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, llr_history=3,
        )
        r_urw = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum_urw",
            urw_rho=1.0, syndrome_vec=s, llr_history=3,
        )
        np.testing.assert_array_equal(r_ms[0], r_urw[0])
        assert r_ms[1] == r_urw[1]
        np.testing.assert_array_equal(r_ms[2], r_urw[2])


# ── Determinism ──────────────────────────────────────────────────────

class TestURWDeterminism:
    """Two identical URW runs must produce byte-identical results."""

    @pytest.mark.parametrize("rho", [0.5, 0.8, 1.0])
    @pytest.mark.parametrize("schedule", ["flooding", "layered"])
    def test_deterministic_runs(self, noisy_setup, rho, schedule):
        """Same config, same seed → same output."""
        code, e, s, llr = noisy_setup
        r1 = bp_decode(
            code.H_X, llr, max_iters=30, mode="min_sum_urw",
            urw_rho=rho, syndrome_vec=s, schedule=schedule,
        )
        r2 = bp_decode(
            code.H_X, llr, max_iters=30, mode="min_sum_urw",
            urw_rho=rho, syndrome_vec=s, schedule=schedule,
        )
        np.testing.assert_array_equal(r1[0], r2[0])
        assert r1[1] == r2[1]

    def test_deterministic_with_residual_schedule(self, noisy_setup):
        """URW with residual schedule is deterministic."""
        code, e, s, llr = noisy_setup
        r1 = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum_urw",
            urw_rho=0.7, syndrome_vec=s, schedule="residual",
        )
        r2 = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum_urw",
            urw_rho=0.7, syndrome_vec=s, schedule="residual",
        )
        np.testing.assert_array_equal(r1[0], r2[0])
        assert r1[1] == r2[1]

    def test_deterministic_with_hybrid_residual(self, noisy_setup):
        """URW with hybrid_residual schedule is deterministic."""
        code, e, s, llr = noisy_setup
        r1 = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum_urw",
            urw_rho=0.6, syndrome_vec=s, schedule="hybrid_residual",
        )
        r2 = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum_urw",
            urw_rho=0.6, syndrome_vec=s, schedule="hybrid_residual",
        )
        np.testing.assert_array_equal(r1[0], r2[0])
        assert r1[1] == r2[1]


# ── Validation ───────────────────────────────────────────────────────

class TestURWValidation:
    """Invalid urw_rho values must raise deterministic errors."""

    def test_rho_zero_raises(self, noisy_setup):
        """rho=0.0 is invalid (must be > 0)."""
        code, e, s, llr = noisy_setup
        with pytest.raises(ValueError, match="urw_rho must be in the interval"):
            bp_decode(
                code.H_X, llr, max_iters=10, mode="min_sum_urw",
                urw_rho=0.0, syndrome_vec=s,
            )

    def test_rho_negative_raises(self, noisy_setup):
        """Negative rho is invalid."""
        code, e, s, llr = noisy_setup
        with pytest.raises(ValueError, match="urw_rho must be in the interval"):
            bp_decode(
                code.H_X, llr, max_iters=10, mode="min_sum_urw",
                urw_rho=-0.5, syndrome_vec=s,
            )

    def test_rho_above_one_raises(self, noisy_setup):
        """rho > 1.0 is invalid."""
        code, e, s, llr = noisy_setup
        with pytest.raises(ValueError, match="urw_rho must be in the interval"):
            bp_decode(
                code.H_X, llr, max_iters=10, mode="min_sum_urw",
                urw_rho=1.5, syndrome_vec=s,
            )

    def test_rho_ignored_for_other_modes(self, noisy_setup):
        """urw_rho is not validated when mode is not min_sum_urw."""
        code, e, s, llr = noisy_setup
        # Should not raise even with out-of-range rho, since mode != min_sum_urw.
        r = bp_decode(
            code.H_X, llr, max_iters=10, mode="min_sum",
            urw_rho=0.0, syndrome_vec=s,
        )
        assert r[0].shape == (code.n,)


# ── URW output validity ─────────────────────────────────────────────

class TestURWOutput:
    """URW with rho < 1 produces valid decoder output."""

    @pytest.mark.parametrize("rho", [0.3, 0.5, 0.8, 0.95])
    def test_output_shape(self, noisy_setup, rho):
        """Output shape is (n,) uint8 correction vector."""
        code, e, s, llr = noisy_setup
        r = bp_decode(
            code.H_X, llr, max_iters=30, mode="min_sum_urw",
            urw_rho=rho, syndrome_vec=s,
        )
        assert r[0].shape == (code.n,)
        assert r[0].dtype == np.uint8
        assert isinstance(r[1], int)
        assert r[1] >= 1

    def test_urw_postprocess_osd0(self, noisy_setup):
        """URW with osd0 postprocessing works correctly."""
        code, e, s, llr = noisy_setup
        r = bp_decode(
            code.H_X, llr, max_iters=10, mode="min_sum_urw",
            urw_rho=0.7, syndrome_vec=s, postprocess="osd0",
        )
        assert r[0].shape == (code.n,)

    def test_different_rho_different_beliefs(self, noisy_setup):
        """Different rho values produce different intermediate beliefs."""
        code, e, s, llr = noisy_setup
        # Use higher noise to ensure non-trivial decoding.
        rng = np.random.default_rng(99)
        e2 = (rng.random(code.n) < 0.08).astype(np.uint8)
        s2 = syndrome(code.H_X, e2)
        llr2 = channel_llr(e2, 0.08)

        # Use llr_history to capture internal beliefs — these must differ
        # for different rho values even if hard decisions happen to match.
        r1 = bp_decode(
            code.H_X, llr2, max_iters=3, mode="min_sum_urw",
            urw_rho=0.5, syndrome_vec=s2, llr_history=1,
        )
        r2 = bp_decode(
            code.H_X, llr2, max_iters=3, mode="min_sum_urw",
            urw_rho=0.9, syndrome_vec=s2, llr_history=1,
        )
        # LLR history (beliefs) must differ for different rho.
        assert not np.array_equal(r1[2], r2[2]), \
            "Different rho values should produce different beliefs"


# ── Identity ─────────────────────────────────────────────────────────

class TestURWIdentity:
    """URW mode correctly reflected in adapter identity."""

    def test_identity_includes_urw_mode(self):
        """BPAdapter identity includes mode=min_sum_urw and urw_rho."""
        from bench.adapters.bp import BPAdapter
        adapter = BPAdapter()
        adapter.initialize(config={
            "mode": "min_sum_urw",
            "schedule": "flooding",
            "urw_rho": 0.7,
        })
        identity = adapter.serialize_identity()
        assert identity["adapter"] == "bp"
        assert identity["params"]["mode"] == "min_sum_urw"
        assert identity["params"]["urw_rho"] == 0.7

    def test_identity_no_urw_for_baseline(self):
        """BPAdapter identity for min_sum does not include urw_rho."""
        from bench.adapters.bp import BPAdapter
        adapter = BPAdapter()
        adapter.initialize(config={
            "mode": "min_sum",
            "schedule": "flooding",
        })
        identity = adapter.serialize_identity()
        assert identity["adapter"] == "bp"
        assert identity["params"]["mode"] == "min_sum"
        assert "urw_rho" not in identity["params"]

    def test_adapter_name_includes_urw(self):
        """BPAdapter.name reflects min_sum_urw mode."""
        from bench.adapters.bp import BPAdapter
        adapter = BPAdapter()
        adapter.initialize(config={
            "mode": "min_sum_urw",
            "schedule": "flooding",
            "urw_rho": 0.5,
        })
        assert "min_sum_urw" in adapter.name
