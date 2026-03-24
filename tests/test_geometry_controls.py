"""
Tests for v3.9.1 geometry field controls.

Verifies:
  1. geometry_strength scaling is deterministic
  2. normalize_geometry produces unit variance
  3. baseline behavior unchanged when features disabled
  4. DPS harness new modes execute correctly
"""

from __future__ import annotations

import numpy as np
import pytest

from qec_qldpc_codes import bp_decode, syndrome, channel_llr, create_code
from qec.decoder.rpc import RPCConfig, StructuralConfig
from qec.channel.geometry import (
    centered_syndrome_field,
    syndrome_field,
    pseudo_prior_bias,
    apply_pseudo_prior,
)


# ── Fixtures ────────────────────────────────────────────────────────

SEED = 42
DISTANCE = 3
P = 0.02
MAX_ITERS = 50
BP_MODE = "min_sum"


@pytest.fixture
def code():
    return create_code(name="rate_0.50", lifting_size=DISTANCE, seed=SEED)


@pytest.fixture
def H(code):
    return code.H_X


@pytest.fixture
def instance(H):
    """Generate an instance with a non-trivial syndrome."""
    rng = np.random.default_rng(SEED)
    n = H.shape[1]
    # Find an error that produces a non-zero syndrome for meaningful tests.
    for _ in range(100):
        e = (rng.random(n) < P).astype(np.uint8)
        s = syndrome(H, e)
        if np.any(s):
            break
    llr = channel_llr(e, P)
    return {"e": e, "s": s, "llr": llr}


# ── Helpers ─────────────────────────────────────────────────────────

def _apply_geometry_pipeline(H, s, structural):
    """Reproduce the adapter/harness geometry pipeline for testing."""
    llr_used = None

    if structural.centered_field:
        llr_used = centered_syndrome_field(H, s)
    elif structural.pseudo_prior:
        llr_used = syndrome_field(H, s)

    if structural.pseudo_prior:
        bias = pseudo_prior_bias(H, s)
        if llr_used is None:
            llr_used = syndrome_field(H, s)
        llr_used = apply_pseudo_prior(
            llr_used, bias, structural.pseudo_prior_strength,
        )

    if llr_used is None:
        return None

    geometry_active = structural.centered_field or structural.pseudo_prior
    if geometry_active:
        if structural.normalize_geometry:
            std = np.std(np.asarray(llr_used, dtype=np.float64))
            llr_used = np.asarray(llr_used, dtype=np.float64) / (std + 1e-12)
        if structural.geometry_strength != 1.0:
            llr_used = structural.geometry_strength * np.asarray(
                llr_used, dtype=np.float64
            )

    return llr_used


# ── Test 1: geometry_strength scaling is deterministic ──────────────

class TestGeometryStrength:
    """Tests for geometry_strength scaling."""

    def test_strength_scales_llr(self, H, instance):
        """geometry_strength=2.0 should produce exactly 2x the base LLR."""
        s = instance["s"]

        cfg_base = StructuralConfig(centered_field=True, geometry_strength=1.0)
        cfg_scaled = StructuralConfig(centered_field=True, geometry_strength=2.0)

        llr_base = _apply_geometry_pipeline(H, s, cfg_base)
        llr_scaled = _apply_geometry_pipeline(H, s, cfg_scaled)

        np.testing.assert_allclose(llr_scaled, 2.0 * llr_base, atol=1e-14)

    def test_strength_deterministic(self, H, instance):
        """Running the same geometry_strength config twice gives identical results."""
        s = instance["s"]
        cfg = StructuralConfig(centered_field=True, geometry_strength=1.5)

        llr_1 = _apply_geometry_pipeline(H, s, cfg)
        llr_2 = _apply_geometry_pipeline(H, s, cfg)

        np.testing.assert_array_equal(llr_1, llr_2)

    def test_strength_1_is_identity(self, H, instance):
        """geometry_strength=1.0 should not alter the LLR."""
        s = instance["s"]

        cfg_default = StructuralConfig(centered_field=True)
        cfg_explicit = StructuralConfig(centered_field=True, geometry_strength=1.0)

        llr_default = _apply_geometry_pipeline(H, s, cfg_default)
        llr_explicit = _apply_geometry_pipeline(H, s, cfg_explicit)

        np.testing.assert_array_equal(llr_default, llr_explicit)

    def test_strength_with_pseudo_prior(self, H, instance):
        """geometry_strength scales the full pipeline including pseudo_prior."""
        s = instance["s"]

        cfg_base = StructuralConfig(
            centered_field=True, pseudo_prior=True, geometry_strength=1.0,
        )
        cfg_scaled = StructuralConfig(
            centered_field=True, pseudo_prior=True, geometry_strength=3.0,
        )

        llr_base = _apply_geometry_pipeline(H, s, cfg_base)
        llr_scaled = _apply_geometry_pipeline(H, s, cfg_scaled)

        np.testing.assert_allclose(llr_scaled, 3.0 * llr_base, atol=1e-14)


# ── Test 2: normalize_geometry produces unit variance ──────────────

class TestNormalizeGeometry:
    """Tests for normalize_geometry."""

    def test_normalized_unit_variance(self, H, instance):
        """When normalize_geometry=True, the output should have unit variance."""
        s = instance["s"]
        cfg = StructuralConfig(centered_field=True, normalize_geometry=True)

        llr = _apply_geometry_pipeline(H, s, cfg)
        assert llr is not None
        np.testing.assert_allclose(np.std(llr), 1.0, atol=1e-6)

    def test_normalized_deterministic(self, H, instance):
        """Normalized output is deterministic."""
        s = instance["s"]
        cfg = StructuralConfig(centered_field=True, normalize_geometry=True)

        llr_1 = _apply_geometry_pipeline(H, s, cfg)
        llr_2 = _apply_geometry_pipeline(H, s, cfg)

        np.testing.assert_array_equal(llr_1, llr_2)

    def test_normalized_with_pseudo_prior(self, H, instance):
        """Normalization works when combined with pseudo_prior."""
        s = instance["s"]
        cfg = StructuralConfig(
            centered_field=True, pseudo_prior=True, normalize_geometry=True,
        )

        llr = _apply_geometry_pipeline(H, s, cfg)
        assert llr is not None
        np.testing.assert_allclose(np.std(llr), 1.0, atol=1e-6)

    def test_normalized_then_strength(self, H, instance):
        """normalize then strength: std should equal geometry_strength."""
        s = instance["s"]
        cfg = StructuralConfig(
            centered_field=True,
            normalize_geometry=True,
            geometry_strength=2.0,
        )

        llr = _apply_geometry_pipeline(H, s, cfg)
        assert llr is not None
        # After normalization std=1, then * 2.0 => std=2.0
        np.testing.assert_allclose(np.std(llr), 2.0, atol=1e-6)

    def test_normalize_disabled_by_default(self, H, instance):
        """Default StructuralConfig has normalize_geometry=False."""
        cfg = StructuralConfig()
        assert cfg.normalize_geometry is False


# ── Test 3: baseline behavior unchanged when features disabled ──────

class TestBaselineInvariance:
    """Verify baseline is not affected by new fields at default values."""

    def test_default_config_unchanged(self):
        """Default StructuralConfig should have geometry_strength=1.0 and
        normalize_geometry=False."""
        cfg = StructuralConfig()
        assert cfg.geometry_strength == 1.0
        assert cfg.normalize_geometry is False

    def test_baseline_decode_unchanged(self, H, instance):
        """Baseline decode (no geometry interventions) is unaffected."""
        e = instance["e"]
        s = instance["s"]
        llr = instance["llr"]

        # Decode with default structural config (all disabled).
        result = bp_decode(
            H, llr,
            max_iters=MAX_ITERS,
            mode=BP_MODE,
            schedule="flooding",
            syndrome_vec=s,
        )
        correction_1, iters_1 = result[0], result[1]

        # Decode again — should be byte-identical.
        result2 = bp_decode(
            H, llr,
            max_iters=MAX_ITERS,
            mode=BP_MODE,
            schedule="flooding",
            syndrome_vec=s,
        )
        correction_2, iters_2 = result2[0], result2[1]

        np.testing.assert_array_equal(correction_1, correction_2)
        assert iters_1 == iters_2

    def test_no_geometry_when_disabled(self, H, instance):
        """When centered_field and pseudo_prior are both False,
        _apply_geometry_pipeline returns None (no modification)."""
        s = instance["s"]
        cfg = StructuralConfig(geometry_strength=2.0, normalize_geometry=True)

        llr = _apply_geometry_pipeline(H, s, cfg)
        assert llr is None, "Geometry pipeline should not activate without centered_field or pseudo_prior"


# ── Test 4: DPS harness new modes execute correctly ────────────────

class TestDPSHarnessModes:
    """Verify DPS harness new geometry control modes."""

    def test_new_modes_defined(self):
        """New modes are present in MODES dict."""
        from bench_reports.dps_v381_eval import MODES, MODE_ORDER

        for mode_name in ("centered_strong", "centered_normalized",
                          "centered_prior_normalized"):
            assert mode_name in MODES, f"Mode {mode_name} missing from MODES"
            assert mode_name in MODE_ORDER, f"Mode {mode_name} missing from MODE_ORDER"

    def test_centered_strong_config(self):
        """centered_strong has correct structural config."""
        from bench_reports.dps_v381_eval import MODES

        cfg = MODES["centered_strong"]["structural"]
        assert cfg.centered_field is True
        assert cfg.pseudo_prior is False
        assert cfg.geometry_strength == 2.0
        assert cfg.normalize_geometry is False

    def test_centered_normalized_config(self):
        """centered_normalized has correct structural config."""
        from bench_reports.dps_v381_eval import MODES

        cfg = MODES["centered_normalized"]["structural"]
        assert cfg.centered_field is True
        assert cfg.normalize_geometry is True
        assert cfg.geometry_strength == 1.0

    def test_centered_prior_normalized_config(self):
        """centered_prior_normalized has correct structural config."""
        from bench_reports.dps_v381_eval import MODES

        cfg = MODES["centered_prior_normalized"]["structural"]
        assert cfg.centered_field is True
        assert cfg.pseudo_prior is True
        assert cfg.normalize_geometry is True

    def test_new_modes_execute(self, H, instance):
        """All three new modes run without error."""
        from bench_reports.dps_v381_eval import run_mode

        instances = [instance]
        for mode_name in ("centered_strong", "centered_normalized",
                          "centered_prior_normalized"):
            result = run_mode(
                mode_name, H, instances,
                max_iters=MAX_ITERS, bp_mode=BP_MODE,
            )
            assert "fer" in result
            assert "frame_errors" in result
            assert result["trials"] == 1

    def test_new_modes_deterministic(self, H, instance):
        """New modes produce identical results on repeated runs."""
        from bench_reports.dps_v381_eval import run_mode

        instances = [instance]
        for mode_name in ("centered_strong", "centered_normalized",
                          "centered_prior_normalized"):
            r1 = run_mode(mode_name, H, instances, max_iters=MAX_ITERS, bp_mode=BP_MODE)
            r2 = run_mode(mode_name, H, instances, max_iters=MAX_ITERS, bp_mode=BP_MODE)
            assert r1["fer"] == r2["fer"], f"Determinism failure for {mode_name}"
            assert r1["frame_errors"] == r2["frame_errors"]

    def test_mode_order_length(self):
        """MODE_ORDER contains all defined modes."""
        from bench_reports.dps_v381_eval import MODES, MODE_ORDER

        assert len(MODE_ORDER) == len(MODES)
        assert set(MODE_ORDER) == set(MODES.keys())
