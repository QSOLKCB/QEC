"""
Tests for v3.8.1 DPS evaluation harness.

Verifies:
- Harness does not modify decoder internals
- RPC modes → added_rows_mean > 0
- Non-RPC modes → added_rows_mean == 0
- Determinism holds (two identical runs produce identical results)
- All 4 modes dispatch correctly
"""

import numpy as np
import pytest

from qec_qldpc_codes import bp_decode, syndrome, channel_llr, create_code
from qec.decoder.rpc import RPCConfig, StructuralConfig

from bench_reports.dps_v381_eval import (
    MODES,
    MODE_ORDER,
    run_mode,
    run_evaluation,
    run_determinism_check,
    compute_dps_slope,
    _pre_generate_instances,
)


# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────

@pytest.fixture
def small_code():
    """Small rate-0.50 code for fast tests."""
    return create_code(name="rate_0.50", lifting_size=3, seed=42)


@pytest.fixture
def small_instances(small_code):
    """Pre-generated instances for the small code."""
    rng = np.random.default_rng(42)
    return _pre_generate_instances(small_code.H_X, 0.02, 20, rng)


# ──────────────────────────────────────────────────────────────────
# Mode definitions
# ──────────────────────────────────────────────────────────────────

class TestModeDefinitions:

    def test_all_modes_defined(self):
        """All expected modes are defined."""
        expected = {
            "baseline", "rpc_only", "geom_v1_only", "rpc_geom",
            "centered", "prior", "centered_prior",
            "geom_centered", "geom_centered_prior",
            "rpc_centered", "rpc_centered_prior",
            "centered_strong", "centered_normalized",
            "centered_prior_normalized",
        }
        assert set(MODES.keys()) == expected
        assert len(MODES) == 14

    def test_mode_order_matches_keys(self):
        """MODE_ORDER lists all mode keys."""
        assert set(MODE_ORDER) == set(MODES.keys())
        assert len(MODE_ORDER) == 14

    def test_baseline_is_flooding_no_rpc(self):
        cfg = MODES["baseline"]
        assert cfg["schedule"] == "flooding"
        assert cfg["structural"].rpc.enabled is False

    def test_rpc_only_is_flooding_with_rpc(self):
        cfg = MODES["rpc_only"]
        assert cfg["schedule"] == "flooding"
        assert cfg["structural"].rpc.enabled is True

    def test_geom_v1_only_is_geom_no_rpc(self):
        cfg = MODES["geom_v1_only"]
        assert cfg["schedule"] == "geom_v1"
        assert cfg["structural"].rpc.enabled is False

    def test_rpc_geom_is_geom_with_rpc(self):
        cfg = MODES["rpc_geom"]
        assert cfg["schedule"] == "geom_v1"
        assert cfg["structural"].rpc.enabled is True


# ──────────────────────────────────────────────────────────────────
# Harness does not modify decoder
# ──────────────────────────────────────────────────────────────────

class TestDecoderInvariance:

    def test_decoder_output_unchanged_after_harness(self, small_code, small_instances):
        """Running the harness does not alter bp_decode behavior."""
        H = small_code.H_X
        inst = small_instances[0]

        # Decode before running harness.
        c_before, i_before = bp_decode(
            H, inst["llr"], max_iters=20, mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
        )

        # Run harness (all modes).
        for mode_name in MODE_ORDER:
            run_mode(mode_name, H, small_instances, max_iters=20)

        # Decode after running harness — must be identical.
        c_after, i_after = bp_decode(
            H, inst["llr"], max_iters=20, mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
        )

        np.testing.assert_array_equal(c_before, c_after)
        assert i_before == i_after

    def test_H_matrix_not_mutated(self, small_code, small_instances):
        """The H matrix is not mutated by the harness."""
        H = small_code.H_X.copy()
        H_original = H.copy()

        for mode_name in MODE_ORDER:
            run_mode(mode_name, H, small_instances, max_iters=20)

        np.testing.assert_array_equal(H, H_original)


# ──────────────────────────────────────────────────────────────────
# RPC activation audit
# ──────────────────────────────────────────────────────────────────

class TestRPCActivation:

    def test_rpc_modes_have_added_rows(self, small_code, small_instances):
        """RPC-enabled modes must have added_rows_mean > 0."""
        H = small_code.H_X
        for mode_name in ("rpc_only", "rpc_geom"):
            result = run_mode(mode_name, H, small_instances, max_iters=20)
            audit = result["audit_summary"]
            assert audit["added_rows_mean"] > 0, (
                f"{mode_name}: expected added_rows_mean > 0, "
                f"got {audit['added_rows_mean']}"
            )

    def test_non_rpc_modes_have_zero_added_rows(self, small_code, small_instances):
        """Non-RPC modes must have added_rows_mean == 0."""
        H = small_code.H_X
        for mode_name in ("baseline", "geom_v1_only"):
            result = run_mode(mode_name, H, small_instances, max_iters=20)
            audit = result["audit_summary"]
            assert audit["added_rows_mean"] == 0, (
                f"{mode_name}: expected added_rows_mean == 0, "
                f"got {audit['added_rows_mean']}"
            )


# ──────────────────────────────────────────────────────────────────
# Determinism
# ──────────────────────────────────────────────────────────────────

class TestDeterminism:

    def test_determinism_check_passes(self):
        """Built-in determinism check must pass."""
        result = run_determinism_check(
            seed=42, distance=3, p=0.02, trials=20, max_iters=20,
        )
        assert result["passed"], (
            f"Determinism check failed: "
            f"fer_match={result['fer_match']}, "
            f"audit_match={result['audit_match']}"
        )

    def test_run_mode_deterministic(self, small_code):
        """Two identical run_mode calls produce identical FER and audit."""
        H = small_code.H_X
        results = []
        for _ in range(2):
            rng = np.random.default_rng(42)
            instances = _pre_generate_instances(H, 0.02, 20, rng)
            r = run_mode("baseline", H, instances, max_iters=20)
            results.append(r)

        assert results[0]["fer"] == results[1]["fer"]
        assert results[0]["audit_summary"] == results[1]["audit_summary"]

    def test_all_modes_deterministic(self, small_code):
        """All modes produce deterministic results across two runs."""
        H = small_code.H_X
        for mode_name in MODE_ORDER:
            results = []
            for _ in range(2):
                rng = np.random.default_rng(42)
                instances = _pre_generate_instances(H, 0.02, 20, rng)
                r = run_mode(mode_name, H, instances, max_iters=20)
                results.append(r)
            assert results[0]["fer"] == results[1]["fer"], (
                f"{mode_name}: FER mismatch"
            )


# ──────────────────────────────────────────────────────────────────
# All 4 modes dispatch correctly
# ──────────────────────────────────────────────────────────────────

class TestModeDispatch:

    def test_all_modes_run(self, small_code, small_instances):
        """All 4 modes run without error and produce valid output."""
        H = small_code.H_X
        for mode_name in MODE_ORDER:
            result = run_mode(mode_name, H, small_instances, max_iters=20)
            assert "fer" in result
            assert "frame_errors" in result
            assert "trials" in result
            assert "audit_summary" in result
            assert result["trials"] == len(small_instances)
            assert 0.0 <= result["fer"] <= 1.0

    def test_baseline_uses_flooding(self, small_code, small_instances):
        """Baseline mode uses flooding schedule (verified by mode config)."""
        assert MODES["baseline"]["schedule"] == "flooding"
        # Also verify it runs.
        H = small_code.H_X
        result = run_mode("baseline", H, small_instances, max_iters=20)
        assert result["trials"] == len(small_instances)

    def test_geom_v1_mode_runs(self, small_code, small_instances):
        """geom_v1 modes run without error."""
        H = small_code.H_X
        for mode_name in ("geom_v1_only", "rpc_geom"):
            result = run_mode(mode_name, H, small_instances, max_iters=20)
            assert result["trials"] == len(small_instances)


# ──────────────────────────────────────────────────────────────────
# DPS slope computation
# ──────────────────────────────────────────────────────────────────

class TestDPSSlope:

    def test_slope_with_constant_fer(self):
        """Constant FER across distances → slope ≈ 0."""
        fer_by_d = {3: 0.5, 5: 0.5, 7: 0.5}
        slope = compute_dps_slope(fer_by_d)
        assert abs(slope) < 1e-10

    def test_slope_with_decreasing_fer(self):
        """Decreasing FER → negative slope."""
        fer_by_d = {3: 0.5, 5: 0.1, 7: 0.01}
        slope = compute_dps_slope(fer_by_d)
        assert slope < 0

    def test_slope_with_increasing_fer(self):
        """Increasing FER → positive slope (inverted)."""
        fer_by_d = {3: 0.01, 5: 0.1, 7: 0.5}
        slope = compute_dps_slope(fer_by_d)
        assert slope > 0

    def test_slope_single_distance(self):
        """Single distance → slope = 0."""
        fer_by_d = {3: 0.5}
        slope = compute_dps_slope(fer_by_d)
        assert slope == 0.0


# ──────────────────────────────────────────────────────────────────
# Full evaluation (small scale)
# ──────────────────────────────────────────────────────────────────

class TestFullEvaluation:

    def test_evaluation_runs(self):
        """Full evaluation with small parameters runs without error."""
        result = run_evaluation(
            seed=42,
            distances=[3, 5],
            p_values=[0.02],
            trials=10,
            max_iters=20,
        )
        assert "results" in result
        assert "slopes" in result
        assert "audits" in result
        assert "config" in result

        # All 4 modes should be present.
        for mode_name in MODE_ORDER:
            assert mode_name in result["results"]
            assert mode_name in result["slopes"]
            assert mode_name in result["audits"]

    def test_evaluation_slopes_are_floats(self):
        """All slopes are finite floats."""
        result = run_evaluation(
            seed=42,
            distances=[3, 5],
            p_values=[0.02],
            trials=10,
            max_iters=20,
        )
        for mode_name in MODE_ORDER:
            for p, slope in result["slopes"][mode_name].items():
                assert isinstance(slope, float)
                assert np.isfinite(slope)
