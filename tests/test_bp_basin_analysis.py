"""
Tests for BP basin-of-attraction analysis (v4.9.0).

Validates:
  - Correct basin: all perturbations converge to correct fixed point
  - Mixed basin: perturbations split between correct and incorrect
  - Boundary detection: basin_boundary_eps detected when classification flips
  - Determinism: identical outputs on repeated runs
  - JSON serializability of all outputs
"""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from qec.diagnostics.bp_basin_analysis import (
    compute_bp_basin_analysis,
    DEFAULT_EPS_VALUES,
    DEFAULT_PERTURBATION_PATTERNS,
)


# ── Mock helpers ─────────────────────────────────────────────────────

# We mock bp_decode and syndrome to avoid needing the full decoder stack.
# This lets us test basin analysis logic in isolation.


def _make_mock_decode_always_correct(H, n_vars, max_iters):
    """Return a mock bp_decode that always produces correct decoding.

    Returns (correction, iters, llr_history, energy_trace) where
    correction yields zero syndrome residual.
    """
    def mock_bp_decode(H_arg, llr_arg, **kwargs):
        correction = np.zeros(n_vars, dtype=np.uint8)
        iters = 10
        # Build plausible LLR history: monotonically growing magnitudes.
        llr_hist = np.tile(llr_arg, (max_iters, 1)) * np.linspace(
            1.0, 3.0, max_iters,
        ).reshape(-1, 1)
        # Energy trace: monotonically decreasing, stabilizing.
        energy_trace = [max(1.0, 100.0 - 10.0 * t) for t in range(max_iters)]
        return (correction, iters, llr_hist, energy_trace)
    return mock_bp_decode


def _make_mock_decode_flip_at_eps(H, n_vars, max_iters, flip_eps):
    """Return a mock bp_decode that produces incorrect decoding when
    perturbation magnitude >= flip_eps."""
    def mock_bp_decode(H_arg, llr_arg, **kwargs):
        # Detect perturbation magnitude by comparing to a baseline.
        # We use mean absolute deviation from the original LLR.
        iters = 10
        llr_hist = np.tile(llr_arg, (max_iters, 1)) * np.linspace(
            1.0, 3.0, max_iters,
        ).reshape(-1, 1)
        energy_trace = [max(1.0, 100.0 - 10.0 * t) for t in range(max_iters)]

        # Check if perturbation is large enough to flip.
        # We embed the flip threshold in the correction: nonzero correction
        # produces nonzero syndrome residual.
        perturbation_mag = float(np.max(np.abs(llr_arg - _baseline_llr[0])))
        if perturbation_mag >= flip_eps - 1e-12:
            correction = np.ones(n_vars, dtype=np.uint8)
        else:
            correction = np.zeros(n_vars, dtype=np.uint8)
        return (correction, iters, llr_hist, energy_trace)
    return mock_bp_decode


# Storage for baseline LLR (used by flip mock).
_baseline_llr = [None]


def _mock_syndrome_zero(H, correction):
    """Mock syndrome that returns all zeros (correct decode)."""
    return np.zeros(H.shape[0], dtype=np.uint8)


def _mock_syndrome_from_correction(H, correction):
    """Mock syndrome: zero if correction is zero, nonzero otherwise."""
    if np.any(correction):
        s = np.ones(H.shape[0], dtype=np.uint8)
    else:
        s = np.zeros(H.shape[0], dtype=np.uint8)
    return s


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def small_H():
    """Small parity-check matrix for testing."""
    return np.array([
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
    ], dtype=np.uint8)


@pytest.fixture
def small_llr():
    """Small LLR vector."""
    return np.array([2.0, -1.5, 0.8, -0.3, 1.2], dtype=np.float64)


@pytest.fixture
def small_syndrome(small_H):
    """Zero syndrome (no error)."""
    return np.zeros(small_H.shape[0], dtype=np.uint8)


# ── Tests ────────────────────────────────────────────────────────────


class TestCorrectBasin:
    """All perturbations converge to correct fixed point."""

    @patch("src.qec_qldpc_codes.syndrome",
           side_effect=_mock_syndrome_zero)
    @patch("src.qec_qldpc_codes.bp_decode")
    def test_all_correct(self, mock_decode, mock_synd,
                         small_H, small_llr, small_syndrome):
        n_vars = small_H.shape[1]
        max_iters = 20
        mock_decode.side_effect = _make_mock_decode_always_correct(
            small_H, n_vars, max_iters,
        )

        result = compute_bp_basin_analysis(
            H=small_H,
            llr=small_llr,
            baseline_fixed_point_type="correct_fixed_point",
            max_iters=max_iters,
            bp_mode="min_sum",
            schedule="flooding",
            syndrome_vec=small_syndrome,
            syndrome_original=small_syndrome,
        )

        expected_total = len(DEFAULT_EPS_VALUES) * len(DEFAULT_PERTURBATION_PATTERNS)
        assert result["num_perturbations"] == expected_total
        assert result["num_correct"] == expected_total
        assert result["num_incorrect"] == 0
        assert result["num_degenerate"] == 0
        assert result["basin_correct_probability"] == 1.0
        assert result["basin_incorrect_probability"] == 0.0
        assert result["basin_boundary_eps"] is None

    @patch("src.qec_qldpc_codes.syndrome",
           side_effect=_mock_syndrome_zero)
    @patch("src.qec_qldpc_codes.bp_decode")
    def test_sorted_keys(self, mock_decode, mock_synd,
                         small_H, small_llr, small_syndrome):
        n_vars = small_H.shape[1]
        max_iters = 20
        mock_decode.side_effect = _make_mock_decode_always_correct(
            small_H, n_vars, max_iters,
        )

        result = compute_bp_basin_analysis(
            H=small_H,
            llr=small_llr,
            baseline_fixed_point_type="correct_fixed_point",
            max_iters=max_iters,
            bp_mode="min_sum",
            schedule="flooding",
            syndrome_vec=small_syndrome,
            syndrome_original=small_syndrome,
        )
        keys = list(result.keys())
        assert keys == sorted(keys)


class TestMixedBasin:
    """Perturbations split between correct and incorrect."""

    @patch("src.qec_qldpc_codes.syndrome",
           side_effect=_mock_syndrome_from_correction)
    @patch("src.qec_qldpc_codes.bp_decode")
    def test_mixed_results(self, mock_decode, mock_synd,
                           small_H, small_llr, small_syndrome):
        n_vars = small_H.shape[1]
        max_iters = 20
        # Flip at eps=1e-3 — so smaller perturbations remain correct,
        # larger ones become incorrect.
        flip_eps = 1e-3
        _baseline_llr[0] = small_llr.copy()
        mock_decode.side_effect = _make_mock_decode_flip_at_eps(
            small_H, n_vars, max_iters, flip_eps,
        )

        result = compute_bp_basin_analysis(
            H=small_H,
            llr=small_llr,
            baseline_fixed_point_type="correct_fixed_point",
            max_iters=max_iters,
            bp_mode="min_sum",
            schedule="flooding",
            syndrome_vec=small_syndrome,
            syndrome_original=small_syndrome,
        )

        # Should have a mix of correct and incorrect.
        assert result["num_correct"] > 0
        assert result["num_incorrect"] > 0
        assert result["num_correct"] + result["num_incorrect"] == result["num_perturbations"]
        assert result["basin_correct_probability"] > 0.0
        assert result["basin_incorrect_probability"] > 0.0
        # Sum of probabilities must equal 1.
        total_prob = (
            result["basin_correct_probability"]
            + result["basin_incorrect_probability"]
            + result["basin_degenerate_probability"]
            + result["basin_no_convergence_probability"]
        )
        assert abs(total_prob - 1.0) < 1e-12


class TestBoundaryDetection:
    """Verify basin_boundary_eps detected when classification flips."""

    @patch("src.qec_qldpc_codes.syndrome",
           side_effect=_mock_syndrome_from_correction)
    @patch("src.qec_qldpc_codes.bp_decode")
    def test_boundary_detected(self, mock_decode, mock_synd,
                               small_H, small_llr, small_syndrome):
        n_vars = small_H.shape[1]
        max_iters = 20
        # Flip at eps=2e-3: perturbations with |pattern * eps| >= 2e-3 flip.
        flip_eps = 2e-3
        _baseline_llr[0] = small_llr.copy()
        mock_decode.side_effect = _make_mock_decode_flip_at_eps(
            small_H, n_vars, max_iters, flip_eps,
        )

        result = compute_bp_basin_analysis(
            H=small_H,
            llr=small_llr,
            baseline_fixed_point_type="correct_fixed_point",
            max_iters=max_iters,
            bp_mode="min_sum",
            schedule="flooding",
            syndrome_vec=small_syndrome,
            syndrome_original=small_syndrome,
        )

        # Boundary should be detected.
        assert result["basin_boundary_eps"] is not None
        assert isinstance(result["basin_boundary_eps"], float)
        assert result["basin_boundary_eps"] > 0.0

    @patch("src.qec_qldpc_codes.syndrome",
           side_effect=_mock_syndrome_zero)
    @patch("src.qec_qldpc_codes.bp_decode")
    def test_no_boundary_when_stable(self, mock_decode, mock_synd,
                                     small_H, small_llr, small_syndrome):
        """No boundary detected when all perturbations match baseline."""
        n_vars = small_H.shape[1]
        max_iters = 20
        mock_decode.side_effect = _make_mock_decode_always_correct(
            small_H, n_vars, max_iters,
        )

        result = compute_bp_basin_analysis(
            H=small_H,
            llr=small_llr,
            baseline_fixed_point_type="correct_fixed_point",
            max_iters=max_iters,
            bp_mode="min_sum",
            schedule="flooding",
            syndrome_vec=small_syndrome,
            syndrome_original=small_syndrome,
        )

        assert result["basin_boundary_eps"] is None


class TestDeterminism:
    """Repeated runs must produce identical outputs."""

    @patch("src.qec_qldpc_codes.syndrome",
           side_effect=_mock_syndrome_zero)
    @patch("src.qec_qldpc_codes.bp_decode")
    def test_determinism(self, mock_decode, mock_synd,
                         small_H, small_llr, small_syndrome):
        n_vars = small_H.shape[1]
        max_iters = 20
        mock_decode.side_effect = _make_mock_decode_always_correct(
            small_H, n_vars, max_iters,
        )

        kwargs = dict(
            H=small_H,
            llr=small_llr,
            baseline_fixed_point_type="correct_fixed_point",
            max_iters=max_iters,
            bp_mode="min_sum",
            schedule="flooding",
            syndrome_vec=small_syndrome,
            syndrome_original=small_syndrome,
        )
        r1 = compute_bp_basin_analysis(**kwargs)

        # Reset mock for second run.
        mock_decode.side_effect = _make_mock_decode_always_correct(
            small_H, n_vars, max_iters,
        )
        r2 = compute_bp_basin_analysis(**kwargs)

        assert r1 == r2


class TestJSONSerializable:
    """All outputs must be JSON-serializable."""

    @patch("src.qec_qldpc_codes.syndrome",
           side_effect=_mock_syndrome_zero)
    @patch("src.qec_qldpc_codes.bp_decode")
    def test_json_serializable(self, mock_decode, mock_synd,
                               small_H, small_llr, small_syndrome):
        n_vars = small_H.shape[1]
        max_iters = 20
        mock_decode.side_effect = _make_mock_decode_always_correct(
            small_H, n_vars, max_iters,
        )

        result = compute_bp_basin_analysis(
            H=small_H,
            llr=small_llr,
            baseline_fixed_point_type="correct_fixed_point",
            max_iters=max_iters,
            bp_mode="min_sum",
            schedule="flooding",
            syndrome_vec=small_syndrome,
            syndrome_original=small_syndrome,
        )

        serialized = json.dumps(result, sort_keys=True)
        deserialized = json.loads(serialized)
        assert deserialized["basin_correct_probability"] == result["basin_correct_probability"]
        assert deserialized["num_perturbations"] == result["num_perturbations"]

    @patch("src.qec_qldpc_codes.syndrome",
           side_effect=_mock_syndrome_from_correction)
    @patch("src.qec_qldpc_codes.bp_decode")
    def test_json_serializable_with_boundary(self, mock_decode, mock_synd,
                                             small_H, small_llr, small_syndrome):
        """JSON serializable even when basin_boundary_eps is not None."""
        n_vars = small_H.shape[1]
        max_iters = 20
        flip_eps = 1e-3
        _baseline_llr[0] = small_llr.copy()
        mock_decode.side_effect = _make_mock_decode_flip_at_eps(
            small_H, n_vars, max_iters, flip_eps,
        )

        result = compute_bp_basin_analysis(
            H=small_H,
            llr=small_llr,
            baseline_fixed_point_type="correct_fixed_point",
            max_iters=max_iters,
            bp_mode="min_sum",
            schedule="flooding",
            syndrome_vec=small_syndrome,
            syndrome_original=small_syndrome,
        )

        serialized = json.dumps(result, sort_keys=True)
        deserialized = json.loads(serialized)
        assert deserialized == result


class TestCustomEpsValues:
    """Custom epsilon values and perturbation patterns."""

    @patch("src.qec_qldpc_codes.syndrome",
           side_effect=_mock_syndrome_zero)
    @patch("src.qec_qldpc_codes.bp_decode")
    def test_custom_eps(self, mock_decode, mock_synd,
                        small_H, small_llr, small_syndrome):
        n_vars = small_H.shape[1]
        max_iters = 20
        mock_decode.side_effect = _make_mock_decode_always_correct(
            small_H, n_vars, max_iters,
        )

        custom_eps = [0.01, 0.02]
        custom_patterns = [1.0, -1.0]

        result = compute_bp_basin_analysis(
            H=small_H,
            llr=small_llr,
            baseline_fixed_point_type="correct_fixed_point",
            max_iters=max_iters,
            bp_mode="min_sum",
            schedule="flooding",
            syndrome_vec=small_syndrome,
            syndrome_original=small_syndrome,
            eps_values=custom_eps,
            perturbation_patterns=custom_patterns,
        )

        expected_total = len(custom_eps) * len(custom_patterns)
        assert result["num_perturbations"] == expected_total
