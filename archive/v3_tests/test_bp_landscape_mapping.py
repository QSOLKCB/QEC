"""
Tests for BP attractor landscape mapping (v5.0.0).

Validates:
  - Single attractor: all perturbations converge to same attractor
  - Multiple attractors: perturbations split across attractors
  - Pseudocodeword detection: incorrect fixed point stable under perturbation
  - Determinism: identical outputs on repeated runs
  - JSON serializability of all outputs
"""

from __future__ import annotations

import json
from unittest.mock import patch
from zlib import crc32

import numpy as np
import pytest

from qec.diagnostics.bp_landscape_mapping import (
    compute_bp_landscape_map,
    _compute_attractor_id,
    DEFAULT_EPS_VALUES,
    DEFAULT_PERTURBATION_PATTERNS,
)


# ── Mock helpers ─────────────────────────────────────────────────────

# We mock bp_decode and syndrome to avoid needing the full decoder stack.


def _make_mock_decode_single_attractor(H, n_vars, max_iters):
    """Mock bp_decode that always produces the same final LLR sign pattern
    and correct decoding (zero syndrome residual)."""
    def mock_bp_decode(H_arg, llr_arg, **kwargs):
        correction = np.zeros(n_vars, dtype=np.uint8)
        iters = 10
        # All runs produce the same final LLR sign pattern regardless
        # of perturbation — single attractor.
        # Use varied magnitudes to avoid degenerate classification.
        final_llr = np.array([3.5, -2.1, 1.7, -4.0, 0.9], dtype=np.float64)
        llr_hist = np.tile(final_llr, (max_iters, 1))
        energy_trace = [max(1.0, 100.0 - 10.0 * t) for t in range(max_iters)]
        return (correction, iters, llr_hist, energy_trace)
    return mock_bp_decode


def _make_mock_decode_two_attractors(H, n_vars, max_iters):
    """Mock bp_decode that produces two distinct attractor sign patterns.

    First half of calls produce attractor A, second half produce attractor B.
    Both are correct decodes.  Uses varied magnitudes to avoid degenerate
    classification.
    """
    _call_count = [0]

    def mock_bp_decode(H_arg, llr_arg, **kwargs):
        correction = np.zeros(n_vars, dtype=np.uint8)
        iters = 10
        idx = _call_count[0]
        _call_count[0] += 1
        # Alternate between two attractors based on call order.
        total_perturbations = len(DEFAULT_EPS_VALUES) * len(DEFAULT_PERTURBATION_PATTERNS)
        if idx < total_perturbations // 2:
            final_llr = np.array([3.5, -2.1, 1.7, -4.0, 0.9], dtype=np.float64)
        else:
            final_llr = np.array([-2.8, 1.5, -3.2, 0.7, -4.1], dtype=np.float64)
        llr_hist = np.tile(final_llr, (max_iters, 1))
        energy_trace = [max(1.0, 100.0 - 10.0 * t) for t in range(max_iters)]
        return (correction, iters, llr_hist, energy_trace)
    return mock_bp_decode


def _make_mock_decode_pseudocodeword(H, n_vars, max_iters):
    """Mock bp_decode that produces an incorrect fixed point that is stable
    under small perturbations (pseudocodeword behavior).

    All runs produce the same sign pattern and nonzero syndrome residual.
    Uses varied magnitudes to avoid degenerate classification.
    """
    def mock_bp_decode(H_arg, llr_arg, **kwargs):
        # Nonzero correction → nonzero syndrome residual → incorrect FP.
        correction = np.ones(n_vars, dtype=np.uint8)
        iters = 10
        final_llr = np.array([3.5, -2.1, 1.7, -4.0, 0.9], dtype=np.float64)
        llr_hist = np.tile(final_llr, (max_iters, 1))
        energy_trace = [max(1.0, 100.0 - 10.0 * t) for t in range(max_iters)]
        return (correction, iters, llr_hist, energy_trace)
    return mock_bp_decode


def _make_mock_decode_unstable_incorrect(H, n_vars, max_iters):
    """Mock bp_decode where the first perturbations produce incorrect FP
    but pseudocodeword probing escapes to correct FP.

    Main landscape perturbations: all incorrect.
    Pseudocodeword probe perturbations: escape to correct.
    Uses varied magnitudes to avoid degenerate classification.
    """
    _total_calls = [0]
    _landscape_calls = len(DEFAULT_EPS_VALUES) * len(DEFAULT_PERTURBATION_PATTERNS)

    def mock_bp_decode(H_arg, llr_arg, **kwargs):
        idx = _total_calls[0]
        _total_calls[0] += 1
        iters = 10
        final_llr = np.array([3.5, -2.1, 1.7, -4.0, 0.9], dtype=np.float64)
        llr_hist = np.tile(final_llr, (max_iters, 1))
        energy_trace = [max(1.0, 100.0 - 10.0 * t) for t in range(max_iters)]

        if idx < _landscape_calls:
            # Landscape phase: incorrect fixed point.
            correction = np.ones(n_vars, dtype=np.uint8)
        else:
            # Pseudocodeword probe phase: escapes to correct.
            correction = np.zeros(n_vars, dtype=np.uint8)
        return (correction, iters, llr_hist, energy_trace)
    return mock_bp_decode


def _mock_syndrome_zero(H, correction):
    """Mock syndrome that returns all zeros (correct decode)."""
    return np.zeros(H.shape[0], dtype=np.uint8)


def _mock_syndrome_from_correction(H, correction):
    """Mock syndrome: zero if correction is zero, nonzero otherwise."""
    if np.any(correction):
        return np.ones(H.shape[0], dtype=np.uint8)
    return np.zeros(H.shape[0], dtype=np.uint8)


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


class TestSingleAttractor:
    """All perturbations converge to the same attractor."""

    @patch("src.qec_qldpc_codes.syndrome",
           side_effect=_mock_syndrome_zero)
    @patch("src.qec_qldpc_codes.bp_decode")
    def test_single_attractor(self, mock_decode, mock_synd,
                              small_H, small_llr, small_syndrome):
        n_vars = small_H.shape[1]
        max_iters = 20
        mock_decode.side_effect = _make_mock_decode_single_attractor(
            small_H, n_vars, max_iters,
        )

        result = compute_bp_landscape_map(
            H=small_H,
            llr=small_llr,
            max_iters=max_iters,
            bp_mode="min_sum",
            schedule="flooding",
            syndrome_vec=small_syndrome,
            syndrome_original=small_syndrome,
        )

        assert result["num_attractors"] == 1
        assert result["largest_basin_fraction"] == 1.0
        assert result["correct_attractor_fraction"] == 1.0
        assert result["incorrect_attractor_fraction"] == 0.0
        assert result["degenerate_attractor_fraction"] == 0.0
        assert result["num_pseudocodewords"] == 0
        assert result["pseudocodeword_fraction"] == 0.0
        assert result["pseudocodeword_ids"] == []
        assert len(result["attractor_distribution"]) == 1

    @patch("src.qec_qldpc_codes.syndrome",
           side_effect=_mock_syndrome_zero)
    @patch("src.qec_qldpc_codes.bp_decode")
    def test_sorted_keys(self, mock_decode, mock_synd,
                         small_H, small_llr, small_syndrome):
        n_vars = small_H.shape[1]
        max_iters = 20
        mock_decode.side_effect = _make_mock_decode_single_attractor(
            small_H, n_vars, max_iters,
        )

        result = compute_bp_landscape_map(
            H=small_H,
            llr=small_llr,
            max_iters=max_iters,
            bp_mode="min_sum",
            schedule="flooding",
            syndrome_vec=small_syndrome,
            syndrome_original=small_syndrome,
        )
        keys = list(result.keys())
        assert keys == sorted(keys)


class TestMultipleAttractors:
    """Perturbations split across multiple attractors."""

    @patch("src.qec_qldpc_codes.syndrome",
           side_effect=_mock_syndrome_zero)
    @patch("src.qec_qldpc_codes.bp_decode")
    def test_two_attractors(self, mock_decode, mock_synd,
                            small_H, small_llr, small_syndrome):
        n_vars = small_H.shape[1]
        max_iters = 20
        mock_decode.side_effect = _make_mock_decode_two_attractors(
            small_H, n_vars, max_iters,
        )

        result = compute_bp_landscape_map(
            H=small_H,
            llr=small_llr,
            max_iters=max_iters,
            bp_mode="min_sum",
            schedule="flooding",
            syndrome_vec=small_syndrome,
            syndrome_original=small_syndrome,
        )

        assert result["num_attractors"] == 2
        assert len(result["attractor_distribution"]) == 2
        # Largest basin should be at most 1.0 and at least 0.5.
        assert 0.5 <= result["largest_basin_fraction"] <= 1.0
        # Total fractions should sum to 1.
        total_frac = sum(
            info["fraction"]
            for info in result["attractor_distribution"].values()
        )
        assert abs(total_frac - 1.0) < 1e-12

    @patch("src.qec_qldpc_codes.syndrome",
           side_effect=_mock_syndrome_zero)
    @patch("src.qec_qldpc_codes.bp_decode")
    def test_attractor_distribution_structure(self, mock_decode, mock_synd,
                                              small_H, small_llr,
                                              small_syndrome):
        n_vars = small_H.shape[1]
        max_iters = 20
        mock_decode.side_effect = _make_mock_decode_two_attractors(
            small_H, n_vars, max_iters,
        )

        result = compute_bp_landscape_map(
            H=small_H,
            llr=small_llr,
            max_iters=max_iters,
            bp_mode="min_sum",
            schedule="flooding",
            syndrome_vec=small_syndrome,
            syndrome_original=small_syndrome,
        )

        for aid_str, info in result["attractor_distribution"].items():
            assert "type" in info
            assert "count" in info
            assert "fraction" in info
            assert isinstance(info["count"], int)
            assert isinstance(info["fraction"], float)
            assert isinstance(info["type"], str)


class TestPseudocodewordDetection:
    """Incorrect fixed point stable under perturbation → pseudocodeword."""

    @patch("src.qec_qldpc_codes.syndrome",
           side_effect=_mock_syndrome_from_correction)
    @patch("src.qec_qldpc_codes.bp_decode")
    def test_pseudocodeword_detected(self, mock_decode, mock_synd,
                                     small_H, small_llr, small_syndrome):
        n_vars = small_H.shape[1]
        max_iters = 20
        mock_decode.side_effect = _make_mock_decode_pseudocodeword(
            small_H, n_vars, max_iters,
        )

        result = compute_bp_landscape_map(
            H=small_H,
            llr=small_llr,
            max_iters=max_iters,
            bp_mode="min_sum",
            schedule="flooding",
            syndrome_vec=small_syndrome,
            syndrome_original=small_syndrome,
        )

        assert result["num_pseudocodewords"] >= 1
        assert result["pseudocodeword_fraction"] > 0.0
        assert len(result["pseudocodeword_ids"]) >= 1
        assert result["incorrect_attractor_fraction"] > 0.0

    @patch("src.qec_qldpc_codes.syndrome",
           side_effect=_mock_syndrome_from_correction)
    @patch("src.qec_qldpc_codes.bp_decode")
    def test_unstable_incorrect_not_pseudocodeword(self, mock_decode, mock_synd,
                                                    small_H, small_llr,
                                                    small_syndrome):
        """Incorrect FP that escapes under small perturbation is NOT a
        pseudocodeword."""
        n_vars = small_H.shape[1]
        max_iters = 20
        mock_decode.side_effect = _make_mock_decode_unstable_incorrect(
            small_H, n_vars, max_iters,
        )

        result = compute_bp_landscape_map(
            H=small_H,
            llr=small_llr,
            max_iters=max_iters,
            bp_mode="min_sum",
            schedule="flooding",
            syndrome_vec=small_syndrome,
            syndrome_original=small_syndrome,
        )

        assert result["num_pseudocodewords"] == 0
        assert result["pseudocodeword_fraction"] == 0.0
        assert result["pseudocodeword_ids"] == []


class TestDeterminism:
    """Repeated runs must produce identical outputs."""

    @patch("src.qec_qldpc_codes.syndrome",
           side_effect=_mock_syndrome_zero)
    @patch("src.qec_qldpc_codes.bp_decode")
    def test_determinism(self, mock_decode, mock_synd,
                         small_H, small_llr, small_syndrome):
        n_vars = small_H.shape[1]
        max_iters = 20
        mock_decode.side_effect = _make_mock_decode_single_attractor(
            small_H, n_vars, max_iters,
        )

        kwargs = dict(
            H=small_H,
            llr=small_llr,
            max_iters=max_iters,
            bp_mode="min_sum",
            schedule="flooding",
            syndrome_vec=small_syndrome,
            syndrome_original=small_syndrome,
        )
        r1 = compute_bp_landscape_map(**kwargs)

        # Reset mock for second run.
        mock_decode.side_effect = _make_mock_decode_single_attractor(
            small_H, n_vars, max_iters,
        )
        r2 = compute_bp_landscape_map(**kwargs)

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
        mock_decode.side_effect = _make_mock_decode_single_attractor(
            small_H, n_vars, max_iters,
        )

        result = compute_bp_landscape_map(
            H=small_H,
            llr=small_llr,
            max_iters=max_iters,
            bp_mode="min_sum",
            schedule="flooding",
            syndrome_vec=small_syndrome,
            syndrome_original=small_syndrome,
        )

        serialized = json.dumps(result, sort_keys=True)
        deserialized = json.loads(serialized)
        assert deserialized["num_attractors"] == result["num_attractors"]
        assert deserialized["num_pseudocodewords"] == result["num_pseudocodewords"]
        assert deserialized["largest_basin_fraction"] == result["largest_basin_fraction"]

    @patch("src.qec_qldpc_codes.syndrome",
           side_effect=_mock_syndrome_from_correction)
    @patch("src.qec_qldpc_codes.bp_decode")
    def test_json_serializable_with_pseudocodewords(self, mock_decode, mock_synd,
                                                     small_H, small_llr,
                                                     small_syndrome):
        n_vars = small_H.shape[1]
        max_iters = 20
        mock_decode.side_effect = _make_mock_decode_pseudocodeword(
            small_H, n_vars, max_iters,
        )

        result = compute_bp_landscape_map(
            H=small_H,
            llr=small_llr,
            max_iters=max_iters,
            bp_mode="min_sum",
            schedule="flooding",
            syndrome_vec=small_syndrome,
            syndrome_original=small_syndrome,
        )

        serialized = json.dumps(result, sort_keys=True)
        deserialized = json.loads(serialized)
        assert deserialized == result


class TestAttractorId:
    """Attractor ID computation using CRC32."""

    def test_same_sign_pattern_same_id(self):
        """Identical sign patterns produce identical attractor IDs."""
        llr1 = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
        llr2 = np.array([0.5, -0.1, 10.0, -0.001, 99.0])
        # Same sign pattern.
        assert _compute_attractor_id(llr1) == _compute_attractor_id(llr2)

    def test_different_sign_pattern_different_id(self):
        """Different sign patterns produce different attractor IDs."""
        llr1 = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
        llr2 = np.array([-1.0, 2.0, -3.0, 4.0, -5.0])
        assert _compute_attractor_id(llr1) != _compute_attractor_id(llr2)

    def test_deterministic(self):
        """Same input produces same ID across calls."""
        llr = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
        id1 = _compute_attractor_id(llr)
        id2 = _compute_attractor_id(llr)
        assert id1 == id2

    def test_non_negative(self):
        """Attractor ID is a non-negative integer."""
        llr = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
        aid = _compute_attractor_id(llr)
        assert isinstance(aid, int)
        assert aid >= 0
