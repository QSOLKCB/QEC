"""
Deterministic tests for ternary trapping diagnostics.

Tests verify:
- correct detection of zero regions
- frustration metric correctness
- persistent zero detection
- trapping indicator computation
- deterministic outputs
- integration with ternary decoder
"""

from __future__ import annotations

import numpy as np
import pytest

from src.qec.decoder.ternary.ternary_trapping import (
    detect_zero_regions,
    compute_frustration_index,
    detect_persistent_zero_states,
    estimate_trapping_indicator,
)


# ── detect_zero_regions ──────────────────────────────────────────────


class TestDetectZeroRegions:
    """Tests for zero-region detection."""

    def test_no_zeros(self):
        msgs = np.array([1, -1, 1, -1], dtype=np.int8)
        result = detect_zero_regions(msgs)
        assert result["region_ids"] == []
        assert result["region_sizes"] == []
        assert result["node_indices"] == []

    def test_all_zeros(self):
        msgs = np.array([0, 0, 0, 0], dtype=np.int8)
        result = detect_zero_regions(msgs)
        assert result["region_ids"] == [0]
        assert result["region_sizes"] == [4]
        assert result["node_indices"] == [[0, 1, 2, 3]]

    def test_single_zero(self):
        msgs = np.array([1, 0, -1], dtype=np.int8)
        result = detect_zero_regions(msgs)
        assert result["region_ids"] == [0]
        assert result["region_sizes"] == [1]
        assert result["node_indices"] == [[1]]

    def test_multiple_regions(self):
        msgs = np.array([0, 0, 1, 0, -1, 0, 0], dtype=np.int8)
        result = detect_zero_regions(msgs)
        assert result["region_ids"] == [0, 1, 2]
        assert result["region_sizes"] == [2, 1, 2]
        assert result["node_indices"] == [[0, 1], [3], [5, 6]]

    def test_empty_input(self):
        msgs = np.array([], dtype=np.int8)
        result = detect_zero_regions(msgs)
        assert result["region_ids"] == []
        assert result["region_sizes"] == []
        assert result["node_indices"] == []

    def test_determinism(self):
        msgs = np.array([0, 1, 0, 0, -1, 0], dtype=np.int8)
        r1 = detect_zero_regions(msgs)
        r2 = detect_zero_regions(msgs)
        assert r1 == r2

    def test_node_indices_sorted(self):
        msgs = np.array([0, 0, 0], dtype=np.int8)
        result = detect_zero_regions(msgs)
        for indices in result["node_indices"]:
            assert indices == sorted(indices)

    def test_2d_input_flattened(self):
        msgs = np.array([[0, 1], [0, 0]], dtype=np.int8)
        result = detect_zero_regions(msgs)
        # Flattened: [0, 1, 0, 0]
        assert result["region_ids"] == [0, 1]
        assert result["region_sizes"] == [1, 2]


# ── compute_frustration_index ────────────────────────────────────────


class TestComputeFrustrationIndex:
    """Tests for frustration index metric."""

    def test_all_decided(self):
        msgs = np.array([1, -1, 1, -1], dtype=np.int8)
        fi = compute_frustration_index(msgs)
        assert fi == np.float64(0.0)
        assert isinstance(fi, np.float64)

    def test_all_undecided(self):
        msgs = np.array([0, 0, 0, 0], dtype=np.int8)
        fi = compute_frustration_index(msgs)
        assert fi == np.float64(1.0)

    def test_half_undecided(self):
        msgs = np.array([1, 0, -1, 0], dtype=np.int8)
        fi = compute_frustration_index(msgs)
        assert fi == np.float64(0.5)

    def test_empty(self):
        msgs = np.array([], dtype=np.int8)
        fi = compute_frustration_index(msgs)
        assert fi == np.float64(0.0)

    def test_single_zero(self):
        msgs = np.array([0], dtype=np.int8)
        fi = compute_frustration_index(msgs)
        assert fi == np.float64(1.0)

    def test_single_nonzero(self):
        msgs = np.array([1], dtype=np.int8)
        fi = compute_frustration_index(msgs)
        assert fi == np.float64(0.0)

    def test_determinism(self):
        msgs = np.array([1, 0, -1, 0, 1], dtype=np.int8)
        assert compute_frustration_index(msgs) == compute_frustration_index(msgs)

    def test_returns_float64(self):
        msgs = np.array([0, 1, 0], dtype=np.int8)
        fi = compute_frustration_index(msgs)
        assert isinstance(fi, np.float64)


# ── detect_persistent_zero_states ────────────────────────────────────


class TestDetectPersistentZeroStates:
    """Tests for persistent zero-state detection."""

    def test_all_persistent(self):
        h = [
            np.array([0, 0, 0], dtype=np.int8),
            np.array([0, 0, 0], dtype=np.int8),
            np.array([0, 0, 0], dtype=np.int8),
        ]
        result = detect_persistent_zero_states(h)
        assert result == [0, 1, 2]

    def test_none_persistent(self):
        h = [
            np.array([0, 1, 0], dtype=np.int8),
            np.array([1, 0, -1], dtype=np.int8),
        ]
        result = detect_persistent_zero_states(h)
        assert result == []

    def test_partial_persistent(self):
        h = [
            np.array([0, 0, 1, 0], dtype=np.int8),
            np.array([0, 1, -1, 0], dtype=np.int8),
            np.array([0, 0, 1, 0], dtype=np.int8),
        ]
        result = detect_persistent_zero_states(h)
        assert result == [0, 3]

    def test_single_iteration(self):
        h = [np.array([0, 1, 0], dtype=np.int8)]
        result = detect_persistent_zero_states(h)
        assert result == [0, 2]

    def test_empty_history(self):
        result = detect_persistent_zero_states([])
        assert result == []

    def test_sorted_output(self):
        h = [
            np.array([0, 1, 0, 0], dtype=np.int8),
            np.array([0, -1, 0, 0], dtype=np.int8),
        ]
        result = detect_persistent_zero_states(h)
        assert result == sorted(result)

    def test_determinism(self):
        h = [
            np.array([0, 1, 0], dtype=np.int8),
            np.array([0, 0, 0], dtype=np.int8),
        ]
        r1 = detect_persistent_zero_states(h)
        r2 = detect_persistent_zero_states(h)
        assert r1 == r2


# ── estimate_trapping_indicator ──────────────────────────────────────


class TestEstimateTrappingIndicator:
    """Tests for trapping indicator estimation."""

    def test_all_decided_no_conflicts(self):
        msgs = np.array([1, 1, 1, 1], dtype=np.int8)
        H = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.float64)
        ti = estimate_trapping_indicator(msgs, H)
        assert isinstance(ti, np.float64)
        # zero_density=0, conflict_density=0, check_fraction depends on syndrome
        assert ti >= np.float64(0.0)
        assert ti <= np.float64(1.0)

    def test_all_undecided(self):
        msgs = np.array([0, 0, 0, 0], dtype=np.int8)
        H = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.float64)
        ti = estimate_trapping_indicator(msgs, H)
        assert isinstance(ti, np.float64)
        # zero_density=1.0, conflict_density=0 (no nonzero to conflict),
        # check_fraction=0 (all zero sums to 0)
        expected = np.float64((1.0 + 0.0 + 0.0) / 3.0)
        assert np.isclose(ti, expected)

    def test_high_conflict(self):
        msgs = np.array([1, -1, 1, -1], dtype=np.int8)
        H = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.float64)
        ti = estimate_trapping_indicator(msgs, H)
        assert isinstance(ti, np.float64)
        assert ti > np.float64(0.0)

    def test_empty_messages(self):
        msgs = np.array([], dtype=np.int8)
        H = np.zeros((0, 0), dtype=np.float64)
        ti = estimate_trapping_indicator(msgs, H)
        assert ti == np.float64(0.0)

    def test_determinism(self):
        msgs = np.array([1, 0, -1, 0], dtype=np.int8)
        H = np.array([[1, 1, 1, 0], [0, 1, 1, 1]], dtype=np.float64)
        t1 = estimate_trapping_indicator(msgs, H)
        t2 = estimate_trapping_indicator(msgs, H)
        assert t1 == t2

    def test_returns_float64(self):
        msgs = np.array([1, 0, -1], dtype=np.int8)
        H = np.array([[1, 1, 1]], dtype=np.float64)
        ti = estimate_trapping_indicator(msgs, H)
        assert isinstance(ti, np.float64)

    def test_range_bounded(self):
        msgs = np.array([0, 1, -1, 0, 1], dtype=np.int8)
        H = np.array([[1, 1, 0, 0, 0], [0, 0, 1, 1, 1]], dtype=np.float64)
        ti = estimate_trapping_indicator(msgs, H)
        assert np.float64(0.0) <= ti <= np.float64(1.0)


# ── Integration with ternary decoder ─────────────────────────────────


class TestIntegrationWithDecoder:
    """Integration tests using the ternary decoder."""

    def test_decoder_output_zero_regions(self):
        from src.qec.decoder.ternary.ternary_decoder import run_ternary_decoder

        H = np.array([
            [1, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 1],
        ], dtype=np.float64)
        received = np.array([1.0, 0.0, -1.0, 0.0], dtype=np.float64)
        result = run_ternary_decoder(H, received, max_iterations=10)
        msgs = result["final_messages"]

        regions = detect_zero_regions(msgs)
        assert "region_ids" in regions
        assert "region_sizes" in regions
        assert "node_indices" in regions

    def test_decoder_output_frustration(self):
        from src.qec.decoder.ternary.ternary_decoder import run_ternary_decoder

        H = np.array([
            [1, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 1],
        ], dtype=np.float64)
        received = np.array([1.0, 0.0, -1.0, 0.0], dtype=np.float64)
        result = run_ternary_decoder(H, received, max_iterations=10)
        msgs = result["final_messages"]

        fi = compute_frustration_index(msgs)
        assert isinstance(fi, np.float64)
        assert np.float64(0.0) <= fi <= np.float64(1.0)

    def test_decoder_output_trapping_indicator(self):
        from src.qec.decoder.ternary.ternary_decoder import run_ternary_decoder

        H = np.array([
            [1, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 1],
        ], dtype=np.float64)
        received = np.array([1.0, 0.0, -1.0, 0.0], dtype=np.float64)
        result = run_ternary_decoder(H, received, max_iterations=10)
        msgs = result["final_messages"]

        ti = estimate_trapping_indicator(msgs, H)
        assert isinstance(ti, np.float64)
        assert np.float64(0.0) <= ti <= np.float64(1.0)

    def test_full_pipeline_determinism(self):
        from src.qec.decoder.ternary.ternary_decoder import run_ternary_decoder

        H = np.array([
            [1, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 1],
        ], dtype=np.float64)
        received = np.array([1.0, 0.0, -1.0, 0.0], dtype=np.float64)

        # Run twice
        r1 = run_ternary_decoder(H, received, max_iterations=10)
        r2 = run_ternary_decoder(H, received, max_iterations=10)

        m1 = r1["final_messages"]
        m2 = r2["final_messages"]

        assert np.array_equal(m1, m2)
        assert detect_zero_regions(m1) == detect_zero_regions(m2)
        assert compute_frustration_index(m1) == compute_frustration_index(m2)
        assert estimate_trapping_indicator(m1, H) == estimate_trapping_indicator(m2, H)


# ── API wrapper tests ────────────────────────────────────────────────


class TestAPIWrappers:
    """Test that the analysis API properly exposes trapping functions."""

    def test_api_detect_zero_regions(self):
        from src.qec.analysis.api import detect_zero_regions as api_zr
        msgs = np.array([0, 1, 0, 0], dtype=np.int8)
        result = api_zr(msgs)
        assert result["region_ids"] == [0, 1]

    def test_api_compute_frustration_index(self):
        from src.qec.analysis.api import compute_frustration_index as api_fi
        msgs = np.array([0, 0, 1, 1], dtype=np.int8)
        assert api_fi(msgs) == np.float64(0.5)

    def test_api_detect_persistent_zero_states(self):
        from src.qec.analysis.api import detect_persistent_zero_states as api_pz
        h = [
            np.array([0, 1, 0], dtype=np.int8),
            np.array([0, -1, 0], dtype=np.int8),
        ]
        assert api_pz(h) == [0, 2]

    def test_api_estimate_trapping_indicator(self):
        from src.qec.analysis.api import estimate_trapping_indicator as api_ti
        msgs = np.array([1, -1], dtype=np.int8)
        H = np.array([[1, 1]], dtype=np.float64)
        ti = api_ti(msgs, H)
        assert isinstance(ti, np.float64)
