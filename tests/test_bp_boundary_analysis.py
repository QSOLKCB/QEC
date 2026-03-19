"""
Tests for BP boundary analysis (v5.3.0).

Validates:
  - Successful boundary detection (crossing found)
  - No-crossing scenario (all directions stay in same basin)
  - Determinism across runs
  - JSON serializability of all outputs
  - Edge case: no valid directions
  - Direction generation correctness
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from qec.diagnostics.bp_boundary_analysis import (
    compute_bp_boundary_analysis,
    generate_deterministic_directions,
    DEFAULT_PARAMS,
)


# ── Mock decoder functions ───────────────────────────────────────────


def _make_decoder_fixed(correction: np.ndarray):
    """Create a decoder that always returns the same correction."""
    fixed = np.array(correction, dtype=np.float64)

    def decoder_fn(llr):
        return np.copy(fixed)

    return decoder_fn


def _make_decoder_boundary_at_eps(baseline_correction: np.ndarray,
                                   alt_correction: np.ndarray,
                                   threshold: float):
    """Create a decoder that switches attractor when perturbation exceeds threshold.

    The first call establishes the baseline LLR.  Subsequent calls compare
    the L2 distance from the baseline to determine if the attractor changes.
    """
    _baseline_llr = None

    def decoder_fn(llr):
        nonlocal _baseline_llr
        llr_arr = np.asarray(llr, dtype=np.float64)

        if _baseline_llr is None:
            _baseline_llr = np.copy(llr_arr)

        dist = np.linalg.norm(llr_arr - _baseline_llr)
        if dist >= threshold - 1e-12:
            return np.copy(alt_correction)
        return np.copy(baseline_correction)

    return decoder_fn


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def small_llr():
    """Small LLR vector."""
    return np.array([2.0, -1.5, 0.8, -0.3, 1.2], dtype=np.float64)


@pytest.fixture
def small_H():
    """Small parity-check matrix (3x5)."""
    return np.array([
        [1, 1, 0, 0, 1],
        [0, 1, 1, 1, 0],
        [1, 0, 1, 0, 1],
    ], dtype=np.float64)


@pytest.fixture
def baseline_correction():
    """Baseline correction vector."""
    return np.array([0, 0, 0, 0, 0], dtype=np.float64)


@pytest.fixture
def alt_correction():
    """Alternative correction vector (different attractor)."""
    return np.array([1, 0, 1, 0, 0], dtype=np.float64)


# ── Tests: Direction Generation ──────────────────────────────────────


class TestDirectionGeneration:
    """Deterministic direction generation."""

    def test_direction_count(self, small_H, small_llr):
        """Check expected number of directions."""
        params = {**DEFAULT_PARAMS, "least_reliable_k": 3}
        directions = generate_deterministic_directions(small_H, small_llr, params)

        # 3 H rows * 2 (±) + 3 least reliable * 2 (±) + 1 global sign * 2 (±)
        expected = 3 * 2 + 3 * 2 + 1 * 2
        assert len(directions) == expected

    def test_direction_ordering_deterministic(self, small_H, small_llr):
        """Directions must be identical across calls."""
        params = {**DEFAULT_PARAMS}
        d1 = generate_deterministic_directions(small_H, small_llr, params)
        d2 = generate_deterministic_directions(small_H, small_llr, params)

        assert len(d1) == len(d2)
        for a, b in zip(d1, d2):
            np.testing.assert_array_equal(a, b)

    def test_directions_are_list(self, small_H, small_llr):
        """Directions must be a Python list, not set or dict."""
        params = {**DEFAULT_PARAMS}
        directions = generate_deterministic_directions(small_H, small_llr, params)
        assert isinstance(directions, list)

    def test_parity_rows_first(self, small_H, small_llr):
        """First directions must be from parity-check rows."""
        params = {**DEFAULT_PARAMS, "least_reliable_k": 2}
        directions = generate_deterministic_directions(small_H, small_llr, params)

        # First direction should be normalized first row of H.
        row0 = small_H[0].astype(np.float64)
        expected = row0 / np.linalg.norm(row0)
        np.testing.assert_allclose(directions[0], expected)

    def test_empty_H(self, small_llr):
        """Empty parity-check matrix produces only bit + sign directions."""
        H_empty = np.zeros((0, 5), dtype=np.float64)
        params = {**DEFAULT_PARAMS, "least_reliable_k": 2}
        directions = generate_deterministic_directions(H_empty, small_llr, params)

        # 0 H rows + 2 least reliable * 2 (±) + 1 global sign * 2 (±)
        expected = 0 + 2 * 2 + 1 * 2
        assert len(directions) == expected


# ── Tests: Boundary Detection ────────────────────────────────────────


class TestBoundaryDetection:
    """Successful boundary crossing detection."""

    def test_boundary_found(self, small_llr, small_H,
                            baseline_correction, alt_correction):
        """Boundary detected when perturbation crosses threshold."""
        threshold = 1.0
        decoder_fn = _make_decoder_boundary_at_eps(
            baseline_correction, alt_correction, threshold,
        )
        result = compute_bp_boundary_analysis(
            llr_vector=small_llr,
            decoder_fn=decoder_fn,
            parity_check_matrix=small_H,
            params={"epsilon_max": 5.0, "delta": 1e-4},
        )

        assert result["boundary_crossed"] is True
        assert result["boundary_eps"] is not None
        assert result["boundary_eps"] > 0
        assert result["boundary_direction"] is not None
        assert isinstance(result["boundary_direction"], list)
        assert len(result["boundary_direction"]) == small_llr.shape[0]
        assert result["num_directions"] >= 1

    def test_boundary_eps_is_float(self, small_llr, small_H,
                                    baseline_correction, alt_correction):
        """boundary_eps must be a float when crossing is found."""
        decoder_fn = _make_decoder_boundary_at_eps(
            baseline_correction, alt_correction, 0.5,
        )
        result = compute_bp_boundary_analysis(
            llr_vector=small_llr,
            decoder_fn=decoder_fn,
            parity_check_matrix=small_H,
        )

        assert isinstance(result["boundary_eps"], float)

    def test_baseline_attractor_is_list(self, small_llr, small_H,
                                         baseline_correction, alt_correction):
        """baseline_attractor must be a list (JSON-serializable)."""
        decoder_fn = _make_decoder_boundary_at_eps(
            baseline_correction, alt_correction, 0.5,
        )
        result = compute_bp_boundary_analysis(
            llr_vector=small_llr,
            decoder_fn=decoder_fn,
            parity_check_matrix=small_H,
        )

        assert isinstance(result["baseline_attractor"], list)


# ── Tests: No Crossing ───────────────────────────────────────────────


class TestNoCrossing:
    """All perturbations stay in the same attractor basin."""

    def test_no_crossing(self, small_llr, small_H, baseline_correction):
        """No boundary when decoder always returns same correction."""
        decoder_fn = _make_decoder_fixed(baseline_correction)
        result = compute_bp_boundary_analysis(
            llr_vector=small_llr,
            decoder_fn=decoder_fn,
            parity_check_matrix=small_H,
        )

        assert result["boundary_crossed"] is False
        assert result["boundary_eps"] is None
        assert result["boundary_direction"] is None
        assert result["num_directions"] > 0

    def test_no_crossing_high_threshold(self, small_llr, small_H,
                                         baseline_correction, alt_correction):
        """No crossing when threshold exceeds epsilon_max."""
        decoder_fn = _make_decoder_boundary_at_eps(
            baseline_correction, alt_correction, 100.0,
        )
        result = compute_bp_boundary_analysis(
            llr_vector=small_llr,
            decoder_fn=decoder_fn,
            parity_check_matrix=small_H,
            params={"epsilon_max": 5.0},
        )

        assert result["boundary_crossed"] is False
        assert result["boundary_eps"] is None


# ── Tests: Edge Cases ────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases and special inputs."""

    def test_zero_llr(self, small_H, baseline_correction, alt_correction):
        """Zero LLR vector produces valid result."""
        llr_zero = np.zeros(5, dtype=np.float64)
        decoder_fn = _make_decoder_fixed(baseline_correction)
        result = compute_bp_boundary_analysis(
            llr_vector=llr_zero,
            decoder_fn=decoder_fn,
            parity_check_matrix=small_H,
        )

        assert result["boundary_crossed"] is False
        assert result["num_directions"] >= 0

    def test_no_directions(self, baseline_correction):
        """Empty H and zero LLR — sign direction has zero norm."""
        H_empty = np.zeros((0, 3), dtype=np.float64)
        llr_zero = np.zeros(3, dtype=np.float64)
        decoder_fn = _make_decoder_fixed(baseline_correction[:3])
        result = compute_bp_boundary_analysis(
            llr_vector=llr_zero,
            decoder_fn=decoder_fn,
            parity_check_matrix=H_empty,
            params={"least_reliable_k": 0},
        )

        # With least_reliable_k=0 and zero LLR (sign norm=0), no directions.
        # Actually least_reliable_k=0 means k=min(0,n)=0 so no bit dirs.
        # And zero LLR means sign_vec=[0,0,0], norm=0, so no sign dir.
        # But H is empty so no H dirs either.
        assert result["boundary_crossed"] is False
        assert result["num_directions"] == 0
        assert result["boundary_eps"] is None
        assert result["boundary_direction"] is None


# ── Tests: Determinism ───────────────────────────────────────────────


class TestDeterminism:
    """Repeated runs must produce identical outputs."""

    def test_determinism_no_crossing(self, small_llr, small_H,
                                      baseline_correction):
        """Repeated no-crossing runs produce identical results."""
        decoder_fn1 = _make_decoder_fixed(baseline_correction)
        r1 = compute_bp_boundary_analysis(
            llr_vector=small_llr,
            decoder_fn=decoder_fn1,
            parity_check_matrix=small_H,
        )

        decoder_fn2 = _make_decoder_fixed(baseline_correction)
        r2 = compute_bp_boundary_analysis(
            llr_vector=small_llr,
            decoder_fn=decoder_fn2,
            parity_check_matrix=small_H,
        )

        assert r1 == r2

    def test_determinism_with_crossing(self, small_llr, small_H,
                                        baseline_correction, alt_correction):
        """Repeated crossing runs produce identical results."""
        decoder_fn1 = _make_decoder_boundary_at_eps(
            baseline_correction, alt_correction, 1.0,
        )
        r1 = compute_bp_boundary_analysis(
            llr_vector=small_llr,
            decoder_fn=decoder_fn1,
            parity_check_matrix=small_H,
            params={"delta": 1e-4},
        )

        decoder_fn2 = _make_decoder_boundary_at_eps(
            baseline_correction, alt_correction, 1.0,
        )
        r2 = compute_bp_boundary_analysis(
            llr_vector=small_llr,
            decoder_fn=decoder_fn2,
            parity_check_matrix=small_H,
            params={"delta": 1e-4},
        )

        assert r1 == r2


# ── Tests: JSON Serializability ──────────────────────────────────────


class TestJSONSerializable:
    """All outputs must be JSON-serializable."""

    def test_json_serializable_no_crossing(self, small_llr, small_H,
                                            baseline_correction):
        """No-crossing result is JSON-serializable."""
        decoder_fn = _make_decoder_fixed(baseline_correction)
        result = compute_bp_boundary_analysis(
            llr_vector=small_llr,
            decoder_fn=decoder_fn,
            parity_check_matrix=small_H,
        )

        serialized = json.dumps(result, sort_keys=True)
        deserialized = json.loads(serialized)
        assert deserialized == result

    def test_json_serializable_with_crossing(self, small_llr, small_H,
                                              baseline_correction,
                                              alt_correction):
        """Crossing result is JSON-serializable."""
        decoder_fn = _make_decoder_boundary_at_eps(
            baseline_correction, alt_correction, 1.0,
        )
        result = compute_bp_boundary_analysis(
            llr_vector=small_llr,
            decoder_fn=decoder_fn,
            parity_check_matrix=small_H,
            params={"delta": 1e-4},
        )

        serialized = json.dumps(result, sort_keys=True)
        deserialized = json.loads(serialized)
        assert deserialized == result

    def test_result_keys(self, small_llr, small_H, baseline_correction):
        """Result contains exactly the expected keys."""
        decoder_fn = _make_decoder_fixed(baseline_correction)
        result = compute_bp_boundary_analysis(
            llr_vector=small_llr,
            decoder_fn=decoder_fn,
            parity_check_matrix=small_H,
        )

        expected_keys = {
            "baseline_attractor",
            "boundary_crossed",
            "boundary_direction",
            "boundary_eps",
            "num_directions",
        }
        assert set(result.keys()) == expected_keys
