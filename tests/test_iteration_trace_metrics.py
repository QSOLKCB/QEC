"""
Tests for v4.3.0 — Deterministic Iteration-Trace Diagnostics.

Covers:
  - Determinism (run twice → identical results)
  - No input mutation
  - Oscillation detection (synthetic oscillating LLR traces)
  - Stable convergence (monotonic energy → CIS ≈ 0)
  - Trapping set detection (persistent error nodes)
  - Correction cycling (synthetic repeated patterns)
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from qec.diagnostics.iteration_trace import (
    compute_belief_oscillation_index,
    compute_convergence_instability_score,
    compute_correction_vector_fluctuation,
    compute_iteration_trace_metrics,
    compute_oscillation_depth,
    compute_persistent_error_indicator,
)


# ── Helpers ─────────────────────────────────────────────────────────


def _make_oscillating_llr_trace(n_vars: int, n_iters: int) -> list[np.ndarray]:
    """Create a trace that oscillates sign every iteration."""
    trace = []
    for t in range(n_iters):
        sign = 1.0 if t % 2 == 0 else -1.0
        trace.append(np.full(n_vars, sign * 2.0, dtype=np.float64))
    return trace


def _make_stable_llr_trace(n_vars: int, n_iters: int) -> list[np.ndarray]:
    """Create a trace with stable positive LLRs (converged)."""
    return [np.full(n_vars, 3.0, dtype=np.float64) for _ in range(n_iters)]


def _make_trapping_llr_trace(
    n_vars: int, n_iters: int, trapped_indices: list[int],
) -> list[np.ndarray]:
    """Create a trace where trapped nodes always have negative LLR."""
    trace = []
    for _ in range(n_iters):
        llr = np.full(n_vars, 5.0, dtype=np.float64)
        for idx in trapped_indices:
            llr[idx] = -3.0
        trace.append(llr)
    return trace


def _make_monotonic_energy_trace(n_iters: int) -> list[float]:
    """Monotonically decreasing energy trace (stable convergence)."""
    return [100.0 - float(t) for t in range(n_iters)]


def _make_cycling_corrections(
    n_vars: int, n_iters: int,
) -> list[np.ndarray]:
    """Correction vectors that alternate between two patterns."""
    a = np.zeros(n_vars, dtype=np.float64)
    b = np.ones(n_vars, dtype=np.float64)
    return [a.copy() if t % 2 == 0 else b.copy() for t in range(n_iters)]


# ── Determinism Tests ───────────────────────────────────────────────


class TestDeterminism:
    """Run each diagnostic twice with identical inputs → identical results."""

    def test_pei_determinism(self):
        trace = _make_trapping_llr_trace(10, 20, [0, 3, 7])
        r1 = compute_persistent_error_indicator(trace)
        r2 = compute_persistent_error_indicator(trace)
        np.testing.assert_array_equal(r1["pei_vector"], r2["pei_vector"])
        assert r1["pei_count"] == r2["pei_count"]
        assert r1["pei_ratio"] == r2["pei_ratio"]

    def test_boi_determinism(self):
        trace = _make_oscillating_llr_trace(10, 20)
        r1 = compute_belief_oscillation_index(trace)
        r2 = compute_belief_oscillation_index(trace)
        np.testing.assert_array_equal(r1["boi_vector"], r2["boi_vector"])
        assert r1["boi_mean"] == r2["boi_mean"]
        assert r1["boi_max"] == r2["boi_max"]

    def test_od_determinism(self):
        trace = _make_oscillating_llr_trace(10, 20)
        r1 = compute_oscillation_depth(trace)
        r2 = compute_oscillation_depth(trace)
        np.testing.assert_array_equal(r1["od_vector"], r2["od_vector"])
        assert r1["od_mean"] == r2["od_mean"]
        assert r1["od_max"] == r2["od_max"]

    def test_cis_determinism(self):
        trace = [10.0, 9.5, 9.0, 8.8, 8.7, 8.6, 8.5, 8.4]
        r1 = compute_convergence_instability_score(trace)
        r2 = compute_convergence_instability_score(trace)
        assert r1["cis"] == r2["cis"]

    def test_cvf_determinism(self):
        corrs = _make_cycling_corrections(10, 20)
        r1 = compute_correction_vector_fluctuation(corrs)
        r2 = compute_correction_vector_fluctuation(corrs)
        assert r1["cvf_mean"] == r2["cvf_mean"]
        assert r1["cvf_max"] == r2["cvf_max"]

    def test_composite_determinism(self):
        llr_trace = _make_oscillating_llr_trace(10, 20)
        energy_trace = _make_monotonic_energy_trace(20)
        corrections = _make_cycling_corrections(10, 20)
        r1 = compute_iteration_trace_metrics(llr_trace, energy_trace, corrections)
        r2 = compute_iteration_trace_metrics(llr_trace, energy_trace, corrections)
        # Check all scalar fields match.
        assert r1["convergence_instability_score"] == r2["convergence_instability_score"]
        assert r1["correction_vector_fluctuation"] == r2["correction_vector_fluctuation"]
        np.testing.assert_array_equal(
            r1["persistent_error_indicator"]["pei_vector"],
            r2["persistent_error_indicator"]["pei_vector"],
        )
        np.testing.assert_array_equal(
            r1["belief_oscillation_index"]["boi_vector"],
            r2["belief_oscillation_index"]["boi_vector"],
        )


# ── No Input Mutation Tests ─────────────────────────────────────────


class TestNoInputMutation:
    """Verify that input traces are not modified."""

    def test_pei_no_mutation(self):
        trace = _make_trapping_llr_trace(10, 20, [1, 5])
        original = [arr.copy() for arr in trace]
        compute_persistent_error_indicator(trace)
        for orig, cur in zip(original, trace):
            np.testing.assert_array_equal(orig, cur)

    def test_boi_no_mutation(self):
        trace = _make_oscillating_llr_trace(10, 15)
        original = [arr.copy() for arr in trace]
        compute_belief_oscillation_index(trace)
        for orig, cur in zip(original, trace):
            np.testing.assert_array_equal(orig, cur)

    def test_od_no_mutation(self):
        trace = _make_oscillating_llr_trace(10, 15)
        original = [arr.copy() for arr in trace]
        compute_oscillation_depth(trace)
        for orig, cur in zip(original, trace):
            np.testing.assert_array_equal(orig, cur)

    def test_cis_no_mutation(self):
        trace = [10.0, 9.5, 9.0, 8.5]
        original = list(trace)
        compute_convergence_instability_score(trace)
        assert trace == original

    def test_cvf_no_mutation(self):
        corrs = _make_cycling_corrections(10, 10)
        original = [arr.copy() for arr in corrs]
        compute_correction_vector_fluctuation(corrs)
        for orig, cur in zip(original, corrs):
            np.testing.assert_array_equal(orig, cur)

    def test_composite_no_mutation(self):
        llr = _make_oscillating_llr_trace(8, 12)
        energy = _make_monotonic_energy_trace(12)
        corrs = _make_cycling_corrections(8, 12)
        llr_orig = [a.copy() for a in llr]
        energy_orig = list(energy)
        corrs_orig = [a.copy() for a in corrs]
        compute_iteration_trace_metrics(llr, energy, corrs)
        for a, b in zip(llr_orig, llr):
            np.testing.assert_array_equal(a, b)
        assert energy == energy_orig
        for a, b in zip(corrs_orig, corrs):
            np.testing.assert_array_equal(a, b)


# ── Oscillation Detection Tests ─────────────────────────────────────


class TestOscillationDetection:
    """Synthetic oscillating LLR traces should yield high BOI and OD."""

    def test_boi_oscillating(self):
        trace = _make_oscillating_llr_trace(5, 20)
        result = compute_belief_oscillation_index(trace)
        # Every node flips sign every iteration: 19 flips each.
        assert result["boi_max"] == 19
        assert result["boi_mean"] == 19.0
        np.testing.assert_array_equal(
            result["boi_vector"], np.full(5, 19, dtype=np.int32),
        )

    def test_boi_stable(self):
        trace = _make_stable_llr_trace(5, 20)
        result = compute_belief_oscillation_index(trace)
        assert result["boi_max"] == 0
        assert result["boi_mean"] == 0.0

    def test_od_oscillating(self):
        trace = _make_oscillating_llr_trace(5, 20)
        result = compute_oscillation_depth(trace, window=10)
        # Oscillates between +2 and -2 → depth = 4.
        np.testing.assert_allclose(result["od_vector"], np.full(5, 4.0))
        assert result["od_max"] == pytest.approx(4.0)
        assert result["od_mean"] == pytest.approx(4.0)

    def test_od_stable(self):
        trace = _make_stable_llr_trace(5, 20)
        result = compute_oscillation_depth(trace, window=10)
        np.testing.assert_allclose(result["od_vector"], np.zeros(5))
        assert result["od_max"] == pytest.approx(0.0)


# ── Stable Convergence Tests ────────────────────────────────────────


class TestStableConvergence:
    """Monotonically decreasing energy → CIS ≈ 0 (up to float variance)."""

    def test_constant_energy(self):
        trace = [5.0] * 20
        result = compute_convergence_instability_score(trace, window=10)
        assert result["cis"] == pytest.approx(0.0, abs=1e-15)

    def test_monotonic_energy_low_cis(self):
        # Linearly decreasing: variance of arithmetic sequence.
        trace = _make_monotonic_energy_trace(20)
        result = compute_convergence_instability_score(trace, window=10)
        # var of [90,91,...,99] = 8.25
        expected_var = np.var(np.array(trace[-10:], dtype=np.float64))
        assert result["cis"] == pytest.approx(expected_var)

    def test_erratic_energy_high_cis(self):
        trace = [100.0, 10.0, 100.0, 10.0, 100.0, 10.0]
        result = compute_convergence_instability_score(trace, window=6)
        assert result["cis"] > 100.0  # High variance.


# ── Trapping Set Detection Tests ────────────────────────────────────


class TestTrappingSetDetection:
    """PEI should flag nodes with persistent negative LLR."""

    def test_known_trapped_nodes(self):
        trapped = [2, 4, 7]
        trace = _make_trapping_llr_trace(10, 20, trapped)
        result = compute_persistent_error_indicator(trace, window=5)
        assert result["pei_count"] == len(trapped)
        assert result["pei_ratio"] == pytest.approx(len(trapped) / 10.0)
        for idx in trapped:
            assert result["pei_vector"][idx] == 1
        for idx in range(10):
            if idx not in trapped:
                assert result["pei_vector"][idx] == 0

    def test_no_trapping(self):
        trace = _make_stable_llr_trace(10, 20)
        result = compute_persistent_error_indicator(trace, window=5)
        assert result["pei_count"] == 0
        assert result["pei_ratio"] == 0.0

    def test_short_trace_window_clamp(self):
        # Trace shorter than window: should still work.
        trace = _make_trapping_llr_trace(5, 3, [0, 1])
        result = compute_persistent_error_indicator(trace, window=10)
        assert result["pei_count"] == 2

    def test_empty_trace(self):
        result = compute_persistent_error_indicator([], window=5)
        assert result["pei_count"] == 0


# ── Correction Cycling Tests ────────────────────────────────────────


class TestCorrectionCycling:
    """CVF should detect alternating correction vectors."""

    def test_cycling_corrections(self):
        corrs = _make_cycling_corrections(10, 20)
        result = compute_correction_vector_fluctuation(corrs)
        # Alternating between zeros and ones: ||diff|| = sqrt(10).
        expected_norm = float(np.sqrt(10.0))
        assert result["cvf_mean"] == pytest.approx(expected_norm)
        assert result["cvf_max"] == pytest.approx(expected_norm)

    def test_constant_corrections(self):
        corrs = [np.ones(10, dtype=np.float64) for _ in range(20)]
        result = compute_correction_vector_fluctuation(corrs)
        assert result["cvf_mean"] == pytest.approx(0.0)
        assert result["cvf_max"] == pytest.approx(0.0)

    def test_single_correction(self):
        result = compute_correction_vector_fluctuation(
            [np.array([1.0, 0.0, 1.0])],
        )
        assert result["cvf_mean"] == 0.0
        assert result["cvf_max"] == 0.0

    def test_empty_corrections(self):
        result = compute_correction_vector_fluctuation([])
        assert result["cvf_mean"] == 0.0
        assert result["cvf_max"] == 0.0


# ── BOI Zero-Handling Tests ─────────────────────────────────────────


class TestBOIZeroHandling:
    """Zeros in LLR traces must be treated as non-negative (sign = +1)."""

    def test_zero_to_positive_no_flip(self):
        """Transition from 0.0 to +1.0 should not count as a sign flip."""
        trace = [
            np.array([0.0, 0.0], dtype=np.float64),
            np.array([1.0, 2.0], dtype=np.float64),
        ]
        result = compute_belief_oscillation_index(trace)
        np.testing.assert_array_equal(result["boi_vector"], [0, 0])

    def test_zero_to_negative_is_flip(self):
        """Transition from 0.0 to -1.0 should count as a sign flip."""
        trace = [
            np.array([0.0, 0.0], dtype=np.float64),
            np.array([-1.0, -2.0], dtype=np.float64),
        ]
        result = compute_belief_oscillation_index(trace)
        np.testing.assert_array_equal(result["boi_vector"], [1, 1])

    def test_positive_to_zero_no_flip(self):
        """Transition from +1.0 to 0.0 should not count as a sign flip."""
        trace = [
            np.array([1.0], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
        ]
        result = compute_belief_oscillation_index(trace)
        np.testing.assert_array_equal(result["boi_vector"], [0])

    def test_negative_to_zero_is_flip(self):
        """Transition from -1.0 to 0.0 should count as a sign flip."""
        trace = [
            np.array([-1.0], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
        ]
        result = compute_belief_oscillation_index(trace)
        np.testing.assert_array_equal(result["boi_vector"], [1])

    def test_zero_to_zero_no_flip(self):
        """Transition from 0.0 to 0.0 should not count as a sign flip."""
        trace = [
            np.array([0.0], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
        ]
        result = compute_belief_oscillation_index(trace)
        np.testing.assert_array_equal(result["boi_vector"], [0])


# ── Composite None-CVF Tests ────────────────────────────────────────


class TestCompositeNoneCVF:
    """When correction_vectors is None, CVF fields should be None."""

    def test_none_corrections(self):
        llr = _make_oscillating_llr_trace(5, 10)
        energy = _make_monotonic_energy_trace(10)
        result = compute_iteration_trace_metrics(llr, energy, None)
        assert result["correction_vector_fluctuation"]["cvf_mean"] is None
        assert result["correction_vector_fluctuation"]["cvf_max"] is None

    def test_empty_corrections(self):
        llr = _make_oscillating_llr_trace(5, 10)
        energy = _make_monotonic_energy_trace(10)
        result = compute_iteration_trace_metrics(llr, energy, [])
        assert result["correction_vector_fluctuation"]["cvf_mean"] is None
        assert result["correction_vector_fluctuation"]["cvf_max"] is None

    def test_single_correction_vector(self):
        llr = _make_oscillating_llr_trace(5, 10)
        energy = _make_monotonic_energy_trace(10)
        result = compute_iteration_trace_metrics(
            llr, energy, [np.ones(5)],
        )
        assert result["correction_vector_fluctuation"]["cvf_mean"] is None
        assert result["correction_vector_fluctuation"]["cvf_max"] is None

    def test_determinism_with_none_corrections(self):
        llr = _make_oscillating_llr_trace(5, 10)
        energy = _make_monotonic_energy_trace(10)
        r1 = compute_iteration_trace_metrics(llr, energy, None)
        r2 = compute_iteration_trace_metrics(llr, energy, None)
        assert r1["convergence_instability_score"] == r2["convergence_instability_score"]
        assert r1["correction_vector_fluctuation"] == r2["correction_vector_fluctuation"]


# ── Composite Metric Tests ──────────────────────────────────────────


class TestCompositeMetric:
    """Test the unified compute_iteration_trace_metrics function."""

    def test_output_structure(self):
        llr = _make_oscillating_llr_trace(5, 10)
        energy = _make_monotonic_energy_trace(10)
        corrs = _make_cycling_corrections(5, 10)
        result = compute_iteration_trace_metrics(llr, energy, corrs)

        assert "persistent_error_indicator" in result
        assert "belief_oscillation_index" in result
        assert "oscillation_depth" in result
        assert "convergence_instability_score" in result
        assert "correction_vector_fluctuation" in result

        # CIS is a float, not a dict.
        assert isinstance(result["convergence_instability_score"], float)

        # Sub-dicts have expected keys.
        pei = result["persistent_error_indicator"]
        assert "pei_vector" in pei
        assert "pei_count" in pei
        assert "pei_ratio" in pei

        boi = result["belief_oscillation_index"]
        assert "boi_vector" in boi
        assert "boi_mean" in boi
        assert "boi_max" in boi

        od = result["oscillation_depth"]
        assert "od_vector" in od
        assert "od_mean" in od
        assert "od_max" in od

        cvf = result["correction_vector_fluctuation"]
        assert "cvf_mean" in cvf
        assert "cvf_max" in cvf

    def test_composite_matches_individual(self):
        llr = _make_oscillating_llr_trace(8, 15)
        energy = [50.0, 45.0, 42.0, 40.0, 38.0, 37.5, 37.0, 36.8,
                  36.6, 36.5, 36.4, 36.3, 36.2, 36.1, 36.0]
        corrs = _make_cycling_corrections(8, 15)

        composite = compute_iteration_trace_metrics(llr, energy, corrs)
        pei = compute_persistent_error_indicator(llr)
        boi = compute_belief_oscillation_index(llr)
        od = compute_oscillation_depth(llr)
        cis = compute_convergence_instability_score(energy)
        cvf = compute_correction_vector_fluctuation(corrs)

        np.testing.assert_array_equal(
            composite["persistent_error_indicator"]["pei_vector"],
            pei["pei_vector"],
        )
        assert composite["persistent_error_indicator"]["pei_count"] == pei["pei_count"]
        np.testing.assert_array_equal(
            composite["belief_oscillation_index"]["boi_vector"],
            boi["boi_vector"],
        )
        assert composite["oscillation_depth"]["od_mean"] == od["od_mean"]
        assert composite["convergence_instability_score"] == cis["cis"]
        assert composite["correction_vector_fluctuation"]["cvf_mean"] == cvf["cvf_mean"]
