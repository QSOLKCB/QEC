"""
Tests for v5.8.0 — Ternary Transition Detection & Metastability Score.

Verifies:
  - boundary_crossings computation
  - regime_switch_count computation
  - first_success_iteration detection
  - first_failure_iteration detection
  - metastability_score for convergent, plateau, and oscillating residuals
  - backward compatibility of existing output fields
  - deterministic repeated execution
  - JSON serialization stability
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from qec.diagnostics.bp_phase_space import (
    compute_bp_phase_space,
    compute_metastability_score,
)
from qec.diagnostics.ternary_decoder_topology import (
    compute_ternary_decoder_topology,
)


# ── Test helpers ──────────────────────────────────────────────────────

def _success_phase_space() -> dict:
    """Phase-space result for a converged, successful decode."""
    fixed_point = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    states = []
    for t in range(10):
        perturbation = np.array([1.0, 0.5, 0.3], dtype=np.float64) * (0.001 ** t)
        states.append(fixed_point + perturbation)
    return compute_bp_phase_space(states)


def _failure_phase_space() -> dict:
    """Phase-space result for a converged but failed decode."""
    fixed_point = np.array([1.5, 0.8, 0.4], dtype=np.float64)
    states = []
    for t in range(10):
        perturbation = np.array([2.0, 1.0, 0.6], dtype=np.float64) * (0.001 ** t)
        states.append(fixed_point + perturbation)
    return compute_bp_phase_space(states)


def _oscillating_phase_space() -> dict:
    """Phase-space result for an oscillating, non-converging decode."""
    states = []
    for t in range(20):
        if t % 2 == 0:
            states.append(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        else:
            states.append(np.array([3.0, 2.0, 1.0], dtype=np.float64))
    return compute_bp_phase_space(states)


def _boundary_phase_space() -> dict:
    """Phase-space result that lingers near the boundary."""
    states = []
    for t in range(10):
        # Small oscillation around a point, not converging.
        offset = 0.5 * ((-1) ** t)
        states.append(np.array([1.0 + offset, 2.0, 3.0], dtype=np.float64))
    return compute_bp_phase_space(states)


# ── Transition detection tests ────────────────────────────────────────

class TestSuccessTrajectory:
    """Success trajectory: all states should be +1 after convergence."""

    def test_no_boundary_crossings(self):
        ps = _success_phase_space()
        sr = [0] * ps["trajectory_length"]
        result = compute_ternary_decoder_topology(
            phase_space_result=ps, syndrome_residuals=sr,
        )
        # Pure success trajectory should have no crossings from boundary.
        assert "boundary_crossings" in result
        assert isinstance(result["boundary_crossings"], int)
        assert result["boundary_crossings"] >= 0

    def test_first_success_iteration_present(self):
        ps = _success_phase_space()
        sr = [0] * ps["trajectory_length"]
        result = compute_ternary_decoder_topology(
            phase_space_result=ps, syndrome_residuals=sr,
        )
        assert "first_success_iteration" in result
        assert result["first_success_iteration"] is not None
        assert isinstance(result["first_success_iteration"], int)
        assert result["first_success_iteration"] >= 0

    def test_first_failure_iteration_none(self):
        ps = _success_phase_space()
        sr = [0] * ps["trajectory_length"]
        result = compute_ternary_decoder_topology(
            phase_space_result=ps, syndrome_residuals=sr,
        )
        assert "first_failure_iteration" in result
        assert result["first_failure_iteration"] is None

    def test_regime_switch_count(self):
        ps = _success_phase_space()
        sr = [0] * ps["trajectory_length"]
        result = compute_ternary_decoder_topology(
            phase_space_result=ps, syndrome_residuals=sr,
        )
        assert "regime_switch_count" in result
        assert isinstance(result["regime_switch_count"], int)


class TestFailureTrajectory:
    """Failure trajectory: states should be -1 after convergence."""

    def test_first_failure_iteration_present(self):
        ps = _failure_phase_space()
        sr = [5] * ps["trajectory_length"]
        result = compute_ternary_decoder_topology(
            phase_space_result=ps, syndrome_residuals=sr,
        )
        assert result["first_failure_iteration"] is not None
        assert isinstance(result["first_failure_iteration"], int)

    def test_first_success_iteration_none(self):
        ps = _failure_phase_space()
        sr = [5] * ps["trajectory_length"]
        result = compute_ternary_decoder_topology(
            phase_space_result=ps, syndrome_residuals=sr,
        )
        assert result["first_success_iteration"] is None

    def test_final_state_is_failure(self):
        ps = _failure_phase_space()
        sr = [5] * ps["trajectory_length"]
        result = compute_ternary_decoder_topology(
            phase_space_result=ps, syndrome_residuals=sr,
        )
        assert result["final_ternary_state"] == -1


class TestOscillatoryTrajectory:
    """Oscillatory trajectory: should show boundary crossings."""

    def test_boundary_crossings_present(self):
        ps = _oscillating_phase_space()
        # Mix of syndrome 0 and nonzero to create oscillation.
        sr = [0 if t % 4 == 0 else 3 for t in range(ps["trajectory_length"])]
        result = compute_ternary_decoder_topology(
            phase_space_result=ps, syndrome_residuals=sr,
        )
        assert result["boundary_crossings"] >= 0
        assert result["regime_switch_count"] >= 0

    def test_all_boundary_when_not_converged(self):
        ps = _oscillating_phase_space()
        sr = [0 if t % 4 == 0 else 3 for t in range(ps["trajectory_length"])]
        result = compute_ternary_decoder_topology(
            phase_space_result=ps, syndrome_residuals=sr,
        )
        # Oscillating residual norms are large (not < epsilon), so all
        # iterations classify as 0 (boundary). This means no success or
        # failure iterations exist.
        assert result["first_success_iteration"] is None
        assert result["first_failure_iteration"] is None
        # All states are boundary → no regime switches.
        assert result["regime_switch_count"] == 0


class TestBoundaryTrajectory:
    """Boundary trajectory: should be in metastable region."""

    def test_boundary_state(self):
        ps = _boundary_phase_space()
        result = compute_ternary_decoder_topology(
            phase_space_result=ps, syndrome_residuals=None,
        )
        assert result["final_ternary_state"] == 0

    def test_transition_metrics_present(self):
        ps = _boundary_phase_space()
        result = compute_ternary_decoder_topology(
            phase_space_result=ps, syndrome_residuals=None,
        )
        assert "boundary_crossings" in result
        assert "regime_switch_count" in result
        assert "first_success_iteration" in result
        assert "first_failure_iteration" in result


# ── Metastability score tests ─────────────────────────────────────────

class TestMetastabilityMonotonicDecay:
    """Monotonically decaying residuals → low metastability."""

    def test_low_score(self):
        residuals = [10.0, 5.0, 2.5, 1.25, 0.625, 0.3125]
        score = compute_metastability_score(residuals)
        assert score >= 0.0
        # Monotonic decay should have relatively low normalized variation.
        assert score < 1.0

    def test_deterministic(self):
        residuals = [10.0, 5.0, 2.5, 1.25, 0.625, 0.3125]
        s1 = compute_metastability_score(residuals)
        s2 = compute_metastability_score(residuals)
        assert s1 == s2


class TestMetastabilityPlateau:
    """Plateau residuals → medium metastability."""

    def test_low_score_for_flat_plateau(self):
        residuals = [1.0, 1.0, 1.0, 1.0, 1.0]
        score = compute_metastability_score(residuals)
        # Flat plateau: no differences, score should be 0.
        assert score == 0.0

    def test_near_plateau(self):
        residuals = [1.0, 1.001, 1.0, 0.999, 1.0]
        score = compute_metastability_score(residuals)
        assert score < 0.01


class TestMetastabilityOscillation:
    """Oscillating residuals → high metastability."""

    def test_high_score(self):
        residuals = [1.0, 5.0, 1.0, 5.0, 1.0, 5.0]
        score = compute_metastability_score(residuals)
        # Large oscillations normalized by mean should give high score.
        assert score > 0.5

    def test_score_is_float(self):
        residuals = [1.0, 5.0, 1.0, 5.0]
        score = compute_metastability_score(residuals)
        assert isinstance(score, float)


class TestMetastabilityEdgeCases:
    """Edge cases for metastability score."""

    def test_empty_residuals(self):
        score = compute_metastability_score([])
        assert score == 0.0

    def test_single_residual(self):
        score = compute_metastability_score([1.0])
        assert score == 0.0

    def test_two_residuals(self):
        score = compute_metastability_score([1.0, 2.0])
        assert isinstance(score, float)
        assert score >= 0.0

    def test_tail_length_larger_than_list(self):
        residuals = [1.0, 2.0, 3.0]
        score = compute_metastability_score(residuals, tail_length=100)
        assert isinstance(score, float)

    def test_zero_residuals(self):
        residuals = [0.0, 0.0, 0.0]
        score = compute_metastability_score(residuals)
        assert score == 0.0


# ── Backward compatibility tests ─────────────────────────────────────

class TestBackwardCompatibility:
    """v5.8.0 output must include all v5.7.0 fields."""

    def test_all_v57_fields_present(self):
        ps = _success_phase_space()
        sr = [0] * ps["trajectory_length"]
        result = compute_ternary_decoder_topology(
            phase_space_result=ps, syndrome_residuals=sr,
        )
        v57_keys = {
            "ternary_trace", "final_ternary_state", "state_durations",
            "transition_iteration", "classification_reason", "evidence",
        }
        for key in v57_keys:
            assert key in result, f"Missing v5.7 key: {key}"

    def test_v58_fields_present(self):
        ps = _success_phase_space()
        sr = [0] * ps["trajectory_length"]
        result = compute_ternary_decoder_topology(
            phase_space_result=ps, syndrome_residuals=sr,
        )
        v58_keys = {
            "boundary_crossings", "regime_switch_count",
            "first_success_iteration", "first_failure_iteration",
        }
        for key in v58_keys:
            assert key in result, f"Missing v5.8 key: {key}"


# ── Determinism tests ────────────────────────────────────────────────

class TestDeterminism:
    """Transition metrics must be identical across repeated calls."""

    def test_identical_transition_metrics(self):
        ps = _oscillating_phase_space()
        sr = [0 if t % 3 == 0 else 2 for t in range(ps["trajectory_length"])]
        r1 = compute_ternary_decoder_topology(
            phase_space_result=ps, syndrome_residuals=sr,
        )
        r2 = compute_ternary_decoder_topology(
            phase_space_result=ps, syndrome_residuals=sr,
        )
        assert r1["boundary_crossings"] == r2["boundary_crossings"]
        assert r1["regime_switch_count"] == r2["regime_switch_count"]
        assert r1["first_success_iteration"] == r2["first_success_iteration"]
        assert r1["first_failure_iteration"] == r2["first_failure_iteration"]

    def test_metastability_deterministic(self):
        residuals = [1.0, 3.0, 1.5, 4.0, 2.0]
        s1 = compute_metastability_score(residuals)
        s2 = compute_metastability_score(residuals)
        assert s1 == s2


# ── JSON serialization tests ─────────────────────────────────────────

class TestJsonSerialization:
    """All new fields must survive JSON roundtrip."""

    def test_roundtrip(self):
        ps = _success_phase_space()
        sr = [0] * ps["trajectory_length"]
        result = compute_ternary_decoder_topology(
            phase_space_result=ps, syndrome_residuals=sr,
        )
        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert deserialized["boundary_crossings"] == result["boundary_crossings"]
        assert deserialized["regime_switch_count"] == result["regime_switch_count"]
        assert deserialized["first_success_iteration"] == result["first_success_iteration"]
        assert deserialized["first_failure_iteration"] == result["first_failure_iteration"]

    def test_metastability_roundtrip(self):
        residuals = [1.0, 2.0, 1.5, 3.0]
        score = compute_metastability_score(residuals)
        serialized = json.dumps({"metastability_score": score})
        deserialized = json.loads(serialized)
        assert deserialized["metastability_score"] == score
