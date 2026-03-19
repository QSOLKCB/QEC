"""
Tests for v5.7.0 — BP Phase-Space Explorer.

Verifies correct residual norm computation, projection coordinate
correctness, oscillation scoring, deterministic repeated execution,
JSON serialization stability, and input validation.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from qec.diagnostics.bp_phase_space import (
    compute_bp_phase_space,
)


# ── Test helpers ──────────────────────────────────────────────────────

def _converging_trajectory(n: int = 6, length: int = 10) -> list[np.ndarray]:
    """Trajectory that converges to a fixed point."""
    states = []
    for t in range(length):
        decay = 1.0 / (1.0 + t)
        state = np.array([decay * (i + 1) for i in range(n)], dtype=np.float64)
        states.append(state)
    return states


def _constant_trajectory(n: int = 6, length: int = 5) -> list[np.ndarray]:
    """Trajectory at a fixed point (all states identical)."""
    state = np.array([1.0 * (i + 1) for i in range(n)], dtype=np.float64)
    return [state.copy() for _ in range(length)]


def _oscillating_trajectory(n: int = 6, length: int = 20) -> list[np.ndarray]:
    """Trajectory with oscillating residual norms (alternating step sizes).

    Constructs states with cumulative displacements that alternate
    between large and small steps, producing residual norms that
    oscillate and thus high sign-change rate in first differences.
    """
    states = [np.zeros(n, dtype=np.float64)]
    for t in range(1, length):
        prev = states[-1]
        # Alternate between large (2.0) and small (0.1) step sizes.
        step_size = 2.0 if t % 2 == 1 else 0.1
        states.append(prev + np.ones(n, dtype=np.float64) * step_size)
    return states


def _single_state_trajectory(n: int = 6) -> list[np.ndarray]:
    """Trajectory with only one state."""
    return [np.ones(n, dtype=np.float64)]


# ── Input validation tests ───────────────────────────────────────────

class TestInputValidation:
    """Verify correct error handling for invalid inputs."""

    def test_empty_trajectory_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            compute_bp_phase_space([])

    def test_inconsistent_dimensions_raise(self):
        states = [
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0]),
        ]
        with pytest.raises(ValueError, match="dimension"):
            compute_bp_phase_space(states)

    def test_zero_dimension_raises(self):
        states = [np.array([], dtype=np.float64)]
        with pytest.raises(ValueError, match="non-zero dimension"):
            compute_bp_phase_space(states)

    def test_spectral_basis_wrong_ndim_raises(self):
        states = _converging_trajectory()
        basis = np.ones(6, dtype=np.float64)  # 1-D, should be 2-D
        with pytest.raises(ValueError, match="2-D"):
            compute_bp_phase_space(states, spectral_basis=basis)

    def test_spectral_basis_too_few_rows_raises(self):
        states = _converging_trajectory(n=6)
        basis = np.eye(4, 2, dtype=np.float64)  # 4 rows < 6
        with pytest.raises(ValueError, match="rows"):
            compute_bp_phase_space(states, spectral_basis=basis)


# ── Residual norm tests ──────────────────────────────────────────────

class TestResidualNorms:
    """Verify correct residual norm computation."""

    def test_constant_trajectory_zero_residuals(self):
        states = _constant_trajectory()
        out = compute_bp_phase_space(states)
        for r in out["residual_norms"]:
            assert r == 0.0

    def test_residual_count(self):
        states = _converging_trajectory(length=10)
        out = compute_bp_phase_space(states)
        assert len(out["residual_norms"]) == 9  # length - 1

    def test_single_state_no_residuals(self):
        states = _single_state_trajectory()
        out = compute_bp_phase_space(states)
        assert out["residual_norms"] == []

    def test_known_residual_value(self):
        """Two-state trajectory with known L2 norm."""
        s0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        s1 = np.array([3.0, 4.0, 0.0], dtype=np.float64)
        out = compute_bp_phase_space([s0, s1])
        assert abs(out["residual_norms"][0] - 5.0) < 1e-12

    def test_converging_residuals_decrease(self):
        states = _converging_trajectory(length=20)
        out = compute_bp_phase_space(states)
        norms = out["residual_norms"]
        # Overall trend should decrease (first > last).
        assert norms[0] > norms[-1]


# ── Phase coordinate tests ──────────────────────────────────────────

class TestPhaseCoordinates:
    """Verify projection coordinate correctness."""

    def test_default_projection_uses_first_3_coords(self):
        n = 6
        states = _converging_trajectory(n=n, length=3)
        out = compute_bp_phase_space(states)
        # Default: min(3, 6) = 3 coordinates.
        assert len(out["phase_coordinates"][0]) == 3
        # Should be first 3 components of the state.
        for t, state in enumerate(states):
            for k in range(3):
                assert abs(out["phase_coordinates"][t][k] - float(state[k])) < 1e-12

    def test_small_dimension_fewer_coords(self):
        states = [np.array([1.0, 2.0], dtype=np.float64)]
        out = compute_bp_phase_space(states)
        # min(3, 2) = 2 coordinates.
        assert len(out["phase_coordinates"][0]) == 2

    def test_spectral_basis_projection(self):
        """Projection onto identity basis should return state values."""
        n = 4
        states = [np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)]
        basis = np.eye(n, 2, dtype=np.float64)  # first 2 standard basis
        out = compute_bp_phase_space(states, spectral_basis=basis)
        assert len(out["phase_coordinates"][0]) == 2
        assert abs(out["phase_coordinates"][0][0] - 1.0) < 1e-12
        assert abs(out["phase_coordinates"][0][1] - 2.0) < 1e-12

    def test_final_phase_coordinate(self):
        states = _converging_trajectory(length=5)
        out = compute_bp_phase_space(states)
        assert out["final_phase_coordinate"] == out["phase_coordinates"][-1]

    def test_coordinate_count_matches_trajectory(self):
        states = _converging_trajectory(length=7)
        out = compute_bp_phase_space(states)
        assert len(out["phase_coordinates"]) == 7


# ── Oscillation score tests ─────────────────────────────────────────

class TestOscillationScore:
    """Verify oscillation score computation."""

    def test_constant_trajectory_zero_oscillation(self):
        states = _constant_trajectory()
        out = compute_bp_phase_space(states)
        assert out["oscillation_score"] == 0.0

    def test_single_state_zero_oscillation(self):
        states = _single_state_trajectory()
        out = compute_bp_phase_space(states)
        assert out["oscillation_score"] == 0.0

    def test_oscillating_trajectory_high_score(self):
        states = _oscillating_trajectory(length=20)
        out = compute_bp_phase_space(states)
        # Alternating states should produce high oscillation.
        assert out["oscillation_score"] > 0.5

    def test_oscillation_score_bounded(self):
        states = _oscillating_trajectory(length=50)
        out = compute_bp_phase_space(states)
        assert 0.0 <= out["oscillation_score"] <= 1.0


# ── Output structure tests ──────────────────────────────────────────

class TestOutputStructure:
    """Verify all required keys are present and have correct types."""

    REQUIRED_KEYS = {
        "trajectory_length": int,
        "state_dimension": int,
        "residual_norms": list,
        "phase_coordinates": list,
        "final_phase_coordinate": list,
        "oscillation_score": float,
    }

    def test_all_keys_present(self):
        out = compute_bp_phase_space(_converging_trajectory())
        for key in self.REQUIRED_KEYS:
            assert key in out, f"Missing key: {key}"

    def test_all_types_correct(self):
        out = compute_bp_phase_space(_converging_trajectory())
        for key, expected_type in self.REQUIRED_KEYS.items():
            assert isinstance(out[key], expected_type), (
                f"Key {key}: expected {expected_type}, got {type(out[key])}"
            )


# ── Determinism tests ────────────────────────────────────────────────

class TestDeterminism:
    """Verify byte-identical outputs across repeated calls."""

    def test_converging_determinism(self):
        states = _converging_trajectory()
        out1 = compute_bp_phase_space(states)
        out2 = compute_bp_phase_space(states)
        j1 = json.dumps(out1, sort_keys=True)
        j2 = json.dumps(out2, sort_keys=True)
        assert j1 == j2

    def test_oscillating_determinism(self):
        states = _oscillating_trajectory()
        out1 = compute_bp_phase_space(states)
        out2 = compute_bp_phase_space(states)
        j1 = json.dumps(out1, sort_keys=True)
        j2 = json.dumps(out2, sort_keys=True)
        assert j1 == j2

    def test_with_basis_determinism(self):
        states = _converging_trajectory(n=6)
        basis = np.eye(6, 3, dtype=np.float64)
        out1 = compute_bp_phase_space(states, spectral_basis=basis)
        out2 = compute_bp_phase_space(states, spectral_basis=basis)
        j1 = json.dumps(out1, sort_keys=True)
        j2 = json.dumps(out2, sort_keys=True)
        assert j1 == j2


# ── JSON serialization tests ────────────────────────────────────────

class TestJsonSerialization:
    """Verify JSON roundtrip stability."""

    def test_json_roundtrip(self):
        out = compute_bp_phase_space(_converging_trajectory())
        s = json.dumps(out, sort_keys=True)
        roundtrip = json.loads(s)
        assert roundtrip == out

    def test_json_roundtrip_single_state(self):
        out = compute_bp_phase_space(_single_state_trajectory())
        s = json.dumps(out, sort_keys=True)
        roundtrip = json.loads(s)
        assert roundtrip == out

    def test_all_values_json_types(self):
        out = compute_bp_phase_space(_converging_trajectory())
        s = json.dumps(out)
        assert isinstance(s, str)


# ── No-mutation test ─────────────────────────────────────────────────

class TestNoMutation:
    """Verify inputs are not mutated."""

    def test_trajectory_not_mutated(self):
        states = _converging_trajectory()
        copies = [s.copy() for s in states]
        compute_bp_phase_space(states)
        for orig, copy in zip(states, copies):
            assert np.array_equal(orig, copy)

    def test_basis_not_mutated(self):
        states = _converging_trajectory(n=6)
        basis = np.eye(6, 3, dtype=np.float64)
        basis_copy = basis.copy()
        compute_bp_phase_space(states, spectral_basis=basis)
        assert np.array_equal(basis, basis_copy)
