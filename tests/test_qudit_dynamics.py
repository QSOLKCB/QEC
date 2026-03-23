"""Tests for qudit_dynamics — deterministic qudit measurement layer.

Covers:
  1. Determinism (repeat runs identical)
  2. No mutation of input DFA
  3. Correct trajectory length
  4. Valid basis vectors (one-hot)
  5. Qudit normalization (||psi|| = 1)
  6. Syndrome shape consistency
  7. Stable ordering
  8. Empty / single-state edge cases
"""

import copy

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qec.experiments.qudit_dynamics import (
    analyze_syndrome_evolution,
    embed_state_to_qudit,
    measure_trajectory,
    run_qudit_dynamics,
    state_to_basis_vector,
    trajectory_to_states,
)
from qudit_stabilizer import QuditStabilizerCode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_dfa():
    """Three-state DFA with deterministic transitions."""
    return {
        "states": [0, 1, 2],
        "alphabet": [10, 20],
        "transitions": {
            0: {10: 1, 20: 2},
            1: {10: 2, 20: 0},
            2: {10: 0, 20: 1},
        },
        "initial_state": 0,
    }


def _single_state_dfa():
    """Single-state self-loop DFA."""
    return {
        "states": [0],
        "alphabet": [1],
        "transitions": {0: {1: 0}},
        "initial_state": 0,
    }


def _absorbing_dfa():
    """DFA with a state that has no outgoing transitions."""
    return {
        "states": [0, 1],
        "alphabet": [5],
        "transitions": {
            0: {5: 1},
            1: {},
        },
        "initial_state": 0,
    }


def _make_stabilizer_code(d, n_qudits):
    """Build a simple stabilizer code for testing."""
    generators = []
    a = np.zeros(n_qudits, dtype=int)
    b = np.zeros(n_qudits, dtype=int)
    a[0] = 1
    generators.append((a.copy(), b.copy()))
    a2 = np.zeros(n_qudits, dtype=int)
    b2 = np.zeros(n_qudits, dtype=int)
    b2[0] = 1
    generators.append((a2.copy(), b2.copy()))
    return QuditStabilizerCode(d, n_qudits, generators)


# ---------------------------------------------------------------------------
# PART A — state_to_basis_vector
# ---------------------------------------------------------------------------


class TestStateToBasisVector:
    def test_one_hot(self):
        vec = state_to_basis_vector(1, 3)
        expected = np.array([0.0, 1.0, 0.0])
        np.testing.assert_array_equal(vec, expected)

    def test_first_state(self):
        vec = state_to_basis_vector(0, 4)
        assert vec[0] == 1.0
        assert np.sum(vec) == 1.0

    def test_last_state(self):
        vec = state_to_basis_vector(2, 3)
        assert vec[2] == 1.0
        assert np.count_nonzero(vec) == 1

    def test_invalid_state_id(self):
        with pytest.raises(ValueError):
            state_to_basis_vector(5, 3)

    def test_negative_state_id(self):
        with pytest.raises(ValueError):
            state_to_basis_vector(-1, 3)

    def test_zero_num_states(self):
        with pytest.raises(ValueError):
            state_to_basis_vector(0, 0)


# ---------------------------------------------------------------------------
# PART A — trajectory_to_states
# ---------------------------------------------------------------------------


class TestTrajectoryToStates:
    def test_length(self):
        dfa = _simple_dfa()
        traj = trajectory_to_states(dfa, 0, 5)
        assert len(traj) == 6  # steps + 1

    def test_starts_with_start_state(self):
        dfa = _simple_dfa()
        traj = trajectory_to_states(dfa, 0, 3)
        assert traj[0] == 0

    def test_determinism(self):
        dfa = _simple_dfa()
        t1 = trajectory_to_states(dfa, 0, 10)
        t2 = trajectory_to_states(dfa, 0, 10)
        assert t1 == t2

    def test_no_mutation(self):
        dfa = _simple_dfa()
        original = copy.deepcopy(dfa)
        trajectory_to_states(dfa, 0, 5)
        assert dfa == original

    def test_single_state_loop(self):
        dfa = _single_state_dfa()
        traj = trajectory_to_states(dfa, 0, 5)
        assert traj == [0, 0, 0, 0, 0, 0]

    def test_absorbing_state(self):
        dfa = _absorbing_dfa()
        traj = trajectory_to_states(dfa, 0, 4)
        # 0 -> 1, then stuck at 1
        assert traj == [0, 1, 1, 1, 1]

    def test_zero_steps(self):
        dfa = _simple_dfa()
        traj = trajectory_to_states(dfa, 0, 0)
        assert traj == [0]

    def test_uses_smallest_symbol(self):
        """Deterministic: always picks smallest symbol in sorted alphabet."""
        dfa = _simple_dfa()
        traj = trajectory_to_states(dfa, 0, 1)
        # alphabet sorted = [10, 20], so symbol 10 is used: 0 -> 1
        assert traj == [0, 1]


# ---------------------------------------------------------------------------
# PART B — embed_state_to_qudit
# ---------------------------------------------------------------------------


class TestEmbedStateToQudit:
    def test_normalization(self):
        vec = state_to_basis_vector(0, 3)
        qudit = embed_state_to_qudit(vec, 2)
        norm = np.sqrt(np.vdot(qudit, qudit).real)
        np.testing.assert_almost_equal(norm, 1.0)

    def test_dimension(self):
        vec = state_to_basis_vector(0, 3)
        qudit = embed_state_to_qudit(vec, 2)
        # d=2, need k=2 so 2^2=4 >= 3
        assert len(qudit) == 4

    def test_dimension_exact_power(self):
        vec = state_to_basis_vector(0, 4)
        qudit = embed_state_to_qudit(vec, 2)
        # d=2, k=2, 2^2=4 == 4
        assert len(qudit) == 4

    def test_correct_index(self):
        vec = state_to_basis_vector(2, 5)
        qudit = embed_state_to_qudit(vec, 3)
        # d=3, k=2, 3^2=9 >= 5
        assert len(qudit) == 9
        assert qudit[2] == 1.0 + 0j
        assert np.count_nonzero(qudit) == 1

    def test_invalid_d(self):
        vec = state_to_basis_vector(0, 3)
        with pytest.raises(ValueError):
            embed_state_to_qudit(vec, 1)

    def test_determinism(self):
        vec = state_to_basis_vector(1, 3)
        q1 = embed_state_to_qudit(vec, 2)
        q2 = embed_state_to_qudit(vec, 2)
        np.testing.assert_array_equal(q1, q2)


# ---------------------------------------------------------------------------
# PART C — measure_trajectory
# ---------------------------------------------------------------------------


class TestMeasureTrajectory:
    def test_empty_trajectory(self):
        result = measure_trajectory([], 2, None)
        assert result["states"] == []
        assert result["stabilizer_values"] == []
        assert result["syndromes"] == []

    def test_result_keys(self):
        dfa = _simple_dfa()
        traj = trajectory_to_states(dfa, 0, 3)
        # d=2, need k=2 for 3 states -> dim=4, n_qudits=2
        code = _make_stabilizer_code(2, 2)
        result = measure_trajectory(traj, 2, code)
        assert "states" in result
        assert "stabilizer_values" in result
        assert "syndromes" in result

    def test_lengths_match_trajectory(self):
        dfa = _simple_dfa()
        traj = trajectory_to_states(dfa, 0, 4)
        code = _make_stabilizer_code(2, 2)
        result = measure_trajectory(traj, 2, code)
        assert len(result["states"]) == len(traj)
        assert len(result["stabilizer_values"]) == len(traj)
        assert len(result["syndromes"]) == len(traj)

    def test_syndrome_shape(self):
        dfa = _simple_dfa()
        traj = trajectory_to_states(dfa, 0, 2)
        code = _make_stabilizer_code(2, 2)
        result = measure_trajectory(traj, 2, code)
        for synd in result["syndromes"]:
            assert synd.shape == (2,)  # 2 generators

    def test_determinism(self):
        dfa = _simple_dfa()
        traj = trajectory_to_states(dfa, 0, 3)
        code = _make_stabilizer_code(2, 2)
        r1 = measure_trajectory(traj, 2, code)
        r2 = measure_trajectory(traj, 2, code)
        for s1, s2 in zip(r1["syndromes"], r2["syndromes"]):
            np.testing.assert_array_equal(s1, s2)

    def test_qudit_normalization(self):
        dfa = _simple_dfa()
        traj = trajectory_to_states(dfa, 0, 3)
        code = _make_stabilizer_code(2, 2)
        result = measure_trajectory(traj, 2, code)
        for state in result["states"]:
            norm = np.sqrt(np.vdot(state, state).real)
            np.testing.assert_almost_equal(norm, 1.0)


# ---------------------------------------------------------------------------
# PART D — analyze_syndrome_evolution
# ---------------------------------------------------------------------------


class TestAnalyzeSyndromeEvolution:
    def test_empty(self):
        result = analyze_syndrome_evolution([])
        assert result["unique_syndromes"] == []
        assert result["transition_count"] == 0
        assert result["stable_regions"] == []
        assert result["change_points"] == []

    def test_single_syndrome(self):
        syndromes = [np.array([0, 1])]
        result = analyze_syndrome_evolution(syndromes)
        assert result["unique_syndromes"] == [(0, 1)]
        assert result["transition_count"] == 0
        assert result["stable_regions"] == [(0, 1)]
        assert result["change_points"] == []

    def test_all_same(self):
        syndromes = [np.array([0, 0])] * 5
        result = analyze_syndrome_evolution(syndromes)
        assert len(result["unique_syndromes"]) == 1
        assert result["transition_count"] == 0
        assert result["stable_regions"] == [(0, 5)]

    def test_all_different(self):
        syndromes = [
            np.array([0]),
            np.array([1]),
            np.array([2]),
        ]
        result = analyze_syndrome_evolution(syndromes)
        assert len(result["unique_syndromes"]) == 3
        assert result["transition_count"] == 2
        assert result["change_points"] == [1, 2]
        assert result["stable_regions"] == [(0, 1), (1, 1), (2, 1)]

    def test_runs(self):
        syndromes = [
            np.array([0]),
            np.array([0]),
            np.array([1]),
            np.array([1]),
            np.array([1]),
            np.array([0]),
        ]
        result = analyze_syndrome_evolution(syndromes)
        assert result["stable_regions"] == [(0, 2), (2, 3), (5, 1)]
        assert result["change_points"] == [2, 5]
        assert result["transition_count"] == 2

    def test_unique_sorted(self):
        syndromes = [
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([1, 0]),
        ]
        result = analyze_syndrome_evolution(syndromes)
        assert result["unique_syndromes"] == [(0, 1), (1, 0)]

    def test_determinism(self):
        syndromes = [np.array([i % 3]) for i in range(10)]
        r1 = analyze_syndrome_evolution(syndromes)
        r2 = analyze_syndrome_evolution(syndromes)
        assert r1 == r2


# ---------------------------------------------------------------------------
# PART E — run_qudit_dynamics (full pipeline)
# ---------------------------------------------------------------------------


class TestRunQuditDynamics:
    def test_full_pipeline(self):
        dfa = _simple_dfa()
        code = _make_stabilizer_code(2, 2)
        result = run_qudit_dynamics(dfa, 0, 5, 2, code)
        assert "trajectory" in result
        assert "qudit" in result
        assert "syndrome_analysis" in result
        assert len(result["trajectory"]) == 6

    def test_determinism(self):
        dfa = _simple_dfa()
        code = _make_stabilizer_code(2, 2)
        r1 = run_qudit_dynamics(dfa, 0, 5, 2, code)
        r2 = run_qudit_dynamics(dfa, 0, 5, 2, code)
        assert r1["trajectory"] == r2["trajectory"]
        assert r1["syndrome_analysis"] == r2["syndrome_analysis"]
        for s1, s2 in zip(r1["qudit"]["syndromes"], r2["qudit"]["syndromes"]):
            np.testing.assert_array_equal(s1, s2)

    def test_no_mutation(self):
        dfa = _simple_dfa()
        original = copy.deepcopy(dfa)
        code = _make_stabilizer_code(2, 2)
        run_qudit_dynamics(dfa, 0, 5, 2, code)
        assert dfa == original

    def test_single_state(self):
        dfa = _single_state_dfa()
        code = _make_stabilizer_code(2, 1)
        result = run_qudit_dynamics(dfa, 0, 3, 2, code)
        assert result["trajectory"] == [0, 0, 0, 0]
        assert result["syndrome_analysis"]["transition_count"] == 0

    def test_qudit_states_normalized(self):
        dfa = _simple_dfa()
        code = _make_stabilizer_code(2, 2)
        result = run_qudit_dynamics(dfa, 0, 3, 2, code)
        for state in result["qudit"]["states"]:
            norm = np.sqrt(np.vdot(state, state).real)
            np.testing.assert_almost_equal(norm, 1.0)
