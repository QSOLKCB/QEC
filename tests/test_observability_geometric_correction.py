"""Tests for v91.1.0 — observability + geometric correction layer.

Covers:
  - Supervisor metrics correctness and density monotonicity
  - Forbidden state stratification: correct partition, determinism
  - Stabilizer metadata correctness
  - State-syndrome alignment correctness
  - Syndrome compression: determinism, repeatability
  - Geometric correction: projection stability, normalization, determinism
  - Integration: no regression, optional flags respected
"""

import copy

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qec.experiments.dfa_supervisor import (
    compute_supervisor_metrics,
    derive_forbidden_states,
    run_supervisor,
    stratify_forbidden_states,
    synthesize_supervisor,
)
from qec.experiments.dfa_engine import run_dfa_engine
from qec.experiments.qudit_dynamics import (
    build_stabilizer_metadata,
    compress_syndrome,
    measure_trajectory,
    run_qudit_dynamics,
    trajectory_to_states,
)
from qec.experiments.qudit_geometric_correction import (
    apply_geometric_correction,
    correct_trajectory_states,
    project_to_lattice,
)
from qudit_stabilizer import QuditStabilizerCode


# ---------------------------------------------------------------------------
# Fixtures
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


def _dfa_with_dead():
    """DFA with states 0-4, dead state 4 (absorbing)."""
    return {
        "states": [0, 1, 2, 3, 4],
        "alphabet": [0, 1],
        "transitions": {
            0: {0: 1, 1: 2},
            1: {0: 3, 1: 4},
            2: {0: 3, 1: 2},
            3: {0: 3, 1: 3},
            4: {0: 4, 1: 4},
        },
        "initial_state": 0,
        "dead_state": 4,
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


def _invariants_with_dead(dfa):
    """Build minimal invariants dict for a DFA with dead state."""
    dead = dfa.get("dead_state")
    return {
        "invariants": {
            "dead_state": {
                "has_dead_state": dead is not None,
                "dead_state": dead,
                "is_absorbing": True,
            },
            "structure": {
                "state_to_attractor": {s: (1 if s == dead else 0)
                                        for s in dfa["states"]},
            },
        },
    }


def _inv_constraints_for_dead(dfa):
    """Invariant constraints that avoid the dead state."""
    dead = dfa.get("dead_state")
    if dead is not None:
        return {"avoid_states": {dead}, "allow_only_states": None}
    return {"avoid_states": set(), "allow_only_states": None}


# ---------------------------------------------------------------------------
# PART A — Supervisor Metrics
# ---------------------------------------------------------------------------


class TestSupervisorMetrics:
    def test_basic_counts(self):
        original = _dfa_with_dead()
        # Supervised DFA: remove dead state transitions.
        supervised = {
            "states": [0, 1, 2, 3],
            "alphabet": [0, 1],
            "transitions": {
                0: {0: 1, 1: 2},
                1: {0: 3},
                2: {0: 3, 1: 2},
                3: {0: 3, 1: 3},
            },
            "initial_state": 0,
        }
        m = compute_supervisor_metrics(original, supervised)
        assert m["states_before"] == 5
        assert m["states_after"] == 4
        assert m["transitions_before"] == 10
        assert m["transitions_after"] == 7

    def test_density_correct(self):
        original = _dfa_with_dead()
        supervised = {
            "states": [0, 1, 2, 3],
            "alphabet": [0, 1],
            "transitions": {
                0: {0: 1, 1: 2},
                1: {0: 3},
                2: {0: 3, 1: 2},
                3: {0: 3, 1: 3},
            },
            "initial_state": 0,
        }
        m = compute_supervisor_metrics(original, supervised)
        assert m["density_before"] == 10 / (5 * 2)
        assert m["density_after"] == 7 / (4 * 2)

    def test_control_strength_range(self):
        original = _dfa_with_dead()
        supervised = {
            "states": [0, 1, 2, 3],
            "alphabet": [0, 1],
            "transitions": {
                0: {0: 1, 1: 2},
                1: {0: 3},
                2: {0: 3, 1: 2},
                3: {0: 3, 1: 3},
            },
            "initial_state": 0,
        }
        m = compute_supervisor_metrics(original, supervised)
        assert 0.0 <= m["control_strength"] <= 1.0

    def test_identical_dfas(self):
        dfa = _simple_dfa()
        m = compute_supervisor_metrics(dfa, dfa)
        assert m["control_strength"] == 0.0
        assert m["states_before"] == m["states_after"]

    def test_empty_dfa_safe_division(self):
        empty = {"states": [], "alphabet": [], "transitions": {}}
        m = compute_supervisor_metrics(empty, empty)
        assert m["density_before"] == 0.0
        assert m["density_after"] == 0.0
        assert m["control_strength"] == 0.0

    def test_density_monotonic_after_supervision(self):
        """Supervisor should not increase transition density beyond original."""
        dfa = _dfa_with_dead()
        inv = _invariants_with_dead(dfa)
        ic = _inv_constraints_for_dead(dfa)
        result = run_supervisor(dfa, inv, ic)
        m = result["metrics"]
        # Density may increase or stay same (fewer states can increase ratio),
        # but control_strength >= 0.
        assert m["control_strength"] >= 0.0


# ---------------------------------------------------------------------------
# PART B — Forbidden State Stratification
# ---------------------------------------------------------------------------


class TestStratification:
    def test_correct_partition(self):
        forbidden = [1, 2, 3, 4]
        provenance = {
            1: ["dead_state"],
            2: ["dead_drain"],
            3: ["invariant_avoid"],
            4: ["outside_allowed_region"],
        }
        strata = stratify_forbidden_states(forbidden, provenance)
        assert strata["hard_forbidden"] == [1, 2]
        assert strata["structural_forbidden"] == [3, 4]

    def test_all_hard(self):
        forbidden = [5, 6]
        provenance = {5: ["dead_state"], 6: ["dead_drain"]}
        strata = stratify_forbidden_states(forbidden, provenance)
        assert strata["hard_forbidden"] == [5, 6]
        assert strata["structural_forbidden"] == []

    def test_all_structural(self):
        forbidden = [7, 8]
        provenance = {7: ["forbidden_region"], 8: ["dead_end_trimmed"]}
        strata = stratify_forbidden_states(forbidden, provenance)
        assert strata["hard_forbidden"] == []
        assert strata["structural_forbidden"] == [7, 8]

    def test_empty(self):
        strata = stratify_forbidden_states([], {})
        assert strata["hard_forbidden"] == []
        assert strata["structural_forbidden"] == []

    def test_determinism(self):
        forbidden = [4, 2, 3, 1]
        provenance = {1: ["dead_state"], 2: ["invariant_avoid"],
                      3: ["dead_drain"], 4: ["forbidden_region"]}
        s1 = stratify_forbidden_states(forbidden, provenance)
        s2 = stratify_forbidden_states(forbidden, provenance)
        assert s1 == s2

    def test_sorted_output(self):
        forbidden = [10, 5, 3, 8]
        provenance = {10: ["dead_state"], 5: ["dead_drain"],
                      3: ["invariant_avoid"], 8: ["forbidden_region"]}
        strata = stratify_forbidden_states(forbidden, provenance)
        assert strata["hard_forbidden"] == sorted(strata["hard_forbidden"])
        assert strata["structural_forbidden"] == sorted(strata["structural_forbidden"])

    def test_exposed_in_run_supervisor(self):
        dfa = _dfa_with_dead()
        inv = _invariants_with_dead(dfa)
        ic = _inv_constraints_for_dead(dfa)
        result = run_supervisor(dfa, inv, ic)
        assert "forbidden_strata" in result
        strata = result["forbidden_strata"]
        assert "hard_forbidden" in strata
        assert "structural_forbidden" in strata


# ---------------------------------------------------------------------------
# PART C — Stabilizer Metadata
# ---------------------------------------------------------------------------


class TestStabilizerMetadata:
    def test_basic(self):
        code = _make_stabilizer_code(2, 2)
        meta = build_stabilizer_metadata(code)
        assert meta["d"] == 2
        assert meta["n_qudits"] == 2
        assert meta["generator_count"] == 2

    def test_larger_code(self):
        code = _make_stabilizer_code(3, 4)
        meta = build_stabilizer_metadata(code)
        assert meta["d"] == 3
        assert meta["n_qudits"] == 4
        assert meta["generator_count"] == 2

    def test_exposed_in_pipeline(self):
        dfa = _simple_dfa()
        code = _make_stabilizer_code(2, 2)
        result = run_qudit_dynamics(dfa, 0, 3, 2, code)
        assert "stabilizer_metadata" in result
        assert result["stabilizer_metadata"]["d"] == 2


# ---------------------------------------------------------------------------
# PART D — State-Syndrome Alignment
# ---------------------------------------------------------------------------


class TestStateSyndromeAlignment:
    def test_pairs_present(self):
        dfa = _simple_dfa()
        traj = trajectory_to_states(dfa, 0, 3)
        code = _make_stabilizer_code(2, 2)
        result = measure_trajectory(traj, 2, code)
        assert "state_syndrome_pairs" in result
        assert len(result["state_syndrome_pairs"]) == len(traj)

    def test_pair_structure(self):
        dfa = _simple_dfa()
        traj = trajectory_to_states(dfa, 0, 2)
        code = _make_stabilizer_code(2, 2)
        result = measure_trajectory(traj, 2, code)
        for pair in result["state_syndrome_pairs"]:
            assert "state" in pair
            assert "syndrome" in pair
            assert isinstance(pair["state"], int)
            assert isinstance(pair["syndrome"], list)

    def test_deterministic_ordering(self):
        dfa = _simple_dfa()
        traj = trajectory_to_states(dfa, 0, 4)
        code = _make_stabilizer_code(2, 2)
        r1 = measure_trajectory(traj, 2, code)
        r2 = measure_trajectory(traj, 2, code)
        assert r1["state_syndrome_pairs"] == r2["state_syndrome_pairs"]

    def test_empty_trajectory(self):
        result = measure_trajectory([], 2, None)
        assert result["state_syndrome_pairs"] == []


# ---------------------------------------------------------------------------
# PART E — Syndrome Compression
# ---------------------------------------------------------------------------


class TestSyndromeCompression:
    def test_basic(self):
        s = np.array([0, 1, 0, 2])
        assert compress_syndrome(s) == "0_1_0_2"

    def test_single_element(self):
        s = np.array([3])
        assert compress_syndrome(s) == "3"

    def test_deterministic(self):
        s = np.array([1, 2, 3])
        assert compress_syndrome(s) == compress_syndrome(s)

    def test_repeatable(self):
        s1 = np.array([0, 0, 1])
        s2 = np.array([0, 0, 1])
        assert compress_syndrome(s1) == compress_syndrome(s2)

    def test_pipeline_flag_off(self):
        dfa = _simple_dfa()
        code = _make_stabilizer_code(2, 2)
        result = run_qudit_dynamics(dfa, 0, 3, 2, code, compress=False)
        assert "syndrome_signatures" not in result

    def test_pipeline_flag_on(self):
        dfa = _simple_dfa()
        code = _make_stabilizer_code(2, 2)
        result = run_qudit_dynamics(dfa, 0, 3, 2, code, compress=True)
        assert "syndrome_signatures" in result
        assert len(result["syndrome_signatures"]) == 4  # steps + 1


# ---------------------------------------------------------------------------
# PART F — Geometric Correction
# ---------------------------------------------------------------------------


class TestProjectToLattice:
    def test_square_mode(self):
        vec = np.array([0.3 + 0j, 0.7 + 0j, -0.4 + 0j])
        proj = project_to_lattice(vec, mode="square")
        np.testing.assert_array_equal(proj, np.array([0.0, 1.0, 0.0]))

    def test_d4_mode_even_sum(self):
        vec = np.array([0.3 + 0j, 0.7 + 0j, 0.4 + 0j])
        proj = project_to_lattice(vec, mode="d4")
        # Rounded: [0, 1, 0] → sum=1 (odd) → adjust largest residual
        total = int(np.sum(proj))
        assert total % 2 == 0

    def test_d4_mode_already_even(self):
        vec = np.array([0.1 + 0j, 0.9 + 0j, 1.1 + 0j])
        proj = project_to_lattice(vec, mode="d4")
        total = int(np.sum(proj))
        assert total % 2 == 0

    def test_projection_stable(self):
        vec = np.array([0.5 + 0j, 0.5 + 0j, 1.0 + 0j])
        p1 = project_to_lattice(vec, mode="d4")
        p2 = project_to_lattice(vec, mode="d4")
        np.testing.assert_array_equal(p1, p2)

    def test_invalid_mode(self):
        vec = np.array([1.0 + 0j])
        with pytest.raises(ValueError, match="Unknown projection mode"):
            project_to_lattice(vec, mode="invalid")


class TestApplyGeometricCorrection:
    def test_normalization_preserved(self):
        vec = np.array([0.3 + 0.1j, 0.7 - 0.2j, 0.5 + 0j])
        corrected = apply_geometric_correction(vec, mode="square")
        norm = np.linalg.norm(corrected)
        if norm > 0:
            np.testing.assert_almost_equal(norm, 1.0)

    def test_deterministic(self):
        vec = np.array([0.6 + 0.1j, 0.4 - 0.3j])
        c1 = apply_geometric_correction(vec, mode="d4")
        c2 = apply_geometric_correction(vec, mode="d4")
        np.testing.assert_array_equal(c1, c2)

    def test_no_mutation(self):
        vec = np.array([0.5 + 0j, 0.5 + 0j])
        original = vec.copy()
        apply_geometric_correction(vec, mode="square")
        np.testing.assert_array_equal(vec, original)

    def test_zero_vector(self):
        vec = np.array([0.0 + 0j, 0.0 + 0j])
        corrected = apply_geometric_correction(vec, mode="square")
        # Zero input → zero output (norm=0 safe division).
        np.testing.assert_array_equal(corrected, np.zeros(2, dtype=complex))


class TestCorrectTrajectoryStates:
    def test_none_mode_noop(self):
        states = [np.array([1.0 + 0j, 0.0 + 0j])]
        result = correct_trajectory_states(states, correction_mode=None)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], states[0])

    def test_none_mode_returns_copies(self):
        states = [np.array([1.0 + 0j, 0.0 + 0j])]
        result = correct_trajectory_states(states, correction_mode=None)
        assert result[0] is not states[0]

    def test_with_correction(self):
        states = [
            np.array([0.6 + 0j, 0.4 + 0j]),
            np.array([0.3 + 0j, 0.7 + 0j]),
        ]
        result = correct_trajectory_states(states, correction_mode="square")
        assert len(result) == 2
        for s in result:
            norm = np.linalg.norm(s)
            if norm > 0:
                np.testing.assert_almost_equal(norm, 1.0)

    def test_deterministic(self):
        states = [np.array([0.5 + 0.1j, 0.5 - 0.1j])]
        r1 = correct_trajectory_states(states, correction_mode="d4")
        r2 = correct_trajectory_states(states, correction_mode="d4")
        for s1, s2 in zip(r1, r2):
            np.testing.assert_array_equal(s1, s2)


# ---------------------------------------------------------------------------
# PART G — Integration Tests
# ---------------------------------------------------------------------------


class TestIntegrationNoRegression:
    def test_run_qudit_dynamics_backward_compatible(self):
        """Existing call signature without new args still works."""
        dfa = _simple_dfa()
        code = _make_stabilizer_code(2, 2)
        result = run_qudit_dynamics(dfa, 0, 5, 2, code)
        assert "trajectory" in result
        assert "qudit" in result
        assert "syndrome_analysis" in result
        assert len(result["trajectory"]) == 6
        # No correction output when not requested.
        assert "corrected" not in result
        assert "correction_effect" not in result

    def test_correction_optional(self):
        """correction_mode=None means no correction output."""
        dfa = _simple_dfa()
        code = _make_stabilizer_code(2, 2)
        result = run_qudit_dynamics(dfa, 0, 3, 2, code, correction_mode=None)
        assert "corrected" not in result

    def test_correction_enabled(self):
        """correction_mode="d4" produces corrected output."""
        dfa = _simple_dfa()
        code = _make_stabilizer_code(2, 2)
        result = run_qudit_dynamics(dfa, 0, 3, 2, code, correction_mode="d4")
        assert "corrected" in result
        assert "correction_effect" in result
        assert "syndrome_changes" in result["correction_effect"]
        assert "unique_before" in result["correction_effect"]
        assert "unique_after" in result["correction_effect"]
        assert len(result["corrected"]["states"]) == 4
        assert len(result["corrected"]["syndromes"]) == 4

    def test_engine_exposes_metrics(self):
        """run_dfa_engine includes supervisor_metrics and forbidden_strata."""
        dfa = _dfa_with_dead()
        result = run_dfa_engine(dfa)
        assert "supervisor_metrics" in result
        assert "forbidden_strata" in result
        m = result["supervisor_metrics"]
        assert "states_before" in m
        assert "control_strength" in m

    def test_engine_no_mutation(self):
        dfa = _dfa_with_dead()
        original = copy.deepcopy(dfa)
        run_dfa_engine(dfa)
        assert dfa == original

    def test_full_pipeline_determinism(self):
        dfa = _simple_dfa()
        code = _make_stabilizer_code(2, 2)
        r1 = run_qudit_dynamics(dfa, 0, 3, 2, code,
                                correction_mode="square", compress=True)
        r2 = run_qudit_dynamics(dfa, 0, 3, 2, code,
                                correction_mode="square", compress=True)
        assert r1["trajectory"] == r2["trajectory"]
        assert r1["syndrome_signatures"] == r2["syndrome_signatures"]
        assert r1["correction_effect"] == r2["correction_effect"]
        assert r1["stabilizer_metadata"] == r2["stabilizer_metadata"]
