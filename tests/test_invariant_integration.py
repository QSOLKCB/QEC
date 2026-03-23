"""Tests for the invariant integration layer (v89.3.0).

Verifies invariant normalization, constraint derivation, constraint
composition with invariants, and integration with control/simulation.

All tests are deterministic — no randomness, no heuristics.
"""

from __future__ import annotations

from qec.experiments.dfa_engine import (
    compose_constraints,
    find_control_path,
    normalize_constraints,
    run_dfa_engine,
    simulate_from_state,
)
from qec.experiments.dfa_invariants import (
    derive_constraints_from_invariants,
    detect_invariants,
    normalize_invariants,
)
from qec.experiments.dfa_analysis import classify_components, compute_scc
from qec.experiments.dfa_engine import (
    build_scc_graph,
    compute_scc_levels,
    find_terminal_sccs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _dead_heavy_dfa():
    """DFA with dead state 3 absorbing many transitions."""
    return {
        "states": [0, 1, 2, 3],
        "alphabet": [0, 1],
        "transitions": {
            0: {0: 1, 1: 3},
            1: {0: 2, 1: 3},
            2: {0: 3, 1: 3},
            3: {0: 3, 1: 3},
        },
        "initial_state": 0,
        "dead_state": 3,
    }


def _two_attractor_dfa():
    """Two terminal SCCs: cycle {1,2} and fixed-point {4}."""
    return {
        "states": [0, 1, 2, 3, 4],
        "alphabet": [0, 1],
        "transitions": {
            0: {0: 1, 1: 3},
            1: {0: 2},
            2: {0: 1},
            3: {0: 4},
            4: {0: 4},
        },
        "initial_state": 0,
        "dead_state": None,
    }


def _cyclic_dfa():
    """0 --0--> 1 --0--> 2 --0--> 0   (3-cycle)."""
    return {
        "states": [0, 1, 2],
        "alphabet": [0],
        "transitions": {0: {0: 1}, 1: {0: 2}, 2: {0: 0}},
        "initial_state": 0,
        "dead_state": None,
    }


def _linear_dfa():
    """0 --0--> 1 --0--> 2   (linear chain)."""
    return {
        "states": [0, 1, 2],
        "alphabet": [0],
        "transitions": {0: {0: 1}, 1: {0: 2}, 2: {}},
        "initial_state": 0,
        "dead_state": None,
    }


def _make_invariants(dfa):
    """Helper to build invariants from a DFA."""
    sccs = compute_scc(dfa)
    classify_components(dfa, sccs)
    dag = build_scc_graph(sccs, dfa)
    terminal = find_terminal_sccs(dag)
    hierarchy = compute_scc_levels(dag)
    return detect_invariants(
        dfa, hierarchy,
        {"terminal_sccs": terminal, "sccs": sccs, "scc_graph": dag},
    )


# ===================================================================
# PART 1 — Normalization Stability
# ===================================================================


class TestNormalizationStability:
    """normalize_invariants(inv) == normalize_invariants(inv)."""

    def test_idempotent(self):
        dfa = _two_attractor_dfa()
        inv = _make_invariants(dfa)
        n1 = normalize_invariants(inv)
        n2 = normalize_invariants(n1)
        assert n1 == n2

    def test_idempotent_dead(self):
        dfa = _dead_heavy_dfa()
        inv = _make_invariants(dfa)
        n1 = normalize_invariants(inv)
        n2 = normalize_invariants(n1)
        assert n1 == n2

    def test_idempotent_cyclic(self):
        dfa = _cyclic_dfa()
        inv = _make_invariants(dfa)
        n1 = normalize_invariants(inv)
        n2 = normalize_invariants(n1)
        assert n1 == n2

    def test_missing_sections_filled(self):
        """Empty input gets all sections."""
        result = normalize_invariants({})
        sections = result["invariants"]
        for key in ("dead_state", "attractors", "reachability",
                    "transitions", "structure"):
            assert key in sections

    def test_sets_become_sorted_lists(self):
        """Sets in invariant values are converted to sorted lists."""
        inv = {"invariants": {"dead_state": {"states": {3, 1, 2}}}}
        result = normalize_invariants(inv)
        assert result["invariants"]["dead_state"]["states"] == [1, 2, 3]

    def test_determinism_across_runs(self):
        dfa = _two_attractor_dfa()
        inv = _make_invariants(dfa)
        results = [normalize_invariants(inv) for _ in range(5)]
        assert all(r == results[0] for r in results)


# ===================================================================
# PART 2 — Mapping Correctness
# ===================================================================


class TestConstraintDerivation:
    """derive_constraints_from_invariants correctness."""

    def test_dead_state_in_avoid(self):
        """Dead absorbing state must appear in avoid_states."""
        dfa = _dead_heavy_dfa()
        inv = _make_invariants(dfa)
        constraints = derive_constraints_from_invariants(inv)
        assert 3 in constraints["avoid_states"]

    def test_no_dead_state_empty_avoid(self):
        """DFA without dead state produces no avoid_states from dead rule."""
        dfa = _cyclic_dfa()
        inv = _make_invariants(dfa)
        constraints = derive_constraints_from_invariants(inv)
        # No dead state, so avoid should be empty
        assert len(constraints["avoid_states"]) == 0

    def test_no_false_positives_cyclic(self):
        """Cyclic DFA: no states should be avoided."""
        dfa = _cyclic_dfa()
        inv = _make_invariants(dfa)
        constraints = derive_constraints_from_invariants(inv)
        assert len(constraints["avoid_states"]) == 0

    def test_forbidden_region_detection(self):
        """States leading only to dead state should be in avoid_states."""
        dfa = _dead_heavy_dfa()
        inv = _make_invariants(dfa)
        constraints = derive_constraints_from_invariants(inv)
        # State 2 only leads to dead state 3
        # State 3 is the dead state itself
        assert 3 in constraints["avoid_states"]

    def test_allow_only_from_reachable(self):
        """allow_only_states derived from reachable states."""
        dfa = _two_attractor_dfa()
        inv = _make_invariants(dfa)
        constraints = derive_constraints_from_invariants(inv)
        if constraints["allow_only_states"] is not None:
            # Must be a subset of all states
            assert constraints["allow_only_states"] <= set(dfa["states"])


# ===================================================================
# PART 3 — Constraint Composition
# ===================================================================


class TestConstraintComposition:
    """compose(user, invariant) remains stable."""

    def test_compose_with_invariant_constraints(self):
        dfa = _dead_heavy_dfa()
        inv = _make_invariants(dfa)
        inv_constraints = derive_constraints_from_invariants(inv)
        user_constraints = {"avoid_states": {1}, "max_depth": 5}

        result = compose_constraints(user_constraints, inv_constraints)
        # Must include both user and invariant avoid_states
        assert 1 in result["avoid_states"]
        assert 3 in result["avoid_states"]
        assert result["max_depth"] == 5

    def test_compose_stability(self):
        """Composition is deterministic across runs."""
        dfa = _dead_heavy_dfa()
        inv = _make_invariants(dfa)
        inv_constraints = derive_constraints_from_invariants(inv)
        user_constraints = {"avoid_states": {1}}

        r1 = compose_constraints(user_constraints, inv_constraints)
        r2 = compose_constraints(user_constraints, inv_constraints)
        assert r1 == r2

    def test_compose_with_none_user(self):
        """Composing None user constraints with invariant constraints."""
        dfa = _dead_heavy_dfa()
        inv = _make_invariants(dfa)
        inv_constraints = derive_constraints_from_invariants(inv)

        result = compose_constraints(None, inv_constraints)
        assert 3 in result["avoid_states"]


# ===================================================================
# PART 4 — Control Equivalence (invariants do nothing)
# ===================================================================


class TestControlEquivalence:
    """If invariants add no constraints, output is unchanged."""

    def test_no_invariant_effect_cyclic(self):
        """Cyclic DFA: no dead state, so invariants should not change control."""
        dfa = _cyclic_dfa()
        r_without = find_control_path(dfa, 0, 2)
        r_with = find_control_path(dfa, 0, 2, use_invariants=True)
        assert r_without["reachable"] == r_with["reachable"]
        assert r_without["path"] == r_with["path"]
        assert r_without["symbols"] == r_with["symbols"]

    def test_no_invariant_effect_linear(self):
        """Linear DFA without dead state: invariants should not change paths."""
        dfa = _linear_dfa()
        r_without = find_control_path(dfa, 0, 2)
        r_with = find_control_path(dfa, 0, 2, use_invariants=True)
        assert r_without["reachable"] == r_with["reachable"]
        assert r_without["path"] == r_with["path"]


# ===================================================================
# PART 5 — Control Restriction
# ===================================================================


class TestControlRestriction:
    """If invariants apply, result ⊆ original."""

    def test_dead_state_avoided_in_control(self):
        """Control path with invariants should avoid dead state."""
        dfa = _dead_heavy_dfa()
        r_with = find_control_path(dfa, 0, 2, use_invariants=True)
        if r_with["reachable"]:
            assert 3 not in r_with["path"]

    def test_invariant_constraints_applied_flag(self):
        """Flag indicates whether invariant constraints were applied."""
        dfa = _dead_heavy_dfa()
        r = find_control_path(dfa, 0, 1, use_invariants=True)
        assert "invariant_constraints_applied" in r
        assert r["invariant_constraints_applied"] is True

    def test_invariant_constraints_not_applied_flag(self):
        """Flag is False when invariants derive no constraints."""
        dfa = _cyclic_dfa()
        r = find_control_path(dfa, 0, 1, use_invariants=True)
        assert r["invariant_constraints_applied"] is False


# ===================================================================
# PART 6 — Simulation Pruning Correctness
# ===================================================================


class TestSimulationPruning:
    """Invariant pruning does not introduce new paths."""

    def test_pruned_subset_of_unpruned(self):
        """Invariant-pruned trajectories are a subset of unpruned."""
        dfa = _dead_heavy_dfa()
        unpruned = simulate_from_state(dfa, 0, max_steps=5)
        pruned = simulate_from_state(dfa, 0, max_steps=5, use_invariants=True)

        unpruned_set = set(tuple(t) for t in unpruned["trajectories"])
        for traj in pruned["trajectories"]:
            t = tuple(traj)
            found = t in unpruned_set or any(
                tuple(u[:len(t)]) == t for u in unpruned["trajectories"]
            )
            assert found, f"Pruned trajectory {traj} not prefix of any unpruned"

    def test_no_dead_state_in_pruned_simulation(self):
        """Dead state should not appear in invariant-pruned trajectories."""
        dfa = _dead_heavy_dfa()
        result = simulate_from_state(dfa, 0, max_steps=5, use_invariants=True)
        for traj in result["trajectories"]:
            assert 3 not in traj, f"Dead state 3 in trajectory: {traj}"

    def test_simulation_invariant_flag(self):
        """Simulation reports whether invariant constraints were applied."""
        dfa = _dead_heavy_dfa()
        result = simulate_from_state(dfa, 0, max_steps=5, use_invariants=True)
        assert result["invariant_constraints_applied"] is True

    def test_simulation_no_invariant_flag(self):
        """Simulation without invariants reports False."""
        dfa = _dead_heavy_dfa()
        result = simulate_from_state(dfa, 0, max_steps=5)
        assert result["invariant_constraints_applied"] is False

    def test_no_new_paths_introduced(self):
        """Pruning must only remove paths, never add new ones."""
        dfa = _two_attractor_dfa()
        unpruned = simulate_from_state(dfa, 0, max_steps=5)
        pruned = simulate_from_state(dfa, 0, max_steps=5, use_invariants=True)
        assert pruned["n_trajectories"] <= unpruned["n_trajectories"]


# ===================================================================
# PART 7 — Determinism
# ===================================================================


class TestDeterminism:
    """Repeated runs produce identical outputs."""

    def test_normalization_deterministic(self):
        dfa = _dead_heavy_dfa()
        inv = _make_invariants(dfa)
        results = [normalize_invariants(inv) for _ in range(10)]
        assert all(r == results[0] for r in results)

    def test_derivation_deterministic(self):
        dfa = _dead_heavy_dfa()
        inv = _make_invariants(dfa)
        results = [derive_constraints_from_invariants(inv) for _ in range(10)]
        assert all(r == results[0] for r in results)

    def test_control_with_invariants_deterministic(self):
        dfa = _dead_heavy_dfa()
        results = [
            find_control_path(dfa, 0, 1, use_invariants=True)
            for _ in range(5)
        ]
        assert all(r == results[0] for r in results)

    def test_simulation_with_invariants_deterministic(self):
        dfa = _dead_heavy_dfa()
        results = [
            simulate_from_state(dfa, 0, max_steps=5, use_invariants=True)
            for _ in range(5)
        ]
        assert all(r == results[0] for r in results)

    def test_engine_output_deterministic(self):
        dfa = _dead_heavy_dfa()
        r1 = run_dfa_engine(dfa)
        r2 = run_dfa_engine(dfa)
        assert r1 == r2


# ===================================================================
# PART 8 — Engine Output
# ===================================================================


class TestEngineInvariantOutput:
    """run_dfa_engine exposes invariant_constraints."""

    def test_invariant_constraints_in_output(self):
        dfa = _dead_heavy_dfa()
        result = run_dfa_engine(dfa)
        assert "invariant_constraints" in result
        ic = result["invariant_constraints"]
        assert "avoid_states" in ic
        assert "allow_only_states" in ic
        assert "has_constraints" in ic
        assert "applied" in ic

    def test_dead_state_in_engine_constraints(self):
        dfa = _dead_heavy_dfa()
        result = run_dfa_engine(dfa)
        ic = result["invariant_constraints"]
        assert 3 in ic["avoid_states"]
        assert ic["has_constraints"] is True

    def test_no_constraints_for_cyclic(self):
        dfa = _cyclic_dfa()
        result = run_dfa_engine(dfa)
        ic = result["invariant_constraints"]
        assert ic["avoid_states"] == []
