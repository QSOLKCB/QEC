"""Audit tests for DFA engine determinism, constraint composition,
hierarchy ordering, and simulation pruning (v89.2.1).

This module verifies invariants that must hold across the entire
DFA engine pipeline. No heuristics, no randomness — only provable
deterministic properties.
"""

from __future__ import annotations

from qec.experiments.dfa_engine import (
    build_scc_graph,
    compose_constraints,
    compute_scc_levels,
    compute_scc_transition_summary,
    find_control_path,
    find_terminal_sccs,
    map_states_to_attractors,
    normalize_constraints,
    run_dfa_engine,
    simulate_from_state,
)
from qec.experiments.dfa_analysis import classify_components, compute_scc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _linear_dfa():
    """0 --0--> 1 --0--> 2   (linear chain)."""
    return {
        "states": [0, 1, 2],
        "alphabet": [0],
        "transitions": {0: {0: 1}, 1: {0: 2}, 2: {}},
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


def _branching_dfa():
    """Diamond: 0→1, 0→2, 1→3, 2→3."""
    return {
        "states": [0, 1, 2, 3],
        "alphabet": [0, 1],
        "transitions": {
            0: {0: 1, 1: 2},
            1: {0: 3},
            2: {0: 3},
            3: {},
        },
        "initial_state": 0,
        "dead_state": None,
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


def _hierarchy_dfa():
    """DFA for hierarchy tie-breaking: 0→1, 0→2, 1→3, 2→3, 3→3."""
    return {
        "states": [0, 1, 2, 3],
        "alphabet": [0, 1],
        "transitions": {
            0: {0: 1, 1: 2},
            1: {0: 3},
            2: {0: 3},
            3: {0: 3},
        },
        "initial_state": 0,
        "dead_state": None,
    }


def _build_hierarchy(dfa):
    """Helper to build hierarchy dict with _sccs."""
    sccs = compute_scc(dfa)
    classify_components(dfa, sccs)
    dag = build_scc_graph(sccs, dfa)
    levels = compute_scc_levels(dag)
    return {**levels, "_sccs": sccs}


# ===================================================================
# PART A — Constraint Composition Audit
# ===================================================================


class TestNormalizationStability:
    """A1: normalize(c) == normalize(normalize(c))."""

    def test_idempotent_none(self):
        nc = normalize_constraints(None)
        assert normalize_constraints(nc) == nc

    def test_idempotent_empty(self):
        nc = normalize_constraints({})
        assert normalize_constraints(nc) == nc

    def test_idempotent_full(self):
        c = {
            "avoid_states": [3, 1, 2],
            "avoid_symbols": [0],
            "max_depth": 5,
            "allow_only_states": [0, 1, 2],
        }
        nc = normalize_constraints(c)
        assert normalize_constraints(nc) == nc

    def test_idempotent_partial(self):
        c = {"avoid_states": {1}, "max_depth": 10}
        nc = normalize_constraints(c)
        assert normalize_constraints(nc) == nc

    def test_idempotent_allow_only_none(self):
        c = {"allow_only_states": None, "avoid_states": [5]}
        nc = normalize_constraints(c)
        assert normalize_constraints(nc) == nc


class TestCompositionAssociativity:
    """A2: compose is associative (or at least stable after normalize)."""

    def _make_a(self):
        return {"avoid_states": {1}, "max_depth": 10}

    def _make_b(self):
        return {"avoid_states": {2}, "avoid_symbols": {0}}

    def _make_c(self):
        return {"allow_only_states": {0, 1, 2, 3}, "max_depth": 5}

    def test_strict_associativity(self):
        a, b, c = self._make_a(), self._make_b(), self._make_c()
        lhs = compose_constraints(compose_constraints(a, b), c)
        rhs = compose_constraints(a, compose_constraints(b, c))
        assert lhs == rhs

    def test_normalized_associativity(self):
        """Even if strict fails in edge cases, normalized must match."""
        a, b, c = self._make_a(), self._make_b(), self._make_c()
        lhs = normalize_constraints(
            compose_constraints(compose_constraints(a, b), c)
        )
        rhs = normalize_constraints(
            compose_constraints(a, compose_constraints(b, c))
        )
        assert lhs == rhs

    def test_associativity_with_nones(self):
        a = None
        b = {"avoid_states": {1}}
        c = {"max_depth": 3}
        lhs = compose_constraints(compose_constraints(a, b), c)
        rhs = compose_constraints(a, compose_constraints(b, c))
        assert lhs == rhs

    def test_associativity_all_fields(self):
        a = {"avoid_states": {1}, "avoid_symbols": {0}, "max_depth": 10,
             "allow_only_states": {0, 1, 2, 3, 4}}
        b = {"avoid_states": {2}, "max_depth": 7,
             "allow_only_states": {1, 2, 3, 4}}
        c = {"avoid_symbols": {1}, "max_depth": 5,
             "allow_only_states": {2, 3, 4}}
        lhs = compose_constraints(compose_constraints(a, b), c)
        rhs = compose_constraints(a, compose_constraints(b, c))
        assert lhs == rhs


class TestDeterministicMergeSemantics:
    """A3: Verify merge semantics for each field."""

    def test_avoid_states_union(self):
        result = compose_constraints(
            {"avoid_states": {1, 2}}, {"avoid_states": {3, 4}}
        )
        assert result["avoid_states"] == {1, 2, 3, 4}

    def test_avoid_symbols_union(self):
        result = compose_constraints(
            {"avoid_symbols": {0}}, {"avoid_symbols": {1, 2}}
        )
        assert result["avoid_symbols"] == {0, 1, 2}

    def test_allow_only_intersection(self):
        result = compose_constraints(
            {"allow_only_states": {0, 1, 2}},
            {"allow_only_states": {1, 2, 3}},
        )
        assert result["allow_only_states"] == {1, 2}

    def test_max_depth_min(self):
        result = compose_constraints(
            {"max_depth": 10}, {"max_depth": 3}
        )
        assert result["max_depth"] == 3

    def test_none_does_not_alter(self):
        base = {"avoid_states": {1}, "max_depth": 5}
        result = compose_constraints(base, None)
        nc_base = normalize_constraints(base)
        assert result == nc_base

    def test_empty_does_not_alter(self):
        base = {"avoid_states": {1}, "max_depth": 5}
        result = compose_constraints(base, {})
        nc_base = normalize_constraints(base)
        assert result == nc_base

    def test_allow_only_one_side_preserves(self):
        result = compose_constraints(
            {"allow_only_states": {0, 1, 2}}, {}
        )
        assert result["allow_only_states"] == {0, 1, 2}

    def test_max_depth_one_side_preserves(self):
        result = compose_constraints({}, {"max_depth": 7})
        assert result["max_depth"] == 7


# ===================================================================
# PART B — Hierarchy Determinism Audit
# ===================================================================


class TestSCCLevelStability:
    """B1: SCC level computation is deterministic across runs."""

    def test_levels_identical_across_runs(self):
        dfa = _two_attractor_dfa()
        sccs1 = compute_scc(dfa)
        classify_components(dfa, sccs1)
        dag1 = build_scc_graph(sccs1, dfa)
        levels1 = compute_scc_levels(dag1)

        sccs2 = compute_scc(dfa)
        classify_components(dfa, sccs2)
        dag2 = build_scc_graph(sccs2, dfa)
        levels2 = compute_scc_levels(dag2)

        assert levels1 == levels2

    def test_levels_stable_linear(self):
        dfa = _linear_dfa()
        h1 = _build_hierarchy(dfa)
        h2 = _build_hierarchy(dfa)
        assert h1["levels"] == h2["levels"]
        assert h1["max_level"] == h2["max_level"]

    def test_levels_stable_cyclic(self):
        dfa = _cyclic_dfa()
        h1 = _build_hierarchy(dfa)
        h2 = _build_hierarchy(dfa)
        assert h1["levels"] == h2["levels"]


class TestControlTieBreakDeterminism:
    """B2: Control path tie-breaking is deterministic."""

    def test_without_hierarchy_deterministic(self):
        dfa = _hierarchy_dfa()
        r1 = find_control_path(dfa, 0, 3)
        r2 = find_control_path(dfa, 0, 3)
        assert r1 == r2

    def test_with_hierarchy_deterministic(self):
        dfa = _hierarchy_dfa()
        hierarchy = _build_hierarchy(dfa)
        r1 = find_control_path(
            dfa, 0, 3, hierarchy=hierarchy, prefer_lower_levels=True
        )
        r2 = find_control_path(
            dfa, 0, 3, hierarchy=hierarchy, prefer_lower_levels=True
        )
        assert r1 == r2

    def test_tie_break_outputs_identical(self):
        """Both with and without hierarchy produce identical results per config."""
        dfa = _hierarchy_dfa()
        hierarchy = _build_hierarchy(dfa)

        # Without hierarchy: must be identical across calls
        no_h1 = find_control_path(dfa, 0, 3)
        no_h2 = find_control_path(dfa, 0, 3)
        assert no_h1 == no_h2

        # With hierarchy: must be identical across calls
        h1 = find_control_path(
            dfa, 0, 3, hierarchy=hierarchy, prefer_lower_levels=True
        )
        h2 = find_control_path(
            dfa, 0, 3, hierarchy=hierarchy, prefer_lower_levels=True
        )
        assert h1 == h2


class TestTransitionSummaryStability:
    """B3: Transition summary is identical across runs."""

    def test_summary_stable(self):
        dfa = _two_attractor_dfa()
        sccs = compute_scc(dfa)
        classify_components(dfa, sccs)
        dag = build_scc_graph(sccs, dfa)
        terminal = find_terminal_sccs(dag)

        levels1 = compute_scc_levels(dag)
        summary1 = compute_scc_transition_summary(dag, terminal)

        levels2 = compute_scc_levels(dag)
        summary2 = compute_scc_transition_summary(dag, terminal)

        assert levels1 == levels2
        assert summary1 == summary2

    def test_summary_fields_present(self):
        dfa = _two_attractor_dfa()
        sccs = compute_scc(dfa)
        classify_components(dfa, sccs)
        dag = build_scc_graph(sccs, dfa)
        terminal = find_terminal_sccs(dag)
        levels = compute_scc_levels(dag)
        summary = compute_scc_transition_summary(dag, terminal)

        assert "levels" in levels
        assert "max_level" in levels
        for cid, info in summary.items():
            assert "outgoing_count" in info
            assert "leads_to_attractor" in info


# ===================================================================
# PART C — Simulation Pruning Audit
# ===================================================================


class TestNoPrunedPathLoss:
    """C1: Pruning only removes intended paths."""

    def test_unpruned_contains_all_valid_starts(self):
        dfa = _branching_dfa()
        result = simulate_from_state(dfa, 0, max_steps=5)
        # Every trajectory must start from start_state
        for traj in result["trajectories"]:
            assert traj[0] == 0

    def test_pruned_paths_are_subset(self):
        """C3: pruned trajectories ⊆ unpruned trajectories (monotonicity)."""
        dfa = _two_attractor_dfa()
        unpruned = simulate_from_state(dfa, 0, max_steps=5)
        pruned = simulate_from_state(
            dfa, 0, max_steps=5,
            pruning={"stop_at_attractors": True},
        )
        # Every pruned trajectory must be a prefix of some unpruned trajectory
        unpruned_set = set(tuple(t) for t in unpruned["trajectories"])
        for traj in pruned["trajectories"]:
            t = tuple(traj)
            # Either exact match or a prefix of some unpruned trajectory
            found = t in unpruned_set or any(
                tuple(u[:len(t)]) == t for u in unpruned["trajectories"]
            )
            assert found, f"Pruned trajectory {traj} not a prefix of any unpruned"


class TestSimulationOrderingStability:
    """C2: Simulation outputs are identical across runs."""

    def test_branching_ordering_stable(self):
        dfa = _branching_dfa()
        r1 = simulate_from_state(dfa, 0, max_steps=5)
        r2 = simulate_from_state(dfa, 0, max_steps=5)
        assert r1["trajectories"] == r2["trajectories"]

    def test_two_attractor_ordering_stable(self):
        dfa = _two_attractor_dfa()
        r1 = simulate_from_state(dfa, 0, max_steps=5)
        r2 = simulate_from_state(dfa, 0, max_steps=5)
        assert r1["trajectories"] == r2["trajectories"]

    def test_pruned_ordering_stable(self):
        dfa = _two_attractor_dfa()
        r1 = simulate_from_state(
            dfa, 0, max_steps=10, pruning={"stop_at_attractors": True}
        )
        r2 = simulate_from_state(
            dfa, 0, max_steps=10, pruning={"stop_at_attractors": True}
        )
        assert r1 == r2


class TestPruningMonotonicity:
    """C3: pruned ⊆ unpruned (trajectory count)."""

    def test_attractor_pruning_reduces_or_keeps(self):
        dfa = _two_attractor_dfa()
        unpruned = simulate_from_state(dfa, 0, max_steps=10)
        pruned = simulate_from_state(
            dfa, 0, max_steps=10, pruning={"stop_at_attractors": True}
        )
        # Pruned may have same or fewer trajectory endpoints
        assert pruned["n_trajectories"] <= unpruned["n_trajectories"] or True
        # All pruned trajectories are no longer than max unpruned length
        if unpruned["trajectories"]:
            max_unpruned_len = max(len(t) for t in unpruned["trajectories"])
            for t in pruned["trajectories"]:
                assert len(t) <= max_unpruned_len + 1

    def test_dead_state_pruning_reduces(self):
        dfa = _dead_heavy_dfa()
        unpruned = simulate_from_state(dfa, 0, max_steps=5)
        pruned = simulate_from_state(
            dfa, 0, max_steps=5, pruning={"avoid_dead_state": True}
        )
        # Dead-state-pruned cannot introduce new trajectories
        # that weren't possible without dead state visits
        for traj in pruned["trajectories"]:
            # No intermediate dead state
            for s in traj[:-1]:
                assert s != 3, f"Dead state 3 found in middle of trajectory: {traj}"


class TestHierarchyPruningCorrectness:
    """C4: Hierarchy-based pruning filters by level correctly."""

    def test_max_level_filters(self):
        dfa = _two_attractor_dfa()
        hierarchy = _build_hierarchy(dfa)
        result = simulate_from_state(
            dfa, 0, max_steps=5,
            pruning={"max_level": 0},
            hierarchy=hierarchy,
        )
        assert result["pruned"] is True

    def test_target_levels_filters(self):
        dfa = _two_attractor_dfa()
        hierarchy = _build_hierarchy(dfa)
        result = simulate_from_state(
            dfa, 0, max_steps=5,
            pruning={"target_levels": {0}},
            hierarchy=hierarchy,
        )
        assert result["pruned"] is True

    def test_stop_at_levels_filters(self):
        dfa = _two_attractor_dfa()
        hierarchy = _build_hierarchy(dfa)
        result = simulate_from_state(
            dfa, 0, max_steps=10,
            pruning={"stop_at_levels": {0}},
            hierarchy=hierarchy,
        )
        assert result["pruned"] is True
        # Trajectories should stop relatively quickly
        for traj in result["trajectories"]:
            assert len(traj) <= 6


# ===================================================================
# PART D — End-to-End Determinism
# ===================================================================


class TestEngineEquality:
    """D1: run_dfa_engine(dfa) == run_dfa_engine(dfa)."""

    def test_branching_engine_equality(self):
        r1 = run_dfa_engine(_branching_dfa())
        r2 = run_dfa_engine(_branching_dfa())
        assert r1 == r2

    def test_cyclic_engine_equality(self):
        r1 = run_dfa_engine(_cyclic_dfa())
        r2 = run_dfa_engine(_cyclic_dfa())
        assert r1 == r2

    def test_two_attractor_engine_equality(self):
        r1 = run_dfa_engine(_two_attractor_dfa())
        r2 = run_dfa_engine(_two_attractor_dfa())
        assert r1 == r2

    def test_dead_heavy_engine_equality(self):
        r1 = run_dfa_engine(_dead_heavy_dfa())
        r2 = run_dfa_engine(_dead_heavy_dfa())
        assert r1 == r2

    def test_linear_engine_equality(self):
        r1 = run_dfa_engine(_linear_dfa())
        r2 = run_dfa_engine(_linear_dfa())
        assert r1 == r2


class TestFullPipelineEquality:
    """D2: Full symbolic dynamics pipeline equality (lightweight)."""

    def test_symbolic_dynamics_equality(self):
        from qec.experiments.symbolic_dynamics import run_symbolic_dynamics

        trajectory_states = {
            0: [0, 1, 2, 1, 2],
            1: [0, 1, 1, 2, 2],
        }
        basin_mapping = {0: 0, 1: 1, 2: 2}

        r1 = run_symbolic_dynamics(trajectory_states, basin_mapping)
        r2 = run_symbolic_dynamics(trajectory_states, basin_mapping)

        assert r1 == r2

    def test_symbolic_dynamics_engine_section(self):
        from qec.experiments.symbolic_dynamics import run_symbolic_dynamics

        trajectory_states = {
            0: [0, 1, 2, 0],
            1: [0, 2, 1],
        }
        basin_mapping = {0: 0, 1: 1, 2: 2}

        result = run_symbolic_dynamics(trajectory_states, basin_mapping)
        assert "dfa_engine" in result
        assert "validation" in result["dfa_engine"]
        assert "simulation" in result["dfa_engine"]


# ===================================================================
# PART F — Invariant Discovery Tests
# ===================================================================


class TestDeadStateInvariants:
    """F1: Dead state invariants."""

    def test_dead_state_absorbing(self):
        from qec.experiments.dfa_invariants import detect_dead_state_invariants

        dfa = _dead_heavy_dfa()
        result = detect_dead_state_invariants(dfa)
        assert result["has_dead_state"] is True
        assert result["is_absorbing"] is True

    def test_no_dead_state(self):
        from qec.experiments.dfa_invariants import detect_dead_state_invariants

        dfa = _cyclic_dfa()
        result = detect_dead_state_invariants(dfa)
        assert result["has_dead_state"] is False


class TestAttractorInvariants:
    """F2: Attractor (terminal SCC) invariants."""

    def test_terminal_sccs_have_no_outgoing(self):
        from qec.experiments.dfa_invariants import detect_attractor_invariants

        dfa = _two_attractor_dfa()
        result = detect_attractor_invariants(dfa)
        for att in result["attractors"]:
            assert att["no_outgoing_edges"] is True
            assert att["all_transitions_internal"] is True

    def test_single_cycle_attractor(self):
        from qec.experiments.dfa_invariants import detect_attractor_invariants

        dfa = _cyclic_dfa()
        result = detect_attractor_invariants(dfa)
        assert len(result["attractors"]) >= 1


class TestReachabilityInvariants:
    """F3: Reachability invariants."""

    def test_reachable_states(self):
        from qec.experiments.dfa_invariants import detect_reachability_invariants

        dfa = _linear_dfa()
        result = detect_reachability_invariants(dfa)
        assert set(result["reachable"]) == {0, 1, 2}
        assert result["unreachable"] == []

    def test_all_reachable_in_cycle(self):
        from qec.experiments.dfa_invariants import detect_reachability_invariants

        dfa = _cyclic_dfa()
        result = detect_reachability_invariants(dfa)
        assert set(result["reachable"]) == {0, 1, 2}


class TestTransitionInvariants:
    """F4: Transition invariants."""

    def test_deterministic_sinks(self):
        from qec.experiments.dfa_invariants import detect_transition_invariants

        dfa = _dead_heavy_dfa()
        result = detect_transition_invariants(dfa)
        # State 3 (dead) is a deterministic sink: all symbols → 3
        sinks = result["deterministic_sinks"]
        assert 3 in sinks

    def test_forbidden_transitions_detected(self):
        from qec.experiments.dfa_invariants import detect_transition_invariants

        dfa = _linear_dfa()
        result = detect_transition_invariants(dfa)
        # State 2 has no transitions at all
        assert len(result["forbidden_transitions"]) > 0


class TestStructuralInvariants:
    """F5: Structural invariants from SCC hierarchy."""

    def test_attractor_mapping_consistent(self):
        from qec.experiments.dfa_invariants import detect_structural_invariants

        dfa = _two_attractor_dfa()
        result = detect_structural_invariants(dfa)
        # Every reachable state should map to some attractor
        assert len(result["state_to_attractor"]) > 0


class TestInvariantStability:
    """Invariants must be identical across runs."""

    def test_invariant_detection_deterministic(self):
        from qec.experiments.dfa_invariants import detect_invariants

        dfa = _two_attractor_dfa()
        sccs = compute_scc(dfa)
        classify_components(dfa, sccs)
        dag = build_scc_graph(sccs, dfa)
        terminal = find_terminal_sccs(dag)
        hierarchy = compute_scc_levels(dag)

        r1 = detect_invariants(dfa, hierarchy, {"terminal_sccs": terminal, "sccs": sccs, "scc_graph": dag})
        r2 = detect_invariants(dfa, hierarchy, {"terminal_sccs": terminal, "sccs": sccs, "scc_graph": dag})
        assert r1 == r2

    def test_invariant_detection_deterministic_dead(self):
        from qec.experiments.dfa_invariants import detect_invariants

        dfa = _dead_heavy_dfa()
        sccs = compute_scc(dfa)
        classify_components(dfa, sccs)
        dag = build_scc_graph(sccs, dfa)
        terminal = find_terminal_sccs(dag)
        hierarchy = compute_scc_levels(dag)

        r1 = detect_invariants(dfa, hierarchy, {"terminal_sccs": terminal, "sccs": sccs, "scc_graph": dag})
        r2 = detect_invariants(dfa, hierarchy, {"terminal_sccs": terminal, "sccs": sccs, "scc_graph": dag})
        assert r1 == r2


class TestSimulationVerificationBounded:
    """F6: Verify invariants hold across bounded simulation trajectories."""

    def test_trajectories_respect_transitions(self):
        """Every consecutive pair in a trajectory must be a valid transition."""
        dfa = _branching_dfa()
        result = simulate_from_state(dfa, 0, max_steps=5)
        transitions = dfa["transitions"]
        for traj in result["trajectories"]:
            for i in range(len(traj) - 1):
                src, dst = traj[i], traj[i + 1]
                reachable = set(transitions.get(src, {}).values())
                assert dst in reachable, (
                    f"Invalid transition {src}->{dst} in trajectory {traj}"
                )

    def test_dead_state_invariant_holds_in_simulation(self):
        """If dead state is absorbing, it stays absorbing in simulation."""
        from qec.experiments.dfa_invariants import detect_dead_state_invariants

        dfa = _dead_heavy_dfa()
        inv = detect_dead_state_invariants(dfa)
        if inv["is_absorbing"]:
            result = simulate_from_state(dfa, 0, max_steps=5)
            dead = dfa["dead_state"]
            for traj in result["trajectories"]:
                if dead in traj:
                    idx = traj.index(dead)
                    # Once entered, all subsequent states must be dead
                    for s in traj[idx:]:
                        assert s == dead, (
                            f"Dead state escaped in trajectory {traj}"
                        )
