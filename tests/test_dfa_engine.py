"""Tests for the deterministic DFA engine (v89.1.0).

Covers: validation, simulation, control, SCC meta-graph,
attractor mapping, constrained control, simulation pruning,
attractor hierarchy, determinism, and edge cases.
"""

from __future__ import annotations

from qec.experiments.dfa_engine import (
    build_scc_graph,
    compute_scc_levels,
    compute_scc_transition_summary,
    find_control_path,
    find_terminal_sccs,
    get_next_states,
    map_states_to_attractors,
    run_dfa_engine,
    simulate_from_state,
    step,
    validate_sequence,
)
from qec.experiments.dfa_analysis import classify_components, compute_scc


# ---------------------------------------------------------------------------
# Fixtures — small deterministic DFAs
# ---------------------------------------------------------------------------


def _linear_dfa():
    """0 --0--> 1 --0--> 2   (linear chain, alphabet={0})."""
    return {
        "states": [0, 1, 2],
        "alphabet": [0],
        "transitions": {0: {0: 1}, 1: {0: 2}, 2: {}},
        "initial_state": 0,
        "dead_state": None,
    }


def _cyclic_dfa():
    """0 --0--> 1 --0--> 2 --0--> 0   (3-cycle, alphabet={0})."""
    return {
        "states": [0, 1, 2],
        "alphabet": [0],
        "transitions": {0: {0: 1}, 1: {0: 2}, 2: {0: 0}},
        "initial_state": 0,
        "dead_state": None,
    }


def _branching_dfa():
    """0 --0--> 1, 0 --1--> 2, 1 --0--> 3, 2 --0--> 3   (diamond)."""
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


def _single_state_dfa():
    """Single state with self-loop on symbol 0."""
    return {
        "states": [0],
        "alphabet": [0],
        "transitions": {0: {0: 0}},
        "initial_state": 0,
        "dead_state": None,
    }


def _dead_heavy_dfa():
    """0 --0--> 1, 0 --1--> 3(dead), 1 --0--> 2, 1 --1--> 3, 2 --0--> 3, 2 --1--> 3."""
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
    """Two terminal SCCs: cycle {1,2} and fixed-point {4}.
    0 --0--> 1, 0 --1--> 3
    1 --0--> 2
    2 --0--> 1     (cycle attractor)
    3 --0--> 4
    4 --0--> 4     (fixed-point attractor)
    """
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


# ===================================================================
# PART A — Traversal Primitives
# ===================================================================


class TestGetNextStates:
    def test_linear(self):
        assert get_next_states(_linear_dfa(), 0) == [1]
        assert get_next_states(_linear_dfa(), 2) == []

    def test_branching(self):
        assert get_next_states(_branching_dfa(), 0) == [1, 2]

    def test_self_loop(self):
        assert get_next_states(_single_state_dfa(), 0) == [0]


class TestStep:
    def test_valid(self):
        assert step(_linear_dfa(), 0, 0) == 1

    def test_invalid_symbol(self):
        assert step(_linear_dfa(), 2, 0) is None

    def test_invalid_state(self):
        assert step(_linear_dfa(), 99, 0) is None


# ===================================================================
# PART B — Validation
# ===================================================================


class TestValidateSequence:
    def test_valid_sequence(self):
        result = validate_sequence(_linear_dfa(), [0, 1, 2])
        assert result["valid"] is True
        assert result["failure_index"] is None

    def test_invalid_transition(self):
        result = validate_sequence(_linear_dfa(), [0, 2])
        assert result["valid"] is False
        assert result["failure_index"] == 0

    def test_single_state(self):
        result = validate_sequence(_linear_dfa(), [0])
        assert result["valid"] is True

    def test_empty_sequence(self):
        result = validate_sequence(_linear_dfa(), [])
        assert result["valid"] is True

    def test_cyclic_valid(self):
        result = validate_sequence(_cyclic_dfa(), [0, 1, 2, 0])
        assert result["valid"] is True

    def test_failure_index_mid_sequence(self):
        result = validate_sequence(_linear_dfa(), [0, 1, 0])
        assert result["valid"] is False
        assert result["failure_index"] == 1


# ===================================================================
# PART C — Simulation
# ===================================================================


class TestSimulation:
    def test_linear_bounded(self):
        result = simulate_from_state(_linear_dfa(), 0, max_steps=5)
        assert result["n_trajectories"] >= 1
        # Only one trajectory: 0 → 1 → 2
        assert [0, 1, 2] in result["trajectories"]

    def test_cyclic_stops(self):
        result = simulate_from_state(_cyclic_dfa(), 0, max_steps=10)
        # Must detect cycle and stop
        for traj in result["trajectories"]:
            # No trajectory should be longer than n_states + 1 (cycle marker)
            assert len(traj) <= 4  # [0, 1, 2, 0] at most

    def test_no_duplicates(self):
        result = simulate_from_state(_branching_dfa(), 0, max_steps=5)
        seen = set()
        for traj in result["trajectories"]:
            key = tuple(traj)
            assert key not in seen, f"Duplicate trajectory: {traj}"
            seen.add(key)

    def test_deterministic_ordering(self):
        r1 = simulate_from_state(_branching_dfa(), 0, max_steps=5)
        r2 = simulate_from_state(_branching_dfa(), 0, max_steps=5)
        assert r1 == r2

    def test_single_state_self_loop(self):
        result = simulate_from_state(_single_state_dfa(), 0, max_steps=3)
        assert result["n_trajectories"] >= 1
        # Should contain the self-loop trajectory [0, 0]
        assert [0, 0] in result["trajectories"]


# ===================================================================
# PART D — Control
# ===================================================================


class TestControl:
    def test_reachable_path(self):
        result = find_control_path(_linear_dfa(), 0, 2)
        assert result["reachable"] is True
        assert result["path"] == [0, 1, 2]
        assert result["length"] == 2

    def test_unreachable_target(self):
        result = find_control_path(_linear_dfa(), 2, 0)
        assert result["reachable"] is False
        assert result["path"] == []
        assert result["length"] == 0

    def test_shortest_path(self):
        result = find_control_path(_branching_dfa(), 0, 3)
        assert result["reachable"] is True
        assert result["length"] == 2
        # Either [0,1,3] or [0,2,3] — both length 2.
        assert len(result["path"]) == 3

    def test_start_is_target(self):
        result = find_control_path(_linear_dfa(), 0, 0)
        assert result["reachable"] is True
        assert result["path"] == [0]
        assert result["length"] == 0

    def test_multi_target(self):
        result = find_control_path(
            _branching_dfa(), 0, target_state=-1, target_states={2, 3}
        )
        assert result["reachable"] is True
        # Should reach 2 first (closer).
        assert result["path"][-1] in {2, 3}
        assert result["length"] <= 2

    def test_symbols_correct(self):
        result = find_control_path(_linear_dfa(), 0, 2)
        assert result["symbols"] == [0, 0]


# ===================================================================
# PART E — SCC Meta-Graph / Attractors
# ===================================================================


class TestSCCGraph:
    def test_dag_structure(self):
        dfa = _two_attractor_dfa()
        sccs = compute_scc(dfa)
        classify_components(dfa, sccs)
        dag = build_scc_graph(sccs, dfa)

        # DAG should be a dict of int → list[int].
        assert isinstance(dag, dict)
        for cid, targets in dag.items():
            assert isinstance(targets, list)
            # No self-edges in condensation.
            assert cid not in targets

    def test_terminal_sccs(self):
        dfa = _two_attractor_dfa()
        sccs = compute_scc(dfa)
        classify_components(dfa, sccs)
        dag = build_scc_graph(sccs, dfa)
        terminals = find_terminal_sccs(dag)

        # Two attractors: the {1,2} cycle and the {4} fixed point.
        assert len(terminals) >= 2

    def test_single_state_terminal(self):
        dfa = _single_state_dfa()
        sccs = compute_scc(dfa)
        classify_components(dfa, sccs)
        dag = build_scc_graph(sccs, dfa)
        terminals = find_terminal_sccs(dag)
        assert len(terminals) == 1

    def test_linear_terminal(self):
        dfa = _linear_dfa()
        sccs = compute_scc(dfa)
        classify_components(dfa, sccs)
        dag = build_scc_graph(sccs, dfa)
        terminals = find_terminal_sccs(dag)
        # State 2 is terminal (no outgoing).
        assert len(terminals) >= 1


# ===================================================================
# PART E (continued) — Basin Mapping
# ===================================================================


class TestBasinMapping:
    def test_two_attractor_basins(self):
        dfa = _two_attractor_dfa()
        sccs = compute_scc(dfa)
        classify_components(dfa, sccs)
        dag = build_scc_graph(sccs, dfa)
        terminals = find_terminal_sccs(dag)
        result = map_states_to_attractors(dfa, sccs, dag, terminals)

        assert "attractors" in result
        assert "basins" in result

        # Every state should map to some attractor.
        for s in dfa["states"]:
            assert s in result["basins"]

        # States 1 and 2 should map to the same attractor.
        assert result["basins"][1] == result["basins"][2]

        # States 3 and 4 should map to the same attractor.
        assert result["basins"][3] == result["basins"][4]

    def test_single_attractor(self):
        dfa = _cyclic_dfa()
        sccs = compute_scc(dfa)
        classify_components(dfa, sccs)
        dag = build_scc_graph(sccs, dfa)
        terminals = find_terminal_sccs(dag)
        result = map_states_to_attractors(dfa, sccs, dag, terminals)

        # All states in one cycle → one attractor.
        attractor_ids = set(result["basins"].values())
        assert len(attractor_ids) == 1


# ===================================================================
# PART F — Engine Wrapper
# ===================================================================


class TestRunDFAEngine:
    def test_returns_all_keys(self):
        result = run_dfa_engine(_branching_dfa())
        assert "validation" in result
        assert "simulation" in result
        assert "control" in result
        assert "attractors" in result

    def test_determinism(self):
        r1 = run_dfa_engine(_branching_dfa())
        r2 = run_dfa_engine(_branching_dfa())
        assert r1 == r2

    def test_cyclic_dfa(self):
        result = run_dfa_engine(_cyclic_dfa())
        assert "attractors" in result
        assert len(result["attractors"]["terminal_sccs"]) >= 1

    def test_single_state(self):
        result = run_dfa_engine(_single_state_dfa())
        assert result["attractors"]["terminal_sccs"]

    def test_dead_heavy(self):
        result = run_dfa_engine(_dead_heavy_dfa())
        assert "control" in result
        assert "attractors" in result


# ===================================================================
# Determinism — same DFA → identical outputs
# ===================================================================


class TestDeterminism:
    def test_validate_determinism(self):
        seq = [0, 1, 2]
        r1 = validate_sequence(_linear_dfa(), seq)
        r2 = validate_sequence(_linear_dfa(), seq)
        assert r1 == r2

    def test_simulate_determinism(self):
        r1 = simulate_from_state(_branching_dfa(), 0, 5)
        r2 = simulate_from_state(_branching_dfa(), 0, 5)
        assert r1 == r2

    def test_control_determinism(self):
        r1 = find_control_path(_branching_dfa(), 0, 3)
        r2 = find_control_path(_branching_dfa(), 0, 3)
        assert r1 == r2

    def test_scc_graph_determinism(self):
        dfa = _two_attractor_dfa()
        sccs1 = compute_scc(dfa)
        classify_components(dfa, sccs1)
        dag1 = build_scc_graph(sccs1, dfa)

        sccs2 = compute_scc(dfa)
        classify_components(dfa, sccs2)
        dag2 = build_scc_graph(sccs2, dfa)

        assert dag1 == dag2


# ===================================================================
# v89.1 — Constrained Control
# ===================================================================


class TestConstrainedControl:
    def test_avoid_states_blocks_path(self):
        # Linear: 0->1->2. Avoiding state 1 should block the path.
        result = find_control_path(
            _linear_dfa(), 0, 2, constraints={"avoid_states": {1}}
        )
        assert result["reachable"] is False
        assert result["constrained"] is True

    def test_avoid_symbols_blocks_path(self):
        # Linear uses symbol 0 for all transitions.
        result = find_control_path(
            _linear_dfa(), 0, 2, constraints={"avoid_symbols": {0}}
        )
        assert result["reachable"] is False
        assert result["constrained"] is True

    def test_max_depth_limits_search(self):
        # Path 0->1->2 has length 2; max_depth=1 should block it.
        result = find_control_path(
            _linear_dfa(), 0, 2, constraints={"max_depth": 1}
        )
        assert result["reachable"] is False
        assert result["constrained"] is True

    def test_max_depth_allows_short_path(self):
        result = find_control_path(
            _linear_dfa(), 0, 1, constraints={"max_depth": 2}
        )
        assert result["reachable"] is True
        assert result["length"] == 1

    def test_no_constraints_backward_compatible(self):
        r1 = find_control_path(_linear_dfa(), 0, 2)
        assert r1["reachable"] is True
        assert r1["constrained"] is False

    def test_empty_constraints(self):
        result = find_control_path(
            _linear_dfa(), 0, 2, constraints={}
        )
        assert result["reachable"] is True
        assert result["constrained"] is False


# ===================================================================
# v89.1 — Simulation Pruning
# ===================================================================


class TestSimulationPruning:
    def test_stop_at_attractors(self):
        dfa = _two_attractor_dfa()
        result = simulate_from_state(
            dfa, 0, max_steps=10,
            pruning={"stop_at_attractors": True},
        )
        assert result["pruned"] is True
        # Trajectories should not endlessly traverse attractor cycles.
        for traj in result["trajectories"]:
            assert len(traj) <= 5

    def test_avoid_dead_state(self):
        dfa = _dead_heavy_dfa()
        result = simulate_from_state(
            dfa, 0, max_steps=5,
            pruning={"avoid_dead_state": True},
        )
        assert result["pruned"] is True
        # Dead state (3) should not appear as an expanded node.
        for traj in result["trajectories"]:
            # Dead state may appear as terminal but not in the middle.
            if 3 in traj:
                assert traj[-1] == 3 or traj.index(3) == len(traj) - 1

    def test_no_pruning_backward_compatible(self):
        r1 = simulate_from_state(_branching_dfa(), 0, 5)
        assert r1["pruned"] is False

    def test_pruned_simulation_deterministic(self):
        dfa = _two_attractor_dfa()
        r1 = simulate_from_state(dfa, 0, 10, pruning={"stop_at_attractors": True})
        r2 = simulate_from_state(dfa, 0, 10, pruning={"stop_at_attractors": True})
        assert r1 == r2


# ===================================================================
# v89.1 — Attractor Hierarchy
# ===================================================================


class TestAttractorHierarchy:
    def test_terminal_sccs_level_zero(self):
        dfa = _two_attractor_dfa()
        sccs = compute_scc(dfa)
        classify_components(dfa, sccs)
        dag = build_scc_graph(sccs, dfa)
        terminal = find_terminal_sccs(dag)
        result = compute_scc_levels(dag)

        for t in terminal:
            assert result["levels"][t] == 0

    def test_simple_dag_levels(self):
        dfa = _linear_dfa()
        sccs = compute_scc(dfa)
        classify_components(dfa, sccs)
        dag = build_scc_graph(sccs, dfa)
        result = compute_scc_levels(dag)

        assert result["max_level"] >= 1

    def test_single_state_level(self):
        dfa = _single_state_dfa()
        sccs = compute_scc(dfa)
        classify_components(dfa, sccs)
        dag = build_scc_graph(sccs, dfa)
        result = compute_scc_levels(dag)

        assert result["max_level"] == 0

    def test_transition_summary(self):
        dfa = _two_attractor_dfa()
        sccs = compute_scc(dfa)
        classify_components(dfa, sccs)
        dag = build_scc_graph(sccs, dfa)
        terminal = find_terminal_sccs(dag)
        summary = compute_scc_transition_summary(dag, terminal)

        for t in terminal:
            assert summary[t]["outgoing_count"] == 0
            assert summary[t]["leads_to_attractor"] is True

    def test_engine_includes_hierarchy(self):
        result = run_dfa_engine(_two_attractor_dfa())
        assert "attractor_hierarchy" in result
        assert "levels" in result["attractor_hierarchy"]
        assert "max_level" in result["attractor_hierarchy"]
        assert "transition_summary" in result["attractor_hierarchy"]


# ===================================================================
# v89.1 — Edge Cases
# ===================================================================


class TestEdgeCases:
    def test_single_state_constrained_control(self):
        result = find_control_path(
            _single_state_dfa(), 0, 0, constraints={"max_depth": 0}
        )
        assert result["reachable"] is True
        assert result["length"] == 0

    def test_dead_state_only_transitions(self):
        dfa = _dead_heavy_dfa()
        result = simulate_from_state(
            dfa, 2, max_steps=3,
            pruning={"avoid_dead_state": True},
        )
        # State 2 transitions only to dead state 3
        assert result["n_trajectories"] >= 1
