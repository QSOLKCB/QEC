"""Tests for DFA structural analysis (v88.3.0).

Covers: determinism, SCC correctness, branching, entropy, dead states, edge cases.
"""

import math

from qec.experiments.dfa_analysis import (
    analyze_dfa,
    build_adjacency,
    classify_components,
    compute_branching,
    compute_complexity,
    compute_scc,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dfa(states, alphabet, transitions, initial_state=0, dead_state=None):
    """Build a minimal DFA dict for testing."""
    dfa = {
        "states": list(states),
        "alphabet": list(alphabet),
        "transitions": transitions,
        "initial_state": initial_state,
    }
    if dead_state is not None:
        dfa["dead_state"] = dead_state
    return dfa


def _linear_dfa():
    """0 -a-> 1 -a-> 2 (linear chain, no cycles)."""
    return _make_dfa(
        states=[0, 1, 2],
        alphabet=[0],
        transitions={0: {0: 1}, 1: {0: 2}, 2: {}},
    )


def _cycle_dfa():
    """0 -a-> 1 -a-> 2 -a-> 0 (single 3-state cycle)."""
    return _make_dfa(
        states=[0, 1, 2],
        alphabet=[0],
        transitions={0: {0: 1}, 1: {0: 2}, 2: {0: 0}},
    )


def _branching_dfa():
    """0 -a-> 1, 0 -b-> 2 (state 0 branches to two targets)."""
    return _make_dfa(
        states=[0, 1, 2],
        alphabet=[0, 1],
        transitions={0: {0: 1, 1: 2}, 1: {}, 2: {}},
    )


def _dead_state_dfa():
    """0 -a-> 1, 0 -b-> 3(dead), 1 -a-> 1, 1 -b-> 3(dead), 3 -a-> 3, 3 -b-> 3."""
    return _make_dfa(
        states=[0, 1, 3],
        alphabet=[0, 1],
        transitions={
            0: {0: 1, 1: 3},
            1: {0: 1, 1: 3},
            3: {0: 3, 1: 3},
        },
        dead_state=3,
    )


def _self_loop_dfa():
    """Single state with self-loop."""
    return _make_dfa(
        states=[0],
        alphabet=[0],
        transitions={0: {0: 0}},
    )


def _empty_dfa():
    """No states."""
    return _make_dfa(states=[], alphabet=[], transitions={})


def _disconnected_dfa():
    """Two components: 0->1 and 2->3->2 (cycle)."""
    return _make_dfa(
        states=[0, 1, 2, 3],
        alphabet=[0],
        transitions={0: {0: 1}, 1: {}, 2: {0: 3}, 3: {0: 2}},
    )


# ---------------------------------------------------------------------------
# 1. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_dfa_identical_analysis(self):
        """Same DFA must produce byte-identical analysis across calls."""
        dfa = _cycle_dfa()
        r1 = analyze_dfa(dfa)
        r2 = analyze_dfa(dfa)
        assert r1 == r2

    def test_determinism_branching(self):
        dfa = _branching_dfa()
        r1 = analyze_dfa(dfa)
        r2 = analyze_dfa(dfa)
        assert r1 == r2

    def test_determinism_dead_state(self):
        dfa = _dead_state_dfa()
        r1 = analyze_dfa(dfa)
        r2 = analyze_dfa(dfa)
        assert r1 == r2


# ---------------------------------------------------------------------------
# 2. SCC Correctness
# ---------------------------------------------------------------------------

class TestSCC:
    def test_simple_cycle(self):
        dfa = _cycle_dfa()
        sccs = compute_scc(dfa)
        # All three states in one SCC
        all_states = set()
        for scc in sccs:
            all_states.update(scc["states"])
        assert all_states == {0, 1, 2}
        # Should have exactly one SCC containing all states
        cycle_sccs = [s for s in sccs if len(s["states"]) == 3]
        assert len(cycle_sccs) == 1

    def test_linear_no_cycle(self):
        dfa = _linear_dfa()
        sccs = compute_scc(dfa)
        # All SCCs should be singletons (no cycles)
        for scc in sccs:
            assert len(scc["states"]) == 1

    def test_disconnected_graph(self):
        dfa = _disconnected_dfa()
        sccs = compute_scc(dfa)
        # Should have: {0}, {1}, {2,3}
        sizes = sorted(len(s["states"]) for s in sccs)
        assert sizes == [1, 1, 2]
        # The cycle component should contain 2 and 3
        cycle_scc = [s for s in sccs if len(s["states"]) == 2][0]
        assert set(cycle_scc["states"]) == {2, 3}

    def test_scc_ids_sequential(self):
        dfa = _disconnected_dfa()
        sccs = compute_scc(dfa)
        ids = [s["id"] for s in sccs]
        assert ids == list(range(len(sccs)))

    def test_classify_cycle(self):
        dfa = _cycle_dfa()
        sccs = compute_scc(dfa)
        result = classify_components(dfa, sccs)
        assert result["n_cycles"] >= 1
        assert result["largest_cycle"] == 3

    def test_classify_transient(self):
        dfa = _linear_dfa()
        sccs = compute_scc(dfa)
        result = classify_components(dfa, sccs)
        assert result["n_cycles"] == 0
        transients = [s for s in result["components"] if s["type"] == "transient"]
        assert len(transients) == 3

    def test_classify_fixed_point(self):
        dfa = _self_loop_dfa()
        sccs = compute_scc(dfa)
        result = classify_components(dfa, sccs)
        fixed = [s for s in result["components"] if s["type"] == "fixed_point"]
        assert len(fixed) == 1

    def test_cycle_fraction(self):
        dfa = _cycle_dfa()
        sccs = compute_scc(dfa)
        result = classify_components(dfa, sccs)
        assert result["cycle_fraction"] == 1.0

        dfa2 = _linear_dfa()
        sccs2 = compute_scc(dfa2)
        result2 = classify_components(dfa2, sccs2)
        assert result2["cycle_fraction"] == 0.0


# ---------------------------------------------------------------------------
# 3. Branching Correctness
# ---------------------------------------------------------------------------

class TestBranching:
    def test_known_branching(self):
        dfa = _branching_dfa()
        b = compute_branching(dfa)
        # State 0 has 2 unique targets, states 1 and 2 have 0
        assert b["max_branching"] == 2
        assert b["min_branching"] == 0

    def test_uniform_branching(self):
        dfa = _cycle_dfa()
        b = compute_branching(dfa)
        # Each state has exactly 1 target
        assert b["mean_branching"] == 1.0
        assert b["max_branching"] == 1
        assert b["min_branching"] == 1

    def test_density_complete(self):
        # All transitions present and distinct
        dfa = _make_dfa(
            states=[0, 1],
            alphabet=[0, 1],
            transitions={0: {0: 0, 1: 1}, 1: {0: 0, 1: 1}},
        )
        b = compute_branching(dfa)
        assert b["density"] == 1.0

    def test_density_sparse(self):
        dfa = _linear_dfa()
        b = compute_branching(dfa)
        # 2 transitions out of 3 possible (3 states * 1 symbol)
        assert abs(b["density"] - 2.0 / 3.0) < 1e-10


# ---------------------------------------------------------------------------
# 4. Entropy Sanity
# ---------------------------------------------------------------------------

class TestEntropy:
    def test_entropy_increases_with_branching(self):
        # Single target: entropy = log(1) = 0
        dfa1 = _cycle_dfa()
        c1 = compute_complexity(dfa1)

        # Two targets for state 0: entropy = log(2)
        dfa2 = _branching_dfa()
        c2 = compute_complexity(dfa2)

        assert c2["branching_entropy"] > c1["branching_entropy"]

    def test_entropy_zero_no_branching(self):
        dfa = _cycle_dfa()
        c = compute_complexity(dfa)
        assert c["branching_entropy"] == 0.0

    def test_entropy_log2_for_binary_branching(self):
        # State with 2 unique targets: entropy = log(2)
        dfa = _make_dfa(
            states=[0],
            alphabet=[0, 1],
            transitions={0: {0: 0, 1: 1}},  # need state 1 but it's not in states
        )
        # Adjust: add state 1
        dfa = _make_dfa(
            states=[0, 1],
            alphabet=[0, 1],
            transitions={0: {0: 0, 1: 1}, 1: {0: 0, 1: 1}},
        )
        c = compute_complexity(dfa)
        assert abs(c["branching_entropy"] - math.log(2)) < 1e-10


# ---------------------------------------------------------------------------
# 5. Dead State Tracking
# ---------------------------------------------------------------------------

class TestDeadState:
    def test_dead_state_detected(self):
        dfa = _dead_state_dfa()
        c = compute_complexity(dfa)
        assert c["dead_state"]["has_dead_state"] is True
        assert c["dead_state"]["transitions_to_dead"] > 0

    def test_dead_proportion(self):
        dfa = _dead_state_dfa()
        c = compute_complexity(dfa)
        # 4 transitions to dead out of 6 total:
        # 0-b->3, 1-b->3, 3-a->3, 3-b->3
        assert abs(c["dead_state"]["dead_proportion"] - 4.0 / 6.0) < 1e-10

    def test_no_dead_state(self):
        dfa = _cycle_dfa()
        c = compute_complexity(dfa)
        assert c["dead_state"]["has_dead_state"] is False
        assert c["dead_state"]["transitions_to_dead"] == 0


# ---------------------------------------------------------------------------
# 6. Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_state_self_loop(self):
        dfa = _self_loop_dfa()
        result = analyze_dfa(dfa)
        assert result["branching"]["mean_branching"] == 1.0
        assert result["cycles"]["n_scc"] == 1

    def test_empty_dfa(self):
        dfa = _empty_dfa()
        result = analyze_dfa(dfa)
        assert result["branching"]["mean_branching"] == 0.0
        assert result["cycles"]["n_scc"] == 0
        assert result["complexity"]["branching_entropy"] == 0.0

    def test_no_mutation_of_input(self):
        """analyze_dfa must not modify the input DFA."""
        dfa = _cycle_dfa()
        import copy
        original = copy.deepcopy(dfa)
        analyze_dfa(dfa)
        assert dfa == original


# ---------------------------------------------------------------------------
# 7. Adjacency
# ---------------------------------------------------------------------------

class TestAdjacency:
    def test_adjacency_sorted(self):
        dfa = _branching_dfa()
        adj = build_adjacency(dfa)
        assert adj[0] == [1, 2]
        assert adj[1] == []
        assert adj[2] == []

    def test_adjacency_self_loop(self):
        dfa = _self_loop_dfa()
        adj = build_adjacency(dfa)
        assert adj[0] == [0]


# ---------------------------------------------------------------------------
# 8. Path Diversity
# ---------------------------------------------------------------------------

class TestPathDiversity:
    def test_path_diversity_cycle(self):
        dfa = _cycle_dfa()
        c = compute_complexity(dfa)
        # 3 unique transitions / 3 states = 1.0
        assert c["path_diversity"] == 1.0

    def test_path_diversity_branching(self):
        dfa = _branching_dfa()
        c = compute_complexity(dfa)
        # 2 unique transitions / 3 states
        assert abs(c["path_diversity"] - 2.0 / 3.0) < 1e-10
