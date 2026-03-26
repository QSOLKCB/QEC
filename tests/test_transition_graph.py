"""Tests for transition graph analysis (v102.5.0).

Verifies:
- correct edge counts
- deterministic ordering
- node stats correctness
- ranking correctness
- pattern detection
- edge cases (empty input, single-type sequences, self-loops)
- integration via run_transition_graph_analysis
"""

from __future__ import annotations

import copy

from qec.analysis.transition_graph import (
    build_transition_graph,
    compute_node_stats,
    detect_transition_patterns,
    extract_dominant_flows,
    rank_transitions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run(strategies):
    """Build a run dict from a list of (name, design_score) tuples."""
    return {
        "strategies": [
            {
                "name": name,
                "metrics": {
                    "design_score": score,
                    "confidence_efficiency": 0.5,
                    "consistency_gap": 0.1,
                    "revival_strength": 0.0,
                },
            }
            for name, score in strategies
        ],
    }


def _make_runs_with_scores(name, scores):
    """Build a list of runs where a single strategy has varying scores."""
    return [_make_run([(name, s)]) for s in scores]


# ---------------------------------------------------------------------------
# build_transition_graph tests
# ---------------------------------------------------------------------------


class TestBuildTransitionGraph:
    """Tests for build_transition_graph."""

    def test_empty_trajectories(self):
        graph = build_transition_graph({})
        assert graph == {}

    def test_single_entry_no_edges(self):
        graph = build_transition_graph({"A": ["stable_core"]})
        assert graph == {}

    def test_single_transition(self):
        graph = build_transition_graph(
            {"A": ["stable_core", "steady_improver"]}
        )
        assert graph == {("stable_core", "steady_improver"): 1}

    def test_multiple_transitions_same_strategy(self):
        graph = build_transition_graph(
            {"A": ["a", "b", "a"]}
        )
        assert graph == {("a", "b"): 1, ("b", "a"): 1}

    def test_self_loop(self):
        graph = build_transition_graph(
            {"A": ["a", "a", "a"]}
        )
        assert graph == {("a", "a"): 2}

    def test_aggregates_across_strategies(self):
        graph = build_transition_graph({
            "A": ["a", "b"],
            "B": ["a", "b"],
        })
        assert graph == {("a", "b"): 2}

    def test_multiple_edges(self):
        graph = build_transition_graph({
            "A": ["a", "b", "c"],
            "B": ["b", "c"],
        })
        assert graph[("a", "b")] == 1
        assert graph[("b", "c")] == 2

    def test_deterministic_iteration(self):
        traj = {"B": ["a", "b"], "A": ["a", "b"]}
        g1 = build_transition_graph(traj)
        g2 = build_transition_graph(copy.deepcopy(traj))
        assert g1 == g2

    def test_no_mutation(self):
        traj = {"A": ["a", "b", "c"]}
        original = copy.deepcopy(traj)
        build_transition_graph(traj)
        assert traj == original

    def test_skip_short_sequences(self):
        graph = build_transition_graph({
            "A": [],
            "B": ["x"],
            "C": ["x", "y"],
        })
        assert graph == {("x", "y"): 1}


# ---------------------------------------------------------------------------
# compute_node_stats tests
# ---------------------------------------------------------------------------


class TestComputeNodeStats:
    """Tests for compute_node_stats."""

    def test_empty_graph(self):
        stats = compute_node_stats({})
        assert stats == {}

    def test_single_edge(self):
        stats = compute_node_stats({("a", "b"): 3})
        assert stats["a"]["in_degree"] == 0
        assert stats["a"]["out_degree"] == 3
        assert stats["a"]["total_flow"] == 3
        assert stats["b"]["in_degree"] == 3
        assert stats["b"]["out_degree"] == 0
        assert stats["b"]["total_flow"] == 3

    def test_self_loop_counts_both(self):
        stats = compute_node_stats({("a", "a"): 2})
        assert stats["a"]["in_degree"] == 2
        assert stats["a"]["out_degree"] == 2
        assert stats["a"]["total_flow"] == 4

    def test_multiple_edges(self):
        graph = {("a", "b"): 5, ("b", "c"): 3, ("c", "a"): 1}
        stats = compute_node_stats(graph)
        assert stats["a"]["in_degree"] == 1
        assert stats["a"]["out_degree"] == 5
        assert stats["b"]["in_degree"] == 5
        assert stats["b"]["out_degree"] == 3
        assert stats["c"]["in_degree"] == 3
        assert stats["c"]["out_degree"] == 1

    def test_sorted_keys(self):
        graph = {("c", "a"): 1, ("b", "a"): 1}
        stats = compute_node_stats(graph)
        assert list(stats.keys()) == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# rank_transitions tests
# ---------------------------------------------------------------------------


class TestRankTransitions:
    """Tests for rank_transitions."""

    def test_empty_graph(self):
        assert rank_transitions({}) == []

    def test_single_edge(self):
        ranked = rank_transitions({("a", "b"): 3})
        assert ranked == [("a", "b", 3)]

    def test_descending_by_count(self):
        graph = {("a", "b"): 5, ("c", "d"): 10, ("e", "f"): 1}
        ranked = rank_transitions(graph)
        assert ranked[0] == ("c", "d", 10)
        assert ranked[1] == ("a", "b", 5)
        assert ranked[2] == ("e", "f", 1)

    def test_lexicographic_tiebreak(self):
        graph = {("b", "c"): 3, ("a", "b"): 3, ("a", "c"): 3}
        ranked = rank_transitions(graph)
        assert ranked == [
            ("a", "b", 3),
            ("a", "c", 3),
            ("b", "c", 3),
        ]

    def test_determinism(self):
        graph = {("x", "y"): 2, ("a", "b"): 2}
        r1 = rank_transitions(graph)
        r2 = rank_transitions(copy.deepcopy(graph))
        assert r1 == r2


# ---------------------------------------------------------------------------
# extract_dominant_flows tests
# ---------------------------------------------------------------------------


class TestExtractDominantFlows:
    """Tests for extract_dominant_flows."""

    def test_empty_graph(self):
        assert extract_dominant_flows({}) == []

    def test_default_threshold(self):
        graph = {("a", "b"): 1, ("c", "d"): 2, ("e", "f"): 5}
        flows = extract_dominant_flows(graph)
        assert len(flows) == 2
        assert flows[0] == ("e", "f", 5)
        assert flows[1] == ("c", "d", 2)

    def test_custom_threshold(self):
        graph = {("a", "b"): 1, ("c", "d"): 3, ("e", "f"): 5}
        flows = extract_dominant_flows(graph, threshold=4)
        assert len(flows) == 1
        assert flows[0] == ("e", "f", 5)

    def test_threshold_one(self):
        graph = {("a", "b"): 1}
        flows = extract_dominant_flows(graph, threshold=1)
        assert len(flows) == 1

    def test_all_below_threshold(self):
        graph = {("a", "b"): 1, ("c", "d"): 1}
        flows = extract_dominant_flows(graph, threshold=5)
        assert flows == []


# ---------------------------------------------------------------------------
# detect_transition_patterns tests
# ---------------------------------------------------------------------------


class TestDetectTransitionPatterns:
    """Tests for detect_transition_patterns."""

    def test_empty_graph(self):
        patterns = detect_transition_patterns({})
        assert patterns["bidirectional"] == []
        assert patterns["self_loops"] == []
        assert patterns["sources"] == []
        assert patterns["sinks"] == []

    def test_bidirectional(self):
        graph = {("a", "b"): 3, ("b", "a"): 2}
        patterns = detect_transition_patterns(graph)
        assert ("a", "b") in patterns["bidirectional"]

    def test_bidirectional_sorted(self):
        graph = {("b", "a"): 1, ("a", "b"): 1, ("c", "d"): 1, ("d", "c"): 1}
        patterns = detect_transition_patterns(graph)
        assert patterns["bidirectional"] == [("a", "b"), ("c", "d")]

    def test_self_loops(self):
        graph = {("a", "a"): 5, ("b", "b"): 2}
        patterns = detect_transition_patterns(graph)
        assert patterns["self_loops"] == ["a", "b"]

    def test_sources(self):
        graph = {("a", "b"): 1, ("a", "c"): 1}
        patterns = detect_transition_patterns(graph)
        assert "a" in patterns["sources"]
        assert "b" not in patterns["sources"]

    def test_sinks(self):
        graph = {("a", "b"): 1, ("c", "b"): 1}
        patterns = detect_transition_patterns(graph)
        assert "b" in patterns["sinks"]
        assert "a" not in patterns["sinks"]

    def test_no_source_or_sink_in_cycle(self):
        graph = {("a", "b"): 1, ("b", "a"): 1}
        patterns = detect_transition_patterns(graph)
        assert patterns["sources"] == []
        assert patterns["sinks"] == []

    def test_self_loop_not_bidirectional(self):
        graph = {("a", "a"): 3}
        patterns = detect_transition_patterns(graph)
        assert patterns["bidirectional"] == []
        assert patterns["self_loops"] == ["a"]

    def test_combined_patterns(self):
        graph = {
            ("a", "b"): 2,
            ("b", "a"): 1,
            ("c", "c"): 3,
            ("d", "b"): 1,
            ("a", "e"): 1,
        }
        patterns = detect_transition_patterns(graph)
        assert ("a", "b") in patterns["bidirectional"]
        assert "c" in patterns["self_loops"]
        assert "d" in patterns["sources"]
        assert "e" in patterns["sinks"]


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestRunTransitionGraphAnalysis:
    """Integration tests via strategy_adapter.run_transition_graph_analysis."""

    def test_basic_integration(self):
        from qec.analysis.strategy_adapter import run_transition_graph_analysis

        runs = _make_runs_with_scores("alpha", [0.5, 0.5, 0.5])
        result = run_transition_graph_analysis(runs)

        assert "transition_graph" in result
        assert "node_stats" in result
        assert "ranked_transitions" in result
        assert "transition_patterns" in result
        # Also includes evolution keys.
        assert "type_trajectories" in result
        assert "evolution" in result

    def test_format_summary(self):
        from qec.analysis.strategy_adapter import (
            format_transition_graph_summary,
            run_transition_graph_analysis,
        )

        runs = _make_runs_with_scores("beta", [0.5, 0.6, 0.5])
        result = run_transition_graph_analysis(runs)
        summary = format_transition_graph_summary(result)

        assert "=== Transition Graph ===" in summary

    def test_format_summary_disabled(self):
        from qec.analysis.strategy_adapter import (
            format_transition_graph_summary,
            run_transition_graph_analysis,
        )

        runs = _make_runs_with_scores("gamma", [0.5, 0.5, 0.5])
        result = run_transition_graph_analysis(runs)
        summary = format_transition_graph_summary(
            result, show_transition_graph=False
        )

        assert "=== Transition Graph ===" not in summary
        assert "=== Evolution Analysis ===" in summary

    def test_determinism_integration(self):
        from qec.analysis.strategy_adapter import run_transition_graph_analysis

        runs = _make_runs_with_scores("delta", [0.5, 0.6, 0.4, 0.5])
        r1 = run_transition_graph_analysis(runs)
        r2 = run_transition_graph_analysis(copy.deepcopy(runs))

        assert r1["transition_graph"] == r2["transition_graph"]
        assert r1["node_stats"] == r2["node_stats"]
        assert r1["ranked_transitions"] == r2["ranked_transitions"]
        assert r1["transition_patterns"] == r2["transition_patterns"]

    def test_multiple_strategies(self):
        from qec.analysis.strategy_adapter import run_transition_graph_analysis

        runs = [
            _make_run([("A", 0.5), ("B", 0.6)]),
            _make_run([("A", 0.5), ("B", 0.7)]),
            _make_run([("A", 0.5), ("B", 0.8)]),
        ]
        result = run_transition_graph_analysis(runs)
        assert isinstance(result["transition_graph"], dict)
        assert isinstance(result["ranked_transitions"], list)

    def test_no_mutation(self):
        from qec.analysis.strategy_adapter import run_transition_graph_analysis

        runs = _make_runs_with_scores("epsilon", [0.5, 0.6])
        original = copy.deepcopy(runs)
        run_transition_graph_analysis(runs)
        assert runs == original

    def test_format_shows_node_stats(self):
        from qec.analysis.strategy_adapter import (
            format_transition_graph_summary,
            run_transition_graph_analysis,
        )

        runs = [
            _make_run([("A", 0.5)]),
            _make_run([("A", 0.6)]),
            _make_run([("A", 0.5)]),
        ]
        result = run_transition_graph_analysis(runs)
        summary = format_transition_graph_summary(result)

        # Should have graph structure if transitions exist.
        if result["ranked_transitions"]:
            assert "Top Transitions:" in summary
            assert "→" in summary

    def test_empty_runs(self):
        from qec.analysis.strategy_adapter import run_transition_graph_analysis

        result = run_transition_graph_analysis([])
        assert result["transition_graph"] == {}
        assert result["node_stats"] == {}
        assert result["ranked_transitions"] == []
