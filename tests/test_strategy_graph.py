"""Tests for strategy graph and policy topology analysis (v103.9.0).

Verifies:
- correct transition counts
- deterministic graph construction
- correct node stats
- policy stability metrics
- transition pattern detection
- topology classification
- no mutation of inputs
- integration correctness via strategy_adapter
"""

from __future__ import annotations

import copy

from qec.analysis.strategy_graph import (
    ROUND_PRECISION,
    build_policy_graph,
    classify_policy_topology,
    compute_policy_node_stats,
    compute_policy_stability,
    detect_policy_patterns,
    format_strategy_graph_summary,
)


# ---------------------------------------------------------------------------
# build_policy_graph
# ---------------------------------------------------------------------------


class TestBuildPolicyGraph:
    """Tests for ``build_policy_graph``."""

    def test_basic_transitions(self):
        history = ["stability_first", "balanced", "balanced", "sync_first"]
        graph = build_policy_graph(history)
        assert graph == {
            ("stability_first", "balanced"): 1,
            ("balanced", "balanced"): 1,
            ("balanced", "sync_first"): 1,
        }

    def test_empty_history(self):
        assert build_policy_graph([]) == {}

    def test_single_entry(self):
        assert build_policy_graph(["balanced"]) == {}

    def test_all_same(self):
        history = ["balanced", "balanced", "balanced"]
        graph = build_policy_graph(history)
        assert graph == {("balanced", "balanced"): 2}

    def test_alternating(self):
        history = ["a", "b", "a", "b"]
        graph = build_policy_graph(history)
        assert graph == {("a", "b"): 2, ("b", "a"): 1}

    def test_no_mutation(self):
        history = ["a", "b", "c"]
        original = list(history)
        build_policy_graph(history)
        assert history == original

    def test_deterministic(self):
        history = ["x", "y", "z", "x", "y"]
        g1 = build_policy_graph(history)
        g2 = build_policy_graph(history)
        assert g1 == g2


# ---------------------------------------------------------------------------
# compute_policy_node_stats
# ---------------------------------------------------------------------------


class TestComputePolicyNodeStats:
    """Tests for ``compute_policy_node_stats``."""

    def test_basic_stats(self):
        graph = {
            ("a", "b"): 2,
            ("b", "c"): 1,
            ("c", "a"): 1,
        }
        stats = compute_policy_node_stats(graph)
        assert stats["a"]["out_degree"] == 2
        assert stats["a"]["in_degree"] == 1
        assert stats["a"]["total_flow"] == 3
        assert stats["b"]["out_degree"] == 1
        assert stats["b"]["in_degree"] == 2
        assert stats["c"]["out_degree"] == 1
        assert stats["c"]["in_degree"] == 1

    def test_empty_graph(self):
        assert compute_policy_node_stats({}) == {}

    def test_self_loop(self):
        graph = {("a", "a"): 3}
        stats = compute_policy_node_stats(graph)
        assert stats["a"]["in_degree"] == 3
        assert stats["a"]["out_degree"] == 3
        assert stats["a"]["total_flow"] == 6

    def test_sorted_output(self):
        graph = {("z", "a"): 1, ("m", "z"): 1}
        stats = compute_policy_node_stats(graph)
        assert list(stats.keys()) == ["a", "m", "z"]

    def test_no_mutation(self):
        graph = {("a", "b"): 1}
        original = dict(graph)
        compute_policy_node_stats(graph)
        assert graph == original


# ---------------------------------------------------------------------------
# compute_policy_stability
# ---------------------------------------------------------------------------


class TestComputePolicyStability:
    """Tests for ``compute_policy_stability``."""

    def test_empty_history(self):
        result = compute_policy_stability([])
        assert result["self_loop_ratio"] == 0.0
        assert result["switch_rate"] == 0.0
        assert result["dominant_policy"] == ""
        assert result["longest_streak"] == 0

    def test_single_entry(self):
        result = compute_policy_stability(["a"])
        assert result["dominant_policy"] == "a"
        assert result["longest_streak"] == 1

    def test_all_same(self):
        result = compute_policy_stability(["a", "a", "a", "a"])
        assert result["self_loop_ratio"] == 1.0
        assert result["switch_rate"] == 0.0
        assert result["dominant_policy"] == "a"
        assert result["longest_streak"] == 4

    def test_all_different(self):
        result = compute_policy_stability(["a", "b", "c"])
        assert result["self_loop_ratio"] == 0.0
        assert result["switch_rate"] == 1.0
        assert result["longest_streak"] == 1

    def test_mixed(self):
        history = ["a", "a", "b", "a", "a", "a"]
        result = compute_policy_stability(history)
        # 5 transitions: a->a, a->b, b->a, a->a, a->a
        # self-loops: 3, switches: 2
        assert result["self_loop_ratio"] == round(3 / 5, ROUND_PRECISION)
        assert result["switch_rate"] == round(2 / 5, ROUND_PRECISION)
        assert result["dominant_policy"] == "a"
        assert result["longest_streak"] == 3

    def test_dominant_tie_lexicographic(self):
        history = ["a", "b", "a", "b"]
        result = compute_policy_stability(history)
        # Both have count 2, "a" wins lexicographically
        assert result["dominant_policy"] == "a"

    def test_no_mutation(self):
        history = ["a", "b", "c"]
        original = list(history)
        compute_policy_stability(history)
        assert history == original

    def test_deterministic(self):
        history = ["x", "y", "x", "y", "x"]
        r1 = compute_policy_stability(history)
        r2 = compute_policy_stability(history)
        assert r1 == r2


# ---------------------------------------------------------------------------
# detect_policy_patterns
# ---------------------------------------------------------------------------


class TestDetectPolicyPatterns:
    """Tests for ``detect_policy_patterns``."""

    def test_empty_graph(self):
        result = detect_policy_patterns({})
        assert result["bidirectional"] == []
        assert result["self_loops"] == []
        assert result["sources"] == []
        assert result["sinks"] == []

    def test_bidirectional(self):
        graph = {("a", "b"): 1, ("b", "a"): 1}
        result = detect_policy_patterns(graph)
        assert result["bidirectional"] == [("a", "b")]

    def test_self_loops(self):
        graph = {("a", "a"): 3, ("b", "b"): 1}
        result = detect_policy_patterns(graph)
        assert result["self_loops"] == ["a", "b"]

    def test_sources_and_sinks(self):
        graph = {("a", "b"): 1, ("b", "c"): 1}
        result = detect_policy_patterns(graph)
        assert result["sources"] == ["a"]
        assert result["sinks"] == ["c"]

    def test_no_mutation(self):
        graph = {("a", "b"): 1, ("b", "a"): 1}
        original = dict(graph)
        detect_policy_patterns(graph)
        assert graph == original


# ---------------------------------------------------------------------------
# classify_policy_topology
# ---------------------------------------------------------------------------


class TestClassifyPolicyTopology:
    """Tests for ``classify_policy_topology``."""

    def test_empty_graph(self):
        assert classify_policy_topology({}, {}) == "stable"

    def test_stable_mostly_self_loops(self):
        graph = {("a", "a"): 8, ("a", "b"): 1, ("b", "a"): 1}
        stats = compute_policy_node_stats(graph)
        assert classify_policy_topology(graph, stats) == "stable"

    def test_converging(self):
        graph = {
            ("a", "c"): 3,
            ("b", "c"): 3,
            ("d", "c"): 2,
        }
        stats = compute_policy_node_stats(graph)
        assert classify_policy_topology(graph, stats) == "converging"

    def test_diverging(self):
        graph = {
            ("a", "b"): 3,
            ("a", "c"): 3,
            ("a", "d"): 2,
        }
        stats = compute_policy_node_stats(graph)
        assert classify_policy_topology(graph, stats) == "diverging"

    def test_cyclic(self):
        graph = {
            ("a", "b"): 3,
            ("b", "a"): 3,
            ("c", "d"): 1,
            ("d", "c"): 1,
        }
        stats = compute_policy_node_stats(graph)
        assert classify_policy_topology(graph, stats) == "cyclic"

    def test_mixed(self):
        graph = {
            ("a", "b"): 1,
            ("b", "c"): 1,
            ("c", "d"): 1,
            ("d", "a"): 1,
            ("a", "c"): 1,
        }
        stats = compute_policy_node_stats(graph)
        assert classify_policy_topology(graph, stats) == "mixed"

    def test_deterministic(self):
        graph = {("a", "b"): 2, ("b", "c"): 1, ("c", "a"): 1}
        stats = compute_policy_node_stats(graph)
        t1 = classify_policy_topology(graph, stats)
        t2 = classify_policy_topology(graph, stats)
        assert t1 == t2


# ---------------------------------------------------------------------------
# format_strategy_graph_summary
# ---------------------------------------------------------------------------


class TestFormatStrategyGraphSummary:
    """Tests for ``format_strategy_graph_summary``."""

    def test_basic_format(self):
        result = {
            "graph": {
                ("balanced", "balanced"): 5,
                ("stability_first", "balanced"): 3,
                ("balanced", "sync_first"): 2,
            },
            "topology": "converging",
            "stability": {
                "dominant_policy": "balanced",
                "switch_rate": 0.4,
                "self_loop_ratio": 0.5,
                "longest_streak": 5,
            },
            "patterns": {
                "bidirectional": [],
                "self_loops": ["balanced"],
                "sources": ["stability_first"],
                "sinks": ["sync_first"],
            },
        }
        text = format_strategy_graph_summary(result)
        assert "=== Strategy Graph ===" in text
        assert "Topology: converging" in text
        assert "Dominant Policy: balanced" in text
        assert "balanced -> balanced: 5" in text

    def test_empty_result(self):
        result = {"graph": {}, "topology": "stable", "stability": {}, "patterns": {}}
        text = format_strategy_graph_summary(result)
        assert "=== Strategy Graph ===" in text
        assert "Topology: stable" in text


# ---------------------------------------------------------------------------
# Integration: build + stats + stability + patterns + topology
# ---------------------------------------------------------------------------


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline(self):
        history = [
            "stability_first", "balanced", "balanced",
            "balanced", "sync_first", "balanced",
        ]
        graph = build_policy_graph(history)
        stats = compute_policy_node_stats(graph)
        stability = compute_policy_stability(history)
        patterns = detect_policy_patterns(graph)
        topology = classify_policy_topology(graph, stats)

        # Graph has correct edges.
        assert ("balanced", "balanced") in graph
        assert ("stability_first", "balanced") in graph
        assert ("balanced", "sync_first") in graph
        assert ("sync_first", "balanced") in graph

        # Stats are consistent.
        for node, node_stats in stats.items():
            assert node_stats["total_flow"] == (
                node_stats["in_degree"] + node_stats["out_degree"]
            )

        # Stability is consistent.
        assert stability["dominant_policy"] == "balanced"
        assert stability["longest_streak"] == 3
        assert (
            abs(stability["self_loop_ratio"] + stability["switch_rate"] - 1.0)
            < 1e-9
        )

        # Topology is a valid string.
        assert topology in ("stable", "converging", "diverging", "cyclic", "mixed")

    def test_no_mutation_full_pipeline(self):
        history = ["a", "b", "a", "c"]
        original = list(history)

        graph = build_policy_graph(history)
        graph_copy = dict(graph)

        compute_policy_node_stats(graph)
        assert graph == graph_copy

        compute_policy_stability(history)
        assert history == original

        detect_policy_patterns(graph)
        assert graph == graph_copy

        stats = compute_policy_node_stats(graph)
        classify_policy_topology(graph, stats)
        assert graph == graph_copy
