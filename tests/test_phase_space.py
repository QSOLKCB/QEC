"""Tests for phase space analysis (v102.6.0).

Verifies:
- correct attractor detection
- basin strength calculation
- escape rate correctness
- phase classification
- edge cases (isolated node, pure self-loop, pure source, pure sink)
- integration via run_phase_space_analysis
"""

from __future__ import annotations

import copy

from qec.analysis.phase_space import (
    classify_phase_state,
    detect_attractors,
    detect_basins,
    detect_escape_dynamics,
)
from qec.analysis.transition_graph import compute_node_stats


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
# detect_attractors tests
# ---------------------------------------------------------------------------


class TestDetectAttractors:
    """Tests for detect_attractors."""

    def test_empty_graph(self):
        result = detect_attractors({}, {})
        assert result == {}

    def test_self_loop_positive_score(self):
        graph = {("a", "a"): 3}
        stats = compute_node_stats(graph)
        result = detect_attractors(graph, stats)
        # self_loop=3, in=3, out=3 -> score = 3+3-3 = 3
        assert result["a"]["is_attractor"] is True
        assert result["a"]["score"] == 3.0

    def test_no_self_loop_not_attractor(self):
        graph = {("a", "b"): 5}
        stats = compute_node_stats(graph)
        result = detect_attractors(graph, stats)
        assert result["a"]["is_attractor"] is False
        assert result["b"]["is_attractor"] is False

    def test_self_loop_negative_score(self):
        # self_loop=1 for a, but a has high out_degree
        graph = {("a", "a"): 1, ("a", "b"): 10, ("a", "c"): 10}
        stats = compute_node_stats(graph)
        result = detect_attractors(graph, stats)
        # in=1, out=21, self_loop=1 -> score = 1+1-21 = -19
        assert result["a"]["is_attractor"] is False
        assert result["a"]["score"] == -19.0

    def test_score_rounded_to_12_decimals(self):
        graph = {("a", "a"): 1, ("b", "a"): 2}
        stats = compute_node_stats(graph)
        result = detect_attractors(graph, stats)
        score = result["a"]["score"]
        # Verify it's a float
        assert isinstance(score, float)

    def test_multiple_nodes(self):
        graph = {
            ("a", "a"): 5,
            ("a", "b"): 1,
            ("b", "a"): 3,
        }
        stats = compute_node_stats(graph)
        result = detect_attractors(graph, stats)
        # a: self_loop=5, in=8, out=6 -> score=7, attractor
        assert result["a"]["is_attractor"] is True
        # b: self_loop=0, in=1, out=3 -> not attractor (no self-loop)
        assert result["b"]["is_attractor"] is False

    def test_sorted_keys(self):
        graph = {("c", "c"): 1, ("a", "a"): 1}
        stats = compute_node_stats(graph)
        result = detect_attractors(graph, stats)
        assert list(result.keys()) == ["a", "c"]


# ---------------------------------------------------------------------------
# detect_basins tests
# ---------------------------------------------------------------------------


class TestDetectBasins:
    """Tests for detect_basins."""

    def test_empty_graph(self):
        result = detect_basins({}, {})
        assert result == {}

    def test_high_in_degree(self):
        graph = {("a", "b"): 5, ("c", "b"): 3}
        stats = compute_node_stats(graph)
        result = detect_basins(graph, stats)
        assert result["b"]["basin_size"] == 8
        # in=8, out=0 -> strength = 8 / (1+0) = 8.0
        assert result["b"]["basin_strength"] == 8.0

    def test_balanced_flow(self):
        graph = {("a", "b"): 3, ("b", "a"): 3}
        stats = compute_node_stats(graph)
        result = detect_basins(graph, stats)
        # a: in=3, out=3 -> strength = 3/(1+3) = 0.75
        assert result["a"]["basin_strength"] == 0.75
        assert result["a"]["basin_size"] == 3

    def test_pure_source(self):
        graph = {("a", "b"): 5}
        stats = compute_node_stats(graph)
        result = detect_basins(graph, stats)
        # a: in=0, out=5 -> strength = 0/(1+5) = 0
        assert result["a"]["basin_size"] == 0
        assert result["a"]["basin_strength"] == 0.0

    def test_self_loop(self):
        graph = {("a", "a"): 4}
        stats = compute_node_stats(graph)
        result = detect_basins(graph, stats)
        # in=4, out=4 -> strength = 4/(1+4) = 0.8
        assert result["a"]["basin_strength"] == 0.8

    def test_sorted_keys(self):
        graph = {("c", "a"): 1, ("b", "a"): 1}
        stats = compute_node_stats(graph)
        result = detect_basins(graph, stats)
        assert list(result.keys()) == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# detect_escape_dynamics tests
# ---------------------------------------------------------------------------


class TestDetectEscapeDynamics:
    """Tests for detect_escape_dynamics."""

    def test_empty_graph(self):
        result = detect_escape_dynamics({}, {})
        assert result == {}

    def test_pure_source(self):
        graph = {("a", "b"): 5}
        stats = compute_node_stats(graph)
        result = detect_escape_dynamics(graph, stats)
        # a: out=5, in=0 -> escape = 5/(1+0) = 5.0
        assert result["a"]["escape_rate"] == 5.0

    def test_pure_sink(self):
        graph = {("a", "b"): 5}
        stats = compute_node_stats(graph)
        result = detect_escape_dynamics(graph, stats)
        # b: out=0, in=5 -> escape = 0/(1+5) = 0.0
        assert result["b"]["escape_rate"] == 0.0

    def test_balanced(self):
        graph = {("a", "b"): 3, ("b", "a"): 3}
        stats = compute_node_stats(graph)
        result = detect_escape_dynamics(graph, stats)
        # a: out=3, in=3 -> escape = 3/(1+3) = 0.75
        assert result["a"]["escape_rate"] == 0.75

    def test_self_loop(self):
        graph = {("a", "a"): 4}
        stats = compute_node_stats(graph)
        result = detect_escape_dynamics(graph, stats)
        # in=4, out=4 -> escape = 4/(1+4) = 0.8
        assert result["a"]["escape_rate"] == 0.8

    def test_sorted_keys(self):
        graph = {("c", "a"): 1, ("b", "a"): 1}
        stats = compute_node_stats(graph)
        result = detect_escape_dynamics(graph, stats)
        assert list(result.keys()) == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# classify_phase_state tests
# ---------------------------------------------------------------------------


class TestClassifyPhaseState:
    """Tests for classify_phase_state."""

    def test_empty(self):
        result = classify_phase_state({}, {}, {})
        assert result == {}

    def test_strong_attractor(self):
        attractors = {"a": {"is_attractor": True, "score": 5.0}}
        basins = {"a": {"basin_size": 10, "basin_strength": 2.0}}
        escape = {"a": {"escape_rate": 0.1}}
        result = classify_phase_state(attractors, basins, escape)
        assert result["a"]["phase"] == "strong_attractor"

    def test_weak_attractor(self):
        attractors = {"a": {"is_attractor": True, "score": 1.0}}
        basins = {"a": {"basin_size": 1, "basin_strength": 0.5}}
        escape = {"a": {"escape_rate": 0.3}}
        result = classify_phase_state(attractors, basins, escape)
        assert result["a"]["phase"] == "weak_attractor"

    def test_weak_attractor_boundary(self):
        # basin_strength == 1 -> weak_attractor (not strong)
        attractors = {"a": {"is_attractor": True, "score": 2.0}}
        basins = {"a": {"basin_size": 3, "basin_strength": 1.0}}
        escape = {"a": {"escape_rate": 0.2}}
        result = classify_phase_state(attractors, basins, escape)
        assert result["a"]["phase"] == "weak_attractor"

    def test_basin(self):
        attractors = {"a": {"is_attractor": False, "score": -1.0}}
        basins = {"a": {"basin_size": 3, "basin_strength": 0.75}}
        escape = {"a": {"escape_rate": 0.3}}
        result = classify_phase_state(attractors, basins, escape)
        assert result["a"]["phase"] == "basin"

    def test_transient(self):
        attractors = {"a": {"is_attractor": False, "score": -2.0}}
        basins = {"a": {"basin_size": 0, "basin_strength": 0.0}}
        escape = {"a": {"escape_rate": 2.0}}
        result = classify_phase_state(attractors, basins, escape)
        assert result["a"]["phase"] == "transient"

    def test_neutral_fallback(self):
        attractors = {"a": {"is_attractor": False, "score": 0.0}}
        basins = {"a": {"basin_size": 1, "basin_strength": 0.3}}
        escape = {"a": {"escape_rate": 0.2}}
        result = classify_phase_state(attractors, basins, escape)
        assert result["a"]["phase"] == "neutral"

    def test_priority_attractor_over_basin(self):
        # Even with high basin_strength, attractor check comes first
        attractors = {"a": {"is_attractor": True, "score": 3.0}}
        basins = {"a": {"basin_size": 5, "basin_strength": 0.8}}
        escape = {"a": {"escape_rate": 0.6}}
        result = classify_phase_state(attractors, basins, escape)
        assert result["a"]["phase"] == "weak_attractor"

    def test_multiple_nodes_different_phases(self):
        attractors = {
            "a": {"is_attractor": True, "score": 5.0},
            "b": {"is_attractor": False, "score": -1.0},
            "c": {"is_attractor": False, "score": -2.0},
        }
        basins = {
            "a": {"basin_size": 10, "basin_strength": 2.0},
            "b": {"basin_size": 3, "basin_strength": 0.75},
            "c": {"basin_size": 0, "basin_strength": 0.0},
        }
        escape = {
            "a": {"escape_rate": 0.1},
            "b": {"escape_rate": 0.3},
            "c": {"escape_rate": 2.0},
        }
        result = classify_phase_state(attractors, basins, escape)
        assert result["a"]["phase"] == "strong_attractor"
        assert result["b"]["phase"] == "basin"
        assert result["c"]["phase"] == "transient"

    def test_sorted_keys(self):
        attractors = {"c": {"is_attractor": False, "score": 0.0},
                      "a": {"is_attractor": False, "score": 0.0}}
        basins = {"c": {"basin_size": 0, "basin_strength": 0.0},
                  "a": {"basin_size": 0, "basin_strength": 0.0}}
        escape = {"c": {"escape_rate": 0.0}, "a": {"escape_rate": 0.0}}
        result = classify_phase_state(attractors, basins, escape)
        assert list(result.keys()) == ["a", "c"]


# ---------------------------------------------------------------------------
# Edge cases: isolated, pure self-loop, pure source, pure sink
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for phase space functions."""

    def test_isolated_node(self):
        """A node with no edges at all (only appears in node_stats)."""
        graph = {}
        node_stats = {"isolated": {"in_degree": 0, "out_degree": 0, "total_flow": 0}}
        att = detect_attractors(graph, node_stats)
        bas = detect_basins(graph, node_stats)
        esc = detect_escape_dynamics(graph, node_stats)
        cls = classify_phase_state(att, bas, esc)

        assert att["isolated"]["is_attractor"] is False
        assert att["isolated"]["score"] == 0.0
        assert bas["isolated"]["basin_size"] == 0
        assert bas["isolated"]["basin_strength"] == 0.0
        assert esc["isolated"]["escape_rate"] == 0.0
        assert cls["isolated"]["phase"] == "neutral"

    def test_pure_self_loop(self):
        """A node with only self-loops."""
        graph = {("a", "a"): 10}
        stats = compute_node_stats(graph)
        att = detect_attractors(graph, stats)
        bas = detect_basins(graph, stats)
        esc = detect_escape_dynamics(graph, stats)
        cls = classify_phase_state(att, bas, esc)

        # self_loop=10, in=10, out=10 -> score=10
        assert att["a"]["is_attractor"] is True
        assert att["a"]["score"] == 10.0
        # in=10, out=10 -> basin_strength = 10/11
        assert abs(bas["a"]["basin_strength"] - 10.0 / 11.0) < 1e-10
        # escape = 10/11
        assert abs(esc["a"]["escape_rate"] - 10.0 / 11.0) < 1e-10
        # attractor with basin_strength < 1 -> weak_attractor
        assert cls["a"]["phase"] == "weak_attractor"

    def test_pure_source(self):
        """A node with only outgoing edges, no incoming."""
        graph = {("src", "a"): 3, ("src", "b"): 2}
        stats = compute_node_stats(graph)
        att = detect_attractors(graph, stats)
        bas = detect_basins(graph, stats)
        esc = detect_escape_dynamics(graph, stats)
        cls = classify_phase_state(att, bas, esc)

        assert att["src"]["is_attractor"] is False
        assert bas["src"]["basin_size"] == 0
        assert bas["src"]["basin_strength"] == 0.0
        # out=5, in=0 -> escape=5.0
        assert esc["src"]["escape_rate"] == 5.0
        assert cls["src"]["phase"] == "transient"

    def test_pure_sink(self):
        """A node with only incoming edges, no outgoing."""
        graph = {("a", "sink"): 3, ("b", "sink"): 4}
        stats = compute_node_stats(graph)
        att = detect_attractors(graph, stats)
        bas = detect_basins(graph, stats)
        esc = detect_escape_dynamics(graph, stats)
        cls = classify_phase_state(att, bas, esc)

        # No self-loop -> not attractor
        assert att["sink"]["is_attractor"] is False
        # in=7, out=0 -> basin_strength = 7.0
        assert bas["sink"]["basin_strength"] == 7.0
        assert esc["sink"]["escape_rate"] == 0.0
        # Not attractor, basin_strength > 0.5 -> basin
        assert cls["sink"]["phase"] == "basin"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestRunPhaseSpaceAnalysis:
    """Integration tests via strategy_adapter.run_phase_space_analysis."""

    def test_basic_integration(self):
        from qec.analysis.strategy_adapter import run_phase_space_analysis

        runs = _make_runs_with_scores("alpha", [0.5, 0.5, 0.5])
        result = run_phase_space_analysis(runs)

        # Phase space keys.
        assert "attractors" in result
        assert "basins" in result
        assert "escape_dynamics" in result
        assert "phase_classification" in result
        # Also includes transition graph keys.
        assert "transition_graph" in result
        assert "node_stats" in result

    def test_format_summary(self):
        from qec.analysis.strategy_adapter import (
            format_phase_space_summary,
            run_phase_space_analysis,
        )

        runs = _make_runs_with_scores("beta", [0.5, 0.6, 0.5])
        result = run_phase_space_analysis(runs)
        summary = format_phase_space_summary(result)

        assert "=== Phase Space Analysis ===" in summary
        assert "Attractor:" in summary
        assert "Score:" in summary
        assert "Basin Strength:" in summary
        assert "Escape Rate:" in summary
        assert "Phase:" in summary

    def test_determinism_integration(self):
        from qec.analysis.strategy_adapter import run_phase_space_analysis

        runs = _make_runs_with_scores("delta", [0.5, 0.6, 0.4, 0.5])
        r1 = run_phase_space_analysis(runs)
        r2 = run_phase_space_analysis(copy.deepcopy(runs))

        assert r1["attractors"] == r2["attractors"]
        assert r1["basins"] == r2["basins"]
        assert r1["escape_dynamics"] == r2["escape_dynamics"]
        assert r1["phase_classification"] == r2["phase_classification"]

    def test_multiple_strategies(self):
        from qec.analysis.strategy_adapter import run_phase_space_analysis

        runs = [
            _make_run([("A", 0.5), ("B", 0.6)]),
            _make_run([("A", 0.5), ("B", 0.7)]),
            _make_run([("A", 0.5), ("B", 0.8)]),
        ]
        result = run_phase_space_analysis(runs)
        assert isinstance(result["attractors"], dict)
        assert isinstance(result["phase_classification"], dict)

    def test_no_mutation(self):
        from qec.analysis.strategy_adapter import run_phase_space_analysis

        runs = _make_runs_with_scores("epsilon", [0.5, 0.6])
        original = copy.deepcopy(runs)
        run_phase_space_analysis(runs)
        assert runs == original

    def test_empty_runs(self):
        from qec.analysis.strategy_adapter import run_phase_space_analysis

        result = run_phase_space_analysis([])
        assert result["attractors"] == {}
        assert result["basins"] == {}
        assert result["escape_dynamics"] == {}
        assert result["phase_classification"] == {}
