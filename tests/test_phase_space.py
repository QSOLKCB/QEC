"""Tests for phase space analysis (v102.6.1).

Verifies:
- correct attractor detection
- basin strength calculation (bounded to [0, 1))
- escape rate correctness (bounded to [0, 1))
- phase classification with strict priority ordering
- edge cases (isolated node, pure self-loop, pure source, pure sink)
- integration via run_phase_space_analysis
"""

from __future__ import annotations

import copy

from qec.analysis.phase_space import (
    BASIN_THRESHOLD,
    ESCAPE_THRESHOLD,
    ROUND_PRECISION,
    classify_phase_state,
    detect_attractors,
    detect_basins,
    detect_escape_dynamics,
)
from qec.analysis.transition_graph import compute_node_stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _norm(x: float) -> float:
    """Apply the x/(1+x) normalization used by phase_space."""
    return round(x / (1 + x), ROUND_PRECISION)


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

    def test_score_rounded_to_precision(self):
        graph = {("a", "a"): 1, ("b", "a"): 2}
        stats = compute_node_stats(graph)
        result = detect_attractors(graph, stats)
        score = result["a"]["score"]
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
        # in=8, out=0 -> raw=8/1=8, strength=8/9
        assert result["b"]["basin_strength"] == _norm(8.0)

    def test_balanced_flow(self):
        graph = {("a", "b"): 3, ("b", "a"): 3}
        stats = compute_node_stats(graph)
        result = detect_basins(graph, stats)
        # a: in=3, out=3 -> raw=3/4=0.75, strength=0.75/1.75=3/7
        assert result["a"]["basin_strength"] == _norm(0.75)
        assert result["a"]["basin_size"] == 3

    def test_pure_source(self):
        graph = {("a", "b"): 5}
        stats = compute_node_stats(graph)
        result = detect_basins(graph, stats)
        # a: in=0, out=5 -> raw=0, strength=0
        assert result["a"]["basin_size"] == 0
        assert result["a"]["basin_strength"] == 0.0

    def test_self_loop(self):
        graph = {("a", "a"): 4}
        stats = compute_node_stats(graph)
        result = detect_basins(graph, stats)
        # in=4, out=4 -> raw=4/5=0.8, strength=0.8/1.8=4/9
        assert result["a"]["basin_strength"] == _norm(0.8)

    def test_sorted_keys(self):
        graph = {("c", "a"): 1, ("b", "a"): 1}
        stats = compute_node_stats(graph)
        result = detect_basins(graph, stats)
        assert list(result.keys()) == ["a", "b", "c"]

    def test_bounded_output(self):
        """Basin strength must always be in [0, 1)."""
        graph = {("a", "b"): 100, ("c", "b"): 200}
        stats = compute_node_stats(graph)
        result = detect_basins(graph, stats)
        for node in result:
            assert 0.0 <= result[node]["basin_strength"] < 1.0


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
        # a: out=5, in=0 -> raw=5/1=5, rate=5/6
        assert result["a"]["escape_rate"] == _norm(5.0)

    def test_pure_sink(self):
        graph = {("a", "b"): 5}
        stats = compute_node_stats(graph)
        result = detect_escape_dynamics(graph, stats)
        # b: out=0, in=5 -> raw=0, rate=0
        assert result["b"]["escape_rate"] == 0.0

    def test_balanced(self):
        graph = {("a", "b"): 3, ("b", "a"): 3}
        stats = compute_node_stats(graph)
        result = detect_escape_dynamics(graph, stats)
        # a: out=3, in=3 -> raw=3/4=0.75, rate=0.75/1.75=3/7
        assert result["a"]["escape_rate"] == _norm(0.75)

    def test_self_loop(self):
        graph = {("a", "a"): 4}
        stats = compute_node_stats(graph)
        result = detect_escape_dynamics(graph, stats)
        # in=4, out=4 -> raw=4/5=0.8, rate=0.8/1.8=4/9
        assert result["a"]["escape_rate"] == _norm(0.8)

    def test_sorted_keys(self):
        graph = {("c", "a"): 1, ("b", "a"): 1}
        stats = compute_node_stats(graph)
        result = detect_escape_dynamics(graph, stats)
        assert list(result.keys()) == ["a", "b", "c"]

    def test_bounded_output(self):
        """Escape rate must always be in [0, 1)."""
        graph = {("a", "b"): 100, ("a", "c"): 200}
        stats = compute_node_stats(graph)
        result = detect_escape_dynamics(graph, stats)
        for node in result:
            assert 0.0 <= result[node]["escape_rate"] < 1.0


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
        basins = {"a": {"basin_size": 10, "basin_strength": 0.8}}
        escape = {"a": {"escape_rate": 0.1}}
        result = classify_phase_state(attractors, basins, escape)
        assert result["a"]["phase"] == "strong_attractor"

    def test_weak_attractor(self):
        attractors = {"a": {"is_attractor": True, "score": 1.0}}
        basins = {"a": {"basin_size": 1, "basin_strength": 0.3}}
        escape = {"a": {"escape_rate": 0.3}}
        result = classify_phase_state(attractors, basins, escape)
        assert result["a"]["phase"] == "weak_attractor"

    def test_weak_attractor_boundary(self):
        # basin_strength == BASIN_THRESHOLD -> not > threshold -> weak
        attractors = {"a": {"is_attractor": True, "score": 2.0}}
        basins = {"a": {"basin_size": 3, "basin_strength": BASIN_THRESHOLD}}
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
        escape = {"a": {"escape_rate": 0.8}}
        result = classify_phase_state(attractors, basins, escape)
        assert result["a"]["phase"] == "transient"

    def test_neutral_fallback(self):
        attractors = {"a": {"is_attractor": False, "score": 0.0}}
        basins = {"a": {"basin_size": 1, "basin_strength": 0.3}}
        escape = {"a": {"escape_rate": 0.2}}
        result = classify_phase_state(attractors, basins, escape)
        assert result["a"]["phase"] == "neutral"

    def test_priority_attractor_over_basin(self):
        # Attractor with high basin_strength -> strong_attractor, not basin
        attractors = {"a": {"is_attractor": True, "score": 3.0}}
        basins = {"a": {"basin_size": 5, "basin_strength": 0.8}}
        escape = {"a": {"escape_rate": 0.6}}
        result = classify_phase_state(attractors, basins, escape)
        assert result["a"]["phase"] == "strong_attractor"

    def test_basin_over_transient_priority(self):
        # Basin dominates transient even when escape_rate is high
        attractors = {"a": {"is_attractor": False, "score": -1.0}}
        basins = {"a": {"basin_size": 3, "basin_strength": 0.8}}
        escape = {"a": {"escape_rate": 0.9}}
        result = classify_phase_state(attractors, basins, escape)
        assert result["a"]["phase"] == "basin"

    def test_multiple_nodes_different_phases(self):
        attractors = {
            "a": {"is_attractor": True, "score": 5.0},
            "b": {"is_attractor": False, "score": -1.0},
            "c": {"is_attractor": False, "score": -2.0},
        }
        basins = {
            "a": {"basin_size": 10, "basin_strength": 0.9},
            "b": {"basin_size": 3, "basin_strength": 0.75},
            "c": {"basin_size": 0, "basin_strength": 0.0},
        }
        escape = {
            "a": {"escape_rate": 0.1},
            "b": {"escape_rate": 0.3},
            "c": {"escape_rate": 0.8},
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

    def test_uses_constants(self):
        """Verify thresholds match exported constants."""
        # Just above BASIN_THRESHOLD -> strong_attractor
        attractors = {"a": {"is_attractor": True, "score": 1.0}}
        basins = {"a": {"basin_size": 2, "basin_strength": BASIN_THRESHOLD + 0.01}}
        escape = {"a": {"escape_rate": 0.0}}
        result = classify_phase_state(attractors, basins, escape)
        assert result["a"]["phase"] == "strong_attractor"

        # Just above ESCAPE_THRESHOLD -> transient
        attractors2 = {"b": {"is_attractor": False, "score": 0.0}}
        basins2 = {"b": {"basin_size": 0, "basin_strength": 0.0}}
        escape2 = {"b": {"escape_rate": ESCAPE_THRESHOLD + 0.01}}
        result2 = classify_phase_state(attractors2, basins2, escape2)
        assert result2["b"]["phase"] == "transient"


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
        # in=10, out=10 -> raw=10/11, strength=norm(10/11)=10/21
        assert abs(bas["a"]["basin_strength"] - _norm(10.0 / 11.0)) < 1e-10
        assert bas["a"]["basin_strength"] < 1.0  # bounded
        # escape: raw=10/11, rate=norm(10/11)=10/21
        assert abs(esc["a"]["escape_rate"] - _norm(10.0 / 11.0)) < 1e-10
        assert esc["a"]["escape_rate"] < 1.0  # bounded
        # 10/21 ≈ 0.476 < BASIN_THRESHOLD -> weak_attractor
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
        # out=5, in=0 -> raw=5, rate=5/6
        assert esc["src"]["escape_rate"] == _norm(5.0)
        assert esc["src"]["escape_rate"] < 1.0  # bounded
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
        # in=7, out=0 -> raw=7, strength=7/8
        assert bas["sink"]["basin_strength"] == _norm(7.0)
        assert bas["sink"]["basin_strength"] < 1.0  # bounded
        assert esc["sink"]["escape_rate"] == 0.0
        # Not attractor, basin_strength > BASIN_THRESHOLD -> basin
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

    def test_bounded_outputs_integration(self):
        """All basin_strength and escape_rate values must be in [0, 1)."""
        from qec.analysis.strategy_adapter import run_phase_space_analysis

        runs = _make_runs_with_scores("zeta", [0.5, 0.6, 0.4, 0.5, 0.7])
        result = run_phase_space_analysis(runs)

        for node, bas in result["basins"].items():
            assert 0.0 <= bas["basin_strength"] < 1.0, (
                f"{node}: basin_strength={bas['basin_strength']}"
            )
        for node, esc in result["escape_dynamics"].items():
            assert 0.0 <= esc["escape_rate"] < 1.0, (
                f"{node}: escape_rate={esc['escape_rate']}"
            )
