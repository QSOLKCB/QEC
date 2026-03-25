"""Tests for v102.0.0 strategy analysis modules.

Verifies:
- embedding is deterministic
- clustering is stable
- Pareto front is correct
- representation comparison is accurate
- no mutation of inputs
- consistent outputs across repeated calls
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

import pytest

from qec.analysis.pareto_analysis import compute_pareto_front
from qec.analysis.representation_analysis import compare_representations
from qec.analysis.strategy_clustering import cluster_strategies
from qec.analysis.strategy_embedding import embed_strategies_2d
from qec.visualization.strategy_map import render_strategy_map


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _make_strategy(
    name: str,
    design_score: float = 0.5,
    confidence_efficiency: float = 0.5,
    temporal_stability: float = 0.5,
    trust_modulation: float = 0.5,
    consistency_gap: float = 0.0,
    revival_strength: float = 0.0,
    state_system: str = "ternary",
) -> Dict[str, Any]:
    return {
        "name": name,
        "state_system": state_system,
        "metrics": {
            "design_score": design_score,
            "confidence_efficiency": confidence_efficiency,
            "temporal_stability": temporal_stability,
            "trust_modulation": trust_modulation,
            "consistency_gap": consistency_gap,
            "revival_strength": revival_strength,
        },
    }


def _sample_strategies() -> List[Dict[str, Any]]:
    return [
        _make_strategy("alpha", 0.9, 0.8, 0.7, 0.6, 0.1, 0.2, "ternary"),
        _make_strategy("beta", 0.85, 0.75, 0.65, 0.55, 0.1, 0.1, "ternary"),
        _make_strategy("gamma", 0.3, 0.4, 0.5, 0.6, 0.0, 0.0, "quaternary"),
        _make_strategy("delta", 0.7, 0.7, 0.7, 0.7, 0.0, 0.3, "quaternary"),
        _make_strategy("epsilon", 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, "ternary"),
    ]


# ---------------------------------------------------------------------------
# Embedding tests
# ---------------------------------------------------------------------------


class TestStrategyEmbedding:
    def test_empty(self):
        assert embed_strategies_2d([]) == []

    def test_basic_embedding(self):
        strats = _sample_strategies()
        result = embed_strategies_2d(strats)

        assert len(result) == len(strats)
        for e in result:
            assert "name" in e
            assert "x" in e
            assert "y" in e
            assert isinstance(e["x"], float)
            assert isinstance(e["y"], float)

    def test_deterministic(self):
        strats = _sample_strategies()
        r1 = embed_strategies_2d(strats)
        r2 = embed_strategies_2d(strats)
        assert r1 == r2

    def test_sorted_by_name(self):
        strats = _sample_strategies()
        result = embed_strategies_2d(strats)
        names = [e["name"] for e in result]
        assert names == sorted(names)

    def test_no_mutation(self):
        strats = _sample_strategies()
        original = copy.deepcopy(strats)
        embed_strategies_2d(strats)
        assert strats == original

    def test_coordinate_values(self):
        s = _make_strategy("test", design_score=0.8, consistency_gap=0.1,
                           confidence_efficiency=0.6, revival_strength=0.2)
        result = embed_strategies_2d([s])
        assert len(result) == 1
        assert result[0]["x"] == round(0.8 - 0.1, 12)  # 0.7
        assert result[0]["y"] == round(0.6 + 0.2, 12)  # 0.8

    def test_rounding(self):
        strats = _sample_strategies()
        result = embed_strategies_2d(strats)
        for e in result:
            # Check 12-decimal rounding by verifying string representation
            x_str = f"{e['x']:.12f}"
            assert float(x_str) == e["x"]


# ---------------------------------------------------------------------------
# Pareto front tests
# ---------------------------------------------------------------------------


class TestParetoFront:
    def test_empty(self):
        assert compute_pareto_front([]) == []

    def test_single_strategy(self):
        s = _make_strategy("only")
        result = compute_pareto_front([s])
        assert len(result) == 1
        assert result[0]["name"] == "only"

    def test_dominated_removed(self):
        # alpha dominates beta on all metrics
        strats = [
            _make_strategy("alpha", 0.9, 0.9, 0.9, 0.9),
            _make_strategy("beta", 0.5, 0.5, 0.5, 0.5),
        ]
        result = compute_pareto_front(strats)
        assert len(result) == 1
        assert result[0]["name"] == "alpha"

    def test_non_dominated_kept(self):
        # Neither dominates the other
        strats = [
            _make_strategy("alpha", 0.9, 0.3, 0.5, 0.5),
            _make_strategy("beta", 0.3, 0.9, 0.5, 0.5),
        ]
        result = compute_pareto_front(strats)
        assert len(result) == 2

    def test_sorted_by_design_score_desc_then_name(self):
        strats = [
            _make_strategy("beta", 0.9, 0.3, 0.5, 0.5),
            _make_strategy("alpha", 0.3, 0.9, 0.5, 0.5),
        ]
        result = compute_pareto_front(strats)
        assert result[0]["name"] == "beta"   # higher design_score
        assert result[1]["name"] == "alpha"

    def test_deterministic(self):
        strats = _sample_strategies()
        r1 = compute_pareto_front(strats)
        r2 = compute_pareto_front(strats)
        assert r1 == r2

    def test_no_mutation(self):
        strats = _sample_strategies()
        original = copy.deepcopy(strats)
        compute_pareto_front(strats)
        assert strats == original


# ---------------------------------------------------------------------------
# Clustering tests
# ---------------------------------------------------------------------------


class TestStrategyClustering:
    def test_empty(self):
        result = cluster_strategies([])
        assert result == {"clusters": []}

    def test_single_strategy(self):
        result = cluster_strategies([_make_strategy("only")])
        assert len(result["clusters"]) == 1
        assert result["clusters"][0]["representative"] == "only"
        assert result["clusters"][0]["size"] == 1

    def test_identical_strategies_cluster(self):
        strats = [
            _make_strategy("a", 0.5, 0.5, 0.5, 0.5),
            _make_strategy("b", 0.5, 0.5, 0.5, 0.5),
        ]
        result = cluster_strategies(strats, threshold=0.9)
        assert len(result["clusters"]) == 1
        assert result["clusters"][0]["size"] == 2
        assert result["clusters"][0]["representative"] == "a"

    def test_different_strategies_separate(self):
        strats = [
            _make_strategy("a", 0.9, 0.9, 0.9, 0.9),
            _make_strategy("b", 0.1, 0.1, 0.1, 0.1),
        ]
        result = cluster_strategies(strats, threshold=0.9)
        assert len(result["clusters"]) == 2

    def test_deterministic(self):
        strats = _sample_strategies()
        r1 = cluster_strategies(strats)
        r2 = cluster_strategies(strats)
        assert r1 == r2

    def test_no_mutation(self):
        strats = _sample_strategies()
        original = copy.deepcopy(strats)
        cluster_strategies(strats)
        assert strats == original

    def test_cluster_structure(self):
        result = cluster_strategies(_sample_strategies())
        for c in result["clusters"]:
            assert "representative" in c
            assert "members" in c
            assert "size" in c
            assert c["size"] == len(c["members"])
            assert c["representative"] in c["members"]

    def test_all_strategies_assigned(self):
        strats = _sample_strategies()
        result = cluster_strategies(strats)
        all_members = []
        for c in result["clusters"]:
            all_members.extend(c["members"])
        names = sorted(s["name"] for s in strats)
        assert sorted(all_members) == names


# ---------------------------------------------------------------------------
# Representation comparison tests
# ---------------------------------------------------------------------------


class TestRepresentationComparison:
    def test_empty(self):
        result = compare_representations([])
        assert result["ternary"]["count"] == 0
        assert result["quaternary"]["count"] == 0

    def test_split_correct(self):
        strats = _sample_strategies()
        result = compare_representations(strats)
        # 3 ternary, 2 quaternary
        assert result["ternary"]["count"] == 3
        assert result["quaternary"]["count"] == 2

    def test_avg_design_score(self):
        strats = [
            _make_strategy("a", design_score=0.6, state_system="ternary"),
            _make_strategy("b", design_score=0.4, state_system="ternary"),
        ]
        result = compare_representations(strats)
        assert result["ternary"]["avg_design_score"] == round(0.5, 12)

    def test_best_strategy(self):
        strats = [
            _make_strategy("low", design_score=0.3, state_system="quaternary"),
            _make_strategy("high", design_score=0.9, state_system="quaternary"),
        ]
        result = compare_representations(strats)
        assert result["quaternary"]["best"] == "high"

    def test_deterministic(self):
        strats = _sample_strategies()
        r1 = compare_representations(strats)
        r2 = compare_representations(strats)
        assert r1 == r2

    def test_no_mutation(self):
        strats = _sample_strategies()
        original = copy.deepcopy(strats)
        compare_representations(strats)
        assert strats == original

    def test_missing_state_system_skipped(self):
        """Strategies without state_system are not silently assigned to ternary."""
        strats = [
            _make_strategy("a", design_score=0.5, state_system="ternary"),
            {"name": "no_system", "metrics": {"design_score": 0.9}},
        ]
        result = compare_representations(strats)
        assert result["ternary"]["count"] == 1
        assert result["quaternary"]["count"] == 0


# ---------------------------------------------------------------------------
# Strategy map visualization tests
# ---------------------------------------------------------------------------


class TestStrategyMap:
    def test_empty(self):
        result = render_strategy_map([])
        assert "no strategies" in result

    def test_basic_render(self):
        embedded = [
            {"name": "a", "x": 0.0, "y": 0.0},
            {"name": "b", "x": 1.0, "y": 1.0},
        ]
        result = render_strategy_map(embedded)
        assert "Strategy Map" in result
        assert "*" in result
        assert "o" in result

    def test_deterministic(self):
        embedded = [
            {"name": "a", "x": 0.5, "y": 0.3},
            {"name": "b", "x": 0.8, "y": 0.9},
            {"name": "c", "x": 0.1, "y": 0.7},
        ]
        r1 = render_strategy_map(embedded)
        r2 = render_strategy_map(embedded)
        assert r1 == r2

    def test_single_point(self):
        embedded = [{"name": "only", "x": 0.5, "y": 0.5}]
        result = render_strategy_map(embedded)
        assert "*" in result
        assert "only" in result

    def test_best_marked(self):
        embedded = [
            {"name": "low", "x": 0.1, "y": 0.1},
            {"name": "high", "x": 0.9, "y": 0.9},
        ]
        result = render_strategy_map(embedded)
        assert "Best: high" in result


# ---------------------------------------------------------------------------
# Consistent outputs across runs
# ---------------------------------------------------------------------------


class TestConsistentOutputs:
    def test_full_pipeline_deterministic(self):
        """End-to-end: all analysis outputs are identical across runs."""
        strats = _sample_strategies()

        for _ in range(3):
            e = embed_strategies_2d(strats)
            c = cluster_strategies(strats)
            p = compute_pareto_front(strats)
            r = compare_representations(strats)
            m = render_strategy_map(e)

        # Just verify the last run is consistent with a fresh run
        e2 = embed_strategies_2d(strats)
        c2 = cluster_strategies(strats)
        p2 = compute_pareto_front(strats)
        r2 = compare_representations(strats)
        m2 = render_strategy_map(e2)

        assert e == e2
        assert c == c2
        assert p == p2
        assert r == r2
        assert m == m2
