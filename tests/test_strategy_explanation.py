"""Tests for v102.1.0 strategy explanation modules.

Verifies:
- explain_strategy returns correct components and dominant factors
- compare_strategies correctly identifies better/worse metrics
- explain_pareto identifies strengths for Pareto front members
- all outputs are deterministic
- no mutation of inputs
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

import pytest

from qec.analysis.strategy_explanation import compare_strategies, explain_strategy
from qec.analysis.pareto_explanation import explain_pareto


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


# ---------------------------------------------------------------------------
# explain_strategy tests
# ---------------------------------------------------------------------------


class TestExplainStrategy:
    def test_basic_fields(self) -> None:
        s = _make_strategy("alpha", design_score=0.9, confidence_efficiency=0.8)
        result = explain_strategy(s)
        assert result["name"] == "alpha"
        assert isinstance(result["score"], float)
        assert "components" in result
        assert "dominant_factors" in result

    def test_components_match_metrics(self) -> None:
        s = _make_strategy(
            "beta",
            design_score=0.7,
            confidence_efficiency=0.6,
            consistency_gap=0.1,
            revival_strength=0.3,
        )
        result = explain_strategy(s)
        comps = result["components"]
        assert comps["design_score"] == 0.7
        assert comps["confidence_efficiency"] == 0.6
        assert comps["consistency_gap"] == 0.1
        assert comps["revival_strength"] == 0.3

    def test_dominant_factors_sorted_by_abs_value(self) -> None:
        s = _make_strategy(
            "gamma",
            design_score=0.9,
            confidence_efficiency=0.1,
            revival_strength=0.5,
        )
        result = explain_strategy(s)
        factors = result["dominant_factors"]
        assert factors == ["design_score", "revival_strength", "confidence_efficiency"]

    def test_determinism(self) -> None:
        s = _make_strategy("delta", 0.8, 0.7, 0.6, 0.5, 0.1, 0.2)
        r1 = explain_strategy(s)
        r2 = explain_strategy(s)
        assert r1 == r2

    def test_no_mutation(self) -> None:
        s = _make_strategy("epsilon", 0.5, 0.5)
        original = copy.deepcopy(s)
        explain_strategy(s)
        assert s == original

    def test_missing_metrics(self) -> None:
        s = {"name": "empty", "metrics": {}}
        result = explain_strategy(s)
        assert result["name"] == "empty"
        assert result["components"]["design_score"] == 0.0

    def test_score_from_underscore_score(self) -> None:
        s = _make_strategy("scored", design_score=0.5)
        s["_score"] = 0.75
        result = explain_strategy(s)
        assert result["score"] == 0.75


# ---------------------------------------------------------------------------
# compare_strategies tests
# ---------------------------------------------------------------------------


class TestCompareStrategies:
    def test_a_better_on_all(self) -> None:
        a = _make_strategy("a", design_score=0.9, confidence_efficiency=0.8, consistency_gap=0.1)
        b = _make_strategy("b", design_score=0.5, confidence_efficiency=0.4, consistency_gap=0.3)
        result = compare_strategies(a, b)
        assert "design_score" in result["better_on"]
        assert "confidence_efficiency" in result["better_on"]
        assert "consistency_gap" in result["better_on"]  # lower is better
        assert result["worse_on"] == []

    def test_a_worse_on_all(self) -> None:
        a = _make_strategy("a", design_score=0.3, confidence_efficiency=0.2, consistency_gap=0.5)
        b = _make_strategy("b", design_score=0.9, confidence_efficiency=0.8, consistency_gap=0.1)
        result = compare_strategies(a, b)
        assert result["better_on"] == []
        assert len(result["worse_on"]) == 3

    def test_tied_metrics(self) -> None:
        a = _make_strategy("a", design_score=0.5, confidence_efficiency=0.5, consistency_gap=0.1)
        b = _make_strategy("b", design_score=0.5, confidence_efficiency=0.5, consistency_gap=0.1)
        result = compare_strategies(a, b)
        assert result["better_on"] == []
        assert result["worse_on"] == []

    def test_mixed_comparison(self) -> None:
        a = _make_strategy("a", design_score=0.9, confidence_efficiency=0.3, consistency_gap=0.2)
        b = _make_strategy("b", design_score=0.5, confidence_efficiency=0.8, consistency_gap=0.1)
        result = compare_strategies(a, b)
        assert "design_score" in result["better_on"]
        assert "confidence_efficiency" in result["worse_on"]
        assert "consistency_gap" in result["worse_on"]  # a=0.2 > b=0.1, lower is better

    def test_consistency_gap_lower_is_better(self) -> None:
        a = _make_strategy("a", consistency_gap=0.05)
        b = _make_strategy("b", consistency_gap=0.2)
        result = compare_strategies(a, b)
        assert "consistency_gap" in result["better_on"]

    def test_determinism(self) -> None:
        a = _make_strategy("a", 0.9, 0.8)
        b = _make_strategy("b", 0.5, 0.4)
        r1 = compare_strategies(a, b)
        r2 = compare_strategies(a, b)
        assert r1 == r2

    def test_no_mutation(self) -> None:
        a = _make_strategy("a", 0.9, 0.8)
        b = _make_strategy("b", 0.5, 0.4)
        a_orig = copy.deepcopy(a)
        b_orig = copy.deepcopy(b)
        compare_strategies(a, b)
        assert a == a_orig
        assert b == b_orig


# ---------------------------------------------------------------------------
# explain_pareto tests
# ---------------------------------------------------------------------------


class TestExplainPareto:
    def test_empty(self) -> None:
        assert explain_pareto([]) == []

    def test_single_strategy(self) -> None:
        s = _make_strategy("solo", design_score=0.8, confidence_efficiency=0.7)
        result = explain_pareto([s])
        assert len(result) == 1
        assert result[0]["name"] == "solo"
        assert "design_score" in result[0]["strengths"]
        assert "confidence_efficiency" in result[0]["strengths"]

    def test_multiple_strategies_different_strengths(self) -> None:
        a = _make_strategy("a", design_score=0.9, confidence_efficiency=0.3,
                           temporal_stability=0.5, trust_modulation=0.5)
        b = _make_strategy("b", design_score=0.3, confidence_efficiency=0.9,
                           temporal_stability=0.5, trust_modulation=0.5)
        result = explain_pareto([a, b])
        assert result[0]["name"] == "a"
        assert "design_score" in result[0]["strengths"]
        assert "confidence_efficiency" not in result[0]["strengths"]
        assert result[1]["name"] == "b"
        assert "confidence_efficiency" in result[1]["strengths"]
        assert "design_score" not in result[1]["strengths"]

    def test_tied_metric_both_get_strength(self) -> None:
        a = _make_strategy("a", design_score=0.8)
        b = _make_strategy("b", design_score=0.8)
        result = explain_pareto([a, b])
        assert "design_score" in result[0]["strengths"]
        assert "design_score" in result[1]["strengths"]

    def test_zero_metric_not_strength(self) -> None:
        a = _make_strategy("a", design_score=0.0, confidence_efficiency=0.0,
                           temporal_stability=0.0, trust_modulation=0.0)
        result = explain_pareto([a])
        assert result[0]["strengths"] == []

    def test_determinism(self) -> None:
        strategies = [
            _make_strategy("a", 0.9, 0.3),
            _make_strategy("b", 0.3, 0.9),
        ]
        r1 = explain_pareto(strategies)
        r2 = explain_pareto(strategies)
        assert r1 == r2

    def test_no_mutation(self) -> None:
        strategies = [_make_strategy("a", 0.9, 0.8)]
        original = copy.deepcopy(strategies)
        explain_pareto(strategies)
        assert strategies == original

    def test_preserves_input_order(self) -> None:
        strategies = [
            _make_strategy("z", 0.5, 0.5),
            _make_strategy("a", 0.9, 0.3),
            _make_strategy("m", 0.3, 0.9),
        ]
        result = explain_pareto(strategies)
        assert [r["name"] for r in result] == ["z", "a", "m"]
