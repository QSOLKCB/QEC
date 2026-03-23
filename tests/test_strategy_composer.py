"""Tests for strategy_composer (v98.0.0)."""

import pytest

from qec.analysis.strategy_composer import (
    blend_numeric,
    compose_strategies,
    group_by_action_type,
    resolve_discrete,
)


# ---------------------------------------------------------------------------
# BLEND NUMERIC
# ---------------------------------------------------------------------------


class TestBlendNumeric:
    def test_equal_weights(self):
        result = blend_numeric([0.3, 0.7], [1.0, 1.0])
        assert result == 0.5

    def test_weighted(self):
        result = blend_numeric([0.2, 0.8], [3.0, 1.0])
        # (0.2*3 + 0.8*1) / 4 = 1.4/4 = 0.35
        assert result == 0.35

    def test_zero_weights_fallback(self):
        result = blend_numeric([0.4, 0.6], [0.0, 0.0])
        assert result == 0.5

    def test_single_value(self):
        result = blend_numeric([0.5], [1.0])
        assert result == 0.5

    def test_empty(self):
        assert blend_numeric([], []) == 0.0

    def test_deterministic(self):
        for _ in range(100):
            assert blend_numeric([0.3, 0.7], [1.0, 2.0]) == blend_numeric(
                [0.3, 0.7], [1.0, 2.0]
            )


# ---------------------------------------------------------------------------
# RESOLVE DISCRETE
# ---------------------------------------------------------------------------


class TestResolveDiscrete:
    def test_confidence_wins(self):
        result = resolve_discrete(
            ["sequential", "parallel"], [0.9, 0.1]
        )
        assert result == "sequential"

    def test_agreement_tiebreak(self):
        result = resolve_discrete(
            ["sequential", "sequential", "parallel"],
            [0.3, 0.3, 0.6],
        )
        assert result == "sequential"  # 0.6 total, count=2

    def test_lexicographic_tiebreak(self):
        result = resolve_discrete(["b", "a"], [0.5, 0.5])
        assert result == "a"

    def test_single(self):
        assert resolve_discrete(["hard"], [0.5]) == "hard"

    def test_empty(self):
        assert resolve_discrete([], []) == ""


# ---------------------------------------------------------------------------
# GROUP BY ACTION TYPE
# ---------------------------------------------------------------------------


class TestGroupByActionType:
    def test_single_group(self):
        strategies = [
            {"action_type": "adjust_damping", "params": {"alpha": 0.5}, "confidence": 0.8},
            {"action_type": "adjust_damping", "params": {"alpha": 0.3}, "confidence": 0.6},
        ]
        groups = group_by_action_type(strategies)
        assert len(groups) == 1
        assert len(groups["adjust_damping"]) == 2

    def test_multiple_groups(self):
        strategies = [
            {"action_type": "adjust_damping", "params": {"alpha": 0.5}, "confidence": 0.8},
            {"action_type": "schedule_update", "params": {"mode": "sequential"}, "confidence": 0.6},
        ]
        groups = group_by_action_type(strategies)
        assert len(groups) == 2


# ---------------------------------------------------------------------------
# COMPOSE STRATEGIES (INTEGRATION)
# ---------------------------------------------------------------------------


class TestComposeStrategies:
    def test_empty(self):
        assert compose_strategies([]) == []

    def test_single_strategy(self):
        strategies = [
            {
                "action_type": "adjust_damping",
                "params": {"alpha": 0.5},
                "confidence": 0.8,
                "law_id": "L1",
            }
        ]
        result = compose_strategies(strategies)
        assert len(result) == 1
        assert result[0]["action_type"] == "adjust_damping"
        assert result[0]["params"]["alpha"] == 0.5

    def test_numeric_blending(self):
        strategies = [
            {
                "action_type": "adjust_damping",
                "params": {"alpha": 0.3},
                "confidence": 0.8,
                "law_id": "L1",
            },
            {
                "action_type": "adjust_damping",
                "params": {"alpha": 0.7},
                "confidence": 0.8,
                "law_id": "L2",
            },
        ]
        result = compose_strategies(strategies)
        assert len(result) == 1
        # Equal weights -> average: 0.5
        assert result[0]["params"]["alpha"] == 0.5

    def test_discrete_resolution(self):
        strategies = [
            {
                "action_type": "schedule_update",
                "params": {"mode": "sequential"},
                "confidence": 0.9,
                "law_id": "L1",
            },
            {
                "action_type": "schedule_update",
                "params": {"mode": "parallel"},
                "confidence": 0.3,
                "law_id": "L2",
            },
        ]
        result = compose_strategies(strategies)
        assert len(result) == 1
        assert result[0]["params"]["mode"] == "sequential"

    def test_multiple_action_types(self):
        strategies = [
            {
                "action_type": "adjust_damping",
                "params": {"alpha": 0.5},
                "confidence": 0.8,
                "law_id": "L1",
            },
            {
                "action_type": "schedule_update",
                "params": {"mode": "parallel"},
                "confidence": 0.6,
                "law_id": "L2",
            },
        ]
        result = compose_strategies(strategies)
        assert len(result) == 2
        types = [r["action_type"] for r in result]
        assert "adjust_damping" in types
        assert "schedule_update" in types

    def test_source_ids_tracked(self):
        strategies = [
            {
                "action_type": "adjust_damping",
                "params": {"alpha": 0.3},
                "confidence": 0.8,
                "law_id": "L1",
            },
            {
                "action_type": "adjust_damping",
                "params": {"alpha": 0.7},
                "confidence": 0.6,
                "law_id": "L2",
            },
        ]
        result = compose_strategies(strategies)
        assert sorted(result[0]["source_ids"]) == ["L1", "L2"]

    def test_deterministic_output(self):
        strategies = [
            {
                "action_type": "adjust_damping",
                "params": {"alpha": 0.3},
                "confidence": 0.8,
                "law_id": "L1",
            },
            {
                "action_type": "adjust_damping",
                "params": {"alpha": 0.7},
                "confidence": 0.4,
                "law_id": "L2",
            },
            {
                "action_type": "schedule_update",
                "params": {"mode": "sequential"},
                "confidence": 0.6,
                "law_id": "L3",
            },
        ]
        baseline = compose_strategies(strategies)
        for _ in range(100):
            assert compose_strategies(strategies) == baseline

    def test_confidence_averaging(self):
        strategies = [
            {
                "action_type": "adjust_damping",
                "params": {"alpha": 0.5},
                "confidence": 0.8,
                "law_id": "L1",
            },
            {
                "action_type": "adjust_damping",
                "params": {"alpha": 0.5},
                "confidence": 0.4,
                "law_id": "L2",
            },
        ]
        result = compose_strategies(strategies)
        assert result[0]["confidence"] == 0.6  # (0.8 + 0.4) / 2
