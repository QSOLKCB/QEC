"""Tests for basin-aware strategy transition layer."""

from __future__ import annotations

import copy
from typing import Any, Dict

import pytest

from qec.analysis.strategy_transition import (
    compute_transition,
    extract_state,
    score_strategy,
    select_next_strategy,
    select_strategy,
    should_transition,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_metrics(
    regime: str = "stable",
    basin_score: float = 0.8,
    phi: float = 0.9,
    consistency: float = 0.85,
    divergence: float = 0.1,
    curvature: float = 0.1,
    curvature_var: float = 0.05,
    resonance: float = 0.1,
    complexity: float = 0.1,
) -> Dict[str, Any]:
    """Build a minimal metrics dict matching the expected schema."""
    return {
        "attractor": {
            "regime": regime,
            "basin_score": basin_score,
        },
        "field": {
            "phi_alignment": phi,
            "curvature": {
                "abs_curvature": curvature,
                "curvature_variation": curvature_var,
            },
            "resonance": resonance,
            "complexity": complexity,
        },
        "multiscale": {
            "scale_consistency": consistency,
            "scale_divergence": divergence,
        },
    }


def _make_strategies() -> Dict[str, Dict[str, Any]]:
    """Build a deterministic set of test strategies."""
    return {
        "s_damp": {
            "action_type": "damping",
            "params": {"alpha": 0.1},
            "confidence": 0.9,
        },
        "s_scale": {
            "action_type": "scaling",
            "params": {"alpha": 0.5, "beta": 0.3},
            "confidence": 0.7,
        },
        "s_rotate": {
            "action_type": "rotation",
            "params": {"theta": 1.0},
            "confidence": 0.6,
        },
        "s_reweight": {
            "action_type": "reweight",
            "params": {"w": 0.5},
            "confidence": 0.8,
        },
    }


# ---------------------------------------------------------------------------
# extract_state
# ---------------------------------------------------------------------------

class TestExtractState:
    def test_extracts_all_fields(self):
        metrics = _make_metrics()
        state = extract_state(metrics)
        assert state["regime"] == "stable"
        assert state["basin_score"] == pytest.approx(0.8)
        assert state["phi"] == pytest.approx(0.9)
        assert state["consistency"] == pytest.approx(0.85)
        assert state["divergence"] == pytest.approx(0.1)
        assert state["curvature"] == pytest.approx(0.1)
        assert state["resonance"] == pytest.approx(0.1)
        assert state["complexity"] == pytest.approx(0.1)

    def test_defaults_on_missing(self):
        state = extract_state({})
        assert state["regime"] == "mixed"
        assert state["basin_score"] == 0.0

    def test_no_mutation(self):
        metrics = _make_metrics()
        original = copy.deepcopy(metrics)
        extract_state(metrics)
        assert metrics == original


# ---------------------------------------------------------------------------
# score_strategy
# ---------------------------------------------------------------------------

class TestScoreStrategy:
    def test_unstable_prefers_damping(self):
        state = extract_state(_make_metrics(regime="unstable", curvature=0.8))
        strategies = _make_strategies()
        damp_score = score_strategy(state, strategies["s_damp"])
        rotate_score = score_strategy(state, strategies["s_rotate"])
        assert damp_score > rotate_score

    def test_oscillatory_prefers_damping(self):
        state = extract_state(
            _make_metrics(regime="oscillatory", resonance=0.8)
        )
        strategies = _make_strategies()
        damp_score = score_strategy(state, strategies["s_damp"])
        scale_score = score_strategy(state, strategies["s_scale"])
        assert damp_score > scale_score

    def test_transitional_prefers_reweight(self):
        state = extract_state(
            _make_metrics(regime="transitional", divergence=0.6)
        )
        strategies = _make_strategies()
        reweight_score = score_strategy(state, strategies["s_reweight"])
        damp_score = score_strategy(state, strategies["s_damp"])
        assert reweight_score > damp_score

    def test_stable_gives_low_scores(self):
        state = extract_state(_make_metrics(regime="stable"))
        strategies = _make_strategies()
        for s in strategies.values():
            assert score_strategy(state, s) <= 0.2

    def test_score_bounded_zero_one(self):
        for regime in ("stable", "unstable", "oscillatory", "transitional", "mixed"):
            state = extract_state(_make_metrics(regime=regime))
            for s in _make_strategies().values():
                sc = score_strategy(state, s)
                assert 0.0 <= sc <= 1.0

    def test_deterministic(self):
        state = extract_state(_make_metrics(regime="unstable"))
        s = _make_strategies()["s_damp"]
        scores = [score_strategy(state, s) for _ in range(10)]
        assert len(set(scores)) == 1


# ---------------------------------------------------------------------------
# select_strategy
# ---------------------------------------------------------------------------

class TestSelectStrategy:
    def test_unstable_selects_damping(self):
        state = extract_state(_make_metrics(regime="unstable"))
        result = select_strategy(state, _make_strategies())
        assert result["id"] == "s_damp"

    def test_transitional_selects_reweight(self):
        state = extract_state(
            _make_metrics(regime="transitional", divergence=0.6)
        )
        result = select_strategy(state, _make_strategies())
        assert result["id"] == "s_reweight"

    def test_stable_selects_minimal(self):
        state = extract_state(_make_metrics(regime="stable"))
        result = select_strategy(state, _make_strategies())
        # all scores are low; tie-break by confidence then simplicity then id
        assert result["score"] <= 0.2

    def test_empty_strategies(self):
        state = extract_state(_make_metrics())
        result = select_strategy(state, {})
        assert result["id"] == ""
        assert result["strategy"] is None

    def test_tiebreak_confidence(self):
        """When two strategies have same score, higher confidence wins."""
        state = extract_state(_make_metrics(regime="mixed"))
        strats = {
            "a": {"action_type": "damping", "params": {"x": 1}, "confidence": 0.9},
            "b": {"action_type": "damping", "params": {"x": 1}, "confidence": 0.5},
        }
        result = select_strategy(state, strats)
        assert result["id"] == "a"

    def test_tiebreak_fewer_actions(self):
        """When score and confidence tie, fewer params wins."""
        state = extract_state(_make_metrics(regime="mixed"))
        strats = {
            "a": {"action_type": "damping", "params": {"x": 1, "y": 2}, "confidence": 0.5},
            "b": {"action_type": "damping", "params": {"x": 1}, "confidence": 0.5},
        }
        result = select_strategy(state, strats)
        assert result["id"] == "b"

    def test_tiebreak_lexicographic(self):
        """When everything ties, lexicographic id wins."""
        state = extract_state(_make_metrics(regime="mixed"))
        strats = {
            "b_strat": {"action_type": "damping", "params": {"x": 1}, "confidence": 0.5},
            "a_strat": {"action_type": "damping", "params": {"x": 1}, "confidence": 0.5},
        }
        result = select_strategy(state, strats)
        assert result["id"] == "a_strat"

    def test_deterministic_selection(self):
        state = extract_state(_make_metrics(regime="unstable"))
        strats = _make_strategies()
        results = [select_strategy(state, strats)["id"] for _ in range(20)]
        assert len(set(results)) == 1


# ---------------------------------------------------------------------------
# should_transition
# ---------------------------------------------------------------------------

class TestShouldTransition:
    def test_regime_change_triggers(self):
        prev = {"regime": "stable", "basin_score": 0.8}
        curr = {"regime": "unstable", "basin_score": 0.8}
        assert should_transition(prev, curr) is True

    def test_basin_decrease_triggers(self):
        prev = {"regime": "stable", "basin_score": 0.8}
        curr = {"regime": "stable", "basin_score": 0.5}
        assert should_transition(prev, curr) is True

    def test_no_change_no_transition(self):
        prev = {"regime": "stable", "basin_score": 0.8}
        curr = {"regime": "stable", "basin_score": 0.8}
        assert should_transition(prev, curr) is False

    def test_basin_increase_no_transition(self):
        prev = {"regime": "stable", "basin_score": 0.5}
        curr = {"regime": "stable", "basin_score": 0.9}
        assert should_transition(prev, curr) is False


# ---------------------------------------------------------------------------
# compute_transition
# ---------------------------------------------------------------------------

class TestComputeTransition:
    def test_basic_transition(self):
        prev = {"id": "s_damp", "strategy": None, "score": 0.9}
        new = {"id": "s_scale", "strategy": None, "score": 0.7}
        result = compute_transition(prev, new)
        assert result["from"] == "s_damp"
        assert result["to"] == "s_scale"
        assert isinstance(result["change"], str)

    def test_unstable_to_stable_label(self):
        prev_state = {"regime": "unstable"}
        curr_state = {"regime": "stable"}
        result = compute_transition(
            {"id": "a"}, {"id": "b"}, prev_state, curr_state,
        )
        assert result["change"] == "increase_stability"

    def test_oscillatory_to_stable_label(self):
        result = compute_transition(
            {"id": "a"}, {"id": "b"},
            {"regime": "oscillatory"}, {"regime": "stable"},
        )
        assert result["change"] == "reduce_oscillation"


# ---------------------------------------------------------------------------
# select_next_strategy (master)
# ---------------------------------------------------------------------------

class TestSelectNextStrategy:
    def test_basic_call(self):
        metrics = _make_metrics(regime="unstable")
        strats = _make_strategies()
        result = select_next_strategy(metrics, strats)
        assert "strategy" in result
        assert "state" in result
        assert result["transition"] is None

    def test_with_transition(self):
        prev_metrics = _make_metrics(regime="stable", basin_score=0.9)
        curr_metrics = _make_metrics(regime="unstable", basin_score=0.3)
        strats = _make_strategies()

        prev_result = select_next_strategy(prev_metrics, strats)

        curr_result = select_next_strategy(
            curr_metrics, strats,
            prev_strategy=prev_result["strategy"],
            prev_state=prev_result["state"],
        )
        assert curr_result["transition"] is not None
        assert curr_result["transition"]["change"] == "reduce_instability"

    def test_no_transition_when_stable(self):
        strats = _make_strategies()
        m1 = _make_metrics(regime="stable", basin_score=0.8)
        m2 = _make_metrics(regime="stable", basin_score=0.85)

        r1 = select_next_strategy(m1, strats)
        r2 = select_next_strategy(
            m2, strats,
            prev_strategy=r1["strategy"],
            prev_state=r1["state"],
        )
        assert r2["transition"] is None

    def test_no_mutation_of_inputs(self):
        metrics = _make_metrics(regime="unstable")
        strats = _make_strategies()
        m_copy = copy.deepcopy(metrics)
        s_copy = copy.deepcopy(strats)
        select_next_strategy(metrics, strats)
        assert metrics == m_copy
        assert strats == s_copy

    def test_deterministic_output(self):
        metrics = _make_metrics(regime="oscillatory", resonance=0.7)
        strats = _make_strategies()
        results = [
            select_next_strategy(metrics, strats)["strategy"]["id"]
            for _ in range(10)
        ]
        assert len(set(results)) == 1
