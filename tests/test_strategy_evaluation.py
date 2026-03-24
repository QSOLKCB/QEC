"""Tests for strategy_evaluation module (v98.9.0).

Covers improvement detection, degradation, neutral cases, outcome
classification, score bounds, determinism, and no-mutation guarantees.
"""

from __future__ import annotations

import copy
from typing import Any, Dict

import pytest

from qec.analysis.strategy_evaluation import (
    classify_outcome,
    compute_state_delta,
    evaluate_improvement,
    evaluate_strategy,
    extract_eval_state,
    update_history,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metrics(
    regime: str = "stable",
    basin_score: float = 0.5,
    phi: float = 0.5,
    consistency: float = 0.5,
    divergence: float = 0.1,
    abs_curvature: float = 0.1,
    resonance: float = 0.1,
    complexity: float = 0.1,
) -> Dict[str, Any]:
    """Build a synthetic full-metrics dict."""
    return {
        "attractor": {
            "regime": regime,
            "basin_score": basin_score,
        },
        "field": {
            "phi_alignment": phi,
            "curvature": {"abs_curvature": abs_curvature},
            "resonance": resonance,
            "complexity": complexity,
        },
        "multiscale": {
            "scale_consistency": consistency,
            "scale_divergence": divergence,
        },
    }


def _make_state(
    regime: str = "stable",
    basin_score: float = 0.5,
    phi: float = 0.5,
    consistency: float = 0.5,
    divergence: float = 0.1,
    curvature: float = 0.1,
    resonance: float = 0.1,
    complexity: float = 0.1,
) -> Dict[str, Any]:
    return {
        "regime": regime,
        "basin_score": basin_score,
        "phi": phi,
        "consistency": consistency,
        "divergence": divergence,
        "curvature": curvature,
        "resonance": resonance,
        "complexity": complexity,
    }


# ---------------------------------------------------------------------------
# 1. extract_eval_state
# ---------------------------------------------------------------------------

class TestExtractEvalState:
    def test_returns_expected_keys(self):
        m = _make_metrics()
        state = extract_eval_state(m)
        expected_keys = {
            "regime", "basin_score", "phi", "consistency",
            "divergence", "curvature", "resonance", "complexity",
        }
        assert set(state.keys()) == expected_keys

    def test_values_match_input(self):
        m = _make_metrics(
            regime="unstable", basin_score=0.3, phi=0.7,
            consistency=0.8, divergence=0.4, abs_curvature=0.6,
            resonance=0.2, complexity=0.9,
        )
        state = extract_eval_state(m)
        assert state["regime"] == "unstable"
        assert state["basin_score"] == pytest.approx(0.3)
        assert state["phi"] == pytest.approx(0.7)
        assert state["curvature"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# 2. compute_state_delta
# ---------------------------------------------------------------------------

class TestComputeStateDelta:
    def test_all_deltas_present(self):
        prev = _make_state(basin_score=0.3)
        curr = _make_state(basin_score=0.7)
        delta = compute_state_delta(prev, curr)
        assert "basin_score_delta" in delta
        assert "phi_delta" in delta
        assert "consistency_delta" in delta
        assert "curvature_delta" in delta
        assert "divergence_delta" in delta
        assert "resonance_delta" in delta
        assert "complexity_delta" in delta

    def test_delta_values(self):
        prev = _make_state(basin_score=0.3, curvature=0.5)
        curr = _make_state(basin_score=0.7, curvature=0.2)
        delta = compute_state_delta(prev, curr)
        assert delta["basin_score_delta"] == pytest.approx(0.4)
        assert delta["curvature_delta"] == pytest.approx(-0.3)

    def test_zero_delta_for_identical_states(self):
        s = _make_state()
        delta = compute_state_delta(s, s)
        for v in delta.values():
            assert v == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 3. evaluate_improvement — improvement detected
# ---------------------------------------------------------------------------

class TestEvaluateImprovement:
    def test_improvement_detected(self):
        prev = _make_state(basin_score=0.3, consistency=0.4, curvature=0.5, divergence=0.3)
        curr = _make_state(basin_score=0.8, consistency=0.8, curvature=0.1, divergence=0.1)
        result = evaluate_improvement(prev, curr)
        assert result["improved"] is True
        assert result["score"] > 0.0
        assert result["direction"] == "improved"

    def test_degradation_detected(self):
        prev = _make_state(basin_score=0.8, consistency=0.8, curvature=0.1, divergence=0.1)
        curr = _make_state(basin_score=0.3, consistency=0.4, curvature=0.5, divergence=0.3)
        result = evaluate_improvement(prev, curr)
        assert result["improved"] is False
        assert result["score"] < 0.0
        assert result["direction"] == "degraded"

    def test_neutral_case(self):
        s = _make_state()
        result = evaluate_improvement(s, s)
        assert result["direction"] == "neutral"
        assert result["improved"] is False
        assert result["score"] == pytest.approx(0.0)

    def test_score_clamped_upper(self):
        prev = _make_state(basin_score=0.0, consistency=0.0, curvature=1.0, divergence=1.0)
        curr = _make_state(basin_score=1.0, consistency=1.0, curvature=0.0, divergence=0.0)
        result = evaluate_improvement(prev, curr)
        assert result["score"] <= 1.0

    def test_score_clamped_lower(self):
        prev = _make_state(basin_score=1.0, consistency=1.0, curvature=0.0, divergence=0.0)
        curr = _make_state(basin_score=0.0, consistency=0.0, curvature=1.0, divergence=1.0)
        result = evaluate_improvement(prev, curr)
        assert result["score"] >= -1.0


# ---------------------------------------------------------------------------
# 4. classify_outcome
# ---------------------------------------------------------------------------

class TestClassifyOutcome:
    def test_recovered(self):
        prev = _make_state(regime="unstable")
        curr = _make_state(regime="stable")
        assert classify_outcome(prev, curr) == "recovered"

    def test_damped(self):
        prev = _make_state(regime="oscillatory")
        curr = _make_state(regime="stable")
        assert classify_outcome(prev, curr) == "damped"

    def test_stabilized(self):
        prev = _make_state(regime="stable", basin_score=0.4)
        curr = _make_state(regime="stable", basin_score=0.8)
        assert classify_outcome(prev, curr) == "stabilized"

    def test_regressed(self):
        prev = _make_state(regime="stable")
        curr = _make_state(regime="unstable")
        assert classify_outcome(prev, curr) == "regressed"

    def test_neutral_same_regime(self):
        prev = _make_state(regime="stable", basin_score=0.5)
        curr = _make_state(regime="stable", basin_score=0.5)
        assert classify_outcome(prev, curr) == "neutral"

    def test_neutral_mixed_to_transitional(self):
        prev = _make_state(regime="mixed")
        curr = _make_state(regime="transitional")
        assert classify_outcome(prev, curr) == "neutral"


# ---------------------------------------------------------------------------
# 5. update_history
# ---------------------------------------------------------------------------

class TestUpdateHistory:
    def test_appends_evaluation(self):
        history = [{"score": 0.1}]
        evaluation = {"score": 0.5}
        updated = update_history(history, evaluation)
        assert len(updated) == 2
        assert updated[-1]["score"] == 0.5

    def test_caps_at_max(self):
        history = [{"score": float(i)} for i in range(20)]
        evaluation = {"score": 99.0}
        updated = update_history(history, evaluation)
        assert len(updated) == 20
        assert updated[-1]["score"] == 99.0
        assert updated[0]["score"] == 1.0  # oldest dropped

    def test_does_not_mutate_input(self):
        history = [{"score": 0.1}]
        original = list(history)
        update_history(history, {"score": 0.5})
        assert history == original


# ---------------------------------------------------------------------------
# 6. evaluate_strategy (master)
# ---------------------------------------------------------------------------

class TestEvaluateStrategy:
    def test_basic_structure(self):
        prev = _make_metrics(regime="unstable", basin_score=0.2)
        curr = _make_metrics(regime="stable", basin_score=0.8)
        result = evaluate_strategy(prev, curr)
        assert "delta" in result
        assert "evaluation" in result
        assert "outcome" in result
        assert "history" not in result

    def test_with_history(self):
        prev = _make_metrics()
        curr = _make_metrics(basin_score=0.7)
        result = evaluate_strategy(prev, curr, history=[])
        assert "history" in result
        assert len(result["history"]) == 1

    def test_outcome_matches_classify(self):
        prev = _make_metrics(regime="unstable", basin_score=0.2)
        curr = _make_metrics(regime="stable", basin_score=0.8)
        result = evaluate_strategy(prev, curr)
        assert result["outcome"] == "recovered"

    def test_evaluation_matches_improvement(self):
        prev = _make_metrics(basin_score=0.3, consistency=0.4)
        curr = _make_metrics(basin_score=0.8, consistency=0.8)
        result = evaluate_strategy(prev, curr)
        assert result["evaluation"]["improved"] is True


# ---------------------------------------------------------------------------
# 7. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_repeated_calls_identical(self):
        prev = _make_metrics(regime="oscillatory", basin_score=0.3, phi=0.4)
        curr = _make_metrics(regime="stable", basin_score=0.7, phi=0.8)
        r1 = evaluate_strategy(prev, curr, history=[])
        r2 = evaluate_strategy(prev, curr, history=[])
        assert r1["delta"] == r2["delta"]
        assert r1["evaluation"] == r2["evaluation"]
        assert r1["outcome"] == r2["outcome"]

    def test_no_mutation_of_inputs(self):
        prev = _make_metrics(regime="unstable", basin_score=0.2)
        curr = _make_metrics(regime="stable", basin_score=0.8)
        prev_copy = copy.deepcopy(prev)
        curr_copy = copy.deepcopy(curr)
        evaluate_strategy(prev, curr)
        assert prev == prev_copy
        assert curr == curr_copy
