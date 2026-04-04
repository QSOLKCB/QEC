"""Tests for the deterministic adaptation layer (v99.0.0).

Covers:
- history scoring
- adaptation bias
- regime weights
- adaptive scoring
- trajectory scoring
- master adaptive selection
- determinism guarantees
- no-mutation guarantees
"""

from __future__ import annotations

import pytest
import copy
from typing import Any, Dict, List

from qec.analysis.strategy_adaptation import (
    REGIME_EVAL_WEIGHTS,
    compute_adaptation_bias,
    compute_strategy_history_score,
    compute_trajectory_score,
    get_regime_weights,
    score_strategy_adaptive,
    select_strategy_adaptive,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_eval(score: float, direction: str = "improved") -> Dict[str, Any]:
    """Build a minimal evaluation entry."""
    return {"score": score, "direction": direction}


def _make_history(
    scores: List[float],
    directions: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """Build a history list from score values."""
    if directions is None:
        directions = [
            "improved" if s > 0.01 else ("degraded" if s < -0.01 else "neutral")
            for s in scores
        ]
    return [_make_eval(s, d) for s, d in zip(scores, directions)]


def _make_state(regime: str = "mixed", **kwargs: float) -> Dict[str, Any]:
    """Build a flat state dict."""
    state = {
        "regime": regime,
        "basin_score": 0.5,
        "phi": 0.5,
        "consistency": 0.5,
        "divergence": 0.1,
        "curvature": 0.1,
        "resonance": 0.1,
        "complexity": 0.1,
    }
    state.update(kwargs)
    return state


def _make_strategy(
    action_type: str = "damping",
    confidence: float = 0.0,
    **params: Any,
) -> Dict[str, Any]:
    """Build a minimal strategy dict."""
    return {
        "action_type": action_type,
        "params": params if params else {"alpha": 0.1},
        "confidence": confidence,
    }


def _make_strategies() -> Dict[str, Any]:
    """Build a small deterministic strategy set."""
    return {
        "s1": _make_strategy("damping", 0.5, alpha=0.1),
        "s2": _make_strategy("scaling", 0.3, beta=0.2),
        "s3": _make_strategy("rotation", 0.1, theta=1.0, phi=0.0),
    }


# ===================================================================
# 1. History Scoring
# ===================================================================

class TestComputeStrategyHistoryScore:
    """Tests for compute_strategy_history_score."""

    def test_empty_history(self) -> None:
        result = compute_strategy_history_score([])
        assert result["avg_score"] == 0.0
        assert result["improvement_rate"] == 0.0
        assert result["stability"] == 1.0

    def test_all_positive(self) -> None:
        history = _make_history([0.5, 0.3, 0.4])
        result = compute_strategy_history_score(history)
        assert result["avg_score"] == pytest.approx((0.5 + 0.3 + 0.4) / 3)
        assert result["improvement_rate"] == 1.0
        assert result["stability"] == 0.0  # all |score| >= 0.1

    def test_all_negative(self) -> None:
        history = _make_history([-0.5, -0.3, -0.4])
        result = compute_strategy_history_score(history)
        assert result["avg_score"] < 0.0
        assert result["improvement_rate"] == 0.0

    def test_mixed_history(self) -> None:
        history = _make_history([0.5, -0.3, 0.05])
        result = compute_strategy_history_score(history)
        # improvement: first and third (0.05 > 0.01 → "improved")
        assert result["improvement_rate"] == 2.0 / 3.0
        # stability: only last entry (|0.05| < 0.1)
        assert result["stability"] == 1.0 / 3.0

    def test_all_stable(self) -> None:
        history = _make_history([0.01, -0.02, 0.05])
        result = compute_strategy_history_score(history)
        assert result["stability"] == 1.0  # all |score| < 0.1


# ===================================================================
# 2. Adaptation Bias
# ===================================================================

class TestComputeAdaptationBias:
    """Tests for compute_adaptation_bias."""

    def test_empty_history_zero_bias(self) -> None:
        assert compute_adaptation_bias([]) == 0.0

    def test_positive_history_positive_bias(self) -> None:
        history = _make_history([0.5, 0.6, 0.7])
        bias = compute_adaptation_bias(history)
        assert bias > 0.0

    def test_negative_history_negative_bias(self) -> None:
        history = _make_history([-0.5, -0.6, -0.7])
        bias = compute_adaptation_bias(history)
        assert bias < 0.0

    def test_bias_clamped_upper(self) -> None:
        # Extreme positive history
        history = _make_history([1.0] * 20)
        bias = compute_adaptation_bias(history)
        assert bias <= 0.2

    def test_bias_clamped_lower(self) -> None:
        # Extreme negative history
        history = _make_history([-1.0] * 20)
        bias = compute_adaptation_bias(history)
        assert bias >= -0.2

    def test_bias_increases_with_improvement(self) -> None:
        h1 = _make_history([0.1, 0.1])
        h2 = _make_history([0.5, 0.5])
        assert compute_adaptation_bias(h2) > compute_adaptation_bias(h1)

    def test_bias_decreases_with_degradation(self) -> None:
        h1 = _make_history([-0.1, -0.1])
        h2 = _make_history([-0.5, -0.5])
        assert compute_adaptation_bias(h2) < compute_adaptation_bias(h1)


# ===================================================================
# 3. Regime Weights
# ===================================================================

class TestGetRegimeWeights:
    """Tests for get_regime_weights."""

    def test_known_regimes(self) -> None:
        for regime in ("unstable", "oscillatory", "stable"):
            weights = get_regime_weights(regime)
            assert isinstance(weights, dict)
            assert len(weights) > 0
            assert weights == REGIME_EVAL_WEIGHTS[regime]

    def test_unknown_regime_empty(self) -> None:
        assert get_regime_weights("unknown_regime") == {}

    def test_returns_copy(self) -> None:
        w1 = get_regime_weights("unstable")
        w2 = get_regime_weights("unstable")
        assert w1 is not w2  # different dict objects
        w1["extra"] = 999
        assert "extra" not in get_regime_weights("unstable")


# ===================================================================
# 4. Adaptive Strategy Scoring
# ===================================================================

class TestScoreStrategyAdaptive:
    """Tests for score_strategy_adaptive."""

    def test_no_history_equals_base(self) -> None:
        state = _make_state("unstable")
        strategy = _make_strategy("damping")
        result = score_strategy_adaptive(strategy, state, [])
        assert result["bias"] == 0.0
        assert result["score"] == result["base_score"]

    def test_positive_history_increases_score(self) -> None:
        state = _make_state("unstable")
        strategy = _make_strategy("damping")
        history = _make_history([0.5, 0.6, 0.7])
        result = score_strategy_adaptive(strategy, state, history)
        assert result["score"] > result["base_score"]
        assert result["bias"] > 0.0

    def test_negative_history_decreases_score(self) -> None:
        state = _make_state("unstable")
        strategy = _make_strategy("damping")
        history = _make_history([-0.5, -0.6, -0.7])
        result = score_strategy_adaptive(strategy, state, history)
        assert result["score"] < result["base_score"]
        assert result["bias"] < 0.0

    def test_score_clamped_to_unit(self) -> None:
        state = _make_state("unstable")
        strategy = _make_strategy("damping")
        history = _make_history([1.0] * 20)
        result = score_strategy_adaptive(strategy, state, history)
        assert 0.0 <= result["score"] <= 1.0

    def test_deterministic(self) -> None:
        state = _make_state("oscillatory")
        strategy = _make_strategy("rotation")
        history = _make_history([0.3, -0.1, 0.2])
        results = [
            score_strategy_adaptive(strategy, state, history)
            for _ in range(50)
        ]
        assert all(r == results[0] for r in results)


# ===================================================================
# 5. Trajectory Scoring
# ===================================================================

class TestComputeTrajectoryScore:
    """Tests for compute_trajectory_score."""

    def test_empty_history(self) -> None:
        assert compute_trajectory_score([]) == 0.0

    def test_single_entry(self) -> None:
        history = _make_history([0.5])
        assert compute_trajectory_score(history) == 0.5

    def test_weights_favor_recent(self) -> None:
        # [old=-1, recent=+1] with weights [1, 2]
        # weighted = (1*-1 + 2*1) / 3 = 1/3 > 0
        history = _make_history([-1.0, 1.0])
        score = compute_trajectory_score(history)
        assert score > 0.0

    def test_uniform_scores(self) -> None:
        history = _make_history([0.3, 0.3, 0.3])
        assert abs(compute_trajectory_score(history) - 0.3) < 1e-10

    def test_longer_history(self) -> None:
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        history = _make_history(scores)
        # weights = [1, 2, 3, 4, 5], total=15
        expected = (1*0.1 + 2*0.2 + 3*0.3 + 4*0.4 + 5*0.5) / 15
        assert abs(compute_trajectory_score(history) - expected) < 1e-10


# ===================================================================
# 6. Master Adaptive Selection
# ===================================================================

class TestSelectStrategyAdaptive:
    """Tests for select_strategy_adaptive."""

    def test_empty_strategies(self) -> None:
        result = select_strategy_adaptive({}, _make_state(), [])
        assert result["selected"] == ""
        assert result["strategy"] is None

    def test_selects_best(self) -> None:
        strategies = _make_strategies()
        state = _make_state("unstable")
        result = select_strategy_adaptive(strategies, state, [])
        assert result["selected"] in strategies
        assert result["score"] > 0.0

    def test_history_influences_selection(self) -> None:
        strategies = _make_strategies()
        state = _make_state("mixed")
        r_no_hist = select_strategy_adaptive(strategies, state, [])
        r_pos_hist = select_strategy_adaptive(
            strategies, state, _make_history([0.8, 0.9, 0.7]),
        )
        # Scores should differ due to bias
        assert r_pos_hist["bias"] > 0.0
        assert r_pos_hist["score"] != r_no_hist["score"]

    def test_trajectory_score_included(self) -> None:
        strategies = _make_strategies()
        state = _make_state("unstable")
        history = _make_history([0.3, 0.5])
        result = select_strategy_adaptive(strategies, state, history)
        assert result["trajectory_score"] > 0.0

    def test_deterministic_selection(self) -> None:
        strategies = _make_strategies()
        state = _make_state("oscillatory")
        history = _make_history([0.2, -0.1, 0.3])
        results = [
            select_strategy_adaptive(strategies, state, history)
            for _ in range(50)
        ]
        assert all(r["selected"] == results[0]["selected"] for r in results)
        assert all(r["score"] == results[0]["score"] for r in results)

    def test_returns_diagnostics(self) -> None:
        strategies = _make_strategies()
        state = _make_state("stable")
        history = _make_history([0.1, 0.2])
        result = select_strategy_adaptive(strategies, state, history)
        assert "selected" in result
        assert "score" in result
        assert "bias" in result
        assert "trajectory_score" in result


# ===================================================================
# 7. No-Mutation Guarantees
# ===================================================================

class TestNoMutation:
    """Verify that adaptation functions do not mutate inputs."""

    def test_history_score_no_mutation(self) -> None:
        history = _make_history([0.3, -0.2, 0.5])
        original = copy.deepcopy(history)
        compute_strategy_history_score(history)
        assert history == original

    def test_bias_no_mutation(self) -> None:
        history = _make_history([0.3, -0.2])
        original = copy.deepcopy(history)
        compute_adaptation_bias(history)
        assert history == original

    def test_adaptive_score_no_mutation(self) -> None:
        state = _make_state("unstable")
        strategy = _make_strategy("damping")
        history = _make_history([0.5])
        state_copy = copy.deepcopy(state)
        strategy_copy = copy.deepcopy(strategy)
        history_copy = copy.deepcopy(history)
        score_strategy_adaptive(strategy, state, history)
        assert state == state_copy
        assert strategy == strategy_copy
        assert history == history_copy

    def test_select_adaptive_no_mutation(self) -> None:
        strategies = _make_strategies()
        state = _make_state("mixed")
        history = _make_history([0.3, 0.1])
        strat_copy = copy.deepcopy(strategies)
        state_copy = copy.deepcopy(state)
        hist_copy = copy.deepcopy(history)
        select_strategy_adaptive(strategies, state, history)
        assert strategies == strat_copy
        assert state == state_copy
        assert history == hist_copy

    def test_trajectory_no_mutation(self) -> None:
        history = _make_history([0.1, 0.2, 0.3])
        original = copy.deepcopy(history)
        compute_trajectory_score(history)
        assert history == original


# ===================================================================
# 8. Deterministic Repeatability
# ===================================================================

class TestDeterministicRepeatability:
    """Verify all functions produce identical output across runs."""

    def test_history_score_repeatable(self) -> None:
        history = _make_history([0.3, -0.2, 0.5, 0.1])
        results = [compute_strategy_history_score(history) for _ in range(100)]
        assert all(r == results[0] for r in results)

    def test_bias_repeatable(self) -> None:
        history = _make_history([0.3, -0.2, 0.5])
        results = [compute_adaptation_bias(history) for _ in range(100)]
        assert all(r == results[0] for r in results)

    def test_trajectory_repeatable(self) -> None:
        history = _make_history([0.1, 0.2, 0.3, 0.4])
        results = [compute_trajectory_score(history) for _ in range(100)]
        assert all(r == results[0] for r in results)

    def test_adaptive_selection_repeatable(self) -> None:
        strategies = _make_strategies()
        state = _make_state("transitional")
        history = _make_history([0.2, -0.1, 0.4, 0.3])
        results = [
            select_strategy_adaptive(strategies, state, history)
            for _ in range(100)
        ]
        assert all(r["selected"] == results[0]["selected"] for r in results)
        assert all(r["score"] == results[0]["score"] for r in results)
        assert all(r["bias"] == results[0]["bias"] for r in results)


# ===================================================================
# 9. Fallback Path
# ===================================================================

class TestFallbackPath:
    """Test that existing logic is preserved when no history provided."""

    def test_no_history_zero_bias(self) -> None:
        state = _make_state("unstable")
        strategy = _make_strategy("damping")
        result = score_strategy_adaptive(strategy, state, [])
        assert result["bias"] == 0.0

    def test_no_history_selection_matches_base(self) -> None:
        """With empty history, adaptive selection should match base scoring."""
        strategies = _make_strategies()
        state = _make_state("unstable")
        result = select_strategy_adaptive(strategies, state, [])
        # bias should be zero, trajectory should be zero
        assert result["bias"] == 0.0
        assert result["trajectory_score"] == 0.0
