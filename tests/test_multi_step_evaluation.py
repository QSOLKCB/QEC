"""Tests for deterministic multi-step evaluation (v99.4.0)."""

from __future__ import annotations

import copy

from qec.analysis.multi_step_evaluation import (
    _best_follow_up_delta,
    compute_multi_step_factor,
    compute_multi_step_factors,
    compute_two_step_value,
    get_expected_transition_outcomes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transition_memory():
    """Build a transition memory with known outcomes.

    Layout:
      From (unstable, basin_2):
        s1 -> (stable, basin_4): mean_delta=0.3, success_rate=0.8, count=5
        s1 -> (transitional, basin_3): mean_delta=0.1, success_rate=0.6, count=3
        s2 -> (stable, basin_4): mean_delta=0.2, success_rate=0.7, count=4

      From (stable, basin_4):
        s1 -> (stable, basin_4): mean_delta=0.05, success_rate=0.9, count=10
        s3 -> (stable, basin_4): mean_delta=0.1, success_rate=1.0, count=2

      From (transitional, basin_3):
        s2 -> (stable, basin_4): mean_delta=0.4, success_rate=0.9, count=6
    """
    return {
        ("unstable", "basin_2", "s1", "stable", "basin_4"): {
            "count": 5, "mean_delta": 0.3, "success_rate": 0.8,
        },
        ("unstable", "basin_2", "s1", "transitional", "basin_3"): {
            "count": 3, "mean_delta": 0.1, "success_rate": 0.6,
        },
        ("unstable", "basin_2", "s2", "stable", "basin_4"): {
            "count": 4, "mean_delta": 0.2, "success_rate": 0.7,
        },
        ("stable", "basin_4", "s1", "stable", "basin_4"): {
            "count": 10, "mean_delta": 0.05, "success_rate": 0.9,
        },
        ("stable", "basin_4", "s3", "stable", "basin_4"): {
            "count": 2, "mean_delta": 0.1, "success_rate": 1.0,
        },
        ("transitional", "basin_3", "s2", "stable", "basin_4"): {
            "count": 6, "mean_delta": 0.4, "success_rate": 0.9,
        },
    }


# ---------------------------------------------------------------------------
# 1. get_expected_transition_outcomes
# ---------------------------------------------------------------------------


class TestGetExpectedTransitionOutcomes:

    def test_returns_matching_outcomes(self):
        tm = _make_transition_memory()
        outcomes = get_expected_transition_outcomes(
            "unstable", "basin_2", "s1", tm,
        )
        assert len(outcomes) == 2
        # Sorted deterministically
        states = [o[0] for o in outcomes]
        assert ("stable", "basin_4") in states
        assert ("transitional", "basin_3") in states

    def test_returns_correct_values(self):
        tm = _make_transition_memory()
        outcomes = get_expected_transition_outcomes(
            "unstable", "basin_2", "s1", tm,
        )
        outcome_dict = {o[0]: (o[1], o[2]) for o in outcomes}
        assert outcome_dict[("stable", "basin_4")] == (0.3, 0.8)
        assert outcome_dict[("transitional", "basin_3")] == (0.1, 0.6)

    def test_no_match_returns_empty(self):
        tm = _make_transition_memory()
        outcomes = get_expected_transition_outcomes(
            "unstable", "basin_2", "s99", tm,
        )
        assert outcomes == []

    def test_empty_memory_returns_empty(self):
        outcomes = get_expected_transition_outcomes(
            "unstable", "basin_2", "s1", {},
        )
        assert outcomes == []

    def test_deterministic(self):
        tm = _make_transition_memory()
        a = get_expected_transition_outcomes("unstable", "basin_2", "s1", tm)
        b = get_expected_transition_outcomes("unstable", "basin_2", "s1", tm)
        assert a == b

    def test_no_mutation(self):
        tm = _make_transition_memory()
        frozen = copy.deepcopy(tm)
        get_expected_transition_outcomes("unstable", "basin_2", "s1", tm)
        assert tm == frozen


# ---------------------------------------------------------------------------
# 2. _best_follow_up_delta
# ---------------------------------------------------------------------------


class TestBestFollowUpDelta:

    def test_finds_best_strategy(self):
        tm = _make_transition_memory()
        # From (stable, basin_4): s1 has 0.05, s3 has 0.1
        best = _best_follow_up_delta("stable", "basin_4", tm)
        assert best == 0.1

    def test_single_strategy(self):
        tm = _make_transition_memory()
        # From (transitional, basin_3): only s2 with 0.4
        best = _best_follow_up_delta("transitional", "basin_3", tm)
        assert abs(best - 0.4) < 1e-10

    def test_no_data_returns_zero(self):
        tm = _make_transition_memory()
        best = _best_follow_up_delta("unknown", "basin_0", tm)
        assert best == 0.0

    def test_empty_memory_returns_zero(self):
        assert _best_follow_up_delta("stable", "basin_4", {}) == 0.0

    def test_deterministic(self):
        tm = _make_transition_memory()
        a = _best_follow_up_delta("stable", "basin_4", tm)
        b = _best_follow_up_delta("stable", "basin_4", tm)
        assert a == b


# ---------------------------------------------------------------------------
# 3. compute_two_step_value
# ---------------------------------------------------------------------------


class TestComputeTwoStepValue:

    def test_basic_two_step(self):
        tm = _make_transition_memory()
        # s1 from (unstable, basin_2):
        #   -> (stable, basin_4): delta1=0.3, count=5
        #     best_delta2 from (stable, basin_4) = 0.1 (s3)
        #     combined = 0.3 + 0.1 = 0.4
        #   -> (transitional, basin_3): delta1=0.1, count=3
        #     best_delta2 from (transitional, basin_3) = 0.4 (s2)
        #     combined = 0.1 + 0.4 = 0.5
        # weighted_mean = (0.4*5 + 0.5*3) / (5+3) = (2.0+1.5)/8 = 0.4375
        value = compute_two_step_value("unstable", "basin_2", "s1", tm)
        assert abs(value - 0.4375) < 1e-10

    def test_single_outcome(self):
        tm = _make_transition_memory()
        # s2 from (unstable, basin_2):
        #   -> (stable, basin_4): delta1=0.2, count=4
        #     best_delta2 from (stable, basin_4) = 0.1
        #     combined = 0.2 + 0.1 = 0.3
        # weighted_mean = 0.3
        value = compute_two_step_value("unstable", "basin_2", "s2", tm)
        assert abs(value - 0.3) < 1e-10

    def test_no_data_returns_zero(self):
        tm = _make_transition_memory()
        value = compute_two_step_value("unstable", "basin_2", "s99", tm)
        assert value == 0.0

    def test_empty_memory_returns_zero(self):
        value = compute_two_step_value("unstable", "basin_2", "s1", {})
        assert value == 0.0

    def test_deterministic(self):
        tm = _make_transition_memory()
        a = compute_two_step_value("unstable", "basin_2", "s1", tm)
        b = compute_two_step_value("unstable", "basin_2", "s1", tm)
        assert a == b

    def test_no_mutation(self):
        tm = _make_transition_memory()
        frozen = copy.deepcopy(tm)
        compute_two_step_value("unstable", "basin_2", "s1", tm)
        assert tm == frozen


# ---------------------------------------------------------------------------
# 4. compute_multi_step_factor
# ---------------------------------------------------------------------------


class TestComputeMultiStepFactor:

    def test_zero_returns_one(self):
        assert compute_multi_step_factor(0.0) == 1.0

    def test_positive_value_above_one(self):
        factor = compute_multi_step_factor(0.5, max_abs_value=1.0)
        assert factor > 1.0
        assert factor <= 1.2

    def test_negative_value_below_one(self):
        factor = compute_multi_step_factor(-0.5, max_abs_value=1.0)
        assert factor < 1.0
        assert factor >= 0.8

    def test_clamped_upper(self):
        factor = compute_multi_step_factor(100.0, max_abs_value=1.0)
        assert factor == 1.2

    def test_clamped_lower(self):
        factor = compute_multi_step_factor(-100.0, max_abs_value=1.0)
        assert factor == 0.8

    def test_exact_alpha_scaling(self):
        # normalized = 1.0/1.0 = 1.0, factor = 1 + 0.2*1.0 = 1.2
        factor = compute_multi_step_factor(1.0, max_abs_value=1.0)
        assert abs(factor - 1.2) < 1e-10

    def test_normalization(self):
        # value=0.5, max_abs=2.0 -> normalized=0.25 -> factor=1.05
        factor = compute_multi_step_factor(0.5, max_abs_value=2.0)
        assert abs(factor - 1.05) < 1e-10

    def test_safe_with_zero_max_abs(self):
        # Should not divide by zero
        factor = compute_multi_step_factor(0.5, max_abs_value=0.0)
        assert 0.8 <= factor <= 1.2

    def test_deterministic(self):
        a = compute_multi_step_factor(0.3, max_abs_value=0.5)
        b = compute_multi_step_factor(0.3, max_abs_value=0.5)
        assert a == b


# ---------------------------------------------------------------------------
# 5. compute_multi_step_factors (batch)
# ---------------------------------------------------------------------------


class TestComputeMultiStepFactors:

    def test_basic_factors(self):
        tm = _make_transition_memory()
        factors = compute_multi_step_factors(
            "unstable", "basin_2", ["s1", "s2"], tm,
        )
        assert "s1" in factors
        assert "s2" in factors
        assert 0.8 <= factors["s1"] <= 1.2
        assert 0.8 <= factors["s2"] <= 1.2

    def test_relative_ordering(self):
        tm = _make_transition_memory()
        factors = compute_multi_step_factors(
            "unstable", "basin_2", ["s1", "s2"], tm,
        )
        # s1 has higher two-step value (0.4375 vs 0.3) so higher factor
        assert factors["s1"] >= factors["s2"]

    def test_empty_strategies(self):
        tm = _make_transition_memory()
        factors = compute_multi_step_factors("unstable", "basin_2", [], tm)
        assert factors == {}

    def test_empty_memory_all_neutral(self):
        factors = compute_multi_step_factors(
            "unstable", "basin_2", ["s1", "s2"], {},
        )
        assert factors["s1"] == 1.0
        assert factors["s2"] == 1.0

    def test_unknown_strategy_neutral(self):
        tm = _make_transition_memory()
        factors = compute_multi_step_factors(
            "unstable", "basin_2", ["s99"], tm,
        )
        assert factors["s99"] == 1.0

    def test_deterministic(self):
        tm = _make_transition_memory()
        a = compute_multi_step_factors("unstable", "basin_2", ["s1", "s2"], tm)
        b = compute_multi_step_factors("unstable", "basin_2", ["s1", "s2"], tm)
        assert a == b

    def test_no_mutation(self):
        tm = _make_transition_memory()
        frozen = copy.deepcopy(tm)
        compute_multi_step_factors("unstable", "basin_2", ["s1", "s2"], tm)
        assert tm == frozen


# ---------------------------------------------------------------------------
# 6. Integration: multi-step factor in scoring pipeline
# ---------------------------------------------------------------------------


class TestMultiStepScoringIntegration:

    def test_score_includes_multi_step_factor(self):
        from qec.analysis.strategy_memory import (
            compute_regime_key,
            compute_attractor_id,
            score_strategy_with_memory,
        )
        tm = _make_transition_memory()
        state = {
            "regime": "unstable",
            "basin_score": 0.5,
            "phi": 0.5,
            "consistency": 0.5,
            "divergence": 0.1,
            "curvature": 0.1,
            "resonance": 0.1,
            "complexity": 0.1,
        }
        strategy = {"action_type": "damping", "params": {"alpha": 0.1}}
        regime_key = compute_regime_key(
            "unstable", compute_attractor_id(0.5),
        )
        history = [{"score": 0.3, "direction": "improved"}]
        memory = {}

        result = score_strategy_with_memory(
            strategy, state, history, memory, "s1",
            regime_key=regime_key,
            transition_memory=tm,
        )
        assert "multi_step_factor" in result
        assert 0.8 <= result["multi_step_factor"] <= 1.2

    def test_no_transition_memory_factor_is_one(self):
        from qec.analysis.strategy_memory import score_strategy_with_memory
        state = {
            "regime": "unstable",
            "basin_score": 0.5,
            "phi": 0.5,
            "consistency": 0.5,
            "divergence": 0.1,
            "curvature": 0.1,
            "resonance": 0.1,
            "complexity": 0.1,
        }
        strategy = {"action_type": "damping", "params": {"alpha": 0.1}}
        result = score_strategy_with_memory(
            strategy, state, [], {}, "s1",
        )
        assert result["multi_step_factor"] == 1.0

    def test_select_with_multi_step(self):
        from qec.analysis.strategy_memory import (
            compute_regime_key,
            compute_attractor_id,
            select_strategy_with_memory,
        )
        tm = _make_transition_memory()
        state = {
            "regime": "unstable",
            "basin_score": 0.5,
            "phi": 0.5,
            "consistency": 0.5,
            "divergence": 0.1,
            "curvature": 0.1,
            "resonance": 0.1,
            "complexity": 0.1,
        }
        strategies = {
            "s1": {"action_type": "damping", "params": {"alpha": 0.1}, "confidence": 0.5},
            "s2": {"action_type": "scaling", "params": {"beta": 0.2}, "confidence": 0.5},
        }
        regime_key = compute_regime_key(
            "unstable", compute_attractor_id(0.5),
        )
        history = [{"score": 0.3, "direction": "improved"}]

        result = select_strategy_with_memory(
            strategies, state, history, {},
            regime_key=regime_key,
            transition_memory=tm,
        )
        assert "multi_step_factor" in result
        assert 0.8 <= result["multi_step_factor"] <= 1.2

    def test_select_empty_strategies_has_multi_step_factor(self):
        from qec.analysis.strategy_memory import select_strategy_with_memory
        result = select_strategy_with_memory(
            {}, {"regime": "unstable"}, [], {},
        )
        assert result["multi_step_factor"] == 1.0

    def test_integration_deterministic(self):
        from qec.analysis.strategy_memory import (
            compute_regime_key,
            compute_attractor_id,
            select_strategy_with_memory,
        )
        tm = _make_transition_memory()
        state = {
            "regime": "unstable",
            "basin_score": 0.5,
            "phi": 0.5,
            "consistency": 0.5,
            "divergence": 0.1,
            "curvature": 0.1,
            "resonance": 0.1,
            "complexity": 0.1,
        }
        strategies = {
            "s1": {"action_type": "damping", "params": {"alpha": 0.1}, "confidence": 0.5},
            "s2": {"action_type": "scaling", "params": {"beta": 0.2}, "confidence": 0.5},
        }
        regime_key = compute_regime_key(
            "unstable", compute_attractor_id(0.5),
        )
        history = [{"score": 0.3, "direction": "improved"}]

        a = select_strategy_with_memory(
            strategies, state, history, {},
            regime_key=regime_key, transition_memory=tm,
        )
        b = select_strategy_with_memory(
            strategies, state, history, {},
            regime_key=regime_key, transition_memory=tm,
        )
        assert a == b

    def test_no_mutation_of_transition_memory(self):
        from qec.analysis.strategy_memory import (
            compute_regime_key,
            compute_attractor_id,
            select_strategy_with_memory,
        )
        tm = _make_transition_memory()
        frozen = copy.deepcopy(tm)
        state = {
            "regime": "unstable",
            "basin_score": 0.5,
            "phi": 0.5,
            "consistency": 0.5,
            "divergence": 0.1,
            "curvature": 0.1,
            "resonance": 0.1,
            "complexity": 0.1,
        }
        strategies = {
            "s1": {"action_type": "damping", "params": {"alpha": 0.1}, "confidence": 0.5},
        }
        regime_key = compute_regime_key(
            "unstable", compute_attractor_id(0.5),
        )
        select_strategy_with_memory(
            strategies, state, [], {},
            regime_key=regime_key, transition_memory=tm,
        )
        assert tm == frozen


# ---------------------------------------------------------------------------
# 7. Regression: existing behavior preserved
# ---------------------------------------------------------------------------


class TestNoRegression:

    def test_score_without_transition_memory_unchanged(self):
        """v99.1.0 flat path should be unaffected."""
        from qec.analysis.strategy_memory import score_strategy_with_memory
        state = {
            "regime": "unstable",
            "basin_score": 0.5,
            "phi": 0.5,
            "consistency": 0.5,
            "divergence": 0.1,
            "curvature": 0.1,
            "resonance": 0.1,
            "complexity": 0.1,
        }
        strategy = {"action_type": "damping", "params": {"alpha": 0.1}}
        result = score_strategy_with_memory(
            strategy, state, [], {}, "s1",
        )
        # v99.1.0 path: no regime_key, no transition_memory
        assert result["transition_bias"] == 1.0
        assert result["multi_step_factor"] == 1.0

    def test_score_with_regime_key_only_unchanged(self):
        """v99.2.0 additive path should be unaffected."""
        from qec.analysis.strategy_memory import (
            compute_regime_key,
            compute_attractor_id,
            score_strategy_with_memory,
        )
        state = {
            "regime": "unstable",
            "basin_score": 0.5,
            "phi": 0.5,
            "consistency": 0.5,
            "divergence": 0.1,
            "curvature": 0.1,
            "resonance": 0.1,
            "complexity": 0.1,
        }
        strategy = {"action_type": "damping", "params": {"alpha": 0.1}}
        regime_key = compute_regime_key(
            "unstable", compute_attractor_id(0.5),
        )
        result = score_strategy_with_memory(
            strategy, state, [], {}, "s1",
            regime_key=regime_key,
        )
        # v99.2.0 path: regime_key but no transition_memory
        assert result["transition_bias"] == 1.0
        assert result["multi_step_factor"] == 1.0
