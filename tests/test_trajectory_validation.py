"""Tests for trajectory validation & policy constraints (v99.9.0)."""

from __future__ import annotations

from qec.analysis.trajectory_validation import (
    compute_guardrail_penalty,
    compute_monotonicity,
    compute_strategy_consistency,
    compute_trajectory_diagnostics,
    compute_trajectory_score,
    validate_transition,
)
from qec.analysis.policy_signal_robustness import compute_robust_score


# ---------------------------------------------------------------------------
# 1. Transition Validation
# ---------------------------------------------------------------------------


class TestValidateTransition:

    def test_improvement_above_one(self):
        before = {"score": 0.3, "energy": 0.7, "coherence": 0.4}
        after = {"score": 0.8, "energy": 0.3, "coherence": 0.8}
        result = validate_transition(before, after)
        assert result > 1.0

    def test_degradation_below_one(self):
        before = {"score": 0.8, "energy": 0.3, "coherence": 0.8}
        after = {"score": 0.3, "energy": 0.7, "coherence": 0.4}
        result = validate_transition(before, after)
        assert result < 1.0

    def test_neutral_near_one(self):
        metrics = {"score": 0.5, "energy": 0.5, "coherence": 0.5}
        result = validate_transition(metrics, metrics)
        assert abs(result - 1.0) < 1e-9

    def test_bounded_low(self):
        before = {"score": 1.0, "energy": 0.0, "coherence": 1.0}
        after = {"score": 0.0, "energy": 1.0, "coherence": 0.0}
        result = validate_transition(before, after)
        assert result >= 0.7

    def test_bounded_high(self):
        before = {"score": 0.0, "energy": 1.0, "coherence": 0.0}
        after = {"score": 1.0, "energy": 0.0, "coherence": 1.0}
        result = validate_transition(before, after)
        assert result <= 1.1

    def test_missing_keys_default_to_zero(self):
        result = validate_transition({}, {})
        assert 0.7 <= result <= 1.1

    def test_deterministic(self):
        before = {"score": 0.4, "energy": 0.6, "coherence": 0.5}
        after = {"score": 0.6, "energy": 0.4, "coherence": 0.7}
        a = validate_transition(before, after)
        b = validate_transition(before, after)
        assert a == b

    def test_no_mutation(self):
        before = {"score": 0.4, "energy": 0.6, "coherence": 0.5}
        after = {"score": 0.6, "energy": 0.4, "coherence": 0.7}
        before_copy = dict(before)
        after_copy = dict(after)
        validate_transition(before, after)
        assert before == before_copy
        assert after == after_copy


# ---------------------------------------------------------------------------
# 2. Monotonicity Constraint
# ---------------------------------------------------------------------------


class TestComputeMonotonicity:

    def test_consistent_improvement(self):
        history = [
            {"score": 0.1},
            {"score": 0.3},
            {"score": 0.5},
            {"score": 0.7},
            {"score": 0.9},
        ]
        result = compute_monotonicity(history)
        assert result > 1.0

    def test_consistent_improvement_max(self):
        history = [{"score": float(i)} for i in range(10)]
        result = compute_monotonicity(history)
        assert abs(result - 1.2) < 1e-9

    def test_consistent_degradation(self):
        history = [
            {"score": 0.9},
            {"score": 0.7},
            {"score": 0.5},
            {"score": 0.3},
            {"score": 0.1},
        ]
        result = compute_monotonicity(history)
        assert result < 1.0

    def test_mixed_neutral(self):
        history = [
            {"score": 0.5},
            {"score": 0.7},
            {"score": 0.4},
            {"score": 0.6},
        ]
        result = compute_monotonicity(history)
        # 2 positive, 1 negative → ratio = 2/3 → ~1.067
        assert 0.8 <= result <= 1.2

    def test_empty_history(self):
        assert compute_monotonicity([]) == 1.0

    def test_single_entry(self):
        assert compute_monotonicity([{"score": 0.5}]) == 1.0

    def test_flat_history(self):
        history = [{"score": 0.5}] * 5
        assert compute_monotonicity(history) == 1.0

    def test_bounded(self):
        # All degrading
        history = [{"score": 1.0 - i * 0.1} for i in range(10)]
        result = compute_monotonicity(history)
        assert result >= 0.8
        assert result <= 1.2

    def test_deterministic(self):
        history = [{"score": 0.1 * i} for i in range(5)]
        a = compute_monotonicity(history)
        b = compute_monotonicity(history)
        assert a == b


# ---------------------------------------------------------------------------
# 3. Strategy Consistency Constraint
# ---------------------------------------------------------------------------


class TestComputeStrategyConsistency:

    def test_no_switching(self):
        history = [{"strategy": "A"}] * 5
        result = compute_strategy_consistency(history)
        assert abs(result - 1.05) < 1e-9

    def test_rapid_switching(self):
        history = [
            {"strategy": "A"},
            {"strategy": "B"},
            {"strategy": "A"},
            {"strategy": "B"},
            {"strategy": "A"},
        ]
        result = compute_strategy_consistency(history)
        assert result < 1.0

    def test_max_switching_penalty(self):
        history = [
            {"strategy": "A"},
            {"strategy": "B"},
            {"strategy": "A"},
            {"strategy": "B"},
        ]
        result = compute_strategy_consistency(history)
        # switch_rate = 1.0 → consistency = 0.85
        assert abs(result - 0.85) < 1e-9

    def test_partial_switching(self):
        history = [
            {"strategy": "A"},
            {"strategy": "A"},
            {"strategy": "B"},
            {"strategy": "B"},
        ]
        result = compute_strategy_consistency(history)
        # 1 switch in 3 transitions → switch_rate = 1/3
        # consistency = 1.05 - 0.2*(1/3) ≈ 0.983
        assert 0.85 <= result <= 1.05

    def test_empty_history(self):
        assert compute_strategy_consistency([]) == 1.0

    def test_single_entry(self):
        assert compute_strategy_consistency([{"strategy": "A"}]) == 1.0

    def test_bounded(self):
        # Every step switches
        history = [{"strategy": chr(65 + i)} for i in range(10)]
        result = compute_strategy_consistency(history)
        assert result >= 0.85
        assert result <= 1.05

    def test_deterministic(self):
        history = [{"strategy": "A"}, {"strategy": "B"}, {"strategy": "A"}]
        a = compute_strategy_consistency(history)
        b = compute_strategy_consistency(history)
        assert a == b

    def test_missing_strategy_key(self):
        history = [{}, {}, {}]
        result = compute_strategy_consistency(history)
        # All empty string → no switching → 1.05
        assert abs(result - 1.05) < 1e-9


# ---------------------------------------------------------------------------
# 4. Trajectory Score
# ---------------------------------------------------------------------------


class TestComputeTrajectoryScore:

    def test_all_neutral(self):
        result = compute_trajectory_score()
        assert abs(result - 1.0) < 1e-9

    def test_all_positive(self):
        result = compute_trajectory_score(1.1, 1.2, 1.05)
        expected = 1.1 * 1.2 * 1.05
        assert abs(result - min(1.2, expected)) < 1e-9

    def test_all_negative(self):
        result = compute_trajectory_score(0.7, 0.8, 0.85)
        expected = 0.7 * 0.8 * 0.85
        assert result >= 0.7

    def test_bounded_high(self):
        result = compute_trajectory_score(1.1, 1.2, 1.05)
        assert result <= 1.2

    def test_bounded_low(self):
        result = compute_trajectory_score(0.7, 0.8, 0.85)
        assert result >= 0.7

    def test_deterministic(self):
        a = compute_trajectory_score(0.9, 1.1, 0.95)
        b = compute_trajectory_score(0.9, 1.1, 0.95)
        assert a == b


# ---------------------------------------------------------------------------
# 5. Integration: Scoring adjusts with trajectory_score
# ---------------------------------------------------------------------------


class TestIntegration:

    def test_trajectory_score_in_robust_score(self):
        base = compute_robust_score(0.5)
        with_trajectory = compute_robust_score(0.5, trajectory_score=0.9)
        assert with_trajectory < base

    def test_trajectory_boost(self):
        base = compute_robust_score(0.5)
        with_trajectory = compute_robust_score(0.5, trajectory_score=1.15)
        assert with_trajectory > base

    def test_backward_compatible(self):
        """Default trajectory_score=1.0 preserves old behavior."""
        old = compute_robust_score(
            0.6, 0.95, 1.1, 1.05,
            adaptation_modulation=1.0,
            cycle_penalty=1.0,
        )
        new = compute_robust_score(
            0.6, 0.95, 1.1, 1.05,
            adaptation_modulation=1.0,
            cycle_penalty=1.0,
            trajectory_score=1.0,
        )
        assert abs(old - new) < 1e-15

    def test_all_factors_multiplicative(self):
        score = compute_robust_score(
            base_score=0.8,
            stability_weight=0.95,
            transition_bias=1.05,
            multi_step_factor=1.02,
            adaptation_modulation=1.1,
            cycle_penalty=0.95,
            trajectory_score=0.95,
        )
        expected = 0.8 * 0.95 * 1.05 * 1.02 * 1.1 * 0.95 * 0.95
        assert abs(score - max(0.0, min(1.0, expected))) < 1e-9

    def test_deterministic_full_pipeline(self):
        before = {"score": 0.3, "energy": 0.6, "coherence": 0.4}
        after = {"score": 0.6, "energy": 0.4, "coherence": 0.7}
        score_hist = [{"score": 0.3}, {"score": 0.4}, {"score": 0.6}]
        strat_hist = [
            {"strategy": "A"},
            {"strategy": "A"},
            {"strategy": "B"},
        ]

        val = validate_transition(before, after)
        mono = compute_monotonicity(score_hist)
        cons = compute_strategy_consistency(strat_hist)
        traj = compute_trajectory_score(val, mono, cons)

        result = compute_robust_score(0.6, trajectory_score=traj)

        # Repeat and verify identical
        val2 = validate_transition(before, after)
        mono2 = compute_monotonicity(score_hist)
        cons2 = compute_strategy_consistency(strat_hist)
        traj2 = compute_trajectory_score(val2, mono2, cons2)
        result2 = compute_robust_score(0.6, trajectory_score=traj2)

        assert result == result2
        assert val == val2
        assert mono == mono2
        assert cons == cons2
        assert traj == traj2


# ---------------------------------------------------------------------------
# 6. Guardrail Penalty
# ---------------------------------------------------------------------------


class TestGuardrailPenalty:

    def test_no_penalty_good_validation(self):
        result = compute_guardrail_penalty(0.9, 0.5, 0.6)
        assert result == 1.0

    def test_no_penalty_small_energy_increase(self):
        result = compute_guardrail_penalty(0.7, 0.5, 0.6)
        assert result == 1.0  # energy_increase = 0.1 < 0.2

    def test_penalty_bad_validation_and_energy_increase(self):
        result = compute_guardrail_penalty(0.72, 0.3, 0.7)
        assert 0.9 <= result < 1.0

    def test_bounded(self):
        result = compute_guardrail_penalty(0.7, 0.0, 1.0)
        assert result >= 0.9

    def test_deterministic(self):
        a = compute_guardrail_penalty(0.72, 0.3, 0.8)
        b = compute_guardrail_penalty(0.72, 0.3, 0.8)
        assert a == b


# ---------------------------------------------------------------------------
# 7. Diagnostics Output
# ---------------------------------------------------------------------------


class TestTrajectoryDiagnostics:

    def test_returns_expected_keys(self):
        result = compute_trajectory_diagnostics()
        assert "validation_score" in result
        assert "monotonicity" in result
        assert "strategy_consistency" in result
        assert "trajectory_score" in result
        assert "guardrail_penalty" in result

    def test_defaults_neutral(self):
        result = compute_trajectory_diagnostics()
        assert result["validation_score"] == 1.0
        assert result["monotonicity"] == 1.0
        assert result["strategy_consistency"] == 1.0
        assert result["trajectory_score"] == 1.0
        assert result["guardrail_penalty"] == 1.0

    def test_with_all_inputs(self):
        before = {"score": 0.3, "energy": 0.6, "coherence": 0.4}
        after = {"score": 0.6, "energy": 0.4, "coherence": 0.7}
        score_hist = [{"score": 0.3}, {"score": 0.5}, {"score": 0.7}]
        strat_hist = [{"strategy": "A"}, {"strategy": "A"}, {"strategy": "B"}]

        result = compute_trajectory_diagnostics(
            before_metrics=before,
            after_metrics=after,
            score_history=score_hist,
            strategy_history=strat_hist,
        )

        assert 0.7 <= result["validation_score"] <= 1.1
        assert 0.8 <= result["monotonicity"] <= 1.2
        assert 0.85 <= result["strategy_consistency"] <= 1.05
        assert 0.7 <= result["trajectory_score"] <= 1.2
        assert 0.9 <= result["guardrail_penalty"] <= 1.0

    def test_deterministic(self):
        before = {"score": 0.3, "energy": 0.6, "coherence": 0.4}
        after = {"score": 0.6, "energy": 0.4, "coherence": 0.7}
        a = compute_trajectory_diagnostics(before, after)
        b = compute_trajectory_diagnostics(before, after)
        assert a == b


# ---------------------------------------------------------------------------
# 8. Stress: Multi-step trajectory stability
# ---------------------------------------------------------------------------


class TestDeterministicStress:

    def test_multi_step_trajectory_100_runs(self):
        """Run full trajectory pipeline 100 times, verify identical."""
        before = {"score": 0.3, "energy": 0.6, "coherence": 0.4}
        after = {"score": 0.7, "energy": 0.3, "coherence": 0.8}
        score_hist = [{"score": 0.1 * i} for i in range(1, 8)]
        strat_hist = [
            {"strategy": "A"},
            {"strategy": "A"},
            {"strategy": "B"},
            {"strategy": "B"},
            {"strategy": "A"},
            {"strategy": "C"},
            {"strategy": "C"},
        ]

        first_diag = compute_trajectory_diagnostics(
            before, after, score_hist, strat_hist,
        )
        first_val = validate_transition(before, after)
        first_mono = compute_monotonicity(score_hist)
        first_cons = compute_strategy_consistency(strat_hist)
        first_traj = compute_trajectory_score(first_val, first_mono, first_cons)
        first_guard = compute_guardrail_penalty(first_val, 0.6, 0.3)
        first_robust = compute_robust_score(
            0.6, 0.9, 1.1, 1.0, 1.05, 0.95, first_traj,
        )

        for _ in range(99):
            assert compute_trajectory_diagnostics(
                before, after, score_hist, strat_hist,
            ) == first_diag
            assert validate_transition(before, after) == first_val
            assert compute_monotonicity(score_hist) == first_mono
            assert compute_strategy_consistency(strat_hist) == first_cons
            assert compute_trajectory_score(
                first_val, first_mono, first_cons,
            ) == first_traj
            assert compute_guardrail_penalty(first_val, 0.6, 0.3) == first_guard
            assert compute_robust_score(
                0.6, 0.9, 1.1, 1.0, 1.05, 0.95, first_traj,
            ) == first_robust
