"""Tests for v130.0.0 policy feedback controller."""

from __future__ import annotations

import dataclasses

import pytest

from qec.analysis.policy_feedback_controller import (
    FeedbackState,
    evaluate_policy_effectiveness,
    recommend_policy_adjustment,
    run_policy_feedback_controller,
)


class TestFeedbackState:
    def test_frozen_immutability(self):
        state = FeedbackState(
            previous_risk_score=0.6,
            current_risk_score=0.4,
            previous_policy="observe",
            current_policy="stabilize",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            state.current_policy = "recover"


class TestEvaluatePolicyEffectiveness:
    def test_improvement_detection(self):
        state = FeedbackState(0.7, 0.5, "observe", "stabilize")
        feedback = evaluate_policy_effectiveness(state)
        assert feedback["risk_delta"] == pytest.approx(0.2)
        assert feedback["policy_improved"] is True
        assert feedback["policy_degraded"] is False
        assert feedback["stagnation_detected"] is False

    def test_degradation_detection(self):
        state = FeedbackState(0.4, 0.6, "stabilize", "observe")
        feedback = evaluate_policy_effectiveness(state)
        assert feedback["risk_delta"] == pytest.approx(-0.2)
        assert feedback["policy_improved"] is False
        assert feedback["policy_degraded"] is True
        assert feedback["stagnation_detected"] is False

    def test_stagnation_detection(self):
        state = FeedbackState(0.5, 0.46, "observe", "observe")
        feedback = evaluate_policy_effectiveness(state)
        assert feedback["risk_delta"] == pytest.approx(0.04)
        assert feedback["policy_improved"] is False
        assert feedback["policy_degraded"] is False
        assert feedback["stagnation_detected"] is True


class TestRecommendPolicyAdjustment:
    def test_maintain_action(self):
        adjustment = recommend_policy_adjustment(
            {
                "policy_improved": True,
                "policy_degraded": False,
                "stagnation_detected": False,
                "current_risk_score": 0.3,
            }
        )
        assert adjustment == {
            "adjustment_required": False,
            "recommended_action": "maintain",
        }

    def test_escalate_action(self):
        adjustment = recommend_policy_adjustment(
            {
                "policy_improved": False,
                "policy_degraded": True,
                "stagnation_detected": False,
                "current_risk_score": 0.7,
            }
        )
        assert adjustment == {
            "adjustment_required": True,
            "recommended_action": "escalate",
        }

    def test_recover_action(self):
        adjustment = recommend_policy_adjustment(
            {
                "policy_improved": False,
                "policy_degraded": False,
                "stagnation_detected": True,
                "current_risk_score": 0.7,
            }
        )
        assert adjustment == {
            "adjustment_required": True,
            "recommended_action": "recover",
        }

    def test_fail_safe_override(self):
        adjustment = recommend_policy_adjustment(
            {
                "policy_improved": True,
                "policy_degraded": False,
                "stagnation_detected": False,
                "current_risk_score": 0.95,
            }
        )
        assert adjustment == {
            "adjustment_required": True,
            "recommended_action": "fail_safe",
        }


class TestPolicyFeedbackController:
    def test_deterministic_repeatability(self):
        def build_result():
            state = FeedbackState(0.85, 0.7, "observe", "stabilize")
            return run_policy_feedback_controller(state)

        assert build_result() == build_result()

    def test_schema_stability(self):
        state = FeedbackState(0.8, 0.7, "observe", "stabilize")
        result = run_policy_feedback_controller(state)

        assert tuple(result.keys()) == (
            "feedback_analysis",
            "policy_adjustment",
            "feedback_ready",
        )
        assert tuple(result["feedback_analysis"].keys()) == (
            "risk_delta",
            "policy_improved",
            "policy_degraded",
            "stagnation_detected",
        )
        assert tuple(result["policy_adjustment"].keys()) == (
            "adjustment_required",
            "recommended_action",
        )
        assert result["feedback_ready"] is True
