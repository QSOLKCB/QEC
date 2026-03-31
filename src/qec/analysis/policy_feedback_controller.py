"""v130.0.0 — Deterministic policy feedback controller.

This module provides a deterministic feedback analysis layer for policy
tracking and recovery optimization.
"""

from __future__ import annotations

from dataclasses import dataclass


EPSILON = 1e-12


@dataclass(frozen=True)
class FeedbackState:
    """Immutable feedback state for policy effectiveness analysis."""

    previous_risk_score: float
    current_risk_score: float
    previous_policy: str
    current_policy: str


def evaluate_policy_effectiveness(state: FeedbackState) -> dict:
    """Evaluate deterministic policy effectiveness from two risk observations."""
    risk_delta = state.previous_risk_score - state.current_risk_score
    return {
        "risk_delta": risk_delta,
        "policy_improved": risk_delta > 0.10,
        "policy_degraded": risk_delta < -0.10,
        "stagnation_detected": abs(risk_delta) <= 0.05 + EPSILON,
    }


def recommend_policy_adjustment(feedback: dict) -> dict:
    """Recommend deterministic policy adjustment action from feedback metrics."""
    current_risk_score = float(feedback.get("current_risk_score", 0.0))

    if current_risk_score >= 0.95:
        recommended_action = "fail_safe"
    elif feedback["policy_improved"]:
        recommended_action = "maintain"
    elif feedback["policy_degraded"]:
        recommended_action = "escalate"
    elif feedback["stagnation_detected"]:
        recommended_action = "recover"
    else:
        recommended_action = "maintain"

    return {
        "adjustment_required": recommended_action != "maintain",
        "recommended_action": recommended_action,
    }


def run_policy_feedback_controller(state: FeedbackState) -> dict:
    """Run deterministic policy feedback analysis and adjustment recommendation."""
    feedback_analysis = evaluate_policy_effectiveness(state)
    policy_adjustment = recommend_policy_adjustment(
        {
            **feedback_analysis,
            "current_risk_score": state.current_risk_score,
        }
    )
    return {
        "feedback_analysis": feedback_analysis,
        "policy_adjustment": policy_adjustment,
        "feedback_ready": True,
    }


__all__ = [
    "FeedbackState",
    "evaluate_policy_effectiveness",
    "recommend_policy_adjustment",
    "run_policy_feedback_controller",
]
