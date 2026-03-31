"""v128.0.0 — Deterministic adaptive policy orchestrator.

Selects a stable policy decision from fused invariant risk outputs (v127)
using deterministic label mapping plus a strict fail-safe threshold.
"""

from __future__ import annotations

import math
from typing import TypedDict


class PolicyDecision(TypedDict):
    policy_label: str
    policy_score: float
    policy_escalated: bool
    fail_safe_required: bool


class AdaptivePolicyOrchestrationResult(TypedDict):
    fused_risk_score: float
    fused_risk_label: str
    policy_decision: PolicyDecision
    orchestration_ready: bool


def normalize_policy_score(value: float) -> float:
    """Normalize a policy score to [0.0, 1.0] deterministically."""
    numeric = float(value)

    if math.isnan(numeric):
        return 0.0
    if math.isinf(numeric):
        return 1.0 if numeric > 0.0 else 0.0
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric


def select_policy(fused_result: dict) -> PolicyDecision:
    """Select deterministic policy label/flags from fused risk output."""
    fused_risk_score = normalize_policy_score(fused_result.get("fused_risk_score", 0.0))
    fused_risk_label = str(fused_result.get("fused_risk_label", "critical"))

    if fused_risk_score >= 0.90:
        policy_label = "fail_safe"
    elif fused_risk_label == "safe":
        policy_label = "observe"
    elif fused_risk_label == "warning":
        policy_label = "stabilize"
    else:
        policy_label = "intervene"

    policy_escalated = policy_label in ("stabilize", "intervene", "fail_safe")
    fail_safe_required = policy_label == "fail_safe"

    return {
        "policy_label": policy_label,
        "policy_score": fused_risk_score,
        "policy_escalated": policy_escalated,
        "fail_safe_required": fail_safe_required,
    }


def run_adaptive_policy_orchestrator(
    fused_result: dict,
) -> AdaptivePolicyOrchestrationResult:
    """Run deterministic policy orchestration from fused risk result."""
    fused_risk_score = normalize_policy_score(fused_result.get("fused_risk_score", 0.0))
    fused_risk_label = str(fused_result.get("fused_risk_label", "critical"))

    return {
        "fused_risk_score": fused_risk_score,
        "fused_risk_label": fused_risk_label,
        "policy_decision": select_policy(
            {
                "fused_risk_score": fused_risk_score,
                "fused_risk_label": fused_risk_label,
            }
        ),
        "orchestration_ready": True,
    }
