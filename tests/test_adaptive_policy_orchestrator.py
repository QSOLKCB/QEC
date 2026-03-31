"""Tests for adaptive_policy_orchestrator.py — v128.0.0."""

from __future__ import annotations

import math

from qec.analysis.adaptive_policy_orchestrator import (
    normalize_policy_score,
    run_adaptive_policy_orchestrator,
    select_policy,
)


def test_safe_label_maps_to_observe():
    result = select_policy({"fused_risk_score": 0.10, "fused_risk_label": "safe"})
    assert result["policy_label"] == "observe"


def test_warning_label_maps_to_stabilize():
    result = select_policy({"fused_risk_score": 0.50, "fused_risk_label": "warning"})
    assert result["policy_label"] == "stabilize"


def test_critical_label_maps_to_intervene():
    result = select_policy({"fused_risk_score": 0.70, "fused_risk_label": "critical"})
    assert result["policy_label"] == "intervene"


def test_fail_safe_forced_at_or_above_threshold():
    at_threshold = select_policy({"fused_risk_score": 0.90, "fused_risk_label": "safe"})
    above_threshold = select_policy(
        {"fused_risk_score": 0.95, "fused_risk_label": "warning"}
    )

    assert at_threshold["policy_label"] == "fail_safe"
    assert above_threshold["policy_label"] == "fail_safe"


def test_nan_inf_normalization():
    assert normalize_policy_score(float("nan")) == 0.0
    assert normalize_policy_score(float("inf")) == 1.0
    assert normalize_policy_score(float("-inf")) == 0.0
    assert normalize_policy_score(-0.1) == 0.0
    assert normalize_policy_score(0.0) == 0.0
    assert normalize_policy_score(0.25) == 0.25
    assert normalize_policy_score(1.0) == 1.0
    assert normalize_policy_score(1.1) == 1.0


def test_deterministic_repeatability():
    fused_result = {"fused_risk_score": 0.33, "fused_risk_label": "warning"}
    first = run_adaptive_policy_orchestrator(fused_result)
    second = run_adaptive_policy_orchestrator(fused_result)
    assert first == second


def test_schema_stability():
    result = run_adaptive_policy_orchestrator(
        {"fused_risk_score": 0.42, "fused_risk_label": "warning"}
    )
    assert list(result.keys()) == [
        "fused_risk_score",
        "fused_risk_label",
        "policy_decision",
        "orchestration_ready",
    ]
    assert isinstance(result["fused_risk_score"], float)
    assert isinstance(result["fused_risk_label"], str)
    assert isinstance(result["policy_decision"], dict)
    assert isinstance(result["orchestration_ready"], bool)

    policy_decision = result["policy_decision"]
    assert list(policy_decision.keys()) == [
        "policy_label",
        "policy_score",
        "policy_escalated",
        "fail_safe_required",
    ]
    assert isinstance(policy_decision["policy_label"], str)
    assert isinstance(policy_decision["policy_score"], float)
    assert isinstance(policy_decision["policy_escalated"], bool)
    assert isinstance(policy_decision["fail_safe_required"], bool)
    assert math.isfinite(policy_decision["policy_score"])


def test_escalation_correctness():
    observe = select_policy({"fused_risk_score": 0.10, "fused_risk_label": "safe"})
    stabilize = select_policy({"fused_risk_score": 0.50, "fused_risk_label": "warning"})
    intervene = select_policy({"fused_risk_score": 0.80, "fused_risk_label": "critical"})
    fail_safe = select_policy({"fused_risk_score": 0.95, "fused_risk_label": "safe"})

    assert observe["policy_escalated"] is False
    assert stabilize["policy_escalated"] is True
    assert intervene["policy_escalated"] is True
    assert fail_safe["policy_escalated"] is True


def test_fail_safe_correctness():
    non_fail_safe = select_policy({"fused_risk_score": 0.80, "fused_risk_label": "critical"})
    fail_safe = select_policy({"fused_risk_score": 0.90, "fused_risk_label": "warning"})

    assert non_fail_safe["fail_safe_required"] is False
    assert fail_safe["fail_safe_required"] is True


def test_threshold_boundary_at_point_nine_zero():
    below = select_policy({"fused_risk_score": 0.899999, "fused_risk_label": "critical"})
    at = select_policy({"fused_risk_score": 0.90, "fused_risk_label": "critical"})

    assert below["policy_label"] == "intervene"
    assert at["policy_label"] == "fail_safe"
