"""Tests for invariant_fusion_engine.py — v127.0.0."""

from __future__ import annotations

import math

from qec.analysis.invariant_fusion_engine import (
    compute_control_risk,
    compute_logical_risk,
    compute_topology_risk,
    fuse_invariants,
    normalize_risk_signal,
)


def test_valid_safe_safe_path():
    result = fuse_invariants(
        {"proof_status": "valid"},
        {"topology_risk": "safe"},
        {"fail_safe_triggered": False, "hysteresis_active": False},
    )
    assert result["logical_risk"] == 0.0
    assert result["topology_risk"] == 0.0
    assert result["control_risk"] == 0.0
    assert result["fused_risk_score"] == 0.0
    assert result["fused_risk_label"] == "safe"


def test_partial_proof_path():
    assert compute_logical_risk({"proof_status": "partial"}) == 0.5


def test_violated_proof_path():
    assert compute_logical_risk({"proof_status": "violated"}) == 1.0


def test_critical_topology_path():
    assert compute_topology_risk({"topology_risk": "critical"}) == 1.0


def test_fail_safe_control_path():
    assert compute_control_risk(
        {"fail_safe_triggered": True, "hysteresis_active": False}
    ) == 1.0


def test_weighted_fusion_correctness():
    result = fuse_invariants(
        {"proof_status": "partial"},
        {"topology_risk": "critical"},
        {"fail_safe_triggered": False, "hysteresis_active": True},
    )
    expected = (0.4 * 0.5) + (0.3 * 1.0) + (0.3 * 0.5)
    assert result["fused_risk_score"] == expected


def test_label_threshold_correctness():
    safe_result = fuse_invariants(
        {"proof_status": "valid"},
        {"topology_risk": "safe"},
        {"fail_safe_triggered": False, "hysteresis_active": True},
    )
    warning_result = fuse_invariants(
        {"proof_status": "partial"},
        {"topology_risk": "warning"},
        {"fail_safe_triggered": False, "hysteresis_active": False},
    )
    critical_result = fuse_invariants(
        {"proof_status": "violated"},
        {"topology_risk": "critical"},
        {"fail_safe_triggered": False, "hysteresis_active": False},
    )

    assert safe_result["fused_risk_label"] == "safe"
    assert warning_result["fused_risk_label"] == "warning"
    assert critical_result["fused_risk_label"] == "critical"


def test_nan_inf_normalization():
    assert normalize_risk_signal(float("nan")) == 0.0
    assert normalize_risk_signal(float("inf")) == 1.0
    assert normalize_risk_signal(float("-inf")) == 0.0
    assert normalize_risk_signal(-0.1) == 0.0
    assert normalize_risk_signal(0.0) == 0.0
    assert normalize_risk_signal(0.25) == 0.25
    assert normalize_risk_signal(1.0) == 1.0
    assert normalize_risk_signal(1.5) == 1.0


def test_deterministic_repeatability():
    inputs = (
        {"proof_status": "partial"},
        {"topology_risk": "warning"},
        {"fail_safe_triggered": False, "hysteresis_active": True},
    )
    first = fuse_invariants(*inputs)
    second = fuse_invariants(*inputs)
    assert first == second


def test_schema_stability():
    result = fuse_invariants(
        {"proof_status": "valid"},
        {"topology_risk": "safe"},
        {"fail_safe_triggered": False, "hysteresis_active": False},
    )
    assert list(result.keys()) == [
        "logical_risk",
        "topology_risk",
        "control_risk",
        "fused_risk_score",
        "fused_risk_label",
    ]
    assert isinstance(result["logical_risk"], float)
    assert isinstance(result["topology_risk"], float)
    assert isinstance(result["control_risk"], float)
    assert isinstance(result["fused_risk_score"], float)
    assert isinstance(result["fused_risk_label"], str)
    assert math.isfinite(result["fused_risk_score"])
