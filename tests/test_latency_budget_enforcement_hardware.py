# SPDX-License-Identifier: MIT
"""Deterministic tests for v138.2.1 latency-budget enforcement hardware."""

from __future__ import annotations

import json

from qec.runtime.fpga_asic_control_module import build_hardware_control_dispatch
from qec.runtime.latency_budget_enforcement_hardware import (
    LatencyBudgetReceipt,
    LatencyBudgetPolicy,
    LatencyEnforcementDecision,
    compute_latency_violation_class,
    enforce_latency_budget,
    replay_timing_projection,
    validate_latency_budget,
)


def _dispatch(*, target_family: str = "fpga"):
    return build_hardware_control_dispatch(
        execution_hash="e" * 64,
        package_hash="a" * 64,
        lane_id="lane-0",
        lane_family="surface_code",
        hardware_target={
            "target_family": target_family,
            "target_name": "deterministic_target",
            "target_class": "accelerator",
            "supported_lane_families": ["surface_code", "qldpc"],
            "latency_budget_ns": 100,
            "throughput_budget_ops": 1000,
            "metadata": {"fabric": "simulated"},
        },
        dispatch_policy="strict_deterministic",
        projected_base_latency_ns=50,
        priority_rank=0,
        lane_count=1,
        metadata={"build": "v138.2.1"},
    )


def _policy(*, action: str = "throttle") -> LatencyBudgetPolicy:
    return LatencyBudgetPolicy(
        policy_id="policy-main",
        max_latency_ns=100,
        hard_limit_ns=150,
        violation_action=action,
        recovery_mode="bounded",
        metadata={"tier": "default"},
    )


def test_nominal_path():
    enforcement = enforce_latency_budget(
        dispatch_receipt=_dispatch(),
        projected_latency_ns=90,
        budget_policy=_policy(),
    )
    assert enforcement.decision.decision == "allow"
    assert enforcement.validation.valid is True
    assert enforcement.receipt.within_budget is True


def test_warning_path():
    enforcement = enforce_latency_budget(
        dispatch_receipt=_dispatch(),
        projected_latency_ns=110,
        budget_policy=_policy(),
    )
    assert compute_latency_violation_class(projected_latency_ns=110, policy=_policy()) == "warning"
    assert enforcement.decision.decision == "throttle"
    assert enforcement.validation.valid is True


def test_violation_path():
    enforcement = enforce_latency_budget(
        dispatch_receipt=_dispatch(),
        projected_latency_ns=140,
        budget_policy=_policy(),
    )
    assert compute_latency_violation_class(projected_latency_ns=140, policy=_policy()) == "violation"
    assert enforcement.decision.decision == "throttle"
    assert enforcement.validation.valid is True


def test_hard_breach_path():
    enforcement = enforce_latency_budget(
        dispatch_receipt=_dispatch(),
        projected_latency_ns=151,
        budget_policy=_policy(),
    )
    assert compute_latency_violation_class(projected_latency_ns=151, policy=_policy()) == "hard_breach"
    assert enforcement.decision.hard_limit_breached is True
    assert enforcement.decision.decision == "reject"


def test_invalid_policy_rejection():
    enforcement = enforce_latency_budget(
        dispatch_receipt=_dispatch(),
        projected_latency_ns=110,
        budget_policy={
            "policy_id": "invalid",
            "max_latency_ns": 0,
            "hard_limit_ns": 10,
            "violation_action": "throttle",
            "recovery_mode": "bounded",
            "metadata": {},
        },
    )
    assert enforcement.validation.valid is False
    assert "policy.max_latency_ns must be > 0" in enforcement.validation.errors


def test_stable_hash_equality():
    a = enforce_latency_budget(dispatch_receipt=_dispatch(), projected_latency_ns=95, budget_policy=_policy())
    b = enforce_latency_budget(dispatch_receipt=_dispatch(), projected_latency_ns=95, budget_policy=_policy())
    assert a.receipt.receipt_hash == b.receipt.receipt_hash
    assert a.to_canonical_json().encode("utf-8") == b.to_canonical_json().encode("utf-8")


def test_canonical_json_round_trip():
    enforcement = enforce_latency_budget(dispatch_receipt=_dispatch(), projected_latency_ns=98, budget_policy=_policy())
    payload = json.loads(enforcement.to_canonical_json())
    assert payload["receipt"]["receipt_hash"] == enforcement.receipt.receipt_hash
    assert payload["decision"]["dispatch_id"] == enforcement.decision.dispatch_id


def test_semantic_tamper_detection():
    enforcement = enforce_latency_budget(dispatch_receipt=_dispatch(), projected_latency_ns=90, budget_policy=_policy())
    tampered = {
        "policy": enforcement.policy.to_dict(),
        "decision": {
            **enforcement.decision.to_dict(),
            "within_budget": False,
        },
        "receipt": enforcement.receipt.to_dict(),
        "validation": enforcement.validation.to_dict(),
        "target_family": enforcement.target_family,
    }
    report = validate_latency_budget(tampered)
    assert report.valid is False
    assert "decision.within_budget invariant violated" in report.errors


def test_non_hard_violation_decision_tamper_detection():
    enforcement = enforce_latency_budget(dispatch_receipt=_dispatch(), projected_latency_ns=140, budget_policy=_policy())

    tampered_decision = LatencyEnforcementDecision(
        dispatch_id=enforcement.decision.dispatch_id,
        projected_latency_ns=enforcement.decision.projected_latency_ns,
        within_budget=enforcement.decision.within_budget,
        hard_limit_breached=enforcement.decision.hard_limit_breached,
        decision="allow",
        enforcement_reason=enforcement.decision.enforcement_reason,
        metadata=enforcement.decision.metadata,
    )
    tampered_receipt = LatencyBudgetReceipt(
        policy_hash=enforcement.receipt.policy_hash,
        dispatch_hash=enforcement.receipt.dispatch_hash,
        decision_hash=tampered_decision.stable_hash(),
        within_budget=enforcement.receipt.within_budget,
        receipt_hash="",
    )
    tampered_receipt = LatencyBudgetReceipt(
        policy_hash=tampered_receipt.policy_hash,
        dispatch_hash=tampered_receipt.dispatch_hash,
        decision_hash=tampered_receipt.decision_hash,
        within_budget=tampered_receipt.within_budget,
        receipt_hash=tampered_receipt.stable_hash(),
    )

    tampered = {
        "policy": enforcement.policy.to_dict(),
        "decision": tampered_decision.to_dict(),
        "receipt": tampered_receipt.to_dict(),
        "validation": enforcement.validation.to_dict(),
        "target_family": enforcement.target_family,
    }
    report = validate_latency_budget(tampered)
    assert report.valid is False
    assert "decision.decision is inconsistent with policy semantics" in report.errors


def test_replay_timing_lineage_stability():
    enforcement = enforce_latency_budget(dispatch_receipt=_dispatch(), projected_latency_ns=120, budget_policy=_policy())
    a = replay_timing_projection(enforcement)
    b = replay_timing_projection(enforcement)
    assert a == b


def test_reroute_is_limited_to_simulation_shadow():
    enforcement = enforce_latency_budget(
        dispatch_receipt=_dispatch(target_family="simulation_shadow"),
        projected_latency_ns=140,
        budget_policy=_policy(action="reroute"),
    )
    assert enforcement.decision.decision == "reroute"


def test_none_metadata_policy_is_valid():
    """_normalize_policy() must treat explicit None metadata as {} without TypeError."""
    enforcement = enforce_latency_budget(
        dispatch_receipt=_dispatch(),
        projected_latency_ns=110,
        budget_policy={
            "policy_id": "policy-none-meta",
            "max_latency_ns": 100,
            "hard_limit_ns": 150,
            "violation_action": "throttle",
            "recovery_mode": "bounded",
            "metadata": None,
        },
    )
    assert enforcement.decision.decision == "throttle"
    assert enforcement.validation.valid is True


def test_violation_action_allow_honored():
    """violation_action='allow' must produce an 'allow' decision for warning/violation classes."""
    enforcement = enforce_latency_budget(
        dispatch_receipt=_dispatch(),
        projected_latency_ns=120,
        budget_policy=_policy(action="allow"),
    )
    assert compute_latency_violation_class(projected_latency_ns=120, policy=_policy(action="allow")) == "warning"
    assert enforcement.decision.decision == "allow"
    assert enforcement.validation.valid is True


def test_violation_action_reject_honored():
    """violation_action='reject' must produce a 'reject' decision for warning/violation classes."""
    enforcement = enforce_latency_budget(
        dispatch_receipt=_dispatch(),
        projected_latency_ns=120,
        budget_policy=_policy(action="reject"),
    )
    assert enforcement.decision.decision == "reject"
    assert enforcement.validation.valid is True


def test_target_family_tamper_detection():
    """Replacing target_family in the serialised enforcement must be caught by validation."""
    enforcement = enforce_latency_budget(
        dispatch_receipt=_dispatch(target_family="fpga"),
        projected_latency_ns=140,
        budget_policy=_policy(action="reroute"),
    )
    # Original decision must be 'throttle' (reroute not valid for fpga)
    assert enforcement.decision.decision == "throttle"

    # Tamper: replace target_family to make it look like reroute was correct
    tampered = {**enforcement.to_dict(), "target_family": "simulation_shadow"}
    report = validate_latency_budget(tampered)
    assert report.valid is False
    assert "decision.decision is inconsistent with policy semantics" in report.errors
