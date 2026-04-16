# SPDX-License-Identifier: MIT
"""Deterministic tests for v138.2.3 thermal / power budget receipt pack."""

from __future__ import annotations

import json

from qec.runtime.fpga_asic_control_module import build_hardware_control_dispatch
from qec.runtime.latency_budget_enforcement_hardware import LatencyBudgetPolicy, enforce_latency_budget
from qec.runtime.throughput_scaling_study import ThroughputScalingPolicy, build_throughput_scaling_study
from qec.runtime.thermal_power_budget_receipt_pack import (
    ThermalPowerPolicy,
    budget_replay_projection,
    build_thermal_power_budget_pack,
    validate_thermal_power_budget_pack,
)


def _dispatch(*, lane_count: int, dispatch_suffix: str, target_family: str = "fpga"):
    return build_hardware_control_dispatch(
        execution_hash=("e" * 63) + dispatch_suffix,
        package_hash="a" * 64,
        lane_id=f"lane-{dispatch_suffix}",
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
        lane_count=lane_count,
        metadata={"build": "v138.2.3", "lane_count": lane_count},
    )


def _enforcement(*, lane_count: int, projected_latency_ns: int, dispatch_suffix: str):
    dispatch = _dispatch(lane_count=lane_count, dispatch_suffix=dispatch_suffix)
    enforcement = enforce_latency_budget(
        dispatch_receipt=dispatch,
        projected_latency_ns=projected_latency_ns,
        budget_policy=LatencyBudgetPolicy(
            policy_id="latency-policy",
            max_latency_ns=100,
            hard_limit_ns=140,
            violation_action="throttle",
            recovery_mode="bounded",
            metadata={"tier": "default"},
        ),
    )
    decision_metadata = dict(enforcement.decision.metadata)
    decision_metadata["lane_count"] = lane_count
    return {
        "policy": enforcement.policy.to_dict(),
        "decision": {**enforcement.decision.to_dict(), "metadata": decision_metadata},
        "receipt": enforcement.receipt.to_dict(),
        "validation": enforcement.validation.to_dict(),
        "target_family": enforcement.target_family,
    }


def _throughput_study(effective_mode: str = "linear"):
    enforcements = (
        _enforcement(lane_count=1, projected_latency_ns=90, dispatch_suffix="1"),
        _enforcement(lane_count=2, projected_latency_ns=90, dispatch_suffix="2"),
        _enforcement(lane_count=3, projected_latency_ns=120, dispatch_suffix="3"),
    )
    return build_throughput_scaling_study(
        enforcement_set=enforcements,
        scaling_policy=ThroughputScalingPolicy(
            policy_id="throughput-policy",
            scaling_mode=effective_mode,
            max_parallel_lanes=4,
            target_ops_per_window=1000,
            degradation_mode="none",
            metadata={"study": "v138.2.3"},
        ),
    )


def _policy(**overrides):
    base = {
        "policy_id": "thermal-power-policy",
        "max_thermal_units": 30,
        "max_power_units": 35,
        "thermal_mode": "linear",
        "power_mode": "proportional",
        "metadata": {"release": "v138.2.3"},
    }
    base.update(overrides)
    return ThermalPowerPolicy(**base)


def test_same_input_same_bytes():
    throughput_study = _throughput_study()
    a = build_thermal_power_budget_pack(throughput_study=throughput_study, thermal_power_policy=_policy())
    b = build_thermal_power_budget_pack(throughput_study=throughput_study, thermal_power_policy=_policy())
    assert a.to_canonical_json().encode("utf-8") == b.to_canonical_json().encode("utf-8")


def test_thermal_budget_nominal():
    study = build_thermal_power_budget_pack(
        throughput_study=_throughput_study(),
        thermal_power_policy=_policy(max_thermal_units=40, thermal_mode="linear"),
    )
    assert all(receipt.within_thermal_budget for receipt in study.thermal_receipts)


def test_thermal_budget_exceeded():
    study = build_thermal_power_budget_pack(
        throughput_study=_throughput_study(),
        thermal_power_policy=_policy(max_thermal_units=10, thermal_mode="linear"),
    )
    assert any(not receipt.within_thermal_budget for receipt in study.thermal_receipts)


def test_power_budget_nominal():
    study = build_thermal_power_budget_pack(
        throughput_study=_throughput_study(),
        thermal_power_policy=_policy(max_power_units=50, power_mode="proportional"),
    )
    assert all(receipt.within_power_budget for receipt in study.power_receipts)


def test_power_budget_exceeded():
    study = build_thermal_power_budget_pack(
        throughput_study=_throughput_study(),
        thermal_power_policy=_policy(max_power_units=10, power_mode="proportional"),
    )
    assert any(not receipt.within_power_budget for receipt in study.power_receipts)


def test_pressure_score_bounds():
    study = build_thermal_power_budget_pack(
        throughput_study=_throughput_study(),
        thermal_power_policy=_policy(max_thermal_units=3, max_power_units=3),
    )
    assert all(0.0 <= receipt.thermal_pressure_score <= 1.0 for receipt in study.thermal_receipts)
    assert all(0.0 <= receipt.power_pressure_score <= 1.0 for receipt in study.power_receipts)


def test_invalid_policy_rejection():
    study = build_thermal_power_budget_pack(
        throughput_study=_throughput_study(),
        thermal_power_policy={
            "policy_id": "invalid",
            "max_thermal_units": 0,
            "max_power_units": 35,
            "thermal_mode": "linear",
            "power_mode": "proportional",
            "metadata": {},
        },
    )
    assert study.validation.valid is False
    assert "policy.max_thermal_units must be > 0" in study.validation.errors


def test_negative_payload_rejection():
    clean = build_thermal_power_budget_pack(throughput_study=_throughput_study(), thermal_power_policy=_policy())
    tampered = clean.to_dict()
    tampered["thermal_receipts"][0]["projected_thermal_units"] = -1
    report = validate_thermal_power_budget_pack(tampered)
    assert report.valid is False
    assert "projected_thermal_units must be >= 0" in report.errors[0]


def test_receipt_tamper_detection():
    study = build_thermal_power_budget_pack(throughput_study=_throughput_study(), thermal_power_policy=_policy())
    tampered = {**study.to_dict(), "receipt_pack": {**study.receipt_pack.to_dict(), "receipt_hash": "0" * 64}}
    report = validate_thermal_power_budget_pack(tampered)
    assert report.valid is False
    assert "receipt_pack.receipt_hash mismatch" in report.errors


def test_canonical_json_round_trip():
    study = build_thermal_power_budget_pack(throughput_study=_throughput_study(), thermal_power_policy=_policy())
    payload = json.loads(study.to_canonical_json())
    assert payload["receipt_pack"]["receipt_hash"] == study.receipt_pack.receipt_hash
    assert payload["policy"]["policy_id"] == study.policy.policy_id


def test_replay_projection_stability():
    study = build_thermal_power_budget_pack(throughput_study=_throughput_study(), thermal_power_policy=_policy())
    a = budget_replay_projection(study)
    b = budget_replay_projection(study)
    assert a == b
