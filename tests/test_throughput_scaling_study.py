# SPDX-License-Identifier: MIT
"""Deterministic tests for v138.2.2 throughput scaling study."""

from __future__ import annotations

import json

from qec.runtime.fpga_asic_control_module import build_hardware_control_dispatch
from qec.runtime.latency_budget_enforcement_hardware import LatencyBudgetPolicy, enforce_latency_budget
from qec.runtime.throughput_scaling_study import (
    ThroughputScalingPolicy,
    build_throughput_scaling_study,
    throughput_replay_projection,
    validate_throughput_scaling_study,
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
        metadata={"build": "v138.2.2", "lane_count": lane_count},
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


def _policy(*, scaling_mode: str = "linear", degradation_mode: str = "none") -> ThroughputScalingPolicy:
    return ThroughputScalingPolicy(
        policy_id="throughput-policy",
        scaling_mode=scaling_mode,
        max_parallel_lanes=2,
        target_ops_per_window=150,
        degradation_mode=degradation_mode,
        metadata={"study": "v138.2.2"},
    )


def test_same_input_same_bytes():
    enforcements = (_enforcement(lane_count=1, projected_latency_ns=90, dispatch_suffix="1"),)
    a = build_throughput_scaling_study(enforcement_set=enforcements, scaling_policy=_policy())
    b = build_throughput_scaling_study(enforcement_set=enforcements, scaling_policy=_policy())
    assert a.to_canonical_json().encode("utf-8") == b.to_canonical_json().encode("utf-8")


def test_deterministic_sample_ordering():
    one = _enforcement(lane_count=2, projected_latency_ns=90, dispatch_suffix="1")
    two = _enforcement(lane_count=1, projected_latency_ns=95, dispatch_suffix="2")
    a = build_throughput_scaling_study(enforcement_set=(one, two), scaling_policy=_policy())
    b = build_throughput_scaling_study(enforcement_set=(two, one), scaling_policy=_policy())
    assert tuple(s.sample_id for s in a.samples) == tuple(s.sample_id for s in b.samples)


def test_linear_mode_scaling():
    enforcements = (
        _enforcement(lane_count=1, projected_latency_ns=90, dispatch_suffix="1"),
        _enforcement(lane_count=2, projected_latency_ns=90, dispatch_suffix="2"),
    )
    study = build_throughput_scaling_study(enforcement_set=enforcements, scaling_policy=_policy(scaling_mode="linear"))
    assert study.samples[-1].projected_ops_per_window == 200
    assert study.samples[-1].effective_ops_per_window == 150


def test_saturating_mode_scaling():
    enforcements = (
        _enforcement(lane_count=1, projected_latency_ns=90, dispatch_suffix="1"),
        _enforcement(lane_count=2, projected_latency_ns=90, dispatch_suffix="2"),
        _enforcement(lane_count=3, projected_latency_ns=90, dispatch_suffix="3"),
    )
    study = build_throughput_scaling_study(enforcement_set=enforcements, scaling_policy=_policy(scaling_mode="saturating"))
    assert study.samples[-1].projected_ops_per_window == 300
    assert study.samples[-1].effective_ops_per_window == 200


def test_bounded_mesh_mode_scaling():
    enforcements = (
        _enforcement(lane_count=1, projected_latency_ns=90, dispatch_suffix="1"),
        _enforcement(lane_count=2, projected_latency_ns=90, dispatch_suffix="2"),
    )
    study = build_throughput_scaling_study(enforcement_set=enforcements, scaling_policy=_policy(scaling_mode="bounded_mesh"))
    assert study.samples[-1].projected_ops_per_window == 200
    assert study.samples[-1].effective_ops_per_window == 170


def test_degradation_soft_throttle_behavior():
    enforcements = (
        _enforcement(lane_count=1, projected_latency_ns=90, dispatch_suffix="1"),
        _enforcement(lane_count=3, projected_latency_ns=90, dispatch_suffix="2"),
    )
    study = build_throughput_scaling_study(
        enforcement_set=enforcements,
        scaling_policy=_policy(scaling_mode="saturating", degradation_mode="soft_throttle"),
    )
    assert study.samples[-1].effective_ops_per_window == 180


def test_degradation_hard_cap_behavior():
    enforcements = (
        _enforcement(lane_count=1, projected_latency_ns=90, dispatch_suffix="1"),
        _enforcement(lane_count=2, projected_latency_ns=90, dispatch_suffix="2"),
        _enforcement(lane_count=3, projected_latency_ns=90, dispatch_suffix="3"),
    )
    study = build_throughput_scaling_study(
        enforcement_set=enforcements,
        scaling_policy=_policy(scaling_mode="linear", degradation_mode="hard_cap"),
    )
    assert study.samples[-1].effective_ops_per_window == 150


def test_invalid_policy_rejection():
    study = build_throughput_scaling_study(
        enforcement_set=(_enforcement(lane_count=1, projected_latency_ns=90, dispatch_suffix="1"),),
        scaling_policy={
            "policy_id": "invalid",
            "scaling_mode": "linear",
            "max_parallel_lanes": 0,
            "target_ops_per_window": 150,
            "degradation_mode": "none",
            "metadata": {},
        },
    )
    assert study.validation.valid is False
    assert "policy.max_parallel_lanes must be > 0" in study.validation.errors


def test_saturation_score_bounds():
    enforcements = (
        _enforcement(lane_count=1, projected_latency_ns=90, dispatch_suffix="1"),
        _enforcement(lane_count=2, projected_latency_ns=90, dispatch_suffix="2"),
        _enforcement(lane_count=3, projected_latency_ns=90, dispatch_suffix="3"),
        _enforcement(lane_count=4, projected_latency_ns=90, dispatch_suffix="4"),
    )
    study = build_throughput_scaling_study(enforcement_set=enforcements, scaling_policy=_policy(scaling_mode="linear"))
    assert all(0.0 <= sample.saturation_score <= 1.0 for sample in study.samples)


def test_receipt_tamper_detection():
    study = build_throughput_scaling_study(
        enforcement_set=(_enforcement(lane_count=1, projected_latency_ns=90, dispatch_suffix="1"),),
        scaling_policy=_policy(),
    )
    tampered = {
        **study.to_dict(),
        "receipt": {**study.receipt.to_dict(), "receipt_hash": "0" * 64},
    }
    report = validate_throughput_scaling_study(tampered)
    assert report.valid is False
    assert "receipt.receipt_hash mismatch" in report.errors


def test_canonical_json_round_trip():
    study = build_throughput_scaling_study(
        enforcement_set=(_enforcement(lane_count=1, projected_latency_ns=90, dispatch_suffix="1"),),
        scaling_policy=_policy(),
    )
    payload = json.loads(study.to_canonical_json())
    assert payload["receipt"]["receipt_hash"] == study.receipt.receipt_hash
    assert payload["policy"]["policy_id"] == study.policy.policy_id


def test_replay_projection_stability():
    study = build_throughput_scaling_study(
        enforcement_set=(
            _enforcement(lane_count=1, projected_latency_ns=90, dispatch_suffix="1"),
            _enforcement(lane_count=2, projected_latency_ns=120, dispatch_suffix="2"),
        ),
        scaling_policy=_policy(scaling_mode="saturating"),
    )
    a = throughput_replay_projection(study)
    b = throughput_replay_projection(study)
    assert a == b
