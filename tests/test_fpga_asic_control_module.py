# SPDX-License-Identifier: MIT
"""Deterministic tests for v138.2.0 FPGA / ASIC control module."""

from __future__ import annotations

import json

import pytest

from qec.runtime.fpga_asic_control_module import (
    HardwareControlValidationError,
    HardwareTarget,
    build_hardware_control_dispatch,
    shadow_replay_projection,
    validate_hardware_control_dispatch,
)


def _target(*, family: str = "fpga", lanes: tuple[str, ...] = ("surface_code", "qldpc"), budget: int = 100) -> HardwareTarget:
    return HardwareTarget(
        target_family=family,
        target_name="xilinx_u55n",
        target_class="accelerator",
        supported_lane_families=lanes,
        latency_budget_ns=budget,
        throughput_budget_ops=1000,
        metadata={"fabric": "simulated"},
    )


def _build(*, priority_rank: int = 0, budget: int = 100, lane_family: str = "surface_code"):
    return build_hardware_control_dispatch(
        execution_hash="e" * 64,
        package_hash="a" * 64,
        lane_id="lane-0",
        lane_family=lane_family,
        hardware_target=_target(budget=budget),
        dispatch_policy="strict_deterministic",
        projected_base_latency_ns=50,
        priority_rank=priority_rank,
        lane_count=2,
        metadata={"region": "us-east", "build": "v138.2.0"},
    )


def test_same_input_same_bytes():
    a = _build()
    b = _build()
    assert a.to_canonical_json().encode("utf-8") == b.to_canonical_json().encode("utf-8")


def test_latency_within_budget():
    dispatch = _build(budget=200)
    assert dispatch.latency_receipt.projected_dispatch_ns == 60
    assert dispatch.latency_receipt.within_budget is True


def test_latency_budget_violation():
    dispatch = _build(budget=55)
    assert dispatch.latency_receipt.projected_dispatch_ns == 60
    assert dispatch.latency_receipt.within_budget is False


def test_unsupported_target_rejection():
    with pytest.raises(HardwareControlValidationError):
        build_hardware_control_dispatch(
            execution_hash="e" * 64,
            package_hash="a" * 64,
            lane_id="lane-0",
            lane_family="surface_code",
            hardware_target={
                "target_family": "gpu",
                "target_name": "invalid",
                "target_class": "accelerator",
                "supported_lane_families": ["surface_code"],
                "latency_budget_ns": 100,
                "throughput_budget_ops": 100,
                "metadata": {},
            },
            dispatch_policy="strict_deterministic",
            projected_base_latency_ns=50,
            priority_rank=0,
        )


def test_lane_incompatibility_rejection():
    with pytest.raises(HardwareControlValidationError):
        _build(lane_family="bosonic_concat")


def test_receipt_hash_stable():
    dispatch = _build()
    expected = dispatch.control_receipt.stable_hash()
    assert dispatch.control_receipt.receipt_hash == expected


def test_canonical_json_round_trip():
    dispatch = _build()
    payload = json.loads(dispatch.to_canonical_json())
    assert payload["control_receipt"]["receipt_hash"] == dispatch.control_receipt.receipt_hash
    assert payload["dispatch"]["dispatch_id"] == dispatch.dispatch.dispatch_id


def test_replay_projection_stable():
    dispatch = _build()
    a = shadow_replay_projection(dispatch)
    b = shadow_replay_projection(dispatch)
    assert a == b


def test_validation_tamper_detection():
    dispatch = _build()
    tampered = {
        "target": dispatch.target.to_dict(),
        "dispatch": dispatch.dispatch.to_dict(),
        "latency_receipt": dispatch.latency_receipt.to_dict(),
        "control_receipt": {
            **dispatch.control_receipt.to_dict(),
            "receipt_hash": "0" * 64,
        },
    }
    report = validate_hardware_control_dispatch(tampered)
    assert report.valid is False
    assert "receipt_hash mismatch" in report.errors


def test_priority_rank_changes_dispatch_hash():
    a = _build(priority_rank=0)
    b = _build(priority_rank=1)
    assert a.dispatch.stable_hash() != b.dispatch.stable_hash()
