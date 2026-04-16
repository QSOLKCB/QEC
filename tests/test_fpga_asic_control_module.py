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


def test_validation_missing_lane_family_returns_invalid_report():
    dispatch = _build()
    tampered = {
        "target": dispatch.target.to_dict(),
        "dispatch": {
            **dispatch.dispatch.to_dict(),
            "metadata": {"region": "us-east"},
        },
        "latency_receipt": dispatch.latency_receipt.to_dict(),
        "control_receipt": dispatch.control_receipt.to_dict(),
    }
    report = validate_hardware_control_dispatch(tampered)
    assert report.valid is False
    assert "dispatch.metadata.lane_family must be non-empty" in report.errors


def test_validation_rejects_non_boolean_latency_within_budget():
    dispatch = _build()
    tampered = {
        "target": dispatch.target.to_dict(),
        "dispatch": dispatch.dispatch.to_dict(),
        "latency_receipt": {
            **dispatch.latency_receipt.to_dict(),
            "within_budget": "false",
        },
        "control_receipt": dispatch.control_receipt.to_dict(),
    }
    report = validate_hardware_control_dispatch(tampered)
    assert report.valid is False
    assert len(report.errors) == 1
    assert report.errors[0].startswith("normalization_failed: latency_receipt.within_budget must be a boolean")


def test_build_rejects_short_execution_hash():
    with pytest.raises(HardwareControlValidationError, match="execution_hash"):
        build_hardware_control_dispatch(
            execution_hash="short",
            package_hash="a" * 64,
            lane_id="lane-0",
            lane_family="surface_code",
            hardware_target=_target(),
            dispatch_policy="strict_deterministic",
            projected_base_latency_ns=50,
            priority_rank=0,
        )


def test_build_rejects_non_hex_package_hash():
    with pytest.raises(HardwareControlValidationError, match="package_hash"):
        build_hardware_control_dispatch(
            execution_hash="e" * 64,
            package_hash="z" * 64,  # 'z' is not a hex character
            lane_id="lane-0",
            lane_family="surface_code",
            hardware_target=_target(),
            dispatch_policy="strict_deterministic",
            projected_base_latency_ns=50,
            priority_rank=0,
        )


def test_normalize_target_rejects_string_lane_families():
    with pytest.raises(HardwareControlValidationError, match="supported_lane_families"):
        build_hardware_control_dispatch(
            execution_hash="e" * 64,
            package_hash="a" * 64,
            lane_id="lane-0",
            lane_family="s",
            hardware_target={
                "target_family": "fpga",
                "target_name": "test",
                "target_class": "accelerator",
                "supported_lane_families": "surface_code",  # bare string instead of sequence
                "latency_budget_ns": 100,
                "throughput_budget_ops": 100,
                "metadata": {},
            },
            dispatch_policy="strict_deterministic",
            projected_base_latency_ns=50,
            priority_rank=0,
        )


def test_validation_detects_target_family_mismatch():
    """dispatch.target_family that differs from target.target_family must be caught."""
    dispatch = _build()
    tampered_dispatch_dict = {
        **dispatch.dispatch.to_dict(),
        "target_family": "asic",  # target is "fpga", dispatch says "asic"
    }
    report = validate_hardware_control_dispatch(
        {
            "target": dispatch.target.to_dict(),
            "dispatch": tampered_dispatch_dict,
            "latency_receipt": dispatch.latency_receipt.to_dict(),
            "control_receipt": dispatch.control_receipt.to_dict(),
        }
    )
    assert report.valid is False
    assert any("target_family" in e for e in report.errors)


def test_validation_detects_dispatch_id_tampering():
    """Tampering dispatch_id (even with recomputed downstream hashes) must be caught."""
    from qec.runtime.fpga_asic_control_module import (
        ControlDispatchIntent,
        HardwareControlDispatch,
        HardwareControlReceipt,
        _stable_hash,
    )

    dispatch = _build()

    # Build a tampered dispatch intent with a different dispatch_id but consistent downstream hashes
    tampered_intent = ControlDispatchIntent(
        dispatch_id="d" * 64,
        lane_id=dispatch.dispatch.lane_id,
        package_hash=dispatch.dispatch.package_hash,
        execution_hash=dispatch.dispatch.execution_hash,
        target_family=dispatch.dispatch.target_family,
        dispatch_policy=dispatch.dispatch.dispatch_policy,
        priority_rank=dispatch.dispatch.priority_rank,
        metadata=dict(dispatch.dispatch.metadata),
    )
    new_dispatch_hash = tampered_intent.stable_hash()
    new_control_payload = {
        "dispatch_hash": new_dispatch_hash,
        "target_hash": dispatch.target.stable_hash(),
        "latency_hash": dispatch.latency_receipt.latency_hash,
        "validation_passed": True,
    }
    tampered_control = HardwareControlReceipt(
        dispatch_hash=new_dispatch_hash,
        target_hash=dispatch.target.stable_hash(),
        latency_hash=dispatch.latency_receipt.latency_hash,
        validation_passed=True,
        receipt_hash=_stable_hash(new_control_payload),
    )
    tampered_obj = HardwareControlDispatch(
        target=dispatch.target,
        dispatch=tampered_intent,
        latency_receipt=dispatch.latency_receipt,
        control_receipt=tampered_control,
    )
    report = validate_hardware_control_dispatch(tampered_obj)
    assert report.valid is False
    assert any("dispatch_id" in e for e in report.errors)


def test_validation_detects_within_budget_tampering():
    """Flipping within_budget (with recomputed latency_hash and receipt_hash) must be caught."""
    from qec.runtime.fpga_asic_control_module import (
        HardwareControlDispatch,
        HardwareControlReceipt,
        LatencyTruthReceipt,
        _stable_hash,
    )

    dispatch = _build(budget=200)  # projected=60, within_budget=True
    assert dispatch.latency_receipt.within_budget is True

    tampered_latency = LatencyTruthReceipt(
        latency_budget_ns=dispatch.latency_receipt.latency_budget_ns,
        projected_dispatch_ns=dispatch.latency_receipt.projected_dispatch_ns,
        within_budget=False,  # flipped — semantically invalid
        latency_hash=_stable_hash({
            "latency_budget_ns": dispatch.latency_receipt.latency_budget_ns,
            "projected_dispatch_ns": dispatch.latency_receipt.projected_dispatch_ns,
            "within_budget": False,
        }),
    )
    new_control_payload = {
        "dispatch_hash": dispatch.control_receipt.dispatch_hash,
        "target_hash": dispatch.control_receipt.target_hash,
        "latency_hash": tampered_latency.latency_hash,
        "validation_passed": True,
    }
    tampered_control = HardwareControlReceipt(
        dispatch_hash=dispatch.control_receipt.dispatch_hash,
        target_hash=dispatch.control_receipt.target_hash,
        latency_hash=tampered_latency.latency_hash,
        validation_passed=True,
        receipt_hash=_stable_hash(new_control_payload),
    )
    tampered_obj = HardwareControlDispatch(
        target=dispatch.target,
        dispatch=dispatch.dispatch,
        latency_receipt=tampered_latency,
        control_receipt=tampered_control,
    )
    report = validate_hardware_control_dispatch(tampered_obj)
    assert report.valid is False
    assert any("within_budget" in e for e in report.errors)
