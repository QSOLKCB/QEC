from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.hardware_alignment_layer import (
    HARDWARE_ALIGNMENT_MODULE_VERSION,
    ControlSignalIntent,
    HardwareAlignmentDecision,
    HardwareAlignmentReceipt,
    HardwareConstraintProfile,
    align_control_signals_to_hardware,
)


def _hash(label: str) -> str:
    return sha256_hex({"seed": label})


def _signal(
    *,
    signal_id: str = "s-1",
    recommendation: str = "MAINTAIN_POLICY",
    control_priority: int = 2,
    control_confidence: float = 0.9,
    required_capability: str = "cap-alpha",
    risk_tolerance: float = 0.2,
) -> ControlSignalIntent:
    return ControlSignalIntent(
        signal_id=signal_id,
        source_receipt_hash=_hash(f"src::{signal_id}"),
        recommendation=recommendation,
        control_priority=control_priority,
        control_confidence=control_confidence,
        required_capability=required_capability,
        risk_tolerance=risk_tolerance,
    )


def _hardware(
    *,
    hardware_id: str = "h-1",
    capabilities: tuple[str, ...] = ("cap-alpha",),
    max_control_priority: int = 3,
    stability_rating: float = 0.9,
    latency_class: str = "LOW",
    replay_supported: bool = True,
) -> HardwareConstraintProfile:
    return HardwareConstraintProfile(
        hardware_id=hardware_id,
        capabilities=capabilities,
        max_control_priority=max_control_priority,
        stability_rating=stability_rating,
        latency_class=latency_class,
        replay_supported=replay_supported,
        constraint_hash=_hash(f"constraint::{hardware_id}"),
    )


def test_empty_inputs_return_empty_receipt() -> None:
    receipt = align_control_signals_to_hardware((), ())
    assert receipt.module_version == HARDWARE_ALIGNMENT_MODULE_VERSION
    assert receipt.signal_count == 0
    assert receipt.hardware_profile_count == 0
    assert receipt.alignment_decisions == tuple()
    assert receipt.aligned_count == 0
    assert receipt.degraded_count == 0
    assert receipt.blocked_count == 0
    assert receipt.overall_alignment_score == 0.0


def test_fully_aligned_signal_profile() -> None:
    signal = _signal(control_confidence=0.8, risk_tolerance=0.1)
    hardware = _hardware(stability_rating=0.95, latency_class="MEDIUM")
    receipt = align_control_signals_to_hardware((signal,), (hardware,))
    decision = receipt.alignment_decisions[0]
    assert decision.alignment_status == "ALIGNED"
    assert decision.mapping_score == 0.88
    assert decision.constraint_violations == tuple()
    assert decision.selected_capability == "cap-alpha"
    assert decision.decision_reason == "aligned"


def test_replay_unsupported_blocks_alignment() -> None:
    receipt = align_control_signals_to_hardware((_signal(),), (_hardware(replay_supported=False),))
    decision = receipt.alignment_decisions[0]
    assert decision.alignment_status == "REPLAY_UNSUPPORTED"
    assert decision.mapping_score == 0.0
    assert decision.selected_capability == "NONE"
    assert decision.decision_reason == "replay_not_supported"


def test_capability_mismatch_blocks_alignment() -> None:
    receipt = align_control_signals_to_hardware((_signal(required_capability="cap-z"),), (_hardware(capabilities=("cap-alpha", "cap-beta")),))
    decision = receipt.alignment_decisions[0]
    assert decision.alignment_status == "CAPABILITY_MISMATCH"
    assert decision.mapping_score == 0.0
    assert decision.selected_capability == "NONE"
    assert decision.decision_reason == "capability_missing"


def test_priority_exceeded_blocks_alignment() -> None:
    receipt = align_control_signals_to_hardware((_signal(control_priority=5),), (_hardware(max_control_priority=4),))
    decision = receipt.alignment_decisions[0]
    assert decision.alignment_status == "PRIORITY_EXCEEDED"
    assert decision.mapping_score == 0.0
    assert decision.selected_capability == "cap-alpha"
    assert decision.decision_reason == "priority_exceeded"


def test_unstable_hardware_blocks_alignment() -> None:
    receipt = align_control_signals_to_hardware((_signal(risk_tolerance=0.7),), (_hardware(stability_rating=0.61),))
    decision = receipt.alignment_decisions[0]
    assert decision.alignment_status == "UNSTABLE_HARDWARE"
    assert decision.mapping_score == 0.61
    assert decision.selected_capability == "cap-alpha"
    assert decision.decision_reason == "stability_below_tolerance"


def test_degraded_alignment_from_high_latency() -> None:
    receipt = align_control_signals_to_hardware((_signal(),), (_hardware(latency_class="HIGH", stability_rating=0.95),))
    decision = receipt.alignment_decisions[0]
    assert decision.alignment_status == "DEGRADED_ALIGNMENT"
    assert decision.constraint_violations == ("high_latency",)
    assert decision.decision_reason == "degraded_but_usable"


def test_degraded_alignment_from_low_stability() -> None:
    receipt = align_control_signals_to_hardware((_signal(risk_tolerance=0.5),), (_hardware(stability_rating=0.74, latency_class="LOW"),))
    decision = receipt.alignment_decisions[0]
    assert decision.alignment_status == "DEGRADED_ALIGNMENT"
    assert decision.constraint_violations == ("low_stability",)
    assert decision.decision_reason == "degraded_but_usable"


def test_deterministic_ordering_independent_of_input_order() -> None:
    signals = (
        _signal(signal_id="s-2", required_capability="cap-beta"),
        _signal(signal_id="s-1", required_capability="cap-alpha"),
    )
    hardware = (
        _hardware(hardware_id="h-2", capabilities=("cap-alpha", "cap-beta")),
        _hardware(hardware_id="h-1", capabilities=("cap-alpha", "cap-beta")),
    )
    left = align_control_signals_to_hardware(signals, hardware)
    right = align_control_signals_to_hardware(tuple(reversed(signals)), tuple(reversed(hardware)))
    assert tuple((d.signal_id, d.hardware_id) for d in left.alignment_decisions) == (
        ("s-1", "h-1"),
        ("s-1", "h-2"),
        ("s-2", "h-1"),
        ("s-2", "h-2"),
    )
    assert left.to_dict() == right.to_dict()
    assert left.stable_hash() == right.stable_hash()


def test_duplicate_signal_ids_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate signal_id"):
        align_control_signals_to_hardware(
            (_signal(signal_id="dup"), _signal(signal_id="dup", required_capability="cap-beta")),
            (_hardware(capabilities=("cap-alpha", "cap-beta")),),
        )


def test_duplicate_hardware_ids_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate hardware_id"):
        align_control_signals_to_hardware(
            (_signal(),),
            (_hardware(hardware_id="dup"), _hardware(hardware_id="dup", capabilities=("cap-alpha", "cap-beta"))),
        )


def test_invalid_sha256_rejected() -> None:
    with pytest.raises(ValueError, match="source_receipt_hash must be a valid SHA-256 hex"):
        ControlSignalIntent(
            signal_id="s-1",
            source_receipt_hash="xyz",
            recommendation="MAINTAIN_POLICY",
            control_priority=0,
            control_confidence=0.5,
            required_capability="cap-alpha",
            risk_tolerance=0.1,
        )


def test_invalid_recommendation_rejected() -> None:
    with pytest.raises(ValueError, match="supported governance label"):
        _signal(recommendation="DO_NOTHING")


def test_invalid_capability_token_rejected() -> None:
    with pytest.raises(ValueError, match="required_capability must be a non-empty canonical string"):
        _signal(required_capability=" cap-alpha")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_confidence_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="control_confidence must be bounded"):
        _signal(control_confidence=value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_stability_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="stability_rating must be bounded"):
        _hardware(stability_rating=value)


def test_receipt_count_mismatch_rejected() -> None:
    decision = HardwareAlignmentDecision(
        signal_id="s-1",
        hardware_id="h-1",
        alignment_status="ALIGNED",
        mapping_score=0.7,
        constraint_violations=tuple(),
        selected_capability="cap-alpha",
        decision_reason="aligned",
    )
    with pytest.raises(ValueError, match="aligned/degraded/blocked counts must match"):
        HardwareAlignmentReceipt(
            module_version=HARDWARE_ALIGNMENT_MODULE_VERSION,
            signal_count=1,
            hardware_profile_count=1,
            alignment_decisions=(decision,),
            aligned_count=0,
            degraded_count=0,
            blocked_count=1,
            overall_alignment_score=0.7,
        )


def test_duplicate_decision_pair_rejected() -> None:
    decision = HardwareAlignmentDecision(
        signal_id="s-1",
        hardware_id="h-1",
        alignment_status="ALIGNED",
        mapping_score=0.7,
        constraint_violations=tuple(),
        selected_capability="cap-alpha",
        decision_reason="aligned",
    )
    with pytest.raises(ValueError, match=r"duplicate \(signal_id, hardware_id\)"):
        HardwareAlignmentReceipt(
            module_version=HARDWARE_ALIGNMENT_MODULE_VERSION,
            signal_count=1,
            hardware_profile_count=2,
            alignment_decisions=(decision, decision),
            aligned_count=2,
            degraded_count=0,
            blocked_count=0,
            overall_alignment_score=0.7,
        )


def test_frozen_dataclass_immutability() -> None:
    receipt = align_control_signals_to_hardware((_signal(),), (_hardware(),))
    with pytest.raises(FrozenInstanceError):
        receipt.signal_count = 99


def test_canonical_json_hash_replay_stability() -> None:
    receipt_a = align_control_signals_to_hardware((_signal(),), (_hardware(),))
    receipt_b = HardwareAlignmentReceipt(
        module_version=receipt_a.module_version,
        signal_count=receipt_a.signal_count,
        hardware_profile_count=receipt_a.hardware_profile_count,
        alignment_decisions=receipt_a.alignment_decisions,
        aligned_count=receipt_a.aligned_count,
        degraded_count=receipt_a.degraded_count,
        blocked_count=receipt_a.blocked_count,
        overall_alignment_score=receipt_a.overall_alignment_score,
        stable_hash_input=receipt_a.stable_hash(),
    )
    assert receipt_a.to_canonical_json() == receipt_b.to_canonical_json()
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()
    assert receipt_a.stable_hash() == receipt_b.stable_hash()
