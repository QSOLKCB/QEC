from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.execution_bridge import (
    EXECUTION_BRIDGE_MODULE_VERSION,
    ExecutionBridgeReceipt,
    ExecutionValidationResult,
    SimulatedActuationRequest,
    SimulatedActuationResult,
    simulate_execution_bridge,
)
from qec.analysis.hardware_alignment_layer import (
    HARDWARE_ALIGNMENT_MODULE_VERSION,
    HardwareAlignmentDecision,
    HardwareAlignmentReceipt,
)


def _hash(label: str) -> str:
    return sha256_hex({"seed": label})


def _decision(
    *,
    signal_id: str,
    hardware_id: str,
    alignment_status: str,
    mapping_score: float,
    selected_capability: str,
    decision_reason: str,
    constraint_violations: tuple[str, ...],
) -> HardwareAlignmentDecision:
    return HardwareAlignmentDecision(
        signal_id=signal_id,
        hardware_id=hardware_id,
        alignment_status=alignment_status,
        mapping_score=mapping_score,
        constraint_violations=constraint_violations,
        selected_capability=selected_capability,
        decision_reason=decision_reason,
    )


def _alignment_receipt(decisions: tuple[HardwareAlignmentDecision, ...]) -> HardwareAlignmentReceipt:
    sorted_decisions = tuple(sorted(decisions, key=lambda d: (d.signal_id, d.hardware_id)))
    signal_count = len({d.signal_id for d in sorted_decisions})
    hardware_count = len({d.hardware_id for d in sorted_decisions})
    aligned_count = sum(1 for d in sorted_decisions if d.alignment_status == "ALIGNED")
    degraded_count = sum(1 for d in sorted_decisions if d.alignment_status == "DEGRADED_ALIGNMENT")
    blocked_count = len(sorted_decisions) - aligned_count - degraded_count
    overall_score = round(sum(d.mapping_score for d in sorted_decisions) / len(sorted_decisions), 12) if sorted_decisions else 0.0
    return HardwareAlignmentReceipt(
        module_version=HARDWARE_ALIGNMENT_MODULE_VERSION,
        signal_count=signal_count,
        hardware_profile_count=hardware_count,
        alignment_decisions=sorted_decisions,
        aligned_count=aligned_count,
        degraded_count=degraded_count,
        blocked_count=blocked_count,
        overall_alignment_score=overall_score,
    )


def test_empty_alignment_returns_empty_receipt() -> None:
    alignment = HardwareAlignmentReceipt(
        module_version=HARDWARE_ALIGNMENT_MODULE_VERSION,
        signal_count=0,
        hardware_profile_count=0,
        alignment_decisions=tuple(),
        aligned_count=0,
        degraded_count=0,
        blocked_count=0,
        overall_alignment_score=0.0,
    )
    receipt = simulate_execution_bridge(alignment)
    assert receipt.module_version == EXECUTION_BRIDGE_MODULE_VERSION
    assert receipt.request_count == 0
    assert receipt.actuation_results == tuple()
    assert receipt.validation_results == tuple()
    assert receipt.overall_consistency_score == 0.0


@pytest.mark.parametrize(
    ("alignment_status", "decision_reason", "constraint_violations"),
    (
        ("REPLAY_UNSUPPORTED", "replay_not_supported", ("replay_not_supported",)),
        ("CAPABILITY_MISMATCH", "capability_missing", ("capability_missing",)),
        ("PRIORITY_EXCEEDED", "priority_exceeded", ("priority_exceeded",)),
    ),
)
def test_blocked_alignment_produces_simulated_blocked(
    alignment_status: str,
    decision_reason: str,
    constraint_violations: tuple[str, ...],
) -> None:
    alignment = _alignment_receipt(
        (
            _decision(
                signal_id="s-1",
                hardware_id="h-1",
                alignment_status=alignment_status,
                mapping_score=0.0,
                selected_capability="NONE" if alignment_status != "PRIORITY_EXCEEDED" else "cap-alpha",
                decision_reason=decision_reason,
                constraint_violations=constraint_violations,
            ),
        )
    )
    receipt = simulate_execution_bridge(alignment)
    actuation = receipt.actuation_results[0]
    validation = receipt.validation_results[0]
    assert actuation.actuation_status == "SIMULATED_BLOCKED"
    assert actuation.effect_strength == 0.0
    assert actuation.effect_class == "none"
    assert validation.validation_status == "VALID"
    assert validation.consistency_score == 0.0


def test_no_op_actuation_result_is_supported() -> None:
    simulation_hash = sha256_hex(
        {
            "signal_id": "s-1",
            "hardware_id": "h-1",
            "actuation_status": "NO_OP",
            "effect_strength": 0.0,
            "effect_class": "none",
        }
    )
    result = SimulatedActuationResult(
        signal_id="s-1",
        hardware_id="h-1",
        actuation_status="NO_OP",
        effect_strength=0.0,
        effect_class="none",
        simulation_hash=simulation_hash,
    )
    assert result.actuation_status == "NO_OP"
    assert result.effect_strength == 0.0
    assert result.effect_class == "none"


def test_aligned_alignment_produces_simulated_success() -> None:
    alignment = _alignment_receipt(
        (
            _decision(
                signal_id="s-1",
                hardware_id="h-1",
                alignment_status="ALIGNED",
                mapping_score=0.8,
                selected_capability="cap-alpha",
                decision_reason="aligned",
                constraint_violations=tuple(),
            ),
        )
    )
    receipt = simulate_execution_bridge(alignment)
    actuation = receipt.actuation_results[0]
    validation = receipt.validation_results[0]
    assert actuation.actuation_status == "SIMULATED_SUCCESS"
    assert actuation.effect_strength == 0.8
    assert actuation.effect_class == "aligned"
    assert validation.validation_status == "VALID"
    assert validation.consistency_score == 0.8


def test_degraded_alignment_produces_simulated_degraded() -> None:
    alignment = _alignment_receipt(
        (
            _decision(
                signal_id="s-1",
                hardware_id="h-1",
                alignment_status="DEGRADED_ALIGNMENT",
                mapping_score=0.61,
                selected_capability="cap-alpha",
                decision_reason="degraded_but_usable",
                constraint_violations=("low_stability",),
            ),
        )
    )
    receipt = simulate_execution_bridge(alignment)
    actuation = receipt.actuation_results[0]
    assert actuation.actuation_status == "SIMULATED_DEGRADED"
    assert actuation.effect_strength == 0.61
    assert actuation.effect_class == "degraded"


def test_unstable_hardware_produces_simulated_degraded() -> None:
    alignment = _alignment_receipt(
        (
            _decision(
                signal_id="s-1",
                hardware_id="h-1",
                alignment_status="UNSTABLE_HARDWARE",
                mapping_score=0.4,
                selected_capability="cap-alpha",
                decision_reason="stability_below_tolerance",
                constraint_violations=("stability_below_tolerance",),
            ),
        )
    )
    receipt = simulate_execution_bridge(alignment)
    actuation = receipt.actuation_results[0]
    validation = receipt.validation_results[0]
    assert actuation.actuation_status == "SIMULATED_DEGRADED"
    assert actuation.effect_strength == 0.4
    assert actuation.effect_class == "unstable"
    assert validation.validation_status == "VALID"


def test_deterministic_ordering_independent_of_input_order() -> None:
    decisions = (
        _decision(
            signal_id="s-2",
            hardware_id="h-2",
            alignment_status="ALIGNED",
            mapping_score=0.9,
            selected_capability="cap-alpha",
            decision_reason="aligned",
            constraint_violations=tuple(),
        ),
        _decision(
            signal_id="s-2",
            hardware_id="h-1",
            alignment_status="REPLAY_UNSUPPORTED",
            mapping_score=0.0,
            selected_capability="NONE",
            decision_reason="replay_not_supported",
            constraint_violations=("replay_not_supported",),
        ),
        _decision(
            signal_id="s-1",
            hardware_id="h-2",
            alignment_status="ALIGNED",
            mapping_score=0.6,
            selected_capability="cap-alpha",
            decision_reason="aligned",
            constraint_violations=tuple(),
        ),
        _decision(
            signal_id="s-1",
            hardware_id="h-1",
            alignment_status="DEGRADED_ALIGNMENT",
            mapping_score=0.3,
            selected_capability="cap-alpha",
            decision_reason="degraded_but_usable",
            constraint_violations=("high_latency",),
        ),
    )
    left = simulate_execution_bridge(_alignment_receipt(decisions))
    right = simulate_execution_bridge(_alignment_receipt(tuple(reversed(decisions))))
    assert tuple((r.signal_id, r.hardware_id) for r in left.actuation_results) == (
        ("s-1", "h-1"),
        ("s-1", "h-2"),
        ("s-2", "h-1"),
        ("s-2", "h-2"),
    )
    assert left.to_dict() == right.to_dict()
    assert left.stable_hash() == right.stable_hash()


def test_invalid_alignment_status_rejected_in_request() -> None:
    with pytest.raises(ValueError, match="alignment_status is not supported"):
        SimulatedActuationRequest(
            signal_id="s-1",
            hardware_id="h-1",
            alignment_status="UNKNOWN",
            selected_capability="cap-alpha",
            mapping_score=0.2,
            input_hash=_hash("input"),
        )


def test_nan_mapping_score_rejected_in_request() -> None:
    with pytest.raises(ValueError, match="mapping_score must be bounded"):
        SimulatedActuationRequest(
            signal_id="s-1",
            hardware_id="h-1",
            alignment_status="ALIGNED",
            selected_capability="cap-alpha",
            mapping_score=float("nan"),
            input_hash=_hash("input"),
        )


def test_blocked_status_requires_zero_mapping_score_in_request() -> None:
    with pytest.raises(ValueError, match="mapping_score must be 0.0 for blocked hard-failure statuses"):
        SimulatedActuationRequest(
            signal_id="s-1",
            hardware_id="h-1",
            alignment_status="CAPABILITY_MISMATCH",
            selected_capability="NONE",
            mapping_score=0.1,
            input_hash=_hash("input"),
        )


def test_replay_unsupported_requires_none_capability_in_request() -> None:
    with pytest.raises(ValueError, match="selected_capability must be NONE"):
        SimulatedActuationRequest(
            signal_id="s-1",
            hardware_id="h-1",
            alignment_status="REPLAY_UNSUPPORTED",
            selected_capability="cap-alpha",
            mapping_score=0.0,
            input_hash=_hash("input"),
        )


def test_aligned_requires_non_none_capability_in_request() -> None:
    with pytest.raises(ValueError, match="selected_capability must be a capability token"):
        SimulatedActuationRequest(
            signal_id="s-1",
            hardware_id="h-1",
            alignment_status="ALIGNED",
            selected_capability="NONE",
            mapping_score=0.8,
            input_hash=_hash("input"),
        )


def test_success_result_requires_aligned_effect_class() -> None:
    simulation_hash = sha256_hex(
        {
            "signal_id": "s-1",
            "hardware_id": "h-1",
            "actuation_status": "SIMULATED_SUCCESS",
            "effect_strength": 0.3,
            "effect_class": "degraded",
        }
    )
    with pytest.raises(ValueError, match="effect_class must be aligned"):
        SimulatedActuationResult(
            signal_id="s-1",
            hardware_id="h-1",
            actuation_status="SIMULATED_SUCCESS",
            effect_strength=0.3,
            effect_class="degraded",
            simulation_hash=simulation_hash,
        )


def test_blocked_result_requires_none_and_zero_effect() -> None:
    simulation_hash = sha256_hex(
        {
            "signal_id": "s-1",
            "hardware_id": "h-1",
            "actuation_status": "SIMULATED_BLOCKED",
            "effect_strength": 0.1,
            "effect_class": "none",
        }
    )
    with pytest.raises(ValueError, match="effect_strength must be 0.0"):
        SimulatedActuationResult(
            signal_id="s-1",
            hardware_id="h-1",
            actuation_status="SIMULATED_BLOCKED",
            effect_strength=0.1,
            effect_class="none",
            simulation_hash=simulation_hash,
        )

def test_validation_mismatch_can_be_encoded_as_inconsistent() -> None:
    validation = ExecutionValidationResult(
        signal_id="s-1",
        hardware_id="h-1",
        validation_status="INCONSISTENT",
        consistency_score=0.0,
        validation_reason="status_mapping_mismatch",
    )
    assert validation.validation_status == "INCONSISTENT"
    assert validation.consistency_score == 0.0


def test_frozen_immutability() -> None:
    alignment = _alignment_receipt(
        (
            _decision(
                signal_id="s-1",
                hardware_id="h-1",
                alignment_status="ALIGNED",
                mapping_score=0.8,
                selected_capability="cap-alpha",
                decision_reason="aligned",
                constraint_violations=tuple(),
            ),
        )
    )
    receipt = simulate_execution_bridge(alignment)
    with pytest.raises(FrozenInstanceError):
        receipt.request_count = 3


def test_canonical_json_stability() -> None:
    alignment = _alignment_receipt(
        (
            _decision(
                signal_id="s-1",
                hardware_id="h-1",
                alignment_status="ALIGNED",
                mapping_score=0.5,
                selected_capability="cap-alpha",
                decision_reason="aligned",
                constraint_violations=tuple(),
            ),
        )
    )
    receipt_a = simulate_execution_bridge(alignment)
    receipt_b = ExecutionBridgeReceipt(
        module_version=receipt_a.module_version,
        request_count=receipt_a.request_count,
        actuation_results=receipt_a.actuation_results,
        validation_results=receipt_a.validation_results,
        overall_consistency_score=receipt_a.overall_consistency_score,
        stable_hash_input=receipt_a.stable_hash(),
    )
    assert receipt_a.to_canonical_json() == receipt_b.to_canonical_json()
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_hash_replay_stability_for_actuation_result() -> None:
    simulation_hash = sha256_hex(
        {
            "signal_id": "s-1",
            "hardware_id": "h-1",
            "actuation_status": "SIMULATED_SUCCESS",
            "effect_strength": 0.25,
            "effect_class": "aligned",
        }
    )
    result_a = SimulatedActuationResult(
        signal_id="s-1",
        hardware_id="h-1",
        actuation_status="SIMULATED_SUCCESS",
        effect_strength=0.25,
        effect_class="aligned",
        simulation_hash=simulation_hash,
    )
    result_b = SimulatedActuationResult(
        signal_id=result_a.signal_id,
        hardware_id=result_a.hardware_id,
        actuation_status=result_a.actuation_status,
        effect_strength=result_a.effect_strength,
        effect_class=result_a.effect_class,
        simulation_hash=result_a.simulation_hash,
        stable_hash_input=result_a.stable_hash(),
    )
    assert result_a.to_canonical_json() == result_b.to_canonical_json()
    assert result_a.stable_hash() == result_b.stable_hash()
