from __future__ import annotations

import json

import pytest

from qec.analysis.autonomous_anomaly_detection_kernel import (
    AnomalyDecision,
    AnomalySignal,
    AutonomousAnomalyReceipt,
)
from qec.analysis.deterministic_rollback_planning_engine import (
    RollbackCandidate,
    RollbackPlan,
    RollbackPlanReceipt,
)
from qec.analysis.policy_adaptation_kernel import (
    AdaptivePolicyDecision,
    AdaptivePolicySignal,
    PolicyAdaptationReceipt,
)
from qec.analysis.recovery_validation_kernel import (
    RecoveryValidationDecision,
    RecoveryValidationReceipt,
    RecoveryValidationSignal,
)
from qec.analysis.self_healing_integration_bridge import (
    SELF_HEALING_INTEGRATION_BRIDGE_VERSION,
    SelfHealingInputs,
    evaluate_self_healing_integration_bridge,
)


def _anomaly_receipt(*, anomaly_score: float, anomaly_label: str) -> AutonomousAnomalyReceipt:
    escalation_by_label = {
        "nominal": 0,
        "watch": 1,
        "recover": 2,
        "critical": 3,
    }
    return AutonomousAnomalyReceipt(
        version="v141.0",
        signal=AnomalySignal(
            anomaly_score=anomaly_score,
            anomaly_confidence=0.8,
            pressure_component=anomaly_score,
            instability_component=anomaly_score,
            conflict_component=anomaly_score,
            dominant_signal="thermal",
        ),
        decision=AnomalyDecision(
            anomaly_label=anomaly_label,
            recovery_ready=anomaly_label in {"recover", "critical"},
            escalation_rank=escalation_by_label[anomaly_label],
            rationale=f"a::{anomaly_label}",
        ),
        control_mode="autonomous_anomaly_advisory",
        observatory_only=True,
    )


def _rollback_receipt(*, selected_action: str, rollback_strength: float, severity_rank: int, confidence: float) -> RollbackPlanReceipt:
    action_order = ("none", "soft_reset", "partial_rollback", "full_rollback")
    candidates = tuple(
        RollbackCandidate(
            candidate_type=action,
            priority_score=0.25,
            justification=f"j::{action}",
        )
        for action in action_order
    )
    return RollbackPlanReceipt(
        version="v141.1",
        plan=RollbackPlan(
            selected_action=selected_action,
            severity_rank=severity_rank,
            rollback_strength=rollback_strength,
            confidence=confidence,
            candidates=candidates,
        ),
        control_mode="rollback_planning_advisory",
        observatory_only=True,
    )


def _policy_receipt(*, policy_bias: float, control_gain: float, adaptation_label: str) -> PolicyAdaptationReceipt:
    label_to_rank = {
        "hold": 0,
        "tune": 1,
        "constrain": 2,
        "harden": 3,
    }
    return PolicyAdaptationReceipt(
        version="v141.2",
        signal=AdaptivePolicySignal(
            adaptation_pressure=policy_bias,
            rollback_component=policy_bias,
            confidence_component=0.5,
            severity_component=0.5,
            selected_action="soft_reset",
        ),
        decision=AdaptivePolicyDecision(
            adaptation_label=adaptation_label,
            adaptation_rank=label_to_rank[adaptation_label],
            policy_bias=policy_bias,
            control_gain=control_gain,
            rationale=f"p::{adaptation_label}",
        ),
        control_mode="policy_adaptation_advisory",
        observatory_only=True,
    )


def _validation_receipt(
    *,
    validation_pressure: float,
    recovery_alignment: float,
    validation_label: str,
    recovery_viable: bool,
    validation_confidence: float,
) -> RecoveryValidationReceipt:
    rank_by_label = {
        "stable": 0,
        "monitor": 1,
        "validate_recovery": 2,
        "recovery_failed": 3,
    }
    return RecoveryValidationReceipt(
        version="v141.3",
        signal=RecoveryValidationSignal(
            anomaly_component=validation_pressure,
            confidence_component=validation_confidence,
            policy_bias_component=validation_pressure,
            control_gain_component=validation_pressure,
            validation_pressure=validation_pressure,
            recovery_alignment=recovery_alignment,
        ),
        decision=RecoveryValidationDecision(
            validation_label=validation_label,
            validation_rank=rank_by_label[validation_label],
            recovery_viable=recovery_viable,
            validation_confidence=validation_confidence,
            rationale=f"v::{validation_label}",
        ),
        control_mode="recovery_validation_advisory",
        observatory_only=True,
    )


def _inputs(**kwargs: object) -> SelfHealingInputs:
    return SelfHealingInputs(
        anomaly_receipt=_anomaly_receipt(
            anomaly_score=kwargs["anomaly_score"],
            anomaly_label=kwargs["anomaly_label"],
        ),
        rollback_receipt=_rollback_receipt(
            selected_action=kwargs["selected_action"],
            rollback_strength=kwargs["rollback_strength"],
            severity_rank=kwargs["severity_rank"],
            confidence=kwargs["rollback_confidence"],
        ),
        policy_receipt=_policy_receipt(
            policy_bias=kwargs["policy_bias"],
            control_gain=kwargs["control_gain"],
            adaptation_label=kwargs["adaptation_label"],
        ),
        validation_receipt=_validation_receipt(
            validation_pressure=kwargs["validation_pressure"],
            recovery_alignment=kwargs["recovery_alignment"],
            validation_label=kwargs["validation_label"],
            recovery_viable=kwargs["recovery_viable"],
            validation_confidence=kwargs["validation_confidence"],
        ),
    )


def test_deterministic_replay_identical_json_and_hash() -> None:
    inputs = _inputs(
        anomaly_score=0.55,
        anomaly_label="watch",
        selected_action="partial_rollback",
        rollback_strength=0.61,
        severity_rank=2,
        rollback_confidence=0.75,
        policy_bias=0.57,
        control_gain=0.66,
        adaptation_label="constrain",
        validation_pressure=0.58,
        recovery_alignment=0.95,
        validation_label="validate_recovery",
        recovery_viable=True,
        validation_confidence=0.82,
    )

    first = evaluate_self_healing_integration_bridge(inputs)
    second = evaluate_self_healing_integration_bridge(inputs)

    assert first.to_canonical_json() == second.to_canonical_json()
    assert first.stable_hash == second.stable_hash


def test_abort_path_recovery_failed_maps_to_abort() -> None:
    receipt = evaluate_self_healing_integration_bridge(
        _inputs(
            anomaly_score=0.8,
            anomaly_label="critical",
            selected_action="full_rollback",
            rollback_strength=0.9,
            severity_rank=3,
            rollback_confidence=0.6,
            policy_bias=0.4,
            control_gain=0.5,
            adaptation_label="harden",
            validation_pressure=0.9,
            recovery_alignment=0.2,
            validation_label="recovery_failed",
            recovery_viable=False,
            validation_confidence=0.3,
        )
    )

    assert receipt.decision.directive_label == "abort"
    assert receipt.decision.directive_rank == 5


def test_no_action_path_stable_maps_to_no_action() -> None:
    receipt = evaluate_self_healing_integration_bridge(
        _inputs(
            anomaly_score=0.1,
            anomaly_label="nominal",
            selected_action="none",
            rollback_strength=0.05,
            severity_rank=0,
            rollback_confidence=0.9,
            policy_bias=0.1,
            control_gain=0.1,
            adaptation_label="hold",
            validation_pressure=0.1,
            recovery_alignment=1.0,
            validation_label="stable",
            recovery_viable=True,
            validation_confidence=0.95,
        )
    )

    assert receipt.decision.directive_label == "no_action"
    assert receipt.decision.directive_rank == 0


def test_observe_path_monitor_maps_to_observe() -> None:
    receipt = evaluate_self_healing_integration_bridge(
        _inputs(
            anomaly_score=0.3,
            anomaly_label="watch",
            selected_action="soft_reset",
            rollback_strength=0.2,
            severity_rank=1,
            rollback_confidence=0.7,
            policy_bias=0.25,
            control_gain=0.35,
            adaptation_label="tune",
            validation_pressure=0.32,
            recovery_alignment=0.9,
            validation_label="monitor",
            recovery_viable=True,
            validation_confidence=0.75,
        )
    )

    assert receipt.decision.directive_label == "observe"
    assert receipt.decision.directive_rank == 1


def test_apply_paths_validate_recovery_uses_rollback_mapping() -> None:
    soft = evaluate_self_healing_integration_bridge(
        _inputs(
            anomaly_score=0.6,
            anomaly_label="recover",
            selected_action="soft_reset",
            rollback_strength=0.4,
            severity_rank=1,
            rollback_confidence=0.7,
            policy_bias=0.55,
            control_gain=0.6,
            adaptation_label="tune",
            validation_pressure=0.6,
            recovery_alignment=0.95,
            validation_label="validate_recovery",
            recovery_viable=True,
            validation_confidence=0.8,
        )
    )
    partial = evaluate_self_healing_integration_bridge(
        _inputs(
            anomaly_score=0.6,
            anomaly_label="recover",
            selected_action="partial_rollback",
            rollback_strength=0.6,
            severity_rank=2,
            rollback_confidence=0.7,
            policy_bias=0.55,
            control_gain=0.6,
            adaptation_label="constrain",
            validation_pressure=0.6,
            recovery_alignment=0.95,
            validation_label="validate_recovery",
            recovery_viable=True,
            validation_confidence=0.8,
        )
    )
    full = evaluate_self_healing_integration_bridge(
        _inputs(
            anomaly_score=0.6,
            anomaly_label="recover",
            selected_action="full_rollback",
            rollback_strength=0.8,
            severity_rank=3,
            rollback_confidence=0.7,
            policy_bias=0.55,
            control_gain=0.6,
            adaptation_label="harden",
            validation_pressure=0.6,
            recovery_alignment=0.95,
            validation_label="validate_recovery",
            recovery_viable=True,
            validation_confidence=0.8,
        )
    )

    assert soft.decision.directive_label == "apply_soft"
    assert partial.decision.directive_label == "apply_partial"
    assert full.decision.directive_label == "apply_full"


def test_coherence_behavior_aligned_signals_yield_higher_coherence() -> None:
    aligned = evaluate_self_healing_integration_bridge(
        _inputs(
            anomaly_score=0.6,
            anomaly_label="recover",
            selected_action="partial_rollback",
            rollback_strength=0.7,
            severity_rank=2,
            rollback_confidence=0.7,
            policy_bias=0.61,
            control_gain=0.6,
            adaptation_label="constrain",
            validation_pressure=0.69,
            recovery_alignment=0.95,
            validation_label="monitor",
            recovery_viable=True,
            validation_confidence=0.8,
        )
    )
    misaligned = evaluate_self_healing_integration_bridge(
        _inputs(
            anomaly_score=0.6,
            anomaly_label="recover",
            selected_action="partial_rollback",
            rollback_strength=0.7,
            severity_rank=2,
            rollback_confidence=0.7,
            policy_bias=0.1,
            control_gain=0.6,
            adaptation_label="hold",
            validation_pressure=0.2,
            recovery_alignment=0.3,
            validation_label="monitor",
            recovery_viable=True,
            validation_confidence=0.8,
        )
    )

    assert aligned.signal.coherence_score > misaligned.signal.coherence_score


def test_dominant_factor_selects_highest_contributor_with_tie_break() -> None:
    anomaly_dominant = evaluate_self_healing_integration_bridge(
        _inputs(
            anomaly_score=0.8,
            anomaly_label="critical",
            selected_action="soft_reset",
            rollback_strength=0.4,
            severity_rank=1,
            rollback_confidence=0.7,
            policy_bias=0.4,
            control_gain=0.5,
            adaptation_label="tune",
            validation_pressure=0.4,
            recovery_alignment=0.7,
            validation_label="monitor",
            recovery_viable=True,
            validation_confidence=0.8,
        )
    )
    tie_prefers_anomaly = evaluate_self_healing_integration_bridge(
        _inputs(
            anomaly_score=0.7,
            anomaly_label="recover",
            selected_action="partial_rollback",
            rollback_strength=0.7,
            severity_rank=2,
            rollback_confidence=0.7,
            policy_bias=0.7,
            control_gain=0.5,
            adaptation_label="constrain",
            validation_pressure=0.7,
            recovery_alignment=0.9,
            validation_label="monitor",
            recovery_viable=True,
            validation_confidence=0.8,
        )
    )

    assert anomaly_dominant.decision.dominant_factor == "anomaly"
    assert tie_prefers_anomaly.decision.dominant_factor == "anomaly"


def test_confidence_increases_with_validation_confidence_and_coherence() -> None:
    high = evaluate_self_healing_integration_bridge(
        _inputs(
            anomaly_score=0.5,
            anomaly_label="watch",
            selected_action="soft_reset",
            rollback_strength=0.5,
            severity_rank=1,
            rollback_confidence=0.7,
            policy_bias=0.5,
            control_gain=0.5,
            adaptation_label="tune",
            validation_pressure=0.5,
            recovery_alignment=1.0,
            validation_label="monitor",
            recovery_viable=True,
            validation_confidence=0.95,
        )
    )
    low = evaluate_self_healing_integration_bridge(
        _inputs(
            anomaly_score=0.5,
            anomaly_label="watch",
            selected_action="soft_reset",
            rollback_strength=0.5,
            severity_rank=1,
            rollback_confidence=0.7,
            policy_bias=0.1,
            control_gain=0.5,
            adaptation_label="hold",
            validation_pressure=0.1,
            recovery_alignment=0.2,
            validation_label="monitor",
            recovery_viable=True,
            validation_confidence=0.20,
        )
    )

    assert high.decision.integration_confidence > low.decision.integration_confidence


def test_recovery_permission_matches_validation_viability() -> None:
    allowed = evaluate_self_healing_integration_bridge(
        _inputs(
            anomaly_score=0.3,
            anomaly_label="watch",
            selected_action="soft_reset",
            rollback_strength=0.3,
            severity_rank=1,
            rollback_confidence=0.7,
            policy_bias=0.3,
            control_gain=0.3,
            adaptation_label="tune",
            validation_pressure=0.3,
            recovery_alignment=0.9,
            validation_label="monitor",
            recovery_viable=True,
            validation_confidence=0.8,
        )
    )
    denied = evaluate_self_healing_integration_bridge(
        _inputs(
            anomaly_score=0.9,
            anomaly_label="critical",
            selected_action="full_rollback",
            rollback_strength=0.9,
            severity_rank=3,
            rollback_confidence=0.6,
            policy_bias=0.2,
            control_gain=0.4,
            adaptation_label="hold",
            validation_pressure=0.9,
            recovery_alignment=0.2,
            validation_label="recovery_failed",
            recovery_viable=False,
            validation_confidence=0.3,
        )
    )

    assert allowed.decision.recovery_permitted is True
    assert denied.decision.recovery_permitted is False


def test_validation_invalid_inputs_raise_value_error() -> None:
    with pytest.raises(ValueError, match="anomaly_receipt must be AutonomousAnomalyReceipt"):
        SelfHealingInputs(  # type: ignore[arg-type]
            anomaly_receipt="bad",
            rollback_receipt=_rollback_receipt(
                selected_action="none",
                rollback_strength=0.1,
                severity_rank=0,
                confidence=0.9,
            ),
            policy_receipt=_policy_receipt(
                policy_bias=0.1,
                control_gain=0.1,
                adaptation_label="hold",
            ),
            validation_receipt=_validation_receipt(
                validation_pressure=0.1,
                recovery_alignment=0.9,
                validation_label="stable",
                recovery_viable=True,
                validation_confidence=0.9,
            ),
        )

    bad_validation_inputs = _inputs(
        anomaly_score=0.2,
        anomaly_label="watch",
        selected_action="soft_reset",
        rollback_strength=0.2,
        severity_rank=1,
        rollback_confidence=0.7,
        policy_bias=0.2,
        control_gain=0.2,
        adaptation_label="tune",
        validation_pressure=0.2,
        recovery_alignment=0.9,
        validation_label="monitor",
        recovery_viable=True,
        validation_confidence=0.8,
    )
    object.__setattr__(bad_validation_inputs.validation_receipt.decision, "validation_label", "bad")
    with pytest.raises(ValueError, match="invalid validation_label"):
        evaluate_self_healing_integration_bridge(bad_validation_inputs)


def test_hash_stability_repeated_runs_identical() -> None:
    inputs = _inputs(
        anomaly_score=0.62,
        anomaly_label="recover",
        selected_action="partial_rollback",
        rollback_strength=0.64,
        severity_rank=2,
        rollback_confidence=0.73,
        policy_bias=0.63,
        control_gain=0.67,
        adaptation_label="constrain",
        validation_pressure=0.65,
        recovery_alignment=0.96,
        validation_label="validate_recovery",
        recovery_viable=True,
        validation_confidence=0.85,
    )

    hashes = tuple(evaluate_self_healing_integration_bridge(inputs).stable_hash for _ in range(5))
    assert hashes == (hashes[0],) * 5


def test_canonical_serialization_replay_safe() -> None:
    receipt = evaluate_self_healing_integration_bridge(
        _inputs(
            anomaly_score=0.44,
            anomaly_label="watch",
            selected_action="soft_reset",
            rollback_strength=0.41,
            severity_rank=1,
            rollback_confidence=0.72,
            policy_bias=0.46,
            control_gain=0.43,
            adaptation_label="tune",
            validation_pressure=0.45,
            recovery_alignment=0.98,
            validation_label="monitor",
            recovery_viable=True,
            validation_confidence=0.77,
        )
    )

    payload = receipt.to_dict()
    assert tuple(payload.keys()) == (
        "version",
        "signal",
        "decision",
        "control_mode",
        "observatory_only",
        "stable_hash",
    )
    assert payload["version"] == SELF_HEALING_INTEGRATION_BRIDGE_VERSION
    assert receipt.to_canonical_json() == json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
