from __future__ import annotations

import json

import pytest

from qec.analysis.autonomous_anomaly_detection_kernel import (
    AnomalyDecision,
    AnomalySignal,
    AutonomousAnomalyReceipt,
)
from qec.analysis.policy_adaptation_kernel import (
    AdaptivePolicyDecision,
    AdaptivePolicySignal,
    PolicyAdaptationReceipt,
)
from qec.analysis.recovery_validation_kernel import (
    RECOVERY_VALIDATION_KERNEL_VERSION,
    RecoveryValidationInputs,
    evaluate_recovery_validation_kernel,
)


def _anomaly_receipt(
    *,
    anomaly_label: str,
    anomaly_score: float,
    anomaly_confidence: float,
    recovery_ready: bool,
) -> AutonomousAnomalyReceipt:
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
            anomaly_confidence=anomaly_confidence,
            pressure_component=anomaly_score,
            instability_component=anomaly_score,
            conflict_component=anomaly_score,
            dominant_signal="thermal",
        ),
        decision=AnomalyDecision(
            anomaly_label=anomaly_label,
            recovery_ready=recovery_ready,
            escalation_rank=escalation_by_label[anomaly_label],
            rationale=f"a::{anomaly_label}",
        ),
        control_mode="autonomous_anomaly_advisory",
        observatory_only=True,
    )


def _policy_receipt(
    *,
    adaptation_label: str,
    policy_bias: float,
    control_gain: float,
    adaptation_pressure: float,
) -> PolicyAdaptationReceipt:
    label_to_rank = {
        "hold": 0,
        "tune": 1,
        "constrain": 2,
        "harden": 3,
    }
    return PolicyAdaptationReceipt(
        version="v141.2",
        signal=AdaptivePolicySignal(
            adaptation_pressure=adaptation_pressure,
            rollback_component=adaptation_pressure,
            confidence_component=1.0 - adaptation_pressure,
            severity_component=adaptation_pressure,
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


def _inputs(**kwargs: object) -> RecoveryValidationInputs:
    return RecoveryValidationInputs(
        anomaly_receipt=_anomaly_receipt(
            anomaly_label=kwargs["anomaly_label"],
            anomaly_score=kwargs["anomaly_score"],
            anomaly_confidence=kwargs["anomaly_confidence"],
            recovery_ready=kwargs["recovery_ready"],
        ),
        policy_receipt=_policy_receipt(
            adaptation_label=kwargs["adaptation_label"],
            policy_bias=kwargs["policy_bias"],
            control_gain=kwargs["control_gain"],
            adaptation_pressure=kwargs["adaptation_pressure"],
        ),
    )


def test_deterministic_replay_identical_json_and_hash() -> None:
    inputs = _inputs(
        anomaly_label="watch",
        anomaly_score=0.45,
        anomaly_confidence=0.82,
        recovery_ready=True,
        adaptation_label="tune",
        policy_bias=0.46,
        control_gain=0.50,
        adaptation_pressure=0.30,
    )

    first = evaluate_recovery_validation_kernel(inputs)
    second = evaluate_recovery_validation_kernel(inputs)

    assert first.to_canonical_json() == second.to_canonical_json()
    assert first.stable_hash == second.stable_hash


def test_stable_path_low_anomaly_and_not_recovery_ready() -> None:
    receipt = evaluate_recovery_validation_kernel(
        _inputs(
            anomaly_label="nominal",
            anomaly_score=0.05,
            anomaly_confidence=0.95,
            recovery_ready=False,
            adaptation_label="hold",
            policy_bias=0.05,
            control_gain=0.05,
            adaptation_pressure=0.05,
        )
    )

    assert receipt.decision.validation_label == "stable"
    assert receipt.decision.validation_rank == 0


def test_monitor_path_moderate_pressure() -> None:
    receipt = evaluate_recovery_validation_kernel(
        _inputs(
            anomaly_label="watch",
            anomaly_score=0.40,
            anomaly_confidence=0.80,
            recovery_ready=True,
            adaptation_label="tune",
            policy_bias=0.40,
            control_gain=0.40,
            adaptation_pressure=0.40,
        )
    )

    assert receipt.signal.validation_pressure < 0.50
    assert receipt.decision.validation_label == "monitor"
    assert receipt.decision.validation_rank == 1


def test_validate_recovery_path_elevated_pressure_with_alignment() -> None:
    receipt = evaluate_recovery_validation_kernel(
        _inputs(
            anomaly_label="recover",
            anomaly_score=0.65,
            anomaly_confidence=0.70,
            recovery_ready=True,
            adaptation_label="constrain",
            policy_bias=0.67,
            control_gain=0.70,
            adaptation_pressure=0.68,
        )
    )

    assert 0.50 <= receipt.signal.validation_pressure < 0.75
    assert receipt.signal.recovery_alignment >= 0.25
    assert receipt.decision.validation_label == "validate_recovery"
    assert receipt.decision.validation_rank == 2


def test_recovery_failed_path_critical_and_poor_alignment() -> None:
    receipt = evaluate_recovery_validation_kernel(
        _inputs(
            anomaly_label="critical",
            anomaly_score=0.90,
            anomaly_confidence=0.80,
            recovery_ready=True,
            adaptation_label="hold",
            policy_bias=0.10,
            control_gain=0.40,
            adaptation_pressure=0.30,
        )
    )

    assert receipt.signal.recovery_alignment < 0.25
    assert receipt.decision.validation_label == "recovery_failed"
    assert receipt.decision.validation_rank == 3


def test_recovery_viability_false_only_for_recovery_failed() -> None:
    stable = evaluate_recovery_validation_kernel(
        _inputs(
            anomaly_label="nominal",
            anomaly_score=0.05,
            anomaly_confidence=0.95,
            recovery_ready=False,
            adaptation_label="hold",
            policy_bias=0.05,
            control_gain=0.05,
            adaptation_pressure=0.05,
        )
    )
    monitor = evaluate_recovery_validation_kernel(
        _inputs(
            anomaly_label="watch",
            anomaly_score=0.40,
            anomaly_confidence=0.80,
            recovery_ready=True,
            adaptation_label="tune",
            policy_bias=0.40,
            control_gain=0.40,
            adaptation_pressure=0.40,
        )
    )
    validate_recovery = evaluate_recovery_validation_kernel(
        _inputs(
            anomaly_label="recover",
            anomaly_score=0.65,
            anomaly_confidence=0.70,
            recovery_ready=True,
            adaptation_label="constrain",
            policy_bias=0.67,
            control_gain=0.70,
            adaptation_pressure=0.68,
        )
    )
    failed = evaluate_recovery_validation_kernel(
        _inputs(
            anomaly_label="critical",
            anomaly_score=0.90,
            anomaly_confidence=0.80,
            recovery_ready=True,
            adaptation_label="hold",
            policy_bias=0.10,
            control_gain=0.40,
            adaptation_pressure=0.30,
        )
    )

    assert stable.decision.recovery_viable is True
    assert monitor.decision.recovery_viable is True
    assert validate_recovery.decision.recovery_viable is True
    assert failed.decision.recovery_viable is False


def test_alignment_behavior_closer_bias_yields_higher_alignment() -> None:
    close = evaluate_recovery_validation_kernel(
        _inputs(
            anomaly_label="watch",
            anomaly_score=0.55,
            anomaly_confidence=0.80,
            recovery_ready=True,
            adaptation_label="tune",
            policy_bias=0.56,
            control_gain=0.50,
            adaptation_pressure=0.30,
        )
    )
    far = evaluate_recovery_validation_kernel(
        _inputs(
            anomaly_label="watch",
            anomaly_score=0.55,
            anomaly_confidence=0.80,
            recovery_ready=True,
            adaptation_label="tune",
            policy_bias=0.10,
            control_gain=0.50,
            adaptation_pressure=0.30,
        )
    )

    assert close.signal.recovery_alignment > far.signal.recovery_alignment


def test_confidence_behavior_increases_with_confidence_and_alignment() -> None:
    high = evaluate_recovery_validation_kernel(
        _inputs(
            anomaly_label="watch",
            anomaly_score=0.45,
            anomaly_confidence=0.95,
            recovery_ready=True,
            adaptation_label="tune",
            policy_bias=0.45,
            control_gain=0.50,
            adaptation_pressure=0.30,
        )
    )
    low = evaluate_recovery_validation_kernel(
        _inputs(
            anomaly_label="watch",
            anomaly_score=0.45,
            anomaly_confidence=0.40,
            recovery_ready=True,
            adaptation_label="tune",
            policy_bias=0.10,
            control_gain=0.50,
            adaptation_pressure=0.30,
        )
    )

    assert high.decision.validation_confidence > low.decision.validation_confidence


def test_validation_incorrect_input_types_raise_value_error() -> None:
    with pytest.raises(ValueError, match="anomaly_receipt must be AutonomousAnomalyReceipt"):
        RecoveryValidationInputs(  # type: ignore[arg-type]
            anomaly_receipt="bad",
            policy_receipt=_policy_receipt(
                adaptation_label="hold",
                policy_bias=0.2,
                control_gain=0.2,
                adaptation_pressure=0.2,
            ),
        )

    with pytest.raises(ValueError, match="inputs must be RecoveryValidationInputs"):
        evaluate_recovery_validation_kernel("bad-input")  # type: ignore[arg-type]


def test_hash_stability_repeated_runs_identical() -> None:
    inputs = _inputs(
        anomaly_label="recover",
        anomaly_score=0.61,
        anomaly_confidence=0.77,
        recovery_ready=True,
        adaptation_label="constrain",
        policy_bias=0.64,
        control_gain=0.62,
        adaptation_pressure=0.60,
    )

    hashes = tuple(evaluate_recovery_validation_kernel(inputs).stable_hash for _ in range(5))
    assert hashes == (hashes[0],) * 5


def test_canonical_serialization_replay_safe() -> None:
    receipt = evaluate_recovery_validation_kernel(
        _inputs(
            anomaly_label="watch",
            anomaly_score=0.41,
            anomaly_confidence=0.78,
            recovery_ready=True,
            adaptation_label="tune",
            policy_bias=0.39,
            control_gain=0.45,
            adaptation_pressure=0.40,
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
    assert payload["version"] == RECOVERY_VALIDATION_KERNEL_VERSION
    assert receipt.to_canonical_json() == json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
