from __future__ import annotations

import json

import pytest

from qec.analysis.autonomous_anomaly_detection_kernel import (
    AUTONOMOUS_ANOMALY_DETECTION_KERNEL_VERSION,
    AnomalyDetectionInputs,
    AutonomousAnomalyReceipt,
    evaluate_autonomous_anomaly_detection_kernel,
)
from qec.analysis.hardware_feedback_consensus_bridge import (
    ControlSignalVote,
    HardwareConsensusDecision,
    HardwareConsensusReceipt,
)


def _consensus_receipt(
    *,
    consensus_pressure: float,
    consensus_stability: float,
    consensus_confidence: float,
    conflict_count: int,
    dominant_signal: str,
    action_label: str,
) -> HardwareConsensusReceipt:
    votes = (
        ControlSignalVote(
            signal_name="thermal",
            pressure=consensus_pressure,
            stability=consensus_stability,
            severity_rank=0,
            action_label="stable",
        ),
        ControlSignalVote(
            signal_name="latency",
            pressure=consensus_pressure,
            stability=consensus_stability,
            severity_rank=0,
            action_label="stable",
        ),
        ControlSignalVote(
            signal_name="timing",
            pressure=consensus_pressure,
            stability=consensus_stability,
            severity_rank=0,
            action_label="stable",
        ),
        ControlSignalVote(
            signal_name="power",
            pressure=consensus_pressure,
            stability=consensus_stability,
            severity_rank=0,
            action_label="stable",
        ),
    )
    return HardwareConsensusReceipt(
        version="v140.4",
        signal_votes=votes,
        decision=HardwareConsensusDecision(
            consensus_pressure=consensus_pressure,
            consensus_stability=consensus_stability,
            consensus_confidence=consensus_confidence,
            conflict_count=conflict_count,
            dominant_signal=dominant_signal,
            action_label=action_label,
        ),
        control_mode="hardware_consensus_advisory",
        observatory_only=True,
    )


def _inputs(**kwargs: object) -> AnomalyDetectionInputs:
    return AnomalyDetectionInputs(consensus_receipt=_consensus_receipt(**kwargs))


def test_deterministic_replay_identical_json_and_hash() -> None:
    inputs = _inputs(
        consensus_pressure=0.42,
        consensus_stability=0.75,
        consensus_confidence=0.81,
        conflict_count=1,
        dominant_signal="thermal",
        action_label="bias_adjust",
    )

    first = evaluate_autonomous_anomaly_detection_kernel(inputs)
    second = evaluate_autonomous_anomaly_detection_kernel(inputs)

    assert first.to_canonical_json() == second.to_canonical_json()
    assert first.stable_hash == second.stable_hash


def test_nominal_path_classification() -> None:
    receipt = evaluate_autonomous_anomaly_detection_kernel(
        _inputs(
            consensus_pressure=0.10,
            consensus_stability=0.95,
            consensus_confidence=0.95,
            conflict_count=0,
            dominant_signal="latency",
            action_label="stable",
        )
    )

    assert receipt.decision.anomaly_label == "nominal"
    assert receipt.decision.escalation_rank == 0
    assert receipt.decision.recovery_ready is False


def test_watch_path_classification() -> None:
    receipt = evaluate_autonomous_anomaly_detection_kernel(
        _inputs(
            consensus_pressure=0.40,
            consensus_stability=0.70,
            consensus_confidence=0.80,
            conflict_count=1,
            dominant_signal="timing",
            action_label="bias_adjust",
        )
    )

    assert receipt.decision.anomaly_label == "watch"
    assert receipt.decision.escalation_rank == 1


def test_recover_path_classification_sets_recovery_ready() -> None:
    receipt = evaluate_autonomous_anomaly_detection_kernel(
        _inputs(
            consensus_pressure=0.80,
            consensus_stability=0.40,
            consensus_confidence=0.80,
            conflict_count=2,
            dominant_signal="power",
            action_label="reduce_load",
        )
    )

    assert receipt.decision.anomaly_label == "recover"
    assert receipt.decision.escalation_rank == 2
    assert receipt.decision.recovery_ready is True


def test_critical_override_emergency_align_forces_critical() -> None:
    receipt = evaluate_autonomous_anomaly_detection_kernel(
        _inputs(
            consensus_pressure=0.02,
            consensus_stability=0.98,
            consensus_confidence=0.90,
            conflict_count=0,
            dominant_signal="thermal",
            action_label="emergency_align",
        )
    )

    assert receipt.decision.anomaly_label == "critical"
    assert receipt.decision.escalation_rank == 3
    assert receipt.decision.recovery_ready is True


def test_conflict_contribution_increases_anomaly_score() -> None:
    low_conflict = evaluate_autonomous_anomaly_detection_kernel(
        _inputs(
            consensus_pressure=0.40,
            consensus_stability=0.70,
            consensus_confidence=0.85,
            conflict_count=0,
            dominant_signal="latency",
            action_label="bias_adjust",
        )
    )
    high_conflict = evaluate_autonomous_anomaly_detection_kernel(
        _inputs(
            consensus_pressure=0.40,
            consensus_stability=0.70,
            consensus_confidence=0.85,
            conflict_count=3,
            dominant_signal="latency",
            action_label="bias_adjust",
        )
    )

    assert high_conflict.signal.anomaly_score > low_conflict.signal.anomaly_score


def test_confidence_behavior_tracks_consensus_confidence() -> None:
    high_consensus_conf = evaluate_autonomous_anomaly_detection_kernel(
        _inputs(
            consensus_pressure=0.30,
            consensus_stability=0.80,
            consensus_confidence=0.95,
            conflict_count=1,
            dominant_signal="timing",
            action_label="bias_adjust",
        )
    )
    low_consensus_conf = evaluate_autonomous_anomaly_detection_kernel(
        _inputs(
            consensus_pressure=0.30,
            consensus_stability=0.80,
            consensus_confidence=0.30,
            conflict_count=1,
            dominant_signal="timing",
            action_label="bias_adjust",
        )
    )

    assert low_consensus_conf.signal.anomaly_confidence < high_consensus_conf.signal.anomaly_confidence


def test_validation_incorrect_input_type_raises_value_error() -> None:
    with pytest.raises(ValueError, match="inputs must be AnomalyDetectionInputs"):
        evaluate_autonomous_anomaly_detection_kernel("bad-input")  # type: ignore[arg-type]


def test_hash_stability_repeated_runs_identical() -> None:
    inputs = _inputs(
        consensus_pressure=0.66,
        consensus_stability=0.51,
        consensus_confidence=0.73,
        conflict_count=2,
        dominant_signal="power",
        action_label="reduce_load",
    )

    hashes = tuple(evaluate_autonomous_anomaly_detection_kernel(inputs).stable_hash for _ in range(5))
    assert hashes == (hashes[0],) * 5


def test_canonical_serialization_replay_safe() -> None:
    receipt = evaluate_autonomous_anomaly_detection_kernel(
        _inputs(
            consensus_pressure=0.52,
            consensus_stability=0.44,
            consensus_confidence=0.61,
            conflict_count=2,
            dominant_signal="thermal",
            action_label="reduce_load",
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
    assert payload["version"] == AUTONOMOUS_ANOMALY_DETECTION_KERNEL_VERSION
    assert receipt.to_canonical_json() == json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def test_receipt_stable_hash_derived_internally() -> None:
    receipt = evaluate_autonomous_anomaly_detection_kernel(
        _inputs(
            consensus_pressure=0.35,
            consensus_stability=0.65,
            consensus_confidence=0.75,
            conflict_count=1,
            dominant_signal="power",
            action_label="bias_adjust",
        )
    )

    assert isinstance(receipt, AutonomousAnomalyReceipt)
    assert receipt.stable_hash == receipt.stable_hash_value()
