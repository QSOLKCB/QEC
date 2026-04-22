from __future__ import annotations

import json

import pytest

from qec.analysis.autonomous_anomaly_detection_kernel import (
    AnomalyDecision,
    AnomalySignal,
    AutonomousAnomalyReceipt,
)
from qec.analysis.deterministic_rollback_planning_engine import (
    DETERMINISTIC_ROLLBACK_PLANNING_ENGINE_VERSION,
    RollbackPlanningInputs,
    evaluate_deterministic_rollback_planning_engine,
)


def _anomaly_receipt(
    *,
    anomaly_label: str,
    anomaly_score: float,
    anomaly_confidence: float,
    dominant_signal: str,
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
            dominant_signal=dominant_signal,
        ),
        decision=AnomalyDecision(
            anomaly_label=anomaly_label,
            recovery_ready=anomaly_label in {"recover", "critical"},
            escalation_rank=escalation_by_label[anomaly_label],
            rationale=f"r::{dominant_signal}",
        ),
        control_mode="autonomous_anomaly_advisory",
        observatory_only=True,
    )


def _inputs(**kwargs: object) -> RollbackPlanningInputs:
    return RollbackPlanningInputs(anomaly_receipt=_anomaly_receipt(**kwargs))


def test_deterministic_replay_identical_json_and_hash() -> None:
    inputs = _inputs(
        anomaly_label="recover",
        anomaly_score=0.62,
        anomaly_confidence=0.77,
        dominant_signal="thermal",
    )

    first = evaluate_deterministic_rollback_planning_engine(inputs)
    second = evaluate_deterministic_rollback_planning_engine(inputs)

    assert first.to_canonical_json() == second.to_canonical_json()
    assert first.stable_hash == second.stable_hash


def test_candidate_completeness_and_fixed_order() -> None:
    receipt = evaluate_deterministic_rollback_planning_engine(
        _inputs(
            anomaly_label="watch",
            anomaly_score=0.41,
            anomaly_confidence=0.68,
            dominant_signal="latency",
        )
    )

    assert tuple(candidate.candidate_type for candidate in receipt.plan.candidates) == (
        "none",
        "soft_reset",
        "partial_rollback",
        "full_rollback",
    )


def test_selection_mapping_from_anomaly_label() -> None:
    assert evaluate_deterministic_rollback_planning_engine(
        _inputs(
            anomaly_label="nominal",
            anomaly_score=0.10,
            anomaly_confidence=0.20,
            dominant_signal="timing",
        )
    ).plan.selected_action == "none"

    assert evaluate_deterministic_rollback_planning_engine(
        _inputs(
            anomaly_label="watch",
            anomaly_score=0.30,
            anomaly_confidence=0.40,
            dominant_signal="timing",
        )
    ).plan.selected_action == "soft_reset"

    assert evaluate_deterministic_rollback_planning_engine(
        _inputs(
            anomaly_label="recover",
            anomaly_score=0.60,
            anomaly_confidence=0.40,
            dominant_signal="timing",
        )
    ).plan.selected_action == "partial_rollback"

    assert evaluate_deterministic_rollback_planning_engine(
        _inputs(
            anomaly_label="critical",
            anomaly_score=0.90,
            anomaly_confidence=0.40,
            dominant_signal="timing",
        )
    ).plan.selected_action == "full_rollback"


def test_strength_scaling_increases_with_anomaly_score() -> None:
    low = evaluate_deterministic_rollback_planning_engine(
        _inputs(
            anomaly_label="watch",
            anomaly_score=0.20,
            anomaly_confidence=0.80,
            dominant_signal="power",
        )
    )
    high = evaluate_deterministic_rollback_planning_engine(
        _inputs(
            anomaly_label="watch",
            anomaly_score=0.70,
            anomaly_confidence=0.80,
            dominant_signal="power",
        )
    )

    assert high.plan.rollback_strength > low.plan.rollback_strength


def test_confidence_propagation_matches_anomaly_confidence() -> None:
    receipt = evaluate_deterministic_rollback_planning_engine(
        _inputs(
            anomaly_label="recover",
            anomaly_score=0.55,
            anomaly_confidence=0.73,
            dominant_signal="thermal",
        )
    )

    assert receipt.plan.confidence == 0.73


def test_justification_determinism_identical_inputs_identical_strings() -> None:
    inputs = _inputs(
        anomaly_label="critical",
        anomaly_score=0.84,
        anomaly_confidence=0.49,
        dominant_signal="timing",
    )

    first = evaluate_deterministic_rollback_planning_engine(inputs)
    second = evaluate_deterministic_rollback_planning_engine(inputs)

    assert tuple(c.justification for c in first.plan.candidates) == tuple(
        c.justification for c in second.plan.candidates
    )


def test_validation_incorrect_input_type_raises_value_error() -> None:
    with pytest.raises(ValueError, match="inputs must be RollbackPlanningInputs"):
        evaluate_deterministic_rollback_planning_engine("bad-input")  # type: ignore[arg-type]


def test_hash_stability_repeated_runs_identical() -> None:
    inputs = _inputs(
        anomaly_label="recover",
        anomaly_score=0.66,
        anomaly_confidence=0.58,
        dominant_signal="power",
    )

    hashes = tuple(evaluate_deterministic_rollback_planning_engine(inputs).stable_hash for _ in range(5))
    assert hashes == (hashes[0],) * 5


def test_canonical_serialization_replay_safe() -> None:
    receipt = evaluate_deterministic_rollback_planning_engine(
        _inputs(
            anomaly_label="watch",
            anomaly_score=0.43,
            anomaly_confidence=0.61,
            dominant_signal="latency",
        )
    )

    payload = receipt.to_dict()
    assert tuple(payload.keys()) == (
        "version",
        "plan",
        "control_mode",
        "observatory_only",
        "stable_hash",
    )
    assert payload["version"] == DETERMINISTIC_ROLLBACK_PLANNING_ENGINE_VERSION
    assert receipt.to_canonical_json() == json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
