from dataclasses import FrozenInstanceError

import pytest

from qec.analysis import adversarial_determinism_battery as battery
from qec.analysis.adversarial_determinism_battery import (
    AdversarialCase,
    AdversarialDeterminismReceipt,
    AdversarialResult,
    run_adversarial_determinism_battery,
)


def _valid_artifact() -> dict[str, object]:
    return {
        "issues": [
            {
                "source": "CODEX",
                "category": "DETERMINISM",
                "severity": "HIGH",
                "target_path": "src/qec/analysis/sample_kernel.py",
                "summary": "Preserve deterministic canonical ordering.",
                "invariant": "DETERMINISTIC_ORDERING",
            }
        ]
    }


def test_field_order_shuffle_identical_output() -> None:
    receipt_1 = run_adversarial_determinism_battery(_valid_artifact())
    receipt_2 = run_adversarial_determinism_battery(_valid_artifact())

    assert receipt_1.to_canonical_json() == receipt_2.to_canonical_json()
    assert receipt_1.stable_hash() == receipt_2.stable_hash()


def test_expected_rejections_and_valid_paths() -> None:
    receipt = run_adversarial_determinism_battery(_valid_artifact())

    assert receipt.battery_status == "ALL_PASS"
    assert receipt.pass_count == 9
    assert receipt.fail_count == 0
    assert receipt.case_count == 9


def test_replay_and_hash_stability_across_runs() -> None:
    receipt_a = run_adversarial_determinism_battery(_valid_artifact())
    receipt_b = run_adversarial_determinism_battery(_valid_artifact())

    assert receipt_a.determinism_pass is True
    assert receipt_a.hash_stability_pass is True
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_false_positive_detection_is_not_triggered() -> None:
    receipt = run_adversarial_determinism_battery(_valid_artifact())
    assert receipt.false_positive_detected is False


def test_empty_artifact_receipt() -> None:
    receipt = run_adversarial_determinism_battery({"issues": []})
    assert receipt.battery_status == "EMPTY"
    assert receipt.case_count == 0


def test_frozen_dataclass_immutability() -> None:
    case = AdversarialCase(
        case_id="ADV-1",
        artifact_type="ISSUE_ARTIFACT",
        perturbation_type="FIELD_ORDER_SHUFFLE",
        original_hash="a" * 64,
        perturbed_hash="b" * 64,
        expected_outcome="VALID",
    )
    result = AdversarialResult(
        case_id="ADV-1",
        observed_status="VALID",
        expected_outcome="VALID",
        determinism_preserved=True,
        hash_stable=True,
        validity_preserved=True,
    )
    receipt = AdversarialDeterminismReceipt(
        schema_version="1.0",
        module_version="v148.5",
        battery_status="ALL_PASS",
        input_artifact_hash="c" * 64,
        case_count=1,
        pass_count=1,
        fail_count=0,
        determinism_pass=True,
        hash_stability_pass=True,
        false_positive_detected=False,
    )

    with pytest.raises(FrozenInstanceError):
        case.case_id = "ADV-2"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.observed_status = "REJECTED"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        receipt.battery_status = "HAS_FAILURE"  # type: ignore[misc]


def test_exception_path_is_not_treated_as_deterministic_success(monkeypatch: pytest.MonkeyPatch) -> None:
    case = AdversarialCase(
        case_id="ADV-exc",
        artifact_type="ISSUE_ARTIFACT",
        perturbation_type="MISSING_FIELD",
        original_hash="a" * 64,
        perturbed_hash="b" * 64,
        expected_outcome="REJECTED",
    )
    calls = {"count": 0}

    def _flaky_run_pipeline(_: dict[str, object]) -> object:
        calls["count"] += 1
        if calls["count"] == 1:
            raise ValueError("first-run-failure")
        return ("VALID", None, None, None, None)

    monkeypatch.setattr(battery, "_run_pipeline", _flaky_run_pipeline)

    result = battery._evaluate_case(case, {"issues": []})

    assert result.observed_status == "REJECTED"
    assert result.validity_preserved is True
    assert result.determinism_preserved is False
    assert result.hash_stable is False
