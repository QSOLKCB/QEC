# SPDX-License-Identifier: MIT
"""Deterministic adversarial perturbation battery for analysis-layer receipts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any

from qec.analysis.counterfactual_replay_kernel import CounterfactualReplayReceipt, run_counterfactual_replay
from qec.analysis.fix_proposal_kernel import FixProposalReceipt, generate_fix_proposals
from qec.analysis.fix_validation_kernel import FixValidationReceipt, validate_fix_proposals
from qec.analysis.issue_normalization_kernel import IssueNormalizationReceipt, normalize_review_issues

_SCHEMA_VERSION = "1.0"
_MODULE_VERSION = "v148.5"

_ALLOWED_PERTURBATION_TYPES = frozenset(
    {
        "FIELD_ORDER_SHUFFLE",
        "DUPLICATE_ENTRY",
        "MISSING_FIELD",
        "INVALID_ENUM",
        "NULL_BYTE_INJECTION",
        "PATH_VARIATION",
        "FLOAT_PRECISION_DRIFT",
        "EMPTY_PAYLOAD",
        "STRUCTURAL_CORRUPTION",
    }
)

_EXPECTED_OUTCOME_BY_PERTURBATION = {
    "FIELD_ORDER_SHUFFLE": "VALID",
    "DUPLICATE_ENTRY": "REJECTED",
    "MISSING_FIELD": "REJECTED",
    "INVALID_ENUM": "REJECTED",
    "NULL_BYTE_INJECTION": "REJECTED",
    "PATH_VARIATION": "VALID",
    "FLOAT_PRECISION_DRIFT": "VALID",
    "EMPTY_PAYLOAD": "REJECTED",
    "STRUCTURAL_CORRUPTION": "REJECTED",
}

_ALLOWED_OBSERVED_STATUS = frozenset({"REJECTED", "VALID", "INVALID", "UNSAFE", "INSUFFICIENT"})


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _canonicalize(value: Any) -> Any:
    if isinstance(value, bool | str) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return float(format(value, ".12g"))
    if isinstance(value, Mapping):
        return {k: _canonicalize(value[k]) for k in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonicalize(item) for item in value]
    return str(value)


def _stable_hash_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _status_from_validation(validation: FixValidationReceipt) -> str:
    if validation.validation_status == "ALL_VALID":
        return "VALID"
    if validation.validation_status == "HAS_UNSAFE":
        return "UNSAFE"
    if validation.validation_status == "HAS_INSUFFICIENT":
        return "INSUFFICIENT"
    return "INVALID"


def _validate_artifact_shape(artifact: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    issues = artifact.get("issues")
    if not isinstance(issues, Sequence) or isinstance(issues, (str, bytes, bytearray)):
        raise ValueError("artifact must contain an 'issues' sequence")
    normalized = tuple(issues)
    for issue in normalized:
        if not isinstance(issue, Mapping):
            raise ValueError("artifact issues must be mappings")
    return normalized


def _run_pipeline(artifact: Mapping[str, Any]) -> tuple[str, IssueNormalizationReceipt, FixProposalReceipt, FixValidationReceipt, CounterfactualReplayReceipt]:
    issues = _validate_artifact_shape(artifact)
    normalization = normalize_review_issues(issues)
    proposals = generate_fix_proposals(normalization)
    validation = validate_fix_proposals(proposals)
    replay = run_counterfactual_replay(validation)
    return _status_from_validation(validation), normalization, proposals, validation, replay


@dataclass(frozen=True)
class AdversarialCase:
    case_id: str
    artifact_type: str
    perturbation_type: str
    original_hash: str
    perturbed_hash: str
    expected_outcome: str

    def __post_init__(self) -> None:
        if self.perturbation_type not in _ALLOWED_PERTURBATION_TYPES:
            raise ValueError(f"invalid perturbation_type: {self.perturbation_type}")
        if self.expected_outcome not in {"VALID", "REJECTED"}:
            raise ValueError(f"invalid expected_outcome: {self.expected_outcome}")

    def _payload_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "artifact_type": self.artifact_type,
            "perturbation_type": self.perturbation_type,
            "original_hash": self.original_hash,
            "perturbed_hash": self.perturbed_hash,
            "expected_outcome": self.expected_outcome,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_dict()
        payload["stable_hash"] = self.stable_hash()
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _stable_hash_payload(self._payload_dict())


@dataclass(frozen=True)
class AdversarialResult:
    case_id: str
    observed_status: str
    expected_outcome: str
    determinism_preserved: bool
    hash_stable: bool
    validity_preserved: bool

    def __post_init__(self) -> None:
        if self.observed_status not in _ALLOWED_OBSERVED_STATUS:
            raise ValueError(f"invalid observed_status: {self.observed_status}")

    def _payload_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "observed_status": self.observed_status,
            "expected_outcome": self.expected_outcome,
            "determinism_preserved": self.determinism_preserved,
            "hash_stable": self.hash_stable,
            "validity_preserved": self.validity_preserved,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_dict()
        payload["stable_hash"] = self.stable_hash()
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _stable_hash_payload(self._payload_dict())


@dataclass(frozen=True)
class AdversarialDeterminismReceipt:
    schema_version: str
    module_version: str
    battery_status: str
    input_artifact_hash: str
    case_count: int
    pass_count: int
    fail_count: int
    determinism_pass: bool
    hash_stability_pass: bool
    false_positive_detected: bool

    def _payload_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "module_version": self.module_version,
            "battery_status": self.battery_status,
            "input_artifact_hash": self.input_artifact_hash,
            "case_count": self.case_count,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "determinism_pass": self.determinism_pass,
            "hash_stability_pass": self.hash_stability_pass,
            "false_positive_detected": self.false_positive_detected,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_dict()
        payload["stable_hash"] = self.stable_hash()
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _stable_hash_payload(self._payload_dict())


def _build_case_id(artifact_type: str, perturbation_type: str, original_hash: str, perturbed_hash: str) -> str:
    payload = {
        "artifact_type": artifact_type,
        "perturbation_type": perturbation_type,
        "original_hash": original_hash,
        "perturbed_hash": perturbed_hash,
    }
    return f"ADV-{_stable_hash_payload(payload)[:16]}"


def _issue_with_updated_path(issue: Mapping[str, Any], path: str) -> dict[str, Any]:
    clone = dict(issue)
    clone["target_path"] = path
    return clone


def _generate_perturbations(artifact: Mapping[str, Any]) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    issues = tuple(_validate_artifact_shape(artifact))
    if len(issues) == 0:
        return ()
    first = dict(issues[0])

    field_order_issue = {k: first[k] for k in tuple(sorted(first.keys(), reverse=True))}

    missing_field_issue = dict(first)
    missing_field_issue.pop("summary", None)
    missing_field_issue.pop("body", None)

    invalid_enum_issue = dict(first)
    invalid_enum_issue["category"] = "NOT_A_REAL_CATEGORY"

    null_byte_issue = _issue_with_updated_path(first, f"{first.get('target_path', 'src/qec/analysis/kernel.py')}\x00")

    path_variation_issue = _issue_with_updated_path(first, f"./{str(first.get('target_path', '')).replace('/', '\\\\')}")

    float_drift_issue = dict(first)
    float_drift_issue["confidence"] = 0.30000000000000004

    return (
        ("DUPLICATE_ENTRY", {"issues": [dict(first), dict(first)]}),
        ("EMPTY_PAYLOAD", {}),
        ("FIELD_ORDER_SHUFFLE", {"issues": [field_order_issue]}),
        ("FLOAT_PRECISION_DRIFT", {"issues": [float_drift_issue]}),
        ("INVALID_ENUM", {"issues": [invalid_enum_issue]}),
        ("MISSING_FIELD", {"issues": [missing_field_issue]}),
        ("NULL_BYTE_INJECTION", {"issues": [null_byte_issue]}),
        ("PATH_VARIATION", {"issues": [path_variation_issue]}),
        ("STRUCTURAL_CORRUPTION", {"issues": "not-a-sequence"}),
    )


def _evaluate_case(case: AdversarialCase, perturbed_artifact: Mapping[str, Any]) -> AdversarialResult:
    expected = case.expected_outcome
    try:
        run1 = _run_pipeline(perturbed_artifact)
        run2 = _run_pipeline(perturbed_artifact)
        observed = run1[0]

        determinism_preserved = (
            run1[1].stable_hash() == run2[1].stable_hash()
            and run1[2].stable_hash() == run2[2].stable_hash()
            and run1[3].stable_hash() == run2[3].stable_hash()
            and run1[4].stable_hash() == run2[4].stable_hash()
            and run1[1].to_canonical_bytes() == run2[1].to_canonical_bytes()
            and run1[2].to_canonical_bytes() == run2[2].to_canonical_bytes()
            and run1[3].to_canonical_bytes() == run2[3].to_canonical_bytes()
            and run1[4].to_canonical_bytes() == run2[4].to_canonical_bytes()
        )
        hash_stable = run1[4].stable_hash() == run2[4].stable_hash()
    except Exception:
        observed = "REJECTED"
        determinism_preserved = True
        hash_stable = True

    validity_preserved = observed == expected
    return AdversarialResult(
        case_id=case.case_id,
        observed_status=observed,
        expected_outcome=expected,
        determinism_preserved=determinism_preserved,
        hash_stable=hash_stable,
        validity_preserved=validity_preserved,
    )


def run_adversarial_determinism_battery(artifact: Mapping[str, Any]) -> AdversarialDeterminismReceipt:
    if not isinstance(artifact, Mapping):
        raise ValueError("artifact must be a mapping")

    normalized_input = _canonicalize(artifact)
    input_artifact_hash = _stable_hash_payload({"artifact": normalized_input})

    perturbations = _generate_perturbations(normalized_input)
    if len(perturbations) == 0:
        return AdversarialDeterminismReceipt(
            schema_version=_SCHEMA_VERSION,
            module_version=_MODULE_VERSION,
            battery_status="EMPTY",
            input_artifact_hash=input_artifact_hash,
            case_count=0,
            pass_count=0,
            fail_count=0,
            determinism_pass=True,
            hash_stability_pass=True,
            false_positive_detected=False,
        )

    case_results: list[AdversarialResult] = []
    for perturbation_type, perturbed in perturbations:
        perturbed_hash = _stable_hash_payload({"artifact": _canonicalize(perturbed)})
        case = AdversarialCase(
            case_id=_build_case_id("ISSUE_ARTIFACT", perturbation_type, input_artifact_hash, perturbed_hash),
            artifact_type="ISSUE_ARTIFACT",
            perturbation_type=perturbation_type,
            original_hash=input_artifact_hash,
            perturbed_hash=perturbed_hash,
            expected_outcome=_EXPECTED_OUTCOME_BY_PERTURBATION[perturbation_type],
        )
        case_results.append(_evaluate_case(case, perturbed))

    case_count = len(case_results)
    pass_count = sum(1 for result in case_results if result.validity_preserved)
    fail_count = case_count - pass_count
    determinism_pass = all(result.determinism_preserved for result in case_results)
    hash_stability_pass = all(result.hash_stable for result in case_results)
    false_positive_detected = any(
        result.expected_outcome == "REJECTED" and result.observed_status in {"VALID"}
        for result in case_results
    )

    if false_positive_detected:
        battery_status = "FALSE_POSITIVE_DETECTED"
    elif not hash_stability_pass:
        battery_status = "HASH_UNSTABLE"
    elif not determinism_pass:
        battery_status = "DETERMINISM_BROKEN"
    elif fail_count > 0:
        battery_status = "HAS_FAILURE"
    else:
        battery_status = "ALL_PASS"

    return AdversarialDeterminismReceipt(
        schema_version=_SCHEMA_VERSION,
        module_version=_MODULE_VERSION,
        battery_status=battery_status,
        input_artifact_hash=input_artifact_hash,
        case_count=case_count,
        pass_count=pass_count,
        fail_count=fail_count,
        determinism_pass=determinism_pass,
        hash_stability_pass=hash_stability_pass,
        false_positive_detected=false_positive_detected,
    )


__all__ = [
    "AdversarialCase",
    "AdversarialResult",
    "AdversarialDeterminismReceipt",
    "run_adversarial_determinism_battery",
]
