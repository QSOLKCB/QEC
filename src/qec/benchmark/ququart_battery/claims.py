"""Fail-closed validation for claims made about ququart battery evidence."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from qec.sonify.canonical import canonical_sha256

CLAIMS_SCHEMA = "qec.ququart-report-claims.v1"
VALIDATION_SCHEMA = "qec.ququart-report-claim-validation.v1"
ANALYSIS_SCOPE = "finite_code_capacity"
PERMITTED_CURVE_LANGUAGE = (
    "low_error_quadratic_regime",
    "intermediate_error_regime",
    "high_error_regime",
    "finite_code_crossover",
)
HARDWARE_DECLARATION_FIELDS = (
    "device_specification",
    "circuit_model",
    "leakage_model",
    "timing_model",
    "readout_model",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ReportClaimError(ValueError):
    """Raised when prose-level claims disagree with machine evidence."""


def _integer(row: Mapping[str, Any], key: str) -> int:
    value = row.get(key, 0)
    if isinstance(value, bool):
        raise TypeError(f"{key} must be an integer")
    return int(value)


def derive_evidence_facts(
    fault_rows: Sequence[Mapping[str, Any]],
    harmonic_rows: Sequence[Mapping[str, Any]],
    *,
    lane_symmetry: Mapping[str, Any] | None = None,
) -> dict[str, int | bool | str]:
    """Derive reportable facts directly from generated evidence tables."""

    expected_accept = tuple(
        row for row in fault_rows
        if row.get("expected") == "accept_and_correct_all"
    )
    expected_reject = tuple(
        row for row in fault_rows
        if row.get("expected") == "reject_all"
    )
    deterministic_evaluations = sum(
        _integer(row, "errors_tested") for row in fault_rows
    )
    expected_accept_evaluations = sum(
        _integer(row, "errors_tested") for row in expected_accept
    )
    adversarial_rejection_evaluations = sum(
        _integer(row, "errors_tested") for row in expected_reject
    )
    acceptance_violations = sum(
        _integer(row, "errors_tested") - _integer(row, "successful")
        for row in expected_accept
    )
    rejection_violations = sum(
        _integer(row, "accepted") for row in expected_reject
    )
    adversarial_false_accepts = sum(
        _integer(row, "false_accepts") for row in expected_reject
    )

    end_to_end_trials = sum(_integer(row, "trials") for row in harmonic_rows)
    receiver_rejections = sum(
        _integer(row, "receiver_rejections") for row in harmonic_rows
    )
    decoder_rejections = sum(
        _integer(row, "decoder_rejections") for row in harmonic_rows
    )
    receiver_false_trust = sum(
        _integer(
            row,
            "receiver_false_trust"
            if "receiver_false_trust" in row
            else "incorrect_trusted_syndrome",
        )
        for row in harmonic_rows
    )
    accepted_logical_residuals = sum(
        _integer(
            row,
            "accepted_logical_residual"
            if "accepted_logical_residual" in row
            else "false_accepts",
        )
        for row in harmonic_rows
    )

    return {
        "analysis_scope": ANALYSIS_SCOPE,
        "deterministic_fault_cases": len(fault_rows),
        "deterministic_fault_evaluations": deterministic_evaluations,
        "expected_accept_evaluations": expected_accept_evaluations,
        "adversarial_rejection_evaluations": adversarial_rejection_evaluations,
        "deterministic_acceptance_violations": acceptance_violations,
        "deterministic_rejection_violations": rejection_violations,
        "deterministic_adversarial_false_accepts": adversarial_false_accepts,
        "end_to_end_cells": len(harmonic_rows),
        "end_to_end_trials": end_to_end_trials,
        "receiver_rejections": receiver_rejections,
        "decoder_rejections": decoder_rejections,
        "receiver_false_trust": receiver_false_trust,
        "accepted_logical_residuals": accepted_logical_residuals,
        "lane_symmetry_verified": bool(
            lane_symmetry
            and lane_symmetry.get("weight_enumerators_equal") is True
        ),
    }


def derived_report_claims(
    facts: Mapping[str, Any],
) -> dict[str, object]:
    """Create a conservative claim declaration entirely from evidence facts."""

    claims: dict[str, object] = {
        "schema": CLAIMS_SCHEMA,
        "analysis_scope": ANALYSIS_SCOPE,
        "curve_language": list(PERMITTED_CURVE_LANGUAGE[:3]),
        "threshold_claim": False,
        "end_to_end_cells": _integer(facts, "end_to_end_cells"),
        "deterministic_fault_evaluations": _integer(
            facts, "deterministic_fault_evaluations"
        ),
        "expected_accept_evaluations": _integer(
            facts, "expected_accept_evaluations"
        ),
        "adversarial_rejection_evaluations": _integer(
            facts, "adversarial_rejection_evaluations"
        ),
        "zero_deterministic_adversarial_false_accepts": (
            _integer(facts, "deterministic_adversarial_false_accepts") == 0
        ),
        "zero_receiver_false_trust": (
            _integer(facts, "receiver_false_trust") == 0
        ),
        "lane_symmetry_verified": bool(facts.get("lane_symmetry_verified")),
        "all_tests_passed": False,
        "test_status": "not_claimed_by_generated_report",
        "hardware_claim": False,
        "hardware_declarations": {},
        "artifact_match_claims": [],
    }
    claims["sha256"] = canonical_sha256(claims)
    return claims


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ReportClaimError(f"{field} must be a full lowercase SHA-256 digest")
    return value


def validate_report_claims(
    claims: Mapping[str, Any],
    facts: Mapping[str, Any],
    *,
    test_receipt: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Validate report claims against evidence and return a hash-bound receipt."""

    errors: list[str] = []
    if claims.get("schema") != CLAIMS_SCHEMA:
        errors.append(f"claims schema must be {CLAIMS_SCHEMA}")
    if claims.get("analysis_scope") != facts.get("analysis_scope"):
        errors.append("analysis scope does not match generated evidence")

    exact_fields = (
        "end_to_end_cells",
        "deterministic_fault_evaluations",
        "expected_accept_evaluations",
        "adversarial_rejection_evaluations",
    )
    for field in exact_fields:
        try:
            claimed = _integer(claims, field)
            actual = _integer(facts, field)
        except (TypeError, ValueError) as error:
            errors.append(str(error))
            continue
        if claimed != actual:
            errors.append(f"{field} claim {claimed} does not match evidence {actual}")

    permitted = set(PERMITTED_CURVE_LANGUAGE)
    language = claims.get("curve_language", [])
    if isinstance(language, str) or not isinstance(language, Sequence):
        errors.append("curve_language must be a sequence of controlled terms")
    else:
        unknown = sorted(str(term) for term in language if term not in permitted)
        if unknown:
            errors.append("uncontrolled curve language: " + ", ".join(unknown))

    if claims.get("threshold_claim") is not False:
        errors.append(
            "threshold claims require a separately declared code-family scaling study"
        )

    expected_zero_adversarial = (
        _integer(facts, "deterministic_adversarial_false_accepts") == 0
    )
    if claims.get("zero_deterministic_adversarial_false_accepts") is not expected_zero_adversarial:
        errors.append(
            "zero deterministic adversarial false-accept claim disagrees with evidence"
        )

    expected_zero_false_trust = _integer(facts, "receiver_false_trust") == 0
    if claims.get("zero_receiver_false_trust") is not expected_zero_false_trust:
        errors.append("zero receiver-false-trust claim disagrees with evidence")

    if claims.get("lane_symmetry_verified") is not bool(
        facts.get("lane_symmetry_verified")
    ):
        errors.append("lane-symmetry claim disagrees with exact-channel evidence")

    if claims.get("all_tests_passed") is True:
        if not test_receipt:
            errors.append("all-tests-passed claim requires a test receipt")
        else:
            if test_receipt.get("status") != "passed":
                errors.append("test receipt does not report passed status")
            try:
                _require_sha256(test_receipt.get("sha256"), "test_receipt.sha256")
            except ReportClaimError as error:
                errors.append(str(error))

    hardware_claim = claims.get("hardware_claim") is True
    declarations = claims.get("hardware_declarations", {})
    if hardware_claim:
        if not isinstance(declarations, Mapping):
            errors.append("hardware declarations must be an object")
        else:
            missing = [
                field for field in HARDWARE_DECLARATION_FIELDS
                if not isinstance(declarations.get(field), str)
                or not declarations.get(field).strip()
            ]
            if missing:
                errors.append(
                    "hardware claim missing declarations: " + ", ".join(missing)
                )

    artifact_claims = claims.get("artifact_match_claims", [])
    if isinstance(artifact_claims, str) or not isinstance(artifact_claims, Sequence):
        errors.append("artifact_match_claims must be a sequence")
    else:
        for index, claim in enumerate(artifact_claims):
            if not isinstance(claim, Mapping):
                errors.append(f"artifact_match_claims[{index}] must be an object")
                continue
            try:
                expected = _require_sha256(
                    claim.get("expected_sha256"),
                    f"artifact_match_claims[{index}].expected_sha256",
                )
                observed = _require_sha256(
                    claim.get("observed_sha256"),
                    f"artifact_match_claims[{index}].observed_sha256",
                )
            except ReportClaimError as error:
                errors.append(str(error))
                continue
            if expected != observed:
                errors.append(f"artifact match claim {index} has unequal hashes")

    if errors:
        raise ReportClaimError("; ".join(errors))

    normalized_claims = dict(claims)
    normalized_facts = dict(facts)
    receipt: dict[str, object] = {
        "schema": VALIDATION_SCHEMA,
        "passed": True,
        "claims_sha256": canonical_sha256(normalized_claims),
        "facts_sha256": canonical_sha256(normalized_facts),
        "checks": {
            "numeric_claims_match": True,
            "controlled_curve_language": True,
            "threshold_claim_permitted": False,
            "false_accept_claims_match": True,
            "lane_symmetry_claim_matches": True,
            "hardware_contract_enforced": True,
            "artifact_matches_require_full_sha256": True,
            "test_claim_requires_receipt": True,
        },
    }
    receipt["sha256"] = canonical_sha256(receipt)
    return receipt
