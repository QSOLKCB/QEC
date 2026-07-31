import pytest

from qec.benchmark.ququart_battery.claims import (
    ReportClaimError,
    derive_evidence_facts,
    derived_report_claims,
    validate_report_claims,
)


FAULT_ROWS = (
    {
        "fault_case": "clean",
        "errors_tested": 75,
        "accepted": 75,
        "successful": 75,
        "false_accepts": 0,
        "expected": "accept_and_correct_all",
    },
    {
        "fault_case": "reject",
        "errors_tested": 75,
        "accepted": 0,
        "successful": 0,
        "false_accepts": 0,
        "expected": "reject_all",
    },
)
HARMONIC_ROWS = (
    {
        "trials": 100,
        "receiver_rejections": 8,
        "decoder_rejections": 2,
        "receiver_false_trust": 0,
        "accepted_logical_residual": 1,
    },
    {
        "trials": 100,
        "receiver_rejections": 90,
        "decoder_rejections": 0,
        "receiver_false_trust": 1,
        "accepted_logical_residual": 0,
    },
)


def _facts():
    return derive_evidence_facts(
        FAULT_ROWS,
        HARMONIC_ROWS,
        lane_symmetry={"weight_enumerators_equal": True},
    )


def test_derived_claims_validate_against_their_evidence():
    facts = _facts()
    claims = derived_report_claims(facts)
    receipt = validate_report_claims(claims, facts)
    assert receipt["passed"] is True
    assert len(receipt["sha256"]) == 64
    assert claims["zero_receiver_false_trust"] is False


def test_claimed_cell_count_must_match_evidence():
    facts = _facts()
    claims = derived_report_claims(facts)
    claims["end_to_end_cells"] = 112
    with pytest.raises(ReportClaimError, match="end_to_end_cells"):
        validate_report_claims(claims, facts)


def test_threshold_and_test_pass_claims_fail_closed():
    facts = _facts()
    claims = derived_report_claims(facts)
    claims["threshold_claim"] = True
    with pytest.raises(ReportClaimError, match="threshold"):
        validate_report_claims(claims, facts)

    claims = derived_report_claims(facts)
    claims["all_tests_passed"] = True
    with pytest.raises(ReportClaimError, match="test receipt"):
        validate_report_claims(claims, facts)


def test_hardware_and_artifact_claims_require_complete_evidence():
    facts = _facts()
    claims = derived_report_claims(facts)
    claims["hardware_claim"] = True
    with pytest.raises(ReportClaimError, match="hardware claim missing"):
        validate_report_claims(claims, facts)

    claims = derived_report_claims(facts)
    claims["artifact_match_claims"] = [{
        "expected_sha256": "a" * 64,
        "observed_sha256": "a" * 8,
    }]
    with pytest.raises(ReportClaimError, match="full lowercase SHA-256"):
        validate_report_claims(claims, facts)
