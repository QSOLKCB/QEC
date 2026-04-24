from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json

import pytest

from qec.analysis.fix_proposal_kernel import FixProposal, FixProposalReceipt, FixProposalSet
from qec.analysis.fix_validation_kernel import (
    FixValidation,
    validate_fix_proposals,
)


def _proposal(
    *,
    proposal_id: str,
    issue_hash: str,
    category: str,
    severity: str,
    fix_strategy: str,
    invariant_preserved: str,
    target_path: str = "src/qec/analysis/example.py",
    fix_summary: str = "Deterministic fix summary",
    deterministic_safe: bool = True,
) -> FixProposal:
    return FixProposal(
        proposal_id=proposal_id,
        issue_hash=issue_hash,
        category=category,
        severity=severity,
        target_path=target_path,
        fix_strategy=fix_strategy,
        fix_summary=fix_summary,
        invariant_preserved=invariant_preserved,
        deterministic_safe=deterministic_safe,
    )


def _proposal_receipt(proposals: tuple[FixProposal, ...]) -> FixProposalReceipt:
    hashes = tuple(item.stable_hash() for item in proposals)
    proposal_set_hash = hashlib.sha256("|".join(hashes).encode("utf-8")).hexdigest()
    proposal_set = FixProposalSet(
        proposals=proposals,
        proposal_count=len(proposals),
        proposal_set_hash=proposal_set_hash,
    )
    status = "EMPTY" if len(proposals) == 0 else "PROPOSED"
    return FixProposalReceipt(
        schema_version="1.0",
        module_version="v148.2",
        proposal_status=status,
        input_issue_set_hash="issueset-hash",
        proposal_set=proposal_set,
        proposal_count=len(proposals),
    )


def test_valid_proposal_and_mapping_correctness() -> None:
    proposal = _proposal(
        proposal_id="PROPOSAL-0001",
        issue_hash="issue-0001",
        category="CANONICALIZATION",
        severity="MEDIUM",
        fix_strategy="CANONICALIZATION_FIX",
        invariant_preserved="CANONICAL_JSON",
    )

    receipt = validate_fix_proposals(_proposal_receipt((proposal,)))

    assert receipt.validation_status == "ALL_VALID"
    assert receipt.valid_count == 1
    validation = receipt.validation_set.validations[0]
    assert validation.validation_status == "VALID"
    assert validation.invariant_preserved is True
    assert validation.consistency_score == 1.0


def test_replay_stability_hash_and_bytes() -> None:
    proposal = _proposal(
        proposal_id="PROPOSAL-0002",
        issue_hash="issue-0002",
        category="ORDERING",
        severity="HIGH",
        fix_strategy="ORDERING_FIX",
        invariant_preserved="DETERMINISTIC_ORDERING",
    )

    first = validate_fix_proposals(_proposal_receipt((proposal,)))
    second = validate_fix_proposals(_proposal_receipt((proposal,)))

    assert first.to_canonical_json() == second.to_canonical_json()
    assert first.to_canonical_bytes() == second.to_canonical_bytes()
    assert first.stable_hash() == second.stable_hash()


def test_sorting_stability() -> None:
    valid = _proposal(
        proposal_id="PROPOSAL-B",
        issue_hash="issue-b",
        category="VALIDATION",
        severity="HIGH",
        fix_strategy="VALIDATION_HARDEN",
        invariant_preserved="FAIL_FAST_VALIDATION",
    )
    invalid = _proposal(
        proposal_id="PROPOSAL-A",
        issue_hash="issue-a",
        category="VALIDATION",
        severity="HIGH",
        fix_strategy="VALIDATION_HARDEN",
        invariant_preserved="WRONG_INVARIANT",
    )

    receipt = validate_fix_proposals(_proposal_receipt((valid, invalid)))

    statuses_then_ids = tuple((item.validation_status, item.proposal_id) for item in receipt.validation_set.validations)
    assert statuses_then_ids == (("INVALID", "PROPOSAL-A"), ("VALID", "PROPOSAL-B"))


def test_invalid_detection_from_consistency_mismatch() -> None:
    proposal = _proposal(
        proposal_id="PROPOSAL-0003",
        issue_hash="issue-0003",
        category="HASH_INTEGRITY",
        severity="LOW",
        fix_strategy="ORDERING_FIX",
        invariant_preserved="DETERMINISTIC_ORDERING",
    )

    receipt = validate_fix_proposals(_proposal_receipt((proposal,)))

    assert receipt.validation_status == "HAS_INVALID"
    assert receipt.invalid_count == 1
    assert receipt.validation_set.validations[0].consistency_score == 0.0


def test_unsafe_detection_scope_and_determinism() -> None:
    proposal = _proposal(
        proposal_id="PROPOSAL-0004",
        issue_hash="issue-0004",
        category="ORDERING",
        severity="MEDIUM",
        fix_strategy="ORDERING_FIX",
        invariant_preserved="DETERMINISTIC_ORDERING",
        target_path="src/qec/decoder/core.py",
        fix_summary="Use current timestamp and external API call",
    )

    receipt = validate_fix_proposals(_proposal_receipt((proposal,)))

    assert receipt.validation_status == "HAS_UNSAFE"
    assert receipt.unsafe_count == 1
    validation = receipt.validation_set.validations[0]
    assert validation.validation_status == "UNSAFE"
    assert validation.deterministic_safe is False
    assert validation.scope_compliant is False


def test_insufficient_no_action() -> None:
    proposal = _proposal(
        proposal_id="PROPOSAL-0005",
        issue_hash="issue-0005",
        category="UNKNOWN",
        severity="LOW",
        fix_strategy="NO_ACTION",
        invariant_preserved="NONE",
    )

    receipt = validate_fix_proposals(_proposal_receipt((proposal,)))

    assert receipt.validation_status == "HAS_INSUFFICIENT"
    assert receipt.insufficient_count == 1
    assert receipt.validation_set.validations[0].consistency_score == 0.5


def test_empty_input() -> None:
    receipt = validate_fix_proposals(_proposal_receipt(tuple()))

    assert receipt.validation_status == "EMPTY"
    assert receipt.validation_set.validation_count == 0
    assert receipt.valid_count == 0
    assert receipt.invalid_count == 0
    assert receipt.unsafe_count == 0
    assert receipt.insufficient_count == 0




def test_unsafe_when_proposal_marks_non_deterministic() -> None:
    proposal = _proposal(
        proposal_id="PROPOSAL-0007",
        issue_hash="issue-0007",
        category="ORDERING",
        severity="MEDIUM",
        fix_strategy="ORDERING_FIX",
        invariant_preserved="DETERMINISTIC_ORDERING",
        deterministic_safe=False,
        fix_summary="Deterministic wording only",
    )

    receipt = validate_fix_proposals(_proposal_receipt((proposal,)))

    assert receipt.validation_status == "HAS_UNSAFE"
    validation = receipt.validation_set.validations[0]
    assert validation.validation_status == "UNSAFE"
    assert validation.deterministic_safe is False


def test_test_addition_requires_allowed_invariant() -> None:
    valid = _proposal(
        proposal_id="PROPOSAL-0008",
        issue_hash="issue-0008",
        category="TEST_COVERAGE",
        severity="LOW",
        fix_strategy="TEST_ADDITION",
        invariant_preserved="TEST_COVERAGE",
    )
    invalid = _proposal(
        proposal_id="PROPOSAL-0009",
        issue_hash="issue-0009",
        category="TEST_COVERAGE",
        severity="LOW",
        fix_strategy="TEST_ADDITION",
        invariant_preserved="NOT_ALLOWED",
    )

    receipt = validate_fix_proposals(_proposal_receipt((valid, invalid)))

    statuses = {item.proposal_id: item.validation_status for item in receipt.validation_set.validations}
    assert statuses["PROPOSAL-0008"] == "VALID"
    assert statuses["PROPOSAL-0009"] == "INVALID"

def test_frozen_dataclass_immutability() -> None:
    validation = FixValidation(
        proposal_id="PROPOSAL-IMMUT",
        issue_hash="issue-immut",
        fix_strategy="ORDERING_FIX",
        validation_status="VALID",
        invariant_preserved=True,
        deterministic_safe=True,
        scope_compliant=True,
        consistency_score=1.0,
    )

    with pytest.raises(FrozenInstanceError):
        validation.validation_status = "INVALID"  # type: ignore[misc]


def test_canonical_json_ordering() -> None:
    proposal = _proposal(
        proposal_id="PROPOSAL-0006",
        issue_hash="issue-0006",
        category="IMMUTABILITY",
        severity="CRITICAL",
        fix_strategy="IMMUTABILITY_ENFORCEMENT",
        invariant_preserved="FROZEN_DATACLASS",
    )

    receipt = validate_fix_proposals(_proposal_receipt((proposal,)))
    expected = json.dumps(
        receipt.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    assert receipt.to_canonical_json() == expected
