from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qec.analysis.fix_proposal_kernel import FixProposal, FixProposalSet, generate_fix_proposals
from qec.analysis.issue_normalization_kernel import normalize_review_issues


def _make_receipt() -> object:
    raw_issues = [
        {
            "source": "MANUAL",
            "severity": "CRITICAL",
            "category": "HASH_INTEGRITY",
            "target_path": "src/qec/analysis/a.py",
            "summary": " Ensure stable_hash is validated using canonical SHA-256 check ",
            "invariant": "STABLE_HASH",
        },
        {
            "source": "MANUAL",
            "severity": "HIGH",
            "category": "SCOPE_GUARDRAIL",
            "target_path": "src/qec/analysis/b.py",
            "summary": "Restrict changes to analysis layer only",
            "invariant": "ANALYSIS_LAYER_ONLY",
        },
    ]
    return normalize_review_issues(raw_issues)


def test_proposal_generation_and_replay_stability() -> None:
    issue_receipt = _make_receipt()

    receipt_a = generate_fix_proposals(issue_receipt)
    receipt_b = generate_fix_proposals(issue_receipt)

    assert receipt_a.proposal_status == "PROPOSED"
    assert receipt_a.to_dict() == receipt_b.to_dict()
    assert receipt_a.to_canonical_json() == receipt_b.to_canonical_json()
    assert receipt_a.stable_hash() == receipt_b.stable_hash()
    assert receipt_a.proposal_set.proposal_set_hash == receipt_b.proposal_set.proposal_set_hash


def test_category_strategy_mapping() -> None:
    category_cases = [
        ("DETERMINISM", "ORDERING_FIX"),
        ("CANONICALIZATION", "CANONICALIZATION_FIX"),
        ("HASH_INTEGRITY", "HASH_VALIDATION_FIX"),
        ("IMMUTABILITY", "IMMUTABILITY_ENFORCEMENT"),
        ("VALIDATION", "VALIDATION_HARDEN"),
        ("BOUNDS", "BOUNDS_ENFORCEMENT"),
        ("ORDERING", "ORDERING_FIX"),
        ("TEST_COVERAGE", "TEST_ADDITION"),
        ("SCOPE_GUARDRAIL", "SCOPE_RESTRICTION"),
        ("UNKNOWN", "NO_ACTION"),
    ]

    raw_issues = [
        {
            "source": "UNKNOWN",
            "severity": "LOW",
            "category": category,
            "target_path": f"src/qec/analysis/{index}.py",
            "summary": f"summary for {category}",
            "invariant": "UNKNOWN",
        }
        for index, (category, _) in enumerate(category_cases)
    ]

    issue_receipt = normalize_review_issues(raw_issues)
    proposal_receipt = generate_fix_proposals(issue_receipt)

    expected_by_category = {category: strategy for category, strategy in category_cases}
    actual_by_category = {
        proposal.category: proposal.fix_strategy
        for proposal in proposal_receipt.proposal_set.proposals
    }
    assert actual_by_category == expected_by_category


def test_sorting_stability_for_shuffled_input() -> None:
    raw_issues = [
        {
            "source": "MANUAL",
            "severity": "LOW",
            "category": "VALIDATION",
            "target_path": "z.py",
            "summary": "validate inputs",
            "invariant": "FAIL_FAST_VALIDATION",
        },
        {
            "source": "MANUAL",
            "severity": "CRITICAL",
            "category": "ORDERING",
            "target_path": "a.py",
            "summary": "enforce deterministic ordering",
            "invariant": "DETERMINISTIC_ORDERING",
        },
    ]

    receipt_forward = generate_fix_proposals(normalize_review_issues(raw_issues))
    receipt_reverse = generate_fix_proposals(normalize_review_issues(list(reversed(raw_issues))))

    assert receipt_forward.to_canonical_json() == receipt_reverse.to_canonical_json()
    assert tuple(p.proposal_id for p in receipt_forward.proposal_set.proposals) == tuple(
        p.proposal_id for p in receipt_reverse.proposal_set.proposals
    )


def test_duplicate_proposal_hash_rejected() -> None:
    proposal = FixProposal(
        proposal_id="PROPOSAL-0000000000000000",
        issue_hash="abc",
        category="UNKNOWN",
        severity="LOW",
        target_path="x.py",
        fix_strategy="NO_ACTION",
        fix_summary="No action",
        invariant_preserved="UNKNOWN",
        deterministic_safe=True,
    )
    with pytest.raises(ValueError, match="duplicate proposal hashes"):
        # direct set creation is the deterministic duplicate-rejection boundary
        _ = FixProposalSet(proposals=(proposal, proposal), proposal_count=2, proposal_set_hash="x")


def test_empty_case() -> None:
    proposal_receipt = generate_fix_proposals(normalize_review_issues([]))

    assert proposal_receipt.proposal_status == "EMPTY"
    assert proposal_receipt.proposal_count == 0
    assert proposal_receipt.proposal_set.proposal_count == 0


def test_invalid_input_rejected() -> None:
    with pytest.raises(ValueError, match="IssueNormalizationReceipt"):
        generate_fix_proposals({"bad": "value"})  # type: ignore[arg-type]


def test_frozen_dataclass_immutability() -> None:
    receipt = generate_fix_proposals(_make_receipt())
    with pytest.raises(FrozenInstanceError):
        receipt.proposal_status = "EMPTY"  # type: ignore[misc]


def test_canonical_json_ordering_stability() -> None:
    issue_receipt = _make_receipt()
    proposal_receipt = generate_fix_proposals(issue_receipt)
    first_json = proposal_receipt.to_canonical_json()
    second_json = proposal_receipt.to_canonical_json()

    assert first_json == second_json
    assert proposal_receipt.to_canonical_bytes() == first_json.encode("utf-8")
