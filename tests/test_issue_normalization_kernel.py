from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qec.analysis.issue_normalization_kernel import normalize_review_issues


def test_valid_normalization_and_replay_stability() -> None:
    raw_issues = [
        {
            "source": "GITHUB_REVIEW",
            "severity": "HIGH",
            "category": "ORDERING",
            "target_path": "src\\qec\\analysis\\api.py",
            "summary": "Sort issues deterministically",
            "invariant": "DETERMINISTIC_ORDERING",
        },
        {
            "source": "MANUAL",
            "severity": "LOW",
            "target_path": "src/qec/analysis/issue_normalization_kernel.py",
            "body": "Canonical json output must be identical across runs. Extra details here.",
        },
    ]

    receipt_a = normalize_review_issues(raw_issues)
    receipt_b = normalize_review_issues(raw_issues)

    assert receipt_a.normalization_status == "NORMALIZED"
    assert receipt_a.to_dict() == receipt_b.to_dict()
    assert receipt_a.to_canonical_json() == receipt_b.to_canonical_json()
    assert receipt_a.issue_set_hash == receipt_b.issue_set_hash
    assert receipt_a.stable_hash() == receipt_b.stable_hash()


def test_dict_ordering_stability() -> None:
    issue_a = {
        "source": "COPILOT",
        "severity": "MEDIUM",
        "target_path": "src/qec/analysis/mod.py",
        "summary": "Normalize canonical JSON payload",
    }
    issue_b = {
        "summary": "Normalize canonical JSON payload",
        "target_path": "src/qec/analysis/mod.py",
        "severity": "MEDIUM",
        "source": "COPILOT",
    }

    receipt_a = normalize_review_issues([issue_a])
    receipt_b = normalize_review_issues([issue_b])

    assert receipt_a.to_dict() == receipt_b.to_dict()
    assert receipt_a.stable_hash() == receipt_b.stable_hash()


def test_sorting_stability_for_shuffled_input() -> None:
    first = {
        "source": "UNKNOWN",
        "severity": "LOW",
        "summary": "validation should reject malformed payloads",
        "target_path": "z.py",
    }
    second = {
        "source": "UNKNOWN",
        "severity": "CRITICAL",
        "summary": "hash integrity must remain stable",
        "target_path": "a.py",
    }

    receipt_forward = normalize_review_issues([first, second])
    receipt_reverse = normalize_review_issues([second, first])

    forward_issues = receipt_forward.canonical_issue_set.issues
    reverse_issues = receipt_reverse.canonical_issue_set.issues

    assert tuple(issue.issue_id for issue in forward_issues) == tuple(issue.issue_id for issue in reverse_issues)
    assert receipt_forward.to_canonical_json() == receipt_reverse.to_canonical_json()


def test_missing_category_and_invariant_classification() -> None:
    receipt = normalize_review_issues(
        [
            {
                "source": "UNKNOWN",
                "summary": "Stable hash should use sha256 canonical payload",
                "target_path": "",
            }
        ]
    )
    issue = receipt.canonical_issue_set.issues[0]
    assert issue.category == "HASH_INTEGRITY"
    assert issue.invariant == "STABLE_HASH"


def test_missing_severity_defaults_to_medium() -> None:
    receipt = normalize_review_issues(
        [
            {
                "source": "MANUAL",
                "summary": "validate malformed records",
                "target_path": "",
            }
        ]
    )
    assert receipt.canonical_issue_set.issues[0].severity == "MEDIUM"


def test_empty_list_produces_empty_status() -> None:
    receipt = normalize_review_issues([])
    assert receipt.normalization_status == "EMPTY"
    assert receipt.input_issue_count == 0
    assert receipt.issue_count == 0


@pytest.mark.parametrize(
    "payload",
    [
        [
            {
                "source": "MANUAL",
                "summary": "same issue",
                "target_path": "x.py",
            },
            {
                "source": "MANUAL",
                "summary": "same issue",
                "target_path": "x.py",
            },
        ],
    ],
)
def test_duplicate_normalized_issues_rejected(payload: list[dict[str, str]]) -> None:
    with pytest.raises(ValueError, match="duplicate normalized issues"):
        normalize_review_issues(payload)


@pytest.mark.parametrize(
    "payload,expected",
    [
        ([{"source": "BAD", "summary": "x", "target_path": ""}], "invalid source"),
        ([{"category": "BAD", "summary": "x", "target_path": ""}], "invalid category"),
        ([{"severity": "BAD", "summary": "x", "target_path": ""}], "invalid severity"),
        ([{"invariant": "BAD", "summary": "x", "target_path": ""}], "invalid invariant"),
    ],
)
def test_invalid_enums_rejected(payload: list[dict[str, str]], expected: str) -> None:
    with pytest.raises(ValueError, match=expected):
        normalize_review_issues(payload)


def test_receipt_is_frozen() -> None:
    receipt = normalize_review_issues([{"summary": "test coverage regression", "target_path": ""}])
    with pytest.raises(FrozenInstanceError):
        receipt.normalization_status = "INVALID_INPUT"  # type: ignore[misc]


def test_null_byte_path_rejected() -> None:
    with pytest.raises(ValueError, match="null bytes"):
        normalize_review_issues([{"summary": "x", "target_path": "abc\x00def"}])
