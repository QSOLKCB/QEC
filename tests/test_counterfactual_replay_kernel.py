from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json

import pytest

from qec.analysis.counterfactual_replay_kernel import (
    CounterfactualComparison,
    run_counterfactual_replay,
)
from qec.analysis.fix_validation_kernel import (
    FixValidation,
    FixValidationReceipt,
    FixValidationSet,
)


def _validation(
    *,
    proposal_id: str,
    issue_hash: str,
    fix_strategy: str,
    validation_status: str = "VALID",
    invariant_preserved: bool = True,
    deterministic_safe: bool = True,
    scope_compliant: bool = True,
    consistency_score: float = 1.0,
) -> FixValidation:
    return FixValidation(
        proposal_id=proposal_id,
        issue_hash=issue_hash,
        fix_strategy=fix_strategy,
        validation_status=validation_status,
        invariant_preserved=invariant_preserved,
        deterministic_safe=deterministic_safe,
        scope_compliant=scope_compliant,
        consistency_score=consistency_score,
    )


def _validation_receipt(validations: tuple[FixValidation, ...]) -> FixValidationReceipt:
    hashes = tuple(sorted(item.stable_hash() for item in validations))
    validation_set_hash = hashlib.sha256("|".join(hashes).encode("utf-8")).hexdigest()
    validation_set = FixValidationSet(
        validations=validations,
        validation_count=len(validations),
        validation_set_hash=validation_set_hash,
    )
    statuses = {item.validation_status for item in validations}
    if len(validations) == 0:
        status = "EMPTY"
    elif statuses == {"VALID"}:
        status = "ALL_VALID"
    elif "UNSAFE" in statuses:
        status = "HAS_UNSAFE"
    elif "INVALID" in statuses:
        status = "HAS_INVALID"
    else:
        status = "HAS_INSUFFICIENT"
    return FixValidationReceipt(
        schema_version="1.0",
        module_version="v148.3",
        validation_status=status,
        input_proposal_set_hash="proposal-set-hash",
        validation_set=validation_set,
        valid_count=sum(1 for item in validations if item.validation_status == "VALID"),
        invalid_count=sum(1 for item in validations if item.validation_status == "INVALID"),
        unsafe_count=sum(1 for item in validations if item.validation_status == "UNSAFE"),
        insufficient_count=sum(1 for item in validations if item.validation_status == "INSUFFICIENT"),
    )


def test_single_valid_proposal_is_necessary() -> None:
    receipt = run_counterfactual_replay(
        _validation_receipt(
            (
                _validation(
                    proposal_id="PROPOSAL-1",
                    issue_hash="issue-a",
                    fix_strategy="ORDERING_FIX",
                ),
            )
        )
    )

    assert receipt.replay_status == "ALL_RESOLVED"
    assert receipt.necessary_count == 1
    comparison = receipt.replay_set.comparisons[0]
    assert comparison.dominance_status == "NECESSARY"
    assert comparison.necessity_score == 1.0


def test_multiple_identical_proposals_are_equivalent() -> None:
    receipt = run_counterfactual_replay(
        _validation_receipt(
            (
                _validation(proposal_id="PROPOSAL-2", issue_hash="issue-e", fix_strategy="ORDERING_FIX"),
                _validation(proposal_id="PROPOSAL-3", issue_hash="issue-e", fix_strategy="ORDERING_FIX"),
            )
        )
    )

    assert receipt.replay_status == "HAS_EQUIVALENT"
    assert receipt.equivalent_count == 2
    assert all(item.dominance_status == "EQUIVALENT" for item in receipt.replay_set.comparisons)


def test_non_representative_identical_vectors_are_equivalent() -> None:
    # Three non-dominated proposals: NR2 and NR3 share the same vector but are
    # NOT the representative (NR1 sorts first).  The fix must still label them
    # EQUIVALENT and give them an identical equivalence_class that differs from NR1's.
    receipt = run_counterfactual_replay(
        _validation_receipt(
            (
                _validation(
                    proposal_id="PROPOSAL-NR1",
                    issue_hash="issue-nr",
                    fix_strategy="CANONICALIZATION_FIX",
                    invariant_preserved=False,
                ),
                _validation(
                    proposal_id="PROPOSAL-NR2",
                    issue_hash="issue-nr",
                    fix_strategy="ORDERING_FIX",
                    invariant_preserved=True,
                ),
                _validation(
                    proposal_id="PROPOSAL-NR3",
                    issue_hash="issue-nr",
                    fix_strategy="ORDERING_FIX",
                    invariant_preserved=True,
                ),
            )
        )
    )

    statuses = {item.proposal_id: item.dominance_status for item in receipt.replay_set.comparisons}
    eq_classes = {item.proposal_id: item.equivalence_class for item in receipt.replay_set.comparisons}

    assert statuses["PROPOSAL-NR2"] == "EQUIVALENT"
    assert statuses["PROPOSAL-NR3"] == "EQUIVALENT"
    assert eq_classes["PROPOSAL-NR2"] == eq_classes["PROPOSAL-NR3"]
    assert eq_classes["PROPOSAL-NR1"] != eq_classes["PROPOSAL-NR2"]
    assert receipt.equivalent_count == 2


def test_clear_dominance_case() -> None:
    receipt = run_counterfactual_replay(
        _validation_receipt(
            (
                _validation(proposal_id="PROPOSAL-4", issue_hash="issue-d", fix_strategy="VALIDATION_HARDEN"),
                _validation(proposal_id="PROPOSAL-5", issue_hash="issue-d", fix_strategy="NO_ACTION"),
            )
        )
    )

    statuses = {item.proposal_id: item.dominance_status for item in receipt.replay_set.comparisons}
    assert statuses["PROPOSAL-4"] == "NECESSARY"
    assert statuses["PROPOSAL-5"] == "DOMINATED"
    assert receipt.necessary_count == 1
    assert receipt.dominated_count == 1


def test_mixed_ambiguity_unresolved() -> None:
    receipt = run_counterfactual_replay(
        _validation_receipt(
            (
                _validation(proposal_id="PROPOSAL-6", issue_hash="issue-u", fix_strategy="ORDERING_FIX", invariant_preserved=True),
                _validation(proposal_id="PROPOSAL-7", issue_hash="issue-u", fix_strategy="CANONICALIZATION_FIX", invariant_preserved=False),
            )
        )
    )

    statuses = {item.proposal_id: item.dominance_status for item in receipt.replay_set.comparisons}
    assert statuses["PROPOSAL-6"] == "UNRESOLVED"
    assert statuses["PROPOSAL-7"] == "UNRESOLVED"
    assert receipt.replay_status == "HAS_UNRESOLVED"
    assert receipt.unresolved_count == 2


def test_replay_stability_hash_equality() -> None:
    validations = (
        _validation(proposal_id="PROPOSAL-8", issue_hash="issue-s", fix_strategy="ORDERING_FIX"),
        _validation(proposal_id="PROPOSAL-9", issue_hash="issue-s", fix_strategy="ORDERING_FIX"),
    )

    first = run_counterfactual_replay(_validation_receipt(validations))
    second = run_counterfactual_replay(_validation_receipt(validations))

    assert first.to_canonical_json() == second.to_canonical_json()
    assert first.to_canonical_bytes() == second.to_canonical_bytes()
    assert first.stable_hash() == second.stable_hash()


def test_sorting_stability_with_shuffled_input() -> None:
    a = _validation(proposal_id="PROPOSAL-A", issue_hash="issue-z", fix_strategy="NO_ACTION")
    b = _validation(proposal_id="PROPOSAL-B", issue_hash="issue-z", fix_strategy="VALIDATION_HARDEN")
    c = _validation(proposal_id="PROPOSAL-C", issue_hash="issue-y", fix_strategy="ORDERING_FIX")

    one = run_counterfactual_replay(_validation_receipt((a, b, c)))
    two = run_counterfactual_replay(_validation_receipt((c, b, a)))

    assert one.to_canonical_json() == two.to_canonical_json()


def test_empty_case() -> None:
    receipt = run_counterfactual_replay(_validation_receipt(tuple()))

    assert receipt.replay_status == "EMPTY"
    assert receipt.replay_set.comparison_count == 0
    assert receipt.necessary_count == 0
    assert receipt.equivalent_count == 0
    assert receipt.dominated_count == 0
    assert receipt.unresolved_count == 0


def test_frozen_dataclass_immutability() -> None:
    comparison = CounterfactualComparison(
        proposal_id="PROPOSAL-IMMUT",
        comparison_group_id="group-1",
        dominance_status="NECESSARY",
        equivalence_class="eq-1",
        necessity_score=1.0,
    )

    with pytest.raises(FrozenInstanceError):
        comparison.dominance_status = "DOMINATED"  # type: ignore[misc]


def test_canonical_json_ordering() -> None:
    receipt = run_counterfactual_replay(
        _validation_receipt(
            (
                _validation(proposal_id="PROPOSAL-10", issue_hash="issue-canon", fix_strategy="ORDERING_FIX"),
            )
        )
    )

    expected = json.dumps(
        receipt.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    assert receipt.to_canonical_json() == expected
