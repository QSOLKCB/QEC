# SPDX-License-Identifier: MIT
"""Deterministic counterfactual replay kernel for validated fix proposals."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from typing import Any

from qec.analysis.fix_validation_kernel import FixValidation, FixValidationReceipt

_ALLOWED_DOMINANCE_STATUS = frozenset({"NECESSARY", "EQUIVALENT", "DOMINATED", "UNRESOLVED"})
_ALLOWED_REPLAY_STATUS = frozenset({"ALL_RESOLVED", "HAS_EQUIVALENT", "HAS_DOMINATED", "HAS_UNRESOLVED", "EMPTY"})

_STRATEGY_PRIORITY = {
    "VALIDATION_HARDEN": 9,
    "CANONICALIZATION_FIX": 8,
    "HASH_VALIDATION_FIX": 7,
    "IMMUTABILITY_ENFORCEMENT": 6,
    "ORDERING_FIX": 5,
    "BOUNDS_ENFORCEMENT": 4,
    "TEST_ADDITION": 3,
    "SCOPE_RESTRICTION": 2,
    "NO_ACTION": 1,
}

_DOMINANCE_RANK = {
    "NECESSARY": 0,
    "EQUIVALENT": 1,
    "DOMINATED": 2,
    "UNRESOLVED": 3,
}

_NECESSITY_SCORE = {
    "NECESSARY": 1.0,
    "EQUIVALENT": 0.75,
    "UNRESOLVED": 0.5,
    "DOMINATED": 0.0,
}


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _severity_rank(validation: FixValidation) -> int:
    # Severity is not available on FixValidation; this symbolic kernel remains deterministic.
    return 0


def _invariant_strength(validation: FixValidation) -> int:
    return 1 if validation.invariant_preserved else 0


def _proposal_vector(validation: FixValidation) -> tuple[int, int, int]:
    return (
        _severity_rank(validation),
        _STRATEGY_PRIORITY.get(validation.fix_strategy, 0),
        _invariant_strength(validation),
    )


@dataclass(frozen=True)
class CounterfactualComparison:
    proposal_id: str
    comparison_group_id: str
    dominance_status: str
    equivalence_class: str
    necessity_score: float

    def __post_init__(self) -> None:
        if self.dominance_status not in _ALLOWED_DOMINANCE_STATUS:
            raise ValueError(f"invalid dominance_status: {self.dominance_status}")

    def _payload_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "comparison_group_id": self.comparison_group_id,
            "dominance_status": self.dominance_status,
            "equivalence_class": self.equivalence_class,
            "necessity_score": self.necessity_score,
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
        return hashlib.sha256(_canonical_json(self._payload_dict()).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CounterfactualReplaySet:
    comparisons: tuple[CounterfactualComparison, ...]
    comparison_count: int
    replay_set_hash: str

    def __post_init__(self) -> None:
        if self.comparison_count != len(self.comparisons):
            raise ValueError("comparison_count must match len(comparisons)")

    def _payload_dict(self) -> dict[str, Any]:
        return {
            "comparisons": [item.to_dict() for item in self.comparisons],
            "comparison_count": self.comparison_count,
            "replay_set_hash": self.replay_set_hash,
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
        return hashlib.sha256(_canonical_json(self._payload_dict()).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CounterfactualReplayReceipt:
    schema_version: str
    module_version: str
    replay_status: str
    input_validation_set_hash: str
    replay_set: CounterfactualReplaySet
    necessary_count: int
    equivalent_count: int
    dominated_count: int
    unresolved_count: int

    def __post_init__(self) -> None:
        if self.replay_status not in _ALLOWED_REPLAY_STATUS:
            raise ValueError(f"invalid replay_status: {self.replay_status}")

    def _payload_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "module_version": self.module_version,
            "replay_status": self.replay_status,
            "input_validation_set_hash": self.input_validation_set_hash,
            "replay_set": self.replay_set.to_dict(),
            "necessary_count": self.necessary_count,
            "equivalent_count": self.equivalent_count,
            "dominated_count": self.dominated_count,
            "unresolved_count": self.unresolved_count,
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
        return hashlib.sha256(_canonical_json(self._payload_dict()).encode("utf-8")).hexdigest()


def _comparison_sort_key(item: CounterfactualComparison) -> tuple[str, int, float, str, str]:
    return (
        item.comparison_group_id,
        _DOMINANCE_RANK[item.dominance_status],
        -item.necessity_score,
        item.proposal_id,
        item.stable_hash(),
    )


def _build_replay_set(comparisons: tuple[CounterfactualComparison, ...]) -> CounterfactualReplaySet:
    sorted_comparisons = tuple(sorted(comparisons, key=_comparison_sort_key))
    comparison_hashes = tuple(item.stable_hash() for item in sorted_comparisons)
    replay_set_hash = hashlib.sha256("|".join(comparison_hashes).encode("utf-8")).hexdigest()
    return CounterfactualReplaySet(
        comparisons=sorted_comparisons,
        comparison_count=len(sorted_comparisons),
        replay_set_hash=replay_set_hash,
    )


def _group_id_for_issue(issue_hash: str) -> str:
    return issue_hash


def _status_from_counts(
    comparison_count: int,
    equivalent_count: int,
    dominated_count: int,
    unresolved_count: int,
) -> str:
    if comparison_count == 0:
        return "EMPTY"
    if unresolved_count > 0:
        return "HAS_UNRESOLVED"
    if equivalent_count > 0:
        return "HAS_EQUIVALENT"
    if dominated_count > 0:
        return "HAS_DOMINATED"
    return "ALL_RESOLVED"


def _classify_group(issue_hash: str, validations: tuple[FixValidation, ...]) -> tuple[CounterfactualComparison, ...]:
    if len(validations) == 1:
        only = validations[0]
        return (
            CounterfactualComparison(
                proposal_id=only.proposal_id,
                comparison_group_id=_group_id_for_issue(issue_hash),
                dominance_status="NECESSARY",
                equivalence_class=f"eq-{issue_hash}-0",
                necessity_score=_NECESSITY_SCORE["NECESSARY"],
            ),
        )

    vectors = {validation.proposal_id: _proposal_vector(validation) for validation in validations}
    dominated_by_id: dict[str, bool] = {}
    for validation in validations:
        proposal_id = validation.proposal_id
        vector = vectors[proposal_id]
        dominated = False
        for other in validations:
            if other.proposal_id == proposal_id:
                continue
            other_vector = vectors[other.proposal_id]
            if all(a <= b for a, b in zip(vector, other_vector, strict=True)) and any(
                a < b for a, b in zip(vector, other_vector, strict=True)
            ):
                dominated = True
        dominated_by_id[proposal_id] = dominated

    nondominated_ids = tuple(
        sorted(item.proposal_id for item in validations if not dominated_by_id[item.proposal_id])
    )

    # Group all proposals by their vector to detect every equivalence class deterministically.
    vector_groups: dict[tuple[int, int, int], list[str]] = {}
    for validation in validations:
        vec = vectors[validation.proposal_id]
        vector_groups.setdefault(vec, []).append(validation.proposal_id)

    equivalence_sets: dict[tuple[int, int, int], tuple[str, ...]] = {
        vec: tuple(sorted(ids))
        for vec, ids in vector_groups.items()
        if len(ids) > 1
    }

    comparisons: list[CounterfactualComparison] = []
    for validation in validations:
        proposal_id = validation.proposal_id
        vec = vectors[proposal_id]
        if len(nondominated_ids) == 1 and proposal_id == nondominated_ids[0]:
            status = "NECESSARY"
        elif vec in equivalence_sets and not dominated_by_id[proposal_id]:
            status = "EQUIVALENT"
        elif dominated_by_id[proposal_id]:
            status = "DOMINATED"
        else:
            status = "UNRESOLVED"

        if vec in equivalence_sets:
            eq_id = "-".join(equivalence_sets[vec])
            equivalence_class = f"eq-{issue_hash}-{eq_id}"
        else:
            equivalence_class = f"eq-{issue_hash}-{proposal_id}"

        comparisons.append(
            CounterfactualComparison(
                proposal_id=proposal_id,
                comparison_group_id=_group_id_for_issue(issue_hash),
                dominance_status=status,
                equivalence_class=equivalence_class,
                necessity_score=_NECESSITY_SCORE[status],
            )
        )

    return tuple(comparisons)


def run_counterfactual_replay(validation_receipt: FixValidationReceipt) -> CounterfactualReplayReceipt:
    if not isinstance(validation_receipt, FixValidationReceipt):
        raise ValueError("validation_receipt must be a FixValidationReceipt")

    valid = tuple(item for item in validation_receipt.validation_set.validations if item.validation_status == "VALID")
    grouped: dict[str, list[FixValidation]] = defaultdict(list)
    for item in valid:
        grouped[item.issue_hash].append(item)

    comparisons: list[CounterfactualComparison] = []
    for issue_hash in sorted(grouped):
        group = tuple(sorted(grouped[issue_hash], key=lambda item: item.proposal_id))
        comparisons.extend(_classify_group(issue_hash=issue_hash, validations=group))

    replay_set = _build_replay_set(tuple(comparisons))
    necessary_count = sum(1 for item in replay_set.comparisons if item.dominance_status == "NECESSARY")
    equivalent_count = sum(1 for item in replay_set.comparisons if item.dominance_status == "EQUIVALENT")
    dominated_count = sum(1 for item in replay_set.comparisons if item.dominance_status == "DOMINATED")
    unresolved_count = sum(1 for item in replay_set.comparisons if item.dominance_status == "UNRESOLVED")

    return CounterfactualReplayReceipt(
        schema_version="1.0",
        module_version="v148.4",
        replay_status=_status_from_counts(
            comparison_count=replay_set.comparison_count,
            equivalent_count=equivalent_count,
            dominated_count=dominated_count,
            unresolved_count=unresolved_count,
        ),
        input_validation_set_hash=validation_receipt.validation_set.validation_set_hash,
        replay_set=replay_set,
        necessary_count=necessary_count,
        equivalent_count=equivalent_count,
        dominated_count=dominated_count,
        unresolved_count=unresolved_count,
    )


__all__ = [
    "CounterfactualComparison",
    "CounterfactualReplaySet",
    "CounterfactualReplayReceipt",
    "run_counterfactual_replay",
]
