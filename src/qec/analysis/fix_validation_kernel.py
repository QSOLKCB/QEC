# SPDX-License-Identifier: MIT
"""Deterministic validation kernel for advisory fix proposals."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from typing import Any

from qec.analysis.fix_proposal_kernel import (
    FixProposal,
    FixProposalReceipt,
    FixProposalSet,
    _CATEGORY_TO_STRATEGY as _PROPOSAL_CATEGORY_TO_STRATEGY,
)

_ALLOWED_VALIDATION_STATUS = frozenset({"VALID", "INVALID", "UNSAFE", "INSUFFICIENT"})
_ALLOWED_RECEIPT_STATUS = frozenset({"ALL_VALID", "HAS_INVALID", "HAS_UNSAFE", "HAS_INSUFFICIENT", "EMPTY"})
_ALLOWED_SEVERITIES = frozenset({"LOW", "MEDIUM", "HIGH", "CRITICAL"})

_CATEGORY_TO_STRATEGY = _PROPOSAL_CATEGORY_TO_STRATEGY

_STRATEGY_TO_INVARIANT = {
    "VALIDATION_HARDEN": "FAIL_FAST_VALIDATION",
    "CANONICALIZATION_FIX": "CANONICAL_JSON",
    "HASH_VALIDATION_FIX": "STABLE_HASH",
    "IMMUTABILITY_ENFORCEMENT": "FROZEN_DATACLASS",
    "ORDERING_FIX": "DETERMINISTIC_ORDERING",
    "BOUNDS_ENFORCEMENT": "BOUNDED_OUTPUT",
    "SCOPE_RESTRICTION": "ANALYSIS_LAYER_ONLY",
}

_ALLOWED_TEST_ADDITION_INVARIANTS = frozenset({"TEST_COVERAGE"})

_VALIDATION_STATUS_RANK = {"INVALID": 0, "INSUFFICIENT": 1, "UNSAFE": 2, "VALID": 3}

_FORBIDDEN_DYNAMIC_TOKENS = (
    "http://",
    "https://",
    "timestamp",
    "datetime",
    "today",
    "now",
    "random",
    "uuid",
)

_FORBIDDEN_SCOPE_TOKENS = (
    "decoder",
    "runtime",
    "mutate",
    "mutation",
    "i/o",
    "external",
    "network",
    "api call",
)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


@dataclass(frozen=True)
class FixValidation:
    proposal_id: str
    issue_hash: str
    fix_strategy: str
    validation_status: str
    invariant_preserved: bool
    deterministic_safe: bool
    scope_compliant: bool
    consistency_score: float

    def __post_init__(self) -> None:
        if self.validation_status not in _ALLOWED_VALIDATION_STATUS:
            raise ValueError(f"invalid validation_status: {self.validation_status}")

    def _payload_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "issue_hash": self.issue_hash,
            "fix_strategy": self.fix_strategy,
            "validation_status": self.validation_status,
            "invariant_preserved": self.invariant_preserved,
            "deterministic_safe": self.deterministic_safe,
            "scope_compliant": self.scope_compliant,
            "consistency_score": self.consistency_score,
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
class FixValidationSet:
    validations: tuple[FixValidation, ...]
    validation_count: int
    validation_set_hash: str

    def __post_init__(self) -> None:
        if self.validation_count != len(self.validations):
            raise ValueError("validation_count must match len(validations)")

    def _payload_dict(self) -> dict[str, Any]:
        return {
            "validations": [item.to_dict() for item in self.validations],
            "validation_count": self.validation_count,
            "validation_set_hash": self.validation_set_hash,
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
class FixValidationReceipt:
    schema_version: str
    module_version: str
    validation_status: str
    input_proposal_set_hash: str
    validation_set: FixValidationSet
    valid_count: int
    invalid_count: int
    unsafe_count: int
    insufficient_count: int

    def __post_init__(self) -> None:
        if self.validation_status not in _ALLOWED_RECEIPT_STATUS:
            raise ValueError(f"invalid validation_status: {self.validation_status}")

    def _payload_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "module_version": self.module_version,
            "validation_status": self.validation_status,
            "input_proposal_set_hash": self.input_proposal_set_hash,
            "validation_set": self.validation_set.to_dict(),
            "valid_count": self.valid_count,
            "invalid_count": self.invalid_count,
            "unsafe_count": self.unsafe_count,
            "insufficient_count": self.insufficient_count,
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


def _proposal_is_deterministic(proposal: FixProposal) -> bool:
    text = proposal.fix_summary.strip()
    if text != " ".join(text.split()):
        return False
    lowered = text.lower()
    return not any(token in lowered for token in _FORBIDDEN_DYNAMIC_TOKENS)


def _normalize_target_path(target_path: str) -> str:
    """Return a canonical relative path string for scope checks."""
    normalized = target_path.strip().lower().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    parts = tuple(part for part in normalized.split("/") if part not in {"", "."})
    return "/".join(parts)


def _proposal_scope_compliant(proposal: FixProposal) -> bool:
    normalized_target = _normalize_target_path(proposal.target_path)
    target_parts = tuple(normalized_target.split("/")) if normalized_target else ()
    if target_parts[:3] == ("src", "qec", "decoder"):
        return False
    lowered_summary = proposal.fix_summary.lower()
    return not any(token in lowered_summary for token in _FORBIDDEN_SCOPE_TOKENS)


def _proposal_invariant_preserved(proposal: FixProposal) -> bool:
    if proposal.fix_strategy == "NO_ACTION":
        return False
    if proposal.fix_strategy == "TEST_ADDITION":
        return proposal.invariant_preserved in _ALLOWED_TEST_ADDITION_INVARIANTS
    expected = _STRATEGY_TO_INVARIANT.get(proposal.fix_strategy)
    return expected == proposal.invariant_preserved


def _proposal_consistent(proposal: FixProposal) -> bool:
    if proposal.severity not in _ALLOWED_SEVERITIES:
        return False
    expected_strategy = _CATEGORY_TO_STRATEGY.get(proposal.category, "NO_ACTION")
    if expected_strategy != proposal.fix_strategy:
        return False
    if proposal.fix_strategy in _STRATEGY_TO_INVARIANT:
        return proposal.invariant_preserved == _STRATEGY_TO_INVARIANT[proposal.fix_strategy]
    if proposal.fix_strategy == "TEST_ADDITION":
        return proposal.invariant_preserved in _ALLOWED_TEST_ADDITION_INVARIANTS
    return True


def _evaluate_proposal(proposal: FixProposal) -> FixValidation:
    deterministic_safe = proposal.deterministic_safe and _proposal_is_deterministic(proposal)
    scope_compliant = _proposal_scope_compliant(proposal)
    invariant_preserved = _proposal_invariant_preserved(proposal)
    consistent = _proposal_consistent(proposal)

    if not deterministic_safe or not scope_compliant:
        status = "UNSAFE"
        score = 0.0
    elif proposal.fix_strategy == "NO_ACTION":
        status = "INSUFFICIENT"
        score = 0.5
    elif not consistent or not invariant_preserved:
        status = "INVALID"
        score = 0.0
    else:
        status = "VALID"
        score = 1.0

    return FixValidation(
        proposal_id=proposal.proposal_id,
        issue_hash=proposal.issue_hash,
        fix_strategy=proposal.fix_strategy,
        validation_status=status,
        invariant_preserved=invariant_preserved,
        deterministic_safe=deterministic_safe,
        scope_compliant=scope_compliant,
        consistency_score=score,
    )


def _validation_sort_key(item: FixValidation) -> tuple[int, str, str, str]:
    return (_VALIDATION_STATUS_RANK[item.validation_status], item.fix_strategy, item.proposal_id, item.stable_hash())


def _build_validation_set(proposals: tuple[FixProposal, ...]) -> FixValidationSet:
    validations = tuple(_evaluate_proposal(proposal) for proposal in proposals)
    sorted_validations = tuple(sorted(validations, key=_validation_sort_key))
    validation_hashes = tuple(item.stable_hash() for item in sorted_validations)
    validation_set_hash = hashlib.sha256("|".join(validation_hashes).encode("utf-8")).hexdigest()
    return FixValidationSet(
        validations=sorted_validations,
        validation_count=len(sorted_validations),
        validation_set_hash=validation_set_hash,
    )


def _receipt_status(validation_set: FixValidationSet) -> str:
    if validation_set.validation_count == 0:
        return "EMPTY"
    statuses = tuple(item.validation_status for item in validation_set.validations)
    if "UNSAFE" in statuses:
        return "HAS_UNSAFE"
    if "INVALID" in statuses:
        return "HAS_INVALID"
    if "INSUFFICIENT" in statuses:
        return "HAS_INSUFFICIENT"
    return "ALL_VALID"


def validate_fix_proposals(proposal_receipt: FixProposalReceipt) -> FixValidationReceipt:
    if not isinstance(proposal_receipt, FixProposalReceipt):
        raise ValueError("proposal_receipt must be a FixProposalReceipt")

    proposal_set: FixProposalSet = proposal_receipt.proposal_set
    validation_set = _build_validation_set(proposal_set.proposals)

    valid_count = sum(1 for item in validation_set.validations if item.validation_status == "VALID")
    invalid_count = sum(1 for item in validation_set.validations if item.validation_status == "INVALID")
    unsafe_count = sum(1 for item in validation_set.validations if item.validation_status == "UNSAFE")
    insufficient_count = sum(1 for item in validation_set.validations if item.validation_status == "INSUFFICIENT")

    return FixValidationReceipt(
        schema_version="1.0",
        module_version="v148.3",
        validation_status=_receipt_status(validation_set),
        input_proposal_set_hash=proposal_set.proposal_set_hash,
        validation_set=validation_set,
        valid_count=valid_count,
        invalid_count=invalid_count,
        unsafe_count=unsafe_count,
        insufficient_count=insufficient_count,
    )


__all__ = [
    "FixValidation",
    "FixValidationSet",
    "FixValidationReceipt",
    "validate_fix_proposals",
]
