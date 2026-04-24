# SPDX-License-Identifier: MIT
"""Deterministic advisory fix proposal generation from canonical issue receipts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from typing import Any

from qec.analysis.issue_normalization_kernel import IssueNormalizationReceipt

_SEVERITY_RANK = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

_CATEGORY_TO_STRATEGY = {
    "DETERMINISM": "ORDERING_FIX",
    "CANONICALIZATION": "CANONICALIZATION_FIX",
    "HASH_INTEGRITY": "HASH_VALIDATION_FIX",
    "IMMUTABILITY": "IMMUTABILITY_ENFORCEMENT",
    "VALIDATION": "VALIDATION_HARDEN",
    "BOUNDS": "BOUNDS_ENFORCEMENT",
    "ORDERING": "ORDERING_FIX",
    "TEST_COVERAGE": "TEST_ADDITION",
    "SCOPE_GUARDRAIL": "SCOPE_RESTRICTION",
    "UNKNOWN": "NO_ACTION",
    "NAMING": "NO_ACTION",
    "DOCS": "NO_ACTION",
}

_ALLOWED_FIX_STRATEGIES = frozenset(
    {
        "VALIDATION_HARDEN",
        "CANONICALIZATION_FIX",
        "HASH_VALIDATION_FIX",
        "IMMUTABILITY_ENFORCEMENT",
        "ORDERING_FIX",
        "BOUNDS_ENFORCEMENT",
        "TEST_ADDITION",
        "SCOPE_RESTRICTION",
        "NO_ACTION",
    }
)



def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().split())


@dataclass(frozen=True)
class FixProposal:
    proposal_id: str
    issue_hash: str
    category: str
    severity: str
    target_path: str
    fix_strategy: str
    fix_summary: str
    invariant_preserved: str
    deterministic_safe: bool

    def __post_init__(self) -> None:
        if self.fix_strategy not in _ALLOWED_FIX_STRATEGIES:
            raise ValueError(f"invalid fix_strategy: {self.fix_strategy}")
        if not self.proposal_id.startswith("PROPOSAL-"):
            raise ValueError("proposal_id must start with PROPOSAL-")

    def _payload_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "issue_hash": self.issue_hash,
            "category": self.category,
            "severity": self.severity,
            "target_path": self.target_path,
            "fix_strategy": self.fix_strategy,
            "fix_summary": self.fix_summary,
            "invariant_preserved": self.invariant_preserved,
            "deterministic_safe": self.deterministic_safe,
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
class FixProposalSet:
    proposals: tuple[FixProposal, ...]
    proposal_count: int
    proposal_set_hash: str

    def __post_init__(self) -> None:
        if len(self.proposals) != self.proposal_count:
            raise ValueError("proposal_count must match the number of proposals")
        proposal_hashes = tuple(proposal.stable_hash() for proposal in self.proposals)
        if len(set(proposal_hashes)) != len(proposal_hashes):
            raise ValueError("duplicate proposal hashes are not allowed")

    def _payload_dict(self) -> dict[str, Any]:
        return {
            "proposals": [proposal.to_dict() for proposal in self.proposals],
            "proposal_count": self.proposal_count,
            "proposal_set_hash": self.proposal_set_hash,
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
class FixProposalReceipt:
    schema_version: str
    module_version: str
    proposal_status: str
    input_issue_set_hash: str
    proposal_set: FixProposalSet
    proposal_count: int

    def __post_init__(self) -> None:
        if self.proposal_status not in {"PROPOSED", "EMPTY"}:
            raise ValueError(f"invalid proposal_status: {self.proposal_status}")
        if self.proposal_count != self.proposal_set.proposal_count:
            raise ValueError("proposal_count must match proposal_set.proposal_count")

    def _payload_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "module_version": self.module_version,
            "proposal_status": self.proposal_status,
            "input_issue_set_hash": self.input_issue_set_hash,
            "proposal_set": self.proposal_set.to_dict(),
            "proposal_count": self.proposal_count,
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


def _proposal_sort_key(item: tuple[FixProposal, str]) -> tuple[int, str, str, str, str]:
    proposal, stable_hash = item
    return (
        -_SEVERITY_RANK[proposal.severity],
        proposal.category,
        proposal.target_path,
        proposal.fix_strategy,
        stable_hash,
    )


def _build_proposal_set(proposals: tuple[FixProposal, ...]) -> FixProposalSet:
    proposal_hash_pairs = tuple((proposal, proposal.stable_hash()) for proposal in proposals)
    sorted_pairs = tuple(sorted(proposal_hash_pairs, key=_proposal_sort_key))
    sorted_proposals = tuple(proposal for proposal, _ in sorted_pairs)
    hashes = tuple(stable_hash for _, stable_hash in sorted_pairs)
    if len(set(hashes)) != len(hashes):
        raise ValueError("duplicate proposal hashes are not allowed")
    proposal_set_hash = hashlib.sha256("|".join(hashes).encode("utf-8")).hexdigest()
    return FixProposalSet(
        proposals=sorted_proposals,
        proposal_count=len(sorted_proposals),
        proposal_set_hash=proposal_set_hash,
    )


def _derive_fix_strategy(category: str) -> str:
    return _CATEGORY_TO_STRATEGY.get(category, "NO_ACTION")


def _build_proposal_id(issue_hash: str, fix_strategy: str) -> str:
    payload = {"issue_hash": issue_hash, "fix_strategy": fix_strategy}
    return f"PROPOSAL-{hashlib.sha256(_canonical_json(payload).encode('utf-8')).hexdigest()[:16]}"


def _proposal_from_issue(issue_hash: str, issue: Any) -> FixProposal:
    fix_strategy = _derive_fix_strategy(issue.category)
    fix_summary = _normalize_text(issue.summary)

    return FixProposal(
        proposal_id=_build_proposal_id(issue_hash=issue_hash, fix_strategy=fix_strategy),
        issue_hash=issue_hash,
        category=issue.category,
        severity=issue.severity,
        target_path=issue.target_path,
        fix_strategy=fix_strategy,
        fix_summary=fix_summary,
        invariant_preserved=issue.invariant,
        deterministic_safe=True,
    )


def generate_fix_proposals(issue_receipt: IssueNormalizationReceipt) -> FixProposalReceipt:
    if not isinstance(issue_receipt, IssueNormalizationReceipt):
        raise ValueError("issue_receipt must be an IssueNormalizationReceipt")

    issues = issue_receipt.canonical_issue_set.issues
    issue_hashes = tuple(issue.stable_hash() for issue in issues)
    proposals = tuple(_proposal_from_issue(issue_hash, issue) for issue_hash, issue in zip(issue_hashes, issues, strict=True))
    proposal_set = _build_proposal_set(proposals)

    status = "EMPTY" if proposal_set.proposal_count == 0 else "PROPOSED"

    return FixProposalReceipt(
        schema_version="1.0",
        module_version="v148.2",
        proposal_status=status,
        input_issue_set_hash=issue_receipt.issue_set_hash,
        proposal_set=proposal_set,
        proposal_count=proposal_set.proposal_count,
    )


__all__ = [
    "FixProposal",
    "FixProposalSet",
    "FixProposalReceipt",
    "generate_fix_proposals",
]
