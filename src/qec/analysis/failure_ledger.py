# SPDX-License-Identifier: MIT
"""v148.7 — Failure Ledger (Expanded).

Deterministic, immutable failure ledger linking review issues, proposals,
validations, counterfactual outcomes, and replay receipts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qec.analysis.adversarial_determinism_battery import AdversarialDeterminismReceipt
from qec.analysis.canonical_hashing import canonical_bytes, canonical_json, sha256_hex
from qec.analysis.counterfactual_replay_kernel import CounterfactualReplayReceipt
from qec.analysis.cross_environment_replay_kernel import CrossEnvironmentReplayReceipt
from qec.analysis.fix_proposal_kernel import FixProposalReceipt
from qec.analysis.fix_validation_kernel import FixValidationReceipt
from qec.analysis.issue_normalization_kernel import IssueNormalizationReceipt

SCHEMA_VERSION = "1.0"
MODULE_VERSION = "v148.7"

_FAILURE_TYPES: tuple[str, ...] = (
    "VALIDATION_FAILURE",
    "MISMATCH",
    "UNSAFE",
    "INSUFFICIENT",
    "REPLAY_FAILURE",
    "DETERMINISM_FAILURE",
    "ADVERSARIAL_FAILURE",
    "CROSS_ENV_FAILURE",
    "UNKNOWN_FAILURE",
)

_SEVERITIES: tuple[str, ...] = ("CRITICAL", "HIGH", "MEDIUM", "LOW")
_SEVERITY_RANK = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1, "LOW": 0}


@dataclass(frozen=True)
class FailureRecord:
    failure_id: str
    source_module: str
    failure_type: str
    severity: str
    origin_hash: str
    description: str

    def __post_init__(self) -> None:
        if self.failure_type not in _FAILURE_TYPES:
            raise ValueError(f"invalid failure_type: {self.failure_type}")
        if self.severity not in _SEVERITIES:
            raise ValueError(f"invalid severity: {self.severity}")

    def _payload_dict(self) -> dict[str, Any]:
        return {
            "failure_id": self.failure_id,
            "source_module": self.source_module,
            "failure_type": self.failure_type,
            "severity": self.severity,
            "origin_hash": self.origin_hash,
            "description": self.description,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_dict()
        payload["stable_hash"] = self.stable_hash()
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self._payload_dict())


@dataclass(frozen=True)
class FailureLedgerEntry:
    entry_id: str
    failure_record: FailureRecord
    issue_hash: str
    proposal_hash: str
    validation_hash: str
    counterfactual_hash: str
    adversarial_hash: str
    cross_env_hash: str
    lineage_chain: tuple[str, str, str, str, str, str, str]

    def __post_init__(self) -> None:
        expected = (
            self.failure_record.origin_hash,
            self.issue_hash,
            self.proposal_hash,
            self.validation_hash,
            self.counterfactual_hash,
            self.adversarial_hash,
            self.cross_env_hash,
        )
        if self.lineage_chain != expected:
            raise ValueError("lineage_chain must match linked hash fields")

    def _payload_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "failure_record": self.failure_record.to_dict(),
            "issue_hash": self.issue_hash,
            "proposal_hash": self.proposal_hash,
            "validation_hash": self.validation_hash,
            "counterfactual_hash": self.counterfactual_hash,
            "adversarial_hash": self.adversarial_hash,
            "cross_env_hash": self.cross_env_hash,
            "lineage_chain": self.lineage_chain,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_dict()
        payload["stable_hash"] = self.stable_hash()
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self._payload_dict())


@dataclass(frozen=True)
class FailureLedger:
    entries: tuple[FailureLedgerEntry, ...]
    entry_count: int
    ledger_hash: str

    def __post_init__(self) -> None:
        if self.entry_count != len(self.entries):
            raise ValueError("entry_count must match len(entries)")

    def _payload_dict(self) -> dict[str, Any]:
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "entry_count": self.entry_count,
            "ledger_hash": self.ledger_hash,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_dict()
        payload["stable_hash"] = self.stable_hash()
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self._payload_dict())


@dataclass(frozen=True)
class FailureLedgerReceipt:
    schema_version: str
    module_version: str
    ledger_status: str
    input_hash: str
    ledger: FailureLedger
    failure_count: int
    typed_counts: dict[str, int]
    suppression_rate: int

    def __post_init__(self) -> None:
        if self.ledger_status not in {"VALID", "INVALID_INPUT", "EMPTY"}:
            raise ValueError(f"invalid ledger_status: {self.ledger_status}")
        if self.failure_count != self.ledger.entry_count:
            raise ValueError("failure_count must match ledger.entry_count")
        if self.suppression_rate != 0:
            raise ValueError("suppression_rate must be 0")

        expected_keys = tuple(sorted(_FAILURE_TYPES))
        actual_keys = tuple(sorted(self.typed_counts.keys()))
        if actual_keys != expected_keys:
            raise ValueError("typed_counts must include all failure_type categories")

    def _payload_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "module_version": self.module_version,
            "ledger_status": self.ledger_status,
            "input_hash": self.input_hash,
            "ledger": self.ledger.to_dict(),
            "failure_count": self.failure_count,
            "typed_counts": {key: self.typed_counts[key] for key in sorted(self.typed_counts)},
            "suppression_rate": self.suppression_rate,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_dict()
        payload["stable_hash"] = self.stable_hash()
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self._payload_dict())


def _build_failure_id(source_module: str, failure_type: str, origin_hash: str, description: str) -> str:
    return f"FAIL-{sha256_hex({'source_module': source_module, 'failure_type': failure_type, 'origin_hash': origin_hash, 'description': description})[:16]}"


def _build_entry_id(failure: FailureRecord, lineage_chain: tuple[str, str, str, str, str, str, str]) -> str:
    return f"LEDGER-{sha256_hex({'failure_hash': failure.stable_hash(), 'lineage_chain': lineage_chain})[:16]}"


def _build_entry(
    *,
    failure: FailureRecord,
    issue_hash: str,
    proposal_hash: str,
    validation_hash: str,
    counterfactual_hash: str,
    adversarial_hash: str,
    cross_env_hash: str,
) -> FailureLedgerEntry:
    lineage_chain = (
        failure.origin_hash,
        issue_hash,
        proposal_hash,
        validation_hash,
        counterfactual_hash,
        adversarial_hash,
        cross_env_hash,
    )
    return FailureLedgerEntry(
        entry_id=_build_entry_id(failure, lineage_chain),
        failure_record=failure,
        issue_hash=issue_hash,
        proposal_hash=proposal_hash,
        validation_hash=validation_hash,
        counterfactual_hash=counterfactual_hash,
        adversarial_hash=adversarial_hash,
        cross_env_hash=cross_env_hash,
        lineage_chain=lineage_chain,
    )


def _entry_sort_key(entry: FailureLedgerEntry) -> tuple[int, str, str, str]:
    return (
        -_SEVERITY_RANK[entry.failure_record.severity],
        entry.failure_record.failure_type,
        entry.entry_id,
        entry.stable_hash(),
    )


def _empty_typed_counts() -> dict[str, int]:
    return {key: 0 for key in sorted(_FAILURE_TYPES)}


def _severity_for_validation(status: str) -> str:
    if status == "UNSAFE":
        return "CRITICAL"
    if status == "INVALID":
        return "HIGH"
    if status == "INSUFFICIENT":
        return "MEDIUM"
    return "LOW"


def build_failure_ledger(
    issue_receipt: IssueNormalizationReceipt,
    proposal_receipt: FixProposalReceipt,
    validation_receipt: FixValidationReceipt,
    counterfactual_receipt: CounterfactualReplayReceipt,
    adversarial_receipt: AdversarialDeterminismReceipt,
    cross_env_receipt: CrossEnvironmentReplayReceipt,
) -> FailureLedgerReceipt:
    if not isinstance(issue_receipt, IssueNormalizationReceipt):
        raise ValueError("issue_receipt must be an IssueNormalizationReceipt")
    if not isinstance(proposal_receipt, FixProposalReceipt):
        raise ValueError("proposal_receipt must be a FixProposalReceipt")
    if not isinstance(validation_receipt, FixValidationReceipt):
        raise ValueError("validation_receipt must be a FixValidationReceipt")
    if not isinstance(counterfactual_receipt, CounterfactualReplayReceipt):
        raise ValueError("counterfactual_receipt must be a CounterfactualReplayReceipt")
    if not isinstance(adversarial_receipt, AdversarialDeterminismReceipt):
        raise ValueError("adversarial_receipt must be an AdversarialDeterminismReceipt")
    if not isinstance(cross_env_receipt, CrossEnvironmentReplayReceipt):
        raise ValueError("cross_env_receipt must be a CrossEnvironmentReplayReceipt")

    issue_hashes = tuple(issue.stable_hash() for issue in issue_receipt.canonical_issue_set.issues)
    proposals_by_id = {proposal.proposal_id: proposal for proposal in proposal_receipt.proposal_set.proposals}
    proposal_hash_by_id = {proposal_id: proposal.stable_hash() for proposal_id, proposal in proposals_by_id.items()}
    validation_by_id = {validation.proposal_id: validation for validation in validation_receipt.validation_set.validations}
    counterfactual_by_id = {
        comparison.proposal_id: comparison for comparison in counterfactual_receipt.replay_set.comparisons
    }

    adversarial_hash = adversarial_receipt.stable_hash()
    cross_env_hash = cross_env_receipt.stable_hash

    entries: list[FailureLedgerEntry] = []
    extracted_failure_ids: list[str] = []

    for validation in validation_receipt.validation_set.validations:
        if validation.validation_status == "VALID":
            continue
        failure_type = (
            "UNSAFE"
            if validation.validation_status == "UNSAFE"
            else "INSUFFICIENT"
            if validation.validation_status == "INSUFFICIENT"
            else "VALIDATION_FAILURE"
        )
        severity = _severity_for_validation(validation.validation_status)
        description = (
            f"validation status {validation.validation_status} "
            f"for proposal {validation.proposal_id}"
        )
        failure = FailureRecord(
            failure_id=_build_failure_id(
                "qec.analysis.fix_validation_kernel",
                failure_type,
                validation.stable_hash(),
                description,
            ),
            source_module="qec.analysis.fix_validation_kernel",
            failure_type=failure_type,
            severity=severity,
            origin_hash=validation.stable_hash(),
            description=description,
        )
        entry = _build_entry(
            failure=failure,
            issue_hash=validation.issue_hash if validation.issue_hash in issue_hashes else "NONE",
            proposal_hash=proposal_hash_by_id.get(validation.proposal_id, "NONE"),
            validation_hash=validation.stable_hash(),
            counterfactual_hash=counterfactual_by_id.get(validation.proposal_id).stable_hash()
            if validation.proposal_id in counterfactual_by_id
            else "NONE",
            adversarial_hash=adversarial_hash,
            cross_env_hash=cross_env_hash,
        )
        entries.append(entry)
        extracted_failure_ids.append(failure.failure_id)

    for comparison in counterfactual_receipt.replay_set.comparisons:
        if comparison.dominance_status not in {"DOMINATED", "UNRESOLVED"}:
            continue
        description = (
            f"counterfactual dominance {comparison.dominance_status} "
            f"for proposal {comparison.proposal_id}"
        )
        failure = FailureRecord(
            failure_id=_build_failure_id(
                "qec.analysis.counterfactual_replay_kernel",
                "REPLAY_FAILURE",
                comparison.stable_hash(),
                description,
            ),
            source_module="qec.analysis.counterfactual_replay_kernel",
            failure_type="REPLAY_FAILURE",
            severity="MEDIUM",
            origin_hash=comparison.stable_hash(),
            description=description,
        )
        issue_hash = comparison.comparison_group_id if comparison.comparison_group_id in issue_hashes else "NONE"
        entry = _build_entry(
            failure=failure,
            issue_hash=issue_hash,
            proposal_hash=proposal_hash_by_id.get(comparison.proposal_id, "NONE"),
            validation_hash=validation_by_id.get(comparison.proposal_id).stable_hash()
            if comparison.proposal_id in validation_by_id
            else "NONE",
            counterfactual_hash=comparison.stable_hash(),
            adversarial_hash=adversarial_hash,
            cross_env_hash=cross_env_hash,
        )
        entries.append(entry)
        extracted_failure_ids.append(failure.failure_id)

    if adversarial_receipt.fail_count > 0 or adversarial_receipt.false_positive_detected:
        description = f"adversarial battery status {adversarial_receipt.battery_status}"
        failure = FailureRecord(
            failure_id=_build_failure_id(
                "qec.analysis.adversarial_determinism_battery",
                "ADVERSARIAL_FAILURE",
                adversarial_hash,
                description,
            ),
            source_module="qec.analysis.adversarial_determinism_battery",
            failure_type="ADVERSARIAL_FAILURE",
            severity="HIGH",
            origin_hash=adversarial_hash,
            description=description,
        )
        entries.append(
            _build_entry(
                failure=failure,
                issue_hash="NONE",
                proposal_hash="NONE",
                validation_hash="NONE",
                counterfactual_hash="NONE",
                adversarial_hash=adversarial_hash,
                cross_env_hash=cross_env_hash,
            )
        )
        extracted_failure_ids.append(failure.failure_id)

    if not adversarial_receipt.determinism_pass or not adversarial_receipt.hash_stability_pass:
        description = (
            f"adversarial determinism={adversarial_receipt.determinism_pass} "
            f"hash_stability={adversarial_receipt.hash_stability_pass}"
        )
        failure = FailureRecord(
            failure_id=_build_failure_id(
                "qec.analysis.adversarial_determinism_battery",
                "DETERMINISM_FAILURE",
                adversarial_hash,
                description,
            ),
            source_module="qec.analysis.adversarial_determinism_battery",
            failure_type="DETERMINISM_FAILURE",
            severity="CRITICAL",
            origin_hash=adversarial_hash,
            description=description,
        )
        entries.append(
            _build_entry(
                failure=failure,
                issue_hash="NONE",
                proposal_hash="NONE",
                validation_hash="NONE",
                counterfactual_hash="NONE",
                adversarial_hash=adversarial_hash,
                cross_env_hash=cross_env_hash,
            )
        )
        extracted_failure_ids.append(failure.failure_id)

    if cross_env_receipt.comparison.comparison_status == "MISMATCH":
        description = f"cross-environment mismatch classification {cross_env_receipt.comparison.mismatch_classification}"
        failure = FailureRecord(
            failure_id=_build_failure_id(
                "qec.analysis.cross_environment_replay_kernel",
                "MISMATCH",
                cross_env_hash,
                description,
            ),
            source_module="qec.analysis.cross_environment_replay_kernel",
            failure_type="MISMATCH",
            severity="HIGH",
            origin_hash=cross_env_hash,
            description=description,
        )
        entries.append(
            _build_entry(
                failure=failure,
                issue_hash="NONE",
                proposal_hash="NONE",
                validation_hash="NONE",
                counterfactual_hash="NONE",
                adversarial_hash=adversarial_hash,
                cross_env_hash=cross_env_hash,
            )
        )
        extracted_failure_ids.append(failure.failure_id)

    if cross_env_receipt.receipt_status == "INSUFFICIENT_ENVIRONMENTS" or cross_env_receipt.failure_recorded:
        description = f"cross-environment status {cross_env_receipt.receipt_status}"
        failure = FailureRecord(
            failure_id=_build_failure_id(
                "qec.analysis.cross_environment_replay_kernel",
                "CROSS_ENV_FAILURE",
                cross_env_hash,
                description,
            ),
            source_module="qec.analysis.cross_environment_replay_kernel",
            failure_type="CROSS_ENV_FAILURE",
            severity="HIGH",
            origin_hash=cross_env_hash,
            description=description,
        )
        entries.append(
            _build_entry(
                failure=failure,
                issue_hash="NONE",
                proposal_hash="NONE",
                validation_hash="NONE",
                counterfactual_hash="NONE",
                adversarial_hash=adversarial_hash,
                cross_env_hash=cross_env_hash,
            )
        )
        extracted_failure_ids.append(failure.failure_id)

    if not cross_env_receipt.determinism_preserved:
        description = "cross-environment determinism not preserved"
        failure = FailureRecord(
            failure_id=_build_failure_id(
                "qec.analysis.cross_environment_replay_kernel",
                "DETERMINISM_FAILURE",
                cross_env_hash,
                description,
            ),
            source_module="qec.analysis.cross_environment_replay_kernel",
            failure_type="DETERMINISM_FAILURE",
            severity="CRITICAL",
            origin_hash=cross_env_hash,
            description=description,
        )
        entries.append(
            _build_entry(
                failure=failure,
                issue_hash="NONE",
                proposal_hash="NONE",
                validation_hash="NONE",
                counterfactual_hash="NONE",
                adversarial_hash=adversarial_hash,
                cross_env_hash=cross_env_hash,
            )
        )
        extracted_failure_ids.append(failure.failure_id)

    sorted_entries = tuple(sorted(entries, key=_entry_sort_key))
    ledger_hash = sha256_hex({"entry_hashes": tuple(entry.stable_hash() for entry in sorted_entries)})
    ledger = FailureLedger(entries=sorted_entries, entry_count=len(sorted_entries), ledger_hash=ledger_hash)

    typed_counts = _empty_typed_counts()
    for entry in sorted_entries:
        typed_counts[entry.failure_record.failure_type] += 1

    represented = tuple(entry.failure_record.failure_id for entry in sorted_entries)
    if tuple(extracted_failure_ids) != represented:
        if len(tuple(extracted_failure_ids)) != len(represented) or set(extracted_failure_ids) != set(represented):
            raise ValueError("suppression detected: extracted failures not fully represented")

    input_hash = sha256_hex(
        {
            "issue_receipt_hash": issue_receipt.stable_hash(),
            "proposal_receipt_hash": proposal_receipt.stable_hash(),
            "validation_receipt_hash": validation_receipt.stable_hash(),
            "counterfactual_receipt_hash": counterfactual_receipt.stable_hash(),
            "adversarial_receipt_hash": adversarial_receipt.stable_hash(),
            "cross_env_receipt_hash": cross_env_receipt.stable_hash,
        }
    )

    status = "EMPTY" if len(sorted_entries) == 0 else "VALID"

    return FailureLedgerReceipt(
        schema_version=SCHEMA_VERSION,
        module_version=MODULE_VERSION,
        ledger_status=status,
        input_hash=input_hash,
        ledger=ledger,
        failure_count=len(sorted_entries),
        typed_counts={key: typed_counts[key] for key in sorted(typed_counts)},
        suppression_rate=0,
    )


__all__ = [
    "FailureRecord",
    "FailureLedgerEntry",
    "FailureLedger",
    "FailureLedgerReceipt",
    "build_failure_ledger",
]
