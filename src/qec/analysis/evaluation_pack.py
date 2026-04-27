# SPDX-License-Identifier: MIT
"""v148.9 — Evaluation pack receipt builder for bundled analysis receipts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from qec.analysis.adversarial_determinism_battery import AdversarialDeterminismReceipt
from qec.analysis.canonical_hashing import canonical_bytes, canonical_json, sha256_hex
from qec.analysis.counterfactual_replay_kernel import CounterfactualReplayReceipt
from qec.analysis.cross_environment_replay_kernel import CrossEnvironmentReplayReceipt
from qec.analysis.failure_ledger import FailureLedgerReceipt
from qec.analysis.fix_proposal_kernel import FixProposalReceipt
from qec.analysis.fix_validation_kernel import FixValidationReceipt
from qec.analysis.governance_validation_kernel import GovernanceValidationReceipt
from qec.analysis.issue_normalization_kernel import IssueNormalizationReceipt
from qec.analysis.real_workload_injection import DeterministicWorkloadReceipt

SCHEMA_VERSION = "v148.9"
MODULE_VERSION = "v148.9"

ITEM_TYPE_GOVERNANCE_VALIDATION = "GOVERNANCE_VALIDATION"
ITEM_TYPE_ISSUE_NORMALIZATION = "ISSUE_NORMALIZATION"
ITEM_TYPE_FIX_PROPOSAL = "FIX_PROPOSAL"
ITEM_TYPE_FIX_VALIDATION = "FIX_VALIDATION"
ITEM_TYPE_COUNTERFACTUAL_REPLAY = "COUNTERFACTUAL_REPLAY"
ITEM_TYPE_ADVERSARIAL_DETERMINISM = "ADVERSARIAL_DETERMINISM"
ITEM_TYPE_CROSS_ENVIRONMENT_REPLAY = "CROSS_ENVIRONMENT_REPLAY"
ITEM_TYPE_FAILURE_LEDGER = "FAILURE_LEDGER"
ITEM_TYPE_DETERMINISTIC_WORKLOAD = "DETERMINISTIC_WORKLOAD"

ALLOWED_ITEM_TYPES: tuple[str, ...] = (
    ITEM_TYPE_ADVERSARIAL_DETERMINISM,
    ITEM_TYPE_COUNTERFACTUAL_REPLAY,
    ITEM_TYPE_CROSS_ENVIRONMENT_REPLAY,
    ITEM_TYPE_DETERMINISTIC_WORKLOAD,
    ITEM_TYPE_FAILURE_LEDGER,
    ITEM_TYPE_FIX_PROPOSAL,
    ITEM_TYPE_FIX_VALIDATION,
    ITEM_TYPE_GOVERNANCE_VALIDATION,
    ITEM_TYPE_ISSUE_NORMALIZATION,
)

_ALLOWED_PACK_STATUSES = frozenset({"COMPLETE", "PARTIAL", "HAS_FAILURES", "EMPTY", "INVALID_INPUT"})

_STATUS_FIELDS: tuple[str, ...] = (
    "validation_status",
    "normalization_status",
    "proposal_status",
    "replay_status",
    "battery_status",
    "receipt_status",
    "ledger_status",
    "workload_status",
)

_CLASS_TO_ITEM_TYPE = {
    GovernanceValidationReceipt: ITEM_TYPE_GOVERNANCE_VALIDATION,
    IssueNormalizationReceipt: ITEM_TYPE_ISSUE_NORMALIZATION,
    FixProposalReceipt: ITEM_TYPE_FIX_PROPOSAL,
    FixValidationReceipt: ITEM_TYPE_FIX_VALIDATION,
    CounterfactualReplayReceipt: ITEM_TYPE_COUNTERFACTUAL_REPLAY,
    AdversarialDeterminismReceipt: ITEM_TYPE_ADVERSARIAL_DETERMINISM,
    CrossEnvironmentReplayReceipt: ITEM_TYPE_CROSS_ENVIRONMENT_REPLAY,
    FailureLedgerReceipt: ITEM_TYPE_FAILURE_LEDGER,
    DeterministicWorkloadReceipt: ITEM_TYPE_DETERMINISTIC_WORKLOAD,
}

_CLASS_NAME_TO_ITEM_TYPE = {
    "GovernanceValidationReceipt": ITEM_TYPE_GOVERNANCE_VALIDATION,
    "IssueNormalizationReceipt": ITEM_TYPE_ISSUE_NORMALIZATION,
    "FixProposalReceipt": ITEM_TYPE_FIX_PROPOSAL,
    "FixValidationReceipt": ITEM_TYPE_FIX_VALIDATION,
    "CounterfactualReplayReceipt": ITEM_TYPE_COUNTERFACTUAL_REPLAY,
    "AdversarialDeterminismReceipt": ITEM_TYPE_ADVERSARIAL_DETERMINISM,
    "CrossEnvironmentReplayReceipt": ITEM_TYPE_CROSS_ENVIRONMENT_REPLAY,
    "FailureLedgerReceipt": ITEM_TYPE_FAILURE_LEDGER,
    "DeterministicWorkloadReceipt": ITEM_TYPE_DETERMINISTIC_WORKLOAD,
}



def _is_hex_hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)



def _extract_receipt_hash(receipt: Any) -> str:
    stable_hash_attr = getattr(receipt, "stable_hash", None)
    hash_value: Any
    if callable(stable_hash_attr):
        hash_value = stable_hash_attr()
    else:
        hash_value = stable_hash_attr
    if not _is_hex_hash(hash_value):
        raise ValueError(
            "receipt.stable_hash must be or return a valid 64-char lowercase SHA-256 hex string"
        )
    return hash_value



def _extract_payload_mapping(receipt: Any) -> Mapping[str, Any]:
    to_dict = getattr(receipt, "to_dict", None)
    if not callable(to_dict):
        raise ValueError("receipt must provide to_dict()")
    payload = to_dict()
    if not isinstance(payload, Mapping):
        raise ValueError("receipt.to_dict() must return a mapping")
    return payload



def _canonical_payload_hash(receipt: Any) -> str:
    payload = dict(_extract_payload_mapping(receipt))
    payload.pop("stable_hash", None)
    return sha256_hex(payload)



def _detect_item_type(receipt: Any) -> str:
    for cls, item_type in _CLASS_TO_ITEM_TYPE.items():
        if isinstance(receipt, cls):
            return item_type
    class_name = receipt.__class__.__name__
    if class_name in _CLASS_NAME_TO_ITEM_TYPE:
        return _CLASS_NAME_TO_ITEM_TYPE[class_name]
    raise ValueError(f"unsupported receipt type: {class_name}")



def _extract_status(receipt: Any) -> str:
    for field_name in _STATUS_FIELDS:
        value = getattr(receipt, field_name, None)
        if isinstance(value, str) and value:
            return value
    return "UNKNOWN"



def _failure_delta(receipt: Any) -> int:
    if isinstance(receipt, FailureLedgerReceipt):
        return int(receipt.failure_count)
    if isinstance(receipt, AdversarialDeterminismReceipt):
        return int(receipt.fail_count)
    if isinstance(receipt, CrossEnvironmentReplayReceipt):
        return 1 if bool(receipt.failure_recorded) else 0
    if isinstance(receipt, FixValidationReceipt):
        return int(receipt.invalid_count) + int(receipt.unsafe_count) + int(receipt.insufficient_count)
    if isinstance(receipt, CounterfactualReplayReceipt):
        return int(receipt.dominated_count) + int(receipt.unresolved_count)
    return 0



def _determinism_failed(receipt: Any) -> bool:
    if isinstance(receipt, AdversarialDeterminismReceipt):
        return (not bool(receipt.determinism_pass)) or (not bool(receipt.hash_stability_pass)) or int(receipt.fail_count) > 0
    if isinstance(receipt, CrossEnvironmentReplayReceipt):
        return not bool(receipt.determinism_preserved)
    return False


@dataclass(frozen=True)
class EvaluationPackItem:
    item_id: str
    item_type: str
    receipt_hash: str
    canonical_payload_hash: str
    status: str

    def __post_init__(self) -> None:
        if self.item_type not in ALLOWED_ITEM_TYPES:
            raise ValueError("invalid item_type")
        if not self.item_id:
            raise ValueError("item_id must be non-empty")
        if not _is_hex_hash(self.receipt_hash):
            raise ValueError("receipt_hash must be SHA-256 hex")
        if not _is_hex_hash(self.canonical_payload_hash):
            raise ValueError("canonical_payload_hash must be SHA-256 hex")
        if not isinstance(self.status, str) or not self.status:
            raise ValueError("status must be a non-empty string")

    def _payload(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "item_type": self.item_type,
            "receipt_hash": self.receipt_hash,
            "canonical_payload_hash": self.canonical_payload_hash,
            "status": self.status,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "stable_hash": self.stable_hash()}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self._payload())


@dataclass(frozen=True)
class EvaluationPackSummary:
    item_count: int
    type_counts: Mapping[str, int]
    status_counts: Mapping[str, int]
    failure_count: int
    determinism_preserved: bool
    bundle_complete: bool

    def __post_init__(self) -> None:
        if self.item_count < 0:
            raise ValueError("item_count must be non-negative")
        normalized_type_counts = {item_type: int(self.type_counts.get(item_type, 0)) for item_type in ALLOWED_ITEM_TYPES}
        object.__setattr__(self, "type_counts", normalized_type_counts)

        ordered_status_counts = {key: int(self.status_counts[key]) for key in sorted(self.status_counts)}
        object.__setattr__(self, "status_counts", ordered_status_counts)

        if self.failure_count < 0:
            raise ValueError("failure_count must be non-negative")

    def _payload(self) -> dict[str, Any]:
        return {
            "item_count": self.item_count,
            "type_counts": {key: self.type_counts[key] for key in ALLOWED_ITEM_TYPES},
            "status_counts": {key: self.status_counts[key] for key in sorted(self.status_counts)},
            "failure_count": self.failure_count,
            "determinism_preserved": self.determinism_preserved,
            "bundle_complete": self.bundle_complete,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "stable_hash": self.stable_hash()}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self._payload())


@dataclass(frozen=True)
class EvaluationPackReceipt:
    schema_version: str
    module_version: str
    pack_status: str
    pack_id: str
    items: tuple[EvaluationPackItem, ...]
    summary: EvaluationPackSummary

    def __post_init__(self) -> None:
        if self.pack_status not in _ALLOWED_PACK_STATUSES:
            raise ValueError("invalid pack_status")
        if not _is_hex_hash(self.pack_id):
            raise ValueError("pack_id must be SHA-256 hex")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "module_version": self.module_version,
            "pack_status": self.pack_status,
            "pack_id": self.pack_id,
            "items": [item.to_dict() for item in self.items],
            "summary": self.summary.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "stable_hash": self.stable_hash()}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return sha256_hex(self._payload())



def _pack_status(summary: EvaluationPackSummary) -> str:
    if summary.item_count == 0:
        return "EMPTY"
    if summary.failure_count > 0:
        return "HAS_FAILURES"
    if summary.bundle_complete:
        return "COMPLETE"
    return "PARTIAL"



def build_evaluation_pack(receipts: Sequence[Any]) -> EvaluationPackReceipt:
    if not isinstance(receipts, Sequence):
        raise ValueError("receipts must be a sequence")

    normalized = tuple(receipts)
    items: list[EvaluationPackItem] = []
    seen_hashes: set[str] = set()
    receipt_types_seen: set[str] = set()
    status_counter: Counter[str] = Counter()
    failure_count = 0
    determinism_preserved = True

    for receipt in normalized:
        item_type = _detect_item_type(receipt)
        receipt_hash = _extract_receipt_hash(receipt)
        if receipt_hash in seen_hashes:
            raise ValueError("duplicate receipt hash detected")
        seen_hashes.add(receipt_hash)

        status = _extract_status(receipt)
        item_id = f"{item_type}-{receipt_hash[:16]}"
        item = EvaluationPackItem(
            item_id=item_id,
            item_type=item_type,
            receipt_hash=receipt_hash,
            canonical_payload_hash=_canonical_payload_hash(receipt),
            status=status,
        )
        items.append(item)

        receipt_types_seen.add(item_type)
        status_counter[status] += 1
        failure_count += _failure_delta(receipt)
        if _determinism_failed(receipt):
            determinism_preserved = False

    sorted_items = tuple(
        sorted(
            items,
            key=lambda item: (item.item_type, item.status, item.receipt_hash, item.item_id),
        )
    )

    type_counter = Counter(item.item_type for item in sorted_items)
    type_counts = {item_type: int(type_counter[item_type]) for item_type in ALLOWED_ITEM_TYPES}
    status_counts = {key: int(status_counter[key]) for key in sorted(status_counter)}

    summary = EvaluationPackSummary(
        item_count=len(sorted_items),
        type_counts=type_counts,
        status_counts=status_counts,
        failure_count=failure_count,
        determinism_preserved=determinism_preserved,
        bundle_complete=all(item_type in receipt_types_seen for item_type in ALLOWED_ITEM_TYPES),
    )

    pack_id = sha256_hex(
        {
            "items": [item.stable_hash() for item in sorted_items],
            "summary_hash": summary.stable_hash(),
            "schema_version": SCHEMA_VERSION,
            "module_version": MODULE_VERSION,
        }
    )

    return EvaluationPackReceipt(
        schema_version=SCHEMA_VERSION,
        module_version=MODULE_VERSION,
        pack_status=_pack_status(summary),
        pack_id=pack_id,
        items=sorted_items,
        summary=summary,
    )


__all__ = [
    "ALLOWED_ITEM_TYPES",
    "EvaluationPackItem",
    "EvaluationPackSummary",
    "EvaluationPackReceipt",
    "build_evaluation_pack",
]
