from __future__ import annotations

from dataclasses import dataclass
from re import compile as re_compile

from qec.analysis.agent_governance_fence import GovernanceDecision
from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.res_rag_semantic_field import SemanticFieldReceipt

_SHA256_HEX_RE = re_compile(r"^[0-9a-f]{64}$")


def _is_sha256_hex(value: str) -> bool:
    return isinstance(value, str) and _SHA256_HEX_RE.fullmatch(value) is not None


def _governance_entry_payload(governance_decision_hash: str, occurrence_count: int, source_rule_id_sets: tuple[tuple[str, ...], ...]) -> dict[str, object]:
    return {
        "governance_decision_hash": governance_decision_hash,
        "occurrence_count": occurrence_count,
        "source_rule_id_sets": source_rule_id_sets,
    }


def _semantic_entry_payload(semantic_field_hash: str, occurrence_count: int, source_document_ids: tuple[str, ...]) -> dict[str, object]:
    return {
        "semantic_field_hash": semantic_field_hash,
        "occurrence_count": occurrence_count,
        "source_document_ids": source_document_ids,
    }


def _governance_receipt_entry_payload(entry: "GovernanceCompressionEntry") -> dict[str, object]:
    return {
        "governance_decision_hash": entry.governance_decision_hash,
        "occurrence_count": entry.occurrence_count,
        "source_rule_id_sets": entry.source_rule_id_sets,
        "compression_entry_hash": entry.compression_entry_hash,
    }


def _semantic_receipt_entry_payload(entry: "SemanticCompressionEntry") -> dict[str, object]:
    return {
        "semantic_field_hash": entry.semantic_field_hash,
        "occurrence_count": entry.occurrence_count,
        "source_document_ids": entry.source_document_ids,
        "compression_entry_hash": entry.compression_entry_hash,
    }


def _validate_governance_entry(entry: "GovernanceCompressionEntry") -> None:
    if not _is_sha256_hex(entry.governance_decision_hash):
        raise ValueError("INVALID_INPUT")
    if not isinstance(entry.occurrence_count, int) or isinstance(entry.occurrence_count, bool):
        raise ValueError("INVALID_INPUT")
    if entry.occurrence_count < 2:
        raise ValueError("INSUFFICIENT_OCCURRENCES_FOR_COMPRESSION")
    if not isinstance(entry.source_rule_id_sets, tuple) or any(not isinstance(s, tuple) for s in entry.source_rule_id_sets):
        raise ValueError("INVALID_INPUT")
    if len(entry.source_rule_id_sets) != entry.occurrence_count:
        raise ValueError("INVALID_INPUT")
    if any(any(not isinstance(v, str) for v in s) for s in entry.source_rule_id_sets):
        raise ValueError("INVALID_INPUT")
    if any(tuple(sorted(s)) != s for s in entry.source_rule_id_sets):
        raise ValueError("INVALID_INPUT")
    if tuple(sorted(entry.source_rule_id_sets)) != entry.source_rule_id_sets:
        raise ValueError("INVALID_INPUT")
    if not _is_sha256_hex(entry.compression_entry_hash):
        raise ValueError("INVALID_INPUT")
    expected = sha256_hex(_governance_entry_payload(entry.governance_decision_hash, entry.occurrence_count, entry.source_rule_id_sets))
    if entry.compression_entry_hash != expected:
        raise ValueError("HASH_MISMATCH")


def _validate_governance_receipt(receipt: "GovernanceCompressionReceipt") -> None:
    if not isinstance(receipt.entries, tuple) or any(not isinstance(e, GovernanceCompressionEntry) for e in receipt.entries):
        raise ValueError("INVALID_INPUT")
    if tuple(sorted(receipt.entries, key=lambda e: e.governance_decision_hash)) != receipt.entries:
        raise ValueError("INVALID_INPUT")
    if len({e.governance_decision_hash for e in receipt.entries}) != len(receipt.entries):
        raise ValueError("IDENTITY_COLLISION")
    if not isinstance(receipt.total_compressed_decisions, int) or isinstance(receipt.total_compressed_decisions, bool):
        raise ValueError("INVALID_INPUT")
    if receipt.total_compressed_decisions != sum(e.occurrence_count for e in receipt.entries):
        raise ValueError("INVALID_INPUT")
    if not _is_sha256_hex(receipt.governance_compression_receipt_hash):
        raise ValueError("INVALID_INPUT")
    expected = sha256_hex({"entries": [
        _governance_receipt_entry_payload(e) for e in receipt.entries
    ], "total_compressed_decisions": receipt.total_compressed_decisions})
    if receipt.governance_compression_receipt_hash != expected:
        raise ValueError("HASH_MISMATCH")


def _validate_semantic_entry(entry: "SemanticCompressionEntry") -> None:
    if not _is_sha256_hex(entry.semantic_field_hash):
        raise ValueError("INVALID_INPUT")
    if not isinstance(entry.occurrence_count, int) or isinstance(entry.occurrence_count, bool):
        raise ValueError("INVALID_INPUT")
    if entry.occurrence_count < 2:
        raise ValueError("INSUFFICIENT_OCCURRENCES_FOR_COMPRESSION")
    if not isinstance(entry.source_document_ids, tuple) or any(not isinstance(d, str) for d in entry.source_document_ids):
        raise ValueError("INVALID_INPUT")
    if len(entry.source_document_ids) != entry.occurrence_count or tuple(sorted(entry.source_document_ids)) != entry.source_document_ids:
        raise ValueError("INVALID_INPUT")
    if not _is_sha256_hex(entry.compression_entry_hash):
        raise ValueError("INVALID_INPUT")
    expected = sha256_hex(_semantic_entry_payload(entry.semantic_field_hash, entry.occurrence_count, entry.source_document_ids))
    if entry.compression_entry_hash != expected:
        raise ValueError("HASH_MISMATCH")


def _validate_semantic_receipt(receipt: "SemanticCompressionReceipt") -> None:
    if not isinstance(receipt.entries, tuple) or any(not isinstance(e, SemanticCompressionEntry) for e in receipt.entries):
        raise ValueError("INVALID_INPUT")
    if tuple(sorted(receipt.entries, key=lambda e: e.semantic_field_hash)) != receipt.entries:
        raise ValueError("INVALID_INPUT")
    if len({e.semantic_field_hash for e in receipt.entries}) != len(receipt.entries):
        raise ValueError("IDENTITY_COLLISION")
    if not isinstance(receipt.total_compressed_fields, int) or isinstance(receipt.total_compressed_fields, bool):
        raise ValueError("INVALID_INPUT")
    if receipt.total_compressed_fields != sum(e.occurrence_count for e in receipt.entries):
        raise ValueError("INVALID_INPUT")
    if not _is_sha256_hex(receipt.semantic_compression_receipt_hash):
        raise ValueError("INVALID_INPUT")
    expected = sha256_hex({"entries": [
        _semantic_receipt_entry_payload(e) for e in receipt.entries
    ], "total_compressed_fields": receipt.total_compressed_fields})
    if receipt.semantic_compression_receipt_hash != expected:
        raise ValueError("HASH_MISMATCH")


@dataclass(frozen=True)
class GovernanceCompressionEntry:
    governance_decision_hash: str
    occurrence_count: int
    source_rule_id_sets: tuple[tuple[str, ...], ...]
    compression_entry_hash: str

    def __post_init__(self) -> None:
        _validate_governance_entry(self)


@dataclass(frozen=True)
class GovernanceCompressionReceipt:
    entries: tuple[GovernanceCompressionEntry, ...]
    total_compressed_decisions: int
    governance_compression_receipt_hash: str

    def __post_init__(self) -> None:
        _validate_governance_receipt(self)


@dataclass(frozen=True)
class SemanticCompressionEntry:
    semantic_field_hash: str
    occurrence_count: int
    source_document_ids: tuple[str, ...]
    compression_entry_hash: str

    def __post_init__(self) -> None:
        _validate_semantic_entry(self)


@dataclass(frozen=True)
class SemanticCompressionReceipt:
    entries: tuple[SemanticCompressionEntry, ...]
    total_compressed_fields: int
    semantic_compression_receipt_hash: str

    def __post_init__(self) -> None:
        _validate_semantic_receipt(self)


def build_governance_compression_receipt(decisions: list[GovernanceDecision]) -> GovernanceCompressionReceipt:
    if not isinstance(decisions, list) or any(not isinstance(d, GovernanceDecision) for d in decisions):
        raise ValueError("INVALID_INPUT")
    groups: dict[str, list[tuple[str, ...]]] = {}
    for decision in decisions:
        source_ids = tuple(sorted(decision.matched_rule_ids))
        groups.setdefault(decision.decision_hash, []).append(source_ids)
    entries: list[GovernanceCompressionEntry] = []
    for decision_hash, sources in groups.items():
        if len(sources) < 2:
            continue
        source_sets = tuple(sorted(sources))
        occ = len(source_sets)
        eh = sha256_hex(_governance_entry_payload(decision_hash, occ, source_sets))
        entries.append(GovernanceCompressionEntry(decision_hash, occ, source_sets, eh))
    ordered = tuple(sorted(entries, key=lambda e: e.governance_decision_hash))
    if not ordered:
        raise ValueError("INSUFFICIENT_OCCURRENCES_FOR_COMPRESSION")
    total = sum(e.occurrence_count for e in ordered)
    rh = sha256_hex({"entries": [_governance_receipt_entry_payload(e) for e in ordered], "total_compressed_decisions": total})
    return GovernanceCompressionReceipt(ordered, total, rh)


def build_semantic_compression_receipt(semantic_field_receipts: list[SemanticFieldReceipt]) -> SemanticCompressionReceipt:
    if not isinstance(semantic_field_receipts, list) or any(not isinstance(s, SemanticFieldReceipt) for s in semantic_field_receipts):
        raise ValueError("INVALID_INPUT")
    groups: dict[str, list[str]] = {}
    for receipt in semantic_field_receipts:
        groups.setdefault(receipt.semantic_field_hash, []).append(receipt.canonical_hash)
    entries: list[SemanticCompressionEntry] = []
    for semantic_hash, doc_ids in groups.items():
        if len(doc_ids) < 2:
            continue
        source_document_ids = tuple(sorted(doc_ids))
        occ = len(source_document_ids)
        eh = sha256_hex(_semantic_entry_payload(semantic_hash, occ, source_document_ids))
        entries.append(SemanticCompressionEntry(semantic_hash, occ, source_document_ids, eh))
    ordered = tuple(sorted(entries, key=lambda e: e.semantic_field_hash))
    if not ordered:
        raise ValueError("INSUFFICIENT_OCCURRENCES_FOR_COMPRESSION")
    total = sum(e.occurrence_count for e in ordered)
    rh = sha256_hex({"entries": [_semantic_receipt_entry_payload(e) for e in ordered], "total_compressed_fields": total})
    return SemanticCompressionReceipt(ordered, total, rh)


def validate_governance_compression_receipt(r: GovernanceCompressionReceipt) -> bool:
    if not isinstance(r, GovernanceCompressionReceipt):
        raise ValueError("INVALID_INPUT")
    for entry in r.entries:
        _validate_governance_entry(entry)
    _validate_governance_receipt(r)
    return True


def validate_semantic_compression_receipt(r: SemanticCompressionReceipt) -> bool:
    if not isinstance(r, SemanticCompressionReceipt):
        raise ValueError("INVALID_INPUT")
    for entry in r.entries:
        _validate_semantic_entry(entry)
    _validate_semantic_receipt(r)
    return True
