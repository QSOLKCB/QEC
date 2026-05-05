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


@dataclass(frozen=True)
class GovernanceCompressionEntry:
    governance_decision_hash: str
    occurrence_count: int
    source_rule_id_sets: tuple[tuple[str, ...], ...]
    compression_entry_hash: str

    def __post_init__(self) -> None:
        if not _is_sha256_hex(self.governance_decision_hash):
            raise ValueError("INVALID_INPUT")
        if not isinstance(self.occurrence_count, int) or isinstance(self.occurrence_count, bool):
            raise ValueError("INVALID_INPUT")
        if self.occurrence_count < 2:
            raise ValueError("INSUFFICIENT_OCCURRENCES_FOR_COMPRESSION")
        if not isinstance(self.source_rule_id_sets, tuple) or any(not isinstance(s, tuple) for s in self.source_rule_id_sets):
            raise ValueError("INVALID_INPUT")
        if len(self.source_rule_id_sets) != self.occurrence_count:
            raise ValueError("INVALID_INPUT")
        if any(tuple(sorted(s)) != s or any(not isinstance(v, str) for v in s) for s in self.source_rule_id_sets):
            raise ValueError("INVALID_INPUT")
        if tuple(sorted(self.source_rule_id_sets)) != self.source_rule_id_sets:
            raise ValueError("INVALID_INPUT")
        if not _is_sha256_hex(self.compression_entry_hash):
            raise ValueError("INVALID_INPUT")
        expected = sha256_hex(_governance_entry_payload(self.governance_decision_hash, self.occurrence_count, self.source_rule_id_sets))
        if self.compression_entry_hash != expected:
            raise ValueError("HASH_MISMATCH")


@dataclass(frozen=True)
class GovernanceCompressionReceipt:
    entries: tuple[GovernanceCompressionEntry, ...]
    total_compressed_decisions: int
    governance_compression_receipt_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.entries, tuple) or any(not isinstance(e, GovernanceCompressionEntry) for e in self.entries):
            raise ValueError("INVALID_INPUT")
        if tuple(sorted(self.entries, key=lambda e: e.governance_decision_hash)) != self.entries:
            raise ValueError("INVALID_INPUT")
        if len({e.governance_decision_hash for e in self.entries}) != len(self.entries):
            raise ValueError("IDENTITY_COLLISION")
        if not isinstance(self.total_compressed_decisions, int) or isinstance(self.total_compressed_decisions, bool):
            raise ValueError("INVALID_INPUT")
        if self.total_compressed_decisions != sum(e.occurrence_count for e in self.entries):
            raise ValueError("INVALID_INPUT")
        expected = sha256_hex({"entries": [
            _governance_receipt_entry_payload(e) for e in self.entries
        ], "total_compressed_decisions": self.total_compressed_decisions})
        if self.governance_compression_receipt_hash != expected:
            raise ValueError("HASH_MISMATCH")


@dataclass(frozen=True)
class SemanticCompressionEntry:
    semantic_field_hash: str
    occurrence_count: int
    source_document_ids: tuple[str, ...]
    compression_entry_hash: str

    def __post_init__(self) -> None:
        if not _is_sha256_hex(self.semantic_field_hash):
            raise ValueError("INVALID_INPUT")
        if not isinstance(self.occurrence_count, int) or isinstance(self.occurrence_count, bool):
            raise ValueError("INVALID_INPUT")
        if self.occurrence_count < 2:
            raise ValueError("INSUFFICIENT_OCCURRENCES_FOR_COMPRESSION")
        if not isinstance(self.source_document_ids, tuple) or any(not isinstance(d, str) for d in self.source_document_ids):
            raise ValueError("INVALID_INPUT")
        if len(self.source_document_ids) != self.occurrence_count or tuple(sorted(self.source_document_ids)) != self.source_document_ids:
            raise ValueError("INVALID_INPUT")
        if not _is_sha256_hex(self.compression_entry_hash):
            raise ValueError("INVALID_INPUT")
        expected = sha256_hex(_semantic_entry_payload(self.semantic_field_hash, self.occurrence_count, self.source_document_ids))
        if self.compression_entry_hash != expected:
            raise ValueError("HASH_MISMATCH")


@dataclass(frozen=True)
class SemanticCompressionReceipt:
    entries: tuple[SemanticCompressionEntry, ...]
    total_compressed_fields: int
    semantic_compression_receipt_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.entries, tuple) or any(not isinstance(e, SemanticCompressionEntry) for e in self.entries):
            raise ValueError("INVALID_INPUT")
        if tuple(sorted(self.entries, key=lambda e: e.semantic_field_hash)) != self.entries:
            raise ValueError("INVALID_INPUT")
        if len({e.semantic_field_hash for e in self.entries}) != len(self.entries):
            raise ValueError("IDENTITY_COLLISION")
        if not isinstance(self.total_compressed_fields, int) or isinstance(self.total_compressed_fields, bool):
            raise ValueError("INVALID_INPUT")
        if self.total_compressed_fields != sum(e.occurrence_count for e in self.entries):
            raise ValueError("INVALID_INPUT")
        expected = sha256_hex({"entries": [
            _semantic_receipt_entry_payload(e) for e in self.entries
        ], "total_compressed_fields": self.total_compressed_fields})
        if self.semantic_compression_receipt_hash != expected:
            raise ValueError("HASH_MISMATCH")


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
    for e in r.entries:
        GovernanceCompressionEntry(e.governance_decision_hash, e.occurrence_count, e.source_rule_id_sets, e.compression_entry_hash)
    GovernanceCompressionReceipt(r.entries, r.total_compressed_decisions, r.governance_compression_receipt_hash)
    return True


def validate_semantic_compression_receipt(r: SemanticCompressionReceipt) -> bool:
    if not isinstance(r, SemanticCompressionReceipt):
        raise ValueError("INVALID_INPUT")
    for e in r.entries:
        SemanticCompressionEntry(e.semantic_field_hash, e.occurrence_count, e.source_document_ids, e.compression_entry_hash)
    SemanticCompressionReceipt(r.entries, r.total_compressed_fields, r.semantic_compression_receipt_hash)
    return True
