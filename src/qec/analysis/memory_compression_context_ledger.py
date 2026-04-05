"""Deterministic Layer-4 memory compression + context ledger (v137.1.13).

LAWS preserved by this module:
- CANONICAL_MEMORY_COMPACTION_LAW
- STABLE_CONTEXT_HASH_CHAIN
- REPLAY_SAFE_CONTEXT_SNAPSHOT_INVARIANT
- DETERMINISTIC_LEDGER_MINIMIZATION_LAW
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

MEMORY_COMPRESSION_CONTEXT_LEDGER_VERSION: str = "v137.1.13"
ROUND_DIGITS: int = 12
GENESIS_HASH: str = "0" * 64

CANONICAL_MEMORY_COMPACTION_LAW: str = "CANONICAL_MEMORY_COMPACTION_LAW"
STABLE_CONTEXT_HASH_CHAIN: str = "STABLE_CONTEXT_HASH_CHAIN"
REPLAY_SAFE_CONTEXT_SNAPSHOT_INVARIANT: str = "REPLAY_SAFE_CONTEXT_SNAPSHOT_INVARIANT"
DETERMINISTIC_LEDGER_MINIMIZATION_LAW: str = "DETERMINISTIC_LEDGER_MINIMIZATION_LAW"


@dataclass(frozen=True)
class ContextItem:
    item_id: str
    category: str
    content: str
    priority: int
    source_ref: str
    tags: tuple[str, ...]
    bounded: bool
    item_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "category": self.category,
            "content": self.content,
            "priority": self.priority,
            "source_ref": self.source_ref,
            "tags": list(self.tags),
            "bounded": self.bounded,
            "item_hash": self.item_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class CanonicalMemorySnapshot:
    snapshot_id: str
    retained_item_ids: tuple[str, ...]
    compacted_item_ids: tuple[str, ...]
    deduplicated_count: int
    retention_ratio: float
    snapshot_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "retained_item_ids": list(self.retained_item_ids),
            "compacted_item_ids": list(self.compacted_item_ids),
            "deduplicated_count": self.deduplicated_count,
            "retention_ratio": _round_float(self.retention_ratio),
            "snapshot_hash": self.snapshot_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class ContextCompressionReport:
    original_count: int
    retained_count: int
    compacted_count: int
    compression_ratio: float
    deterministic: bool
    report_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_count": self.original_count,
            "retained_count": self.retained_count,
            "compacted_count": self.compacted_count,
            "compression_ratio": _round_float(self.compression_ratio),
            "deterministic": self.deterministic,
            "report_hash": self.report_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class ContextLedgerEntry:
    sequence_id: int
    snapshot_hash: str
    parent_hash: str
    retained_count: int
    compression_ratio: float
    entry_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "snapshot_hash": self.snapshot_hash,
            "parent_hash": self.parent_hash,
            "retained_count": self.retained_count,
            "compression_ratio": _round_float(self.compression_ratio),
            "entry_hash": self.entry_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class ContextLedger:
    entries: tuple[ContextLedgerEntry, ...]
    head_hash: str
    chain_valid: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "head_hash": self.head_hash,
            "chain_valid": self.chain_valid,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class ContextTransitionReport:
    total_items: int
    unique_items: int
    duplicate_items: int
    minimized: bool
    report_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_items": self.total_items,
            "unique_items": self.unique_items,
            "duplicate_items": self.duplicate_items,
            "minimized": self.minimized,
            "report_hash": self.report_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class ContextCompactionPlan:
    retained_ids: tuple[str, ...]
    removed_ids: tuple[str, ...]
    canonical_order: tuple[str, ...]
    plan_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "retained_ids": list(self.retained_ids),
            "removed_ids": list(self.removed_ids),
            "canonical_order": list(self.canonical_order),
            "plan_hash": self.plan_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _normalize_string(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be a non-empty string")
    return normalized


def _round_float(value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("float value must be finite")
    return round(numeric, ROUND_DIGITS)


def _bounded_ratio(numerator: int, denominator: int) -> float:
    ratio = numerator / max(denominator, 1)
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError("ratio must be bounded in [0,1]")
    return _round_float(ratio)


def _context_item_payload(item_id: str, category: str, content: str, priority: int, source_ref: str, tags: tuple[str, ...], bounded: bool) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "category": category,
        "content": content,
        "priority": priority,
        "source_ref": source_ref,
        "tags": tuple(tags),
        "bounded": bool(bounded),
    }


def _normalize_tags(tags: Any) -> tuple[str, ...]:
    if tags is None:
        return ()
    if isinstance(tags, str):
        raise ValueError("tags must be an iterable of strings")
    normalized: set[str] = set()
    for value in tags:
        if not isinstance(value, str):
            raise ValueError("tag must be a string")
        tag = value.strip()
        if not tag:
            raise ValueError("tag must be non-empty after trimming")
        normalized.add(tag)
    return tuple(sorted(normalized))


def _normalize_item_like(item: Any) -> dict[str, Any]:
    if isinstance(item, ContextItem):
        return {
            "item_id": item.item_id,
            "category": item.category,
            "content": item.content,
            "priority": item.priority,
            "source_ref": item.source_ref,
            "tags": item.tags,
            "bounded": item.bounded,
        }
    if isinstance(item, Mapping):
        return {
            "item_id": item.get("item_id"),
            "category": item.get("category"),
            "content": item.get("content"),
            "priority": item.get("priority"),
            "source_ref": item.get("source_ref"),
            "tags": item.get("tags", ()),
            "bounded": item.get("bounded", True),
        }
    if isinstance(item, tuple):
        if len(item) != 7:
            raise ValueError("tuple context item must have exactly 7 fields")
        return {
            "item_id": item[0],
            "category": item[1],
            "content": item[2],
            "priority": item[3],
            "source_ref": item[4],
            "tags": item[5],
            "bounded": item[6],
        }
    raise ValueError("context item must be a mapping, tuple, or ContextItem")


def _entry_payload(sequence_id: int, snapshot_hash: str, parent_hash: str, retained_count: int, compression_ratio: float) -> dict[str, Any]:
    return {
        "sequence_id": sequence_id,
        "snapshot_hash": snapshot_hash,
        "parent_hash": parent_hash,
        "retained_count": retained_count,
        "compression_ratio": _round_float(compression_ratio),
    }


def empty_context_ledger() -> ContextLedger:
    return ContextLedger(entries=(), head_hash=GENESIS_HASH, chain_valid=True)


def normalize_context_items(items: Iterable[Any]) -> tuple[ContextItem, ...]:
    """Normalize context into canonical deterministic items.

    CANONICAL_MEMORY_COMPACTION_LAW: context compaction starts from explicit,
    deterministic normalization only.
    """
    normalized_items: list[ContextItem] = []
    seen_ids: set[str] = set()

    for item in items:
        raw = _normalize_item_like(item)
        item_id = _normalize_string("item_id", raw["item_id"])
        category = _normalize_string("category", raw["category"])
        content = _normalize_string("content", raw["content"])
        source_ref = _normalize_string("source_ref", raw["source_ref"])
        priority = raw["priority"]
        if not isinstance(priority, int) or isinstance(priority, bool):
            raise ValueError("priority must be an integer and not bool")
        tags = _normalize_tags(raw["tags"])
        bounded = bool(raw["bounded"])
        if item_id in seen_ids:
            raise ValueError(f"duplicate item_id: {item_id}")
        seen_ids.add(item_id)

        payload = _context_item_payload(item_id, category, content, priority, source_ref, tags, bounded)
        normalized_items.append(ContextItem(**payload, item_hash=_hash_sha256(payload)))

    normalized_items.sort(key=lambda x: (-x.priority, x.category, x.item_id))
    return tuple(normalized_items)


def deduplicate_context_items(items: Iterable[ContextItem]) -> tuple[tuple[ContextItem, ...], tuple[str, ...]]:
    """Deterministically collapse equivalent items using explicit fields only."""
    grouped: dict[tuple[str, str, str, tuple[str, ...]], list[ContextItem]] = {}
    for item in items:
        key = (item.content, item.category, item.source_ref, item.tags)
        grouped.setdefault(key, []).append(item)

    retained: list[ContextItem] = []
    compacted: list[str] = []

    for _, group in sorted(grouped.items(), key=lambda x: x[0]):
        ranked = sorted(group, key=lambda i: (-i.priority, i.item_id))
        retained.append(ranked[0])
        compacted.extend(i.item_id for i in ranked[1:])

    retained_sorted = tuple(sorted(retained, key=lambda x: (-x.priority, x.category, x.item_id)))
    compacted_sorted = tuple(sorted(compacted))
    return retained_sorted, compacted_sorted


def build_context_compaction_plan(retained_items: Iterable[ContextItem], compacted_item_ids: Iterable[str]) -> ContextCompactionPlan:
    retained_ids = tuple(sorted(item.item_id for item in retained_items))
    removed_ids = tuple(sorted(compacted_item_ids))
    canonical_order = tuple(sorted(retained_ids + removed_ids))
    payload = {
        "retained_ids": list(retained_ids),
        "removed_ids": list(removed_ids),
        "canonical_order": list(canonical_order),
    }
    return ContextCompactionPlan(
        retained_ids=retained_ids,
        removed_ids=removed_ids,
        canonical_order=canonical_order,
        plan_hash=_hash_sha256(payload),
    )


def build_canonical_memory_snapshot(retained_items: Iterable[ContextItem], compacted_item_ids: Iterable[str], original_count: int) -> CanonicalMemorySnapshot:
    retained_ids = tuple(sorted(item.item_id for item in retained_items))
    compacted_ids = tuple(sorted(compacted_item_ids))
    deduplicated_count = len(compacted_ids)
    retention_ratio = _bounded_ratio(len(retained_ids), original_count)
    payload = {
        "retained_item_ids": list(retained_ids),
        "compacted_item_ids": list(compacted_ids),
        "deduplicated_count": deduplicated_count,
        "retention_ratio": retention_ratio,
    }
    snapshot_hash = _hash_sha256(payload)
    return CanonicalMemorySnapshot(
        snapshot_id=f"snapshot_{snapshot_hash[:16]}",
        retained_item_ids=retained_ids,
        compacted_item_ids=compacted_ids,
        deduplicated_count=deduplicated_count,
        retention_ratio=retention_ratio,
        snapshot_hash=snapshot_hash,
    )


def compute_context_compression_report(original_count: int, retained_count: int, compacted_count: int) -> ContextCompressionReport:
    if min(original_count, retained_count, compacted_count) < 0:
        raise ValueError("counts must be non-negative")
    if retained_count > original_count:
        raise ValueError("retained_count cannot exceed original_count")
    if compacted_count > original_count:
        raise ValueError("compacted_count cannot exceed original_count")
    if retained_count + compacted_count != original_count:
        raise ValueError("retained_count + compacted_count must equal original_count")

    compression_ratio = _bounded_ratio(compacted_count, original_count)
    payload = {
        "original_count": original_count,
        "retained_count": retained_count,
        "compacted_count": compacted_count,
        "compression_ratio": compression_ratio,
        "deterministic": True,
    }
    return ContextCompressionReport(
        original_count=original_count,
        retained_count=retained_count,
        compacted_count=compacted_count,
        compression_ratio=compression_ratio,
        deterministic=True,
        report_hash=_hash_sha256(payload),
    )


def validate_context_ledger(ledger: ContextLedger) -> bool:
    """Pure bool validator for STABLE_CONTEXT_HASH_CHAIN integrity."""
    expected_parent = GENESIS_HASH
    expected_head = GENESIS_HASH

    for idx, entry in enumerate(ledger.entries):
        if entry.sequence_id != idx:
            return False
        if entry.parent_hash != expected_parent:
            return False
        payload = _entry_payload(
            sequence_id=entry.sequence_id,
            snapshot_hash=entry.snapshot_hash,
            parent_hash=entry.parent_hash,
            retained_count=entry.retained_count,
            compression_ratio=entry.compression_ratio,
        )
        if entry.entry_hash != _hash_sha256(payload):
            return False
        expected_head = _hash_sha256({"parent": expected_head, "entry": entry.entry_hash})
        expected_parent = entry.entry_hash

    computed_valid = ledger.head_hash == expected_head
    return computed_valid and ledger.chain_valid == computed_valid


def append_context_ledger_entry(
    prior_ledger: ContextLedger,
    snapshot_hash: str,
    retained_count: int,
    compression_ratio: float,
) -> ContextLedger:
    if not validate_context_ledger(prior_ledger):
        raise ValueError("prior ledger is malformed or invalid")
    if retained_count < 0:
        raise ValueError("retained_count must be non-negative")
    ratio = _round_float(compression_ratio)
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError("compression_ratio must be bounded in [0,1]")

    sequence_id = len(prior_ledger.entries)
    parent_hash = prior_ledger.entries[-1].entry_hash if prior_ledger.entries else GENESIS_HASH
    payload = _entry_payload(sequence_id, snapshot_hash, parent_hash, retained_count, ratio)
    entry_hash = _hash_sha256(payload)
    new_entry = ContextLedgerEntry(
        sequence_id=sequence_id,
        snapshot_hash=snapshot_hash,
        parent_hash=parent_hash,
        retained_count=retained_count,
        compression_ratio=ratio,
        entry_hash=entry_hash,
    )
    new_entries = prior_ledger.entries + (new_entry,)
    new_head = _hash_sha256({"parent": prior_ledger.head_hash, "entry": entry_hash})
    ledger = ContextLedger(entries=new_entries, head_hash=new_head, chain_valid=True)
    if not validate_context_ledger(ledger):
        raise ValueError("failed to append a valid context ledger entry")
    return ledger


def minimize_context_ledger(ledger: ContextLedger, threshold: int = 8) -> ContextLedger:
    """Deterministic minimized ledger rebuild.

    DETERMINISTIC_LEDGER_MINIMIZATION_LAW rule:
    - length <= threshold: return unchanged
    - otherwise: deterministically fold prefix into a canonical base entry and
      keep the latest tail entries, then rebuild sequence IDs from zero.
    """
    if threshold < 1:
        raise ValueError("threshold must be >= 1")
    if not validate_context_ledger(ledger):
        raise ValueError("ledger must be valid before minimization")
    if len(ledger.entries) <= threshold:
        return ledger

    tail_keep = threshold - 1
    prefix = ledger.entries[: len(ledger.entries) - tail_keep]
    tail = ledger.entries[len(ledger.entries) - tail_keep :]

    base_payload = {
        "folded_entry_hashes": [entry.entry_hash for entry in prefix],
        "folded_count": len(prefix),
        "tail_anchor_parent": tail[0].parent_hash if tail else GENESIS_HASH,
    }
    base_snapshot_hash = _hash_sha256(base_payload)
    base_retained = sum(entry.retained_count for entry in prefix)
    if prefix:
        base_ratio = _round_float(sum(entry.compression_ratio for entry in prefix) / len(prefix))
    else:
        base_ratio = 0.0

    rebuilt_entries: list[ContextLedgerEntry] = []
    parent_hash = GENESIS_HASH
    head_hash = GENESIS_HASH

    seed_entries = [
        (base_snapshot_hash, base_retained, base_ratio),
        *[(entry.snapshot_hash, entry.retained_count, entry.compression_ratio) for entry in tail],
    ]

    for idx, (snapshot_hash, retained_count, compression_ratio) in enumerate(seed_entries):
        payload = _entry_payload(idx, snapshot_hash, parent_hash, retained_count, compression_ratio)
        entry_hash = _hash_sha256(payload)
        entry = ContextLedgerEntry(
            sequence_id=idx,
            snapshot_hash=snapshot_hash,
            parent_hash=parent_hash,
            retained_count=retained_count,
            compression_ratio=_round_float(compression_ratio),
            entry_hash=entry_hash,
        )
        rebuilt_entries.append(entry)
        head_hash = _hash_sha256({"parent": head_hash, "entry": entry_hash})
        parent_hash = entry_hash

    minimized = ContextLedger(entries=tuple(rebuilt_entries), head_hash=head_hash, chain_valid=True)
    if not validate_context_ledger(minimized):
        raise ValueError("minimized ledger failed validation")
    return minimized


def run_memory_compression_context_ledger(
    items: Iterable[Any],
    prior_ledger: ContextLedger | None = None,
    minimization_threshold: int = 8,
) -> tuple[
    tuple[ContextItem, ...],
    CanonicalMemorySnapshot,
    ContextCompressionReport,
    ContextTransitionReport,
    ContextLedger,
    ContextLedger,
]:
    """Main deterministic orchestration wrapper for memory compression."""
    normalized = normalize_context_items(items)
    retained_items, compacted_ids = deduplicate_context_items(normalized)
    snapshot = build_canonical_memory_snapshot(
        retained_items=retained_items,
        compacted_item_ids=compacted_ids,
        original_count=len(normalized),
    )
    compression_report = compute_context_compression_report(
        original_count=len(normalized),
        retained_count=len(retained_items),
        compacted_count=len(compacted_ids),
    )
    base_ledger = prior_ledger if prior_ledger is not None else empty_context_ledger()
    updated_ledger = append_context_ledger_entry(
        prior_ledger=base_ledger,
        snapshot_hash=snapshot.snapshot_hash,
        retained_count=len(retained_items),
        compression_ratio=compression_report.compression_ratio,
    )
    minimized_ledger = minimize_context_ledger(updated_ledger, threshold=minimization_threshold)

    transition_payload = {
        "total_items": len(normalized),
        "unique_items": len(retained_items),
        "duplicate_items": len(compacted_ids),
        "minimized": len(minimized_ledger.entries) < len(updated_ledger.entries),
    }
    transition_report = ContextTransitionReport(
        total_items=transition_payload["total_items"],
        unique_items=transition_payload["unique_items"],
        duplicate_items=transition_payload["duplicate_items"],
        minimized=transition_payload["minimized"],
        report_hash=_hash_sha256(transition_payload),
    )

    return (
        retained_items,
        snapshot,
        compression_report,
        transition_report,
        updated_ledger,
        minimized_ledger,
    )
