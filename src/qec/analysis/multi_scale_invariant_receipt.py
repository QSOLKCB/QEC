from __future__ import annotations

from dataclasses import dataclass

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.subgraph_invariant_pattern import (
    SubgraphInvariantPattern,
    SubgraphOccurrence,
    _is_sorted_pair_tuple,
    _is_sorted_str_tuple,
    _pattern_hash_payload_raw,
    _pattern_id_payload,
    _require_constraint_edge_pairs,
)

_VALID_SCALES = {0, 1, 2}


@dataclass(frozen=True)
class ScaleLevelSummary:
    pattern_id: str
    scale_index: int
    occurrence_hashes: tuple[str, ...]
    occurrence_count: int
    scale_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.scale_index, int) or isinstance(self.scale_index, bool) or self.scale_index not in _VALID_SCALES:
            raise ValueError("INVALID_SCALE_INDEX")
        if tuple(sorted(self.occurrence_hashes)) != self.occurrence_hashes:
            raise ValueError("INVALID_INPUT")
        if not isinstance(self.occurrence_count, int) or isinstance(self.occurrence_count, bool):
            raise ValueError("INVALID_INPUT")
        if self.occurrence_count != len(self.occurrence_hashes):
            raise ValueError("OCCURRENCE_COUNT_MISMATCH")

        expected_hash = sha256_hex(
            _scale_hash_payload(
                self.pattern_id,
                self.scale_index,
                self.occurrence_hashes,
                self.occurrence_count,
            )
        )
        if self.scale_hash != expected_hash:
            raise ValueError("HASH_MISMATCH")


@dataclass(frozen=True)
class MultiScaleInvariantReceipt:
    """Deterministic aggregated receipt for a pattern across observed scales.

    Empty occurrences are valid.

    If no occurrences are provided:
    - scale_summaries is an empty tuple
    - total_occurrence_count == 0
    - receipt_hash is still deterministically computed

    No implicit scale_index entries are created.
    Only observed scales are represented.
    """

    pattern: SubgraphInvariantPattern
    scale_summaries: tuple[ScaleLevelSummary, ...]
    total_occurrence_count: int
    receipt_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.pattern, SubgraphInvariantPattern):
            raise ValueError("INVALID_INPUT")
        if not isinstance(self.scale_summaries, tuple) or any(not isinstance(s, ScaleLevelSummary) for s in self.scale_summaries):
            raise ValueError("INVALID_INPUT")

        sorted_summaries = tuple(sorted(self.scale_summaries, key=lambda s: s.scale_index))
        if sorted_summaries != self.scale_summaries:
            raise ValueError("INVALID_INPUT")

        seen: set[int] = set()
        for summary in self.scale_summaries:
            if summary.scale_index in seen:
                raise ValueError("INVALID_INPUT")
            seen.add(summary.scale_index)
            if summary.pattern_id != self.pattern.pattern_id:
                raise ValueError("INVALID_INPUT")

        if not isinstance(self.total_occurrence_count, int) or isinstance(self.total_occurrence_count, bool):
            raise ValueError("INVALID_INPUT")
        expected_total = sum(summary.occurrence_count for summary in self.scale_summaries)
        if self.total_occurrence_count != expected_total:
            raise ValueError("OCCURRENCE_COUNT_MISMATCH")

        expected_hash = sha256_hex(
            _receipt_hash_payload(self.pattern, self.scale_summaries, self.total_occurrence_count)
        )
        if self.receipt_hash != expected_hash:
            raise ValueError("HASH_MISMATCH")


def _scale_hash_payload(
    pattern_id: str,
    scale_index: int,
    occurrence_hashes: tuple[str, ...],
    occurrence_count: int,
) -> dict[str, object]:
    return {
        "pattern_id": pattern_id,
        "scale_index": scale_index,
        "occurrence_hashes": occurrence_hashes,
        "occurrence_count": occurrence_count,
    }


def _pattern_payload(pattern: SubgraphInvariantPattern) -> dict[str, object]:
    return {
        "pattern_id": pattern.pattern_id,
        "node_label_multiset": pattern.node_label_multiset,
        "constraint_edge_pairs": [list(p) for p in pattern.constraint_edge_pairs],
        "pattern_hash": pattern.pattern_hash,
    }


def _summary_payload(summary: ScaleLevelSummary) -> dict[str, object]:
    return {
        "pattern_id": summary.pattern_id,
        "scale_index": summary.scale_index,
        "occurrence_hashes": summary.occurrence_hashes,
        "occurrence_count": summary.occurrence_count,
        "scale_hash": summary.scale_hash,
    }


def _receipt_hash_payload(
    pattern: SubgraphInvariantPattern,
    scale_summaries: tuple[ScaleLevelSummary, ...],
    total_occurrence_count: int,
) -> dict[str, object]:
    return {
        "pattern": _pattern_payload(pattern),
        "scale_summaries": [_summary_payload(s) for s in scale_summaries],
        "total_occurrence_count": total_occurrence_count,
    }


def build_scale_level_summary(
    pattern_id: str,
    scale_index: int,
    occurrences: list[SubgraphOccurrence],
) -> ScaleLevelSummary:
    filtered = [o for o in occurrences if o.scale_index == scale_index and o.pattern_id == pattern_id]
    occurrence_hashes = tuple(sorted(o.occurrence_hash for o in filtered))
    occurrence_count = len(occurrence_hashes)
    scale_hash = sha256_hex(_scale_hash_payload(pattern_id, scale_index, occurrence_hashes, occurrence_count))
    return ScaleLevelSummary(pattern_id, scale_index, occurrence_hashes, occurrence_count, scale_hash)


def build_multi_scale_invariant_receipt(
    pattern: SubgraphInvariantPattern,
    occurrences: list[SubgraphOccurrence],
) -> MultiScaleInvariantReceipt:
    """Build a deterministic receipt from observed occurrences only.

    Only scale_index values present in occurrences are included.
    No implicit scale insertion occurs.
    Missing scales (e.g., no scale_index=1 occurrences) are not represented.
    """

    by_scale: dict[int, list[SubgraphOccurrence]] = {}
    for occurrence in occurrences:
        if occurrence.pattern_id != pattern.pattern_id:
            raise ValueError("INVALID_INPUT")
        by_scale.setdefault(occurrence.scale_index, []).append(occurrence)

    summaries = tuple(
        sorted(
            [
                build_scale_level_summary(pattern.pattern_id, scale_index, scale_occurrences)
                for scale_index, scale_occurrences in by_scale.items()
            ],
            key=lambda summary: summary.scale_index,
        )
    )

    total_occurrence_count = sum(summary.occurrence_count for summary in summaries)
    receipt_hash = sha256_hex(_receipt_hash_payload(pattern, summaries, total_occurrence_count))
    return MultiScaleInvariantReceipt(pattern, summaries, total_occurrence_count, receipt_hash)


def validate_multi_scale_invariant_receipt(
    receipt: MultiScaleInvariantReceipt,
) -> bool:
    """Validate receipt integrity including nested pattern and summary objects.

    This function revalidates all nested objects to prevent accepting tampered receipts
    where an attacker mutated nested data and recomputed the outer receipt_hash.
    """
    if not isinstance(receipt.pattern, SubgraphInvariantPattern):
        raise ValueError("INVALID_INPUT")

    # Revalidate the nested pattern by reconstructing it
    pattern = receipt.pattern
    if not _is_sorted_str_tuple(pattern.node_label_multiset):
        raise ValueError("INVALID_INPUT")
    if not _is_sorted_pair_tuple(pattern.constraint_edge_pairs):
        raise ValueError("INVALID_INPUT")
    _require_constraint_edge_pairs(pattern.constraint_edge_pairs)

    expected_pattern_id = sha256_hex(_pattern_id_payload(pattern.node_label_multiset, pattern.constraint_edge_pairs))
    if pattern.pattern_id != expected_pattern_id:
        raise ValueError("HASH_MISMATCH")
    expected_pattern_hash = sha256_hex(_pattern_hash_payload_raw(pattern.pattern_id, pattern.node_label_multiset, pattern.constraint_edge_pairs))
    if pattern.pattern_hash != expected_pattern_hash:
        raise ValueError("HASH_MISMATCH")

    # Validate total_occurrence_count type
    if not isinstance(receipt.total_occurrence_count, int) or isinstance(receipt.total_occurrence_count, bool):
        raise ValueError("INVALID_INPUT")

    expected_scale_summaries = tuple(sorted(receipt.scale_summaries, key=lambda s: s.scale_index))
    if receipt.scale_summaries != expected_scale_summaries:
        raise ValueError("INVALID_INPUT")

    seen: set[int] = set()
    for summary in receipt.scale_summaries:
        # Validate scale_index type, non-bool, and membership in valid scales
        if not isinstance(summary.scale_index, int) or isinstance(summary.scale_index, bool) or summary.scale_index not in _VALID_SCALES:
            raise ValueError("INVALID_SCALE_INDEX")

        if summary.scale_index in seen:
            raise ValueError("INVALID_INPUT")
        seen.add(summary.scale_index)

        if summary.pattern_id != receipt.pattern.pattern_id:
            raise ValueError("INVALID_INPUT")

        # Validate occurrence_count type
        if not isinstance(summary.occurrence_count, int) or isinstance(summary.occurrence_count, bool):
            raise ValueError("INVALID_INPUT")

        # Validate occurrence_hashes is sorted
        if tuple(sorted(summary.occurrence_hashes)) != summary.occurrence_hashes:
            raise ValueError("INVALID_INPUT")

        # Validate occurrence_count matches len(occurrence_hashes)
        if summary.occurrence_count != len(summary.occurrence_hashes):
            raise ValueError("OCCURRENCE_COUNT_MISMATCH")

        expected_scale_hash = sha256_hex(
            _scale_hash_payload(
                summary.pattern_id,
                summary.scale_index,
                summary.occurrence_hashes,
                summary.occurrence_count,
            )
        )
        if summary.scale_hash != expected_scale_hash:
            raise ValueError("HASH_MISMATCH")

    expected_total = sum(summary.occurrence_count for summary in receipt.scale_summaries)
    if receipt.total_occurrence_count != expected_total:
        raise ValueError("OCCURRENCE_COUNT_MISMATCH")

    expected_receipt_hash = sha256_hex(
        _receipt_hash_payload(receipt.pattern, receipt.scale_summaries, receipt.total_occurrence_count)
    )
    if receipt.receipt_hash != expected_receipt_hash:
        raise ValueError("HASH_MISMATCH")
    return True
