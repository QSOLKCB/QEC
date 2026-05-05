from __future__ import annotations

from dataclasses import dataclass
from re import compile as re_compile
from types import MappingProxyType

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.subgraph_invariant_pattern import (
    SubgraphInvariantPatternReceipt,
    SubgraphOccurrence,
    _VALID_SCALES,
)

_SHA256_HEX_RE = re_compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SierpinskiCompressionEntry:
    pattern_hash: str
    scale_index: int
    occurrence_count: int
    source_node_id_sets: tuple[tuple[str, ...], ...]
    compression_entry_hash: str

    def __post_init__(self) -> None:
        _validate_compression_entry(self)


@dataclass(frozen=True)
class SierpinskiCompressionReceipt:
    multi_scale_invariant_receipt_hash: str
    compression_entries: tuple[SierpinskiCompressionEntry, ...]
    total_compressed_occurrences: int
    total_entries: int
    sierpinski_compression_receipt_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.multi_scale_invariant_receipt_hash, str) or _SHA256_HEX_RE.fullmatch(self.multi_scale_invariant_receipt_hash) is None:
            raise ValueError("INVALID_INPUT")
        if not isinstance(self.compression_entries, tuple) or any(not isinstance(e, SierpinskiCompressionEntry) for e in self.compression_entries):
            raise ValueError("INVALID_INPUT")
        if tuple(sorted(self.compression_entries, key=lambda e: (e.scale_index, e.pattern_hash))) != self.compression_entries:
            raise ValueError("INVALID_INPUT")
        if len({(e.scale_index, e.pattern_hash) for e in self.compression_entries}) != len(self.compression_entries):
            raise ValueError("INVALID_INPUT")
        if any(e.occurrence_count < 2 for e in self.compression_entries):
            raise ValueError("INSUFFICIENT_OCCURRENCES_FOR_COMPRESSION")
        if not isinstance(self.total_entries, int) or isinstance(self.total_entries, bool):
            raise ValueError("INVALID_INPUT")
        if self.total_entries != len(self.compression_entries):
            raise ValueError("INVALID_INPUT")
        if not isinstance(self.total_compressed_occurrences, int) or isinstance(self.total_compressed_occurrences, bool):
            raise ValueError("INVALID_INPUT")
        if self.total_compressed_occurrences != sum(e.occurrence_count for e in self.compression_entries):
            raise ValueError("INVALID_INPUT")
        expected_hash = sha256_hex(_receipt_hash_payload(self.multi_scale_invariant_receipt_hash, self.compression_entries, self.total_compressed_occurrences, self.total_entries))
        if self.sierpinski_compression_receipt_hash != expected_hash:
            raise ValueError("HASH_MISMATCH")


@dataclass(frozen=True)
class SierpinskiCompressionBuildResult:
    """Explicit, frozen context artifact required for deterministic decompression.
    Contains pattern_id_map and canonical original_occurrences.
    No hidden runtime state is used."""
    receipt: SierpinskiCompressionReceipt
    pattern_id_map: MappingProxyType[str, str]
    original_occurrences: tuple[SubgraphOccurrence, ...]


def _entry_hash_payload(pattern_hash: str, scale_index: int, occurrence_count: int, source_node_id_sets: tuple[tuple[str, ...], ...]) -> dict[str, object]:
    return {"pattern_hash": pattern_hash, "scale_index": scale_index, "occurrence_count": occurrence_count, "source_node_id_sets": source_node_id_sets}


def _receipt_hash_payload(multi_scale_invariant_receipt_hash: str, compression_entries: tuple[SierpinskiCompressionEntry, ...], total_compressed_occurrences: int, total_entries: int) -> dict[str, object]:
    return {
        "multi_scale_invariant_receipt_hash": multi_scale_invariant_receipt_hash,
        "compression_entries": [_entry_hash_payload(e.pattern_hash, e.scale_index, e.occurrence_count, e.source_node_id_sets) for e in compression_entries],
        "total_compressed_occurrences": total_compressed_occurrences,
        "total_entries": total_entries,
    }


def _validate_compression_entry(entry: SierpinskiCompressionEntry) -> None:
    """Shared validation logic for a SierpinskiCompressionEntry.

    Called from SierpinskiCompressionEntry.__post_init__ and validate_sierpinski_compression_receipt
    to enforce all structural and hash invariants in a single place.

    Args:
        entry: The SierpinskiCompressionEntry to validate.

    Raises:
        ValueError("INVALID_INPUT"): If any field has an invalid type, format, or ordering.
        ValueError("INSUFFICIENT_OCCURRENCES_FOR_COMPRESSION"): If occurrence_count < 2.
        ValueError("HASH_MISMATCH"): If compression_entry_hash does not match the recomputed hash.
    """
    if not isinstance(entry.pattern_hash, str) or _SHA256_HEX_RE.fullmatch(entry.pattern_hash) is None:
        raise ValueError("INVALID_INPUT")
    if not isinstance(entry.scale_index, int) or isinstance(entry.scale_index, bool) or entry.scale_index not in _VALID_SCALES:
        raise ValueError("INVALID_INPUT")
    if not isinstance(entry.occurrence_count, int) or isinstance(entry.occurrence_count, bool):
        raise ValueError("INVALID_INPUT")
    if entry.occurrence_count < 2:
        raise ValueError("INSUFFICIENT_OCCURRENCES_FOR_COMPRESSION")
    if not isinstance(entry.source_node_id_sets, tuple) or any(not isinstance(s, tuple) for s in entry.source_node_id_sets):
        raise ValueError("INVALID_INPUT")
    if entry.occurrence_count != len(entry.source_node_id_sets):
        raise ValueError("INVALID_INPUT")
    if any(any(not isinstance(v, str) for v in s) for s in entry.source_node_id_sets):
        raise ValueError("INVALID_INPUT")
    if any(tuple(sorted(s)) != s for s in entry.source_node_id_sets):
        raise ValueError("INVALID_INPUT")
    if tuple(sorted(entry.source_node_id_sets)) != entry.source_node_id_sets:
        raise ValueError("INVALID_INPUT")
    expected_hash = sha256_hex(_entry_hash_payload(entry.pattern_hash, entry.scale_index, entry.occurrence_count, entry.source_node_id_sets))
    if entry.compression_entry_hash != expected_hash:
        raise ValueError("HASH_MISMATCH")


def build_sierpinski_compression_context(multi_scale_invariant_receipt_hash: str, pattern_receipts: list[SubgraphInvariantPatternReceipt]) -> SierpinskiCompressionBuildResult:
    """Build a deterministic compression context anchored to a multi-scale invariant receipt hash.

    The hash anchor binds the compression artifact to an external multi-scale analysis result.
    Cross-pattern compression is intentionally supported; the anchor is a hash reference only.

    Args:
        multi_scale_invariant_receipt_hash: A valid SHA-256 hex string that anchors this
            compression artifact to an external multi-scale invariant receipt.
        pattern_receipts: A list of SubgraphInvariantPatternReceipt objects whose occurrences
            are candidates for compression. Patterns with fewer than 2 occurrences per scale
            are silently skipped.

    Returns:
        A SierpinskiCompressionBuildResult containing the frozen receipt, an immutable
        pattern_id_map, and the canonical original_occurrences required for decompression.

    Raises:
        ValueError("INVALID_INPUT"): If multi_scale_invariant_receipt_hash is not a valid
            SHA-256 hex string, pattern_receipts is not a list, or any receipt is invalid.
    """
    if not isinstance(multi_scale_invariant_receipt_hash, str) or _SHA256_HEX_RE.fullmatch(multi_scale_invariant_receipt_hash) is None:
        raise ValueError("INVALID_INPUT")
    if not isinstance(pattern_receipts, list):
        raise ValueError("INVALID_INPUT")
    ms_hash = multi_scale_invariant_receipt_hash

    entries: list[SierpinskiCompressionEntry] = []
    pattern_id_map: dict[str, str] = {}
    original_occurrences: list[SubgraphOccurrence] = []

    for pr in pattern_receipts:
        if not isinstance(pr, SubgraphInvariantPatternReceipt):
            raise ValueError("INVALID_INPUT")
        if pr.total_occurrence_count != len(pr.occurrences):
            raise ValueError("INVALID_INPUT")
        if any(o.pattern_id != pr.pattern.pattern_id for o in pr.occurrences):
            raise ValueError("INVALID_INPUT")
        pattern_hash = pr.pattern.pattern_hash
        pattern_id = pr.pattern.pattern_id
        if pattern_hash in pattern_id_map and pattern_id_map[pattern_hash] != pattern_id:
            raise ValueError("INVALID_INPUT")
        pattern_id_map[pattern_hash] = pattern_id
        if pr.total_occurrence_count < 2:
            continue
        by_scale: dict[int, list[SubgraphOccurrence]] = {}
        for occ in pr.occurrences:
            by_scale.setdefault(occ.scale_index, []).append(occ)
        for scale_index, scale_occurrences in by_scale.items():
            if len(scale_occurrences) < 2:
                continue
            source_node_id_sets = tuple(sorted(tuple(sorted(occ.source_node_ids)) for occ in scale_occurrences))
            entry_hash = sha256_hex(_entry_hash_payload(pattern_hash, scale_index, len(scale_occurrences), source_node_id_sets))
            entries.append(SierpinskiCompressionEntry(pattern_hash, scale_index, len(scale_occurrences), source_node_id_sets, entry_hash))
            original_occurrences.extend(scale_occurrences)

    compression_entries = tuple(sorted(entries, key=lambda e: (e.scale_index, e.pattern_hash)))
    total_occ = sum(e.occurrence_count for e in compression_entries)
    receipt_hash = sha256_hex(_receipt_hash_payload(ms_hash, compression_entries, total_occ, len(compression_entries)))
    receipt = SierpinskiCompressionReceipt(ms_hash, compression_entries, total_occ, len(compression_entries), receipt_hash)
    canonical_original = tuple(sorted(original_occurrences, key=lambda o: o.occurrence_hash))
    return SierpinskiCompressionBuildResult(receipt, MappingProxyType(dict(sorted(pattern_id_map.items()))), canonical_original)


def build_sierpinski_compression_receipt(multi_scale_invariant_receipt_hash: str, pattern_receipts: list[SubgraphInvariantPatternReceipt]) -> SierpinskiCompressionReceipt:
    return build_sierpinski_compression_context(multi_scale_invariant_receipt_hash, pattern_receipts).receipt


def validate_sierpinski_compression_receipt(receipt: SierpinskiCompressionReceipt) -> bool:
    if not isinstance(receipt, SierpinskiCompressionReceipt):
        raise ValueError("INVALID_INPUT")
    if not isinstance(receipt.multi_scale_invariant_receipt_hash, str) or _SHA256_HEX_RE.fullmatch(receipt.multi_scale_invariant_receipt_hash) is None:
        raise ValueError("INVALID_INPUT")
    if not isinstance(receipt.compression_entries, tuple) or any(not isinstance(e, SierpinskiCompressionEntry) for e in receipt.compression_entries):
        raise ValueError("INVALID_INPUT")
    if tuple(sorted(receipt.compression_entries, key=lambda e: (e.scale_index, e.pattern_hash))) != receipt.compression_entries:
        raise ValueError("INVALID_INPUT")
    if len({(e.scale_index, e.pattern_hash) for e in receipt.compression_entries}) != len(receipt.compression_entries):
        raise ValueError("INVALID_INPUT")

    for entry in receipt.compression_entries:
        _validate_compression_entry(entry)

    if not isinstance(receipt.total_entries, int) or isinstance(receipt.total_entries, bool) or receipt.total_entries != len(receipt.compression_entries):
        raise ValueError("INVALID_INPUT")
    if not isinstance(receipt.total_compressed_occurrences, int) or isinstance(receipt.total_compressed_occurrences, bool):
        raise ValueError("INVALID_INPUT")
    if receipt.total_compressed_occurrences != sum(e.occurrence_count for e in receipt.compression_entries):
        raise ValueError("INVALID_INPUT")
    if receipt.sierpinski_compression_receipt_hash != sha256_hex(_receipt_hash_payload(receipt.multi_scale_invariant_receipt_hash, receipt.compression_entries, receipt.total_compressed_occurrences, receipt.total_entries)):
        raise ValueError("HASH_MISMATCH")
    return True


def decompress_sierpinski_receipt(receipt: SierpinskiCompressionReceipt, pattern_id_map: dict[str, str], original_occurrences: tuple[SubgraphOccurrence, ...]) -> list[SubgraphOccurrence]:
    """Decompression returns occurrences sorted by occurrence_hash (canonical order).

    Invariant: decompress(compress(x)) == x (under canonical occurrence ordering).
    """
    validate_sierpinski_compression_receipt(receipt)
    if not isinstance(pattern_id_map, dict) or not isinstance(original_occurrences, tuple):
        raise ValueError("DECOMPRESSION_IDENTITY_FAILURE")
    if any(not isinstance(o, SubgraphOccurrence) for o in original_occurrences):
        raise ValueError("DECOMPRESSION_IDENTITY_FAILURE")
    reconstructed: list[SubgraphOccurrence] = []
    for entry in receipt.compression_entries:
        if entry.pattern_hash not in pattern_id_map:
            raise ValueError("DECOMPRESSION_IDENTITY_FAILURE")
        pattern_id = pattern_id_map[entry.pattern_hash]
        if not isinstance(pattern_id, str):
            raise ValueError("DECOMPRESSION_IDENTITY_FAILURE")
        for source_node_ids in entry.source_node_id_sets:
            occ_hash = sha256_hex({"pattern_id": pattern_id, "scale_index": entry.scale_index, "source_node_ids": source_node_ids})
            try:
                occ = SubgraphOccurrence(pattern_id, entry.scale_index, source_node_ids, occ_hash)
            except ValueError:
                raise ValueError("DECOMPRESSION_IDENTITY_FAILURE") from None
            reconstructed.append(occ)
    reconstructed_sorted = sorted(reconstructed, key=lambda o: o.occurrence_hash)
    original_sorted = sorted(original_occurrences, key=lambda o: o.occurrence_hash)
    if reconstructed_sorted != original_sorted:
        raise ValueError("DECOMPRESSION_IDENTITY_FAILURE")
    return reconstructed_sorted
