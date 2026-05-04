from __future__ import annotations

from dataclasses import dataclass
import re

from qec.analysis.atomic_semantic_lattice_contract import SemanticLatticeGraph
from qec.analysis.canonical_hashing import sha256_hex

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_VALID_SCALES = {0, 1, 2}


@dataclass(frozen=True)
class SubgraphInvariantPattern:
    pattern_id: str
    node_label_multiset: tuple[str, ...]
    constraint_edge_pairs: tuple[tuple[str, str], ...]
    pattern_hash: str

    def __post_init__(self) -> None:
        if not _is_sorted_str_tuple(self.node_label_multiset):
            raise ValueError("INVALID_INPUT")
        if not _is_sorted_pair_tuple(self.constraint_edge_pairs):
            raise ValueError("INVALID_INPUT")
        _require_constraint_edge_pairs(self.constraint_edge_pairs)

        expected_pattern_id = sha256_hex(
            _pattern_id_payload(self.node_label_multiset, self.constraint_edge_pairs)
        )
        if self.pattern_id != expected_pattern_id:
            raise ValueError("HASH_MISMATCH")

        expected_pattern_hash = sha256_hex(_pattern_hash_payload_raw(self.pattern_id, self.node_label_multiset, self.constraint_edge_pairs))
        if self.pattern_hash != expected_pattern_hash:
            raise ValueError("HASH_MISMATCH")


@dataclass(frozen=True)
class SubgraphOccurrence:
    pattern_id: str
    scale_index: int
    source_node_ids: tuple[str, ...]
    occurrence_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.pattern_id, str) or _SHA256_HEX_RE.fullmatch(self.pattern_id) is None:
            raise ValueError("INVALID_INPUT")
        _require_scale_index(self.scale_index)
        if not _is_sorted_str_tuple(self.source_node_ids):
            raise ValueError("INVALID_INPUT")

        expected_occurrence_hash = sha256_hex(_occurrence_hash_payload(self.pattern_id, self.scale_index, self.source_node_ids))
        if self.occurrence_hash != expected_occurrence_hash:
            raise ValueError("HASH_MISMATCH")


@dataclass(frozen=True)
class SubgraphInvariantPatternReceipt:
    pattern: SubgraphInvariantPattern
    occurrences: tuple[SubgraphOccurrence, ...]
    total_occurrence_count: int
    receipt_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.pattern, SubgraphInvariantPattern):
            raise ValueError("INVALID_INPUT")
        if not isinstance(self.occurrences, tuple) or any(not isinstance(o, SubgraphOccurrence) for o in self.occurrences):
            raise ValueError("INVALID_INPUT")
        if any(o.pattern_id != self.pattern.pattern_id for o in self.occurrences):
            raise ValueError("INVALID_INPUT")
        if tuple(sorted(self.occurrences, key=lambda o: o.occurrence_hash)) != self.occurrences:
            raise ValueError("INVALID_INPUT")
        if not isinstance(self.total_occurrence_count, int) or isinstance(self.total_occurrence_count, bool):
            raise ValueError("INVALID_INPUT")
        if self.total_occurrence_count != len(self.occurrences):
            raise ValueError("OCCURRENCE_COUNT_MISMATCH")

        expected_receipt_hash = sha256_hex(_receipt_hash_payload(self.pattern, self.occurrences, self.total_occurrence_count))
        if self.receipt_hash != expected_receipt_hash:
            raise ValueError("HASH_MISMATCH")


def _is_sorted_str_tuple(value: object) -> bool:
    return isinstance(value, tuple) and all(isinstance(v, str) for v in value) and tuple(sorted(value)) == value


def _is_sorted_pair_tuple(value: object) -> bool:
    if not isinstance(value, tuple):
        return False
    for item in value:
        if not isinstance(item, tuple) or len(item) != 2:
            return False
        if not isinstance(item[0], str) or not isinstance(item[1], str):
            return False
    return tuple(sorted(value)) == value


def _require_scale_index(scale_index: int) -> None:
    if not isinstance(scale_index, int) or isinstance(scale_index, bool) or scale_index not in _VALID_SCALES:
        raise ValueError("INVALID_SCALE_INDEX")


def _require_constraint_edge_pairs(pairs: tuple[tuple[str, str], ...]) -> None:
    for constraint_type, constraint_hash in pairs:
        if _SHA256_HEX_RE.fullmatch(constraint_hash) is None:
            raise ValueError("INVALID_CONSTRAINT_HASH")


def _pattern_id_payload(node_labels: tuple[str, ...], constraint_edge_pairs: tuple[tuple[str, str], ...]) -> dict[str, object]:
    return {
        "node_label_multiset": node_labels,
        "constraint_edge_pairs": [list(p) for p in constraint_edge_pairs],
    }


def _pattern_hash_payload_raw(
    pattern_id: str,
    node_label_multiset: tuple[str, ...],
    constraint_edge_pairs: tuple[tuple[str, str], ...],
) -> dict[str, object]:
    return {
        "pattern_id": pattern_id,
        "node_label_multiset": node_label_multiset,
        "constraint_edge_pairs": [list(p) for p in constraint_edge_pairs],
    }


def _occurrence_hash_payload(pattern_id: str, scale_index: int, source_node_ids: tuple[str, ...]) -> dict[str, object]:
    return {"pattern_id": pattern_id, "scale_index": scale_index, "source_node_ids": source_node_ids}


def _pattern_payload_for_receipt(pattern: SubgraphInvariantPattern) -> dict[str, object]:
    return {
        "pattern_id": pattern.pattern_id,
        "node_label_multiset": pattern.node_label_multiset,
        "constraint_edge_pairs": [list(p) for p in pattern.constraint_edge_pairs],
        "pattern_hash": pattern.pattern_hash,
    }


def _occurrence_payload_for_receipt(occurrence: SubgraphOccurrence) -> dict[str, object]:
    return {
        "pattern_id": occurrence.pattern_id,
        "scale_index": occurrence.scale_index,
        "source_node_ids": occurrence.source_node_ids,
        "occurrence_hash": occurrence.occurrence_hash,
    }


def _receipt_hash_payload(pattern: SubgraphInvariantPattern, occurrences: tuple[SubgraphOccurrence, ...], total_count: int) -> dict[str, object]:
    return {
        "pattern": _pattern_payload_for_receipt(pattern),
        "occurrences": [_occurrence_payload_for_receipt(o) for o in occurrences],
        "total_occurrence_count": total_count,
    }


def build_subgraph_invariant_pattern(node_labels: list[str], constraint_edge_pairs: list[tuple[str, str]]) -> SubgraphInvariantPattern:
    if not all(isinstance(lbl, str) for lbl in node_labels):
        raise ValueError("INVALID_INPUT")
    if not all(isinstance(p, tuple) and len(p) == 2 and isinstance(p[0], str) and isinstance(p[1], str) for p in constraint_edge_pairs):
        raise ValueError("INVALID_INPUT")
    node_multiset = tuple(sorted(node_labels))
    edge_pairs = tuple(sorted(constraint_edge_pairs))
    _require_constraint_edge_pairs(edge_pairs)
    pattern_id = sha256_hex(_pattern_id_payload(node_multiset, edge_pairs))
    pattern_hash = sha256_hex(_pattern_hash_payload_raw(pattern_id, node_multiset, edge_pairs))
    return SubgraphInvariantPattern(pattern_id, node_multiset, edge_pairs, pattern_hash)


def match_graph_to_pattern(pattern: SubgraphInvariantPattern, graph: SemanticLatticeGraph, scale_index: int) -> list[SubgraphOccurrence]:
    """Validate that the entire graph matches a pattern and return a single occurrence if it does.

    This function checks if the whole graph's node label multiset and constraint edge pairs
    exactly match the pattern. It does NOT perform subgraph enumeration. Returns an empty
    list if the graph does not match, or a single-element list with a SubgraphOccurrence
    representing all nodes in the graph.
    """
    _require_scale_index(scale_index)
    node_ids = {n.node_id for n in graph.nodes}
    for edge in graph.edges:
        if edge.source_node_id not in node_ids or edge.target_node_id not in node_ids:
            raise ValueError("INVALID_NODE_ID")

    node_labels = tuple(sorted(node.node_type for node in graph.nodes))
    edge_pairs = tuple(sorted((edge.constraint_type, edge.constraint_payload_hash) for edge in graph.edges))
    if (node_labels, edge_pairs) != (
        pattern.node_label_multiset,
        pattern.constraint_edge_pairs,
    ):
        return []

    source_node_ids = tuple(sorted(node_ids))
    occurrence_hash = sha256_hex(_occurrence_hash_payload(pattern.pattern_id, scale_index, source_node_ids))
    return [SubgraphOccurrence(pattern.pattern_id, scale_index, source_node_ids, occurrence_hash)]


def build_subgraph_invariant_pattern_receipt(pattern: SubgraphInvariantPattern, occurrences: list[SubgraphOccurrence]) -> SubgraphInvariantPatternReceipt:
    normalized: list[SubgraphOccurrence] = []
    for o in occurrences:
        _require_scale_index(o.scale_index)
        source_ids = tuple(sorted(o.source_node_ids))
        occurrence_hash = sha256_hex(_occurrence_hash_payload(o.pattern_id, o.scale_index, source_ids))
        normalized.append(SubgraphOccurrence(o.pattern_id, o.scale_index, source_ids, occurrence_hash))
    sorted_occurrences = tuple(sorted(normalized, key=lambda o: o.occurrence_hash))
    total = len(sorted_occurrences)
    receipt_hash = sha256_hex(_receipt_hash_payload(pattern, sorted_occurrences, total))
    return SubgraphInvariantPatternReceipt(pattern, sorted_occurrences, total, receipt_hash)


def validate_subgraph_invariant_pattern_receipt(receipt: SubgraphInvariantPatternReceipt) -> bool:
    """Validate receipt integrity including nested pattern and occurrence objects.

    This function revalidates all nested objects to prevent accepting tampered receipts
    where an attacker mutated nested data and recomputed the outer receipt_hash.
    """
    # Revalidate the nested pattern by reconstructing it
    pattern = receipt.pattern
    expected_pattern_id = sha256_hex(_pattern_id_payload(pattern.node_label_multiset, pattern.constraint_edge_pairs))
    if pattern.pattern_id != expected_pattern_id:
        raise ValueError("HASH_MISMATCH")
    expected_pattern_hash = sha256_hex(_pattern_hash_payload_raw(pattern.pattern_id, pattern.node_label_multiset, pattern.constraint_edge_pairs))
    if pattern.pattern_hash != expected_pattern_hash:
        raise ValueError("HASH_MISMATCH")
    if not _is_sorted_str_tuple(pattern.node_label_multiset):
        raise ValueError("INVALID_INPUT")
    if not _is_sorted_pair_tuple(pattern.constraint_edge_pairs):
        raise ValueError("INVALID_INPUT")
    _require_constraint_edge_pairs(pattern.constraint_edge_pairs)

    # Revalidate each occurrence
    for occ in receipt.occurrences:
        if occ.pattern_id != pattern.pattern_id:
            raise ValueError("INVALID_INPUT")
        _require_scale_index(occ.scale_index)
        if not _is_sorted_str_tuple(occ.source_node_ids):
            raise ValueError("INVALID_INPUT")
        expected_occ_hash = sha256_hex(_occurrence_hash_payload(occ.pattern_id, occ.scale_index, occ.source_node_ids))
        if occ.occurrence_hash != expected_occ_hash:
            raise ValueError("HASH_MISMATCH")

    # Validate occurrence ordering
    if tuple(sorted(receipt.occurrences, key=lambda o: o.occurrence_hash)) != receipt.occurrences:
        raise ValueError("INVALID_INPUT")

    # Validate count and receipt hash
    if receipt.total_occurrence_count != len(receipt.occurrences):
        raise ValueError("OCCURRENCE_COUNT_MISMATCH")
    expected_hash = sha256_hex(_receipt_hash_payload(receipt.pattern, receipt.occurrences, receipt.total_occurrence_count))
    if expected_hash != receipt.receipt_hash:
        raise ValueError("HASH_MISMATCH")
    return True
