from __future__ import annotations

import pytest

from qec.analysis.atomic_semantic_lattice_contract import AtomicLatticeBounds, ConstraintEdgeReceipt, SemanticLatticeNode, build_semantic_lattice_graph
from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis import subgraph_invariant_pattern as sip
from qec.analysis.subgraph_invariant_pattern import (
    SubgraphInvariantPattern,
    SubgraphInvariantPatternReceipt,
    SubgraphOccurrence,
    build_subgraph_invariant_pattern,
    build_subgraph_invariant_pattern_receipt,
    match_graph_to_pattern,
    validate_subgraph_invariant_pattern_receipt,
)


def _bounds() -> AtomicLatticeBounds:
    b = AtomicLatticeBounds("b", "153.0", 5, 5, 5, "")
    return AtomicLatticeBounds(**{**b.__dict__, "bounds_hash": b.stable_hash()})


def _node(nid: str, coord: tuple[int, int, int], node_type: str) -> SemanticLatticeNode:
    n = SemanticLatticeNode(nid, node_type, coord, "a" * 64, "b" * 64, {}, "")
    return SemanticLatticeNode(**{**n.__dict__, "node_hash": n.stable_hash()})


def _edge(eid: str, s: str, t: str, ctype: str, payload: dict[str, object]) -> ConstraintEdgeReceipt:
    p_hash = sha256_hex(payload)
    e = ConstraintEdgeReceipt(eid, s, t, ctype, payload, p_hash, "", "")
    e = ConstraintEdgeReceipt(**{**e.__dict__, "edge_hash": e._edge_stable_hash()})
    return ConstraintEdgeReceipt(**{**e.__dict__, "receipt_hash": e.stable_hash()})


def _graph():
    n1 = _node("n1", (0, 0, 0), "A")
    n2 = _node("n2", (1, 0, 0), "B")
    e1 = _edge("e1", "n1", "n2", "adjacent", {"k": "v"})
    return build_semantic_lattice_graph("l", "153.9", _bounds(), (n1, n2), (e1,))


def test_subgraph_invariant_pattern_determinism():
    h = sha256_hex({"k": "v"})
    a = build_subgraph_invariant_pattern(["B", "A"], [("adjacent", h)])
    b = build_subgraph_invariant_pattern(["A", "B"], [("adjacent", h)])
    assert a.pattern_id == b.pattern_id
    assert a.pattern_hash == b.pattern_hash
    for _ in range(5):
        c = build_subgraph_invariant_pattern(["A", "B"], [("adjacent", h)])
        assert c.pattern_id == a.pattern_id
        assert c.pattern_hash == a.pattern_hash


def test_subgraph_invariant_pattern_rejects_invalid_constraint_hash():
    with pytest.raises(ValueError, match="INVALID_CONSTRAINT_HASH"):
        build_subgraph_invariant_pattern(["A"], [("x", "not-a-hash")])


def test_subgraph_occurrence_scale_index_validation():
    g = _graph()
    p = build_subgraph_invariant_pattern(["A", "B"], [("adjacent", g.edges[0].constraint_payload_hash)])
    with pytest.raises(ValueError, match="INVALID_SCALE_INDEX"):
        match_graph_to_pattern(p, g, 3)
    with pytest.raises(ValueError, match="INVALID_SCALE_INDEX"):
        match_graph_to_pattern(p, g, True)
    with pytest.raises(ValueError, match="INVALID_SCALE_INDEX"):
        match_graph_to_pattern(p, g, False)


def test_subgraph_occurrence_source_node_validation():
    class FakeNode:
        def __init__(self, node_id: str, node_type: str):
            self.node_id = node_id
            self.node_type = node_type

    class FakeEdge:
        def __init__(self):
            self.source_node_id = "n1"
            self.target_node_id = "missing"
            self.constraint_type = "adjacent"
            self.constraint_payload_hash = "a" * 64

    class FakeGraph:
        nodes = (FakeNode("n1", "A"),)
        edges = (FakeEdge(),)

    p = build_subgraph_invariant_pattern(["A"], [("adjacent", "a" * 64)])
    with pytest.raises(ValueError, match="INVALID_NODE_ID"):
        match_graph_to_pattern(p, FakeGraph(), 0)


def test_subgraph_invariant_pattern_receipt_determinism():
    g = _graph()
    p = build_subgraph_invariant_pattern(["A", "B"], [("adjacent", g.edges[0].constraint_payload_hash)])
    occ = match_graph_to_pattern(p, g, 0)
    r1 = build_subgraph_invariant_pattern_receipt(p, occ)
    r2 = build_subgraph_invariant_pattern_receipt(p, list(reversed(occ)))
    assert r1.receipt_hash == r2.receipt_hash


def test_subgraph_invariant_pattern_receipt_count_mismatch():
    g = _graph()
    p = build_subgraph_invariant_pattern(["A", "B"], [("adjacent", g.edges[0].constraint_payload_hash)])
    r = build_subgraph_invariant_pattern_receipt(p, match_graph_to_pattern(p, g, 0))
    object.__setattr__(r, "total_occurrence_count", r.total_occurrence_count + 1)
    with pytest.raises(ValueError, match="OCCURRENCE_COUNT_MISMATCH"):
        validate_subgraph_invariant_pattern_receipt(r)


def test_subgraph_invariant_pattern_receipt_hash_mismatch():
    g = _graph()
    p = build_subgraph_invariant_pattern(["A", "B"], [("adjacent", g.edges[0].constraint_payload_hash)])
    r = build_subgraph_invariant_pattern_receipt(p, match_graph_to_pattern(p, g, 0))
    object.__setattr__(r, "receipt_hash", "0" * 64)
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_subgraph_invariant_pattern_receipt(r)


def test_subgraph_occurrences_sorted_by_hash():
    p = build_subgraph_invariant_pattern(["A", "B"], [("adjacent", sha256_hex({"k": "v"}))])
    occ_a = SubgraphOccurrence(p.pattern_id, 0, ("a",), sha256_hex({"pattern_id": p.pattern_id, "scale_index": 0, "source_node_ids": ("a",)}))
    occ_b = SubgraphOccurrence(p.pattern_id, 0, ("b",), sha256_hex({"pattern_id": p.pattern_id, "scale_index": 0, "source_node_ids": ("b",)}))
    receipt = build_subgraph_invariant_pattern_receipt(p, [occ_b, occ_a])
    hashes = [o.occurrence_hash for o in receipt.occurrences]
    assert hashes == sorted(hashes)


def test_cross_run_hash_stability():
    g = _graph()
    p_hashes = set()
    r_hashes = set()
    for _ in range(10):
        p = build_subgraph_invariant_pattern(["A", "B"], [("adjacent", g.edges[0].constraint_payload_hash)])
        r = build_subgraph_invariant_pattern_receipt(p, match_graph_to_pattern(p, g, 1))
        p_hashes.add((p.pattern_id, p.pattern_hash))
        r_hashes.add(r.receipt_hash)
    assert len(p_hashes) == 1
    assert len(r_hashes) == 1


def test_direct_pattern_hash_and_id_validation_and_ordering():
    p = build_subgraph_invariant_pattern(["A", "B"], [("adjacent", "a" * 64)])
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        SubgraphInvariantPattern("0" * 64, p.node_label_multiset, p.constraint_edge_pairs, p.pattern_hash)
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        SubgraphInvariantPattern(p.pattern_id, p.node_label_multiset, p.constraint_edge_pairs, "0" * 64)
    with pytest.raises(ValueError, match="INVALID_CONSTRAINT_HASH"):
        SubgraphInvariantPattern(p.pattern_id, p.node_label_multiset, (("bad", "notahash"),), p.pattern_hash)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        SubgraphInvariantPattern(p.pattern_id, ("B", "A"), p.constraint_edge_pairs, p.pattern_hash)


def test_occurrence_self_validation():
    p = build_subgraph_invariant_pattern(["A"], [("x", "a" * 64)])
    good_hash = sha256_hex({"pattern_id": p.pattern_id, "scale_index": 0, "source_node_ids": ("n1",)})
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        SubgraphOccurrence(p.pattern_id, 0, ("n1",), "0" * 64)
    with pytest.raises(ValueError, match="INVALID_SCALE_INDEX"):
        SubgraphOccurrence(p.pattern_id, True, ("n1",), good_hash)
    with pytest.raises(ValueError, match="INVALID_SCALE_INDEX"):
        SubgraphOccurrence(p.pattern_id, False, ("n1",), good_hash)
    with pytest.raises(ValueError, match="INVALID_SCALE_INDEX"):
        SubgraphOccurrence(p.pattern_id, 3, ("n1",), good_hash)
    unsorted_hash = sha256_hex({"pattern_id": p.pattern_id, "scale_index": 0, "source_node_ids": ("b", "a")})
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        SubgraphOccurrence(p.pattern_id, 0, ("b", "a"), unsorted_hash)


def test_receipt_self_validation():
    g = _graph()
    p = build_subgraph_invariant_pattern(["A", "B"], [("adjacent", g.edges[0].constraint_payload_hash)])
    occ = tuple(match_graph_to_pattern(p, g, 0))
    r = build_subgraph_invariant_pattern_receipt(p, list(occ))

    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        SubgraphInvariantPatternReceipt(r.pattern, r.occurrences, r.total_occurrence_count, "0" * 64)
    with pytest.raises(ValueError, match="OCCURRENCE_COUNT_MISMATCH"):
        SubgraphInvariantPatternReceipt(r.pattern, r.occurrences, r.total_occurrence_count + 1, r.receipt_hash)

    occ_a = SubgraphOccurrence(p.pattern_id, 0, ("a",), sha256_hex({"pattern_id": p.pattern_id, "scale_index": 0, "source_node_ids": ("a",)}))
    occ_b = SubgraphOccurrence(p.pattern_id, 0, ("b",), sha256_hex({"pattern_id": p.pattern_id, "scale_index": 0, "source_node_ids": ("b",)}))
    occ_unsorted = (occ_b, occ_a) if occ_b.occurrence_hash < occ_a.occurrence_hash else (occ_a, occ_b)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        SubgraphInvariantPatternReceipt(r.pattern, tuple(reversed(occ_unsorted)), 2, r.receipt_hash)

    p2 = build_subgraph_invariant_pattern(["A"], [("x", "a" * 64)])
    wrong_occ = SubgraphOccurrence(p2.pattern_id, 0, ("n1",), sha256_hex({"pattern_id": p2.pattern_id, "scale_index": 0, "source_node_ids": ("n1",)}))
    mismatch_hash = sha256_hex({"pattern": {"pattern_id": p.pattern_id, "node_label_multiset": p.node_label_multiset, "constraint_edge_pairs": [list(pair) for pair in p.constraint_edge_pairs], "pattern_hash": p.pattern_hash}, "occurrences": [{"pattern_id": wrong_occ.pattern_id, "scale_index": wrong_occ.scale_index, "source_node_ids": wrong_occ.source_node_ids, "occurrence_hash": wrong_occ.occurrence_hash}], "total_occurrence_count": 1})
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        SubgraphInvariantPatternReceipt(p, (wrong_occ,), 1, mismatch_hash)


def test_builder_instantiates_pattern_once(monkeypatch):
    real_cls = sip.SubgraphInvariantPattern
    calls = {"count": 0}

    def wrapper(*args, **kwargs):
        calls["count"] += 1
        return real_cls(*args, **kwargs)

    monkeypatch.setattr(sip, "SubgraphInvariantPattern", wrapper)
    h = sha256_hex({"k": "v"})
    out = sip.build_subgraph_invariant_pattern(["B", "A"], [("adjacent", h)])
    assert calls["count"] == 1
    assert isinstance(out, real_cls)


def test_v154_0_scope_guard():
    import qec.analysis.subgraph_invariant_pattern as m

    assert not hasattr(m, "MultiScaleInvariantReceipt")
    assert not hasattr(m, "SierpinskiCompressionReceipt")


def test_validate_receipt_detects_tampered_nested_pattern():
    """Test that validate_subgraph_invariant_pattern_receipt catches tampered pattern data."""
    g = _graph()
    p = build_subgraph_invariant_pattern(["A", "B"], [("adjacent", g.edges[0].constraint_payload_hash)])
    r = build_subgraph_invariant_pattern_receipt(p, match_graph_to_pattern(p, g, 0))

    # Create a tampered pattern with invalid pattern_id but otherwise valid structure
    tampered_pattern = SubgraphInvariantPattern.__new__(SubgraphInvariantPattern)
    object.__setattr__(tampered_pattern, "pattern_id", "f" * 64)  # Invalid pattern_id
    object.__setattr__(tampered_pattern, "node_label_multiset", p.node_label_multiset)
    object.__setattr__(tampered_pattern, "constraint_edge_pairs", p.constraint_edge_pairs)
    object.__setattr__(tampered_pattern, "pattern_hash", p.pattern_hash)

    # Recompute receipt_hash with tampered pattern
    from qec.analysis.subgraph_invariant_pattern import _receipt_hash_payload
    tampered_receipt_hash = sha256_hex(_receipt_hash_payload(tampered_pattern, r.occurrences, r.total_occurrence_count))

    # Create tampered receipt bypassing __post_init__
    tampered_receipt = SubgraphInvariantPatternReceipt.__new__(SubgraphInvariantPatternReceipt)
    object.__setattr__(tampered_receipt, "pattern", tampered_pattern)
    object.__setattr__(tampered_receipt, "occurrences", r.occurrences)
    object.__setattr__(tampered_receipt, "total_occurrence_count", r.total_occurrence_count)
    object.__setattr__(tampered_receipt, "receipt_hash", tampered_receipt_hash)

    # validate should reject due to tampered pattern_id
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_subgraph_invariant_pattern_receipt(tampered_receipt)


def test_validate_receipt_detects_tampered_occurrence_scale_index():
    """Test that validate_subgraph_invariant_pattern_receipt catches tampered occurrence scale_index."""
    g = _graph()
    p = build_subgraph_invariant_pattern(["A", "B"], [("adjacent", g.edges[0].constraint_payload_hash)])
    r = build_subgraph_invariant_pattern_receipt(p, match_graph_to_pattern(p, g, 0))

    # Create a tampered occurrence with invalid scale_index
    original_occ = r.occurrences[0]
    tampered_occ = SubgraphOccurrence.__new__(SubgraphOccurrence)
    object.__setattr__(tampered_occ, "pattern_id", original_occ.pattern_id)
    object.__setattr__(tampered_occ, "scale_index", 99)  # Invalid scale_index
    object.__setattr__(tampered_occ, "source_node_ids", original_occ.source_node_ids)
    object.__setattr__(tampered_occ, "occurrence_hash", original_occ.occurrence_hash)

    # Recompute receipt_hash with tampered occurrence
    from qec.analysis.subgraph_invariant_pattern import _receipt_hash_payload
    tampered_occurrences = (tampered_occ,)
    tampered_receipt_hash = sha256_hex(_receipt_hash_payload(p, tampered_occurrences, 1))

    # Create tampered receipt bypassing __post_init__
    tampered_receipt = SubgraphInvariantPatternReceipt.__new__(SubgraphInvariantPatternReceipt)
    object.__setattr__(tampered_receipt, "pattern", p)
    object.__setattr__(tampered_receipt, "occurrences", tampered_occurrences)
    object.__setattr__(tampered_receipt, "total_occurrence_count", 1)
    object.__setattr__(tampered_receipt, "receipt_hash", tampered_receipt_hash)

    # validate should reject due to invalid scale_index
    with pytest.raises(ValueError, match="INVALID_SCALE_INDEX"):
        validate_subgraph_invariant_pattern_receipt(tampered_receipt)


def test_validate_receipt_detects_tampered_occurrence_hash():
    """Test that validate_subgraph_invariant_pattern_receipt catches corrupted occurrence_hash."""
    g = _graph()
    p = build_subgraph_invariant_pattern(["A", "B"], [("adjacent", g.edges[0].constraint_payload_hash)])
    r = build_subgraph_invariant_pattern_receipt(p, match_graph_to_pattern(p, g, 0))

    # Create a tampered occurrence with corrupted occurrence_hash
    original_occ = r.occurrences[0]
    tampered_occ = SubgraphOccurrence.__new__(SubgraphOccurrence)
    object.__setattr__(tampered_occ, "pattern_id", original_occ.pattern_id)
    object.__setattr__(tampered_occ, "scale_index", original_occ.scale_index)
    object.__setattr__(tampered_occ, "source_node_ids", original_occ.source_node_ids)
    object.__setattr__(tampered_occ, "occurrence_hash", "a" * 64)  # Corrupted hash

    # Recompute receipt_hash with tampered occurrence
    from qec.analysis.subgraph_invariant_pattern import _receipt_hash_payload
    tampered_occurrences = (tampered_occ,)
    tampered_receipt_hash = sha256_hex(_receipt_hash_payload(p, tampered_occurrences, 1))

    # Create tampered receipt bypassing __post_init__
    tampered_receipt = SubgraphInvariantPatternReceipt.__new__(SubgraphInvariantPatternReceipt)
    object.__setattr__(tampered_receipt, "pattern", p)
    object.__setattr__(tampered_receipt, "occurrences", tampered_occurrences)
    object.__setattr__(tampered_receipt, "total_occurrence_count", 1)
    object.__setattr__(tampered_receipt, "receipt_hash", tampered_receipt_hash)

    # validate should reject due to corrupted occurrence_hash
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_subgraph_invariant_pattern_receipt(tampered_receipt)


def test_build_subgraph_invariant_pattern_rejects_non_string_inputs():
    """Test that build_subgraph_invariant_pattern rejects non-string node labels."""
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_subgraph_invariant_pattern([123, "A"], [("x", "a" * 64)])
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_subgraph_invariant_pattern(["A"], [(123, "a" * 64)])
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_subgraph_invariant_pattern(["A"], [("x", 123)])
