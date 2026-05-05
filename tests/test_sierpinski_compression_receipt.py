import pytest

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.multi_scale_invariant_receipt import build_multi_scale_invariant_receipt
from qec.analysis.sierpinski_compression_receipt import (
    SierpinskiCompressionBuildResult,
    SierpinskiCompressionEntry,
    SierpinskiCompressionReceipt,
    build_sierpinski_compression_context,
    build_sierpinski_compression_receipt,
    decompress_sierpinski_receipt,
    validate_sierpinski_compression_receipt,
)
from qec.analysis.subgraph_invariant_pattern import SubgraphOccurrence, build_subgraph_invariant_pattern, build_subgraph_invariant_pattern_receipt


def _occ(pattern_id: str, scale: int, src: tuple[str, ...]) -> SubgraphOccurrence:
    return SubgraphOccurrence(pattern_id, scale, src, sha256_hex({"pattern_id": pattern_id, "scale_index": scale, "source_node_ids": src}))


def test_compression_only_for_n_ge_2():
    p = build_subgraph_invariant_pattern(["A"], [("edge", "a" * 64)])
    pr = build_subgraph_invariant_pattern_receipt(p, [_occ(p.pattern_id, 0, ("n0",)), _occ(p.pattern_id, 0, ("n1",)), _occ(p.pattern_id, 1, ("n2",))])
    ms = build_multi_scale_invariant_receipt(p, list(pr.occurrences))
    r = build_sierpinski_compression_receipt(ms, [pr])
    assert r.total_entries == 1


def test_deterministic_compression_across_runs():
    p = build_subgraph_invariant_pattern(["A"], [("edge", "a" * 64)])
    occ = [_occ(p.pattern_id, 1, ("n2",)), _occ(p.pattern_id, 1, ("n1",)), _occ(p.pattern_id, 1, ("n3",))]
    pr1 = build_subgraph_invariant_pattern_receipt(p, occ)
    pr2 = build_subgraph_invariant_pattern_receipt(p, list(reversed(occ)))
    ms = build_multi_scale_invariant_receipt(p, occ)
    assert build_sierpinski_compression_receipt(ms, [pr1]) == build_sierpinski_compression_receipt(ms, [pr2])


def test_ordering_invariance():
    p1 = build_subgraph_invariant_pattern(["A"], [("edge", "a" * 64)])
    p2 = build_subgraph_invariant_pattern(["B"], [("edge", "b" * 64)])
    pr1 = build_subgraph_invariant_pattern_receipt(p1, [_occ(p1.pattern_id, 2, ("a0",)), _occ(p1.pattern_id, 2, ("a1",))])
    pr2 = build_subgraph_invariant_pattern_receipt(p2, [_occ(p2.pattern_id, 0, ("b0",)), _occ(p2.pattern_id, 0, ("b1",))])
    ms = build_multi_scale_invariant_receipt(p1, list(pr1.occurrences))
    r = build_sierpinski_compression_receipt(ms, [pr1, pr2])
    assert tuple((e.scale_index, e.pattern_hash) for e in r.compression_entries) == tuple(sorted((e.scale_index, e.pattern_hash) for e in r.compression_entries))


def test_invalid_occurrence_count_lt_2_rejection():
    with pytest.raises(ValueError, match="INSUFFICIENT_OCCURRENCES_FOR_COMPRESSION"):
        SierpinskiCompressionEntry("a" * 64, 0, 1, (("n",),), "0" * 64)


def test_tampered_compression_entry_hash_detection():
    p = build_subgraph_invariant_pattern(["A"], [("edge", "a" * 64)])
    pr = build_subgraph_invariant_pattern_receipt(p, [_occ(p.pattern_id, 0, ("n0",)), _occ(p.pattern_id, 0, ("n1",))])
    ms = build_multi_scale_invariant_receipt(p, list(pr.occurrences))
    r = build_sierpinski_compression_receipt(ms, [pr])
    e = r.compression_entries[0]
    bad = SierpinskiCompressionEntry.__new__(SierpinskiCompressionEntry)
    for k, v in e.__dict__.items():
        object.__setattr__(bad, k, v)
    object.__setattr__(bad, "compression_entry_hash", "0" * 64)
    bad_receipt = SierpinskiCompressionReceipt.__new__(SierpinskiCompressionReceipt)
    object.__setattr__(bad_receipt, "multi_scale_invariant_receipt_hash", r.multi_scale_invariant_receipt_hash)
    object.__setattr__(bad_receipt, "compression_entries", (bad,))
    object.__setattr__(bad_receipt, "total_compressed_occurrences", r.total_compressed_occurrences)
    object.__setattr__(bad_receipt, "total_entries", r.total_entries)
    object.__setattr__(bad_receipt, "sierpinski_compression_receipt_hash", r.sierpinski_compression_receipt_hash)
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_sierpinski_compression_receipt(bad_receipt)


def test_tampered_receipt_hash_detection():
    p = build_subgraph_invariant_pattern(["A"], [("edge", "a" * 64)])
    pr = build_subgraph_invariant_pattern_receipt(p, [_occ(p.pattern_id, 0, ("n0",)), _occ(p.pattern_id, 0, ("n1",))])
    ms = build_multi_scale_invariant_receipt(p, list(pr.occurrences))
    r = build_sierpinski_compression_receipt(ms, [pr])
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        SierpinskiCompressionReceipt(r.multi_scale_invariant_receipt_hash, r.compression_entries, r.total_compressed_occurrences, r.total_entries, "0" * 64)


def test_decompression_identity_check_and_no_global_state():
    p = build_subgraph_invariant_pattern(["A"], [("edge", "a" * 64)])
    occ = [_occ(p.pattern_id, 2, ("n2",)), _occ(p.pattern_id, 2, ("n3",))]
    pr = build_subgraph_invariant_pattern_receipt(p, occ)
    ms = build_multi_scale_invariant_receipt(p, occ)
    ctx: SierpinskiCompressionBuildResult = build_sierpinski_compression_context(ms, [pr])
    assert decompress_sierpinski_receipt(ctx.receipt, dict(ctx.pattern_id_map), ctx.original_occurrences) == sorted(occ, key=lambda o: o.occurrence_hash)


def test_decompression_requires_explicit_pattern_id_map_and_original_occurrences():
    p = build_subgraph_invariant_pattern(["A"], [("edge", "a" * 64)])
    occ = [_occ(p.pattern_id, 0, ("n0",)), _occ(p.pattern_id, 0, ("n1",))]
    pr = build_subgraph_invariant_pattern_receipt(p, occ)
    ms = build_multi_scale_invariant_receipt(p, occ)
    ctx = build_sierpinski_compression_context(ms, [pr])
    with pytest.raises(ValueError, match="DECOMPRESSION_IDENTITY_FAILURE"):
        decompress_sierpinski_receipt(ctx.receipt, {}, ctx.original_occurrences)
    with pytest.raises(ValueError, match="DECOMPRESSION_IDENTITY_FAILURE"):
        decompress_sierpinski_receipt(ctx.receipt, dict(ctx.pattern_id_map), tuple())


def test_parent_hash_excludes_child_self_hash():
    p = build_subgraph_invariant_pattern(["A"], [("edge", "a" * 64)])
    occ = [_occ(p.pattern_id, 0, ("n0",)), _occ(p.pattern_id, 0, ("n1",))]
    pr = build_subgraph_invariant_pattern_receipt(p, occ)
    ms = build_multi_scale_invariant_receipt(p, occ)
    r = build_sierpinski_compression_receipt(ms, [pr])
    e = r.compression_entries[0]
    tampered_entry = SierpinskiCompressionEntry.__new__(SierpinskiCompressionEntry)
    object.__setattr__(tampered_entry, "pattern_hash", e.pattern_hash)
    object.__setattr__(tampered_entry, "scale_index", e.scale_index)
    object.__setattr__(tampered_entry, "occurrence_count", e.occurrence_count)
    object.__setattr__(tampered_entry, "source_node_id_sets", e.source_node_id_sets)
    object.__setattr__(tampered_entry, "compression_entry_hash", "f" * 64)
    expected_parent_hash = sha256_hex({"multi_scale_invariant_receipt_hash": r.multi_scale_invariant_receipt_hash, "compression_entries": [{"pattern_hash": e.pattern_hash, "scale_index": e.scale_index, "occurrence_count": e.occurrence_count, "source_node_id_sets": e.source_node_id_sets}], "total_compressed_occurrences": r.total_compressed_occurrences, "total_entries": r.total_entries})
    assert expected_parent_hash == r.sierpinski_compression_receipt_hash
    bad_receipt = SierpinskiCompressionReceipt.__new__(SierpinskiCompressionReceipt)
    object.__setattr__(bad_receipt, "multi_scale_invariant_receipt_hash", r.multi_scale_invariant_receipt_hash)
    object.__setattr__(bad_receipt, "compression_entries", (tampered_entry,))
    object.__setattr__(bad_receipt, "total_compressed_occurrences", r.total_compressed_occurrences)
    object.__setattr__(bad_receipt, "total_entries", r.total_entries)
    object.__setattr__(bad_receipt, "sierpinski_compression_receipt_hash", r.sierpinski_compression_receipt_hash)
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_sierpinski_compression_receipt(bad_receipt)


def test_no_silent_multi_scale_hash_fallback():
    p = build_subgraph_invariant_pattern(["A"], [("edge", "a" * 64)])
    pr = build_subgraph_invariant_pattern_receipt(p, [_occ(p.pattern_id, 0, ("n0",)), _occ(p.pattern_id, 0, ("n1",))])
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_sierpinski_compression_receipt(object(), [pr])


def test_structural_sort_validation_and_occurrence_lt_2_validator_path():
    p = build_subgraph_invariant_pattern(["A"], [("edge", "a" * 64)])
    pr = build_subgraph_invariant_pattern_receipt(p, [_occ(p.pattern_id, 0, ("n0",)), _occ(p.pattern_id, 0, ("n1",))])
    ms = build_multi_scale_invariant_receipt(p, list(pr.occurrences))
    r = build_sierpinski_compression_receipt(ms, [pr])
    e = r.compression_entries[0]
    bad_inner = SierpinskiCompressionEntry.__new__(SierpinskiCompressionEntry)
    object.__setattr__(bad_inner, "pattern_hash", e.pattern_hash)
    object.__setattr__(bad_inner, "scale_index", e.scale_index)
    object.__setattr__(bad_inner, "occurrence_count", e.occurrence_count)
    object.__setattr__(bad_inner, "source_node_id_sets", (("z", "a"), ("n1",)))
    object.__setattr__(bad_inner, "compression_entry_hash", sha256_hex({"pattern_hash": e.pattern_hash, "scale_index": e.scale_index, "occurrence_count": e.occurrence_count, "source_node_id_sets": (("z", "a"), ("n1",))}))
    bad_r1 = SierpinskiCompressionReceipt.__new__(SierpinskiCompressionReceipt)
    object.__setattr__(bad_r1, "multi_scale_invariant_receipt_hash", r.multi_scale_invariant_receipt_hash)
    object.__setattr__(bad_r1, "compression_entries", (bad_inner,))
    object.__setattr__(bad_r1, "total_compressed_occurrences", 2)
    object.__setattr__(bad_r1, "total_entries", 1)
    object.__setattr__(bad_r1, "sierpinski_compression_receipt_hash", r.sierpinski_compression_receipt_hash)
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_sierpinski_compression_receipt(bad_r1)

    bad_count = SierpinskiCompressionEntry.__new__(SierpinskiCompressionEntry)
    object.__setattr__(bad_count, "pattern_hash", e.pattern_hash)
    object.__setattr__(bad_count, "scale_index", e.scale_index)
    object.__setattr__(bad_count, "occurrence_count", 1)
    object.__setattr__(bad_count, "source_node_id_sets", (("n0",),))
    object.__setattr__(bad_count, "compression_entry_hash", sha256_hex({"pattern_hash": e.pattern_hash, "scale_index": e.scale_index, "occurrence_count": 1, "source_node_id_sets": (("n0",),)}))
    bad_r2 = SierpinskiCompressionReceipt.__new__(SierpinskiCompressionReceipt)
    object.__setattr__(bad_r2, "multi_scale_invariant_receipt_hash", r.multi_scale_invariant_receipt_hash)
    object.__setattr__(bad_r2, "compression_entries", (bad_count,))
    object.__setattr__(bad_r2, "total_compressed_occurrences", 1)
    object.__setattr__(bad_r2, "total_entries", 1)
    object.__setattr__(bad_r2, "sierpinski_compression_receipt_hash", sha256_hex({"multi_scale_invariant_receipt_hash": r.multi_scale_invariant_receipt_hash, "compression_entries": [{"pattern_hash": e.pattern_hash, "scale_index": e.scale_index, "occurrence_count": 1, "source_node_id_sets": (("n0",),)}], "total_compressed_occurrences": 1, "total_entries": 1}))
    with pytest.raises(ValueError, match="INSUFFICIENT_OCCURRENCES_FOR_COMPRESSION"):
        validate_sierpinski_compression_receipt(bad_r2)


def test_bool_count_rejection_and_multi_pattern_compression_included_and_empty_and_cross_scale():
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        SierpinskiCompressionEntry("a" * 64, 0, True, (("n",), ("m",)), "0" * 64)

    p1 = build_subgraph_invariant_pattern(["A"], [("edge", "a" * 64)])
    p2 = build_subgraph_invariant_pattern(["B"], [("edge", "b" * 64)])
    o1 = [_occ(p1.pattern_id, 0, ("a0",)), _occ(p1.pattern_id, 0, ("a1",))]
    o2 = [_occ(p2.pattern_id, 1, ("b0",)), _occ(p2.pattern_id, 1, ("b1",))]
    pr1 = build_subgraph_invariant_pattern_receipt(p1, o1)
    pr2 = build_subgraph_invariant_pattern_receipt(p2, o2)
    ms = build_multi_scale_invariant_receipt(p1, o1)
    rr = build_sierpinski_compression_receipt(ms, [pr1, pr2])
    assert rr.total_entries == 2

    singleton = build_subgraph_invariant_pattern_receipt(p1, [_occ(p1.pattern_id, 2, ("solo",))])
    empty = build_sierpinski_compression_receipt(ms, [singleton])
    assert empty.compression_entries == ()

    cross = build_subgraph_invariant_pattern_receipt(p1, [_occ(p1.pattern_id, 0, ("n0",)), _occ(p1.pattern_id, 0, ("n1",)), _occ(p1.pattern_id, 2, ("n2",)), _occ(p1.pattern_id, 2, ("n3",))])
    cross_r = build_sierpinski_compression_receipt(build_multi_scale_invariant_receipt(p1, list(cross.occurrences)), [cross])
    assert tuple((e.scale_index, e.occurrence_count) for e in cross_r.compression_entries) == ((0, 2), (2, 2))

    with pytest.raises(ValueError, match="INVALID_INPUT"):
        SierpinskiCompressionReceipt(rr.multi_scale_invariant_receipt_hash, rr.compression_entries, rr.total_compressed_occurrences, True, rr.sierpinski_compression_receipt_hash)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        SierpinskiCompressionReceipt(rr.multi_scale_invariant_receipt_hash, rr.compression_entries, True, rr.total_entries, rr.sierpinski_compression_receipt_hash)
