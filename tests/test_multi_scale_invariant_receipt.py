import pytest

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.multi_scale_invariant_receipt import (
    MultiScaleInvariantReceipt,
    ScaleLevelSummary,
    build_multi_scale_invariant_receipt,
    build_scale_level_summary,
    validate_multi_scale_invariant_receipt,
)
from qec.analysis.subgraph_invariant_pattern import SubgraphOccurrence, build_subgraph_invariant_pattern


def _pattern_and_occurrences():
    pattern = build_subgraph_invariant_pattern(["A"], [("edge", "a" * 64)])
    occ = [
        SubgraphOccurrence(pattern.pattern_id, 2, ("n2",), sha256_hex({"pattern_id": pattern.pattern_id, "scale_index": 2, "source_node_ids": ("n2",)})),
        SubgraphOccurrence(pattern.pattern_id, 0, ("n0",), sha256_hex({"pattern_id": pattern.pattern_id, "scale_index": 0, "source_node_ids": ("n0",)})),
        SubgraphOccurrence(pattern.pattern_id, 1, ("n1",), sha256_hex({"pattern_id": pattern.pattern_id, "scale_index": 1, "source_node_ids": ("n1",)})),
        SubgraphOccurrence(pattern.pattern_id, 1, ("n1b",), sha256_hex({"pattern_id": pattern.pattern_id, "scale_index": 1, "source_node_ids": ("n1b",)})),
    ]
    return pattern, occ


def test_scale_grouping_correctness():
    pattern, occ = _pattern_and_occurrences()
    receipt = build_multi_scale_invariant_receipt(pattern, occ)
    assert tuple(s.scale_index for s in receipt.scale_summaries) == (0, 1, 2)
    counts = {s.scale_index: s.occurrence_count for s in receipt.scale_summaries}
    assert counts == {0: 1, 1: 2, 2: 1}


def test_deterministic_receipt_across_runs_and_unordered_input():
    pattern, occ = _pattern_and_occurrences()
    r1 = build_multi_scale_invariant_receipt(pattern, occ)
    r2 = build_multi_scale_invariant_receipt(pattern, list(reversed(occ)))
    r3 = build_multi_scale_invariant_receipt(pattern, list(occ))
    assert r1 == r2 == r3


def test_duplicate_scale_index_rejection():
    pattern, occ = _pattern_and_occurrences()
    s0 = build_scale_level_summary(pattern.pattern_id, 0, occ)
    s0_dup = build_scale_level_summary(pattern.pattern_id, 0, occ)
    h = sha256_hex(
        {
            "pattern": {
                "pattern_id": pattern.pattern_id,
                "node_label_multiset": pattern.node_label_multiset,
                "constraint_edge_pairs": [list(p) for p in pattern.constraint_edge_pairs],
                "pattern_hash": pattern.pattern_hash,
            },
            "scale_summaries": [
                {
                    "pattern_id": s0.pattern_id,
                    "scale_index": s0.scale_index,
                    "occurrence_hashes": s0.occurrence_hashes,
                    "occurrence_count": s0.occurrence_count,
                    "scale_hash": s0.scale_hash,
                },
                {
                    "pattern_id": s0_dup.pattern_id,
                    "scale_index": s0_dup.scale_index,
                    "occurrence_hashes": s0_dup.occurrence_hashes,
                    "occurrence_count": s0_dup.occurrence_count,
                    "scale_hash": s0_dup.scale_hash,
                },
            ],
            "total_occurrence_count": 2,
        }
    )
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        MultiScaleInvariantReceipt(pattern, (s0, s0_dup), 2, h)


def test_occurrence_count_mismatch():
    pattern, occ = _pattern_and_occurrences()
    receipt = build_multi_scale_invariant_receipt(pattern, occ)
    with pytest.raises(ValueError, match="OCCURRENCE_COUNT_MISMATCH"):
        MultiScaleInvariantReceipt(receipt.pattern, receipt.scale_summaries, receipt.total_occurrence_count + 1, receipt.receipt_hash)


def test_tampered_scale_hash_detection():
    pattern, occ = _pattern_and_occurrences()
    summary = build_scale_level_summary(pattern.pattern_id, 1, occ)
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        ScaleLevelSummary(summary.pattern_id, summary.scale_index, summary.occurrence_hashes, summary.occurrence_count, "0" * 64)


def test_tampered_receipt_hash_detection():
    pattern, occ = _pattern_and_occurrences()
    receipt = build_multi_scale_invariant_receipt(pattern, occ)
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        MultiScaleInvariantReceipt(receipt.pattern, receipt.scale_summaries, receipt.total_occurrence_count, "0" * 64)


def test_bool_scale_rejection():
    pattern, occ = _pattern_and_occurrences()
    with pytest.raises(ValueError, match="INVALID_SCALE_INDEX"):
        build_scale_level_summary(pattern.pattern_id, True, occ)


def test_empty_occurrences_valid_empty_receipt():
    pattern = build_subgraph_invariant_pattern(["A"], [("edge", "a" * 64)])
    receipt = build_multi_scale_invariant_receipt(pattern, [])
    assert receipt.scale_summaries == ()
    assert receipt.total_occurrence_count == 0
    assert validate_multi_scale_invariant_receipt(receipt) is True


def test_empty_multi_scale_receipt():
    pattern = build_subgraph_invariant_pattern(["A"], [("edge", "a" * 64)])
    receipt = build_multi_scale_invariant_receipt(pattern=pattern, occurrences=[])
    assert receipt.scale_summaries == ()
    assert receipt.total_occurrence_count == 0
    assert validate_multi_scale_invariant_receipt(receipt) is True


def test_cross_scale_aggregation_correctness():
    pattern, occ = _pattern_and_occurrences()
    receipt = build_multi_scale_invariant_receipt(pattern, occ)
    assert receipt.total_occurrence_count == len(occ)
    assert validate_multi_scale_invariant_receipt(receipt)


def test_validate_catches_tampered_nested_summary_hash():
    pattern, occ = _pattern_and_occurrences()
    receipt = build_multi_scale_invariant_receipt(pattern, occ)
    original = receipt.scale_summaries[0]

    tampered_summary = ScaleLevelSummary.__new__(ScaleLevelSummary)
    object.__setattr__(tampered_summary, "pattern_id", original.pattern_id)
    object.__setattr__(tampered_summary, "scale_index", original.scale_index)
    object.__setattr__(tampered_summary, "occurrence_hashes", original.occurrence_hashes)
    object.__setattr__(tampered_summary, "occurrence_count", original.occurrence_count)
    object.__setattr__(tampered_summary, "scale_hash", "0" * 64)

    tampered = MultiScaleInvariantReceipt.__new__(MultiScaleInvariantReceipt)
    object.__setattr__(tampered, "pattern", receipt.pattern)
    object.__setattr__(tampered, "scale_summaries", (tampered_summary,) + receipt.scale_summaries[1:])
    object.__setattr__(tampered, "total_occurrence_count", receipt.total_occurrence_count)
    object.__setattr__(tampered, "receipt_hash", receipt.receipt_hash)

    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_multi_scale_invariant_receipt(tampered)


def test_scale_summary_pattern_id_mismatch():
    pattern, occ = _pattern_and_occurrences()
    wrong_pattern = build_subgraph_invariant_pattern(["B"], [("edge", "b" * 64)])
    summary = build_scale_level_summary(
        pattern_id=wrong_pattern.pattern_id,
        scale_index=0,
        occurrences=occ,
    )

    with pytest.raises(ValueError, match="INVALID_INPUT"):
        MultiScaleInvariantReceipt(
            pattern=pattern,
            scale_summaries=(summary,),
            total_occurrence_count=1,
            receipt_hash="invalid",
        )


def test_bool_occurrence_count_rejection_in_scale_summary():
    """ScaleLevelSummary rejects boolean occurrence_count (bool is subclass of int)."""
    pattern, occ = _pattern_and_occurrences()
    summary = build_scale_level_summary(pattern.pattern_id, 0, occ)
    # Try to create summary with bool occurrence_count bypassing __post_init__ would fail
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        ScaleLevelSummary(
            pattern_id=summary.pattern_id,
            scale_index=summary.scale_index,
            occurrence_hashes=summary.occurrence_hashes,
            occurrence_count=True,
            scale_hash=summary.scale_hash,
        )


def test_bool_total_occurrence_count_rejection_in_receipt():
    """MultiScaleInvariantReceipt rejects boolean total_occurrence_count."""
    pattern, occ = _pattern_and_occurrences()
    receipt = build_multi_scale_invariant_receipt(pattern, occ)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        MultiScaleInvariantReceipt(
            pattern=receipt.pattern,
            scale_summaries=receipt.scale_summaries,
            total_occurrence_count=True,
            receipt_hash=receipt.receipt_hash,
        )


def test_mismatched_pattern_id_in_builder_rejected():
    """build_multi_scale_invariant_receipt rejects occurrences with mismatched pattern_id."""
    pattern, occ = _pattern_and_occurrences()
    other_pattern = build_subgraph_invariant_pattern(["B"], [("edge", "b" * 64)])

    # Create an occurrence with a different pattern_id
    foreign_occ = SubgraphOccurrence(
        other_pattern.pattern_id, 0, ("n_other",),
        sha256_hex({"pattern_id": other_pattern.pattern_id, "scale_index": 0, "source_node_ids": ("n_other",)})
    )

    mixed_occurrences = occ + [foreign_occ]
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_multi_scale_invariant_receipt(pattern, mixed_occurrences)


def test_validate_catches_tampered_pattern_id():
    """validate_multi_scale_invariant_receipt catches tampered pattern.pattern_id."""
    pattern, occ = _pattern_and_occurrences()
    receipt = build_multi_scale_invariant_receipt(pattern, occ)

    # Tamper pattern_id via __new__
    tampered_pattern = type(receipt.pattern).__new__(type(receipt.pattern))
    object.__setattr__(tampered_pattern, "pattern_id", "0" * 64)
    object.__setattr__(tampered_pattern, "node_label_multiset", receipt.pattern.node_label_multiset)
    object.__setattr__(tampered_pattern, "constraint_edge_pairs", receipt.pattern.constraint_edge_pairs)
    object.__setattr__(tampered_pattern, "pattern_hash", receipt.pattern.pattern_hash)

    tampered_receipt = MultiScaleInvariantReceipt.__new__(MultiScaleInvariantReceipt)
    object.__setattr__(tampered_receipt, "pattern", tampered_pattern)
    object.__setattr__(tampered_receipt, "scale_summaries", receipt.scale_summaries)
    object.__setattr__(tampered_receipt, "total_occurrence_count", receipt.total_occurrence_count)
    object.__setattr__(tampered_receipt, "receipt_hash", receipt.receipt_hash)

    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_multi_scale_invariant_receipt(tampered_receipt)


def test_validate_catches_tampered_pattern_hash():
    """validate_multi_scale_invariant_receipt catches tampered pattern.pattern_hash."""
    pattern, occ = _pattern_and_occurrences()
    receipt = build_multi_scale_invariant_receipt(pattern, occ)

    tampered_pattern = type(receipt.pattern).__new__(type(receipt.pattern))
    object.__setattr__(tampered_pattern, "pattern_id", receipt.pattern.pattern_id)
    object.__setattr__(tampered_pattern, "node_label_multiset", receipt.pattern.node_label_multiset)
    object.__setattr__(tampered_pattern, "constraint_edge_pairs", receipt.pattern.constraint_edge_pairs)
    object.__setattr__(tampered_pattern, "pattern_hash", "0" * 64)

    tampered_receipt = MultiScaleInvariantReceipt.__new__(MultiScaleInvariantReceipt)
    object.__setattr__(tampered_receipt, "pattern", tampered_pattern)
    object.__setattr__(tampered_receipt, "scale_summaries", receipt.scale_summaries)
    object.__setattr__(tampered_receipt, "total_occurrence_count", receipt.total_occurrence_count)
    object.__setattr__(tampered_receipt, "receipt_hash", receipt.receipt_hash)

    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_multi_scale_invariant_receipt(tampered_receipt)


def test_validate_catches_tampered_scale_index_type():
    """validate_multi_scale_invariant_receipt catches non-int or bool scale_index."""
    pattern, occ = _pattern_and_occurrences()
    receipt = build_multi_scale_invariant_receipt(pattern, occ)
    original_summary = receipt.scale_summaries[0]

    # Tamper scale_index to a boolean
    tampered_summary = ScaleLevelSummary.__new__(ScaleLevelSummary)
    object.__setattr__(tampered_summary, "pattern_id", original_summary.pattern_id)
    object.__setattr__(tampered_summary, "scale_index", True)  # bool is subclass of int
    object.__setattr__(tampered_summary, "occurrence_hashes", original_summary.occurrence_hashes)
    object.__setattr__(tampered_summary, "occurrence_count", original_summary.occurrence_count)
    object.__setattr__(tampered_summary, "scale_hash", original_summary.scale_hash)

    tampered_receipt = MultiScaleInvariantReceipt.__new__(MultiScaleInvariantReceipt)
    object.__setattr__(tampered_receipt, "pattern", receipt.pattern)
    object.__setattr__(tampered_receipt, "scale_summaries", (tampered_summary,) + receipt.scale_summaries[1:])
    object.__setattr__(tampered_receipt, "total_occurrence_count", receipt.total_occurrence_count)
    object.__setattr__(tampered_receipt, "receipt_hash", receipt.receipt_hash)

    with pytest.raises(ValueError, match="INVALID_SCALE_INDEX"):
        validate_multi_scale_invariant_receipt(tampered_receipt)


def test_validate_catches_unsorted_occurrence_hashes():
    """validate_multi_scale_invariant_receipt catches unsorted occurrence_hashes in summaries."""
    pattern, occ = _pattern_and_occurrences()
    receipt = build_multi_scale_invariant_receipt(pattern, occ)
    original_summary = receipt.scale_summaries[1]  # scale 1 has 2 hashes

    # Tamper occurrence_hashes to be unsorted (reverse them)
    if len(original_summary.occurrence_hashes) < 2:
        pytest.skip("Need at least 2 hashes to test sorting")

    unsorted_hashes = tuple(reversed(original_summary.occurrence_hashes))
    if unsorted_hashes == original_summary.occurrence_hashes:
        pytest.skip("Hashes are already the same reversed")

    tampered_summary = ScaleLevelSummary.__new__(ScaleLevelSummary)
    object.__setattr__(tampered_summary, "pattern_id", original_summary.pattern_id)
    object.__setattr__(tampered_summary, "scale_index", original_summary.scale_index)
    object.__setattr__(tampered_summary, "occurrence_hashes", unsorted_hashes)
    object.__setattr__(tampered_summary, "occurrence_count", original_summary.occurrence_count)
    object.__setattr__(tampered_summary, "scale_hash", original_summary.scale_hash)

    # Replace the summary at scale 1
    new_summaries = tuple(
        tampered_summary if s.scale_index == 1 else s for s in receipt.scale_summaries
    )

    tampered_receipt = MultiScaleInvariantReceipt.__new__(MultiScaleInvariantReceipt)
    object.__setattr__(tampered_receipt, "pattern", receipt.pattern)
    object.__setattr__(tampered_receipt, "scale_summaries", new_summaries)
    object.__setattr__(tampered_receipt, "total_occurrence_count", receipt.total_occurrence_count)
    object.__setattr__(tampered_receipt, "receipt_hash", receipt.receipt_hash)

    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_multi_scale_invariant_receipt(tampered_receipt)


def test_validate_catches_occurrence_count_mismatch_in_summary():
    """validate_multi_scale_invariant_receipt catches summary with mismatched occurrence_count."""
    pattern, occ = _pattern_and_occurrences()
    receipt = build_multi_scale_invariant_receipt(pattern, occ)
    original_summary = receipt.scale_summaries[0]

    # Tamper occurrence_count to not match len(occurrence_hashes)
    tampered_summary = ScaleLevelSummary.__new__(ScaleLevelSummary)
    object.__setattr__(tampered_summary, "pattern_id", original_summary.pattern_id)
    object.__setattr__(tampered_summary, "scale_index", original_summary.scale_index)
    object.__setattr__(tampered_summary, "occurrence_hashes", original_summary.occurrence_hashes)
    object.__setattr__(tampered_summary, "occurrence_count", original_summary.occurrence_count + 99)
    object.__setattr__(tampered_summary, "scale_hash", original_summary.scale_hash)

    tampered_receipt = MultiScaleInvariantReceipt.__new__(MultiScaleInvariantReceipt)
    object.__setattr__(tampered_receipt, "pattern", receipt.pattern)
    object.__setattr__(tampered_receipt, "scale_summaries", (tampered_summary,) + receipt.scale_summaries[1:])
    # Adjust total_occurrence_count to match the sum of tampered summaries
    new_total = tampered_summary.occurrence_count + sum(s.occurrence_count for s in receipt.scale_summaries[1:])
    object.__setattr__(tampered_receipt, "total_occurrence_count", new_total)
    object.__setattr__(tampered_receipt, "receipt_hash", receipt.receipt_hash)

    with pytest.raises(ValueError, match="OCCURRENCE_COUNT_MISMATCH"):
        validate_multi_scale_invariant_receipt(tampered_receipt)


def test_build_scale_level_summary_filters_by_pattern_id():
    """build_scale_level_summary only includes occurrences matching both scale_index and pattern_id."""
    pattern, occ = _pattern_and_occurrences()
    other_pattern = build_subgraph_invariant_pattern(["B"], [("edge", "b" * 64)])

    # Create an occurrence with a different pattern_id at scale 0
    foreign_occ = SubgraphOccurrence(
        other_pattern.pattern_id, 0, ("n_other",),
        sha256_hex({"pattern_id": other_pattern.pattern_id, "scale_index": 0, "source_node_ids": ("n_other",)})
    )

    # Build summary for pattern.pattern_id at scale 0 from mixed list
    mixed = occ + [foreign_occ]
    summary = build_scale_level_summary(pattern.pattern_id, 0, mixed)

    # Only the occurrence matching pattern.pattern_id and scale_index=0 should be included
    expected_count = sum(1 for o in occ if o.scale_index == 0 and o.pattern_id == pattern.pattern_id)
    assert summary.occurrence_count == expected_count
    assert summary.occurrence_count == 1  # From _pattern_and_occurrences, scale 0 has 1 occurrence
