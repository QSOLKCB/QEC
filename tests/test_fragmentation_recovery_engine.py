from __future__ import annotations

import pytest

from qec.analysis.episodic_memory_lifting import lift_raw_records_to_episodic_memory
from qec.analysis.fragmentation_recovery_engine import (
    BOUNDED_FRAGMENT_RECOVERY_SCORE,
    DETERMINISTIC_CHAIN_REPAIR_RULE,
    FRAGMENTATION_RECOVERY_LAW,
    REPLAY_SAFE_RECOVERY_CHAIN,
    export_fragmentation_recovery_bytes,
    generate_fragmentation_recovery_receipt,
    recover_fragmented_compression_chain,
)
from qec.analysis.hash_preserving_memory_compression import compress_semantic_theme_memory
from qec.analysis.semantic_memory_sonification import project_compressed_memory_to_sonification
from qec.analysis.semantic_theme_compaction import compact_episodic_memory_to_semantic_themes


def _records() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "record_id": f"theme:delta-{i:03d}" if i % 7 == 0 else f"ops-{i:03d}",
            "sequence_index": i,
            "source_id": "src-a" if i < 40 else "src-b",
            "provenance_id": "prov",
            "state_token": "S" if i % 2 == 0 else "T",
            "task_completed": bool(i % 9 == 0),
            "is_reset": bool(i > 0 and i % 13 == 0),
        }
        for i in range(70)
    )


def _compressed_artifact():
    episodic = lift_raw_records_to_episodic_memory(_records())
    semantic = compact_episodic_memory_to_semantic_themes(episodic)
    return compress_semantic_theme_memory(semantic)


def test_identical_input_is_byte_identical_and_deterministic() -> None:
    compressed = _compressed_artifact()
    observed = tuple(compressed.records)

    a = recover_fragmented_compression_chain(
        compressed,
        observed_records=observed,
        enable_fragmentation_recovery=True,
    )
    b = recover_fragmented_compression_chain(
        compressed,
        observed_records=observed,
        enable_fragmentation_recovery=True,
    )

    assert a == b
    assert a.to_canonical_bytes() == b.to_canonical_bytes()
    assert export_fragmentation_recovery_bytes(a) == export_fragmentation_recovery_bytes(b)


def test_deterministic_fracture_detection_stable_tie_breaking_and_partial_repair() -> None:
    compressed = _compressed_artifact()
    observed = (
        compressed.records[3],
        compressed.records[1],
        compressed.records[1],
        compressed.records[4],
        compressed.records[0],
    )

    repaired_a = recover_fragmented_compression_chain(
        compressed,
        observed_records=observed,
        enable_fragmentation_recovery=True,
    )
    repaired_b = recover_fragmented_compression_chain(
        compressed,
        observed_records=observed,
        enable_fragmentation_recovery=True,
    )

    assert repaired_a.lineage_report.disordered_theme_indices == repaired_b.lineage_report.disordered_theme_indices
    assert repaired_a.lineage_report.missing_theme_indices == repaired_b.lineage_report.missing_theme_indices
    assert repaired_a.lineage_report.classification == "partial_fragment_repair"
    assert repaired_a.repaired_records == compressed.records


def test_bounded_scores_and_invariants() -> None:
    compressed = _compressed_artifact()
    observed = (compressed.records[0],)
    repaired = recover_fragmented_compression_chain(
        compressed,
        observed_records=observed,
        enable_fragmentation_recovery=True,
    )

    assert 0.0 <= repaired.continuity_score <= 1.0
    assert 0.0 <= repaired.recovery_confidence <= 1.0
    assert 0.0 <= repaired.fracture_severity <= 1.0
    assert repaired.law_invariants == (
        FRAGMENTATION_RECOVERY_LAW,
        DETERMINISTIC_CHAIN_REPAIR_RULE,
        REPLAY_SAFE_RECOVERY_CHAIN,
        BOUNDED_FRAGMENT_RECOVERY_SCORE,
    )


def test_fail_fast_on_malformed_or_invalid_inputs() -> None:
    compressed = _compressed_artifact()
    with pytest.raises(ValueError, match="enable_fragmentation_recovery must be explicitly True"):
        recover_fragmented_compression_chain(compressed)

    bad_record = type(compressed.records[0])(
        theme_id=compressed.records[0].theme_id,
        theme_index=compressed.records[0].theme_index,
        source_theme_hash="not-hex",
        source_replay_identity_hash=compressed.records[0].source_replay_identity_hash,
        source_parent_theme_hash=compressed.records[0].source_parent_theme_hash,
        signature_ref=compressed.records[0].signature_ref,
        reason_ref=compressed.records[0].reason_ref,
        episode_hashes_ref=compressed.records[0].episode_hashes_ref,
        compression_record_hash=compressed.records[0].compression_record_hash,
    )
    with pytest.raises(ValueError, match="source_theme_hash"):
        recover_fragmented_compression_chain(
            compressed,
            observed_records=(bad_record,),
            enable_fragmentation_recovery=True,
        )


def test_preserves_valid_compression_hashes_and_noop_behavior() -> None:
    compressed = _compressed_artifact()
    repaired = recover_fragmented_compression_chain(
        compressed,
        observed_records=tuple(compressed.records),
        enable_fragmentation_recovery=True,
    )

    assert repaired.lineage_report.classification == "no_op"
    assert repaired.lineage_report.preserved_hash_count == len(compressed.records)
    assert tuple(r.compression_record_hash for r in repaired.repaired_records) == tuple(
        r.compression_record_hash for r in compressed.records
    )


def test_preserves_sonification_lineage_when_present() -> None:
    compressed = _compressed_artifact()
    sonification = project_compressed_memory_to_sonification(compressed)
    observed = (
        compressed.records[2],
        compressed.records[0],
    )

    repaired = recover_fragmented_compression_chain(
        compressed,
        observed_records=observed,
        sonification_spec=sonification,
        enable_fragmentation_recovery=True,
    )

    assert repaired.sonification_repair_metadata is not None
    assert repaired.sonification_repair_metadata.source_sonification_spec_hash == sonification.sonification_spec_hash
    assert repaired.sonification_repair_metadata.preserved_audio_projection_hash == sonification.audio_projection_hash


def test_replay_safe_receipt_generation() -> None:
    compressed = _compressed_artifact()
    repaired = recover_fragmented_compression_chain(
        compressed,
        observed_records=(compressed.records[0], compressed.records[-1]),
        enable_fragmentation_recovery=True,
    )
    receipt_a = generate_fragmentation_recovery_receipt(repaired)
    receipt_b = generate_fragmentation_recovery_receipt(repaired)

    assert receipt_a.receipt_hash == receipt_b.receipt_hash
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_unrecoverable_fragmentation_classification() -> None:
    compressed = _compressed_artifact()
    wrong = type(compressed.records[0])(
        theme_id=compressed.records[0].theme_id,
        theme_index=0,
        source_theme_hash=compressed.records[1].source_theme_hash,
        source_replay_identity_hash=compressed.records[1].source_replay_identity_hash,
        source_parent_theme_hash=compressed.records[1].source_parent_theme_hash,
        signature_ref=compressed.records[1].signature_ref,
        reason_ref=compressed.records[1].reason_ref,
        episode_hashes_ref=compressed.records[1].episode_hashes_ref,
        compression_record_hash=compressed.records[1].compression_record_hash,
    )

    repaired = recover_fragmented_compression_chain(
        compressed,
        observed_records=(wrong,),
        enable_fragmentation_recovery=True,
    )
    assert repaired.lineage_report.classification == "unrecoverable_corruption"
    assert repaired.recovery_confidence == 0.0
