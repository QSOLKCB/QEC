from __future__ import annotations

import pytest

from qec.analysis.episodic_memory_lifting import lift_raw_records_to_episodic_memory
from qec.analysis.hash_preserving_memory_compression import (
    DETERMINISTIC_MEMORY_COMPRESSION_RULE,
    HASH_PRESERVING_COMPRESSION_LAW,
    REPLAY_SAFE_COMPRESSION_CHAIN_INVARIANT,
    compress_semantic_theme_memory,
    export_compressed_memory_bytes,
    generate_compression_receipt,
)
from qec.analysis.semantic_theme_compaction import compact_episodic_memory_to_semantic_themes


def _records() -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for i in range(60):
        rows.append(
            {
                "record_id": f"topic:alpha-{i:03d}" if i % 3 == 0 else f"ops-{i:03d}",
                "sequence_index": i,
                "source_id": "s1" if i < 30 else "s2",
                "provenance_id": "p1",
                "state_token": "A" if i % 4 else "B",
                "task_completed": bool(i % 7 == 0),
                "is_reset": bool(i % 11 == 0 and i > 0),
            }
        )
    return tuple(rows)


def _semantic_artifact():
    episodic = lift_raw_records_to_episodic_memory(_records())
    return compact_episodic_memory_to_semantic_themes(episodic)


def test_deterministic_compression_output_and_hash_stability() -> None:
    semantic = _semantic_artifact()
    a = compress_semantic_theme_memory(semantic)
    b = compress_semantic_theme_memory(semantic)

    assert a == b
    assert a.compression_hash == b.compression_hash
    assert a.to_canonical_bytes() == b.to_canonical_bytes()
    assert a.stable_hash() == b.stable_hash()


def test_compression_bounds_and_lineage_preservation() -> None:
    semantic = _semantic_artifact()
    compressed = compress_semantic_theme_memory(semantic)

    assert 0.0 <= compressed.compression_ratio <= 1.0
    assert compressed.source_artifact_hash == semantic.artifact_hash
    assert compressed.preserved_theme_hashes == tuple(theme.theme_hash for theme in semantic.themes)
    assert compressed.compression_chain_head == semantic.themes[-1].replay_identity_hash
    assert compressed.law_invariants == (
        HASH_PRESERVING_COMPRESSION_LAW,
        DETERMINISTIC_MEMORY_COMPRESSION_RULE,
        REPLAY_SAFE_COMPRESSION_CHAIN_INVARIANT,
    )


def test_repeated_run_byte_identity_stress() -> None:
    semantic = _semantic_artifact()
    blobs = [export_compressed_memory_bytes(compress_semantic_theme_memory(semantic)) for _ in range(40)]
    assert len(set(blobs)) == 1


def test_receipt_stability() -> None:
    compressed = compress_semantic_theme_memory(_semantic_artifact())
    receipt_a = generate_compression_receipt(compressed)
    receipt_b = generate_compression_receipt(compressed)
    assert receipt_a.receipt_hash == receipt_b.receipt_hash
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_fail_fast_malformed_input() -> None:
    with pytest.raises(ValueError, match="artifact must be a SemanticThemeArtifact"):
        compress_semantic_theme_memory(object())

    semantic = _semantic_artifact()
    bad = type(semantic)(
        schema_version=semantic.schema_version,
        source_artifact_hash=semantic.source_artifact_hash,
        episode_count=semantic.episode_count,
        theme_count=semantic.theme_count,
        theme_ids=("only-one",),
        themes=semantic.themes,
        episode_to_theme=semantic.episode_to_theme,
        compaction_ratio=semantic.compaction_ratio,
        law_invariants=semantic.law_invariants,
        assignment_precedence=semantic.assignment_precedence,
        artifact_hash=semantic.artifact_hash,
    )
    with pytest.raises(ValueError, match="theme_count must match theme_ids length"):
        compress_semantic_theme_memory(bad)
