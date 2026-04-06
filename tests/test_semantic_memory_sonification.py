from __future__ import annotations

import pytest

from qec.analysis.episodic_memory_lifting import lift_raw_records_to_episodic_memory
from qec.analysis.hash_preserving_memory_compression import compress_semantic_theme_memory
from qec.analysis.semantic_memory_sonification import (
    DETERMINISTIC_SONIFICATION_PROJECTION_RULE,
    export_mp3_projection_manifest,
    export_sonification_spec_bytes,
    generate_sonification_receipt,
    project_compressed_memory_to_sonification,
)
from qec.analysis.semantic_theme_compaction import compact_episodic_memory_to_semantic_themes


def _records() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "record_id": f"theme:gamma-{i:03d}" if i % 5 == 0 else f"nav-{i:03d}",
            "sequence_index": i,
            "source_id": "src",
            "provenance_id": "prov",
            "state_token": "X" if i % 3 else "Y",
            "task_completed": bool(i % 6 == 0),
            "is_reset": bool(i % 10 == 0 and i > 0),
        }
        for i in range(45)
    )


def _compressed_artifact():
    episodic = lift_raw_records_to_episodic_memory(_records())
    semantic = compact_episodic_memory_to_semantic_themes(episodic)
    return compress_semantic_theme_memory(semantic)


def test_deterministic_sonification_spec_and_hash() -> None:
    compressed = _compressed_artifact()
    spec_a = project_compressed_memory_to_sonification(compressed)
    spec_b = project_compressed_memory_to_sonification(compressed)

    assert spec_a == spec_b
    assert spec_a.sonification_spec_hash == spec_b.sonification_spec_hash
    assert spec_a.to_canonical_bytes() == spec_b.to_canonical_bytes()
    assert spec_a.law_invariants == (DETERMINISTIC_SONIFICATION_PROJECTION_RULE,)


def test_repeated_run_audio_spec_identity_stress() -> None:
    compressed = _compressed_artifact()
    blobs = [export_sonification_spec_bytes(project_compressed_memory_to_sonification(compressed)) for _ in range(30)]
    assert len(set(blobs)) == 1


def test_receipt_stability_and_event_shape() -> None:
    compressed = _compressed_artifact()
    spec = project_compressed_memory_to_sonification(compressed)
    receipt_a = generate_sonification_receipt(spec)
    receipt_b = generate_sonification_receipt(spec)

    assert receipt_a.receipt_hash == receipt_b.receipt_hash
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()
    assert all(len(event) == 5 for event in spec.event_sequence)


def test_mp3_manifest_is_non_authoritative_convenience_only() -> None:
    compressed = _compressed_artifact()
    spec = project_compressed_memory_to_sonification(compressed)
    canonical_before = spec.to_canonical_bytes()

    mp3_manifest = export_mp3_projection_manifest(spec, include_mp3_manifest=True)
    assert b'"authoritative":false' in mp3_manifest
    assert spec.to_canonical_bytes() == canonical_before


def test_fail_fast_malformed_input() -> None:
    with pytest.raises(ValueError, match="artifact must be a CompressedMemoryArtifact"):
        project_compressed_memory_to_sonification(object())

    compressed = _compressed_artifact()
    with pytest.raises(ValueError, match="sample_rate_hz must be fixed at 48000"):
        project_compressed_memory_to_sonification(compressed, sample_rate_hz=44_100)
    with pytest.raises(ValueError, match="include_mp3_manifest must be explicitly True"):
        export_mp3_projection_manifest(project_compressed_memory_to_sonification(compressed))
