from __future__ import annotations

import pytest

from qec.analysis.episodic_memory_lifting import lift_raw_records_to_episodic_memory
from qec.analysis.semantic_theme_compaction import (
    ThemeCompactionConfig,
    compact_episodic_memory_to_semantic_themes,
    export_semantic_theme_bytes,
    generate_semantic_theme_receipt,
)


def _records() -> tuple[dict[str, object], ...]:
    return (
        {"record_id": "nav-000", "sequence_index": 0, "source_id": "s1", "provenance_id": "p1", "state_token": "A"},
        {"record_id": "nav-001", "sequence_index": 1, "source_id": "s1", "provenance_id": "p1", "state_token": "A", "task_completed": True},
        {"record_id": "nav-002", "sequence_index": 2, "source_id": "s1", "provenance_id": "p1", "state_token": "A"},
        {"record_id": "ops-000", "sequence_index": 3, "source_id": "s1", "provenance_id": "p1", "state_token": "A", "is_reset": True},
        {"record_id": "ops-001", "sequence_index": 4, "source_id": "s1", "provenance_id": "p1", "state_token": "A"},
        {"record_id": "ops-002", "sequence_index": 5, "source_id": "s1", "provenance_id": "p1", "state_token": "A", "task_completed": True},
        {"record_id": "ops-003", "sequence_index": 6, "source_id": "s1", "provenance_id": "p1", "state_token": "A"},
        {"record_id": "topic:alpha-000", "sequence_index": 7, "source_id": "s1", "provenance_id": "p1", "state_token": "A", "is_reset": True},
        {"record_id": "topic:alpha-001", "sequence_index": 8, "source_id": "s1", "provenance_id": "p1", "state_token": "A"},
    )


def _episodic_artifact():
    return lift_raw_records_to_episodic_memory(_records(), config=None)


def test_deterministic_theme_grouping() -> None:
    episodic = _episodic_artifact()
    a = compact_episodic_memory_to_semantic_themes(episodic)
    b = compact_episodic_memory_to_semantic_themes(episodic)

    assert a == b
    assert a.to_canonical_bytes() == b.to_canonical_bytes()


def test_stable_theme_ordering() -> None:
    episodic = _episodic_artifact()
    artifact = compact_episodic_memory_to_semantic_themes(episodic)
    assert artifact.theme_ids == tuple(sorted(artifact.theme_ids))
    assert tuple(theme.theme_index for theme in artifact.themes) == tuple(range(artifact.theme_count))


def test_exact_symbolic_motif_grouping_by_parent_chain() -> None:
    episodic = _episodic_artifact()
    artifact = compact_episodic_memory_to_semantic_themes(episodic)
    assert artifact.theme_count < episodic.episode_count
    assert any(theme.compaction_reasons == ("task_completion",) for theme in artifact.themes)
    assert any(theme.compaction_reasons == ("boundary_signature",) for theme in artifact.themes)


def test_provenance_continuity_grouping_by_parent_chain() -> None:
    episodic = _episodic_artifact()
    altered_eps = tuple(
        type(ep)(
            episode_id=ep.episode_id,
            episode_index=ep.episode_index,
            record_ids=ep.record_ids,
            start_sequence_index=ep.start_sequence_index,
            end_sequence_index=ep.end_sequence_index,
            boundary_reasons=(),
            parent_episode_hash="chain-group",
            episode_hash=ep.episode_hash,
            replay_identity_hash=ep.replay_identity_hash,
        )
        for ep in episodic.episodes
    )
    altered = type(episodic)(
        schema_version=episodic.schema_version,
        source_sequence_hash=episodic.source_sequence_hash,
        total_records=episodic.total_records,
        episode_count=episodic.episode_count,
        episode_ids=episodic.episode_ids,
        episodes=altered_eps,
        law_invariants=episodic.law_invariants,
        artifact_hash=episodic.artifact_hash,
    )
    artifact = compact_episodic_memory_to_semantic_themes(altered)
    assert artifact.theme_count == 1
    assert artifact.themes[0].compaction_reasons == ("parent_chain",)


def test_lexical_prefix_grouping() -> None:
    episodic = _episodic_artifact()
    base = episodic.episodes
    altered_eps = (
        type(base[0])(
            episode_id=base[0].episode_id,
            episode_index=base[0].episode_index,
            record_ids=base[0].record_ids,
            start_sequence_index=base[0].start_sequence_index,
            end_sequence_index=base[0].end_sequence_index,
            boundary_reasons=(),
            parent_episode_hash="parent-0",
            episode_hash=base[0].episode_hash,
            replay_identity_hash=base[0].replay_identity_hash,
        ),
        type(base[1])(
            episode_id=base[1].episode_id,
            episode_index=base[1].episode_index,
            record_ids=base[1].record_ids,
            start_sequence_index=base[1].start_sequence_index,
            end_sequence_index=base[1].end_sequence_index,
            boundary_reasons=(),
            parent_episode_hash="parent-1",
            episode_hash=base[1].episode_hash,
            replay_identity_hash=base[1].replay_identity_hash,
        ),
    )
    altered = type(episodic)(
        schema_version=episodic.schema_version,
        source_sequence_hash=episodic.source_sequence_hash,
        total_records=episodic.total_records,
        episode_count=2,
        episode_ids=(base[0].episode_id, base[1].episode_id),
        episodes=altered_eps,
        law_invariants=episodic.law_invariants,
        artifact_hash=episodic.artifact_hash,
    )
    artifact = compact_episodic_memory_to_semantic_themes(altered)
    assert artifact.theme_count == 1
    assert artifact.themes[0].compaction_reasons == ("record_prefix",)


def test_canonical_export_stability() -> None:
    artifact = compact_episodic_memory_to_semantic_themes(_episodic_artifact())
    assert artifact.to_canonical_json() == artifact.to_canonical_json()


def test_stable_receipt_hash() -> None:
    artifact = compact_episodic_memory_to_semantic_themes(_episodic_artifact())
    receipt_a = generate_semantic_theme_receipt(artifact)
    receipt_b = generate_semantic_theme_receipt(artifact)
    assert receipt_a.receipt_hash == receipt_b.receipt_hash
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_repeated_run_byte_identity_stress() -> None:
    episodic = _episodic_artifact()
    blobs = [export_semantic_theme_bytes(compact_episodic_memory_to_semantic_themes(episodic)) for _ in range(25)]
    assert len(set(blobs)) == 1


def test_fail_fast_malformed_input() -> None:
    with pytest.raises(ValueError, match="artifact must be an EpisodicMemoryArtifact"):
        compact_episodic_memory_to_semantic_themes(object())

    episodic = _episodic_artifact()
    bad = type(episodic)(
        schema_version=episodic.schema_version,
        source_sequence_hash=episodic.source_sequence_hash,
        total_records=episodic.total_records,
        episode_count=episodic.episode_count,
        episode_ids=("wrong",),
        episodes=episodic.episodes,
        law_invariants=episodic.law_invariants,
        artifact_hash=episodic.artifact_hash,
    )
    with pytest.raises(ValueError, match="episode_count must match episode_ids length"):
        compact_episodic_memory_to_semantic_themes(bad)


def test_adversarial_episode_ordering_normalization() -> None:
    episodic = _episodic_artifact()
    reversed_eps = tuple(reversed(episodic.episodes))
    reversed_ids = tuple(ep.episode_id for ep in reversed_eps)
    shuffled = type(episodic)(
        schema_version=episodic.schema_version,
        source_sequence_hash=episodic.source_sequence_hash,
        total_records=episodic.total_records,
        episode_count=episodic.episode_count,
        episode_ids=reversed_ids,
        episodes=reversed_eps,
        law_invariants=episodic.law_invariants,
        artifact_hash=episodic.artifact_hash,
    )

    normalized = compact_episodic_memory_to_semantic_themes(
        shuffled,
        config=ThemeCompactionConfig(normalize_episode_order=True),
    )
    baseline = compact_episodic_memory_to_semantic_themes(_episodic_artifact())

    assert normalized.to_canonical_bytes() == baseline.to_canonical_bytes()
