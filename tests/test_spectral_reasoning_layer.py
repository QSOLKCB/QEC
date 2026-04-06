from __future__ import annotations

from dataclasses import replace

import pytest

from qec.analysis.arithmetic_topology_correspondence_engine import build_arithmetic_topology_correspondence
from qec.analysis.e8_symmetry_projection_layer import build_e8_symmetry_projection
from qec.analysis.episodic_memory_lifting import lift_raw_records_to_episodic_memory
from qec.analysis.fragmentation_recovery_engine import recover_fragmented_compression_chain
from qec.analysis.hash_preserving_memory_compression import CompressedMemoryArtifact, compress_semantic_theme_memory
from qec.analysis.manifold_traversal_planner import build_manifold_traversal_plan
from qec.analysis.multimodal_feature_schema import build_multimodal_feature_schema
from qec.analysis.polytope_reasoning_engine import build_polytope_reasoning_engine
from qec.analysis.semantic_theme_compaction import compact_episodic_memory_to_semantic_themes
from qec.analysis.spectral_reasoning_layer import (
    BOUNDED_SPECTRAL_SCORE_RULE,
    DETERMINISTIC_BAND_ORDERING_RULE,
    REPLAY_SAFE_SPECTRAL_IDENTITY_RULE,
    SPECTRAL_REASONING_LAYER_LAW,
    build_spectral_reasoning_layer,
    export_spectral_reasoning_bytes,
    generate_spectral_reasoning_receipt,
)
from qec.analysis.topological_graph_kernel import build_topological_graph_kernel
from qec.analysis.topology_divergence_battery import run_topology_divergence_battery


def _records() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "record_id": f"theme:spectral-{i:03d}" if i % 2 == 0 else f"spectral-{i:03d}",
            "sequence_index": i,
            "source_id": "src-a" if i < 12 else "src-b",
            "provenance_id": "prov-a" if i % 3 == 0 else "prov-b",
            "state_token": "A" if i % 2 == 0 else "B",
            "task_completed": bool(i % 4 == 0),
            "is_reset": bool(i > 0 and i % 11 == 0),
        }
        for i in range(24)
    )


def _compressed_artifact() -> CompressedMemoryArtifact:
    episodic = lift_raw_records_to_episodic_memory(_records())
    semantic = compact_episodic_memory_to_semantic_themes(episodic)
    return compress_semantic_theme_memory(semantic)


def _schema_artifact() -> object:
    compressed = _compressed_artifact()
    seed = tuple(range(min(6, len(compressed.records))))
    observed = tuple(compressed.records[idx] for idx in seed)
    recovery = recover_fragmented_compression_chain(
        compressed,
        observed_records=observed,
        enable_fragmentation_recovery=True,
    )
    graph = build_topological_graph_kernel(compressed, recovery)
    polytope = build_polytope_reasoning_engine(graph)
    symmetry = build_e8_symmetry_projection(polytope, graph_artifact=graph)
    traversal = build_manifold_traversal_plan(symmetry, polytope_artifact=polytope, graph_artifact=graph)
    divergence = run_topology_divergence_battery(
        traversal,
        symmetry_artifact=symmetry,
        polytope_artifact=polytope,
        graph_artifact=graph,
    )
    correspondence = build_arithmetic_topology_correspondence(
        divergence,
        traversal_artifact=traversal,
        symmetry_artifact=symmetry,
        polytope_artifact=polytope,
        graph_artifact=graph,
    )
    return build_multimodal_feature_schema(correspondence, float_payload={"zeta": 0.25, "alpha": 0.5})


def test_identical_input_produces_byte_identical_spectral_artifacts() -> None:
    schema = _schema_artifact()

    artifact_a = build_spectral_reasoning_layer(schema)
    artifact_b = build_spectral_reasoning_layer(schema)

    assert artifact_a == artifact_b
    assert artifact_a.to_canonical_bytes() == artifact_b.to_canonical_bytes()
    assert export_spectral_reasoning_bytes(artifact_a) == export_spectral_reasoning_bytes(artifact_b)


def test_deterministic_repeated_runs() -> None:
    schema = _schema_artifact()

    artifacts = [
        export_spectral_reasoning_bytes(build_spectral_reasoning_layer(schema, signal_payload=(3.0, 1.0, 2.0)))
        for _ in range(25)
    ]
    assert len(set(artifacts)) == 1


def test_stable_band_ordering() -> None:
    schema = _schema_artifact()
    artifact = build_spectral_reasoning_layer(schema)

    band_keys = tuple((band.band_index, band.band_name) for band in artifact.spectral_bands)
    assert band_keys == ((0, "low_band"), (1, "mid_band"), (2, "high_band"), (3, "residual_band"))


def test_stable_feature_ordering() -> None:
    schema = _schema_artifact()
    artifact = build_spectral_reasoning_layer(schema)

    feature_indexes = tuple(
        feature.source_feature_index
        for band in artifact.spectral_bands
        for feature in band.features
    )
    assert feature_indexes == tuple(range(len(feature_indexes)))


def test_stable_hash_invariants() -> None:
    schema = _schema_artifact()
    artifact = build_spectral_reasoning_layer(schema)

    assert artifact.law_invariants == (
        SPECTRAL_REASONING_LAYER_LAW,
        DETERMINISTIC_BAND_ORDERING_RULE,
        REPLAY_SAFE_SPECTRAL_IDENTITY_RULE,
        BOUNDED_SPECTRAL_SCORE_RULE,
    )
    assert artifact.stable_hash() == artifact.spectral_reasoning_hash


def test_fail_fast_malformed_schema_input() -> None:
    schema = _schema_artifact()
    invalid = replace(schema, feature_schema_hash="bad-hash")

    with pytest.raises(ValueError, match="feature_schema_hash must match stable_hash"):
        build_spectral_reasoning_layer(invalid)


def test_canonical_band_normalization() -> None:
    schema = _schema_artifact()
    artifact = build_spectral_reasoning_layer(schema)

    normalized = tuple(feature.normalized_magnitude for band in artifact.spectral_bands for feature in band.features)
    assert all(0.0 <= value <= 1.0 for value in normalized)
    assert max(normalized) == 1.0


def test_receipt_determinism() -> None:
    schema = _schema_artifact()
    artifact = build_spectral_reasoning_layer(schema)

    receipt_a = generate_spectral_reasoning_receipt(artifact)
    receipt_b = generate_spectral_reasoning_receipt(artifact)

    assert receipt_a == receipt_b
    assert receipt_a.stable_hash() == receipt_a.receipt_hash
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_lineage_hash_preservation() -> None:
    schema = _schema_artifact()
    artifact = build_spectral_reasoning_layer(schema)

    assert artifact.source_correspondence_hash == schema.source_correspondence_hash
    assert artifact.source_feature_schema_hash == schema.feature_schema_hash


def test_bounded_score_checks() -> None:
    schema = _schema_artifact()
    artifact = build_spectral_reasoning_layer(schema)

    scores = (
        artifact.spectral_coherence_score,
        artifact.band_consistency_score,
        artifact.normalization_integrity_score,
        artifact.feature_projection_score,
        artifact.overall_spectral_score,
    )
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_nan_infinity_rejection() -> None:
    schema = _schema_artifact()

    with pytest.raises(ValueError, match="finite"):
        build_spectral_reasoning_layer(schema, signal_payload=(1.0, float("nan")))

    with pytest.raises(ValueError, match="finite"):
        build_spectral_reasoning_layer(schema, signal_payload=(complex(1.0, float("inf")),))


def test_schema_identity_controls_spectral_identity() -> None:
    schema = _schema_artifact()

    baseline = build_spectral_reasoning_layer(schema)
    with_float_payload = build_spectral_reasoning_layer(schema, signal_payload=(4.0, 1.0, 2.0))
    with_complex_payload = build_spectral_reasoning_layer(schema, signal_payload=(complex(4.0, 0.0), complex(0.0, 1.0)))

    assert baseline.to_canonical_bytes() == with_float_payload.to_canonical_bytes()
    assert baseline.to_canonical_bytes() == with_complex_payload.to_canonical_bytes()
    assert baseline.spectral_reasoning_hash == with_float_payload.spectral_reasoning_hash
    assert baseline.spectral_reasoning_hash == with_complex_payload.spectral_reasoning_hash
