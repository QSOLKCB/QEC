from __future__ import annotations

from dataclasses import replace

import pytest

from qec.analysis.arithmetic_topology_correspondence_engine import build_arithmetic_topology_correspondence
from qec.analysis.e8_symmetry_projection_layer import build_e8_symmetry_projection
from qec.analysis.episodic_memory_lifting import lift_raw_records_to_episodic_memory
from qec.analysis.fragmentation_recovery_engine import recover_fragmented_compression_chain
from qec.analysis.hash_preserving_memory_compression import CompressedMemoryArtifact, compress_semantic_theme_memory
from qec.analysis.manifold_traversal_planner import build_manifold_traversal_plan
from qec.analysis.multimodal_feature_schema import (
    BOUNDED_SCHEMA_SCORE_RULE,
    DETERMINISTIC_FEATURE_ORDERING_RULE,
    MULTIMODAL_FEATURE_SCHEMA_LAW,
    REPLAY_SAFE_SCHEMA_IDENTITY_RULE,
    build_multimodal_feature_schema,
    export_multimodal_feature_schema_bytes,
    generate_multimodal_feature_schema_receipt,
)
from qec.analysis.polytope_reasoning_engine import build_polytope_reasoning_engine
from qec.analysis.semantic_theme_compaction import compact_episodic_memory_to_semantic_themes
from qec.analysis.topological_graph_kernel import build_topological_graph_kernel
from qec.analysis.topology_divergence_battery import run_topology_divergence_battery


def _records() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "record_id": f"theme:multimodal-{i:03d}" if i % 2 == 0 else f"multimodal-{i:03d}",
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


def _correspondence() -> object:
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
    return build_arithmetic_topology_correspondence(
        divergence,
        traversal_artifact=traversal,
        symmetry_artifact=symmetry,
        polytope_artifact=polytope,
        graph_artifact=graph,
    )


def test_identical_input_produces_byte_identical_schema_artifacts() -> None:
    correspondence = _correspondence()

    artifact_a = build_multimodal_feature_schema(correspondence)
    artifact_b = build_multimodal_feature_schema(correspondence)

    assert artifact_a == artifact_b
    assert artifact_a.to_canonical_bytes() == artifact_b.to_canonical_bytes()
    assert export_multimodal_feature_schema_bytes(artifact_a) == export_multimodal_feature_schema_bytes(artifact_b)


def test_deterministic_repeated_runs() -> None:
    correspondence = _correspondence()

    artifacts = [
        export_multimodal_feature_schema_bytes(
            build_multimodal_feature_schema(
                correspondence,
                float_payload={"zeta": 0.25, "alpha": 0.5},
                int_payload={"beta": 2, "alpha": 1},
                str_payload={"mode": "deterministic"},
                tuple_feature_streams=(("sensor", "temperature", 3.5), ("telecom", "band", "n78")),
            )
        )
        for _ in range(25)
    ]
    assert len(set(artifacts)) == 1


def test_stable_feature_ordering() -> None:
    correspondence = _correspondence()
    artifact = build_multimodal_feature_schema(
        correspondence,
        float_payload={"zeta": 0.25, "alpha": 0.5},
        int_payload={"beta": 2, "alpha": 1},
        tuple_feature_streams=(("telecom", "carrier", "x"), ("sensor", "accel", 4.0)),
    )

    keys = tuple((f.feature_family, f.feature_name, f.feature_index) for f in artifact.features)
    assert keys == tuple(sorted(keys, key=lambda row: (row[0], row[1], row[2])))


def test_stable_namespace_ordering() -> None:
    correspondence = _correspondence()
    artifact = build_multimodal_feature_schema(
        correspondence,
        float_payload={"b": 1.0, "a": 2.0},
        int_payload={"x": 1},
        str_payload={"mode": "m"},
        tuple_feature_streams=(("telecom", "carrier", "x"), ("sensor", "accel", 4.0)),
    )

    namespace_keys = tuple((n.namespace_index, n.feature_namespace) for n in artifact.namespaces)
    assert namespace_keys == tuple(sorted(namespace_keys, key=lambda row: (row[0], row[1])))


def test_stable_hash_invariants() -> None:
    correspondence = _correspondence()
    artifact = build_multimodal_feature_schema(correspondence)

    assert artifact.law_invariants == (
        MULTIMODAL_FEATURE_SCHEMA_LAW,
        DETERMINISTIC_FEATURE_ORDERING_RULE,
        REPLAY_SAFE_SCHEMA_IDENTITY_RULE,
        BOUNDED_SCHEMA_SCORE_RULE,
    )
    assert artifact.stable_hash() == artifact.feature_schema_hash


def test_fail_fast_malformed_input() -> None:
    correspondence = _correspondence()

    with pytest.raises(ValueError, match="tuple_feature_streams entries"):
        build_multimodal_feature_schema(correspondence, tuple_feature_streams=(("a",),))

    with pytest.raises(ValueError, match="float_payload values must be finite"):
        build_multimodal_feature_schema(correspondence, float_payload={"bad": float("inf")})


def test_canonical_ordering_normalization() -> None:
    correspondence = _correspondence()

    artifact_unsorted = build_multimodal_feature_schema(
        correspondence,
        float_payload={"zeta": 0.25, "alpha": 0.5},
        int_payload={"beta": 2, "alpha": 1},
        str_payload={"zzz": "x", "aaa": "y"},
    )
    artifact_sorted = build_multimodal_feature_schema(
        correspondence,
        float_payload={"alpha": 0.5, "zeta": 0.25},
        int_payload={"alpha": 1, "beta": 2},
        str_payload={"aaa": "y", "zzz": "x"},
    )

    assert tuple(f.to_dict() for f in artifact_unsorted.features) == tuple(f.to_dict() for f in artifact_sorted.features)
    assert tuple(n.to_dict() for n in artifact_unsorted.namespaces) == tuple(n.to_dict() for n in artifact_sorted.namespaces)
    assert artifact_unsorted.feature_ordering_score < 1.0
    assert artifact_sorted.feature_ordering_score == 1.0


def test_receipt_determinism() -> None:
    correspondence = _correspondence()
    artifact = build_multimodal_feature_schema(correspondence, float_payload={"alpha": 0.5})

    receipt_a = generate_multimodal_feature_schema_receipt(artifact)
    receipt_b = generate_multimodal_feature_schema_receipt(artifact)

    assert receipt_a == receipt_b
    assert receipt_a.stable_hash() == receipt_a.receipt_hash
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_lineage_hash_preservation() -> None:
    correspondence = _correspondence()
    artifact = build_multimodal_feature_schema(correspondence)

    assert artifact.source_graph_hash == correspondence.source_graph_hash
    assert artifact.source_polytope_hash == correspondence.source_polytope_hash
    assert artifact.source_symmetry_hash == correspondence.source_symmetry_hash
    assert artifact.source_traversal_hash == correspondence.source_traversal_hash
    assert artifact.source_divergence_hash == correspondence.source_divergence_hash
    assert artifact.source_correspondence_hash == correspondence.correspondence_hash


def test_namespace_versioning_checks() -> None:
    correspondence = _correspondence()
    artifact = build_multimodal_feature_schema(
        correspondence,
        float_payload={"a": 1.0},
        int_payload={"x": 1},
        tuple_feature_streams=(("sensor", "temp", 2.0),),
    )

    assert artifact.schema_version == 1
    assert all(feature.feature_schema_version == 1 for feature in artifact.features)
    assert all(namespace.feature_schema_version == 1 for namespace in artifact.namespaces)
    assert all(namespace.feature_namespace.endswith(".v1") for namespace in artifact.namespaces)


def test_fail_fast_correspondence_hash_mismatch() -> None:
    correspondence = _correspondence()
    invalid = replace(correspondence, correspondence_hash="bad-hash")

    with pytest.raises(ValueError, match="correspondence_hash must match stable_hash"):
        build_multimodal_feature_schema(invalid)
