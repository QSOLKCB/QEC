from __future__ import annotations

import pytest

from qec.analysis.e8_symmetry_projection_layer import build_e8_symmetry_projection
from qec.analysis.episodic_memory_lifting import lift_raw_records_to_episodic_memory
from qec.analysis.fragmentation_recovery_engine import recover_fragmented_compression_chain
from qec.analysis.hash_preserving_memory_compression import CompressedMemoryArtifact, compress_semantic_theme_memory
from qec.analysis.manifold_traversal_planner import (
    BOUNDED_TRAVERSAL_SCORE_RULE,
    DETERMINISTIC_PATH_ORDERING_RULE,
    MANIFOLD_TRAVERSAL_LAW,
    REPLAY_SAFE_TRAVERSAL_IDENTITY_RULE,
    build_manifold_traversal_plan,
    export_manifold_traversal_bytes,
    generate_manifold_traversal_receipt,
)
from qec.analysis.polytope_reasoning_engine import build_polytope_reasoning_engine
from qec.analysis.semantic_theme_compaction import compact_episodic_memory_to_semantic_themes
from qec.analysis.topological_graph_kernel import build_topological_graph_kernel


def _records() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "record_id": f"theme:manifold-{i:03d}" if i % 3 == 0 else f"manifold-{i:03d}",
            "sequence_index": i,
            "source_id": "src-a" if i < 16 else "src-b",
            "provenance_id": "prov-a" if i % 4 == 0 else "prov-b",
            "state_token": "X" if i % 2 == 0 else "Y",
            "task_completed": bool(i % 5 == 0),
            "is_reset": bool(i > 0 and i % 9 == 0),
        }
        for i in range(30)
    )


def _compressed_artifact() -> CompressedMemoryArtifact:
    episodic = lift_raw_records_to_episodic_memory(_records())
    semantic = compact_episodic_memory_to_semantic_themes(episodic)
    return compress_semantic_theme_memory(semantic)


def _graph_from_observed_records(observed_indices: tuple[int, ...], compressed: CompressedMemoryArtifact | None = None):
    if compressed is None:
        compressed = _compressed_artifact()
    observed = tuple(compressed.records[idx] for idx in observed_indices)
    recovery = recover_fragmented_compression_chain(
        compressed,
        observed_records=observed,
        enable_fragmentation_recovery=True,
    )
    return build_topological_graph_kernel(compressed, recovery)


def _symmetry_from_observed_records(observed_indices: tuple[int, ...], compressed: CompressedMemoryArtifact | None = None):
    graph = _graph_from_observed_records(observed_indices, compressed)
    polytope = build_polytope_reasoning_engine(graph)
    return build_e8_symmetry_projection(polytope)


def test_identical_input_produces_byte_identical_traversal_artifacts() -> None:
    compressed = _compressed_artifact()
    symmetry = _symmetry_from_observed_records(tuple(range(len(compressed.records))), compressed)

    traversal_a = build_manifold_traversal_plan(symmetry)
    traversal_b = build_manifold_traversal_plan(symmetry)

    assert traversal_a == traversal_b
    assert traversal_a.to_canonical_bytes() == traversal_b.to_canonical_bytes()
    assert export_manifold_traversal_bytes(traversal_a) == export_manifold_traversal_bytes(traversal_b)


def test_deterministic_repeated_runs() -> None:
    symmetry = _symmetry_from_observed_records((0, 2, 4))

    artifacts = [export_manifold_traversal_bytes(build_manifold_traversal_plan(symmetry)) for _ in range(25)]
    assert len(set(artifacts)) == 1


def test_stable_path_ordering() -> None:
    symmetry = _symmetry_from_observed_records((0, 1, 3))
    traversal = build_manifold_traversal_plan(symmetry)

    keys = tuple((path.path_index, path.path_id) for path in traversal.paths)
    assert keys == tuple(sorted(keys))


def test_stable_node_ordering() -> None:
    symmetry = _symmetry_from_observed_records((0, 2, 3))
    traversal = build_manifold_traversal_plan(symmetry)

    keys = tuple((node.node_index, node.source_basis_index, node.node_id) for node in traversal.nodes)
    assert keys == tuple(sorted(keys))


def test_stable_hash_invariants() -> None:
    symmetry = _symmetry_from_observed_records((0, 1, 2))
    traversal = build_manifold_traversal_plan(symmetry)

    assert traversal.law_invariants == (
        MANIFOLD_TRAVERSAL_LAW,
        DETERMINISTIC_PATH_ORDERING_RULE,
        REPLAY_SAFE_TRAVERSAL_IDENTITY_RULE,
        BOUNDED_TRAVERSAL_SCORE_RULE,
    )
    assert traversal.stable_hash() == traversal.traversal_hash


def test_fail_fast_malformed_symmetry_input() -> None:
    symmetry = _symmetry_from_observed_records((0, 1))
    invalid_symmetry = type(symmetry)(
        schema_version=symmetry.schema_version,
        source_graph_hash=symmetry.source_graph_hash,
        source_polytope_hash=symmetry.source_polytope_hash,
        source_replay_identity_hash=symmetry.source_replay_identity_hash,
        projection=symmetry.projection,
        symmetry_hash="bad-hash",
    )

    with pytest.raises(ValueError, match="symmetry_hash must match stable_hash"):
        build_manifold_traversal_plan(invalid_symmetry)


def test_perfect_continuity_produces_high_traversal_score() -> None:
    compressed = _compressed_artifact()
    symmetry = _symmetry_from_observed_records(tuple(range(len(compressed.records))), compressed)
    traversal = build_manifold_traversal_plan(symmetry)

    assert traversal.path_continuity_score > 0.9
    assert traversal.symmetry_route_integrity_score > 0.9
    assert traversal.overall_traversal_score > 0.88


def test_fragmented_topology_degrades_traversal_score() -> None:
    compressed = _compressed_artifact()
    pristine = build_manifold_traversal_plan(_symmetry_from_observed_records(tuple(range(len(compressed.records))), compressed))
    fragmented = build_manifold_traversal_plan(_symmetry_from_observed_records((0,)))

    assert fragmented.path_continuity_score < pristine.path_continuity_score
    assert fragmented.overall_traversal_score < pristine.overall_traversal_score


def test_receipt_determinism() -> None:
    symmetry = _symmetry_from_observed_records((0, 2))
    traversal = build_manifold_traversal_plan(symmetry)

    receipt_a = generate_manifold_traversal_receipt(traversal)
    receipt_b = generate_manifold_traversal_receipt(traversal)

    assert receipt_a == receipt_b
    assert receipt_a.stable_hash() == receipt_a.receipt_hash
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_lineage_hash_preservation() -> None:
    graph = _graph_from_observed_records((0, 1, 3))
    polytope = build_polytope_reasoning_engine(graph)
    symmetry = build_e8_symmetry_projection(polytope, graph_artifact=graph)
    traversal = build_manifold_traversal_plan(symmetry, polytope_artifact=polytope, graph_artifact=graph)

    assert traversal.source_graph_hash == graph.graph_hash
    assert traversal.source_polytope_hash == polytope.polytope_hash
    assert traversal.source_symmetry_hash == symmetry.symmetry_hash


def test_lineage_validation_fail_fast() -> None:
    graph = _graph_from_observed_records((0, 1, 2))
    polytope = build_polytope_reasoning_engine(graph)
    symmetry = build_e8_symmetry_projection(polytope, graph_artifact=graph)

    other_graph = _graph_from_observed_records((0,))
    with pytest.raises(ValueError, match="graph_hash must match symmetry_artifact.source_graph_hash"):
        build_manifold_traversal_plan(symmetry, graph_artifact=other_graph)
