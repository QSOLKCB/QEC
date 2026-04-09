from __future__ import annotations

from dataclasses import replace

import pytest

from qec.analysis.arithmetic_topology_correspondence_engine import (
    ARITHMETIC_TOPOLOGY_CORRESPONDENCE_LAW,
    BOUNDED_CORRESPONDENCE_SCORE_RULE,
    DETERMINISTIC_WITNESS_ORDERING_RULE,
    REPLAY_SAFE_CORRESPONDENCE_IDENTITY_RULE,
    build_arithmetic_topology_correspondence,
    export_arithmetic_correspondence_bytes,
    generate_arithmetic_correspondence_receipt,
)
from qec.analysis.e8_symmetry_projection_layer import build_e8_symmetry_projection
from qec.analysis.episodic_memory_lifting import lift_raw_records_to_episodic_memory
from qec.analysis.fragmentation_recovery_engine import recover_fragmented_compression_chain
from qec.analysis.hash_preserving_memory_compression import CompressedMemoryArtifact, compress_semantic_theme_memory
from qec.analysis.manifold_traversal_planner import build_manifold_traversal_plan
from qec.analysis.polytope_reasoning_engine import build_polytope_reasoning_engine
from qec.analysis.semantic_theme_compaction import compact_episodic_memory_to_semantic_themes
from qec.analysis.topological_graph_kernel import build_topological_graph_kernel
from qec.analysis.topology_divergence_battery import run_topology_divergence_battery


def _records() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "record_id": f"theme:correspondence-{i:03d}" if i % 3 == 0 else f"correspondence-{i:03d}",
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


def _pipeline(observed_indices: tuple[int, ...], compressed: CompressedMemoryArtifact | None = None):
    if compressed is None:
        compressed = _compressed_artifact()
    observed = tuple(compressed.records[idx] for idx in observed_indices)
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
    return divergence, traversal, symmetry, polytope, graph


def _prefix_indices(compressed: CompressedMemoryArtifact, count: int) -> tuple[int, ...]:
    upper = min(count, len(compressed.records))
    return tuple(range(upper))


def test_identical_input_produces_byte_identical_correspondence_artifacts() -> None:
    divergence, _, _, _, _ = _pipeline((0, 2, 4))

    artifact_a = build_arithmetic_topology_correspondence(divergence)
    artifact_b = build_arithmetic_topology_correspondence(divergence)

    assert artifact_a == artifact_b
    assert artifact_a.to_canonical_bytes() == artifact_b.to_canonical_bytes()
    assert export_arithmetic_correspondence_bytes(artifact_a) == export_arithmetic_correspondence_bytes(artifact_b)


def test_deterministic_repeated_runs() -> None:
    divergence, _, _, _, _ = _pipeline((0, 1, 3))

    artifacts = [export_arithmetic_correspondence_bytes(build_arithmetic_topology_correspondence(divergence)) for _ in range(25)]
    assert len(set(artifacts)) == 1


def test_stable_witness_ordering() -> None:
    compressed = _compressed_artifact()
    divergence, _, _, _, _ = _pipeline(_prefix_indices(compressed, 6), compressed)
    artifact = build_arithmetic_topology_correspondence(divergence)

    witness_keys = tuple((w.witness_index, w.anchor_node_id, w.scenario_id, w.witness_id) for w in artifact.witnesses)
    assert witness_keys == tuple(sorted(witness_keys))


def test_stable_primitive_ordering() -> None:
    compressed = _compressed_artifact()
    divergence, _, _, _, _ = _pipeline(_prefix_indices(compressed, 6), compressed)
    artifact = build_arithmetic_topology_correspondence(divergence)

    primitive_keys = tuple(
        (p.scenario_id, p.path_id, p.arithmetic_step, p.segment_id, p.primitive_id)
        for p in artifact.primitives
    )
    assert primitive_keys == tuple(sorted(primitive_keys))


def test_stable_hash_invariants() -> None:
    divergence, _, _, _, _ = _pipeline((0, 1, 2))
    artifact = build_arithmetic_topology_correspondence(divergence)

    assert artifact.law_invariants == (
        ARITHMETIC_TOPOLOGY_CORRESPONDENCE_LAW,
        DETERMINISTIC_WITNESS_ORDERING_RULE,
        REPLAY_SAFE_CORRESPONDENCE_IDENTITY_RULE,
        BOUNDED_CORRESPONDENCE_SCORE_RULE,
    )
    assert artifact.stable_hash() == artifact.correspondence_hash


def test_fail_fast_malformed_divergence_input() -> None:
    divergence, _, _, _, _ = _pipeline((0, 1, 2))
    invalid = replace(divergence, divergence_hash="bad-hash")

    with pytest.raises(ValueError, match="divergence_hash must match stable_hash"):
        build_arithmetic_topology_correspondence(invalid)


def test_perfect_continuity_produces_high_correspondence_score() -> None:
    compressed = _compressed_artifact()
    divergence, _, _, _, _ = _pipeline(tuple(range(len(compressed.records))), compressed)
    artifact = build_arithmetic_topology_correspondence(divergence)

    assert artifact.witness_consistency_score > 0.8
    assert artifact.topology_arithmetic_coherence_score > 0.78
    assert artifact.overall_correspondence_score > 0.8


def test_fragmented_topology_degrades_coherence() -> None:
    compressed = _compressed_artifact()
    pristine_divergence, _, _, _, _ = _pipeline(tuple(range(len(compressed.records))), compressed)
    fragmented_divergence, _, _, _, _ = _pipeline((0,), compressed)

    pristine = build_arithmetic_topology_correspondence(pristine_divergence)
    fragmented = build_arithmetic_topology_correspondence(fragmented_divergence)

    assert fragmented.witness_consistency_score < pristine.witness_consistency_score
    assert fragmented.topology_arithmetic_coherence_score < pristine.topology_arithmetic_coherence_score
    assert fragmented.overall_correspondence_score < pristine.overall_correspondence_score


def test_monotonic_degradation_across_fragmentation_levels() -> None:
    compressed = _compressed_artifact()
    dense_divergence, _, _, _, _ = _pipeline(tuple(range(len(compressed.records))), compressed)
    medium_seed = _prefix_indices(compressed, max(2, len(compressed.records)))
    medium_divergence, _, _, _, _ = _pipeline(medium_seed[::2], compressed)
    sparse_divergence, _, _, _, _ = _pipeline((0,), compressed)

    dense = build_arithmetic_topology_correspondence(dense_divergence)
    medium = build_arithmetic_topology_correspondence(medium_divergence)
    sparse = build_arithmetic_topology_correspondence(sparse_divergence)

    coherence_scores = (dense.topology_arithmetic_coherence_score, medium.topology_arithmetic_coherence_score, sparse.topology_arithmetic_coherence_score)
    assert max(coherence_scores) - min(coherence_scores) > 0.05
    assert dense.topology_arithmetic_coherence_score >= sparse.topology_arithmetic_coherence_score
    assert dense.overall_correspondence_score >= sparse.overall_correspondence_score


def test_receipt_determinism() -> None:
    divergence, _, _, _, _ = _pipeline((0, 2))
    artifact = build_arithmetic_topology_correspondence(divergence)

    receipt_a = generate_arithmetic_correspondence_receipt(artifact)
    receipt_b = generate_arithmetic_correspondence_receipt(artifact)

    assert receipt_a == receipt_b
    assert receipt_a.stable_hash() == receipt_a.receipt_hash
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_lineage_hash_preservation() -> None:
    divergence, traversal, symmetry, polytope, graph = _pipeline((0, 1, 3))
    artifact = build_arithmetic_topology_correspondence(
        divergence,
        traversal_artifact=traversal,
        symmetry_artifact=symmetry,
        polytope_artifact=polytope,
        graph_artifact=graph,
    )

    assert artifact.source_graph_hash == graph.graph_hash
    assert artifact.source_polytope_hash == polytope.polytope_hash
    assert artifact.source_symmetry_hash == symmetry.symmetry_hash
    assert artifact.source_traversal_hash == traversal.traversal_hash
    assert artifact.source_divergence_hash == divergence.divergence_hash


def test_lineage_validation_fail_fast() -> None:
    divergence, traversal, symmetry, polytope, _ = _pipeline((0, 1, 2))
    _, _, _, _, other_graph = _pipeline((0,))

    with pytest.raises(ValueError, match="graph_hash must match divergence_artifact.source_graph_hash"):
        build_arithmetic_topology_correspondence(
            divergence,
            traversal_artifact=traversal,
            symmetry_artifact=symmetry,
            polytope_artifact=polytope,
            graph_artifact=other_graph,
        )
