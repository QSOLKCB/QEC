from __future__ import annotations

import pytest

from qec.analysis.e8_symmetry_projection_layer import (
    BOUNDED_PROJECTION_SCORE_RULE,
    DETERMINISTIC_BASIS_ORDERING_RULE,
    E8_SYMMETRY_PROJECTION_LAW,
    REPLAY_SAFE_SYMMETRY_IDENTITY_RULE,
    build_e8_symmetry_projection,
    export_e8_projection_bytes,
    generate_e8_projection_receipt,
)
from qec.analysis.episodic_memory_lifting import lift_raw_records_to_episodic_memory
from qec.analysis.fragmentation_recovery_engine import recover_fragmented_compression_chain
from qec.analysis.hash_preserving_memory_compression import compress_semantic_theme_memory
from qec.analysis.polytope_reasoning_engine import build_polytope_reasoning_engine
from qec.analysis.semantic_theme_compaction import compact_episodic_memory_to_semantic_themes
from qec.analysis.topological_graph_kernel import build_topological_graph_kernel


def _records() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "record_id": f"theme:e8-{i:03d}" if i % 3 == 0 else f"e8-{i:03d}",
            "sequence_index": i,
            "source_id": "src-a" if i < 16 else "src-b",
            "provenance_id": "prov-a" if i % 4 == 0 else "prov-b",
            "state_token": "X" if i % 2 == 0 else "Y",
            "task_completed": bool(i % 5 == 0),
            "is_reset": bool(i > 0 and i % 9 == 0),
        }
        for i in range(30)
    )


def _compressed_artifact():
    episodic = lift_raw_records_to_episodic_memory(_records())
    semantic = compact_episodic_memory_to_semantic_themes(episodic)
    return compress_semantic_theme_memory(semantic)


def _graph_from_observed_records(observed_indices: tuple[int, ...]):
    compressed = _compressed_artifact()
    observed = tuple(compressed.records[idx] for idx in observed_indices)
    recovery = recover_fragmented_compression_chain(
        compressed,
        observed_records=observed,
        enable_fragmentation_recovery=True,
    )
    return build_topological_graph_kernel(compressed, recovery)


def _polytope_from_observed_records(observed_indices: tuple[int, ...]):
    return build_polytope_reasoning_engine(_graph_from_observed_records(observed_indices))


def test_identical_input_produces_byte_identical_symmetry_artifacts() -> None:
    polytope = _polytope_from_observed_records(tuple(range(len(_compressed_artifact().records))))

    symmetry_a = build_e8_symmetry_projection(polytope)
    symmetry_b = build_e8_symmetry_projection(polytope)

    assert symmetry_a == symmetry_b
    assert symmetry_a.to_canonical_bytes() == symmetry_b.to_canonical_bytes()
    assert export_e8_projection_bytes(symmetry_a) == export_e8_projection_bytes(symmetry_b)


def test_deterministic_repeated_runs() -> None:
    polytope = _polytope_from_observed_records((0, 2, 4))

    artifacts = [export_e8_projection_bytes(build_e8_symmetry_projection(polytope)) for _ in range(25)]
    assert len(set(artifacts)) == 1


def test_stable_basis_ordering() -> None:
    polytope = _polytope_from_observed_records((0, 1, 3))
    symmetry = build_e8_symmetry_projection(polytope)

    keys = tuple((vector.basis_index, vector.basis_id) for vector in symmetry.projection.vectors)
    assert keys == tuple(sorted(keys))


def test_stable_coordinate_ordering() -> None:
    polytope = _polytope_from_observed_records((0, 2, 3))
    symmetry = build_e8_symmetry_projection(polytope)

    for vector in symmetry.projection.vectors:
        assert vector.coordinate_order == tuple(f"x{i}" for i in range(8))
        assert len(vector.normalized_coordinates) == len(vector.coordinate_order)


def test_stable_hash_invariants() -> None:
    polytope = _polytope_from_observed_records((0, 1, 2))
    symmetry = build_e8_symmetry_projection(polytope)

    assert symmetry.projection.law_invariants == (
        E8_SYMMETRY_PROJECTION_LAW,
        DETERMINISTIC_BASIS_ORDERING_RULE,
        REPLAY_SAFE_SYMMETRY_IDENTITY_RULE,
        BOUNDED_PROJECTION_SCORE_RULE,
    )
    assert symmetry.projection.stable_hash() == symmetry.projection.symmetry_hash
    assert symmetry.stable_hash() == symmetry.symmetry_hash


def test_fail_fast_malformed_polytope_input() -> None:
    polytope = _polytope_from_observed_records((0, 1))
    invalid_polytope = type(polytope)(
        schema_version=polytope.schema_version,
        source_graph_hash=polytope.source_graph_hash,
        source_replay_identity_hash=polytope.source_replay_identity_hash,
        vertex_count=polytope.vertex_count + 1,
        face_count=polytope.face_count,
        vertices=polytope.vertices,
        faces=polytope.faces,
        vertex_connectivity_score=polytope.vertex_connectivity_score,
        face_continuity_score=polytope.face_continuity_score,
        dimensional_consistency_score=polytope.dimensional_consistency_score,
        polytope_integrity_score=polytope.polytope_integrity_score,
        overall_polytope_score=polytope.overall_polytope_score,
        law_invariants=polytope.law_invariants,
        polytope_hash=polytope.polytope_hash,
    )

    with pytest.raises(ValueError, match="vertex_count must match vertices length"):
        build_e8_symmetry_projection(invalid_polytope)


def test_perfect_continuity_produces_high_symmetry_score() -> None:
    polytope = _polytope_from_observed_records(tuple(range(len(_compressed_artifact().records))))
    symmetry = build_e8_symmetry_projection(polytope)

    assert symmetry.projection.symmetry_alignment_score > 0.85
    assert symmetry.projection.lattice_continuity_score > 0.95
    assert symmetry.projection.overall_symmetry_score > 0.9


def test_fragmented_topology_degrades_symmetry_score() -> None:
    pristine = build_e8_symmetry_projection(_polytope_from_observed_records(tuple(range(len(_compressed_artifact().records)))))
    fragmented = build_e8_symmetry_projection(_polytope_from_observed_records((0,)))

    assert fragmented.projection.lattice_continuity_score < pristine.projection.lattice_continuity_score
    assert fragmented.projection.overall_symmetry_score < pristine.projection.overall_symmetry_score


def test_receipt_determinism() -> None:
    polytope = _polytope_from_observed_records((0, 2))
    symmetry = build_e8_symmetry_projection(polytope)

    receipt_a = generate_e8_projection_receipt(symmetry)
    receipt_b = generate_e8_projection_receipt(symmetry)

    assert receipt_a == receipt_b
    assert receipt_a.stable_hash() == receipt_a.receipt_hash
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_lineage_hash_preservation() -> None:
    graph = _graph_from_observed_records((0, 1, 3))
    polytope = build_polytope_reasoning_engine(graph)
    symmetry = build_e8_symmetry_projection(polytope, graph_artifact=graph)

    assert symmetry.source_graph_hash == graph.graph_hash
    assert symmetry.source_polytope_hash == polytope.polytope_hash
    assert symmetry.projection.source_graph_hash == graph.graph_hash
    assert symmetry.projection.source_polytope_hash == polytope.polytope_hash


def test_graph_lineage_validation_fail_fast() -> None:
    graph = _graph_from_observed_records((0, 1, 2))
    polytope = build_polytope_reasoning_engine(graph)

    other_graph = _graph_from_observed_records((0,))
    with pytest.raises(ValueError, match="graph_hash must match polytope_artifact.source_graph_hash"):
        build_e8_symmetry_projection(polytope, graph_artifact=other_graph)
