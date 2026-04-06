from __future__ import annotations

import pytest

from qec.analysis.episodic_memory_lifting import lift_raw_records_to_episodic_memory
from qec.analysis.fragmentation_recovery_engine import recover_fragmented_compression_chain
from qec.analysis.hash_preserving_memory_compression import compress_semantic_theme_memory
from qec.analysis.polytope_reasoning_engine import (
    DETERMINISTIC_FACE_ORDERING_RULE,
    DETERMINISTIC_VERTEX_ORDERING_RULE,
    POLYTOPE_REASONING_LAW,
    REPLAY_SAFE_POLYTOPE_IDENTITY_RULE,
    build_polytope_reasoning_engine,
    export_polytope_bytes,
    generate_polytope_receipt,
)
from qec.analysis.semantic_theme_compaction import compact_episodic_memory_to_semantic_themes
from qec.analysis.topological_graph_kernel import build_topological_graph_kernel


def _records() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "record_id": f"theme:poly-{i:03d}" if i % 3 == 0 else f"poly-{i:03d}",
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


def test_identical_input_produces_byte_identical_polytope_artifacts() -> None:
    graph = _graph_from_observed_records(tuple(range(len(_compressed_artifact().records))))

    polytope_a = build_polytope_reasoning_engine(graph)
    polytope_b = build_polytope_reasoning_engine(graph)

    assert polytope_a == polytope_b
    assert polytope_a.to_canonical_bytes() == polytope_b.to_canonical_bytes()
    assert export_polytope_bytes(polytope_a) == export_polytope_bytes(polytope_b)


def test_deterministic_repeated_runs() -> None:
    graph = _graph_from_observed_records((0, 2, 4))

    artifacts = [export_polytope_bytes(build_polytope_reasoning_engine(graph)) for _ in range(25)]
    assert len(set(artifacts)) == 1


def test_stable_vertex_ordering() -> None:
    graph = _graph_from_observed_records((0, 1, 3))
    polytope = build_polytope_reasoning_engine(graph)

    keys = tuple((vertex.theme_index, vertex.source_node_id, vertex.vertex_id) for vertex in polytope.vertices)
    assert keys == tuple(sorted(keys))


def test_stable_face_ordering() -> None:
    graph = _graph_from_observed_records((0, 2, 3))
    polytope = build_polytope_reasoning_engine(graph)

    keys = tuple((face.face_index, face.face_id, face.vertex_ids, face.source_edge_ids) for face in polytope.faces)
    assert keys == tuple(sorted(keys))


def test_stable_hash_invariants() -> None:
    graph = _graph_from_observed_records((0, 1, 2))
    polytope = build_polytope_reasoning_engine(graph)

    assert polytope.law_invariants == (
        POLYTOPE_REASONING_LAW,
        DETERMINISTIC_VERTEX_ORDERING_RULE,
        DETERMINISTIC_FACE_ORDERING_RULE,
        REPLAY_SAFE_POLYTOPE_IDENTITY_RULE,
    )
    assert polytope.stable_hash() == polytope.polytope_hash


def test_fail_fast_malformed_graph_input() -> None:
    graph = _graph_from_observed_records((0, 1))
    invalid_graph = type(graph)(
        schema_version=graph.schema_version,
        source_compression_hash=graph.source_compression_hash,
        source_replay_identity_hash=graph.source_replay_identity_hash,
        recovered_chain_head=graph.recovered_chain_head,
        node_count=graph.node_count + 1,
        edge_count=graph.edge_count,
        nodes=graph.nodes,
        edges=graph.edges,
        connectivity_score=graph.connectivity_score,
        continuity_graph_score=graph.continuity_graph_score,
        lineage_integrity_score=graph.lineage_integrity_score,
        overall_topology_score=graph.overall_topology_score,
        law_invariants=graph.law_invariants,
        graph_hash=graph.graph_hash,
    )

    with pytest.raises(ValueError, match="node_count must match nodes length"):
        build_polytope_reasoning_engine(invalid_graph)


def test_perfect_chain_graph_produces_max_polytope_score() -> None:
    graph = _graph_from_observed_records(tuple(range(len(_compressed_artifact().records))))
    polytope = build_polytope_reasoning_engine(graph)

    assert polytope.vertex_connectivity_score == 1.0
    assert polytope.face_continuity_score == 1.0
    # dimensional_consistency_score is bounded below 1.0 for large chains due to
    # source_edge_uniqueness penalizing shared edges across consecutive faces.
    assert polytope.dimensional_consistency_score > 0.8
    assert polytope.polytope_integrity_score > 0.9
    assert polytope.overall_polytope_score > 0.9


def test_fragmented_graph_degrades_polytope_score() -> None:
    pristine = build_polytope_reasoning_engine(_graph_from_observed_records(tuple(range(len(_compressed_artifact().records)))))
    fragmented = build_polytope_reasoning_engine(_graph_from_observed_records((0,)))

    assert fragmented.face_continuity_score < pristine.face_continuity_score
    assert fragmented.overall_polytope_score < pristine.overall_polytope_score


def test_receipt_determinism() -> None:
    graph = _graph_from_observed_records((0, 2))
    polytope = build_polytope_reasoning_engine(graph)

    receipt_a = generate_polytope_receipt(polytope)
    receipt_b = generate_polytope_receipt(polytope)

    assert receipt_a == receipt_b
    assert receipt_a.stable_hash() == receipt_a.receipt_hash
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_graph_hash_lineage_preservation() -> None:
    graph = _graph_from_observed_records((0, 1, 3))
    polytope = build_polytope_reasoning_engine(graph)

    assert polytope.source_graph_hash == graph.graph_hash
