from __future__ import annotations

import pytest

from qec.analysis.episodic_memory_lifting import lift_raw_records_to_episodic_memory
from qec.analysis.fragmentation_recovery_engine import recover_fragmented_compression_chain
from qec.analysis.hash_preserving_memory_compression import compress_semantic_theme_memory
from qec.analysis.semantic_theme_compaction import compact_episodic_memory_to_semantic_themes
from qec.analysis.topological_graph_kernel import (
    DETERMINISTIC_EDGE_ORDERING_RULE,
    DETERMINISTIC_NODE_ORDERING_RULE,
    REPLAY_SAFE_GRAPH_IDENTITY_RULE,
    TOPOLOGICAL_GRAPH_KERNEL_LAW,
    build_topological_graph_kernel,
    export_topological_graph_bytes,
    generate_topological_graph_receipt,
)


def _records() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "record_id": f"theme:tgk-{i:03d}" if i % 5 == 0 else f"ops-{i:03d}",
            "sequence_index": i,
            "source_id": "src-a" if i < 20 else "src-b",
            "provenance_id": "prov-a" if i % 4 == 0 else "prov-b",
            "state_token": "S" if i % 2 == 0 else "T",
            "task_completed": bool(i % 7 == 0),
            "is_reset": bool(i > 0 and i % 11 == 0),
        }
        for i in range(36)
    )


def _compressed_artifact():
    episodic = lift_raw_records_to_episodic_memory(_records())
    semantic = compact_episodic_memory_to_semantic_themes(episodic)
    return compress_semantic_theme_memory(semantic)


def test_identical_input_produces_byte_identical_graph_artifacts() -> None:
    compressed = _compressed_artifact()
    recovery = recover_fragmented_compression_chain(
        compressed,
        observed_records=tuple(compressed.records),
        enable_fragmentation_recovery=True,
    )

    graph_a = build_topological_graph_kernel(compressed, recovery)
    graph_b = build_topological_graph_kernel(compressed, recovery)

    assert graph_a == graph_b
    assert graph_a.to_canonical_bytes() == graph_b.to_canonical_bytes()
    assert export_topological_graph_bytes(graph_a) == export_topological_graph_bytes(graph_b)


def test_deterministic_repeated_graph_construction() -> None:
    compressed = _compressed_artifact()
    recovery = recover_fragmented_compression_chain(
        compressed,
        observed_records=(compressed.records[3], compressed.records[0]),
        enable_fragmentation_recovery=True,
    )

    artifacts = [export_topological_graph_bytes(build_topological_graph_kernel(compressed, recovery)) for _ in range(25)]
    assert len(set(artifacts)) == 1


def test_stable_node_ordering() -> None:
    compressed = _compressed_artifact()
    recovery = recover_fragmented_compression_chain(
        compressed,
        observed_records=(compressed.records[2], compressed.records[0]),
        enable_fragmentation_recovery=True,
    )
    graph = build_topological_graph_kernel(compressed, recovery)

    keys = tuple((node.theme_index, node.node_id) for node in graph.nodes)
    assert keys == tuple(sorted(keys))


def test_stable_edge_ordering() -> None:
    compressed = _compressed_artifact()
    recovery = recover_fragmented_compression_chain(
        compressed,
        observed_records=(compressed.records[4], compressed.records[1], compressed.records[0]),
        enable_fragmentation_recovery=True,
    )
    graph = build_topological_graph_kernel(compressed, recovery)

    keys = tuple((edge.source_node_id, edge.target_node_id, edge.edge_type, edge.edge_id) for edge in graph.edges)
    assert keys == tuple(sorted(keys))


def test_stable_graph_hash_invariants() -> None:
    compressed = _compressed_artifact()
    recovery = recover_fragmented_compression_chain(
        compressed,
        observed_records=(compressed.records[1], compressed.records[0]),
        enable_fragmentation_recovery=True,
    )
    graph = build_topological_graph_kernel(compressed, recovery)

    assert graph.law_invariants == (
        TOPOLOGICAL_GRAPH_KERNEL_LAW,
        DETERMINISTIC_NODE_ORDERING_RULE,
        DETERMINISTIC_EDGE_ORDERING_RULE,
        REPLAY_SAFE_GRAPH_IDENTITY_RULE,
    )
    assert graph.stable_hash() == graph.graph_hash


def test_fail_fast_malformed_node_lineage() -> None:
    compressed = _compressed_artifact()
    bad_record = type(compressed.records[1])(
        theme_id=compressed.records[1].theme_id,
        theme_index=1,
        source_theme_hash=compressed.records[1].source_theme_hash,
        source_replay_identity_hash=compressed.records[1].source_replay_identity_hash,
        source_parent_theme_hash=compressed.records[0].source_parent_theme_hash,
        signature_ref=compressed.records[1].signature_ref,
        reason_ref=compressed.records[1].reason_ref,
        episode_hashes_ref=compressed.records[1].episode_hashes_ref,
        compression_record_hash=compressed.records[1].compression_record_hash,
    )
    invalid = type(compressed)(
        schema_version=compressed.schema_version,
        source_artifact_hash=compressed.source_artifact_hash,
        source_theme_count=compressed.source_theme_count,
        compressed_record_count=compressed.compressed_record_count,
        signature_table=compressed.signature_table,
        reason_table=compressed.reason_table,
        episode_hash_chain_table=compressed.episode_hash_chain_table,
        records=(compressed.records[0], bad_record) + compressed.records[2:],
        preserved_theme_hashes=compressed.preserved_theme_hashes,
        source_compaction_ratio=compressed.source_compaction_ratio,
        compression_ratio=compressed.compression_ratio,
        compression_chain_head=compressed.compression_chain_head,
        replay_identity_hash=compressed.replay_identity_hash,
        law_invariants=compressed.law_invariants,
        compression_hash=compressed.compression_hash,
    )
    recovery = recover_fragmented_compression_chain(
        compressed,
        observed_records=tuple(compressed.records),
        enable_fragmentation_recovery=True,
    )

    with pytest.raises(ValueError, match="invalid lineage structure"):
        build_topological_graph_kernel(invalid, recovery)


def test_continuity_edge_correctness() -> None:
    compressed = _compressed_artifact()
    recovery = recover_fragmented_compression_chain(
        compressed,
        observed_records=(compressed.records[3], compressed.records[0]),
        enable_fragmentation_recovery=True,
    )
    graph = build_topological_graph_kernel(compressed, recovery)

    assert graph.edge_count == max(0, graph.node_count - 1)
    assert all(edge.edge_type == "continuity" for edge in graph.edges)
    assert all(0.0 <= edge.continuity_weight <= 1.0 for edge in graph.edges)


def test_perfect_chain_graph_is_max_connectivity() -> None:
    compressed = _compressed_artifact()
    recovery = recover_fragmented_compression_chain(
        compressed,
        observed_records=tuple(compressed.records),
        enable_fragmentation_recovery=True,
    )
    graph = build_topological_graph_kernel(compressed, recovery)

    assert graph.connectivity_score == 1.0
    assert graph.continuity_graph_score == 1.0
    assert graph.lineage_integrity_score == 1.0
    assert graph.overall_topology_score == 1.0


def test_fragmented_chain_graph_degrades() -> None:
    compressed = _compressed_artifact()
    pristine = recover_fragmented_compression_chain(
        compressed,
        observed_records=tuple(compressed.records),
        enable_fragmentation_recovery=True,
    )
    fragmented = recover_fragmented_compression_chain(
        compressed,
        observed_records=(compressed.records[4], compressed.records[0]),
        enable_fragmentation_recovery=True,
    )

    pristine_graph = build_topological_graph_kernel(compressed, pristine)
    fragmented_graph = build_topological_graph_kernel(compressed, fragmented)

    assert fragmented_graph.continuity_graph_score < pristine_graph.continuity_graph_score
    assert fragmented_graph.overall_topology_score < pristine_graph.overall_topology_score


def test_receipt_determinism() -> None:
    compressed = _compressed_artifact()
    recovery = recover_fragmented_compression_chain(
        compressed,
        observed_records=(compressed.records[2], compressed.records[0]),
        enable_fragmentation_recovery=True,
    )
    graph = build_topological_graph_kernel(compressed, recovery)

    receipt_a = generate_topological_graph_receipt(graph)
    receipt_b = generate_topological_graph_receipt(graph)

    assert receipt_a == receipt_b
    assert receipt_a.stable_hash() == receipt_a.receipt_hash
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_fail_fast_mismatched_preserved_theme_hashes() -> None:
    compressed = _compressed_artifact()
    recovery = recover_fragmented_compression_chain(
        compressed,
        observed_records=tuple(compressed.records),
        enable_fragmentation_recovery=True,
    )
    invalid_recovery = type(recovery)(
        schema_version=recovery.schema_version,
        source_compression_hash=recovery.source_compression_hash,
        source_replay_identity_hash=recovery.source_replay_identity_hash,
        repaired_chain_head=recovery.repaired_chain_head,
        repaired_records=recovery.repaired_records,
        preserved_theme_hashes=tuple(reversed(recovery.preserved_theme_hashes)),
        continuity_score=recovery.continuity_score,
        recovery_confidence=recovery.recovery_confidence,
        fracture_severity=recovery.fracture_severity,
        lineage_report=recovery.lineage_report,
        sonification_repair_metadata=recovery.sonification_repair_metadata,
        replay_safe=recovery.replay_safe,
        law_invariants=recovery.law_invariants,
        recovery_artifact_hash=recovery.recovery_artifact_hash,
    )

    with pytest.raises(ValueError, match="preserved_theme_hashes must match source_artifact"):
        build_topological_graph_kernel(compressed, invalid_recovery)


def test_fragmented_chain_connectivity_and_lineage_degrade() -> None:
    compressed = _compressed_artifact()
    pristine = recover_fragmented_compression_chain(
        compressed,
        observed_records=tuple(compressed.records),
        enable_fragmentation_recovery=True,
    )
    fragmented = recover_fragmented_compression_chain(
        compressed,
        observed_records=(compressed.records[4], compressed.records[0]),
        enable_fragmentation_recovery=True,
    )

    pristine_graph = build_topological_graph_kernel(compressed, pristine)
    fragmented_graph = build_topological_graph_kernel(compressed, fragmented)

    assert fragmented_graph.connectivity_score < pristine_graph.connectivity_score
    assert fragmented_graph.lineage_integrity_score < pristine_graph.lineage_integrity_score
