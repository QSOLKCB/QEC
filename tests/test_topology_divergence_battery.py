from __future__ import annotations

from dataclasses import replace

import pytest

from qec.analysis.e8_symmetry_projection_layer import build_e8_symmetry_projection
from qec.analysis.episodic_memory_lifting import lift_raw_records_to_episodic_memory
from qec.analysis.fragmentation_recovery_engine import recover_fragmented_compression_chain
from qec.analysis.hash_preserving_memory_compression import CompressedMemoryArtifact, compress_semantic_theme_memory
from qec.analysis.manifold_traversal_planner import (
    ManifoldNode,
    ManifoldTraversalResult,
    TraversalPath,
    build_manifold_traversal_plan,
)
from qec.analysis.polytope_reasoning_engine import build_polytope_reasoning_engine
from qec.analysis.semantic_theme_compaction import compact_episodic_memory_to_semantic_themes
from qec.analysis.topological_graph_kernel import build_topological_graph_kernel
from qec.analysis.topology_divergence_battery import (
    BOUNDED_DIVERGENCE_SCORE_RULE,
    DETERMINISTIC_SCENARIO_ORDERING_RULE,
    REPLAY_SAFE_DIVERGENCE_IDENTITY_RULE,
    TOPOLOGY_DIVERGENCE_LAW,
    export_topology_divergence_bytes,
    generate_topology_divergence_receipt,
    run_topology_divergence_battery,
)


def _records() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "record_id": f"theme:divergence-{i:03d}" if i % 3 == 0 else f"divergence-{i:03d}",
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


def _traversal_from_observed_records(observed_indices: tuple[int, ...], compressed: CompressedMemoryArtifact | None = None):
    graph = _graph_from_observed_records(observed_indices, compressed)
    polytope = build_polytope_reasoning_engine(graph)
    symmetry = build_e8_symmetry_projection(polytope, graph_artifact=graph)
    traversal = build_manifold_traversal_plan(symmetry, polytope_artifact=polytope, graph_artifact=graph)
    return traversal, symmetry, polytope, graph


def _manual_traversal(path_scores: tuple[float, ...]) -> ManifoldTraversalResult:
    nodes = tuple(
        ManifoldNode(
            node_id=f"n-{idx}",
            node_index=idx,
            source_basis_id=f"b-{idx}",
            source_basis_index=idx,
            coordinate_order=("x", "y"),
            manifold_coordinates=(float(idx), float(idx + 1)),
            continuity_weight=1.0,
            alignment_weight=1.0,
        )
        for idx in range(4)
    )

    paths = tuple(
        TraversalPath(
            path_id=f"p-{idx}",
            path_index=idx,
            node_ids=(f"n-{idx}", f"n-{idx+1}"),
            path_length=2,
            path_continuity_score=score,
            path_alignment_score=score,
            route_integrity_score=score,
            traversal_efficiency_score=score,
            path_score=score,
        )
        for idx, score in enumerate(path_scores)
    )

    result = ManifoldTraversalResult(
        schema_version=1,
        source_graph_hash="graph-h",
        source_polytope_hash="polytope-h",
        source_symmetry_hash="symmetry-h",
        source_replay_identity_hash="replay-h",
        node_count=len(nodes),
        path_count=len(paths),
        nodes=nodes,
        paths=paths,
        path_continuity_score=float(sum(path_scores) / len(path_scores)),
        manifold_alignment_score=1.0,
        symmetry_route_integrity_score=float(sum(path_scores) / len(path_scores)),
        traversal_efficiency_score=float(sum(path_scores) / len(path_scores)),
        overall_traversal_score=float(sum(path_scores) / len(path_scores)),
        law_invariants=("x",),
        traversal_hash="",
    )
    return replace(result, traversal_hash=result.stable_hash())


def _manual_traversal_split() -> ManifoldTraversalResult:
    """Create a traversal with node n-1 as a split node (connects to n-0, n-2, and n-3).

    Topology:
        p-0: n-0 -> n-1 -> n-2
        p-1: n-0 -> n-1 -> n-3
    n-1 has three neighbours so it is detected as a split node and gets its own scenario.
    """
    nodes = tuple(
        ManifoldNode(
            node_id=f"n-{idx}",
            node_index=idx,
            source_basis_id=f"b-{idx}",
            source_basis_index=idx,
            coordinate_order=("x", "y"),
            manifold_coordinates=(float(idx), float(idx + 1)),
            continuity_weight=1.0,
            alignment_weight=1.0,
        )
        for idx in range(4)
    )
    paths = (
        TraversalPath(
            path_id="p-0",
            path_index=0,
            node_ids=("n-0", "n-1", "n-2"),
            path_length=3,
            path_continuity_score=0.9,
            path_alignment_score=0.9,
            route_integrity_score=0.9,
            traversal_efficiency_score=0.9,
            path_score=0.9,
        ),
        TraversalPath(
            path_id="p-1",
            path_index=1,
            node_ids=("n-0", "n-1", "n-3"),
            path_length=3,
            path_continuity_score=0.8,
            path_alignment_score=0.8,
            route_integrity_score=0.8,
            traversal_efficiency_score=0.8,
            path_score=0.8,
        ),
    )
    result = ManifoldTraversalResult(
        schema_version=1,
        source_graph_hash="graph-h-split",
        source_polytope_hash="polytope-h-split",
        source_symmetry_hash="symmetry-h-split",
        source_replay_identity_hash="replay-h-split",
        node_count=4,
        path_count=2,
        nodes=nodes,
        paths=paths,
        path_continuity_score=0.85,
        manifold_alignment_score=0.85,
        symmetry_route_integrity_score=0.85,
        traversal_efficiency_score=0.85,
        overall_traversal_score=0.85,
        law_invariants=("x",),
        traversal_hash="",
    )
    return replace(result, traversal_hash=result.stable_hash())


def test_identical_input_produces_byte_identical_divergence_artifacts() -> None:
    traversal, _, _, _ = _traversal_from_observed_records((0, 2, 4))

    artifact_a = run_topology_divergence_battery(traversal)
    artifact_b = run_topology_divergence_battery(traversal)

    assert artifact_a == artifact_b
    assert artifact_a.to_canonical_bytes() == artifact_b.to_canonical_bytes()
    assert export_topology_divergence_bytes(artifact_a) == export_topology_divergence_bytes(artifact_b)


def test_deterministic_repeated_runs() -> None:
    traversal, _, _, _ = _traversal_from_observed_records((0, 1, 3))

    artifacts = [export_topology_divergence_bytes(run_topology_divergence_battery(traversal)) for _ in range(25)]
    assert len(set(artifacts)) == 1


def test_stable_scenario_ordering() -> None:
    # Use a split-node traversal so there are multiple scenarios to order.
    traversal = _manual_traversal_split()
    artifact = run_topology_divergence_battery(traversal)

    # The baseline scenario (sentinel) must always be first.
    anchor_ids = tuple(scenario.anchor_node_id for scenario in artifact.scenarios)
    assert anchor_ids[0] == "baseline"
    # Remaining scenarios are ordered by anchor_node_id (the deterministic sort key).
    assert anchor_ids[1:] == tuple(sorted(anchor_ids[1:]))
    # All scenario IDs are unique content hashes — ordering is driven by content, not insertion.
    scenario_ids = tuple(scenario.scenario_id for scenario in artifact.scenarios)
    assert len(set(scenario_ids)) == len(scenario_ids)


def test_stable_segment_ordering() -> None:
    # Each path in _manual_traversal has exactly one edge (edge_index=0 for all segments),
    # so the effective sort key is (path_id, start_node_id, end_node_id, segment_id) —
    # a fully content-derived ordering that is independent of segment_index.
    traversal = _manual_traversal((0.9, 0.8, 0.7))
    artifact = run_topology_divergence_battery(traversal)

    for scenario in artifact.scenarios:
        content_keys = tuple(
            (segment.path_id, segment.start_node_id, segment.end_node_id, segment.segment_id)
            for segment in scenario.segments
        )
        assert content_keys == tuple(sorted(content_keys))


def test_stable_hash_invariants() -> None:
    traversal = _manual_traversal((0.9, 0.9, 0.9))
    artifact = run_topology_divergence_battery(traversal)

    assert artifact.law_invariants == (
        TOPOLOGY_DIVERGENCE_LAW,
        DETERMINISTIC_SCENARIO_ORDERING_RULE,
        REPLAY_SAFE_DIVERGENCE_IDENTITY_RULE,
        BOUNDED_DIVERGENCE_SCORE_RULE,
    )
    assert artifact.stable_hash() == artifact.divergence_hash


def test_fail_fast_malformed_traversal_input() -> None:
    traversal = _manual_traversal((0.9, 0.9, 0.9))
    invalid = replace(traversal, traversal_hash="bad-hash")

    with pytest.raises(ValueError, match="traversal_hash must match stable_hash"):
        run_topology_divergence_battery(invalid)


def test_fail_fast_baseline_node_id_rejected() -> None:
    """Traversal inputs with node_id == 'baseline' must be rejected to prevent sentinel collision."""
    traversal = _manual_traversal((0.9, 0.9, 0.9))
    bad_nodes = (replace(traversal.nodes[0], node_id="baseline"),) + traversal.nodes[1:]
    invalid = replace(traversal, nodes=bad_nodes, node_count=len(bad_nodes), traversal_hash="")
    invalid = replace(invalid, traversal_hash=invalid.stable_hash())

    with pytest.raises(ValueError, match="reserved"):
        run_topology_divergence_battery(invalid)


def test_split_entropy_nonzero_on_split_node() -> None:
    """split_entropy_score must be > 0 for a scenario whose anchor is a true split node."""
    traversal = _manual_traversal_split()
    artifact = run_topology_divergence_battery(traversal)

    # The split-node scenario (anchor_node_id == "n-1") must have non-zero entropy.
    split_scenarios = [s for s in artifact.scenarios if s.anchor_node_id != "baseline"]
    assert split_scenarios, "expected at least one non-baseline scenario"
    assert all(s.split_entropy_score > 0.0 for s in split_scenarios)


def test_perfect_continuity_produces_low_divergence() -> None:
    traversal = _manual_traversal((1.0, 1.0, 1.0))
    artifact = run_topology_divergence_battery(traversal)

    assert artifact.path_fragmentation_score < 0.05
    assert artifact.overall_divergence_score < 0.2
    assert artifact.traversal_resilience_score > 0.9


def test_fragmented_topology_degrades_resilience() -> None:
    pristine = run_topology_divergence_battery(_manual_traversal((1.0, 1.0, 1.0)))
    fragmented = run_topology_divergence_battery(_manual_traversal((0.2, 0.2, 0.2)))

    assert fragmented.path_fragmentation_score > pristine.path_fragmentation_score
    assert fragmented.overall_divergence_score > pristine.overall_divergence_score
    assert fragmented.traversal_resilience_score < pristine.traversal_resilience_score


def test_monotonic_degradation_assertions() -> None:
    pristine = run_topology_divergence_battery(_manual_traversal((1.0, 1.0, 1.0)))
    moderate = run_topology_divergence_battery(_manual_traversal((0.7, 0.7, 0.7)))
    fragmented = run_topology_divergence_battery(_manual_traversal((0.3, 0.3, 0.3)))

    assert pristine.overall_divergence_score < moderate.overall_divergence_score < fragmented.overall_divergence_score
    assert pristine.traversal_resilience_score > moderate.traversal_resilience_score > fragmented.traversal_resilience_score


def test_receipt_determinism() -> None:
    traversal = _manual_traversal((0.8, 0.8, 0.8))
    artifact = run_topology_divergence_battery(traversal)

    receipt_a = generate_topology_divergence_receipt(artifact)
    receipt_b = generate_topology_divergence_receipt(artifact)

    assert receipt_a == receipt_b
    assert receipt_a.stable_hash() == receipt_a.receipt_hash
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_lineage_hash_preservation() -> None:
    traversal, symmetry, polytope, graph = _traversal_from_observed_records((0, 1, 2, 3))
    artifact = run_topology_divergence_battery(
        traversal,
        symmetry_artifact=symmetry,
        polytope_artifact=polytope,
        graph_artifact=graph,
    )

    assert artifact.source_graph_hash == graph.graph_hash
    assert artifact.source_polytope_hash == polytope.polytope_hash
    assert artifact.source_symmetry_hash == symmetry.symmetry_hash
    assert artifact.source_traversal_hash == traversal.traversal_hash
