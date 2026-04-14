"""v137.16.3 — Tests for replay-safe reasoning graph.

Deterministic replay coverage for normalization, build, validation,
traversal, and canonical export stability.
"""

from __future__ import annotations

import hashlib

import pytest

from qec.memory.replay_safe_reasoning_graph import (
    ReasoningGraphError,
    build_replay_safe_reasoning_graph,
    normalize_replay_safe_reasoning_input,
    traverse_replay_safe_reasoning_graph,
    validate_replay_safe_reasoning_graph,
)


def _lh(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _base_payload():
    return {
        "graph_id": "reasoning.v137.16.3",
        "memory_graph": {"graph_id": "mem.v137.16.0"},
        "decision_dag": {"dag_id": "dag.v137.16.1"},
        "topology_index": {"index_id": "topo.v137.16.2"},
        "release_traces": [{"trace_id": "rel.v137.16.3"}],
        "proof_traces": [{"trace_id": "proof.kernel"}],
        "test_traces": [{"trace_id": "test.reasoning"}],
        "artifact_traces": [{"trace_id": "artifact.trace"}],
        "nodes": [
            {
                "node_id": "n-memory",
                "reasoning_kind": "memory",
                "source_ref": "memory_graph:mem.v137.16.0",
                "lineage_hash": _lh("memory"),
                "reasoning_epoch": 0,
            },
            {
                "node_id": "n-decision",
                "reasoning_kind": "decision",
                "source_ref": "decision_dag:dag.v137.16.1",
                "lineage_hash": _lh("decision"),
                "reasoning_epoch": 1,
            },
            {
                "node_id": "n-topology",
                "reasoning_kind": "topology",
                "source_ref": "topology_index:topo.v137.16.2",
                "lineage_hash": _lh("topology"),
                "reasoning_epoch": 2,
            },
            {
                "node_id": "n-proof",
                "reasoning_kind": "proof",
                "source_ref": "proof:proof.kernel",
                "lineage_hash": _lh("proof"),
                "reasoning_epoch": 3,
            },
            {
                "node_id": "n-test",
                "reasoning_kind": "test",
                "source_ref": "test:test.reasoning",
                "lineage_hash": _lh("test"),
                "reasoning_epoch": 4,
            },
            {
                "node_id": "n-release",
                "reasoning_kind": "release",
                "source_ref": "release:rel.v137.16.3",
                "lineage_hash": _lh("release"),
                "reasoning_epoch": 5,
            },
            {
                "node_id": "n-artifact",
                "reasoning_kind": "artifact",
                "source_ref": "artifact:artifact.trace",
                "lineage_hash": _lh("artifact"),
                "reasoning_epoch": 6,
            },
        ],
        "edges": [
            {
                "edge_id": "e-support",
                "source_node_id": "n-memory",
                "target_node_id": "n-decision",
                "reasoning_relation": "supports",
                "edge_weight": 1.0,
                "reasoning_epoch": 1,
            },
            {
                "edge_id": "e-dep",
                "source_node_id": "n-decision",
                "target_node_id": "n-topology",
                "reasoning_relation": "depends_on",
                "edge_weight": 1.0,
                "reasoning_epoch": 2,
            },
            {
                "edge_id": "e-verify",
                "source_node_id": "n-proof",
                "target_node_id": "n-test",
                "reasoning_relation": "verifies",
                "edge_weight": 1.0,
                "reasoning_epoch": 3,
            },
            {
                "edge_id": "e-index",
                "source_node_id": "n-topology",
                "target_node_id": "n-artifact",
                "reasoning_relation": "indexes",
                "edge_weight": 1.0,
                "reasoning_epoch": 4,
            },
            {
                "edge_id": "e-replay",
                "source_node_id": "n-release",
                "target_node_id": "n-artifact",
                "reasoning_relation": "replays",
                "edge_weight": 1.0,
                "reasoning_epoch": 5,
            },
            {
                "edge_id": "e-derive",
                "source_node_id": "n-release",
                "target_node_id": "n-memory",
                "reasoning_relation": "derives_from",
                "edge_weight": 1.0,
                "reasoning_epoch": 6,
            },
        ],
    }


def test_repeated_run_byte_identity():
    payload = _base_payload()
    g1 = build_replay_safe_reasoning_graph(payload)
    g2 = build_replay_safe_reasoning_graph(payload)
    assert g1.to_canonical_bytes() == g2.to_canonical_bytes()


def test_repeated_run_reasoning_hash_identity():
    payload = _base_payload()
    hashes = {build_replay_safe_reasoning_graph(payload).reasoning_hash for _ in range(5)}
    assert len(hashes) == 1


def test_duplicate_node_rejection():
    payload = _base_payload()
    payload["nodes"].append(dict(payload["nodes"][0]))
    with pytest.raises(ReasoningGraphError, match="duplicate reasoning node id"):
        normalize_replay_safe_reasoning_input(payload)


def test_duplicate_edge_rejection():
    payload = _base_payload()
    payload["edges"].append(dict(payload["edges"][0]))
    with pytest.raises(ReasoningGraphError, match="duplicate reasoning edge id"):
        normalize_replay_safe_reasoning_input(payload)


def test_invalid_cross_artifact_reference_rejection():
    payload = _base_payload()
    payload["nodes"][0]["source_ref"] = "memory_graph:missing"
    with pytest.raises(ReasoningGraphError, match="invalid cross-artifact reference"):
        normalize_replay_safe_reasoning_input(payload)


def test_invalid_reasoning_kind_rejection():
    payload = _base_payload()
    payload["nodes"][0]["reasoning_kind"] = "unknown"
    with pytest.raises(ReasoningGraphError, match="unsupported reasoning kind"):
        normalize_replay_safe_reasoning_input(payload)


def test_invalid_relation_rejection():
    payload = _base_payload()
    payload["edges"][0]["reasoning_relation"] = "unknown"
    with pytest.raises(ReasoningGraphError, match="unsupported relation"):
        normalize_replay_safe_reasoning_input(payload)


def test_deterministic_reasoning_traversal():
    graph = build_replay_safe_reasoning_graph(_base_payload())
    a = traverse_replay_safe_reasoning_graph(graph, "reasoning")
    b = traverse_replay_safe_reasoning_graph(graph, "reasoning")
    assert a.visited_nodes == b.visited_nodes
    assert a.visited_edges == b.visited_edges
    assert a.traversal_hash == b.traversal_hash
    assert a.to_canonical_bytes() == b.to_canonical_bytes()


def test_deterministic_dependency_traversal():
    graph = build_replay_safe_reasoning_graph(_base_payload())
    a = traverse_replay_safe_reasoning_graph(graph, "dependency")
    b = traverse_replay_safe_reasoning_graph(graph, "dependency")
    assert a.visited_nodes == b.visited_nodes
    assert a.visited_edges == b.visited_edges
    assert a.traversal_hash == b.traversal_hash
    assert a.to_canonical_bytes() == b.to_canonical_bytes()


def test_deterministic_verification_traversal():
    graph = build_replay_safe_reasoning_graph(_base_payload())
    a = traverse_replay_safe_reasoning_graph(graph, "verification")
    b = traverse_replay_safe_reasoning_graph(graph, "verification")
    assert a.visited_nodes == b.visited_nodes
    assert a.visited_edges == b.visited_edges
    assert a.traversal_hash == b.traversal_hash
    assert a.to_canonical_bytes() == b.to_canonical_bytes()


def test_deterministic_replay_traversal():
    graph = build_replay_safe_reasoning_graph(_base_payload())
    a = traverse_replay_safe_reasoning_graph(graph, "replay")
    b = traverse_replay_safe_reasoning_graph(graph, "replay")
    assert a.visited_nodes == b.visited_nodes
    assert a.visited_edges == b.visited_edges
    assert a.traversal_hash == b.traversal_hash
    assert a.to_canonical_bytes() == b.to_canonical_bytes()


def test_canonical_export_stability():
    graph = build_replay_safe_reasoning_graph(_base_payload())
    assert graph.to_canonical_json() == graph.to_canonical_json()
    assert graph.to_canonical_bytes() == graph.to_canonical_bytes()


def test_validation_report_flags_invalid_reference():
    payload = _base_payload()
    payload["nodes"][0]["source_ref"] = "memory_graph:missing"
    report = validate_replay_safe_reasoning_graph(payload)
    assert report.is_valid is False
    assert report.cross_artifact_reference_validity_ok is False


def test_validation_report_accepts_built_graph_object():
    graph = build_replay_safe_reasoning_graph(_base_payload())
    report = validate_replay_safe_reasoning_graph(graph)
    assert report.is_valid is True
    assert report.violations == ()
