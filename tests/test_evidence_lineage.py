from __future__ import annotations

import pytest

from qec.analysis.evidence_lineage import (
    build_evidence_receipt,
    compile_evidence_graph,
    normalize_evidence_graph,
    stable_evidence_hash,
)


def _raw_graph() -> dict[str, object]:
    return {
        "claim_id": "claim-001",
        "experiment_hash": "exp-hash-001",
        "nodes": [
            {
                "node_id": "m-2",
                "node_type": "measurement",
                "source_hash": "src-m2",
                "source_kind": "measurement_log",
                "metadata": {"z": 1, "a": 2},
                "linked_experiment_hash": "exp-hash-001",
                "linked_measurement_ids": ["m-1", "m-2"],
            },
            {
                "node_id": "claim-1",
                "node_type": "claim",
                "source_hash": "src-claim",
                "source_kind": "claim_doc",
                "metadata": {"title": "A"},
                "linked_experiment_hash": "",
                "linked_measurement_ids": [],
            },
            {
                "node_id": "m-1",
                "node_type": "measurement",
                "source_hash": "src-m1",
                "source_kind": "measurement_log",
                "metadata": {"k": "v"},
                "linked_experiment_hash": "exp-hash-001",
                "linked_measurement_ids": ["m-1"],
            },
            {
                "node_id": "exp-1",
                "node_type": "experiment",
                "source_hash": "src-exp",
                "source_kind": "experiment_spec",
                "metadata": {"batch": 7},
                "linked_experiment_hash": "exp-hash-001",
                "linked_measurement_ids": [],
            },
            {
                "node_id": "criterion-1",
                "node_type": "criterion",
                "source_hash": "src-c",
                "source_kind": "criteria_doc",
                "metadata": {"min": 0.9},
                "linked_experiment_hash": "exp-hash-001",
                "linked_measurement_ids": [],
            },
            {
                "node_id": "result-1",
                "node_type": "result",
                "source_hash": "src-r",
                "source_kind": "result_artifact",
                "metadata": {"pass": True},
                "linked_experiment_hash": "exp-hash-001",
                "linked_measurement_ids": [],
            },
            {
                "node_id": "receipt-1",
                "node_type": "receipt",
                "source_hash": "src-receipt",
                "source_kind": "receipt",
                "metadata": {"version": 1},
                "linked_experiment_hash": "exp-hash-001",
                "linked_measurement_ids": [],
            },
        ],
        "edges": [
            {"from_node_id": "result-1", "to_node_id": "exp-1", "relation_type": "produced_by"},
            {"from_node_id": "m-1", "to_node_id": "exp-1", "relation_type": "derives_from"},
            {"from_node_id": "claim-1", "to_node_id": "exp-1", "relation_type": "supports"},
            {"from_node_id": "result-1", "to_node_id": "criterion-1", "relation_type": "validates"},
            {"from_node_id": "receipt-1", "to_node_id": "result-1", "relation_type": "linked_to"},
        ],
    }


def test_deterministic_graph_hash_stability() -> None:
    raw = _raw_graph()
    graph_a, _ = compile_evidence_graph(raw)
    graph_b, _ = compile_evidence_graph(raw)
    assert stable_evidence_hash(graph_a) == stable_evidence_hash(graph_b)


def test_canonical_ordering() -> None:
    graph, _ = compile_evidence_graph(_raw_graph())
    node_keys = tuple((n.node_type, n.node_id) for n in graph.nodes)
    edge_keys = tuple((e.from_node_id, e.to_node_id, e.relation_type) for e in graph.edges)
    assert node_keys == tuple(sorted(node_keys))
    assert edge_keys == tuple(sorted(edge_keys))


def test_duplicate_node_rejection() -> None:
    raw = _raw_graph()
    raw["nodes"] = list(raw["nodes"]) + [dict(raw["nodes"][0])]  # type: ignore[index]
    with pytest.raises(ValueError, match="duplicate node IDs"):
        compile_evidence_graph(raw)


def test_duplicate_edge_rejection() -> None:
    raw = _raw_graph()
    raw["edges"] = list(raw["edges"]) + [dict(raw["edges"][0])]  # type: ignore[index]
    with pytest.raises(ValueError, match="duplicate edges"):
        compile_evidence_graph(raw)


def test_unknown_references_rejection() -> None:
    raw = _raw_graph()
    bad = dict(raw["edges"][0])  # type: ignore[index]
    bad["to_node_id"] = "missing-node"
    raw["edges"] = [bad]
    with pytest.raises(ValueError, match="unknown node IDs"):
        compile_evidence_graph(raw)


def test_invalid_edge_type_rejection() -> None:
    raw = _raw_graph()
    bad = dict(raw["edges"][0])  # type: ignore[index]
    bad["relation_type"] = "invalid"
    raw["edges"] = [bad]
    with pytest.raises(ValueError, match="unsupported edge type"):
        compile_evidence_graph(raw)


def test_receipt_stability() -> None:
    graph, _ = compile_evidence_graph(_raw_graph())
    receipt_a = build_evidence_receipt(graph)
    receipt_b = build_evidence_receipt(graph)
    assert receipt_a == receipt_b
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_same_input_same_bytes() -> None:
    graph_a, _ = compile_evidence_graph(_raw_graph())
    graph_b, _ = compile_evidence_graph(_raw_graph())
    assert graph_a.to_canonical_bytes() == graph_b.to_canonical_bytes()


def test_caller_mutation_defensive_copy_behavior() -> None:
    raw = _raw_graph()
    graph = normalize_evidence_graph(raw)
    raw["nodes"][0]["metadata"]["z"] = 999  # type: ignore[index]
    mutated_node = next(node for node in graph.nodes if node.node_id == "m-2")
    assert mutated_node.metadata["z"] == 1
