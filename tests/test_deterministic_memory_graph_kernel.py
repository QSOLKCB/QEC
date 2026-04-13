"""v137.16.0 — Tests for the Deterministic Memory Graph Kernel.

Byte-for-byte replay tests covering graph normalization, construction,
validation, and deterministic traversal. These tests must pass identically
on repeated execution.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qec.memory.deterministic_memory_graph_kernel import (
    DeterministicMemoryGraph,
    ERR_DUPLICATE_EDGE,
    ERR_DUPLICATE_NODE,
    ERR_EDGE_KIND_INVALID,
    ERR_EDGE_REF_SOURCE_MISSING,
    ERR_EDGE_REF_TARGET_MISSING,
    ERR_EDGE_WEIGHT_INVALID,
    ERR_EDGE_WEIGHT_NEGATIVE,
    ERR_LINEAGE_INVALID,
    ERR_NODE_KIND_INVALID,
    ERR_PAYLOAD_INVALID,
    GraphValidationError,
    MemoryGraphEdge,
    MemoryGraphExecutionReceipt,
    MemoryGraphNode,
    MemoryGraphValidationReport,
    build_deterministic_memory_graph,
    normalize_deterministic_memory_graph,
    traverse_deterministic_memory_graph,
    validate_deterministic_memory_graph,
)


def _lh(ch: str) -> str:
    return ch * 64


def _node(
    node_id: str,
    node_kind: str = "module",
    creation_epoch: int = 0,
    payload=None,
    lineage_hash=None,
):
    return {
        "node_id": node_id,
        "node_kind": node_kind,
        "node_payload": payload if payload is not None else {"name": node_id},
        "lineage_hash": lineage_hash if lineage_hash is not None else _lh("a"),
        "creation_epoch": creation_epoch,
    }


def _edge(
    edge_id: str,
    source: str,
    target: str,
    edge_kind: str = "depends_on",
    edge_weight: float = 1.0,
    creation_epoch: int = 1,
):
    return {
        "edge_id": edge_id,
        "source_node_id": source,
        "target_node_id": target,
        "edge_kind": edge_kind,
        "edge_weight": edge_weight,
        "creation_epoch": creation_epoch,
    }


def _simple_graph_input():
    return {
        "graph_id": "g-simple",
        "nodes": [
            _node("n-a", "release", 0, {"version": "137.16.0"}, _lh("1")),
            _node("n-b", "module", 1, {"path": "src/qec/memory"}, _lh("2")),
            _node("n-c", "proof", 2, {"kind": "state_safety"}, _lh("3")),
            _node("n-d", "test", 3, {"name": "t_memory"}, _lh("4")),
        ],
        "edges": [
            _edge("e-ab", "n-a", "n-b", "depends_on", 1.0, 1),
            _edge("e-bc", "n-b", "n-c", "verified_by", 2.0, 2),
            _edge("e-cd", "n-c", "n-d", "tests", 1.5, 3),
            _edge("e-ad", "n-a", "n-d", "derived_from", 0.5, 4),
        ],
    }


# --------------------------------------------------------------------------
# Repeated-run byte identity / hash identity
# --------------------------------------------------------------------------


def test_repeated_build_byte_identity():
    raw = _simple_graph_input()
    g1 = build_deterministic_memory_graph(raw)
    g2 = build_deterministic_memory_graph(raw)
    assert g1.to_canonical_bytes() == g2.to_canonical_bytes()
    assert g1.to_canonical_json() == g2.to_canonical_json()
    assert g1.as_hash_payload() == g2.as_hash_payload()


def test_repeated_build_graph_hash_identity():
    raw = _simple_graph_input()
    hashes = {
        build_deterministic_memory_graph(raw).graph_hash for _ in range(5)
    }
    assert len(hashes) == 1


def test_build_is_order_independent():
    raw = _simple_graph_input()
    shuffled = dict(raw)
    shuffled["nodes"] = list(reversed(raw["nodes"]))
    shuffled["edges"] = list(reversed(raw["edges"]))
    a = build_deterministic_memory_graph(raw)
    b = build_deterministic_memory_graph(shuffled)
    assert a.graph_hash == b.graph_hash
    assert a.to_canonical_bytes() == b.to_canonical_bytes()


def test_node_and_edge_canonical_exports():
    raw = _simple_graph_input()
    g = build_deterministic_memory_graph(raw)
    for node in g.nodes:
        assert isinstance(node, MemoryGraphNode)
        assert node.to_canonical_bytes() == node.to_canonical_bytes()
        assert (
            node.to_canonical_json()
            == node.to_canonical_json()
        )
    for edge in g.edges:
        assert isinstance(edge, MemoryGraphEdge)
        assert edge.to_canonical_bytes() == edge.to_canonical_bytes()


def test_dataclasses_are_frozen():
    raw = _simple_graph_input()
    g = build_deterministic_memory_graph(raw)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        g.nodes[0].node_id = "mutated"  # type: ignore[misc]
    with pytest.raises((FrozenInstanceError, AttributeError)):
        g.edges[0].edge_weight = 99.0  # type: ignore[misc]
    with pytest.raises((FrozenInstanceError, AttributeError)):
        g.graph_hash = "mutated"  # type: ignore[misc]


# --------------------------------------------------------------------------
# Rejection / fail-fast law
# --------------------------------------------------------------------------


def test_duplicate_node_rejection():
    raw = _simple_graph_input()
    raw["nodes"].append(_node("n-a", "module", 5))
    with pytest.raises(ValueError, match="duplicate node id"):
        build_deterministic_memory_graph(raw)


def test_duplicate_edge_rejection():
    raw = _simple_graph_input()
    raw["edges"].append(_edge("e-ab", "n-b", "n-c", "depends_on", 1.0, 1))
    with pytest.raises(ValueError, match="duplicate edge id"):
        build_deterministic_memory_graph(raw)


def test_invalid_edge_reference_source():
    raw = _simple_graph_input()
    raw["edges"].append(
        _edge("e-missing", "n-missing", "n-b", "depends_on", 1.0, 1)
    )
    with pytest.raises(ValueError, match="missing source node"):
        build_deterministic_memory_graph(raw)


def test_invalid_edge_reference_target():
    raw = _simple_graph_input()
    raw["edges"].append(
        _edge("e-missing", "n-a", "n-nope", "depends_on", 1.0, 1)
    )
    with pytest.raises(ValueError, match="missing target node"):
        build_deterministic_memory_graph(raw)


def test_invalid_node_kind_rejection():
    raw = _simple_graph_input()
    raw["nodes"][0]["node_kind"] = "wizard"
    with pytest.raises(ValueError, match="unsupported node kind"):
        build_deterministic_memory_graph(raw)


def test_invalid_edge_kind_rejection():
    raw = _simple_graph_input()
    raw["edges"][0]["edge_kind"] = "haunts"
    with pytest.raises(ValueError, match="unsupported edge kind"):
        build_deterministic_memory_graph(raw)


def test_malformed_payload_rejection():
    raw = _simple_graph_input()
    raw["nodes"][0]["node_payload"] = "not-a-mapping"
    with pytest.raises(ValueError, match="malformed payload"):
        build_deterministic_memory_graph(raw)


def test_negative_weight_rejection():
    raw = _simple_graph_input()
    raw["edges"][0]["edge_weight"] = -0.1
    with pytest.raises(ValueError, match="negative edge weight"):
        build_deterministic_memory_graph(raw)


def test_malformed_lineage_hash_rejection():
    raw = _simple_graph_input()
    raw["nodes"][0]["lineage_hash"] = "not-a-hash"
    with pytest.raises(ValueError, match="malformed lineage hash"):
        build_deterministic_memory_graph(raw)


def test_nan_payload_value_rejection():
    raw = _simple_graph_input()
    raw["nodes"][0]["node_payload"] = {"bad": float("nan")}
    with pytest.raises(ValueError, match="malformed payload"):
        build_deterministic_memory_graph(raw)


def test_missing_top_level_fields():
    with pytest.raises(ValueError, match="missing required graph fields"):
        build_deterministic_memory_graph({"graph_id": "x", "nodes": []})


# --------------------------------------------------------------------------
# Validation report law
# --------------------------------------------------------------------------


def test_validation_report_valid_graph():
    raw = _simple_graph_input()
    report = validate_deterministic_memory_graph(raw)
    assert isinstance(report, MemoryGraphValidationReport)
    assert report.is_valid is True
    assert report.node_count == 4
    assert report.edge_count == 4
    assert report.violations == ()
    # Determinism: report bytes are stable
    again = validate_deterministic_memory_graph(raw)
    assert report.to_canonical_bytes() == again.to_canonical_bytes()


def test_validation_report_captures_multiple_failures():
    raw = _simple_graph_input()
    raw["nodes"][1]["lineage_hash"] = "nope"  # n-b has bad lineage
    raw["edges"][1]["edge_weight"] = -2.0  # e-bc has negative weight
    # duplicate edge id on a well-formed pair (both endpoints still present)
    raw["edges"].append(
        _edge("e-cd", "n-a", "n-d", "depends_on", 1.0, 5)
    )
    report = validate_deterministic_memory_graph(raw)
    assert report.is_valid is False
    assert report.lineage_validity_ok is False
    assert report.weight_validity_ok is False
    assert report.uniqueness_ok is False
    assert any("duplicate edge id" in v for v in report.violations)
    assert any("lineage" in v for v in report.violations)
    assert any("weight" in v for v in report.violations)


# --------------------------------------------------------------------------
# Traversal determinism
# --------------------------------------------------------------------------


def _lineage_graph_input():
    return {
        "graph_id": "g-lineage",
        "nodes": [
            _node("r0", "release", 0, {"v": 0}, _lh("a")),
            _node("r1", "release", 1, {"v": 1}, _lh("b")),
            _node("r2", "release", 2, {"v": 2}, _lh("c")),
            _node("m1", "module", 1, {"m": 1}, _lh("d")),
            _node("p1", "proof", 2, {"p": 1}, _lh("e")),
        ],
        "edges": [
            # lineage chain
            _edge("l01", "r1", "r0", "derived_from", 1.0, 1),
            _edge("l12", "r2", "r1", "supersedes", 1.0, 2),
            # non-lineage edges to confirm they are filtered out
            _edge("d1", "r1", "m1", "depends_on", 1.0, 1),
            _edge("v1", "r1", "p1", "verified_by", 1.0, 2),
        ],
    }


def test_traversal_bfs_deterministic_order():
    raw = _simple_graph_input()
    g = build_deterministic_memory_graph(raw)
    r1 = traverse_deterministic_memory_graph(g, "n-a", "bfs")
    r2 = traverse_deterministic_memory_graph(g, "n-a", "bfs")
    assert r1.visited_nodes == r2.visited_nodes
    assert r1.visited_edges == r2.visited_edges
    assert r1.traversal_hash == r2.traversal_hash
    # n-a first, then n-b (epoch 1), then n-d (epoch 4 via derived_from from
    # n-a), then n-c (reached from n-b via verified_by).
    assert r1.visited_nodes[0] == "n-a"
    assert set(r1.visited_nodes) == {"n-a", "n-b", "n-c", "n-d"}
    assert len(r1.visited_edges) == 3  # 4 nodes reached via 3 frontier edges


def test_traversal_dfs_deterministic_order():
    raw = _simple_graph_input()
    g = build_deterministic_memory_graph(raw)
    r1 = traverse_deterministic_memory_graph(g, "n-a", "dfs")
    r2 = traverse_deterministic_memory_graph(g, "n-a", "dfs")
    assert r1.visited_nodes == r2.visited_nodes
    assert r1.visited_edges == r2.visited_edges
    assert r1.traversal_hash == r2.traversal_hash
    assert r1.visited_nodes[0] == "n-a"
    # DFS should produce identical trace under repeated calls
    assert len({
        traverse_deterministic_memory_graph(g, "n-a", "dfs").traversal_hash
        for _ in range(5)
    }) == 1


def test_traversal_lineage_filters_edge_kinds():
    raw = _lineage_graph_input()
    g = build_deterministic_memory_graph(raw)
    r = traverse_deterministic_memory_graph(g, "r2", "lineage")
    assert r.visited_nodes == ("r2", "r1", "r0")
    # Only lineage edges (supersedes, derived_from) should be traversed.
    assert set(r.visited_edges) == {"l12", "l01"}
    # Repeated lineage traversal is byte-stable
    r2 = traverse_deterministic_memory_graph(g, "r2", "lineage")
    assert r.to_canonical_bytes() == r2.to_canonical_bytes()


def test_traversal_dependency_filters_edge_kinds():
    raw = _lineage_graph_input()
    g = build_deterministic_memory_graph(raw)
    r = traverse_deterministic_memory_graph(g, "r1", "dependency")
    # Only depends_on edges: r1 -> m1.
    assert r.visited_nodes == ("r1", "m1")
    assert r.visited_edges == ("d1",)


def test_traversal_rejects_unknown_mode():
    raw = _simple_graph_input()
    g = build_deterministic_memory_graph(raw)
    with pytest.raises(ValueError, match="unsupported traversal mode"):
        traverse_deterministic_memory_graph(g, "n-a", "astral")


def test_traversal_rejects_unknown_start():
    raw = _simple_graph_input()
    g = build_deterministic_memory_graph(raw)
    with pytest.raises(ValueError, match="start node not in graph"):
        traverse_deterministic_memory_graph(g, "n-missing", "bfs")


def test_traversal_receipt_is_frozen_dataclass():
    raw = _simple_graph_input()
    g = build_deterministic_memory_graph(raw)
    r = traverse_deterministic_memory_graph(g, "n-a", "bfs")
    assert isinstance(r, MemoryGraphExecutionReceipt)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        r.traversal_hash = "mutated"  # type: ignore[misc]
    with pytest.raises((FrozenInstanceError, AttributeError)):
        r.visited_nodes = ()  # type: ignore[misc]


# --------------------------------------------------------------------------
# Canonical export stability
# --------------------------------------------------------------------------


def test_canonical_export_stability_graph_and_receipt():
    raw = _simple_graph_input()
    g = build_deterministic_memory_graph(raw)
    r = traverse_deterministic_memory_graph(g, "n-a", "bfs")
    # Byte-stable across many re-exports
    graph_bytes = {g.to_canonical_bytes() for _ in range(5)}
    receipt_bytes = {r.to_canonical_bytes() for _ in range(5)}
    assert len(graph_bytes) == 1
    assert len(receipt_bytes) == 1
    # Hash payload equals canonical bytes
    assert g.as_hash_payload() == g.to_canonical_bytes()
    assert r.as_hash_payload() == r.to_canonical_bytes()


def test_normalize_returns_canonical_ordering():
    raw = _simple_graph_input()
    graph_id, nodes, edges = normalize_deterministic_memory_graph(raw)
    assert graph_id == "g-simple"
    # Nodes sorted by (creation_epoch, node_kind, node_id)
    assert [n.node_id for n in nodes] == ["n-a", "n-b", "n-c", "n-d"]
    # Edges sorted by (creation_epoch, edge_kind, edge_id)
    assert [e.edge_id for e in edges] == ["e-ab", "e-bc", "e-cd", "e-ad"]


def test_build_accepts_existing_graph_instance():
    raw = _simple_graph_input()
    g1 = build_deterministic_memory_graph(raw)
    g2 = build_deterministic_memory_graph(g1)
    assert g1.graph_hash == g2.graph_hash
    assert g1.to_canonical_bytes() == g2.to_canonical_bytes()


# --------------------------------------------------------------------------
# Hardening regression tests — structured error codes, stable payload
# context, validation-flag correctness, and FrozenInstanceError-based
# immutability.
# --------------------------------------------------------------------------


def _raise_code(raw) -> str:
    """Run build + normalize and return the GraphValidationError code."""
    with pytest.raises(GraphValidationError) as excinfo:
        build_deterministic_memory_graph(raw)
    return excinfo.value.code


def test_structured_error_code_duplicate_node():
    raw = _simple_graph_input()
    raw["nodes"].append(_node("n-a", "module", 5))
    assert _raise_code(raw) == ERR_DUPLICATE_NODE


def test_structured_error_code_duplicate_edge():
    raw = _simple_graph_input()
    raw["edges"].append(_edge("e-ab", "n-b", "n-c", "depends_on", 1.0, 1))
    assert _raise_code(raw) == ERR_DUPLICATE_EDGE


def test_structured_error_code_missing_source_ref():
    raw = _simple_graph_input()
    raw["edges"].append(
        _edge("e-missing-src", "n-missing", "n-b", "depends_on", 1.0, 1)
    )
    assert _raise_code(raw) == ERR_EDGE_REF_SOURCE_MISSING


def test_structured_error_code_missing_target_ref():
    raw = _simple_graph_input()
    raw["edges"].append(
        _edge("e-missing-tgt", "n-a", "n-missing", "depends_on", 1.0, 1)
    )
    assert _raise_code(raw) == ERR_EDGE_REF_TARGET_MISSING


def test_structured_error_code_node_kind_invalid():
    raw = _simple_graph_input()
    raw["nodes"][0]["node_kind"] = "wizard"
    assert _raise_code(raw) == ERR_NODE_KIND_INVALID


def test_structured_error_code_edge_kind_invalid():
    raw = _simple_graph_input()
    raw["edges"][0]["edge_kind"] = "haunts"
    assert _raise_code(raw) == ERR_EDGE_KIND_INVALID


def test_structured_error_code_lineage_invalid():
    raw = _simple_graph_input()
    raw["nodes"][0]["lineage_hash"] = "not-a-hash"
    assert _raise_code(raw) == ERR_LINEAGE_INVALID


def test_structured_error_code_negative_weight():
    raw = _simple_graph_input()
    raw["edges"][0]["edge_weight"] = -3.5
    assert _raise_code(raw) == ERR_EDGE_WEIGHT_NEGATIVE


def test_structured_error_code_weight_invalid():
    raw = _simple_graph_input()
    raw["edges"][0]["edge_weight"] = "not-a-number"
    assert _raise_code(raw) == ERR_EDGE_WEIGHT_INVALID


def test_structured_error_code_payload_invalid():
    raw = _simple_graph_input()
    raw["nodes"][0]["node_payload"] = "not-a-mapping"
    assert _raise_code(raw) == ERR_PAYLOAD_INVALID


def test_payload_error_path_context_root():
    raw = _simple_graph_input()
    raw["nodes"][0]["node_payload"] = "not-a-mapping"
    with pytest.raises(GraphValidationError) as excinfo:
        build_deterministic_memory_graph(raw)
    msg = str(excinfo.value)
    assert "malformed payload in node n-a at path=node_payload" in msg
    assert excinfo.value.code == ERR_PAYLOAD_INVALID


def test_payload_error_path_context_nested_mapping():
    raw = _simple_graph_input()
    raw["nodes"][0]["node_payload"] = {
        "metadata": {"hash": float("nan")}
    }
    with pytest.raises(GraphValidationError) as excinfo:
        build_deterministic_memory_graph(raw)
    msg = str(excinfo.value)
    assert (
        "malformed payload in node n-a at path=node_payload.metadata.hash"
        in msg
    )


def test_payload_error_path_context_list_index():
    raw = _simple_graph_input()
    raw["nodes"][0]["node_payload"] = {"values": [1, object(), 3]}
    with pytest.raises(GraphValidationError) as excinfo:
        build_deterministic_memory_graph(raw)
    msg = str(excinfo.value)
    # The list index path token is deterministic and human-readable.
    assert (
        "malformed payload in node n-a at path=node_payload.values.[1]"
        in msg
    )


def test_payload_error_path_is_deterministic_across_runs():
    raw = _simple_graph_input()
    raw["nodes"][0]["node_payload"] = {
        "metadata": {"hash": float("inf")}
    }
    msgs = set()
    for _ in range(5):
        with pytest.raises(GraphValidationError) as excinfo:
            build_deterministic_memory_graph(raw)
        msgs.add(str(excinfo.value))
    assert len(msgs) == 1


def test_validation_report_duplicate_node_flag_correctness():
    # Build a valid base graph, then append a duplicate of n-a with a
    # different epoch so the second _normalize_node call still succeeds
    # and the duplicate is detected by the set check.
    raw = _simple_graph_input()
    raw["nodes"].append(_node("n-a", "module", 9))
    report = validate_deterministic_memory_graph(raw)
    assert report.is_valid is False
    assert report.uniqueness_ok is False
    # Other channels must stay green — the flag classifier must not flip
    # unrelated report channels.
    assert report.node_validity_ok is True
    assert report.edge_validity_ok is True
    assert report.payload_validity_ok is True
    assert report.lineage_validity_ok is True
    assert report.weight_validity_ok is True
    assert any("duplicate node id: n-a" in v for v in report.violations)


def test_validation_report_invalid_edge_reference_flag_correctness():
    raw = _simple_graph_input()
    raw["edges"].append(
        _edge("e-orphan", "n-a", "n-ghost", "depends_on", 1.0, 9)
    )
    report = validate_deterministic_memory_graph(raw)
    assert report.is_valid is False
    assert report.edge_validity_ok is False
    # Orthogonal channels remain green.
    assert report.uniqueness_ok is True
    assert report.node_validity_ok is True
    assert report.payload_validity_ok is True
    assert report.lineage_validity_ok is True
    assert report.weight_validity_ok is True
    assert any(
        "e-orphan references missing target node n-ghost" in v
        for v in report.violations
    )


def test_validation_report_payload_channel_isolation():
    # Corrupt an isolated node's payload (n-iso is not referenced by any
    # edge) so the payload failure cannot cascade into edge_validity_ok.
    raw = _simple_graph_input()
    raw["nodes"].append(
        _node("n-iso", "module", 7, {"bad": float("nan")}, _lh("9"))
    )
    report = validate_deterministic_memory_graph(raw)
    assert report.is_valid is False
    assert report.payload_validity_ok is False
    # Orthogonal channels must remain green.
    assert report.lineage_validity_ok is True
    assert report.weight_validity_ok is True
    assert report.node_validity_ok is True
    assert report.edge_validity_ok is True
    assert report.uniqueness_ok is True


def test_validation_report_bytes_stable_across_runs():
    raw = _simple_graph_input()
    raw["nodes"][0]["lineage_hash"] = "nope"
    raw["edges"][0]["edge_weight"] = -9.0
    bytes_set = {
        validate_deterministic_memory_graph(raw).to_canonical_bytes()
        for _ in range(5)
    }
    assert len(bytes_set) == 1


def test_graph_validation_error_is_value_error_subclass():
    # Back-compat: callers catching ValueError must still work.
    raw = _simple_graph_input()
    raw["nodes"][0]["node_kind"] = "wizard"
    with pytest.raises(ValueError):
        build_deterministic_memory_graph(raw)


def test_immutability_node_frozen_instance_error():
    raw = _simple_graph_input()
    g = build_deterministic_memory_graph(raw)
    with pytest.raises(FrozenInstanceError):
        g.nodes[0].node_id = "mutated"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        g.nodes[0].node_kind = "release"  # type: ignore[misc]


def test_immutability_edge_frozen_instance_error():
    raw = _simple_graph_input()
    g = build_deterministic_memory_graph(raw)
    with pytest.raises(FrozenInstanceError):
        g.edges[0].edge_weight = 999.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        g.edges[0].source_node_id = "x"  # type: ignore[misc]


def test_immutability_receipt_frozen_instance_error():
    raw = _simple_graph_input()
    g = build_deterministic_memory_graph(raw)
    r = traverse_deterministic_memory_graph(g, "n-a", "bfs")
    with pytest.raises(FrozenInstanceError):
        r.traversal_hash = "mutated"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        r.visited_nodes = ()  # type: ignore[misc]
