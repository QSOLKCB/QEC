"""v137.16.1 — Tests for the Decision DAG Compiler.

Byte-for-byte replay tests covering DAG normalization, cycle rejection,
deterministic topological ordering, traversal modes, and canonical export
stability. These tests must pass identically on repeated execution.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qec.memory.decision_dag_compiler import (
    CompiledDecisionDAG,
    DecisionDAGEdge,
    DecisionDAGError,
    DecisionDAGExecutionReceipt,
    DecisionDAGNode,
    DecisionDAGValidationReport,
    ERR_DAG_CYCLE,
    ERR_DAG_FIELDS_MISSING,
    ERR_DAG_ID_INVALID,
    ERR_DAG_TYPE_INVALID,
    ERR_TRAVERSAL_CYCLE_STATE_INVALID,
    ERR_TRAVERSAL_DAG_TYPE_INVALID,
    ERR_TRAVERSAL_MODE_INVALID,
    ERR_TRAVERSAL_START_INVALID,
    compile_decision_dag,
    normalize_decision_dag_input,
    traverse_decision_dag,
    validate_decision_dag,
)


# --------------------------------------------------------------------------
# Deterministic fixture helpers.
# --------------------------------------------------------------------------


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


def _simple_dag_input():
    """Four-node chain with a cross edge. Valid DAG:

        n-a --(depends_on)--> n-b --(verified_by)--> n-c --(tests)--> n-d
         \\__________________(derived_from)_______________________/
    """
    return {
        "dag_id": "dag-simple",
        "nodes": [
            _node("n-a", "release", 0, {"version": "137.16.1"}, _lh("1")),
            _node("n-b", "module", 1, {"path": "src/qec/memory"}, _lh("2")),
            _node("n-c", "proof", 2, {"kind": "cycle_free"}, _lh("3")),
            _node("n-d", "test", 3, {"name": "t_dag"}, _lh("4")),
        ],
        "edges": [
            _edge("e-ab", "n-a", "n-b", "depends_on", 1.0, 1),
            _edge("e-bc", "n-b", "n-c", "verified_by", 2.0, 2),
            _edge("e-cd", "n-c", "n-d", "tests", 1.5, 3),
            _edge("e-ad", "n-a", "n-d", "derived_from", 0.5, 4),
        ],
    }


def _cyclic_dag_input():
    return {
        "dag_id": "dag-cyclic",
        "nodes": [
            _node("n-a", "module", 0, {"name": "a"}, _lh("1")),
            _node("n-b", "module", 1, {"name": "b"}, _lh("2")),
            _node("n-c", "module", 2, {"name": "c"}, _lh("3")),
        ],
        "edges": [
            _edge("e-ab", "n-a", "n-b", "depends_on", 1.0, 1),
            _edge("e-bc", "n-b", "n-c", "depends_on", 1.0, 2),
            _edge("e-ca", "n-c", "n-a", "depends_on", 1.0, 3),
        ],
    }


def _lineage_dag_input():
    return {
        "dag_id": "dag-lineage",
        "nodes": [
            _node("r-0", "release", 0, {"v": "0"}, _lh("1")),
            _node("r-1", "release", 1, {"v": "1"}, _lh("2")),
            _node("r-2", "release", 2, {"v": "2"}, _lh("3")),
            _node("m-x", "module", 1, {"m": "x"}, _lh("4")),
        ],
        "edges": [
            _edge("e-01", "r-0", "r-1", "derived_from", 1.0, 1),
            _edge("e-12", "r-1", "r-2", "supersedes", 1.0, 2),
            _edge("e-1x", "r-1", "m-x", "depends_on", 1.0, 3),
        ],
    }


# --------------------------------------------------------------------------
# Repeated-run byte identity / hash identity.
# --------------------------------------------------------------------------


def test_repeated_compile_byte_identity():
    raw = _simple_dag_input()
    d1 = compile_decision_dag(raw)
    d2 = compile_decision_dag(raw)
    assert d1.to_canonical_bytes() == d2.to_canonical_bytes()
    assert d1.to_canonical_json() == d2.to_canonical_json()
    assert d1.as_hash_payload() == d2.as_hash_payload()


def test_repeated_compile_dag_hash_identity():
    raw = _simple_dag_input()
    hashes = {compile_decision_dag(raw).dag_hash for _ in range(5)}
    assert len(hashes) == 1


def test_compile_is_input_order_independent():
    raw = _simple_dag_input()
    shuffled = dict(raw)
    shuffled["nodes"] = list(reversed(raw["nodes"]))
    shuffled["edges"] = list(reversed(raw["edges"]))
    a = compile_decision_dag(raw)
    b = compile_decision_dag(shuffled)
    assert a.dag_hash == b.dag_hash
    assert a.to_canonical_bytes() == b.to_canonical_bytes()
    assert a.topological_order == b.topological_order


def test_compile_from_compiled_is_idempotent():
    raw = _simple_dag_input()
    a = compile_decision_dag(raw)
    b = compile_decision_dag(a)
    assert a.dag_hash == b.dag_hash
    assert a.to_canonical_bytes() == b.to_canonical_bytes()


# --------------------------------------------------------------------------
# Topological ordering.
# --------------------------------------------------------------------------


def test_topological_order_is_valid_linear_extension():
    raw = _simple_dag_input()
    dag = compile_decision_dag(raw)
    position = {nid: i for i, nid in enumerate(dag.topological_order)}
    for edge in dag.edges:
        assert position[edge.source_node_id] < position[edge.target_node_id]


def test_topological_order_deterministic_across_runs():
    raw = _simple_dag_input()
    orders = {compile_decision_dag(raw).topological_order for _ in range(5)}
    assert len(orders) == 1


def test_topological_order_covers_all_nodes():
    raw = _simple_dag_input()
    dag = compile_decision_dag(raw)
    assert set(dag.topological_order) == {n.node_id for n in dag.nodes}
    assert len(dag.topological_order) == len(dag.nodes)


def test_nodes_stored_in_topological_order():
    raw = _simple_dag_input()
    dag = compile_decision_dag(raw)
    assert tuple(n.node_id for n in dag.nodes) == dag.topological_order


# --------------------------------------------------------------------------
# Canonical export stability.
# --------------------------------------------------------------------------


def test_canonical_export_stability():
    raw = _simple_dag_input()
    dag = compile_decision_dag(raw)
    # Exports are repeatable.
    assert dag.to_canonical_bytes() == dag.to_canonical_bytes()
    assert dag.to_canonical_json() == dag.to_canonical_json()
    # Node and edge exports are individually repeatable.
    for node in dag.nodes:
        assert isinstance(node, DecisionDAGNode)
        assert node.to_canonical_bytes() == node.to_canonical_bytes()
    for edge in dag.edges:
        assert isinstance(edge, DecisionDAGEdge)
        assert edge.to_canonical_bytes() == edge.to_canonical_bytes()


def test_dataclasses_are_frozen():
    raw = _simple_dag_input()
    dag = compile_decision_dag(raw)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        dag.nodes[0].node_id = "mutated"  # type: ignore[misc]
    with pytest.raises((FrozenInstanceError, AttributeError)):
        dag.edges[0].edge_weight = 99.0  # type: ignore[misc]
    with pytest.raises((FrozenInstanceError, AttributeError)):
        dag.dag_hash = "mutated"  # type: ignore[misc]


# --------------------------------------------------------------------------
# Fail-fast rejection law.
# --------------------------------------------------------------------------


def test_cycle_rejection_in_normalize():
    raw = _cyclic_dag_input()
    with pytest.raises(DecisionDAGError) as excinfo:
        normalize_decision_dag_input(raw)
    assert excinfo.value.code == ERR_DAG_CYCLE


def test_cycle_rejection_in_compile():
    raw = _cyclic_dag_input()
    with pytest.raises(DecisionDAGError) as excinfo:
        compile_decision_dag(raw)
    assert excinfo.value.code == ERR_DAG_CYCLE


def test_self_loop_rejection():
    raw = _simple_dag_input()
    raw["edges"].append(
        _edge("e-loop", "n-b", "n-b", "depends_on", 1.0, 99)
    )
    with pytest.raises(DecisionDAGError) as excinfo:
        compile_decision_dag(raw)
    assert excinfo.value.code == ERR_DAG_CYCLE


def test_duplicate_node_rejection():
    raw = _simple_dag_input()
    raw["nodes"].append(_node("n-a", "module", 5))
    with pytest.raises(DecisionDAGError) as excinfo:
        compile_decision_dag(raw)
    assert excinfo.value.code == "duplicate_node"


def test_duplicate_edge_rejection():
    raw = _simple_dag_input()
    raw["edges"].append(
        _edge("e-ab", "n-b", "n-c", "depends_on", 1.0, 99)
    )
    with pytest.raises(DecisionDAGError) as excinfo:
        compile_decision_dag(raw)
    assert excinfo.value.code == "duplicate_edge"


def test_missing_edge_target_rejection():
    raw = _simple_dag_input()
    raw["edges"].append(
        _edge("e-missing", "n-a", "n-ghost", "depends_on", 1.0, 99)
    )
    with pytest.raises(DecisionDAGError):
        compile_decision_dag(raw)


def test_missing_top_level_field_rejection():
    raw = _simple_dag_input()
    del raw["edges"]
    with pytest.raises(DecisionDAGError) as excinfo:
        compile_decision_dag(raw)
    assert excinfo.value.code == ERR_DAG_FIELDS_MISSING


def test_invalid_dag_id_rejection():
    raw = _simple_dag_input()
    raw["dag_id"] = "   "
    with pytest.raises(DecisionDAGError) as excinfo:
        compile_decision_dag(raw)
    assert excinfo.value.code == ERR_DAG_ID_INVALID


def test_invalid_input_type_rejection():
    with pytest.raises(DecisionDAGError) as excinfo:
        compile_decision_dag("not-a-mapping")  # type: ignore[arg-type]
    assert excinfo.value.code == ERR_DAG_TYPE_INVALID


# --------------------------------------------------------------------------
# Validation report.
# --------------------------------------------------------------------------


def test_validate_returns_valid_report_for_clean_dag():
    raw = _simple_dag_input()
    report = validate_decision_dag(raw)
    assert isinstance(report, DecisionDAGValidationReport)
    assert report.is_valid is True
    assert report.structure_ok is True
    assert report.uniqueness_ok is True
    assert report.cycle_free_ok is True
    assert report.node_count == 4
    assert report.edge_count == 4
    assert report.violations == ()


def test_validate_reports_cycle_without_raising():
    raw = _cyclic_dag_input()
    report = validate_decision_dag(raw)
    assert report.is_valid is False
    assert report.cycle_free_ok is False
    assert any(ERR_DAG_CYCLE in v for v in report.violations)


def test_validate_reports_duplicate_node():
    raw = _simple_dag_input()
    raw["nodes"].append(_node("n-a", "module", 5))
    report = validate_decision_dag(raw)
    assert report.is_valid is False
    assert report.uniqueness_ok is False


def test_validate_report_is_canonical():
    raw = _simple_dag_input()
    r1 = validate_decision_dag(raw)
    r2 = validate_decision_dag(raw)
    assert r1.to_canonical_bytes() == r2.to_canonical_bytes()


# --------------------------------------------------------------------------
# Deterministic traversal.
# --------------------------------------------------------------------------


def test_traverse_topological_deterministic():
    raw = _simple_dag_input()
    dag = compile_decision_dag(raw)
    r1 = traverse_decision_dag(dag, "n-a", "topological")
    r2 = traverse_decision_dag(dag, "n-a", "topological")
    assert r1.to_canonical_bytes() == r2.to_canonical_bytes()
    assert r1.visited_nodes == ("n-a", "n-b", "n-c", "n-d")
    # Every edge in the simple DAG is induced from n-a.
    assert set(r1.visited_edges) == {"e-ab", "e-bc", "e-cd", "e-ad"}
    assert r1.traversal_hash == r2.traversal_hash


def test_traverse_topological_respects_linear_extension():
    raw = _simple_dag_input()
    dag = compile_decision_dag(raw)
    receipt = traverse_decision_dag(dag, "n-a", "topological")
    position = {nid: i for i, nid in enumerate(dag.topological_order)}
    visited = list(receipt.visited_nodes)
    assert visited == sorted(visited, key=lambda nid: position[nid])


def test_traverse_topological_from_mid_node():
    raw = _simple_dag_input()
    dag = compile_decision_dag(raw)
    receipt = traverse_decision_dag(dag, "n-b", "topological")
    assert receipt.visited_nodes == ("n-b", "n-c", "n-d")
    assert set(receipt.visited_edges) == {"e-bc", "e-cd"}


def test_traverse_lineage_mode():
    raw = _lineage_dag_input()
    dag = compile_decision_dag(raw)
    receipt = traverse_decision_dag(dag, "r-0", "lineage")
    assert receipt.traversal_mode == "lineage"
    assert receipt.visited_nodes[0] == "r-0"
    # Lineage BFS must not cross depends_on edges.
    assert "m-x" not in receipt.visited_nodes
    assert "r-1" in receipt.visited_nodes
    assert "r-2" in receipt.visited_nodes
    assert "e-1x" not in receipt.visited_edges


def test_traverse_dependency_mode():
    raw = _lineage_dag_input()
    dag = compile_decision_dag(raw)
    receipt = traverse_decision_dag(dag, "r-1", "dependency")
    # dependency BFS only follows depends_on edges.
    assert receipt.visited_nodes == ("r-1", "m-x")
    assert receipt.visited_edges == ("e-1x",)


def test_traverse_critical_path_deterministic():
    raw = _simple_dag_input()
    dag = compile_decision_dag(raw)
    r1 = traverse_decision_dag(dag, "n-a", "critical_path")
    r2 = traverse_decision_dag(dag, "n-a", "critical_path")
    assert r1.to_canonical_bytes() == r2.to_canonical_bytes()
    # Longest weighted path from n-a is n-a -> n-b -> n-c -> n-d
    # (weights 1.0 + 2.0 + 1.5 = 4.5), greater than n-a -> n-d (0.5).
    assert r1.visited_nodes == ("n-a", "n-b", "n-c", "n-d")
    assert r1.visited_edges == ("e-ab", "e-bc", "e-cd")


def test_traverse_critical_path_start_without_outgoing():
    raw = _simple_dag_input()
    dag = compile_decision_dag(raw)
    receipt = traverse_decision_dag(dag, "n-d", "critical_path")
    assert receipt.visited_nodes == ("n-d",)
    assert receipt.visited_edges == ()


def test_traverse_rejects_unknown_mode():
    raw = _simple_dag_input()
    dag = compile_decision_dag(raw)
    with pytest.raises(DecisionDAGError) as excinfo:
        traverse_decision_dag(dag, "n-a", "wormhole")
    assert excinfo.value.code == ERR_TRAVERSAL_MODE_INVALID


def test_traverse_rejects_unknown_start():
    raw = _simple_dag_input()
    dag = compile_decision_dag(raw)
    with pytest.raises(DecisionDAGError) as excinfo:
        traverse_decision_dag(dag, "n-ghost", "topological")
    assert excinfo.value.code == ERR_TRAVERSAL_START_INVALID


def test_traverse_rejects_invalid_dag_type():
    with pytest.raises(DecisionDAGError) as excinfo:
        traverse_decision_dag("not-a-dag", "n-a", "topological")  # type: ignore[arg-type]
    assert excinfo.value.code == ERR_TRAVERSAL_DAG_TYPE_INVALID


def test_traverse_rejects_invalid_cycle_state():
    raw = _simple_dag_input()
    dag = compile_decision_dag(raw)
    broken = CompiledDecisionDAG(
        dag_id=dag.dag_id,
        nodes=dag.nodes,
        edges=dag.edges,
        topological_order=("n-b", "n-a", "n-c", "n-d"),
        dag_hash=dag.dag_hash,
    )
    with pytest.raises(DecisionDAGError) as excinfo:
        traverse_decision_dag(broken, "n-a", "topological")
    assert excinfo.value.code == ERR_TRAVERSAL_CYCLE_STATE_INVALID


def test_traverse_receipt_hash_determinism():
    raw = _simple_dag_input()
    dag = compile_decision_dag(raw)
    hashes = {
        traverse_decision_dag(dag, "n-a", "topological").traversal_hash
        for _ in range(5)
    }
    assert len(hashes) == 1


def test_traverse_receipt_is_frozen():
    raw = _simple_dag_input()
    dag = compile_decision_dag(raw)
    receipt = traverse_decision_dag(dag, "n-a", "topological")
    assert isinstance(receipt, DecisionDAGExecutionReceipt)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        receipt.traversal_hash = "mutated"  # type: ignore[misc]
