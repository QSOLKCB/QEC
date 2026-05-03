from types import MappingProxyType

import pytest

from qec.analysis.atomic_semantic_lattice_contract import (
    AtomicLatticeBounds,
    ConstraintEdgeReceipt,
    LatticeStateReceipt,
    SemanticLatticeNode,
    build_lattice_state_receipt,
    build_semantic_lattice_graph,
    build_topology_stability_receipt,
    validate_lattice_state_receipt,
    validate_topology_stability_receipt,
)


def _bounds() -> AtomicLatticeBounds:
    b = AtomicLatticeBounds("atomic-5", "153.0", 5, 5, 5, "")
    return AtomicLatticeBounds(**{**b.__dict__, "bounds_hash": b.stable_hash()})


def _node(node_id: str, coord: tuple[int, int, int], node_type: str = "semantic") -> SemanticLatticeNode:
    n = SemanticLatticeNode(node_id, node_type, coord, "c-ref", "payload", {"k": "v"}, "")
    return SemanticLatticeNode(**{**n.__dict__, "node_hash": n.stable_hash()})


def _edge(edge_id: str, source: str, target: str, payload: dict[str, object] | None = None) -> ConstraintEdgeReceipt:
    p = payload or {"kind": "hard"}
    from qec.analysis.canonical_hashing import sha256_hex
    from qec.analysis.atomic_semantic_lattice_contract import _ensure_json_safe

    _ensure_json_safe(p)
    payload_hash = sha256_hex(p)
    temp = ConstraintEdgeReceipt(edge_id, source, target, "adjacent", p, payload_hash, "", "")
    full = ConstraintEdgeReceipt(
        edge_id,
        source,
        target,
        "adjacent",
        p,
        temp.constraint_payload_hash or temp.__dict__["constraint_payload_hash"],
        temp._edge_stable_hash(),
        "",
    )
    return ConstraintEdgeReceipt(**{**full.__dict__, "receipt_hash": full.stable_hash()})


def _graph():
    n1, n2 = _node("n1", (0, 0, 0)), _node("n2", (1, 0, 0))
    e1 = _edge("e1", "n1", "n2")
    return build_semantic_lattice_graph("lattice-a", "153.0", _bounds(), (n1, n2), (e1,))


def test_same_lattice_inputs_different_order_produce_same_graph_hash():
    n1, n2 = _node("n1", (0, 0, 0)), _node("n2", (1, 0, 0))
    e1 = _edge("e1", "n1", "n2")
    g1 = build_semantic_lattice_graph("lattice", "153.0", _bounds(), (n1, n2), (e1,))
    g2 = build_semantic_lattice_graph("lattice", "153.0", _bounds(), (n2, n1), (e1,))
    assert g1.graph_hash == g2.graph_hash


def test_duplicate_coordinate_rejected():
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_semantic_lattice_graph("l", "153.0", _bounds(), (_node("n1", (0, 0, 0)), _node("n2", (0, 0, 0))), ())


def test_edge_endpoint_must_exist():
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_semantic_lattice_graph("l", "153.0", _bounds(), (_node("n1", (0, 0, 0)),), (_edge("e1", "n1", "missing"),))


def test_topology_stability_rejects_mismatched_lattice_receipt():
    g1 = _graph()
    g2 = build_semantic_lattice_graph("other", "153.0", _bounds(), (_node("n1", (0, 0, 0)), _node("n2", (1, 0, 0))), (_edge("e1", "n1", "n2"),))
    r1 = build_lattice_state_receipt(g1)
    t = build_topology_stability_receipt(g1, r1)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_topology_stability_receipt(t, g2, r1)


def test_v153_0_does_not_implement_router_or_readout_behavior():
    g = _graph()
    for attr in ("route", "resolve", "readout", "project", "traverse"):
        assert not hasattr(g, attr)


def test_bounds_and_node_and_edge_determinism_and_json_safety_and_immutability():
    b1 = _bounds()
    b2 = _bounds()
    assert b1.bounds_hash == b2.bounds_hash
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        AtomicLatticeBounds("a", "153.0", 0, 5, 5, "")
    n1 = _node("n", (0, 1, 2))
    n2 = _node("n", (0, 1, 2))
    assert n1.node_hash == n2.node_hash
    assert isinstance(n1.node_metadata, MappingProxyType)
    with pytest.raises(TypeError):
        n1.node_metadata["x"] = 1
    e1 = _edge("e", "s", "t")
    e2 = _edge("e", "s", "t")
    assert e1.edge_hash == e2.edge_hash
    assert e1.receipt_hash == e2.receipt_hash
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        _edge("e", "s", "t", {"bad": float("nan")})


def test_graph_ordering_and_changes_and_receipts_and_tamper_rejection():
    g = _graph()
    assert g.nodes[0].node_id == "n1"
    assert g.edges[0].edge_id == "e1"
    g_changed_node = build_semantic_lattice_graph("lattice-a", "153.0", _bounds(), (_node("n1", (0, 0, 0), "semantic-x"), _node("n2", (1, 0, 0))), (_edge("e1", "n1", "n2"),))
    g_changed_edge = build_semantic_lattice_graph("lattice-a", "153.0", _bounds(), (_node("n1", (0, 0, 0)), _node("n2", (1, 0, 0))), (_edge("e1", "n1", "n2", {"kind": "soft"}),))
    assert g.graph_hash != g_changed_node.graph_hash
    assert g.graph_hash != g_changed_edge.graph_hash
    receipt = build_lattice_state_receipt(g)
    validate_lattice_state_receipt(receipt, g)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        LatticeStateReceipt(**{**receipt.__dict__, "receipt_hash": "0" * 64})
    t = build_topology_stability_receipt(g, receipt)
    validate_topology_stability_receipt(t, g, receipt)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_topology_stability_receipt(type(t)(**{**t.__dict__, "topology_hash": "0" * 64}), g, receipt)


def test_duplicate_ids_oob_and_semantic_edge_duplicate_rejected_and_self_hash_exclusion():
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_semantic_lattice_graph("l", "153.0", _bounds(), (_node("n1", (0, 0, 0)), _node("n1", (1, 0, 0))), ())
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_semantic_lattice_graph("l", "153.0", _bounds(), (_node("n1", (5, 0, 0)),), ())
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_semantic_lattice_graph("l", "153.0", _bounds(), (_node("n1", (0, 0, 0)), _node("n2", (1, 0, 0))), (_edge("e1", "n1", "n2"), _edge("e2", "n1", "n2")))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_semantic_lattice_graph("l", "153.0", _bounds(), (_node("n1", (0, 0, 0)), _node("n2", (1, 0, 0))), (_edge("e1", "n1", "n2"), _edge("e1", "n2", "n1")))

    b = _bounds()
    assert "bounds_hash" in b.to_dict()
    assert "bounds_hash" not in b.to_canonical_json()


def test_graph_structural_node_tamper_rejected():
    g = _graph()
    tampered_node = _node("n1", (0, 0, 1))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        type(g)(
            lattice_id=g.lattice_id,
            lattice_version=g.lattice_version,
            bounds_hash=g.bounds_hash,
            node_hashes=g.node_hashes,
            edge_receipt_hashes=g.edge_receipt_hashes,
            nodes=(tampered_node, g.nodes[1]),
            edges=g.edges,
            graph_hash=g.graph_hash,
        )


def test_graph_structural_edge_tamper_rejected():
    g = _graph()
    tampered_edge = _edge("e1", "n2", "n1")
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        type(g)(
            lattice_id=g.lattice_id,
            lattice_version=g.lattice_version,
            bounds_hash=g.bounds_hash,
            node_hashes=g.node_hashes,
            edge_receipt_hashes=g.edge_receipt_hashes,
            nodes=g.nodes,
            edges=(tampered_edge,),
            graph_hash=g.graph_hash,
        )


def test_lattice_state_validator_rejects_desynchronized_node_hashes():
    g = _graph()
    r = build_lattice_state_receipt(g)
    object.__setattr__(g, "node_hashes", ("0" * 64, g.node_hashes[1]))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_lattice_state_receipt(r, g)


def test_lattice_state_validator_rejects_desynchronized_edge_hashes():
    g = _graph()
    r = build_lattice_state_receipt(g)
    object.__setattr__(g, "edge_receipt_hashes", ("0" * 64,))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_lattice_state_receipt(r, g)


def test_topology_stability_validator_rejects_structurally_forged_graph():
    g = _graph()
    r = build_lattice_state_receipt(g)
    t = build_topology_stability_receipt(g, r)
    object.__setattr__(g.nodes[0], "semantic_payload_hash", "tampered")
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_topology_stability_receipt(t, g, r)


def test_receipt_self_hash_enforcement_and_unsigned_pattern():
    g = _graph()
    r = build_lattice_state_receipt(g)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        LatticeStateReceipt(**{**r.__dict__, "receipt_hash": "1" * 64})
    unsigned = LatticeStateReceipt(**{**r.__dict__, "receipt_hash": ""})
    assert unsigned.receipt_hash == ""

    t = build_topology_stability_receipt(g, r)
    from qec.analysis.atomic_semantic_lattice_contract import TopologyStabilityReceipt
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        TopologyStabilityReceipt(**{**t.__dict__, "receipt_hash": "2" * 64})
    unsigned_t = TopologyStabilityReceipt(**{**t.__dict__, "receipt_hash": ""})
    assert unsigned_t.receipt_hash == ""


def test_validate_lattice_state_receipt_rejects_forged_lattice_id():
    """Regression test: receipt.lattice_id must match graph.lattice_id."""
    g = _graph()
    r = build_lattice_state_receipt(g)
    # Forge a receipt with different lattice_id but same graph_hash
    forged = LatticeStateReceipt(
        lattice_id="forged-id",
        lattice_version=r.lattice_version,
        bounds_hash=r.bounds_hash,
        node_set_hash=r.node_set_hash,
        edge_set_hash=r.edge_set_hash,
        graph_hash=r.graph_hash,
        receipt_hash="",
    )
    forged = LatticeStateReceipt(**{**forged.__dict__, "receipt_hash": forged.stable_hash()})
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_lattice_state_receipt(forged, g)


def test_validate_lattice_state_receipt_rejects_forged_lattice_version():
    """Regression test: receipt.lattice_version must match graph.lattice_version."""
    g = _graph()
    r = build_lattice_state_receipt(g)
    # Forge a receipt with different lattice_version but same graph_hash
    forged = LatticeStateReceipt(
        lattice_id=r.lattice_id,
        lattice_version="forged-version",
        bounds_hash=r.bounds_hash,
        node_set_hash=r.node_set_hash,
        edge_set_hash=r.edge_set_hash,
        graph_hash=r.graph_hash,
        receipt_hash="",
    )
    forged = LatticeStateReceipt(**{**forged.__dict__, "receipt_hash": forged.stable_hash()})
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_lattice_state_receipt(forged, g)


def test_graph_internal_consistency_rejects_dangling_edges():
    """Regression test: edges with endpoints absent from node set must be rejected."""
    from qec.analysis.atomic_semantic_lattice_contract import SemanticLatticeGraph

    n1, n2 = _node("n1", (0, 0, 0)), _node("n2", (1, 0, 0))
    # Create an edge that references a node not in the node set
    dangling_edge = _edge("e1", "n1", "nonexistent")
    # Build a valid graph first, then tamper it
    g = build_semantic_lattice_graph("lattice-a", "153.0", _bounds(), (n1, n2), (_edge("e1", "n1", "n2"),))
    # Attempt to create graph with dangling edge bypassing build function
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        SemanticLatticeGraph(
            lattice_id=g.lattice_id,
            lattice_version=g.lattice_version,
            bounds_hash=g.bounds_hash,
            node_hashes=g.node_hashes,
            edge_receipt_hashes=(dangling_edge.receipt_hash,),
            nodes=g.nodes,
            edges=(dangling_edge,),
            graph_hash="",
        )
