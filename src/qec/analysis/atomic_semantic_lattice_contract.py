from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from qec.analysis.canonical_hashing import canonical_json, sha256_hex
from qec.analysis.layer_spec_contract import (
    _deep_freeze,
    _ensure_json_safe as _base_ensure_json_safe,
)


def _ensure_json_safe(value: Any) -> None:
    """Validate JSON-safety, extending base check with callable rejection."""
    if callable(value):
        raise ValueError("INVALID_INPUT")
    _base_ensure_json_safe(value)


def _canonical_key(value: Any) -> tuple[str, str]:
    return (type(value).__name__, str(value))


def _freeze_mapping(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    """Freeze a mapping using canonical key ordering."""
    ordered: dict[str, Any] = {}
    for key in sorted(mapping, key=_canonical_key):
        ordered[key] = _deep_freeze(mapping[key])
    return MappingProxyType(ordered)


@dataclass(frozen=True)
class AtomicLatticeBounds:
    bounds_id: str
    bounds_version: str
    x_size: int
    y_size: int
    z_size: int
    bounds_hash: str

    def __post_init__(self) -> None:
        if not self.bounds_id or not self.bounds_version:
            raise ValueError("INVALID_INPUT")
        for size in (self.x_size, self.y_size, self.z_size):
            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                raise ValueError("INVALID_INPUT")
        if self.bounds_hash and self.bounds_hash != self.stable_hash():
            raise ValueError("INVALID_INPUT")

    def _canonical_payload(self) -> dict:
        payload = {
            "bounds_id": self.bounds_id,
            "bounds_version": self.bounds_version,
            "x_size": self.x_size,
            "y_size": self.y_size,
            "z_size": self.z_size,
        }
        _ensure_json_safe(payload)
        return payload

    def to_dict(self) -> dict:
        payload = dict(self._canonical_payload())
        payload["bounds_hash"] = self.bounds_hash
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self._canonical_payload())

    def stable_hash(self) -> str:
        return sha256_hex(self._canonical_payload())


@dataclass(frozen=True)
class SemanticLatticeNode:
    node_id: str
    node_type: str
    coordinate: tuple[int, int, int]
    canonical_ref_hash: str
    semantic_payload_hash: str
    node_metadata: Mapping[str, Any]
    node_hash: str

    def __post_init__(self) -> None:
        if not self.node_id or not self.node_type or not self.canonical_ref_hash or not self.semantic_payload_hash:
            raise ValueError("INVALID_INPUT")
        if not isinstance(self.coordinate, tuple) or len(self.coordinate) != 3:
            raise ValueError("INVALID_INPUT")
        for axis in self.coordinate:
            if not isinstance(axis, int) or isinstance(axis, bool):
                raise ValueError("INVALID_INPUT")
        object.__setattr__(self, "node_metadata", _freeze_mapping(dict(self.node_metadata)))
        _ensure_json_safe(self._canonical_payload())
        if self.node_hash and self.node_hash != self.stable_hash():
            raise ValueError("INVALID_INPUT")

    def _canonical_payload(self) -> dict:
        payload = {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "coordinate": list(self.coordinate),
            "canonical_ref_hash": self.canonical_ref_hash,
            "semantic_payload_hash": self.semantic_payload_hash,
            "node_metadata": dict(self.node_metadata),
        }
        _ensure_json_safe(payload)
        return payload

    def to_dict(self) -> dict:
        payload = dict(self._canonical_payload())
        payload["node_hash"] = self.node_hash
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self._canonical_payload())

    def stable_hash(self) -> str:
        return sha256_hex(self._canonical_payload())


@dataclass(frozen=True)
class ConstraintEdgeReceipt:
    edge_id: str
    source_node_id: str
    target_node_id: str
    constraint_type: str
    constraint_payload: Mapping[str, Any]
    constraint_payload_hash: str
    edge_hash: str
    receipt_hash: str

    def __post_init__(self) -> None:
        if not self.edge_id or not self.source_node_id or not self.target_node_id or not self.constraint_type:
            raise ValueError("INVALID_INPUT")
        object.__setattr__(self, "constraint_payload", _freeze_mapping(dict(self.constraint_payload)))
        _ensure_json_safe(dict(self.constraint_payload))
        payload_hash = sha256_hex(dict(self.constraint_payload))
        if self.constraint_payload_hash and self.constraint_payload_hash != payload_hash:
            raise ValueError("INVALID_INPUT")
        if self.edge_hash and self.edge_hash != self._edge_stable_hash():
            raise ValueError("INVALID_INPUT")
        if self.receipt_hash and self.receipt_hash != self.stable_hash():
            raise ValueError("INVALID_INPUT")

    def _edge_payload(self) -> dict:
        payload = {
            "edge_id": self.edge_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "constraint_type": self.constraint_type,
            "constraint_payload_hash": self.constraint_payload_hash,
        }
        _ensure_json_safe(payload)
        return payload

    def _canonical_payload(self) -> dict:
        payload = {
            "edge_id": self.edge_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "constraint_type": self.constraint_type,
            "constraint_payload": dict(self.constraint_payload),
            "constraint_payload_hash": self.constraint_payload_hash,
            "edge_hash": self.edge_hash,
        }
        _ensure_json_safe(payload)
        return payload

    def _edge_stable_hash(self) -> str:
        return sha256_hex(self._edge_payload())

    def to_dict(self) -> dict:
        payload = dict(self._canonical_payload())
        payload["receipt_hash"] = self.receipt_hash
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self._canonical_payload())

    def stable_hash(self) -> str:
        return sha256_hex(self._canonical_payload())


@dataclass(frozen=True)
class SemanticLatticeGraph:
    lattice_id: str
    lattice_version: str
    bounds_hash: str
    node_hashes: tuple[str, ...]
    edge_receipt_hashes: tuple[str, ...]
    nodes: tuple[SemanticLatticeNode, ...]
    edges: tuple[ConstraintEdgeReceipt, ...]
    graph_hash: str

    def __post_init__(self) -> None:
        if not self.lattice_id or not self.lattice_version or not self.bounds_hash:
            raise ValueError("INVALID_INPUT")
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "edges", tuple(self.edges))
        object.__setattr__(self, "node_hashes", tuple(self.node_hashes))
        object.__setattr__(self, "edge_receipt_hashes", tuple(self.edge_receipt_hashes))
        _validate_graph_internal_consistency(self)

    def _canonical_payload(self) -> dict:
        payload = {
            "lattice_id": self.lattice_id,
            "lattice_version": self.lattice_version,
            "bounds_hash": self.bounds_hash,
            "node_hashes": list(self.node_hashes),
            "edge_receipt_hashes": list(self.edge_receipt_hashes),
        }
        _ensure_json_safe(payload)
        return payload

    def to_dict(self) -> dict:
        payload = dict(self._canonical_payload())
        payload["nodes"] = [node.to_dict() for node in self.nodes]
        payload["edges"] = [edge.to_dict() for edge in self.edges]
        payload["graph_hash"] = self.graph_hash
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self._canonical_payload())

    def stable_hash(self) -> str:
        return sha256_hex(self._canonical_payload())


@dataclass(frozen=True)
class LatticeStateReceipt:
    lattice_id: str
    lattice_version: str
    bounds_hash: str
    node_set_hash: str
    edge_set_hash: str
    graph_hash: str
    receipt_hash: str

    def __post_init__(self) -> None:
        if not self.lattice_id or not self.lattice_version or not self.bounds_hash or not self.node_set_hash or not self.edge_set_hash or not self.graph_hash:
            raise ValueError("INVALID_INPUT")
        if self.receipt_hash and self.receipt_hash != self.stable_hash():
            raise ValueError("INVALID_INPUT")

    def _canonical_payload(self) -> dict:
        payload = {
            "lattice_id": self.lattice_id,
            "lattice_version": self.lattice_version,
            "bounds_hash": self.bounds_hash,
            "node_set_hash": self.node_set_hash,
            "edge_set_hash": self.edge_set_hash,
            "graph_hash": self.graph_hash,
        }
        _ensure_json_safe(payload)
        return payload

    def to_dict(self) -> dict:
        payload = dict(self._canonical_payload())
        payload["receipt_hash"] = self.receipt_hash
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self._canonical_payload())

    def stable_hash(self) -> str:
        return sha256_hex(self._canonical_payload())


@dataclass(frozen=True)
class TopologyStabilityReceipt:
    lattice_state_receipt_hash: str
    graph_hash: str
    bounds_hash: str
    node_set_hash: str
    edge_set_hash: str
    topology_hash: str
    receipt_hash: str

    def __post_init__(self) -> None:
        if (
            not self.lattice_state_receipt_hash
            or not self.graph_hash
            or not self.bounds_hash
            or not self.node_set_hash
            or not self.edge_set_hash
            or not self.topology_hash
        ):
            raise ValueError("INVALID_INPUT")
        if self.receipt_hash and self.receipt_hash != self.stable_hash():
            raise ValueError("INVALID_INPUT")

    def _topology_payload(self) -> dict:
        payload = {
            "graph_hash": self.graph_hash,
            "bounds_hash": self.bounds_hash,
            "node_set_hash": self.node_set_hash,
            "edge_set_hash": self.edge_set_hash,
        }
        _ensure_json_safe(payload)
        return payload

    def _canonical_payload(self) -> dict:
        payload = {
            "lattice_state_receipt_hash": self.lattice_state_receipt_hash,
            "graph_hash": self.graph_hash,
            "bounds_hash": self.bounds_hash,
            "node_set_hash": self.node_set_hash,
            "edge_set_hash": self.edge_set_hash,
            "topology_hash": self.topology_hash,
        }
        _ensure_json_safe(payload)
        return payload

    def to_dict(self) -> dict:
        payload = dict(self._canonical_payload())
        payload["receipt_hash"] = self.receipt_hash
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self._canonical_payload())

    def stable_hash(self) -> str:
        return sha256_hex(self._canonical_payload())


def build_semantic_lattice_graph(
    lattice_id: str,
    lattice_version: str,
    bounds: AtomicLatticeBounds,
    nodes: tuple[SemanticLatticeNode, ...],
    edges: tuple[ConstraintEdgeReceipt, ...],
) -> SemanticLatticeGraph:
    if not lattice_id or not lattice_version:
        raise ValueError("INVALID_INPUT")
    if bounds.bounds_hash != bounds.stable_hash():
        raise ValueError("INVALID_INPUT")

    sorted_nodes = tuple(sorted(nodes, key=lambda n: (n.coordinate[0], n.coordinate[1], n.coordinate[2], n.node_id, n.node_hash)))
    seen_ids: set[str] = set()
    seen_coords: set[tuple[int, int, int]] = set()
    for node in sorted_nodes:
        if node.node_hash != node.stable_hash():
            raise ValueError("INVALID_INPUT")
        if node.node_id in seen_ids or node.coordinate in seen_coords:
            raise ValueError("INVALID_INPUT")
        if not (0 <= node.coordinate[0] < bounds.x_size and 0 <= node.coordinate[1] < bounds.y_size and 0 <= node.coordinate[2] < bounds.z_size):
            raise ValueError("INVALID_INPUT")
        seen_ids.add(node.node_id)
        seen_coords.add(node.coordinate)

    sorted_edges = tuple(sorted(edges, key=lambda e: (e.source_node_id, e.target_node_id, e.constraint_type, e.edge_id, e.edge_hash)))
    seen_edge_ids: set[str] = set()
    seen_semantic: set[tuple[str, str, str, str]] = set()
    for edge in sorted_edges:
        if edge.source_node_id not in seen_ids or edge.target_node_id not in seen_ids:
            raise ValueError("INVALID_INPUT")
        if edge.edge_id in seen_edge_ids:
            raise ValueError("INVALID_INPUT")
        semantic_key = (edge.source_node_id, edge.target_node_id, edge.constraint_type, edge.constraint_payload_hash)
        if semantic_key in seen_semantic:
            raise ValueError("INVALID_INPUT")
        if edge.constraint_payload_hash != sha256_hex(dict(edge.constraint_payload)):
            raise ValueError("INVALID_INPUT")
        if edge.edge_hash != edge._edge_stable_hash() or edge.receipt_hash != edge.stable_hash():
            raise ValueError("INVALID_INPUT")
        seen_edge_ids.add(edge.edge_id)
        seen_semantic.add(semantic_key)

    node_hashes = tuple(node.node_hash for node in sorted_nodes)
    edge_receipt_hashes = tuple(edge.receipt_hash for edge in sorted_edges)
    graph = SemanticLatticeGraph(
        lattice_id=lattice_id,
        lattice_version=lattice_version,
        bounds_hash=bounds.stable_hash(),
        node_hashes=node_hashes,
        edge_receipt_hashes=edge_receipt_hashes,
        nodes=sorted_nodes,
        edges=sorted_edges,
        graph_hash="",
    )
    return SemanticLatticeGraph(**{**graph.__dict__, "graph_hash": graph.stable_hash()})


def _validate_graph_internal_consistency(graph: SemanticLatticeGraph) -> None:
    if len(graph.nodes) != len(graph.node_hashes):
        raise ValueError("INVALID_INPUT")
    if len(graph.edges) != len(graph.edge_receipt_hashes):
        raise ValueError("INVALID_INPUT")

    expected_nodes = tuple(sorted(graph.nodes, key=lambda n: (n.coordinate[0], n.coordinate[1], n.coordinate[2], n.node_id, n.node_hash)))
    expected_node_hashes: list[str] = []
    valid_node_ids: set[str] = set()
    for index, node in enumerate(expected_nodes):
        canonical_hash = node.stable_hash()
        if node.node_hash != canonical_hash:
            raise ValueError("INVALID_INPUT")
        expected_node_hashes.append(canonical_hash)
        if graph.node_hashes[index] != canonical_hash:
            raise ValueError("INVALID_INPUT")
        valid_node_ids.add(node.node_id)
    if tuple(expected_node_hashes) != graph.node_hashes:
        raise ValueError("INVALID_INPUT")
    if tuple(graph.nodes) != expected_nodes:
        raise ValueError("INVALID_INPUT")

    expected_edges = tuple(sorted(graph.edges, key=lambda e: (e.source_node_id, e.target_node_id, e.constraint_type, e.edge_id, e.edge_hash)))
    expected_edge_receipt_hashes: list[str] = []
    for index, edge in enumerate(expected_edges):
        if edge.source_node_id not in valid_node_ids or edge.target_node_id not in valid_node_ids:
            raise ValueError("INVALID_INPUT")
        if edge.constraint_payload_hash != sha256_hex(dict(edge.constraint_payload)):
            raise ValueError("INVALID_INPUT")
        if edge.edge_hash != edge._edge_stable_hash():
            raise ValueError("INVALID_INPUT")
        if edge.receipt_hash != edge.stable_hash():
            raise ValueError("INVALID_INPUT")
        expected_edge_receipt_hashes.append(edge.receipt_hash)
        if graph.edge_receipt_hashes[index] != edge.receipt_hash:
            raise ValueError("INVALID_INPUT")
    if tuple(expected_edge_receipt_hashes) != graph.edge_receipt_hashes:
        raise ValueError("INVALID_INPUT")
    if tuple(graph.edges) != expected_edges:
        raise ValueError("INVALID_INPUT")

    if graph.graph_hash and graph.stable_hash() != graph.graph_hash:
        raise ValueError("INVALID_INPUT")


def build_lattice_state_receipt(graph: SemanticLatticeGraph) -> LatticeStateReceipt:
    if graph.graph_hash != graph.stable_hash():
        raise ValueError("INVALID_INPUT")
    node_set_hash = sha256_hex(list(graph.node_hashes))
    edge_set_hash = sha256_hex(list(graph.edge_receipt_hashes))
    receipt = LatticeStateReceipt(graph.lattice_id, graph.lattice_version, graph.bounds_hash, node_set_hash, edge_set_hash, graph.graph_hash, "")
    return LatticeStateReceipt(**{**receipt.__dict__, "receipt_hash": receipt.stable_hash()})


def build_topology_stability_receipt(graph: SemanticLatticeGraph, lattice_receipt: LatticeStateReceipt) -> TopologyStabilityReceipt:
    validate_lattice_state_receipt(lattice_receipt, graph)
    topology_hash = sha256_hex({
        "graph_hash": graph.graph_hash,
        "bounds_hash": graph.bounds_hash,
        "node_set_hash": lattice_receipt.node_set_hash,
        "edge_set_hash": lattice_receipt.edge_set_hash,
    })
    receipt = TopologyStabilityReceipt(
        lattice_state_receipt_hash=lattice_receipt.receipt_hash,
        graph_hash=graph.graph_hash,
        bounds_hash=graph.bounds_hash,
        node_set_hash=lattice_receipt.node_set_hash,
        edge_set_hash=lattice_receipt.edge_set_hash,
        topology_hash=topology_hash,
        receipt_hash="",
    )
    return TopologyStabilityReceipt(**{**receipt.__dict__, "receipt_hash": receipt.stable_hash()})


def validate_lattice_state_receipt(receipt: LatticeStateReceipt, graph: SemanticLatticeGraph) -> None:
    _validate_graph_internal_consistency(graph)
    if receipt.lattice_id != graph.lattice_id:
        raise ValueError("INVALID_INPUT")
    if receipt.lattice_version != graph.lattice_version:
        raise ValueError("INVALID_INPUT")
    if graph.graph_hash != graph.stable_hash() or receipt.graph_hash != graph.graph_hash:
        raise ValueError("INVALID_INPUT")
    if receipt.bounds_hash != graph.bounds_hash:
        raise ValueError("INVALID_INPUT")
    if receipt.node_set_hash != sha256_hex(list(graph.node_hashes)):
        raise ValueError("INVALID_INPUT")
    if receipt.edge_set_hash != sha256_hex(list(graph.edge_receipt_hashes)):
        raise ValueError("INVALID_INPUT")
    if receipt.receipt_hash != LatticeStateReceipt(**{**receipt.__dict__, "receipt_hash": ""}).stable_hash():
        raise ValueError("INVALID_INPUT")


def validate_topology_stability_receipt(topology_receipt: TopologyStabilityReceipt, graph: SemanticLatticeGraph, lattice_receipt: LatticeStateReceipt) -> None:
    _validate_graph_internal_consistency(graph)
    validate_lattice_state_receipt(lattice_receipt, graph)
    if topology_receipt.lattice_state_receipt_hash != lattice_receipt.receipt_hash:
        raise ValueError("INVALID_INPUT")
    if topology_receipt.graph_hash != graph.graph_hash:
        raise ValueError("INVALID_INPUT")
    if topology_receipt.bounds_hash != graph.bounds_hash:
        raise ValueError("INVALID_INPUT")
    if topology_receipt.node_set_hash != lattice_receipt.node_set_hash:
        raise ValueError("INVALID_INPUT")
    if topology_receipt.edge_set_hash != lattice_receipt.edge_set_hash:
        raise ValueError("INVALID_INPUT")
    expected_topology_hash = sha256_hex({
        "graph_hash": graph.graph_hash,
        "bounds_hash": graph.bounds_hash,
        "node_set_hash": lattice_receipt.node_set_hash,
        "edge_set_hash": lattice_receipt.edge_set_hash,
    })
    if topology_receipt.topology_hash != expected_topology_hash:
        raise ValueError("INVALID_INPUT")
    if topology_receipt.receipt_hash != TopologyStabilityReceipt(**{**topology_receipt.__dict__, "receipt_hash": ""}).stable_hash():
        raise ValueError("INVALID_INPUT")


for _name in ("apply", "execute", "run", "traverse", "resolve", "route", "readout", "project", "render"):
    if hasattr(SemanticLatticeGraph, _name) or hasattr(LatticeStateReceipt, _name) or hasattr(TopologyStabilityReceipt, _name):
        raise RuntimeError("INVALID_INPUT")
