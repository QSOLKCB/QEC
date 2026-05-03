from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from qec.analysis.atomic_semantic_lattice_contract import SemanticLatticeGraph, _validate_graph_internal_consistency
from qec.analysis.canonical_hashing import canonical_json, sha256_hex
from qec.analysis.layer_spec_contract import _deep_freeze, _ensure_json_safe
from qec.analysis.layered_state_receipt import LayeredReceipt

MAX_LAYERED_NODE_BINDINGS = 128
MAX_LAYERED_EDGE_BINDINGS = 128
_ALLOWED_NODE_BINDING_ROLES = {"BASE_STATE", "LAYER_STATE", "LAYER_PAYLOAD", "LAYER_INVARIANT", "LAYER_BOUNDARY", "DERIVED_OBSERVABLE"}
_ALLOWED_EDGE_BINDING_ROLES = {"LAYER_CONSTRAINT", "LAYER_COMPATIBILITY", "LAYER_BOUNDARY", "LAYER_DERIVATION", "LAYER_REMOVAL_PATH", "LAYER_EQUIVALENCE"}
_ALLOWED_PROJECTION_POLICIES = {"STRICT_EXPLICIT_BINDINGS"}


def _freeze_mapping(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({k: _deep_freeze(mapping[k]) for k in sorted(mapping)})


def _validate_graph(graph: SemanticLatticeGraph) -> None:
    _validate_graph_internal_consistency(graph)
    if graph.graph_hash != graph.stable_hash():
        raise ValueError("INVALID_INPUT")


def _layered_topology_integrity_payload(
    projection_id: str,
    semantic_lattice_graph_hash: str,
    layered_receipt_hash: str,
    base_hash: str,
    layered_hash: str,
    layer_spec_hash: str,
    layer_payload_hash: str,
    node_binding_hashes: tuple[str, ...],
    edge_binding_hashes: tuple[str, ...],
    node_binding_count: int,
    edge_binding_count: int,
) -> dict[str, Any]:
    payload = {
        "projection_id": projection_id,
        "semantic_lattice_graph_hash": semantic_lattice_graph_hash,
        "layered_receipt_hash": layered_receipt_hash,
        "base_hash": base_hash,
        "layered_hash": layered_hash,
        "layer_spec_hash": layer_spec_hash,
        "layer_payload_hash": layer_payload_hash,
        "node_binding_hashes": list(node_binding_hashes),
        "edge_binding_hashes": list(edge_binding_hashes),
        "node_binding_count": node_binding_count,
        "edge_binding_count": edge_binding_count,
    }
    _ensure_json_safe(payload)
    return payload


def _compute_topology_integrity_hash(payload: Mapping[str, Any]) -> str:
    return sha256_hex(dict(payload))


def _compute_topology_receipt_hash(payload: Mapping[str, Any], topology_integrity_hash: str) -> str:
    canonical_payload = dict(payload, topology_integrity_hash=topology_integrity_hash)
    _ensure_json_safe(canonical_payload)
    return sha256_hex(canonical_payload)


def _layered_lattice_projection_payload(
    projection_id: str, lattice_id: str, lattice_version: str, semantic_lattice_graph_hash: str, layered_receipt_hash: str,
    base_hash: str, layer_spec_hash: str, layer_payload_hash: str, layered_hash: str, projection_spec_hash: str,
    topology_integrity_receipt_hash: str, node_binding_count: int, edge_binding_count: int, layered_lattice_projection_hash: str
) -> dict[str, Any]:
    payload = {"projection_id": projection_id, "lattice_id": lattice_id, "lattice_version": lattice_version, "semantic_lattice_graph_hash": semantic_lattice_graph_hash, "layered_receipt_hash": layered_receipt_hash, "base_hash": base_hash, "layer_spec_hash": layer_spec_hash, "layer_payload_hash": layer_payload_hash, "layered_hash": layered_hash, "projection_spec_hash": projection_spec_hash, "topology_integrity_receipt_hash": topology_integrity_receipt_hash, "node_binding_count": node_binding_count, "edge_binding_count": edge_binding_count, "layered_lattice_projection_hash": layered_lattice_projection_hash}
    _ensure_json_safe(payload)
    return payload


def _compute_layered_lattice_projection_hash(
    semantic_lattice_graph_hash: str,
    layered_receipt_hash: str,
    projection_spec_hash: str,
    topology_integrity_receipt_hash: str,
    node_bindings_payload: list[dict[str, Any]],
    edge_bindings_payload: list[dict[str, Any]],
) -> str:
    return sha256_hex({"semantic_lattice_graph_hash": semantic_lattice_graph_hash, "layered_receipt_hash": layered_receipt_hash, "projection_spec_hash": projection_spec_hash, "topology_integrity_receipt_hash": topology_integrity_receipt_hash, "node_bindings": node_bindings_payload, "edge_bindings": edge_bindings_payload})


def _compute_projection_receipt_hash(payload: Mapping[str, Any]) -> str:
    return sha256_hex(dict(payload))


@dataclass(frozen=True)
class LayeredNodeBinding:
    binding_id: str
    layer_id: str
    node_id: str
    base_hash: str
    layered_hash: str
    layer_spec_hash: str
    layer_payload_hash: str
    binding_role: str
    binding_metadata: Mapping[str, Any]
    binding_hash: str

    def __post_init__(self) -> None:
        if not self.binding_id or not self.layer_id or not self.node_id:
            raise ValueError("INVALID_INPUT")
        if self.binding_role not in _ALLOWED_NODE_BINDING_ROLES:
            raise ValueError("INVALID_INPUT")
        object.__setattr__(self, "binding_metadata", _freeze_mapping(dict(self.binding_metadata)))
        _ensure_json_safe(self._canonical_payload())
        if self.binding_hash and self.binding_hash != self.stable_hash():
            raise ValueError("INVALID_INPUT")

    def _canonical_payload(self) -> dict:
        p = {"binding_id": self.binding_id, "layer_id": self.layer_id, "node_id": self.node_id, "base_hash": self.base_hash, "layered_hash": self.layered_hash, "layer_spec_hash": self.layer_spec_hash, "layer_payload_hash": self.layer_payload_hash, "binding_role": self.binding_role, "binding_metadata": dict(self.binding_metadata)}
        _ensure_json_safe(p)
        return p

    def to_dict(self) -> dict:
        return dict(self._canonical_payload(), binding_hash=self.binding_hash)

    def to_canonical_json(self) -> str:
        return canonical_json(self._canonical_payload())

    def stable_hash(self) -> str:
        return sha256_hex(self._canonical_payload())


@dataclass(frozen=True)
class LayeredEdgeBinding:
    binding_id: str
    layer_id: str
    edge_id: str
    base_hash: str
    layered_hash: str
    layer_spec_hash: str
    layer_payload_hash: str
    binding_role: str
    binding_metadata: Mapping[str, Any]
    binding_hash: str

    def __post_init__(self) -> None:
        if not self.binding_id or not self.layer_id or not self.edge_id:
            raise ValueError("INVALID_INPUT")
        if self.binding_role not in _ALLOWED_EDGE_BINDING_ROLES:
            raise ValueError("INVALID_INPUT")
        object.__setattr__(self, "binding_metadata", _freeze_mapping(dict(self.binding_metadata)))
        _ensure_json_safe(self._canonical_payload())
        if self.binding_hash and self.binding_hash != self.stable_hash():
            raise ValueError("INVALID_INPUT")

    def _canonical_payload(self) -> dict:
        p = {"binding_id": self.binding_id, "layer_id": self.layer_id, "edge_id": self.edge_id, "base_hash": self.base_hash, "layered_hash": self.layered_hash, "layer_spec_hash": self.layer_spec_hash, "layer_payload_hash": self.layer_payload_hash, "binding_role": self.binding_role, "binding_metadata": dict(self.binding_metadata)}
        _ensure_json_safe(p)
        return p

    def to_dict(self) -> dict:
        return dict(self._canonical_payload(), binding_hash=self.binding_hash)

    def stable_hash(self) -> str:
        return sha256_hex(self._canonical_payload())


@dataclass(frozen=True)
class LayeredLatticeProjectionSpec:
    projection_id: str
    projection_version: str
    node_bindings: tuple[LayeredNodeBinding, ...]
    edge_bindings: tuple[LayeredEdgeBinding, ...]
    projection_policy: str
    spec_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_bindings", tuple(self.node_bindings))
        object.__setattr__(self, "edge_bindings", tuple(self.edge_bindings))
        if not self.projection_id or not self.projection_version:
            raise ValueError("INVALID_INPUT")
        if self.projection_policy not in _ALLOWED_PROJECTION_POLICIES:
            raise ValueError("INVALID_INPUT")
        if not self.node_bindings and not self.edge_bindings:
            raise ValueError("INVALID_INPUT")
        if len(self.node_bindings) > MAX_LAYERED_NODE_BINDINGS or len(self.edge_bindings) > MAX_LAYERED_EDGE_BINDINGS:
            raise ValueError("INVALID_INPUT")
        if tuple(sorted(self.node_bindings, key=lambda b: (b.binding_id, b.node_id, b.binding_hash))) != self.node_bindings:
            raise ValueError("INVALID_INPUT")
        if tuple(sorted(self.edge_bindings, key=lambda b: (b.binding_id, b.edge_id, b.binding_hash))) != self.edge_bindings:
            raise ValueError("INVALID_INPUT")
        if len({b.binding_id for b in self.node_bindings}) != len(self.node_bindings):
            raise ValueError("INVALID_INPUT")
        if len({b.binding_id for b in self.edge_bindings}) != len(self.edge_bindings):
            raise ValueError("INVALID_INPUT")
        if len({b.node_id for b in self.node_bindings}) != len(self.node_bindings):
            raise ValueError("INVALID_INPUT")
        if len({b.edge_id for b in self.edge_bindings}) != len(self.edge_bindings):
            raise ValueError("INVALID_INPUT")
        if self.spec_hash and self.spec_hash != self.stable_hash():
            raise ValueError("INVALID_INPUT")

    def _canonical_payload(self) -> dict:
        p = {"projection_id": self.projection_id, "projection_version": self.projection_version, "node_bindings": [b._canonical_payload() for b in self.node_bindings], "edge_bindings": [b._canonical_payload() for b in self.edge_bindings], "projection_policy": self.projection_policy}
        _ensure_json_safe(p)
        return p

    def stable_hash(self) -> str:
        return sha256_hex(self._canonical_payload())

    def to_dict(self) -> dict:
        return dict(self._canonical_payload(), spec_hash=self.spec_hash)


@dataclass(frozen=True)
class LayeredTopologyIntegrityReceipt:
    projection_id: str
    semantic_lattice_graph_hash: str
    layered_receipt_hash: str
    base_hash: str
    layered_hash: str
    layer_spec_hash: str
    layer_payload_hash: str
    node_binding_hashes: tuple[str, ...]
    edge_binding_hashes: tuple[str, ...]
    node_binding_count: int
    edge_binding_count: int
    topology_integrity_hash: str
    receipt_hash: str
    def __post_init__(self)->None:
        if self.topology_integrity_hash and self.topology_integrity_hash != self._topology_hash(): raise ValueError("INVALID_INPUT")
        if self.receipt_hash and self.receipt_hash != self.stable_hash(): raise ValueError("INVALID_INPUT")
    def _topology_payload(self)->dict:
        return {"projection_id":self.projection_id,"semantic_lattice_graph_hash":self.semantic_lattice_graph_hash,"layered_receipt_hash":self.layered_receipt_hash,"base_hash":self.base_hash,"layered_hash":self.layered_hash,"layer_spec_hash":self.layer_spec_hash,"layer_payload_hash":self.layer_payload_hash,"node_binding_hashes":list(self.node_binding_hashes),"edge_binding_hashes":list(self.edge_binding_hashes),"node_binding_count":self.node_binding_count,"edge_binding_count":self.edge_binding_count}
    def _topology_hash(self)->str:
        return sha256_hex(self._topology_payload())
    def _canonical_payload(self)->dict:
        p=dict(self._topology_payload(),topology_integrity_hash=self.topology_integrity_hash); _ensure_json_safe(p); return p
    def stable_hash(self)->str: return sha256_hex(self._canonical_payload())


@dataclass(frozen=True)
class LayeredLatticeProjectionReceipt:
    projection_id: str
    lattice_id: str
    lattice_version: str
    semantic_lattice_graph_hash: str
    layered_receipt_hash: str
    base_hash: str
    layer_spec_hash: str
    layer_payload_hash: str
    layered_hash: str
    projection_spec_hash: str
    topology_integrity_receipt_hash: str
    node_binding_count: int
    edge_binding_count: int
    layered_lattice_projection_hash: str
    receipt_hash: str
    def __post_init__(self)->None:
        if self.receipt_hash and self.receipt_hash != self.stable_hash(): raise ValueError("INVALID_INPUT")
    def _canonical_payload(self)->dict:
        p={"projection_id":self.projection_id,"lattice_id":self.lattice_id,"lattice_version":self.lattice_version,"semantic_lattice_graph_hash":self.semantic_lattice_graph_hash,"layered_receipt_hash":self.layered_receipt_hash,"base_hash":self.base_hash,"layer_spec_hash":self.layer_spec_hash,"layer_payload_hash":self.layer_payload_hash,"layered_hash":self.layered_hash,"projection_spec_hash":self.projection_spec_hash,"topology_integrity_receipt_hash":self.topology_integrity_receipt_hash,"node_binding_count":self.node_binding_count,"edge_binding_count":self.edge_binding_count,"layered_lattice_projection_hash":self.layered_lattice_projection_hash}
        _ensure_json_safe(p); return p
    def stable_hash(self)->str: return sha256_hex(self._canonical_payload())
    def to_dict(self)->dict: return dict(self._canonical_payload(),receipt_hash=self.receipt_hash)


def build_layered_lattice_projection_spec(projection_id: str, projection_version: str, node_binding_specs: tuple[LayeredNodeBinding | Mapping[str, Any], ...], edge_binding_specs: tuple[LayeredEdgeBinding | Mapping[str, Any], ...], projection_policy: str = "STRICT_EXPLICIT_BINDINGS") -> LayeredLatticeProjectionSpec:
    n=[]; e=[]
    for x in node_binding_specs:
        b=x if isinstance(x,LayeredNodeBinding) else LayeredNodeBinding(**x)
        n.append(LayeredNodeBinding(**{**b.__dict__,"binding_hash":b.stable_hash()}))
    for x in edge_binding_specs:
        b=x if isinstance(x,LayeredEdgeBinding) else LayeredEdgeBinding(**x)
        e.append(LayeredEdgeBinding(**{**b.__dict__,"binding_hash":b.stable_hash()}))
    spec=LayeredLatticeProjectionSpec(projection_id,projection_version,tuple(sorted(n,key=lambda b:(b.binding_id,b.node_id,b.binding_hash))),tuple(sorted(e,key=lambda b:(b.binding_id,b.edge_id,b.binding_hash))),projection_policy,"")
    return LayeredLatticeProjectionSpec(**{**spec.__dict__,"spec_hash":spec.stable_hash()})


def build_layered_topology_integrity_receipt(graph: SemanticLatticeGraph, layered_receipt: LayeredReceipt, projection_spec: LayeredLatticeProjectionSpec) -> LayeredTopologyIntegrityReceipt:
    _validate_graph(graph)
    if layered_receipt.receipt_hash != layered_receipt.stable_hash(): raise ValueError("INVALID_INPUT")
    if projection_spec.spec_hash != projection_spec.stable_hash(): raise ValueError("INVALID_INPUT")
    node_ids={n.node_id for n in graph.nodes}; edge_ids={e.edge_id for e in graph.edges}
    for b in projection_spec.node_bindings:
        if b.node_id not in node_ids or b.base_hash!=layered_receipt.base_hash or b.layered_hash!=layered_receipt.layered_hash or b.layer_spec_hash!=layered_receipt.layer_spec_hash or b.layer_payload_hash!=layered_receipt.layer_payload_hash or b.binding_hash!=b.stable_hash(): raise ValueError("INVALID_INPUT")
    for b in projection_spec.edge_bindings:
        if b.edge_id not in edge_ids or b.base_hash!=layered_receipt.base_hash or b.layered_hash!=layered_receipt.layered_hash or b.layer_spec_hash!=layered_receipt.layer_spec_hash or b.layer_payload_hash!=layered_receipt.layer_payload_hash or b.binding_hash!=b.stable_hash(): raise ValueError("INVALID_INPUT")
    payload = _layered_topology_integrity_payload(projection_spec.projection_id, graph.stable_hash(), layered_receipt.stable_hash(), layered_receipt.base_hash, layered_receipt.layered_hash, layered_receipt.layer_spec_hash, layered_receipt.layer_payload_hash, tuple(b.binding_hash for b in projection_spec.node_bindings), tuple(b.binding_hash for b in projection_spec.edge_bindings), len(projection_spec.node_bindings), len(projection_spec.edge_bindings))
    topology_integrity_hash = _compute_topology_integrity_hash(payload)
    receipt_hash = _compute_topology_receipt_hash(payload, topology_integrity_hash)
    return LayeredTopologyIntegrityReceipt(**{**payload, "node_binding_hashes": tuple(payload["node_binding_hashes"]), "edge_binding_hashes": tuple(payload["edge_binding_hashes"]), "topology_integrity_hash": topology_integrity_hash, "receipt_hash": receipt_hash})


def build_layered_lattice_projection_receipt(graph: SemanticLatticeGraph, layered_receipt: LayeredReceipt, projection_spec: LayeredLatticeProjectionSpec) -> LayeredLatticeProjectionReceipt:
    top=build_layered_topology_integrity_receipt(graph,layered_receipt,projection_spec)
    proj_hash=_compute_layered_lattice_projection_hash(graph.stable_hash(), layered_receipt.stable_hash(), projection_spec.stable_hash(), top.receipt_hash, [b._canonical_payload() for b in projection_spec.node_bindings], [b._canonical_payload() for b in projection_spec.edge_bindings])
    payload = _layered_lattice_projection_payload(projection_spec.projection_id, graph.lattice_id, graph.lattice_version, graph.stable_hash(), layered_receipt.stable_hash(), layered_receipt.base_hash, layered_receipt.layer_spec_hash, layered_receipt.layer_payload_hash, layered_receipt.layered_hash, projection_spec.stable_hash(), top.receipt_hash, len(projection_spec.node_bindings), len(projection_spec.edge_bindings), proj_hash)
    rec=LayeredLatticeProjectionReceipt(**{**payload, "receipt_hash": _compute_projection_receipt_hash(payload)})
    validate_layered_lattice_projection_receipt(rec,graph,layered_receipt,projection_spec)
    return rec


def validate_layered_lattice_projection_receipt(receipt: LayeredLatticeProjectionReceipt, graph: SemanticLatticeGraph, layered_receipt: LayeredReceipt, projection_spec: LayeredLatticeProjectionSpec) -> None:
    _validate_graph(graph)
    if layered_receipt.receipt_hash != layered_receipt.stable_hash() or projection_spec.spec_hash!=projection_spec.stable_hash(): raise ValueError("INVALID_INPUT")
    top=build_layered_topology_integrity_receipt(graph,layered_receipt,projection_spec)
    proj_hash=_compute_layered_lattice_projection_hash(graph.stable_hash(), layered_receipt.stable_hash(), projection_spec.stable_hash(), top.receipt_hash, [b._canonical_payload() for b in projection_spec.node_bindings], [b._canonical_payload() for b in projection_spec.edge_bindings])
    payload = _layered_lattice_projection_payload(projection_spec.projection_id, graph.lattice_id, graph.lattice_version, graph.stable_hash(), layered_receipt.stable_hash(), layered_receipt.base_hash, layered_receipt.layer_spec_hash, layered_receipt.layer_payload_hash, layered_receipt.layered_hash, projection_spec.stable_hash(), top.receipt_hash, len(projection_spec.node_bindings), len(projection_spec.edge_bindings), proj_hash)
    expected=LayeredLatticeProjectionReceipt(**{**payload, "receipt_hash": _compute_projection_receipt_hash(payload)})
    if receipt.__dict__!=expected.__dict__: raise ValueError("INVALID_INPUT")


_FORBIDDEN_V153_3_SCOPE_ATTRIBUTES = ("apply", "execute", "run", "traverse", "pathfind", "resolve", "readout", "search", "mask", "hilber", "hilbert", "markov")


def _assert_no_v153_3_forbidden_scope() -> None:
    for _cls in (LayeredNodeBinding, LayeredEdgeBinding, LayeredLatticeProjectionSpec, LayeredTopologyIntegrityReceipt, LayeredLatticeProjectionReceipt):
        for _name in _FORBIDDEN_V153_3_SCOPE_ATTRIBUTES:
            if hasattr(_cls, _name):
                raise RuntimeError("INVALID_STATE")


_assert_no_v153_3_forbidden_scope()

__all__ = [
    "LayeredNodeBinding", "LayeredEdgeBinding", "LayeredLatticeProjectionSpec", "LayeredTopologyIntegrityReceipt", "LayeredLatticeProjectionReceipt",
    "build_layered_lattice_projection_spec", "build_layered_topology_integrity_receipt", "build_layered_lattice_projection_receipt", "validate_layered_lattice_projection_receipt",
]
