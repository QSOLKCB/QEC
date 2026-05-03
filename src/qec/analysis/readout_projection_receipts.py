from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from qec.analysis.atomic_semantic_lattice_contract import SemanticLatticeGraph, _validate_graph_internal_consistency
from qec.analysis.canonical_hashing import canonical_json, sha256_hex
from qec.analysis.layer_spec_contract import _ensure_json_safe
from qec.analysis.router_lattice_paths import ResolvedLatticePathSet, RouterPathSpec, SpecialPathIndex

MAX_READOUT_FIELDS = 128
_ALLOWED_SOURCE_TYPES = {"NODE", "EDGE", "PATH"}
_ALLOWED_PROJECTION_MODES = {"IDENTITY_HASH", "METADATA_VALUE", "COORDINATE", "CONSTRAINT_PAYLOAD_VALUE", "PATH_IDENTITY"}




def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({k: _deep_freeze(value[k]) for k in sorted(value)})
    if isinstance(value, list):
        return tuple(_deep_freeze(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_deep_freeze(v) for v in value)
    return value


def _freeze_mapping(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    return _deep_freeze(dict(mapping))


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _deep_thaw(value[k]) for k in sorted(value)}
    if isinstance(value, tuple):
        return [_deep_thaw(v) for v in value]
    if isinstance(value, list):
        return [_deep_thaw(v) for v in value]
    return value


def _resolve_key_path(value: Any, key_path: tuple[str, ...]) -> Any:
    current = value
    for key in key_path:
        if not isinstance(current, Mapping) or key not in current:
            raise ValueError("INVALID_INPUT")
        current = current[key]
    _ensure_json_safe(current)
    return _deep_freeze(current)


def _validate_graph(graph: SemanticLatticeGraph) -> None:
    _validate_graph_internal_consistency(graph)
    if graph.graph_hash != graph.stable_hash():
        raise ValueError("INVALID_INPUT")


@dataclass(frozen=True)
class ReadoutFieldSpec:
    field_id: str
    source_type: str
    path_id: str
    source_id: str
    projection_mode: str
    key_path: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "key_path", tuple(self.key_path))
        if not self.field_id or not self.path_id or not self.source_id:
            raise ValueError("INVALID_INPUT")
        if self.source_type not in _ALLOWED_SOURCE_TYPES or self.projection_mode not in _ALLOWED_PROJECTION_MODES:
            raise ValueError("INVALID_INPUT")
        if any((not isinstance(k, str) or not k) for k in self.key_path):
            raise ValueError("INVALID_INPUT")
        if self.projection_mode in {"METADATA_VALUE", "CONSTRAINT_PAYLOAD_VALUE"} and not self.key_path:
            raise ValueError("INVALID_INPUT")
        if self.projection_mode == "COORDINATE" and self.source_type != "NODE":
            raise ValueError("INVALID_INPUT")
        if self.projection_mode == "CONSTRAINT_PAYLOAD_VALUE" and self.source_type != "EDGE":
            raise ValueError("INVALID_INPUT")
        if self.projection_mode == "PATH_IDENTITY" and self.source_type != "PATH":
            raise ValueError("INVALID_INPUT")
        if self.source_type == "PATH" and self.source_id != self.path_id:
            raise ValueError("INVALID_INPUT")

    def _canonical_payload(self) -> dict:
        p = {"field_id": self.field_id, "source_type": self.source_type, "path_id": self.path_id, "source_id": self.source_id, "projection_mode": self.projection_mode, "key_path": list(self.key_path)}
        _ensure_json_safe(p)
        return p


@dataclass(frozen=True)
class ReadoutProjectionSpec:
    projection_id: str
    fields: tuple[ReadoutFieldSpec, ...]
    readout_projection_spec_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", tuple(self.fields))
        if not self.projection_id or not self.fields or len(self.fields) > MAX_READOUT_FIELDS:
            raise ValueError("INVALID_INPUT")
        if tuple(sorted(self.fields, key=lambda f: (f.field_id, f.source_type, f.path_id, f.source_id, f.projection_mode, f.key_path))) != self.fields:
            raise ValueError("INVALID_INPUT")
        if len({f.field_id for f in self.fields}) != len(self.fields):
            raise ValueError("INVALID_INPUT")
        if self.readout_projection_spec_hash and self.readout_projection_spec_hash != self.stable_hash():
            raise ValueError("INVALID_INPUT")

    def _canonical_payload(self) -> dict:
        p = {"projection_id": self.projection_id, "fields": [f._canonical_payload() for f in self.fields]}
        _ensure_json_safe(p)
        return p

    def stable_hash(self) -> str:
        return sha256_hex(self._canonical_payload())


@dataclass(frozen=True)
class ProjectedReadoutField:
    field_id: str
    source_type: str
    path_id: str
    source_id: str
    projection_mode: str
    projected_value: Any
    projected_value_hash: str
    projected_field_hash: str

    def __post_init__(self) -> None:
        if isinstance(self.projected_value, Mapping):
            object.__setattr__(self, "projected_value", _freeze_mapping(self.projected_value))
        else:
            object.__setattr__(self, "projected_value", _deep_freeze(self.projected_value))
        _ensure_json_safe(self.projected_value)
        if self.projected_value_hash != sha256_hex(self.projected_value):
            raise ValueError("INVALID_INPUT")
        if self.projected_field_hash and self.projected_field_hash != self.stable_hash():
            raise ValueError("INVALID_INPUT")

    def _canonical_payload(self) -> dict:
        p = {"field_id": self.field_id, "source_type": self.source_type, "path_id": self.path_id, "source_id": self.source_id, "projection_mode": self.projection_mode, "projected_value": _deep_thaw(self.projected_value), "projected_value_hash": self.projected_value_hash}
        _ensure_json_safe(p)
        return p

    def to_dict(self) -> dict:
        return dict(self._canonical_payload(), projected_field_hash=self.projected_field_hash)

    def stable_hash(self) -> str:
        return sha256_hex(self._canonical_payload())


@dataclass(frozen=True)
class ReadoutProjectionSet:
    graph_hash: str
    router_path_spec_hash: str
    special_path_index_hash: str
    resolved_path_hash: str
    readout_projection_spec_hash: str
    projected_fields: tuple[ProjectedReadoutField, ...]
    field_count: int
    projection_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "projected_fields", tuple(self.projected_fields))
        if self.field_count != len(self.projected_fields):
            raise ValueError("INVALID_INPUT")
        if tuple(sorted(self.projected_fields, key=lambda f: (f.field_id, f.projected_field_hash))) != self.projected_fields:
            raise ValueError("INVALID_INPUT")
        if self.projection_hash and self.projection_hash != self.stable_hash():
            raise ValueError("INVALID_INPUT")

    def _canonical_payload(self) -> dict:
        p = {"graph_hash": self.graph_hash, "router_path_spec_hash": self.router_path_spec_hash, "special_path_index_hash": self.special_path_index_hash, "resolved_path_hash": self.resolved_path_hash, "readout_projection_spec_hash": self.readout_projection_spec_hash, "projected_fields": [f._canonical_payload() for f in self.projected_fields], "field_count": self.field_count}
        _ensure_json_safe(p)
        return p

    def stable_hash(self) -> str:
        return sha256_hex(self._canonical_payload())

    def to_dict(self) -> dict:
        payload = dict(self._canonical_payload())
        payload["projected_fields"] = [field.to_dict() for field in self.projected_fields]
        payload["projection_hash"] = self.projection_hash
        return payload


@dataclass(frozen=True)
class ReadoutProjectionReceipt:
    lattice_id: str
    lattice_version: str
    semantic_lattice_graph_hash: str
    router_path_spec_hash: str
    special_path_index_hash: str
    resolved_path_hash: str
    readout_projection_spec_hash: str
    projection_hash: str
    field_count: int
    receipt_hash: str

    def __post_init__(self) -> None:
        if self.receipt_hash and self.receipt_hash != self.stable_hash():
            raise ValueError("INVALID_INPUT")

    def _canonical_payload(self) -> dict:
        p = {"lattice_id": self.lattice_id, "lattice_version": self.lattice_version, "semantic_lattice_graph_hash": self.semantic_lattice_graph_hash, "router_path_spec_hash": self.router_path_spec_hash, "special_path_index_hash": self.special_path_index_hash, "resolved_path_hash": self.resolved_path_hash, "readout_projection_spec_hash": self.readout_projection_spec_hash, "projection_hash": self.projection_hash, "field_count": self.field_count}
        _ensure_json_safe(p)
        return p

    def stable_hash(self) -> str:
        return sha256_hex(self._canonical_payload())

    def to_dict(self) -> dict:
        return dict(self._canonical_payload(), receipt_hash=self.receipt_hash)


def build_readout_projection_spec(projection_id: str, field_specs: tuple[ReadoutFieldSpec | Mapping[str, Any], ...]) -> ReadoutProjectionSpec:
    fields = []
    for raw in field_specs:
        f = raw if isinstance(raw, ReadoutFieldSpec) else ReadoutFieldSpec(**raw)
        fields.append(f)
    ordered = tuple(sorted(fields, key=lambda f: (f.field_id, f.source_type, f.path_id, f.source_id, f.projection_mode, f.key_path)))
    spec = ReadoutProjectionSpec(projection_id, ordered, "")
    return ReadoutProjectionSpec(**{**spec.__dict__, "readout_projection_spec_hash": spec.stable_hash()})


def project_readout_fields(graph: SemanticLatticeGraph, resolved_path_set: ResolvedLatticePathSet, readout_projection_spec: ReadoutProjectionSpec, router_path_spec: RouterPathSpec, special_path_index: SpecialPathIndex) -> ReadoutProjectionSet:
    _validate_graph(graph)
    if resolved_path_set.graph_hash != graph.stable_hash() or resolved_path_set.resolved_path_hash != resolved_path_set.stable_hash(): raise ValueError("INVALID_INPUT")
    if resolved_path_set.router_path_spec_hash != router_path_spec.stable_hash() or resolved_path_set.special_path_index_hash != special_path_index.stable_hash(): raise ValueError("INVALID_INPUT")
    if readout_projection_spec.readout_projection_spec_hash != readout_projection_spec.stable_hash(): raise ValueError("INVALID_INPUT")
    paths = {p.path_id: p for p in resolved_path_set.resolved_paths}
    nodes = {n.node_id: n for n in graph.nodes}
    edges = {e.edge_id: e for e in graph.edges}
    out = []
    for f in readout_projection_spec.fields:
        if f.path_id not in paths: raise ValueError("INVALID_INPUT")
        path = paths[f.path_id]
        if f.source_type == "NODE":
            if f.source_id not in nodes or f.source_id not in path.node_ids: raise ValueError("INVALID_INPUT")
            src = nodes[f.source_id]
            val = src.node_hash if f.projection_mode == "IDENTITY_HASH" else (list(src.coordinate) if f.projection_mode == "COORDINATE" else _resolve_key_path(src.node_metadata, f.key_path))
        elif f.source_type == "EDGE":
            if f.source_id not in edges or f.source_id not in path.edge_ids: raise ValueError("INVALID_INPUT")
            src = edges[f.source_id]
            val = src.edge_hash if f.projection_mode == "IDENTITY_HASH" else _resolve_key_path(src.constraint_payload, f.key_path)
        else:
            if f.source_id != f.path_id: raise ValueError("INVALID_INPUT")
            val = path.path_hash if f.projection_mode == "IDENTITY_HASH" else {"path_id": path.path_id, "path_hash": path.path_hash}
        vhash = sha256_hex(val)
        pf = ProjectedReadoutField(f.field_id, f.source_type, f.path_id, f.source_id, f.projection_mode, val, vhash, "")
        out.append(ProjectedReadoutField(**{**pf.__dict__, "projected_field_hash": pf.stable_hash()}))
    ordered = tuple(sorted(out, key=lambda x: (x.field_id, x.projected_field_hash)))
    pset = ReadoutProjectionSet(graph.stable_hash(), router_path_spec.stable_hash(), special_path_index.stable_hash(), resolved_path_set.resolved_path_hash, readout_projection_spec.stable_hash(), ordered, len(ordered), "")
    return ReadoutProjectionSet(**{**pset.__dict__, "projection_hash": pset.stable_hash()})


def build_readout_projection_receipt(graph: SemanticLatticeGraph, router_path_spec: RouterPathSpec, special_path_index: SpecialPathIndex, resolved_path_set: ResolvedLatticePathSet, readout_projection_spec: ReadoutProjectionSpec) -> ReadoutProjectionReceipt:
    pset = project_readout_fields(graph, resolved_path_set, readout_projection_spec, router_path_spec, special_path_index)
    r = ReadoutProjectionReceipt(graph.lattice_id, graph.lattice_version, graph.stable_hash(), router_path_spec.stable_hash(), special_path_index.stable_hash(), resolved_path_set.stable_hash(), readout_projection_spec.stable_hash(), pset.projection_hash, pset.field_count, "")
    out = ReadoutProjectionReceipt(**{**r.__dict__, "receipt_hash": r.stable_hash()})
    validate_readout_projection_receipt(out, graph, router_path_spec, special_path_index, resolved_path_set, readout_projection_spec)
    return out


def validate_readout_projection_receipt(receipt: ReadoutProjectionReceipt, graph: SemanticLatticeGraph, router_path_spec: RouterPathSpec, special_path_index: SpecialPathIndex, resolved_path_set: ResolvedLatticePathSet, readout_projection_spec: ReadoutProjectionSpec) -> None:
    pset = project_readout_fields(graph, resolved_path_set, readout_projection_spec, router_path_spec, special_path_index)
    if receipt.semantic_lattice_graph_hash != graph.stable_hash() or receipt.router_path_spec_hash != router_path_spec.stable_hash() or receipt.special_path_index_hash != special_path_index.stable_hash(): raise ValueError("INVALID_INPUT")
    if receipt.resolved_path_hash != resolved_path_set.stable_hash() or receipt.readout_projection_spec_hash != readout_projection_spec.stable_hash(): raise ValueError("INVALID_INPUT")
    if receipt.projection_hash != pset.projection_hash or receipt.field_count != pset.field_count: raise ValueError("INVALID_INPUT")
    if receipt.receipt_hash != ReadoutProjectionReceipt(**{**receipt.__dict__, "receipt_hash": ""}).stable_hash(): raise ValueError("INVALID_INPUT")


_FORBIDDEN_V153_2_SCOPE_ATTRIBUTES = (
    "apply", "execute", "run", "readout", "project", "traverse", "search", "mask", "hilber", "hilbert",
)


def _assert_no_v153_2_forbidden_scope() -> None:
    for _cls in (ReadoutFieldSpec, ReadoutProjectionSpec, ProjectedReadoutField, ReadoutProjectionSet, ReadoutProjectionReceipt):
        for _name in _FORBIDDEN_V153_2_SCOPE_ATTRIBUTES:
            if hasattr(_cls, _name):
                raise RuntimeError("INVALID_STATE")


_assert_no_v153_2_forbidden_scope()

__all__ = [
    "ReadoutFieldSpec", "ReadoutProjectionSpec", "ProjectedReadoutField", "ReadoutProjectionSet", "ReadoutProjectionReceipt",
    "build_readout_projection_spec", "project_readout_fields", "build_readout_projection_receipt", "validate_readout_projection_receipt", "MAX_READOUT_FIELDS",
]
