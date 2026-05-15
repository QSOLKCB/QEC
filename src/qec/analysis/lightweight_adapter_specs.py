from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

from .optimization_contracts import OptimizationContract, validate_optimization_contract, _ALLOWED_NEXT_RECEIPTS

_SCHEMA_VERSION = "LIGHTWEIGHT_ADAPTER_SPEC_V1"
_SPEC_MODE = "DETERMINISTIC_LIGHTWEIGHT_ADAPTER_SPEC"
_MAX_OPERATION_SPECS = 64
_MAX_BOUNDARY_SPECS = 64
_MAX_CAPABILITY_SPECS = 64
_MAX_NAME_LENGTH = 128
_MAX_REASON_LENGTH = 256
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

_ALLOWED_ADAPTER_STATUS = {"ADAPTER_SPEC_DRAFT", "ADAPTER_SPEC_READY", "ADAPTER_SPEC_BLOCKED"}
_ALLOWED_ADAPTER_KIND = {
    "IMPORT_SURFACE_ADAPTER", "PLOTTING_RENDER_ADAPTER", "DATAFRAME_SCHEMA_ADAPTER", "SPARSE_DENSE_BOUNDARY_ADAPTER",
    "QUANTUM_BACKEND_ADAPTER", "AUDIO_MIDI_ADAPTER", "INTERNAL_QEC_ADAPTER", "HASH_ONLY_EQUIVALENCE_ADAPTER", "EXACT_JSON_EQUIVALENCE_ADAPTER",
}
_ALLOWED_BOUNDARY_KIND = {"INPUT_BOUNDARY", "OUTPUT_BOUNDARY", "ERROR_BOUNDARY", "AVAILABILITY_BOUNDARY", "POLICY_BOUNDARY"}
_ALLOWED_OPERATION_KIND = {
    "NORMALIZE_INPUT", "CANONICALIZE_OUTPUT", "VALIDATE_SHAPE_DTYPE", "VALIDATE_HASH_ONLY", "VALIDATE_ORDERED_SEQUENCE",
    "VALIDATE_SET_LIKE_SEQUENCE", "DECLARE_UNAVAILABLE", "DECLARE_ERROR", "EXPORT_CANONICAL_PAYLOAD",
}
_ALLOWED_CAPABILITY_KIND = {
    "READ_ONLY", "CANONICAL_JSON_OUTPUT", "HASH_ONLY_OUTPUT", "SHAPE_DTYPE_OUTPUT", "ORDERED_SEQUENCE_OUTPUT",
    "SET_LIKE_SEQUENCE_OUTPUT", "ERROR_RESULT_OUTPUT", "UNAVAILABLE_RESULT_OUTPUT", "NO_BACKEND_EXECUTION", "NO_NETWORK_EXECUTION", "NO_RUNTIME_IMPORT",
}
_SCOPE_TO_ADAPTER_KIND = {
    "IMPORT_SURFACE_REDUCTION": "IMPORT_SURFACE_ADAPTER",
    "TOP_LEVEL_IMPORT_DEFERRAL": "IMPORT_SURFACE_ADAPTER",
    "REPEATED_IMPORT_COLLAPSE": "IMPORT_SURFACE_ADAPTER",
    "PLOTTING_RENDER_BYPASS": "PLOTTING_RENDER_ADAPTER",
    "DATAFRAME_SCHEMA_CACHE_REVIEW": "DATAFRAME_SCHEMA_ADAPTER",
    "SPARSE_DENSE_BOUNDARY_REVIEW": "SPARSE_DENSE_BOUNDARY_ADAPTER",
    "QUANTUM_BACKEND_ADAPTER_REVIEW": "QUANTUM_BACKEND_ADAPTER",
    "AUDIO_MIDI_ADAPTER_REVIEW": "AUDIO_MIDI_ADAPTER",
    "INTERNAL_QEC_FASTPATH_REVIEW": "INTERNAL_QEC_ADAPTER",
    "HASH_ONLY_EQUIVALENCE_REVIEW": "HASH_ONLY_EQUIVALENCE_ADAPTER",
    "EXACT_JSON_EQUIVALENCE_REVIEW": "EXACT_JSON_EQUIVALENCE_ADAPTER",
}


def _canonical_json(obj: Any) -> str: return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
def _hash_payload(obj: Any) -> str: return hashlib.sha256(_canonical_json(obj).encode("utf-8")).hexdigest()
def _base_payload(x: Any, key: str) -> dict[str, Any]: d = x.to_dict(); d.pop(key); return d

def _validate_hash_format(v: str) -> None:
    if not isinstance(v, str) or _HASH_RE.fullmatch(v) is None: raise ValueError("INVALID_HASH_FORMAT")

def _bounded(v: str, max_len: int = _MAX_NAME_LENGTH) -> bool: return isinstance(v, str) and bool(v) and len(v) <= max_len


@dataclass(frozen=True)
class AdapterBoundarySpec:
    boundary_index: int
    boundary_kind: str
    boundary_name: str
    dependency_name: str
    allowed_payload_kind: str
    validation_policy: str
    reason: str
    boundary_hash: str
    def to_dict(self) -> dict[str, Any]: return self.__dict__.copy()
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class AdapterOperationSpec:
    operation_index: int
    operation_kind: str
    operation_name: str
    dependency_name: str
    input_boundary_hashes: tuple[str, ...]
    output_boundary_hashes: tuple[str, ...]
    required: bool
    reason: str
    operation_hash: str
    def to_dict(self) -> dict[str, Any]: return {**self.__dict__, "input_boundary_hashes": list(self.input_boundary_hashes), "output_boundary_hashes": list(self.output_boundary_hashes)}
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class AdapterCapabilitySpec:
    capability_index: int
    capability_kind: str
    dependency_name: str
    capability_name: str
    enabled: bool
    reason: str
    capability_hash: str
    def to_dict(self) -> dict[str, Any]: return self.__dict__.copy()
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class LightweightAdapterSpec:
    schema_version: str
    spec_mode: str
    adapter_name: str
    adapter_status: str
    adapter_kind: str
    dependency_name: str
    source_optimization_contract_hash: str
    source_opportunity_hash: str
    optimization_scope: str
    contract_required_next_receipt: str
    boundary_count: int
    operation_count: int
    capability_count: int
    boundaries: tuple[AdapterBoundarySpec, ...]
    operations: tuple[AdapterOperationSpec, ...]
    capabilities: tuple[AdapterCapabilitySpec, ...]
    first_boundary_hash: str
    final_boundary_hash: str
    first_operation_hash: str
    final_operation_hash: str
    first_capability_hash: str
    final_capability_hash: str
    lightweight_adapter_spec_hash: str
    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "boundaries": [x.to_dict() for x in self.boundaries], "operations": [x.to_dict() for x in self.operations], "capabilities": [x.to_dict() for x in self.capabilities]}
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")


def build_adapter_boundary_spec(**kwargs: Any) -> AdapterBoundarySpec:
    k = dict(kwargs); k.pop("boundary_hash", None)
    x = AdapterBoundarySpec(boundary_hash="", **k)
    validate_adapter_boundary_spec(x, allow_blank_hash=True)
    return AdapterBoundarySpec(**{**x.to_dict(), "boundary_hash": _hash_payload(_base_payload(x, "boundary_hash"))})


def build_adapter_operation_spec(**kwargs: Any) -> AdapterOperationSpec:
    k = dict(kwargs); k.pop("operation_hash", None)
    k["input_boundary_hashes"] = tuple(k.get("input_boundary_hashes", ()))
    k["output_boundary_hashes"] = tuple(k.get("output_boundary_hashes", ()))
    x = AdapterOperationSpec(operation_hash="", **k)
    validate_adapter_operation_spec(x, allow_blank_hash=True)
    return AdapterOperationSpec(**{**x.__dict__, "operation_hash": _hash_payload(_base_payload(x, "operation_hash"))})


def build_adapter_capability_spec(**kwargs: Any) -> AdapterCapabilitySpec:
    k = dict(kwargs); k.pop("capability_hash", None)
    x = AdapterCapabilitySpec(capability_hash="", **k)
    validate_adapter_capability_spec(x, allow_blank_hash=True)
    return AdapterCapabilitySpec(**{**x.to_dict(), "capability_hash": _hash_payload(_base_payload(x, "capability_hash"))})


def validate_adapter_boundary_spec(boundary: AdapterBoundarySpec, allow_blank_hash: bool = False) -> bool:
    if not isinstance(boundary, AdapterBoundarySpec): raise ValueError("INVALID_INPUT")
    if not isinstance(boundary.boundary_index, int) or isinstance(boundary.boundary_index, bool) or boundary.boundary_index < 0: raise ValueError("INVALID_INPUT")
    if boundary.boundary_kind not in _ALLOWED_BOUNDARY_KIND: raise ValueError("INVALID_BOUNDARY_KIND")
    if not _bounded(boundary.boundary_name): raise ValueError("INVALID_INPUT")
    if not _bounded(boundary.dependency_name): raise ValueError("INVALID_INPUT")
    if not _bounded(boundary.allowed_payload_kind): raise ValueError("INVALID_INPUT")
    if not _bounded(boundary.validation_policy): raise ValueError("INVALID_INPUT")
    if not isinstance(boundary.reason, str) or len(boundary.reason) > _MAX_REASON_LENGTH: raise ValueError("INVALID_INPUT")
    exp = _hash_payload(_base_payload(boundary, "boundary_hash"))
    if boundary.boundary_hash == "" and allow_blank_hash: return True
    _validate_hash_format(boundary.boundary_hash)
    if boundary.boundary_hash != exp: raise ValueError("HASH_MISMATCH")
    return True


def validate_adapter_operation_spec(operation: AdapterOperationSpec, allow_blank_hash: bool = False) -> bool:
    if not isinstance(operation, AdapterOperationSpec): raise ValueError("INVALID_INPUT")
    if not isinstance(operation.operation_index, int) or isinstance(operation.operation_index, bool) or operation.operation_index < 0: raise ValueError("INVALID_INPUT")
    if operation.operation_kind not in _ALLOWED_OPERATION_KIND: raise ValueError("INVALID_OPERATION_KIND")
    if not _bounded(operation.operation_name): raise ValueError("INVALID_INPUT")
    if not _bounded(operation.dependency_name): raise ValueError("INVALID_INPUT")
    if not isinstance(operation.input_boundary_hashes, tuple) or not isinstance(operation.output_boundary_hashes, tuple): raise ValueError("INVALID_INPUT")
    for h in operation.input_boundary_hashes + operation.output_boundary_hashes: _validate_hash_format(h)
    if not isinstance(operation.required, bool): raise ValueError("INVALID_INPUT")
    if not isinstance(operation.reason, str) or len(operation.reason) > _MAX_REASON_LENGTH: raise ValueError("INVALID_INPUT")
    exp = _hash_payload(_base_payload(operation, "operation_hash"))
    if operation.operation_hash == "" and allow_blank_hash: return True
    _validate_hash_format(operation.operation_hash)
    if operation.operation_hash != exp: raise ValueError("HASH_MISMATCH")
    return True


def validate_adapter_capability_spec(capability: AdapterCapabilitySpec, allow_blank_hash: bool = False) -> bool:
    if not isinstance(capability, AdapterCapabilitySpec): raise ValueError("INVALID_INPUT")
    if not isinstance(capability.capability_index, int) or isinstance(capability.capability_index, bool) or capability.capability_index < 0: raise ValueError("INVALID_INPUT")
    if capability.capability_kind not in _ALLOWED_CAPABILITY_KIND: raise ValueError("INVALID_CAPABILITY_KIND")
    if not _bounded(capability.dependency_name): raise ValueError("INVALID_INPUT")
    if not _bounded(capability.capability_name): raise ValueError("INVALID_INPUT")
    if not isinstance(capability.enabled, bool): raise ValueError("INVALID_INPUT")
    if not isinstance(capability.reason, str) or len(capability.reason) > _MAX_REASON_LENGTH: raise ValueError("INVALID_INPUT")
    exp = _hash_payload(_base_payload(capability, "capability_hash"))
    if capability.capability_hash == "" and allow_blank_hash: return True
    _validate_hash_format(capability.capability_hash)
    if capability.capability_hash != exp: raise ValueError("HASH_MISMATCH")
    return True


def build_lightweight_adapter_spec(contract: OptimizationContract, adapter_name: str, adapter_status: str, adapter_kind: str, dependency_name: str, boundaries, operations, capabilities) -> LightweightAdapterSpec:
    validate_optimization_contract(contract)
    if adapter_status not in _ALLOWED_ADAPTER_STATUS: raise ValueError("INVALID_ADAPTER_STATUS")
    if adapter_kind not in _ALLOWED_ADAPTER_KIND: raise ValueError("INVALID_ADAPTER_KIND")
    if not _bounded(adapter_name): raise ValueError("INVALID_INPUT")
    if not _bounded(dependency_name): raise ValueError("INVALID_INPUT")
    bs = tuple(sorted(tuple(boundaries), key=lambda x: x.boundary_index)); os = tuple(sorted(tuple(operations), key=lambda x: x.operation_index)); cs = tuple(sorted(tuple(capabilities), key=lambda x: x.capability_index))
    if len(bs) > _MAX_BOUNDARY_SPECS or len(os) > _MAX_OPERATION_SPECS or len(cs) > _MAX_CAPABILITY_SPECS: raise ValueError("INVALID_INPUT")
    for x in bs: validate_adapter_boundary_spec(x)
    for x in os: validate_adapter_operation_spec(x)
    for x in cs: validate_adapter_capability_spec(x)
    if tuple(x.boundary_index for x in bs) != tuple(range(len(bs))): raise ValueError("BOUNDARY_ORDER_MISMATCH")
    if tuple(x.operation_index for x in os) != tuple(range(len(os))): raise ValueError("OPERATION_ORDER_MISMATCH")
    if tuple(x.capability_index for x in cs) != tuple(range(len(cs))): raise ValueError("CAPABILITY_ORDER_MISMATCH")
    all_bh = {b.boundary_hash for b in bs}
    for o in os:
        if any(h not in all_bh for h in (o.input_boundary_hashes + o.output_boundary_hashes)): raise ValueError("OPERATION_BOUNDARY_MISMATCH")
    s = LightweightAdapterSpec(_SCHEMA_VERSION, _SPEC_MODE, adapter_name, adapter_status, adapter_kind, dependency_name, contract.optimization_contract_hash, contract.source_opportunity_hash, contract.optimization_scope, contract.required_next_receipt, len(bs), len(os), len(cs), bs, os, cs, bs[0].boundary_hash if bs else "", bs[-1].boundary_hash if bs else "", os[0].operation_hash if os else "", os[-1].operation_hash if os else "", cs[0].capability_hash if cs else "", cs[-1].capability_hash if cs else "", "")
    return LightweightAdapterSpec(**{**s.__dict__, "lightweight_adapter_spec_hash": _hash_payload(_base_payload(s, "lightweight_adapter_spec_hash"))})


def _derive_adapter_kind(scope: str) -> str:
    if scope not in _SCOPE_TO_ADAPTER_KIND: raise ValueError("INVALID_OPTIMIZATION_SCOPE")
    return _SCOPE_TO_ADAPTER_KIND[scope]


def build_lightweight_adapter_spec_from_contract(contract: OptimizationContract, *, adapter_name: str | None = None) -> LightweightAdapterSpec:
    validate_optimization_contract(contract)
    if contract.contract_status != "CONTRACT_READY": raise ValueError("INVALID_INPUT")
    if contract.required_next_receipt not in _ALLOWED_NEXT_RECEIPTS: raise ValueError("INVALID_INPUT")
    adapter_kind = _derive_adapter_kind(contract.optimization_scope)
    dep = contract.dependency_name
    name = adapter_name or f"{dep}_{adapter_kind.lower()}"
    b_defs = [
        ("INPUT_BOUNDARY", "input_payload", "CANONICAL_JSON", "EXACT_CANONICAL_JSON", "Adapter input boundary declaration."),
        ("OUTPUT_BOUNDARY", "output_payload", "CANONICAL_JSON", "EXACT_CANONICAL_JSON", "Adapter output boundary declaration."),
        ("ERROR_BOUNDARY", "error_payload", "ERROR_RESULT", "DECLARED_ERROR_MATCH", "Declared error boundary only."),
        ("POLICY_BOUNDARY", "policy_payload", "POLICY_DECLARATION", "CONTRACT_POLICY_ONLY", "Contract policy boundary declaration."),
    ]
    if adapter_kind == "IMPORT_SURFACE_ADAPTER": b_defs.append(("AVAILABILITY_BOUNDARY", "availability_payload", "UNAVAILABLE_RESULT", "DECLARED_UNAVAILABLE_MATCH", "Declared unavailable boundary only."))
    bs = tuple(build_adapter_boundary_spec(boundary_index=i, boundary_kind=k, boundary_name=n, dependency_name=dep, allowed_payload_kind=ap, validation_policy=vp, reason=r) for i, (k, n, ap, vp, r) in enumerate(b_defs))
    in_h, out_h = (bs[0].boundary_hash,), (bs[1].boundary_hash,)
    op_kinds = ["NORMALIZE_INPUT", "CANONICALIZE_OUTPUT", "EXPORT_CANONICAL_PAYLOAD"]
    op_kinds += {
        "HASH_ONLY_EQUIVALENCE_ADAPTER": ["VALIDATE_HASH_ONLY"],
        "SPARSE_DENSE_BOUNDARY_ADAPTER": ["VALIDATE_SHAPE_DTYPE"],
        "DATAFRAME_SCHEMA_ADAPTER": ["VALIDATE_SHAPE_DTYPE", "VALIDATE_ORDERED_SEQUENCE"],
        "AUDIO_MIDI_ADAPTER": ["VALIDATE_ORDERED_SEQUENCE"],
        "QUANTUM_BACKEND_ADAPTER": ["VALIDATE_SHAPE_DTYPE", "VALIDATE_HASH_ONLY"],
        "INTERNAL_QEC_ADAPTER": ["VALIDATE_HASH_ONLY", "EXPORT_CANONICAL_PAYLOAD"],
        "IMPORT_SURFACE_ADAPTER": ["DECLARE_UNAVAILABLE", "DECLARE_ERROR"],
    }.get(adapter_kind, [])
    seen = set(); op_kinds = [x for x in op_kinds if (x not in seen and not seen.add(x))]
    os = tuple(build_adapter_operation_spec(operation_index=i, operation_kind=k, operation_name=f"{k.lower()}::{dep}", dependency_name=dep, input_boundary_hashes=in_h, output_boundary_hashes=out_h, required=True, reason=f"Deterministic adapter operation declaration: {k}") for i, k in enumerate(op_kinds))
    cap_kinds = ["READ_ONLY", "CANONICAL_JSON_OUTPUT", "NO_BACKEND_EXECUTION", "NO_NETWORK_EXECUTION", "NO_RUNTIME_IMPORT"] + {
        "HASH_ONLY_EQUIVALENCE_ADAPTER": ["HASH_ONLY_OUTPUT"],
        "SPARSE_DENSE_BOUNDARY_ADAPTER": ["SHAPE_DTYPE_OUTPUT"],
        "DATAFRAME_SCHEMA_ADAPTER": ["SHAPE_DTYPE_OUTPUT", "ORDERED_SEQUENCE_OUTPUT"],
        "AUDIO_MIDI_ADAPTER": ["ORDERED_SEQUENCE_OUTPUT"],
        "IMPORT_SURFACE_ADAPTER": ["ERROR_RESULT_OUTPUT", "UNAVAILABLE_RESULT_OUTPUT"],
    }.get(adapter_kind, [])
    seen = set(); cap_kinds = [x for x in cap_kinds if (x not in seen and not seen.add(x))]
    cs = tuple(build_adapter_capability_spec(capability_index=i, capability_kind=k, dependency_name=dep, capability_name=f"{k.lower()}::{dep}", enabled=True, reason=f"Deterministic adapter capability declaration: {k}") for i, k in enumerate(cap_kinds))
    return build_lightweight_adapter_spec(contract, name, "ADAPTER_SPEC_READY", adapter_kind, dep, bs, os, cs)


def validate_lightweight_adapter_spec(spec: LightweightAdapterSpec) -> bool:
    if not isinstance(spec, LightweightAdapterSpec): raise ValueError("INVALID_INPUT")
    if spec.schema_version != _SCHEMA_VERSION: raise ValueError("INVALID_SCHEMA_VERSION")
    if spec.spec_mode != _SPEC_MODE: raise ValueError("INVALID_SPEC_MODE")
    if spec.adapter_status not in _ALLOWED_ADAPTER_STATUS: raise ValueError("INVALID_ADAPTER_STATUS")
    if spec.adapter_kind not in _ALLOWED_ADAPTER_KIND: raise ValueError("INVALID_ADAPTER_KIND")
    if spec.optimization_scope in _SCOPE_TO_ADAPTER_KIND and spec.adapter_kind != _derive_adapter_kind(spec.optimization_scope): raise ValueError("ADAPTER_KIND_SCOPE_MISMATCH")
    _validate_hash_format(spec.source_optimization_contract_hash); _validate_hash_format(spec.source_opportunity_hash)
    for x in spec.boundaries: validate_adapter_boundary_spec(x)
    for x in spec.operations: validate_adapter_operation_spec(x)
    for x in spec.capabilities: validate_adapter_capability_spec(x)
    for x in spec.boundaries:
        if x.dependency_name != spec.dependency_name: raise ValueError("DEPENDENCY_NAME_MISMATCH")
    for x in spec.operations:
        if x.dependency_name != spec.dependency_name: raise ValueError("DEPENDENCY_NAME_MISMATCH")
    for x in spec.capabilities:
        if x.dependency_name != spec.dependency_name: raise ValueError("DEPENDENCY_NAME_MISMATCH")
    if tuple(x.boundary_index for x in spec.boundaries) != tuple(range(len(spec.boundaries))): raise ValueError("BOUNDARY_ORDER_MISMATCH")
    if tuple(x.operation_index for x in spec.operations) != tuple(range(len(spec.operations))): raise ValueError("OPERATION_ORDER_MISMATCH")
    if tuple(x.capability_index for x in spec.capabilities) != tuple(range(len(spec.capabilities))): raise ValueError("CAPABILITY_ORDER_MISMATCH")
    if (spec.boundary_count, spec.operation_count, spec.capability_count) != (len(spec.boundaries), len(spec.operations), len(spec.capabilities)): raise ValueError("ADAPTER_SPEC_COUNT_MISMATCH")
    if spec.first_boundary_hash != (spec.boundaries[0].boundary_hash if spec.boundaries else "") or spec.final_boundary_hash != (spec.boundaries[-1].boundary_hash if spec.boundaries else ""): raise ValueError("BOUNDARY_ORDER_MISMATCH")
    if spec.first_operation_hash != (spec.operations[0].operation_hash if spec.operations else "") or spec.final_operation_hash != (spec.operations[-1].operation_hash if spec.operations else ""): raise ValueError("OPERATION_ORDER_MISMATCH")
    if spec.first_capability_hash != (spec.capabilities[0].capability_hash if spec.capabilities else "") or spec.final_capability_hash != (spec.capabilities[-1].capability_hash if spec.capabilities else ""): raise ValueError("CAPABILITY_ORDER_MISMATCH")
    all_bh = {b.boundary_hash for b in spec.boundaries}
    for o in spec.operations:
        if any(h not in all_bh for h in (o.input_boundary_hashes + o.output_boundary_hashes)): raise ValueError("OPERATION_BOUNDARY_MISMATCH")
    txt = spec.to_canonical_json().lower()
    if "speedup" in txt or "benchmark" in txt: raise ValueError("SPEEDUP_CLAIM_FORBIDDEN")
    if "implementation complete" in txt or "adapter implemented" in txt or "fast path implemented" in txt: raise ValueError("IMPLEMENTATION_CLAIM_FORBIDDEN")
    exp = _hash_payload(_base_payload(spec, "lightweight_adapter_spec_hash"))
    _validate_hash_format(spec.lightweight_adapter_spec_hash)
    if spec.lightweight_adapter_spec_hash != exp: raise ValueError("HASH_MISMATCH")
    return True


def validate_adapter_spec_matches_contract(spec: LightweightAdapterSpec, contract: OptimizationContract) -> bool:
    if not isinstance(spec, LightweightAdapterSpec) or not isinstance(contract, OptimizationContract): raise ValueError("INVALID_INPUT")
    validate_lightweight_adapter_spec(spec)
    rebuilt = build_lightweight_adapter_spec_from_contract(contract, adapter_name=spec.adapter_name)
    if rebuilt.to_dict() != spec.to_dict(): raise ValueError("LIGHTWEIGHT_ADAPTER_SPEC_MISMATCH")
    return True
