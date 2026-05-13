from __future__ import annotations

from dataclasses import dataclass, fields
import re
from typing import Any

from .canonical_hashing import canonical_bytes, canonical_json, sha256_hex
from .material_encoding_receipt import (
    EncodingEntry,
    MaterialEncodingReceipt,
    SubstrateDriftReceipt,
    validate_encoding_entry,
    validate_material_encoding_receipt,
    validate_material_encoding_receipt_with_state,
    validate_substrate_drift_receipt,
    validate_substrate_drift_receipt_with_materials,
)
from .substrate_state_receipt import (
    PredicateEvaluationResult,
    SubstrateStateReceipt,
    validate_predicate_evaluation_result,
    validate_substrate_state_receipt,
)

_ERR_INVALID_INPUT = "INVALID_INPUT"
_ERR_INVALID_HASH_FORMAT = "INVALID_HASH_FORMAT"
_ERR_HASH_MISMATCH = "HASH_MISMATCH"
_ERR_INVALID_SUBSYSTEM_TYPE = "INVALID_SUBSYSTEM_TYPE"
_ERR_INVALID_SUBSYSTEM_LABEL = "INVALID_SUBSYSTEM_LABEL"
_ERR_INVALID_SUBSYSTEM_SUBSTRATE_CLASS = "INVALID_SUBSYSTEM_SUBSTRATE_CLASS"
_ERR_INVALID_SUBSYSTEM_SUBSTRATE_DRIFT_CLASS = "INVALID_SUBSYSTEM_SUBSTRATE_DRIFT_CLASS"
_ERR_SUBSYSTEM_CLASSIFICATION_AMBIGUOUS = "SUBSYSTEM_CLASSIFICATION_AMBIGUOUS"
_ERR_SUBSYSTEM_COUNT_MISMATCH = "SUBSYSTEM_COUNT_MISMATCH"
_ERR_SUBSYSTEM_ENTRY_MISMATCH = "SUBSYSTEM_ENTRY_MISMATCH"
_ERR_SUBSYSTEM_ENTRY_ORDER_MISMATCH = "SUBSYSTEM_ENTRY_ORDER_MISMATCH"
_ERR_DUPLICATE_SUBSYSTEM_ENTRY = "DUPLICATE_SUBSYSTEM_ENTRY"
_ERR_SUBSYSTEM_SUBSTRATE_CLASS_MISMATCH = "SUBSYSTEM_SUBSTRATE_CLASS_MISMATCH"
_ERR_SUBSYSTEM_SUBSTRATE_DRIFT_CLASS_MISMATCH = "SUBSYSTEM_SUBSTRATE_DRIFT_CLASS_MISMATCH"
_ERR_SUBSYSTEM_RECEIPT_MISMATCH = "SUBSYSTEM_RECEIPT_MISMATCH"

_MAX_SUBSYSTEM_ENCODING_ENTRIES = 1_000
_MAX_SUBSYSTEM_LABEL_LENGTH = 128
_MAX_DRIFT_HASHES = 1_000

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LABEL_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_ALLOWED_SUBSYSTEM_TYPES = frozenset({"LAYER", "MASK", "ROUTER", "READOUT"})
_ALLOWED_SUBSYSTEM_CLASSES = frozenset({"SUBSYSTEM_SUBSTRATE_COMPATIBLE", "SUBSYSTEM_SUBSTRATE_INCOMPATIBLE", "SUBSYSTEM_SUBSTRATE_EMPTY"})
_ALLOWED_SUBSYSTEM_DRIFT_CLASSES = frozenset({"SUBSYSTEM_SUBSTRATE_DRIFT_CLEAN", "SUBSYSTEM_SUBSTRATE_DRIFT_CHANGED", "SUBSYSTEM_SUBSTRATE_DRIFT_INCOMPLETE", "SUBSYSTEM_SUBSTRATE_DRIFT_UNEXPECTED", "SUBSYSTEM_SUBSTRATE_DRIFT_EMPTY"})
_ERR_DRIFT_HASH_NOT_IN_SUBSYSTEM = "DRIFT_HASH_NOT_IN_SUBSYSTEM"


def get_allowed_substrate_subsystem_types() -> frozenset[str]: return _ALLOWED_SUBSYSTEM_TYPES

def get_allowed_subsystem_substrate_classes() -> frozenset[str]: return _ALLOWED_SUBSYSTEM_CLASSES

def get_allowed_subsystem_substrate_drift_classes() -> frozenset[str]: return _ALLOWED_SUBSYSTEM_DRIFT_CLASSES

def _validate_sha(v: object) -> str:
    if not isinstance(v, str) or _SHA256_RE.fullmatch(v) is None: raise ValueError(_ERR_INVALID_HASH_FORMAT)
    return v

def _validate_int(v: object, err: str) -> int:
    if not isinstance(v, int) or isinstance(v, bool): raise ValueError(err)
    return v

def _subsystem_substrate_receipt_payload(**kwargs: Any) -> dict[str, Any]: return kwargs

def _derive_subsystem_class(count: int, failed: int) -> str:
    if count == 0: return "SUBSYSTEM_SUBSTRATE_EMPTY"
    if failed == 0: return "SUBSYSTEM_SUBSTRATE_COMPATIBLE"
    return "SUBSYSTEM_SUBSTRATE_INCOMPATIBLE"

def _derive_subsystem_drift_class(count: int, missing_count: int, unexpected_count: int, drift_count: int) -> str:
    if count == 0: return "SUBSYSTEM_SUBSTRATE_DRIFT_EMPTY"
    if missing_count > 0: return "SUBSYSTEM_SUBSTRATE_DRIFT_INCOMPLETE"
    if unexpected_count > 0: return "SUBSYSTEM_SUBSTRATE_DRIFT_UNEXPECTED"
    if drift_count > 0: return "SUBSYSTEM_SUBSTRATE_DRIFT_CHANGED"
    return "SUBSYSTEM_SUBSTRATE_DRIFT_CLEAN"

def _classify_encoding_entry_subsystem(encoding_entry: EncodingEntry, predicate_result: PredicateEvaluationResult, substrate_profile_id: str) -> str | None:
    validate_encoding_entry(encoding_entry)
    validate_predicate_evaluation_result(predicate_result)
    if not isinstance(substrate_profile_id, str): raise ValueError(_ERR_INVALID_INPUT)
    toks = list(predicate_result.field_path)
    toks.extend(predicate_result.predicate_id.split("_"))
    toks.extend(encoding_entry.encoding_label.split("_"))
    toks.extend([predicate_result.predicate_id, encoding_entry.encoding_label, substrate_profile_id])
    matches = set()
    # Exact-token matching: only match if the token exactly equals one of the allowed forms
    _LAYER_TOKENS = frozenset({"layer", "layers", "Layer", "Layers", "LAYER", "LAYERS", "Layered", "LayerToken"})
    _MASK_TOKENS = frozenset({"mask", "masks", "Mask", "Masks", "MASK", "MASKS", "SearchMask", "MaskCollision", "MaskToken"})
    _ROUTER_TOKENS = frozenset({"router", "routers", "Router", "Routers", "ROUTER", "ROUTERS", "RouterToken"})
    _READOUT_TOKENS = frozenset({"readout", "readouts", "Readout", "Readouts", "READOUT", "READOUTS", "ReadoutToken"})
    for t in toks:
        if not isinstance(t, str): continue
        if t in _LAYER_TOKENS: matches.add("LAYER")
        if t in _MASK_TOKENS: matches.add("MASK")
        if t in _ROUTER_TOKENS: matches.add("ROUTER")
        if t in _READOUT_TOKENS: matches.add("READOUT")
    if len(matches) > 1: raise ValueError(_ERR_SUBSYSTEM_CLASSIFICATION_AMBIGUOUS)
    return next(iter(matches)) if matches else None

def _select_entries_for_subsystem(substrate_state_receipt: SubstrateStateReceipt, material_encoding_receipt: MaterialEncodingReceipt, subsystem_type: str) -> tuple[tuple[EncodingEntry, PredicateEvaluationResult], ...]:
    if subsystem_type not in _ALLOWED_SUBSYSTEM_TYPES: raise ValueError(_ERR_INVALID_SUBSYSTEM_TYPE)
    validate_substrate_state_receipt(substrate_state_receipt)
    validate_material_encoding_receipt_with_state(material_encoding_receipt, substrate_state_receipt)
    by_hash = {r.predicate_evaluation_result_hash: r for r in substrate_state_receipt.predicate_evaluation_results}
    out: list[tuple[EncodingEntry, PredicateEvaluationResult]] = []
    for e in material_encoding_receipt.encoding_entries:
        r = by_hash.get(e.predicate_evaluation_result_hash)
        if r is None: raise ValueError(_ERR_SUBSYSTEM_ENTRY_MISMATCH)
        cls = _classify_encoding_entry_subsystem(e, r, material_encoding_receipt.substrate_profile_id)
        if cls == subsystem_type: out.append((e, r))
    return tuple(out)

def _build_common(ssr: SubstrateStateReceipt, mer: MaterialEncodingReceipt, sdr: SubstrateDriftReceipt, subsystem_type: str, subsystem_label: str, observed_mer: MaterialEncodingReceipt | None = None) -> dict[str, Any]:
    validate_substrate_state_receipt(ssr)
    validate_material_encoding_receipt_with_state(mer, ssr)
    validate_substrate_drift_receipt(sdr)
    if sdr.expected_material_encoding_receipt_hash != mer.material_encoding_receipt_hash: raise ValueError(_ERR_SUBSYSTEM_ENTRY_MISMATCH)
    if sdr.observed_material_encoding_receipt_hash is not None:
        # Use provided observed_mer if available, otherwise fall back to mer (clean case)
        obs = observed_mer if observed_mer is not None else mer
        validate_substrate_drift_receipt_with_materials(sdr, mer, obs)
    if mer.substrate_contract_hash != ssr.substrate_contract_hash or sdr.substrate_contract_hash != ssr.substrate_contract_hash: raise ValueError(_ERR_SUBSYSTEM_ENTRY_MISMATCH)
    if mer.substrate_profile_id != ssr.substrate_profile_id or sdr.substrate_profile_id != ssr.substrate_profile_id: raise ValueError(_ERR_SUBSYSTEM_ENTRY_MISMATCH)
    sel = _select_entries_for_subsystem(ssr, mer, subsystem_type)
    prh = tuple(r.predicate_evaluation_result_hash for e, r in sel)
    ehs = tuple(e.encoding_entry_hash for e, r in sel)
    passed = sum(1 for e, _ in sel if e.encoded_status == "ENCODED_PASS")
    failed = len(sel) - passed
    set_ehs = set(ehs)
    drifted = tuple(h for h in sdr.drifted_encoding_entry_hashes if h in set_ehs)
    missing = tuple(h for h in sdr.missing_encoding_entry_hashes if h in set_ehs)
    unexpected = tuple(h for h in sdr.unexpected_encoding_entry_hashes if h in set_ehs)
    drift_count = len(drifted) + len(missing) + len(unexpected)
    return _subsystem_substrate_receipt_payload(
        substrate_state_receipt_hash=ssr.substrate_state_receipt_hash,
        material_encoding_receipt_hash=mer.material_encoding_receipt_hash,
        substrate_drift_receipt_hash=sdr.substrate_drift_receipt_hash,
        substrate_contract_hash=ssr.substrate_contract_hash,
        substrate_profile_id=ssr.substrate_profile_id,
        subsystem_type=subsystem_type,
        subsystem_label=subsystem_label,
        predicate_evaluation_result_hashes=prh,
        encoding_entry_hashes=ehs,
        subsystem_entry_count=len(sel),
        passed_encoding_count=passed,
        failed_encoding_count=failed,
        drifted_encoding_entry_hashes=drifted,
        missing_encoding_entry_hashes=missing,
        unexpected_encoding_entry_hashes=unexpected,
        drift_count=drift_count,
        subsystem_substrate_class=_derive_subsystem_class(len(sel), failed),
        subsystem_substrate_drift_class=_derive_subsystem_drift_class(len(sel), len(missing), len(unexpected), drift_count),
    )

def _validate_core(receipt: object, etype: str, elabel: str, hfield: str) -> bool:
    try:
        # Use isinstance check against base class instead of string-based type checking
        if not isinstance(receipt, LayerSubstrateCompatibilityReceipt): raise ValueError(_ERR_INVALID_INPUT)
        _validate_sha(getattr(receipt, hfield))
        payload = {f.name: getattr(receipt, f.name) for f in fields(receipt) if f.name not in {hfield, "mask_substrate_receipt_hash", "router_substrate_receipt_hash", "readout_substrate_receipt_hash"}}
        for k in ("substrate_state_receipt_hash","material_encoding_receipt_hash","substrate_drift_receipt_hash","substrate_contract_hash"): _validate_sha(payload[k])
        if payload["subsystem_type"] != etype: raise ValueError(_ERR_INVALID_SUBSYSTEM_TYPE)
        if not isinstance(payload["subsystem_label"], str) or not payload["subsystem_label"] or len(payload["subsystem_label"]) > _MAX_SUBSYSTEM_LABEL_LENGTH or _LABEL_RE.fullmatch(payload["subsystem_label"]) is None or payload["subsystem_label"] != elabel: raise ValueError(_ERR_INVALID_SUBSYSTEM_LABEL)
        for k, mx in (("predicate_evaluation_result_hashes", _MAX_SUBSYSTEM_ENCODING_ENTRIES), ("encoding_entry_hashes", _MAX_SUBSYSTEM_ENCODING_ENTRIES), ("drifted_encoding_entry_hashes", _MAX_DRIFT_HASHES), ("missing_encoding_entry_hashes", _MAX_DRIFT_HASHES), ("unexpected_encoding_entry_hashes", _MAX_DRIFT_HASHES)):
            v = payload[k]
            if not isinstance(v, tuple) or len(v) > mx: raise ValueError(_ERR_INVALID_INPUT)
            seen = set()
            for h in v:
                _validate_sha(h)
                if h in seen: raise ValueError(_ERR_DUPLICATE_SUBSYSTEM_ENTRY)
                seen.add(h)
        c = _validate_int(payload["subsystem_entry_count"], _ERR_SUBSYSTEM_COUNT_MISMATCH); p = _validate_int(payload["passed_encoding_count"], _ERR_SUBSYSTEM_COUNT_MISMATCH); f = _validate_int(payload["failed_encoding_count"], _ERR_SUBSYSTEM_COUNT_MISMATCH); d = _validate_int(payload["drift_count"], _ERR_SUBSYSTEM_COUNT_MISMATCH)
        if c != len(payload["encoding_entry_hashes"]) or c != len(payload["predicate_evaluation_result_hashes"]) or p + f != c: raise ValueError(_ERR_SUBSYSTEM_COUNT_MISMATCH)
        if d != len(payload["drifted_encoding_entry_hashes"]) + len(payload["missing_encoding_entry_hashes"]) + len(payload["unexpected_encoding_entry_hashes"]): raise ValueError(_ERR_SUBSYSTEM_COUNT_MISMATCH)
        # P2 fix: Verify drift hashes are members of encoding_entry_hashes
        encoding_entry_set = set(payload["encoding_entry_hashes"])
        for drift_key in ("drifted_encoding_entry_hashes", "missing_encoding_entry_hashes", "unexpected_encoding_entry_hashes"):
            for h in payload[drift_key]:
                if h not in encoding_entry_set: raise ValueError(_ERR_DRIFT_HASH_NOT_IN_SUBSYSTEM)
        if payload["subsystem_substrate_class"] not in _ALLOWED_SUBSYSTEM_CLASSES: raise ValueError(_ERR_INVALID_SUBSYSTEM_SUBSTRATE_CLASS)
        if payload["subsystem_substrate_drift_class"] not in _ALLOWED_SUBSYSTEM_DRIFT_CLASSES: raise ValueError(_ERR_INVALID_SUBSYSTEM_SUBSTRATE_DRIFT_CLASS)
        if payload["subsystem_substrate_class"] != _derive_subsystem_class(c, f): raise ValueError(_ERR_SUBSYSTEM_SUBSTRATE_CLASS_MISMATCH)
        if payload["subsystem_substrate_drift_class"] != _derive_subsystem_drift_class(c, len(payload["missing_encoding_entry_hashes"]), len(payload["unexpected_encoding_entry_hashes"]), d): raise ValueError(_ERR_SUBSYSTEM_SUBSTRATE_DRIFT_CLASS_MISMATCH)
        if sha256_hex(payload) != getattr(receipt, hfield): raise ValueError(_ERR_HASH_MISMATCH)
        return True
    except (TypeError, AttributeError) as e:
        raise ValueError(_ERR_INVALID_INPUT) from e

@dataclass(frozen=True)
class LayerSubstrateCompatibilityReceipt:
    substrate_state_receipt_hash: str; material_encoding_receipt_hash: str; substrate_drift_receipt_hash: str; substrate_contract_hash: str; substrate_profile_id: str
    subsystem_type: str; subsystem_label: str; predicate_evaluation_result_hashes: tuple[str, ...]; encoding_entry_hashes: tuple[str, ...]
    subsystem_entry_count: int; passed_encoding_count: int; failed_encoding_count: int
    drifted_encoding_entry_hashes: tuple[str, ...]; missing_encoding_entry_hashes: tuple[str, ...]; unexpected_encoding_entry_hashes: tuple[str, ...]
    drift_count: int; subsystem_substrate_class: str; subsystem_substrate_drift_class: str; layer_substrate_compatibility_receipt_hash: str
    def __post_init__(self) -> None: validate_layer_substrate_compatibility_receipt(self)
    def to_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        for k in ("predicate_evaluation_result_hashes","encoding_entry_hashes","drifted_encoding_entry_hashes","missing_encoding_entry_hashes","unexpected_encoding_entry_hashes"): d[k] = list(d[k])
        return d
    def to_canonical_json(self) -> str: return canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return canonical_bytes(self.to_dict())

@dataclass(frozen=True)
class MaskSubstrateReceipt(LayerSubstrateCompatibilityReceipt):
    mask_substrate_receipt_hash: str
    def __post_init__(self) -> None: validate_mask_substrate_receipt(self)
    def to_dict(self) -> dict[str, Any]: return {**super().to_dict(), "mask_substrate_receipt_hash": self.mask_substrate_receipt_hash}

@dataclass(frozen=True)
class RouterSubstrateReceipt(LayerSubstrateCompatibilityReceipt):
    router_substrate_receipt_hash: str
    def __post_init__(self) -> None: validate_router_substrate_receipt(self)
    def to_dict(self) -> dict[str, Any]: return {**super().to_dict(), "router_substrate_receipt_hash": self.router_substrate_receipt_hash}

@dataclass(frozen=True)
class ReadoutSubstrateReceipt(LayerSubstrateCompatibilityReceipt):
    readout_substrate_receipt_hash: str
    def __post_init__(self) -> None: validate_readout_substrate_receipt(self)
    def to_dict(self) -> dict[str, Any]: return {**super().to_dict(), "readout_substrate_receipt_hash": self.readout_substrate_receipt_hash}

def build_layer_substrate_compatibility_receipt(substrate_state_receipt: SubstrateStateReceipt, material_encoding_receipt: MaterialEncodingReceipt, substrate_drift_receipt: SubstrateDriftReceipt, observed_material_encoding_receipt: MaterialEncodingReceipt | None = None) -> LayerSubstrateCompatibilityReceipt:
    p = _build_common(substrate_state_receipt, material_encoding_receipt, substrate_drift_receipt, "LAYER", "LAYER_SUBSTRATE_COMPATIBILITY", observed_material_encoding_receipt)
    return LayerSubstrateCompatibilityReceipt(**p, layer_substrate_compatibility_receipt_hash=sha256_hex(p))

def build_mask_substrate_receipt(substrate_state_receipt: SubstrateStateReceipt, material_encoding_receipt: MaterialEncodingReceipt, substrate_drift_receipt: SubstrateDriftReceipt, observed_material_encoding_receipt: MaterialEncodingReceipt | None = None) -> MaskSubstrateReceipt:
    p = _build_common(substrate_state_receipt, material_encoding_receipt, substrate_drift_receipt, "MASK", "MASK_SUBSTRATE", observed_material_encoding_receipt)
    return MaskSubstrateReceipt(**p, layer_substrate_compatibility_receipt_hash=sha256_hex(p), mask_substrate_receipt_hash=sha256_hex({**p, "layer_substrate_compatibility_receipt_hash": sha256_hex(p)}))

def build_router_substrate_receipt(substrate_state_receipt: SubstrateStateReceipt, material_encoding_receipt: MaterialEncodingReceipt, substrate_drift_receipt: SubstrateDriftReceipt, observed_material_encoding_receipt: MaterialEncodingReceipt | None = None) -> RouterSubstrateReceipt:
    p = _build_common(substrate_state_receipt, material_encoding_receipt, substrate_drift_receipt, "ROUTER", "ROUTER_SUBSTRATE", observed_material_encoding_receipt)
    return RouterSubstrateReceipt(**p, layer_substrate_compatibility_receipt_hash=sha256_hex(p), router_substrate_receipt_hash=sha256_hex({**p, "layer_substrate_compatibility_receipt_hash": sha256_hex(p)}))

def build_readout_substrate_receipt(substrate_state_receipt: SubstrateStateReceipt, material_encoding_receipt: MaterialEncodingReceipt, substrate_drift_receipt: SubstrateDriftReceipt, observed_material_encoding_receipt: MaterialEncodingReceipt | None = None) -> ReadoutSubstrateReceipt:
    p = _build_common(substrate_state_receipt, material_encoding_receipt, substrate_drift_receipt, "READOUT", "READOUT_SUBSTRATE", observed_material_encoding_receipt)
    return ReadoutSubstrateReceipt(**p, layer_substrate_compatibility_receipt_hash=sha256_hex(p), readout_substrate_receipt_hash=sha256_hex({**p, "layer_substrate_compatibility_receipt_hash": sha256_hex(p)}))

def validate_layer_substrate_compatibility_receipt(receipt: LayerSubstrateCompatibilityReceipt) -> bool: return _validate_core(receipt, "LAYER", "LAYER_SUBSTRATE_COMPATIBILITY", "layer_substrate_compatibility_receipt_hash")
def validate_mask_substrate_receipt(receipt: MaskSubstrateReceipt) -> bool:
    if not _validate_core(receipt, "MASK", "MASK_SUBSTRATE", "layer_substrate_compatibility_receipt_hash"): return False
    if sha256_hex({f.name: getattr(receipt, f.name) for f in fields(receipt) if f.name != "mask_substrate_receipt_hash"}) != receipt.mask_substrate_receipt_hash: raise ValueError(_ERR_HASH_MISMATCH)
    return True

def validate_router_substrate_receipt(receipt: RouterSubstrateReceipt) -> bool:
    if not _validate_core(receipt, "ROUTER", "ROUTER_SUBSTRATE", "layer_substrate_compatibility_receipt_hash"): return False
    if sha256_hex({f.name: getattr(receipt, f.name) for f in fields(receipt) if f.name != "router_substrate_receipt_hash"}) != receipt.router_substrate_receipt_hash: raise ValueError(_ERR_HASH_MISMATCH)
    return True

def validate_readout_substrate_receipt(receipt: ReadoutSubstrateReceipt) -> bool:
    if not _validate_core(receipt, "READOUT", "READOUT_SUBSTRATE", "layer_substrate_compatibility_receipt_hash"): return False
    if sha256_hex({f.name: getattr(receipt, f.name) for f in fields(receipt) if f.name != "readout_substrate_receipt_hash"}) != receipt.readout_substrate_receipt_hash: raise ValueError(_ERR_HASH_MISMATCH)
    return True

def _complete_validate(receipt: Any, ssr: SubstrateStateReceipt, mer: MaterialEncodingReceipt, sdr: SubstrateDriftReceipt, build_fn: Any, validate_fn: Any, hash_field: str, observed_mer: MaterialEncodingReceipt | None = None) -> bool:
    validate_fn(receipt)
    exp = build_fn(ssr, mer, sdr, observed_mer)
    if receipt.to_dict() != exp.to_dict():
        # Use explicit hash field name instead of relying on field order
        if getattr(receipt, hash_field) != getattr(exp, hash_field): raise ValueError(_ERR_HASH_MISMATCH)
        raise ValueError(_ERR_SUBSYSTEM_RECEIPT_MISMATCH)
    return True

def validate_layer_substrate_compatibility_receipt_with_artifacts(receipt: LayerSubstrateCompatibilityReceipt, substrate_state_receipt: SubstrateStateReceipt, material_encoding_receipt: MaterialEncodingReceipt, substrate_drift_receipt: SubstrateDriftReceipt, observed_material_encoding_receipt: MaterialEncodingReceipt | None = None) -> bool:
    return _complete_validate(receipt, substrate_state_receipt, material_encoding_receipt, substrate_drift_receipt, build_layer_substrate_compatibility_receipt, validate_layer_substrate_compatibility_receipt, "layer_substrate_compatibility_receipt_hash", observed_material_encoding_receipt)

def validate_mask_substrate_receipt_with_artifacts(receipt: MaskSubstrateReceipt, substrate_state_receipt: SubstrateStateReceipt, material_encoding_receipt: MaterialEncodingReceipt, substrate_drift_receipt: SubstrateDriftReceipt, observed_material_encoding_receipt: MaterialEncodingReceipt | None = None) -> bool:
    return _complete_validate(receipt, substrate_state_receipt, material_encoding_receipt, substrate_drift_receipt, build_mask_substrate_receipt, validate_mask_substrate_receipt, "mask_substrate_receipt_hash", observed_material_encoding_receipt)

def validate_router_substrate_receipt_with_artifacts(receipt: RouterSubstrateReceipt, substrate_state_receipt: SubstrateStateReceipt, material_encoding_receipt: MaterialEncodingReceipt, substrate_drift_receipt: SubstrateDriftReceipt, observed_material_encoding_receipt: MaterialEncodingReceipt | None = None) -> bool:
    return _complete_validate(receipt, substrate_state_receipt, material_encoding_receipt, substrate_drift_receipt, build_router_substrate_receipt, validate_router_substrate_receipt, "router_substrate_receipt_hash", observed_material_encoding_receipt)

def validate_readout_substrate_receipt_with_artifacts(receipt: ReadoutSubstrateReceipt, substrate_state_receipt: SubstrateStateReceipt, material_encoding_receipt: MaterialEncodingReceipt, substrate_drift_receipt: SubstrateDriftReceipt, observed_material_encoding_receipt: MaterialEncodingReceipt | None = None) -> bool:
    return _complete_validate(receipt, substrate_state_receipt, material_encoding_receipt, substrate_drift_receipt, build_readout_substrate_receipt, validate_readout_substrate_receipt, "readout_substrate_receipt_hash", observed_material_encoding_receipt)
