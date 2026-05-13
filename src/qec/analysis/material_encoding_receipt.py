from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .canonical_hashing import canonical_bytes, canonical_json, sha256_hex
from .substrate_constraint_contract import SubstrateContract, validate_substrate_contract
from .substrate_state_receipt import (
    PredicateEvaluationResult,
    SubstrateStateReceipt,
    validate_predicate_evaluation_result,
    validate_substrate_state_receipt,
    validate_substrate_state_receipt_with_contract,
)

_ERR_INVALID_INPUT = "INVALID_INPUT"
_ERR_INVALID_HASH_FORMAT = "INVALID_HASH_FORMAT"
_ERR_HASH_MISMATCH = "HASH_MISMATCH"
_ERR_INVALID_ENCODING_MODE = "INVALID_ENCODING_MODE"
_ERR_INVALID_ENCODING_LABEL = "INVALID_ENCODING_LABEL"
_ERR_INVALID_ENCODED_STATUS = "INVALID_ENCODED_STATUS"
_ERR_ENCODING_INDEX_OUT_OF_BOUNDS = "ENCODING_INDEX_OUT_OF_BOUNDS"
_ERR_ENCODING_COUNT_MISMATCH = "ENCODING_COUNT_MISMATCH"
_ERR_ENCODING_ORDER_MISMATCH = "ENCODING_ORDER_MISMATCH"
_ERR_DUPLICATE_ENCODING_ENTRY = "DUPLICATE_ENCODING_ENTRY"
_ERR_ENCODING_ENTRY_MISMATCH = "ENCODING_ENTRY_MISMATCH"
_ERR_MATERIAL_ENCODING_CLASS_MISMATCH = "MATERIAL_ENCODING_CLASS_MISMATCH"
_ERR_MATERIAL_ENCODING_RECEIPT_MISMATCH = "MATERIAL_ENCODING_RECEIPT_MISMATCH"
_ERR_INVALID_SUBSTRATE_DRIFT_CLASS = "INVALID_SUBSTRATE_DRIFT_CLASS"
_ERR_SUBSTRATE_DRIFT_CLASS_MISMATCH = "SUBSTRATE_DRIFT_CLASS_MISMATCH"
_ERR_SUBSTRATE_DRIFT_COUNT_MISMATCH = "SUBSTRATE_DRIFT_COUNT_MISMATCH"
_ERR_SUBSTRATE_DRIFT_RECEIPT_MISMATCH = "SUBSTRATE_DRIFT_RECEIPT_MISMATCH"

_MAX_ENCODING_ENTRIES = 1_000
_MAX_ENCODING_INDEX = 999
_MAX_ENCODING_LABEL_LENGTH = 96
_MAX_DRIFT_ENTRY_HASHES = 1_000

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LABEL_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_ALLOWED_ENCODED_STATUSES = frozenset({"ENCODED_PASS", "ENCODED_FAIL"})
_ALLOWED_MATERIAL_ENCODING_CLASSES = frozenset({"MATERIAL_ENCODING_COMPATIBLE", "MATERIAL_ENCODING_INCOMPATIBLE"})
_ALLOWED_SUBSTRATE_DRIFT_CLASSES = frozenset({"SUBSTRATE_DRIFT_CLEAN", "SUBSTRATE_DRIFT_CHANGED", "SUBSTRATE_DRIFT_INCOMPLETE", "SUBSTRATE_DRIFT_UNEXPECTED"})
_ENCODING_MODE = "DETERMINISTIC_SUBSTRATE_MATERIAL_ENCODING"


def get_allowed_encoded_statuses() -> frozenset[str]: return _ALLOWED_ENCODED_STATUSES

def get_allowed_material_encoding_classes() -> frozenset[str]: return _ALLOWED_MATERIAL_ENCODING_CLASSES

def get_allowed_substrate_drift_classes() -> frozenset[str]: return _ALLOWED_SUBSTRATE_DRIFT_CLASSES

def _validate_sha(v: object) -> str:
    if not isinstance(v, str) or _SHA256_RE.fullmatch(v) is None: raise ValueError(_ERR_INVALID_HASH_FORMAT)
    return v

def _encoding_entry_payload(**kwargs: Any) -> dict[str, Any]: return kwargs

def _material_encoding_receipt_payload(**kwargs: Any) -> dict[str, Any]:
    p = dict(kwargs); p["encoding_entries"] = [e.to_dict() for e in kwargs["encoding_entries"]]; return p

def _substrate_drift_receipt_payload(**kwargs: Any) -> dict[str, Any]: return kwargs

def _encoded_value_hash_from_result(result: PredicateEvaluationResult) -> str:
    return sha256_hex({"predicate_evaluation_result_hash": result.predicate_evaluation_result_hash, "predicate_id": result.predicate_id, "predicate_kind": result.predicate_kind, "observed_json_type": result.observed_json_type, "observed_value_hash": result.observed_value_hash, "passed": result.passed, "evaluation_status": result.evaluation_status})

@dataclass(frozen=True)
class EncodingEntry:
    substrate_state_receipt_hash: str; substrate_contract_hash: str; substrate_profile_id: str
    predicate_evaluation_result_hash: str; predicate_id: str; predicate_kind: str
    encoding_index: int; encoding_label: str; encoded_status: str; encoded_value_hash: str; encoding_entry_hash: str
    def __post_init__(self) -> None: validate_encoding_entry(self)
    def to_dict(self) -> dict[str, Any]: return {k: getattr(self, k) for k in self.__dataclass_fields__}
    def to_canonical_json(self) -> str: return canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return canonical_bytes(self.to_dict())

@dataclass(frozen=True)
class MaterialEncodingReceipt:
    substrate_state_receipt_hash: str; substrate_contract_hash: str; substrate_profile_id: str; encoding_mode: str
    encoding_entries: tuple[EncodingEntry, ...]; encoding_entry_count: int; passed_encoding_count: int; failed_encoding_count: int
    material_encoding_class: str; material_encoding_receipt_hash: str
    def __post_init__(self) -> None: validate_material_encoding_receipt(self)
    def to_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}; d["encoding_entries"] = [e.to_dict() for e in self.encoding_entries]; return d
    def to_canonical_json(self) -> str: return canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return canonical_bytes(self.to_dict())

@dataclass(frozen=True)
class SubstrateDriftReceipt:
    expected_material_encoding_receipt_hash: str; observed_material_encoding_receipt_hash: str | None; substrate_contract_hash: str; substrate_profile_id: str
    expected_encoding_entry_hashes: tuple[str, ...]; observed_encoding_entry_hashes: tuple[str, ...]; drifted_encoding_entry_hashes: tuple[str, ...]
    missing_encoding_entry_hashes: tuple[str, ...]; unexpected_encoding_entry_hashes: tuple[str, ...]; encoding_entry_count: int; drift_count: int
    substrate_drift_class: str; substrate_drift_receipt_hash: str
    def __post_init__(self) -> None: validate_substrate_drift_receipt(self)
    def to_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        for k in ("expected_encoding_entry_hashes","observed_encoding_entry_hashes","drifted_encoding_entry_hashes","missing_encoding_entry_hashes","unexpected_encoding_entry_hashes"): d[k] = list(d[k])
        return d
    def to_canonical_json(self) -> str: return canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return canonical_bytes(self.to_dict())

def _class_from_failed(failed: int) -> str: return "MATERIAL_ENCODING_COMPATIBLE" if failed == 0 else "MATERIAL_ENCODING_INCOMPATIBLE"
def _status_from_passed(p: bool) -> str: return "ENCODED_PASS" if p else "ENCODED_FAIL"
def _drift_class(obs_hash: str | None, missing: tuple[str,...], unexpected: tuple[str,...], drift_count: int) -> str:
    if obs_hash is None or len(missing) > 0: return "SUBSTRATE_DRIFT_INCOMPLETE"
    if len(unexpected) > 0: return "SUBSTRATE_DRIFT_UNEXPECTED"
    if drift_count > 0: return "SUBSTRATE_DRIFT_CHANGED"
    return "SUBSTRATE_DRIFT_CLEAN"

def build_encoding_entry(substrate_state_receipt: SubstrateStateReceipt, predicate_evaluation_result: PredicateEvaluationResult, encoding_index: int) -> EncodingEntry:
    validate_substrate_state_receipt(substrate_state_receipt); validate_predicate_evaluation_result(predicate_evaluation_result)
    if not isinstance(encoding_index, int) or isinstance(encoding_index, bool) or not (0 <= encoding_index <= _MAX_ENCODING_INDEX): raise ValueError(_ERR_ENCODING_INDEX_OUT_OF_BOUNDS)
    matches = [r for r in substrate_state_receipt.predicate_evaluation_results if r.predicate_evaluation_result_hash == predicate_evaluation_result.predicate_evaluation_result_hash]
    if len(matches) != 1: raise ValueError(_ERR_INVALID_INPUT)
    label = f"ENCODING_{encoding_index:06d}_{predicate_evaluation_result.predicate_id}"
    status = _status_from_passed(predicate_evaluation_result.passed); evh = _encoded_value_hash_from_result(predicate_evaluation_result)
    p = _encoding_entry_payload(substrate_state_receipt_hash=substrate_state_receipt.substrate_state_receipt_hash, substrate_contract_hash=substrate_state_receipt.substrate_contract_hash, substrate_profile_id=substrate_state_receipt.substrate_profile_id, predicate_evaluation_result_hash=predicate_evaluation_result.predicate_evaluation_result_hash, predicate_id=predicate_evaluation_result.predicate_id, predicate_kind=predicate_evaluation_result.predicate_kind, encoding_index=encoding_index, encoding_label=label, encoded_status=status, encoded_value_hash=evh)
    return EncodingEntry(**p, encoding_entry_hash=sha256_hex(p))

def validate_encoding_entry(entry: EncodingEntry) -> bool:
    try:
        if not isinstance(entry, EncodingEntry): raise ValueError(_ERR_INVALID_INPUT)
        for h in (entry.substrate_state_receipt_hash,entry.substrate_contract_hash,entry.predicate_evaluation_result_hash,entry.encoded_value_hash,entry.encoding_entry_hash): _validate_sha(h)
        if not isinstance(entry.encoding_index, int) or isinstance(entry.encoding_index, bool) or not (0 <= entry.encoding_index <= _MAX_ENCODING_INDEX): raise ValueError(_ERR_ENCODING_INDEX_OUT_OF_BOUNDS)
        if not isinstance(entry.encoding_label, str) or not entry.encoding_label or len(entry.encoding_label) > _MAX_ENCODING_LABEL_LENGTH or _LABEL_RE.fullmatch(entry.encoding_label) is None: raise ValueError(_ERR_INVALID_ENCODING_LABEL)
        if entry.encoded_status not in _ALLOWED_ENCODED_STATUSES: raise ValueError(_ERR_INVALID_ENCODED_STATUS)
        p = _encoding_entry_payload(substrate_state_receipt_hash=entry.substrate_state_receipt_hash, substrate_contract_hash=entry.substrate_contract_hash, substrate_profile_id=entry.substrate_profile_id, predicate_evaluation_result_hash=entry.predicate_evaluation_result_hash, predicate_id=entry.predicate_id, predicate_kind=entry.predicate_kind, encoding_index=entry.encoding_index, encoding_label=entry.encoding_label, encoded_status=entry.encoded_status, encoded_value_hash=entry.encoded_value_hash)
        if sha256_hex(p) != entry.encoding_entry_hash: raise ValueError(_ERR_HASH_MISMATCH)
        return True
    except (TypeError, AttributeError) as e: raise ValueError(_ERR_INVALID_INPUT) from e

def build_material_encoding_receipt(substrate_state_receipt: SubstrateStateReceipt) -> MaterialEncodingReceipt:
    validate_substrate_state_receipt(substrate_state_receipt)
    entries = tuple(build_encoding_entry(substrate_state_receipt, r, i) for i, r in enumerate(substrate_state_receipt.predicate_evaluation_results))
    if len(entries) == 0 or len(entries) > _MAX_ENCODING_ENTRIES: raise ValueError(_ERR_ENCODING_COUNT_MISMATCH)
    passed = sum(1 for e in entries if e.encoded_status == "ENCODED_PASS"); failed = len(entries)-passed; mclass = _class_from_failed(failed)
    p = _material_encoding_receipt_payload(substrate_state_receipt_hash=substrate_state_receipt.substrate_state_receipt_hash, substrate_contract_hash=substrate_state_receipt.substrate_contract_hash, substrate_profile_id=substrate_state_receipt.substrate_profile_id, encoding_mode=_ENCODING_MODE, encoding_entries=entries, encoding_entry_count=len(entries), passed_encoding_count=passed, failed_encoding_count=failed, material_encoding_class=mclass)
    return MaterialEncodingReceipt(**{k:v for k,v in p.items() if k!="encoding_entries"}, encoding_entries=entries, material_encoding_receipt_hash=sha256_hex(p))

def validate_material_encoding_receipt(receipt: MaterialEncodingReceipt) -> bool:
    try:
        if not isinstance(receipt, MaterialEncodingReceipt): raise ValueError(_ERR_INVALID_INPUT)
        for h in (receipt.substrate_state_receipt_hash, receipt.substrate_contract_hash, receipt.material_encoding_receipt_hash): _validate_sha(h)
        if receipt.encoding_mode != _ENCODING_MODE: raise ValueError(_ERR_INVALID_ENCODING_MODE)
        if not isinstance(receipt.encoding_entries, tuple) or not (1 <= len(receipt.encoding_entries) <= _MAX_ENCODING_ENTRIES): raise ValueError(_ERR_ENCODING_COUNT_MISMATCH)
        ids_i=set(); ids_h=set(); ids_p=set(); sort_key=[]
        for e in receipt.encoding_entries:
            validate_encoding_entry(e)
            if e.encoding_index in ids_i or e.predicate_evaluation_result_hash in ids_h or e.predicate_id in ids_p: raise ValueError(_ERR_DUPLICATE_ENCODING_ENTRY)
            ids_i.add(e.encoding_index); ids_h.add(e.predicate_evaluation_result_hash); ids_p.add(e.predicate_id); sort_key.append((e.encoding_index,e.predicate_id,e.predicate_kind,e.encoding_entry_hash))
        if tuple(sort_key) != tuple(sorted(sort_key)): raise ValueError(_ERR_ENCODING_ORDER_MISMATCH)
        for c in (receipt.encoding_entry_count, receipt.passed_encoding_count, receipt.failed_encoding_count):
            if not isinstance(c, int) or isinstance(c, bool): raise ValueError(_ERR_INVALID_INPUT)
        if receipt.encoding_entry_count != len(receipt.encoding_entries): raise ValueError(_ERR_ENCODING_COUNT_MISMATCH)
        passed = sum(1 for e in receipt.encoding_entries if e.encoded_status=="ENCODED_PASS"); failed = len(receipt.encoding_entries)-passed
        if receipt.passed_encoding_count != passed or receipt.failed_encoding_count != failed: raise ValueError(_ERR_ENCODING_COUNT_MISMATCH)
        if receipt.material_encoding_class not in _ALLOWED_MATERIAL_ENCODING_CLASSES: raise ValueError(_ERR_MATERIAL_ENCODING_CLASS_MISMATCH)
        if receipt.material_encoding_class != _class_from_failed(failed): raise ValueError(_ERR_MATERIAL_ENCODING_CLASS_MISMATCH)
        p = _material_encoding_receipt_payload(substrate_state_receipt_hash=receipt.substrate_state_receipt_hash, substrate_contract_hash=receipt.substrate_contract_hash, substrate_profile_id=receipt.substrate_profile_id, encoding_mode=receipt.encoding_mode, encoding_entries=receipt.encoding_entries, encoding_entry_count=receipt.encoding_entry_count, passed_encoding_count=receipt.passed_encoding_count, failed_encoding_count=receipt.failed_encoding_count, material_encoding_class=receipt.material_encoding_class)
        if sha256_hex(p) != receipt.material_encoding_receipt_hash: raise ValueError(_ERR_HASH_MISMATCH)
        return True
    except (TypeError, AttributeError) as e: raise ValueError(_ERR_INVALID_INPUT) from e

def build_substrate_drift_receipt(expected_material_encoding_receipt: MaterialEncodingReceipt, observed_material_encoding_receipt: MaterialEncodingReceipt | None = None) -> SubstrateDriftReceipt:
    validate_material_encoding_receipt(expected_material_encoding_receipt)
    if observed_material_encoding_receipt is not None:
        validate_material_encoding_receipt(observed_material_encoding_receipt)
        if observed_material_encoding_receipt.substrate_contract_hash != expected_material_encoding_receipt.substrate_contract_hash or observed_material_encoding_receipt.substrate_profile_id != expected_material_encoding_receipt.substrate_profile_id:
            raise ValueError(_ERR_MATERIAL_ENCODING_RECEIPT_MISMATCH)
    exp = expected_material_encoding_receipt.encoding_entries; obs = tuple() if observed_material_encoding_receipt is None else observed_material_encoding_receipt.encoding_entries
    em, om = {e.predicate_id:e for e in exp}, {e.predicate_id:e for e in obs}
    if len(em)!=len(exp) or len(om)!=len(obs): raise ValueError(_ERR_DUPLICATE_ENCODING_ENTRY)
    expected_hashes = tuple(e.encoding_entry_hash for e in exp); observed_hashes = tuple(e.encoding_entry_hash for e in obs)
    drifted=[]; missing=[]; unexpected=[]
    for e in exp:
        if e.predicate_id not in om: missing.append(e.encoding_entry_hash)
        elif om[e.predicate_id].encoding_entry_hash != e.encoding_entry_hash: drifted.append(e.encoding_entry_hash)
    for o in obs:
        if o.predicate_id not in em: unexpected.append(o.encoding_entry_hash)
    drifted_t, missing_t, unexpected_t = tuple(drifted), tuple(missing), tuple(unexpected)
    drift_count = len(drifted_t)+len(missing_t)+len(unexpected_t)
    obs_hash = None if observed_material_encoding_receipt is None else observed_material_encoding_receipt.material_encoding_receipt_hash
    dclass = _drift_class(obs_hash, missing_t, unexpected_t, drift_count)
    p = _substrate_drift_receipt_payload(expected_material_encoding_receipt_hash=expected_material_encoding_receipt.material_encoding_receipt_hash, observed_material_encoding_receipt_hash=obs_hash, substrate_contract_hash=expected_material_encoding_receipt.substrate_contract_hash, substrate_profile_id=expected_material_encoding_receipt.substrate_profile_id, expected_encoding_entry_hashes=expected_hashes, observed_encoding_entry_hashes=observed_hashes, drifted_encoding_entry_hashes=drifted_t, missing_encoding_entry_hashes=missing_t, unexpected_encoding_entry_hashes=unexpected_t, encoding_entry_count=len(exp), drift_count=drift_count, substrate_drift_class=dclass)
    return SubstrateDriftReceipt(**p, substrate_drift_receipt_hash=sha256_hex(p))

def validate_substrate_drift_receipt(receipt: SubstrateDriftReceipt) -> bool:
    try:
        if not isinstance(receipt, SubstrateDriftReceipt): raise ValueError(_ERR_INVALID_INPUT)
        for h in (receipt.expected_material_encoding_receipt_hash, receipt.substrate_contract_hash, receipt.substrate_drift_receipt_hash): _validate_sha(h)
        if receipt.observed_material_encoding_receipt_hash is not None: _validate_sha(receipt.observed_material_encoding_receipt_hash)
        tuples = (receipt.expected_encoding_entry_hashes, receipt.observed_encoding_entry_hashes, receipt.drifted_encoding_entry_hashes, receipt.missing_encoding_entry_hashes, receipt.unexpected_encoding_entry_hashes)
        for t in tuples:
            if not isinstance(t, tuple) or len(t) > _MAX_DRIFT_ENTRY_HASHES: raise ValueError(_ERR_INVALID_INPUT)
            for h in t: _validate_sha(h)
        for c in (receipt.encoding_entry_count, receipt.drift_count):
            if not isinstance(c, int) or isinstance(c, bool): raise ValueError(_ERR_INVALID_INPUT)
        if receipt.encoding_entry_count != len(receipt.expected_encoding_entry_hashes): raise ValueError(_ERR_SUBSTRATE_DRIFT_COUNT_MISMATCH)
        drift_count = len(receipt.drifted_encoding_entry_hashes)+len(receipt.missing_encoding_entry_hashes)+len(receipt.unexpected_encoding_entry_hashes)
        if receipt.drift_count != drift_count: raise ValueError(_ERR_SUBSTRATE_DRIFT_COUNT_MISMATCH)
        if receipt.substrate_drift_class not in _ALLOWED_SUBSTRATE_DRIFT_CLASSES: raise ValueError(_ERR_INVALID_SUBSTRATE_DRIFT_CLASS)
        if receipt.substrate_drift_class != _drift_class(receipt.observed_material_encoding_receipt_hash, receipt.missing_encoding_entry_hashes, receipt.unexpected_encoding_entry_hashes, receipt.drift_count): raise ValueError(_ERR_SUBSTRATE_DRIFT_CLASS_MISMATCH)
        p = _substrate_drift_receipt_payload(expected_material_encoding_receipt_hash=receipt.expected_material_encoding_receipt_hash, observed_material_encoding_receipt_hash=receipt.observed_material_encoding_receipt_hash, substrate_contract_hash=receipt.substrate_contract_hash, substrate_profile_id=receipt.substrate_profile_id, expected_encoding_entry_hashes=receipt.expected_encoding_entry_hashes, observed_encoding_entry_hashes=receipt.observed_encoding_entry_hashes, drifted_encoding_entry_hashes=receipt.drifted_encoding_entry_hashes, missing_encoding_entry_hashes=receipt.missing_encoding_entry_hashes, unexpected_encoding_entry_hashes=receipt.unexpected_encoding_entry_hashes, encoding_entry_count=receipt.encoding_entry_count, drift_count=receipt.drift_count, substrate_drift_class=receipt.substrate_drift_class)
        if sha256_hex(p) != receipt.substrate_drift_receipt_hash: raise ValueError(_ERR_HASH_MISMATCH)
        return True
    except (TypeError, AttributeError) as e: raise ValueError(_ERR_INVALID_INPUT) from e

def validate_encoding_entry_with_state(entry: EncodingEntry, substrate_state_receipt: SubstrateStateReceipt, predicate_evaluation_result: PredicateEvaluationResult) -> bool:
    validate_encoding_entry(entry); validate_substrate_state_receipt(substrate_state_receipt); validate_predicate_evaluation_result(predicate_evaluation_result)
    rebuilt = build_encoding_entry(substrate_state_receipt, predicate_evaluation_result, entry.encoding_index)
    if rebuilt.to_dict() != entry.to_dict(): raise ValueError(_ERR_ENCODING_ENTRY_MISMATCH)
    return True

def validate_material_encoding_receipt_with_state(receipt: MaterialEncodingReceipt, substrate_state_receipt: SubstrateStateReceipt) -> bool:
    validate_material_encoding_receipt(receipt); validate_substrate_state_receipt(substrate_state_receipt)
    rebuilt = build_material_encoding_receipt(substrate_state_receipt)
    if rebuilt.to_dict() != receipt.to_dict(): raise ValueError(_ERR_MATERIAL_ENCODING_RECEIPT_MISMATCH)
    return True

def validate_substrate_drift_receipt_with_materials(receipt: SubstrateDriftReceipt, expected_material_encoding_receipt: MaterialEncodingReceipt, observed_material_encoding_receipt: MaterialEncodingReceipt | None = None) -> bool:
    validate_substrate_drift_receipt(receipt); validate_material_encoding_receipt(expected_material_encoding_receipt)
    if observed_material_encoding_receipt is not None: validate_material_encoding_receipt(observed_material_encoding_receipt)
    rebuilt = build_substrate_drift_receipt(expected_material_encoding_receipt, observed_material_encoding_receipt)
    if rebuilt.to_dict() != receipt.to_dict(): raise ValueError(_ERR_SUBSTRATE_DRIFT_RECEIPT_MISMATCH)
    return True
