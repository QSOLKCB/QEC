from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qec.analysis.canonical_hashing import CanonicalHashingError, canonical_json, canonicalize_json, sha256_hex

_VERSION = "v151.7"

_EXTRACTION_ORDER = (
    "CONFIG_DRIFT", "SCHEMA_DRIFT", "LOCALE_DRIFT", "FIELD_DRIFT", "PARTIAL_EXTRACTION",
    "BACKEND_CONFIG_DRIFT", "CANONICALIZATION_RULE_DRIFT", "BACKEND_INCONSISTENCY", "CANONICAL_OUTPUT_DRIFT",
)
_RESRAG_ORDER = (
    "RES_RAG_MAPPING_DRIFT", "RESONANCE_CLASSIFIER_DRIFT", "TOLERANCE_DRIFT", "SEMANTIC_FIELD_DRIFT",
    "RES_STATE_DRIFT", "RAG_STATE_DRIFT", "RESONANCE_OUTPUT_DRIFT",
)
_ALLOWED_DRIFT = set(_EXTRACTION_ORDER) | set(_RESRAG_ORDER)
_ALLOWED_REASONS = {
    "CONFIG_HASH_CHANGED", "SCHEMA_HASH_CHANGED", "LOCALE_HASH_CHANGED", "QUERY_FIELDS_CHANGED", "EXTRACTED_FIELDS_CHANGED",
    "PARTIAL_EXTRACTION_DETECTED", "BACKEND_CONFIG_CHANGED", "EXTRACTION_HASH_CHANGED_UNDER_SAME_CONFIG",
    "CANONICALIZATION_RULES_CHANGED", "CANONICAL_HASH_CHANGED_UNDER_SAME_RULES", "RES_RAG_MAPPING_CHANGED",
    "RESONANCE_CLASSIFIER_CHANGED", "TOLERANCE_HASH_CHANGED", "SEMANTIC_FIELD_HASH_CHANGED", "RES_HASH_CHANGED",
    "RAG_HASH_CHANGED", "RESONANCE_RECEIPT_HASH_CHANGED",
}
_REASON_BY_DRIFT = {
    "CONFIG_DRIFT": "CONFIG_HASH_CHANGED", "SCHEMA_DRIFT": "SCHEMA_HASH_CHANGED", "LOCALE_DRIFT": "LOCALE_HASH_CHANGED",
    "FIELD_DRIFT": {"query": "QUERY_FIELDS_CHANGED", "extracted": "EXTRACTED_FIELDS_CHANGED"},
    "PARTIAL_EXTRACTION": "PARTIAL_EXTRACTION_DETECTED", "BACKEND_CONFIG_DRIFT": "BACKEND_CONFIG_CHANGED",
    "BACKEND_INCONSISTENCY": "EXTRACTION_HASH_CHANGED_UNDER_SAME_CONFIG", "CANONICALIZATION_RULE_DRIFT": "CANONICALIZATION_RULES_CHANGED",
    "CANONICAL_OUTPUT_DRIFT": "CANONICAL_HASH_CHANGED_UNDER_SAME_RULES", "RES_RAG_MAPPING_DRIFT": "RES_RAG_MAPPING_CHANGED",
    "RESONANCE_CLASSIFIER_DRIFT": "RESONANCE_CLASSIFIER_CHANGED", "TOLERANCE_DRIFT": "TOLERANCE_HASH_CHANGED",
    "SEMANTIC_FIELD_DRIFT": "SEMANTIC_FIELD_HASH_CHANGED", "RES_STATE_DRIFT": "RES_HASH_CHANGED",
    "RAG_STATE_DRIFT": "RAG_HASH_CHANGED", "RESONANCE_OUTPUT_DRIFT": "RESONANCE_RECEIPT_HASH_CHANGED",
}


def _invalid() -> ValueError: return ValueError("INVALID_INPUT")

def _sha(value: Any) -> str:
    try: return sha256_hex(value)
    except CanonicalHashingError as exc: raise _invalid() from exc

def _canonical_json(value: Any) -> str:
    try: return canonical_json(value)
    except CanonicalHashingError as exc: raise _invalid() from exc

def _json_safe(value: Any) -> Any:
    try:
        v = canonicalize_json(value)
    except CanonicalHashingError as exc:
        raise _invalid() from exc
    if isinstance(v, dict):
        for k in v:
            if not isinstance(k, str) or k == "": raise _invalid()
        return {k: _json_safe(vv) for k, vv in v.items()}
    if isinstance(v, tuple): return tuple(_json_safe(x) for x in v)
    return v

def _is_sha(v: Any) -> bool: return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)
def _require_str(v: Any) -> str:
    if isinstance(v, bool) or not isinstance(v, str) or not v or v.strip() != v: raise _invalid()
    return v

def _require_sha(v: Any) -> str:
    if not _is_sha(v): raise _invalid()
    return v

def _require_tuple_str(v: Any, *, allow_empty: bool = False, unique: bool = False) -> tuple[str, ...]:
    if not isinstance(v, tuple): raise _invalid()
    if not allow_empty and not v: raise _invalid()
    out: list[str] = []
    for item in v:
        out.append(_require_str(item))
    if unique and len(set(out)) != len(out): raise _invalid()
    return tuple(out)

def _mk_sha() -> str: return "0" * 64


def _extraction_snapshot_payload(x: "ExtractionDeterminismSnapshot") -> dict[str, Any]:
    return {"snapshot_id": x.snapshot_id, "raw_bytes_hash": x.raw_bytes_hash, "extraction_config_hash": x.extraction_config_hash, "schema_hash": x.schema_hash, "query_fields": x.query_fields, "extracted_field_names": x.extracted_field_names, "locale_hash": x.locale_hash, "backend_config_hash": x.backend_config_hash, "canonicalization_rules_hash": x.canonicalization_rules_hash, "extraction_hash": x.extraction_hash, "canonical_hash": x.canonical_hash}

def _resrag_snapshot_payload(x: "RESRAGDeterminismSnapshot") -> dict[str, Any]:
    return {"snapshot_id": x.snapshot_id, "canonical_hash": x.canonical_hash, "res_hash": x.res_hash, "rag_hash": x.rag_hash, "semantic_field_hash": x.semantic_field_hash, "res_rag_mapping_hash": x.res_rag_mapping_hash, "governance_context_hash": x.governance_context_hash, "resonance_classifier_hash": x.resonance_classifier_hash, "tolerance_hash": x.tolerance_hash, "resonance_receipt_hash": x.resonance_receipt_hash}

def _drift_case_payload(x: "DeterminismDriftCase") -> dict[str, Any]:
    return {"case_id": x.case_id, "drift_type": x.drift_type, "baseline_value_hash": x.baseline_value_hash, "observed_value_hash": x.observed_value_hash, "target": x.target}

def _drift_result_payload(x: "DeterminismDriftResult") -> dict[str, Any]:
    return {"case_id": x.case_id, "drift_type": x.drift_type, "detected": x.detected, "severity": x.severity, "reason": x.reason, "case_hash": x.case_hash}

def _extraction_receipt_payload(x: "ExtractionDeterminismReceipt") -> dict[str, Any]:
    return {"version": x.version, "baseline_snapshot_hash": x.baseline_snapshot_hash, "observed_snapshot_hash": x.observed_snapshot_hash, "raw_bytes_hash": x.raw_bytes_hash, "results": tuple(r.to_dict() for r in x.results), "result_count": x.result_count, "reject_count": x.reject_count, "flag_count": x.flag_count, "status": x.status}

def _resrag_receipt_payload(x: "RESRAGDeterminismReceipt") -> dict[str, Any]:
    return {"version": x.version, "baseline_snapshot_hash": x.baseline_snapshot_hash, "observed_snapshot_hash": x.observed_snapshot_hash, "canonical_hash": x.canonical_hash, "results": tuple(r.to_dict() for r in x.results), "result_count": x.result_count, "reject_count": x.reject_count, "flag_count": x.flag_count, "status": x.status}


@dataclass(frozen=True)
class ExtractionDeterminismSnapshot:
    snapshot_id: str; raw_bytes_hash: str; extraction_config_hash: str; schema_hash: str; query_fields: tuple[str, ...]; extracted_field_names: tuple[str, ...]; locale_hash: str; backend_config_hash: str; canonicalization_rules_hash: str; extraction_hash: str; canonical_hash: str; snapshot_hash: str
    def __post_init__(self) -> None:
        _require_str(self.snapshot_id)
        for h in (self.raw_bytes_hash, self.extraction_config_hash, self.schema_hash, self.locale_hash, self.backend_config_hash, self.canonicalization_rules_hash, self.extraction_hash, self.canonical_hash, self.snapshot_hash): _require_sha(h)
        _require_tuple_str(self.query_fields, unique=True); _require_tuple_str(self.extracted_field_names, unique=True)
        if self.computed_stable_hash() != self.snapshot_hash: raise _invalid()
    def to_dict(self) -> dict[str, Any]: return {**_extraction_snapshot_payload(self), "snapshot_hash": self.snapshot_hash}
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")
    def computed_stable_hash(self) -> str: return _sha(_extraction_snapshot_payload(self))

@dataclass(frozen=True)
class RESRAGDeterminismSnapshot:
    snapshot_id: str; canonical_hash: str; res_hash: str; rag_hash: str; semantic_field_hash: str; res_rag_mapping_hash: str; governance_context_hash: str; resonance_classifier_hash: str; tolerance_hash: str; resonance_receipt_hash: str; snapshot_hash: str
    def __post_init__(self) -> None:
        _require_str(self.snapshot_id)
        for h in (self.canonical_hash, self.res_hash, self.rag_hash, self.semantic_field_hash, self.res_rag_mapping_hash, self.governance_context_hash, self.resonance_classifier_hash, self.tolerance_hash, self.resonance_receipt_hash, self.snapshot_hash): _require_sha(h)
        if self.computed_stable_hash() != self.snapshot_hash: raise _invalid()
    def to_dict(self) -> dict[str, Any]: return {**_resrag_snapshot_payload(self), "snapshot_hash": self.snapshot_hash}
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")
    def computed_stable_hash(self) -> str: return _sha(_resrag_snapshot_payload(self))

@dataclass(frozen=True)
class DeterminismDriftCase:
    case_id: str; drift_type: str; baseline_value_hash: str; observed_value_hash: str; target: Any; case_hash: str
    def __post_init__(self) -> None:
        _require_str(self.case_id)
        if self.drift_type not in _ALLOWED_DRIFT: raise _invalid()
        _require_sha(self.baseline_value_hash); _require_sha(self.observed_value_hash)
        object.__setattr__(self, "target", _json_safe(self.target))
        if self.computed_stable_hash() != self.case_hash: raise _invalid()
    def to_dict(self) -> dict[str, Any]: return {**_drift_case_payload(self), "case_hash": self.case_hash}
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")
    def computed_stable_hash(self) -> str: return _sha(_drift_case_payload(self))

@dataclass(frozen=True)
class DeterminismDriftResult:
    case_id: str; drift_type: str; detected: bool; severity: str; reason: str; case_hash: str; result_hash: str
    def __post_init__(self) -> None:
        _require_str(self.case_id)
        if self.drift_type not in _ALLOWED_DRIFT or not isinstance(self.detected, bool) or self.severity not in {"REJECT", "FLAG"} or self.reason not in _ALLOWED_REASONS: raise _invalid()
        _require_sha(self.case_hash); _require_sha(self.result_hash)
        if self.computed_stable_hash() != self.result_hash: raise _invalid()
    def to_dict(self) -> dict[str, Any]: return {**_drift_result_payload(self), "result_hash": self.result_hash}
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")
    def computed_stable_hash(self) -> str: return _sha(_drift_result_payload(self))


def _validate_sorted_results(results: tuple[DeterminismDriftResult, ...]) -> None:
    ordered = tuple(sorted(results, key=lambda r: (r.drift_type, r.case_id, r.result_hash)))
    if ordered != results: raise _invalid()

@dataclass(frozen=True)
class ExtractionDeterminismReceipt:
    version: str; baseline_snapshot_hash: str; observed_snapshot_hash: str; raw_bytes_hash: str; results: tuple[DeterminismDriftResult, ...]; result_count: int; reject_count: int; flag_count: int; status: str; stable_hash: str
    def __post_init__(self) -> None:
        if self.version != _VERSION or self.status not in {"EXTRACTION_DETERMINISM_VALIDATED", "EXTRACTION_DETERMINISM_DRIFT_DETECTED"}: raise _invalid()
        for h in (self.baseline_snapshot_hash, self.observed_snapshot_hash, self.raw_bytes_hash, self.stable_hash): _require_sha(h)
        if not isinstance(self.results, tuple) or any(not isinstance(r, DeterminismDriftResult) for r in self.results): raise _invalid()
        _validate_sorted_results(self.results)
        if self.result_count != len(self.results): raise _invalid()
        rj = sum(1 for r in self.results if r.detected and r.severity == "REJECT"); fl = sum(1 for r in self.results if r.detected and r.severity == "FLAG")
        if self.reject_count != rj or self.flag_count != fl: raise _invalid()
        expected_status = "EXTRACTION_DETERMINISM_DRIFT_DETECTED" if any(r.detected for r in self.results) else "EXTRACTION_DETERMINISM_VALIDATED"
        if self.status != expected_status or self.computed_stable_hash() != self.stable_hash: raise _invalid()
    def to_dict(self) -> dict[str, Any]: return {**_extraction_receipt_payload(self), "stable_hash": self.stable_hash}
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")
    def computed_stable_hash(self) -> str: return _sha(_extraction_receipt_payload(self))

@dataclass(frozen=True)
class RESRAGDeterminismReceipt:
    version: str; baseline_snapshot_hash: str; observed_snapshot_hash: str; canonical_hash: str; results: tuple[DeterminismDriftResult, ...]; result_count: int; reject_count: int; flag_count: int; status: str; stable_hash: str
    def __post_init__(self) -> None:
        if self.version != _VERSION or self.status not in {"RES_RAG_DETERMINISM_VALIDATED", "RES_RAG_DETERMINISM_DRIFT_DETECTED"}: raise _invalid()
        for h in (self.baseline_snapshot_hash, self.observed_snapshot_hash, self.canonical_hash, self.stable_hash): _require_sha(h)
        if not isinstance(self.results, tuple) or any(not isinstance(r, DeterminismDriftResult) for r in self.results): raise _invalid()
        _validate_sorted_results(self.results)
        if self.result_count != len(self.results): raise _invalid()
        rj = sum(1 for r in self.results if r.detected and r.severity == "REJECT"); fl = sum(1 for r in self.results if r.detected and r.severity == "FLAG")
        if self.reject_count != rj or self.flag_count != fl: raise _invalid()
        expected_status = "RES_RAG_DETERMINISM_DRIFT_DETECTED" if any(r.detected for r in self.results) else "RES_RAG_DETERMINISM_VALIDATED"
        if self.status != expected_status or self.computed_stable_hash() != self.stable_hash: raise _invalid()
    def to_dict(self) -> dict[str, Any]: return {**_resrag_receipt_payload(self), "stable_hash": self.stable_hash}
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")
    def computed_stable_hash(self) -> str: return _sha(_resrag_receipt_payload(self))


def _mk_case(prefix: str, drift_type: str, baseline_hash: str, observed_hash: str, target: Any, suffix: str = "") -> DeterminismDriftCase:
    cid = f"{prefix}:{drift_type}{suffix}"
    payload = {"case_id": cid, "drift_type": drift_type, "baseline_value_hash": baseline_hash, "observed_value_hash": observed_hash, "target": target}
    return DeterminismDriftCase(**payload, case_hash=_sha(payload))

def _mk_result(case: DeterminismDriftCase, severity: str, reason: str) -> DeterminismDriftResult:
    payload = {"case_id": case.case_id, "drift_type": case.drift_type, "detected": True, "severity": severity, "reason": reason, "case_hash": case.case_hash}
    return DeterminismDriftResult(**payload, result_hash=_sha(payload))


def run_extraction_determinism_enforcement(baseline_snapshot: ExtractionDeterminismSnapshot, observed_snapshot: ExtractionDeterminismSnapshot) -> ExtractionDeterminismReceipt:
    if not isinstance(baseline_snapshot, ExtractionDeterminismSnapshot) or not isinstance(observed_snapshot, ExtractionDeterminismSnapshot): raise _invalid()
    if baseline_snapshot.computed_stable_hash() != baseline_snapshot.snapshot_hash or observed_snapshot.computed_stable_hash() != observed_snapshot.snapshot_hash: raise _invalid()
    if baseline_snapshot.raw_bytes_hash != observed_snapshot.raw_bytes_hash: raise _invalid()
    out: list[DeterminismDriftResult] = []
    b, o = baseline_snapshot, observed_snapshot
    if b.extraction_config_hash != o.extraction_config_hash: out.append(_mk_result(_mk_case("extraction", "CONFIG_DRIFT", b.extraction_config_hash, o.extraction_config_hash, "extraction_config_hash"), "REJECT", _REASON_BY_DRIFT["CONFIG_DRIFT"]))
    if b.schema_hash != o.schema_hash: out.append(_mk_result(_mk_case("extraction", "SCHEMA_DRIFT", b.schema_hash, o.schema_hash, "schema_hash"), "REJECT", _REASON_BY_DRIFT["SCHEMA_DRIFT"]))
    if b.locale_hash != o.locale_hash: out.append(_mk_result(_mk_case("extraction", "LOCALE_DRIFT", b.locale_hash, o.locale_hash, "locale_hash"), "REJECT", _REASON_BY_DRIFT["LOCALE_DRIFT"]))
    if b.query_fields != o.query_fields: out.append(_mk_result(_mk_case("extraction", "FIELD_DRIFT", _sha(b.query_fields), _sha(o.query_fields), "query_fields", ":query"), "REJECT", _REASON_BY_DRIFT["FIELD_DRIFT"]["query"]))
    if o.extracted_field_names != o.query_fields: out.append(_mk_result(_mk_case("extraction", "PARTIAL_EXTRACTION", _sha(o.query_fields), _sha(o.extracted_field_names), "observed_fields", ":observed"), "REJECT", _REASON_BY_DRIFT["PARTIAL_EXTRACTION"]))
    if b.extracted_field_names != b.query_fields: out.append(_mk_result(_mk_case("extraction", "PARTIAL_EXTRACTION", _sha(b.query_fields), _sha(b.extracted_field_names), "baseline_fields", ":baseline"), "REJECT", _REASON_BY_DRIFT["PARTIAL_EXTRACTION"]))
    if b.extracted_field_names != o.extracted_field_names: out.append(_mk_result(_mk_case("extraction", "FIELD_DRIFT", _sha(b.extracted_field_names), _sha(o.extracted_field_names), "extracted_field_names", ":extracted"), "REJECT", _REASON_BY_DRIFT["FIELD_DRIFT"]["extracted"]))
    if b.backend_config_hash != o.backend_config_hash: out.append(_mk_result(_mk_case("extraction", "BACKEND_CONFIG_DRIFT", b.backend_config_hash, o.backend_config_hash, "backend_config_hash"), "REJECT", _REASON_BY_DRIFT["BACKEND_CONFIG_DRIFT"]))
    if b.canonicalization_rules_hash != o.canonicalization_rules_hash: out.append(_mk_result(_mk_case("extraction", "CANONICALIZATION_RULE_DRIFT", b.canonicalization_rules_hash, o.canonicalization_rules_hash, "canonicalization_rules_hash"), "REJECT", _REASON_BY_DRIFT["CANONICALIZATION_RULE_DRIFT"]))
    if b.extraction_config_hash == o.extraction_config_hash and b.extraction_hash != o.extraction_hash: out.append(_mk_result(_mk_case("extraction", "BACKEND_INCONSISTENCY", b.extraction_hash, o.extraction_hash, "extraction_hash"), "REJECT", _REASON_BY_DRIFT["BACKEND_INCONSISTENCY"]))
    if b.canonicalization_rules_hash == o.canonicalization_rules_hash and b.canonical_hash != o.canonical_hash: out.append(_mk_result(_mk_case("extraction", "CANONICAL_OUTPUT_DRIFT", b.canonical_hash, o.canonical_hash, "canonical_hash"), "REJECT", _REASON_BY_DRIFT["CANONICAL_OUTPUT_DRIFT"]))
    results = tuple(sorted(out, key=lambda r: (r.drift_type, r.case_id, r.result_hash)))
    status = "EXTRACTION_DETERMINISM_DRIFT_DETECTED" if results else "EXTRACTION_DETERMINISM_VALIDATED"
    payload = {"version": _VERSION, "baseline_snapshot_hash": b.snapshot_hash, "observed_snapshot_hash": o.snapshot_hash, "raw_bytes_hash": b.raw_bytes_hash, "results": results, "result_count": len(results), "reject_count": sum(1 for r in results if r.severity == "REJECT"), "flag_count": sum(1 for r in results if r.severity == "FLAG"), "status": status}
    return ExtractionDeterminismReceipt(**payload, stable_hash=_sha({**payload, "results": tuple(r.to_dict() for r in results)}))


def run_res_rag_determinism_enforcement(baseline_snapshot: RESRAGDeterminismSnapshot, observed_snapshot: RESRAGDeterminismSnapshot) -> RESRAGDeterminismReceipt:
    if not isinstance(baseline_snapshot, RESRAGDeterminismSnapshot) or not isinstance(observed_snapshot, RESRAGDeterminismSnapshot): raise _invalid()
    if baseline_snapshot.computed_stable_hash() != baseline_snapshot.snapshot_hash or observed_snapshot.computed_stable_hash() != observed_snapshot.snapshot_hash: raise _invalid()
    if baseline_snapshot.canonical_hash != observed_snapshot.canonical_hash: raise _invalid()
    b, o = baseline_snapshot, observed_snapshot
    out: list[DeterminismDriftResult] = []
    if b.res_rag_mapping_hash != o.res_rag_mapping_hash: out.append(_mk_result(_mk_case("resrag", "RES_RAG_MAPPING_DRIFT", b.res_rag_mapping_hash, o.res_rag_mapping_hash, "res_rag_mapping_hash"), "REJECT", _REASON_BY_DRIFT["RES_RAG_MAPPING_DRIFT"]))
    if b.resonance_classifier_hash != o.resonance_classifier_hash: out.append(_mk_result(_mk_case("resrag", "RESONANCE_CLASSIFIER_DRIFT", b.resonance_classifier_hash, o.resonance_classifier_hash, "resonance_classifier_hash"), "REJECT", _REASON_BY_DRIFT["RESONANCE_CLASSIFIER_DRIFT"]))
    if b.tolerance_hash != o.tolerance_hash: out.append(_mk_result(_mk_case("resrag", "TOLERANCE_DRIFT", b.tolerance_hash, o.tolerance_hash, "tolerance_hash"), "FLAG", _REASON_BY_DRIFT["TOLERANCE_DRIFT"]))
    if b.semantic_field_hash != o.semantic_field_hash: out.append(_mk_result(_mk_case("resrag", "SEMANTIC_FIELD_DRIFT", b.semantic_field_hash, o.semantic_field_hash, "semantic_field_hash"), "REJECT", _REASON_BY_DRIFT["SEMANTIC_FIELD_DRIFT"]))
    if b.res_hash != o.res_hash: out.append(_mk_result(_mk_case("resrag", "RES_STATE_DRIFT", b.res_hash, o.res_hash, "res_hash"), "REJECT", _REASON_BY_DRIFT["RES_STATE_DRIFT"]))
    if b.rag_hash != o.rag_hash: out.append(_mk_result(_mk_case("resrag", "RAG_STATE_DRIFT", b.rag_hash, o.rag_hash, "rag_hash"), "REJECT", _REASON_BY_DRIFT["RAG_STATE_DRIFT"]))
    if b.resonance_receipt_hash != o.resonance_receipt_hash: out.append(_mk_result(_mk_case("resrag", "RESONANCE_OUTPUT_DRIFT", b.resonance_receipt_hash, o.resonance_receipt_hash, "resonance_receipt_hash"), "REJECT", _REASON_BY_DRIFT["RESONANCE_OUTPUT_DRIFT"]))
    results = tuple(sorted(out, key=lambda r: (r.drift_type, r.case_id, r.result_hash)))
    status = "RES_RAG_DETERMINISM_DRIFT_DETECTED" if results else "RES_RAG_DETERMINISM_VALIDATED"
    payload = {"version": _VERSION, "baseline_snapshot_hash": b.snapshot_hash, "observed_snapshot_hash": o.snapshot_hash, "canonical_hash": b.canonical_hash, "results": results, "result_count": len(results), "reject_count": sum(1 for r in results if r.severity == "REJECT"), "flag_count": sum(1 for r in results if r.severity == "FLAG"), "status": status}
    return RESRAGDeterminismReceipt(**payload, stable_hash=_sha({**payload, "results": tuple(r.to_dict() for r in results)}))
