from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from qec.analysis.canonical_hashing import CanonicalHashingError, canonical_json, canonicalize_json, sha256_hex
from qec.analysis.canonicalization_engine import CanonicalDocument
from qec.analysis.res_rag_resonance_validation import ResonanceValidationReceipt
from qec.analysis.res_rag_semantic_field import SemanticFieldReceipt

_VERSION = "v151.4"
_ALLOWED_RULE_TYPES = {"REQUIRED_FIELD", "MONEY_TOTAL_EQUALS_SUM", "DATE_ORDER", "CURRENCY_CONSISTENCY", "DUPLICATE_IDENTITY", "ALLOW_RESONANCE_CLASSES"}
_ALLOWED_CASE_SOURCES = {"CANONICAL_RULE", "RESONANCE_RESULT", "DIGITAL_DECAY"}
_ALLOWED_FAILURE_TYPES = {"INVALID_FIELD", "INCONSISTENT_VALUE", "DUPLICATE_IDENTITY", "CROSS_FIELD_CONFLICT", "LAYOUT_AMBIGUITY", "RES_RAG_DIVERGENCE", "UNSUPPORTED_RAG_CLAIM", "GROUNDING_FAILURE", "SEMANTIC_CONTRADICTION"}
_ALLOWED_FAILURE_SUBTYPES = {"MISSING_REQUIRED_FIELD", "NULL_REQUIRED_FIELD", "TOTAL_MISMATCH", "DATE_SEQUENCE_VIOLATION", "CURRENCY_INCONSISTENCY", "DUPLICATE_IDENTITY_VALUE", "LAYOUT_AMBIGUITY_DETECTED", "RESONANCE_DIVERGENT", "RESONANCE_CONTRADICTORY", "RESONANCE_UNSUPPORTED", "EVIDENCE_WITHOUT_INTERPRETATION", "CLAIM_WITHOUT_EVIDENCE", "UNSUPPORTED_GENERATED_CLAIM", "GROUNDING_MISSING"}
_ALLOWED_SEVERITIES = {"REJECT", "FLAG"}
_ALLOWED_STATUS = {"EXTRACTION_VALIDATED", "ADVERSARIAL_FAILURE_DETECTED"}
_ALLOWED_REASONS = {"REQUIRED_FIELD_MISSING", "REQUIRED_FIELD_NULL", "MONEY_TOTAL_MISMATCH", "DATE_ORDER_VIOLATION", "CURRENCY_MISMATCH", "DUPLICATE_IDENTITY_VALUE", "LAYOUT_AMBIGUITY_DETECTED", "RESONANCE_CONTRADICTORY", "RESONANCE_DIVERGENT", "RESONANCE_UNSUPPORTED", "CLAIM_WITHOUT_EVIDENCE", "EVIDENCE_WITHOUT_INTERPRETATION", "UNSUPPORTED_GENERATED_CLAIM"}


def _invalid() -> ValueError:
    return ValueError("INVALID_INPUT")


def _canonical_json(value: Any) -> str:
    try:
        return canonical_json(value)
    except CanonicalHashingError as exc:
        raise _invalid() from exc


def _sha(value: Any) -> str:
    try:
        return sha256_hex(value)
    except CanonicalHashingError as exc:
        raise _invalid() from exc


def _canon(value: Any) -> Any:
    try:
        return canonicalize_json(value)
    except CanonicalHashingError as exc:
        raise _invalid() from exc




def _case_payload(*, case_id: str, case_source: str, rule_id: str | None, target: Any, failure_type: str, failure_subtype: str) -> dict[str, Any]:
    return {"case_id": case_id, "case_source": case_source, "rule_id": rule_id, "target": target, "failure_type": failure_type, "failure_subtype": failure_subtype}


def _result_payload(*, case_id: str, failure_type: str, failure_subtype: str, severity: str, detected: bool, reason: str, case_hash: str) -> dict[str, Any]:
    return {"case_id": case_id, "failure_type": failure_type, "failure_subtype": failure_subtype, "severity": severity, "detected": detected, "reason": reason, "case_hash": case_hash}


def _receipt_payload(*, version: str, canonical_hash: str, semantic_field_hash: str, resonance_receipt_hash: str, rule_set_hash: str, digital_decay_hash: str, results: tuple[ExtractionValidationResult, ...], result_count: int, reject_count: int, flag_count: int, status: str) -> dict[str, Any]:
    return {"version": version, "canonical_hash": canonical_hash, "semantic_field_hash": semantic_field_hash, "resonance_receipt_hash": resonance_receipt_hash, "rule_set_hash": rule_set_hash, "digital_decay_hash": digital_decay_hash, "results": results, "result_count": result_count, "reject_count": reject_count, "flag_count": flag_count, "status": status}

def _json_safe(value: Any) -> Any:
    v = _canon(value)
    if isinstance(v, Mapping):
        safe: dict[str, Any] = {}
        for k, vv in v.items():
            if not isinstance(k, str) or k == "":
                raise _invalid()
            safe[k] = _json_safe(vv)
        return MappingProxyType(safe)
    if isinstance(v, list):
        return tuple(_json_safe(i) for i in v)
    if isinstance(v, tuple):
        return tuple(_json_safe(i) for i in v)
    return v


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _thaw_json(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(v) for v in value]
    return value


def _resonance_failure_for_class(resonance_class: str, reason: str) -> tuple[str, str, str, str] | None:
    if resonance_class == "CONTRADICTORY":
        return ("SEMANTIC_CONTRADICTION", "RESONANCE_CONTRADICTORY", "REJECT", "RESONANCE_CONTRADICTORY")
    if resonance_class == "DIVERGENT":
        return ("RES_RAG_DIVERGENCE", "RESONANCE_DIVERGENT", "REJECT", "RESONANCE_DIVERGENT")
    if resonance_class == "UNSUPPORTED":
        if reason == "CLAIM_WITHOUT_EVIDENCE":
            return ("GROUNDING_FAILURE", "CLAIM_WITHOUT_EVIDENCE", "REJECT", "CLAIM_WITHOUT_EVIDENCE")
        return ("UNSUPPORTED_RAG_CLAIM", "UNSUPPORTED_GENERATED_CLAIM", "REJECT", "UNSUPPORTED_GENERATED_CLAIM")
    if resonance_class == "PARTIAL" and reason == "EVIDENCE_WITHOUT_INTERPRETATION":
        return ("GROUNDING_FAILURE", "EVIDENCE_WITHOUT_INTERPRETATION", "FLAG", "EVIDENCE_WITHOUT_INTERPRETATION")
    return None


def _money(value: Any) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    a = value.get("amount_minor_units")
    c = value.get("currency_code")
    e = value.get("minor_unit_exponent")
    if not isinstance(a, int) or isinstance(a, bool) or not isinstance(c, str) or c == "" or not isinstance(e, int) or isinstance(e, bool):
        return None
    return value


@dataclass(frozen=True)
class ExtractionValidationRule:
    rule_id: str
    rule_type: str
    parameters: Any
    severity: str
    rule_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.rule_id, str) or self.rule_id == "" or self.rule_type not in _ALLOWED_RULE_TYPES or self.severity not in _ALLOWED_SEVERITIES:
            raise _invalid()
        object.__setattr__(self, "parameters", _json_safe(self.parameters))
        if self.computed_stable_hash() != self.rule_hash:
            raise _invalid()

    def to_dict(self) -> dict[str, Any]:
        return {"rule_id": self.rule_id, "rule_type": self.rule_type, "parameters": _thaw_json(self.parameters), "severity": self.severity, "rule_hash": self.rule_hash}
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")
    def computed_stable_hash(self) -> str: return _sha({k: v for k, v in self.to_dict().items() if k != "rule_hash"})


@dataclass(frozen=True)
class ExtractionValidationCase:
    case_id: str; case_source: str; rule_id: str | None; target: Any; failure_type: str; failure_subtype: str; case_hash: str
    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or self.case_id == "" or self.case_source not in _ALLOWED_CASE_SOURCES:
            raise _invalid()
        if self.case_source != "RESONANCE_RESULT" and self.rule_id is None:
            raise _invalid()
        if self.failure_type not in _ALLOWED_FAILURE_TYPES or self.failure_subtype not in _ALLOWED_FAILURE_SUBTYPES:
            raise _invalid()
        object.__setattr__(self, "target", _json_safe(self.target))
        if self.computed_stable_hash() != self.case_hash:
            raise _invalid()
    def to_dict(self) -> dict[str, Any]: return {"case_id": self.case_id, "case_source": self.case_source, "rule_id": self.rule_id, "target": _thaw_json(self.target), "failure_type": self.failure_type, "failure_subtype": self.failure_subtype, "case_hash": self.case_hash}
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")
    def computed_stable_hash(self) -> str: return _sha({k: v for k, v in self.to_dict().items() if k != "case_hash"})


@dataclass(frozen=True)
class ExtractionValidationResult:
    case_id: str; failure_type: str; failure_subtype: str; severity: str; detected: bool; reason: str; case_hash: str; result_hash: str
    def __post_init__(self) -> None:
        if self.failure_type not in _ALLOWED_FAILURE_TYPES or self.failure_subtype not in _ALLOWED_FAILURE_SUBTYPES or self.reason not in _ALLOWED_REASONS:
            raise _invalid()
        if not isinstance(self.detected, bool) or (self.detected and self.severity not in _ALLOWED_SEVERITIES):
            raise _invalid()
        if self.computed_stable_hash() != self.result_hash:
            raise _invalid()
    def to_dict(self) -> dict[str, Any]: return {"case_id": self.case_id, "failure_type": self.failure_type, "failure_subtype": self.failure_subtype, "severity": self.severity, "detected": self.detected, "reason": self.reason, "case_hash": self.case_hash, "result_hash": self.result_hash}
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")
    def computed_stable_hash(self) -> str: return _sha({k: v for k, v in self.to_dict().items() if k != "result_hash"})


@dataclass(frozen=True)
class DigitalDecaySignature:
    failure_count: int; reject_count: int; flag_count: int; failure_type_counts: Mapping[str, int]; failure_subtype_counts: Mapping[str, int]; ordered_failure_hashes: tuple[str, ...]; decay_signature_hash: str
    def __post_init__(self) -> None:
        if self.failure_count != self.reject_count + self.flag_count or self.failure_count != len(self.ordered_failure_hashes):
            raise _invalid()
        if self.computed_stable_hash() != self.decay_signature_hash:
            raise _invalid()
    def to_dict(self) -> dict[str, Any]: return {"failure_count": self.failure_count, "reject_count": self.reject_count, "flag_count": self.flag_count, "failure_type_counts": dict(self.failure_type_counts), "failure_subtype_counts": dict(self.failure_subtype_counts), "ordered_failure_hashes": self.ordered_failure_hashes, "decay_signature_hash": self.decay_signature_hash}
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")
    def computed_stable_hash(self) -> str: return _sha({k: v for k, v in self.to_dict().items() if k != "decay_signature_hash"})


@dataclass(frozen=True)
class ExtractionValidationReceipt:
    version: str; canonical_hash: str; semantic_field_hash: str; resonance_receipt_hash: str; rule_set_hash: str; digital_decay_hash: str; results: tuple[ExtractionValidationResult, ...]; result_count: int; reject_count: int; flag_count: int; status: str; stable_hash: str
    def __post_init__(self) -> None:
        if self.version != _VERSION or self.status not in _ALLOWED_STATUS:
            raise _invalid()
        if tuple(sorted(self.results, key=lambda r: (r.failure_type, r.failure_subtype, r.case_id, r.result_hash))) != self.results:
            raise _invalid()
        if self.result_count != len(self.results) or self.reject_count != sum(1 for r in self.results if r.severity == "REJECT") or self.flag_count != sum(1 for r in self.results if r.severity == "FLAG"):
            raise _invalid()
        expected = "ADVERSARIAL_FAILURE_DETECTED" if self.results else "EXTRACTION_VALIDATED"
        if self.status != expected:
            raise _invalid()
        if self.computed_stable_hash() != self.stable_hash:
            raise _invalid()
    def to_dict(self) -> dict[str, Any]: return {"version": self.version, "canonical_hash": self.canonical_hash, "semantic_field_hash": self.semantic_field_hash, "resonance_receipt_hash": self.resonance_receipt_hash, "rule_set_hash": self.rule_set_hash, "digital_decay_hash": self.digital_decay_hash, "results": tuple(r.to_dict() for r in self.results), "result_count": self.result_count, "reject_count": self.reject_count, "flag_count": self.flag_count, "status": self.status, "stable_hash": self.stable_hash}
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")
    def computed_stable_hash(self) -> str: return _sha({k: v for k, v in self.to_dict().items() if k != "stable_hash"})


def run_adversarial_extraction_validation(canonical_document: CanonicalDocument, semantic_field_receipt: SemanticFieldReceipt, resonance_receipt: ResonanceValidationReceipt, rules: Sequence[ExtractionValidationRule]) -> ExtractionValidationReceipt:
    if not isinstance(canonical_document, CanonicalDocument) or not isinstance(semantic_field_receipt, SemanticFieldReceipt) or not isinstance(resonance_receipt, ResonanceValidationReceipt) or not isinstance(rules, Sequence):
        raise _invalid()
    if canonical_document.canonical_hash != _sha(canonical_document.canonical_payload):
        raise _invalid()
    if semantic_field_receipt.computed_stable_hash() != semantic_field_receipt.stable_hash or resonance_receipt.computed_stable_hash() != resonance_receipt.stable_hash:
        raise _invalid()
    if semantic_field_receipt.canonical_hash != canonical_document.canonical_hash or resonance_receipt.canonical_hash != canonical_document.canonical_hash or resonance_receipt.semantic_field_hash != semantic_field_receipt.semantic_field_hash:
        raise _invalid()
    rs = tuple(sorted(rules, key=lambda r: (r.rule_type, r.rule_id, r.rule_hash)))
    if any(not isinstance(r, ExtractionValidationRule) for r in rs) or len({r.rule_id for r in rs}) != len(rs):
        raise _invalid()
    results: list[ExtractionValidationResult] = []
    payload = canonical_document.canonical_payload
    for i, rule in enumerate(rs):
        p = rule.parameters
        if not isinstance(p, Mapping):
            raise _invalid()
        if rule.rule_type == "REQUIRED_FIELD":
            fn = p.get("field_name")
            if not isinstance(fn, str) or fn == "": raise _invalid()
            if fn not in payload:
                results.append(_mk(rule, i, "INVALID_FIELD", "MISSING_REQUIRED_FIELD", "REQUIRED_FIELD_MISSING", {"field_name": fn}))
            elif payload[fn] is None:
                results.append(_mk(rule, i, "INVALID_FIELD", "NULL_REQUIRED_FIELD", "REQUIRED_FIELD_NULL", {"field_name": fn}))
        elif rule.rule_type == "DATE_ORDER":
            e, l, allow = p.get("earlier_field"), p.get("later_field"), p.get("allow_equal")
            if not isinstance(e, str) or not isinstance(l, str) or not isinstance(allow, bool): raise _invalid()
            ed, ld = payload.get(e), payload.get(l)
            if not isinstance(ed, str) or not isinstance(ld, str): raise _invalid()
            try: ev, lv = date.fromisoformat(ed), date.fromisoformat(ld)
            except ValueError as exc: raise _invalid() from exc
            bad = lv < ev or (not allow and lv == ev)
            if bad: results.append(_mk(rule, i, "CROSS_FIELD_CONFLICT", "DATE_SEQUENCE_VIOLATION", "DATE_ORDER_VIOLATION", {"earlier_field": e, "later_field": l}))
        elif rule.rule_type == "CURRENCY_CONSISTENCY":
            fields = p.get("fields")
            if not isinstance(fields, (list, tuple)) or not fields or any(not isinstance(f, str) or f == "" for f in fields):
                raise _invalid()
            monies = [_money(payload.get(f)) for f in fields]
            if any(m is None for m in monies):
                raise _invalid()
            base = monies[0]
            if base is None:
                raise _invalid()
            if any((m["currency_code"] != base["currency_code"] or m["minor_unit_exponent"] != base["minor_unit_exponent"]) for m in monies[1:] if m is not None):
                results.append(_mk(rule, i, "INCONSISTENT_VALUE", "CURRENCY_INCONSISTENCY", "CURRENCY_MISMATCH", {"fields": fields}))
        elif rule.rule_type == "MONEY_TOTAL_EQUALS_SUM":
            tgt = p.get("target_field")
            comps = p.get("component_fields")
            tol = p.get("tolerance_minor_units")
            if not isinstance(tgt, str) or tgt == "" or not isinstance(comps, (list, tuple)) or not comps or any(not isinstance(c, str) or c == "" for c in comps) or not (isinstance(tol, int) and not isinstance(tol, bool)) or tol < 0:
                raise _invalid()
            t = _money(payload.get(tgt))
            cms = [_money(payload.get(c)) for c in comps]
            if t is None or any(m is None for m in cms):
                raise _invalid()
            if any((m["currency_code"] != t["currency_code"] or m["minor_unit_exponent"] != t["minor_unit_exponent"]) for m in cms if m is not None):
                results.append(_mk(rule, i, "INCONSISTENT_VALUE", "CURRENCY_INCONSISTENCY", "CURRENCY_MISMATCH", {"target_field": tgt, "component_fields": comps}))
            else:
                total = sum(m["amount_minor_units"] for m in cms if m is not None)
                if abs(t["amount_minor_units"] - total) > tol:
                    results.append(_mk(rule, i, "INCONSISTENT_VALUE", "TOTAL_MISMATCH", "MONEY_TOTAL_MISMATCH", {"target_field": tgt, "component_fields": comps}))
        elif rule.rule_type == "DUPLICATE_IDENTITY":
            fn = p.get("field_name")
            vals = payload.get(fn)
            if not isinstance(fn, str) or fn == "" or not isinstance(vals, tuple):
                raise _invalid()
            seen: set[str] = set()
            dup = False
            for v in vals:
                if isinstance(v, (dict, list, tuple, Mapping)):
                    raise _invalid()
                h = _canonical_json(v)
                if h in seen:
                    dup = True
                    break
                seen.add(h)
            if dup:
                results.append(_mk(rule, i, "DUPLICATE_IDENTITY", "DUPLICATE_IDENTITY_VALUE", "DUPLICATE_IDENTITY_VALUE", {"field_name": fn}))
        elif rule.rule_type == "ALLOW_RESONANCE_CLASSES":
            allowed = p.get("allowed_classes")
            if not isinstance(allowed, (list, tuple)) or any(not isinstance(x, str) for x in allowed):
                raise _invalid()
            allowed_set = set(allowed)
            for rr in resonance_receipt.results:
                if rr.resonance_class in allowed_set:
                    continue
                mapped = _resonance_failure_for_class(rr.resonance_class, rr.reason)
                if mapped is not None:
                    ft, fs, _, reason = mapped
                    results.append(_mk(rule, i, ft, fs, reason, {"field_name": rr.field_name}))
    for rr in resonance_receipt.results:
        mapped = _resonance_failure_for_class(rr.resonance_class, rr.reason)
        if mapped is not None:
            ft, fs, sev, reason = mapped
            if rr.resonance_class == "UNSUPPORTED" and reason == "UNSUPPORTED_GENERATED_CLAIM":
                fs = "RESONANCE_UNSUPPORTED"
            results.append(_mk_res(rr.case_id, ft, fs, sev, reason, {"field_name": rr.field_name}))

    out = tuple(sorted(results, key=lambda r: (r.failure_type, r.failure_subtype, r.case_id, r.result_hash)))
    rule_set_hash = _sha(tuple(r.to_dict() for r in rs))
    d = _build_decay(out)
    receipt_payload = _receipt_payload(version=_VERSION, canonical_hash=canonical_document.canonical_hash, semantic_field_hash=semantic_field_receipt.semantic_field_hash, resonance_receipt_hash=resonance_receipt.stable_hash, rule_set_hash=rule_set_hash, digital_decay_hash=d.decay_signature_hash, results=out, result_count=len(out), reject_count=sum(1 for r in out if r.severity == "REJECT"), flag_count=sum(1 for r in out if r.severity == "FLAG"), status="ADVERSARIAL_FAILURE_DETECTED" if out else "EXTRACTION_VALIDATED")
    stable = _sha({**receipt_payload, "results": tuple(r.to_dict() for r in out)})
    return ExtractionValidationReceipt(**receipt_payload, stable_hash=stable)


def _mk(rule: ExtractionValidationRule, idx: int, ft: str, fst: str, reason: str, target: Any) -> ExtractionValidationResult:
    case_id = f"rule:{rule.rule_id}:{idx}"
    cp = _case_payload(case_id=case_id, case_source="CANONICAL_RULE", rule_id=rule.rule_id, target=target, failure_type=ft, failure_subtype=fst)
    case = ExtractionValidationCase(**cp, case_hash=_sha(cp))
    rp = _result_payload(case_id=case.case_id, failure_type=ft, failure_subtype=fst, severity=rule.severity, detected=True, reason=reason, case_hash=case.case_hash)
    return ExtractionValidationResult(**rp, result_hash=_sha(rp))


def _mk_res(case_id: str, ft: str, fst: str, sev: str, reason: str, target: Any) -> ExtractionValidationResult:
    cid = f"res:{case_id}:{ft}:{fst}"
    cp = _case_payload(case_id=cid, case_source="RESONANCE_RESULT", rule_id=None, target=target, failure_type=ft, failure_subtype=fst)
    case = ExtractionValidationCase(**cp, case_hash=_sha(cp))
    rp = _result_payload(case_id=case.case_id, failure_type=ft, failure_subtype=fst, severity=sev, detected=True, reason=reason, case_hash=case.case_hash)
    return ExtractionValidationResult(**rp, result_hash=_sha(rp))


def _build_decay(results: tuple[ExtractionValidationResult, ...]) -> DigitalDecaySignature:
    ft: dict[str, int] = {}
    fs: dict[str, int] = {}
    for r in results:
        ft[r.failure_type] = ft.get(r.failure_type, 0) + 1
        fs[r.failure_subtype] = fs.get(r.failure_subtype, 0) + 1
    ordered = tuple(sorted(r.result_hash for r in results))
    payload = {"failure_count": len(results), "reject_count": sum(1 for r in results if r.severity == "REJECT"), "flag_count": sum(1 for r in results if r.severity == "FLAG"), "failure_type_counts": dict(sorted(ft.items())), "failure_subtype_counts": dict(sorted(fs.items())), "ordered_failure_hashes": ordered}
    return DigitalDecaySignature(**payload, decay_signature_hash=_sha(payload))
