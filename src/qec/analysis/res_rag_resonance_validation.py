from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from qec.analysis.canonical_hashing import CanonicalHashingError, canonical_json, canonicalize_json, sha256_hex
from qec.analysis.res_rag_semantic_field import RESState, RAGState, SemanticFieldReceipt

_VERSION = "v151.3"

_ALLOWED_CASE_TYPES = {"CLAIM_TO_EVIDENCE", "EVIDENCE_WITHOUT_INTERPRETATION"}
_ALLOWED_CLASSES = {"IDENTICAL", "ALIGNED", "PARTIAL", "DIVERGENT", "CONTRADICTORY", "UNSUPPORTED"}
_ALLOWED_REASONS = {
    "FIELD_VALUE_IDENTICAL",
    "FIELD_PRESENT_ALIGNED",
    "FIELD_SUBSET_PARTIAL",
    "FIELD_SUBSET_DIVERGENT",
    "FIELD_VALUE_CONTRADICTORY",
    "CLAIM_WITHOUT_EVIDENCE",
    "EVIDENCE_WITHOUT_INTERPRETATION",
    "UNSUPPORTED_CLAIM_TYPE",
    "UNSUPPORTED_CLAIM_SHAPE",
}
_ALLOWED_STATUS = {"RESONANCE_VALIDATED", "RESONANCE_DIVERGENCE_DETECTED"}
_VALID_CLASS_REASON = {
    ("IDENTICAL", "FIELD_VALUE_IDENTICAL"),
    ("ALIGNED", "FIELD_PRESENT_ALIGNED"),
    ("PARTIAL", "FIELD_SUBSET_PARTIAL"),
    ("PARTIAL", "EVIDENCE_WITHOUT_INTERPRETATION"),
    ("DIVERGENT", "FIELD_SUBSET_DIVERGENT"),
    ("CONTRADICTORY", "FIELD_VALUE_CONTRADICTORY"),
    ("UNSUPPORTED", "CLAIM_WITHOUT_EVIDENCE"),
    ("UNSUPPORTED", "UNSUPPORTED_CLAIM_TYPE"),
    ("UNSUPPORTED", "UNSUPPORTED_CLAIM_SHAPE"),
}


def _invalid() -> ValueError:
    return ValueError("INVALID_INPUT")


def _canonical_json(value: Any) -> str:
    try:
        return canonical_json(value)
    except CanonicalHashingError as exc:
        raise _invalid() from exc


def _canon(value: Any) -> Any:
    try:
        return canonicalize_json(value)
    except CanonicalHashingError as exc:
        raise _invalid() from exc


def _sha(value: Any) -> str:
    try:
        return sha256_hex(value)
    except CanonicalHashingError as exc:
        raise _invalid() from exc


def _is_subset(claim_value: Any, evidence_value: Any) -> bool:
    if not isinstance(claim_value, Mapping) or not isinstance(evidence_value, Mapping):
        return False
    for k, v in claim_value.items():
        if k not in evidence_value:
            return False
        rv = evidence_value[k]
        if isinstance(v, Mapping):
            if not isinstance(rv, Mapping) or not _is_subset(v, rv):
                return False
        else:
            if _canon(v) != _canon(rv):
                return False
    return True


@dataclass(frozen=True)
class ResonanceCase:
    case_id: str
    case_type: str
    field_name: str
    claim_id: str | None
    claim_hash: str | None
    evidence_value_hash: str | None
    case_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or self.case_id == "":
            raise _invalid()
        if self.case_type not in _ALLOWED_CASE_TYPES:
            raise _invalid()
        if not isinstance(self.field_name, str) or self.field_name == "":
            raise _invalid()
        if self.case_type == "EVIDENCE_WITHOUT_INTERPRETATION":
            if self.claim_id is not None or self.claim_hash is not None:
                raise _invalid()
            if not isinstance(self.evidence_value_hash, str) or self.evidence_value_hash == "":
                raise _invalid()
        else:
            if not isinstance(self.claim_id, str) or self.claim_id == "":
                raise _invalid()
            if not isinstance(self.claim_hash, str) or self.claim_hash == "":
                raise _invalid()
        if self.computed_stable_hash() != self.case_hash:
            raise _invalid()

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "case_type": self.case_type,
            "field_name": self.field_name,
            "claim_id": self.claim_id,
            "claim_hash": self.claim_hash,
            "evidence_value_hash": self.evidence_value_hash,
            "case_hash": self.case_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def computed_stable_hash(self) -> str:
        return _sha({k: v for k, v in self.to_dict().items() if k != "case_hash"})


@dataclass(frozen=True)
class ResonanceResult:
    case_id: str
    field_name: str
    resonance_class: str
    reason: str
    case_hash: str
    result_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or self.case_id == "":
            raise _invalid()
        if not isinstance(self.field_name, str) or self.field_name == "":
            raise _invalid()
        if self.resonance_class not in _ALLOWED_CLASSES or self.reason not in _ALLOWED_REASONS:
            raise _invalid()
        if (self.resonance_class, self.reason) not in _VALID_CLASS_REASON:
            raise _invalid()
        if self.computed_stable_hash() != self.result_hash:
            raise _invalid()

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "field_name": self.field_name,
            "resonance_class": self.resonance_class,
            "reason": self.reason,
            "case_hash": self.case_hash,
            "result_hash": self.result_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def computed_stable_hash(self) -> str:
        return _sha({k: v for k, v in self.to_dict().items() if k != "result_hash"})


@dataclass(frozen=True)
class ResonanceValidationReceipt:
    version: str
    semantic_field_hash: str
    canonical_hash: str
    res_hash: str
    rag_hash: str
    aggregate_resonance_class: str
    status: str
    results: tuple[ResonanceResult, ...]
    result_count: int
    identical_count: int
    aligned_count: int
    partial_count: int
    divergent_count: int
    contradictory_count: int
    unsupported_count: int
    stable_hash: str

    def __post_init__(self) -> None:
        if self.version != _VERSION or self.aggregate_resonance_class not in _ALLOWED_CLASSES or self.status not in _ALLOWED_STATUS:
            raise _invalid()
        if self.semantic_field_hash != _semantic_field_lineage_hash(self.canonical_hash, self.res_hash, self.rag_hash):
            raise _invalid()
        if not isinstance(self.results, tuple) or any(not isinstance(r, ResonanceResult) for r in self.results):
            raise _invalid()
        if tuple(sorted(self.results, key=lambda r: (r.field_name, r.case_id, r.result_hash))) != self.results:
            raise _invalid()
        counts = {k: 0 for k in _ALLOWED_CLASSES}
        for r in self.results:
            counts[r.resonance_class] += 1
        if self.result_count != len(self.results):
            raise _invalid()
        if (self.identical_count != counts["IDENTICAL"] or self.aligned_count != counts["ALIGNED"] or self.partial_count != counts["PARTIAL"] or self.divergent_count != counts["DIVERGENT"] or self.contradictory_count != counts["CONTRADICTORY"] or self.unsupported_count != counts["UNSUPPORTED"]):
            raise _invalid()
        if self.aggregate_resonance_class != _aggregate(self.results):
            raise _invalid()
        if self.status != _status_for(self.aggregate_resonance_class):
            raise _invalid()
        if self.computed_stable_hash() != self.stable_hash:
            raise _invalid()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "semantic_field_hash": self.semantic_field_hash,
            "canonical_hash": self.canonical_hash,
            "res_hash": self.res_hash,
            "rag_hash": self.rag_hash,
            "aggregate_resonance_class": self.aggregate_resonance_class,
            "status": self.status,
            "results": tuple(r.to_dict() for r in self.results),
            "result_count": self.result_count,
            "identical_count": self.identical_count,
            "aligned_count": self.aligned_count,
            "partial_count": self.partial_count,
            "divergent_count": self.divergent_count,
            "contradictory_count": self.contradictory_count,
            "unsupported_count": self.unsupported_count,
            "stable_hash": self.stable_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def computed_stable_hash(self) -> str:
        return _sha({k: v for k, v in self.to_dict().items() if k != "stable_hash"})


def _aggregate(results: tuple[ResonanceResult, ...]) -> str:
    classes = {r.resonance_class for r in results}
    if "CONTRADICTORY" in classes:
        return "CONTRADICTORY"
    if "DIVERGENT" in classes:
        return "DIVERGENT"
    if "UNSUPPORTED" in classes:
        return "UNSUPPORTED"
    if "PARTIAL" in classes:
        return "PARTIAL"
    if results and all(r.resonance_class == "IDENTICAL" for r in results):
        return "IDENTICAL"
    return "ALIGNED"


def _status_for(aggregate: str) -> str:
    if aggregate in {"IDENTICAL", "ALIGNED"}:
        return "RESONANCE_VALIDATED"
    return "RESONANCE_DIVERGENCE_DETECTED"




def _semantic_field_lineage_hash(canonical_hash: str, res_hash: str, rag_hash: str) -> str:
    return _sha({"canonical_hash": canonical_hash, "res_hash": res_hash, "rag_hash": rag_hash})


def _case_payload(*, case_id: str, case_type: str, field_name: str, claim_id: str | None, claim_hash: str | None, evidence_value_hash: str | None) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "case_type": case_type,
        "field_name": field_name,
        "claim_id": claim_id,
        "claim_hash": claim_hash,
        "evidence_value_hash": evidence_value_hash,
    }


def _result_payload(*, case_id: str, field_name: str, resonance_class: str, reason: str, case_hash: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "field_name": field_name,
        "resonance_class": resonance_class,
        "reason": reason,
        "case_hash": case_hash,
    }


def _build_case(*, case_id: str, case_type: str, field_name: str, claim_id: str | None, claim_hash: str | None, evidence_value_hash: str | None) -> ResonanceCase:
    payload = _case_payload(case_id=case_id, case_type=case_type, field_name=field_name, claim_id=claim_id, claim_hash=claim_hash, evidence_value_hash=evidence_value_hash)
    return ResonanceCase(**payload, case_hash=_sha(payload))


def _build_result(*, case: ResonanceCase, field_name: str, resonance_class: str, reason: str) -> ResonanceResult:
    payload = _result_payload(case_id=case.case_id, field_name=field_name, resonance_class=resonance_class, reason=reason, case_hash=case.case_hash)
    return ResonanceResult(**payload, result_hash=_sha(payload))


def run_res_rag_resonance_validation(semantic_field_receipt: SemanticFieldReceipt, res_state: RESState, rag_state: RAGState) -> ResonanceValidationReceipt:
    if not isinstance(semantic_field_receipt, SemanticFieldReceipt) or not isinstance(res_state, RESState) or not isinstance(rag_state, RAGState):
        raise _invalid()
    if semantic_field_receipt.computed_stable_hash() != semantic_field_receipt.stable_hash:
        raise _invalid()
    if res_state.computed_stable_hash() != res_state.res_hash or rag_state.computed_stable_hash() != rag_state.rag_hash:
        raise _invalid()
    if semantic_field_receipt.canonical_hash != res_state.canonical_document_hash or semantic_field_receipt.canonical_hash != rag_state.canonical_document_hash:
        raise _invalid()
    if semantic_field_receipt.res_hash != res_state.res_hash or semantic_field_receipt.rag_hash != rag_state.rag_hash:
        raise _invalid()

    evidence_index = {e.field_name: e for e in res_state.evidence_fields}
    referenced_fields: set[str] = set()
    results: list[ResonanceResult] = []

    for claim in rag_state.generated_claims:
        payload = claim.claim_payload
        if not isinstance(payload, Mapping) or "claim_type" not in payload or "field_name" not in payload:
            claim_type = payload.get("claim_type") if isinstance(payload, Mapping) else None
            reason = "UNSUPPORTED_CLAIM_SHAPE"
            field_name = payload.get("field_name") if isinstance(payload, Mapping) and isinstance(payload.get("field_name"), str) and payload.get("field_name") else "unsupported"
            ev = evidence_index.get(field_name)
            c = _build_case(case_id=f"claim:{claim.claim_id}", case_type="CLAIM_TO_EVIDENCE", field_name=field_name, claim_id=claim.claim_id, claim_hash=claim.claim_hash, evidence_value_hash=ev.value_hash if ev else None)
            results.append(_build_result(case=c, field_name=field_name, resonance_class="UNSUPPORTED", reason=reason))
            continue
        claim_type = payload["claim_type"]
        field_name = payload["field_name"]
        if not isinstance(claim_type, str) or not isinstance(field_name, str) or field_name == "":
            c = _build_case(case_id=f"claim:{claim.claim_id}", case_type="CLAIM_TO_EVIDENCE", field_name="unsupported", claim_id=claim.claim_id, claim_hash=claim.claim_hash, evidence_value_hash=None)
            results.append(_build_result(case=c, field_name=c.field_name, resonance_class="UNSUPPORTED", reason="UNSUPPORTED_CLAIM_SHAPE"))
            continue
        evidence = evidence_index.get(field_name)
        if evidence is not None and claim_type in {"FIELD_EQUALS", "FIELD_PRESENT", "FIELD_SUBSET"}:
            referenced_fields.add(field_name)
        case = _build_case(case_id=f"claim:{claim.claim_id}", case_type="CLAIM_TO_EVIDENCE", field_name=field_name, claim_id=claim.claim_id, claim_hash=claim.claim_hash, evidence_value_hash=evidence.value_hash if evidence else None)

        if claim_type == "FIELD_EQUALS":
            if evidence is None:
                rc, reason = "UNSUPPORTED", "CLAIM_WITHOUT_EVIDENCE"
            elif "claim_value" not in payload:
                rc, reason = "UNSUPPORTED", "UNSUPPORTED_CLAIM_SHAPE"
            elif _canon(payload["claim_value"]) == _canon(evidence.canonical_value):
                rc, reason = "IDENTICAL", "FIELD_VALUE_IDENTICAL"
            else:
                rc, reason = "CONTRADICTORY", "FIELD_VALUE_CONTRADICTORY"
        elif claim_type == "FIELD_PRESENT":
            rc, reason = ("ALIGNED", "FIELD_PRESENT_ALIGNED") if evidence is not None else ("UNSUPPORTED", "CLAIM_WITHOUT_EVIDENCE")
        elif claim_type == "FIELD_SUBSET":
            if evidence is None:
                rc, reason = "UNSUPPORTED", "CLAIM_WITHOUT_EVIDENCE"
            elif "claim_value" not in payload:
                rc, reason = "UNSUPPORTED", "UNSUPPORTED_CLAIM_SHAPE"
            else:
                cv = payload["claim_value"]
                ev = evidence.canonical_value
                if _canon(cv) == _canon(ev):
                    rc, reason = "IDENTICAL", "FIELD_VALUE_IDENTICAL"
                elif _is_subset(cv, ev):
                    rc, reason = "PARTIAL", "FIELD_SUBSET_PARTIAL"
                else:
                    rc, reason = "DIVERGENT", "FIELD_SUBSET_DIVERGENT"
        else:
            rc, reason = "UNSUPPORTED", "UNSUPPORTED_CLAIM_TYPE"

        results.append(_build_result(case=case, field_name=case.field_name, resonance_class=rc, reason=reason))

    for field_name, ev in evidence_index.items():
        if field_name not in referenced_fields:
            case = _build_case(case_id=f"evidence:{field_name}", case_type="EVIDENCE_WITHOUT_INTERPRETATION", field_name=field_name, claim_id=None, claim_hash=None, evidence_value_hash=ev.value_hash)
            results.append(_build_result(case=case, field_name=field_name, resonance_class="PARTIAL", reason="EVIDENCE_WITHOUT_INTERPRETATION"))

    sorted_results = tuple(sorted(results, key=lambda r: (r.field_name, r.case_id, r.result_hash)))
    counts = {k: 0 for k in _ALLOWED_CLASSES}
    for r in sorted_results:
        counts[r.resonance_class] += 1
    aggregate = _aggregate(sorted_results)
    status = _status_for(aggregate)
    receipt = ResonanceValidationReceipt(
        version=_VERSION,
        semantic_field_hash=semantic_field_receipt.semantic_field_hash,
        canonical_hash=semantic_field_receipt.canonical_hash,
        res_hash=res_state.res_hash,
        rag_hash=rag_state.rag_hash,
        aggregate_resonance_class=aggregate,
        status=status,
        results=sorted_results,
        result_count=len(sorted_results),
        identical_count=counts["IDENTICAL"],
        aligned_count=counts["ALIGNED"],
        partial_count=counts["PARTIAL"],
        divergent_count=counts["DIVERGENT"],
        contradictory_count=counts["CONTRADICTORY"],
        unsupported_count=counts["UNSUPPORTED"],
        stable_hash=_sha({
            "version": _VERSION,
            "semantic_field_hash": semantic_field_receipt.semantic_field_hash,
            "canonical_hash": semantic_field_receipt.canonical_hash,
            "res_hash": res_state.res_hash,
            "rag_hash": rag_state.rag_hash,
            "aggregate_resonance_class": aggregate,
            "status": status,
            "results": tuple(r.to_dict() for r in sorted_results),
            "result_count": len(sorted_results),
            "identical_count": counts["IDENTICAL"],
            "aligned_count": counts["ALIGNED"],
            "partial_count": counts["PARTIAL"],
            "divergent_count": counts["DIVERGENT"],
            "contradictory_count": counts["CONTRADICTORY"],
            "unsupported_count": counts["UNSUPPORTED"],
        }),
    )
    if receipt.computed_stable_hash() != receipt.stable_hash:
        raise _invalid()
    return receipt
