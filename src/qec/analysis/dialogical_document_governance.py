from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping

from qec.analysis.adversarial_extraction_validation import ExtractionValidationReceipt
from qec.analysis.canonical_hashing import CanonicalHashingError, canonical_json, canonicalize_json, sha256_hex
from qec.analysis.canonicalization_engine import CanonicalDocument
from qec.analysis.res_rag_resonance_validation import ResonanceValidationReceipt
from qec.analysis.res_rag_semantic_field import RAGState, RESState

_VERSION = "v151.5"
_ALLOWED_ROLES = (
    "EXTRACTION_AUDITOR",
    "RES_GROUNDING_AGENT",
    "RAG_INTERPRETATION_AGENT",
    "SEMANTIC_RESONANCE_VALIDATOR",
    "RECONCILER",
    "ARBITRATOR",
)
_CORE_ROLES = _ALLOWED_ROLES[:4]
_ALLOWED_DECISIONS = {"ACCEPT", "REJECT", "REPAIR", "ESCALATE", "ABSTAIN"}
_ALLOWED_REASONS = {
    "VALIDATION_CLEAN", "VALIDATION_REJECT_FAILURES", "VALIDATION_FLAG_FAILURES",
    "GROUNDING_CLEAN", "GROUNDING_FAILURE_REPAIRABLE", "GROUNDING_FAILURE_REJECTED",
    "RAG_CLAIMS_ALIGNED", "RAG_CLAIMS_UNSUPPORTED", "RAG_CLAIMS_CONTRADICTORY",
    "RESONANCE_VALIDATED", "RESONANCE_PARTIAL", "RESONANCE_DIVERGENT", "RESONANCE_CONTRADICTORY", "RESONANCE_UNSUPPORTED",
    "RECONCILED_ACCEPT", "RECONCILED_REJECT", "RECONCILED_REPAIR", "RECONCILED_ESCALATE", "RECONCILED_ABSTAIN",
    "ARBITRATED_ACCEPT", "ARBITRATED_REJECT", "ARBITRATED_REPAIR", "ARBITRATED_ESCALATE", "ARBITRATED_ABSTAIN",
    "NO_APPLICABLE_EVIDENCE",
}


def _invalid() -> ValueError: return ValueError("INVALID_INPUT")

def _canonical_json(value: Any) -> str:
    try: return canonical_json(value)
    except CanonicalHashingError as exc: raise _invalid() from exc

def _canon(value: Any) -> Any:
    try: return canonicalize_json(value)
    except CanonicalHashingError as exc: raise _invalid() from exc

def _sha(value: Any) -> str:
    try: return sha256_hex(value)
    except CanonicalHashingError as exc: raise _invalid() from exc

def _json_safe(value: Any) -> Any:
    v = _canon(value)
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        raise _invalid()
    if isinstance(v, Mapping):
        out: dict[str, Any] = {}
        for k, vv in v.items():
            if not isinstance(k, str) or k == "": raise _invalid()
            out[k] = _json_safe(vv)
        return MappingProxyType(out)
    if isinstance(v, list):
        return tuple(_json_safe(x) for x in v)
    if isinstance(v, tuple):
        return tuple(_json_safe(x) for x in v)
    return v

def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping): return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple): return [_thaw(v) for v in value]
    return value

def _is_sha256_hex(v: str) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


@dataclass(frozen=True)
class GovernanceAgentDecision:
    agent_role: str
    decision: str
    reason: str
    input_hashes: tuple[str, ...]
    decision_basis: Any
    agent_decision_hash: str

    def __post_init__(self) -> None:
        if self.agent_role not in _ALLOWED_ROLES or self.decision not in _ALLOWED_DECISIONS or self.reason not in _ALLOWED_REASONS: raise _invalid()
        if not isinstance(self.input_hashes, tuple) or not self.input_hashes or any(not _is_sha256_hex(h) for h in self.input_hashes): raise _invalid()
        object.__setattr__(self, "input_hashes", tuple(sorted(set(self.input_hashes))))
        object.__setattr__(self, "decision_basis", _json_safe(self.decision_basis))
        if self.computed_stable_hash() != self.agent_decision_hash: raise _invalid()
    def to_dict(self) -> dict[str, Any]:
        return {"agent_role": self.agent_role, "decision": self.decision, "reason": self.reason, "input_hashes": list(self.input_hashes), "decision_basis": _thaw(self.decision_basis), "agent_decision_hash": self.agent_decision_hash}
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")
    def computed_stable_hash(self) -> str: return _sha({k: v for k, v in self.to_dict().items() if k != "agent_decision_hash"})


@dataclass(frozen=True)
class GovernanceDecisionSet:
    agent_decisions: tuple[GovernanceAgentDecision, ...]
    decision_set_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.agent_decisions, tuple) or len(self.agent_decisions) != len(_ALLOWED_ROLES): raise _invalid()
        if any(not isinstance(d, GovernanceAgentDecision) for d in self.agent_decisions): raise _invalid()
        roles = tuple(d.agent_role for d in self.agent_decisions)
        if roles != _ALLOWED_ROLES or len(set(roles)) != len(roles): raise _invalid()
        if self.computed_stable_hash() != self.decision_set_hash: raise _invalid()
    def to_dict(self) -> dict[str, Any]:
        return {"agent_decisions": [d.to_dict() for d in self.agent_decisions], "decision_set_hash": self.decision_set_hash}
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")
    def computed_stable_hash(self) -> str: return _sha({"agent_decisions": tuple(d.to_dict() for d in self.agent_decisions)})


@dataclass(frozen=True)
class DialogicalGovernanceReceipt:
    version: str; canonical_hash: str; res_hash: str; rag_hash: str; resonance_receipt_hash: str; extraction_validation_hash: str; decision_set_hash: str
    final_decision: str; final_reason: str; agent_decisions: tuple[GovernanceAgentDecision, ...]
    decision_count: int; accept_count: int; reject_count: int; repair_count: int; escalate_count: int; abstain_count: int; status: str; stable_hash: str
    def __post_init__(self) -> None:
        if self.version != _VERSION or self.status != "GOVERNANCE_DECIDED" or self.final_decision not in _ALLOWED_DECISIONS or self.final_reason not in _ALLOWED_REASONS: raise _invalid()
        if not isinstance(self.agent_decisions, tuple) or any(not isinstance(d, GovernanceAgentDecision) for d in self.agent_decisions): raise _invalid()
        if tuple(d.agent_role for d in self.agent_decisions) != _ALLOWED_ROLES: raise _invalid()
        if self.decision_count != len(self.agent_decisions): raise _invalid()
        counts = {k: sum(1 for d in self.agent_decisions if d.decision == k) for k in _ALLOWED_DECISIONS}
        if (self.accept_count != counts["ACCEPT"] or self.reject_count != counts["REJECT"] or self.repair_count != counts["REPAIR"] or self.escalate_count != counts["ESCALATE"] or self.abstain_count != counts["ABSTAIN"]): raise _invalid()
        if self.agent_decisions[-1].decision != self.final_decision or self.agent_decisions[-1].reason != self.final_reason: raise _invalid()
        expected_ds_hash = _sha({"agent_decisions": tuple(d.to_dict() for d in self.agent_decisions)})
        if self.decision_set_hash != expected_ds_hash: raise _invalid()
        if self.computed_stable_hash() != self.stable_hash: raise _invalid()
    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "canonical_hash": self.canonical_hash, "res_hash": self.res_hash, "rag_hash": self.rag_hash, "resonance_receipt_hash": self.resonance_receipt_hash, "extraction_validation_hash": self.extraction_validation_hash, "decision_set_hash": self.decision_set_hash, "final_decision": self.final_decision, "final_reason": self.final_reason, "agent_decisions": [d.to_dict() for d in self.agent_decisions], "decision_count": self.decision_count, "accept_count": self.accept_count, "reject_count": self.reject_count, "repair_count": self.repair_count, "escalate_count": self.escalate_count, "abstain_count": self.abstain_count, "status": self.status, "stable_hash": self.stable_hash}
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")
    def computed_stable_hash(self) -> str: return _sha({k: v for k, v in self.to_dict().items() if k != "stable_hash"})


def _mk(role: str, decision: str, reason: str, hashes: tuple[str, ...], basis: Any) -> GovernanceAgentDecision:
    p = {"agent_role": role, "decision": decision, "reason": reason, "input_hashes": tuple(sorted(set(hashes))), "decision_basis": basis}
    return GovernanceAgentDecision(**p, agent_decision_hash=_sha(p))


def run_dialogical_document_governance(canonical_document: CanonicalDocument, res_state: RESState, rag_state: RAGState, resonance_receipt: ResonanceValidationReceipt, extraction_validation_receipt: ExtractionValidationReceipt) -> DialogicalGovernanceReceipt:
    if not isinstance(canonical_document, CanonicalDocument) or not isinstance(res_state, RESState) or not isinstance(rag_state, RAGState) or not isinstance(resonance_receipt, ResonanceValidationReceipt) or not isinstance(extraction_validation_receipt, ExtractionValidationReceipt): raise _invalid()
    if canonical_document.canonical_hash != _sha(canonical_document.canonical_payload): raise _invalid()
    if res_state.computed_stable_hash() != res_state.res_hash or rag_state.computed_stable_hash() != rag_state.rag_hash: raise _invalid()
    if resonance_receipt.computed_stable_hash() != resonance_receipt.stable_hash or extraction_validation_receipt.computed_stable_hash() != extraction_validation_receipt.stable_hash: raise _invalid()
    if res_state.canonical_document_hash != canonical_document.canonical_hash or rag_state.canonical_document_hash != canonical_document.canonical_hash: raise _invalid()
    if resonance_receipt.canonical_hash != canonical_document.canonical_hash or resonance_receipt.res_hash != res_state.res_hash or resonance_receipt.rag_hash != rag_state.rag_hash: raise _invalid()
    if extraction_validation_receipt.canonical_hash != canonical_document.canonical_hash or extraction_validation_receipt.resonance_receipt_hash != resonance_receipt.stable_hash: raise _invalid()
    if extraction_validation_receipt.semantic_field_hash != resonance_receipt.semantic_field_hash: raise _invalid()

    r = extraction_validation_receipt.results
    ex = _mk("EXTRACTION_AUDITOR", "REJECT" if extraction_validation_receipt.reject_count > 0 else ("REPAIR" if extraction_validation_receipt.flag_count > 0 else "ACCEPT"), "VALIDATION_REJECT_FAILURES" if extraction_validation_receipt.reject_count > 0 else ("VALIDATION_FLAG_FAILURES" if extraction_validation_receipt.flag_count > 0 else "VALIDATION_CLEAN"), (extraction_validation_receipt.stable_hash,), {"reject_count": extraction_validation_receipt.reject_count, "flag_count": extraction_validation_receipt.flag_count})
    has_gr_rej = any(x.failure_type == "GROUNDING_FAILURE" and x.severity == "REJECT" for x in r)
    has_gr_flag = any(x.failure_type == "GROUNDING_FAILURE" and x.severity == "FLAG" for x in r)
    rg = _mk("RES_GROUNDING_AGENT", "REJECT" if has_gr_rej else ("REPAIR" if has_gr_flag else "ACCEPT"), "GROUNDING_FAILURE_REJECTED" if has_gr_rej else ("GROUNDING_FAILURE_REPAIRABLE" if has_gr_flag else "GROUNDING_CLEAN"), (extraction_validation_receipt.stable_hash, res_state.res_hash), {"result_count": extraction_validation_receipt.result_count})
    has_con = any(x.failure_type == "SEMANTIC_CONTRADICTION" for x in r)
    has_uns = any(x.failure_type == "UNSUPPORTED_RAG_CLAIM" for x in r)
    ri = _mk("RAG_INTERPRETATION_AGENT", "REJECT" if (has_con or has_uns) else "ACCEPT", "RAG_CLAIMS_CONTRADICTORY" if has_con else ("RAG_CLAIMS_UNSUPPORTED" if has_uns else "RAG_CLAIMS_ALIGNED"), (extraction_validation_receipt.stable_hash, rag_state.rag_hash), {"result_count": extraction_validation_receipt.result_count})
    arc = resonance_receipt.aggregate_resonance_class
    if arc in {"IDENTICAL", "ALIGNED"}: sd, sr = "ACCEPT", "RESONANCE_VALIDATED"
    elif arc == "PARTIAL": sd, sr = "REPAIR", "RESONANCE_PARTIAL"
    elif arc == "DIVERGENT": sd, sr = "REJECT", "RESONANCE_DIVERGENT"
    elif arc == "CONTRADICTORY": sd, sr = "REJECT", "RESONANCE_CONTRADICTORY"
    elif arc == "UNSUPPORTED": sd, sr = "REJECT", "RESONANCE_UNSUPPORTED"
    else: raise _invalid()
    sv = _mk("SEMANTIC_RESONANCE_VALIDATOR", sd, sr, (resonance_receipt.stable_hash,), {"aggregate_resonance_class": arc})

    core = (ex, rg, ri, sv)
    cds = tuple(d.decision for d in core)
    if "REJECT" in cds: rd, rr = "REJECT", "RECONCILED_REJECT"
    elif "REPAIR" in cds: rd, rr = "REPAIR", "RECONCILED_REPAIR"
    elif "ESCALATE" in cds: rd, rr = "ESCALATE", "RECONCILED_ESCALATE"
    elif all(d == "ACCEPT" for d in cds): rd, rr = "ACCEPT", "RECONCILED_ACCEPT"
    else: rd, rr = "ABSTAIN", "RECONCILED_ABSTAIN"
    rec = _mk("RECONCILER", rd, rr, tuple(d.agent_decision_hash for d in core), {"core_decisions": cds})
    amap = {"REJECT": "ARBITRATED_REJECT", "REPAIR": "ARBITRATED_REPAIR", "ESCALATE": "ARBITRATED_ESCALATE", "ACCEPT": "ARBITRATED_ACCEPT", "ABSTAIN": "ARBITRATED_ABSTAIN"}
    arb = _mk("ARBITRATOR", rec.decision, amap[rec.decision], (rec.agent_decision_hash,) + tuple(d.agent_decision_hash for d in core), {"reconciler_decision": rec.decision})
    ads = (ex, rg, ri, sv, rec, arb)
    ds = GovernanceDecisionSet(ads, _sha({"agent_decisions": tuple(d.to_dict() for d in ads)}))
    counts = {k: sum(1 for d in ads if d.decision == k) for k in _ALLOWED_DECISIONS}
    payload = {"version": _VERSION, "canonical_hash": canonical_document.canonical_hash, "res_hash": res_state.res_hash, "rag_hash": rag_state.rag_hash, "resonance_receipt_hash": resonance_receipt.stable_hash, "extraction_validation_hash": extraction_validation_receipt.stable_hash, "decision_set_hash": ds.decision_set_hash, "final_decision": arb.decision, "final_reason": arb.reason, "agent_decisions": ads, "decision_count": len(ads), "accept_count": counts["ACCEPT"], "reject_count": counts["REJECT"], "repair_count": counts["REPAIR"], "escalate_count": counts["ESCALATE"], "abstain_count": counts["ABSTAIN"], "status": "GOVERNANCE_DECIDED"}
    return DialogicalGovernanceReceipt(**payload, stable_hash=_sha({k: (tuple(d.to_dict() for d in v) if k == "agent_decisions" else v) for k, v in payload.items()}))
