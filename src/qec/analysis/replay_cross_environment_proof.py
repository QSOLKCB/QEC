from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping

from qec.analysis.canonical_hashing import (
    CanonicalHashingError,
    canonical_json as _canonical_json,
    canonicalize_json as _canonicalize_base,
    sha256_hex as _sha256_hex,
)

SCHEMA_VERSION = "v151.8"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _invalid() -> ValueError:
    return ValueError("INVALID_INPUT")


def _canonicalize(value: Any) -> Any:
    try:
        return _canonicalize_base(value)
    except CanonicalHashingError:
        raise _invalid()


def _cjson(value: Any) -> str:
    try:
        return _canonical_json(value)
    except CanonicalHashingError:
        raise _invalid()


def _sha(value: Any) -> str:
    try:
        return _sha256_hex(value)
    except CanonicalHashingError:
        raise _invalid()


def _req_str(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, str) or value.strip() != value or value == "":
        raise _invalid()
    return value


def _req_hash(value: Any) -> str:
    v = _req_str(value)
    if not _HEX64.match(v):
        raise _invalid()
    return v


def _require_sorted_results(results: tuple["ReplayDivergenceResult", ...]) -> None:
    key = lambda r: (r.divergence_type, r.case_id, r.result_hash)
    if tuple(sorted(results, key=key)) != results:
        raise _invalid()


@dataclass(frozen=True)
class ReplayEnvironment:
    environment_id: str
    python_version: str
    platform_id: str
    dependency_lock_hash: str
    backend_config_hash: str
    numeric_profile_hash: str
    environment_hash: str

    def __post_init__(self) -> None:
        _req_str(self.environment_id); _req_str(self.python_version); _req_str(self.platform_id)
        _req_hash(self.dependency_lock_hash); _req_hash(self.backend_config_hash); _req_hash(self.numeric_profile_hash); _req_hash(self.environment_hash)
        if self.environment_hash != self.computed_stable_hash(): raise _invalid()

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment_id": self.environment_id,
            "python_version": self.python_version,
            "platform_id": self.platform_id,
            "dependency_lock_hash": self.dependency_lock_hash,
            "backend_config_hash": self.backend_config_hash,
            "numeric_profile_hash": self.numeric_profile_hash,
            "environment_hash": self.environment_hash,
        }

    def to_canonical_json(self) -> str:
        return _cjson(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def computed_stable_hash(self) -> str:
        return _sha(_environment_payload(self))


def _environment_payload(v: ReplayEnvironment) -> dict[str, Any]:
    return {"environment_id": v.environment_id,"python_version": v.python_version,"platform_id": v.platform_id,"dependency_lock_hash": v.dependency_lock_hash,"backend_config_hash": v.backend_config_hash,"numeric_profile_hash": v.numeric_profile_hash}

@dataclass(frozen=True)
class ExtractionReplayEvidence:
    evidence_id: str
    environment_hash: str
    raw_bytes_hash: str
    extraction_config_hash: str
    schema_hash: str
    query_fields_hash: str
    locale_hash: str
    backend_config_hash: str
    canonicalization_rules_hash: str
    numeric_profile_hash: str
    extraction_hash: str
    canonical_hash: str
    evidence_hash: str

    def __post_init__(self) -> None:
        _req_str(self.evidence_id)
        for f in (
            "environment_hash",
            "raw_bytes_hash",
            "extraction_config_hash",
            "schema_hash",
            "query_fields_hash",
            "locale_hash",
            "backend_config_hash",
            "canonicalization_rules_hash",
            "numeric_profile_hash",
            "extraction_hash",
            "canonical_hash",
            "evidence_hash",
        ):
            _req_hash(getattr(self, f))
        if self.evidence_hash != self.computed_stable_hash():
            raise _invalid()

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in (
            "evidence_id",
            "environment_hash",
            "raw_bytes_hash",
            "extraction_config_hash",
            "schema_hash",
            "query_fields_hash",
            "locale_hash",
            "backend_config_hash",
            "canonicalization_rules_hash",
            "numeric_profile_hash",
            "extraction_hash",
            "canonical_hash",
            "evidence_hash",
        )}

    def to_canonical_json(self) -> str:
        return _cjson(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def computed_stable_hash(self) -> str:
        return _sha(_extraction_evidence_payload(self))

def _extraction_evidence_payload(v:ExtractionReplayEvidence)->dict[str,Any]: d=v.to_dict().copy(); d.pop("evidence_hash"); return d

_ALLOWED_ARC=("IDENTICAL","ALIGNED","PARTIAL","DIVERGENT","CONTRADICTORY","UNSUPPORTED")
@dataclass(frozen=True)
class ResonanceReplayEvidence:
    evidence_id:str; environment_hash:str; canonical_hash:str; res_hash:str; rag_hash:str; semantic_field_hash:str; res_rag_mapping_hash:str; governance_context_hash:str; resonance_classifier_hash:str; tolerance_hash:str; resonance_receipt_hash:str; aggregate_resonance_class:str; evidence_hash:str
    def __post_init__(self)->None:
        _req_str(self.evidence_id); _req_str(self.aggregate_resonance_class)
        if self.aggregate_resonance_class not in _ALLOWED_ARC: raise _invalid()
        for f in ("environment_hash","canonical_hash","res_hash","rag_hash","semantic_field_hash","res_rag_mapping_hash","governance_context_hash","resonance_classifier_hash","tolerance_hash","resonance_receipt_hash","evidence_hash"): _req_hash(getattr(self,f))
        if self.evidence_hash!=self.computed_stable_hash(): raise _invalid()
    def to_dict(self)->dict[str,Any]: return {k:getattr(self,k) for k in ("evidence_id","environment_hash","canonical_hash","res_hash","rag_hash","semantic_field_hash","res_rag_mapping_hash","governance_context_hash","resonance_classifier_hash","tolerance_hash","resonance_receipt_hash","aggregate_resonance_class","evidence_hash")}
    def to_canonical_json(self)->str:return _cjson(self.to_dict())
    def to_canonical_bytes(self)->bytes:return self.to_canonical_json().encode("utf-8")
    def computed_stable_hash(self)->str:return _sha(_resonance_evidence_payload(self))

def _resonance_evidence_payload(v:ResonanceReplayEvidence)->dict[str,Any]: d=v.to_dict().copy(); d.pop("evidence_hash"); return d

_ALLOWED_GD=("ACCEPT","REJECT","REPAIR","ESCALATE","ABSTAIN")
@dataclass(frozen=True)
class RealWorldReplayEvidence:
    evidence_id:str; environment_hash:str; canonical_hash:str; semantic_field_hash:str; resonance_receipt_hash:str; validation_hash:str; governance_hash:str; local_proof_hash:str; distributed_convergence_hash:str; final_proof_hash:str; governance_decision:str; evidence_hash:str
    def __post_init__(self)->None:
        _req_str(self.evidence_id); _req_str(self.governance_decision)
        if self.governance_decision not in _ALLOWED_GD: raise _invalid()
        for f in ("environment_hash","canonical_hash","semantic_field_hash","resonance_receipt_hash","validation_hash","governance_hash","local_proof_hash","distributed_convergence_hash","final_proof_hash","evidence_hash"): _req_hash(getattr(self,f))
        if self.evidence_hash!=self.computed_stable_hash(): raise _invalid()
    def to_dict(self)->dict[str,Any]: return {k:getattr(self,k) for k in ("evidence_id","environment_hash","canonical_hash","semantic_field_hash","resonance_receipt_hash","validation_hash","governance_hash","local_proof_hash","distributed_convergence_hash","final_proof_hash","governance_decision","evidence_hash")}
    def to_canonical_json(self)->str:return _cjson(self.to_dict())
    def to_canonical_bytes(self)->bytes:return self.to_canonical_json().encode("utf-8")
    def computed_stable_hash(self)->str:return _sha(_real_world_evidence_payload(self))

def _real_world_evidence_payload(v:RealWorldReplayEvidence)->dict[str,Any]: d=v.to_dict().copy(); d.pop("evidence_hash"); return d

_ALLOWED_DT=("FLOATING_POINT_DRIFT","ENVIRONMENT_DIVERGENCE","BACKEND_INCONSISTENCY","CANONICALIZATION_DRIFT","SEMANTIC_FIELD_DRIFT","RES_STATE_DRIFT","RAG_STATE_DRIFT","RESONANCE_CLASSIFICATION_DRIFT","VALIDATION_DRIFT","GOVERNANCE_DIVERGENCE","PROOF_CHAIN_DRIFT","FINAL_PROOF_DRIFT")
@dataclass(frozen=True)
class ReplayDivergenceCase:
    case_id:str; divergence_type:str; baseline_value_hash:str; observed_value_hash:str; target:dict[str,Any]; case_hash:str
    def __post_init__(self)->None:
        _req_str(self.case_id); _req_str(self.divergence_type)
        if self.divergence_type not in _ALLOWED_DT: raise _invalid()
        _req_hash(self.baseline_value_hash); _req_hash(self.observed_value_hash); _canonicalize(self.target)
        if self.case_hash != "":
            _req_hash(self.case_hash)
            if self.case_hash!=self.computed_stable_hash(): raise _invalid()
    def to_dict(self)->dict[str,Any]: return {"case_id":self.case_id,"divergence_type":self.divergence_type,"baseline_value_hash":self.baseline_value_hash,"observed_value_hash":self.observed_value_hash,"target":self.target,"case_hash":self.case_hash}
    def to_canonical_json(self)->str:return _cjson(self.to_dict())
    def to_canonical_bytes(self)->bytes:return self.to_canonical_json().encode("utf-8")
    def computed_stable_hash(self)->str:return _sha(_replay_case_payload(self))

def _replay_case_payload(v:ReplayDivergenceCase)->dict[str,Any]: d=v.to_dict().copy(); d.pop("case_hash"); return d

_ALLOWED_SEV=("REJECT","FLAG")
_ALLOWED_REASON=("NUMERIC_PROFILE_CHANGED","ENVIRONMENT_HASH_CHANGED_WITH_OUTPUT_DRIFT","BACKEND_OUTPUT_CHANGED","CANONICAL_HASH_CHANGED","SEMANTIC_FIELD_HASH_CHANGED","RES_HASH_CHANGED","RAG_HASH_CHANGED","RESONANCE_CLASS_CHANGED","RESONANCE_RECEIPT_HASH_CHANGED","VALIDATION_HASH_CHANGED","GOVERNANCE_HASH_CHANGED","GOVERNANCE_DECISION_CHANGED","LOCAL_PROOF_HASH_CHANGED","DISTRIBUTED_CONVERGENCE_HASH_CHANGED","FINAL_PROOF_HASH_CHANGED")
@dataclass(frozen=True)
class ReplayDivergenceResult:
    case_id:str; divergence_type:str; detected:bool; severity:str; reason:str; case_hash:str; result_hash:str
    def __post_init__(self)->None:
        _req_str(self.case_id); _req_str(self.divergence_type); _req_str(self.severity); _req_str(self.reason)
        if self.divergence_type not in _ALLOWED_DT or self.severity not in _ALLOWED_SEV or self.reason not in _ALLOWED_REASON or not isinstance(self.detected,bool): raise _invalid()
        _req_hash(self.case_hash)
        if self.result_hash != "":
            _req_hash(self.result_hash)
            if self.result_hash!=self.computed_stable_hash(): raise _invalid()
    def to_dict(self)->dict[str,Any]: return {"case_id":self.case_id,"divergence_type":self.divergence_type,"detected":self.detected,"severity":self.severity,"reason":self.reason,"case_hash":self.case_hash,"result_hash":self.result_hash}
    def to_canonical_json(self)->str:return _cjson(self.to_dict())
    def to_canonical_bytes(self)->bytes:return self.to_canonical_json().encode("utf-8")
    def computed_stable_hash(self)->str:return _sha(_replay_result_payload(self))

def _replay_result_payload(v:ReplayDivergenceResult)->dict[str,Any]: d=v.to_dict().copy(); d.pop("result_hash"); return d

@dataclass(frozen=True)
class ExtractionReplayReceipt:
    version:str; baseline_evidence_hash:str; observed_evidence_hash:str; raw_bytes_hash:str; results:tuple[ReplayDivergenceResult,...]; result_count:int; reject_count:int; flag_count:int; status:str; stable_hash:str
    def __post_init__(self)->None:
        if self.version!=SCHEMA_VERSION or self.status not in ("EXTRACTION_REPLAY_VALIDATED","EXTRACTION_REPLAY_DIVERGENCE_DETECTED"): raise _invalid()
        _req_hash(self.baseline_evidence_hash); _req_hash(self.observed_evidence_hash); _req_hash(self.raw_bytes_hash); _require_sorted_results(self.results)
        if self.stable_hash != "":
            _req_hash(self.stable_hash)
        if self.result_count!=len(self.results) or self.reject_count!=sum(1 for r in self.results if r.detected and r.severity=="REJECT") or self.flag_count!=sum(1 for r in self.results if r.detected and r.severity=="FLAG"): raise _invalid()
        expected="EXTRACTION_REPLAY_DIVERGENCE_DETECTED" if any(r.detected for r in self.results) else "EXTRACTION_REPLAY_VALIDATED"
        if self.status!=expected: raise _invalid()
        if self.stable_hash != "" and self.stable_hash!=self.computed_stable_hash(): raise _invalid()
    def to_dict(self)->dict[str,Any]: return {"version":self.version,"baseline_evidence_hash":self.baseline_evidence_hash,"observed_evidence_hash":self.observed_evidence_hash,"raw_bytes_hash":self.raw_bytes_hash,"results":tuple(r.to_dict() for r in self.results),"result_count":self.result_count,"reject_count":self.reject_count,"flag_count":self.flag_count,"status":self.status,"stable_hash":self.stable_hash}
    def to_canonical_json(self)->str:return _cjson(self.to_dict())
    def to_canonical_bytes(self)->bytes:return self.to_canonical_json().encode("utf-8")
    def computed_stable_hash(self)->str:return _sha(_extraction_replay_receipt_payload(self))

def _extraction_replay_receipt_payload(v:ExtractionReplayReceipt)->dict[str,Any]: d=v.to_dict().copy(); d.pop("stable_hash"); return d

@dataclass(frozen=True)
class ResonanceReplayReceipt:
    version:str; baseline_evidence_hash:str; observed_evidence_hash:str; canonical_hash:str; results:tuple[ReplayDivergenceResult,...]; result_count:int; reject_count:int; flag_count:int; status:str; stable_hash:str
    def __post_init__(self)->None:
        if self.version!=SCHEMA_VERSION or self.status not in ("RESONANCE_REPLAY_VALIDATED","RESONANCE_REPLAY_DIVERGENCE_DETECTED"): raise _invalid()
        _req_hash(self.baseline_evidence_hash); _req_hash(self.observed_evidence_hash); _req_hash(self.canonical_hash); _require_sorted_results(self.results)
        if self.stable_hash != "":
            _req_hash(self.stable_hash)
        if self.result_count!=len(self.results) or self.reject_count!=sum(1 for r in self.results if r.detected and r.severity=="REJECT") or self.flag_count!=sum(1 for r in self.results if r.detected and r.severity=="FLAG"): raise _invalid()
        expected="RESONANCE_REPLAY_DIVERGENCE_DETECTED" if any(r.detected for r in self.results) else "RESONANCE_REPLAY_VALIDATED"
        if self.status!=expected: raise _invalid()
        if self.stable_hash != "" and self.stable_hash!=self.computed_stable_hash(): raise _invalid()
    def to_dict(self)->dict[str,Any]: return {"version":self.version,"baseline_evidence_hash":self.baseline_evidence_hash,"observed_evidence_hash":self.observed_evidence_hash,"canonical_hash":self.canonical_hash,"results":tuple(r.to_dict() for r in self.results),"result_count":self.result_count,"reject_count":self.reject_count,"flag_count":self.flag_count,"status":self.status,"stable_hash":self.stable_hash}
    def to_canonical_json(self)->str:return _cjson(self.to_dict())
    def to_canonical_bytes(self)->bytes:return self.to_canonical_json().encode("utf-8")
    def computed_stable_hash(self)->str:return _sha(_resonance_replay_receipt_payload(self))

def _resonance_replay_receipt_payload(v:ResonanceReplayReceipt)->dict[str,Any]: d=v.to_dict().copy(); d.pop("stable_hash"); return d

@dataclass(frozen=True)
class RealWorldReplayProofReceipt:
    version:str; baseline_evidence_hash:str; observed_evidence_hash:str; canonical_hash:str; semantic_field_hash:str; extraction_replay_hash:str; resonance_replay_hash:str; results:tuple[ReplayDivergenceResult,...]; result_count:int; reject_count:int; flag_count:int; status:str; stable_hash:str
    def __post_init__(self)->None:
        if self.version!=SCHEMA_VERSION or self.status not in ("REAL_WORLD_REPLAY_VALIDATED","REAL_WORLD_REPLAY_DIVERGENCE_DETECTED"): raise _invalid()
        for f in ("baseline_evidence_hash","observed_evidence_hash","canonical_hash","semantic_field_hash","extraction_replay_hash","resonance_replay_hash"): _req_hash(getattr(self,f))
        if self.stable_hash != "":
            _req_hash(self.stable_hash)
        _require_sorted_results(self.results)
        if self.result_count!=len(self.results) or self.reject_count!=sum(1 for r in self.results if r.detected and r.severity=="REJECT") or self.flag_count!=sum(1 for r in self.results if r.detected and r.severity=="FLAG"): raise _invalid()
        expected="REAL_WORLD_REPLAY_DIVERGENCE_DETECTED" if any(r.detected for r in self.results) else "REAL_WORLD_REPLAY_VALIDATED"
        if self.status!=expected: raise _invalid()
        if self.stable_hash != "" and self.stable_hash!=self.computed_stable_hash(): raise _invalid()
    def to_dict(self)->dict[str,Any]: return {"version":self.version,"baseline_evidence_hash":self.baseline_evidence_hash,"observed_evidence_hash":self.observed_evidence_hash,"canonical_hash":self.canonical_hash,"semantic_field_hash":self.semantic_field_hash,"extraction_replay_hash":self.extraction_replay_hash,"resonance_replay_hash":self.resonance_replay_hash,"results":tuple(r.to_dict() for r in self.results),"result_count":self.result_count,"reject_count":self.reject_count,"flag_count":self.flag_count,"status":self.status,"stable_hash":self.stable_hash}
    def to_canonical_json(self)->str:return _cjson(self.to_dict())
    def to_canonical_bytes(self)->bytes:return self.to_canonical_json().encode("utf-8")
    def computed_stable_hash(self)->str:return _sha(_real_world_replay_receipt_payload(self))

def _real_world_replay_receipt_payload(v:RealWorldReplayProofReceipt)->dict[str,Any]: d=v.to_dict().copy(); d.pop("stable_hash"); return d


def _mk_result(prefix:str,dtype:str,b:str,o:str,reason:str,severity:str)->ReplayDivergenceResult:
    c=ReplayDivergenceCase(f"{prefix}:{dtype}",dtype,b,o,{"target":dtype},"")
    c=ReplayDivergenceCase(c.case_id,c.divergence_type,c.baseline_value_hash,c.observed_value_hash,c.target,c.computed_stable_hash())
    r=ReplayDivergenceResult(c.case_id,dtype,True,severity,reason,c.case_hash,"")
    return ReplayDivergenceResult(r.case_id,r.divergence_type,r.detected,r.severity,r.reason,r.case_hash,r.computed_stable_hash())


def run_extraction_replay_validation(baseline_evidence: ExtractionReplayEvidence, observed_evidence: ExtractionReplayEvidence) -> ExtractionReplayReceipt:
    if not isinstance(baseline_evidence, ExtractionReplayEvidence) or not isinstance(observed_evidence, ExtractionReplayEvidence): raise _invalid()
    if baseline_evidence.evidence_hash != baseline_evidence.computed_stable_hash() or observed_evidence.evidence_hash != observed_evidence.computed_stable_hash(): raise _invalid()
    for k in ("raw_bytes_hash","extraction_config_hash","schema_hash","query_fields_hash","locale_hash","canonicalization_rules_hash"):
        if getattr(baseline_evidence,k)!=getattr(observed_evidence,k): raise _invalid()
    results=[]
    if baseline_evidence.numeric_profile_hash!=observed_evidence.numeric_profile_hash and baseline_evidence.canonical_hash!=observed_evidence.canonical_hash: results.append(_mk_result("extraction_replay","FLOATING_POINT_DRIFT",baseline_evidence.numeric_profile_hash,observed_evidence.numeric_profile_hash,"NUMERIC_PROFILE_CHANGED","FLAG"))
    if baseline_evidence.extraction_hash!=observed_evidence.extraction_hash: results.append(_mk_result("extraction_replay","BACKEND_INCONSISTENCY",baseline_evidence.extraction_hash,observed_evidence.extraction_hash,"BACKEND_OUTPUT_CHANGED","REJECT"))
    if baseline_evidence.canonical_hash!=observed_evidence.canonical_hash: results.append(_mk_result("extraction_replay","CANONICALIZATION_DRIFT",baseline_evidence.canonical_hash,observed_evidence.canonical_hash,"CANONICAL_HASH_CHANGED","REJECT"))
    if baseline_evidence.environment_hash!=observed_evidence.environment_hash and (baseline_evidence.extraction_hash!=observed_evidence.extraction_hash or baseline_evidence.canonical_hash!=observed_evidence.canonical_hash): results.append(_mk_result("extraction_replay","ENVIRONMENT_DIVERGENCE",baseline_evidence.environment_hash,observed_evidence.environment_hash,"ENVIRONMENT_HASH_CHANGED_WITH_OUTPUT_DRIFT","REJECT"))
    tup = tuple(sorted(results, key=lambda r: (r.divergence_type, r.case_id, r.result_hash)))
    status = "EXTRACTION_REPLAY_DIVERGENCE_DETECTED" if tup else "EXTRACTION_REPLAY_VALIDATED"
    rec = ExtractionReplayReceipt(SCHEMA_VERSION, baseline_evidence.evidence_hash, observed_evidence.evidence_hash, baseline_evidence.raw_bytes_hash, tup, len(tup), sum(1 for r in tup if r.severity == "REJECT"), sum(1 for r in tup if r.severity == "FLAG"), status, "")
    return ExtractionReplayReceipt(rec.version, rec.baseline_evidence_hash, rec.observed_evidence_hash, rec.raw_bytes_hash, rec.results, rec.result_count, rec.reject_count, rec.flag_count, rec.status, rec.computed_stable_hash())


def run_resonance_replay_validation(baseline_evidence: ResonanceReplayEvidence, observed_evidence: ResonanceReplayEvidence) -> ResonanceReplayReceipt:
    if not isinstance(baseline_evidence, ResonanceReplayEvidence) or not isinstance(observed_evidence, ResonanceReplayEvidence): raise _invalid()
    for k in ("canonical_hash","res_rag_mapping_hash","governance_context_hash","resonance_classifier_hash","tolerance_hash"):
        if getattr(baseline_evidence,k)!=getattr(observed_evidence,k): raise _invalid()
    results=[]
    if baseline_evidence.semantic_field_hash!=observed_evidence.semantic_field_hash: results.append(_mk_result("resonance_replay","SEMANTIC_FIELD_DRIFT",baseline_evidence.semantic_field_hash,observed_evidence.semantic_field_hash,"SEMANTIC_FIELD_HASH_CHANGED","REJECT"))
    if baseline_evidence.res_hash!=observed_evidence.res_hash: results.append(_mk_result("resonance_replay","RES_STATE_DRIFT",baseline_evidence.res_hash,observed_evidence.res_hash,"RES_HASH_CHANGED","REJECT"))
    if baseline_evidence.rag_hash!=observed_evidence.rag_hash: results.append(_mk_result("resonance_replay","RAG_STATE_DRIFT",baseline_evidence.rag_hash,observed_evidence.rag_hash,"RAG_HASH_CHANGED","REJECT"))
    if baseline_evidence.aggregate_resonance_class!=observed_evidence.aggregate_resonance_class: results.append(_mk_result("resonance_replay","RESONANCE_CLASSIFICATION_DRIFT",_sha(baseline_evidence.aggregate_resonance_class),_sha(observed_evidence.aggregate_resonance_class),"RESONANCE_CLASS_CHANGED","REJECT"))
    if baseline_evidence.resonance_receipt_hash!=observed_evidence.resonance_receipt_hash: results.append(_mk_result("resonance_replay","RESONANCE_CLASSIFICATION_DRIFT",baseline_evidence.resonance_receipt_hash,observed_evidence.resonance_receipt_hash,"RESONANCE_RECEIPT_HASH_CHANGED","REJECT"))
    tup=tuple(sorted(results,key=lambda r:(r.divergence_type,r.case_id,r.result_hash)))
    status="RESONANCE_REPLAY_DIVERGENCE_DETECTED" if tup else "RESONANCE_REPLAY_VALIDATED"
    rec=ResonanceReplayReceipt(SCHEMA_VERSION,baseline_evidence.evidence_hash,observed_evidence.evidence_hash,baseline_evidence.canonical_hash,tup,len(tup),sum(1 for r in tup if r.severity=="REJECT"),sum(1 for r in tup if r.severity=="FLAG"),status,"")
    return ResonanceReplayReceipt(rec.version,rec.baseline_evidence_hash,rec.observed_evidence_hash,rec.canonical_hash,rec.results,rec.result_count,rec.reject_count,rec.flag_count,rec.status,rec.computed_stable_hash())


def run_real_world_replay_proof(baseline_evidence: RealWorldReplayEvidence, observed_evidence: RealWorldReplayEvidence, extraction_replay_receipt: ExtractionReplayReceipt, resonance_replay_receipt: ResonanceReplayReceipt) -> RealWorldReplayProofReceipt:
    if not isinstance(baseline_evidence, RealWorldReplayEvidence) or not isinstance(observed_evidence, RealWorldReplayEvidence) or not isinstance(extraction_replay_receipt, ExtractionReplayReceipt) or not isinstance(resonance_replay_receipt, ResonanceReplayReceipt): raise _invalid()
    if baseline_evidence.canonical_hash!=observed_evidence.canonical_hash: raise _invalid()
    if extraction_replay_receipt.stable_hash!=extraction_replay_receipt.computed_stable_hash() or resonance_replay_receipt.stable_hash!=resonance_replay_receipt.computed_stable_hash(): raise _invalid()
    if baseline_evidence.canonical_hash!=resonance_replay_receipt.canonical_hash: raise _invalid()
    results=[]
    if baseline_evidence.semantic_field_hash!=observed_evidence.semantic_field_hash: results.append(_mk_result("real_world_replay","SEMANTIC_FIELD_DRIFT",baseline_evidence.semantic_field_hash,observed_evidence.semantic_field_hash,"SEMANTIC_FIELD_HASH_CHANGED","REJECT"))
    if baseline_evidence.validation_hash!=observed_evidence.validation_hash: results.append(_mk_result("real_world_replay","VALIDATION_DRIFT",baseline_evidence.validation_hash,observed_evidence.validation_hash,"VALIDATION_HASH_CHANGED","REJECT"))
    if baseline_evidence.governance_hash!=observed_evidence.governance_hash: results.append(_mk_result("real_world_replay","GOVERNANCE_DIVERGENCE",baseline_evidence.governance_hash,observed_evidence.governance_hash,"GOVERNANCE_HASH_CHANGED","REJECT"))
    if baseline_evidence.governance_decision!=observed_evidence.governance_decision: results.append(_mk_result("real_world_replay","GOVERNANCE_DIVERGENCE",_sha(baseline_evidence.governance_decision),_sha(observed_evidence.governance_decision),"GOVERNANCE_DECISION_CHANGED","REJECT"))
    if baseline_evidence.local_proof_hash!=observed_evidence.local_proof_hash: results.append(_mk_result("real_world_replay","PROOF_CHAIN_DRIFT",baseline_evidence.local_proof_hash,observed_evidence.local_proof_hash,"LOCAL_PROOF_HASH_CHANGED","REJECT"))
    if baseline_evidence.distributed_convergence_hash!=observed_evidence.distributed_convergence_hash: results.append(_mk_result("real_world_replay","PROOF_CHAIN_DRIFT",baseline_evidence.distributed_convergence_hash,observed_evidence.distributed_convergence_hash,"DISTRIBUTED_CONVERGENCE_HASH_CHANGED","REJECT"))
    if baseline_evidence.final_proof_hash!=observed_evidence.final_proof_hash: results.append(_mk_result("real_world_replay","FINAL_PROOF_DRIFT",baseline_evidence.final_proof_hash,observed_evidence.final_proof_hash,"FINAL_PROOF_HASH_CHANGED","REJECT"))
    if baseline_evidence.resonance_receipt_hash!=observed_evidence.resonance_receipt_hash: results.append(_mk_result("real_world_replay","RESONANCE_CLASSIFICATION_DRIFT",baseline_evidence.resonance_receipt_hash,observed_evidence.resonance_receipt_hash,"RESONANCE_RECEIPT_HASH_CHANGED","REJECT"))
    tup=tuple(sorted(results,key=lambda r:(r.divergence_type,r.case_id,r.result_hash)))
    status="REAL_WORLD_REPLAY_DIVERGENCE_DETECTED" if tup else "REAL_WORLD_REPLAY_VALIDATED"
    rec=RealWorldReplayProofReceipt(SCHEMA_VERSION,baseline_evidence.evidence_hash,observed_evidence.evidence_hash,baseline_evidence.canonical_hash,baseline_evidence.semantic_field_hash,extraction_replay_receipt.stable_hash,resonance_replay_receipt.stable_hash,tup,len(tup),sum(1 for r in tup if r.severity=="REJECT"),sum(1 for r in tup if r.severity=="FLAG"),status,"")
    return RealWorldReplayProofReceipt(rec.version,rec.baseline_evidence_hash,rec.observed_evidence_hash,rec.canonical_hash,rec.semantic_field_hash,rec.extraction_replay_hash,rec.resonance_replay_hash,rec.results,rec.result_count,rec.reject_count,rec.flag_count,rec.status,rec.computed_stable_hash())
