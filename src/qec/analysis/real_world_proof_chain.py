from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qec.analysis.adversarial_extraction_validation import ExtractionValidationReceipt
from qec.analysis.canonical_hashing import CanonicalHashingError, canonical_json, sha256_hex
from qec.analysis.canonicalization_engine import CanonicalizationReceipt
from qec.analysis.dialogical_document_governance import DialogicalGovernanceReceipt
from qec.analysis.distributed_convergence_proof import (
    DISTRIBUTED_CONVERGENCE_MISMATCH,
    VALIDATED,
    DistributedConvergenceReceipt,
    DistributedNodeConvergenceEvidence,
)
from qec.analysis.extraction_boundary import ExtractionReceipt
from qec.analysis.res_rag_resonance_validation import ResonanceValidationReceipt
from qec.analysis.res_rag_semantic_field import SemanticFieldReceipt

_VERSION = "v151.6"

_LINKS_ALL = (
    "EXTRACTION",
    "CANONICALIZATION",
    "SEMANTIC_FIELD",
    "RESONANCE_VALIDATION",
    "EXTRACTION_VALIDATION",
    "DIALOGICAL_GOVERNANCE",
    "DISTRIBUTED_CONVERGENCE",
)


def _invalid() -> ValueError:
    return ValueError("INVALID_INPUT")


def _sha(value: Any) -> str:
    try:
        return sha256_hex(value)
    except CanonicalHashingError as exc:
        raise _invalid() from exc


def _canonical_json(value: Any) -> str:
    try:
        return canonical_json(value)
    except CanonicalHashingError as exc:
        raise _invalid() from exc


def _is_sha(v: Any) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


def _require_str(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, str) or value.strip() != value or not value:
        raise _invalid()
    return value


@dataclass(frozen=True)
class ProofChainLink:
    link_name: str
    receipt_type: str
    receipt_hash: str
    lineage_hashes: tuple[str, ...]
    link_hash: str

    def __post_init__(self) -> None:
        if self.link_name not in _LINKS_ALL:
            raise _invalid()
        if not isinstance(self.receipt_type, str) or self.receipt_type == "":
            raise _invalid()
        if not _is_sha(self.receipt_hash):
            raise _invalid()
        if not isinstance(self.lineage_hashes, tuple) or any(not _is_sha(h) for h in self.lineage_hashes):
            raise _invalid()
        if self.link_name != "EXTRACTION" and not self.lineage_hashes:
            raise _invalid()
        if self.computed_stable_hash() != self.link_hash:
            raise _invalid()

    def to_dict(self) -> dict[str, Any]:
        return {"link_name": self.link_name, "receipt_type": self.receipt_type, "receipt_hash": self.receipt_hash, "lineage_hashes": self.lineage_hashes, "link_hash": self.link_hash}
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")
    def computed_stable_hash(self) -> str: return _sha({k: v for k, v in self.to_dict().items() if k != "link_hash"})


@dataclass(frozen=True)
class LocalRealWorldProof:
    version: str; raw_bytes_hash: str; extraction_hash: str; canonical_hash: str; res_hash: str; rag_hash: str; semantic_field_hash: str; resonance_hash: str; validation_hash: str; governance_hash: str; proof_links: tuple[ProofChainLink, ...]; local_proof_hash: str
    def __post_init__(self) -> None:
        if self.version != _VERSION: raise _invalid()
        for h in (self.raw_bytes_hash, self.extraction_hash, self.canonical_hash, self.res_hash, self.rag_hash, self.semantic_field_hash, self.resonance_hash, self.validation_hash, self.governance_hash):
            if not _is_sha(h): raise _invalid()
        if not isinstance(self.proof_links, tuple) or len(self.proof_links) != 6 or any(not isinstance(p, ProofChainLink) for p in self.proof_links): raise _invalid()
        if tuple(p.link_name for p in self.proof_links) != _LINKS_ALL[:6]: raise _invalid()
        if self.computed_stable_hash() != self.local_proof_hash: raise _invalid()
    def to_dict(self) -> dict[str, Any]: return {"version": self.version, "raw_bytes_hash": self.raw_bytes_hash, "extraction_hash": self.extraction_hash, "canonical_hash": self.canonical_hash, "res_hash": self.res_hash, "rag_hash": self.rag_hash, "semantic_field_hash": self.semantic_field_hash, "resonance_hash": self.resonance_hash, "validation_hash": self.validation_hash, "governance_hash": self.governance_hash, "proof_links": tuple(p.to_dict() for p in self.proof_links), "local_proof_hash": self.local_proof_hash}
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")
    def computed_stable_hash(self) -> str: return _sha({k: v for k, v in self.to_dict().items() if k != "local_proof_hash"})


@dataclass(frozen=True)
class RealWorldDistributedEvidence:
    node_id: str; node_role: str; local_proof_hash: str; governance_hash: str; validation_hash: str; resonance_hash: str; semantic_field_hash: str; distributed_evidence_hash: str
    def __post_init__(self) -> None:
        _require_str(self.node_id)
        _require_str(self.node_role)
        for h in (self.local_proof_hash, self.governance_hash, self.validation_hash, self.resonance_hash, self.semantic_field_hash):
            if not _is_sha(h): raise _invalid()
        if self.computed_stable_hash() != self.distributed_evidence_hash: raise _invalid()
    def to_dict(self) -> dict[str, Any]: return {"node_id": self.node_id, "node_role": self.node_role, "local_proof_hash": self.local_proof_hash, "governance_hash": self.governance_hash, "validation_hash": self.validation_hash, "resonance_hash": self.resonance_hash, "semantic_field_hash": self.semantic_field_hash, "distributed_evidence_hash": self.distributed_evidence_hash}
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")
    def computed_stable_hash(self) -> str: return _sha({k: v for k, v in self.to_dict().items() if k != "distributed_evidence_hash"})


@dataclass(frozen=True)
class RealWorldProofReceipt:
    version: str; raw_bytes_hash: str; extraction_hash: str; canonical_hash: str; res_hash: str; rag_hash: str; semantic_field_hash: str; resonance_hash: str; validation_hash: str; governance_hash: str; local_proof_hash: str; distributed_convergence_hash: str; final_proof_hash: str; proof_links: tuple[ProofChainLink, ...]; status: str; stable_hash: str
    def __post_init__(self) -> None:
        if self.version != _VERSION or self.status not in ("REAL_WORLD_PROOF_VALIDATED", "DISTRIBUTED_CONVERGENCE_MISMATCH"): raise _invalid()
        for h in (self.raw_bytes_hash, self.extraction_hash, self.canonical_hash, self.res_hash, self.rag_hash, self.semantic_field_hash, self.resonance_hash, self.validation_hash, self.governance_hash, self.local_proof_hash, self.distributed_convergence_hash, self.final_proof_hash, self.stable_hash):
            if not _is_sha(h): raise _invalid()
        if not isinstance(self.proof_links, tuple) or len(self.proof_links) != 7 or any(not isinstance(p, ProofChainLink) for p in self.proof_links): raise _invalid()
        if tuple(p.link_name for p in self.proof_links) != _LINKS_ALL: raise _invalid()
        if self.final_proof_hash != _sha({"local_proof_hash": self.local_proof_hash, "distributed_convergence_hash": self.distributed_convergence_hash, "status": self.status}): raise _invalid()
        if self.computed_stable_hash() != self.stable_hash: raise _invalid()
    def to_dict(self) -> dict[str, Any]: return {"version": self.version, "raw_bytes_hash": self.raw_bytes_hash, "extraction_hash": self.extraction_hash, "canonical_hash": self.canonical_hash, "res_hash": self.res_hash, "rag_hash": self.rag_hash, "semantic_field_hash": self.semantic_field_hash, "resonance_hash": self.resonance_hash, "validation_hash": self.validation_hash, "governance_hash": self.governance_hash, "local_proof_hash": self.local_proof_hash, "distributed_convergence_hash": self.distributed_convergence_hash, "final_proof_hash": self.final_proof_hash, "proof_links": tuple(p.to_dict() for p in self.proof_links), "status": self.status, "stable_hash": self.stable_hash}
    def to_canonical_json(self) -> str: return _canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes: return self.to_canonical_json().encode("utf-8")
    def computed_stable_hash(self) -> str: return _sha({k: v for k, v in self.to_dict().items() if k != "stable_hash"})


def _link(link_name: str, receipt_type: str, receipt_hash: str, lineage_hashes: tuple[str, ...]) -> ProofChainLink:
    payload = {"link_name": link_name, "receipt_type": receipt_type, "receipt_hash": receipt_hash, "lineage_hashes": lineage_hashes}
    return ProofChainLink(**payload, link_hash=_sha(payload))


def build_local_real_world_proof(extraction_receipt: ExtractionReceipt, canonicalization_receipt: CanonicalizationReceipt, semantic_field_receipt: SemanticFieldReceipt, resonance_receipt: ResonanceValidationReceipt, extraction_validation_receipt: ExtractionValidationReceipt, governance_receipt: DialogicalGovernanceReceipt) -> LocalRealWorldProof:
    if not isinstance(extraction_receipt, ExtractionReceipt) or not isinstance(canonicalization_receipt, CanonicalizationReceipt) or not isinstance(semantic_field_receipt, SemanticFieldReceipt) or not isinstance(resonance_receipt, ResonanceValidationReceipt) or not isinstance(extraction_validation_receipt, ExtractionValidationReceipt) or not isinstance(governance_receipt, DialogicalGovernanceReceipt):
        raise _invalid()
    if extraction_receipt.computed_stable_hash() != extraction_receipt.stable_hash or canonicalization_receipt.computed_stable_hash() != canonicalization_receipt.stable_hash or semantic_field_receipt.computed_stable_hash() != semantic_field_receipt.stable_hash or resonance_receipt.computed_stable_hash() != resonance_receipt.stable_hash or extraction_validation_receipt.computed_stable_hash() != extraction_validation_receipt.stable_hash or governance_receipt.computed_stable_hash() != governance_receipt.stable_hash:
        raise _invalid()
    if extraction_receipt.extraction_hash != canonicalization_receipt.extraction_hash: raise _invalid()
    if canonicalization_receipt.canonical_hash != semantic_field_receipt.canonical_hash: raise _invalid()
    if semantic_field_receipt.semantic_field_hash != resonance_receipt.semantic_field_hash: raise _invalid()
    if resonance_receipt.stable_hash != extraction_validation_receipt.resonance_receipt_hash: raise _invalid()
    if extraction_validation_receipt.stable_hash != governance_receipt.extraction_validation_hash: raise _invalid()
    if governance_receipt.canonical_hash != canonicalization_receipt.canonical_hash: raise _invalid()

    links = (
        _link("EXTRACTION", "ExtractionReceipt", extraction_receipt.stable_hash, tuple()),
        _link("CANONICALIZATION", "CanonicalizationReceipt", canonicalization_receipt.stable_hash, (extraction_receipt.extraction_hash,)),
        _link("SEMANTIC_FIELD", "SemanticFieldReceipt", semantic_field_receipt.stable_hash, (canonicalization_receipt.canonical_hash, semantic_field_receipt.res_hash, semantic_field_receipt.rag_hash)),
        _link("RESONANCE_VALIDATION", "ResonanceValidationReceipt", resonance_receipt.stable_hash, (semantic_field_receipt.semantic_field_hash,)),
        _link("EXTRACTION_VALIDATION", "ExtractionValidationReceipt", extraction_validation_receipt.stable_hash, (resonance_receipt.stable_hash,)),
        _link("DIALOGICAL_GOVERNANCE", "DialogicalGovernanceReceipt", governance_receipt.stable_hash, (extraction_validation_receipt.stable_hash,)),
    )
    payload = {"version": _VERSION, "raw_bytes_hash": extraction_receipt.raw_bytes_hash, "extraction_hash": extraction_receipt.extraction_hash, "canonical_hash": canonicalization_receipt.canonical_hash, "res_hash": semantic_field_receipt.res_hash, "rag_hash": semantic_field_receipt.rag_hash, "semantic_field_hash": semantic_field_receipt.semantic_field_hash, "resonance_hash": resonance_receipt.stable_hash, "validation_hash": extraction_validation_receipt.stable_hash, "governance_hash": governance_receipt.stable_hash, "proof_links": links}
    return LocalRealWorldProof(**payload, local_proof_hash=_sha({**payload, "proof_links": tuple(p.to_dict() for p in links)}))


def build_real_world_distributed_evidence(node_id: str, node_role: str, local_proof: LocalRealWorldProof) -> RealWorldDistributedEvidence:
    if not isinstance(local_proof, LocalRealWorldProof): raise _invalid()
    _require_str(node_id)
    _require_str(node_role)
    if local_proof.computed_stable_hash() != local_proof.local_proof_hash: raise _invalid()
    payload = {"node_id": node_id, "node_role": node_role, "local_proof_hash": local_proof.local_proof_hash, "governance_hash": local_proof.governance_hash, "validation_hash": local_proof.validation_hash, "resonance_hash": local_proof.resonance_hash, "semantic_field_hash": local_proof.semantic_field_hash}
    return RealWorldDistributedEvidence(**payload, distributed_evidence_hash=_sha(payload))


def to_distributed_node_convergence_evidence(evidence: RealWorldDistributedEvidence) -> DistributedNodeConvergenceEvidence:
    if not isinstance(evidence, RealWorldDistributedEvidence) or evidence.computed_stable_hash() != evidence.distributed_evidence_hash: raise _invalid()
    payload = {"node_id": evidence.node_id, "node_role": evidence.node_role, "convergence_hash": evidence.resonance_hash, "governance_hash": evidence.governance_hash, "adversarial_hash": evidence.validation_hash, "final_proof_hash": evidence.local_proof_hash, "metadata": {"semantic_field_hash": evidence.semantic_field_hash, "distributed_evidence_hash": evidence.distributed_evidence_hash, "source": "v151.6_real_world_proof_chain"}}
    return DistributedNodeConvergenceEvidence(**payload, evidence_hash=_sha(payload))


def run_real_world_proof_chain(extraction_receipt: ExtractionReceipt, canonicalization_receipt: CanonicalizationReceipt, semantic_field_receipt: SemanticFieldReceipt, resonance_receipt: ResonanceValidationReceipt, extraction_validation_receipt: ExtractionValidationReceipt, governance_receipt: DialogicalGovernanceReceipt, distributed_convergence_receipt: DistributedConvergenceReceipt) -> RealWorldProofReceipt:
    local = build_local_real_world_proof(extraction_receipt, canonicalization_receipt, semantic_field_receipt, resonance_receipt, extraction_validation_receipt, governance_receipt)
    if not isinstance(distributed_convergence_receipt, DistributedConvergenceReceipt) or distributed_convergence_receipt.computed_stable_hash() != distributed_convergence_receipt.stable_hash:
        raise _invalid()
    if distributed_convergence_receipt.status == VALIDATED:
        if distributed_convergence_receipt.reference_final_proof_hash != local.local_proof_hash:
            raise _invalid()
        status = "REAL_WORLD_PROOF_VALIDATED"
    elif distributed_convergence_receipt.status == DISTRIBUTED_CONVERGENCE_MISMATCH:
        if distributed_convergence_receipt.reference_final_proof_hash != local.local_proof_hash:
            raise _invalid()
        status = "DISTRIBUTED_CONVERGENCE_MISMATCH"
    else:
        raise _invalid()
    links = local.proof_links + (_link("DISTRIBUTED_CONVERGENCE", "DistributedConvergenceReceipt", distributed_convergence_receipt.stable_hash, (local.local_proof_hash, distributed_convergence_receipt.reference_final_proof_hash)),)
    final_proof_hash = _sha({"local_proof_hash": local.local_proof_hash, "distributed_convergence_hash": distributed_convergence_receipt.stable_hash, "status": status})
    payload = {"version": _VERSION, "raw_bytes_hash": local.raw_bytes_hash, "extraction_hash": local.extraction_hash, "canonical_hash": local.canonical_hash, "res_hash": local.res_hash, "rag_hash": local.rag_hash, "semantic_field_hash": local.semantic_field_hash, "resonance_hash": local.resonance_hash, "validation_hash": local.validation_hash, "governance_hash": local.governance_hash, "local_proof_hash": local.local_proof_hash, "distributed_convergence_hash": distributed_convergence_receipt.stable_hash, "final_proof_hash": final_proof_hash, "proof_links": links, "status": status}
    return RealWorldProofReceipt(**payload, stable_hash=_sha({**payload, "proof_links": tuple(p.to_dict() for p in links)}))
