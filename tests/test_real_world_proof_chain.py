from __future__ import annotations

import dataclasses
import json

import pytest

from qec.analysis.adversarial_extraction_validation import ExtractionValidationRule, run_adversarial_extraction_validation
from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.canonicalization_engine import CanonicalDocument, CanonicalizationReceipt
from qec.analysis.dialogical_document_governance import run_dialogical_document_governance
from qec.analysis.distributed_convergence_proof import run_distributed_convergence_proof
from qec.analysis.extraction_boundary import ExtractionReceipt
from qec.analysis.real_world_proof_chain import (
    LocalRealWorldProof,
    RealWorldProofReceipt,
    build_local_real_world_proof,
    build_real_world_distributed_evidence,
    run_real_world_proof_chain,
    to_distributed_node_convergence_evidence,
)
from qec.analysis.res_rag_resonance_validation import run_res_rag_resonance_validation
from qec.analysis.res_rag_semantic_field import GeneratedClaim, GovernanceContext, run_res_rag_semantic_field


def _doc(payload: dict[str, object]) -> CanonicalDocument:
    return CanonicalDocument("v151.1", "1" * 64, "2" * 64, "3" * 64, payload, json.dumps(payload, sort_keys=True, separators=(",", ":")), sha256_hex(payload))


def _claim(cid: str, payload: object) -> GeneratedClaim:
    return GeneratedClaim(cid, f"claim-{cid}", payload, sha256_hex({"claim_id": cid, "claim_text": f"claim-{cid}", "claim_payload": payload}))


def _ctx() -> GovernanceContext:
    p = {"mode": "strict"}
    return GovernanceContext("ctx", p, sha256_hex({"context_id": "ctx", "schema_version": "v151.3", "context_payload": p, "allowed_keys": ("mode",)}))


def _build_receipts():
    doc = _doc({"x": 1})
    sf = run_res_rag_semantic_field(doc, [_claim("1", {"claim_type": "FIELD_PRESENT", "field_name": "x"})], _ctx())
    from qec.analysis.res_rag_semantic_field import EvidenceField, RESState, RAGState, SourceConstraint
    evidence = tuple(sorted((EvidenceField(k, v, sha256_hex({"field_name": k, "canonical_value": v})) for k, v in doc.canonical_payload.items()), key=lambda e: (e.field_name, e.value_hash)))
    fsh = sha256_hex(tuple(sorted(e.field_name for e in evidence)))
    constraints = tuple(sorted([
        SourceConstraint("CANONICAL_DOCUMENT_HASH", doc.canonical_hash, sha256_hex({"constraint_type": "CANONICAL_DOCUMENT_HASH", "constraint_value": doc.canonical_hash})),
        SourceConstraint("CANONICAL_SCHEMA_HASH", doc.schema_hash, sha256_hex({"constraint_type": "CANONICAL_SCHEMA_HASH", "constraint_value": doc.schema_hash})),
        SourceConstraint("CANONICAL_LOCALE_HASH", doc.locale_hash, sha256_hex({"constraint_type": "CANONICAL_LOCALE_HASH", "constraint_value": doc.locale_hash})),
        SourceConstraint("CANONICAL_EXTRACTION_HASH", doc.extraction_hash, sha256_hex({"constraint_type": "CANONICAL_EXTRACTION_HASH", "constraint_value": doc.extraction_hash})),
        SourceConstraint("FIELD_SET_HASH", fsh, sha256_hex({"constraint_type": "FIELD_SET_HASH", "constraint_value": fsh})),
    ], key=lambda c: (c.constraint_type, c.constraint_hash)))
    gfh = sha256_hex(tuple(e.to_dict() for e in evidence))
    res = RESState("v151.2", doc.canonical_hash, gfh, evidence, constraints, sha256_hex({"version": "v151.2", "canonical_document_hash": doc.canonical_hash, "grounded_field_hash": gfh, "evidence_fields": tuple(e.to_dict() for e in evidence), "source_constraints": tuple(c.to_dict() for c in constraints)}))
    gc = tuple(sorted([_claim("1", {"claim_type": "FIELD_PRESENT", "field_name": "x"})], key=lambda c: (c.claim_id, c.claim_hash)))
    gh = _ctx().governance_context_hash
    ih = sha256_hex({"generated_claims": tuple(c.to_dict() for c in gc), "governance_context_hash": gh})
    rag = RAGState("v151.2", doc.canonical_hash, ih, gc, gh, sha256_hex({"version": "v151.2", "canonical_document_hash": doc.canonical_hash, "interpretation_hash": ih, "generated_claims": tuple(c.to_dict() for c in gc), "governance_context_hash": gh}))
    rr = run_res_rag_resonance_validation(sf, res, rag)
    evr = run_adversarial_extraction_validation(doc, sf, rr, [])
    gov = run_dialogical_document_governance(doc, res, rag, rr, evr)
    exp = {"version": "v151.0", "raw_bytes_hash": "1" * 64, "extraction_config_hash": "a" * 64, "input_hash": "b" * 64, "config_hash": "c" * 64, "extraction_hash": doc.extraction_hash, "query_fields": ("x",), "determinism_status": "CONSISTENT"}
    extraction = ExtractionReceipt(**exp, stable_hash=sha256_hex(exp))
    can_payload = {"version": "v151.1", "extraction_hash": extraction.extraction_hash, "schema_hash": doc.schema_hash, "locale_hash": doc.locale_hash, "canonical_hash": doc.canonical_hash, "canonical_document": doc, "status": "CANONICALIZED"}
    can = CanonicalizationReceipt(**can_payload, stable_hash=sha256_hex({**can_payload, "canonical_document": doc.to_dict()}))
    return extraction, can, sf, rr, evr, gov


def test_local_chain_and_mapping_and_final_receipt() -> None:
    ex, can, sf, rr, evr, gov = _build_receipts()
    local = build_local_real_world_proof(ex, can, sf, rr, evr, gov)
    assert isinstance(local, LocalRealWorldProof)
    assert tuple(x.link_name for x in local.proof_links) == ("EXTRACTION", "CANONICALIZATION", "SEMANTIC_FIELD", "RESONANCE_VALIDATION", "EXTRACTION_VALIDATION", "DIALOGICAL_GOVERNANCE")
    assert local.local_proof_hash == local.computed_stable_hash()

    e = build_real_world_distributed_evidence("n1", "CONTROL", local)
    n = to_distributed_node_convergence_evidence(e)
    assert n.final_proof_hash == local.local_proof_hash
    assert n.metadata["semantic_field_hash"] == local.semantic_field_hash
    assert n.metadata["source"] == "v151.6_real_world_proof_chain"

    dcr = run_distributed_convergence_proof("scn", (n,), expected_final_proof_hash=local.local_proof_hash)
    out = run_real_world_proof_chain(ex, can, sf, rr, evr, gov, dcr)
    assert isinstance(out, RealWorldProofReceipt)
    assert out.status == "REAL_WORLD_PROOF_VALIDATED"
    assert out.final_proof_hash == sha256_hex({"local_proof_hash": out.local_proof_hash, "distributed_convergence_hash": out.distributed_convergence_hash, "status": out.status})


def test_mismatch_and_invalid_paths_and_determinism() -> None:
    ex, can, sf, rr, evr, gov = _build_receipts()
    local = build_local_real_world_proof(ex, can, sf, rr, evr, gov)
    # Use two nodes with local.local_proof_hash so it becomes the reference (majority).
    n0 = to_distributed_node_convergence_evidence(build_real_world_distributed_evidence("n0", "CONTROL", local))
    n1 = to_distributed_node_convergence_evidence(build_real_world_distributed_evidence("n1", "CONTROL", local))
    other_hash = sha256_hex({"x": "other"})
    from qec.analysis.distributed_convergence_proof import DistributedNodeConvergenceEvidence
    p = {"node_id": "n2", "node_role": "CONTROL", "convergence_hash": local.resonance_hash, "governance_hash": local.governance_hash, "adversarial_hash": local.validation_hash, "final_proof_hash": other_hash, "metadata": {"semantic_field_hash": local.semantic_field_hash, "distributed_evidence_hash": sha256_hex({"x": 1}), "source": "v151.6_real_world_proof_chain"}}
    n2 = DistributedNodeConvergenceEvidence(**p, evidence_hash=sha256_hex(p))
    dcr = run_distributed_convergence_proof("scn2", (n0, n1, n2))
    assert dcr.reference_final_proof_hash == local.local_proof_hash
    out = run_real_world_proof_chain(ex, can, sf, rr, evr, gov, dcr)
    assert out.status == "DISTRIBUTED_CONVERGENCE_MISMATCH"

    out2 = run_real_world_proof_chain(ex, can, sf, rr, evr, gov, dcr)
    assert out.stable_hash == out2.stable_hash

    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        run_real_world_proof_chain(dataclasses.replace(ex, extraction_hash="f" * 64), can, sf, rr, evr, gov, dcr)
    with pytest.raises(dataclasses.FrozenInstanceError):
        out.status = "X"  # type: ignore[misc]
    json.dumps(out.to_dict())


def test_mismatch_receipt_from_unrelated_proof_is_rejected() -> None:
    ex, can, sf, rr, evr, gov = _build_receipts()
    local = build_local_real_world_proof(ex, can, sf, rr, evr, gov)
    # Build a distributed convergence receipt for a completely different proof hash.
    unrelated_hash = sha256_hex({"proof": "unrelated"})
    from qec.analysis.distributed_convergence_proof import DistributedNodeConvergenceEvidence
    p = {"node_id": "n1", "node_role": "CONTROL", "convergence_hash": local.resonance_hash, "governance_hash": local.governance_hash, "adversarial_hash": local.validation_hash, "final_proof_hash": unrelated_hash, "metadata": {}}
    n_unrelated = DistributedNodeConvergenceEvidence(**p, evidence_hash=sha256_hex(p))
    dcr_unrelated = run_distributed_convergence_proof("scn_unrelated", (n_unrelated,), expected_final_proof_hash=unrelated_hash)
    # That receipt is VALIDATED but references the wrong local proof — must be rejected.
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        run_real_world_proof_chain(ex, can, sf, rr, evr, gov, dcr_unrelated)


def test_node_id_role_whitespace_and_bool_rejected() -> None:
    ex, can, sf, rr, evr, gov = _build_receipts()
    local = build_local_real_world_proof(ex, can, sf, rr, evr, gov)
    for bad in (" n1", "n1 ", " ", True, False):
        with pytest.raises(ValueError, match="^INVALID_INPUT$"):
            build_real_world_distributed_evidence(bad, "CONTROL", local)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="^INVALID_INPUT$"):
            build_real_world_distributed_evidence("n1", bad, local)  # type: ignore[arg-type]


def test_proof_receipt_rejects_non_link_elements() -> None:
    ex, can, sf, rr, evr, gov = _build_receipts()
    local = build_local_real_world_proof(ex, can, sf, rr, evr, gov)
    n1 = to_distributed_node_convergence_evidence(build_real_world_distributed_evidence("n1", "CONTROL", local))
    dcr = run_distributed_convergence_proof("scn", (n1,), expected_final_proof_hash=local.local_proof_hash)
    good = run_real_world_proof_chain(ex, can, sf, rr, evr, gov, dcr)
    # Replace one proof link with a plain dict — must raise INVALID_INPUT, not AttributeError.
    bad_links = good.proof_links[:6] + ({"link_name": "DISTRIBUTED_CONVERGENCE"},)  # type: ignore[assignment]
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        dataclasses.replace(good, proof_links=bad_links)
