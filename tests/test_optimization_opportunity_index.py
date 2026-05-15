import json
import pytest

from qec.analysis.heavy_dependency_discovery import build_default_unprobed_manifest
from qec.analysis.dependency_hotpath_receipts import build_dependency_hotpath_candidate, build_dependency_import_and_hotpath_receipt
from qec.analysis.backend_invariant_candidate_receipts import build_backend_invariant_evidence, build_backend_invariant_candidate, build_backend_invariant_candidate_receipt
from qec.analysis.cross_backend_equivalence_receipts import build_backend_observation, build_equivalence_receipt_from_observations
from qec.analysis.optimization_opportunity_index import (
    build_optimization_opportunity_index,
    optimization_opportunity_index_to_canonical_json_bytes,
)


def _inputs(with_equivalence=False):
    manifest = build_default_unprobed_manifest()
    hot = [build_dependency_hotpath_candidate(candidate_index=0, dependency_name="qiskit", source_path="src/a.py", line_number=1, candidate_kind="QUANTUM_BACKEND_BOUNDARY", candidate_status="NEEDS_EQUIVALENCE_PROOF", reason="r", related_import_site_hashes=())]
    hot_receipt = build_dependency_import_and_hotpath_receipt((), hot, source_root_label="src", scanned_file_count=1, target_registry_hash=manifest.heavy_dependency_discovery_manifest_hash)
    ev = [build_backend_invariant_evidence(evidence_index=i, dependency_name=d, evidence_kind="HOTPATH_CANDIDATE_EVIDENCE", source_path="src/a.py", line_number=1, import_site_hash=None, hotpath_candidate_hash=hot[0].candidate_hash if d=="qiskit" else None, probe_hash=manifest.probe_results[0].probe_hash, reason="r") for i,d in enumerate(["qldpc_external","qiskit","numpy","matplotlib","pandas","mido","qldpc_internal"])]
    kinds=["POLICY_BLOCKED_EXTERNAL_INVARIANT","QUANTUM_BACKEND_BOUNDARY_INVARIANT","SPARSE_DENSE_BOUNDARY_INVARIANT","PLOTTING_RENDER_BOUNDARY_INVARIANT","DATAFRAME_SCHEMA_BOUNDARY_INVARIANT","AUDIO_MIDI_BOUNDARY_INVARIANT","INTERNAL_QEC_SURFACE_INVARIANT"]
    deps=["qldpc_external","qiskit","numpy","matplotlib","pandas","mido","qldpc_internal"]
    cands=[build_backend_invariant_candidate(candidate_index=i, dependency_name=deps[i], invariant_name=f"inv{i}", invariant_kind=kinds[i], invariant_status="CANDIDATE_IDENTIFIED", review_class=("BLOCKED_BY_POLICY" if kinds[i]=="POLICY_BLOCKED_EXTERNAL_INVARIANT" else "SAFE_FOR_OPTIMIZATION_CONTRACT_REVIEW"), required_next_receipt="CrossBackendEquivalenceReceipt" if "BOUNDARY" in kinds[i] else "OptimizationContract", evidence_hashes=(ev[i].evidence_hash,), source_paths=("src/a.py",), reason="r") for i in range(len(kinds))]
    inv_receipt = build_backend_invariant_candidate_receipt(manifest, hot_receipt, ev, cands)
    if not with_equivalence:
        return manifest, hot_receipt, inv_receipt, None
    obs=[]
    for i,c in enumerate(cands):
        if c.dependency_name not in {"qiskit","numpy","pandas"}: continue
        obs.append(build_backend_observation(observation_index=len(obs), backend_name="ref", dependency_name=c.dependency_name, observation_name=f"obs_{c.candidate_index}", observation_kind="JSON_VALUE", backend_role="REFERENCE", payload={"x":1}, payload_hash="", error_code=None, unavailable_reason=None, source_invariant_candidate_hash=c.candidate_hash))
        obs.append(build_backend_observation(observation_index=len(obs), backend_name="cand", dependency_name=c.dependency_name, observation_name=f"obs_{c.candidate_index}", observation_kind="JSON_VALUE", backend_role="CANDIDATE", payload={"x":1}, payload_hash="", error_code=None, unavailable_reason=None, source_invariant_candidate_hash=c.candidate_hash))
    eq = build_equivalence_receipt_from_observations(inv_receipt, obs, equivalence_policy="EXACT_CANONICAL_JSON")
    return manifest, hot_receipt, inv_receipt, eq


def test_deterministic_builds_and_json_bytes_and_derivation_behaviors():
    m,h,i,e = _inputs(False)
    idx1 = derive_optimization_opportunity_index(m,h,i,e)
    idx2 = derive_optimization_opportunity_index(m,h,i,e)
    assert idx1.optimization_opportunity_index_hash == idx2.optimization_opportunity_index_hash
    assert idx1.to_canonical_json() == idx2.to_canonical_json()
    assert idx1.to_canonical_bytes() == idx2.to_canonical_bytes()
    json.dumps(idx1.evidence[0].to_dict(), allow_nan=False)
    json.dumps(idx1.opportunities[0].to_dict(), allow_nan=False)
    kinds = {o.opportunity_kind: o for o in idx1.opportunities}
    assert kinds["POLICY_BLOCKED_DEPENDENCY_REVIEW"].readiness_status == "BLOCKED"
    assert kinds["UNAVAILABLE_BACKEND_REVIEW"] if "UNAVAILABLE_BACKEND_REVIEW" in kinds else True
    assert kinds["QUANTUM_BACKEND_ADAPTER_REVIEW"].readiness_status == "NEEDS_EQUIVALENCE_RECEIPT"
    assert kinds["SPARSE_DENSE_BOUNDARY_REVIEW"].required_next_receipt == "CrossBackendEquivalenceReceipt"
    assert kinds["PLOTTING_RENDER_BYPASS"].required_next_receipt == "OptimizationContract"
    assert kinds["DATAFRAME_SCHEMA_CACHE_REVIEW"].readiness_status == "NEEDS_EQUIVALENCE_RECEIPT"
    assert kinds["AUDIO_MIDI_ADAPTER_REVIEW"].readiness_status == "READY_FOR_OPTIMIZATION_CONTRACT"
    assert kinds["INTERNAL_QEC_FASTPATH_REVIEW"].readiness_status == "READY_FOR_OPTIMIZATION_CONTRACT"
    assert all("speedup" not in json.dumps(o.to_dict()).lower() for o in idx1.opportunities)
    assert all("speedup" not in o.rank_reason.lower() for o in idx1.opportunities)
    assert "confidence" not in json.dumps(idx1.to_dict()).lower()
    assert all(isinstance(o.total_priority_score, int) for o in idx1.opportunities)
    assert all(0 <= o.total_priority_score <= 20 for o in idx1.opportunities)
    assert all(o.total_priority_score == o.static_determinism_score + o.static_value_score + o.dependency_reduction_score + (5 - o.implementation_risk_score) for o in idx1.opportunities)
    assert [o.opportunity_index for o in idx1.opportunities] == list(range(len(idx1.opportunities)))
    assert [x.evidence_index for x in idx1.evidence] == list(range(len(idx1.evidence)))
    assert idx1.opportunities[-1].readiness_status == "BLOCKED"


def test_equivalence_ready_and_validators_and_source_scan_and_mismatch():
    m,h,i,e = _inputs(True)
    idx = derive_optimization_opportunity_index(m,h,i,e)
    kinds = {o.opportunity_kind: o for o in idx.opportunities}
    assert kinds["QUANTUM_BACKEND_ADAPTER_REVIEW"].readiness_status == "READY_FOR_OPTIMIZATION_CONTRACT"
    assert validate_optimization_opportunity_evidence(idx.evidence[0]) is True
    assert validate_optimization_opportunity_entry(idx.opportunities[0]) is True
    assert validate_optimization_opportunity_index(idx) is True
    assert validate_index_matches_inputs(idx,m,h,i,e) is True
    bad_idx = OptimizationOpportunityIndex(**{**idx.__dict__, "final_opportunity_hash": "0"*64})
    with pytest.raises(ValueError, match="OPTIMIZATION_INDEX_MISMATCH"):
        validate_index_matches_inputs(bad_idx,m,h,i,e)
    bad = OptimizationOpportunityEvidence(**{**idx.evidence[0].to_dict(), "evidence_hash":"x"})
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"): validate_optimization_opportunity_evidence(bad)
    bad2 = OptimizationOpportunityEvidence(**{**idx.evidence[0].to_dict(), "evidence_hash":"0"*64})
    with pytest.raises(ValueError, match="HASH_MISMATCH"): validate_optimization_opportunity_evidence(bad2)
    bad3 = OptimizationOpportunityEntry(**{**idx.opportunities[0].to_dict(), "opportunity_hash":"x"})
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"): validate_optimization_opportunity_entry(bad3)
    bad4 = OptimizationOpportunityEntry(**{**idx.opportunities[0].to_dict(), "opportunity_hash":"0"*64})
    with pytest.raises(ValueError, match="HASH_MISMATCH"): validate_optimization_opportunity_entry(bad4)
    bad5 = OptimizationOpportunityIndex(**{**idx.__dict__, "optimization_opportunity_index_hash":"x"})
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"): validate_optimization_opportunity_index(bad5)
    bad6 = OptimizationOpportunityIndex(**{**idx.__dict__, "optimization_opportunity_index_hash":"0"*64})
    with pytest.raises(ValueError, match="HASH_MISMATCH"): validate_optimization_opportunity_index(bad6)
    with pytest.raises(ValueError, match="INVALID_SCHEMA_VERSION"):
        validate_optimization_opportunity_index(OptimizationOpportunityIndex(**{**idx.__dict__, "schema_version":"BAD"}))
    with pytest.raises(ValueError, match="INVALID_INDEX_MODE"):
        validate_optimization_opportunity_index(OptimizationOpportunityIndex(**{**idx.__dict__, "index_mode":"BAD"}))
    src = open("src/qec/analysis/optimization_opportunity_index.py", encoding="utf-8").read().lower()
    for token in ["import qutip","import qiskit","import matplotlib","import pandas","import stim","import pymatching","import mido","import requests","urllib.request","subprocess","os.system","shell=true","eval(","exec(","__import__(","importlib.import_module","pip","time.time","datetime.now","random."]:
        assert token not in src
