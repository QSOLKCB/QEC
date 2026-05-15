import json
import pytest

from qec.analysis.heavy_dependency_discovery import build_default_unprobed_manifest
from qec.analysis.dependency_hotpath_receipts import build_dependency_hotpath_candidate, build_dependency_import_and_hotpath_receipt
from qec.analysis.backend_invariant_candidate_receipts import build_backend_invariant_evidence, build_backend_invariant_candidate, build_backend_invariant_candidate_receipt
from qec.analysis.optimization_opportunity_index import derive_optimization_opportunity_index
from qec.analysis.optimization_contracts import *


def _index():
    m = build_default_unprobed_manifest()
    hot = [build_dependency_hotpath_candidate(candidate_index=0, dependency_name="qiskit", source_path="src/a.py", line_number=1, candidate_kind="QUANTUM_BACKEND_BOUNDARY", candidate_status="NEEDS_EQUIVALENCE_PROOF", reason="r", related_import_site_hashes=())]
    hr = build_dependency_import_and_hotpath_receipt((), hot, source_root_label="src", scanned_file_count=1, target_registry_hash=m.heavy_dependency_discovery_manifest_hash)
    deps = ["qldpc_external", "qiskit", "numpy", "matplotlib", "pandas", "mido", "qldpc_internal"]
    kinds=["POLICY_BLOCKED_EXTERNAL_INVARIANT","QUANTUM_BACKEND_BOUNDARY_INVARIANT","SPARSE_DENSE_BOUNDARY_INVARIANT","PLOTTING_RENDER_BOUNDARY_INVARIANT","DATAFRAME_SCHEMA_BOUNDARY_INVARIANT","AUDIO_MIDI_BOUNDARY_INVARIANT","INTERNAL_QEC_SURFACE_INVARIANT"]
    ev=[build_backend_invariant_evidence(evidence_index=i, dependency_name=d, evidence_kind="HOTPATH_CANDIDATE_EVIDENCE", source_path="src/a.py", line_number=1, import_site_hash=None, hotpath_candidate_hash=hot[0].candidate_hash if d=="qiskit" else None, probe_hash=m.probe_results[0].probe_hash, reason="r") for i,d in enumerate(deps)]
    c=[build_backend_invariant_candidate(candidate_index=i, dependency_name=deps[i], invariant_name=f"inv{i}", invariant_kind=kinds[i], invariant_status="CANDIDATE_IDENTIFIED", review_class=("BLOCKED_BY_POLICY" if i==0 else "SAFE_FOR_OPTIMIZATION_CONTRACT_REVIEW"), required_next_receipt=("CrossBackendEquivalenceReceipt" if i in (1,2,4) else "OptimizationContract"), evidence_hashes=(ev[i].evidence_hash,), source_paths=("src/a.py",), reason="r") for i in range(len(deps))]
    ir = build_backend_invariant_candidate_receipt(m, hr, ev, c)
    return derive_optimization_opportunity_index(m, hr, ir, None)


def _ready_contract():
    idx = _index()
    ready = next(o for o in idx.opportunities if o.readiness_status == "READY_FOR_OPTIMIZATION_CONTRACT")
    return idx, build_optimization_contract_from_opportunity(idx, ready.opportunity_hash)


def test_contract_artifacts_and_determinism_and_readiness_paths():
    idx, c1 = _ready_contract(); _, c2 = _ready_contract()
    assert c1.optimization_contract_hash == c2.optimization_contract_hash
    assert c1.to_canonical_json() == c2.to_canonical_json(); assert c1.to_canonical_bytes() == c2.to_canonical_bytes()
    assert validate_optimization_contract(c1) is True
    json.dumps(c1.preconditions[0].to_dict(), allow_nan=False); json.dumps(c1.equivalence_requirements[0].to_dict(), allow_nan=False); json.dumps(c1.rollback_conditions[0].to_dict(), allow_nan=False)
    assert all(p.source_opportunity_hash == c1.source_opportunity_hash for p in c1.preconditions)
    assert all(r.source_opportunity_hash == c1.source_opportunity_hash for r in c1.equivalence_requirements)
    assert all(r.source_opportunity_hash == c1.source_opportunity_hash for r in c1.rollback_conditions)
    kinds = {p.precondition_kind for p in c1.preconditions}
    assert {"OPPORTUNITY_READY","OPPORTUNITY_HASH_BOUND","EVIDENCE_HASH_BOUND","EQUIVALENCE_POLICY_DECLARED","BENCHMARK_NOT_CLAIMED","DECODER_UNTOUCHED","NO_HEAVY_IMPORT_EXECUTION","NO_NETWORK_EXECUTION","ROLLBACK_DECLARED"}.issubset(kinds)
    assert len(c1.equivalence_requirements) >= 1
    assert {x.rollback_kind for x in c1.rollback_conditions} >= {"REJECT_ON_HASH_MISMATCH","REJECT_ON_EQUIVALENCE_FAILURE","DISABLE_FAST_PATH","REJECT_ON_POLICY_VIOLATION"}
    txt = c1.to_canonical_json().lower(); assert "claims speedup" not in txt and "implementation complete" not in txt
    with pytest.raises(ValueError, match="OPPORTUNITY_NOT_FOUND"): build_optimization_contract_from_opportunity(idx, "0"*64)
    bad_eq = next(o for o in idx.opportunities if o.readiness_status == "NEEDS_EQUIVALENCE_RECEIPT")
    with pytest.raises(ValueError, match="OPPORTUNITY_NOT_READY"): build_optimization_contract_from_opportunity(idx, bad_eq.opportunity_hash)
    bad_block = next(o for o in idx.opportunities if o.readiness_status == "BLOCKED")
    with pytest.raises(ValueError, match="OPPORTUNITY_NOT_READY"): build_optimization_contract_from_opportunity(idx, bad_block.opportunity_hash)


def test_hash_and_schema_validations_and_match_validator_and_source_scan():
    idx, c = _ready_contract()
    p = c.preconditions[0]; r = c.equivalence_requirements[0]; b = c.rollback_conditions[0]
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"): validate_optimization_contract_precondition(OptimizationContractPrecondition(**{**p.to_dict(), "precondition_hash":"x"}))
    with pytest.raises(ValueError, match="HASH_MISMATCH"): validate_optimization_contract_precondition(OptimizationContractPrecondition(**{**p.to_dict(), "precondition_hash":"0"*64}))
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"): validate_optimization_equivalence_requirement(OptimizationEquivalenceRequirement(**{**r.to_dict(), "requirement_hash":"x"}))
    with pytest.raises(ValueError, match="HASH_MISMATCH"): validate_optimization_equivalence_requirement(OptimizationEquivalenceRequirement(**{**r.to_dict(), "requirement_hash":"0"*64}))
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"): validate_optimization_rollback_condition(OptimizationRollbackCondition(**{**b.to_dict(), "rollback_hash":"x"}))
    with pytest.raises(ValueError, match="HASH_MISMATCH"): validate_optimization_rollback_condition(OptimizationRollbackCondition(**{**b.to_dict(), "rollback_hash":"0"*64}))
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"): validate_optimization_contract(OptimizationContract(**{**c.__dict__, "optimization_contract_hash":"x"}))
    with pytest.raises(ValueError, match="HASH_MISMATCH"): validate_optimization_contract(OptimizationContract(**{**c.__dict__, "optimization_contract_hash":"0"*64}))
    with pytest.raises(ValueError, match="INVALID_SCHEMA_VERSION"): validate_optimization_contract(OptimizationContract(**{**c.__dict__, "schema_version":"BAD"}))
    with pytest.raises(ValueError, match="INVALID_CONTRACT_MODE"): validate_optimization_contract(OptimizationContract(**{**c.__dict__, "contract_mode":"BAD"}))
    with pytest.raises(ValueError, match="PRECONDITION_ORDER_MISMATCH"): validate_optimization_contract(OptimizationContract(**{**c.__dict__, "first_precondition_hash":"0"*64}))
    assert c.precondition_count == len(c.preconditions) and c.equivalence_requirement_count == len(c.equivalence_requirements) and c.rollback_condition_count == len(c.rollback_conditions)
    assert [x.precondition_index for x in c.preconditions] == list(range(len(c.preconditions)))
    assert [x.requirement_index for x in c.equivalence_requirements] == list(range(len(c.equivalence_requirements)))
    assert [x.rollback_index for x in c.rollback_conditions] == list(range(len(c.rollback_conditions)))
    assert validate_contract_matches_opportunity(c, idx) is True
    with pytest.raises(ValueError, match="OPTIMIZATION_CONTRACT_MISMATCH"): validate_contract_matches_opportunity(OptimizationContract(**{**c.__dict__, "contract_name":"other"}), idx)
    src = open("src/qec/analysis/optimization_contracts.py", encoding="utf-8").read().lower()
    for token in ["import qutip","import qiskit","import matplotlib","import pandas","import stim","import pymatching","import mido","import requests","urllib.request","subprocess","os.system","shell=true","eval(","exec(","__import__(","importlib.import_module","pip","time.time","datetime.now","random."]:
        assert token not in src
