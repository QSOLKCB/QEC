from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import pytest

from qec.analysis.heavy_dependency_discovery import build_default_unprobed_manifest
from qec.analysis.dependency_hotpath_receipts import build_dependency_hotpath_candidate, build_dependency_import_and_hotpath_receipt
from qec.analysis.backend_invariant_candidate_receipts import build_backend_invariant_candidate, build_backend_invariant_candidate_receipt
from qec.analysis.optimization_opportunity_index import derive_optimization_opportunity_index
from qec.analysis.optimization_contracts import build_optimization_contract_from_opportunity
from qec.analysis.lightweight_adapter_specs import build_lightweight_adapter_spec_from_contract
from qec.analysis.cached_canonical_kernel_receipts import build_cached_canonical_kernel_receipt_from_adapter
from qec.analysis.fast_path_equivalence_receipts import build_fast_path_observation, build_fast_path_comparison_case, build_fast_path_equivalence_receipt
from qec.analysis.optimization_implementation_receipts import *

def _chain():
    m = build_default_unprobed_manifest()
    hot = [build_dependency_hotpath_candidate(candidate_index=0, dependency_name="qiskit", source_path="src/a.py", line_number=1, candidate_kind="QUANTUM_BACKEND_BOUNDARY", candidate_status="NEEDS_EQUIVALENCE_PROOF", reason="r", related_import_site_hashes=())]
    hr = build_dependency_import_and_hotpath_receipt((), hot, source_root_label="src", scanned_file_count=1, target_registry_hash=m.heavy_dependency_discovery_manifest_hash)
    deps = ["qldpc_external", "qiskit", "numpy", "matplotlib", "pandas", "mido", "qldpc_internal"]
    kinds=["POLICY_BLOCKED_EXTERNAL_INVARIANT","QUANTUM_BACKEND_BOUNDARY_INVARIANT","SPARSE_DENSE_BOUNDARY_INVARIANT","PLOTTING_RENDER_BOUNDARY_INVARIANT","DATAFRAME_SCHEMA_BOUNDARY_INVARIANT","AUDIO_MIDI_BOUNDARY_INVARIANT","INTERNAL_QEC_SURFACE_INVARIANT"]
    cands=[build_backend_invariant_candidate(candidate_index=i, dependency_name=deps[i], invariant_name=f"inv{i}", invariant_kind=kinds[i], invariant_status="CANDIDATE_IDENTIFIED", review_class=("BLOCKED_BY_POLICY" if i==0 else "SAFE_FOR_OPTIMIZATION_CONTRACT_REVIEW"), required_next_receipt=("CrossBackendEquivalenceReceipt" if i in (1,2,4) else "OptimizationContract"), evidence_hashes=(), source_paths=("src/a.py",), reason="r") for i in range(len(deps))]
    ir = build_backend_invariant_candidate_receipt(m, hr, (), cands)
    idx = derive_optimization_opportunity_index(m, hr, ir, None)
    ready = next(o for o in idx.opportunities if o.readiness_status == "READY_FOR_OPTIMIZATION_CONTRACT")
    contract = build_optimization_contract_from_opportunity(idx, ready.opportunity_hash)
    adapter = build_lightweight_adapter_spec_from_contract(contract)
    cache = build_cached_canonical_kernel_receipt_from_adapter(contract, adapter)
    kh = cache.kernel_descriptors[0].kernel_hash
    r=build_fast_path_observation(observation_index=0, observation_role="REFERENCE", observation_kind="HASH_ONLY", observation_name="r", dependency_name=contract.dependency_name, source_kernel_hash=kh, source_cached_canonical_kernel_receipt_hash=cache.cached_canonical_kernel_receipt_hash, payload=None, payload_hash="a"*64, shape=None, dtype=None, ordered_sequence=None, set_like_sequence=None, error_code=None, unavailable_reason=None, reason="r")
    cnd=build_fast_path_observation(observation_index=1, observation_role="CANDIDATE", observation_kind="HASH_ONLY", observation_name="c", dependency_name=contract.dependency_name, source_kernel_hash=kh, source_cached_canonical_kernel_receipt_hash=cache.cached_canonical_kernel_receipt_hash, payload=None, payload_hash="a"*64, shape=None, dtype=None, ordered_sequence=None, set_like_sequence=None, error_code=None, unavailable_reason=None, reason="r")
    case=build_fast_path_comparison_case(case_index=0, case_name="k", equivalence_policy="EXACT_HASH", reference_observation_hash=r.observation_hash, candidate_observation_hash=cnd.observation_hash, source_kernel_hash=kh, source_cached_canonical_kernel_receipt_hash=cache.cached_canonical_kernel_receipt_hash, reason="r")
    fast=build_fast_path_equivalence_receipt(contract,adapter,cache,"FAST_PATH_EQUIVALENCE_PASSED",(r,cnd),(case,))
    return contract,adapter,cache,fast

def test_optimization_implementation_child_hash_stability():
    contract,adapter,cache,fast=_chain(); rec=build_optimization_implementation_receipt_from_fast_path(contract,adapter,cache,fast)
    g=rec.guards[0]; assert g.guard_hash==build_optimization_implementation_guard(**{**g.to_dict(),"guard_hash":""}).guard_hash
    assert g.to_canonical_bytes()==build_optimization_implementation_guard(**{**g.to_dict(),"guard_hash":""}).to_canonical_bytes()

def test_optimization_implementation_receipt_hash_stability():
    c,a,ca,f=_chain(); r1=build_optimization_implementation_receipt_from_fast_path(c,a,ca,f); r2=build_optimization_implementation_receipt_from_fast_path(c,a,ca,f)
    assert r1.optimization_implementation_receipt_hash==r2.optimization_implementation_receipt_hash

def test_optimization_implementation_from_fast_path_lineage():
    c,a,ca,f=_chain(); r=build_optimization_implementation_receipt_from_fast_path(c,a,ca,f)
    assert validate_optimization_implementation_receipt_matches_inputs(r,c,a,ca,f)

def test_optimization_implementation_hash_validation():
    c,a,ca,f=_chain(); r=build_optimization_implementation_receipt_from_fast_path(c,a,ca,f)
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"): validate_optimization_implementation_receipt(replace(r, optimization_implementation_receipt_hash="x"))

def test_optimization_implementation_source_scan_and_decoder_boundary():
    text=Path("src/qec/analysis/optimization_implementation_receipts.py").read_text(encoding="utf-8")
    for token in ["import numpy","import scipy","import pandas","import matplotlib","import qiskit","subprocess","urllib","eval(","exec(","time.time","datetime.now","random."]:
        assert token not in text
    for p in Path("src/qec/decoder").glob("**/*.py"):
        t=p.read_text(encoding="utf-8")
        assert "OptimizationImplementationReceipt" not in t and "optimization_implementation_receipt_hash" not in t
