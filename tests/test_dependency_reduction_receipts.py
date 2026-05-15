from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import pytest

from qec.analysis.heavy_dependency_discovery import build_default_unprobed_manifest
from qec.analysis.dependency_hotpath_receipts import build_dependency_hotpath_candidate, build_dependency_import_and_hotpath_receipt
from qec.analysis.backend_invariant_candidate_receipts import build_backend_invariant_candidate, build_backend_invariant_candidate_receipt
from qec.analysis.cross_backend_equivalence_receipts import build_backend_observation, build_cross_backend_comparison_case, build_cross_backend_equivalence_receipt
from qec.analysis.optimization_opportunity_index import derive_optimization_opportunity_index
from qec.analysis.optimization_contracts import build_optimization_contract_from_opportunity
from qec.analysis.lightweight_adapter_specs import build_lightweight_adapter_spec_from_contract
from qec.analysis.cached_canonical_kernel_receipts import build_cached_canonical_kernel_receipt_from_adapter
from qec.analysis.fast_path_equivalence_receipts import build_fast_path_observation, build_fast_path_comparison_case, build_fast_path_equivalence_receipt
from qec.analysis.optimization_implementation_receipts import build_optimization_implementation_receipt_from_fast_path
from qec.analysis.dependency_reduction_receipts import *

def _chain():
    m = build_default_unprobed_manifest()
    hot = [build_dependency_hotpath_candidate(candidate_index=0, dependency_name="qiskit", source_path="src/a.py", line_number=1, candidate_kind="QUANTUM_BACKEND_BOUNDARY", candidate_status="NEEDS_EQUIVALENCE_PROOF", reason="r", related_import_site_hashes=())]
    hr = build_dependency_import_and_hotpath_receipt((), hot, source_root_label="src", scanned_file_count=1, target_registry_hash=m.heavy_dependency_discovery_manifest_hash)
    deps=["qldpc_external","qiskit","numpy","matplotlib","pandas","mido","qldpc_internal"]; kinds=["POLICY_BLOCKED_EXTERNAL_INVARIANT","QUANTUM_BACKEND_BOUNDARY_INVARIANT","SPARSE_DENSE_BOUNDARY_INVARIANT","PLOTTING_RENDER_BOUNDARY_INVARIANT","DATAFRAME_SCHEMA_BOUNDARY_INVARIANT","AUDIO_MIDI_BOUNDARY_INVARIANT","INTERNAL_QEC_SURFACE_INVARIANT"]
    c=[build_backend_invariant_candidate(candidate_index=i,dependency_name=deps[i],invariant_name=f"inv{i}",invariant_kind=kinds[i],invariant_status="CANDIDATE_IDENTIFIED",review_class=("BLOCKED_BY_POLICY" if i==0 else "SAFE_FOR_OPTIMIZATION_CONTRACT_REVIEW"),required_next_receipt=("CrossBackendEquivalenceReceipt" if i in (1,2,4) else "OptimizationContract"),evidence_hashes=(),source_paths=("src/a.py",),reason="r") for i in range(len(deps))]
    ir = build_backend_invariant_candidate_receipt(m, hr, (), c)
    o1=build_backend_observation(observation_index=0, backend_name="r", dependency_name="qiskit", observation_name="n", observation_kind="JSON_VALUE", backend_role="REFERENCE", payload={"x":1}, error_code=None, unavailable_reason=None, source_invariant_candidate_hash=None)
    o2=build_backend_observation(observation_index=1, backend_name="c", dependency_name="qiskit", observation_name="n", observation_kind="JSON_VALUE", backend_role="CANDIDATE", payload={"x":1}, error_code=None, unavailable_reason=None, source_invariant_candidate_hash=None)
    cc=build_cross_backend_comparison_case(case_index=0, case_name="n", equivalence_policy="EXACT_CANONICAL_JSON", reference_observation_hash=o1.observation_hash, candidate_observation_hashes=(o2.observation_hash,), source_candidate_hash=None, case_reason="r")
    eq = build_cross_backend_equivalence_receipt(ir, [o1,o2], [cc])
    idx = derive_optimization_opportunity_index(m, hr, ir, eq)
    ready = next(o for o in idx.opportunities if o.readiness_status == "READY_FOR_OPTIMIZATION_CONTRACT")
    contract = build_optimization_contract_from_opportunity(idx, ready.opportunity_hash)
    adapter = build_lightweight_adapter_spec_from_contract(contract)
    cache = build_cached_canonical_kernel_receipt_from_adapter(contract, adapter)
    kh = cache.kernel_descriptors[0].kernel_hash
    r=build_fast_path_observation(observation_index=0,observation_role="REFERENCE",observation_kind="HASH_ONLY",observation_name="r",dependency_name=contract.dependency_name,source_kernel_hash=kh,source_cached_canonical_kernel_receipt_hash=cache.cached_canonical_kernel_receipt_hash,payload=None,payload_hash="a"*64,shape=None,dtype=None,ordered_sequence=None,set_like_sequence=None,error_code=None,unavailable_reason=None,reason="r")
    cnd=build_fast_path_observation(observation_index=1,observation_role="CANDIDATE",observation_kind="HASH_ONLY",observation_name="c",dependency_name=contract.dependency_name,source_kernel_hash=kh,source_cached_canonical_kernel_receipt_hash=cache.cached_canonical_kernel_receipt_hash,payload=None,payload_hash="a"*64,shape=None,dtype=None,ordered_sequence=None,set_like_sequence=None,error_code=None,unavailable_reason=None,reason="r")
    case=build_fast_path_comparison_case(case_index=0,case_name="k",equivalence_policy="EXACT_HASH",reference_observation_hash=r.observation_hash,candidate_observation_hash=cnd.observation_hash,source_kernel_hash=kh,source_cached_canonical_kernel_receipt_hash=cache.cached_canonical_kernel_receipt_hash,reason="r")
    fp=build_fast_path_equivalence_receipt(contract,adapter,cache,"FAST_PATH_EQUIVALENCE_PASSED",(r,cnd),(case,))
    imp=build_optimization_implementation_receipt_from_fast_path(contract,adapter,cache,fp)
    return m,hr,ir,eq,idx,contract,adapter,cache,fp,imp

def test_dependency_reduction_child_hash_stability():
    c=_chain(); r=build_dependency_reduction_receipt_from_implementation(*c)
    t=r.targets[0]
    assert build_dependency_reduction_target(**{**t.to_dict(),"reduction_target_hash":""}).reduction_target_hash==t.reduction_target_hash

def test_dependency_reduction_receipt_hash_stability():
    c=_chain(); r1=build_dependency_reduction_receipt_from_implementation(*c); r2=build_dependency_reduction_receipt_from_implementation(*c)
    assert r1.dependency_reduction_receipt_hash==r2.dependency_reduction_receipt_hash

def test_dependency_reduction_from_implementation_lineage():
    c=_chain(); r=build_dependency_reduction_receipt_from_implementation(*c)
    assert validate_dependency_reduction_receipt_matches_inputs(r,*c)

def test_dependency_reduction_hash_validation():
    c=_chain(); r=build_dependency_reduction_receipt_from_implementation(*c)
    with pytest.raises(ValueError,match="INVALID_HASH_FORMAT"): validate_dependency_reduction_receipt(replace(r, dependency_reduction_receipt_hash="x"))

def test_dependency_reduction_source_scan_and_decoder_boundary():
    text=Path("src/qec/analysis/dependency_reduction_receipts.py").read_text(encoding="utf-8")
    for token in ["import numpy","import scipy","import pandas","import matplotlib","import qutip","import qiskit","import qiskit_aer","import stim","import pymatching","import mido","import qldpc","import requests","urllib","subprocess","importlib.import_module","__import__(","eval(","exec(","os.system","shell=True","pip","time.time","datetime.now","random."]:
        assert token not in text
    for p in Path("src/qec/decoder").glob("**/*.py"):
        t=p.read_text(encoding="utf-8")
        assert "DependencyReductionReceipt" not in t and "dependency_reduction_receipt_hash" not in t
