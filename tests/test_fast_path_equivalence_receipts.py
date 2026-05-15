from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import math
import pytest

import qec.analysis.fast_path_equivalence_receipts as fper
from qec.analysis.fast_path_equivalence_receipts import *
from qec.analysis.heavy_dependency_discovery import build_default_unprobed_manifest
from qec.analysis.dependency_hotpath_receipts import build_dependency_hotpath_candidate, build_dependency_import_and_hotpath_receipt
from qec.analysis.backend_invariant_candidate_receipts import build_backend_invariant_candidate, build_backend_invariant_candidate_receipt
from qec.analysis.optimization_opportunity_index import derive_optimization_opportunity_index
from qec.analysis.optimization_contracts import build_optimization_contract_from_opportunity
from qec.analysis.lightweight_adapter_specs import build_lightweight_adapter_spec_from_contract
from qec.analysis.cached_canonical_kernel_receipts import build_cached_canonical_kernel_receipt_from_adapter


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
    return contract, adapter, cache, kh


def _obs(i, role, kind, **kw):
    c, _, cache, kh = _chain()
    return build_fast_path_observation(observation_index=i, observation_role=role, observation_kind=kind, observation_name=f"n{i}", dependency_name=c.dependency_name, source_kernel_hash=kh, source_cached_canonical_kernel_receipt_hash=cache.cached_canonical_kernel_receipt_hash, payload=kw.get("payload"), payload_hash=kw.get("payload_hash", "a" * 64), shape=kw.get("shape"), dtype=kw.get("dtype"), ordered_sequence=kw.get("ordered_sequence"), set_like_sequence=kw.get("set_like_sequence"), error_code=kw.get("error_code"), unavailable_reason=kw.get("unavailable_reason"), reason="r")


def test_fast_path_observation_hash_stability():
    for kind, extra in [("CANONICAL_JSON", {"payload": {"a": 1}}), ("HASH_ONLY", {"payload_hash": "b" * 64}), ("STRUCTURAL_SHAPE_DTYPE", {"shape": (2,), "dtype": "f64"}), ("ORDERED_SEQUENCE", {"ordered_sequence": (1, 2)}), ("SET_LIKE_SEQUENCE", {"set_like_sequence": (2, 1)}), ("DECLARED_UNAVAILABLE", {"unavailable_reason": "u"}), ("DECLARED_ERROR", {"error_code": "E"})]:
        o = _obs(0, "REFERENCE", kind, **extra)
        rebuilt = build_fast_path_observation(**{**o.to_dict(), "observation_hash": ""})
        assert o.observation_hash == rebuilt.observation_hash
        assert o.to_canonical_json() == rebuilt.to_canonical_json()
        assert o.to_canonical_bytes() == rebuilt.to_canonical_bytes()


def test_fast_path_comparison_policy_pass_and_fail():
    checks = [
        ("EXACT_CANONICAL_JSON", _obs(0, "REFERENCE", "CANONICAL_JSON", payload={"x": 1}), _obs(1, "CANDIDATE", "CANONICAL_JSON", payload={"x": 2}), "CANONICAL_JSON_MISMATCH"),
        ("EXACT_HASH", _obs(0, "REFERENCE", "HASH_ONLY", payload_hash="a" * 64), _obs(1, "CANDIDATE", "HASH_ONLY", payload_hash="b" * 64), "HASH_MISMATCH"),
        ("STRUCTURAL_SHAPE_DTYPE", _obs(0, "REFERENCE", "STRUCTURAL_SHAPE_DTYPE", shape=(2,), dtype="f64"), _obs(1, "CANDIDATE", "STRUCTURAL_SHAPE_DTYPE", shape=(3,), dtype="f64"), "SHAPE_DTYPE_MISMATCH"),
        ("ORDERED_SEQUENCE_EXACT", _obs(0, "REFERENCE", "ORDERED_SEQUENCE", ordered_sequence=(1, 2)), _obs(1, "CANDIDATE", "ORDERED_SEQUENCE", ordered_sequence=(2, 1)), "ORDERED_SEQUENCE_MISMATCH"),
        ("SET_LIKE_SORTED_EXACT", _obs(0, "REFERENCE", "SET_LIKE_SEQUENCE", set_like_sequence=(2, 1)), _obs(1, "CANDIDATE", "SET_LIKE_SEQUENCE", set_like_sequence=(1, 3)), "SET_LIKE_SEQUENCE_MISMATCH"),
        ("DECLARED_UNAVAILABLE_MATCH", _obs(0, "REFERENCE", "DECLARED_UNAVAILABLE", unavailable_reason="a"), _obs(1, "CANDIDATE", "DECLARED_UNAVAILABLE", unavailable_reason="b"), "DECLARED_UNAVAILABLE_MISMATCH"),
        ("DECLARED_ERROR_MATCH", _obs(0, "REFERENCE", "DECLARED_ERROR", error_code="E1"), _obs(1, "CANDIDATE", "DECLARED_ERROR", error_code="E2"), "DECLARED_ERROR_MISMATCH"),
    ]
    for pol, r, c, fail in checks:
        assert fper._evaluate_case(pol, r, r) == (True, None)
        assert fper._evaluate_case(pol, r, c) == (False, fail)


def test_fast_path_sequence_policy_kind_mismatch_is_structured_failure():
    assert fper._evaluate_case("ORDERED_SEQUENCE_EXACT", _obs(0, "REFERENCE", "HASH_ONLY", payload_hash="a" * 64), _obs(1, "CANDIDATE", "HASH_ONLY", payload_hash="b" * 64)) == (False, "OBSERVATION_KIND_POLICY_MISMATCH")
    assert fper._evaluate_case("SET_LIKE_SORTED_EXACT", _obs(0, "REFERENCE", "HASH_ONLY", payload_hash="a" * 64), _obs(1, "CANDIDATE", "HASH_ONLY", payload_hash="b" * 64)) == (False, "OBSERVATION_KIND_POLICY_MISMATCH")


def test_fast_path_receipt_validation_handles_sequence_policy_kind_mismatch():
    contract, adapter, cache, kh = _chain()
    r = _obs(0, "REFERENCE", "HASH_ONLY", payload_hash="a" * 64)
    c = _obs(1, "CANDIDATE", "HASH_ONLY", payload_hash="b" * 64)
    case = build_fast_path_comparison_case(case_index=0, case_name="k", equivalence_policy="ORDERED_SEQUENCE_EXACT", reference_observation_hash=r.observation_hash, candidate_observation_hash=c.observation_hash, source_kernel_hash=kh, source_cached_canonical_kernel_receipt_hash=cache.cached_canonical_kernel_receipt_hash, reason="r")
    rec = build_fast_path_equivalence_receipt(contract, adapter, cache, "FAST_PATH_EQUIVALENCE_FAILED", (r, c), (case,))
    assert rec.comparison_results[0].failure_code == "OBSERVATION_KIND_POLICY_MISMATCH"
    assert validate_fast_path_equivalence_receipt(rec)


def test_fast_path_equivalence_receipt_from_cache_lineage():
    contract, adapter, cache, kh = _chain()
    r = _obs(0, "REFERENCE", "HASH_ONLY", payload_hash="a" * 64)
    c = _obs(1, "CANDIDATE", "HASH_ONLY", payload_hash="a" * 64)
    case = build_fast_path_comparison_case(case_index=0, case_name="k", equivalence_policy="EXACT_HASH", reference_observation_hash=r.observation_hash, candidate_observation_hash=c.observation_hash, source_kernel_hash=kh, source_cached_canonical_kernel_receipt_hash=cache.cached_canonical_kernel_receipt_hash, reason="r")
    rec = build_fast_path_equivalence_receipt_from_cache(contract, adapter, cache, (r, c), (case,), equivalence_status="FAST_PATH_EQUIVALENCE_PASSED")
    assert validate_fast_path_equivalence_receipt_matches_inputs(rec, contract, adapter, cache)


def test_fast_path_equivalence_receipt_rejects_lineage_mismatch():
    contract, adapter, cache, kh = _chain()
    r = _obs(0, "REFERENCE", "HASH_ONLY", payload_hash="a" * 64)
    c = _obs(1, "CANDIDATE", "HASH_ONLY", payload_hash="a" * 64)
    case = build_fast_path_comparison_case(case_index=0, case_name="k", equivalence_policy="EXACT_HASH", reference_observation_hash=r.observation_hash, candidate_observation_hash=c.observation_hash, source_kernel_hash=kh, source_cached_canonical_kernel_receipt_hash=cache.cached_canonical_kernel_receipt_hash, reason="r")
    rec = build_fast_path_equivalence_receipt(contract, adapter, cache, "FAST_PATH_EQUIVALENCE_PASSED", (r, c), (case,))
    with pytest.raises(ValueError, match="HASH_MISMATCH|RECEIPT_CONTRACT_MISMATCH"): validate_fast_path_equivalence_receipt_matches_inputs(replace(rec, source_optimization_contract_hash="0" * 64), contract, adapter, cache)


def test_fast_path_result_re_evaluation():
    contract, adapter, cache, kh = _chain()
    r = _obs(0, "REFERENCE", "HASH_ONLY", payload_hash="a" * 64)
    c = _obs(1, "CANDIDATE", "HASH_ONLY", payload_hash="b" * 64)
    case = build_fast_path_comparison_case(case_index=0, case_name="k", equivalence_policy="EXACT_HASH", reference_observation_hash=r.observation_hash, candidate_observation_hash=c.observation_hash, source_kernel_hash=kh, source_cached_canonical_kernel_receipt_hash=cache.cached_canonical_kernel_receipt_hash, reason="r")
    rec = build_fast_path_equivalence_receipt(contract, adapter, cache, "FAST_PATH_EQUIVALENCE_FAILED", (r, c), (case,))
    bad = replace(rec.comparison_results[0], equivalence_passed=True, failure_code=None)
    bad = replace(bad, result_hash=build_fast_path_comparison_result(**{**bad.to_dict(), "result_hash": ""}).result_hash)
    with pytest.raises(ValueError, match="RESULT_ORDER_MISMATCH|RESULT_EVALUATION_MISMATCH"):
        validate_fast_path_equivalence_receipt(replace(rec, comparison_results=(bad,)))


def test_fast_path_hash_validation():
    o = _obs(0, "REFERENCE", "HASH_ONLY", payload_hash="a" * 64)
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"): validate_fast_path_observation(replace(o, observation_hash="x"))
    with pytest.raises(ValueError, match="HASH_MISMATCH"): validate_fast_path_observation(replace(o, observation_hash="0" * 64))


def test_fast_path_counts_and_ordering():
    contract, adapter, cache, kh = _chain()
    r = _obs(0, "REFERENCE", "HASH_ONLY", payload_hash="a" * 64)
    c = _obs(1, "CANDIDATE", "HASH_ONLY", payload_hash="a" * 64)
    case = build_fast_path_comparison_case(case_index=0, case_name="k", equivalence_policy="EXACT_HASH", reference_observation_hash=r.observation_hash, candidate_observation_hash=c.observation_hash, source_kernel_hash=kh, source_cached_canonical_kernel_receipt_hash=cache.cached_canonical_kernel_receipt_hash, reason="r")
    rec = build_fast_path_equivalence_receipt(contract, adapter, cache, "FAST_PATH_EQUIVALENCE_PASSED", (r, c), (case,))
    with pytest.raises(ValueError, match="COUNT_MISMATCH"): validate_fast_path_equivalence_receipt(replace(rec, observation_count=999))
    with pytest.raises(ValueError, match="INVALID_INPUT"): build_fast_path_observation(**{**r.to_dict(), "observation_index": True, "observation_hash": ""})


def test_fast_path_payload_validation():
    with pytest.raises(ValueError, match="INVALID_PAYLOAD"): fper._canonical_payload_hash({"x": math.nan})


def test_fast_path_source_scan_and_decoder_boundary():
    text = Path("src/qec/analysis/fast_path_equivalence_receipts.py").read_text(encoding="utf-8")
    for token in ["import numpy", "import scipy", "import pandas", "import matplotlib", "import qutip", "import qiskit", "import qiskit_aer", "import stim", "import pymatching", "import mido", "import qldpc", "import requests", "urllib", "subprocess", "importlib.import_module", "eval(", "exec(", "os.system", "shell=True", "pip", "time.time", "datetime.now", "random."]:
        assert token not in text
    for p in Path("src/qec/decoder").glob("**/*.py"):
        t = p.read_text(encoding="utf-8")
        assert "FastPathEquivalenceReceipt" not in t and "fast_path_equivalence_receipt_hash" not in t


def test_fast_path_no_scope_creep():
    contract, adapter, cache, kh = _chain()
    r = _obs(0, "REFERENCE", "HASH_ONLY", payload_hash="a" * 64)
    c = _obs(1, "CANDIDATE", "HASH_ONLY", payload_hash="a" * 64)
    case = build_fast_path_comparison_case(case_index=0, case_name="k", equivalence_policy="EXACT_HASH", reference_observation_hash=r.observation_hash, candidate_observation_hash=c.observation_hash, source_kernel_hash=kh, source_cached_canonical_kernel_receipt_hash=cache.cached_canonical_kernel_receipt_hash, reason="r")
    rec = build_fast_path_equivalence_receipt(contract, adapter, cache, "FAST_PATH_EQUIVALENCE_PASSED", (r, c), (case,))
    txt = rec.to_canonical_json().lower()
    for token in ["speedup", "benchmark proven", "runtime cache", "memoization", "fast path implemented", "implementation complete", "dependency reduction"]:
        assert token not in txt
