from __future__ import annotations

import json
from pathlib import Path

import pytest

from qec.analysis.backend_invariant_candidate_receipts import build_backend_invariant_candidate, build_backend_invariant_candidate_receipt
import qec.analysis.cross_backend_equivalence_receipts as cber
from qec.analysis.cross_backend_equivalence_receipts import *
from qec.analysis.dependency_hotpath_receipts import build_dependency_hotpath_candidate, build_dependency_import_and_hotpath_receipt, build_dependency_import_site
from qec.analysis.heavy_dependency_discovery import build_default_unprobed_manifest


def _inv_receipt():
    m = build_default_unprobed_manifest()
    s = build_dependency_import_site(dependency_name="qiskit", import_name="qiskit", source_path="a.py", line_number=1, import_kind="IMPORT", import_placement="MODULE_TOP_LEVEL", imported_symbol=None, is_heavy_target=True)
    h = build_dependency_hotpath_candidate(candidate_index=0, dependency_name="qiskit", source_path="a.py", line_number=1, candidate_kind="QUANTUM_BACKEND_BOUNDARY", candidate_status="NEEDS_EQUIVALENCE_PROOF", reason="r", related_import_site_hashes=(s.import_site_hash,))
    hp = build_dependency_import_and_hotpath_receipt([s], [h], source_root_label="src", scanned_file_count=1)
    c = build_backend_invariant_candidate(candidate_index=0, dependency_name="qiskit", invariant_name="inv", invariant_kind="UNAVAILABLE_BACKEND_INVARIANT", invariant_status="CANDIDATE_IDENTIFIED", review_class="NEEDS_EQUIVALENCE_RECEIPT", evidence_hashes=(), source_paths=("a.py",), reason="r", required_next_receipt="CrossBackendEquivalenceReceipt")
    return build_backend_invariant_candidate_receipt(m, hp, [], [c]), c


def _obs(i, role, name="n", payload=None, kind="JSON_VALUE", e=None, u=None, ch=None):
    return build_backend_observation(observation_index=i, backend_name=f"b{role}{i}", dependency_name="qiskit", observation_name=name, observation_kind=kind, backend_role=role, payload=payload, error_code=e, unavailable_reason=u, source_invariant_candidate_hash=ch)


def test_hash_determinism_builders_and_receipt_and_json_exports():
    inv, c = _inv_receipt()
    o1 = _obs(0, "REFERENCE", payload={"a": 1}, ch=c.candidate_hash)
    o2 = _obs(1, "CANDIDATE", payload={"a": 1}, ch=c.candidate_hash)
    case = build_cross_backend_comparison_case(case_index=0, case_name="n", equivalence_policy="EXACT_CANONICAL_JSON", reference_observation_hash=o1.observation_hash, candidate_observation_hashes=(o2.observation_hash,), source_candidate_hash=c.candidate_hash, case_reason="r")
    r1 = evaluate_cross_backend_case(case, {o1.observation_hash: o1, o2.observation_hash: o2}, result_index=0)
    r2 = evaluate_cross_backend_case(case, {o1.observation_hash: o1, o2.observation_hash: o2}, result_index=0)
    assert o1.observation_hash == _obs(0, "REFERENCE", payload={"a": 1}, ch=c.candidate_hash).observation_hash
    assert case.case_hash == build_cross_backend_comparison_case(**{**case.to_dict(), "case_hash": ""}).case_hash
    assert r1.result_hash == r2.result_hash
    receipt = build_cross_backend_equivalence_receipt(inv, [o1, o2], [case])
    receipt2 = build_cross_backend_equivalence_receipt(inv, [o1, o2], [case])
    assert receipt.cross_backend_equivalence_receipt_hash == receipt2.cross_backend_equivalence_receipt_hash
    json.loads(o1.to_canonical_json()); json.loads(case.to_canonical_json()); json.loads(r1.to_canonical_json())
    assert receipt.to_canonical_json() == receipt2.to_canonical_json()
    assert receipt.to_canonical_bytes() == receipt2.to_canonical_bytes()


@pytest.mark.parametrize("policy,ref,cand,kind,status", [
    ("EXACT_CANONICAL_JSON", {"x": 1}, {"x": 1}, "JSON_VALUE", "EQUIVALENT"),
    ("EXACT_CANONICAL_JSON", {"x": 1}, {"x": 2}, "JSON_VALUE", "NOT_EQUIVALENT"),
    ("EXACT_HASH", [1], [1], "JSON_VECTOR", "EQUIVALENT"),
    ("STRUCTURAL_SHAPE_DTYPE", {"shape": [2], "dtype": "f64", "layout": "C"}, {"shape": [2], "dtype": "f64", "layout": "C"}, "JSON_SHAPE_DTYPE", "EQUIVALENT"),
    ("STRUCTURAL_SHAPE_DTYPE", {"shape": [2], "dtype": "f64", "layout": "C"}, {"shape": [3], "dtype": "f64", "layout": "C"}, "JSON_SHAPE_DTYPE", "NOT_EQUIVALENT"),
    ("ORDERED_SEQUENCE_EXACT", [1,2], [1,2], "JSON_VECTOR", "EQUIVALENT"),
    ("ORDERED_SEQUENCE_EXACT", [1,2], [2,1], "JSON_VECTOR", "NOT_EQUIVALENT"),
    ("SET_LIKE_SORTED_EXACT", [1,2], [2,1], "JSON_VECTOR", "EQUIVALENT"),
])
def test_policies(policy, ref, cand, kind, status):
    o1 = _obs(0, "REFERENCE", payload=ref, kind=kind)
    o2 = _obs(1, "CANDIDATE", payload=cand, kind=kind)
    case = build_cross_backend_comparison_case(case_index=0, case_name="n", equivalence_policy=policy, reference_observation_hash=o1.observation_hash, candidate_observation_hashes=(o2.observation_hash,), source_candidate_hash=None, case_reason="r")
    res = evaluate_cross_backend_case(case, {o1.observation_hash: o1, o2.observation_hash: o2}, result_index=0)
    assert res.result_status == status


def test_declared_unavailable_error_and_invalid_payload_policy():
    u1 = _obs(0, "REFERENCE", kind="UNAVAILABLE_RESULT", u="u")
    u2 = _obs(1, "CANDIDATE", kind="UNAVAILABLE_RESULT", u="u")
    c1 = build_cross_backend_comparison_case(case_index=0, case_name="u", equivalence_policy="DECLARED_UNAVAILABLE_MATCH", reference_observation_hash=u1.observation_hash, candidate_observation_hashes=(u2.observation_hash,), source_candidate_hash=None, case_reason="r")
    assert evaluate_cross_backend_case(c1, {u1.observation_hash: u1, u2.observation_hash: u2}, result_index=0).result_status == "EQUIVALENT"
    e1 = _obs(0, "REFERENCE", kind="ERROR_RESULT", e="E")
    e2 = _obs(1, "CANDIDATE", kind="ERROR_RESULT", e="E")
    c2 = build_cross_backend_comparison_case(case_index=0, case_name="e", equivalence_policy="DECLARED_ERROR_MATCH", reference_observation_hash=e1.observation_hash, candidate_observation_hashes=(e2.observation_hash,), source_candidate_hash=None, case_reason="r")
    assert evaluate_cross_backend_case(c2, {e1.observation_hash: e1, e2.observation_hash: e2}, result_index=0).result_status == "EQUIVALENT"
    bad = build_cross_backend_comparison_case(case_index=0, case_name="b", equivalence_policy="ORDERED_SEQUENCE_EXACT", reference_observation_hash=e1.observation_hash, candidate_observation_hashes=(e2.observation_hash,), source_candidate_hash=None, case_reason="r")
    assert evaluate_cross_backend_case(bad, {e1.observation_hash: e1, e2.observation_hash: e2}, result_index=0).result_status == "INVALID_OBSERVATION"


def test_missing_and_ambiguous_reference_and_contiguity_and_hash_validation_and_schema_mode_counts():
    inv, c = _inv_receipt()
    r = _obs(0, "REFERENCE", name="x", payload=1, ch=c.candidate_hash)
    a = _obs(1, "CANDIDATE", name="x", payload=1, ch=c.candidate_hash)
    with pytest.raises(ValueError, match="MISSING_REFERENCE_OBSERVATION"):
        build_equivalence_receipt_from_observations(inv, [a], equivalence_policy="EXACT_HASH")
    with pytest.raises(ValueError, match="MISSING_CANDIDATE_OBSERVATION"):
        build_equivalence_receipt_from_observations(inv, [r], equivalence_policy="EXACT_HASH")
    with pytest.raises(ValueError, match="AMBIGUOUS_REFERENCE_OBSERVATION"):
        build_equivalence_receipt_from_observations(inv, [r, _obs(2, "REFERENCE", name="x", payload=1, ch=c.candidate_hash), a], equivalence_policy="EXACT_HASH")
    case = build_cross_backend_comparison_case(case_index=0, case_name="x", equivalence_policy="EXACT_HASH", reference_observation_hash=r.observation_hash, candidate_observation_hashes=(a.observation_hash,), source_candidate_hash=c.candidate_hash, case_reason="r")
    rcpt = build_cross_backend_equivalence_receipt(inv, [r, a], [case])
    assert validate_cross_backend_equivalence_receipt(rcpt)
    assert rcpt.observation_count == 2 and rcpt.comparison_case_count == 1 and rcpt.comparison_result_count == 1
    assert "cross_backend_equivalence_receipt_hash" in rcpt.to_dict()
    with pytest.raises(ValueError, match="OBSERVATION_ORDER_MISMATCH"):
        build_cross_backend_equivalence_receipt(inv, [_obs(1, "REFERENCE", payload=1), _obs(2, "CANDIDATE", payload=1)], [case])
    with pytest.raises(ValueError, match="CASE_ORDER_MISMATCH"):
        build_cross_backend_equivalence_receipt(inv, [r, a], [build_cross_backend_comparison_case(case_index=1, case_name="x", equivalence_policy="EXACT_HASH", reference_observation_hash=r.observation_hash, candidate_observation_hashes=(a.observation_hash,), source_candidate_hash=None, case_reason="r")])
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_backend_observation(BackendObservation(**{**r.to_dict(), "observation_hash": "x"}))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_backend_observation(BackendObservation(**{**r.to_dict(), "observation_hash": "0"*64}))
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_cross_backend_comparison_case(CrossBackendComparisonCase(**{**case.to_dict(), "case_hash": "x", "candidate_observation_hashes": tuple(case.candidate_observation_hashes)}))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_cross_backend_comparison_case(CrossBackendComparisonCase(**{**case.to_dict(), "case_hash": "0"*64, "candidate_observation_hashes": tuple(case.candidate_observation_hashes)}))
    res = rcpt.comparison_results[0]
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_cross_backend_comparison_result(CrossBackendComparisonResult(**{**res.to_dict(), "result_hash": "x", "candidate_payload_hashes": tuple(res.candidate_payload_hashes)}))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_cross_backend_comparison_result(CrossBackendComparisonResult(**{**res.to_dict(), "result_hash": "0"*64, "candidate_payload_hashes": tuple(res.candidate_payload_hashes)}))
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_cross_backend_equivalence_receipt(CrossBackendEquivalenceReceipt(**{**rcpt.__dict__, "cross_backend_equivalence_receipt_hash": "x"}))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_cross_backend_equivalence_receipt(CrossBackendEquivalenceReceipt(**{**rcpt.__dict__, "cross_backend_equivalence_receipt_hash": "0"*64}))
    with pytest.raises(ValueError, match="INVALID_SCHEMA_VERSION"):
        validate_cross_backend_equivalence_receipt(CrossBackendEquivalenceReceipt(**{**rcpt.__dict__, "schema_version": "x"}))
    with pytest.raises(ValueError, match="INVALID_EQUIVALENCE_MODE"):
        validate_cross_backend_equivalence_receipt(CrossBackendEquivalenceReceipt(**{**rcpt.__dict__, "equivalence_mode": "x"}))
    with pytest.raises(ValueError, match="CROSS_BACKEND_RECEIPT_MISMATCH"):
        validate_receipt_matches_inputs(rcpt, inv, [r], [case])


def test_no_fuzzy_policy_and_source_scan_forbidden_tokens():
    assert not any("ULP" in x or "TOLERANCE" in x or "FUZZY" in x for x in cber._ALLOWED_EQUIVALENCE_POLICIES)
    text = Path("src/qec/analysis/cross_backend_equivalence_receipts.py").read_text(encoding="utf-8")
    for token in ["import qutip", "import qiskit", "import matplotlib", "import pandas", "import stim", "import pymatching", "import mido", "import requests", "urllib.request", "subprocess", "os.system", "shell=True", "eval(", "exec(", "__import__(", "importlib.import_module", "pip", "time.time", "datetime.now", "random."]:
        assert token not in text
