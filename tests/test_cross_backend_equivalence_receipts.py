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


def test_hash_only_observation_with_explicit_hash():
    """Issue 1: HASH_ONLY observations can carry backend-supplied hash without payload"""
    inv, c = _inv_receipt()
    explicit_hash = "a" * 64
    o = build_backend_observation(observation_index=0, backend_name="b", dependency_name="qiskit", observation_name="n", observation_kind="HASH_ONLY", backend_role="REFERENCE", payload=None, payload_hash=explicit_hash, error_code=None, unavailable_reason=None, source_invariant_candidate_hash=c.candidate_hash)
    assert o.payload_hash == explicit_hash
    assert o.payload is None
    assert validate_backend_observation(o)


def test_invalid_json_payload_normalized():
    """Issue 2: Invalid JSON payloads raise INVALID_PAYLOAD"""
    inv, c = _inv_receipt()
    # Non-serializable object - error happens in builder before validation
    try:
        build_backend_observation(observation_index=0, backend_name="b", dependency_name="qiskit", observation_name="n", observation_kind="JSON_VALUE", backend_role="REFERENCE", payload=object(), error_code=None, unavailable_reason=None, source_invariant_candidate_hash=c.candidate_hash)
        assert False, "Should have raised"
    except (TypeError, ValueError) as e:
        # Either TypeError from json.dumps or ValueError from validation
        pass
    
    # Test oversized payload
    huge_payload = {"data": "x" * 20000}
    with pytest.raises(ValueError, match="INVALID_PAYLOAD"):
        build_backend_observation(observation_index=0, backend_name="b", dependency_name="qiskit", observation_name="n", observation_kind="JSON_VALUE", backend_role="REFERENCE", payload=huge_payload, error_code=None, unavailable_reason=None, source_invariant_candidate_hash=c.candidate_hash)


def test_declared_policies_require_explicit_fields():
    """Issue 3: Declared error/unavailable policies require explicit error_code/unavailable_reason"""
    inv, c = _inv_receipt()
    # Missing unavailable_reason
    u1 = _obs(0, "REFERENCE", kind="UNAVAILABLE_RESULT", u=None)
    u2 = _obs(1, "CANDIDATE", kind="UNAVAILABLE_RESULT", u=None)
    case_u = build_cross_backend_comparison_case(case_index=0, case_name="u", equivalence_policy="DECLARED_UNAVAILABLE_MATCH", reference_observation_hash=u1.observation_hash, candidate_observation_hashes=(u2.observation_hash,), source_candidate_hash=None, case_reason="r")
    res_u = evaluate_cross_backend_case(case_u, {u1.observation_hash: u1, u2.observation_hash: u2}, result_index=0)
    assert res_u.result_status == "NOT_EQUIVALENT"
    
    # Missing error_code
    e1 = _obs(0, "REFERENCE", kind="ERROR_RESULT", e=None)
    e2 = _obs(1, "CANDIDATE", kind="ERROR_RESULT", e=None)
    case_e = build_cross_backend_comparison_case(case_index=0, case_name="e", equivalence_policy="DECLARED_ERROR_MATCH", reference_observation_hash=e1.observation_hash, candidate_observation_hashes=(e2.observation_hash,), source_candidate_hash=None, case_reason="r")
    res_e = evaluate_cross_backend_case(case_e, {e1.observation_hash: e1, e2.observation_hash: e2}, result_index=0)
    assert res_e.result_status == "NOT_EQUIVALENT"


def test_source_candidate_hash_validation():
    """Issue 4: Validate observation/case source_invariant_candidate_hash belongs to receipt"""
    inv, c = _inv_receipt()
    fake_hash = "f" * 64
    o1 = _obs(0, "REFERENCE", payload=1, ch=fake_hash)
    o2 = _obs(1, "CANDIDATE", payload=1, ch=c.candidate_hash)
    case = build_cross_backend_comparison_case(case_index=0, case_name="x", equivalence_policy="EXACT_HASH", reference_observation_hash=o1.observation_hash, candidate_observation_hashes=(o2.observation_hash,), source_candidate_hash=c.candidate_hash, case_reason="r")
    with pytest.raises(ValueError, match="INVALID_SOURCE_CANDIDATE_HASH"):
        build_cross_backend_equivalence_receipt(inv, [o1, o2], [case])
    
    # Also test case with invalid source_candidate_hash
    o3 = _obs(0, "REFERENCE", payload=1, ch=c.candidate_hash)
    o4 = _obs(1, "CANDIDATE", payload=1, ch=c.candidate_hash)
    case2 = build_cross_backend_comparison_case(case_index=0, case_name="x", equivalence_policy="EXACT_HASH", reference_observation_hash=o3.observation_hash, candidate_observation_hashes=(o4.observation_hash,), source_candidate_hash=fake_hash, case_reason="r")
    with pytest.raises(ValueError, match="INVALID_SOURCE_CANDIDATE_HASH"):
        build_cross_backend_equivalence_receipt(inv, [o3, o4], [case2])


def test_grouping_by_source_identity():
    """Issue 5: Grouping includes source identity, not just observation_name"""
    m = build_default_unprobed_manifest()
    s = build_dependency_import_site(dependency_name="qiskit", import_name="qiskit", source_path="a.py", line_number=1, import_kind="IMPORT", import_placement="MODULE_TOP_LEVEL", imported_symbol=None, is_heavy_target=True)
    h = build_dependency_hotpath_candidate(candidate_index=0, dependency_name="qiskit", source_path="a.py", line_number=1, candidate_kind="QUANTUM_BACKEND_BOUNDARY", candidate_status="NEEDS_EQUIVALENCE_PROOF", reason="r", related_import_site_hashes=(s.import_site_hash,))
    hp = build_dependency_import_and_hotpath_receipt([s], [h], source_root_label="src", scanned_file_count=1)
    
    # Create two candidates for different sources
    c1 = build_backend_invariant_candidate(candidate_index=0, dependency_name="qiskit", invariant_name="inv", invariant_kind="UNAVAILABLE_BACKEND_INVARIANT", invariant_status="CANDIDATE_IDENTIFIED", review_class="NEEDS_EQUIVALENCE_RECEIPT", evidence_hashes=(), source_paths=("a.py",), reason="r", required_next_receipt="CrossBackendEquivalenceReceipt")
    c2 = build_backend_invariant_candidate(candidate_index=1, dependency_name="qiskit", invariant_name="inv2", invariant_kind="UNAVAILABLE_BACKEND_INVARIANT", invariant_status="CANDIDATE_IDENTIFIED", review_class="NEEDS_EQUIVALENCE_RECEIPT", evidence_hashes=(), source_paths=("b.py",), reason="r", required_next_receipt="CrossBackendEquivalenceReceipt")
    inv = build_backend_invariant_candidate_receipt(m, hp, [], [c1, c2])
    
    # Same observation_name but different source_invariant_candidate_hash
    o1 = _obs(0, "REFERENCE", name="shared", payload=1, ch=c1.candidate_hash)
    o2 = _obs(1, "CANDIDATE", name="shared", payload=1, ch=c1.candidate_hash)
    o3 = _obs(2, "REFERENCE", name="shared", payload=2, ch=c2.candidate_hash)
    o4 = _obs(3, "CANDIDATE", name="shared", payload=2, ch=c2.candidate_hash)
    
    receipt = build_equivalence_receipt_from_observations(inv, [o1, o2, o3, o4], equivalence_policy="EXACT_HASH")
    # Should create 2 cases, one per source
    assert receipt.comparison_case_count == 2
    assert all(r.result_status == "EQUIVALENT" for r in receipt.comparison_results)


def test_all_count_fields_validated():
    """Issue 6 (P1): Validate all receipt count fields"""
    inv, c = _inv_receipt()
    r = _obs(0, "REFERENCE", payload=1, ch=c.candidate_hash)
    a = _obs(1, "CANDIDATE", payload=1, ch=c.candidate_hash)
    case = build_cross_backend_comparison_case(case_index=0, case_name="x", equivalence_policy="EXACT_HASH", reference_observation_hash=r.observation_hash, candidate_observation_hashes=(a.observation_hash,), source_candidate_hash=c.candidate_hash, case_reason="r")
    rcpt = build_cross_backend_equivalence_receipt(inv, [r, a], [case])
    
    # Test observation_count
    with pytest.raises(ValueError, match="OBSERVATION_COUNT_MISMATCH"):
        validate_cross_backend_equivalence_receipt(CrossBackendEquivalenceReceipt(**{**rcpt.__dict__, "observation_count": 999}))
    
    # Test comparison_case_count
    with pytest.raises(ValueError, match="CASE_COUNT_MISMATCH"):
        validate_cross_backend_equivalence_receipt(CrossBackendEquivalenceReceipt(**{**rcpt.__dict__, "comparison_case_count": 999}))
    
    # Test comparison_result_count
    with pytest.raises(ValueError, match="RESULT_COUNT_MISMATCH"):
        validate_cross_backend_equivalence_receipt(CrossBackendEquivalenceReceipt(**{**rcpt.__dict__, "comparison_result_count": 999}))
    
    # Test not_equivalent_count
    with pytest.raises(ValueError, match="NOT_EQUIVALENT_COUNT_MISMATCH"):
        validate_cross_backend_equivalence_receipt(CrossBackendEquivalenceReceipt(**{**rcpt.__dict__, "not_equivalent_count": 999}))
    
    # Test incomplete_count
    with pytest.raises(ValueError, match="INCOMPLETE_COUNT_MISMATCH"):
        validate_cross_backend_equivalence_receipt(CrossBackendEquivalenceReceipt(**{**rcpt.__dict__, "incomplete_count": 999}))
    
    # Test blocked_by_policy_count
    with pytest.raises(ValueError, match="BLOCKED_BY_POLICY_COUNT_MISMATCH"):
        validate_cross_backend_equivalence_receipt(CrossBackendEquivalenceReceipt(**{**rcpt.__dict__, "blocked_by_policy_count": 999}))
    
    # Test invalid_observation_count
    with pytest.raises(ValueError, match="INVALID_OBSERVATION_COUNT_MISMATCH"):
        validate_cross_backend_equivalence_receipt(CrossBackendEquivalenceReceipt(**{**rcpt.__dict__, "invalid_observation_count": 999}))


def test_result_matches_case_and_observations():
    """Issue 7: Verify embedded results match their cases and observations"""
    inv, c = _inv_receipt()
    r = _obs(0, "REFERENCE", payload=1, ch=c.candidate_hash)
    a = _obs(1, "CANDIDATE", payload=2, ch=c.candidate_hash)  # Different payload
    case = build_cross_backend_comparison_case(case_index=0, case_name="x", equivalence_policy="EXACT_HASH", reference_observation_hash=r.observation_hash, candidate_observation_hashes=(a.observation_hash,), source_candidate_hash=c.candidate_hash, case_reason="r")
    rcpt = build_cross_backend_equivalence_receipt(inv, [r, a], [case])
    
    # Forge a result with wrong status
    forged_result = build_cross_backend_comparison_result(result_index=0, case_hash=case.case_hash, equivalence_policy=case.equivalence_policy, result_status="EQUIVALENT", reference_payload_hash=r.payload_hash, candidate_payload_hashes=(a.payload_hash,), mismatch_reason=None)
    forged_receipt = CrossBackendEquivalenceReceipt(**{**rcpt.__dict__, "comparison_results": (forged_result,), "equivalent_count": 1, "not_equivalent_count": 0})
    
    with pytest.raises(ValueError, match="RESULT_EVALUATION_MISMATCH"):
        validate_cross_backend_equivalence_receipt(forged_receipt)


def test_case_field_validation():
    """Issue 8: Validate case_index, case_name, case_reason"""
    inv, c = _inv_receipt()
    r = _obs(0, "REFERENCE", payload=1, ch=c.candidate_hash)
    a = _obs(1, "CANDIDATE", payload=1, ch=c.candidate_hash)
    
    # Invalid case_index
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_cross_backend_comparison_case(case_index=-1, case_name="x", equivalence_policy="EXACT_HASH", reference_observation_hash=r.observation_hash, candidate_observation_hashes=(a.observation_hash,), source_candidate_hash=None, case_reason="r")
    
    # Empty case_name
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_cross_backend_comparison_case(case_index=0, case_name="", equivalence_policy="EXACT_HASH", reference_observation_hash=r.observation_hash, candidate_observation_hashes=(a.observation_hash,), source_candidate_hash=None, case_reason="r")
    
    # Oversized case_reason
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_cross_backend_comparison_case(case_index=0, case_name="x", equivalence_policy="EXACT_HASH", reference_observation_hash=r.observation_hash, candidate_observation_hashes=(a.observation_hash,), source_candidate_hash=None, case_reason="x" * 300)


def test_result_field_validation():
    """Issue 9: Validate result_index and mismatch_reason"""
    inv, c = _inv_receipt()
    r = _obs(0, "REFERENCE", payload=1, ch=c.candidate_hash)
    a = _obs(1, "CANDIDATE", payload=1, ch=c.candidate_hash)
    case = build_cross_backend_comparison_case(case_index=0, case_name="x", equivalence_policy="EXACT_HASH", reference_observation_hash=r.observation_hash, candidate_observation_hashes=(a.observation_hash,), source_candidate_hash=None, case_reason="r")
    
    # Invalid result_index
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_cross_backend_comparison_result(result_index=-1, case_hash=case.case_hash, equivalence_policy="EXACT_HASH", result_status="EQUIVALENT", reference_payload_hash=r.payload_hash, candidate_payload_hashes=(a.payload_hash,), mismatch_reason=None)
    
    # Oversized mismatch_reason
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_cross_backend_comparison_result(result_index=0, case_hash=case.case_hash, equivalence_policy="EXACT_HASH", result_status="NOT_EQUIVALENT", reference_payload_hash=r.payload_hash, candidate_payload_hashes=(a.payload_hash,), mismatch_reason="x" * 300)


def test_observation_role_validation():
    """Issue 10: Validate observation roles in case evaluation"""
    inv, c = _inv_receipt()
    # Reference observation with wrong role
    r = _obs(0, "CANDIDATE", payload=1, ch=c.candidate_hash)  # Should be REFERENCE
    a = _obs(1, "CANDIDATE", payload=1, ch=c.candidate_hash)
    case = build_cross_backend_comparison_case(case_index=0, case_name="x", equivalence_policy="EXACT_HASH", reference_observation_hash=r.observation_hash, candidate_observation_hashes=(a.observation_hash,), source_candidate_hash=None, case_reason="r")
    
    with pytest.raises(ValueError, match="INVALID_BACKEND_ROLE"):
        evaluate_cross_backend_case(case, {r.observation_hash: r, a.observation_hash: a}, result_index=0)
    
    # Candidate with REFERENCE role
    r2 = _obs(0, "REFERENCE", payload=1, ch=c.candidate_hash)
    a2 = _obs(1, "REFERENCE", payload=1, ch=c.candidate_hash)  # Should be CANDIDATE
    case2 = build_cross_backend_comparison_case(case_index=0, case_name="x", equivalence_policy="EXACT_HASH", reference_observation_hash=r2.observation_hash, candidate_observation_hashes=(a2.observation_hash,), source_candidate_hash=None, case_reason="r")
    
    with pytest.raises(ValueError, match="INVALID_BACKEND_ROLE"):
        evaluate_cross_backend_case(case2, {r2.observation_hash: r2, a2.observation_hash: a2}, result_index=0)
    
    # Reference included as its own candidate - this triggers INVALID_BACKEND_ROLE because reference has REFERENCE role
    r3 = _obs(0, "REFERENCE", payload=1, ch=c.candidate_hash)
    case3 = build_cross_backend_comparison_case(case_index=0, case_name="x", equivalence_policy="EXACT_HASH", reference_observation_hash=r3.observation_hash, candidate_observation_hashes=(r3.observation_hash,), source_candidate_hash=None, case_reason="r")
    
    with pytest.raises(ValueError, match="INVALID_BACKEND_ROLE"):
        evaluate_cross_backend_case(case3, {r3.observation_hash: r3}, result_index=0)

