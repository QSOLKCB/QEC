from dataclasses import FrozenInstanceError, replace
import json
import pytest

from qec.analysis.canonical_hashing import canonical_json
from qec.analysis.substrate_constraint_contract import build_substrate_constraint_predicate, build_substrate_contract
from qec.analysis.substrate_state_receipt import (
    PredicateEvaluationResult,
    SubstrateStateReceipt,
    _ERR_CANONICAL_JSON_TOO_LARGE,
    _ERR_HASH_MISMATCH,
    _ERR_INVALID_CANONICAL_JSON,
    _ERR_INVALID_EVALUATION_STATUS,
    _ERR_INVALID_HASH_FORMAT,
    _ERR_PREDICATE_COUNT_MISMATCH,
    _ERR_PREDICATE_EVALUATION_ORDER_MISMATCH,
    _ERR_SOURCE_ARTIFACT_HASH_MISMATCH,
    _ERR_SUBSTRATE_CONTRACT_MISMATCH,
    _ERR_SUBSTRATE_STATE_CLASS_MISMATCH,
    build_predicate_evaluation_result,
    build_substrate_state_receipt,
    validate_predicate_evaluation_result,
    validate_predicate_evaluation_result_with_contract,
    validate_substrate_state_receipt,
    validate_substrate_state_receipt_with_contract,
)


def _mk_contract(source, preds):
    txt = canonical_json(source)
    from qec.analysis.substrate_state_receipt import _canonical_json_text_hash
    return txt, build_substrate_contract("Artifact", _canonical_json_text_hash(txt), "Profile", preds)


def test_predicate_basics_and_freeze_and_exports():
    pred = build_substrate_constraint_predicate("P", "STRING_EQUALS", ["a"], {"value": "x"})
    src, contract = _mk_contract({"a": "x"}, [pred])
    r1 = build_predicate_evaluation_result(contract, pred, src)
    r2 = build_predicate_evaluation_result(contract, pred, src)
    assert r1.predicate_evaluation_result_hash == r2.predicate_evaluation_result_hash
    assert r1.to_canonical_json() == r2.to_canonical_json()
    assert r1.to_canonical_bytes() == r2.to_canonical_bytes()
    with pytest.raises(FrozenInstanceError):
        r1.passed = False
    with pytest.raises(ValueError, match=_ERR_SUBSTRATE_CONTRACT_MISMATCH):
        other = build_substrate_constraint_predicate("Q", "FIELD_PRESENT", ["z"], {})
        build_predicate_evaluation_result(contract, other, src)
    with pytest.raises(ValueError, match=_ERR_INVALID_HASH_FORMAT):
        validate_predicate_evaluation_result(replace(r1, predicate_evaluation_result_hash="bad"))
    with pytest.raises(ValueError, match=_ERR_HASH_MISMATCH):
        validate_predicate_evaluation_result(replace(r1, predicate_evaluation_result_hash="0" * 64))
    with pytest.raises(ValueError, match=_ERR_INVALID_HASH_FORMAT):
        validate_predicate_evaluation_result(replace(r1, observed_value_hash="A" * 64))
    with pytest.raises(ValueError, match=_ERR_INVALID_EVALUATION_STATUS):
        validate_predicate_evaluation_result(replace(r1, evaluation_status="NOPE"))


def test_predicate_semantics_matrix():
    src_obj = {"obj": {}, "arr": [], "s": "ok", "i": 4, "b": True, "n": None, "h": "a" * 64}
    src, _ = _mk_contract(src_obj, [build_substrate_constraint_predicate("tmp", "FIELD_PRESENT", ["obj"], {})])
    cases = [
        ("FIELD_PRESENT", ["obj"], {}, True, "PREDICATE_PASSED"),
        ("FIELD_ABSENT", ["obj"], {}, False, "FIELD_PRESENT_UNEXPECTED"),
        ("FIELD_TYPE", ["obj"], {"json_type": "object"}, True, "PREDICATE_PASSED"),
        ("STRING_EQUALS", ["s"], {"value": "no"}, False, "VALUE_MISMATCH"),
        ("STRING_IN_SET", ["s"], {"allowed_values": ["ok"]}, True, "PREDICATE_PASSED"),
        ("INTEGER_RANGE", ["i"], {"min_value": 5, "max_value": 9}, False, "INTEGER_RANGE_MISMATCH"),
        ("BOOLEAN_EQUALS", ["b"], {"value": True}, True, "PREDICATE_PASSED"),
        ("HASH_FORMAT", ["h"], {}, True, "PREDICATE_PASSED"),
        ("CANONICAL_BYTES_MAX", [], {"max_bytes": len(src.encode())}, True, "PREDICATE_PASSED"),
    ]
    for i, (kind, path, params, exp_pass, exp_status) in enumerate(cases):
        p = build_substrate_constraint_predicate(f"P{i}", kind, path, params)
        _, c = _mk_contract(src_obj, [p])
        r = build_predicate_evaluation_result(c, p, src)
        assert r.passed is exp_pass
        assert r.evaluation_status == exp_status
    pm = build_substrate_constraint_predicate("pm", "FIELD_TYPE", ["missing"], {"json_type": "string"})
    _, cm = _mk_contract(src_obj, [pm])
    assert build_predicate_evaluation_result(cm, pm, src).evaluation_status == "FIELD_MISSING"


def test_receipt_basics_and_counts_and_tamper_checks():
    p1 = build_substrate_constraint_predicate("A", "FIELD_PRESENT", ["a"], {})
    p2 = build_substrate_constraint_predicate("B", "STRING_EQUALS", ["a"], {"value": "z"})
    src, c = _mk_contract({"a": "z"}, [p1, p2])
    receipt = build_substrate_state_receipt(c, src)
    assert receipt.substrate_state_class == "SUBSTRATE_STATE_COMPATIBLE"
    assert receipt.predicate_count == 2 and receipt.passed_count == 2 and receipt.failed_count == 0 and receipt.all_predicates_passed
    with pytest.raises(FrozenInstanceError):
        receipt.predicate_count = 1
    with pytest.raises(ValueError, match=_ERR_INVALID_HASH_FORMAT):
        validate_substrate_state_receipt(replace(receipt, substrate_state_receipt_hash="bad"))
    with pytest.raises(ValueError, match=_ERR_HASH_MISMATCH):
        validate_substrate_state_receipt(replace(receipt, substrate_state_receipt_hash="0" * 64))
    with pytest.raises(ValueError, match=_ERR_PREDICATE_COUNT_MISMATCH):
        validate_substrate_state_receipt(replace(receipt, predicate_count=True))
    with pytest.raises(ValueError, match=_ERR_SUBSTRATE_STATE_CLASS_MISMATCH):
        validate_substrate_state_receipt(replace(receipt, substrate_state_class="BAD"))
    unsorted_results = tuple(reversed(receipt.predicate_evaluation_results))
    with pytest.raises(ValueError, match=_ERR_PREDICATE_EVALUATION_ORDER_MISMATCH):
        SubstrateStateReceipt(receipt.source_artifact_type, receipt.source_artifact_hash, receipt.source_canonical_json_hash, receipt.substrate_contract_hash, receipt.substrate_profile_id, unsorted_results, 2, 2, 0, True, "SUBSTRATE_STATE_COMPATIBLE", receipt.substrate_state_receipt_hash)


def test_complete_validators_and_canonical_boundaries_and_scope_scan():
    p = build_substrate_constraint_predicate("A", "CANONICAL_BYTES_MAX", [], {"max_bytes": 100})
    src, c = _mk_contract([1, 2, 3], [p])
    r = build_predicate_evaluation_result(c, p, src)
    receipt = build_substrate_state_receipt(c, src)
    assert validate_predicate_evaluation_result_with_contract(r, c, p, src)
    assert validate_substrate_state_receipt_with_contract(receipt, c, src)
    with pytest.raises(ValueError, match=_ERR_SOURCE_ARTIFACT_HASH_MISMATCH):
        build_substrate_state_receipt(c, canonical_json([1, 2]))
    with pytest.raises(ValueError, match=_ERR_INVALID_CANONICAL_JSON):
        build_substrate_state_receipt(c, '{"a": 1}')
    with pytest.raises(ValueError, match=_ERR_INVALID_CANONICAL_JSON):
        build_substrate_state_receipt(c, '{"a":NaN}')
    with pytest.raises(ValueError, match=_ERR_CANONICAL_JSON_TOO_LARGE):
        build_substrate_state_receipt(c, '"' + ('a' * 1_000_001) + '"')
    text = open("src/qec/analysis/substrate_state_receipt.py", encoding="utf-8").read().lower()
    for forbidden in ["materialencodingreceipt", "substratedriftreceipt", "importlib", "subprocess", "exec(", "eval(", "gpu", "hardware", "global_truth"]:
        assert forbidden not in text
