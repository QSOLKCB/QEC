from dataclasses import FrozenInstanceError, replace

import pytest

from qec.analysis.canonical_hashing import canonical_json, sha256_hex
from qec.analysis.material_encoding_receipt import (
    EncodingEntry,
    MaterialEncodingReceipt,
    SubstrateDriftReceipt,
    build_encoding_entry,
    build_material_encoding_receipt,
    build_substrate_drift_receipt,
    validate_encoding_entry,
    validate_encoding_entry_with_state,
    validate_material_encoding_receipt,
    validate_material_encoding_receipt_with_state,
    validate_substrate_drift_receipt,
    validate_substrate_drift_receipt_with_materials,
)
from qec.analysis.substrate_constraint_contract import build_substrate_constraint_predicate, build_substrate_contract
from qec.analysis.substrate_state_receipt import build_substrate_state_receipt


def _state(payload=None):
    payload = {"a": "x", "n": 2} if payload is None else payload
    cjson = canonical_json(payload)
    p1 = build_substrate_constraint_predicate("P_A", "STRING_EQUALS", ["a"], {"value": "x"})
    p2 = build_substrate_constraint_predicate("P_N", "INTEGER_RANGE", ["n"], {"min_value": 3, "max_value": 9})
    contract = build_substrate_contract("CANONICAL_JSON", sha256_hex({"canonical_json": cjson}), "PROFILE_V1582", [p1, p2])
    return build_substrate_state_receipt(contract, cjson)


def test_encoding_entry_basics_and_validation():
    s = _state(); r0, r1 = s.predicate_evaluation_results
    e0 = build_encoding_entry(s, r0, 0); e0b = build_encoding_entry(s, r0, 0); e1 = build_encoding_entry(s, r1, 1)
    assert e0.encoding_entry_hash == e0b.encoding_entry_hash
    assert e0.encoded_status == "ENCODED_PASS" and e1.encoded_status == "ENCODED_FAIL"
    assert e0.encoded_value_hash == e0b.encoded_value_hash
    assert validate_encoding_entry(e0)
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"): validate_encoding_entry(replace(e0, encoding_entry_hash="A" * 64))
    with pytest.raises(ValueError, match="HASH_MISMATCH"): validate_encoding_entry(replace(e0, encoding_entry_hash="0" * 64))
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"): validate_encoding_entry(replace(e0, encoded_value_hash="A" * 64))
    with pytest.raises(ValueError, match="INVALID_ENCODED_STATUS"): validate_encoding_entry(replace(e0, encoded_status="BAD"))
    with pytest.raises(ValueError, match="ENCODING_INDEX_OUT_OF_BOUNDS"): validate_encoding_entry(replace(e0, encoding_index=True))
    with pytest.raises(ValueError, match="ENCODING_INDEX_OUT_OF_BOUNDS"): validate_encoding_entry(replace(e0, encoding_index=1000))
    with pytest.raises(ValueError, match="INVALID_ENCODING_LABEL"): validate_encoding_entry(replace(e0, encoding_label="bad-label"))
    s_other = _state({"a": "x", "n": 4})
    with pytest.raises(ValueError):
        build_encoding_entry(s, s_other.predicate_evaluation_results[0], 0)
    with pytest.raises(FrozenInstanceError): e0.encoding_label = "X"
    assert e0.to_canonical_json() == e0b.to_canonical_json() and e0.to_canonical_bytes() == e0b.to_canonical_bytes()


def test_material_encoding_receipt_basics_and_validation():
    s = _state(); m = build_material_encoding_receipt(s); m2 = build_material_encoding_receipt(s)
    assert m.material_encoding_receipt_hash == m2.material_encoding_receipt_hash
    assert len(m.encoding_entries) == len(s.predicate_evaluation_results)
    assert validate_material_encoding_receipt(m)
    assert m.material_encoding_class == "MATERIAL_ENCODING_INCOMPATIBLE"
    good = m.encoding_entries
    with pytest.raises(ValueError, match="ENCODING_ORDER_MISMATCH"): MaterialEncodingReceipt(**{**m.__dict__, "encoding_entries": (good[1], good[0])})
    with pytest.raises(ValueError, match="DUPLICATE_ENCODING_ENTRY"): MaterialEncodingReceipt(**{**m.__dict__, "encoding_entries": (good[0], good[0])})
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"): validate_material_encoding_receipt(replace(m, material_encoding_receipt_hash="A" * 64))
    with pytest.raises(ValueError, match="HASH_MISMATCH"): validate_material_encoding_receipt(replace(m, material_encoding_receipt_hash="0" * 64))
    with pytest.raises(ValueError, match="INVALID_ENCODING_MODE"): validate_material_encoding_receipt(replace(m, encoding_mode="BAD"))
    with pytest.raises(ValueError, match="MATERIAL_ENCODING_CLASS_MISMATCH"): validate_material_encoding_receipt(replace(m, material_encoding_class="MATERIAL_ENCODING_COMPATIBLE"))
    with pytest.raises(ValueError): validate_material_encoding_receipt(replace(m, encoding_entries=(replace(good[0], encoding_label="BROKEN_LABEL", encoding_entry_hash=good[0].encoding_entry_hash), good[1])))
    with pytest.raises(FrozenInstanceError): m.encoding_entry_count = 7
    assert m.to_canonical_json() == m2.to_canonical_json() and m.to_canonical_bytes() == m2.to_canonical_bytes()


def test_substrate_drift_receipt_basics_and_validation():
    s = _state(); expected = build_material_encoding_receipt(s); observed = build_material_encoding_receipt(s)
    d = build_substrate_drift_receipt(expected, observed); assert d.substrate_drift_class == "SUBSTRATE_DRIFT_CLEAN"
    assert build_substrate_drift_receipt(expected, None).substrate_drift_class == "SUBSTRATE_DRIFT_INCOMPLETE"
    e0 = observed.encoding_entries[0]
    changed_e0_payload = {**e0.to_dict(), "encoded_status": "ENCODED_FAIL"}
    changed_e0_payload.pop("encoding_entry_hash")
    changed_e0 = EncodingEntry(**changed_e0_payload, encoding_entry_hash=sha256_hex(changed_e0_payload))
    changed_entries = (changed_e0, observed.encoding_entries[1])
    changed_payload = {**observed.__dict__, "encoding_entries": changed_entries, "passed_encoding_count": 0, "failed_encoding_count": 2}
    changed_payload["material_encoding_receipt_hash"] = sha256_hex({
        "substrate_state_receipt_hash": observed.substrate_state_receipt_hash,
        "substrate_contract_hash": observed.substrate_contract_hash,
        "substrate_profile_id": observed.substrate_profile_id,
        "encoding_mode": observed.encoding_mode,
        "encoding_entries": [x.to_dict() for x in changed_entries],
        "encoding_entry_count": 2,
        "passed_encoding_count": 0,
        "failed_encoding_count": 2,
        "material_encoding_class": "MATERIAL_ENCODING_INCOMPATIBLE",
    })
    changed_obs = MaterialEncodingReceipt(**changed_payload)
    assert build_substrate_drift_receipt(expected, changed_obs).substrate_drift_class == "SUBSTRATE_DRIFT_CHANGED"
    d2 = build_substrate_drift_receipt(expected, MaterialEncodingReceipt(**{**expected.__dict__}))
    assert validate_substrate_drift_receipt(d2)
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"): validate_substrate_drift_receipt(replace(d, substrate_drift_receipt_hash="A" * 64))
    with pytest.raises(ValueError, match="HASH_MISMATCH"): validate_substrate_drift_receipt(replace(d, substrate_drift_receipt_hash="0" * 64))
    with pytest.raises(ValueError, match="INVALID_SUBSTRATE_DRIFT_CLASS"): validate_substrate_drift_receipt(replace(d, substrate_drift_class="BAD"))
    with pytest.raises(ValueError, match="SUBSTRATE_DRIFT_CLASS_MISMATCH"): validate_substrate_drift_receipt(replace(d, substrate_drift_class="SUBSTRATE_DRIFT_CHANGED"))
    with pytest.raises(FrozenInstanceError): d.drift_count = 1


def test_complete_validators_and_boundaries_and_scope_scan():
    s = _state(); e = build_encoding_entry(s, s.predicate_evaluation_results[0], 0); m = build_material_encoding_receipt(s); d = build_substrate_drift_receipt(m, m)
    assert validate_encoding_entry_with_state(e, s, s.predicate_evaluation_results[0])
    assert validate_material_encoding_receipt_with_state(m, s)
    assert validate_substrate_drift_receipt_with_materials(d, m, m)
    with pytest.raises(ValueError): validate_encoding_entry(object())
    with pytest.raises(ValueError): validate_material_encoding_receipt(object())
    with pytest.raises(ValueError): validate_substrate_drift_receipt(object())
    assert isinstance(m.to_dict()["encoding_entries"], list)
    assert isinstance(d.to_dict()["expected_encoding_entry_hashes"], list)
    with open("src/qec/analysis/material_encoding_receipt.py", "r", encoding="utf-8") as f:
        txt = f.read().lower()
    for token in ["layersubstratecompatibilityreceipt", "masksubstratereceipt", "routersubstratereceipt", "readoutsubstratereceipt", "hardware", "cpu", "gpu", "device", "physical_substrate", "gameplay", "render", "step_world", "execute_action", "run_game", "importlib", "__import__(", "subprocess", "exec(", "eval(", "random", "time.time", "datetime.now", "probability", "probabilistic", "neural", "learned_policy", "recursive", "global_truth"]:
        assert token not in txt
