from dataclasses import FrozenInstanceError, replace

import pytest

from qec.analysis.substrate_constraint_contract import *


def _valid_hash() -> str:
    return "a" * 64


def _mk_pred(kind: str = "FIELD_PRESENT", pid: str = "P1", path=("root",), params=None):
    return build_substrate_constraint_predicate(pid, kind, path, params)


def test_predicate_basics_and_hash_failures_and_stability():
    p1 = _mk_pred()
    p2 = _mk_pred()
    assert p1.substrate_constraint_predicate_hash == p2.substrate_constraint_predicate_hash
    for k, params, fp in [
        ("FIELD_PRESENT", {}, ("a",)), ("FIELD_ABSENT", {}, ("a",)), ("FIELD_TYPE", {"json_type": "string"}, ("a",)),
        ("STRING_EQUALS", {"value": "x"}, ("a",)), ("STRING_IN_SET", {"allowed_values": ["a", "b"]}, ("a",)),
        ("INTEGER_RANGE", {"min_value": 1, "max_value": 2}, ("a",)), ("BOOLEAN_EQUALS", {"value": True}, ("a",)),
        ("HASH_FORMAT", {}, ("a",)), ("CANONICAL_BYTES_MAX", {"max_bytes": 1}, ()),
    ]:
        assert _mk_pred(k, f"P_{k}", fp, params)
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_substrate_constraint_predicate(replace(p1, predicate_parameters_hash="bad"))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_substrate_constraint_predicate(replace(p1, predicate_parameters_hash="b" * 64))
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_substrate_constraint_predicate(replace(p1, substrate_constraint_predicate_hash="bad"))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_substrate_constraint_predicate(replace(p1, substrate_constraint_predicate_hash="b" * 64))
    for bad in ["", "1BAD", "a" * 97]:
        with pytest.raises(ValueError, match="INVALID_PREDICATE_ID"):
            _mk_pred(pid=bad)
    with pytest.raises(ValueError, match="INVALID_PREDICATE_KIND"):
        _mk_pred(kind="NOPE")
    with pytest.raises(ValueError, match="INVALID_FIELD_PATH"):
        _mk_pred(path=("0",))
    for k in get_allowed_substrate_predicate_kinds() - {"CANONICAL_BYTES_MAX"}:
        with pytest.raises(ValueError, match="INVALID_FIELD_PATH"):
            _mk_pred(kind=k, path=(), params={} if k in {"FIELD_PRESENT", "FIELD_ABSENT", "HASH_FORMAT"} else {"json_type": "string"} if k=="FIELD_TYPE" else {"value":"x"} if k=="STRING_EQUALS" else {"allowed_values":["x"]} if k=="STRING_IN_SET" else {"min_value":None,"max_value":None} if k=="INTEGER_RANGE" else {"value":True})
    assert _mk_pred(kind="CANONICAL_BYTES_MAX", path=(), params={"max_bytes": 2})
    with pytest.raises(ValueError, match="INVALID_PREDICATE_PARAMETERS"):
        replace(p1, canonical_predicate_parameters='{"b":1,"a":2}')
    with pytest.raises(ValueError, match="PREDICATE_PARAMETER_TOO_LARGE"):
        _mk_pred(kind="STRING_EQUALS", params={"value": "x" * 9000})
    with pytest.raises(ValueError, match="INVALID_PREDICATE_PARAMETERS"):
        _mk_pred(params=("x",))
    with pytest.raises(ValueError, match="INVALID_PREDICATE_PARAMETERS"):
        _mk_pred(kind="INTEGER_RANGE", params={"min_value": 1.0, "max_value": 2})
    with pytest.raises(ValueError, match="INVALID_PREDICATE_PARAMETERS"):
        _mk_pred(kind="INTEGER_RANGE", params={"min_value": True, "max_value": 2})
    with pytest.raises(FrozenInstanceError):
        p1.predicate_id = "X"
    assert p1.to_canonical_json() == p2.to_canonical_json()
    assert p1.to_canonical_bytes() == p2.to_canonical_bytes()


def test_predicate_parameter_semantics():
    assert _mk_pred("FIELD_PRESENT", "A", ("x",), {})
    assert _mk_pred("FIELD_ABSENT", "B", ("x",), {})
    for t in ["object", "array", "string", "integer", "boolean", "null"]:
        assert _mk_pred("FIELD_TYPE", f"C{t}", ("x",), {"json_type": t})
    for bad in ["number", 1, None]:
        with pytest.raises(ValueError): _mk_pred("FIELD_TYPE", "CT", ("x",), {"json_type": bad})
    assert _mk_pred("STRING_EQUALS", "D", ("x",), {"value": "ok"})
    with pytest.raises(ValueError): _mk_pred("STRING_EQUALS", "D2", ("x",), {"value": 1})
    assert _mk_pred("STRING_IN_SET", "E", ("x",), {"allowed_values": ["a", "b"]})
    for bad in [{"allowed_values": []}, {"allowed_values": ["a", "a"]}, {"allowed_values": ["b", "a"]}, {"allowed_values": ["a", 1]}]:
        with pytest.raises(ValueError): _mk_pred("STRING_IN_SET", "E2", ("x",), bad)
    assert _mk_pred("INTEGER_RANGE", "F", ("x",), {"min_value": None, "max_value": 3})
    for bad in [
        {"min_value": True, "max_value": 3}, {"min_value": 1.2, "max_value": 3}, {"min_value": 4, "max_value": 3},
        {"min_value": 1_000_000_000_001, "max_value": None},
    ]:
        with pytest.raises(ValueError): _mk_pred("INTEGER_RANGE", "F2", ("x",), bad)
    assert _mk_pred("BOOLEAN_EQUALS", "G", ("x",), {"value": False})
    for bad in [0, 1, "true", None]:
        with pytest.raises(ValueError): _mk_pred("BOOLEAN_EQUALS", "G2", ("x",), {"value": bad})
    assert _mk_pred("HASH_FORMAT", "H", ("x",), {})
    assert _mk_pred("CANONICAL_BYTES_MAX", "I", (), {"max_bytes": 1})
    for bad in [True, 0, -1, 1_000_001]:
        with pytest.raises(ValueError): _mk_pred("CANONICAL_BYTES_MAX", "I2", (), {"max_bytes": bad})
    with pytest.raises(ValueError): _mk_pred("FIELD_PRESENT", "J", ("x",), {"x": 1})
    with pytest.raises(ValueError): _mk_pred("STRING_EQUALS", "K", ("x",), {})


def test_contract_basics_and_validation_paths():
    pz = _mk_pred("STRING_EQUALS", "Z", ("z",), {"value": "z"})
    pa = _mk_pred("FIELD_PRESENT", "A", ("a",), {})
    c1 = build_substrate_contract("PerturbationStabilityProof", _valid_hash(), "ProfileA", [pz, pa])
    c2 = build_substrate_contract("PerturbationStabilityProof", _valid_hash(), "ProfileA", [pa, pz])
    assert c1.substrate_contract_hash == c2.substrate_contract_hash
    assert tuple(p.predicate_id for p in c1.predicates) == ("A", "Z")
    with pytest.raises(ValueError, match="PREDICATE_ORDER_MISMATCH"):
        replace(c1, predicates=(pz, pa), substrate_contract_hash="f" * 64)
    with pytest.raises(ValueError, match="DUPLICATE_PREDICATE"):
        build_substrate_contract("PerturbationStabilityProof", _valid_hash(), "ProfileA", [pa, pa])
    with pytest.raises(ValueError):
        build_substrate_contract("PerturbationStabilityProof", _valid_hash(), "ProfileA", [])
    with pytest.raises(ValueError, match="PREDICATE_COUNT_MISMATCH"):
        validate_substrate_contract(replace(c1, predicate_count=3))
    with pytest.raises(ValueError, match="PREDICATE_COUNT_MISMATCH"):
        validate_substrate_contract(replace(c1, predicate_count=True))
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        build_substrate_contract("PerturbationStabilityProof", "bad", "ProfileA", [pa])
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        build_substrate_contract("PerturbationStabilityProof", "A" * 64, "ProfileA", [pa])
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_substrate_contract(replace(c1, substrate_contract_hash="bad"))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_substrate_contract(replace(c1, substrate_contract_hash="b" * 64))
    with pytest.raises(ValueError, match="INVALID_SOURCE_ARTIFACT_TYPE"):
        build_substrate_contract("", _valid_hash(), "ProfileA", [pa])
    with pytest.raises(ValueError, match="INVALID_SUBSTRATE_PROFILE"):
        build_substrate_contract("PerturbationStabilityProof", _valid_hash(), "", [pa])
    with pytest.raises(ValueError, match="INVALID_SUBSTRATE_MODE"):
        validate_substrate_contract(replace(c1, substrate_mode="OTHER"))
    with pytest.raises(ValueError):
        tampered_p = replace(pa, canonical_predicate_parameters='{"x":1}')
        t_payload = {"source_artifact_type": c1.source_artifact_type, "source_artifact_hash": c1.source_artifact_hash, "substrate_profile_id": c1.substrate_profile_id, "substrate_mode": c1.substrate_mode, "predicates": (tampered_p.to_dict(), pz.to_dict()), "predicate_count": 2}
        t_contract = replace(c1, predicates=(tampered_p, pz), substrate_contract_hash=sha256_hex(t_payload))
        validate_substrate_contract(t_contract)
    with pytest.raises(FrozenInstanceError):
        c1.substrate_profile_id = "B"
    assert c1.to_canonical_json() == c2.to_canonical_json()
    assert c1.to_canonical_bytes() == c2.to_canonical_bytes()


def test_complete_validator_and_scope_boundary_scan():
    p1 = _mk_pred("FIELD_PRESENT", "A", ("a",), {})
    p2 = _mk_pred("BOOLEAN_EQUALS", "B", ("b",), {"value": True})
    c = build_substrate_contract("PerturbationStabilityProof", _valid_hash(), "ProfileA", [p1, p2])
    assert validate_substrate_contract_matches_predicates(c, [p1, p2])
    with pytest.raises(ValueError):
        validate_substrate_contract_matches_predicates(c, [p1])
    c_missing = build_substrate_contract(c.source_artifact_type, c.source_artifact_hash, c.substrate_profile_id, [p1])
    with pytest.raises(ValueError):
        validate_substrate_contract_matches_predicates(c_missing, [p1, p2])
    with pytest.raises(ValueError):
        validate_substrate_contract_matches_predicates(c, [p1, p2, _mk_pred("HASH_FORMAT", "C", ("c",), {})])
    with pytest.raises(ValueError):
        validate_substrate_contract_matches_predicates(replace(c, source_artifact_hash="b" * 64), [p1, p2])
    with pytest.raises(ValueError):
        validate_substrate_contract_matches_predicates(replace(c, substrate_profile_id="ProfileB"), [p1, p2])
    text = open("src/qec/analysis/substrate_constraint_contract.py", "r", encoding="utf-8").read().lower()
    banned = ["predicateevaluationresult","substratestatereceipt","materialencodingreceipt","substratedriftreceipt","layersubstratecompatibilityreceipt","masksubstratereceipt","routersubstratereceipt","readoutsubstratereceipt","hardware","cpu","gpu","device","physical_substrate","gameplay","render","step_world","execute_action","run_game","importlib","__import__(","subprocess","exec(","eval(","random","time.time","datetime.now","probability","probabilistic","neural","learned_policy","recursive","global_truth"]
    assert all(x not in text for x in banned)
