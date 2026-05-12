from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
import pytest

from qec.analysis.perturbation_contract import (
    PerturbationContract,
    PerturbationResult,
    _ERR_CANONICAL_JSON_TOO_LARGE,
    _ERR_HASH_MISMATCH,
    _ERR_INVALID_CANONICAL_JSON,
    _ERR_INVALID_FIELD_PATH,
    _ERR_INVALID_HASH_FORMAT,
    _ERR_INVALID_INPUT,
    _ERR_INVALID_OPERATION_PARAMETERS,
    _ERR_INVALID_OPERATION_TYPE,
    _ERR_OPERATION_PARAMETER_TOO_LARGE,
    _ERR_INVALID_PERTURBATION_MODE,
    _ERR_INVALID_TARGET_ARTIFACT_TYPE,
    _ERR_INTEGER_BOUND_VIOLATION,
    _ERR_PERTURBATION_CONTRACT_MISMATCH,
    _ERR_PERTURBATION_RESULT_MISMATCH,
    _ERR_TARGET_FIELD_ALREADY_EXISTS,
    _ERR_TARGET_FIELD_NOT_FOUND,
    _ERR_TARGET_FIELD_TYPE_MISMATCH,
    _ERR_TARGET_PARENT_NOT_OBJECT,
    _MAX_ABS_INTEGER_VALUE,
    _MAX_CANONICAL_JSON_BYTES,
    _MAX_OPERATION_PARAMETER_BYTES,
    _canonical_json_text_hash,
    _perturbation_result_payload,
    _MAX_FIELD_PATH_DEPTH,
    build_perturbation_contract,
    apply_perturbation_contract,
    get_allowed_perturbation_operation_types,
    validate_perturbation_contract,
    validate_perturbation_result,
    validate_perturbation_result_with_contract,
)
from qec.analysis.canonical_hashing import canonical_json, sha256_hex

H = "a" * 64



def _make_rehashed_result(base: PerturbationResult, **overrides) -> PerturbationResult:
    fields = {
        "perturbation_contract_hash": base.perturbation_contract_hash,
        "target_artifact_type": base.target_artifact_type,
        "target_artifact_hash": base.target_artifact_hash,
        "target_field_path": base.target_field_path,
        "operation_type": base.operation_type,
        "original_canonical_json_hash": base.original_canonical_json_hash,
        "perturbed_canonical_json": base.perturbed_canonical_json,
        "perturbed_canonical_json_hash": base.perturbed_canonical_json_hash,
        "changed": base.changed,
    }
    fields.update(overrides)
    if "perturbed_canonical_json" in overrides and "perturbed_canonical_json_hash" not in overrides:
        fields["perturbed_canonical_json_hash"] = _canonical_json_text_hash(fields["perturbed_canonical_json"])
    result_hash = sha256_hex(
        _perturbation_result_payload(
            fields["perturbation_contract_hash"],
            fields["target_artifact_type"],
            fields["target_artifact_hash"],
            fields["target_field_path"],
            fields["operation_type"],
            fields["original_canonical_json_hash"],
            fields["perturbed_canonical_json"],
            fields["perturbed_canonical_json_hash"],
            fields["changed"],
        )
    )
    return PerturbationResult(**fields, perturbation_result_hash=result_hash)
def test_contract_basics_and_mutation_and_canonical_exports():
    c1 = build_perturbation_contract("Artifact", H, ["x"], "REPLACE_VALUE", {"value": 1})
    c2 = build_perturbation_contract("Artifact", H, ["x"], "REPLACE_VALUE", {"value": 1})
    assert c1.perturbation_contract_hash == c2.perturbation_contract_hash
    assert c1.to_canonical_json() == c2.to_canonical_json()
    assert c1.to_canonical_bytes() == c2.to_canonical_bytes()
    with pytest.raises(FrozenInstanceError):
        c1.target_artifact_type = "X"

    with pytest.raises(ValueError, match=_ERR_INVALID_HASH_FORMAT):
        build_perturbation_contract("Artifact", "bad", ["x"], "REMOVE_FIELD", {})
    with pytest.raises(ValueError, match=_ERR_INVALID_TARGET_ARTIFACT_TYPE):
        build_perturbation_contract("1bad", H, ["x"], "REMOVE_FIELD", {})
    with pytest.raises(ValueError, match=_ERR_INVALID_OPERATION_TYPE):
        build_perturbation_contract("Artifact", H, ["x"], "NOPE", {})
    with pytest.raises(ValueError, match=_ERR_INVALID_FIELD_PATH):
        build_perturbation_contract("Artifact", H, [], "REMOVE_FIELD", {})
    with pytest.raises(ValueError, match=_ERR_INVALID_FIELD_PATH):
        build_perturbation_contract("Artifact", H, ["x"] * (_MAX_FIELD_PATH_DEPTH + 1), "REMOVE_FIELD", {})
    with pytest.raises(ValueError, match=_ERR_INVALID_INPUT):
        build_perturbation_contract("Artifact", H, ["x"], "REMOVE_FIELD", {}, max_result_bytes=True)

    with pytest.raises(ValueError, match=_ERR_INVALID_PERTURBATION_MODE):
        PerturbationContract(
            target_artifact_type=c1.target_artifact_type,
            target_artifact_hash=c1.target_artifact_hash,
            perturbation_mode="BAD",
            target_field_path=c1.target_field_path,
            operation_type=c1.operation_type,
            canonical_operation_parameters=c1.canonical_operation_parameters,
            operation_parameters_hash=c1.operation_parameters_hash,
            max_result_bytes=c1.max_result_bytes,
            perturbation_contract_hash=c1.perturbation_contract_hash,
        )
    with pytest.raises(ValueError, match=_ERR_HASH_MISMATCH):
        validate_perturbation_contract(replace(c1, operation_parameters_hash="b" * 64))
    with pytest.raises(ValueError, match=_ERR_HASH_MISMATCH):
        validate_perturbation_contract(replace(c1, perturbation_contract_hash="b" * 64))
    with pytest.raises(ValueError, match=_ERR_INVALID_HASH_FORMAT):
        validate_perturbation_contract(replace(c1, perturbation_contract_hash="bad"))


def test_parameter_validation_and_operation_types():
    assert get_allowed_perturbation_operation_types() == frozenset({"REPLACE_VALUE", "ADD_FIELD", "REMOVE_FIELD", "INTEGER_DELTA", "BOOLEAN_FLIP"})
    build_perturbation_contract("Artifact", H, ["x"], "REPLACE_VALUE", {"value": [1, {"k": None}]})
    build_perturbation_contract("Artifact", H, ["x"], "ADD_FIELD", {"value": False})
    build_perturbation_contract("Artifact", H, ["x"], "REMOVE_FIELD", {})
    build_perturbation_contract("Artifact", H, ["x"], "BOOLEAN_FLIP", {})
    build_perturbation_contract("Artifact", H, ["x"], "INTEGER_DELTA", {"delta": 1, "min_value": None, "max_value": 2})
    for p in ({"value": 1, "x": 2}, {}, {"delta": True, "min_value": None, "max_value": None}, {"delta": 1.2, "min_value": None, "max_value": None}, {"value": (1,)}, {"value": {"k": b"x"}}, {"value": {"": 1}}):
        with pytest.raises(ValueError):
            build_perturbation_contract("Artifact", H, ["x"], "REPLACE_VALUE" if "value" in p else "INTEGER_DELTA", p)
    with pytest.raises(ValueError, match=_ERR_INTEGER_BOUND_VIOLATION):
        build_perturbation_contract("Artifact", H, ["x"], "INTEGER_DELTA", {"delta": 10**12, "min_value": None, "max_value": None})


def test_application_and_result_integrity_and_no_mutation_and_scan():
    original = canonical_json({"a": {"b": 1}, "f": True, "n": 4})
    parsed_a = json.loads(original)

    r1 = apply_perturbation_contract(build_perturbation_contract("Artifact", H, ["a", "b"], "REPLACE_VALUE", {"value": 9}), original)
    assert json.loads(r1.perturbed_canonical_json)["a"]["b"] == 9
    apply_perturbation_contract(build_perturbation_contract("Artifact", H, ["a", "c"], "ADD_FIELD", {"value": 2}), original)
    with pytest.raises(ValueError, match=_ERR_TARGET_FIELD_ALREADY_EXISTS):
        apply_perturbation_contract(build_perturbation_contract("Artifact", H, ["a", "b"], "ADD_FIELD", {"value": 2}), original)
    apply_perturbation_contract(build_perturbation_contract("Artifact", H, ["a", "b"], "REMOVE_FIELD", {}), original)
    with pytest.raises(ValueError, match=_ERR_TARGET_FIELD_NOT_FOUND):
        apply_perturbation_contract(build_perturbation_contract("Artifact", H, ["a", "z"], "REMOVE_FIELD", {}), original)
    apply_perturbation_contract(build_perturbation_contract("Artifact", H, ["n"], "INTEGER_DELTA", {"delta": 2, "min_value": 0, "max_value": 10}), original)
    with pytest.raises(ValueError, match=_ERR_TARGET_FIELD_TYPE_MISMATCH):
        apply_perturbation_contract(build_perturbation_contract("Artifact", H, ["f"], "INTEGER_DELTA", {"delta": 1, "min_value": None, "max_value": None}), original)
    with pytest.raises(ValueError, match=_ERR_TARGET_FIELD_TYPE_MISMATCH):
        apply_perturbation_contract(build_perturbation_contract("Artifact", H, ["a"], "INTEGER_DELTA", {"delta": 1, "min_value": None, "max_value": None}), original)
    with pytest.raises(ValueError, match=_ERR_INTEGER_BOUND_VIOLATION):
        apply_perturbation_contract(build_perturbation_contract("Artifact", H, ["n"], "INTEGER_DELTA", {"delta": 2, "min_value": 10, "max_value": 20}), original)
    apply_perturbation_contract(build_perturbation_contract("Artifact", H, ["f"], "BOOLEAN_FLIP", {}), original)
    with pytest.raises(ValueError, match=_ERR_TARGET_FIELD_TYPE_MISMATCH):
        apply_perturbation_contract(build_perturbation_contract("Artifact", H, ["n"], "BOOLEAN_FLIP", {}), original)
    with pytest.raises(ValueError, match=_ERR_TARGET_FIELD_NOT_FOUND):
        apply_perturbation_contract(build_perturbation_contract("Artifact", H, ["x", "y"], "REMOVE_FIELD", {}), original)
    with pytest.raises(ValueError, match=_ERR_TARGET_PARENT_NOT_OBJECT):
        apply_perturbation_contract(build_perturbation_contract("Artifact", H, ["n", "x"], "REMOVE_FIELD", {}), original)
    with pytest.raises(ValueError, match=_ERR_INVALID_CANONICAL_JSON):
        apply_perturbation_contract(build_perturbation_contract("Artifact", H, ["a"], "REMOVE_FIELD", {}), canonical_json([1, 2]))

    assert original == canonical_json(parsed_a)
    assert json.loads(original) == parsed_a
    assert r1.perturbation_result_hash == apply_perturbation_contract(build_perturbation_contract("Artifact", H, ["a", "b"], "REPLACE_VALUE", {"value": 9}), original).perturbation_result_hash
    assert r1.perturbation_result_hash != apply_perturbation_contract(build_perturbation_contract("Artifact", H, ["a", "b"], "REPLACE_VALUE", {"value": 9}), canonical_json({"a": {"b": 2}, "f": True, "n": 4})).perturbation_result_hash

    validate_perturbation_result(r1)
    with pytest.raises(ValueError, match=_ERR_INVALID_HASH_FORMAT):
        validate_perturbation_result(replace(r1, original_canonical_json_hash="bad"))
    with pytest.raises(ValueError, match=_ERR_INVALID_HASH_FORMAT):
        validate_perturbation_result(replace(r1, perturbed_canonical_json_hash="bad"))
    with pytest.raises(ValueError, match=_ERR_INVALID_HASH_FORMAT):
        validate_perturbation_result(replace(r1, perturbation_result_hash="bad"))
    with pytest.raises(ValueError, match=_ERR_HASH_MISMATCH):
        validate_perturbation_result(replace(r1, perturbation_result_hash="b" * 64))
    with pytest.raises(ValueError, match=_ERR_INVALID_CANONICAL_JSON):
        validate_perturbation_result(replace(r1, perturbed_canonical_json='{"z":2, "a":1}'))
    with pytest.raises(ValueError, match=_ERR_CANONICAL_JSON_TOO_LARGE):
        validate_perturbation_result(replace(r1, perturbed_canonical_json="x" * (_MAX_CANONICAL_JSON_BYTES + 1)))

    c = build_perturbation_contract("Artifact", H, ["a", "b"], "REPLACE_VALUE", {"value": 9})
    validate_perturbation_result_with_contract(r1, c, original)
    c_other = build_perturbation_contract("Artifact", H, ["a", "b"], "REPLACE_VALUE", {"value": 8})
    r_other = apply_perturbation_contract(c_other, original)
    with pytest.raises(ValueError, match=_ERR_PERTURBATION_CONTRACT_MISMATCH):
        validate_perturbation_result_with_contract(r_other, c, original)
    with pytest.raises(ValueError):
        validate_perturbation_result_with_contract(replace(r1, target_artifact_hash="c" * 64), c, original)
    with pytest.raises(ValueError):
        validate_perturbation_result_with_contract(replace(r1, target_field_path=("a",)), c, original)
    with pytest.raises(ValueError):
        validate_perturbation_result_with_contract(replace(r1, operation_type="REMOVE_FIELD"), c, original)
    with pytest.raises(ValueError):
        validate_perturbation_result_with_contract(replace(r1, changed=not r1.changed), c, original)
    with pytest.raises(FrozenInstanceError):
        r1.changed = False
    assert r1.to_canonical_json().encode("utf-8") == r1.to_canonical_bytes()

    import qec.analysis.perturbation_contract as pc_module
    with open(pc_module.__file__, "r", encoding="utf-8") as f:
        text = f.read()
    banned = ["EnergyMatrix", "PerturbationMatrix", "SemanticStressReceipt", "PerturbationStabilityProof", "SubstrateContract", "RecursiveProofReceipt", "RealityLoopProofReceipt", "GlobalTruthReceipt", "gameplay", "render", "step_world", "execute_action", "run_game", "importlib", "__import__(", "subprocess", "exec(", "eval(", "random", "time.time", "datetime.now", "probability", "probabilistic", "neural", "learned_policy"]
    for token in banned:
        assert token not in text


def test_oversized_operation_parameters_rejected():
    big_value = "x" * (_MAX_OPERATION_PARAMETER_BYTES + 1)
    with pytest.raises(ValueError, match=_ERR_OPERATION_PARAMETER_TOO_LARGE):
        build_perturbation_contract("Artifact", H, ["x"], "REPLACE_VALUE", {"value": big_value})


def test_integer_delta_min_greater_than_max_rejected():
    with pytest.raises(ValueError, match=_ERR_INTEGER_BOUND_VIOLATION):
        build_perturbation_contract(
            "Artifact", H, ["x"], "INTEGER_DELTA", {"delta": 0, "min_value": 10, "max_value": 5}
        )


def test_integer_delta_bool_bounds_rejected():
    for bad_params in (
        {"delta": 0, "min_value": True, "max_value": None},
        {"delta": 0, "min_value": None, "max_value": False},
    ):
        with pytest.raises(ValueError, match=_ERR_INVALID_OPERATION_PARAMETERS):
            build_perturbation_contract("Artifact", H, ["x"], "INTEGER_DELTA", bad_params)


def test_complete_validator_rejects_wrong_original_json():
    from qec.analysis.canonical_hashing import canonical_json as cj

    original = cj({"a": 1})
    different = cj({"a": 2})
    c = build_perturbation_contract("Artifact", H, ["a"], "REPLACE_VALUE", {"value": 99})
    r = apply_perturbation_contract(c, original)
    with pytest.raises(ValueError, match=_ERR_PERTURBATION_RESULT_MISMATCH):
        validate_perturbation_result_with_contract(r, c, different)


def test_max_result_bytes_range_violations():
    with pytest.raises(ValueError, match=_ERR_INVALID_INPUT):
        build_perturbation_contract("Artifact", H, ["x"], "REMOVE_FIELD", {}, max_result_bytes=0)
    with pytest.raises(ValueError, match=_ERR_INVALID_INPUT):
        build_perturbation_contract(
            "Artifact", H, ["x"], "REMOVE_FIELD", {}, max_result_bytes=_MAX_CANONICAL_JSON_BYTES + 1
        )


def test_validate_contract_rejects_non_canonical_operation_parameters():
    c = build_perturbation_contract("Artifact", H, ["x"], "REPLACE_VALUE", {"value": 1})
    non_canonical = '{ "value": 1}'
    with pytest.raises(ValueError, match=_ERR_INVALID_CANONICAL_JSON):
        validate_perturbation_contract(
            replace(
                c,
                canonical_operation_parameters=non_canonical,
                operation_parameters_hash="b" * 64,
                perturbation_contract_hash="b" * 64,
            )
        )


def test_complete_validator_result_mismatch_rehashed_fields():
    original = canonical_json({"a": 1})
    c = build_perturbation_contract("Artifact", H, ["a"], "REPLACE_VALUE", {"value": 9})
    r = apply_perturbation_contract(c, original)

    with pytest.raises(ValueError, match=_ERR_PERTURBATION_RESULT_MISMATCH):
        validate_perturbation_result_with_contract(_make_rehashed_result(r, target_artifact_hash="b" * 64), c, original)
    with pytest.raises(ValueError, match=_ERR_PERTURBATION_RESULT_MISMATCH):
        validate_perturbation_result_with_contract(_make_rehashed_result(r, target_field_path=("b",)), c, original)
    with pytest.raises(ValueError, match=_ERR_PERTURBATION_RESULT_MISMATCH):
        validate_perturbation_result_with_contract(_make_rehashed_result(r, operation_type="ADD_FIELD"), c, original)
    with pytest.raises(ValueError, match=_ERR_PERTURBATION_RESULT_MISMATCH):
        validate_perturbation_result_with_contract(_make_rehashed_result(r, changed=not r.changed), c, original)
    with pytest.raises(ValueError, match=_ERR_PERTURBATION_RESULT_MISMATCH):
        validate_perturbation_result_with_contract(
            _make_rehashed_result(r, perturbed_canonical_json=canonical_json({"a": 1234})), c, original
        )


def test_complete_validator_contract_specific_max_result_bytes():
    original = canonical_json({"a": 1})
    c = build_perturbation_contract("Artifact", H, ["a"], "REPLACE_VALUE", {"value": 2}, max_result_bytes=12)
    base = apply_perturbation_contract(c, original)
    bigger = _make_rehashed_result(base, perturbed_canonical_json=canonical_json({"a": 1234567890}))
    assert validate_perturbation_result(bigger) is True
    with pytest.raises(ValueError, match=_ERR_CANONICAL_JSON_TOO_LARGE):
        validate_perturbation_result_with_contract(bigger, c, original)


def test_malformed_canonical_json_exact_errors():
    c = build_perturbation_contract("Artifact", H, ["a"], "REPLACE_VALUE", {"value": 1})
    for bad in ("NaN", "Infinity", "-Infinity", '{"x":NaN}'):
        with pytest.raises(ValueError, match=_ERR_INVALID_CANONICAL_JSON):
            apply_perturbation_contract(c, bad)

        with pytest.raises(ValueError, match=_ERR_INVALID_CANONICAL_JSON):
            validate_perturbation_contract(
                replace(c, canonical_operation_parameters=bad, operation_parameters_hash="b" * 64, perturbation_contract_hash="b" * 64)
            )

    r = apply_perturbation_contract(c, canonical_json({"a": 1}))
    for bad in ("NaN", "Infinity", "-Infinity", '{"x":NaN}'):
        with pytest.raises(ValueError, match=_ERR_INVALID_CANONICAL_JSON):
            validate_perturbation_result(replace(r, perturbed_canonical_json=bad))


def test_uppercase_hash_format_rejections():
    up = "A" * 64
    with pytest.raises(ValueError, match=_ERR_INVALID_HASH_FORMAT):
        build_perturbation_contract("Artifact", up, ["x"], "REMOVE_FIELD", {})

    c = build_perturbation_contract("Artifact", H, ["x"], "REPLACE_VALUE", {"value": 1})
    with pytest.raises(ValueError, match=_ERR_INVALID_HASH_FORMAT):
        validate_perturbation_contract(replace(c, operation_parameters_hash=up))
    with pytest.raises(ValueError, match=_ERR_INVALID_HASH_FORMAT):
        validate_perturbation_contract(replace(c, perturbation_contract_hash=up))

    r = apply_perturbation_contract(c, canonical_json({"x": 1}))
    with pytest.raises(ValueError, match=_ERR_INVALID_HASH_FORMAT):
        validate_perturbation_result(replace(r, perturbation_contract_hash=up))
    with pytest.raises(ValueError, match=_ERR_INVALID_HASH_FORMAT):
        validate_perturbation_result(replace(r, target_artifact_hash=up))
    with pytest.raises(ValueError, match=_ERR_INVALID_HASH_FORMAT):
        validate_perturbation_result(replace(r, original_canonical_json_hash=up))
    with pytest.raises(ValueError, match=_ERR_INVALID_HASH_FORMAT):
        validate_perturbation_result(replace(r, perturbed_canonical_json_hash=up))
    with pytest.raises(ValueError, match=_ERR_INVALID_HASH_FORMAT):
        validate_perturbation_result(replace(r, perturbation_result_hash=up))


def test_tuple_and_numeric_bound_coverage():
    with pytest.raises(ValueError, match=_ERR_INVALID_OPERATION_PARAMETERS):
        build_perturbation_contract("Artifact", H, ["x"], "ADD_FIELD", {"value": (1,)})
    with pytest.raises(ValueError, match=_ERR_INVALID_OPERATION_PARAMETERS):
        build_perturbation_contract("Artifact", H, ["x"], "ADD_FIELD", {"value": [1, (2,)]})
    with pytest.raises(ValueError, match=_ERR_INVALID_OPERATION_PARAMETERS):
        build_perturbation_contract("Artifact", H, ["x"], "REPLACE_VALUE", {"value": {"k": (3,)}})

    with pytest.raises(ValueError, match=_ERR_INVALID_OPERATION_PARAMETERS):
        build_perturbation_contract("Artifact", H, ["x"], "INTEGER_DELTA", {"delta": 0, "min_value": 1.1, "max_value": None})
    with pytest.raises(ValueError, match=_ERR_INVALID_OPERATION_PARAMETERS):
        build_perturbation_contract("Artifact", H, ["x"], "INTEGER_DELTA", {"delta": 0, "min_value": None, "max_value": 1.1})
    with pytest.raises(ValueError, match=_ERR_INTEGER_BOUND_VIOLATION):
        build_perturbation_contract("Artifact", H, ["x"], "INTEGER_DELTA", {"delta": 0, "min_value": _MAX_ABS_INTEGER_VALUE + 1, "max_value": None})
    with pytest.raises(ValueError, match=_ERR_INTEGER_BOUND_VIOLATION):
        build_perturbation_contract("Artifact", H, ["x"], "INTEGER_DELTA", {"delta": 0, "min_value": None, "max_value": _MAX_ABS_INTEGER_VALUE + 1})


def test_max_result_bytes_applies_to_result_not_input():
    """Verify that max_result_bytes only constrains the result, not the input.

    A perturbation that shrinks a large input below max_result_bytes should succeed.
    """
    large_value = "x" * 100
    original = canonical_json({"a": large_value})
    original_size = len(original.encode("utf-8"))
    small_max = 20
    assert original_size > small_max

    c = build_perturbation_contract("Artifact", H, ["a"], "REPLACE_VALUE", {"value": 1}, max_result_bytes=small_max)
    r = apply_perturbation_contract(c, original)
    assert len(r.perturbed_canonical_json.encode("utf-8")) <= small_max
    assert json.loads(r.perturbed_canonical_json)["a"] == 1

    c_remove = build_perturbation_contract("Artifact", H, ["a"], "REMOVE_FIELD", {}, max_result_bytes=small_max)
    r_remove = apply_perturbation_contract(c_remove, original)
    assert len(r_remove.perturbed_canonical_json.encode("utf-8")) <= small_max

    c_grow = build_perturbation_contract("Artifact", H, ["a"], "REPLACE_VALUE", {"value": "y" * 200}, max_result_bytes=small_max)
    with pytest.raises(ValueError, match=_ERR_CANONICAL_JSON_TOO_LARGE):
        apply_perturbation_contract(c_grow, original)


def test_validate_contract_rejects_non_string_canonical_operation_parameters():
    """Verify that validate_perturbation_contract rejects non-string canonical_operation_parameters."""
    c = build_perturbation_contract("Artifact", H, ["x"], "REPLACE_VALUE", {"value": 1})
    bad_contract = object.__new__(PerturbationContract)
    object.__setattr__(bad_contract, "target_artifact_type", c.target_artifact_type)
    object.__setattr__(bad_contract, "target_artifact_hash", c.target_artifact_hash)
    object.__setattr__(bad_contract, "perturbation_mode", c.perturbation_mode)
    object.__setattr__(bad_contract, "target_field_path", c.target_field_path)
    object.__setattr__(bad_contract, "operation_type", c.operation_type)
    object.__setattr__(bad_contract, "canonical_operation_parameters", None)
    object.__setattr__(bad_contract, "operation_parameters_hash", c.operation_parameters_hash)
    object.__setattr__(bad_contract, "max_result_bytes", c.max_result_bytes)
    object.__setattr__(bad_contract, "perturbation_contract_hash", c.perturbation_contract_hash)
    with pytest.raises(ValueError, match=_ERR_INVALID_CANONICAL_JSON):
        validate_perturbation_contract(bad_contract)
