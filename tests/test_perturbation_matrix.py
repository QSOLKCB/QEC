from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import pytest

from qec.analysis.canonical_hashing import canonical_json, sha256_hex
from qec.analysis.perturbation_contract import build_perturbation_contract, apply_perturbation_contract
from qec.analysis.perturbation_matrix import (
    _ERR_DUPLICATE_MATRIX_ENTRY,_ERR_ENERGY_RECEIPT_MISMATCH,_ERR_ENTRY_CONTRACT_RESULT_MISMATCH,
    _ERR_ENTRY_OPERATION_MISMATCH,_ERR_ENTRY_TARGET_MISMATCH,_ERR_HASH_MISMATCH,_ERR_IMPACT_SCORE_MISMATCH,
    _ERR_INVALID_ENTRY_LABEL,_ERR_INVALID_HASH_FORMAT,_ERR_INVALID_INPUT,_ERR_MATRIX_COUNT_MISMATCH,
    _ERR_MATRIX_DIMENSION_MISMATCH,_ERR_MATRIX_INDEX_OUT_OF_BOUNDS,_ERR_MATRIX_ORDER_MISMATCH,
    _MAX_TARGET_ARTIFACT_TYPE_LENGTH,
    PerturbationMatrix, PerturbationMatrixEntry, build_energy_matrix_receipt, build_perturbation_matrix, build_perturbation_matrix_entry,
    validate_energy_matrix_receipt, validate_perturbation_matrix, validate_perturbation_matrix_entry,
    validate_perturbation_matrix_entry_with_contract_result,
)

H = "a" * 64


def _mk(operation_type: str, params: dict, path: list[str], original: str, row: int, col: int, label: str):
    c = build_perturbation_contract("Artifact", H, path, operation_type, params)
    r = apply_perturbation_contract(c, original)
    e = build_perturbation_matrix_entry(row, col, label, c, r)
    return c, r, e


def _make_rehashed_matrix(matrix: PerturbationMatrix, **overrides) -> PerturbationMatrix:
    fields = {
        "matrix_label": matrix.matrix_label,
        "matrix_mode": matrix.matrix_mode,
        "entries": matrix.entries,
        "row_count": matrix.row_count,
        "column_count": matrix.column_count,
        "entry_count": matrix.entry_count,
        "changed_entry_count": matrix.changed_entry_count,
        "unchanged_entry_count": matrix.unchanged_entry_count,
    }
    fields.update(overrides)
    return PerturbationMatrix(**fields, perturbation_matrix_hash=sha256_hex(PerturbationMatrix(**fields, perturbation_matrix_hash=matrix.perturbation_matrix_hash)._hash_payload()))


def _make_rehashed_receipt(receipt, **overrides):
    fields = {
        "perturbation_matrix_hash": receipt.perturbation_matrix_hash,
        "matrix_label": receipt.matrix_label,
        "matrix_mode": receipt.matrix_mode,
        "total_integer_impact_score": receipt.total_integer_impact_score,
        "changed_entry_count": receipt.changed_entry_count,
        "unchanged_entry_count": receipt.unchanged_entry_count,
        "operation_type_counts": receipt.operation_type_counts,
        "target_artifact_type_counts": receipt.target_artifact_type_counts,
        "perturbation_matrix": receipt.perturbation_matrix,
    }
    fields.update(overrides)
    return type(receipt)(**fields, energy_matrix_receipt_hash=sha256_hex(type(receipt)(**fields, energy_matrix_receipt_hash=receipt.energy_matrix_receipt_hash)._hash_payload()))


def test_entry_basics_and_validation_edges():
    original = canonical_json({"a": 1, "b": True})
    c1, r1, e1 = _mk("REPLACE_VALUE", {"value": 2}, ["a"], original, 0, 0, "ENTRY_A")
    _, r2, e2 = _mk("REPLACE_VALUE", {"value": 1}, ["a"], original, 0, 1, "ENTRY_B")
    assert e1.perturbation_matrix_entry_hash == build_perturbation_matrix_entry(0, 0, "ENTRY_A", c1, r1).perturbation_matrix_entry_hash
    assert e1.integer_impact_score == 1
    assert r2.changed is False and e2.integer_impact_score == 0
    with pytest.raises(ValueError, match=_ERR_INVALID_HASH_FORMAT):
        validate_perturbation_matrix_entry(replace(e1, perturbation_matrix_entry_hash="bad"))
    with pytest.raises(ValueError, match=_ERR_HASH_MISMATCH):
        validate_perturbation_matrix_entry(replace(e1, perturbation_matrix_entry_hash="b" * 64))
    with pytest.raises(ValueError, match=_ERR_INVALID_HASH_FORMAT):
        build_perturbation_matrix_entry(0, 0, "ENTRY_X", replace(c1, perturbation_contract_hash="bad"), r1)
    with pytest.raises(ValueError, match=_ERR_INVALID_HASH_FORMAT):
        build_perturbation_matrix_entry(0, 0, "ENTRY_X", c1, replace(r1, perturbation_result_hash="bad"))
    for rr, cc in ((True, 0), (0, False), (-1, 0), (0, 1000)):
        with pytest.raises(ValueError, match=_ERR_MATRIX_INDEX_OUT_OF_BOUNDS):
            build_perturbation_matrix_entry(rr, cc, "ENTRY_X", c1, r1)
    with pytest.raises(ValueError, match=_ERR_INVALID_ENTRY_LABEL):
        build_perturbation_matrix_entry(0, 0, "bad", c1, r1)
    c1_other = build_perturbation_contract("Artifact", H, ["a"], "REPLACE_VALUE", {"value": 3})
    with pytest.raises(ValueError, match=_ERR_ENTRY_CONTRACT_RESULT_MISMATCH):
        build_perturbation_matrix_entry(0, 0, "ENTRY_X", c1_other, r1)
    c_target = build_perturbation_contract("Other", H, ["a"], "REPLACE_VALUE", {"value": 2})
    r_target = apply_perturbation_contract(c_target, original)
    with pytest.raises(ValueError):
        build_perturbation_matrix_entry(0, 0, "ENTRY_X", c_target, r1)
    with pytest.raises(ValueError):
        build_perturbation_matrix_entry(0, 0, "ENTRY_X", c1, r_target)
    with pytest.raises(ValueError, match=_ERR_IMPACT_SCORE_MISMATCH):
        validate_perturbation_matrix_entry(replace(e1, integer_impact_score=0, perturbation_matrix_entry_hash=e1.perturbation_matrix_entry_hash))
    with pytest.raises(ValueError, match=_ERR_IMPACT_SCORE_MISMATCH):
        validate_perturbation_matrix_entry(replace(e1, changed=False, perturbation_matrix_entry_hash=e1.perturbation_matrix_entry_hash))
    with pytest.raises(FrozenInstanceError):
        e1.entry_label = "X"
    assert e1.to_canonical_json().encode("utf-8") == e1.to_canonical_bytes()


def test_matrix_basics_and_tamper():
    original = canonical_json({"a": 1, "b": True, "o": {"n": 1}})
    _, _, e1 = _mk("REPLACE_VALUE", {"value": 2}, ["a"], original, 1, 1, "ENTRY_Z")
    _, _, e2 = _mk("BOOLEAN_FLIP", {}, ["b"], original, 0, 0, "ENTRY_A")
    m1 = build_perturbation_matrix("MATRIX_A", [e1, e2])
    m2 = build_perturbation_matrix("MATRIX_A", [e2, e1])
    assert m1.perturbation_matrix_hash == m2.perturbation_matrix_hash
    assert [e.entry_label for e in m1.entries] == ["ENTRY_A", "ENTRY_Z"]
    with pytest.raises(ValueError, match=_ERR_MATRIX_ORDER_MISMATCH):
        PerturbationMatrix("MATRIX_A", "CANONICAL_PERTURBATION_MATRIX", (e1, e2), 2, 2, 2, 2, 0, "b" * 64)
    e_dup_coord = build_perturbation_matrix_entry(1, 1, "ENTRY_B", build_perturbation_contract("Artifact", H, ["a"], "REPLACE_VALUE", {"value": 2}), apply_perturbation_contract(build_perturbation_contract("Artifact", H, ["a"], "REPLACE_VALUE", {"value": 2}), original))
    with pytest.raises(ValueError, match=_ERR_DUPLICATE_MATRIX_ENTRY):
        build_perturbation_matrix("MATRIX_A", [e1, e_dup_coord])
    e_dup_label = build_perturbation_matrix_entry(2, 2, "ENTRY_Z", build_perturbation_contract("Artifact", H, ["a"], "REPLACE_VALUE", {"value": 2}), apply_perturbation_contract(build_perturbation_contract("Artifact", H, ["a"], "REPLACE_VALUE", {"value": 2}), original))
    with pytest.raises(ValueError, match=_ERR_DUPLICATE_MATRIX_ENTRY):
        build_perturbation_matrix("MATRIX_A", [e1, e_dup_label])
    with pytest.raises(ValueError, match=_ERR_INVALID_INPUT):
        build_perturbation_matrix("MATRIX_A", [])
    import qec.analysis.perturbation_matrix as pm
    old = pm._MAX_MATRIX_ENTRIES
    pm._MAX_MATRIX_ENTRIES = 1
    try:
        with pytest.raises(ValueError, match=_ERR_MATRIX_COUNT_MISMATCH):
            build_perturbation_matrix("MATRIX_A", [e1, e2])
    finally:
        pm._MAX_MATRIX_ENTRIES = old
    with pytest.raises(ValueError, match=_ERR_MATRIX_DIMENSION_MISMATCH):
        validate_perturbation_matrix(replace(m1, row_count=99))
    with pytest.raises(ValueError, match=_ERR_MATRIX_DIMENSION_MISMATCH):
        validate_perturbation_matrix(replace(m1, column_count=99))
    with pytest.raises(ValueError, match=_ERR_MATRIX_COUNT_MISMATCH):
        validate_perturbation_matrix(replace(m1, entry_count=99))
    with pytest.raises(ValueError, match=_ERR_MATRIX_COUNT_MISMATCH):
        validate_perturbation_matrix(replace(m1, changed_entry_count=99))
    with pytest.raises(ValueError, match=_ERR_MATRIX_COUNT_MISMATCH):
        validate_perturbation_matrix(replace(m1, unchanged_entry_count=99))
    with pytest.raises(ValueError, match=_ERR_INVALID_HASH_FORMAT):
        validate_perturbation_matrix(replace(m1, perturbation_matrix_hash="bad"))
    with pytest.raises(ValueError, match=_ERR_HASH_MISMATCH):
        validate_perturbation_matrix(replace(m1, perturbation_matrix_hash="b" * 64))
    with pytest.raises(ValueError, match=_ERR_IMPACT_SCORE_MISMATCH):
        PerturbationMatrixEntry(**{**e1.to_dict(), "integer_impact_score": 0, "perturbation_matrix_entry_hash": sha256_hex({**e1._hash_payload(), "integer_impact_score": 0})})
    with pytest.raises(FrozenInstanceError):
        m1.matrix_label = "X"
    assert m1.to_canonical_json().encode("utf-8") == m1.to_canonical_bytes()


def test_receipt_and_complete_validator_and_scope_scan():
    original = canonical_json({"a": 1, "b": True})
    c1, r1, e1 = _mk("REPLACE_VALUE", {"value": 2}, ["a"], original, 0, 0, "ENTRY_A")
    c2, r2, e2 = _mk("BOOLEAN_FLIP", {}, ["b"], original, 0, 1, "ENTRY_B")
    m = build_perturbation_matrix("MATRIX_A", [e1, e2])
    rec1 = build_energy_matrix_receipt(m)
    rec2 = build_energy_matrix_receipt(m)
    assert rec1.energy_matrix_receipt_hash == rec2.energy_matrix_receipt_hash
    assert rec1.total_integer_impact_score == sum(e.integer_impact_score for e in m.entries)
    assert rec1.changed_entry_count == m.changed_entry_count
    assert rec1.unchanged_entry_count == m.unchanged_entry_count
    assert rec1.operation_type_counts == tuple(sorted(rec1.operation_type_counts, key=lambda x: x[0]))
    assert rec1.target_artifact_type_counts == tuple(sorted(rec1.target_artifact_type_counts, key=lambda x: x[0]))
    with pytest.raises(ValueError, match=_ERR_INVALID_HASH_FORMAT):
        validate_energy_matrix_receipt(replace(rec1, energy_matrix_receipt_hash="bad"))
    with pytest.raises(ValueError, match=_ERR_HASH_MISMATCH):
        validate_energy_matrix_receipt(replace(rec1, energy_matrix_receipt_hash="b" * 64))
    with pytest.raises(ValueError, match=_ERR_IMPACT_SCORE_MISMATCH):
        validate_energy_matrix_receipt(replace(rec1, total_integer_impact_score=999))
    with pytest.raises(ValueError, match=_ERR_INVALID_INPUT):
        validate_energy_matrix_receipt(replace(rec1, operation_type_counts=(("X", 1),)))
    with pytest.raises(ValueError, match=_ERR_ENERGY_RECEIPT_MISMATCH):
        validate_energy_matrix_receipt(replace(rec1, target_artifact_type_counts=(("X", 1),)))
    with pytest.raises(ValueError, match=_ERR_IMPACT_SCORE_MISMATCH):
        validate_energy_matrix_receipt(replace(rec1, perturbation_matrix=replace(m, entries=(replace(e1, integer_impact_score=0, perturbation_matrix_entry_hash=e1.perturbation_matrix_entry_hash), e2))))
    with pytest.raises(FrozenInstanceError):
        rec1.matrix_label = "X"
    assert rec1.to_canonical_json().encode("utf-8") == rec1.to_canonical_bytes()

    assert validate_perturbation_matrix_entry_with_contract_result(e1, c1, r1) is True
    with pytest.raises(ValueError, match=_ERR_ENTRY_CONTRACT_RESULT_MISMATCH):
        validate_perturbation_matrix_entry_with_contract_result(e1, c2, r1)
    with pytest.raises(ValueError, match=_ERR_ENTRY_CONTRACT_RESULT_MISMATCH):
        validate_perturbation_matrix_entry_with_contract_result(e1, c1, r2)
    changed_target = PerturbationMatrixEntry(**{**e1.to_dict(), "target_artifact_hash": "b" * 64, "perturbation_matrix_entry_hash": sha256_hex({**e1._hash_payload(), "target_artifact_hash": "b" * 64})})
    rehashed_target = changed_target
    with pytest.raises(ValueError, match=_ERR_ENTRY_CONTRACT_RESULT_MISMATCH):
        validate_perturbation_matrix_entry_with_contract_result(rehashed_target, c1, r1)
    with pytest.raises(ValueError, match=_ERR_ENTRY_CONTRACT_RESULT_MISMATCH):
        validate_perturbation_matrix_entry_with_contract_result(e2, c1, r1)

    import qec.analysis.perturbation_matrix as pm_module
    with open(pm_module.__file__, "r", encoding="utf-8") as f:
        text = f.read()
    banned = ["SemanticStressReceipt", "PerturbationStabilityProof", "SubstrateContract", "RecursiveProofReceipt", "RealityLoopProofReceipt", "GlobalTruthReceipt", "gameplay", "render", "step_world", "execute_action", "run_game", "importlib", "__import__(", "subprocess", "exec(", "eval(", "random", "time.time", "datetime.now", "probability", "probabilistic", "neural", "learned_policy", "physical_energy", "joule", "voltage", "current"]
    for token in banned:
        assert token not in text


def test_hardening_matrix_and_receipt_validation_edges():
    original = canonical_json({"a": 1, "b": True})
    _, _, e1 = _mk("REPLACE_VALUE", {"value": 2}, ["a"], original, 0, 0, "ENTRY_A")
    m = build_perturbation_matrix("MATRIX_A", [e1])
    r = build_energy_matrix_receipt(m)

    with pytest.raises(ValueError, match=_ERR_INVALID_INPUT):
        PerturbationMatrix("MATRIX_A", "CANONICAL_PERTURBATION_MATRIX", (object(),), 1, 1, 1, 1, 0, "b" * 64)
    for k, v in (("row_count", True), ("column_count", True), ("entry_count", True), ("changed_entry_count", True), ("unchanged_entry_count", False)):
        with pytest.raises(ValueError, match=_ERR_INVALID_INPUT):
            validate_perturbation_matrix(_make_rehashed_matrix(m, **{k: v}))
    for k, v in (("changed_entry_count", True), ("unchanged_entry_count", False)):
        with pytest.raises(ValueError, match=_ERR_INVALID_INPUT):
            validate_energy_matrix_receipt(_make_rehashed_receipt(r, **{k: v}))
    with pytest.raises(ValueError, match=_ERR_INVALID_INPUT):
        validate_energy_matrix_receipt(_make_rehashed_receipt(r, operation_type_counts=(("REPLACE_VALUE", True),)))
    with pytest.raises(ValueError, match=_ERR_INVALID_INPUT):
        validate_energy_matrix_receipt(_make_rehashed_receipt(r, target_artifact_type_counts=(("Artifact", True),)))

    bad_operation_shapes = [
        [("REPLACE_VALUE", 1)],
        (["REPLACE_VALUE", 1],),
        (("REPLACE_VALUE",),),
        ((1, 1),),
        (("REPLACE_VALUE", 0),),
        (("REPLACE_VALUE", -1),),
        (("ZZ", 1),),
        (("REPLACE_VALUE", 1), ("REPLACE_VALUE", 1)),
    ]
    for value in bad_operation_shapes:
        with pytest.raises(ValueError, match=_ERR_INVALID_INPUT):
            validate_energy_matrix_receipt(_make_rehashed_receipt(r, operation_type_counts=value))
    with pytest.raises(ValueError, match=_ERR_INVALID_INPUT):
        validate_energy_matrix_receipt(_make_rehashed_receipt(r, operation_type_counts=(("REPLACE_VALUE", 1), ("ADD_FIELD", 1))))

    bad_target_shapes = [
        [("Artifact", 1)],
        (["Artifact", 1],),
        (("Artifact",),),
        ((1, 1),),
        (("", 1),),
        (("Artifact", 0),),
        (("Artifact", -1),),
        (("Artifact", 1), ("Artifact", 1)),
        # Invalid artifact type format (spaces, punctuation, too long)
        (("Invalid Type", 1),),
        (("invalid-type", 1),),
        (("123invalid", 1),),
        (("a" * (_MAX_TARGET_ARTIFACT_TYPE_LENGTH + 1), 1),),  # exceeds max length
    ]
    for value in bad_target_shapes:
        with pytest.raises(ValueError, match=_ERR_INVALID_INPUT):
            validate_energy_matrix_receipt(_make_rehashed_receipt(r, target_artifact_type_counts=value))

    upper = "A" * 64
    with pytest.raises(ValueError, match=_ERR_INVALID_HASH_FORMAT):
        validate_perturbation_matrix_entry(PerturbationMatrixEntry(**{**e1.to_dict(), "perturbation_contract_hash": upper, "perturbation_matrix_entry_hash": sha256_hex({**e1._hash_payload(), "perturbation_contract_hash": upper})}))
    with pytest.raises(ValueError, match=_ERR_INVALID_HASH_FORMAT):
        validate_perturbation_matrix_entry(PerturbationMatrixEntry(**{**e1.to_dict(), "perturbation_result_hash": upper, "perturbation_matrix_entry_hash": sha256_hex({**e1._hash_payload(), "perturbation_result_hash": upper})}))
    with pytest.raises(ValueError, match=_ERR_INVALID_HASH_FORMAT):
        validate_perturbation_matrix_entry(PerturbationMatrixEntry(**{**e1.to_dict(), "target_artifact_hash": upper, "perturbation_matrix_entry_hash": sha256_hex({**e1._hash_payload(), "target_artifact_hash": upper})}))
    with pytest.raises(ValueError, match=_ERR_INVALID_HASH_FORMAT):
        validate_perturbation_matrix_entry(replace(e1, perturbation_matrix_entry_hash=upper))
    with pytest.raises(ValueError, match=_ERR_INVALID_HASH_FORMAT):
        validate_perturbation_matrix(replace(m, perturbation_matrix_hash=upper))
    with pytest.raises(ValueError, match=_ERR_INVALID_HASH_FORMAT):
        validate_energy_matrix_receipt(_make_rehashed_receipt(r, perturbation_matrix_hash=upper))
    with pytest.raises(ValueError, match=_ERR_INVALID_HASH_FORMAT):
        validate_energy_matrix_receipt(replace(r, energy_matrix_receipt_hash=upper))

    with pytest.raises(ValueError, match=_ERR_ENERGY_RECEIPT_MISMATCH):
        validate_energy_matrix_receipt(_make_rehashed_receipt(r, perturbation_matrix_hash="b" * 64))
    with pytest.raises(ValueError, match=_ERR_ENERGY_RECEIPT_MISMATCH):
        validate_energy_matrix_receipt(_make_rehashed_receipt(r, operation_type_counts=()))
    with pytest.raises(ValueError, match=_ERR_ENERGY_RECEIPT_MISMATCH):
        validate_energy_matrix_receipt(_make_rehashed_receipt(r, operation_type_counts=(("REPLACE_VALUE", 2),)))
    with pytest.raises(ValueError, match=_ERR_ENERGY_RECEIPT_MISMATCH):
        validate_energy_matrix_receipt(_make_rehashed_receipt(r, operation_type_counts=(("BOOLEAN_FLIP", 1), ("REPLACE_VALUE", 1))))
    with pytest.raises(ValueError, match=_ERR_ENERGY_RECEIPT_MISMATCH):
        validate_energy_matrix_receipt(_make_rehashed_receipt(r, target_artifact_type_counts=()))
    with pytest.raises(ValueError, match=_ERR_ENERGY_RECEIPT_MISMATCH):
        validate_energy_matrix_receipt(_make_rehashed_receipt(r, target_artifact_type_counts=(("Artifact", 2),)))
    # O(1) length cap: max_target_types = entry_count (1), so 2 types exceeds the limit
    with pytest.raises(ValueError, match=_ERR_INVALID_INPUT):
        validate_energy_matrix_receipt(_make_rehashed_receipt(r, target_artifact_type_counts=(("Artifact", 1), ("Other", 1))))
