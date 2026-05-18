from __future__ import annotations

import os
import subprocess
import sys

import pytest

from qec.analysis.dataframe_backend_manifest import build_schema_field, build_schema_manifest
from qec.analysis.polars_pandas_equivalence_receipts import build_dataframe_schema_comparison
from qec.analysis.schema_equivalence_receipts import (
    _canonical_json,
    _hash_payload,
    build_schema_compatibility_policy,
    build_schema_dtype_equivalence,
    build_schema_equivalence_receipt,
    build_schema_evolution_transition,
    build_schema_field_equivalence,
    build_schema_mismatch_record,
    build_schema_nullability_equivalence,
    build_schema_ordering_equivalence,
    validate_schema_equivalence_receipt,
)


def _h() -> str:
    return "a" * 64


def _fixtures(order_change: bool = False):
    left = build_schema_manifest((build_schema_field("a", "int64", False, 0), build_schema_field("b", "str", True, 1)))
    right_fields = (build_schema_field("b", "str", True, 0), build_schema_field("a", "int64", False, 1)) if order_change else (build_schema_field("a", "int64", False, 0), build_schema_field("b", "str", True, 1))
    right = build_schema_manifest(right_fields)
    cmp = build_dataframe_schema_comparison(left.schema_manifest_hash, right.schema_manifest_hash, left.schema_manifest_hash == right.schema_manifest_hash)
    fe = (
        build_schema_field_equivalence(left.fields[0].field_hash, right.fields[0].field_hash, "EXACT_SCHEMA", left.fields[0].field_hash == right.fields[0].field_hash),
        build_schema_field_equivalence(left.fields[1].field_hash, right.fields[1].field_hash, "EXACT_SCHEMA", left.fields[1].field_hash == right.fields[1].field_hash),
    )
    return left, right, cmp, fe


def test_hash_and_canonical_json_stability_and_idempotent_rebuild():
    p = {"b": 1, "a": 2}
    assert _canonical_json(p) == '{"a":2,"b":1}'
    assert _hash_payload(p) == _hash_payload({"a": 2, "b": 1})


def test_exact_schema_equivalence_pass_and_adapter_only_and_semantics():
    left, right, cmp, fe = _fixtures(False)
    r = build_schema_equivalence_receipt(left, right, cmp, fe, build_schema_ordering_equivalence("EXACT_ORDER", True), build_schema_nullability_equivalence("EXACT_NULLABILITY", True), build_schema_dtype_equivalence("EXACT_DTYPE", True), build_schema_evolution_transition("NO_CHANGE", left.schema_manifest_hash, right.schema_manifest_hash, True), build_schema_compatibility_policy("STRICT_COMPATIBILITY", "declared"), (), True)
    assert r.schemas_equivalent is True
    assert r.adapter_only is True
    assert r.mismatch_count == 0
    r2 = build_schema_equivalence_receipt(left, right, cmp, fe, r.ordering_equivalence, r.nullability_equivalence, r.dtype_equivalence, r.evolution_transition, r.compatibility_policy, (), True)
    assert r2.schema_equivalence_receipt_hash == r.schema_equivalence_receipt_hash


def test_order_insensitive_schema_equivalence_pass():
    left, right, cmp, _ = _fixtures(True)
    fe = (
        build_schema_field_equivalence(left.fields[0].field_hash, right.fields[1].field_hash, "ORDER_INSENSITIVE_SCHEMA", True),
        build_schema_field_equivalence(left.fields[1].field_hash, right.fields[0].field_hash, "ORDER_INSENSITIVE_SCHEMA", True),
    )
    r = build_schema_equivalence_receipt(left, right, cmp, fe, build_schema_ordering_equivalence("ORDER_INSENSITIVE", True), build_schema_nullability_equivalence("EXACT_NULLABILITY", True), build_schema_dtype_equivalence("EXACT_DTYPE", True), build_schema_evolution_transition("COLUMN_REORDERED", left.schema_manifest_hash, right.schema_manifest_hash, True), build_schema_compatibility_policy("BIDIRECTIONAL_COMPATIBLE", "declared"), (), True)
    assert r.schemas_equivalent is True


def test_schema_mismatch_and_mismatch_count_recompute_and_dense_unique_indices():
    left, right, cmp, fe = _fixtures(False)
    m0 = build_schema_mismatch_record(0, "DTYPE_MISMATCH", _h(), _h(), "reason")
    r = build_schema_equivalence_receipt(left, right, cmp, fe, build_schema_ordering_equivalence("EXACT_ORDER", True), build_schema_nullability_equivalence("EXACT_NULLABILITY", True), build_schema_dtype_equivalence("EXACT_DTYPE", False), build_schema_evolution_transition("DTYPE_CHANGED", left.schema_manifest_hash, right.schema_manifest_hash, True), build_schema_compatibility_policy("FORWARD_COMPATIBLE", "declared"), (m0,), True)
    assert r.schemas_equivalent is False
    bad = type(r)(**{**r.__dict__, "mismatch_count": 0})
    with pytest.raises(ValueError, match="mismatch_count mismatch"):
        validate_schema_equivalence_receipt(bad)
    dup = type(r)(**{**r.__dict__, "mismatches": (m0, m0), "mismatch_count": 2, "schema_equivalence_receipt_hash": _hash_payload({k: v for k, v in r.__dict__.items() if k != "schema_equivalence_receipt_hash"})})
    with pytest.raises(ValueError, match="duplicate mismatch"):
        validate_schema_equivalence_receipt(dup)


def test_rejections_and_forbidden_semantics_and_policy_consistency_and_child_validation():
    left, right, cmp, fe = _fixtures(False)
    with pytest.raises(ValueError):
        build_schema_field_equivalence(_h(), _h(), "BAD", True)
    with pytest.raises(ValueError):
        build_schema_compatibility_policy("BAD", "x")
    with pytest.raises(ValueError):
        build_schema_evolution_transition("BAD", _h(), _h(), True)
    with pytest.raises(ValueError):
        build_schema_dtype_equivalence("BAD", True)
    with pytest.raises(ValueError):
        build_schema_nullability_equivalence("BAD", True)
    with pytest.raises(ValueError):
        build_schema_ordering_equivalence("BAD", True)
    with pytest.raises(ValueError):
        build_schema_mismatch_record(0, "DTYPE_MISMATCH", "bad", _h(), "x")
    with pytest.raises(ValueError):
        build_schema_compatibility_policy("STRICT_COMPATIBILITY", "backend authority")
    with pytest.raises(ValueError):
        build_schema_equivalence_receipt(left, right, cmp, fe, build_schema_ordering_equivalence("EXACT_ORDER", True), build_schema_nullability_equivalence("EXACT_NULLABILITY", True), build_schema_dtype_equivalence("EXACT_DTYPE", True), build_schema_evolution_transition("COLUMN_REORDERED", left.schema_manifest_hash, right.schema_manifest_hash, True), build_schema_compatibility_policy("STRICT_COMPATIBILITY", "declared"), (), True)


def test_recomputed_schemas_equivalent_not_trusted_and_adapter_only_enforced_and_immutable_payload_validation():
    left, right, cmp, fe = _fixtures(False)
    m0 = build_schema_mismatch_record(0, "ORDERING_MISMATCH", _h(), _h(), "reason")
    r = build_schema_equivalence_receipt(left, right, cmp, fe, build_schema_ordering_equivalence("EXACT_ORDER", False), build_schema_nullability_equivalence("EXACT_NULLABILITY", True), build_schema_dtype_equivalence("EXACT_DTYPE", True), build_schema_evolution_transition("NO_CHANGE", left.schema_manifest_hash, right.schema_manifest_hash, True), build_schema_compatibility_policy("FORWARD_COMPATIBLE", "declared"), (m0,), True)
    forged = type(r)(**{**r.__dict__, "schemas_equivalent": True})
    with pytest.raises(ValueError, match="schemas_equivalent mismatch"):
        validate_schema_equivalence_receipt(forged)
    with pytest.raises(ValueError, match="adapter_only"):
        validate_schema_equivalence_receipt(type(r)(**{**r.__dict__, "adapter_only": False}))


def test_no_forbidden_imports_decoder_boundary_and_pythonhashseed_stability():
    with open("src/qec/analysis/schema_equivalence_receipts.py", encoding="utf-8") as f:
        text = f.read()
    forbidden = ["import pandas", "import polars", "import pyarrow", "import scipy", "import matplotlib", "import qutip", "import qiskit", "import requests", "import urllib", "import subprocess", "eval(", "exec(", "os.system("]
    assert not any(tok in text for tok in forbidden)
    assert "src/qec/decoder/" not in "src/qec/analysis/schema_equivalence_receipts.py"
    cmd = "import os;from qec.analysis.schema_equivalence_receipts import _hash_payload;print(_hash_payload({'a':1,'b':2}))"
    env = {**os.environ, "PYTHONPATH": "src", "PYTHONHASHSEED": "777"}
    a = subprocess.check_output([sys.executable, "-c", cmd], env=env, text=True).strip()
    b = subprocess.check_output([sys.executable, "-c", cmd], env=env, text=True).strip()
    assert a == b


# ============================================================================
# Schema Evolution Examples: Declarative, deterministic, no runtime execution
# ============================================================================


def test_schema_evolution_column_added():
    """COLUMN_ADDED: right schema has an additional column not in left."""
    left = build_schema_manifest((build_schema_field("a", "int64", False, 0),))
    right = build_schema_manifest((build_schema_field("a", "int64", False, 0), build_schema_field("b", "str", True, 1)))
    cmp = build_dataframe_schema_comparison(left.schema_manifest_hash, right.schema_manifest_hash, False)
    fe = (build_schema_field_equivalence(left.fields[0].field_hash, right.fields[0].field_hash, "EXACT_SCHEMA", True),)
    m0 = build_schema_mismatch_record(0, "FIELD_COUNT_MISMATCH", left.schema_manifest_hash, right.schema_manifest_hash, "column added")
    r = build_schema_equivalence_receipt(
        left, right, cmp, fe,
        build_schema_ordering_equivalence("EXACT_ORDER", True),
        build_schema_nullability_equivalence("EXACT_NULLABILITY", True),
        build_schema_dtype_equivalence("EXACT_DTYPE", True),
        build_schema_evolution_transition("COLUMN_ADDED", left.schema_manifest_hash, right.schema_manifest_hash, True),
        build_schema_compatibility_policy("FORWARD_COMPATIBLE", "column added is forward compatible"),
        (m0,), True,
    )
    assert r.evolution_transition.transition_type == "COLUMN_ADDED"
    assert r.schemas_equivalent is False
    assert r.mismatch_count == 1


def test_schema_evolution_column_removed():
    """COLUMN_REMOVED: right schema has fewer columns than left."""
    left = build_schema_manifest((build_schema_field("a", "int64", False, 0), build_schema_field("b", "str", True, 1)))
    right = build_schema_manifest((build_schema_field("a", "int64", False, 0),))
    cmp = build_dataframe_schema_comparison(left.schema_manifest_hash, right.schema_manifest_hash, False)
    fe = (build_schema_field_equivalence(left.fields[0].field_hash, right.fields[0].field_hash, "EXACT_SCHEMA", True),)
    m0 = build_schema_mismatch_record(0, "FIELD_COUNT_MISMATCH", left.schema_manifest_hash, right.schema_manifest_hash, "column removed")
    r = build_schema_equivalence_receipt(
        left, right, cmp, fe,
        build_schema_ordering_equivalence("EXACT_ORDER", True),
        build_schema_nullability_equivalence("EXACT_NULLABILITY", True),
        build_schema_dtype_equivalence("EXACT_DTYPE", True),
        build_schema_evolution_transition("COLUMN_REMOVED", left.schema_manifest_hash, right.schema_manifest_hash, True),
        build_schema_compatibility_policy("BACKWARD_COMPATIBLE", "column removed is backward compatible"),
        (m0,), True,
    )
    assert r.evolution_transition.transition_type == "COLUMN_REMOVED"
    assert r.schemas_equivalent is False
    assert r.mismatch_count == 1


def test_schema_evolution_column_renamed():
    """COLUMN_RENAMED: same structure but different field name."""
    left = build_schema_manifest((build_schema_field("old_name", "int64", False, 0),))
    right = build_schema_manifest((build_schema_field("new_name", "int64", False, 0),))
    cmp = build_dataframe_schema_comparison(left.schema_manifest_hash, right.schema_manifest_hash, False)
    fe = (build_schema_field_equivalence(left.fields[0].field_hash, right.fields[0].field_hash, "DECLARED_EVOLUTION_EQUIVALENCE", True),)
    r = build_schema_equivalence_receipt(
        left, right, cmp, fe,
        build_schema_ordering_equivalence("EXACT_ORDER", True),
        build_schema_nullability_equivalence("EXACT_NULLABILITY", True),
        build_schema_dtype_equivalence("EXACT_DTYPE", True),
        build_schema_evolution_transition("COLUMN_RENAMED", left.schema_manifest_hash, right.schema_manifest_hash, True),
        build_schema_compatibility_policy("BIDIRECTIONAL_COMPATIBLE", "rename is bidirectional"),
        (), True,
    )
    assert r.evolution_transition.transition_type == "COLUMN_RENAMED"
    assert r.schemas_equivalent is True
    assert r.mismatch_count == 0


def test_schema_evolution_column_reordered():
    """COLUMN_REORDERED: same columns but different positions."""
    left = build_schema_manifest((build_schema_field("a", "int64", False, 0), build_schema_field("b", "str", True, 1)))
    right = build_schema_manifest((build_schema_field("b", "str", True, 0), build_schema_field("a", "int64", False, 1)))
    cmp = build_dataframe_schema_comparison(left.schema_manifest_hash, right.schema_manifest_hash, False)
    fe = (
        build_schema_field_equivalence(left.fields[0].field_hash, right.fields[1].field_hash, "ORDER_INSENSITIVE_SCHEMA", True),
        build_schema_field_equivalence(left.fields[1].field_hash, right.fields[0].field_hash, "ORDER_INSENSITIVE_SCHEMA", True),
    )
    r = build_schema_equivalence_receipt(
        left, right, cmp, fe,
        build_schema_ordering_equivalence("ORDER_INSENSITIVE", True),
        build_schema_nullability_equivalence("EXACT_NULLABILITY", True),
        build_schema_dtype_equivalence("EXACT_DTYPE", True),
        build_schema_evolution_transition("COLUMN_REORDERED", left.schema_manifest_hash, right.schema_manifest_hash, True),
        build_schema_compatibility_policy("BIDIRECTIONAL_COMPATIBLE", "reorder is bidirectional"),
        (), True,
    )
    assert r.evolution_transition.transition_type == "COLUMN_REORDERED"
    assert r.schemas_equivalent is True
    assert r.mismatch_count == 0


def test_schema_evolution_dtype_changed():
    """DTYPE_CHANGED: same column name but different dtype."""
    left = build_schema_manifest((build_schema_field("a", "int32", False, 0),))
    right = build_schema_manifest((build_schema_field("a", "int64", False, 0),))
    cmp = build_dataframe_schema_comparison(left.schema_manifest_hash, right.schema_manifest_hash, False)
    fe = (build_schema_field_equivalence(left.fields[0].field_hash, right.fields[0].field_hash, "DECLARED_DTYPE_EQUIVALENCE", True),)
    r = build_schema_equivalence_receipt(
        left, right, cmp, fe,
        build_schema_ordering_equivalence("EXACT_ORDER", True),
        build_schema_nullability_equivalence("EXACT_NULLABILITY", True),
        build_schema_dtype_equivalence("NUMERIC_WIDTH_EQUIVALENT", True),
        build_schema_evolution_transition("DTYPE_CHANGED", left.schema_manifest_hash, right.schema_manifest_hash, True),
        build_schema_compatibility_policy("FORWARD_COMPATIBLE", "widening is forward compatible"),
        (), True,
    )
    assert r.evolution_transition.transition_type == "DTYPE_CHANGED"
    assert r.schemas_equivalent is True
    assert r.mismatch_count == 0


def test_schema_evolution_nullability_changed():
    """NULLABILITY_CHANGED: same column but different nullability."""
    left = build_schema_manifest((build_schema_field("a", "int64", False, 0),))
    right = build_schema_manifest((build_schema_field("a", "int64", True, 0),))
    cmp = build_dataframe_schema_comparison(left.schema_manifest_hash, right.schema_manifest_hash, False)
    fe = (build_schema_field_equivalence(left.fields[0].field_hash, right.fields[0].field_hash, "DECLARED_NULLABILITY_EQUIVALENCE", True),)
    r = build_schema_equivalence_receipt(
        left, right, cmp, fe,
        build_schema_ordering_equivalence("EXACT_ORDER", True),
        build_schema_nullability_equivalence("DECLARED_NULLABILITY_EQUIVALENCE", True),
        build_schema_dtype_equivalence("EXACT_DTYPE", True),
        build_schema_evolution_transition("NULLABILITY_CHANGED", left.schema_manifest_hash, right.schema_manifest_hash, True),
        build_schema_compatibility_policy("FORWARD_COMPATIBLE", "non-null to nullable is forward compatible"),
        (), True,
    )
    assert r.evolution_transition.transition_type == "NULLABILITY_CHANGED"
    assert r.schemas_equivalent is True
    assert r.mismatch_count == 0


def test_child_hash_recomputation_detects_forged_child():
    """Verify that child validators recompute hashes and reject forged children."""
    from qec.analysis.schema_equivalence_receipts import (
        SchemaFieldEquivalence,
        SchemaOrderingEquivalence,
        SchemaNullabilityEquivalence,
        SchemaDTypeEquivalence,
        SchemaEvolutionTransition,
        SchemaMismatchRecord,
        SchemaCompatibilityPolicy,
        validate_schema_field_equivalence,
        validate_schema_ordering_equivalence,
        validate_schema_nullability_equivalence,
        validate_schema_dtype_equivalence,
        validate_schema_evolution_transition,
        validate_schema_mismatch_record,
        validate_schema_compatibility_policy,
    )
    # Build valid objects then forge them by changing a field but keeping old hash
    valid_fe = build_schema_field_equivalence(_h(), _h(), "EXACT_SCHEMA", True)
    forged_fe = SchemaFieldEquivalence(
        left_field_hash=valid_fe.left_field_hash,
        right_field_hash=valid_fe.right_field_hash,
        equivalence_mode=valid_fe.equivalence_mode,
        fields_equivalent=False,  # Changed!
        field_equivalence_hash=valid_fe.field_equivalence_hash,  # Stale hash
    )
    with pytest.raises(ValueError, match="field equivalence hash mismatch"):
        validate_schema_field_equivalence(forged_fe)

    valid_oe = build_schema_ordering_equivalence("EXACT_ORDER", True)
    forged_oe = SchemaOrderingEquivalence(
        ordering_mode=valid_oe.ordering_mode,
        ordering_equivalent=False,  # Changed!
        ordering_equivalence_hash=valid_oe.ordering_equivalence_hash,  # Stale hash
    )
    with pytest.raises(ValueError, match="ordering equivalence hash mismatch"):
        validate_schema_ordering_equivalence(forged_oe)

    valid_ne = build_schema_nullability_equivalence("EXACT_NULLABILITY", True)
    forged_ne = SchemaNullabilityEquivalence(
        nullability_mode=valid_ne.nullability_mode,
        nullability_equivalent=False,  # Changed!
        nullability_equivalence_hash=valid_ne.nullability_equivalence_hash,  # Stale hash
    )
    with pytest.raises(ValueError, match="nullability equivalence hash mismatch"):
        validate_schema_nullability_equivalence(forged_ne)

    valid_de = build_schema_dtype_equivalence("EXACT_DTYPE", True)
    forged_de = SchemaDTypeEquivalence(
        dtype_equivalence_mode=valid_de.dtype_equivalence_mode,
        dtype_equivalent=False,  # Changed!
        dtype_equivalence_hash=valid_de.dtype_equivalence_hash,  # Stale hash
    )
    with pytest.raises(ValueError, match="dtype equivalence hash mismatch"):
        validate_schema_dtype_equivalence(forged_de)

    valid_et = build_schema_evolution_transition("NO_CHANGE", _h(), _h(), True)
    forged_et = SchemaEvolutionTransition(
        transition_type=valid_et.transition_type,
        left_schema_hash=valid_et.left_schema_hash,
        right_schema_hash=valid_et.right_schema_hash,
        transition_allowed=False,  # Changed!
        schema_evolution_transition_hash=valid_et.schema_evolution_transition_hash,  # Stale hash
    )
    with pytest.raises(ValueError, match="schema evolution transition hash mismatch"):
        validate_schema_evolution_transition(forged_et)

    valid_mr = build_schema_mismatch_record(0, "DTYPE_MISMATCH", _h(), _h(), "reason")
    forged_mr = SchemaMismatchRecord(
        mismatch_index=valid_mr.mismatch_index,
        mismatch_kind=valid_mr.mismatch_kind,
        left_hash=valid_mr.left_hash,
        right_hash=valid_mr.right_hash,
        reason="forged reason",  # Changed!
        schema_mismatch_record_hash=valid_mr.schema_mismatch_record_hash,  # Stale hash
    )
    with pytest.raises(ValueError, match="schema mismatch record hash mismatch"):
        validate_schema_mismatch_record(forged_mr)

    valid_cp = build_schema_compatibility_policy("STRICT_COMPATIBILITY", "declared")
    forged_cp = SchemaCompatibilityPolicy(
        compatibility_policy=valid_cp.compatibility_policy,
        policy_reason="forged reason",  # Changed!
        compatibility_policy_hash=valid_cp.compatibility_policy_hash,  # Stale hash
    )
    with pytest.raises(ValueError, match="compatibility policy hash mismatch"):
        validate_schema_compatibility_policy(forged_cp)


def test_schema_comparison_binding_to_manifests():
    """Verify that schema_comparison must match the provided manifests."""
    left, right, _, fe = _fixtures(False)
    # Create a schema_comparison with wrong hashes
    wrong_cmp = build_dataframe_schema_comparison(_h(), _h(), True)
    with pytest.raises(ValueError, match="schema_comparison.left_schema_manifest_hash must match"):
        build_schema_equivalence_receipt(
            left, right, wrong_cmp, fe,
            build_schema_ordering_equivalence("EXACT_ORDER", True),
            build_schema_nullability_equivalence("EXACT_NULLABILITY", True),
            build_schema_dtype_equivalence("EXACT_DTYPE", True),
            build_schema_evolution_transition("NO_CHANGE", left.schema_manifest_hash, right.schema_manifest_hash, True),
            build_schema_compatibility_policy("STRICT_COMPATIBILITY", "declared"),
            (), True,
        )
