from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from qec.analysis import polars_pandas_equivalence_receipts as pper


def _digest(name: str, out: str = "a" * 64, schema: str = "b" * 64):
    return pper.build_dataframe_output_digest(name, "0" * 64, out, 10, 2, schema)


def _policy(mode: str = "EXACT_DIGEST", row: str = "DECLARED_SORT_KEYS", keys=("id",), rounding: str = "IEEE754"):
    return pper.build_dataframe_equivalence_policy(mode, "EXACT_DTYPE_MATCH", row, keys, "NULL_EQUALS_NULL", rounding)


def _schema(schema="b" * 64):
    return pper.build_dataframe_schema_comparison(schema, schema, True)


def _receipt(mode="EXACT_DIGEST", left_out="a" * 64, right_out="a" * 64, mismatches=()):
    return pper.build_polars_pandas_equivalence_receipt("POLARS", "PANDAS", _digest("POLARS", left_out), _digest("PANDAS", right_out), _policy(mode), _schema(), mismatches)


def test_hash_stability_and_canonical_json_stability_and_idempotent_rebuild():
    r1 = _receipt()
    r2 = _receipt()
    assert r1.polars_pandas_equivalence_receipt_hash == r2.polars_pandas_equivalence_receipt_hash
    assert pper._canonical_json({"a": 1, "b": 2}) == pper._canonical_json({"b": 2, "a": 1})


def test_exact_digest_and_schema_only_equivalence_pass():
    assert _receipt(mode="EXACT_DIGEST").equivalence_passed is True
    assert _receipt(mode="SCHEMA_ONLY", left_out="1" * 64, right_out="2" * 64).equivalence_passed is True


def test_output_digest_mismatch_and_schema_mismatch_failure_and_mismatch_semantics():
    with pytest.raises(ValueError):
        pper.validate_polars_pandas_equivalence_receipt(_receipt(left_out="1" * 64, right_out="2" * 64))
    # Create digests with different schema hashes to match the bad_schema
    left_digest = _digest("POLARS", schema="b" * 64)
    right_digest = _digest("PANDAS", schema="c" * 64)
    bad_schema = pper.build_dataframe_schema_comparison("b" * 64, "c" * 64, False)
    mismatch = pper.build_dataframe_mismatch_record(0, "SCHEMA_HASH_MISMATCH", "b" * 64, "c" * 64, "schema mismatch")
    r = pper.build_polars_pandas_equivalence_receipt("POLARS", "PANDAS", left_digest, right_digest, _policy(), bad_schema, (mismatch,))
    assert r.equivalence_passed is False


def test_mismatch_count_recompute_duplicate_dense_malformed_hash_rejections():
    mm0 = pper.build_dataframe_mismatch_record(0, "OUTPUT_DIGEST_MISMATCH", "1" * 64, "2" * 64, "digest mismatch")
    mm1 = pper.build_dataframe_mismatch_record(1, "ROW_COUNT_MISMATCH", "3" * 64, "4" * 64, "row mismatch")
    r = _receipt(mismatches=(mm0, mm1))
    tampered = pper.PolarsPandasEquivalenceReceipt(**{**r.__dict__, "mismatch_count": 1})
    with pytest.raises(ValueError):
        pper.validate_polars_pandas_equivalence_receipt(tampered)
    dup = pper.PolarsPandasEquivalenceReceipt(**{**r.__dict__, "mismatches": (mm0, mm0), "mismatch_count": 2})
    with pytest.raises(ValueError):
        pper.validate_polars_pandas_equivalence_receipt(dup)
    sparse = pper.PolarsPandasEquivalenceReceipt(**{**r.__dict__, "mismatches": (mm1,), "mismatch_count": 1, "equivalence_passed": False})
    with pytest.raises(ValueError):
        pper.validate_polars_pandas_equivalence_receipt(sparse)
    bad = pper.DataframeOutputDigest(**{**_digest("POLARS").__dict__, "canonical_output_hash": "abc"})
    with pytest.raises(ValueError):
        pper.validate_dataframe_output_digest(bad)


def test_invalid_policy_values_and_sort_key_rules():
    with pytest.raises(ValueError):
        pper.build_dataframe_equivalence_policy("BAD", "EXACT_DTYPE_MATCH", "DECLARED_SORT_KEYS", ("id",), "NULL_EQUALS_NULL", "IEEE754")
    with pytest.raises(ValueError):
        pper.build_dataframe_equivalence_policy("EXACT_DIGEST", "BAD", "DECLARED_SORT_KEYS", ("id",), "NULL_EQUALS_NULL", "IEEE754")
    with pytest.raises(ValueError):
        pper.build_dataframe_equivalence_policy("EXACT_DIGEST", "EXACT_DTYPE_MATCH", "BAD", ("id",), "NULL_EQUALS_NULL", "IEEE754")
    with pytest.raises(ValueError):
        pper.build_dataframe_equivalence_policy("EXACT_DIGEST", "EXACT_DTYPE_MATCH", "DECLARED_SORT_KEYS", ("id",), "BAD", "IEEE754")
    with pytest.raises(ValueError):
        pper.build_dataframe_equivalence_policy("EXACT_DIGEST", "EXACT_DTYPE_MATCH", "DECLARED_SORT_KEYS", ("id",), "NULL_EQUALS_NULL", "BAD")
    with pytest.raises(ValueError):
        pper.build_dataframe_equivalence_policy("EXACT_DIGEST", "EXACT_DTYPE_MATCH", "DECLARED_SORT_KEYS", (), "NULL_EQUALS_NULL", "IEEE754")
    with pytest.raises(ValueError):
        pper.build_dataframe_equivalence_policy("EXACT_DIGEST", "EXACT_DTYPE_MATCH", "PRESERVE_INPUT_ORDER", ("id",), "NULL_EQUALS_NULL", "IEEE754")


def test_equivalence_passed_recomputed_adapter_only_and_child_validation_schema_validation():
    r = _receipt()
    with pytest.raises(ValueError):
        pper.validate_polars_pandas_equivalence_receipt(pper.PolarsPandasEquivalenceReceipt(**{**r.__dict__, "equivalence_passed": False}))
    with pytest.raises(ValueError):
        pper.validate_polars_pandas_equivalence_receipt(pper.PolarsPandasEquivalenceReceipt(**{**r.__dict__, "adapter_only": False}))
    bad_left = pper.DataframeOutputDigest(**{**r.left_output_digest.__dict__, "output_digest_hash": "0" * 64})
    with pytest.raises(ValueError):
        pper.validate_polars_pandas_equivalence_receipt(pper.PolarsPandasEquivalenceReceipt(**{**r.__dict__, "left_output_digest": bad_left}))
    with pytest.raises(ValueError):
        pper.validate_dataframe_schema_comparison(pper.DataframeSchemaComparison("0" * 64, "1" * 64, True, "0" * 64))


def test_no_forbidden_imports_decoder_boundary_immutable_payload_pythonhashseed_forbidden_claims():
    # AST-based forbidden import check
    source_path = Path("src/qec/analysis/polars_pandas_equivalence_receipts.py")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_modules = {"pandas", "polars", "pyarrow", "requests", "subprocess", "urllib", "os"}
    decoder_modules = {"qec.decoder"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name.split(".")[0]
                assert module_name not in forbidden_modules, f"Forbidden import: {alias.name}"
                assert alias.name not in decoder_modules and not alias.name.startswith("qec.decoder"), f"Forbidden decoder import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_name = node.module.split(".")[0]
                assert module_name not in forbidden_modules, f"Forbidden import from: {node.module}"
                assert node.module not in decoder_modules and not node.module.startswith("qec.decoder"), f"Forbidden decoder import from: {node.module}"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in {"eval", "exec", "__import__"}, f"Forbidden call: {node.func.id}"
    r = _receipt()
    with pytest.raises(TypeError):
        pper.validate_polars_pandas_equivalence_receipt(pper.PolarsPandasEquivalenceReceipt(**{**r.__dict__, "mismatches": []}))
    with pytest.raises(ValueError):
        pper.build_dataframe_mismatch_record(0, "OUTPUT_DIGEST_MISMATCH", "0" * 64, "1" * 64, "speedup proven")
    code = (
        "from qec.analysis.polars_pandas_equivalence_receipts import *;"
        "l=build_dataframe_output_digest('POLARS','0'*64,'a'*64,1,1,'b'*64);"
        "r=build_dataframe_output_digest('PANDAS','0'*64,'a'*64,1,1,'b'*64);"
        "p=build_dataframe_equivalence_policy('EXACT_DIGEST','EXACT_DTYPE_MATCH','DECLARED_SORT_KEYS',('id',),'NULL_EQUALS_NULL','IEEE754');"
        "s=build_dataframe_schema_comparison('b'*64,'b'*64,True);"
        "x=build_polars_pandas_equivalence_receipt('POLARS','PANDAS',l,r,p,s);"
        "print(x.polars_pandas_equivalence_receipt_hash)"
    )
    out1 = subprocess.check_output([sys.executable, "-c", code], env={"PYTHONPATH": "src", "PYTHONHASHSEED": "0"}, text=True).strip()
    out2 = subprocess.check_output([sys.executable, "-c", code], env={"PYTHONPATH": "src", "PYTHONHASHSEED": "1"}, text=True).strip()
    assert out1 == out2


def test_strict_string_type_validation():
    """Test that backend_name and other string fields must be actual strings."""
    # backend_name must be a string, not a list
    with pytest.raises(TypeError, match="backend_name must be a string"):
        pper.build_dataframe_output_digest(["POLARS"], "0" * 64, "a" * 64, 10, 2, "b" * 64)  # type: ignore[arg-type]
    # equivalence_mode must be a string
    with pytest.raises(TypeError, match="equivalence_mode must be a string"):
        pper.build_dataframe_equivalence_policy(123, "EXACT_DTYPE_MATCH", "DECLARED_SORT_KEYS", ("id",), "NULL_EQUALS_NULL", "IEEE754")  # type: ignore[arg-type]


def test_declared_sort_keys_rejects_plain_string():
    """Test that declared_sort_keys rejects a plain string."""
    with pytest.raises(TypeError, match="declared_sort_keys must be a sequence"):
        pper.build_dataframe_equivalence_policy("EXACT_DIGEST", "EXACT_DTYPE_MATCH", "DECLARED_SORT_KEYS", "column_name", "NULL_EQUALS_NULL", "IEEE754")  # type: ignore[arg-type]


def test_declared_sort_keys_must_be_tuple_in_validator():
    """Test that declared_sort_keys must be a tuple in the validator."""
    policy = _policy()
    bad_policy = pper.DataframeEquivalencePolicy(
        equivalence_mode=policy.equivalence_mode,
        dtype_policy=policy.dtype_policy,
        row_order_policy=policy.row_order_policy,
        declared_sort_keys=["id"],  # type: ignore[arg-type]
        null_policy=policy.null_policy,
        rounding_policy=policy.rounding_policy,
        equivalence_policy_hash=policy.equivalence_policy_hash,
    )
    with pytest.raises(TypeError, match="declared_sort_keys must be a tuple"):
        pper.validate_dataframe_equivalence_policy(bad_policy)


def test_declared_sort_keys_must_be_unique():
    """Test that declared_sort_keys must have unique elements."""
    with pytest.raises(ValueError, match="declared_sort_keys must be unique"):
        pper.build_dataframe_equivalence_policy("EXACT_DIGEST", "EXACT_DTYPE_MATCH", "DECLARED_SORT_KEYS", ("id", "id"), "NULL_EQUALS_NULL", "IEEE754")


def test_declared_rounding_bound_mode():
    """Test that DECLARED_ROUNDING_BOUND mode allows different digests."""
    # DECLARED_ROUNDING_BOUND requires non-EXACT rounding policy
    with pytest.raises(ValueError, match="DECLARED_ROUNDING_BOUND requires non-EXACT rounding policy"):
        pper.build_dataframe_equivalence_policy("DECLARED_ROUNDING_BOUND", "EXACT_DTYPE_MATCH", "DECLARED_SORT_KEYS", ("id",), "NULL_EQUALS_NULL", "EXACT")
    # DECLARED_ROUNDING_BOUND with IEEE754 rounding should allow different digests
    policy = pper.build_dataframe_equivalence_policy("DECLARED_ROUNDING_BOUND", "EXACT_DTYPE_MATCH", "DECLARED_SORT_KEYS", ("id",), "NULL_EQUALS_NULL", "IEEE754")
    left = _digest("POLARS", "1" * 64)
    right = _digest("PANDAS", "2" * 64)
    schema = _schema()
    receipt = pper.build_polars_pandas_equivalence_receipt("POLARS", "PANDAS", left, right, policy, schema)
    assert receipt.equivalence_passed is True


def test_schema_comparison_hash_binding_to_output_digests():
    """Test P1: schema_comparison hashes must match output_digest schema_manifest_hash."""
    left = _digest("POLARS", schema="a" * 64)
    right = _digest("PANDAS", schema="a" * 64)
    # Create schema comparison with different hashes
    bad_schema = pper.build_dataframe_schema_comparison("b" * 64, "b" * 64, True)
    policy = _policy()
    # Build receipt manually to bypass builder validation
    receipt = pper.PolarsPandasEquivalenceReceipt(
        schema_version="POLARS_PANDAS_EQUIVALENCE_RECEIPT_V1",
        left_backend_name="POLARS",
        right_backend_name="PANDAS",
        left_output_digest=left,
        right_output_digest=right,
        equivalence_policy=policy,
        schema_comparison=bad_schema,
        mismatches=(),
        mismatch_count=0,
        lazy_plan_canonical_receipt_hash=None,
        equivalence_passed=True,
        adapter_only=True,
        polars_pandas_equivalence_receipt_hash="0" * 64,
    )
    with pytest.raises(ValueError, match="schema_comparison.left_schema_manifest_hash must match left_output_digest.schema_manifest_hash"):
        pper.validate_polars_pandas_equivalence_receipt(receipt)


def test_manifest_hash_binding_to_output_digests():
    """Test P2: backend_manifest hash must match output_digest.backend_manifest_hash."""
    from qec.analysis.dataframe_backend_manifest import (
        build_dataframe_backend_manifest,
        build_execution_policy,
        build_null_policy,
        build_ordering_policy,
        build_precision_policy,
        build_schema_field,
        build_schema_manifest,
    )
    # Create a valid manifest
    fields = [build_schema_field("a", "int64", False, 0)]
    sm = build_schema_manifest(fields)
    ep = build_execution_policy("HYBRID", True, True)
    pp = build_precision_policy("IEEE754", 64)
    op = build_ordering_policy("PRESERVE_INPUT_ORDER", [])
    np = build_null_policy("DECLARED_NULLABLE", True)
    manifest = build_dataframe_backend_manifest("POLARS", "1.0.0", True, sm, ep, pp, op, np)
    # Create a receipt with a different backend_manifest_hash
    left = _digest("POLARS")  # Uses "0" * 64 as backend_manifest_hash
    right = _digest("PANDAS")
    receipt = _receipt()
    # Validate with mismatched manifest
    with pytest.raises(ValueError, match="left_backend_manifest hash must match left_output_digest.backend_manifest_hash"):
        pper.validate_polars_pandas_equivalence_receipt(receipt, left_backend_manifest=manifest)

