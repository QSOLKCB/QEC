from __future__ import annotations

import ast
import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest

from qec.analysis.dataframe_backend_manifest import (
    DataframeExecutionPolicy,
    DataframeNullPolicy,
    DataframeOrderingPolicy,
    DataframePrecisionPolicy,
    DataframeSchemaField,
    DataframeSchemaManifest,
    _canonical_json,
    build_dataframe_backend_manifest,
    build_execution_policy,
    build_null_policy,
    build_ordering_policy,
    build_precision_policy,
    build_schema_field,
    build_schema_manifest,
    validate_dataframe_backend_manifest,
    validate_execution_policy,
    validate_null_policy,
    validate_ordering_policy,
    validate_precision_policy,
    validate_schema_field,
    validate_schema_manifest,
)


def _manifest():
    fields = [
        build_schema_field("b", "int64", False, 1),
        build_schema_field("a", "float64", False, 0),  # Changed to nullable=False
    ]
    sm = build_schema_manifest(fields)
    ep = build_execution_policy("HYBRID", True, True)
    pp = build_precision_policy("IEEE754", 64)
    op = build_ordering_policy("DECLARED_SORT_KEYS", ["a", "b"])
    np = build_null_policy("DECLARED_NULLABLE", True)
    return build_dataframe_backend_manifest("PANDAS", "2.0.0", True, sm, ep, pp, op, np)


def test_hash_stability_and_manifest_idempotence():
    m1 = _manifest()
    m2 = _manifest()
    assert m1.dataframe_backend_manifest_hash == m2.dataframe_backend_manifest_hash


def test_canonical_json_stability():
    p1 = {"b": 1, "a": [2, 1]}
    p2 = {"a": [2, 1], "b": 1}
    assert _canonical_json(p1) == _canonical_json(p2)


def test_deterministic_ordering_fields_sorted_by_position():
    sm = _manifest().schema_manifest
    assert [f.field_name for f in sm.fields] == ["a", "b"]


def test_schema_field_uniqueness_rejection_and_duplicate_field_rejection():
    f0 = build_schema_field("a", "int64", True, 0)
    f1 = build_schema_field("a", "float64", False, 1)
    with pytest.raises(ValueError, match="duplicate field names"):
        build_schema_manifest([f0, f1])


def test_dense_field_positions_rejection():
    f0 = build_schema_field("a", "int64", True, 0)
    f2 = build_schema_field("b", "float64", False, 2)
    with pytest.raises(ValueError, match="dense"):
        build_schema_manifest([f0, f2])


def test_malformed_hash_rejection():
    m = _manifest()
    bad = dataclasses.replace(m, dataframe_backend_manifest_hash="xyz")
    with pytest.raises(ValueError, match="64-character"):
        validate_dataframe_backend_manifest(bad)


def test_invalid_backend_rejection():
    m = _manifest()
    bad = dataclasses.replace(m, backend_name="DUCKDB")
    with pytest.raises(ValueError, match="invalid backend"):
        validate_dataframe_backend_manifest(bad)


def test_invalid_execution_mode_rejection():
    m = _manifest()
    bad_ep = dataclasses.replace(m.execution_policy, execution_mode="STREAMING")
    bad = dataclasses.replace(m, execution_policy=bad_ep)
    with pytest.raises(ValueError, match="invalid execution mode"):
        validate_dataframe_backend_manifest(bad)


def test_invalid_rounding_policy_rejection():
    m = _manifest()
    bad_pp = dataclasses.replace(m.precision_policy, rounding_policy="BANKERS")
    bad = dataclasses.replace(m, precision_policy=bad_pp)
    with pytest.raises(ValueError, match="invalid rounding policy"):
        validate_dataframe_backend_manifest(bad)


def test_invalid_ordering_policy_rejection():
    m = _manifest()
    bad_op = dataclasses.replace(m.ordering_policy, ordering_policy="AUTO")
    bad = dataclasses.replace(m, ordering_policy=bad_op)
    with pytest.raises(ValueError, match="invalid ordering policy"):
        validate_dataframe_backend_manifest(bad)


def test_invalid_null_policy_rejection():
    m = _manifest()
    bad_np = dataclasses.replace(m.null_policy, null_policy="NA_POLICY")
    bad = dataclasses.replace(m, null_policy=bad_np)
    with pytest.raises(ValueError, match="invalid null policy"):
        validate_dataframe_backend_manifest(bad)


def test_adapter_only_enforcement():
    m = _manifest()
    bad = dataclasses.replace(m, adapter_only=False)
    with pytest.raises(ValueError, match="adapter_only"):
        validate_dataframe_backend_manifest(bad)


def test_no_forbidden_imports_and_no_runtime_execution_semantics():
    text = Path("src/qec/analysis/dataframe_backend_manifest.py").read_text(encoding="utf-8")
    forbidden = ["import pandas", "import polars", "import pyarrow", "import requests", "subprocess", "eval(", "exec(", "os.system"]
    for token in forbidden:
        assert token not in text


def test_decoder_boundary_enforcement():
    assert Path("src/qec/decoder").exists()


def test_immutable_payload_validation():
    m = _manifest()
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.backend_version = "x"  # type: ignore[misc]


def test_schema_count_recomputation_rejection():
    sm = _manifest().schema_manifest
    bad = dataclasses.replace(sm, schema_field_count=999)
    with pytest.raises(ValueError, match="count mismatch"):
        validate_schema_manifest(bad)


def test_stable_hashing_across_pythonhashseed():
    code = """
from qec.analysis.dataframe_backend_manifest import build_schema_field, build_schema_manifest, build_execution_policy, build_precision_policy, build_ordering_policy, build_null_policy, build_dataframe_backend_manifest
f=[build_schema_field('a','float64',False,0),build_schema_field('b','int64',False,1)]
sm=build_schema_manifest(f)
e=build_execution_policy('HYBRID',True,True)
p=build_precision_policy('IEEE754',64)
o=build_ordering_policy('DECLARED_SORT_KEYS',['a','b'])
n=build_null_policy('DECLARED_NULLABLE',True)
m=build_dataframe_backend_manifest('PANDAS','2.0.0',True,sm,e,p,o,n)
print(m.dataframe_backend_manifest_hash)
"""
    env0 = {"PYTHONPATH": "src", "PYTHONHASHSEED": "0"}
    env1 = {"PYTHONPATH": "src", "PYTHONHASHSEED": "1"}
    h0 = subprocess.check_output([sys.executable, "-c", code], env=env0, text=True).strip()
    h1 = subprocess.check_output([sys.executable, "-c", code], env=env1, text=True).strip()
    assert h0 == h1


# ============================================================================
# New tests for strict type validation
# ============================================================================


def test_strict_bool_validation_nullable():
    """Test that nullable must be a boolean, not a truthy value."""
    with pytest.raises(TypeError, match="nullable must be a boolean"):
        build_schema_field("a", "int64", "False", 0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="nullable must be a boolean"):
        build_schema_field("a", "int64", 1, 0)  # type: ignore[arg-type]


def test_strict_int_validation_field_position():
    """Test that field_position must be an integer, not a float."""
    with pytest.raises(TypeError, match="field_position must be an integer"):
        build_schema_field("a", "int64", True, 1.9)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="field_position must be an integer"):
        build_schema_field("a", "int64", True, True)  # type: ignore[arg-type]


def test_strict_str_validation_field_name():
    """Test that field_name must be a string."""
    with pytest.raises(TypeError, match="field_name must be a string"):
        build_schema_field(["a"], "int64", True, 0)  # type: ignore[arg-type]


def test_strict_bool_validation_execution_policy():
    """Test that execution policy booleans must be actual booleans."""
    with pytest.raises(TypeError, match="lazy_execution_allowed must be a boolean"):
        build_execution_policy("HYBRID", "True", True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="eager_execution_allowed must be a boolean"):
        build_execution_policy("HYBRID", True, 1)  # type: ignore[arg-type]


def test_strict_int_validation_precision_bits():
    """Test that float_precision_bits must be an integer."""
    with pytest.raises(TypeError, match="float_precision_bits must be an integer"):
        build_precision_policy("IEEE754", 63.9)  # type: ignore[arg-type]


def test_strict_bool_validation_null_policy():
    """Test that allow_null_values must be a boolean."""
    with pytest.raises(TypeError, match="allow_null_values must be a boolean"):
        build_null_policy("DECLARED_NULLABLE", "True")  # type: ignore[arg-type]


def test_declared_sort_keys_rejects_plain_string():
    """Test that declared_sort_keys rejects a plain string."""
    with pytest.raises(TypeError, match="declared_sort_keys must be a sequence of strings"):
        build_ordering_policy("DECLARED_SORT_KEYS", "column_name")  # type: ignore[arg-type]


def test_contradictory_null_policy_rejection():
    """Test that NULLS_FORBIDDEN with allow_null_values=True is rejected."""
    with pytest.raises(ValueError, match="contradictory null policy"):
        build_null_policy("NULLS_FORBIDDEN", True)


def test_declared_sort_keys_must_exist_in_schema():
    """Test that declared_sort_keys must reference existing schema fields."""
    fields = [
        build_schema_field("a", "int64", False, 0),
        build_schema_field("b", "float64", False, 1),
    ]
    sm = build_schema_manifest(fields)
    ep = build_execution_policy("HYBRID", True, True)
    pp = build_precision_policy("IEEE754", 64)
    op = build_ordering_policy("DECLARED_SORT_KEYS", ["a", "c"])  # "c" doesn't exist
    np = build_null_policy("DECLARED_NULLABLE", True)
    with pytest.raises(ValueError, match="declared_sort_key 'c' not found in schema fields"):
        build_dataframe_backend_manifest("PANDAS", "2.0.0", True, sm, ep, pp, op, np)


def test_null_policy_reconciliation_with_field_nullability():
    """Test that nullable fields are rejected when null policy forbids nulls."""
    fields = [
        build_schema_field("a", "int64", True, 0),  # nullable=True
    ]
    sm = build_schema_manifest(fields)
    ep = build_execution_policy("HYBRID", True, True)
    pp = build_precision_policy("IEEE754", 64)
    op = build_ordering_policy("PRESERVE_INPUT_ORDER", [])
    np = build_null_policy("NULLS_FORBIDDEN", False)
    with pytest.raises(ValueError, match="field 'a' is nullable but null policy forbids nulls"):
        build_dataframe_backend_manifest("PANDAS", "2.0.0", True, sm, ep, pp, op, np)


def test_null_policy_reconciliation_allow_null_values_false():
    """Test that nullable fields are rejected when allow_null_values=False."""
    fields = [
        build_schema_field("a", "int64", True, 0),  # nullable=True
    ]
    sm = build_schema_manifest(fields)
    ep = build_execution_policy("HYBRID", True, True)
    pp = build_precision_policy("IEEE754", 64)
    op = build_ordering_policy("PRESERVE_INPUT_ORDER", [])
    np = build_null_policy("DECLARED_NULLABLE", False)  # allow_null_values=False
    with pytest.raises(ValueError, match="field 'a' is nullable but null policy forbids nulls"):
        build_dataframe_backend_manifest("PANDAS", "2.0.0", True, sm, ep, pp, op, np)


def test_backend_version_must_be_non_empty():
    """Test that backend_version must be non-empty."""
    fields = [build_schema_field("a", "int64", False, 0)]
    sm = build_schema_manifest(fields)
    ep = build_execution_policy("HYBRID", True, True)
    pp = build_precision_policy("IEEE754", 64)
    op = build_ordering_policy("PRESERVE_INPUT_ORDER", [])
    np = build_null_policy("DECLARED_NULLABLE", True)
    with pytest.raises(ValueError, match="backend_version must be non-empty"):
        build_dataframe_backend_manifest("PANDAS", "", True, sm, ep, pp, op, np)


def test_canonical_field_ordering_validation():
    """Test that validate_schema_manifest enforces canonical field ordering."""
    f0 = build_schema_field("a", "int64", False, 0)
    f1 = build_schema_field("b", "float64", False, 1)
    # Build a manifest with correct ordering
    sm = build_schema_manifest([f0, f1])
    # Manually create a manifest with wrong ordering (b before a)
    bad_sm = DataframeSchemaManifest(
        fields=(f1, f0),  # Wrong order
        schema_field_count=2,
        schema_manifest_hash=sm.schema_manifest_hash,  # Use same hash
    )
    with pytest.raises(ValueError, match="fields must be in canonical order"):
        validate_schema_manifest(bad_sm)


def test_fields_must_be_tuple():
    """Test that fields must be a tuple, not a list."""
    f0 = build_schema_field("a", "int64", False, 0)
    sm = build_schema_manifest([f0])
    # Manually create a manifest with a list instead of tuple
    bad_sm = DataframeSchemaManifest(
        fields=[f0],  # type: ignore[arg-type]
        schema_field_count=1,
        schema_manifest_hash=sm.schema_manifest_hash,
    )
    with pytest.raises(TypeError, match="fields must be a tuple"):
        validate_schema_manifest(bad_sm)


def test_declared_sort_keys_must_be_tuple():
    """Test that declared_sort_keys must be a tuple, not a list."""
    op = build_ordering_policy("DECLARED_SORT_KEYS", ["a", "b"])
    # Manually create a policy with a list instead of tuple
    bad_op = DataframeOrderingPolicy(
        ordering_policy="DECLARED_SORT_KEYS",
        declared_sort_keys=["a", "b"],  # type: ignore[arg-type]
        ordering_policy_hash=op.ordering_policy_hash,
    )
    with pytest.raises(TypeError, match="declared_sort_keys must be a tuple"):
        validate_ordering_policy(bad_op)


def test_schema_field_count_must_be_integer():
    """Test that schema_field_count must be an integer."""
    f0 = build_schema_field("a", "int64", False, 0)
    sm = build_schema_manifest([f0])
    # Manually create a manifest with True instead of 1 (True == 1 but type is bool)
    bad_sm = DataframeSchemaManifest(
        fields=(f0,),
        schema_field_count=True,  # type: ignore[arg-type]
        schema_manifest_hash=sm.schema_manifest_hash,
    )
    with pytest.raises(TypeError, match="schema_field_count must be an integer"):
        validate_schema_manifest(bad_sm)


def test_policy_type_validation():
    """Test that policy objects must be correct dataclass types."""
    m = _manifest()
    # Replace execution_policy with a dict
    bad = dataclasses.replace(m, execution_policy={"execution_mode": "HYBRID"})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="execution_policy must be DataframeExecutionPolicy"):
        validate_dataframe_backend_manifest(bad)


def test_allow_nan_rejection():
    """Test that NaN/Infinity values are rejected in canonical JSON."""
    import math
    with pytest.raises(ValueError):
        _canonical_json({"value": math.nan})
    with pytest.raises(ValueError):
        _canonical_json({"value": math.inf})


def test_no_forbidden_imports_ast_based():
    """Test that no forbidden imports exist using AST analysis."""
    source_path = Path("src/qec/analysis/dataframe_backend_manifest.py")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    
    forbidden_modules = {"pandas", "polars", "pyarrow", "requests", "subprocess", "os"}
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name.split(".")[0]
                assert module_name not in forbidden_modules, f"Forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_name = node.module.split(".")[0]
                assert module_name not in forbidden_modules, f"Forbidden import from: {node.module}"
        elif isinstance(node, ast.Call):
            # Check for __import__, importlib.import_module, eval, exec
            if isinstance(node.func, ast.Name):
                assert node.func.id not in {"eval", "exec", "__import__"}, f"Forbidden call: {node.func.id}"
            elif isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    if node.func.value.id == "importlib" and node.func.attr == "import_module":
                        # Check if importing forbidden module
                        if node.args and isinstance(node.args[0], ast.Constant):
                            module_name = str(node.args[0].value).split(".")[0]
                            assert module_name not in forbidden_modules, f"Forbidden importlib.import_module: {node.args[0].value}"


def test_nullable_fields_allowed_when_null_policy_permits():
    """Test that nullable fields are allowed when null policy permits nulls."""
    fields = [
        build_schema_field("a", "int64", True, 0),  # nullable=True
        build_schema_field("b", "float64", False, 1),
    ]
    sm = build_schema_manifest(fields)
    ep = build_execution_policy("HYBRID", True, True)
    pp = build_precision_policy("IEEE754", 64)
    op = build_ordering_policy("PRESERVE_INPUT_ORDER", [])
    np = build_null_policy("DECLARED_NULLABLE", True)  # allow_null_values=True
    # This should succeed
    m = build_dataframe_backend_manifest("PANDAS", "2.0.0", True, sm, ep, pp, op, np)
    assert m.schema_manifest.fields[0].nullable is True
