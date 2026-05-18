from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest

from qec.analysis.dataframe_backend_manifest import (
    _canonical_json,
    build_dataframe_backend_manifest,
    build_execution_policy,
    build_null_policy,
    build_ordering_policy,
    build_precision_policy,
    build_schema_field,
    build_schema_manifest,
    validate_dataframe_backend_manifest,
    validate_schema_manifest,
)


def _manifest():
    fields = [
        build_schema_field("b", "int64", False, 1),
        build_schema_field("a", "float64", True, 0),
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
f=[build_schema_field('a','float64',True,0),build_schema_field('b','int64',False,1)]
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
