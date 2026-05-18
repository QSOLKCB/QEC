from __future__ import annotations

import subprocess
import sys

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
    bad_schema = pper.build_dataframe_schema_comparison("b" * 64, "c" * 64, False)
    mismatch = pper.build_dataframe_mismatch_record(0, "SCHEMA_HASH_MISMATCH", "b" * 64, "c" * 64, "schema mismatch")
    r = pper.build_polars_pandas_equivalence_receipt("POLARS", "PANDAS", _digest("POLARS"), _digest("PANDAS"), _policy(), bad_schema, (mismatch,))
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
    src = open("src/qec/analysis/polars_pandas_equivalence_receipts.py", encoding="utf-8").read()
    forbidden = ["import pandas", "import polars", "import pyarrow", "import requests", "import subprocess", "urllib", "from qec.decoder", "import qec.decoder"]
    assert not any(x in src for x in forbidden)
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
