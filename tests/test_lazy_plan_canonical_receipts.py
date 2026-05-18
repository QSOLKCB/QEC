from __future__ import annotations

import subprocess
import sys

import pytest

from qec.analysis import lazy_plan_canonical_receipts as lpr


def _op(i: int = 0, payload: dict[str, object] | None = None):
    return lpr.build_lazy_plan_operation(i, "SELECT", ("a",), ("b",), payload or {"expr": "a"})


def _receipt():
    projection = lpr.build_lazy_plan_projection(("b",))
    filt = lpr.build_lazy_plan_filter("b > 0", ("b",))
    agg = lpr.build_lazy_plan_aggregation(("b",), ("sum",))
    join = lpr.build_lazy_plan_join("INNER", ("id",), ("id",))
    sort = lpr.build_lazy_plan_sort(("b",), ("ASCENDING",))
    st = lpr.build_lazy_plan_schema_transition("NO_SCHEMA_CHANGE", "0" * 64, "1" * 64)
    boundary = lpr.build_lazy_execution_boundary("FULLY_LAZY", False, True)
    return lpr.build_lazy_plan_canonical_receipt("POLARS", "1.0", True, (_op(0),), projection, (filt,), (agg,), (join,), (sort,), (st,), boundary)


def test_hash_and_canonical_json_stability_and_idempotent_rebuild():
    r1 = _receipt()
    r2 = _receipt()
    assert r1.lazy_plan_canonical_receipt_hash == r2.lazy_plan_canonical_receipt_hash
    assert lpr._canonical_json({"a": 1, "b": 2}) == lpr._canonical_json({"b": 2, "a": 1})


def test_deterministic_operation_ordering_and_count_recompute():
    p = lpr.build_lazy_plan_projection(("x",))
    r = lpr.build_lazy_plan_canonical_receipt("POLARS", "1", True, (_op(1), _op(0)), p)
    assert [x.operation_index for x in r.operations] == [0, 1]
    assert r.operation_count == 2


def test_dense_and_duplicate_operation_rejection():
    p = lpr.build_lazy_plan_projection(("x",))
    with pytest.raises(ValueError):
        lpr.build_lazy_plan_canonical_receipt("POLARS", "1", True, (_op(0), _op(2)), p)
    with pytest.raises(ValueError):
        lpr.build_lazy_plan_canonical_receipt("POLARS", "1", True, (_op(0), _op(0)), p)


def test_malformed_hash_rejection():
    op = _op(0)
    bad = lpr.LazyPlanOperation(**{**op.__dict__, "operation_hash": "abc"})
    with pytest.raises(ValueError):
        lpr.validate_lazy_plan_operation(bad)


def test_invalid_types_and_semantics_rejections():
    with pytest.raises(ValueError):
        lpr.build_lazy_plan_join("BAD", ("a",), ("a",))
    with pytest.raises(ValueError):
        lpr.build_lazy_plan_operation(0, "BAD", (), (), {})
    with pytest.raises(ValueError):
        lpr.build_lazy_execution_boundary("BAD", False, False)
    with pytest.raises(ValueError):
        lpr.build_lazy_plan_schema_transition("BAD", "0" * 64, "1" * 64)
    with pytest.raises(ValueError):
        lpr.build_lazy_plan_sort(("a", "a"), ("ASCENDING", "DESCENDING"))
    with pytest.raises(ValueError):
        lpr.build_lazy_plan_sort(("a",), ("ASCENDING", "DESCENDING"))


def test_adapter_only_strict_bool_enforcement():
    r = _receipt()
    bad = lpr.LazyPlanCanonicalReceipt(**{**r.__dict__, "adapter_only": 1})
    with pytest.raises(TypeError):
        lpr.validate_lazy_plan_canonical_receipt(bad)


def test_no_forbidden_imports_and_decoder_untouched():
    src = open("src/qec/analysis/lazy_plan_canonical_receipts.py", encoding="utf-8").read()
    forbidden = ["import pandas", "import polars", "import pyarrow", "import requests", "import subprocess", "urllib"]
    assert not any(x in src for x in forbidden)
    assert "src/qec/decoder/" not in "src/qec/analysis/lazy_plan_canonical_receipts.py"


def test_immutable_payload_validation_and_runtime_semantics_rejection():
    op = _op(0)
    with pytest.raises(TypeError):
        lpr.validate_lazy_plan_operation(lpr.LazyPlanOperation(0, "SELECT", (), (), {"a": 1}, op.operation_hash))
    with pytest.raises(ValueError):
        lpr.build_lazy_plan_filter("query executed", ("a",))


def test_pythonhashseed_replay_stability():
    code = (
        "from qec.analysis.lazy_plan_canonical_receipts import *;"
        "op=build_lazy_plan_operation(0,'SELECT',('a',),('b',),{'expr':'a'});"
        "p=build_lazy_plan_projection(('b',));"
        "r=build_lazy_plan_canonical_receipt('POLARS','1',True,(op,),p);"
        "print(r.lazy_plan_canonical_receipt_hash)"
    )
    out1 = subprocess.check_output([sys.executable, "-c", code], env={"PYTHONPATH": "src", "PYTHONHASHSEED": "0"}, text=True).strip()
    out2 = subprocess.check_output([sys.executable, "-c", code], env={"PYTHONPATH": "src", "PYTHONHASHSEED": "1"}, text=True).strip()
    assert out1 == out2


def test_schema_transition_stability():
    a = lpr.build_lazy_plan_schema_transition("DTYPE_CHANGED", "0" * 64, "1" * 64)
    b = lpr.build_lazy_plan_schema_transition("DTYPE_CHANGED", "0" * 64, "1" * 64)
    assert a.schema_transition_hash == b.schema_transition_hash
