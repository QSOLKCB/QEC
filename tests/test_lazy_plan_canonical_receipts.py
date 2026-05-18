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
    # Verify no imports from qec.decoder in the source
    assert "from qec.decoder" not in src
    assert "import qec.decoder" not in src


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


def test_filter_hash_recomputation_rejects_tampered_filter():
    """Verify that validate_lazy_plan_filter recomputes and verifies the hash."""
    valid_filter = lpr.build_lazy_plan_filter("x > 0", ("x",))
    # Create a tampered filter with mismatched expression but valid hash format
    tampered = lpr.LazyPlanFilter(
        filter_expression="y < 10",  # Different expression
        referenced_columns=("y",),   # Different columns
        filter_hash=valid_filter.filter_hash,  # Reuse old hash
    )
    with pytest.raises(ValueError, match="filter hash mismatch"):
        lpr.validate_lazy_plan_filter(tampered)


def test_execution_boundary_strict_bool_enforcement():
    """Verify that validate_lazy_execution_boundary enforces strict bool types."""
    valid_boundary = lpr.build_lazy_execution_boundary("FULLY_LAZY", False, True)
    # Create boundary with integer instead of bool for eager_materialization_allowed
    tampered_eager = lpr.LazyExecutionBoundary(
        execution_boundary="FULLY_LAZY",
        eager_materialization_allowed=0,  # int instead of bool
        lazy_reordering_allowed=True,
        execution_boundary_hash=valid_boundary.execution_boundary_hash,
    )
    with pytest.raises(TypeError, match="eager_materialization_allowed must be strict bool"):
        lpr.validate_lazy_execution_boundary(tampered_eager)
    # Create boundary with integer instead of bool for lazy_reordering_allowed
    tampered_lazy = lpr.LazyExecutionBoundary(
        execution_boundary="FULLY_LAZY",
        eager_materialization_allowed=False,
        lazy_reordering_allowed=1,  # int instead of bool
        execution_boundary_hash=valid_boundary.execution_boundary_hash,
    )
    with pytest.raises(TypeError, match="lazy_reordering_allowed must be strict bool"):
        lpr.validate_lazy_execution_boundary(tampered_lazy)


def test_receipt_validates_nested_components():
    """Verify that validate_lazy_plan_canonical_receipt validates nested components."""
    # Build a valid receipt first
    valid_receipt = _receipt()
    # Create a receipt with an invalid nested join type
    invalid_join = lpr.LazyPlanJoin(
        join_type="INVALID_JOIN",  # Invalid join type
        left_keys=("id",),
        right_keys=("id",),
        join_hash="0" * 64,  # Fake hash
    )
    # Manually construct a receipt with the invalid join
    tampered_receipt = lpr.LazyPlanCanonicalReceipt(
        schema_version=valid_receipt.schema_version,
        backend_name=valid_receipt.backend_name,
        backend_version=valid_receipt.backend_version,
        adapter_only=valid_receipt.adapter_only,
        operations=valid_receipt.operations,
        projection=valid_receipt.projection,
        filters=valid_receipt.filters,
        aggregations=valid_receipt.aggregations,
        joins=(invalid_join,),  # Invalid join
        sorts=valid_receipt.sorts,
        schema_transitions=valid_receipt.schema_transitions,
        execution_boundary=valid_receipt.execution_boundary,
        operation_count=valid_receipt.operation_count,
        lazy_plan_canonical_receipt_hash=valid_receipt.lazy_plan_canonical_receipt_hash,
    )
    with pytest.raises(ValueError, match="invalid join type"):
        lpr.validate_lazy_plan_canonical_receipt(tampered_receipt)


def test_non_string_keys_rejected():
    """Verify that non-string mapping keys are rejected."""
    from qec.analysis.canonical_hashing import CanonicalHashingError
    with pytest.raises(CanonicalHashingError, match="payload keys must be strings"):
        lpr._canonical_json({1: "value"})  # Integer key should be rejected
