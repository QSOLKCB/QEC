from __future__ import annotations

import os
import subprocess
import sys

import pytest

from qec.analysis import columnar_optimization_opportunity_receipts as coor
from qec.analysis.lazy_plan_canonical_receipts import build_lazy_plan_canonical_receipt, build_lazy_plan_operation, build_lazy_plan_projection
from qec.analysis.schema_equivalence_receipts import (
    build_schema_compatibility_policy,
    build_schema_dtype_equivalence,
    build_schema_equivalence_receipt,
    build_schema_evolution_transition,
    build_schema_field_equivalence,
    build_schema_nullability_equivalence,
    build_schema_ordering_equivalence,
)
from qec.analysis.dataframe_backend_manifest import build_schema_field, build_schema_manifest
from qec.analysis.polars_pandas_equivalence_receipts import build_dataframe_schema_comparison


def _lazy_receipt():
    op = build_lazy_plan_operation(0, "SELECT", ("a",), ("a",), {"expr": "a"})
    return build_lazy_plan_canonical_receipt("POLARS", "1", True, (op,), build_lazy_plan_projection(("a",)))


def _schema_receipt():
    s = build_schema_manifest((build_schema_field("a", "int64", False, 0),))
    c = build_dataframe_schema_comparison(s.schema_manifest_hash, s.schema_manifest_hash, True)
    return build_schema_equivalence_receipt(
        s, s, c,
        (build_schema_field_equivalence(s.fields[0].field_hash, s.fields[0].field_hash, "EXACT_SCHEMA", True),),
        build_schema_ordering_equivalence("EXACT_ORDER", True),
        build_schema_nullability_equivalence("EXACT_NULLABILITY", True),
        build_schema_dtype_equivalence("EXACT_DTYPE", True),
        build_schema_evolution_transition("NO_CHANGE", s.schema_manifest_hash, s.schema_manifest_hash, True),
        build_schema_compatibility_policy("STRICT_COMPATIBILITY", "declared"),
        (),
        True,
    )


def _build(status="ELIGIBLE", pre_ok=True, risk="LOW"):
    lr, sr = _lazy_receipt(), _schema_receipt()
    return coor.build_columnar_optimization_opportunity_receipt(
        lr.lazy_plan_canonical_receipt_hash,
        sr.schema_equivalence_receipt_hash,
        coor.build_columnar_optimization_candidate("COLUMN_PRUNING", status),
        coor.build_columnar_optimization_scope("LOCAL_OPERATION", (0,)),
        (coor.build_optimization_precondition(0, "safe", pre_ok),),
        (coor.build_optimization_constraint(0, "ROW_ORDER_MUST_PRESERVE", "declared"),),
        (coor.build_optimization_risk_declaration(0, risk, "declared"),),
        lazy_plan_canonical_receipt=lr,
        schema_equivalence_receipt=sr,
    )


def test_hash_and_canonical_json_stability_and_idempotent_rebuild():
    assert coor._canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    r1, r2 = _build(), _build()
    assert r1.columnar_optimization_opportunity_receipt_hash == r2.columnar_optimization_opportunity_receipt_hash


def test_eligibility_semantics_and_recomputed_not_trusted():
    assert _build().optimization_eligible is True
    assert _build(status="BLOCKED").optimization_eligible is False
    assert _build(status="REQUIRES_EQUIVALENCE_PROOF").optimization_eligible is False
    assert _build(status="REQUIRES_SCHEMA_PROOF").optimization_eligible is False
    assert _build(status="DECLARED_UNSAFE").optimization_eligible is False
    assert _build(risk="REPLAY_CRITICAL").optimization_eligible is False
    assert _build(pre_ok=False).optimization_eligible is False
    good = _build()
    bad = type(good)(**{**good.__dict__, "optimization_eligible": False})
    with pytest.raises(ValueError, match="optimization_eligible"):
        coor.validate_columnar_optimization_opportunity_receipt(bad)


def test_dense_unique_indices_and_invalid_enums_and_hash_rejection():
    r = _build()
    with pytest.raises(ValueError):
        coor.validate_columnar_optimization_opportunity_receipt(type(r)(**{**r.__dict__, "preconditions": (coor.build_optimization_precondition(1, "x", True),), "precondition_count": 1, "columnar_optimization_opportunity_receipt_hash": coor._hash_payload(coor._base_payload({**r.__dict__, "preconditions": (coor.build_optimization_precondition(1, 'x', True),), "precondition_count": 1}, "columnar_optimization_opportunity_receipt_hash"))}))
    with pytest.raises(ValueError):
        coor.build_columnar_optimization_candidate("BAD", "ELIGIBLE")
    with pytest.raises(ValueError):
        coor.build_columnar_optimization_scope("BAD", (0,))
    with pytest.raises(ValueError):
        coor.build_optimization_risk_declaration(0, "BAD", "x")
    with pytest.raises(ValueError):
        coor.build_columnar_optimization_candidate("COLUMN_PRUNING", "BAD")
    with pytest.raises(ValueError):
        coor.build_optimization_constraint(0, "BAD", "x")
    with pytest.raises(ValueError):
        coor.validate_optimization_precondition(type(r.preconditions[0])(**{**r.preconditions[0].__dict__, "optimization_precondition_hash": "abc"}))


def test_adapter_only_child_validation_lineage_binding_and_forbidden_semantics():
    r = _build()
    with pytest.raises(ValueError, match="adapter_only"):
        coor.validate_columnar_optimization_opportunity_receipt(type(r)(**{**r.__dict__, "adapter_only": False}))
    lr, sr = _lazy_receipt(), _schema_receipt()
    with pytest.raises(ValueError, match="lazy plan lineage"):
        coor.build_columnar_optimization_opportunity_receipt("b" * 64, sr.schema_equivalence_receipt_hash, r.optimization_candidate, r.optimization_scope, r.preconditions, r.constraints, r.risks, lazy_plan_canonical_receipt=lr, schema_equivalence_receipt=sr)
    with pytest.raises(ValueError, match="schema equivalence lineage"):
        coor.build_columnar_optimization_opportunity_receipt(lr.lazy_plan_canonical_receipt_hash, "b" * 64, r.optimization_candidate, r.optimization_scope, r.preconditions, r.constraints, r.risks, lazy_plan_canonical_receipt=lr, schema_equivalence_receipt=sr)
    with pytest.raises(ValueError):
        coor.build_optimization_constraint(0, "ROW_ORDER_MUST_PRESERVE", "optimization authority")


def test_no_forbidden_imports_decoder_boundary_immutable_and_pythonhashseed_stability():
    text = open("src/qec/analysis/columnar_optimization_opportunity_receipts.py", encoding="utf-8").read()
    forbidden = ["import pandas", "import polars", "import pyarrow", "import scipy", "import matplotlib", "import qutip", "import qiskit", "import requests", "import urllib", "import subprocess", "eval(", "exec(", "os.system("]
    assert not any(tok in text for tok in forbidden)
    assert "from qec.decoder" not in text and "import qec.decoder" not in text
    with pytest.raises(Exception):
        coor._canonical_json({1: "x"})
    code = "from qec.analysis.columnar_optimization_opportunity_receipts import _hash_payload;print(_hash_payload({'a':1,'b':2}))"
    env = {**os.environ, "PYTHONPATH": "src", "PYTHONHASHSEED": "1"}
    a = subprocess.check_output([sys.executable, "-c", code], env=env, text=True).strip()
    b = subprocess.check_output([sys.executable, "-c", code], env=env, text=True).strip()
    assert a == b
