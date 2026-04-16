# SPDX-License-Identifier: MIT
"""Deterministic tests for v138.3.1 quadratic tension functional kernel."""

from __future__ import annotations

import json
import math

import pytest

from qec.runtime.quadratic_tension_functional_kernel import (
    QuadraticTensionFunctionalValidationError,
    compute_quadratic_tension,
    build_quadratic_tension_functional,
    quadratic_tension_projection,
    validate_quadratic_tension_functional,
)
from qec.runtime.runtime_admissibility_projection_engine import AdmissibleSubspace, project_runtime_state


def _projection(inadmissible: bool = True):
    subspace = AdmissibleSubspace(
        subspace_id="uft-id-subspace-01",
        dimension=3,
        basis_vectors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        metadata={"domain": "runtime", "version": "v138.3.0"},
    )
    state = (2.0, -1.5, 3.0) if inadmissible else (2.0, -1.5, 0.0)
    return project_runtime_state({"state_id": "state-001", "state": state}, subspace)


def test_same_input_same_bytes():
    a = build_quadratic_tension_functional(_projection())
    b = build_quadratic_tension_functional(_projection())
    assert a.to_canonical_json().encode("utf-8") == b.to_canonical_json().encode("utf-8")


def test_admissible_state_zero_tension():
    functional = build_quadratic_tension_functional(_projection(inadmissible=False))
    assert functional.admissible is True
    assert functional.tension_value <= 1e-12


def test_inadmissible_state_positive_tension():
    functional = build_quadratic_tension_functional(_projection(inadmissible=True))
    assert functional.admissible is False
    assert functional.tension_value > 0.0


def test_deterministic_term_ordering():
    terms, _ = compute_quadratic_tension((3.0, -1.0, 2.0))
    assert tuple((t.coordinate_index, t.term_id) for t in terms) == (
        (0, "term-000000"),
        (1, "term-000001"),
        (2, "term-000002"),
    )


def test_dimension_mismatch_rejection():
    projection = _projection().to_dict()
    projection["residual_vector"] = [0.0, 0.0]
    with pytest.raises(QuadraticTensionFunctionalValidationError):
        build_quadratic_tension_functional(projection)


def test_nan_inf_rejection():
    with pytest.raises(QuadraticTensionFunctionalValidationError):
        compute_quadratic_tension((1.0, math.nan))
    with pytest.raises(QuadraticTensionFunctionalValidationError):
        compute_quadratic_tension((1.0, math.inf))


def test_canonical_json_round_trip():
    functional = build_quadratic_tension_functional(_projection())
    payload = json.loads(functional.to_canonical_json())
    rebuilt = build_quadratic_tension_functional(
        {
            "state_id": payload["state_id"],
            "input_state": payload["input_state"],
            "projected_state": payload["projected_state"],
            "residual_vector": payload["residual_vector"],
            "admissible": payload["admissible"],
            "receipt": {"receipt_hash": payload["receipt"]["projection_receipt_hash"]},
        }
    )
    assert rebuilt.stable_hash() == functional.stable_hash()


def test_receipt_tamper_detection():
    functional = build_quadratic_tension_functional(_projection())
    payload = functional.to_dict()
    payload["receipt"]["tension_hash"] = "0" * 64
    report = validate_quadratic_tension_functional(payload)
    assert "receipt.tension_hash mismatch" in report.errors


def test_projection_stability():
    functional = build_quadratic_tension_functional(_projection())
    assert quadratic_tension_projection(functional) == quadratic_tension_projection(functional)


def test_recomputed_tension_equality():
    functional = build_quadratic_tension_functional(_projection())
    recomputed = sum(term.squared_component for term in functional.terms)
    assert math.isclose(functional.tension_value, recomputed, rel_tol=0.0, abs_tol=1e-12)


def test_malformed_projection_mapping_rejection():
    with pytest.raises(QuadraticTensionFunctionalValidationError):
        build_quadratic_tension_functional({"state_id": "x", "input_state": [1.0]})


def test_residual_term_non_negativity_enforcement():
    functional = build_quadratic_tension_functional(_projection())
    payload = functional.to_dict()
    payload["terms"][0]["squared_component"] = -1.0
    report = validate_quadratic_tension_functional(payload)
    assert any("squared_component must be >= 0" in err for err in report.errors)
