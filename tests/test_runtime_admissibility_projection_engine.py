# SPDX-License-Identifier: MIT
"""Deterministic tests for v138.3.0 runtime admissibility projection engine."""

from __future__ import annotations

import json
import math

import pytest

from qec.runtime.runtime_admissibility_projection_engine import (
    AdmissibleSubspace,
    ProjectionProofReceipt,
    RuntimeAdmissibilityProjectionValidationError,
    compute_runtime_residual,
    project_runtime_state,
    runtime_projection_summary,
    validate_runtime_projection,
)


def _subspace() -> AdmissibleSubspace:
    return AdmissibleSubspace(
        subspace_id="uft-id-subspace-01",
        dimension=3,
        basis_vectors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        metadata={"domain": "runtime", "version": "v138.3.0"},
    )


def _runtime_state():
    return {"state_id": "state-001", "state": (2.0, -1.5, 0.0)}


def test_same_input_same_bytes():
    a = project_runtime_state(_runtime_state(), _subspace())
    b = project_runtime_state(_runtime_state(), _subspace())
    assert a.to_canonical_json().encode("utf-8") == b.to_canonical_json().encode("utf-8")


def test_admissible_projection_success():
    projected = project_runtime_state({"state_id": "state-ok", "state": (1.0, 2.0, 0.0)}, _subspace())
    assert projected.residual.admissible is True
    assert projected.projected_state == (1.0, 2.0, 0.0)
    assert projected.validation_errors == ()


def test_inadmissible_state_projection():
    projected = project_runtime_state({"state_id": "state-bad", "state": (1.0, 2.0, 3.0)}, _subspace())
    assert projected.residual.admissible is False
    assert projected.projected_state == (1.0, 2.0, 0.0)
    assert projected.residual.residual_vector == (0.0, 0.0, 3.0)


def test_dimension_mismatch_rejection():
    with pytest.raises(RuntimeAdmissibilityProjectionValidationError):
        project_runtime_state({"state_id": "bad-dim", "state": (1.0, 2.0)}, _subspace())


def test_nan_inf_rejection():
    with pytest.raises(RuntimeAdmissibilityProjectionValidationError):
        project_runtime_state({"state_id": "nan-state", "state": (1.0, math.nan, 0.0)}, _subspace())

    with pytest.raises(RuntimeAdmissibilityProjectionValidationError):
        compute_runtime_residual((1.0, 2.0, 3.0), ((1.0, 0.0, 0.0), (0.0, math.inf, 0.0)))


def test_empty_basis_rejection():
    with pytest.raises(RuntimeAdmissibilityProjectionValidationError):
        compute_runtime_residual((1.0, 2.0), ())


def test_canonical_json_round_trip():
    projection = project_runtime_state(_runtime_state(), _subspace())
    payload = json.loads(projection.to_canonical_json())
    rebuilt = project_runtime_state(
        {"state_id": payload["state_id"], "state": tuple(payload["input_state"])},
        _subspace(),
    )
    assert rebuilt.receipt.receipt_hash == projection.receipt.receipt_hash


def test_receipt_tamper_detection():
    projection = project_runtime_state(_runtime_state(), _subspace())
    forged = ProjectionProofReceipt(
        input_state_hash=projection.receipt.input_state_hash,
        projected_state_hash=projection.receipt.projected_state_hash,
        subspace_hash=projection.receipt.subspace_hash,
        admissible=False,
        proof_hash=projection.receipt.proof_hash,
        receipt_hash="",
    )
    tampered = {
        **projection.to_dict(),
        "receipt": {
            **projection.receipt.to_dict(),
            "admissible": False,
            "receipt_hash": forged.stable_hash(),
        },
    }
    errors = validate_runtime_projection(tampered)
    assert "receipt.proof_hash lineage mismatch" in errors


def test_projection_summary_stability():
    projection = project_runtime_state(_runtime_state(), _subspace())
    assert runtime_projection_summary(projection) == runtime_projection_summary(projection)


def test_deterministic_residual_computation():
    first = compute_runtime_residual((1.0, 2.0, 3.0), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    second = compute_runtime_residual((1.0, 2.0, 3.0), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    assert first.residual_vector == (0.0, 0.0, 3.0)
    assert first.residual_norm == 3.0
    assert first.to_canonical_json() == second.to_canonical_json()
