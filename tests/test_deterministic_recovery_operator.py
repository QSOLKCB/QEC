# SPDX-License-Identifier: MIT
"""Deterministic tests for v138.3.2 deterministic recovery operator."""

from __future__ import annotations

import json
import math

import pytest

from qec.runtime.deterministic_recovery_operator import (
    DeterministicRecoveryOperatorValidationError,
    compute_recovery_state,
    build_deterministic_recovery_state,
    recovery_projection,
    validate_deterministic_recovery_state,
)
from qec.runtime.quadratic_tension_functional_kernel import build_quadratic_tension_functional
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


def _recovery(inadmissible: bool = True):
    projection = _projection(inadmissible=inadmissible)
    tension = build_quadratic_tension_functional(projection)
    return build_deterministic_recovery_state(projection, tension)


def test_same_input_same_bytes():
    a = _recovery()
    b = _recovery()
    assert a.to_canonical_json().encode("utf-8") == b.to_canonical_json().encode("utf-8")


def test_zero_tension_identity_recovery():
    recovery = _recovery(inadmissible=False)
    assert recovery.tension_value <= 1e-12
    assert recovery.recovered_state == recovery.projected_state


def test_positive_tension_projection_recovery():
    recovery = _recovery(inadmissible=True)
    assert recovery.tension_value > 0.0
    assert recovery.recovered_state == recovery.projected_state


def test_deterministic_step_ordering():
    recovery = _recovery()
    expected = tuple((i, f"recovery-step-{i:06d}") for i in range(len(recovery.recovery_steps)))
    assert tuple((s.coordinate_index, s.step_id) for s in recovery.recovery_steps) == expected


def test_dimension_mismatch_rejection():
    with pytest.raises(DeterministicRecoveryOperatorValidationError):
        compute_recovery_state((1.0, 2.0), (1.0,), tension_value=0.0)


def test_nan_inf_rejection():
    with pytest.raises(DeterministicRecoveryOperatorValidationError):
        compute_recovery_state((1.0, math.nan), (1.0, 2.0), tension_value=0.0)
    with pytest.raises(DeterministicRecoveryOperatorValidationError):
        compute_recovery_state((1.0, 2.0), (1.0, 2.0), tension_value=math.inf)


def test_canonical_json_round_trip():
    recovery = _recovery()
    payload = json.loads(recovery.to_canonical_json())
    rebuilt = build_deterministic_recovery_state(
        {
            "state_id": payload["state_id"],
            "input_state": payload["input_state"],
            "projected_state": payload["projected_state"],
        },
        {
            "state_id": payload["state_id"],
            "input_state": payload["input_state"],
            "projected_state": payload["projected_state"],
            "tension_value": payload["tension_value"],
            "receipt": {"tension_hash": payload["receipt"]["tension_hash"]},
        },
    )
    assert rebuilt.stable_hash() == recovery.stable_hash()


def test_receipt_tamper_detection():
    recovery = _recovery()
    payload = recovery.to_dict()
    payload["receipt"]["recovery_hash"] = "0" * 64
    report = validate_deterministic_recovery_state(payload)
    assert "receipt.recovery_hash mismatch" in report.errors


def test_projection_stability():
    recovery = _recovery()
    assert recovery_projection(recovery) == recovery_projection(recovery)


def test_recovery_magnitude_equality():
    recovery = _recovery()
    recomputed = math.sqrt(sum(step.delta * step.delta for step in recovery.recovery_steps))
    assert math.isclose(recovery.recovery_magnitude, recomputed, rel_tol=0.0, abs_tol=1e-12)


def test_malformed_mapping_rejection():
    with pytest.raises(DeterministicRecoveryOperatorValidationError):
        build_deterministic_recovery_state({"state_id": "x"}, {"state_id": "x"})


def test_step_delta_consistency():
    recovery = _recovery()
    payload = recovery.to_dict()
    payload["recovery_steps"][0]["delta"] = 999.0
    report = validate_deterministic_recovery_state(payload)
    assert any("delta mismatch" in error for error in report.errors)


def test_forged_tension_value_rejected():
    """P1: builder must reject a tension_value that does not match state geometry."""
    projection = _projection(inadmissible=True)
    tension = build_quadratic_tension_functional(projection)
    tension_map = tension.to_dict()
    # Forge a zero tension even though the residual is nonzero.
    tension_map["tension_value"] = 0.0
    with pytest.raises(DeterministicRecoveryOperatorValidationError, match="does not match state geometry"):
        build_deterministic_recovery_state(projection, tension_map)


def test_validate_invalid_type_returns_report():
    """P2: validate_deterministic_recovery_state must not crash on non-Mapping input."""
    for bad_input in (42, None, [1, 2, 3]):
        report = validate_deterministic_recovery_state(bad_input)  # type: ignore[arg-type]
        assert not report.valid
        assert report.error_count >= 1
        assert any("Mapping" in err or "DeterministicRecoveryState" in err for err in report.errors)


def test_duplicate_coordinate_index_rejected():
    """Validator must reject a payload with duplicate coordinate indices."""
    recovery = _recovery()
    payload = recovery.to_dict()
    # Duplicate step 0's index onto step 1 — creates a gap and a duplicate.
    payload["recovery_steps"][1]["coordinate_index"] = 0
    report = validate_deterministic_recovery_state(payload)
    assert any("coordinate_index" in err for err in report.errors)
