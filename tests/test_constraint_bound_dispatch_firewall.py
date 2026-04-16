# SPDX-License-Identifier: MIT
"""Deterministic tests for v138.3.3 constraint-bound dispatch firewall."""

from __future__ import annotations

import json
import math

import pytest

from qec.runtime.constraint_bound_dispatch_firewall import (
    ConstraintBoundDispatchFirewallValidationError,
    DispatchConstraint,
    build_constraint_bound_dispatch_firewall,
    dispatch_firewall_projection,
    evaluate_dispatch_constraints,
    validate_constraint_bound_dispatch_firewall,
)
from qec.runtime.deterministic_recovery_operator import build_deterministic_recovery_state
from qec.runtime.quadratic_tension_functional_kernel import build_quadratic_tension_functional
from qec.runtime.runtime_admissibility_projection_engine import AdmissibleSubspace, project_runtime_state


def _projection(state=(2.0, -1.5, 3.0)):
    subspace = AdmissibleSubspace(
        subspace_id="uft-id-subspace-01",
        dimension=3,
        basis_vectors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        metadata={"domain": "runtime", "version": "v138.3.0"},
    )
    return project_runtime_state({"state_id": "state-001", "state": state}, subspace)


def _lineage(state=(2.0, -1.5, 3.0)):
    projection = _projection(state)
    tension = build_quadratic_tension_functional(projection)
    recovery = build_deterministic_recovery_state(projection, tension)
    return recovery, tension


def _constraints(*, tension_threshold=10.0, recovery_threshold=10.0):
    return (
        DispatchConstraint(
            constraint_id="c-recovery",
            constraint_type="recovery_magnitude",
            threshold=float(recovery_threshold),
            operator="<=",
            metadata={"hard": True},
        ),
        DispatchConstraint(
            constraint_id="c-tension",
            constraint_type="tension_value",
            threshold=float(tension_threshold),
            operator="<=",
            metadata={"hard": True},
        ),
    )


def test_same_input_same_bytes():
    recovery, tension = _lineage()
    a = build_constraint_bound_dispatch_firewall(recovery, tension, _constraints())
    b = build_constraint_bound_dispatch_firewall(recovery, tension, _constraints())
    assert a.to_canonical_json().encode("utf-8") == b.to_canonical_json().encode("utf-8")


def test_allow_verdict():
    recovery, tension = _lineage(state=(2.0, -1.5, 0.0))
    firewall = build_constraint_bound_dispatch_firewall(recovery, tension, _constraints())
    assert firewall.verdict.verdict == "allow"


def test_deny_verdict():
    recovery, tension = _lineage()
    firewall = build_constraint_bound_dispatch_firewall(recovery, tension, _constraints(tension_threshold=0.0))
    assert firewall.verdict.verdict == "deny"


def test_recover_only_verdict():
    # Use raw dicts to construct a firewall with admissible=True upstream and
    # recovery_magnitude=5.0. This bypasses the normal lineage pipeline so we can
    # test the recover_only verdict path directly (with a real lineage, admissible=True
    # implies zero residual and therefore zero recovery_magnitude, which would give
    # an allow verdict instead).
    state_id = "test-recover-state"
    recovery_dict = {
        "state_id": state_id,
        "tension_value": 2.0,
        "recovery_magnitude": 5.0,
        "receipt": {
            "input_state_hash": "a" * 64,
            "recovery_hash": "b" * 64,
        },
    }
    tension_dict = {"state_id": state_id, "tension_value": 2.0, "admissible": True}
    firewall = build_constraint_bound_dispatch_firewall(recovery_dict, tension_dict, _constraints())
    assert firewall.recovery_magnitude > 1e-12
    assert firewall.verdict.verdict == "recover_only"


def test_deterministic_constraint_ordering():
    recovery, tension = _lineage()
    firewall = build_constraint_bound_dispatch_firewall(recovery, tension, tuple(reversed(_constraints())))
    assert tuple((c.constraint_type, c.constraint_id) for c in firewall.constraints) == (
        ("recovery_magnitude", "c-recovery"),
        ("tension_value", "c-tension"),
    )


def test_invalid_operator_rejection():
    recovery, tension = _lineage()
    bad = ({"constraint_id": "bad", "constraint_type": "tension_value", "threshold": 1.0, "operator": "!=", "metadata": {}},)
    with pytest.raises(ConstraintBoundDispatchFirewallValidationError):
        build_constraint_bound_dispatch_firewall(recovery, tension, bad)


def test_nan_inf_rejection():
    with pytest.raises(ConstraintBoundDispatchFirewallValidationError):
        evaluate_dispatch_constraints(math.nan, 0.0, _constraints())
    with pytest.raises(ConstraintBoundDispatchFirewallValidationError):
        evaluate_dispatch_constraints(0.0, math.inf, _constraints())


def test_canonical_json_round_trip():
    recovery, tension = _lineage()
    firewall = build_constraint_bound_dispatch_firewall(recovery, tension, _constraints())
    payload = json.loads(firewall.to_canonical_json())
    rebuilt = build_constraint_bound_dispatch_firewall(
        {
            "state_id": payload["state_id"],
            "tension_value": payload["tension_value"],
            "recovery_magnitude": payload["recovery_magnitude"],
            "receipt": {
                "input_state_hash": payload["receipt"]["input_state_hash"],
                "recovery_hash": payload["receipt"]["recovery_hash"],
            },
        },
        {
            "state_id": payload["state_id"],
            "tension_value": payload["tension_value"],
            "admissible": payload["upstream_admissible"],
        },
        payload["constraints"],
    )
    assert rebuilt.stable_hash() == firewall.stable_hash()


def test_receipt_tamper_detection():
    recovery, tension = _lineage()
    firewall = build_constraint_bound_dispatch_firewall(recovery, tension, _constraints())
    payload = firewall.to_dict()
    payload["receipt"]["firewall_hash"] = "0" * 64
    report = validate_constraint_bound_dispatch_firewall(payload)
    assert "receipt.firewall_hash mismatch" in report.errors


def test_projection_stability():
    recovery, tension = _lineage()
    firewall = build_constraint_bound_dispatch_firewall(recovery, tension, _constraints())
    assert dispatch_firewall_projection(firewall) == dispatch_firewall_projection(firewall)


def test_malformed_mapping_rejection():
    with pytest.raises(ConstraintBoundDispatchFirewallValidationError):
        build_constraint_bound_dispatch_firewall({"state_id": "x"}, {"state_id": "x"}, ())


def test_verdict_consistency_under_repeated_runs():
    recovery, tension = _lineage()
    expected = build_constraint_bound_dispatch_firewall(recovery, tension, _constraints()).verdict.to_dict()
    for _ in range(20):
        firewall = build_constraint_bound_dispatch_firewall(recovery, tension, _constraints())
        assert firewall.verdict.to_dict() == expected


def test_inadmissible_upstream_deny():
    # Default _lineage() produces tension.admissible=False (large z-residual).
    # The firewall must deny dispatch and expose upstream_admissible=False.
    recovery, tension = _lineage()
    firewall = build_constraint_bound_dispatch_firewall(recovery, tension, _constraints())
    assert firewall.verdict.verdict == "deny"
    assert not firewall.upstream_admissible
    assert firewall.verdict.reason == "upstream_admissibility_failure"


def test_soft_constraint_failure_gives_recover_only():
    # A constraint with hard=False should not deny; it should degrade to recover_only.
    state_id = "test-soft-fail"
    recovery_dict = {
        "state_id": state_id,
        "tension_value": 5.0,
        "recovery_magnitude": 0.0,
        "receipt": {"input_state_hash": "c" * 64, "recovery_hash": "d" * 64},
    }
    tension_dict = {"state_id": state_id, "tension_value": 5.0, "admissible": True}
    # tension_value=5.0 violates threshold=1.0 but constraint is soft (hard=False).
    soft_constraint = (
        DispatchConstraint(
            constraint_id="c-soft",
            constraint_type="tension_value",
            threshold=1.0,
            operator="<=",
            metadata={"hard": False},
        ),
    )
    firewall = build_constraint_bound_dispatch_firewall(recovery_dict, tension_dict, soft_constraint)
    assert firewall.verdict.verdict == "recover_only"


def test_metadata_none_raises_validation_error():
    with pytest.raises(ConstraintBoundDispatchFirewallValidationError):
        _evaluate_with_bad_metadata(metadata_value=None)


def test_metadata_non_mapping_raises_validation_error():
    with pytest.raises(ConstraintBoundDispatchFirewallValidationError):
        _evaluate_with_bad_metadata(metadata_value="not-a-mapping")


def _evaluate_with_bad_metadata(*, metadata_value):
    bad_constraint = {
        "constraint_id": "c-bad-meta",
        "constraint_type": "tension_value",
        "threshold": 10.0,
        "operator": "<=",
        "metadata": metadata_value,
    }
    evaluate_dispatch_constraints(1.0, 0.0, [bad_constraint])
