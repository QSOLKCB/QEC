# SPDX-License-Identifier: MIT
"""Deterministic tests for v138.3.5 proof-carrying runtime bridge."""

from __future__ import annotations

import json

import pytest

from qec.runtime.constraint_bound_dispatch_firewall import (
    DispatchConstraint,
    build_constraint_bound_dispatch_firewall,
)
from qec.runtime.hardware_admissibility_proof_pack import build_hardware_admissibility_proof_pack
from qec.runtime.proof_carrying_runtime_bridge import (
    ProofCarryingRuntimeBridgeValidationError,
    build_proof_carrying_runtime_bridge,
    runtime_bridge_projection,
    validate_proof_carrying_runtime_bridge,
)
from qec.runtime.deterministic_recovery_operator import build_deterministic_recovery_state
from qec.runtime.quadratic_tension_functional_kernel import build_quadratic_tension_functional
from qec.runtime.runtime_admissibility_projection_engine import AdmissibleSubspace, project_runtime_state


def _projection(state=(2.0, -1.5, 0.0)):
    subspace = AdmissibleSubspace(
        subspace_id="uft-id-subspace-01",
        dimension=3,
        basis_vectors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        metadata={"domain": "runtime", "version": "v138.3.0"},
    )
    return project_runtime_state({"state_id": "state-001", "state": state}, subspace)


def _lineage_allow(state=(2.0, -1.5, 0.0)):
    projection = _projection(state)
    tension = build_quadratic_tension_functional(projection)
    recovery = build_deterministic_recovery_state(projection, tension)
    constraints = (
        DispatchConstraint(
            constraint_id="c-recovery",
            constraint_type="recovery_magnitude",
            threshold=10.0,
            operator="<=",
            metadata={"hard": True},
        ),
        DispatchConstraint(
            constraint_id="c-tension",
            constraint_type="tension_value",
            threshold=10.0,
            operator="<=",
            metadata={"hard": True},
        ),
    )
    firewall = build_constraint_bound_dispatch_firewall(recovery, tension, constraints)
    proof_pack = build_hardware_admissibility_proof_pack(projection, tension, recovery, firewall)
    return proof_pack, firewall


def _lineage_deny():
    projection = _projection()
    tension = build_quadratic_tension_functional(projection)
    recovery = build_deterministic_recovery_state(projection, tension)
    # threshold=1000.0 with ">=" is guaranteed to fail for any realistic recovery_magnitude
    constraints = (
        DispatchConstraint(
            constraint_id="c-deny",
            constraint_type="recovery_magnitude",
            threshold=1000.0,
            operator=">=",
            metadata={"hard": True},
        ),
    )
    firewall = build_constraint_bound_dispatch_firewall(recovery, tension, constraints)
    proof_pack = build_hardware_admissibility_proof_pack(projection, tension, recovery, firewall)
    return proof_pack, firewall


def _lineage_recover_only():
    projection = _projection()
    tension = build_quadratic_tension_functional(projection)
    recovery = build_deterministic_recovery_state(projection, tension)
    # same infeasible threshold as deny but with hard=False → soft failure → recover_only verdict
    constraints = (
        DispatchConstraint(
            constraint_id="c-soft",
            constraint_type="recovery_magnitude",
            threshold=1000.0,
            operator=">=",
            metadata={"hard": False},
        ),
    )
    firewall = build_constraint_bound_dispatch_firewall(recovery, tension, constraints)
    proof_pack = build_hardware_admissibility_proof_pack(projection, tension, recovery, firewall)
    return proof_pack, firewall


def _bridge_payload_with_verdict(verdict: str) -> dict:
    if verdict == "deny":
        proof_pack, firewall = _lineage_deny()
    elif verdict == "recover_only":
        proof_pack, firewall = _lineage_recover_only()
    else:
        proof_pack, firewall = _lineage_allow()
    bridge = build_proof_carrying_runtime_bridge(proof_pack, firewall)
    return bridge.to_dict()


def test_same_input_same_bytes():
    proof_pack, firewall = _lineage_allow()
    a = build_proof_carrying_runtime_bridge(proof_pack, firewall)
    b = build_proof_carrying_runtime_bridge(proof_pack, firewall)
    assert a.to_canonical_json().encode("utf-8") == b.to_canonical_json().encode("utf-8")


def test_allow_authorized():
    proof_pack, firewall = _lineage_allow()
    bridge = build_proof_carrying_runtime_bridge(proof_pack, firewall)
    assert bridge.verdict == "allow"
    assert bridge.authorized


def test_deny_unauthorized():
    payload = _bridge_payload_with_verdict("deny")
    report = validate_proof_carrying_runtime_bridge(payload)
    assert report.valid
    assert payload["authorized"] is False


def test_recover_only_unauthorized():
    payload = _bridge_payload_with_verdict("recover_only")
    report = validate_proof_carrying_runtime_bridge(payload)
    assert report.valid
    assert payload["authorized"] is False


def test_lineage_mismatch_rejection():
    proof_pack, firewall = _lineage_allow()
    bad_firewall = firewall.to_dict()
    bad_firewall["state_id"] = "state-other"
    with pytest.raises(ProofCarryingRuntimeBridgeValidationError, match="upstream validation|lineage mismatch"):
        build_proof_carrying_runtime_bridge(proof_pack, bad_firewall)


def test_invalid_verdict_rejection():
    proof_pack, firewall = _lineage_allow()
    bad_firewall = firewall.to_dict()
    bad_firewall["verdict"]["verdict"] = "invalid"
    with pytest.raises(ProofCarryingRuntimeBridgeValidationError, match="upstream validation|lineage mismatch|unsupported"):
        build_proof_carrying_runtime_bridge(proof_pack, bad_firewall)


def test_canonical_json_round_trip():
    proof_pack, firewall = _lineage_allow()
    bridge = build_proof_carrying_runtime_bridge(proof_pack, firewall)
    payload = json.loads(bridge.to_canonical_json())
    assert json.loads(json.dumps(payload, sort_keys=True, separators=(",", ":"))) == payload


def test_receipt_tamper_detection():
    proof_pack, firewall = _lineage_allow()
    bridge = build_proof_carrying_runtime_bridge(proof_pack, firewall)
    payload = bridge.to_dict()
    payload["receipt"]["bridge_hash"] = "0" * 64
    report = validate_proof_carrying_runtime_bridge(payload)
    assert "receipt.bridge_hash mismatch" in report.errors


def test_projection_stability():
    proof_pack, firewall = _lineage_allow()
    bridge = build_proof_carrying_runtime_bridge(proof_pack, firewall)
    assert runtime_bridge_projection(bridge) == runtime_bridge_projection(bridge)


def test_repeated_run_bridge_stability():
    proof_pack, firewall = _lineage_allow()
    expected = build_proof_carrying_runtime_bridge(proof_pack, firewall).stable_hash()
    for _ in range(20):
        bridge = build_proof_carrying_runtime_bridge(proof_pack, firewall)
        assert bridge.stable_hash() == expected
