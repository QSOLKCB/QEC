# SPDX-License-Identifier: MIT
"""Deterministic tests for v138.3.4 hardware admissibility proof pack."""

from __future__ import annotations

import json

import pytest

from qec.runtime.constraint_bound_dispatch_firewall import (
    DispatchConstraint,
    build_constraint_bound_dispatch_firewall,
)
from qec.runtime.deterministic_recovery_operator import build_deterministic_recovery_state
from qec.runtime.hardware_admissibility_proof_pack import (
    HardwareAdmissibilityProofPackValidationError,
    build_hardware_admissibility_proof_pack,
    hardware_proof_projection,
    validate_hardware_admissibility_proof_pack,
)
from qec.runtime.quadratic_tension_functional_kernel import build_quadratic_tension_functional
from qec.runtime.runtime_admissibility_projection_engine import (
    AdmissibleSubspace,
    project_runtime_state,
)


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
    return projection, tension, recovery, firewall


def test_same_input_same_bytes():
    projection, tension, recovery, firewall = _lineage()
    a = build_hardware_admissibility_proof_pack(projection, tension, recovery, firewall)
    b = build_hardware_admissibility_proof_pack(projection, tension, recovery, firewall)
    assert a.to_canonical_json().encode("utf-8") == b.to_canonical_json().encode("utf-8")


def test_deterministic_component_ordering():
    projection, tension, recovery, firewall = _lineage()
    pack = build_hardware_admissibility_proof_pack(projection, tension, recovery, firewall)
    assert tuple((c.component_type, c.component_id) for c in pack.components) == (
        ("admissibility", "state-001:admissibility"),
        ("firewall", "state-001:firewall"),
        ("recovery", "state-001:recovery"),
        ("tension", "state-001:tension"),
    )


def test_duplicate_component_rejection():
    projection, tension, recovery, firewall = _lineage()
    pack = build_hardware_admissibility_proof_pack(projection, tension, recovery, firewall)
    payload = pack.to_dict()
    payload["components"][1]["component_id"] = payload["components"][0]["component_id"]
    report = validate_hardware_admissibility_proof_pack(payload)
    assert "duplicate component ids are not allowed" in report.errors


def test_lineage_mismatch_rejection():
    projection, tension, recovery, firewall = _lineage()
    bad_tension = tension.to_dict()
    bad_tension["state_id"] = "state-other"
    with pytest.raises(HardwareAdmissibilityProofPackValidationError, match="lineage mismatch"):
        build_hardware_admissibility_proof_pack(projection, bad_tension, recovery, firewall)


def test_invalid_verdict_rejection():
    projection, tension, recovery, firewall = _lineage()
    bad_firewall = firewall.to_dict()
    bad_firewall["verdict"]["verdict"] = "unsupported"
    with pytest.raises(HardwareAdmissibilityProofPackValidationError, match="unsupported verdict"):
        build_hardware_admissibility_proof_pack(projection, tension, recovery, bad_firewall)


def test_canonical_json_round_trip():
    projection, tension, recovery, firewall = _lineage()
    pack = build_hardware_admissibility_proof_pack(projection, tension, recovery, firewall)
    payload = json.loads(pack.to_canonical_json())
    rebuilt = build_hardware_admissibility_proof_pack(projection, tension, recovery, firewall)
    assert json.loads(rebuilt.to_canonical_json()) == payload


def test_receipt_tamper_detection():
    projection, tension, recovery, firewall = _lineage()
    pack = build_hardware_admissibility_proof_pack(projection, tension, recovery, firewall)
    payload = pack.to_dict()
    payload["receipt"]["proof_pack_hash"] = "0" * 64
    report = validate_hardware_admissibility_proof_pack(payload)
    assert "receipt.proof_pack_hash mismatch" in report.errors


def test_projection_stability():
    projection, tension, recovery, firewall = _lineage()
    pack = build_hardware_admissibility_proof_pack(projection, tension, recovery, firewall)
    assert hardware_proof_projection(pack) == hardware_proof_projection(pack)


def test_malformed_mapping_rejection():
    with pytest.raises(HardwareAdmissibilityProofPackValidationError):
        build_hardware_admissibility_proof_pack({"state_id": "x"}, {"state_id": "x"}, {"state_id": "x"}, {"state_id": "x"})


def test_repeated_run_proof_stability():
    projection, tension, recovery, firewall = _lineage()
    expected = build_hardware_admissibility_proof_pack(projection, tension, recovery, firewall).stable_hash()
    for _ in range(20):
        pack = build_hardware_admissibility_proof_pack(projection, tension, recovery, firewall)
        assert pack.stable_hash() == expected


def test_null_component_metadata_does_not_crash():
    projection, tension, recovery, firewall = _lineage()
    pack = build_hardware_admissibility_proof_pack(projection, tension, recovery, firewall)
    payload = pack.to_dict()
    payload["components"][0]["metadata"] = None
    report = validate_hardware_admissibility_proof_pack(payload)
    assert not report.valid


def test_missing_required_component_type_reports_error():
    projection, tension, recovery, firewall = _lineage()
    pack = build_hardware_admissibility_proof_pack(projection, tension, recovery, firewall)
    payload = pack.to_dict()
    payload["components"] = [c for c in payload["components"] if c["component_type"] != "tension"]
    report = validate_hardware_admissibility_proof_pack(payload)
    assert any("required component type missing: tension" in e for e in report.errors)
