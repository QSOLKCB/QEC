"""Deterministic tests for v137.4.1 capability + privilege boundary engine."""

from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest

from qec.analysis.capability_privilege_boundary_engine import (
    CAPABILITY_PRIVILEGE_BOUNDARY_ENGINE_VERSION,
    CapabilityGrant,
    authorize_action,
    export_capability_audit_bytes,
    generate_privilege_receipt,
    replay_privilege_decision,
    validate_capability_scope,
)


def _h(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _grant() -> CapabilityGrant:
    return CapabilityGrant(
        capability_id="cap-001",
        actor_id="actor-7",
        scope=("sys:queue", "sys:queue:write", "sys:queue:read"),
        allowed_actions=("enqueue", "dequeue"),
        parent_sovereignty_event_hash=_h("sovereignty-root"),
    )


def test_frozen_dataclasses() -> None:
    grant = _grant()
    with pytest.raises(dataclasses.FrozenInstanceError):
        grant.actor_id = "other"  # type: ignore[misc]


def test_repeated_run_determinism() -> None:
    grant = _grant()
    reference = authorize_action(grant, "actor-7", "enqueue", ("sys:queue",))
    for _ in range(100):
        replay = authorize_action(grant, "actor-7", "enqueue", ("sys:queue",))
        assert replay == reference
        assert replay.to_canonical_bytes() == reference.to_canonical_bytes()


def test_identical_inputs_identical_decision_bytes() -> None:
    grant = _grant()
    d1 = authorize_action(grant, "actor-7", "enqueue", ("sys:queue", "sys:queue"))
    d2 = authorize_action(grant, "actor-7", "enqueue", ("sys:queue",))
    assert d1.to_canonical_bytes() == d2.to_canonical_bytes()


def test_stable_decision_hash() -> None:
    grant = _grant()
    decision = authorize_action(grant, "actor-7", "dequeue", ("sys:queue:read",))
    decision_again = authorize_action(grant, "actor-7", "dequeue", ("sys:queue:read",))
    assert decision.stable_decision_hash == decision_again.stable_decision_hash


def test_privilege_boundary_enforcement() -> None:
    grant = _grant()
    denied = authorize_action(grant, "actor-7", "delete", ("sys:queue",))
    assert denied.privilege_verdict == "DENY"


def test_scope_mismatch_rejection() -> None:
    grant = _grant()
    denied = authorize_action(grant, "actor-7", "enqueue", ("sys:admin",))
    assert denied.privilege_verdict == "DENY"


def test_validate_capability_scope() -> None:
    assert validate_capability_scope(("a", "b", "c"), ("a", "b")) is True
    assert validate_capability_scope(("a", "b"), ("a", "z")) is False


def test_receipt_stability() -> None:
    grant = _grant()
    decision = authorize_action(grant, "actor-7", "enqueue", ("sys:queue",))
    r1 = generate_privilege_receipt(decision)
    r2 = generate_privilege_receipt(decision)
    assert r1 == r2
    assert hashlib.sha256(r1.receipt_bytes).hexdigest() == r1.receipt_hash


def test_replay_rejects_non_canonical_json() -> None:
    grant = _grant()
    decision = authorize_action(grant, "actor-7", "enqueue", ("sys:queue",))
    payload = json.loads(decision.to_canonical_bytes())
    non_canonical = json.dumps(payload, indent=2).encode("utf-8")
    with pytest.raises(ValueError, match="canonical"):
        replay_privilege_decision(non_canonical, grant)


def test_replay_fidelity() -> None:
    grant = _grant()
    decision = authorize_action(grant, "actor-7", "enqueue", ("sys:queue", "sys:queue:write"))
    replayed = replay_privilege_decision(decision.to_canonical_bytes(), grant)
    assert replayed == decision


def test_export_capability_audit_bytes_deterministic_ordering() -> None:
    grant = _grant()
    d1 = authorize_action(grant, "actor-7", "enqueue", ("sys:queue",))
    d2 = authorize_action(grant, "actor-7", "dequeue", ("sys:queue:read",))
    b1 = export_capability_audit_bytes((d1, d2))
    b2 = export_capability_audit_bytes((d2, d1))
    assert b1 == b2


def test_decision_law_fields_present() -> None:
    grant = _grant()
    decision = authorize_action(grant, "actor-7", "enqueue", ("sys:queue",))
    payload = json.loads(decision.to_canonical_bytes().decode("utf-8"))
    required = {
        "capability_id",
        "actor_id",
        "action_id",
        "scope",
        "privilege_verdict",
        "stable_decision_hash",
        "parent_sovereignty_event_hash",
        "schema_version",
    }
    assert required.issubset(payload.keys())
    assert payload["schema_version"] == CAPABILITY_PRIVILEGE_BOUNDARY_ENGINE_VERSION


def test_fail_fast_invalid_input_handling() -> None:
    with pytest.raises(ValueError):
        CapabilityGrant(
            capability_id="",
            actor_id="actor-7",
            scope=("sys:queue",),
            allowed_actions=("enqueue",),
            parent_sovereignty_event_hash=_h("root"),
        )

    grant = _grant()
    with pytest.raises(ValueError):
        authorize_action(grant, "", "enqueue", ("sys:queue",))
    with pytest.raises(ValueError):
        validate_capability_scope(("a",), ())
    with pytest.raises(ValueError):
        replay_privilege_decision(b"", grant)
    with pytest.raises(ValueError):
        export_capability_audit_bytes(("not-a-decision",))  # type: ignore[arg-type]
