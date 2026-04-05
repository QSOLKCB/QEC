"""Capability + Privilege Boundary Engine (v137.4.1).

Deterministic Layer-4 authorization kernel for capability grants,
privilege boundary decisions, replay-safe receipts, and canonical audit export.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Sequence

CAPABILITY_PRIVILEGE_BOUNDARY_ENGINE_VERSION: str = "v137.4.1"


# ---------------------------------------------------------------------------
# Canonical helpers
# ---------------------------------------------------------------------------

def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _canonical_bytes(payload: Any) -> bytes:
    return _canonical_json(payload).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _require_non_empty_str(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    return normalized


def _normalize_str_tuple(name: str, values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of strings")
    normalized: list[str] = []
    for idx, value in enumerate(values):
        normalized.append(_require_non_empty_str(f"{name}[{idx}]", value))
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    return tuple(sorted(set(normalized)))


def _validate_hash(name: str, value: Any) -> str:
    normalized = _require_non_empty_str(name, value)
    if len(normalized) != 64:
        raise ValueError(f"{name} must be a 64-char SHA-256 hex digest")
    for ch in normalized:
        if ch not in "0123456789abcdef":
            raise ValueError(f"{name} must be lowercase hex")
    return normalized


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CapabilityGrant:
    capability_id: str
    actor_id: str
    scope: tuple[str, ...]
    allowed_actions: tuple[str, ...]
    parent_sovereignty_event_hash: str
    schema_version: str = CAPABILITY_PRIVILEGE_BOUNDARY_ENGINE_VERSION
    capability_hash: str = field(init=False)

    def __post_init__(self) -> None:
        capability_id = _require_non_empty_str("capability_id", self.capability_id)
        actor_id = _require_non_empty_str("actor_id", self.actor_id)
        scope = _normalize_str_tuple("scope", self.scope)
        allowed_actions = _normalize_str_tuple("allowed_actions", self.allowed_actions)
        parent_hash = _validate_hash("parent_sovereignty_event_hash", self.parent_sovereignty_event_hash)
        schema_version = _require_non_empty_str("schema_version", self.schema_version)

        object.__setattr__(self, "capability_id", capability_id)
        object.__setattr__(self, "actor_id", actor_id)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "allowed_actions", allowed_actions)
        object.__setattr__(self, "parent_sovereignty_event_hash", parent_hash)
        object.__setattr__(self, "schema_version", schema_version)

        payload = {
            "capability_id": capability_id,
            "actor_id": actor_id,
            "scope": list(scope),
            "allowed_actions": list(allowed_actions),
            "parent_sovereignty_event_hash": parent_hash,
            "schema_version": schema_version,
        }
        object.__setattr__(self, "capability_hash", _sha256_hex(payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "actor_id": self.actor_id,
            "scope": list(self.scope),
            "allowed_actions": list(self.allowed_actions),
            "parent_sovereignty_event_hash": self.parent_sovereignty_event_hash,
            "schema_version": self.schema_version,
            "capability_hash": self.capability_hash,
        }


@dataclass(frozen=True)
class PrivilegeDecision:
    capability_id: str
    actor_id: str
    action_id: str
    scope: tuple[str, ...]
    privilege_verdict: str
    stable_decision_hash: str
    parent_sovereignty_event_hash: str
    schema_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "actor_id": self.actor_id,
            "action_id": self.action_id,
            "scope": list(self.scope),
            "privilege_verdict": self.privilege_verdict,
            "stable_decision_hash": self.stable_decision_hash,
            "parent_sovereignty_event_hash": self.parent_sovereignty_event_hash,
            "schema_version": self.schema_version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class CapabilityReceipt:
    decision_hash: str
    receipt_hash: str
    schema_version: str
    receipt_bytes: bytes


# ---------------------------------------------------------------------------
# Policy/authorization engine
# ---------------------------------------------------------------------------

def validate_capability_scope(capability_scope: Sequence[str], action_scope: Sequence[str]) -> bool:
    normalized_capability_scope = _normalize_str_tuple("capability_scope", capability_scope)
    normalized_action_scope = _normalize_str_tuple("action_scope", action_scope)
    return set(normalized_action_scope).issubset(set(normalized_capability_scope))


def _decision_hash_payload(
    capability_id: str,
    actor_id: str,
    action_id: str,
    scope: tuple[str, ...],
    privilege_verdict: str,
    parent_sovereignty_event_hash: str,
    schema_version: str,
) -> dict[str, Any]:
    return {
        "capability_id": capability_id,
        "actor_id": actor_id,
        "action_id": action_id,
        "scope": list(scope),
        "privilege_verdict": privilege_verdict,
        "parent_sovereignty_event_hash": parent_sovereignty_event_hash,
        "schema_version": schema_version,
    }


def authorize_action(
    grant: CapabilityGrant,
    actor_id: str,
    action_id: str,
    action_scope: Sequence[str],
) -> PrivilegeDecision:
    if not isinstance(grant, CapabilityGrant):
        raise ValueError("grant must be a CapabilityGrant")

    actor = _require_non_empty_str("actor_id", actor_id)
    action = _require_non_empty_str("action_id", action_id)
    normalized_scope = _normalize_str_tuple("action_scope", action_scope)

    scope_allowed = validate_capability_scope(grant.scope, normalized_scope)
    action_allowed = action in grant.allowed_actions
    actor_allowed = actor == grant.actor_id

    verdict = "ALLOW" if (scope_allowed and action_allowed and actor_allowed) else "DENY"
    payload = _decision_hash_payload(
        capability_id=grant.capability_id,
        actor_id=actor,
        action_id=action,
        scope=normalized_scope,
        privilege_verdict=verdict,
        parent_sovereignty_event_hash=grant.parent_sovereignty_event_hash,
        schema_version=grant.schema_version,
    )
    decision_hash = _sha256_hex(payload)

    return PrivilegeDecision(
        capability_id=grant.capability_id,
        actor_id=actor,
        action_id=action,
        scope=normalized_scope,
        privilege_verdict=verdict,
        stable_decision_hash=decision_hash,
        parent_sovereignty_event_hash=grant.parent_sovereignty_event_hash,
        schema_version=grant.schema_version,
    )


def generate_privilege_receipt(decision: PrivilegeDecision) -> CapabilityReceipt:
    if not isinstance(decision, PrivilegeDecision):
        raise ValueError("decision must be a PrivilegeDecision")

    decision_payload = decision.to_dict()
    decision_bytes = _canonical_bytes(decision_payload)
    receipt_payload = {
        "decision_hash": decision.stable_decision_hash,
        "decision_payload": decision_payload,
        "schema_version": decision.schema_version,
    }
    receipt_hash = _sha256_hex(receipt_payload)

    return CapabilityReceipt(
        decision_hash=decision.stable_decision_hash,
        receipt_hash=receipt_hash,
        schema_version=decision.schema_version,
        receipt_bytes=decision_bytes,
    )


def replay_privilege_decision(decision_bytes: bytes, grant: CapabilityGrant) -> PrivilegeDecision:
    if not isinstance(decision_bytes, bytes) or len(decision_bytes) == 0:
        raise ValueError("decision_bytes must be non-empty bytes")
    if not isinstance(grant, CapabilityGrant):
        raise ValueError("grant must be a CapabilityGrant")

    try:
        payload = json.loads(decision_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("decision_bytes must be UTF-8 canonical JSON") from exc

    capability_id = _require_non_empty_str("payload.capability_id", payload.get("capability_id"))
    actor_id = _require_non_empty_str("payload.actor_id", payload.get("actor_id"))
    action_id = _require_non_empty_str("payload.action_id", payload.get("action_id"))
    scope = _normalize_str_tuple("payload.scope", payload.get("scope", ()))
    verdict = _require_non_empty_str("payload.privilege_verdict", payload.get("privilege_verdict"))
    parent_hash = _validate_hash(
        "payload.parent_sovereignty_event_hash", payload.get("parent_sovereignty_event_hash")
    )
    schema_version = _require_non_empty_str("payload.schema_version", payload.get("schema_version"))

    if capability_id != grant.capability_id:
        raise ValueError("decision capability_id does not match grant")
    if parent_hash != grant.parent_sovereignty_event_hash:
        raise ValueError("decision parent sovereignty hash does not match grant")
    if schema_version != grant.schema_version:
        raise ValueError("decision schema_version does not match grant")

    replayed = authorize_action(
        grant=grant,
        actor_id=actor_id,
        action_id=action_id,
        action_scope=scope,
    )
    if replayed.privilege_verdict != verdict:
        raise ValueError("decision verdict mismatch on replay")

    claimed_hash = _require_non_empty_str("payload.stable_decision_hash", payload.get("stable_decision_hash"))
    if replayed.stable_decision_hash != claimed_hash:
        raise ValueError("decision hash mismatch on replay")

    return replayed


def export_capability_audit_bytes(decisions: Sequence[PrivilegeDecision]) -> bytes:
    if isinstance(decisions, (str, bytes)):
        raise ValueError("decisions must be a sequence of PrivilegeDecision")

    normalized: list[PrivilegeDecision] = []
    for idx, decision in enumerate(decisions):
        if not isinstance(decision, PrivilegeDecision):
            raise ValueError(f"decisions[{idx}] must be a PrivilegeDecision")
        normalized.append(decision)

    ordered = sorted(
        normalized,
        key=lambda d: (
            d.stable_decision_hash,
            d.capability_id,
            d.actor_id,
            d.action_id,
            d.scope,
            d.privilege_verdict,
            d.parent_sovereignty_event_hash,
            d.schema_version,
        ),
    )
    payload = {
        "schema_version": CAPABILITY_PRIVILEGE_BOUNDARY_ENGINE_VERSION,
        "decision_count": len(ordered),
        "decisions": [item.to_dict() for item in ordered],
        "audit_hash": _sha256_hex([item.to_dict() for item in ordered]),
    }
    return _canonical_bytes(payload)


__all__ = [
    "CAPABILITY_PRIVILEGE_BOUNDARY_ENGINE_VERSION",
    "CapabilityGrant",
    "PrivilegeDecision",
    "CapabilityReceipt",
    "authorize_action",
    "validate_capability_scope",
    "generate_privilege_receipt",
    "export_capability_audit_bytes",
    "replay_privilege_decision",
]
