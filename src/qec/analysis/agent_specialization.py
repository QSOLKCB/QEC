"""Deterministic role-constrained agent specialization for analysis receipts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qec.analysis.canonical_hashing import canonicalize_json, sha256_hex
from qec.analysis.canonical_identity import canonical_hash_identity


class AgentRole(str):
    CONTROL = "control"
    VALIDATION = "validation"
    REPAIR = "repair"
    ADVERSARIAL = "adversarial"
    COMPRESSION = "compression"


_ALLOWED_ROLES = frozenset(
    {
        AgentRole.CONTROL,
        AgentRole.VALIDATION,
        AgentRole.REPAIR,
        AgentRole.ADVERSARIAL,
        AgentRole.COMPRESSION,
    }
)


def _invalid_input() -> ValueError:
    return ValueError("INVALID_INPUT")


def _validate_role(role: object) -> str:
    if isinstance(role, str) and role in _ALLOWED_ROLES:
        return role
    raise _invalid_input()


def _canonical_payload(payload: object) -> tuple[tuple[str, Any], ...]:
    if not isinstance(payload, tuple):
        raise _invalid_input()
    out: list[tuple[str, Any]] = []
    prev_key: str | None = None
    for item in payload:
        if not isinstance(item, tuple) or len(item) != 2:
            raise _invalid_input()
        key, value = item
        if not isinstance(key, str):
            raise _invalid_input()
        if prev_key is not None and key <= prev_key:
            raise _invalid_input()
        prev_key = key
        out.append((key, canonicalize_json(value)))
    return tuple(out)


def _decision_payload_hash(*, agent_id: str, agent_role: str, decision_payload: tuple[tuple[str, Any], ...]) -> str:
    return sha256_hex(
        {
            "agent_id": agent_id,
            "agent_role": agent_role,
            "decision_payload": decision_payload,
        }
    )


@dataclass(frozen=True)
class RoleAgentDecision:
    agent_id: str
    agent_role: str
    decision_hash: str
    decision_payload: tuple[tuple[str, Any], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.agent_id, str) or not self.agent_id:
            raise _invalid_input()
        role = _validate_role(self.agent_role)
        payload = _canonical_payload(self.decision_payload)
        expected_hash = _decision_payload_hash(
            agent_id=self.agent_id,
            agent_role=role,
            decision_payload=payload,
        )
        if self.decision_hash != expected_hash:
            raise _invalid_input()


@dataclass(frozen=True)
class RoleDecisionState:
    decisions: tuple[RoleAgentDecision, ...]


@dataclass(frozen=True)
class RoleDecisionReceipt:
    role_state: RoleDecisionState
    input_memory_hashes: tuple[str, ...]
    role_decision_hash: str


def build_role_decision_state(
    input_memory_hashes: tuple[str, ...],
    decisions: tuple[RoleAgentDecision, ...] | list[RoleAgentDecision],
) -> RoleDecisionReceipt:
    identity = canonical_hash_identity(input_memory_hashes)
    role_to_hash: dict[str, str] = {}
    unique: dict[tuple[str, str], RoleAgentDecision] = {}

    for decision in decisions:
        if not isinstance(decision, RoleAgentDecision):
            raise _invalid_input()
        role = _validate_role(decision.agent_role)
        existing_hash = role_to_hash.get(role)
        if existing_hash is None:
            role_to_hash[role] = decision.decision_hash
        elif existing_hash != decision.decision_hash:
            raise _invalid_input()
        key = (role, decision.decision_hash)
        if key in unique:
            raise _invalid_input()
        unique[key] = decision

    ordered = tuple(sorted(unique.values(), key=lambda item: (item.agent_role, item.decision_hash, item.agent_id)))
    role_state = RoleDecisionState(decisions=ordered)
    role_decision_hash = sha256_hex(
        {
            "decisions": [
                {
                    "agent_id": item.agent_id,
                    "agent_role": item.agent_role,
                    "decision_hash": item.decision_hash,
                    "decision_payload": item.decision_payload,
                }
                for item in role_state.decisions
            ],
            "input_memory_hashes": identity,
        }
    )
    return RoleDecisionReceipt(
        role_state=role_state,
        input_memory_hashes=identity,
        role_decision_hash=role_decision_hash,
    )


__all__ = [
    "AgentRole",
    "RoleAgentDecision",
    "RoleDecisionState",
    "RoleDecisionReceipt",
    "build_role_decision_state",
]
