"""This module adheres to the QEC identity and hashing surface contract.

See: qec.analysis.identity_hashing_contract.get_identity_hashing_contract()
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qec.analysis.canonical_hashing import CanonicalHashingError, canonicalize_json, sha256_hex
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
        try:
            canonical_value = canonicalize_json(value)
        except CanonicalHashingError as exc:
            raise _invalid_input() from exc
        out.append((key, canonical_value))
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

    @classmethod
    def create(cls, *, agent_id: str, agent_role: str, decision_payload: tuple[tuple[str, Any], ...]) -> "RoleAgentDecision":
        role = _validate_role(agent_role)
        payload = _canonical_payload(decision_payload)
        decision_hash = _decision_payload_hash(
            agent_id=agent_id,
            agent_role=role,
            decision_payload=payload,
        )
        return cls(
            agent_id=agent_id,
            agent_role=role,
            decision_hash=decision_hash,
            decision_payload=payload,
        )

    def __post_init__(self) -> None:
        if not isinstance(self.agent_id, str) or not self.agent_id:
            raise _invalid_input()
        role = _validate_role(self.agent_role)
        payload = _canonical_payload(self.decision_payload)
        object.__setattr__(self, "agent_role", role)
        object.__setattr__(self, "decision_payload", payload)
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

    def __post_init__(self) -> None:
        prev_key: tuple[str, str, str] | None = None
        for decision in self.decisions:
            if not isinstance(decision, RoleAgentDecision):
                raise _invalid_input()
            key = (decision.agent_role, decision.decision_hash, decision.agent_id)
            if prev_key is not None and key <= prev_key:
                raise _invalid_input()
            prev_key = key


@dataclass(frozen=True)
class RoleDecisionReceipt:
    role_state: RoleDecisionState
    input_memory_hashes: tuple[str, ...]
    role_decision_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.role_state, RoleDecisionState):
            raise _invalid_input()
        identity = canonical_hash_identity(self.input_memory_hashes)
        object.__setattr__(self, "input_memory_hashes", identity)
        expected_hash = _role_decision_hash(input_memory_hashes=identity, role_state=self.role_state)
        if self.role_decision_hash != expected_hash:
            raise _invalid_input()


def _role_decision_hash(*, input_memory_hashes: tuple[str, ...], role_state: RoleDecisionState) -> str:
    return sha256_hex(
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
            "input_memory_hashes": input_memory_hashes,
        }
    )


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
    role_decision_hash = _role_decision_hash(input_memory_hashes=identity, role_state=role_state)
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
