"""v139.3 — Cross-Node Recovery Runtime.

Deterministic analysis-only cross-node recovery receipt generation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

from qec.analysis.canonical_hashing import canonical_bytes, canonical_json, sha256_hex

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

ALLOWED_RECOVERY_ACTION_TYPES: tuple[str, ...] = (
    "compare_sync_state",
    "compare_replay_state",
    "compare_proof_state",
    "hold_node",
    "recover_replay",
    "recover_proof",
    "rejoin_node",
    "emit_recovery_view",
)
CROSS_NODE_RECOVERY_SCHEMA_VERSION = "v139.3"


def _is_sha256_hex(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _validate_non_empty_str(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty str")


def _validate_bool(value: bool, field_name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be bool")


def _validate_fraction(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric in [0,1]")
    out = float(value)
    if not math.isfinite(out) or not 0.0 <= out <= 1.0:
        raise ValueError(f"{field_name} must be finite numeric in [0,1]")
    return out


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


@dataclass(frozen=True)
class RecoveryNodeInput:
    node_id: str
    node_role: str
    epoch_index: int
    state_hash: str
    replay_identity: str
    log_hash: str
    proof_bundle_hash: str
    sync_admissible: bool
    replay_admissible: bool
    proof_admissible: bool
    sync_confidence: float
    replay_confidence: float
    proof_confidence: float
    sync_risk: float
    replay_risk: float
    proof_risk: float

    def __post_init__(self) -> None:
        _validate_non_empty_str(self.node_id, "node_id")
        _validate_non_empty_str(self.node_role, "node_role")
        if isinstance(self.epoch_index, bool) or not isinstance(self.epoch_index, int) or self.epoch_index < 0:
            raise ValueError("epoch_index must be int >= 0")
        for hash_field in ("state_hash", "replay_identity", "log_hash", "proof_bundle_hash"):
            if not _is_sha256_hex(getattr(self, hash_field)):
                raise ValueError(f"{hash_field} must be 64-char lowercase sha256 hex")
        for field_name in ("sync_admissible", "replay_admissible", "proof_admissible"):
            _validate_bool(getattr(self, field_name), field_name)
        for field_name in (
            "sync_confidence",
            "replay_confidence",
            "proof_confidence",
            "sync_risk",
            "replay_risk",
            "proof_risk",
        ):
            object.__setattr__(self, field_name, _validate_fraction(getattr(self, field_name), field_name))

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "node_id": self.node_id,
            "node_role": self.node_role,
            "epoch_index": self.epoch_index,
            "state_hash": self.state_hash,
            "replay_identity": self.replay_identity,
            "log_hash": self.log_hash,
            "proof_bundle_hash": self.proof_bundle_hash,
            "sync_admissible": self.sync_admissible,
            "replay_admissible": self.replay_admissible,
            "proof_admissible": self.proof_admissible,
            "sync_confidence": self.sync_confidence,
            "replay_confidence": self.replay_confidence,
            "proof_confidence": self.proof_confidence,
            "sync_risk": self.sync_risk,
            "replay_risk": self.replay_risk,
            "proof_risk": self.proof_risk,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash_value(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class RecoveryPolicy:
    minimum_sync_confidence: float
    minimum_replay_confidence: float
    minimum_proof_confidence: float
    maximum_sync_risk: float
    maximum_replay_risk: float
    maximum_proof_risk: float
    require_sync_admissibility: bool
    require_replay_admissibility: bool
    require_proof_admissibility: bool
    require_epoch_alignment: bool
    allow_role_mixing: bool
    allow_partial_cluster_recovery: bool

    def __post_init__(self) -> None:
        for field_name in (
            "minimum_sync_confidence",
            "minimum_replay_confidence",
            "minimum_proof_confidence",
            "maximum_sync_risk",
            "maximum_replay_risk",
            "maximum_proof_risk",
        ):
            object.__setattr__(self, field_name, _validate_fraction(getattr(self, field_name), field_name))
        for field_name in (
            "require_sync_admissibility",
            "require_replay_admissibility",
            "require_proof_admissibility",
            "require_epoch_alignment",
            "allow_role_mixing",
            "allow_partial_cluster_recovery",
        ):
            _validate_bool(getattr(self, field_name), field_name)

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "minimum_sync_confidence": self.minimum_sync_confidence,
            "minimum_replay_confidence": self.minimum_replay_confidence,
            "minimum_proof_confidence": self.minimum_proof_confidence,
            "maximum_sync_risk": self.maximum_sync_risk,
            "maximum_replay_risk": self.maximum_replay_risk,
            "maximum_proof_risk": self.maximum_proof_risk,
            "require_sync_admissibility": self.require_sync_admissibility,
            "require_replay_admissibility": self.require_replay_admissibility,
            "require_proof_admissibility": self.require_proof_admissibility,
            "require_epoch_alignment": self.require_epoch_alignment,
            "allow_role_mixing": self.allow_role_mixing,
            "allow_partial_cluster_recovery": self.allow_partial_cluster_recovery,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash_value(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class RecoveryNodeStatus:
    node_id: str
    admissible: bool
    epoch_aligned: bool
    role_aligned: bool
    sync_ok: bool
    replay_ok: bool
    proof_ok: bool
    recoverable: bool
    requires_hold: bool
    requires_replay_recovery: bool
    requires_proof_recovery: bool
    requires_full_rejoin: bool
    recovery_confidence: float
    recovery_risk: float
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_non_empty_str(self.node_id, "node_id")
        for field_name in (
            "admissible",
            "epoch_aligned",
            "role_aligned",
            "sync_ok",
            "replay_ok",
            "proof_ok",
            "recoverable",
            "requires_hold",
            "requires_replay_recovery",
            "requires_proof_recovery",
            "requires_full_rejoin",
        ):
            _validate_bool(getattr(self, field_name), field_name)
        object.__setattr__(self, "recovery_confidence", _validate_fraction(self.recovery_confidence, "recovery_confidence"))
        object.__setattr__(self, "recovery_risk", _validate_fraction(self.recovery_risk, "recovery_risk"))
        if not isinstance(self.reasons, tuple) or any(not isinstance(item, str) or not item for item in self.reasons):
            raise ValueError("reasons must be tuple[str, ...] of non-empty strings")

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "node_id": self.node_id,
            "admissible": self.admissible,
            "epoch_aligned": self.epoch_aligned,
            "role_aligned": self.role_aligned,
            "sync_ok": self.sync_ok,
            "replay_ok": self.replay_ok,
            "proof_ok": self.proof_ok,
            "recoverable": self.recoverable,
            "requires_hold": self.requires_hold,
            "requires_replay_recovery": self.requires_replay_recovery,
            "requires_proof_recovery": self.requires_proof_recovery,
            "requires_full_rejoin": self.requires_full_rejoin,
            "recovery_confidence": self.recovery_confidence,
            "recovery_risk": self.recovery_risk,
            "reasons": self.reasons,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash_value(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class RecoveryAction:
    action_index: int
    action_type: str
    source_node_id: str
    target_node_id: str
    blocking: bool
    ready: bool
    detail: str

    def __post_init__(self) -> None:
        if isinstance(self.action_index, bool) or not isinstance(self.action_index, int) or self.action_index < 0:
            raise ValueError("action_index must be int >= 0")
        if self.action_type not in ALLOWED_RECOVERY_ACTION_TYPES:
            raise ValueError("invalid action_type")
        _validate_non_empty_str(self.source_node_id, "source_node_id")
        _validate_non_empty_str(self.target_node_id, "target_node_id")
        _validate_bool(self.blocking, "blocking")
        _validate_bool(self.ready, "ready")
        _validate_non_empty_str(self.detail, "detail")

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "action_index": self.action_index,
            "action_type": self.action_type,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "blocking": self.blocking,
            "ready": self.ready,
            "detail": self.detail,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash_value(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class CrossNodeRecoveryReceipt:
    node_inputs: tuple[RecoveryNodeInput, ...]
    policy_snapshot: RecoveryPolicy
    node_statuses: tuple[RecoveryNodeStatus, ...]
    recovery_actions: tuple[RecoveryAction, ...]
    cluster_epoch: int
    reference_node_id: str
    structurally_consistent: bool
    recovery_ready: bool
    recovery_confidence: float
    recovery_risk: float
    rationale: tuple[str, ...]
    schema_version: str
    replay_identity: str
    stable_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.node_inputs, tuple) or not self.node_inputs:
            raise ValueError("node_inputs must be non-empty tuple")
        if any(not isinstance(item, RecoveryNodeInput) for item in self.node_inputs):
            raise ValueError("node_inputs must contain RecoveryNodeInput")
        if not isinstance(self.policy_snapshot, RecoveryPolicy):
            raise ValueError("policy_snapshot must be RecoveryPolicy")
        if not isinstance(self.node_statuses, tuple) or any(not isinstance(item, RecoveryNodeStatus) for item in self.node_statuses):
            raise ValueError("node_statuses must be tuple[RecoveryNodeStatus, ...]")
        if not isinstance(self.recovery_actions, tuple) or any(not isinstance(item, RecoveryAction) for item in self.recovery_actions):
            raise ValueError("recovery_actions must be tuple[RecoveryAction, ...]")
        if isinstance(self.cluster_epoch, bool) or not isinstance(self.cluster_epoch, int) or self.cluster_epoch < 0:
            raise ValueError("cluster_epoch must be int >= 0")
        _validate_non_empty_str(self.reference_node_id, "reference_node_id")
        _validate_bool(self.structurally_consistent, "structurally_consistent")
        _validate_bool(self.recovery_ready, "recovery_ready")
        object.__setattr__(self, "recovery_confidence", _validate_fraction(self.recovery_confidence, "recovery_confidence"))
        object.__setattr__(self, "recovery_risk", _validate_fraction(self.recovery_risk, "recovery_risk"))
        if not isinstance(self.rationale, tuple) or any(not isinstance(item, str) or not item for item in self.rationale):
            raise ValueError("rationale must be tuple[str, ...] of non-empty strings")
        _validate_non_empty_str(self.schema_version, "schema_version")
        if not _is_sha256_hex(self.replay_identity):
            raise ValueError("replay_identity must be 64-char lowercase sha256 hex")
        expected_replay_identity = _compute_replay_identity(
            self.node_inputs,
            self.policy_snapshot,
            self.cluster_epoch,
            self.reference_node_id,
        )
        if self.replay_identity != expected_replay_identity:
            raise ValueError("replay_identity mismatch with receipt contents")
        if not _is_sha256_hex(self.stable_hash):
            raise ValueError("stable_hash must be 64-char lowercase sha256 hex")
        if self.stable_hash_value() != self.stable_hash:
            raise ValueError("stable_hash must match stable_hash_value")

    def _hash_payload(self) -> dict[str, _JSONValue]:
        return _build_receipt_hash_payload(
            node_inputs=self.node_inputs,
            policy_snapshot=self.policy_snapshot,
            node_statuses=self.node_statuses,
            recovery_actions=self.recovery_actions,
            cluster_epoch=self.cluster_epoch,
            reference_node_id=self.reference_node_id,
            structurally_consistent=self.structurally_consistent,
            recovery_ready=self.recovery_ready,
            recovery_confidence=self.recovery_confidence,
            recovery_risk=self.recovery_risk,
            rationale=self.rationale,
            schema_version=self.schema_version,
            replay_identity=self.replay_identity,
        )

    def to_dict(self) -> dict[str, _JSONValue]:
        payload = self._hash_payload()
        payload["stable_hash"] = self.stable_hash
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash_value(self) -> str:
        return sha256_hex(self._hash_payload())


def _build_replay_identity_payload(
    node_inputs: tuple[RecoveryNodeInput, ...],
    policy: RecoveryPolicy,
    cluster_epoch: int,
    reference_node_id: str,
) -> dict[str, _JSONValue]:
    return {
        "node_inputs": tuple(node.to_dict() for node in node_inputs),
        "policy_snapshot": policy.to_dict(),
        "cluster_epoch": cluster_epoch,
        "reference_node_id": reference_node_id,
        "schema_version": CROSS_NODE_RECOVERY_SCHEMA_VERSION,
    }


def _compute_replay_identity(
    node_inputs: tuple[RecoveryNodeInput, ...],
    policy: RecoveryPolicy,
    cluster_epoch: int,
    reference_node_id: str,
) -> str:
    return sha256_hex(_build_replay_identity_payload(node_inputs, policy, cluster_epoch, reference_node_id))


def _build_receipt_hash_payload(
    *,
    node_inputs: tuple[RecoveryNodeInput, ...],
    policy_snapshot: RecoveryPolicy,
    node_statuses: tuple[RecoveryNodeStatus, ...],
    recovery_actions: tuple[RecoveryAction, ...],
    cluster_epoch: int,
    reference_node_id: str,
    structurally_consistent: bool,
    recovery_ready: bool,
    recovery_confidence: float,
    recovery_risk: float,
    rationale: tuple[str, ...],
    schema_version: str,
    replay_identity: str,
) -> dict[str, _JSONValue]:
    return {
        "node_inputs": tuple(item.to_dict() for item in node_inputs),
        "policy_snapshot": policy_snapshot.to_dict(),
        "node_statuses": tuple(item.to_dict() for item in node_statuses),
        "recovery_actions": tuple(item.to_dict() for item in recovery_actions),
        "cluster_epoch": cluster_epoch,
        "reference_node_id": reference_node_id,
        "structurally_consistent": structurally_consistent,
        "recovery_ready": recovery_ready,
        "recovery_confidence": recovery_confidence,
        "recovery_risk": recovery_risk,
        "rationale": rationale,
        "schema_version": schema_version,
        "replay_identity": replay_identity,
    }


def _validate_inputs(node_inputs: tuple[RecoveryNodeInput, ...]) -> tuple[RecoveryNodeInput, ...]:
    if not isinstance(node_inputs, tuple) or not node_inputs:
        raise ValueError("node_inputs must be a non-empty tuple")
    if any(not isinstance(item, RecoveryNodeInput) for item in node_inputs):
        raise ValueError("node_inputs must contain RecoveryNodeInput values")
    ordered = tuple(sorted(node_inputs, key=lambda item: item.node_id))
    node_ids = tuple(node.node_id for node in ordered)
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("duplicate node_id")
    return ordered


def _cluster_epoch(node_inputs: tuple[RecoveryNodeInput, ...]) -> int:
    counts = Counter(node.epoch_index for node in node_inputs)
    return min((-count, epoch) for epoch, count in counts.items())[1]


def _passes_required_admissibility(node: RecoveryNodeInput, policy: RecoveryPolicy) -> bool:
    return (
        (not policy.require_sync_admissibility or node.sync_admissible)
        and (not policy.require_replay_admissibility or node.replay_admissible)
        and (not policy.require_proof_admissibility or node.proof_admissible)
    )


def _average_confidence(node: RecoveryNodeInput) -> float:
    return (node.sync_confidence + node.replay_confidence + node.proof_confidence) / 3.0


def _average_risk(node: RecoveryNodeInput) -> float:
    return (node.sync_risk + node.replay_risk + node.proof_risk) / 3.0


def _select_reference_node(node_inputs: tuple[RecoveryNodeInput, ...], policy: RecoveryPolicy) -> RecoveryNodeInput:
    gated = tuple(node for node in node_inputs if _passes_required_admissibility(node, policy))
    pool = gated if gated else node_inputs
    ranked = sorted(pool, key=lambda node: (-_average_confidence(node), _average_risk(node), node.node_id))
    return ranked[0]


def _node_status(
    node: RecoveryNodeInput,
    policy: RecoveryPolicy,
    cluster_epoch: int,
    reference_role: str,
) -> RecoveryNodeStatus:
    epoch_aligned = node.epoch_index == cluster_epoch
    role_aligned = policy.allow_role_mixing or node.node_role == reference_role

    sync_ok = (
        (node.sync_admissible or not policy.require_sync_admissibility)
        and node.sync_confidence >= policy.minimum_sync_confidence
        and node.sync_risk <= policy.maximum_sync_risk
    )
    replay_ok = (
        (node.replay_admissible or not policy.require_replay_admissibility)
        and node.replay_confidence >= policy.minimum_replay_confidence
        and node.replay_risk <= policy.maximum_replay_risk
    )
    proof_ok = (
        (node.proof_admissible or not policy.require_proof_admissibility)
        and node.proof_confidence >= policy.minimum_proof_confidence
        and node.proof_risk <= policy.maximum_proof_risk
    )

    admissible = _passes_required_admissibility(node, policy)
    layers_ok = sync_ok and replay_ok and proof_ok
    gating_block = (policy.require_epoch_alignment and not epoch_aligned) or (not role_aligned)
    recoverable = layers_ok and not gating_block

    requires_replay_recovery = sync_ok and proof_ok and (not replay_ok) and (not gating_block)
    requires_proof_recovery = sync_ok and replay_ok and (not proof_ok) and (not gating_block)
    requires_full_rejoin = (not sync_ok) or gating_block
    requires_hold = (not admissible) or ((not sync_ok) and (not replay_ok) and (not proof_ok))

    reasons: list[str] = []
    reasons.append("epoch aligned" if epoch_aligned else "epoch mismatch")
    reasons.append("role aligned" if role_aligned else "role mismatch")
    reasons.append("sync policy satisfied" if sync_ok else "sync policy failed")
    reasons.append("replay policy satisfied" if replay_ok else "replay policy failed")
    reasons.append("proof policy satisfied" if proof_ok else "proof policy failed")
    if requires_replay_recovery:
        reasons.append("replay recovery required")
    if requires_proof_recovery:
        reasons.append("proof recovery required")
    if requires_full_rejoin:
        reasons.append("full rejoin required")
    if requires_hold:
        reasons.append("hold required")
    if recoverable:
        reasons.append("recoverable")

    confidence = _clamp01((_average_confidence(node) + (1.0 if recoverable else 0.0)) / 2.0)
    risk = _clamp01((_average_risk(node) + (0.0 if recoverable else 1.0)) / 2.0)

    return RecoveryNodeStatus(
        node_id=node.node_id,
        admissible=admissible,
        epoch_aligned=epoch_aligned,
        role_aligned=role_aligned,
        sync_ok=sync_ok,
        replay_ok=replay_ok,
        proof_ok=proof_ok,
        recoverable=recoverable,
        requires_hold=requires_hold,
        requires_replay_recovery=requires_replay_recovery,
        requires_proof_recovery=requires_proof_recovery,
        requires_full_rejoin=requires_full_rejoin,
        recovery_confidence=confidence,
        recovery_risk=risk,
        reasons=tuple(reasons),
    )


def _evaluate_structural_consistency(
    statuses: tuple[RecoveryNodeStatus, ...],
    policy: RecoveryPolicy,
) -> bool:
    if not statuses:
        return False
    if policy.require_epoch_alignment and any(not status.epoch_aligned for status in statuses):
        return False
    if not policy.allow_role_mixing and any(not status.role_aligned for status in statuses):
        return False
    if not policy.allow_partial_cluster_recovery and any(not status.recoverable for status in statuses):
        return False
    return True


def _build_actions(
    statuses: tuple[RecoveryNodeStatus, ...],
    reference_node_id: str,
    recovery_ready: bool,
) -> tuple[RecoveryAction, ...]:
    actions: list[RecoveryAction] = []

    for action_type in ("compare_sync_state", "compare_replay_state", "compare_proof_state"):
        for status in statuses:
            if status.node_id == reference_node_id:
                continue
            actions.append(
                RecoveryAction(
                    action_index=0,
                    action_type=action_type,
                    source_node_id=reference_node_id,
                    target_node_id=status.node_id,
                    blocking=False,
                    ready=True,
                    detail=f"deterministic {action_type} against reference",
                )
            )

    for status in statuses:
        if status.requires_hold:
            actions.append(
                RecoveryAction(
                    action_index=0,
                    action_type="hold_node",
                    source_node_id=reference_node_id,
                    target_node_id=status.node_id,
                    blocking=True,
                    ready=True,
                    detail="node requires hold before rejoin",
                )
            )

    for status in statuses:
        if status.requires_replay_recovery:
            actions.append(
                RecoveryAction(
                    action_index=0,
                    action_type="recover_replay",
                    source_node_id=reference_node_id,
                    target_node_id=status.node_id,
                    blocking=True,
                    ready=True,
                    detail="node requires replay recovery",
                )
            )

    for status in statuses:
        if status.requires_proof_recovery:
            actions.append(
                RecoveryAction(
                    action_index=0,
                    action_type="recover_proof",
                    source_node_id=reference_node_id,
                    target_node_id=status.node_id,
                    blocking=True,
                    ready=True,
                    detail="node requires proof recovery",
                )
            )

    for status in statuses:
        if status.recoverable:
            actions.append(
                RecoveryAction(
                    action_index=0,
                    action_type="rejoin_node",
                    source_node_id=reference_node_id,
                    target_node_id=status.node_id,
                    blocking=False,
                    ready=True,
                    detail="node eligible for deterministic rejoin",
                )
            )

    actions.append(
        RecoveryAction(
            action_index=0,
            action_type="emit_recovery_view",
            source_node_id=reference_node_id,
            target_node_id=reference_node_id,
            blocking=not recovery_ready,
            ready=True,
            detail="emit deterministic advisory recovery receipt",
        )
    )

    return tuple(
        RecoveryAction(
            action_index=index,
            action_type=action.action_type,
            source_node_id=action.source_node_id,
            target_node_id=action.target_node_id,
            blocking=action.blocking,
            ready=action.ready,
            detail=action.detail,
        )
        for index, action in enumerate(actions)
    )


def _build_rationale(
    statuses: tuple[RecoveryNodeStatus, ...],
    policy: RecoveryPolicy,
    recovery_ready: bool,
) -> tuple[str, ...]:
    reasons: list[str] = ["reference recovery node selected deterministically"]
    reasons.append("sync admissibility satisfies policy" if any(s.sync_ok for s in statuses) else "sync admissibility below policy")
    if any(status.requires_replay_recovery for status in statuses):
        reasons.append("replay recovery required for node")
    if any(status.requires_proof_recovery for status in statuses):
        reasons.append("proof recovery required for node")
    if not policy.allow_role_mixing and any(not status.role_aligned for status in statuses):
        reasons.append("role mismatch disallowed by policy")
    if policy.allow_partial_cluster_recovery:
        reasons.append("partial cluster recovery allowed by policy")
    reasons.append("cross-node recovery ready" if recovery_ready else "cross-node recovery not ready")
    return tuple(reasons)


def run_cross_node_recovery_runtime(
    node_inputs: tuple[RecoveryNodeInput, ...],
    policy: RecoveryPolicy,
) -> CrossNodeRecoveryReceipt:
    normalized = _validate_inputs(node_inputs)
    if not isinstance(policy, RecoveryPolicy):
        raise ValueError("policy must be RecoveryPolicy")

    cluster_epoch = _cluster_epoch(normalized)
    reference = _select_reference_node(normalized, policy)

    statuses = tuple(
        _node_status(node, policy, cluster_epoch, reference.node_role)
        for node in normalized
    )
    structurally_consistent = _evaluate_structural_consistency(statuses, policy)
    reference_status = next(status for status in statuses if status.node_id == reference.node_id)

    if policy.allow_partial_cluster_recovery:
        plan_exists = any(s.recoverable or s.requires_replay_recovery or s.requires_proof_recovery for s in statuses)
        recovery_ready = structurally_consistent and reference_status.admissible and any(s.recoverable for s in statuses) and plan_exists
    else:
        recovery_ready = structurally_consistent and reference_status.admissible and all(s.recoverable for s in statuses)

    actions = _build_actions(statuses, reference.node_id, recovery_ready)
    rationale = _build_rationale(statuses, policy, recovery_ready)
    recovery_confidence = _clamp01(sum(status.recovery_confidence for status in statuses) / len(statuses))
    recovery_risk = _clamp01(sum(status.recovery_risk for status in statuses) / len(statuses))

    replay_identity = _compute_replay_identity(normalized, policy, cluster_epoch, reference.node_id)
    stable_hash = sha256_hex(
        _build_receipt_hash_payload(
            node_inputs=normalized,
            policy_snapshot=policy,
            node_statuses=statuses,
            recovery_actions=actions,
            cluster_epoch=cluster_epoch,
            reference_node_id=reference.node_id,
            structurally_consistent=structurally_consistent,
            recovery_ready=recovery_ready,
            recovery_confidence=recovery_confidence,
            recovery_risk=recovery_risk,
            rationale=rationale,
            schema_version=CROSS_NODE_RECOVERY_SCHEMA_VERSION,
            replay_identity=replay_identity,
        )
    )

    return CrossNodeRecoveryReceipt(
        node_inputs=normalized,
        policy_snapshot=policy,
        node_statuses=statuses,
        recovery_actions=actions,
        cluster_epoch=cluster_epoch,
        reference_node_id=reference.node_id,
        structurally_consistent=structurally_consistent,
        recovery_ready=recovery_ready,
        recovery_confidence=recovery_confidence,
        recovery_risk=recovery_risk,
        rationale=rationale,
        schema_version=CROSS_NODE_RECOVERY_SCHEMA_VERSION,
        replay_identity=replay_identity,
        stable_hash=stable_hash,
    )


def export_cross_node_recovery_runtime_bytes(receipt: CrossNodeRecoveryReceipt) -> bytes:
    if not isinstance(receipt, CrossNodeRecoveryReceipt):
        raise ValueError("receipt must be CrossNodeRecoveryReceipt")
    return receipt.to_canonical_bytes()


__all__ = [
    "ALLOWED_RECOVERY_ACTION_TYPES",
    "CROSS_NODE_RECOVERY_SCHEMA_VERSION",
    "RecoveryNodeInput",
    "RecoveryPolicy",
    "RecoveryNodeStatus",
    "RecoveryAction",
    "CrossNodeRecoveryReceipt",
    "run_cross_node_recovery_runtime",
    "export_cross_node_recovery_runtime_bytes",
]
