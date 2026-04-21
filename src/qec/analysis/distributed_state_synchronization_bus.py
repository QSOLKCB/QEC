from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

DISTRIBUTED_SYNC_BUS_SCHEMA_VERSION = "v139.0"
_ALLOWED_SYNC_ACTION_TYPES: tuple[str, ...] = (
    "compare_state",
    "compare_replay",
    "align_epoch",
    "align_hash",
    "hold_node",
    "admit_node",
    "emit_cluster_view",
)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return _canonical_json(payload).encode("utf-8")


def _sha256_hex(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _validate_sha256_hex(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{field_name} must be a 64-char SHA-256 hex")
    return normalized


def _validate_non_empty_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.strip()
    if normalized == "":
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _validate_non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an int")
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return value


def _validate_unit_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    if not (0.0 <= normalized <= 1.0):
        raise ValueError(f"{field_name} must be in [0,1]")
    return normalized


def _validate_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be bool")
    return value


def _normalize_metadata(metadata: Mapping[str, str] | None) -> Mapping[str, str] | None:
    if metadata is None:
        return None
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")
    normalized: dict[str, str] = {}
    for key in sorted(metadata.keys()):
        if not isinstance(key, str):
            raise ValueError("metadata keys must be strings")
        value = metadata[key]
        if not isinstance(value, str):
            raise ValueError("metadata values must be strings")
        normalized[key] = value
    return MappingProxyType(normalized)


@dataclass(frozen=True)
class NodeStateSnapshot:
    node_id: str
    node_role: str
    epoch_index: int
    state_hash: str
    replay_identity: str
    logical_stability: float
    projected_loss: float
    hardware_alignment: float
    execution_efficiency: float
    orchestration_depth: int
    metadata: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _validate_non_empty_str(self.node_id, field_name="node_id"))
        object.__setattr__(self, "node_role", _validate_non_empty_str(self.node_role, field_name="node_role"))
        object.__setattr__(self, "epoch_index", _validate_non_negative_int(self.epoch_index, field_name="epoch_index"))
        object.__setattr__(self, "state_hash", _validate_sha256_hex(self.state_hash, field_name="state_hash"))
        object.__setattr__(self, "replay_identity", _validate_sha256_hex(self.replay_identity, field_name="replay_identity"))
        object.__setattr__(
            self,
            "logical_stability",
            _validate_unit_float(self.logical_stability, field_name="logical_stability"),
        )
        object.__setattr__(self, "projected_loss", _validate_unit_float(self.projected_loss, field_name="projected_loss"))
        object.__setattr__(
            self,
            "hardware_alignment",
            _validate_unit_float(self.hardware_alignment, field_name="hardware_alignment"),
        )
        object.__setattr__(
            self,
            "execution_efficiency",
            _validate_unit_float(self.execution_efficiency, field_name="execution_efficiency"),
        )
        object.__setattr__(
            self,
            "orchestration_depth",
            _validate_non_negative_int(self.orchestration_depth, field_name="orchestration_depth"),
        )
        object.__setattr__(self, "metadata", _normalize_metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_role": self.node_role,
            "epoch_index": self.epoch_index,
            "state_hash": self.state_hash,
            "replay_identity": self.replay_identity,
            "logical_stability": self.logical_stability,
            "projected_loss": self.projected_loss,
            "hardware_alignment": self.hardware_alignment,
            "execution_efficiency": self.execution_efficiency,
            "orchestration_depth": self.orchestration_depth,
            "metadata": None if self.metadata is None else dict(self.metadata),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash_value(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class SyncBusPolicy:
    require_matching_epoch: bool
    require_matching_state_hash: bool
    maximum_projected_loss_delta: float
    minimum_logical_stability: float
    minimum_hardware_alignment: float
    allow_role_mixing: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "require_matching_epoch",
            _validate_bool(self.require_matching_epoch, field_name="require_matching_epoch"),
        )
        object.__setattr__(
            self,
            "require_matching_state_hash",
            _validate_bool(self.require_matching_state_hash, field_name="require_matching_state_hash"),
        )
        object.__setattr__(
            self,
            "allow_role_mixing",
            _validate_bool(self.allow_role_mixing, field_name="allow_role_mixing"),
        )
        object.__setattr__(
            self,
            "maximum_projected_loss_delta",
            _validate_unit_float(self.maximum_projected_loss_delta, field_name="maximum_projected_loss_delta"),
        )
        object.__setattr__(
            self,
            "minimum_logical_stability",
            _validate_unit_float(self.minimum_logical_stability, field_name="minimum_logical_stability"),
        )
        object.__setattr__(
            self,
            "minimum_hardware_alignment",
            _validate_unit_float(self.minimum_hardware_alignment, field_name="minimum_hardware_alignment"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "require_matching_epoch": self.require_matching_epoch,
            "require_matching_state_hash": self.require_matching_state_hash,
            "maximum_projected_loss_delta": self.maximum_projected_loss_delta,
            "minimum_logical_stability": self.minimum_logical_stability,
            "minimum_hardware_alignment": self.minimum_hardware_alignment,
            "allow_role_mixing": self.allow_role_mixing,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash_value(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class NodeSyncStatus:
    node_id: str
    admissible: bool
    epoch_aligned: bool
    state_hash_aligned: bool
    replay_identity_aligned: bool
    stability_ok: bool
    hardware_ok: bool
    loss_delta_ok: bool
    sync_confidence: float
    sync_risk: float
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _validate_non_empty_str(self.node_id, field_name="node_id"))
        for field in (
            "admissible",
            "epoch_aligned",
            "state_hash_aligned",
            "replay_identity_aligned",
            "stability_ok",
            "hardware_ok",
            "loss_delta_ok",
        ):
            object.__setattr__(self, field, _validate_bool(getattr(self, field), field_name=field))
        object.__setattr__(self, "sync_confidence", _validate_unit_float(self.sync_confidence, field_name="sync_confidence"))
        object.__setattr__(self, "sync_risk", _validate_unit_float(self.sync_risk, field_name="sync_risk"))
        if not isinstance(self.reasons, tuple) or any(not isinstance(r, str) or r.strip() == "" for r in self.reasons):
            raise ValueError("reasons must be tuple[str, ...] with non-empty entries")

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "admissible": self.admissible,
            "epoch_aligned": self.epoch_aligned,
            "state_hash_aligned": self.state_hash_aligned,
            "replay_identity_aligned": self.replay_identity_aligned,
            "stability_ok": self.stability_ok,
            "hardware_ok": self.hardware_ok,
            "loss_delta_ok": self.loss_delta_ok,
            "sync_confidence": self.sync_confidence,
            "sync_risk": self.sync_risk,
            "reasons": self.reasons,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash_value(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class SyncAction:
    action_index: int
    action_type: str
    source_node_id: str
    target_node_id: str
    blocking: bool
    ready: bool
    detail: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_index", _validate_non_negative_int(self.action_index, field_name="action_index"))
        if self.action_type not in _ALLOWED_SYNC_ACTION_TYPES:
            raise ValueError("invalid action type")
        object.__setattr__(self, "source_node_id", _validate_non_empty_str(self.source_node_id, field_name="source_node_id"))
        object.__setattr__(self, "target_node_id", _validate_non_empty_str(self.target_node_id, field_name="target_node_id"))
        object.__setattr__(self, "blocking", _validate_bool(self.blocking, field_name="blocking"))
        object.__setattr__(self, "ready", _validate_bool(self.ready, field_name="ready"))
        object.__setattr__(self, "detail", _validate_non_empty_str(self.detail, field_name="detail"))

    def to_dict(self) -> dict[str, Any]:
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
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash_value(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class DistributedSyncReceipt:
    node_snapshots: tuple[NodeStateSnapshot, ...]
    policy_snapshot: SyncBusPolicy
    node_statuses: tuple[NodeSyncStatus, ...]
    sync_actions: tuple[SyncAction, ...]
    cluster_epoch: int
    cluster_state_hash: str
    structurally_consistent: bool
    cluster_ready: bool
    sync_confidence: float
    sync_risk: float
    rationale: tuple[str, ...]
    schema_version: str
    replay_identity: str
    stable_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.node_snapshots, tuple) or len(self.node_snapshots) == 0:
            raise ValueError("node_snapshots must be a non-empty tuple")
        if not isinstance(self.policy_snapshot, SyncBusPolicy):
            raise ValueError("policy_snapshot must be SyncBusPolicy")
        if not isinstance(self.node_statuses, tuple):
            raise ValueError("node_statuses must be tuple")
        if not isinstance(self.sync_actions, tuple):
            raise ValueError("sync_actions must be tuple")
        object.__setattr__(self, "cluster_epoch", _validate_non_negative_int(self.cluster_epoch, field_name="cluster_epoch"))
        object.__setattr__(self, "cluster_state_hash", _validate_sha256_hex(self.cluster_state_hash, field_name="cluster_state_hash"))
        object.__setattr__(
            self,
            "structurally_consistent",
            _validate_bool(self.structurally_consistent, field_name="structurally_consistent"),
        )
        object.__setattr__(self, "cluster_ready", _validate_bool(self.cluster_ready, field_name="cluster_ready"))
        object.__setattr__(self, "sync_confidence", _validate_unit_float(self.sync_confidence, field_name="sync_confidence"))
        object.__setattr__(self, "sync_risk", _validate_unit_float(self.sync_risk, field_name="sync_risk"))
        if not isinstance(self.rationale, tuple) or any(not isinstance(r, str) or r.strip() == "" for r in self.rationale):
            raise ValueError("rationale must be tuple[str, ...] with non-empty entries")
        object.__setattr__(self, "schema_version", _validate_non_empty_str(self.schema_version, field_name="schema_version"))
        object.__setattr__(self, "replay_identity", _validate_sha256_hex(self.replay_identity, field_name="replay_identity"))
        object.__setattr__(self, "stable_hash", _validate_sha256_hex(self.stable_hash, field_name="stable_hash"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_snapshots": [node.to_dict() for node in self.node_snapshots],
            "policy_snapshot": self.policy_snapshot.to_dict(),
            "node_statuses": [status.to_dict() for status in self.node_statuses],
            "sync_actions": [action.to_dict() for action in self.sync_actions],
            "cluster_epoch": self.cluster_epoch,
            "cluster_state_hash": self.cluster_state_hash,
            "structurally_consistent": self.structurally_consistent,
            "cluster_ready": self.cluster_ready,
            "sync_confidence": self.sync_confidence,
            "sync_risk": self.sync_risk,
            "rationale": self.rationale,
            "schema_version": self.schema_version,
            "replay_identity": self.replay_identity,
            "stable_hash": self.stable_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash_value(self) -> str:
        return _sha256_hex(_receipt_hash_payload(self))


def _canonicalize_snapshots(node_snapshots: Sequence[NodeStateSnapshot]) -> tuple[NodeStateSnapshot, ...]:
    canonical = tuple(node_snapshots)
    if len(canonical) == 0:
        raise ValueError("node_snapshots must be non-empty")

    for index, node in enumerate(canonical):
        if not isinstance(node, NodeStateSnapshot):
            raise TypeError(
                f"node_snapshots[{index}] must be a NodeStateSnapshot, got {type(node).__name__}"
            )
    ids = [node.node_id for node in canonical]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate node_id")
    return tuple(sorted(canonical, key=lambda n: n.node_id))


def _cluster_epoch(node_snapshots: tuple[NodeStateSnapshot, ...]) -> int:
    counts = Counter(snapshot.epoch_index for snapshot in node_snapshots)
    return min((-count, epoch) for epoch, count in counts.items())[1]


def _cluster_role(node_snapshots: tuple[NodeStateSnapshot, ...]) -> str:
    counts = Counter(snapshot.node_role for snapshot in node_snapshots)
    return min((-count, role) for role, count in counts.items())[1]


def _select_reference_node(
    node_snapshots: tuple[NodeStateSnapshot, ...],
    policy: SyncBusPolicy,
    cluster_role: str,
    cluster_epoch: int,
) -> NodeStateSnapshot:
    if policy.allow_role_mixing:
        eligible = node_snapshots
    else:
        eligible = tuple(node for node in node_snapshots if node.node_role == cluster_role)
        if not eligible:
            eligible = node_snapshots

    admissible_seed = tuple(
        node
        for node in eligible
        if node.logical_stability >= policy.minimum_logical_stability
        and node.hardware_alignment >= policy.minimum_hardware_alignment
    )
    pool = admissible_seed if admissible_seed else eligible

    epoch_aligned_nodes = tuple(node for node in pool if node.epoch_index == cluster_epoch)
    if policy.require_matching_epoch and epoch_aligned_nodes:
        pool = epoch_aligned_nodes
    return sorted(
        pool,
        key=lambda node: (
            -node.logical_stability,
            -node.hardware_alignment,
            node.projected_loss,
            node.node_id,
        ),
    )[0]


def _build_node_statuses(
    node_snapshots: tuple[NodeStateSnapshot, ...],
    reference_node: NodeStateSnapshot,
    policy: SyncBusPolicy,
    cluster_epoch: int,
    cluster_state_hash: str,
    cluster_role: str,
) -> tuple[NodeSyncStatus, ...]:
    statuses: list[NodeSyncStatus] = []
    for node in node_snapshots:
        epoch_aligned = node.epoch_index == cluster_epoch
        state_hash_aligned = node.state_hash == cluster_state_hash
        replay_identity_aligned = node.replay_identity == reference_node.replay_identity
        stability_ok = node.logical_stability >= policy.minimum_logical_stability
        hardware_ok = node.hardware_alignment >= policy.minimum_hardware_alignment
        loss_delta_ok = abs(node.projected_loss - reference_node.projected_loss) <= policy.maximum_projected_loss_delta
        role_ok = policy.allow_role_mixing or node.node_role == cluster_role

        reasons: list[str] = []
        if not stability_ok:
            reasons.append("stability threshold unmet")
        if not hardware_ok:
            reasons.append("hardware alignment threshold unmet")
        if not loss_delta_ok:
            reasons.append("projected loss delta exceeds policy")
        if not role_ok:
            reasons.append("role mixing disabled")
        if policy.require_matching_epoch and not epoch_aligned:
            reasons.append("epoch mismatch")
        if policy.require_matching_state_hash and not state_hash_aligned:
            reasons.append("state hash mismatch")

        admissible = stability_ok and hardware_ok and loss_delta_ok and role_ok
        if policy.require_matching_epoch:
            admissible = admissible and epoch_aligned
        if policy.require_matching_state_hash:
            admissible = admissible and state_hash_aligned

        confidence = float((
            (1.0 if epoch_aligned else 0.0)
            + (1.0 if state_hash_aligned else 0.0)
            + (1.0 if replay_identity_aligned else 0.0)
            + (1.0 if stability_ok else 0.0)
            + (1.0 if hardware_ok else 0.0)
            + (1.0 if loss_delta_ok else 0.0)
        ) / 6.0)
        risk = float(1.0 - confidence)

        statuses.append(
            NodeSyncStatus(
                node_id=node.node_id,
                admissible=admissible,
                epoch_aligned=epoch_aligned,
                state_hash_aligned=state_hash_aligned,
                replay_identity_aligned=replay_identity_aligned,
                stability_ok=stability_ok,
                hardware_ok=hardware_ok,
                loss_delta_ok=loss_delta_ok,
                sync_confidence=confidence,
                sync_risk=risk,
                reasons=tuple(reasons),
            )
        )
    return tuple(statuses)


def _build_sync_actions(
    node_snapshots: tuple[NodeStateSnapshot, ...],
    node_statuses: tuple[NodeSyncStatus, ...],
    reference_node: NodeStateSnapshot,
    policy: SyncBusPolicy,
    cluster_ready: bool,
) -> tuple[SyncAction, ...]:
    actions: list[SyncAction] = []
    status_by_node = {status.node_id: status for status in node_statuses}

    def _append_action(**kwargs: Any) -> None:
        actions.append(SyncAction(action_index=len(actions), **kwargs))

    for node in node_snapshots:
        if node.node_id == reference_node.node_id:
            continue
        status = status_by_node[node.node_id]
        _append_action(
            action_type="compare_state",
            source_node_id=reference_node.node_id,
            target_node_id=node.node_id,
            blocking=False,
            ready=True,
            detail="compare node state hash against reference",
        )
        _append_action(
            action_type="compare_replay",
            source_node_id=reference_node.node_id,
            target_node_id=node.node_id,
            blocking=False,
            ready=True,
            detail="compare replay identity against reference",
        )
        epoch_alignment_required = (
            not status.epoch_aligned and policy.require_matching_epoch
        )
        _append_action(
            action_type="align_epoch",
            source_node_id=node.node_id,
            target_node_id=reference_node.node_id,
            blocking=epoch_alignment_required,
            ready=epoch_alignment_required,
            detail=(
                "align node epoch to cluster epoch"
                if policy.require_matching_epoch
                else "epoch differs from cluster but policy permits mismatch"
            ),
        )
        hash_alignment_required = (
            not status.state_hash_aligned and policy.require_matching_state_hash
        )
        _append_action(
            action_type="align_hash",
            source_node_id=node.node_id,
            target_node_id=reference_node.node_id,
            blocking=hash_alignment_required,
            ready=hash_alignment_required,
            detail=(
                "align node state hash to reference hash"
                if policy.require_matching_state_hash
                else "state hash differs from reference, but policy permits mismatch"
            ),
        )

    for status in node_statuses:
        actions.append(
            SyncAction(
                action_index=len(actions),
                action_type="admit_node" if status.admissible else "hold_node",
                source_node_id=status.node_id,
                target_node_id=reference_node.node_id,
                blocking=not status.admissible,
                ready=status.admissible,
                detail="node admissible for synchronization" if status.admissible else "node held by policy",
            )
        )

    actions.append(
        SyncAction(
            action_index=len(actions),
            action_type="emit_cluster_view",
            source_node_id=reference_node.node_id,
            target_node_id=reference_node.node_id,
            blocking=not cluster_ready,
            ready=True,
            detail="emit deterministic advisory cluster synchronization view",
        )
    )
    return tuple(actions)


def _build_rationale(
    node_statuses: tuple[NodeSyncStatus, ...],
    cluster_ready: bool,
    structural_consistency: bool,
    require_matching_epoch: bool = True,
    require_matching_state_hash: bool = True,
) -> tuple[str, ...]:
    """Build human-readable readiness rationale aligned with policy constraints."""
    out: list[str] = ["reference node selected deterministically"]

    if all(status.epoch_aligned for status in node_statuses):
        out.append("epoch alignment satisfied")
    elif require_matching_epoch:
        out.append("epoch mismatch blocks cluster readiness")
    else:
        out.append("epoch mismatch noted but allowed by policy")

    if all(status.state_hash_aligned for status in node_statuses):
        out.append("state hash alignment satisfied")
    elif require_matching_state_hash:
        out.append("state hash mismatch blocks cluster readiness")
    else:
        out.append("state hash mismatch noted but allowed by policy")

    if all(status.loss_delta_ok for status in node_statuses):
        out.append("projected loss delta within policy bounds")
    else:
        out.append("projected loss delta exceeds policy bounds")

    held_nodes = tuple(sorted(status.node_id for status in node_statuses if not status.admissible))
    for node_id in held_nodes:
        out.append(f"node held due to policy constraints: {node_id}")

    if structural_consistency:
        out.append("structural consistency satisfied")
    else:
        out.append("structural consistency failed")

    out.append("cluster synchronization ready" if cluster_ready else "cluster synchronization not ready")
    return tuple(out)


def _receipt_replay_payload(
    *,
    node_snapshots: tuple[NodeStateSnapshot, ...],
    policy: SyncBusPolicy,
    node_statuses: tuple[NodeSyncStatus, ...],
    sync_actions: tuple[SyncAction, ...],
    cluster_epoch: int,
    cluster_state_hash: str,
    structurally_consistent: bool,
    cluster_ready: bool,
    sync_confidence: float,
    sync_risk: float,
    rationale: tuple[str, ...],
    schema_version: str,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "node_snapshots": [node.to_dict() for node in node_snapshots],
        "policy_snapshot": policy.to_dict(),
        "node_statuses": [status.to_dict() for status in node_statuses],
        "sync_actions": [action.to_dict() for action in sync_actions],
        "cluster_epoch": cluster_epoch,
        "cluster_state_hash": cluster_state_hash,
        "structurally_consistent": structurally_consistent,
        "cluster_ready": cluster_ready,
        "sync_confidence": sync_confidence,
        "sync_risk": sync_risk,
        "rationale": rationale,
    }


def _receipt_hash_payload(receipt: DistributedSyncReceipt) -> dict[str, Any]:
    payload = receipt.to_dict()
    payload.pop("stable_hash")
    return payload


def build_distributed_sync_receipt(
    node_snapshots: Sequence[NodeStateSnapshot],
    policy: SyncBusPolicy,
    *,
    schema_version: str = DISTRIBUTED_SYNC_BUS_SCHEMA_VERSION,
) -> DistributedSyncReceipt:
    if not isinstance(policy, SyncBusPolicy):
        raise ValueError("policy must be SyncBusPolicy")
    schema = _validate_non_empty_str(schema_version, field_name="schema_version")

    canonical_nodes = _canonicalize_snapshots(node_snapshots)
    cluster_epoch = _cluster_epoch(canonical_nodes)
    cluster_role = _cluster_role(canonical_nodes)
    reference_node = _select_reference_node(canonical_nodes, policy, cluster_role, cluster_epoch)
    cluster_state_hash = reference_node.state_hash

    node_statuses = _build_node_statuses(canonical_nodes, reference_node, policy, cluster_epoch, cluster_state_hash, cluster_role)
    admissible = tuple(status for status in node_statuses if status.admissible)

    structural_consistency = all(
        status.replay_identity_aligned
        and (status.epoch_aligned or not policy.require_matching_epoch)
        and (status.state_hash_aligned or not policy.require_matching_state_hash)
        for status in node_statuses
    )

    required_constraints_ok = True
    if policy.require_matching_epoch:
        required_constraints_ok = required_constraints_ok and all(status.epoch_aligned for status in admissible)
    if policy.require_matching_state_hash:
        required_constraints_ok = required_constraints_ok and all(status.state_hash_aligned for status in admissible)

    cluster_ready = bool(structural_consistency and len(admissible) > 0 and required_constraints_ok)

    sync_confidence = float(sum(status.sync_confidence for status in node_statuses) / len(node_statuses))
    sync_risk = float(1.0 - sync_confidence)

    sync_actions = _build_sync_actions(canonical_nodes, node_statuses, reference_node, policy, cluster_ready)
    rationale = _build_rationale(
        node_statuses,
        cluster_ready,
        structural_consistency,
        require_matching_epoch=policy.require_matching_epoch,
        require_matching_state_hash=policy.require_matching_state_hash,
    )

    replay_payload = _receipt_replay_payload(
        node_snapshots=canonical_nodes,
        policy=policy,
        node_statuses=node_statuses,
        sync_actions=sync_actions,
        cluster_epoch=cluster_epoch,
        cluster_state_hash=cluster_state_hash,
        structurally_consistent=structural_consistency,
        cluster_ready=cluster_ready,
        sync_confidence=sync_confidence,
        sync_risk=sync_risk,
        rationale=rationale,
        schema_version=schema,
    )
    replay_identity = _sha256_hex(replay_payload)

    preliminary = DistributedSyncReceipt(
        node_snapshots=canonical_nodes,
        policy_snapshot=policy,
        node_statuses=node_statuses,
        sync_actions=sync_actions,
        cluster_epoch=cluster_epoch,
        cluster_state_hash=cluster_state_hash,
        structurally_consistent=structural_consistency,
        cluster_ready=cluster_ready,
        sync_confidence=sync_confidence,
        sync_risk=sync_risk,
        rationale=rationale,
        schema_version=schema,
        replay_identity=replay_identity,
        stable_hash="0" * 64,
    )
    stable_hash = _sha256_hex(_receipt_hash_payload(preliminary))

    return DistributedSyncReceipt(
        node_snapshots=canonical_nodes,
        policy_snapshot=policy,
        node_statuses=node_statuses,
        sync_actions=sync_actions,
        cluster_epoch=cluster_epoch,
        cluster_state_hash=cluster_state_hash,
        structurally_consistent=structural_consistency,
        cluster_ready=cluster_ready,
        sync_confidence=sync_confidence,
        sync_risk=sync_risk,
        rationale=rationale,
        schema_version=schema,
        replay_identity=replay_identity,
        stable_hash=stable_hash,
    )
