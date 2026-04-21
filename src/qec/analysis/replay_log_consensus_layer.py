"""v139.1 — Replay Log Consensus Layer.

Deterministic advisory consensus analysis for node-local replay logs.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
import hashlib
import math
import re
from typing import Any, Mapping

from .canonical_hashing import canonical_bytes, canonical_json

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

SCHEMA_VERSION = "v139.1"
ALLOWED_CONSENSUS_ACTION_TYPES: tuple[str, ...] = (
    "compare_prefix",
    "compare_log_hash",
    "align_epoch",
    "hold_node",
    "admit_log",
    "flag_divergence",
    "emit_consensus_view",
)

_SHA256_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _is_valid_sha256_hex(value: str) -> bool:
    return bool(_SHA256_HEX_RE.fullmatch(value))


def _require_non_empty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_int_non_bool(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _require_non_negative_int(value: Any, field_name: str) -> int:
    as_int = _require_int_non_bool(value, field_name)
    if as_int < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return as_int


def _require_float01(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    as_float = float(value)
    if not math.isfinite(as_float) or not (0.0 <= as_float <= 1.0):
        raise ValueError(f"{field_name} must be finite in [0,1]")
    return as_float


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _sha256_hex(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class ReplayLogEntry:
    sequence_index: int
    event_type: str
    event_hash: str
    replay_identity: str
    payload_hash: str
    deterministic_order_key: str

    def __post_init__(self) -> None:
        if _require_non_negative_int(self.sequence_index, "sequence_index") < 0:
            raise ValueError("sequence_index must be >= 0")
        _require_non_empty_str(self.event_type, "event_type")
        _require_non_empty_str(self.deterministic_order_key, "deterministic_order_key")
        for field_name in ("event_hash", "replay_identity", "payload_hash"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not _is_valid_sha256_hex(value):
                raise ValueError(f"{field_name} must be a valid SHA-256 hex string")

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "sequence_index": self.sequence_index,
            "event_type": self.event_type,
            "event_hash": self.event_hash,
            "replay_identity": self.replay_identity,
            "payload_hash": self.payload_hash,
            "deterministic_order_key": self.deterministic_order_key,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash_value(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class NodeReplayLog:
    node_id: str
    node_role: str
    epoch_index: int
    entries: tuple[ReplayLogEntry, ...]
    log_hash: str
    metadata: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        _require_non_empty_str(self.node_id, "node_id")
        _require_non_empty_str(self.node_role, "node_role")
        _require_non_negative_int(self.epoch_index, "epoch_index")
        if not isinstance(self.entries, tuple) or len(self.entries) == 0:
            raise ValueError("entries must be a non-empty tuple")
        if any(not isinstance(entry, ReplayLogEntry) for entry in self.entries):
            raise ValueError("entries must contain ReplayLogEntry values")

        sequence_indices = tuple(entry.sequence_index for entry in self.entries)
        if len(set(sequence_indices)) != len(sequence_indices):
            raise ValueError("duplicate sequence_index within node log")
        if sequence_indices != tuple(sorted(sequence_indices)):
            raise ValueError("entries must be ordered by sequence_index ascending")

        if not isinstance(self.log_hash, str) or not _is_valid_sha256_hex(self.log_hash):
            raise ValueError("log_hash must be a valid SHA-256 hex string")

        if self.metadata is None:
            object.__setattr__(self, "metadata", None)
        else:
            if not isinstance(self.metadata, Mapping):
                raise ValueError("metadata must be a mapping[str, str] or None")
            if any(not isinstance(k, str) or not isinstance(v, str) for k, v in self.metadata.items()):
                raise ValueError("metadata must only contain string keys and values")
            normalized = {k: v for k, v in self.metadata.items()}
            object.__setattr__(self, "metadata", MappingProxyType({k: normalized[k] for k in sorted(normalized)}))

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "node_id": self.node_id,
            "node_role": self.node_role,
            "epoch_index": self.epoch_index,
            "entries": tuple(entry.to_dict() for entry in self.entries),
            "log_hash": self.log_hash,
            "metadata": None if self.metadata is None else dict(self.metadata),
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash_value(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class ReplayConsensusPolicy:
    require_matching_epoch: bool
    require_full_prefix_agreement: bool
    allow_length_skew: bool
    maximum_length_delta: int
    require_matching_log_hash: bool
    minimum_consensus_fraction: float
    allow_role_mixing: bool

    def __post_init__(self) -> None:
        for field_name in (
            "require_matching_epoch",
            "require_full_prefix_agreement",
            "allow_length_skew",
            "require_matching_log_hash",
            "allow_role_mixing",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be bool")
        _require_non_negative_int(self.maximum_length_delta, "maximum_length_delta")
        _require_float01(self.minimum_consensus_fraction, "minimum_consensus_fraction")

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "require_matching_epoch": self.require_matching_epoch,
            "require_full_prefix_agreement": self.require_full_prefix_agreement,
            "allow_length_skew": self.allow_length_skew,
            "maximum_length_delta": self.maximum_length_delta,
            "require_matching_log_hash": self.require_matching_log_hash,
            "minimum_consensus_fraction": float(self.minimum_consensus_fraction),
            "allow_role_mixing": self.allow_role_mixing,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash_value(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class NodeReplayConsensusStatus:
    node_id: str
    admissible: bool
    epoch_aligned: bool
    log_hash_aligned: bool
    prefix_aligned: bool
    length_delta_ok: bool
    consensus_fraction_ok: bool
    matching_prefix_length: int
    local_log_length: int
    consensus_fraction: float
    consensus_risk: float
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_non_empty_str(self.node_id, "node_id")
        _require_non_negative_int(self.matching_prefix_length, "matching_prefix_length")
        _require_non_negative_int(self.local_log_length, "local_log_length")
        _require_float01(self.consensus_fraction, "consensus_fraction")
        _require_float01(self.consensus_risk, "consensus_risk")
        if not isinstance(self.reasons, tuple) or any(not isinstance(reason, str) for reason in self.reasons):
            raise ValueError("reasons must be tuple[str, ...]")

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "node_id": self.node_id,
            "admissible": self.admissible,
            "epoch_aligned": self.epoch_aligned,
            "log_hash_aligned": self.log_hash_aligned,
            "prefix_aligned": self.prefix_aligned,
            "length_delta_ok": self.length_delta_ok,
            "consensus_fraction_ok": self.consensus_fraction_ok,
            "matching_prefix_length": self.matching_prefix_length,
            "local_log_length": self.local_log_length,
            "consensus_fraction": float(self.consensus_fraction),
            "consensus_risk": float(self.consensus_risk),
            "reasons": self.reasons,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash_value(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class ReplayConsensusAction:
    action_index: int
    action_type: str
    source_node_id: str
    target_node_id: str
    blocking: bool
    ready: bool
    detail: str

    def __post_init__(self) -> None:
        _require_non_negative_int(self.action_index, "action_index")
        if self.action_type not in ALLOWED_CONSENSUS_ACTION_TYPES:
            raise ValueError(f"unsupported action_type: {self.action_type}")
        _require_non_empty_str(self.source_node_id, "source_node_id")
        _require_non_empty_str(self.target_node_id, "target_node_id")
        _require_non_empty_str(self.detail, "detail")

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
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class ReplayLogConsensusReceipt:
    node_replay_logs: tuple[NodeReplayLog, ...]
    policy_snapshot: ReplayConsensusPolicy
    node_statuses: tuple[NodeReplayConsensusStatus, ...]
    consensus_actions: tuple[ReplayConsensusAction, ...]
    cluster_epoch: int
    reference_node_id: str
    reference_log_hash: str
    structurally_consistent: bool
    consensus_ready: bool
    consensus_confidence: float
    consensus_risk: float
    rationale: tuple[str, ...]
    schema_version: str
    replay_identity: str
    stable_hash: str

    def __post_init__(self) -> None:
        _require_non_negative_int(self.cluster_epoch, "cluster_epoch")
        _require_non_empty_str(self.reference_node_id, "reference_node_id")
        if not _is_valid_sha256_hex(self.reference_log_hash):
            raise ValueError("reference_log_hash must be SHA-256 hex")
        _require_float01(self.consensus_confidence, "consensus_confidence")
        _require_float01(self.consensus_risk, "consensus_risk")
        _require_non_empty_str(self.schema_version, "schema_version")
        for field_name in ("replay_identity", "stable_hash"):
            value = getattr(self, field_name)
            if not _is_valid_sha256_hex(value):
                raise ValueError(f"{field_name} must be SHA-256 hex")

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "node_replay_logs": tuple(log.to_dict() for log in self.node_replay_logs),
            "policy_snapshot": self.policy_snapshot.to_dict(),
            "node_statuses": tuple(status.to_dict() for status in self.node_statuses),
            "consensus_actions": tuple(action.to_dict() for action in self.consensus_actions),
            "cluster_epoch": self.cluster_epoch,
            "reference_node_id": self.reference_node_id,
            "reference_log_hash": self.reference_log_hash,
            "structurally_consistent": self.structurally_consistent,
            "consensus_ready": self.consensus_ready,
            "consensus_confidence": float(self.consensus_confidence),
            "consensus_risk": float(self.consensus_risk),
            "rationale": self.rationale,
            "schema_version": self.schema_version,
            "replay_identity": self.replay_identity,
            "stable_hash": self.stable_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("stable_hash")
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash_value(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


def _matching_prefix_length(reference_entries: tuple[ReplayLogEntry, ...], candidate_entries: tuple[ReplayLogEntry, ...]) -> int:
    reference_hashes = tuple(entry.stable_hash_value() for entry in reference_entries)
    candidate_hashes = tuple(entry.stable_hash_value() for entry in candidate_entries)

    matched = 0
    for reference_hash, candidate_hash in zip(reference_hashes, candidate_hashes):
        if reference_hash != candidate_hash:
            break
        matched += 1
    return matched


def _cluster_epoch(logs: tuple[NodeReplayLog, ...]) -> int:
    counter = Counter(log.epoch_index for log in logs)
    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return int(ranked[0][0])


def _select_reference_log(logs: tuple[NodeReplayLog, ...], policy: ReplayConsensusPolicy, cluster_epoch: int) -> NodeReplayLog:
    candidates = logs
    if policy.require_matching_epoch:
        aligned = tuple(log for log in logs if log.epoch_index == cluster_epoch)
        if len(aligned) > 0:
            candidates = aligned
    ordered = sorted(candidates, key=lambda log: (-len(log.entries), log.log_hash, log.node_id))
    return ordered[0]


def _compute_node_status(
    node_log: NodeReplayLog,
    reference_log: NodeReplayLog,
    cluster_epoch: int,
    policy: ReplayConsensusPolicy,
) -> NodeReplayConsensusStatus:
    matching_prefix_length = _matching_prefix_length(reference_log.entries, node_log.entries)
    local_log_length = len(node_log.entries)
    reference_length = len(reference_log.entries)
    epoch_aligned = node_log.epoch_index == cluster_epoch
    log_hash_aligned = node_log.log_hash == reference_log.log_hash
    prefix_aligned = matching_prefix_length == min(reference_length, local_log_length)

    if policy.allow_length_skew:
        length_delta_ok = abs(reference_length - local_log_length) <= policy.maximum_length_delta
    else:
        length_delta_ok = reference_length == local_log_length

    consensus_fraction = float(matching_prefix_length) / float(max(reference_length, 1))
    consensus_fraction_ok = consensus_fraction >= policy.minimum_consensus_fraction

    admissible = True
    reasons: list[str] = []

    if policy.require_matching_epoch and not epoch_aligned:
        admissible = False
        reasons.append("epoch mismatch")
    else:
        reasons.append("epoch aligned" if epoch_aligned else "epoch mismatch tolerated")

    if policy.require_matching_log_hash and not log_hash_aligned:
        admissible = False
        reasons.append("log hash mismatch")
    else:
        reasons.append("log hash aligned" if log_hash_aligned else "log hash mismatch tolerated")

    if policy.require_full_prefix_agreement and not prefix_aligned:
        admissible = False
        reasons.append("prefix mismatch")
    else:
        reasons.append("prefix aligned" if prefix_aligned else "prefix mismatch tolerated")

    if not length_delta_ok:
        admissible = False
        reasons.append("length delta exceeds policy")
    else:
        reasons.append("length delta within policy")

    if not consensus_fraction_ok:
        admissible = False
        reasons.append("consensus fraction below threshold")
    else:
        reasons.append("consensus fraction satisfies threshold")

    return NodeReplayConsensusStatus(
        node_id=node_log.node_id,
        admissible=admissible,
        epoch_aligned=epoch_aligned,
        log_hash_aligned=log_hash_aligned,
        prefix_aligned=prefix_aligned,
        length_delta_ok=length_delta_ok,
        consensus_fraction_ok=consensus_fraction_ok,
        matching_prefix_length=matching_prefix_length,
        local_log_length=local_log_length,
        consensus_fraction=consensus_fraction,
        consensus_risk=_clamp01(1.0 - consensus_fraction),
        reasons=tuple(reasons),
    )


def _synthesize_actions(
    statuses: tuple[NodeReplayConsensusStatus, ...],
    reference_node_id: str,
    consensus_ready: bool,
) -> tuple[ReplayConsensusAction, ...]:
    actions: list[ReplayConsensusAction] = []

    def _add(action_type: str, source_node_id: str, target_node_id: str, blocking: bool, ready: bool, detail: str) -> None:
        actions.append(
            ReplayConsensusAction(
                action_index=len(actions),
                action_type=action_type,
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                blocking=blocking,
                ready=ready,
                detail=detail,
            )
        )

    for status in statuses:
        _add("compare_prefix", reference_node_id, status.node_id, blocking=not status.prefix_aligned, ready=True, detail="deterministic prefix comparison complete")
        _add("compare_log_hash", reference_node_id, status.node_id, blocking=not status.log_hash_aligned, ready=True, detail="deterministic log hash comparison complete")
        _add("align_epoch", reference_node_id, status.node_id, blocking=not status.epoch_aligned, ready=status.epoch_aligned, detail="deterministic epoch alignment evaluated")
        _add("hold_node", reference_node_id, status.node_id, blocking=not status.admissible, ready=not status.admissible, detail="node held" if not status.admissible else "node hold not required")
        _add("admit_log", status.node_id, reference_node_id, blocking=not status.admissible, ready=status.admissible, detail="node log admissible" if status.admissible else "node log not admissible")
        _add("flag_divergence", status.node_id, reference_node_id, blocking=not status.admissible, ready=not status.admissible, detail="node divergence flagged" if not status.admissible else "no divergence flag required")

    _add(
        "emit_consensus_view",
        reference_node_id,
        reference_node_id,
        blocking=not consensus_ready,
        ready=consensus_ready,
        detail="consensus view ready" if consensus_ready else "consensus view blocked",
    )

    return tuple(actions)


def _assemble_rationale(
    policy: ReplayConsensusPolicy,
    statuses: tuple[NodeReplayConsensusStatus, ...],
    structurally_consistent: bool,
    consensus_ready: bool,
) -> tuple[str, ...]:
    rationale: list[str] = ["reference log selected deterministically"]

    if all(status.epoch_aligned for status in statuses):
        rationale.append("epoch alignment satisfied")
    elif policy.require_matching_epoch:
        rationale.append("epoch mismatch blocks consensus readiness")
    else:
        rationale.append("epoch mismatch tolerated by policy")

    if all(status.log_hash_aligned for status in statuses):
        rationale.append("log hash alignment satisfied")
    elif policy.require_matching_log_hash:
        rationale.append("log hash mismatch blocks consensus readiness")
    else:
        rationale.append("log hash mismatch tolerated by policy")

    if all(status.prefix_aligned for status in statuses):
        rationale.append("prefix agreement satisfies policy")
    elif policy.require_full_prefix_agreement:
        rationale.append("prefix divergence blocks consensus readiness")
    else:
        rationale.append("prefix divergence tolerated by policy")

    rationale.append("length skew allowed by policy" if policy.allow_length_skew else "length skew blocked by policy")
    if any(not status.admissible for status in statuses):
        rationale.append("node held due to replay divergence")
    rationale.append("replay consensus ready" if consensus_ready else "replay consensus not ready")
    rationale.append("structurally consistent" if structurally_consistent else "structural consistency violated")

    return tuple(rationale)


def _validate_and_sort_logs(node_replay_logs: tuple[NodeReplayLog, ...]) -> tuple[NodeReplayLog, ...]:
    if len(node_replay_logs) == 0:
        raise ValueError("node replay logs must be non-empty")
    if any(not isinstance(log, NodeReplayLog) for log in node_replay_logs):
        raise ValueError("node replay logs must contain NodeReplayLog values")

    sorted_logs = tuple(sorted(node_replay_logs, key=lambda item: item.node_id))
    node_ids = tuple(log.node_id for log in sorted_logs)
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("duplicate node_id values are not allowed")
    return sorted_logs


def _structural_consistency(statuses: tuple[NodeReplayConsensusStatus, ...], policy: ReplayConsensusPolicy) -> bool:
    for status in statuses:
        if policy.require_matching_epoch and not status.epoch_aligned:
            return False
        if policy.require_matching_log_hash and not status.log_hash_aligned:
            return False
        if policy.require_full_prefix_agreement and not status.prefix_aligned:
            return False
        if not status.length_delta_ok:
            return False
        if not status.consensus_fraction_ok:
            return False
    return True


def _receipt_hash_payload(receipt: ReplayLogConsensusReceipt) -> dict[str, _JSONValue]:
    return receipt.to_hash_payload_dict()


def build_replay_log_consensus_receipt(
    node_replay_logs: tuple[NodeReplayLog, ...],
    policy: ReplayConsensusPolicy,
) -> ReplayLogConsensusReceipt:
    if not isinstance(policy, ReplayConsensusPolicy):
        raise ValueError("policy must be ReplayConsensusPolicy")

    logs = _validate_and_sort_logs(node_replay_logs)
    cluster_epoch = _cluster_epoch(logs)
    reference_log = _select_reference_log(logs, policy, cluster_epoch)

    statuses = tuple(
        _compute_node_status(
            node_log=node_log,
            reference_log=reference_log,
            cluster_epoch=cluster_epoch,
            policy=policy,
        )
        for node_log in logs
    )

    structurally_consistent = _structural_consistency(statuses, policy)

    status_by_id = {status.node_id: status for status in statuses}
    reference_status = status_by_id[reference_log.node_id]
    admissible_count = sum(1 for status in statuses if status.admissible)
    consensus_confidence = _clamp01(float(admissible_count) / float(len(statuses)))
    consensus_risk = _clamp01(1.0 - consensus_confidence)

    consensus_ready = bool(structurally_consistent and reference_status.admissible and admissible_count >= 1)

    actions = _synthesize_actions(statuses, reference_log.node_id, consensus_ready)

    replay_identity = _sha256_hex(
        {
            "schema_version": SCHEMA_VERSION,
            "cluster_epoch": cluster_epoch,
            "reference_node_id": reference_log.node_id,
            "reference_log_hash": reference_log.log_hash,
            "node_log_hashes": tuple((log.node_id, log.log_hash) for log in logs),
            "policy": policy.to_dict(),
        }
    )

    rationale = _assemble_rationale(policy, statuses, structurally_consistent, consensus_ready)

    provisional = ReplayLogConsensusReceipt(
        node_replay_logs=logs,
        policy_snapshot=policy,
        node_statuses=statuses,
        consensus_actions=actions,
        cluster_epoch=cluster_epoch,
        reference_node_id=reference_log.node_id,
        reference_log_hash=reference_log.log_hash,
        structurally_consistent=structurally_consistent,
        consensus_ready=consensus_ready,
        consensus_confidence=consensus_confidence,
        consensus_risk=consensus_risk,
        rationale=rationale,
        schema_version=SCHEMA_VERSION,
        replay_identity=replay_identity,
        stable_hash="0" * 64,
    )
    stable_hash = _sha256_hex(_receipt_hash_payload(provisional))

    return ReplayLogConsensusReceipt(
        node_replay_logs=logs,
        policy_snapshot=policy,
        node_statuses=statuses,
        consensus_actions=actions,
        cluster_epoch=cluster_epoch,
        reference_node_id=reference_log.node_id,
        reference_log_hash=reference_log.log_hash,
        structurally_consistent=structurally_consistent,
        consensus_ready=consensus_ready,
        consensus_confidence=consensus_confidence,
        consensus_risk=consensus_risk,
        rationale=rationale,
        schema_version=SCHEMA_VERSION,
        replay_identity=replay_identity,
        stable_hash=stable_hash,
    )


__all__ = [
    "ALLOWED_CONSENSUS_ACTION_TYPES",
    "SCHEMA_VERSION",
    "NodeReplayConsensusStatus",
    "NodeReplayLog",
    "ReplayConsensusAction",
    "ReplayConsensusPolicy",
    "ReplayLogConsensusReceipt",
    "ReplayLogEntry",
    "build_replay_log_consensus_receipt",
]
