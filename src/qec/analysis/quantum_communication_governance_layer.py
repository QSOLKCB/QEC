from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

_PRECISION = 12
_VERSION = "v137.1.18"


def _round64(value: float) -> float:
    return float(round(float(value), _PRECISION))


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return _canonical_json(payload).encode("utf-8")


def _sha256_hex_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_hex_mapping(payload: Mapping[str, Any]) -> str:
    return _sha256_hex_bytes(_canonical_json_bytes(payload))


def _norm_token(value: str, *, name: str) -> str:
    token = str(value).strip()
    if not token:
        raise ValueError(f"{name} must be a non-empty token")
    return token


def _norm_metric_items(values: Sequence[tuple[str, float]]) -> tuple[tuple[str, float], ...]:
    normalized: list[tuple[str, float]] = []
    seen_keys: set[str] = set()
    for key, value in values:
        norm_key = _norm_token(key, name="metric key")
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError("metric values must be finite float64-compatible values")
        if norm_key in seen_keys:
            raise ValueError(f"duplicate metric key is not allowed: {norm_key}")
        seen_keys.add(norm_key)
        normalized.append((norm_key, _round64(numeric_value)))
    return tuple(sorted(normalized, key=lambda item: item[0]))


def _norm_flags(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted({_norm_token(v, name="governance flag") for v in values}))


@dataclass(frozen=True)
class NodeState:
    node_id: str
    epoch: int
    state_hash: str
    metrics: tuple[tuple[str, float], ...]
    governance_flags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "epoch": int(self.epoch),
            "state_hash": self.state_hash,
            "metrics": [[k, _round64(v)] for k, v in self.metrics],
            "governance_flags": list(self.governance_flags),
        }


@dataclass(frozen=True)
class ReplaySignatureEntry:
    index: int
    node_id: str
    node_state_hash: str
    parent_signature: str
    signature: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "node_id": self.node_id,
            "node_state_hash": self.node_state_hash,
            "parent_signature": self.parent_signature,
            "signature": self.signature,
        }


@dataclass(frozen=True)
class ReplaySignatureChain:
    entries: tuple[ReplaySignatureEntry, ...]
    head_signature: str
    replay_identity: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "head_signature": self.head_signature,
            "replay_identity": self.replay_identity,
        }


@dataclass(frozen=True)
class DriftProvenance:
    node_pair: tuple[str, str]
    metric_key: str
    delta: float
    provenance_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_pair": [self.node_pair[0], self.node_pair[1]],
            "metric_key": self.metric_key,
            "delta": _round64(self.delta),
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class QuantumCommunicationGovernanceReport:
    synchronized_states: tuple[NodeState, ...]
    replay_chain: ReplaySignatureChain
    drift_provenance: tuple[DriftProvenance, ...]
    governance_trust_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": _VERSION,
            "synchronized_states": [state.to_dict() for state in self.synchronized_states],
            "replay_chain": self.replay_chain.to_dict(),
            "drift_provenance": [record.to_dict() for record in self.drift_provenance],
            "governance_trust_score": _round64(self.governance_trust_score),
            "simulation_only": True,
            "governance_only": True,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


def normalize_node_state(node_state: NodeState) -> NodeState:
    node_id = _norm_token(node_state.node_id, name="node_id")
    state_hash = _norm_token(node_state.state_hash, name="state_hash")
    epoch = int(node_state.epoch)
    if epoch < 0:
        raise ValueError("epoch must be >= 0")
    metrics = _norm_metric_items(node_state.metrics)
    flags = _norm_flags(node_state.governance_flags)
    return NodeState(node_id=node_id, epoch=epoch, state_hash=state_hash, metrics=metrics, governance_flags=flags)


def _node_state_digest(node_state: NodeState) -> str:
    return _sha256_hex_mapping(node_state.to_dict())


def synchronize_node_states(node_states: Sequence[NodeState]) -> tuple[NodeState, ...]:
    if not node_states:
        raise ValueError("node_states must not be empty")
    normalized = [normalize_node_state(state) for state in node_states]
    by_node: dict[str, list[NodeState]] = {}
    for state in normalized:
        by_node.setdefault(state.node_id, []).append(state)

    synchronized: list[NodeState] = []
    for node_id in sorted(by_node.keys()):
        candidates = by_node[node_id]
        selected = sorted(candidates, key=lambda s: (s.epoch, s.state_hash, _node_state_digest(s)))[-1]
        synchronized.append(selected)
    return tuple(synchronized)


def build_secure_replay_signature_chain(
    synchronized_states: Sequence[NodeState],
    *,
    replay_namespace: str = "quantum-communication-governance",
) -> ReplaySignatureChain:
    namespace = _norm_token(replay_namespace, name="replay_namespace")
    if not synchronized_states:
        raise ValueError("synchronized_states must not be empty")
    entries: list[ReplaySignatureEntry] = []
    parent_signature = _sha256_hex_mapping({"namespace": namespace, "version": _VERSION})
    for index, state in enumerate(synchronized_states):
        node_state_hash = _node_state_digest(state)
        signature = _sha256_hex_mapping(
            {
                "index": int(index),
                "node_id": state.node_id,
                "node_state_hash": node_state_hash,
                "parent_signature": parent_signature,
            }
        )
        entry = ReplaySignatureEntry(
            index=index,
            node_id=state.node_id,
            node_state_hash=node_state_hash,
            parent_signature=parent_signature,
            signature=signature,
        )
        entries.append(entry)
        parent_signature = signature

    head = parent_signature
    replay_identity = _sha256_hex_mapping(
        {
            "namespace": namespace,
            "version": _VERSION,
            "head_signature": head,
            "entry_signatures": [entry.signature for entry in entries],
        }
    )
    return ReplaySignatureChain(entries=tuple(entries), head_signature=head, replay_identity=replay_identity)


def compute_cross_node_drift_provenance(synchronized_states: Sequence[NodeState]) -> tuple[DriftProvenance, ...]:
    metric_maps: dict[str, dict[str, float]] = {}
    seen_node_ids: set[str] = set()
    for state in synchronized_states:
        if state.node_id in seen_node_ids:
            raise ValueError(f"Duplicate node_id in synchronized_states: {state.node_id}")
        seen_node_ids.add(state.node_id)
        metric_maps[state.node_id] = dict(state.metrics)
    node_ids = tuple(sorted(metric_maps.keys()))
    output: list[DriftProvenance] = []
    for left_index, left_node in enumerate(node_ids):
        for right_node in node_ids[left_index + 1 :]:
            left_metrics = metric_maps[left_node]
            right_metrics = metric_maps[right_node]
            all_keys = tuple(sorted(set(left_metrics.keys()) | set(right_metrics.keys())))
            for key in all_keys:
                delta = _round64(float(right_metrics.get(key, 0.0)) - float(left_metrics.get(key, 0.0)))
                payload = {
                    "node_pair": [left_node, right_node],
                    "metric_key": key,
                    "delta": delta,
                }
                output.append(
                    DriftProvenance(
                        node_pair=(left_node, right_node),
                        metric_key=key,
                        delta=delta,
                        provenance_hash=_sha256_hex_mapping(payload),
                    )
                )
    return tuple(output)


def compute_governance_trust_score(
    synchronized_states: Sequence[NodeState],
    drift_provenance: Sequence[DriftProvenance],
) -> float:
    if not synchronized_states:
        raise ValueError("synchronized_states must not be empty")
    trusted_nodes = sum(1 for state in synchronized_states if "trusted" in state.governance_flags)
    trusted_ratio = float(trusted_nodes) / float(len(synchronized_states))
    avg_abs_drift = (
        sum(abs(float(record.delta)) for record in drift_provenance) / float(len(drift_provenance))
        if drift_provenance
        else 0.0
    )
    bounded_drift_penalty = min(max(avg_abs_drift, 0.0), 1.0)
    trust = (0.65 * trusted_ratio) + (0.35 * (1.0 - bounded_drift_penalty))
    return min(max(_round64(trust), 0.0), 1.0)


def run_quantum_communication_governance_layer(
    node_states: Sequence[NodeState],
    *,
    replay_namespace: str = "quantum-communication-governance",
) -> QuantumCommunicationGovernanceReport:
    synchronized = synchronize_node_states(node_states)
    replay_chain = build_secure_replay_signature_chain(synchronized, replay_namespace=replay_namespace)
    drift_provenance = compute_cross_node_drift_provenance(synchronized)
    trust_score = compute_governance_trust_score(synchronized, drift_provenance)
    return QuantumCommunicationGovernanceReport(
        synchronized_states=synchronized,
        replay_chain=replay_chain,
        drift_provenance=drift_provenance,
        governance_trust_score=trust_score,
    )
