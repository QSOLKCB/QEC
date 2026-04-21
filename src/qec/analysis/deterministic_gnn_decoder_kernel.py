# SPDX-License-Identifier: MIT
"""v138.7.0 — deterministic GNN decoder kernel."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import types
from typing import Any

from qec.analysis.canonical_hashing import (
    CanonicalHashingError,
    canonical_bytes,
    canonical_json,
    canonicalize_json,
    sha256_hex,
)

RELEASE_VERSION = "v138.7.0"
RUNTIME_KIND = "deterministic_gnn_decoder_kernel"

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | Mapping[str, "_JSONValue"]


class DeterministicGNNKernelError(ValueError):
    """Raised when deterministic GNN kernel inputs or invariants are invalid."""


def _canonicalize_json(value: Any) -> _JSONValue:
    try:
        return canonicalize_json(value)
    except CanonicalHashingError as exc:
        raise DeterministicGNNKernelError(str(exc)) from exc


def _canonical_json(value: Any) -> str:
    try:
        return canonical_json(value)
    except CanonicalHashingError as exc:
        raise DeterministicGNNKernelError(str(exc)) from exc


def _canonical_bytes(value: Any) -> bytes:
    try:
        return canonical_bytes(value)
    except CanonicalHashingError as exc:
        raise DeterministicGNNKernelError(str(exc)) from exc


def _sha256_hex(value: Any) -> str:
    try:
        return sha256_hex(value)
    except CanonicalHashingError as exc:
        raise DeterministicGNNKernelError(str(exc)) from exc


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def _round_score(value: float, digits: int) -> float:
    return round(_clamp01(value), digits)


def _immutable_mapping(mapping: Mapping[str, Any]) -> Mapping[str, _JSONValue]:
    canonical = _canonicalize_json(mapping)
    if not isinstance(canonical, dict):
        raise DeterministicGNNKernelError("mapping must serialize as an object")
    return types.MappingProxyType(canonical)


def _validate_finite_number(
    value: Any,
    *,
    field_name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise DeterministicGNNKernelError(f"{field_name} must not be a bool")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DeterministicGNNKernelError(f"{field_name} must be numeric") from exc
    if not math.isfinite(number):
        raise DeterministicGNNKernelError(f"{field_name} must be finite")
    if minimum is not None and number < minimum:
        raise DeterministicGNNKernelError(f"{field_name} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise DeterministicGNNKernelError(f"{field_name} must be <= {maximum}")
    return number


def _validate_int_config_value(value: Any, *, field_name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise DeterministicGNNKernelError(f"{field_name} must not be a bool")
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise DeterministicGNNKernelError(f"{field_name} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DeterministicGNNKernelError(f"{field_name} must be an integer") from exc
    if minimum is not None and number < minimum:
        raise DeterministicGNNKernelError(f"{field_name} must be >= {minimum}")
    return number


@dataclass(frozen=True)
class DeterministicGNNKernelConfig:
    num_rounds: int
    self_weight: float
    neighbor_weight: float
    syndrome_weight: float
    hardware_weight: float
    residual_weight: float
    damping_factor: float
    score_round_digits: int
    top_k: int
    convergence_epsilon: float
    normalization_policy: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "num_rounds": self.num_rounds,
            "self_weight": self.self_weight,
            "neighbor_weight": self.neighbor_weight,
            "syndrome_weight": self.syndrome_weight,
            "hardware_weight": self.hardware_weight,
            "residual_weight": self.residual_weight,
            "damping_factor": self.damping_factor,
            "score_round_digits": self.score_round_digits,
            "top_k": self.top_k,
            "convergence_epsilon": self.convergence_epsilon,
            "normalization_policy": self.normalization_policy,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class SyndromeGraphNode:
    node_id: str
    syndrome: float
    parity: float
    defect: float
    hardware_sideband: Mapping[str, _JSONValue]

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "node_id": self.node_id,
            "syndrome": self.syndrome,
            "parity": self.parity,
            "defect": self.defect,
            "hardware_sideband": _canonicalize_json(self.hardware_sideband),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class SyndromeGraphEdge:
    edge_id: str
    source_node_id: str
    target_node_id: str
    coupling_weight: float
    edge_sideband: Mapping[str, _JSONValue]

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "edge_id": self.edge_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "coupling_weight": self.coupling_weight,
            "edge_sideband": _canonicalize_json(self.edge_sideband),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class GNNMessageSnapshot:
    round_index: int
    node_messages: Mapping[str, float]
    max_delta: float

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "round_index": self.round_index,
            "node_messages": _canonicalize_json(self.node_messages),
            "max_delta": self.max_delta,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class GNNCorrectionProposal:
    proposal_id: str
    target_nodes: tuple[str, ...]
    target_edges: tuple[str, ...]
    action_class: str
    proposal_score: float
    confidence: float
    rationale: Mapping[str, _JSONValue]

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "proposal_id": self.proposal_id,
            "target_nodes": self.target_nodes,
            "target_edges": self.target_edges,
            "action_class": self.action_class,
            "proposal_score": self.proposal_score,
            "confidence": self.confidence,
            "rationale": _canonicalize_json(self.rationale),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class DeterministicGNNKernelReceipt:
    release_version: str
    runtime_kind: str
    config_hash: str
    graph_hash: str
    input_hash: str
    round_count: int
    converged: bool
    proposal_count: int
    top_proposal_hash: str | None
    kernel_result_hash: str
    replay_identity: str
    decoder_core_modified: bool
    receipt_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "release_version": self.release_version,
            "runtime_kind": self.runtime_kind,
            "config_hash": self.config_hash,
            "graph_hash": self.graph_hash,
            "input_hash": self.input_hash,
            "round_count": self.round_count,
            "converged": self.converged,
            "proposal_count": self.proposal_count,
            "top_proposal_hash": self.top_proposal_hash,
            "kernel_result_hash": self.kernel_result_hash,
            "replay_identity": self.replay_identity,
            "decoder_core_modified": self.decoder_core_modified,
            "receipt_hash": self.receipt_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("receipt_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())

    def __post_init__(self) -> None:
        if self.receipt_hash and self.receipt_hash != self.stable_hash():
            raise DeterministicGNNKernelError("receipt_hash must match stable_hash payload")


@dataclass(frozen=True)
class DeterministicGNNKernelResult:
    release_version: str
    runtime_kind: str
    config: DeterministicGNNKernelConfig
    nodes: tuple[SyndromeGraphNode, ...]
    edges: tuple[SyndromeGraphEdge, ...]
    message_snapshots: tuple[GNNMessageSnapshot, ...]
    final_node_scores: Mapping[str, float]
    proposals: tuple[GNNCorrectionProposal, ...]
    round_count: int
    converged: bool
    convergence_delta: float
    config_hash: str
    graph_hash: str
    input_hash: str
    replay_identity: str
    result_hash: str
    receipt: DeterministicGNNKernelReceipt

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "release_version": self.release_version,
            "runtime_kind": self.runtime_kind,
            "config": self.config.to_dict(),
            "nodes": tuple(node.to_dict() for node in self.nodes),
            "edges": tuple(edge.to_dict() for edge in self.edges),
            "message_snapshots": tuple(snapshot.to_dict() for snapshot in self.message_snapshots),
            "final_node_scores": _canonicalize_json(self.final_node_scores),
            "proposals": tuple(proposal.to_dict() for proposal in self.proposals),
            "round_count": self.round_count,
            "converged": self.converged,
            "convergence_delta": self.convergence_delta,
            "config_hash": self.config_hash,
            "graph_hash": self.graph_hash,
            "input_hash": self.input_hash,
            "replay_identity": self.replay_identity,
            "result_hash": self.result_hash,
            "receipt": self.receipt.to_dict(),
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("result_hash")
        payload["receipt"] = self.receipt.to_hash_payload_dict()
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


def _normalize_config(config: DeterministicGNNKernelConfig | Mapping[str, Any]) -> DeterministicGNNKernelConfig:
    if isinstance(config, DeterministicGNNKernelConfig):
        raw_config: Mapping[str, Any] = config.to_dict()
    else:
        if not isinstance(config, Mapping):
            raise DeterministicGNNKernelError("config must be a DeterministicGNNKernelConfig or mapping")
        raw_config = config

    normalized = DeterministicGNNKernelConfig(
        num_rounds=_validate_int_config_value(raw_config.get("num_rounds", 0), field_name="num_rounds", minimum=0),
        self_weight=_validate_finite_number(raw_config.get("self_weight", 0.0), field_name="self_weight", minimum=0.0, maximum=1.0),
        neighbor_weight=_validate_finite_number(raw_config.get("neighbor_weight", 0.0), field_name="neighbor_weight", minimum=0.0, maximum=1.0),
        syndrome_weight=_validate_finite_number(raw_config.get("syndrome_weight", 0.0), field_name="syndrome_weight", minimum=0.0, maximum=1.0),
        hardware_weight=_validate_finite_number(raw_config.get("hardware_weight", 0.0), field_name="hardware_weight", minimum=0.0, maximum=1.0),
        residual_weight=_validate_finite_number(raw_config.get("residual_weight", 0.0), field_name="residual_weight", minimum=0.0, maximum=1.0),
        damping_factor=_validate_finite_number(raw_config.get("damping_factor", 0.0), field_name="damping_factor", minimum=0.0, maximum=1.0),
        score_round_digits=_validate_int_config_value(raw_config.get("score_round_digits", 6), field_name="score_round_digits", minimum=0),
        top_k=_validate_int_config_value(raw_config.get("top_k", 1), field_name="top_k", minimum=1),
        convergence_epsilon=_validate_finite_number(raw_config.get("convergence_epsilon", 1e-6), field_name="convergence_epsilon", minimum=0.0),
        normalization_policy=str(raw_config.get("normalization_policy", "clamp_0_1")),
    )
    if normalized.normalization_policy != "clamp_0_1":
        raise DeterministicGNNKernelError("normalization_policy must be 'clamp_0_1'")
    return normalized


def _normalize_nodes(raw_nodes: Sequence[SyndromeGraphNode | Mapping[str, Any]]) -> tuple[SyndromeGraphNode, ...]:
    normalized: list[SyndromeGraphNode] = []
    for raw in raw_nodes:
        if isinstance(raw, SyndromeGraphNode):
            raw_node: Mapping[str, Any] = {
                "node_id": raw.node_id,
                "syndrome": raw.syndrome,
                "parity": raw.parity,
                "defect": raw.defect,
                "hardware_sideband": raw.hardware_sideband,
            }
        else:
            if not isinstance(raw, Mapping):
                raise DeterministicGNNKernelError("node entries must be SyndromeGraphNode or mappings")
            raw_node = raw
        node = SyndromeGraphNode(
            node_id=str(raw_node.get("node_id", "")),
            syndrome=_validate_finite_number(raw_node.get("syndrome", 0.0), field_name="node.syndrome"),
            parity=_validate_finite_number(raw_node.get("parity", 0.0), field_name="node.parity"),
            defect=_validate_finite_number(raw_node.get("defect", 0.0), field_name="node.defect"),
            hardware_sideband=_immutable_mapping(raw_node.get("hardware_sideband", {})),
        )
        if not node.node_id:
            raise DeterministicGNNKernelError("node_id must be non-empty")
        normalized.append(node)

    if not normalized:
        raise DeterministicGNNKernelError("graph must contain at least one node")

    ids = [node.node_id for node in normalized]
    if len(ids) != len(set(ids)):
        raise DeterministicGNNKernelError("duplicate node ids are not allowed")
    if ids != sorted(ids):
        raise DeterministicGNNKernelError("node ordering must already be canonical")
    return tuple(normalized)


def _normalize_edges(raw_edges: Sequence[SyndromeGraphEdge | Mapping[str, Any]], node_ids: set[str]) -> tuple[SyndromeGraphEdge, ...]:
    normalized: list[SyndromeGraphEdge] = []
    for raw in raw_edges:
        if isinstance(raw, SyndromeGraphEdge):
            raw_edge: Mapping[str, Any] = {
                "edge_id": raw.edge_id,
                "source_node_id": raw.source_node_id,
                "target_node_id": raw.target_node_id,
                "coupling_weight": raw.coupling_weight,
                "edge_sideband": raw.edge_sideband,
            }
        else:
            if not isinstance(raw, Mapping):
                raise DeterministicGNNKernelError("edge entries must be SyndromeGraphEdge or mappings")
            raw_edge = raw
        edge = SyndromeGraphEdge(
            edge_id=str(raw_edge.get("edge_id", "")),
            source_node_id=str(raw_edge.get("source_node_id", "")),
            target_node_id=str(raw_edge.get("target_node_id", "")),
            coupling_weight=_validate_finite_number(raw_edge.get("coupling_weight", 0.0), field_name="edge.coupling_weight", minimum=0.0, maximum=1.0),
            edge_sideband=_immutable_mapping(raw_edge.get("edge_sideband", {})),
        )
        if not edge.edge_id:
            raise DeterministicGNNKernelError("edge_id must be non-empty")
        if edge.source_node_id not in node_ids or edge.target_node_id not in node_ids:
            raise DeterministicGNNKernelError("edge references unknown node ids")
        if edge.source_node_id == edge.target_node_id:
            raise DeterministicGNNKernelError("self-loop edges are not allowed")
        normalized.append(edge)

    edge_ids = [edge.edge_id for edge in normalized]
    if len(edge_ids) != len(set(edge_ids)):
        raise DeterministicGNNKernelError("duplicate edge ids are not allowed")

    canonical_order = sorted(normalized, key=lambda edge: (edge.source_node_id, edge.target_node_id, edge.edge_id))
    if list(normalized) != canonical_order:
        raise DeterministicGNNKernelError("edge ordering must already be canonical")
    return tuple(normalized)


def _hardware_scalar(node: SyndromeGraphNode) -> float:
    if not node.hardware_sideband:
        return 0.0
    values: list[float] = []
    for key in sorted(node.hardware_sideband.keys()):
        value = node.hardware_sideband[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DeterministicGNNKernelError(f"hardware_sideband[{key!r}] must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise DeterministicGNNKernelError(f"hardware_sideband[{key!r}] must be finite")
        values.append(number)
    return sum(values) / float(len(values))


def _action_class(score: float) -> str:
    if score >= 0.67:
        return "apply_primary_correction"
    if score >= 0.34:
        return "apply_secondary_correction"
    return "monitor_only"


def _result_core_payload(
    *,
    cfg: DeterministicGNNKernelConfig,
    nodes: tuple[SyndromeGraphNode, ...],
    edges: tuple[SyndromeGraphEdge, ...],
    snapshots: tuple[GNNMessageSnapshot, ...],
    final_scores: Mapping[str, float],
    ranked_proposals: tuple[GNNCorrectionProposal, ...],
    converged: bool,
    max_delta: float,
    config_hash: str,
    graph_hash: str,
    input_hash: str,
) -> dict[str, _JSONValue]:
    return {
        "release_version": RELEASE_VERSION,
        "runtime_kind": RUNTIME_KIND,
        "config": cfg.to_dict(),
        "nodes": tuple(node.to_dict() for node in nodes),
        "edges": tuple(edge.to_dict() for edge in edges),
        "message_snapshots": tuple(snapshot.to_dict() for snapshot in snapshots),
        "final_node_scores": _canonicalize_json(final_scores),
        "proposals": tuple(proposal.to_dict() for proposal in ranked_proposals),
        "round_count": len(snapshots),
        "converged": converged,
        "convergence_delta": round(max_delta, cfg.score_round_digits),
        "config_hash": config_hash,
        "graph_hash": graph_hash,
        "input_hash": input_hash,
    }


def build_deterministic_gnn_decoder_kernel(
    *,
    config: DeterministicGNNKernelConfig | Mapping[str, Any],
    nodes: Sequence[SyndromeGraphNode | Mapping[str, Any]],
    edges: Sequence[SyndromeGraphEdge | Mapping[str, Any]],
) -> DeterministicGNNKernelResult:
    cfg = _normalize_config(config)
    normalized_nodes = _normalize_nodes(nodes)
    node_id_set = {node.node_id for node in normalized_nodes}
    normalized_edges = _normalize_edges(edges, node_id_set)

    adjacency: dict[str, list[tuple[str, float, str]]] = defaultdict(list)
    for edge in normalized_edges:
        adjacency[edge.source_node_id].append((edge.target_node_id, edge.coupling_weight, edge.edge_id))
        adjacency[edge.target_node_id].append((edge.source_node_id, edge.coupling_weight, edge.edge_id))
    for node_id in adjacency:
        adjacency[node_id].sort(key=lambda item: (item[0], item[2]))

    node_ids = tuple(node.node_id for node in normalized_nodes)
    base_signals: dict[str, float] = {}
    for node in normalized_nodes:
        base_signals[node.node_id] = _round_score(
            (cfg.syndrome_weight * node.syndrome)
            + (cfg.hardware_weight * _hardware_scalar(node))
            + (cfg.residual_weight * node.defect)
            + (cfg.self_weight * node.parity),
            cfg.score_round_digits,
        )

    state = {node_id: base_signals[node_id] for node_id in node_ids}
    snapshots: list[GNNMessageSnapshot] = []
    max_delta = 0.0
    converged = cfg.num_rounds == 0

    for round_index in range(cfg.num_rounds):
        next_state: dict[str, float] = {}
        max_delta = 0.0
        for node_id in node_ids:
            neighbors = adjacency.get(node_id, [])
            neighbor_signal = 0.0
            if neighbors:
                total = 0.0
                for neighbor_id, coupling, _ in neighbors:
                    total += state[neighbor_id] * coupling
                neighbor_signal = total / float(len(neighbors))
            raw = (
                (cfg.self_weight * state[node_id])
                + (cfg.neighbor_weight * neighbor_signal)
                + base_signals[node_id]
            )
            damped = (cfg.damping_factor * state[node_id]) + ((1.0 - cfg.damping_factor) * raw)
            bounded = _round_score(damped, cfg.score_round_digits)
            next_state[node_id] = bounded
            max_delta = max(max_delta, abs(bounded - state[node_id]))

        snapshot = GNNMessageSnapshot(
            round_index=round_index,
            node_messages=types.MappingProxyType({node_id: next_state[node_id] for node_id in node_ids}),
            max_delta=round(max_delta, cfg.score_round_digits),
        )
        snapshots.append(snapshot)
        state = next_state

        if max_delta <= cfg.convergence_epsilon:
            converged = True
            break

    proposals: list[GNNCorrectionProposal] = []
    for node in normalized_nodes:
        score = _round_score(state[node.node_id], cfg.score_round_digits)
        confidence = _round_score(
            0.5 * score + 0.5 * _clamp01(1.0 - min(1.0, max_delta)),
            cfg.score_round_digits,
        )
        rationale = {
            "base_signal": base_signals[node.node_id],
            "final_score": score,
            "neighbor_degree": float(len(adjacency.get(node.node_id, []))),
        }
        proposals.append(
            GNNCorrectionProposal(
                proposal_id=f"proposal::{node.node_id}",
                target_nodes=(node.node_id,),
                target_edges=tuple(edge_id for _, _, edge_id in adjacency.get(node.node_id, [])),
                action_class=_action_class(score),
                proposal_score=score,
                confidence=confidence,
                rationale=_immutable_mapping(rationale),
            )
        )

    proposals.sort(
        key=lambda proposal: (
            -proposal.proposal_score,
            -proposal.confidence,
            proposal.target_nodes,
            proposal.proposal_id,
        )
    )
    ranked_proposals = tuple(proposals[: min(cfg.top_k, len(proposals))])

    final_scores = types.MappingProxyType({node_id: state[node_id] for node_id in node_ids})
    config_hash = cfg.stable_hash()
    graph_hash = _sha256_hex(
        {
            "nodes": tuple(node.to_dict() for node in normalized_nodes),
            "edges": tuple(edge.to_dict() for edge in normalized_edges),
        }
    )
    input_hash = _sha256_hex(
        {
            "config": cfg.to_dict(),
            "nodes": tuple(node.to_dict() for node in normalized_nodes),
            "edges": tuple(edge.to_dict() for edge in normalized_edges),
        }
    )

    snapshots_tuple = tuple(snapshots)
    result_hash = _sha256_hex(
        _result_core_payload(
            cfg=cfg,
            nodes=normalized_nodes,
            edges=normalized_edges,
            snapshots=snapshots_tuple,
            final_scores=final_scores,
            ranked_proposals=ranked_proposals,
            converged=converged,
            max_delta=max_delta,
            config_hash=config_hash,
            graph_hash=graph_hash,
            input_hash=input_hash,
        )
    )
    replay_identity = _sha256_hex({"result_hash": result_hash, "input_hash": input_hash})
    receipt_payload = DeterministicGNNKernelReceipt(
        release_version=RELEASE_VERSION,
        runtime_kind=RUNTIME_KIND,
        config_hash=config_hash,
        graph_hash=graph_hash,
        input_hash=input_hash,
        round_count=len(snapshots_tuple),
        converged=converged,
        proposal_count=len(ranked_proposals),
        top_proposal_hash=ranked_proposals[0].stable_hash() if ranked_proposals else None,
        kernel_result_hash=result_hash,
        replay_identity=replay_identity,
        decoder_core_modified=False,
        receipt_hash="",
    )
    receipt = DeterministicGNNKernelReceipt(**{**receipt_payload.to_dict(), "receipt_hash": receipt_payload.stable_hash()})
    result = DeterministicGNNKernelResult(
        release_version=RELEASE_VERSION,
        runtime_kind=RUNTIME_KIND,
        config=cfg,
        nodes=normalized_nodes,
        edges=normalized_edges,
        message_snapshots=snapshots_tuple,
        final_node_scores=final_scores,
        proposals=ranked_proposals,
        round_count=len(snapshots_tuple),
        converged=converged,
        convergence_delta=round(max_delta, cfg.score_round_digits),
        config_hash=config_hash,
        graph_hash=graph_hash,
        input_hash=input_hash,
        replay_identity=replay_identity,
        result_hash=result_hash,
        receipt=receipt,
    )

    if result.receipt.receipt_hash != result.receipt.stable_hash():
        raise DeterministicGNNKernelError("receipt hash mismatch")
    if result.receipt.kernel_result_hash != result.result_hash:
        raise DeterministicGNNKernelError("receipt kernel_result_hash mismatch")
    if any(not (0.0 <= score <= 1.0) for score in result.final_node_scores.values()):
        raise DeterministicGNNKernelError("final scores must remain bounded in [0,1]")
    return result
