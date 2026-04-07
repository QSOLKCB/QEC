"""v137.10.0 — Hypothesis Lattice.

Deterministic Layer-4 reasoning consumer of atomic signal observatory artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Any, Mapping

from qec.analysis.atomic_signal_transduction_observatory import AtomicSignalTransductionObservatoryResult
from qec.analysis.cross_modal_replay_certification import CrossModalReplayCertificationResult
from qec.analysis.rf_equalization_and_ground_station_compensation import RFEqualizationResult
from qec.analysis.satellite_signal_baseline_and_orbital_noise import SatelliteBaselineResult
from qec.analysis.telecom_line_recovery_and_sync import TelecomRecoveryResult

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

_HYPOTHESIS_LATTICE_VERSION = 1
# Only the full-chain profile is currently implemented.  The profile value is
# metadata-only: it influences the deterministic lattice_id/hash but does not
# alter node/edge construction or scoring.  Additional profiles must be added
# here only when their distinct construction semantics are implemented.
_LATTICE_PROFILE_ORDER: tuple[str, ...] = ("full_chain_hypothesis_lattice",)

HYPOTHESIS_LATTICE_LAW = "HYPOTHESIS_LATTICE_LAW"
DETERMINISTIC_LATTICE_ORDERING_RULE = "DETERMINISTIC_LATTICE_ORDERING_RULE"
REPLAY_SAFE_LATTICE_IDENTITY_RULE = "REPLAY_SAFE_LATTICE_IDENTITY_RULE"
BOUNDED_LATTICE_SCORE_RULE = "BOUNDED_LATTICE_SCORE_RULE"


def _canonicalize_json(value: Any) -> _JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float values are not allowed")
        return value
    if isinstance(value, tuple):
        return tuple(_canonicalize_json(v) for v in value)
    if isinstance(value, list):
        return tuple(_canonicalize_json(v) for v in value)
    if isinstance(value, Mapping):
        keys = tuple(value.keys())
        if any(not isinstance(k, str) for k in keys):
            raise ValueError("payload keys must be strings")
        out: dict[str, _JSONValue] = {}
        for key in sorted(keys):
            out[key] = _canonicalize_json(value[key])
        return out
    raise ValueError(f"unsupported canonical payload type: {type(value)!r}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonicalize_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_hex(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _clamp01(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("score must be numeric")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError("score must be finite")
    return min(1.0, max(0.0, score))


def _mean(values: tuple[float, ...], default: float = 1.0) -> float:
    if not values:
        return _clamp01(default)
    return _clamp01(float(sum(values) / len(values)))


def _validate_unit_interval(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a numeric value")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{name} must be finite and in [0.0, 1.0]")


def _validate_profile(lattice_profile: str) -> None:
    if lattice_profile not in _LATTICE_PROFILE_ORDER:
        raise ValueError(
            f"lattice_profile must be one of {_LATTICE_PROFILE_ORDER}; "
            f"received {lattice_profile!r}"
        )


def _validate_optional_fixture_payload(score_fixture: tuple[float, ...] | None) -> None:
    if score_fixture is None:
        return
    if not isinstance(score_fixture, tuple):
        raise ValueError("score_fixture must be a tuple")
    for value in score_fixture:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("score_fixture values must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError("score_fixture values must be finite")


def _finalize_identity(obj: Any, hash_field: str) -> Any:
    return replace(obj, **{hash_field: obj.stable_hash()})


def _validate_observatory_artifact(observatory_artifact: AtomicSignalTransductionObservatoryResult) -> None:
    if not isinstance(observatory_artifact, AtomicSignalTransductionObservatoryResult):
        raise ValueError("observatory_artifact must be a AtomicSignalTransductionObservatoryResult")
    if observatory_artifact.stable_hash() != observatory_artifact.atomic_signal_observatory_hash:
        raise ValueError("observatory_artifact atomic_signal_observatory_hash must match stable_hash")
    if observatory_artifact.observation_count != len(observatory_artifact.observations):
        raise ValueError("observatory_artifact observation_count must match len(observations)")
    if observatory_artifact.window_count != len(observatory_artifact.windows):
        raise ValueError("observatory_artifact window_count must match len(windows)")
    if observatory_artifact.observation_count <= 0 or not observatory_artifact.observations:
        raise ValueError("observatory_artifact must contain at least one observation")
    if observatory_artifact.window_count <= 0 or not observatory_artifact.windows:
        raise ValueError("observatory_artifact must contain at least one window")

    expected_observation_order = tuple(
        sorted(
            observatory_artifact.observations,
            key=lambda o: (o.observation_index, o.observatory_profile, o.observation_id),
        )
    )
    if observatory_artifact.observations != expected_observation_order:
        raise ValueError("observatory_artifact observations must be in canonical deterministic order")

    expected_window_order = tuple(
        sorted(
            observatory_artifact.windows,
            key=lambda w: (w.window_index, w.window_id, w.window_hash),
        )
    )
    if observatory_artifact.windows != expected_window_order:
        raise ValueError("observatory_artifact windows must be in canonical deterministic order")

    # Derive expected lineage from the top-level source_* fields so that forged
    # source_* mutations that are internally consistent with the observatory hash
    # cannot propagate into nested observations/windows with mismatched lineage.
    expected_nested_lineage: tuple[str, ...] = (
        observatory_artifact.source_feature_schema_hash,
        observatory_artifact.source_spectral_reasoning_hash,
        observatory_artifact.source_copper_channel_battery_hash,
        observatory_artifact.source_telecom_recovery_hash,
        observatory_artifact.source_satellite_baseline_hash,
        observatory_artifact.source_rf_equalization_hash,
        observatory_artifact.source_replay_certification_hash,
    )

    observation_ids: frozenset[str] = frozenset(
        o.observation_id for o in observatory_artifact.observations
    )

    for observation in observatory_artifact.observations:
        if observation.observation_hash != observation.stable_hash():
            raise ValueError("observatory_artifact observation_hash must match stable_hash")
        if observation.observation_id != observation.observation_hash:
            raise ValueError("observatory_artifact observation_id must equal observation_hash")
        if observation.lineage_chain != expected_nested_lineage:
            raise ValueError(
                "observatory_artifact observation lineage_chain must match source fields"
            )

    for window in observatory_artifact.windows:
        if window.window_hash != window.stable_hash():
            raise ValueError("observatory_artifact window_hash must match stable_hash")
        if window.lineage_chain != expected_nested_lineage:
            raise ValueError(
                "observatory_artifact window lineage_chain must match source fields"
            )
        if len(window.observation_ids) != 2:
            raise ValueError(
                f"window {window.window_id!r} must reference exactly 2 observation IDs; "
                f"got {len(window.observation_ids)}"
            )
        missing = tuple(oid for oid in window.observation_ids if oid not in observation_ids)
        if missing:
            raise ValueError(
                f"window {window.window_id!r} references unknown observation IDs: {missing!r}"
            )


def _validate_direct_lineage(
    observatory_artifact: AtomicSignalTransductionObservatoryResult,
    *,
    replay_certification_artifact: CrossModalReplayCertificationResult | None,
    rf_artifact: RFEqualizationResult | None,
    satellite_artifact: SatelliteBaselineResult | None,
    telecom_artifact: TelecomRecoveryResult | None,
) -> None:
    if replay_certification_artifact is not None:
        if not isinstance(replay_certification_artifact, CrossModalReplayCertificationResult):
            raise ValueError("replay_certification_artifact must be a CrossModalReplayCertificationResult")
        if replay_certification_artifact.stable_hash() != replay_certification_artifact.replay_certification_hash:
            raise ValueError("replay_certification_artifact replay_certification_hash must match stable_hash")
        if replay_certification_artifact.replay_certification_hash != observatory_artifact.source_replay_certification_hash:
            raise ValueError("direct lineage mismatch: replay_certification_hash")

    if rf_artifact is not None:
        if not isinstance(rf_artifact, RFEqualizationResult):
            raise ValueError("rf_artifact must be a RFEqualizationResult")
        if rf_artifact.stable_hash() != rf_artifact.rf_equalization_hash:
            raise ValueError("rf_artifact rf_equalization_hash must match stable_hash")
        if rf_artifact.rf_equalization_hash != observatory_artifact.source_rf_equalization_hash:
            raise ValueError("direct lineage mismatch: rf_equalization_hash")

    if satellite_artifact is not None:
        if not isinstance(satellite_artifact, SatelliteBaselineResult):
            raise ValueError("satellite_artifact must be a SatelliteBaselineResult")
        if satellite_artifact.stable_hash() != satellite_artifact.satellite_baseline_hash:
            raise ValueError("satellite_artifact satellite_baseline_hash must match stable_hash")
        if satellite_artifact.satellite_baseline_hash != observatory_artifact.source_satellite_baseline_hash:
            raise ValueError("direct lineage mismatch: satellite_baseline_hash")

    if telecom_artifact is not None:
        if not isinstance(telecom_artifact, TelecomRecoveryResult):
            raise ValueError("telecom_artifact must be a TelecomRecoveryResult")
        if telecom_artifact.stable_hash() != telecom_artifact.telecom_recovery_hash:
            raise ValueError("telecom_artifact telecom_recovery_hash must match stable_hash")
        if telecom_artifact.telecom_recovery_hash != observatory_artifact.source_telecom_recovery_hash:
            raise ValueError("direct lineage mismatch: telecom_recovery_hash")


@dataclass(frozen=True)
class HypothesisNode:
    node_id: str
    node_index: int
    node_kind: str
    source_artifact_id: str
    lineage_chain: tuple[str, ...]
    hypothesis_confidence: float
    causal_support_score: float
    observability_support_score: float
    node_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "node_id": self.node_id,
            "node_index": self.node_index,
            "node_kind": self.node_kind,
            "source_artifact_id": self.source_artifact_id,
            "lineage_chain": self.lineage_chain,
            "hypothesis_confidence": self.hypothesis_confidence,
            "causal_support_score": self.causal_support_score,
            "observability_support_score": self.observability_support_score,
            "node_hash": self.node_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("node_id")
        payload.pop("node_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class HypothesisEdge:
    edge_id: str
    edge_index: int
    from_node_id: str
    to_node_id: str
    edge_kind: str
    lineage_chain: tuple[str, ...]
    causal_alignment_score: float
    edge_integrity_score: float
    edge_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "edge_id": self.edge_id,
            "edge_index": self.edge_index,
            "from_node_id": self.from_node_id,
            "to_node_id": self.to_node_id,
            "edge_kind": self.edge_kind,
            "lineage_chain": self.lineage_chain,
            "causal_alignment_score": self.causal_alignment_score,
            "edge_integrity_score": self.edge_integrity_score,
            "edge_hash": self.edge_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("edge_id")
        payload.pop("edge_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class HypothesisLattice:
    lattice_id: str
    lattice_profile: str
    lineage_chain: tuple[str, ...]
    nodes: tuple[HypothesisNode, ...]
    node_count: int
    edges: tuple[HypothesisEdge, ...]
    edge_count: int
    lattice_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "lattice_id": self.lattice_id,
            "lattice_profile": self.lattice_profile,
            "lineage_chain": self.lineage_chain,
            "nodes": tuple(node.to_dict() for node in self.nodes),
            "node_count": self.node_count,
            "edges": tuple(edge.to_dict() for edge in self.edges),
            "edge_count": self.edge_count,
            "lattice_hash": self.lattice_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("lattice_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class HypothesisLatticeResult:
    hypothesis_lattice_version: int
    source_feature_schema_hash: str
    source_spectral_reasoning_hash: str
    source_copper_channel_battery_hash: str
    source_telecom_recovery_hash: str
    source_satellite_baseline_hash: str
    source_rf_equalization_hash: str
    source_replay_certification_hash: str
    source_atomic_signal_observatory_hash: str
    hypothesis_lattice_id: str
    lattice_profile: str
    lattice: HypothesisLattice
    node_consistency_score: float
    edge_integrity_score: float
    lineage_reasoning_score: float
    causal_alignment_score: float
    overall_lattice_score: float
    law_invariants: tuple[str, ...]
    hypothesis_lattice_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "hypothesis_lattice_version": self.hypothesis_lattice_version,
            "source_feature_schema_hash": self.source_feature_schema_hash,
            "source_spectral_reasoning_hash": self.source_spectral_reasoning_hash,
            "source_copper_channel_battery_hash": self.source_copper_channel_battery_hash,
            "source_telecom_recovery_hash": self.source_telecom_recovery_hash,
            "source_satellite_baseline_hash": self.source_satellite_baseline_hash,
            "source_rf_equalization_hash": self.source_rf_equalization_hash,
            "source_replay_certification_hash": self.source_replay_certification_hash,
            "source_atomic_signal_observatory_hash": self.source_atomic_signal_observatory_hash,
            "hypothesis_lattice_id": self.hypothesis_lattice_id,
            "lattice_profile": self.lattice_profile,
            "lattice": self.lattice.to_dict(),
            "node_consistency_score": self.node_consistency_score,
            "edge_integrity_score": self.edge_integrity_score,
            "lineage_reasoning_score": self.lineage_reasoning_score,
            "causal_alignment_score": self.causal_alignment_score,
            "overall_lattice_score": self.overall_lattice_score,
            "law_invariants": self.law_invariants,
            "hypothesis_lattice_hash": self.hypothesis_lattice_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("hypothesis_lattice_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class HypothesisLatticeReceipt:
    hypothesis_lattice_version: int
    source_feature_schema_hash: str
    source_spectral_reasoning_hash: str
    source_copper_channel_battery_hash: str
    source_telecom_recovery_hash: str
    source_satellite_baseline_hash: str
    source_rf_equalization_hash: str
    source_replay_certification_hash: str
    source_atomic_signal_observatory_hash: str
    hypothesis_lattice_id: str
    lattice_profile: str
    node_count: int
    edge_count: int
    overall_lattice_score: float
    hypothesis_lattice_hash: str
    receipt_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "hypothesis_lattice_version": self.hypothesis_lattice_version,
            "source_feature_schema_hash": self.source_feature_schema_hash,
            "source_spectral_reasoning_hash": self.source_spectral_reasoning_hash,
            "source_copper_channel_battery_hash": self.source_copper_channel_battery_hash,
            "source_telecom_recovery_hash": self.source_telecom_recovery_hash,
            "source_satellite_baseline_hash": self.source_satellite_baseline_hash,
            "source_rf_equalization_hash": self.source_rf_equalization_hash,
            "source_replay_certification_hash": self.source_replay_certification_hash,
            "source_atomic_signal_observatory_hash": self.source_atomic_signal_observatory_hash,
            "hypothesis_lattice_id": self.hypothesis_lattice_id,
            "lattice_profile": self.lattice_profile,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "overall_lattice_score": self.overall_lattice_score,
            "hypothesis_lattice_hash": self.hypothesis_lattice_hash,
            "receipt_hash": self.receipt_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("receipt_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


def _lineage_chain(observatory_artifact: AtomicSignalTransductionObservatoryResult) -> tuple[str, ...]:
    return (
        observatory_artifact.source_feature_schema_hash,
        observatory_artifact.source_spectral_reasoning_hash,
        observatory_artifact.source_copper_channel_battery_hash,
        observatory_artifact.source_telecom_recovery_hash,
        observatory_artifact.source_satellite_baseline_hash,
        observatory_artifact.source_rf_equalization_hash,
        observatory_artifact.source_replay_certification_hash,
        observatory_artifact.atomic_signal_observatory_hash,
    )


def _build_nodes(
    observatory_artifact: AtomicSignalTransductionObservatoryResult,
    *,
    lineage_chain: tuple[str, ...],
) -> tuple[HypothesisNode, ...]:
    nodes: list[HypothesisNode] = []

    for observation in observatory_artifact.observations:
        node = HypothesisNode(
            node_id="",
            node_index=observation.observation_index,
            node_kind="observation_hypothesis",
            source_artifact_id=observation.observation_id,
            lineage_chain=lineage_chain,
            hypothesis_confidence=_mean(
                (
                    observation.transduction_integrity_score,
                    observation.observation_alignment_score,
                    observation.replay_visibility_score,
                ),
                default=1.0,
            ),
            causal_support_score=observation.observation_alignment_score,
            observability_support_score=observation.replay_visibility_score,
            node_hash="",
        )
        node_hash = _finalize_identity(node, "node_hash").node_hash
        nodes.append(replace(node, node_id=node_hash, node_hash=node_hash))

    base_index = len(observatory_artifact.observations)
    for window in observatory_artifact.windows:
        node = HypothesisNode(
            node_id="",
            node_index=base_index + window.window_index,
            node_kind="transition_hypothesis",
            source_artifact_id=window.window_id,
            lineage_chain=lineage_chain,
            hypothesis_confidence=_mean(
                (
                    window.transduction_integrity_score,
                    window.window_consistency_score,
                    window.replay_visibility_score,
                ),
                default=1.0,
            ),
            causal_support_score=window.window_consistency_score,
            observability_support_score=window.replay_visibility_score,
            node_hash="",
        )
        node_hash = _finalize_identity(node, "node_hash").node_hash
        nodes.append(replace(node, node_id=node_hash, node_hash=node_hash))

    return tuple(sorted(nodes, key=lambda n: (n.node_index, n.node_kind, n.node_id)))


def _build_edges(
    observatory_artifact: AtomicSignalTransductionObservatoryResult,
    node_by_source_id: Mapping[str, HypothesisNode],
    *,
    lineage_chain: tuple[str, ...],
) -> tuple[HypothesisEdge, ...]:
    edges: list[HypothesisEdge] = []
    for window in observatory_artifact.windows:
        window_node = node_by_source_id[window.window_id]
        source_observation_id, target_observation_id = window.observation_ids
        source_node = node_by_source_id[source_observation_id]
        target_node = node_by_source_id[target_observation_id]

        source_edge = HypothesisEdge(
            edge_id="",
            edge_index=len(edges),
            from_node_id=source_node.node_id,
            to_node_id=window_node.node_id,
            edge_kind="observation_to_transition",
            lineage_chain=lineage_chain,
            causal_alignment_score=_mean(
                (
                    source_node.causal_support_score,
                    window_node.causal_support_score,
                ),
                default=1.0,
            ),
            edge_integrity_score=_mean(
                (
                    source_node.hypothesis_confidence,
                    window_node.hypothesis_confidence,
                ),
                default=1.0,
            ),
            edge_hash="",
        )
        finalized_source = _finalize_identity(source_edge, "edge_hash")
        edges.append(replace(finalized_source, edge_id=finalized_source.edge_hash))

        target_edge = HypothesisEdge(
            edge_id="",
            edge_index=len(edges),
            from_node_id=window_node.node_id,
            to_node_id=target_node.node_id,
            edge_kind="transition_to_observation",
            lineage_chain=lineage_chain,
            causal_alignment_score=_mean(
                (
                    window_node.causal_support_score,
                    target_node.causal_support_score,
                ),
                default=1.0,
            ),
            edge_integrity_score=_mean(
                (
                    window_node.hypothesis_confidence,
                    target_node.hypothesis_confidence,
                ),
                default=1.0,
            ),
            edge_hash="",
        )
        finalized_target = _finalize_identity(target_edge, "edge_hash")
        edges.append(replace(finalized_target, edge_id=finalized_target.edge_hash))

    return tuple(
        sorted(
            edges,
            key=lambda e: (e.edge_index, e.from_node_id, e.to_node_id, e.edge_id),
        )
    )


def _validate_lattice_connectivity(nodes: tuple[HypothesisNode, ...], edges: tuple[HypothesisEdge, ...]) -> None:
    adjacency: dict[str, set[str]] = {node.node_id: set() for node in nodes}
    for edge in edges:
        if edge.from_node_id not in adjacency or edge.to_node_id not in adjacency:
            raise ValueError("lattice edges must reference known nodes")
        adjacency[edge.from_node_id].add(edge.to_node_id)
        adjacency[edge.to_node_id].add(edge.from_node_id)

    if not adjacency:
        raise ValueError("lattice must contain nodes")

    stack = [nodes[0].node_id]
    visited: set[str] = set()
    while stack:
        node_id = stack.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        stack.extend(sorted(adjacency[node_id] - visited))

    if len(visited) != len(nodes):
        raise ValueError("disconnected lattice is not allowed")


def _validate_lattice(lattice: HypothesisLattice, expected_lineage_chain: tuple[str, ...]) -> None:
    if lattice.lattice_hash != lattice.stable_hash():
        raise ValueError("lattice_hash must match stable_hash")
    if lattice.node_count != len(lattice.nodes):
        raise ValueError("lattice node_count must match len(nodes)")
    if lattice.edge_count != len(lattice.edges):
        raise ValueError("lattice edge_count must match len(edges)")
    if lattice.lineage_chain != expected_lineage_chain:
        raise ValueError("lattice lineage_chain must match expected lineage")

    expected_node_order = tuple(
        sorted(lattice.nodes, key=lambda n: (n.node_index, n.node_kind, n.node_id))
    )
    if lattice.nodes != expected_node_order:
        raise ValueError("lattice nodes must be in canonical deterministic order")

    expected_edge_order = tuple(
        sorted(lattice.edges, key=lambda e: (e.edge_index, e.from_node_id, e.to_node_id, e.edge_id))
    )
    if lattice.edges != expected_edge_order:
        raise ValueError("lattice edges must be in canonical deterministic order")

    for node in lattice.nodes:
        if node.node_hash != node.stable_hash():
            raise ValueError("lattice node_hash must match stable_hash")
        if node.node_id != node.node_hash:
            raise ValueError("lattice node_id must equal node_hash")
        if node.lineage_chain != expected_lineage_chain:
            raise ValueError("lattice node lineage_chain mismatch")

    for edge in lattice.edges:
        if edge.edge_hash != edge.stable_hash():
            raise ValueError("lattice edge_hash must match stable_hash")
        if edge.edge_id != edge.edge_hash:
            raise ValueError("lattice edge_id must equal edge_hash")
        if edge.lineage_chain != expected_lineage_chain:
            raise ValueError("lattice edge lineage_chain mismatch")

    _validate_lattice_connectivity(lattice.nodes, lattice.edges)


def build_hypothesis_lattice(
    observatory_artifact: AtomicSignalTransductionObservatoryResult,
    *,
    lattice_profile: str = "full_chain_hypothesis_lattice",
    score_fixture: tuple[float, ...] | None = None,
    replay_certification_artifact: CrossModalReplayCertificationResult | None = None,
    rf_artifact: RFEqualizationResult | None = None,
    satellite_artifact: SatelliteBaselineResult | None = None,
    telecom_artifact: TelecomRecoveryResult | None = None,
) -> HypothesisLatticeResult:
    """Build deterministic replay-safe hypothesis lattice artifacts."""

    _validate_profile(lattice_profile)
    _validate_optional_fixture_payload(score_fixture)
    _validate_observatory_artifact(observatory_artifact)
    _validate_direct_lineage(
        observatory_artifact,
        replay_certification_artifact=replay_certification_artifact,
        rf_artifact=rf_artifact,
        satellite_artifact=satellite_artifact,
        telecom_artifact=telecom_artifact,
    )

    lineage_chain = _lineage_chain(observatory_artifact)
    nodes = _build_nodes(observatory_artifact, lineage_chain=lineage_chain)
    node_by_source_id = {node.source_artifact_id: node for node in nodes}
    edges = _build_edges(
        observatory_artifact,
        node_by_source_id,
        lineage_chain=lineage_chain,
    )
    _validate_lattice_connectivity(nodes, edges)

    lattice = HypothesisLattice(
        lattice_id=_sha256_hex(
            {
                "source_atomic_signal_observatory_hash": observatory_artifact.atomic_signal_observatory_hash,
                "hypothesis_lattice_version": _HYPOTHESIS_LATTICE_VERSION,
                "lattice_profile": lattice_profile,
            }
        ),
        lattice_profile=lattice_profile,
        lineage_chain=lineage_chain,
        nodes=nodes,
        node_count=len(nodes),
        edges=edges,
        edge_count=len(edges),
        lattice_hash="",
    )
    lattice = _finalize_identity(lattice, "lattice_hash")
    _validate_lattice(lattice, lineage_chain)

    node_consistency_score = _mean(tuple(node.hypothesis_confidence for node in nodes), default=1.0)
    edge_integrity_score = _mean(tuple(edge.edge_integrity_score for edge in edges), default=1.0)
    lineage_reasoning_score = _mean(
        (
            1.0 if lattice.lineage_chain == lineage_chain else 0.0,
            1.0 if all(node.lineage_chain == lineage_chain for node in nodes) else 0.0,
            1.0 if all(edge.lineage_chain == lineage_chain for edge in edges) else 0.0,
        ),
        default=1.0,
    )
    causal_alignment_score = _mean(tuple(edge.causal_alignment_score for edge in edges), default=1.0)
    overall_lattice_score = _mean(
        (
            node_consistency_score,
            edge_integrity_score,
            lineage_reasoning_score,
            causal_alignment_score,
        ),
        default=1.0,
    )

    for score_name, score in (
        ("node_consistency_score", node_consistency_score),
        ("edge_integrity_score", edge_integrity_score),
        ("lineage_reasoning_score", lineage_reasoning_score),
        ("causal_alignment_score", causal_alignment_score),
        ("overall_lattice_score", overall_lattice_score),
    ):
        _validate_unit_interval(score, score_name)

    result = HypothesisLatticeResult(
        hypothesis_lattice_version=_HYPOTHESIS_LATTICE_VERSION,
        source_feature_schema_hash=observatory_artifact.source_feature_schema_hash,
        source_spectral_reasoning_hash=observatory_artifact.source_spectral_reasoning_hash,
        source_copper_channel_battery_hash=observatory_artifact.source_copper_channel_battery_hash,
        source_telecom_recovery_hash=observatory_artifact.source_telecom_recovery_hash,
        source_satellite_baseline_hash=observatory_artifact.source_satellite_baseline_hash,
        source_rf_equalization_hash=observatory_artifact.source_rf_equalization_hash,
        source_replay_certification_hash=observatory_artifact.source_replay_certification_hash,
        source_atomic_signal_observatory_hash=observatory_artifact.atomic_signal_observatory_hash,
        hypothesis_lattice_id=lattice.lattice_id,
        lattice_profile=lattice_profile,
        lattice=lattice,
        node_consistency_score=node_consistency_score,
        edge_integrity_score=edge_integrity_score,
        lineage_reasoning_score=lineage_reasoning_score,
        causal_alignment_score=causal_alignment_score,
        overall_lattice_score=overall_lattice_score,
        law_invariants=(
            HYPOTHESIS_LATTICE_LAW,
            DETERMINISTIC_LATTICE_ORDERING_RULE,
            REPLAY_SAFE_LATTICE_IDENTITY_RULE,
            BOUNDED_LATTICE_SCORE_RULE,
        ),
        hypothesis_lattice_hash="",
    )
    return _finalize_identity(result, "hypothesis_lattice_hash")


def export_hypothesis_lattice_bytes(artifact: HypothesisLatticeResult) -> bytes:
    if not isinstance(artifact, HypothesisLatticeResult):
        raise ValueError("artifact must be a HypothesisLatticeResult")
    return artifact.to_canonical_bytes()


def generate_hypothesis_lattice_receipt(
    artifact: HypothesisLatticeResult,
) -> HypothesisLatticeReceipt:
    if not isinstance(artifact, HypothesisLatticeResult):
        raise ValueError("artifact must be a HypothesisLatticeResult")
    if artifact.stable_hash() != artifact.hypothesis_lattice_hash:
        raise ValueError("artifact hypothesis_lattice_hash must match stable_hash")

    receipt = HypothesisLatticeReceipt(
        hypothesis_lattice_version=artifact.hypothesis_lattice_version,
        source_feature_schema_hash=artifact.source_feature_schema_hash,
        source_spectral_reasoning_hash=artifact.source_spectral_reasoning_hash,
        source_copper_channel_battery_hash=artifact.source_copper_channel_battery_hash,
        source_telecom_recovery_hash=artifact.source_telecom_recovery_hash,
        source_satellite_baseline_hash=artifact.source_satellite_baseline_hash,
        source_rf_equalization_hash=artifact.source_rf_equalization_hash,
        source_replay_certification_hash=artifact.source_replay_certification_hash,
        source_atomic_signal_observatory_hash=artifact.source_atomic_signal_observatory_hash,
        hypothesis_lattice_id=artifact.hypothesis_lattice_id,
        lattice_profile=artifact.lattice_profile,
        node_count=artifact.lattice.node_count,
        edge_count=artifact.lattice.edge_count,
        overall_lattice_score=artifact.overall_lattice_score,
        hypothesis_lattice_hash=artifact.hypothesis_lattice_hash,
        receipt_hash="",
    )
    return _finalize_identity(receipt, "receipt_hash")


__all__ = [
    "BOUNDED_LATTICE_SCORE_RULE",
    "DETERMINISTIC_LATTICE_ORDERING_RULE",
    "HYPOTHESIS_LATTICE_LAW",
    "REPLAY_SAFE_LATTICE_IDENTITY_RULE",
    "HypothesisEdge",
    "HypothesisLattice",
    "HypothesisLatticeReceipt",
    "HypothesisLatticeResult",
    "HypothesisNode",
    "build_hypothesis_lattice",
    "export_hypothesis_lattice_bytes",
    "generate_hypothesis_lattice_receipt",
]
