"""v137.13.1 — Morphology Transition Kernel.

Deterministic Layer-4 transition analysis over synthetic geometry trajectories.
Builds ordered morphology states, transition edges, bounded metrics, and
receipt-chain artifacts with canonical export semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import json
import math
from typing import Any, Mapping

from qec.analysis.synthetic_signal_geometry_kernel import (
    SCHEMA_VERSION as GEOMETRY_SCHEMA_VERSION,
    SignalGeometryTrajectory,
)

SCHEMA_VERSION = "v137.13.1"
_DECIMAL_PLACES = Decimal("0.000000000001")

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

_STATE_PRIORITY = (
    "stable",
    "resonant",
    "oscillatory",
    "diverging",
    "converging",
    "transitional",
)


# ---------------------------------------------------------------------------
# Canonical + deterministic helpers
# ---------------------------------------------------------------------------


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
        return {k: _canonicalize_json(value[k]) for k in sorted(keys)}
    raise ValueError(f"unsupported canonical payload type: {type(value)!r}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonicalize_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _canonical_bytes(value: Any) -> bytes:
    return _canonical_json(value).encode("utf-8")


def _sha256_hex(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def _quantize(value: float, *, field_name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return float(Decimal(str(value)).quantize(_DECIMAL_PLACES, rounding=ROUND_HALF_EVEN))


def _quantize_unit(value: float, *, field_name: str) -> float:
    q = _quantize(value, field_name=field_name)
    if q < 0.0 or q > 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]")
    return q


def _quantized_str(value: float, *, field_name: str) -> str:
    q = Decimal(str(_quantize(value, field_name=field_name))).quantize(
        _DECIMAL_PLACES,
        rounding=ROUND_HALF_EVEN,
    )
    return str(q)


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MorphologyTransitionConfig:
    schema_version: str = SCHEMA_VERSION
    kernel_version: str = SCHEMA_VERSION
    stable_continuity_threshold: float = 0.080000000000
    resonant_continuity_threshold: float = 0.200000000000
    diverging_delta_threshold: float = 0.050000000000
    converging_delta_threshold: float = 0.050000000000

    def __post_init__(self) -> None:
        for field_name in (
            "stable_continuity_threshold",
            "resonant_continuity_threshold",
            "diverging_delta_threshold",
            "converging_delta_threshold",
        ):
            value = getattr(self, field_name)
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{field_name} must be in [0, 1]")
        if self.resonant_continuity_threshold < self.stable_continuity_threshold:
            raise ValueError("resonant_continuity_threshold must be >= stable_continuity_threshold")

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "kernel_version": self.kernel_version,
            "stable_continuity_threshold": _quantized_str(
                self.stable_continuity_threshold, field_name="stable_continuity_threshold"
            ),
            "resonant_continuity_threshold": _quantized_str(
                self.resonant_continuity_threshold, field_name="resonant_continuity_threshold"
            ),
            "diverging_delta_threshold": _quantized_str(
                self.diverging_delta_threshold, field_name="diverging_delta_threshold"
            ),
            "converging_delta_threshold": _quantized_str(
                self.converging_delta_threshold, field_name="converging_delta_threshold"
            ),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_sha256(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class MorphologyState:
    state_id: str
    source_index: int
    state_label: str
    activity_centroid: float
    spike_density_coordinate: float
    continuity_coordinate: float
    state_score: float

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "state_id": self.state_id,
            "source_index": self.source_index,
            "state_label": self.state_label,
            "activity_centroid": _quantized_str(self.activity_centroid, field_name="activity_centroid"),
            "spike_density_coordinate": _quantized_str(
                self.spike_density_coordinate, field_name="spike_density_coordinate"
            ),
            "continuity_coordinate": _quantized_str(
                self.continuity_coordinate, field_name="continuity_coordinate"
            ),
            "state_score": _quantized_str(self.state_score, field_name="state_score"),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_sha256(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class MorphologyTransitionEdge:
    source_index: int
    target_index: int
    source_state: str
    target_state: str
    transition_magnitude: float
    stability_delta: float
    continuity_delta: float

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "source_index": self.source_index,
            "target_index": self.target_index,
            "source_state": self.source_state,
            "target_state": self.target_state,
            "transition_magnitude": _quantized_str(
                self.transition_magnitude, field_name="transition_magnitude"
            ),
            "stability_delta": _quantized_str(self.stability_delta, field_name="stability_delta"),
            "continuity_delta": _quantized_str(self.continuity_delta, field_name="continuity_delta"),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_sha256(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class MorphologyTransitionPath:
    config: MorphologyTransitionConfig
    input_trajectory_hash: str
    states: tuple[MorphologyState, ...]
    edges: tuple[MorphologyTransitionEdge, ...]
    stable_hash: str
    schema_version: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "config": self.config.to_dict(),
            "input_trajectory_hash": self.input_trajectory_hash,
            "states": tuple(state.to_dict() for state in self.states),
            "edges": tuple(edge.to_dict() for edge in self.edges),
            "stable_hash": self.stable_hash,
            "schema_version": self.schema_version,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("stable_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_sha256(self) -> str:
        return self.stable_hash


@dataclass(frozen=True)
class MorphologyTransitionResult:
    path: MorphologyTransitionPath
    transition_integrity_score: float
    phase_stability_score: float
    morphology_consistency_score: float
    transition_continuity_score: float
    stable_hash: str
    schema_version: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "path": self.path.to_dict(),
            "transition_integrity_score": _quantized_str(
                self.transition_integrity_score, field_name="transition_integrity_score"
            ),
            "phase_stability_score": _quantized_str(
                self.phase_stability_score, field_name="phase_stability_score"
            ),
            "morphology_consistency_score": _quantized_str(
                self.morphology_consistency_score, field_name="morphology_consistency_score"
            ),
            "transition_continuity_score": _quantized_str(
                self.transition_continuity_score, field_name="transition_continuity_score"
            ),
            "stable_hash": self.stable_hash,
            "schema_version": self.schema_version,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("stable_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_sha256(self) -> str:
        return self.stable_hash


@dataclass(frozen=True)
class MorphologyTransitionReceipt:
    receipt_hash: str
    kernel_version: str
    schema_version: str
    input_trajectory_hash: str
    output_stable_hash: str
    state_count: int
    edge_count: int
    receipt_chain: tuple[str, ...]
    transition_integrity_score: float
    phase_stability_score: float
    morphology_consistency_score: float
    transition_continuity_score: float
    validation_passed: bool

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "receipt_hash": self.receipt_hash,
            "kernel_version": self.kernel_version,
            "schema_version": self.schema_version,
            "input_trajectory_hash": self.input_trajectory_hash,
            "output_stable_hash": self.output_stable_hash,
            "state_count": self.state_count,
            "edge_count": self.edge_count,
            "receipt_chain": self.receipt_chain,
            "transition_integrity_score": _quantized_str(
                self.transition_integrity_score, field_name="transition_integrity_score"
            ),
            "phase_stability_score": _quantized_str(
                self.phase_stability_score, field_name="phase_stability_score"
            ),
            "morphology_consistency_score": _quantized_str(
                self.morphology_consistency_score, field_name="morphology_consistency_score"
            ),
            "transition_continuity_score": _quantized_str(
                self.transition_continuity_score, field_name="transition_continuity_score"
            ),
            "validation_passed": self.validation_passed,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("receipt_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_sha256(self) -> str:
        return self.receipt_hash


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_config(config: MorphologyTransitionConfig) -> MorphologyTransitionConfig:
    if config.schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {config.schema_version}")
    if config.kernel_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported kernel version: {config.kernel_version}")
    return config


def _validate_geometry_trajectory(trajectory: SignalGeometryTrajectory) -> SignalGeometryTrajectory:
    if trajectory.schema_version != GEOMETRY_SCHEMA_VERSION:
        raise ValueError("schema mismatch with geometry trajectory")
    if trajectory.node_count <= 0:
        raise ValueError("empty paths are not allowed")
    if len(trajectory.nodes) != trajectory.node_count:
        raise ValueError("invalid ordering: node_count mismatch")
    if len(trajectory.edges) != max(0, trajectory.node_count - 1):
        raise ValueError("invalid ordering: edge count mismatch")
    previous = -1
    for node in trajectory.nodes:
        if node.frame_index <= previous:
            raise ValueError("invalid ordering: nodes must be strictly ordered")
        previous = node.frame_index
        for value in (
            node.activity_centroid,
            node.spike_density_coordinate,
            node.continuity_coordinate,
        ):
            if not math.isfinite(value):
                raise ValueError("non-finite metrics in trajectory")
    if trajectory.stable_hash != _sha256_hex(trajectory.to_hash_payload_dict()):
        raise ValueError("broken lineage: trajectory stable_hash mismatch")
    return trajectory


def _validate_path(path: MorphologyTransitionPath) -> None:
    if len(path.states) == 0:
        raise ValueError("empty paths are not allowed")
    state_ids = tuple(state.state_id for state in path.states)
    if len(set(state_ids)) != len(state_ids):
        raise ValueError("duplicate state ids detected")
    prev_index = -1
    for state in path.states:
        if state.source_index <= prev_index:
            raise ValueError("invalid ordering: state indices must be strictly increasing")
        prev_index = state.source_index
        if state.state_label not in _STATE_PRIORITY:
            raise ValueError("schema mismatch: invalid state label")
        for value in (
            state.activity_centroid,
            state.spike_density_coordinate,
            state.continuity_coordinate,
            state.state_score,
        ):
            if not math.isfinite(value):
                raise ValueError("non-finite metrics in state")

    if len(path.edges) != max(0, len(path.states) - 1):
        raise ValueError("invalid ordering: edge cardinality mismatch")

    for i, edge in enumerate(path.edges):
        source = path.states[i]
        target = path.states[i + 1]
        if edge.source_index != source.source_index or edge.target_index != target.source_index:
            raise ValueError("invalid ordering: edge indices do not align")
        if edge.source_state != source.state_label or edge.target_state != target.state_label:
            raise ValueError("schema mismatch: edge/state label mismatch")
        for value in (edge.transition_magnitude, edge.stability_delta, edge.continuity_delta):
            if not math.isfinite(value):
                raise ValueError("non-finite metrics in edges")

    if path.stable_hash != _sha256_hex(path.to_hash_payload_dict()):
        raise ValueError("broken lineage: path stable_hash mismatch")


# ---------------------------------------------------------------------------
# State detection
# ---------------------------------------------------------------------------


def _state_score_for_label(
    *,
    label: str,
    continuity: float,
    signed_delta: float,
    direction_flip: bool,
    config: MorphologyTransitionConfig,
) -> float:
    abs_delta = abs(signed_delta)
    if label == "stable":
        return _quantize_unit(
            1.0 - ((continuity + abs_delta) * 0.5),
            field_name="state_score",
        )
    if label == "resonant":
        closeness = 1.0 - (continuity / max(config.resonant_continuity_threshold, 1e-12))
        return _quantize_unit(closeness, field_name="state_score")
    if label == "oscillatory":
        return 1.0 if direction_flip else 0.0
    if label == "diverging":
        return _quantize_unit(abs_delta, field_name="state_score")
    if label == "converging":
        return _quantize_unit(abs_delta, field_name="state_score")
    return _quantize_unit(1.0 - continuity, field_name="state_score")


def detect_morphology_states(
    trajectory: SignalGeometryTrajectory,
    config: MorphologyTransitionConfig | None = None,
) -> tuple[MorphologyState, ...]:
    """Convert ordered geometry nodes into deterministic morphology states."""
    effective_config = _validate_config(config or MorphologyTransitionConfig())
    valid_trajectory = _validate_geometry_trajectory(trajectory)

    nodes = valid_trajectory.nodes
    states: list[MorphologyState] = []

    for idx, node in enumerate(nodes):
        prev_centroid = nodes[idx - 1].activity_centroid if idx > 0 else node.activity_centroid
        next_centroid = nodes[idx + 1].activity_centroid if idx < len(nodes) - 1 else node.activity_centroid
        signed_delta = _quantize(
            node.activity_centroid - prev_centroid,
            field_name="signed_delta",
        )
        next_signed_delta = _quantize(
            next_centroid - node.activity_centroid,
            field_name="next_signed_delta",
        )
        direction_flip = (signed_delta * next_signed_delta) < 0.0

        continuity = _quantize_unit(node.continuity_coordinate, field_name="continuity_coordinate")
        abs_delta = abs(signed_delta)

        rules: dict[str, bool] = {
            "stable": continuity <= effective_config.stable_continuity_threshold
            and abs_delta <= effective_config.diverging_delta_threshold,
            "resonant": continuity <= effective_config.resonant_continuity_threshold
            and abs_delta <= effective_config.diverging_delta_threshold,
            "oscillatory": direction_flip,
            "diverging": signed_delta >= effective_config.diverging_delta_threshold,
            "converging": signed_delta <= -effective_config.converging_delta_threshold,
            "transitional": True,
        }

        chosen_label = "transitional"
        for label in _STATE_PRIORITY:
            if rules[label]:
                chosen_label = label
                break

        score = _state_score_for_label(
            label=chosen_label,
            continuity=continuity,
            signed_delta=signed_delta,
            direction_flip=direction_flip,
            config=effective_config,
        )

        seed = {
            "trajectory_hash": trajectory.stable_hash,
            "source_index": node.frame_index,
            "state_label": chosen_label,
            "position": idx,
        }

        states.append(
            MorphologyState(
                state_id=_sha256_hex(seed),
                source_index=node.frame_index,
                state_label=chosen_label,
                activity_centroid=_quantize_unit(node.activity_centroid, field_name="activity_centroid"),
                spike_density_coordinate=_quantize_unit(
                    node.spike_density_coordinate,
                    field_name="spike_density_coordinate",
                ),
                continuity_coordinate=continuity,
                state_score=score,
            )
        )

    return tuple(states)


# ---------------------------------------------------------------------------
# Transition path + metrics
# ---------------------------------------------------------------------------


def build_transition_path(
    trajectory: SignalGeometryTrajectory,
    config: MorphologyTransitionConfig | None = None,
) -> MorphologyTransitionPath:
    """Build deterministic morphology transition path from a geometry trajectory."""
    effective_config = _validate_config(config or MorphologyTransitionConfig())
    valid_trajectory = _validate_geometry_trajectory(trajectory)
    states = detect_morphology_states(valid_trajectory, effective_config)

    edges: list[MorphologyTransitionEdge] = []
    for i in range(len(states) - 1):
        source = states[i]
        target = states[i + 1]
        stability_delta = _quantize(target.state_score - source.state_score, field_name="stability_delta")
        continuity_delta = _quantize(
            target.continuity_coordinate - source.continuity_coordinate,
            field_name="continuity_delta",
        )
        magnitude = _quantize_unit(
            abs(target.activity_centroid - source.activity_centroid),
            field_name="transition_magnitude",
        )
        edges.append(
            MorphologyTransitionEdge(
                source_index=source.source_index,
                target_index=target.source_index,
                source_state=source.state_label,
                target_state=target.state_label,
                transition_magnitude=magnitude,
                stability_delta=stability_delta,
                continuity_delta=continuity_delta,
            )
        )

    proto = MorphologyTransitionPath(
        config=effective_config,
        input_trajectory_hash=valid_trajectory.stable_hash,
        states=states,
        edges=tuple(edges),
        stable_hash="",
        schema_version=effective_config.schema_version,
    )
    path = MorphologyTransitionPath(
        config=proto.config,
        input_trajectory_hash=proto.input_trajectory_hash,
        states=proto.states,
        edges=proto.edges,
        stable_hash=_sha256_hex(proto.to_hash_payload_dict()),
        schema_version=proto.schema_version,
    )
    _validate_path(path)
    return path


def _compute_kernel_metrics(path: MorphologyTransitionPath) -> dict[str, float]:
    edges = path.edges
    states = path.states

    if len(edges) == 0:
        integrity = 1.0
        continuity_score = 1.0
    else:
        contiguous = sum(
            1
            for i, edge in enumerate(edges)
            if edge.source_index == states[i].source_index and edge.target_index == states[i + 1].source_index
        )
        integrity = float(contiguous) / float(len(edges))
        mean_abs_cont = sum(abs(edge.continuity_delta) for edge in edges) / float(len(edges))
        continuity_score = 1.0 - _clamp01(mean_abs_cont)

    stable_like = sum(1 for state in states if state.state_label in ("stable", "resonant"))
    phase_stability = float(stable_like) / float(len(states))

    if len(edges) == 0:
        consistency = 1.0
    else:
        changes = sum(1 for edge in edges if edge.source_state != edge.target_state)
        consistency = 1.0 - (float(changes) / float(len(edges)))

    return {
        "transition_integrity_score": _quantize_unit(integrity, field_name="transition_integrity_score"),
        "phase_stability_score": _quantize_unit(phase_stability, field_name="phase_stability_score"),
        "morphology_consistency_score": _quantize_unit(
            consistency,
            field_name="morphology_consistency_score",
        ),
        "transition_continuity_score": _quantize_unit(
            continuity_score,
            field_name="transition_continuity_score",
        ),
    }


def compute_transition_similarity(
    path_a: MorphologyTransitionPath,
    path_b: MorphologyTransitionPath,
) -> dict[str, float]:
    """Compute deterministic bounded similarity between morphology paths."""
    _validate_path(path_a)
    _validate_path(path_b)

    states_a = path_a.states
    states_b = path_b.states
    min_len = min(len(states_a), len(states_b))
    max_len = max(len(states_a), len(states_b))
    if max_len == 0:
        raise ValueError("cannot compare empty paths")

    length_overlap = float(min_len) / float(max_len)
    if min_len == 0:
        label_alignment = 0.0
        centroid_similarity = 0.0
    else:
        label_matches = sum(
            1 for i in range(min_len) if states_a[i].state_label == states_b[i].state_label
        )
        label_alignment = float(label_matches) / float(min_len)

        centroid_diffs = tuple(
            abs(states_a[i].activity_centroid - states_b[i].activity_centroid)
            for i in range(min_len)
        )
        centroid_similarity = 1.0 - _clamp01(sum(centroid_diffs) / float(min_len))

    structural_similarity = _clamp01((0.6 * label_alignment) + (0.4 * centroid_similarity))
    path_overlap = _clamp01(length_overlap * structural_similarity)

    return {
        "length_overlap": _quantize_unit(length_overlap, field_name="length_overlap"),
        "label_alignment": _quantize_unit(label_alignment, field_name="label_alignment"),
        "structural_similarity": _quantize_unit(
            structural_similarity,
            field_name="structural_similarity",
        ),
        "path_overlap": _quantize_unit(path_overlap, field_name="path_overlap"),
    }


def run_morphology_transition_kernel(
    trajectory: SignalGeometryTrajectory,
    config: MorphologyTransitionConfig | None = None,
) -> tuple[MorphologyTransitionResult, MorphologyTransitionReceipt]:
    """Run the full deterministic morphology transition kernel pipeline."""
    effective_config = _validate_config(config or MorphologyTransitionConfig())
    path = build_transition_path(trajectory, effective_config)
    metrics = _compute_kernel_metrics(path)

    result_proto = MorphologyTransitionResult(
        path=path,
        transition_integrity_score=metrics["transition_integrity_score"],
        phase_stability_score=metrics["phase_stability_score"],
        morphology_consistency_score=metrics["morphology_consistency_score"],
        transition_continuity_score=metrics["transition_continuity_score"],
        stable_hash="",
        schema_version=effective_config.schema_version,
    )
    result = MorphologyTransitionResult(
        path=result_proto.path,
        transition_integrity_score=result_proto.transition_integrity_score,
        phase_stability_score=result_proto.phase_stability_score,
        morphology_consistency_score=result_proto.morphology_consistency_score,
        transition_continuity_score=result_proto.transition_continuity_score,
        stable_hash=_sha256_hex(result_proto.to_hash_payload_dict()),
        schema_version=result_proto.schema_version,
    )

    chain_seed = (
        effective_config.stable_sha256(),
        path.stable_hash,
        result.stable_hash,
    )
    chain_tip = _sha256_hex({"receipt_chain_seed": chain_seed})
    receipt_chain = chain_seed + (chain_tip,)

    receipt_proto = MorphologyTransitionReceipt(
        receipt_hash="",
        kernel_version=effective_config.kernel_version,
        schema_version=effective_config.schema_version,
        input_trajectory_hash=path.input_trajectory_hash,
        output_stable_hash=result.stable_hash,
        state_count=len(path.states),
        edge_count=len(path.edges),
        receipt_chain=receipt_chain,
        transition_integrity_score=result.transition_integrity_score,
        phase_stability_score=result.phase_stability_score,
        morphology_consistency_score=result.morphology_consistency_score,
        transition_continuity_score=result.transition_continuity_score,
        validation_passed=True,
    )
    receipt = MorphologyTransitionReceipt(
        receipt_hash=_sha256_hex(receipt_proto.to_hash_payload_dict()),
        kernel_version=receipt_proto.kernel_version,
        schema_version=receipt_proto.schema_version,
        input_trajectory_hash=receipt_proto.input_trajectory_hash,
        output_stable_hash=receipt_proto.output_stable_hash,
        state_count=receipt_proto.state_count,
        edge_count=receipt_proto.edge_count,
        receipt_chain=receipt_proto.receipt_chain,
        transition_integrity_score=receipt_proto.transition_integrity_score,
        phase_stability_score=receipt_proto.phase_stability_score,
        morphology_consistency_score=receipt_proto.morphology_consistency_score,
        transition_continuity_score=receipt_proto.transition_continuity_score,
        validation_passed=receipt_proto.validation_passed,
    )

    if receipt.receipt_chain[0] != effective_config.stable_sha256():
        raise ValueError("broken lineage: receipt chain root mismatch")
    if receipt.receipt_chain[1] != path.stable_hash:
        raise ValueError("broken lineage: receipt chain path mismatch")
    if receipt.receipt_chain[2] != result.stable_hash:
        raise ValueError("broken lineage: receipt chain result mismatch")

    return result, receipt


def build_ascii_transition_summary(result: MorphologyTransitionResult) -> str:
    """Build deterministic ASCII summary for morphology transition results."""
    return "\n".join(
        (
            f"Morphology Transition Kernel — {result.schema_version}",
            f"  States:      {len(result.path.states)}",
            f"  Edges:       {len(result.path.edges)}",
            f"  Integrity:   {_quantized_str(result.transition_integrity_score, field_name='integrity')}",
            f"  Stability:   {_quantized_str(result.phase_stability_score, field_name='stability')}",
            f"  Consistency: {_quantized_str(result.morphology_consistency_score, field_name='consistency')}",
            f"  Continuity:  {_quantized_str(result.transition_continuity_score, field_name='continuity')}",
            f"  Hash:        {result.stable_hash[:16]}...",
        )
    )


__all__ = [
    "SCHEMA_VERSION",
    "MorphologyTransitionConfig",
    "MorphologyState",
    "MorphologyTransitionEdge",
    "MorphologyTransitionPath",
    "MorphologyTransitionResult",
    "MorphologyTransitionReceipt",
    "detect_morphology_states",
    "build_transition_path",
    "compute_transition_similarity",
    "run_morphology_transition_kernel",
    "build_ascii_transition_summary",
]
