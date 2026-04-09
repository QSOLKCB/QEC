"""v137.13.0 — Synthetic Signal Geometry Kernel.

Deterministic Layer-4 geometry kernel for synthetic signal morphology
and state-space abstraction.  Converts deterministic hybrid signal traces
into canonical geometric representations with shape abstraction,
trajectory similarity, and transition geometry.

This is formal signal geometry only — no biological claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import json
import math
import types
from typing import Any, Mapping

from qec.analysis.hybrid_signal_interface import (
    SCHEMA_VERSION as HYBRID_SCHEMA_VERSION,
    HybridSignalTrace,
)

SCHEMA_VERSION = "v137.13.0"
_DECIMAL_PLACES = Decimal("0.000000000001")

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

# ---------------------------------------------------------------------------
# Canonical serialisation helpers (deterministic, sorted, no spaces)
# ---------------------------------------------------------------------------


def _canonicalize_json(value: Any) -> _JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float values are not allowed")
        return value
    if callable(value):
        raise ValueError("callable leakage in payload is not allowed")
    if isinstance(value, tuple):
        return tuple(_canonicalize_json(item) for item in value)
    if isinstance(value, list):
        return tuple(_canonicalize_json(item) for item in value)
    if isinstance(value, Mapping):
        keys = tuple(value.keys())
        if any(not isinstance(key, str) for key in keys):
            raise ValueError("payload keys must be strings")
        return {key: _canonicalize_json(value[key]) for key in sorted(keys)}
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


def _quantize_unit_float(value: float, field_name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    quantized = Decimal(str(value)).quantize(_DECIMAL_PLACES, rounding=ROUND_HALF_EVEN)
    as_float = float(quantized)
    if as_float < 0.0 or as_float > 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]")
    return as_float


def _quantized_str(value: float, field_name: str) -> str:
    quantized = Decimal(str(_quantize_unit_float(value, field_name))).quantize(
        _DECIMAL_PLACES,
        rounding=ROUND_HALF_EVEN,
    )
    return str(quantized)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


# ---------------------------------------------------------------------------
# Shape morphology thresholds used by deterministic classification rules
# ---------------------------------------------------------------------------

_SPARSE_DENSITY_THRESHOLD = 0.1

# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticSignalGeometryConfig:
    """Configuration for the geometry kernel."""

    schema_version: str = SCHEMA_VERSION
    kernel_version: str = SCHEMA_VERSION
    enforce_strict_ordering: bool = True
    enforce_unique_frame_ids: bool = True
    coordinate_dimensions: int = 3

    def __post_init__(self) -> None:
        if self.coordinate_dimensions != 3:
            raise ValueError(
                "SyntheticSignalGeometryConfig.coordinate_dimensions must be 3; "
                "the current geometry kernel implementation is 3D-only."
            )

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "kernel_version": self.kernel_version,
            "enforce_strict_ordering": self.enforce_strict_ordering,
            "enforce_unique_frame_ids": self.enforce_unique_frame_ids,
            "coordinate_dimensions": self.coordinate_dimensions,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class SignalGeometryNode:
    """A single frame projected into geometry space."""

    frame_index: int
    activity_centroid: float
    spike_density_coordinate: float
    continuity_coordinate: float
    trajectory_vector: tuple[float, ...]
    stable_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "frame_index": self.frame_index,
            "activity_centroid": _quantized_str(self.activity_centroid, "activity_centroid"),
            "spike_density_coordinate": _quantized_str(self.spike_density_coordinate, "spike_density_coordinate"),
            "continuity_coordinate": _quantized_str(self.continuity_coordinate, "continuity_coordinate"),
            "trajectory_vector": tuple(
                _quantized_str(v, "trajectory_vector entry") for v in self.trajectory_vector
            ),
            "stable_hash": self.stable_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("stable_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class SignalGeometryEdge:
    """Deterministic edge between consecutive geometry nodes."""

    source_index: int
    target_index: int
    centroid_delta: float
    spike_density_delta: float
    continuity_delta: float
    edge_magnitude: float

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "source_index": self.source_index,
            "target_index": self.target_index,
            "centroid_delta": _quantized_str(
                _clamp01(abs(self.centroid_delta)), "centroid_delta"
            ),
            "spike_density_delta": _quantized_str(
                _clamp01(abs(self.spike_density_delta)), "spike_density_delta"
            ),
            "continuity_delta": _quantized_str(
                _clamp01(abs(self.continuity_delta)), "continuity_delta"
            ),
            "edge_magnitude": _quantized_str(self.edge_magnitude, "edge_magnitude"),
        }


@dataclass(frozen=True)
class SignalGeometryTrajectory:
    """Ordered sequence of geometry nodes with transition edges."""

    config: SyntheticSignalGeometryConfig
    input_trace_hash: str
    node_count: int
    nodes: tuple[SignalGeometryNode, ...]
    edges: tuple[SignalGeometryEdge, ...]
    shape_label: str
    shape_scores: Mapping[str, float]
    stable_hash: str
    schema_version: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "config": self.config.to_dict(),
            "input_trace_hash": self.input_trace_hash,
            "node_count": self.node_count,
            "nodes": tuple(node.to_dict() for node in self.nodes),
            "edges": tuple(edge.to_dict() for edge in self.edges),
            "shape_label": self.shape_label,
            "shape_scores": {
                k: _quantized_str(v, f"shape_scores.{k}")
                for k, v in sorted(self.shape_scores.items())
            },
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


@dataclass(frozen=True)
class SignalGeometryKernelResult:
    """Full kernel result including trajectory and bounded metrics."""

    trajectory: SignalGeometryTrajectory
    geometry_integrity_score: float
    continuity_score: float
    similarity_score: float
    path_stability_score: float
    stable_hash: str
    schema_version: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "trajectory": self.trajectory.to_dict(),
            "geometry_integrity_score": _quantized_str(
                self.geometry_integrity_score, "geometry_integrity_score"
            ),
            "continuity_score": _quantized_str(
                self.continuity_score, "continuity_score"
            ),
            "similarity_score": _quantized_str(
                self.similarity_score, "similarity_score"
            ),
            "path_stability_score": _quantized_str(
                self.path_stability_score, "path_stability_score"
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


@dataclass(frozen=True)
class SignalGeometryReceipt:
    """Auditable receipt for a geometry kernel run."""

    receipt_hash: str
    kernel_version: str
    schema_version: str
    input_trace_hash: str
    output_stable_hash: str
    node_count: int
    shape_label: str
    geometry_integrity_score: float
    continuity_score: float
    similarity_score: float
    path_stability_score: float
    validation_passed: bool

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "receipt_hash": self.receipt_hash,
            "kernel_version": self.kernel_version,
            "schema_version": self.schema_version,
            "input_trace_hash": self.input_trace_hash,
            "output_stable_hash": self.output_stable_hash,
            "node_count": self.node_count,
            "shape_label": self.shape_label,
            "geometry_integrity_score": _quantized_str(
                self.geometry_integrity_score, "geometry_integrity_score"
            ),
            "continuity_score": _quantized_str(
                self.continuity_score, "continuity_score"
            ),
            "similarity_score": _quantized_str(
                self.similarity_score, "similarity_score"
            ),
            "path_stability_score": _quantized_str(
                self.path_stability_score, "path_stability_score"
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


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_config(config: SyntheticSignalGeometryConfig) -> SyntheticSignalGeometryConfig:
    if config.schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {config.schema_version}")
    if config.kernel_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported kernel version: {config.kernel_version}")
    if config.coordinate_dimensions < 1:
        raise ValueError("coordinate_dimensions must be >= 1")
    return config


def _validate_trace(trace: HybridSignalTrace) -> HybridSignalTrace:
    if trace.schema_version != HYBRID_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported hybrid trace schema version: {trace.schema_version}"
        )
    if trace.frame_count <= 0:
        raise ValueError("trace frame_count must be > 0")
    if len(trace.frames) != trace.frame_count:
        raise ValueError("trace frames length must equal frame_count")
    computed_hash = _sha256_hex(trace.to_hash_payload_dict())
    if trace.stable_hash != computed_hash:
        raise ValueError("trace stable_hash does not match computed hash")
    return trace


def _validate_nodes(
    nodes: tuple[SignalGeometryNode, ...],
    config: SyntheticSignalGeometryConfig,
) -> None:
    if len(nodes) == 0:
        raise ValueError("nodes must be non-empty")
    seen_indices: set[int] = set()
    last_index = -1
    for node in nodes:
        if config.enforce_strict_ordering and node.frame_index <= last_index:
            raise ValueError("nodes must be strictly ordered by frame_index")
        if config.enforce_unique_frame_ids and node.frame_index in seen_indices:
            raise ValueError("duplicate frame_index detected")
        seen_indices.add(node.frame_index)
        last_index = node.frame_index
        if len(node.trajectory_vector) != config.coordinate_dimensions:
            raise ValueError(
                "trajectory_vector length must equal config.coordinate_dimensions"
            )
        for v in node.trajectory_vector:
            if not math.isfinite(v):
                raise ValueError("trajectory_vector contains non-finite value")


# ---------------------------------------------------------------------------
# Geometry projection
# ---------------------------------------------------------------------------


def project_trace_to_geometry(
    trace: HybridSignalTrace,
    config: SyntheticSignalGeometryConfig | None = None,
) -> tuple[SignalGeometryNode, ...]:
    """Project hybrid signal frames into deterministic geometry nodes."""
    effective_config = _validate_config(config or SyntheticSignalGeometryConfig())
    _validate_trace(trace)

    nodes: list[SignalGeometryNode] = []
    prev_centroid: float | None = None

    for frame in trace.frames:
        node_count = len(frame.node_state_lane)
        if node_count == 0:
            raise ValueError("frame node_state_lane must be non-empty")

        activity_centroid = _quantize_unit_float(
            sum(frame.node_state_lane) / float(node_count),
            "activity_centroid",
        )

        spike_count = sum(frame.spike_event_lane)
        spike_density = _quantize_unit_float(
            float(spike_count) / float(node_count),
            "spike_density_coordinate",
        )

        if prev_centroid is None:
            continuity = 0.0
        else:
            continuity = _quantize_unit_float(
                abs(activity_centroid - prev_centroid),
                "continuity_coordinate",
            )

        trajectory_vector = (activity_centroid, spike_density, continuity)

        proto = SignalGeometryNode(
            frame_index=frame.time_index,
            activity_centroid=activity_centroid,
            spike_density_coordinate=spike_density,
            continuity_coordinate=continuity,
            trajectory_vector=trajectory_vector,
            stable_hash="",
        )
        nodes.append(
            SignalGeometryNode(
                frame_index=proto.frame_index,
                activity_centroid=proto.activity_centroid,
                spike_density_coordinate=proto.spike_density_coordinate,
                continuity_coordinate=proto.continuity_coordinate,
                trajectory_vector=proto.trajectory_vector,
                stable_hash=_sha256_hex(proto.to_hash_payload_dict()),
            )
        )

        prev_centroid = activity_centroid

    return tuple(nodes)


# ---------------------------------------------------------------------------
# Transition geometry
# ---------------------------------------------------------------------------


def _build_transition_edges(
    nodes: tuple[SignalGeometryNode, ...],
) -> tuple[SignalGeometryEdge, ...]:
    """Build deterministic edges between consecutive geometry nodes."""
    edges: list[SignalGeometryEdge] = []
    for i in range(len(nodes) - 1):
        src = nodes[i]
        tgt = nodes[i + 1]
        cd = tgt.activity_centroid - src.activity_centroid
        sd = tgt.spike_density_coordinate - src.spike_density_coordinate
        cond = tgt.continuity_coordinate - src.continuity_coordinate
        magnitude = math.sqrt(cd * cd + sd * sd + cond * cond)
        magnitude = _quantize_unit_float(_clamp01(magnitude), "edge_magnitude")
        edges.append(
            SignalGeometryEdge(
                source_index=src.frame_index,
                target_index=tgt.frame_index,
                centroid_delta=cd,
                spike_density_delta=sd,
                continuity_delta=cond,
                edge_magnitude=magnitude,
            )
        )
    return tuple(edges)


# ---------------------------------------------------------------------------
# Shape abstraction (deterministic rules only)
# ---------------------------------------------------------------------------


def _compute_shape_scores(
    nodes: tuple[SignalGeometryNode, ...],
) -> dict[str, float]:
    """Compute deterministic shape scores from geometry nodes."""
    centroids = tuple(n.activity_centroid for n in nodes)
    densities = tuple(n.spike_density_coordinate for n in nodes)
    n = len(centroids)

    # Monotonic score: fraction of consecutive pairs moving in one direction
    if n < 2:
        monotonic_score = 1.0
        linear_score = 1.0
        cyclic_score = 0.0
        resonant_score = 0.0
    else:
        increasing = sum(1 for i in range(n - 1) if centroids[i + 1] >= centroids[i])
        decreasing = sum(1 for i in range(n - 1) if centroids[i + 1] <= centroids[i])
        monotonic_score = _clamp01(max(increasing, decreasing) / float(n - 1))

        # Linear score: R² of centroid vs time index (least squares)
        mean_t = (n - 1) / 2.0
        mean_c = sum(centroids) / float(n)
        ss_tot = sum((c - mean_c) ** 2 for c in centroids)
        ss_xy = sum((i - mean_t) * (centroids[i] - mean_c) for i in range(n))
        ss_tt = sum((i - mean_t) ** 2 for i in range(n))
        if ss_tot > 0.0 and ss_tt > 0.0:
            r_squared = (ss_xy * ss_xy) / (ss_tt * ss_tot)
            linear_score = _clamp01(r_squared)
        else:
            linear_score = 1.0

        # Cyclic score: closeness of first and last centroid
        cyclic_distance = abs(centroids[-1] - centroids[0])
        cyclic_score = _clamp01(1.0 - cyclic_distance / max(max(centroids) - min(centroids), 1e-12))

        # Resonant score: fraction of direction flips
        flips = sum(
            1 for i in range(1, n - 1)
            if (centroids[i] - centroids[i - 1]) * (centroids[i + 1] - centroids[i]) < 0
        )
        resonant_score = _clamp01(flips / float(n - 2)) if n > 2 else 0.0

    # Clustered score: inverse of centroid variance
    mean_c_all = sum(centroids) / float(n)
    variance = sum((c - mean_c_all) ** 2 for c in centroids) / float(n)
    clustered_score = _clamp01(1.0 - math.sqrt(variance) * 4.0)

    # Sparse score: fraction of frames with low spike density
    sparse_count = sum(1 for d in densities if d < _SPARSE_DENSITY_THRESHOLD)
    sparse_score = _clamp01(float(sparse_count) / float(n))

    return {
        "clustered": _quantize_unit_float(clustered_score, "clustered_score"),
        "cyclic": _quantize_unit_float(cyclic_score, "cyclic_score"),
        "linear": _quantize_unit_float(linear_score, "linear_score"),
        "monotonic": _quantize_unit_float(monotonic_score, "monotonic_score"),
        "resonant": _quantize_unit_float(resonant_score, "resonant_score"),
        "sparse": _quantize_unit_float(sparse_score, "sparse_score"),
    }


def _classify_shape(scores: dict[str, float]) -> str:
    """Deterministic shape classification by highest score with tie-breaking."""
    # Priority order for tie-breaking: alphabetical (deterministic)
    priority = ("clustered", "cyclic", "linear", "monotonic", "resonant", "sparse")
    best_label = priority[0]
    best_score = scores.get(best_label, 0.0)
    for label in priority[1:]:
        s = scores.get(label, 0.0)
        if s > best_score:
            best_score = s
            best_label = label
    return best_label


# ---------------------------------------------------------------------------
# Trajectory similarity
# ---------------------------------------------------------------------------


def compute_geometry_similarity(
    traj_a: SignalGeometryTrajectory,
    traj_b: SignalGeometryTrajectory,
) -> dict[str, float]:
    """Compute bounded [0,1] similarity metrics between two trajectories."""
    nodes_a = traj_a.nodes
    nodes_b = traj_b.nodes
    min_len = min(len(nodes_a), len(nodes_b))
    max_len = max(len(nodes_a), len(nodes_b))

    if max_len == 0:
        raise ValueError("cannot compute similarity on empty trajectories")

    # Length overlap ratio
    length_overlap = float(min_len) / float(max_len)

    # Geometric continuity: mean absolute centroid difference over shared range
    if min_len == 0:
        centroid_similarity = 0.0
        density_similarity = 0.0
        continuity_similarity = 0.0
    else:
        centroid_diffs = tuple(
            abs(nodes_a[i].activity_centroid - nodes_b[i].activity_centroid)
            for i in range(min_len)
        )
        centroid_similarity = _clamp01(1.0 - sum(centroid_diffs) / float(min_len))

        density_diffs = tuple(
            abs(nodes_a[i].spike_density_coordinate - nodes_b[i].spike_density_coordinate)
            for i in range(min_len)
        )
        density_similarity = _clamp01(1.0 - sum(density_diffs) / float(min_len))

        continuity_diffs = tuple(
            abs(nodes_a[i].continuity_coordinate - nodes_b[i].continuity_coordinate)
            for i in range(min_len)
        )
        continuity_similarity = _clamp01(1.0 - sum(continuity_diffs) / float(min_len))

    # Structural similarity: weighted combination
    structural_similarity = _clamp01(
        0.4 * centroid_similarity + 0.3 * density_similarity + 0.3 * continuity_similarity
    )

    # Trajectory overlap: combines length overlap and structural similarity
    trajectory_overlap = _clamp01(length_overlap * structural_similarity)

    # Centroid drift: how much final centroids diverge
    if len(nodes_a) > 0 and len(nodes_b) > 0:
        final_drift = abs(
            nodes_a[-1].activity_centroid - nodes_b[-1].activity_centroid
        )
        centroid_drift_score = _clamp01(1.0 - final_drift)
    else:
        centroid_drift_score = 0.0

    return {
        "geometric_continuity": _quantize_unit_float(
            centroid_similarity, "geometric_continuity"
        ),
        "structural_similarity": _quantize_unit_float(
            structural_similarity, "structural_similarity"
        ),
        "trajectory_overlap": _quantize_unit_float(
            trajectory_overlap, "trajectory_overlap"
        ),
        "centroid_drift_score": _quantize_unit_float(
            centroid_drift_score, "centroid_drift_score"
        ),
    }


# ---------------------------------------------------------------------------
# Trajectory builder
# ---------------------------------------------------------------------------


def build_geometry_trajectory(
    trace: HybridSignalTrace,
    config: SyntheticSignalGeometryConfig | None = None,
) -> SignalGeometryTrajectory:
    """Build a complete geometry trajectory from a hybrid signal trace."""
    effective_config = _validate_config(config or SyntheticSignalGeometryConfig())
    _validate_trace(trace)

    nodes = project_trace_to_geometry(trace, effective_config)
    _validate_nodes(nodes, effective_config)
    edges = _build_transition_edges(nodes)
    shape_scores = _compute_shape_scores(nodes)
    shape_label = _classify_shape(shape_scores)

    immutable_scores: Mapping[str, float] = types.MappingProxyType(shape_scores)

    proto = SignalGeometryTrajectory(
        config=effective_config,
        input_trace_hash=trace.stable_hash,
        node_count=len(nodes),
        nodes=nodes,
        edges=edges,
        shape_label=shape_label,
        shape_scores=immutable_scores,
        stable_hash="",
        schema_version=effective_config.schema_version,
    )
    return SignalGeometryTrajectory(
        config=proto.config,
        input_trace_hash=proto.input_trace_hash,
        node_count=proto.node_count,
        nodes=proto.nodes,
        edges=proto.edges,
        shape_label=proto.shape_label,
        shape_scores=proto.shape_scores,
        stable_hash=_sha256_hex(proto.to_hash_payload_dict()),
        schema_version=proto.schema_version,
    )


# ---------------------------------------------------------------------------
# Kernel metrics
# ---------------------------------------------------------------------------


def _compute_kernel_metrics(
    trajectory: SignalGeometryTrajectory,
) -> dict[str, float]:
    """Compute bounded [0,1] kernel-level metrics."""
    nodes = trajectory.nodes
    edges = trajectory.edges
    n = len(nodes)

    # Geometry integrity: all nodes have valid hashes, edges are contiguous
    valid_hashes = sum(1 for nd in nodes if len(nd.stable_hash) == 64)
    hash_integrity = float(valid_hashes) / float(n) if n > 0 else 0.0

    edge_integrity = 1.0
    if len(edges) > 0:
        for i, edge in enumerate(edges):
            if edge.source_index != nodes[i].frame_index:
                edge_integrity = 0.0
                break
            if edge.target_index != nodes[i + 1].frame_index:
                edge_integrity = 0.0
                break
    geometry_integrity = _clamp01((hash_integrity + edge_integrity) / 2.0)

    # Continuity score: mean continuity coordinate (lower = more continuous)
    if n > 0:
        mean_continuity = sum(nd.continuity_coordinate for nd in nodes) / float(n)
        continuity_score = _clamp01(1.0 - mean_continuity)
    else:
        continuity_score = 0.0

    # Similarity score: self-similarity (always 1.0 for a single trajectory)
    similarity_score = 1.0

    # Path stability: inverse of mean edge magnitude (lower magnitude = more stable)
    if len(edges) > 0:
        mean_magnitude = sum(e.edge_magnitude for e in edges) / float(len(edges))
        path_stability = _clamp01(1.0 - mean_magnitude)
    else:
        path_stability = 1.0

    return {
        "geometry_integrity_score": _quantize_unit_float(
            geometry_integrity, "geometry_integrity_score"
        ),
        "continuity_score": _quantize_unit_float(
            continuity_score, "continuity_score"
        ),
        "similarity_score": _quantize_unit_float(
            similarity_score, "similarity_score"
        ),
        "path_stability_score": _quantize_unit_float(
            path_stability, "path_stability_score"
        ),
    }


# ---------------------------------------------------------------------------
# Kernel runner
# ---------------------------------------------------------------------------


def run_signal_geometry_kernel(
    trace: HybridSignalTrace,
    config: SyntheticSignalGeometryConfig | None = None,
) -> tuple[SignalGeometryKernelResult, SignalGeometryReceipt]:
    """Run the full signal geometry kernel pipeline.

    Returns (result, receipt) where both are frozen, deterministic, and
    receipt-chain auditable.
    """
    effective_config = _validate_config(config or SyntheticSignalGeometryConfig())
    trajectory = build_geometry_trajectory(trace, effective_config)
    metrics = _compute_kernel_metrics(trajectory)

    result_proto = SignalGeometryKernelResult(
        trajectory=trajectory,
        geometry_integrity_score=metrics["geometry_integrity_score"],
        continuity_score=metrics["continuity_score"],
        similarity_score=metrics["similarity_score"],
        path_stability_score=metrics["path_stability_score"],
        stable_hash="",
        schema_version=effective_config.schema_version,
    )
    result = SignalGeometryKernelResult(
        trajectory=result_proto.trajectory,
        geometry_integrity_score=result_proto.geometry_integrity_score,
        continuity_score=result_proto.continuity_score,
        similarity_score=result_proto.similarity_score,
        path_stability_score=result_proto.path_stability_score,
        stable_hash=_sha256_hex(result_proto.to_hash_payload_dict()),
        schema_version=result_proto.schema_version,
    )

    receipt_proto = SignalGeometryReceipt(
        receipt_hash="",
        kernel_version=effective_config.kernel_version,
        schema_version=effective_config.schema_version,
        input_trace_hash=trace.stable_hash,
        output_stable_hash=result.stable_hash,
        node_count=trajectory.node_count,
        shape_label=trajectory.shape_label,
        geometry_integrity_score=result.geometry_integrity_score,
        continuity_score=result.continuity_score,
        similarity_score=result.similarity_score,
        path_stability_score=result.path_stability_score,
        validation_passed=True,
    )
    receipt = SignalGeometryReceipt(
        receipt_hash=_sha256_hex(receipt_proto.to_hash_payload_dict()),
        kernel_version=receipt_proto.kernel_version,
        schema_version=receipt_proto.schema_version,
        input_trace_hash=receipt_proto.input_trace_hash,
        output_stable_hash=receipt_proto.output_stable_hash,
        node_count=receipt_proto.node_count,
        shape_label=receipt_proto.shape_label,
        geometry_integrity_score=receipt_proto.geometry_integrity_score,
        continuity_score=receipt_proto.continuity_score,
        similarity_score=receipt_proto.similarity_score,
        path_stability_score=receipt_proto.path_stability_score,
        validation_passed=receipt_proto.validation_passed,
    )

    return result, receipt


# ---------------------------------------------------------------------------
# Optional ASCII summary
# ---------------------------------------------------------------------------


def build_ascii_geometry_summary(result: SignalGeometryKernelResult) -> str:
    """Build a deterministic ASCII summary of the geometry kernel result."""
    traj = result.trajectory
    lines = [
        f"Signal Geometry Kernel — {result.schema_version}",
        f"  Shape:      {traj.shape_label}",
        f"  Nodes:      {traj.node_count}",
        f"  Edges:      {len(traj.edges)}",
        f"  Integrity:  {_quantized_str(result.geometry_integrity_score, 'integrity')}",
        f"  Continuity: {_quantized_str(result.continuity_score, 'continuity')}",
        f"  Similarity: {_quantized_str(result.similarity_score, 'similarity')}",
        f"  Stability:  {_quantized_str(result.path_stability_score, 'stability')}",
        f"  Hash:       {result.stable_hash[:16]}...",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "SCHEMA_VERSION",
    "SyntheticSignalGeometryConfig",
    "SignalGeometryNode",
    "SignalGeometryEdge",
    "SignalGeometryTrajectory",
    "SignalGeometryKernelResult",
    "SignalGeometryReceipt",
    "project_trace_to_geometry",
    "compute_geometry_similarity",
    "build_geometry_trajectory",
    "run_signal_geometry_kernel",
    "build_ascii_geometry_summary",
]
