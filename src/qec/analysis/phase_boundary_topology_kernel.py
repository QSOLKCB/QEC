"""v137.13.2 — Phase Boundary Topology Kernel.

Deterministic Layer-4 topology analysis over morphology transition paths.
Builds phase regions, boundary crossings, bounded metrics, and receipt-chain
artifacts with canonical export semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import json
import math
from typing import Any, Mapping

from qec.analysis.morphology_transition_kernel import (
    SCHEMA_VERSION as MORPHOLOGY_SCHEMA_VERSION,
    MorphologyTransitionPath,
)

SCHEMA_VERSION = "v137.13.2"
_DECIMAL_PLACES = Decimal("0.000000000001")

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

_REGION_LABEL_PRIORITY = (
    "stable_region",
    "oscillatory_region",
    "resonant_region",
    "divergence_region",
    "transitional_region",
)

_BOUNDARY_LABEL_PRIORITY = (
    "continuity_break",
    "state_flip",
    "drift_jump",
    "topology_inversion",
    "region_transition",
)

_MORPH_LABEL_TO_REGION = {
    "stable": "stable_region",
    "oscillatory": "oscillatory_region",
    "resonant": "resonant_region",
    "diverging": "divergence_region",
    "converging": "transitional_region",
    "transitional": "transitional_region",
}


def _canonicalize_json(value: Any) -> _JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float values are not allowed")
        return value
    if callable(value):
        raise ValueError("callable values are not allowed in canonical payloads")
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


def _sha256_hex(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


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


@dataclass(frozen=True)
class PhaseBoundaryTopologyConfig:
    schema_version: str = SCHEMA_VERSION
    kernel_version: str = SCHEMA_VERSION
    continuity_break_threshold: float = 0.200000000000
    drift_jump_threshold: float = 0.200000000000
    state_flip_delta_threshold: float = 0.100000000000

    def __post_init__(self) -> None:
        for field_name in (
            "continuity_break_threshold",
            "drift_jump_threshold",
            "state_flip_delta_threshold",
        ):
            value = getattr(self, field_name)
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{field_name} must be in [0, 1]")

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "kernel_version": self.kernel_version,
            "continuity_break_threshold": _quantized_str(
                self.continuity_break_threshold,
                field_name="continuity_break_threshold",
            ),
            "drift_jump_threshold": _quantized_str(
                self.drift_jump_threshold,
                field_name="drift_jump_threshold",
            ),
            "state_flip_delta_threshold": _quantized_str(
                self.state_flip_delta_threshold,
                field_name="state_flip_delta_threshold",
            ),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_sha256(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class PhaseRegion:
    region_id: str
    source_start_index: int
    source_end_index: int
    region_label: str
    region_score: float
    continuity_mean: float
    morphology_mean: float

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "region_id": self.region_id,
            "source_start_index": self.source_start_index,
            "source_end_index": self.source_end_index,
            "region_label": self.region_label,
            "region_score": _quantized_str(self.region_score, field_name="region_score"),
            "continuity_mean": _quantized_str(self.continuity_mean, field_name="continuity_mean"),
            "morphology_mean": _quantized_str(self.morphology_mean, field_name="morphology_mean"),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_sha256(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class PhaseBoundaryEdge:
    source_region_index: int
    target_region_index: int
    boundary_type: str
    boundary_magnitude: float
    continuity_delta: float
    morphology_delta: float

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "source_region_index": self.source_region_index,
            "target_region_index": self.target_region_index,
            "boundary_type": self.boundary_type,
            "boundary_magnitude": _quantized_str(self.boundary_magnitude, field_name="boundary_magnitude"),
            "continuity_delta": _quantized_str(self.continuity_delta, field_name="continuity_delta"),
            "morphology_delta": _quantized_str(self.morphology_delta, field_name="morphology_delta"),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_sha256(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class PhaseTopologyPath:
    config: PhaseBoundaryTopologyConfig
    input_transition_hash: str
    regions: tuple[PhaseRegion, ...]
    boundaries: tuple[PhaseBoundaryEdge, ...]
    stable_hash: str
    schema_version: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "config": self.config.to_dict(),
            "input_transition_hash": self.input_transition_hash,
            "regions": tuple(region.to_dict() for region in self.regions),
            "boundaries": tuple(boundary.to_dict() for boundary in self.boundaries),
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
class PhaseBoundaryTopologyResult:
    path: PhaseTopologyPath
    boundary_integrity_score: float
    topology_stability_score: float
    region_consistency_score: float
    boundary_continuity_score: float
    stable_hash: str
    schema_version: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "path": self.path.to_dict(),
            "boundary_integrity_score": _quantized_str(
                self.boundary_integrity_score,
                field_name="boundary_integrity_score",
            ),
            "topology_stability_score": _quantized_str(
                self.topology_stability_score,
                field_name="topology_stability_score",
            ),
            "region_consistency_score": _quantized_str(
                self.region_consistency_score,
                field_name="region_consistency_score",
            ),
            "boundary_continuity_score": _quantized_str(
                self.boundary_continuity_score,
                field_name="boundary_continuity_score",
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
class PhaseBoundaryTopologyReceipt:
    receipt_hash: str
    kernel_version: str
    schema_version: str
    input_transition_hash: str
    output_stable_hash: str
    region_count: int
    boundary_count: int
    receipt_chain: tuple[str, ...]
    boundary_integrity_score: float
    topology_stability_score: float
    region_consistency_score: float
    boundary_continuity_score: float
    validation_passed: bool

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "receipt_hash": self.receipt_hash,
            "kernel_version": self.kernel_version,
            "schema_version": self.schema_version,
            "input_transition_hash": self.input_transition_hash,
            "output_stable_hash": self.output_stable_hash,
            "region_count": self.region_count,
            "boundary_count": self.boundary_count,
            "receipt_chain": self.receipt_chain,
            "boundary_integrity_score": _quantized_str(
                self.boundary_integrity_score,
                field_name="boundary_integrity_score",
            ),
            "topology_stability_score": _quantized_str(
                self.topology_stability_score,
                field_name="topology_stability_score",
            ),
            "region_consistency_score": _quantized_str(
                self.region_consistency_score,
                field_name="region_consistency_score",
            ),
            "boundary_continuity_score": _quantized_str(
                self.boundary_continuity_score,
                field_name="boundary_continuity_score",
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


def _validate_config(config: PhaseBoundaryTopologyConfig) -> PhaseBoundaryTopologyConfig:
    if config.schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {config.schema_version}")
    if config.kernel_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported kernel version: {config.kernel_version}")
    return config


def _validate_transition_path(path: MorphologyTransitionPath) -> MorphologyTransitionPath:
    if path.schema_version != MORPHOLOGY_SCHEMA_VERSION:
        raise ValueError("schema mismatch with morphology transition path")
    if path.stable_hash != _sha256_hex(path.to_hash_payload_dict()):
        raise ValueError("broken lineage: transition path stable_hash mismatch")
    if len(path.states) == 0:
        raise ValueError("empty path")

    state_ids = tuple(state.state_id for state in path.states)
    if len(set(state_ids)) != len(state_ids):
        raise ValueError("duplicate state ids")

    if len(path.edges) != max(0, len(path.states) - 1):
        raise ValueError("invalid ordering")

    previous_index = -1
    for state in path.states:
        if state.state_label not in _MORPH_LABEL_TO_REGION:
            raise ValueError("schema mismatch")
        if state.source_index <= previous_index:
            raise ValueError("invalid ordering")
        previous_index = state.source_index
        for value in (
            state.activity_centroid,
            state.spike_density_coordinate,
            state.continuity_coordinate,
            state.state_score,
        ):
            if not math.isfinite(value):
                raise ValueError("non-finite metrics")

    for i, edge in enumerate(path.edges):
        source = path.states[i]
        target = path.states[i + 1]
        if edge.source_index != source.source_index or edge.target_index != target.source_index:
            raise ValueError("invalid ordering: edge indices do not align")
        if edge.source_state != source.state_label or edge.target_state != target.state_label:
            raise ValueError("schema mismatch: edge/state label mismatch")
        for value in (edge.transition_magnitude, edge.stability_delta, edge.continuity_delta):
            if not math.isfinite(value):
                raise ValueError("non-finite metrics")

    return path


def _validate_topology_path(path: PhaseTopologyPath) -> None:
    if path.schema_version != SCHEMA_VERSION:
        raise ValueError("schema mismatch")
    if len(path.regions) == 0:
        raise ValueError("empty path")

    region_ids = tuple(region.region_id for region in path.regions)
    if len(set(region_ids)) != len(region_ids):
        raise ValueError("duplicate region ids")

    previous_start = -1
    for region in path.regions:
        if region.region_label not in _REGION_LABEL_PRIORITY:
            raise ValueError("schema mismatch")
        if region.source_start_index <= previous_start:
            raise ValueError("invalid ordering")
        if region.source_end_index < region.source_start_index:
            raise ValueError("invalid ordering")
        previous_start = region.source_start_index
        for value in (region.region_score, region.continuity_mean, region.morphology_mean):
            if not math.isfinite(value):
                raise ValueError("non-finite metrics")

    if len(path.boundaries) != max(0, len(path.regions) - 1):
        raise ValueError("invalid ordering")

    for i, boundary in enumerate(path.boundaries):
        if boundary.boundary_type not in _BOUNDARY_LABEL_PRIORITY:
            raise ValueError("schema mismatch")
        if boundary.source_region_index != i or boundary.target_region_index != i + 1:
            raise ValueError("invalid ordering")
        for value in (boundary.boundary_magnitude, boundary.continuity_delta, boundary.morphology_delta):
            if not math.isfinite(value):
                raise ValueError("non-finite metrics")

    if path.stable_hash != _sha256_hex(path.to_hash_payload_dict()):
        raise ValueError("broken lineage")


def _state_to_region_label(state_label: str) -> str:
    return _MORPH_LABEL_TO_REGION[state_label]


def detect_phase_regions(
    transition_path: MorphologyTransitionPath,
    config: PhaseBoundaryTopologyConfig | None = None,
) -> tuple[PhaseRegion, ...]:
    """Convert ordered morphology states into deterministic phase regions."""
    _validate_config(config or PhaseBoundaryTopologyConfig())
    valid_path = _validate_transition_path(transition_path)

    grouped: list[tuple[str, int, int]] = []
    states = valid_path.states
    start_idx = 0
    current_label = _state_to_region_label(states[0].state_label)

    for idx in range(1, len(states)):
        label = _state_to_region_label(states[idx].state_label)
        if label != current_label:
            grouped.append((current_label, start_idx, idx - 1))
            start_idx = idx
            current_label = label
    grouped.append((current_label, start_idx, len(states) - 1))

    regions: list[PhaseRegion] = []
    for position, (label, left, right) in enumerate(grouped):
        chunk = states[left : right + 1]
        width = float(len(chunk))
        region_score = _quantize_unit(
            sum(state.state_score for state in chunk) / width,
            field_name="region_score",
        )
        continuity_mean = _quantize_unit(
            sum(state.continuity_coordinate for state in chunk) / width,
            field_name="continuity_mean",
        )
        morphology_mean = _quantize_unit(
            sum(state.activity_centroid for state in chunk) / width,
            field_name="morphology_mean",
        )
        seed = {
            "input_transition_hash": valid_path.stable_hash,
            "source_start_index": chunk[0].source_index,
            "source_end_index": chunk[-1].source_index,
            "region_label": label,
            "position": position,
        }
        regions.append(
            PhaseRegion(
                region_id=_sha256_hex(seed),
                source_start_index=chunk[0].source_index,
                source_end_index=chunk[-1].source_index,
                region_label=label,
                region_score=region_score,
                continuity_mean=continuity_mean,
                morphology_mean=morphology_mean,
            )
        )

    return tuple(regions)


def _choose_boundary_type(
    *,
    source_label: str,
    target_label: str,
    continuity_delta_abs: float,
    morphology_delta_abs: float,
    region_score_delta_abs: float,
    config: PhaseBoundaryTopologyConfig,
) -> str:
    rules: dict[str, bool] = {
        "continuity_break": continuity_delta_abs >= config.continuity_break_threshold,
        "state_flip": region_score_delta_abs >= config.state_flip_delta_threshold,
        "drift_jump": morphology_delta_abs >= config.drift_jump_threshold,
        "topology_inversion": (
            source_label in ("stable_region", "resonant_region") and target_label == "divergence_region"
        )
        or (
            source_label == "divergence_region"
            and target_label in ("stable_region", "resonant_region")
        ),
        "region_transition": True,
    }
    for boundary_label in _BOUNDARY_LABEL_PRIORITY:
        if rules[boundary_label]:
            return boundary_label
    return "region_transition"


def build_phase_topology_path(
    transition_path: MorphologyTransitionPath,
    config: PhaseBoundaryTopologyConfig | None = None,
) -> PhaseTopologyPath:
    """Build deterministic phase topology path from a morphology transition path."""
    effective_config = _validate_config(config or PhaseBoundaryTopologyConfig())
    valid_transition_path = _validate_transition_path(transition_path)
    regions = detect_phase_regions(valid_transition_path, effective_config)

    boundaries: list[PhaseBoundaryEdge] = []
    for index in range(len(regions) - 1):
        source = regions[index]
        target = regions[index + 1]
        continuity_delta = _quantize(
            target.continuity_mean - source.continuity_mean,
            field_name="continuity_delta",
        )
        morphology_delta = _quantize(
            target.morphology_mean - source.morphology_mean,
            field_name="morphology_delta",
        )
        boundary_magnitude = _quantize_unit(
            (abs(continuity_delta) + abs(morphology_delta)) * 0.5,
            field_name="boundary_magnitude",
        )
        boundary_type = _choose_boundary_type(
            source_label=source.region_label,
            target_label=target.region_label,
            continuity_delta_abs=abs(continuity_delta),
            morphology_delta_abs=abs(morphology_delta),
            region_score_delta_abs=abs(target.region_score - source.region_score),
            config=effective_config,
        )

        boundaries.append(
            PhaseBoundaryEdge(
                source_region_index=index,
                target_region_index=index + 1,
                boundary_type=boundary_type,
                boundary_magnitude=boundary_magnitude,
                continuity_delta=continuity_delta,
                morphology_delta=morphology_delta,
            )
        )

    proto = PhaseTopologyPath(
        config=effective_config,
        input_transition_hash=valid_transition_path.stable_hash,
        regions=regions,
        boundaries=tuple(boundaries),
        stable_hash="",
        schema_version=effective_config.schema_version,
    )
    path = PhaseTopologyPath(
        config=proto.config,
        input_transition_hash=proto.input_transition_hash,
        regions=proto.regions,
        boundaries=proto.boundaries,
        stable_hash=_sha256_hex(proto.to_hash_payload_dict()),
        schema_version=proto.schema_version,
    )
    _validate_topology_path(path)
    return path


def _compute_boundary_integrity_score(path: PhaseTopologyPath) -> float:
    """Measure boundary coherence across adjacent valid boundaries."""
    if len(path.boundaries) <= 1:
        return 1.0

    pair_scores: list[float] = []
    for previous_boundary, current_boundary in zip(
        path.boundaries,
        path.boundaries[1:],
    ):
        continuity_coherence = 1.0 - _clamp01(
            abs(current_boundary.continuity_delta - previous_boundary.continuity_delta)
        )
        morphology_coherence = 1.0 - _clamp01(
            abs(current_boundary.morphology_delta - previous_boundary.morphology_delta)
        )
        pair_scores.append((continuity_coherence + morphology_coherence) / 2.0)

    return sum(pair_scores) / float(len(pair_scores))


def _compute_kernel_metrics(path: PhaseTopologyPath) -> dict[str, float]:
    if len(path.boundaries) == 0:
        boundary_integrity = 1.0
        continuity_score = 1.0
        region_consistency = 1.0
    else:
        boundary_integrity = _compute_boundary_integrity_score(path)
        continuity_score = 1.0 - _clamp01(
            sum(abs(boundary.continuity_delta) for boundary in path.boundaries)
            / float(len(path.boundaries))
        )
        region_consistency = 1.0 - _clamp01(
            sum(abs(boundary.morphology_delta) for boundary in path.boundaries)
            / float(len(path.boundaries))
        )

    stable_like = sum(
        1 for region in path.regions if region.region_label in ("stable_region", "resonant_region")
    )
    topology_stability = float(stable_like) / float(len(path.regions))

    return {
        "boundary_integrity_score": _quantize_unit(
            boundary_integrity,
            field_name="boundary_integrity_score",
        ),
        "topology_stability_score": _quantize_unit(
            topology_stability,
            field_name="topology_stability_score",
        ),
        "region_consistency_score": _quantize_unit(
            region_consistency,
            field_name="region_consistency_score",
        ),
        "boundary_continuity_score": _quantize_unit(
            continuity_score,
            field_name="boundary_continuity_score",
        ),
    }


def compute_phase_similarity(
    path_a: PhaseTopologyPath,
    path_b: PhaseTopologyPath,
) -> dict[str, float]:
    """Compute deterministic bounded similarity between phase topology paths."""
    _validate_topology_path(path_a)
    _validate_topology_path(path_b)

    min_len = min(len(path_a.regions), len(path_b.regions))
    max_len = max(len(path_a.regions), len(path_b.regions))

    length_overlap = float(min_len) / float(max_len)
    label_alignment = float(
        sum(1 for i in range(min_len) if path_a.regions[i].region_label == path_b.regions[i].region_label)
    ) / float(min_len)
    morphology_similarity = 1.0 - _clamp01(
        sum(
            abs(path_a.regions[i].morphology_mean - path_b.regions[i].morphology_mean)
            for i in range(min_len)
        )
        / float(min_len)
    )

    topology_similarity = _clamp01((0.6 * label_alignment) + (0.4 * morphology_similarity))
    path_overlap = _clamp01(length_overlap * topology_similarity)

    return {
        "length_overlap": _quantize_unit(length_overlap, field_name="length_overlap"),
        "label_alignment": _quantize_unit(label_alignment, field_name="label_alignment"),
        "topology_similarity": _quantize_unit(topology_similarity, field_name="topology_similarity"),
        "path_overlap": _quantize_unit(path_overlap, field_name="path_overlap"),
    }


def run_phase_boundary_topology_kernel(
    transition_path: MorphologyTransitionPath,
    config: PhaseBoundaryTopologyConfig | None = None,
) -> tuple[PhaseBoundaryTopologyResult, PhaseBoundaryTopologyReceipt]:
    """Run the deterministic phase boundary topology kernel pipeline."""
    effective_config = _validate_config(config or PhaseBoundaryTopologyConfig())
    path = build_phase_topology_path(transition_path, effective_config)
    metrics = _compute_kernel_metrics(path)

    result_proto = PhaseBoundaryTopologyResult(
        path=path,
        boundary_integrity_score=metrics["boundary_integrity_score"],
        topology_stability_score=metrics["topology_stability_score"],
        region_consistency_score=metrics["region_consistency_score"],
        boundary_continuity_score=metrics["boundary_continuity_score"],
        stable_hash="",
        schema_version=effective_config.schema_version,
    )
    result = PhaseBoundaryTopologyResult(
        path=result_proto.path,
        boundary_integrity_score=result_proto.boundary_integrity_score,
        topology_stability_score=result_proto.topology_stability_score,
        region_consistency_score=result_proto.region_consistency_score,
        boundary_continuity_score=result_proto.boundary_continuity_score,
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

    receipt_proto = PhaseBoundaryTopologyReceipt(
        receipt_hash="",
        kernel_version=effective_config.kernel_version,
        schema_version=effective_config.schema_version,
        input_transition_hash=path.input_transition_hash,
        output_stable_hash=result.stable_hash,
        region_count=len(path.regions),
        boundary_count=len(path.boundaries),
        receipt_chain=receipt_chain,
        boundary_integrity_score=result.boundary_integrity_score,
        topology_stability_score=result.topology_stability_score,
        region_consistency_score=result.region_consistency_score,
        boundary_continuity_score=result.boundary_continuity_score,
        validation_passed=True,
    )
    receipt = PhaseBoundaryTopologyReceipt(
        receipt_hash=_sha256_hex(receipt_proto.to_hash_payload_dict()),
        kernel_version=receipt_proto.kernel_version,
        schema_version=receipt_proto.schema_version,
        input_transition_hash=receipt_proto.input_transition_hash,
        output_stable_hash=receipt_proto.output_stable_hash,
        region_count=receipt_proto.region_count,
        boundary_count=receipt_proto.boundary_count,
        receipt_chain=receipt_proto.receipt_chain,
        boundary_integrity_score=receipt_proto.boundary_integrity_score,
        topology_stability_score=receipt_proto.topology_stability_score,
        region_consistency_score=receipt_proto.region_consistency_score,
        boundary_continuity_score=receipt_proto.boundary_continuity_score,
        validation_passed=receipt_proto.validation_passed,
    )

    if receipt.receipt_chain[0] != effective_config.stable_sha256():
        raise ValueError("broken lineage: receipt chain root mismatch")
    if receipt.receipt_chain[1] != path.stable_hash:
        raise ValueError("broken lineage: receipt chain path mismatch")
    if receipt.receipt_chain[2] != result.stable_hash:
        raise ValueError("broken lineage: receipt chain result mismatch")

    return result, receipt


def build_ascii_phase_summary(result: PhaseBoundaryTopologyResult) -> str:
    """Build deterministic ASCII summary for phase boundary topology results."""
    return "\n".join(
        (
            f"Phase Boundary Topology Kernel — {result.schema_version}",
            f"  Regions:      {len(result.path.regions)}",
            f"  Boundaries:   {len(result.path.boundaries)}",
            f"  Integrity:    {_quantized_str(result.boundary_integrity_score, field_name='integrity')}",
            f"  Stability:    {_quantized_str(result.topology_stability_score, field_name='stability')}",
            f"  Consistency:  {_quantized_str(result.region_consistency_score, field_name='consistency')}",
            f"  Continuity:   {_quantized_str(result.boundary_continuity_score, field_name='continuity')}",
            f"  Hash:         {result.stable_hash[:16]}...",
        )
    )


__all__ = [
    "SCHEMA_VERSION",
    "PhaseBoundaryTopologyConfig",
    "PhaseRegion",
    "PhaseBoundaryEdge",
    "PhaseTopologyPath",
    "PhaseBoundaryTopologyResult",
    "PhaseBoundaryTopologyReceipt",
    "detect_phase_regions",
    "build_phase_topology_path",
    "compute_phase_similarity",
    "run_phase_boundary_topology_kernel",
    "build_ascii_phase_summary",
]
