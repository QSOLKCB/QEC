"""v137.13.3 — Region Correspondence Kernel.

Deterministic Layer-4 cross-path region alignment and correspondence scoring
for PhaseTopologyPath artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import json
import math
from typing import Any, Mapping

from qec.analysis.phase_boundary_topology_kernel import (
    SCHEMA_VERSION as PHASE_TOPOLOGY_SCHEMA_VERSION,
    PhaseTopologyPath,
)

SCHEMA_VERSION = "v137.13.3"
_DECIMAL_PLACES = Decimal("0.000000000001")

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]


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


def _is_sha256_hex(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _region_span_overlap(source_start: int, source_end: int, target_start: int, target_end: int) -> float:
    left = max(source_start, target_start)
    right = min(source_end, target_end)
    intersection = max(0, right - left + 1)
    source_len = source_end - source_start + 1
    target_len = target_end - target_start + 1
    union = source_len + target_len - intersection
    if union <= 0:
        return 0.0
    return float(intersection) / float(union)


def _compute_region_boundary_coherence(path: PhaseTopologyPath, region_index: int) -> float:
    continuity_values: list[float] = []
    if region_index > 0:
        boundary = path.boundaries[region_index - 1]
        continuity_values.append(abs(boundary.continuity_delta))
        continuity_values.append(abs(boundary.morphology_delta))
    if region_index < len(path.boundaries):
        boundary = path.boundaries[region_index]
        continuity_values.append(abs(boundary.continuity_delta))
        continuity_values.append(abs(boundary.morphology_delta))
    if len(continuity_values) == 0:
        return 1.0
    return 1.0 - _clamp01(sum(continuity_values) / float(len(continuity_values)))


@dataclass(frozen=True)
class RegionCorrespondenceConfig:
    schema_version: str = SCHEMA_VERSION
    kernel_version: str = SCHEMA_VERSION
    label_match_weight: float = 0.400000000000
    morphology_weight: float = 0.300000000000
    span_overlap_weight: float = 0.200000000000
    boundary_coherence_weight: float = 0.100000000000

    def __post_init__(self) -> None:
        for field_name in (
            "label_match_weight",
            "morphology_weight",
            "span_overlap_weight",
            "boundary_coherence_weight",
        ):
            value = getattr(self, field_name)
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{field_name} must be in [0, 1]")
        weight_sum = (
            self.label_match_weight
            + self.morphology_weight
            + self.span_overlap_weight
            + self.boundary_coherence_weight
        )
        if _quantize(weight_sum, field_name="weight_sum") != 1.0:
            raise ValueError("weights must sum to 1.0")

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "kernel_version": self.kernel_version,
            "label_match_weight": _quantized_str(self.label_match_weight, field_name="label_match_weight"),
            "morphology_weight": _quantized_str(self.morphology_weight, field_name="morphology_weight"),
            "span_overlap_weight": _quantized_str(self.span_overlap_weight, field_name="span_overlap_weight"),
            "boundary_coherence_weight": _quantized_str(
                self.boundary_coherence_weight,
                field_name="boundary_coherence_weight",
            ),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_sha256(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class RegionCorrespondencePair:
    pair_id: str
    source_path_index: int
    target_path_index: int
    source_region_index: int
    target_region_index: int
    source_region_id: str
    target_region_id: str
    source_region_label: str
    target_region_label: str
    region_alignment_score: float
    topology_correspondence_score: float
    boundary_coherence_score: float
    global_correspondence_score: float

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "pair_id": self.pair_id,
            "source_path_index": self.source_path_index,
            "target_path_index": self.target_path_index,
            "source_region_index": self.source_region_index,
            "target_region_index": self.target_region_index,
            "source_region_id": self.source_region_id,
            "target_region_id": self.target_region_id,
            "source_region_label": self.source_region_label,
            "target_region_label": self.target_region_label,
            "region_alignment_score": _quantized_str(
                self.region_alignment_score,
                field_name="region_alignment_score",
            ),
            "topology_correspondence_score": _quantized_str(
                self.topology_correspondence_score,
                field_name="topology_correspondence_score",
            ),
            "boundary_coherence_score": _quantized_str(
                self.boundary_coherence_score,
                field_name="boundary_coherence_score",
            ),
            "global_correspondence_score": _quantized_str(
                self.global_correspondence_score,
                field_name="global_correspondence_score",
            ),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_sha256(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class RegionCorrespondenceMap:
    source_path_index: int
    target_path_index: int
    pairs: tuple[RegionCorrespondencePair, ...]
    region_alignment_score: float
    topology_correspondence_score: float
    boundary_coherence_score: float
    global_correspondence_score: float
    stable_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "source_path_index": self.source_path_index,
            "target_path_index": self.target_path_index,
            "pairs": tuple(pair.to_dict() for pair in self.pairs),
            "region_alignment_score": _quantized_str(self.region_alignment_score, field_name="region_alignment_score"),
            "topology_correspondence_score": _quantized_str(
                self.topology_correspondence_score,
                field_name="topology_correspondence_score",
            ),
            "boundary_coherence_score": _quantized_str(
                self.boundary_coherence_score,
                field_name="boundary_coherence_score",
            ),
            "global_correspondence_score": _quantized_str(
                self.global_correspondence_score,
                field_name="global_correspondence_score",
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

    def stable_sha256(self) -> str:
        return self.stable_hash


@dataclass(frozen=True)
class RegionCorrespondenceResult:
    config: RegionCorrespondenceConfig
    correspondence_maps: tuple[RegionCorrespondenceMap, ...]
    path_hashes: tuple[str, ...]
    region_alignment_score: float
    topology_correspondence_score: float
    boundary_coherence_score: float
    global_correspondence_score: float
    stable_hash: str
    schema_version: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "config": self.config.to_dict(),
            "correspondence_maps": tuple(map_.to_dict() for map_ in self.correspondence_maps),
            "path_hashes": self.path_hashes,
            "region_alignment_score": _quantized_str(self.region_alignment_score, field_name="region_alignment_score"),
            "topology_correspondence_score": _quantized_str(
                self.topology_correspondence_score,
                field_name="topology_correspondence_score",
            ),
            "boundary_coherence_score": _quantized_str(
                self.boundary_coherence_score,
                field_name="boundary_coherence_score",
            ),
            "global_correspondence_score": _quantized_str(
                self.global_correspondence_score,
                field_name="global_correspondence_score",
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
class RegionCorrespondenceReceipt:
    receipt_hash: str
    kernel_version: str
    schema_version: str
    input_path_hashes: tuple[str, ...]
    output_stable_hash: str
    map_count: int
    pair_count: int
    receipt_chain: tuple[str, ...]
    region_alignment_score: float
    topology_correspondence_score: float
    boundary_coherence_score: float
    global_correspondence_score: float
    validation_passed: bool

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "receipt_hash": self.receipt_hash,
            "kernel_version": self.kernel_version,
            "schema_version": self.schema_version,
            "input_path_hashes": self.input_path_hashes,
            "output_stable_hash": self.output_stable_hash,
            "map_count": self.map_count,
            "pair_count": self.pair_count,
            "receipt_chain": self.receipt_chain,
            "region_alignment_score": _quantized_str(self.region_alignment_score, field_name="region_alignment_score"),
            "topology_correspondence_score": _quantized_str(
                self.topology_correspondence_score,
                field_name="topology_correspondence_score",
            ),
            "boundary_coherence_score": _quantized_str(
                self.boundary_coherence_score,
                field_name="boundary_coherence_score",
            ),
            "global_correspondence_score": _quantized_str(
                self.global_correspondence_score,
                field_name="global_correspondence_score",
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


def _validate_config(config: RegionCorrespondenceConfig) -> RegionCorrespondenceConfig:
    if config.schema_version != SCHEMA_VERSION:
        raise ValueError("schema mismatch")
    if config.kernel_version != SCHEMA_VERSION:
        raise ValueError("schema mismatch")
    return config


def _validate_phase_topology_path(path: PhaseTopologyPath) -> None:
    if path.schema_version != PHASE_TOPOLOGY_SCHEMA_VERSION:
        raise ValueError("schema mismatch")
    if len(path.regions) == 0:
        raise ValueError("empty path")
    if path.stable_hash != _sha256_hex(path.to_hash_payload_dict()):
        raise ValueError("broken lineage")
    if not _is_sha256_hex(path.stable_hash):
        raise ValueError("broken lineage")

    prior_start = -1
    region_ids: set[str] = set()
    for region in path.regions:
        if region.region_id in region_ids:
            raise ValueError("schema mismatch")
        region_ids.add(region.region_id)
        if region.source_start_index <= prior_start or region.source_end_index < region.source_start_index:
            raise ValueError("invalid ordering")
        prior_start = region.source_start_index
        for value in (region.region_score, region.continuity_mean, region.morphology_mean):
            if not math.isfinite(value):
                raise ValueError("non-finite metrics")

    if len(path.boundaries) != max(0, len(path.regions) - 1):
        raise ValueError("invalid ordering")
    for i, boundary in enumerate(path.boundaries):
        if boundary.source_region_index != i or boundary.target_region_index != i + 1:
            raise ValueError("invalid ordering")
        for value in (boundary.boundary_magnitude, boundary.continuity_delta, boundary.morphology_delta):
            if not math.isfinite(value):
                raise ValueError("non-finite metrics")


def _candidate_scores(
    source_path: PhaseTopologyPath,
    target_path: PhaseTopologyPath,
    source_index: int,
    target_index: int,
    config: RegionCorrespondenceConfig,
) -> tuple[float, float, float, float]:
    source_region = source_path.regions[source_index]
    target_region = target_path.regions[target_index]

    label_score = 1.0 if source_region.region_label == target_region.region_label else 0.0
    morphology_score = 1.0 - _clamp01(abs(source_region.morphology_mean - target_region.morphology_mean))
    span_score = _region_span_overlap(
        source_region.source_start_index,
        source_region.source_end_index,
        target_region.source_start_index,
        target_region.source_end_index,
    )
    source_boundary = _compute_region_boundary_coherence(source_path, source_index)
    target_boundary = _compute_region_boundary_coherence(target_path, target_index)
    boundary_score = 1.0 - _clamp01(abs(source_boundary - target_boundary))

    region_alignment = _clamp01(
        (config.label_match_weight * label_score)
        + (config.morphology_weight * morphology_score)
        + (config.span_overlap_weight * span_score)
        + (config.boundary_coherence_weight * boundary_score)
    )
    topology_correspondence = _clamp01((0.6 * label_score) + (0.4 * span_score))
    boundary_coherence = _clamp01(boundary_score)
    global_score = _clamp01(
        (0.5 * region_alignment) + (0.3 * topology_correspondence) + (0.2 * boundary_coherence)
    )

    return (
        _quantize_unit(region_alignment, field_name="region_alignment_score"),
        _quantize_unit(topology_correspondence, field_name="topology_correspondence_score"),
        _quantize_unit(boundary_coherence, field_name="boundary_coherence_score"),
        _quantize_unit(global_score, field_name="global_correspondence_score"),
    )


def _validate_correspondence_map(correspondence_map: RegionCorrespondenceMap) -> None:
    if correspondence_map.stable_hash != _sha256_hex(correspondence_map.to_hash_payload_dict()):
        raise ValueError("broken lineage")
    if len(correspondence_map.pairs) == 0:
        raise ValueError("empty path")
    if correspondence_map.source_path_index < 0 or correspondence_map.target_path_index <= correspondence_map.source_path_index:
        raise ValueError("invalid ordering")

    source_indices = tuple(pair.source_region_index for pair in correspondence_map.pairs)
    target_indices = tuple(pair.target_region_index for pair in correspondence_map.pairs)
    if len(set(source_indices)) != len(source_indices) or len(set(target_indices)) != len(target_indices):
        raise ValueError("duplicate region mappings")

    previous_source = -1
    for pair in correspondence_map.pairs:
        if pair.source_region_index <= previous_source:
            raise ValueError("invalid ordering")
        previous_source = pair.source_region_index
        if pair.source_path_index != correspondence_map.source_path_index:
            raise ValueError("schema mismatch")
        if pair.target_path_index != correspondence_map.target_path_index:
            raise ValueError("schema mismatch")
        for value in (
            pair.region_alignment_score,
            pair.topology_correspondence_score,
            pair.boundary_coherence_score,
            pair.global_correspondence_score,
        ):
            if not math.isfinite(value):
                raise ValueError("non-finite metrics")
            if value < 0.0 or value > 1.0:
                raise ValueError("non-finite metrics")


def build_region_correspondence_map(
    source_path: PhaseTopologyPath,
    target_path: PhaseTopologyPath,
    *,
    source_path_index: int = 0,
    target_path_index: int = 1,
    config: RegionCorrespondenceConfig | None = None,
) -> RegionCorrespondenceMap:
    """Build a deterministic one-to-one region correspondence map."""
    effective_config = _validate_config(config or RegionCorrespondenceConfig())
    _validate_phase_topology_path(source_path)
    _validate_phase_topology_path(target_path)

    if source_path_index < 0 or target_path_index < 0 or source_path_index >= target_path_index:
        raise ValueError("invalid ordering")

    unmatched_target_indices = set(range(len(target_path.regions)))
    pairs: list[RegionCorrespondencePair] = []

    for source_region_index in range(len(source_path.regions)):
        if len(unmatched_target_indices) == 0:
            break

        scored_candidates: list[tuple[float, int, str, tuple[float, float, float, float]]] = []
        for target_region_index in unmatched_target_indices:
            scores = _candidate_scores(
                source_path,
                target_path,
                source_region_index,
                target_region_index,
                effective_config,
            )
            target_label = target_path.regions[target_region_index].region_label
            scored_candidates.append((scores[3], target_region_index, target_label, scores))

        # tie-break: highest score, then lowest target index, then lexicographic label
        scored_candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        _, target_region_index, _, scores = scored_candidates[0]
        unmatched_target_indices.remove(target_region_index)

        source_region = source_path.regions[source_region_index]
        target_region = target_path.regions[target_region_index]
        pair_seed = {
            "source_path_index": source_path_index,
            "target_path_index": target_path_index,
            "source_region_index": source_region_index,
            "target_region_index": target_region_index,
            "source_region_id": source_region.region_id,
            "target_region_id": target_region.region_id,
            "scores": scores,
        }
        pairs.append(
            RegionCorrespondencePair(
                pair_id=_sha256_hex(pair_seed),
                source_path_index=source_path_index,
                target_path_index=target_path_index,
                source_region_index=source_region_index,
                target_region_index=target_region_index,
                source_region_id=source_region.region_id,
                target_region_id=target_region.region_id,
                source_region_label=source_region.region_label,
                target_region_label=target_region.region_label,
                region_alignment_score=scores[0],
                topology_correspondence_score=scores[1],
                boundary_coherence_score=scores[2],
                global_correspondence_score=scores[3],
            )
        )

    source_indices = tuple(pair.source_region_index for pair in pairs)
    target_indices = tuple(pair.target_region_index for pair in pairs)
    if len(set(source_indices)) != len(source_indices) or len(set(target_indices)) != len(target_indices):
        raise ValueError("duplicate region mappings")

    if len(pairs) == 0:
        raise ValueError("empty path")

    region_alignment = _quantize_unit(
        sum(pair.region_alignment_score for pair in pairs) / float(len(pairs)),
        field_name="region_alignment_score",
    )
    topology_correspondence = _quantize_unit(
        sum(pair.topology_correspondence_score for pair in pairs) / float(len(pairs)),
        field_name="topology_correspondence_score",
    )
    boundary_coherence = _quantize_unit(
        sum(pair.boundary_coherence_score for pair in pairs) / float(len(pairs)),
        field_name="boundary_coherence_score",
    )
    global_correspondence = _quantize_unit(
        sum(pair.global_correspondence_score for pair in pairs) / float(len(pairs)),
        field_name="global_correspondence_score",
    )

    proto = RegionCorrespondenceMap(
        source_path_index=source_path_index,
        target_path_index=target_path_index,
        pairs=tuple(pairs),
        region_alignment_score=region_alignment,
        topology_correspondence_score=topology_correspondence,
        boundary_coherence_score=boundary_coherence,
        global_correspondence_score=global_correspondence,
        stable_hash="",
    )
    return RegionCorrespondenceMap(
        source_path_index=proto.source_path_index,
        target_path_index=proto.target_path_index,
        pairs=proto.pairs,
        region_alignment_score=proto.region_alignment_score,
        topology_correspondence_score=proto.topology_correspondence_score,
        boundary_coherence_score=proto.boundary_coherence_score,
        global_correspondence_score=proto.global_correspondence_score,
        stable_hash=_sha256_hex(proto.to_hash_payload_dict()),
    )


def compute_region_correspondence_similarity(
    correspondence_map: RegionCorrespondenceMap,
) -> dict[str, float]:
    """Return bounded map-level similarity metrics."""
    _validate_correspondence_map(correspondence_map)

    return {
        "region_alignment_score": _quantize_unit(
            correspondence_map.region_alignment_score,
            field_name="region_alignment_score",
        ),
        "topology_correspondence_score": _quantize_unit(
            correspondence_map.topology_correspondence_score,
            field_name="topology_correspondence_score",
        ),
        "boundary_coherence_score": _quantize_unit(
            correspondence_map.boundary_coherence_score,
            field_name="boundary_coherence_score",
        ),
        "global_correspondence_score": _quantize_unit(
            correspondence_map.global_correspondence_score,
            field_name="global_correspondence_score",
        ),
    }


def run_region_correspondence_kernel(
    paths: tuple[PhaseTopologyPath, ...],
    config: RegionCorrespondenceConfig | None = None,
) -> tuple[RegionCorrespondenceResult, RegionCorrespondenceReceipt]:
    """Run deterministic cross-path correspondence over two or more paths."""
    effective_config = _validate_config(config or RegionCorrespondenceConfig())
    if len(paths) < 2:
        raise ValueError("empty path")

    for path in paths:
        _validate_phase_topology_path(path)

    maps: list[RegionCorrespondenceMap] = []
    for i in range(len(paths) - 1):
        for j in range(i + 1, len(paths)):
            maps.append(
                build_region_correspondence_map(
                    paths[i],
                    paths[j],
                    source_path_index=i,
                    target_path_index=j,
                    config=effective_config,
                )
            )

    path_hashes = tuple(path.stable_hash for path in paths)

    region_alignment = _quantize_unit(
        sum(item.region_alignment_score for item in maps) / float(len(maps)),
        field_name="region_alignment_score",
    )
    topology_correspondence = _quantize_unit(
        sum(item.topology_correspondence_score for item in maps) / float(len(maps)),
        field_name="topology_correspondence_score",
    )
    boundary_coherence = _quantize_unit(
        sum(item.boundary_coherence_score for item in maps) / float(len(maps)),
        field_name="boundary_coherence_score",
    )
    global_correspondence = _quantize_unit(
        sum(item.global_correspondence_score for item in maps) / float(len(maps)),
        field_name="global_correspondence_score",
    )

    result_proto = RegionCorrespondenceResult(
        config=effective_config,
        correspondence_maps=tuple(maps),
        path_hashes=path_hashes,
        region_alignment_score=region_alignment,
        topology_correspondence_score=topology_correspondence,
        boundary_coherence_score=boundary_coherence,
        global_correspondence_score=global_correspondence,
        stable_hash="",
        schema_version=effective_config.schema_version,
    )
    result = RegionCorrespondenceResult(
        config=result_proto.config,
        correspondence_maps=result_proto.correspondence_maps,
        path_hashes=result_proto.path_hashes,
        region_alignment_score=result_proto.region_alignment_score,
        topology_correspondence_score=result_proto.topology_correspondence_score,
        boundary_coherence_score=result_proto.boundary_coherence_score,
        global_correspondence_score=result_proto.global_correspondence_score,
        stable_hash=_sha256_hex(result_proto.to_hash_payload_dict()),
        schema_version=result_proto.schema_version,
    )

    chain_seed = (
        effective_config.stable_sha256(),
        _sha256_hex(path_hashes),
        result.stable_hash,
    )
    chain_tip = _sha256_hex({"receipt_chain_seed": chain_seed})
    receipt_chain = chain_seed + (chain_tip,)

    receipt_proto = RegionCorrespondenceReceipt(
        receipt_hash="",
        kernel_version=effective_config.kernel_version,
        schema_version=effective_config.schema_version,
        input_path_hashes=path_hashes,
        output_stable_hash=result.stable_hash,
        map_count=len(result.correspondence_maps),
        pair_count=sum(len(map_.pairs) for map_ in result.correspondence_maps),
        receipt_chain=receipt_chain,
        region_alignment_score=result.region_alignment_score,
        topology_correspondence_score=result.topology_correspondence_score,
        boundary_coherence_score=result.boundary_coherence_score,
        global_correspondence_score=result.global_correspondence_score,
        validation_passed=True,
    )
    receipt = RegionCorrespondenceReceipt(
        receipt_hash=_sha256_hex(receipt_proto.to_hash_payload_dict()),
        kernel_version=receipt_proto.kernel_version,
        schema_version=receipt_proto.schema_version,
        input_path_hashes=receipt_proto.input_path_hashes,
        output_stable_hash=receipt_proto.output_stable_hash,
        map_count=receipt_proto.map_count,
        pair_count=receipt_proto.pair_count,
        receipt_chain=receipt_proto.receipt_chain,
        region_alignment_score=receipt_proto.region_alignment_score,
        topology_correspondence_score=receipt_proto.topology_correspondence_score,
        boundary_coherence_score=receipt_proto.boundary_coherence_score,
        global_correspondence_score=receipt_proto.global_correspondence_score,
        validation_passed=receipt_proto.validation_passed,
    )

    if receipt.receipt_chain[0] != effective_config.stable_sha256():
        raise ValueError("broken lineage")
    if receipt.receipt_chain[2] != result.stable_hash:
        raise ValueError("broken lineage")

    return result, receipt


def build_ascii_correspondence_summary(result: RegionCorrespondenceResult) -> str:
    """Build deterministic ASCII summary for correspondence results."""
    return "\n".join(
        (
            f"Region Correspondence Kernel — {result.schema_version}",
            f"  Paths:                {len(result.path_hashes)}",
            f"  Maps:                 {len(result.correspondence_maps)}",
            f"  Alignment:            {_quantized_str(result.region_alignment_score, field_name='region_alignment_score')}",
            f"  Topology:             {_quantized_str(result.topology_correspondence_score, field_name='topology_correspondence_score')}",
            f"  Boundary Coherence:   {_quantized_str(result.boundary_coherence_score, field_name='boundary_coherence_score')}",
            f"  Global:               {_quantized_str(result.global_correspondence_score, field_name='global_correspondence_score')}",
            f"  Hash:                 {result.stable_hash[:16]}...",
        )
    )


__all__ = [
    "SCHEMA_VERSION",
    "RegionCorrespondenceConfig",
    "RegionCorrespondencePair",
    "RegionCorrespondenceMap",
    "RegionCorrespondenceResult",
    "RegionCorrespondenceReceipt",
    "build_region_correspondence_map",
    "compute_region_correspondence_similarity",
    "run_region_correspondence_kernel",
    "build_ascii_correspondence_summary",
]
