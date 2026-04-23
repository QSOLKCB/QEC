from __future__ import annotations

"""v144.1 — Deterministic Stress Lattice (QMC Coverage Layer).

Deterministic analysis-layer stress coverage kernel for offline benchmarking.
"""

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any

ROUND_DIGITS: int = 12
MAX_POINT_COUNT: int = 4096
HALTON_METHOD: str = "halton"
LATTICE_METHOD: str = "lattice"
SUPPORTED_METHODS: tuple[str, ...] = (HALTON_METHOD, LATTICE_METHOD)
PARTIAL_COVERAGE_THRESHOLD: float = 0.40
DENSE_COVERAGE_THRESHOLD: float = 0.75
_CANONICAL_AXIS_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_PRIMES: tuple[int, ...] = (
    2,
    3,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
    53,
)


def _round12(value: float) -> float:
    return float(round(float(value), ROUND_DIGITS))


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _canonical_bytes(payload: Any) -> bytes:
    return _canonical_json(payload).encode("utf-8")


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _validate_hash(value: str, *, field: str) -> str:
    token = str(value)
    if len(token) != 64 or any(ch not in "0123456789abcdef" for ch in token):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return token


def _validate_axis_name(name: str) -> str:
    token = str(name)
    if not _CANONICAL_AXIS_NAME_PATTERN.fullmatch(token):
        raise ValueError("axis name must match ^[a-z][a-z0-9_]*$")
    return token


def _validate_bound(value: float, *, field: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field} must be finite")
    if numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{field} must be within [0.0, 1.0]")
    return numeric


def _halton_value(index: int, base: int) -> float:
    if index < 1:
        raise ValueError("halton index must be >= 1")
    result = 0.0
    fraction = 1.0 / float(base)
    n = index
    while n > 0:
        n, remainder = divmod(n, base)
        result += fraction * float(remainder)
        fraction /= float(base)
    return result


@dataclass(frozen=True)
class StressAxis:
    name: str
    lower_bound: float
    upper_bound: float

    def __post_init__(self) -> None:
        canonical_name = _validate_axis_name(self.name)
        lower = _validate_bound(self.lower_bound, field="lower_bound")
        upper = _validate_bound(self.upper_bound, field="upper_bound")
        if lower > upper:
            raise ValueError("lower_bound must be <= upper_bound")
        object.__setattr__(self, "name", canonical_name)
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "lower_bound": _round12(self.lower_bound),
            "upper_bound": _round12(self.upper_bound),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())


@dataclass(frozen=True)
class StressPoint:
    coordinates: dict[str, float]
    point_index: int
    signature: str
    stable_hash: str

    def __post_init__(self) -> None:
        if int(self.point_index) < 0:
            raise ValueError("point_index must be >= 0")
        if not isinstance(self.coordinates, dict) or not self.coordinates:
            raise ValueError("coordinates must be a non-empty mapping")

        normalized: dict[str, float] = {}
        for name in sorted(self.coordinates.keys()):
            canonical_name = _validate_axis_name(name)
            value = float(self.coordinates[name])
            if not math.isfinite(value):
                raise ValueError("coordinates must contain finite values")
            if value < 0.0 or value > 1.0:
                raise ValueError("coordinate values must be in [0.0, 1.0]")
            normalized[canonical_name] = value

        expected_signature = "|".join(
            f"{axis}={_round12(normalized[axis]):.{ROUND_DIGITS}f}" for axis in sorted(normalized.keys())
        )
        if self.signature != expected_signature:
            raise ValueError("signature mismatch")

        expected_hash = _sha256_hex(self.to_hash_payload_dict())
        if _validate_hash(self.stable_hash, field="stable_hash") != expected_hash:
            raise ValueError("stable_hash mismatch")

        object.__setattr__(self, "coordinates", normalized)
        object.__setattr__(self, "point_index", int(self.point_index))

    def to_hash_payload_dict(self) -> dict[str, Any]:
        return {
            "coordinates": {key: _round12(self.coordinates[key]) for key in sorted(self.coordinates.keys())},
            "point_index": int(self.point_index),
            "signature": self.signature,
        }

    def computed_stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = self.to_hash_payload_dict()
        payload["stable_hash"] = self.stable_hash
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())


@dataclass(frozen=True)
class StressCoverageReceipt:
    axis_names: list[str]
    point_count: int
    method: str
    points: list[StressPoint]
    min_per_axis: dict[str, float]
    max_per_axis: dict[str, float]
    mean_per_axis: dict[str, float]
    coverage_score: float
    classification: str
    stable_hash: str

    def __post_init__(self) -> None:
        axis_names = list(self.axis_names)
        if not axis_names:
            raise ValueError("axis_names must not be empty")
        canonical_axis_names = sorted(_validate_axis_name(name) for name in axis_names)
        if axis_names != canonical_axis_names:
            raise ValueError("axis_names must be sorted canonically")
        if len(set(canonical_axis_names)) != len(canonical_axis_names):
            raise ValueError("axis_names must be unique")

        if self.method not in SUPPORTED_METHODS:
            raise ValueError(f"method must be one of {SUPPORTED_METHODS}")

        if int(self.point_count) < 1:
            raise ValueError("point_count must be >= 1")
        if int(self.point_count) > MAX_POINT_COUNT:
            raise ValueError(f"point_count must be <= {MAX_POINT_COUNT}")

        if len(self.points) != int(self.point_count):
            raise ValueError("points length must equal point_count")
        for idx, point in enumerate(self.points):
            if point.point_index != idx:
                raise ValueError("points must be ordered by point_index")
            if sorted(point.coordinates.keys()) != canonical_axis_names:
                raise ValueError("point coordinates must exactly match axis_names")

        for metric_name, metric in (
            ("min_per_axis", self.min_per_axis),
            ("max_per_axis", self.max_per_axis),
            ("mean_per_axis", self.mean_per_axis),
        ):
            if sorted(metric.keys()) != canonical_axis_names:
                raise ValueError(f"{metric_name} keys must match axis_names")
            for axis in canonical_axis_names:
                value = float(metric[axis])
                if not math.isfinite(value):
                    raise ValueError(f"{metric_name}[{axis}] must be finite")
                if value < 0.0 or value > 1.0:
                    raise ValueError(f"{metric_name}[{axis}] must be in [0.0, 1.0]")

        coverage = float(self.coverage_score)
        if not math.isfinite(coverage) or coverage < 0.0 or coverage > 1.0:
            raise ValueError("coverage_score must be in [0.0, 1.0]")

        if self.classification not in {"sparse", "partial", "dense"}:
            raise ValueError("classification must be one of: sparse, partial, dense")

        expected_hash = _sha256_hex(self.to_hash_payload_dict())
        if _validate_hash(self.stable_hash, field="stable_hash") != expected_hash:
            raise ValueError("stable_hash mismatch")

    def to_hash_payload_dict(self) -> dict[str, Any]:
        return {
            "axis_names": list(self.axis_names),
            "point_count": int(self.point_count),
            "method": self.method,
            "points": [point.to_dict() for point in self.points],
            "min_per_axis": {key: _round12(self.min_per_axis[key]) for key in self.axis_names},
            "max_per_axis": {key: _round12(self.max_per_axis[key]) for key in self.axis_names},
            "mean_per_axis": {key: _round12(self.mean_per_axis[key]) for key in self.axis_names},
            "coverage_score": _round12(self.coverage_score),
            "classification": self.classification,
        }

    def computed_stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = self.to_hash_payload_dict()
        payload["stable_hash"] = self.stable_hash
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())


def generate_halton_points(axes: list[StressAxis], point_count: int) -> list[dict[str, float]]:
    if len(axes) > len(_PRIMES):
        raise ValueError(f"halton supports at most {len(_PRIMES)} axes")

    points: list[dict[str, float]] = []
    for point_index in range(point_count):
        coordinate_map: dict[str, float] = {}
        for axis_index, axis in enumerate(axes):
            u = _halton_value(point_index + 1, _PRIMES[axis_index])
            span = axis.upper_bound - axis.lower_bound
            coordinate_map[axis.name] = axis.lower_bound + (span * u)
        points.append(coordinate_map)
    return points


def generate_lattice_points(axes: list[StressAxis], point_count: int) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    for point_index in range(point_count):
        coordinate_map: dict[str, float] = {}
        for axis_index, axis in enumerate(axes):
            stride = (2 * axis_index) + 1
            numerator = ((point_index * stride) % point_count) + 0.5
            u = numerator / float(point_count)
            span = axis.upper_bound - axis.lower_bound
            coordinate_map[axis.name] = axis.lower_bound + (span * u)
        points.append(coordinate_map)
    return points


def _classify_coverage(score: float) -> str:
    if score >= DENSE_COVERAGE_THRESHOLD:
        return "dense"
    if score >= PARTIAL_COVERAGE_THRESHOLD:
        return "partial"
    return "sparse"


def generate_stress_lattice(
    axes: list[StressAxis],
    point_count: int,
    method: str = HALTON_METHOD,
) -> StressCoverageReceipt:
    if not axes:
        raise ValueError("axes must not be empty")

    bounded_point_count = int(point_count)
    if bounded_point_count < 1:
        raise ValueError("point_count must be >= 1")
    if bounded_point_count > MAX_POINT_COUNT:
        raise ValueError(f"point_count must be <= {MAX_POINT_COUNT}")

    canonical_axes = sorted(axes, key=lambda axis: axis.name)
    axis_names = [axis.name for axis in canonical_axes]
    if len(set(axis_names)) != len(axis_names):
        raise ValueError("axis names must be unique")

    if method not in SUPPORTED_METHODS:
        raise ValueError(f"unsupported method: {method}")

    raw_points = (
        generate_halton_points(canonical_axes, bounded_point_count)
        if method == HALTON_METHOD
        else generate_lattice_points(canonical_axes, bounded_point_count)
    )

    points: list[StressPoint] = []
    for point_index, coordinate_map in enumerate(raw_points):
        signature = "|".join(
            f"{axis}={_round12(coordinate_map[axis]):.{ROUND_DIGITS}f}" for axis in axis_names
        )
        coordinates = {axis: float(coordinate_map[axis]) for axis in axis_names}
        point_payload = {
            "coordinates": {key: _round12(coordinates[key]) for key in axis_names},
            "point_index": int(point_index),
            "signature": signature,
        }
        points.append(
            StressPoint(
                coordinates=coordinates,
                point_index=point_index,
                signature=signature,
                stable_hash=_sha256_hex(point_payload),
            )
        )

    min_per_axis: dict[str, float] = {}
    max_per_axis: dict[str, float] = {}
    mean_per_axis: dict[str, float] = {}
    span_scores: list[float] = []

    for axis in canonical_axes:
        values = [point.coordinates[axis.name] for point in points]
        axis_min = min(values)
        axis_max = max(values)
        axis_mean = sum(values) / float(len(values))
        min_per_axis[axis.name] = _round12(axis_min)
        max_per_axis[axis.name] = _round12(axis_max)
        mean_per_axis[axis.name] = _round12(axis_mean)

        span = axis.upper_bound - axis.lower_bound
        normalized_span = 1.0 if span == 0.0 else max(0.0, min(1.0, (axis_max - axis_min) / span))
        span_scores.append(normalized_span)

    coverage_score = _round12(sum(span_scores) / float(len(span_scores)))
    classification = _classify_coverage(coverage_score)

    payload_without_hash = {
        "axis_names": list(axis_names),
        "point_count": int(bounded_point_count),
        "method": method,
        "points": [point.to_dict() for point in points],
        "min_per_axis": {key: _round12(min_per_axis[key]) for key in axis_names},
        "max_per_axis": {key: _round12(max_per_axis[key]) for key in axis_names},
        "mean_per_axis": {key: _round12(mean_per_axis[key]) for key in axis_names},
        "coverage_score": _round12(coverage_score),
        "classification": classification,
    }
    stable_hash = _sha256_hex(payload_without_hash)

    return StressCoverageReceipt(
        axis_names=axis_names,
        point_count=bounded_point_count,
        method=method,
        points=points,
        min_per_axis=min_per_axis,
        max_per_axis=max_per_axis,
        mean_per_axis=mean_per_axis,
        coverage_score=coverage_score,
        classification=classification,
        stable_hash=stable_hash,
    )
