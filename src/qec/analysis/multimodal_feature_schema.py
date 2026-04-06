"""v137.9.0 — Multimodal Feature Schema.

Deterministic Layer-4 schema envelope built from correspondence artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Any, Mapping

from qec.analysis.arithmetic_topology_correspondence_engine import ArithmeticTopologyCorrespondenceResult

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

_FEATURE_SCHEMA_VERSION = 1
_NAMESPACE_SCHEMA_VERSION = 1

MULTIMODAL_FEATURE_SCHEMA_LAW = "MULTIMODAL_FEATURE_SCHEMA_LAW"
DETERMINISTIC_FEATURE_ORDERING_RULE = "DETERMINISTIC_FEATURE_ORDERING_RULE"
REPLAY_SAFE_SCHEMA_IDENTITY_RULE = "REPLAY_SAFE_SCHEMA_IDENTITY_RULE"
BOUNDED_SCHEMA_SCORE_RULE = "BOUNDED_SCHEMA_SCORE_RULE"


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


def _canonical_bytes(value: Any) -> bytes:
    return _canonical_json(value).encode("utf-8")


def _sha256_hex(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


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
        raise ValueError(f"{name} must be finite number in [0.0, 1.0]")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{name} must be finite number in [0.0, 1.0]")


def _normalize_feature_value(value: str | int | float) -> str | int | float:
    if isinstance(value, bool):
        raise ValueError("feature values must not be bool")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("feature float values must be finite")
        return float(value)
    if isinstance(value, str):
        return value
    raise ValueError("feature values must be str, int, or float")


def _validate_correspondence_artifact(artifact: ArithmeticTopologyCorrespondenceResult) -> None:
    if not isinstance(artifact, ArithmeticTopologyCorrespondenceResult):
        raise ValueError("correspondence_artifact must be an ArithmeticTopologyCorrespondenceResult")
    if artifact.stable_hash() != artifact.correspondence_hash:
        raise ValueError("correspondence_artifact correspondence_hash must match stable_hash")


def _collect_base_features(correspondence_artifact: ArithmeticTopologyCorrespondenceResult) -> tuple[tuple[str, str, str | int | float], ...]:
    features: list[tuple[str, str, str | int | float]] = [
        ("correspondence", "witness_count", int(correspondence_artifact.witness_count)),
        ("correspondence", "primitive_count", int(correspondence_artifact.primitive_count)),
        ("correspondence", "witness_consistency_score", float(correspondence_artifact.witness_consistency_score)),
        (
            "correspondence",
            "split_arithmetic_alignment_score",
            float(correspondence_artifact.split_arithmetic_alignment_score),
        ),
        (
            "correspondence",
            "divergence_mapping_integrity_score",
            float(correspondence_artifact.divergence_mapping_integrity_score),
        ),
        (
            "correspondence",
            "topology_arithmetic_coherence_score",
            float(correspondence_artifact.topology_arithmetic_coherence_score),
        ),
        ("correspondence", "overall_correspondence_score", float(correspondence_artifact.overall_correspondence_score)),
    ]

    for witness in correspondence_artifact.witnesses:
        prefix = f"witness.{witness.witness_index:06d}"
        features.extend(
            (
                ("witness", f"{prefix}.scenario_id", witness.scenario_id),
                ("witness", f"{prefix}.anchor_node_id", witness.anchor_node_id),
                ("witness", f"{prefix}.path_count", int(witness.path_count)),
                ("witness", f"{prefix}.split_count", int(witness.split_count)),
                ("witness", f"{prefix}.segment_count", int(witness.segment_count)),
                ("witness", f"{prefix}.arithmetic_mass", float(witness.arithmetic_mass)),
                ("witness", f"{prefix}.divergence_complement_score", float(witness.divergence_complement_score)),
                ("witness", f"{prefix}.witness_consistency_score", float(witness.witness_consistency_score)),
            )
        )

    for primitive in correspondence_artifact.primitives:
        prefix = f"primitive.{primitive.primitive_index:06d}"
        features.extend(
            (
                ("primitive", f"{prefix}.scenario_id", primitive.scenario_id),
                ("primitive", f"{prefix}.segment_id", primitive.segment_id),
                ("primitive", f"{prefix}.path_id", primitive.path_id),
                ("primitive", f"{prefix}.arithmetic_step", int(primitive.arithmetic_step)),
                ("primitive", f"{prefix}.split_pressure_score", float(primitive.split_pressure_score)),
                ("primitive", f"{prefix}.segment_divergence_score", float(primitive.segment_divergence_score)),
                ("primitive", f"{prefix}.arithmetic_alignment_score", float(primitive.arithmetic_alignment_score)),
            )
        )
    return tuple(features)


def _normalize_optional_payloads(
    *,
    float_payload: Mapping[str, float] | None,
    int_payload: Mapping[str, int] | None,
    str_payload: Mapping[str, str] | None,
    tuple_feature_streams: tuple[tuple[str, str, str | int | float], ...] | None,
) -> tuple[tuple[str, str, str | int | float], ...]:
    features: list[tuple[str, str, str | int | float]] = []

    for family, payload in (("float", float_payload), ("int", int_payload), ("str", str_payload)):
        if payload is None:
            continue
        if not isinstance(payload, Mapping):
            raise ValueError(f"{family}_payload must be a mapping[str, {family}]")
        for key, value in payload.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{family}_payload keys must be non-empty strings")
            if family == "float":
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("float_payload values must be finite float-compatible numbers")
                normalized_value = float(value)
                if not math.isfinite(normalized_value):
                    raise ValueError("float_payload values must be finite")
            elif family == "int":
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError("int_payload values must be int")
                normalized_value = int(value)
            else:
                if not isinstance(value, str):
                    raise ValueError("str_payload values must be str")
                normalized_value = value
            features.append((family, key, normalized_value))

    if tuple_feature_streams is not None:
        if not isinstance(tuple_feature_streams, tuple):
            raise ValueError("tuple_feature_streams must be a tuple")
        for entry in tuple_feature_streams:
            if not isinstance(entry, tuple) or len(entry) not in (2, 3):
                raise ValueError("tuple_feature_streams entries must be tuple(name, value) or tuple(family, name, value)")
            family = "tuple_stream"
            name: str
            value: str | int | float
            if len(entry) == 2:
                name_obj, value_obj = entry
                name = str(name_obj)
                value = _normalize_feature_value(value_obj)
            else:
                family_obj, name_obj, value_obj = entry
                if not isinstance(family_obj, str) or not family_obj:
                    raise ValueError("tuple feature family must be non-empty string")
                family = family_obj
                name = str(name_obj)
                value = _normalize_feature_value(value_obj)
            if not name:
                raise ValueError("tuple feature name must be non-empty string")
            features.append((family, name, value))

    return tuple(features)


@dataclass(frozen=True)
class MultimodalFeature:
    feature_name: str
    feature_family: str
    feature_namespace: str
    feature_schema_version: int
    feature_value: str | int | float
    feature_index: int

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "feature_name": self.feature_name,
            "feature_family": self.feature_family,
            "feature_namespace": self.feature_namespace,
            "feature_schema_version": self.feature_schema_version,
            "feature_value": self.feature_value,
            "feature_index": self.feature_index,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class FeatureNamespace:
    feature_namespace: str
    feature_family: str
    feature_schema_version: int
    namespace_index: int
    feature_count: int
    namespace_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "feature_namespace": self.feature_namespace,
            "feature_family": self.feature_family,
            "feature_schema_version": self.feature_schema_version,
            "namespace_index": self.namespace_index,
            "feature_count": self.feature_count,
            "namespace_hash": self.namespace_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("namespace_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class MultimodalFeatureSchemaResult:
    schema_version: int
    source_graph_hash: str
    source_polytope_hash: str
    source_symmetry_hash: str
    source_traversal_hash: str
    source_divergence_hash: str
    source_correspondence_hash: str
    feature_count: int
    namespace_count: int
    features: tuple[MultimodalFeature, ...]
    namespaces: tuple[FeatureNamespace, ...]
    schema_integrity_score: float
    namespace_consistency_score: float
    feature_ordering_score: float
    payload_normalization_score: float
    overall_schema_score: float
    law_invariants: tuple[str, ...]
    feature_schema_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "source_graph_hash": self.source_graph_hash,
            "source_polytope_hash": self.source_polytope_hash,
            "source_symmetry_hash": self.source_symmetry_hash,
            "source_traversal_hash": self.source_traversal_hash,
            "source_divergence_hash": self.source_divergence_hash,
            "source_correspondence_hash": self.source_correspondence_hash,
            "feature_count": self.feature_count,
            "namespace_count": self.namespace_count,
            "features": tuple(feature.to_dict() for feature in self.features),
            "namespaces": tuple(namespace.to_dict() for namespace in self.namespaces),
            "schema_integrity_score": self.schema_integrity_score,
            "namespace_consistency_score": self.namespace_consistency_score,
            "feature_ordering_score": self.feature_ordering_score,
            "payload_normalization_score": self.payload_normalization_score,
            "overall_schema_score": self.overall_schema_score,
            "law_invariants": self.law_invariants,
            "feature_schema_hash": self.feature_schema_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("feature_schema_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class MultimodalFeatureSchemaReceipt:
    schema_version: int
    source_graph_hash: str
    source_polytope_hash: str
    source_symmetry_hash: str
    source_traversal_hash: str
    source_divergence_hash: str
    source_correspondence_hash: str
    feature_schema_hash: str
    feature_count: int
    namespace_count: int
    overall_schema_score: float
    receipt_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "source_graph_hash": self.source_graph_hash,
            "source_polytope_hash": self.source_polytope_hash,
            "source_symmetry_hash": self.source_symmetry_hash,
            "source_traversal_hash": self.source_traversal_hash,
            "source_divergence_hash": self.source_divergence_hash,
            "source_correspondence_hash": self.source_correspondence_hash,
            "feature_schema_hash": self.feature_schema_hash,
            "feature_count": self.feature_count,
            "namespace_count": self.namespace_count,
            "overall_schema_score": self.overall_schema_score,
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


def build_multimodal_feature_schema(
    correspondence_artifact: ArithmeticTopologyCorrespondenceResult,
    *,
    float_payload: Mapping[str, float] | None = None,
    int_payload: Mapping[str, int] | None = None,
    str_payload: Mapping[str, str] | None = None,
    tuple_feature_streams: tuple[tuple[str, str, str | int | float], ...] | None = None,
) -> MultimodalFeatureSchemaResult:
    _validate_correspondence_artifact(correspondence_artifact)

    base_features = _collect_base_features(correspondence_artifact)
    optional_features = _normalize_optional_payloads(
        float_payload=float_payload,
        int_payload=int_payload,
        str_payload=str_payload,
        tuple_feature_streams=tuple_feature_streams,
    )
    raw_features = base_features + optional_features

    optional_keys = tuple((family, name) for family, name, _ in optional_features)
    sorted_optional_keys = tuple(sorted(optional_keys))
    ordering_matches = sum(1 for a, b in zip(optional_keys, sorted_optional_keys) if a == b)
    feature_ordering_score = 1.0 if not optional_keys else _clamp01(float(ordering_matches / len(optional_keys)))

    canonical_raw_features = tuple(sorted(raw_features, key=lambda x: (x[0], x[1], _canonical_json(x[2]))))
    if len(set((family, name) for family, name, _ in canonical_raw_features)) != len(canonical_raw_features):
        raise ValueError("duplicate feature identity detected after canonical normalization")

    features: list[MultimodalFeature] = []
    for index, (family, name, value) in enumerate(canonical_raw_features):
        namespace = f"{family}.v{_NAMESPACE_SCHEMA_VERSION}"
        features.append(
            MultimodalFeature(
                feature_name=name,
                feature_family=family,
                feature_namespace=namespace,
                feature_schema_version=_FEATURE_SCHEMA_VERSION,
                feature_value=_normalize_feature_value(value),
                feature_index=index,
            )
        )

    namespace_counts: dict[str, int] = {}
    namespace_families: dict[str, str] = {}
    for feature in features:
        namespace_counts[feature.feature_namespace] = namespace_counts.get(feature.feature_namespace, 0) + 1
        namespace_families[feature.feature_namespace] = feature.feature_family

    namespace_names = tuple(sorted(namespace_counts.keys()))
    namespaces = tuple(
        FeatureNamespace(
            feature_namespace=name,
            feature_family=namespace_families[name],
            feature_schema_version=_NAMESPACE_SCHEMA_VERSION,
            namespace_index=index,
            feature_count=namespace_counts[name],
            namespace_hash="",
        )
        for index, name in enumerate(namespace_names)
    )
    namespaces = tuple(replace(ns, namespace_hash=ns.stable_hash()) for ns in namespaces)

    schema_integrity_score = 1.0
    namespace_consistency_score = 1.0
    for namespace in namespaces:
        if namespace.namespace_hash != namespace.stable_hash():
            raise ValueError("namespace_hash must match stable_hash")
        realized_count = sum(1 for feature in features if feature.feature_namespace == namespace.feature_namespace)
        if realized_count != namespace.feature_count:
            raise ValueError("namespace feature_count must match realized namespace members")
        namespace_consistency_score = _clamp01(
            namespace_consistency_score
            * (1.0 if realized_count == namespace.feature_count else float(realized_count / max(namespace.feature_count, 1)))
        )

    payload_normalization_score = 1.0 if len(canonical_raw_features) == len(raw_features) else _clamp01(
        float(len(canonical_raw_features) / max(len(raw_features), 1))
    )

    overall_schema_score = _mean(
        (
            schema_integrity_score,
            namespace_consistency_score,
            feature_ordering_score,
            payload_normalization_score,
        ),
        default=1.0,
    )
    for name, value in (
        ("schema_integrity_score", schema_integrity_score),
        ("namespace_consistency_score", namespace_consistency_score),
        ("feature_ordering_score", feature_ordering_score),
        ("payload_normalization_score", payload_normalization_score),
        ("overall_schema_score", overall_schema_score),
    ):
        _validate_unit_interval(value, name)

    result = MultimodalFeatureSchemaResult(
        schema_version=_FEATURE_SCHEMA_VERSION,
        source_graph_hash=correspondence_artifact.source_graph_hash,
        source_polytope_hash=correspondence_artifact.source_polytope_hash,
        source_symmetry_hash=correspondence_artifact.source_symmetry_hash,
        source_traversal_hash=correspondence_artifact.source_traversal_hash,
        source_divergence_hash=correspondence_artifact.source_divergence_hash,
        source_correspondence_hash=correspondence_artifact.correspondence_hash,
        feature_count=len(features),
        namespace_count=len(namespaces),
        features=tuple(features),
        namespaces=namespaces,
        schema_integrity_score=schema_integrity_score,
        namespace_consistency_score=namespace_consistency_score,
        feature_ordering_score=feature_ordering_score,
        payload_normalization_score=payload_normalization_score,
        overall_schema_score=overall_schema_score,
        law_invariants=(
            MULTIMODAL_FEATURE_SCHEMA_LAW,
            DETERMINISTIC_FEATURE_ORDERING_RULE,
            REPLAY_SAFE_SCHEMA_IDENTITY_RULE,
            BOUNDED_SCHEMA_SCORE_RULE,
        ),
        feature_schema_hash="",
    )
    return replace(result, feature_schema_hash=result.stable_hash())


def export_multimodal_feature_schema_bytes(artifact: MultimodalFeatureSchemaResult) -> bytes:
    if not isinstance(artifact, MultimodalFeatureSchemaResult):
        raise ValueError("artifact must be a MultimodalFeatureSchemaResult")
    return artifact.to_canonical_bytes()


def generate_multimodal_feature_schema_receipt(
    artifact: MultimodalFeatureSchemaResult,
) -> MultimodalFeatureSchemaReceipt:
    if not isinstance(artifact, MultimodalFeatureSchemaResult):
        raise ValueError("artifact must be a MultimodalFeatureSchemaResult")
    if artifact.stable_hash() != artifact.feature_schema_hash:
        raise ValueError("artifact feature_schema_hash must match stable_hash")
    receipt = MultimodalFeatureSchemaReceipt(
        schema_version=artifact.schema_version,
        source_graph_hash=artifact.source_graph_hash,
        source_polytope_hash=artifact.source_polytope_hash,
        source_symmetry_hash=artifact.source_symmetry_hash,
        source_traversal_hash=artifact.source_traversal_hash,
        source_divergence_hash=artifact.source_divergence_hash,
        source_correspondence_hash=artifact.source_correspondence_hash,
        feature_schema_hash=artifact.feature_schema_hash,
        feature_count=artifact.feature_count,
        namespace_count=artifact.namespace_count,
        overall_schema_score=artifact.overall_schema_score,
        receipt_hash="",
    )
    return replace(receipt, receipt_hash=receipt.stable_hash())


__all__ = [
    "BOUNDED_SCHEMA_SCORE_RULE",
    "DETERMINISTIC_FEATURE_ORDERING_RULE",
    "MULTIMODAL_FEATURE_SCHEMA_LAW",
    "REPLAY_SAFE_SCHEMA_IDENTITY_RULE",
    "FeatureNamespace",
    "MultimodalFeature",
    "MultimodalFeatureSchemaReceipt",
    "MultimodalFeatureSchemaResult",
    "build_multimodal_feature_schema",
    "export_multimodal_feature_schema_bytes",
    "generate_multimodal_feature_schema_receipt",
]
