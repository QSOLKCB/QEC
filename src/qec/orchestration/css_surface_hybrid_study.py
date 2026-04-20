"""v138.4.3 — deterministic CSS / surface hybrid comparative study module."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping

_RELEASE_VERSION = "v138.4.3"
_STUDY_KIND = "css_surface_hybrid_study"
_EXPECTED_SOURCE_RELEASE_VERSION = "v138.4.2"
_EXPECTED_SOURCE_EXPERIMENT_KIND = "ternary_asic_experiment_module"

_ALLOWED_CLASSIFICATIONS = (
    "css_aligned",
    "surface_favorable",
    "ternary_favorable",
    "hybrid_balanced",
    "hybrid_divergent",
)

_REQUIRED_SOURCE_METRICS = (
    "asic_compatibility_score",
    "execution_feasibility_score",
    "timing_efficiency_score",
    "power_efficiency_score",
    "thermal_stability_score",
    "memory_feasibility_score",
    "bounded_experiment_confidence",
)

_REQUIRED_PROFILE_FIELDS = (
    "pipeline_depth_class",
    "lane_parallelism_class",
    "timing_regime",
    "power_regime",
    "thermal_regime",
    "memory_pressure_class",
)

_ALLOWED_TIMING_REGIMES = ("tight", "moderate", "relaxed")
_ALLOWED_MEMORY_PRESSURE_CLASSES = ("low", "moderate", "high")


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _stable_hash(data: Any) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def _require_non_empty_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _require_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return int(value)


def _bounded(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    numeric = float(value)
    if math.isnan(numeric) or math.isinf(numeric):
        raise ValueError(f"{field} must be finite")
    if numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{field} must be within [0,1]")
    return numeric


def _canonical_mapping(value: Mapping[str, float], *, field: str, required_keys: tuple[str, ...]) -> Mapping[str, float]:
    missing = [key for key in required_keys if key not in value]
    if missing:
        raise ValueError(f"{field} missing keys: {', '.join(missing)}")
    normalized: dict[str, float] = {}
    for key in sorted(required_keys):
        normalized[key] = _bounded(value[key], field=f"{field}.{key}")
    return MappingProxyType(normalized)


def _normalize_correction(raw: Any, *, field: str) -> tuple[int, ...]:
    correction = tuple(raw)
    if len(correction) == 0:
        raise ValueError(f"{field} must be non-empty")
    normalized: list[int] = []
    for idx, value in enumerate(correction):
        symbol = _require_int(value, field=f"{field}[{idx}]")
        if symbol not in (0, 1, 2):
            raise ValueError(f"{field}[{idx}] must be one of 0, 1, 2")
        normalized.append(symbol)
    return tuple(normalized)


def _normalize_source_experiment(source_experiment_receipt: Any) -> dict[str, Any]:
    if isinstance(source_experiment_receipt, Mapping):
        source = dict(source_experiment_receipt)
    elif hasattr(source_experiment_receipt, "to_dict") and callable(source_experiment_receipt.to_dict):
        candidate = source_experiment_receipt.to_dict()
        if not isinstance(candidate, Mapping):
            raise ValueError("source_experiment_receipt.to_dict() must return a mapping")
        source = dict(candidate)
    else:
        raise ValueError("source_experiment_receipt must be mapping-compatible")

    required_fields = (
        "release_version",
        "experiment_kind",
        "source_dispatch_release_version",
        "source_dispatch_kind",
        "source_dispatch_receipt_hash",
        "source_lane_receipt_hash",
        "source_selected_target",
        "canonical_selected_correction",
        "correction_length",
        "execution_profile",
        "metric_bundle",
        "advisory_only",
        "hardware_execution_performed",
        "decoder_core_modified",
        "receipt_hash",
    )
    missing = [key for key in required_fields if key not in source]
    if missing:
        raise ValueError(f"source_experiment_receipt missing required fields: {', '.join(missing)}")

    source["release_version"] = _require_non_empty_text(
        source["release_version"], field="source_experiment_receipt.release_version"
    )
    source["experiment_kind"] = _require_non_empty_text(
        source["experiment_kind"], field="source_experiment_receipt.experiment_kind"
    )
    if source["release_version"] != _EXPECTED_SOURCE_RELEASE_VERSION:
        raise ValueError(
            f"source_experiment_receipt.release_version must be {_EXPECTED_SOURCE_RELEASE_VERSION}"
        )
    if source["experiment_kind"] != _EXPECTED_SOURCE_EXPERIMENT_KIND:
        raise ValueError(
            f"source_experiment_receipt.experiment_kind must be {_EXPECTED_SOURCE_EXPERIMENT_KIND}"
        )

    for key in (
        "source_dispatch_release_version",
        "source_dispatch_kind",
        "source_dispatch_receipt_hash",
        "source_lane_receipt_hash",
        "source_selected_target",
        "receipt_hash",
    ):
        source[key] = _require_non_empty_text(source[key], field=f"source_experiment_receipt.{key}")

    correction = _normalize_correction(
        source["canonical_selected_correction"], field="source_experiment_receipt.canonical_selected_correction"
    )
    correction_length = _require_int(source["correction_length"], field="source_experiment_receipt.correction_length")
    if correction_length != len(correction):
        raise ValueError("source_experiment_receipt.correction_length must match canonical_selected_correction length")
    source["canonical_selected_correction"] = correction
    source["correction_length"] = correction_length

    execution_profile = source["execution_profile"]
    if not isinstance(execution_profile, Mapping):
        raise ValueError("source_experiment_receipt.execution_profile must be a mapping")
    normalized_profile: dict[str, str] = {}
    for key in _REQUIRED_PROFILE_FIELDS:
        if key not in execution_profile:
            raise ValueError(f"source_experiment_receipt.execution_profile missing key: {key}")
        normalized_profile[key] = _require_non_empty_text(
            execution_profile[key], field=f"source_experiment_receipt.execution_profile.{key}"
        )
    if normalized_profile["timing_regime"] not in _ALLOWED_TIMING_REGIMES:
        raise ValueError(
            "source_experiment_receipt.execution_profile.timing_regime unsupported: "
            + normalized_profile["timing_regime"]
        )
    if normalized_profile["memory_pressure_class"] not in _ALLOWED_MEMORY_PRESSURE_CLASSES:
        raise ValueError(
            "source_experiment_receipt.execution_profile.memory_pressure_class unsupported: "
            + normalized_profile["memory_pressure_class"]
        )
    source["execution_profile"] = MappingProxyType(normalized_profile)

    metrics = source["metric_bundle"]
    if not isinstance(metrics, Mapping):
        raise ValueError("source_experiment_receipt.metric_bundle must be a mapping")
    source["metric_bundle"] = _canonical_mapping(
        metrics,
        field="source_experiment_receipt.metric_bundle",
        required_keys=_REQUIRED_SOURCE_METRICS,
    )

    for field_name, expected in (
        ("advisory_only", True),
        ("hardware_execution_performed", False),
        ("decoder_core_modified", False),
    ):
        if not isinstance(source[field_name], bool):
            raise ValueError(f"source_experiment_receipt.{field_name} must be boolean")
        if source[field_name] is not expected:
            raise ValueError(f"source_experiment_receipt.{field_name} must be {expected}")

    expected_hash = _stable_hash(_source_experiment_hash_payload(source))
    if source["receipt_hash"] != expected_hash:
        raise ValueError("source_experiment_receipt.receipt_hash mismatch")

    return source


def _source_experiment_hash_payload(source: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "release_version": source["release_version"],
        "experiment_kind": source["experiment_kind"],
        "source_dispatch_release_version": source["source_dispatch_release_version"],
        "source_dispatch_kind": source["source_dispatch_kind"],
        "source_dispatch_receipt_hash": source["source_dispatch_receipt_hash"],
        "source_lane_receipt_hash": source["source_lane_receipt_hash"],
        "source_selected_target": source["source_selected_target"],
        "canonical_selected_correction": tuple(source["canonical_selected_correction"]),
        "correction_length": source["correction_length"],
        "execution_profile": {key: source["execution_profile"][key] for key in sorted(_REQUIRED_PROFILE_FIELDS)},
        "pipeline_depth_class": source["execution_profile"]["pipeline_depth_class"],
        "lane_parallelism_class": source["execution_profile"]["lane_parallelism_class"],
        "timing_regime": source["execution_profile"]["timing_regime"],
        "power_regime": source["execution_profile"]["power_regime"],
        "thermal_regime": source["execution_profile"]["thermal_regime"],
        "memory_pressure_class": source["execution_profile"]["memory_pressure_class"],
        "metric_bundle": {key: source["metric_bundle"][key] for key in sorted(_REQUIRED_SOURCE_METRICS)},
        "asic_compatibility_score": source["metric_bundle"]["asic_compatibility_score"],
        "execution_feasibility_score": source["metric_bundle"]["execution_feasibility_score"],
        "timing_efficiency_score": source["metric_bundle"]["timing_efficiency_score"],
        "power_efficiency_score": source["metric_bundle"]["power_efficiency_score"],
        "thermal_stability_score": source["metric_bundle"]["thermal_stability_score"],
        "memory_feasibility_score": source["metric_bundle"]["memory_feasibility_score"],
        "bounded_experiment_confidence": source["metric_bundle"]["bounded_experiment_confidence"],
        "advisory_only": source["advisory_only"],
        "hardware_execution_performed": source["hardware_execution_performed"],
        "decoder_core_modified": source["decoder_core_modified"],
    }
    return payload


def _project_ternary_to_css(correction: tuple[int, ...]) -> "CSSSurfaceProjection":
    projected: list[tuple[int, int]] = []
    css_weight = 0
    x_weight = 0
    z_weight = 0
    for idx, symbol in enumerate(correction):
        if symbol == 0:
            pair = (0, 0)
        elif symbol == 1:
            pair = (1, 0)
            css_weight += 1
            x_weight += 1
        elif symbol == 2:
            pair = (0, 1)
            css_weight += 1
            z_weight += 1
        else:
            raise ValueError(f"canonical_selected_correction[{idx}] must be one of 0, 1, 2")
        projected.append(pair)

    ternary_weight = sum(1 for symbol in correction if symbol != 0)
    overlap_count = sum(1 for symbol, pair in zip(correction, projected) if (symbol == 0 and pair == (0, 0)) or (symbol in (1, 2) and pair != (0, 0)))
    divergence_count = len(correction) - overlap_count

    return CSSSurfaceProjection(
        projected_binary_sequence=tuple(projected),
        projection_weight=css_weight,
        ternary_weight=ternary_weight,
        x_projection_weight=x_weight,
        z_projection_weight=z_weight,
        overlap_count=overlap_count,
        divergence_count=divergence_count,
    )


def _normalize_study_policy_overrides(study_policy_overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    policy = {
        "force_hybrid_classification": None,
        "css_aligned_threshold": 0.82,
        "balanced_threshold": 0.70,
        "divergent_threshold": 0.35,
        "surface_favorable_margin": 0.08,
    }
    if study_policy_overrides is None:
        return policy
    if not isinstance(study_policy_overrides, Mapping):
        raise ValueError("study_policy_overrides must be a mapping when provided")

    for key in sorted(study_policy_overrides.keys()):
        if key not in policy:
            raise ValueError(f"study_policy_overrides.{key} is not supported")
    if "force_hybrid_classification" in study_policy_overrides:
        label = _require_non_empty_text(
            study_policy_overrides["force_hybrid_classification"],
            field="study_policy_overrides.force_hybrid_classification",
        )
        if label not in _ALLOWED_CLASSIFICATIONS:
            raise ValueError(
                "study_policy_overrides.force_hybrid_classification unsupported: " + label
            )
        policy["force_hybrid_classification"] = label

    for key in (
        "css_aligned_threshold",
        "balanced_threshold",
        "divergent_threshold",
        "surface_favorable_margin",
    ):
        if key in study_policy_overrides:
            policy[key] = _bounded(study_policy_overrides[key], field=f"study_policy_overrides.{key}")

    if policy["divergent_threshold"] > policy["balanced_threshold"]:
        raise ValueError("study_policy_overrides are contradictory: divergent_threshold cannot exceed balanced_threshold")
    if policy["balanced_threshold"] > policy["css_aligned_threshold"]:
        raise ValueError("study_policy_overrides are contradictory: balanced_threshold cannot exceed css_aligned_threshold")
    return policy


def _normalize_study_constraints(study_constraints: Mapping[str, Any] | None) -> dict[str, Any]:
    constraints = {
        "required_hybrid_classification": None,
        "minimum_surface_alignment_score": 0.0,
        "minimum_ternary_preservation_score": 0.0,
        "minimum_hybrid_overlap_score": 0.0,
    }
    if study_constraints is None:
        return constraints
    if not isinstance(study_constraints, Mapping):
        raise ValueError("study_constraints must be a mapping when provided")

    for key in sorted(study_constraints.keys()):
        if key not in constraints:
            raise ValueError(f"study_constraints.{key} is not supported")

    if "required_hybrid_classification" in study_constraints:
        label = _require_non_empty_text(
            study_constraints["required_hybrid_classification"],
            field="study_constraints.required_hybrid_classification",
        )
        if label not in _ALLOWED_CLASSIFICATIONS:
            raise ValueError(f"study_constraints.required_hybrid_classification unsupported: {label}")
        constraints["required_hybrid_classification"] = label

    for key in (
        "minimum_surface_alignment_score",
        "minimum_ternary_preservation_score",
        "minimum_hybrid_overlap_score",
    ):
        if key in study_constraints:
            constraints[key] = _bounded(study_constraints[key], field=f"study_constraints.{key}")

    return constraints


def _derive_metrics(source: Mapping[str, Any], projection: "CSSSurfaceProjection") -> "HybridComparisonMetricSet":
    correction_length = len(source["canonical_selected_correction"])
    profile = source["execution_profile"]
    bundle = source["metric_bundle"]

    imbalance = abs(projection.x_projection_weight - projection.z_projection_weight) / correction_length
    css_projection_consistency = _bounded(1.0 - imbalance, field="css_projection_consistency_score")

    neutral_ratio = 1.0 - (projection.projection_weight / correction_length)
    profile_bonus = {
        "relaxed": 0.05,
        "moderate": 0.0,
        "tight": -0.05,
    }[profile["timing_regime"]]
    surface_alignment = _bounded(
        max(
            0.0,
            min(
                1.0,
                (0.50 * bundle["timing_efficiency_score"])
                + (0.30 * bundle["memory_feasibility_score"])
                + (0.20 * neutral_ratio)
                + profile_bonus,
            ),
        ),
        field="surface_alignment_score",
    )

    preservation_penalty = {
        "high": 0.12,
        "moderate": 0.06,
        "low": 0.0,
    }[profile["memory_pressure_class"]]
    ternary_preservation = _bounded(
        max(
            0.0,
            min(
                1.0,
                (0.45 * bundle["asic_compatibility_score"])
                + (0.35 * bundle["execution_feasibility_score"])
                + (0.20 * bundle["bounded_experiment_confidence"])
                - preservation_penalty,
            ),
        ),
        field="ternary_preservation_score",
    )

    overlap_ratio = projection.overlap_count / correction_length
    hybrid_overlap = _bounded(
        max(0.0, min(1.0, (0.70 * overlap_ratio) + (0.30 * css_projection_consistency))),
        field="hybrid_overlap_score",
    )

    cross_domain_stability = _bounded(
        max(
            0.0,
            min(
                1.0,
                0.30 * css_projection_consistency
                + 0.30 * hybrid_overlap
                + 0.20 * bundle["thermal_stability_score"]
                + 0.20 * bundle["bounded_experiment_confidence"],
            ),
        ),
        field="cross_domain_stability_score",
    )

    bounded_confidence = _bounded(
        (
            css_projection_consistency
            + surface_alignment
            + ternary_preservation
            + hybrid_overlap
            + cross_domain_stability
        )
        / 5.0,
        field="bounded_hybrid_confidence",
    )

    return HybridComparisonMetricSet(
        css_projection_consistency_score=css_projection_consistency,
        surface_alignment_score=surface_alignment,
        ternary_preservation_score=ternary_preservation,
        hybrid_overlap_score=hybrid_overlap,
        cross_domain_stability_score=cross_domain_stability,
        bounded_hybrid_confidence=bounded_confidence,
    )


def _classify_study(
    metrics: "HybridComparisonMetricSet",
    policy: Mapping[str, Any],
) -> "CSSSurfaceHybridStudyDecision":
    if policy["force_hybrid_classification"] is not None:
        classification = policy["force_hybrid_classification"]
    elif metrics.hybrid_overlap_score <= policy["divergent_threshold"] and metrics.cross_domain_stability_score <= policy["divergent_threshold"]:
        classification = "hybrid_divergent"
    elif (
        metrics.css_projection_consistency_score >= policy["css_aligned_threshold"]
        and metrics.hybrid_overlap_score >= policy["css_aligned_threshold"]
        and metrics.surface_alignment_score >= policy["balanced_threshold"]
        and metrics.ternary_preservation_score >= policy["balanced_threshold"]
    ):
        classification = "css_aligned"
    elif (
        metrics.surface_alignment_score >= policy["balanced_threshold"]
        and metrics.ternary_preservation_score >= policy["balanced_threshold"]
        and abs(metrics.surface_alignment_score - metrics.ternary_preservation_score) <= policy["surface_favorable_margin"]
    ):
        classification = "hybrid_balanced"
    elif metrics.surface_alignment_score > metrics.ternary_preservation_score + policy["surface_favorable_margin"]:
        classification = "surface_favorable"
    elif metrics.ternary_preservation_score > metrics.surface_alignment_score + policy["surface_favorable_margin"]:
        classification = "ternary_favorable"
    elif metrics.bounded_hybrid_confidence >= policy["balanced_threshold"]:
        classification = "hybrid_balanced"
    else:
        classification = "hybrid_divergent"

    recommendation = {
        "css_aligned": "Preserve CSS/surface framing as primary comparative view.",
        "surface_favorable": "Prefer surface-style simplification for bounded deployment studies.",
        "ternary_favorable": "Preserve ternary/qutrit framing for high-fidelity comparative studies.",
        "hybrid_balanced": "Use blended reporting across CSS/surface and ternary framings.",
        "hybrid_divergent": "Escalate to deeper validation before cross-domain adoption.",
    }[classification]

    return CSSSurfaceHybridStudyDecision(
        hybrid_classification=classification,
        hybrid_recommendation=recommendation,
    )


def _apply_constraints(
    constraints: Mapping[str, Any],
    *,
    metrics: "HybridComparisonMetricSet",
    decision: "CSSSurfaceHybridStudyDecision",
    policy: Mapping[str, Any],
) -> None:
    required = constraints["required_hybrid_classification"]
    forced = policy["force_hybrid_classification"]
    if forced is not None and required is not None and forced != required:
        raise ValueError("study_constraints are contradictory with study_policy_overrides.force_hybrid_classification")

    if required is not None and decision.hybrid_classification != required:
        raise ValueError("study_constraints.required_hybrid_classification not satisfied")

    if metrics.surface_alignment_score < constraints["minimum_surface_alignment_score"]:
        raise ValueError("study_constraints.minimum_surface_alignment_score not satisfied")
    if metrics.ternary_preservation_score < constraints["minimum_ternary_preservation_score"]:
        raise ValueError("study_constraints.minimum_ternary_preservation_score not satisfied")
    if metrics.hybrid_overlap_score < constraints["minimum_hybrid_overlap_score"]:
        raise ValueError("study_constraints.minimum_hybrid_overlap_score not satisfied")


@dataclass(frozen=True)
class CSSSurfaceProjection:
    projected_binary_sequence: tuple[tuple[int, int], ...]
    projection_weight: int
    ternary_weight: int
    x_projection_weight: int
    z_projection_weight: int
    overlap_count: int
    divergence_count: int

    def __post_init__(self) -> None:
        sequence = tuple(tuple(pair) for pair in self.projected_binary_sequence)
        normalized: list[tuple[int, int]] = []
        for idx, pair in enumerate(sequence):
            if len(pair) != 2:
                raise ValueError(f"projected_binary_sequence[{idx}] must contain exactly two entries")
            x = _require_int(pair[0], field=f"projected_binary_sequence[{idx}][0]")
            z = _require_int(pair[1], field=f"projected_binary_sequence[{idx}][1]")
            if (x, z) not in ((0, 0), (1, 0), (0, 1)):
                raise ValueError(f"projected_binary_sequence[{idx}] must be one of (0,0), (1,0), (0,1)")
            normalized.append((x, z))
        object.__setattr__(self, "projected_binary_sequence", tuple(normalized))

        expected_len = len(normalized)
        for field_name in (
            "projection_weight",
            "ternary_weight",
            "x_projection_weight",
            "z_projection_weight",
            "overlap_count",
            "divergence_count",
        ):
            raw = getattr(self, field_name)
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise ValueError(f"{field_name} must be an integer")
            if raw < 0 or raw > expected_len:
                raise ValueError(f"{field_name} out of valid range")

    def to_dict(self) -> dict[str, Any]:
        return {
            "projected_binary_sequence": self.projected_binary_sequence,
            "projection_weight": self.projection_weight,
            "ternary_weight": self.ternary_weight,
            "x_projection_weight": self.x_projection_weight,
            "z_projection_weight": self.z_projection_weight,
            "overlap_count": self.overlap_count,
            "divergence_count": self.divergence_count,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class HybridComparisonMetricSet:
    css_projection_consistency_score: float
    surface_alignment_score: float
    ternary_preservation_score: float
    hybrid_overlap_score: float
    cross_domain_stability_score: float
    bounded_hybrid_confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "css_projection_consistency_score",
            _bounded(self.css_projection_consistency_score, field="css_projection_consistency_score"),
        )
        object.__setattr__(self, "surface_alignment_score", _bounded(self.surface_alignment_score, field="surface_alignment_score"))
        object.__setattr__(self, "ternary_preservation_score", _bounded(self.ternary_preservation_score, field="ternary_preservation_score"))
        object.__setattr__(self, "hybrid_overlap_score", _bounded(self.hybrid_overlap_score, field="hybrid_overlap_score"))
        object.__setattr__(
            self,
            "cross_domain_stability_score",
            _bounded(self.cross_domain_stability_score, field="cross_domain_stability_score"),
        )
        object.__setattr__(
            self,
            "bounded_hybrid_confidence",
            _bounded(self.bounded_hybrid_confidence, field="bounded_hybrid_confidence"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "css_projection_consistency_score": self.css_projection_consistency_score,
            "surface_alignment_score": self.surface_alignment_score,
            "ternary_preservation_score": self.ternary_preservation_score,
            "hybrid_overlap_score": self.hybrid_overlap_score,
            "cross_domain_stability_score": self.cross_domain_stability_score,
            "bounded_hybrid_confidence": self.bounded_hybrid_confidence,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class CSSSurfaceHybridStudyDecision:
    hybrid_classification: str
    hybrid_recommendation: str

    def __post_init__(self) -> None:
        classification = _require_non_empty_text(self.hybrid_classification, field="hybrid_classification")
        if classification not in _ALLOWED_CLASSIFICATIONS:
            raise ValueError(f"hybrid_classification unsupported: {classification}")
        object.__setattr__(self, "hybrid_classification", classification)
        object.__setattr__(
            self,
            "hybrid_recommendation",
            _require_non_empty_text(self.hybrid_recommendation, field="hybrid_recommendation"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "hybrid_classification": self.hybrid_classification,
            "hybrid_recommendation": self.hybrid_recommendation,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class CSSSurfaceHybridStudyReceipt:
    release_version: str
    study_kind: str
    source_experiment_release_version: str
    source_experiment_kind: str
    source_experiment_receipt_hash: str
    source_dispatch_receipt_hash: str
    source_lane_receipt_hash: str
    source_selected_target: str
    canonical_selected_correction: tuple[int, ...]
    correction_length: int
    execution_profile: Mapping[str, str]
    source_metric_bundle: Mapping[str, float]
    binary_css_projection: CSSSurfaceProjection
    hybrid_metrics: HybridComparisonMetricSet
    study_decision: CSSSurfaceHybridStudyDecision
    advisory_only: bool
    hardware_execution_performed: bool
    decoder_core_modified: bool
    receipt_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "release_version", _require_non_empty_text(self.release_version, field="release_version"))
        object.__setattr__(self, "study_kind", _require_non_empty_text(self.study_kind, field="study_kind"))
        object.__setattr__(
            self,
            "source_experiment_release_version",
            _require_non_empty_text(self.source_experiment_release_version, field="source_experiment_release_version"),
        )
        object.__setattr__(
            self,
            "source_experiment_kind",
            _require_non_empty_text(self.source_experiment_kind, field="source_experiment_kind"),
        )
        for field_name in (
            "source_experiment_receipt_hash",
            "source_dispatch_receipt_hash",
            "source_lane_receipt_hash",
            "source_selected_target",
            "receipt_hash",
        ):
            object.__setattr__(self, field_name, _require_non_empty_text(getattr(self, field_name), field=field_name))

        correction = _normalize_correction(self.canonical_selected_correction, field="canonical_selected_correction")
        object.__setattr__(self, "canonical_selected_correction", correction)
        length = _require_int(self.correction_length, field="correction_length")
        if length != len(correction):
            raise ValueError("correction_length must match canonical_selected_correction length")
        object.__setattr__(self, "correction_length", length)

        if not isinstance(self.execution_profile, Mapping):
            raise ValueError("execution_profile must be a mapping")
        profile = {
            key: _require_non_empty_text(self.execution_profile[key], field=f"execution_profile.{key}")
            for key in sorted(_REQUIRED_PROFILE_FIELDS)
        }
        object.__setattr__(self, "execution_profile", MappingProxyType(profile))

        if not isinstance(self.source_metric_bundle, Mapping):
            raise ValueError("source_metric_bundle must be a mapping")
        object.__setattr__(
            self,
            "source_metric_bundle",
            _canonical_mapping(self.source_metric_bundle, field="source_metric_bundle", required_keys=_REQUIRED_SOURCE_METRICS),
        )

        if not isinstance(self.advisory_only, bool):
            raise ValueError("advisory_only must be boolean")
        if not isinstance(self.hardware_execution_performed, bool):
            raise ValueError("hardware_execution_performed must be boolean")
        if not isinstance(self.decoder_core_modified, bool):
            raise ValueError("decoder_core_modified must be boolean")

    def to_dict(self) -> dict[str, Any]:
        metrics = self.hybrid_metrics.to_dict()
        projection = self.binary_css_projection
        decision = self.study_decision
        return {
            "release_version": self.release_version,
            "study_kind": self.study_kind,
            "source_experiment_release_version": self.source_experiment_release_version,
            "source_experiment_kind": self.source_experiment_kind,
            "source_experiment_receipt_hash": self.source_experiment_receipt_hash,
            "source_dispatch_receipt_hash": self.source_dispatch_receipt_hash,
            "source_lane_receipt_hash": self.source_lane_receipt_hash,
            "source_selected_target": self.source_selected_target,
            "canonical_selected_correction": self.canonical_selected_correction,
            "correction_length": self.correction_length,
            "execution_profile": dict(self.execution_profile),
            "source_metric_bundle": dict(self.source_metric_bundle),
            "binary_css_projection": projection.projected_binary_sequence,
            "projection_weight": projection.projection_weight,
            "ternary_weight": projection.ternary_weight,
            "projection_x_weight": projection.x_projection_weight,
            "projection_z_weight": projection.z_projection_weight,
            "projection_overlap_count": projection.overlap_count,
            "projection_divergence_count": projection.divergence_count,
            "hybrid_classification": decision.hybrid_classification,
            "hybrid_recommendation": decision.hybrid_recommendation,
            "metric_bundle": metrics,
            "css_projection_consistency_score": metrics["css_projection_consistency_score"],
            "surface_alignment_score": metrics["surface_alignment_score"],
            "ternary_preservation_score": metrics["ternary_preservation_score"],
            "hybrid_overlap_score": metrics["hybrid_overlap_score"],
            "cross_domain_stability_score": metrics["cross_domain_stability_score"],
            "bounded_hybrid_confidence": metrics["bounded_hybrid_confidence"],
            "advisory_only": self.advisory_only,
            "hardware_execution_performed": self.hardware_execution_performed,
            "decoder_core_modified": self.decoder_core_modified,
            "receipt_hash": self.receipt_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("receipt_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _stable_hash(self.to_hash_payload_dict())



def run_css_surface_hybrid_study(
    source_experiment_receipt: Any,
    *,
    study_policy_overrides: Mapping[str, Any] | None = None,
    study_constraints: Mapping[str, Any] | None = None,
) -> CSSSurfaceHybridStudyReceipt:
    source = _normalize_source_experiment(source_experiment_receipt)
    policy = _normalize_study_policy_overrides(study_policy_overrides)
    constraints = _normalize_study_constraints(study_constraints)

    projection = _project_ternary_to_css(source["canonical_selected_correction"])
    metrics = _derive_metrics(source, projection)
    decision = _classify_study(metrics, policy)
    _apply_constraints(constraints, metrics=metrics, decision=decision, policy=policy)

    provisional = CSSSurfaceHybridStudyReceipt(
        release_version=_RELEASE_VERSION,
        study_kind=_STUDY_KIND,
        source_experiment_release_version=source["release_version"],
        source_experiment_kind=source["experiment_kind"],
        source_experiment_receipt_hash=source["receipt_hash"],
        source_dispatch_receipt_hash=source["source_dispatch_receipt_hash"],
        source_lane_receipt_hash=source["source_lane_receipt_hash"],
        source_selected_target=source["source_selected_target"],
        canonical_selected_correction=source["canonical_selected_correction"],
        correction_length=source["correction_length"],
        execution_profile=source["execution_profile"],
        source_metric_bundle=source["metric_bundle"],
        binary_css_projection=projection,
        hybrid_metrics=metrics,
        study_decision=decision,
        advisory_only=True,
        hardware_execution_performed=False,
        decoder_core_modified=False,
        receipt_hash="pending",
    )

    return CSSSurfaceHybridStudyReceipt(
        release_version=provisional.release_version,
        study_kind=provisional.study_kind,
        source_experiment_release_version=provisional.source_experiment_release_version,
        source_experiment_kind=provisional.source_experiment_kind,
        source_experiment_receipt_hash=provisional.source_experiment_receipt_hash,
        source_dispatch_receipt_hash=provisional.source_dispatch_receipt_hash,
        source_lane_receipt_hash=provisional.source_lane_receipt_hash,
        source_selected_target=provisional.source_selected_target,
        canonical_selected_correction=provisional.canonical_selected_correction,
        correction_length=provisional.correction_length,
        execution_profile=provisional.execution_profile,
        source_metric_bundle=provisional.source_metric_bundle,
        binary_css_projection=provisional.binary_css_projection,
        hybrid_metrics=provisional.hybrid_metrics,
        study_decision=provisional.study_decision,
        advisory_only=provisional.advisory_only,
        hardware_execution_performed=provisional.hardware_execution_performed,
        decoder_core_modified=provisional.decoder_core_modified,
        receipt_hash=provisional.stable_hash(),
    )


__all__ = [
    "CSSSurfaceProjection",
    "HybridComparisonMetricSet",
    "CSSSurfaceHybridStudyDecision",
    "CSSSurfaceHybridStudyReceipt",
    "run_css_surface_hybrid_study",
]
