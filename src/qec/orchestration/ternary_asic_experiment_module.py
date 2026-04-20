"""v138.4.2 — deterministic ternary ASIC experiment modeling module."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping

_RELEASE_VERSION = "v138.4.2"
_EXPERIMENT_KIND = "ternary_asic_experiment_module"
_EXPECTED_SOURCE_RELEASE_VERSION = "v138.4.1"
_EXPECTED_SOURCE_DISPATCH_KIND = "qutrit_hardware_dispatch_path"
_ASIC_COMPATIBLE_TARGET = "qutrit_asic_lane"

_REQUIRED_SOURCE_METRICS = (
    "target_compatibility_score",
    "dispatch_readiness_score",
    "timing_feasibility_score",
    "resource_feasibility_score",
    "constraint_safety_score",
    "bounded_dispatch_confidence",
)
_REQUIRED_SOURCE_FIELDS = (
    "release_version",
    "dispatch_kind",
    "source_lane_receipt_hash",
    "selected_target",
    "canonical_selected_correction",
    "dispatch_eligible",
    "dispatch_metric_bundle",
    "advisory_only",
    "hardware_execution_performed",
    "decoder_core_modified",
    "receipt_hash",
)
_REQUIRED_SOURCE_HASH_BOUND_FIELDS = (
    "source_lane_kind",
    "source_selected_candidate_id",
    "timing_class",
    "resource_class",
    "constraint_status",
    "constraint_report",
)

_ALLOWED_PIPELINE_DEPTH = ("shallow", "medium", "deep")
_ALLOWED_LANE_PARALLELISM = ("serial", "dual_lane", "multi_lane")
_ALLOWED_TIMING_REGIMES = ("tight", "moderate", "relaxed")
_ALLOWED_POWER_REGIMES = ("low", "medium", "high")
_ALLOWED_THERMAL_REGIMES = ("cool", "warm", "hot")
_ALLOWED_MEMORY_PRESSURE = ("low", "moderate", "high")


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
    normalized = {}
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


def _extract_source_dispatch(source_dispatch_receipt: Any) -> dict[str, Any]:
    if isinstance(source_dispatch_receipt, Mapping):
        source = dict(source_dispatch_receipt)
    elif hasattr(source_dispatch_receipt, "to_dict") and callable(source_dispatch_receipt.to_dict):
        candidate = source_dispatch_receipt.to_dict()
        if not isinstance(candidate, Mapping):
            raise ValueError("source_dispatch_receipt.to_dict() must return a mapping")
        source = dict(candidate)
    else:
        raise ValueError("source_dispatch_receipt must be mapping-compatible")

    missing = [key for key in _REQUIRED_SOURCE_FIELDS if key not in source]
    if missing:
        raise ValueError(f"source_dispatch_receipt missing required fields: {', '.join(missing)}")

    source["release_version"] = _require_non_empty_text(
        source["release_version"], field="source_dispatch_receipt.release_version"
    )
    source["dispatch_kind"] = _require_non_empty_text(source["dispatch_kind"], field="source_dispatch_receipt.dispatch_kind")
    if source["release_version"] != _EXPECTED_SOURCE_RELEASE_VERSION:
        raise ValueError(f"source_dispatch_receipt.release_version must be {_EXPECTED_SOURCE_RELEASE_VERSION}")
    if source["dispatch_kind"] != _EXPECTED_SOURCE_DISPATCH_KIND:
        raise ValueError(f"source_dispatch_receipt.dispatch_kind must be {_EXPECTED_SOURCE_DISPATCH_KIND}")

    source["source_lane_receipt_hash"] = _require_non_empty_text(
        source["source_lane_receipt_hash"], field="source_dispatch_receipt.source_lane_receipt_hash"
    )

    if isinstance(source["selected_target"], Mapping):
        selected_target = _require_non_empty_text(source["selected_target"].get("target_name"), field="source_dispatch_receipt.selected_target.target_name")
    else:
        selected_target = _require_non_empty_text(source["selected_target"], field="source_dispatch_receipt.selected_target")
    source["selected_target"] = selected_target

    source["canonical_selected_correction"] = _normalize_correction(
        source["canonical_selected_correction"], field="source_dispatch_receipt.canonical_selected_correction"
    )

    if not isinstance(source["dispatch_eligible"], bool):
        raise ValueError("source_dispatch_receipt.dispatch_eligible must be boolean")
    if source["dispatch_eligible"] is not True:
        raise ValueError("source_dispatch_receipt.dispatch_eligible must be True")

    if not isinstance(source["advisory_only"], bool):
        raise ValueError("source_dispatch_receipt.advisory_only must be boolean")
    if source["advisory_only"] is not True:
        raise ValueError("source_dispatch_receipt.advisory_only must be True")

    if not isinstance(source["hardware_execution_performed"], bool):
        raise ValueError("source_dispatch_receipt.hardware_execution_performed must be boolean")
    if source["hardware_execution_performed"] is not False:
        raise ValueError("source_dispatch_receipt.hardware_execution_performed must be False")

    if not isinstance(source["decoder_core_modified"], bool):
        raise ValueError("source_dispatch_receipt.decoder_core_modified must be boolean")
    if source["decoder_core_modified"] is not False:
        raise ValueError("source_dispatch_receipt.decoder_core_modified must be False")

    if source["selected_target"] != _ASIC_COMPATIBLE_TARGET:
        raise ValueError(
            "source_dispatch_receipt.selected_target is not ASIC-compatible; expected qutrit_asic_lane"
        )

    if not isinstance(source["dispatch_metric_bundle"], Mapping):
        raise ValueError("source_dispatch_receipt.dispatch_metric_bundle must be a mapping")
    source["dispatch_metric_bundle"] = _canonical_mapping(
        source["dispatch_metric_bundle"],
        field="source_dispatch_receipt.dispatch_metric_bundle",
        required_keys=_REQUIRED_SOURCE_METRICS,
    )

    source["receipt_hash"] = _require_non_empty_text(source["receipt_hash"], field="source_dispatch_receipt.receipt_hash")
    expected_hash = _stable_hash(_source_dispatch_hash_payload(source))
    if source["receipt_hash"] != expected_hash:
        raise ValueError("source_dispatch_receipt.receipt_hash mismatch")

    return source


def _source_dispatch_hash_payload(source: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in _REQUIRED_SOURCE_HASH_BOUND_FIELDS if key not in source]
    if missing:
        raise ValueError(
            "source_dispatch_receipt missing required hash-bound fields: " + ", ".join(missing)
        )

    return {
        "release_version": source["release_version"],
        "dispatch_kind": source["dispatch_kind"],
        "source_lane_kind": source["source_lane_kind"],
        "source_lane_receipt_hash": source["source_lane_receipt_hash"],
        "source_selected_candidate_id": source["source_selected_candidate_id"],
        "canonical_selected_correction": tuple(source["canonical_selected_correction"]),
        "selected_target": source["selected_target"],
        "dispatch_eligible": source["dispatch_eligible"],
        "timing_class": source["timing_class"],
        "resource_class": source["resource_class"],
        "constraint_status": source["constraint_status"],
        "dispatch_metric_bundle": {k: source["dispatch_metric_bundle"][k] for k in _REQUIRED_SOURCE_METRICS},
        "constraint_report": source["constraint_report"],
        "advisory_only": source["advisory_only"],
        "hardware_execution_performed": source["hardware_execution_performed"],
        "decoder_core_modified": source["decoder_core_modified"],
    }


def _normalize_profile_overrides(profile_overrides: Mapping[str, Any] | None) -> dict[str, str]:
    allowed = {
        "pipeline_depth_class": _ALLOWED_PIPELINE_DEPTH,
        "lane_parallelism_class": _ALLOWED_LANE_PARALLELISM,
        "timing_regime": _ALLOWED_TIMING_REGIMES,
        "power_regime": _ALLOWED_POWER_REGIMES,
        "thermal_regime": _ALLOWED_THERMAL_REGIMES,
        "memory_pressure_class": _ALLOWED_MEMORY_PRESSURE,
    }
    overrides: dict[str, str] = {}
    if profile_overrides is None:
        return overrides
    if not isinstance(profile_overrides, Mapping):
        raise ValueError("profile_overrides must be a mapping when provided")
    for key in sorted(profile_overrides.keys()):
        if key not in allowed:
            raise ValueError(f"profile_overrides.{key} is not supported")
        value = _require_non_empty_text(profile_overrides[key], field=f"profile_overrides.{key}")
        if value not in allowed[key]:
            raise ValueError(f"profile_overrides.{key} unsupported value: {value}")
        overrides[key] = value
    return overrides


def _normalize_constraints(experiment_constraints: Mapping[str, Any] | None) -> dict[str, str | None]:
    constraints: dict[str, str | None] = {
        "required_timing_regime": None,
        "max_power_regime": None,
        "max_thermal_regime": None,
        "required_lane_parallelism_class": None,
    }
    if experiment_constraints is None:
        return constraints
    if not isinstance(experiment_constraints, Mapping):
        raise ValueError("experiment_constraints must be a mapping when provided")

    if "required_timing_regime" in experiment_constraints:
        value = _require_non_empty_text(
            experiment_constraints["required_timing_regime"], field="experiment_constraints.required_timing_regime"
        )
        if value not in _ALLOWED_TIMING_REGIMES:
            raise ValueError(f"experiment_constraints.required_timing_regime unsupported: {value}")
        constraints["required_timing_regime"] = value

    if "max_power_regime" in experiment_constraints:
        value = _require_non_empty_text(experiment_constraints["max_power_regime"], field="experiment_constraints.max_power_regime")
        if value not in _ALLOWED_POWER_REGIMES:
            raise ValueError(f"experiment_constraints.max_power_regime unsupported: {value}")
        constraints["max_power_regime"] = value

    if "max_thermal_regime" in experiment_constraints:
        value = _require_non_empty_text(
            experiment_constraints["max_thermal_regime"], field="experiment_constraints.max_thermal_regime"
        )
        if value not in _ALLOWED_THERMAL_REGIMES:
            raise ValueError(f"experiment_constraints.max_thermal_regime unsupported: {value}")
        constraints["max_thermal_regime"] = value

    if "required_lane_parallelism_class" in experiment_constraints:
        value = _require_non_empty_text(
            experiment_constraints["required_lane_parallelism_class"],
            field="experiment_constraints.required_lane_parallelism_class",
        )
        if value not in _ALLOWED_LANE_PARALLELISM:
            raise ValueError(f"experiment_constraints.required_lane_parallelism_class unsupported: {value}")
        constraints["required_lane_parallelism_class"] = value

    if constraints["required_lane_parallelism_class"] == "multi_lane" and constraints["max_power_regime"] == "low":
        raise ValueError("experiment_constraints are contradictory: multi_lane cannot be paired with max_power_regime=low")
    if constraints["required_timing_regime"] == "tight" and constraints["max_thermal_regime"] == "cool":
        raise ValueError("experiment_constraints are contradictory: tight timing cannot require max_thermal_regime=cool")

    return constraints


def _profile_from_source(source: Mapping[str, Any]) -> dict[str, str]:
    metrics = source["dispatch_metric_bundle"]
    correction = source["canonical_selected_correction"]
    correction_len = len(correction)

    if correction_len <= 8:
        pipeline_depth = "shallow"
        memory_pressure = "low"
    elif correction_len <= 32:
        pipeline_depth = "medium"
        memory_pressure = "moderate"
    else:
        pipeline_depth = "deep"
        memory_pressure = "high"

    readiness_signal = 0.55 * metrics["dispatch_readiness_score"] + 0.45 * metrics["target_compatibility_score"]
    if readiness_signal >= 0.88:
        lane_parallelism = "multi_lane"
    elif readiness_signal >= 0.65:
        lane_parallelism = "dual_lane"
    else:
        lane_parallelism = "serial"

    if correction_len > 32 or metrics["timing_feasibility_score"] < 0.65:
        timing_regime = "tight"
    elif correction_len > 8 or metrics["timing_feasibility_score"] < 0.82:
        timing_regime = "moderate"
    else:
        timing_regime = "relaxed"

    if correction_len > 40 or metrics["resource_feasibility_score"] < 0.55:
        power_regime = "high"
    elif correction_len > 12 or metrics["resource_feasibility_score"] < 0.75:
        power_regime = "medium"
    else:
        power_regime = "low"

    if correction_len > 40 or metrics["constraint_safety_score"] < 0.60:
        thermal_regime = "hot"
    elif correction_len > 16 or metrics["constraint_safety_score"] < 0.80:
        thermal_regime = "warm"
    else:
        thermal_regime = "cool"

    return {
        "pipeline_depth_class": pipeline_depth,
        "lane_parallelism_class": lane_parallelism,
        "timing_regime": timing_regime,
        "power_regime": power_regime,
        "thermal_regime": thermal_regime,
        "memory_pressure_class": memory_pressure,
    }


def _apply_constraints(profile: Mapping[str, str], constraints: Mapping[str, str | None]) -> None:
    if constraints["required_timing_regime"] is not None and profile["timing_regime"] != constraints["required_timing_regime"]:
        raise ValueError("experiment profile violates experiment_constraints.required_timing_regime")
    if constraints["required_lane_parallelism_class"] is not None and profile["lane_parallelism_class"] != constraints["required_lane_parallelism_class"]:
        raise ValueError("experiment profile violates experiment_constraints.required_lane_parallelism_class")

    power_order = {"low": 0, "medium": 1, "high": 2}
    if constraints["max_power_regime"] is not None and power_order[profile["power_regime"]] > power_order[constraints["max_power_regime"]]:
        raise ValueError("experiment profile violates experiment_constraints.max_power_regime")

    thermal_order = {"cool": 0, "warm": 1, "hot": 2}
    if constraints["max_thermal_regime"] is not None and thermal_order[profile["thermal_regime"]] > thermal_order[constraints["max_thermal_regime"]]:
        raise ValueError("experiment profile violates experiment_constraints.max_thermal_regime")


def _compute_metrics(source: Mapping[str, Any], profile: Mapping[str, str]) -> dict[str, float]:
    bundle = source["dispatch_metric_bundle"]
    correction_length = len(source["canonical_selected_correction"])

    timing_regime_penalty = {"tight": 0.12, "moderate": 0.05, "relaxed": 0.0}
    power_regime_penalty = {"low": 0.0, "medium": 0.08, "high": 0.18}
    thermal_regime_penalty = {"cool": 0.0, "warm": 0.08, "hot": 0.18}
    memory_pressure_penalty = {"low": 0.0, "moderate": 0.07, "high": 0.16}

    asic_compatibility = _bounded(bundle["target_compatibility_score"], field="asic_compatibility_score")
    execution_feasibility = _bounded(
        max(
            0.0,
            (
                bundle["dispatch_readiness_score"]
                + bundle["timing_feasibility_score"]
                + bundle["resource_feasibility_score"]
                + bundle["constraint_safety_score"]
            )
            / 4.0
            - memory_pressure_penalty[profile["memory_pressure_class"]],
        ),
        field="execution_feasibility_score",
    )
    timing_efficiency = _bounded(
        max(0.0, bundle["timing_feasibility_score"] - timing_regime_penalty[profile["timing_regime"]]),
        field="timing_efficiency_score",
    )
    power_efficiency = _bounded(
        max(0.0, bundle["resource_feasibility_score"] - power_regime_penalty[profile["power_regime"]]),
        field="power_efficiency_score",
    )
    thermal_stability = _bounded(
        max(0.0, bundle["constraint_safety_score"] - thermal_regime_penalty[profile["thermal_regime"]]),
        field="thermal_stability_score",
    )
    base_memory = _bounded(max(0.0, 1.0 - min(1.0, correction_length / 128.0)), field="memory_base")
    memory_feasibility = _bounded(
        max(0.0, base_memory - memory_pressure_penalty[profile["memory_pressure_class"]]),
        field="memory_feasibility_score",
    )
    bounded_confidence = _bounded(
        (
            asic_compatibility
            + execution_feasibility
            + timing_efficiency
            + power_efficiency
            + thermal_stability
            + memory_feasibility
            + bundle["bounded_dispatch_confidence"]
        )
        / 7.0,
        field="bounded_experiment_confidence",
    )

    return {
        "asic_compatibility_score": asic_compatibility,
        "execution_feasibility_score": execution_feasibility,
        "timing_efficiency_score": timing_efficiency,
        "power_efficiency_score": power_efficiency,
        "thermal_stability_score": thermal_stability,
        "memory_feasibility_score": memory_feasibility,
        "bounded_experiment_confidence": bounded_confidence,
    }


@dataclass(frozen=True)
class TernaryASICExperimentInput:
    source_dispatch_release_version: str
    source_dispatch_kind: str
    source_dispatch_receipt_hash: str
    source_lane_receipt_hash: str
    source_selected_target: str
    canonical_selected_correction: tuple[int, ...]
    dispatch_metric_bundle: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_dispatch_release_version", _require_non_empty_text(self.source_dispatch_release_version, field="source_dispatch_release_version"))
        object.__setattr__(self, "source_dispatch_kind", _require_non_empty_text(self.source_dispatch_kind, field="source_dispatch_kind"))
        object.__setattr__(self, "source_dispatch_receipt_hash", _require_non_empty_text(self.source_dispatch_receipt_hash, field="source_dispatch_receipt_hash"))
        object.__setattr__(self, "source_lane_receipt_hash", _require_non_empty_text(self.source_lane_receipt_hash, field="source_lane_receipt_hash"))
        object.__setattr__(self, "source_selected_target", _require_non_empty_text(self.source_selected_target, field="source_selected_target"))
        object.__setattr__(self, "canonical_selected_correction", _normalize_correction(self.canonical_selected_correction, field="canonical_selected_correction"))
        if not isinstance(self.dispatch_metric_bundle, Mapping):
            raise ValueError("dispatch_metric_bundle must be a mapping")
        object.__setattr__(
            self,
            "dispatch_metric_bundle",
            _canonical_mapping(self.dispatch_metric_bundle, field="dispatch_metric_bundle", required_keys=_REQUIRED_SOURCE_METRICS),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_dispatch_release_version": self.source_dispatch_release_version,
            "source_dispatch_kind": self.source_dispatch_kind,
            "source_dispatch_receipt_hash": self.source_dispatch_receipt_hash,
            "source_lane_receipt_hash": self.source_lane_receipt_hash,
            "source_selected_target": self.source_selected_target,
            "canonical_selected_correction": self.canonical_selected_correction,
            "dispatch_metric_bundle": dict(self.dispatch_metric_bundle),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class TernaryASICExecutionProfile:
    pipeline_depth_class: str
    lane_parallelism_class: str
    timing_regime: str
    power_regime: str
    thermal_regime: str
    memory_pressure_class: str

    def __post_init__(self) -> None:
        if self.pipeline_depth_class not in _ALLOWED_PIPELINE_DEPTH:
            raise ValueError(f"pipeline_depth_class unsupported: {self.pipeline_depth_class}")
        if self.lane_parallelism_class not in _ALLOWED_LANE_PARALLELISM:
            raise ValueError(f"lane_parallelism_class unsupported: {self.lane_parallelism_class}")
        if self.timing_regime not in _ALLOWED_TIMING_REGIMES:
            raise ValueError(f"timing_regime unsupported: {self.timing_regime}")
        if self.power_regime not in _ALLOWED_POWER_REGIMES:
            raise ValueError(f"power_regime unsupported: {self.power_regime}")
        if self.thermal_regime not in _ALLOWED_THERMAL_REGIMES:
            raise ValueError(f"thermal_regime unsupported: {self.thermal_regime}")
        if self.memory_pressure_class not in _ALLOWED_MEMORY_PRESSURE:
            raise ValueError(f"memory_pressure_class unsupported: {self.memory_pressure_class}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_depth_class": self.pipeline_depth_class,
            "lane_parallelism_class": self.lane_parallelism_class,
            "timing_regime": self.timing_regime,
            "power_regime": self.power_regime,
            "thermal_regime": self.thermal_regime,
            "memory_pressure_class": self.memory_pressure_class,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class TernaryASICExperimentMetrics:
    metric_bundle: Mapping[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.metric_bundle, Mapping):
            raise ValueError("metric_bundle must be a mapping")
        required = (
            "asic_compatibility_score",
            "execution_feasibility_score",
            "timing_efficiency_score",
            "power_efficiency_score",
            "thermal_stability_score",
            "memory_feasibility_score",
            "bounded_experiment_confidence",
        )
        object.__setattr__(self, "metric_bundle", _canonical_mapping(self.metric_bundle, field="metric_bundle", required_keys=required))

    def to_dict(self) -> dict[str, Any]:
        return {"metric_bundle": dict(self.metric_bundle)}

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class TernaryASICExperimentReceipt:
    release_version: str
    experiment_kind: str
    experiment_input: TernaryASICExperimentInput
    execution_profile: TernaryASICExecutionProfile
    experiment_metrics: TernaryASICExperimentMetrics
    advisory_only: bool
    hardware_execution_performed: bool
    decoder_core_modified: bool
    receipt_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "release_version", _require_non_empty_text(self.release_version, field="release_version"))
        object.__setattr__(self, "experiment_kind", _require_non_empty_text(self.experiment_kind, field="experiment_kind"))
        if not isinstance(self.advisory_only, bool):
            raise ValueError("advisory_only must be boolean")
        if not isinstance(self.hardware_execution_performed, bool):
            raise ValueError("hardware_execution_performed must be boolean")
        if not isinstance(self.decoder_core_modified, bool):
            raise ValueError("decoder_core_modified must be boolean")
        object.__setattr__(self, "receipt_hash", _require_non_empty_text(self.receipt_hash, field="receipt_hash"))

    def to_dict(self) -> dict[str, Any]:
        profile = self.execution_profile
        metrics = self.experiment_metrics.metric_bundle
        source = self.experiment_input
        correction = source.canonical_selected_correction
        return {
            "release_version": self.release_version,
            "experiment_kind": self.experiment_kind,
            "source_dispatch_release_version": source.source_dispatch_release_version,
            "source_dispatch_kind": source.source_dispatch_kind,
            "source_dispatch_receipt_hash": source.source_dispatch_receipt_hash,
            "source_lane_receipt_hash": source.source_lane_receipt_hash,
            "source_selected_target": source.source_selected_target,
            "canonical_selected_correction": correction,
            "correction_length": len(correction),
            "execution_profile": profile.to_dict(),
            "pipeline_depth_class": profile.pipeline_depth_class,
            "lane_parallelism_class": profile.lane_parallelism_class,
            "timing_regime": profile.timing_regime,
            "power_regime": profile.power_regime,
            "thermal_regime": profile.thermal_regime,
            "memory_pressure_class": profile.memory_pressure_class,
            "metric_bundle": dict(metrics),
            "asic_compatibility_score": metrics["asic_compatibility_score"],
            "execution_feasibility_score": metrics["execution_feasibility_score"],
            "timing_efficiency_score": metrics["timing_efficiency_score"],
            "power_efficiency_score": metrics["power_efficiency_score"],
            "thermal_stability_score": metrics["thermal_stability_score"],
            "memory_feasibility_score": metrics["memory_feasibility_score"],
            "bounded_experiment_confidence": metrics["bounded_experiment_confidence"],
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

    @property
    def source_dispatch_release_version(self) -> str:
        return self.experiment_input.source_dispatch_release_version

    @property
    def source_dispatch_kind(self) -> str:
        return self.experiment_input.source_dispatch_kind

    @property
    def source_dispatch_receipt_hash(self) -> str:
        return self.experiment_input.source_dispatch_receipt_hash

    @property
    def source_lane_receipt_hash(self) -> str:
        return self.experiment_input.source_lane_receipt_hash

    @property
    def source_selected_target(self) -> str:
        return self.experiment_input.source_selected_target

    @property
    def canonical_selected_correction(self) -> tuple[int, ...]:
        return self.experiment_input.canonical_selected_correction

    @property
    def correction_length(self) -> int:
        return len(self.experiment_input.canonical_selected_correction)

    @property
    def pipeline_depth_class(self) -> str:
        return self.execution_profile.pipeline_depth_class

    @property
    def lane_parallelism_class(self) -> str:
        return self.execution_profile.lane_parallelism_class

    @property
    def timing_regime(self) -> str:
        return self.execution_profile.timing_regime

    @property
    def power_regime(self) -> str:
        return self.execution_profile.power_regime

    @property
    def thermal_regime(self) -> str:
        return self.execution_profile.thermal_regime

    @property
    def memory_pressure_class(self) -> str:
        return self.execution_profile.memory_pressure_class

    @property
    def metric_bundle(self) -> Mapping[str, float]:
        return self.experiment_metrics.metric_bundle

    @property
    def asic_compatibility_score(self) -> float:
        return self.experiment_metrics.metric_bundle["asic_compatibility_score"]

    @property
    def execution_feasibility_score(self) -> float:
        return self.experiment_metrics.metric_bundle["execution_feasibility_score"]

    @property
    def timing_efficiency_score(self) -> float:
        return self.experiment_metrics.metric_bundle["timing_efficiency_score"]

    @property
    def power_efficiency_score(self) -> float:
        return self.experiment_metrics.metric_bundle["power_efficiency_score"]

    @property
    def thermal_stability_score(self) -> float:
        return self.experiment_metrics.metric_bundle["thermal_stability_score"]

    @property
    def memory_feasibility_score(self) -> float:
        return self.experiment_metrics.metric_bundle["memory_feasibility_score"]

    @property
    def bounded_experiment_confidence(self) -> float:
        return self.experiment_metrics.metric_bundle["bounded_experiment_confidence"]


def run_ternary_asic_experiment(
    source_dispatch_receipt: Any,
    *,
    profile_overrides: Mapping[str, Any] | None = None,
    experiment_constraints: Mapping[str, Any] | None = None,
) -> TernaryASICExperimentReceipt:
    source = _extract_source_dispatch(source_dispatch_receipt)
    overrides = _normalize_profile_overrides(profile_overrides)
    constraints = _normalize_constraints(experiment_constraints)

    profile_dict = _profile_from_source(source)
    for key in sorted(overrides.keys()):
        profile_dict[key] = overrides[key]
    _apply_constraints(profile_dict, constraints)

    profile = TernaryASICExecutionProfile(**profile_dict)
    metrics = TernaryASICExperimentMetrics(metric_bundle=_compute_metrics(source, profile.to_dict()))
    experiment_input = TernaryASICExperimentInput(
        source_dispatch_release_version=source["release_version"],
        source_dispatch_kind=source["dispatch_kind"],
        source_dispatch_receipt_hash=source["receipt_hash"],
        source_lane_receipt_hash=source["source_lane_receipt_hash"],
        source_selected_target=source["selected_target"],
        canonical_selected_correction=source["canonical_selected_correction"],
        dispatch_metric_bundle=source["dispatch_metric_bundle"],
    )

    provisional = TernaryASICExperimentReceipt(
        release_version=_RELEASE_VERSION,
        experiment_kind=_EXPERIMENT_KIND,
        experiment_input=experiment_input,
        execution_profile=profile,
        experiment_metrics=metrics,
        advisory_only=True,
        hardware_execution_performed=False,
        decoder_core_modified=False,
        receipt_hash="pending",
    )
    return TernaryASICExperimentReceipt(
        release_version=provisional.release_version,
        experiment_kind=provisional.experiment_kind,
        experiment_input=provisional.experiment_input,
        execution_profile=provisional.execution_profile,
        experiment_metrics=provisional.experiment_metrics,
        advisory_only=provisional.advisory_only,
        hardware_execution_performed=provisional.hardware_execution_performed,
        decoder_core_modified=provisional.decoder_core_modified,
        receipt_hash=provisional.stable_hash(),
    )


__all__ = [
    "TernaryASICExperimentInput",
    "TernaryASICExecutionProfile",
    "TernaryASICExperimentMetrics",
    "TernaryASICExperimentReceipt",
    "run_ternary_asic_experiment",
]
