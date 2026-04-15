"""v137.20.5 — Governance Forecast Topology Reconciliation Kernel.

Narrow additive Layer-4 advisory-only orchestration release.

Canonical flow:
forecast_series
+ topology_series
+ replay_horizon_series
-> reconciliation_analysis
+ stable_receipt

Invariants:
- no randomness
- no async
- no external I/O
- deterministic ordering
- canonical JSON + stable SHA-256
- validator must never raise
- preserve advisory-only semantics
- no mutation of inputs
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Sequence, Tuple


_METRIC_ORDER: Tuple[str, ...] = (
    "forecast_topology_delta_score",
    "reconciliation_alignment_score",
    "horizon_reconciliation_gradient",
    "forecast_drift_pressure",
    "replay_reconciliation_stability_score",
    "reconciliation_confidence_score",
)
_METRIC_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(_METRIC_ORDER)}


_ADVISORY_THRESHOLDS: Tuple[Tuple[str, float], ...] = (
    ("stable_reconciliation", 0.05),
    ("minor_reconciliation_variation", 0.20),
    ("moderate_reconciliation_instability", 0.50),
)


def _canonical_json(data: Any) -> str:
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _field(source: Any, name: str, default: Any = None) -> Any:
    try:
        if isinstance(source, Mapping):
            return source.get(name, default)
        return getattr(source, name, default)
    except Exception:
        return default


def _safe_text(value: Any) -> str:
    try:
        if value is None:
            return ""
        return str(value)
    except Exception:
        return ""


def _safe_bool(value: Any) -> bool:
    return value is True


def _safe_nonneg_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        parsed = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(parsed) or parsed < 0.0:
        return 0.0
    return parsed


def _safe_series(value: Any) -> Tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return ()


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


@dataclass(frozen=True)
class _ForecastNode:
    forecast_id: str
    topology_id: str
    forecast_stability: float
    forecast_alignment: float
    forecast_pressure: float
    replay_identity: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "forecast_id": self.forecast_id,
            "topology_id": self.topology_id,
            "forecast_stability": self.forecast_stability,
            "forecast_alignment": self.forecast_alignment,
            "forecast_pressure": self.forecast_pressure,
            "replay_identity": self.replay_identity,
        }


@dataclass(frozen=True)
class _TopologyNode:
    topology_id: str
    observed_stability: float
    observed_alignment: float
    observed_pressure: float
    continuity_ok: bool
    replay_identity: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topology_id": self.topology_id,
            "observed_stability": self.observed_stability,
            "observed_alignment": self.observed_alignment,
            "observed_pressure": self.observed_pressure,
            "continuity_ok": self.continuity_ok,
            "replay_identity": self.replay_identity,
        }


@dataclass(frozen=True)
class _HorizonNode:
    horizon_id: str
    horizon_step: float
    reconciliation_delta: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizon_id": self.horizon_id,
            "horizon_step": self.horizon_step,
            "reconciliation_delta": self.reconciliation_delta,
        }


@dataclass(frozen=True)
class ForecastTopologyReconciliationScenario:
    scenario_id: str
    forecast_series: Tuple[Mapping[str, Any], ...]
    topology_series: Tuple[Mapping[str, Any], ...]
    replay_horizon_series: Tuple[Mapping[str, Any], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "forecast_series": [dict(item) for item in self.forecast_series],
            "topology_series": [dict(item) for item in self.topology_series],
            "replay_horizon_series": [dict(item) for item in self.replay_horizon_series],
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_canonical_json().encode("utf-8"))


@dataclass(frozen=True)
class ForecastTopologyReconciliationMetric:
    metric_name: str
    metric_order: int
    metric_value: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "metric_order": self.metric_order,
            "metric_value": self.metric_value,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_canonical_json().encode("utf-8"))


@dataclass(frozen=True)
class ForecastTopologyReconciliationReceipt:
    scenario_hash: str
    metrics_hash: str
    reconciliation_hash: str
    reconciliation_confidence_score: float
    advisory_output: str
    receipt_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_hash": self.scenario_hash,
            "metrics_hash": self.metrics_hash,
            "reconciliation_hash": self.reconciliation_hash,
            "reconciliation_confidence_score": self.reconciliation_confidence_score,
            "advisory_output": self.advisory_output,
            "receipt_hash": self.receipt_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.receipt_hash


@dataclass(frozen=True)
class GovernanceForecastTopologyReconciliationKernel:
    scenario: ForecastTopologyReconciliationScenario
    metrics: Tuple[ForecastTopologyReconciliationMetric, ...]
    reconciliation_analysis: Mapping[str, Any]
    advisory_output: str
    violations: Tuple[str, ...]
    receipt: ForecastTopologyReconciliationReceipt
    reconciliation_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario.to_dict(),
            "metrics": [metric.to_dict() for metric in self.metrics],
            "reconciliation_analysis": dict(self.reconciliation_analysis),
            "advisory_output": self.advisory_output,
            "violations": list(self.violations),
            "receipt": self.receipt.to_dict(),
            "reconciliation_hash": self.reconciliation_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.reconciliation_hash


def _normalize_forecast_node(raw: Any, index: int) -> _ForecastNode:
    forecast_id = _safe_text(_field(raw, "forecast_id", "")).strip() or f"forecast_{index}"
    topology_id = (
        _safe_text(_field(raw, "topology_id", "")).strip()
        or _safe_text(_field(raw, "forecast_topology_id", "")).strip()
        or forecast_id
    )
    forecast_stability = _clamp01(
        _safe_nonneg_float(
            _field(
                raw,
                "forecast_stability",
                _field(
                    raw,
                    "forecast_coherence_score",
                    _field(raw, "topology_forecast_alignment", 0.0),
                ),
            )
        )
    )
    forecast_alignment = _clamp01(
        _safe_nonneg_float(
            _field(raw, "forecast_alignment", _field(raw, "topology_forecast_alignment", 0.0))
        )
    )
    forecast_pressure = _clamp01(
        _safe_nonneg_float(
            _field(raw, "forecast_pressure", _field(raw, "forecast_surface_pressure", 0.0))
        )
    )
    replay_identity = _safe_text(_field(raw, "replay_identity", "")).strip()
    return _ForecastNode(
        forecast_id=forecast_id,
        topology_id=topology_id,
        forecast_stability=forecast_stability,
        forecast_alignment=forecast_alignment,
        forecast_pressure=forecast_pressure,
        replay_identity=replay_identity,
    )


def _normalize_topology_node(raw: Any, index: int) -> _TopologyNode:
    topology_id = _safe_text(_field(raw, "topology_id", "")).strip() or f"topology_{index}"
    observed_stability = _clamp01(
        _safe_nonneg_float(
            _field(raw, "observed_stability", _field(raw, "topology_stability_score", _field(raw, "coherence", 0.0)))
        )
    )
    observed_alignment = _clamp01(
        _safe_nonneg_float(_field(raw, "observed_alignment", _field(raw, "alignment", 0.0)))
    )
    observed_pressure = _clamp01(
        _safe_nonneg_float(_field(raw, "observed_pressure", _field(raw, "pressure", 0.0)))
    )
    continuity_ok = _safe_bool(_field(raw, "continuity_ok", False))
    replay_identity = _safe_text(_field(raw, "replay_identity", "")).strip()
    return _TopologyNode(
        topology_id=topology_id,
        observed_stability=observed_stability,
        observed_alignment=observed_alignment,
        observed_pressure=observed_pressure,
        continuity_ok=continuity_ok,
        replay_identity=replay_identity,
    )


def _normalize_horizon_node(raw: Any, index: int) -> _HorizonNode:
    horizon_id = _safe_text(_field(raw, "horizon_id", "")).strip() or f"horizon_{index}"
    horizon_step = _safe_nonneg_float(_field(raw, "horizon_step", _field(raw, "horizon", index + 1)))
    if horizon_step <= 0.0:
        horizon_step = float(index + 1)
    reconciliation_delta = _clamp01(
        _safe_nonneg_float(_field(raw, "reconciliation_delta", _field(raw, "projection_delta", 0.0)))
    )
    return _HorizonNode(
        horizon_id=horizon_id,
        horizon_step=horizon_step,
        reconciliation_delta=reconciliation_delta,
    )


def _ordered_forecast_nodes(series: Sequence[Any]) -> Tuple[_ForecastNode, ...]:
    return tuple(_normalize_forecast_node(raw, index=i) for i, raw in enumerate(series))


def _ordered_topology_nodes(series: Sequence[Any]) -> Tuple[_TopologyNode, ...]:
    return tuple(_normalize_topology_node(raw, index=i) for i, raw in enumerate(series))


def _ordered_horizon_nodes(series: Sequence[Any]) -> Tuple[_HorizonNode, ...]:
    return tuple(_normalize_horizon_node(raw, index=i) for i, raw in enumerate(series))


def build_forecast_topology_reconciliation_scenario(
    *,
    scenario_id: str,
    forecast_series: Any,
    topology_series: Any,
    replay_horizon_series: Any,
) -> ForecastTopologyReconciliationScenario:
    normalized_id = _safe_text(scenario_id).strip()
    forecasts = _ordered_forecast_nodes(_safe_series(forecast_series))
    topologies = _ordered_topology_nodes(_safe_series(topology_series))
    horizons = _ordered_horizon_nodes(_safe_series(replay_horizon_series))
    return ForecastTopologyReconciliationScenario(
        scenario_id=normalized_id,
        forecast_series=tuple(node.to_dict() for node in forecasts),
        topology_series=tuple(node.to_dict() for node in topologies),
        replay_horizon_series=tuple(node.to_dict() for node in horizons),
    )


def _metric(name: str, value: float) -> ForecastTopologyReconciliationMetric:
    return ForecastTopologyReconciliationMetric(
        metric_name=name,
        metric_order=_METRIC_INDEX[name],
        metric_value=float(value),
    )


def _compute_metrics(
    scenario: ForecastTopologyReconciliationScenario,
) -> Tuple[ForecastTopologyReconciliationMetric, ...]:
    forecasts = _ordered_forecast_nodes(scenario.forecast_series)
    topologies = _ordered_topology_nodes(scenario.topology_series)
    horizons = _ordered_horizon_nodes(scenario.replay_horizon_series)

    topologies_by_id: Dict[str, list[_TopologyNode]] = {}
    for topology in topologies:
        topologies_by_id.setdefault(topology.topology_id, []).append(topology)

    topology_match_index: Dict[str, int] = {}
    matched_pairs: list[Tuple[_ForecastNode, _TopologyNode]] = []
    for forecast in forecasts:
        group = topologies_by_id.get(forecast.topology_id)
        if not group:
            continue
        group_index = topology_match_index.get(forecast.topology_id, 0)
        if group_index >= len(group):
            continue
        matched_pairs.append((forecast, group[group_index]))
        topology_match_index[forecast.topology_id] = group_index + 1

    pair_count = len(matched_pairs)
    if pair_count > 0:
        delta_sum = 0.0
        alignment_error_sum = 0.0
        drift_pressure_sum = 0.0
        replay_stability_sum = 0.0
        for forecast, topology in matched_pairs:
            delta_sum += abs(forecast.forecast_stability - topology.observed_stability)
            alignment_error_sum += abs(forecast.forecast_alignment - topology.observed_alignment)
            drift_pressure_sum += abs(forecast.forecast_pressure - topology.observed_pressure)
            replay_match = 1.0 if (
                not forecast.replay_identity
                or not topology.replay_identity
                or forecast.replay_identity == topology.replay_identity
            ) else 0.0
            continuity = 1.0 if topology.continuity_ok else 0.0
            replay_stability_sum += _clamp01((replay_match + continuity) * 0.5)

        forecast_topology_delta_score = _clamp01(delta_sum / float(pair_count))
        reconciliation_alignment_score = _clamp01(1.0 - (alignment_error_sum / float(pair_count)))
        forecast_drift_pressure = _clamp01(drift_pressure_sum / float(pair_count))
        replay_reconciliation_stability_score = _clamp01(replay_stability_sum / float(pair_count))
    else:
        forecast_topology_delta_score = 1.0
        reconciliation_alignment_score = 0.0
        forecast_drift_pressure = 1.0
        replay_reconciliation_stability_score = 0.0

    if horizons:
        weighted_delta_sum = 0.0
        total_weight = 0.0
        for node in horizons:
            total_weight += node.horizon_step
            weighted_delta_sum += node.horizon_step * node.reconciliation_delta
        horizon_reconciliation_gradient = _clamp01(
            weighted_delta_sum / total_weight if total_weight > 0.0 else 0.0
        )
    else:
        horizon_reconciliation_gradient = 0.0

    composite_pressure = _clamp01(
        (
            forecast_topology_delta_score
            + (1.0 - reconciliation_alignment_score)
            + horizon_reconciliation_gradient
            + forecast_drift_pressure
            + (1.0 - replay_reconciliation_stability_score)
        )
        / 5.0
    )
    reconciliation_confidence_score = _clamp01(1.0 - composite_pressure)

    return (
        _metric("forecast_topology_delta_score", forecast_topology_delta_score),
        _metric("reconciliation_alignment_score", reconciliation_alignment_score),
        _metric("horizon_reconciliation_gradient", horizon_reconciliation_gradient),
        _metric("forecast_drift_pressure", forecast_drift_pressure),
        _metric("replay_reconciliation_stability_score", replay_reconciliation_stability_score),
        _metric("reconciliation_confidence_score", reconciliation_confidence_score),
    )


def _metrics_hash(metrics: Sequence[ForecastTopologyReconciliationMetric]) -> str:
    payload = [metric.to_dict() for metric in metrics]
    return _sha256_hex(_canonical_json(payload).encode("utf-8"))


def _advisory_from_pressure(composite_pressure: float) -> str:
    bounded = _clamp01(composite_pressure)
    for advisory, threshold in _ADVISORY_THRESHOLDS:
        if bounded <= threshold:
            return advisory
    return "severe_reconciliation_instability"


def _reconciliation_hash(
    *,
    scenario: ForecastTopologyReconciliationScenario,
    metrics: Sequence[ForecastTopologyReconciliationMetric],
    violations: Sequence[str],
) -> str:
    payload = {
        "scenario": scenario.to_dict(),
        "metrics": [metric.to_dict() for metric in metrics],
        "violations": list(violations),
    }
    return _sha256_hex(_canonical_json(payload).encode("utf-8"))


def validate_forecast_topology_reconciliation(kernel: Any) -> Tuple[str, ...]:
    violations: list[str] = []
    try:
        scenario = _field(kernel, "scenario", None)
        metrics = tuple(_safe_series(_field(kernel, "metrics", ())))
        receipt = _field(kernel, "receipt", None)

        if scenario is None:
            violations.append("missing_scenario")
            return tuple(violations)

        if not _safe_text(_field(scenario, "scenario_id", "")).strip():
            violations.append("empty_scenario_id")

        forecast_series = _safe_series(_field(scenario, "forecast_series", ()))
        topology_series = _safe_series(_field(scenario, "topology_series", ()))

        if not forecast_series:
            violations.append("empty_forecast_series")
        if not topology_series:
            violations.append("empty_topology_series")

        for idx, row in enumerate(forecast_series):
            topology_id = _safe_text(_field(row, "topology_id", "")).strip()
            if not topology_id:
                violations.append(f"malformed_forecast_row:{idx}")

        for idx, row in enumerate(topology_series):
            topology_id = _safe_text(_field(row, "topology_id", "")).strip()
            if not topology_id:
                violations.append(f"malformed_topology_row:{idx}")

        metric_names = tuple(_safe_text(_field(metric, "metric_name", "")) for metric in metrics)
        if metric_names != _METRIC_ORDER:
            violations.append("metric_order_mismatch")

        by_name: Dict[str, float] = {}
        for metric in metrics:
            name = _safe_text(_field(metric, "metric_name", ""))
            value_raw = _field(metric, "metric_value", 0.0)
            try:
                value = float(value_raw)
            except Exception:
                violations.append(f"metric_non_numeric:{name or '?'}")
                continue
            if not math.isfinite(value):
                violations.append(f"metric_non_finite:{name or '?'}")
                continue
            by_name[name] = value

        for metric_name in _METRIC_ORDER:
            if metric_name in by_name and (by_name[metric_name] < 0.0 or by_name[metric_name] > 1.0):
                violations.append(f"metric_out_of_bounds:{metric_name}")

        if receipt is not None:
            actual_metrics_hash = _safe_text(_field(receipt, "metrics_hash", "")).strip()
            if actual_metrics_hash:
                expected_metrics_hash = _metrics_hash(metrics)
                if expected_metrics_hash != actual_metrics_hash:
                    violations.append("receipt_metrics_hash_mismatch")

            actual_scenario_hash = _safe_text(_field(receipt, "scenario_hash", "")).strip()
            if actual_scenario_hash:
                expected_scenario_hash = scenario.stable_hash()
                if expected_scenario_hash != actual_scenario_hash:
                    violations.append("receipt_scenario_hash_mismatch")

            actual_reconciliation_hash = _safe_text(_field(receipt, "reconciliation_hash", "")).strip()
            expected_reconciliation_hash = _reconciliation_hash(
                scenario=scenario,
                metrics=metrics,
                violations=tuple(sorted(set(violations))),
            )
            if actual_reconciliation_hash and expected_reconciliation_hash:
                if actual_reconciliation_hash != expected_reconciliation_hash:
                    violations.append("receipt_reconciliation_hash_mismatch")

            actual_receipt_hash = _safe_text(_field(receipt, "receipt_hash", "")).strip()
            if actual_receipt_hash:
                confidence_raw = _field(receipt, "reconciliation_confidence_score", 0.0)
                try:
                    confidence = float(confidence_raw)
                except Exception:
                    confidence = 0.0
                expected_receipt_payload = {
                    "scenario_hash": _safe_text(_field(receipt, "scenario_hash", "")).strip(),
                    "metrics_hash": _safe_text(_field(receipt, "metrics_hash", "")).strip(),
                    "reconciliation_hash": _safe_text(_field(receipt, "reconciliation_hash", "")).strip(),
                    "reconciliation_confidence_score": confidence,
                    "advisory_output": _safe_text(_field(receipt, "advisory_output", "")).strip(),
                }
                expected_receipt_hash = _sha256_hex(_canonical_json(expected_receipt_payload).encode("utf-8"))
                if expected_receipt_hash != actual_receipt_hash:
                    violations.append("receipt_hash_mismatch")
    except Exception as exc:
        violations.append(f"validator_internal_error:{_safe_text(exc)}")

    return tuple(sorted(set(violations)))


def build_forecast_topology_reconciliation_receipt(
    *,
    scenario: ForecastTopologyReconciliationScenario,
    metrics: Sequence[ForecastTopologyReconciliationMetric],
    reconciliation_hash: str,
    advisory_output: str,
) -> ForecastTopologyReconciliationReceipt:
    confidence = 0.0
    for metric in metrics:
        if metric.metric_name == "reconciliation_confidence_score":
            confidence = float(metric.metric_value)
            break

    payload = {
        "scenario_hash": scenario.stable_hash(),
        "metrics_hash": _metrics_hash(metrics),
        "reconciliation_hash": _safe_text(reconciliation_hash).strip(),
        "reconciliation_confidence_score": _clamp01(confidence),
        "advisory_output": _safe_text(advisory_output).strip(),
    }
    receipt_hash = _sha256_hex(_canonical_json(payload).encode("utf-8"))
    return ForecastTopologyReconciliationReceipt(
        scenario_hash=payload["scenario_hash"],
        metrics_hash=payload["metrics_hash"],
        reconciliation_hash=payload["reconciliation_hash"],
        reconciliation_confidence_score=payload["reconciliation_confidence_score"],
        advisory_output=payload["advisory_output"],
        receipt_hash=receipt_hash,
    )


def run_governance_forecast_topology_reconciliation(
    *,
    scenario: ForecastTopologyReconciliationScenario,
) -> GovernanceForecastTopologyReconciliationKernel:
    metrics = _compute_metrics(scenario)
    by_name = {metric.metric_name: metric.metric_value for metric in metrics}
    composite_pressure = _clamp01(
        (
            by_name.get("forecast_topology_delta_score", 0.0)
            + (1.0 - by_name.get("reconciliation_alignment_score", 0.0))
            + by_name.get("horizon_reconciliation_gradient", 0.0)
            + by_name.get("forecast_drift_pressure", 0.0)
            + (1.0 - by_name.get("replay_reconciliation_stability_score", 0.0))
        )
        / 5.0
    )
    advisory_output = _advisory_from_pressure(composite_pressure)
    reconciliation_analysis = {
        "scenario_id": scenario.scenario_id,
        "scenario_hash": scenario.stable_hash(),
        "metrics_hash": _metrics_hash(metrics),
        "composite_pressure": composite_pressure,
        "advisory_output": advisory_output,
    }

    provisional = GovernanceForecastTopologyReconciliationKernel(
        scenario=scenario,
        metrics=metrics,
        reconciliation_analysis=reconciliation_analysis,
        advisory_output=advisory_output,
        violations=(),
        receipt=ForecastTopologyReconciliationReceipt("", "", "", 0.0, "", ""),
        reconciliation_hash="",
    )
    violations = validate_forecast_topology_reconciliation(provisional)
    reconciliation_hash = _reconciliation_hash(scenario=scenario, metrics=metrics, violations=violations)
    receipt = build_forecast_topology_reconciliation_receipt(
        scenario=scenario,
        metrics=metrics,
        reconciliation_hash=reconciliation_hash,
        advisory_output=advisory_output,
    )
    return GovernanceForecastTopologyReconciliationKernel(
        scenario=scenario,
        metrics=metrics,
        reconciliation_analysis=reconciliation_analysis,
        advisory_output=advisory_output,
        violations=violations,
        receipt=receipt,
        reconciliation_hash=reconciliation_hash,
    )


def compare_forecast_topology_replay(
    baseline: GovernanceForecastTopologyReconciliationKernel,
    replay: GovernanceForecastTopologyReconciliationKernel,
) -> Dict[str, Any]:
    baseline_by_name = {metric.metric_name: metric.metric_value for metric in baseline.metrics}
    replay_by_name = {metric.metric_name: metric.metric_value for metric in replay.metrics}

    metric_delta = tuple(
        (
            name,
            float(replay_by_name.get(name, 0.0) - baseline_by_name.get(name, 0.0)),
        )
        for name in _METRIC_ORDER
    )

    mismatches: list[str] = []
    if baseline.scenario.stable_hash() != replay.scenario.stable_hash():
        mismatches.append("scenario_hash")
    if baseline.reconciliation_hash != replay.reconciliation_hash:
        mismatches.append("reconciliation_hash")
    if baseline.receipt.receipt_hash != replay.receipt.receipt_hash:
        mismatches.append("receipt_hash")

    return {
        "is_stable_replay": len(mismatches) == 0,
        "mismatches": tuple(mismatches),
        "metric_delta": metric_delta,
        "baseline_hash": baseline.stable_hash(),
        "replay_hash": replay.stable_hash(),
    }


def summarize_forecast_topology_reconciliation(
    kernel: GovernanceForecastTopologyReconciliationKernel,
) -> str:
    lines = [
        f"scenario_id={kernel.scenario.scenario_id}",
        f"scenario_hash={kernel.scenario.stable_hash()}",
        f"reconciliation_hash={kernel.reconciliation_hash}",
        f"receipt_hash={kernel.receipt.receipt_hash}",
        f"advisory_output={kernel.advisory_output}",
        "metrics:",
    ]
    for metric in kernel.metrics:
        lines.append(f"- {metric.metric_order}:{metric.metric_name}={metric.metric_value:.12f}")
    if kernel.violations:
        lines.append("violations:")
        for violation in kernel.violations:
            lines.append(f"- {violation}")
    else:
        lines.append("violations: none")
    return "\n".join(lines)


__all__ = [
    "ForecastTopologyReconciliationScenario",
    "ForecastTopologyReconciliationMetric",
    "ForecastTopologyReconciliationReceipt",
    "GovernanceForecastTopologyReconciliationKernel",
    "build_forecast_topology_reconciliation_scenario",
    "run_governance_forecast_topology_reconciliation",
    "validate_forecast_topology_reconciliation",
    "build_forecast_topology_reconciliation_receipt",
    "compare_forecast_topology_replay",
    "summarize_forecast_topology_reconciliation",
]
