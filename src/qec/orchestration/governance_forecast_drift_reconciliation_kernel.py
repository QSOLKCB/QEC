"""v137.19.5 — Governance Forecast Drift Reconciliation Kernel.

Additive deterministic reconciliation-analysis over forecast and realized artifacts.

Canonical model:
- forecast_series + realized_evolution_series + realized_drift_series
  -> reconciliation_analysis + reconciliation_receipt

Invariants:
- no randomness
- no async
- no external I/O
- deterministic ordering
- canonical JSON + SHA-256
- validator must never raise
- no mutation of inputs
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Sequence, Tuple


_METRIC_ORDER: Tuple[str, ...] = (
    "forecast_error_rate",
    "realized_vs_projected_drift_delta",
    "stability_prediction_accuracy",
    "continuity_forecast_error",
    "severity_forecast_residual",
    "replay_reconciliation_score",
    "forecast_confidence_calibration",
)
_METRIC_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(_METRIC_ORDER)}


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


def _safe_nonneg_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
    except Exception:
        return 0
    return parsed if parsed >= 0 else 0


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
    evolution_id: str
    basin_id: str
    projected_drift: float
    projected_stability: float
    projected_severity: float
    continuity_expected: bool
    forecast_confidence: float
    replay_identity: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evolution_id": self.evolution_id,
            "basin_id": self.basin_id,
            "projected_drift": self.projected_drift,
            "projected_stability": self.projected_stability,
            "projected_severity": self.projected_severity,
            "continuity_expected": self.continuity_expected,
            "forecast_confidence": self.forecast_confidence,
            "replay_identity": self.replay_identity,
        }


@dataclass(frozen=True)
class _RealizedEvolutionNode:
    evolution_id: str
    topology_id: str
    basin_id: str
    continuity_ok: bool
    severity: float
    stability_realized: float
    replay_identity: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evolution_id": self.evolution_id,
            "topology_id": self.topology_id,
            "basin_id": self.basin_id,
            "continuity_ok": self.continuity_ok,
            "severity": self.severity,
            "stability_realized": self.stability_realized,
            "replay_identity": self.replay_identity,
        }


@dataclass(frozen=True)
class _RealizedDriftNode:
    drift_id: str
    from_basin: str
    to_basin: str
    transition_count: int
    drift_magnitude: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drift_id": self.drift_id,
            "from_basin": self.from_basin,
            "to_basin": self.to_basin,
            "transition_count": self.transition_count,
            "drift_magnitude": self.drift_magnitude,
        }


@dataclass(frozen=True)
class ForecastReconciliationScenario:
    scenario_id: str
    forecast_series: Tuple[Mapping[str, Any], ...]
    realized_evolution_series: Tuple[Mapping[str, Any], ...]
    realized_drift_series: Tuple[Mapping[str, Any], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "forecast_series": [dict(item) for item in self.forecast_series],
            "realized_evolution_series": [dict(item) for item in self.realized_evolution_series],
            "realized_drift_series": [dict(item) for item in self.realized_drift_series],
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_canonical_json().encode("utf-8"))


@dataclass(frozen=True)
class ForecastReconciliationMetric:
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
class ForecastReconciliationReceipt:
    scenario_hash: str
    metrics_hash: str
    reconciliation_hash: str
    replay_reconciliation_score: float
    receipt_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_hash": self.scenario_hash,
            "metrics_hash": self.metrics_hash,
            "reconciliation_hash": self.reconciliation_hash,
            "replay_reconciliation_score": self.replay_reconciliation_score,
            "receipt_hash": self.receipt_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.receipt_hash


@dataclass(frozen=True)
class GovernanceForecastDriftReconciliationKernel:
    scenario: ForecastReconciliationScenario
    metrics: Tuple[ForecastReconciliationMetric, ...]
    violations: Tuple[str, ...]
    receipt: ForecastReconciliationReceipt
    reconciliation_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario.to_dict(),
            "metrics": [metric.to_dict() for metric in self.metrics],
            "violations": list(self.violations),
            "receipt": self.receipt.to_dict(),
            "reconciliation_hash": self.reconciliation_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.reconciliation_hash


def _normalize_forecast_node(raw: Any, index: int) -> _ForecastNode:
    evolution_id = _safe_text(_field(raw, "evolution_id", "")).strip() or f"forecast_{index}"
    basin_id = _safe_text(_field(raw, "basin_id", _field(raw, "decision_basin", ""))).strip() or "unknown"
    projected_drift = _safe_nonneg_float(_field(raw, "projected_drift", _field(raw, "drift_projection", 0.0)))
    projected_stability = _clamp01(
        _safe_nonneg_float(_field(raw, "projected_stability", _field(raw, "stability_forecast", 1.0)))
    )
    projected_severity = _clamp01(
        _safe_nonneg_float(_field(raw, "projected_severity", _field(raw, "severity_projection", 0.0)))
    )
    continuity_expected = _safe_bool(_field(raw, "continuity_expected", _field(raw, "continuity_ok", False)))
    forecast_confidence = _clamp01(_safe_nonneg_float(_field(raw, "forecast_confidence", 0.0)))
    replay_identity = _safe_text(_field(raw, "replay_identity", "")).strip()
    return _ForecastNode(
        evolution_id=evolution_id,
        basin_id=basin_id,
        projected_drift=projected_drift,
        projected_stability=projected_stability,
        projected_severity=projected_severity,
        continuity_expected=continuity_expected,
        forecast_confidence=forecast_confidence,
        replay_identity=replay_identity,
    )


def _normalize_realized_evolution_node(raw: Any, index: int) -> _RealizedEvolutionNode:
    evolution_id = (
        _safe_text(_field(raw, "evolution_id", "")).strip()
        or _safe_text(_field(raw, "topology_id", "")).strip()
        or f"evolution_{index}"
    )
    topology_id = _safe_text(_field(raw, "topology_id", "")).strip()
    basin_id = _safe_text(_field(raw, "basin_id", _field(raw, "decision_basin", ""))).strip() or "unknown"
    continuity_ok = _safe_bool(_field(raw, "continuity_ok", False))
    severity = _clamp01(_safe_nonneg_float(_field(raw, "severity", _field(raw, "topology_severity", 0.0))))
    stability_realized = _clamp01(
        _safe_nonneg_float(
            _field(raw, "stability_realized", _field(raw, "stability", 1.0))
        )
    )
    replay_identity = _safe_text(_field(raw, "replay_identity", "")).strip()
    return _RealizedEvolutionNode(
        evolution_id=evolution_id,
        topology_id=topology_id,
        basin_id=basin_id,
        continuity_ok=continuity_ok,
        severity=severity,
        stability_realized=stability_realized,
        replay_identity=replay_identity,
    )


def _normalize_realized_drift_node(raw: Any, index: int) -> _RealizedDriftNode:
    drift_id = _safe_text(_field(raw, "drift_id", "")).strip() or f"drift_{index}"
    from_basin = _safe_text(_field(raw, "from_basin", "")).strip() or "unknown"
    to_basin = _safe_text(_field(raw, "to_basin", "")).strip() or "unknown"
    transition_count = _safe_nonneg_int(_field(raw, "transition_count", 0))
    drift_magnitude = _safe_nonneg_float(_field(raw, "drift_magnitude", 0.0))
    return _RealizedDriftNode(
        drift_id=drift_id,
        from_basin=from_basin,
        to_basin=to_basin,
        transition_count=transition_count,
        drift_magnitude=drift_magnitude,
    )


def _ordered_forecast_nodes(series: Sequence[Any]) -> Tuple[_ForecastNode, ...]:
    return tuple(_normalize_forecast_node(raw, index=i) for i, raw in enumerate(series))


def _ordered_realized_evolution_nodes(series: Sequence[Any]) -> Tuple[_RealizedEvolutionNode, ...]:
    return tuple(_normalize_realized_evolution_node(raw, index=i) for i, raw in enumerate(series))


def _ordered_realized_drift_nodes(series: Sequence[Any]) -> Tuple[_RealizedDriftNode, ...]:
    return tuple(_normalize_realized_drift_node(raw, index=i) for i, raw in enumerate(series))


def build_forecast_reconciliation_scenario(
    *,
    scenario_id: str,
    forecast_series: Any,
    realized_evolution_series: Any,
    realized_drift_series: Any,
) -> ForecastReconciliationScenario:
    normalized_id = _safe_text(scenario_id).strip()
    normalized_forecast = _ordered_forecast_nodes(_safe_series(forecast_series))
    normalized_evolution = _ordered_realized_evolution_nodes(_safe_series(realized_evolution_series))
    normalized_drift = _ordered_realized_drift_nodes(_safe_series(realized_drift_series))
    return ForecastReconciliationScenario(
        scenario_id=normalized_id,
        forecast_series=tuple(node.to_dict() for node in normalized_forecast),
        realized_evolution_series=tuple(node.to_dict() for node in normalized_evolution),
        realized_drift_series=tuple(node.to_dict() for node in normalized_drift),
    )


def _metric(name: str, value: float) -> ForecastReconciliationMetric:
    return ForecastReconciliationMetric(
        metric_name=name,
        metric_order=_METRIC_INDEX[name],
        metric_value=float(value),
    )


def _compute_metrics(scenario: ForecastReconciliationScenario) -> Tuple[ForecastReconciliationMetric, ...]:
    forecasts = _ordered_forecast_nodes(scenario.forecast_series)
    evolutions = _ordered_realized_evolution_nodes(scenario.realized_evolution_series)
    drifts = _ordered_realized_drift_nodes(scenario.realized_drift_series)

    evolutions_by_id: Dict[str, list[_RealizedEvolutionNode]] = {}
    for realized in evolutions:
        evolutions_by_id.setdefault(realized.evolution_id, []).append(realized)

    evolution_match_index: Dict[str, int] = {}
    matched_pairs: list[Tuple[_ForecastNode, _RealizedEvolutionNode]] = []
    for forecast in forecasts:
        realized_group = evolutions_by_id.get(forecast.evolution_id)
        if not realized_group:
            continue
        group_index = evolution_match_index.get(forecast.evolution_id, 0)
        if group_index >= len(realized_group):
            continue
        matched_pairs.append((forecast, realized_group[group_index]))
        evolution_match_index[forecast.evolution_id] = group_index + 1

    pair_count = len(matched_pairs)
    if pair_count > 0:
        error_sum = 0.0
        stability_error_sum = 0.0
        continuity_error_sum = 0.0
        severity_residual_sum = 0.0
        replay_match_sum = 0.0
        for forecast, realized in matched_pairs:
            realized_instability = _clamp01((1.0 - realized.stability_realized + realized.severity) * 0.5)
            projected_instability = _clamp01((1.0 - forecast.projected_stability + forecast.projected_severity) * 0.5)
            error_sum += abs(projected_instability - realized_instability)
            stability_error_sum += abs(forecast.projected_stability - realized.stability_realized)
            continuity_error_sum += abs((1.0 if forecast.continuity_expected else 0.0) - (1.0 if realized.continuity_ok else 0.0))
            severity_residual_sum += abs(forecast.projected_severity - realized.severity)
            replay_match = 1.0 if (
                forecast.evolution_id == realized.evolution_id
                and (
                    not forecast.replay_identity
                    or not realized.replay_identity
                    or forecast.replay_identity == realized.replay_identity
                )
            ) else 0.0
            replay_match_sum += replay_match

        forecast_error_rate = _clamp01(error_sum / float(pair_count))
        stability_prediction_accuracy = _clamp01(1.0 - (stability_error_sum / float(pair_count)))
        continuity_forecast_error = _clamp01(continuity_error_sum / float(pair_count))
        severity_forecast_residual = _clamp01(severity_residual_sum / float(pair_count))
        replay_reconciliation_score = _clamp01(replay_match_sum / float(pair_count))
    else:
        forecast_error_rate = 1.0
        stability_prediction_accuracy = 0.0
        continuity_forecast_error = 1.0
        severity_forecast_residual = 1.0
        replay_reconciliation_score = 0.0

    avg_projected_drift = 0.0 if not forecasts else sum(item.projected_drift for item in forecasts) / float(len(forecasts))
    avg_realized_drift = 0.0 if not drifts else sum(item.drift_magnitude for item in drifts) / float(len(drifts))
    realized_vs_projected_drift_delta = _clamp01(abs(avg_realized_drift - avg_projected_drift))

    avg_confidence = 0.0 if not forecasts else sum(item.forecast_confidence for item in forecasts) / float(len(forecasts))
    expected_confidence = _clamp01(1.0 - forecast_error_rate)
    forecast_confidence_calibration = _clamp01(1.0 - abs(avg_confidence - expected_confidence))

    return (
        _metric("forecast_error_rate", forecast_error_rate),
        _metric("realized_vs_projected_drift_delta", realized_vs_projected_drift_delta),
        _metric("stability_prediction_accuracy", stability_prediction_accuracy),
        _metric("continuity_forecast_error", continuity_forecast_error),
        _metric("severity_forecast_residual", severity_forecast_residual),
        _metric("replay_reconciliation_score", replay_reconciliation_score),
        _metric("forecast_confidence_calibration", forecast_confidence_calibration),
    )


def _metrics_hash(metrics: Sequence[ForecastReconciliationMetric]) -> str:
    payload = [metric.to_dict() for metric in metrics]
    return _sha256_hex(_canonical_json(payload).encode("utf-8"))


def _reconciliation_hash(
    *,
    scenario: ForecastReconciliationScenario,
    metrics: Sequence[ForecastReconciliationMetric],
    violations: Sequence[str],
) -> str:
    body = {
        "scenario": scenario.to_dict(),
        "metrics": [metric.to_dict() for metric in metrics],
        "violations": list(violations),
    }
    return _sha256_hex(_canonical_json(body).encode("utf-8"))


def validate_forecast_reconciliation(kernel: Any) -> Tuple[str, ...]:
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
        realized_evolution_series = _safe_series(_field(scenario, "realized_evolution_series", ()))
        realized_drift_series = _safe_series(_field(scenario, "realized_drift_series", ()))

        if not forecast_series:
            violations.append("empty_forecast_series")
        if not realized_evolution_series:
            violations.append("empty_realized_evolution_series")

        for idx, row in enumerate(forecast_series):
            evolution_id = _safe_text(_field(row, "evolution_id", "")).strip()
            if not evolution_id:
                violations.append(f"malformed_forecast_row:{idx}")

        for idx, row in enumerate(realized_evolution_series):
            evolution_id = _safe_text(_field(row, "evolution_id", "")).strip()
            if not evolution_id:
                violations.append(f"malformed_realized_evolution_row:{idx}")

        for idx, row in enumerate(realized_drift_series):
            from_basin = _safe_text(_field(row, "from_basin", "")).strip()
            to_basin = _safe_text(_field(row, "to_basin", "")).strip()
            if not from_basin or not to_basin:
                violations.append(f"malformed_realized_drift_row:{idx}")

        names = tuple(_safe_text(_field(m, "metric_name", "")) for m in metrics)
        if names != _METRIC_ORDER:
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

        for key in _METRIC_ORDER:
            if key in by_name and (by_name[key] < 0.0 or by_name[key] > 1.0):
                violations.append(f"metric_out_of_bounds:{key}")

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
            expected_reconciliation_hash = _safe_text(_field(kernel, "reconciliation_hash", "")).strip()
            if actual_reconciliation_hash and expected_reconciliation_hash:
                if actual_reconciliation_hash != expected_reconciliation_hash:
                    violations.append("receipt_reconciliation_hash_mismatch")

            actual_receipt_hash = _safe_text(_field(receipt, "receipt_hash", "")).strip()
            if actual_receipt_hash:
                replay_score_raw = _field(receipt, "replay_reconciliation_score", 0.0)
                try:
                    replay_score = float(replay_score_raw)
                except Exception:
                    replay_score = 0.0
                expected_receipt_payload = {
                    "scenario_hash": _safe_text(_field(receipt, "scenario_hash", "")).strip(),
                    "metrics_hash": _safe_text(_field(receipt, "metrics_hash", "")).strip(),
                    "reconciliation_hash": _safe_text(_field(receipt, "reconciliation_hash", "")).strip(),
                    "replay_reconciliation_score": replay_score,
                }
                expected_receipt_hash = _sha256_hex(_canonical_json(expected_receipt_payload).encode("utf-8"))
                if expected_receipt_hash != actual_receipt_hash:
                    violations.append("receipt_hash_mismatch")
    except Exception as exc:
        violations.append(f"validator_internal_error:{_safe_text(exc)}")

    return tuple(sorted(set(violations)))


def build_forecast_reconciliation_receipt(
    *,
    scenario: ForecastReconciliationScenario,
    metrics: Sequence[ForecastReconciliationMetric],
    reconciliation_hash: str,
) -> ForecastReconciliationReceipt:
    replay_reconciliation_score = 0.0
    for metric in metrics:
        if metric.metric_name == "replay_reconciliation_score":
            replay_reconciliation_score = float(metric.metric_value)
            break

    payload = {
        "scenario_hash": scenario.stable_hash(),
        "metrics_hash": _metrics_hash(metrics),
        "reconciliation_hash": _safe_text(reconciliation_hash).strip(),
        "replay_reconciliation_score": replay_reconciliation_score,
    }
    receipt_hash = _sha256_hex(_canonical_json(payload).encode("utf-8"))
    return ForecastReconciliationReceipt(
        scenario_hash=payload["scenario_hash"],
        metrics_hash=payload["metrics_hash"],
        reconciliation_hash=payload["reconciliation_hash"],
        replay_reconciliation_score=replay_reconciliation_score,
        receipt_hash=receipt_hash,
    )


def run_governance_forecast_reconciliation(
    *,
    scenario: ForecastReconciliationScenario,
) -> GovernanceForecastDriftReconciliationKernel:
    metrics = _compute_metrics(scenario)
    provisional = GovernanceForecastDriftReconciliationKernel(
        scenario=scenario,
        metrics=metrics,
        violations=(),
        receipt=ForecastReconciliationReceipt("", "", "", 0.0, ""),
        reconciliation_hash="",
    )
    violations = validate_forecast_reconciliation(provisional)
    reconciliation_hash = _reconciliation_hash(scenario=scenario, metrics=metrics, violations=violations)
    receipt = build_forecast_reconciliation_receipt(
        scenario=scenario,
        metrics=metrics,
        reconciliation_hash=reconciliation_hash,
    )
    return GovernanceForecastDriftReconciliationKernel(
        scenario=scenario,
        metrics=metrics,
        violations=violations,
        receipt=receipt,
        reconciliation_hash=reconciliation_hash,
    )


def compare_forecast_reconciliation_replay(
    baseline: GovernanceForecastDriftReconciliationKernel,
    replay: GovernanceForecastDriftReconciliationKernel,
) -> Dict[str, Any]:
    metric_delta: Dict[str, float] = {}
    baseline_by_name = {m.metric_name: m.metric_value for m in baseline.metrics}
    replay_by_name = {m.metric_name: m.metric_value for m in replay.metrics}

    for name in _METRIC_ORDER:
        metric_delta[name] = float(replay_by_name.get(name, 0.0) - baseline_by_name.get(name, 0.0))

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
        "metric_delta": tuple((name, metric_delta[name]) for name in _METRIC_ORDER),
        "baseline_hash": baseline.stable_hash(),
        "replay_hash": replay.stable_hash(),
    }


def summarize_forecast_reconciliation(kernel: GovernanceForecastDriftReconciliationKernel) -> str:
    lines = [
        f"scenario_id={kernel.scenario.scenario_id}",
        f"scenario_hash={kernel.scenario.stable_hash()}",
        f"reconciliation_hash={kernel.reconciliation_hash}",
        f"receipt_hash={kernel.receipt.receipt_hash}",
        "metrics:",
    ]
    for metric in kernel.metrics:
        lines.append(f"- {metric.metric_order}:{metric.metric_name}={metric.metric_value:.12f}")
    if kernel.violations:
        lines.append("violations:")
        for item in kernel.violations:
            lines.append(f"- {item}")
    else:
        lines.append("violations: none")
    return "\n".join(lines)


__all__ = [
    "ForecastReconciliationScenario",
    "ForecastReconciliationMetric",
    "ForecastReconciliationReceipt",
    "GovernanceForecastDriftReconciliationKernel",
    "build_forecast_reconciliation_scenario",
    "run_governance_forecast_reconciliation",
    "validate_forecast_reconciliation",
    "build_forecast_reconciliation_receipt",
    "compare_forecast_reconciliation_replay",
    "summarize_forecast_reconciliation",
]
