"""v137.19.4 — Governance Stability Forecast Kernel.

Additive deterministic forecast-analysis over topology evolution artifacts.

Canonical model:
- evolution_series + drift_series -> forecast_analysis + forecast_receipt

Invariants:
- no randomness
- no async
- no external I/O
- deterministic ordering
- canonical JSON + SHA-256
- validator never raises
- no mutation of inputs
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Sequence, Tuple


_METRIC_ORDER: Tuple[str, ...] = (
    "projected_instability_risk",
    "basin_decay_forecast",
    "transition_acceleration_score",
    "replay_stability_forecast",
    "severity_projection_score",
    "continuity_failure_risk",
    "forecast_confidence_score",
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
class _EvolutionNode:
    evolution_id: str
    basin_id: str
    continuity_ok: bool
    severity: float
    replay_identity: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evolution_id": self.evolution_id,
            "basin_id": self.basin_id,
            "continuity_ok": self.continuity_ok,
            "severity": self.severity,
            "replay_identity": self.replay_identity,
        }


@dataclass(frozen=True)
class _DriftNode:
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
class StabilityForecastScenario:
    scenario_id: str
    evolution_series: Tuple[Mapping[str, Any], ...]
    drift_series: Tuple[Mapping[str, Any], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "evolution_series": [dict(item) for item in self.evolution_series],
            "drift_series": [dict(item) for item in self.drift_series],
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_canonical_json().encode("utf-8"))


@dataclass(frozen=True)
class StabilityForecastMetric:
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
class StabilityForecastReceipt:
    scenario_hash: str
    metrics_hash: str
    forecast_hash: str
    severity_projection_score: float
    receipt_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_hash": self.scenario_hash,
            "metrics_hash": self.metrics_hash,
            "forecast_hash": self.forecast_hash,
            "severity_projection_score": self.severity_projection_score,
            "receipt_hash": self.receipt_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.receipt_hash


@dataclass(frozen=True)
class GovernanceStabilityForecastKernel:
    scenario: StabilityForecastScenario
    metrics: Tuple[StabilityForecastMetric, ...]
    violations: Tuple[str, ...]
    receipt: StabilityForecastReceipt
    forecast_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario.to_dict(),
            "metrics": [metric.to_dict() for metric in self.metrics],
            "violations": list(self.violations),
            "receipt": self.receipt.to_dict(),
            "forecast_hash": self.forecast_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.forecast_hash


def _normalize_evolution_node(raw: Any, index: int) -> _EvolutionNode:
    evolution_id = _safe_text(_field(raw, "evolution_id", "")).strip() or f"evolution_{index}"
    basin_id = _safe_text(_field(raw, "basin_id", _field(raw, "decision_basin", ""))).strip() or "unknown"
    continuity_ok = _safe_bool(_field(raw, "continuity_ok", False))
    severity = _safe_nonneg_float(_field(raw, "severity", _field(raw, "topology_severity", 0.0)))
    replay_identity = _safe_text(_field(raw, "replay_identity", "")).strip()
    return _EvolutionNode(
        evolution_id=evolution_id,
        basin_id=basin_id,
        continuity_ok=continuity_ok,
        severity=severity,
        replay_identity=replay_identity,
    )


def _normalize_drift_node(raw: Any, index: int) -> _DriftNode:
    drift_id = _safe_text(_field(raw, "drift_id", "")).strip() or f"drift_{index}"
    from_basin = _safe_text(_field(raw, "from_basin", "")).strip() or "unknown"
    to_basin = _safe_text(_field(raw, "to_basin", "")).strip() or "unknown"
    transition_count = _safe_nonneg_int(_field(raw, "transition_count", 1))
    drift_magnitude = _safe_nonneg_float(_field(raw, "drift_magnitude", 0.0))
    return _DriftNode(
        drift_id=drift_id,
        from_basin=from_basin,
        to_basin=to_basin,
        transition_count=transition_count,
        drift_magnitude=drift_magnitude,
    )


def _ordered_evolution_nodes(series: Sequence[Any]) -> Tuple[_EvolutionNode, ...]:
    return tuple(_normalize_evolution_node(raw, index=i) for i, raw in enumerate(series))


def _ordered_drift_nodes(series: Sequence[Any]) -> Tuple[_DriftNode, ...]:
    return tuple(_normalize_drift_node(raw, index=i) for i, raw in enumerate(series))


def build_stability_forecast_scenario(
    *,
    scenario_id: str,
    evolution_series: Any,
    drift_series: Any,
) -> StabilityForecastScenario:
    normalized_id = _safe_text(scenario_id).strip()
    normalized_evolution = _ordered_evolution_nodes(_safe_series(evolution_series))
    normalized_drift = _ordered_drift_nodes(_safe_series(drift_series))
    return StabilityForecastScenario(
        scenario_id=normalized_id,
        evolution_series=tuple(node.to_dict() for node in normalized_evolution),
        drift_series=tuple(node.to_dict() for node in normalized_drift),
    )


def _metric(name: str, value: float) -> StabilityForecastMetric:
    return StabilityForecastMetric(
        metric_name=name,
        metric_order=_METRIC_INDEX[name],
        metric_value=float(value),
    )


def _compute_metrics(scenario: StabilityForecastScenario) -> Tuple[StabilityForecastMetric, ...]:
    evolutions = _ordered_evolution_nodes(scenario.evolution_series)
    drifts = _ordered_drift_nodes(scenario.drift_series)

    evolution_count = len(evolutions)
    drift_count = len(drifts)

    basin_changes = 0
    stable_replay_pairs = 0
    increasing_transition_pairs = 0
    for i in range(1, evolution_count):
        prev = evolutions[i - 1]
        curr = evolutions[i]
        if prev.basin_id != curr.basin_id:
            basin_changes += 1
        if prev.replay_identity and prev.replay_identity == curr.replay_identity:
            stable_replay_pairs += 1

    for i in range(1, drift_count):
        if drifts[i].transition_count > drifts[i - 1].transition_count:
            increasing_transition_pairs += 1

    basin_counts: Dict[str, int] = {}
    continuity_true = 0
    severity_sum = 0.0
    for node in evolutions:
        basin_counts[node.basin_id] = basin_counts.get(node.basin_id, 0) + 1
        if node.continuity_ok:
            continuity_true += 1
        severity_sum += node.severity

    dominant_basin_count = max(basin_counts.values()) if basin_counts else 0
    basin_decay_forecast = 0.0
    if evolution_count > 0:
        basin_decay_forecast = 1.0 - (float(dominant_basin_count) / float(evolution_count))

    continuity_failure_risk = 0.0
    if evolution_count > 0:
        continuity_failure_risk = 1.0 - (float(continuity_true) / float(evolution_count))

    replay_stability_forecast = 1.0
    if evolution_count >= 2:
        replay_stability_forecast = float(stable_replay_pairs) / float(evolution_count - 1)

    avg_severity = 0.0 if evolution_count == 0 else (severity_sum / float(evolution_count))

    transition_acceleration_score = 0.0
    if drift_count >= 2:
        transition_acceleration_score = float(increasing_transition_pairs) / float(drift_count - 1)

    total_drift_magnitude = sum(node.drift_magnitude for node in drifts)
    avg_drift_magnitude = 0.0 if drift_count == 0 else total_drift_magnitude / float(drift_count)

    projected_instability_risk = _clamp01(
        0.25 * _clamp01(float(basin_changes) / float(max(1, evolution_count - 1)))
        + 0.25 * _clamp01(continuity_failure_risk)
        + 0.25 * _clamp01(avg_severity)
        + 0.25 * _clamp01(avg_drift_magnitude)
    )

    severity_projection_score = _clamp01(
        0.6 * _clamp01(avg_severity)
        + 0.2 * _clamp01(transition_acceleration_score)
        + 0.2 * _clamp01(avg_drift_magnitude)
    )

    forecast_confidence_score = _clamp01(
        0.5 * _clamp01(replay_stability_forecast)
        + 0.3 * (1.0 - _clamp01(projected_instability_risk))
        + 0.2 * (1.0 - _clamp01(basin_decay_forecast))
    )

    metrics = (
        _metric("projected_instability_risk", projected_instability_risk),
        _metric("basin_decay_forecast", _clamp01(basin_decay_forecast)),
        _metric("transition_acceleration_score", _clamp01(transition_acceleration_score)),
        _metric("replay_stability_forecast", _clamp01(replay_stability_forecast)),
        _metric("severity_projection_score", severity_projection_score),
        _metric("continuity_failure_risk", _clamp01(continuity_failure_risk)),
        _metric("forecast_confidence_score", forecast_confidence_score),
    )
    return metrics


def _metrics_hash(metrics: Sequence[StabilityForecastMetric]) -> str:
    payload = [metric.to_dict() for metric in metrics]
    return _sha256_hex(_canonical_json(payload).encode("utf-8"))


def _forecast_hash(
    *,
    scenario: StabilityForecastScenario,
    metrics: Sequence[StabilityForecastMetric],
    violations: Sequence[str],
) -> str:
    body = {
        "scenario": scenario.to_dict(),
        "metrics": [metric.to_dict() for metric in metrics],
        "violations": list(violations),
    }
    return _sha256_hex(_canonical_json(body).encode("utf-8"))


def validate_stability_forecast(kernel: Any) -> Tuple[str, ...]:
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

        evolution_series = _safe_series(_field(scenario, "evolution_series", ()))
        drift_series = _safe_series(_field(scenario, "drift_series", ()))

        if not evolution_series:
            violations.append("empty_evolution_series")

        for idx, row in enumerate(evolution_series):
            basin = _safe_text(_field(row, "basin_id", "")).strip()
            if not basin:
                violations.append(f"malformed_evolution_row:{idx}")

        for idx, row in enumerate(drift_series):
            from_basin = _safe_text(_field(row, "from_basin", "")).strip()
            to_basin = _safe_text(_field(row, "to_basin", "")).strip()
            if not from_basin or not to_basin:
                violations.append(f"malformed_drift_row:{idx}")

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
    except Exception as exc:
        violations.append(f"validator_internal_error:{_safe_text(exc)}")

    return tuple(sorted(set(violations)))


def build_stability_forecast_receipt(
    *,
    scenario: StabilityForecastScenario,
    metrics: Sequence[StabilityForecastMetric],
    forecast_hash: str,
) -> StabilityForecastReceipt:
    severity_projection_score = 0.0
    for metric in metrics:
        if metric.metric_name == "severity_projection_score":
            severity_projection_score = float(metric.metric_value)
            break

    payload = {
        "scenario_hash": scenario.stable_hash(),
        "metrics_hash": _metrics_hash(metrics),
        "forecast_hash": _safe_text(forecast_hash).strip(),
        "severity_projection_score": severity_projection_score,
    }
    receipt_hash = _sha256_hex(_canonical_json(payload).encode("utf-8"))
    return StabilityForecastReceipt(
        scenario_hash=payload["scenario_hash"],
        metrics_hash=payload["metrics_hash"],
        forecast_hash=payload["forecast_hash"],
        severity_projection_score=severity_projection_score,
        receipt_hash=receipt_hash,
    )


def run_governance_stability_forecast(
    *,
    scenario: StabilityForecastScenario,
) -> GovernanceStabilityForecastKernel:
    metrics = _compute_metrics(scenario)
    provisional = GovernanceStabilityForecastKernel(
        scenario=scenario,
        metrics=metrics,
        violations=(),
        receipt=StabilityForecastReceipt("", "", "", 0.0, ""),
        forecast_hash="",
    )
    violations = validate_stability_forecast(provisional)
    forecast_hash = _forecast_hash(scenario=scenario, metrics=metrics, violations=violations)
    receipt = build_stability_forecast_receipt(
        scenario=scenario,
        metrics=metrics,
        forecast_hash=forecast_hash,
    )
    return GovernanceStabilityForecastKernel(
        scenario=scenario,
        metrics=metrics,
        violations=violations,
        receipt=receipt,
        forecast_hash=forecast_hash,
    )


def compare_stability_forecast_replay(
    baseline: GovernanceStabilityForecastKernel,
    replay: GovernanceStabilityForecastKernel,
) -> Dict[str, Any]:
    metric_delta: Dict[str, float] = {}
    baseline_by_name = {m.metric_name: m.metric_value for m in baseline.metrics}
    replay_by_name = {m.metric_name: m.metric_value for m in replay.metrics}

    for name in _METRIC_ORDER:
        metric_delta[name] = float(replay_by_name.get(name, 0.0) - baseline_by_name.get(name, 0.0))

    mismatches: list[str] = []
    if baseline.scenario.stable_hash() != replay.scenario.stable_hash():
        mismatches.append("scenario_hash")
    if baseline.forecast_hash != replay.forecast_hash:
        mismatches.append("forecast_hash")
    if baseline.receipt.receipt_hash != replay.receipt.receipt_hash:
        mismatches.append("receipt_hash")

    return {
        "is_stable_replay": len(mismatches) == 0,
        "mismatches": tuple(mismatches),
        "metric_delta": tuple((name, metric_delta[name]) for name in _METRIC_ORDER),
        "baseline_hash": baseline.stable_hash(),
        "replay_hash": replay.stable_hash(),
    }


def summarize_stability_forecast(kernel: GovernanceStabilityForecastKernel) -> str:
    lines = [
        f"scenario_id={kernel.scenario.scenario_id}",
        f"scenario_hash={kernel.scenario.stable_hash()}",
        f"forecast_hash={kernel.forecast_hash}",
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
    "StabilityForecastScenario",
    "StabilityForecastMetric",
    "StabilityForecastReceipt",
    "GovernanceStabilityForecastKernel",
    "build_stability_forecast_scenario",
    "run_governance_stability_forecast",
    "validate_stability_forecast",
    "build_stability_forecast_receipt",
    "compare_stability_forecast_replay",
    "summarize_stability_forecast",
]
