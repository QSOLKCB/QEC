"""v137.20.4 — Governance Topology Forecast Stability Kernel.

Narrow additive Layer-4 advisory-only orchestration release.

Canonical flow:
- topology_stability_series + replay_horizon_series
  -> topology_forecast_analysis + topology_forecast_receipt

Invariants:
- no randomness
- no async
- no external I/O
- deterministic ordering
- canonical JSON + stable SHA-256
- validator never raises
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
    "forecast_coherence_score",
    "horizon_projection_gradient",
    "forecast_surface_pressure",
    "topology_forecast_alignment",
    "replay_forecast_stability_score",
    "forecast_confidence_score",
)
_METRIC_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(_METRIC_ORDER)}


_ADVISORY_THRESHOLDS: Tuple[Tuple[str, float], ...] = (
    ("stable_forecast", 0.05),
    ("minor_forecast_variation", 0.20),
    ("moderate_forecast_instability", 0.50),
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
class _TopologyNode:
    topology_id: str
    coherence: float
    pressure: float
    alignment: float
    continuity_ok: bool
    replay_identity: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topology_id": self.topology_id,
            "coherence": self.coherence,
            "pressure": self.pressure,
            "alignment": self.alignment,
            "continuity_ok": self.continuity_ok,
            "replay_identity": self.replay_identity,
        }


@dataclass(frozen=True)
class _HorizonNode:
    horizon_id: str
    horizon_step: float
    forecast_pressure: float
    projection_delta: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizon_id": self.horizon_id,
            "horizon_step": self.horizon_step,
            "forecast_pressure": self.forecast_pressure,
            "projection_delta": self.projection_delta,
        }


@dataclass(frozen=True)
class TopologyForecastScenario:
    scenario_id: str
    topology_stability_series: Tuple[Mapping[str, Any], ...]
    replay_horizon_series: Tuple[Mapping[str, Any], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "topology_stability_series": [dict(item) for item in self.topology_stability_series],
            "replay_horizon_series": [dict(item) for item in self.replay_horizon_series],
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_canonical_json().encode("utf-8"))


@dataclass(frozen=True)
class TopologyForecastMetric:
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
class TopologyForecastReceipt:
    scenario_hash: str
    metrics_hash: str
    forecast_hash: str
    forecast_surface_pressure: float
    advisory_output: str
    receipt_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_hash": self.scenario_hash,
            "metrics_hash": self.metrics_hash,
            "forecast_hash": self.forecast_hash,
            "forecast_surface_pressure": self.forecast_surface_pressure,
            "advisory_output": self.advisory_output,
            "receipt_hash": self.receipt_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.receipt_hash


@dataclass(frozen=True)
class GovernanceTopologyForecastStabilityKernel:
    scenario: TopologyForecastScenario
    metrics: Tuple[TopologyForecastMetric, ...]
    topology_forecast_analysis: Mapping[str, Any]
    advisory_output: str
    violations: Tuple[str, ...]
    receipt: TopologyForecastReceipt
    forecast_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario.to_dict(),
            "metrics": [metric.to_dict() for metric in self.metrics],
            "topology_forecast_analysis": dict(self.topology_forecast_analysis),
            "advisory_output": self.advisory_output,
            "violations": list(self.violations),
            "receipt": self.receipt.to_dict(),
            "forecast_hash": self.forecast_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.forecast_hash


def _normalize_topology_node(raw: Any, index: int) -> _TopologyNode:
    topology_id = _safe_text(_field(raw, "topology_id", "")).strip() or f"topology_{index}"
    coherence = _clamp01(
        _safe_nonneg_float(
            _field(raw, "coherence", _field(raw, "topology_stability_score", _field(raw, "stability", 0.0)))
        )
    )
    pressure = _clamp01(_safe_nonneg_float(_field(raw, "pressure", _field(raw, "surface_pressure", 0.0))))
    alignment = _clamp01(_safe_nonneg_float(_field(raw, "alignment", _field(raw, "topology_alignment", 0.0))))
    continuity_ok = _safe_bool(_field(raw, "continuity_ok", False))
    replay_identity = _safe_text(_field(raw, "replay_identity", "")).strip()
    return _TopologyNode(
        topology_id=topology_id,
        coherence=coherence,
        pressure=pressure,
        alignment=alignment,
        continuity_ok=continuity_ok,
        replay_identity=replay_identity,
    )


def _normalize_horizon_node(raw: Any, index: int) -> _HorizonNode:
    horizon_id = _safe_text(_field(raw, "horizon_id", "")).strip() or f"horizon_{index}"
    horizon_step = _safe_nonneg_float(_field(raw, "horizon_step", _field(raw, "horizon", index)))
    forecast_pressure = _clamp01(
        _safe_nonneg_float(_field(raw, "forecast_pressure", _field(raw, "pressure", _field(raw, "surface_pressure", 0.0))))
    )
    projection_delta = _clamp01(_safe_nonneg_float(_field(raw, "projection_delta", _field(raw, "delta", 0.0))))
    return _HorizonNode(
        horizon_id=horizon_id,
        horizon_step=horizon_step,
        forecast_pressure=forecast_pressure,
        projection_delta=projection_delta,
    )


def _ordered_topology_nodes(series: Sequence[Any]) -> Tuple[_TopologyNode, ...]:
    return tuple(_normalize_topology_node(raw, index=i) for i, raw in enumerate(series))


def _ordered_horizon_nodes(series: Sequence[Any]) -> Tuple[_HorizonNode, ...]:
    return tuple(
        sorted(
            (_normalize_horizon_node(raw, index=i) for i, raw in enumerate(series)),
            key=lambda item: (item.horizon_step, item.horizon_id, _canonical_json(item.to_dict())),
        )
    )


def build_topology_forecast_scenario(
    *,
    scenario_id: str,
    topology_stability_series: Any,
    replay_horizon_series: Any,
) -> TopologyForecastScenario:
    normalized_topology = _ordered_topology_nodes(_safe_series(topology_stability_series))
    normalized_horizons = _ordered_horizon_nodes(_safe_series(replay_horizon_series))
    return TopologyForecastScenario(
        scenario_id=_safe_text(scenario_id).strip(),
        topology_stability_series=tuple(node.to_dict() for node in normalized_topology),
        replay_horizon_series=tuple(node.to_dict() for node in normalized_horizons),
    )


def _metric(name: str, value: float) -> TopologyForecastMetric:
    return TopologyForecastMetric(
        metric_name=name,
        metric_order=_METRIC_INDEX[name],
        metric_value=float(value),
    )


def _compute_metrics(scenario: TopologyForecastScenario) -> Tuple[TopologyForecastMetric, ...]:
    topology_nodes = _ordered_topology_nodes(scenario.topology_stability_series)
    horizon_nodes = _ordered_horizon_nodes(scenario.replay_horizon_series)

    topology_count = len(topology_nodes)
    horizon_count = len(horizon_nodes)

    avg_coherence = 0.0 if topology_count == 0 else sum(node.coherence for node in topology_nodes) / float(topology_count)
    avg_alignment = 0.0 if topology_count == 0 else sum(node.alignment for node in topology_nodes) / float(topology_count)
    continuity_ratio = (
        0.0
        if topology_count == 0
        else sum(1.0 for node in topology_nodes if node.continuity_ok) / float(topology_count)
    )

    horizon_pressure = 0.0
    horizon_gradient = 0.0
    horizon_delta = 0.0
    if horizon_count > 0:
        horizon_pressure = sum(node.forecast_pressure for node in horizon_nodes) / float(horizon_count)
        horizon_delta = sum(node.projection_delta for node in horizon_nodes) / float(horizon_count)
        if horizon_count > 1:
            gradient_sum = 0.0
            for idx in range(1, horizon_count):
                left = horizon_nodes[idx - 1]
                right = horizon_nodes[idx]
                step_delta = right.horizon_step - left.horizon_step
                if step_delta <= 0.0:
                    continue
                gradient_sum += abs(right.forecast_pressure - left.forecast_pressure) / step_delta
            horizon_gradient = gradient_sum / float(horizon_count - 1)

    replay_pairs = 0
    replay_stable = 0
    for idx in range(1, topology_count):
        replay_pairs += 1
        left = topology_nodes[idx - 1]
        right = topology_nodes[idx]
        if left.replay_identity and left.replay_identity == right.replay_identity:
            replay_stable += 1

    replay_forecast_stability_score = 1.0 if replay_pairs == 0 else float(replay_stable) / float(replay_pairs)

    forecast_coherence_score = _clamp01(0.7 * _clamp01(avg_coherence) + 0.3 * _clamp01(continuity_ratio))
    horizon_projection_gradient = _clamp01(horizon_gradient)

    forecast_surface_pressure = _clamp01(
        0.45 * (1.0 - forecast_coherence_score)
        + 0.25 * _clamp01(horizon_projection_gradient)
        + 0.20 * (1.0 - _clamp01(avg_alignment))
        + 0.10 * _clamp01(horizon_pressure)
    )

    topology_forecast_alignment = _clamp01(0.8 * _clamp01(avg_alignment) + 0.2 * (1.0 - _clamp01(horizon_delta)))

    forecast_confidence_score = _clamp01(
        0.45 * _clamp01(forecast_coherence_score)
        + 0.35 * _clamp01(replay_forecast_stability_score)
        + 0.20 * (1.0 - _clamp01(forecast_surface_pressure))
    )

    return (
        _metric("forecast_coherence_score", forecast_coherence_score),
        _metric("horizon_projection_gradient", horizon_projection_gradient),
        _metric("forecast_surface_pressure", forecast_surface_pressure),
        _metric("topology_forecast_alignment", topology_forecast_alignment),
        _metric("replay_forecast_stability_score", _clamp01(replay_forecast_stability_score)),
        _metric("forecast_confidence_score", forecast_confidence_score),
    )


def _metrics_hash(metrics: Sequence[TopologyForecastMetric]) -> str:
    return _sha256_hex(_canonical_json([metric.to_dict() for metric in metrics]).encode("utf-8"))


def _classify_advisory(forecast_surface_pressure: float) -> str:
    pressure = _clamp01(forecast_surface_pressure)
    for label, threshold in _ADVISORY_THRESHOLDS:
        if pressure <= threshold:
            return label
    return "severe_forecast_instability"


def _forecast_analysis(*, metrics: Sequence[TopologyForecastMetric], advisory_output: str) -> Dict[str, Any]:
    by_name = {metric.metric_name: metric.metric_value for metric in metrics}
    return {
        "composite_pressure": _clamp01(float(by_name.get("forecast_surface_pressure", 0.0))),
        "advisory_output": advisory_output,
        "metric_order": _METRIC_ORDER,
        "replay_horizon_ordering": "ascending_horizon_step_then_id",
    }


def _forecast_hash(
    *,
    scenario: TopologyForecastScenario,
    metrics: Sequence[TopologyForecastMetric],
    topology_forecast_analysis: Mapping[str, Any],
    violations: Sequence[str],
) -> str:
    payload = {
        "scenario": scenario.to_dict(),
        "metrics": [metric.to_dict() for metric in metrics],
        "topology_forecast_analysis": dict(topology_forecast_analysis),
        "violations": list(violations),
    }
    return _sha256_hex(_canonical_json(payload).encode("utf-8"))


def validate_topology_forecast_stability(kernel: Any) -> Tuple[str, ...]:
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

        topology_series = _safe_series(_field(scenario, "topology_stability_series", ()))
        horizon_series = _safe_series(_field(scenario, "replay_horizon_series", ()))

        if not topology_series:
            violations.append("empty_topology_stability_series")
        if not horizon_series:
            violations.append("empty_replay_horizon_series")

        for idx, row in enumerate(topology_series):
            topology_id = _safe_text(_field(row, "topology_id", "")).strip()
            if not topology_id:
                violations.append(f"malformed_topology_row:{idx}")

        last_horizon = -1.0
        for idx, row in enumerate(horizon_series):
            horizon_id = _safe_text(_field(row, "horizon_id", "")).strip()
            horizon_step = _safe_nonneg_float(_field(row, "horizon_step", -1.0))
            if not horizon_id:
                violations.append(f"malformed_horizon_row:{idx}")
            if horizon_step < last_horizon:
                violations.append("horizon_ordering_violation")
                break
            last_horizon = horizon_step

        names = tuple(_safe_text(_field(metric, "metric_name", "")) for metric in metrics)
        if names != _METRIC_ORDER:
            violations.append("metric_order_mismatch")

        metric_map: Dict[str, float] = {}
        for metric in metrics:
            name = _safe_text(_field(metric, "metric_name", "")).strip()
            value_raw = _field(metric, "metric_value", 0.0)
            try:
                value = float(value_raw)
            except Exception:
                violations.append(f"metric_non_numeric:{name or '?'}")
                continue
            if not math.isfinite(value):
                violations.append(f"metric_non_finite:{name or '?'}")
                continue
            metric_map[name] = value

        for name in _METRIC_ORDER:
            if name in metric_map and (metric_map[name] < 0.0 or metric_map[name] > 1.0):
                violations.append(f"metric_out_of_bounds:{name}")

        if receipt is not None:
            metrics_hash = _safe_text(_field(receipt, "metrics_hash", "")).strip()
            if metrics_hash:
                if metrics_hash != _metrics_hash(metrics):
                    violations.append("receipt_metrics_hash_mismatch")
    except Exception as exc:
        violations.append(f"validator_internal_error:{_safe_text(exc)}")

    return tuple(sorted(set(violations)))


def build_topology_forecast_receipt(
    *,
    scenario: TopologyForecastScenario,
    metrics: Sequence[TopologyForecastMetric],
    forecast_hash: str,
) -> TopologyForecastReceipt:
    metric_map = {metric.metric_name: metric.metric_value for metric in metrics}
    forecast_surface_pressure = _clamp01(float(metric_map.get("forecast_surface_pressure", 0.0)))
    advisory_output = _classify_advisory(forecast_surface_pressure)
    payload = {
        "scenario_hash": scenario.stable_hash(),
        "metrics_hash": _metrics_hash(metrics),
        "forecast_hash": _safe_text(forecast_hash).strip(),
        "forecast_surface_pressure": forecast_surface_pressure,
        "advisory_output": advisory_output,
    }
    receipt_hash = _sha256_hex(_canonical_json(payload).encode("utf-8"))
    return TopologyForecastReceipt(
        scenario_hash=payload["scenario_hash"],
        metrics_hash=payload["metrics_hash"],
        forecast_hash=payload["forecast_hash"],
        forecast_surface_pressure=forecast_surface_pressure,
        advisory_output=advisory_output,
        receipt_hash=receipt_hash,
    )


def run_governance_topology_forecast_stability(
    *,
    scenario: TopologyForecastScenario,
) -> GovernanceTopologyForecastStabilityKernel:
    metrics = _compute_metrics(scenario)
    metric_map = {metric.metric_name: metric.metric_value for metric in metrics}
    advisory_output = _classify_advisory(float(metric_map.get("forecast_surface_pressure", 0.0)))
    topology_forecast_analysis = _forecast_analysis(metrics=metrics, advisory_output=advisory_output)

    provisional = GovernanceTopologyForecastStabilityKernel(
        scenario=scenario,
        metrics=metrics,
        topology_forecast_analysis=topology_forecast_analysis,
        advisory_output=advisory_output,
        violations=(),
        receipt=TopologyForecastReceipt("", "", "", 0.0, "stable_forecast", ""),
        forecast_hash="",
    )
    violations = validate_topology_forecast_stability(provisional)
    forecast_hash = _forecast_hash(
        scenario=scenario,
        metrics=metrics,
        topology_forecast_analysis=topology_forecast_analysis,
        violations=violations,
    )
    receipt = build_topology_forecast_receipt(
        scenario=scenario,
        metrics=metrics,
        forecast_hash=forecast_hash,
    )

    return GovernanceTopologyForecastStabilityKernel(
        scenario=scenario,
        metrics=metrics,
        topology_forecast_analysis=topology_forecast_analysis,
        advisory_output=advisory_output,
        violations=violations,
        receipt=receipt,
        forecast_hash=forecast_hash,
    )


def compare_topology_forecast_replay(
    baseline: GovernanceTopologyForecastStabilityKernel,
    replay: GovernanceTopologyForecastStabilityKernel,
) -> Dict[str, Any]:
    baseline_by_name = {metric.metric_name: metric.metric_value for metric in baseline.metrics}
    replay_by_name = {metric.metric_name: metric.metric_value for metric in replay.metrics}

    metric_delta = tuple(
        (name, float(replay_by_name.get(name, 0.0) - baseline_by_name.get(name, 0.0))) for name in _METRIC_ORDER
    )

    mismatches: list[str] = []
    if baseline.scenario.stable_hash() != replay.scenario.stable_hash():
        mismatches.append("scenario_hash")
    if baseline.forecast_hash != replay.forecast_hash:
        mismatches.append("forecast_hash")
    if baseline.receipt.receipt_hash != replay.receipt.receipt_hash:
        mismatches.append("receipt_hash")
    if baseline.advisory_output != replay.advisory_output:
        mismatches.append("advisory_output")

    return {
        "is_stable_replay": len(mismatches) == 0,
        "mismatches": tuple(mismatches),
        "metric_delta": metric_delta,
        "baseline_hash": baseline.stable_hash(),
        "replay_hash": replay.stable_hash(),
    }


def summarize_topology_forecast_stability(kernel: GovernanceTopologyForecastStabilityKernel) -> str:
    lines = [
        f"scenario_id={kernel.scenario.scenario_id}",
        f"scenario_hash={kernel.scenario.stable_hash()}",
        f"forecast_hash={kernel.forecast_hash}",
        f"receipt_hash={kernel.receipt.receipt_hash}",
        f"advisory_output={kernel.advisory_output}",
        "metrics:",
    ]
    for metric in kernel.metrics:
        lines.append(f"- {metric.metric_order}:{metric.metric_name}={metric.metric_value:.12f}")
    lines.append("violations:")
    if kernel.violations:
        for violation in kernel.violations:
            lines.append(f"- {violation}")
    else:
        lines.append("- none")
    return "\n".join(lines)


__all__ = [
    "TopologyForecastScenario",
    "TopologyForecastMetric",
    "TopologyForecastReceipt",
    "GovernanceTopologyForecastStabilityKernel",
    "build_topology_forecast_scenario",
    "run_governance_topology_forecast_stability",
    "validate_topology_forecast_stability",
    "build_topology_forecast_receipt",
    "compare_topology_forecast_replay",
    "summarize_topology_forecast_stability",
]
