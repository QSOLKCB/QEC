"""v137.20.0 — Governance Forecast Feedback Control Kernel.

Additive analysis-and-control layer that converts deterministic reconciliation
residuals and realized drift signals into bounded, advisory feedback-control
recommendations.

Canonical model:

    reconciliation_series + drift_series
        -> control_analysis + control_receipt

This kernel is strictly advisory:

- no mutation of decoder state
- no mutation of inputs
- no autonomous execution
- no policy enforcement
- no write actions

Invariants:

- no randomness
- no async
- no external I/O
- deterministic ordering
- canonical JSON + SHA-256
- validator must never raise
- malformed input returns deterministic violations
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Sequence, Tuple


_METRIC_ORDER: Tuple[str, ...] = (
    "residual_control_pressure",
    "stability_correction_score",
    "continuity_recovery_priority",
    "severity_feedback_gain",
    "replay_control_stability",
    "calibration_feedback_signal",
    "bounded_control_confidence",
)
_METRIC_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(_METRIC_ORDER)}


_RECOMMENDATION_ORDER: Tuple[str, ...] = (
    "observe_only",
    "monitor",
    "stabilize",
    "recalibrate",
    "isolate",
)
_RECOMMENDATION_RANK: Dict[str, int] = {
    name: idx for idx, name in enumerate(_RECOMMENDATION_ORDER)
}


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
class _ReconciliationNode:
    reconciliation_id: str
    residual_magnitude: float
    stability_residual: float
    continuity_residual: float
    severity_residual: float
    replay_residual: float
    calibration_residual: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reconciliation_id": self.reconciliation_id,
            "residual_magnitude": self.residual_magnitude,
            "stability_residual": self.stability_residual,
            "continuity_residual": self.continuity_residual,
            "severity_residual": self.severity_residual,
            "replay_residual": self.replay_residual,
            "calibration_residual": self.calibration_residual,
        }

    def max_category_residual(self) -> float:
        return max(
            self.stability_residual,
            self.continuity_residual,
            self.severity_residual,
            self.replay_residual,
            self.calibration_residual,
        )


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
class ForecastFeedbackControlScenario:
    scenario_id: str
    reconciliation_series: Tuple[Mapping[str, Any], ...]
    drift_series: Tuple[Mapping[str, Any], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "reconciliation_series": [dict(item) for item in self.reconciliation_series],
            "drift_series": [dict(item) for item in self.drift_series],
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_canonical_json().encode("utf-8"))


@dataclass(frozen=True)
class ForecastFeedbackControlMetric:
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
class ForecastFeedbackControlReceipt:
    scenario_hash: str
    metrics_hash: str
    recommendations_hash: str
    control_hash: str
    bounded_control_confidence: float
    receipt_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_hash": self.scenario_hash,
            "metrics_hash": self.metrics_hash,
            "recommendations_hash": self.recommendations_hash,
            "control_hash": self.control_hash,
            "bounded_control_confidence": self.bounded_control_confidence,
            "receipt_hash": self.receipt_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.receipt_hash


@dataclass(frozen=True)
class GovernanceForecastFeedbackControlKernel:
    scenario: ForecastFeedbackControlScenario
    metrics: Tuple[ForecastFeedbackControlMetric, ...]
    recommendations: Tuple[Tuple[str, str], ...]
    aggregate_recommendation: str
    violations: Tuple[str, ...]
    receipt: ForecastFeedbackControlReceipt
    control_hash: str
    advisory_only: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario.to_dict(),
            "metrics": [metric.to_dict() for metric in self.metrics],
            "recommendations": [[rid, rec] for rid, rec in self.recommendations],
            "aggregate_recommendation": self.aggregate_recommendation,
            "violations": list(self.violations),
            "receipt": self.receipt.to_dict(),
            "control_hash": self.control_hash,
            "advisory_only": bool(self.advisory_only),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.control_hash


def _normalize_reconciliation_node(raw: Any, index: int) -> _ReconciliationNode:
    reconciliation_id = (
        _safe_text(_field(raw, "reconciliation_id", "")).strip()
        or _safe_text(_field(raw, "evolution_id", "")).strip()
        or f"reconciliation_{index}"
    )
    residual_magnitude = _clamp01(
        _safe_nonneg_float(_field(raw, "residual_magnitude", 0.0))
    )
    stability_residual = _clamp01(
        _safe_nonneg_float(_field(raw, "stability_residual", 0.0))
    )
    continuity_residual = _clamp01(
        _safe_nonneg_float(_field(raw, "continuity_residual", 0.0))
    )
    severity_residual = _clamp01(
        _safe_nonneg_float(_field(raw, "severity_residual", 0.0))
    )
    replay_residual = _clamp01(
        _safe_nonneg_float(_field(raw, "replay_residual", 0.0))
    )
    calibration_residual = _clamp01(
        _safe_nonneg_float(_field(raw, "calibration_residual", 0.0))
    )
    return _ReconciliationNode(
        reconciliation_id=reconciliation_id,
        residual_magnitude=residual_magnitude,
        stability_residual=stability_residual,
        continuity_residual=continuity_residual,
        severity_residual=severity_residual,
        replay_residual=replay_residual,
        calibration_residual=calibration_residual,
    )


def _normalize_drift_node(raw: Any, index: int) -> _DriftNode:
    drift_id = _safe_text(_field(raw, "drift_id", "")).strip() or f"drift_{index}"
    from_basin = _safe_text(_field(raw, "from_basin", "")).strip() or "unknown"
    to_basin = _safe_text(_field(raw, "to_basin", "")).strip() or "unknown"
    transition_count = _safe_nonneg_int(_field(raw, "transition_count", 0))
    drift_magnitude = _clamp01(_safe_nonneg_float(_field(raw, "drift_magnitude", 0.0)))
    return _DriftNode(
        drift_id=drift_id,
        from_basin=from_basin,
        to_basin=to_basin,
        transition_count=transition_count,
        drift_magnitude=drift_magnitude,
    )


def _ordered_reconciliation_nodes(series: Sequence[Any]) -> Tuple[_ReconciliationNode, ...]:
    return tuple(
        _normalize_reconciliation_node(raw, index=i) for i, raw in enumerate(series)
    )


def _ordered_drift_nodes(series: Sequence[Any]) -> Tuple[_DriftNode, ...]:
    return tuple(_normalize_drift_node(raw, index=i) for i, raw in enumerate(series))


def build_feedback_control_scenario(
    *,
    scenario_id: str,
    reconciliation_series: Any,
    drift_series: Any,
) -> ForecastFeedbackControlScenario:
    normalized_id = _safe_text(scenario_id).strip()
    normalized_recs = _ordered_reconciliation_nodes(_safe_series(reconciliation_series))
    normalized_drifts = _ordered_drift_nodes(_safe_series(drift_series))
    return ForecastFeedbackControlScenario(
        scenario_id=normalized_id,
        reconciliation_series=tuple(node.to_dict() for node in normalized_recs),
        drift_series=tuple(node.to_dict() for node in normalized_drifts),
    )


def _metric(name: str, value: float) -> ForecastFeedbackControlMetric:
    return ForecastFeedbackControlMetric(
        metric_name=name,
        metric_order=_METRIC_INDEX[name],
        metric_value=float(value),
    )


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def _compute_metrics(
    scenario: ForecastFeedbackControlScenario,
) -> Tuple[ForecastFeedbackControlMetric, ...]:
    recs = _ordered_reconciliation_nodes(scenario.reconciliation_series)
    drifts = _ordered_drift_nodes(scenario.drift_series)

    residual_magnitudes = tuple(node.residual_magnitude for node in recs)
    stability_residuals = tuple(node.stability_residual for node in recs)
    continuity_residuals = tuple(node.continuity_residual for node in recs)
    severity_residuals = tuple(node.severity_residual for node in recs)
    replay_residuals = tuple(node.replay_residual for node in recs)
    calibration_residuals = tuple(node.calibration_residual for node in recs)

    drift_magnitudes = tuple(node.drift_magnitude for node in drifts)

    avg_residual = _mean(residual_magnitudes)
    avg_drift = _mean(drift_magnitudes)

    # Pressure blends residual magnitude and realized drift.
    residual_control_pressure = _clamp01(
        (avg_residual * 0.7) + (avg_drift * 0.3)
    )

    stability_correction_score = _clamp01(_mean(stability_residuals))
    continuity_recovery_priority = _clamp01(_mean(continuity_residuals))
    severity_feedback_gain = _clamp01(_mean(severity_residuals))
    replay_control_stability = _clamp01(1.0 - _mean(replay_residuals))
    calibration_feedback_signal = _clamp01(_mean(calibration_residuals))

    if recs:
        max_per_entry = tuple(node.max_category_residual() for node in recs)
        worst_case = _mean(max_per_entry)
    else:
        worst_case = 0.0
    bounded_control_confidence = _clamp01(
        1.0 - (0.5 * residual_control_pressure + 0.5 * worst_case)
    )

    return (
        _metric("residual_control_pressure", residual_control_pressure),
        _metric("stability_correction_score", stability_correction_score),
        _metric("continuity_recovery_priority", continuity_recovery_priority),
        _metric("severity_feedback_gain", severity_feedback_gain),
        _metric("replay_control_stability", replay_control_stability),
        _metric("calibration_feedback_signal", calibration_feedback_signal),
        _metric("bounded_control_confidence", bounded_control_confidence),
    )


def _entry_recommendation(node: _ReconciliationNode) -> str:
    """Deterministic advisory mapping for a single reconciliation entry."""

    worst_category = node.max_category_residual()
    pressure = max(node.residual_magnitude, worst_category)

    if pressure <= 0.0:
        return "observe_only"
    if pressure < 0.2:
        return "monitor"

    # isolate if severity dominates and is extreme
    if node.severity_residual >= 0.8 and node.severity_residual >= worst_category:
        return "isolate"
    if node.continuity_residual >= 0.8 and node.continuity_residual >= worst_category:
        return "isolate"

    if pressure >= 0.8:
        if node.calibration_residual >= node.stability_residual:
            return "recalibrate"
        return "stabilize"

    if pressure >= 0.6:
        if node.calibration_residual >= node.stability_residual:
            return "recalibrate"
        return "stabilize"

    if pressure >= 0.4:
        return "stabilize"

    return "monitor"


def _compute_recommendations(
    scenario: ForecastFeedbackControlScenario,
) -> Tuple[Tuple[str, str], ...]:
    recs = _ordered_reconciliation_nodes(scenario.reconciliation_series)
    return tuple((node.reconciliation_id, _entry_recommendation(node)) for node in recs)


def _aggregate_recommendation(
    metrics: Sequence[ForecastFeedbackControlMetric],
    recommendations: Sequence[Tuple[str, str]],
) -> str:
    by_name = {m.metric_name: m.metric_value for m in metrics}
    pressure = float(by_name.get("residual_control_pressure", 0.0))
    severity = float(by_name.get("severity_feedback_gain", 0.0))
    continuity = float(by_name.get("continuity_recovery_priority", 0.0))
    calibration = float(by_name.get("calibration_feedback_signal", 0.0))
    stability = float(by_name.get("stability_correction_score", 0.0))

    if not recommendations:
        return "observe_only"

    worst_rank = 0
    for _rid, rec in recommendations:
        rank = _RECOMMENDATION_RANK.get(rec, 0)
        if rank > worst_rank:
            worst_rank = rank

    if pressure <= 0.0 and severity <= 0.0 and continuity <= 0.0:
        metric_based = "observe_only"
    elif pressure < 0.2:
        metric_based = "monitor"
    elif severity >= 0.8 or continuity >= 0.8:
        metric_based = "isolate"
    elif pressure >= 0.6 and calibration >= stability:
        metric_based = "recalibrate"
    elif pressure >= 0.4:
        metric_based = "stabilize"
    else:
        metric_based = "monitor"

    metric_rank = _RECOMMENDATION_RANK[metric_based]
    final_rank = max(worst_rank, metric_rank)
    return _RECOMMENDATION_ORDER[final_rank]


def _metrics_hash(metrics: Sequence[ForecastFeedbackControlMetric]) -> str:
    payload = [metric.to_dict() for metric in metrics]
    return _sha256_hex(_canonical_json(payload).encode("utf-8"))


def _recommendations_hash(
    recommendations: Sequence[Tuple[str, str]],
    aggregate_recommendation: str,
) -> str:
    payload = {
        "recommendations": [[rid, rec] for rid, rec in recommendations],
        "aggregate_recommendation": aggregate_recommendation,
    }
    return _sha256_hex(_canonical_json(payload).encode("utf-8"))


def _control_hash(
    *,
    scenario: ForecastFeedbackControlScenario,
    metrics: Sequence[ForecastFeedbackControlMetric],
    recommendations: Sequence[Tuple[str, str]],
    aggregate_recommendation: str,
    violations: Sequence[str],
) -> str:
    body = {
        "scenario": scenario.to_dict(),
        "metrics": [metric.to_dict() for metric in metrics],
        "recommendations": [[rid, rec] for rid, rec in recommendations],
        "aggregate_recommendation": aggregate_recommendation,
        "violations": list(violations),
    }
    return _sha256_hex(_canonical_json(body).encode("utf-8"))


def validate_feedback_control(kernel: Any) -> Tuple[str, ...]:
    violations: list[str] = []
    try:
        scenario = _field(kernel, "scenario", None)
        metrics = tuple(_safe_series(_field(kernel, "metrics", ())))
        recommendations = tuple(_safe_series(_field(kernel, "recommendations", ())))
        aggregate_recommendation = _safe_text(
            _field(kernel, "aggregate_recommendation", "")
        ).strip()
        receipt = _field(kernel, "receipt", None)
        advisory_only = _field(kernel, "advisory_only", True)

        if scenario is None:
            violations.append("missing_scenario")
            return tuple(sorted(set(violations)))

        if not _safe_text(_field(scenario, "scenario_id", "")).strip():
            violations.append("empty_scenario_id")

        reconciliation_series = _safe_series(
            _field(scenario, "reconciliation_series", ())
        )
        drift_series = _safe_series(_field(scenario, "drift_series", ()))

        if not reconciliation_series:
            violations.append("empty_reconciliation_series")

        for idx, row in enumerate(reconciliation_series):
            rec_id = _safe_text(_field(row, "reconciliation_id", "")).strip()
            if not rec_id:
                violations.append(f"malformed_reconciliation_row:{idx}")

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

        for idx, entry in enumerate(recommendations):
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                violations.append(f"malformed_recommendation:{idx}")
                continue
            rec_name = _safe_text(entry[1]).strip()
            if rec_name not in _RECOMMENDATION_RANK:
                violations.append(f"unknown_recommendation:{idx}")

        if aggregate_recommendation and aggregate_recommendation not in _RECOMMENDATION_RANK:
            violations.append("unknown_aggregate_recommendation")

        if advisory_only is not True:
            violations.append("advisory_only_flag_disabled")

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

            actual_recommendations_hash = _safe_text(
                _field(receipt, "recommendations_hash", "")
            ).strip()
            if actual_recommendations_hash:
                expected_recommendations_hash = _recommendations_hash(
                    recommendations, aggregate_recommendation
                )
                if expected_recommendations_hash != actual_recommendations_hash:
                    violations.append("receipt_recommendations_hash_mismatch")

            actual_control_hash = _safe_text(_field(receipt, "control_hash", "")).strip()
            expected_control_hash = _safe_text(_field(kernel, "control_hash", "")).strip()
            if actual_control_hash and expected_control_hash:
                if actual_control_hash != expected_control_hash:
                    violations.append("receipt_control_hash_mismatch")

            actual_receipt_hash = _safe_text(_field(receipt, "receipt_hash", "")).strip()
            if actual_receipt_hash:
                bounded_raw = _field(receipt, "bounded_control_confidence", 0.0)
                try:
                    bounded_value = float(bounded_raw)
                except Exception:
                    bounded_value = 0.0
                expected_payload = {
                    "scenario_hash": _safe_text(_field(receipt, "scenario_hash", "")).strip(),
                    "metrics_hash": _safe_text(_field(receipt, "metrics_hash", "")).strip(),
                    "recommendations_hash": _safe_text(
                        _field(receipt, "recommendations_hash", "")
                    ).strip(),
                    "control_hash": _safe_text(_field(receipt, "control_hash", "")).strip(),
                    "bounded_control_confidence": bounded_value,
                }
                expected_receipt_hash = _sha256_hex(
                    _canonical_json(expected_payload).encode("utf-8")
                )
                if expected_receipt_hash != actual_receipt_hash:
                    violations.append("receipt_hash_mismatch")
    except Exception as exc:
        violations.append(f"validator_internal_error:{_safe_text(exc)}")

    return tuple(sorted(set(violations)))


def build_feedback_control_receipt(
    *,
    scenario: ForecastFeedbackControlScenario,
    metrics: Sequence[ForecastFeedbackControlMetric],
    recommendations: Sequence[Tuple[str, str]],
    aggregate_recommendation: str,
    control_hash: str,
) -> ForecastFeedbackControlReceipt:
    bounded_control_confidence = 0.0
    for metric in metrics:
        if metric.metric_name == "bounded_control_confidence":
            bounded_control_confidence = float(metric.metric_value)
            break

    payload = {
        "scenario_hash": scenario.stable_hash(),
        "metrics_hash": _metrics_hash(metrics),
        "recommendations_hash": _recommendations_hash(
            recommendations, aggregate_recommendation
        ),
        "control_hash": _safe_text(control_hash).strip(),
        "bounded_control_confidence": bounded_control_confidence,
    }
    receipt_hash = _sha256_hex(_canonical_json(payload).encode("utf-8"))
    return ForecastFeedbackControlReceipt(
        scenario_hash=payload["scenario_hash"],
        metrics_hash=payload["metrics_hash"],
        recommendations_hash=payload["recommendations_hash"],
        control_hash=payload["control_hash"],
        bounded_control_confidence=bounded_control_confidence,
        receipt_hash=receipt_hash,
    )


def run_governance_feedback_control(
    *,
    scenario: ForecastFeedbackControlScenario,
) -> GovernanceForecastFeedbackControlKernel:
    metrics = _compute_metrics(scenario)
    recommendations = _compute_recommendations(scenario)
    aggregate_recommendation = _aggregate_recommendation(metrics, recommendations)
    provisional = GovernanceForecastFeedbackControlKernel(
        scenario=scenario,
        metrics=metrics,
        recommendations=recommendations,
        aggregate_recommendation=aggregate_recommendation,
        violations=(),
        receipt=ForecastFeedbackControlReceipt("", "", "", "", 0.0, ""),
        control_hash="",
        advisory_only=True,
    )
    violations = validate_feedback_control(provisional)
    control_hash = _control_hash(
        scenario=scenario,
        metrics=metrics,
        recommendations=recommendations,
        aggregate_recommendation=aggregate_recommendation,
        violations=violations,
    )
    receipt = build_feedback_control_receipt(
        scenario=scenario,
        metrics=metrics,
        recommendations=recommendations,
        aggregate_recommendation=aggregate_recommendation,
        control_hash=control_hash,
    )
    return GovernanceForecastFeedbackControlKernel(
        scenario=scenario,
        metrics=metrics,
        recommendations=recommendations,
        aggregate_recommendation=aggregate_recommendation,
        violations=violations,
        receipt=receipt,
        control_hash=control_hash,
        advisory_only=True,
    )


def compare_feedback_control_replay(
    baseline: GovernanceForecastFeedbackControlKernel,
    replay: GovernanceForecastFeedbackControlKernel,
) -> Dict[str, Any]:
    metric_delta: Dict[str, float] = {}
    baseline_by_name = {m.metric_name: m.metric_value for m in baseline.metrics}
    replay_by_name = {m.metric_name: m.metric_value for m in replay.metrics}

    for name in _METRIC_ORDER:
        metric_delta[name] = float(
            replay_by_name.get(name, 0.0) - baseline_by_name.get(name, 0.0)
        )

    mismatches: list[str] = []
    if baseline.scenario.stable_hash() != replay.scenario.stable_hash():
        mismatches.append("scenario_hash")
    if baseline.control_hash != replay.control_hash:
        mismatches.append("control_hash")
    if baseline.receipt.receipt_hash != replay.receipt.receipt_hash:
        mismatches.append("receipt_hash")
    if baseline.aggregate_recommendation != replay.aggregate_recommendation:
        mismatches.append("aggregate_recommendation")

    return {
        "is_stable_replay": len(mismatches) == 0,
        "mismatches": tuple(mismatches),
        "metric_delta": tuple((name, metric_delta[name]) for name in _METRIC_ORDER),
        "baseline_hash": baseline.stable_hash(),
        "replay_hash": replay.stable_hash(),
        "baseline_aggregate_recommendation": baseline.aggregate_recommendation,
        "replay_aggregate_recommendation": replay.aggregate_recommendation,
    }


def summarize_feedback_control(
    kernel: GovernanceForecastFeedbackControlKernel,
) -> str:
    lines = [
        f"scenario_id={kernel.scenario.scenario_id}",
        f"scenario_hash={kernel.scenario.stable_hash()}",
        f"control_hash={kernel.control_hash}",
        f"receipt_hash={kernel.receipt.receipt_hash}",
        f"aggregate_recommendation={kernel.aggregate_recommendation}",
        f"advisory_only={bool(kernel.advisory_only)}",
        "metrics:",
    ]
    for metric in kernel.metrics:
        lines.append(
            f"- {metric.metric_order}:{metric.metric_name}={metric.metric_value:.12f}"
        )
    if kernel.recommendations:
        lines.append("recommendations:")
        for rid, rec in kernel.recommendations:
            lines.append(f"- {rid}={rec}")
    else:
        lines.append("recommendations: none")
    if kernel.violations:
        lines.append("violations:")
        for item in kernel.violations:
            lines.append(f"- {item}")
    else:
        lines.append("violations: none")
    return "\n".join(lines)


__all__ = [
    "ForecastFeedbackControlScenario",
    "ForecastFeedbackControlMetric",
    "ForecastFeedbackControlReceipt",
    "GovernanceForecastFeedbackControlKernel",
    "build_feedback_control_scenario",
    "run_governance_feedback_control",
    "validate_feedback_control",
    "build_feedback_control_receipt",
    "compare_feedback_control_replay",
    "summarize_feedback_control",
]
