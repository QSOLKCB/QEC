"""v137.20.0 — Governance Forecast Feedback Control Kernel.

Additive deterministic advisory layer that converts reconciliation residuals
and drift observations into bounded control recommendations.

Canonical model:
- reconciliation_series + drift_series
  -> control_analysis + advisory_receipt

Invariants:
- no randomness
- no async
- no external I/O
- no mutation of inputs
- advisory only — no autonomous execution
- deterministic ordering
- canonical JSON + SHA-256
- validator must never raise
- bounded advisory outputs drawn from a fixed lattice
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
    "bounded_control_confidence",
)
_METRIC_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(_METRIC_ORDER)}


_ADVISORY_OUTPUTS: Tuple[str, ...] = (
    "observe_only",
    "monitor",
    "stabilize",
    "recalibrate",
    "isolate",
)
_ADVISORY_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(_ADVISORY_OUTPUTS)}


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
    stability_accuracy: float
    continuity_error: float
    severity_residual: float
    replay_score: float
    confidence_calibration: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reconciliation_id": self.reconciliation_id,
            "residual_magnitude": self.residual_magnitude,
            "stability_accuracy": self.stability_accuracy,
            "continuity_error": self.continuity_error,
            "severity_residual": self.severity_residual,
            "replay_score": self.replay_score,
            "confidence_calibration": self.confidence_calibration,
        }


@dataclass(frozen=True)
class _DriftNode:
    drift_id: str
    from_basin: str
    to_basin: str
    transition_count: int
    drift_magnitude: float
    drift_severity: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drift_id": self.drift_id,
            "from_basin": self.from_basin,
            "to_basin": self.to_basin,
            "transition_count": self.transition_count,
            "drift_magnitude": self.drift_magnitude,
            "drift_severity": self.drift_severity,
        }


@dataclass(frozen=True)
class FeedbackControlScenario:
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
class FeedbackControlMetric:
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
class FeedbackControlReceipt:
    scenario_hash: str
    metrics_hash: str
    advisory_hash: str
    analysis_hash: str
    bounded_control_confidence: float
    advisory_output: str
    receipt_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_hash": self.scenario_hash,
            "metrics_hash": self.metrics_hash,
            "advisory_hash": self.advisory_hash,
            "analysis_hash": self.analysis_hash,
            "bounded_control_confidence": self.bounded_control_confidence,
            "advisory_output": self.advisory_output,
            "receipt_hash": self.receipt_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.receipt_hash


@dataclass(frozen=True)
class GovernanceForecastFeedbackControlKernel:
    scenario: FeedbackControlScenario
    metrics: Tuple[FeedbackControlMetric, ...]
    advisory_output: str
    advisory_rationale: Tuple[str, ...]
    violations: Tuple[str, ...]
    receipt: FeedbackControlReceipt
    analysis_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario.to_dict(),
            "metrics": [metric.to_dict() for metric in self.metrics],
            "advisory_output": self.advisory_output,
            "advisory_rationale": list(self.advisory_rationale),
            "violations": list(self.violations),
            "receipt": self.receipt.to_dict(),
            "analysis_hash": self.analysis_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.analysis_hash


def _normalize_reconciliation_node(raw: Any, index: int) -> _ReconciliationNode:
    reconciliation_id = (
        _safe_text(_field(raw, "reconciliation_id", "")).strip()
        or _safe_text(_field(raw, "scenario_id", "")).strip()
        or f"reconciliation_{index}"
    )
    residual_magnitude = _clamp01(
        _safe_nonneg_float(
            _field(raw, "residual_magnitude", _field(raw, "forecast_error_rate", 0.0))
        )
    )
    stability_accuracy = _clamp01(
        _safe_nonneg_float(
            _field(raw, "stability_accuracy", _field(raw, "stability_prediction_accuracy", 1.0))
        )
    )
    continuity_error = _clamp01(
        _safe_nonneg_float(
            _field(raw, "continuity_error", _field(raw, "continuity_forecast_error", 0.0))
        )
    )
    severity_residual = _clamp01(
        _safe_nonneg_float(
            _field(raw, "severity_residual", _field(raw, "severity_forecast_residual", 0.0))
        )
    )
    replay_score = _clamp01(
        _safe_nonneg_float(
            _field(raw, "replay_score", _field(raw, "replay_reconciliation_score", 1.0))
        )
    )
    confidence_calibration = _clamp01(
        _safe_nonneg_float(
            _field(
                raw,
                "confidence_calibration",
                _field(raw, "forecast_confidence_calibration", 1.0),
            )
        )
    )
    return _ReconciliationNode(
        reconciliation_id=reconciliation_id,
        residual_magnitude=residual_magnitude,
        stability_accuracy=stability_accuracy,
        continuity_error=continuity_error,
        severity_residual=severity_residual,
        replay_score=replay_score,
        confidence_calibration=confidence_calibration,
    )


def _normalize_drift_node(raw: Any, index: int) -> _DriftNode:
    drift_id = _safe_text(_field(raw, "drift_id", "")).strip() or f"drift_{index}"
    from_basin = _safe_text(_field(raw, "from_basin", "")).strip() or "unknown"
    to_basin = _safe_text(_field(raw, "to_basin", "")).strip() or "unknown"
    transition_count = _safe_nonneg_int(_field(raw, "transition_count", 0))
    drift_magnitude = _clamp01(_safe_nonneg_float(_field(raw, "drift_magnitude", 0.0)))
    drift_severity = _clamp01(
        _safe_nonneg_float(_field(raw, "drift_severity", _field(raw, "severity", 0.0)))
    )
    return _DriftNode(
        drift_id=drift_id,
        from_basin=from_basin,
        to_basin=to_basin,
        transition_count=transition_count,
        drift_magnitude=drift_magnitude,
        drift_severity=drift_severity,
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
) -> FeedbackControlScenario:
    normalized_id = _safe_text(scenario_id).strip()
    normalized_reconciliation = _ordered_reconciliation_nodes(
        _safe_series(reconciliation_series)
    )
    normalized_drift = _ordered_drift_nodes(_safe_series(drift_series))
    return FeedbackControlScenario(
        scenario_id=normalized_id,
        reconciliation_series=tuple(node.to_dict() for node in normalized_reconciliation),
        drift_series=tuple(node.to_dict() for node in normalized_drift),
    )


def _metric(name: str, value: float) -> FeedbackControlMetric:
    return FeedbackControlMetric(
        metric_name=name,
        metric_order=_METRIC_INDEX[name],
        metric_value=float(value),
    )


def _compute_metrics(
    scenario: FeedbackControlScenario,
) -> Tuple[FeedbackControlMetric, ...]:
    reconciliations = _ordered_reconciliation_nodes(scenario.reconciliation_series)
    drifts = _ordered_drift_nodes(scenario.drift_series)

    recon_count = len(reconciliations)
    if recon_count > 0:
        avg_residual = sum(n.residual_magnitude for n in reconciliations) / float(recon_count)
        avg_stability_accuracy = sum(n.stability_accuracy for n in reconciliations) / float(recon_count)
        avg_continuity_error = sum(n.continuity_error for n in reconciliations) / float(recon_count)
        avg_severity_residual = sum(n.severity_residual for n in reconciliations) / float(recon_count)
        avg_replay_score = sum(n.replay_score for n in reconciliations) / float(recon_count)
        avg_confidence_calibration = sum(
            n.confidence_calibration for n in reconciliations
        ) / float(recon_count)
    else:
        avg_residual = 1.0
        avg_stability_accuracy = 0.0
        avg_continuity_error = 1.0
        avg_severity_residual = 1.0
        avg_replay_score = 0.0
        avg_confidence_calibration = 0.0

    drift_count = len(drifts)
    if drift_count > 0:
        avg_drift_magnitude = sum(n.drift_magnitude for n in drifts) / float(drift_count)
        avg_drift_severity = sum(n.drift_severity for n in drifts) / float(drift_count)
    else:
        avg_drift_magnitude = 0.0
        avg_drift_severity = 0.0

    # residual_control_pressure — bounded combined residual with drift contribution
    residual_control_pressure = _clamp01(
        0.5 * avg_residual + 0.3 * avg_drift_magnitude + 0.2 * avg_severity_residual
    )

    # stability_correction_score — how much corrective action the stability gap suggests
    stability_correction_score = _clamp01(1.0 - avg_stability_accuracy)

    # continuity_recovery_priority — bounded priority for continuity recovery
    continuity_recovery_priority = _clamp01(
        0.7 * avg_continuity_error + 0.3 * avg_drift_severity
    )

    # severity_feedback_gain — bounded gain combining severity channels
    severity_feedback_gain = _clamp01(
        0.6 * avg_severity_residual + 0.4 * avg_drift_severity
    )

    # replay_control_stability — deterministic replay confidence of control path
    replay_control_stability = _clamp01(avg_replay_score)

    # bounded_control_confidence — combined bounded confidence in control advisory
    stability_confidence = _clamp01(avg_stability_accuracy)
    pressure_slack = _clamp01(1.0 - residual_control_pressure)
    bounded_control_confidence = _clamp01(
        0.4 * replay_control_stability
        + 0.3 * stability_confidence
        + 0.2 * pressure_slack
        + 0.1 * avg_confidence_calibration
    )

    return (
        _metric("residual_control_pressure", residual_control_pressure),
        _metric("stability_correction_score", stability_correction_score),
        _metric("continuity_recovery_priority", continuity_recovery_priority),
        _metric("severity_feedback_gain", severity_feedback_gain),
        _metric("replay_control_stability", replay_control_stability),
        _metric("bounded_control_confidence", bounded_control_confidence),
    )


def _metrics_by_name(
    metrics: Sequence[FeedbackControlMetric],
) -> Dict[str, float]:
    return {m.metric_name: float(m.metric_value) for m in metrics}


def _derive_advisory(
    metrics: Sequence[FeedbackControlMetric],
    scenario: FeedbackControlScenario,
) -> Tuple[str, Tuple[str, ...]]:
    by_name = _metrics_by_name(metrics)
    rationale: list[str] = []

    has_inputs = bool(scenario.reconciliation_series) or bool(scenario.drift_series)
    if not has_inputs:
        rationale.append("no_inputs")
        return "observe_only", tuple(rationale)

    residual_pressure = by_name.get("residual_control_pressure", 0.0)
    stability_correction = by_name.get("stability_correction_score", 0.0)
    continuity_priority = by_name.get("continuity_recovery_priority", 0.0)
    severity_gain = by_name.get("severity_feedback_gain", 0.0)
    replay_stability = by_name.get("replay_control_stability", 1.0)
    bounded_confidence = by_name.get("bounded_control_confidence", 0.0)

    # Deterministic advisory escalation lattice (fixed thresholds).
    advisory: str
    if residual_pressure >= 0.80 or severity_gain >= 0.85 or continuity_priority >= 0.85:
        advisory = "isolate"
        rationale.append("critical_residual_pressure")
    elif residual_pressure >= 0.55 or continuity_priority >= 0.6 or stability_correction >= 0.7:
        advisory = "recalibrate"
        rationale.append("elevated_recalibration_demand")
    elif residual_pressure >= 0.30 or stability_correction >= 0.4 or severity_gain >= 0.4:
        advisory = "stabilize"
        rationale.append("moderate_stabilization_demand")
    elif residual_pressure >= 0.10 or stability_correction >= 0.2:
        advisory = "monitor"
        rationale.append("low_control_pressure")
    else:
        advisory = "observe_only"
        rationale.append("nominal_state")

    if replay_stability < 0.5:
        rationale.append("replay_instability")
    if bounded_confidence < 0.3:
        rationale.append("low_bounded_confidence")

    return advisory, tuple(rationale)


def _metrics_hash(metrics: Sequence[FeedbackControlMetric]) -> str:
    payload = [metric.to_dict() for metric in metrics]
    return _sha256_hex(_canonical_json(payload).encode("utf-8"))


def _advisory_hash(advisory_output: str, advisory_rationale: Sequence[str]) -> str:
    payload = {
        "advisory_output": _safe_text(advisory_output),
        "advisory_rationale": list(advisory_rationale),
    }
    return _sha256_hex(_canonical_json(payload).encode("utf-8"))


def _analysis_hash(
    *,
    scenario: FeedbackControlScenario,
    metrics: Sequence[FeedbackControlMetric],
    advisory_output: str,
    advisory_rationale: Sequence[str],
    violations: Sequence[str],
) -> str:
    body = {
        "scenario": scenario.to_dict(),
        "metrics": [metric.to_dict() for metric in metrics],
        "advisory_output": _safe_text(advisory_output),
        "advisory_rationale": list(advisory_rationale),
        "violations": list(violations),
    }
    return _sha256_hex(_canonical_json(body).encode("utf-8"))


def validate_feedback_control(kernel: Any) -> Tuple[str, ...]:
    violations: list[str] = []
    try:
        scenario = _field(kernel, "scenario", None)
        metrics = tuple(_safe_series(_field(kernel, "metrics", ())))
        advisory_output = _safe_text(_field(kernel, "advisory_output", "")).strip()
        advisory_rationale = tuple(
            _safe_text(item)
            for item in _safe_series(_field(kernel, "advisory_rationale", ()))
        )
        receipt = _field(kernel, "receipt", None)

        if scenario is None:
            violations.append("missing_scenario")
            return tuple(sorted(set(violations)))

        if not _safe_text(_field(scenario, "scenario_id", "")).strip():
            violations.append("empty_scenario_id")

        reconciliation_series = _safe_series(
            _field(scenario, "reconciliation_series", ())
        )
        drift_series = _safe_series(_field(scenario, "drift_series", ()))

        if not reconciliation_series and not drift_series:
            violations.append("empty_input_series")

        for idx, row in enumerate(reconciliation_series):
            reconciliation_id = _safe_text(
                _field(row, "reconciliation_id", "")
            ).strip()
            if not reconciliation_id:
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

        if advisory_output and advisory_output not in _ADVISORY_INDEX:
            violations.append("advisory_output_unknown")

        if receipt is not None:
            actual_metrics_hash = _safe_text(_field(receipt, "metrics_hash", "")).strip()
            if actual_metrics_hash:
                expected_metrics_hash = _metrics_hash(metrics)
                if expected_metrics_hash != actual_metrics_hash:
                    violations.append("receipt_metrics_hash_mismatch")

            actual_scenario_hash = _safe_text(_field(receipt, "scenario_hash", "")).strip()
            if actual_scenario_hash:
                try:
                    expected_scenario_hash = scenario.stable_hash()
                except Exception:
                    expected_scenario_hash = ""
                if expected_scenario_hash and expected_scenario_hash != actual_scenario_hash:
                    violations.append("receipt_scenario_hash_mismatch")

            actual_advisory_hash = _safe_text(_field(receipt, "advisory_hash", "")).strip()
            if actual_advisory_hash:
                expected_advisory_hash = _advisory_hash(advisory_output, advisory_rationale)
                if expected_advisory_hash != actual_advisory_hash:
                    violations.append("receipt_advisory_hash_mismatch")

            actual_analysis_hash = _safe_text(_field(receipt, "analysis_hash", "")).strip()
            expected_analysis_hash = _safe_text(
                _field(kernel, "analysis_hash", "")
            ).strip()
            if actual_analysis_hash and expected_analysis_hash:
                if actual_analysis_hash != expected_analysis_hash:
                    violations.append("receipt_analysis_hash_mismatch")

            receipt_advisory_output = _safe_text(
                _field(receipt, "advisory_output", "")
            ).strip()
            if receipt_advisory_output and advisory_output:
                if receipt_advisory_output != advisory_output:
                    violations.append("receipt_advisory_output_mismatch")

            actual_receipt_hash = _safe_text(_field(receipt, "receipt_hash", "")).strip()
            if actual_receipt_hash:
                confidence_raw = _field(receipt, "bounded_control_confidence", 0.0)
                try:
                    confidence = float(confidence_raw)
                except Exception:
                    confidence = 0.0
                expected_receipt_payload = {
                    "scenario_hash": _safe_text(
                        _field(receipt, "scenario_hash", "")
                    ).strip(),
                    "metrics_hash": _safe_text(
                        _field(receipt, "metrics_hash", "")
                    ).strip(),
                    "advisory_hash": _safe_text(
                        _field(receipt, "advisory_hash", "")
                    ).strip(),
                    "analysis_hash": _safe_text(
                        _field(receipt, "analysis_hash", "")
                    ).strip(),
                    "bounded_control_confidence": confidence,
                    "advisory_output": _safe_text(
                        _field(receipt, "advisory_output", "")
                    ).strip(),
                }
                expected_receipt_hash = _sha256_hex(
                    _canonical_json(expected_receipt_payload).encode("utf-8")
                )
                if expected_receipt_hash != actual_receipt_hash:
                    violations.append("receipt_hash_mismatch")
    except Exception as exc:
        violations.append(f"validator_internal_error:{_safe_text(exc)}")

    return tuple(sorted(set(violations)))


def build_feedback_control_receipt(
    *,
    scenario: FeedbackControlScenario,
    metrics: Sequence[FeedbackControlMetric],
    advisory_output: str,
    advisory_rationale: Sequence[str],
    analysis_hash: str,
) -> FeedbackControlReceipt:
    bounded_control_confidence = 0.0
    for metric in metrics:
        if metric.metric_name == "bounded_control_confidence":
            bounded_control_confidence = float(metric.metric_value)
            break

    normalized_advisory = _safe_text(advisory_output).strip()
    normalized_rationale = tuple(_safe_text(item) for item in advisory_rationale)

    payload = {
        "scenario_hash": scenario.stable_hash(),
        "metrics_hash": _metrics_hash(metrics),
        "advisory_hash": _advisory_hash(normalized_advisory, normalized_rationale),
        "analysis_hash": _safe_text(analysis_hash).strip(),
        "bounded_control_confidence": bounded_control_confidence,
        "advisory_output": normalized_advisory,
    }
    receipt_hash = _sha256_hex(_canonical_json(payload).encode("utf-8"))
    return FeedbackControlReceipt(
        scenario_hash=payload["scenario_hash"],
        metrics_hash=payload["metrics_hash"],
        advisory_hash=payload["advisory_hash"],
        analysis_hash=payload["analysis_hash"],
        bounded_control_confidence=bounded_control_confidence,
        advisory_output=normalized_advisory,
        receipt_hash=receipt_hash,
    )


def run_governance_feedback_control(
    *,
    scenario: FeedbackControlScenario,
) -> GovernanceForecastFeedbackControlKernel:
    metrics = _compute_metrics(scenario)
    advisory_output, advisory_rationale = _derive_advisory(metrics, scenario)

    provisional = GovernanceForecastFeedbackControlKernel(
        scenario=scenario,
        metrics=metrics,
        advisory_output=advisory_output,
        advisory_rationale=advisory_rationale,
        violations=(),
        receipt=FeedbackControlReceipt("", "", "", "", 0.0, "", ""),
        analysis_hash="",
    )
    violations = validate_feedback_control(provisional)
    analysis_hash = _analysis_hash(
        scenario=scenario,
        metrics=metrics,
        advisory_output=advisory_output,
        advisory_rationale=advisory_rationale,
        violations=violations,
    )
    receipt = build_feedback_control_receipt(
        scenario=scenario,
        metrics=metrics,
        advisory_output=advisory_output,
        advisory_rationale=advisory_rationale,
        analysis_hash=analysis_hash,
    )
    return GovernanceForecastFeedbackControlKernel(
        scenario=scenario,
        metrics=metrics,
        advisory_output=advisory_output,
        advisory_rationale=advisory_rationale,
        violations=violations,
        receipt=receipt,
        analysis_hash=analysis_hash,
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
    if baseline.analysis_hash != replay.analysis_hash:
        mismatches.append("analysis_hash")
    if baseline.receipt.receipt_hash != replay.receipt.receipt_hash:
        mismatches.append("receipt_hash")
    if baseline.advisory_output != replay.advisory_output:
        mismatches.append("advisory_output")

    return {
        "is_stable_replay": len(mismatches) == 0,
        "mismatches": tuple(mismatches),
        "metric_delta": tuple((name, metric_delta[name]) for name in _METRIC_ORDER),
        "baseline_hash": baseline.stable_hash(),
        "replay_hash": replay.stable_hash(),
    }


def summarize_feedback_control(
    kernel: GovernanceForecastFeedbackControlKernel,
) -> str:
    lines = [
        f"scenario_id={kernel.scenario.scenario_id}",
        f"scenario_hash={kernel.scenario.stable_hash()}",
        f"analysis_hash={kernel.analysis_hash}",
        f"receipt_hash={kernel.receipt.receipt_hash}",
        f"advisory_output={kernel.advisory_output}",
        "metrics:",
    ]
    for metric in kernel.metrics:
        lines.append(
            f"- {metric.metric_order}:{metric.metric_name}={metric.metric_value:.12f}"
        )
    if kernel.advisory_rationale:
        lines.append("rationale:")
        for item in kernel.advisory_rationale:
            lines.append(f"- {item}")
    else:
        lines.append("rationale: none")
    if kernel.violations:
        lines.append("violations:")
        for item in kernel.violations:
            lines.append(f"- {item}")
    else:
        lines.append("violations: none")
    return "\n".join(lines)


__all__ = [
    "FeedbackControlScenario",
    "FeedbackControlMetric",
    "FeedbackControlReceipt",
    "GovernanceForecastFeedbackControlKernel",
    "build_feedback_control_scenario",
    "run_governance_feedback_control",
    "validate_feedback_control",
    "build_feedback_control_receipt",
    "compare_feedback_control_replay",
    "summarize_feedback_control",
]
