"""v137.20.1 — Governance Control Stability Calibration Kernel.

Additive advisory layer that converts deterministic reconciliation residuals
together with a prior governance feedback-control analysis into bounded
calibration advisories for governance control thresholds and gains.

Canonical model:

    reconciliation_series + control_analysis
        -> calibration_analysis + calibration_receipt

This kernel is strictly advisory:

- no decoder mutation
- no input mutation
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
    "threshold_drift_score",
    "control_gain_calibration",
    "residual_confidence_adjustment",
    "continuity_threshold_pressure",
    "stability_gain_normalization",
    "calibration_confidence_score",
)
_METRIC_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(_METRIC_ORDER)}


_ADVISORY_ORDER: Tuple[str, ...] = (
    "hold_thresholds",
    "soft_adjust",
    "moderate_recalibration",
    "aggressive_recalibration",
)
_ADVISORY_RANK: Dict[str, int] = {name: idx for idx, name in enumerate(_ADVISORY_ORDER)}


_CONTROL_KEYS: Tuple[str, ...] = (
    "bounded_control_confidence",
    "calibration_feedback_signal",
    "continuity_recovery_priority",
    "replay_control_stability",
    "residual_control_pressure",
    "severity_feedback_gain",
    "stability_correction_score",
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


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def _control_float(control: Any, key: str, default: float) -> float:
    raw = _field(control, key, None)
    if raw is None:
        return _clamp01(default)
    return _clamp01(_safe_nonneg_float(raw))


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


def _normalize_reconciliation_node(raw: Any, index: int) -> _ReconciliationNode:
    rid = (
        _safe_text(_field(raw, "reconciliation_id", "")).strip()
        or f"reconciliation_{index}"
    )
    return _ReconciliationNode(
        reconciliation_id=rid,
        residual_magnitude=_clamp01(
            _safe_nonneg_float(_field(raw, "residual_magnitude", 0.0))
        ),
        stability_residual=_clamp01(
            _safe_nonneg_float(_field(raw, "stability_residual", 0.0))
        ),
        continuity_residual=_clamp01(
            _safe_nonneg_float(_field(raw, "continuity_residual", 0.0))
        ),
        severity_residual=_clamp01(
            _safe_nonneg_float(_field(raw, "severity_residual", 0.0))
        ),
        replay_residual=_clamp01(
            _safe_nonneg_float(_field(raw, "replay_residual", 0.0))
        ),
        calibration_residual=_clamp01(
            _safe_nonneg_float(_field(raw, "calibration_residual", 0.0))
        ),
    )


def _ordered_reconciliation_nodes(
    series: Sequence[Any],
) -> Tuple[_ReconciliationNode, ...]:
    return tuple(
        _normalize_reconciliation_node(raw, index=i) for i, raw in enumerate(series)
    )


def _normalize_control_analysis(raw: Any) -> Dict[str, Any]:
    flat: Dict[str, float] = {key: 0.0 for key in _CONTROL_KEYS}
    # Neutral default so absent control signal does not synthesise pressure.
    flat["bounded_control_confidence"] = 1.0

    if raw is None:
        out: Dict[str, Any] = {key: flat[key] for key in _CONTROL_KEYS}
        out["aggregate_recommendation"] = ""
        return out

    metrics_seq = _safe_series(_field(raw, "metrics", None))
    for metric in metrics_seq:
        name = _safe_text(_field(metric, "metric_name", "")).strip()
        if name in flat:
            flat[name] = _clamp01(
                _safe_nonneg_float(_field(metric, "metric_value", 0.0))
            )

    if isinstance(raw, Mapping):
        for key in _CONTROL_KEYS:
            if key in raw:
                flat[key] = _clamp01(_safe_nonneg_float(raw.get(key, 0.0)))

    aggregate = _safe_text(_field(raw, "aggregate_recommendation", "")).strip()

    out = {key: flat[key] for key in _CONTROL_KEYS}
    out["aggregate_recommendation"] = aggregate
    return out


@dataclass(frozen=True)
class ControlCalibrationScenario:
    scenario_id: str
    reconciliation_series: Tuple[Mapping[str, Any], ...]
    control_analysis: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "reconciliation_series": [
                dict(item) for item in self.reconciliation_series
            ],
            "control_analysis": dict(self.control_analysis),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_canonical_json().encode("utf-8"))


@dataclass(frozen=True)
class ControlCalibrationMetric:
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
class ControlCalibrationReceipt:
    scenario_hash: str
    metrics_hash: str
    advisories_hash: str
    calibration_hash: str
    calibration_confidence_score: float
    receipt_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_hash": self.scenario_hash,
            "metrics_hash": self.metrics_hash,
            "advisories_hash": self.advisories_hash,
            "calibration_hash": self.calibration_hash,
            "calibration_confidence_score": self.calibration_confidence_score,
            "receipt_hash": self.receipt_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.receipt_hash


@dataclass(frozen=True)
class GovernanceControlStabilityCalibrationKernel:
    scenario: ControlCalibrationScenario
    metrics: Tuple[ControlCalibrationMetric, ...]
    advisories: Tuple[Tuple[str, str], ...]
    aggregate_advisory: str
    violations: Tuple[str, ...]
    receipt: ControlCalibrationReceipt
    calibration_hash: str
    advisory_only: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario.to_dict(),
            "metrics": [m.to_dict() for m in self.metrics],
            "advisories": [[rid, rec] for rid, rec in self.advisories],
            "aggregate_advisory": self.aggregate_advisory,
            "violations": list(self.violations),
            "receipt": self.receipt.to_dict(),
            "calibration_hash": self.calibration_hash,
            "advisory_only": bool(self.advisory_only),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.calibration_hash


def build_control_calibration_scenario(
    *,
    scenario_id: str,
    reconciliation_series: Any,
    control_analysis: Any,
) -> ControlCalibrationScenario:
    normalized_id = _safe_text(scenario_id).strip()
    nodes = _ordered_reconciliation_nodes(_safe_series(reconciliation_series))
    normalized_control = _normalize_control_analysis(control_analysis)
    return ControlCalibrationScenario(
        scenario_id=normalized_id,
        reconciliation_series=tuple(node.to_dict() for node in nodes),
        control_analysis=normalized_control,
    )


def _metric(name: str, value: float) -> ControlCalibrationMetric:
    return ControlCalibrationMetric(
        metric_name=name,
        metric_order=_METRIC_INDEX[name],
        metric_value=float(value),
    )


def _compute_metrics(
    scenario: ControlCalibrationScenario,
) -> Tuple[ControlCalibrationMetric, ...]:
    nodes = _ordered_reconciliation_nodes(scenario.reconciliation_series)
    control = scenario.control_analysis

    avg_residual = _mean(tuple(n.residual_magnitude for n in nodes))
    avg_stability = _mean(tuple(n.stability_residual for n in nodes))
    avg_continuity = _mean(tuple(n.continuity_residual for n in nodes))

    residual_pressure = _control_float(control, "residual_control_pressure", 0.0)
    severity_gain = _control_float(control, "severity_feedback_gain", 0.0)
    stability_correction = _control_float(control, "stability_correction_score", 0.0)
    continuity_priority = _control_float(control, "continuity_recovery_priority", 0.0)
    bounded_confidence = _control_float(control, "bounded_control_confidence", 1.0)

    threshold_drift_score = _clamp01(0.5 * avg_residual + 0.5 * residual_pressure)
    control_gain_calibration = _clamp01(0.6 * residual_pressure + 0.4 * severity_gain)
    residual_confidence_adjustment = _clamp01(1.0 - bounded_confidence)
    continuity_threshold_pressure = _clamp01(
        0.5 * avg_continuity + 0.5 * continuity_priority
    )
    stability_gain_normalization = _clamp01(
        1.0 - (0.5 * stability_correction + 0.5 * avg_stability)
    )
    calibration_confidence_score = _clamp01(
        1.0
        - (
            0.4 * threshold_drift_score
            + 0.3 * residual_confidence_adjustment
            + 0.3 * control_gain_calibration
        )
    )

    return (
        _metric("threshold_drift_score", threshold_drift_score),
        _metric("control_gain_calibration", control_gain_calibration),
        _metric("residual_confidence_adjustment", residual_confidence_adjustment),
        _metric("continuity_threshold_pressure", continuity_threshold_pressure),
        _metric("stability_gain_normalization", stability_gain_normalization),
        _metric("calibration_confidence_score", calibration_confidence_score),
    )


def _entry_advisory(node: _ReconciliationNode) -> str:
    pressure = max(node.residual_magnitude, node.max_category_residual())
    if pressure < 0.2:
        return "hold_thresholds"
    if pressure < 0.5:
        return "soft_adjust"
    if pressure < 0.8:
        return "moderate_recalibration"
    return "aggressive_recalibration"


def _compute_advisories(
    scenario: ControlCalibrationScenario,
) -> Tuple[Tuple[str, str], ...]:
    nodes = _ordered_reconciliation_nodes(scenario.reconciliation_series)
    return tuple((node.reconciliation_id, _entry_advisory(node)) for node in nodes)


def _aggregate_advisory(
    metrics: Sequence[ControlCalibrationMetric],
    advisories: Sequence[Tuple[str, str]],
) -> str:
    by_name = {m.metric_name: m.metric_value for m in metrics}
    drift = float(by_name.get("threshold_drift_score", 0.0))
    gain = float(by_name.get("control_gain_calibration", 0.0))
    confidence_adj = float(by_name.get("residual_confidence_adjustment", 0.0))
    pressure = max(drift, gain, confidence_adj)

    if pressure < 0.2:
        metric_based = "hold_thresholds"
    elif pressure < 0.5:
        metric_based = "soft_adjust"
    elif pressure < 0.8:
        metric_based = "moderate_recalibration"
    else:
        metric_based = "aggressive_recalibration"

    metric_rank = _ADVISORY_RANK[metric_based]
    worst_rank = 0
    for _rid, rec in advisories:
        rank = _ADVISORY_RANK.get(rec, 0)
        if rank > worst_rank:
            worst_rank = rank

    return _ADVISORY_ORDER[max(metric_rank, worst_rank)]


def _metrics_hash(metrics: Sequence[ControlCalibrationMetric]) -> str:
    payload = [m.to_dict() for m in metrics]
    return _sha256_hex(_canonical_json(payload).encode("utf-8"))


def _advisories_hash(
    advisories: Sequence[Tuple[str, str]],
    aggregate_advisory: str,
) -> str:
    payload = {
        "advisories": [[rid, rec] for rid, rec in advisories],
        "aggregate_advisory": aggregate_advisory,
    }
    return _sha256_hex(_canonical_json(payload).encode("utf-8"))


def _calibration_hash(
    *,
    scenario: ControlCalibrationScenario,
    metrics: Sequence[ControlCalibrationMetric],
    advisories: Sequence[Tuple[str, str]],
    aggregate_advisory: str,
    violations: Sequence[str],
) -> str:
    body = {
        "scenario": scenario.to_dict(),
        "metrics": [m.to_dict() for m in metrics],
        "advisories": [[rid, rec] for rid, rec in advisories],
        "aggregate_advisory": aggregate_advisory,
        "violations": list(violations),
    }
    return _sha256_hex(_canonical_json(body).encode("utf-8"))


def validate_control_calibration(kernel: Any) -> Tuple[str, ...]:
    violations: list[str] = []
    try:
        scenario = _field(kernel, "scenario", None)
        metrics = tuple(_safe_series(_field(kernel, "metrics", ())))
        advisories = tuple(_safe_series(_field(kernel, "advisories", ())))
        aggregate = _safe_text(_field(kernel, "aggregate_advisory", "")).strip()
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
        if not reconciliation_series:
            violations.append("empty_reconciliation_series")

        for idx, row in enumerate(reconciliation_series):
            rec_id = _safe_text(_field(row, "reconciliation_id", "")).strip()
            if not rec_id:
                violations.append(f"malformed_reconciliation_row:{idx}")

        control_analysis = _field(scenario, "control_analysis", None)
        if control_analysis is None:
            violations.append("missing_control_analysis")
        elif not isinstance(control_analysis, Mapping):
            violations.append("malformed_control_analysis")

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

        for idx, entry in enumerate(advisories):
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                violations.append(f"malformed_advisory:{idx}")
                continue
            adv_name = _safe_text(entry[1]).strip()
            if adv_name not in _ADVISORY_RANK:
                violations.append(f"unknown_advisory:{idx}")

        if aggregate and aggregate not in _ADVISORY_RANK:
            violations.append("unknown_aggregate_advisory")

        if advisory_only is not True:
            violations.append("advisory_only_flag_disabled")

        if receipt is not None:
            actual_metrics_hash = _safe_text(
                _field(receipt, "metrics_hash", "")
            ).strip()
            if actual_metrics_hash:
                expected_metrics_hash = _metrics_hash(metrics)
                if expected_metrics_hash != actual_metrics_hash:
                    violations.append("receipt_metrics_hash_mismatch")

            actual_scenario_hash = _safe_text(
                _field(receipt, "scenario_hash", "")
            ).strip()
            if actual_scenario_hash and hasattr(scenario, "stable_hash"):
                try:
                    expected_scenario_hash = scenario.stable_hash()
                except Exception:
                    expected_scenario_hash = ""
                if expected_scenario_hash and expected_scenario_hash != actual_scenario_hash:
                    violations.append("receipt_scenario_hash_mismatch")

            actual_advisories_hash = _safe_text(
                _field(receipt, "advisories_hash", "")
            ).strip()
            if actual_advisories_hash:
                expected_advisories_hash = _advisories_hash(advisories, aggregate)
                if expected_advisories_hash != actual_advisories_hash:
                    violations.append("receipt_advisories_hash_mismatch")

            actual_calibration_hash = _safe_text(
                _field(receipt, "calibration_hash", "")
            ).strip()
            expected_calibration_hash = _safe_text(
                _field(kernel, "calibration_hash", "")
            ).strip()
            if actual_calibration_hash and expected_calibration_hash:
                if actual_calibration_hash != expected_calibration_hash:
                    violations.append("receipt_calibration_hash_mismatch")

            actual_receipt_hash = _safe_text(
                _field(receipt, "receipt_hash", "")
            ).strip()
            if actual_receipt_hash:
                conf_raw = _field(receipt, "calibration_confidence_score", 0.0)
                try:
                    conf_value = float(conf_raw)
                except Exception:
                    conf_value = 0.0
                expected_payload = {
                    "scenario_hash": _safe_text(
                        _field(receipt, "scenario_hash", "")
                    ).strip(),
                    "metrics_hash": _safe_text(
                        _field(receipt, "metrics_hash", "")
                    ).strip(),
                    "advisories_hash": _safe_text(
                        _field(receipt, "advisories_hash", "")
                    ).strip(),
                    "calibration_hash": _safe_text(
                        _field(receipt, "calibration_hash", "")
                    ).strip(),
                    "calibration_confidence_score": conf_value,
                }
                expected_receipt_hash = _sha256_hex(
                    _canonical_json(expected_payload).encode("utf-8")
                )
                if expected_receipt_hash != actual_receipt_hash:
                    violations.append("receipt_hash_mismatch")
    except Exception as exc:
        violations.append(f"validator_internal_error:{_safe_text(exc)}")

    return tuple(sorted(set(violations)))


def build_control_calibration_receipt(
    *,
    scenario: ControlCalibrationScenario,
    metrics: Sequence[ControlCalibrationMetric],
    advisories: Sequence[Tuple[str, str]],
    aggregate_advisory: str,
    calibration_hash: str,
) -> ControlCalibrationReceipt:
    confidence_score = 0.0
    for m in metrics:
        if m.metric_name == "calibration_confidence_score":
            confidence_score = float(m.metric_value)
            break

    payload = {
        "scenario_hash": scenario.stable_hash(),
        "metrics_hash": _metrics_hash(metrics),
        "advisories_hash": _advisories_hash(advisories, aggregate_advisory),
        "calibration_hash": _safe_text(calibration_hash).strip(),
        "calibration_confidence_score": confidence_score,
    }
    receipt_hash = _sha256_hex(_canonical_json(payload).encode("utf-8"))
    return ControlCalibrationReceipt(
        scenario_hash=payload["scenario_hash"],
        metrics_hash=payload["metrics_hash"],
        advisories_hash=payload["advisories_hash"],
        calibration_hash=payload["calibration_hash"],
        calibration_confidence_score=confidence_score,
        receipt_hash=receipt_hash,
    )


def run_governance_control_calibration(
    *,
    scenario: ControlCalibrationScenario,
) -> GovernanceControlStabilityCalibrationKernel:
    metrics = _compute_metrics(scenario)
    advisories = _compute_advisories(scenario)
    aggregate_advisory = _aggregate_advisory(metrics, advisories)
    provisional = GovernanceControlStabilityCalibrationKernel(
        scenario=scenario,
        metrics=metrics,
        advisories=advisories,
        aggregate_advisory=aggregate_advisory,
        violations=(),
        receipt=ControlCalibrationReceipt("", "", "", "", 0.0, ""),
        calibration_hash="",
        advisory_only=True,
    )
    violations = validate_control_calibration(provisional)
    calibration_hash = _calibration_hash(
        scenario=scenario,
        metrics=metrics,
        advisories=advisories,
        aggregate_advisory=aggregate_advisory,
        violations=violations,
    )
    receipt = build_control_calibration_receipt(
        scenario=scenario,
        metrics=metrics,
        advisories=advisories,
        aggregate_advisory=aggregate_advisory,
        calibration_hash=calibration_hash,
    )
    return GovernanceControlStabilityCalibrationKernel(
        scenario=scenario,
        metrics=metrics,
        advisories=advisories,
        aggregate_advisory=aggregate_advisory,
        violations=violations,
        receipt=receipt,
        calibration_hash=calibration_hash,
        advisory_only=True,
    )


def compare_control_calibration_replay(
    baseline: GovernanceControlStabilityCalibrationKernel,
    replay: GovernanceControlStabilityCalibrationKernel,
) -> Dict[str, Any]:
    baseline_by_name = {m.metric_name: m.metric_value for m in baseline.metrics}
    replay_by_name = {m.metric_name: m.metric_value for m in replay.metrics}
    metric_delta: Dict[str, float] = {}
    for name in _METRIC_ORDER:
        metric_delta[name] = float(
            replay_by_name.get(name, 0.0) - baseline_by_name.get(name, 0.0)
        )

    mismatches: list[str] = []
    if baseline.scenario.stable_hash() != replay.scenario.stable_hash():
        mismatches.append("scenario_hash")
    if baseline.calibration_hash != replay.calibration_hash:
        mismatches.append("calibration_hash")
    if baseline.receipt.receipt_hash != replay.receipt.receipt_hash:
        mismatches.append("receipt_hash")
    if baseline.aggregate_advisory != replay.aggregate_advisory:
        mismatches.append("aggregate_advisory")

    return {
        "is_stable_replay": len(mismatches) == 0,
        "mismatches": tuple(mismatches),
        "metric_delta": tuple((name, metric_delta[name]) for name in _METRIC_ORDER),
        "baseline_hash": baseline.stable_hash(),
        "replay_hash": replay.stable_hash(),
        "baseline_aggregate_advisory": baseline.aggregate_advisory,
        "replay_aggregate_advisory": replay.aggregate_advisory,
    }


def summarize_control_calibration(
    kernel: GovernanceControlStabilityCalibrationKernel,
) -> str:
    lines = [
        f"scenario_id={kernel.scenario.scenario_id}",
        f"scenario_hash={kernel.scenario.stable_hash()}",
        f"calibration_hash={kernel.calibration_hash}",
        f"receipt_hash={kernel.receipt.receipt_hash}",
        f"aggregate_advisory={kernel.aggregate_advisory}",
        f"advisory_only={bool(kernel.advisory_only)}",
        "metrics:",
    ]
    for m in kernel.metrics:
        lines.append(
            f"- {m.metric_order}:{m.metric_name}={m.metric_value:.12f}"
        )
    if kernel.advisories:
        lines.append("advisories:")
        for rid, rec in kernel.advisories:
            lines.append(f"- {rid}={rec}")
    else:
        lines.append("advisories: none")
    if kernel.violations:
        lines.append("violations:")
        for v in kernel.violations:
            lines.append(f"- {v}")
    else:
        lines.append("violations: none")
    return "\n".join(lines)


__all__ = [
    "ControlCalibrationScenario",
    "ControlCalibrationMetric",
    "ControlCalibrationReceipt",
    "GovernanceControlStabilityCalibrationKernel",
    "build_control_calibration_scenario",
    "run_governance_control_calibration",
    "validate_control_calibration",
    "build_control_calibration_receipt",
    "compare_control_calibration_replay",
    "summarize_control_calibration",
]
