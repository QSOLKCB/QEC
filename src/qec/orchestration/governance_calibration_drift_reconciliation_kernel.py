"""v137.20.2 — Governance Calibration Drift Reconciliation Kernel.

Additive deterministic reconciliation-analysis over calibration outputs
compared across replay horizons. Advisory-only. No decoder touch.

Canonical model:
- calibration_series + prior_calibration_series
  -> drift_reconciliation_analysis + drift_reconciliation_receipt

Invariants:
- no randomness
- no async
- no external I/O
- deterministic ordering
- canonical JSON + SHA-256
- validator must never raise
- no mutation of inputs
- advisory-only, never executes actions
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Sequence, Tuple


_METRIC_ORDER: Tuple[str, ...] = (
    "threshold_drift_delta",
    "confidence_reconciliation_score",
    "continuity_drift_pressure",
    "stability_recovery_alignment",
    "cross_replay_drift_score",
    "drift_reconciliation_confidence",
)
_METRIC_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(_METRIC_ORDER)}

_ADVISORY_ORDER: Tuple[str, ...] = (
    "stable_alignment",
    "minor_drift_reconcile",
    "moderate_drift_reconcile",
    "severe_drift_reconcile",
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


def _safe_finite_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        parsed = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(parsed):
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
class _CalibrationNode:
    calibration_id: str
    threshold: float
    confidence: float
    continuity_signal: bool
    stability_level: float
    replay_identity: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calibration_id": self.calibration_id,
            "threshold": self.threshold,
            "confidence": self.confidence,
            "continuity_signal": self.continuity_signal,
            "stability_level": self.stability_level,
            "replay_identity": self.replay_identity,
        }


@dataclass(frozen=True)
class CalibrationDriftScenario:
    scenario_id: str
    calibration_series: Tuple[Mapping[str, Any], ...]
    prior_calibration_series: Tuple[Mapping[str, Any], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "calibration_series": [dict(item) for item in self.calibration_series],
            "prior_calibration_series": [dict(item) for item in self.prior_calibration_series],
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_canonical_json().encode("utf-8"))


@dataclass(frozen=True)
class CalibrationDriftMetric:
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
class CalibrationDriftReceipt:
    scenario_hash: str
    metrics_hash: str
    advisory_hash: str
    reconciliation_hash: str
    drift_reconciliation_confidence: float
    receipt_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_hash": self.scenario_hash,
            "metrics_hash": self.metrics_hash,
            "advisory_hash": self.advisory_hash,
            "reconciliation_hash": self.reconciliation_hash,
            "drift_reconciliation_confidence": self.drift_reconciliation_confidence,
            "receipt_hash": self.receipt_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.receipt_hash


@dataclass(frozen=True)
class GovernanceCalibrationDriftReconciliationKernel:
    scenario: CalibrationDriftScenario
    metrics: Tuple[CalibrationDriftMetric, ...]
    advisory: Tuple[str, ...]
    violations: Tuple[str, ...]
    receipt: CalibrationDriftReceipt
    reconciliation_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario.to_dict(),
            "metrics": [metric.to_dict() for metric in self.metrics],
            "advisory": list(self.advisory),
            "violations": list(self.violations),
            "receipt": self.receipt.to_dict(),
            "reconciliation_hash": self.reconciliation_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.reconciliation_hash


def _normalize_calibration_node(raw: Any, index: int) -> _CalibrationNode:
    calibration_id = _safe_text(_field(raw, "calibration_id", "")).strip() or f"calibration_{index}"
    threshold = _clamp01(_safe_finite_float(_field(raw, "threshold", 0.0)))
    confidence = _clamp01(_safe_finite_float(_field(raw, "confidence", 0.0)))
    continuity_signal = _safe_bool(_field(raw, "continuity_signal", _field(raw, "continuity_ok", False)))
    stability_level = _clamp01(
        _safe_finite_float(_field(raw, "stability_level", _field(raw, "stability", 1.0)))
    )
    replay_identity = _safe_text(_field(raw, "replay_identity", "")).strip()
    return _CalibrationNode(
        calibration_id=calibration_id,
        threshold=threshold,
        confidence=confidence,
        continuity_signal=continuity_signal,
        stability_level=stability_level,
        replay_identity=replay_identity,
    )


def _ordered_calibration_nodes(series: Sequence[Any]) -> Tuple[_CalibrationNode, ...]:
    return tuple(_normalize_calibration_node(raw, index=i) for i, raw in enumerate(series))


def build_calibration_drift_scenario(
    *,
    scenario_id: str,
    calibration_series: Any,
    prior_calibration_series: Any,
) -> CalibrationDriftScenario:
    normalized_id = _safe_text(scenario_id).strip()
    normalized_current = _ordered_calibration_nodes(_safe_series(calibration_series))
    normalized_prior = _ordered_calibration_nodes(_safe_series(prior_calibration_series))
    return CalibrationDriftScenario(
        scenario_id=normalized_id,
        calibration_series=tuple(node.to_dict() for node in normalized_current),
        prior_calibration_series=tuple(node.to_dict() for node in normalized_prior),
    )


def _metric(name: str, value: float) -> CalibrationDriftMetric:
    return CalibrationDriftMetric(
        metric_name=name,
        metric_order=_METRIC_INDEX[name],
        metric_value=float(value),
    )


def _compute_metrics(scenario: CalibrationDriftScenario) -> Tuple[CalibrationDriftMetric, ...]:
    current = _ordered_calibration_nodes(scenario.calibration_series)
    prior = _ordered_calibration_nodes(scenario.prior_calibration_series)

    prior_by_id: Dict[str, list[_CalibrationNode]] = {}
    for node in prior:
        prior_by_id.setdefault(node.calibration_id, []).append(node)

    match_index: Dict[str, int] = {}
    matched_pairs: list[Tuple[_CalibrationNode, _CalibrationNode]] = []
    for node in current:
        group = prior_by_id.get(node.calibration_id)
        if not group:
            continue
        idx = match_index.get(node.calibration_id, 0)
        if idx >= len(group):
            continue
        matched_pairs.append((node, group[idx]))
        match_index[node.calibration_id] = idx + 1

    pair_count = len(matched_pairs)
    if pair_count > 0:
        threshold_delta_sum = 0.0
        confidence_delta_sum = 0.0
        continuity_flip_sum = 0.0
        stability_delta_sum = 0.0
        replay_mismatch_sum = 0.0
        for cur, pri in matched_pairs:
            threshold_delta_sum += abs(cur.threshold - pri.threshold)
            confidence_delta_sum += abs(cur.confidence - pri.confidence)
            continuity_flip_sum += (
                0.0
                if cur.continuity_signal == pri.continuity_signal
                else 1.0
            )
            stability_delta_sum += abs(cur.stability_level - pri.stability_level)
            replay_mismatch = 0.0
            if cur.replay_identity and pri.replay_identity:
                replay_mismatch = 0.0 if cur.replay_identity == pri.replay_identity else 1.0
            elif cur.replay_identity != pri.replay_identity:
                replay_mismatch = 1.0
            replay_mismatch_sum += replay_mismatch

        threshold_drift_delta = _clamp01(threshold_delta_sum / float(pair_count))
        confidence_reconciliation_score = _clamp01(1.0 - (confidence_delta_sum / float(pair_count)))
        continuity_drift_pressure = _clamp01(continuity_flip_sum / float(pair_count))
        stability_recovery_alignment = _clamp01(1.0 - (stability_delta_sum / float(pair_count)))
        cross_replay_drift_score = _clamp01(replay_mismatch_sum / float(pair_count))
    else:
        threshold_drift_delta = 1.0
        confidence_reconciliation_score = 0.0
        continuity_drift_pressure = 1.0
        stability_recovery_alignment = 0.0
        cross_replay_drift_score = 1.0

    # Composite confidence: high when drift pressure is low and alignment is high.
    pressure = (
        threshold_drift_delta
        + continuity_drift_pressure
        + cross_replay_drift_score
        + (1.0 - confidence_reconciliation_score)
        + (1.0 - stability_recovery_alignment)
    ) / 5.0
    drift_reconciliation_confidence = _clamp01(1.0 - pressure)

    return (
        _metric("threshold_drift_delta", threshold_drift_delta),
        _metric("confidence_reconciliation_score", confidence_reconciliation_score),
        _metric("continuity_drift_pressure", continuity_drift_pressure),
        _metric("stability_recovery_alignment", stability_recovery_alignment),
        _metric("cross_replay_drift_score", cross_replay_drift_score),
        _metric("drift_reconciliation_confidence", drift_reconciliation_confidence),
    )


def _compute_advisory(metrics: Sequence[CalibrationDriftMetric]) -> Tuple[str, ...]:
    by_name = {m.metric_name: float(m.metric_value) for m in metrics}
    threshold_drift_delta = _clamp01(by_name.get("threshold_drift_delta", 1.0))
    continuity_drift_pressure = _clamp01(by_name.get("continuity_drift_pressure", 1.0))
    confidence_reconciliation_score = _clamp01(by_name.get("confidence_reconciliation_score", 0.0))
    stability_recovery_alignment = _clamp01(by_name.get("stability_recovery_alignment", 0.0))
    cross_replay_drift_score = _clamp01(by_name.get("cross_replay_drift_score", 1.0))

    composite_pressure = (
        threshold_drift_delta
        + continuity_drift_pressure
        + cross_replay_drift_score
        + (1.0 - confidence_reconciliation_score)
        + (1.0 - stability_recovery_alignment)
    ) / 5.0

    if composite_pressure <= 0.05:
        label = "stable_alignment"
    elif composite_pressure <= 0.20:
        label = "minor_drift_reconcile"
    elif composite_pressure <= 0.50:
        label = "moderate_drift_reconcile"
    else:
        label = "severe_drift_reconcile"
    return (label,)


def _metrics_hash(metrics: Sequence[CalibrationDriftMetric]) -> str:
    payload = [metric.to_dict() for metric in metrics]
    return _sha256_hex(_canonical_json(payload).encode("utf-8"))


def _advisory_hash(advisory: Sequence[str]) -> str:
    return _sha256_hex(_canonical_json(list(advisory)).encode("utf-8"))


def _reconciliation_hash(
    *,
    scenario: CalibrationDriftScenario,
    metrics: Sequence[CalibrationDriftMetric],
    advisory: Sequence[str],
    violations: Sequence[str],
) -> str:
    body = {
        "scenario": scenario.to_dict(),
        "metrics": [metric.to_dict() for metric in metrics],
        "advisory": list(advisory),
        "violations": list(violations),
    }
    return _sha256_hex(_canonical_json(body).encode("utf-8"))


def validate_calibration_drift_reconciliation(kernel: Any) -> Tuple[str, ...]:
    violations: list[str] = []
    try:
        scenario = _field(kernel, "scenario", None)
        metrics = tuple(_safe_series(_field(kernel, "metrics", ())))
        advisory = tuple(_safe_series(_field(kernel, "advisory", ())))
        receipt = _field(kernel, "receipt", None)

        if scenario is None:
            violations.append("missing_scenario")
            return tuple(sorted(set(violations)))

        if not _safe_text(_field(scenario, "scenario_id", "")).strip():
            violations.append("empty_scenario_id")

        calibration_series = _safe_series(_field(scenario, "calibration_series", ()))
        prior_calibration_series = _safe_series(_field(scenario, "prior_calibration_series", ()))

        if not calibration_series:
            violations.append("empty_calibration_series")
        if not prior_calibration_series:
            violations.append("empty_prior_calibration_series")

        for idx, row in enumerate(calibration_series):
            calibration_id = _safe_text(_field(row, "calibration_id", "")).strip()
            if not calibration_id:
                violations.append(f"malformed_calibration_row:{idx}")

        for idx, row in enumerate(prior_calibration_series):
            calibration_id = _safe_text(_field(row, "calibration_id", "")).strip()
            if not calibration_id:
                violations.append(f"malformed_prior_calibration_row:{idx}")

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

        if not advisory:
            violations.append("empty_advisory")
        else:
            for label in advisory:
                if _safe_text(label) not in _ADVISORY_ORDER:
                    violations.append(f"unknown_advisory_label:{_safe_text(label)}")

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

            actual_advisory_hash = _safe_text(_field(receipt, "advisory_hash", "")).strip()
            if actual_advisory_hash:
                expected_advisory_hash = _advisory_hash(advisory)
                if expected_advisory_hash != actual_advisory_hash:
                    violations.append("receipt_advisory_hash_mismatch")

            actual_reconciliation_hash = _safe_text(_field(receipt, "reconciliation_hash", "")).strip()
            expected_reconciliation_hash = _safe_text(_field(kernel, "reconciliation_hash", "")).strip()
            if actual_reconciliation_hash and expected_reconciliation_hash:
                if actual_reconciliation_hash != expected_reconciliation_hash:
                    violations.append("receipt_reconciliation_hash_mismatch")

            actual_receipt_hash = _safe_text(_field(receipt, "receipt_hash", "")).strip()
            if actual_receipt_hash:
                confidence_raw = _field(receipt, "drift_reconciliation_confidence", 0.0)
                try:
                    confidence_value = float(confidence_raw)
                except Exception:
                    confidence_value = 0.0
                expected_receipt_payload = {
                    "scenario_hash": _safe_text(_field(receipt, "scenario_hash", "")).strip(),
                    "metrics_hash": _safe_text(_field(receipt, "metrics_hash", "")).strip(),
                    "advisory_hash": _safe_text(_field(receipt, "advisory_hash", "")).strip(),
                    "reconciliation_hash": _safe_text(_field(receipt, "reconciliation_hash", "")).strip(),
                    "drift_reconciliation_confidence": confidence_value,
                }
                expected_receipt_hash = _sha256_hex(_canonical_json(expected_receipt_payload).encode("utf-8"))
                if expected_receipt_hash != actual_receipt_hash:
                    violations.append("receipt_hash_mismatch")
    except Exception as exc:
        violations.append(f"validator_internal_error:{_safe_text(exc)}")

    return tuple(sorted(set(violations)))


def build_calibration_drift_receipt(
    *,
    scenario: CalibrationDriftScenario,
    metrics: Sequence[CalibrationDriftMetric],
    advisory: Sequence[str],
    reconciliation_hash: str,
) -> CalibrationDriftReceipt:
    drift_reconciliation_confidence = 0.0
    for metric in metrics:
        if metric.metric_name == "drift_reconciliation_confidence":
            drift_reconciliation_confidence = float(metric.metric_value)
            break

    payload = {
        "scenario_hash": scenario.stable_hash(),
        "metrics_hash": _metrics_hash(metrics),
        "advisory_hash": _advisory_hash(advisory),
        "reconciliation_hash": _safe_text(reconciliation_hash).strip(),
        "drift_reconciliation_confidence": drift_reconciliation_confidence,
    }
    receipt_hash = _sha256_hex(_canonical_json(payload).encode("utf-8"))
    return CalibrationDriftReceipt(
        scenario_hash=payload["scenario_hash"],
        metrics_hash=payload["metrics_hash"],
        advisory_hash=payload["advisory_hash"],
        reconciliation_hash=payload["reconciliation_hash"],
        drift_reconciliation_confidence=drift_reconciliation_confidence,
        receipt_hash=receipt_hash,
    )


def run_governance_calibration_drift_reconciliation(
    *,
    scenario: CalibrationDriftScenario,
) -> GovernanceCalibrationDriftReconciliationKernel:
    metrics = _compute_metrics(scenario)
    advisory = _compute_advisory(metrics)
    provisional = GovernanceCalibrationDriftReconciliationKernel(
        scenario=scenario,
        metrics=metrics,
        advisory=advisory,
        violations=(),
        receipt=CalibrationDriftReceipt("", "", "", "", 0.0, ""),
        reconciliation_hash="",
    )
    violations = validate_calibration_drift_reconciliation(provisional)
    reconciliation_hash = _reconciliation_hash(
        scenario=scenario,
        metrics=metrics,
        advisory=advisory,
        violations=violations,
    )
    receipt = build_calibration_drift_receipt(
        scenario=scenario,
        metrics=metrics,
        advisory=advisory,
        reconciliation_hash=reconciliation_hash,
    )
    return GovernanceCalibrationDriftReconciliationKernel(
        scenario=scenario,
        metrics=metrics,
        advisory=advisory,
        violations=violations,
        receipt=receipt,
        reconciliation_hash=reconciliation_hash,
    )


def compare_calibration_drift_replay(
    baseline: GovernanceCalibrationDriftReconciliationKernel,
    replay: GovernanceCalibrationDriftReconciliationKernel,
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
    if tuple(baseline.advisory) != tuple(replay.advisory):
        mismatches.append("advisory")

    return {
        "is_stable_replay": len(mismatches) == 0,
        "mismatches": tuple(mismatches),
        "metric_delta": tuple((name, metric_delta[name]) for name in _METRIC_ORDER),
        "baseline_hash": baseline.stable_hash(),
        "replay_hash": replay.stable_hash(),
    }


def summarize_calibration_drift_reconciliation(
    kernel: GovernanceCalibrationDriftReconciliationKernel,
) -> str:
    lines = [
        f"scenario_id={kernel.scenario.scenario_id}",
        f"scenario_hash={kernel.scenario.stable_hash()}",
        f"reconciliation_hash={kernel.reconciliation_hash}",
        f"receipt_hash={kernel.receipt.receipt_hash}",
        "metrics:",
    ]
    for metric in kernel.metrics:
        lines.append(f"- {metric.metric_order}:{metric.metric_name}={metric.metric_value:.12f}")
    lines.append("advisory:")
    for label in kernel.advisory:
        lines.append(f"- {label}")
    if kernel.violations:
        lines.append("violations:")
        for item in kernel.violations:
            lines.append(f"- {item}")
    else:
        lines.append("violations: none")
    return "\n".join(lines)


__all__ = [
    "CalibrationDriftScenario",
    "CalibrationDriftMetric",
    "CalibrationDriftReceipt",
    "GovernanceCalibrationDriftReconciliationKernel",
    "build_calibration_drift_scenario",
    "run_governance_calibration_drift_reconciliation",
    "validate_calibration_drift_reconciliation",
    "build_calibration_drift_receipt",
    "compare_calibration_drift_replay",
    "summarize_calibration_drift_reconciliation",
]
