"""v137.20.3 — Governance Drift Topology Stability Kernel.

Additive deterministic topology-analysis layer that converts drift
reconciliation outputs into a bounded stability topology across replay
horizons. Advisory-only. No decoder touch.

Canonical model:
- drift_reconciliation_series + replay_horizon_series
  -> topology_stability_analysis + topology_stability_receipt

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
    "topology_coherence_score",
    "horizon_stability_gradient",
    "drift_surface_pressure",
    "continuity_topology_alignment",
    "cross_horizon_stability_score",
    "topology_confidence_score",
)
_METRIC_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(_METRIC_ORDER)}

_ADVISORY_ORDER: Tuple[str, ...] = (
    "stable_topology",
    "minor_topology_variation",
    "moderate_topology_instability",
    "severe_topology_instability",
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
class _DriftReconciliationNode:
    reconciliation_id: str
    drift_reconciliation_confidence: float
    threshold_drift_delta: float
    continuity_drift_pressure: float
    stability_recovery_alignment: float
    cross_replay_drift_score: float
    advisory_label: str
    replay_identity: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reconciliation_id": self.reconciliation_id,
            "drift_reconciliation_confidence": self.drift_reconciliation_confidence,
            "threshold_drift_delta": self.threshold_drift_delta,
            "continuity_drift_pressure": self.continuity_drift_pressure,
            "stability_recovery_alignment": self.stability_recovery_alignment,
            "cross_replay_drift_score": self.cross_replay_drift_score,
            "advisory_label": self.advisory_label,
            "replay_identity": self.replay_identity,
        }


@dataclass(frozen=True)
class _HorizonNode:
    horizon_id: str
    horizon_index: int
    continuity_flag: bool
    replay_identity: str
    weight: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizon_id": self.horizon_id,
            "horizon_index": self.horizon_index,
            "continuity_flag": self.continuity_flag,
            "replay_identity": self.replay_identity,
            "weight": self.weight,
        }


def _normalize_drift_node(raw: Any, index: int) -> _DriftReconciliationNode:
    reconciliation_id = (
        _safe_text(_field(raw, "reconciliation_id", "")).strip()
        or _safe_text(_field(raw, "scenario_id", "")).strip()
        or f"drift_{index}"
    )
    confidence = _clamp01(
        _safe_finite_float(_field(raw, "drift_reconciliation_confidence", 0.0))
    )
    threshold_drift_delta = _clamp01(
        _safe_finite_float(_field(raw, "threshold_drift_delta", 0.0))
    )
    continuity_drift_pressure = _clamp01(
        _safe_finite_float(_field(raw, "continuity_drift_pressure", 0.0))
    )
    stability_recovery_alignment = _clamp01(
        _safe_finite_float(_field(raw, "stability_recovery_alignment", 0.0))
    )
    cross_replay_drift_score = _clamp01(
        _safe_finite_float(_field(raw, "cross_replay_drift_score", 0.0))
    )
    advisory_label = _safe_text(_field(raw, "advisory_label", "")).strip()
    replay_identity = _safe_text(_field(raw, "replay_identity", "")).strip()
    return _DriftReconciliationNode(
        reconciliation_id=reconciliation_id,
        drift_reconciliation_confidence=confidence,
        threshold_drift_delta=threshold_drift_delta,
        continuity_drift_pressure=continuity_drift_pressure,
        stability_recovery_alignment=stability_recovery_alignment,
        cross_replay_drift_score=cross_replay_drift_score,
        advisory_label=advisory_label,
        replay_identity=replay_identity,
    )


def _normalize_horizon_node(raw: Any, index: int) -> _HorizonNode:
    horizon_id = (
        _safe_text(_field(raw, "horizon_id", "")).strip() or f"horizon_{index}"
    )
    try:
        horizon_index_raw = int(_field(raw, "horizon_index", index))
    except Exception:
        horizon_index_raw = index
    if horizon_index_raw < 0:
        horizon_index_raw = 0
    continuity_flag = _safe_bool(_field(raw, "continuity_flag", False))
    replay_identity = _safe_text(_field(raw, "replay_identity", "")).strip()
    weight = _clamp01(_safe_finite_float(_field(raw, "weight", 1.0)))
    return _HorizonNode(
        horizon_id=horizon_id,
        horizon_index=horizon_index_raw,
        continuity_flag=continuity_flag,
        replay_identity=replay_identity,
        weight=weight,
    )


def _ordered_drift_nodes(series: Sequence[Any]) -> Tuple[_DriftReconciliationNode, ...]:
    return tuple(_normalize_drift_node(raw, index=i) for i, raw in enumerate(series))


def _ordered_horizon_nodes(series: Sequence[Any]) -> Tuple[_HorizonNode, ...]:
    return tuple(_normalize_horizon_node(raw, index=i) for i, raw in enumerate(series))


@dataclass(frozen=True)
class DriftTopologyScenario:
    scenario_id: str
    drift_reconciliation_series: Tuple[Mapping[str, Any], ...]
    replay_horizon_series: Tuple[Mapping[str, Any], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "drift_reconciliation_series": [dict(item) for item in self.drift_reconciliation_series],
            "replay_horizon_series": [dict(item) for item in self.replay_horizon_series],
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_canonical_json().encode("utf-8"))


@dataclass(frozen=True)
class DriftTopologyMetric:
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
class DriftTopologyReceipt:
    scenario_hash: str
    metrics_hash: str
    advisory_hash: str
    topology_hash: str
    topology_confidence_score: float
    receipt_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_hash": self.scenario_hash,
            "metrics_hash": self.metrics_hash,
            "advisory_hash": self.advisory_hash,
            "topology_hash": self.topology_hash,
            "topology_confidence_score": self.topology_confidence_score,
            "receipt_hash": self.receipt_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.receipt_hash


@dataclass(frozen=True)
class GovernanceDriftTopologyStabilityKernel:
    scenario: DriftTopologyScenario
    metrics: Tuple[DriftTopologyMetric, ...]
    advisory: Tuple[str, ...]
    violations: Tuple[str, ...]
    receipt: DriftTopologyReceipt
    topology_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario.to_dict(),
            "metrics": [metric.to_dict() for metric in self.metrics],
            "advisory": list(self.advisory),
            "violations": list(self.violations),
            "receipt": self.receipt.to_dict(),
            "topology_hash": self.topology_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.topology_hash


def build_drift_topology_scenario(
    *,
    scenario_id: str,
    drift_reconciliation_series: Any,
    replay_horizon_series: Any,
) -> DriftTopologyScenario:
    normalized_id = _safe_text(scenario_id).strip()
    normalized_drifts = _ordered_drift_nodes(_safe_series(drift_reconciliation_series))
    normalized_horizons = _ordered_horizon_nodes(_safe_series(replay_horizon_series))
    return DriftTopologyScenario(
        scenario_id=normalized_id,
        drift_reconciliation_series=tuple(node.to_dict() for node in normalized_drifts),
        replay_horizon_series=tuple(node.to_dict() for node in normalized_horizons),
    )


def _metric(name: str, value: float) -> DriftTopologyMetric:
    return DriftTopologyMetric(
        metric_name=name,
        metric_order=_METRIC_INDEX[name],
        metric_value=float(value),
    )


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def _std_dev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((v - mean) * (v - mean) for v in values) / float(len(values))
    return math.sqrt(variance)


def _compute_metrics(scenario: DriftTopologyScenario) -> Tuple[DriftTopologyMetric, ...]:
    drift_nodes = _ordered_drift_nodes(scenario.drift_reconciliation_series)
    horizon_nodes = _ordered_horizon_nodes(scenario.replay_horizon_series)

    # Horizons traversed in fixed deterministic order by (horizon_index, horizon_id).
    ordered_horizons = tuple(
        sorted(
            horizon_nodes,
            key=lambda n: (n.horizon_index, n.horizon_id),
        )
    )

    confidences: Tuple[float, ...] = tuple(
        node.drift_reconciliation_confidence for node in drift_nodes
    )

    # Topology coherence: 1 - spread of confidences (max - min).
    if confidences:
        spread = max(confidences) - min(confidences)
        topology_coherence_score = _clamp01(1.0 - spread)
    else:
        topology_coherence_score = 0.0

    # Horizon stability gradient: 1 - mean consecutive absolute difference.
    if len(confidences) >= 2:
        diffs = [
            abs(confidences[i + 1] - confidences[i])
            for i in range(len(confidences) - 1)
        ]
        horizon_stability_gradient = _clamp01(1.0 - _mean(diffs))
    elif len(confidences) == 1:
        horizon_stability_gradient = 1.0
    else:
        horizon_stability_gradient = 0.0

    # Drift surface pressure: average of per-node pressures.
    if drift_nodes:
        pressures = []
        for node in drift_nodes:
            node_pressure = (
                node.threshold_drift_delta
                + node.continuity_drift_pressure
                + node.cross_replay_drift_score
                + (1.0 - node.drift_reconciliation_confidence)
                + (1.0 - node.stability_recovery_alignment)
            ) / 5.0
            pressures.append(_clamp01(node_pressure))
        drift_surface_pressure = _clamp01(_mean(pressures))
    else:
        drift_surface_pressure = 1.0

    # Continuity topology alignment: fraction of horizons with continuity_flag.
    if ordered_horizons:
        continuity_true = sum(1 for node in ordered_horizons if node.continuity_flag)
        continuity_topology_alignment = _clamp01(
            float(continuity_true) / float(len(ordered_horizons))
        )
    else:
        continuity_topology_alignment = 0.0

    # Cross horizon stability: 1 - std-dev of confidences across horizons.
    cross_horizon_stability_score = _clamp01(1.0 - _std_dev(confidences))

    # Composite topology confidence.
    composite_pressure = (
        (1.0 - topology_coherence_score)
        + (1.0 - horizon_stability_gradient)
        + drift_surface_pressure
        + (1.0 - continuity_topology_alignment)
        + (1.0 - cross_horizon_stability_score)
    ) / 5.0
    topology_confidence_score = _clamp01(1.0 - composite_pressure)

    return (
        _metric("topology_coherence_score", topology_coherence_score),
        _metric("horizon_stability_gradient", horizon_stability_gradient),
        _metric("drift_surface_pressure", drift_surface_pressure),
        _metric("continuity_topology_alignment", continuity_topology_alignment),
        _metric("cross_horizon_stability_score", cross_horizon_stability_score),
        _metric("topology_confidence_score", topology_confidence_score),
    )


def _composite_topology_pressure(metrics: Sequence[DriftTopologyMetric]) -> float:
    by_name = {m.metric_name: float(m.metric_value) for m in metrics}
    topology_coherence = _clamp01(by_name.get("topology_coherence_score", 0.0))
    horizon_gradient = _clamp01(by_name.get("horizon_stability_gradient", 0.0))
    surface_pressure = _clamp01(by_name.get("drift_surface_pressure", 1.0))
    continuity_alignment = _clamp01(by_name.get("continuity_topology_alignment", 0.0))
    cross_horizon = _clamp01(by_name.get("cross_horizon_stability_score", 0.0))

    return _clamp01(
        (
            (1.0 - topology_coherence)
            + (1.0 - horizon_gradient)
            + surface_pressure
            + (1.0 - continuity_alignment)
            + (1.0 - cross_horizon)
        )
        / 5.0
    )


def _compute_advisory(metrics: Sequence[DriftTopologyMetric]) -> Tuple[str, ...]:
    pressure = _composite_topology_pressure(metrics)
    if pressure <= 0.05:
        label = "stable_topology"
    elif pressure <= 0.20:
        label = "minor_topology_variation"
    elif pressure <= 0.50:
        label = "moderate_topology_instability"
    else:
        label = "severe_topology_instability"
    return (label,)


def _metrics_hash(metrics: Sequence[DriftTopologyMetric]) -> str:
    payload = [metric.to_dict() for metric in metrics]
    return _sha256_hex(_canonical_json(payload).encode("utf-8"))


def _advisory_hash(advisory: Sequence[str]) -> str:
    return _sha256_hex(_canonical_json(list(advisory)).encode("utf-8"))


def _topology_hash(
    *,
    scenario: DriftTopologyScenario,
    metrics: Sequence[DriftTopologyMetric],
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


def validate_drift_topology_stability(kernel: Any) -> Tuple[str, ...]:
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

        drift_series = _safe_series(_field(scenario, "drift_reconciliation_series", ()))
        horizon_series = _safe_series(_field(scenario, "replay_horizon_series", ()))

        if not drift_series:
            violations.append("empty_drift_reconciliation_series")
        if not horizon_series:
            violations.append("empty_replay_horizon_series")

        for idx, row in enumerate(drift_series):
            reconciliation_id = _safe_text(_field(row, "reconciliation_id", "")).strip()
            if not reconciliation_id:
                violations.append(f"malformed_drift_row:{idx}")

        for idx, row in enumerate(horizon_series):
            horizon_id = _safe_text(_field(row, "horizon_id", "")).strip()
            if not horizon_id:
                violations.append(f"malformed_horizon_row:{idx}")

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
                expected_metrics_hash = _metrics_hash(
                    tuple(m for m in metrics if isinstance(m, DriftTopologyMetric))
                )
                if expected_metrics_hash != actual_metrics_hash:
                    violations.append("receipt_metrics_hash_mismatch")

            actual_scenario_hash = _safe_text(_field(receipt, "scenario_hash", "")).strip()
            if actual_scenario_hash and isinstance(scenario, DriftTopologyScenario):
                expected_scenario_hash = scenario.stable_hash()
                if expected_scenario_hash != actual_scenario_hash:
                    violations.append("receipt_scenario_hash_mismatch")

            actual_advisory_hash = _safe_text(_field(receipt, "advisory_hash", "")).strip()
            if actual_advisory_hash:
                expected_advisory_hash = _advisory_hash(advisory)
                if expected_advisory_hash != actual_advisory_hash:
                    violations.append("receipt_advisory_hash_mismatch")

            actual_topology_hash = _safe_text(_field(receipt, "topology_hash", "")).strip()
            expected_topology_hash = _safe_text(_field(kernel, "topology_hash", "")).strip()
            if actual_topology_hash and expected_topology_hash:
                if actual_topology_hash != expected_topology_hash:
                    violations.append("receipt_topology_hash_mismatch")

            actual_receipt_hash = _safe_text(_field(receipt, "receipt_hash", "")).strip()
            if actual_receipt_hash:
                confidence_raw = _field(receipt, "topology_confidence_score", 0.0)
                try:
                    confidence_value = float(confidence_raw)
                except Exception:
                    confidence_value = 0.0
                expected_receipt_payload = {
                    "scenario_hash": _safe_text(_field(receipt, "scenario_hash", "")).strip(),
                    "metrics_hash": _safe_text(_field(receipt, "metrics_hash", "")).strip(),
                    "advisory_hash": _safe_text(_field(receipt, "advisory_hash", "")).strip(),
                    "topology_hash": _safe_text(_field(receipt, "topology_hash", "")).strip(),
                    "topology_confidence_score": confidence_value,
                }
                expected_receipt_hash = _sha256_hex(
                    _canonical_json(expected_receipt_payload).encode("utf-8")
                )
                if expected_receipt_hash != actual_receipt_hash:
                    violations.append("receipt_hash_mismatch")
    except Exception as exc:
        violations.append(f"validator_internal_error:{_safe_text(exc)}")

    return tuple(sorted(set(violations)))


def build_drift_topology_receipt(
    *,
    scenario: DriftTopologyScenario,
    metrics: Sequence[DriftTopologyMetric],
    advisory: Sequence[str],
    topology_hash: str,
) -> DriftTopologyReceipt:
    topology_confidence_score = 0.0
    for metric in metrics:
        if metric.metric_name == "topology_confidence_score":
            topology_confidence_score = float(metric.metric_value)
            break

    payload = {
        "scenario_hash": scenario.stable_hash(),
        "metrics_hash": _metrics_hash(metrics),
        "advisory_hash": _advisory_hash(advisory),
        "topology_hash": _safe_text(topology_hash).strip(),
        "topology_confidence_score": topology_confidence_score,
    }
    receipt_hash = _sha256_hex(_canonical_json(payload).encode("utf-8"))
    return DriftTopologyReceipt(
        scenario_hash=payload["scenario_hash"],
        metrics_hash=payload["metrics_hash"],
        advisory_hash=payload["advisory_hash"],
        topology_hash=payload["topology_hash"],
        topology_confidence_score=topology_confidence_score,
        receipt_hash=receipt_hash,
    )


def run_governance_drift_topology_stability(
    *,
    scenario: DriftTopologyScenario,
) -> GovernanceDriftTopologyStabilityKernel:
    metrics = _compute_metrics(scenario)
    advisory = _compute_advisory(metrics)
    provisional = GovernanceDriftTopologyStabilityKernel(
        scenario=scenario,
        metrics=metrics,
        advisory=advisory,
        violations=(),
        receipt=DriftTopologyReceipt("", "", "", "", 0.0, ""),
        topology_hash="",
    )
    violations = validate_drift_topology_stability(provisional)
    topology_hash = _topology_hash(
        scenario=scenario,
        metrics=metrics,
        advisory=advisory,
        violations=violations,
    )
    receipt = build_drift_topology_receipt(
        scenario=scenario,
        metrics=metrics,
        advisory=advisory,
        topology_hash=topology_hash,
    )
    return GovernanceDriftTopologyStabilityKernel(
        scenario=scenario,
        metrics=metrics,
        advisory=advisory,
        violations=violations,
        receipt=receipt,
        topology_hash=topology_hash,
    )


def compare_drift_topology_replay(
    baseline: GovernanceDriftTopologyStabilityKernel,
    replay: GovernanceDriftTopologyStabilityKernel,
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
    if baseline.topology_hash != replay.topology_hash:
        mismatches.append("topology_hash")
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


def summarize_drift_topology_stability(
    kernel: GovernanceDriftTopologyStabilityKernel,
) -> str:
    lines = [
        f"scenario_id={kernel.scenario.scenario_id}",
        f"scenario_hash={kernel.scenario.stable_hash()}",
        f"topology_hash={kernel.topology_hash}",
        f"receipt_hash={kernel.receipt.receipt_hash}",
        "metrics:",
    ]
    for metric in kernel.metrics:
        lines.append(
            f"- {metric.metric_order}:{metric.metric_name}={metric.metric_value:.12f}"
        )
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
    "DriftTopologyScenario",
    "DriftTopologyMetric",
    "DriftTopologyReceipt",
    "GovernanceDriftTopologyStabilityKernel",
    "build_drift_topology_scenario",
    "run_governance_drift_topology_stability",
    "validate_drift_topology_stability",
    "build_drift_topology_receipt",
    "compare_drift_topology_replay",
    "summarize_drift_topology_stability",
]
