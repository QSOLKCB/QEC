"""v137.19.3 — Governance Topology Evolution Kernel.

Additive deterministic temporal-evolution analysis over governance stability
Topology artifacts.

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
    "topology_change_rate",
    "basin_persistence_score",
    "evolution_transition_density",
    "topology_drift_velocity",
    "continuity_decay_score",
    "severity_evolution_score",
    "replay_evolution_stability",
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
class _TopologyNode:
    topology_id: str
    basin_id: str
    continuity_ok: bool
    severity: float
    replay_identity: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topology_id": self.topology_id,
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
class TopologyEvolutionScenario:
    scenario_id: str
    topology_series: Tuple[Mapping[str, Any], ...]
    drift_series: Tuple[Mapping[str, Any], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "topology_series": [dict(item) for item in self.topology_series],
            "drift_series": [dict(item) for item in self.drift_series],
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_canonical_json().encode("utf-8"))


@dataclass(frozen=True)
class TopologyEvolutionMetric:
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
class TopologyEvolutionReceipt:
    scenario_hash: str
    metrics_hash: str
    evolution_hash: str
    severity_evolution_score: float
    receipt_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_hash": self.scenario_hash,
            "metrics_hash": self.metrics_hash,
            "evolution_hash": self.evolution_hash,
            "severity_evolution_score": self.severity_evolution_score,
            "receipt_hash": self.receipt_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.receipt_hash


@dataclass(frozen=True)
class GovernanceTopologyEvolutionKernel:
    scenario: TopologyEvolutionScenario
    metrics: Tuple[TopologyEvolutionMetric, ...]
    violations: Tuple[str, ...]
    receipt: TopologyEvolutionReceipt
    evolution_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario.to_dict(),
            "metrics": [metric.to_dict() for metric in self.metrics],
            "violations": list(self.violations),
            "receipt": self.receipt.to_dict(),
            "evolution_hash": self.evolution_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.evolution_hash


def _normalize_topology_node(raw: Any, index: int) -> _TopologyNode:
    topology_id = _safe_text(_field(raw, "topology_id", "")).strip() or f"topology_{index}"
    basin_id = _safe_text(_field(raw, "basin_id", _field(raw, "decision_basin", ""))).strip() or "unknown"
    continuity_ok = _safe_bool(_field(raw, "continuity_ok", False))
    severity = _safe_nonneg_float(_field(raw, "severity", _field(raw, "topology_severity", 0.0)))
    replay_identity = _safe_text(_field(raw, "replay_identity", "")).strip()
    return _TopologyNode(
        topology_id=topology_id,
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


def _ordered_topology_nodes(series: Sequence[Any]) -> Tuple[_TopologyNode, ...]:
    """Normalize topology nodes preserving caller-provided order (for temporal metrics)."""
    return tuple(_normalize_topology_node(raw, index=i) for i, raw in enumerate(series))


def _ordered_drift_nodes(series: Sequence[Any]) -> Tuple[_DriftNode, ...]:
    """Normalize drift nodes preserving caller-provided order (for temporal metrics)."""
    return tuple(_normalize_drift_node(raw, index=i) for i, raw in enumerate(series))


def build_topology_evolution_scenario(
    *,
    scenario_id: str,
    topology_series: Any,
    drift_series: Any,
) -> TopologyEvolutionScenario:
    normalized_id = _safe_text(scenario_id).strip()
    # Preserve caller-provided order: temporal sequence must not be reordered.
    normalized_topology = _ordered_topology_nodes(_safe_series(topology_series))
    normalized_drift = _ordered_drift_nodes(_safe_series(drift_series))
    return TopologyEvolutionScenario(
        scenario_id=normalized_id,
        topology_series=tuple(node.to_dict() for node in normalized_topology),
        drift_series=tuple(node.to_dict() for node in normalized_drift),
    )


def _metric(name: str, value: float) -> TopologyEvolutionMetric:
    return TopologyEvolutionMetric(
        metric_name=name,
        metric_order=_METRIC_INDEX[name],
        metric_value=float(value),
    )


def _compute_metrics(scenario: TopologyEvolutionScenario) -> Tuple[TopologyEvolutionMetric, ...]:
    # Preserve temporal order for all adjacent-pair metrics; canonical sorting
    # is confined to hashing/serialization paths only.
    topologies = _ordered_topology_nodes(scenario.topology_series)
    drifts = _ordered_drift_nodes(scenario.drift_series)

    topology_count = len(topologies)
    drift_count = len(drifts)

    if topology_count >= 2:
        changes = 0
        stable_replay_pairs = 0
        for i in range(1, topology_count):
            prev = topologies[i - 1]
            curr = topologies[i]
            if prev.basin_id != curr.basin_id:
                changes += 1
            if prev.replay_identity and prev.replay_identity == curr.replay_identity:
                stable_replay_pairs += 1
        topology_change_rate = float(changes) / float(topology_count - 1)
        replay_evolution_stability = float(stable_replay_pairs) / float(topology_count - 1)
    else:
        topology_change_rate = 0.0
        replay_evolution_stability = 1.0

    basin_counts: Dict[str, int] = {}
    continuity_true = 0
    severity_sum = 0.0
    for node in topologies:
        basin_counts[node.basin_id] = basin_counts.get(node.basin_id, 0) + 1
        if node.continuity_ok:
            continuity_true += 1
        severity_sum += node.severity

    dominant = max(basin_counts.values()) if basin_counts else 0
    basin_persistence_score = 0.0 if topology_count == 0 else float(dominant) / float(topology_count)

    total_transitions = sum(node.transition_count for node in drifts)
    total_magnitude = sum(node.drift_magnitude for node in drifts)
    possible_transition_edges = max(1, topology_count - 1)
    evolution_transition_density = float(total_transitions) / float(possible_transition_edges)
    topology_drift_velocity = 0.0 if drift_count == 0 else float(total_magnitude) / float(drift_count)

    continuity_decay_score = (
        0.0 if topology_count == 0 else 1.0 - (float(continuity_true) / float(topology_count))
    )
    avg_severity = 0.0 if topology_count == 0 else severity_sum / float(topology_count)
    severity_evolution_score = _clamp01(
        0.4 * _clamp01(avg_severity)
        + 0.2 * _clamp01(topology_change_rate)
        + 0.2 * _clamp01(continuity_decay_score)
        + 0.2 * _clamp01(evolution_transition_density)
    )

    metrics = (
        _metric("topology_change_rate", topology_change_rate),
        _metric("basin_persistence_score", basin_persistence_score),
        _metric("evolution_transition_density", evolution_transition_density),
        _metric("topology_drift_velocity", topology_drift_velocity),
        _metric("continuity_decay_score", continuity_decay_score),
        _metric("severity_evolution_score", severity_evolution_score),
        _metric("replay_evolution_stability", replay_evolution_stability),
    )
    return metrics


def _metrics_hash(metrics: Sequence[TopologyEvolutionMetric]) -> str:
    payload = [metric.to_dict() for metric in metrics]
    return _sha256_hex(_canonical_json(payload).encode("utf-8"))


def _evolution_hash(
    *,
    scenario: TopologyEvolutionScenario,
    metrics: Sequence[TopologyEvolutionMetric],
    violations: Sequence[str],
) -> str:
    body = {
        "scenario": scenario.to_dict(),
        "metrics": [metric.to_dict() for metric in metrics],
        "violations": list(violations),
    }
    return _sha256_hex(_canonical_json(body).encode("utf-8"))


def validate_topology_evolution(
    kernel: Any,
) -> Tuple[str, ...]:
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

        topology_series = _safe_series(_field(scenario, "topology_series", ()))
        drift_series = _safe_series(_field(scenario, "drift_series", ()))

        if not topology_series:
            violations.append("empty_topology_series")

        for idx, row in enumerate(topology_series):
            basin = _safe_text(_field(row, "basin_id", "")).strip()
            if not basin:
                violations.append(f"malformed_topology_row:{idx}")

        for idx, row in enumerate(drift_series):
            from_basin = _safe_text(_field(row, "from_basin", "")).strip()
            to_basin = _safe_text(_field(row, "to_basin", "")).strip()
            if not from_basin or not to_basin:
                violations.append(f"malformed_drift_row:{idx}")

        names = tuple(_safe_text(_field(m, "metric_name", "")) for m in metrics)
        if names != _METRIC_ORDER:
            violations.append("metric_order_mismatch")

        for metric in metrics:
            value = _field(metric, "metric_value", 0.0)
            try:
                value_f = float(value)
            except Exception:
                violations.append(f"metric_non_numeric:{_safe_text(_field(metric, 'metric_name', '?'))}")
                continue
            if not math.isfinite(value_f):
                violations.append(f"metric_non_finite:{_safe_text(_field(metric, 'metric_name', '?'))}")

        score_fields = (
            "topology_change_rate",
            "basin_persistence_score",
            "continuity_decay_score",
            "severity_evolution_score",
            "replay_evolution_stability",
        )
        by_name: Dict[str, float] = {}
        for metric in metrics:
            key = _safe_text(_field(metric, "metric_name", ""))
            if key:
                try:
                    by_name[key] = float(_field(metric, "metric_value", 0.0))
                except Exception:
                    by_name[key] = -1.0
        for key in score_fields:
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


def build_topology_evolution_receipt(
    *,
    scenario: TopologyEvolutionScenario,
    metrics: Sequence[TopologyEvolutionMetric],
    evolution_hash: str,
) -> TopologyEvolutionReceipt:
    severity = 0.0
    for metric in metrics:
        if metric.metric_name == "severity_evolution_score":
            severity = float(metric.metric_value)
            break
    payload = {
        "scenario_hash": scenario.stable_hash(),
        "metrics_hash": _metrics_hash(metrics),
        "evolution_hash": _safe_text(evolution_hash).strip(),
        "severity_evolution_score": severity,
    }
    receipt_hash = _sha256_hex(_canonical_json(payload).encode("utf-8"))
    return TopologyEvolutionReceipt(
        scenario_hash=payload["scenario_hash"],
        metrics_hash=payload["metrics_hash"],
        evolution_hash=payload["evolution_hash"],
        severity_evolution_score=severity,
        receipt_hash=receipt_hash,
    )


def run_governance_topology_evolution(
    *,
    scenario: TopologyEvolutionScenario,
) -> GovernanceTopologyEvolutionKernel:
    metrics = _compute_metrics(scenario)
    provisional = GovernanceTopologyEvolutionKernel(
        scenario=scenario,
        metrics=metrics,
        violations=(),
        receipt=TopologyEvolutionReceipt("", "", "", 0.0, ""),
        evolution_hash="",
    )
    violations = validate_topology_evolution(provisional)
    evolution_hash = _evolution_hash(scenario=scenario, metrics=metrics, violations=violations)
    receipt = build_topology_evolution_receipt(
        scenario=scenario,
        metrics=metrics,
        evolution_hash=evolution_hash,
    )
    kernel = GovernanceTopologyEvolutionKernel(
        scenario=scenario,
        metrics=metrics,
        violations=violations,
        receipt=receipt,
        evolution_hash=evolution_hash,
    )
    return kernel


def compare_topology_evolution_replay(
    baseline: GovernanceTopologyEvolutionKernel,
    replay: GovernanceTopologyEvolutionKernel,
) -> Dict[str, Any]:
    metric_delta: Dict[str, float] = {}
    baseline_by_name = {m.metric_name: m.metric_value for m in baseline.metrics}
    replay_by_name = {m.metric_name: m.metric_value for m in replay.metrics}
    for name in _METRIC_ORDER:
        metric_delta[name] = float(replay_by_name.get(name, 0.0) - baseline_by_name.get(name, 0.0))

    mismatches: list[str] = []
    if baseline.scenario.stable_hash() != replay.scenario.stable_hash():
        mismatches.append("scenario_hash")
    if baseline.evolution_hash != replay.evolution_hash:
        mismatches.append("evolution_hash")
    if baseline.receipt.receipt_hash != replay.receipt.receipt_hash:
        mismatches.append("receipt_hash")

    report = {
        "is_stable_replay": len(mismatches) == 0,
        "mismatches": tuple(mismatches),
        "metric_delta": tuple((name, metric_delta[name]) for name in _METRIC_ORDER),
        "baseline_hash": baseline.stable_hash(),
        "replay_hash": replay.stable_hash(),
    }
    return report


def summarize_topology_evolution(kernel: GovernanceTopologyEvolutionKernel) -> str:
    lines = [
        f"scenario_id={kernel.scenario.scenario_id}",
        f"scenario_hash={kernel.scenario.stable_hash()}",
        f"evolution_hash={kernel.evolution_hash}",
        f"receipt_hash={kernel.receipt.receipt_hash}",
        "metrics:",
    ]
    for metric in kernel.metrics:
        lines.append(
            f"- {metric.metric_order}:{metric.metric_name}={metric.metric_value:.12f}"
        )
    if kernel.violations:
        lines.append("violations:")
        for v in kernel.violations:
            lines.append(f"- {v}")
    else:
        lines.append("violations: none")
    return "\n".join(lines)


__all__ = [
    "TopologyEvolutionScenario",
    "TopologyEvolutionMetric",
    "TopologyEvolutionReceipt",
    "GovernanceTopologyEvolutionKernel",
    "build_topology_evolution_scenario",
    "run_governance_topology_evolution",
    "validate_topology_evolution",
    "build_topology_evolution_receipt",
    "compare_topology_evolution_replay",
    "summarize_topology_evolution",
]
