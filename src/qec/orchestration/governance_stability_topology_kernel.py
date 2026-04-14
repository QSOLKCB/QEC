"""v137.19.2 — Governance Stability Topology Kernel.

Deterministic topology-analysis layer over benchmark + policy drift artifacts.

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
from typing import Any, Dict, Mapping, Tuple


_METRIC_ORDER: Tuple[str, ...] = (
    "stability_cluster_count",
    "dominant_decision_basin",
    "drift_transition_density",
    "replay_basin_stability",
    "continuity_surface_entropy",
    "boundary_failure_topology_score",
    "topology_severity_score",
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


def _canonical_bytes(data: Any) -> bytes:
    return _canonical_json(data).encode("utf-8")


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_text(value: Any) -> str:
    try:
        if value is None:
            return ""
        return str(value)
    except Exception:
        return ""


def _safe_nonneg_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
    except Exception:
        return 0
    return parsed if parsed >= 0 else 0


def _safe_bool(value: Any) -> bool:
    return value is True


def _field(source: Any, name: str, default: Any = None) -> Any:
    try:
        if isinstance(source, Mapping):
            return source.get(name, default)
        return getattr(source, name, default)
    except Exception:
        return default


def _safe_series(value: Any) -> Tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return ()


@dataclass(frozen=True)
class _BenchmarkNode:
    benchmark_id: str
    decision_basin: str
    boundary_failures: int
    continuity_ok: bool
    replay_identity: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "decision_basin": self.decision_basin,
            "boundary_failures": self.boundary_failures,
            "continuity_ok": self.continuity_ok,
            "replay_identity": self.replay_identity,
        }


@dataclass(frozen=True)
class _DriftNode:
    drift_id: str
    from_basin: str
    to_basin: str
    transition_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drift_id": self.drift_id,
            "from_basin": self.from_basin,
            "to_basin": self.to_basin,
            "transition_count": self.transition_count,
        }


def _normalize_benchmark_node(raw: Any, index: int) -> _BenchmarkNode:
    benchmark_id = _safe_text(_field(raw, "benchmark_id", "")).strip() or f"benchmark_{index}"
    decision_basin = _safe_text(_field(raw, "decision_basin", _field(raw, "decision", ""))).strip()
    if not decision_basin:
        decision_basin = "unknown"
    return _BenchmarkNode(
        benchmark_id=benchmark_id,
        decision_basin=decision_basin,
        boundary_failures=_safe_nonneg_int(_field(raw, "boundary_failures", 0)),
        continuity_ok=_safe_bool(_field(raw, "continuity_ok", False)),
        replay_identity=_safe_text(_field(raw, "replay_identity", "")).strip(),
    )


def _normalize_drift_node(raw: Any, index: int) -> _DriftNode:
    drift_id = _safe_text(_field(raw, "drift_id", "")).strip() or f"drift_{index}"
    from_basin = _safe_text(_field(raw, "from_basin", "")).strip() or "unknown"
    to_basin = _safe_text(_field(raw, "to_basin", "")).strip() or "unknown"
    return _DriftNode(
        drift_id=drift_id,
        from_basin=from_basin,
        to_basin=to_basin,
        transition_count=_safe_nonneg_int(_field(raw, "transition_count", 1)) or 1,
    )


@dataclass(frozen=True)
class StabilityTopologyScenario:
    scenario_id: str
    benchmark_series: Tuple[_BenchmarkNode, ...]
    drift_series: Tuple[_DriftNode, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "benchmark_series": [entry.to_dict() for entry in self.benchmark_series],
            "drift_series": [entry.to_dict() for entry in self.drift_series],
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_canonical_bytes())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class StabilityTopologyMetric:
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
class StabilityTopologyReceipt:
    scenario_hash: str
    metrics_hash: str
    topology_hash: str
    topology_severity_score: float
    receipt_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_hash": self.scenario_hash,
            "metrics_hash": self.metrics_hash,
            "topology_hash": self.topology_hash,
            "topology_severity_score": self.topology_severity_score,
            "receipt_hash": self.receipt_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.receipt_hash


@dataclass(frozen=True)
class GovernanceStabilityTopologyKernel:
    scenario: StabilityTopologyScenario
    metrics: Tuple[StabilityTopologyMetric, ...]
    violations: Tuple[str, ...]
    receipt: StabilityTopologyReceipt
    topology_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario.to_dict(),
            "metrics": [metric.to_dict() for metric in self.metrics],
            "violations": list(self.violations),
            "receipt": self.receipt.to_dict(),
            "topology_hash": self.topology_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.topology_hash


def _benchmark_node_sort_key(node: Any) -> Tuple[str, str, str, str]:
    """Return a deterministic sort key for normalized benchmark nodes."""
    node_payload = node.to_dict() if hasattr(node, "to_dict") else node
    return (
        _safe_text(getattr(node, "benchmark_id", "")),
        _safe_text(getattr(node, "decision_basin", "")),
        _safe_text(getattr(node, "replay_identity", "")),
        _canonical_json(node_payload),
    )


def _drift_node_sort_key(node: Any) -> Tuple[str, str, str, int, str]:
    """Return a deterministic sort key for normalized drift nodes."""
    node_payload = node.to_dict() if hasattr(node, "to_dict") else node
    return (
        _safe_text(getattr(node, "drift_id", "")),
        _safe_text(getattr(node, "from_basin", "")),
        _safe_text(getattr(node, "to_basin", "")),
        int(getattr(node, "transition_count", 0)),
        _canonical_json(node_payload),
    )


def build_stability_topology_scenario(
    *,
    scenario_id: str,
    benchmark_series: Any,
    drift_series: Any,
) -> StabilityTopologyScenario:
    normalized_benchmarks = tuple(
        sorted(
            (
                _normalize_benchmark_node(item, index)
                for index, item in enumerate(_safe_series(benchmark_series))
            ),
            key=_benchmark_node_sort_key,
        )
    )
    normalized_drifts = tuple(
        sorted(
            (
                _normalize_drift_node(item, index)
                for index, item in enumerate(_safe_series(drift_series))
            ),
            key=_drift_node_sort_key,
        )
    )
    return StabilityTopologyScenario(
        scenario_id=_safe_text(scenario_id).strip(),
        benchmark_series=normalized_benchmarks,
        drift_series=normalized_drifts,
    )


def _bounded01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def _compute_metrics(scenario: StabilityTopologyScenario) -> Tuple[StabilityTopologyMetric, ...]:
    benchmark_series = scenario.benchmark_series
    drift_series = scenario.drift_series

    basin_counts: Dict[str, int] = {}
    boundary_total = 0
    continuity_true = 0
    replay_matches = 0
    prev_replay_key = ""

    for idx, node in enumerate(benchmark_series):
        basin_counts[node.decision_basin] = basin_counts.get(node.decision_basin, 0) + 1
        boundary_total += node.boundary_failures
        if node.continuity_ok:
            continuity_true += 1
        replay_key = f"{node.decision_basin}|{node.replay_identity}"
        if idx > 0 and replay_key == prev_replay_key:
            replay_matches += 1
        prev_replay_key = replay_key

    bench_n = len(benchmark_series)
    clusters = float(len(basin_counts))
    dominant = 0.0
    if bench_n > 0:
        dominant = float(max(basin_counts.values())) / float(bench_n)

    total_transitions = sum(node.transition_count for node in drift_series)
    drift_n = len(drift_series)
    drift_transition_density = 0.0
    if drift_n > 0:
        drift_transition_density = float(total_transitions) / float(drift_n)
        drift_transition_density = _bounded01(drift_transition_density / 10.0)

    replay_basin_stability = 1.0
    if bench_n > 1:
        replay_basin_stability = float(replay_matches) / float(bench_n - 1)

    continuity_surface_entropy = 0.0
    if bench_n > 0:
        p_true = float(continuity_true) / float(bench_n)
        p_false = 1.0 - p_true
        parts = []
        if p_true > 0.0:
            parts.append(-p_true * math.log(p_true, 2.0))
        if p_false > 0.0:
            parts.append(-p_false * math.log(p_false, 2.0))
        continuity_surface_entropy = sum(parts)

    boundary_failure_topology_score = 0.0
    if bench_n > 0:
        boundary_failure_topology_score = float(boundary_total) / float(boundary_total + bench_n)

    topology_severity_score = _bounded01(
        0.22 * (1.0 - _bounded01(dominant))
        + 0.18 * _bounded01(drift_transition_density)
        + 0.16 * (1.0 - _bounded01(replay_basin_stability))
        + 0.14 * _bounded01(continuity_surface_entropy)
        + 0.20 * _bounded01(boundary_failure_topology_score)
        + 0.10 * _bounded01(min(1.0, clusters / 10.0))
    )

    values = {
        "stability_cluster_count": clusters,
        "dominant_decision_basin": _bounded01(dominant),
        "drift_transition_density": _bounded01(drift_transition_density),
        "replay_basin_stability": _bounded01(replay_basin_stability),
        "continuity_surface_entropy": _bounded01(continuity_surface_entropy),
        "boundary_failure_topology_score": _bounded01(boundary_failure_topology_score),
        "topology_severity_score": topology_severity_score,
    }
    return tuple(
        StabilityTopologyMetric(
            metric_name=name,
            metric_order=_METRIC_INDEX[name],
            metric_value=float(values[name]),
        )
        for name in _METRIC_ORDER
    )


def build_stability_topology_receipt(
    *,
    scenario_hash: str,
    metrics_hash: str,
    topology_hash: str,
    topology_severity_score: float,
) -> StabilityTopologyReceipt:
    payload = {
        "metrics_hash": metrics_hash,
        "scenario_hash": scenario_hash,
        "topology_hash": topology_hash,
        "topology_severity_score": float(topology_severity_score),
    }
    receipt_hash = _sha256_hex(_canonical_bytes(payload))
    return StabilityTopologyReceipt(
        scenario_hash=scenario_hash,
        metrics_hash=metrics_hash,
        topology_hash=topology_hash,
        topology_severity_score=float(topology_severity_score),
        receipt_hash=receipt_hash,
    )


def run_governance_stability_topology(scenario: Any) -> GovernanceStabilityTopologyKernel:
    violations: list[str] = []
    if isinstance(scenario, StabilityTopologyScenario):
        safe_scenario = scenario
    else:
        safe_scenario = build_stability_topology_scenario(
            scenario_id="",
            benchmark_series=(),
            drift_series=(),
        )
        violations.append("malformed_scenario_input")

    if not safe_scenario.scenario_id:
        violations.append("empty_scenario_id")
    if len(safe_scenario.benchmark_series) == 0:
        violations.append("empty_benchmark_series")
    if len(safe_scenario.drift_series) == 0:
        violations.append("empty_drift_series")

    metrics = _compute_metrics(safe_scenario)
    metrics_hash = _sha256_hex(_canonical_bytes([metric.to_dict() for metric in metrics]))
    scenario_hash = safe_scenario.stable_hash()
    severity = next(
        (metric.metric_value for metric in metrics if metric.metric_name == "topology_severity_score"),
        0.0,
    )

    ordered_violations = tuple(sorted(set(violations)))
    topology_body = {
        "metrics_hash": metrics_hash,
        "scenario_hash": scenario_hash,
        "violations": list(ordered_violations),
    }
    topology_hash = _sha256_hex(_canonical_bytes(topology_body))

    receipt = build_stability_topology_receipt(
        scenario_hash=scenario_hash,
        metrics_hash=metrics_hash,
        topology_hash=topology_hash,
        topology_severity_score=severity,
    )
    return GovernanceStabilityTopologyKernel(
        scenario=safe_scenario,
        metrics=metrics,
        violations=ordered_violations,
        receipt=receipt,
        topology_hash=topology_hash,
    )


def validate_stability_topology(analysis: Any) -> Tuple[str, ...]:
    try:
        if not isinstance(analysis, GovernanceStabilityTopologyKernel):
            return ("malformed_governance_stability_topology",)
        violations: list[str] = []
        if not isinstance(analysis.scenario, StabilityTopologyScenario):
            violations.append("malformed_scenario")
        if not isinstance(analysis.metrics, tuple):
            violations.append("malformed_metrics_collection")
        else:
            names = tuple(getattr(metric, "metric_name", "") for metric in analysis.metrics)
            if names != _METRIC_ORDER:
                violations.append("metric_ordering_mismatch")
            for metric in analysis.metrics:
                if not isinstance(metric, StabilityTopologyMetric):
                    violations.append("malformed_metric_entry")
                    break
                if metric.metric_order != _METRIC_INDEX.get(metric.metric_name, -1):
                    violations.append("metric_order_index_mismatch")
                    break
                if metric.metric_value < 0.0:
                    violations.append("negative_metric_value")
                    break

        if not isinstance(analysis.receipt, StabilityTopologyReceipt):
            violations.append("malformed_receipt")
        else:
            expected_receipt = build_stability_topology_receipt(
                scenario_hash=analysis.receipt.scenario_hash,
                metrics_hash=analysis.receipt.metrics_hash,
                topology_hash=analysis.receipt.topology_hash,
                topology_severity_score=analysis.receipt.topology_severity_score,
            )
            if expected_receipt.receipt_hash != analysis.receipt.receipt_hash:
                violations.append("receipt_hash_mismatch")
            if isinstance(analysis.scenario, StabilityTopologyScenario):
                if analysis.receipt.scenario_hash != analysis.scenario.stable_hash():
                    violations.append("scenario_hash_mismatch")
            if isinstance(analysis.metrics, tuple):
                expected_metrics_hash = _sha256_hex(
                    _canonical_bytes(
                        [
                            metric.to_dict()
                            for metric in analysis.metrics
                            if isinstance(metric, StabilityTopologyMetric)
                        ]
                    )
                )
                if analysis.receipt.metrics_hash != expected_metrics_hash:
                    violations.append("metrics_hash_mismatch")

        expected_topology_hash = _sha256_hex(
            _canonical_bytes(
                {
                    "metrics_hash": analysis.receipt.metrics_hash
                    if isinstance(analysis.receipt, StabilityTopologyReceipt)
                    else "",
                    "scenario_hash": analysis.receipt.scenario_hash
                    if isinstance(analysis.receipt, StabilityTopologyReceipt)
                    else "",
                    "violations": list(analysis.violations),
                }
            )
        )
        if analysis.topology_hash != expected_topology_hash:
            violations.append("topology_hash_body_mismatch")
        return tuple(sorted(set(violations)))
    except Exception as exc:  # pragma: no cover
        return (f"validator_error:{type(exc).__name__}",)


def compare_stability_topology_replay(analysis_a: Any, analysis_b: Any) -> Dict[str, Any]:
    try:
        if not isinstance(analysis_a, GovernanceStabilityTopologyKernel) or not isinstance(
            analysis_b, GovernanceStabilityTopologyKernel
        ):
            return {
                "match": False,
                "mismatch_fields": ("type",),
                "analysis_a_hash": _safe_text(_field(analysis_a, "topology_hash", "")),
                "analysis_b_hash": _safe_text(_field(analysis_b, "topology_hash", "")),
            }
        mismatches = []
        if analysis_a.topology_hash != analysis_b.topology_hash:
            mismatches.append("topology_hash")
        if analysis_a.scenario.stable_hash() != analysis_b.scenario.stable_hash():
            mismatches.append("scenario_hash")
        if tuple(metric.stable_hash() for metric in analysis_a.metrics) != tuple(
            metric.stable_hash() for metric in analysis_b.metrics
        ):
            mismatches.append("metrics")
        if analysis_a.receipt.stable_hash() != analysis_b.receipt.stable_hash():
            mismatches.append("receipt_hash")
        if analysis_a.violations != analysis_b.violations:
            mismatches.append("violations")
        return {
            "match": len(mismatches) == 0,
            "mismatch_fields": tuple(mismatches),
            "analysis_a_hash": analysis_a.topology_hash,
            "analysis_b_hash": analysis_b.topology_hash,
        }
    except Exception as exc:  # pragma: no cover
        return {
            "match": False,
            "mismatch_fields": (f"compare_error:{type(exc).__name__}",),
            "analysis_a_hash": "",
            "analysis_b_hash": "",
        }


def summarize_stability_topology(analysis: Any) -> Dict[str, Any]:
    try:
        if not isinstance(analysis, GovernanceStabilityTopologyKernel):
            return {
                "valid": False,
                "scenario_id": "",
                "topology_severity_score": 0.0,
                "metric_values": {},
                "violations": ("malformed_governance_stability_topology",),
                "topology_hash": "",
                "receipt_hash": "",
            }
        validated = validate_stability_topology(analysis)
        metric_values = {metric.metric_name: metric.metric_value for metric in analysis.metrics}
        return {
            "valid": len(validated) == 0 and len(analysis.violations) == 0,
            "scenario_id": analysis.scenario.scenario_id,
            "topology_severity_score": analysis.receipt.topology_severity_score,
            "metric_values": dict(sorted(metric_values.items())),
            "violations": tuple(sorted(set(tuple(analysis.violations) + tuple(validated)))),
            "topology_hash": analysis.topology_hash,
            "receipt_hash": analysis.receipt.receipt_hash,
        }
    except Exception as exc:  # pragma: no cover
        return {
            "valid": False,
            "scenario_id": "",
            "topology_severity_score": 0.0,
            "metric_values": {},
            "violations": (f"summary_error:{type(exc).__name__}",),
            "topology_hash": "",
            "receipt_hash": "",
        }
