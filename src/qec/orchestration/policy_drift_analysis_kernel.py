"""v137.19.1 — Policy Drift Analysis Kernel.

Deterministic analysis layer that measures policy drift across benchmark runs
and simulation variants.

This module is an analysis + metrics + receipt layer only:
- no external I/O
- no async
- no wall-clock reads
- no randomness
- deterministic ordering everywhere
- canonical JSON + stable SHA-256 hashing
- zero mutation of input benchmark artifacts
- validator never raises
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, Mapping, Tuple


_METRIC_ORDER: Tuple[str, ...] = (
    "allow_drift_rate",
    "deny_drift_rate",
    "decision_surface_delta",
    "boundary_failure_delta",
    "continuity_delta",
    "replay_stability_delta",
    "trace_length_delta",
    "drift_severity_score",
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
    try:
        if isinstance(value, bool):
            return int(value)
        parsed = int(value)
    except Exception:
        return 0
    if parsed < 0:
        return 0
    return parsed


def _safe_bool(value: Any) -> bool:
    return value is True


def _field(source: Any, name: str, default: Any = None) -> Any:
    try:
        if isinstance(source, Mapping):
            return source.get(name, default)
        return getattr(source, name, default)
    except Exception:
        return default


def _safe_decision_surface(value: Any) -> Tuple[Tuple[str, str], ...]:
    result: set = set()
    try:
        if isinstance(value, Mapping):
            iterable: Tuple[Any, ...] = tuple(value.items())
        elif isinstance(value, (list, tuple)):
            iterable = tuple(value)
        else:
            iterable = ()
        for item in iterable:
            try:
                if isinstance(item, Mapping):
                    key = _safe_text(item.get("key", item.get("decision_key", "")))
                    dec = _safe_text(item.get("decision", item.get("value", "")))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    key = _safe_text(item[0])
                    dec = _safe_text(item[1])
                else:
                    continue
                key = key.strip()
                dec = dec.strip()
                if not key:
                    continue
                result.add((key, dec))
            except Exception:
                continue
    except Exception:
        pass
    return tuple(sorted(result))


@dataclass(frozen=True)
class _BenchmarkSnapshot:
    benchmark_id: str
    allow_count: int
    deny_count: int
    decision_surface: Tuple[Tuple[str, str], ...]
    boundary_failures: int
    continuity_ok: bool
    replay_identity: str
    trace_length: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "allow_count": self.allow_count,
            "deny_count": self.deny_count,
            "decision_surface": [list(pair) for pair in self.decision_surface],
            "boundary_failures": self.boundary_failures,
            "continuity_ok": self.continuity_ok,
            "replay_identity": self.replay_identity,
            "trace_length": self.trace_length,
        }


def _normalize_benchmark(data: Any, fallback_id: str) -> _BenchmarkSnapshot:
    raw_id = _field(data, "benchmark_id", None)
    benchmark_id = _safe_text(raw_id).strip() or fallback_id
    return _BenchmarkSnapshot(
        benchmark_id=benchmark_id,
        allow_count=_safe_nonneg_int(_field(data, "allow_count", 0)),
        deny_count=_safe_nonneg_int(_field(data, "deny_count", 0)),
        decision_surface=_safe_decision_surface(_field(data, "decision_surface", ())),
        boundary_failures=_safe_nonneg_int(_field(data, "boundary_failures", 0)),
        continuity_ok=_safe_bool(_field(data, "continuity_ok", False)),
        replay_identity=_safe_text(_field(data, "replay_identity", "")).strip(),
        trace_length=_safe_nonneg_int(_field(data, "trace_length", 0)),
    )


@dataclass(frozen=True)
class PolicyDriftScenario:
    scenario_id: str
    benchmark_a: _BenchmarkSnapshot
    benchmark_b: _BenchmarkSnapshot

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "benchmark_a": self.benchmark_a.to_dict(),
            "benchmark_b": self.benchmark_b.to_dict(),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_canonical_json().encode("utf-8"))


@dataclass(frozen=True)
class PolicyDriftMetric:
    metric_name: str
    metric_order: int
    value_a: float
    value_b: float
    delta: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "metric_order": self.metric_order,
            "value_a": self.value_a,
            "value_b": self.value_b,
            "delta": self.delta,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_canonical_json().encode("utf-8"))


@dataclass(frozen=True)
class PolicyDriftReceipt:
    scenario_hash: str
    metrics_hash: str
    drift_severity_score: float
    analysis_hash: str
    receipt_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_hash": self.scenario_hash,
            "metrics_hash": self.metrics_hash,
            "drift_severity_score": self.drift_severity_score,
            "analysis_hash": self.analysis_hash,
            "receipt_hash": self.receipt_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.receipt_hash


@dataclass(frozen=True)
class PolicyDriftAnalysisKernel:
    scenario: PolicyDriftScenario
    metrics: Tuple[PolicyDriftMetric, ...]
    violations: Tuple[str, ...]
    receipt: PolicyDriftReceipt
    analysis_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario.to_dict(),
            "metrics": [metric.to_dict() for metric in self.metrics],
            "violations": list(self.violations),
            "receipt": self.receipt.to_dict(),
            "analysis_hash": self.analysis_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.analysis_hash


def build_policy_drift_scenario(
    *,
    scenario_id: str,
    benchmark_a: Any,
    benchmark_b: Any,
) -> PolicyDriftScenario:
    normalized_scenario_id = _safe_text(scenario_id).strip()
    snapshot_a = _normalize_benchmark(benchmark_a, fallback_id="benchmark_a")
    snapshot_b = _normalize_benchmark(benchmark_b, fallback_id="benchmark_b")
    return PolicyDriftScenario(
        scenario_id=normalized_scenario_id,
        benchmark_a=snapshot_a,
        benchmark_b=snapshot_b,
    )


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _make_metric(
    name: str, value_a: float, value_b: float, delta: float
) -> PolicyDriftMetric:
    return PolicyDriftMetric(
        metric_name=name,
        metric_order=_METRIC_INDEX[name],
        value_a=float(value_a),
        value_b=float(value_b),
        delta=float(delta),
    )


def _compute_drift_metrics(
    snapshot_a: _BenchmarkSnapshot, snapshot_b: _BenchmarkSnapshot
) -> Tuple[PolicyDriftMetric, ...]:
    total_a = snapshot_a.allow_count + snapshot_a.deny_count
    total_b = snapshot_b.allow_count + snapshot_b.deny_count
    allow_a = _rate(snapshot_a.allow_count, total_a)
    allow_b = _rate(snapshot_b.allow_count, total_b)
    deny_a = _rate(snapshot_a.deny_count, total_a)
    deny_b = _rate(snapshot_b.deny_count, total_b)
    allow_drift = abs(allow_a - allow_b)
    deny_drift = abs(deny_a - deny_b)

    set_a = set(snapshot_a.decision_surface)
    set_b = set(snapshot_b.decision_surface)
    union = set_a | set_b
    if union:
        surface_delta = float(len(set_a ^ set_b)) / float(len(union))
    else:
        surface_delta = 0.0

    boundary_delta = float(
        abs(snapshot_a.boundary_failures - snapshot_b.boundary_failures)
    )
    continuity_delta = 0.0 if snapshot_a.continuity_ok == snapshot_b.continuity_ok else 1.0
    replay_delta = (
        0.0 if snapshot_a.replay_identity == snapshot_b.replay_identity else 1.0
    )
    trace_delta = float(abs(snapshot_a.trace_length - snapshot_b.trace_length))

    severity = (
        0.2 * allow_drift
        + 0.2 * deny_drift
        + 0.2 * surface_delta
        + 0.1 * min(1.0, boundary_delta / 10.0)
        + 0.1 * continuity_delta
        + 0.1 * replay_delta
        + 0.1 * min(1.0, trace_delta / 100.0)
    )

    return (
        _make_metric("allow_drift_rate", allow_a, allow_b, allow_drift),
        _make_metric("deny_drift_rate", deny_a, deny_b, deny_drift),
        _make_metric(
            "decision_surface_delta",
            float(len(set_a)),
            float(len(set_b)),
            surface_delta,
        ),
        _make_metric(
            "boundary_failure_delta",
            float(snapshot_a.boundary_failures),
            float(snapshot_b.boundary_failures),
            boundary_delta,
        ),
        _make_metric(
            "continuity_delta",
            1.0 if snapshot_a.continuity_ok else 0.0,
            1.0 if snapshot_b.continuity_ok else 0.0,
            continuity_delta,
        ),
        _make_metric(
            "replay_stability_delta",
            1.0 if snapshot_a.replay_identity else 0.0,
            1.0 if snapshot_b.replay_identity else 0.0,
            replay_delta,
        ),
        _make_metric(
            "trace_length_delta",
            float(snapshot_a.trace_length),
            float(snapshot_b.trace_length),
            trace_delta,
        ),
        _make_metric("drift_severity_score", 0.0, 0.0, severity),
    )


def build_policy_drift_receipt(
    *,
    scenario_hash: str,
    metrics_hash: str,
    drift_severity_score: float,
    analysis_hash: str,
) -> PolicyDriftReceipt:
    preimage = {
        "analysis_hash": analysis_hash,
        "drift_severity_score": float(drift_severity_score),
        "metrics_hash": metrics_hash,
        "scenario_hash": scenario_hash,
    }
    receipt_hash = _sha256_hex(_canonical_bytes(preimage))
    return PolicyDriftReceipt(
        scenario_hash=scenario_hash,
        metrics_hash=metrics_hash,
        drift_severity_score=float(drift_severity_score),
        analysis_hash=analysis_hash,
        receipt_hash=receipt_hash,
    )


def run_policy_drift_analysis(scenario: Any) -> PolicyDriftAnalysisKernel:
    violations: list = []

    if isinstance(scenario, PolicyDriftScenario):
        safe_scenario = scenario
    else:
        violations.append("malformed_scenario_input")
        safe_scenario = PolicyDriftScenario(
            scenario_id="",
            benchmark_a=_normalize_benchmark(None, fallback_id="benchmark_a"),
            benchmark_b=_normalize_benchmark(None, fallback_id="benchmark_b"),
        )

    if not safe_scenario.scenario_id:
        violations.append("empty_scenario_id")
    if not safe_scenario.benchmark_a.benchmark_id:
        violations.append("empty_benchmark_a_id")
    if not safe_scenario.benchmark_b.benchmark_id:
        violations.append("empty_benchmark_b_id")

    metrics = _compute_drift_metrics(
        safe_scenario.benchmark_a, safe_scenario.benchmark_b
    )
    metrics_payload = [metric.to_dict() for metric in metrics]
    metrics_hash = _sha256_hex(_canonical_bytes(metrics_payload))
    severity = next(
        (metric.delta for metric in metrics if metric.metric_name == "drift_severity_score"),
        0.0,
    )
    scenario_hash = safe_scenario.stable_hash()

    ordered_violations = tuple(sorted(set(violations)))
    analysis_body = {
        "metrics_hash": metrics_hash,
        "scenario_hash": scenario_hash,
        "violations": list(ordered_violations),
    }
    analysis_hash = _sha256_hex(_canonical_bytes(analysis_body))

    receipt = build_policy_drift_receipt(
        scenario_hash=scenario_hash,
        metrics_hash=metrics_hash,
        drift_severity_score=severity,
        analysis_hash=analysis_hash,
    )

    return PolicyDriftAnalysisKernel(
        scenario=safe_scenario,
        metrics=metrics,
        violations=ordered_violations,
        receipt=receipt,
        analysis_hash=analysis_hash,
    )


def validate_policy_drift_analysis(analysis: Any) -> Tuple[str, ...]:
    try:
        if not isinstance(analysis, PolicyDriftAnalysisKernel):
            return ("malformed_policy_drift_analysis",)

        violations: list = []

        if not isinstance(analysis.scenario, PolicyDriftScenario):
            violations.append("malformed_scenario")

        if not isinstance(analysis.metrics, tuple):
            violations.append("malformed_metrics_collection")
        else:
            names = tuple(
                getattr(metric, "metric_name", "") for metric in analysis.metrics
            )
            if names != _METRIC_ORDER:
                violations.append("metric_ordering_mismatch")
            for metric in analysis.metrics:
                if not isinstance(metric, PolicyDriftMetric):
                    violations.append("malformed_metric_entry")
                    break
                if metric.delta < 0.0:
                    violations.append("negative_metric_delta")
                    break
                if metric.metric_order != _METRIC_INDEX.get(metric.metric_name, -1):
                    violations.append("metric_order_index_mismatch")
                    break

        receipt = analysis.receipt
        if not isinstance(receipt, PolicyDriftReceipt):
            violations.append("malformed_receipt")
        else:
            expected_receipt = build_policy_drift_receipt(
                scenario_hash=receipt.scenario_hash,
                metrics_hash=receipt.metrics_hash,
                drift_severity_score=receipt.drift_severity_score,
                analysis_hash=receipt.analysis_hash,
            )
            if expected_receipt.receipt_hash != receipt.receipt_hash:
                violations.append("receipt_hash_mismatch")

            if isinstance(analysis.scenario, PolicyDriftScenario):
                if receipt.scenario_hash != analysis.scenario.stable_hash():
                    violations.append("scenario_hash_mismatch")

            if isinstance(analysis.metrics, tuple):
                metrics_payload = [
                    metric.to_dict()
                    for metric in analysis.metrics
                    if isinstance(metric, PolicyDriftMetric)
                ]
                expected_metrics_hash = _sha256_hex(_canonical_bytes(metrics_payload))
                if receipt.metrics_hash != expected_metrics_hash:
                    violations.append("metrics_hash_mismatch")

            if receipt.analysis_hash != analysis.analysis_hash:
                violations.append("analysis_hash_linkage_mismatch")

        expected_analysis_body = {
            "metrics_hash": analysis.receipt.metrics_hash
            if isinstance(analysis.receipt, PolicyDriftReceipt)
            else "",
            "scenario_hash": analysis.receipt.scenario_hash
            if isinstance(analysis.receipt, PolicyDriftReceipt)
            else "",
            "violations": list(analysis.violations),
        }
        expected_analysis_hash = _sha256_hex(_canonical_bytes(expected_analysis_body))
        if analysis.analysis_hash != expected_analysis_hash:
            violations.append("analysis_hash_body_mismatch")

        return tuple(sorted(set(violations)))
    except Exception as exc:  # pragma: no cover
        return (f"validator_error:{type(exc).__name__}",)


def compare_policy_drift_replay(analysis_a: Any, analysis_b: Any) -> Dict[str, Any]:
    try:
        if not isinstance(analysis_a, PolicyDriftAnalysisKernel) or not isinstance(
            analysis_b, PolicyDriftAnalysisKernel
        ):
            return {
                "match": False,
                "mismatch_fields": ("type",),
                "analysis_a_hash": _safe_text(_field(analysis_a, "analysis_hash", "")),
                "analysis_b_hash": _safe_text(_field(analysis_b, "analysis_hash", "")),
            }

        mismatches: list = []
        if analysis_a.analysis_hash != analysis_b.analysis_hash:
            mismatches.append("analysis_hash")
        if analysis_a.scenario.stable_hash() != analysis_b.scenario.stable_hash():
            mismatches.append("scenario_hash")
        metrics_a = tuple(metric.stable_hash() for metric in analysis_a.metrics)
        metrics_b = tuple(metric.stable_hash() for metric in analysis_b.metrics)
        if metrics_a != metrics_b:
            mismatches.append("metrics")
        if analysis_a.receipt.receipt_hash != analysis_b.receipt.receipt_hash:
            mismatches.append("receipt_hash")
        if analysis_a.violations != analysis_b.violations:
            mismatches.append("violations")

        return {
            "match": len(mismatches) == 0,
            "mismatch_fields": tuple(mismatches),
            "analysis_a_hash": analysis_a.analysis_hash,
            "analysis_b_hash": analysis_b.analysis_hash,
        }
    except Exception as exc:  # pragma: no cover
        return {
            "match": False,
            "mismatch_fields": (f"compare_error:{type(exc).__name__}",),
            "analysis_a_hash": "",
            "analysis_b_hash": "",
        }


def summarize_policy_drift(analysis: Any) -> Dict[str, Any]:
    try:
        if not isinstance(analysis, PolicyDriftAnalysisKernel):
            return {
                "valid": False,
                "scenario_id": "",
                "drift_severity_score": 0.0,
                "metric_deltas": {},
                "violations": ("malformed_policy_drift_analysis",),
                "analysis_hash": "",
                "receipt_hash": "",
            }
        metric_deltas = {
            metric.metric_name: metric.delta for metric in analysis.metrics
        }
        return {
            "valid": len(analysis.violations) == 0,
            "scenario_id": analysis.scenario.scenario_id,
            "drift_severity_score": analysis.receipt.drift_severity_score,
            "metric_deltas": dict(sorted(metric_deltas.items())),
            "violations": tuple(analysis.violations),
            "analysis_hash": analysis.analysis_hash,
            "receipt_hash": analysis.receipt.receipt_hash,
        }
    except Exception as exc:  # pragma: no cover
        return {
            "valid": False,
            "scenario_id": "",
            "drift_severity_score": 0.0,
            "metric_deltas": {},
            "violations": (f"summary_error:{type(exc).__name__}",),
            "analysis_hash": "",
            "receipt_hash": "",
        }
