"""v137.21.1 — Correlated-Noise Suppression Kernel.

Narrow additive Layer-4 advisory-only orchestration release.

Canonical flow:
noise_series
+ topology_series
+ replay_horizon
-> suppression_analysis
+ suppression_receipt

Invariants:
- suppression-before-correction remains side-band only
- decoder core remains untouched
- no randomness
- no async
- no external I/O
- deterministic ordering
- canonical JSON + stable SHA-256
- validator must never raise
- no input mutation
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Sequence, Tuple


_METRIC_ORDER: Tuple[str, ...] = (
    "spatial_correlation_score",
    "temporal_correlation_score",
    "topology_noise_pressure",
    "suppression_alignment_score",
    "residual_noise_score",
    "suppression_confidence_score",
)
_METRIC_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(_METRIC_ORDER)}

_ADVISORY_THRESHOLDS: Tuple[Tuple[str, float], ...] = (
    ("no_suppression_required", 0.05),
    ("mild_correlated_suppression", 0.20),
    ("moderate_correlated_suppression", 0.50),
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


@dataclass(frozen=True)
class _NoiseEntry:
    noise_id: str
    topology_id: str
    temporal_index: float
    noise_level: float
    correlation_hint: float
    residual_noise: float
    replay_identity: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "noise_id": self.noise_id,
            "topology_id": self.topology_id,
            "temporal_index": self.temporal_index,
            "noise_level": self.noise_level,
            "correlation_hint": self.correlation_hint,
            "residual_noise": self.residual_noise,
            "replay_identity": self.replay_identity,
        }


@dataclass(frozen=True)
class _TopologyEntry:
    topology_id: str
    adjacency_pressure: float
    coupling_strength: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topology_id": self.topology_id,
            "adjacency_pressure": self.adjacency_pressure,
            "coupling_strength": self.coupling_strength,
        }


@dataclass(frozen=True)
class _HorizonEntry:
    horizon_id: str
    horizon_step: float
    expected_suppression: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizon_id": self.horizon_id,
            "horizon_step": self.horizon_step,
            "expected_suppression": self.expected_suppression,
        }


@dataclass(frozen=True)
class CorrelatedNoiseNode:
    scenario_id: str
    noise_series: Tuple[Mapping[str, Any], ...]
    topology_series: Tuple[Mapping[str, Any], ...]
    replay_horizon: Tuple[Mapping[str, Any], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "noise_series": [dict(item) for item in self.noise_series],
            "topology_series": [dict(item) for item in self.topology_series],
            "replay_horizon": [dict(item) for item in self.replay_horizon],
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_canonical_json().encode("utf-8"))


@dataclass(frozen=True)
class SuppressionMetric:
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
class SuppressionReceipt:
    scenario_hash: str
    metrics_hash: str
    suppression_hash: str
    suppression_confidence_score: float
    advisory_output: str
    sideband_only: bool
    receipt_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_hash": self.scenario_hash,
            "metrics_hash": self.metrics_hash,
            "suppression_hash": self.suppression_hash,
            "suppression_confidence_score": self.suppression_confidence_score,
            "advisory_output": self.advisory_output,
            "sideband_only": self.sideband_only,
            "receipt_hash": self.receipt_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _receipt_hash_from_payload(self.to_dict())


@dataclass(frozen=True)
class CorrelatedNoiseSuppressionKernel:
    scenario: CorrelatedNoiseNode
    metrics: Tuple[SuppressionMetric, ...]
    suppression_analysis: Mapping[str, Any]
    advisory_output: str
    violations: Tuple[str, ...]
    suppression_receipt: SuppressionReceipt
    suppression_hash: str
    sideband_only: bool
    decoder_untouched: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario.to_dict(),
            "metrics": [metric.to_dict() for metric in self.metrics],
            "suppression_analysis": dict(self.suppression_analysis),
            "advisory_output": self.advisory_output,
            "violations": list(self.violations),
            "suppression_receipt": self.suppression_receipt.to_dict(),
            "suppression_hash": self.suppression_hash,
            "sideband_only": self.sideband_only,
            "decoder_untouched": self.decoder_untouched,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.suppression_hash


def _normalize_noise_entry(raw: Any, index: int) -> _NoiseEntry:
    noise_id = _safe_text(_field(raw, "noise_id", "")).strip() or f"noise_{index}"
    topology_id = _safe_text(_field(raw, "topology_id", "")).strip() or f"topology_{index}"
    temporal_index = _safe_nonneg_float(_field(raw, "temporal_index", _field(raw, "time_index", index)))
    noise_level = _clamp01(
        _safe_nonneg_float(_field(raw, "noise_level", _field(raw, "correlated_noise", 0.0)))
    )
    correlation_hint = _clamp01(_safe_nonneg_float(_field(raw, "correlation_hint", noise_level)))
    residual_noise = _clamp01(_safe_nonneg_float(_field(raw, "residual_noise", 0.0)))
    replay_identity = _safe_text(_field(raw, "replay_identity", "")).strip()
    return _NoiseEntry(
        noise_id=noise_id,
        topology_id=topology_id,
        temporal_index=temporal_index,
        noise_level=noise_level,
        correlation_hint=correlation_hint,
        residual_noise=residual_noise,
        replay_identity=replay_identity,
    )


def _normalize_topology_entry(raw: Any, index: int) -> _TopologyEntry:
    topology_id = _safe_text(_field(raw, "topology_id", "")).strip() or f"topology_{index}"
    adjacency_pressure = _clamp01(
        _safe_nonneg_float(_field(raw, "adjacency_pressure", _field(raw, "noise_pressure", 0.0)))
    )
    coupling_strength = _clamp01(
        _safe_nonneg_float(_field(raw, "coupling_strength", _field(raw, "correlation_coupling", 0.0)))
    )
    return _TopologyEntry(
        topology_id=topology_id,
        adjacency_pressure=adjacency_pressure,
        coupling_strength=coupling_strength,
    )


def _normalize_horizon_entry(raw: Any, index: int) -> _HorizonEntry:
    horizon_id = _safe_text(_field(raw, "horizon_id", "")).strip() or f"horizon_{index}"
    horizon_step = _safe_nonneg_float(_field(raw, "horizon_step", _field(raw, "horizon", index + 1)))
    if horizon_step <= 0.0:
        horizon_step = float(index + 1)
    expected_suppression = _clamp01(
        _safe_nonneg_float(_field(raw, "expected_suppression", _field(raw, "suppression_target", 0.0)))
    )
    return _HorizonEntry(
        horizon_id=horizon_id,
        horizon_step=horizon_step,
        expected_suppression=expected_suppression,
    )


def _ordered_noise(series: Sequence[Any]) -> Tuple[_NoiseEntry, ...]:
    normalized = tuple(_normalize_noise_entry(raw, i) for i, raw in enumerate(series))
    return tuple(sorted(normalized, key=lambda n: (n.temporal_index, n.noise_id)))


def _ordered_topology(series: Sequence[Any]) -> Tuple[_TopologyEntry, ...]:
    normalized = tuple(_normalize_topology_entry(raw, i) for i, raw in enumerate(series))
    return tuple(sorted(normalized, key=lambda t: (t.topology_id, t.adjacency_pressure, t.coupling_strength)))


def _ordered_horizon(series: Sequence[Any]) -> Tuple[_HorizonEntry, ...]:
    normalized = tuple(_normalize_horizon_entry(raw, i) for i, raw in enumerate(series))
    return tuple(sorted(normalized, key=lambda h: (h.horizon_step, h.horizon_id)))


def build_correlated_noise_scenario(
    *,
    scenario_id: str,
    noise_series: Any,
    topology_series: Any,
    replay_horizon: Any,
) -> CorrelatedNoiseNode:
    return CorrelatedNoiseNode(
        scenario_id=_safe_text(scenario_id).strip(),
        noise_series=tuple(node.to_dict() for node in _ordered_noise(_safe_series(noise_series))),
        topology_series=tuple(node.to_dict() for node in _ordered_topology(_safe_series(topology_series))),
        replay_horizon=tuple(node.to_dict() for node in _ordered_horizon(_safe_series(replay_horizon))),
    )


def _metric(name: str, value: float) -> SuppressionMetric:
    return SuppressionMetric(
        metric_name=name,
        metric_order=_METRIC_INDEX[name],
        metric_value=_clamp01(value),
    )


def _metrics_hash(metrics: Sequence[SuppressionMetric]) -> str:
    return _sha256_hex(_canonical_json([metric.to_dict() for metric in metrics]).encode("utf-8"))


def _receipt_payload(
    *,
    scenario_hash: str,
    metrics_hash: str,
    suppression_hash: str,
    suppression_confidence_score: float,
    advisory_output: str,
    sideband_only: bool,
) -> Dict[str, Any]:
    return {
        "scenario_hash": scenario_hash,
        "metrics_hash": metrics_hash,
        "suppression_hash": suppression_hash,
        "suppression_confidence_score": _clamp01(suppression_confidence_score),
        "advisory_output": advisory_output,
        "sideband_only": bool(sideband_only),
    }


def _receipt_hash_from_payload(payload: Mapping[str, Any]) -> str:
    canonical_payload = _receipt_payload(
        scenario_hash=_safe_text(_field(payload, "scenario_hash", "")),
        metrics_hash=_safe_text(_field(payload, "metrics_hash", "")),
        suppression_hash=_safe_text(_field(payload, "suppression_hash", "")),
        suppression_confidence_score=_safe_nonneg_float(_field(payload, "suppression_confidence_score", 0.0)),
        advisory_output=_safe_text(_field(payload, "advisory_output", "")),
        sideband_only=_field(payload, "sideband_only", False) is True,
    )
    return _sha256_hex(_canonical_json(canonical_payload).encode("utf-8"))


def _advisory_from_pressure(composite_pressure: float) -> str:
    bounded = _clamp01(composite_pressure)
    for advisory, threshold in _ADVISORY_THRESHOLDS:
        if bounded <= threshold:
            return advisory
    return "severe_correlated_suppression"


def _compute_metrics(scenario: CorrelatedNoiseNode) -> Tuple[SuppressionMetric, ...]:
    noise = _ordered_noise(scenario.noise_series)
    topology = _ordered_topology(scenario.topology_series)
    horizon = _ordered_horizon(scenario.replay_horizon)

    if len(noise) >= 2:
        temporal_delta_sum = 0.0
        for left, right in zip(noise[:-1], noise[1:]):
            temporal_delta_sum += abs(right.noise_level - left.noise_level)
        temporal_correlation_score = _clamp01(1.0 - (temporal_delta_sum / float(len(noise) - 1)))
    elif noise:
        temporal_correlation_score = noise[0].correlation_hint
    else:
        temporal_correlation_score = 0.0

    group_levels: Dict[str, list[float]] = {}
    hint_sum = 0.0
    for node in noise:
        group_levels.setdefault(node.topology_id, []).append(node.noise_level)
        hint_sum += node.correlation_hint

    spread_sum = 0.0
    spread_count = 0
    for levels in group_levels.values():
        if len(levels) < 2:
            continue
        spread_sum += max(levels) - min(levels)
        spread_count += 1
    if spread_count > 0:
        spatial_correlation_score = _clamp01(1.0 - (spread_sum / float(spread_count)))
    elif noise:
        spatial_correlation_score = _clamp01(hint_sum / float(len(noise)))
    else:
        spatial_correlation_score = 0.0

    topology_lookup = {node.topology_id for node in noise}
    topology_pressure_sum = 0.0
    topology_pressure_count = 0
    for node in topology:
        if topology_lookup and node.topology_id not in topology_lookup:
            continue
        topology_pressure_sum += (node.adjacency_pressure + node.coupling_strength) * 0.5
        topology_pressure_count += 1
    topology_noise_pressure = _clamp01(
        topology_pressure_sum / float(topology_pressure_count) if topology_pressure_count else 0.0
    )

    if horizon:
        weighted_target = 0.0
        total_weight = 0.0
        for node in horizon:
            total_weight += node.horizon_step
            weighted_target += node.horizon_step * node.expected_suppression
        expected_suppression = weighted_target / total_weight if total_weight > 0.0 else 0.0
    else:
        expected_suppression = 0.0

    observed_hint = _clamp01(hint_sum / float(len(noise))) if noise else 0.0
    suppression_alignment_score = _clamp01(1.0 - abs(expected_suppression - observed_hint))

    residual_noise_score = _clamp01(
        sum(node.residual_noise for node in noise) / float(len(noise)) if noise else 0.0
    )

    composite_pressure = _clamp01(
        (
            temporal_correlation_score
            + spatial_correlation_score
            + topology_noise_pressure
            + residual_noise_score
        )
        / 4.0
    )
    suppression_confidence_score = _clamp01(1.0 - (composite_pressure * (1.0 - 0.25 * suppression_alignment_score)))

    return (
        _metric("spatial_correlation_score", spatial_correlation_score),
        _metric("temporal_correlation_score", temporal_correlation_score),
        _metric("topology_noise_pressure", topology_noise_pressure),
        _metric("suppression_alignment_score", suppression_alignment_score),
        _metric("residual_noise_score", residual_noise_score),
        _metric("suppression_confidence_score", suppression_confidence_score),
    )


def _suppression_hash(
    *,
    scenario: CorrelatedNoiseNode,
    metrics: Sequence[SuppressionMetric],
    advisory_output: str,
    violations: Sequence[str],
) -> str:
    payload = {
        "scenario": scenario.to_dict(),
        "metrics": [metric.to_dict() for metric in metrics],
        "advisory_output": advisory_output,
        "violations": list(violations),
        "sideband_only": True,
        "decoder_untouched": True,
    }
    return _sha256_hex(_canonical_json(payload).encode("utf-8"))


def build_suppression_receipt(
    *,
    scenario: CorrelatedNoiseNode,
    metrics: Sequence[SuppressionMetric],
    advisory_output: str,
    suppression_hash: str,
) -> SuppressionReceipt:
    metrics_tuple = tuple(metrics)
    confidence_score = 0.0
    for metric in metrics_tuple:
        if metric.metric_name == "suppression_confidence_score":
            confidence_score = metric.metric_value
            break

    payload = _receipt_payload(
        scenario_hash=scenario.stable_hash(),
        metrics_hash=_metrics_hash(metrics_tuple),
        suppression_hash=suppression_hash,
        suppression_confidence_score=confidence_score,
        advisory_output=advisory_output,
        sideband_only=True,
    )
    receipt_hash = _receipt_hash_from_payload(payload)
    return SuppressionReceipt(receipt_hash=receipt_hash, **payload)


def validate_correlated_noise_suppression(kernel: Any) -> Tuple[str, ...]:
    violations: list[str] = []
    try:
        scenario = _field(kernel, "scenario", None)
        metrics = tuple(_safe_series(_field(kernel, "metrics", ())))
        receipt = _field(kernel, "suppression_receipt", None)

        if scenario is None:
            violations.append("missing_scenario")
            return tuple(violations)

        if not _safe_text(_field(scenario, "scenario_id", "")).strip():
            violations.append("empty_scenario_id")

        if _safe_series(_field(scenario, "noise_series", ())) == ():
            violations.append("empty_noise_series")
        if _safe_series(_field(scenario, "topology_series", ())) == ():
            violations.append("empty_topology_series")

        metric_names = tuple(_safe_text(_field(metric, "metric_name", "")) for metric in metrics)
        if metric_names != _METRIC_ORDER:
            violations.append("metric_order_mismatch")

        for metric in metrics:
            name = _safe_text(_field(metric, "metric_name", ""))
            raw_value = _field(metric, "metric_value", 0.0)
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                violations.append(f"metric_non_numeric:{name}")
                continue

            if not math.isfinite(value):
                violations.append(f"metric_non_finite:{name}")
                continue

            if name in _METRIC_INDEX and (value < 0.0 or value > 1.0):
                violations.append(f"metric_out_of_bounds:{name}")

        sideband_only = _field(kernel, "sideband_only", False)
        if sideband_only is not True:
            violations.append("sideband_only_violation")

        decoder_untouched = _field(kernel, "decoder_untouched", False)
        if decoder_untouched is not True:
            violations.append("decoder_boundary_violation")

        advisory_output = _safe_text(_field(kernel, "advisory_output", ""))
        if advisory_output not in {
            "no_suppression_required",
            "mild_correlated_suppression",
            "moderate_correlated_suppression",
            "severe_correlated_suppression",
        }:
            violations.append("invalid_advisory_output")

        if receipt is not None:
            if _field(receipt, "sideband_only", False) is not True:
                violations.append("receipt_sideband_violation")
            if _safe_text(_field(receipt, "scenario_hash", "")) != scenario.stable_hash():
                violations.append("receipt_scenario_hash_mismatch")
            if _safe_text(_field(receipt, "metrics_hash", "")) != _metrics_hash(metrics):
                violations.append("receipt_metrics_hash_mismatch")
            expected_receipt_hash = _receipt_hash_from_payload(_field(receipt, "to_dict", lambda: {})())
            if _safe_text(_field(receipt, "receipt_hash", "")) != expected_receipt_hash:
                violations.append("receipt_hash_mismatch")
    except Exception:
        violations.append("validator_internal_exception")

    deduped = sorted(set(violations))
    return tuple(deduped)


def run_correlated_noise_suppression(
    *,
    scenario: CorrelatedNoiseNode,
) -> CorrelatedNoiseSuppressionKernel:
    metrics = _compute_metrics(scenario)
    by_name = {metric.metric_name: metric.metric_value for metric in metrics}
    composite_pressure = _clamp01(
        (
            by_name.get("temporal_correlation_score", 0.0)
            + by_name.get("spatial_correlation_score", 0.0)
            + by_name.get("topology_noise_pressure", 0.0)
            + by_name.get("residual_noise_score", 0.0)
        )
        / 4.0
    )
    advisory_output = _advisory_from_pressure(composite_pressure)

    suppression_analysis = {
        "composite_pressure": composite_pressure,
        "suppression_before_correction_sideband_only": True,
        "decoder_semantics_modified": False,
        "noise_series_count": len(scenario.noise_series),
        "topology_series_count": len(scenario.topology_series),
        "replay_horizon_count": len(scenario.replay_horizon),
    }
    provisional_hash = _suppression_hash(
        scenario=scenario,
        metrics=metrics,
        advisory_output=advisory_output,
        violations=(),
    )
    receipt = build_suppression_receipt(
        scenario=scenario,
        metrics=metrics,
        advisory_output=advisory_output,
        suppression_hash=provisional_hash,
    )

    provisional = CorrelatedNoiseSuppressionKernel(
        scenario=scenario,
        metrics=metrics,
        suppression_analysis=suppression_analysis,
        advisory_output=advisory_output,
        violations=(),
        suppression_receipt=receipt,
        suppression_hash=provisional_hash,
        sideband_only=True,
        decoder_untouched=True,
    )
    violations = validate_correlated_noise_suppression(provisional)
    final_hash = _suppression_hash(
        scenario=scenario,
        metrics=metrics,
        advisory_output=advisory_output,
        violations=violations,
    )
    final_receipt = build_suppression_receipt(
        scenario=scenario,
        metrics=metrics,
        advisory_output=advisory_output,
        suppression_hash=final_hash,
    )

    return CorrelatedNoiseSuppressionKernel(
        scenario=scenario,
        metrics=metrics,
        suppression_analysis=suppression_analysis,
        advisory_output=advisory_output,
        violations=violations,
        suppression_receipt=final_receipt,
        suppression_hash=final_hash,
        sideband_only=True,
        decoder_untouched=True,
    )


def compare_suppression_replay(
    baseline: CorrelatedNoiseSuppressionKernel,
    replay: CorrelatedNoiseSuppressionKernel,
) -> Dict[str, Any]:
    report = {
        "baseline_hash": baseline.stable_hash(),
        "replay_hash": replay.stable_hash(),
        "hash_match": baseline.stable_hash() == replay.stable_hash(),
        "baseline_advisory_output": baseline.advisory_output,
        "replay_advisory_output": replay.advisory_output,
        "advisory_match": baseline.advisory_output == replay.advisory_output,
        "baseline_receipt_hash": baseline.suppression_receipt.receipt_hash,
        "replay_receipt_hash": replay.suppression_receipt.receipt_hash,
        "receipt_match": baseline.suppression_receipt.receipt_hash == replay.suppression_receipt.receipt_hash,
        "sideband_only": baseline.sideband_only and replay.sideband_only,
        "decoder_untouched": baseline.decoder_untouched and replay.decoder_untouched,
    }
    report["comparison_hash"] = _sha256_hex(_canonical_json(report).encode("utf-8"))
    return report


def summarize_correlated_noise_suppression(kernel: CorrelatedNoiseSuppressionKernel) -> str:
    return (
        "CorrelatedNoiseSuppressionKernel("
        f"scenario_id={kernel.scenario.scenario_id},"
        f"advisory_output={kernel.advisory_output},"
        f"suppression_hash={kernel.suppression_hash},"
        f"receipt_hash={kernel.suppression_receipt.receipt_hash},"
        f"sideband_only={kernel.sideband_only},"
        f"decoder_untouched={kernel.decoder_untouched}"
        ")"
    )
