"""v137.21.2 — Latency & Throughput Budget Ledger.

Deterministic side-band accounting of timing and throughput budgets for replay-safe
validation. Decoder semantics and logical artifacts remain untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Sequence, Tuple, Union

LEDGER_VERSION = "v137.21.2"
METRIC_ORDER: Tuple[str, ...] = (
    "latency_budget_compliance",
    "throughput_budget_compliance",
    "backlog_pressure_score",
    "scheduling_stability_score",
    "timing_variation_score",
    "budget_confidence_score",
)
ADVISORY_STATES: Tuple[str, ...] = (
    "within_budget",
    "near_budget_limit",
    "budget_pressure",
    "budget_violation",
)

SampleLike = Union["LatencyThroughputSample", Mapping[str, Any]]
LedgerLike = Union["LatencyThroughputBudgetLedger", Mapping[str, Any]]


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def _normalize_float(value: Any, default: float = 0.0) -> Tuple[float, str | None]:
    if value is None:
        return float(default), "missing"
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default), "malformed"
    if math.isnan(parsed):
        return float(default), "nan"
    if math.isinf(parsed):
        return float(default), "pos_inf" if parsed > 0 else "neg_inf"
    return parsed, None


def _normalize_text(value: Any, default: str) -> str:
    text = str(value).strip() if value is not None else ""
    return text if text else default


def _stable_sample_sort_key(sample: "LatencyThroughputSample") -> Tuple[int, str, float, float, float, Tuple[str, ...]]:
    return (
        sample.sample_index,
        sample.sample_id,
        sample.latency_ms,
        sample.throughput_units,
        sample.backlog_units,
        sample.normalization_flags,
    )


def _normalize_sample(raw: SampleLike, fallback_index: int) -> "LatencyThroughputSample":
    if isinstance(raw, LatencyThroughputSample):
        return raw
    if not isinstance(raw, Mapping):
        raw = {"sample_id": f"sample-{fallback_index}", "latency_ms": raw}

    sample_index_raw = raw.get("sample_index", fallback_index)
    try:
        sample_index = int(sample_index_raw)
    except (TypeError, ValueError):
        sample_index = fallback_index

    sample_id = _normalize_text(raw.get("sample_id"), f"sample-{fallback_index}")

    latency_ms, latency_flag = _normalize_float(raw.get("latency_ms"), default=0.0)
    throughput_units, throughput_flag = _normalize_float(raw.get("throughput_units"), default=0.0)
    backlog_units, backlog_flag = _normalize_float(raw.get("backlog_units"), default=0.0)

    flags = tuple(sorted(flag for flag in (latency_flag, throughput_flag, backlog_flag) if flag is not None))

    return LatencyThroughputSample(
        sample_index=sample_index,
        sample_id=sample_id,
        latency_ms=max(0.0, float(latency_ms)),
        throughput_units=max(0.0, float(throughput_units)),
        backlog_units=max(0.0, float(backlog_units)),
        normalization_flags=flags,
    )


def _normalize_series(series: Any) -> Tuple[LatencyThroughputSample, ...]:
    if not isinstance(series, Sequence) or isinstance(series, (str, bytes)):
        return ()
    samples = tuple(_normalize_sample(item, idx) for idx, item in enumerate(series))
    return tuple(sorted(samples, key=_stable_sample_sort_key))


def _mean(values: Tuple[float, ...], default: float = 0.0) -> float:
    if not values:
        return float(default)
    return float(sum(values) / float(len(values)))


def _normalize_requirements(raw: Any) -> Dict[str, float]:
    requirements = raw if isinstance(raw, Mapping) else {}
    latency_budget_ms, _ = _normalize_float(requirements.get("latency_budget_ms"), default=1.0)
    min_throughput_units, _ = _normalize_float(requirements.get("min_throughput_units"), default=1.0)
    max_backlog_units, _ = _normalize_float(requirements.get("max_backlog_units"), default=1.0)
    max_timing_variation_ms, _ = _normalize_float(requirements.get("max_timing_variation_ms"), default=1.0)

    return {
        "latency_budget_ms": max(1e-12, abs(float(latency_budget_ms))),
        "min_throughput_units": max(1e-12, abs(float(min_throughput_units))),
        "max_backlog_units": max(1e-12, abs(float(max_backlog_units))),
        "max_timing_variation_ms": max(1e-12, abs(float(max_timing_variation_ms))),
    }


@dataclass(frozen=True)
class LatencyThroughputSample:
    sample_index: int
    sample_id: str
    latency_ms: float
    throughput_units: float
    backlog_units: float
    normalization_flags: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "sample_id": self.sample_id,
            "latency_ms": self.latency_ms,
            "throughput_units": self.throughput_units,
            "backlog_units": self.backlog_units,
            "normalization_flags": list(self.normalization_flags),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_canonical_json().encode("utf-8"))


@dataclass(frozen=True)
class BudgetMetric:
    metric_name: str
    metric_value: float

    def to_dict(self) -> Dict[str, Any]:
        return {"metric_name": self.metric_name, "metric_value": self.metric_value}

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_canonical_json().encode("utf-8"))


@dataclass(frozen=True)
class BudgetReceipt:
    ledger_version: str
    advisory_state: str
    logical_replay_identity: str
    logical_outputs_valid: bool
    timing_budget_exceeded: bool
    composite_budget_pressure: float
    ledger_hash: str
    receipt_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ledger_version": self.ledger_version,
            "advisory_state": self.advisory_state,
            "logical_replay_identity": self.logical_replay_identity,
            "logical_outputs_valid": self.logical_outputs_valid,
            "timing_budget_exceeded": self.timing_budget_exceeded,
            "composite_budget_pressure": self.composite_budget_pressure,
            "ledger_hash": self.ledger_hash,
            "receipt_hash": self.receipt_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.receipt_hash


@dataclass(frozen=True)
class LatencyThroughputBudgetLedger:
    ledger_version: str
    timing_series: Tuple[LatencyThroughputSample, ...]
    throughput_series: Tuple[LatencyThroughputSample, ...]
    budget_requirements: Dict[str, float]
    budget_analysis: Tuple[BudgetMetric, ...]
    advisory_state: str
    composite_budget_pressure: float
    budget_receipt: BudgetReceipt
    normalization_notes: Tuple[str, ...]
    ledger_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ledger_version": self.ledger_version,
            "timing_series": [sample.to_dict() for sample in self.timing_series],
            "throughput_series": [sample.to_dict() for sample in self.throughput_series],
            "budget_requirements": dict(self.budget_requirements),
            "budget_analysis": [metric.to_dict() for metric in self.budget_analysis],
            "advisory_state": self.advisory_state,
            "composite_budget_pressure": self.composite_budget_pressure,
            "budget_receipt": self.budget_receipt.to_dict(),
            "normalization_notes": list(self.normalization_notes),
            "ledger_hash": self.ledger_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.ledger_hash


def _calculate_metrics(
    timing_series: Tuple[LatencyThroughputSample, ...],
    throughput_series: Tuple[LatencyThroughputSample, ...],
    budget_requirements: Mapping[str, float],
) -> Tuple[Tuple[BudgetMetric, ...], float, str]:
    latency_budget = float(budget_requirements["latency_budget_ms"])
    throughput_budget = float(budget_requirements["min_throughput_units"])
    backlog_budget = float(budget_requirements["max_backlog_units"])
    variation_budget = float(budget_requirements["max_timing_variation_ms"])

    latency_values = tuple(sample.latency_ms for sample in timing_series)
    throughput_values = tuple(sample.throughput_units for sample in throughput_series)
    backlog_values = tuple(sample.backlog_units for sample in throughput_series)

    latency_compliance = _mean(tuple(1.0 if value <= latency_budget else 0.0 for value in latency_values), default=1.0)
    throughput_compliance = _mean(tuple(1.0 if value >= throughput_budget else 0.0 for value in throughput_values), default=1.0)

    backlog_pressure = _mean(tuple(_clamp01(value / backlog_budget) for value in backlog_values), default=0.0)

    if len(throughput_values) <= 1:
        scheduling_stability = 1.0
    else:
        deltas = tuple(abs(throughput_values[i] - throughput_values[i - 1]) for i in range(1, len(throughput_values)))
        scheduling_stability = _clamp01(1.0 - _mean(tuple(_clamp01(delta / throughput_budget) for delta in deltas), default=0.0))

    latency_mean = _mean(latency_values, default=0.0)
    variation = _mean(tuple(abs(v - latency_mean) for v in latency_values), default=0.0)
    timing_variation = _clamp01(variation / variation_budget)

    confidence = _clamp01(
        _mean(
            (
                latency_compliance,
                throughput_compliance,
                1.0 - backlog_pressure,
                scheduling_stability,
                1.0 - timing_variation,
            ),
            default=1.0,
        )
    )

    pressure_components = (
        1.0 - latency_compliance,
        1.0 - throughput_compliance,
        backlog_pressure,
        timing_variation,
    )
    composite_pressure = _clamp01(_mean(pressure_components, default=0.0))

    if composite_pressure <= 0.05:
        advisory = "within_budget"
    elif composite_pressure <= 0.20:
        advisory = "near_budget_limit"
    elif composite_pressure <= 0.50:
        advisory = "budget_pressure"
    else:
        advisory = "budget_violation"

    values = {
        "latency_budget_compliance": _clamp01(latency_compliance),
        "throughput_budget_compliance": _clamp01(throughput_compliance),
        "backlog_pressure_score": _clamp01(backlog_pressure),
        "scheduling_stability_score": _clamp01(scheduling_stability),
        "timing_variation_score": _clamp01(timing_variation),
        "budget_confidence_score": _clamp01(confidence),
    }
    metrics = tuple(BudgetMetric(metric_name=name, metric_value=values[name]) for name in METRIC_ORDER)
    return metrics, composite_pressure, advisory


def build_budget_receipt(ledger: LedgerLike) -> BudgetReceipt:
    normalized = ledger if isinstance(ledger, LatencyThroughputBudgetLedger) else run_latency_throughput_budget_ledger(**build_latency_throughput_scenario(ledger))
    timing_exceeded = normalized.advisory_state == "budget_violation"
    receipt_body = {
        "ledger_version": normalized.ledger_version,
        "advisory_state": normalized.advisory_state,
        "logical_replay_identity": "logical-replay-unchanged",
        "logical_outputs_valid": True,
        "timing_budget_exceeded": timing_exceeded,
        "composite_budget_pressure": normalized.composite_budget_pressure,
        "ledger_hash": normalized.ledger_hash,
    }
    receipt_hash = _sha256_hex(_canonical_json(receipt_body).encode("utf-8"))
    return BudgetReceipt(receipt_hash=receipt_hash, **receipt_body)


def build_latency_throughput_scenario(
    payload: Mapping[str, Any] | None = None,
    *,
    timing_series: Sequence[Any] | None = None,
    throughput_series: Sequence[Any] | None = None,
    budget_requirements: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    src = payload if isinstance(payload, Mapping) else {}
    timing_raw = timing_series if timing_series is not None else src.get("timing_series", ())
    throughput_raw = throughput_series if throughput_series is not None else src.get("throughput_series", ())
    requirements_raw = budget_requirements if budget_requirements is not None else src.get("budget_requirements", {})

    timing_norm = _normalize_series(timing_raw)
    throughput_norm = _normalize_series(throughput_raw)
    requirements_norm = _normalize_requirements(requirements_raw)

    notes = []
    for sample in timing_norm + throughput_norm:
        for flag in sample.normalization_flags:
            notes.append(f"{sample.sample_id}:{flag}")

    return {
        "timing_series": timing_norm,
        "throughput_series": throughput_norm,
        "budget_requirements": requirements_norm,
        "normalization_notes": tuple(sorted(notes)),
    }


def run_latency_throughput_budget_ledger(
    timing_series: Sequence[Any] | None = None,
    throughput_series: Sequence[Any] | None = None,
    budget_requirements: Mapping[str, Any] | None = None,
    normalization_notes: Sequence[str] = (),
) -> LatencyThroughputBudgetLedger:
    timing_norm = _normalize_series(timing_series)
    throughput_norm = _normalize_series(throughput_series)
    requirements_norm = _normalize_requirements(budget_requirements)

    metrics, composite_pressure, advisory = _calculate_metrics(timing_norm, throughput_norm, requirements_norm)

    ledger_body = {
        "ledger_version": LEDGER_VERSION,
        "timing_series": [s.to_dict() for s in timing_norm],
        "throughput_series": [s.to_dict() for s in throughput_norm],
        "budget_requirements": dict(requirements_norm),
        "budget_analysis": [m.to_dict() for m in metrics],
        "advisory_state": advisory,
        "composite_budget_pressure": composite_pressure,
        "normalization_notes": list(tuple(sorted(str(n) for n in normalization_notes))),
    }
    ledger_hash = _sha256_hex(_canonical_json(ledger_body).encode("utf-8"))

    partial = LatencyThroughputBudgetLedger(
        ledger_version=LEDGER_VERSION,
        timing_series=timing_norm,
        throughput_series=throughput_norm,
        budget_requirements=requirements_norm,
        budget_analysis=metrics,
        advisory_state=advisory,
        composite_budget_pressure=composite_pressure,
        budget_receipt=BudgetReceipt(
            ledger_version=LEDGER_VERSION,
            advisory_state=advisory,
            logical_replay_identity="logical-replay-unchanged",
            logical_outputs_valid=True,
            timing_budget_exceeded=advisory == "budget_violation",
            composite_budget_pressure=composite_pressure,
            ledger_hash=ledger_hash,
            receipt_hash="",
        ),
        normalization_notes=tuple(sorted(str(n) for n in normalization_notes)),
        ledger_hash=ledger_hash,
    )
    receipt = build_budget_receipt(partial)
    return LatencyThroughputBudgetLedger(
        ledger_version=partial.ledger_version,
        timing_series=partial.timing_series,
        throughput_series=partial.throughput_series,
        budget_requirements=partial.budget_requirements,
        budget_analysis=partial.budget_analysis,
        advisory_state=partial.advisory_state,
        composite_budget_pressure=partial.composite_budget_pressure,
        budget_receipt=receipt,
        normalization_notes=partial.normalization_notes,
        ledger_hash=partial.ledger_hash,
    )


def _as_ledger(raw: LedgerLike) -> LatencyThroughputBudgetLedger:
    if isinstance(raw, LatencyThroughputBudgetLedger):
        return raw
    if isinstance(raw, Mapping):
        scenario = build_latency_throughput_scenario(raw)
        return run_latency_throughput_budget_ledger(**scenario)
    return run_latency_throughput_budget_ledger()


def validate_latency_throughput_budget_ledger(ledger: LedgerLike) -> Dict[str, Any]:
    violations = []
    try:
        normalized = _as_ledger(ledger)
        if normalized.ledger_version != LEDGER_VERSION:
            violations.append("ledger version drift")
        if tuple(metric.metric_name for metric in normalized.budget_analysis) != METRIC_ORDER:
            violations.append("metric ordering drift")
        for metric in normalized.budget_analysis:
            if not (0.0 <= metric.metric_value <= 1.0):
                violations.append(f"metric out of bounds: {metric.metric_name}")
        if normalized.advisory_state not in ADVISORY_STATES:
            violations.append("invalid advisory state")
        if normalized.budget_receipt.ledger_hash != normalized.ledger_hash:
            violations.append("receipt ledger hash mismatch")
        if normalized.budget_receipt.receipt_hash != build_budget_receipt(normalized).receipt_hash:
            violations.append("receipt hash drift")
    except Exception as exc:  # nosec - validator must never raise
        return {
            "is_valid": False,
            "violations": (f"normalization failure: {type(exc).__name__}:{exc}",),
            "ledger_hash": None,
        }

    return {
        "is_valid": len(violations) == 0,
        "violations": tuple(violations),
        "ledger_hash": normalized.ledger_hash,
    }


def compare_budget_replay(left: LedgerLike, right: LedgerLike) -> Dict[str, Any]:
    try:
        left_ledger = _as_ledger(left)
        right_ledger = _as_ledger(right)
    except Exception as exc:
        return {
            "replay_stable": False,
            "violations": (f"normalization failure: {type(exc).__name__}:{exc}",),
            "left_hash": None,
            "right_hash": None,
        }

    violations = []
    if left_ledger.ledger_hash != right_ledger.ledger_hash:
        violations.append("ledger hash mismatch")
    if left_ledger.budget_receipt.receipt_hash != right_ledger.budget_receipt.receipt_hash:
        violations.append("receipt hash mismatch")
    if left_ledger.advisory_state != right_ledger.advisory_state:
        violations.append("advisory mismatch")

    return {
        "replay_stable": len(violations) == 0,
        "violations": tuple(violations),
        "left_hash": left_ledger.ledger_hash,
        "right_hash": right_ledger.ledger_hash,
    }


def summarize_latency_throughput_budget(ledger: LedgerLike) -> Dict[str, Any]:
    normalized = _as_ledger(ledger)
    return {
        "ledger_version": normalized.ledger_version,
        "timing_sample_count": len(normalized.timing_series),
        "throughput_sample_count": len(normalized.throughput_series),
        "advisory_state": normalized.advisory_state,
        "composite_budget_pressure": normalized.composite_budget_pressure,
        "metric_order": METRIC_ORDER,
        "metrics": tuple((metric.metric_name, metric.metric_value) for metric in normalized.budget_analysis),
        "logical_outputs_valid": normalized.budget_receipt.logical_outputs_valid,
        "timing_budget_exceeded": normalized.budget_receipt.timing_budget_exceeded,
        "ledger_hash": normalized.ledger_hash,
    }
