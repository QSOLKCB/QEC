"""v147.3.0 — Retro Trace Forecast Kernel."""

from __future__ import annotations

from dataclasses import dataclass

from qec.analysis.canonical_hashing import canonical_bytes, canonical_json, sha256_hex
from qec.analysis.closed_loop_simulation_kernel import round12, validate_sha256_hex, validate_unit_interval
from qec.analysis.retro_trace_intake_bridge import RetroTraceReceipt

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

_MIN_HORIZON = 1
_MAX_HORIZON = 256
_CLASS_STABLE = "STABLE"
_CLASS_DRIFT = "DRIFT"
_CLASS_UNSTABLE = "UNSTABLE"


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return round12(value)


def _extract_features(retro_trace: RetroTraceReceipt) -> dict[str, float]:
    metrics = dict(retro_trace.trace_metrics)
    timing = retro_trace.normalized_timing
    trace_length = max(1, retro_trace.trace_length)
    timing_density = _clamp01(float(len(timing)) / float(trace_length))
    sparsity = _clamp01(float(metrics.get("input_sparsity", 1.0)))
    ordering_integrity = _clamp01(float(metrics.get("event_order_integrity", 0.0)))

    max_cycle = int(timing[-1]) if timing else 0
    event_rate = _clamp01(float(retro_trace.trace_length) / float(max_cycle + 1))

    if len(timing) >= 2:
        gradients = tuple(float(timing[idx] - timing[idx - 1]) for idx in range(1, len(timing)))
        span = float(max(1, timing[-1] - timing[0]))
        normalized_gradients = tuple((value * float(len(gradients))) / span for value in gradients)
        grad_mean = _clamp01(sum(normalized_gradients) / float(len(normalized_gradients)))
        grad_var = sum((value - grad_mean) ** 2 for value in normalized_gradients) / float(len(normalized_gradients))
        gradient_stability = _clamp01(1.0 - min(1.0, grad_var))
        avg_delta = sum(gradients) / float(len(gradients))
    else:
        grad_mean = 0.0
        gradient_stability = 1.0 if timing else 0.0
        avg_delta = 1.0

    return {
        "timing_density": timing_density,
        "sparsity": sparsity,
        "ordering_integrity": ordering_integrity,
        "event_rate": event_rate,
        "normalized_timing_gradient": grad_mean,
        "gradient_stability": gradient_stability,
        "average_timing_delta": float(max(1.0, avg_delta)),
        "last_timing": float(max_cycle),
    }


@dataclass(frozen=True)
class RetroTraceForecastStep:
    step_index: int
    projected_timing: int
    projected_event_density: float
    stability_score: float
    _stable_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.step_index, int) or self.step_index < 1:
            raise ValueError("step_index must be positive int")
        if not isinstance(self.projected_timing, int) or self.projected_timing < 0:
            raise ValueError("projected_timing must be non-negative int")
        object.__setattr__(
            self,
            "projected_event_density",
            _clamp01(
                validate_unit_interval(
                    self.projected_event_density,
                    "projected_event_density",
                )
            ),
        )
        object.__setattr__(
            self,
            "stability_score",
            _clamp01(
                validate_unit_interval(
                    self.stability_score,
                    "stability_score",
                )
            ),
        )
        if self._stable_hash != sha256_hex(self._payload_without_hash()):
            raise ValueError("stable_hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "step_index": self.step_index,
            "projected_timing": self.projected_timing,
            "projected_event_density": round12(self.projected_event_density),
            "stability_score": round12(self.stability_score),
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return self._stable_hash


@dataclass(frozen=True)
class RetroTraceForecastSeries:
    horizon: int
    features: tuple[tuple[str, float], ...]
    steps: tuple[RetroTraceForecastStep, ...]
    _stable_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.horizon, int) or not (_MIN_HORIZON <= self.horizon <= _MAX_HORIZON):
            raise ValueError(f"horizon must be in [{_MIN_HORIZON},{_MAX_HORIZON}]")
        if not isinstance(self.features, tuple):
            raise ValueError("features must be tuple")
        expected_keys = (
            "average_timing_delta",
            "event_rate",
            "gradient_stability",
            "last_timing",
            "normalized_timing_gradient",
            "ordering_integrity",
            "sparsity",
            "timing_density",
        )
        observed_keys = tuple(name for name, _ in self.features)
        if observed_keys != expected_keys:
            raise ValueError("features must use canonical deterministic key ordering")
        normalized_features = []
        for name, value in self.features:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"features[{name}] must be numeric")
            normalized_features.append((name, round12(float(value))))
        object.__setattr__(self, "features", tuple(normalized_features))
        if not isinstance(self.steps, tuple) or len(self.steps) != self.horizon:
            raise ValueError("steps length must equal horizon")
        if any(not isinstance(step, RetroTraceForecastStep) for step in self.steps):
            raise ValueError("steps contains invalid item")
        for idx, step in enumerate(self.steps, start=1):
            if step.step_index != idx:
                raise ValueError("steps must use contiguous canonical ordering")
            if idx > 1 and step.projected_timing < self.steps[idx - 2].projected_timing:
                raise ValueError("projected_timing must be monotonic")
        if self._stable_hash != sha256_hex(self._payload_without_hash()):
            raise ValueError("stable_hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "horizon": self.horizon,
            "features": tuple((k, round12(v)) for k, v in self.features),
            "steps": tuple(item.to_dict() for item in self.steps),
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return self._stable_hash


@dataclass(frozen=True)
class RetroTraceForecastSummary:
    overall_stability_forecast: float
    collapse_risk_classification: str
    _stable_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "overall_stability_forecast",
            validate_unit_interval(self.overall_stability_forecast, "overall_stability_forecast"),
        )
        if self.collapse_risk_classification not in (_CLASS_STABLE, _CLASS_DRIFT, _CLASS_UNSTABLE):
            raise ValueError("collapse_risk_classification must be STABLE|DRIFT|UNSTABLE")
        if self._stable_hash != sha256_hex(self._payload_without_hash()):
            raise ValueError("stable_hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "overall_stability_forecast": round12(self.overall_stability_forecast),
            "collapse_risk_classification": self.collapse_risk_classification,
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return self._stable_hash


@dataclass(frozen=True)
class RetroTraceForecastReceipt:
    retro_trace_hash: str
    series: RetroTraceForecastSeries
    summary: RetroTraceForecastSummary
    _stable_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "retro_trace_hash", validate_sha256_hex(self.retro_trace_hash, "retro_trace_hash"))
        if not isinstance(self.series, RetroTraceForecastSeries):
            raise ValueError("series must be RetroTraceForecastSeries")
        if not isinstance(self.summary, RetroTraceForecastSummary):
            raise ValueError("summary must be RetroTraceForecastSummary")
        if self.summary.collapse_risk_classification != _classify(self.summary.overall_stability_forecast):
            raise ValueError("summary classification mismatch")
        recomputed = _clamp01(
            sum(step.stability_score for step in self.series.steps) / float(max(1, len(self.series.steps)))
        )
        if round12(recomputed) != round12(self.summary.overall_stability_forecast):
            raise ValueError("summary overall_stability_forecast mismatch")
        if self._stable_hash != sha256_hex(self._payload_without_hash()):
            raise ValueError("stable_hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "retro_trace_hash": self.retro_trace_hash,
            "series": self.series.to_dict(),
            "summary": self.summary.to_dict(),
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return self._stable_hash


def _classify(stability: float) -> str:
    if stability >= 0.68:
        return _CLASS_STABLE
    if stability >= 0.50:
        return _CLASS_DRIFT
    return _CLASS_UNSTABLE


def forecast_retro_trace(
    retro_trace: RetroTraceReceipt,
    horizon: int,
) -> RetroTraceForecastReceipt:
    if not isinstance(retro_trace, RetroTraceReceipt):
        raise ValueError("retro_trace must be RetroTraceReceipt")
    if not isinstance(horizon, int) or not (_MIN_HORIZON <= horizon <= _MAX_HORIZON):
        raise ValueError(f"horizon must be int in [{_MIN_HORIZON},{_MAX_HORIZON}]")
    retro_payload = retro_trace.to_dict()
    observed_hash = str(retro_payload.pop("stable_hash"))
    if observed_hash != sha256_hex(retro_payload):
        raise ValueError("retro_trace stable_hash mismatch")

    features = _extract_features(retro_trace)
    initial_density = features["timing_density"]
    target_density = _clamp01((1.0 - features["sparsity"] + features["event_rate"] + features["ordering_integrity"]) / 3.0)
    density_trend = round12((target_density - initial_density) * 0.18)

    baseline_delta = max(1, int(round(features["average_timing_delta"] * (1.0 + features["normalized_timing_gradient"] * 0.5))))
    delta_growth = (1.0 - features["sparsity"]) * 1.25 + features["event_rate"] * 0.5
    decay = 0.90 + (0.08 * features["gradient_stability"])

    projected_time = int(round(features["last_timing"]))
    current_density = initial_density
    current_delta = baseline_delta
    steps = []
    for step_index in range(1, horizon + 1):
        current_delta = max(1, int(round((float(current_delta) * decay) + delta_growth)))
        projected_time = projected_time + current_delta

        current_density = _clamp01(current_density + density_trend)
        density_alignment = _clamp01(1.0 - abs(current_density - target_density))
        base_stability = (
            features["ordering_integrity"] * 0.20
            + features["gradient_stability"] * 0.20
            + (1.0 - features["sparsity"]) * 0.25
            + density_alignment * 0.15
            + features["event_rate"] * 0.20
        )
        stability = _clamp01(base_stability - (0.015 * float(step_index - 1)))

        step_payload = {
            "step_index": step_index,
            "projected_timing": projected_time,
            "projected_event_density": round12(current_density),
            "stability_score": round12(stability),
        }
        steps.append(
            RetroTraceForecastStep(
                step_index=step_index,
                projected_timing=projected_time,
                projected_event_density=current_density,
                stability_score=stability,
                _stable_hash=sha256_hex(step_payload),
            )
        )

    series_features = tuple(sorted((name, round12(value)) for name, value in features.items()))
    series_payload = {
        "horizon": horizon,
        "features": series_features,
        "steps": tuple(step.to_dict() for step in steps),
    }
    series = RetroTraceForecastSeries(
        horizon=horizon,
        features=series_features,
        steps=tuple(steps),
        _stable_hash=sha256_hex(series_payload),
    )

    overall_stability = _clamp01(sum(step.stability_score for step in steps) / float(len(steps)))
    summary_payload = {
        "overall_stability_forecast": round12(overall_stability),
        "collapse_risk_classification": _classify(overall_stability),
    }
    summary = RetroTraceForecastSummary(
        overall_stability_forecast=overall_stability,
        collapse_risk_classification=_classify(overall_stability),
        _stable_hash=sha256_hex(summary_payload),
    )

    receipt_payload = {
        "retro_trace_hash": retro_trace.stable_hash,
        "series": series.to_dict(),
        "summary": summary.to_dict(),
    }
    return RetroTraceForecastReceipt(
        retro_trace_hash=retro_trace.stable_hash,
        series=series,
        summary=summary,
        _stable_hash=sha256_hex(receipt_payload),
    )


__all__ = [
    "RetroTraceForecastStep",
    "RetroTraceForecastSeries",
    "RetroTraceForecastSummary",
    "RetroTraceForecastReceipt",
    "forecast_retro_trace",
]
