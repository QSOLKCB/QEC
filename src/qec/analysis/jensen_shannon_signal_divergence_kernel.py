"""v137.14.0 — Jensen–Shannon Signal Divergence Kernel.

Deterministic Layer-4 signal divergence with canonical report/receipt export.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import json
import math
from typing import Any, Mapping

SCHEMA_VERSION = "v137.14.0"
_DECIMAL_PLACES = Decimal("0.000000000001")

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]


def _canonicalize_json(value: Any) -> _JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float values are not allowed")
        return value
    if isinstance(value, tuple):
        return tuple(_canonicalize_json(v) for v in value)
    if isinstance(value, list):
        return tuple(_canonicalize_json(v) for v in value)
    if isinstance(value, Mapping):
        keys = tuple(value.keys())
        if any(not isinstance(k, str) for k in keys):
            raise ValueError("payload keys must be strings")
        return {k: _canonicalize_json(value[k]) for k in sorted(keys)}
    raise ValueError(f"unsupported canonical payload type: {type(value)!r}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonicalize_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_hex(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _quantize(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite float")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite float")
    return float(Decimal(str(float(value))).quantize(_DECIMAL_PLACES, rounding=ROUND_HALF_EVEN))


def _quantized_str(value: float, *, name: str) -> str:
    q = Decimal(str(_quantize(value, name=name))).quantize(_DECIMAL_PLACES, rounding=ROUND_HALF_EVEN)
    return str(q)


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


def _normalized_entropy(probabilities: tuple[float, ...]) -> float:
    n = len(probabilities)
    if n <= 1:
        return 0.0
    entropy = 0.0
    for p in probabilities:
        if p > 0.0:
            entropy -= p * math.log2(p)
    denom = math.log2(n)
    if denom <= 0.0:
        return 0.0
    return _clamp01(entropy / denom)


def _validate_distribution(distribution: "SignalDistribution", *, name: str) -> None:
    if not isinstance(distribution, SignalDistribution):
        raise ValueError(f"{name} must be a SignalDistribution")
    if len(distribution.labels) == 0:
        raise ValueError(f"{name} must contain at least one label")
    if len(distribution.labels) != len(distribution.probabilities):
        raise ValueError(f"{name} labels and probabilities length mismatch")
    if distribution.labels != tuple(sorted(distribution.labels)):
        raise ValueError(f"{name} labels must be sorted")

    probs = tuple(float(p) for p in distribution.probabilities)
    for idx, p in enumerate(probs):
        if not math.isfinite(p):
            raise ValueError(f"{name} probabilities[{idx}] must be finite")
        if p < 0.0 or p > 1.0:
            raise ValueError(f"{name} probabilities[{idx}] must be in [0,1]")

    if abs(sum(probs) - 1.0) > 1e-12:
        raise ValueError(f"{name} probabilities must sum to 1 within 1e-12")


@dataclass(frozen=True)
class SignalDistribution:
    labels: tuple[str, ...]
    probabilities: tuple[float, ...]
    distribution_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "labels": self.labels,
            "probabilities": tuple(
                _quantized_str(value, name=f"probabilities[{i}]") for i, value in enumerate(self.probabilities)
            ),
            "distribution_hash": self.distribution_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class SignalDivergenceResult:
    source_distribution_hash: str
    target_distribution_hash: str
    js_divergence_score: float
    distribution_overlap_score: float
    entropy_alignment_score: float
    global_information_geometry_score: float
    result_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "source_distribution_hash": self.source_distribution_hash,
            "target_distribution_hash": self.target_distribution_hash,
            "js_divergence_score": _quantized_str(self.js_divergence_score, name="js_divergence_score"),
            "distribution_overlap_score": _quantized_str(
                self.distribution_overlap_score,
                name="distribution_overlap_score",
            ),
            "entropy_alignment_score": _quantized_str(self.entropy_alignment_score, name="entropy_alignment_score"),
            "global_information_geometry_score": _quantized_str(
                self.global_information_geometry_score,
                name="global_information_geometry_score",
            ),
            "result_hash": self.result_hash,
        }


@dataclass(frozen=True)
class SignalDivergenceReport:
    schema_version: str
    source_distribution: SignalDistribution
    target_distribution: SignalDistribution
    divergence_result: SignalDivergenceResult
    report_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "source_distribution": self.source_distribution.to_dict(),
            "target_distribution": self.target_distribution.to_dict(),
            "divergence_result": self.divergence_result.to_dict(),
            "report_hash": self.report_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class SignalDivergenceReceipt:
    report_hash: str
    byte_length: int
    validation_passed: bool
    receipt_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "report_hash": self.report_hash,
            "byte_length": self.byte_length,
            "validation_passed": self.validation_passed,
            "receipt_hash": self.receipt_hash,
        }


def build_signal_distribution(signal_weights: Mapping[str, float]) -> SignalDistribution:
    if not isinstance(signal_weights, Mapping):
        raise ValueError("signal_weights must be a mapping")
    if len(signal_weights) == 0:
        raise ValueError("signal_weights must be non-empty")

    pairs: list[tuple[str, float]] = []
    for label in sorted(signal_weights.keys()):
        if not isinstance(label, str) or not label:
            raise ValueError("all labels must be non-empty strings")
        raw = signal_weights[label]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError("all signal weights must be finite numbers")
        weight = float(raw)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("all signal weights must be finite and non-negative")
        pairs.append((label, weight))

    total = sum(weight for _, weight in pairs)
    if total <= 0.0:
        raise ValueError("signal weights must have positive total")

    labels = tuple(label for label, _ in pairs)
    probabilities = tuple(_quantize(weight / total, name=f"probability[{i}]") for i, (_, weight) in enumerate(pairs))

    residual = _quantize(1.0 - sum(probabilities), name="residual")
    adjusted = list(probabilities)
    adjusted[-1] = _quantize(adjusted[-1] + residual, name="adjusted_probability")
    normalized_probabilities = tuple(adjusted)

    payload = {
        "labels": labels,
        "probabilities": tuple(_quantized_str(v, name=f"probability[{i}]") for i, v in enumerate(normalized_probabilities)),
    }
    distribution_hash = _sha256_hex(payload)
    dist = SignalDistribution(labels=labels, probabilities=normalized_probabilities, distribution_hash=distribution_hash)
    _validate_distribution(dist, name="distribution")
    if distribution_hash != _sha256_hex({"labels": labels, "probabilities": tuple(_quantized_str(v, name="p") for v in normalized_probabilities)}):
        raise ValueError("distribution_hash mismatch")
    return dist


def compute_jensen_shannon_divergence(
    source_distribution: SignalDistribution,
    target_distribution: SignalDistribution,
) -> SignalDivergenceResult:
    _validate_distribution(source_distribution, name="source_distribution")
    _validate_distribution(target_distribution, name="target_distribution")

    labels = tuple(sorted(set(source_distribution.labels) | set(target_distribution.labels)))
    source_map = dict(zip(source_distribution.labels, source_distribution.probabilities))
    target_map = dict(zip(target_distribution.labels, target_distribution.probabilities))

    p = tuple(source_map.get(label, 0.0) for label in labels)
    q = tuple(target_map.get(label, 0.0) for label in labels)
    m = tuple(0.5 * (pv + qv) for pv, qv in zip(p, q))

    def _kld(a: tuple[float, ...], b: tuple[float, ...]) -> float:
        value = 0.0
        for av, bv in zip(a, b):
            if av > 0.0:
                value += av * math.log2(av / bv)
        return value

    js_divergence = _clamp01(0.5 * _kld(p, m) + 0.5 * _kld(q, m))
    overlap = _clamp01(sum(min(pv, qv) for pv, qv in zip(p, q)))
    entropy_alignment = _clamp01(1.0 - abs(_normalized_entropy(p) - _normalized_entropy(q)))
    global_score = _clamp01((1.0 - js_divergence + overlap + entropy_alignment) / 3.0)

    payload = {
        "source_distribution_hash": source_distribution.distribution_hash,
        "target_distribution_hash": target_distribution.distribution_hash,
        "js_divergence_score": _quantized_str(js_divergence, name="js_divergence_score"),
        "distribution_overlap_score": _quantized_str(overlap, name="distribution_overlap_score"),
        "entropy_alignment_score": _quantized_str(entropy_alignment, name="entropy_alignment_score"),
        "global_information_geometry_score": _quantized_str(global_score, name="global_information_geometry_score"),
    }
    return SignalDivergenceResult(
        source_distribution_hash=source_distribution.distribution_hash,
        target_distribution_hash=target_distribution.distribution_hash,
        js_divergence_score=_quantize(js_divergence, name="js_divergence_score"),
        distribution_overlap_score=_quantize(overlap, name="distribution_overlap_score"),
        entropy_alignment_score=_quantize(entropy_alignment, name="entropy_alignment_score"),
        global_information_geometry_score=_quantize(global_score, name="global_information_geometry_score"),
        result_hash=_sha256_hex(payload),
    )


def run_signal_divergence_kernel(
    source_signal_weights: Mapping[str, float],
    target_signal_weights: Mapping[str, float],
) -> tuple[SignalDivergenceReport, SignalDivergenceReceipt]:
    source_distribution = build_signal_distribution(source_signal_weights)
    target_distribution = build_signal_distribution(target_signal_weights)
    result = compute_jensen_shannon_divergence(source_distribution, target_distribution)

    report_payload = {
        "schema_version": SCHEMA_VERSION,
        "source_distribution": source_distribution.to_dict(),
        "target_distribution": target_distribution.to_dict(),
        "divergence_result": result.to_dict(),
    }
    report_hash = _sha256_hex(report_payload)
    report = SignalDivergenceReport(
        schema_version=SCHEMA_VERSION,
        source_distribution=source_distribution,
        target_distribution=target_distribution,
        divergence_result=result,
        report_hash=report_hash,
    )

    report_bytes = report.to_canonical_bytes()
    receipt_payload = {
        "report_hash": report.report_hash,
        "byte_length": len(report_bytes),
        "validation_passed": True,
    }
    receipt = SignalDivergenceReceipt(
        report_hash=report.report_hash,
        byte_length=len(report_bytes),
        validation_passed=True,
        receipt_hash=_sha256_hex(receipt_payload),
    )
    return report, receipt
