"""v137.14.3 — Optimal Transport Geometry Kernel.

Deterministic Layer-4 transport-distance kernel that complements:

* v137.14.0 ``jensen_shannon_signal_divergence_kernel``
* v137.14.1 ``fisher_rao_geometry_approximation_layer``
* v137.14.2 ``bregman_f_divergence_correspondence_engine``

This layer approximates the Wasserstein-1 / Earth Mover transport
distance between two valid :class:`SignalDistribution` artifacts using
the classical one-dimensional closed form over ordered (sorted-label)
bins::

    W1 = Σ_i |CDF_p(i) - CDF_q(i)| / (N - 1)

All exported metrics follow the engine-wide convention::

    0 = identical
    1 = maximally separated

The ``compute_transport_alignment`` primitive returns the raw
``1 - W1`` alignment value (``1 = perfectly aligned``, ``0 = maximally
misaligned``), while the ``transport_alignment_score`` exported in the
result is its complement so that, like every other score in the
transport-geometry result, ``0`` marks identical distributions and
``1`` marks maximally separated distributions.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import json
import math
from typing import Any, Mapping

from qec.analysis.jensen_shannon_signal_divergence_kernel import (
    SignalDistribution,
    build_signal_distribution,
)

SCHEMA_VERSION = "v137.14.3"
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


def _quantized_decimal(value: float, *, name: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite float")
    coerced = float(value)
    if not math.isfinite(coerced):
        raise ValueError(f"{name} must be a finite float")
    return Decimal(str(coerced)).quantize(_DECIMAL_PLACES, rounding=ROUND_HALF_EVEN)


def _quantize(value: float, *, name: str) -> float:
    return float(_quantized_decimal(value, name=name))


def _quantized_str(value: float, *, name: str) -> str:
    return str(_quantized_decimal(value, name=name))


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


def _validate_distribution(distribution: SignalDistribution, *, name: str) -> None:
    if not isinstance(distribution, SignalDistribution):
        raise ValueError(f"{name} must be a SignalDistribution")
    if len(distribution.labels) == 0:
        raise ValueError(f"{name} must contain at least one label")
    if len(distribution.labels) != len(distribution.probabilities):
        raise ValueError(f"{name} labels and probabilities length mismatch")
    if distribution.labels != tuple(sorted(distribution.labels)):
        raise ValueError(f"{name} labels must be sorted")
    if len(set(distribution.labels)) != len(distribution.labels):
        raise ValueError(f"{name} labels must be unique")

    probs = tuple(float(p) for p in distribution.probabilities)
    for idx, p in enumerate(probs):
        if not math.isfinite(p):
            raise ValueError(f"{name} probabilities[{idx}] must be finite")
        if p < 0.0 or p > 1.0:
            raise ValueError(f"{name} probabilities[{idx}] must be in [0,1]")

    if abs(sum(probs) - 1.0) > 1e-12:
        raise ValueError(f"{name} probabilities must sum to 1 within 1e-12")


def _aligned_pq(
    source: SignalDistribution,
    target: SignalDistribution,
) -> tuple[tuple[str, ...], tuple[float, ...], tuple[float, ...]]:
    labels = tuple(sorted(set(source.labels) | set(target.labels)))
    smap = dict(zip(source.labels, source.probabilities))
    tmap = dict(zip(target.labels, target.probabilities))
    p = tuple(float(smap.get(label, 0.0)) for label in labels)
    q = tuple(float(tmap.get(label, 0.0)) for label in labels)
    return labels, p, q


def _cdf(probabilities: tuple[float, ...]) -> tuple[float, ...]:
    cdf: list[float] = []
    running = 0.0
    for p in probabilities:
        running += p
        cdf.append(running)
    # Clamp the terminal value to exactly 1.0 to neutralize floating
    # accumulation noise; every distribution has already been validated
    # to sum to 1 within 1e-12.
    if cdf:
        cdf[-1] = 1.0
    return tuple(cdf)


def _cdf_absolute_differences(
    p: tuple[float, ...],
    q: tuple[float, ...],
) -> tuple[float, ...]:
    cdf_p = _cdf(p)
    cdf_q = _cdf(q)
    return tuple(abs(a - b) for a, b in zip(cdf_p, cdf_q))


# ------------------------------------------------------------------
# Dataclasses
# ------------------------------------------------------------------


@dataclass(frozen=True)
class TransportGeometryConfig:
    """Configuration for the optimal transport geometry kernel.

    The kernel is fully deterministic; configuration only carries
    schema lineage so receipts chain cleanly with upstream divergence
    engines.
    """

    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, _JSONValue]:
        return {"schema_version": self.schema_version}

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    @property
    def config_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class TransportGeometryResult:
    source_distribution_hash: str
    target_distribution_hash: str
    wasserstein_distance_score: float
    transport_alignment_score: float
    cumulative_flow_consistency_score: float
    global_transport_geometry_score: float
    result_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "source_distribution_hash": self.source_distribution_hash,
            "target_distribution_hash": self.target_distribution_hash,
            "wasserstein_distance_score": _quantized_str(
                self.wasserstein_distance_score, name="wasserstein_distance_score"
            ),
            "transport_alignment_score": _quantized_str(
                self.transport_alignment_score, name="transport_alignment_score"
            ),
            "cumulative_flow_consistency_score": _quantized_str(
                self.cumulative_flow_consistency_score,
                name="cumulative_flow_consistency_score",
            ),
            "global_transport_geometry_score": _quantized_str(
                self.global_transport_geometry_score,
                name="global_transport_geometry_score",
            ),
            "result_hash": self.result_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class TransportGeometryReport:
    schema_version: str
    config: TransportGeometryConfig
    source_distribution: SignalDistribution
    target_distribution: SignalDistribution
    transport_result: TransportGeometryResult
    report_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "config": self.config.to_dict(),
            "source_distribution": self.source_distribution.to_dict(),
            "target_distribution": self.target_distribution.to_dict(),
            "transport_result": self.transport_result.to_dict(),
            "report_hash": self.report_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class TransportGeometryReceipt:
    report_hash: str
    config_hash: str
    source_distribution_hash: str
    target_distribution_hash: str
    result_hash: str
    byte_length: int
    validation_passed: bool
    receipt_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "report_hash": self.report_hash,
            "config_hash": self.config_hash,
            "source_distribution_hash": self.source_distribution_hash,
            "target_distribution_hash": self.target_distribution_hash,
            "result_hash": self.result_hash,
            "byte_length": self.byte_length,
            "validation_passed": self.validation_passed,
            "receipt_hash": self.receipt_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


# ------------------------------------------------------------------
# Public transport primitives
# ------------------------------------------------------------------


def compute_wasserstein_distance(
    source_distribution: SignalDistribution,
    target_distribution: SignalDistribution,
) -> float:
    """Normalized Wasserstein-1 / Earth Mover distance in ``[0, 1]``.

    Uses the classical one-dimensional closed form::

        W1 = Σ_i |CDF_p(i) - CDF_q(i)| / (N - 1)

    over the union of the source and target labels in sorted order.
    The sum has ``N`` non-negative terms bounded by ``1`` each, with
    the terminal term identically zero (both CDFs reach ``1``), so the
    numerator is bounded above by ``N - 1`` and the normalized score
    sits in ``[0, 1]``. Symmetric in its arguments, ``0`` when the
    distributions are identical, ``1`` when they are maximally
    separated (disjoint indicator supports at opposite ends).

    A degenerate single-bin distribution has ``N = 1`` and both
    distributions collapse to the same point mass; the normalized
    distance is defined to be exactly ``0`` in that case.
    """
    _validate_distribution(source_distribution, name="source_distribution")
    _validate_distribution(target_distribution, name="target_distribution")
    _, p, q = _aligned_pq(source_distribution, target_distribution)
    n = len(p)
    if n <= 1:
        return 0.0
    diffs = _cdf_absolute_differences(p, q)
    raw = sum(diffs) / float(n - 1)
    return _clamp01(raw)


def compute_transport_alignment(
    source_distribution: SignalDistribution,
    target_distribution: SignalDistribution,
) -> float:
    """Raw transport alignment value ``1 - W1``.

    Uses :func:`compute_wasserstein_distance` and returns its
    complement. This is the raw *alignment* convention:

    * ``1`` — perfectly aligned (identical distributions)
    * ``0`` — maximally misaligned (disjoint indicator supports)

    The complementary ``transport_alignment_score`` exported in
    :class:`TransportGeometryResult` is ``1 - this value`` so that it
    follows the engine-wide ``0 = identical`` convention.
    """
    return _clamp01(1.0 - compute_wasserstein_distance(source_distribution, target_distribution))


# ------------------------------------------------------------------
# Engine
# ------------------------------------------------------------------


def _compute_transport_result(
    source: SignalDistribution,
    target: SignalDistribution,
) -> TransportGeometryResult:
    # Delegate Wasserstein to the public primitive so the kernel and
    # the standalone transport function share a single source of
    # truth. ``compute_transport_alignment`` returns the *raw*
    # alignment (``1 = identical``), so we take its complement to
    # obtain the engine-convention score where ``0 = identical``.
    wasserstein_score = compute_wasserstein_distance(source, target)
    alignment_score = _clamp01(1.0 - compute_transport_alignment(source, target))

    # Cumulative-flow consistency signal: the Kolmogorov–Smirnov
    # supremum ``max_i |CDF_p(i) - CDF_q(i)|``. ``0`` means the
    # cumulative flows are identical, ``1`` means maximally separated.
    _, p, q = _aligned_pq(source, target)
    if len(p) == 0:
        consistency_raw = 0.0
    else:
        diffs = _cdf_absolute_differences(p, q)
        consistency_raw = max(diffs) if diffs else 0.0
    consistency_score = _clamp01(consistency_raw)

    global_score = _clamp01(
        (wasserstein_score + alignment_score + consistency_score) / 3.0
    )

    payload = {
        "source_distribution_hash": source.distribution_hash,
        "target_distribution_hash": target.distribution_hash,
        "wasserstein_distance_score": _quantized_str(
            wasserstein_score, name="wasserstein_distance_score"
        ),
        "transport_alignment_score": _quantized_str(
            alignment_score, name="transport_alignment_score"
        ),
        "cumulative_flow_consistency_score": _quantized_str(
            consistency_score, name="cumulative_flow_consistency_score"
        ),
        "global_transport_geometry_score": _quantized_str(
            global_score, name="global_transport_geometry_score"
        ),
    }
    return TransportGeometryResult(
        source_distribution_hash=source.distribution_hash,
        target_distribution_hash=target.distribution_hash,
        wasserstein_distance_score=_quantize(
            wasserstein_score, name="wasserstein_distance_score"
        ),
        transport_alignment_score=_quantize(
            alignment_score, name="transport_alignment_score"
        ),
        cumulative_flow_consistency_score=_quantize(
            consistency_score, name="cumulative_flow_consistency_score"
        ),
        global_transport_geometry_score=_quantize(
            global_score, name="global_transport_geometry_score"
        ),
        result_hash=_sha256_hex(payload),
    )


def run_optimal_transport_geometry_kernel(
    source_signal_weights: Mapping[str, float],
    target_signal_weights: Mapping[str, float],
    config: TransportGeometryConfig | None = None,
) -> tuple[TransportGeometryReport, TransportGeometryReceipt]:
    """Run the full optimal transport geometry kernel.

    Returns a ``(report, receipt)`` pair. The receipt's
    ``validation_passed`` attests that the report hash was
    independently re-derived from the report object's own canonical
    form, mirroring how an external consumer would derive it.
    """
    cfg = config if config is not None else TransportGeometryConfig()
    if not isinstance(cfg, TransportGeometryConfig):
        raise ValueError("config must be a TransportGeometryConfig")

    # ``build_signal_distribution`` validates its output internally, so
    # we do not re-validate the distributions here.
    source_distribution = build_signal_distribution(source_signal_weights)
    target_distribution = build_signal_distribution(target_signal_weights)
    result = _compute_transport_result(source_distribution, target_distribution)

    report_payload = {
        "schema_version": SCHEMA_VERSION,
        "config": cfg.to_dict(),
        "source_distribution": source_distribution.to_dict(),
        "target_distribution": target_distribution.to_dict(),
        "transport_result": result.to_dict(),
    }
    report = TransportGeometryReport(
        schema_version=SCHEMA_VERSION,
        config=cfg,
        source_distribution=source_distribution,
        target_distribution=target_distribution,
        transport_result=result,
        report_hash=_sha256_hex(report_payload),
    )

    # Round-trip self-verification: derive the hash from the report
    # object's own canonical form (with its self-referential
    # ``report_hash`` field stripped), exactly as an external consumer
    # would. This catches any drift between the dataclass and the
    # payload that was hashed.
    report_bytes = report.to_canonical_bytes()
    round_trip_payload = {
        k: v for k, v in report.to_dict().items() if k != "report_hash"
    }
    recomputed_report_hash = _sha256_hex(round_trip_payload)
    validation_passed = (
        recomputed_report_hash == report.report_hash and len(report_bytes) > 0
    )
    if not validation_passed:
        raise ValueError(
            "optimal transport geometry report failed self-verification"
        )

    receipt_payload = {
        "report_hash": report.report_hash,
        "config_hash": cfg.config_hash,
        "source_distribution_hash": source_distribution.distribution_hash,
        "target_distribution_hash": target_distribution.distribution_hash,
        "result_hash": result.result_hash,
        "byte_length": len(report_bytes),
        "validation_passed": validation_passed,
    }
    receipt = TransportGeometryReceipt(
        report_hash=report.report_hash,
        config_hash=cfg.config_hash,
        source_distribution_hash=source_distribution.distribution_hash,
        target_distribution_hash=target_distribution.distribution_hash,
        result_hash=result.result_hash,
        byte_length=len(report_bytes),
        validation_passed=validation_passed,
        receipt_hash=_sha256_hex(receipt_payload),
    )
    return report, receipt


def build_ascii_transport_summary(
    report: TransportGeometryReport,
) -> str:
    """Deterministic ASCII summary of a transport geometry report."""
    if not isinstance(report, TransportGeometryReport):
        raise ValueError("report must be a TransportGeometryReport")
    r = report.transport_result
    lines = (
        f"# Optimal Transport Geometry Kernel ({report.schema_version})",
        f"source_hash: {r.source_distribution_hash}",
        f"target_hash: {r.target_distribution_hash}",
        f"wasserstein_distance_score:         {_quantized_str(r.wasserstein_distance_score, name='wasserstein_distance_score')}",
        f"transport_alignment_score:          {_quantized_str(r.transport_alignment_score, name='transport_alignment_score')}",
        f"cumulative_flow_consistency_score:  {_quantized_str(r.cumulative_flow_consistency_score, name='cumulative_flow_consistency_score')}",
        f"global_transport_geometry_score:    {_quantized_str(r.global_transport_geometry_score, name='global_transport_geometry_score')}",
        f"report_hash: {report.report_hash}",
    )
    return "\n".join(lines)
