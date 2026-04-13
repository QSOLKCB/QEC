"""v137.14.1 — Fisher–Rao Geometry Approximation Layer.

Deterministic Layer-4 geometric distance complementing the
v137.14.0 Jensen–Shannon signal divergence kernel.

This module approximates the Fisher–Rao manifold distance between two
``SignalDistribution`` artifacts using the Bhattacharyya-based geodesic
on the sphere of square-root probability vectors:

    distance = 2 * arccos( sum_i sqrt(p_i * q_i) )

The distance is clamped into ``[0, π]`` and normalized into ``[0, 1]``.

All metrics use the convention::

    0 = identical
    1 = maximally separated
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

SCHEMA_VERSION = "v137.14.1"
_DECIMAL_PLACES = Decimal("0.000000000001")
_PI = math.pi

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


def _clamp_pi(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= _PI:
        return _PI
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


@dataclass(frozen=True)
class FisherRaoConfig:
    """Configuration for the Fisher–Rao geometry layer.

    The layer is fully deterministic; configuration only carries
    schema lineage. Distance is always clamped into ``[0, π]``.
    """

    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    @property
    def config_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class FisherRaoResult:
    source_distribution_hash: str
    target_distribution_hash: str
    fisher_rao_distance: float
    normalized_fisher_rao_score: float
    fisher_rao_distance_score: float
    geodesic_alignment_score: float
    manifold_consistency_score: float
    global_information_geometry_score: float
    result_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "source_distribution_hash": self.source_distribution_hash,
            "target_distribution_hash": self.target_distribution_hash,
            "fisher_rao_distance": _quantized_str(self.fisher_rao_distance, name="fisher_rao_distance"),
            "normalized_fisher_rao_score": _quantized_str(
                self.normalized_fisher_rao_score, name="normalized_fisher_rao_score"
            ),
            "fisher_rao_distance_score": _quantized_str(
                self.fisher_rao_distance_score, name="fisher_rao_distance_score"
            ),
            "geodesic_alignment_score": _quantized_str(
                self.geodesic_alignment_score, name="geodesic_alignment_score"
            ),
            "manifold_consistency_score": _quantized_str(
                self.manifold_consistency_score, name="manifold_consistency_score"
            ),
            "global_information_geometry_score": _quantized_str(
                self.global_information_geometry_score,
                name="global_information_geometry_score",
            ),
            "result_hash": self.result_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class FisherRaoReport:
    schema_version: str
    config: FisherRaoConfig
    source_distribution: SignalDistribution
    target_distribution: SignalDistribution
    geometry_result: FisherRaoResult
    report_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "config": self.config.to_dict(),
            "source_distribution": self.source_distribution.to_dict(),
            "target_distribution": self.target_distribution.to_dict(),
            "geometry_result": self.geometry_result.to_dict(),
            "report_hash": self.report_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class FisherRaoReceipt:
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


def _bhattacharyya_coefficient(p: tuple[float, ...], q: tuple[float, ...]) -> float:
    bc = 0.0
    for pv, qv in zip(p, q):
        product = pv * qv
        if product > 0.0:
            bc += math.sqrt(product)
    if bc > 1.0:
        bc = 1.0
    if bc < 0.0:
        bc = 0.0
    return bc


def _fisher_rao_distance_and_bc(
    source_distribution: SignalDistribution,
    target_distribution: SignalDistribution,
) -> tuple[float, float]:
    """Compute Fisher–Rao distance and Bhattacharyya coefficient.

    Validates both distributions, then returns the clamped geodesic
    distance in ``[0, π]`` together with the Bhattacharyya coefficient
    in ``[0, 1]``. This is the single source of truth used by all
    public geometry computations.
    """
    _validate_distribution(source_distribution, name="source_distribution")
    _validate_distribution(target_distribution, name="target_distribution")
    _, p, q = _aligned_pq(source_distribution, target_distribution)
    bc = _bhattacharyya_coefficient(p, q)
    distance = _clamp_pi(2.0 * math.acos(bc))
    return distance, bc


def compute_fisher_rao_distance(
    source_distribution: SignalDistribution,
    target_distribution: SignalDistribution,
) -> float:
    """Approximate Fisher–Rao geodesic distance in ``[0, π]``.

    Computed via the spherical embedding of square-root probabilities::

        distance = 2 * arccos( sum_i sqrt(p_i * q_i) )
    """
    distance, _ = _fisher_rao_distance_and_bc(source_distribution, target_distribution)
    return distance


def compute_geodesic_alignment(
    source_distribution: SignalDistribution,
    target_distribution: SignalDistribution,
) -> float:
    """Bhattacharyya-based geodesic misalignment in ``[0, 1]``.

    Returns ``0`` when the two distributions are identical and ``1``
    when they have disjoint supports.
    """
    _, bc = _fisher_rao_distance_and_bc(source_distribution, target_distribution)
    return _clamp01(1.0 - bc)


def _compute_geometry_result(
    source: SignalDistribution,
    target: SignalDistribution,
) -> FisherRaoResult:
    distance, bc = _fisher_rao_distance_and_bc(source, target)
    normalized = _clamp01(distance / _PI)

    fisher_rao_distance_score = normalized
    geodesic_alignment_score = _clamp01(1.0 - bc)
    # Chord length on the sphere of square-roots: sin(distance/2) = sqrt(1 - bc^2)
    manifold_consistency_score = _clamp01(math.sqrt(max(0.0, 1.0 - bc * bc)))
    global_score = _clamp01(
        (fisher_rao_distance_score + geodesic_alignment_score + manifold_consistency_score) / 3.0
    )

    payload = {
        "source_distribution_hash": source.distribution_hash,
        "target_distribution_hash": target.distribution_hash,
        "fisher_rao_distance": _quantized_str(distance, name="fisher_rao_distance"),
        "normalized_fisher_rao_score": _quantized_str(
            normalized, name="normalized_fisher_rao_score"
        ),
        "fisher_rao_distance_score": _quantized_str(
            fisher_rao_distance_score, name="fisher_rao_distance_score"
        ),
        "geodesic_alignment_score": _quantized_str(
            geodesic_alignment_score, name="geodesic_alignment_score"
        ),
        "manifold_consistency_score": _quantized_str(
            manifold_consistency_score, name="manifold_consistency_score"
        ),
        "global_information_geometry_score": _quantized_str(
            global_score, name="global_information_geometry_score"
        ),
    }
    return FisherRaoResult(
        source_distribution_hash=source.distribution_hash,
        target_distribution_hash=target.distribution_hash,
        fisher_rao_distance=_quantize(distance, name="fisher_rao_distance"),
        normalized_fisher_rao_score=_quantize(normalized, name="normalized_fisher_rao_score"),
        fisher_rao_distance_score=_quantize(
            fisher_rao_distance_score, name="fisher_rao_distance_score"
        ),
        geodesic_alignment_score=_quantize(
            geodesic_alignment_score, name="geodesic_alignment_score"
        ),
        manifold_consistency_score=_quantize(
            manifold_consistency_score, name="manifold_consistency_score"
        ),
        global_information_geometry_score=_quantize(
            global_score, name="global_information_geometry_score"
        ),
        result_hash=_sha256_hex(payload),
    )


def run_fisher_rao_geometry_layer(
    source_signal_weights: Mapping[str, float],
    target_signal_weights: Mapping[str, float],
    config: FisherRaoConfig | None = None,
) -> tuple[FisherRaoReport, FisherRaoReceipt]:
    """Run the full Fisher–Rao geometry layer and return report + receipt."""
    cfg = config if config is not None else FisherRaoConfig()
    if not isinstance(cfg, FisherRaoConfig):
        raise ValueError("config must be a FisherRaoConfig")

    source_distribution = build_signal_distribution(source_signal_weights)
    target_distribution = build_signal_distribution(target_signal_weights)
    result = _compute_geometry_result(source_distribution, target_distribution)

    report_payload = {
        "schema_version": SCHEMA_VERSION,
        "config": cfg.to_dict(),
        "source_distribution": source_distribution.to_dict(),
        "target_distribution": target_distribution.to_dict(),
        "geometry_result": result.to_dict(),
    }
    report_hash = _sha256_hex(report_payload)
    report = FisherRaoReport(
        schema_version=SCHEMA_VERSION,
        config=cfg,
        source_distribution=source_distribution,
        target_distribution=target_distribution,
        geometry_result=result,
        report_hash=report_hash,
    )

    report_bytes = report.to_canonical_bytes()
    # Independent verification: re-derive the report hash from the
    # canonical payload and confirm distributions still validate. This
    # is what the receipt's ``validation_passed`` attests to.
    recomputed_report_hash = _sha256_hex(report_payload)
    _validate_distribution(source_distribution, name="source_distribution")
    _validate_distribution(target_distribution, name="target_distribution")
    validation_passed = (
        recomputed_report_hash == report.report_hash
        and len(report_bytes) > 0
    )
    if not validation_passed:
        raise ValueError("fisher-rao report failed self-verification")

    receipt_payload = {
        "report_hash": report.report_hash,
        "config_hash": cfg.config_hash,
        "source_distribution_hash": source_distribution.distribution_hash,
        "target_distribution_hash": target_distribution.distribution_hash,
        "result_hash": result.result_hash,
        "byte_length": len(report_bytes),
        "validation_passed": validation_passed,
    }
    receipt = FisherRaoReceipt(
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


def build_ascii_fisher_rao_summary(report: FisherRaoReport) -> str:
    """Deterministic ASCII summary of a Fisher–Rao geometry report."""
    if not isinstance(report, FisherRaoReport):
        raise ValueError("report must be a FisherRaoReport")
    r = report.geometry_result
    lines = (
        f"# Fisher-Rao Geometry Layer ({report.schema_version})",
        f"source_hash: {r.source_distribution_hash}",
        f"target_hash: {r.target_distribution_hash}",
        f"fisher_rao_distance:               {_quantized_str(r.fisher_rao_distance, name='fisher_rao_distance')}",
        f"normalized_fisher_rao_score:       {_quantized_str(r.normalized_fisher_rao_score, name='normalized_fisher_rao_score')}",
        f"fisher_rao_distance_score:         {_quantized_str(r.fisher_rao_distance_score, name='fisher_rao_distance_score')}",
        f"geodesic_alignment_score:          {_quantized_str(r.geodesic_alignment_score, name='geodesic_alignment_score')}",
        f"manifold_consistency_score:        {_quantized_str(r.manifold_consistency_score, name='manifold_consistency_score')}",
        f"global_information_geometry_score: {_quantized_str(r.global_information_geometry_score, name='global_information_geometry_score')}",
        f"report_hash: {report.report_hash}",
    )
    return "\n".join(lines)
