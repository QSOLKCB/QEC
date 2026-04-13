"""v137.14.2 — Bregman / f-Divergence Correspondence Engine.

Deterministic Layer-4 correspondence layer that maps several divergence
families onto a single bounded comparison space.

This module complements:

* v137.14.0 ``jensen_shannon_signal_divergence_kernel``
* v137.14.1 ``fisher_rao_geometry_approximation_layer``

It exports KL divergence, total variation distance, and a symmetric
Bregman / f-divergence alignment signal through a single canonical
report + receipt pipeline. All exported metrics follow the convention::

    0 = identical
    1 = maximally separated

The ``compute_bregman_alignment`` primitive returns the raw
``1 - normalized_KL`` alignment value (``1 = perfectly aligned``,
``0 = maximally misaligned``) as specified mathematically, while the
``bregman_alignment_score`` exported in the result is its complement so
that, like every other score in the correspondence result, ``0``
marks identical distributions and ``1`` marks maximally separated
distributions.
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

SCHEMA_VERSION = "v137.14.2"
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
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    labels = tuple(sorted(set(source.labels) | set(target.labels)))
    smap = dict(zip(source.labels, source.probabilities))
    tmap = dict(zip(target.labels, target.probabilities))
    p = tuple(float(smap.get(label, 0.0)) for label in labels)
    q = tuple(float(tmap.get(label, 0.0)) for label in labels)
    return p, q


def _raw_kl_bits(p: tuple[float, ...], q: tuple[float, ...]) -> float:
    """Forward KL divergence ``D_KL(p || q)`` in bits.

    Returns ``math.inf`` if any index has ``p_i > 0`` while ``q_i == 0``.
    Returns exactly ``0.0`` when the distributions are identical.
    """
    kl = 0.0
    for pv, qv in zip(p, q):
        if pv > 0.0:
            if qv <= 0.0:
                return math.inf
            kl += pv * math.log2(pv / qv)
    if kl < 0.0:
        return 0.0
    return kl


def _normalize_kl(kl_bits: float) -> float:
    """Map ``[0, +inf]`` KL (bits) into ``[0, 1]`` via ``1 - 2**(-kl)``.

    ``kl = 0`` maps to ``0``. ``kl = +inf`` maps to ``1``. The mapping
    is strictly monotone so ordering between KL values is preserved.
    """
    if math.isinf(kl_bits) and kl_bits > 0.0:
        return 1.0
    if kl_bits <= 0.0:
        return 0.0
    if not math.isfinite(kl_bits):
        raise ValueError("kl_bits must be finite or +inf")
    return _clamp01(1.0 - math.pow(2.0, -kl_bits))


def _symmetric_kl_bits(p: tuple[float, ...], q: tuple[float, ...]) -> float:
    forward = _raw_kl_bits(p, q)
    reverse = _raw_kl_bits(q, p)
    if math.isinf(forward) or math.isinf(reverse):
        return math.inf
    return 0.5 * (forward + reverse)


# ------------------------------------------------------------------
# Dataclasses
# ------------------------------------------------------------------


@dataclass(frozen=True)
class DivergenceCorrespondenceConfig:
    """Configuration for the correspondence engine.

    The engine is fully deterministic; configuration only carries
    schema lineage so receipts chain cleanly.
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
class DivergenceCorrespondenceResult:
    source_distribution_hash: str
    target_distribution_hash: str
    kl_divergence_score: float
    total_variation_score: float
    bregman_alignment_score: float
    divergence_family_consistency_score: float
    global_divergence_correspondence_score: float
    result_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "source_distribution_hash": self.source_distribution_hash,
            "target_distribution_hash": self.target_distribution_hash,
            "kl_divergence_score": _quantized_str(
                self.kl_divergence_score, name="kl_divergence_score"
            ),
            "total_variation_score": _quantized_str(
                self.total_variation_score, name="total_variation_score"
            ),
            "bregman_alignment_score": _quantized_str(
                self.bregman_alignment_score, name="bregman_alignment_score"
            ),
            "divergence_family_consistency_score": _quantized_str(
                self.divergence_family_consistency_score,
                name="divergence_family_consistency_score",
            ),
            "global_divergence_correspondence_score": _quantized_str(
                self.global_divergence_correspondence_score,
                name="global_divergence_correspondence_score",
            ),
            "result_hash": self.result_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class DivergenceCorrespondenceReport:
    schema_version: str
    config: DivergenceCorrespondenceConfig
    source_distribution: SignalDistribution
    target_distribution: SignalDistribution
    correspondence_result: DivergenceCorrespondenceResult
    report_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "config": self.config.to_dict(),
            "source_distribution": self.source_distribution.to_dict(),
            "target_distribution": self.target_distribution.to_dict(),
            "correspondence_result": self.correspondence_result.to_dict(),
            "report_hash": self.report_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class DivergenceCorrespondenceReceipt:
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
# Public divergence primitives
# ------------------------------------------------------------------


def compute_kl_divergence(
    source_distribution: SignalDistribution,
    target_distribution: SignalDistribution,
) -> float:
    """Normalized KL divergence score in ``[0, 1]``.

    Computes the forward KL divergence ``D_KL(p || q) = Σ p_i log2(p_i / q_i)``
    in bits and maps it into ``[0, 1]`` via the bijection
    ``1 - 2**(-kl)``. Returns ``0`` when the distributions are
    identical and ``1`` when the target places zero mass where the
    source has positive mass (infinite KL).
    """
    _validate_distribution(source_distribution, name="source_distribution")
    _validate_distribution(target_distribution, name="target_distribution")
    p, q = _aligned_pq(source_distribution, target_distribution)
    return _normalize_kl(_raw_kl_bits(p, q))


def compute_total_variation_distance(
    source_distribution: SignalDistribution,
    target_distribution: SignalDistribution,
) -> float:
    """Total variation distance ``0.5 * Σ |p_i - q_i|`` in ``[0, 1]``.

    Returns ``0`` for identical distributions and ``1`` for
    distributions with disjoint supports. Symmetric in its arguments.
    """
    _validate_distribution(source_distribution, name="source_distribution")
    _validate_distribution(target_distribution, name="target_distribution")
    p, q = _aligned_pq(source_distribution, target_distribution)
    return _clamp01(0.5 * sum(abs(pv - qv) for pv, qv in zip(p, q)))


def compute_bregman_alignment(
    source_distribution: SignalDistribution,
    target_distribution: SignalDistribution,
) -> float:
    """Raw Bregman / f-divergence alignment value ``1 - normalized_KL``.

    Uses the symmetric KL Bregman divergence::

        B(p, q) = 0.5 * (D_KL(p || q) + D_KL(q || p))

    and returns ``1 - _normalize_kl(B)``. This is the raw *alignment*
    convention from the spec:

    * ``1`` — perfectly aligned (identical distributions)
    * ``0`` — maximally misaligned

    The complementary ``bregman_alignment_score`` exported in
    :class:`DivergenceCorrespondenceResult` is ``1 - this value`` so
    that it follows the engine-wide ``0 = identical`` convention.
    """
    _validate_distribution(source_distribution, name="source_distribution")
    _validate_distribution(target_distribution, name="target_distribution")
    p, q = _aligned_pq(source_distribution, target_distribution)
    return _clamp01(1.0 - _normalize_kl(_symmetric_kl_bits(p, q)))


# ------------------------------------------------------------------
# Engine
# ------------------------------------------------------------------


def _compute_correspondence_result(
    source: SignalDistribution,
    target: SignalDistribution,
) -> DivergenceCorrespondenceResult:
    p, q = _aligned_pq(source, target)

    kl_score = _normalize_kl(_raw_kl_bits(p, q))
    tv_score = _clamp01(0.5 * sum(abs(pv - qv) for pv, qv in zip(p, q)))
    bregman_score = _normalize_kl(_symmetric_kl_bits(p, q))

    # Family-consistency signal: range of the three divergence-family
    # scores. ``0`` means perfect agreement, ``1`` means maximal
    # disagreement. It is not averaged into the global score; it is a
    # separate meta-signal exported alongside it.
    family_range = max(kl_score, tv_score, bregman_score) - min(
        kl_score, tv_score, bregman_score
    )
    consistency_score = _clamp01(family_range)

    global_score = _clamp01((kl_score + tv_score + bregman_score) / 3.0)

    payload = {
        "source_distribution_hash": source.distribution_hash,
        "target_distribution_hash": target.distribution_hash,
        "kl_divergence_score": _quantized_str(kl_score, name="kl_divergence_score"),
        "total_variation_score": _quantized_str(
            tv_score, name="total_variation_score"
        ),
        "bregman_alignment_score": _quantized_str(
            bregman_score, name="bregman_alignment_score"
        ),
        "divergence_family_consistency_score": _quantized_str(
            consistency_score, name="divergence_family_consistency_score"
        ),
        "global_divergence_correspondence_score": _quantized_str(
            global_score, name="global_divergence_correspondence_score"
        ),
    }
    return DivergenceCorrespondenceResult(
        source_distribution_hash=source.distribution_hash,
        target_distribution_hash=target.distribution_hash,
        kl_divergence_score=_quantize(kl_score, name="kl_divergence_score"),
        total_variation_score=_quantize(tv_score, name="total_variation_score"),
        bregman_alignment_score=_quantize(
            bregman_score, name="bregman_alignment_score"
        ),
        divergence_family_consistency_score=_quantize(
            consistency_score, name="divergence_family_consistency_score"
        ),
        global_divergence_correspondence_score=_quantize(
            global_score, name="global_divergence_correspondence_score"
        ),
        result_hash=_sha256_hex(payload),
    )


def run_divergence_correspondence_engine(
    source_signal_weights: Mapping[str, float],
    target_signal_weights: Mapping[str, float],
    config: DivergenceCorrespondenceConfig | None = None,
) -> tuple[DivergenceCorrespondenceReport, DivergenceCorrespondenceReceipt]:
    """Run the full Bregman / f-divergence correspondence engine.

    Returns a ``(report, receipt)`` pair. The receipt's ``validation_passed``
    attests that the report hash was independently re-derived from the
    canonical payload and that both distributions pass validation.
    """
    cfg = config if config is not None else DivergenceCorrespondenceConfig()
    if not isinstance(cfg, DivergenceCorrespondenceConfig):
        raise ValueError("config must be a DivergenceCorrespondenceConfig")

    source_distribution = build_signal_distribution(source_signal_weights)
    target_distribution = build_signal_distribution(target_signal_weights)
    result = _compute_correspondence_result(source_distribution, target_distribution)

    report_payload = {
        "schema_version": SCHEMA_VERSION,
        "config": cfg.to_dict(),
        "source_distribution": source_distribution.to_dict(),
        "target_distribution": target_distribution.to_dict(),
        "correspondence_result": result.to_dict(),
    }
    report_hash = _sha256_hex(report_payload)
    report = DivergenceCorrespondenceReport(
        schema_version=SCHEMA_VERSION,
        config=cfg,
        source_distribution=source_distribution,
        target_distribution=target_distribution,
        correspondence_result=result,
        report_hash=report_hash,
    )

    report_bytes = report.to_canonical_bytes()
    recomputed_report_hash = _sha256_hex(report_payload)
    _validate_distribution(source_distribution, name="source_distribution")
    _validate_distribution(target_distribution, name="target_distribution")
    validation_passed = (
        recomputed_report_hash == report.report_hash and len(report_bytes) > 0
    )
    if not validation_passed:
        raise ValueError(
            "divergence correspondence report failed self-verification"
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
    receipt = DivergenceCorrespondenceReceipt(
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


def build_ascii_divergence_correspondence_summary(
    report: DivergenceCorrespondenceReport,
) -> str:
    """Deterministic ASCII summary of a correspondence report."""
    if not isinstance(report, DivergenceCorrespondenceReport):
        raise ValueError("report must be a DivergenceCorrespondenceReport")
    r = report.correspondence_result
    lines = (
        f"# Bregman / f-Divergence Correspondence Engine ({report.schema_version})",
        f"source_hash: {r.source_distribution_hash}",
        f"target_hash: {r.target_distribution_hash}",
        f"kl_divergence_score:                    {_quantized_str(r.kl_divergence_score, name='kl_divergence_score')}",
        f"total_variation_score:                  {_quantized_str(r.total_variation_score, name='total_variation_score')}",
        f"bregman_alignment_score:                {_quantized_str(r.bregman_alignment_score, name='bregman_alignment_score')}",
        f"divergence_family_consistency_score:    {_quantized_str(r.divergence_family_consistency_score, name='divergence_family_consistency_score')}",
        f"global_divergence_correspondence_score: {_quantized_str(r.global_divergence_correspondence_score, name='global_divergence_correspondence_score')}",
        f"report_hash: {report.report_hash}",
    )
    return "\n".join(lines)
