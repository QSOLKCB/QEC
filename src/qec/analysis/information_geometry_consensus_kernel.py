"""v137.14.4 — Information Geometry Consensus Kernel.

Deterministic Layer-4 consensus kernel that aggregates geometry-layer
scores and anchors a replay-safe receipt chain to upstream receipts.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import json
import math
from typing import Any, Mapping

SCHEMA_VERSION = "v137.14.4"
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


def _validate_score(value: float, *, name: str) -> float:
    score = _quantize(value, name=name)
    if score < 0.0 or score > 1.0:
        raise ValueError(f"{name} must be in [0,1]")
    return score


def _validate_receipt_hash(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a hex string")
    if len(value) != 64:
        raise ValueError(f"{name} must be 64 hex chars")
    if any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be lowercase hex")
    return value


@dataclass(frozen=True)
class InformationGeometryConsensusConfig:
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
class InformationGeometryConsensusResult:
    js_divergence_score: float
    fisher_rao_distance_score: float
    global_divergence_correspondence_score: float
    global_transport_geometry_score: float
    geometry_consensus_score: float
    geometry_dispersion_score: float
    manifold_agreement_score: float
    geometry_stability_score: float
    global_information_consensus_score: float
    result_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "js_divergence_score": _quantized_str(self.js_divergence_score, name="js_divergence_score"),
            "fisher_rao_distance_score": _quantized_str(
                self.fisher_rao_distance_score, name="fisher_rao_distance_score"
            ),
            "global_divergence_correspondence_score": _quantized_str(
                self.global_divergence_correspondence_score,
                name="global_divergence_correspondence_score",
            ),
            "global_transport_geometry_score": _quantized_str(
                self.global_transport_geometry_score, name="global_transport_geometry_score"
            ),
            "geometry_consensus_score": _quantized_str(
                self.geometry_consensus_score, name="geometry_consensus_score"
            ),
            "geometry_dispersion_score": _quantized_str(
                self.geometry_dispersion_score, name="geometry_dispersion_score"
            ),
            "manifold_agreement_score": _quantized_str(
                self.manifold_agreement_score, name="manifold_agreement_score"
            ),
            "geometry_stability_score": _quantized_str(
                self.geometry_stability_score, name="geometry_stability_score"
            ),
            "global_information_consensus_score": _quantized_str(
                self.global_information_consensus_score,
                name="global_information_consensus_score",
            ),
            "result_hash": self.result_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class InformationGeometryConsensusReport:
    schema_version: str
    config: InformationGeometryConsensusConfig
    js_divergence_receipt_hash: str
    fisher_rao_receipt_hash: str
    divergence_correspondence_receipt_hash: str
    transport_geometry_receipt_hash: str
    consensus_result: InformationGeometryConsensusResult
    report_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "config": self.config.to_dict(),
            "js_divergence_receipt_hash": self.js_divergence_receipt_hash,
            "fisher_rao_receipt_hash": self.fisher_rao_receipt_hash,
            "divergence_correspondence_receipt_hash": self.divergence_correspondence_receipt_hash,
            "transport_geometry_receipt_hash": self.transport_geometry_receipt_hash,
            "consensus_result": self.consensus_result.to_dict(),
            "report_hash": self.report_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class InformationGeometryConsensusReceipt:
    report_hash: str
    config_hash: str
    js_divergence_receipt_hash: str
    fisher_rao_receipt_hash: str
    divergence_correspondence_receipt_hash: str
    transport_geometry_receipt_hash: str
    result_hash: str
    byte_length: int
    validation_passed: bool
    receipt_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "report_hash": self.report_hash,
            "config_hash": self.config_hash,
            "js_divergence_receipt_hash": self.js_divergence_receipt_hash,
            "fisher_rao_receipt_hash": self.fisher_rao_receipt_hash,
            "divergence_correspondence_receipt_hash": self.divergence_correspondence_receipt_hash,
            "transport_geometry_receipt_hash": self.transport_geometry_receipt_hash,
            "result_hash": self.result_hash,
            "byte_length": self.byte_length,
            "validation_passed": self.validation_passed,
            "receipt_hash": self.receipt_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


def _compute_consensus_result(
    js_divergence_score: float,
    fisher_rao_distance_score: float,
    global_divergence_correspondence_score: float,
    global_transport_geometry_score: float,
) -> InformationGeometryConsensusResult:
    js = _validate_score(js_divergence_score, name="js_divergence_score")
    fr = _validate_score(fisher_rao_distance_score, name="fisher_rao_distance_score")
    dc = _validate_score(
        global_divergence_correspondence_score,
        name="global_divergence_correspondence_score",
    )
    ot = _validate_score(global_transport_geometry_score, name="global_transport_geometry_score")

    values = (js, fr, dc, ot)
    # Convert divergence-style inputs (0=identical, 1=maximally separated) into
    # alignment space (0=maximally separated, 1=identical) before aggregating so
    # that higher consensus scores indicate stronger agreement, not divergence.
    alignment_values = tuple(1.0 - v for v in values)
    consensus = _clamp01(sum(alignment_values) / 4.0)
    mean = consensus
    mad = sum(abs(v - mean) for v in alignment_values) / 4.0
    dispersion = _clamp01(2.0 * mad)
    agreement = _clamp01(1.0 - dispersion)
    stability = _clamp01(1.0 - (max(alignment_values) - min(alignment_values)))
    global_consensus = _clamp01((consensus + agreement + stability) / 3.0)

    payload = {
        "js_divergence_score": _quantized_str(js, name="js_divergence_score"),
        "fisher_rao_distance_score": _quantized_str(fr, name="fisher_rao_distance_score"),
        "global_divergence_correspondence_score": _quantized_str(
            dc, name="global_divergence_correspondence_score"
        ),
        "global_transport_geometry_score": _quantized_str(
            ot, name="global_transport_geometry_score"
        ),
        "geometry_consensus_score": _quantized_str(consensus, name="geometry_consensus_score"),
        "geometry_dispersion_score": _quantized_str(dispersion, name="geometry_dispersion_score"),
        "manifold_agreement_score": _quantized_str(agreement, name="manifold_agreement_score"),
        "geometry_stability_score": _quantized_str(stability, name="geometry_stability_score"),
        "global_information_consensus_score": _quantized_str(
            global_consensus, name="global_information_consensus_score"
        ),
    }

    return InformationGeometryConsensusResult(
        js_divergence_score=js,
        fisher_rao_distance_score=fr,
        global_divergence_correspondence_score=dc,
        global_transport_geometry_score=ot,
        geometry_consensus_score=_quantize(consensus, name="geometry_consensus_score"),
        geometry_dispersion_score=_quantize(dispersion, name="geometry_dispersion_score"),
        manifold_agreement_score=_quantize(agreement, name="manifold_agreement_score"),
        geometry_stability_score=_quantize(stability, name="geometry_stability_score"),
        global_information_consensus_score=_quantize(
            global_consensus, name="global_information_consensus_score"
        ),
        result_hash=_sha256_hex(payload),
    )


def run_information_geometry_consensus_kernel(
    js_divergence_score: float,
    fisher_rao_distance_score: float,
    global_divergence_correspondence_score: float,
    global_transport_geometry_score: float,
    js_divergence_receipt_hash: str,
    fisher_rao_receipt_hash: str,
    divergence_correspondence_receipt_hash: str,
    transport_geometry_receipt_hash: str,
    config: InformationGeometryConsensusConfig | None = None,
) -> tuple[InformationGeometryConsensusReport, InformationGeometryConsensusReceipt]:
    cfg = config if config is not None else InformationGeometryConsensusConfig()
    if not isinstance(cfg, InformationGeometryConsensusConfig):
        raise ValueError("config must be an InformationGeometryConsensusConfig")

    js_receipt = _validate_receipt_hash(js_divergence_receipt_hash, name="js_divergence_receipt_hash")
    fr_receipt = _validate_receipt_hash(fisher_rao_receipt_hash, name="fisher_rao_receipt_hash")
    dc_receipt = _validate_receipt_hash(
        divergence_correspondence_receipt_hash,
        name="divergence_correspondence_receipt_hash",
    )
    ot_receipt = _validate_receipt_hash(
        transport_geometry_receipt_hash,
        name="transport_geometry_receipt_hash",
    )

    result = _compute_consensus_result(
        js_divergence_score=js_divergence_score,
        fisher_rao_distance_score=fisher_rao_distance_score,
        global_divergence_correspondence_score=global_divergence_correspondence_score,
        global_transport_geometry_score=global_transport_geometry_score,
    )

    report_payload = {
        "schema_version": SCHEMA_VERSION,
        "config": cfg.to_dict(),
        "js_divergence_receipt_hash": js_receipt,
        "fisher_rao_receipt_hash": fr_receipt,
        "divergence_correspondence_receipt_hash": dc_receipt,
        "transport_geometry_receipt_hash": ot_receipt,
        "consensus_result": result.to_dict(),
    }
    report = InformationGeometryConsensusReport(
        schema_version=SCHEMA_VERSION,
        config=cfg,
        js_divergence_receipt_hash=js_receipt,
        fisher_rao_receipt_hash=fr_receipt,
        divergence_correspondence_receipt_hash=dc_receipt,
        transport_geometry_receipt_hash=ot_receipt,
        consensus_result=result,
        report_hash=_sha256_hex(report_payload),
    )

    report_bytes = report.to_canonical_bytes()
    round_trip_payload = {k: v for k, v in report.to_dict().items() if k != "report_hash"}
    recomputed_report_hash = _sha256_hex(round_trip_payload)
    validation_passed = recomputed_report_hash == report.report_hash and len(report_bytes) > 0
    if not validation_passed:
        raise ValueError("information geometry consensus report failed self-verification")

    receipt_payload = {
        "report_hash": report.report_hash,
        "config_hash": cfg.config_hash,
        "js_divergence_receipt_hash": js_receipt,
        "fisher_rao_receipt_hash": fr_receipt,
        "divergence_correspondence_receipt_hash": dc_receipt,
        "transport_geometry_receipt_hash": ot_receipt,
        "result_hash": result.result_hash,
        "byte_length": len(report_bytes),
        "validation_passed": validation_passed,
    }
    receipt = InformationGeometryConsensusReceipt(
        report_hash=report.report_hash,
        config_hash=cfg.config_hash,
        js_divergence_receipt_hash=js_receipt,
        fisher_rao_receipt_hash=fr_receipt,
        divergence_correspondence_receipt_hash=dc_receipt,
        transport_geometry_receipt_hash=ot_receipt,
        result_hash=result.result_hash,
        byte_length=len(report_bytes),
        validation_passed=validation_passed,
        receipt_hash=_sha256_hex(receipt_payload),
    )
    return report, receipt


def build_ascii_information_geometry_consensus_summary(
    report: InformationGeometryConsensusReport,
) -> str:
    """Return a deterministic ASCII summary for a consensus report."""
    if not isinstance(report, InformationGeometryConsensusReport):
        raise ValueError("report must be an InformationGeometryConsensusReport")
    r = report.consensus_result
    lines = (
        f"# Information Geometry Consensus Kernel ({report.schema_version})",
        f"js_divergence_receipt_hash: {report.js_divergence_receipt_hash}",
        f"fisher_rao_receipt_hash: {report.fisher_rao_receipt_hash}",
        f"divergence_correspondence_receipt_hash: {report.divergence_correspondence_receipt_hash}",
        f"transport_geometry_receipt_hash: {report.transport_geometry_receipt_hash}",
        f"geometry_consensus_score:           {_quantized_str(r.geometry_consensus_score, name='geometry_consensus_score')}",
        f"geometry_dispersion_score:          {_quantized_str(r.geometry_dispersion_score, name='geometry_dispersion_score')}",
        f"manifold_agreement_score:           {_quantized_str(r.manifold_agreement_score, name='manifold_agreement_score')}",
        f"geometry_stability_score:           {_quantized_str(r.geometry_stability_score, name='geometry_stability_score')}",
        f"global_information_consensus_score: {_quantized_str(r.global_information_consensus_score, name='global_information_consensus_score')}",
        f"report_hash: {report.report_hash}",
    )
    return "\n".join(lines)
