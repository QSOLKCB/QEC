"""v137.14.5 — Information-Geometric Certification Pack.

Deterministic Layer-4 certification capstone for the v137.14.x
information-geometry ladder.

This module consumes already-computed, normalized supervisory metrics and
produces a replay-safe certification result, report, and receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import json
import math
from typing import Any, Mapping

SCHEMA_VERSION = "v137.14.5"
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
class InformationGeometricCertificationInput:
    js_divergence_score: float
    fisher_rao_distance_score: float
    global_divergence_correspondence_score: float
    global_transport_geometry_score: float
    global_information_consensus_score: float
    js_divergence_receipt_hash: str
    fisher_rao_receipt_hash: str
    divergence_correspondence_receipt_hash: str
    transport_geometry_receipt_hash: str
    consensus_receipt_hash: str
    geometry_stability_score: float = 1.0
    manifold_agreement_score: float = 1.0
    coverage_ratio: float = 1.0

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "js_divergence_score": _quantized_str(self.js_divergence_score, name="js_divergence_score"),
            "fisher_rao_distance_score": _quantized_str(self.fisher_rao_distance_score, name="fisher_rao_distance_score"),
            "global_divergence_correspondence_score": _quantized_str(
                self.global_divergence_correspondence_score,
                name="global_divergence_correspondence_score",
            ),
            "global_transport_geometry_score": _quantized_str(
                self.global_transport_geometry_score,
                name="global_transport_geometry_score",
            ),
            "global_information_consensus_score": _quantized_str(
                self.global_information_consensus_score,
                name="global_information_consensus_score",
            ),
            "js_divergence_receipt_hash": self.js_divergence_receipt_hash,
            "fisher_rao_receipt_hash": self.fisher_rao_receipt_hash,
            "divergence_correspondence_receipt_hash": self.divergence_correspondence_receipt_hash,
            "transport_geometry_receipt_hash": self.transport_geometry_receipt_hash,
            "consensus_receipt_hash": self.consensus_receipt_hash,
            "geometry_stability_score": _quantized_str(self.geometry_stability_score, name="geometry_stability_score"),
            "manifold_agreement_score": _quantized_str(self.manifold_agreement_score, name="manifold_agreement_score"),
            "coverage_ratio": _quantized_str(self.coverage_ratio, name="coverage_ratio"),
        }

    def as_hash_payload(self) -> dict[str, _JSONValue]:
        return self.to_dict()


@dataclass(frozen=True)
class InformationGeometricCertificationConfig:
    schema_version: str = SCHEMA_VERSION
    divergence_consistency_weight: float = 0.24
    manifold_consistency_weight: float = 0.22
    transport_consistency_weight: float = 0.18
    consensus_certainty_weight: float = 0.20
    coverage_completeness_weight: float = 0.16

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "divergence_consistency_weight": _quantized_str(
                self.divergence_consistency_weight,
                name="divergence_consistency_weight",
            ),
            "manifold_consistency_weight": _quantized_str(
                self.manifold_consistency_weight,
                name="manifold_consistency_weight",
            ),
            "transport_consistency_weight": _quantized_str(
                self.transport_consistency_weight,
                name="transport_consistency_weight",
            ),
            "consensus_certainty_weight": _quantized_str(
                self.consensus_certainty_weight,
                name="consensus_certainty_weight",
            ),
            "coverage_completeness_weight": _quantized_str(
                self.coverage_completeness_weight,
                name="coverage_completeness_weight",
            ),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def as_hash_payload(self) -> dict[str, _JSONValue]:
        return self.to_dict()

    @property
    def config_hash(self) -> str:
        return _sha256_hex(self.as_hash_payload())


@dataclass(frozen=True)
class InformationGeometricCertificationResult:
    divergence_consistency_score: float
    manifold_consistency_score: float
    transport_consistency_score: float
    consensus_certainty_score: float
    coverage_completeness_score: float
    global_information_geometry_certification_score: float
    result_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "divergence_consistency_score": _quantized_str(
                self.divergence_consistency_score,
                name="divergence_consistency_score",
            ),
            "manifold_consistency_score": _quantized_str(
                self.manifold_consistency_score,
                name="manifold_consistency_score",
            ),
            "transport_consistency_score": _quantized_str(
                self.transport_consistency_score,
                name="transport_consistency_score",
            ),
            "consensus_certainty_score": _quantized_str(
                self.consensus_certainty_score,
                name="consensus_certainty_score",
            ),
            "coverage_completeness_score": _quantized_str(
                self.coverage_completeness_score,
                name="coverage_completeness_score",
            ),
            "global_information_geometry_certification_score": _quantized_str(
                self.global_information_geometry_certification_score,
                name="global_information_geometry_certification_score",
            ),
            "result_hash": self.result_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def as_hash_payload(self) -> dict[str, _JSONValue]:
        payload = self.to_dict().copy()
        payload.pop("result_hash", None)
        return payload


@dataclass(frozen=True)
class InformationGeometricCertificationReport:
    schema_version: str
    config: InformationGeometricCertificationConfig
    certification_input: InformationGeometricCertificationInput
    certification_result: InformationGeometricCertificationResult
    summary_text: str
    report_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "config": self.config.to_dict(),
            "certification_input": self.certification_input.to_dict(),
            "certification_result": self.certification_result.to_dict(),
            "summary_text": self.summary_text,
            "report_hash": self.report_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def as_hash_payload(self) -> dict[str, _JSONValue]:
        payload = self.to_dict().copy()
        payload.pop("report_hash", None)
        return payload


@dataclass(frozen=True)
class InformationGeometricCertificationReceipt:
    report_hash: str
    config_hash: str
    js_divergence_receipt_hash: str
    fisher_rao_receipt_hash: str
    divergence_correspondence_receipt_hash: str
    transport_geometry_receipt_hash: str
    consensus_receipt_hash: str
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
            "consensus_receipt_hash": self.consensus_receipt_hash,
            "result_hash": self.result_hash,
            "byte_length": self.byte_length,
            "validation_passed": self.validation_passed,
            "receipt_hash": self.receipt_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def as_hash_payload(self) -> dict[str, _JSONValue]:
        payload = self.to_dict().copy()
        payload.pop("receipt_hash", None)
        return payload


_REQUIRED_FIELDS = (
    "js_divergence_score",
    "fisher_rao_distance_score",
    "global_divergence_correspondence_score",
    "global_transport_geometry_score",
    "global_information_consensus_score",
    "js_divergence_receipt_hash",
    "fisher_rao_receipt_hash",
    "divergence_correspondence_receipt_hash",
    "transport_geometry_receipt_hash",
    "consensus_receipt_hash",
)


def _normalize_input(
    certification_input: InformationGeometricCertificationInput | Mapping[str, Any],
) -> InformationGeometricCertificationInput:
    if isinstance(certification_input, InformationGeometricCertificationInput):
        candidate = certification_input
    elif isinstance(certification_input, Mapping):
        missing = tuple(name for name in _REQUIRED_FIELDS if name not in certification_input)
        if missing:
            raise ValueError(f"missing required fields: {', '.join(missing)}")
        candidate = InformationGeometricCertificationInput(
            js_divergence_score=certification_input["js_divergence_score"],
            fisher_rao_distance_score=certification_input["fisher_rao_distance_score"],
            global_divergence_correspondence_score=certification_input[
                "global_divergence_correspondence_score"
            ],
            global_transport_geometry_score=certification_input[
                "global_transport_geometry_score"
            ],
            global_information_consensus_score=certification_input[
                "global_information_consensus_score"
            ],
            js_divergence_receipt_hash=certification_input["js_divergence_receipt_hash"],
            fisher_rao_receipt_hash=certification_input["fisher_rao_receipt_hash"],
            divergence_correspondence_receipt_hash=certification_input[
                "divergence_correspondence_receipt_hash"
            ],
            transport_geometry_receipt_hash=certification_input[
                "transport_geometry_receipt_hash"
            ],
            consensus_receipt_hash=certification_input["consensus_receipt_hash"],
            geometry_stability_score=certification_input.get("geometry_stability_score", 1.0),
            manifold_agreement_score=certification_input.get("manifold_agreement_score", 1.0),
            coverage_ratio=certification_input.get("coverage_ratio", 1.0),
        )
    else:
        raise ValueError("certification_input must be InformationGeometricCertificationInput or mapping")

    return InformationGeometricCertificationInput(
        js_divergence_score=_validate_score(candidate.js_divergence_score, name="js_divergence_score"),
        fisher_rao_distance_score=_validate_score(
            candidate.fisher_rao_distance_score,
            name="fisher_rao_distance_score",
        ),
        global_divergence_correspondence_score=_validate_score(
            candidate.global_divergence_correspondence_score,
            name="global_divergence_correspondence_score",
        ),
        global_transport_geometry_score=_validate_score(
            candidate.global_transport_geometry_score,
            name="global_transport_geometry_score",
        ),
        global_information_consensus_score=_validate_score(
            candidate.global_information_consensus_score,
            name="global_information_consensus_score",
        ),
        js_divergence_receipt_hash=_validate_receipt_hash(
            candidate.js_divergence_receipt_hash,
            name="js_divergence_receipt_hash",
        ),
        fisher_rao_receipt_hash=_validate_receipt_hash(
            candidate.fisher_rao_receipt_hash,
            name="fisher_rao_receipt_hash",
        ),
        divergence_correspondence_receipt_hash=_validate_receipt_hash(
            candidate.divergence_correspondence_receipt_hash,
            name="divergence_correspondence_receipt_hash",
        ),
        transport_geometry_receipt_hash=_validate_receipt_hash(
            candidate.transport_geometry_receipt_hash,
            name="transport_geometry_receipt_hash",
        ),
        consensus_receipt_hash=_validate_receipt_hash(
            candidate.consensus_receipt_hash,
            name="consensus_receipt_hash",
        ),
        geometry_stability_score=_validate_score(
            candidate.geometry_stability_score,
            name="geometry_stability_score",
        ),
        manifold_agreement_score=_validate_score(
            candidate.manifold_agreement_score,
            name="manifold_agreement_score",
        ),
        coverage_ratio=_validate_score(candidate.coverage_ratio, name="coverage_ratio"),
    )


def _normalize_config(
    config: InformationGeometricCertificationConfig | None,
) -> InformationGeometricCertificationConfig:
    cfg = config if config is not None else InformationGeometricCertificationConfig()
    if not isinstance(cfg, InformationGeometricCertificationConfig):
        raise ValueError("config must be an InformationGeometricCertificationConfig")

    w0 = _validate_score(cfg.divergence_consistency_weight, name="divergence_consistency_weight")
    w1 = _validate_score(cfg.manifold_consistency_weight, name="manifold_consistency_weight")
    w2 = _validate_score(cfg.transport_consistency_weight, name="transport_consistency_weight")
    w3 = _validate_score(cfg.consensus_certainty_weight, name="consensus_certainty_weight")
    w4 = _validate_score(cfg.coverage_completeness_weight, name="coverage_completeness_weight")
    total = _quantize(w0 + w1 + w2 + w3 + w4, name="weight_total")
    if abs(total - 1.0) > 1e-12:
        raise ValueError("certification weights must sum to 1")

    return InformationGeometricCertificationConfig(
        schema_version=cfg.schema_version,
        divergence_consistency_weight=w0,
        manifold_consistency_weight=w1,
        transport_consistency_weight=w2,
        consensus_certainty_weight=w3,
        coverage_completeness_weight=w4,
    )


def _compute_certification_result(
    normalized_input: InformationGeometricCertificationInput,
    config: InformationGeometricCertificationConfig,
) -> InformationGeometricCertificationResult:
    divergence_alignment = _clamp01(
        1.0
        - (
            normalized_input.js_divergence_score
            + normalized_input.fisher_rao_distance_score
            + normalized_input.global_divergence_correspondence_score
        )
        / 3.0
    )
    manifold_consistency = _clamp01(
        (
            (1.0 - normalized_input.fisher_rao_distance_score)
            + normalized_input.manifold_agreement_score
            + normalized_input.geometry_stability_score
        )
        / 3.0
    )
    transport_consistency = _clamp01(1.0 - normalized_input.global_transport_geometry_score)
    consensus_certainty = _clamp01(
        (normalized_input.global_information_consensus_score + normalized_input.geometry_stability_score) / 2.0
    )
    coverage_completeness = _clamp01(normalized_input.coverage_ratio)

    global_certification = _clamp01(
        config.divergence_consistency_weight * divergence_alignment
        + config.manifold_consistency_weight * manifold_consistency
        + config.transport_consistency_weight * transport_consistency
        + config.consensus_certainty_weight * consensus_certainty
        + config.coverage_completeness_weight * coverage_completeness
    )

    payload = {
        "divergence_consistency_score": _quantized_str(
            divergence_alignment,
            name="divergence_consistency_score",
        ),
        "manifold_consistency_score": _quantized_str(
            manifold_consistency,
            name="manifold_consistency_score",
        ),
        "transport_consistency_score": _quantized_str(
            transport_consistency,
            name="transport_consistency_score",
        ),
        "consensus_certainty_score": _quantized_str(
            consensus_certainty,
            name="consensus_certainty_score",
        ),
        "coverage_completeness_score": _quantized_str(
            coverage_completeness,
            name="coverage_completeness_score",
        ),
        "global_information_geometry_certification_score": _quantized_str(
            global_certification,
            name="global_information_geometry_certification_score",
        ),
    }

    return InformationGeometricCertificationResult(
        divergence_consistency_score=_quantize(
            divergence_alignment,
            name="divergence_consistency_score",
        ),
        manifold_consistency_score=_quantize(
            manifold_consistency,
            name="manifold_consistency_score",
        ),
        transport_consistency_score=_quantize(
            transport_consistency,
            name="transport_consistency_score",
        ),
        consensus_certainty_score=_quantize(
            consensus_certainty,
            name="consensus_certainty_score",
        ),
        coverage_completeness_score=_quantize(
            coverage_completeness,
            name="coverage_completeness_score",
        ),
        global_information_geometry_certification_score=_quantize(
            global_certification,
            name="global_information_geometry_certification_score",
        ),
        result_hash=_sha256_hex(payload),
    )


def build_ascii_information_geometric_certification_summary(
    report: InformationGeometricCertificationReport,
) -> str:
    if not isinstance(report, InformationGeometricCertificationReport):
        raise ValueError("report must be an InformationGeometricCertificationReport")
    r = report.certification_result
    i = report.certification_input
    lines = (
        f"# Information-Geometric Certification Pack ({report.schema_version})",
        f"js_divergence_receipt_hash: {i.js_divergence_receipt_hash}",
        f"fisher_rao_receipt_hash: {i.fisher_rao_receipt_hash}",
        f"divergence_correspondence_receipt_hash: {i.divergence_correspondence_receipt_hash}",
        f"transport_geometry_receipt_hash: {i.transport_geometry_receipt_hash}",
        f"consensus_receipt_hash: {i.consensus_receipt_hash}",
        f"divergence_consistency_score:                  {_quantized_str(r.divergence_consistency_score, name='divergence_consistency_score')}",
        f"manifold_consistency_score:                    {_quantized_str(r.manifold_consistency_score, name='manifold_consistency_score')}",
        f"transport_consistency_score:                   {_quantized_str(r.transport_consistency_score, name='transport_consistency_score')}",
        f"consensus_certainty_score:                     {_quantized_str(r.consensus_certainty_score, name='consensus_certainty_score')}",
        f"coverage_completeness_score:                   {_quantized_str(r.coverage_completeness_score, name='coverage_completeness_score')}",
        f"global_information_geometry_certification_score: {_quantized_str(r.global_information_geometry_certification_score, name='global_information_geometry_certification_score')}",
        f"report_hash: {report.report_hash}",
    )
    return "\n".join(lines)


def run_information_geometric_certification_pack(
    certification_input: InformationGeometricCertificationInput | Mapping[str, Any],
    config: InformationGeometricCertificationConfig | None = None,
) -> tuple[InformationGeometricCertificationReport, InformationGeometricCertificationReceipt]:
    normalized_input = _normalize_input(certification_input)
    normalized_config = _normalize_config(config)
    result = _compute_certification_result(normalized_input, normalized_config)

    report_seed = {
        "schema_version": SCHEMA_VERSION,
        "config": normalized_config.to_dict(),
        "certification_input": normalized_input.to_dict(),
        "certification_result": result.to_dict(),
    }
    report_hash = _sha256_hex(report_seed)
    temp_report = InformationGeometricCertificationReport(
        schema_version=SCHEMA_VERSION,
        config=normalized_config,
        certification_input=normalized_input,
        certification_result=result,
        summary_text="",
        report_hash=report_hash,
    )
    summary = build_ascii_information_geometric_certification_summary(temp_report)

    report = InformationGeometricCertificationReport(
        schema_version=SCHEMA_VERSION,
        config=normalized_config,
        certification_input=normalized_input,
        certification_result=result,
        summary_text=summary,
        report_hash=report_hash,
    )

    report_bytes = report.to_canonical_bytes()
    validation_passed = _sha256_hex(report_seed) == report.report_hash and len(report_bytes) > 0
    if not validation_passed:
        raise ValueError("information-geometric certification report failed self-verification")

    receipt_payload = {
        "report_hash": report.report_hash,
        "config_hash": report.config.config_hash,
        "js_divergence_receipt_hash": normalized_input.js_divergence_receipt_hash,
        "fisher_rao_receipt_hash": normalized_input.fisher_rao_receipt_hash,
        "divergence_correspondence_receipt_hash": normalized_input.divergence_correspondence_receipt_hash,
        "transport_geometry_receipt_hash": normalized_input.transport_geometry_receipt_hash,
        "consensus_receipt_hash": normalized_input.consensus_receipt_hash,
        "result_hash": result.result_hash,
        "byte_length": len(report_bytes),
        "validation_passed": validation_passed,
    }
    receipt = InformationGeometricCertificationReceipt(
        report_hash=report.report_hash,
        config_hash=report.config.config_hash,
        js_divergence_receipt_hash=normalized_input.js_divergence_receipt_hash,
        fisher_rao_receipt_hash=normalized_input.fisher_rao_receipt_hash,
        divergence_correspondence_receipt_hash=normalized_input.divergence_correspondence_receipt_hash,
        transport_geometry_receipt_hash=normalized_input.transport_geometry_receipt_hash,
        consensus_receipt_hash=normalized_input.consensus_receipt_hash,
        result_hash=result.result_hash,
        byte_length=len(report_bytes),
        validation_passed=validation_passed,
        receipt_hash=_sha256_hex(receipt_payload),
    )
    return report, receipt
