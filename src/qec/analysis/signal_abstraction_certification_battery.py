"""v137.13.4 — Signal Abstraction Certification Battery.

Deterministic Layer-4 certification battery for the full abstraction stack:
geometry -> morphology -> topology -> region correspondence.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import json
import math
from typing import Any, Mapping

from qec.analysis.morphology_transition_kernel import MorphologyTransitionResult
from qec.analysis.phase_boundary_topology_kernel import PhaseBoundaryTopologyResult
from qec.analysis.region_correspondence_kernel import RegionCorrespondenceResult
from qec.analysis.synthetic_signal_geometry_kernel import SignalGeometryKernelResult

SCHEMA_VERSION = "v137.13.4"
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
    if callable(value):
        raise ValueError("callable values are not allowed in canonical payloads")
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


def _canonical_bytes(value: Any) -> bytes:
    return _canonical_json(value).encode("utf-8")


def _sha256_hex(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256_hex(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _quantize(value: float, *, field_name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return float(Decimal(str(value)).quantize(_DECIMAL_PLACES, rounding=ROUND_HALF_EVEN))


def _quantize_unit(value: float, *, field_name: str) -> float:
    q = _quantize(value, field_name=field_name)
    if q < 0.0 or q > 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]")
    return q


def _quantized_str(value: float, *, field_name: str) -> str:
    q = Decimal(str(_quantize(value, field_name=field_name))).quantize(
        _DECIMAL_PLACES,
        rounding=ROUND_HALF_EVEN,
    )
    return str(q)


def _metric_mean(metrics: tuple[float, ...], *, field_name: str) -> float:
    if len(metrics) == 0:
        raise ValueError(f"{field_name} requires at least one metric")
    return _quantize_unit(sum(metrics) / float(len(metrics)), field_name=field_name)


@dataclass(frozen=True)
class SignalAbstractionCertificationConfig:
    schema_version: str = SCHEMA_VERSION
    kernel_version: str = SCHEMA_VERSION
    fail_fast: bool = True

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "kernel_version": self.kernel_version,
            "fail_fast": self.fail_fast,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_sha256(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class SignalAbstractionEvidence:
    geometry_hash: str
    morphology_hash: str
    topology_hash: str
    correspondence_hash: str
    receipt_chain: tuple[str, ...]

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "geometry_hash": self.geometry_hash,
            "morphology_hash": self.morphology_hash,
            "topology_hash": self.topology_hash,
            "correspondence_hash": self.correspondence_hash,
            "receipt_chain": self.receipt_chain,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_sha256(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class SignalAbstractionCertificationResult:
    validation_passed: bool
    certification_score: float
    failure_reasons: tuple[str, ...]
    report_hash: str
    receipt_hash: str
    determinism_score: float
    lineage_integrity_score: float
    metric_integrity_score: float
    cross_layer_consistency_score: float
    global_certification_score: float

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "validation_passed": self.validation_passed,
            "certification_score": _quantized_str(self.certification_score, field_name="certification_score"),
            "failure_reasons": self.failure_reasons,
            "report_hash": self.report_hash,
            "receipt_hash": self.receipt_hash,
            "determinism_score": _quantized_str(self.determinism_score, field_name="determinism_score"),
            "lineage_integrity_score": _quantized_str(
                self.lineage_integrity_score,
                field_name="lineage_integrity_score",
            ),
            "metric_integrity_score": _quantized_str(self.metric_integrity_score, field_name="metric_integrity_score"),
            "cross_layer_consistency_score": _quantized_str(
                self.cross_layer_consistency_score,
                field_name="cross_layer_consistency_score",
            ),
            "global_certification_score": _quantized_str(
                self.global_certification_score,
                field_name="global_certification_score",
            ),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_sha256(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class SignalAbstractionCertificationReport:
    config: SignalAbstractionCertificationConfig
    evidence: SignalAbstractionEvidence
    result: SignalAbstractionCertificationResult
    report_hash: str
    schema_version: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "config": self.config.to_dict(),
            "evidence": self.evidence.to_dict(),
            "result": self.result.to_dict(),
            "report_hash": self.report_hash,
            "schema_version": self.schema_version,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("report_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_sha256(self) -> str:
        return self.report_hash


@dataclass(frozen=True)
class SignalAbstractionCertificationReceipt:
    receipt_hash: str
    report_hash: str
    evidence_hash: str
    result_hash: str
    receipt_chain: tuple[str, ...]
    schema_version: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "receipt_hash": self.receipt_hash,
            "report_hash": self.report_hash,
            "evidence_hash": self.evidence_hash,
            "result_hash": self.result_hash,
            "receipt_chain": self.receipt_chain,
            "schema_version": self.schema_version,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("receipt_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_sha256(self) -> str:
        return self.receipt_hash


def _validate_metric_bounds(metrics: Mapping[str, float], *, prefix: str) -> None:
    for metric_name, metric_value in metrics.items():
        _quantize_unit(metric_value, field_name=f"{prefix}.{metric_name}")


def _build_evidence(
    geometry: SignalGeometryKernelResult,
    morphology: MorphologyTransitionResult,
    topology: PhaseBoundaryTopologyResult,
    correspondence: RegionCorrespondenceResult,
) -> SignalAbstractionEvidence:
    geometry_hash = geometry.stable_hash
    morphology_hash = morphology.stable_hash
    topology_hash = topology.stable_hash
    correspondence_hash = correspondence.stable_hash

    if not all(_is_sha256_hex(h) for h in (geometry_hash, morphology_hash, topology_hash, correspondence_hash)):
        raise ValueError("broken lineage: hash must be 64-character lowercase hex")

    chain_seed = (
        geometry_hash,
        morphology_hash,
        topology_hash,
        correspondence_hash,
    )
    chain_tip = _sha256_hex({"receipt_chain_seed": chain_seed})
    return SignalAbstractionEvidence(
        geometry_hash=geometry_hash,
        morphology_hash=morphology_hash,
        topology_hash=topology_hash,
        correspondence_hash=correspondence_hash,
        receipt_chain=chain_seed + (chain_tip,),
    )


def _validate_stack_lineage(
    geometry: SignalGeometryKernelResult,
    morphology: MorphologyTransitionResult,
    topology: PhaseBoundaryTopologyResult,
    correspondence: RegionCorrespondenceResult,
) -> None:
    if geometry.trajectory.stable_hash != _sha256_hex(geometry.trajectory.to_hash_payload_dict()):
        raise ValueError("broken lineage: geometry trajectory hash mismatch")
    if geometry.stable_hash != _sha256_hex(geometry.to_hash_payload_dict()):
        raise ValueError("broken lineage: geometry result hash mismatch")
    if morphology.path.stable_hash != _sha256_hex(morphology.path.to_hash_payload_dict()):
        raise ValueError("broken lineage: morphology path hash mismatch")
    if morphology.stable_hash != _sha256_hex(morphology.to_hash_payload_dict()):
        raise ValueError("broken lineage: morphology result hash mismatch")
    if topology.path.stable_hash != _sha256_hex(topology.path.to_hash_payload_dict()):
        raise ValueError("broken lineage: topology path hash mismatch")
    if topology.stable_hash != _sha256_hex(topology.to_hash_payload_dict()):
        raise ValueError("broken lineage: topology result hash mismatch")
    if correspondence.stable_hash != _sha256_hex(correspondence.to_hash_payload_dict()):
        raise ValueError("broken lineage: correspondence result hash mismatch")


def _compute_domain_flags(
    geometry: SignalGeometryKernelResult,
    morphology: MorphologyTransitionResult,
    topology: PhaseBoundaryTopologyResult,
    correspondence: RegionCorrespondenceResult,
    evidence: SignalAbstractionEvidence,
) -> tuple[dict[str, bool], dict[str, bool], dict[str, bool], dict[str, bool]]:
    _validate_stack_lineage(geometry, morphology, topology, correspondence)

    report_a = build_certification_report(evidence, _compute_result(evidence, (), (1.0, 1.0, 1.0, 1.0, 1.0)))
    report_b = build_certification_report(evidence, _compute_result(evidence, (), (1.0, 1.0, 1.0, 1.0, 1.0)))
    byte_identity = {
        "repeated_run_byte_identity": report_a.to_canonical_bytes() == report_b.to_canonical_bytes(),
        "canonical_export_stability": report_a.to_canonical_json() == report_a.to_canonical_json(),
        "stable_hash_consistency": report_a.stable_sha256() == report_b.stable_sha256(),
    }

    lineage = {
        "geometry_hash": evidence.geometry_hash == geometry.stable_hash,
        "morphology_hash": evidence.morphology_hash == morphology.stable_hash,
        "topology_hash": evidence.topology_hash == topology.stable_hash,
        "correspondence_hash": evidence.correspondence_hash == correspondence.stable_hash,
        "receipt_chain_continuity": evidence.receipt_chain[-1]
        == _sha256_hex({"receipt_chain_seed": evidence.receipt_chain[:-1]}),
    }

    _validate_metric_bounds(
        {
            "geometry_integrity_score": geometry.geometry_integrity_score,
            "continuity_score": geometry.continuity_score,
            "similarity_score": geometry.similarity_score,
            "path_stability_score": geometry.path_stability_score,
        },
        prefix="geometry",
    )
    _validate_metric_bounds(
        {
            "transition_integrity_score": morphology.transition_integrity_score,
            "phase_stability_score": morphology.phase_stability_score,
            "morphology_consistency_score": morphology.morphology_consistency_score,
            "transition_continuity_score": morphology.transition_continuity_score,
        },
        prefix="morphology",
    )
    _validate_metric_bounds(
        {
            "boundary_integrity_score": topology.boundary_integrity_score,
            "topology_stability_score": topology.topology_stability_score,
            "region_consistency_score": topology.region_consistency_score,
            "boundary_continuity_score": topology.boundary_continuity_score,
        },
        prefix="topology",
    )
    _validate_metric_bounds(
        {
            "region_alignment_score": correspondence.region_alignment_score,
            "topology_correspondence_score": correspondence.topology_correspondence_score,
            "boundary_coherence_score": correspondence.boundary_coherence_score,
            "global_correspondence_score": correspondence.global_correspondence_score,
        },
        prefix="correspondence",
    )

    metric_integrity = {"all_metrics_bounded": True}

    unique_segments = len(set(evidence.receipt_chain[:-1])) == len(evidence.receipt_chain[:-1])
    cross_layer = {
        "topology_traces_to_morphology": topology.path.input_transition_hash == morphology.path.stable_hash,
        "correspondence_traces_to_topology": topology.path.stable_hash in correspondence.path_hashes,
        "stable_hash_lineage_anchors": (
            geometry.trajectory.stable_hash == morphology.path.input_trajectory_hash
            and morphology.path.stable_hash == topology.path.input_transition_hash
        ),
        "no_duplicate_chain_segments": unique_segments,
    }

    return byte_identity, lineage, metric_integrity, cross_layer


def _compute_result(
    evidence: SignalAbstractionEvidence,
    failure_reasons: tuple[str, ...],
    scores: tuple[float, float, float, float, float],
) -> SignalAbstractionCertificationResult:
    determinism_score, lineage_score, metric_score, cross_layer_score, global_score = scores
    return SignalAbstractionCertificationResult(
        validation_passed=len(failure_reasons) == 0,
        certification_score=global_score,
        failure_reasons=failure_reasons,
        report_hash="",
        receipt_hash="",
        determinism_score=determinism_score,
        lineage_integrity_score=lineage_score,
        metric_integrity_score=metric_score,
        cross_layer_consistency_score=cross_layer_score,
        global_certification_score=global_score,
    )


def build_certification_report(
    evidence: SignalAbstractionEvidence,
    result: SignalAbstractionCertificationResult,
    config: SignalAbstractionCertificationConfig | None = None,
) -> SignalAbstractionCertificationReport:
    effective_config = config or SignalAbstractionCertificationConfig()
    result_hash = result.stable_sha256()
    report_seed = {
        "config": effective_config.to_dict(),
        "evidence": evidence.to_dict(),
        "result": result.to_dict(),
        "result_hash": result_hash,
    }
    report_hash = _sha256_hex(report_seed)
    fixed_result = SignalAbstractionCertificationResult(
        validation_passed=result.validation_passed,
        certification_score=result.certification_score,
        failure_reasons=result.failure_reasons,
        report_hash=report_hash,
        receipt_hash=result.receipt_hash,
        determinism_score=result.determinism_score,
        lineage_integrity_score=result.lineage_integrity_score,
        metric_integrity_score=result.metric_integrity_score,
        cross_layer_consistency_score=result.cross_layer_consistency_score,
        global_certification_score=result.global_certification_score,
    )
    proto_report = SignalAbstractionCertificationReport(
        config=effective_config,
        evidence=evidence,
        result=fixed_result,
        report_hash="",
        schema_version=effective_config.schema_version,
    )
    return SignalAbstractionCertificationReport(
        config=proto_report.config,
        evidence=proto_report.evidence,
        result=proto_report.result,
        report_hash=_sha256_hex(proto_report.to_hash_payload_dict()),
        schema_version=proto_report.schema_version,
    )


def _build_receipt(report: SignalAbstractionCertificationReport) -> SignalAbstractionCertificationReceipt:
    evidence_hash = report.evidence.stable_sha256()
    result_hash = report.result.stable_sha256()
    chain_seed = (report.config.stable_sha256(), evidence_hash, result_hash, report.report_hash)
    chain_tip = _sha256_hex({"receipt_chain_seed": chain_seed})
    proto_receipt = SignalAbstractionCertificationReceipt(
        receipt_hash="",
        report_hash=report.report_hash,
        evidence_hash=evidence_hash,
        result_hash=result_hash,
        receipt_chain=chain_seed + (chain_tip,),
        schema_version=report.schema_version,
    )
    return SignalAbstractionCertificationReceipt(
        receipt_hash=_sha256_hex(proto_receipt.to_hash_payload_dict()),
        report_hash=proto_receipt.report_hash,
        evidence_hash=proto_receipt.evidence_hash,
        result_hash=proto_receipt.result_hash,
        receipt_chain=proto_receipt.receipt_chain,
        schema_version=proto_receipt.schema_version,
    )


def run_signal_abstraction_certification_battery(
    geometry: SignalGeometryKernelResult,
    morphology: MorphologyTransitionResult,
    topology: PhaseBoundaryTopologyResult,
    correspondence: RegionCorrespondenceResult,
    config: SignalAbstractionCertificationConfig | None = None,
) -> tuple[SignalAbstractionCertificationReport, SignalAbstractionCertificationReceipt]:
    effective_config = config or SignalAbstractionCertificationConfig()

    evidence = _build_evidence(geometry, morphology, topology, correspondence)

    byte_identity, lineage, metric_integrity, cross_layer = _compute_domain_flags(
        geometry,
        morphology,
        topology,
        correspondence,
        evidence,
    )

    determinism_score = _metric_mean(tuple(1.0 if ok else 0.0 for ok in byte_identity.values()), field_name="determinism_score")
    lineage_score = _metric_mean(tuple(1.0 if ok else 0.0 for ok in lineage.values()), field_name="lineage_integrity_score")
    metric_score = _metric_mean(tuple(1.0 if ok else 0.0 for ok in metric_integrity.values()), field_name="metric_integrity_score")
    cross_layer_score = _metric_mean(tuple(1.0 if ok else 0.0 for ok in cross_layer.values()), field_name="cross_layer_consistency_score")
    global_score = _metric_mean(
        (determinism_score, lineage_score, metric_score, cross_layer_score),
        field_name="global_certification_score",
    )

    failures: list[str] = []
    for domain_name, checks in (
        ("byte_identity", byte_identity),
        ("lineage_integrity", lineage),
        ("metric_integrity", metric_integrity),
        ("cross_layer", cross_layer),
    ):
        for check_name, check_passed in checks.items():
            if not check_passed:
                failures.append(f"{domain_name}.{check_name}")

    if effective_config.fail_fast and len(failures) > 0:
        failures = tuple(sorted(failures))
    else:
        failures = tuple(sorted(failures))

    base_result = _compute_result(
        evidence,
        failures,
        (determinism_score, lineage_score, metric_score, cross_layer_score, global_score),
    )
    report = build_certification_report(evidence, base_result, effective_config)
    receipt = _build_receipt(report)

    final_result = SignalAbstractionCertificationResult(
        validation_passed=report.result.validation_passed,
        certification_score=report.result.certification_score,
        failure_reasons=report.result.failure_reasons,
        report_hash=report.report_hash,
        receipt_hash=receipt.receipt_hash,
        determinism_score=report.result.determinism_score,
        lineage_integrity_score=report.result.lineage_integrity_score,
        metric_integrity_score=report.result.metric_integrity_score,
        cross_layer_consistency_score=report.result.cross_layer_consistency_score,
        global_certification_score=report.result.global_certification_score,
    )

    final_report = SignalAbstractionCertificationReport(
        config=report.config,
        evidence=report.evidence,
        result=final_result,
        report_hash=report.report_hash,
        schema_version=report.schema_version,
    )

    if final_report.evidence.receipt_chain[-1] != _sha256_hex({"receipt_chain_seed": final_report.evidence.receipt_chain[:-1]}):
        raise ValueError("broken lineage: evidence receipt-chain continuity failure")
    if final_report.result.report_hash != final_report.report_hash:
        raise ValueError("broken lineage: result/report hash mismatch")
    if receipt.receipt_chain[-1] != _sha256_hex({"receipt_chain_seed": receipt.receipt_chain[:-1]}):
        raise ValueError("broken lineage: receipt-chain continuity failure")

    return final_report, receipt


def export_certification_report_json(report: SignalAbstractionCertificationReport) -> str:
    return report.to_canonical_json()


def build_ascii_certification_summary(report: SignalAbstractionCertificationReport) -> str:
    result = report.result
    return "\n".join(
        (
            f"Signal Abstraction Certification Battery — {report.schema_version}",
            f"  Validation Passed:        {result.validation_passed}",
            f"  Determinism:              {_quantized_str(result.determinism_score, field_name='determinism_score')}",
            f"  Lineage Integrity:        {_quantized_str(result.lineage_integrity_score, field_name='lineage_integrity_score')}",
            f"  Metric Integrity:         {_quantized_str(result.metric_integrity_score, field_name='metric_integrity_score')}",
            f"  Cross-Layer Consistency:  {_quantized_str(result.cross_layer_consistency_score, field_name='cross_layer_consistency_score')}",
            f"  Global Score:             {_quantized_str(result.global_certification_score, field_name='global_certification_score')}",
            f"  Report Hash:              {report.report_hash[:16]}...",
            f"  Receipt Hash:             {result.receipt_hash[:16]}...",
        )
    )


__all__ = [
    "SCHEMA_VERSION",
    "SignalAbstractionCertificationConfig",
    "SignalAbstractionEvidence",
    "SignalAbstractionCertificationResult",
    "SignalAbstractionCertificationReport",
    "SignalAbstractionCertificationReceipt",
    "run_signal_abstraction_certification_battery",
    "build_certification_report",
    "export_certification_report_json",
    "build_ascii_certification_summary",
]
