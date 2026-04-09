"""v137.12.3 — Hybrid Replay Certification.

Deterministic certification layer for synthetic hybrid replay artifacts:
substrate simulation -> hybrid signal interface -> benchmark battery.
Simulation-first only: this module certifies synthetic replay integrity and
makes no biological or physiological claims.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping

from qec.analysis.hybrid_signal_interface import (
    SCHEMA_VERSION as HYBRID_INTERFACE_SCHEMA_VERSION,
    HybridSignalReceipt,
    HybridSignalTrace,
)
from qec.analysis.neuromorphic_substrate_simulator import (
    SCHEMA_VERSION as SUBSTRATE_SCHEMA_VERSION,
    SubstrateSimulationReport,
    stable_substrate_report_hash,
)
from qec.benchmark.bio_signal_benchmark_battery import (
    SCHEMA_VERSION as BENCHMARK_SCHEMA_VERSION,
    BioSignalBenchmarkBatteryReport,
)

SCHEMA_VERSION = "v137.12.3"
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_METRIC_KEYS: tuple[str, ...] = (
    "continuity_score",
    "replay_fidelity_score",
    "scaling_efficiency_score",
    "threshold_response_score",
)
_REQUIRED_CHANNEL_ORDER: tuple[str, ...] = (
    "node_state_lane",
    "spike_event_lane",
    "threshold_reset_lane",
    "aggregate_activity_lane",
)

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
        return tuple(_canonicalize_json(item) for item in value)
    if isinstance(value, list):
        return tuple(_canonicalize_json(item) for item in value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value.keys()):
            raise ValueError("payload keys must be strings")
        return {key: _canonicalize_json(value[key]) for key in sorted(value.keys())}
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


def _require_hex64(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not _HEX64_RE.match(value):
        raise ValueError(f"{field_name} must be a 64-character lowercase hex hash")
    return value


def _require_finite_metric(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field_name} must be finite")
    return numeric


def _bounded_score(*, passed: tuple[bool, ...]) -> float:
    return 1.0 if all(passed) else 0.0


@dataclass(frozen=True)
class HybridReplayCertificationConfig:
    schema_version: str = SCHEMA_VERSION
    require_strict_ordering: bool = True
    require_metric_identity: bool = True

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "require_strict_ordering": self.require_strict_ordering,
            "require_metric_identity": self.require_metric_identity,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())

    def stable_receipt_chain(self) -> tuple[str, ...]:
        return (self.stable_hash(),)


@dataclass(frozen=True)
class HybridReplayEvidence:
    substrate_hash: str
    trace_hash: str
    benchmark_hash: str
    substrate_receipt_hash: str
    interface_receipt_hash: str
    benchmark_result_hashes: tuple[str, ...]
    metric_snapshot: Mapping[str, float]
    node_order: tuple[int, ...]
    frame_order: tuple[int, ...]
    channel_order: tuple[str, ...]
    receipt_order: tuple[str, ...]
    summary_metric_order: tuple[str, ...]

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "substrate_hash": self.substrate_hash,
            "trace_hash": self.trace_hash,
            "benchmark_hash": self.benchmark_hash,
            "substrate_receipt_hash": self.substrate_receipt_hash,
            "interface_receipt_hash": self.interface_receipt_hash,
            "benchmark_result_hashes": self.benchmark_result_hashes,
            "metric_snapshot": {key: float(self.metric_snapshot[key]) for key in sorted(self.metric_snapshot.keys())},
            "node_order": self.node_order,
            "frame_order": self.frame_order,
            "channel_order": self.channel_order,
            "receipt_order": self.receipt_order,
            "summary_metric_order": self.summary_metric_order,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())

    def stable_receipt_chain(self) -> tuple[str, ...]:
        return (
            self.substrate_receipt_hash,
            self.interface_receipt_hash,
            self.stable_hash(),
        )


@dataclass(frozen=True)
class HybridReplayCertificationResult:
    byte_identity_passed: bool
    hash_lineage_passed: bool
    structural_replay_passed: bool
    metric_replay_passed: bool
    cross_layer_passed: bool
    validation_passed: bool
    certification_score: float
    validation_flags: Mapping[str, bool]
    receipt_chain: tuple[str, ...]

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "byte_identity_passed": self.byte_identity_passed,
            "hash_lineage_passed": self.hash_lineage_passed,
            "structural_replay_passed": self.structural_replay_passed,
            "metric_replay_passed": self.metric_replay_passed,
            "cross_layer_passed": self.cross_layer_passed,
            "validation_passed": self.validation_passed,
            "certification_score": self.certification_score,
            "validation_flags": {key: bool(self.validation_flags[key]) for key in sorted(self.validation_flags.keys())},
            "receipt_chain": self.receipt_chain,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())

    def stable_receipt_chain(self) -> tuple[str, ...]:
        return (*self.receipt_chain, self.stable_hash())


@dataclass(frozen=True)
class HybridReplayCertificationReport:
    certification_id: str
    config: HybridReplayCertificationConfig
    evidence: HybridReplayEvidence
    result: HybridReplayCertificationResult
    report_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "certification_id": self.certification_id,
            "config": self.config.to_dict(),
            "evidence": self.evidence.to_dict(),
            "result": self.result.to_dict(),
            "report_hash": self.report_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("report_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())

    def stable_receipt_chain(self) -> tuple[str, ...]:
        return (*self.result.stable_receipt_chain(), self.report_hash)


def certify_byte_identity(reference_bytes: bytes, replay_bytes: tuple[bytes, ...]) -> bool:
    if not isinstance(reference_bytes, (bytes, bytearray)) or len(reference_bytes) == 0:
        raise ValueError("reference_bytes must be non-empty bytes")
    if len(replay_bytes) == 0:
        raise ValueError("replay_bytes must be non-empty")
    for idx, candidate in enumerate(replay_bytes):
        if not isinstance(candidate, (bytes, bytearray)) or len(candidate) == 0:
            raise ValueError(f"replay_bytes[{idx}] must be non-empty bytes")
        if bytes(candidate) != bytes(reference_bytes):
            return False
    return True


def certify_hash_lineage(
    *,
    substrate_hash: str,
    trace_hash: str,
    benchmark_hash: str,
    report_payload_hash: str,
) -> bool:
    _require_hex64(substrate_hash, "substrate_hash")
    _require_hex64(trace_hash, "trace_hash")
    _require_hex64(benchmark_hash, "benchmark_hash")
    _require_hex64(report_payload_hash, "report_payload_hash")
    chain = _sha256_hex({
        "substrate_hash": substrate_hash,
        "trace_hash": trace_hash,
        "benchmark_hash": benchmark_hash,
        "report_payload_hash": report_payload_hash,
    })
    return isinstance(chain, str) and len(chain) == 64


def certify_structural_replay(
    *,
    trace: HybridSignalTrace,
    benchmark_report: BioSignalBenchmarkBatteryReport,
    receipt_order: tuple[str, ...],
) -> bool:
    node_order = tuple(trace.node_ids)
    frame_order = tuple(frame.time_index for frame in trace.frames)
    channel_order = tuple(trace.config.channel_names)

    expected_nodes = tuple(range(len(node_order)))
    expected_frames = tuple(range(trace.frame_count))
    expected_receipt_order = tuple(sorted(receipt_order))
    summary_metric_order = tuple(sorted(benchmark_report.aggregate_metrics.keys()))

    return (
        node_order == expected_nodes
        and frame_order == expected_frames
        and channel_order == _REQUIRED_CHANNEL_ORDER
        and receipt_order == expected_receipt_order
        and summary_metric_order == tuple(sorted(summary_metric_order))
    )


def certify_metric_replay(
    reference_metrics: Mapping[str, float],
    replay_metrics: Mapping[str, float],
) -> bool:
    for key in _REQUIRED_METRIC_KEYS:
        if key not in reference_metrics or key not in replay_metrics:
            raise ValueError(f"missing required metric: {key}")
        left = _require_finite_metric(float(reference_metrics[key]), f"reference_metrics[{key}]")
        right = _require_finite_metric(float(replay_metrics[key]), f"replay_metrics[{key}]")
        if left != right:
            return False
    return True


def _validate_artifact_integrity(
    *,
    substrate_report: SubstrateSimulationReport,
    trace: HybridSignalTrace,
    interface_receipt: HybridSignalReceipt,
    benchmark_report: BioSignalBenchmarkBatteryReport,
    config: HybridReplayCertificationConfig,
) -> None:
    if config.schema_version != SCHEMA_VERSION:
        raise ValueError("schema mismatch: certification config schema")
    if substrate_report.schema_version != SUBSTRATE_SCHEMA_VERSION:
        raise ValueError("schema mismatch: substrate report schema")
    if trace.schema_version != HYBRID_INTERFACE_SCHEMA_VERSION:
        raise ValueError("schema mismatch: trace schema")
    if interface_receipt.schema_version != HYBRID_INTERFACE_SCHEMA_VERSION:
        raise ValueError("schema mismatch: interface receipt schema")
    if benchmark_report.schema_version != BENCHMARK_SCHEMA_VERSION:
        raise ValueError("schema mismatch: benchmark schema")

    if substrate_report.stable_hash != stable_substrate_report_hash(substrate_report):
        raise ValueError("invalid hashes: substrate report stable_hash mismatch")
    _require_hex64(substrate_report.receipt.receipt_hash, "substrate receipt hash")
    recalculated_substrate_receipt_hash = _sha256_hex(substrate_report.receipt.to_hash_payload_dict())
    if substrate_report.receipt.receipt_hash != recalculated_substrate_receipt_hash:
        raise ValueError("invalid hashes: substrate receipt hash mismatch")

    _require_hex64(interface_receipt.receipt_hash, "interface receipt hash")
    recalculated_interface_receipt_hash = _sha256_hex(interface_receipt.to_hash_payload_dict())
    if interface_receipt.receipt_hash != recalculated_interface_receipt_hash:
        raise ValueError("invalid hashes: interface receipt hash mismatch")

    if not interface_receipt.validation_passed:
        raise ValueError("missing required receipts: invalid interface receipt")
    if not substrate_report.receipt.validation_passed:
        raise ValueError("missing required receipts: invalid substrate receipt")

    if trace.frame_count <= 0 or len(trace.frames) != trace.frame_count:
        raise ValueError("malformed traces: invalid frame count")
    if trace.frame_count != len(tuple(range(trace.frame_count))):
        raise ValueError("malformed traces: invalid frame cardinality")
    if trace.node_ids != tuple(range(len(trace.node_ids))):
        raise ValueError("invalid ordering: node ordering must be dense deterministic")
    _require_hex64(trace.stable_hash, "trace stable_hash")
    recalculated_trace_hash = _sha256_hex(trace.to_hash_payload_dict())
    if trace.stable_hash != recalculated_trace_hash:
        raise ValueError("invalid hashes: trace stable_hash mismatch")

    seen_frame_hashes: set[str] = set()
    for expected_idx, frame in enumerate(trace.frames):
        if frame.time_index != expected_idx:
            raise ValueError("invalid ordering: frame ordering")
        _require_hex64(frame.stable_hash, "frame stable_hash")
        recalculated_frame_hash = _sha256_hex(frame.to_hash_payload_dict())
        if frame.stable_hash != recalculated_frame_hash:
            raise ValueError("invalid hashes: frame stable_hash mismatch")
        if frame.stable_hash in seen_frame_hashes:
            raise ValueError("duplicate identities: frame hash collision")
        seen_frame_hashes.add(frame.stable_hash)

    if interface_receipt.output_stable_hash != trace.stable_hash:
        raise ValueError("broken lineage: interface receipt output hash mismatch")
    if interface_receipt.input_stable_hash != substrate_report.stable_hash:
        raise ValueError("broken lineage: interface receipt input hash mismatch")
    if trace.input_stable_hash != substrate_report.stable_hash:
        raise ValueError("broken lineage: trace input hash mismatch")

    result_hashes: list[str] = []
    has_anchor_case = False
    for result in benchmark_report.results:
        for key, value in result.metrics.items():
            _require_finite_metric(float(value), f"benchmark metric {key}")
        _require_hex64(result.case.trace_hash, "benchmark case trace_hash")
        if result.case.trace_hash == trace.stable_hash:
            has_anchor_case = True
        recalculated_result_hash = _sha256_hex(result.to_hash_payload_dict())
        if result.stable_hash != recalculated_result_hash:
            raise ValueError("invalid hashes: benchmark result hash mismatch")
        result_hashes.append(result.stable_hash)

    if len(result_hashes) != len(set(result_hashes)):
        raise ValueError("duplicate identities: duplicate benchmark result stable_hash")
    if not has_anchor_case:
        raise ValueError("broken lineage: benchmark report missing anchor trace hash")

    recalculated_battery_hash = _sha256_hex(benchmark_report.to_hash_payload_dict())
    if benchmark_report.stable_hash != recalculated_battery_hash:
        raise ValueError("invalid hashes: benchmark report hash mismatch")


def build_hybrid_replay_certificate(
    *,
    config: HybridReplayCertificationConfig,
    evidence: HybridReplayEvidence,
    result: HybridReplayCertificationResult,
) -> HybridReplayCertificationReport:
    certification_id = _sha256_hex({
        "schema_version": SCHEMA_VERSION,
        "config_hash": config.stable_hash(),
        "evidence_hash": evidence.stable_hash(),
        "result_hash": result.stable_hash(),
    })
    proto = HybridReplayCertificationReport(
        certification_id=certification_id,
        config=config,
        evidence=evidence,
        result=result,
        report_hash="",
    )
    return HybridReplayCertificationReport(
        certification_id=proto.certification_id,
        config=proto.config,
        evidence=proto.evidence,
        result=proto.result,
        report_hash=proto.stable_hash(),
    )


def run_hybrid_replay_certification(
    substrate_report: SubstrateSimulationReport,
    trace: HybridSignalTrace,
    interface_receipt: HybridSignalReceipt,
    benchmark_report: BioSignalBenchmarkBatteryReport,
    *,
    config: HybridReplayCertificationConfig | None = None,
    replay_trace: HybridSignalTrace | None = None,
    replay_benchmark_report: BioSignalBenchmarkBatteryReport | None = None,
) -> HybridReplayCertificationReport:
    effective_config = config or HybridReplayCertificationConfig()
    _validate_artifact_integrity(
        substrate_report=substrate_report,
        trace=trace,
        interface_receipt=interface_receipt,
        benchmark_report=benchmark_report,
        config=effective_config,
    )

    replay_trace_effective = trace if replay_trace is None else replay_trace
    replay_benchmark_effective = benchmark_report if replay_benchmark_report is None else replay_benchmark_report

    if replay_trace_effective.to_canonical_bytes() != trace.to_canonical_bytes():
        raise ValueError("invalid ordering: replay trace mismatch")

    replay_reference = benchmark_report.to_canonical_bytes()
    replay_candidate = replay_benchmark_effective.to_canonical_bytes()
    byte_identity_passed = certify_byte_identity(replay_reference, (replay_candidate,))

    receipt_order = tuple(sorted((substrate_report.receipt.receipt_hash, interface_receipt.receipt_hash)))
    structural_replay_passed = certify_structural_replay(
        trace=trace,
        benchmark_report=benchmark_report,
        receipt_order=receipt_order,
    )

    metric_replay_passed = certify_metric_replay(
        benchmark_report.aggregate_metrics,
        replay_benchmark_effective.aggregate_metrics,
    )

    cross_layer_passed = (
        trace.input_stable_hash == substrate_report.stable_hash
        and interface_receipt.input_stable_hash == substrate_report.stable_hash
        and interface_receipt.output_stable_hash == trace.stable_hash
        and any(result.case.trace_hash == trace.stable_hash for result in benchmark_report.results)
    )

    metric_snapshot = {
        key: _require_finite_metric(float(benchmark_report.aggregate_metrics[key]), f"aggregate metric {key}")
        for key in _REQUIRED_METRIC_KEYS
    }
    evidence = HybridReplayEvidence(
        substrate_hash=substrate_report.stable_hash,
        trace_hash=trace.stable_hash,
        benchmark_hash=benchmark_report.stable_hash,
        substrate_receipt_hash=substrate_report.receipt.receipt_hash,
        interface_receipt_hash=interface_receipt.receipt_hash,
        benchmark_result_hashes=tuple(sorted(result.stable_hash for result in benchmark_report.results)),
        metric_snapshot=metric_snapshot,
        node_order=trace.node_ids,
        frame_order=tuple(frame.time_index for frame in trace.frames),
        channel_order=trace.config.channel_names,
        receipt_order=receipt_order,
        summary_metric_order=tuple(sorted(benchmark_report.aggregate_metrics.keys())),
    )

    lineage_payload_hash = _sha256_hex({
        "substrate_hash": evidence.substrate_hash,
        "trace_hash": evidence.trace_hash,
        "benchmark_hash": evidence.benchmark_hash,
        "evidence_hash": evidence.stable_hash(),
    })
    hash_lineage_passed = certify_hash_lineage(
        substrate_hash=evidence.substrate_hash,
        trace_hash=evidence.trace_hash,
        benchmark_hash=evidence.benchmark_hash,
        report_payload_hash=lineage_payload_hash,
    )

    validation_flags = {
        "schemas_valid": True,
        "hashes_valid": True,
        "receipts_present": True,
        "ordering_valid": structural_replay_passed,
    }
    domain_passes = (
        byte_identity_passed,
        hash_lineage_passed,
        structural_replay_passed,
        metric_replay_passed,
        cross_layer_passed,
    )
    result = HybridReplayCertificationResult(
        byte_identity_passed=byte_identity_passed,
        hash_lineage_passed=hash_lineage_passed,
        structural_replay_passed=structural_replay_passed,
        metric_replay_passed=metric_replay_passed,
        cross_layer_passed=cross_layer_passed,
        validation_passed=all(validation_flags.values()),
        certification_score=_bounded_score(passed=domain_passes),
        validation_flags=validation_flags,
        receipt_chain=evidence.stable_receipt_chain(),
    )
    return build_hybrid_replay_certificate(config=effective_config, evidence=evidence, result=result)


__all__ = [
    "SCHEMA_VERSION",
    "HybridReplayCertificationConfig",
    "HybridReplayEvidence",
    "HybridReplayCertificationResult",
    "HybridReplayCertificationReport",
    "certify_byte_identity",
    "certify_hash_lineage",
    "certify_structural_replay",
    "certify_metric_replay",
    "build_hybrid_replay_certificate",
    "run_hybrid_replay_certification",
]
