"""v137.12.4 — Experimental Research Pack.

Deterministic research artifact bundle for the full v137.12.x synthetic stack:
substrate simulation -> hybrid signal interface -> benchmark battery -> replay
certification.  Simulation-first only: this module packages synthetic
deterministic artifacts and makes no biological or physiological claims.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

from qec.analysis.hybrid_replay_certification import (
    SCHEMA_VERSION as CERTIFICATION_SCHEMA_VERSION,
    HybridReplayCertificationReport,
)
from qec.analysis.hybrid_signal_interface import (
    SCHEMA_VERSION as INTERFACE_SCHEMA_VERSION,
    HybridSignalReceipt,
    HybridSignalTrace,
)
from qec.analysis.neuromorphic_substrate_simulator import (
    SCHEMA_VERSION as SUBSTRATE_SCHEMA_VERSION,
    SubstrateSimulationReport,
)
from qec.benchmark.bio_signal_benchmark_battery import (
    SCHEMA_VERSION as BENCHMARK_SCHEMA_VERSION,
    BioSignalBenchmarkBatteryReport,
)

SCHEMA_VERSION = "v137.12.4"

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

_EXPECTED_SCHEMA_VERSIONS: dict[str, str] = {
    "substrate": SUBSTRATE_SCHEMA_VERSION,
    "interface": INTERFACE_SCHEMA_VERSION,
    "benchmark": BENCHMARK_SCHEMA_VERSION,
    "certification": CERTIFICATION_SCHEMA_VERSION,
}


# ---------------------------------------------------------------------------
# Canonical serialisation helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _require_bounded(value: float, field_name: str) -> float:
    """Require *value* is a finite float in [0, 1]."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    fval = float(value)
    if not math.isfinite(fval):
        raise ValueError(f"{field_name} must be finite")
    if fval < 0.0 or fval > 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]")
    return fval


def _stable_clamp(value: float) -> float:
    """Clamp a finite float into [0, 1]."""
    return max(0.0, min(1.0, float(value)))


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExperimentalResearchPackConfig:
    """Configuration for building a v137.12.4 research pack."""

    schema_version: str = SCHEMA_VERSION
    release_version: str = "v137.12.4"
    substrate_schema_version: str = SUBSTRATE_SCHEMA_VERSION
    interface_schema_version: str = INTERFACE_SCHEMA_VERSION
    benchmark_schema_version: str = BENCHMARK_SCHEMA_VERSION
    certification_schema_version: str = CERTIFICATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "release_version": self.release_version,
            "substrate_schema_version": self.substrate_schema_version,
            "interface_schema_version": self.interface_schema_version,
            "benchmark_schema_version": self.benchmark_schema_version,
            "certification_schema_version": self.certification_schema_version,
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
class ExperimentalArtifactManifest:
    """Canonical manifest recording artifact metadata for the research pack."""

    release_version: str
    module_schema_versions: Mapping[str, str]
    artifact_hashes: Mapping[str, str]
    artifact_byte_sizes: Mapping[str, int]
    frame_count: int
    node_count: int
    benchmark_case_count: int
    certification_passed: bool
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "release_version": self.release_version,
            "module_schema_versions": {
                k: self.module_schema_versions[k]
                for k in sorted(self.module_schema_versions.keys())
            },
            "artifact_hashes": {
                k: self.artifact_hashes[k]
                for k in sorted(self.artifact_hashes.keys())
            },
            "artifact_byte_sizes": {
                k: self.artifact_byte_sizes[k]
                for k in sorted(self.artifact_byte_sizes.keys())
            },
            "frame_count": self.frame_count,
            "node_count": self.node_count,
            "benchmark_case_count": self.benchmark_case_count,
            "certification_passed": self.certification_passed,
            "schema_version": self.schema_version,
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
class ExperimentalResearchPack:
    """Deterministic research pack bundling the full v137.12.x artifact stack."""

    config: ExperimentalResearchPackConfig
    manifest: ExperimentalArtifactManifest
    substrate_summary: Mapping[str, _JSONValue]
    trace_summary: Mapping[str, _JSONValue]
    benchmark_summary: Mapping[str, _JSONValue]
    certification_summary: Mapping[str, _JSONValue]
    summary_metrics: Mapping[str, float]
    lineage_hash_chain: tuple[str, ...]
    stable_hash: str
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "config": self.config.to_dict(),
            "manifest": self.manifest.to_dict(),
            "substrate_summary": {
                k: self.substrate_summary[k]
                for k in sorted(self.substrate_summary.keys())
            },
            "trace_summary": {
                k: self.trace_summary[k]
                for k in sorted(self.trace_summary.keys())
            },
            "benchmark_summary": {
                k: self.benchmark_summary[k]
                for k in sorted(self.benchmark_summary.keys())
            },
            "certification_summary": {
                k: self.certification_summary[k]
                for k in sorted(self.certification_summary.keys())
            },
            "summary_metrics": {
                k: float(self.summary_metrics[k])
                for k in sorted(self.summary_metrics.keys())
            },
            "lineage_hash_chain": self.lineage_hash_chain,
            "stable_hash": self.stable_hash,
            "schema_version": self.schema_version,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("stable_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_receipt_chain(self) -> tuple[str, ...]:
        return (*self.lineage_hash_chain, self.stable_hash)


@dataclass(frozen=True)
class ExperimentalResearchReceipt:
    """Deterministic receipt certifying a completed research pack."""

    receipt_hash: str
    pack_hash: str
    manifest_hash: str
    certification_passed: bool
    global_reproducibility_score: float
    receipt_chain: tuple[str, ...]
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "receipt_hash": self.receipt_hash,
            "pack_hash": self.pack_hash,
            "manifest_hash": self.manifest_hash,
            "certification_passed": self.certification_passed,
            "global_reproducibility_score": self.global_reproducibility_score,
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

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())

    def stable_receipt_chain(self) -> tuple[str, ...]:
        return (*self.receipt_chain, self.receipt_hash)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_research_pack_inputs(
    *,
    substrate_report: SubstrateSimulationReport,
    trace: HybridSignalTrace,
    interface_receipt: HybridSignalReceipt,
    benchmark_report: BioSignalBenchmarkBatteryReport,
    certification_report: HybridReplayCertificationReport,
    config: ExperimentalResearchPackConfig,
) -> None:
    """Fail fast on broken lineage, missing artifacts, or schema mismatches."""
    # Schema version checks
    if config.schema_version != SCHEMA_VERSION:
        raise ValueError("schema mismatch: research pack config schema")
    if substrate_report.schema_version != config.substrate_schema_version:
        raise ValueError("schema mismatch: substrate report schema")
    if trace.schema_version != config.interface_schema_version:
        raise ValueError("schema mismatch: trace schema")
    if interface_receipt.schema_version != config.interface_schema_version:
        raise ValueError("schema mismatch: interface receipt schema")
    if benchmark_report.schema_version != config.benchmark_schema_version:
        raise ValueError("schema mismatch: benchmark report schema")
    if certification_report.config.schema_version != config.certification_schema_version:
        raise ValueError("schema mismatch: certification report schema")

    # Lineage checks
    if trace.input_stable_hash != substrate_report.stable_hash:
        raise ValueError("broken lineage: trace input hash mismatch")
    if interface_receipt.input_stable_hash != substrate_report.stable_hash:
        raise ValueError("broken lineage: interface receipt input hash mismatch")
    if interface_receipt.output_stable_hash != trace.stable_hash:
        raise ValueError("broken lineage: interface receipt output hash mismatch")

    # Certification evidence lineage
    if certification_report.evidence.substrate_hash != substrate_report.stable_hash:
        raise ValueError("broken lineage: certification substrate hash mismatch")
    if certification_report.evidence.trace_hash != trace.stable_hash:
        raise ValueError("broken lineage: certification trace hash mismatch")
    if certification_report.evidence.benchmark_hash != benchmark_report.stable_hash:
        raise ValueError("broken lineage: certification benchmark hash mismatch")

    # Missing artifacts
    if not substrate_report.receipt.validation_passed:
        raise ValueError("missing artifact: invalid substrate receipt")
    if not interface_receipt.validation_passed:
        raise ValueError("missing artifact: invalid interface receipt")
    if len(benchmark_report.results) == 0:
        raise ValueError("missing artifact: benchmark has no results")
    if len(trace.frames) == 0:
        raise ValueError("missing artifact: trace has no frames")

    # Hash integrity
    recalculated_report_hash = certification_report.stable_hash()
    if certification_report.report_hash != recalculated_report_hash:
        raise ValueError("hash mismatch: certification report hash")

    # Duplicate identity check
    benchmark_result_hashes = tuple(r.stable_hash for r in benchmark_report.results)
    if len(benchmark_result_hashes) != len(set(benchmark_result_hashes)):
        raise ValueError("duplicate artifact identity: benchmark result hashes")

    frame_hashes = tuple(f.stable_hash for f in trace.frames)
    if len(frame_hashes) != len(set(frame_hashes)):
        raise ValueError("duplicate artifact identity: frame hashes")


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_artifact_manifest(
    *,
    substrate_report: SubstrateSimulationReport,
    trace: HybridSignalTrace,
    interface_receipt: HybridSignalReceipt,
    benchmark_report: BioSignalBenchmarkBatteryReport,
    certification_report: HybridReplayCertificationReport,
    config: ExperimentalResearchPackConfig,
) -> ExperimentalArtifactManifest:
    """Build a deterministic manifest recording all artifact metadata."""
    return ExperimentalArtifactManifest(
        release_version=config.release_version,
        module_schema_versions={
            "substrate": config.substrate_schema_version,
            "interface": config.interface_schema_version,
            "benchmark": config.benchmark_schema_version,
            "certification": config.certification_schema_version,
            "research_pack": SCHEMA_VERSION,
        },
        artifact_hashes={
            "substrate_report": substrate_report.stable_hash,
            "trace": trace.stable_hash,
            "interface_receipt": interface_receipt.receipt_hash,
            "benchmark_report": benchmark_report.stable_hash,
            "certification_report": certification_report.report_hash,
        },
        artifact_byte_sizes={
            "substrate_report": len(substrate_report.to_canonical_bytes()),
            "trace": len(trace.to_canonical_bytes()),
            "interface_receipt": len(interface_receipt.to_canonical_bytes()),
            "benchmark_report": len(benchmark_report.to_canonical_bytes()),
            "certification_report": len(certification_report.to_canonical_bytes()),
        },
        frame_count=trace.frame_count,
        node_count=len(trace.node_ids),
        benchmark_case_count=len(benchmark_report.results),
        certification_passed=certification_report.result.validation_passed,
    )


def _compute_summary_metrics(
    *,
    substrate_report: SubstrateSimulationReport,
    trace: HybridSignalTrace,
    interface_receipt: HybridSignalReceipt,
    benchmark_report: BioSignalBenchmarkBatteryReport,
    certification_report: HybridReplayCertificationReport,
) -> dict[str, float]:
    """Compute deterministic bounded summary metrics for the research pack."""
    # Simulation integrity: 1.0 if substrate receipt valid and hash recomputes
    simulation_integrity_score = _stable_clamp(
        1.0 if substrate_report.receipt.validation_passed else 0.0,
    )

    # Interface integrity: 1.0 if interface receipt valid and lineage holds
    lineage_ok = (
        interface_receipt.input_stable_hash == substrate_report.stable_hash
        and interface_receipt.output_stable_hash == trace.stable_hash
    )
    interface_integrity_score = _stable_clamp(
        1.0 if (interface_receipt.validation_passed and lineage_ok) else 0.0,
    )

    # Benchmark integrity: proportion of results with finite metrics
    valid_results = 0
    for result in benchmark_report.results:
        if all(math.isfinite(float(v)) for v in result.metrics.values()):
            valid_results += 1
    benchmark_integrity_score = _stable_clamp(
        valid_results / len(benchmark_report.results)
        if len(benchmark_report.results) > 0
        else 0.0,
    )

    # Certification score: directly from certification result
    certification_score = _stable_clamp(
        certification_report.result.certification_score,
    )

    # Global reproducibility: all-or-nothing across all domains
    global_reproducibility_score = _stable_clamp(
        1.0
        if (
            simulation_integrity_score == 1.0
            and interface_integrity_score == 1.0
            and benchmark_integrity_score == 1.0
            and certification_score == 1.0
        )
        else 0.0,
    )

    return {
        "simulation_integrity_score": simulation_integrity_score,
        "interface_integrity_score": interface_integrity_score,
        "benchmark_integrity_score": benchmark_integrity_score,
        "certification_score": certification_score,
        "global_reproducibility_score": global_reproducibility_score,
    }


def _build_substrate_summary(
    substrate_report: SubstrateSimulationReport,
) -> dict[str, _JSONValue]:
    return {
        "simulation_id": substrate_report.input.simulation_id,
        "node_count": substrate_report.input.node_count,
        "time_steps": substrate_report.input.time_steps,
        "spike_count": substrate_report.receipt.spike_count,
        "stable_hash": substrate_report.stable_hash,
        "receipt_hash": substrate_report.receipt.receipt_hash,
        "schema_version": substrate_report.schema_version,
    }


def _build_trace_summary(
    trace: HybridSignalTrace,
) -> dict[str, _JSONValue]:
    return {
        "frame_count": trace.frame_count,
        "node_count": len(trace.node_ids),
        "input_stable_hash": trace.input_stable_hash,
        "stable_hash": trace.stable_hash,
        "schema_version": trace.schema_version,
    }


def _build_benchmark_summary(
    benchmark_report: BioSignalBenchmarkBatteryReport,
) -> dict[str, _JSONValue]:
    return {
        "case_count": len(benchmark_report.results),
        "aggregate_metrics": {
            k: float(benchmark_report.aggregate_metrics[k])
            for k in sorted(benchmark_report.aggregate_metrics.keys())
        },
        "stable_hash": benchmark_report.stable_hash,
        "schema_version": benchmark_report.schema_version,
    }


def _build_certification_summary(
    certification_report: HybridReplayCertificationReport,
) -> dict[str, _JSONValue]:
    return {
        "certification_id": certification_report.certification_id,
        "validation_passed": certification_report.result.validation_passed,
        "certification_score": certification_report.result.certification_score,
        "byte_identity_passed": certification_report.result.byte_identity_passed,
        "hash_lineage_passed": certification_report.result.hash_lineage_passed,
        "structural_replay_passed": certification_report.result.structural_replay_passed,
        "metric_replay_passed": certification_report.result.metric_replay_passed,
        "cross_layer_passed": certification_report.result.cross_layer_passed,
        "report_hash": certification_report.report_hash,
        "schema_version": certification_report.config.schema_version,
    }


def build_experimental_research_pack(
    *,
    substrate_report: SubstrateSimulationReport,
    trace: HybridSignalTrace,
    interface_receipt: HybridSignalReceipt,
    benchmark_report: BioSignalBenchmarkBatteryReport,
    certification_report: HybridReplayCertificationReport,
    config: ExperimentalResearchPackConfig | None = None,
) -> ExperimentalResearchPack:
    """Build a deterministic research pack from the full v137.12.x artifact stack."""
    effective_config = config or ExperimentalResearchPackConfig()

    _validate_research_pack_inputs(
        substrate_report=substrate_report,
        trace=trace,
        interface_receipt=interface_receipt,
        benchmark_report=benchmark_report,
        certification_report=certification_report,
        config=effective_config,
    )

    manifest = build_artifact_manifest(
        substrate_report=substrate_report,
        trace=trace,
        interface_receipt=interface_receipt,
        benchmark_report=benchmark_report,
        certification_report=certification_report,
        config=effective_config,
    )

    summary_metrics = _compute_summary_metrics(
        substrate_report=substrate_report,
        trace=trace,
        interface_receipt=interface_receipt,
        benchmark_report=benchmark_report,
        certification_report=certification_report,
    )

    # Validate all metrics are bounded
    for key, value in summary_metrics.items():
        _require_bounded(value, key)

    lineage_hash_chain = (
        substrate_report.stable_hash,
        trace.stable_hash,
        interface_receipt.receipt_hash,
        benchmark_report.stable_hash,
        certification_report.report_hash,
        manifest.stable_hash(),
    )

    proto = ExperimentalResearchPack(
        config=effective_config,
        manifest=manifest,
        substrate_summary=_build_substrate_summary(substrate_report),
        trace_summary=_build_trace_summary(trace),
        benchmark_summary=_build_benchmark_summary(benchmark_report),
        certification_summary=_build_certification_summary(certification_report),
        summary_metrics=summary_metrics,
        lineage_hash_chain=lineage_hash_chain,
        stable_hash="",
        schema_version=SCHEMA_VERSION,
    )
    return ExperimentalResearchPack(
        config=proto.config,
        manifest=proto.manifest,
        substrate_summary=proto.substrate_summary,
        trace_summary=proto.trace_summary,
        benchmark_summary=proto.benchmark_summary,
        certification_summary=proto.certification_summary,
        summary_metrics=proto.summary_metrics,
        lineage_hash_chain=proto.lineage_hash_chain,
        stable_hash=_sha256_hex(proto.to_hash_payload_dict()),
        schema_version=proto.schema_version,
    )


def build_research_receipt(
    pack: ExperimentalResearchPack,
) -> ExperimentalResearchReceipt:
    """Build a deterministic receipt for a completed research pack."""
    if not pack.stable_hash:
        raise ValueError("invalid byte export: pack has no stable hash")

    manifest_hash = pack.manifest.stable_hash()
    certification_passed = pack.manifest.certification_passed
    global_reproducibility_score = _require_bounded(
        float(pack.summary_metrics["global_reproducibility_score"]),
        "global_reproducibility_score",
    )

    receipt_chain = pack.stable_receipt_chain()

    proto = ExperimentalResearchReceipt(
        receipt_hash="",
        pack_hash=pack.stable_hash,
        manifest_hash=manifest_hash,
        certification_passed=certification_passed,
        global_reproducibility_score=global_reproducibility_score,
        receipt_chain=receipt_chain,
    )
    return ExperimentalResearchReceipt(
        receipt_hash=proto.stable_hash(),
        pack_hash=proto.pack_hash,
        manifest_hash=proto.manifest_hash,
        certification_passed=proto.certification_passed,
        global_reproducibility_score=proto.global_reproducibility_score,
        receipt_chain=proto.receipt_chain,
    )


def export_research_pack_json(
    pack: ExperimentalResearchPack,
    receipt: ExperimentalResearchReceipt,
) -> str:
    """Export the full research pack as a canonical JSON string."""
    if receipt.pack_hash != pack.stable_hash:
        raise ValueError("malformed receipt: receipt pack_hash does not match pack")
    payload = {
        "pack": pack.to_dict(),
        "receipt": receipt.to_dict(),
    }
    return _canonical_json(payload)


def build_ascii_research_summary(
    pack: ExperimentalResearchPack,
    receipt: ExperimentalResearchReceipt,
) -> str:
    """Build a human-readable ASCII summary of the research pack."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("EXPERIMENTAL RESEARCH PACK SUMMARY")
    lines.append(f"Release: {pack.config.release_version}")
    lines.append(f"Schema:  {pack.schema_version}")
    lines.append("=" * 72)
    lines.append("")
    lines.append("--- Artifact Manifest ---")
    lines.append(f"  Substrate hash:      {pack.manifest.artifact_hashes['substrate_report']}")
    lines.append(f"  Trace hash:          {pack.manifest.artifact_hashes['trace']}")
    lines.append(f"  Interface receipt:    {pack.manifest.artifact_hashes['interface_receipt']}")
    lines.append(f"  Benchmark hash:      {pack.manifest.artifact_hashes['benchmark_report']}")
    lines.append(f"  Certification hash:  {pack.manifest.artifact_hashes['certification_report']}")
    lines.append(f"  Frame count:         {pack.manifest.frame_count}")
    lines.append(f"  Node count:          {pack.manifest.node_count}")
    lines.append(f"  Benchmark cases:     {pack.manifest.benchmark_case_count}")
    lines.append(f"  Certification:       {'PASSED' if pack.manifest.certification_passed else 'FAILED'}")
    lines.append("")
    lines.append("--- Summary Metrics ---")
    for key in sorted(pack.summary_metrics.keys()):
        lines.append(f"  {key}: {pack.summary_metrics[key]:.6f}")
    lines.append("")
    lines.append("--- Lineage Hash Chain ---")
    for idx, h in enumerate(pack.lineage_hash_chain):
        lines.append(f"  [{idx}] {h}")
    lines.append("")
    lines.append("--- Receipt ---")
    lines.append(f"  Pack hash:     {receipt.pack_hash}")
    lines.append(f"  Receipt hash:  {receipt.receipt_hash}")
    lines.append(f"  Manifest hash: {receipt.manifest_hash}")
    lines.append(f"  Reproducibility score: {receipt.global_reproducibility_score:.6f}")
    lines.append("=" * 72)
    return "\n".join(lines)


def run_experimental_research_pack(
    *,
    substrate_report: SubstrateSimulationReport,
    trace: HybridSignalTrace,
    interface_receipt: HybridSignalReceipt,
    benchmark_report: BioSignalBenchmarkBatteryReport,
    certification_report: HybridReplayCertificationReport,
    config: ExperimentalResearchPackConfig | None = None,
) -> tuple[ExperimentalResearchPack, ExperimentalResearchReceipt]:
    """Run the full research pack pipeline: build pack, build receipt, return both."""
    pack = build_experimental_research_pack(
        substrate_report=substrate_report,
        trace=trace,
        interface_receipt=interface_receipt,
        benchmark_report=benchmark_report,
        certification_report=certification_report,
        config=config,
    )
    receipt = build_research_receipt(pack)
    return pack, receipt


__all__ = [
    "SCHEMA_VERSION",
    "ExperimentalResearchPackConfig",
    "ExperimentalArtifactManifest",
    "ExperimentalResearchPack",
    "ExperimentalResearchReceipt",
    "build_experimental_research_pack",
    "build_artifact_manifest",
    "build_research_receipt",
    "export_research_pack_json",
    "build_ascii_research_summary",
    "run_experimental_research_pack",
]
