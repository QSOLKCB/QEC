"""Terminal Kernel Handoff Release (v137.1.20).

Deterministic Layer-4 terminal handoff/freeze module for the v137.1.x line.

Guarantees:
- frozen dataclasses
- canonical JSON / bytes export
- stable SHA-256 replay identity
- deterministic ordering and deterministic tie-breaking
- fail-fast + bounded validation
- no stochastic behavior and no external I/O
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

TERMINAL_KERNEL_HANDOFF_VERSION: str = "v137.1.20"
_ROUND_DIGITS: int = 12
_MAX_BENCHMARK_SAMPLES: int = 512
_MAX_ADVISORIES: int = 256
_MAX_OBSERVATORY_METRICS: int = 512


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _round_float(value: float) -> float:
    return round(float(value), _ROUND_DIGITS)


def _require_finite(name: str, value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _require_non_empty_str(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class BenchmarkSample:
    benchmark_id: str
    operation_count: int
    determinism_score: float
    sample_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "operation_count": self.operation_count,
            "determinism_score": _round_float(self.determinism_score),
            "sample_hash": self.sample_hash,
        }


@dataclass(frozen=True)
class DeterministicBenchmarkFreezeArtifact:
    line_version: str
    sample_count: int
    baseline_operation_count: int
    mean_determinism_score: float
    frozen_samples: tuple[BenchmarkSample, ...]
    benchmark_freeze_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "line_version": self.line_version,
            "sample_count": self.sample_count,
            "baseline_operation_count": self.baseline_operation_count,
            "mean_determinism_score": _round_float(self.mean_determinism_score),
            "frozen_samples": [sample.to_dict() for sample in self.frozen_samples],
            "benchmark_freeze_hash": self.benchmark_freeze_hash,
        }


@dataclass(frozen=True)
class ReplayProvenanceAdvisoryObservatoryFreezeReport:
    line_version: str
    replay_anchor: str
    provenance_anchor: str
    advisory_codes: tuple[str, ...]
    observatory_metrics: tuple[tuple[str, float], ...]
    report_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "line_version": self.line_version,
            "replay_anchor": self.replay_anchor,
            "provenance_anchor": self.provenance_anchor,
            "advisory_codes": list(self.advisory_codes),
            "observatory_metrics": [[k, _round_float(v)] for k, v in self.observatory_metrics],
            "report_hash": self.report_hash,
        }


@dataclass(frozen=True)
class V1372MigrationContractArtifact:
    from_version: str
    to_version: str
    contract_id: str
    required_capabilities: tuple[str, ...]
    deterministic_invariants: tuple[str, ...]
    contract_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_version": self.from_version,
            "to_version": self.to_version,
            "contract_id": self.contract_id,
            "required_capabilities": list(self.required_capabilities),
            "deterministic_invariants": list(self.deterministic_invariants),
            "contract_hash": self.contract_hash,
        }


@dataclass(frozen=True)
class PlanningKernelBootstrapSchemaArtifact:
    schema_id: str
    schema_version: str
    planning_kernel: str
    required_fields: tuple[str, ...]
    strict_mode: bool
    bootstrap_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "planning_kernel": self.planning_kernel,
            "required_fields": list(self.required_fields),
            "strict_mode": self.strict_mode,
            "bootstrap_hash": self.bootstrap_hash,
        }


@dataclass(frozen=True)
class TerminalKernelHandoffRelease:
    version: str
    benchmark_freeze: DeterministicBenchmarkFreezeArtifact
    freeze_report: ReplayProvenanceAdvisoryObservatoryFreezeReport
    migration_contract: V1372MigrationContractArtifact
    bootstrap_schema: PlanningKernelBootstrapSchemaArtifact
    replay_identity: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "benchmark_freeze": self.benchmark_freeze.to_dict(),
            "freeze_report": self.freeze_report.to_dict(),
            "migration_contract": self.migration_contract.to_dict(),
            "bootstrap_schema": self.bootstrap_schema.to_dict(),
            "replay_identity": self.replay_identity,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


def _normalize_benchmark_samples(samples: Sequence[Mapping[str, Any]]) -> tuple[BenchmarkSample, ...]:
    if len(samples) == 0:
        raise ValueError("benchmark_samples must be non-empty")
    if len(samples) > _MAX_BENCHMARK_SAMPLES:
        raise ValueError(f"benchmark_samples supports at most {_MAX_BENCHMARK_SAMPLES} samples")

    normalized: list[BenchmarkSample] = []
    for idx, sample in enumerate(samples):
        benchmark_id = _require_non_empty_str(f"benchmark_samples[{idx}].benchmark_id", sample.get("benchmark_id"))

        operation_count_value = sample.get("operation_count", -1)
        if isinstance(operation_count_value, bool) or not isinstance(operation_count_value, int):
            raise ValueError(f"benchmark_samples[{idx}].operation_count must be an integer")
        operation_count = operation_count_value
        if operation_count < 0 or operation_count > 10**12:
            raise ValueError(f"benchmark_samples[{idx}].operation_count must be in [0, 1e12]")

        determinism_score = _require_finite(
            f"benchmark_samples[{idx}].determinism_score", sample.get("determinism_score", float("nan"))
        )
        if determinism_score < 0.0 or determinism_score > 1.0:
            raise ValueError(f"benchmark_samples[{idx}].determinism_score must be in [0, 1]")

        payload = {
            "benchmark_id": benchmark_id,
            "operation_count": operation_count,
            "determinism_score": _round_float(determinism_score),
        }
        normalized.append(
            BenchmarkSample(
                benchmark_id=benchmark_id,
                operation_count=operation_count,
                determinism_score=payload["determinism_score"],
                sample_hash=_hash_sha256(payload),
            )
        )

    # Deterministic ordering with explicit tie-breaking.
    return tuple(
        sorted(
            normalized,
            key=lambda s: (
                -s.determinism_score,
                s.operation_count,
                s.benchmark_id,
                s.sample_hash,
            ),
        )
    )


def _build_benchmark_freeze(samples: tuple[BenchmarkSample, ...]) -> DeterministicBenchmarkFreezeArtifact:
    baseline_operation_count = int(min(sample.operation_count for sample in samples))
    mean_score = _round_float(sum(sample.determinism_score for sample in samples) / len(samples))
    payload = {
        "line_version": TERMINAL_KERNEL_HANDOFF_VERSION,
        "sample_count": len(samples),
        "baseline_operation_count": baseline_operation_count,
        "mean_determinism_score": mean_score,
        "frozen_samples": [s.to_dict() for s in samples],
    }
    return DeterministicBenchmarkFreezeArtifact(
        line_version=TERMINAL_KERNEL_HANDOFF_VERSION,
        sample_count=len(samples),
        baseline_operation_count=baseline_operation_count,
        mean_determinism_score=mean_score,
        frozen_samples=samples,
        benchmark_freeze_hash=_hash_sha256(payload),
    )


def _build_freeze_report(
    replay_anchor: str,
    provenance_anchor: str,
    advisory_codes: Sequence[str],
    observatory_metrics: Mapping[str, float],
) -> ReplayProvenanceAdvisoryObservatoryFreezeReport:
    if len(advisory_codes) > _MAX_ADVISORIES:
        raise ValueError(f"advisory_codes supports at most {_MAX_ADVISORIES} entries")
    if len(observatory_metrics) > _MAX_OBSERVATORY_METRICS:
        raise ValueError(f"observatory_metrics supports at most {_MAX_OBSERVATORY_METRICS} entries")

    normalized_advisories = tuple(sorted({_require_non_empty_str("advisory_code", code) for code in advisory_codes}))

    normalized_metrics: list[tuple[str, float]] = []
    for key, raw_value in observatory_metrics.items():
        name = _require_non_empty_str("observatory_metric_key", key)
        value = _require_finite(f"observatory_metrics[{name}]", raw_value)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"observatory_metrics[{name}] must be in [0, 1]")
        normalized_metrics.append((name, _round_float(value)))
    normalized_metrics.sort(key=lambda item: item[0])

    metrics_tuple = tuple(normalized_metrics)
    payload = {
        "line_version": TERMINAL_KERNEL_HANDOFF_VERSION,
        "replay_anchor": replay_anchor,
        "provenance_anchor": provenance_anchor,
        "advisory_codes": list(normalized_advisories),
        "observatory_metrics": [[k, v] for k, v in metrics_tuple],
    }
    return ReplayProvenanceAdvisoryObservatoryFreezeReport(
        line_version=TERMINAL_KERNEL_HANDOFF_VERSION,
        replay_anchor=replay_anchor,
        provenance_anchor=provenance_anchor,
        advisory_codes=normalized_advisories,
        observatory_metrics=metrics_tuple,
        report_hash=_hash_sha256(payload),
    )


def _build_migration_contract() -> V1372MigrationContractArtifact:
    required_capabilities = (
        "canonical_freeze_summary",
        "deterministic_benchmark_freeze",
        "planning_kernel_bootstrap_schema",
        "replay_safe_handoff_identity",
    )
    deterministic_invariants = (
        "canonical_json_export",
        "deterministic_ordering",
        "fail_fast_validation",
        "same_input_same_bytes",
        "sha256_replay_identity",
    )
    payload = {
        "from_version": TERMINAL_KERNEL_HANDOFF_VERSION,
        "to_version": "v137.2.x",
        "contract_id": "v1371x_to_v1372x_terminal_contract",
        "required_capabilities": list(required_capabilities),
        "deterministic_invariants": list(deterministic_invariants),
    }
    return V1372MigrationContractArtifact(
        from_version=TERMINAL_KERNEL_HANDOFF_VERSION,
        to_version="v137.2.x",
        contract_id="v1371x_to_v1372x_terminal_contract",
        required_capabilities=required_capabilities,
        deterministic_invariants=deterministic_invariants,
        contract_hash=_hash_sha256(payload),
    )


def _build_bootstrap_schema() -> PlanningKernelBootstrapSchemaArtifact:
    required_fields = (
        "contract_hash",
        "deterministic_mode",
        "handoff_replay_identity",
        "input_digest",
        "kernel_epoch",
        "planning_goal",
        "trace_digest",
    )
    payload = {
        "schema_id": "autonomous_planning_kernel.bootstrap",
        "schema_version": "v1",
        "planning_kernel": "Autonomous Planning Kernel",
        "required_fields": list(required_fields),
        "strict_mode": True,
    }
    return PlanningKernelBootstrapSchemaArtifact(
        schema_id="autonomous_planning_kernel.bootstrap",
        schema_version="v1",
        planning_kernel="Autonomous Planning Kernel",
        required_fields=required_fields,
        strict_mode=True,
        bootstrap_hash=_hash_sha256(payload),
    )


def build_terminal_kernel_handoff_release(
    *,
    benchmark_samples: Sequence[Mapping[str, Any]],
    replay_anchor: str,
    provenance_anchor: str,
    advisory_codes: Sequence[str],
    observatory_metrics: Mapping[str, float],
    enable_terminal_handoff_freeze: bool = False,
) -> TerminalKernelHandoffRelease:
    """Build canonical terminal handoff artifacts for the v137.1.x line.

    This endpoint is explicitly opt-in via `enable_terminal_handoff_freeze`.
    """

    if not enable_terminal_handoff_freeze:
        raise ValueError("enable_terminal_handoff_freeze must be True for terminal handoff release")

    replay_anchor_value = _require_non_empty_str("replay_anchor", replay_anchor)
    provenance_anchor_value = _require_non_empty_str("provenance_anchor", provenance_anchor)

    samples = _normalize_benchmark_samples(benchmark_samples)
    benchmark_freeze = _build_benchmark_freeze(samples)
    freeze_report = _build_freeze_report(
        replay_anchor=replay_anchor_value,
        provenance_anchor=provenance_anchor_value,
        advisory_codes=advisory_codes,
        observatory_metrics=observatory_metrics,
    )
    migration_contract = _build_migration_contract()
    bootstrap_schema = _build_bootstrap_schema()

    replay_payload = {
        "version": TERMINAL_KERNEL_HANDOFF_VERSION,
        "benchmark_freeze_hash": benchmark_freeze.benchmark_freeze_hash,
        "freeze_report_hash": freeze_report.report_hash,
        "migration_contract_hash": migration_contract.contract_hash,
        "bootstrap_schema_hash": bootstrap_schema.bootstrap_hash,
    }
    replay_identity = _hash_sha256(replay_payload)

    return TerminalKernelHandoffRelease(
        version=TERMINAL_KERNEL_HANDOFF_VERSION,
        benchmark_freeze=benchmark_freeze,
        freeze_report=freeze_report,
        migration_contract=migration_contract,
        bootstrap_schema=bootstrap_schema,
        replay_identity=replay_identity,
    )


__all__ = [
    "TERMINAL_KERNEL_HANDOFF_VERSION",
    "BenchmarkSample",
    "DeterministicBenchmarkFreezeArtifact",
    "ReplayProvenanceAdvisoryObservatoryFreezeReport",
    "V1372MigrationContractArtifact",
    "PlanningKernelBootstrapSchemaArtifact",
    "TerminalKernelHandoffRelease",
    "build_terminal_kernel_handoff_release",
]
