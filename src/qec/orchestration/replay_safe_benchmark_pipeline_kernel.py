"""v137.17.2 — Replay-Safe Benchmark Pipeline Kernel.

Deterministic bounded benchmark pipeline construction and traversal over
DeterministicExperimentSchedule artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

from qec.orchestration.deterministic_experiment_scheduling_kernel import (
    DeterministicExperimentSchedule,
    ScheduledExperiment,
)


VALID_BENCHMARK_STAGE_KINDS: Tuple[str, ...] = (
    "prepare",
    "execute",
    "verify",
    "aggregate",
    "audit",
)

VALID_BENCHMARK_RESULT_KINDS: Tuple[str, ...] = (
    "metric",
    "verification",
    "audit",
    "artifact",
)

VALID_BENCHMARK_TRAVERSAL_MODES: Tuple[str, ...] = (
    "execution",
    "verification",
    "audit",
    "artifact",
)


BenchmarkStageLike = Union["BenchmarkStage", Mapping[str, Any]]
BenchmarkResultLike = Union["BenchmarkResult", Mapping[str, Any]]
PipelineLike = Union["ReplaySafeBenchmarkPipeline", Mapping[str, Any]]
ScheduleLike = Union[DeterministicExperimentSchedule, Mapping[str, Any]]


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _canonical_bytes(data: Any) -> bytes:
    return _canonical_json(data).encode("utf-8")


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a non-empty string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _require_non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer, not bool")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be a non-negative integer")
    if parsed < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return parsed


@dataclass(frozen=True)
class BenchmarkStage:
    stage_id: str
    stage_kind: str
    input_ref: str
    stage_order: int
    stage_epoch: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "stage_kind": self.stage_kind,
            "input_ref": self.input_ref,
            "stage_order": self.stage_order,
            "stage_epoch": self.stage_epoch,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class BenchmarkResult:
    result_id: str
    stage_id: str
    experiment_id: str
    result_kind: str
    result_hash: str
    result_epoch: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "result_id": self.result_id,
            "stage_id": self.stage_id,
            "experiment_id": self.experiment_id,
            "result_kind": self.result_kind,
            "result_hash": self.result_hash,
            "result_epoch": self.result_epoch,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class ReplaySafeBenchmarkPipeline:
    pipeline_id: str
    stages: Tuple[BenchmarkStage, ...]
    results: Tuple[BenchmarkResult, ...]
    pipeline_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "stages": [stage.to_dict() for stage in self.stages],
            "results": [result.to_dict() for result in self.results],
            "pipeline_hash": self.pipeline_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class BenchmarkPipelineValidationReport:
    pipeline_id: str
    is_valid: bool
    uniqueness_ok: bool
    stage_validity_ok: bool
    result_validity_ok: bool
    ordering_validity_ok: bool
    lineage_validity_ok: bool
    violations: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "is_valid": self.is_valid,
            "uniqueness_ok": self.uniqueness_ok,
            "stage_validity_ok": self.stage_validity_ok,
            "result_validity_ok": self.result_validity_ok,
            "ordering_validity_ok": self.ordering_validity_ok,
            "lineage_validity_ok": self.lineage_validity_ok,
            "violations": list(self.violations),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class BenchmarkPipelineExecutionReceipt:
    receipt_id: str
    pipeline_id: str
    pipeline_hash: str
    traversal_mode: str
    ordered_stage_trace: Tuple[str, ...]
    ordered_result_trace: Tuple[str, ...]
    traversal_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "pipeline_id": self.pipeline_id,
            "pipeline_hash": self.pipeline_hash,
            "traversal_mode": self.traversal_mode,
            "ordered_stage_trace": list(self.ordered_stage_trace),
            "ordered_result_trace": list(self.ordered_result_trace),
            "traversal_hash": self.traversal_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


def _stage_sort_key(stage: BenchmarkStage) -> Tuple[int, int, str]:
    return (stage.stage_epoch, stage.stage_order, stage.stage_id)


def _result_sort_key(result: BenchmarkResult) -> Tuple[int, str]:
    return (result.result_epoch, result.result_id)


def _normalize_schedule(schedule: ScheduleLike) -> DeterministicExperimentSchedule:
    if isinstance(schedule, DeterministicExperimentSchedule):
        return schedule
    if not isinstance(schedule, Mapping):
        raise ValueError("schedule must be mapping or DeterministicExperimentSchedule")

    raw_experiments = schedule.get("scheduled_experiments", ())
    scheduled_experiments: List[ScheduledExperiment] = []
    for raw in raw_experiments:
        if isinstance(raw, ScheduledExperiment):
            scheduled_experiments.append(raw)
        elif isinstance(raw, Mapping):
            scheduled_experiments.append(
                ScheduledExperiment(
                    experiment_id=_require_non_empty_string(raw.get("experiment_id", ""), "experiment_id"),
                    task_id=_require_non_empty_string(raw.get("task_id", ""), "task_id"),
                    lane_id=_require_non_empty_string(raw.get("lane_id", ""), "lane_id"),
                    execution_slot=_require_non_negative_int(raw.get("execution_slot", 0), "execution_slot"),
                    priority=_require_non_negative_int(raw.get("priority", 0), "priority"),
                    schedule_epoch=_require_non_negative_int(raw.get("schedule_epoch", 0), "schedule_epoch"),
                )
            )
        else:
            raise ValueError("scheduled experiment must be mapping or ScheduledExperiment")

    return DeterministicExperimentSchedule(
        schedule_id=_require_non_empty_string(schedule.get("schedule_id", ""), "schedule_id"),
        lanes=tuple(schedule.get("lanes", ())),
        scheduled_experiments=tuple(scheduled_experiments),
        schedule_hash=_require_non_empty_string(schedule.get("schedule_hash", ""), "schedule_hash"),
    )


def _normalize_stage(raw: BenchmarkStageLike) -> BenchmarkStage:
    if isinstance(raw, BenchmarkStage):
        stage = BenchmarkStage(
            stage_id=_require_non_empty_string(raw.stage_id, "stage_id"),
            stage_kind=_require_non_empty_string(raw.stage_kind, "stage_kind"),
            input_ref=_require_non_empty_string(raw.input_ref, "input_ref"),
            stage_order=_require_non_negative_int(raw.stage_order, "stage_order"),
            stage_epoch=_require_non_negative_int(raw.stage_epoch, "stage_epoch"),
        )
    else:
        if not isinstance(raw, Mapping):
            raise ValueError("stage must be mapping or BenchmarkStage")
        stage = BenchmarkStage(
            stage_id=_require_non_empty_string(raw.get("stage_id", ""), "stage_id"),
            stage_kind=_require_non_empty_string(raw.get("stage_kind", ""), "stage_kind"),
            input_ref=_require_non_empty_string(raw.get("input_ref", ""), "input_ref"),
            stage_order=_require_non_negative_int(raw.get("stage_order", 0), "stage_order"),
            stage_epoch=_require_non_negative_int(raw.get("stage_epoch", 0), "stage_epoch"),
        )

    if stage.stage_kind not in VALID_BENCHMARK_STAGE_KINDS:
        raise ValueError(f"invalid stage kind: {stage.stage_kind}")
    return stage


def _normalize_result(raw: BenchmarkResultLike) -> BenchmarkResult:
    if isinstance(raw, BenchmarkResult):
        result = BenchmarkResult(
            result_id=_require_non_empty_string(raw.result_id, "result_id"),
            stage_id=_require_non_empty_string(raw.stage_id, "stage_id"),
            experiment_id=_require_non_empty_string(raw.experiment_id, "experiment_id"),
            result_kind=_require_non_empty_string(raw.result_kind, "result_kind"),
            result_hash=_require_non_empty_string(raw.result_hash, "result_hash"),
            result_epoch=_require_non_negative_int(raw.result_epoch, "result_epoch"),
        )
    else:
        if not isinstance(raw, Mapping):
            raise ValueError("result must be mapping or BenchmarkResult")
        result = BenchmarkResult(
            result_id=_require_non_empty_string(raw.get("result_id", ""), "result_id"),
            stage_id=_require_non_empty_string(raw.get("stage_id", ""), "stage_id"),
            experiment_id=_require_non_empty_string(raw.get("experiment_id", ""), "experiment_id"),
            result_kind=_require_non_empty_string(raw.get("result_kind", ""), "result_kind"),
            result_hash=_require_non_empty_string(raw.get("result_hash", ""), "result_hash"),
            result_epoch=_require_non_negative_int(raw.get("result_epoch", 0), "result_epoch"),
        )

    if result.result_kind not in VALID_BENCHMARK_RESULT_KINDS:
        raise ValueError(f"invalid result kind: {result.result_kind}")
    return result


def _lineage_result_hash(pipeline_id: str, schedule_hash: str, result: BenchmarkResult) -> str:
    payload = {
        "pipeline_id": pipeline_id,
        "schedule_hash": schedule_hash,
        "result_id": result.result_id,
        "stage_id": result.stage_id,
        "experiment_id": result.experiment_id,
        "result_kind": result.result_kind,
        "result_epoch": result.result_epoch,
    }
    return _sha256_hex(_canonical_bytes(payload))


def normalize_benchmark_pipeline_input(
    schedule: ScheduleLike,
    stages: Sequence[BenchmarkStageLike],
    results: Sequence[BenchmarkResultLike],
) -> Tuple[DeterministicExperimentSchedule, Tuple[BenchmarkStage, ...], Tuple[BenchmarkResult, ...]]:
    normalized_schedule = _normalize_schedule(schedule)
    normalized_stages = tuple(_normalize_stage(stage) for stage in stages)
    normalized_results = tuple(_normalize_result(result) for result in results)

    stage_ids = tuple(stage.stage_id for stage in normalized_stages)
    result_ids = tuple(result.result_id for result in normalized_results)
    if len(stage_ids) != len(set(stage_ids)):
        raise ValueError("duplicate stage IDs")
    if len(result_ids) != len(set(result_ids)):
        raise ValueError("duplicate result IDs")

    experiment_ids = {exp.experiment_id for exp in normalized_schedule.scheduled_experiments}
    for stage in normalized_stages:
        if stage.input_ref not in experiment_ids:
            raise ValueError(f"missing experiment refs: {stage.input_ref}")

    per_epoch_orders: Dict[int, List[int]] = {}
    for stage in normalized_stages:
        per_epoch_orders.setdefault(stage.stage_epoch, []).append(stage.stage_order)
    for orders in per_epoch_orders.values():
        unique_orders = sorted(set(orders))
        if unique_orders != list(range(len(unique_orders))):
            raise ValueError("invalid stage order")

    ordered_stages = tuple(sorted(normalized_stages, key=_stage_sort_key))
    ordered_results = tuple(sorted(normalized_results, key=_result_sort_key))
    return normalized_schedule, ordered_stages, ordered_results


def _compute_pipeline_hash(
    pipeline_id: str,
    schedule_id: str,
    schedule_hash: str,
    stages: Tuple[BenchmarkStage, ...],
    results: Tuple[BenchmarkResult, ...],
) -> str:
    payload = {
        "pipeline_id": pipeline_id,
        "schedule_id": schedule_id,
        "schedule_hash": schedule_hash,
        "stages": [stage.to_dict() for stage in stages],
        "results": [result.to_dict() for result in results],
    }
    return _sha256_hex(_canonical_bytes(payload))


def build_replay_safe_benchmark_pipeline(
    pipeline_id: str,
    schedule: ScheduleLike,
    stages: Sequence[BenchmarkStageLike],
    results: Sequence[BenchmarkResultLike],
) -> ReplaySafeBenchmarkPipeline:
    pipeline_id = _require_non_empty_string(pipeline_id, "pipeline_id")
    normalized_schedule, ordered_stages, ordered_results = normalize_benchmark_pipeline_input(schedule, stages, results)

    stage_by_id = {stage.stage_id: stage for stage in ordered_stages}
    normalized_lineage_results: List[BenchmarkResult] = []
    for result in ordered_results:
        stage = stage_by_id.get(result.stage_id)
        if stage is None:
            raise ValueError(f"missing stage refs: {result.stage_id}")
        if result.experiment_id != stage.input_ref:
            raise ValueError("lineage mismatch between stage input_ref and result experiment_id")

        stable_hash = _lineage_result_hash(pipeline_id, normalized_schedule.schedule_hash, result)
        normalized_lineage_results.append(
            BenchmarkResult(
                result_id=result.result_id,
                stage_id=result.stage_id,
                experiment_id=result.experiment_id,
                result_kind=result.result_kind,
                result_hash=stable_hash,
                result_epoch=result.result_epoch,
            )
        )

    ordered_lineage_results = tuple(sorted(normalized_lineage_results, key=_result_sort_key))
    pipeline_hash = _compute_pipeline_hash(
        pipeline_id,
        normalized_schedule.schedule_id,
        normalized_schedule.schedule_hash,
        ordered_stages,
        ordered_lineage_results,
    )

    return ReplaySafeBenchmarkPipeline(
        pipeline_id=pipeline_id,
        stages=ordered_stages,
        results=ordered_lineage_results,
        pipeline_hash=pipeline_hash,
    )


def validate_replay_safe_benchmark_pipeline(pipeline: PipelineLike) -> BenchmarkPipelineValidationReport:
    if isinstance(pipeline, ReplaySafeBenchmarkPipeline):
        candidate = pipeline
    else:
        if not isinstance(pipeline, Mapping):
            raise ValueError("pipeline must be mapping or ReplaySafeBenchmarkPipeline")
        candidate = ReplaySafeBenchmarkPipeline(
            pipeline_id=_require_non_empty_string(pipeline.get("pipeline_id", ""), "pipeline_id"),
            stages=tuple(_normalize_stage(raw) for raw in pipeline.get("stages", ())),
            results=tuple(_normalize_result(raw) for raw in pipeline.get("results", ())),
            pipeline_hash=_require_non_empty_string(pipeline.get("pipeline_hash", ""), "pipeline_hash"),
        )

    violations: List[str] = []
    uniqueness_ok = True
    stage_validity_ok = True
    result_validity_ok = True
    ordering_validity_ok = True
    lineage_validity_ok = True

    stage_ids = tuple(stage.stage_id for stage in candidate.stages)
    result_ids = tuple(result.result_id for result in candidate.results)
    if len(stage_ids) != len(set(stage_ids)):
        uniqueness_ok = False
        violations.append("duplicate_stage_ids")
    if len(result_ids) != len(set(result_ids)):
        uniqueness_ok = False
        violations.append("duplicate_result_ids")

    for stage in candidate.stages:
        if stage.stage_kind not in VALID_BENCHMARK_STAGE_KINDS:
            stage_validity_ok = False
            violations.append("invalid_stage_kind")
            break

    for result in candidate.results:
        if result.result_kind not in VALID_BENCHMARK_RESULT_KINDS:
            result_validity_ok = False
            violations.append("invalid_result_kind")
            break

    if tuple(sorted(candidate.stages, key=_stage_sort_key)) != candidate.stages:
        ordering_validity_ok = False
        violations.append("stage_ordering_invalid")
    if tuple(sorted(candidate.results, key=_result_sort_key)) != candidate.results:
        ordering_validity_ok = False
        violations.append("result_ordering_invalid")

    stage_by_id = {stage.stage_id: stage for stage in candidate.stages}
    for result in candidate.results:
        stage = stage_by_id.get(result.stage_id)
        if stage is None or stage.input_ref != result.experiment_id:
            lineage_validity_ok = False
            violations.append("lineage_invalid")
            break

    is_valid = all((uniqueness_ok, stage_validity_ok, result_validity_ok, ordering_validity_ok, lineage_validity_ok))
    return BenchmarkPipelineValidationReport(
        pipeline_id=candidate.pipeline_id,
        is_valid=is_valid,
        uniqueness_ok=uniqueness_ok,
        stage_validity_ok=stage_validity_ok,
        result_validity_ok=result_validity_ok,
        ordering_validity_ok=ordering_validity_ok,
        lineage_validity_ok=lineage_validity_ok,
        violations=tuple(violations),
    )


def traverse_replay_safe_benchmark_pipeline(
    pipeline: ReplaySafeBenchmarkPipeline,
    traversal_mode: str,
) -> BenchmarkPipelineExecutionReceipt:
    traversal_mode = _require_non_empty_string(traversal_mode, "traversal_mode")
    if traversal_mode not in VALID_BENCHMARK_TRAVERSAL_MODES:
        raise ValueError(f"unsupported traversal mode: {traversal_mode}")

    stages = pipeline.stages
    results = pipeline.results

    if traversal_mode == "execution":
        ordered_stages = stages
        ordered_results = results
    elif traversal_mode == "verification":
        ordered_stages = tuple(stage for stage in stages if stage.stage_kind in ("verify", "audit"))
        stage_ids = {stage.stage_id for stage in ordered_stages}
        ordered_results = tuple(
            result for result in results if result.stage_id in stage_ids or result.result_kind == "verification"
        )
    elif traversal_mode == "audit":
        ordered_stages = tuple(stage for stage in stages if stage.stage_kind == "audit")
        stage_ids = {stage.stage_id for stage in ordered_stages}
        ordered_results = tuple(result for result in results if result.stage_id in stage_ids or result.result_kind == "audit")
    else:
        ordered_stages = tuple(stage for stage in stages if stage.stage_kind in ("aggregate", "audit"))
        stage_ids = {stage.stage_id for stage in ordered_stages}
        ordered_results = tuple(result for result in results if result.stage_id in stage_ids or result.result_kind == "artifact")

    ordered_stage_trace = tuple(stage.stage_id for stage in ordered_stages)
    ordered_result_trace = tuple(result.result_id for result in ordered_results)

    receipt_id = f"receipt::{pipeline.pipeline_id}::{traversal_mode}"
    traversal_payload = {
        "receipt_id": receipt_id,
        "pipeline_id": pipeline.pipeline_id,
        "pipeline_hash": pipeline.pipeline_hash,
        "traversal_mode": traversal_mode,
        "ordered_stage_trace": list(ordered_stage_trace),
        "ordered_result_trace": list(ordered_result_trace),
    }
    traversal_hash = _sha256_hex(_canonical_bytes(traversal_payload))

    return BenchmarkPipelineExecutionReceipt(
        receipt_id=receipt_id,
        pipeline_id=pipeline.pipeline_id,
        pipeline_hash=pipeline.pipeline_hash,
        traversal_mode=traversal_mode,
        ordered_stage_trace=ordered_stage_trace,
        ordered_result_trace=ordered_result_trace,
        traversal_hash=traversal_hash,
    )
