from __future__ import annotations

import pytest

from qec.orchestration.autonomous_research_orchestration_kernel import (
    OrchestrationStep,
    ResearchTask,
    build_autonomous_research_plan,
)
from qec.orchestration.deterministic_experiment_scheduling_kernel import (
    SchedulingLane,
    build_deterministic_experiment_schedule,
)
from qec.orchestration.replay_safe_benchmark_pipeline_kernel import (
    BenchmarkResult,
    BenchmarkStage,
    build_replay_safe_benchmark_pipeline,
    normalize_benchmark_pipeline_input,
    traverse_replay_safe_benchmark_pipeline,
)


def _sample_schedule():
    tasks = (
        ResearchTask(task_id="task_prepare", task_kind="experiment", source_ref="node:0", priority=0, task_epoch=0),
        ResearchTask(task_id="task_execute", task_kind="benchmark", source_ref="node:1", priority=1, task_epoch=0),
        ResearchTask(task_id="task_verify", task_kind="validation", source_ref="node:2", priority=2, task_epoch=1),
        ResearchTask(task_id="task_audit", task_kind="release_audit", source_ref="node:3", priority=3, task_epoch=1),
    )
    steps = (
        OrchestrationStep(step_id="exp_prepare", task_id="task_prepare", execution_order=0, dependency_refs=(), step_epoch=0),
        OrchestrationStep(
            step_id="exp_execute",
            task_id="task_execute",
            execution_order=1,
            dependency_refs=("exp_prepare",),
            step_epoch=0,
        ),
        OrchestrationStep(
            step_id="exp_verify",
            task_id="task_verify",
            execution_order=2,
            dependency_refs=("exp_execute",),
            step_epoch=1,
        ),
        OrchestrationStep(
            step_id="exp_audit",
            task_id="task_audit",
            execution_order=3,
            dependency_refs=("exp_verify",),
            step_epoch=1,
        ),
    )
    plan = build_autonomous_research_plan("plan_v137_17_2", tasks, steps)
    lanes = (
        SchedulingLane(lane_id="lane_experiment_0", lane_kind="experiment", capacity=1, lane_epoch=0),
        SchedulingLane(lane_id="lane_benchmark_0", lane_kind="benchmark", capacity=1, lane_epoch=0),
        SchedulingLane(lane_id="lane_validation_0", lane_kind="validation", capacity=1, lane_epoch=0),
        SchedulingLane(lane_id="lane_audit_0", lane_kind="release_audit", capacity=1, lane_epoch=0),
    )
    return build_deterministic_experiment_schedule("schedule_v137_17_2", plan, lanes)


def _sample_stages_and_results(schedule):
    exp_ids = [exp.experiment_id for exp in schedule.scheduled_experiments]
    stages = (
        BenchmarkStage(stage_id="stage_prepare", stage_kind="prepare", input_ref=exp_ids[0], stage_order=0, stage_epoch=0),
        BenchmarkStage(stage_id="stage_execute", stage_kind="execute", input_ref=exp_ids[1], stage_order=1, stage_epoch=0),
        BenchmarkStage(stage_id="stage_verify", stage_kind="verify", input_ref=exp_ids[2], stage_order=0, stage_epoch=1),
        BenchmarkStage(stage_id="stage_audit", stage_kind="audit", input_ref=exp_ids[3], stage_order=1, stage_epoch=1),
    )
    results = (
        BenchmarkResult(
            result_id="result_metric",
            stage_id="stage_execute",
            experiment_id=exp_ids[1],
            result_kind="metric",
            result_hash="placeholder",
            result_epoch=0,
        ),
        BenchmarkResult(
            result_id="result_verification",
            stage_id="stage_verify",
            experiment_id=exp_ids[2],
            result_kind="verification",
            result_hash="placeholder",
            result_epoch=1,
        ),
        BenchmarkResult(
            result_id="result_audit",
            stage_id="stage_audit",
            experiment_id=exp_ids[3],
            result_kind="audit",
            result_hash="placeholder",
            result_epoch=2,
        ),
        BenchmarkResult(
            result_id="result_artifact",
            stage_id="stage_audit",
            experiment_id=exp_ids[3],
            result_kind="artifact",
            result_hash="placeholder",
            result_epoch=3,
        ),
    )
    return stages, results


def test_repeated_run_byte_identity() -> None:
    schedule = _sample_schedule()
    stages, results = _sample_stages_and_results(schedule)
    a = build_replay_safe_benchmark_pipeline("pipeline_a", schedule, stages, results)
    b = build_replay_safe_benchmark_pipeline("pipeline_a", schedule, stages, results)
    assert a.to_canonical_bytes() == b.to_canonical_bytes()


def test_repeated_run_pipeline_hash_identity() -> None:
    schedule = _sample_schedule()
    stages, results = _sample_stages_and_results(schedule)
    a = build_replay_safe_benchmark_pipeline("pipeline_hash", schedule, stages, results)
    b = build_replay_safe_benchmark_pipeline("pipeline_hash", schedule, stages, results)
    assert a.pipeline_hash == b.pipeline_hash


def test_duplicate_stage_rejection() -> None:
    schedule = _sample_schedule()
    stages, results = _sample_stages_and_results(schedule)
    with pytest.raises(ValueError, match="duplicate stage IDs"):
        normalize_benchmark_pipeline_input(schedule, stages + (stages[0],), results)


def test_duplicate_result_rejection() -> None:
    schedule = _sample_schedule()
    stages, results = _sample_stages_and_results(schedule)
    with pytest.raises(ValueError, match="duplicate result IDs"):
        normalize_benchmark_pipeline_input(schedule, stages, results + (results[0],))


def test_invalid_stage_order_rejection() -> None:
    schedule = _sample_schedule()
    stages, results = _sample_stages_and_results(schedule)
    bad_stages = (
        BenchmarkStage(stage_id="stage_prepare", stage_kind="prepare", input_ref=stages[0].input_ref, stage_order=0, stage_epoch=0),
        BenchmarkStage(stage_id="stage_execute", stage_kind="execute", input_ref=stages[1].input_ref, stage_order=2, stage_epoch=0),
    )
    with pytest.raises(ValueError, match="invalid stage order"):
        normalize_benchmark_pipeline_input(schedule, bad_stages, results)


def test_deterministic_execution_traversal() -> None:
    schedule = _sample_schedule()
    stages, results = _sample_stages_and_results(schedule)
    pipeline = build_replay_safe_benchmark_pipeline("pipeline_exec", schedule, stages, results)
    a = traverse_replay_safe_benchmark_pipeline(pipeline, "execution")
    b = traverse_replay_safe_benchmark_pipeline(pipeline, "execution")
    assert a.ordered_stage_trace == b.ordered_stage_trace
    assert a.ordered_result_trace == b.ordered_result_trace
    assert a.traversal_hash == b.traversal_hash


def test_deterministic_verification_traversal() -> None:
    schedule = _sample_schedule()
    stages, results = _sample_stages_and_results(schedule)
    pipeline = build_replay_safe_benchmark_pipeline("pipeline_verification", schedule, stages, results)
    a = traverse_replay_safe_benchmark_pipeline(pipeline, "verification")
    b = traverse_replay_safe_benchmark_pipeline(pipeline, "verification")
    assert a.ordered_stage_trace == b.ordered_stage_trace
    assert a.ordered_result_trace == b.ordered_result_trace
    assert a.traversal_hash == b.traversal_hash


def test_deterministic_audit_traversal() -> None:
    schedule = _sample_schedule()
    stages, results = _sample_stages_and_results(schedule)
    pipeline = build_replay_safe_benchmark_pipeline("pipeline_audit", schedule, stages, results)
    a = traverse_replay_safe_benchmark_pipeline(pipeline, "audit")
    b = traverse_replay_safe_benchmark_pipeline(pipeline, "audit")
    assert a.ordered_stage_trace == b.ordered_stage_trace
    assert a.ordered_result_trace == b.ordered_result_trace
    assert a.traversal_hash == b.traversal_hash


def test_deterministic_artifact_traversal() -> None:
    schedule = _sample_schedule()
    stages, results = _sample_stages_and_results(schedule)
    pipeline = build_replay_safe_benchmark_pipeline("pipeline_artifact", schedule, stages, results)
    a = traverse_replay_safe_benchmark_pipeline(pipeline, "artifact")
    b = traverse_replay_safe_benchmark_pipeline(pipeline, "artifact")
    assert a.ordered_stage_trace == b.ordered_stage_trace
    assert a.ordered_result_trace == b.ordered_result_trace
    assert a.traversal_hash == b.traversal_hash


def test_canonical_export_stability() -> None:
    schedule = _sample_schedule()
    stages, results = _sample_stages_and_results(schedule)
    pipeline = build_replay_safe_benchmark_pipeline("pipeline_export", schedule, stages, results)
    assert pipeline.to_canonical_json() == pipeline.to_canonical_json()
    assert pipeline.to_canonical_bytes() == pipeline.to_canonical_bytes()
