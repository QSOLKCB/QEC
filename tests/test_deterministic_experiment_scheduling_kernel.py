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
    normalize_experiment_schedule_input,
    traverse_deterministic_experiment_schedule,
)


def _sample_plan():
    tasks = (
        ResearchTask(task_id="task_proof", task_kind="proof", source_ref="node:0", priority=0, task_epoch=0),
        ResearchTask(task_id="task_validation", task_kind="validation", source_ref="node:1", priority=1, task_epoch=0),
        ResearchTask(task_id="task_benchmark", task_kind="benchmark", source_ref="node:2", priority=2, task_epoch=1),
        ResearchTask(task_id="task_release", task_kind="release_audit", source_ref="node:3", priority=3, task_epoch=1),
    )
    steps = (
        OrchestrationStep(step_id="exp_proof", task_id="task_proof", execution_order=0, dependency_refs=(), step_epoch=0),
        OrchestrationStep(
            step_id="exp_validation",
            task_id="task_validation",
            execution_order=1,
            dependency_refs=("exp_proof",),
            step_epoch=0,
        ),
        OrchestrationStep(
            step_id="exp_benchmark",
            task_id="task_benchmark",
            execution_order=2,
            dependency_refs=("exp_validation",),
            step_epoch=1,
        ),
        OrchestrationStep(
            step_id="exp_release",
            task_id="task_release",
            execution_order=3,
            dependency_refs=("exp_benchmark",),
            step_epoch=1,
        ),
    )
    return build_autonomous_research_plan("plan_v137_17_1", tasks, steps)


def _sample_lanes() -> tuple[SchedulingLane, ...]:
    return (
        SchedulingLane(lane_id="lane_validation_0", lane_kind="validation", capacity=1, lane_epoch=0),
        SchedulingLane(lane_id="lane_benchmark_0", lane_kind="benchmark", capacity=1, lane_epoch=0),
        SchedulingLane(lane_id="lane_release_0", lane_kind="release_audit", capacity=1, lane_epoch=0),
        SchedulingLane(lane_id="lane_proof_0", lane_kind="proof", capacity=2, lane_epoch=0),
    )


def test_repeated_run_byte_identity() -> None:
    plan = _sample_plan()
    a = build_deterministic_experiment_schedule("schedule_a", plan, _sample_lanes())
    b = build_deterministic_experiment_schedule("schedule_a", plan, _sample_lanes())
    assert a.to_canonical_bytes() == b.to_canonical_bytes()


def test_repeated_run_schedule_hash_identity() -> None:
    plan = _sample_plan()
    a = build_deterministic_experiment_schedule("schedule_hash", plan, _sample_lanes())
    b = build_deterministic_experiment_schedule("schedule_hash", plan, _sample_lanes())
    assert a.schedule_hash == b.schedule_hash


def test_duplicate_experiment_rejection() -> None:
    tasks = (ResearchTask(task_id="task_proof", task_kind="proof", source_ref="node:0", priority=0, task_epoch=0),)
    steps = (
        OrchestrationStep(step_id="exp_x", task_id="task_proof", execution_order=0, dependency_refs=(), step_epoch=0),
        OrchestrationStep(step_id="exp_x", task_id="task_proof", execution_order=1, dependency_refs=(), step_epoch=0),
    )
    with pytest.raises(ValueError, match="duplicate step IDs"):
        bad_plan = build_autonomous_research_plan("plan_dup_exp", tasks, steps)
        build_deterministic_experiment_schedule("schedule_dup_exp", bad_plan, _sample_lanes())


def test_duplicate_lane_rejection() -> None:
    plan = _sample_plan()
    lanes = _sample_lanes() + (_sample_lanes()[0],)
    with pytest.raises(ValueError, match="duplicate lane IDs"):
        normalize_experiment_schedule_input(plan, lanes)


def test_invalid_capacity_rejection() -> None:
    plan = _sample_plan()
    lanes = (
        SchedulingLane(lane_id="lane_validation_0", lane_kind="validation", capacity=-1, lane_epoch=0),
        SchedulingLane(lane_id="lane_benchmark_0", lane_kind="benchmark", capacity=1, lane_epoch=0),
        SchedulingLane(lane_id="lane_release_0", lane_kind="release_audit", capacity=1, lane_epoch=0),
        SchedulingLane(lane_id="lane_proof_0", lane_kind="proof", capacity=1, lane_epoch=0),
    )
    with pytest.raises(ValueError, match="capacity must be non-negative"):
        normalize_experiment_schedule_input(plan, lanes)


def test_deterministic_execution_traversal() -> None:
    schedule = build_deterministic_experiment_schedule("schedule_exec", _sample_plan(), _sample_lanes())
    a = traverse_deterministic_experiment_schedule(schedule, "execution")
    b = traverse_deterministic_experiment_schedule(schedule, "execution")
    assert a.ordered_experiment_trace == b.ordered_experiment_trace
    assert a.ordered_lane_trace == b.ordered_lane_trace
    assert a.traversal_hash == b.traversal_hash


def test_deterministic_lane_traversal() -> None:
    schedule = build_deterministic_experiment_schedule("schedule_lane", _sample_plan(), _sample_lanes())
    a = traverse_deterministic_experiment_schedule(schedule, "lane")
    b = traverse_deterministic_experiment_schedule(schedule, "lane")
    assert a.ordered_experiment_trace == b.ordered_experiment_trace
    assert a.ordered_lane_trace == b.ordered_lane_trace
    assert a.traversal_hash == b.traversal_hash


def test_deterministic_audit_traversal() -> None:
    schedule = build_deterministic_experiment_schedule("schedule_audit", _sample_plan(), _sample_lanes())
    a = traverse_deterministic_experiment_schedule(schedule, "audit")
    b = traverse_deterministic_experiment_schedule(schedule, "audit")
    assert a.ordered_experiment_trace == b.ordered_experiment_trace
    assert a.ordered_lane_trace == b.ordered_lane_trace
    assert a.traversal_hash == b.traversal_hash


def test_capacity_traversal() -> None:
    schedule = build_deterministic_experiment_schedule("schedule_capacity", _sample_plan(), _sample_lanes())
    a = traverse_deterministic_experiment_schedule(schedule, "capacity")
    b = traverse_deterministic_experiment_schedule(schedule, "capacity")
    assert a.ordered_experiment_trace == b.ordered_experiment_trace
    assert a.ordered_lane_trace == b.ordered_lane_trace
    assert a.traversal_hash == b.traversal_hash


def test_canonical_export_stability() -> None:
    schedule = build_deterministic_experiment_schedule("schedule_export", _sample_plan(), _sample_lanes())
    assert schedule.to_canonical_json() == schedule.to_canonical_json()
    assert schedule.to_canonical_bytes() == schedule.to_canonical_bytes()
