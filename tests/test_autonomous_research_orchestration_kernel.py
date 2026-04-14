from __future__ import annotations

import pytest

from qec.orchestration.autonomous_research_orchestration_kernel import (
    ResearchTask,
    OrchestrationStep,
    build_autonomous_research_plan,
    normalize_research_orchestration_input,
    traverse_autonomous_research_plan,
)


def _sample_tasks() -> tuple[ResearchTask, ...]:
    return (
        ResearchTask(
            task_id="task_benchmark",
            task_kind="benchmark",
            source_ref="reasoning_graph:node_2",
            priority=2,
            task_epoch=1,
        ),
        ResearchTask(
            task_id="task_proof",
            task_kind="proof",
            source_ref="reasoning_graph:node_0",
            priority=0,
            task_epoch=0,
        ),
        ResearchTask(
            task_id="task_experiment",
            task_kind="experiment",
            source_ref="reasoning_graph:node_1",
            priority=1,
            task_epoch=0,
        ),
    )


def _sample_steps() -> tuple[OrchestrationStep, ...]:
    return (
        OrchestrationStep(
            step_id="step_validate",
            task_id="task_experiment",
            execution_order=2,
            dependency_refs=("step_experiment",),
            step_epoch=0,
        ),
        OrchestrationStep(
            step_id="step_benchmark",
            task_id="task_benchmark",
            execution_order=3,
            dependency_refs=("step_validate",),
            step_epoch=1,
        ),
        OrchestrationStep(
            step_id="step_proof",
            task_id="task_proof",
            execution_order=0,
            dependency_refs=(),
            step_epoch=0,
        ),
        OrchestrationStep(
            step_id="step_experiment",
            task_id="task_experiment",
            execution_order=1,
            dependency_refs=("step_proof",),
            step_epoch=0,
        ),
    )


def test_repeated_run_byte_identity() -> None:
    a = build_autonomous_research_plan("plan_v137_17_0", _sample_tasks(), _sample_steps())
    b = build_autonomous_research_plan("plan_v137_17_0", _sample_tasks(), _sample_steps())
    assert a.to_canonical_bytes() == b.to_canonical_bytes()


def test_repeated_run_plan_hash_identity() -> None:
    a = build_autonomous_research_plan("plan_v137_17_0", _sample_tasks(), _sample_steps())
    b = build_autonomous_research_plan("plan_v137_17_0", _sample_tasks(), _sample_steps())
    assert a.plan_hash == b.plan_hash


def test_duplicate_task_rejection() -> None:
    tasks = _sample_tasks() + (_sample_tasks()[0],)
    with pytest.raises(ValueError, match="duplicate task IDs"):
        normalize_research_orchestration_input(tasks, _sample_steps())


def test_duplicate_step_rejection() -> None:
    steps = _sample_steps() + (_sample_steps()[0],)
    with pytest.raises(ValueError, match="duplicate step IDs"):
        normalize_research_orchestration_input(_sample_tasks(), steps)


def test_invalid_dependency_rejection() -> None:
    steps = (
        OrchestrationStep(
            step_id="step_a",
            task_id="task_proof",
            execution_order=0,
            dependency_refs=("missing",),
            step_epoch=0,
        ),
    )
    with pytest.raises(ValueError, match="missing dependency refs"):
        normalize_research_orchestration_input((_sample_tasks()[1],), steps)


def test_deterministic_execution_traversal() -> None:
    plan = build_autonomous_research_plan("plan_exec", _sample_tasks(), _sample_steps())
    a = traverse_autonomous_research_plan(plan, "execution")
    b = traverse_autonomous_research_plan(plan, "execution")
    assert a.ordered_step_trace == b.ordered_step_trace
    assert a.ordered_task_trace == b.ordered_task_trace
    assert a.traversal_hash == b.traversal_hash


def test_deterministic_dependency_traversal() -> None:
    plan = build_autonomous_research_plan("plan_dep", _sample_tasks(), _sample_steps())
    a = traverse_autonomous_research_plan(plan, "dependency")
    b = traverse_autonomous_research_plan(plan, "dependency")
    assert a.ordered_step_trace == b.ordered_step_trace
    assert a.traversal_hash == b.traversal_hash


def test_deterministic_audit_traversal() -> None:
    plan = build_autonomous_research_plan("plan_audit", _sample_tasks(), _sample_steps())
    a = traverse_autonomous_research_plan(plan, "audit")
    b = traverse_autonomous_research_plan(plan, "audit")
    assert a.ordered_step_trace == b.ordered_step_trace
    assert a.traversal_hash == b.traversal_hash


def test_canonical_export_stability() -> None:
    plan = build_autonomous_research_plan("plan_export", _sample_tasks(), _sample_steps())
    assert plan.to_canonical_json() == plan.to_canonical_json()
    assert plan.to_canonical_bytes() == plan.to_canonical_bytes()
