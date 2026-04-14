from __future__ import annotations

import math

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
)
from qec.orchestration.research_trace_lineage_kernel import (
    ResearchTraceEdge,
    ResearchTraceNode,
    build_research_trace_lineage,
    normalize_research_trace_input,
    traverse_research_trace_lineage,
    validate_research_trace_lineage,
)


def _sample_plan_schedule_pipeline():
    tasks = (
        ResearchTask(task_id="task_proof", task_kind="proof", source_ref="node:0", priority=0, task_epoch=0),
        ResearchTask(task_id="task_validation", task_kind="validation", source_ref="node:1", priority=1, task_epoch=0),
    )
    steps = (
        OrchestrationStep(step_id="exp_proof", task_id="task_proof", execution_order=0, dependency_refs=(), step_epoch=0),
        OrchestrationStep(
            step_id="exp_validation",
            task_id="task_validation",
            execution_order=1,
            dependency_refs=("exp_proof",),
            step_epoch=1,
        ),
    )
    plan = build_autonomous_research_plan("plan_v137_17_3", tasks, steps)

    lanes = (
        SchedulingLane(lane_id="lane_proof_0", lane_kind="proof", capacity=1, lane_epoch=0),
        SchedulingLane(lane_id="lane_validation_0", lane_kind="validation", capacity=1, lane_epoch=0),
    )
    schedule = build_deterministic_experiment_schedule("schedule_v137_17_3", plan, lanes)

    stages = (
        BenchmarkStage(stage_id="stage_prepare", stage_kind="prepare", input_ref="exp_proof", stage_order=0, stage_epoch=0),
        BenchmarkStage(stage_id="stage_verify", stage_kind="verify", input_ref="exp_validation", stage_order=0, stage_epoch=1),
    )
    results = (
        BenchmarkResult(
            result_id="result_metric",
            stage_id="stage_prepare",
            experiment_id="exp_proof",
            result_kind="metric",
            result_hash="0" * 64,
            result_epoch=0,
        ),
        BenchmarkResult(
            result_id="result_verification",
            stage_id="stage_verify",
            experiment_id="exp_validation",
            result_kind="verification",
            result_hash="1" * 64,
            result_epoch=1,
        ),
    )
    pipeline = build_replay_safe_benchmark_pipeline("pipeline_v137_17_3", schedule, stages, results)
    return plan, schedule, pipeline


def test_repeated_run_byte_identity() -> None:
    plan, schedule, pipeline = _sample_plan_schedule_pipeline()
    a = build_research_trace_lineage(
        "lineage_repeat_bytes",
        plan,
        schedule,
        pipeline,
        audit_refs=("audit:primary",),
        verification_refs=("verification:primary",),
        artifact_refs=("artifact:bundle",),
    )
    b = build_research_trace_lineage(
        "lineage_repeat_bytes",
        plan,
        schedule,
        pipeline,
        audit_refs=("audit:primary",),
        verification_refs=("verification:primary",),
        artifact_refs=("artifact:bundle",),
    )
    assert a.to_canonical_bytes() == b.to_canonical_bytes()


def test_repeated_run_lineage_hash_identity() -> None:
    plan, schedule, pipeline = _sample_plan_schedule_pipeline()
    a = build_research_trace_lineage("lineage_repeat_hash", plan, schedule, pipeline)
    b = build_research_trace_lineage("lineage_repeat_hash", plan, schedule, pipeline)
    assert a.lineage_hash == b.lineage_hash


def test_duplicate_trace_rejection() -> None:
    node = ResearchTraceNode(
        trace_id="dup",
        trace_kind="plan",
        source_ref="plan:dup",
        trace_hash="a" * 64,
        trace_epoch=0,
    )
    with pytest.raises(ValueError, match="duplicate trace IDs"):
        normalize_research_trace_input((node, node), ())


def test_duplicate_edge_rejection() -> None:
    node_a = ResearchTraceNode("a", "plan", "plan:a", "a" * 64, 0)
    node_b = ResearchTraceNode("b", "schedule", "schedule:b", "b" * 64, 0)
    edge = ResearchTraceEdge("e", "b", "a", "derives_from", 1.0, 0)
    with pytest.raises(ValueError, match="duplicate edge IDs"):
        normalize_research_trace_input((node_a, node_b), (edge, edge))


def test_invalid_reference_rejection() -> None:
    node = ResearchTraceNode("a", "plan", "plan:a", "a" * 64, 0)
    edge = ResearchTraceEdge("e", "missing", "a", "derives_from", 1.0, 0)
    with pytest.raises(ValueError, match="invalid edge references"):
        normalize_research_trace_input((node,), (edge,))


def test_invalid_trace_kind_rejection() -> None:
    bad = {
        "trace_id": "n0",
        "trace_kind": "unsupported",
        "source_ref": "plan:n0",
        "trace_hash": "a" * 64,
        "trace_epoch": 0,
    }
    with pytest.raises(ValueError, match="invalid trace kind"):
        normalize_research_trace_input((bad,), ())


def test_invalid_relation_kind_rejection() -> None:
    node_a = ResearchTraceNode("a", "plan", "plan:a", "a" * 64, 0)
    node_b = ResearchTraceNode("b", "schedule", "schedule:b", "b" * 64, 0)
    bad_edge = {
        "edge_id": "e0",
        "source_trace_id": "b",
        "target_trace_id": "a",
        "relation_kind": "unsupported",
        "edge_weight": 1.0,
        "trace_epoch": 0,
    }
    with pytest.raises(ValueError, match="invalid relation kind"):
        normalize_research_trace_input((node_a, node_b), (bad_edge,))


def test_tampered_lineage_hash_rejection() -> None:
    plan, schedule, pipeline = _sample_plan_schedule_pipeline()
    lineage = build_research_trace_lineage("lineage_tamper", plan, schedule, pipeline)
    tampered = {
        "lineage_id": lineage.lineage_id,
        "nodes": [node.to_dict() for node in lineage.nodes],
        "edges": [edge.to_dict() for edge in lineage.edges],
        "lineage_hash": "f" * 64,
    }
    report = validate_research_trace_lineage(tampered)
    assert not report.is_valid
    assert "lineage_hash_mismatch" in report.violations


def test_deterministic_lineage_traversal() -> None:
    plan, schedule, pipeline = _sample_plan_schedule_pipeline()
    lineage = build_research_trace_lineage("lineage_mode", plan, schedule, pipeline)
    a = traverse_research_trace_lineage(lineage, "lineage")
    b = traverse_research_trace_lineage(lineage, "lineage")
    assert a.ordered_node_trace == b.ordered_node_trace
    assert a.ordered_edge_trace == b.ordered_edge_trace
    assert a.traversal_hash == b.traversal_hash


def test_deterministic_verification_traversal() -> None:
    plan, schedule, pipeline = _sample_plan_schedule_pipeline()
    lineage = build_research_trace_lineage(
        "verification_mode",
        plan,
        schedule,
        pipeline,
        verification_refs=("verification:v0",),
    )
    a = traverse_research_trace_lineage(lineage, "verification")
    b = traverse_research_trace_lineage(lineage, "verification")
    assert a.ordered_node_trace == b.ordered_node_trace
    assert a.ordered_edge_trace == b.ordered_edge_trace
    assert a.traversal_hash == b.traversal_hash


def test_deterministic_audit_traversal() -> None:
    plan, schedule, pipeline = _sample_plan_schedule_pipeline()
    lineage = build_research_trace_lineage("audit_mode", plan, schedule, pipeline, audit_refs=("audit:v0",))
    a = traverse_research_trace_lineage(lineage, "audit")
    b = traverse_research_trace_lineage(lineage, "audit")
    assert a.ordered_node_trace == b.ordered_node_trace
    assert a.ordered_edge_trace == b.ordered_edge_trace
    assert a.traversal_hash == b.traversal_hash


def test_deterministic_replay_traversal() -> None:
    plan, schedule, pipeline = _sample_plan_schedule_pipeline()
    lineage = build_research_trace_lineage("replay_mode", plan, schedule, pipeline, artifact_refs=("artifact:v0",))
    a = traverse_research_trace_lineage(lineage, "replay")
    b = traverse_research_trace_lineage(lineage, "replay")
    assert a.ordered_node_trace == b.ordered_node_trace
    assert a.ordered_edge_trace == b.ordered_edge_trace
    assert a.traversal_hash == b.traversal_hash


def test_canonical_export_stability() -> None:
    plan, schedule, pipeline = _sample_plan_schedule_pipeline()
    lineage = build_research_trace_lineage("canonical_export", plan, schedule, pipeline)
    assert lineage.to_canonical_json() == lineage.to_canonical_json()
    assert lineage.to_canonical_bytes() == lineage.to_canonical_bytes()


def test_negative_edge_weight_rejection() -> None:
    node_a = ResearchTraceNode("a", "plan", "plan:a", "a" * 64, 0)
    node_b = ResearchTraceNode("b", "schedule", "schedule:b", "b" * 64, 0)
    bad_edge = {
        "edge_id": "e0",
        "source_trace_id": "b",
        "target_trace_id": "a",
        "relation_kind": "derives_from",
        "edge_weight": -1.0,
        "trace_epoch": 0,
    }
    with pytest.raises(ValueError, match="non-negative and finite"):
        normalize_research_trace_input((node_a, node_b), (bad_edge,))


def test_nan_edge_weight_rejection() -> None:
    node_a = ResearchTraceNode("a", "plan", "plan:a", "a" * 64, 0)
    node_b = ResearchTraceNode("b", "schedule", "schedule:b", "b" * 64, 0)
    bad_edge = {
        "edge_id": "e0",
        "source_trace_id": "b",
        "target_trace_id": "a",
        "relation_kind": "derives_from",
        "edge_weight": math.nan,
        "trace_epoch": 0,
    }
    with pytest.raises(ValueError, match="non-negative and finite"):
        normalize_research_trace_input((node_a, node_b), (bad_edge,))


def test_inf_edge_weight_rejection() -> None:
    node_a = ResearchTraceNode("a", "plan", "plan:a", "a" * 64, 0)
    node_b = ResearchTraceNode("b", "schedule", "schedule:b", "b" * 64, 0)
    bad_edge = {
        "edge_id": "e0",
        "source_trace_id": "b",
        "target_trace_id": "a",
        "relation_kind": "derives_from",
        "edge_weight": math.inf,
        "trace_epoch": 0,
    }
    with pytest.raises(ValueError, match="non-negative and finite"):
        normalize_research_trace_input((node_a, node_b), (bad_edge,))


def test_unsupported_traversal_mode_rejection() -> None:
    plan, schedule, pipeline = _sample_plan_schedule_pipeline()
    lineage = build_research_trace_lineage("traversal_mode_test", plan, schedule, pipeline)
    with pytest.raises(ValueError, match="unsupported traversal mode"):
        traverse_research_trace_lineage(lineage, "unknown_mode")
