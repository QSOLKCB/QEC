from __future__ import annotations

import dataclasses

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
from qec.orchestration.research_trace_lineage_kernel import build_research_trace_lineage
from qec.orchestration.deterministic_research_audit_kernel import run_deterministic_research_audit
from qec.orchestration.dataflow_research_ledger_kernel import (
    build_dataflow_research_ledger,
    compute_dataflow_continuity,
    traverse_dataflow_research_ledger,
    validate_dataflow_research_ledger,
)


def _sample_artifacts():
    tasks = (
        ResearchTask(task_id="task_plan", task_kind="proof", source_ref="node:0", priority=0, task_epoch=0),
        ResearchTask(task_id="task_schedule", task_kind="validation", source_ref="node:1", priority=1, task_epoch=1),
    )
    steps = (
        OrchestrationStep(step_id="step_plan", task_id="task_plan", execution_order=0, dependency_refs=(), step_epoch=0),
        OrchestrationStep(step_id="step_schedule", task_id="task_schedule", execution_order=1, dependency_refs=("step_plan",), step_epoch=1),
    )
    plan = build_autonomous_research_plan("plan_v137_17_5", tasks, steps)

    lanes = (
        SchedulingLane(lane_id="lane_proof_0", lane_kind="proof", capacity=1, lane_epoch=0),
        SchedulingLane(lane_id="lane_validation_0", lane_kind="validation", capacity=1, lane_epoch=0),
    )
    schedule = build_deterministic_experiment_schedule("schedule_v137_17_5", plan, lanes)

    stages = (
        BenchmarkStage(stage_id="stage_prepare", stage_kind="prepare", input_ref="step_plan", stage_order=0, stage_epoch=0),
        BenchmarkStage(stage_id="stage_verify", stage_kind="verify", input_ref="step_schedule", stage_order=0, stage_epoch=1),
    )
    results = (
        BenchmarkResult(
            result_id="result_metric",
            stage_id="stage_prepare",
            experiment_id="step_plan",
            result_kind="metric",
            result_hash="0" * 64,
            result_epoch=0,
        ),
        BenchmarkResult(
            result_id="result_verification",
            stage_id="stage_verify",
            experiment_id="step_schedule",
            result_kind="verification",
            result_hash="1" * 64,
            result_epoch=1,
        ),
    )
    pipeline = build_replay_safe_benchmark_pipeline("pipeline_v137_17_5", schedule, stages, results)
    lineage = build_research_trace_lineage("lineage_v137_17_5", plan, schedule, pipeline)
    audit = run_deterministic_research_audit("audit_v137_17_5", plan, schedule, pipeline, lineage)
    return plan, schedule, pipeline, lineage, audit


def test_repeated_build_deterministic_json_hash() -> None:
    plan, schedule, pipeline, lineage, audit = _sample_artifacts()
    a = build_dataflow_research_ledger("ledger_repeat", plan, schedule, pipeline, lineage, audit)
    b = build_dataflow_research_ledger("ledger_repeat", plan, schedule, pipeline, lineage, audit)
    assert a.to_canonical_json() == b.to_canonical_json()
    assert a.stable_hash() == b.stable_hash()


def test_traversal_order_stable_across_runs() -> None:
    plan, schedule, pipeline, lineage, audit = _sample_artifacts()
    ledger = build_dataflow_research_ledger("ledger_traversal", plan, schedule, pipeline, lineage, audit)
    a = traverse_dataflow_research_ledger(ledger, "full")
    b = traverse_dataflow_research_ledger(ledger, "full")
    assert a.ordered_stage_trace == b.ordered_stage_trace
    assert a.ordered_edge_trace == b.ordered_edge_trace
    assert a.traversal_hash == b.traversal_hash


def test_valid_full_chain_builds_cleanly() -> None:
    plan, schedule, pipeline, lineage, audit = _sample_artifacts()
    ledger = build_dataflow_research_ledger("ledger_chain", plan, schedule, pipeline, lineage, audit)
    assert tuple(entry.stage_name for entry in ledger.entries) == ("plan", "schedule", "pipeline", "lineage", "audit")
    assert validate_dataflow_research_ledger(ledger).is_valid is True


def test_missing_upstream_linkage_detected() -> None:
    plan, schedule, pipeline, lineage, audit = _sample_artifacts()
    ledger = build_dataflow_research_ledger("ledger_missing_upstream", plan, schedule, pipeline, lineage, audit)
    broken_entry = dataclasses.replace(ledger.entries[1], upstream_hash_link="", continuity_ok=True)
    tampered = dataclasses.replace(ledger, entries=(ledger.entries[0], broken_entry, *ledger.entries[2:]))
    with pytest.raises(ValueError, match="missing_required_upstream_link"):
        validate_dataflow_research_ledger(tampered)


def test_hash_drift_between_linked_stages_detected() -> None:
    plan, schedule, pipeline, lineage, audit = _sample_artifacts()
    ledger = build_dataflow_research_ledger("ledger_hash_drift", plan, schedule, pipeline, lineage, audit)
    broken_entry = dataclasses.replace(ledger.entries[2], upstream_hash_link="f" * 64)
    tampered = dataclasses.replace(ledger, entries=(ledger.entries[0], ledger.entries[1], broken_entry, *ledger.entries[3:]))
    with pytest.raises(ValueError, match="hash_drift_between_linked_stages"):
        validate_dataflow_research_ledger(tampered)


def test_duplicate_or_out_of_order_ordinals_rejected() -> None:
    plan, schedule, pipeline, lineage, audit = _sample_artifacts()
    ledger = build_dataflow_research_ledger("ledger_bad_ordinals", plan, schedule, pipeline, lineage, audit)
    duplicate = dataclasses.replace(ledger.entries[1], stage_ordinal=0)
    duplicate_ledger = dataclasses.replace(ledger, entries=(ledger.entries[0], duplicate, *ledger.entries[2:]))
    with pytest.raises(ValueError, match="duplicate_stage_ordinals"):
        validate_dataflow_research_ledger(duplicate_ledger)

    regressed = dataclasses.replace(ledger.entries[3], stage_ordinal=10)
    regressed_ledger = dataclasses.replace(ledger, entries=(ledger.entries[0], ledger.entries[1], ledger.entries[2], regressed, ledger.entries[4]))
    with pytest.raises(ValueError, match="impossible_stage_regression"):
        validate_dataflow_research_ledger(regressed_ledger)


def test_continuity_summary_truthful_for_healthy_chain() -> None:
    plan, schedule, pipeline, lineage, audit = _sample_artifacts()
    ledger = build_dataflow_research_ledger("ledger_summary", plan, schedule, pipeline, lineage, audit)
    summary = compute_dataflow_continuity(ledger.entries)
    assert summary.total_stages == 5
    assert summary.linked_stages == 4
    assert summary.broken_links == 0
    assert summary.terminal_stage == "audit"
    assert summary.continuity_ok is True


def test_traversal_modes_deterministic_subsets() -> None:
    plan, schedule, pipeline, lineage, audit = _sample_artifacts()
    ledger = build_dataflow_research_ledger("ledger_modes", plan, schedule, pipeline, lineage, audit)

    full_a = traverse_dataflow_research_ledger(ledger, "full")
    full_b = traverse_dataflow_research_ledger(ledger, "full")
    assert full_a.ordered_stage_trace == full_b.ordered_stage_trace

    continuity = traverse_dataflow_research_ledger(ledger, "continuity")
    critical = traverse_dataflow_research_ledger(ledger, "critical")
    receipt = traverse_dataflow_research_ledger(ledger, "receipt")

    assert len(continuity.ordered_stage_trace) == 4
    assert len(critical.ordered_stage_trace) == 1
    assert len(receipt.ordered_stage_trace) == 5


def test_malformed_entries_fail_validation() -> None:
    plan, schedule, pipeline, lineage, audit = _sample_artifacts()
    ledger = build_dataflow_research_ledger("ledger_malformed", plan, schedule, pipeline, lineage, audit)
    malformed = dataclasses.replace(ledger.entries[2], predecessor_stage="plan")
    tampered = dataclasses.replace(ledger, entries=(ledger.entries[0], ledger.entries[1], malformed, ledger.entries[3], ledger.entries[4]))
    with pytest.raises(ValueError, match="malformed_receipt_chain_structure"):
        validate_dataflow_research_ledger(tampered)


def test_empty_and_minimal_input_behavior_explicit() -> None:
    with pytest.raises(ValueError, match="plan_hash"):
        build_dataflow_research_ledger(
            "ledger_empty",
            {"plan_id": "p"},
            {"schedule_id": "s", "schedule_hash": "0" * 64},
            {"pipeline_id": "pi", "pipeline_hash": "1" * 64},
            {"lineage_id": "l", "lineage_hash": "2" * 64},
            {"audit_id": "a", "audit_hash": "3" * 64},
        )

    with pytest.raises(ValueError, match="empty_ledger_entries"):
        validate_dataflow_research_ledger({"ledger_id": "x", "entries": ()})
