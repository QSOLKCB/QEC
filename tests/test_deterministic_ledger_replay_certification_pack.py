from __future__ import annotations

import dataclasses
import json

import pytest

from qec.orchestration.autonomous_research_orchestration_kernel import (
    OrchestrationStep,
    ResearchTask,
    build_autonomous_research_plan,
)
from qec.orchestration.dataflow_research_ledger_kernel import (
    build_dataflow_research_ledger,
    traverse_dataflow_research_ledger,
)
from qec.orchestration.deterministic_experiment_scheduling_kernel import (
    SchedulingLane,
    build_deterministic_experiment_schedule,
)
from qec.orchestration.deterministic_ledger_replay_certification_pack import (
    build_replay_certification_receipt,
    certify_ledger_replay,
    certify_traversal_replay,
    compare_replay_runs,
    validate_ledger_replay_certification,
)
from qec.orchestration.deterministic_research_audit_kernel import run_deterministic_research_audit
from qec.orchestration.replay_safe_benchmark_pipeline_kernel import (
    BenchmarkResult,
    BenchmarkStage,
    build_replay_safe_benchmark_pipeline,
)
from qec.orchestration.research_trace_lineage_kernel import build_research_trace_lineage


def _sample_ledger():
    tasks = (
        ResearchTask(task_id="task_plan", task_kind="proof", source_ref="node:0", priority=0, task_epoch=0),
        ResearchTask(task_id="task_schedule", task_kind="validation", source_ref="node:1", priority=1, task_epoch=1),
    )
    steps = (
        OrchestrationStep(step_id="step_plan", task_id="task_plan", execution_order=0, dependency_refs=(), step_epoch=0),
        OrchestrationStep(step_id="step_schedule", task_id="task_schedule", execution_order=1, dependency_refs=("step_plan",), step_epoch=1),
    )
    plan = build_autonomous_research_plan("plan_v137_17_6", tasks, steps)

    lanes = (
        SchedulingLane(lane_id="lane_proof_0", lane_kind="proof", capacity=1, lane_epoch=0),
        SchedulingLane(lane_id="lane_validation_0", lane_kind="validation", capacity=1, lane_epoch=0),
    )
    schedule = build_deterministic_experiment_schedule("schedule_v137_17_6", plan, lanes)

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
    pipeline = build_replay_safe_benchmark_pipeline("pipeline_v137_17_6", schedule, stages, results)
    lineage = build_research_trace_lineage("lineage_v137_17_6", plan, schedule, pipeline)
    audit = run_deterministic_research_audit("audit_v137_17_6", plan, schedule, pipeline, lineage)
    return build_dataflow_research_ledger("ledger_v137_17_6", plan, schedule, pipeline, lineage, audit)


def test_identical_repeated_certification_produces_same_bytes_hash() -> None:
    ledger = _sample_ledger()
    a = certify_ledger_replay(ledger)
    b = certify_ledger_replay(ledger)
    assert a.to_canonical_json() == b.to_canonical_json()
    assert a.stable_hash() == b.stable_hash()


def test_rebuilt_ledger_certifies_successfully() -> None:
    ledger = _sample_ledger()
    pack = certify_ledger_replay(ledger)
    assert pack.report.replay_stable is True
    assert validate_ledger_replay_certification(pack) is True


def test_tampered_ledger_fails_certification() -> None:
    ledger = _sample_ledger()
    tampered = ledger.to_dict()
    tampered["ledger_hash"] = "f" * 64
    with pytest.raises(ValueError, match="malformed ledger input"):
        certify_ledger_replay(tampered)


def test_tampered_traversal_receipt_fails() -> None:
    ledger = _sample_ledger()
    receipt = traverse_dataflow_research_ledger(ledger, "full")
    bad_receipt = dataclasses.replace(receipt, traversal_hash="f" * 64)
    with pytest.raises(ValueError, match="replay certification failed"):
        certify_ledger_replay(ledger, expected_receipts={"full": bad_receipt})


def test_traversal_order_drift_fails() -> None:
    ledger = _sample_ledger()
    receipt = traverse_dataflow_research_ledger(ledger, "full")
    bad_receipt = dataclasses.replace(receipt, ordered_stage_trace=tuple(reversed(receipt.ordered_stage_trace)))
    with pytest.raises(ValueError, match="replay certification failed"):
        certify_ledger_replay(ledger, expected_receipts={"full": bad_receipt})


def test_continuity_summary_drift_fails() -> None:
    ledger = _sample_ledger()
    tampered = ledger.to_dict()
    tampered["continuity_summary"] = {
        "total_stages": 5,
        "linked_stages": 4,
        "broken_links": 1,
        "terminal_stage": "audit",
        "continuity_ok": False,
    }
    report = compare_replay_runs(ledger, tampered)
    assert report.replay_stable is False
    assert any(entry.continuity_summary_match is False for entry in report.entries)


def test_subset_traversal_replay_remains_stable() -> None:
    ledger = _sample_ledger()
    pack = certify_ledger_replay(ledger, traversal_modes=("continuity", "critical", "receipt"))
    assert pack.report.replay_stable is True
    assert pack.report.traversal_modes == ("continuity", "critical", "receipt")


def test_malformed_ledger_rejected() -> None:
    with pytest.raises(ValueError, match="malformed ledger input"):
        certify_ledger_replay({"ledger_id": "x", "entries": []})


def test_deterministic_repeated_receipt_generation_stable() -> None:
    ledger = _sample_ledger()
    pack = certify_ledger_replay(ledger)
    a = build_replay_certification_receipt(pack.report)
    b = build_replay_certification_receipt(pack.report)
    assert a.to_canonical_json() == b.to_canonical_json()
    assert a.stable_hash() == b.stable_hash()


def test_certification_hash_reproducible() -> None:
    ledger = _sample_ledger()
    a = certify_ledger_replay(ledger)
    b = certify_ledger_replay(ledger)
    assert a.report.certification_hash == b.report.certification_hash
    assert a.report.stable_hash() == b.report.stable_hash()


def test_canonical_json_byte_equality() -> None:
    ledger = _sample_ledger()
    pack = certify_ledger_replay(ledger)
    canonical_a = pack.to_canonical_json().encode("utf-8")
    canonical_b = json.dumps(pack.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
    assert canonical_a == canonical_b


def test_certify_traversal_replay_deterministic() -> None:
    ledger = _sample_ledger()
    rebuilt = json.loads(ledger.to_canonical_json())
    entry = certify_traversal_replay(ledger, rebuilt, "full")
    assert entry.replay_stable is True
    report = compare_replay_runs(ledger, rebuilt)
    assert report.replay_stable is True


def test_validate_certification_rejects_tampered_report_hash() -> None:
    ledger = _sample_ledger()
    pack = certify_ledger_replay(ledger)
    tampered_report = dataclasses.replace(pack.report, certification_hash="f" * 64)
    tampered_pack = dataclasses.replace(pack, report=tampered_report)
    with pytest.raises(ValueError, match="report certification_hash mismatch"):
        validate_ledger_replay_certification(tampered_pack)


def test_validate_certification_rejects_tampered_receipt_report_hash() -> None:
    ledger = _sample_ledger()
    pack = certify_ledger_replay(ledger)
    tampered_receipt = dataclasses.replace(pack.receipt, report_hash="f" * 64)
    tampered_pack = dataclasses.replace(pack, receipt=tampered_receipt)
    with pytest.raises(ValueError, match="receipt report_hash mismatch"):
        validate_ledger_replay_certification(tampered_pack)


def test_validate_certification_rejects_tampered_receipt_ledger_id() -> None:
    ledger = _sample_ledger()
    pack = certify_ledger_replay(ledger)
    tampered_receipt = dataclasses.replace(pack.receipt, ledger_id="tampered-id")
    tampered_pack = dataclasses.replace(pack, receipt=tampered_receipt)
    with pytest.raises(ValueError, match="receipt ledger_id mismatch"):
        validate_ledger_replay_certification(tampered_pack)


def test_certify_ledger_replay_raises_on_unexpected_receipt_mode() -> None:
    ledger = _sample_ledger()
    receipt = traverse_dataflow_research_ledger(ledger, "full")
    with pytest.raises(ValueError, match="expected_receipts contains modes not present in traversal_modes"):
        certify_ledger_replay(ledger, traversal_modes=("continuity",), expected_receipts={"full": receipt})
