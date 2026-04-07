from __future__ import annotations

import pytest

from qec.analysis.heterogeneous_scheduler import (
    _sha256_hex,
    build_epoch_schedule,
    build_schedule_receipt,
    compile_scheduler_report,
    normalize_epoch_tasks,
)


def _base_tasks() -> list[dict[str, object]]:
    return [
        {
            "task_id": "task-b",
            "task_type": "identity_pass",
            "lane_id": "cpu_lane",
            "epoch_id": "epoch-001",
            "task_order": 2,
            "dependency_ids": [],
            "payload_hash": "hash-b",
            "schema_version": 1,
        },
        {
            "task_id": "task-a",
            "task_type": "matrix_offload",
            "lane_id": "integer_lane_0",
            "epoch_id": "epoch-001",
            "task_order": 1,
            "dependency_ids": [],
            "payload_hash": "hash-a",
            "schema_version": 1,
        },
    ]


def test_stable_dispatch_ordering() -> None:
    report = compile_scheduler_report({"tasks": _base_tasks(), "schema_version": 1})
    assert report.schedule.dispatch_order == ("task-a", "task-b")


def test_dependency_enforcement() -> None:
    tasks = _base_tasks()
    tasks.append(
        {
            "task_id": "task-c",
            "task_type": "lookup_transform",
            "lane_id": "integer_lane_1",
            "epoch_id": "epoch-001",
            "task_order": 1,
            "dependency_ids": ["task-b"],
            "payload_hash": "hash-c",
            "schema_version": 1,
        }
    )
    report = compile_scheduler_report({"tasks": tasks, "schema_version": 1})
    assert report.schedule.dispatch_order.index("task-b") < report.schedule.dispatch_order.index("task-c")


def test_barrier_behavior() -> None:
    tasks = [
        {
            "task_id": "t1",
            "task_type": "identity_pass",
            "lane_id": "cpu_lane",
            "epoch_id": "epoch-001",
            "task_order": 1,
            "dependency_ids": [],
            "payload_hash": "h1",
            "schema_version": 1,
        },
        {
            "task_id": "bar",
            "task_type": "merge_barrier",
            "lane_id": "fixed_function_0",
            "epoch_id": "epoch-001",
            "task_order": 2,
            "dependency_ids": [],
            "payload_hash": "hb",
            "schema_version": 1,
        },
        {
            "task_id": "t2",
            "task_type": "bitfield_transform",
            "lane_id": "integer_lane_1",
            "epoch_id": "epoch-001",
            "task_order": 3,
            "dependency_ids": [],
            "payload_hash": "h2",
            "schema_version": 1,
        },
    ]
    schedule = build_epoch_schedule(normalize_epoch_tasks({"tasks": tasks, "schema_version": 1}))
    assert schedule.dispatch_order == ("t1", "bar", "t2")
    assert schedule.barrier_count == 1


def test_duplicate_task_rejection() -> None:
    tasks = _base_tasks()
    tasks.append({**tasks[0]})
    with pytest.raises(ValueError, match="duplicate task_id within epoch"):
        compile_scheduler_report({"tasks": tasks, "schema_version": 1})


def test_cycle_dependency_rejection() -> None:
    tasks = [
        {
            "task_id": "a",
            "task_type": "identity_pass",
            "lane_id": "cpu_lane",
            "epoch_id": "epoch-001",
            "task_order": 1,
            "dependency_ids": ["b"],
            "payload_hash": "ha",
            "schema_version": 1,
        },
        {
            "task_id": "b",
            "task_type": "identity_pass",
            "lane_id": "cpu_lane",
            "epoch_id": "epoch-001",
            "task_order": 1,
            "dependency_ids": ["a"],
            "payload_hash": "hb",
            "schema_version": 1,
        },
    ]
    with pytest.raises(ValueError, match="cycle"):
        compile_scheduler_report({"tasks": tasks, "schema_version": 1})


def test_cross_epoch_leakage_rejection() -> None:
    tasks = _base_tasks()
    tasks.append(
        {
            "task_id": "task-z",
            "task_type": "identity_pass",
            "lane_id": "cpu_lane",
            "epoch_id": "epoch-002",
            "task_order": 3,
            "dependency_ids": ["task-a"],
            "payload_hash": "hash-z",
            "schema_version": 1,
        }
    )
    with pytest.raises(ValueError, match="same epoch_id"):
        compile_scheduler_report({"tasks": tasks, "schema_version": 1})


def test_stable_hashes_and_receipts() -> None:
    report_a = compile_scheduler_report({"tasks": _base_tasks(), "schema_version": 1})
    report_b = compile_scheduler_report({"tasks": _base_tasks(), "schema_version": 1})
    assert report_a.schedule.stable_schedule_hash == report_b.schedule.stable_schedule_hash
    assert report_a.receipt.receipt_hash == report_b.receipt.receipt_hash


def test_repeated_runs_byte_identical() -> None:
    artifacts = tuple(compile_scheduler_report({"tasks": _base_tasks(), "schema_version": 1}).to_canonical_bytes() for _ in range(10))
    assert len(set(artifacts)) == 1


def test_schema_rejection() -> None:
    with pytest.raises(ValueError, match="unsupported schema version"):
        compile_scheduler_report({"tasks": _base_tasks(), "schema_version": 2})


def test_lane_validation() -> None:
    tasks = _base_tasks()
    tasks[0]["lane_id"] = "gpu_lane"
    with pytest.raises(ValueError, match="unsupported lane_id"):
        compile_scheduler_report({"tasks": tasks, "schema_version": 1})


def test_lexicographic_tie_break_stability() -> None:
    tasks = [
        {
            "task_id": "task-z",
            "task_type": "identity_pass",
            "lane_id": "cpu_lane",
            "epoch_id": "epoch-001",
            "task_order": 1,
            "dependency_ids": [],
            "payload_hash": "hz",
            "schema_version": 1,
        },
        {
            "task_id": "task-a",
            "task_type": "identity_pass",
            "lane_id": "cpu_lane",
            "epoch_id": "epoch-001",
            "task_order": 1,
            "dependency_ids": [],
            "payload_hash": "ha",
            "schema_version": 1,
        },
    ]
    report = compile_scheduler_report({"tasks": tasks, "schema_version": 1})
    assert report.schedule.dispatch_order == ("task-a", "task-z")


def test_missing_dependency_rejected() -> None:
    tasks = _base_tasks()
    tasks[1]["dependency_ids"] = ["not-there"]
    with pytest.raises(ValueError, match="missing dependency IDs"):
        compile_scheduler_report({"tasks": tasks, "schema_version": 1})


def test_receipt_stability() -> None:
    schedule = build_epoch_schedule(normalize_epoch_tasks({"tasks": _base_tasks(), "schema_version": 1}))
    receipt_a = build_schedule_receipt(schedule)
    receipt_b = build_schedule_receipt(schedule)
    assert receipt_a == receipt_b


def test_schedule_hash_verifiable_from_payload() -> None:
    schedule = build_epoch_schedule(normalize_epoch_tasks({"tasks": _base_tasks(), "schema_version": 1}))
    recomputed = _sha256_hex(schedule.to_hash_payload_dict())
    assert recomputed == schedule.stable_schedule_hash
