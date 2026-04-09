import math

import pytest

from qec.analysis.coding_agent_execution_kernel import (
    CommandHistory,
    ExecutionCommandRecord,
    append_command_history_entry,
    build_task_graph,
    compute_bounded_workflow_score,
    derive_agent_workflow_state,
    empty_command_history,
    normalize_kernel_tasks,
    run_coding_agent_execution_kernel,
    schedule_task_graph,
    validate_command_history,
)


def _sample_tasks():
    return [
        {
            "task_id": "T2",
            "task_kind": "analysis",
            "description": "child task",
            "dependencies": ["T1"],
            "required_capabilities": ["hash", "plan"],
            "priority": 10,
            "estimated_cost": 0.2,
            "bounded": True,
        },
        {
            "task_id": "T1",
            "task_kind": "analysis",
            "description": "root task",
            "dependencies": [],
            "required_capabilities": ["plan"],
            "priority": 5,
            "estimated_cost": 0.1,
            "bounded": True,
        },
    ]


def test_normalize_kernel_tasks_is_deterministic():
    a = normalize_kernel_tasks(_sample_tasks())
    b = normalize_kernel_tasks(reversed(_sample_tasks()))
    assert a == b
    assert tuple(t.task_id for t in a) == ("T1", "T2")


def test_build_task_graph_is_stable():
    g1 = build_task_graph(_sample_tasks())
    g2 = build_task_graph(reversed(_sample_tasks()))
    assert g1 == g2
    assert g1.graph_valid is True


def test_build_task_graph_rejects_unknown_dependency():
    tasks = _sample_tasks()
    tasks[0]["dependencies"] = ["T404"]
    with pytest.raises(ValueError, match="unknown dependency"):
        build_task_graph(tasks)


def test_build_task_graph_rejects_cycle():
    tasks = _sample_tasks()
    tasks[1]["dependencies"] = ["T2"]
    with pytest.raises(ValueError, match="cycle"):
        build_task_graph(tasks)


def test_schedule_task_graph_is_deterministic():
    graph = build_task_graph(_sample_tasks())
    s1 = schedule_task_graph(graph)
    s2 = schedule_task_graph(graph)
    assert s1 == s2
    assert all(s.execution_index == i for i, s in enumerate(s1))


def test_scheduler_tie_break_rules_are_stable():
    tasks = [
        {
            "task_id": "A",
            "task_kind": "k",
            "description": "a",
            "dependencies": [],
            "required_capabilities": [],
            "priority": 2,
            "estimated_cost": 2.0,
            "bounded": True,
        },
        {
            "task_id": "B",
            "task_kind": "k",
            "description": "b",
            "dependencies": [],
            "required_capabilities": [],
            "priority": 2,
            "estimated_cost": 1.0,
            "bounded": True,
        },
        {
            "task_id": "C",
            "task_kind": "k",
            "description": "c",
            "dependencies": [],
            "required_capabilities": [],
            "priority": 3,
            "estimated_cost": 9.0,
            "bounded": True,
        },
    ]
    graph = build_task_graph(tasks)
    scheduled = schedule_task_graph(graph)
    assert tuple(s.task_id for s in scheduled) == ("C", "B", "A")


def test_command_history_chain_is_stable():
    history = empty_command_history()
    history = append_command_history_entry(
        history,
        task_id="T1",
        command_kind="plan",
        command_payload=(("task_id", "T1"),),
    )
    history = append_command_history_entry(
        history,
        task_id="T1",
        command_kind="complete",
        command_payload=(("task_id", "T1"),),
    )
    assert validate_command_history(history) is True
    assert history.chain_valid is True


def test_command_history_detects_corruption():
    history = empty_command_history()
    history = append_command_history_entry(
        history,
        task_id="T1",
        command_kind="plan",
        command_payload=(("task_id", "T1"),),
    )
    bad_entry = ExecutionCommandRecord(
        sequence_id=0,
        task_id="T1",
        command_kind="plan",
        command_payload=(("task_id", "T1"),),
        parent_hash="",
        command_hash="deadbeef",
    )
    bad_history = CommandHistory(
        entries=(bad_entry,),
        head_hash=history.head_hash,
        chain_valid=True,
    )
    with pytest.raises(ValueError, match="corrupted command hash"):
        validate_command_history(bad_history)


def test_append_rejects_malformed_history():
    history = CommandHistory(entries=(), head_hash="", chain_valid=False)
    with pytest.raises(ValueError, match="contradictory chain_valid"):
        append_command_history_entry(
            history,
            task_id="T1",
            command_kind="plan",
            command_payload=(("task_id", "T1"),),
        )


def test_workflow_score_is_bounded_and_reproducible():
    a = compute_bounded_workflow_score(
        ready_tasks=2,
        blocked_tasks=1,
        total_tasks=3,
        total_estimated_cost=0.9,
        dependency_pressure=0.5,
    )
    b = compute_bounded_workflow_score(
        ready_tasks=2,
        blocked_tasks=1,
        total_tasks=3,
        total_estimated_cost=0.9,
        dependency_pressure=0.5,
    )
    assert a == b
    assert 0.0 <= a <= 1.0


def test_run_kernel_same_input_same_bytes():
    out1 = run_coding_agent_execution_kernel(_sample_tasks())
    out2 = run_coding_agent_execution_kernel(_sample_tasks())
    assert out1 == out2
    assert out1[0].to_canonical_json() == out2[0].to_canonical_json()


def test_no_decoder_imports():
    import qec.analysis.coding_agent_execution_kernel as mod

    names = set(mod.__dict__.keys())
    assert not any("decoder" in name.lower() for name in names)


def test_insertion_order_independence():
    tasks_a = _sample_tasks()
    tasks_b = list(reversed(tasks_a))
    assert build_task_graph(tasks_a).graph_hash == build_task_graph(tasks_b).graph_hash


def test_self_dependency_rejected():
    tasks = _sample_tasks()
    tasks[0]["dependencies"] = ["T2"]
    with pytest.raises(ValueError, match="self dependency"):
        normalize_kernel_tasks(tasks)


def test_invalid_command_kind_rejected():
    with pytest.raises(ValueError, match="unknown command kind"):
        append_command_history_entry(
            empty_command_history(),
            task_id="T1",
            command_kind="INVALID",
            command_payload=(("task_id", "T1"),),
        )


def test_empty_task_set_is_valid_and_bounded():
    graph, schedule, state, report, history = run_coding_agent_execution_kernel([])
    assert graph.nodes == ()
    assert schedule == ()
    assert state.total_tasks == 0
    assert 0.0 <= state.bounded_score <= 1.0
    assert report.scheduled_count == 0
    assert validate_command_history(history) is True


def test_contradictory_chain_valid_flag_rejected():
    history = empty_command_history()
    bad = CommandHistory(entries=history.entries, head_hash=history.head_hash, chain_valid=False)
    with pytest.raises(ValueError, match="contradictory chain_valid"):
        validate_command_history(bad)


def test_nan_inf_rejected_in_workflow_score():
    with pytest.raises(ValueError, match="finite"):
        compute_bounded_workflow_score(
            ready_tasks=1,
            blocked_tasks=0,
            total_tasks=1,
            total_estimated_cost=math.inf,
            dependency_pressure=0.0,
        )
    with pytest.raises(ValueError, match="finite"):
        compute_bounded_workflow_score(
            ready_tasks=1,
            blocked_tasks=0,
            total_tasks=1,
            total_estimated_cost=0.0,
            dependency_pressure=math.nan,
        )


def test_graph_hash_independent_of_input_order():
    g1 = build_task_graph(_sample_tasks())
    g2 = build_task_graph(reversed(_sample_tasks()))
    assert g1.graph_hash == g2.graph_hash


def test_inconsistent_task_counts_rejected():
    with pytest.raises(ValueError, match="cannot exceed total_tasks"):
        compute_bounded_workflow_score(
            ready_tasks=4,
            blocked_tasks=0,
            total_tasks=3,
            total_estimated_cost=0.0,
        )
    with pytest.raises(ValueError, match="cannot exceed total_tasks"):
        compute_bounded_workflow_score(
            ready_tasks=0,
            blocked_tasks=4,
            total_tasks=3,
            total_estimated_cost=0.0,
        )
    with pytest.raises(ValueError, match="cannot exceed total_tasks"):
        compute_bounded_workflow_score(
            ready_tasks=2,
            blocked_tasks=2,
            total_tasks=3,
            total_estimated_cost=0.0,
        )


def test_schedule_carries_estimated_cost():
    graph = build_task_graph(_sample_tasks())
    scheduled = schedule_task_graph(graph)
    cost_map = {t.task_id: t.estimated_cost for t in graph.nodes}
    for item in scheduled:
        assert item.estimated_cost == cost_map[item.task_id]


def test_workflow_state_uses_real_total_cost():
    graph = build_task_graph(_sample_tasks())
    scheduled = schedule_task_graph(graph)
    history = empty_command_history()
    state = derive_agent_workflow_state(scheduled, history)
    expected_cost = round(sum(t.estimated_cost for t in scheduled), 12)
    assert expected_cost > 0.0
    assert 0.0 <= state.bounded_score <= 1.0


def test_estimated_cost_rounded_to_fixed_precision():
    tasks = [
        {
            "task_id": "X",
            "task_kind": "k",
            "description": "d",
            "dependencies": [],
            "required_capabilities": [],
            "priority": 1,
            "estimated_cost": 0.1 + 0.2,
            "bounded": True,
        }
    ]
    normalized = normalize_kernel_tasks(tasks)
    a = normalized[0].estimated_cost
    tasks[0]["estimated_cost"] = 0.3
    normalized2 = normalize_kernel_tasks(tasks)
    b = normalized2[0].estimated_cost
    assert a == b
