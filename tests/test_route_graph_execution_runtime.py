from __future__ import annotations

import pytest

from qec.analysis.route_graph_execution_runtime import (
    GENESIS_HASH,
    ROUTE_GRAPH_EXECUTION_RUNTIME_VERSION,
    advance_path_state,
    execute_route_graph,
    export_execution_bytes,
    generate_execution_receipt,
)


def _world_state() -> dict[str, object]:
    return {
        "agent": "runtime-v137-6",
        "energy": 2.0,
        "flags": ["deterministic", "replay_safe"],
        "state": {"goal": 1.0, "start": 0.0},
    }


def _source_plan_hash() -> str:
    return "a" * 64


def _route_graph() -> dict[str, tuple[str, ...]]:
    return {
        "start": ("branch_b", "branch_a"),
        "branch_a": ("goal",),
        "branch_b": ("goal",),
        "goal": tuple(),
    }


def test_repeated_run_determinism_identical_bytes_and_hashes() -> None:
    execution_a = execute_route_graph(
        _source_plan_hash(),
        _route_graph(),
        initial_node="start",
        world_state=_world_state(),
        max_path_length=5,
        enable_v137_6_route_runtime=True,
    )
    execution_b = execute_route_graph(
        _source_plan_hash(),
        _route_graph(),
        initial_node="start",
        world_state=_world_state(),
        max_path_length=5,
        enable_v137_6_route_runtime=True,
    )

    assert execution_a == execution_b
    assert export_execution_bytes(execution_a) == export_execution_bytes(execution_b)
    assert execution_a.stable_execution_hash == execution_b.stable_execution_hash


def test_identical_inputs_produce_identical_receipt_bytes() -> None:
    execution = execute_route_graph(
        _source_plan_hash(),
        _route_graph(),
        initial_node="start",
        world_state=_world_state(),
        max_path_length=5,
        enable_v137_6_route_runtime=True,
    )
    receipt_a = generate_execution_receipt(execution)
    receipt_b = generate_execution_receipt(execution)

    assert receipt_a == receipt_b
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_bounded_path_length_is_enforced() -> None:
    execution = execute_route_graph(
        _source_plan_hash(),
        _route_graph(),
        initial_node="start",
        world_state=_world_state(),
        max_path_length=2,
        enable_v137_6_route_runtime=True,
    )

    assert execution.path_length == 2
    assert execution.executed_route == ("start", "branch_a")


def test_stable_execution_identity_chain_and_receipt_law_fields() -> None:
    execution = execute_route_graph(
        _source_plan_hash(),
        _route_graph(),
        initial_node="start",
        world_state=_world_state(),
        max_path_length=5,
        enable_v137_6_route_runtime=True,
    )
    receipt = generate_execution_receipt(execution)

    assert execution.execution_identity_chain[0] == GENESIS_HASH
    assert execution.schema_version == ROUTE_GRAPH_EXECUTION_RUNTIME_VERSION

    assert receipt.source_plan_hash == execution.source_plan_hash
    assert receipt.executed_route_hash == execution.executed_route_hash
    assert receipt.path_length == execution.path_length
    assert receipt.execution_stability_score == execution.execution_stability_score
    assert receipt.final_world_state_hash == execution.final_world_state_hash
    assert receipt.stable_execution_hash == execution.stable_execution_hash


def test_deterministic_branch_tie_breaking_prefers_lexicographic_target() -> None:
    graph = {
        "start": ("zeta", "alpha", "middle"),
        "alpha": ("goal",),
        "middle": ("goal",),
        "zeta": ("goal",),
        "goal": tuple(),
    }
    execution = execute_route_graph(
        _source_plan_hash(),
        graph,
        initial_node="start",
        world_state=_world_state(),
        max_path_length=3,
        enable_v137_6_route_runtime=True,
    )

    assert execution.executed_route[1] == "alpha"


def test_fail_fast_invalid_input_handling() -> None:
    with pytest.raises(ValueError, match="enable_v137_6_route_runtime"):
        execute_route_graph(
            _source_plan_hash(),
            _route_graph(),
            initial_node="start",
            world_state=_world_state(),
            max_path_length=5,
        )

    with pytest.raises(ValueError, match="source_plan_hash must be a lowercase SHA-256"):
        execute_route_graph(
            "ABC",
            _route_graph(),
            initial_node="start",
            world_state=_world_state(),
            max_path_length=5,
            enable_v137_6_route_runtime=True,
        )

    with pytest.raises(ValueError, match="world_state must normalize to a non-empty mapping"):
        execute_route_graph(
            _source_plan_hash(),
            _route_graph(),
            initial_node="start",
            world_state={},
            max_path_length=5,
            enable_v137_6_route_runtime=True,
        )

    with pytest.raises(ValueError, match="max_path_length must be >= 1"):
        execute_route_graph(
            _source_plan_hash(),
            _route_graph(),
            initial_node="start",
            world_state=_world_state(),
            max_path_length=0,
            enable_v137_6_route_runtime=True,
        )

    with pytest.raises(ValueError, match="initial_node must exist in route_graph"):
        execute_route_graph(
            _source_plan_hash(),
            _route_graph(),
            initial_node="unknown",
            world_state=_world_state(),
            max_path_length=5,
            enable_v137_6_route_runtime=True,
        )

    with pytest.raises(ValueError, match="duplicate branch targets"):
        execute_route_graph(
            _source_plan_hash(),
            {"start": ("a", "a")},
            initial_node="start",
            world_state=_world_state(),
            max_path_length=5,
            enable_v137_6_route_runtime=True,
        )


def test_advance_path_state_deterministic_progression() -> None:
    next_path = advance_path_state(("start",), _route_graph(), max_path_length=4)
    assert next_path == ("start", "branch_a")

    bounded_path = advance_path_state(("start", "branch_a"), _route_graph(), max_path_length=2)
    assert bounded_path == ("start", "branch_a")
