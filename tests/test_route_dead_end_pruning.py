from __future__ import annotations

import pytest

from qec.analysis.route_dead_end_pruning import (
    GENESIS_HASH,
    ROUTE_DEAD_END_PRUNING_VERSION,
    analyze_dead_end_pruning,
    compute_dead_end_pressure,
    export_dead_end_pruning_bytes,
    generate_dead_end_pruning_receipt,
    prune_route_frontier,
)
from qec.analysis.route_graph_execution_runtime import execute_route_graph


def _source_plan_hash() -> str:
    return "b" * 64


def _world_state() -> dict[str, object]:
    return {
        "agent": "runtime-v137-6-2",
        "energy": 3.0,
        "state": {"goal": 1.0, "start": 0.0},
        "flags": ["deterministic", "replay_safe"],
    }


def _graph_with_dead_ends() -> dict[str, tuple[str, ...]]:
    return {
        "start": ("branch_z", "branch_a", "branch_b"),
        "branch_a": ("goal",),
        "branch_b": ("cycle",),
        "branch_z": ("cycle",),
        "cycle": ("cycle",),
        "goal": tuple(),
    }


def _fully_reachable_graph() -> dict[str, tuple[str, ...]]:
    return {
        "start": ("alpha", "beta"),
        "alpha": ("goal",),
        "beta": ("goal",),
        "goal": tuple(),
    }


def test_deterministic_detection_same_input_same_bytes_and_hashes() -> None:
    a = analyze_dead_end_pruning(
        _source_plan_hash(),
        _graph_with_dead_ends(),
        current_path=("start",),
        max_path_length=4,
        enable_v137_6_2_dead_end_pruning=True,
    )
    b = analyze_dead_end_pruning(
        _source_plan_hash(),
        _graph_with_dead_ends(),
        current_path=("start",),
        max_path_length=4,
        enable_v137_6_2_dead_end_pruning=True,
    )

    assert a == b
    assert export_dead_end_pruning_bytes(a) == export_dead_end_pruning_bytes(b)
    assert a.stable_pruning_hash == b.stable_pruning_hash
    assert a.schema_version == ROUTE_DEAD_END_PRUNING_VERSION
    assert a.pruning_identity_chain[0] == GENESIS_HASH


def test_stable_tie_breaking_dead_end_order_is_lexicographic() -> None:
    artifact = analyze_dead_end_pruning(
        _source_plan_hash(),
        _graph_with_dead_ends(),
        current_path=("start",),
        max_path_length=3,
        enable_v137_6_2_dead_end_pruning=True,
    )

    assert artifact.examined_candidates == ("branch_a", "branch_b", "branch_z")
    assert artifact.dead_end_candidates == ("branch_b", "branch_z")
    assert artifact.survivor_candidates == ("branch_a",)


def test_bounded_pressure_invariant_holds() -> None:
    assert compute_dead_end_pressure(0, 0) == 0.0
    assert 0.0 <= compute_dead_end_pressure(5, 0) <= 1.0
    assert 0.0 <= compute_dead_end_pressure(5, 5) <= 1.0
    assert 0.0 <= compute_dead_end_pressure(5, 2) <= 1.0


def test_truthful_no_prune_path_when_all_candidates_reach_terminal() -> None:
    survivors, artifact = prune_route_frontier(
        _source_plan_hash(),
        _fully_reachable_graph(),
        current_path=("start",),
        max_path_length=3,
        enable_v137_6_2_dead_end_pruning=True,
    )

    assert artifact.dead_end_candidates == tuple()
    assert survivors == artifact.examined_candidates
    assert artifact.dead_end_pressure_score == 0.0


def test_truthful_prune_path_removes_dead_end_branches_preserving_survivors() -> None:
    survivors, artifact = prune_route_frontier(
        _source_plan_hash(),
        _graph_with_dead_ends(),
        current_path=("start",),
        max_path_length=3,
        enable_v137_6_2_dead_end_pruning=True,
    )

    assert survivors == ("branch_a",)
    assert artifact.dead_end_candidates == ("branch_b", "branch_z")
    assert artifact.dead_end_pressure_score == pytest.approx(2.0 / 3.0)


def test_replay_safe_canonical_export_and_receipt_stability() -> None:
    artifact = analyze_dead_end_pruning(
        _source_plan_hash(),
        _graph_with_dead_ends(),
        current_path=("start",),
        max_path_length=3,
        enable_v137_6_2_dead_end_pruning=True,
    )

    receipt_a = generate_dead_end_pruning_receipt(artifact)
    receipt_b = generate_dead_end_pruning_receipt(artifact)

    assert artifact.to_canonical_json().encode("utf-8") == artifact.to_canonical_bytes()
    assert receipt_a == receipt_b
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_fail_fast_validation_for_malformed_inputs() -> None:
    with pytest.raises(ValueError, match="enable_v137_6_2_dead_end_pruning"):
        analyze_dead_end_pruning(
            _source_plan_hash(),
            _graph_with_dead_ends(),
            current_path=("start",),
            max_path_length=3,
        )

    with pytest.raises(ValueError, match="source_plan_hash must be a lowercase SHA-256"):
        analyze_dead_end_pruning(
            "BAD",
            _graph_with_dead_ends(),
            current_path=("start",),
            max_path_length=3,
            enable_v137_6_2_dead_end_pruning=True,
        )

    with pytest.raises(ValueError, match="dead_end_count must be <= total_candidates_examined"):
        compute_dead_end_pressure(1, 2)

    with pytest.raises(ValueError, match="current_path terminal node must exist in route_graph"):
        analyze_dead_end_pruning(
            _source_plan_hash(),
            _graph_with_dead_ends(),
            current_path=("unknown",),
            max_path_length=3,
            enable_v137_6_2_dead_end_pruning=True,
        )


def test_integration_sanity_route_runtime_uses_pruned_frontier_without_breaking_execution() -> None:
    survivors, artifact = prune_route_frontier(
        _source_plan_hash(),
        _graph_with_dead_ends(),
        current_path=("start",),
        max_path_length=3,
        enable_v137_6_2_dead_end_pruning=True,
    )

    pruned_graph = dict(_graph_with_dead_ends())
    pruned_graph["start"] = survivors

    execution = execute_route_graph(
        _source_plan_hash(),
        pruned_graph,
        initial_node="start",
        world_state=_world_state(),
        max_path_length=3,
        enable_v137_6_route_runtime=True,
    )

    assert artifact.survivor_candidates == survivors
    assert execution.executed_route == ("start", "branch_a", "goal")
