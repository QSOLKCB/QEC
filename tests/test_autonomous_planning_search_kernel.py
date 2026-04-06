from __future__ import annotations

import pytest

from qec.analysis.autonomous_planning_search_kernel import (
    AUTONOMOUS_PLANNING_SEARCH_KERNEL_VERSION,
    GENESIS_HASH,
    export_plan_bytes,
    generate_plan_receipt,
    synthesize_plan_ir,
)


def _world_state() -> dict[str, object]:
    return {
        "agent": "planner-v137-6",
        "budget": 3.0,
        "state": {"mid": 1.0, "start": 0.0, "goal": 2.0},
        "flags": ["deterministic", "replay_safe"],
    }


def _objective() -> dict[str, object]:
    return {
        "preferred_nodes": ["goal", "mid"],
        "policy": "min_depth_max_coverage",
    }


def _routes() -> tuple[tuple[str, ...], ...]:
    return (
        ("start", "goal"),
        ("start", "mid", "goal"),
        ("start", "alt", "goal"),
        ("start", "very", "deep", "goal"),
    )


def test_repeated_run_determinism_identical_bytes_and_hashes() -> None:
    plan_a = synthesize_plan_ir(
        _world_state(),
        _objective(),
        _routes(),
        search_depth=2,
        enable_v137_6_search=True,
    )
    plan_b = synthesize_plan_ir(
        _world_state(),
        _objective(),
        _routes(),
        search_depth=2,
        enable_v137_6_search=True,
    )

    assert plan_a == plan_b
    assert export_plan_bytes(plan_a) == export_plan_bytes(plan_b)
    assert plan_a.stable_plan_hash == plan_b.stable_plan_hash


def test_identical_inputs_produce_identical_receipt_bytes() -> None:
    plan = synthesize_plan_ir(
        _world_state(),
        _objective(),
        _routes(),
        search_depth=2,
        enable_v137_6_search=True,
    )
    receipt_a = generate_plan_receipt(plan)
    receipt_b = generate_plan_receipt(plan)

    assert receipt_a == receipt_b
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_bounded_search_depth_excludes_deeper_routes() -> None:
    plan = synthesize_plan_ir(
        _world_state(),
        _objective(),
        _routes(),
        search_depth=2,
        enable_v137_6_search=True,
    )

    assert plan.search_depth == 2
    assert plan.selected_route != ("start", "very", "deep", "goal")


def test_stable_search_identity_chain_and_receipt_law_fields() -> None:
    plan = synthesize_plan_ir(
        _world_state(),
        _objective(),
        _routes(),
        search_depth=2,
        enable_v137_6_search=True,
    )
    receipt = generate_plan_receipt(plan)

    assert plan.search_identity_chain[0] == GENESIS_HASH
    assert len(plan.search_identity_chain) == 1 + 3
    assert plan.schema_version == AUTONOMOUS_PLANNING_SEARCH_KERNEL_VERSION

    assert receipt.world_state_hash == plan.world_state_hash
    assert receipt.objective_hash == plan.objective_hash
    assert receipt.search_depth == plan.search_depth
    assert receipt.search_stability_score == plan.search_stability_score
    assert receipt.selected_route_hash == plan.selected_route_hash
    assert receipt.stable_plan_hash == plan.stable_plan_hash


def test_route_tie_determinism_prefers_shortest_then_lexicographic() -> None:
    objective = {
        "preferred_nodes": ["goal"],
        "policy": "tie_break",
    }
    routes = (
        ("start", "goal"),
        ("start", "aux", "goal"),
    )
    plan = synthesize_plan_ir(
        _world_state(),
        objective,
        routes,
        search_depth=3,
        enable_v137_6_search=True,
    )

    assert plan.selected_route == ("start", "goal")


def test_fail_fast_invalid_input_handling() -> None:
    with pytest.raises(ValueError, match="enable_v137_6_search"):
        synthesize_plan_ir(_world_state(), _objective(), _routes(), search_depth=2)

    with pytest.raises(ValueError, match="world_state must normalize to a non-empty mapping"):
        synthesize_plan_ir({}, _objective(), _routes(), search_depth=2, enable_v137_6_search=True)

    with pytest.raises(ValueError, match="objective must normalize to a non-empty mapping"):
        synthesize_plan_ir(_world_state(), {}, _routes(), search_depth=2, enable_v137_6_search=True)

    with pytest.raises(ValueError, match="search_depth must be >= 1"):
        synthesize_plan_ir(_world_state(), _objective(), _routes(), search_depth=0, enable_v137_6_search=True)

    with pytest.raises(ValueError, match="no candidate routes satisfy bounded search_depth"):
        synthesize_plan_ir(
            _world_state(),
            _objective(),
            (("start", "a", "b", "goal"),),
            search_depth=2,
            enable_v137_6_search=True,
        )
