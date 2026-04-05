from __future__ import annotations

import json

import pytest

from qec.analysis.autonomous_planning_kernel import (
    GENESIS_HASH,
    export_policy_ledger_canonical_bytes,
    run_autonomous_planning_kernel,
)


def _world_state() -> dict[str, object]:
    return {
        "start": 0.2,
        "mid": 0.7,
        "goal": 1.4,
        "agent": "planner-rt",
        "flags": ["safe", "runtime"],
    }


def _routes() -> tuple[tuple[str, ...], ...]:
    return (
        ("start", "mid", "goal"),
        ("start", "goal"),
        ("start", "alt", "goal"),
    )


def _edge_costs() -> dict[tuple[str, str], float]:
    return {
        ("start", "mid"): 0.2,
        ("mid", "goal"): 0.2,
        ("start", "goal"): 0.5,
        ("start", "alt"): 0.2,
        ("alt", "goal"): 0.4,
    }


def _bounds() -> dict[str, tuple[float, float]]:
    return {
        "start": (0.0, 0.5),
        "mid": (0.0, 0.6),
        "goal": (0.0, 1.0),
    }


def test_repeated_run_determinism_same_bytes() -> None:
    a = run_autonomous_planning_kernel(
        _world_state(),
        _routes(),
        edge_costs=_edge_costs(),
        world_state_bounds=_bounds(),
        enable_v137_2_runtime=True,
    )
    b = run_autonomous_planning_kernel(
        _world_state(),
        _routes(),
        edge_costs=_edge_costs(),
        world_state_bounds=_bounds(),
        enable_v137_2_runtime=True,
    )
    assert a == b
    assert a.to_canonical_bytes() == b.to_canonical_bytes()


def test_canonical_json_and_bytes_stability() -> None:
    report = run_autonomous_planning_kernel(
        _world_state(),
        _routes(),
        edge_costs=_edge_costs(),
        world_state_bounds=_bounds(),
        enable_v137_2_runtime=True,
    )
    canonical_json = report.to_canonical_json()
    reparsed = json.loads(canonical_json)
    assert json.dumps(reparsed, sort_keys=True, separators=(",", ":"), ensure_ascii=True) == canonical_json
    assert report.to_canonical_bytes() == canonical_json.encode("utf-8")


def test_stable_plan_identity_chain_and_execution_hash_chain() -> None:
    report = run_autonomous_planning_kernel(
        _world_state(),
        _routes(),
        edge_costs=_edge_costs(),
        world_state_bounds=_bounds(),
        enable_v137_2_runtime=True,
    )
    assert len(report.plan_identity_chain) == 6
    for item in report.plan_identity_chain:
        assert len(item) == 64
        int(item, 16)

    steps = report.execution_artifact.steps
    if steps:
        assert steps[0].prior_hash == GENESIS_HASH
        for idx in range(1, len(steps)):
            assert steps[idx].prior_hash == steps[idx - 1].step_hash
        assert report.execution_artifact.execution_head_hash == steps[-1].step_hash


def test_deterministic_route_execution_and_control_synthesis() -> None:
    report = run_autonomous_planning_kernel(
        _world_state(),
        _routes(),
        edge_costs=_edge_costs(),
        world_state_bounds=_bounds(),
        enable_v137_2_runtime=True,
    )
    # Deterministic tie-break: objective desc, route length asc, lexicographic asc.
    assert report.control_synthesis.selected_route == ("start", "goal")
    assert tuple(report.execution_artifact.route) == report.control_synthesis.selected_route


def test_bounded_world_state_policy_outputs_and_stable_policy_ledger() -> None:
    report = run_autonomous_planning_kernel(
        _world_state(),
        _routes(),
        edge_costs=_edge_costs(),
        world_state_bounds=_bounds(),
        enable_v137_2_runtime=True,
    )
    statuses = {item.key: item.status for item in report.bounded_world_state.bounds}
    assert statuses["start"] == "within"
    assert statuses["mid"] == "clamped_high"
    assert statuses["goal"] == "clamped_high"

    exported_a = export_policy_ledger_canonical_bytes(report.policy_ledger)
    exported_b = export_policy_ledger_canonical_bytes(report.policy_ledger)
    assert exported_a == exported_b
    assert report.policy_ledger.head_hash == report.policy_ledger.entries[-1].entry_hash


def test_fail_fast_invalid_input_handling() -> None:
    with pytest.raises(ValueError, match="enable_v137_2_runtime"):
        run_autonomous_planning_kernel(_world_state(), _routes(), edge_costs=_edge_costs(), world_state_bounds=_bounds())

    with pytest.raises(ValueError, match="must normalize to a non-empty mapping"):
        run_autonomous_planning_kernel(
            {},
            _routes(),
            edge_costs=_edge_costs(),
            world_state_bounds=_bounds(),
            enable_v137_2_runtime=True,
        )

    with pytest.raises(ValueError, match="candidate route at index 0"):
        run_autonomous_planning_kernel(
            _world_state(),
            ((),),
            edge_costs=_edge_costs(),
            world_state_bounds=_bounds(),
            enable_v137_2_runtime=True,
        )

    with pytest.raises(ValueError, match="edge costs must be finite and >= 0"):
        run_autonomous_planning_kernel(
            _world_state(),
            _routes(),
            edge_costs={("start", "goal"): -1.0},
            world_state_bounds=_bounds(),
            enable_v137_2_runtime=True,
        )

    with pytest.raises(ValueError, match="world_state bound key missing numeric value"):
        run_autonomous_planning_kernel(
            _world_state(),
            _routes(),
            edge_costs=_edge_costs(),
            world_state_bounds={"not_present": (0.0, 1.0)},
            enable_v137_2_runtime=True,
        )
