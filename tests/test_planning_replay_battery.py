from __future__ import annotations

from collections import OrderedDict

import pytest

from qec.analysis.policy_constrained_planner import (
    analyze_policy_constrained_frontier,
    generate_policy_decision_receipt,
)
from qec.analysis.planning_replay_battery import (
    GENESIS_HASH,
    PlanningReplayCertificationArtifact,
    certify_planning_replay_battery,
    export_planning_replay_certification_bytes,
    generate_planning_replay_certification_receipt,
    run_adversarial_replay_harness,
)
from qec.analysis.route_dead_end_pruning import (
    analyze_dead_end_pruning,
    generate_dead_end_pruning_receipt,
)


def _world_state() -> dict[str, object]:
    return {
        "agent": "planning-replay-v137-6-4",
        "energy": 4.0,
        "state": {"goal": 1.0, "start": 0.0, "alpha": 0.5},
        "flags": ["deterministic", "replay_safe"],
    }


def _objective() -> dict[str, object]:
    return {
        "preferred_nodes": ["goal", "alpha"],
        "policy": "certification",
    }


def _candidate_routes() -> tuple[tuple[str, ...], ...]:
    return (
        ("start", "alpha", "goal"),
        ("start", "beta", "goal"),
        ("start", "gamma", "goal"),
    )


def _route_graph() -> dict[str, tuple[str, ...]]:
    return {
        "start": ("gamma", "alpha", "beta"),
        "alpha": ("goal", "loop"),
        "beta": ("trap",),
        "gamma": ("goal",),
        "goal": tuple(),
        "loop": ("loop",),
        "trap": ("trap",),
    }


def _policy_rules() -> dict[str, object]:
    return {
        "forbidden_nodes": ("beta",),
        "required_terminal_subset": ("goal",),
        "must_pass_through_nodes": ("alpha",),
    }


def test_byte_identity_replay_same_input_same_bytes() -> None:
    artifact_a = certify_planning_replay_battery(
        _world_state(),
        _objective(),
        _candidate_routes(),
        _route_graph(),
        initial_node="start",
        search_depth=3,
        max_path_length=1,
        policy_rules=_policy_rules(),
        replay_run_count=2,
        stress_run_count=1,
        enable_v137_6_4_replay_battery=True,
    )
    artifact_b = certify_planning_replay_battery(
        _world_state(),
        _objective(),
        _candidate_routes(),
        _route_graph(),
        initial_node="start",
        search_depth=3,
        max_path_length=1,
        policy_rules=_policy_rules(),
        replay_run_count=2,
        stress_run_count=1,
        enable_v137_6_4_replay_battery=True,
    )

    assert artifact_a == artifact_b
    assert export_planning_replay_certification_bytes(artifact_a) == export_planning_replay_certification_bytes(artifact_b)


def test_frontier_stability_pruning_stability_and_identity_chain() -> None:
    artifact = certify_planning_replay_battery(
        _world_state(),
        _objective(),
        _candidate_routes(),
        _route_graph(),
        initial_node="start",
        search_depth=3,
        max_path_length=1,
        policy_rules=_policy_rules(),
        replay_run_count=3,
        stress_run_count=2,
        enable_v137_6_4_replay_battery=True,
    )

    assert artifact.frontier_order_stable is True
    assert artifact.pruning_hash_stable is True
    assert artifact.policy_hash_stable is True
    assert artifact.byte_identity_verified is True
    assert artifact.replay_identity_chain[0] == GENESIS_HASH
    assert len(artifact.replay_identity_chain) == artifact.replay_run_count + 1
    assert 0.0 <= artifact.certification_score <= 1.0


def test_pruning_artifact_and_receipt_hash_stability() -> None:
    artifact_a = analyze_dead_end_pruning(
        "d" * 64,
        _route_graph(),
        current_path=("start",),
        max_path_length=3,
        enable_v137_6_2_dead_end_pruning=True,
    )
    artifact_b = analyze_dead_end_pruning(
        "d" * 64,
        _route_graph(),
        current_path=("start",),
        max_path_length=3,
        enable_v137_6_2_dead_end_pruning=True,
    )

    receipt_a = generate_dead_end_pruning_receipt(artifact_a)
    receipt_b = generate_dead_end_pruning_receipt(artifact_b)
    assert artifact_a.stable_pruning_hash == artifact_b.stable_pruning_hash
    assert receipt_a.receipt_hash == receipt_b.receipt_hash


def test_policy_receipt_replay_identity_stability() -> None:
    policy_a = analyze_policy_constrained_frontier(
        "e" * 64,
        _route_graph(),
        current_path=("start",),
        frontier_candidates=("alpha", "beta", "gamma"),
        max_path_length=3,
        policy_rules=_policy_rules(),
        enable_v137_6_3_policy_constraints=True,
    )
    policy_b = analyze_policy_constrained_frontier(
        "e" * 64,
        _route_graph(),
        current_path=("start",),
        frontier_candidates=("gamma", "alpha", "beta"),
        max_path_length=3,
        policy_rules=_policy_rules(),
        enable_v137_6_3_policy_constraints=True,
    )

    receipt_a = generate_policy_decision_receipt(policy_a)
    receipt_b = generate_policy_decision_receipt(policy_b)
    assert policy_a.stable_policy_hash == policy_b.stable_policy_hash
    assert receipt_a.receipt_hash == receipt_b.receipt_hash


def test_end_to_end_receipt_stability_and_type() -> None:
    artifact = certify_planning_replay_battery(
        _world_state(),
        _objective(),
        _candidate_routes(),
        _route_graph(),
        initial_node="start",
        search_depth=3,
        max_path_length=1,
        policy_rules=_policy_rules(),
        replay_run_count=2,
        stress_run_count=1,
        enable_v137_6_4_replay_battery=True,
    )
    receipt_a = generate_planning_replay_certification_receipt(artifact)
    receipt_b = generate_planning_replay_certification_receipt(artifact)

    assert isinstance(artifact, PlanningReplayCertificationArtifact)
    assert receipt_a == receipt_b
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_malformed_input_replay_is_deterministic() -> None:
    signatures_a = run_adversarial_replay_harness(
        "f" * 64,
        _route_graph(),
        current_path=("start",),
        frontier_candidates=("alpha", "beta", "gamma"),
        max_path_length=3,
    )
    signatures_b = run_adversarial_replay_harness(
        "f" * 64,
        _route_graph(),
        current_path=("start",),
        frontier_candidates=("alpha", "beta", "gamma"),
        max_path_length=3,
    )

    assert signatures_a == signatures_b
    assert len(signatures_a) >= 8


def test_adversarial_ordering_replay_normalizes_to_identical_receipts() -> None:
    scrambled_world_state = OrderedDict(
        [
            ("flags", ["deterministic", "replay_safe"]),
            ("state", OrderedDict([("alpha", 0.5), ("start", 0.0), ("goal", 1.0)])),
            ("energy", 4.0),
            ("agent", "planning-replay-v137-6-4"),
        ]
    )
    scrambled_objective = OrderedDict([("policy", "certification"), ("preferred_nodes", ["goal", "alpha"])])
    scrambled_graph = OrderedDict(
        [
            ("trap", ("trap",)),
            ("goal", tuple()),
            ("beta", ("trap",)),
            ("start", ("beta", "gamma", "alpha")),
            ("gamma", ("goal",)),
            ("loop", ("loop",)),
            ("alpha", ("loop", "goal")),
        ]
    )

    base = certify_planning_replay_battery(
        _world_state(),
        _objective(),
        _candidate_routes(),
        _route_graph(),
        initial_node="start",
        search_depth=3,
        max_path_length=1,
        policy_rules=_policy_rules(),
        replay_run_count=2,
        stress_run_count=1,
        enable_v137_6_4_replay_battery=True,
    )
    scrambled = certify_planning_replay_battery(
        scrambled_world_state,
        scrambled_objective,
        _candidate_routes(),
        scrambled_graph,
        initial_node="start",
        search_depth=3,
        max_path_length=1,
        policy_rules=OrderedDict([("must_pass_through_nodes", ("alpha",)), ("required_terminal_subset", ("goal",)), ("forbidden_nodes", ("beta",))]),
        replay_run_count=2,
        stress_run_count=1,
        enable_v137_6_4_replay_battery=True,
    )

    assert generate_planning_replay_certification_receipt(base).receipt_hash == generate_planning_replay_certification_receipt(scrambled).receipt_hash


def test_repeated_run_stress_replay_identity() -> None:
    artifact = certify_planning_replay_battery(
        _world_state(),
        _objective(),
        _candidate_routes(),
        _route_graph(),
        initial_node="start",
        search_depth=3,
        max_path_length=1,
        policy_rules=_policy_rules(),
        replay_run_count=2,
        stress_run_count=10,
        enable_v137_6_4_replay_battery=True,
    )

    assert artifact.byte_identity_verified is True
    assert artifact.adversarial_cases_passed is True
    assert artifact.certification_score == pytest.approx(1.0)


def test_certification_route_graph_hash_matches_pruning_artifact() -> None:
    """route_graph_hash on the certification artifact must equal the pruning artifact's route_graph_hash."""
    from qec.analysis.autonomous_planning_search_kernel import synthesize_plan_ir
    from qec.analysis.route_graph_execution_runtime import execute_route_graph

    plan = synthesize_plan_ir(
        _world_state(),
        _objective(),
        _candidate_routes(),
        search_depth=3,
        enable_v137_6_search=True,
    )
    execution = execute_route_graph(
        plan.stable_plan_hash,
        _route_graph(),
        initial_node="start",
        world_state=_world_state(),
        max_path_length=1,
        enable_v137_6_route_runtime=True,
    )
    pruning = analyze_dead_end_pruning(
        plan.stable_plan_hash,
        _route_graph(),
        current_path=execution.executed_route,
        max_path_length=1,
        enable_v137_6_2_dead_end_pruning=True,
    )
    artifact = certify_planning_replay_battery(
        _world_state(),
        _objective(),
        _candidate_routes(),
        _route_graph(),
        initial_node="start",
        search_depth=3,
        max_path_length=1,
        policy_rules=_policy_rules(),
        replay_run_count=2,
        stress_run_count=1,
        enable_v137_6_4_replay_battery=True,
    )
    assert artifact.route_graph_hash == pruning.route_graph_hash


def test_fail_fast_validation_for_battery_flags_and_counts() -> None:
    with pytest.raises(ValueError, match="enable_v137_6_4_replay_battery"):
        certify_planning_replay_battery(
            _world_state(),
            _objective(),
            _candidate_routes(),
            _route_graph(),
            initial_node="start",
            search_depth=3,
            max_path_length=1,
            policy_rules=_policy_rules(),
        )

    with pytest.raises(ValueError, match="replay_run_count must be >= 2"):
        certify_planning_replay_battery(
            _world_state(),
            _objective(),
            _candidate_routes(),
            _route_graph(),
            initial_node="start",
            search_depth=3,
            max_path_length=1,
            policy_rules=_policy_rules(),
            replay_run_count=1,
            enable_v137_6_4_replay_battery=True,
        )

    with pytest.raises(ValueError, match="stress_run_count must be >= 1"):
        certify_planning_replay_battery(
            _world_state(),
            _objective(),
            _candidate_routes(),
            _route_graph(),
            initial_node="start",
            search_depth=3,
            max_path_length=1,
            policy_rules=_policy_rules(),
            stress_run_count=0,
            enable_v137_6_4_replay_battery=True,
        )

