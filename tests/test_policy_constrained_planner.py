from __future__ import annotations

import pytest
from typing import Any

from qec.analysis.policy_constrained_planner import (
    GENESIS_HASH,
    POLICY_CONSTRAINED_PLANNER_VERSION,
    admit_policy_constrained_frontier,
    analyze_policy_constrained_frontier,
    compute_policy_pressure,
    export_policy_decision_bytes,
    generate_policy_decision_receipt,
)
from qec.analysis.route_dead_end_pruning import prune_route_frontier


def _source_plan_hash() -> str:
    return "c" * 64


def _graph() -> dict[str, tuple[str, ...]]:
    return {
        "start": ("alpha", "beta", "gamma"),
        "alpha": ("must",),
        "beta": ("forbidden_terminal",),
        "gamma": ("goal",),
        "must": ("goal",),
        "forbidden_terminal": tuple(),
        "goal": tuple(),
    }


def test_deterministic_policy_verdicts_same_input_same_bytes() -> None:
    rules = {
        "max_depth": 3,
        "forbidden_nodes": ("beta",),
        "required_terminal_subset": ("goal",),
        "must_pass_through_nodes": ("must",),
    }
    a = analyze_policy_constrained_frontier(
        _source_plan_hash(),
        _graph(),
        current_path=("start",),
        frontier_candidates=("gamma", "alpha", "beta"),
        max_path_length=4,
        policy_rules=rules,
        enable_v137_6_3_policy_constraints=True,
    )
    b = analyze_policy_constrained_frontier(
        _source_plan_hash(),
        _graph(),
        current_path=("start",),
        frontier_candidates=("gamma", "alpha", "beta"),
        max_path_length=4,
        policy_rules=rules,
        enable_v137_6_3_policy_constraints=True,
    )

    assert a == b
    assert export_policy_decision_bytes(a) == export_policy_decision_bytes(b)
    assert a.stable_policy_hash == b.stable_policy_hash
    assert a.schema_version == POLICY_CONSTRAINED_PLANNER_VERSION
    assert a.policy_identity_chain[0] == GENESIS_HASH


def test_stable_ordering_and_rule_enforcement() -> None:
    artifact = analyze_policy_constrained_frontier(
        _source_plan_hash(),
        _graph(),
        current_path=("start",),
        frontier_candidates=("gamma", "beta", "alpha"),
        max_path_length=4,
        policy_rules={
            "forbidden_nodes": ("beta",),
            "forbidden_transitions": (("start", "gamma"),),
            "required_terminal_subset": ("goal",),
        },
        enable_v137_6_3_policy_constraints=True,
    )

    assert artifact.examined_candidates == ("alpha", "beta", "gamma")
    assert artifact.rejected_candidates == ("beta", "gamma")
    assert artifact.admitted_candidates == ("alpha",)
    assert artifact.violated_rules == (
        "forbidden_nodes",
        "forbidden_transitions",
        "required_terminal_subset",
    )


def test_max_depth_required_terminal_and_pressure_bounded() -> None:
    artifact = analyze_policy_constrained_frontier(
        _source_plan_hash(),
        _graph(),
        current_path=("start", "alpha"),
        frontier_candidates=("must",),
        max_path_length=4,
        policy_rules={"max_depth": 2, "required_terminal_subset": ("goal",)},
        enable_v137_6_3_policy_constraints=True,
    )

    assert artifact.rejected_candidates == ("must",)
    assert "max_depth" in artifact.violated_rules
    assert 0.0 <= artifact.policy_pressure_score <= 1.0
    assert compute_policy_pressure(3, 0) == 0.0
    assert compute_policy_pressure(3, 3) == 1.0


def test_must_pass_through_generates_constrained_status_and_warning_pressure() -> None:
    artifact = analyze_policy_constrained_frontier(
        _source_plan_hash(),
        _graph(),
        current_path=("start",),
        frontier_candidates=("alpha", "gamma"),
        max_path_length=4,
        policy_rules={"must_pass_through_nodes": ("must",)},
        enable_v137_6_3_policy_constraints=True,
    )

    assert artifact.constrained_candidates == ("alpha",)
    assert artifact.admitted_candidates == ("alpha",)
    assert artifact.rejected_candidates == ("gamma",)
    constrained_decision = next(d for d in artifact.candidate_decisions if d.candidate_node == "alpha")
    assert constrained_decision.status == "constrained"
    assert constrained_decision.warning_pressure == pytest.approx(0.5)


def test_canonical_export_and_receipt_replay_stability() -> None:
    artifact = analyze_policy_constrained_frontier(
        _source_plan_hash(),
        _graph(),
        current_path=("start",),
        frontier_candidates=("alpha", "beta", "gamma"),
        max_path_length=4,
        policy_rules={"forbidden_nodes": ("beta",)},
        enable_v137_6_3_policy_constraints=True,
    )

    receipt_a = generate_policy_decision_receipt(artifact)
    receipt_b = generate_policy_decision_receipt(artifact)

    assert artifact.to_canonical_json().encode("utf-8") == artifact.to_canonical_bytes()
    assert receipt_a == receipt_b
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_fail_fast_validation_for_malformed_inputs() -> None:
    with pytest.raises(ValueError, match="enable_v137_6_3_policy_constraints"):
        analyze_policy_constrained_frontier(
            _source_plan_hash(),
            _graph(),
            current_path=("start",),
            frontier_candidates=("alpha",),
            max_path_length=4,
            policy_rules={},
        )

    with pytest.raises(ValueError, match="unsupported policy_rules keys"):
        analyze_policy_constrained_frontier(
            _source_plan_hash(),
            _graph(),
            current_path=("start",),
            frontier_candidates=("alpha",),
            max_path_length=4,
            policy_rules={"unknown_rule": 1},
            enable_v137_6_3_policy_constraints=True,
        )

    with pytest.raises(ValueError, match="rejected_candidates must be <= total_candidates_examined"):
        compute_policy_pressure(1, 2)


def test_frontier_candidate_absent_from_graph_raises_value_error() -> None:
    with pytest.raises(ValueError, match="not in route_graph universe"):
        analyze_policy_constrained_frontier(
            _source_plan_hash(),
            _graph(),
            current_path=("start",),
            frontier_candidates=("alpha", "unknown_node"),
            max_path_length=4,
            policy_rules={},
            enable_v137_6_3_policy_constraints=True,
        )


def test_policy_rules_field_is_immutable_and_replay_bytes_stable() -> None:
    mutable_rules: dict[str, Any] = {"forbidden_nodes": ("beta",)}
    artifact = analyze_policy_constrained_frontier(
        _source_plan_hash(),
        _graph(),
        current_path=("start",),
        frontier_candidates=("alpha",),
        max_path_length=4,
        policy_rules=mutable_rules,
        enable_v137_6_3_policy_constraints=True,
    )

    original_bytes = artifact.to_canonical_bytes()

    # Mutating the original input dict must not affect the artifact snapshot.
    mutable_rules["forbidden_nodes"] = ("alpha",)
    mutable_rules["max_depth"] = 1
    assert artifact.to_canonical_bytes() == original_bytes

    # policy_rules stored on the artifact must not be a plain mutable dict.
    assert not isinstance(artifact.policy_rules, dict)


def test_integration_sanity_after_v137_6_2_pruning_flow() -> None:
    pruned_frontier, _ = prune_route_frontier(
        _source_plan_hash(),
        _graph(),
        current_path=("start",),
        max_path_length=4,
        enable_v137_6_2_dead_end_pruning=True,
    )

    admitted, artifact = admit_policy_constrained_frontier(
        _source_plan_hash(),
        _graph(),
        current_path=("start",),
        frontier_candidates=pruned_frontier,
        max_path_length=4,
        policy_rules={"forbidden_nodes": ("beta",)},
        enable_v137_6_3_policy_constraints=True,
    )

    assert admitted == ("alpha", "gamma")
    assert artifact.rejected_candidates == ("beta",)
