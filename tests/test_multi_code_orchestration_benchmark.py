from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qec.analysis.multi_code_orchestration_benchmark import (
    BenchmarkPolicy,
    OrchestrationCandidate,
    benchmark_multi_code_orchestration,
)


def _policy(*, require_cross_family_benefit: bool = False) -> BenchmarkPolicy:
    return BenchmarkPolicy(
        minimum_selection_confidence=0.6,
        minimum_migration_confidence=0.6,
        maximum_projected_loss=0.5,
        maximum_migration_overhead=0.5,
        require_cross_family_benefit=require_cross_family_benefit,
        weights={
            "stability_gain": 1.0,
            "loss_reduction": 1.0,
            "hardware_gain": 1.0,
            "efficiency_gain": 1.0,
            "overhead_penalty": 1.0,
            "confidence_gain": 1.0,
        },
    )


def _candidate(
    candidate_id: str,
    *,
    source_family: str = "surface",
    target_family: str = "surface",
    selection_confidence: float = 0.8,
    migration_confidence: float = 0.8,
    logical_stability: float = 0.8,
    projected_loss: float = 0.2,
    hardware_alignment: float = 0.8,
    execution_efficiency: float = 0.8,
    migration_overhead: float = 0.2,
    orchestration_depth: int = 0,
) -> OrchestrationCandidate:
    return OrchestrationCandidate(
        candidate_id=candidate_id,
        source_family=source_family,
        target_family=target_family,
        selection_confidence=selection_confidence,
        migration_confidence=migration_confidence,
        logical_stability=logical_stability,
        projected_loss=projected_loss,
        hardware_alignment=hardware_alignment,
        execution_efficiency=execution_efficiency,
        migration_overhead=migration_overhead,
        orchestration_depth=orchestration_depth,
    )


def test_benchmark_success_with_baseline_and_cross_family_candidates() -> None:
    receipt = benchmark_multi_code_orchestration(
        [
            _candidate("baseline", orchestration_depth=0),
            _candidate("selected", orchestration_depth=1, logical_stability=0.85),
            _candidate("migrated", source_family="surface", target_family="toric", orchestration_depth=2, logical_stability=0.9),
        ],
        _policy(),
    )

    assert receipt.comparison.best_candidate_id == "migrated"
    assert receipt.comparison.improvement_margin > 0.0
    assert receipt.comparison.deterministic_ranking[0] == "migrated"


def test_deterministic_baseline_selection() -> None:
    receipt = benchmark_multi_code_orchestration(
        [
            _candidate("baseline_low", orchestration_depth=0, logical_stability=0.7),
            _candidate("baseline_high", orchestration_depth=0, logical_stability=0.8),
            _candidate("x", orchestration_depth=1, logical_stability=0.75),
        ],
        _policy(),
    )
    assert receipt.comparison.baseline_utility == next(
        s.orchestration_utility for s in receipt.benchmark_scores if s.candidate_id == "baseline_high"
    )


def test_no_baseline_rejected() -> None:
    with pytest.raises(ValueError, match="baseline"):
        benchmark_multi_code_orchestration(
            [_candidate("x", orchestration_depth=1)],
            _policy(),
        )


def test_inadmissible_candidate_visibility_in_receipt() -> None:
    receipt = benchmark_multi_code_orchestration(
        [
            _candidate("baseline", orchestration_depth=0),
            _candidate("inadmissible", orchestration_depth=1, selection_confidence=0.2),
        ],
        _policy(),
    )
    score = next(s for s in receipt.benchmark_scores if s.candidate_id == "inadmissible")
    assert score.admissible is False
    assert "selection_confidence_below_minimum" in score.reasons


def test_best_admissible_comparison_logic() -> None:
    receipt = benchmark_multi_code_orchestration(
        [
            _candidate("baseline", orchestration_depth=0, logical_stability=0.5, projected_loss=0.4),
            _candidate("inadmissible_best", orchestration_depth=2, logical_stability=1.0, selection_confidence=0.2, migration_confidence=1.0, projected_loss=0.0, hardware_alignment=1.0, execution_efficiency=1.0, migration_overhead=0.0),
            _candidate("admissible_best", orchestration_depth=1, logical_stability=0.7, projected_loss=0.3, hardware_alignment=0.7, execution_efficiency=0.7),
        ],
        _policy(),
    )
    assert receipt.comparison.best_overall_candidate_id == "inadmissible_best"
    assert receipt.comparison.best_candidate_id == "admissible_best"


def test_require_cross_family_benefit_behavior() -> None:
    receipt = benchmark_multi_code_orchestration(
        [
            _candidate("baseline", orchestration_depth=0),
            _candidate("best_same_family", orchestration_depth=1, logical_stability=0.95),
            _candidate("cross_family_worse", source_family="surface", target_family="toric", orchestration_depth=2, logical_stability=0.7),
        ],
        _policy(require_cross_family_benefit=True),
    )
    assert receipt.comparison.best_candidate_id == "best_same_family"
    assert receipt.comparison.cross_family_winner is False


def test_duplicate_candidate_rejection() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        benchmark_multi_code_orchestration(
            [_candidate("dup"), _candidate("dup")],
            _policy(),
        )


def test_invalid_numeric_and_bool_rejection() -> None:
    with pytest.raises(ValueError, match="selection_confidence"):
        _candidate("bad", selection_confidence=True)
    with pytest.raises(ValueError, match="weight"):
        BenchmarkPolicy(
            minimum_selection_confidence=0.5,
            minimum_migration_confidence=0.5,
            maximum_projected_loss=0.5,
            maximum_migration_overhead=0.5,
            require_cross_family_benefit=False,
            weights={
                "stability_gain": 1.0,
                "loss_reduction": 1.0,
                "hardware_gain": 1.0,
                "efficiency_gain": 1.0,
                "overhead_penalty": True,
                "confidence_gain": 1.0,
            },
        )


def test_canonical_json_stability_and_hash_determinism() -> None:
    candidates = [
        _candidate("baseline", orchestration_depth=0),
        _candidate("migrated", source_family="surface", target_family="toric", orchestration_depth=2, logical_stability=0.9),
    ]
    receipt_a = benchmark_multi_code_orchestration(candidates, _policy())
    receipt_b = benchmark_multi_code_orchestration(tuple(candidates), _policy())

    assert receipt_a.to_canonical_json() == receipt_b.to_canonical_json()
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()
    assert receipt_a.replay_identity == receipt_b.replay_identity
    assert receipt_a.stable_hash == receipt_b.stable_hash


def test_stable_hash_value_reproduces_stored_stable_hash() -> None:
    receipt = benchmark_multi_code_orchestration(
        [
            _candidate("baseline", orchestration_depth=0),
            _candidate("migrated", source_family="surface", target_family="toric", orchestration_depth=1),
        ],
        _policy(),
    )

    assert receipt.stable_hash_value() == receipt.stable_hash


def test_receipt_canonical_payload_contains_schema_version() -> None:
    receipt = benchmark_multi_code_orchestration(
        [
            _candidate("baseline", orchestration_depth=0),
            _candidate("migrated", source_family="surface", target_family="toric", orchestration_depth=1),
        ],
        _policy(),
    )

    assert receipt.to_dict()["schema_version"] == "1"
    assert '"schema_version":"1"' in receipt.to_canonical_json()
    assert b'"schema_version":"1"' in receipt.to_canonical_bytes()


def test_policy_weights_are_frozen_against_caller_mutation() -> None:
    source_weights = {
        "stability_gain": 1.0,
        "loss_reduction": 1.0,
        "hardware_gain": 1.0,
        "efficiency_gain": 1.0,
        "overhead_penalty": 1.0,
        "confidence_gain": 1.0,
    }
    policy = BenchmarkPolicy(
        minimum_selection_confidence=0.6,
        minimum_migration_confidence=0.6,
        maximum_projected_loss=0.5,
        maximum_migration_overhead=0.5,
        require_cross_family_benefit=False,
        weights=source_weights,
    )
    source_weights["stability_gain"] = 99.0
    assert policy.weights["stability_gain"] == 1.0
    with pytest.raises(TypeError):
        policy.weights["stability_gain"] = 2.0  # type: ignore[index]


def test_unknown_benchmark_weight_keys_rejected() -> None:
    with pytest.raises(ValueError, match="unexpected benchmark weight keys"):
        BenchmarkPolicy(
            minimum_selection_confidence=0.6,
            minimum_migration_confidence=0.6,
            maximum_projected_loss=0.5,
            maximum_migration_overhead=0.5,
            require_cross_family_benefit=False,
            weights={
                "stability_gain": 1.0,
                "loss_reduction": 1.0,
                "hardware_gain": 1.0,
                "efficiency_gain": 1.0,
                "overhead_penalty": 1.0,
                "confidence_gain": 1.0,
                "extra_weight": 1.0,
            },
        )


def test_frozen_dataclass_immutability() -> None:
    candidate = _candidate("immut")
    with pytest.raises(FrozenInstanceError):
        candidate.candidate_id = "changed"  # type: ignore[misc]


def test_deterministic_ranking_order_with_ties() -> None:
    receipt = benchmark_multi_code_orchestration(
        [
            _candidate("baseline", orchestration_depth=0, logical_stability=0.6),
            _candidate("bbb", orchestration_depth=1, logical_stability=0.8),
            _candidate("aaa", orchestration_depth=1, logical_stability=0.8),
        ],
        _policy(),
    )

    assert receipt.comparison.deterministic_ranking[:2] == ("aaa", "bbb")
