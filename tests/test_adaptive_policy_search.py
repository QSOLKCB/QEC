"""Tests for adaptive_policy_search — v134.7.0.

Deterministic replay tests for the adaptive policy search layer.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qec.sims.control_policy_benchmark import PolicyBenchmarkResult
from qec.sims.adaptive_policy_search import (
    AdaptivePolicySearchResult,
    search_adaptive_policy,
    _refinement_factor_for,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    name: str,
    score: float,
    *,
    mean_final_amplitude: float = 0.5,
    steps_to_first_stability: int = 5,
    recovery_count: int = 1,
    oscillation_count: int = 2,
) -> PolicyBenchmarkResult:
    return PolicyBenchmarkResult(
        policy_name=name,
        mean_final_amplitude=mean_final_amplitude,
        steps_to_first_stability=steps_to_first_stability,
        recovery_count=recovery_count,
        oscillation_count=oscillation_count,
        score=score,
    )


# ---------------------------------------------------------------------------
# AdaptivePolicySearchResult immutability
# ---------------------------------------------------------------------------

class TestAdaptivePolicySearchResultImmutability:

    def test_frozen(self) -> None:
        result = AdaptivePolicySearchResult(
            best_policy_name="nominal",
            best_score=0.8,
            candidate_damping_factor=0.95,
            search_iteration=0,
            improvement_delta=0.1,
        )
        with pytest.raises(FrozenInstanceError):
            result.best_score = 0.99  # type: ignore[misc]

    def test_fields_preserved(self) -> None:
        result = AdaptivePolicySearchResult(
            best_policy_name="aggressive_damping",
            best_score=0.75,
            candidate_damping_factor=0.9,
            search_iteration=3,
            improvement_delta=0.05,
        )
        assert result.best_policy_name == "aggressive_damping"
        assert result.best_score == 0.75
        assert result.candidate_damping_factor == 0.9
        assert result.search_iteration == 3
        assert result.improvement_delta == 0.05


# ---------------------------------------------------------------------------
# Refinement factor mapping
# ---------------------------------------------------------------------------

class TestRefinementFactor:

    def test_aggressive_damping(self) -> None:
        assert _refinement_factor_for("aggressive_damping") == 0.9

    def test_nominal(self) -> None:
        assert _refinement_factor_for("nominal") == 0.95

    def test_recovery_first(self) -> None:
        assert _refinement_factor_for("recovery_first") == 0.8

    def test_unknown_policy_neutral(self) -> None:
        assert _refinement_factor_for("unknown_policy") == 1.0


# ---------------------------------------------------------------------------
# search_adaptive_policy — best selection
# ---------------------------------------------------------------------------

class TestSearchBestSelection:

    def test_selects_highest_score(self) -> None:
        results = (
            _make_result("nominal", 0.6),
            _make_result("aggressive_damping", 0.8),
            _make_result("recovery_first", 0.5),
        )
        out = search_adaptive_policy(results)
        assert out.best_policy_name == "aggressive_damping"
        assert out.best_score == 0.8

    def test_tie_broken_by_name(self) -> None:
        results = (
            _make_result("nominal", 0.7),
            _make_result("aggressive_damping", 0.7),
        )
        out = search_adaptive_policy(results)
        # Tie on score — alphabetically first name wins.
        assert out.best_policy_name == "aggressive_damping"

    def test_single_result(self) -> None:
        results = (_make_result("nominal", 0.9),)
        out = search_adaptive_policy(results)
        assert out.best_policy_name == "nominal"
        assert out.best_score == 0.9
        assert out.improvement_delta == 0.0


# ---------------------------------------------------------------------------
# search_adaptive_policy — damping refinement
# ---------------------------------------------------------------------------

class TestSearchDampingRefinement:

    def test_aggressive_damping_candidate(self) -> None:
        results = (
            _make_result("aggressive_damping", 0.9),
            _make_result("nominal", 0.5),
        )
        out = search_adaptive_policy(results, base_damping=1.0)
        assert out.candidate_damping_factor == pytest.approx(0.9)

    def test_nominal_candidate(self) -> None:
        results = (
            _make_result("nominal", 0.9),
            _make_result("aggressive_damping", 0.5),
        )
        out = search_adaptive_policy(results, base_damping=1.0)
        assert out.candidate_damping_factor == pytest.approx(0.95)

    def test_recovery_first_candidate(self) -> None:
        results = (
            _make_result("recovery_first", 0.9),
            _make_result("nominal", 0.5),
        )
        out = search_adaptive_policy(results, base_damping=1.0)
        assert out.candidate_damping_factor == pytest.approx(0.8)

    def test_custom_base_damping(self) -> None:
        results = (_make_result("nominal", 0.9),)
        out = search_adaptive_policy(results, base_damping=0.5)
        assert out.candidate_damping_factor == pytest.approx(0.5 * 0.95)

    def test_unknown_policy_neutral_damping(self) -> None:
        results = (_make_result("custom_policy", 0.9),)
        out = search_adaptive_policy(results, base_damping=0.7)
        assert out.candidate_damping_factor == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# search_adaptive_policy — improvement delta
# ---------------------------------------------------------------------------

class TestSearchImprovementDelta:

    def test_delta_between_top_two(self) -> None:
        results = (
            _make_result("aggressive_damping", 0.85),
            _make_result("nominal", 0.60),
            _make_result("recovery_first", 0.40),
        )
        out = search_adaptive_policy(results)
        assert out.improvement_delta == pytest.approx(0.25)

    def test_single_result_zero_delta(self) -> None:
        results = (_make_result("nominal", 0.9),)
        out = search_adaptive_policy(results)
        assert out.improvement_delta == 0.0

    def test_tied_scores_zero_delta(self) -> None:
        results = (
            _make_result("aggressive_damping", 0.7),
            _make_result("nominal", 0.7),
        )
        out = search_adaptive_policy(results)
        assert out.improvement_delta == 0.0


# ---------------------------------------------------------------------------
# search_adaptive_policy — iteration tracking
# ---------------------------------------------------------------------------

class TestSearchIteration:

    def test_default_iteration_zero(self) -> None:
        results = (_make_result("nominal", 0.5),)
        out = search_adaptive_policy(results)
        assert out.search_iteration == 0

    def test_explicit_iteration(self) -> None:
        results = (_make_result("nominal", 0.5),)
        out = search_adaptive_policy(results, search_iteration=7)
        assert out.search_iteration == 7


# ---------------------------------------------------------------------------
# search_adaptive_policy — error handling
# ---------------------------------------------------------------------------

class TestSearchErrors:

    def test_empty_results_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            search_adaptive_policy(())


# ---------------------------------------------------------------------------
# Deterministic replay
# ---------------------------------------------------------------------------

class TestDeterministicReplay:

    def test_identical_inputs_identical_outputs(self) -> None:
        results = (
            _make_result("aggressive_damping", 0.85),
            _make_result("nominal", 0.60),
            _make_result("recovery_first", 0.40),
        )
        out_a = search_adaptive_policy(results, search_iteration=1, base_damping=0.5)
        out_b = search_adaptive_policy(results, search_iteration=1, base_damping=0.5)
        assert out_a == out_b

    def test_input_order_invariance(self) -> None:
        r1 = _make_result("aggressive_damping", 0.85)
        r2 = _make_result("nominal", 0.60)
        r3 = _make_result("recovery_first", 0.40)
        out_a = search_adaptive_policy((r1, r2, r3))
        out_b = search_adaptive_policy((r3, r1, r2))
        assert out_a == out_b

    def test_result_is_frozen(self) -> None:
        results = (_make_result("nominal", 0.5),)
        out = search_adaptive_policy(results)
        with pytest.raises(FrozenInstanceError):
            out.best_score = 999.0  # type: ignore[misc]
