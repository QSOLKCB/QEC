"""
Tests for the deterministic co-evolution evaluation layer.

Verifies:
- Deterministic ordering of rule_results
- Correct best rule selection
- Reproducibility across runs
- Integration with experiment runner
- dtype correctness
"""

from __future__ import annotations

import numpy as np
import pytest

from src.qec.decoder.ternary.ternary_coevolution import (
    evaluate_graph_decoder_pair,
    evaluate_rule_population,
    select_best_rule,
)
from src.qec.decoder.ternary.ternary_rule_variants import RULE_REGISTRY


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def small_parity_matrix() -> np.ndarray:
    """3x6 parity check matrix for testing."""
    return np.array(
        [
            [1, 1, 0, 1, 0, 0],
            [0, 1, 1, 0, 1, 0],
            [1, 0, 1, 0, 0, 1],
        ],
        dtype=np.float64,
    )


@pytest.fixture()
def received_vector(small_parity_matrix: np.ndarray) -> np.ndarray:
    """All-ones received vector matching matrix width."""
    return np.ones(small_parity_matrix.shape[1], dtype=np.float64)


# ---------------------------------------------------------------------------
# Tests for evaluate_graph_decoder_pair
# ---------------------------------------------------------------------------


class TestEvaluateGraphDecoderPair:
    """Tests for evaluate_graph_decoder_pair."""

    def test_returns_required_keys(
        self, small_parity_matrix: np.ndarray, received_vector: np.ndarray
    ) -> None:
        result = evaluate_graph_decoder_pair(
            small_parity_matrix, received_vector, "majority"
        )
        required_keys = {
            "rule_name",
            "stability",
            "entropy",
            "conflict_density",
            "trapping_indicator",
            "converged",
            "iterations",
        }
        assert set(result.keys()) == required_keys

    def test_rule_name_matches(
        self, small_parity_matrix: np.ndarray, received_vector: np.ndarray
    ) -> None:
        for rule_name in sorted(RULE_REGISTRY.keys()):
            result = evaluate_graph_decoder_pair(
                small_parity_matrix, received_vector, rule_name
            )
            assert result["rule_name"] == rule_name

    def test_dtype_correctness(
        self, small_parity_matrix: np.ndarray, received_vector: np.ndarray
    ) -> None:
        result = evaluate_graph_decoder_pair(
            small_parity_matrix, received_vector, "majority"
        )
        assert isinstance(result["stability"], np.float64)
        assert isinstance(result["entropy"], np.float64)
        assert isinstance(result["conflict_density"], np.float64)
        assert isinstance(result["trapping_indicator"], np.float64)
        assert isinstance(result["converged"], bool)
        assert isinstance(result["iterations"], int)
        assert isinstance(result["rule_name"], str)

    def test_metric_bounds(
        self, small_parity_matrix: np.ndarray, received_vector: np.ndarray
    ) -> None:
        result = evaluate_graph_decoder_pair(
            small_parity_matrix, received_vector, "majority"
        )
        for key in ("stability", "entropy", "conflict_density", "trapping_indicator"):
            assert 0.0 <= float(result[key]) <= 1.0, f"{key} out of bounds"

    def test_determinism(
        self, small_parity_matrix: np.ndarray, received_vector: np.ndarray
    ) -> None:
        r1 = evaluate_graph_decoder_pair(
            small_parity_matrix, received_vector, "majority"
        )
        r2 = evaluate_graph_decoder_pair(
            small_parity_matrix, received_vector, "majority"
        )
        assert r1 == r2

    def test_invalid_rule_raises(
        self, small_parity_matrix: np.ndarray, received_vector: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="Unknown rule"):
            evaluate_graph_decoder_pair(
                small_parity_matrix, received_vector, "nonexistent_rule"
            )


# ---------------------------------------------------------------------------
# Tests for evaluate_rule_population
# ---------------------------------------------------------------------------


class TestEvaluateRulePopulation:
    """Tests for evaluate_rule_population."""

    def test_returns_all_rules(
        self, small_parity_matrix: np.ndarray, received_vector: np.ndarray
    ) -> None:
        results = evaluate_rule_population(small_parity_matrix, received_vector)
        result_names = {r["rule_name"] for r in results}
        assert result_names == set(RULE_REGISTRY.keys())

    def test_length_matches_registry(
        self, small_parity_matrix: np.ndarray, received_vector: np.ndarray
    ) -> None:
        results = evaluate_rule_population(small_parity_matrix, received_vector)
        assert len(results) == len(RULE_REGISTRY)

    def test_deterministic_ordering(
        self, small_parity_matrix: np.ndarray, received_vector: np.ndarray
    ) -> None:
        r1 = evaluate_rule_population(small_parity_matrix, received_vector)
        r2 = evaluate_rule_population(small_parity_matrix, received_vector)
        for a, b in zip(r1, r2):
            assert a["rule_name"] == b["rule_name"]
            assert a["stability"] == b["stability"]
            assert a["entropy"] == b["entropy"]
            assert a["conflict_density"] == b["conflict_density"]
            assert a["trapping_indicator"] == b["trapping_indicator"]
            assert a["converged"] == b["converged"]
            assert a["iterations"] == b["iterations"]

    def test_sorted_by_stability_descending(
        self, small_parity_matrix: np.ndarray, received_vector: np.ndarray
    ) -> None:
        results = evaluate_rule_population(small_parity_matrix, received_vector)
        for i in range(len(results) - 1):
            s_curr = float(results[i]["stability"])
            s_next = float(results[i + 1]["stability"])
            if s_curr == s_next:
                # Tie-break: rule_name ascending
                assert results[i]["rule_name"] <= results[i + 1]["rule_name"]
            else:
                assert s_curr >= s_next

    def test_all_entries_have_correct_keys(
        self, small_parity_matrix: np.ndarray, received_vector: np.ndarray
    ) -> None:
        results = evaluate_rule_population(small_parity_matrix, received_vector)
        required_keys = {
            "rule_name",
            "stability",
            "entropy",
            "conflict_density",
            "trapping_indicator",
            "converged",
            "iterations",
        }
        for r in results:
            assert set(r.keys()) == required_keys

    def test_reproducibility_across_runs(
        self, small_parity_matrix: np.ndarray, received_vector: np.ndarray
    ) -> None:
        """Three runs must produce identical results."""
        runs = [
            evaluate_rule_population(small_parity_matrix, received_vector)
            for _ in range(3)
        ]
        for i in range(1, len(runs)):
            assert len(runs[0]) == len(runs[i])
            for a, b in zip(runs[0], runs[i]):
                assert a == b


# ---------------------------------------------------------------------------
# Tests for select_best_rule
# ---------------------------------------------------------------------------


class TestSelectBestRule:
    """Tests for select_best_rule."""

    def test_returns_required_keys(
        self, small_parity_matrix: np.ndarray, received_vector: np.ndarray
    ) -> None:
        results = evaluate_rule_population(small_parity_matrix, received_vector)
        best = select_best_rule(results)
        assert set(best.keys()) == {"best_rule", "best_score"}

    def test_best_rule_is_string(
        self, small_parity_matrix: np.ndarray, received_vector: np.ndarray
    ) -> None:
        results = evaluate_rule_population(small_parity_matrix, received_vector)
        best = select_best_rule(results)
        assert isinstance(best["best_rule"], str)

    def test_best_score_is_float64(
        self, small_parity_matrix: np.ndarray, received_vector: np.ndarray
    ) -> None:
        results = evaluate_rule_population(small_parity_matrix, received_vector)
        best = select_best_rule(results)
        assert isinstance(best["best_score"], np.float64)

    def test_best_rule_in_registry(
        self, small_parity_matrix: np.ndarray, received_vector: np.ndarray
    ) -> None:
        results = evaluate_rule_population(small_parity_matrix, received_vector)
        best = select_best_rule(results)
        assert best["best_rule"] in RULE_REGISTRY

    def test_best_has_highest_stability(
        self, small_parity_matrix: np.ndarray, received_vector: np.ndarray
    ) -> None:
        results = evaluate_rule_population(small_parity_matrix, received_vector)
        best = select_best_rule(results)
        max_stability = max(float(r["stability"]) for r in results)
        assert float(best["best_score"]) == max_stability

    def test_deterministic_selection(
        self, small_parity_matrix: np.ndarray, received_vector: np.ndarray
    ) -> None:
        results = evaluate_rule_population(small_parity_matrix, received_vector)
        b1 = select_best_rule(results)
        b2 = select_best_rule(results)
        assert b1["best_rule"] == b2["best_rule"]
        assert b1["best_score"] == b2["best_score"]

    def test_tiebreak_by_rule_name(self) -> None:
        """When stabilities are equal, lowest rule_name wins."""
        tied_results = [
            {
                "rule_name": "beta_rule",
                "stability": np.float64(0.75),
                "entropy": np.float64(0.5),
                "conflict_density": np.float64(0.1),
                "trapping_indicator": np.float64(0.2),
                "converged": True,
                "iterations": 5,
            },
            {
                "rule_name": "alpha_rule",
                "stability": np.float64(0.75),
                "entropy": np.float64(0.6),
                "conflict_density": np.float64(0.2),
                "trapping_indicator": np.float64(0.3),
                "converged": True,
                "iterations": 3,
            },
        ]
        best = select_best_rule(tied_results)
        assert best["best_rule"] == "alpha_rule"
        assert best["best_score"] == np.float64(0.75)

    def test_empty_results(self) -> None:
        best = select_best_rule([])
        assert best["best_rule"] == ""
        assert best["best_score"] == np.float64(0.0)


# ---------------------------------------------------------------------------
# Integration with experiment runner
# ---------------------------------------------------------------------------


class TestExperimentIntegration:
    """Tests for co-evolution integration with discovery_run."""

    def test_coevolution_disabled_by_default(self) -> None:
        from src.qec.experiments.discovery_run import run_discovery_experiment

        spec = {
            "num_variables": 6,
            "num_checks": 3,
            "variable_degree": 3,
            "check_degree": 6,
        }
        artifact = run_discovery_experiment(
            spec,
            num_generations=1,
            population_size=2,
            base_seed=42,
            output_path="artifacts/test_coev_disabled.json",
        )
        assert "decoder_rule_population" not in artifact["results"]

    def test_coevolution_enabled_produces_results(self) -> None:
        from src.qec.experiments.discovery_run import run_discovery_experiment

        spec = {
            "num_variables": 6,
            "num_checks": 3,
            "variable_degree": 3,
            "check_degree": 6,
        }
        artifact = run_discovery_experiment(
            spec,
            num_generations=1,
            population_size=2,
            base_seed=42,
            enable_coevolution=True,
            output_path="artifacts/test_coev_enabled.json",
        )
        assert "decoder_rule_population" in artifact["results"]
        assert "best_decoder_rule" in artifact["results"]
        assert "best_decoder_score" in artifact["results"]

    def test_coevolution_population_has_all_rules(self) -> None:
        from src.qec.experiments.discovery_run import run_discovery_experiment

        spec = {
            "num_variables": 6,
            "num_checks": 3,
            "variable_degree": 3,
            "check_degree": 6,
        }
        artifact = run_discovery_experiment(
            spec,
            num_generations=1,
            population_size=2,
            base_seed=42,
            enable_coevolution=True,
            output_path="artifacts/test_coev_pop.json",
        )
        pop = artifact["results"]["decoder_rule_population"]
        rule_names = {r["rule_name"] for r in pop}
        assert rule_names == set(RULE_REGISTRY.keys())

    def test_coevolution_deterministic(self) -> None:
        from src.qec.experiments.discovery_run import run_discovery_experiment

        spec = {
            "num_variables": 6,
            "num_checks": 3,
            "variable_degree": 3,
            "check_degree": 6,
        }
        a1 = run_discovery_experiment(
            spec,
            num_generations=1,
            population_size=2,
            base_seed=42,
            enable_coevolution=True,
            output_path="artifacts/test_coev_det1.json",
        )
        a2 = run_discovery_experiment(
            spec,
            num_generations=1,
            population_size=2,
            base_seed=42,
            enable_coevolution=True,
            output_path="artifacts/test_coev_det2.json",
        )
        assert a1["results"]["decoder_rule_population"] == a2["results"]["decoder_rule_population"]
        assert a1["results"]["best_decoder_rule"] == a2["results"]["best_decoder_rule"]
        assert a1["results"]["best_decoder_score"] == a2["results"]["best_decoder_score"]

    def test_coevolution_does_not_break_existing_rule_experiments(self) -> None:
        from src.qec.experiments.discovery_run import run_discovery_experiment

        spec = {
            "num_variables": 6,
            "num_checks": 3,
            "variable_degree": 3,
            "check_degree": 6,
        }
        artifact = run_discovery_experiment(
            spec,
            num_generations=1,
            population_size=2,
            base_seed=42,
            enable_decoder_rule_experiments=True,
            enable_coevolution=True,
            output_path="artifacts/test_coev_both.json",
        )
        # Both features produce output
        assert "decoder_rule_population" in artifact["results"]
        assert "decoder_rule_metrics" in artifact["results"]


# ---------------------------------------------------------------------------
# API wrapper tests
# ---------------------------------------------------------------------------


class TestAPIWrappers:
    """Tests for analysis API wrappers."""

    def test_evaluate_graph_decoder_pair_wrapper(
        self, small_parity_matrix: np.ndarray, received_vector: np.ndarray
    ) -> None:
        from src.qec.analysis.api import evaluate_graph_decoder_pair as api_fn

        result = api_fn(small_parity_matrix, received_vector, "majority")
        assert "rule_name" in result
        assert result["rule_name"] == "majority"

    def test_evaluate_rule_population_wrapper(
        self, small_parity_matrix: np.ndarray, received_vector: np.ndarray
    ) -> None:
        from src.qec.analysis.api import evaluate_rule_population as api_fn

        results = api_fn(small_parity_matrix, received_vector)
        assert len(results) == len(RULE_REGISTRY)

    def test_select_best_rule_wrapper(
        self, small_parity_matrix: np.ndarray, received_vector: np.ndarray
    ) -> None:
        from src.qec.analysis.api import (
            evaluate_rule_population as pop_fn,
            select_best_rule as best_fn,
        )

        results = pop_fn(small_parity_matrix, received_vector)
        best = best_fn(results)
        assert "best_rule" in best
        assert "best_score" in best
        assert best["best_rule"] in RULE_REGISTRY
