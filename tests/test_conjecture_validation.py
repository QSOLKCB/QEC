"""Deterministic tests for conjecture validation experiments (v62.1.0)."""

from __future__ import annotations

import numpy as np
import pytest

from src.qec.analysis.conjecture_validation import (
    evaluate_conjecture,
    find_conjecture_counterexamples,
    design_validation_experiment,
)


def _make_linear_conjecture() -> dict:
    """Create a simple linear conjecture with known coefficients."""
    return {
        "conjecture_id": "spectral_conjecture_000",
        "model_type": "linear",
        "equation": "estimated_threshold ≈ 0.1 + 0.2 * spectral_radius",
        "fit_quality": 0.9,
        "coefficients": [0.1, 0.2, -0.05, 0.3, -0.1],
    }


def _make_phase_profile(
    phase_id: int,
    spectral_radius: float = 1.0,
    bethe_min: float = -0.1,
    bp_stability: float = 0.8,
    trapping_density: float = 0.2,
    estimated_threshold: float = 0.5,
) -> dict:
    """Create a deterministic phase profile."""
    return {
        "phase_id": phase_id,
        "phase_label": "stable_bp_phase",
        "spectral_radius": float(spectral_radius),
        "bethe_hessian_min_eigenvalue": float(bethe_min),
        "bp_stability_score": float(bp_stability),
        "trapping_density": float(trapping_density),
        "estimated_threshold": float(estimated_threshold),
    }


def _make_phase_profiles(n: int = 5) -> list[dict]:
    """Create a list of deterministic phase profiles."""
    profiles = []
    for i in range(n):
        profiles.append(
            _make_phase_profile(
                phase_id=i,
                spectral_radius=1.0 + 0.1 * i,
                bethe_min=-0.1 - 0.05 * i,
                bp_stability=0.8 - 0.05 * i,
                trapping_density=0.2 + 0.03 * i,
                estimated_threshold=0.5 + 0.02 * i,
            )
        )
    return profiles


class TestEvaluateConjecture:
    """Tests for evaluate_conjecture determinism and correctness."""

    def test_returns_all_fields(self):
        conjecture = _make_linear_conjecture()
        profile = _make_phase_profile(0)
        result = evaluate_conjecture(conjecture, profile)
        assert "predicted_value" in result
        assert "observed_value" in result
        assert "absolute_error" in result
        assert "squared_error" in result

    def test_float64_types(self):
        conjecture = _make_linear_conjecture()
        profile = _make_phase_profile(0)
        result = evaluate_conjecture(conjecture, profile)
        for key in ("predicted_value", "observed_value", "absolute_error", "squared_error"):
            assert isinstance(result[key], float)

    def test_deterministic_across_calls(self):
        conjecture = _make_linear_conjecture()
        profile = _make_phase_profile(0)
        r1 = evaluate_conjecture(conjecture, profile)
        r2 = evaluate_conjecture(conjecture, profile)
        assert r1 == r2

    def test_observed_matches_profile(self):
        conjecture = _make_linear_conjecture()
        profile = _make_phase_profile(0, estimated_threshold=0.42)
        result = evaluate_conjecture(conjecture, profile)
        assert result["observed_value"] == pytest.approx(0.42)

    def test_squared_error_is_square_of_absolute(self):
        conjecture = _make_linear_conjecture()
        profile = _make_phase_profile(0)
        result = evaluate_conjecture(conjecture, profile)
        assert result["squared_error"] == pytest.approx(
            result["absolute_error"] ** 2
        )

    def test_zero_error_on_perfect_prediction(self):
        """A constant conjecture matching the target should have zero error."""
        profile = _make_phase_profile(0, estimated_threshold=0.5)
        conjecture = {"conjecture_id": "const", "model_type": "constant", "target_value": 0.5}
        result = evaluate_conjecture(conjecture, profile)
        assert result["absolute_error"] == pytest.approx(0.0)
        assert result["squared_error"] == pytest.approx(0.0)

    def test_different_profiles_different_results(self):
        conjecture = _make_linear_conjecture()
        p1 = _make_phase_profile(0, spectral_radius=1.0)
        p2 = _make_phase_profile(1, spectral_radius=2.0)
        r1 = evaluate_conjecture(conjecture, p1)
        r2 = evaluate_conjecture(conjecture, p2)
        assert r1["predicted_value"] != r2["predicted_value"]


class TestFindConjectureCounterexamples:
    """Tests for find_conjecture_counterexamples determinism."""

    def test_returns_list(self):
        conjecture = _make_linear_conjecture()
        profiles = _make_phase_profiles(5)
        result = find_conjecture_counterexamples(conjecture, profiles)
        assert isinstance(result, list)

    def test_counterexample_fields(self):
        conjecture = _make_linear_conjecture()
        profiles = _make_phase_profiles(5)
        result = find_conjecture_counterexamples(
            conjecture, profiles, error_threshold=0.0
        )
        if result:
            rec = result[0]
            assert "phase_id" in rec
            assert "predicted" in rec
            assert "observed" in rec
            assert "error" in rec

    def test_sorted_by_descending_error(self):
        conjecture = _make_linear_conjecture()
        profiles = _make_phase_profiles(10)
        result = find_conjecture_counterexamples(
            conjecture, profiles, error_threshold=0.0
        )
        if len(result) >= 2:
            for i in range(len(result) - 1):
                assert result[i]["error"] >= result[i + 1]["error"]

    def test_deterministic_across_calls(self):
        conjecture = _make_linear_conjecture()
        profiles = _make_phase_profiles(5)
        r1 = find_conjecture_counterexamples(conjecture, profiles)
        r2 = find_conjecture_counterexamples(conjecture, profiles)
        assert r1 == r2

    def test_max_counterexamples_limit(self):
        conjecture = _make_linear_conjecture()
        profiles = _make_phase_profiles(10)
        result = find_conjecture_counterexamples(
            conjecture, profiles, error_threshold=0.0, max_counterexamples=3
        )
        assert len(result) <= 3

    def test_high_threshold_filters_all(self):
        conjecture = {"conjecture_id": "const", "model_type": "constant", "target_value": 0.5}
        profiles = [_make_phase_profile(0, estimated_threshold=0.5)]
        result = find_conjecture_counterexamples(
            conjecture, profiles, error_threshold=1.0
        )
        assert len(result) == 0

    def test_empty_profiles(self):
        conjecture = _make_linear_conjecture()
        result = find_conjecture_counterexamples(conjecture, [])
        assert result == []


class TestDesignValidationExperiment:
    """Tests for design_validation_experiment determinism."""

    def test_returns_list(self):
        conjecture = _make_linear_conjecture()
        profiles = _make_phase_profiles(5)
        result = design_validation_experiment(conjecture, profiles)
        assert isinstance(result, list)

    def test_target_fields(self):
        conjecture = _make_linear_conjecture()
        profiles = _make_phase_profiles(5)
        result = design_validation_experiment(conjecture, profiles)
        if result:
            target = result[0]
            assert "target_vector" in target
            assert "source_phase_id" in target
            assert "prediction_error" in target
            assert "priority" in target

    def test_target_vector_float64(self):
        conjecture = _make_linear_conjecture()
        profiles = _make_phase_profiles(5)
        result = design_validation_experiment(conjecture, profiles)
        if result:
            vec = result[0]["target_vector"]
            assert isinstance(vec, list)
            for v in vec:
                assert isinstance(v, float)

    def test_deterministic_across_calls(self):
        conjecture = _make_linear_conjecture()
        profiles = _make_phase_profiles(5)
        r1 = design_validation_experiment(conjecture, profiles)
        r2 = design_validation_experiment(conjecture, profiles)
        assert r1 == r2

    def test_num_targets_limit(self):
        conjecture = _make_linear_conjecture()
        profiles = _make_phase_profiles(10)
        result = design_validation_experiment(
            conjecture, profiles, num_targets=3
        )
        assert len(result) <= 3

    def test_empty_archive(self):
        conjecture = _make_linear_conjecture()
        result = design_validation_experiment(conjecture, [])
        assert result == []

    def test_priority_in_zero_one(self):
        conjecture = _make_linear_conjecture()
        profiles = _make_phase_profiles(5)
        result = design_validation_experiment(conjecture, profiles)
        for target in result:
            assert 0.0 <= target["priority"] <= 1.0

    def test_highest_error_first(self):
        conjecture = _make_linear_conjecture()
        profiles = _make_phase_profiles(5)
        result = design_validation_experiment(conjecture, profiles)
        if len(result) >= 2:
            for i in range(len(result) - 1):
                assert result[i]["prediction_error"] >= result[i + 1]["prediction_error"]


class TestDiscoveryEngineIntegration:
    """Tests for discovery engine integration with conjecture validation."""

    def test_conjecture_validation_disabled_by_default(self):
        from src.qec.discovery.discovery_engine import run_structure_discovery

        spec = {
            "num_variables": 10,
            "num_checks": 5,
            "variable_degree": 3,
            "check_degree": 6,
        }
        result = run_structure_discovery(
            spec, num_generations=2, population_size=4
        )
        assert "conjecture_validation_results" not in result
        assert "conjecture_counterexamples_phase" not in result
        assert "validation_experiment_targets" not in result

    def test_conjecture_validation_opt_in(self):
        from src.qec.discovery.discovery_engine import run_structure_discovery

        spec = {
            "num_variables": 10,
            "num_checks": 5,
            "variable_degree": 3,
            "check_degree": 6,
        }
        result = run_structure_discovery(
            spec,
            num_generations=2,
            population_size=4,
            base_seed=42,
            enable_conjecture_validation=True,
            conjecture_validation_interval=1,
            enable_phase_characterization=True,
            phase_characterization_interval=1,
            enable_theory_synthesis=True,
            theory_synthesis_interval=1,
        )
        assert "conjecture_validation_results" in result
        assert "conjecture_counterexamples_phase" in result
        assert "validation_experiment_targets" in result
        assert isinstance(result["conjecture_validation_results"], list)
        assert isinstance(result["conjecture_counterexamples_phase"], list)
        assert isinstance(result["validation_experiment_targets"], list)

    def test_generation_summary_fields_present(self):
        from src.qec.discovery.discovery_engine import run_structure_discovery

        spec = {
            "num_variables": 10,
            "num_checks": 5,
            "variable_degree": 3,
            "check_degree": 6,
        }
        result = run_structure_discovery(
            spec,
            num_generations=2,
            population_size=4,
            base_seed=42,
            enable_conjecture_validation=True,
            conjecture_validation_interval=1,
            enable_phase_characterization=True,
            phase_characterization_interval=1,
            enable_theory_synthesis=True,
            theory_synthesis_interval=1,
        )
        summary = result["generation_summaries"][-1]
        assert "num_conjectures_tested" in summary
        assert "num_counterexamples_found" in summary

    def test_generation_summary_fields_absent_when_disabled(self):
        from src.qec.discovery.discovery_engine import run_structure_discovery

        spec = {
            "num_variables": 10,
            "num_checks": 5,
            "variable_degree": 3,
            "check_degree": 6,
        }
        result = run_structure_discovery(
            spec, num_generations=2, population_size=4
        )
        summary = result["generation_summaries"][-1]
        assert "num_conjectures_tested" not in summary
        assert "num_counterexamples_found" not in summary
