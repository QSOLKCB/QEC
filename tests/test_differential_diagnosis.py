"""Tests for differential diagnosis engine (v104.2.0).

Covers:
- diagnostic feature extraction
- rule-based scoring for each failure mode
- deterministic ranking
- explanation generation
- full pipeline
- formatter output
- integration with system diagnostics structure
- no mutation of inputs
- determinism guarantees
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

from qec.analysis.differential_diagnosis import (
    FAILURE_MODES,
    explain_diagnosis,
    extract_diagnostic_features,
    format_differential_diagnosis,
    rank_diagnoses,
    run_differential_diagnosis,
    score_failure_modes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    *,
    angular_velocity: float = 0.0,
    spiral_score: float = 0.0,
    curvature: float = 0.0,
    axis_lock: float = 0.0,
    total_displacement: float = 0.0,
    displacement_variance: float = 0.0,
    mean_instability: float = 0.0,
    plane_coupling_score: float = 0.0,
    dimensional_activity: int = 0,
    convergence: str = "low",
    oscillation: str = "low",
    basin_switch_risk: str = "low",
    metastable: str = "low",
    convergence_rate: float = 0.0,
    system_stability: float = 0.0,
    volatility_score: float = 0.0,
    topology_type: str = "unknown",
    control_signal: float = 0.0,
) -> Dict[str, Any]:
    """Build a system diagnostics result dict for testing."""
    return {
        "global_metrics": {
            "convergence_rate": convergence_rate,
            "system_stability": system_stability,
            "volatility_score": volatility_score,
            "topology_type": topology_type,
        },
        "trajectory_geometry": {
            "rotation_metrics": {
                "angular_velocity": angular_velocity,
                "spiral_score": spiral_score,
                "curvature": curvature,
                "axis_lock": axis_lock,
                "total_displacement": total_displacement,
                "displacement_variance": displacement_variance,
                "mean_instability": mean_instability,
                "control_signal": control_signal,
            },
            "coupling_metrics": {
                "plane_coupling_score": plane_coupling_score,
                "dimensional_activity": dimensional_activity,
            },
            "predictions": {
                "convergence": convergence,
                "oscillation": oscillation,
                "basin_switch_risk": basin_switch_risk,
                "metastable": metastable,
            },
        },
    }


# ---------------------------------------------------------------------------
# Feature Extraction Tests
# ---------------------------------------------------------------------------


class TestExtractDiagnosticFeatures:
    def test_extracts_all_expected_keys(self) -> None:
        result = _make_result(angular_velocity=0.7, spiral_score=0.3)
        features = extract_diagnostic_features(result)
        expected_keys = {
            "angular_velocity", "spiral_score", "curvature", "axis_lock",
            "total_displacement", "displacement_variance", "mean_instability",
            "plane_coupling_score", "dimensional_activity",
            "convergence_rate", "system_stability", "volatility_score",
            "topology_type", "convergence_prediction", "oscillation_prediction",
            "basin_switch_risk", "metastable_prediction", "control_signal",
        }
        assert set(features.keys()) == expected_keys

    def test_defaults_on_empty_input(self) -> None:
        features = extract_diagnostic_features({})
        assert features["angular_velocity"] == 0.0
        assert features["topology_type"] == "unknown"
        assert features["convergence_prediction"] == "unknown"

    def test_preserves_values(self) -> None:
        result = _make_result(
            angular_velocity=0.82,
            convergence_rate=0.21,
            system_stability=0.65,
        )
        features = extract_diagnostic_features(result)
        assert features["angular_velocity"] == 0.82
        assert features["convergence_rate"] == 0.21
        assert features["system_stability"] == 0.65

    def test_no_mutation(self) -> None:
        result = _make_result(angular_velocity=0.5)
        original = copy.deepcopy(result)
        extract_diagnostic_features(result)
        assert result == original


# ---------------------------------------------------------------------------
# Scoring Tests
# ---------------------------------------------------------------------------


class TestScoreFailureModes:
    def test_returns_all_failure_modes(self) -> None:
        features = extract_diagnostic_features(_make_result())
        scores = score_failure_modes(features)
        assert set(scores.keys()) == set(FAILURE_MODES)

    def test_scores_in_range(self) -> None:
        features = extract_diagnostic_features(_make_result(
            angular_velocity=0.8, curvature=0.6, system_stability=0.5,
        ))
        scores = score_failure_modes(features)
        for mode, score in scores.items():
            assert 0.0 <= score <= 1.0, f"{mode} score {score} out of range"

    def test_oscillatory_trap_scores_high(self) -> None:
        """High angular velocity + low convergence -> oscillatory trap."""
        features = extract_diagnostic_features(_make_result(
            angular_velocity=0.8,
            convergence_rate=0.05,
            system_stability=0.6,
            oscillation="high",
        ))
        scores = score_failure_modes(features)
        assert scores["oscillatory_trap"] >= 0.7

    def test_metastable_plateau_scores_high(self) -> None:
        """Low displacement + high stability + low curvature -> metastable."""
        features = extract_diagnostic_features(_make_result(
            total_displacement=0.02,
            system_stability=0.8,
            curvature=0.05,
            metastable="likely",
        ))
        scores = score_failure_modes(features)
        assert scores["metastable_plateau"] >= 0.7

    def test_basin_switch_scores_high(self) -> None:
        """High curvature + high coupling + basin risk -> basin switch."""
        features = extract_diagnostic_features(_make_result(
            curvature=0.7,
            plane_coupling_score=0.6,
            system_stability=0.4,
            basin_switch_risk="high",
        ))
        scores = score_failure_modes(features)
        assert scores["basin_switch_instability"] >= 0.7

    def test_healthy_convergence_scores_high(self) -> None:
        """High spiral + high convergence + high stability -> healthy."""
        features = extract_diagnostic_features(_make_result(
            spiral_score=0.8,
            convergence_rate=0.9,
            system_stability=0.85,
            angular_velocity=0.05,
            convergence="likely",
        ))
        scores = score_failure_modes(features)
        assert scores["healthy_convergence"] >= 0.7

    def test_control_overshoot_scores_high(self) -> None:
        """High control + high curvature + oscillatory -> overshoot."""
        features = extract_diagnostic_features(_make_result(
            control_signal=0.7,
            curvature=0.6,
            angular_velocity=0.6,
            volatility_score=0.5,
            displacement_variance=0.4,
        ))
        scores = score_failure_modes(features)
        assert scores["control_overshoot"] >= 0.7

    def test_underconstrained_scores_high(self) -> None:
        """Low coupling + low axis_lock + high dim activity -> underconstrained."""
        features = extract_diagnostic_features(_make_result(
            plane_coupling_score=0.05,
            axis_lock=0.05,
            dimensional_activity=7,
            system_stability=0.05,
            convergence_rate=0.05,
        ))
        scores = score_failure_modes(features)
        assert scores["underconstrained_dynamics"] >= 0.7

    def test_determinism(self) -> None:
        """Same input -> same scores."""
        features = extract_diagnostic_features(_make_result(
            angular_velocity=0.6, curvature=0.4, system_stability=0.5,
        ))
        scores1 = score_failure_modes(features)
        scores2 = score_failure_modes(features)
        assert scores1 == scores2


# ---------------------------------------------------------------------------
# Ranking Tests
# ---------------------------------------------------------------------------


class TestRankDiagnoses:
    def test_sorted_by_score_desc(self) -> None:
        scores = {
            "oscillatory_trap": 0.8,
            "metastable_plateau": 0.3,
            "healthy_convergence": 0.6,
        }
        ranked = rank_diagnoses(scores)
        assert ranked[0] == ("oscillatory_trap", 0.8)
        assert ranked[1] == ("healthy_convergence", 0.6)
        assert ranked[2] == ("metastable_plateau", 0.3)

    def test_ties_broken_by_name_asc(self) -> None:
        scores = {
            "basin_switch_instability": 0.5,
            "control_overshoot": 0.5,
            "oscillatory_trap": 0.5,
        }
        ranked = rank_diagnoses(scores)
        names = [name for name, _ in ranked]
        assert names == [
            "basin_switch_instability",
            "control_overshoot",
            "oscillatory_trap",
        ]

    def test_empty_scores(self) -> None:
        ranked = rank_diagnoses({})
        assert ranked == []

    def test_determinism(self) -> None:
        scores = {"a": 0.5, "b": 0.5, "c": 0.3}
        r1 = rank_diagnoses(scores)
        r2 = rank_diagnoses(scores)
        assert r1 == r2


# ---------------------------------------------------------------------------
# Explanation Tests
# ---------------------------------------------------------------------------


class TestExplainDiagnosis:
    def test_oscillatory_trap_explanation(self) -> None:
        features = extract_diagnostic_features(_make_result(
            angular_velocity=0.8, convergence_rate=0.1, system_stability=0.6,
        ))
        explanation = explain_diagnosis(features, "oscillatory_trap")
        assert "angular velocity" in explanation
        assert "0.80" in explanation
        assert "rotation" in explanation.lower()

    def test_healthy_convergence_explanation(self) -> None:
        features = extract_diagnostic_features(_make_result(
            spiral_score=0.7, convergence_rate=0.9, system_stability=0.8,
        ))
        explanation = explain_diagnosis(features, "healthy_convergence")
        assert "spiral score" in explanation
        assert "convergence" in explanation.lower()

    def test_unknown_diagnosis(self) -> None:
        features = extract_diagnostic_features(_make_result())
        explanation = explain_diagnosis(features, "nonexistent_mode")
        assert "Unknown" in explanation

    def test_explanation_is_string(self) -> None:
        features = extract_diagnostic_features(_make_result(
            curvature=0.6, plane_coupling_score=0.5,
        ))
        explanation = explain_diagnosis(features, "basin_switch_instability")
        assert isinstance(explanation, str)
        assert len(explanation) > 0


# ---------------------------------------------------------------------------
# Full Pipeline Tests
# ---------------------------------------------------------------------------


class TestRunDifferentialDiagnosis:
    def test_returns_expected_keys(self) -> None:
        result = _make_result(angular_velocity=0.7, convergence_rate=0.1)
        diagnosis = run_differential_diagnosis(result)
        assert "features" in diagnosis
        assert "scores" in diagnosis
        assert "ranked" in diagnosis
        assert "primary_diagnosis" in diagnosis
        assert "diagnosis_confidence" in diagnosis
        assert "explanations" in diagnosis

    def test_primary_diagnosis_is_top_ranked(self) -> None:
        result = _make_result(
            angular_velocity=0.8,
            convergence_rate=0.05,
            system_stability=0.6,
            oscillation="high",
        )
        diagnosis = run_differential_diagnosis(result)
        ranked = diagnosis["ranked"]
        assert diagnosis["primary_diagnosis"] == ranked[0][0]
        assert diagnosis["diagnosis_confidence"] == ranked[0][1]

    def test_empty_input(self) -> None:
        diagnosis = run_differential_diagnosis({})
        assert diagnosis["primary_diagnosis"] in FAILURE_MODES
        assert 0.0 <= diagnosis["diagnosis_confidence"] <= 1.0

    def test_no_mutation(self) -> None:
        result = _make_result(angular_velocity=0.5, curvature=0.3)
        original = copy.deepcopy(result)
        run_differential_diagnosis(result)
        assert result == original

    def test_determinism(self) -> None:
        result = _make_result(
            angular_velocity=0.6,
            curvature=0.4,
            system_stability=0.5,
            plane_coupling_score=0.3,
        )
        d1 = run_differential_diagnosis(result)
        d2 = run_differential_diagnosis(result)
        assert d1 == d2

    def test_explanations_only_for_nonzero_scores(self) -> None:
        result = _make_result()
        diagnosis = run_differential_diagnosis(result)
        for name in diagnosis["explanations"]:
            assert diagnosis["scores"][name] > 0.0


# ---------------------------------------------------------------------------
# Formatter Tests
# ---------------------------------------------------------------------------


class TestFormatDifferentialDiagnosis:
    def test_contains_header(self) -> None:
        result = _make_result(angular_velocity=0.7, convergence_rate=0.1)
        diagnosis = run_differential_diagnosis(result)
        output = format_differential_diagnosis(diagnosis)
        assert "=== Differential Diagnosis ===" in output

    def test_contains_primary_diagnosis(self) -> None:
        result = _make_result(
            angular_velocity=0.8,
            convergence_rate=0.05,
            system_stability=0.6,
            oscillation="high",
        )
        diagnosis = run_differential_diagnosis(result)
        output = format_differential_diagnosis(diagnosis)
        assert "Primary Diagnosis:" in output

    def test_contains_explanation(self) -> None:
        result = _make_result(
            angular_velocity=0.8,
            convergence_rate=0.05,
            system_stability=0.6,
        )
        diagnosis = run_differential_diagnosis(result)
        output = format_differential_diagnosis(diagnosis)
        assert "Explanation:" in output

    def test_ranked_entries_formatted(self) -> None:
        result = _make_result(
            angular_velocity=0.8,
            convergence_rate=0.05,
            system_stability=0.6,
        )
        diagnosis = run_differential_diagnosis(result)
        output = format_differential_diagnosis(diagnosis)
        # Should contain numbered entries.
        assert "1." in output

    def test_output_is_string(self) -> None:
        diagnosis = run_differential_diagnosis({})
        output = format_differential_diagnosis(diagnosis)
        assert isinstance(output, str)


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_oscillatory_scenario_primary(self) -> None:
        """Full pipeline: oscillatory input -> oscillatory_trap primary."""
        result = _make_result(
            angular_velocity=0.9,
            convergence_rate=0.02,
            system_stability=0.7,
            oscillation="high",
            curvature=0.1,
            spiral_score=0.05,
        )
        diagnosis = run_differential_diagnosis(result)
        assert diagnosis["primary_diagnosis"] == "oscillatory_trap"

    def test_healthy_scenario_primary(self) -> None:
        """Full pipeline: healthy input -> healthy_convergence primary."""
        result = _make_result(
            spiral_score=0.8,
            convergence_rate=0.95,
            system_stability=0.9,
            angular_velocity=0.02,
            convergence="likely",
        )
        diagnosis = run_differential_diagnosis(result)
        assert diagnosis["primary_diagnosis"] == "healthy_convergence"

    def test_metastable_scenario_primary(self) -> None:
        """Full pipeline: metastable input -> metastable_plateau primary."""
        result = _make_result(
            total_displacement=0.01,
            system_stability=0.85,
            curvature=0.02,
            metastable="likely",
            convergence_rate=0.0,
            angular_velocity=0.0,
            spiral_score=0.0,
        )
        diagnosis = run_differential_diagnosis(result)
        assert diagnosis["primary_diagnosis"] == "metastable_plateau"

    def test_all_failure_modes_in_scores(self) -> None:
        """All seven failure modes appear in every diagnosis."""
        result = _make_result()
        diagnosis = run_differential_diagnosis(result)
        for mode in FAILURE_MODES:
            assert mode in diagnosis["scores"]

    def test_ranked_length_matches_failure_modes(self) -> None:
        result = _make_result()
        diagnosis = run_differential_diagnosis(result)
        assert len(diagnosis["ranked"]) == len(FAILURE_MODES)
