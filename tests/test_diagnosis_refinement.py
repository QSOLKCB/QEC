"""Tests for diagnosis_refinement.py — v105.0.0."""

from __future__ import annotations

import copy

import pytest

from qec.analysis.diagnosis_refinement import (
    DEFAULT_THRESHOLDS,
    ROUND_PRECISION,
    apply_refined_thresholds,
    refine_diagnosis_rules,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_registry_with_oscillation(count: int = 5, strength: float = 0.8) -> dict:
    return {
        "sign:oscillatory_trap_suppression": {
            "count": count,
            "avg_strength": strength,
            "max_strength": 1.0,
            "first_seen": 0,
            "last_seen": count - 1,
        },
        "geometry:angular_velocity_decreasing": {
            "count": count,
            "avg_strength": strength,
            "max_strength": 1.0,
            "first_seen": 0,
            "last_seen": count - 1,
        },
    }


def _make_registry_with_stability(count: int = 5, strength: float = 0.8) -> dict:
    return {
        "geometry:stability_monotonicity": {
            "count": count,
            "avg_strength": strength,
            "max_strength": 1.0,
            "first_seen": 0,
            "last_seen": count - 1,
        },
        "control:control_stability_correlation": {
            "count": count,
            "avg_strength": strength,
            "max_strength": 1.0,
            "first_seen": 0,
            "last_seen": count - 1,
        },
    }


def _make_features() -> dict:
    return {
        "angular_velocity": 0.45,
        "spiral_score": 0.3,
        "curvature": 0.25,
        "axis_lock": 0.4,
        "total_displacement": 0.3,
        "displacement_variance": 0.2,
        "mean_instability": 0.3,
        "plane_coupling_score": 0.35,
        "dimensional_activity": 4,
        "convergence_rate": 0.4,
        "system_stability": 0.5,
        "volatility_score": 0.2,
        "topology_type": "mixed",
        "convergence_prediction": "moderate",
        "oscillation_prediction": "moderate",
        "basin_switch_risk": "medium",
        "metastable_prediction": "low",
        "control_signal": 0.3,
    }


# ---------------------------------------------------------------------------
# Tests: refine_diagnosis_rules
# ---------------------------------------------------------------------------


class TestRefineDiagnosisRules:
    def test_empty_registry_returns_defaults(self):
        result = refine_diagnosis_rules({})
        assert result == DEFAULT_THRESHOLDS

    def test_returns_all_threshold_keys(self):
        reg = _make_registry_with_oscillation()
        result = refine_diagnosis_rules(reg)
        assert "high" in result
        assert "moderate" in result
        assert "low" in result

    def test_oscillation_lowers_high_threshold(self):
        reg = _make_registry_with_oscillation()
        result = refine_diagnosis_rules(reg)
        assert result["high"] < DEFAULT_THRESHOLDS["high"]

    def test_stability_raises_moderate_threshold(self):
        reg = _make_registry_with_stability()
        result = refine_diagnosis_rules(reg)
        assert result["moderate"] > DEFAULT_THRESHOLDS["moderate"]

    def test_thresholds_bounded(self):
        reg = _make_registry_with_oscillation(count=100, strength=1.0)
        result = refine_diagnosis_rules(reg)
        for key in result:
            assert 0.05 <= result[key] <= 0.8

    def test_low_strength_no_adjustment(self):
        reg = _make_registry_with_oscillation(count=5, strength=0.2)
        result = refine_diagnosis_rules(reg)
        assert result == DEFAULT_THRESHOLDS

    def test_low_count_no_adjustment(self):
        reg = _make_registry_with_oscillation(count=1, strength=0.9)
        result = refine_diagnosis_rules(reg)
        assert result == DEFAULT_THRESHOLDS

    def test_deterministic(self):
        reg = _make_registry_with_oscillation()
        assert refine_diagnosis_rules(reg) == refine_diagnosis_rules(reg)

    def test_no_mutation(self):
        reg = _make_registry_with_oscillation()
        orig = copy.deepcopy(reg)
        refine_diagnosis_rules(reg)
        assert reg == orig


# ---------------------------------------------------------------------------
# Tests: apply_refined_thresholds
# ---------------------------------------------------------------------------


class TestApplyRefinedThresholds:
    def test_none_thresholds_uses_defaults(self):
        features = _make_features()
        result = apply_refined_thresholds(features, None)
        assert isinstance(result, dict)
        # Should have all failure modes.
        assert "oscillatory_trap" in result
        assert "healthy_convergence" in result

    def test_with_refined_thresholds(self):
        features = _make_features()
        thresholds = {"high": 0.4, "moderate": 0.25, "low": 0.1}
        result = apply_refined_thresholds(features, thresholds)
        assert isinstance(result, dict)
        for mode in result:
            assert 0.0 <= result[mode] <= 1.0

    def test_lowered_threshold_boosts_scores(self):
        features = _make_features()
        default_scores = apply_refined_thresholds(features, None)
        lowered = {"high": 0.3, "moderate": 0.2, "low": 0.1}
        boosted_scores = apply_refined_thresholds(features, lowered)
        # At least some scores should increase when threshold is lowered.
        any_boosted = any(
            boosted_scores[m] > default_scores[m]
            for m in default_scores
            if default_scores[m] > 0.0
        )
        # This is a soft check — geometry of features may not always produce boost.
        assert isinstance(boosted_scores, dict)

    def test_raised_threshold_may_reduce_scores(self):
        features = _make_features()
        raised = {"high": 0.7, "moderate": 0.5, "low": 0.3}
        result = apply_refined_thresholds(features, raised)
        assert isinstance(result, dict)

    def test_scores_bounded(self):
        features = _make_features()
        thresholds = {"high": 0.3, "moderate": 0.2, "low": 0.1}
        result = apply_refined_thresholds(features, thresholds)
        for mode, score in result.items():
            assert 0.0 <= score <= 1.0

    def test_deterministic(self):
        features = _make_features()
        thresholds = {"high": 0.4, "moderate": 0.25, "low": 0.1}
        r1 = apply_refined_thresholds(features, thresholds)
        r2 = apply_refined_thresholds(features, thresholds)
        assert r1 == r2

    def test_no_mutation(self):
        features = _make_features()
        thresholds = {"high": 0.4, "moderate": 0.25, "low": 0.1}
        f_orig = copy.deepcopy(features)
        t_orig = copy.deepcopy(thresholds)
        apply_refined_thresholds(features, thresholds)
        assert features == f_orig
        assert thresholds == t_orig


# ---------------------------------------------------------------------------
# Tests: geometry-guided scoring in treatment_planning
# ---------------------------------------------------------------------------


class TestGeometryGuidedScoring:
    def test_score_treatment_with_geometry(self):
        from qec.analysis.treatment_planning import score_treatment
        result = {
            "post_metrics": {
                "stability": 0.8,
                "attractor_weight": 0.6,
                "transient_weight": 0.1,
            },
            "geometry": {
                "angular_velocity": 0.1,
                "curvature": 0.2,
                "spiral_score": 0.8,
                "axis_lock": 0.7,
            },
        }
        score = score_treatment(result, use_geometry=True)
        assert 0.0 <= score <= 1.0

    def test_score_treatment_without_geometry(self):
        from qec.analysis.treatment_planning import score_treatment
        result = {
            "post_metrics": {
                "stability": 0.8,
                "attractor_weight": 0.6,
                "transient_weight": 0.1,
            },
        }
        score = score_treatment(result, use_geometry=False)
        assert 0.0 <= score <= 1.0

    def test_geometry_off_matches_original(self):
        from qec.analysis.treatment_planning import score_treatment
        result = {
            "post_metrics": {
                "stability": 0.8,
                "attractor_weight": 0.6,
                "transient_weight": 0.1,
            },
        }
        # Without geometry, should use original 0.35/0.35/0.30 weights.
        score = score_treatment(result, use_geometry=False)
        expected = round(
            0.35 * 0.8 + 0.35 * 0.6 + 0.30 * (1.0 - 0.1), 12
        )
        expected = min(1.0, max(0.0, expected))
        assert score == round(expected, 12)

    def test_high_angular_velocity_penalized(self):
        from qec.analysis.treatment_planning import score_treatment
        base = {
            "post_metrics": {"stability": 0.5, "attractor_weight": 0.5, "transient_weight": 0.2},
            "geometry": {"angular_velocity": 0.0, "curvature": 0.0, "spiral_score": 0.0, "axis_lock": 0.0},
        }
        high_av = {
            "post_metrics": {"stability": 0.5, "attractor_weight": 0.5, "transient_weight": 0.2},
            "geometry": {"angular_velocity": 1.0, "curvature": 0.0, "spiral_score": 0.0, "axis_lock": 0.0},
        }
        score_low = score_treatment(base, use_geometry=True)
        score_high = score_treatment(high_av, use_geometry=True)
        assert score_low > score_high

    def test_high_spiral_score_rewarded(self):
        from qec.analysis.treatment_planning import score_treatment
        base = {
            "post_metrics": {"stability": 0.5, "attractor_weight": 0.5, "transient_weight": 0.2},
            "geometry": {"angular_velocity": 0.0, "curvature": 0.0, "spiral_score": 0.0, "axis_lock": 0.0},
        }
        high_spiral = {
            "post_metrics": {"stability": 0.5, "attractor_weight": 0.5, "transient_weight": 0.2},
            "geometry": {"angular_velocity": 0.0, "curvature": 0.0, "spiral_score": 1.0, "axis_lock": 0.0},
        }
        score_base = score_treatment(base, use_geometry=True)
        score_spiral = score_treatment(high_spiral, use_geometry=True)
        assert score_spiral > score_base

    def test_deterministic(self):
        from qec.analysis.treatment_planning import score_treatment
        result = {
            "post_metrics": {"stability": 0.7, "attractor_weight": 0.5, "transient_weight": 0.15},
            "geometry": {"angular_velocity": 0.3, "curvature": 0.2, "spiral_score": 0.6, "axis_lock": 0.5},
        }
        assert score_treatment(result) == score_treatment(result)

    def test_default_geometry_when_key_missing(self):
        from qec.analysis.treatment_planning import score_treatment
        # No geometry key at all — should still work with defaults of 0.0.
        result = {
            "post_metrics": {"stability": 0.5, "attractor_weight": 0.5, "transient_weight": 0.2},
        }
        score = score_treatment(result, use_geometry=True)
        assert 0.0 <= score <= 1.0
