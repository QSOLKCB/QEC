"""Tests for multi-state analysis (v102.8.0).

Verifies:
- ternary classification thresholds
- deterministic outputs
- membership normalization
- edge cases (all neutral, strong attractor only, mixed states)
- state vector composition
- integration via run_multistate_analysis
"""

from __future__ import annotations

import copy

from qec.analysis.multistate import (
    PHASE_WEIGHTS,
    ROUND_PRECISION,
    STABILITY_HIGH,
    STABILITY_LOW,
    TREND_THRESHOLD,
    build_state_vector,
    classify_ternary,
    compute_multistate,
    compute_phase_membership,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run(strategies):
    """Build a run dict from a list of (name, design_score) tuples."""
    return {
        "strategies": [
            {
                "name": name,
                "metrics": {
                    "design_score": score,
                    "confidence_efficiency": 0.5,
                    "consistency_gap": 0.1,
                    "revival_strength": 0.0,
                },
            }
            for name, score in strategies
        ],
    }


# ---------------------------------------------------------------------------
# classify_ternary tests
# ---------------------------------------------------------------------------


class TestClassifyTernary:
    """Tests for classify_ternary."""

    def test_positive_trend(self):
        result = classify_ternary({"trend": 0.1, "stability": 0.9, "phase": "neutral"})
        assert result["trend_state"] == 1

    def test_negative_trend(self):
        result = classify_ternary({"trend": -0.1, "stability": 0.9, "phase": "neutral"})
        assert result["trend_state"] == -1

    def test_neutral_trend(self):
        result = classify_ternary({"trend": 0.03, "stability": 0.9, "phase": "neutral"})
        assert result["trend_state"] == 0

    def test_trend_at_positive_boundary(self):
        result = classify_ternary({"trend": TREND_THRESHOLD, "stability": 0.5, "phase": "neutral"})
        assert result["trend_state"] == 0

    def test_trend_just_above_threshold(self):
        result = classify_ternary({"trend": TREND_THRESHOLD + 1e-9, "stability": 0.5, "phase": "neutral"})
        assert result["trend_state"] == 1

    def test_trend_at_negative_boundary(self):
        result = classify_ternary({"trend": -TREND_THRESHOLD, "stability": 0.5, "phase": "neutral"})
        assert result["trend_state"] == 0

    def test_high_stability(self):
        result = classify_ternary({"trend": 0.0, "stability": 0.9, "phase": "neutral"})
        assert result["stability_state"] == 1

    def test_low_stability(self):
        result = classify_ternary({"trend": 0.0, "stability": 0.3, "phase": "neutral"})
        assert result["stability_state"] == -1

    def test_mid_stability(self):
        result = classify_ternary({"trend": 0.0, "stability": 0.65, "phase": "neutral"})
        assert result["stability_state"] == 0

    def test_stability_at_high_boundary(self):
        result = classify_ternary({"trend": 0.0, "stability": STABILITY_HIGH, "phase": "neutral"})
        assert result["stability_state"] == 0

    def test_stability_at_low_boundary(self):
        result = classify_ternary({"trend": 0.0, "stability": STABILITY_LOW, "phase": "neutral"})
        assert result["stability_state"] == 0

    def test_stability_below_low(self):
        result = classify_ternary({"trend": 0.0, "stability": STABILITY_LOW - 0.01, "phase": "neutral"})
        assert result["stability_state"] == -1

    def test_phase_strong_attractor(self):
        result = classify_ternary({"trend": 0.0, "stability": 0.5, "phase": "strong_attractor"})
        assert result["phase_state"] == 1

    def test_phase_weak_attractor(self):
        result = classify_ternary({"trend": 0.0, "stability": 0.5, "phase": "weak_attractor"})
        assert result["phase_state"] == 1

    def test_phase_basin(self):
        result = classify_ternary({"trend": 0.0, "stability": 0.5, "phase": "basin"})
        assert result["phase_state"] == 0

    def test_phase_neutral(self):
        result = classify_ternary({"trend": 0.0, "stability": 0.5, "phase": "neutral"})
        assert result["phase_state"] == 0

    def test_phase_transient(self):
        result = classify_ternary({"trend": 0.0, "stability": 0.5, "phase": "transient"})
        assert result["phase_state"] == -1

    def test_empty_metrics(self):
        result = classify_ternary({})
        assert result["trend_state"] == 0
        assert result["stability_state"] == -1
        assert result["phase_state"] == 0

    def test_output_keys(self):
        result = classify_ternary({"trend": 0.0, "stability": 0.5, "phase": "neutral"})
        assert set(result.keys()) == {"trend_state", "stability_state", "phase_state"}

    def test_deterministic(self):
        metrics = {"trend": 0.03, "stability": 0.75, "phase": "basin"}
        results = [classify_ternary(metrics) for _ in range(50)]
        assert all(r == results[0] for r in results)

    def test_no_mutation(self):
        metrics = {"trend": 0.1, "stability": 0.9, "phase": "strong_attractor"}
        original = copy.deepcopy(metrics)
        classify_ternary(metrics)
        assert metrics == original


# ---------------------------------------------------------------------------
# compute_phase_membership tests
# ---------------------------------------------------------------------------


class TestComputePhaseMembership:
    """Tests for compute_phase_membership."""

    def test_empty_input(self):
        result = compute_phase_membership({})
        assert result == {}

    def test_all_neutral(self):
        phase_result = {
            "a": {"phase": "neutral"},
            "b": {"phase": "neutral"},
        }
        result = compute_phase_membership(phase_result)
        assert "neutral" in result
        # neutral weight is 0.0, so all values should be near 0
        assert result["neutral"] == round(0.0 / (0.0 + 1e-12), ROUND_PRECISION)

    def test_strong_attractor_only(self):
        phase_result = {
            "a": {"phase": "strong_attractor"},
        }
        result = compute_phase_membership(phase_result)
        assert "strong_attractor" in result
        expected = round(1.0 / (1.0 + 1e-12), ROUND_PRECISION)
        assert result["strong_attractor"] == expected

    def test_mixed_states(self):
        phase_result = {
            "a": {"phase": "strong_attractor"},
            "b": {"phase": "basin"},
            "c": {"phase": "transient"},
        }
        result = compute_phase_membership(phase_result)
        assert "strong_attractor" in result
        assert "basin" in result
        assert "transient" in result
        # Weights: strong_attractor=1.0, basin=0.5, transient=0.2
        total = 1.0 + 0.5 + 0.2 + 1e-12
        assert result["strong_attractor"] == round(1.0 / total, ROUND_PRECISION)
        assert result["basin"] == round(0.5 / total, ROUND_PRECISION)
        assert result["transient"] == round(0.2 / total, ROUND_PRECISION)

    def test_normalization_sums_near_one(self):
        phase_result = {
            "a": {"phase": "strong_attractor"},
            "b": {"phase": "weak_attractor"},
            "c": {"phase": "basin"},
        }
        result = compute_phase_membership(phase_result)
        total = sum(result.values())
        # Should be very close to 1.0 (epsilon from normalization).
        assert abs(total - 1.0) < 1e-6

    def test_duplicate_phases_accumulate(self):
        phase_result = {
            "a": {"phase": "basin"},
            "b": {"phase": "basin"},
        }
        result = compute_phase_membership(phase_result)
        # Two basins: weight = 0.5 + 0.5 = 1.0
        expected = round(1.0 / (1.0 + 1e-12), ROUND_PRECISION)
        assert result["basin"] == expected

    def test_deterministic(self):
        phase_result = {
            "x": {"phase": "strong_attractor"},
            "y": {"phase": "transient"},
        }
        results = [compute_phase_membership(phase_result) for _ in range(50)]
        assert all(r == results[0] for r in results)

    def test_no_mutation(self):
        phase_result = {"a": {"phase": "basin"}}
        original = copy.deepcopy(phase_result)
        compute_phase_membership(phase_result)
        assert phase_result == original

    def test_unknown_phase_treated_as_zero(self):
        phase_result = {"a": {"phase": "unknown_phase"}}
        result = compute_phase_membership(phase_result)
        assert "unknown_phase" in result
        # Weight 0.0 -> normalized near 0
        assert result["unknown_phase"] == round(0.0 / (0.0 + 1e-12), ROUND_PRECISION)


# ---------------------------------------------------------------------------
# build_state_vector tests
# ---------------------------------------------------------------------------


class TestBuildStateVector:
    """Tests for build_state_vector."""

    def test_basic(self):
        ternary = {"trend_state": 1, "stability_state": 0, "phase_state": -1}
        membership = {"strong_attractor": 0.6, "basin": 0.4}
        result = build_state_vector(ternary, membership)
        assert result["ternary"] == ternary
        assert result["membership"] == membership

    def test_returns_copies(self):
        ternary = {"trend_state": 1, "stability_state": 0, "phase_state": -1}
        membership = {"strong_attractor": 0.6}
        result = build_state_vector(ternary, membership)
        # Modifying result should not affect originals.
        result["ternary"]["trend_state"] = 99
        assert ternary["trend_state"] == 1

    def test_output_keys(self):
        result = build_state_vector({}, {})
        assert set(result.keys()) == {"ternary", "membership"}


# ---------------------------------------------------------------------------
# compute_multistate tests
# ---------------------------------------------------------------------------


class TestComputeMultistate:
    """Tests for compute_multistate."""

    def test_empty_runs(self):
        result = compute_multistate([])
        assert result == {}

    def test_basic_pipeline(self):
        runs = [
            _make_run([("alpha", 0.8), ("beta", 0.3)]),
            _make_run([("alpha", 0.85), ("beta", 0.25)]),
            _make_run([("alpha", 0.9), ("beta", 0.2)]),
        ]
        result = compute_multistate(runs)
        # Should have entries for strategy types from taxonomy.
        assert isinstance(result, dict)
        for name, sv in result.items():
            assert "ternary" in sv
            assert "membership" in sv
            ternary = sv["ternary"]
            assert ternary["trend_state"] in (-1, 0, 1)
            assert ternary["stability_state"] in (-1, 0, 1)
            assert ternary["phase_state"] in (-1, 0, 1)

    def test_deterministic(self):
        runs = [
            _make_run([("alpha", 0.8), ("beta", 0.3)]),
            _make_run([("alpha", 0.85), ("beta", 0.25)]),
        ]
        results = [compute_multistate(runs) for _ in range(50)]
        assert all(r == results[0] for r in results)

    def test_no_mutation(self):
        runs = [
            _make_run([("alpha", 0.8)]),
            _make_run([("alpha", 0.85)]),
        ]
        original = copy.deepcopy(runs)
        compute_multistate(runs)
        assert runs == original

    def test_reuses_precomputed_results(self):
        runs = [
            _make_run([("alpha", 0.8)]),
            _make_run([("alpha", 0.85)]),
        ]
        from qec.analysis.strategy_adapter import (
            run_phase_space_analysis,
            run_trajectory_analysis,
        )

        traj = run_trajectory_analysis(runs)
        phase = run_phase_space_analysis(runs)

        result_with = compute_multistate(
            runs, trajectory_result=traj, phase_space_result=phase,
        )
        result_without = compute_multistate(runs)
        assert result_with == result_without


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestMultistateIntegration:
    """Integration tests via strategy_adapter."""

    def test_run_multistate_analysis(self):
        from qec.analysis.strategy_adapter import run_multistate_analysis

        runs = [
            _make_run([("alpha", 0.8), ("beta", 0.3)]),
            _make_run([("alpha", 0.85), ("beta", 0.25)]),
            _make_run([("alpha", 0.9), ("beta", 0.2)]),
        ]
        result = run_multistate_analysis(runs)
        assert "multistate" in result
        multistate = result["multistate"]
        assert isinstance(multistate, dict)
        for name, sv in multistate.items():
            assert "ternary" in sv
            assert "membership" in sv

    def test_format_multistate_summary(self):
        from qec.analysis.strategy_adapter import format_multistate_summary

        result = {
            "multistate": {
                "alpha": {
                    "ternary": {"trend_state": 1, "stability_state": 0, "phase_state": 1},
                    "membership": {"strong_attractor": 0.6, "basin": 0.3, "transient": 0.1},
                },
            },
        }
        summary = format_multistate_summary(result)
        assert "=== Multi-State Analysis ===" in summary
        assert "Strategy: alpha" in summary
        assert "trend=+1" in summary
        assert "stability=0" in summary
        assert "phase=+1" in summary
        assert "strong_attractor: 0.6" in summary

    def test_format_empty(self):
        from qec.analysis.strategy_adapter import format_multistate_summary

        summary = format_multistate_summary({"multistate": {}})
        assert "=== Multi-State Analysis ===" in summary

    def test_reuses_phase_space_result(self):
        from qec.analysis.strategy_adapter import (
            run_multistate_analysis,
            run_phase_space_analysis,
        )

        runs = [
            _make_run([("alpha", 0.8)]),
            _make_run([("alpha", 0.85)]),
        ]
        phase = run_phase_space_analysis(runs)
        result = run_multistate_analysis(runs, phase_space_result=phase)
        result2 = run_multistate_analysis(runs)
        assert result == result2
