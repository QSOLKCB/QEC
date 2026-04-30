"""Tests for attractor basin analysis and regime transition detection (v98.7)."""

from __future__ import annotations

import copy
import importlib
import sys

import pytest

from qec.analysis.attractor_analysis import (
    analyze_attractors,
    analyze_trajectory,
    classify_regime,
    compute_basin_score,
    detect_transition,
    extract_signals,
)
from qec.experiments.metrics_probe import (
    generate_metric_sequences,
    run_trajectory_experiments,
)


# ---------------------------------------------------------------------------
# Import safety
# ---------------------------------------------------------------------------


def test_import_attractor_analysis_does_not_raise():
    """Importing attractor_analysis must succeed without optional deps."""
    mod = importlib.import_module("qec.analysis.attractor_analysis")
    assert hasattr(mod, "analyze_attractors")


def test_import_does_not_pull_cffi():
    """Importing attractor_analysis must not directly import cffi from QEC code."""
    before = set(sys.modules.keys())
    importlib.import_module("qec.analysis.attractor_analysis")
    after = set(sys.modules.keys())
    new_modules = after - before
    assert not any(
        m.startswith("_cffi_backend") or m.startswith("cffi")
        for m in new_modules
        if "qec" in repr(sys.modules.get(m, ""))
    )


# ---------------------------------------------------------------------------
# Helpers — build synthetic metrics dicts
# ---------------------------------------------------------------------------

def _make_metrics(
    phi: float = 0.5,
    consistency: float = 0.5,
    divergence: float = 0.1,
    abs_curvature: float = 0.1,
    curvature_variation: float = 0.1,
    resonance: float = 0.1,
    complexity: float = 0.1,
) -> dict:
    """Build a minimal metrics dict matching evaluate_metrics output shape."""
    return {
        "field": {
            "phi_alignment": phi,
            "curvature": {
                "abs_curvature": abs_curvature,
                "curvature_variation": curvature_variation,
            },
            "resonance": resonance,
            "complexity": complexity,
        },
        "multiscale": {
            "scale_consistency": consistency,
            "scale_divergence": divergence,
        },
    }


# ---------------------------------------------------------------------------
# extract_signals
# ---------------------------------------------------------------------------

class TestExtractSignals:
    def test_returns_all_keys(self):
        m = _make_metrics()
        s = extract_signals(m)
        expected_keys = {
            "phi", "consistency", "divergence",
            "curvature", "curvature_var", "resonance", "complexity",
        }
        assert set(s.keys()) == expected_keys

    def test_values_match_input(self):
        m = _make_metrics(phi=0.9, consistency=0.85, divergence=0.3)
        s = extract_signals(m)
        assert s["phi"] == pytest.approx(0.9)
        assert s["consistency"] == pytest.approx(0.85)
        assert s["divergence"] == pytest.approx(0.3)

    def test_no_mutation(self):
        m = _make_metrics()
        original = copy.deepcopy(m)
        extract_signals(m)
        assert m == original


# ---------------------------------------------------------------------------
# classify_regime
# ---------------------------------------------------------------------------

class TestClassifyRegime:
    def test_stable(self):
        s = extract_signals(_make_metrics(phi=0.9, consistency=0.9, abs_curvature=0.1))
        assert classify_regime(s) == "stable"

    def test_transitional(self):
        s = extract_signals(_make_metrics(divergence=0.5, consistency=0.6))
        assert classify_regime(s) == "transitional"

    def test_oscillatory(self):
        s = extract_signals(_make_metrics(resonance=0.7, curvature_variation=0.3))
        assert classify_regime(s) == "oscillatory"

    def test_unstable_curvature(self):
        s = extract_signals(_make_metrics(abs_curvature=0.5))
        assert classify_regime(s) == "unstable"

    def test_unstable_complexity(self):
        s = extract_signals(_make_metrics(complexity=0.7))
        assert classify_regime(s) == "unstable"

    def test_mixed_fallback(self):
        s = extract_signals(_make_metrics(
            phi=0.5, consistency=0.3, divergence=0.1,
            abs_curvature=0.1, curvature_variation=0.1,
            resonance=0.1, complexity=0.1,
        ))
        assert classify_regime(s) == "mixed"

    def test_priority_stable_over_oscillatory(self):
        """Stable check comes before oscillatory."""
        s = extract_signals(_make_metrics(
            phi=0.9, consistency=0.9, abs_curvature=0.1,
            resonance=0.8, curvature_variation=0.5,
        ))
        assert classify_regime(s) == "stable"

    def test_priority_transitional_over_unstable(self):
        """Transitional check comes before unstable."""
        s = extract_signals(_make_metrics(
            divergence=0.5, consistency=0.6, abs_curvature=0.5,
        ))
        assert classify_regime(s) == "transitional"


# ---------------------------------------------------------------------------
# compute_basin_score
# ---------------------------------------------------------------------------

class TestBasinScore:
    def test_range_clamped(self):
        # High positive signals
        s = extract_signals(_make_metrics(phi=1.0, consistency=1.0))
        score = compute_basin_score(s)
        assert 0.0 <= score <= 1.0

    def test_clamp_low(self):
        s = extract_signals(_make_metrics(
            phi=0.0, consistency=0.0,
            abs_curvature=10.0, divergence=10.0,
        ))
        assert compute_basin_score(s) == 0.0

    def test_known_value(self):
        s = {
            "phi": 0.5, "consistency": 0.5,
            "divergence": 0.1, "curvature": 0.1,
            "curvature_var": 0.0, "resonance": 0.0, "complexity": 0.0,
        }
        expected = 0.4 * 0.5 + 0.3 * 0.5 - 0.2 * 0.1 - 0.1 * 0.1
        assert compute_basin_score(s) == pytest.approx(expected)

    def test_deterministic(self):
        s = extract_signals(_make_metrics(phi=0.7, consistency=0.6))
        assert compute_basin_score(s) == compute_basin_score(s)


# ---------------------------------------------------------------------------
# detect_transition
# ---------------------------------------------------------------------------

class TestDetectTransition:
    def test_no_transition(self):
        s = extract_signals(_make_metrics(phi=0.9, consistency=0.9, abs_curvature=0.1))
        result = detect_transition(s, s)
        assert result["transition"] is False
        assert result["magnitude"] == pytest.approx(0.0)

    def test_regime_change_triggers(self):
        s1 = extract_signals(_make_metrics(phi=0.9, consistency=0.9, abs_curvature=0.1))
        s2 = extract_signals(_make_metrics(abs_curvature=0.5))
        result = detect_transition(s1, s2)
        assert result["transition"] is True

    def test_large_delta_triggers(self):
        s1 = extract_signals(_make_metrics(divergence=0.0, abs_curvature=0.0))
        s2 = extract_signals(_make_metrics(divergence=0.2, abs_curvature=0.2))
        result = detect_transition(s1, s2)
        assert result["transition"] is True
        assert result["magnitude"] == pytest.approx(0.4)

    def test_small_delta_no_trigger(self):
        s1 = extract_signals(_make_metrics(divergence=0.1, abs_curvature=0.1))
        s2 = extract_signals(_make_metrics(divergence=0.15, abs_curvature=0.1))
        # Same regime (both mixed), delta = 0.05
        result = detect_transition(s1, s2)
        assert result["transition"] is False


# ---------------------------------------------------------------------------
# analyze_trajectory
# ---------------------------------------------------------------------------

class TestAnalyzeTrajectory:
    def test_empty_sequence(self):
        result = analyze_trajectory([])
        assert result["regimes"] == []
        assert result["transitions"] == []

    def test_single_entry(self):
        result = analyze_trajectory([_make_metrics(phi=0.9, consistency=0.9, abs_curvature=0.1)])
        assert len(result["regimes"]) == 1
        assert result["regimes"][0] == "stable"
        assert result["transitions"] == []

    def test_detects_regime_changes(self):
        seq = [
            _make_metrics(phi=0.9, consistency=0.9, abs_curvature=0.1),  # stable
            _make_metrics(abs_curvature=0.5),  # unstable
        ]
        result = analyze_trajectory(seq)
        assert result["regimes"] == ["stable", "unstable"]
        assert len(result["transitions"]) == 1
        assert result["transitions"][0]["transition"] is True

    def test_stable_segments(self):
        seq = [
            _make_metrics(phi=0.9, consistency=0.9, abs_curvature=0.1),
            _make_metrics(abs_curvature=0.5),
            _make_metrics(phi=0.9, consistency=0.9, abs_curvature=0.1),
        ]
        result = analyze_trajectory(seq)
        assert result["stable_segments"] == [0, 2]

    def test_oscillation_flags(self):
        seq = [
            _make_metrics(resonance=0.7, curvature_variation=0.3),
            _make_metrics(phi=0.5, consistency=0.3),
        ]
        result = analyze_trajectory(seq)
        assert result["oscillation_flags"] == [True, False]


# ---------------------------------------------------------------------------
# analyze_attractors (master)
# ---------------------------------------------------------------------------

class TestAnalyzeAttractors:
    def test_returns_all_keys(self):
        m = _make_metrics()
        result = analyze_attractors(m)
        assert "signals" in result
        assert "regime" in result
        assert "basin_score" in result

    def test_deterministic(self):
        m = _make_metrics(phi=0.7, consistency=0.6, divergence=0.2)
        r1 = analyze_attractors(m)
        r2 = analyze_attractors(m)
        assert r1 == r2

    def test_no_mutation(self):
        m = _make_metrics()
        original = copy.deepcopy(m)
        analyze_attractors(m)
        assert m == original


# ---------------------------------------------------------------------------
# Trajectory experiments via metrics_probe sequences (v98.7)
# ---------------------------------------------------------------------------

class TestTrajectoryExperiments:
    """Exercise analyze_trajectory with structured metric sequences."""

    def test_stable_convergence_all_stable(self):
        """Stable convergence sequence stays in stable regime."""
        results = run_trajectory_experiments()
        r = [x for x in results if x["name"] == "stable_convergence"][0]
        assert all(reg == "stable" for reg in r["trajectory"]["regimes"])

    def test_regime_transition_detects_changes(self):
        """Regime transition sequence must detect transitions."""
        results = run_trajectory_experiments()
        r = [x for x in results if x["name"] == "regime_transition"][0]
        traj = r["trajectory"]
        assert len(set(traj["regimes"])) > 1
        assert any(t["transition"] for t in traj["transitions"])

    def test_oscillatory_switching_flags(self):
        """Oscillatory switching must flag oscillatory segments."""
        results = run_trajectory_experiments()
        r = [x for x in results if x["name"] == "oscillatory_switching"][0]
        assert any(r["trajectory"]["oscillation_flags"])

    def test_unstable_escalation_no_stable(self):
        """Unstable escalation must not contain stable segments."""
        results = run_trajectory_experiments()
        r = [x for x in results if x["name"] == "unstable_escalation"][0]
        assert len(r["trajectory"]["stable_segments"]) == 0

    def test_mixed_plateau_no_transitions(self):
        """Mixed plateau must show no transitions."""
        results = run_trajectory_experiments()
        r = [x for x in results if x["name"] == "mixed_plateau"][0]
        assert not any(t["transition"] for t in r["trajectory"]["transitions"])

    def test_trajectory_experiments_deterministic(self):
        """All trajectory experiments must be deterministic."""
        a = run_trajectory_experiments()
        b = run_trajectory_experiments()
        assert a == b

    def test_trajectory_no_mutation_of_sequences(self):
        """Running trajectory experiments must not mutate the sequences."""
        seqs_before = generate_metric_sequences()
        run_trajectory_experiments()
        seqs_after = generate_metric_sequences()
        assert seqs_before == seqs_after
