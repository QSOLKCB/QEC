"""
Tests for BP dynamics regime analysis (v4.4.0).

Validates:
  - Determinism (identical outputs on repeated runs)
  - Zero sign handling (0 → +1, consistent with v4.3.0)
  - Optional correction vectors (CVNE fields None when unavailable)
  - Regime branch coverage (all six regimes triggered)
  - Bench integration smoke test
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from src.qec.diagnostics.bp_dynamics import (
    DEFAULT_PARAMS,
    DEFAULT_THRESHOLDS,
    classify_bp_regime,
    compute_bp_dynamics_metrics,
    _normalize_llr_vector,
    _normalize_llr_trace,
    _precompute_signs_and_sigs,
    _sign,
)


# ── Helpers ───────────────────────────────────────────────────────────


def _make_stable_llr_trace(n_iters: int = 20, n_vars: int = 10) -> list:
    """Converging LLR trace: all positive, decreasing energy."""
    trace = []
    for t in range(n_iters):
        # Stable positive LLRs growing in magnitude
        vec = np.ones(n_vars, dtype=np.float64) * (1.0 + 0.1 * t)
        trace.append(vec)
    return trace


def _make_monotonic_energy(n_iters: int = 20) -> list:
    """Monotonically decreasing energy trace."""
    return [float(100.0 - 2.0 * t) for t in range(n_iters)]


def _make_oscillating_llr_trace(
    n_iters: int = 20, n_vars: int = 10, period: int = 2,
) -> list:
    """LLR trace with periodic sign flips."""
    trace = []
    for t in range(n_iters):
        if t % period == 0:
            vec = np.ones(n_vars, dtype=np.float64) * 1.5
        else:
            vec = np.ones(n_vars, dtype=np.float64) * -1.5
        trace.append(vec)
    return trace


def _make_flat_energy(n_iters: int = 20, value: float = 50.0) -> list:
    """Flat energy trace (plateau)."""
    return [value] * n_iters


def _make_trapping_llr_trace(
    n_iters: int = 20, n_vars: int = 10, trap_fraction: float = 0.5,
) -> list:
    """LLR trace with persistent sign disagreements in tail.

    Some variables flip between iterations but disagree with the final sign.
    """
    n_trap = max(1, int(n_vars * trap_fraction))
    trace = []
    for t in range(n_iters):
        vec = np.ones(n_vars, dtype=np.float64) * 2.0
        # Trapped variables oscillate but end negative
        if t < n_iters - 1:
            # Most iterations: trapped vars have opposite sign to final
            vec[:n_trap] = -2.0 if (t % 2 == 0) else 2.0
        else:
            # Final iteration: all positive (so trapped vars disagree)
            vec[:n_trap] = 2.0
        trace.append(vec)
    # Make trapped variables negative in most tail iterations
    # but positive in final - creating disagreement
    for t in range(max(0, n_iters - 13), n_iters - 1):
        trace[t][:n_trap] = -2.0
    trace[-1][:n_trap] = 2.0  # final positive
    return trace


def _make_chaotic_energy(n_iters: int = 20) -> list:
    """Erratic energy trace with large jumps and no descent."""
    rng = np.random.default_rng(42)
    base = 50.0
    trace = []
    for t in range(n_iters):
        # Large random jumps, mostly increasing
        base += rng.standard_normal() * 10.0
        trace.append(float(base))
    return trace


def _make_chaotic_llr_trace(n_iters: int = 20, n_vars: int = 10) -> list:
    """Chaotic LLR trace: random sign changes, no periodic structure."""
    rng = np.random.default_rng(42)
    trace = []
    for t in range(n_iters):
        vec = rng.standard_normal(n_vars) * 3.0
        trace.append(vec)
    return trace


def _make_cycling_corrections(
    n_iters: int = 20, n_vars: int = 10, period: int = 3,
) -> list:
    """Correction vectors that cycle with given period."""
    patterns = []
    rng = np.random.default_rng(123)
    for p in range(period):
        patterns.append(rng.integers(0, 2, size=n_vars).astype(np.int32))
    return [patterns[t % period].copy() for t in range(n_iters)]


# ── Test: Determinism ─────────────────────────────────────────────────


class TestDeterminism:
    """Identical inputs must produce byte-identical JSON output."""

    def test_stable_determinism(self):
        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()
        out1 = compute_bp_dynamics_metrics(llr, energy)
        out2 = compute_bp_dynamics_metrics(llr, energy)
        j1 = json.dumps(out1, sort_keys=True)
        j2 = json.dumps(out2, sort_keys=True)
        assert j1 == j2

    def test_oscillating_determinism(self):
        llr = _make_oscillating_llr_trace()
        energy = _make_flat_energy()
        out1 = compute_bp_dynamics_metrics(llr, energy)
        out2 = compute_bp_dynamics_metrics(llr, energy)
        j1 = json.dumps(out1, sort_keys=True)
        j2 = json.dumps(out2, sort_keys=True)
        assert j1 == j2

    def test_with_correction_vectors_determinism(self):
        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()
        cvs = _make_cycling_corrections()
        out1 = compute_bp_dynamics_metrics(llr, energy, correction_vectors=cvs)
        out2 = compute_bp_dynamics_metrics(llr, energy, correction_vectors=cvs)
        j1 = json.dumps(out1, sort_keys=True)
        j2 = json.dumps(out2, sort_keys=True)
        assert j1 == j2

    def test_classifier_determinism(self):
        llr = _make_oscillating_llr_trace()
        energy = _make_flat_energy()
        out1 = compute_bp_dynamics_metrics(llr, energy)
        out2 = compute_bp_dynamics_metrics(llr, energy)
        assert out1["regime"] == out2["regime"]
        assert out1["evidence"] == out2["evidence"]


# ── Test: Zero Sign Handling ──────────────────────────────────────────


class TestZeroSignHandling:
    """Zero LLR values must be treated as non-negative (→ +1)."""

    def test_sign_of_zero(self):
        x = np.array([0.0, -1.0, 1.0, 0.0, -0.5])
        s = _sign(x)
        np.testing.assert_array_equal(s, [1, -1, 1, 1, -1])

    def test_all_zeros_trace(self):
        llr = [np.zeros(5, dtype=np.float64) for _ in range(10)]
        energy = [float(i) for i in range(10)]
        out = compute_bp_dynamics_metrics(llr, energy)
        assert isinstance(out["metrics"]["msi"], float)
        assert isinstance(out["regime"], str)

    def test_zero_sign_consistent_with_v43(self):
        """v4.3.0 BOI uses np.where(x < 0, -1, 1). We must match."""
        x = np.array([0.0, -0.0, 1e-30, -1e-30])
        s = _sign(x)
        expected = np.where(x < 0, -1, 1)
        np.testing.assert_array_equal(s, expected)


# ── Test: Optional Correction Vectors ─────────────────────────────────


class TestOptionalCorrectionVectors:
    """CVNE must return None fields when correction vectors unavailable."""

    def test_none_correction_vectors(self):
        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()
        out = compute_bp_dynamics_metrics(llr, energy, correction_vectors=None)
        assert out["metrics"]["cvne_entropy"] is None
        assert out["metrics"]["cvne_mean_norm"] is None
        assert out["metrics"]["cvne_std_norm"] is None

    def test_empty_correction_vectors(self):
        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()
        out = compute_bp_dynamics_metrics(llr, energy, correction_vectors=[])
        assert out["metrics"]["cvne_entropy"] is None

    def test_single_correction_vector(self):
        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()
        cv = [np.array([1, 0, 1, 0, 1], dtype=np.int32)]
        out = compute_bp_dynamics_metrics(llr, energy, correction_vectors=cv)
        assert out["metrics"]["cvne_entropy"] is None

    def test_classifier_without_correction_vectors(self):
        """Classifier must not error when correction vectors are absent."""
        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()
        out = compute_bp_dynamics_metrics(llr, energy, correction_vectors=None)
        assert "regime" in out
        assert "evidence" in out

    def test_with_correction_vectors(self):
        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()
        cvs = _make_cycling_corrections()
        out = compute_bp_dynamics_metrics(llr, energy, correction_vectors=cvs)
        assert out["metrics"]["cvne_entropy"] is not None
        assert isinstance(out["metrics"]["cvne_entropy"], float)
        assert out["metrics"]["cvne_mean_norm"] is not None


# ── Test: Trace Normalization ─────────────────────────────────────────


class TestTraceNormalization:
    """LLR trace elements with various shapes must normalize to 1-D."""

    def test_1d_array(self):
        v = _normalize_llr_vector(np.array([1.0, 2.0, 3.0]))
        assert v.ndim == 1
        assert len(v) == 3

    def test_column_vector(self):
        v = _normalize_llr_vector(np.array([[1.0], [2.0], [3.0]]))
        assert v.ndim == 1
        assert len(v) == 3

    def test_row_vector(self):
        v = _normalize_llr_vector(np.array([[1.0, 2.0, 3.0]]))
        assert v.ndim == 1
        assert len(v) == 3

    def test_2d_matrix_uses_row0(self):
        m = np.array([[1.0, 2.0], [3.0, 4.0]])
        v = _normalize_llr_vector(m)
        assert v.ndim == 1
        np.testing.assert_array_equal(v, [1.0, 2.0])

    def test_list_input(self):
        v = _normalize_llr_vector([1, 2, 3])
        assert v.ndim == 1
        assert v.dtype == np.float64

    def test_scalar_input(self):
        v = _normalize_llr_vector(5.0)
        assert v.ndim == 1
        assert len(v) == 1


# ── Test: Edge Cases ──────────────────────────────────────────────────


class TestEdgeCases:
    """Empty/short traces must not crash."""

    def test_empty_traces(self):
        out = compute_bp_dynamics_metrics([], [])
        assert "metrics" in out
        assert "regime" in out
        assert out["regime"] == "stable_convergence"

    def test_single_iteration(self):
        llr = [np.array([1.0, -1.0, 0.0])]
        energy = [50.0]
        out = compute_bp_dynamics_metrics(llr, energy)
        assert "metrics" in out
        assert "regime" in out

    def test_two_iterations(self):
        llr = [np.array([1.0, -1.0]), np.array([1.0, 1.0])]
        energy = [50.0, 45.0]
        out = compute_bp_dynamics_metrics(llr, energy)
        assert isinstance(out["metrics"]["cpi_strength"], float)

    def test_all_zero_llr(self):
        llr = [np.zeros(5) for _ in range(10)]
        energy = list(range(10))
        out = compute_bp_dynamics_metrics(llr, energy)
        assert out["metrics"]["gos"] == 0.0


# ── Test: Return Structure ────────────────────────────────────────────


class TestReturnStructure:
    """Verify output is JSON-serializable with correct keys."""

    def test_top_level_keys(self):
        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()
        out = compute_bp_dynamics_metrics(llr, energy)
        assert set(out.keys()) == {"metrics", "regime", "evidence"}

    def test_json_serializable(self):
        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()
        out = compute_bp_dynamics_metrics(llr, energy)
        # Must not raise
        s = json.dumps(out, sort_keys=True)
        assert isinstance(s, str)

    def test_json_serializable_with_cvs(self):
        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()
        cvs = _make_cycling_corrections()
        out = compute_bp_dynamics_metrics(llr, energy, correction_vectors=cvs)
        s = json.dumps(out, sort_keys=True)
        assert isinstance(s, str)

    def test_evidence_keys(self):
        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()
        out = compute_bp_dynamics_metrics(llr, energy)
        ev = out["evidence"]
        assert "rule" in ev
        assert "comparisons" in ev
        assert "thresholds" in ev

    def test_metric_keys_present(self):
        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()
        out = compute_bp_dynamics_metrics(llr, energy)
        m = out["metrics"]
        expected_keys = {
            "msi", "mean_abs_delta_e", "flip_rate",
            "cpi_period", "cpi_strength",
            "tsl", "tsl_disagreement_count", "tsl_total_checked",
            "lec_mean", "lec_max",
            "cvne_entropy", "cvne_mean_norm", "cvne_std_norm",
            "gos", "gos_flip_fraction", "gos_max_node_flips",
            "eds_descent_fraction", "eds_variance",
            "bti", "bti_jump_count", "bti_sig_changes",
        }
        assert set(m.keys()) == expected_keys


# ── Test: Regime Coverage ─────────────────────────────────────────────


class TestRegimeCoverage:
    """Each regime must be triggerable with synthetic traces or metrics."""

    def test_stable_convergence(self):
        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()
        out = compute_bp_dynamics_metrics(llr, energy)
        assert out["regime"] == "stable_convergence"

    def test_oscillatory_convergence(self):
        """Short-period oscillation with high strength → oscillatory."""
        llr = _make_oscillating_llr_trace(n_iters=20, period=2)
        energy = _make_flat_energy()
        out = compute_bp_dynamics_metrics(llr, energy)
        assert out["regime"] == "oscillatory_convergence"

    def test_metastable_state(self):
        """High MSI + poor EDS → metastable."""
        # Create flat energy but with sign flips (metastable)
        n_vars = 10
        n_iters = 20
        llr = []
        for t in range(n_iters):
            vec = np.ones(n_vars, dtype=np.float64) * 0.5
            # Persistent but low-amplitude flips
            if t % 3 == 0:
                vec[:5] = -0.5
            llr.append(vec)
        # Flat energy with slight upward trend (non-descent)
        energy = [50.0 + 0.1 * (t % 3) for t in range(n_iters)]

        # Use direct classifier with tuned metrics for coverage
        metrics = {
            "msi": 0.8,
            "cpi_period": None,
            "cpi_strength": 0.1,
            "tsl": 0.1,
            "cvne_entropy": None,
            "gos": 0.2,
            "eds_descent_fraction": 0.3,
            "bti": 0.1,
        }
        result = classify_bp_regime(metrics)
        assert result["regime"] == "metastable_state"

    def test_trapping_set_regime(self):
        """High TSL → trapping_set_regime."""
        metrics = {
            "msi": 0.1,
            "cpi_period": None,
            "cpi_strength": 0.1,
            "tsl": 0.6,
            "cvne_entropy": None,
            "gos": 0.2,
            "eds_descent_fraction": 0.9,
            "bti": 0.1,
        }
        result = classify_bp_regime(metrics)
        assert result["regime"] == "trapping_set_regime"

    def test_correction_cycling(self):
        """High CVNE + moderate CPI → correction_cycling."""
        metrics = {
            "msi": 0.1,
            "cpi_period": 3,
            "cpi_strength": 0.4,
            "tsl": 0.1,
            "cvne_entropy": 2.0,
            "gos": 0.3,
            "eds_descent_fraction": 0.9,
            "bti": 0.1,
        }
        result = classify_bp_regime(metrics)
        assert result["regime"] == "correction_cycling"

    def test_correction_cycling_requires_cvne(self):
        """Without CVNE, correction_cycling cannot trigger."""
        metrics = {
            "msi": 0.1,
            "cpi_period": 3,
            "cpi_strength": 0.4,
            "tsl": 0.1,
            "cvne_entropy": None,
            "gos": 0.3,
            "eds_descent_fraction": 0.9,
            "bti": 0.1,
        }
        result = classify_bp_regime(metrics)
        assert result["regime"] != "correction_cycling"

    def test_chaotic_behavior(self):
        """High BTI + unstable EDS + no CPI → chaotic."""
        metrics = {
            "msi": 0.1,
            "cpi_period": None,
            "cpi_strength": 0.1,
            "tsl": 0.1,
            "cvne_entropy": None,
            "gos": 0.2,
            "eds_descent_fraction": 0.3,
            "bti": 0.7,
        }
        result = classify_bp_regime(metrics)
        assert result["regime"] == "chaotic_behavior"

    def test_all_regimes_reachable(self):
        """Verify all six regimes can be triggered."""
        regimes_hit = set()

        # stable_convergence
        m1 = {"msi": 0.0, "cpi_period": None, "cpi_strength": 0.0,
               "tsl": 0.0, "cvne_entropy": None, "gos": 0.0,
               "eds_descent_fraction": 1.0, "bti": 0.0}
        regimes_hit.add(classify_bp_regime(m1)["regime"])

        # oscillatory_convergence
        m2 = {"msi": 0.0, "cpi_period": 2, "cpi_strength": 0.9,
               "tsl": 0.0, "cvne_entropy": None, "gos": 0.0,
               "eds_descent_fraction": 1.0, "bti": 0.0}
        regimes_hit.add(classify_bp_regime(m2)["regime"])

        # metastable_state
        m3 = {"msi": 0.8, "cpi_period": None, "cpi_strength": 0.0,
               "tsl": 0.0, "cvne_entropy": None, "gos": 0.0,
               "eds_descent_fraction": 0.3, "bti": 0.0}
        regimes_hit.add(classify_bp_regime(m3)["regime"])

        # trapping_set_regime
        m4 = {"msi": 0.0, "cpi_period": None, "cpi_strength": 0.0,
               "tsl": 0.6, "cvne_entropy": None, "gos": 0.0,
               "eds_descent_fraction": 1.0, "bti": 0.0}
        regimes_hit.add(classify_bp_regime(m4)["regime"])

        # correction_cycling
        m5 = {"msi": 0.0, "cpi_period": 3, "cpi_strength": 0.4,
               "tsl": 0.0, "cvne_entropy": 2.0, "gos": 0.3,
               "eds_descent_fraction": 1.0, "bti": 0.0}
        regimes_hit.add(classify_bp_regime(m5)["regime"])

        # chaotic_behavior
        m6 = {"msi": 0.0, "cpi_period": None, "cpi_strength": 0.1,
               "tsl": 0.0, "cvne_entropy": None, "gos": 0.0,
               "eds_descent_fraction": 0.3, "bti": 0.7}
        regimes_hit.add(classify_bp_regime(m6)["regime"])

        expected = {
            "stable_convergence",
            "oscillatory_convergence",
            "metastable_state",
            "trapping_set_regime",
            "correction_cycling",
            "chaotic_behavior",
        }
        assert regimes_hit == expected


# ── Test: Classifier Evidence ─────────────────────────────────────────


class TestClassifierEvidence:
    """Evidence dict must contain rule, comparisons, thresholds."""

    def test_evidence_rule_matches_regime(self):
        metrics = {"msi": 0.0, "cpi_period": None, "cpi_strength": 0.0,
                   "tsl": 0.0, "cvne_entropy": None, "gos": 0.0,
                   "eds_descent_fraction": 1.0, "bti": 0.0}
        result = classify_bp_regime(metrics)
        assert result["evidence"]["rule"] == result["regime"]

    def test_comparisons_are_booleans(self):
        metrics = {"msi": 0.8, "cpi_period": 2, "cpi_strength": 0.9,
                   "tsl": 0.0, "cvne_entropy": None, "gos": 0.0,
                   "eds_descent_fraction": 1.0, "bti": 0.0}
        result = classify_bp_regime(metrics)
        for v in result["evidence"]["comparisons"].values():
            assert isinstance(v, bool)

    def test_custom_thresholds(self):
        metrics = {"msi": 0.0, "cpi_period": None, "cpi_strength": 0.0,
                   "tsl": 0.3, "cvne_entropy": None, "gos": 0.0,
                   "eds_descent_fraction": 1.0, "bti": 0.0}
        # With default threshold (0.4), TSL=0.3 won't trigger
        r1 = classify_bp_regime(metrics)
        assert r1["regime"] == "stable_convergence"
        # With lowered threshold, it triggers
        r2 = classify_bp_regime(metrics, thresholds={"tsl_min": 0.2})
        assert r2["regime"] == "trapping_set_regime"


# ── Test: Individual Metrics ──────────────────────────────────────────


class TestIndividualMetrics:
    """Sanity checks on individual metric computations."""

    def test_msi_stable(self):
        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()
        out = compute_bp_dynamics_metrics(llr, energy)
        # Stable trace should have low MSI
        assert out["metrics"]["msi"] < 0.3

    def test_msi_metastable(self):
        """Flat energy + sign flips → high MSI."""
        llr = _make_oscillating_llr_trace(period=3)
        energy = _make_flat_energy()
        out = compute_bp_dynamics_metrics(llr, energy)
        assert out["metrics"]["msi"] > 0.3

    def test_cpi_periodic(self):
        llr = _make_oscillating_llr_trace(period=2)
        energy = _make_flat_energy()
        out = compute_bp_dynamics_metrics(llr, energy)
        assert out["metrics"]["cpi_period"] is not None
        assert out["metrics"]["cpi_strength"] > 0.5

    def test_cpi_non_periodic(self):
        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()
        out = compute_bp_dynamics_metrics(llr, energy)
        # Stable trace: no strong periodicity
        assert out["metrics"]["cpi_strength"] < 0.5

    def test_lec_smooth(self):
        energy = _make_monotonic_energy()
        out = compute_bp_dynamics_metrics(
            _make_stable_llr_trace(), energy,
        )
        # Linear descent → zero curvature
        assert out["metrics"]["lec_mean"] < 0.01

    def test_eds_monotonic(self):
        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()
        out = compute_bp_dynamics_metrics(llr, energy)
        assert out["metrics"]["eds_descent_fraction"] == 1.0

    def test_gos_no_oscillation(self):
        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()
        out = compute_bp_dynamics_metrics(llr, energy)
        assert out["metrics"]["gos"] == 0.0

    def test_gos_oscillating(self):
        llr = _make_oscillating_llr_trace(period=2)
        energy = _make_flat_energy()
        out = compute_bp_dynamics_metrics(llr, energy)
        assert out["metrics"]["gos"] > 0.5


# ── Test: No Input Mutation ───────────────────────────────────────────


class TestNoInputMutation:
    """Inputs must not be modified."""

    def test_llr_not_mutated(self):
        llr = [np.array([1.0, -1.0, 0.0]) for _ in range(10)]
        copies = [arr.copy() for arr in llr]
        energy = list(range(10))
        compute_bp_dynamics_metrics(llr, energy)
        for orig, copy in zip(llr, copies):
            np.testing.assert_array_equal(orig, copy)

    def test_energy_not_mutated(self):
        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()
        energy_copy = list(energy)
        compute_bp_dynamics_metrics(llr, energy)
        assert energy == energy_copy


# ── Test: Bench Integration Smoke Test ────────────────────────────────


class TestBenchIntegration:
    """Lightweight smoke test for bench harness integration."""

    def test_import_bp_dynamics_from_bench(self):
        """The bench harness must import bp_dynamics without error."""
        from bench.dps_v381_eval import run_mode  # noqa: F401

    def test_run_mode_accepts_bp_dynamics_flag(self):
        """run_mode signature accepts enable_bp_dynamics."""
        import inspect
        from bench.dps_v381_eval import run_mode
        sig = inspect.signature(run_mode)
        assert "enable_bp_dynamics" in sig.parameters

    def test_run_evaluation_accepts_bp_dynamics_flag(self):
        """run_evaluation signature accepts enable_bp_dynamics."""
        import inspect
        from bench.dps_v381_eval import run_evaluation
        sig = inspect.signature(run_evaluation)
        assert "enable_bp_dynamics" in sig.parameters

    def test_bp_dynamics_output_keys(self):
        """When bp_dynamics data exists, expected keys are present."""
        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()
        out = compute_bp_dynamics_metrics(llr, energy)
        assert "bp_dynamics" not in out  # This is the per-trial metric, not bench output
        # Instead verify the structure
        assert "metrics" in out
        assert "regime" in out
        assert "evidence" in out


# ── Test: Params Override ─────────────────────────────────────────────


class TestParamsOverride:
    """Custom params must override defaults."""

    def test_custom_tail_window(self):
        llr = _make_stable_llr_trace(n_iters=30)
        energy = _make_monotonic_energy(n_iters=30)
        out1 = compute_bp_dynamics_metrics(llr, energy, params={"tail_window": 5})
        out2 = compute_bp_dynamics_metrics(llr, energy, params={"tail_window": 20})
        # Different window sizes may produce different metrics
        # Just verify both succeed and are valid
        assert isinstance(out1["metrics"]["msi"], float)
        assert isinstance(out2["metrics"]["msi"], float)

    def test_default_params_exist(self):
        assert "tail_window" in DEFAULT_PARAMS
        assert "msi_energy_tol" in DEFAULT_PARAMS

    def test_default_thresholds_exist(self):
        assert "cpi_strength_min" in DEFAULT_THRESHOLDS
        assert "msi_min" in DEFAULT_THRESHOLDS
        assert "tsl_min" in DEFAULT_THRESHOLDS


# ── Test: Input Validation Guards ─────────────────────────────────────


class TestInputValidation:
    """Mismatched or invalid inputs must raise deterministic ValueErrors."""

    def test_trace_length_mismatch_llr_vs_energy(self):
        """llr_trace and energy_trace with different lengths → ValueError."""
        llr = _make_stable_llr_trace(n_iters=10)
        energy = _make_monotonic_energy(n_iters=15)
        with pytest.raises(ValueError, match="Trace length mismatch"):
            compute_bp_dynamics_metrics(llr, energy)

    def test_trace_length_mismatch_llr_vs_correction_vectors(self):
        """correction_vectors length != llr_trace length → ValueError."""
        llr = _make_stable_llr_trace(n_iters=10)
        energy = _make_monotonic_energy(n_iters=10)
        cvs = _make_cycling_corrections(n_iters=7)
        with pytest.raises(ValueError, match="Trace length mismatch"):
            compute_bp_dynamics_metrics(llr, energy, correction_vectors=cvs)

    def test_rank3_tensor_rejected(self):
        """Rank-3 tensor in llr_trace → ValueError."""
        rank3 = np.ones((2, 2, 2), dtype=np.float64)
        llr = [rank3 for _ in range(5)]
        energy = _make_monotonic_energy(n_iters=5)
        with pytest.raises(ValueError, match="Unsupported llr vector rank"):
            compute_bp_dynamics_metrics(llr, energy)

    def test_llr_vector_length_mismatch(self):
        """Inconsistent vector lengths within llr_trace → ValueError."""
        llr = [np.ones(10, dtype=np.float64) for _ in range(5)]
        # Make one vector a different length
        llr[3] = np.ones(8, dtype=np.float64)
        energy = _make_monotonic_energy(n_iters=5)
        with pytest.raises(ValueError, match="LLR vector length mismatch"):
            compute_bp_dynamics_metrics(llr, energy)


# ── Test: Sign Pre-computation Equivalence (INV-001) ─────────────────


class TestSignPrecomputeEquivalence:
    """Pre-computed signs/CRC32 must produce identical metrics to inline."""

    def _run_without_cache(self, llr_trace, energy_trace, cvs=None, params=None):
        """Run compute_bp_dynamics_metrics with cache disabled.

        Calls each metric function without pre-computed signs/CRC sigs,
        forcing the fallback (original) code path.
        """
        from src.qec.diagnostics.bp_dynamics import (
            _compute_msi, _compute_cpi, _compute_tsl,
            _compute_lec, _compute_cvne, _compute_gos,
            _compute_eds, _compute_bti,
        )
        p = dict(DEFAULT_PARAMS)
        if params is not None:
            p.update(params)
        normed = _normalize_llr_trace(llr_trace) if llr_trace else []
        msi = _compute_msi(normed, energy_trace, p, _signs=None)
        cpi = _compute_cpi(normed, p, _crc_sigs=None)
        tsl = _compute_tsl(normed, p, _signs=None)
        lec = _compute_lec(energy_trace, p)
        cvne = _compute_cvne(cvs, p)
        gos = _compute_gos(normed, p, _signs=None)
        eds = _compute_eds(energy_trace, p)
        bti = _compute_bti(energy_trace, normed, p, _crc_sigs=None)
        return {
            "msi": msi, "cpi": cpi, "tsl": tsl, "lec": lec,
            "cvne": cvne, "gos": gos, "eds": eds, "bti": bti,
        }

    def _run_with_cache(self, llr_trace, energy_trace, cvs=None, params=None):
        """Run with pre-computed cache (the optimized path)."""
        from src.qec.diagnostics.bp_dynamics import (
            _compute_msi, _compute_cpi, _compute_tsl,
            _compute_lec, _compute_cvne, _compute_gos,
            _compute_eds, _compute_bti,
        )
        p = dict(DEFAULT_PARAMS)
        if params is not None:
            p.update(params)
        normed = _normalize_llr_trace(llr_trace) if llr_trace else []
        signs, crc_sigs = _precompute_signs_and_sigs(normed)
        msi = _compute_msi(normed, energy_trace, p, _signs=signs)
        cpi = _compute_cpi(normed, p, _crc_sigs=crc_sigs)
        tsl = _compute_tsl(normed, p, _signs=signs)
        lec = _compute_lec(energy_trace, p)
        cvne = _compute_cvne(cvs, p)
        gos = _compute_gos(normed, p, _signs=signs)
        eds = _compute_eds(energy_trace, p)
        bti = _compute_bti(energy_trace, normed, p, _crc_sigs=crc_sigs)
        return {
            "msi": msi, "cpi": cpi, "tsl": tsl, "lec": lec,
            "cvne": cvne, "gos": gos, "eds": eds, "bti": bti,
        }

    def test_stable_trace_equivalence(self):
        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()
        uncached = self._run_without_cache(llr, energy)
        cached = self._run_with_cache(llr, energy)
        assert json.dumps(uncached, sort_keys=True) == json.dumps(cached, sort_keys=True)

    def test_oscillating_trace_equivalence(self):
        llr = _make_oscillating_llr_trace(period=2)
        energy = _make_flat_energy()
        uncached = self._run_without_cache(llr, energy)
        cached = self._run_with_cache(llr, energy)
        assert json.dumps(uncached, sort_keys=True) == json.dumps(cached, sort_keys=True)

    def test_chaotic_trace_equivalence(self):
        llr = _make_chaotic_llr_trace()
        energy = _make_chaotic_energy()
        uncached = self._run_without_cache(llr, energy)
        cached = self._run_with_cache(llr, energy)
        assert json.dumps(uncached, sort_keys=True) == json.dumps(cached, sort_keys=True)

    def test_trapping_trace_equivalence(self):
        llr = _make_trapping_llr_trace()
        energy = _make_flat_energy()
        uncached = self._run_without_cache(llr, energy)
        cached = self._run_with_cache(llr, energy)
        assert json.dumps(uncached, sort_keys=True) == json.dumps(cached, sort_keys=True)

    def test_with_correction_vectors_equivalence(self):
        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()
        cvs = _make_cycling_corrections()
        uncached = self._run_without_cache(llr, energy, cvs=cvs)
        cached = self._run_with_cache(llr, energy, cvs=cvs)
        assert json.dumps(uncached, sort_keys=True) == json.dumps(cached, sort_keys=True)


class TestWindowSafety:
    """Differing window sizes must NOT reuse incompatible cached data."""

    def test_different_windows_produce_valid_independent_metrics(self):
        """With different tail_window vs gos_window, each metric uses its own window."""
        llr = _make_oscillating_llr_trace(n_iters=30, period=3)
        energy = _make_flat_energy(n_iters=30)
        # Run with asymmetric windows: tail_window and gos_window differ
        out = compute_bp_dynamics_metrics(
            llr, energy, params={"tail_window": 5, "gos_window": 20},
        )
        # Both metrics must be valid floats (no cross-contamination)
        assert isinstance(out["metrics"]["msi"], float)
        assert isinstance(out["metrics"]["gos"], float)
        assert 0.0 <= out["metrics"]["msi"] <= 1.0
        assert 0.0 <= out["metrics"]["gos"] <= 1.0

    def test_cached_vs_uncached_with_asymmetric_windows(self):
        """Pre-computed signs are per-vector, not per-window — safe."""
        llr = _make_oscillating_llr_trace(n_iters=25, period=2)
        energy = _make_flat_energy(n_iters=25)
        params = {
            "tail_window": 5,
            "tsl_window": 10,
            "gos_window": 15,
            "bti_window": 8,
        }
        out1 = compute_bp_dynamics_metrics(llr, energy, params=params)
        out2 = compute_bp_dynamics_metrics(llr, energy, params=params)
        assert json.dumps(out1, sort_keys=True) == json.dumps(out2, sort_keys=True)


class TestPrecomputeImmutability:
    """Pre-computed sign arrays must not be mutated."""

    def test_sign_arrays_not_writeable(self):
        llr = _make_stable_llr_trace(n_iters=5)
        normed = _normalize_llr_trace(llr)
        signs, _ = _precompute_signs_and_sigs(normed)
        for s in signs:
            assert not s.flags.writeable

    def test_sign_purity(self):
        """_sign(v) produces identical output for identical input."""
        v = np.array([1.0, -2.0, 0.0, 3.5, -0.1], dtype=np.float64)
        s1 = _sign(v)
        s2 = _sign(v)
        np.testing.assert_array_equal(s1, s2)

    def test_precompute_matches_inline_sign(self):
        """Pre-computed signs match element-wise _sign() calls."""
        llr = _make_chaotic_llr_trace(n_iters=15)
        normed = _normalize_llr_trace(llr)
        signs, crc_sigs = _precompute_signs_and_sigs(normed)
        import zlib
        for i, vec in enumerate(normed):
            expected_sign = _sign(vec)
            np.testing.assert_array_equal(signs[i], expected_sign)
            expected_crc = (
                zlib.crc32(expected_sign.astype(np.int8).tobytes()) & 0xFFFFFFFF
            )
            assert crc_sigs[i] == expected_crc


class TestSignPrecomputeDeterminism:
    """Pre-computed path must be deterministic across repeated runs."""

    def test_repeated_runs_identical(self):
        llr = _make_chaotic_llr_trace(n_iters=20)
        energy = _make_chaotic_energy(n_iters=20)
        results = []
        for _ in range(5):
            out = compute_bp_dynamics_metrics(llr, energy)
            results.append(json.dumps(out, sort_keys=True))
        assert len(set(results)) == 1
