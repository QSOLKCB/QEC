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

import qec.diagnostics.bp_dynamics as _bp_mod

from qec.diagnostics.bp_dynamics import (
    DEFAULT_PARAMS,
    DEFAULT_THRESHOLDS,
    _CROSS_CALL_CACHE,
    _CROSS_CALL_CACHE_ENABLED,
    _make_cache_key,
    _normalize_param_value,
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
        from qec.diagnostics.bp_dynamics import (
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
        from qec.diagnostics.bp_dynamics import (
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

    def test_asymmetric_windows_are_deterministic(self):
        """Asymmetric metric windows produce deterministic metrics with cached signs."""
        llr = _make_oscillating_llr_trace(n_iters=25, period=2)
        energy = _make_flat_energy(n_iters=25)
        params = {
            "tail_window": 5,
            "tsl_window": 10,
            "gos_window": 15,
            "bti_window": 8,
        }
        # Even when all metric windows are asymmetric, repeated cached evaluations
        # must remain exactly deterministic.
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


# ── Formal Validation: QSOL-BP-INV-001 (v68.5.1) ────────────────────


class TestSliceLevelEquivalence:
    """Prove that cached metrics index the same trace elements as inline.

    For each metric that uses _signs or _crc_sigs, verify that the
    precomputed array, when sliced by the metric's window parameter,
    produces element-identical values to the original inline computation.
    """

    def _get_normed_and_precomputed(self, llr_trace):
        normed = _normalize_llr_trace(llr_trace) if llr_trace else []
        signs, crc_sigs = _precompute_signs_and_sigs(normed)
        return normed, signs, crc_sigs

    def test_msi_slice_alignment(self):
        """MSI: cached _signs[t-1], _signs[t] == _sign(tail[t-1]), _sign(tail[t])."""
        llr = _make_chaotic_llr_trace(n_iters=20, n_vars=8)
        normed, signs, _ = self._get_normed_and_precomputed(llr)
        W = 7  # non-default window
        n = len(normed)
        w = min(W, n)
        # Original path: tail = normed[-w:], iterate pairs
        tail = normed[-w:]
        for t in range(1, len(tail)):
            s_prev_inline = _sign(tail[t - 1])
            s_curr_inline = _sign(tail[t])
            # Cached path: absolute index
            abs_idx_prev = n - w + t - 1
            abs_idx_curr = n - w + t
            np.testing.assert_array_equal(signs[abs_idx_prev], s_prev_inline)
            np.testing.assert_array_equal(signs[abs_idx_curr], s_curr_inline)

    def test_cpi_slice_alignment(self):
        """CPI: _crc_sigs[-w:] == [crc32(_sign(v)) for v in llr[-w:]]."""
        llr = _make_oscillating_llr_trace(n_iters=20, period=3)
        normed, _, crc_sigs = self._get_normed_and_precomputed(llr)
        import zlib as _zlib
        W = 9
        n = len(normed)
        w = min(W, n)
        cached_slice = crc_sigs[-w:]
        inline_sigs = []
        for vec in normed[-w:]:
            s = _sign(vec)
            inline_sigs.append(_zlib.crc32(s.astype(np.int8).tobytes()) & 0xFFFFFFFF)
        assert cached_slice == inline_sigs

    def test_tsl_slice_alignment(self):
        """TSL: cached _signs covers [n-w-1..n-2] and final = _signs[n-1]."""
        llr = _make_trapping_llr_trace(n_iters=20, n_vars=8)
        normed, signs, _ = self._get_normed_and_precomputed(llr)
        W = 10
        n = len(normed)
        w = min(W, n - 1)
        # Final sign
        np.testing.assert_array_equal(signs[-1], _sign(normed[-1]))
        # Tail before final: original = normed[-(w+1):-1]
        tail = normed[-(w + 1):-1]
        for j, vec in enumerate(tail):
            abs_idx = n - w - 1 + j
            np.testing.assert_array_equal(signs[abs_idx], _sign(vec))

    def test_gos_slice_alignment(self):
        """GOS: cached pairs match inline tail pairs."""
        llr = _make_oscillating_llr_trace(n_iters=20, period=2, n_vars=6)
        normed, signs, _ = self._get_normed_and_precomputed(llr)
        W = 8
        n = len(normed)
        w = min(W, n)
        tail = normed[-w:]
        for t in range(1, len(tail)):
            abs_prev = n - w + t - 1
            abs_curr = n - w + t
            np.testing.assert_array_equal(signs[abs_prev], _sign(tail[t - 1]))
            np.testing.assert_array_equal(signs[abs_curr], _sign(tail[t]))

    def test_bti_slice_alignment(self):
        """BTI: _crc_sigs[-w_llr:] == inline CRC32 of tail LLR signs."""
        llr = _make_chaotic_llr_trace(n_iters=20, n_vars=8)
        normed, _, crc_sigs = self._get_normed_and_precomputed(llr)
        import zlib as _zlib
        W = 6
        n = len(normed)
        w_llr = min(W, n)
        cached_slice = crc_sigs[-w_llr:]
        inline_sigs = []
        for vec in normed[-w_llr:]:
            sign_bytes = _sign(vec).astype(np.int8).tobytes()
            inline_sigs.append(_zlib.crc32(sign_bytes) & 0xFFFFFFFF)
        assert cached_slice == inline_sigs


class TestByteLevelEquivalence:
    """Prove CRC32 is computed on byte-identical data between paths.

    This is the CRITICAL proof: the precomputed CRC32 value for trace
    element i must equal the CRC32 that would be computed inline.
    """

    def test_sign_tobytes_identity(self):
        """sign_vec from precompute and inline produce identical bytes."""
        llr = _make_chaotic_llr_trace(n_iters=15, n_vars=12)
        normed = _normalize_llr_trace(llr)
        signs, _ = _precompute_signs_and_sigs(normed)
        for i, vec in enumerate(normed):
            inline_sign = _sign(vec)
            # Byte-level identity
            assert signs[i].astype(np.int8).tobytes() == inline_sign.astype(np.int8).tobytes()

    def test_crc32_identity_from_both_paths(self):
        """CRC32 from precomputed signs == CRC32 from inline signs."""
        import zlib as _zlib
        llr = _make_chaotic_llr_trace(n_iters=15, n_vars=12)
        normed = _normalize_llr_trace(llr)
        signs, crc_sigs = _precompute_signs_and_sigs(normed)
        for i, vec in enumerate(normed):
            # Path A: precomputed CRC
            crc_a = crc_sigs[i]
            # Path B: inline computation
            inline_sign = _sign(vec)
            crc_b = _zlib.crc32(inline_sign.astype(np.int8).tobytes()) & 0xFFFFFFFF
            # Path C: from precomputed sign array (re-derive CRC)
            crc_c = _zlib.crc32(signs[i].astype(np.int8).tobytes()) & 0xFFFFFFFF
            assert crc_a == crc_b == crc_c

    def test_sign_dtype_consistency(self):
        """Precomputed and inline _sign() produce same dtype and shape."""
        llr = _make_chaotic_llr_trace(n_iters=10, n_vars=7)
        normed = _normalize_llr_trace(llr)
        signs, _ = _precompute_signs_and_sigs(normed)
        for i, vec in enumerate(normed):
            inline = _sign(vec)
            assert signs[i].dtype == inline.dtype
            assert signs[i].shape == inline.shape

    def test_int8_cast_determinism(self):
        """astype(int8) is deterministic for _sign output values {-1, +1}."""
        # -1 as int8 = 0xFF, +1 as int8 = 0x01
        v = np.array([1.0, -1.0, 0.0, -0.5, 2.0], dtype=np.float64)
        s = _sign(v)  # [1, -1, 1, -1, 1]
        b1 = s.astype(np.int8).tobytes()
        b2 = s.astype(np.int8).tobytes()
        assert b1 == b2
        # Verify exact byte values
        expected = np.array([1, -1, 1, -1, 1], dtype=np.int8).tobytes()
        assert b1 == expected


class TestPurityProof:
    """Formally verify _sign() is pure (no mutation, no external state).

    Purity proof sketch:
    1. _sign(x) = np.where(x < 0, -1, 1)
    2. np.where is a pure function of its arguments
    3. x < 0 is an element-wise comparison, pure
    4. Neither -1 nor 1 depend on external state
    5. _sign does not read or write global state
    6. _sign does not mutate its input
    Therefore: _sign(x) depends only on the values of x.
    """

    def test_purity_same_values_same_result(self):
        """Same values in different arrays → identical output."""
        a = np.array([1.0, -2.0, 0.0, 3.5], dtype=np.float64)
        b = np.array([1.0, -2.0, 0.0, 3.5], dtype=np.float64)
        assert a is not b  # distinct objects
        np.testing.assert_array_equal(_sign(a), _sign(b))

    def test_purity_no_input_mutation(self):
        """_sign() does not mutate its input array."""
        v = np.array([1.0, -2.0, 0.0], dtype=np.float64)
        v_copy = v.copy()
        _ = _sign(v)
        np.testing.assert_array_equal(v, v_copy)

    def test_purity_independent_of_call_order(self):
        """Calling _sign on different inputs does not affect subsequent calls."""
        v1 = np.array([1.0, -1.0], dtype=np.float64)
        v2 = np.array([-5.0, 5.0], dtype=np.float64)
        # Call on v2 first, then v1
        r2a = _sign(v2).copy()
        r1 = _sign(v1).copy()
        r2b = _sign(v2).copy()
        # r2a and r2b must be identical (no state carried between calls)
        np.testing.assert_array_equal(r2a, r2b)
        np.testing.assert_array_equal(r1, np.array([1, -1]))

    def test_crc32_determinism(self):
        """zlib.crc32 is deterministic for identical byte inputs."""
        import zlib as _zlib
        data = np.array([1, -1, 1, 1, -1], dtype=np.int8).tobytes()
        results = [_zlib.crc32(data) & 0xFFFFFFFF for _ in range(100)]
        assert len(set(results)) == 1


class TestZeroMutationGuarantee:
    """Prove that cached arrays are never mutated during metric computation."""

    def test_write_to_cached_sign_raises(self):
        """Attempting to write to a cached sign array must raise ValueError."""
        llr = _make_stable_llr_trace(n_iters=5)
        normed = _normalize_llr_trace(llr)
        signs, _ = _precompute_signs_and_sigs(normed)
        for s in signs:
            with pytest.raises(ValueError, match="read-only"):
                s[0] = 99

    def test_cached_signs_unchanged_after_full_metric_run(self):
        """After compute_bp_dynamics_metrics, sign values are unchanged."""
        llr = _make_chaotic_llr_trace(n_iters=15, n_vars=8)
        normed = _normalize_llr_trace(llr)
        signs, crc_sigs = _precompute_signs_and_sigs(normed)
        # Snapshot
        sign_snapshots = [s.copy() for s in signs]
        crc_snapshots = list(crc_sigs)
        # Run full metric suite (uses signs internally)
        from qec.diagnostics.bp_dynamics import (
            _compute_msi, _compute_cpi, _compute_tsl,
            _compute_gos, _compute_bti,
        )
        p = dict(DEFAULT_PARAMS)
        energy = _make_chaotic_energy(n_iters=15)
        _compute_msi(normed, energy, p, _signs=signs)
        _compute_cpi(normed, p, _crc_sigs=crc_sigs)
        _compute_tsl(normed, p, _signs=signs)
        _compute_gos(normed, p, _signs=signs)
        _compute_bti(energy, normed, p, _crc_sigs=crc_sigs)
        # Verify no mutation
        for i, s in enumerate(signs):
            np.testing.assert_array_equal(s, sign_snapshots[i])
        assert crc_sigs == crc_snapshots

    def test_normed_llr_unchanged_after_precompute(self):
        """_precompute_signs_and_sigs does not mutate normed_llr."""
        llr = _make_chaotic_llr_trace(n_iters=10)
        normed = _normalize_llr_trace(llr)
        copies = [v.copy() for v in normed]
        _precompute_signs_and_sigs(normed)
        for orig, cpy in zip(normed, copies):
            np.testing.assert_array_equal(orig, cpy)


class TestWindowSafetyFormal:
    """Prove metrics correctly slice precomputed arrays per their own window.

    No assumption of equal window sizes is required because precomputed
    arrays are indexed by absolute trace position, and each metric
    applies its own window parameter independently.
    """

    def test_asymmetric_windows_cached_vs_uncached(self):
        """All 4 distinct window sizes → cached == uncached, bitwise."""
        from qec.diagnostics.bp_dynamics import (
            _compute_msi, _compute_cpi, _compute_tsl,
            _compute_gos, _compute_bti,
        )
        llr = _make_chaotic_llr_trace(n_iters=25, n_vars=8)
        energy = _make_chaotic_energy(n_iters=25)
        normed = _normalize_llr_trace(llr)
        signs, crc_sigs = _precompute_signs_and_sigs(normed)
        # Use 4 different window sizes
        p = dict(DEFAULT_PARAMS)
        p["tail_window"] = 5
        p["tsl_window"] = 10
        p["gos_window"] = 15
        p["bti_window"] = 8
        # Cached
        msi_c = _compute_msi(normed, energy, p, _signs=signs)
        cpi_c = _compute_cpi(normed, p, _crc_sigs=crc_sigs)
        tsl_c = _compute_tsl(normed, p, _signs=signs)
        gos_c = _compute_gos(normed, p, _signs=signs)
        bti_c = _compute_bti(energy, normed, p, _crc_sigs=crc_sigs)
        # Uncached
        msi_u = _compute_msi(normed, energy, p, _signs=None)
        cpi_u = _compute_cpi(normed, p, _crc_sigs=None)
        tsl_u = _compute_tsl(normed, p, _signs=None)
        gos_u = _compute_gos(normed, p, _signs=None)
        bti_u = _compute_bti(energy, normed, p, _crc_sigs=None)
        # Bitwise equality via JSON
        assert json.dumps(msi_c, sort_keys=True) == json.dumps(msi_u, sort_keys=True)
        assert json.dumps(cpi_c, sort_keys=True) == json.dumps(cpi_u, sort_keys=True)
        assert json.dumps(tsl_c, sort_keys=True) == json.dumps(tsl_u, sort_keys=True)
        assert json.dumps(gos_c, sort_keys=True) == json.dumps(gos_u, sort_keys=True)
        assert json.dumps(bti_c, sort_keys=True) == json.dumps(bti_u, sort_keys=True)

    def test_window_larger_than_trace(self):
        """When window > trace length, cached path still correct."""
        llr = _make_oscillating_llr_trace(n_iters=5, period=2)
        energy = _make_flat_energy(n_iters=5)
        # Windows much larger than trace
        params = {"tail_window": 100, "tsl_window": 100,
                  "gos_window": 100, "bti_window": 100}
        out1 = compute_bp_dynamics_metrics(llr, energy, params=params)
        out2 = compute_bp_dynamics_metrics(llr, energy, params=params)
        assert json.dumps(out1, sort_keys=True) == json.dumps(out2, sort_keys=True)

    def test_window_of_2_minimal(self):
        """Minimal window=2 with cached path works correctly."""
        from qec.diagnostics.bp_dynamics import (
            _compute_msi, _compute_gos,
        )
        llr = _make_oscillating_llr_trace(n_iters=10, period=2)
        energy = _make_flat_energy(n_iters=10)
        normed = _normalize_llr_trace(llr)
        signs, _ = _precompute_signs_and_sigs(normed)
        p = dict(DEFAULT_PARAMS)
        p["tail_window"] = 2
        p["gos_window"] = 2
        msi_c = _compute_msi(normed, energy, p, _signs=signs)
        msi_u = _compute_msi(normed, energy, p, _signs=None)
        gos_c = _compute_gos(normed, p, _signs=signs)
        gos_u = _compute_gos(normed, p, _signs=None)
        assert msi_c == msi_u
        assert gos_c == gos_u


class TestRedundancyElimination:
    """Measure actual reduction in _sign() calls via monkeypatching.

    This proves the optimization eliminates redundant computation
    via deterministic call counting, not estimation.
    """

    def test_sign_call_count_reduction(self, monkeypatch):
        """Precomputed path calls _sign() exactly N times (once per vec).

        Without precomputation, _sign() is called multiple times per vec
        across MSI, CPI, TSL, GOS, BTI.
        """
        import qec.diagnostics.bp_dynamics as mod

        call_count = {"n": 0}
        original_sign = mod._sign

        def counting_sign(x):
            call_count["n"] += 1
            return original_sign(x)

        llr = _make_oscillating_llr_trace(n_iters=20, period=2)
        energy = _make_flat_energy(n_iters=20)
        normed = _normalize_llr_trace(llr)
        N = len(normed)

        # --- Path A: uncached (original behavior) ---
        from qec.diagnostics.bp_dynamics import (
            _compute_msi, _compute_cpi, _compute_tsl,
            _compute_gos, _compute_bti,
        )
        p = dict(DEFAULT_PARAMS)
        call_count["n"] = 0
        monkeypatch.setattr(mod, "_sign", counting_sign)
        _compute_msi(normed, energy, p, _signs=None)
        _compute_cpi(normed, p, _crc_sigs=None)
        _compute_tsl(normed, p, _signs=None)
        _compute_gos(normed, p, _signs=None)
        _compute_bti(energy, normed, p, _crc_sigs=None)
        uncached_calls = call_count["n"]

        # --- Path B: cached (optimized) ---
        call_count["n"] = 0
        signs, crc_sigs = _precompute_signs_and_sigs(normed)
        precompute_calls = call_count["n"]
        _compute_msi(normed, energy, p, _signs=signs)
        _compute_cpi(normed, p, _crc_sigs=crc_sigs)
        _compute_tsl(normed, p, _signs=signs)
        _compute_gos(normed, p, _signs=signs)
        _compute_bti(energy, normed, p, _crc_sigs=crc_sigs)
        cached_metric_calls = call_count["n"] - precompute_calls
        total_cached_calls = call_count["n"]

        monkeypatch.undo()

        # Assertions:
        # Precompute calls _sign exactly N times (once per trace element)
        assert precompute_calls == N
        # Cached metrics call _sign 0 times (all reuse precomputed data)
        assert cached_metric_calls == 0
        # Total cached < uncached
        assert total_cached_calls < uncached_calls
        # Uncached must call _sign more than N times (redundant)
        assert uncached_calls > N


# ── Test: Cross-Call Deterministic Reuse (INV-003) ───────────────────


class TestCrossCallReuse:
    """Identical inputs across separate calls must produce identical outputs.

    Validates QSOL-BP-INV-003: cross-call deterministic reuse.
    """

    def setup_method(self):
        """Clear cross-call cache before each test."""
        _CROSS_CALL_CACHE.clear()

    def test_same_inputs_identical_outputs(self):
        """Two calls with identical inputs produce byte-identical JSON."""
        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()
        out1 = compute_bp_dynamics_metrics(llr, energy)
        out2 = compute_bp_dynamics_metrics(llr, energy)
        j1 = json.dumps(out1, sort_keys=True)
        j2 = json.dumps(out2, sort_keys=True)
        assert j1 == j2

    def test_cache_hit_after_first_call(self):
        """Second call with same inputs hits cache (deep copy, not same object)."""
        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()
        out1 = compute_bp_dynamics_metrics(llr, energy)
        assert len(_CROSS_CALL_CACHE) == 1
        out2 = compute_bp_dynamics_metrics(llr, energy)
        # Cache hit returns deep copy — different object, identical content
        assert out1 is not out2
        assert json.dumps(out1, sort_keys=True) == json.dumps(out2, sort_keys=True)

    def test_different_inputs_different_cache_entries(self):
        """Different inputs produce separate cache entries."""
        llr1 = _make_stable_llr_trace()
        energy1 = _make_monotonic_energy()
        llr2 = _make_oscillating_llr_trace()
        energy2 = _make_flat_energy()
        compute_bp_dynamics_metrics(llr1, energy1)
        compute_bp_dynamics_metrics(llr2, energy2)
        assert len(_CROSS_CALL_CACHE) == 2

    def test_different_params_different_cache_entries(self):
        """Same traces but different params → different cache keys."""
        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()
        p1 = dict(DEFAULT_PARAMS, tail_window=5)
        p2 = dict(DEFAULT_PARAMS, tail_window=8)
        key1 = _make_cache_key(llr, energy, None, p1)
        key2 = _make_cache_key(llr, energy, None, p2)
        assert key1 != key2

    def test_cross_helper_reuse(self):
        """Results from different helper-generated identical inputs match."""
        # Generate the same trace twice (fresh helper calls)
        llr_a = _make_oscillating_llr_trace(n_iters=20, period=2)
        energy_a = _make_flat_energy(n_iters=20)
        llr_b = _make_oscillating_llr_trace(n_iters=20, period=2)
        energy_b = _make_flat_energy(n_iters=20)
        out_a = compute_bp_dynamics_metrics(llr_a, energy_a)
        out_b = compute_bp_dynamics_metrics(llr_b, energy_b)
        assert json.dumps(out_a, sort_keys=True) == json.dumps(out_b, sort_keys=True)


class TestCacheTransparency:
    """Cached results must be bitwise identical to uncached results."""

    def setup_method(self):
        _CROSS_CALL_CACHE.clear()

    def test_cached_vs_uncached_stable(self):
        """Cache-returned result matches fresh computation."""
        import qec.diagnostics.bp_dynamics as mod

        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()

        # Compute with cache enabled (populates cache)
        out_cached = compute_bp_dynamics_metrics(llr, energy)

        # Compute with cache disabled (fresh computation)
        _CROSS_CALL_CACHE.clear()
        old_flag = mod._CROSS_CALL_CACHE_ENABLED
        mod._CROSS_CALL_CACHE_ENABLED = False
        try:
            out_fresh = compute_bp_dynamics_metrics(llr, energy)
        finally:
            mod._CROSS_CALL_CACHE_ENABLED = old_flag

        j_cached = json.dumps(out_cached, sort_keys=True)
        j_fresh = json.dumps(out_fresh, sort_keys=True)
        assert j_cached == j_fresh

    def test_cached_vs_uncached_oscillating(self):
        """Cache transparency for oscillating trace."""
        import qec.diagnostics.bp_dynamics as mod

        llr = _make_oscillating_llr_trace()
        energy = _make_flat_energy()

        out_cached = compute_bp_dynamics_metrics(llr, energy)
        _CROSS_CALL_CACHE.clear()
        old_flag = mod._CROSS_CALL_CACHE_ENABLED
        mod._CROSS_CALL_CACHE_ENABLED = False
        try:
            out_fresh = compute_bp_dynamics_metrics(llr, energy)
        finally:
            mod._CROSS_CALL_CACHE_ENABLED = old_flag

        assert json.dumps(out_cached, sort_keys=True) == json.dumps(out_fresh, sort_keys=True)


class TestCacheMutationSafety:
    """Mutating a returned result must not corrupt cached data.

    Proves Step 5 of INV-003 proof: no external mutation path to
    cached data exists (deep copy on both store and retrieval).
    """

    def setup_method(self):
        _CROSS_CALL_CACHE.clear()

    def test_mutate_output_does_not_corrupt_cache(self):
        """CRITICAL: Mutating returned dict does NOT affect cached data."""
        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()

        out1 = compute_bp_dynamics_metrics(llr, energy)
        original_json = json.dumps(out1, sort_keys=True)

        # Aggressively mutate the returned dict
        out1["regime"] = "CORRUPTED"
        out1["metrics"]["msi"] = -999.0
        out1["metrics"]["cpi_strength"] = -888.0
        out1["evidence"]["rule"] = "CORRUPTED"

        # Cache hit must return uncorrupted data (deep copy isolation)
        out2 = compute_bp_dynamics_metrics(llr, energy)
        assert json.dumps(out2, sort_keys=True) == original_json
        assert out2["regime"] != "CORRUPTED"
        assert out2["metrics"]["msi"] != -999.0

    def test_mutate_nested_evidence_does_not_corrupt_cache(self):
        """Mutation of nested evidence dict does not corrupt cache."""
        llr = _make_oscillating_llr_trace()
        energy = _make_flat_energy()

        out1 = compute_bp_dynamics_metrics(llr, energy)
        original_json = json.dumps(out1, sort_keys=True)

        # Mutate deeply nested structure
        out1["evidence"]["thresholds"]["cpi_period_max"] = -1.0
        out1["evidence"]["comparisons"] = {"CORRUPTED": True}

        # Cache must be unaffected
        out2 = compute_bp_dynamics_metrics(llr, energy)
        assert json.dumps(out2, sort_keys=True) == original_json

    def test_first_call_result_isolated_from_cache(self):
        """The first call's return value is independent of the cache copy."""
        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()

        out1 = compute_bp_dynamics_metrics(llr, energy)

        # Verify the cached copy is a different object than out1
        # (deep copy on store means the cache has its own copy)
        assert len(_CROSS_CALL_CACHE) == 1
        cached_value = next(iter(_CROSS_CALL_CACHE.values()))
        assert out1 is not cached_value
        assert out1["metrics"] is not cached_value["metrics"]


class TestCacheDeterminism:
    """Cache behavior must be deterministic across repeated runs."""

    def setup_method(self):
        _CROSS_CALL_CACHE.clear()

    def test_repeated_runs_identical(self):
        """Multiple rounds of cache-populate-and-read produce same results."""
        llr = _make_oscillating_llr_trace(period=3)
        energy = _make_flat_energy()
        results = []
        for _ in range(5):
            _CROSS_CALL_CACHE.clear()
            out = compute_bp_dynamics_metrics(llr, energy)
            results.append(json.dumps(out, sort_keys=True))
        assert all(r == results[0] for r in results)


class TestCrossCallRedundancyElimination:
    """Measure actual reduction in compute_bp_dynamics_metrics calls.

    Uses monkeypatch counting to prove cross-call cache eliminates
    redundant full computations for identical inputs.
    """

    def setup_method(self):
        _CROSS_CALL_CACHE.clear()

    def test_compute_call_count_reduction(self, monkeypatch):
        """Cross-call cache reduces full metric computations."""
        import qec.diagnostics.bp_dynamics as mod

        call_count = {"n": 0}
        original_precompute = mod._precompute_signs_and_sigs

        def counting_precompute(normed_llr):
            call_count["n"] += 1
            return original_precompute(normed_llr)

        monkeypatch.setattr(mod, "_precompute_signs_and_sigs",
                            counting_precompute)

        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()

        # First call: full computation
        compute_bp_dynamics_metrics(llr, energy)
        assert call_count["n"] == 1

        # Subsequent calls with identical inputs: cache hit, no computation
        compute_bp_dynamics_metrics(llr, energy)
        compute_bp_dynamics_metrics(llr, energy)
        compute_bp_dynamics_metrics(llr, energy)
        assert call_count["n"] == 1  # still 1 — all were cache hits

        # Different inputs: new computation
        llr2 = _make_oscillating_llr_trace()
        energy2 = _make_flat_energy()
        compute_bp_dynamics_metrics(llr2, energy2)
        assert call_count["n"] == 2  # one new computation

        # Same different inputs again: cache hit
        compute_bp_dynamics_metrics(llr2, energy2)
        assert call_count["n"] == 2  # still 2

        monkeypatch.undo()

    def test_measurement_summary(self, monkeypatch):
        """Reproduce the typical test-suite pattern and measure reduction.

        Simulates the pattern seen in test_bp_dynamics.py where
        _make_stable_llr_trace() + _make_monotonic_energy() is called
        ~14 times with identical inputs across different test classes.
        """
        import qec.diagnostics.bp_dynamics as mod

        call_count = {"n": 0}
        original_precompute = mod._precompute_signs_and_sigs

        def counting_precompute(normed_llr):
            call_count["n"] += 1
            return original_precompute(normed_llr)

        monkeypatch.setattr(mod, "_precompute_signs_and_sigs",
                            counting_precompute)

        # Pattern 1: stable trace (used ~14 times in test suite)
        llr_s = _make_stable_llr_trace()
        energy_s = _make_monotonic_energy()
        for _ in range(14):
            compute_bp_dynamics_metrics(llr_s, energy_s)

        # Pattern 2: oscillating trace (used ~6 times)
        llr_o = _make_oscillating_llr_trace()
        energy_o = _make_flat_energy()
        for _ in range(6):
            compute_bp_dynamics_metrics(llr_o, energy_o)

        monkeypatch.undo()

        # Without cache: 20 full computations
        # With cache: 2 full computations (one per distinct input)
        assert call_count["n"] == 2
        # Reduction: 20 → 2 = 90%


class TestByteKeyEquivalence:
    """Identical inputs constructed via different code paths must
    produce the same cache key and therefore the same output.

    Proves Step 2 of INV-003 proof: bytes(x) == bytes(y) when the
    underlying float64 values are identical.
    """

    def setup_method(self):
        _CROSS_CALL_CACHE.clear()

    def test_list_vs_array_inputs_same_result(self):
        """LLR trace built from lists vs numpy arrays → same output."""
        n_iters, n_vars = 20, 10
        # Path A: build from Python lists
        llr_lists = [
            [float(1.0 + 0.1 * t)] * n_vars for t in range(n_iters)
        ]
        # Path B: build from numpy arrays (same values)
        llr_arrays = [
            np.ones(n_vars, dtype=np.float64) * (1.0 + 0.1 * t)
            for t in range(n_iters)
        ]
        energy = _make_monotonic_energy(n_iters=n_iters)

        out_a = compute_bp_dynamics_metrics(llr_lists, energy)
        out_b = compute_bp_dynamics_metrics(llr_arrays, energy)
        assert json.dumps(out_a, sort_keys=True) == json.dumps(out_b, sort_keys=True)

    def test_fresh_helper_calls_same_key(self):
        """Two independent helper invocations produce same cache key."""
        from qec.diagnostics.bp_dynamics import _make_cache_key

        llr1 = _make_oscillating_llr_trace(n_iters=15, period=3)
        energy1 = _make_flat_energy(n_iters=15)
        llr2 = _make_oscillating_llr_trace(n_iters=15, period=3)
        energy2 = _make_flat_energy(n_iters=15)

        p = dict(DEFAULT_PARAMS)
        key1 = _make_cache_key(llr1, energy1, None, p)
        key2 = _make_cache_key(llr2, energy2, None, p)
        assert key1 == key2


class TestCorrectionVectorsCacheKey:
    """Validate correction_vectors participation in cache key."""

    def setup_method(self):
        _CROSS_CALL_CACHE.clear()

    def test_correction_vectors_affect_key(self):
        """Same inputs except correction_vectors → different cache keys."""
        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()
        cvs = _make_cycling_corrections()
        p = dict(DEFAULT_PARAMS)
        key_none = _make_cache_key(llr, energy, None, p)
        key_cvs = _make_cache_key(llr, energy, cvs, p)
        assert key_none != key_cvs

    def test_none_vs_empty_correction_vectors(self):
        """None and [] produce SAME cache key."""
        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()
        p = dict(DEFAULT_PARAMS)
        key_none = _make_cache_key(llr, energy, None, p)
        key_empty = _make_cache_key(llr, energy, [], p)
        assert key_none == key_empty


class TestParamNormalization:
    """Validate _normalize_param_value prevents silent key divergence."""

    def test_numpy_scalar_normalized(self):
        """np.float64(5.0) and Python 5.0 produce same normalized value."""
        assert _normalize_param_value(np.float64(5.0)) == 5.0
        assert type(_normalize_param_value(np.float64(5.0))) is float

    def test_numpy_int_normalized(self):
        """np.int64(12) normalizes to Python int."""
        assert _normalize_param_value(np.int64(12)) == 12
        assert type(_normalize_param_value(np.int64(12))) is int

    def test_python_native_passthrough(self):
        """Python native types pass through unchanged."""
        assert _normalize_param_value(5.0) == 5.0
        assert _normalize_param_value(12) == 12
        assert _normalize_param_value("abc") == "abc"

    def test_numpy_params_same_cache_key(self):
        """Params with numpy scalars produce same key as Python natives."""
        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()
        p_native = dict(DEFAULT_PARAMS, tail_window=5)
        p_numpy = dict(DEFAULT_PARAMS, tail_window=np.int64(5))
        key_native = _make_cache_key(llr, energy, None, p_native)
        key_numpy = _make_cache_key(llr, energy, None, p_numpy)
        assert key_native == key_numpy


class TestCacheHitConsistency:
    """Repeated cache hits must return identical results every time."""

    def setup_method(self):
        _CROSS_CALL_CACHE.clear()

    def test_ten_consecutive_hits_identical(self):
        """10 consecutive calls return byte-identical JSON."""
        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()
        results = [
            json.dumps(compute_bp_dynamics_metrics(llr, energy), sort_keys=True)
            for _ in range(10)
        ]
        assert all(r == results[0] for r in results)

    def test_interleaved_patterns_consistent(self):
        """Interleaved calls to different inputs return correct results."""
        llr_s = _make_stable_llr_trace()
        energy_s = _make_monotonic_energy()
        llr_o = _make_oscillating_llr_trace()
        energy_o = _make_flat_energy()

        ref_s = json.dumps(compute_bp_dynamics_metrics(llr_s, energy_s), sort_keys=True)
        ref_o = json.dumps(compute_bp_dynamics_metrics(llr_o, energy_o), sort_keys=True)

        # Interleave 5 rounds
        for _ in range(5):
            assert json.dumps(compute_bp_dynamics_metrics(llr_o, energy_o), sort_keys=True) == ref_o
            assert json.dumps(compute_bp_dynamics_metrics(llr_s, energy_s), sort_keys=True) == ref_s


class TestCacheCounters:
    """Validate _CACHE_HITS and _CACHE_MISSES instrumentation."""

    def setup_method(self):
        _CROSS_CALL_CACHE.clear()
        _bp_mod._CACHE_HITS = 0
        _bp_mod._CACHE_MISSES = 0

    def test_counters_track_hits_and_misses(self):
        """Counters accurately reflect cache behavior."""
        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()

        compute_bp_dynamics_metrics(llr, energy)
        assert _bp_mod._CACHE_MISSES == 1
        assert _bp_mod._CACHE_HITS == 0

        compute_bp_dynamics_metrics(llr, energy)
        assert _bp_mod._CACHE_MISSES == 1
        assert _bp_mod._CACHE_HITS == 1

        compute_bp_dynamics_metrics(llr, energy)
        assert _bp_mod._CACHE_MISSES == 1
        assert _bp_mod._CACHE_HITS == 2

        # Different input → new miss
        llr2 = _make_oscillating_llr_trace()
        energy2 = _make_flat_energy()
        compute_bp_dynamics_metrics(llr2, energy2)
        assert _bp_mod._CACHE_MISSES == 2
        assert _bp_mod._CACHE_HITS == 2
