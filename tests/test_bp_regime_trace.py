"""
Tests for BP regime transition analysis (v4.5.0).

Validates:
  - Determinism (identical outputs on repeated runs)
  - Stable regime (monotonic descent → no transitions)
  - Oscillatory regime (periodic trace → mostly oscillatory)
  - Metastable plateau (long plateau then drop → event detected)
  - Chaotic regime (rapidly varying → high switch_rate)
  - Bench integration smoke test
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from qec.diagnostics.bp_regime_trace import (
    compute_bp_regime_trace,
)


# ── Helpers ───────────────────────────────────────────────────────────


def _make_stable_llr_trace(n_iters: int = 30, n_vars: int = 10) -> list:
    """Converging LLR trace: all positive, increasing magnitude."""
    trace = []
    for t in range(n_iters):
        vec = np.ones(n_vars, dtype=np.float64) * (1.0 + 0.1 * t)
        trace.append(vec)
    return trace


def _make_monotonic_energy(n_iters: int = 30) -> list:
    """Monotonically decreasing energy trace."""
    return [float(100.0 - 2.0 * t) for t in range(n_iters)]


def _make_oscillating_llr_trace(
    n_iters: int = 30, n_vars: int = 10, period: int = 2,
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


def _make_flat_energy(n_iters: int = 30, value: float = 50.0) -> list:
    """Flat energy trace (plateau)."""
    return [value] * n_iters


def _make_plateau_then_drop(
    n_iters: int = 40, plateau_len: int = 30, drop: float = 50.0,
) -> tuple:
    """Create a long plateau followed by a sudden energy drop.

    Returns (llr_trace, energy_trace).
    """
    n_vars = 10
    llr_trace = []
    energy_trace = []

    # Plateau phase: flat energy, signs flip slowly (metastable)
    for t in range(plateau_len):
        vec = np.ones(n_vars, dtype=np.float64) * 0.5
        if t % 3 == 0:
            vec[:5] = -0.5
        llr_trace.append(vec)
        energy_trace.append(50.0 + 0.05 * (t % 3))

    # Drop phase: energy drops, stable convergence
    for t in range(n_iters - plateau_len):
        vec = np.ones(n_vars, dtype=np.float64) * (2.0 + 0.1 * t)
        llr_trace.append(vec)
        energy_trace.append(50.0 - drop - 2.0 * t)

    return llr_trace, energy_trace


def _make_chaotic_llr_trace(n_iters: int = 30, n_vars: int = 10) -> list:
    """Chaotic LLR trace: deterministic pseudo-random sign changes."""
    rng = np.random.default_rng(42)
    trace = []
    for t in range(n_iters):
        vec = rng.standard_normal(n_vars) * 3.0
        trace.append(vec)
    return trace


def _make_chaotic_energy(n_iters: int = 30) -> list:
    """Erratic energy trace with large jumps."""
    rng = np.random.default_rng(42)
    base = 50.0
    trace = []
    for t in range(n_iters):
        base += rng.standard_normal() * 10.0
        trace.append(float(base))
    return trace


# ── Test: Determinism ─────────────────────────────────────────────────


class TestDeterminism:
    """Identical inputs must produce byte-identical JSON output."""

    def test_stable_determinism(self):
        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()
        out1 = compute_bp_regime_trace(llr, energy)
        out2 = compute_bp_regime_trace(llr, energy)
        j1 = json.dumps(out1, sort_keys=True)
        j2 = json.dumps(out2, sort_keys=True)
        assert j1 == j2

    def test_oscillating_determinism(self):
        llr = _make_oscillating_llr_trace()
        energy = _make_flat_energy()
        out1 = compute_bp_regime_trace(llr, energy)
        out2 = compute_bp_regime_trace(llr, energy)
        j1 = json.dumps(out1, sort_keys=True)
        j2 = json.dumps(out2, sort_keys=True)
        assert j1 == j2

    def test_chaotic_determinism(self):
        llr = _make_chaotic_llr_trace()
        energy = _make_chaotic_energy()
        out1 = compute_bp_regime_trace(llr, energy)
        out2 = compute_bp_regime_trace(llr, energy)
        j1 = json.dumps(out1, sort_keys=True)
        j2 = json.dumps(out2, sort_keys=True)
        assert j1 == j2

    def test_plateau_drop_determinism(self):
        llr, energy = _make_plateau_then_drop()
        out1 = compute_bp_regime_trace(llr, energy)
        out2 = compute_bp_regime_trace(llr, energy)
        j1 = json.dumps(out1, sort_keys=True)
        j2 = json.dumps(out2, sort_keys=True)
        assert j1 == j2


# ── Test: Stable Regime ───────────────────────────────────────────────


class TestStableRegime:
    """Monotonic energy descent → stable_convergence, no transitions."""

    def test_switch_rate_zero(self):
        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()
        out = compute_bp_regime_trace(llr, energy)
        assert out["summary"]["switch_rate"] == 0.0

    def test_no_transitions(self):
        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()
        out = compute_bp_regime_trace(llr, energy)
        assert len(out["transitions"]) == 0

    def test_single_regime_in_trace(self):
        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()
        out = compute_bp_regime_trace(llr, energy)
        regimes = set(out["regime_trace"])
        assert len(regimes) == 1
        assert "stable_convergence" in regimes

    def test_freeze_score_one(self):
        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()
        out = compute_bp_regime_trace(llr, energy)
        assert out["summary"]["freeze_score"] == 1.0

    def test_dwell_equals_trace_length(self):
        llr = _make_stable_llr_trace(n_iters=25)
        energy = _make_monotonic_energy(n_iters=25)
        out = compute_bp_regime_trace(llr, energy)
        assert out["summary"]["max_dwell"] == 25


# ── Test: Oscillatory Regime ──────────────────────────────────────────


class TestOscillatoryRegime:
    """Periodic sign pattern → mostly oscillatory_convergence."""

    def test_regime_trace_contains_oscillatory(self):
        llr = _make_oscillating_llr_trace(n_iters=30, period=2)
        energy = _make_flat_energy(n_iters=30)
        out = compute_bp_regime_trace(llr, energy)
        # After enough window buildup, should see oscillatory
        assert "oscillatory_convergence" in out["regime_trace"]

    def test_regime_trace_length(self):
        n = 30
        llr = _make_oscillating_llr_trace(n_iters=n)
        energy = _make_flat_energy(n_iters=n)
        out = compute_bp_regime_trace(llr, energy)
        assert len(out["regime_trace"]) == n


# ── Test: Metastable Plateau ─────────────────────────────────────────


class TestMetastablePlateau:
    """Long plateau then sudden drop → transition with event."""

    def test_has_transition(self):
        llr, energy = _make_plateau_then_drop(n_iters=40, plateau_len=30)
        out = compute_bp_regime_trace(llr, energy)
        # Should have at least one transition
        assert len(out["transitions"]) >= 1

    def test_large_max_dwell(self):
        llr, energy = _make_plateau_then_drop(n_iters=40, plateau_len=30)
        out = compute_bp_regime_trace(llr, energy)
        # The plateau should create a long dwell
        assert out["summary"]["max_dwell"] >= 10

    def test_event_detected(self):
        """Sudden large energy drop should trigger an event."""
        llr, energy = _make_plateau_then_drop(
            n_iters=40, plateau_len=30, drop=50.0,
        )
        out = compute_bp_regime_trace(llr, energy)
        assert any(tr["event"] for tr in out["transitions"])


# ── Test: Chaotic Regime ─────────────────────────────────────────────


class TestChaoticRegime:
    """Rapidly varying synthetic trace → high switch_rate."""

    def test_high_switch_rate(self):
        llr = _make_chaotic_llr_trace(n_iters=30)
        energy = _make_chaotic_energy(n_iters=30)
        out = compute_bp_regime_trace(llr, energy)
        # Chaotic traces should have some transitions
        assert len(out["transitions"]) >= 1

    def test_multiple_transitions(self):
        llr = _make_chaotic_llr_trace(n_iters=30)
        energy = _make_chaotic_energy(n_iters=30)
        out = compute_bp_regime_trace(llr, energy)
        assert out["summary"]["switch_rate"] > 0.0


# ── Test: Return Structure ────────────────────────────────────────────


class TestReturnStructure:
    """Verify output has correct keys and is JSON-serializable."""

    def test_top_level_keys(self):
        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()
        out = compute_bp_regime_trace(llr, energy)
        assert set(out.keys()) == {
            "regime_trace", "transitions", "dwell_times",
            "transition_counts", "summary",
        }

    def test_summary_keys(self):
        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()
        out = compute_bp_regime_trace(llr, energy)
        assert set(out["summary"].keys()) == {
            "switch_rate", "max_dwell", "freeze_score", "num_events",
        }

    def test_json_serializable(self):
        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()
        out = compute_bp_regime_trace(llr, energy)
        s = json.dumps(out, sort_keys=True)
        assert isinstance(s, str)

    def test_json_serializable_with_transitions(self):
        llr, energy = _make_plateau_then_drop()
        out = compute_bp_regime_trace(llr, energy)
        s = json.dumps(out, sort_keys=True)
        assert isinstance(s, str)

    def test_transition_structure(self):
        llr, energy = _make_plateau_then_drop()
        out = compute_bp_regime_trace(llr, energy)
        for tr in out["transitions"]:
            assert "t" in tr
            assert "from" in tr
            assert "to" in tr
            assert "event" in tr
            assert isinstance(tr["t"], int)
            assert isinstance(tr["from"], str)
            assert isinstance(tr["to"], str)
            assert isinstance(tr["event"], bool)

    def test_dwell_times_sorted(self):
        llr, energy = _make_plateau_then_drop()
        out = compute_bp_regime_trace(llr, energy)
        keys = list(out["dwell_times"].keys())
        assert keys == sorted(keys)

    def test_transition_counts_sorted(self):
        llr, energy = _make_plateau_then_drop()
        out = compute_bp_regime_trace(llr, energy)
        keys = list(out["transition_counts"].keys())
        assert keys == sorted(keys)


# ── Test: Edge Cases ──────────────────────────────────────────────────


class TestEdgeCases:
    """Empty/short traces must not crash."""

    def test_empty_traces(self):
        out = compute_bp_regime_trace([], [])
        assert out["regime_trace"] == []
        assert out["transitions"] == []
        assert out["summary"]["switch_rate"] == 0.0

    def test_single_iteration(self):
        llr = [np.array([1.0, -1.0, 0.0])]
        energy = [50.0]
        out = compute_bp_regime_trace(llr, energy)
        assert len(out["regime_trace"]) == 1
        assert len(out["transitions"]) == 0

    def test_two_iterations(self):
        llr = [np.array([1.0, -1.0]), np.array([1.0, 1.0])]
        energy = [50.0, 45.0]
        out = compute_bp_regime_trace(llr, energy)
        assert len(out["regime_trace"]) == 2

    def test_mismatched_trace_lengths_raises(self):
        llr = _make_stable_llr_trace(n_iters=10)
        energy = _make_monotonic_energy(n_iters=9)
        with pytest.raises(ValueError):
            compute_bp_regime_trace(llr, energy)

    def test_window_larger_than_trace(self):
        """Window=16 but trace has only 5 iterations."""
        llr = _make_stable_llr_trace(n_iters=5)
        energy = _make_monotonic_energy(n_iters=5)
        out = compute_bp_regime_trace(llr, energy, window=16)
        assert len(out["regime_trace"]) == 5


# ── Test: No Input Mutation ───────────────────────────────────────────


class TestNoInputMutation:
    """Inputs must not be modified."""

    def test_llr_not_mutated(self):
        llr = [np.array([1.0, -1.0, 0.0]) for _ in range(15)]
        copies = [arr.copy() for arr in llr]
        energy = list(range(15))
        compute_bp_regime_trace(llr, energy)
        for orig, copy in zip(llr, copies):
            np.testing.assert_array_equal(orig, copy)

    def test_energy_not_mutated(self):
        llr = _make_stable_llr_trace()
        energy = _make_monotonic_energy()
        energy_copy = list(energy)
        compute_bp_regime_trace(llr, energy)
        assert energy == energy_copy


# ── Test: Dwell Time Consistency ──────────────────────────────────────


class TestDwellTimeConsistency:
    """Dwell times must sum to total trace length."""

    def test_dwell_sum_equals_trace_length(self):
        llr = _make_stable_llr_trace(n_iters=25)
        energy = _make_monotonic_energy(n_iters=25)
        out = compute_bp_regime_trace(llr, energy)
        total = sum(
            d for dlist in out["dwell_times"].values() for d in dlist
        )
        assert total == 25

    def test_dwell_sum_with_transitions(self):
        llr, energy = _make_plateau_then_drop(n_iters=40)
        out = compute_bp_regime_trace(llr, energy)
        total = sum(
            d for dlist in out["dwell_times"].values() for d in dlist
        )
        assert total == 40

    def test_dwell_sum_chaotic(self):
        llr = _make_chaotic_llr_trace(n_iters=30)
        energy = _make_chaotic_energy(n_iters=30)
        out = compute_bp_regime_trace(llr, energy)
        total = sum(
            d for dlist in out["dwell_times"].values() for d in dlist
        )
        assert total == 30


# ── Test: Bench Integration Smoke Test ────────────────────────────────


class TestBenchIntegration:
    """Lightweight smoke test for bench harness integration."""

    def test_import_regime_trace_from_bench(self):
        """The bench harness must import bp_regime_trace without error."""
        from bench_reports.dps_v381_eval import run_mode  # noqa: F401

    def test_run_mode_accepts_bp_transitions_flag(self):
        """run_mode signature accepts enable_bp_transitions."""
        import inspect
        from bench_reports.dps_v381_eval import run_mode
        sig = inspect.signature(run_mode)
        assert "enable_bp_transitions" in sig.parameters

    def test_run_evaluation_accepts_bp_transitions_flag(self):
        """run_evaluation signature accepts enable_bp_transitions."""
        import inspect
        from bench_reports.dps_v381_eval import run_evaluation
        sig = inspect.signature(run_evaluation)
        assert "enable_bp_transitions" in sig.parameters
