# SPDX-License-Identifier: MIT
"""Tests for control policy sweep benchmarking — v134.6.0."""

from __future__ import annotations

import pytest

from qec.sims.qudit_lattice_engine import (
    QuditLatticeSnapshot,
    build_qudit_lattice,
)
from qec.sims.control_policy_benchmark import (
    BUILTIN_POLICIES,
    PolicyBenchmarkResult,
    compute_policy_score,
    render_benchmark_summary,
    run_policy_benchmark,
)


# ── Helpers ──────────────────────────────────────────────────────


def _make_3x2_lattice() -> QuditLatticeSnapshot:
    """Canonical 3x2 qutrit lattice for deterministic benchmarks."""
    return build_qudit_lattice(width=3, height=2, qudit_dimension=3)


def _make_4x4_lattice() -> QuditLatticeSnapshot:
    """Larger 4x4 qutrit lattice."""
    return build_qudit_lattice(width=4, height=4, qudit_dimension=3)


# ── Immutability ─────────────────────────────────────────────────


class TestImmutability:
    """PolicyBenchmarkResult must be frozen."""

    def test_result_is_frozen(self) -> None:
        result = PolicyBenchmarkResult(
            policy_name="test",
            mean_final_amplitude=0.5,
            mean_steps_to_stability=3.0,
            recovery_count=1,
            oscillation_count=2,
            score=0.8,
        )
        with pytest.raises(AttributeError):
            result.score = 0.0  # type: ignore[misc]

    def test_result_is_frozen_policy_name(self) -> None:
        result = PolicyBenchmarkResult(
            policy_name="test",
            mean_final_amplitude=0.5,
            mean_steps_to_stability=3.0,
            recovery_count=1,
            oscillation_count=2,
            score=0.8,
        )
        with pytest.raises(AttributeError):
            result.policy_name = "changed"  # type: ignore[misc]


# ── Scoring ──────────────────────────────────────────────────────


class TestScoring:
    """Deterministic scoring formula tests."""

    def test_perfect_score(self) -> None:
        """Immediate stability, no recovery, no oscillation."""
        score = compute_policy_score(
            steps=20, first_stable_step=0,
            recovery_count=0, oscillation_count=0,
        )
        assert score == pytest.approx(1.0)

    def test_never_stable(self) -> None:
        """Stability never reached, no recovery, no oscillation."""
        score = compute_policy_score(
            steps=20, first_stable_step=20,
            recovery_count=0, oscillation_count=0,
        )
        assert score == pytest.approx(0.0)

    def test_recovery_penalty(self) -> None:
        """Recovery penalizes score by 0.5 * ratio."""
        score = compute_policy_score(
            steps=20, first_stable_step=0,
            recovery_count=20, oscillation_count=0,
        )
        # 1.0 - 0.5 * 1.0 - 0.0 = 0.5
        assert score == pytest.approx(0.5)

    def test_oscillation_penalty(self) -> None:
        """Oscillation penalizes score by 0.25 * ratio."""
        score = compute_policy_score(
            steps=20, first_stable_step=0,
            recovery_count=0, oscillation_count=20,
        )
        # 1.0 - 0.0 - 0.25 * 1.0 = 0.75
        assert score == pytest.approx(0.75)

    def test_zero_steps(self) -> None:
        """Zero steps returns zero score."""
        score = compute_policy_score(
            steps=0, first_stable_step=0,
            recovery_count=0, oscillation_count=0,
        )
        assert score == 0.0

    def test_combined_penalties(self) -> None:
        """Both recovery and oscillation apply."""
        score = compute_policy_score(
            steps=10, first_stable_step=5,
            recovery_count=2, oscillation_count=4,
        )
        # stability_ratio = 1.0 - 5/10 = 0.5
        # recovery_ratio = 2/10 = 0.2
        # oscillation_ratio = 4/10 = 0.4
        # score = 0.5 - 0.5*0.2 - 0.25*0.4 = 0.5 - 0.1 - 0.1 = 0.3
        assert score == pytest.approx(0.3)


# ── Benchmark runner ─────────────────────────────────────────────


class TestRunPolicyBenchmark:
    """Integration tests for the policy benchmark runner."""

    def test_builtin_policies_run(self) -> None:
        """All three builtin policies execute without error."""
        snap = _make_3x2_lattice()
        results = run_policy_benchmark(snap, steps=10)
        assert len(results) == 3

    def test_result_types(self) -> None:
        """Results are a tuple of PolicyBenchmarkResult."""
        snap = _make_3x2_lattice()
        results = run_policy_benchmark(snap, steps=10)
        assert isinstance(results, tuple)
        assert all(isinstance(r, PolicyBenchmarkResult) for r in results)

    def test_deterministic_replay(self) -> None:
        """Identical inputs produce byte-identical outputs."""
        snap = _make_3x2_lattice()
        results_a = run_policy_benchmark(snap, steps=15)
        results_b = run_policy_benchmark(snap, steps=15)
        assert results_a == results_b

    def test_results_sorted_by_name(self) -> None:
        """Results are sorted by policy_name for deterministic ordering."""
        snap = _make_3x2_lattice()
        results = run_policy_benchmark(snap, steps=10)
        names = [r.policy_name for r in results]
        assert names == sorted(names)

    def test_builtin_policy_names(self) -> None:
        """All three builtin policies are present."""
        snap = _make_3x2_lattice()
        results = run_policy_benchmark(snap, steps=10)
        names = {r.policy_name for r in results}
        assert names == {"nominal", "aggressive_damping", "recovery_first"}

    def test_scores_are_finite(self) -> None:
        """All scores are finite floats."""
        snap = _make_3x2_lattice()
        results = run_policy_benchmark(snap, steps=10)
        for r in results:
            assert isinstance(r.score, float)
            assert r.score == r.score  # not NaN

    def test_amplitudes_nonnegative(self) -> None:
        """Final amplitudes must be non-negative."""
        snap = _make_3x2_lattice()
        results = run_policy_benchmark(snap, steps=20)
        for r in results:
            assert r.mean_final_amplitude >= 0.0

    def test_recovery_counts_nonnegative(self) -> None:
        """Recovery counts are non-negative integers."""
        snap = _make_3x2_lattice()
        results = run_policy_benchmark(snap, steps=10)
        for r in results:
            assert r.recovery_count >= 0
            assert r.oscillation_count >= 0

    def test_custom_policy(self) -> None:
        """A custom single policy can be benchmarked."""
        snap = _make_3x2_lattice()
        custom = (
            ("custom_conservative", (
                ("divergent", "recover", 0.5, True),
                ("fixed_point", "maintain", 1.0, False),
                ("oscillatory", "observe", 1.0, False),
                ("stable", "observe", 1.0, False),
            )),
        )
        results = run_policy_benchmark(snap, policies=custom, steps=10)
        assert len(results) == 1
        assert results[0].policy_name == "custom_conservative"

    def test_larger_lattice(self) -> None:
        """Benchmark runs on a larger lattice without error."""
        snap = _make_4x4_lattice()
        results = run_policy_benchmark(snap, steps=10)
        assert len(results) == 3
        for r in results:
            assert r.mean_final_amplitude >= 0.0


# ── Validation ───────────────────────────────────────────────────


class TestValidation:
    """Input validation tests."""

    def test_steps_must_be_positive(self) -> None:
        snap = _make_3x2_lattice()
        with pytest.raises(ValueError, match="steps must be >= 1"):
            run_policy_benchmark(snap, steps=0)

    def test_empty_policies_rejected(self) -> None:
        snap = _make_3x2_lattice()
        with pytest.raises(ValueError, match="policies must not be empty"):
            run_policy_benchmark(snap, policies=(), steps=10)


# ── Rendering ────────────────────────────────────────────────────


class TestRendering:
    """Benchmark summary rendering tests."""

    def test_render_contains_header(self) -> None:
        snap = _make_3x2_lattice()
        results = run_policy_benchmark(snap, steps=10)
        text = render_benchmark_summary(results)
        assert "Policy Benchmark Summary" in text

    def test_render_contains_all_policies(self) -> None:
        snap = _make_3x2_lattice()
        results = run_policy_benchmark(snap, steps=10)
        text = render_benchmark_summary(results)
        for r in results:
            assert r.policy_name in text

    def test_render_contains_best(self) -> None:
        snap = _make_3x2_lattice()
        results = run_policy_benchmark(snap, steps=10)
        text = render_benchmark_summary(results)
        assert "Best:" in text

    def test_render_deterministic(self) -> None:
        snap = _make_3x2_lattice()
        results = run_policy_benchmark(snap, steps=10)
        text_a = render_benchmark_summary(results)
        text_b = render_benchmark_summary(results)
        assert text_a == text_b

    def test_render_empty_results(self) -> None:
        text = render_benchmark_summary(())
        assert "Policy Benchmark Summary" in text


# ── Determinism across lattice sizes ─────────────────────────────


class TestDeterminismVariants:
    """Verify determinism across different lattice configurations."""

    def test_replay_4x4(self) -> None:
        snap = _make_4x4_lattice()
        a = run_policy_benchmark(snap, steps=20)
        b = run_policy_benchmark(snap, steps=20)
        assert a == b

    def test_step_count_affects_amplitudes(self) -> None:
        """More steps produce lower final amplitudes due to decay."""
        snap = _make_3x2_lattice()
        r10 = run_policy_benchmark(snap, steps=10)
        r20 = run_policy_benchmark(snap, steps=20)
        # More steps = more decay = lower amplitudes
        for a, b in zip(r10, r20):
            assert a.mean_final_amplitude >= b.mean_final_amplitude

    def test_different_initial_amplitude(self) -> None:
        """Different starting amplitudes produce different final amplitudes."""
        snap_low = build_qudit_lattice(
            width=3, height=2, qudit_dimension=3, initial_amplitude=0.5,
        )
        snap_high = build_qudit_lattice(
            width=3, height=2, qudit_dimension=3, initial_amplitude=2.0,
        )
        r_low = run_policy_benchmark(snap_low, steps=10)
        r_high = run_policy_benchmark(snap_high, steps=10)
        amps_low = tuple(r.mean_final_amplitude for r in r_low)
        amps_high = tuple(r.mean_final_amplitude for r in r_high)
        assert amps_low != amps_high
