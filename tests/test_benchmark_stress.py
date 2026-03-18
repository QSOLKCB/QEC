"""Tests for the BP dynamics benchmark + stress baseline (v68.7).

Validates:
- Deterministic output (repeated runs → identical JSON)
- All stress scenarios execute without error
- Classification logic is stable
- Artifact structure is correct
- Trace generators produce valid inputs
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.qec.experiments.benchmark_stress import (
    STRESS_SCENARIOS,
    _derive_seed,
    _make_rng,
    classify_outcome,
    generate_converging_trace,
    generate_diverging_trace,
    generate_high_noise_trace,
    generate_large_window_trace,
    generate_long_iteration_trace,
    generate_oscillating_trace,
    generate_pathological_trace,
    generate_small_window_trace,
    run_benchmark_suite,
    run_single_benchmark,
    serialize_artifact,
)


# ── Determinism tests ────────────────────────────────────────────────


class TestDeterminism:
    """Repeated runs with same seed must produce identical JSON."""

    def test_full_suite_determinism(self):
        a1 = run_benchmark_suite(master_seed=42)
        a2 = run_benchmark_suite(master_seed=42)
        # Remove wall_time (non-deterministic) before comparison
        s1 = _strip_wall_times(a1)
        s2 = _strip_wall_times(a2)
        j1 = json.dumps(s1, sort_keys=True, ensure_ascii=True)
        j2 = json.dumps(s2, sort_keys=True, ensure_ascii=True)
        assert j1 == j2, "Full suite not deterministic across runs"

    def test_single_scenario_determinism(self):
        seed = 12345
        trace = generate_converging_trace(n_vars=20, n_iters=15, seed=seed)
        r1 = run_single_benchmark("test", trace["llr_trace"], trace["energy_trace"], seed)
        r2 = run_single_benchmark("test", trace["llr_trace"], trace["energy_trace"], seed)
        # Metrics must be identical
        assert r1["metrics"] == r2["metrics"]
        assert r1["regime"] == r2["regime"]
        assert r1["outcome"] == r2["outcome"]

    def test_different_seeds_differ(self):
        a1 = run_benchmark_suite(master_seed=42)
        a2 = run_benchmark_suite(master_seed=99)
        # At least some metrics should differ
        m1 = a1["runs"][0]["metrics"]
        m2 = a2["runs"][0]["metrics"]
        assert m1 != m2, "Different seeds should produce different metrics"


class TestSeedDerivation:
    """Sub-seed derivation must be deterministic and collision-resistant."""

    def test_deterministic(self):
        s1 = _derive_seed(42, "test")
        s2 = _derive_seed(42, "test")
        assert s1 == s2

    def test_different_labels_differ(self):
        s1 = _derive_seed(42, "alpha")
        s2 = _derive_seed(42, "beta")
        assert s1 != s2

    def test_different_seeds_differ(self):
        s1 = _derive_seed(1, "test")
        s2 = _derive_seed(2, "test")
        assert s1 != s2

    def test_rng_determinism(self):
        rng1 = _make_rng(42, "test")
        rng2 = _make_rng(42, "test")
        v1 = rng1.standard_normal(10)
        v2 = rng2.standard_normal(10)
        np.testing.assert_array_equal(v1, v2)


# ── Trace generator tests ───────────────────────────────────────────


class TestTraceGenerators:
    """All trace generators must produce valid, deterministic traces."""

    @pytest.mark.parametrize(
        "gen,kwargs",
        [
            (generate_converging_trace, {"n_vars": 20, "n_iters": 15, "seed": 42}),
            (generate_high_noise_trace, {"n_vars": 20, "n_iters": 15, "seed": 42}),
            (generate_oscillating_trace, {"n_vars": 20, "n_iters": 15, "seed": 42}),
            (generate_long_iteration_trace, {"n_vars": 20, "n_iters": 50, "seed": 42}),
            (generate_small_window_trace, {"n_vars": 20, "seed": 42}),
            (generate_large_window_trace, {"n_vars": 50, "n_iters": 40, "seed": 42}),
            (generate_pathological_trace, {"n_vars": 40, "n_iters": 15, "seed": 42}),
            (generate_diverging_trace, {"n_vars": 20, "n_iters": 15, "seed": 42}),
        ],
    )
    def test_trace_valid_structure(self, gen, kwargs):
        trace = gen(**kwargs)
        assert "llr_trace" in trace
        assert "energy_trace" in trace
        assert len(trace["llr_trace"]) == len(trace["energy_trace"])
        assert len(trace["llr_trace"]) > 0

    @pytest.mark.parametrize(
        "gen,kwargs",
        [
            (generate_converging_trace, {"n_vars": 20, "n_iters": 15, "seed": 42}),
            (generate_high_noise_trace, {"n_vars": 20, "n_iters": 15, "seed": 42}),
            (generate_oscillating_trace, {"n_vars": 20, "n_iters": 15, "seed": 42}),
            (generate_diverging_trace, {"n_vars": 20, "n_iters": 15, "seed": 42}),
        ],
    )
    def test_trace_determinism(self, gen, kwargs):
        t1 = gen(**kwargs)
        t2 = gen(**kwargs)
        for i in range(len(t1["llr_trace"])):
            np.testing.assert_array_equal(t1["llr_trace"][i], t2["llr_trace"][i])
        assert t1["energy_trace"] == t2["energy_trace"]

    def test_traces_are_float64(self):
        trace = generate_converging_trace(n_vars=10, n_iters=5, seed=42)
        for llr in trace["llr_trace"]:
            assert llr.dtype == np.float64

    def test_energy_values_are_finite(self):
        for scenario in STRESS_SCENARIOS:
            gen = scenario["generator"]
            kwargs = dict(scenario.get("kwargs", {}))
            kwargs["seed"] = 42
            trace = gen(**kwargs)
            for e in trace["energy_trace"]:
                assert np.isfinite(e), f"Non-finite energy in {scenario['name']}"


# ── Stress scenario execution ───────────────────────────────────────


class TestStressExecution:
    """All stress scenarios must execute without errors."""

    def test_all_scenarios_run(self):
        artifact = run_benchmark_suite(master_seed=42)
        assert artifact["n_scenarios"] == len(STRESS_SCENARIOS)
        for run in artifact["runs"]:
            assert "metrics" in run
            assert "regime" in run
            assert "outcome" in run

    @pytest.mark.parametrize("scenario", STRESS_SCENARIOS, ids=lambda s: s["name"])
    def test_individual_scenario(self, scenario):
        gen = scenario["generator"]
        kwargs = dict(scenario.get("kwargs", {}))
        kwargs["seed"] = 42
        trace = gen(**kwargs)
        result = run_single_benchmark(
            scenario["name"],
            trace["llr_trace"],
            trace["energy_trace"],
            seed=42,
        )
        assert result["scenario"] == scenario["name"]
        assert result["outcome"] in ("converged", "oscillatory", "unstable", "diverged")
        assert isinstance(result["wall_time_seconds"], float)
        assert result["wall_time_seconds"] >= 0


# ── Classification logic ────────────────────────────────────────────


class TestClassification:
    """Outcome classification must be stable and cover all regimes."""

    @pytest.mark.parametrize(
        "regime,expected",
        [
            ("stable_convergence", "converged"),
            ("oscillatory_convergence", "oscillatory"),
            ("metastable_state", "unstable"),
            ("correction_cycling", "unstable"),
            ("chaotic_behavior", "diverged"),
        ],
    )
    def test_regime_mapping(self, regime, expected):
        result = {"regime": regime, "metrics": {}}
        assert classify_outcome(result) == expected

    def test_trapping_set_with_low_eds(self):
        result = {
            "regime": "trapping_set_regime",
            "metrics": {"eds_descent_fraction": 0.1},
        }
        assert classify_outcome(result) == "diverged"

    def test_trapping_set_with_high_eds(self):
        result = {
            "regime": "trapping_set_regime",
            "metrics": {"eds_descent_fraction": 0.8},
        }
        assert classify_outcome(result) == "unstable"

    def test_classification_determinism(self):
        """Same input → same classification, always."""
        trace = generate_oscillating_trace(n_vars=20, n_iters=30, seed=42, period=3)
        r1 = run_single_benchmark("osc", trace["llr_trace"], trace["energy_trace"], 42)
        r2 = run_single_benchmark("osc", trace["llr_trace"], trace["energy_trace"], 42)
        assert r1["outcome"] == r2["outcome"]


# ── Artifact structure ──────────────────────────────────────────────


class TestArtifactStructure:
    """Benchmark artifact must have correct structure and be JSON-serializable."""

    def test_top_level_keys(self):
        artifact = run_benchmark_suite(master_seed=42)
        assert "benchmark_version" in artifact
        assert "master_seed" in artifact
        assert "n_scenarios" in artifact
        assert "outcome_summary" in artifact
        assert "runs" in artifact

    def test_run_keys(self):
        artifact = run_benchmark_suite(master_seed=42)
        required_keys = {
            "scenario", "seed", "n_iterations", "n_vars",
            "wall_time_seconds", "regime", "outcome", "metrics",
            "evidence", "params", "description",
        }
        for run in artifact["runs"]:
            assert required_keys.issubset(set(run.keys())), (
                f"Missing keys in run: {required_keys - set(run.keys())}"
            )

    def test_json_serializable(self):
        artifact = run_benchmark_suite(master_seed=42)
        json_str = serialize_artifact(artifact)
        roundtrip = json.loads(json_str)
        assert roundtrip["master_seed"] == 42
        assert len(roundtrip["runs"]) == len(STRESS_SCENARIOS)

    def test_json_deterministic_ordering(self):
        artifact = run_benchmark_suite(master_seed=42)
        s1 = serialize_artifact(artifact)
        s2 = serialize_artifact(artifact)
        assert s1 == s2, "JSON serialization must be deterministic"

    def test_outcome_summary_counts(self):
        artifact = run_benchmark_suite(master_seed=42)
        total = sum(artifact["outcome_summary"].values())
        assert total == artifact["n_scenarios"]


# ── Helpers ──────────────────────────────────────────────────────────


def _strip_wall_times(artifact: dict) -> dict:
    """Return a copy of artifact with wall_time_seconds removed (for determinism checks)."""
    import copy

    a = copy.deepcopy(artifact)
    a.pop("git_hash", None)
    for run in a.get("runs", []):
        run.pop("wall_time_seconds", None)
    return a
