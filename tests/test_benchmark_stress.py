"""Tests for benchmark_stress framework.

Covers:
- Determinism (byte-identical across runs)
- Oscillation period-2 actually flips signs
- No crash for small n_vars
- Classification fallback to 'unstable'
- Fidelity metric ranges
- All 9 scenarios execute
"""

import json

import numpy as np
import pytest

from src.qec.experiments.benchmark_stress import (
    SCENARIOS,
    _cosine_similarity,
    _derive_seed,
    _quantum_proxy,
    _sign_agreement,
    classify_with_fallback,
    compute_dark_state_mask,
    compute_fidelity,
    results_to_json,
    run_benchmark_stress,
)


class TestDeterminism:
    """Results must be byte-identical across runs."""

    def test_deterministic_seed_derivation(self):
        """SHA-256 seed derivation is deterministic."""
        s1 = _derive_seed("test_label")
        s2 = _derive_seed("test_label")
        assert s1 == s2
        # Different labels produce different seeds
        s3 = _derive_seed("other_label")
        assert s1 != s3

    def test_deterministic_results(self):
        """Full benchmark produces identical JSON across two runs."""
        r1 = run_benchmark_stress(n_vars=10, n_iters=8)
        r2 = run_benchmark_stress(n_vars=10, n_iters=8)

        # Strip timing (non-deterministic)
        for s in r1["scenarios"]:
            s.pop("timing", None)
        for s in r2["scenarios"]:
            s.pop("timing", None)

        j1 = json.dumps(r1, sort_keys=True)
        j2 = json.dumps(r2, sort_keys=True)
        assert j1 == j2, "Results are not byte-identical across runs"

    def test_deterministic_metrics_values(self):
        """Individual metric values are identical across runs."""
        r1 = run_benchmark_stress(n_vars=15, n_iters=10)
        r2 = run_benchmark_stress(n_vars=15, n_iters=10)

        for s1, s2 in zip(r1["scenarios"], r2["scenarios"]):
            assert s1["scenario"] == s2["scenario"]
            assert s1["regime"] == s2["regime"]
            for key in s1["metrics"]:
                v1 = s1["metrics"][key]
                v2 = s2["metrics"][key]
                if v1 is None:
                    assert v2 is None
                else:
                    assert v1 == v2, f"{s1['scenario']}.{key}: {v1} != {v2}"


class TestOscillationFlips:
    """Period-2 oscillation MUST flip sign each step."""

    def test_period2_sign_flip(self):
        """Adjacent LLR vectors in period-2 must have opposite signs."""
        seed = _derive_seed("benchmark_stress_v68.7.2:oscillating_period2")
        rng = np.random.Generator(np.random.PCG64(seed))

        # Find the period2 generator
        gen_fn = None
        for name, fn in SCENARIOS:
            if name == "oscillating_period2":
                gen_fn = fn
                break
        assert gen_fn is not None

        data = gen_fn(rng, 50, 30)
        llr_trace = data["llr_trace"]

        # Check that consecutive iterations have flipped signs
        for t in range(1, len(llr_trace)):
            signs_prev = np.sign(llr_trace[t - 1])
            signs_curr = np.sign(llr_trace[t])
            # With small noise, most signs should flip
            agreement = float(np.mean(signs_prev == signs_curr))
            # Most elements should disagree (flipped)
            assert agreement < 0.15, (
                f"Iteration {t}: sign agreement={agreement:.3f}, "
                f"expected < 0.15 (signs should flip)"
            )

    def test_period2_explicit_sign_pattern(self):
        """Even iterations positive base, odd iterations negative base."""
        seed = _derive_seed("benchmark_stress_v68.7.2:oscillating_period2")
        rng = np.random.Generator(np.random.PCG64(seed))

        gen_fn = dict(SCENARIOS)["oscillating_period2"]
        data = gen_fn(rng, 20, 10)
        llr_trace = data["llr_trace"]

        # Check cosine similarity between even pairs (should be positive)
        cos_even = _cosine_similarity(llr_trace[0], llr_trace[2])
        assert cos_even > 0.99, f"Even pair cosine={cos_even}, expected > 0.99"

        # Check cosine between even and odd (should be negative)
        cos_flip = _cosine_similarity(llr_trace[0], llr_trace[1])
        assert cos_flip < -0.99, f"Even-odd cosine={cos_flip}, expected < -0.99"


class TestSmallNVars:
    """Must not crash for small n_vars values."""

    @pytest.mark.parametrize("n_vars", [1, 2, 3, 4, 5])
    def test_no_crash_small_nvars(self, n_vars):
        """Run all scenarios with very small n_vars."""
        results = run_benchmark_stress(n_vars=n_vars, n_iters=8)
        assert results["n_scenarios"] == 9
        for s in results["scenarios"]:
            assert s["metrics"] is not None
            assert s["regime"] is not None

    def test_pathological_small_nvars(self):
        """Pathological scenario must handle n_vars=1 without crash."""
        seed = _derive_seed("test:pathological_extreme")
        rng = np.random.Generator(np.random.PCG64(seed))
        gen_fn = dict(SCENARIOS)["pathological_extreme"]
        data = gen_fn(rng, 1, 10)
        assert len(data["llr_trace"]) == 10
        assert len(data["energy_trace"]) == 10

    def test_pathological_nvars_3(self):
        """Pathological with n_vars=3: quarter=0 edge handled."""
        seed = _derive_seed("test:pathological_extreme_3")
        rng = np.random.Generator(np.random.PCG64(seed))
        gen_fn = dict(SCENARIOS)["pathological_extreme"]
        data = gen_fn(rng, 3, 5)
        assert len(data["llr_trace"]) == 5
        for llr in data["llr_trace"]:
            assert len(llr) == 3


class TestClassificationFallback:
    """Unknown regime must map to 'unstable'."""

    def test_known_regimes_pass_through(self):
        known = [
            "stable_convergence",
            "oscillatory_convergence",
            "metastable_state",
            "trapping_set_regime",
            "correction_cycling",
            "chaotic_behavior",
        ]
        for regime in known:
            assert classify_with_fallback(regime) == regime

    def test_unknown_regime_maps_to_unstable(self):
        assert classify_with_fallback("unknown_regime") == "unstable"
        assert classify_with_fallback("") == "unstable"
        assert classify_with_fallback("some_new_regime") == "unstable"


class TestFidelityRanges:
    """Fidelity metrics must be in valid ranges."""

    def test_cosine_range(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        c = np.array([-1.0, 0.0, 0.0], dtype=np.float64)

        assert _cosine_similarity(a, a) == pytest.approx(1.0)
        assert _cosine_similarity(a, b) == pytest.approx(0.0)
        assert _cosine_similarity(a, c) == pytest.approx(-1.0)

    def test_cosine_zero_norm(self):
        a = np.zeros(5, dtype=np.float64)
        b = np.ones(5, dtype=np.float64)
        assert _cosine_similarity(a, b) == 0.0

    def test_sign_agreement_range(self):
        a = np.array([1.0, -1.0, 1.0], dtype=np.float64)
        b = np.array([1.0, -1.0, 1.0], dtype=np.float64)
        assert _sign_agreement(a, b) == 1.0

        c = np.array([-1.0, 1.0, -1.0], dtype=np.float64)
        assert _sign_agreement(a, c) == 0.0

    def test_sign_agreement_empty(self):
        assert _sign_agreement(np.array([]), np.array([])) == 1.0

    def test_quantum_proxy_range(self):
        a = np.array([1.0, 0.0], dtype=np.float64)
        b = np.array([1.0, 0.0], dtype=np.float64)
        assert _quantum_proxy(a, b) == pytest.approx(1.0)

        c = np.array([0.0, 1.0], dtype=np.float64)
        assert _quantum_proxy(a, c) == pytest.approx(0.0)

    def test_quantum_proxy_zero_norm(self):
        a = np.zeros(3, dtype=np.float64)
        b = np.ones(3, dtype=np.float64)
        assert _quantum_proxy(a, b) == 0.0

    def test_fidelity_all_scenarios(self):
        """All fidelity values in valid ranges across all scenarios."""
        results = run_benchmark_stress(n_vars=20, n_iters=15)
        for s in results["scenarios"]:
            fid = s["fidelity"]
            assert -1.0 <= fid["cosine"] <= 1.0, (
                f"{s['scenario']}: cosine={fid['cosine']} out of range"
            )
            assert 0.0 <= fid["sign_agreement"] <= 1.0, (
                f"{s['scenario']}: sign_agreement={fid['sign_agreement']} out of range"
            )
            assert 0.0 <= fid["quantum_proxy"] <= 1.0, (
                f"{s['scenario']}: quantum_proxy={fid['quantum_proxy']} out of range"
            )


class TestComputeFidelity:
    """Edge cases for compute_fidelity."""

    def test_single_element_trace(self):
        fid = compute_fidelity([np.array([1.0, 2.0])])
        assert fid["cosine"] == 0.0
        assert fid["sign_agreement"] == 1.0
        assert fid["quantum_proxy"] == 0.0

    def test_empty_trace(self):
        fid = compute_fidelity([])
        assert fid["cosine"] == 0.0


class TestAllScenariosRun:
    """All 9 scenarios must execute without error."""

    def test_scenario_count(self):
        assert len(SCENARIOS) == 9

    def test_scenario_names(self):
        names = [name for name, _ in SCENARIOS]
        expected = [
            "converging_baseline",
            "high_noise",
            "oscillating_period3",
            "oscillating_period2",
            "long_iteration",
            "small_window",
            "large_window",
            "pathological_extreme",
            "diverging",
        ]
        assert names == expected

    def test_all_scenarios_produce_results(self):
        results = run_benchmark_stress(n_vars=10, n_iters=10)
        assert results["n_scenarios"] == 9
        seen = set()
        for s in results["scenarios"]:
            seen.add(s["scenario"])
            assert "metrics" in s
            assert "regime" in s
            assert "fidelity" in s
        assert len(seen) == 9


class TestJsonSerialization:
    """JSON output must be deterministic."""

    def test_json_roundtrip(self):
        results = run_benchmark_stress(n_vars=10, n_iters=8)
        json_str = results_to_json(results)
        parsed = json.loads(json_str)
        assert parsed["n_scenarios"] == 9
        assert parsed["version"] == "v68.8.0"

    def test_json_sorted_keys(self):
        results = run_benchmark_stress(n_vars=10, n_iters=8)
        json_str = results_to_json(results)
        # Verify it's valid JSON with sorted keys
        parsed = json.loads(json_str)
        # Re-serialize with sorted keys and compare
        json_str2 = json.dumps(parsed, sort_keys=True, indent=2)
        assert json_str == json_str2


class TestDarkState:
    """Tests for dark-state detection (v68.8.0)."""

    def test_dark_state_basic(self):
        """Constant vector trace → all dark after step 1."""
        v = np.array([1.0, -2.0, 3.0], dtype=np.float64)
        trace = [v.copy() for _ in range(5)]
        masks = compute_dark_state_mask(trace)
        assert len(masks) == 5
        # First timestep: all False
        assert not np.any(masks[0])
        # All subsequent: all True (identical vectors)
        for t in range(1, 5):
            assert np.all(masks[t])

    def test_dark_state_sign_flip(self):
        """Alternating sign → no dark stability."""
        v = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        trace = [v if t % 2 == 0 else -v for t in range(6)]
        masks = compute_dark_state_mask(trace)
        # First timestep: all False by definition
        assert not np.any(masks[0])
        # All others: no node is dark-stable (signs flip)
        for t in range(1, 6):
            assert not np.any(masks[t])

    def test_dark_state_small_variation(self):
        """Small noise below epsilon → still dark-stable."""
        v = np.array([10.0, -5.0, 3.0], dtype=np.float64)
        tiny = 1e-8  # well below epsilon=1e-6
        trace = [v, v + tiny, v + 2 * tiny, v + 3 * tiny]
        masks = compute_dark_state_mask(trace)
        assert not np.any(masks[0])
        for t in range(1, 4):
            assert np.all(masks[t])

    def test_dark_fraction_range(self):
        """Dark fractions in run_benchmark_stress are in [0, 1]."""
        results = run_benchmark_stress(n_vars=20, n_iters=15)
        for s in results["scenarios"]:
            assert 0.0 <= s["mean_dark_fraction"] <= 1.0, (
                f"{s['scenario']}: mean_dark_fraction={s['mean_dark_fraction']}"
            )
            assert 0.0 <= s["final_dark_fraction"] <= 1.0, (
                f"{s['scenario']}: final_dark_fraction={s['final_dark_fraction']}"
            )

    def test_determinism(self):
        """Repeated compute_dark_state_mask calls produce identical results."""
        rng = np.random.Generator(np.random.PCG64(42))
        trace = [rng.standard_normal(20).astype(np.float64) for _ in range(10)]
        masks_a = compute_dark_state_mask(trace)
        masks_b = compute_dark_state_mask(trace)
        assert len(masks_a) == len(masks_b)
        for ma, mb in zip(masks_a, masks_b):
            np.testing.assert_array_equal(ma, mb)
