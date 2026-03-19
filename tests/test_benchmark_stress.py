"""Tests for benchmark_stress framework.

Covers:
- Determinism (byte-identical across runs)
- Oscillation period-2 actually flips signs
- No crash for small n_vars
- Classification fallback to 'unstable'
- Fidelity metric ranges
- All 9 scenarios execute
- Pareto frontier
- Scoring layer

Version: v6.9.9.5
"""

import json

import numpy as np
import pytest

from src.qec.experiments.benchmark_stress import (
    SCENARIOS,
    _EXCLUDED_KEYS,
    _cosine_similarity,
    _derive_seed,
    _prepare_genome_list,
    _quantum_proxy,
    _sign_agreement,
    apply_decoder_genome,
    build_experiment_table,
    build_pairwise_comparison,
    build_pareto_frontier,
    build_scores,
    classify_with_fallback,
    compute_dark_state_mask,
    compute_fidelity,
    default_decoder_genome,
    fingerprint_decoder_genome,
    normalize_decoder_genome,
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
        assert parsed["version"] == "v6.9.9.5"

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


class TestDecoderGenome:
    """Tests for the decoder genome abstraction (v68.9.0)."""

    def test_default_genome_structure(self):
        """Default genome has expected keys and deterministic values."""
        g = default_decoder_genome()
        assert g["alphabet"] == "binary"
        assert g["clip_value"] is None
        assert g["damping"] == 0.0
        assert g["dark_skip"] is False
        # Deterministic: repeated calls identical
        assert default_decoder_genome() == g

    def test_clipping_effect(self):
        """Clipping bounds all values to [-clip_value, clip_value]."""
        trace = [np.array([10.0, -20.0, 0.5, -0.3], dtype=np.float64)]
        genome = default_decoder_genome()
        genome["clip_value"] = 5.0
        out = apply_decoder_genome(trace, genome)
        assert len(out) == 1
        assert float(np.max(out[0])) <= 5.0
        assert float(np.min(out[0])) >= -5.0
        # Values within range are unchanged
        assert out[0][2] == pytest.approx(0.5)
        assert out[0][3] == pytest.approx(-0.3)

    def test_damping_effect(self):
        """Damping moves v_t toward v_{t-1}."""
        v0 = np.array([10.0, 0.0, -10.0], dtype=np.float64)
        v1 = np.array([0.0, 10.0, 0.0], dtype=np.float64)
        trace = [v0.copy(), v1.copy()]
        genome = default_decoder_genome()
        genome["damping"] = 0.5
        out = apply_decoder_genome(trace, genome)
        # v_t = 0.5 * v1 + 0.5 * v0
        expected = 0.5 * v1 + 0.5 * v0
        np.testing.assert_allclose(out[1], expected)
        # v_0 unchanged
        np.testing.assert_array_equal(out[0], v0)

    def test_ternary_projection(self):
        """Ternary projection maps all values to {-1, 0, +1}."""
        trace = [np.array([5.0, -3.0, 0.0, 1e-13, -1e-13, 2.0], dtype=np.float64)]
        genome = default_decoder_genome()
        genome["alphabet"] = "ternary"
        out = apply_decoder_genome(trace, genome)
        allowed = {-1.0, 0.0, 1.0}
        for val in out[0]:
            assert float(val) in allowed
        assert out[0][0] == 1.0   # positive
        assert out[0][1] == -1.0  # negative
        assert out[0][2] == 0.0   # exact zero
        assert out[0][3] == 0.0   # tiny positive → 0
        assert out[0][4] == 0.0   # tiny negative → 0
        assert out[0][5] == 1.0   # positive

    def test_dark_skip_freezes_nodes(self):
        """Dark skip freezes dark-stable nodes to their previous value."""
        # Construct trace where node 0 is dark-stable (constant)
        # and node 1 changes
        v0 = np.array([5.0, 1.0], dtype=np.float64)
        v1 = np.array([5.0, -3.0], dtype=np.float64)  # node 0 same, node 1 changes
        v2 = np.array([5.0, 7.0], dtype=np.float64)   # node 0 same, node 1 changes
        trace = [v0.copy(), v1.copy(), v2.copy()]
        genome = default_decoder_genome()
        genome["dark_skip"] = True
        out = apply_decoder_genome(trace, genome)
        # Node 0 is dark-stable at t=1,2 → frozen to previous value
        assert out[1][0] == pytest.approx(5.0)
        assert out[2][0] == pytest.approx(5.0)
        # Node 1 changes sign → not dark-stable → not frozen
        assert out[1][1] == pytest.approx(-3.0)

    def test_pipeline_accepts_genome(self):
        """run_benchmark_suite with genome runs without error."""
        genome = default_decoder_genome()
        genome["clip_value"] = 3.0
        genome["damping"] = 0.1
        results = run_benchmark_stress(n_vars=10, n_iters=8, genome=genome)
        assert results["n_scenarios"] == 9
        for s in results["scenarios"]:
            assert s["genome"] == genome
            assert "genome_id" in s
            assert isinstance(s["genome_id"], str)
            assert len(s["genome_id"]) == 12
            assert "metrics" in s
            assert "regime" in s

    def test_determinism_with_genome(self):
        """Repeated runs with same genome produce identical results."""
        genome = {"alphabet": "ternary", "clip_value": 2.0,
                  "damping": 0.3, "dark_skip": True}
        r1 = run_benchmark_stress(n_vars=10, n_iters=8, genome=genome)
        r2 = run_benchmark_stress(n_vars=10, n_iters=8, genome=genome)
        for s in r1["scenarios"]:
            s.pop("timing", None)
        for s in r2["scenarios"]:
            s.pop("timing", None)
        j1 = json.dumps(r1, sort_keys=True)
        j2 = json.dumps(r2, sort_keys=True)
        assert j1 == j2


class TestGenomeHardening:
    """Hardening tests for genome normalization, fingerprinting, and artifact safety (v68.9.1)."""

    def test_genome_normalization_defaults(self):
        """Partial genome fills missing defaults."""
        g = normalize_decoder_genome({"damping": 0.5})
        assert g["alphabet"] == "binary"
        assert g["clip_value"] is None
        assert g["damping"] == 0.5
        assert g["dark_skip"] is False

    def test_genome_normalization_none(self):
        """None input returns full default genome."""
        g = normalize_decoder_genome(None)
        assert g == default_decoder_genome()

    def test_invalid_genome_alphabet(self):
        """Invalid alphabet is rejected."""
        with pytest.raises(ValueError, match="Invalid alphabet"):
            normalize_decoder_genome({"alphabet": "quaternary"})

    def test_invalid_genome_damping_too_high(self):
        """damping >= 1 is rejected."""
        with pytest.raises(ValueError, match="damping"):
            normalize_decoder_genome({"damping": 1.0})

    def test_invalid_genome_damping_negative(self):
        """Negative damping is rejected."""
        with pytest.raises(ValueError, match="damping"):
            normalize_decoder_genome({"damping": -0.1})

    def test_invalid_genome_clip_value_negative(self):
        """Negative clip_value is rejected."""
        with pytest.raises(ValueError, match="clip_value"):
            normalize_decoder_genome({"clip_value": -1.0})

    def test_invalid_genome_unknown_key(self):
        """Unknown key is rejected."""
        with pytest.raises(ValueError, match="Unknown genome keys"):
            normalize_decoder_genome({"alphabet": "binary", "bogus": 42})

    def test_genome_fingerprint_deterministic(self):
        """Same canonical genome gives same fingerprint."""
        g = default_decoder_genome()
        fp1 = fingerprint_decoder_genome(g)
        fp2 = fingerprint_decoder_genome(g)
        assert fp1 == fp2
        assert len(fp1) == 12

    def test_genome_fingerprint_equivalent_inputs(self):
        """Equivalent dicts with different insertion order give same fingerprint after normalization."""
        g1 = normalize_decoder_genome(
            {"dark_skip": True, "alphabet": "ternary", "damping": 0.1, "clip_value": 5.0}
        )
        g2 = normalize_decoder_genome(
            {"alphabet": "ternary", "clip_value": 5.0, "dark_skip": True, "damping": 0.1}
        )
        assert fingerprint_decoder_genome(g1) == fingerprint_decoder_genome(g2)

    def test_output_contains_canonical_genome(self):
        """Caller mutation after run does not affect stored genome."""
        genome = {"alphabet": "ternary", "clip_value": 2.0,
                  "damping": 0.3, "dark_skip": True}
        results = run_benchmark_stress(n_vars=10, n_iters=8, genome=genome)
        # Mutate the caller's dict
        genome["damping"] = 0.99
        genome["bogus_key"] = "injected"
        # Stored genome must remain canonical and unaffected
        for s in results["scenarios"]:
            assert s["genome"]["damping"] == 0.3
            assert "bogus_key" not in s["genome"]
            assert s["genome"]["alphabet"] == "ternary"

    def test_empty_trace_safe(self):
        """apply_decoder_genome handles empty trace for all genome configs."""
        genome = {"alphabet": "ternary", "clip_value": 1.0,
                  "damping": 0.5, "dark_skip": True}
        out = apply_decoder_genome([], genome)
        assert out == []

    def test_single_timestep_trace_safe(self):
        """Single-timestep trace is handled safely (no damping/dark-skip crash)."""
        v = np.array([3.0, -1.0, 0.0], dtype=np.float64)
        genome = {"alphabet": "binary", "clip_value": 2.0,
                  "damping": 0.5, "dark_skip": True}
        out = apply_decoder_genome([v], genome)
        assert len(out) == 1
        # Clipping applied: 3.0 → 2.0
        assert out[0][0] == pytest.approx(2.0)
        assert out[0][1] == pytest.approx(-1.0)

    def test_clip_value_zero(self):
        """clip_value=0 is valid and zeros all values."""
        v = np.array([5.0, -3.0, 0.1], dtype=np.float64)
        genome = normalize_decoder_genome({"clip_value": 0.0})
        out = apply_decoder_genome([v], genome)
        np.testing.assert_array_equal(out[0], np.zeros(3, dtype=np.float64))

    def test_apply_decoder_genome_non_mutating(self):
        """apply_decoder_genome must not mutate input trace or its arrays."""
        v0 = np.array([10.0, -5.0, 0.0], dtype=np.float64)
        v1 = np.array([-2.0, 7.0, 3.0], dtype=np.float64)
        v2 = np.array([1.0, -1.0, 4.0], dtype=np.float64)
        trace = [v0.copy(), v1.copy(), v2.copy()]

        # Preserve originals for comparison
        orig_values = [arr.copy() for arr in trace]
        orig_ids = [id(arr) for arr in trace]

        genome = normalize_decoder_genome(
            {"clip_value": 3.0, "damping": 0.4, "dark_skip": True}
        )
        out = apply_decoder_genome(trace, genome)

        # Value invariants: original arrays unchanged
        for i in range(len(trace)):
            np.testing.assert_array_equal(
                trace[i], orig_values[i],
                err_msg=f"trace[{i}] was mutated in-place",
            )

        # Object invariants: output is a new list with new arrays
        assert out is not trace
        for i in range(len(out)):
            assert out[i] is not trace[i], f"out[{i}] shares identity with trace[{i}]"

        # Structural invariant: original list length unchanged
        assert len(trace) == 3

        # Original array identities unchanged (not replaced)
        for i in range(len(trace)):
            assert id(trace[i]) == orig_ids[i], f"trace[{i}] object was replaced"


class TestGenomeSweep:
    """Tests for deterministic genome sweep evaluation (v69.0.1)."""

    def test_single_vs_sweep_equivalence(self):
        """Single genome via genome= vs genomes=[genome] produce equivalent results."""
        genome = {"clip_value": 3.0, "damping": 0.2}
        r_single = run_benchmark_stress(n_vars=10, n_iters=8, genome=genome)
        r_sweep = run_benchmark_stress(n_vars=10, n_iters=8, genomes=[genome])

        assert r_single["mode"] == "single"
        assert r_sweep["mode"] == "sweep"
        assert len(r_sweep["results"]) == 1

        # Strip timing from both, then compare scenario data
        sweep_inner = r_sweep["results"][0]
        for s in r_single["scenarios"]:
            s.pop("timing", None)
        for s in sweep_inner["scenarios"]:
            s.pop("timing", None)

        # Metadata equivalence
        assert r_single["version"] == sweep_inner["version"]
        assert r_single["base_seed_label"] == sweep_inner["base_seed_label"]
        assert r_single["n_vars"] == sweep_inner["n_vars"]
        assert r_single["n_iters_base"] == sweep_inner["n_iters_base"]

        # Core data must match
        assert r_single["n_scenarios"] == sweep_inner["n_scenarios"]
        j_single = json.dumps(r_single["scenarios"], sort_keys=True)
        j_sweep = json.dumps(sweep_inner["scenarios"], sort_keys=True)
        assert j_single == j_sweep

        # Genome identity equivalence
        for s1, s2 in zip(r_single["scenarios"], sweep_inner["scenarios"]):
            assert s1["genome_id"] == s2["genome_id"]

    def test_multiple_genomes_run(self):
        """Multiple distinct genomes all run and produce unique genome_ids."""
        genomes = [
            {"clip_value": 1.0, "damping": 0.0},
            {"clip_value": 5.0, "damping": 0.3},
            {"alphabet": "ternary", "damping": 0.1},
        ]
        result = run_benchmark_stress(n_vars=10, n_iters=8, genomes=genomes)

        assert result["mode"] == "sweep"
        assert len(result["results"]) == 3

        genome_ids = set()
        for suite in result["results"]:
            assert suite["n_scenarios"] == 9
            # All scenarios in this suite share the same genome_id
            ids_in_suite = {s["genome_id"] for s in suite["scenarios"]}
            assert len(ids_in_suite) == 1
            genome_ids.update(ids_in_suite)

        # All three genomes have distinct ids
        assert len(genome_ids) == 3

    def test_duplicate_genomes_rejected(self):
        """Providing the same genome twice raises ValueError."""
        genome = {"clip_value": 2.0, "damping": 0.1}
        with pytest.raises(ValueError, match="Duplicate genome_id"):
            _prepare_genome_list(genomes=[genome, genome])

    def test_invalid_dual_input_rejected(self):
        """Providing both genome and genomes raises ValueError."""
        with pytest.raises(ValueError, match="Cannot provide both"):
            run_benchmark_stress(
                n_vars=10, n_iters=8,
                genome={"damping": 0.1},
                genomes=[{"damping": 0.2}],
            )

    def test_determinism_of_sweep(self):
        """Two identical sweep runs produce identical outputs."""
        genomes = [
            {"clip_value": 1.0},
            {"clip_value": 5.0, "damping": 0.2},
        ]
        r1 = run_benchmark_stress(n_vars=10, n_iters=8, genomes=genomes)
        r2 = run_benchmark_stress(n_vars=10, n_iters=8, genomes=genomes)

        # Strip timing
        for suite in r1["results"]:
            for s in suite["scenarios"]:
                s.pop("timing", None)
        for suite in r2["results"]:
            for s in suite["scenarios"]:
                s.pop("timing", None)

        j1 = json.dumps(r1, sort_keys=True)
        j2 = json.dumps(r2, sort_keys=True)
        assert j1 == j2, "Sweep results are not deterministic"

    def test_order_preserved(self):
        """Input genome list order is preserved in results."""
        g_a = {"clip_value": 1.0, "damping": 0.0}
        g_b = {"clip_value": 5.0, "damping": 0.3}
        g_c = {"alphabet": "ternary"}

        result = run_benchmark_stress(n_vars=10, n_iters=8, genomes=[g_a, g_b, g_c])

        canonical_a = normalize_decoder_genome(g_a)
        canonical_b = normalize_decoder_genome(g_b)
        canonical_c = normalize_decoder_genome(g_c)

        id_a = fingerprint_decoder_genome(canonical_a)
        id_b = fingerprint_decoder_genome(canonical_b)
        id_c = fingerprint_decoder_genome(canonical_c)

        assert result["results"][0]["scenarios"][0]["genome_id"] == id_a
        assert result["results"][1]["scenarios"][0]["genome_id"] == id_b
        assert result["results"][2]["scenarios"][0]["genome_id"] == id_c


class TestAggregationLayer:
    """Tests for the aggregation layer — build_experiment_table (v69.1.0)."""

    def test_single_table_shape(self):
        """Single-genome run produces rows == number of scenarios."""
        result = run_benchmark_stress(n_vars=10, n_iters=8)
        table = result["table"]
        assert len(table) == len(result["scenarios"])
        # Each row has required fields
        required = {"genome_id", "scenario", "version", "base_seed_label",
                     "n_vars", "n_iters_base"}
        for row in table:
            assert required.issubset(row.keys()), (
                f"Missing fields: {required - row.keys()}"
            )

    def test_sweep_table_shape(self):
        """Sweep with 3 genomes produces rows == genomes x scenarios."""
        genomes = [
            {"clip_value": 1.0, "damping": 0.0},
            {"clip_value": 5.0, "damping": 0.3},
            {"alphabet": "ternary", "damping": 0.1},
        ]
        result = run_benchmark_stress(n_vars=10, n_iters=8, genomes=genomes)
        table = result["table"]
        scenarios_per_genome = len(result["results"][0]["scenarios"])
        expected_rows = len(genomes) * scenarios_per_genome
        assert len(table) == expected_rows

    def test_table_determinism(self):
        """Two identical runs produce identical tables."""
        r1 = run_benchmark_stress(n_vars=10, n_iters=8)
        r2 = run_benchmark_stress(n_vars=10, n_iters=8)
        t1 = r1["table"]
        t2 = r2["table"]
        assert len(t1) == len(t2)
        j1 = json.dumps(t1, sort_keys=True)
        j2 = json.dumps(t2, sort_keys=True)
        assert j1 == j2, "Tables are not deterministic across runs"

    def test_table_matches_raw_results(self):
        """Values in table match values in raw scenario results."""
        result = run_benchmark_stress(n_vars=10, n_iters=8)
        table = result["table"]
        scenarios = result["scenarios"]
        assert len(table) == len(scenarios)
        for row, scenario in zip(table, scenarios):
            assert row["genome_id"] == scenario["genome_id"]
            assert row["scenario"] == scenario["scenario"]
            assert row["n_vars"] == result["n_vars"]
            # Check flattened metrics match
            for k, v in scenario["metrics"].items():
                assert row[k] == v, (
                    f"Mismatch for {scenario['scenario']}.{k}: "
                    f"table={row[k]} vs raw={v}"
                )

    def test_order_preserved(self):
        """Genome and scenario order is preserved in table rows."""
        genomes = [
            {"clip_value": 1.0, "damping": 0.0},
            {"clip_value": 5.0, "damping": 0.3},
        ]
        result = run_benchmark_stress(n_vars=10, n_iters=8, genomes=genomes)
        table = result["table"]

        canonical_a = normalize_decoder_genome(genomes[0])
        canonical_b = normalize_decoder_genome(genomes[1])
        id_a = fingerprint_decoder_genome(canonical_a)
        id_b = fingerprint_decoder_genome(canonical_b)

        scenario_names = [name for name, _ in SCENARIOS]

        # First 9 rows: genome A, in scenario order
        for i, name in enumerate(scenario_names):
            assert table[i]["genome_id"] == id_a
            assert table[i]["scenario"] == name

        # Next 9 rows: genome B, in scenario order
        for i, name in enumerate(scenario_names):
            assert table[9 + i]["genome_id"] == id_b
            assert table[9 + i]["scenario"] == name

    def test_invalid_mode_rejected(self):
        """build_experiment_table rejects result with invalid mode."""
        with pytest.raises(ValueError, match="Invalid result mode"):
            build_experiment_table({"mode": "bogus"})

    def test_missing_mode_rejected(self):
        """build_experiment_table rejects result without mode key."""
        with pytest.raises(ValueError, match="Invalid result mode"):
            build_experiment_table({"scenarios": []})

    def test_missing_scenarios_rejected(self):
        """build_experiment_table rejects suite with missing scenarios."""
        with pytest.raises(ValueError, match="Malformed suite"):
            build_experiment_table({"mode": "single"})

    def test_invalid_scenarios_type_rejected(self):
        """build_experiment_table rejects suite with non-list scenarios."""
        with pytest.raises(ValueError, match="Malformed suite"):
            build_experiment_table({"mode": "single", "scenarios": "not_a_list"})

    def test_metric_key_collision_rejected(self):
        """build_experiment_table rejects metrics that collide with reserved keys."""
        malformed = {
            "mode": "single",
            "version": "test",
            "base_seed_label": "test",
            "n_vars": 10,
            "n_iters_base": 8,
            "scenarios": [{
                "genome_id": "abc",
                "scenario": "test",
                "metrics": {"genome_id": 42},
            }],
        }
        with pytest.raises(ValueError, match="Metric key collision"):
            build_experiment_table(malformed)


class TestPairwiseComparison:
    """Tests for pairwise genome comparison (v69.2.0)."""

    def test_pairwise_count(self):
        """3 genomes × 9 scenarios → 3 pairs per scenario = 27 rows."""
        genomes = [
            {"clip_value": 1.0, "damping": 0.0},
            {"clip_value": 5.0, "damping": 0.3},
            {"alphabet": "ternary", "damping": 0.1},
        ]
        result = run_benchmark_stress(n_vars=10, n_iters=8, genomes=genomes)
        comparisons = result["comparisons"]
        n_scenarios = len(SCENARIOS)
        n_pairs = 3  # C(3, 2)
        assert len(comparisons) == n_scenarios * n_pairs

    def test_single_genome_empty(self):
        """Single genome produces no comparison pairs."""
        result = run_benchmark_stress(n_vars=10, n_iters=8)
        assert result["comparisons"] == []

    def test_delta_correctness(self):
        """Manually verify one metric delta is row_j[metric] - row_i[metric]."""
        genomes = [
            {"clip_value": 1.0, "damping": 0.0},
            {"clip_value": 5.0, "damping": 0.3},
        ]
        result = run_benchmark_stress(n_vars=10, n_iters=8, genomes=genomes)
        table = result["table"]
        comparisons = result["comparisons"]

        # First comparison should be for the first scenario
        first_scenario = SCENARIOS[0][0]
        # Find the two table rows for this scenario
        scenario_rows = [r for r in table if r["scenario"] == first_scenario]
        assert len(scenario_rows) == 2

        # Find the matching comparison
        comp = [c for c in comparisons if c["scenario"] == first_scenario]
        assert len(comp) == 1
        comp = comp[0]

        assert comp["genome_a"] == scenario_rows[0]["genome_id"]
        assert comp["genome_b"] == scenario_rows[1]["genome_id"]

        # Check all numeric metric deltas using module-level _EXCLUDED_KEYS
        for key in scenario_rows[0]:
            if key in _EXCLUDED_KEYS:
                continue
            val_i = scenario_rows[0][key]
            val_j = scenario_rows[1][key]
            if isinstance(val_i, (int, float)) and isinstance(val_j, (int, float)):
                expected_delta = val_j - val_i
                assert comp[f"{key}_delta"] == pytest.approx(expected_delta), (
                    f"Delta mismatch for {key}: got {comp[f'{key}_delta']}, "
                    f"expected {expected_delta}"
                )

    def test_determinism(self):
        """Repeated runs produce identical comparisons."""
        genomes = [
            {"clip_value": 1.0, "damping": 0.0},
            {"clip_value": 5.0, "damping": 0.3},
        ]
        r1 = run_benchmark_stress(n_vars=10, n_iters=8, genomes=genomes)
        r2 = run_benchmark_stress(n_vars=10, n_iters=8, genomes=genomes)
        j1 = json.dumps(r1["comparisons"], sort_keys=True)
        j2 = json.dumps(r2["comparisons"], sort_keys=True)
        assert j1 == j2, "Pairwise comparisons are not deterministic"

    def test_order_preserved(self):
        """Scenario and genome pair ordering is stable across comparisons."""
        genomes = [
            {"clip_value": 1.0, "damping": 0.0},
            {"clip_value": 5.0, "damping": 0.3},
            {"alphabet": "ternary", "damping": 0.1},
        ]
        result = run_benchmark_stress(n_vars=10, n_iters=8, genomes=genomes)
        comparisons = result["comparisons"]

        # Compute expected genome_ids in input order
        canonical = [normalize_decoder_genome(g) for g in genomes]
        gids = [fingerprint_decoder_genome(c) for c in canonical]

        scenario_names = [name for name, _ in SCENARIOS]

        # Expected pairs: (0,1), (0,2), (1,2) — i < j ordering
        expected_pairs = [
            (gids[0], gids[1]),
            (gids[0], gids[2]),
            (gids[1], gids[2]),
        ]

        # 3 pairs per scenario, in scenario order
        idx = 0
        for sc_name in scenario_names:
            sc_comps = comparisons[idx:idx + 3]
            assert len(sc_comps) == 3
            for comp, (ga, gb) in zip(sc_comps, expected_pairs):
                assert comp["scenario"] == sc_name
                assert comp["genome_a"] == ga, (
                    f"Expected genome_a={ga} for scenario={sc_name}, got {comp['genome_a']}"
                )
                assert comp["genome_b"] == gb, (
                    f"Expected genome_b={gb} for scenario={sc_name}, got {comp['genome_b']}"
                )
            idx += 3

    def test_non_numeric_fields_do_not_produce_delta_keys(self):
        """Non-numeric fields (str, dict, None) must not appear as delta keys."""
        synthetic_table = [
            {
                "genome_id": "aaa",
                "scenario": "test_sc",
                "version": "v1",
                "base_seed_label": "test",
                "n_vars": 10,
                "n_iters_base": 8,
                "metric_float": 1.5,
                "metric_int": 3,
                "metric_str": "hello",
                "metric_dict": {"nested": True},
                "metric_none": None,
            },
            {
                "genome_id": "bbb",
                "scenario": "test_sc",
                "version": "v1",
                "base_seed_label": "test",
                "n_vars": 10,
                "n_iters_base": 8,
                "metric_float": 2.5,
                "metric_int": 7,
                "metric_str": "world",
                "metric_dict": {"nested": False},
                "metric_none": None,
            },
        ]
        result = {"table": synthetic_table}
        comps = build_pairwise_comparison(result)

        assert len(comps) == 1
        comp = comps[0]
        # Numeric deltas present
        assert "metric_float_delta" in comp
        assert comp["metric_float_delta"] == pytest.approx(1.0)
        assert "metric_int_delta" in comp
        assert comp["metric_int_delta"] == 4
        # Non-numeric fields must NOT produce delta keys
        assert "metric_str_delta" not in comp
        assert "metric_dict_delta" not in comp
        assert "metric_none_delta" not in comp

    def test_missing_table_raises(self):
        """build_pairwise_comparison raises on missing table."""
        with pytest.raises(ValueError, match="missing 'table'"):
            build_pairwise_comparison({})


class TestParetoFrontier:
    """Tests for Pareto frontier identification (v69.3.1)."""

    def test_single_genome_all_pareto(self):
        """Single genome is trivially Pareto-optimal in every scenario."""
        result = run_benchmark_stress(n_vars=10, n_iters=8)
        pareto = result["pareto"]
        assert len(pareto) == len(SCENARIOS)
        for entry in pareto:
            assert len(entry["pareto_genomes"]) == 1

    def test_pareto_present_in_result(self):
        """run_benchmark_stress includes 'pareto' key."""
        result = run_benchmark_stress(n_vars=10, n_iters=8)
        assert "pareto" in result
        assert isinstance(result["pareto"], list)

    def test_sweep_pareto(self):
        """Sweep with multiple genomes produces Pareto frontier per scenario."""
        genomes = [
            {"clip_value": 1.0, "damping": 0.0},
            {"clip_value": 5.0, "damping": 0.3},
            {"alphabet": "ternary", "damping": 0.1},
        ]
        result = run_benchmark_stress(n_vars=10, n_iters=8, genomes=genomes)
        pareto = result["pareto"]
        assert len(pareto) == len(SCENARIOS)
        for entry in pareto:
            assert "scenario" in entry
            assert "pareto_genomes" in entry
            # At least one genome must be Pareto-optimal
            assert len(entry["pareto_genomes"]) >= 1
            # All pareto genomes must be valid genome_ids from the table
            table_gids = {r["genome_id"] for r in result["table"]}
            for gid in entry["pareto_genomes"]:
                assert gid in table_gids

    def test_determinism(self):
        """Repeated runs produce identical Pareto frontiers."""
        genomes = [
            {"clip_value": 1.0, "damping": 0.0},
            {"clip_value": 5.0, "damping": 0.3},
        ]
        r1 = run_benchmark_stress(n_vars=10, n_iters=8, genomes=genomes)
        r2 = run_benchmark_stress(n_vars=10, n_iters=8, genomes=genomes)
        j1 = json.dumps(r1["pareto"], sort_keys=True)
        j2 = json.dumps(r2["pareto"], sort_keys=True)
        assert j1 == j2, "Pareto frontiers are not deterministic"

    def test_unknown_scenario_raises(self):
        """Pareto frontier raises on unknown scenario in comparisons."""
        result = {
            "table": [
                {"genome_id": "aaa", "scenario": "sc1", "version": "v1",
                 "base_seed_label": "t", "n_vars": 10, "n_iters_base": 8,
                 "metric_a": 1.0},
            ],
            "comparisons": [
                {"scenario": "nonexistent", "genome_a": "aaa", "genome_b": "bbb"},
            ],
        }
        with pytest.raises(ValueError, match="unknown scenario"):
            build_pareto_frontier(result)

    def test_no_numeric_metrics_raises(self):
        """Pareto frontier raises when scenario has no numeric metrics."""
        result = {
            "table": [
                {"genome_id": "aaa", "scenario": "sc1", "version": "v1",
                 "base_seed_label": "t", "n_vars": "not_numeric",
                 "n_iters_base": "not_numeric"},
            ],
        }
        with pytest.raises(ValueError, match="no valid numeric deltas"):
            build_pareto_frontier(result)

    def test_missing_table_raises(self):
        """Pareto frontier raises on missing table."""
        with pytest.raises(ValueError, match="missing 'table'"):
            build_pareto_frontier({})

    def test_scenario_order_preserved(self):
        """Pareto frontier preserves scenario order from the table."""
        result = run_benchmark_stress(n_vars=10, n_iters=8)
        pareto = result["pareto"]
        scenario_names = [name for name, _ in SCENARIOS]
        pareto_scenarios = [entry["scenario"] for entry in pareto]
        assert pareto_scenarios == scenario_names


class TestScoringLayer:
    """Tests for the scoring layer (v69.9.4)."""

    def test_single_genome_score_half(self):
        """Single genome gets score 0.5 (all constant columns)."""
        result = run_benchmark_stress(n_vars=10, n_iters=8)
        scores = result["scores"]
        assert len(scores) == 1
        assert scores[0]["score"] == pytest.approx(0.5)
        assert scores[0]["rank"] == 1

    def test_scores_present_in_result(self):
        """run_benchmark_stress includes 'scores' key."""
        result = run_benchmark_stress(n_vars=10, n_iters=8)
        assert "scores" in result
        assert isinstance(result["scores"], list)

    def test_sweep_scores(self):
        """Sweep with multiple genomes produces ranked scores."""
        genomes = [
            {"clip_value": 1.0, "damping": 0.0},
            {"clip_value": 5.0, "damping": 0.3},
            {"alphabet": "ternary", "damping": 0.1},
        ]
        result = run_benchmark_stress(n_vars=10, n_iters=8, genomes=genomes)
        scores = result["scores"]
        assert len(scores) == 3
        # Scores are in [0, 1]
        for s in scores:
            assert 0.0 <= s["score"] <= 1.0
            assert "genome_id" in s
            assert "rank" in s
        # Ranks are 1, 2, 3
        ranks = [s["rank"] for s in scores]
        assert sorted(ranks) == [1, 2, 3]
        # Scores are descending
        for i in range(len(scores) - 1):
            assert scores[i]["score"] >= scores[i + 1]["score"]

    def test_ranking_descending_with_tiebreak(self):
        """Scores are ranked descending; ties broken by genome_id ascending."""
        genomes = [
            {"clip_value": 1.0, "damping": 0.0},
            {"clip_value": 5.0, "damping": 0.3},
        ]
        result = run_benchmark_stress(n_vars=10, n_iters=8, genomes=genomes)
        scores = result["scores"]
        assert len(scores) == 2
        # First score >= second score
        assert scores[0]["score"] >= scores[1]["score"]
        # If equal scores, genome_id ascending
        if scores[0]["score"] == scores[1]["score"]:
            assert scores[0]["genome_id"] < scores[1]["genome_id"]

    def test_determinism(self):
        """Repeated runs produce identical scores."""
        genomes = [
            {"clip_value": 1.0, "damping": 0.0},
            {"clip_value": 5.0, "damping": 0.3},
        ]
        r1 = run_benchmark_stress(n_vars=10, n_iters=8, genomes=genomes)
        r2 = run_benchmark_stress(n_vars=10, n_iters=8, genomes=genomes)
        j1 = json.dumps(r1["scores"], sort_keys=True)
        j2 = json.dumps(r2["scores"], sort_keys=True)
        assert j1 == j2, "Scores are not deterministic"

    def test_missing_table_raises(self):
        """build_scores raises on missing table."""
        with pytest.raises(ValueError, match="missing 'table'"):
            build_scores({})

    def test_empty_table(self):
        """build_scores returns empty list for empty table."""
        result = {"table": []}
        scores = build_scores(result)
        assert scores == []

    def test_normalization_min_max(self):
        """Min-max normalization produces 0.0 for min and 1.0 for max."""
        # Build a synthetic table with two genomes and one metric
        synthetic_table = [
            {"genome_id": "aaa", "scenario": "sc1", "version": "v1",
             "base_seed_label": "t", "n_vars": 10, "n_iters_base": 8,
             "metric_a": 1.0},
            {"genome_id": "bbb", "scenario": "sc1", "version": "v1",
             "base_seed_label": "t", "n_vars": 10, "n_iters_base": 8,
             "metric_a": 3.0},
        ]
        result = {"table": synthetic_table}
        scores = build_scores(result)
        assert len(scores) == 2
        # Higher metric_a genome gets score 1.0, lower gets 0.0
        score_map = {s["genome_id"]: s["score"] for s in scores}
        assert score_map["bbb"] == pytest.approx(1.0)
        assert score_map["aaa"] == pytest.approx(0.0)
        # bbb ranked first
        assert scores[0]["genome_id"] == "bbb"
        assert scores[0]["rank"] == 1

    def test_constant_column_maps_to_half(self):
        """Constant metric column maps to 0.5 for all genomes."""
        synthetic_table = [
            {"genome_id": "aaa", "scenario": "sc1", "version": "v1",
             "base_seed_label": "t", "n_vars": 10, "n_iters_base": 8,
             "metric_a": 5.0},
            {"genome_id": "bbb", "scenario": "sc1", "version": "v1",
             "base_seed_label": "t", "n_vars": 10, "n_iters_base": 8,
             "metric_a": 5.0},
        ]
        result = {"table": synthetic_table}
        scores = build_scores(result)
        for s in scores:
            assert s["score"] == pytest.approx(0.5)
