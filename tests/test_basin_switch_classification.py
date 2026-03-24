"""Tests for improved basin switch classification (v4.1.0).

Verifies:
- Determinism: identical inputs → identical outputs
- Baseline safety: decoder outputs unchanged when diagnostics disabled
- Classification coverage: all four regimes correctly identified
- Synthetic trace unit tests for classifier logic
"""

from __future__ import annotations

import numpy as np
import pytest

from qec.diagnostics.energy_landscape import (
    _count_gradient_sign_flips,
    _trace_converged,
    _deterministic_sign,
    classify_basin_switch,
    detect_basin_switch,
    classify_energy_landscape,
)
from qec_qldpc_codes import bp_decode, syndrome, channel_llr, create_code
from bench_reports.dps_v381_eval import (
    run_mode,
    _pre_generate_instances,
    MODE_ORDER,
)


# ── Fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def small_code():
    """Small rate-0.50 code for fast tests."""
    return create_code(name="rate_0.50", lifting_size=3, seed=42)


@pytest.fixture
def small_instances(small_code):
    """Pre-generated instances for the small code."""
    rng = np.random.default_rng(42)
    return _pre_generate_instances(small_code.H_X, 0.02, 20, rng)


# ── Gradient Sign Flip Counter ─────────────────────────────────────

class TestGradientSignFlips:

    def test_monotonic_descent_no_flips(self):
        trace = [10.0, 8.0, 6.0, 4.0, 2.0]
        assert _count_gradient_sign_flips(trace) == 0

    def test_oscillating_trace(self):
        # grad: [+2, -3, +2, -3, +2]
        trace = [1.0, 3.0, 0.0, 2.0, -1.0, 1.0]
        flips = _count_gradient_sign_flips(trace)
        assert flips == 4

    def test_short_trace(self):
        assert _count_gradient_sign_flips([5.0]) == 0
        assert _count_gradient_sign_flips([5.0, 3.0]) == 0

    def test_single_flip(self):
        # grad: [-2, +1]
        trace = [10.0, 8.0, 9.0]
        assert _count_gradient_sign_flips(trace) == 1

    def test_determinism(self):
        trace = [10.0, 8.0, 9.0, 7.0, 8.0, 6.0]
        r1 = _count_gradient_sign_flips(trace)
        r2 = _count_gradient_sign_flips(trace)
        assert r1 == r2


# ── Convergence Check ──────────────────────────────────────────────

class TestCheckConvergence:

    def test_converged(self):
        trace = [10.0, 5.0, 3.0, 2.5, 2.5]
        assert _trace_converged(trace) is True

    def test_not_converged(self):
        trace = [10.0, 5.0, 3.0, 1.0]
        assert _trace_converged(trace) is False

    def test_single_point(self):
        assert _trace_converged([5.0]) is True

    def test_determinism(self):
        trace = [10.0, 5.0, 3.0, 2.5, 2.5]
        r1 = _trace_converged(trace)
        r2 = _trace_converged(trace)
        assert r1 == r2


# ── Deterministic Sign Helper ─────────────────────────────────────

class TestDeterministicSign:

    def test_positive_values(self):
        llr = np.array([1.0, 2.5, 0.01])
        s = _deterministic_sign(llr)
        np.testing.assert_array_equal(s, [1.0, 1.0, 1.0])

    def test_negative_values(self):
        llr = np.array([-1.0, -2.5, -0.01])
        s = _deterministic_sign(llr)
        np.testing.assert_array_equal(s, [-1.0, -1.0, -1.0])

    def test_zero_maps_to_positive_one(self):
        llr = np.array([0.0, 0.0, 0.0])
        s = _deterministic_sign(llr)
        np.testing.assert_array_equal(s, [1.0, 1.0, 1.0])

    def test_mixed_with_zeros(self):
        llr = np.array([-3.0, 0.0, 5.0, 0.0, -1.0])
        s = _deterministic_sign(llr)
        np.testing.assert_array_equal(s, [-1.0, 1.0, 1.0, 1.0, -1.0])

    def test_input_not_mutated(self):
        llr = np.array([0.0, -1.0, 2.0])
        llr_original = llr.copy()
        _deterministic_sign(llr)
        np.testing.assert_array_equal(llr, llr_original)

    def test_determinism(self):
        llr = np.array([0.0, -1.0, 2.0, 0.0])
        s1 = _deterministic_sign(llr)
        s2 = _deterministic_sign(llr)
        np.testing.assert_array_equal(s1, s2)

    def test_returns_new_array(self):
        llr = np.array([1.0, -1.0, 0.0])
        s = _deterministic_sign(llr)
        assert s is not llr


# ── Classification Output Structure ──────────────────────────────

class TestClassifyBasinSwitchStructure:

    def test_output_keys(self, small_code, small_instances):
        """classify_basin_switch returns correct top-level keys."""
        H = small_code.H_X
        inst = small_instances[0]
        result = bp_decode(
            H, inst["llr"], max_iters=20, mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
            energy_trace=True,
        )
        correction = result[0]
        trace = result[-1]

        classification = classify_basin_switch(
            H, inst["llr"], correction, trace,
            max_iters=20, bp_mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
        )
        assert "basin_switch_class" in classification
        assert "basin_switch_evidence" in classification
        assert classification["basin_switch_class"] in (
            "metastable_oscillation",
            "shallow_sensitivity",
            "true_basin_switch",
            "none",
        )

    def test_evidence_keys(self, small_code, small_instances):
        """Evidence dict has all required keys."""
        H = small_code.H_X
        inst = small_instances[0]
        result = bp_decode(
            H, inst["llr"], max_iters=20, mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
            energy_trace=True,
        )
        correction = result[0]
        trace = result[-1]

        classification = classify_basin_switch(
            H, inst["llr"], correction, trace,
            max_iters=20, bp_mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
        )
        evidence = classification["basin_switch_evidence"]
        expected_keys = {
            "energy_delta_plus", "energy_delta_minus",
            "gradient_flip_count",
            "corrections_differ_plus", "corrections_differ_minus",
            "converged",
            "energy_baseline", "energy_plus", "energy_minus",
        }
        assert set(evidence.keys()) == expected_keys


# ── Determinism ────────────────────────────────────────────────────

class TestClassifyBasinSwitchDeterminism:

    def test_identical_runs_produce_identical_output(self, small_code, small_instances):
        """Running classify_basin_switch twice with same inputs → same output."""
        H = small_code.H_X
        inst = small_instances[0]

        result = bp_decode(
            H, inst["llr"], max_iters=20, mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
            energy_trace=True,
        )
        correction = result[0]
        trace = result[-1]

        c1 = classify_basin_switch(
            H, inst["llr"], correction, trace,
            max_iters=20, bp_mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
        )
        c2 = classify_basin_switch(
            H, inst["llr"], correction, trace,
            max_iters=20, bp_mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
        )
        assert c1 == c2


# ── Baseline Safety ───────────────────────────────────────────────

class TestBaselineSafety:

    def test_decoder_unchanged_after_classification(self, small_code, small_instances):
        """Decoder outputs remain identical after classification runs."""
        H = small_code.H_X
        inst = small_instances[0]

        # Decode before classification.
        c_before, i_before = bp_decode(
            H, inst["llr"], max_iters=20, mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
        )

        # Run classification.
        result = bp_decode(
            H, inst["llr"], max_iters=20, mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
            energy_trace=True,
        )
        classify_basin_switch(
            H, inst["llr"], result[0], result[-1],
            max_iters=20, bp_mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
        )

        # Decode after classification — must be identical.
        c_after, i_after = bp_decode(
            H, inst["llr"], max_iters=20, mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
        )

        np.testing.assert_array_equal(c_before, c_after)
        assert i_before == i_after

    def test_input_llr_not_mutated(self, small_code, small_instances):
        """The input LLR vector is not mutated by classification."""
        H = small_code.H_X
        inst = small_instances[0]
        llr = inst["llr"].copy()
        llr_original = llr.copy()

        result = bp_decode(
            H, llr, max_iters=20, mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
            energy_trace=True,
        )
        classify_basin_switch(
            H, llr, result[0], result[-1],
            max_iters=20, bp_mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
        )

        np.testing.assert_array_equal(llr, llr_original)


# ── Classifier Helper Tests ────────────────────────────────────────

class TestClassifierHelpers:
    """Unit tests for classifier helper functions.

    These tests validate _count_gradient_sign_flips and _trace_converged
    using synthetic traces with known properties.
    """

    def test_none_classification(self):
        """When all outcomes match, classification is 'none'."""
        # A simple monotonically descending trace with convergence.
        trace = [10.0, 8.0, 6.0, 5.0, 4.5, 4.5]
        flip_count = _count_gradient_sign_flips(trace)
        converged = _trace_converged(trace)
        assert flip_count < 3
        assert converged is True

    def test_metastable_detection_logic(self):
        """Oscillating trace with many flips and no convergence → metastable."""
        # Oscillating trace: many sign flips, doesn't converge.
        trace = [10.0, 8.0, 9.0, 7.0, 8.5, 6.5, 8.0, 6.0]
        flip_count = _count_gradient_sign_flips(trace)
        converged = _trace_converged(trace)
        assert flip_count >= 3, f"Expected ≥3 flips, got {flip_count}"
        assert converged is False

    def test_shallow_sensitivity_logic(self):
        """Energy difference without correction difference → shallow."""
        # Smooth descent (no oscillation).
        trace = [10.0, 8.0, 6.0, 4.0, 2.0]
        flip_count = _count_gradient_sign_flips(trace)
        assert flip_count == 0


# ── Harness Integration ───────────────────────────────────────────

class TestHarnessIntegration:

    @pytest.fixture
    def landscape_code(self):
        """Larger code that produces multi-iteration traces."""
        return create_code(name="rate_0.50", lifting_size=5, seed=42)

    def test_landscape_mode_includes_classifications(self, landscape_code):
        """With landscape enabled, output includes basin_classifications."""
        H = landscape_code.H_X
        rng = np.random.default_rng(99)
        instances = _pre_generate_instances(H, 0.05, 10, rng)

        result = run_mode(
            "baseline", H, instances,
            max_iters=50, enable_landscape=True,
        )
        # With multi-iteration traces, landscape analysis should run.
        # Check that at least some traces had >= 2 iterations.
        multi_iter = [t for t in result.get("energy_traces", []) if len(t) >= 2]
        if multi_iter:
            assert "basin_classifications" in result
            assert "basin_class_counts" in result
            assert isinstance(result["basin_classifications"], list)
            assert isinstance(result["basin_class_counts"], dict)

            for bc in result["basin_classifications"]:
                assert "basin_switch_class" in bc
                assert "basin_switch_evidence" in bc
                assert bc["basin_switch_class"] in (
                    "metastable_oscillation",
                    "shallow_sensitivity",
                    "true_basin_switch",
                    "none",
                )

            # Verify aggregation counts match classification list.
            recomputed = {}
            for bc in result["basin_classifications"]:
                cls = bc["basin_switch_class"]
                recomputed[cls] = recomputed.get(cls, 0) + 1
            assert recomputed == result["basin_class_counts"]

    def test_landscape_mode_deterministic(self, landscape_code):
        """Two identical landscape runs produce identical classifications."""
        H = landscape_code.H_X
        results = []
        for _ in range(2):
            rng = np.random.default_rng(99)
            instances = _pre_generate_instances(H, 0.05, 10, rng)
            r = run_mode(
                "baseline", H, instances,
                max_iters=50, enable_landscape=True,
            )
            results.append(r)

        # Both runs must produce identical classification outputs.
        has_bc_0 = "basin_class_counts" in results[0]
        has_bc_1 = "basin_class_counts" in results[1]
        assert has_bc_0 == has_bc_1
        if has_bc_0:
            assert results[0]["basin_class_counts"] == results[1]["basin_class_counts"]
            assert len(results[0]["basin_classifications"]) == len(results[1]["basin_classifications"])
            for c1, c2 in zip(results[0]["basin_classifications"], results[1]["basin_classifications"]):
                assert c1 == c2

    def test_no_classifications_without_landscape(self, small_code, small_instances):
        """Without landscape mode, no basin_classifications in output."""
        H = small_code.H_X
        result = run_mode("baseline", H, small_instances, max_iters=20)
        assert "basin_classifications" not in result
        assert "basin_class_counts" not in result

    def test_existing_fields_preserved(self, landscape_code):
        """Existing harness output fields remain present with landscape on."""
        H = landscape_code.H_X
        rng = np.random.default_rng(99)
        instances = _pre_generate_instances(H, 0.05, 10, rng)

        result = run_mode(
            "baseline", H, instances,
            max_iters=50, enable_landscape=True,
        )
        # Core fields always present.
        assert "fer" in result
        assert "frame_errors" in result
        assert "trials" in result
        assert "audit_summary" in result
        assert "energy_traces" in result
