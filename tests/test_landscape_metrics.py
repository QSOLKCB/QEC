"""Tests for v4.2.0 deterministic landscape metrics (BSI, AD, EE).

Verifies:
- Basin Stability Index calculation
- Hamming distance correctness
- Escape energy sweep detection
- Directional escape energies
- Deterministic results across repeated runs
- Baseline safety: decoder outputs unchanged
- Input LLR vectors not mutated
"""

from __future__ import annotations

import numpy as np
import pytest

from qec.diagnostics.energy_landscape import (
    _hamming_distance,
    compute_basin_stability_index,
    compute_attractor_distance,
    compute_escape_energy,
    compute_landscape_metrics,
    _ESCAPE_EPSILON_VALUES,
)
from qec_qldpc_codes import bp_decode, syndrome, channel_llr, create_code
from bench_reports.dps_v381_eval import (
    run_mode,
    _pre_generate_instances,
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


# ── Hamming Distance ─────────────────────────────────────────────

class TestHammingDistance:

    def test_identical_arrays(self):
        a = np.array([0, 1, 0, 1, 1], dtype=np.uint8)
        assert _hamming_distance(a, a) == 0

    def test_completely_different(self):
        a = np.array([0, 0, 0], dtype=np.uint8)
        b = np.array([1, 1, 1], dtype=np.uint8)
        assert _hamming_distance(a, b) == 3

    def test_partial_difference(self):
        a = np.array([0, 1, 0, 1], dtype=np.uint8)
        b = np.array([0, 0, 0, 1], dtype=np.uint8)
        assert _hamming_distance(a, b) == 1

    def test_determinism(self):
        a = np.array([1, 0, 1, 0, 1], dtype=np.uint8)
        b = np.array([0, 0, 1, 1, 1], dtype=np.uint8)
        r1 = _hamming_distance(a, b)
        r2 = _hamming_distance(a, b)
        assert r1 == r2

    def test_input_not_mutated(self):
        a = np.array([0, 1, 0], dtype=np.uint8)
        b = np.array([1, 1, 0], dtype=np.uint8)
        a_orig = a.copy()
        b_orig = b.copy()
        _hamming_distance(a, b)
        np.testing.assert_array_equal(a, a_orig)
        np.testing.assert_array_equal(b, b_orig)


# ── Basin Stability Index ────────────────────────────────────────

class TestBasinStabilityIndex:

    def test_fully_stable(self):
        """All perturbations return same correction → BSI = 1.0."""
        c = np.array([0, 1, 0, 1], dtype=np.uint8)
        bsi = compute_basin_stability_index(c, c.copy(), c.copy())
        assert bsi == 1.0

    def test_fully_unstable(self):
        """Both perturbations differ → BSI = 0.0."""
        c_base = np.array([0, 1, 0, 1], dtype=np.uint8)
        c_plus = np.array([1, 1, 0, 1], dtype=np.uint8)
        c_minus = np.array([0, 0, 0, 1], dtype=np.uint8)
        bsi = compute_basin_stability_index(c_base, c_plus, c_minus)
        assert bsi == 0.0

    def test_half_stable(self):
        """One perturbation matches, one differs → BSI = 0.5."""
        c_base = np.array([0, 1, 0, 1], dtype=np.uint8)
        c_plus = c_base.copy()  # matches
        c_minus = np.array([1, 1, 0, 1], dtype=np.uint8)  # differs
        bsi = compute_basin_stability_index(c_base, c_plus, c_minus)
        assert bsi == 0.5

    def test_returns_float(self):
        c = np.array([0, 1], dtype=np.uint8)
        bsi = compute_basin_stability_index(c, c.copy(), c.copy())
        assert isinstance(bsi, float)

    def test_determinism(self):
        c_base = np.array([0, 1, 0, 1], dtype=np.uint8)
        c_plus = np.array([1, 1, 0, 1], dtype=np.uint8)
        c_minus = c_base.copy()
        r1 = compute_basin_stability_index(c_base, c_plus, c_minus)
        r2 = compute_basin_stability_index(c_base, c_plus, c_minus)
        assert r1 == r2


# ── Attractor Distance ───────────────────────────────────────────

class TestAttractorDistance:

    def test_identical_corrections(self):
        """No attractor distance when all corrections match."""
        c = np.array([0, 1, 0, 1], dtype=np.uint8)
        ad = compute_attractor_distance(c, c.copy(), c.copy())
        assert ad["attractor_distance_max"] == 0
        assert ad["attractor_distance_mean"] == 0.0

    def test_asymmetric_distances(self):
        c_base = np.array([0, 0, 0, 0], dtype=np.uint8)
        c_plus = np.array([1, 0, 0, 0], dtype=np.uint8)   # distance 1
        c_minus = np.array([1, 1, 1, 0], dtype=np.uint8)  # distance 3
        ad = compute_attractor_distance(c_base, c_plus, c_minus)
        assert ad["attractor_distance_max"] == 3
        assert ad["attractor_distance_mean"] == 2.0

    def test_output_types(self):
        c = np.array([0, 1], dtype=np.uint8)
        ad = compute_attractor_distance(c, c.copy(), c.copy())
        assert isinstance(ad["attractor_distance_max"], int)
        assert isinstance(ad["attractor_distance_mean"], float)

    def test_determinism(self):
        c_base = np.array([0, 1, 0, 1], dtype=np.uint8)
        c_plus = np.array([1, 1, 0, 0], dtype=np.uint8)
        c_minus = np.array([0, 0, 1, 1], dtype=np.uint8)
        r1 = compute_attractor_distance(c_base, c_plus, c_minus)
        r2 = compute_attractor_distance(c_base, c_plus, c_minus)
        assert r1 == r2


# ── Escape Energy ────────────────────────────────────────────────

class TestEscapeEnergy:

    def test_output_keys(self, small_code, small_instances):
        """Escape energy returns correct keys."""
        H = small_code.H_X
        inst = small_instances[0]
        result = bp_decode(
            H, inst["llr"], max_iters=20, mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
        )
        correction = result[0]

        ee = compute_escape_energy(
            H, inst["llr"], correction,
            max_iters=20, bp_mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
        )
        assert "escape_energy" in ee
        assert "escape_energy_plus" in ee
        assert "escape_energy_minus" in ee

    def test_values_are_none_or_float(self, small_code, small_instances):
        """Escape energy values are either None or float from the sweep."""
        H = small_code.H_X
        inst = small_instances[0]
        result = bp_decode(
            H, inst["llr"], max_iters=20, mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
        )
        correction = result[0]

        ee = compute_escape_energy(
            H, inst["llr"], correction,
            max_iters=20, bp_mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
        )
        for key in ("escape_energy", "escape_energy_plus", "escape_energy_minus"):
            val = ee[key]
            assert val is None or isinstance(val, float)
            if val is not None:
                assert val in _ESCAPE_EPSILON_VALUES

    def test_directional_minimum(self, small_code, small_instances):
        """escape_energy is min of directional values when both exist."""
        H = small_code.H_X
        inst = small_instances[0]
        result = bp_decode(
            H, inst["llr"], max_iters=20, mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
        )
        correction = result[0]

        ee = compute_escape_energy(
            H, inst["llr"], correction,
            max_iters=20, bp_mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
        )
        if ee["escape_energy_plus"] is not None and ee["escape_energy_minus"] is not None:
            assert ee["escape_energy"] == min(
                ee["escape_energy_plus"], ee["escape_energy_minus"]
            )
        elif ee["escape_energy_plus"] is not None:
            assert ee["escape_energy"] == ee["escape_energy_plus"]
        elif ee["escape_energy_minus"] is not None:
            assert ee["escape_energy"] == ee["escape_energy_minus"]
        else:
            assert ee["escape_energy"] is None

    def test_determinism(self, small_code, small_instances):
        """Escape energy is deterministic across runs."""
        H = small_code.H_X
        inst = small_instances[0]
        result = bp_decode(
            H, inst["llr"], max_iters=20, mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
        )
        correction = result[0]

        ee1 = compute_escape_energy(
            H, inst["llr"], correction,
            max_iters=20, bp_mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
        )
        ee2 = compute_escape_energy(
            H, inst["llr"], correction,
            max_iters=20, bp_mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
        )
        assert ee1 == ee2

    def test_input_not_mutated(self, small_code, small_instances):
        """Input LLR is not mutated by escape energy computation."""
        H = small_code.H_X
        inst = small_instances[0]
        llr = inst["llr"].copy()
        llr_orig = llr.copy()

        result = bp_decode(
            H, llr, max_iters=20, mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
        )
        compute_escape_energy(
            H, llr, result[0],
            max_iters=20, bp_mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
        )
        np.testing.assert_array_equal(llr, llr_orig)

    def test_custom_eps_values(self, small_code, small_instances):
        """Custom eps_values are accepted and produce valid results."""
        H = small_code.H_X
        inst = small_instances[0]
        result = bp_decode(
            H, inst["llr"], max_iters=20, mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
        )
        correction = result[0]

        custom_eps = [0.01, 0.02]
        ee = compute_escape_energy(
            H, inst["llr"], correction,
            max_iters=20, bp_mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
            eps_values=custom_eps,
        )
        assert "escape_energy" in ee
        assert "escape_energy_plus" in ee
        assert "escape_energy_minus" in ee
        for key in ("escape_energy", "escape_energy_plus", "escape_energy_minus"):
            val = ee[key]
            assert val is None or isinstance(val, float)
            if val is not None:
                assert val in custom_eps

    def test_empty_eps_values(self, small_code, small_instances):
        """Empty eps_values list produces all-None escape energies."""
        H = small_code.H_X
        inst = small_instances[0]
        result = bp_decode(
            H, inst["llr"], max_iters=20, mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
        )
        correction = result[0]

        ee = compute_escape_energy(
            H, inst["llr"], correction,
            max_iters=20, bp_mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
            eps_values=[],
        )
        assert ee["escape_energy"] is None
        assert ee["escape_energy_plus"] is None
        assert ee["escape_energy_minus"] is None


# ── Composite Landscape Metrics ──────────────────────────────────

class TestComputeLandscapeMetrics:

    def test_output_keys(self, small_code, small_instances):
        """compute_landscape_metrics returns all required keys."""
        H = small_code.H_X
        inst = small_instances[0]
        result = bp_decode(
            H, inst["llr"], max_iters=20, mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
            energy_trace=True,
        )
        correction = result[0]
        trace = result[-1]

        metrics = compute_landscape_metrics(
            H, inst["llr"], correction, trace,
            max_iters=20, bp_mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
        )
        expected_keys = {
            "basin_switch_class",
            "basin_switch_evidence",
            "basin_stability_index",
            "attractor_distance_max",
            "attractor_distance_mean",
            "escape_energy",
            "escape_energy_plus",
            "escape_energy_minus",
        }
        assert set(metrics.keys()) == expected_keys

    def test_basin_switch_class_valid(self, small_code, small_instances):
        """basin_switch_class is one of the four regimes."""
        H = small_code.H_X
        inst = small_instances[0]
        result = bp_decode(
            H, inst["llr"], max_iters=20, mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
            energy_trace=True,
        )
        metrics = compute_landscape_metrics(
            H, inst["llr"], result[0], result[-1],
            max_iters=20, bp_mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
        )
        assert metrics["basin_switch_class"] in (
            "metastable_oscillation",
            "shallow_sensitivity",
            "true_basin_switch",
            "none",
        )

    def test_bsi_range(self, small_code, small_instances):
        """BSI is in [0.0, 1.0]."""
        H = small_code.H_X
        inst = small_instances[0]
        result = bp_decode(
            H, inst["llr"], max_iters=20, mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
            energy_trace=True,
        )
        metrics = compute_landscape_metrics(
            H, inst["llr"], result[0], result[-1],
            max_iters=20, bp_mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
        )
        assert 0.0 <= metrics["basin_stability_index"] <= 1.0

    def test_ad_non_negative(self, small_code, small_instances):
        """Attractor distances are non-negative."""
        H = small_code.H_X
        inst = small_instances[0]
        result = bp_decode(
            H, inst["llr"], max_iters=20, mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
            energy_trace=True,
        )
        metrics = compute_landscape_metrics(
            H, inst["llr"], result[0], result[-1],
            max_iters=20, bp_mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
        )
        assert metrics["attractor_distance_max"] >= 0
        assert metrics["attractor_distance_mean"] >= 0.0

    def test_determinism(self, small_code, small_instances):
        """Identical runs produce identical results."""
        H = small_code.H_X
        inst = small_instances[0]
        result = bp_decode(
            H, inst["llr"], max_iters=20, mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
            energy_trace=True,
        )
        correction = result[0]
        trace = result[-1]

        m1 = compute_landscape_metrics(
            H, inst["llr"], correction, trace,
            max_iters=20, bp_mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
        )
        m2 = compute_landscape_metrics(
            H, inst["llr"], correction, trace,
            max_iters=20, bp_mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
        )
        assert m1 == m2

    def test_bsi_consistency_with_class(self, small_code, small_instances):
        """When class is 'none', BSI should be 1.0."""
        H = small_code.H_X
        for inst in small_instances:
            result = bp_decode(
                H, inst["llr"], max_iters=20, mode="min_sum",
                schedule="flooding", syndrome_vec=inst["s"],
                energy_trace=True,
            )
            metrics = compute_landscape_metrics(
                H, inst["llr"], result[0], result[-1],
                max_iters=20, bp_mode="min_sum",
                schedule="flooding", syndrome_vec=inst["s"],
            )
            if metrics["basin_switch_class"] == "none":
                assert metrics["basin_stability_index"] == 1.0
                assert metrics["attractor_distance_max"] == 0
                assert metrics["attractor_distance_mean"] == 0.0


# ── Baseline Safety ──────────────────────────────────────────────

class TestLandscapeMetricsBaselineSafety:

    def test_decoder_unchanged_after_metrics(self, small_code, small_instances):
        """Decoder outputs remain identical after landscape metrics."""
        H = small_code.H_X
        inst = small_instances[0]

        c_before, i_before = bp_decode(
            H, inst["llr"], max_iters=20, mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
        )

        result = bp_decode(
            H, inst["llr"], max_iters=20, mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
            energy_trace=True,
        )
        compute_landscape_metrics(
            H, inst["llr"], result[0], result[-1],
            max_iters=20, bp_mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
        )

        c_after, i_after = bp_decode(
            H, inst["llr"], max_iters=20, mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
        )
        np.testing.assert_array_equal(c_before, c_after)
        assert i_before == i_after

    def test_input_llr_not_mutated(self, small_code, small_instances):
        """Input LLR is not mutated by landscape metrics."""
        H = small_code.H_X
        inst = small_instances[0]
        llr = inst["llr"].copy()
        llr_orig = llr.copy()

        result = bp_decode(
            H, llr, max_iters=20, mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
            energy_trace=True,
        )
        compute_landscape_metrics(
            H, llr, result[0], result[-1],
            max_iters=20, bp_mode="min_sum",
            schedule="flooding", syndrome_vec=inst["s"],
        )
        np.testing.assert_array_equal(llr, llr_orig)


# ── Harness Integration ──────────────────────────────────────────

_HARNESS_INSTANCES = 5
_HARNESS_MAX_ITERS = 30


class TestHarnessLandscapeMetrics:

    @pytest.fixture
    def landscape_code(self):
        return create_code(name="rate_0.50", lifting_size=3, seed=42)

    def test_landscape_output_has_new_fields(self, landscape_code):
        """With landscape enabled, classifications include v4.2.0 fields."""
        H = landscape_code.H_X
        rng = np.random.default_rng(99)
        instances = _pre_generate_instances(H, 0.50, _HARNESS_INSTANCES, rng)

        result = run_mode(
            "baseline", H, instances,
            max_iters=_HARNESS_MAX_ITERS, enable_landscape=True,
        )

        # Ensure this test always exercises the new landscape metrics.
        multi_iter = [t for t in result.get("energy_traces", []) if len(t) >= 2]
        assert multi_iter, "Expected at least one multi-iteration energy trace when landscape is enabled"

        basin_classifications = result.get("basin_classifications")
        assert basin_classifications, "Expected 'basin_classifications' in landscape result"

        for bc in basin_classifications:
            assert "basin_stability_index" in bc
            assert "attractor_distance_max" in bc
            assert "attractor_distance_mean" in bc
            assert "escape_energy" in bc
            assert "escape_energy_plus" in bc
            assert "escape_energy_minus" in bc

    def test_landscape_deterministic(self, landscape_code):
        """Two identical landscape runs produce identical metrics."""
        H = landscape_code.H_X
        results = []
        for _ in range(2):
            rng = np.random.default_rng(99)
            instances = _pre_generate_instances(H, 0.50, _HARNESS_INSTANCES, rng)
            r = run_mode(
                "baseline", H, instances,
                max_iters=_HARNESS_MAX_ITERS, enable_landscape=True,
            )
            results.append(r)

        # Verify multi-iteration traces exist so assertions are exercised.
        for r in results:
            multi_iter = [t for t in r.get("energy_traces", []) if len(t) >= 2]
            assert multi_iter, "Expected at least one multi-iteration energy trace"

        basin_classifications_0 = results[0].get("basin_classifications")
        basin_classifications_1 = results[1].get("basin_classifications")
        assert basin_classifications_0, "Expected 'basin_classifications' in result"
        assert basin_classifications_1, "Expected 'basin_classifications' in result"

        assert len(basin_classifications_0) == len(basin_classifications_1)
        for c1, c2 in zip(basin_classifications_0, basin_classifications_1):
            assert c1 == c2

        # Verify all landscape metric fields exist in every classification.
        for bc in basin_classifications_0:
            assert "basin_stability_index" in bc
            assert "attractor_distance_max" in bc
            assert "attractor_distance_mean" in bc
            assert "escape_energy" in bc
            assert "escape_energy_plus" in bc
            assert "escape_energy_minus" in bc
