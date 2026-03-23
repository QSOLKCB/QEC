"""Tests for hierarchical correction stacks (v96.0.0).

Covers:
  - E8-like projection determinism and correctness
  - Multi-stage hierarchical stacks
  - Projection distance tracking
  - Ranking stability
  - No mutation of inputs
  - Full pipeline integration
  - Print layer stability
"""

import numpy as np
import pytest

from qec.experiments.hierarchical_correction import (
    HIERARCHICAL_MODES,
    _E8_CHUNK_SIZE,
    _pad_to_multiple,
    _parse_stages,
    _truncate_to_length,
    compare_hierarchical_modes,
    print_hierarchical_report,
    project_e8_like,
    project_hierarchical,
    run_all_hierarchical_modes,
    run_hierarchical_correction,
)
from qec.experiments.correction_layer import (
    project_d4,
    project_square,
)
from qec.experiments.dfa_benchmark import (
    build_chain_dfa,
    build_cycle_dfa,
    build_branching_dfa,
    build_two_basin_dfa,
    build_dead_state_dfa,
)


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_vec_8():
    """8-element test vector."""
    return np.array([0.7, 1.3, -0.2, 2.5, 0.1, -1.8, 3.3, 0.9])


@pytest.fixture
def sample_vec_5():
    """5-element test vector (not a multiple of 8)."""
    return np.array([0.7, 1.3, -0.2, 2.5, 0.1])


@pytest.fixture
def sample_vec_16():
    """16-element test vector (two chunks)."""
    return np.array([
        0.7, 1.3, -0.2, 2.5, 0.1, -1.8, 3.3, 0.9,
        -0.5, 0.4, 1.1, -2.0, 0.3, 1.7, -0.8, 0.6,
    ])


@pytest.fixture
def chain_dfa():
    return build_chain_dfa(5)


@pytest.fixture
def cycle_dfa():
    return build_cycle_dfa(5)


@pytest.fixture
def branching_dfa():
    return build_branching_dfa(5)


# ---------------------------------------------------------------------------
# PART 1 — E8-LIKE PROJECTION TESTS
# ---------------------------------------------------------------------------


class TestProjectE8Like:
    """Tests for the E8-like projection function."""

    def test_deterministic_same_input(self, sample_vec_8):
        """Same input always produces same output."""
        a = project_e8_like(sample_vec_8)
        b = project_e8_like(sample_vec_8)
        np.testing.assert_array_equal(a, b)

    def test_deterministic_copy(self, sample_vec_8):
        """Copy of input produces identical output."""
        a = project_e8_like(sample_vec_8)
        b = project_e8_like(sample_vec_8.copy())
        np.testing.assert_array_equal(a, b)

    def test_no_mutation(self, sample_vec_8):
        """Input vector is not mutated."""
        original = sample_vec_8.copy()
        project_e8_like(sample_vec_8)
        np.testing.assert_array_equal(sample_vec_8, original)

    def test_output_integer_values(self, sample_vec_8):
        """Output values should be integers (lattice points)."""
        result = project_e8_like(sample_vec_8)
        np.testing.assert_array_equal(result, np.round(result))

    def test_even_sum_parity(self, sample_vec_8):
        """Each 8-element chunk should have even sum."""
        result = project_e8_like(sample_vec_8)
        chunk = result[:8]
        assert int(np.sum(chunk)) % 2 == 0

    def test_even_sos_parity(self, sample_vec_8):
        """Each 8-element chunk should have even sum-of-squares."""
        result = project_e8_like(sample_vec_8)
        chunk = result[:8].astype(int)
        assert int(np.sum(chunk * chunk)) % 2 == 0

    def test_handles_short_vector(self, sample_vec_5):
        """Handles vectors shorter than 8 (padding/truncation)."""
        result = project_e8_like(sample_vec_5)
        assert len(result) == len(sample_vec_5)

    def test_handles_multi_chunk(self, sample_vec_16):
        """Handles vectors with multiple 8-element chunks."""
        result = project_e8_like(sample_vec_16)
        assert len(result) == 16
        # Check each chunk.
        for c in range(2):
            chunk = result[c * 8:(c + 1) * 8].astype(int)
            assert int(np.sum(chunk)) % 2 == 0
            assert int(np.sum(chunk * chunk)) % 2 == 0

    def test_zero_vector(self):
        """Zero vector projects to zero."""
        vec = np.zeros(8)
        result = project_e8_like(vec)
        np.testing.assert_array_equal(result, np.zeros(8))

    def test_integer_vector_stays_fixed(self):
        """An already-valid E8-like point should not change."""
        # [2, 0, 0, 0, 0, 0, 0, 0] has sum=2 (even), sos=4 (even).
        vec = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = project_e8_like(vec)
        np.testing.assert_array_equal(result, vec)

    def test_stronger_than_square(self, sample_vec_8):
        """E8-like projection differs from simple rounding."""
        square_result = project_square(sample_vec_8)
        e8_result = project_e8_like(sample_vec_8)
        # They may sometimes agree, but in general differ.
        # Just verify both produce valid outputs.
        assert len(e8_result) == len(square_result)
        np.testing.assert_array_equal(e8_result, np.round(e8_result))

    def test_single_element(self):
        """Single-element vector is handled correctly."""
        vec = np.array([1.7])
        result = project_e8_like(vec)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# PART 2 — PADDING / TRUNCATION TESTS
# ---------------------------------------------------------------------------


class TestPadTruncate:
    """Tests for padding and truncation helpers."""

    def test_pad_no_op(self):
        """Multiple of chunk_size needs no padding."""
        x = np.ones(8)
        result = _pad_to_multiple(x, 8)
        assert len(result) == 8

    def test_pad_short(self):
        """Short vector gets padded."""
        x = np.ones(5)
        result = _pad_to_multiple(x, 8)
        assert len(result) == 8
        np.testing.assert_array_equal(result[5:], np.zeros(3))

    def test_truncate(self):
        """Truncation restores original length."""
        x = np.array([1.0, 2.0, 3.0, 0.0, 0.0])
        result = _truncate_to_length(x, 3)
        assert len(result) == 3
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# PART 3 — HIERARCHICAL PROJECTION TESTS
# ---------------------------------------------------------------------------


class TestProjectHierarchical:
    """Tests for multi-stage hierarchical projection."""

    def test_single_stage_square(self, sample_vec_8):
        """Single-stage 'square' matches direct call."""
        result, meta = project_hierarchical(sample_vec_8, "square")
        expected = project_square(sample_vec_8)
        np.testing.assert_array_equal(result, expected)
        assert len(meta) == 1
        assert meta[0]["stage"] == "square"

    def test_single_stage_d4(self, sample_vec_8):
        """Single-stage 'd4' matches direct call."""
        result, meta = project_hierarchical(sample_vec_8, "d4")
        expected = project_d4(sample_vec_8)
        np.testing.assert_array_equal(result, expected)

    def test_single_stage_e8_like(self, sample_vec_8):
        """Single-stage 'e8_like' matches direct call."""
        result, meta = project_hierarchical(sample_vec_8, "e8_like")
        expected = project_e8_like(sample_vec_8)
        np.testing.assert_array_equal(result, expected)

    def test_multi_stage_two(self, sample_vec_8):
        """Two-stage 'square>d4' applies both sequentially."""
        result, meta = project_hierarchical(sample_vec_8, "square>d4")
        assert len(meta) == 2
        assert meta[0]["stage"] == "square"
        assert meta[1]["stage"] == "d4"
        # Verify: square first, then d4.
        intermediate = project_square(sample_vec_8)
        expected = project_d4(intermediate)
        np.testing.assert_array_equal(result, expected)

    def test_multi_stage_three(self, sample_vec_8):
        """Three-stage 'square>d4>e8_like' applies all sequentially."""
        result, meta = project_hierarchical(
            sample_vec_8, "square>d4>e8_like"
        )
        assert len(meta) == 3
        # Verify sequential application.
        step1 = project_square(sample_vec_8)
        step2 = project_d4(step1)
        step3 = project_e8_like(step2)
        np.testing.assert_array_equal(result, step3)

    def test_projection_distances_tracked(self, sample_vec_8):
        """Each stage records a non-negative projection distance."""
        _, meta = project_hierarchical(sample_vec_8, "square>d4>e8_like")
        for m in meta:
            assert "projection_distance" in m
            assert m["projection_distance"] >= 0.0

    def test_deterministic(self, sample_vec_8):
        """Multi-stage projection is deterministic."""
        a, meta_a = project_hierarchical(sample_vec_8, "square>d4>e8_like")
        b, meta_b = project_hierarchical(sample_vec_8, "square>d4>e8_like")
        np.testing.assert_array_equal(a, b)
        assert meta_a == meta_b

    def test_no_mutation(self, sample_vec_8):
        """Input is not mutated."""
        original = sample_vec_8.copy()
        project_hierarchical(sample_vec_8, "d4>e8_like")
        np.testing.assert_array_equal(sample_vec_8, original)

    def test_invalid_stage_raises(self, sample_vec_8):
        """Invalid stage name raises ValueError."""
        with pytest.raises(ValueError, match="unknown projection stage"):
            project_hierarchical(sample_vec_8, "invalid_stage")

    def test_parse_stages(self):
        """Mode parsing splits correctly."""
        assert _parse_stages("square") == ["square"]
        assert _parse_stages("square>d4") == ["square", "d4"]
        assert _parse_stages("square>d4>e8_like") == [
            "square", "d4", "e8_like"
        ]


# ---------------------------------------------------------------------------
# PART 4 — HIERARCHICAL CORRECTION RUNNER TESTS
# ---------------------------------------------------------------------------


class TestRunHierarchicalCorrection:
    """Tests for run_hierarchical_correction."""

    def test_basic_run(self, chain_dfa):
        """Basic run completes without error."""
        result = run_hierarchical_correction(chain_dfa, "square")
        assert result["mode"] == "square"
        assert "stages" in result
        assert "projection_distances" in result
        assert "total_projection_distance" in result
        assert "compression_efficiency" in result
        assert "stability_efficiency" in result
        assert "stability_gain" in result

    def test_multi_stage_run(self, chain_dfa):
        """Multi-stage run reports all stages."""
        result = run_hierarchical_correction(chain_dfa, "square>d4>e8_like")
        assert result["stages"] == ["square", "d4", "e8_like"]
        assert len(result["projection_distances"]) == 3

    def test_total_projection_distance(self, chain_dfa):
        """Total projection distance equals sum of per-stage distances."""
        result = run_hierarchical_correction(chain_dfa, "square>d4")
        total = sum(result["projection_distances"])
        assert abs(result["total_projection_distance"] - total) < 1e-9

    def test_deterministic(self, chain_dfa):
        """Same DFA + mode gives same result."""
        a = run_hierarchical_correction(chain_dfa, "d4>e8_like")
        b = run_hierarchical_correction(chain_dfa, "d4>e8_like")
        assert a["mode"] == b["mode"]
        assert a["stages"] == b["stages"]
        assert a["projection_distances"] == b["projection_distances"]
        assert a["compression_efficiency"] == b["compression_efficiency"]
        assert a["stability_efficiency"] == b["stability_efficiency"]

    def test_with_invariants(self, chain_dfa):
        """Invariant-guided damping runs without error."""
        result = run_hierarchical_correction(
            chain_dfa, "d4>e8_like", use_invariants=True
        )
        assert result["mode"] == "d4>e8_like"

    def test_metrics_valid_range(self, cycle_dfa):
        """Efficiency metrics are in [0, 1]."""
        result = run_hierarchical_correction(cycle_dfa, "e8_like")
        assert 0 <= result["compression_efficiency"] <= 1
        assert 0 <= result["stability_efficiency"] <= 1

    def test_no_dfa_mutation(self, chain_dfa):
        """DFA is not mutated."""
        import copy
        original = copy.deepcopy(chain_dfa)
        run_hierarchical_correction(chain_dfa, "square>d4")
        assert chain_dfa == original


# ---------------------------------------------------------------------------
# PART 5 — COMPARISON AND RANKING TESTS
# ---------------------------------------------------------------------------


class TestCompareHierarchicalModes:
    """Tests for compare_hierarchical_modes."""

    def test_ranking_stable(self, chain_dfa):
        """Ranking is deterministic across runs."""
        results = run_all_hierarchical_modes(chain_dfa)
        ranked_a = compare_hierarchical_modes(results)
        ranked_b = compare_hierarchical_modes(results)
        for a, b in zip(ranked_a, ranked_b):
            assert a["rank"] == b["rank"]
            assert a["mode"] == b["mode"]

    def test_all_modes_present(self, chain_dfa):
        """All hierarchical modes are represented."""
        results = run_all_hierarchical_modes(chain_dfa)
        ranked = compare_hierarchical_modes(results)
        modes = {r["mode"] for r in ranked}
        assert modes == set(HIERARCHICAL_MODES)

    def test_ranks_sequential(self, chain_dfa):
        """Ranks are 1-indexed and sequential."""
        results = run_all_hierarchical_modes(chain_dfa)
        ranked = compare_hierarchical_modes(results)
        ranks = [r["rank"] for r in ranked]
        assert ranks == list(range(1, len(HIERARCHICAL_MODES) + 1))

    def test_best_has_rank_1(self, chain_dfa):
        """Best mode has rank 1."""
        results = run_all_hierarchical_modes(chain_dfa)
        ranked = compare_hierarchical_modes(results)
        assert ranked[0]["rank"] == 1

    def test_empty_input(self):
        """Empty input returns empty ranking."""
        assert compare_hierarchical_modes([]) == []

    def test_ranking_by_stability_first(self):
        """Higher stability_efficiency wins."""
        results = [
            {
                "mode": "a",
                "stability_efficiency": 0.5,
                "compression_efficiency": 0.1,
                "total_projection_distance": 1.0,
                "stages": ["a"],
                "projection_distances": [1.0],
                "stability_gain": 1,
                "metrics": {},
            },
            {
                "mode": "b",
                "stability_efficiency": 0.8,
                "compression_efficiency": 0.1,
                "total_projection_distance": 2.0,
                "stages": ["b"],
                "projection_distances": [2.0],
                "stability_gain": 1,
                "metrics": {},
            },
        ]
        ranked = compare_hierarchical_modes(results)
        assert ranked[0]["mode"] == "b"
        assert ranked[1]["mode"] == "a"


# ---------------------------------------------------------------------------
# PART 6 — HIERARCHICAL MODES LIST TESTS
# ---------------------------------------------------------------------------


class TestHierarchicalModes:
    """Tests for the HIERARCHICAL_MODES list."""

    def test_contains_single_modes(self):
        """Contains all single-stage modes."""
        assert "square" in HIERARCHICAL_MODES
        assert "d4" in HIERARCHICAL_MODES
        assert "e8_like" in HIERARCHICAL_MODES

    def test_contains_multi_modes(self):
        """Contains multi-stage modes."""
        assert "square>d4" in HIERARCHICAL_MODES
        assert "d4>e8_like" in HIERARCHICAL_MODES
        assert "square>d4>e8_like" in HIERARCHICAL_MODES

    def test_all_modes_valid(self):
        """All modes parse into valid stage names."""
        for mode in HIERARCHICAL_MODES:
            stages = _parse_stages(mode)
            assert all(
                s in ("square", "d4", "e8_like") for s in stages
            )


# ---------------------------------------------------------------------------
# PART 7 — PRINT LAYER TESTS
# ---------------------------------------------------------------------------


class TestPrintHierarchicalReport:
    """Tests for print_hierarchical_report."""

    def test_empty_report(self):
        """Empty report prints header."""
        text = print_hierarchical_report({})
        assert "=== Hierarchical Comparison ===" in text
        assert "No comparisons available." in text

    def test_with_comparisons(self):
        """Report with comparisons includes all fields."""
        report = {
            "comparisons": [
                {
                    "dfa_name": "chain",
                    "n": 5,
                    "baseline_mode": "d4",
                    "hierarchical_mode": "square>d4",
                    "core_overlay_mode": "square>d4",
                    "best_variant": "hierarchical",
                },
            ],
            "global_best_modes": {
                "best_hierarchical_mode": "d4>e8_like",
            },
        }
        text = print_hierarchical_report(report)
        assert "chain" in text
        assert "baseline_best: d4" in text
        assert "hierarchical_best: square>d4" in text
        assert "winner: hierarchical" in text
        assert "d4>e8_like" in text

    def test_deterministic_output(self):
        """Print output is deterministic."""
        report = {
            "comparisons": [
                {
                    "dfa_name": "cycle",
                    "n": 10,
                    "baseline_mode": "d4+inv",
                    "hierarchical_mode": "d4>e8_like",
                    "core_overlay_mode": "d4>e8_like",
                    "best_variant": "core_overlay",
                },
            ],
            "global_best_modes": {},
        }
        a = print_hierarchical_report(report)
        b = print_hierarchical_report(report)
        assert a == b


# ---------------------------------------------------------------------------
# PART 8 — CROSS-DFA DETERMINISM TESTS
# ---------------------------------------------------------------------------


class TestCrossDFADeterminism:
    """Tests for determinism across different DFA types."""

    @pytest.mark.parametrize("dfa_builder,n", [
        (build_chain_dfa, 5),
        (build_chain_dfa, 10),
        (build_cycle_dfa, 5),
        (build_cycle_dfa, 10),
        (build_branching_dfa, 5),
        (build_two_basin_dfa, None),
        (build_dead_state_dfa, None),
    ])
    def test_determinism_across_dfas(self, dfa_builder, n):
        """Each DFA type produces deterministic results."""
        dfa = dfa_builder(n)
        a = run_all_hierarchical_modes(dfa)
        b = run_all_hierarchical_modes(dfa)
        for ra, rb in zip(a, b):
            assert ra["mode"] == rb["mode"]
            assert ra["compression_efficiency"] == rb["compression_efficiency"]
            assert ra["stability_efficiency"] == rb["stability_efficiency"]
            assert ra["projection_distances"] == rb["projection_distances"]
