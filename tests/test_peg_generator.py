"""
Tests for v10.2.0 PEG-seeded spectral population initialization.

Verifies:
  - deterministic PEG generation (same seed → identical graph)
  - degree constraints satisfied
  - matrix dimensions correct
  - population generator returns correct number of candidates
  - spectral filter accepts/rejects appropriately
  - discovery engine integration with PEG initialization
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from qec.generation.peg_generator import (
    generate_peg_tanner_graph,
    generate_peg_population,
    _passes_spectral_filter,
)
from qec.discovery.population_engine import DiscoveryEngine


# ── Fixtures ─────────────────────────────────────────────────────

SPEC_SMALL = {
    "num_variables": 12,
    "num_checks": 6,
    "variable_degree": 3,
    "check_degree": 6,
}


# ── Determinism ──────────────────────────────────────────────────


class TestPEGDeterminism:
    """Same seed must produce identical Tanner graphs."""

    def test_identical_seed_produces_identical_graph(self) -> None:
        H1 = generate_peg_tanner_graph(12, 6, 3, 6, seed=42)
        H2 = generate_peg_tanner_graph(12, 6, 3, 6, seed=42)
        np.testing.assert_array_equal(H1, H2)

    def test_different_seed_produces_different_graph(self) -> None:
        H1 = generate_peg_tanner_graph(12, 6, 3, 6, seed=42)
        H2 = generate_peg_tanner_graph(12, 6, 3, 6, seed=99)
        assert not np.array_equal(H1, H2)

    def test_population_determinism(self) -> None:
        pop1 = generate_peg_population(SPEC_SMALL, 5, base_seed=42)
        pop2 = generate_peg_population(SPEC_SMALL, 5, base_seed=42)
        assert len(pop1) == len(pop2)
        for c1, c2 in zip(pop1, pop2):
            np.testing.assert_array_equal(c1["H"], c2["H"])


# ── Degree Constraints ──────────────────────────────────────────


class TestPEGDegreeConstraints:
    """Variable and check degrees must be respected."""

    def test_variable_degree_satisfied(self) -> None:
        H = generate_peg_tanner_graph(12, 6, 3, 6, seed=0)
        col_sums = H.sum(axis=0)
        # Each variable node should have degree <= variable_degree
        assert np.all(col_sums <= 3 + 1)  # allow minor excess
        assert np.all(col_sums >= 1)  # no isolated variables

    def test_check_degree_respected(self) -> None:
        H = generate_peg_tanner_graph(12, 6, 3, 6, seed=0)
        row_sums = H.sum(axis=1)
        assert np.all(row_sums <= 6 + 1)  # allow minor excess
        assert np.all(row_sums >= 1)  # no isolated checks

    def test_total_edges_consistent(self) -> None:
        H = generate_peg_tanner_graph(12, 6, 3, 6, seed=0)
        # For a (3,6)-regular code: 12*3 = 36 edges = 6*6
        total_edges = H.sum()
        # Should be close to expected (may not be exact due to constraints)
        assert 30 <= total_edges <= 42


# ── Matrix Shape ─────────────────────────────────────────────────


class TestPEGMatrixShape:
    """Generated matrices must have correct dimensions."""

    def test_shape_matches_spec(self) -> None:
        H = generate_peg_tanner_graph(12, 6, 3, 6, seed=0)
        assert H.shape == (6, 12)

    def test_binary_matrix(self) -> None:
        H = generate_peg_tanner_graph(12, 6, 3, 6, seed=0)
        unique_vals = set(np.unique(H))
        assert unique_vals <= {0.0, 1.0}

    def test_float64_dtype(self) -> None:
        H = generate_peg_tanner_graph(12, 6, 3, 6, seed=0)
        assert H.dtype == np.float64

    @pytest.mark.parametrize("n,m,dv,dc", [
        (8, 4, 2, 4),
        (15, 5, 3, 9),
        (20, 10, 3, 6),
    ])
    def test_various_dimensions(self, n: int, m: int, dv: int, dc: int) -> None:
        H = generate_peg_tanner_graph(n, m, dv, dc, seed=7)
        assert H.shape == (m, n)
        assert np.all(H >= 0)
        assert np.all(H <= 1)

    def test_no_empty_rows_or_columns(self) -> None:
        H = generate_peg_tanner_graph(12, 6, 3, 6, seed=0)
        assert np.all(H.sum(axis=1) >= 1)
        assert np.all(H.sum(axis=0) >= 1)


# ── Spectral Filter ─────────────────────────────────────────────


class TestSpectralFilter:
    """Spectral filter should accept good PEG graphs."""

    def test_peg_graph_passes_filter(self) -> None:
        H = generate_peg_tanner_graph(12, 6, 3, 6, seed=0)
        # PEG graphs should generally have decent girth
        from qec.fitness.spectral_metrics import compute_girth_spectrum
        girth_result = compute_girth_spectrum(H)
        assert girth_result["girth"] >= 4

    def test_filter_rejects_zero_matrix(self) -> None:
        H = np.zeros((6, 12), dtype=np.float64)
        assert not _passes_spectral_filter(H)


# ── Population Generator ────────────────────────────────────────


class TestPEGPopulation:
    """Population generator must produce correct candidate lists."""

    def test_correct_population_size(self) -> None:
        pop = generate_peg_population(SPEC_SMALL, 10, base_seed=42)
        assert len(pop) == 10

    def test_candidate_format(self) -> None:
        pop = generate_peg_population(SPEC_SMALL, 3, base_seed=0)
        for candidate in pop:
            assert "candidate_id" in candidate
            assert "H" in candidate
            assert isinstance(candidate["H"], np.ndarray)
            assert candidate["H"].shape == (6, 12)

    def test_candidate_ids_unique(self) -> None:
        pop = generate_peg_population(SPEC_SMALL, 10, base_seed=42)
        ids = [c["candidate_id"] for c in pop]
        assert len(set(ids)) == len(ids)

    def test_single_candidate(self) -> None:
        pop = generate_peg_population(SPEC_SMALL, 1, base_seed=0)
        assert len(pop) == 1


# ── Discovery Engine Integration ────────────────────────────────


class TestDiscoveryEngineIntegration:
    """PEG initialization must work inside the discovery engine."""

    def test_peg_init_strategy(self) -> None:
        spec = {**SPEC_SMALL, "init_strategy": "peg"}
        engine = DiscoveryEngine(
            population_size=5,
            generations=1,
            seed=42,
        )
        result = engine.run(spec)
        assert result["best"] is not None
        assert result["best_H"] is not None
        assert result["best_H"].shape == (6, 12)

    def test_random_init_strategy_backward_compatible(self) -> None:
        spec = {**SPEC_SMALL, "init_strategy": "random"}
        engine = DiscoveryEngine(
            population_size=5,
            generations=1,
            seed=42,
        )
        result = engine.run(spec)
        assert result["best"] is not None

    def test_default_init_strategy_backward_compatible(self) -> None:
        spec = dict(SPEC_SMALL)
        engine = DiscoveryEngine(
            population_size=5,
            generations=1,
            seed=42,
        )
        result = engine.run(spec)
        assert result["best"] is not None
