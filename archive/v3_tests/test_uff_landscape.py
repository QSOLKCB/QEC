"""Tests for v82.2.0 — Invariant Landscape Mapping.

Covers:
- Grid generation (order, size, determinism)
- Sweep execution (determinism, keys, aggregation)
- Edge cases (single point, empty axis)
"""

from __future__ import annotations

import copy
import json
import os
import tempfile

from qec.experiments.uff_landscape import (
    generate_theta_grid,
    run_uff_landscape,
)


# -----------------------------------------------------------------------
# Grid tests
# -----------------------------------------------------------------------

class TestGenerateThetaGrid:
    """Tests for generate_theta_grid."""

    def test_correct_number_of_points(self):
        """Grid size equals product of axis lengths."""
        grid = generate_theta_grid([1.0, 2.0], [3.0], [4.0, 5.0])
        assert len(grid) == 2 * 1 * 2

    def test_deterministic_order(self):
        """Two identical calls produce identical grids."""
        g1 = generate_theta_grid([1.0, 2.0], [3.0, 4.0], [5.0])
        g2 = generate_theta_grid([1.0, 2.0], [3.0, 4.0], [5.0])
        assert g1 == g2

    def test_nested_loop_order(self):
        """V0 is outer, Rc is middle, beta is inner."""
        grid = generate_theta_grid([10.0, 20.0], [1.0, 2.0], [0.5, 1.5])
        # First point: smallest V0, smallest Rc, smallest beta
        assert grid[0] == [10.0, 1.0, 0.5]
        # Second point: smallest V0, smallest Rc, largest beta
        assert grid[1] == [10.0, 1.0, 1.5]
        # Last point: largest V0, largest Rc, largest beta
        assert grid[-1] == [20.0, 2.0, 1.5]

    def test_single_point_grid(self):
        """Single value per axis produces one point."""
        grid = generate_theta_grid([100.0], [5.0], [2.0])
        assert len(grid) == 1
        assert grid[0] == [100.0, 5.0, 2.0]

    def test_empty_axis_returns_empty(self):
        """Empty axis produces empty grid."""
        assert generate_theta_grid([], [1.0], [2.0]) == []
        assert generate_theta_grid([1.0], [], [2.0]) == []
        assert generate_theta_grid([1.0], [2.0], []) == []


# -----------------------------------------------------------------------
# Sweep tests
# -----------------------------------------------------------------------

class TestRunUffLandscape:
    """Tests for run_uff_landscape."""

    def _run_small(self, **kwargs):
        """Run a minimal 2-point sweep."""
        return run_uff_landscape(
            [100.0],
            [1.0],
            [1.0, 2.0],
            **kwargs,
        )

    def test_deterministic_repeated_run(self):
        """Two identical sweeps produce identical results."""
        r1 = self._run_small()
        r2 = self._run_small()
        assert r1["n_points"] == r2["n_points"]
        assert r1["best_theta"] == r2["best_theta"]
        assert r1["worst_theta"] == r2["worst_theta"]
        assert r1["phase_counts"] == r2["phase_counts"]
        for p1, p2 in zip(r1["points"], r2["points"]):
            assert p1["stability_score"] == p2["stability_score"]
            assert p1["phase"] == p2["phase"]

    def test_no_mutation_of_input_arrays(self):
        """Input lists must not be mutated."""
        v0 = [100.0, 150.0]
        rc = [1.0]
        beta = [2.0]
        v0_orig = copy.deepcopy(v0)
        rc_orig = copy.deepcopy(rc)
        beta_orig = copy.deepcopy(beta)
        run_uff_landscape(v0, rc, beta)
        assert v0 == v0_orig
        assert rc == rc_orig
        assert beta == beta_orig

    def test_every_point_has_required_keys(self):
        """Each point in the landscape must have all summary keys."""
        required = {
            "theta", "stability_score", "phase",
            "most_stable", "most_sensitive", "consensus", "verified",
        }
        result = self._run_small()
        for point in result["points"]:
            assert required.issubset(point.keys())

    def test_n_points_correct(self):
        """n_points matches actual number of points."""
        result = self._run_small()
        assert result["n_points"] == len(result["points"])
        assert result["n_points"] == 2

    def test_best_and_worst_theta_present(self):
        """best_theta and worst_theta are valid theta vectors."""
        result = self._run_small()
        assert len(result["best_theta"]) == 3
        assert len(result["worst_theta"]) == 3

    def test_phase_counts_sum_to_n_points(self):
        """Sum of phase counts equals total number of points."""
        result = self._run_small()
        total = sum(result["phase_counts"].values())
        assert total == result["n_points"]

    def test_all_points_consensus_true(self):
        """All points must report consensus == True."""
        result = self._run_small()
        for point in result["points"]:
            assert point["consensus"] is True

    def test_all_points_verified_true(self):
        """All points must report verified == True."""
        result = self._run_small()
        for point in result["points"]:
            assert point["verified"] is True

    def test_empty_axis_returns_empty_landscape(self):
        """Empty axis produces a valid empty landscape."""
        result = run_uff_landscape([], [1.0], [2.0])
        assert result["n_points"] == 0
        assert result["best_theta"] == []
        assert result["worst_theta"] == []
        assert result["phase_counts"] == {}
        assert result["points"] == []

    def test_single_point_landscape(self):
        """Single-point grid returns a valid landscape."""
        result = run_uff_landscape([200.0], [5.0], [2.0])
        assert result["n_points"] == 1
        assert result["best_theta"] == result["worst_theta"]

    def test_output_dir_writes_json(self):
        """When output_dir is set, uff_landscape.json is written."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_uff_landscape(
                [100.0], [1.0], [1.0],
                output_dir=tmpdir,
            )
            out_path = os.path.join(tmpdir, "uff_landscape.json")
            assert os.path.isfile(out_path)
            with open(out_path) as f:
                loaded = json.load(f)
            assert loaded["n_points"] == result["n_points"]
