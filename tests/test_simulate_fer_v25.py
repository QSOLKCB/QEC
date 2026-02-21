"""
Tests for v2.5.0 simulate_fer enhancements: Wilson CI and early termination.
"""

import json

import pytest
import numpy as np

from src.qec_qldpc_codes import create_code
from src.simulation.fer import simulate_fer


@pytest.fixture
def small_code():
    return create_code('rate_0.50', lifting_size=8, seed=42)


# ───────────────────────────────────────────────────────────────────
# Wilson CI Integration
# ───────────────────────────────────────────────────────────────────

class TestSimulateFERWilsonCI:

    def test_ci_method_none_no_ci_key(self, small_code):
        """Default (ci_method=None): no 'ci' key in result."""
        dc = {"mode": "min_sum", "max_iters": 10}
        nc = {"p_grid": [0.01]}
        result = simulate_fer(small_code.H_X, dc, nc, trials=20, seed=42)
        assert "ci" not in result

    def test_ci_method_wilson_adds_ci_key(self, small_code):
        """ci_method='wilson' adds 'ci' key to result."""
        dc = {"mode": "min_sum", "max_iters": 10}
        nc = {"p_grid": [0.01]}
        result = simulate_fer(
            small_code.H_X, dc, nc, trials=20, seed=42,
            ci_method="wilson",
        )
        assert "ci" in result

    def test_ci_output_structure(self, small_code):
        """'ci' dict has required keys: method, alpha, gamma, lower, upper, width."""
        dc = {"mode": "min_sum", "max_iters": 10}
        nc = {"p_grid": [0.01]}
        result = simulate_fer(
            small_code.H_X, dc, nc, trials=20, seed=42,
            ci_method="wilson",
        )
        ci = result["ci"]
        assert ci["method"] == "wilson"
        assert "alpha" in ci
        assert "gamma" in ci
        assert "lower" in ci
        assert "upper" in ci
        assert "width" in ci

    def test_ci_lengths_match_p_grid(self, small_code):
        """lower/upper/width lists have same length as p_grid."""
        dc = {"mode": "min_sum", "max_iters": 10}
        p_grid = [0.001, 0.01, 0.05]
        nc = {"p_grid": p_grid}
        result = simulate_fer(
            small_code.H_X, dc, nc, trials=20, seed=42,
            ci_method="wilson",
        )
        ci = result["ci"]
        assert len(ci["lower"]) == len(p_grid)
        assert len(ci["upper"]) == len(p_grid)
        assert len(ci["width"]) == len(p_grid)

    def test_ci_bounds_valid(self, small_code):
        """0 <= lower <= upper <= 1 for all p values."""
        dc = {"mode": "min_sum", "max_iters": 10}
        nc = {"p_grid": [0.01, 0.05]}
        result = simulate_fer(
            small_code.H_X, dc, nc, trials=30, seed=42,
            ci_method="wilson",
        )
        ci = result["ci"]
        for i in range(len(ci["lower"])):
            assert 0.0 <= ci["lower"][i] <= ci["upper"][i] <= 1.0

    def test_ci_width_matches_bounds(self, small_code):
        """width == upper - lower for each p."""
        dc = {"mode": "min_sum", "max_iters": 10}
        nc = {"p_grid": [0.01, 0.05]}
        result = simulate_fer(
            small_code.H_X, dc, nc, trials=30, seed=42,
            ci_method="wilson",
        )
        ci = result["ci"]
        for i in range(len(ci["width"])):
            assert abs(ci["width"][i] - (ci["upper"][i] - ci["lower"][i])) < 1e-15

    def test_ci_does_not_affect_fer(self, small_code):
        """FER/BER/mean_iters are identical with and without CI."""
        dc = {"mode": "min_sum", "max_iters": 10}
        nc = {"p_grid": [0.01, 0.05]}
        r_no_ci = simulate_fer(
            small_code.H_X, dc, nc, trials=30, seed=42,
        )
        r_ci = simulate_fer(
            small_code.H_X, dc, nc, trials=30, seed=42,
            ci_method="wilson",
        )
        assert r_no_ci["results"]["FER"] == r_ci["results"]["FER"]
        assert r_no_ci["results"]["BER"] == r_ci["results"]["BER"]
        assert r_no_ci["results"]["mean_iters"] == r_ci["results"]["mean_iters"]

    def test_ci_json_serializable(self, small_code):
        """Result with CI is JSON-serializable."""
        dc = {"mode": "min_sum", "max_iters": 10}
        nc = {"p_grid": [0.01]}
        result = simulate_fer(
            small_code.H_X, dc, nc, trials=20, seed=42,
            ci_method="wilson",
        )
        s = json.dumps(result)
        assert isinstance(s, str)

    def test_ci_deterministic(self, small_code):
        """Same seed produces identical CI values."""
        dc = {"mode": "min_sum", "max_iters": 10}
        nc = {"p_grid": [0.01, 0.05]}
        r1 = simulate_fer(
            small_code.H_X, dc, nc, trials=30, seed=42,
            ci_method="wilson",
        )
        r2 = simulate_fer(
            small_code.H_X, dc, nc, trials=30, seed=42,
            ci_method="wilson",
        )
        assert r1["ci"] == r2["ci"]

    def test_invalid_ci_method_raises(self, small_code):
        """Invalid ci_method raises ValueError."""
        dc = {"mode": "min_sum", "max_iters": 10}
        nc = {"p_grid": [0.01]}
        with pytest.raises(ValueError, match="ci_method"):
            simulate_fer(
                small_code.H_X, dc, nc, trials=10, seed=42,
                ci_method="invalid",
            )

    def test_alpha_out_of_range_raises(self, small_code):
        """alpha=0 or alpha=1 raises ValueError."""
        dc = {"mode": "min_sum", "max_iters": 10}
        nc = {"p_grid": [0.01]}
        with pytest.raises(ValueError, match="alpha"):
            simulate_fer(
                small_code.H_X, dc, nc, trials=10, seed=42,
                ci_method="wilson", alpha=0.0,
            )
        with pytest.raises(ValueError, match="alpha"):
            simulate_fer(
                small_code.H_X, dc, nc, trials=10, seed=42,
                ci_method="wilson", alpha=1.0,
            )

    def test_gamma_negative_raises(self, small_code):
        """gamma=-1 raises ValueError."""
        dc = {"mode": "min_sum", "max_iters": 10}
        nc = {"p_grid": [0.01]}
        with pytest.raises(ValueError, match="gamma"):
            simulate_fer(
                small_code.H_X, dc, nc, trials=10, seed=42,
                ci_method="wilson", gamma=-1.0,
            )


# ───────────────────────────────────────────────────────────────────
# Early Termination
# ───────────────────────────────────────────────────────────────────

class TestSimulateFEREarlyStop:

    def test_early_stop_deterministic(self, small_code):
        """Same seed + same epsilon produces identical results."""
        dc = {"mode": "min_sum", "max_iters": 10}
        nc = {"p_grid": [0.05]}
        r1 = simulate_fer(
            small_code.H_X, dc, nc, trials=200, seed=42,
            ci_method="wilson", early_stop_epsilon=0.2,
        )
        r2 = simulate_fer(
            small_code.H_X, dc, nc, trials=200, seed=42,
            ci_method="wilson", early_stop_epsilon=0.2,
        )
        assert r1 == r2

    def test_early_stop_without_ci_raises(self, small_code):
        """early_stop_epsilon without ci_method raises ValueError."""
        dc = {"mode": "min_sum", "max_iters": 10}
        nc = {"p_grid": [0.01]}
        with pytest.raises(ValueError, match="ci_method"):
            simulate_fer(
                small_code.H_X, dc, nc, trials=10, seed=42,
                early_stop_epsilon=0.1,
            )

    def test_early_stop_nonpositive_raises(self, small_code):
        """early_stop_epsilon=0 raises ValueError."""
        dc = {"mode": "min_sum", "max_iters": 10}
        nc = {"p_grid": [0.01]}
        with pytest.raises(ValueError, match="early_stop_epsilon"):
            simulate_fer(
                small_code.H_X, dc, nc, trials=10, seed=42,
                ci_method="wilson", early_stop_epsilon=0.0,
            )

    def test_early_stop_preserves_fer_range(self, small_code):
        """FER is still in [0, 1] with early stop."""
        dc = {"mode": "min_sum", "max_iters": 10}
        nc = {"p_grid": [0.01, 0.05]}
        result = simulate_fer(
            small_code.H_X, dc, nc, trials=100, seed=42,
            ci_method="wilson", early_stop_epsilon=0.2,
        )
        for fer in result["results"]["FER"]:
            assert 0.0 <= fer <= 1.0

    def test_actual_trials_key_present(self, small_code):
        """results['actual_trials'] exists when early_stop_epsilon is set."""
        dc = {"mode": "min_sum", "max_iters": 10}
        nc = {"p_grid": [0.01]}
        result = simulate_fer(
            small_code.H_X, dc, nc, trials=100, seed=42,
            ci_method="wilson", early_stop_epsilon=0.5,
        )
        assert "actual_trials" in result["results"]

    def test_actual_trials_not_present_without_early_stop(self, small_code):
        """results['actual_trials'] absent when early_stop_epsilon is None."""
        dc = {"mode": "min_sum", "max_iters": 10}
        nc = {"p_grid": [0.01]}
        result = simulate_fer(
            small_code.H_X, dc, nc, trials=20, seed=42,
            ci_method="wilson",
        )
        assert "actual_trials" not in result["results"]

    def test_actual_trials_lengths(self, small_code):
        """actual_trials list same length as p_grid."""
        dc = {"mode": "min_sum", "max_iters": 10}
        p_grid = [0.01, 0.05]
        nc = {"p_grid": p_grid}
        result = simulate_fer(
            small_code.H_X, dc, nc, trials=100, seed=42,
            ci_method="wilson", early_stop_epsilon=0.5,
        )
        assert len(result["results"]["actual_trials"]) == len(p_grid)

    def test_large_epsilon_reduces_trials(self, small_code):
        """Very large epsilon stops early (fewer than max trials)."""
        dc = {"mode": "min_sum", "max_iters": 10}
        nc = {"p_grid": [0.05]}
        result = simulate_fer(
            small_code.H_X, dc, nc, trials=500, seed=42,
            ci_method="wilson", early_stop_epsilon=1.0,
        )
        # With epsilon=1.0, any CI width < 1.0 triggers stop,
        # so we expect early termination.
        actual = result["results"]["actual_trials"][0]
        assert actual < 500

    def test_tiny_epsilon_runs_all_trials(self, small_code):
        """Very small epsilon runs all trials."""
        dc = {"mode": "min_sum", "max_iters": 10}
        nc = {"p_grid": [0.01]}
        result = simulate_fer(
            small_code.H_X, dc, nc, trials=20, seed=42,
            ci_method="wilson", early_stop_epsilon=1e-10,
        )
        assert result["results"]["actual_trials"][0] == 20

    def test_early_stop_json_serializable(self, small_code):
        """Result with early stop is JSON-serializable."""
        dc = {"mode": "min_sum", "max_iters": 10}
        nc = {"p_grid": [0.01]}
        result = simulate_fer(
            small_code.H_X, dc, nc, trials=50, seed=42,
            ci_method="wilson", early_stop_epsilon=0.5,
        )
        s = json.dumps(result)
        assert isinstance(s, str)
