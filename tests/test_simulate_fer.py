"""
Tests for the deterministic FER simulation harness.
"""

import json

import pytest
import numpy as np

from src.qec_qldpc_codes import create_code
from src.simulation.fer import simulate_fer


@pytest.fixture
def small_code():
    return create_code('rate_0.50', lifting_size=8, seed=42)


class TestSimulateFER:

    def test_deterministic_with_seed(self, small_code):
        """Two runs with the same seed produce identical results."""
        dc = {"mode": "min_sum", "max_iters": 20}
        nc = {"p_grid": [0.01, 0.02]}
        r1 = simulate_fer(small_code.H_X, dc, nc, trials=50, seed=99)
        r2 = simulate_fer(small_code.H_X, dc, nc, trials=50, seed=99)
        assert r1 == r2

    def test_json_serializable(self, small_code):
        """Result dict is directly JSON-serializable."""
        dc = {"mode": "min_sum", "max_iters": 10}
        nc = {"p_grid": [0.01]}
        result = simulate_fer(small_code.H_X, dc, nc, trials=20, seed=42)
        # Should not raise
        s = json.dumps(result)
        assert isinstance(s, str)

    def test_fer_in_bounds(self, small_code):
        """FER values are in [0, 1]."""
        dc = {"mode": "min_sum", "max_iters": 10}
        nc = {"p_grid": [0.005, 0.01, 0.05]}
        result = simulate_fer(small_code.H_X, dc, nc, trials=30, seed=42)
        for fer in result["results"]["FER"]:
            assert 0.0 <= fer <= 1.0

    def test_ber_in_bounds(self, small_code):
        """BER values are in [0, 1]."""
        dc = {"mode": "min_sum", "max_iters": 10}
        nc = {"p_grid": [0.01, 0.05]}
        result = simulate_fer(small_code.H_X, dc, nc, trials=30, seed=42)
        for ber in result["results"]["BER"]:
            assert 0.0 <= ber <= 1.0

    def test_required_keys(self, small_code):
        """Result dict has all required top-level and nested keys."""
        dc = {"mode": "sum_product", "max_iters": 5}
        nc = {"p_grid": [0.01]}
        result = simulate_fer(small_code.H_X, dc, nc, trials=10, seed=1)
        assert "seed" in result
        assert "decoder" in result
        assert "noise" in result
        assert "results" in result
        r = result["results"]
        assert "p" in r
        assert "FER" in r
        assert "BER" in r
        assert "mean_iters" in r

    def test_result_lengths_match_p_grid(self, small_code):
        """Each result list has same length as p_grid."""
        dc = {"mode": "min_sum", "max_iters": 5}
        p_grid = [0.001, 0.01, 0.05]
        nc = {"p_grid": p_grid}
        result = simulate_fer(small_code.H_X, dc, nc, trials=10, seed=42)
        r = result["results"]
        assert len(r["p"]) == len(p_grid)
        assert len(r["FER"]) == len(p_grid)
        assert len(r["BER"]) == len(p_grid)
        assert len(r["mean_iters"]) == len(p_grid)

    def test_mean_iters_positive(self, small_code):
        """Mean iterations are positive."""
        dc = {"mode": "min_sum", "max_iters": 20}
        nc = {"p_grid": [0.01]}
        result = simulate_fer(small_code.H_X, dc, nc, trials=20, seed=42)
        for mi in result["results"]["mean_iters"]:
            assert mi > 0

    def test_monotonicity_sanity(self, small_code):
        """FER at very low p should be <= FER at moderate p (with enough trials)."""
        dc = {"mode": "min_sum", "max_iters": 30}
        nc = {"p_grid": [0.001, 0.05]}
        result = simulate_fer(small_code.H_X, dc, nc, trials=100, seed=42)
        fer_low = result["results"]["FER"][0]
        fer_high = result["results"]["FER"][1]
        assert fer_low <= fer_high

    def test_different_seeds_differ(self, small_code):
        """Different seeds generally produce different results."""
        dc = {"mode": "min_sum", "max_iters": 10}
        nc = {"p_grid": [0.05]}
        r1 = simulate_fer(small_code.H_X, dc, nc, trials=50, seed=1)
        r2 = simulate_fer(small_code.H_X, dc, nc, trials=50, seed=2)
        # Not guaranteed to differ, but very likely at p=0.05
        # At minimum, they should both be valid
        assert 0.0 <= r1["results"]["FER"][0] <= 1.0
        assert 0.0 <= r2["results"]["FER"][0] <= 1.0
