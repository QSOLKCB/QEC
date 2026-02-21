"""
Tests for the Wilson score confidence interval helpers.
"""

import pytest
import numpy as np

from src.simulation.fer import _probit, _wilson_ci


# ───────────────────────────────────────────────────────────────────
# _probit (inverse normal CDF)
# ───────────────────────────────────────────────────────────────────

class TestProbit:

    def test_probit_median(self):
        """_probit(0.5) should be 0.0."""
        assert abs(_probit(0.5)) < 1e-6

    def test_probit_975(self):
        """_probit(0.975) ≈ 1.96 (used for 95% CI)."""
        z = _probit(0.975)
        assert abs(z - 1.96) < 0.01

    def test_probit_025(self):
        """_probit(0.025) ≈ -1.96."""
        z = _probit(0.025)
        assert abs(z - (-1.96)) < 0.01

    def test_probit_995(self):
        """_probit(0.995) ≈ 2.576 (used for 99% CI)."""
        z = _probit(0.995)
        assert abs(z - 2.576) < 0.01

    def test_probit_symmetry(self):
        """_probit(p) == -_probit(1-p) for several p values."""
        for p in [0.1, 0.2, 0.3, 0.4]:
            z1 = _probit(p)
            z2 = _probit(1.0 - p)
            assert abs(z1 + z2) < 1e-10

    def test_probit_monotonic(self):
        """_probit is strictly increasing."""
        ps = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
        zs = [_probit(p) for p in ps]
        for i in range(len(zs) - 1):
            assert zs[i] < zs[i + 1]

    def test_probit_invalid_0(self):
        """_probit(0) raises ValueError."""
        with pytest.raises(ValueError, match="probit"):
            _probit(0.0)

    def test_probit_invalid_1(self):
        """_probit(1) raises ValueError."""
        with pytest.raises(ValueError, match="probit"):
            _probit(1.0)

    def test_probit_deterministic(self):
        """Same input produces same output."""
        assert _probit(0.025) == _probit(0.025)
        assert _probit(0.975) == _probit(0.975)


# ───────────────────────────────────────────────────────────────────
# _wilson_ci (continuity-corrected Wilson score interval)
# ───────────────────────────────────────────────────────────────────

class TestWilsonCI:

    def test_zero_errors(self):
        """k=0 → lower should be 0.0."""
        lo, hi, w = _wilson_ci(0, 100, alpha=0.05, gamma=1.5)
        assert lo == 0.0
        assert hi > 0.0
        assert w == hi - lo

    def test_all_errors(self):
        """k=n → upper should be 1.0."""
        lo, hi, w = _wilson_ci(100, 100, alpha=0.05, gamma=1.5)
        assert hi == 1.0
        assert lo < 1.0
        assert w == hi - lo

    def test_half_errors(self):
        """k=n/2 → CI should be centred near 0.5."""
        lo, hi, w = _wilson_ci(50, 100, alpha=0.05, gamma=1.5)
        centre = (lo + hi) / 2
        assert 0.4 < centre < 0.6
        assert 0.0 <= lo < 0.5
        assert 0.5 < hi <= 1.0

    def test_width_nonneg(self):
        """Width is always >= 0."""
        for k in [0, 1, 5, 50, 99, 100]:
            _, _, w = _wilson_ci(k, 100, alpha=0.05, gamma=1.5)
            assert w >= 0.0

    def test_bounds_ordered(self):
        """lower <= upper always."""
        for k in [0, 10, 50, 90, 100]:
            lo, hi, w = _wilson_ci(k, 100, alpha=0.05, gamma=1.5)
            assert lo <= hi

    def test_width_matches_bounds(self):
        """width == upper - lower."""
        lo, hi, w = _wilson_ci(30, 200, alpha=0.05, gamma=1.5)
        assert abs(w - (hi - lo)) < 1e-15

    def test_narrows_with_n(self):
        """More trials → smaller CI width for same proportion."""
        _, _, w1 = _wilson_ci(10, 100, alpha=0.05, gamma=1.5)
        _, _, w2 = _wilson_ci(100, 1000, alpha=0.05, gamma=1.5)
        assert w2 < w1

    def test_deterministic(self):
        """Same inputs produce identical outputs."""
        r1 = _wilson_ci(25, 100, alpha=0.05, gamma=1.5)
        r2 = _wilson_ci(25, 100, alpha=0.05, gamma=1.5)
        assert r1 == r2

    def test_lower_alpha_wider(self):
        """Smaller alpha (higher confidence) → wider CI."""
        _, _, w_95 = _wilson_ci(50, 100, alpha=0.05, gamma=1.5)
        _, _, w_99 = _wilson_ci(50, 100, alpha=0.01, gamma=1.5)
        assert w_99 > w_95

    def test_gamma_zero_no_correction(self):
        """gamma=0 disables continuity correction; interval still valid."""
        lo, hi, w = _wilson_ci(50, 100, alpha=0.05, gamma=0)
        assert 0.0 <= lo < hi <= 1.0
