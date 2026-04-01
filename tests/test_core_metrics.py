"""Tests for qec.core.metrics — tolerance hardening and constant contract."""

import numpy as np

from qec.core.metrics import SCALAR_KEYS, compute_resonance


class TestResonanceTolerance:
    """Verify tolerance-based float comparison in compute_resonance."""

    def test_exact_integers_still_match(self):
        """Integer-like values must still detect resonance."""
        # Pattern [1, 2, 1, 2, 1, 2] has period-2 resonance
        values = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        score = compute_resonance(values)
        assert score == 1.0

    def test_near_equal_floats_match(self):
        """Values differing by < atol should be treated as equal."""
        base = [1.0, 2.0, 3.0, 4.0]
        # Repeat with tiny perturbation well within tolerance
        perturbed = [1.0 + 1e-14, 2.0 - 1e-14, 3.0 + 1e-14, 4.0 - 1e-14]
        values = base + perturbed
        score = compute_resonance(values)
        assert score > 0.0, "Near-equal floats should be detected as repeating"

    def test_distinct_floats_do_not_match(self):
        """Values differing by more than tolerance should not match."""
        # No repeating pattern
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        score = compute_resonance(values)
        assert score == 0.0

    def test_deterministic(self):
        """Repeated calls must yield identical results."""
        values = [0.1, 0.2, 0.1 + 1e-13, 0.2 - 1e-13, 0.1, 0.2]
        r1 = compute_resonance(values)
        r2 = compute_resonance(values)
        assert r1 == r2


class TestScalarKeysContract:
    """Verify SCALAR_KEYS public constant contract."""

    def test_importable(self):
        """SCALAR_KEYS must be importable from qec.core.metrics."""
        assert SCALAR_KEYS is not None

    def test_is_tuple(self):
        """SCALAR_KEYS must be a tuple (immutable)."""
        assert isinstance(SCALAR_KEYS, tuple)

    def test_ordering_unchanged(self):
        """SCALAR_KEYS ordering must match canonical contract."""
        expected = (
            "phi_alignment",
            "symmetry_score",
            "triality_balance",
            "nonlinear_response",
            "resonance",
            "complexity",
        )
        assert SCALAR_KEYS == expected

    def test_importable_from_analysis(self):
        """SCALAR_KEYS must also be importable from analysis layer."""
        from qec.analysis.multiscale_metrics import SCALAR_KEYS as analysis_keys
        assert analysis_keys is SCALAR_KEYS
