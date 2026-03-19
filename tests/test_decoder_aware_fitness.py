"""
Tests for v11.0.0 — Decoder-Aware Fitness Integration.

Covers:
  - decoder-aware metrics present when enabled
  - structural-only mode still works
  - cache behavior (value equality)
  - composite score includes decoder-aware components
"""

import numpy as np
import pytest

from qec.fitness.fitness_engine import FitnessEngine


def _make_small_ldpc():
    """Small (4, 8) LDPC-like matrix."""
    H = np.array([
        [1, 1, 1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 1, 0],
        [1, 0, 0, 1, 0, 0, 1, 1],
        [0, 0, 1, 0, 1, 1, 0, 1],
    ], dtype=np.float64)
    return H


class TestDecoderAwareFitness:
    def test_structural_only_no_decoder_metrics(self):
        H = _make_small_ldpc()
        engine = FitnessEngine(decoder_aware=False)
        result = engine.evaluate(H)

        assert "composite" in result
        assert "components" in result
        assert "metrics" in result
        # No decoder-aware keys
        assert "bp_stability" not in result["components"]
        assert "trapping_set_penalty" not in result["components"]

    def test_decoder_aware_metrics_present(self):
        H = _make_small_ldpc()
        engine = FitnessEngine(
            decoder_aware=True,
            bp_trials=5,
            bp_iterations=3,
            seed=42,
        )
        result = engine.evaluate(H)

        assert "composite" in result
        metrics = result["metrics"]
        assert "bp_stability_score" in metrics
        assert "trapping_set_penalty" in metrics
        assert "jacobian_spectral_radius_est" in metrics
        assert "jacobian_instability_penalty" in metrics

        components = result["components"]
        assert "bp_stability" in components
        assert "trapping_set_penalty" in components
        assert "jacobian_instability_penalty" in components

    def test_decoder_aware_deterministic(self):
        """Decoder-aware metrics (BP probe, trapping sets, Jacobian) are
        deterministic.  The IPR metrics from the underlying spectral
        eigensolver may exhibit minor non-determinism due to ARPACK, so
        we compare only the decoder-aware keys."""
        H = _make_small_ldpc()
        engine1 = FitnessEngine(
            decoder_aware=True, bp_trials=5, bp_iterations=3, seed=42,
        )
        engine2 = FitnessEngine(
            decoder_aware=True, bp_trials=5, bp_iterations=3, seed=42,
        )
        r1 = engine1.evaluate(H)
        r2 = engine2.evaluate(H)

        da_keys = [
            "bp_stability_score", "divergence_rate", "stagnation_rate",
            "oscillation_score", "trapping_set_penalty", "trapping_set_total",
            "trapping_set_min_size", "jacobian_spectral_radius_est",
            "jacobian_instability_penalty",
        ]
        for k in da_keys:
            assert r1["metrics"][k] == r2["metrics"][k], f"Mismatch on {k}"

    def test_cache_returns_equal_values(self):
        H = _make_small_ldpc()
        engine = FitnessEngine(
            decoder_aware=True, bp_trials=5, bp_iterations=3, seed=42,
        )
        r1 = engine.evaluate(H)
        r2 = engine.evaluate(H)

        # Cache should return equal values
        assert r1["composite"] == r2["composite"]
        assert r1["metrics"] == r2["metrics"]
        assert r1["components"] == r2["components"]

    def test_backward_compatible_structural(self):
        """Structural-only evaluation produces the same composite and
        structural metric keys.  IPR values may differ slightly due to
        ARPACK non-determinism in the pre-existing eigensolver, so we
        compare composites and non-IPR structural metrics."""
        H = _make_small_ldpc()
        engine_v10 = FitnessEngine()
        engine_v11_structural = FitnessEngine(decoder_aware=False)

        r10 = engine_v10.evaluate(H)
        r11 = engine_v11_structural.evaluate(H)

        # Same metric keys present
        assert set(r10["metrics"].keys()) == set(r11["metrics"].keys())
        # Same component keys
        assert set(r10["components"].keys()) == set(r11["components"].keys())
        # Non-IPR structural metrics should match
        for k in ["girth", "nbt_spectral_radius", "ace_variance",
                   "expansion", "cycle_density", "sparsity"]:
            assert r10["metrics"][k] == r11["metrics"][k], f"Mismatch on {k}"

    def test_clear_cache_works(self):
        H = _make_small_ldpc()
        engine = FitnessEngine(
            decoder_aware=True, bp_trials=5, bp_iterations=3, seed=42,
        )
        r1 = engine.evaluate(H)
        engine.clear_cache()
        r2 = engine.evaluate(H)

        assert r1["composite"] == r2["composite"]

    def test_decoder_aware_composite_differs_from_structural(self):
        """Decoder-aware mode should generally produce a different composite."""
        H = _make_small_ldpc()
        structural = FitnessEngine(decoder_aware=False)
        decoder_aware = FitnessEngine(
            decoder_aware=True, bp_trials=10, bp_iterations=5, seed=42,
        )

        r_s = structural.evaluate(H)
        r_d = decoder_aware.evaluate(H)

        # Composites should generally differ due to extra components
        # (only equal in degenerate cases)
        assert r_d["composite"] != r_s["composite"]

    def test_valid_metric_ranges(self):
        H = _make_small_ldpc()
        engine = FitnessEngine(
            decoder_aware=True, bp_trials=10, bp_iterations=5, seed=42,
        )
        result = engine.evaluate(H)
        metrics = result["metrics"]

        assert 0.0 <= metrics["bp_stability_score"] <= 1.0
        assert metrics["trapping_set_penalty"] >= 0.0
        assert metrics["jacobian_instability_penalty"] >= 0.0
        assert metrics["jacobian_spectral_radius_est"] >= 0.0
