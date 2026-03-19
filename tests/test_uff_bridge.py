"""Tests for UFF-to-QEC bridge layer (v82.0.0)."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from qec.experiments.uff_bridge import (
    _default_v_circ,
    build_sample,
    extract_features,
    run_uff_experiment,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_THETA = [220.0, 5.0, 2.0]
_R = np.linspace(0.1, 20.0, 100)


# ---------------------------------------------------------------------------
# Feature extraction tests
# ---------------------------------------------------------------------------


class TestExtractFeatures:
    """Tests for extract_features."""

    def test_returns_all_keys(self) -> None:
        v = _default_v_circ(_R, _THETA)
        features = extract_features(_R, v)
        assert set(features.keys()) == {
            "energy", "spread", "zcr", "centroid",
            "gradient_energy", "curvature",
        }

    def test_finite_outputs(self) -> None:
        v = _default_v_circ(_R, _THETA)
        features = extract_features(_R, v)
        for key, val in features.items():
            assert np.isfinite(val), f"{key} is not finite: {val}"

    def test_energy_positive_for_positive_curve(self) -> None:
        v = _default_v_circ(_R, _THETA)
        features = extract_features(_R, v)
        assert features["energy"] > 0.0

    def test_deterministic(self) -> None:
        v = _default_v_circ(_R, _THETA)
        f1 = extract_features(_R, v)
        f2 = extract_features(_R, v)
        assert f1 == f2

    def test_gradient_energy_present(self) -> None:
        v = _default_v_circ(_R, _THETA)
        features = extract_features(_R, v)
        assert "gradient_energy" in features

    def test_curvature_present(self) -> None:
        v = _default_v_circ(_R, _THETA)
        features = extract_features(_R, v)
        assert "curvature" in features

    def test_gradient_energy_finite(self) -> None:
        v = _default_v_circ(_R, _THETA)
        features = extract_features(_R, v)
        assert np.isfinite(features["gradient_energy"])

    def test_curvature_finite(self) -> None:
        v = _default_v_circ(_R, _THETA)
        features = extract_features(_R, v)
        assert np.isfinite(features["curvature"])

    def test_gradient_curvature_deterministic(self) -> None:
        v = _default_v_circ(_R, _THETA)
        f1 = extract_features(_R, v)
        f2 = extract_features(_R, v)
        assert f1["gradient_energy"] == f2["gradient_energy"]
        assert f1["curvature"] == f2["curvature"]

    def test_no_input_mutation(self) -> None:
        v = _default_v_circ(_R, _THETA)
        R_copy = _R.copy()
        v_copy = v.copy()
        extract_features(_R, v)
        np.testing.assert_array_equal(_R, R_copy)
        np.testing.assert_array_equal(v, v_copy)


# ---------------------------------------------------------------------------
# Build sample tests
# ---------------------------------------------------------------------------


class TestBuildSample:
    """Tests for build_sample."""

    def test_returns_sonic_keys(self) -> None:
        v = _default_v_circ(_R, _THETA)
        features = extract_features(_R, v)
        sample = build_sample(features)
        assert "rms_energy" in sample
        assert "spectral_centroid_hz" in sample
        assert "spectral_spread_hz" in sample
        assert "zero_crossing_rate" in sample
        assert "fft_top_peaks" in sample

    def test_no_input_mutation(self) -> None:
        features = {"energy": 1.0, "spread": 0.5, "zcr": 3.0, "centroid": 10.0}
        original = copy.deepcopy(features)
        build_sample(features)
        assert features == original


# ---------------------------------------------------------------------------
# Default velocity model tests
# ---------------------------------------------------------------------------


class TestDefaultVCirc:
    """Tests for _default_v_circ."""

    def test_output_shape(self) -> None:
        v = _default_v_circ(_R, _THETA)
        assert v.shape == _R.shape

    def test_finite(self) -> None:
        v = _default_v_circ(_R, _THETA)
        assert np.all(np.isfinite(v))

    def test_zero_Rc(self) -> None:
        v = _default_v_circ(_R, [220.0, 0.0, 2.0])
        np.testing.assert_array_equal(v, np.zeros_like(_R))


# ---------------------------------------------------------------------------
# End-to-end pipeline tests
# ---------------------------------------------------------------------------


class TestRunUffExperiment:
    """Tests for the full UFF→QEC pipeline."""

    def test_end_to_end(self) -> None:
        result = run_uff_experiment(_THETA)
        assert "theta" in result
        assert "features" in result
        assert "probe" in result
        assert "invariants" in result
        assert "trajectory" in result
        assert "verification" in result
        assert "proof" in result
        assert "consensus" in result

    def test_deterministic_output(self) -> None:
        r1 = run_uff_experiment(_THETA)
        r2 = run_uff_experiment(_THETA)
        assert r1["features"] == r2["features"]
        assert r1["verification"]["final_hash"] == r2["verification"]["final_hash"]
        assert r1["consensus"]["consensus_hash"] == r2["consensus"]["consensus_hash"]

    def test_consensus_true(self) -> None:
        result = run_uff_experiment(_THETA)
        assert result["consensus"]["consensus"] is True

    def test_proof_verified(self) -> None:
        result = run_uff_experiment(_THETA)
        assert result["proof"]["verified"] is True

    def test_verification_match(self) -> None:
        result = run_uff_experiment(_THETA)
        assert result["verification"]["match"] is True

    def test_theta_preserved(self) -> None:
        result = run_uff_experiment(_THETA)
        assert result["theta"] == _THETA

    def test_custom_v_circ_fn(self) -> None:
        def flat_curve(R: np.ndarray, theta: list) -> np.ndarray:
            return np.full_like(R, float(theta[0]))

        result = run_uff_experiment([100.0, 1.0, 1.0], v_circ_fn=flat_curve)
        assert result["features"]["energy"] == pytest.approx(100.0)
        assert result["features"]["spread"] == pytest.approx(0.0)

    def test_edge_theta_small(self) -> None:
        result = run_uff_experiment([1e-10, 1e-10, 1.0])
        assert np.isfinite(result["features"]["energy"])

    def test_edge_theta_large(self) -> None:
        result = run_uff_experiment([1e6, 1e3, 10.0])
        for val in result["features"].values():
            assert np.isfinite(val)

    def test_no_input_mutation(self) -> None:
        theta = [220.0, 5.0, 2.0]
        theta_copy = list(theta)
        run_uff_experiment(theta)
        assert theta == theta_copy

    def test_invalid_v_circ_fn(self) -> None:
        with pytest.raises(AssertionError):
            run_uff_experiment([1, 1, 1], v_circ_fn=123)

    def test_bad_shape_v_circ_fn(self) -> None:
        def bad_fn(R: np.ndarray, theta: list) -> list:
            return [1, 2, 3]

        with pytest.raises(AssertionError):
            run_uff_experiment([1, 1, 1], v_circ_fn=bad_fn)
