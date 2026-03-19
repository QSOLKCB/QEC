"""
Tests for BP fixed-point trap analysis (v4.8.0).

Validates:
  - Correct fixed point classification (syndrome weight zero)
  - Incorrect fixed point classification (nonzero syndrome, converged)
  - Degenerate symmetry detection (uniform LLR magnitudes)
  - No convergence detection (energy oscillation)
  - Determinism (identical outputs on repeated runs)
  - JSON serializability of all outputs
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from qec.diagnostics.bp_fixed_point_analysis import (
    compute_bp_fixed_point_analysis,
    DEFAULT_ENERGY_STABILITY_WINDOW,
    DEFAULT_ENERGY_STABILITY_RTOL,
    DEFAULT_ENTROPY_THRESHOLD,
    DEFAULT_VARIANCE_THRESHOLD,
)


# ── Helpers ───────────────────────────────────────────────────────────


def _correct_fp_trace(n_iters: int = 20, n_vars: int = 10) -> dict:
    """Build a trace where BP converges to the correct fixed point.
    Syndrome weight reaches zero, energy stabilizes."""
    rng = np.random.default_rng(seed=42)
    llr_0 = rng.standard_normal(n_vars)
    llr_trace = []
    energy_trace = []
    syndrome_trace = []
    for t in range(n_iters):
        scale = 1.0 + 0.5 * t
        llr_trace.append(llr_0 * scale)
        # Monotonically decreasing energy, stabilizing at the end.
        energy = max(1.0, 100.0 - 10.0 * t)
        energy_trace.append(energy)
        # Syndrome weight drops to zero after a few iterations.
        sw = max(0, 5 - t)
        syndrome_trace.append(sw)
    return {
        "llr_trace": llr_trace,
        "energy_trace": energy_trace,
        "syndrome_trace": syndrome_trace,
        "final_syndrome_weight": 0,
    }


def _incorrect_fp_trace(n_iters: int = 20, n_vars: int = 10) -> dict:
    """Build a trace where BP converges but to an incorrect fixed point.
    Energy stabilizes but syndrome weight remains nonzero."""
    rng = np.random.default_rng(seed=99)
    llr_0 = rng.standard_normal(n_vars)
    llr_trace = []
    energy_trace = []
    syndrome_trace = []
    for t in range(n_iters):
        scale = 1.0 + 0.3 * t
        llr_trace.append(llr_0 * scale)
        # Energy stabilizes at 20.0.
        energy = max(20.0, 100.0 - 8.0 * t)
        energy_trace.append(energy)
        # Syndrome weight stays at 3.
        syndrome_trace.append(3)
    return {
        "llr_trace": llr_trace,
        "energy_trace": energy_trace,
        "syndrome_trace": syndrome_trace,
        "final_syndrome_weight": 3,
    }


def _degenerate_trace(n_iters: int = 20, n_vars: int = 10) -> dict:
    """Build a trace with degenerate symmetry: all LLR magnitudes
    nearly identical (uniform), indicating a symmetric attractor."""
    llr_trace = []
    energy_trace = []
    syndrome_trace = []
    for t in range(n_iters):
        # All LLR values nearly identical magnitude → low entropy, low variance.
        llr_trace.append(np.full(n_vars, 0.5))
        # Energy stabilizes.
        energy = max(10.0, 50.0 - 5.0 * t)
        energy_trace.append(energy)
        syndrome_trace.append(2)
    return {
        "llr_trace": llr_trace,
        "energy_trace": energy_trace,
        "syndrome_trace": syndrome_trace,
        "final_syndrome_weight": 2,
    }


def _no_convergence_trace(n_iters: int = 20, n_vars: int = 10) -> dict:
    """Build a trace where energy oscillates and never stabilizes."""
    rng = np.random.default_rng(seed=77)
    llr_trace = []
    energy_trace = []
    syndrome_trace = []
    for t in range(n_iters):
        llr_trace.append(rng.standard_normal(n_vars) * (1.0 + 0.1 * t))
        # Oscillating energy: never stabilizes.
        energy = 50.0 + 20.0 * ((-1) ** t)
        energy_trace.append(energy)
        syndrome_trace.append(4)
    return {
        "llr_trace": llr_trace,
        "energy_trace": energy_trace,
        "syndrome_trace": syndrome_trace,
        "final_syndrome_weight": 4,
    }


# ── Tests ─────────────────────────────────────────────────────────────


class TestCorrectFixedPoint:
    """Syndrome weight zero and converged → correct_fixed_point."""

    def test_classification(self) -> None:
        data = _correct_fp_trace()
        result = compute_bp_fixed_point_analysis(**data)
        assert result["fixed_point_type"] == "correct_fixed_point"
        assert result["converged"] is True
        assert result["final_syndrome_weight"] == 0
        assert isinstance(result["iterations_to_fixed_point"], int)
        assert isinstance(result["llr_entropy"], float)
        assert isinstance(result["llr_variance"], float)

    def test_sorted_keys(self) -> None:
        data = _correct_fp_trace()
        result = compute_bp_fixed_point_analysis(**data)
        keys = list(result.keys())
        assert keys == sorted(keys)


class TestIncorrectFixedPoint:
    """Nonzero syndrome but energy stabilized → incorrect_fixed_point."""

    def test_classification(self) -> None:
        data = _incorrect_fp_trace()
        result = compute_bp_fixed_point_analysis(**data)
        assert result["fixed_point_type"] == "incorrect_fixed_point"
        assert result["converged"] is True
        assert result["final_syndrome_weight"] == 3

    def test_iterations_to_fixed_point(self) -> None:
        data = _incorrect_fp_trace()
        result = compute_bp_fixed_point_analysis(**data)
        assert result["iterations_to_fixed_point"] >= 0
        assert result["iterations_to_fixed_point"] <= len(data["energy_trace"])


class TestDegenerateFixedPoint:
    """Uniform LLR magnitudes → degenerate_fixed_point."""

    def test_classification(self) -> None:
        data = _degenerate_trace()
        result = compute_bp_fixed_point_analysis(**data)
        assert result["fixed_point_type"] == "degenerate_fixed_point"
        assert result["converged"] is True
        assert result["llr_entropy"] < DEFAULT_ENTROPY_THRESHOLD

    def test_low_variance(self) -> None:
        data = _degenerate_trace()
        result = compute_bp_fixed_point_analysis(**data)
        assert result["llr_variance"] < DEFAULT_VARIANCE_THRESHOLD


class TestNoConvergence:
    """Oscillating energy → no_convergence."""

    def test_classification(self) -> None:
        data = _no_convergence_trace()
        result = compute_bp_fixed_point_analysis(**data)
        assert result["fixed_point_type"] == "no_convergence"
        assert result["converged"] is False


class TestDeterminism:
    """Repeated runs must produce identical outputs."""

    def test_correct_fp_determinism(self) -> None:
        data = _correct_fp_trace()
        r1 = compute_bp_fixed_point_analysis(**data)
        r2 = compute_bp_fixed_point_analysis(**data)
        assert r1 == r2

    def test_incorrect_fp_determinism(self) -> None:
        data = _incorrect_fp_trace()
        r1 = compute_bp_fixed_point_analysis(**data)
        r2 = compute_bp_fixed_point_analysis(**data)
        assert r1 == r2

    def test_degenerate_determinism(self) -> None:
        data = _degenerate_trace()
        r1 = compute_bp_fixed_point_analysis(**data)
        r2 = compute_bp_fixed_point_analysis(**data)
        assert r1 == r2

    def test_no_convergence_determinism(self) -> None:
        data = _no_convergence_trace()
        r1 = compute_bp_fixed_point_analysis(**data)
        r2 = compute_bp_fixed_point_analysis(**data)
        assert r1 == r2


class TestJSONSerializable:
    """All outputs must be JSON-serializable."""

    def test_correct_fp_serializable(self) -> None:
        data = _correct_fp_trace()
        result = compute_bp_fixed_point_analysis(**data)
        serialized = json.dumps(result, sort_keys=True)
        deserialized = json.loads(serialized)
        assert deserialized["fixed_point_type"] == result["fixed_point_type"]
        assert deserialized["converged"] == result["converged"]

    def test_incorrect_fp_serializable(self) -> None:
        data = _incorrect_fp_trace()
        result = compute_bp_fixed_point_analysis(**data)
        serialized = json.dumps(result, sort_keys=True)
        deserialized = json.loads(serialized)
        assert deserialized == result

    def test_degenerate_serializable(self) -> None:
        data = _degenerate_trace()
        result = compute_bp_fixed_point_analysis(**data)
        serialized = json.dumps(result, sort_keys=True)
        deserialized = json.loads(serialized)
        assert deserialized == result

    def test_no_convergence_serializable(self) -> None:
        data = _no_convergence_trace()
        result = compute_bp_fixed_point_analysis(**data)
        serialized = json.dumps(result, sort_keys=True)
        deserialized = json.loads(serialized)
        assert deserialized == result


class TestEdgeCases:
    """Edge cases: empty traces, single iteration, mismatched lengths."""

    def test_empty_traces(self) -> None:
        result = compute_bp_fixed_point_analysis(
            llr_trace=[], energy_trace=[], syndrome_trace=[],
            final_syndrome_weight=0,
        )
        assert result["fixed_point_type"] == "no_convergence"
        assert result["converged"] is False
        assert result["iterations_to_fixed_point"] == 0

    def test_single_iteration(self) -> None:
        llr = [np.array([1.0, -0.5, 2.0])]
        energy = [10.0]
        synd = [1]
        result = compute_bp_fixed_point_analysis(
            llr_trace=llr, energy_trace=energy, syndrome_trace=synd,
            final_syndrome_weight=1,
        )
        # Single iteration cannot establish convergence (need >= 2).
        assert result["converged"] is False
        assert result["fixed_point_type"] == "no_convergence"

    def test_mismatched_llr_energy_raises(self) -> None:
        llr = [np.array([1.0])] * 5
        energy = [1.0] * 3
        synd = [0] * 5
        with pytest.raises(ValueError):
            compute_bp_fixed_point_analysis(
                llr_trace=llr, energy_trace=energy, syndrome_trace=synd,
                final_syndrome_weight=0,
            )

    def test_mismatched_llr_syndrome_raises(self) -> None:
        llr = [np.array([1.0])] * 5
        energy = [1.0] * 5
        synd = [0] * 3
        with pytest.raises(ValueError):
            compute_bp_fixed_point_analysis(
                llr_trace=llr, energy_trace=energy, syndrome_trace=synd,
                final_syndrome_weight=0,
            )


class TestCustomParameters:
    """Custom threshold parameters."""

    def test_custom_entropy_threshold(self) -> None:
        """High entropy threshold should classify more traces as degenerate."""
        data = _incorrect_fp_trace()
        result = compute_bp_fixed_point_analysis(
            **data, entropy_threshold=10.0,
        )
        # With very high entropy threshold, even normal LLR is "degenerate".
        assert result["fixed_point_type"] == "degenerate_fixed_point"

    def test_custom_stability_window(self) -> None:
        data = _correct_fp_trace()
        result = compute_bp_fixed_point_analysis(
            **data, energy_stability_window=3,
        )
        assert result["fixed_point_type"] == "correct_fixed_point"
        assert result["converged"] is True
